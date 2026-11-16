"""``hardware.baseline_compile`` (opt-in): a baseline's windows through ``torch.compile``
(:mod:`hsn.train.baseline_compile`): invisible at its default, taken by the sweep spec, equal to
the eager forward and backward to rounding for every family, and reported by the run records.

Backends on the CPU: inductor (what the key means) on Linux for the compiled-against-eager checks
and the modes; ``aot_eager`` (dynamo and AOTAutograd, eager kernels) for the trainer, the harness
and the runner, where what is checked is the wiring and not the kernels, and wherever inductor's
C++ toolchain is not available (the hosted macOS runner).
"""

from __future__ import annotations

import dataclasses
import functools
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from cards import CUDA_PROFILE
from hsn.baselines.mamba import _plain_scan, _SelectiveScan
from hsn.config import (
    BASELINE_COMPILE_MODEL_NAMES,
    RunConfig,
    canonical_json,
    config_hash,
    load_config,
    load_hardware_profile,
)
from hsn.data.dataset import ByteDataset
from hsn.device import DeviceContext, resolve, seed_everything
from hsn.experiments.protocols import ArmRun, run_protocol
from hsn.experiments.rules import parse_rules
from hsn.experiments.spec import SpecError, expand, load_spec, protocol_params
from hsn.runner import build_model, run_seeds
from hsn.train import baseline_compile
from hsn.train.baseline_compile import (
    CUDA_GRAPH_MODES,
    CompiledBaseline,
    check_baseline_scope,
    compile_baseline,
)
from hsn.train.loop import Trainer
from p1_11_support import (
    GRU_2M,
    REF_A1,
    REPO,
    TINY_GRU_OVERRIDES,
    TINY_TRANSFORMER_OVERRIDES,
    TRANSFORMER_2M,
    disk_overrides,
    docs,
    fake_repo,
    tiny_manifest,  # noqa: F401  (fixture)
    write_yaml,
)
from torch import Tensor, nn

READER = REPO / "configs" / "baselines" / "reader_w64l2.yaml"
MAMBA = REPO / "configs" / "baselines" / "mamba_2m.yaml"
NOMEM = REPO / "configs" / "experiments" / "hsn_t_nomem.yaml"  # HSN-T without its memory
#: family -> (configuration, overrides of a tiny model, compiled functions of the equality test)
TINY: dict[str, tuple[Path, list[str], int]] = {
    # the embeddings and the read-out in both layouts of the recurrence's output, and the
    # evaluation's two
    "gru": (GRU_2M, TINY_GRU_OVERRIDES, 5),
    "transformer": (TRANSFORMER_2M, TINY_TRANSFORMER_OVERRIDES, 2),  # training, evaluation
    "reader": (
        READER,
        ["model.reader_width=16", "model.reader_layers=1", "model.reader_heads=2"],
        2,
    ),
    "selective_ssm": (
        MAMBA,
        [f"baseline.selective_ssm.{key}" for key in ("d_model=16", "n_layers=2", "d_state=4")],
        2,  # one window for every reset pattern, and the evaluation's
    ),
    "hsn_t_nomem": (
        NOMEM,
        [f"model.{key}" for key in ("d=8", "context_dim=8", "hidden=16", "reader_width=16")]
        + ["model.reader_heads=2"],
        2,
    ),
}
FAMILIES = sorted(TINY)
MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
#: inductor's CPU backend needs a C++ compiler with OpenMP (Linux runners have it).
CPU_BACKEND = "inductor" if sys.platform.startswith("linux") else "aot_eager"
B, T = 3, 16
#: Compiled against eager in fp32 on the CPU: 10x the largest difference measured (14 Nov 2026,
#: inductor, every family, both reset patterns, training and evaluation), rounded up: logits and
#: state 4.8e-7; parameter gradients 8.8e-9, and 1.5e-7 for ``selective_ssm`` (its gradients are
#: ten times larger and come from autograd's backward of the scan, not the hand-written one:
#: FAMILY_TOL); and relative to the largest eager value, 2.6e-7 for the logits and state and
#: 8.7e-7 for the gradients (the tiny HSN-T without memory has logits of 1e-2 and gradients of
#: 2e-5, which an absolute bound alone would not test). ``selective_ssm``'s relative differences
#: are 1.6e-7 and 3.9e-7 (15 Nov 2026), so the relative bound holds for it as it is.
TOL = {"values": 5e-6, "grads": 1e-7, "relative": 1e-5}
#: A family's own bounds, where they are larger than TOL's.
FAMILY_TOL: dict[str, dict[str, float]] = {"selective_ssm": {"grads": 2e-6}}
Inputs = tuple[Tensor, Tensor, Any, Tensor]


def _cfg(family: str, *overrides: str, unroll: int = T, batch: int = B) -> RunConfig:
    path, tiny, _ = TINY[family]
    sizes = [f"train.unroll={unroll}", f"train.batch={batch}", f"train.micro_batch={batch}"]
    return load_config(path, [*tiny, *sizes, *overrides])


def _ctx() -> DeviceContext:
    return resolve(load_hardware_profile("cpu"))


def _model(cfg: RunConfig, device: torch.device | str = "cpu") -> nn.Module:
    seed_everything(0)
    return build_model(cfg).train().to(device)


def _tensors(state: Any) -> list[Tensor]:
    """The tensors of a state: none (stateless), one (the GRU) or a tuple of them."""
    if state is None:
        return []
    return [state] if isinstance(state, Tensor) else list(state)


def _inputs(model: nn.Module, *, mid_reset: bool, batch: int = B, seed: int = 1) -> Inputs:
    """Tokens, contexts, a random incoming state and resets at ``t = 0`` (not in every row) and,
    with ``mid_reset``, inside the window; on the model's device."""
    device = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 256, (batch, T), generator=g)
    context = torch.randint(0, 16, (batch, T), generator=g)
    reset = torch.zeros(batch, T, dtype=torch.bool)
    reset[: batch - 1, 0] = True
    if mid_reset:
        reset[1, 5] = True
    state = model.init_state(batch, "cpu")  # type: ignore[operator]
    parts = [(torch.randn(s.shape, generator=g) * 0.1).to(device) for s in _tensors(state)]
    state = None if state is None else parts[0] if isinstance(state, Tensor) else tuple(parts)
    return tokens.to(device), context.to(device), state, reset.to(device)


def _run(fn: Any, model: nn.Module, inputs: Inputs) -> tuple[list[Tensor], list[Tensor]]:
    """``(values, grads)``: the logits and the new state, and the parameter gradients of a loss
    that reads all of them."""
    logits, state, _info = fn(*inputs)
    values = [logits, *_tensors(state)]
    loss = torch.stack([value.float().square().mean() for value in values]).sum()
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
    # copies: under CUDA graphs the outputs are graph memory, which the next window overwrites
    return [value.detach().float().clone() for value in values], [g.clone() for g in grads]


def _diff(got: tuple[list[Tensor], list[Tensor]], ref: tuple[list[Tensor], list[Tensor]]) -> Any:
    """The largest absolute differences, ``{"values": ..., "grads": ...}``, and ``"relative"``,
    the larger of the two as a fraction of the largest reference value of its kind."""
    diff = {
        name: max(float((a - b).abs().max()) for a, b in zip(g, r, strict=True))
        for name, g, r in (("values", got[0], ref[0]), ("grads", got[1], ref[1]))
    }
    largest = {name: max(float(r.abs().max()) for r in ref[i]) for i, name in enumerate(diff)}
    diff["relative"] = max(diff[name] / largest[name] for name in largest)
    return diff


def _tol(family: str) -> dict[str, float]:
    """The bounds of ``family``: TOL's, with its own from FAMILY_TOL in their place."""
    return {**TOL, **FAMILY_TOL.get(family, {})}


def _within(diff: dict[str, float], family: str, scale: float = 1.0) -> bool:
    return all(diff[name] <= scale * bound for name, bound in _tol(family).items())


# --------------------------------------------------------------------------------------------------
# The key and the sweep spec
# --------------------------------------------------------------------------------------------------

#: config_hash at main 80b893d (tests/test_p1_16_config.py), before every opt-in key since.
HASHES = {
    GRU_2M: "c8dcbaa48e3b27ecdde3d078359ab95f7482f49935d076bed34f521aeb88cf60",
    TRANSFORMER_2M: "65066a8915b79a328801b42780ae621f7185f23b553247a70c4f1b4dbd75071f",
}
#: The memory keys of configs/sweeps/p1_11_validation.yaml's arms at main 6bbf53d (before the
#: key), computed with that commit's code: the stored memory and step-time estimates still apply.
MEMORY_KEYS_6BBF53D = {
    "gru384x2_b32_lr3e-3/s0": "ae92d116e289909c",
    "gru128x1_b32_lr3e-3/s0": "4fa2b5e105235138",
    "gru64x1_b32_lr3e-3/s0": "81278b84b4dfab68",
    "hsn_pdg_b32_lr1e-3_bridge/s0": "2bc5d1c4e1d16e2d",
    "hsn_pdg_b32_lr1e-3_bridge_compiled/s0": "52f4f252a79f783d",
}


@pytest.mark.parametrize("path", list(HASHES), ids=lambda p: p.stem)
def test_key_is_opt_in_and_out_of_the_hash_at_its_default(path: Path) -> None:
    base = load_config(path)
    assert base.hardware.baseline_compile == "none"
    assert config_hash(base) == HASHES[path] and "baseline_compile" not in canonical_json(base)
    assert config_hash(load_config(path, ["hardware.baseline_compile=none"])) == HASHES[path]
    hashes = {HASHES[path]}
    for mode in MODES:
        on = load_config(path, [f"hardware.baseline_compile={mode}"])
        assert f'"baseline_compile":"{mode}"' in canonical_json(on)
        hashes.add(config_hash(on))
    assert len(hashes) == 1 + len(MODES)  # every mode is a configuration of its own
    with pytest.raises(ValueError, match="baseline_compile"):
        load_config(path, ["hardware.baseline_compile=fast"])


def test_key_is_for_the_baselines_in_the_scope() -> None:
    """An HSN compiles its window under ``hardware.compile_mode``: the key is refused there and
    taken by every baseline family (``selective_ssm`` among them) and by HSN-T without its
    memory, but not by HSN-T with it. The HSN's key keeps its meaning."""
    with pytest.raises(ValueError, match=r"hardware\.compile_mode"):
        load_config(REF_A1, ["hardware.baseline_compile=default"])
    for path in (GRU_2M, TRANSFORMER_2M, READER, MAMBA):
        on = load_config(path, ["hardware.baseline_compile=default"])
        assert on.model_name in BASELINE_COMPILE_MODEL_NAMES
    # HSN-T without its memory is a stateless model on the generic trainer and takes the key;
    # with its memory it is an HSN
    nomem = load_config(NOMEM, ["hardware.baseline_compile=default"])
    assert (nomem.model_name, nomem.model.memory) == ("hsn_t", False)
    with_memory = ["model.memory=true", "train.optimizer=lazy_adamw"]
    with pytest.raises(ValueError, match=r"hardware\.compile_mode"):
        load_config(NOMEM, [*with_memory, "hardware.baseline_compile=default"])
    cuda = load_config(GRU_2M, ["profile=rtx3060"])  # the profile compiles the HSN by default
    assert cuda.hardware.compile_mode == "reduce-overhead"
    assert cuda.hardware.baseline_compile == "none"


def _arms(tmp: Path, arms: list[dict[str, Any]]) -> dict[str, Any]:
    raw = {
        "name": "unit",
        "base": {"config": "configs/baselines/gru_2m.yaml", "overrides": ["train.batch=8"]},
        "protocol": {"name": "dry_run"},
        "seeds": [0],
        "arms": arms,
    }
    spec, _ = load_spec(write_yaml(tmp / "spec.yaml", raw))
    return {arm.name: arm for arm in expand(spec, repo=fake_repo(tmp))}


def test_spec_accepts_a_compiled_baseline_arm(tmp_path: Path) -> None:
    """A baseline arm may set the key (next to the pinned ``hardware.compile_mode=none``): it is
    an arm of its own with a memory key of its own, and the key at ``none`` is the arm it was.
    So may an arm of family ``hsn_t_nomem`` (HSN-T without its memory). An HSN arm may not set
    it, and the memory keys of the committed validation sweep are those of main before the key
    (and before ``model.memory``)."""
    cuda = ["profile=rtx3060", "hardware.compile_mode=none"]
    arms = _arms(
        tmp_path,
        [
            {"name": "eager", "family": "gru", "overrides": cuda},
            {
                "name": "off",
                "family": "gru",
                "overrides": [*cuda, "hardware.baseline_compile=none"],
            },
            {
                "name": "compiled",
                "family": "gru",
                "overrides": [*cuda, "hardware.baseline_compile=reduce-overhead"],
            },
            {
                "name": "nomem",
                "family": "hsn_t_nomem",
                "config": "configs/experiments/hsn_t_nomem.yaml",
                "overrides": [*cuda, "hardware.baseline_compile=reduce-overhead"],
            },
        ],
    )
    eager, off, compiled = arms["eager"], arms["off"], arms["compiled"]
    assert compiled.config.hardware.baseline_compile == "reduce-overhead"
    assert compiled.config.hardware.compile_mode == "none"
    assert arms["nomem"].family == "hsn_t_nomem" and arms["nomem"].variant == "none"
    assert arms["nomem"].config.hardware.baseline_compile == "reduce-overhead"
    assert (off.arm_id, off.memory_key) == (eager.arm_id, eager.memory_key)
    assert compiled.arm_id != eager.arm_id and compiled.memory_key != eager.memory_key
    hsn = {
        "name": "h",
        "family": "hsn",
        "config": "configs/experiments/g1_ref_a1.yaml",
        "overrides": ["hardware.baseline_compile=default"],
    }
    with pytest.raises(SpecError, match=r"hardware\.compile_mode"):
        _arms(tmp_path, [hsn])
    spec, _ = load_spec(REPO / "configs" / "sweeps" / "p1_11_validation.yaml")
    validation = expand(spec, repo=fake_repo(tmp_path))
    assert {arm.label: arm.memory_key for arm in validation} == MEMORY_KEYS_6BBF53D


# --------------------------------------------------------------------------------------------------
# The compiled window against the eager forward
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_compiled_equals_eager(family: str) -> None:
    """Forward and backward of the compiled window equal the eager model's to rounding, in
    training mode from a non-zero state with resets at the window start and inside the window,
    and in evaluation (``no_grad``, another batch size: a function of its own); once every
    function has compiled nothing compiles again, whatever the memory layout of the window (a
    compiled graph is specialised to its inputs' strides), and ``compile_seconds`` stands
    still."""
    cfg = _cfg(family, "hardware.baseline_compile=default")
    model = _model(cfg)
    compiled = compile_baseline(model, cfg, _ctx(), backend=CPU_BACKEND)
    assert isinstance(compiled, CompiledBaseline) and not compiled.cuda_graphs
    cases = [_inputs(model, mid_reset=False), _inputs(model, mid_reset=True, seed=2)]
    for inputs in cases:
        diff = _diff(_run(compiled, model, inputs), _run(model, model, inputs))
        assert _within(diff, family), diff
    with torch.no_grad():
        small = _inputs(model.eval(), mid_reset=True, batch=2, seed=3)
        ref_logits, ref_state, _ = model(*small)
        logits, state, _ = compiled(*small)
        model.train()
    pairs = zip([logits, *_tensors(state)], [ref_logits, *_tensors(ref_state)], strict=True)
    tol = _tol(family)
    bound = min(tol["values"], tol["relative"] * float(ref_logits.abs().max()))
    assert max(float((a - b).abs().max()) for a, b in pairs) <= bound

    report = compiled.report()
    assert report["functions"] == report["graphs"] == TINY[family][2], report  # one graph each
    assert report["mode"] == "default" and report["cuda_graphs"] is False
    assert report["evaluation"] == "compiled" and report["cudagraph_skips"] == 0
    assert compiled.calls == report["windows"] == 3 and compiled.compile_seconds > 0.0
    seconds = compiled.compile_seconds
    with torch.compiler.set_stance("fail_on_recompile"):
        for tokens, context, state, reset in cases:  # the same windows as non-contiguous views
            wide = [torch.stack([x, x], dim=-1)[..., 0] for x in (tokens, context, reset)]
            got = _run(compiled, model, (wide[0], wide[1], state, wide[2]))
            ref = _run(model, model, (tokens, context, state, reset))
            assert _within(_diff(got, ref), family)
    assert compiled.compile_seconds == seconds and compiled.calls == 5
    assert compiled.graphs == report["graphs"]


def test_plain_scan_equals_the_selective_scan() -> None:
    """The scan the compiled ``selective_ssm`` window runs (plain tensor operations, autograd's
    backward, resets at every position) against the eager one (in-place loop, hand-written
    backward), in float64 from a non-zero state with document starts at the first, a middle and
    the last position: the same values bit for bit, the same gradients to rounding."""
    g = torch.Generator().manual_seed(11)
    B_, T_, C, S = 2, 6, 3, 2

    def rand(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=g, dtype=torch.float64)

    delta, A = rand(B_, T_, C).abs() * 0.5 + 0.05, -rand(C, S).abs() - 0.1
    leaves = [delta, A, rand(B_, T_, C), rand(B_, T_, S), rand(B_, T_, S), rand(B_, C, S)]
    for leaf in leaves:
        leaf.requires_grad_()
    reset = torch.zeros(B_, T_, dtype=torch.bool)
    reset[0, 0] = reset[1, 3] = reset[0, T_ - 1] = True
    cols = tuple(reset.any(dim=0).nonzero().flatten().tolist())
    weights = rand(B_, T_, C), rand(B_, C, S)
    results = []
    for y, last in (_SelectiveScan.apply(*leaves, reset, cols), _plain_scan(*leaves, reset)):
        loss = (y * weights[0]).sum() + (last * weights[1]).sum()
        results.append((y, last, torch.autograd.grad(loss, leaves)))
    (y_a, last_a, grads_a), (y_b, last_b, grads_b) = results
    assert torch.equal(y_a, y_b) and torch.equal(last_a, last_b)
    for a, b in zip(grads_a, grads_b, strict=True):
        assert torch.allclose(a, b, rtol=1e-12, atol=1e-12), float((a - b).abs().max())


@pytest.mark.skipif(CPU_BACKEND != "inductor", reason="inductor needs a C++ toolchain")
@pytest.mark.parametrize("mode", MODES)
def test_every_mode_compiles_and_equals_eager(mode: str) -> None:
    """Each value of the key is a ``torch.compile`` mode this PyTorch takes (the GRU: its two
    graphs are the cheapest to compile). On the CPU none of them records CUDA graphs."""
    cfg = _cfg("gru", f"hardware.baseline_compile={mode}")
    model = _model(cfg)
    compiled = compile_baseline(model, cfg, _ctx())
    assert compiled is not None and compiled.mode == mode and not compiled.cuda_graphs
    inputs = _inputs(model, mid_reset=False)
    diff = _diff(_run(compiled, model, inputs), _run(model, model, inputs))
    assert _within(diff, "gru"), diff
    assert compiled.report()["backend"] == "inductor"


def test_scope_refusals() -> None:
    """Outside the scope the key raises with the reason: a model that is not one of the
    baselines, dropout, and gradient accumulation under CUDA graphs (a CUDA device with a mode
    that records them; the CPU and the modes without CUDA graphs take it)."""
    ctx = _ctx()
    on_cuda = dataclasses.replace(ctx, device=torch.device("cuda"))
    cfg = _cfg("gru", "hardware.baseline_compile=reduce-overhead")
    model = _model(cfg)
    check_baseline_scope(model, cfg, on_cuda)
    assert compile_baseline(model, _cfg("gru"), ctx) is None  # the default builds nothing
    with pytest.raises(NotImplementedError, match="not a baseline in the compiled scope"):
        check_baseline_scope(nn.Linear(2, 2), cfg, ctx)
    for family in ("gru", "transformer"):
        wet = _cfg(family, "hardware.baseline_compile=default", f"baseline.{family}.dropout=0.1")
        with pytest.raises(NotImplementedError, match="dropout 0.1"):
            compile_baseline(_model(wet), wet, ctx)
    for mode in MODES:
        graphs = baseline_compile._uses_cuda_graphs(on_cuda, mode, "inductor")
        assert graphs == (mode in CUDA_GRAPH_MODES)
        accum = _cfg("gru", f"hardware.baseline_compile={mode}", "train.grad_accum=2")
        check_baseline_scope(model, accum, ctx)  # the CPU records no CUDA graphs
        if graphs:
            with pytest.raises(NotImplementedError, match="grad_accum=2"):
                check_baseline_scope(model, accum, on_cuda)
        else:
            check_baseline_scope(model, accum, on_cuda)
    assert not baseline_compile._uses_cuda_graphs(on_cuda, "reduce-overhead", "aot_eager")


# --------------------------------------------------------------------------------------------------
# The trainer and the run records
# --------------------------------------------------------------------------------------------------

TRAIN = (
    "train.lr=0.003",
    "train.warmup_steps=2",
    "train.steps=40",
    "train.eval_every=20",
    "train.ckpt_every=40",
    "train.stats_every=20",
    "data.align_docs=false",  # windows continue documents: resets inside a window
)


@pytest.fixture
def aot_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """The trainer's and the runner's compiled windows through ``aot_eager`` (module
    docstring)."""
    from hsn import runner

    patched = functools.partial(compile_baseline, backend="aot_eager")
    monkeypatch.setattr(baseline_compile, "compile_baseline", patched)
    monkeypatch.setattr(runner, "compile_baseline", patched)


@pytest.mark.usefixtures("aot_eager")
@pytest.mark.parametrize("family", FAMILIES)
def test_trainer_trains_through_the_compiled_window(family: str, tmp_path: Path) -> None:
    """With the key on the generic trainer runs every window, training and evaluation, through
    the compiled window: 40 steps with a finite, falling loss that follows the eager trainer's
    step for step, the same evaluations, and a history that reports the compile as the HSN
    trainer's does (compile steps out of ``train_seconds``). At the default the trainer and its
    history are the ones they were."""
    train, dev = ByteDataset(docs("delayed", "train", 32)), ByteDataset(docs("delayed", "dev", 8))
    histories = {}
    for mode in ("none", "default"):
        cfg = _cfg(
            family,
            *TRAIN,
            "train.deterministic=true",
            f"hardware.baseline_compile={mode}",
            unroll=32,
            batch=4,
        )
        seed_everything(0)
        trainer = Trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / mode)
        trainer.eval_batch = 3  # an evaluation batch that is not the training batch
        histories[mode] = trainer.fit(train, dev, 40)
        compiled = trainer.compiled_baseline
        if mode == "none":
            assert compiled is None and trainer._eval_model() is trainer.model
        else:
            assert isinstance(compiled, CompiledBaseline) and trainer._eval_model() is compiled
            assert compiled.graphs == compiled.report()["functions"]  # nothing recompiled
    eager, comp = histories["none"], histories["default"]
    losses = [record.loss for record in comp.steps]
    assert all(torch.isfinite(torch.tensor(losses)))
    assert sum(losses[-10:]) < 0.9 * sum(losses[:10]), (losses[:10], losses[-10:])
    for a, b in zip(eager.steps, comp.steps, strict=True):
        assert abs(a.loss - b.loss) <= 1e-5 * max(1.0, abs(a.loss)), (a.step, a.loss, b.loss)
        assert abs(a.grad_norm - b.grad_norm) <= 1e-4 * max(1.0, abs(a.grad_norm)), a.step
    assert len(eager.evals) == len(comp.evals) == 2
    for a, b in zip(eager.evals, comp.evals, strict=True):
        assert a["n_positions"] == b["n_positions"] and a["n_windows"] == b["n_windows"]
        for key in ("loss", "accuracy"):
            assert abs(a[key] - b[key]) <= 1e-5 * max(1.0, abs(a[key])), (key, a[key], b[key])

    compile_keys = {"compile_seconds", "compile_steps", "window_compile_seconds"}
    assert eager.measured["compiled_windows"] == 0.0 and not compile_keys & set(eager.measured)
    assert set(comp.measured) == set(eager.measured) | compile_keys
    windows = sum(e["n_windows"] for e in comp.evals)
    assert comp.measured["compiled_windows"] == 40.0 + windows
    assert comp.measured["compile_steps"] >= 1.0 and comp.measured["compile_seconds"] > 0.0
    assert comp.measured["window_compile_seconds"] > 0.0
    stamped = sum(record.seconds for record in comp.steps)
    assert abs(stamped - comp.measured["train_seconds"] - comp.measured["compile_seconds"]) < 1e-6


@pytest.mark.usefixtures("aot_eager")
def test_harness_record_reports_the_compile(tiny_manifest: Path, tmp_path: Path) -> None:  # noqa: F811
    """The sweep harness's record of a compiled baseline arm: the ``compile`` block carries the
    baseline's mode and scope next to the HSN's (still ``none``), the counts are the compiled
    windows', the probes ran on the eager model; the default arm's record has no such entry."""
    records = {}
    for mode in ("none", "default"):
        overrides = [*disk_overrides(tiny_manifest, steps=6), f"hardware.baseline_compile={mode}"]
        arm = ArmRun(
            cfg=load_config(GRU_2M, [*TINY_GRU_OVERRIDES, *overrides]),
            family="gru",
            variant="none",
            variant_params={},
            params=protocol_params(
                {"name": "reads_state", "diag_every": 2, "diag_docs": 8, "probe_batch": 8}
            ),
            run_dir=tmp_path / mode,
            rules=parse_rules(None),
            constants={},
            on_progress=lambda progress: None,
        )
        records[mode] = run_protocol(arm)
    eager, comp = records["none"]["compile"], records["default"]["compile"]
    assert "baseline" not in eager and eager["compiled_windows"] == 0.0
    assert eager["compile_seconds"] is None and eager["compile_steps"] is None
    assert comp["mode"] == "none" and comp["compiled_window"] is False  # the HSN's window
    baseline = comp["baseline"]
    assert baseline["mode"] == "default" and baseline["cuda_graphs"] is False
    assert baseline["backend"] == "aot_eager" and baseline["evaluation"] == "compiled"
    assert "nn.GRU recurrence" in baseline["scope"] and baseline["graphs"] == baseline["functions"]
    assert comp["compiled_windows"] >= 6.0 and comp["compile_steps"] >= 1.0
    assert comp["compile_seconds"] > 0.0 and comp["window_compile_seconds"] > 0.0
    record = records["default"]
    assert abs(record["walls"]["compile_s"] - comp["compile_seconds"]) < 1e-3  # walls round
    assert record["probe_execution"] == "eager" and record["probe"]["n_probed"] > 0
    assert record["final"]["steps"] == 6 and json.dumps(record["compile"])  # plain data


@pytest.mark.usefixtures("aot_eager")
def test_runner_record_reports_the_compile(tiny_manifest: Path, tmp_path: Path) -> None:  # noqa: F811
    """``hsn run`` with the key: ``metrics.json`` says what of the baseline ran compiled, the
    measured numbers carry the compile, the resolved configuration keeps the key, and the profile
    pass runs the compiled windows too (its first windows are the ``compile`` segment)."""
    overrides = [*TINY_GRU_OVERRIDES, *disk_overrides(tiny_manifest, steps=6, eval_every=6)]

    def run(name: str, mode: str) -> Path:
        extra = [f"hardware.baseline_compile={mode}", f"run_id=gru_{name}"]
        runs = run_seeds(GRU_2M, [0], out=tmp_path, overrides=[*overrides, *extra], profile_steps=2)
        return runs[0]

    def read(run_dir: Path, name: str) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads((run_dir / name).read_text(encoding="utf-8"))
        return loaded

    eager_dir, comp_dir = run("eager", "none"), run("compiled", "default")
    eager, comp = read(eager_dir, "metrics.json"), read(comp_dir, "metrics.json")
    assert "baseline_compile" not in eager and eager["measured"]["compiled_windows"] == 0.0
    assert "compile_seconds" not in eager["measured"]
    report = comp["baseline_compile"]
    assert report["mode"] == "default" and report["cuda_graphs"] is False
    assert "nn.GRU recurrence" in report["scope"] and report["windows"] >= 6
    assert comp["measured"]["compile_steps"] >= 1.0 and comp["measured"]["compile_seconds"] > 0.0
    assert read(comp_dir, "run_manifest.json")["measured"]["compile_steps"] >= 1.0
    assert abs(comp["final"]["loss"] - eager["final"]["loss"]) <= 1e-5
    resolved = load_config(comp_dir / "config.resolved.yaml")
    assert resolved.hardware.baseline_compile == "default"
    for name in ("profile_train.json", "profile_serve.json"):
        compiled_segments = read(comp_dir, name)["segments"]
        eager_segments = read(eager_dir, name)["segments"]
        assert "compile" in compiled_segments and "startup" not in compiled_segments
        assert "startup" in eager_segments and "compile" not in eager_segments


# --------------------------------------------------------------------------------------------------
# CUDA (skipped without a card; never run when this file was written: no card was free)
# --------------------------------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["reduce-overhead", "default"])
def test_compiled_equals_eager_cuda(family: str, mode: str) -> None:
    """On the card, in fp32: the compiled window (``reduce-overhead``: warm-up, recording and
    replay of its CUDA graphs) equals the eager forward and backward to rounding, PyTorch replays
    every graph as a CUDA graph (no skip), and a warm window compiles nothing."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    ctx = dataclasses.replace(resolve(load_hardware_profile(CUDA_PROFILE)), autocast_dtype=None)
    cfg = _cfg(family, f"hardware.baseline_compile={mode}")
    model = _model(cfg, ctx.device)
    compiled = compile_baseline(model, cfg, ctx)
    assert compiled is not None and compiled.cuda_graphs == (mode == "reduce-overhead")
    inputs = _inputs(model, mid_reset=False)
    ref = _run(model, model, inputs)
    for _ in range(3):  # compile and warm-up, recording, replay
        got = _run(compiled, model, inputs)
    seconds = compiled.compile_seconds
    with torch.compiler.set_stance("fail_on_recompile"):
        got = _run(compiled, model, inputs)
    # CUDA kernels round differently from the CPU's: 10x the CPU tolerance until measured
    assert _within(_diff(got, ref), family, scale=10.0), _diff(got, ref)
    report = compiled.report()
    assert compiled.compile_seconds == seconds
    assert report["cudagraph_skips"] == 0 and report["graphs"] == report["functions"], report


@pytest.mark.cuda
@pytest.mark.parametrize("family", FAMILIES)
def test_trainer_trains_with_cuda_graphs(family: str, tmp_path: Path) -> None:
    """On the card under the profile's autocast: 40 steps and two evaluations through
    ``reduce-overhead`` (CUDA graphs) with a finite, falling loss; the first two steps compile
    and record, and the last evaluation stays close to the eager trainer's."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    train, dev = ByteDataset(docs("delayed", "train", 32)), ByteDataset(docs("delayed", "dev", 8))
    histories = {}
    for mode in ("none", "reduce-overhead"):
        pins = [f"profile={CUDA_PROFILE}", "hardware.compile_mode=none"]
        cfg = _cfg(family, *TRAIN, *pins, f"hardware.baseline_compile={mode}", unroll=32, batch=4)
        seed_everything(0)
        trainer = Trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / mode)
        histories[mode] = trainer.fit(train, dev, 40)
        if mode != "none":
            compiled = trainer.compiled_baseline
            assert compiled is not None and compiled.cuda_graphs
            assert compiled.report()["cudagraph_skips"] == 0
    eager, comp = histories["none"], histories["reduce-overhead"]
    losses = [record.loss for record in comp.steps]
    assert all(torch.isfinite(torch.tensor(losses)))
    assert sum(losses[-10:]) < 0.9 * sum(losses[:10])
    assert comp.measured["compile_steps"] >= 2.0
    # bf16 autocast: the two runs drift apart by rounding, so the evaluations are compared loosely
    assert abs(comp.evals[-1]["loss"] - eager.evals[-1]["loss"]) <= 0.1 * eager.evals[-1]["loss"]
