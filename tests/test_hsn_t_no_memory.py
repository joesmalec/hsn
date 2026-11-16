"""HSN-T without its memory (``model.memory = false``, ``docs/hsn_t_no_memory.md``).

The switch is opt-in and HSN-T only. Off, it builds :class:`hsn.model.hsn_t.HSNTNoMemory`: the
parts of HSN-T that are not the memory (embeddings, ``E``, the reader, ``S``, ``D``), over all
positions of a window at once and without state. These tests check that the default leaves
every configuration, hash and model as it was, what the memory-off model may read, that it is
HSN-T with every read of the state held at zero (a trained HSN-T's weights load unchanged), its
parameter count, the settings it refuses, its experiment family, that the generic trainer
trains it, and that it compiles as one graph.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import (
    HASH_OMIT_AT_DEFAULT,
    RunConfig,
    canonical_json,
    config_hash,
    load_config,
)
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import SpecError, expand, family_of, load_spec, memory_key
from hsn.model.hsn import HSN
from hsn.model.hsn_t import HSNT, MEMORY_OFF_MODULES, HSNTNoMemory
from hsn.runner import build_model, make_trainer
from hsn.train.loop import SequenceModel, Trainer
from hsn.train.memory_plan import expected_param_count, memory_off_param_count
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import fake_repo, restore_determinism, write_yaml  # noqa: F401
from p1_15_support import datasets, train_overrides
from test_a6_feedback_detach import A5, _cfg, _start, _window
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "configs" / "experiments" / "hsn_t_nomem.yaml"
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
GRU_2M = REPO / "configs" / "baselines" / "gru_2m.yaml"
READER_LM = REPO / "configs" / "baselines" / "reader_w64l2.yaml"

W = 8  # the reader's width on the six-cell model of the A6 tests (d = 8)
READER = ("model_name=hsn_t", "model.reader=transformer", f"model.reader_width={W}")
#: the six-cell HSN-T of the A6 tests, and what switches its memory off
ON = (*A5, *READER, "model.reader_heads=2")
OFF = ("model.memory=false", "train.optimizer=adamw", "train.lambda_p=0", "train.gate_mode=soft")
#: the memory-off recipe at a size the CPU trains in a moment
TINY = ("model.d=8", "model.context_dim=8", "model.hidden=16", "model.reader_width=16")


def _gap(a: Tensor, b: Tensor) -> float:
    return float((a - b).detach().abs().max())


def _off_model(cfg: RunConfig, dtype: torch.dtype = torch.float64) -> HSNTNoMemory:
    """The memory-off model away from its start, so every input column carries a weight."""
    seed_everything(0)
    model = build_model(cfg).to(dtype)
    assert type(model) is HSNTNoMemory
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return model.eval()


# --------------------------------------------------------------------------------------------------
# The key: opt-in, hash, and the model with its memory unchanged
# --------------------------------------------------------------------------------------------------


def test_key_is_opt_in_and_the_default_keeps_its_hash() -> None:
    base = load_config(RECIPE_A6)
    assert base.model.memory is True and "memory" not in json.loads(canonical_json(base))["model"]
    assert config_hash(load_config(RECIPE_A6, ["model.memory=true"])) == config_hash(base)
    assert "model.memory" in HASH_OMIT_AT_DEFAULT  # append only: later keys follow it
    hybrid = ["model_name=hsn_t", "model.reader=transformer", "model.k_select=12"]
    on = load_config(RECIPE_A6, hybrid)
    assert config_hash(load_config(RECIPE_A6, [*hybrid, "model.memory=true"])) == config_hash(on)
    # the arm id reads the non-default values: the key is not among them
    assert "memory" not in on.model.model_dump(exclude_defaults=True)
    off = load_config(RECIPE)
    assert off.model.memory is False and json.loads(canonical_json(off))["model"]["memory"] is False
    assert off.train.optimizer == "adamw" and off.train.lambda_p == 0.0
    # the memory key of the experiment database (it hashes the whole model block), as main
    # 6bbf53d computed it before the key existed: the stored estimates still match
    protocol = {"name": "reads_state", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
    assert memory_key(base, "pdg", protocol) == "c33ec88833506533"
    assert memory_key(on, "pdg", protocol) == "c8bcf4024c9285fe"
    assert memory_key(load_config(GRU_2M), "none", protocol) == "b7c9c795e3ff94f5"
    assert memory_key(off, "none", protocol) != memory_key(
        load_config(RECIPE, ["model.memory=true", "train.lambda_p=1"]), "none", protocol
    )
    # without a unit bank the optimiser's default does not follow the routing
    routed = ["model_name=hsn_t", "model.routing=topk", "model.selected_readout=true"]
    assert load_config(READER_LM, routed).train.optimizer == "lazy_adamw"
    unset = load_config(READER_LM, [*routed, "model.memory=false", "train.lambda_p=0"])
    assert unset.train.optimizer == "adamw"


def test_with_its_memory_the_model_is_built_and_trained_as_before(tmp_path: Path) -> None:
    cfg = _cfg(*ON)
    seed_everything(0)
    model = build_model(cfg)
    assert type(model) is HSNT and isinstance(model, HSN)
    report, expected = model.param_report(), expected_param_count(cfg.model)
    assert all(report[row] == count for row, count in expected.items())
    assert type(make_trainer(model, cfg, resolve(cfg.hardware), run_dir=tmp_path)) is TBPTTTrainer
    seed_everything(0)
    named = build_model(_cfg(*ON, "model.memory=true"))
    for (name, p), (_, q) in zip(model.named_parameters(), named.named_parameters(), strict=True):
        assert torch.equal(p, q), name
    # the two classes are kept apart: neither builds the other's configuration
    off = _cfg(*ON, *OFF)
    for cls in (HSN, HSNT):
        with pytest.raises(ValueError, match="a model of its own"):
            cls(off.model, seed=0, unroll=12)
    with pytest.raises(ValueError, match="with its memory the model is"):
        HSNTNoMemory(cfg.model, seed=0)
    with pytest.raises(TypeError, match="HSN"):
        TBPTTTrainer(build_model(off), off, resolve(off.hardware), run_dir=tmp_path)


# --------------------------------------------------------------------------------------------------
# The memory-off model
# --------------------------------------------------------------------------------------------------


def test_it_is_stateless_and_has_no_memory_modules() -> None:
    cfg = _cfg(*ON, *OFF)
    model = _off_model(cfg)
    assert isinstance(model, SequenceModel) and not isinstance(model, HSN)
    assert [name for name, _ in model.named_children()] == list(MEMORY_OFF_MODULES)
    assert not hasattr(model, "cfg")  # the evaluation loop reads cfg.key_address
    tokens, ctx, reset = _window(cfg)
    logits, state, info = model(tokens, ctx, None, reset)
    assert logits.shape == (*tokens.shape, cfg.model.vocab_size) and state is None and info == {}
    assert model.init_state(3, "cpu") is None
    assert model.reset_state(None, torch.zeros(3, dtype=torch.bool)) is None
    with pytest.raises(ValueError, match="carries no state"):
        model(tokens, ctx, object(), reset)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="context must have shape"):
        model(tokens, ctx[:, :-1], None, reset)


@pytest.mark.parametrize("span", ["null", "4"])
def test_logits_read_no_later_byte_and_nothing_before_a_reset(span: str) -> None:
    cfg = _cfg(*ON, *OFF, f"model.reader_span={span}")
    model = _off_model(cfg)
    tokens, ctx, reset = _window(cfg)
    k, t0 = 4, 8
    reset[0, k] = True  # row 0 starts a document at column k; the other rows do not
    with torch.no_grad():
        logits = model(tokens, ctx, None, reset)[0]
        later, later_ctx = tokens.clone(), ctx.clone()
        later[:, t0:] = (later[:, t0:] + 5) % cfg.model.vocab_size
        later_ctx[:, t0:] = (later_ctx[:, t0:] + 3) % cfg.model.n_context
        changed = model(later, later_ctx, None, reset)[0]
        assert _gap(changed[:, :t0], logits[:, :t0]) <= 1e-12
        assert float((changed[:, t0:] - logits[:, t0:]).abs().amax(dim=-1).min()) > 1e-6
        before = tokens.clone()
        before[:, :k] = (before[:, :k] + 9) % cfg.model.vocab_size
        changed = model(before, ctx, None, reset)[0]
        assert _gap(changed[0, k:], logits[0, k:]) <= 1e-12
        assert _gap(changed[1, k], logits[1, k]) > 1e-6  # a row without the reset reads back
        # the context symbol is read, at its own position only
        other = ctx.clone()
        other[:, t0] = (other[:, t0] + 1) % cfg.model.n_context
        changed = model(tokens, other, None, reset)[0]
        assert _gap(changed[:, t0], logits[:, t0]) > 1e-6
        rest = [t for t in range(tokens.shape[1]) if t != t0]
        assert _gap(changed[:, rest], logits[:, rest]) <= 1e-12


@pytest.mark.parametrize("training", [False, True])
def test_a_trained_hsn_ts_weights_load_and_give_its_logits_with_the_memory_at_zero(
    training: bool,
) -> None:
    """The memory-off model is HSN-T with every read of the state removed: with the weights of
    an HSN-T loaded (the memory's tensors left out) its logits are the logits of that HSN-T's
    own window when ``S`` reads zeros for ``q_{t−1}`` and ``D`` reads zeros for ``q_t`` and
    ``r_t``. The HSN-T still runs its cells, router and predictor; nothing they compute reaches
    the logits."""
    cfg = _cfg(*ON)
    seed_everything(0)
    full = HSNT.from_config(cfg).double()
    with torch.no_grad():
        for p in full.parameters():
            p.add_(0.3 * torch.randn_like(p))
    full.train(training)
    off = build_model(_cfg(*ON, *OFF)).double().train(training)
    assert isinstance(off, HSNTNoMemory)
    state_dict = full.state_dict()
    left_out = {key.split(".")[0] for key in set(state_dict) - set(off.state_dict())}
    assert left_out == {"predictor", "cell", "router", "halting"}
    off.load_memory_on_state(state_dict)
    for name, value in off.state_dict().items():
        assert torch.equal(value, state_dict[name]), name

    d = cfg.model.d
    decode = full._decode

    def decode_without_memory(q: Tensor, u: Tensor, readout: Tensor | None) -> Tensor:
        assert readout is not None and readout.shape[-1] == d + W  # [r ; c]
        r, c = readout[..., :d], readout[..., d:]
        return decode(torch.zeros_like(q), u, torch.cat([torch.zeros_like(r), c], dim=-1))

    full._decode = decode_without_memory  # type: ignore[method-assign]
    full._summary_read = torch.zeros_like  # type: ignore[method-assign]
    tokens, ctx, reset = _window(cfg)
    reset[1, 6] = True
    with torch.no_grad():
        reference = full(tokens, ctx, _start(full, tokens.shape[0]), reset)[0]
        logits = off(tokens, ctx, None, reset)[0]
    assert _gap(logits, reference) <= 1e-12
    # and the memory does matter to the HSN-T itself: the comparison is not empty
    del full._decode, full._summary_read
    with torch.no_grad():
        assert _gap(full(tokens, ctx, _start(full, tokens.shape[0]), reset)[0], logits) > 1e-3
    # a state dict without one of the six modules is refused
    with pytest.raises(RuntimeError, match="Missing key"):
        off.load_memory_on_state({k: v for k, v in state_dict.items() if "reader" not in k})


def test_it_starts_from_hsn_ts_draws_where_they_do_not_depend_on_the_memory() -> None:
    seed_everything(0)
    full = dict(HSNT.from_config(_cfg(*ON)).named_parameters())
    seed_everything(0)
    off = dict(build_model(_cfg(*ON, *OFF)).named_parameters())
    d = _cfg(*ON).model.d
    for name, p in off.items():
        if name.split(".")[0] in ("byte_emb", "ctx_emb", "encoder", "reader"):
            assert torch.equal(p, full[name]), name
    # the columns of c in S, of r and c in D; the rest of the two networks are other draws
    for name, first in (("shared.fc1.weight", -W), ("decoder.fc1.weight", 2 * d)):
        assert torch.equal(off[name][:, first:], full[name][:, first:]), name
        assert not torch.equal(off[name][:, :first], full[name][:, :first]), name


def test_param_report_matches_the_closed_form() -> None:
    cfg = load_config(RECIPE)
    model = build_model(cfg)
    assert isinstance(model, HSNTNoMemory)
    report, expected = model.param_report(), memory_off_param_count(cfg.model)
    assert all(report[name] == expected[name] for name in MEMORY_OFF_MODULES)
    total = sum(p.numel() for p in model.parameters())
    assert report["stored"] == report["active"] == report["trainable"] == total
    assert expected["total"] == total == 627_328
    # S and D are HSN-T's at HSN-T's widths
    hybrid = expected_param_count(
        load_config(RECIPE, ["model.memory=true", "train.lambda_p=1"]).model
    )
    assert (expected["shared"], expected["decoder"]) == (hybrid["shared"], hybrid["decoder"])
    assert model.shared.fc1.in_features == 2 * 64 + 64 + 64
    assert model.decoder.fc1.in_features == 3 * 64 + 64
    # the widened readers of the document: about the GRU baseline's 1,977,856
    for width, layers, count in ((224, 2, 1_942_528), (184, 3, 1_907_192), (160, 4, 1_887_552)):
        keys = [f"model.reader_width={width}", f"model.reader_layers={layers}"]
        wide = load_config(RECIPE, keys)
        assert memory_off_param_count(wide.model)["total"] == count
        assert count == 445_312 + 1_282 * width + layers * (12 * width**2 + 13 * width)
        assert build_model(wide).param_report()["stored"] == count
    with pytest.raises(ValueError, match="has a reader"):
        memory_off_param_count(load_config(RECIPE_A6).model)


# --------------------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        # the switch belongs to HSN-T
        (RECIPE_A6, ["model.memory=false"], "set model.reader=transformer"),
        (GRU_2M, ["model.memory=false"], "set model.reader=transformer"),
        (READER_LM, ["model.memory=false"], "model_name='reader' has no such switch"),
        (RECIPE, ["model.selected_readout=false", "model.routing=dense"], "selected_readout"),
        # the memory's reads of S and D
        (RECIPE, ["model.fast_cells=2", "model.fast_readout=true"], "cells of the memory"),
        (RECIPE, ["model.decoder_shortcut_dropout=0.1"], "there are none"),
        # training settings that need the memory
        (RECIPE, ["train.lambda_p=1.0"], r"train.lambda_p=1.0 \(L_pred"),
        (RECIPE, ["train.lambda_u=0.1"], "train.lambda_u=0.1"),
        (RECIPE, ["model.write_gate=true", "train.lambda_w=0.1"], "train.lambda_w=0.1"),
        (RECIPE, ["train.lambda_c=0.1", "train.aux_enabled=[cost]"], "the routing penalties"),
        (RECIPE, ["train.aux_enabled=[balance]"], r"train.aux_enabled=\['balance'\]"),
        (RECIPE, ["train.route_guard.enabled=true"], "train.route_guard.enabled=true"),
        (RECIPE, ["train.feedback_detach=[summary,eps]"], "the state's feedback reads"),
        (RECIPE, ["train.state_grad_clip=1.0"], "the gradient into the state"),
        (RECIPE, ["train.unit_credit=executed"], "the routed units' credit"),
        (RECIPE, ["train.ste_scope=all"], "straight-through scope"),
        (RECIPE, ["train.gate_mode=ste"], "the routed selection's gradient path"),
        (RECIPE, ["train.soft_until_step=100"], "soft warm-up"),
        (RECIPE, ["train.epsilon_start=0.1", "train.epsilon_end=0.1"], "exploration"),
        (RECIPE, ["train.p_dense=0.1"], "dense fallback"),
        (RECIPE, ["train.cell_lr_mult=0.5"], "the cell's parameter group"),
        (RECIPE, ["train.router_lr_mult=0.5"], "the router's parameter group"),
        (RECIPE, ["train.optimizer=lazy_adamw"], "the unit bank's groups"),
        (
            RECIPE,
            ['train.update_limit.groups={memory: {selectors: ["cell.*"], max_ratio: 0.003}}'],
            r"train.update_limit.groups.memory selects \['cell.\*'\]",
        ),
        (
            RECIPE,
            ['train.gradient_growth.groups={routing: ["router.*", "decoder.*"]}'],
            r"train.gradient_growth.groups.routing selects \['router.\*'\]",
        ),
        (RECIPE, ["hardware.window=fused"], "is not the HSN"),
    ],
)
def test_refusals(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(path, overrides)


def test_one_refusal_names_every_setting_and_the_a6_recipe_names_its_own() -> None:
    hybrid = ["model_name=hsn_t", "model.reader=transformer", "model.memory=false"]
    with pytest.raises(ValueError) as caught:
        load_config(RECIPE_A6, hybrid)
    text = str(caught.value)
    for part in (
        "train.lambda_p=1.0",
        "train.feedback_detach=",
        "train.gate_mode=ste",
        "train.route_guard.enabled=true",
        "train.optimizer=lazy_adamw",
        "train.update_limit.groups.memory",
        "train.update_limit.groups.routing",
    ):
        assert part in text, part


# --------------------------------------------------------------------------------------------------
# The experiment family
# --------------------------------------------------------------------------------------------------


def _arms(tmp: Path, arm: dict[str, Any]) -> list[Any]:
    raw = {
        "name": "unit",
        "base": {"config": "configs/experiments/hsn_t_nomem.yaml", "overrides": []},
        "protocol": {"name": "train_eval", "steps": 40},
        "arms": [{"name": "arm", **arm}],
    }
    path = write_yaml(tmp / f"spec_{len(list(tmp.glob('spec_*')))}.yaml", raw)
    return expand(load_spec(path)[0], repo=fake_repo(tmp))


def test_the_spec_takes_it_as_a_family_of_its_own(tmp_path: Path) -> None:
    assert family_of(load_config(RECIPE)) == "hsn_t_nomem"
    assert family_of(load_config(RECIPE_A6)) == "hsn"
    hybrid = ["model_name=hsn_t", "model.reader=transformer"]
    assert family_of(load_config(RECIPE_A6, hybrid)) == "hsn_t"
    (named,) = _arms(tmp_path, {"family": "hsn_t_nomem"})
    (inferred,) = _arms(tmp_path, {})
    assert named.family == inferred.family == "hsn_t_nomem" and named.arm_id == inferred.arm_id
    assert named.variant == "none" and named.config.model.memory is False
    (wide,) = _arms(tmp_path, {"overrides": ["model.reader_width=224"]})
    assert wide.arm_id != named.arm_id
    with pytest.raises(SpecError, match=r"model.memory=false \(family hsn_t_nomem\)"):
        _arms(tmp_path, {"family": "hsn_t"})
    with pytest.raises(SpecError, match="family hsn_t_nomem but the configuration"):
        _arms(tmp_path, {"family": "hsn_t_nomem", "config": str(READER_LM)})
    with pytest.raises(SpecError, match="trains eager"):  # a baseline to the harness
        _arms(tmp_path, {"overrides": ["hardware.compile_mode=reduce-overhead"]})


# --------------------------------------------------------------------------------------------------
# Training and compiling
# --------------------------------------------------------------------------------------------------


def test_the_generic_trainer_trains_it(tmp_path: Path) -> None:
    """Thirty steps of the recipe at a tiny size; ``key_address`` is set as an HSN-T's
    configuration has it, which the evaluation loop must not take for a model with addresses."""
    steps = 30
    extra = ["model.reader_heads=2", "model.key_address=true", "train.lr=3e-3"]
    cfg = load_config(RECIPE, overrides=[*train_overrides(steps), *TINY, *extra])
    seed_everything(0)
    trainer = make_trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    assert type(trainer) is Trainer and type(trainer.model) is HSNTNoMemory
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.05
    assert history.param_report["stored"] == memory_off_param_count(cfg.model)["total"]
    assert len(history.evals) == 1 and math.isfinite(history.evals[0]["loss"])


def test_the_generic_trainers_own_controllers_run_on_it(tmp_path: Path) -> None:
    """What the generic trainer does for any model stays available: the weight average, the
    spike guard, the decoder's learning-rate group and an update limit on the model's own
    parameter names (the reader's here)."""
    steps = 12
    kept = ["train.weight_ema=0.9", "train.spike_factor=1000", "train.readout_lr_mult=0.5"]
    limit = 'train.update_limit.groups={reader: {selectors: ["reader.*"], max_ratio: 0.01}}'
    sizes = [*TINY, "model.reader_heads=2", "train.lr=3e-3"]
    cfg = load_config(RECIPE, overrides=[*train_overrides(steps), *sizes, *kept, limit])
    seed_everything(0)
    trainer = make_trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    assert type(trainer) is Trainer
    assert trainer.weight_ema is not None and trainer.update_limiter is not None
    decoder = {id(p) for p in trainer.model.decoder.parameters()}
    rates = {
        group["initial_lr"]
        for group in trainer.optimizer.param_groups
        if id(group["params"][0]) in decoder
    }
    assert rates == {0.5 * cfg.train.lr}
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    assert losses[-1] < losses[0]


def test_forward_and_backward_compile_as_one_graph_and_match_eager() -> None:
    """``torch.compile`` with the ``aot_eager`` backend and ``fullgraph``: no graph break, so
    no data-dependent branch, shape or host read in the forward; the logits and every gradient
    equal the eager ones. A second window of the same shape reuses the graph."""
    from torch._dynamo.utils import counters

    cfg = load_config(RECIPE, [*TINY, "model.reader_heads=2"])
    seed_everything(0)
    model = build_model(cfg).train()
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    g = torch.Generator().manual_seed(3)
    try:
        graphs = None
        for _ in range(2):
            tokens = torch.randint(0, 256, (3, 24), generator=g)
            ctx = torch.randint(0, cfg.model.n_context, (3, 24), generator=g)
            reset = torch.rand(3, 24, generator=g) < 0.1
            reset[:, 0] = True
            results = []
            for forward in (model, compiled):
                model.zero_grad(set_to_none=True)
                logits, state, info = forward(tokens, ctx, None, reset)
                logits.square().mean().backward()
                grads = {n: p.grad.clone() for n, p in model.named_parameters()}
                results.append((logits.detach(), grads))
                assert state is None and info == {}
            (ref, ref_grads), (got, got_grads) = results
            assert _gap(got, ref) <= 1e-6
            assert all(_gap(got_grads[n], ref_grads[n]) <= 1e-6 for n in ref_grads)
            assert all(float(grad.abs().sum()) > 0.0 for grad in ref_grads.values())
            if graphs is not None:
                assert counters["stats"]["unique_graphs"] == graphs, "the second window recompiled"
            graphs = counters["stats"]["unique_graphs"]
    finally:
        torch._dynamo.reset()
