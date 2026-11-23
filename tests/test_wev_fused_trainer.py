"""``hardware.event_kernel=fused``: the fused word-event block in training (phase 2 of
``docs/wev_fused_event.md``; the block itself is ``tests/test_wev_fused_event.py``).

* The key is opt-in, out of the configuration hash, the arm id and the memory key at its
  default, and refused at configuration load outside the block's scope, each refusal with its
  reason (the CPU among them: the kernels are CUDA only).
* The trainer runs the block on the training steps with the straight-through gate and the
  compiled window everywhere else: evaluation, a soft warm-up's steps, a statistics step's
  term passes.
* On the CPU the trainer with the block's PyTorch reference equals the eager trainer in
  float64 over some optimiser steps: the losses, every step's raw gradient, the updated
  parameters, the statistics (the refresh rate among them, which trim would otherwise move)
  and the term gradient norms.
* ``train.tbptt_windows > 1``: a window's place in its segment picks its buffers, and the
  gradient of a later window's loss reaches the window before through the block's start state.
* On the card (fp32, inputs whose every decision has a margin): the trainer with the key
  against without it, the same decisions at every window, losses and gradients to the
  compiled window's tolerance; two windows a segment on two sets of buffers; bf16.
* The default path is untouched: no fused object, the same windows.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import (
    HASH_OMIT_AT_DEFAULT,
    REPO_ROOT,
    RunConfig,
    canonical_json,
    config_hash,
    event_kernel_scope_error,
    load_config,
)
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import config_view, memory_key
from hsn.kernels import window as window_kernels
from hsn.losses import TERM_GRAD_NORM_PREFIX
from hsn.model.hsn import HSN
from hsn.runner import build_model
from hsn.train import compile as compile_mod
from hsn.train import event_fused as ef
from hsn.train import tbptt
from hsn.train.compile import CompiledWindow, compile_window, fused_events_of
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import restore_determinism  # noqa: F401
from test_p2_8_compile import _release_compiled_state  # noqa: F401
from test_tbptt_windows import (
    STREAM,
    _capture_grads,
    _capture_params,
    _cfg,
    _long_datasets,
    _params,
)
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

EXPERIMENTS = REPO_ROOT / "configs" / "experiments"
#: the field recipe, and the two it grew from
RECIPE = EXPERIMENTS / "hsn_t_memory_words_carry_bal.yaml"
RECIPES = (RECIPE, EXPERIMENTS / "hsn_t_memory_words_carry.yaml")
#: a CUDA arm's lines (the spec's), without and with the key
CARD = (
    "profile=rtx3060",
    "hardware.compile_mode=reduce-overhead",
    "hardware.graph_alias_index=true",
    "hardware.message_gradients=state",
    "hardware.round_products=merged",
)
KEY = "hardware.event_kernel=fused"


# --------------------------------------------------------------------------------------------------
# The key
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: p.stem)
def test_key_is_opt_in_and_keeps_the_default_hash(recipe: Path) -> None:
    base = load_config(recipe, list(CARD))
    assert base.hardware.event_kernel == "compiled"
    assert "hardware.event_kernel" in HASH_OMIT_AT_DEFAULT
    named = load_config(recipe, [*CARD, "hardware.event_kernel=compiled"])
    assert config_hash(named) == config_hash(base)
    assert "event_kernel" not in canonical_json(base)
    fused = load_config(recipe, [*CARD, KEY])
    assert fused.hardware.event_kernel == "fused"
    assert config_hash(fused) != config_hash(base)
    assert '"event_kernel":"fused"' in canonical_json(fused)
    # nothing else of the configuration moves with the key
    a, b = fused.model_dump(mode="json"), base.model_dump(mode="json")
    assert a["hardware"].pop("event_kernel") == "fused"
    assert b["hardware"].pop("event_kernel") == "compiled"
    assert a == b
    assert event_kernel_scope_error(base) is None


def test_arm_id_and_memory_key_hold_the_key_only_when_set() -> None:
    """An arm without the key keeps its identity and its stored step-time and memory
    estimates; an arm with it is another arm with estimates of its own."""
    protocol = {"name": "train_eval", "eval_batch": 32}
    base = load_config(RECIPE, list(CARD))
    named = load_config(RECIPE, [*CARD, "hardware.event_kernel=compiled"])
    fused = load_config(RECIPE, [*CARD, KEY])
    assert config_view(named) == config_view(base)
    assert "event_kernel" not in config_view(base)["hardware"]
    assert config_view(fused)["hardware"]["event_kernel"] == "fused"
    assert memory_key(named, "pdg", protocol) == memory_key(base, "pdg", protocol)
    assert memory_key(fused, "pdg", protocol) != memory_key(base, "pdg", protocol)


def test_segments_and_a_soft_warm_up_are_inside_the_scope() -> None:
    """``train.tbptt_windows > 1`` (with the compile mode it needs), accumulation without CUDA
    graphs and a soft warm-up load with the key."""
    default = [*CARD[:1], "hardware.compile_mode=default", *CARD[2:], KEY]
    load_config(RECIPE, [*default, "train.tbptt_windows=2", "train.grad_accum=2"])
    load_config(RECIPE, [*default, "train.grad_accum=2"])
    load_config(RECIPE, [*CARD, KEY, "train.soft_until_step=100"])
    load_config(RECIPE, [*CARD, KEY, "train.precision=fp32"])
    load_config(RECIPE, [*CARD, KEY, "model.address_gate_input=u"])


#: What the block does not cover, as overrides of the field recipe on the card, with a word
#: of the reason the configuration gives.
REFUSED: tuple[tuple[tuple[str, ...], str], ...] = (
    # what phase 1 refused
    (("train.epsilon_start=0.1", "train.epsilon_end=0.1"), "exploration"),
    (("train.epsilon_end=0.05", "train.epsilon_steps=10"), "exploration"),
    (("train.p_dense=0.1",), "dense-fallback"),
    (("train.state_grad_clip=1.0",), "per-row gradient limit"),
    (("train.state_grad_ratio=4.0",), "per-row gradient limit"),
    # the device and the window
    (("hardware.compile_mode=none", "hardware.round_products=separate"), "compile_mode=none"),
    (("hardware.master_dtype=fp64", "train.precision=fp64"), "fp32"),
    (("train.precision=fp16",), "fp32 or bf16"),
    # the round
    (("model.write_events=byte", "model.word_address=none"), "no event loop"),
    (("train.ste_scope=all",), "masked round"),
    (("train.unit_credit=softmax",), "executed"),
    (("train.feedback_detach=[summary,eps,state,router]",), "feedback_detach"),
    (("train.feedback_detach=[summary,router]",), "feedback_detach"),
    (("model.cand_norm=none",), "rms"),
    (("model.cell_norm=none",), "rms"),
    (("model.cell_attention_heads=4",), "attention read"),
    (
        ("model.scorer=mlp", "model.address_gate=false", "model.address_gate_input=u"),
        "linear scorer",
    ),
    (("model.scorer_features.exclude=[eps]",), "linear scorer"),
    (("model.cell_units=fixed",), "routed units"),
    # the kernels' shapes
    (("model.n_units=6",), "power of two"),
    (("model.n_cells=128",), "thread per logit"),
    (("model.k_select=20",), "cells a slot"),
    (("model.in_degree=6", "model.random_edges=4"), "in-degree"),
)


@pytest.mark.parametrize(("overrides", "reason"), REFUSED, ids=lambda v: None)
def test_refused_outside_its_scope(overrides: tuple[str, ...], reason: str) -> None:
    """Each refusal names what the block does not cover, at configuration load; without the
    key the same configuration loads (it trains on the compiled window)."""
    without = load_config(RECIPE, [*CARD, *overrides])
    message = event_kernel_scope_error(without)
    assert message is not None and reason in message, message
    with pytest.raises(ValueError, match="hardware.event_kernel=fused is not available") as error:
        load_config(RECIPE, [*CARD, *overrides, KEY])
    assert reason in str(error.value)


def test_refused_on_the_cpu_and_without_an_event_loop() -> None:
    """The kernels are CUDA only: the CPU profile (and MPS) refuse the key. So do a model
    without the memory, the cost term and a gate that is never the straight-through one."""
    with pytest.raises(ValueError, match="CUDA only"):
        load_config(RECIPE, [KEY, "hardware.compile_mode=default"])
    with pytest.raises(ValueError, match="CUDA only"):
        load_config(RECIPE, ["profile=m_series", KEY, "hardware.compile_mode=default"])
    with pytest.raises(ValueError, match="event_kernel"):
        load_config(EXPERIMENTS / "hsn_t_nomem.yaml", ["profile=rtx3060", KEY])
    base = load_config(RECIPE, list(CARD))

    def reason(section: str, **update: Any) -> str:
        part = getattr(base, section).model_copy(update=update)
        return event_kernel_scope_error(base.model_copy(update={section: part})) or ""

    # what the event loop or the compiled window refuses itself at load, whatever the key: the
    # block's own check names each too (a configuration built by hand)
    assert "cost term" in reason("train", aux_enabled=["cost"])
    assert "cost term" in reason("train", lambda_c=0.1)
    assert "ste" in reason("train", gate_mode="soft")
    assert "ste" in reason("train", gate_mode="reinforce")
    assert "temperature ramp" in reason("train", temperature_start=2.0)
    assert "lambda_u" in reason("train", lambda_u=0.1)
    assert "selected_readout" in reason("model", selected_readout=False)
    assert "address_gate_input" in reason("model", address_gate_input="reader_prev")
    assert "write decision" in reason("model", write_gate=True)
    assert "fast cells" in reason("model", fast_cells=8)
    assert "one round" in reason("model", r_max=2)
    assert "ablate_shared_u" in reason("model", ablate_shared_u=True)
    assert "hidden" in reason("model", hidden=2048)
    assert "autograd window" in reason("hardware", window="split")


def test_the_built_model_is_checked_again() -> None:
    """:func:`hsn.train.event_fused.training_scope_error` asks the built model what the
    configuration was asked, at the hard-gate steps whatever the model's mode: the recipe is
    in scope (in a soft warm-up and in evaluation mode too), a model outside it is refused
    with the same reasons, and so is :class:`FusedEvents`."""
    model = build_model(_small())
    assert isinstance(model, HSN)
    assert ef.training_scope_error(model) is None
    assert ef.training_scope_error(model.eval()) is None
    warm = build_model(_small("train.soft_until_step=5"))
    assert isinstance(warm, HSN)
    assert ef.training_scope_error(warm) is None
    assert ef.scope_error(warm.train()) is not None  # this step's gate is soft
    for overrides, reason in (
        (("train.p_dense=0.5",), "p_dense"),
        (("train.epsilon_start=0.2", "train.epsilon_end=0.2"), "exploration"),
        (("train.ste_scope=all",), "masked round"),
        (("train.unit_credit=softmax",), "unit_credit"),
        (("train.state_grad_clip=1.0",), "per-row limit"),
        (("train.gate_mode=soft",), "ste"),
    ):
        other = build_model(_small(*overrides))
        assert isinstance(other, HSN)
        message = ef.training_scope_error(other)
        assert message is not None and reason in message, message
        with pytest.raises(ValueError, match="does not take this model"):
            ef.FusedEvents(other, kernels=False)
    with pytest.raises(ValueError, match="CUDA only"):
        ef.FusedEvents(model)  # the kernels: the model is on the CPU
    with pytest.raises(ValueError, match="segment_windows"):
        ef.FusedEvents(model, kernels=False, segment_windows=0)


def test_no_fused_object_without_the_key() -> None:
    model = build_model(_small())
    assert isinstance(model, HSN)
    assert fused_events_of(model, "compiled") is None
    with pytest.raises(ValueError, match="compiled or fused"):
        fused_events_of(model, "autograd")
    with pytest.raises(ValueError, match="needs a compiled window"):
        compile_window(
            model, resolve(_small().hardware), "none",
            fused_events=ef.FusedEvents(model, kernels=False),
        )  # fmt: skip


# --------------------------------------------------------------------------------------------------
# The trainer on the CPU: the block's PyTorch reference in float64
# --------------------------------------------------------------------------------------------------

#: The word-event recipe's memory keys on the tiny model of the trainer tests: the segment
#: tests' HSN-T (``test_tbptt_windows.HSNT``: top-K with the straight-through gate, executed
#: credit, cut C, A9, A11, A12 on the summary) with both norms, word events with the exact
#: address, two of the four cells a slot and refresh probes after two slots (so that most
#: slots hold a probe), lazy AdamW and the balance term.
EVENTS = (
    "model.cell_norm=rms",
    "model.cand_norm=rms",
    "model.write_events=word",
    "model.word_address=exact",
    "model.word_contexts=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]",  # every context is text
    "model.k_select=2",
    "model.age_max=2",
    "train.optimizer=lazy_adamw",
    "train.aux_enabled=[balance]",
    "train.lambda_b=0.02",
    "train.stats_every=2",
    STREAM,  # documents run across windows: the carried state matters, most windows hold a loss
)
FP64 = ("hardware.master_dtype=fp64", "train.precision=fp64")


def _small(*extra: str, steps: int = 4) -> RunConfig:
    return _cfg(steps, *EVENTS, *extra)


def _perturb(model: torch.nn.Module, scale: float) -> None:
    """Away from the symmetric start (distinct scores: decisions with a margin, and every
    path carries a gradient)."""
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_((scale * torch.randn(p.shape, generator=generator)).to(p))


def _watch(trainer: TBPTTTrainer) -> list[dict[str, Any]]:
    """Every training window of the trainer (not a statistics step's term passes, not the
    evaluation's windows): its records, copied."""
    seen: list[dict[str, Any]] = []
    original, norms = trainer._forward, trainer._compiled_term_norms
    term_pass = [False]

    def terms(*args: Any, **kwargs: Any) -> Any:
        term_pass[0] = True
        try:
            return norms(*args, **kwargs)
        finally:
            term_pass[0] = False

    def spy(batch: dict[str, Tensor], state: Any, **kwargs: Any) -> Any:
        logits, new_state, info = original(batch, state, **kwargs)
        fused = trainer._fused_events
        if trainer.model.training and not term_pass[0]:
            seen.append(
                {
                    key: info[key].detach().clone()
                    for key in ("cell_idx", "unit_idx", "valid", "event_valid", "unit_logits")
                }
                | {
                    "start_grad": bool(state.h.requires_grad),
                    "position": None if fused is None else fused.position,
                    "slots": None if fused is None else fused.slots,
                }
            )
        return logits, new_state, info

    trainer._forward = spy  # type: ignore[method-assign]
    trainer._compiled_term_norms = terms  # type: ignore[method-assign]
    return seen


def _fit(
    cfg: RunConfig,
    run_dir: Path,
    steps: int,
    *,
    scale: float = 0.3,
    seed: int = 0,
) -> dict[str, Any]:
    seed_everything(seed)
    trainer = TBPTTTrainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=run_dir)
    _perturb(trainer.model, scale)
    windows = _watch(trainer)
    grads, params = _capture_grads(trainer), _capture_params(trainer)
    history = trainer.fit(*_long_datasets(), steps)
    return {
        "trainer": trainer,
        "history": history,
        "grads": grads,
        "params": params,
        "windows": windows,
    }


@pytest.fixture
def reference_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """The trainer of a CPU configuration with the fused block on its PyTorch reference: the
    configuration refuses the key on the CPU, so the trainer is handed the block as
    ``hardware.event_kernel=fused`` would build it, with ``kernels=False``. Its compiled
    window is ``aot_eager``'s (as the other compiled-trainer tests on the CPU), in float64
    too (the compiled scope leaves fp64 to the eager gradchecks by rule, not by need)."""

    def reference(model: HSN, kind: str, *, segment_windows: int = 1) -> ef.FusedEvents:
        return ef.FusedEvents(model, kernels=False, segment_windows=segment_windows)

    check = compile_mod.check_compile_scope

    def any_dtype(model: HSN, **kwargs: Any) -> None:
        try:
            check(model, **kwargs)
        except NotImplementedError as error:
            if "fp64" not in str(error):
                raise

    monkeypatch.setattr(tbptt, "fused_events_of", reference)
    monkeypatch.setattr(compile_mod, "check_compile_scope", any_dtype)
    monkeypatch.setattr(
        tbptt, "compile_window", functools.partial(compile_window, backend="aot_eager")
    )


def _near(a: float, b: float, tol: float) -> bool:
    """Two floats within ``tol`` of the larger of 1 and their size; two NaNs (the loss of a
    step without a loss position) are the same."""
    if a != a or b != b:
        return a != a and b != b
    return abs(a - b) <= tol * max(1.0, abs(b))


def _assert_close(got: Tensor | None, want: Tensor | None, tol: float, what: str) -> None:
    assert (got is None) == (want is None), what
    if want is None or got is None:
        return
    scale = max(float(want.abs().max()), 1e-30)
    torch.testing.assert_close(got, want, rtol=0.0, atol=tol * scale, msg=what)


def _assert_same_run(
    got: dict[str, Any],
    ref: dict[str, Any],
    *,
    loss: float,
    grad: float,
    param: float,
    stats: float,
    decisions: bool = True,
) -> None:
    """Two trainer runs agree: every step's loss and raw gradient, the parameters after every
    step, the statistics and the term gradient norms, the evaluation; with ``decisions`` the
    selections of every training window on the slots that hold an event."""
    a, b = got["history"], ref["history"]
    assert len(a.steps) == len(b.steps) > 0
    for x, y in zip(a.steps, b.steps, strict=True):
        assert _near(x.loss, y.loss, loss), (x.step, x.loss, y.loss)
        assert _near(x.total, y.total, loss), (x.step, x.total, y.total)
        assert x.loss_positions == y.loss_positions
        assert list(x.parts) == list(y.parts) and "balance" in x.parts
        for name, value in y.parts.items():
            assert _near(x.parts[name], value, loss), (x.step, name)
        assert x.active_units == y.active_units
        assert set(x.stats) == set(y.stats), set(x.stats) ^ set(y.stats)
        for key, value in y.stats.items():
            # (a term's gradient norm is reduced in fp32 whatever the model's dtype, and the
            # eager report and the compiled term passes sum it in another order)
            tol = max(stats, 1e-5) if key.startswith(TERM_GRAD_NORM_PREFIX) else stats
            assert _near(x.stats[key], value, tol), (x.step, key, x.stats[key], value)
    assert len(got["grads"]) == len(ref["grads"]) == len(b.steps)
    for step, (gs, rs) in enumerate(zip(got["grads"], ref["grads"], strict=True)):
        for index, (g, r) in enumerate(zip(gs, rs, strict=True)):
            _assert_close(g, r, grad, f"gradient {index} of step {step + 1}")
    assert len(got["params"]) == len(ref["params"])
    for step, (ps, rs) in enumerate(zip(got["params"], ref["params"], strict=True)):
        for index, (p, r) in enumerate(zip(ps, rs, strict=True)):
            _assert_close(p, r, param, f"parameter {index} after step {step + 1}")
    assert len(a.evals) == len(b.evals) > 0
    for x, y in zip(a.evals, b.evals, strict=True):
        assert _near(x["loss"], y["loss"], loss), (x["loss"], y["loss"])
    assert sum(step.loss_positions for step in b.steps) > 0  # the run has a loss
    if decisions:
        assert len(got["windows"]) == len(ref["windows"])
        for index, (w, v) in enumerate(zip(got["windows"], ref["windows"], strict=True)):
            assert torch.equal(w["event_valid"], v["event_valid"])
            held = v["event_valid"].any(dim=0)
            for key in ("cell_idx", "unit_idx", "valid"):
                assert torch.equal(w[key][:, held], v[key][:, held]), (index, key)


def test_trainer_with_the_block_equals_the_eager_trainer(
    tmp_path: Path, reference_block: None
) -> None:
    """Four optimiser steps in float64 (lazy AdamW, the balance term, a statistics step every
    second step, an evaluation at the end): the trainer whose training windows run the fused
    block (its PyTorch reference, ``trim`` on) equals the eager trainer. Every training
    window ran the block, on fewer slots than its layout in some; the term passes and the
    evaluation ran the compiled window."""
    steps = 4
    ref = _fit(_small(*FP64, "hardware.compile_mode=none"), tmp_path / "eager", steps)
    got = _fit(_small(*FP64, "hardware.compile_mode=default"), tmp_path / "fused", steps)
    trainer = got["trainer"]
    fused, compiled = trainer._fused_events, trainer.compiled_window
    assert isinstance(fused, ef.FusedEvents) and isinstance(compiled, CompiledWindow)
    assert compiled.fused_events is fused and not fused.kernels
    assert ref["trainer"]._fused_events is None
    assert fused.calls == steps == len(got["windows"])
    measured = got["history"].measured
    assert measured["fused_event_windows"] == steps and measured["fused_event_captures"] == 0
    assert "fused_event_windows" not in ref["history"].measured
    # the term passes of the two statistics steps and the evaluation's windows ran compiled
    assert compiled.calls > steps + 2
    slots = [w["slots"] for w in got["windows"]]
    assert any(ran < layout for ran, layout in slots), slots  # trim left slots out
    assert all(w["position"] == 0 for w in got["windows"])
    _assert_same_run(got, ref, loss=1e-9, grad=1e-8, param=1e-9, stats=1e-8)
    # the statistics that read the selection records (trim zeroes those of the slots behind
    # the last event): the refresh rate is there, not zero, and the same
    last = got["history"].steps[-1]
    assert last.stats["refreshed"] > 0.0 and "term_grad_norm/balance" in last.stats
    assert last.stats["term_grad_norm/task"] > 0.0


def test_refresh_rate_is_over_the_slots_that_hold_an_event(
    tmp_path: Path, reference_block: None
) -> None:
    """The logged refresh rate of an event window is the mean over the slots in which some
    row has an event, so it is the same number with the block (which does not run the slots
    behind them) and without (the compiled blocks run padding rows there, with their probes):
    equal in the two runs, and equal to that mean of the eager window's records."""
    steps = 6  # some rows end a window with a cell past the probes' threshold
    cfg = _small(*FP64, "hardware.compile_mode=none", "train.stats_every=1", steps=steps)
    seed_everything(0)
    trainer = TBPTTTrainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "a")
    _perturb(trainer.model, 0.3)
    records: list[dict[str, Tensor]] = []
    original = trainer._forward

    def spy(batch: dict[str, Tensor], state: Any, **kwargs: Any) -> Any:
        out = original(batch, state, **kwargs)
        if trainer.model.training:  # a training window (the eager trainer has no term pass)
            records.append({k: out[2][k].detach().clone() for k in ("refreshed", "event_valid")})
        return out

    trainer._forward = spy  # type: ignore[method-assign]
    eager = [step.stats["refreshed"] for step in trainer.fit(*_long_datasets(), steps).steps]
    assert len(records) == steps
    behind = 0  # windows whose slots without an event hold probes of padding rows
    for logged, record in zip(eager, records, strict=True):
        held = record["event_valid"].any(dim=0)
        rate = record["refreshed"].double()
        assert 0 < int(held.sum()) < held.numel()  # slots without an event behind the last
        assert logged == pytest.approx(float(rate[held].mean()), rel=1e-6, abs=1e-12)
        if float(rate[~held].sum()) > 0.0:  # not counted: the mean over every slot is another
            behind += 1
            assert logged != pytest.approx(float(rate.mean()), rel=1e-3)
    assert behind > 0 and max(eager) > 0.0
    got = _fit(
        _small(*FP64, "hardware.compile_mode=default", "train.stats_every=1", steps=steps),
        tmp_path / "b",
        steps,
    )
    with_block = [step.stats["refreshed"] for step in got["history"].steps]
    assert with_block == pytest.approx(eager, rel=1e-6, abs=1e-12)


def test_soft_warm_up_steps_stay_on_the_compiled_window(
    tmp_path: Path, reference_block: None
) -> None:
    """With ``train.soft_until_step`` the warm-up's steps run the compiled window (every cell,
    with autograd) and the block takes over at the first hard-gate step; the run equals the
    eager trainer's."""
    steps, warm = 4, 2
    soft = f"train.soft_until_step={warm}"
    ref = _fit(_small(*FP64, soft, "hardware.compile_mode=none"), tmp_path / "eager", steps)
    got = _fit(_small(*FP64, soft, "hardware.compile_mode=default"), tmp_path / "fused", steps)
    fused = got["trainer"]._fused_events
    assert fused is not None and fused.calls == steps - warm
    _assert_same_run(got, ref, loss=1e-9, grad=1e-8, param=1e-9, stats=1e-8)


def test_default_path_has_no_fused_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the key the trainer builds nothing of the block: no fused object in the
    trainer or in its compiled window, the windows are the compiled blocks', and the
    ``measured`` keys are those from before the key."""
    monkeypatch.setattr(
        tbptt, "compile_window", functools.partial(compile_window, backend="aot_eager")
    )
    got = _fit(_small("hardware.compile_mode=default", steps=2), tmp_path, 2)
    trainer = got["trainer"]
    assert trainer._fused_events is None
    compiled = trainer.compiled_window
    assert isinstance(compiled, CompiledWindow) and compiled.fused_events is None
    assert not any(key.startswith("fused_event") for key in got["history"].measured)
    assert compiled.calls >= 2 and all(w["slots"] is None for w in got["windows"])


def test_trainer_refuses_the_key_without_a_compiled_window(tmp_path: Path) -> None:
    """A hand-built configuration (the loaded one is refused): the key without a compiled
    window raises at the trainer, and so does the CPU with one (the kernels)."""
    cfg = _small("hardware.compile_mode=none")
    hardware = cfg.hardware.model_copy(update={"event_kernel": "fused"})
    broken = cfg.model_copy(update={"hardware": hardware})
    with pytest.raises(ValueError, match="needs a compiled window"):
        TBPTTTrainer(build_model(cfg), broken, resolve(cfg.hardware), run_dir=tmp_path)
    cfg = _small("hardware.compile_mode=default")
    hardware = cfg.hardware.model_copy(update={"event_kernel": "fused"})
    broken = cfg.model_copy(update={"hardware": hardware})
    with pytest.raises(ValueError, match="CUDA only"):
        TBPTTTrainer(build_model(cfg), broken, resolve(cfg.hardware), run_dir=tmp_path)


# --------------------------------------------------------------------------------------------------
# train.tbptt_windows > 1
# --------------------------------------------------------------------------------------------------

SEGMENTS = ("train.grad_accum=4", "train.tbptt_windows=2")


def test_segments_of_two_windows_equal_the_eager_trainer(
    tmp_path: Path, reference_block: None
) -> None:
    """``train.tbptt_windows = 2`` with two segments a step, three steps, documents that run
    across windows: the trainer with the block equals the eager trainer (whose gradients hold
    a second window's credit to the first: ``tests/test_tbptt_windows.py``). A segment's
    windows take their places 0 and 1 (a set of buffers each on the card) and the second one
    starts from the first one's state with its gradient."""
    steps = 3
    ref = _fit(_small(*FP64, *SEGMENTS, "hardware.compile_mode=none"), tmp_path / "eager", steps)
    got = _fit(_small(*FP64, *SEGMENTS, "hardware.compile_mode=default"), tmp_path / "fused", steps)
    fused = got["trainer"]._fused_events
    assert fused is not None and fused.segment_windows == 2 and fused.calls == 4 * steps
    assert [w["position"] for w in got["windows"]] == [0, 1, 0, 1] * steps
    assert [w["start_grad"] for w in got["windows"]] == [False, True, False, True] * steps
    _assert_same_run(got, ref, loss=1e-9, grad=1e-8, param=1e-9, stats=1e-8)


def test_gradient_reaches_the_window_before_through_the_block(
    tmp_path: Path, reference_block: None
) -> None:
    """The start-state gradient flows into the window before: the same windows with every
    window detached (``train.tbptt_windows = 1``) give another gradient, on the block as on
    the eager trainer, and the two differences are the same."""
    one = ("train.grad_accum=4", "train.tbptt_windows=1")
    runs = {
        (mode, name): _fit(
            _small(*FP64, *keys, f"hardware.compile_mode={mode}", steps=1),
            tmp_path / f"{mode}-{name}",
            1,
        )
        for mode in ("none", "default")
        for name, keys in (("joined", SEGMENTS), ("cut", one))
    }
    credit: dict[str, list[Tensor]] = {}
    scales: list[float] = []
    for mode in ("none", "default"):
        joined, cut = runs[(mode, "joined")]["grads"][0], runs[(mode, "cut")]["grads"][0]
        pairs = [(a, b) for a, b in zip(joined, cut, strict=True) if a is not None]
        credit[mode] = [a - b for a, b in pairs]
        scales = [float(a.abs().max()) for a, _ in pairs]
        # the credit is a visible share of some parameter's gradient
        pairs_scaled = zip(credit[mode], scales, strict=True)
        shares = [float(c.abs().max()) / max(s, 1e-30) for c, s in pairs_scaled]
        assert max(shares) > 1e-3, max(shares)
    both = zip(credit["default"], credit["none"], scales, strict=True)
    for index, (a, b, scale) in enumerate(both):
        # (each gradient equals the eager one to 1e-8 of its largest entry, so their
        # difference does to twice that)
        worst = float((a - b).abs().max())
        assert worst <= 4e-8 * max(scale, 1e-30), (index, worst, scale, float(b.abs().max()))


def test_a_window_outside_its_segment_is_refused() -> None:
    """A place in the segment beyond ``train.tbptt_windows`` has no buffers: an error, not a
    tape overwritten under a live backward pass."""
    model = build_model(_small())
    assert isinstance(model, HSN)
    fused = ef.FusedEvents(model, kernels=False, segment_windows=2)
    fused.kernels = True  # the place is checked where the buffers are picked
    fused.position = 2
    with pytest.raises(ValueError, match="window 2 of a gradient segment of 2"):
        fused._set()


# --------------------------------------------------------------------------------------------------
# On the card
# --------------------------------------------------------------------------------------------------

needs_card = pytest.mark.skipif(
    not (torch.cuda.is_available() and window_kernels.available()),
    reason="needs a CUDA device and the window kernels (scripts/build_window_kernels.py)",
)

#: The card shape of the block's tests (a power-of-two number of units, a thread per logit) on
#: the trainer tests' model, fp32 products, the compiled window without CUDA graphs
#: (``aot_eager``: the comparison is the trainer's wiring, not inductor's rounding).
ON_CARD = (
    "profile=rtx3060",
    "hardware.compile_mode=default",
    "train.deterministic=false",
    "model.d=16",
    "model.context_dim=16",
    "model.n_cells=16",
    "model.in_degree=3",
    "model.n_units=4",
    "model.k_select=5",
    "model.hidden=32",
    "model.reader_width=16",
    "model.age_max=4",
)
FP32 = ("hardware.autocast_dtype=fp32", "train.precision=fp32")
#: The least gap between the scores that a decision of the card cases separates: a hundred
#: times the rounding of a score in fp32 (the compiled window's tests take the same).
MIN_MARGIN = 1e-4


def _least_margin(run: dict[str, Any], k: int) -> float:
    """The least margin of a run's decisions on its live rows, over its training windows:
    between the last selected and the first unselected cell score, and between a cell's best
    two units."""
    least = float("inf")
    for window in run["windows"]:
        logits = window["unit_logits"].float()  # [B, E, N, M]
        best = logits.sort(dim=-1, descending=True).values
        cells = best[..., 0].sort(dim=-1, descending=True).values
        gaps = [cells[..., k - 1] - cells[..., k], (best[..., 0] - best[..., 1]).amin(dim=-1)]
        live = window["event_valid"]
        if bool(live.any()):
            least = min(least, float(torch.stack(gaps).amin(dim=0)[live].min()))
    return least


@pytest.fixture
def eager_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tbptt, "compile_window", functools.partial(compile_window, backend="aot_eager")
    )


def _card_pair(tmp_path: Path, steps: int, *keys: str, lr: str = "train.lr=1e-5") -> Any:
    """The trainer without and with the key on the card, from a start whose every decision of
    the run has a margin (searched over the perturbation's scale; checked on both runs). A
    small learning rate: Adam turns a gradient entry at the rounding level into a full step
    in either direction, so the two runs' parameters part by the step size in such entries."""
    for scale in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
        base = _small(*ON_CARD, *keys, lr, steps=steps)
        ref = _fit(base, tmp_path / f"off-{scale}", steps, scale=scale)
        if _least_margin(ref, base.model.k_select) <= MIN_MARGIN:
            continue
        got = _fit(_small(*ON_CARD, *keys, lr, KEY, steps=steps), tmp_path / f"on-{scale}", steps,
                   scale=scale)  # fmt: skip
        assert _least_margin(got, base.model.k_select) > 0.5 * MIN_MARGIN
        return ref, got
    raise AssertionError("no start with a margin at every decision")


@needs_card
@pytest.mark.cuda
def test_trainer_with_the_key_equals_without_fp32_cuda(tmp_path: Path, eager_kernels: None) -> None:
    """Three optimiser steps on the card in fp32: the trainer with ``hardware.event_kernel=
    fused`` (the kernels, ``trim``, the parameters' backward on the card, their CUDA graphs)
    against the trainer without it. The same decisions at every window; the losses, the raw
    gradients, the updated parameters and the statistics to the compiled window's CUDA
    tolerance. Every training window ran the block."""
    steps = 3
    ref, got = _card_pair(tmp_path, steps, *FP32)
    fused = got["trainer"]._fused_events
    assert isinstance(fused, ef.FusedEvents) and fused.kernels and fused.calls == steps
    assert ref["trainer"]._fused_events is None
    described = fused.describe()
    assert described["buffer_sets"] == 1 and described["captures"] >= 1
    measured = got["history"].measured
    assert measured["fused_event_windows"] == steps
    assert measured["compile_steps"] >= described["captures"]  # a capture is a compile step
    _assert_same_run(got, ref, loss=1e-5, grad=2e-4, param=1e-3, stats=2e-4)


@needs_card
@pytest.mark.cuda
def test_segments_on_two_sets_of_buffers_fp32_cuda(tmp_path: Path, eager_kernels: None) -> None:
    """``train.tbptt_windows = 2`` on the card: a segment's two windows run on two sets of
    buffers (the first window's tape outlives the second window's forward sweep), the
    segment's one backward pass crosses both blocks, and the run equals the trainer without
    the key, whose gradients hold the second window's credit to the first."""
    steps = 2
    ref, got = _card_pair(tmp_path, steps, *FP32, *SEGMENTS)
    fused = got["trainer"]._fused_events
    assert fused is not None and fused.calls == 4 * steps
    described = fused.describe()
    assert described["buffer_sets"] == 2 and all(described["slot_counts"])
    # one memory pool for the graphs of both sets (their replays take turns on one stream)
    assert fused._sets[0].pool is fused._sets[1].pool is fused._pool is not None
    assert [w["position"] for w in got["windows"]] == [0, 1, 0, 1] * steps
    assert [w["start_grad"] for w in got["windows"]] == [False, True, False, True] * steps
    _assert_same_run(got, ref, loss=1e-5, grad=2e-4, param=1e-3, stats=2e-4)


@needs_card
@pytest.mark.cuda
def test_trainer_with_the_key_under_bf16_cuda(tmp_path: Path, eager_kernels: None) -> None:
    """bf16 autocast, as the runs: the trainer with the key trains (finite losses and
    gradients, every window fused) and stays close to the trainer without it over three
    steps (the two round differently, so near-tied decisions may part)."""
    steps = 3
    keys = (*ON_CARD, "train.lr=1e-4")
    ref = _fit(_small(*keys, steps=steps), tmp_path / "off", steps, scale=0.2)
    got = _fit(_small(*keys, KEY, steps=steps), tmp_path / "on", steps, scale=0.2)
    fused = got["trainer"]._fused_events
    assert fused is not None and fused.calls == steps
    for x, y in zip(got["history"].steps, ref["history"].steps, strict=True):
        assert torch.isfinite(torch.tensor([x.total, x.grad_norm])).all()
        assert x.loss_positions == y.loss_positions
        assert _near(x.loss, y.loss, 0.05), (x.loss, y.loss)  # (NaN without a loss position)
        assert _near(x.total, y.total, 0.05), (x.total, y.total)
    assert sum(step.loss_positions for step in got["history"].steps) > 0
    for gs in got["grads"]:
        assert all(bool(torch.isfinite(g).all()) for g in gs if g is not None)
    assert len(_params(got["trainer"].model)) > 0
