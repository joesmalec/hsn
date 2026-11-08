"""P1.15 tests: the gradient-spike guard (``train.spike_factor``, ``train.spike_window``).

Off by default (today's training: a finite spike is clipped and applied). With a factor, a step
whose finite pre-clip norm exceeds the factor times the running median of the applied norms is
skipped as P1.12 skips a non-finite one (weights, optimiser and AdamW clock untouched, gradients
zeroed, the scheduler advancing), counted and logged; spikes count in P1.12's stop rule; the guard's
state survives a checkpoint; the diagnostics count net of it; the sweep record marks the skipped
steps. On exp 171's kept record the recommended factor skips step 1,269 and none before it. The
compiled window (CPU here, CUDA Graphs under the cuda marker) skips the same steps as eager.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import load_config
from hsn.experiments.diagnostics import Diagnostics
from hsn.experiments.protocols import ArmRun, failure_info, run_protocol
from hsn.experiments.rules import parse_rules
from hsn.experiments.spec import protocol_params
from hsn.train import compile as compile_mod
from hsn.train import loop, tbptt
from hsn.train.compile import compile_window
from hsn.train.loop import SPIKE_MIN_HISTORY, SPIKE_WINDOW, spike_decision
from p1_11_support import (
    REF_A1,
    TINY_HSN_OVERRIDES,
    disk_overrides,
    restore_determinism,  # noqa: F401  (fixture)
    tiny_manifest,  # noqa: F401  (fixture)
)
from p1_15_support import (
    FIXTURES,
    GUARD,
    INF_GRADIENT,
    SPIKE,
    StepProbe,
    datasets,
    gru_config,
    gru_trainer,
    hsn_config,
    hsn_trainer,
    plant,
    plant_by_call,
    plant_gru_spike,
)

pytestmark = pytest.mark.usefixtures("restore_determinism")
CPU_BACKEND = "inductor" if sys.platform.startswith("linux") else "aot_eager"


@pytest.fixture(autouse=True)
def _release_compiled_state() -> Any:
    yield
    torch._dynamo.reset()


# --------------------------------------------------------------------------------------------------
# The rule, and exp 171
# --------------------------------------------------------------------------------------------------


def test_p1_15_spike_decision_rule() -> None:
    history = [1.0] * 19
    assert spike_decision(1e9, history, 100) == (False, None)  # below the minimum history
    history.append(1.0)
    assert spike_decision(100.0, history, 100) == (False, 1.0)  # not above factor x median
    assert spike_decision(100.0001, history, 100) == (True, 1.0)
    # only the last `window` applied norms count, and a small window decides after `window` norms
    assert spike_decision(30.0, [100.0] * 50 + [0.1] * 4, 100, window=4) == (True, 0.1)
    assert spike_decision(30.0, [0.1] * 3, 100, window=4) == (False, None)
    assert SPIKE_WINDOW == 100 and SPIKE_MIN_HISTORY == 20
    assert load_config(REF_A1).train.spike_window == SPIKE_WINDOW  # the config's default
    with pytest.raises(ValueError, match="window"):
        spike_decision(1.0, [1.0], 100, window=0)


def _would_skip(norms: list[Any], factor: float) -> list[int]:
    """The steps the guard skips over a stored record's pre-clip norms (step 1 first): non-finite
    norms are P1.12's skips and never enter the history; a skipped spike does not either."""
    history: list[float] = []
    skipped = []
    for step, g in enumerate(norms, start=1):
        value = float(g)
        if not math.isfinite(value):
            continue
        skip, _median = spike_decision(value, history, factor)
        if skip:
            skipped.append(step)
        else:
            history.append(value)
    return skipped


def test_p1_15_spike_guard_on_exp171_record() -> None:
    """Exp 171's kept per-step record (8 Nov, the lr 2e-3 seed-1 divergence): at the recommended
    factor 100 the guard skips step 1,269 (a finite 2.2e10) and none of the 1,268 steps before it;
    every factor from 5 to 6e10 does the same (the earlier steps peak at 4.5 times their median, the
    jump is 7.0e10 times); a factor of 4 would also skip an early step."""
    data = json.loads((FIXTURES / "exp171_grad_norms.json").read_text())
    norms = data["grad_norms"]
    assert data["first_step"] == 1 and len(norms) == 1278
    assert float(norms[1268]) == pytest.approx(2.2147e10, rel=1e-4)
    head = norms[:1269]  # steps 1 to 1,269; after it every norm is non-finite
    for factor in (5.0, 100.0, 1e3, 1e6, 6e10):
        assert _would_skip(head, factor) == [1269], factor
    early = _would_skip(head, 4.0)
    assert early and early[0] < 1269


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_p1_15_spike_guard_config() -> None:
    cfg = hsn_config(4)
    assert cfg.train.spike_factor is None and cfg.train.spike_window == 100
    assert hsn_config(4, *GUARD).train.spike_factor == 100.0
    with pytest.raises(ValueError, match="spike_window"):
        hsn_config(4, "train.spike_window=4")  # read by the guard only
    with pytest.raises(ValueError, match="spike_factor"):
        hsn_config(4, "train.spike_factor=1.0")  # must be > 1


def test_p1_15_spike_guard_refused_with_grad_scaler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        loop, "make_grad_scaler", lambda ctx: torch.amp.GradScaler("cpu", init_scale=2.0**10)
    )
    with pytest.raises(NotImplementedError, match="GradScaler"):
        hsn_trainer(hsn_config(4, *GUARD), tmp_path / "run")


# --------------------------------------------------------------------------------------------------
# Skips
# --------------------------------------------------------------------------------------------------


def test_p1_15_spike_applied_with_the_guard_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off by default: a planted finite spike is clipped and applied, as today."""
    train_ds, dev_ds = datasets()
    trainer = hsn_trainer(hsn_config(8), tmp_path / "run")
    plant(monkeypatch, trainer, {6: SPIKE})
    with StepProbe(trainer) as probe:
        history = trainer.fit(train_ds, dev_ds, 8)
    assert not probe.unchanged(6) and probe.optimizer_steps[8] == 8
    assert history.steps[5].grad_norm > 1e6  # the spike, applied
    assert trainer.skipped_spike == 0 and trainer.skipped_steps == []
    assert "skipped_spike_steps" not in history.measured
    assert all("skipped_spike" not in r.stats for r in history.steps)  # type: ignore[attr-defined]


def test_p1_15_spike_step_skipped_tbptt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A planted spike at step 6 (window 4, factor 100) is skipped: weights and the optimiser's
    clock untouched, gradients zeroed, the scheduler advancing (the learning rates equal an
    unplanted run's), counted in the records, measured and the log; the spike never enters the
    history; the other steps train and match the unplanted run up to the spike."""
    cfg = hsn_config(8, *GUARD)
    train_ds, dev_ds = datasets()
    clean = hsn_trainer(cfg, tmp_path / "clean")
    clean_history = clean.fit(train_ds, dev_ds, 8)
    assert clean.skipped_spike == 0 and clean_history.measured["skipped_spike_steps"] == 0.0
    trainer = hsn_trainer(cfg, tmp_path / "run")
    plant(monkeypatch, trainer, {6: SPIKE})
    with StepProbe(trainer) as probe, caplog.at_level(logging.WARNING, logger="hsn.train"):
        history = trainer.fit(train_ds, dev_ds, 8)
    assert probe.unchanged(6) and probe.grads_none[6]
    assert not probe.unchanged(7) and probe.optimizer_steps[8] == 7
    assert [r.lr for r in history.steps] == [r.lr for r in clean_history.steps]
    assert [r.loss for r in history.steps[:5]] == [r.loss for r in clean_history.steps[:5]]
    assert trainer.skipped_spike == 1 and trainer.skipped_nonfinite == 0
    assert history.measured["skipped_spike_steps"] == 1.0
    assert [r.stats["skipped_spike"] for r in history.steps] == [0.0] * 5 + [1.0] * 3  # type: ignore[attr-defined]
    (skip,) = trainer.skipped_steps
    assert skip["step"] == 6 and skip["reason"] == SPIKE and skip["grad_norm"] > 1e6
    assert history.steps[5].grad_norm == skip["grad_norm"]  # the record keeps the finite norm
    assert max(trainer._spike_history) < 1e3  # the spike never entered the history
    assert sum("times the median" in m and "step 6" in m for m in caplog.messages) == 1


def test_p1_15_spike_step_skipped_baseline(tmp_path: Path) -> None:
    """The generic trainer (the baselines) applies the same guard."""
    cfg = gru_config(8, *GUARD)
    train_ds, dev_ds = datasets()
    trainer = gru_trainer(cfg, tmp_path / "run")
    handles = plant_gru_spike(trainer, {6})
    with StepProbe(trainer) as probe:
        history = trainer.fit(train_ds, dev_ds, 8)
    for handle in handles:
        handle.remove()
    assert probe.unchanged(6) and probe.grads_none[6] and not probe.unchanged(7)
    assert trainer.skipped_spike == 1 and history.measured["skipped_spike_steps"] == 1.0
    assert [s["step"] for s in trainer.skipped_steps] == [6]


def test_p1_15_spike_and_nonfinite_share_the_stop_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision D4: spike and non-finite skips are one run of consecutive skips (K patched to 3:
    spike, inf, spike raises at the third) and the message names both; the harness classifies it."""
    monkeypatch.setattr(loop, "NONFINITE_MAX_CONSECUTIVE", 3)
    cfg = hsn_config(10, *GUARD)
    train_ds, dev_ds = datasets()
    trainer = hsn_trainer(cfg, tmp_path / "run")
    plant(monkeypatch, trainer, {6: SPIKE, 7: INF_GRADIENT, 8: SPIKE})
    with pytest.raises(RuntimeError, match=r"step 8: 3 consecutive optimizer steps skipped") as err:
        trainer.fit(train_ds, dev_ds, 10)
    assert "1 non-finite, 2 gradient spikes" in str(err.value)
    assert "P1.12 stop rule" in str(err.value)
    info = failure_info(err.value, stage="train", last_step=7, skipped_nonfinite=1, skipped_spike=2)
    assert info["kind"] == "spike" and info["skipped_spike"] == 2
    # without spikes the P1.12 message and kind are unchanged
    monkeypatch.setattr(loop, "NONFINITE_MAX_CONSECUTIVE", 2)
    other = hsn_trainer(cfg, tmp_path / "other")
    plant(monkeypatch, other, {6: INF_GRADIENT, 7: INF_GRADIENT})
    with pytest.raises(RuntimeError) as err2:
        other.fit(train_ds, dev_ds, 10)
    assert "for a non-finite gradient norm (limit 2)" in str(err2.value)
    assert failure_info(err2.value, stage="train", last_step=6)["kind"] == "nonfinite"


def test_p1_15_spike_guard_survives_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard's count and history go into the checkpoint: 4 steps, a resume and 4 more give
    the uninterrupted run's records, its skip included; a malformed history raises."""
    cfg = hsn_config(8, *GUARD, "train.ckpt_every=4")
    train_ds, dev_ds = datasets()
    whole = hsn_trainer(cfg, tmp_path / "whole")
    plant(monkeypatch, whole, {6: SPIKE})
    full = whole.fit(train_ds, dev_ds, 8)
    first = hsn_trainer(cfg, tmp_path / "first")
    plant(monkeypatch, first, {6: SPIKE})
    path = first.fit(train_ds, dev_ds, 4).checkpoints[-1]
    extra = torch.load(path, map_location="cpu", weights_only=True)["extra"]
    assert extra["skipped_spike"] == 0 and len(extra["spike_history"]) == 4
    second = hsn_trainer(cfg, tmp_path / "second")
    assert second.resume(path) == 4
    plant(monkeypatch, second, {6: SPIKE})
    rest = second.fit(train_ds, dev_ds, 8)
    assert [(r.loss, r.grad_norm) for r in rest.steps] == [
        (r.loss, r.grad_norm) for r in full.steps[4:]
    ]
    assert second.skipped_spike == 1 == whole.skipped_spike
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["extra"]["spike_history"] = [1.0] * 5  # longer than the window
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="spike_history"):
        hsn_trainer(cfg, tmp_path / "bad").resume(bad)


def test_p1_15_diagnostics_net_of_spike_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1.11's diagnostics count optimiser calls net of the spike guard's skips too."""
    cfg = hsn_config(8, *GUARD)
    train_ds, dev_ds = datasets()
    trainer = hsn_trainer(cfg, tmp_path / "run")
    diag = Diagnostics(trainer.model, cfg, trainer.ctx, dev_ds, 1)
    diag.attach(trainer)
    plant(monkeypatch, trainer, {6: SPIKE})
    trainer.fit(train_ds, dev_ds, 8)
    diag.detach()
    assert [r["step"] for r in diag.records] == [1, 2, 3, 4, 5, 7, 8]


def _arm(tmp_path: Path, manifest: Path, *overrides: str) -> ArmRun:
    cfg = load_config(REF_A1, [*TINY_HSN_OVERRIDES, *disk_overrides(manifest, steps=8), *overrides])
    params = protocol_params({"name": "reads_state", "diag_every": 2, "probe_batch": 8})
    return ArmRun(
        cfg=cfg,
        family="hsn",
        variant="pdg",
        variant_params={},
        params=params,
        run_dir=tmp_path,
        rules=parse_rules(None),
        constants={},
    )


def test_p1_15_spike_in_the_sweep_record(
    tmp_path: Path,
    tiny_manifest: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record: ``final.skipped_spike_steps``, ``skipped_steps`` and the skipped train entry
    marked; with the guard off ``skipped_spike_steps`` is None and nothing is marked."""
    plant_by_call(monkeypatch, {6: SPIKE})
    record = run_protocol(_arm(tmp_path / "on", tiny_manifest, *GUARD))
    assert record["final"]["skipped_spike_steps"] == 1.0
    assert [s["step"] for s in record["skipped_steps"]] == [6]
    marked = [t["step"] for t in record["train"] if t.get("skipped")]
    assert marked == [6] and record["train"][5]["skipped"] == SPIKE
    assert record["train"][5]["grad_norm"] > 1e6
    plant_by_call(monkeypatch, {})
    off = run_protocol(_arm(tmp_path / "off", tiny_manifest))
    assert off["final"]["skipped_spike_steps"] is None and off["skipped_steps"] == []
    assert not any(t.get("skipped") for t in off["train"])


# --------------------------------------------------------------------------------------------------
# P2.8's compiled window
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compile_mod, "MAX_BLOCK", 4)


class _Events:
    """Dynamo compile and CUDA-Graph recording events per step."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self.count = 0
        self.at: dict[int, int] = {}

    def __call__(self, _: Any) -> None:
        self.count += 1
        self.at[self.trainer.step + 1] = self.count


@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("backend", sorted({"aot_eager", CPU_BACKEND}))
def test_p1_15_spike_skip_compiled_cpu(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2.8's compiled window on the CPU skips the same planted spike as eager, leaves every
    weight unchanged across it, and compiles nothing after the first step."""
    train_ds, dev_ds = datasets()
    eager = hsn_trainer(hsn_config(8, *GUARD), tmp_path / "eager")
    plant(monkeypatch, eager, {6: SPIKE})
    eager.fit(train_ds, dev_ds, 8)
    monkeypatch.setattr(tbptt, "compile_window", functools.partial(compile_window, backend=backend))
    cfg = hsn_config(8, *GUARD, "hardware.compile_mode=reduce-overhead")
    trainer = hsn_trainer(cfg, tmp_path / "compiled")
    assert trainer.compiled_window is not None
    plant(monkeypatch, trainer, {6: SPIKE})
    events = _Events(trainer)
    torch._dynamo.callback_handler.register_start_callback(events)
    try:
        with StepProbe(trainer) as probe:
            history = trainer.fit(train_ds, dev_ds, 8)
    finally:
        torch._dynamo.callback_handler.remove_start_callback(events)
    assert (
        [s["step"] for s in trainer.skipped_steps]
        == [6]
        == [s["step"] for s in eager.skipped_steps]
    )
    assert probe.unchanged(6) and probe.grads_none[6] and not probe.unchanged(7)
    assert history.measured["compile_steps"] == 1.0
    # nothing compiles in steps 2 to 8 (the final evaluation compiles its own variant after them)
    assert not any(2 <= step <= 8 for step in events.at), events.at


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_p1_15_spike_skip_compiled_cuda(
    precision: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On CUDA with CUDA Graphs (reduce-overhead; bf16 is profile:rtx3060's policy): a spike planted
    on a replayed step is skipped, every weight bitwise unchanged, and nothing records after the
    graphs' warm-up and recording (steps 1 and 2)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    from cards import CUDA_PROFILE

    cfg = hsn_config(
        8,
        *GUARD,
        f"profile={CUDA_PROFILE}",
        "hardware.compile_mode=reduce-overhead",
        f"hardware.autocast_dtype={precision}",
        f"train.precision={precision}",
    )
    train_ds, dev_ds = datasets()
    trainer = hsn_trainer(cfg, tmp_path / "run")
    compiled = trainer.compiled_window
    assert compiled is not None and compiled.cuda_graphs and trainer.scaler is None
    plant(monkeypatch, trainer, {6: SPIKE})
    events = _Events(trainer)
    torch._dynamo.callback_handler.register_start_callback(events)
    try:
        with StepProbe(trainer) as probe:
            history = trainer.fit(train_ds, dev_ds, 8)
    finally:
        torch._dynamo.callback_handler.remove_start_callback(events)
    assert [s["step"] for s in trainer.skipped_steps] == [6]
    assert probe.unchanged(6) and probe.grads_none[6] and not probe.unchanged(7)
    assert history.measured["compile_steps"] == 2.0
    # nothing compiles or records in steps 3 to 7 (step 8, a statistics step, and the final
    # evaluation may record graphs of their own: P1.12's CUDA test)
    assert not any(3 <= step <= 7 for step in events.at), events.at
    assert math.isfinite(history.evals[-1]["loss"])
