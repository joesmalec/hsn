"""P1.16 tests: rollback on a guard trigger (``train.rollback_*``, :mod:`hsn.train.rollback`).

Off by default (P1.13's golden runs keep today's training bitwise; with the cadence set and no
trigger the snapshots only read). With it on, planted spikes after the first snapshots roll back:
the weights equal the older snapshot bit for bit, the optimiser is the snapshot's, the windows
between the snapshot and the trigger are never replayed, ``rollback_skip`` more are discarded, every
row starts a fresh document, the learning rate is cut for ``rollback_lr_steps`` steps, nothing
counts in P1.12's stop rule, and the run stops at ``rollback_max + 1`` triggers. A trigger before
the first snapshot is P1.15's skip. The state survives a checkpoint; P1.11's diagnostics count net
of the triggers; P2.8's compiled window keeps its parameters' addresses.
"""

from __future__ import annotations

import functools
import logging
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.experiments.diagnostics import Diagnostics
from hsn.train import compile as compile_mod
from hsn.train import loop, tbptt
from hsn.train.compile import compile_window
from hsn.train.loop import ROLLBACK_LIMIT, StreamLoader
from hsn.train.rollback import RollbackPolicy, SnapshotRing
from p1_11_support import restore_determinism  # noqa: F401  (fixture)
from p1_15_support import (
    GUARD,
    SPIKE,
    StepProbe,
    datasets,
    gru_config,
    gru_trainer,
    hsn_config,
    hsn_trainer,
    plant,
    plant_gru_spike,
)
from torch import nn

pytestmark = pytest.mark.usefixtures("restore_determinism")

#: snapshots every 5 steps; 2 windows discarded; lr x 0.5 for 3 steps; at most 5 rollbacks
ROLLBACK = (
    "train.rollback_every=5",
    "train.rollback_skip=2",
    "train.rollback_lr_mult=0.5",
    "train.rollback_lr_steps=3",
    "train.rollback_max=5",
)


@pytest.fixture(autouse=True)
def _release_compiled_state() -> Any:
    yield
    torch._dynamo.reset()


class _Draws:
    """Every window a training :class:`StreamLoader` (an endless one) emits, in order (a wrapper
    on the class; evaluation loaders are finite and left out)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.windows: list[dict[str, torch.Tensor]] = []
        original = StreamLoader.__next__
        draws = self

        def recorded(loader: StreamLoader) -> dict[str, torch.Tensor]:
            window = original(loader)
            if loader.epochs is None:
                draws.windows.append(window)
            return window

        monkeypatch.setattr(StreamLoader, "__next__", recorded)


# --------------------------------------------------------------------------------------------------
# The snapshot ring
# --------------------------------------------------------------------------------------------------


def _step(model: nn.Module, opt: torch.optim.Optimizer, seed: int) -> None:
    torch.manual_seed(seed)
    loss = model(torch.randn(4, 3)).square().sum()
    opt.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    opt.step()


def test_p1_16_snapshot_ring_restores_in_place() -> None:
    """Two slots, the older restored, in place (same storage), the optimiser exactly the
    snapshot's (moments, step counters; state created after the snapshot dropped)."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 2))
    model.register_buffer("avg", torch.zeros(2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    ring = SnapshotRing(model, opt)
    ring.take(0)  # before any optimiser state exists
    _step(model, opt, 1)
    ring.take(1, [0.5, 0.25])
    assert ring.steps == [0, 1]
    _step(model, opt, 2)
    ring.take(2)
    assert ring.steps == [1, 2]  # at most two: the oldest slot is reused
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    _step(model, opt, 3)  # the reference: snapshot 1 is after step 1
    ptrs = [t.data_ptr() for t in model.state_dict(keep_vars=True).values()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
        model.avg.fill_(7.0)  # type: ignore[operator]
    snap = ring.restore()
    assert snap.step == 1 and ring.steps == [1] and snap.spike_history == [0.5, 0.25]
    assert ptrs == [t.data_ptr() for t in model.state_dict(keep_vars=True).values()]
    # the restored weights are those after step 1 (re-derive them on a fresh copy)
    torch.manual_seed(0)
    ref = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 2))
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-2)
    _step(ref, ref_opt, 1)
    for (name, a), b in zip(model.named_parameters(), ref.parameters(), strict=True):
        assert torch.equal(a, b), name
    assert torch.equal(model.avg, torch.zeros(2))  # type: ignore[arg-type]
    for p, q in zip(model.parameters(), ref.parameters(), strict=True):
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(opt.state[p][key], ref_opt.state[q][key]), key
    assert not torch.equal(model[0].weight, expected["0.weight"])  # snapshot 2 was not taken back
    # a snapshot taken before any optimiser step restores an optimiser without state
    fresh = torch.optim.AdamW(model.parameters(), lr=1e-2)
    empty = SnapshotRing(model, fresh)
    empty.take(0)
    _step(model, fresh, 4)
    assert fresh.state
    empty.restore()
    assert not fresh.state


def test_p1_16_loader_drop_documents() -> None:
    """After :meth:`StreamLoader.drop_documents` every row starts a fresh document at t = 0 (with
    its reset); rows that were mid-document never emit the rest of it."""
    train_ds, _ = datasets()  # documents of 16 to 32 positions
    loader = StreamLoader(
        train_ds, batch=4, unroll=8, seed=0, shuffle=True, align_docs=True, epochs=None
    )
    first = next(loader)
    assert bool(first["reset"][:, 0].all())
    mid = {row.doc for row in loader._rows}
    assert -1 not in mid and len(mid) == 4  # every row is inside a document longer than 8
    assert not bool(next(loader)["reset"][:, 0].any())  # without the drop, rows continue
    loader.drop_documents()
    after = next(loader)
    assert bool(after["reset"][:, 0].all())
    assert not set(after["doc"][:, 0].tolist()) & mid


def test_p1_16_rollback_policy_from_config() -> None:
    assert RollbackPolicy.from_config(hsn_config(4).train) is None
    policy = RollbackPolicy.from_config(hsn_config(4, *ROLLBACK).train)
    assert policy == RollbackPolicy(every=5, skip=2, lr_mult=0.5, lr_steps=3, max_rollbacks=5)
    with pytest.raises(ValueError):
        RollbackPolicy(every=0)


# --------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------


def test_p1_16_rollback_snapshots_only_read(tmp_path: Path) -> None:
    """With the cadence set and no trigger, training is bitwise the training without it."""
    train_ds, dev_ds = datasets()
    off = hsn_trainer(hsn_config(12, *GUARD), tmp_path / "off")
    h_off = off.fit(train_ds, dev_ds, 12)
    on = hsn_trainer(hsn_config(12, *GUARD, *ROLLBACK), tmp_path / "on")
    h_on = on.fit(train_ds, dev_ds, 12)
    assert [r.loss for r in h_on.steps] == [r.loss for r in h_off.steps]
    assert [r.grad_norm for r in h_on.steps] == [r.grad_norm for r in h_off.steps]
    assert [r.lr for r in h_on.steps] == [r.lr for r in h_off.steps]
    for a, b in zip(on.model.parameters(), off.model.parameters(), strict=True):
        assert torch.equal(a, b)
    assert on.rollbacks == [] and h_on.measured["rollbacks"] == 0.0
    assert "rollbacks" not in h_off.measured
    assert on._ring is not None and on._ring.steps == [5, 10]


def test_p1_16_rollback_planted_ramp_survives_hsn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Spikes at steps 22 and 23 (snapshots at 5, 10, 15, 20): both roll back to the snapshot of
    step 15 (the older of 15 and 20, then the only one), bitwise; nothing is skipped; the run
    finishes; the learning rate is halved for the 3 steps after each trigger; 2 windows are
    discarded per rollback and every row starts a fresh document after it."""
    train_ds, dev_ds = datasets()
    clean = hsn_trainer(hsn_config(30, *GUARD), tmp_path / "clean")
    h_clean = clean.fit(train_ds, dev_ds, 30)
    cfg = hsn_config(30, *GUARD, *ROLLBACK)
    trainer = hsn_trainer(cfg, tmp_path / "run")
    plant(monkeypatch, trainer, {22: SPIKE, 23: SPIKE})
    draws = _Draws(monkeypatch)
    ptrs = [p.data_ptr() for p in trainer.model.parameters()]
    with StepProbe(trainer) as probe, caplog.at_level(logging.WARNING, logger="hsn.train"):
        history = trainer.fit(train_ds, dev_ds, 30)
    assert trainer.step == 30 and math.isfinite(history.evals[-1]["loss"])
    assert ptrs == [p.data_ptr() for p in trainer.model.parameters()]
    for step in (22, 23):  # at each trigger's scheduler step the weights are those after step 15
        assert all(
            torch.equal(a, b) for a, b in zip(probe.params[step], probe.params[15], strict=True)
        )
    assert [(r["step"], r["restored_step"], r["lost_steps"]) for r in trainer.rollbacks] == [
        (22, 15, 6),
        (23, 15, 7),
    ]
    assert all(r["kind"] == SPIKE and r["windows_skipped"] == 2 for r in trainer.rollbacks)
    assert trainer.skipped_spike == 0 and trainer.skipped_nonfinite == 0
    assert [s["reason"] for s in trainer.skipped_steps] == ["rollback", "rollback"]
    assert history.measured["rollbacks"] == 2.0
    assert [r.stats["rollbacks"] for r in history.steps][20:24] == [0.0, 1.0, 2.0, 2.0]  # type: ignore[attr-defined]
    ratios = [a.lr / b.lr for a, b in zip(history.steps, h_clean.steps, strict=True)]
    cut = {23, 24, 25, 26}  # 22 + 3 then 23 + 3
    for n, ratio in enumerate(ratios, start=1):
        assert ratio == pytest.approx(0.5 if n in cut else 1.0, rel=1e-12), n
    # 30 training windows plus 2 discarded per rollback; the windows after a rollback reset all rows
    assert len(draws.windows) == 34
    assert bool(draws.windows[24]["reset"][:, 0].all())  # step 23's window (after 22 + 2)
    assert bool(draws.windows[27]["reset"][:, 0].all())  # step 24's window (after 23 + 2 + 2)
    assert sum("rolled back to the snapshot of step 15" in m for m in caplog.messages) == 2
    assert trainer._ring is not None and trainer._ring.steps == [25, 30]


def test_p1_16_rollback_planted_ramp_survives_gru(tmp_path: Path) -> None:
    """The generic trainer (the baselines) rolls back the same way."""
    train_ds, dev_ds = datasets()
    trainer = gru_trainer(gru_config(30, *GUARD, *ROLLBACK), tmp_path / "run")
    handles = plant_gru_spike(trainer, {22, 23})
    try:
        with StepProbe(trainer) as probe:
            history = trainer.fit(train_ds, dev_ds, 30)
    finally:
        for h in handles:
            h.remove()
    assert trainer.step == 30 and history.measured["rollbacks"] == 2.0
    assert [r["restored_step"] for r in trainer.rollbacks] == [15, 15]
    assert all(torch.equal(a, b) for a, b in zip(probe.params[22], probe.params[15], strict=True))
    assert trainer.skipped_spike == 0


def test_p1_16_rollback_before_first_snapshot_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trigger before the first snapshot is P1.15's skip (it counts in P1.12's rule)."""
    train_ds, dev_ds = datasets()
    cfg = hsn_config(12, *GUARD, "train.rollback_every=10")
    trainer = hsn_trainer(cfg, tmp_path / "run")
    plant(monkeypatch, trainer, {7: SPIKE})
    trainer.fit(train_ds, dev_ds, 12)
    assert trainer.rollbacks == [] and trainer.skipped_spike == 1
    assert [s["reason"] for s in trainer.skipped_steps] == [SPIKE]


def test_p1_16_rollback_limit_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At ``rollback_max + 1`` triggers the run raises, naming the rollback limit (the harness's
    failure kind ``rollback``); the earlier triggers are rollbacks, not skips."""
    train_ds, dev_ds = datasets()
    cfg = hsn_config(30, *GUARD, "train.rollback_every=5", "train.rollback_max=2")
    trainer = hsn_trainer(cfg, tmp_path / "run")
    plant(monkeypatch, trainer, dict.fromkeys(range(22, 31), SPIKE))
    with pytest.raises(RuntimeError, match=ROLLBACK_LIMIT) as caught:
        trainer.fit(train_ds, dev_ds, 30)
    assert "P1.12 stop rule" not in str(caught.value)
    assert len(trainer.rollbacks) == 2 and trainer.skipped_spike == 0
    from hsn.experiments.protocols import failure_info

    info = failure_info(caught.value, stage="train", last_step=trainer.step)
    assert info["kind"] == "rollback" and "rollback" in info["reason"]
    from hsn.experiments import protocols

    assert protocols._ROLLBACK_LIMIT == ROLLBACK_LIMIT


def test_p1_16_rollback_nonfinite_trigger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-finite gradient rolls back too (no spike guard needed)."""
    from p1_15_support import INF_GRADIENT

    train_ds, dev_ds = datasets()
    cfg = hsn_config(14, "train.rollback_every=5")
    trainer = hsn_trainer(cfg, tmp_path / "run")
    plant(monkeypatch, trainer, {12: INF_GRADIENT})
    trainer.fit(train_ds, dev_ds, 14)
    assert [(r["step"], r["kind"], r["restored_step"]) for r in trainer.rollbacks] == [
        (12, "nonfinite", 5)
    ]
    assert trainer.skipped_nonfinite == 0
    assert trainer.rollbacks[0]["grad_norm"] in ("inf", "nan")  # kept as text


def test_p1_16_rollback_resume_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The events and the learning-rate cut survive a checkpoint; the snapshots do not (a resumed
    run re-arms at its next multiple of the cadence)."""
    train_ds, dev_ds = datasets()
    cfg = hsn_config(30, *GUARD, *ROLLBACK)
    first = hsn_trainer(cfg, tmp_path / "run")
    plant(monkeypatch, first, {22: SPIKE})
    first.fit(train_ds, dev_ds, 23)  # the checkpoint at the last step, 23
    path = tmp_path / "run" / "step_00000023.pt"
    assert path.is_file()
    second = hsn_trainer(cfg, tmp_path / "resumed")
    assert second.resume(path) == 23
    assert second.rollbacks == first.rollbacks and len(second.rollbacks) == 1
    assert (second._lr_cut_until, second._lr_cut_mult) == (25, 0.5)
    assert second._ring is None  # the plant reads ``first``'s step: no spike from here on
    history = second.fit(train_ds, dev_ds, 30)
    assert second._ring is not None and second._ring.steps == [25, 30]
    lrs = {r.step: r.lr for r in history.steps}
    assert lrs[25] == pytest.approx(0.5 * cfg.train.lr * (1 - 24 / 30), rel=1e-12)
    assert lrs[26] == pytest.approx(cfg.train.lr * (1 - 25 / 30), rel=1e-12)


def test_p1_16_rollback_diagnostics_call_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1.11's diagnostics count optimiser calls net of the rollbacks' trigger steps."""
    train_ds, dev_ds = datasets()
    cfg = hsn_config(24, *GUARD, *ROLLBACK)
    trainer = hsn_trainer(cfg, tmp_path / "run")
    diag = Diagnostics(trainer.model, cfg, trainer.ctx, dev_ds, 1)
    diag.attach(trainer)
    plant(monkeypatch, trainer, {22: SPIKE})
    trainer.fit(train_ds, dev_ds, 24)
    diag.detach()
    assert [r["step"] for r in diag.records] == [*range(1, 22), 23, 24]


def test_p1_16_rollback_refused_with_grad_scaler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        loop, "make_grad_scaler", lambda ctx: torch.amp.GradScaler("cpu", init_scale=2.0**10)
    )
    with pytest.raises(NotImplementedError, match="GradScaler"):
        hsn_trainer(hsn_config(4, "train.rollback_every=5"), tmp_path / "run")


# --------------------------------------------------------------------------------------------------
# P2.8's compiled window
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compile_mod, "MAX_BLOCK", 4)


@pytest.mark.usefixtures("small_blocks")
def test_p1_16_rollback_compiled_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The compiled window (aot_eager on the CPU) rolls back like eager: the same events, the
    parameters' addresses kept, nothing recompiled after the first step."""
    train_ds, dev_ds = datasets()
    eager = hsn_trainer(hsn_config(24, *GUARD, *ROLLBACK), tmp_path / "eager")
    plant(monkeypatch, eager, {22: SPIKE})
    eager.fit(train_ds, dev_ds, 24)
    monkeypatch.setattr(
        tbptt, "compile_window", functools.partial(compile_window, backend="aot_eager")
    )
    cfg = hsn_config(24, *GUARD, *ROLLBACK, "hardware.compile_mode=reduce-overhead")
    trainer = hsn_trainer(cfg, tmp_path / "compiled")
    assert trainer.compiled_window is not None
    plant(monkeypatch, trainer, {22: SPIKE})
    ptrs = [p.data_ptr() for p in trainer.model.parameters()]
    with StepProbe(trainer) as probe:
        history = trainer.fit(train_ds, dev_ds, 24)
    assert ptrs == [p.data_ptr() for p in trainer.model.parameters()]
    assert [(r["step"], r["restored_step"]) for r in trainer.rollbacks] == [(22, 15)]
    assert [(r["step"], r["restored_step"]) for r in eager.rollbacks] == [(22, 15)]
    assert all(torch.equal(a, b) for a, b in zip(probe.params[22], probe.params[15], strict=True))
    assert history.measured["compile_steps"] == 1.0


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
def test_p1_16_rollback_compiled_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On CUDA with CUDA Graphs (reduce-overhead, bf16): a rollback restores the weights in place
    (bitwise the snapshot's, the same addresses), and the graphs replay on: nothing re-records in
    the steps between the rollback and the last statistics step."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    from cards import CUDA_PROFILE

    cfg = hsn_config(
        24,
        *GUARD,
        *ROLLBACK,
        f"profile={CUDA_PROFILE}",
        "hardware.compile_mode=reduce-overhead",
        "hardware.autocast_dtype=bf16",
        "train.precision=bf16",
    )
    train_ds, dev_ds = datasets()
    trainer = hsn_trainer(cfg, tmp_path / "run")
    compiled = trainer.compiled_window
    assert compiled is not None and compiled.cuda_graphs
    plant(monkeypatch, trainer, {22: SPIKE})
    ptrs = [p.data_ptr() for p in trainer.model.parameters()]
    starts: list[int] = []

    def started(_: Any) -> None:
        starts.append(trainer.step + 1)

    torch._dynamo.callback_handler.register_start_callback(started)
    try:
        with StepProbe(trainer) as probe:
            history = trainer.fit(train_ds, dev_ds, 24)
    finally:
        torch._dynamo.callback_handler.remove_start_callback(started)
    assert ptrs == [p.data_ptr() for p in trainer.model.parameters()]
    assert [(r["step"], r["restored_step"]) for r in trainer.rollbacks] == [(22, 15)]
    assert all(torch.equal(a, b) for a, b in zip(probe.params[22], probe.params[15], strict=True))
    assert not any(3 <= s <= 23 for s in starts), starts
    assert math.isfinite(history.evals[-1]["loss"])
