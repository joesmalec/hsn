"""P1.12 tests: the non-finite-gradient guard of both trainers (:mod:`hsn.train.loop`).

Without a loss scaler, a step whose global gradient norm is not finite leaves the weights and the
optimiser state untouched, zeroes the gradients, counts the skip, logs it and lets training
continue; the scheduler keeps its clock. The count is in every TBPTT step record, in the run's
``measured`` block and in the checkpoint (checkpoints written before the guard read 0, a malformed
counter raises). A run that skips step after step raises (the stop rule: too many consecutive
skips, or too large a fraction of the steps). Under fp16 the ``GradScaler`` skips such steps itself
and the guard stays out of its way.

The guard is checked on the eager path and on P2.8's compiled window: CPU (``aot_eager`` and, on
Linux, inductor) and CUDA (inductor with CUDA Graphs, the ``profile:rtx3060`` default). Poison is
planted in the objective (:func:`_plant`), which the trainer assembles eagerly, so it flows back
through the window's epilogue and the compiled blocks' backward (a CUDA-Graph replay) into every
parameter's gradient: either a NaN objective (the G1 seed-1 death at step 3,143) or an infinite
gradient under a finite objective (its step 3,140).
"""

from __future__ import annotations

import functools
import gc
import logging
import math
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest
import torch
from cards import CUDA_PROFILE
from hsn.config import RunConfig, load_config
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example, TaskSpec, generate, load_dataset_spec
from hsn.device import resolve, seed_everything
from hsn.experiments.diagnostics import Diagnostics
from hsn.experiments.protocols import ArmRun, run_protocol
from hsn.experiments.rules import evaluate_verdicts, parse_rules
from hsn.experiments.spec import protocol_params
from hsn.losses import assemble_terms
from hsn.model.hsn import HSN
from hsn.train import compile as compile_mod
from hsn.train import loop, tbptt
from hsn.train.compile import compile_window
from hsn.train.loop import History, Trainer
from hsn.train.tbptt import TBPTTStepRecord, TBPTTTrainer
from p1_11_support import (
    TINY_HSN_OVERRIDES,
    disk_overrides,
    restore_determinism,  # noqa: F401  (fixture)
    tiny_manifest,  # noqa: F401  (fixture)
)
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
GRU_2M = REPO / "configs" / "baselines" / "gru_2m.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
UNROLL = 32
#: inductor's CPU backend needs a C++ compiler with OpenMP (Linux runners have it; P2.8's choice).
CPU_BACKEND = "inductor" if sys.platform.startswith("linux") else "aot_eager"
#: ref_a1 shrunk (the P1.7 / P2.8 trainer-test model).
TINY = (
    "model.d=8",
    "model.context_dim=8",
    "model.n_cells=4",
    "model.in_degree=2",
    "model.n_units=2",
    "model.k_select=2",
    "model.hidden=16",
    "model.scorer_hidden=8",
)
NAN_OBJECTIVE = "nan_objective"
INF_GRADIENT = "inf_gradient"
#: Compiled vs eager trainer, relative, per step's objective (P2.8's trainer tests): CPU, CUDA fp32.
TOL_CPU = 1e-4
TOL_CUDA = 1e-3


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _release_compiled_state() -> Iterator[None]:
    """After every test: dynamo's caches, the CUDA Graph trees and their pools are released (as
    in the P2.8 tests), so later tests see the card as an eager process would."""
    yield
    torch._dynamo.reset()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        from torch._inductor.cudagraph_trees import reset_cudagraph_trees

        reset_cudagraph_trees()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compiled blocks of at most 4 positions: a 32-position window runs 8 blocks, and each
    compile stays short (P2.8's fixture)."""
    monkeypatch.setattr(compile_mod, "MAX_BLOCK", 4)


def _tiny_task() -> TaskSpec:
    task = SPEC.task("kv", "in_range")
    return TaskSpec.model_validate(
        {
            **task.model_dump(),
            "n_train": 32,
            "n_dev": 8,
            "n_test": 8,
            "length_range": [16, UNROLL],
            "dep_len_range": [4, 20],
            "unroll": UNROLL,
        }
    )


def _datasets() -> tuple[ByteDataset, ByteDataset]:
    def docs(split: str, n: int) -> list[Example]:
        return [generate(_tiny_task(), split, i) for i in range(n)]  # type: ignore[arg-type]

    return ByteDataset(docs("train", 32)), ByteDataset(docs("dev", 8))


def _train_overrides(steps: int) -> list[str]:
    """Evaluation, checkpoint and statistics at the last step only; a linear schedule, so every
    step has its own learning rate and a skipped step's scheduler advance is visible."""
    return [
        f"train.unroll={UNROLL}",
        "train.batch=4",
        "train.micro_batch=4",
        f"train.steps={steps}",
        "train.eval_every=1000",
        "train.ckpt_every=1000",
        "train.stats_every=1000",
        "train.warmup_steps=0",
        "train.schedule=linear",
    ]


def _hsn_config(steps: int, *overrides: str) -> RunConfig:
    return load_config(REF_A1, overrides=[*TINY, *_train_overrides(steps), *overrides])


def _gru_config(steps: int) -> RunConfig:
    return load_config(
        GRU_2M,
        overrides=["baseline.gru.d_model=16", "baseline.gru.n_layers=1", *_train_overrides(steps)],
    )


def _hsn_trainer(cfg: RunConfig, run_dir: Path) -> TBPTTTrainer:
    seed_everything(0)
    return TBPTTTrainer(HSN.from_config(cfg), cfg, resolve(cfg.hardware), run_dir=run_dir)


def _gru_trainer(cfg: RunConfig, run_dir: Path) -> Trainer:
    from hsn.runner import build_model

    seed_everything(0)
    return Trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=run_dir)


def _inf(grad: Tensor) -> Tensor:
    return torch.full_like(grad, math.inf)


def _poison(
    kind: str | None,
    out: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    cfg: RunConfig,
    extras: Mapping[str, Tensor] | None,
) -> dict[str, Tensor]:
    """:func:`hsn.losses.assemble_terms` of one window, poisoned by ``kind``: ``nan_objective``
    multiplies the task term by NaN; ``inf_gradient`` keeps the objective finite and returns an
    infinite gradient for the logits; ``None`` leaves it alone."""
    if kind == INF_GRADIENT:
        logits = out["logits"] * 1.0
        logits.register_hook(_inf)
        return assemble_terms({**out, "logits": logits}, batch, cfg, extras)
    terms = assemble_terms(out, batch, cfg, extras)
    if kind == NAN_OBJECTIVE:
        return {**terms, "task": terms["task"] * math.nan}
    assert kind is None, kind
    return terms


def _plant(monkeypatch: pytest.MonkeyPatch, trainer: Trainer, plan: Mapping[int, str]) -> None:
    """Poison the objective of the TBPTT trainer's windows at the steps of ``plan`` (step -> kind,
    :func:`_poison`). The trainer assembles its objective eagerly (outside the compiled blocks), so
    the poison reaches every parameter through the backward, compiled blocks included. Each call
    wraps the real :func:`hsn.losses.assemble_terms`."""

    def planted(
        out: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
        cfg: RunConfig,
        extras: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        # the windows of step n run while trainer.step == n - 1
        return _poison(plan.get(trainer.step + 1), out, batch, cfg, extras)

    monkeypatch.setattr(tbptt, "assemble_terms", planted)


def _plant_by_call(monkeypatch: pytest.MonkeyPatch, plan: Mapping[int, str]) -> None:
    """:func:`_plant` for a trainer built out of reach (a sweep protocol builds its own): ``plan``
    maps the n-th call of the TBPTT trainer's assembly to a kind. On the eager path with one
    window per step (``train.grad_accum = 1``) the n-th call is step n's window."""
    calls = [0]

    def planted(
        out: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
        cfg: RunConfig,
        extras: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        calls[0] += 1
        return _poison(plan.get(calls[0]), out, batch, cfg, extras)

    monkeypatch.setattr(tbptt, "assemble_terms", planted)


class _Probe:
    """What each step left behind, read at the trainer's scheduler step (once per step in both
    trainers, after the optimiser decision): a CPU copy of every parameter, whether every gradient
    is ``None``, the optimiser steps so far, the AdamW step counts and the compile / CUDA-Graph
    recording events so far (dynamo's compilation callbacks: a new frame, a lazy backward, a
    recording). The scheduler's ``step`` is wrapped on its class (for this trainer's scheduler
    only): an instance attribute would land in the scheduler's checkpointed state."""

    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer
        self.params: dict[int, list[Tensor]] = {}
        self.grads_none: dict[int, bool] = {}
        self.optimizer_steps: dict[int, int] = {}
        self.adam_steps: dict[int, list[float]] = {}
        self.events: dict[int, int] = {}
        self._optimizer_steps = 0
        self._events = 0
        self._cls = type(trainer.scheduler)
        self._own = self._cls.__dict__.get("step")  # None: inherited from LRScheduler

    def _record(self) -> None:
        trainer = self.trainer
        n = trainer.step + 1
        params = list(trainer.model.parameters())
        self.params[n] = [p.detach().cpu().clone() for p in params]
        self.grads_none[n] = all(p.grad is None for p in params)
        self.optimizer_steps[n] = self._optimizer_steps
        self.adam_steps[n] = sorted(
            {float(s["step"]) for s in trainer.optimizer.state.values() if "step" in s}
        )
        self.events[n] = self._events

    def __enter__(self) -> _Probe:
        original = self._cls.step
        probe, scheduler = self, self.trainer.scheduler

        def step(sched: Any, *args: Any, **kwargs: Any) -> None:
            if sched is scheduler:
                probe._record()
            original(sched, *args, **kwargs)

        self._cls.step = step  # type: ignore[method-assign]
        self._hook = self.trainer.optimizer.register_step_pre_hook(self._on_optimizer_step)
        torch._dynamo.callback_handler.register_start_callback(self._on_event)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        torch._dynamo.callback_handler.remove_start_callback(self._on_event)
        self._hook.remove()
        if self._own is None:
            del self._cls.step  # the inherited method again
        else:
            self._cls.step = self._own  # type: ignore[method-assign]

    def _on_optimizer_step(self, *_: Any) -> None:
        self._optimizer_steps += 1

    def _on_event(self, _: Any) -> None:
        self._events += 1

    def unchanged(self, step: int) -> bool:
        """Every parameter after ``step`` equals (bitwise) its value after ``step - 1``."""
        before, after = self.params[step - 1], self.params[step]
        return all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def _skipped_steps(history: History) -> list[int]:
    return [r.step for r in history.steps if not math.isfinite(r.grad_norm)]


def _assert_skips(history: History, probe: _Probe, skips: set[int], steps: int) -> None:
    """The skip contract over a run of ``steps`` with ``skips`` planted: exactly those steps
    report a non-finite norm and leave every weight and the optimiser untouched with every gradient
    zeroed (``None``); every other step trains; the counter is in the records and in measured."""
    assert [r.step for r in history.steps] == list(range(1, steps + 1))
    assert _skipped_steps(history) == sorted(skips)
    assert history.measured["skipped_nonfinite_steps"] == float(len(skips))
    taken = 0
    for n in range(1, steps + 1):
        record = history.steps[n - 1]
        assert isinstance(record, TBPTTStepRecord)
        assert record.stats["skipped_nonfinite"] == float(sum(1 for s in skips if s <= n))
        if n in skips:
            assert probe.grads_none[n], f"step {n}: the gradients are zeroed (set to None)"
            assert probe.unchanged(n), f"step {n}: a skipped step leaves every weight untouched"
        else:
            taken += 1
            assert not probe.grads_none[n]
            assert n == 1 or not probe.unchanged(n), f"step {n}: a finite step trains"
            assert math.isfinite(record.total), n
        assert probe.optimizer_steps[n] == taken, n
        assert probe.adam_steps[n] == ([float(taken)] if taken else []), n  # AdamW's own clock


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


# --------------------------------------------------------------------------------------------------
# Eager
# --------------------------------------------------------------------------------------------------


def test_p1_12_nonfinite_step_skipped_tbptt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """TBPTT trainer, eager: an infinite gradient at step 2 (finite objective) skips that step:
    weights, AdamW moments and step counts untouched, gradients zeroed, the skip counted in the
    records, in measured and in the log; the scheduler keeps its clock (the learning rates equal an
    unplanted run's); steps 3 and 4 train; the counter survives a checkpoint round trip."""
    cfg = _hsn_config(4)
    train_ds, dev_ds = _datasets()
    clean = _hsn_trainer(cfg, tmp_path / "clean").fit(train_ds, dev_ds, 4)
    assert _skipped_steps(clean) == [] and clean.measured["skipped_nonfinite_steps"] == 0.0
    trainer = _hsn_trainer(cfg, tmp_path / "run")
    _plant(monkeypatch, trainer, {2: INF_GRADIENT})
    with _Probe(trainer) as probe, caplog.at_level(logging.WARNING, logger="hsn.train"):
        history = trainer.fit(train_ds, dev_ds, 4)
    _assert_skips(history, probe, {2}, 4)
    assert math.isfinite(history.steps[1].total), "the objective of an inf_gradient plant is finite"
    assert history.steps[0].total == clean.steps[0].total  # the step before the plant: unchanged
    assert [r.lr for r in history.steps] == [r.lr for r in clean.steps]
    assert trainer.scheduler.last_epoch == 4
    assert trainer.skipped_nonfinite == 1
    assert any("optimizer step skipped" in m and "step 2" in m for m in caplog.messages)
    assert sum("optimizer step skipped" in m for m in caplog.messages) == 1
    other = _hsn_trainer(cfg, tmp_path / "other")
    assert other.resume(history.checkpoints[-1]) == 4
    assert other.skipped_nonfinite == 1 and other._skip_streak == 0


def test_p1_12_nonfinite_step_skipped_baseline(tmp_path: Path) -> None:
    """The generic P0.5 trainer (the baselines) applies the same guard: an infinite gradient at
    step 2 is skipped and counted, step 3 trains, the counter is in measured and in the
    checkpoint."""
    cfg = _gru_config(3)
    train_ds, dev_ds = _datasets()
    trainer = _gru_trainer(cfg, tmp_path / "run")
    target = next(p for p in trainer.model.parameters() if p.requires_grad)

    def hook(grad: Tensor) -> Tensor:
        return _inf(grad) if trainer.step + 1 == 2 else grad

    handle = target.register_hook(hook)
    with _Probe(trainer) as probe:
        history = trainer.fit(train_ds, dev_ds, 3)
    handle.remove()
    assert _skipped_steps(history) == [2]
    assert trainer.skipped_nonfinite == 1 and history.measured["skipped_nonfinite_steps"] == 1.0
    assert probe.grads_none[2] and probe.unchanged(2) and not probe.unchanged(3)
    assert probe.optimizer_steps == {1: 1, 2: 1, 3: 2}
    assert math.isfinite(history.steps[1].loss)
    other = _gru_trainer(cfg, tmp_path / "other")
    assert other.resume(history.checkpoints[-1]) == 3
    assert other.skipped_nonfinite == 1 and other._skip_streak == 0


@pytest.mark.parametrize("kind", [NAN_OBJECTIVE, INF_GRADIENT])
def test_p1_12_planted_nan_run_completes(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 6-step TBPTT run with poison planted at step 3 runs to its end: one skip, every other
    step finite, a finite final evaluation and the final checkpoint written."""
    cfg = _hsn_config(6)
    train_ds, dev_ds = _datasets()
    trainer = _hsn_trainer(cfg, tmp_path / "run")
    _plant(monkeypatch, trainer, {3: kind})
    with _Probe(trainer) as probe:
        history = trainer.fit(train_ds, dev_ds, 6)
    _assert_skips(history, probe, {3}, 6)
    assert math.isnan(history.steps[2].total) == (kind == NAN_OBJECTIVE)
    last = history.evals[-1]
    assert last["step"] == 6 and math.isfinite(last["loss"]) and math.isfinite(last["accuracy"])
    assert Path(history.checkpoints[-1]).name == "step_00000006.pt"
    assert Path(history.checkpoints[-1]).exists()


@pytest.mark.parametrize(("kind", "dies_at"), [(NAN_OBJECTIVE, 3), (INF_GRADIENT, 4)])
def test_p1_12_planted_nan_kills_a_run_without_the_guard(
    kind: str, dies_at: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plant reproduces the G1 failure when the guard is switched off: a NaN objective raises
    at once; an infinite gradient is clipped into NaN weights and the next step's objective is NaN
    (seed 1 of G1: an infinite norm at step 3,140, a NaN objective at 3,143)."""
    cfg = _hsn_config(6)
    train_ds, dev_ds = _datasets()
    trainer = _hsn_trainer(cfg, tmp_path / "run")
    _plant(monkeypatch, trainer, {3: kind})
    monkeypatch.setattr(Trainer, "_guard_nonfinite", lambda self, grad_norm, step: False)
    with pytest.raises(RuntimeError, match=f"the objective is nan at step {dies_at}"):
        trainer.fit(train_ds, dev_ds, 6)


def test_p1_12_guard_inactive_with_grad_scaler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With a loss scaler (the fp16 policy) the guard stays inactive: the scaler finds the
    infinite gradient, skips the optimiser step itself and backs its scale off; the guard counts
    nothing and logs nothing."""
    cfg = _hsn_config(3)
    train_ds, dev_ds = _datasets()
    trainer = _hsn_trainer(cfg, tmp_path / "run")
    assert trainer.scaler is None  # the CPU policy is fp32; a CPU scaler stands in for fp16
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=2.0**10)
    assert trainer._guard_nonfinite(math.inf, 1) is False and trainer.skipped_nonfinite == 0
    _plant(monkeypatch, trainer, {2: INF_GRADIENT})
    with _Probe(trainer) as probe, caplog.at_level(logging.WARNING, logger="hsn.train"):
        history = trainer.fit(train_ds, dev_ds, 3)
    assert _skipped_steps(history) == [2]
    assert probe.unchanged(2) and not probe.unchanged(3)  # the scaler skipped step 2
    assert trainer.scaler.get_scale() == 2.0**9  # one backoff
    assert trainer.skipped_nonfinite == 0 and history.measured["skipped_nonfinite_steps"] == 0.0
    assert all(
        isinstance(r, TBPTTStepRecord) and r.stats["skipped_nonfinite"] == 0.0
        for r in history.steps
    )
    assert not any("optimizer step skipped" in m for m in caplog.messages)


# --------------------------------------------------------------------------------------------------
# Stop rule
# --------------------------------------------------------------------------------------------------


def test_p1_12_stop_rule_consecutive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run whose every step from step 2 on is non-finite raises at the K-th consecutive skip
    (K patched to 3 here: step 4) with the counts in the message, instead of running on."""
    monkeypatch.setattr(loop, "NONFINITE_MAX_CONSECUTIVE", 3)
    cfg = _hsn_config(8)
    train_ds, dev_ds = _datasets()
    trainer = _hsn_trainer(cfg, tmp_path / "run")
    _plant(monkeypatch, trainer, dict.fromkeys(range(2, 9), INF_GRADIENT))
    with pytest.raises(RuntimeError, match=r"step 4: 3 consecutive optimizer steps skipped") as err:
        trainer.fit(train_ds, dev_ds, 8)
    assert "3 of 4 steps skipped" in str(err.value)
    assert trainer.skipped_nonfinite == 3 and trainer._skip_streak == 3


def test_p1_12_stop_rule_fraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolated skips raise once they exceed the allowed fraction of the steps (patched here: at
    least 2 skips and more than 25 % of the steps): skips at steps 3 and 8 pass (2 of 8 is not more
    than 25 %), the one at step 10 raises (3 of 10); no two of them are consecutive."""
    monkeypatch.setattr(loop, "NONFINITE_MIN_SKIPS", 2)
    monkeypatch.setattr(loop, "NONFINITE_MAX_FRACTION", 0.25)
    cfg = _hsn_config(12)
    train_ds, dev_ds = _datasets()
    trainer = _hsn_trainer(cfg, tmp_path / "run")
    _plant(monkeypatch, trainer, {3: INF_GRADIENT, 8: NAN_OBJECTIVE, 10: INF_GRADIENT})
    with pytest.raises(RuntimeError, match=r"step 10: 3 of 10 steps skipped") as err:
        trainer.fit(train_ds, dev_ds, 12)
    assert "consecutive" not in str(err.value)
    assert trainer.skipped_nonfinite == 3 and trainer._skip_streak == 1


# --------------------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------------------


def _rewrite(path: str | Path, out: Path, **extra: Any) -> Path:
    """A copy of the checkpoint ``path`` at ``out`` with ``extra`` entries replaced (``None``
    removes the entry: a checkpoint written before the guard)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key, value in extra.items():
        if value is None:
            payload["extra"].pop(key)
        else:
            payload["extra"][key] = value
    torch.save(payload, out)
    return out


MALFORMED = [
    ("skipped_nonfinite", -1),
    ("skipped_nonfinite", True),
    ("skipped_nonfinite", 1.0),
    ("skipped_nonfinite", "1"),
    ("skipped_nonfinite_streak", -1),
    ("skipped_nonfinite_streak", 2),  # more consecutive skips than skips
]


def test_p1_12_checkpoint_counters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both counters are in the checkpoint: a run resumed after a skipped last step continues the
    consecutive count (K patched to 2: the next skip raises); a checkpoint written before the
    guard (no counters) resumes with 0; a malformed counter raises. The generic trainer too."""
    cfg = _hsn_config(4)
    train_ds, dev_ds = _datasets()
    trainer = _hsn_trainer(cfg, tmp_path / "run")
    _plant(monkeypatch, trainer, {2: INF_GRADIENT})
    history = trainer.fit(train_ds, dev_ds, 2)
    path = history.checkpoints[-1]
    extra = torch.load(path, map_location="cpu", weights_only=True)["extra"]
    assert extra["skipped_nonfinite"] == 1 and extra["skipped_nonfinite_streak"] == 1
    monkeypatch.setattr(loop, "NONFINITE_MAX_CONSECUTIVE", 2)
    resumed = _hsn_trainer(cfg, tmp_path / "resumed")
    assert resumed.resume(path) == 2
    assert resumed.skipped_nonfinite == 1 and resumed._skip_streak == 1
    _plant(monkeypatch, resumed, {3: INF_GRADIENT})
    with pytest.raises(RuntimeError, match=r"step 3: 2 consecutive optimizer steps skipped"):
        resumed.fit(train_ds, dev_ds, 3)
    old = _rewrite(path, tmp_path / "old.pt", skipped_nonfinite=None, skipped_nonfinite_streak=None)
    fresh = _hsn_trainer(cfg, tmp_path / "fresh")
    assert fresh.resume(old) == 2 and fresh.skipped_nonfinite == 0 and fresh._skip_streak == 0
    for index, (key, value) in enumerate(MALFORMED):
        bad = _rewrite(path, tmp_path / f"bad{index}.pt", **{key: value})
        with pytest.raises(ValueError, match=key):
            _hsn_trainer(cfg, tmp_path / f"bad{index}").resume(bad)
    # the generic trainer reads and checks the same entries
    gru_cfg = _gru_config(2)
    gru = _gru_trainer(gru_cfg, tmp_path / "gru")
    gru_path = gru.fit(train_ds, dev_ds, 1).checkpoints[-1]
    extra = torch.load(gru_path, map_location="cpu", weights_only=True)["extra"]
    assert extra["skipped_nonfinite"] == 0 and extra["skipped_nonfinite_streak"] == 0
    old = _rewrite(
        gru_path, tmp_path / "gru_old.pt", skipped_nonfinite=None, skipped_nonfinite_streak=None
    )
    assert _gru_trainer(gru_cfg, tmp_path / "gru_fresh").resume(old) == 1
    bad = _rewrite(gru_path, tmp_path / "gru_bad.pt", skipped_nonfinite=-1)
    with pytest.raises(ValueError, match="skipped_nonfinite"):
        _gru_trainer(gru_cfg, tmp_path / "gru_bad").resume(bad)


# --------------------------------------------------------------------------------------------------
# P1.11's run record
# --------------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("restore_determinism")
def test_p1_12_sweep_record_carries_skips(
    tiny_manifest: Path,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count reaches P1.11's run record (the field agreed with P1.11): a ``train_eval`` arm
    with an infinite gradient at step 2 of 4 completes, its record carries
    ``final.skipped_nonfinite_steps = 1`` (and ``measured``), its ``train`` series shows the
    skipped step's norm as missing, and a verdict rule on ``final:skipped_nonfinite_steps`` reads
    it."""
    cfg = load_config(REF_A1, [*TINY_HSN_OVERRIDES, *disk_overrides(tiny_manifest, steps=4)])
    arm = ArmRun(
        cfg=cfg,
        family="hsn",
        variant="pdg",
        variant_params={},
        params=protocol_params({"name": "train_eval"}),
        run_dir=tmp_path / "arm",
        rules=parse_rules(None),
        constants={},
    )
    _plant_by_call(monkeypatch, {2: INF_GRADIENT})
    record = run_protocol(arm)
    assert record["final"]["steps"] == 4 and record["stopped"] is None
    assert record["final"]["skipped_nonfinite_steps"] == 1.0
    assert record["measured"]["skipped_nonfinite_steps"] == 1.0
    assert [t["grad_norm"] is None for t in record["train"]] == [False, True, False, False]
    rules = parse_rules(
        {
            "verdicts": {
                "guard": {
                    "cases": [
                        {
                            "label": "skipped",
                            "when": {
                                "metric": "final:skipped_nonfinite_steps",
                                "op": ">",
                                "value": 0,
                            },
                        }
                    ],
                    "default": "clean",
                }
            }
        }
    )
    assert evaluate_verdicts(rules, record, {}, "hsn")["guard"]["label"] == "skipped"


def test_p1_12_diagnostics_survive_a_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P1.11's diagnostics (every step) on a TBPTT run with an infinite gradient at step 3 of 5:
    the run continues (the diagnostics compare the optimiser calls net of the guard's skips instead
    of raising at step 4), the skipped step has no record, and the records resume at step 4."""
    cfg = _hsn_config(5)
    train_ds, dev_ds = _datasets()
    trainer = _hsn_trainer(cfg, tmp_path / "run")
    diag = Diagnostics(trainer.model, cfg, trainer.ctx, dev_ds, 1)
    diag.attach(trainer)
    _plant(monkeypatch, trainer, {3: INF_GRADIENT})
    history = trainer.fit(train_ds, dev_ds, 5)
    diag.detach()
    assert _skipped_steps(history) == [3] and trainer.skipped_nonfinite == 1
    assert [r["step"] for r in diag.records] == [1, 2, 4, 5]
    assert [r["optimizer_calls"] for r in diag.records] == [1, 2, 3, 4]
    assert diag.optimizer_calls == 4
    # any other mismatch still raises
    other = _hsn_trainer(cfg, tmp_path / "other")
    strict = Diagnostics(other.model, cfg, other.ctx, dev_ds, 1)
    strict.attach(other)
    other.skipped_nonfinite = 1  # a count no skip produced
    with pytest.raises(RuntimeError, match="optimiser call"):
        other.fit(train_ds, dev_ds, 1)
    strict.detach()


# --------------------------------------------------------------------------------------------------
# The compiled window (P2.8)
# --------------------------------------------------------------------------------------------------

#: A NaN objective at step 3 and an infinite gradient at step 5 of a 7-step run.
COMPILED_PLAN = {3: NAN_OBJECTIVE, 5: INF_GRADIENT}


def _planted_run(
    cfg: RunConfig, run_dir: Path, monkeypatch: pytest.MonkeyPatch, plan: Mapping[int, str]
) -> tuple[TBPTTTrainer, History, _Probe]:
    train_ds, dev_ds = _datasets()
    trainer = _hsn_trainer(cfg, run_dir)
    _plant(monkeypatch, trainer, plan)
    with _Probe(trainer) as probe:
        history = trainer.fit(train_ds, dev_ds, cfg.train.steps)
    return trainer, history, probe


def _assert_matches_eager(compiled: History, eager: History, tol: float) -> None:
    """Per step: the same skips, and the objectives of the finite steps to ``tol`` (the compiled
    window equals eager to rounding)."""
    assert _skipped_steps(compiled) == _skipped_steps(eager)
    for a, b in zip(compiled.steps, eager.steps, strict=True):
        assert isinstance(a, TBPTTStepRecord) and isinstance(b, TBPTTStepRecord)
        if math.isfinite(b.total):
            assert _close(a.total, b.total, tol), (a.step, a.total, b.total)
        else:
            assert not math.isfinite(a.total), a.step


@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("backend", sorted({"aot_eager", CPU_BACKEND}))
def test_p1_12_compiled_skip_matches_eager(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2.8's compiled window on the CPU (``aot_eager``; inductor on Linux): a NaN objective at
    step 3 and an infinite gradient at step 5 are skipped exactly as on the eager path (the same
    steps, the other steps' objectives equal to rounding), every weight is bitwise unchanged across
    each skip, and nothing compiles after the first step (no recompile from a skipped step)."""
    steps = max(COMPILED_PLAN) + 2
    _, eager, _ = _planted_run(_hsn_config(steps), tmp_path / "eager", monkeypatch, COMPILED_PLAN)
    monkeypatch.setattr(tbptt, "compile_window", functools.partial(compile_window, backend=backend))
    cfg = _hsn_config(steps, "hardware.compile_mode=reduce-overhead")
    trainer, history, probe = _planted_run(cfg, tmp_path / "compiled", monkeypatch, COMPILED_PLAN)
    assert trainer.compiled_window is not None
    assert history.measured["compiled_windows"] >= float(steps)
    _assert_skips(history, probe, set(COMPILED_PLAN), steps)
    _assert_matches_eager(history, eager, TOL_CPU)
    assert history.measured["compile_steps"] == 1.0
    assert probe.events[1] > 0, "the first step compiles"
    assert all(probe.events[n] == probe.events[1] for n in range(2, steps + 1)), probe.events


def _cuda_config(steps: int, precision: str, compile_mode: str) -> RunConfig:
    return _hsn_config(
        steps,
        f"profile={CUDA_PROFILE}",
        f"hardware.compile_mode={compile_mode}",
        f"hardware.autocast_dtype={precision}",
        f"train.precision={precision}",
    )


#: On CUDA the plants come after the CUDA Graph warm-up (step 1) and recording (step 2), on
#: replays: a NaN objective at step 4 and an infinite gradient at step 6 of an 8-step run. The
#: last step is a statistics step, whose extra compiled passes (P2.8) may record graphs of their
#: own; the no-recording check covers the steps before it.
CUDA_PLAN = {4: NAN_OBJECTIVE, 6: INF_GRADIENT}


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_p1_12_compiled_skip_cuda(
    precision: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On CUDA with CUDA Graphs (inductor, ``reduce-overhead``; bf16 is ``profile:rtx3060``'s
    policy, no loss scaler): skips planted on replayed steps leave every weight bitwise unchanged
    and the gradients zeroed, the steps after them train on the same graphs (nothing compiles or
    records after step 2, the recording), and the run completes; in fp32 the objectives equal an
    eager CUDA run's to P2.8's tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    steps = max(CUDA_PLAN) + 2
    cfg = _cuda_config(steps, precision, "reduce-overhead")
    trainer, history, probe = _planted_run(cfg, tmp_path / "compiled", monkeypatch, CUDA_PLAN)
    compiled = trainer.compiled_window
    assert compiled is not None and compiled.cuda_graphs
    assert trainer.scaler is None  # bf16 and fp32: the guard is the only skip
    _assert_skips(history, probe, set(CUDA_PLAN), steps)
    assert history.measured["compile_steps"] == 2.0  # warm-up and recording
    assert all(probe.events[n] == probe.events[2] for n in range(3, steps)), probe.events
    last = history.evals[-1]
    assert math.isfinite(last["loss"]) and math.isfinite(last["accuracy"])
    if precision == "fp32":
        _, eager, _ = _planted_run(
            _cuda_config(steps, precision, "none"), tmp_path / "eager", monkeypatch, CUDA_PLAN
        )
        _assert_matches_eager(history, eager, TOL_CUDA)
