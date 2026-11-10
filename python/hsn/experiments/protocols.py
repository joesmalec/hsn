"""The protocols a sweep arm runs (P1.11): what is trained on what, what is measured after.

* ``reads_state``: the P1.10 audit's reads-state check (``scripts/eff_reads_state.py`` at
  ``f8884ec``): train on the configuration's data (the spec's overrides carry the 5(b) recipe),
  diagnostics every ``diag_every`` steps on the first ``diag_window_docs`` dev documents, then the
  trained model's state statistics on ``diag_docs`` dev documents and the 5(c) edit probe on every
  dev ``delayed/in_range`` document (batched, :mod:`hsn.experiments.probe`).
* ``train_eval``: train and evaluate, with optional diagnostics and the trained state; no probe.
* ``overfit``: the G1 overfit recipe (``scripts/eff_overfit.py``): the first ``n_docs`` training
  documents are both the training and the "dev" set, so every evaluation is their training accuracy;
  the record gives the first evaluation at or above each threshold.
* ``dry_run``: no training and no torch: a synthetic record (evaluations at the arm's evaluation
  steps, an optional synthetic probe), optionally after a sleep or a deliberate failure. It serves
  the queue's tests and checks a spec end to end.

The order of the audit is kept: ``seed_everything(seed)``, build the model, install the variant
(:mod:`hsn.experiments.variants`), build the trainer (which seeds again). New: the early-stop
callback (:func:`hsn.experiments.rules.check_early_stop`, evaluated right after the step's
evaluation and diagnostics), the evaluation batch, and per-phase wall times: setup, the fit, the
trainer's training seconds without the diagnostics hooks' own time, P2.8's compile seconds (the
training steps' share, and every compiled variant's, evaluation included), the evaluations (a
profiler that times the trainer's ``evaluate`` segment only), the diagnostics, the trained state,
the probe and the total. After an early stop the probe and the trained state still run; they are
stored under ``at_stop`` and never feed a verdict.

The record (:func:`run_protocol`'s result; :class:`hsn.experiments.rules.RecordView` reads it) has
``evals`` (``step``, ``accuracy``, ``loss``, ``per_group``; reads-state runs also the audit's
``dev_answer_accuracy`` / ``dev_answer_loss``), ``diagnostics``, ``train`` (every step's loss,
accuracy, gradient norm and learning rate), ``final``, ``trained_state``, ``probe`` (counts;
``probes`` holds every probe), ``overfit``, ``stopped`` / ``early_stop_log`` / ``at_stop``,
``walls``, ``measured``, ``compile`` and ``variant``.

**P1.14.**
- Every training record has ``step_time`` (the 0.5 s guard, :mod:`hsn.experiments.steptime`,
  fed by the step callback; an arm above its ``max_step_s`` raises ``StepTimeExceeded``) and
  ``protocol_version``.
- A train entry with a non-finite loss, accuracy or gradient norm keeps ``null`` there and its
  kind in ``nonfinite``.
- With ``diag_dense_every`` the diagnostics sample the arm's mixed schedule
  (:func:`hsn.experiments.rules.diag_schedule`).
- reads_state version 2 runs probe v2 after v1 (``probe_v2``, ``probes_v2``).
- On any exception the record built so far stays in ``ArmRun.record`` with a ``failure`` block
  (:func:`failure_info`), and the exception propagates unchanged:
  - after a raising fit: the evaluations, diagnostics, every completed step's train record, the
    early-stop log and the walls, read from the live ``History`` the callback holds;
  - otherwise what the earlier stages wrote.
  ``run-arm`` stores it.

**P1.15.**
- Every reads_state evaluation is labelled by query slot (:class:`hsn.experiments.split.QuerySlots`,
  the trainer's ``eval_position_groups``): each evaluation entry carries ``split`` (``first``,
  ``later``, ``later/1`` to ``later/3``: accuracy, loss, ``n``), checked against the frozen
  delayed/in_range metric, and ``final`` carries it flattened (``split/<g>/<key>``).
- The gradient-spike guard (``train.spike_factor``): ``final.skipped_spike_steps`` (``None`` with
  the guard off), ``skipped_steps`` (every step a guard skipped, with its reason), a ``skipped``
  mark on those train entries, and the failure kind ``spike`` when spike skips stopped the run.
- reads_state version 3 runs probe v3 after v1 and v2 (``probe_v3``, ``probes_v3``,
  ``probes_v3_originals``; :mod:`hsn.experiments.probe`).
- ``divergence``: the divergence dump's summary (``train.divergence_dump``, on by default: kind,
  step, norm, path; :mod:`hsn.train.divergence`), in full and partial records; ``None`` without a
  skipped step.

**P1.16.**
- The rollback (``train.rollback_*``, :mod:`hsn.train.rollback`): ``rollbacks`` (every event, in
  full and partial records), ``final.rollbacks`` (the count; ``None`` with the rollback off), the
  trigger steps marked ``skipped: rollback`` on their train entries, and the failure kind
  ``rollback`` at its limit.
- Kept checkpoints (protocol ``keep_every`` / ``keep_last``, execution-only): the trainer writes a
  checkpoint every ``keep_every`` steps (``Trainer.extra_ckpt_every``; the configuration is
  unchanged); after each step the newest ``keep_last`` of those stay, plus the newest one before the
  run's first guard event (a skipped or rolled-back step), which is never pruned; a run that
  finishes deletes them (its final checkpoint follows ``checkpoint``). When the fit fails on a guard
  (kinds ``nonfinite``, ``spike``, ``rollback``), the newest checkpoint on disk (kept or regular)
  before the first guard event (before the failure when there was none) is loaded into the model and
  evaluated (with the split), its state statistics and the protocol version's probes taken, and the
  result stored as ``partial``: ``from_step``, ``checkpoint``, ``reason``, ``final`` (the ``final``
  keys of that evaluation), ``trained``, ``probes`` (summaries) and ``seconds``; a failure while
  re-scoring is kept as ``partial.error`` and never hides the run's own failure. The rules label
  such a record ``failed: … (from step X: …)`` (:mod:`hsn.experiments.rules`). The same assessment
  of the newest saved checkpoint is also stored as ``latest_saved``; it is diagnostic only and
  changes neither the failure nor the conservative partial verdict.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hsn.config import RunConfig
from hsn.experiments.rules import RuleSet, check_early_stop, diag_schedule
from hsn.experiments.spec import (
    DryRunParams,
    OverfitParams,
    ReadsStateParams,
    SpikeDiagnosticParams,
    TrainEvalParams,
)
from hsn.experiments.steptime import StepTimeGuard

__all__ = ["ArmRun", "run_protocol"]

Params = ReadsStateParams | TrainEvalParams | OverfitParams | DryRunParams | SpikeDiagnosticParams


@dataclass
class ArmRun:
    """What a protocol needs to run one arm."""

    cfg: RunConfig
    family: str
    variant: str
    variant_params: dict[str, Any]
    params: Params
    run_dir: Path
    rules: RuleSet
    constants: dict[str, Any]
    on_progress: Callable[[dict[str, Any]], None] = lambda progress: None
    walls: dict[str, float] = field(default_factory=dict)
    # P1.14: the record so far (a protocol fills it as its stages complete) and the current stage;
    # after an exception it holds what exists, with a "failure" block, for the caller to store
    record: dict[str, Any] = field(default_factory=dict)
    stage: str = "setup"

    @property
    def steps(self) -> int:
        return self.params.steps or self.cfg.train.steps


def run_protocol(arm: ArmRun) -> dict[str, Any]:
    """The arm's record. On any exception (P1.14) ``arm.record`` keeps what exists so far with a
    ``failure`` block (:func:`failure_info`) and the exception propagates unchanged."""
    started = time.perf_counter()
    try:
        if isinstance(arm.params, DryRunParams):
            record = _dry_run(arm, arm.params)
        elif isinstance(arm.params, SpikeDiagnosticParams):
            from hsn.experiments.spike import run_diagnostic

            record = run_diagnostic(arm, arm.params)
        else:
            record = _train(arm, arm.params)
    except BaseException as exc:
        partial = arm.record
        partial.setdefault("protocol", arm.params.name)
        partial.setdefault("family", arm.family)
        if "failure" not in partial:
            partial["failure"] = failure_info(
                exc, stage=arm.stage, last_step=int((partial.get("final") or {}).get("steps") or 0)
            )
        arm.walls["total_s"] = time.perf_counter() - started
        partial["walls"] = {k: round(v, 3) for k, v in arm.walls.items()}
        raise
    arm.walls["total_s"] = time.perf_counter() - started
    record["walls"] = {k: round(v, 3) for k, v in arm.walls.items()}
    arm.record = record
    return record


# --------------------------------------------------------------------------------------------------
# Failures (P1.14)
# --------------------------------------------------------------------------------------------------

#: the P1.12 stop rule's messages end with this (hsn.train.loop.Trainer._guard_nonfinite)
_NONFINITE_STOP = "P1.12 stop rule"
#: P1.15: a stop rule message that counted spike skips names them
_SPIKE = "gradient spike"
#: P1.16: the rollback limit's message (hsn.train.loop.ROLLBACK_LIMIT; a literal here, so that the
#: queue side stays torch-free; test_p1_16_rollback_limit_stops checks the two agree)
_ROLLBACK_LIMIT = "P1.16 rollback limit"


def failure_info(
    exc: BaseException,
    *,
    stage: str,
    last_step: int,
    skipped_nonfinite: int | None = None,
    skipped_spike: int | None = None,
) -> dict[str, Any]:
    """The ``failure`` block of a partial record: ``kind`` (``nonfinite``: P1.12's stop rule;
    ``spike``: the same rule with P1.15's spike skips among the skipped steps; ``rollback``:
    P1.16's rollback limit (``train.rollback_max``); ``step_time``: the
    0.5 s rule; ``oom``; ``cancelled``: SIGTERM; ``interrupted``; ``error``), a short ``reason``
    (the verdicts read ``failed: <reason> @ <last_step>``), the error's type and message, the
    traceback's tail, the stage (``setup``, ``train``, ``trained_state``, ``probe``) and
    ``last_step``, the last completed optimiser step. Classifies by type name, so that the queue
    side stays torch-free."""
    import traceback

    name = type(exc).__name__
    message = str(exc)
    first = message.strip().splitlines()[0] if message.strip() else ""
    if name == "Cancelled":
        kind, reason = "cancelled", "cancelled (SIGTERM)"
    elif isinstance(exc, KeyboardInterrupt):
        kind, reason = "interrupted", "interrupted"
    elif name == "StepTimeExceeded":
        kind, reason = "step_time", first
    elif isinstance(exc, TimeoutError):
        kind, reason = "timeout", first
    elif name == "OutOfMemoryError":
        kind, reason = "oom", "out of memory"
    elif isinstance(exc, RuntimeError) and _ROLLBACK_LIMIT in message:  # P1.16
        kind, reason = "rollback", "kept diverging after rollbacks (P1.16 rollback limit)"
    elif isinstance(exc, RuntimeError) and _NONFINITE_STOP in message and _SPIKE in message:
        kind, reason = "spike", "gradient spikes or non-finite gradients (P1.12 stop)"  # P1.15
    elif isinstance(exc, RuntimeError) and _NONFINITE_STOP in message:
        kind, reason = "nonfinite", "non-finite gradients (P1.12 stop)"
    else:
        kind, reason = "error", f"{name}: {first}"[:160]
    if stage not in ("train", "setup") and kind == "error":
        reason = f"{stage}: {reason}"
    out: dict[str, Any] = {
        "kind": kind,
        "reason": reason,
        "error": f"{name}: {message}"[:2000],
        "traceback": "".join(traceback.format_exception(exc))[-4000:],
        "stage": stage,
        "last_step": int(last_step),
    }
    if skipped_nonfinite is not None:
        out["skipped_nonfinite"] = int(skipped_nonfinite)
    if skipped_spike is not None:  # P1.15: only with the spike guard on
        out["skipped_spike"] = int(skipped_spike)
    return out


# --------------------------------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------------------------------


class _EvalTimer:
    """A duck-typed trainer profiler that times the ``evaluate`` segment and nothing else."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self.count = 0

    @contextlib.contextmanager
    def segment(self, name: str, tokens: int = 0) -> Iterator[None]:
        if name != "evaluate":
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds += time.perf_counter() - started
            self.count += 1


def _eval_entry(e: Mapping[str, Any], *, audit_keys: bool) -> dict[str, Any]:
    groups = {
        g: {"accuracy": float(v["accuracy"]), "loss": float(v["loss"])}
        for g, v in (e.get("per_group") or {}).items()
    }
    out: dict[str, Any] = {
        "step": int(e["step"]),
        "accuracy": float(e["accuracy"]),
        "loss": float(e["loss"]),
        "per_group": groups,
        "n_windows": e.get("n_windows"),
    }
    if audit_keys:
        out["dev_answer_accuracy"] = {g: v["accuracy"] for g, v in groups.items()}
        out["dev_answer_loss"] = {g: v["loss"] for g, v in groups.items()}
    if e.get("per_position_group") is not None:  # P1.15: the first/later split
        from hsn.experiments.split import SPLIT_GROUP, split_block

        group = (e.get("per_group") or {}).get("/".join(SPLIT_GROUP)) or {}
        out["split"] = split_block(
            e["per_position_group"], group, total_positions=e.get("n_positions")
        )
    return out


def _train_entry(s: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "step": int(s.step),
        "loss": _finite(s.loss),
        "accuracy": _finite(s.accuracy),
        "grad_norm": _finite(s.grad_norm),
        "lr": float(s.lr),
    }
    # P1.14: a non-finite value stays null above (rules read the step as missing, as before) and
    # its kind (inf, -inf, nan) is kept here, so that a divergence can be read back
    nonfinite = {
        key: repr(float(getattr(s, key)))
        for key in ("loss", "accuracy", "grad_norm")
        if not math.isfinite(float(getattr(s, key)))
    }
    if nonfinite:
        out["nonfinite"] = nonfinite
    return out


def _mark_skipped(train: list[dict[str, Any]], trainer: Any) -> list[dict[str, Any]]:
    """P1.15: a train entry whose step a guard skipped carries ``skipped`` (``nonfinite`` or
    ``spike``); a spike step keeps its finite ``grad_norm``, so a rule can tell it apart."""
    reasons = {int(e["step"]): str(e["reason"]) for e in getattr(trainer, "skipped_steps", [])}
    for entry in train:
        reason = reasons.get(int(entry["step"]))
        if reason is not None:
            entry["skipped"] = reason
    return train


def _finite(x: float) -> float | None:
    return float(x) if math.isfinite(float(x)) else None


def _final(evals: list[dict[str, Any]], train: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"steps": train[-1]["step"] if train else 0}
    if train:
        out["train_loss"] = train[-1]["loss"]
    if evals:
        last = evals[-1]
        out["dev_accuracy"], out["dev_loss"], out["eval_step"] = (
            last["accuracy"],
            last["loss"],
            last["step"],
        )
        for g, v in last["per_group"].items():
            out[f"{g}/accuracy"], out[f"{g}/loss"] = v["accuracy"], v["loss"]
        for g, v in (last.get("split") or {}).items():  # P1.15
            for key in ("accuracy", "loss", "n"):
                out[f"split/{g}/{key}"] = v[key]
    return out


@dataclass
class _Keeper:
    """P1.16: prunes the kept checkpoints after each step (module docstring): the newest
    ``keep_last`` of those at multiples of ``keep_every`` that are not regular ``ckpt_every``
    checkpoints, and always the newest one before the first guard event."""

    keep_every: int
    keep_last: int
    ckpt_every: int
    keep_best: bool = False

    @staticmethod
    def on_disk(history: Any) -> list[tuple[int, Path]]:
        """``(step, path)`` of every checkpoint of the fit still on disk, oldest first."""
        out = set()
        for raw in history.checkpoints:
            path = Path(raw)
            if path.is_file():
                out.add((int(path.stem.split("_")[-1]), path))
        return sorted(out)

    def kept(self, history: Any) -> list[tuple[int, Path]]:
        """The checkpoints written for ``keep_every`` alone (not regular ``ckpt_every`` ones)."""
        return [
            (step, path)
            for step, path in self.on_disk(history)
            if step % self.keep_every == 0 and step % self.ckpt_every != 0
        ]

    @staticmethod
    def best(history: Any) -> dict[str, Any] | None:
        """Best dev accuracy among checkpoints evaluated at their exact saved step.

        Ties keep the earlier checkpoint. No score is inferred for unevaluated checkpoints.
        """
        on_disk = dict(_Keeper.on_disk(history))
        candidates = [
            e
            for e in history.evals
            if int(e["step"]) in on_disk and math.isfinite(float(e["accuracy"]))
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda e: (float(e["accuracy"]), -int(e["step"])))
        step = int(best["step"])
        return {"step": step, "accuracy": float(best["accuracy"]), "path": str(on_disk[step])}

    def prune(self, trainer: Any, history: Any) -> None:
        kept = self.kept(history)
        first = _first_guard_event(trainer)
        pinned = None
        if first is not None:
            before = [k for k in kept if k[0] < first]
            pinned = before[-1] if before else None
        best = self.best(history) if self.keep_best else None
        for entry in kept[: -self.keep_last]:
            if entry != pinned and (best is None or entry[0] != best["step"]):
                entry[1].unlink(missing_ok=True)


def _first_guard_event(trainer: Any) -> int | None:
    """P1.16: the first step a guard skipped or rolled back, or None."""
    steps = [int(s["step"]) for s in getattr(trainer, "skipped_steps", [])]
    return min(steps) if steps else None


class _StepCallback:
    """Progress at evaluations and controller observations; guard/stop rules at their cadence."""

    def __init__(
        self,
        arm: ArmRun,
        diagnostics: Any,
        *,
        audit_keys: bool,
        guard: StepTimeGuard | None = None,
    ) -> None:
        self.arm = arm
        self.diag = diagnostics
        self.audit_keys = audit_keys
        self.guard = guard
        self._compile_seen = 0.0
        self._controller_reported = -1
        self.at_steps = {s for r in arm.rules.early_stop for s in r.at_steps}
        self.stop: dict[str, Any] | None = None
        self.log: list[dict[str, Any]] = []
        self.started = time.perf_counter()
        self.history: Any = None  # P1.14: the live History, read back when the fit raises
        self.keeper: _Keeper | None = None  # P1.16: kept checkpoints (protocol keep_every)

    def __call__(self, trainer: Any, history: Any) -> bool:
        self.history = history
        step = int(trainer.step)
        if self.keeper is not None:
            self.keeper.prune(trainer, history)
        evaluated = bool(history.evals) and int(history.evals[-1]["step"]) == step
        if self.guard is not None and history.steps and int(history.steps[-1].step) == step:
            # a compile or CUDA-graph recording during this step or its evaluation (P2.8)
            window = getattr(trainer, "compiled_window", None)
            compiled_s = float(window.compile_seconds) if window is not None else 0.0
            compiled, self._compile_seen = compiled_s > self._compile_seen, compiled_s
            self.guard.observe(
                step, float(history.steps[-1].seconds), compiled=compiled, evaluated=evaluated
            )  # raises StepTimeExceeded above the arm's max_step_s
        controller = getattr(trainer, "adaptive_lr", None)
        changed = controller is not None and (
            len(controller.state.observations) != self._controller_reported
        )
        if evaluated or changed:
            self.publish_progress(trainer, history)
        elif (
            getattr(trainer, "update_limiter", None) is not None
            and step % trainer.cfg.train.stats_every == 0
        ):
            self.publish_progress(trainer, history)
        if step not in self.at_steps:
            return False
        record = {
            **_adaptive_record(trainer),
            "evals": [_eval_entry(e, audit_keys=False) for e in history.evals],
            "diagnostics": list(self.diag.records) if self.diag is not None else [],
            "train": [_train_entry(s) for s in history.steps],
        }
        stop, log = check_early_stop(
            self.arm.rules, record, step, self.arm.constants, self.arm.family, self.arm.params.name
        )
        self.log.extend(log)
        if stop is not None:
            self.stop = {**stop, "elapsed_s": round(time.perf_counter() - self.started, 1)}
            return True
        return False

    def publish_progress(self, trainer: Any, history: Any, *, phase: str | None = None) -> None:
        from hsn.experiments.controller import controller_progress

        progress: dict[str, Any] = {
            "step": int(trainer.step),
            "of": self.arm.steps,
            "elapsed_s": round(time.perf_counter() - self.started, 1),
            **controller_progress(trainer),
        }
        limiter = getattr(trainer, "update_limiter", None)
        if limiter is not None:
            progress["update_limit"] = limiter.state_dict()
        if history.evals:
            last = history.evals[-1]
            progress.update(dev_accuracy=float(last["accuracy"]), dev_loss=float(last["loss"]))
            if "adaptive_lr" in progress or "update_limit" in progress:
                # A monitoring update must not present an older dev score as a fresh evaluation.
                progress["dev_step"] = int(last["step"])
        if phase is not None:
            progress["phase"] = phase
        self.arm.on_progress(progress)
        if "adaptive_lr" in progress:
            self._controller_reported = int(progress["adaptive_lr"]["observation_count"])


def _adaptive_record(trainer: Any) -> dict[str, Any]:
    """Kept in successful and partial SQLite record_json, including guard/trial decisions."""
    controller = getattr(trainer, "adaptive_lr", None)
    out = {} if controller is None else {"adaptive_lr": controller.state_dict()}
    limiter = getattr(trainer, "update_limiter", None)
    if limiter is not None:
        out["update_limit"] = limiter.state_dict()
    return out


def _compile_info(trainer: Any, measured: Mapping[str, Any]) -> dict[str, Any]:
    """P2.8's public signals, read so that a checkout without P2.8 still works."""
    window = getattr(trainer, "compiled_window", None)
    return {
        "mode": trainer.cfg.hardware.compile_mode,
        "compiled_window": window is not None,
        "compiled_windows": measured.get("compiled_windows"),
        "compile_seconds": measured.get("compile_seconds"),
        "compile_steps": measured.get("compile_steps"),
        # every compiled variant's compile + recording time, evaluation included (P2.8, 0814235)
        "window_compile_seconds": measured.get("window_compile_seconds"),
    }


# --------------------------------------------------------------------------------------------------
# Training protocols
# --------------------------------------------------------------------------------------------------


def _train(
    arm: ArmRun, params: ReadsStateParams | TrainEvalParams | OverfitParams
) -> dict[str, Any]:
    from hsn.data.dataset import ByteDataset
    from hsn.device import resolve, seed_everything
    from hsn.experiments.data_selection import PREFIX, STRATIFIED, cap_dataset, dataset_counts
    from hsn.experiments.diagnostics import Diagnostics, trained_state
    from hsn.experiments.probe import (
        edit_probe,
        edit_probe_v2,
        edit_probe_v3,
        summarize,
        summarize_v2,
        summarize_v3,
    )
    from hsn.experiments.variants import install_variant
    from hsn.runner import build_model, load_datasets, make_trainer

    t0 = time.perf_counter()
    cfg = arm.cfg
    arm.stage = "setup"
    arm.record.update({"protocol": params.name, "family": arm.family})
    if arm.variant not in ("pdg", "none") and cfg.hardware.compile_mode != "none":
        # a variant patches the built model; a compiled path may not call what it patched (as
        # P1.10a's forward stopped calling what gru_cell_direct_input patched): never run one
        # compiled until it is shown to work there (spec expansion refuses it too)
        raise NotImplementedError(
            f"variant {arm.variant} with hardware.compile_mode={cfg.hardware.compile_mode}: "
            "variants run eager only (hardware.compile_mode=none)"
        )
    ctx = resolve(cfg.hardware)  # raises for a compile mode this checkout cannot run
    train_ds, dev_ds = load_datasets(cfg)
    full_train, full_dev = train_ds, dev_ds
    if isinstance(params, OverfitParams):
        train_ds = ByteDataset(list(train_ds.examples[: params.n_docs]))
        dev_ds = train_ds
        train_cap = dev_cap = params.n_docs
        full_dev = full_train  # overfit deliberately evaluates its own training documents
        data_policy = PREFIX
    else:
        train_cap, dev_cap = params.n_train, params.n_dev
        train_ds = cap_dataset(train_ds, train_cap, seed=cfg.seed)
        dev_ds = cap_dataset(dev_ds, dev_cap, seed=cfg.seed)
        data_policy = STRATIFIED
    data_selection = {
        "policy": data_policy,
        "seed": cfg.seed,
        "train": {
            "cap": train_cap,
            "source": "train",
            "full": dataset_counts(full_train),
            "effective": dataset_counts(train_ds),
        },
        "dev": {
            "cap": dev_cap,
            "source": "train" if isinstance(params, OverfitParams) else "dev",
            "full": dataset_counts(full_dev),
            "effective": dataset_counts(dev_ds),
        },
    }
    arm.record["data_selection"] = data_selection  # retained if setup or training fails
    seed_everything(cfg.seed)
    model = build_model(cfg)
    variant = install_variant(model, arm.variant, cfg.seed, arm.variant_params)  # before compile
    arm.record["variant"] = variant
    timer = _EvalTimer()
    trainer = make_trainer(model, cfg, ctx, run_dir=arm.run_dir, profiler=timer)
    trainer.eval_batch = params.eval_batch
    if params.keep_every:  # P1.16: kept checkpoints (a harness setting, not the configuration)
        trainer.extra_ckpt_every = params.keep_every
    # P1.15: what a divergence dump needs to rebuild this arm's model (hsn.train.divergence)
    trainer.dump_notes = {
        "variant": arm.variant,
        "variant_params": dict(arm.variant_params),
        "run_id": cfg.run_id,
        "protocol": params.name,
        "data_selection": {"policy": data_policy, "seed": cfg.seed},
    }
    if isinstance(params, ReadsStateParams):  # P1.15: the first/later split of every evaluation
        from hsn.experiments.split import QuerySlots

        trainer.eval_position_groups = QuerySlots(dev_ds)
    diag_every = params.diag_every
    # P1.14: the mixed cadence samples an explicit schedule; the single cadence keeps diag_every
    dense = params.diag_dense_every > 0
    schedule = diag_schedule(arm.steps, diag_every, params.diag_dense_every, params.diag_windows)
    diag = Diagnostics(
        model,
        cfg,
        ctx,
        dev_ds,
        max(diag_every, 1),
        n_docs=params.diag_window_docs,
        schedule=schedule if dense else None,
    )
    sampling = bool(schedule)
    if sampling:
        diag.attach(trainer)
    audit_keys = isinstance(params, ReadsStateParams)
    guard = StepTimeGuard(
        limit_s=params.max_step_s,
        approval=params.step_time_approval,
        diag_steps=frozenset(schedule),
        stats_every=cfg.train.stats_every,
    )
    callback = _StepCallback(arm, diag if sampling else None, audit_keys=audit_keys, guard=guard)
    if params.keep_every:
        callback.keeper = _Keeper(
            params.keep_every,
            params.keep_last,
            cfg.train.ckpt_every,
            keep_best=params.checkpoint == "retained",
        )
    arm.walls["setup_s"] = time.perf_counter() - t0

    arm.stage = "train"
    t1 = time.perf_counter()
    try:
        history = trainer.fit(train_ds, dev_ds, arm.steps, callback=callback)
    except BaseException as exc:
        # P1.14: what the fit produced before it raised (the P1.12 stop, OOM, SIGTERM, ...)
        arm.walls["fit_s"] = time.perf_counter() - t1
        diag.detach()
        _partial_training(arm, params, trainer, callback, diag, timer, audit_keys)
        spike_on = getattr(trainer, "spike_factor", None) is not None
        arm.record["failure"] = failure_info(
            exc,
            stage="train",
            last_step=int(trainer.step),
            skipped_nonfinite=int(getattr(trainer, "skipped_nonfinite", 0)),
            skipped_spike=int(getattr(trainer, "skipped_spike", 0)) if spike_on else None,
        )
        if callback.keeper is not None and arm.record["failure"]["kind"] in _RESCORED:
            arm.record["partial"] = _rescore_partial(
                arm, params, trainer, model, cfg, ctx, dev_ds, callback, audit_keys
            )
            arm.record["latest_saved"] = _rescore_partial(
                arm,
                params,
                trainer,
                model,
                cfg,
                ctx,
                dev_ds,
                callback,
                audit_keys,
                latest=True,
            )
        raise
    arm.walls["fit_s"] = time.perf_counter() - t1
    diag.detach()
    measured = dict(history.measured)
    arm.walls["train_s"] = float(measured.get("train_seconds", 0.0)) - diag.seconds
    arm.walls["compile_s"] = float(measured.get("compile_seconds") or 0.0)
    # all compiled variants (training, statistics and evaluation windows); eval_s includes the
    # evaluation variant's one-off compile, so steady-state evaluation is eval_s minus its share
    arm.walls["window_compile_s"] = float(measured.get("window_compile_seconds") or 0.0)
    arm.walls["eval_s"] = timer.seconds
    arm.walls["diag_s"] = diag.seconds

    evals = [_eval_entry(e, audit_keys=audit_keys) for e in history.evals]
    train = _mark_skipped([_train_entry(s) for s in history.steps], trainer)
    if evals:
        callback.publish_progress(
            trainer, history, phase="stopped, probing" if callback.stop else "trained, probing"
        )
    full: dict[str, Any] = {
        "protocol": params.name,
        "data_selection": data_selection,
        "protocol_version": getattr(params, "version", 1),  # P1.14
        "family": arm.family,
        "variant": variant,
        "autocast": history.autocast,
        "param_count": measured.get("param_count"),
        "param_report": history.param_report,
        "measured": measured,
        "compile": _compile_info(trainer, measured),
        "evals": evals,
        "diagnostics": diag.records,
        "train": train,
        "final": {
            **_final(evals, train),
            # P1.12's non-finite-gradient guard (agreed 7 Nov): None on a checkout without it, so a
            # rule on final:skipped_nonfinite_steps reads missing, never a false 0
            "skipped_nonfinite_steps": measured.get("skipped_nonfinite_steps"),
            # P1.15: the spike guard's skips; None with the guard off (the guard did not look)
            "skipped_spike_steps": measured.get("skipped_spike_steps"),
            "rollbacks": measured.get("rollbacks"),  # P1.16: None with the rollback off
        },
        "skipped_steps": list(getattr(trainer, "skipped_steps", [])),  # P1.15
        **_adaptive_record(trainer),
        "rollbacks": [dict(r) for r in getattr(trainer, "rollbacks", [])],  # P1.16
        "divergence": getattr(trainer, "divergence", None),  # P1.15: the dump, when written
        "divergences": getattr(trainer, "divergences", []),
        "stopped": callback.stop,
        "early_stop_log": callback.log,
        "eval_count": timer.count,
        "checkpoints": history.checkpoints,
        "step_time": guard.summary(),  # P1.14
    }
    record = arm.record  # P1.14: filled in place, so that a later stage's failure keeps it
    record.clear()
    record.update(full)
    if isinstance(params, OverfitParams):
        record["overfit"] = _overfit_summary(evals, history, params)
    if isinstance(params, (ReadsStateParams, TrainEvalParams)):
        arm.stage = "trained_state"
        t2 = time.perf_counter()
        state = trained_state(model, cfg, ctx, dev_ds, params.diag_docs)
        arm.walls["trained_s"] = time.perf_counter() - t2
        probes: list[dict[str, Any]] | None = None
        summary: dict[str, Any] | None = None
        v2: dict[str, Any] = {}  # P1.14: probe v2 (reads_state version 2 only)
        if isinstance(params, ReadsStateParams):
            arm.stage = "probe"
            t3 = time.perf_counter()
            probes = edit_probe(model, ctx, dev_ds, batch_rows=params.probe_batch)
            summary = summarize(probes)
            arm.walls["probe_s"] = time.perf_counter() - t3
            if params.version >= 2:
                arm.stage = "probe_v2"
                t4 = time.perf_counter()
                probes_v2 = edit_probe_v2(
                    model, ctx, dev_ds, v1=probes, batch_rows=params.probe_batch
                )
                v2 = {"probe_v2": summarize_v2(probes_v2, v1=probes), "probes_v2": probes_v2}
                arm.walls["probe_v2_s"] = time.perf_counter() - t4
            if params.version >= 3:  # P1.15: probe v3, the later queries' key-addressed edits
                arm.stage = "probe_v3"
                t5 = time.perf_counter()
                probes_v3, originals_v3 = edit_probe_v3(
                    model, ctx, dev_ds, batch_rows=params.probe_batch
                )
                v2["probe_v3"] = summarize_v3(probes_v3, originals_v3)
                v2["probes_v3"] = probes_v3
                v2["probes_v3_originals"] = originals_v3
                arm.walls["probe_v3_s"] = time.perf_counter() - t5
        if callback.stop is not None:
            record["at_stop"] = {
                "step": callback.stop["step"],
                "trained_state": state,
                "probe": summary,
                "probes": probes,
                **v2,
            }
            record["trained_state"], record["probe"], record["probes"] = None, None, None
            record.update(dict.fromkeys(v2))
        else:
            record["trained_state"], record["probe"], record["probes"] = state, summary, probes
            record.update(v2)
    if params.checkpoint == "none":
        for path in history.checkpoints:
            Path(path).unlink(missing_ok=True)
    elif params.checkpoint == "retained":
        record["retained_checkpoints"] = [
            {"step": step, "path": str(path)} for step, path in _Keeper.on_disk(history)
        ]
        record["best_checkpoint"] = _Keeper.best(history)
    elif callback.keeper is not None:  # legacy final-only policy
        for _step, kept in callback.keeper.kept(history):
            kept.unlink(missing_ok=True)
    arm.stage = "done"
    return record


def _partial_training(
    arm: ArmRun,
    params: ReadsStateParams | TrainEvalParams | OverfitParams,
    trainer: Any,
    callback: _StepCallback,
    diag: Any,
    timer: _EvalTimer,
    audit_keys: bool,
) -> None:
    """P1.14: the record of a fit that raised, from the live ``History`` the callback was handed
    (``None`` when it raised before its first step completed): evaluations, diagnostics, every
    completed step's train record, the early-stop log, the checkpoints and the walls so far. The
    trainer's ``measured`` block is written only at the end of a fit, so it is not there."""
    history = callback.history
    evals = [_eval_entry(e, audit_keys=audit_keys) for e in (history.evals if history else [])]
    train = _mark_skipped([_train_entry(s) for s in (history.steps if history else [])], trainer)
    skipped = float(getattr(trainer, "skipped_nonfinite", 0))
    spikes = (  # P1.15: None with the guard off
        float(getattr(trainer, "skipped_spike", 0))
        if getattr(trainer, "spike_factor", None) is not None
        else None
    )
    measured: dict[str, Any] = {"skipped_nonfinite_steps": skipped}
    if spikes is not None:
        measured["skipped_spike_steps"] = spikes
    rollbacks = (  # P1.16: None with the rollback off
        float(len(getattr(trainer, "rollbacks", [])))
        if getattr(trainer, "rollback", None) is not None
        else None
    )
    if rollbacks is not None:
        measured["rollbacks"] = rollbacks
    window = getattr(trainer, "compiled_window", None)
    compiled = {"window_compile_seconds": window.compile_seconds} if window is not None else {}
    arm.walls["eval_s"] = timer.seconds
    arm.walls["diag_s"] = diag.seconds
    arm.walls["train_s"] = sum(float(s.seconds) for s in (history.steps if history else []))
    arm.walls["train_s"] -= diag.seconds
    arm.record.update(
        {
            "protocol": params.name,
            "family": arm.family,
            "autocast": history.autocast if history else None,
            "param_report": history.param_report if history else None,
            "measured": measured,
            "compile": _compile_info(trainer, compiled),
            "evals": evals,
            "diagnostics": list(diag.records),
            "train": train,
            "final": {
                **_final(evals, train),
                "skipped_nonfinite_steps": skipped,
                "skipped_spike_steps": spikes,
                "rollbacks": rollbacks,  # P1.16
            },
            "skipped_steps": list(getattr(trainer, "skipped_steps", [])),  # P1.15
            **_adaptive_record(trainer),
            "rollbacks": [dict(r) for r in getattr(trainer, "rollbacks", [])],  # P1.16
            "divergence": getattr(trainer, "divergence", None),  # P1.15: the dump, when written
            "divergences": getattr(trainer, "divergences", []),
            "stopped": None,
            "early_stop_log": list(callback.log),
            "eval_count": timer.count,
            "checkpoints": list(history.checkpoints) if history else [],
            "step_time": callback.guard.summary() if callback.guard is not None else None,
        }
    )


#: P1.16: the failure kinds after which a kept checkpoint is re-scored (the guards' stops)
_RESCORED = ("nonfinite", "spike", "rollback")


def _rescore_partial(
    arm: ArmRun,
    params: ReadsStateParams | TrainEvalParams | OverfitParams,
    trainer: Any,
    model: Any,
    cfg: RunConfig,
    ctx: Any,
    dev_ds: Any,
    callback: _StepCallback,
    audit_keys: bool,
    *,
    latest: bool = False,
) -> dict[str, Any] | None:
    """P1.16: the ``partial`` block of a run that failed on a guard (module docstring), or None
    when no kept checkpoint qualifies. Errors while re-scoring are kept in it, never raised."""
    from hsn.experiments.diagnostics import trained_state
    from hsn.experiments.probe import (
        edit_probe,
        edit_probe_v2,
        edit_probe_v3,
        summarize,
        summarize_v2,
        summarize_v3,
    )
    from hsn.train.checkpoint import load_checkpoint

    history = callback.history
    keeper = callback.keeper
    if history is None or keeper is None:
        return None
    first = _first_guard_event(trainer)
    limit = first if first is not None else int(trainer.step) + 1
    candidates = [k for k in keeper.on_disk(history) if latest or k[0] < limit]
    if not candidates:
        return None
    step, path = candidates[-1]
    reason = (
        f"the newest kept checkpoint before the first guard event (step {first})"
        if first is not None
        else f"the newest kept checkpoint before the failure (step {int(trainer.step)})"
    )
    if latest:
        reason = "the newest saved checkpoint; diagnostic only, not a recovered successful run"
    out: dict[str, Any] = {"from_step": step, "checkpoint": str(path), "reason": reason}
    t0 = time.perf_counter()
    arm.stage = "partial"
    try:
        load_checkpoint(path, model)  # weights only, in place (the compiled window keeps them)
        result = trainer.evaluate(dev_ds)
        entry = _eval_entry({"step": step, **result}, audit_keys=audit_keys)
        train = [_train_entry(s) for s in history.steps if int(s.step) <= step]
        out["final"] = _final([entry], train)
        if isinstance(params, (ReadsStateParams, TrainEvalParams)):
            out["trained"] = trained_state(model, cfg, ctx, dev_ds, params.diag_docs)
        probes: dict[str, Any] = {}
        if isinstance(params, ReadsStateParams):
            v1 = edit_probe(model, ctx, dev_ds, batch_rows=params.probe_batch)
            probes["probe"] = summarize(v1)
            if params.version >= 2:
                v2 = edit_probe_v2(model, ctx, dev_ds, v1=v1, batch_rows=params.probe_batch)
                probes["probe_v2"] = summarize_v2(v2, v1=v1)
            if params.version >= 3:
                v3, originals = edit_probe_v3(model, ctx, dev_ds, batch_rows=params.probe_batch)
                probes["probe_v3"] = summarize_v3(v3, originals)
        out["probes"] = probes
    except Exception as exc:  # a diagnostic of a failed run: never hide the run's own failure
        out["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    arm.stage = "train"
    out["seconds"] = round(time.perf_counter() - t0, 2)
    return out


def _overfit_summary(
    evals: list[dict[str, Any]], history: Any, params: OverfitParams
) -> dict[str, Any]:
    out: dict[str, Any] = {
        f"first_ge_{t:.2f}": next((e["step"] for e in evals if e["accuracy"] >= t), None)
        for t in params.thresholds
    }
    seconds = sorted(s.seconds for s in history.steps)
    out["step_seconds_p50"] = seconds[len(seconds) // 2] if seconds else None
    out["final_train_accuracy"] = evals[-1]["accuracy"] if evals else None
    return out


# --------------------------------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------------------------------


def _dry_run(arm: ArmRun, params: DryRunParams) -> dict[str, Any]:
    t0 = time.perf_counter()
    arm.stage = "train"
    if params.sleep_s:
        time.sleep(params.sleep_s)
    if params.fail:
        raise RuntimeError("dry_run: failing as asked (fail: true)")
    steps, every = arm.steps, arm.cfg.train.eval_every
    eval_steps = sorted({*range(every, steps + 1, every), steps})
    callback_rules = arm.rules
    evals: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "protocol": "dry_run",
        "family": arm.family,
        "variant": {"variant": arm.variant},
    }
    stop: dict[str, Any] | None = None
    log: list[dict[str, Any]] = []
    at_steps = {s for r in callback_rules.early_stop for s in r.at_steps}
    for step in eval_steps:
        evals.append(
            {
                "step": step,
                "accuracy": params.accuracy,
                "loss": -math.log(max(params.accuracy, 1e-6)),
                "per_group": {"delayed/in_range": {"accuracy": params.accuracy, "loss": 1.0}},
            }
        )
        arm.on_progress({"step": step, "of": steps, "dev_accuracy": params.accuracy})
        if step in at_steps and step < steps:
            stop, entries = check_early_stop(
                callback_rules, {"evals": evals}, step, arm.constants, arm.family, "dry_run"
            )
            log.extend(entries)
            if stop is not None:
                break
    train = [
        {"step": s, "loss": 1.0, "accuracy": params.accuracy, "grad_norm": 1.0, "lr": 0.0}
        for s in range(1, (stop["step"] if stop else steps) + 1)
    ]
    record.update(
        {
            "evals": evals,
            "diagnostics": [],
            "train": train,
            "final": _final(evals, train),
            "stopped": stop,
            "early_stop_log": log,
        }
    )
    record.update(params.record)
    arm.walls["setup_s"] = time.perf_counter() - t0
    if stop is not None:
        record["at_stop"] = {"step": stop["step"], "probe": record.pop("probe", None)}
    record.setdefault("probe", None)
    return record
