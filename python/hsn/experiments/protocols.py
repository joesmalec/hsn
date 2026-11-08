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
from hsn.experiments.rules import RuleSet, check_early_stop
from hsn.experiments.spec import (
    DryRunParams,
    OverfitParams,
    ReadsStateParams,
    TrainEvalParams,
)

__all__ = ["ArmRun", "run_protocol"]

Params = ReadsStateParams | TrainEvalParams | OverfitParams | DryRunParams


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


def failure_info(
    exc: BaseException,
    *,
    stage: str,
    last_step: int,
    skipped_nonfinite: int | None = None,
) -> dict[str, Any]:
    """The ``failure`` block of a partial record: ``kind`` (``nonfinite``: P1.12's stop rule;
    ``step_time``: the 0.5 s rule; ``oom``; ``cancelled``: SIGTERM; ``interrupted``; ``error``),
    a short ``reason`` (the verdicts read ``failed: <reason> @ <last_step>``), the error's type and
    message, the traceback's tail, the stage (``setup``, ``train``, ``trained_state``, ``probe``)
    and ``last_step``, the last completed optimiser step. Classifies by type name, so that the
    queue side stays torch-free."""
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
    elif name == "OutOfMemoryError":
        kind, reason = "oom", "out of memory"
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
    return out


def failure_label(failure: Mapping[str, Any]) -> str:
    """The verdict label of a failed arm (P1.14): never a case label, never a finding."""
    return f"failed: {failure.get('reason')} @ {failure.get('last_step')}"


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
    return out


class _StepCallback:
    """Progress at every evaluation; the early-stop rules at their steps."""

    def __init__(self, arm: ArmRun, diagnostics: Any, *, audit_keys: bool) -> None:
        self.arm = arm
        self.diag = diagnostics
        self.audit_keys = audit_keys
        self.at_steps = {s for r in arm.rules.early_stop for s in r.at_steps}
        self.stop: dict[str, Any] | None = None
        self.log: list[dict[str, Any]] = []
        self.started = time.perf_counter()
        self.history: Any = None  # P1.14: the live History, read back when the fit raises

    def __call__(self, trainer: Any, history: Any) -> bool:
        self.history = history
        step = int(trainer.step)
        if history.evals and int(history.evals[-1]["step"]) == step:
            last = history.evals[-1]
            self.arm.on_progress(
                {
                    "step": step,
                    "of": self.arm.steps,
                    "dev_accuracy": float(last["accuracy"]),
                    "dev_loss": float(last["loss"]),
                    "elapsed_s": round(time.perf_counter() - self.started, 1),
                }
            )
        if step not in self.at_steps:
            return False
        record = {
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
    from hsn.experiments.diagnostics import Diagnostics, trained_state
    from hsn.experiments.probe import edit_probe, summarize
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
    if isinstance(params, OverfitParams):
        train_ds = ByteDataset(list(train_ds.examples[: params.n_docs]))
        dev_ds = train_ds
    else:
        if params.n_train:
            train_ds = ByteDataset(list(train_ds.examples[: params.n_train]))
        if params.n_dev:
            dev_ds = ByteDataset(list(dev_ds.examples[: params.n_dev]))
    seed_everything(cfg.seed)
    model = build_model(cfg)
    variant = install_variant(model, arm.variant, cfg.seed, arm.variant_params)  # before compile
    arm.record["variant"] = variant
    timer = _EvalTimer()
    trainer = make_trainer(model, cfg, ctx, run_dir=arm.run_dir, profiler=timer)
    trainer.eval_batch = params.eval_batch
    diag_every = params.diag_every
    diag = Diagnostics(model, cfg, ctx, dev_ds, max(diag_every, 1), n_docs=params.diag_window_docs)
    if diag_every:
        diag.attach(trainer)
    audit_keys = isinstance(params, ReadsStateParams)
    callback = _StepCallback(arm, diag if diag_every else None, audit_keys=audit_keys)
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
        arm.record["failure"] = failure_info(
            exc,
            stage="train",
            last_step=int(trainer.step),
            skipped_nonfinite=int(getattr(trainer, "skipped_nonfinite", 0)),
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
    train = [_train_entry(s) for s in history.steps]
    if evals:  # the trainer does not call back at its last step: report it, and the phase after
        arm.on_progress(
            {
                "step": int(history.steps[-1].step),
                "of": arm.steps,
                "dev_accuracy": evals[-1]["accuracy"],
                "dev_loss": evals[-1]["loss"],
                "phase": "stopped, probing" if callback.stop else "trained, probing",
                "elapsed_s": round(time.perf_counter() - t1, 1),
            }
        )
    full: dict[str, Any] = {
        "protocol": params.name,
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
        },
        "stopped": callback.stop,
        "early_stop_log": callback.log,
        "eval_count": timer.count,
        "checkpoints": history.checkpoints,
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
        if isinstance(params, ReadsStateParams):
            arm.stage = "probe"
            t3 = time.perf_counter()
            probes = edit_probe(model, ctx, dev_ds, batch_rows=params.probe_batch)
            summary = summarize(probes)
            arm.walls["probe_s"] = time.perf_counter() - t3
        if callback.stop is not None:
            record["at_stop"] = {
                "step": callback.stop["step"],
                "trained_state": state,
                "probe": summary,
                "probes": probes,
            }
            record["trained_state"], record["probe"], record["probes"] = None, None, None
        else:
            record["trained_state"], record["probe"], record["probes"] = state, summary, probes
    if params.checkpoint == "none":
        for path in history.checkpoints:
            Path(path).unlink(missing_ok=True)
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
    train = [_train_entry(s) for s in (history.steps if history else [])]
    skipped = float(getattr(trainer, "skipped_nonfinite", 0))
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
            "measured": {"skipped_nonfinite_steps": skipped},
            "compile": _compile_info(trainer, compiled),
            "evals": evals,
            "diagnostics": list(diag.records),
            "train": train,
            "final": {**_final(evals, train), "skipped_nonfinite_steps": skipped},
            "stopped": None,
            "early_stop_log": list(callback.log),
            "eval_count": timer.count,
            "checkpoints": list(history.checkpoints) if history else [],
        }
    )


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
