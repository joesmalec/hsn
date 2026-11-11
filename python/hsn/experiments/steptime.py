"""The step-time guard (P1.14): the user's rule of 7 Nov 2026, "no experiment whose training step
takes longer than 0.5 s". Imports without torch.

**At enqueue** (:func:`enqueue_refusal`, called by :func:`hsn.experiments.db.enqueue_arms` for every
arm it would add; an arm skipped as already queued, running or done is never refused). An arm is
refused, and nothing of the spec is queued, when:

* its expected step time is *known* to exceed its limit ``max_step_s``: the median recorded step
  time of the earlier runs with its memory key (``Database.step_time_estimate``) is above it. This
  refuses with an approval too, since the arm would only be stopped at run time;
* without a ``step_time_approval`` note (who approved, when), when ``max_step_s`` is above
  :data:`STEP_TIME_LIMIT_S`, or when it is an HSN arm on CUDA with ``hardware.compile_mode=none``
  (eager HSN takes 0.73 s dense to 1.7 s top-K per step at B32; the variants are eager by design).

GRU and Transformer arms are never refused for being eager (they take 0.004 to 0.017 s); dry runs
train nothing and are not checked.

**At run time** (:class:`StepTimeGuard`, fed by the protocols' step callback). The first
:data:`STEP_TIME_SKIP` steps are warm-up; compile steps (the compiled window's ``compile_seconds``
grew, the evaluation variant's compile and the CUDA-graph recording included), evaluation steps and
the step after each, diagnostics steps and the trainer's statistics steps never count. Once
:data:`STEP_TIME_WINDOW` eligible steps exist, their median is compared with ``max_step_s``, once;
above it :class:`StepTimeExceeded` is raised and the arm fails early with its partial record
(verdicts ``failed: step time ...``). :meth:`StepTimeGuard.summary` is the record's ``step_time``
block; its ``run_median_s`` (every eligible step of the run) is the enqueue estimate of later arms.
The constants are module constants (as P1.12's stop rule), not keys.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "STEP_TIME_LIMIT_S",
    "STEP_TIME_SKIP",
    "STEP_TIME_WINDOW",
    "StepTimeExceeded",
    "StepTimeGuard",
    "StepTimeRefused",
    "enqueue_refusal",
    "step_time_estimate",
]

#: The user's rule (7 Nov 2026): the default ``max_step_s`` of every protocol.
STEP_TIME_LIMIT_S = 0.5
#: Steps at the start that never count (allocator, autotuning, CUDA-graph warm-up).
STEP_TIME_SKIP = 20
#: Eligible steps whose median is compared with the limit, once.
STEP_TIME_WINDOW = 50

EAGER_HSN = (
    "an HSN arm on CUDA with hardware.compile_mode=none trains eager: 0.73 s (dense) to 1.7 s "
    "(top-K) per step at B32, above the 0.5 s rule"
)


class StepTimeExceeded(RuntimeError):
    """An arm's median step time is above its ``max_step_s`` (the run-time guard)."""


class StepTimeRefused(ValueError):
    """``enqueue`` refused arms under the step-time rule (nothing was queued)."""


@dataclass
class StepTimeGuard:
    """The run-time guard of one arm (module docstring). ``diag_steps``: the steps the diagnostics
    sample; ``stats_every``: the trainer's statistics cadence (``train.stats_every``)."""

    limit_s: float
    approval: str = ""
    diag_steps: Collection[int] = ()
    stats_every: int | None = None
    skip: int = field(default_factory=lambda: STEP_TIME_SKIP)
    window: int = field(default_factory=lambda: STEP_TIME_WINDOW)
    eligible: list[tuple[int, float]] = field(default_factory=list)
    excluded: Counter[str] = field(default_factory=Counter)
    check: dict[str, Any] | None = None
    _evaluated: set[int] = field(default_factory=set)

    def _exclusion(self, step: int, compiled: bool, evaluated: bool) -> str | None:
        if step <= self.skip:
            return "warmup"
        if compiled:
            return "compile"
        if evaluated:
            return "eval"
        if step - 1 in self._evaluated:
            return "after_eval"
        if step in self.diag_steps:
            return "diag"
        if self.stats_every and step % self.stats_every == 0:
            return "stats"
        return None

    def observe(self, step: int, seconds: float, *, compiled: bool, evaluated: bool) -> None:
        """One completed step: its wall seconds, whether a window compiled or recorded during it
        (or during its evaluation), whether it was an evaluation step. Raises
        :class:`StepTimeExceeded` when the window's median is above the limit."""
        why = self._exclusion(step, compiled, evaluated)
        if evaluated:
            self._evaluated.add(step)
        if why is not None:
            self.excluded[why] += 1
            return
        self.eligible.append((step, float(seconds)))
        if self.check is None and len(self.eligible) >= self.window:
            steps = self.eligible[: self.window]
            median = statistics.median(s for _, s in steps)
            self.check = {
                "median_s": median,
                "n": len(steps),
                "first": steps[0][0],
                "last": steps[-1][0],
                "passed": median <= self.limit_s,
            }
            if median > self.limit_s:
                raise StepTimeExceeded(
                    f"step time {median:.3g} s > {self.limit_s:g} s (median of {len(steps)} "
                    f"steps, {steps[0][0]}-{steps[-1][0]})"
                )

    def summary(self) -> dict[str, Any]:
        """The record's ``step_time`` block."""
        times = [s for _, s in self.eligible]
        return {
            "limit_s": self.limit_s,
            "approval": self.approval,
            "skip": self.skip,
            "window": self.window,
            "checked": self.check is not None,
            "check": self.check,
            "run_median_s": statistics.median(times) if times else None,
            "run_n": len(times),
            "excluded": dict(sorted(self.excluded.items())),
        }


def step_time_estimate(rows: Collection[dict[str, Any]]) -> tuple[float, int] | None:
    """The median step time of earlier runs, from rows with ``run_median_s`` (a P1.14 record's
    ``step_time.run_median_s``) or, for records before P1.14, ``train_s`` / ``steps`` /
    ``compile_steps`` (``walls.train_s / (final.steps - measured.compile_steps)``; a failed
    attempt's partial walls are never used that way). ``(median, runs)``, or None."""
    values = []
    for row in rows:
        median = row.get("run_median_s")
        if median is not None:
            values.append(float(median))
            continue
        train_s, steps = row.get("train_s"), row.get("steps")
        if row.get("failed") or train_s is None or not steps:
            continue
        timed = float(steps) - float(row.get("compile_steps") or 0.0)
        if timed > 0 and float(train_s) > 0:
            values.append(float(train_s) / timed)
    if not values:
        return None
    return statistics.median(values), len(values)


def enqueue_refusal(arm: Any, estimate: tuple[float, int] | None) -> str | None:
    """Why the expanded arm ``arm`` (:class:`hsn.experiments.spec.Arm`) may not be queued under the
    step-time rule (module docstring), or None."""
    protocol = arm.protocol
    if protocol.name in ("dry_run", "spike_diagnostic"):
        return None
    limit = float(protocol.max_step_s)
    approval = str(protocol.step_time_approval).strip()
    if estimate is not None and estimate[0] > limit:
        return (
            f"{arm.label}: its expected step time is {estimate[0]:.3g} s (median of {estimate[1]} "
            f"earlier runs with its memory key), above max_step_s {limit:g}: the run-time guard "
            "would stop it; raise max_step_s (with step_time_approval) or change the arm"
        )
    if approval:
        return None
    if limit > STEP_TIME_LIMIT_S:
        return (
            f"{arm.label}: max_step_s {limit:g} is above the {STEP_TIME_LIMIT_S:g} s rule: it "
            'needs step_time_approval: "<who approved, when>" in its protocol'
        )
    hardware = arm.config.hardware
    if arm.family == "hsn" and hardware.device == "cuda" and hardware.compile_mode == "none":
        return (
            f"{arm.label}: {EAGER_HSN}; it needs step_time_approval: "
            '"<who approved, when>" and a max_step_s above its step time in its protocol'
        )
    return None
