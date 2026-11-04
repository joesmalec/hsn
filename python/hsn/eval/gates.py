"""Gate decision functions: efficiency (G2), retention (G3) and composition (G4).

P0.7. Each function applies the thresholds of plan §2.9 and §8.1 (paper §12–§13) to measured
numbers and returns a :class:`GateResult`: ``passed``, one ``reasons`` line per failed criterion
(empty on pass) and a flat ``numbers`` table with every quantity and threshold that entered the
decision, so a gate script can print ``PASS | FAIL`` and store the table in its evidence package.

Units: accuracies, gains, losses and their thresholds are fractions (``0.01`` = 1 pp); relative
changes of latency, energy and perplexity are fractions of the baseline (``0.20`` = 20 %).
``numbers`` holds fractions only; percentage points and percent appear only in the ``reasons``
text (chunk P0.7 pitfall). Thresholds given in percentage points or percent (``5``, ``20``) are
rejected, since they would make a criterion impossible to fail. ``configs/metrics_freeze.json``
(P0.8) must therefore store the gate thresholds as fractions. Thresholds are parameters with the
paper's defaults, so the frozen file is the only other place they live; they may be tightened
before an evaluation, and relaxing one after seeing results requires a new test set and an
explicit revised claim (global rule 3).

Threshold comparisons are inclusive, as the plan states them ("≤ 2 pp", "no task loses more than
5 pp", "≥ 3 pp", "≥ 20 %"), with the tolerance :data:`TOLERANCE` so that a difference of two
accuracies that lands exactly on a bar (``0.90 − 0.85`` is ``0.050000000000000044`` in binary64)
is decided as the plan reads, not by rounding. Confidence-interval checks ("excluding no
improvement", "excluding zero") are strict.

Statistics come from :mod:`hsn.eval.bootstrap`: percentile bootstrap over units (tasks or
repositories × seed, never tokens), paired by seed. Input validation lives in
:mod:`hsn.eval.metrics` and :mod:`hsn.eval.bootstrap`; this module holds decision branches only.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from hsn.eval.bootstrap import PairedSamples, bootstrap_mean
from hsn.eval.metrics import (
    average_final_accuracy,
    matrix_stack,
    new_task_gain,
    per_task_loss,
    require_difference,
    require_fraction,
    require_positive,
    require_threshold,
)

__all__ = [
    "TOLERANCE",
    "Claim",
    "GateResult",
    "QualityKind",
    "composition_gate",
    "efficiency_gate",
    "retention_gate",
]

QualityKind = Literal["accuracy", "perplexity"]
Claim = Literal["latency", "energy"]

#: Tolerance of the inclusive threshold comparisons (module docstring): seven orders of magnitude
#: below one percentage point and far below the ``1/N`` resolution of any measured accuracy.
TOLERANCE: float = 1e-9

_OTHER_OF: dict[str, str] = {"latency": "energy", "energy": "latency"}


@dataclass
class GateResult:
    """Outcome of a gate decision.

    ``passed`` is true when ``reasons`` is empty; ``reasons`` holds one line per failed criterion
    with the value, the threshold and the unit written out; ``numbers`` is a flat table of fractions
    (and counts) covering every quantity and threshold the decision used, for the evidence package.
    """

    passed: bool
    reasons: list[str]
    numbers: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _pp(fraction: float) -> str:
    """A fraction written in percentage points, for reason text only."""
    return f"{100.0 * fraction:.2f} pp"


def _pct(fraction: float) -> str:
    """A relative change written in percent, for reason text only."""
    return f"{100.0 * fraction:.2f} %"


def _ci_text(low: float, high: float, alpha: float) -> str:
    return f"{100.0 * (1.0 - alpha):.0f} % CI [{_pct(low)}, {_pct(high)}]"


# --------------------------------------------------------------------------------------------------
# G2 — efficiency gate (plan §2.9; paper §12 statistical plan; §13 G2)
# --------------------------------------------------------------------------------------------------


def efficiency_gate(
    quality: PairedSamples,
    speed: PairedSamples,
    quality_kind: QualityKind,
    claim: Claim,
    other: PairedSamples | None,
    *,
    accuracy_loss_max: float = 0.01,
    perplexity_loss_max: float = 0.02,
    improvement_min: float = 0.20,
    other_regression_max: float | None = None,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> GateResult:
    """The G2 efficiency gate of plan §2.9 (paper §12): matched quality and a measured win.

    Plan §2.9: "Quality loss ≤ 1 percentage point accuracy (or ≤ 2 % relative perplexity) **and**
    measured latency *or* energy improvement ≥ 20 % with a 95 % paired confidence interval excluding
    no improvement. Do not claim both latency and energy unless both pass; the other must meet its
    non-regression budget." With ``b_u`` the baseline and ``c_u`` the candidate for unit ``u``::

        accuracy:    quality_loss = mean_u (b_u − c_u)          <= accuracy_loss_max   (0.01)
        perplexity:  quality_loss = mean_u ((c_u − b_u) / b_u)  <= perplexity_loss_max (0.02)
        improvement_u = (b_u − c_u) / b_u   on the claimed quantity (``speed`` holds it; ``claim``
                                             names it: latency or energy)
        pass needs  mean_u improvement_u >= improvement_min (0.20)
               and  bootstrap CI(improvement, alpha).low > 0            # excludes no improvement
        other:       improvement of the non-claimed quantity must satisfy
                     CI(other_improvement, alpha).low >= −other_regression_max   # non-inferiority

    ``quality`` and ``speed`` are each paired by unit (task × seed, or batch configuration × seed)
    under identical hardware and audit conditions (§2.9); their unit sets may differ. The quality
    criterion is decided on the point estimate (§2.9 attaches the interval to the improvement); its
    interval is reported in ``numbers``. ``other`` carries the non-claimed quantity and is required
    as an argument: pass ``None`` only when it was not measured, which the result records by leaving
    the ``<other>_*`` numbers out. The plan does not quantify the non-regression budget, so when
    ``other`` is given, ``other_regression_max`` (frozen in ``metrics_freeze.json``) must be given
    too, and a budget without samples is an error. Claiming both quantities means two calls with
    the roles of ``speed`` and ``other`` swapped and both passing (P2.13 pitfall). Compile and
    startup times are reported separately by the efficiency suite and never enter this decision.
    """
    accuracy_loss_max = require_threshold(accuracy_loss_max, "accuracy_loss_max")
    perplexity_loss_max = require_threshold(perplexity_loss_max, "perplexity_loss_max")
    improvement_min = require_threshold(improvement_min, "improvement_min")
    if quality_kind == "accuracy":
        require_fraction(quality.baseline, "quality.baseline")
        require_fraction(quality.candidate, "quality.candidate")
        quality_loss = quality.baseline - quality.candidate
        quality_loss_max = accuracy_loss_max
        loss_text = f"accuracy loss {{}} > {_pp(quality_loss_max)} allowed"
    elif quality_kind == "perplexity":
        require_positive(quality.baseline, "quality.baseline")
        require_positive(quality.candidate, "quality.candidate")
        quality_loss = (quality.candidate - quality.baseline) / quality.baseline
        quality_loss_max = perplexity_loss_max
        loss_text = f"relative perplexity loss {{}} > {_pct(quality_loss_max)} allowed"
    else:
        raise ValueError(f"quality_kind={quality_kind!r} must be 'accuracy' or 'perplexity'")
    if claim not in _OTHER_OF:
        raise ValueError(f"claim={claim!r} must be 'latency' or 'energy'")
    other_name = _OTHER_OF[claim]

    quality_ci = bootstrap_mean(quality_loss, n_boot=n_boot, seed=seed, alpha=alpha)
    require_positive(speed.baseline, f"{claim}.baseline")
    require_positive(speed.candidate, f"{claim}.candidate")
    improvement = (speed.baseline - speed.candidate) / speed.baseline
    speed_ci = bootstrap_mean(improvement, n_boot=n_boot, seed=seed, alpha=alpha)
    fmt = _pp if quality_kind == "accuracy" else _pct

    reasons: list[str] = []
    if quality_ci.estimate > quality_loss_max + TOLERANCE:
        reasons.append(loss_text.format(fmt(quality_ci.estimate)))
    if speed_ci.estimate < improvement_min - TOLERANCE:
        reasons.append(
            f"{claim} improvement {_pct(speed_ci.estimate)} < {_pct(improvement_min)} required"
        )
    if speed_ci.low <= 0.0:
        reasons.append(
            f"{claim} improvement {_ci_text(speed_ci.low, speed_ci.high, alpha)} "
            "includes no improvement"
        )
    numbers: dict[str, float] = {
        f"{quality_kind}_loss": quality_ci.estimate,
        f"{quality_kind}_loss_ci_low": quality_ci.low,
        f"{quality_kind}_loss_ci_high": quality_ci.high,
        f"{quality_kind}_loss_max": quality_loss_max,
        "n_units_quality": quality.n_units,
        f"{claim}_improvement": speed_ci.estimate,
        f"{claim}_improvement_ci_low": speed_ci.low,
        f"{claim}_improvement_ci_high": speed_ci.high,
        "improvement_min": improvement_min,
        f"n_units_{claim}": speed.n_units,
        "alpha": speed_ci.alpha,
        "n_boot": speed_ci.n_boot,
        "seed": speed_ci.seed,
    }
    if other is None:
        if other_regression_max is not None:
            raise ValueError(
                f"other_regression_max={other_regression_max!r} was given but other=None; pass "
                f"the {other_name} samples, or no budget if {other_name} was not measured"
            )
    else:
        if other_regression_max is None:
            raise ValueError(
                f"{other_name} samples were given without other_regression_max, the "
                "non-regression budget of plan §2.9 (frozen in metrics_freeze.json); pass the "
                f"budget, or other=None if {other_name} was not measured"
            )
        budget = require_threshold(other_regression_max, "other_regression_max")
        require_positive(other.baseline, f"{other_name}.baseline")
        require_positive(other.candidate, f"{other_name}.candidate")
        other_improvement = (other.baseline - other.candidate) / other.baseline
        other_ci = bootstrap_mean(other_improvement, n_boot=n_boot, seed=seed, alpha=alpha)
        if other_ci.low < -budget - TOLERANCE:
            reasons.append(
                f"{other_name} non-regression: improvement "
                f"{_ci_text(other_ci.low, other_ci.high, alpha)} reaches below the "
                f"-{_pct(budget)} budget"
            )
        numbers.update(
            {
                f"{other_name}_improvement": other_ci.estimate,
                f"{other_name}_improvement_ci_low": other_ci.low,
                f"{other_name}_improvement_ci_high": other_ci.high,
                "other_regression_max": budget,
                f"n_units_{other_name}": other.n_units,
            }
        )
    return GateResult(passed=not reasons, reasons=reasons, numbers=numbers)


# --------------------------------------------------------------------------------------------------
# G3 — retention gate (plan §8.1, P3.8; paper §13)
# --------------------------------------------------------------------------------------------------


def retention_gate(
    A: ArrayLike,
    mean_forgetting_max: float = 0.02,
    task_loss_max: float = 0.05,
    new_task_gain_min: float = 0.03,
    *,
    control_gain: float,
    new_task_stages: Sequence[int] | None = None,
    baseline_final_accuracy: float | None = None,
    macro_gap_max: float = 0.01,
) -> GateResult:
    """The G3 retention gate (plan §8.1 / P3.8; paper §13; success criterion 2 of paper §1).

    Plan §8.1 G3: "mean forgetting ≤ 2 pp; no task loses > 5 pp; new-task gain ≥ 3 pp over frozen
    control; macro within 1 pt of strongest same-budget baseline". On the accuracy matrix ``A`` of
    plan §2.8 (0-based; see :mod:`hsn.eval.metrics`)::

        forgetting(A)                      <= mean_forgetting_max     (0.02)
        per_task_loss(A)[j]                <= task_loss_max           (0.05)   for every task j
        new_task_gain(A, stages) − control_gain >= new_task_gain_min  (0.03)
        baseline_final_accuracy − average_final_accuracy(A) <= macro_gap_max (0.01)   [optional]

    ``A`` is one ``[T, T]`` matrix or the ``[S, T, T]`` stack of per-seed matrices that P3.7's
    ``run_stream`` produces. Each statistic is computed per seed and then averaged over seeds
    (``numbers["n_seeds"]``); a seed-mean matrix must not be passed instead, because the maximum
    inside ``per_task_loss`` is then biased low (the maximum of a mean is at most the mean of the
    maxima). Per-seed calls feed the per-task table. On a truncated stream the metrics raise on the
    unmeasured entries; the gate must not run there (P3.7), and a PRELIMINARY report may pass the
    fully measured ``k × k`` prefix instead.

    ``control_gain`` is ``new_task_gain(A_control, new_task_stages)`` for the frozen control
    (``0.0`` for a model that cannot learn the new task at all), computed with the same stages;
    pass ``new_task_stages`` to leave out untrained stages (Appendix D stage 8, index 7). The
    fourth criterion needs the strongest same-budget continual baseline, so it is checked only
    when ``baseline_final_accuracy`` is given; P3.8 supplies it after running every arm.
    """
    mean_forgetting_max = require_threshold(mean_forgetting_max, "mean_forgetting_max")
    task_loss_max = require_threshold(task_loss_max, "task_loss_max")
    new_task_gain_min = require_threshold(new_task_gain_min, "new_task_gain_min")
    macro_gap_max = require_threshold(macro_gap_max, "macro_gap_max")
    control = require_difference(control_gain, "control_gain")
    seeds = matrix_stack(A)
    losses = np.mean([per_task_loss(mat) for mat in seeds], axis=0)
    mean_forgetting = float(losses.mean())
    gain = float(np.mean([new_task_gain(mat, new_task_stages) for mat in seeds]))
    final_accuracy = float(np.mean([average_final_accuracy(mat) for mat in seeds]))
    gain_over_control = gain - control

    reasons: list[str] = []
    if mean_forgetting > mean_forgetting_max + TOLERANCE:
        reasons.append(
            f"mean forgetting {_pp(mean_forgetting)} > {_pp(mean_forgetting_max)} allowed"
        )
    over = np.flatnonzero(losses > task_loss_max + TOLERANCE).tolist()
    if over:
        detail = ", ".join([f"task {j}: {_pp(float(losses[j]))}" for j in over])
        reasons.append(f"tasks {over} lose more than {_pp(task_loss_max)} ({detail})")
    if gain_over_control < new_task_gain_min - TOLERANCE:
        reasons.append(
            f"new-task gain {_pp(gain)} is {_pp(gain_over_control)} over the frozen control "
            f"({_pp(control)}), below the {_pp(new_task_gain_min)} required"
        )
    numbers: dict[str, float] = {
        "n_seeds": len(seeds),
        "mean_forgetting": mean_forgetting,
        "mean_forgetting_max": mean_forgetting_max,
        "worst_task_loss": float(losses.max()),
        "task_loss_max": task_loss_max,
        "n_tasks_over_task_loss_max": len(over),
        "new_task_gain": gain,
        "control_gain": control,
        "new_task_gain_over_control": gain_over_control,
        "new_task_gain_min": new_task_gain_min,
        "average_final_accuracy": final_accuracy,
    }
    numbers.update({f"task_loss_{j}": float(loss) for j, loss in enumerate(losses)})
    if baseline_final_accuracy is not None:
        baseline = float(require_fraction(baseline_final_accuracy, "baseline_final_accuracy"))
        gap = baseline - final_accuracy
        if gap > macro_gap_max + TOLERANCE:
            reasons.append(
                f"average final accuracy {_pp(final_accuracy)} is {_pp(gap)} below the strongest "
                f"same-budget baseline ({_pp(baseline)}), more than the {_pp(macro_gap_max)} "
                "allowed"
            )
        numbers.update(
            {
                "baseline_final_accuracy": baseline,
                "macro_gap": gap,
                "macro_gap_max": macro_gap_max,
            }
        )
    return GateResult(passed=not reasons, reasons=reasons, numbers=numbers)


# --------------------------------------------------------------------------------------------------
# G4 — composition gate (plan §1 criterion 3, §8.1, P4.7; paper §13)
# --------------------------------------------------------------------------------------------------


def composition_gate(
    gain: PairedSamples,
    min_gain: float = 0.03,
    *,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> GateResult:
    """The G4 composition gate (plan §1 criterion 3, §8.1 G4, P4.7; paper §13).

    Plan §1: "≥ 3 percentage points better held-out cross-language composition than a matched
    monolithic control, with a confidence interval excluding zero." ``gain`` pairs the control's
    accuracy (``baseline``) with HSN's (``candidate``) per unit — repository or task × seed on the
    stage-8 composition test (Appendix D), bootstrapped over repositories and tasks, paired by
    seed (plan §2.8, P4.7)::

        gain_u = c_u − b_u
        pass needs  mean_u gain_u >= min_gain (0.03)  and  bootstrap CI(gain, alpha).low > 0

    Fractions; ``n_units`` is reported because G4 runs three seeds (reduced certainty).
    """
    min_gain = require_threshold(min_gain, "min_gain")
    require_fraction(gain.baseline, "gain.baseline")
    require_fraction(gain.candidate, "gain.candidate")
    ci = bootstrap_mean(gain.candidate - gain.baseline, n_boot=n_boot, seed=seed, alpha=alpha)
    reasons: list[str] = []
    if ci.estimate < min_gain - TOLERANCE:
        reasons.append(f"composition gain {_pp(ci.estimate)} < {_pp(min_gain)} required")
    if ci.low <= 0.0:
        reasons.append(
            f"composition gain {100.0 * (1.0 - alpha):.0f} % CI [{_pp(ci.low)}, {_pp(ci.high)}] "
            "includes zero"
        )
    numbers: dict[str, float] = {
        "composition_gain": ci.estimate,
        "composition_gain_ci_low": ci.low,
        "composition_gain_ci_high": ci.high,
        "min_gain": min_gain,
        "n_units": gain.n_units,
        "alpha": ci.alpha,
        "n_boot": ci.n_boot,
        "seed": ci.seed,
    }
    return GateResult(passed=not reasons, reasons=reasons, numbers=numbers)
