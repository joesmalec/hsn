"""P0.7 tests: continual-learning metrics, paired bootstrap and gate decisions (plan §2.8–2.9)."""

import json

import numpy as np
import pytest
from hsn.eval.bootstrap import CI, PairedSamples, bootstrap_mean, paired_bootstrap
from hsn.eval.gates import GateResult, composition_gate, efficiency_gate, retention_gate
from hsn.eval.metrics import (
    accuracy_matrix,
    average_final_accuracy,
    backward_transfer,
    forgetting,
    forward_transfer,
    new_task_gain,
    per_task_loss,
)

NAN = float("nan")

# Hand-computed 3 × 3 accuracy matrix (stage × task, fractions); values stay away from thresholds.
A3 = [
    [0.90, 0.10, 0.20],
    [0.80, 0.90, 0.30],
    [0.70, 0.85, 0.95],
]

# --------------------------------------------------------------------------------------------------
# Plan §2.8 metrics
# --------------------------------------------------------------------------------------------------


def test_p0_7_hand_computed_3x3() -> None:
    assert average_final_accuracy(A3) == pytest.approx((0.70 + 0.85 + 0.95) / 3)
    np.testing.assert_allclose(per_task_loss(A3), [0.20, 0.05])  # max over stages 0..1 − final
    assert forgetting(A3) == pytest.approx(0.125)
    assert backward_transfer(A3) == pytest.approx(((0.70 - 0.90) + (0.85 - 0.90)) / 2)
    assert new_task_gain(A3) == pytest.approx(((0.90 - 0.10) + (0.95 - 0.30)) / 2)
    assert new_task_gain(A3, stages=[2]) == pytest.approx(0.95 - 0.30)
    assert forward_transfer([0.30, 0.20], [0.20, 0.25]) == pytest.approx(0.025)

    # accuracy_matrix: 0-based (stage, task) keys, NaN where unmeasured, measured entries kept.
    results = {(i, j): A3[i][j] for i in range(3) for j in range(3) if (i, j) != (0, 2)}
    A = accuracy_matrix(results, 3)
    assert A.shape == (3, 3) and A.dtype == np.float64
    assert np.isnan(A[0, 2]) and np.isnan(A).sum() == 1
    expected = np.where(np.isnan(A), -1.0, np.array(A3))
    assert np.array_equal(np.nan_to_num(A, nan=-1.0), expected)
    assert forgetting(A) == pytest.approx(0.125)  # the NaN sits in the final task's column
    assert np.isnan(accuracy_matrix({}, 2)).all()
    with pytest.raises(ValueError, match="outside"):
        accuracy_matrix({(3, 0): 0.5}, 3)
    with pytest.raises(ValueError, match="outside"):
        accuracy_matrix({(0, -1): 0.5}, 3)
    with pytest.raises(ValueError, match="percentage"):
        accuracy_matrix({(0, 0): 95.0}, 3)
    with pytest.raises(ValueError, match="T="):
        accuracy_matrix({}, 0)
    with pytest.raises(ValueError, match="integer"):
        accuracy_matrix({(0.0, 1): 0.5}, 3)  # type: ignore[dict-item]
    # NumPy integer keys are integers too.
    assert accuracy_matrix({(np.int64(1), np.int64(2)): 0.5}, 3)[1, 2] == 0.5


def test_p0_7_forgetting_ignores_final_stage() -> None:
    # Task 0 is at its best after the final stage: §2.8 takes the maximum over stages 1..T−1 only.
    A = [
        [0.80, 0.10, 0.05],
        [0.70, 0.90, 0.10],
        [0.95, 0.85, 0.90],
    ]
    losses = per_task_loss(A)
    assert losses[0] == pytest.approx(max(0.80, 0.70) - 0.95)  # not max(0.80, 0.70, 0.95) − 0.95
    assert losses[1] == pytest.approx(0.90 - 0.85)
    assert forgetting(A) == pytest.approx(-0.05)
    including_final = np.mean([max(0.80, 0.70, 0.95) - 0.95, max(0.10, 0.90, 0.85) - 0.85])
    assert including_final == pytest.approx(0.025)
    assert forgetting(A) != pytest.approx(including_final)


def test_p0_7_forgetting_can_be_negative() -> None:
    # Every earlier task improves at the final stage, so each loss and the mean are negative. The
    # unmeasured "before training" entries (NaN above the diagonal) are ignored by the maximum.
    A = [
        [0.50, NAN, NAN],
        [0.60, 0.70, NAN],
        [0.80, 0.75, 0.90],
    ]
    np.testing.assert_allclose(per_task_loss(A), [-0.20, -0.05])
    assert forgetting(A) < 0.0
    assert forgetting(A) == pytest.approx(-0.125)
    assert backward_transfer(A) == pytest.approx(((0.80 - 0.50) + (0.75 - 0.70)) / 2)
    assert average_final_accuracy(A) == pytest.approx((0.80 + 0.75 + 0.90) / 3)


def test_p0_7_metrics_reject_percentages_and_shapes() -> None:
    with pytest.raises(ValueError, match="percentage"):
        forgetting([[90.0, 10.0], [80.0, 95.0]])
    with pytest.raises(ValueError, match="square"):
        forgetting([[0.9, 0.1, 0.2], [0.8, 0.9, 0.3]])
    with pytest.raises(ValueError, match="two stages"):
        forgetting([[0.9]])
    with pytest.raises(ValueError, match="infinite"):
        forgetting([[0.9, float("inf")], [0.8, 0.9]])
    with pytest.raises(ValueError, match=r"unmeasured \(NaN\) at \(stage, task\) \[\(0, 0\)\]"):
        per_task_loss([[NAN, 0.1], [0.8, 0.9]])
    # An already-learned task (on or below the diagonal) must be measured at every earlier stage:
    # a missing peak would silently lower the maximum and the loss.
    with pytest.raises(ValueError, match=r"\[\(0, 0\)\]"):
        per_task_loss([[NAN, 0.1, 0.1], [0.80, 0.9, 0.1], [0.79, 0.88, 0.9]])
    with pytest.raises(ValueError, match=r"\[\(1, 0\)\]"):
        per_task_loss([[0.95, 0.1, 0.1], [NAN, 0.9, 0.1], [0.79, 0.88, 0.9]])
    with pytest.raises(ValueError, match=r"\[\(1, 1\)\]"):
        per_task_loss([[0.95, 0.1, 0.1], [0.80, NAN, 0.1], [0.79, 0.88, 0.9]])
    with pytest.raises(ValueError, match="final-stage"):
        per_task_loss([[0.9, 0.1], [NAN, 0.9]])
    with pytest.raises(ValueError, match="final-stage"):
        average_final_accuracy([[0.9, 0.1], [0.8, NAN]])
    with pytest.raises(ValueError, match=r"A\[j, j\]"):
        backward_transfer([[NAN, 0.1], [0.8, 0.9]])
    with pytest.raises(ValueError, match="shape"):
        forward_transfer([0.3, 0.2], [0.2])
    with pytest.raises(ValueError, match="at least one"):
        forward_transfer([], [])
    with pytest.raises(ValueError, match="percentage"):
        forward_transfer([30.0], [20.0])
    with pytest.raises(ValueError, match="outside"):
        new_task_gain(A3, stages=[0])
    with pytest.raises(ValueError, match="outside"):
        new_task_gain(A3, stages=[3])
    with pytest.raises(ValueError, match="empty"):
        new_task_gain(A3, stages=[])
    with pytest.raises(ValueError, match="twice"):
        new_task_gain(A3, stages=[1, 1])
    with pytest.raises(ValueError, match="integer"):
        new_task_gain(A3, stages=[1.5])  # type: ignore[list-item]
    assert new_task_gain(A3, stages=np.array([2])) == pytest.approx(0.95 - 0.30)
    with pytest.raises(ValueError, match=r"A\[j-1, j\]"):
        new_task_gain([[0.5, NAN], [0.6, 0.9]])


# --------------------------------------------------------------------------------------------------
# Plan §2.8–2.9 paired bootstrap
# --------------------------------------------------------------------------------------------------


def test_p0_7_bootstrap_covers_known_mean() -> None:
    true_mean = 0.5
    n_units, n_reps = 30, 300
    rng = np.random.default_rng(1234)
    covered = 0
    for rep in range(n_reps):
        b = rng.normal(0.0, 1.0, size=n_units)
        a = b + true_mean + rng.normal(0.0, 0.5, size=n_units)
        ci = paired_bootstrap(a, b, n_boot=2000, seed=rep)
        assert ci.estimate == pytest.approx(float(np.mean(a - b)))
        assert ci.low <= ci.estimate <= ci.high
        covered += ci.low <= true_mean <= ci.high
    coverage = covered / n_reps
    # Nominal 95 %; the percentile bootstrap under-covers a little at n = 30.
    assert 0.88 <= coverage <= 0.99, coverage

    # Deterministic given the seed; the estimate is the sample mean, whatever the resampling.
    a = np.array([0.5, 0.7, 0.6, 0.8, 0.9])
    b = np.array([0.4, 0.5, 0.6, 0.7, 0.6])
    first = paired_bootstrap(a, b, n_boot=500, seed=7)
    assert first == paired_bootstrap(a, b, n_boot=500, seed=7)
    other_seed = paired_bootstrap(a, b, n_boot=500, seed=8)
    assert (first.low, first.high) != (other_seed.low, other_seed.high)
    assert first == CI(
        low=first.low,
        high=first.high,
        estimate=float(np.mean(a - b)),
        alpha=0.05,
        n_boot=500,
        n_units=5,
        seed=7,
    )
    assert first.low <= first.estimate <= first.high
    # The interval is on the *paired* differences: breaking the pairing (same values, shuffled
    # partners) widens it, because the per-unit noise no longer cancels.
    b_corr = rng.normal(0.0, 1.0, size=40)
    a_corr = b_corr + 0.5 + rng.normal(0.0, 0.05, size=40)
    paired = paired_bootstrap(a_corr, b_corr, n_boot=2000, seed=0)
    shuffled = paired_bootstrap(a_corr, rng.permutation(b_corr), n_boot=2000, seed=0)
    assert paired.high - paired.low < 0.2 * (shuffled.high - shuffled.low)
    # A constant paired difference gives a degenerate interval at that constant.
    const = paired_bootstrap(b + 0.1, b, n_boot=200)
    assert const.low == pytest.approx(0.1) and const.high == pytest.approx(0.1)
    # More resamples than one chunk (1 000 rows) are drawn chunk by chunk.
    big = bootstrap_mean(a - b, n_boot=2500, seed=1)
    assert big.n_boot == 2500 and big.low <= big.estimate <= big.high
    # Pairing is by unit: misaligned lengths, NaN, a single unit and bad parameters raise.
    with pytest.raises(ValueError, match="pairing is by unit"):
        paired_bootstrap(a, b[:-1])
    with pytest.raises(ValueError, match="NaN"):
        paired_bootstrap(np.array([0.5, NAN]), np.array([0.4, 0.4]))
    with pytest.raises(ValueError, match="at least two"):
        bootstrap_mean([0.5])
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_mean([0.5, 0.6], n_boot=0)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_mean([0.5, 0.6], alpha=1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        bootstrap_mean([[0.5, 0.6]])


def test_p0_7_paired_samples_from_mappings() -> None:
    baseline = {("kv", 0): 0.90, ("kv", 1): 0.92, ("brackets", 0): 0.80, ("brackets", 1): 0.81}
    candidate = {("brackets", 1): 0.83, ("kv", 1): 0.91, ("kv", 0): 0.89, ("brackets", 0): 0.82}
    paired = PairedSamples.from_mappings(baseline, candidate)
    assert paired.units == ("('brackets', 0)", "('brackets', 1)", "('kv', 0)", "('kv', 1)")
    np.testing.assert_allclose(paired.baseline, [0.80, 0.81, 0.90, 0.92])
    np.testing.assert_allclose(paired.candidate, [0.82, 0.83, 0.89, 0.91])
    assert paired.n_units == 4
    with pytest.raises(ValueError, match=r"only in baseline: \[\('kv', 1\)\]"):
        PairedSamples.from_mappings(
            baseline, {k: v for k, v in candidate.items() if k != ("kv", 1)}
        )
    with pytest.raises(ValueError, match=r"only in candidate: \[\('kv', 2\)\]"):
        PairedSamples.from_mappings({("kv", 0): 0.9}, {("kv", 0): 0.9, ("kv", 2): 0.8})
    with pytest.raises(ValueError, match="one value per unit"):
        PairedSamples(np.array([0.9, 0.8]), np.array([0.9]))
    with pytest.raises(ValueError, match="unit labels"):
        PairedSamples(np.array([0.9, 0.8]), np.array([0.9, 0.7]), units=("a",))
    with pytest.raises(ValueError, match="unique"):
        PairedSamples(np.array([0.9, 0.8]), np.array([0.9, 0.7]), units=("a", "a"))
    with pytest.raises(ValueError, match="empty"):
        PairedSamples(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="NaN"):
        PairedSamples(np.array([0.9, NAN]), np.array([0.9, 0.7]))
    labelled = PairedSamples([0.9, 0.8], [0.9, 0.7], units=("t0/s0", "t0/s1"))
    assert labelled.units == ("t0/s0", "t0/s1") and labelled.baseline.dtype == np.float64


# --------------------------------------------------------------------------------------------------
# Plan §2.9 efficiency gate (G2)
# --------------------------------------------------------------------------------------------------

ACC_BASELINE = np.array([0.90, 0.91, 0.89, 0.92, 0.90, 0.91])  # accuracy per unit (task × seed)
LATENCY_BASELINE = np.array([10.0, 11.0, 10.5, 9.8, 10.2, 10.1])  # ms per token, per unit


def _quality(loss: float) -> PairedSamples:
    return PairedSamples(ACC_BASELINE, ACC_BASELINE - loss)


def _speed(improvements: list[float]) -> PairedSamples:
    r = np.asarray(improvements, dtype=np.float64)
    return PairedSamples(LATENCY_BASELINE, LATENCY_BASELINE * (1.0 - r))


GOOD_SPEED = _speed([0.30, 0.28, 0.32, 0.29, 0.31, 0.30])


def test_p0_7_efficiency_gate_examples() -> None:
    # Pass: 0.5 pp accuracy loss, ~30 % latency improvement, CI above zero; energy not measured.
    ok = efficiency_gate(_quality(0.005), GOOD_SPEED, "accuracy", "latency", None)
    assert ok.passed and ok.reasons == []
    assert ok.numbers["accuracy_loss"] == pytest.approx(0.005)
    assert ok.numbers["accuracy_loss_max"] == 0.01 and ok.numbers["improvement_min"] == 0.20
    assert ok.numbers["latency_improvement"] == pytest.approx(0.30)
    assert 0.0 < ok.numbers["latency_improvement_ci_low"] <= ok.numbers["latency_improvement"]
    assert ok.numbers["n_units_quality"] == 6 and ok.numbers["n_units_latency"] == 6
    assert "energy_improvement" not in ok.numbers and "other_regression_max" not in ok.numbers

    # Fail on quality: 3 pp accuracy loss with the same speed win.
    bad_quality = efficiency_gate(_quality(0.03), GOOD_SPEED, "accuracy", "latency", None)
    assert not bad_quality.passed and len(bad_quality.reasons) == 1
    assert bad_quality.reasons[0].startswith("accuracy loss 3.00 pp > 1.00 pp allowed")

    # Fail on CI: mean improvement 25 %, but units straddle zero and the 95 % CI reaches below it.
    noisy = _speed([0.8, -0.3] * 3)
    bad_ci = efficiency_gate(_quality(0.005), noisy, "accuracy", "latency", None)
    assert not bad_ci.passed and len(bad_ci.reasons) == 1
    assert "includes no improvement" in bad_ci.reasons[0]
    assert bad_ci.numbers["latency_improvement"] == pytest.approx(0.25)
    assert bad_ci.numbers["latency_improvement_ci_low"] < 0.0

    # Below the 20 % bar although the CI excludes zero: 15 % ± 1 %.
    small = efficiency_gate(_quality(0.005), _speed([0.14, 0.16] * 3), "accuracy", "latency", None)
    assert not small.passed and len(small.reasons) == 1
    assert small.reasons[0].startswith("latency improvement 15.00 % < 20.00 % required")

    # Perplexity: 1.5 % relative loss passes, 3 % fails; the claim here is energy.
    ppl = np.array([20.0, 21.0, 19.5, 20.5, 20.2, 19.8])
    ok_ppl = efficiency_gate(
        PairedSamples(ppl, ppl * 1.015), GOOD_SPEED, "perplexity", "energy", None
    )
    assert ok_ppl.passed and ok_ppl.numbers["perplexity_loss"] == pytest.approx(0.015)
    assert ok_ppl.numbers["perplexity_loss_max"] == 0.02
    assert "energy_improvement" in ok_ppl.numbers and "latency_improvement" not in ok_ppl.numbers
    bad_ppl = efficiency_gate(
        PairedSamples(ppl, ppl * 1.03), GOOD_SPEED, "perplexity", "energy", None
    )
    assert not bad_ppl.passed and len(bad_ppl.reasons) == 1
    assert bad_ppl.reasons[0].startswith("relative perplexity loss 3.00 % > 2.00 % allowed")

    # The non-claimed quantity: within its non-regression budget passes, beyond it fails, and it
    # cannot be given without a budget (the plan leaves that number to metrics_freeze.json).
    energy = np.array([5.0, 5.2, 4.9, 5.1, 5.0, 5.3])
    within = PairedSamples(energy, energy * 1.02)  # 2 % worse
    beyond = PairedSamples(energy, energy * 1.10)  # 10 % worse
    ok_other = efficiency_gate(
        _quality(0.005), GOOD_SPEED, "accuracy", "latency", within, other_regression_max=0.05
    )
    assert ok_other.passed
    assert ok_other.numbers["energy_improvement"] == pytest.approx(-0.02)
    assert ok_other.numbers["other_regression_max"] == 0.05
    assert ok_other.numbers["n_units_energy"] == 6
    bad_other = efficiency_gate(
        _quality(0.005), GOOD_SPEED, "accuracy", "latency", beyond, other_regression_max=0.05
    )
    assert not bad_other.passed and len(bad_other.reasons) == 1
    assert bad_other.reasons[0].startswith("energy non-regression")
    assert bad_other.numbers["energy_improvement"] == pytest.approx(-0.10)
    with pytest.raises(ValueError, match="other_regression_max"):
        efficiency_gate(_quality(0.005), GOOD_SPEED, "accuracy", "latency", within)
    with pytest.raises(ValueError, match="other=None"):
        efficiency_gate(
            _quality(0.005), GOOD_SPEED, "accuracy", "latency", None, other_regression_max=0.05
        )

    # Literal arguments are checked at run time; units are checked before any decision.
    with pytest.raises(ValueError, match="quality_kind"):
        efficiency_gate(_quality(0.005), GOOD_SPEED, "bleu", "latency", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="claim"):
        efficiency_gate(_quality(0.005), GOOD_SPEED, "accuracy", "tokens/s", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="percentage"):
        percent = PairedSamples(ACC_BASELINE * 100, ACC_BASELINE * 100)
        efficiency_gate(percent, GOOD_SPEED, "accuracy", "latency", None)
    with pytest.raises(ValueError, match="strictly positive"):
        zero_latency = PairedSamples(np.zeros(6), np.ones(6))
        efficiency_gate(_quality(0.005), zero_latency, "accuracy", "latency", None)
    with pytest.raises(ValueError, match="improvement_min"):
        efficiency_gate(
            _quality(0.005), GOOD_SPEED, "accuracy", "latency", None, improvement_min=NAN
        )
    # Thresholds written in percentage points or percent would make a criterion unfailable.
    for bad in (
        {"accuracy_loss_max": 1.0},
        {"perplexity_loss_max": 2.0},
        {"improvement_min": 20.0},
    ):
        with pytest.raises(ValueError, match="not percentage points or percent"):
            efficiency_gate(_quality(0.005), GOOD_SPEED, "accuracy", "latency", None, **bad)
    with pytest.raises(ValueError, match="other_regression_max=5"):
        efficiency_gate(
            _quality(0.005), GOOD_SPEED, "accuracy", "latency", within, other_regression_max=5
        )


def test_p0_7_gate_result_json() -> None:
    result = efficiency_gate(_quality(0.005), GOOD_SPEED, "accuracy", "latency", None)
    data = json.loads(result.to_json())
    assert set(data) == {"passed", "reasons", "numbers"}
    assert data["passed"] is True and data["reasons"] == []
    assert data["numbers"] == result.numbers
    assert all(
        isinstance(value, int | float) and not isinstance(value, bool)
        for value in data["numbers"].values()
    )
    failed = GateResult(passed=False, reasons=["x > y"], numbers={"x": 1.0})
    assert json.loads(failed.to_json()) == {
        "passed": False,
        "reasons": ["x > y"],
        "numbers": {"x": 1.0},
    }


# --------------------------------------------------------------------------------------------------
# Plan §8.1 retention gate (G3)
# --------------------------------------------------------------------------------------------------


def _retention_matrix(final_row: list[float]) -> list[list[float]]:
    """T = 4 (stage × task): best earlier accuracies 0.90, 0.80, 0.85; new tasks start at 0.10."""
    return [
        [0.90, 0.10, 0.05, 0.05],
        [0.88, 0.80, 0.10, 0.05],
        [0.85, 0.78, 0.85, 0.10],
        final_row,
    ]


def test_p0_7_retention_gate_per_task_rule() -> None:
    # Mean forgetting 1.33 pp is under the 2 pp bar, but task 0 loses 6 pp: the per-task rule fails.
    A = _retention_matrix([0.84, 0.82, 0.85, 0.90])
    np.testing.assert_allclose(per_task_loss(A), [0.06, -0.02, 0.0])
    result = retention_gate(A, control_gain=0.0)
    assert not result.passed and len(result.reasons) == 1
    assert result.reasons[0].startswith("tasks [0] lose more than 5.00 pp (task 0: 6.00 pp)")
    assert result.numbers["mean_forgetting"] == pytest.approx(0.04 / 3)
    assert result.numbers["worst_task_loss"] == pytest.approx(0.06)
    assert result.numbers["n_tasks_over_task_loss_max"] == 1
    assert result.numbers["task_loss_0"] == pytest.approx(0.06)
    assert result.numbers["task_loss_1"] == pytest.approx(-0.02)
    assert result.numbers["task_loss_2"] == pytest.approx(0.0)
    assert result.numbers["new_task_gain"] == pytest.approx(0.75)
    assert result.numbers["average_final_accuracy"] == pytest.approx(
        (0.84 + 0.82 + 0.85 + 0.90) / 4
    )

    # The same matrix with a 4.5 pp loss on task 0 passes every criterion.
    A_ok = _retention_matrix([0.855, 0.82, 0.85, 0.90])
    ok = retention_gate(A_ok, control_gain=0.0)
    assert ok.passed and ok.reasons == []
    assert ok.numbers["worst_task_loss"] == pytest.approx(0.045)
    assert "baseline_final_accuracy" not in ok.numbers and "macro_gap" not in ok.numbers

    # Mean forgetting over 2 pp with every task within 5 pp fails the mean criterion only.
    mean_fail = retention_gate(_retention_matrix([0.86, 0.76, 0.81, 0.90]), control_gain=0.0)
    assert not mean_fail.passed and len(mean_fail.reasons) == 1
    assert mean_fail.reasons[0].startswith("mean forgetting 4.00 pp > 2.00 pp allowed")

    # New-task gain is judged over the frozen control, with the same stages for both arms.
    gain_fail = retention_gate(A_ok, control_gain=0.74)
    assert not gain_fail.passed and len(gain_fail.reasons) == 1
    assert gain_fail.reasons[0].startswith(
        "new-task gain 75.00 pp is 1.00 pp over the frozen control (74.00 pp), below the 3.00 pp"
    )
    assert gain_fail.numbers["new_task_gain_over_control"] == pytest.approx(0.01)
    subset = retention_gate(A_ok, control_gain=0.0, new_task_stages=[1, 2])
    assert subset.passed and subset.numbers["new_task_gain"] == pytest.approx((0.70 + 0.75) / 2)

    # The optional fourth G3 criterion: macro accuracy within 1 pt of the strongest baseline.
    macro_fail = retention_gate(A_ok, control_gain=0.0, baseline_final_accuracy=0.90)
    assert not macro_fail.passed and len(macro_fail.reasons) == 1
    assert macro_fail.reasons[0].startswith("average final accuracy")
    assert macro_fail.numbers["macro_gap"] == pytest.approx(0.90 - 0.85625)
    assert macro_fail.numbers["baseline_final_accuracy"] == 0.90
    macro_ok = retention_gate(A_ok, control_gain=0.0, baseline_final_accuracy=0.86)
    assert macro_ok.passed and macro_ok.numbers["macro_gap_max"] == 0.01

    # Thresholds and the control gain must be finite numbers: NaN would pass every comparison.
    with pytest.raises(ValueError, match="control_gain"):
        retention_gate(A_ok, control_gain=NAN)
    with pytest.raises(ValueError, match="task_loss_max"):
        retention_gate(A_ok, task_loss_max=NAN, control_gain=0.0)
    # ... and must be fractions: 5 (meant as 5 pp) would make the per-task rule unfailable.
    with pytest.raises(ValueError, match="task_loss_max=5"):
        retention_gate(A_ok, task_loss_max=5, control_gain=0.0)
    with pytest.raises(ValueError, match="mean_forgetting_max=2"):
        retention_gate(A_ok, mean_forgetting_max=2, control_gain=0.0)
    with pytest.raises(ValueError, match="control_gain=3"):
        retention_gate(A_ok, control_gain=3)
    with pytest.raises(ValueError, match="macro_gap_max=1"):
        retention_gate(A_ok, control_gain=0.0, baseline_final_accuracy=0.9, macro_gap_max=1)
    with pytest.raises(ValueError, match="baseline_final_accuracy"):
        retention_gate(A_ok, control_gain=0.0, baseline_final_accuracy=90.0)
    with pytest.raises(ValueError, match="final-stage"):
        retention_gate(_retention_matrix([0.855, NAN, 0.85, 0.90]), control_gain=0.0)


# --------------------------------------------------------------------------------------------------
# Plan §1 criterion 3 / §8.1 composition gate (G4)
# --------------------------------------------------------------------------------------------------


def test_p0_7_composition_gate_examples() -> None:
    control = np.array([0.60, 0.62, 0.58, 0.61, 0.59, 0.63])  # monolithic control, per repo × seed
    ok = composition_gate(PairedSamples(control, control + [0.05, 0.04, 0.06, 0.05, 0.04, 0.06]))
    assert ok.passed and ok.reasons == []
    assert ok.numbers["composition_gain"] == pytest.approx(0.05)
    assert 0.0 < ok.numbers["composition_gain_ci_low"] <= ok.numbers["composition_gain_ci_high"]
    assert ok.numbers["min_gain"] == 0.03 and ok.numbers["n_units"] == 6
    small = composition_gate(PairedSamples(control, control + [0.02, 0.01, 0.02, 0.01, 0.02, 0.02]))
    assert not small.passed and len(small.reasons) == 1
    assert small.reasons[0].startswith("composition gain 1.67 pp < 3.00 pp required")
    noisy = composition_gate(PairedSamples(control, control + [0.30, -0.20] * 3))
    assert not noisy.passed and len(noisy.reasons) == 1
    assert "includes zero" in noisy.reasons[0]
    assert noisy.numbers["composition_gain"] == pytest.approx(0.05)
    assert noisy.numbers["composition_gain_ci_low"] < 0.0
    with pytest.raises(ValueError, match="percentage"):
        composition_gate(PairedSamples(control * 100, control * 100))
    with pytest.raises(ValueError, match="min_gain"):
        composition_gate(PairedSamples(control, control + 0.05), min_gain=NAN)
    with pytest.raises(ValueError, match="min_gain=3"):
        composition_gate(PairedSamples(control, control + 0.05), min_gain=3)


# --------------------------------------------------------------------------------------------------
# Boundaries and per-seed evaluation
# --------------------------------------------------------------------------------------------------


def test_p0_7_gates_decide_exact_boundaries_as_the_plan_reads() -> None:
    # Differences of accuracies that land exactly on a bar round across it in binary64
    # (0.90 − 0.85 > 0.05, 0.83 − 0.80 < 0.03); the gates decide them inclusively, as plan §8.1 and
    # §2.9 state ("no task loses > 5 pp", "≥ 3 pp", "≤ 1 pp", "≥ 20 %").
    exact_5pp = retention_gate(_retention_matrix([0.85, 0.82, 0.85, 0.90]), control_gain=0.0)
    assert exact_5pp.passed and exact_5pp.numbers["task_loss_0"] == pytest.approx(0.05)
    exact_2pp = retention_gate(_retention_matrix([0.88, 0.78, 0.83, 0.90]), control_gain=0.0)
    assert exact_2pp.passed and exact_2pp.numbers["mean_forgetting"] == pytest.approx(0.02)
    exact_3pp = retention_gate([[0.90, 0.05], [0.90, 0.88]], control_gain=0.80)
    assert exact_3pp.passed
    assert exact_3pp.numbers["new_task_gain_over_control"] == pytest.approx(0.03)
    exact_1pp = efficiency_gate(_quality(0.01), GOOD_SPEED, "accuracy", "latency", None)
    assert exact_1pp.passed and exact_1pp.numbers["accuracy_loss"] == pytest.approx(0.01)
    exact_20pct = efficiency_gate(
        _quality(0.005),
        PairedSamples(LATENCY_BASELINE, LATENCY_BASELINE * 0.8),
        "accuracy",
        "latency",
        None,
    )
    assert exact_20pct.passed
    assert exact_20pct.numbers["latency_improvement"] == pytest.approx(0.20)
    control = np.full(6, 0.80)
    exact_3pp_gain = composition_gate(PairedSamples(control, np.full(6, 0.83)))
    assert exact_3pp_gain.passed
    assert exact_3pp_gain.numbers["composition_gain"] == pytest.approx(0.03)


def test_p0_7_retention_gate_evaluates_per_seed() -> None:
    # Two seeds whose task-0 accuracy peaks at different stages: each seed loses 6 pp on task 0,
    # so the per-task rule must fail (mean forgetting stays under 2 pp because task 1 improves).
    # The seed-mean matrix would hide it: its task-0 maximum is only 0.85.
    seed0 = [[0.90, 0.10, 0.10], [0.80, 0.90, 0.10], [0.84, 0.93, 0.90]]
    seed1 = [[0.80, 0.10, 0.10], [0.90, 0.90, 0.10], [0.84, 0.93, 0.90]]
    stacked = retention_gate(np.stack([seed0, seed1]), control_gain=0.0)
    assert not stacked.passed and len(stacked.reasons) == 1
    assert stacked.reasons[0].startswith("tasks [0] lose more than 5.00 pp (task 0: 6.00 pp)")
    assert stacked.numbers["n_seeds"] == 2
    assert stacked.numbers["task_loss_0"] == pytest.approx(0.06)
    assert stacked.numbers["mean_forgetting"] == pytest.approx((0.06 - 0.03) / 2)
    assert stacked.numbers["new_task_gain"] == pytest.approx(0.80)
    seed_mean = retention_gate(np.mean(np.stack([seed0, seed1]), axis=0), control_gain=0.0)
    assert seed_mean.passed and seed_mean.numbers["n_seeds"] == 1  # the biased shortcut
    assert retention_gate(seed0, control_gain=0.0).numbers["n_seeds"] == 1
    with pytest.raises(ValueError, match="empty stack"):
        retention_gate(np.zeros((0, 3, 3)), control_gain=0.0)
    with pytest.raises(ValueError, match=r"\[S, T, T\]"):
        retention_gate(np.zeros(3), control_gain=0.0)
    with pytest.raises(ValueError, match="square"):
        retention_gate(np.zeros((2, 3, 2)), control_gain=0.0)
