"""Continual-learning metrics on the accuracy matrix ``A[i, j]`` (plan §2.8; paper §11).

P0.7. ``A(i, j)`` is the held-out accuracy on task ``j`` after learning stage ``i``, for ``T``
stages (plan §2.8). The paper indexes stages and tasks ``1..T``; this module is 0-based like NumPy:
``A[i, j]`` with ``0 <= i, j < T`` is the paper's ``A(i+1, j+1)``, so stage ``s`` of Appendix D is
row and column ``s − 1`` and the final stage is row ``T − 1``. NaN marks an entry that was not
measured (:func:`accuracy_matrix`). Entries above the diagonal (a task evaluated before it was
learned) may stay NaN and are ignored by the maximum in :func:`per_task_loss`; every entry on or
below the diagonal that a metric needs must be measured, and a metric raises
:class:`ValueError` instead of skipping it (plan §4.3 errors over silence; P3.7: never run the
gate on a truncated stream).

Units: every accuracy is a *fraction* in ``[0, 1]`` and every metric returns a fraction.
Percentage points are ``100 ×`` a fraction and appear only in prose (chunk P0.7 pitfall: never mix
the two). A value outside ``[0, 1]`` — for example ``95.0`` meant as a percentage — raises.
"""

import operator
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "accuracy_matrix",
    "average_final_accuracy",
    "backward_transfer",
    "forgetting",
    "forward_transfer",
    "matrix_stack",
    "new_task_gain",
    "per_task_loss",
    "require_difference",
    "require_finite",
    "require_fraction",
    "require_positive",
    "require_threshold",
]


# --------------------------------------------------------------------------------------------------
# Validation helpers (shared with hsn.eval.gates, which keeps only decision branches)
# --------------------------------------------------------------------------------------------------


def require_finite(value: float, name: str) -> float:
    """``value`` as a float; raises unless it is finite (a NaN threshold would pass every test)."""
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name}={value!r} must be a finite number")
    return out


def require_threshold(value: float, name: str) -> float:
    """A gate threshold as a float: finite and strictly inside ``(−1, 1)``.

    Every threshold of plan §2.9 and §8.1 is a fraction (``0.01`` = 1 pp, ``0.20`` = 20 %); a
    value such as ``5`` (meant as 5 pp) or ``20`` (meant as 20 %) would make a criterion
    impossible to fail, so it raises.
    """
    out = require_finite(value, name)
    if not -1.0 < out < 1.0:
        raise ValueError(
            f"{name}={value!r} must be a fraction (0.01 = 1 pp, 0.20 = 20 %), not percentage "
            "points or percent"
        )
    return out


def require_difference(value: float, name: str) -> float:
    """A difference of two fractions (a gain or a loss) as a float: finite and in ``[−1, 1]``."""
    out = require_finite(value, name)
    if not -1.0 <= out <= 1.0:
        raise ValueError(
            f"{name}={value!r} must be a difference of fractions in [-1, 1], not percentage points"
        )
    return out


def _as_index(value: Any, what: str) -> int:
    """``value`` as an integer index (``operator.index``: no floats, NumPy integers allowed)."""
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{what}={value!r} must be an integer") from exc


def require_fraction(
    values: ArrayLike, name: str, *, allow_nan: bool = False
) -> NDArray[np.float64]:
    """``values`` as a float64 array of fractions in ``[0, 1]`` (NaN only when ``allow_nan``).

    Raises :class:`ValueError` otherwise, so an accuracy given in percentage points (``95.0``)
    fails loudly instead of being compared with a threshold stated as a fraction.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    invalid = ~finite & ~np.isnan(arr) if allow_nan else ~finite
    if invalid.any():
        kind = "infinite" if allow_nan else "NaN or infinite"
        raise ValueError(f"{name} contains {kind} values")
    out_of_range = finite & ((arr < 0.0) | (arr > 1.0))
    if out_of_range.any():
        raise ValueError(
            f"{name} must hold fractions in [0, 1], not percentage points; "
            f"got {float(arr[out_of_range][0])}"
        )
    return arr


def require_positive(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """``values`` as a float64 array of finite, strictly positive numbers.

    For latency, energy and perplexity, which are denominators of relative changes.
    """
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    not_positive = arr <= 0.0
    if not_positive.any():
        raise ValueError(
            f"{name} must be strictly positive (it is the denominator of a relative change); "
            f"got {float(arr[not_positive][0])}"
        )
    return arr


def _as_matrix(A: ArrayLike) -> NDArray[np.float64]:
    """``A`` as a float64 ``[T, T]`` accuracy matrix, ``T >= 2``, entries NaN or in ``[0, 1]``."""
    mat = np.asarray(A, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"A must be a square [T, T] matrix (stage × task), got shape {mat.shape}")
    if mat.shape[0] < 2:
        raise ValueError(
            f"A needs at least two stages for a continual-learning metric, got T={mat.shape[0]}"
        )
    return require_fraction(mat, "A", allow_nan=True)


def matrix_stack(A: ArrayLike) -> list[NDArray[np.float64]]:
    """``A`` as a list of validated ``[T, T]`` matrices.

    One matrix for a ``[T, T]`` input; one per seed for a ``[S, T, T]`` stack (``S >= 1``), the
    shape P3.7 produces (one matrix per seed). Gates compute each statistic per seed and average
    the results: the maximum in :func:`per_task_loss` applied to a seed-mean matrix is biased
    low (the maximum of a mean is at most the mean of the maxima).
    """
    arr = np.asarray(A, dtype=np.float64)
    if arr.ndim == 2:
        return [_as_matrix(arr)]
    if arr.ndim != 3:
        raise ValueError(
            f"A must be [T, T] or [S, T, T] (seed × stage × task), got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("A is an empty stack: at least one seed is needed")
    return [_as_matrix(mat) for mat in arr]


def _measured(values: NDArray[np.float64], what: str, tasks: NDArray[np.intp]) -> None:
    missing = np.isnan(values)
    if missing.any():
        raise ValueError(f"{what} is unmeasured (NaN) for tasks {tasks[missing].tolist()}")


# --------------------------------------------------------------------------------------------------
# Plan §2.8 / paper §11
# --------------------------------------------------------------------------------------------------


def accuracy_matrix(results: dict[tuple[int, int], float], T: int) -> NDArray[np.float64]:
    """Assemble ``A[i, j]`` from ``{(stage, task): accuracy}``; NaN where unmeasured.

    Plan §2.8 / paper §11: ``A(i, j)`` = held-out accuracy on task ``j`` after learning stage ``i``,
    ``T`` stages. Keys are 0-based ``(i, j)`` with ``0 <= i, j < T`` (stage ``s`` of Appendix D is
    index ``s − 1``); a key outside that range or a value outside ``[0, 1]`` raises. Entries absent
    from ``results`` are NaN, which the metrics below treat as "not measured" and refuse to skip
    (P3.7 fills every entry; a truncated stream leaves NaN and is not gate-eligible).
    """
    if T < 1:
        raise ValueError(f"T={T} must be >= 1")
    A = np.full((T, T), np.nan, dtype=np.float64)
    for key, value in results.items():
        i = _as_index(key[0], f"results key {key!r}: stage")
        j = _as_index(key[1], f"results key {key!r}: task")
        if not (0 <= i < T and 0 <= j < T):
            raise ValueError(
                f"result key (stage, task)={key} is outside [0, {T}) "
                "(keys are 0-based: stage s is index s-1)"
            )
        A[i, j] = require_fraction(value, f"results[{key}]")
    return A


def average_final_accuracy(A: ArrayLike) -> float:
    """``AverageFinalAccuracy = (1/T) Σ_{j=1..T} A(T, j)`` (plan §2.8 / paper §11).

    The mean of the final row ``A[T−1, :]``; every task must be measured at the final stage.
    Fraction.
    """
    mat = _as_matrix(A)
    final = mat[-1]
    _measured(final, "final-stage accuracy", np.arange(mat.shape[0]))
    return float(final.mean())


def per_task_loss(A: ArrayLike) -> NDArray[np.float64]:
    """Accuracy lost on each earlier task by the final stage (the "no task loses > 5 pp" rule).

    Plan §2.8 / paper §11, for each task ``j = 1..T−1`` in paper indices::

        loss_j = max_{i ∈ 1..T−1} A(i, j) − A(T, j)

    The maximum runs over stages ``1..T−1`` only — never the final stage — so a task that is at
    its best after the final stage has a *negative* loss. The final task has no loss term (nothing
    was learned after it): the result has length ``T − 1`` and index ``j`` is the 0-based task.
    Every accuracy of an already-learned task (``A[i, j]`` with ``i >= j``) must be measured; an
    unmeasured one raises rather than lowering the maximum. Entries above the diagonal (a task
    before it was learned) may be NaN and are then ignored by the maximum. Fractions.
    ``forgetting`` is the mean of this vector; G3 (plan §8.1) requires every entry ``<= 0.05``.
    """
    mat = _as_matrix(A)
    T = mat.shape[0]
    tasks = np.arange(T - 1)
    earlier = mat[: T - 1, : T - 1]  # stages 0..T-2 × tasks 0..T-2
    final = mat[T - 1, : T - 1]
    learned = np.tri(T - 1, dtype=bool)  # i >= j: task j had been learned by stage i
    missing = np.isnan(earlier) & learned
    if missing.any():
        cells = list(zip(*(axis.tolist() for axis in np.nonzero(missing)), strict=True))
        raise ValueError(
            f"accuracy of already-learned tasks is unmeasured (NaN) at (stage, task) {cells}"
        )
    _measured(final, "final-stage accuracy", tasks)
    best_earlier = np.nanmax(earlier, axis=0)
    return np.asarray(best_earlier - final, dtype=np.float64)


def forgetting(A: ArrayLike) -> float:
    """Mean accuracy lost on earlier tasks by the final stage (plan §2.8 / paper §11)::

        Forgetting = (1/(T−1)) Σ_{j=1..T−1} ( max_{i ∈ 1..T−1} A(i, j) − A(T, j) )    # max to T−1

    Equal to ``mean(per_task_loss(A))``. Negative when earlier tasks are better after the final
    stage than at any earlier stage, because the maximum excludes the final stage by definition.
    Fraction (G3 asks for ``<= 0.02``, i.e. 2 pp).
    """
    return float(per_task_loss(A).mean())


def backward_transfer(A: ArrayLike) -> float:
    """``BackwardTransfer = (1/(T−1)) Σ_{j=1..T−1} ( A(T, j) − A(j, j) )`` (plan §2.8 / paper §11).

    How much later stages changed each earlier task relative to its accuracy right after it was
    learned; positive means later learning helped. Needs the diagonal and the final row measured.
    Fraction.
    """
    mat = _as_matrix(A)
    T = mat.shape[0]
    tasks = np.arange(T - 1)
    final = mat[T - 1, tasks]
    just_learned = mat[tasks, tasks]
    _measured(final, "final-stage accuracy", tasks)
    _measured(just_learned, "accuracy right after learning (A[j, j])", tasks)
    return float((final - just_learned).mean())


def forward_transfer(acc_new_before: ArrayLike, acc_control: ArrayLike) -> float:
    """``ForwardTransfer = mean_u ( acc_new_before_u − acc_control_u )`` (plan §2.8 / paper §11).

    Plan §2.8: "accuracy on a new task before training it, minus a same-budget model without earlier
    tasks". ``acc_new_before[u]`` is the accuracy on a new task before it is trained, for unit ``u``
    (a task, or task × seed); ``acc_control[u]`` is the accuracy of a same-budget control that saw
    no earlier task, aligned by unit. The companion number §2.8 asks to report alongside it —
    examples needed to reach a fixed target — is a count taken by the training harness, not a
    function of accuracies, and is not computed here. Fractions.
    """
    before = require_fraction(acc_new_before, "acc_new_before")
    control = require_fraction(acc_control, "acc_control")
    if before.shape != control.shape:
        raise ValueError(
            f"acc_new_before has shape {before.shape} but acc_control has shape {control.shape}; "
            "they are paired by unit"
        )
    if before.size == 0:
        raise ValueError("forward transfer needs at least one unit")
    return float((before - control).mean())


def new_task_gain(A: ArrayLike, stages: Sequence[int] | None = None) -> float:
    """Mean accuracy gained on each new task by learning it (the G3 "new-task gain" criterion).

    Per stage ``j`` (0-based, ``j >= 1``) and averaged over ``stages``::

        gain_j        = A[j, j] − A[j−1, j]        # after learning task j minus just before it
        new_task_gain = mean_{j ∈ stages} gain_j

    ``stages`` defaults to every stage ``1..T−1`` (stage 0 has no "before" row). Pass a subset when
    a stage is not trained: Appendix D withholds the training data of stage 8 (index 7; the
    composition is tested, not taught), so its gain is zero by construction and must be left
    out. Plan §8.1 G3 /
    paper §13 ask for "new-task gain ≥ 3 pp over a frozen control": compute this function for HSN
    and for the control with the same ``stages`` and compare the two
    (:func:`hsn.eval.gates.retention_gate`). P3.7 evaluates every stage after every stage, so
    ``A[j−1, j]`` exists. Fraction.
    """
    mat = _as_matrix(A)
    T = mat.shape[0]
    chosen = list(range(1, T)) if stages is None else [_as_index(s, "stage") for s in stages]
    if not chosen:
        raise ValueError("stages is empty: at least one stage is needed")
    outside = [stage for stage in chosen if stage < 1 or stage >= T]
    if outside:
        raise ValueError(
            f"stages {outside} are outside 1..{T - 1} (0-based; stage 0 has no earlier row)"
        )
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"stages lists a stage twice: {chosen}")
    idx = np.asarray(chosen, dtype=np.intp)
    after = mat[idx, idx]
    before = mat[idx - 1, idx]
    _measured(after, "accuracy right after learning (A[j, j])", idx)
    _measured(before, "accuracy just before learning (A[j-1, j])", idx)
    return float((after - before).mean())
