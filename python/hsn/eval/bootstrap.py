"""Paired bootstrap confidence intervals (plan §2.8–2.9; paper §12).

P0.7. The statistical plan bootstraps over *units* — tasks or repositories, paired by seed and data
stream — never over tokens (plan §2.8: "Bootstrap over tasks or repositories (not tokens), paired by
seed"). :class:`PairedSamples` holds one baseline and one candidate measurement per unit, aligned by
position and optionally labelled; :meth:`PairedSamples.from_mappings` joins two ``{unit: value}``
mappings on their keys, so a missing or extra seed raises instead of silently shifting the pairing
(chunk P0.7 pitfall: misaligned pairing across seeds).

:func:`bootstrap_mean` is the percentile bootstrap of a mean: resample the ``n`` units with
replacement ``n_boot`` times, take the mean of each resample, and report the ``alpha/2`` and
``1 − alpha/2`` quantiles. It is deterministic given ``seed``. The percentile method under-covers
for very small ``n`` (three seeds × few repositories); :attr:`CI.n_units` is reported so that
reduced certainty stays visible (plan P0.7 context: "three [seeds] for expensive confirmations
(report reduced certainty)").
"""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["CI", "PairedSamples", "bootstrap_mean", "paired_bootstrap"]

#: Rows of resample indices drawn per chunk. Fixed, so a result never depends on available memory;
#: consecutive ``Generator.integers`` draws are identical to one large draw, but this makes the
#: reproducibility a matter of definition rather than of that property.
_BOOT_CHUNK: int = 1_000


def _as_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """``values`` as a 1-D float64 array; raises on wrong rank, empty input or non-finite values."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional (one value per unit), got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError(f"{name} is empty: at least one unit is needed")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} contains NaN or infinite values; an unmeasured unit must be dropped explicitly"
        )
    return arr


@dataclass(frozen=True)
class CI:
    """A two-sided bootstrap confidence interval for a mean over units.

    ``estimate`` is the sample mean of the data (not the mean of the bootstrap means);
    ``[low, high]`` is the ``1 − alpha`` percentile interval from ``n_boot`` resamples of
    ``n_units`` units drawn with ``numpy.random.default_rng(seed)``. The "95 % paired CI" of plan
    §2.9 is ``alpha = 0.05``.
    """

    low: float
    high: float
    estimate: float
    alpha: float
    n_boot: int
    n_units: int
    seed: int


@dataclass(frozen=True, eq=False)
class PairedSamples:
    """One baseline and one candidate measurement per unit, aligned by position.

    A unit is a task or a repository crossed with a seed (plan §2.8), evaluated with the same seed
    and data stream in both arms; ``units`` optionally labels the positions (unique strings) so a
    report can show what was paired. Values must be finite; the unit of measure (accuracy fraction,
    latency, energy, perplexity) is whatever the caller measured and is checked by the gate that
    uses it. Build from ``{unit: value}`` mappings with :meth:`from_mappings` to have the pairing
    checked by key. Units are resampled independently of each other: the seeds of one task count
    as separate units (the unit the chunk's interface names), i.e. they are treated as
    exchangeable.
    """

    baseline: NDArray[np.float64]
    candidate: NDArray[np.float64]
    units: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        baseline = _as_vector(self.baseline, "baseline")
        candidate = _as_vector(self.candidate, "candidate")
        if baseline.shape != candidate.shape:
            raise ValueError(
                f"baseline has {baseline.size} units but candidate has {candidate.size}; "
                "paired samples need one value per unit on both sides"
            )
        if self.units is not None:
            units = tuple(str(unit) for unit in self.units)
            if len(units) != baseline.size:
                raise ValueError(f"{len(units)} unit labels given for {baseline.size} units")
            if len(set(units)) != len(units):
                raise ValueError("unit labels must be unique")
            object.__setattr__(self, "units", units)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "candidate", candidate)

    @property
    def n_units(self) -> int:
        return int(self.baseline.size)

    @classmethod
    def from_mappings[K: Hashable](
        cls, baseline: Mapping[K, float], candidate: Mapping[K, float]
    ) -> Self:
        """Join two ``{unit: value}`` mappings on their keys, e.g. ``{(task, seed): acc}``.

        Every key must be present on both sides: a seed or repository measured in one arm only
        raises (listing the keys) rather than being dropped or paired with its neighbour. Units are
        ordered by ``str(key)`` and that string becomes the label.
        """
        only_baseline = sorted((key for key in baseline if key not in candidate), key=str)
        only_candidate = sorted((key for key in candidate if key not in baseline), key=str)
        if only_baseline or only_candidate:
            raise ValueError(
                "paired samples must cover exactly the same units; "
                f"only in baseline: {only_baseline}; only in candidate: {only_candidate}"
            )
        keys = sorted(baseline, key=str)
        return cls(
            baseline=np.array([baseline[key] for key in keys], dtype=np.float64),
            candidate=np.array([candidate[key] for key in keys], dtype=np.float64),
            units=tuple(str(key) for key in keys),
        )


def bootstrap_mean(
    values: ArrayLike, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> CI:
    """Percentile bootstrap confidence interval of the mean of ``values`` over units.

    Formula (plan §2.9, paper §12 "95 % paired confidence interval"): with per-unit values
    ``d_1..d_n``, draw ``n_boot`` resamples ``d*_b`` of size ``n`` with replacement, let
    ``m*_b = mean(d*_b)``, and report ``[Q_{alpha/2}(m*), Q_{1−alpha/2}(m*)]`` (linearly
    interpolated quantiles); ``estimate = mean(d)``. The units are tasks or repositories × seed,
    never tokens (plan §2.8). At least two units are required; the interval of a constant sample is
    degenerate.
    """
    d = _as_vector(values, "values")
    n = int(d.size)
    if n < 2:
        raise ValueError(f"bootstrap needs at least two units, got {n}")
    if n_boot < 1:
        raise ValueError(f"n_boot={n_boot} must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha={alpha} must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, _BOOT_CHUNK):
        rows = min(_BOOT_CHUNK, n_boot - start)
        idx = rng.integers(0, n, size=(rows, n))
        means[start : start + rows] = d[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    return CI(
        low=float(low),
        high=float(high),
        estimate=float(d.mean()),
        alpha=float(alpha),
        n_boot=int(n_boot),
        n_units=n,
        seed=int(seed),
    )


def paired_bootstrap(
    a: ArrayLike, b: ArrayLike, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> CI:
    """Confidence interval of the mean paired difference ``a − b`` over units.

    ``a`` and ``b`` are aligned by unit (task or repository × seed, the same seed and data stream on
    both sides; plan §2.8): ``d_u = a_u − b_u`` for every unit ``u``, then :func:`bootstrap_mean` on
    ``d``. An interval above zero means ``a`` exceeds ``b``. Plan §2.9 / paper §12.
    """
    a_arr = _as_vector(a, "a")
    b_arr = _as_vector(b, "b")
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"a has {a_arr.size} units but b has {b_arr.size}; pairing is by unit")
    return bootstrap_mean(a_arr - b_arr, n_boot=n_boot, seed=seed, alpha=alpha)
