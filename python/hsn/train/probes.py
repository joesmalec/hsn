"""Counterfactual utility probes (P2.3): targets for the scorer's predicted utility ``U``.

The scorer's ``U_ij`` is meant to predict the *reduction in downstream loss* from executing unit
``j`` on cell ``i`` at position ``t`` (plan §2.5; surprise is not usefulness). A probe measures that
reduction on a training window by running the model forward under ``no_grad`` from the same state,
once with the selection the policy made at ``t`` and once per candidate with that selection edited,
and comparing the task loss over a fixed horizon. These *counterfactual* probes are unrelated to the
*refresh probes* of P2.2 (``Selection.refreshed``), which force stale cells into a selection.

One probe (:func:`run_probe`; :func:`probe_window` is the trainer's entry point)::

    state_t   = the state before position t (probe_window: replay [0, t) with model.forward)
    baseline  = model.step at t with the policy's own selection S, then model.forward over
                [t+1, t+H)                                      L_S  = Σ CE over loss positions
    candidate e = (cell i, unit j), one rollout per candidate column, from the same state and the
                same RNG state:
        i not selected in S:  execute S ∪ {e} at t, continue    delta = L_S − L_{S ∪ e}
        (i, j) a slot of S:   execute S \\ {e} at t, continue    delta = L_{S \\ e} − L_S

``delta > 0`` means the candidate helps (the chunk's sign convention): an added candidate that
lowers the loss and a selected one whose removal raises it both get a positive target. ``CE`` is
the next-symbol cross-entropy summed over the ``loss_mask`` positions of ``[t, t + H)``, so a delta
is the change of the window's summed task loss; a row without a loss position in its horizon has no
target (the pair is invalid). The edit at ``t`` goes through
:meth:`hsn.model.router.Router.intercept` and :meth:`~hsn.model.router.Router.with_candidate`: the
model's own :meth:`~hsn.model.hsn.HSN.step` executes the edited selection, on the sparse round or
the masked round alike, and every later position runs the normal policy of the step (the training
schedule's gate, temperature and
exploration in training mode; position ``t`` itself never runs the dense fallback, because a dense
row does not execute the selection under test).

**Common random numbers.** Every rollout of a probe starts from the RNG state captured after the
prefix (:func:`hsn.train.checkpoint.capture_rng_state`: Python, NumPy, torch CPU, CUDA, MPS), so a
delta measures the candidate, not a different exploration draw, Gumbel sample or dense-fallback
row; on exit every generator is restored to its state on entry, so a probe leaves the training
run's random stream untouched. A probe never writes the state it is given, a weight or a buffer.

**What is probed** (:func:`probe_window`): the position ``t`` is drawn uniformly among the positions
whose horizon holds a loss position of some row (:func:`choose_probe_position`; answers are sparse,
so a uniform ``t`` would waste most probes), and :func:`sample_candidates` takes per row
``n_probe`` alternatives (unselected cells without replacement, each with a uniform unit) and the
``K`` selected slots of the baseline, except the slots P2.2's exploration and refresh probes put
there (``Selection.explored`` / ``refreshed``: the policy did not choose them). ``(−1, −1)`` marks
"no candidate". The draws come from the caller's generator, never the global RNG.

**Splits.** :func:`run_probe` and :func:`probe_window` raise unless ``batch_window["split"]`` is
``"train"``: probes see training targets only, and probe targets on evaluation data would
invalidate every later result. :func:`measure_utility_correlation` is the evaluation of a trained
scorer (the chunk's definition of done on dev probes): it reuses the same rollouts on ``dev`` (or
``train``) windows, refuses ``test``, and returns statistics only — nothing it computes reaches a
weight, the EMA features or a training target.

**Regression and EMA.** :class:`ProbeResult` holds the (features at ``t``, cell, unit, target)
pairs of a probe; :func:`utility_predictions` re-scores the detached features with the router
(gradient into the scorer only) for :func:`hsn.losses.utility_loss`, and :func:`cell_utility`
averages the targets per cell for :meth:`hsn.model.router.EMAFeatures.update`.

Rollouts run one after another, not batched across candidates: batching the candidates as extra
rows would cost about half as much per probe but cannot share the per-row random draws of
exploration, REINFORCE or random routing. Multi-round steps (P2.7) are refused: a probe edits the
single round of a position.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from hsn.data.dataset import ByteDataset
from hsn.model.hsn import HSN
from hsn.model.router import (
    CANDIDATE_ADD,
    CANDIDATE_NONE,
    CANDIDATE_REMOVE,
    SCORED_MODES,
    Router,
    Selection,
)
from hsn.model.state import CellGraphState
from hsn.train.checkpoint import capture_rng_state, restore_rng_state

__all__ = [
    "PROBE_SPLIT",
    "WINDOW_KEYS",
    "ProbeResult",
    "UtilityCorrelation",
    "cell_utility",
    "choose_probe_position",
    "classify_candidates",
    "measure_utility_correlation",
    "probe_window",
    "run_probe",
    "sample_candidates",
    "spearman",
    "utility_predictions",
    "window_split",
]

#: The only split training probes run on.
PROBE_SPLIT = "train"
#: The window tensors a probe reads (the :class:`~hsn.train.loop.StreamLoader` fields).
WINDOW_KEYS: tuple[str, ...] = ("tokens", "targets", "context", "loss_mask", "reset")
_INDEX_DTYPES = (torch.int64, torch.int32)


# --------------------------------------------------------------------------------------------------
# The result
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ProbeResult:
    """One probe of a window at position ``t`` with horizon ``H``.

    ``delta [B, C]`` (fp32) is the target of every candidate (0 where invalid); ``valid [B, C]``
    (bool) marks the pairs with a target (a candidate in the slot and a loss position in the row's
    horizon); ``selected [B, C]`` (bool) the candidates that were in the selection at ``t`` (removal
    probes; the others were added); ``candidates [B, C, 2]`` (int64, ``(cell, unit)``, ``−1`` for no
    candidate); ``features [B, N, 3d+8]`` the detached scorer features at ``t`` (the regression
    inputs, shared by every candidate of a row); ``loss_selected [B]`` (fp32) the baseline's summed
    horizon cross-entropy and ``loss_positions [B]`` (int64) the loss positions it covers.
    ``positions`` counts the input positions executed per row (prefix and every rollout) and
    ``rollouts`` the rollouts, the baseline included."""

    delta: Tensor
    valid: Tensor
    selected: Tensor
    candidates: Tensor
    features: Tensor
    loss_selected: Tensor
    loss_positions: Tensor
    t: int
    horizon: int
    positions: int
    rollouts: int

    @property
    def n_pairs(self) -> int:
        """The number of valid (candidate, target) pairs (one device-to-host copy)."""
        return int(self.valid.sum())

    def pairs(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """The valid pairs flattened: ``(row, cell, unit, target, selected)``, each ``[P]``."""
        rows, cols = self.valid.nonzero(as_tuple=True)
        return (
            rows,
            self.candidates[rows, cols, 0],
            self.candidates[rows, cols, 1],
            self.delta[rows, cols],
            self.selected[rows, cols],
        )


@dataclass(frozen=True)
class _Window:
    tokens: Tensor
    targets: Tensor
    context: Tensor
    loss_mask: Tensor
    reset: Tensor

    @property
    def B(self) -> int:  # noqa: N802 - the plan's shape symbol
        return int(self.tokens.shape[0])

    @property
    def T(self) -> int:  # noqa: N802 - the plan's shape symbol
        return int(self.tokens.shape[1])


@dataclass(frozen=True, eq=False)
class _Baseline:
    state_t: CellGraphState  # the state at t after the reset of position t
    rng: dict[str, Any]  # the RNG state every rollout starts from
    selection: Selection  # the policy's selection at t
    features: Tensor  # the scorer features at t (detached)
    loss: Tensor  # fp32 [B]
    n_loss: Tensor  # int64 [B]


# --------------------------------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------------------------------


def run_probe(
    model: HSN,
    state: CellGraphState,
    batch_window: Mapping[str, Any],
    t: int,
    candidates: Tensor,
    horizon: int,
) -> ProbeResult:
    """Probe ``candidates`` (int64 ``[B, C, 2]``, ``(cell, unit)`` pairs, ``(−1, −1)`` for none) at
    position ``t`` of ``batch_window`` over ``horizon`` positions (module docstring).

    ``state`` is the state *before* position ``t`` (the reset of position ``t`` is applied here);
    it is never mutated. ``batch_window`` maps :data:`WINDOW_KEYS` to ``[B, T]`` tensors on the
    model's device and ``"split"`` to ``"train"`` (anything else raises). Each candidate must be
    either an unselected cell (any unit) or a valid, scorer-chosen slot ``(cell, unit)`` of the
    policy's selection at ``t``; anything else raises. ``t + horizon <= T``. The model must be
    routed and, in training mode, past the soft warm-up (a soft gate has no hard counterfactual).
    Runs under ``no_grad`` and restores every RNG on exit."""
    _require_split(batch_window, (PROBE_SPLIT,), "run_probe")
    window = _prepare(model, state, batch_window, t, horizon)
    cand = _check_candidates(candidates, window.B, model)
    with torch.no_grad(), _rng_preserved():
        base = _baseline(model, state, window, t, horizon)
        return _rollouts(model, window, t, horizon, base, cand, prefix=0)


def probe_window(
    model: HSN,
    state: CellGraphState,
    batch_window: Mapping[str, Any],
    t: int,
    *,
    n_probe: int,
    horizon: int,
    generator: torch.Generator,
) -> ProbeResult:
    """The trainer's probe of one window (module docstring): replay positions ``[0, t)`` from the
    window-start ``state`` (never mutated) with ``model.forward``, take the policy's selection at
    ``t``, draw the candidates with :func:`sample_candidates` (``n_probe`` alternatives plus the
    selected slots, from ``generator``), and run :func:`run_probe`'s rollouts. The same split, model
    and position rules as :func:`run_probe`."""
    _require_split(batch_window, (PROBE_SPLIT,), "probe_window")
    window = _prepare(model, state, batch_window, t, horizon)
    with torch.no_grad(), _rng_preserved():
        before_t = _prefix(model, state, window, t)
        base = _baseline(model, before_t, window, t, horizon)
        cand = sample_candidates(base.selection, n_probe, generator)
        return _rollouts(model, window, t, horizon, base, cand, prefix=t)


def choose_probe_position(
    loss_mask: Tensor, horizon: int | None, generator: torch.Generator
) -> int | None:
    """A probe position ``t`` drawn uniformly from ``generator`` among the positions whose horizon
    ``[t, t + horizon)`` (``horizon=None``: to the end of the window) holds a loss position of some
    row and fits the window; ``None`` when the window has none. ``loss_mask`` is bool ``[B, T]``."""
    if not isinstance(loss_mask, Tensor) or loss_mask.dtype != torch.bool or loss_mask.ndim != 2:
        raise TypeError("loss_mask must be a bool torch.Tensor [B, T]")
    T = int(loss_mask.shape[1])
    any_row = loss_mask.any(dim=0).cpu()  # [T]
    if horizon is None:
        last = any_row.nonzero().flatten()
        if last.numel() == 0:
            return None
        positions = torch.arange(int(last.max()) + 1)
    else:
        _check_horizon(horizon, T)
        counts = torch.cat([torch.zeros(1, dtype=torch.int64), any_row.to(torch.int64).cumsum(0)])
        in_horizon = counts[horizon:] - counts[:-horizon]  # [T − H + 1]: loss positions per start
        positions = (in_horizon > 0).nonzero().flatten()
        if positions.numel() == 0:
            return None
    pick = int(torch.randint(int(positions.numel()), (1,), generator=generator))
    return int(positions[pick])


def sample_candidates(selection: Selection, n_probe: int, generator: torch.Generator) -> Tensor:
    """The candidates of a probe, int64 ``[B, n_probe + K, 2]`` on the selection's device (module
    docstring): ``n_probe`` alternatives per row — distinct cells outside the selection, uniform
    without replacement, each with a uniform unit; ``(−1, −1)`` once a row runs out of unselected
    cells — then the ``K`` slots of the selection, ``(−1, −1)`` for padded, explored and refreshed
    slots. A dense-fallback row gets no candidate. Draws come from ``generator`` (CPU)."""
    if isinstance(n_probe, bool) or not isinstance(n_probe, int) or n_probe < 0:
        raise ValueError(f"n_probe (train.n_probe) must be an int >= 0, got {n_probe!r}")
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise TypeError("generator must be a CPU torch.Generator")
    B, N, M = (int(s) for s in selection.logits.shape)
    device = selection.logits.device
    selected = selection.selected_cells().cpu()
    dense = selection.dense_fallback.cpu()
    keys = torch.rand(B, N, generator=generator).masked_fill(selected, 2.0)  # selected cells last
    order = torch.argsort(keys, dim=1, stable=True)
    take = min(n_probe, N)
    alt_cells = torch.full((B, n_probe), -1, dtype=torch.int64)
    alt_cells[:, :take] = order[:, :take]
    alt_units = torch.randint(0, M, (B, n_probe), generator=generator)
    free = (~selected).sum(dim=1)  # unselected cells per row
    keep = (torch.arange(n_probe)[None, :] < free[:, None]) & ~dense[:, None]
    alt_cells = alt_cells.masked_fill(~keep, -1)
    alt_units = alt_units.masked_fill(~keep, -1)
    chosen = (
        selection.valid
        & ~selection.explored
        & ~selection.refreshed
        & ~selection.dense_fallback[:, None]
    ).cpu()
    sel_cells = selection.cell_idx.cpu().masked_fill(~chosen, -1)
    sel_units = selection.unit_idx.cpu().masked_fill(~chosen, -1)
    cells = torch.cat([alt_cells, sel_cells], dim=1)
    units = torch.cat([alt_units, sel_units], dim=1)
    return torch.stack([cells, units], dim=-1).to(device)


def classify_candidates(selection: Selection, candidates: Tensor) -> tuple[Tensor, Tensor]:
    """``(kind, inconsistent)``, each ``[B, C]``: ``kind`` (int64) is
    :data:`~hsn.model.router.CANDIDATE_ADD` for a cell outside the selection,
    :data:`~hsn.model.router.CANDIDATE_REMOVE` for a valid slot ``(cell, unit)`` the scorer chose
    (neither explored nor refreshed), :data:`~hsn.model.router.CANDIDATE_NONE` otherwise;
    ``inconsistent`` (bool) marks the candidates that are neither none nor one of the two kinds (a
    selected cell with another unit, a flagged slot, a dense-fallback row)."""
    N = int(selection.logits.shape[1])
    cells, units = candidates[..., 0], candidates[..., 1]
    none = cells < 0
    safe = cells.clamp(0, N - 1)
    is_selected = selection.selected_cells().gather(1, safe)  # [B, C]
    slot = (selection.cell_idx[:, None, :] == safe[:, :, None]) & selection.valid[:, None, :]
    same_unit = slot & (selection.unit_idx[:, None, :] == units[:, :, None])
    flagged = slot & (selection.explored | selection.refreshed)[:, None, :]
    dense = selection.dense_fallback[:, None]
    add = ~none & ~is_selected & ~dense
    remove = ~none & same_unit.any(dim=-1) & ~flagged.any(dim=-1) & ~dense
    kind = torch.full_like(cells, CANDIDATE_NONE)
    kind = kind.masked_fill(add, CANDIDATE_ADD).masked_fill(remove, CANDIDATE_REMOVE)
    return kind, ~none & ~add & ~remove


def utility_predictions(router: Router, result: ProbeResult) -> tuple[Tensor, Tensor]:
    """``(U_pred, U_target)``, each ``[P, 1]`` fp32, for :func:`hsn.losses.utility_loss`: the
    router's logit ``U_ij`` re-scored from the probe's detached features (autograd into the scorer
    only) at every valid pair, and the pair's target."""
    if not isinstance(router, Router):
        raise TypeError(f"utility_predictions needs the Router, got {type(router).__name__}")
    rows, cells, units, target, _ = result.pairs()
    logits = router(result.features)  # [B, N, M]
    return logits[rows, cells, units].float()[:, None], target.float()[:, None]


def cell_utility(result: ProbeResult, n_cells: int) -> tuple[Tensor, Tensor]:
    """``(mean, measured)``, each ``[N]``: the mean target of the valid pairs on each cell (0 where
    there is none) and whether the cell has one, the inputs of
    :meth:`hsn.model.router.EMAFeatures.update`."""
    rows, cells, _, target, _ = result.pairs()
    device = result.delta.device
    sums = torch.zeros(n_cells, dtype=torch.float32, device=device).index_add_(0, cells, target)
    counts = torch.zeros(n_cells, dtype=torch.float32, device=device).index_add_(
        0, cells, torch.ones_like(target)
    )
    measured = counts > 0
    return torch.where(measured, sums / counts.clamp(min=1.0), torch.zeros_like(sums)), measured


def window_split(dataset: ByteDataset, doc: Tensor) -> str:
    """The split of the documents a window holds: ``doc`` is the window's ``[B, T]`` document
    index (``−1`` on padding) into ``dataset``, whose per-document meta carries ``"split"``. A
    window without a document, or one that mixes splits, raises."""
    if not isinstance(doc, Tensor) or doc.dtype != torch.int64:
        raise TypeError("doc must be the window's int64 document index tensor")
    ids: list[int] = torch.unique(doc[doc >= 0]).tolist()
    splits = {dataset.meta(i).get("split") for i in ids}
    if len(splits) != 1:
        raise ValueError(
            f"a probe window must hold documents of one split, got {sorted(map(str, splits))}"
        )
    split = splits.pop()
    if not isinstance(split, str):
        raise ValueError(f"document meta carries no split (got {split!r})")
    return split


# --------------------------------------------------------------------------------------------------
# Evaluation of a trained scorer (the chunk's definition of done)
# --------------------------------------------------------------------------------------------------


def spearman(x: Any, y: Any) -> float:
    """Spearman's rank correlation of two equally long 1-D sequences (average ranks for ties, then
    Pearson's correlation of the ranks); NaN when fewer than two values or either is constant."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"spearman: x has {a.size} values but y {b.size}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("spearman: the values must be finite")
    if a.size < 2:
        return math.nan
    ra, rb = _average_ranks(a), _average_ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    if denom == 0.0:
        return math.nan
    return float((ra * rb).sum()) / denom


def _average_ranks(a: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, Any]:
    """1-based ranks with ties at their average rank."""
    order = np.argsort(a, kind="mergesort")
    inverse = np.empty(a.size, dtype=np.intp)
    inverse[order] = np.arange(a.size)
    ordered = a[order]
    first = np.concatenate([[True], ordered[1:] != ordered[:-1]])
    dense = np.cumsum(first)[inverse]  # 1-based tie-group index of every value
    bounds = np.concatenate([np.nonzero(first)[0], [a.size]])
    ranks: np.ndarray[Any, Any] = 0.5 * (bounds[dense] + bounds[dense - 1] + 1)
    return ranks


@dataclass(frozen=True)
class UtilityCorrelation:
    """How well the scorer's ``U`` ranks the measured loss reductions (the chunk's definition of
    done): ``spearman`` is the pooled rank correlation over every valid pair with its 95 %
    bootstrap interval over windows (``ci_low``, ``ci_high``); ``spearman_within`` the mean
    within-position correlation over the (window, row) groups with at least three pairs
    (``n_groups``); ``spearman_added`` / ``spearman_selected`` the pooled correlation of each kind.
    ``horizon`` is the probe horizon (``None``: to the end of each window)."""

    spearman: float
    ci_low: float
    ci_high: float
    spearman_within: float
    n_groups: int
    spearman_added: float
    spearman_selected: float
    n_pairs: int
    n_added: int
    n_selected: int
    n_windows: int
    n_probed_windows: int
    zero_fraction: float
    delta_mean: float
    delta_std: float
    u_mean: float
    u_std: float
    horizon: int | None
    n_probe: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (None if isinstance(value, float) and not math.isfinite(value) else value)
            for name, value in self.__dict__.items()
        }


def measure_utility_correlation(
    model: HSN,
    windows: Iterable[Mapping[str, Any]],
    *,
    n_probe: int,
    horizon: int | None,
    seed: int = 0,
    n_bootstrap: int = 1000,
) -> UtilityCorrelation:
    """Evaluate the scorer against counterfactual probes on ``windows`` (module docstring).

    ``windows`` are consecutive loader windows (the :data:`WINDOW_KEYS` tensors on the model's
    device and ``"split"``, ``"dev"`` or ``"train"``; ``"test"`` raises), processed in order with
    the state carried from window to window as evaluation does. Per window one position is drawn
    by :func:`choose_probe_position` (``horizon=None``: to the end of the window) and probed with
    :func:`sample_candidates` (``n_probe`` alternatives plus the selected slots); every valid pair
    gives ``(U, delta)`` with ``U`` the scorer's logit at the probed (row, cell, unit). Draws come
    from a generator seeded with ``seed`` (the bootstrap from ``numpy`` with the same seed). The
    model is used in the mode it is in (eval for the definition of done); nothing is written."""
    generator = torch.Generator().manual_seed(seed)
    per_window: list[tuple[np.ndarray[Any, Any], ...]] = []
    n_windows = 0
    state: CellGraphState | None = None
    with torch.no_grad():
        for batch_window in windows:
            split = _require_split(
                batch_window, ("dev", PROBE_SPLIT), "measure_utility_correlation"
            )
            del split
            if state is None:
                state = model.init_state(int(batch_window["tokens"].shape[0]))
            window = _prepare(model, state, batch_window, 0, 1)
            n_windows += 1
            t = choose_probe_position(window.loss_mask, horizon, generator)
            if t is not None:
                h = window.T - t if horizon is None else horizon
                with _rng_preserved():
                    before_t = _prefix(model, state, window, t)
                    base = _baseline(model, before_t, window, t, h)
                    cand = sample_candidates(base.selection, n_probe, generator)
                    result = _rollouts(model, window, t, h, base, cand, prefix=t)
                rows, cells, units, target, selected = result.pairs()
                if int(rows.numel()) > 0:
                    logits = model.router(result.features).float()
                    u = logits[rows, cells, units]
                    per_window.append(
                        (
                            u.cpu().numpy(),
                            target.cpu().numpy(),
                            selected.cpu().numpy(),
                            rows.cpu().numpy(),
                        )
                    )
            _, state, _ = model.forward(
                window.tokens, window.context, state, window.reset
            )  # carry the state into the next window
    return _summarise(per_window, n_windows, horizon, n_probe, seed, n_bootstrap)


def _summarise(
    per_window: list[tuple[np.ndarray[Any, Any], ...]],
    n_windows: int,
    horizon: int | None,
    n_probe: int,
    seed: int,
    n_bootstrap: int,
) -> UtilityCorrelation:
    if per_window:
        u = np.concatenate([w[0] for w in per_window]).astype(np.float64)
        d = np.concatenate([w[1] for w in per_window]).astype(np.float64)
        sel = np.concatenate([w[2] for w in per_window]).astype(bool)
    else:
        u = d = np.zeros(0)
        sel = np.zeros(0, dtype=bool)
    within: list[float] = []
    for uw, dw, _, rw in per_window:
        for row in np.unique(rw):
            members = rw == row
            if members.sum() >= 3:
                rho = spearman(uw[members], dw[members])
                if math.isfinite(rho):
                    within.append(rho)
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    if len(per_window) >= 2:
        for _ in range(n_bootstrap):
            pick = rng.integers(0, len(per_window), size=len(per_window))
            ub = np.concatenate([per_window[i][0] for i in pick])
            db = np.concatenate([per_window[i][1] for i in pick])
            rho = spearman(ub, db)
            if math.isfinite(rho):
                boot.append(rho)
    return UtilityCorrelation(
        spearman=spearman(u, d),
        ci_low=float(np.percentile(boot, 2.5)) if boot else math.nan,
        ci_high=float(np.percentile(boot, 97.5)) if boot else math.nan,
        spearman_within=float(np.mean(within)) if within else math.nan,
        n_groups=len(within),
        spearman_added=spearman(u[~sel], d[~sel]),
        spearman_selected=spearman(u[sel], d[sel]),
        n_pairs=int(u.size),
        n_added=int((~sel).sum()),
        n_selected=int(sel.sum()),
        n_windows=n_windows,
        n_probed_windows=len(per_window),
        zero_fraction=float((d == 0.0).mean()) if d.size else math.nan,
        delta_mean=float(d.mean()) if d.size else math.nan,
        delta_std=float(d.std()) if d.size else math.nan,
        u_mean=float(u.mean()) if u.size else math.nan,
        u_std=float(u.std()) if u.size else math.nan,
        horizon=horizon,
        n_probe=n_probe,
        seed=seed,
    )


# --------------------------------------------------------------------------------------------------
# The rollouts
# --------------------------------------------------------------------------------------------------


class _Edit:
    """The edit of one candidate column: the *baseline's* selection with the column's candidates
    (so every rollout edits exactly the selection the candidates were classified against)."""

    def __init__(
        self, router: Router, base: Selection, cells: Tensor, units: Tensor, kind: Tensor
    ) -> None:
        self.router, self.base, self.cells, self.units, self.kind = router, base, cells, units, kind

    def __call__(self, selection: Selection) -> Selection:
        del selection  # the same decision as the baseline's: same state, weights and RNG state
        return self.router.with_candidate(self.base, self.cells, self.units, self.kind)


@contextmanager
def _rng_preserved() -> Iterator[None]:
    saved = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(saved)


def _prefix(model: HSN, state: CellGraphState, window: _Window, t: int) -> CellGraphState:
    """The state before position ``t``: positions ``[0, t)`` under the normal policy."""
    if t == 0:
        return state
    _, before_t, _ = model.forward(
        window.tokens[:, :t], window.context[:, :t], state, window.reset[:, :t]
    )
    return before_t


def _rollout(
    model: HSN,
    state_t: CellGraphState,
    window: _Window,
    t: int,
    horizon: int,
    edit: _Edit | None,
) -> tuple[Tensor, Selection, Tensor]:
    """Position ``t`` with the (edited) selection, then ``[t+1, t+H)`` under the normal policy:
    ``(summed horizon cross-entropy [B], the executed selection at t, the features at t)``."""
    with model.router.intercept(edit) as tap:
        out = model.step(window.tokens[:, t], window.context[:, t], state_t)
    if len(tap.selections) != 1 or len(tap.features) != 1:
        raise NotImplementedError(
            f"a probe edits the single routed round of a position; position {t} ran "
            f"{len(tap.selections)} selections (multi-round steps arrive with P2.7)"
        )
    loss = _ce_sum(
        out.logits[:, None, :], window.targets[:, t : t + 1], window.loss_mask[:, t : t + 1]
    )
    if horizon > 1:
        rest = slice(t + 1, t + horizon)
        logits, _, _ = model.forward(
            window.tokens[:, rest], window.context[:, rest], out.state, window.reset[:, rest]
        )
        loss = loss + _ce_sum(logits, window.targets[:, rest], window.loss_mask[:, rest])
    return loss, tap.selections[0], tap.features[0]


def _baseline(
    model: HSN, state: CellGraphState, window: _Window, t: int, horizon: int
) -> _Baseline:
    state_t = state.reset(window.reset[:, t])
    rng = capture_rng_state()
    loss, selection, features = _rollout(model, state_t, window, t, horizon, None)
    n_loss = window.loss_mask[:, t : t + horizon].sum(dim=1)
    return _Baseline(state_t, rng, selection, features, loss, n_loss)


def _rollouts(
    model: HSN,
    window: _Window,
    t: int,
    horizon: int,
    base: _Baseline,
    candidates: Tensor,
    *,
    prefix: int,
) -> ProbeResult:
    kind, inconsistent = classify_candidates(base.selection, candidates)
    if bool(inconsistent.any()):
        raise ValueError(
            "candidates must be unselected cells or scorer-chosen slots (cell, unit) of the "
            f"selection at t = {t}; {int(inconsistent.sum())} are neither (a selected cell with "
            "another unit, an explored or refreshed slot, or a dense-fallback row)"
        )
    B, C = int(candidates.shape[0]), int(candidates.shape[1])
    delta = torch.zeros(B, C, dtype=torch.float32, device=candidates.device)
    active: list[bool] = (kind != CANDIDATE_NONE).any(dim=0).tolist()
    rollouts = 1
    for c in range(C):
        if not active[c]:
            continue
        restore_rng_state(base.rng)  # common random numbers
        k = kind[:, c]
        edit = _Edit(model.router, base.selection, candidates[:, c, 0], candidates[:, c, 1], k)
        loss_c, _, _ = _rollout(model, base.state_t, window, t, horizon, edit)
        rollouts += 1
        gain = torch.where(k == CANDIDATE_ADD, base.loss - loss_c, loss_c - base.loss)
        delta[:, c] = torch.where(k == CANDIDATE_NONE, torch.zeros_like(gain), gain)
    valid = (kind != CANDIDATE_NONE) & (base.n_loss > 0)[:, None]
    return ProbeResult(
        delta=delta.masked_fill(~valid, 0.0),
        valid=valid,
        selected=kind == CANDIDATE_REMOVE,
        candidates=candidates,
        features=base.features,
        loss_selected=base.loss,
        loss_positions=base.n_loss,
        t=t,
        horizon=horizon,
        positions=prefix + horizon * rollouts,
        rollouts=rollouts,
    )


def _ce_sum(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """``Σ_s CE(logits[:, s], targets[:, s])`` over the ``mask`` positions, fp32 ``[B]``."""
    B, S, V = (int(n) for n in logits.shape)
    safe = targets.masked_fill(~mask, 0)  # padding targets may lie outside the vocabulary
    ce = F.cross_entropy(logits.float().reshape(B * S, V), safe.reshape(B * S), reduction="none")
    return torch.where(mask, ce.reshape(B, S), torch.zeros_like(ce.reshape(B, S))).sum(dim=1)


# --------------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------------


def _require_split(batch_window: Mapping[str, Any], allowed: tuple[str, ...], what: str) -> str:
    if not isinstance(batch_window, Mapping):
        raise TypeError(
            f"{what}: batch_window must be a mapping, got {type(batch_window).__name__}"
        )
    split = batch_window.get("split")
    if split not in allowed:
        raise ValueError(
            f"{what} runs on the {' / '.join(allowed)} split only, got batch_window['split'] = "
            f"{split!r}: probes see training targets only (P2.3), and the test split is reserved "
            "for gate scripts"
        )
    assert isinstance(split, str)
    return split


def _prepare(
    model: HSN, state: CellGraphState, batch_window: Mapping[str, Any], t: int, horizon: int
) -> _Window:
    """Check the model, the state and the window; return the window's tensors."""
    if not isinstance(model, HSN):
        raise TypeError(f"probes run on the HSN module, got {type(model).__name__}")
    routing = model.cfg.routing
    if routing == "dense":
        raise ValueError(
            "model.routing=dense has no selection to probe: every cell executes with soft unit "
            "weights; counterfactual probes need a routed mode (topk, greedy_budget, random, fixed)"
        )
    if (
        model.training
        and routing in SCORED_MODES
        and model.schedule.gate(model.train_step) == "soft"
    ):
        raise ValueError(
            f"step {model.train_step} is in the soft warm-up (train.soft_until_step): a soft gate "
            "executes every cell with a fractional weight and has no hard counterfactual"
        )
    if not isinstance(state, CellGraphState):
        raise TypeError(f"state must be a CellGraphState, got {type(state).__name__}")
    tensors: dict[str, Tensor] = {}
    shape: tuple[int, ...] | None = None
    for key in WINDOW_KEYS:
        value = batch_window.get(key)
        if not isinstance(value, Tensor):
            raise TypeError(f"batch_window[{key!r}] must be a torch.Tensor")
        if shape is None:
            shape = tuple(value.shape)
        if value.ndim != 2 or tuple(value.shape) != shape:
            raise ValueError(
                f"batch_window[{key!r}] must be [B, T] = {shape}, got {tuple(value.shape)}"
            )
        if value.device != model.device:
            raise ValueError(
                f"batch_window[{key!r}] lives on {value.device}, the model on {model.device}"
            )
        if key in ("loss_mask", "reset"):
            if value.dtype != torch.bool:
                raise ValueError(f"batch_window[{key!r}] must be bool, got {value.dtype}")
        elif value.dtype not in _INDEX_DTYPES:
            raise ValueError(f"batch_window[{key!r}] must be int64 or int32, got {value.dtype}")
        tensors[key] = value
    assert shape is not None
    if state.batch_size != shape[0]:
        raise ValueError(f"state has B = {state.batch_size}, the window {shape[0]}")
    _check_horizon(horizon, shape[1])
    if isinstance(t, bool) or not isinstance(t, int) or not 0 <= t <= shape[1] - horizon:
        raise ValueError(
            f"t = {t!r} must be an int with 0 <= t and t + horizon <= T (t + {horizon} <= "
            f"{shape[1]})"
        )
    return _Window(
        tokens=tensors["tokens"],
        targets=tensors["targets"].to(torch.int64),
        context=tensors["context"],
        loss_mask=tensors["loss_mask"],
        reset=tensors["reset"],
    )


def _check_horizon(horizon: int, T: int) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= T:
        raise ValueError(
            f"horizon (train.probe_horizon) must be an int in [1, T = {T}], got {horizon!r}"
        )


def _check_candidates(candidates: Tensor, B: int, model: HSN) -> Tensor:
    N, M = model.cfg.n_cells, model.cfg.n_units
    if not isinstance(candidates, Tensor) or candidates.dtype != torch.int64:
        raise TypeError("candidates must be an int64 torch.Tensor [B, C, 2]")
    if candidates.ndim != 3 or candidates.shape[0] != B or candidates.shape[2] != 2:
        raise ValueError(f"candidates must be [B = {B}, C, 2], got {tuple(candidates.shape)}")
    if candidates.device != model.device:
        raise ValueError(f"candidates live on {candidates.device}, the model on {model.device}")
    cells, units = candidates[..., 0], candidates[..., 1]
    none = (cells == -1) & (units == -1)
    in_range = (cells >= 0) & (cells < N) & (units >= 0) & (units < M)
    if not bool((none | in_range).all()):
        raise ValueError(
            f"every candidate must be (cell in [0, {N}), unit in [0, {M})) or (-1, -1) for none"
        )
    return candidates
