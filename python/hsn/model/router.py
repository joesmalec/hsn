"""Router scorer and its feature builder (P1.5): the fixed ``3d + 8`` per-cell feature layout and
the scorer that maps it to ``M`` unit logits.

Plan §2.5 and Appendix H.2/H.5. Every candidate (cell, unit) is scored from information available at
position ``t`` before any cell is updated (global rule 2). The dense reference (P1.5) and the sparse
model (P2.1) share this one scorer: in dense mode ``α = softmax(logits)`` over the units of every
cell, so the parameter count stays at Appendix C's 1,820,361 (``scorer`` row: ``(3d + 8) × M + M``,
1,608 for A.1); P2.1 adds selection (top-K, budgets, deadlines) beside it in this module. The scorer
is linear by default; ``model.scorer = mlp`` swaps in a two-layer :class:`~hsn.model.encoder.MLP2`
with hidden width ``model.scorer_hidden`` (the C.2 "two-layer scorer" row), which costs more per
position than the four selected cells execute (plan §2.5).

Feature layout (:data:`SCORER_FEATURE_NAMES`, width ``3d + 8``; 200 for A.1), per cell ``i``::

    [ u_t (d) ; ε_t (d) ; h[i, t, r−1] (d) ;
      age_i / age_max ; budget_remaining_fraction ; round_index / R_max ; context_id / n_ctx ;
      ema_utility_i ; ema_selected_i ; pos / unroll ; 1.0 ]

``u_t`` and ``ε_t`` are shared by every cell of a batch row; ``h`` is the cell's state from the
previous round (``state.h``); ``age``/``pos`` come from the
:class:`~hsn.model.state.CellGraphState`.
``budget_remaining_fraction`` is ``B_t`` minus what earlier rounds spent, as a fraction of the dense
step cost (``model.budget`` before round 1); ``round_index`` is 0-based (rounds already completed at
this position; always 0 while ``R_max = 1``), so the feature is ``round_idx / r_max``;
``context_id / n_ctx`` uses the raw context symbol id. ``ema_utility`` and ``ema_selected`` are the
P2.3 running averages and are **zero** until P2.3 passes an :class:`EMAFeatures`; ``pos / unroll``
normalises the session position by ``train.unroll``; the last feature is the constant 1.
``model.scorer_features.exclude`` zeroes the named blocks (an ablation) without changing the width,
so no parameter count changes; a name outside :data:`SCORER_FEATURE_NAMES` raises.

**Selection (P2.1).** :class:`Router` wraps the scorer and turns its logits into a
:class:`Selection` — ``K`` (cell, unit) slots per batch row with a per-row executed cost — under
plan §2.5::

    score_ij = U_ij − λ_cost · c_j          c from the profiled cost table (hsn.kernels.costs)
    S*       = argmax_S Σ score   s.t.   shared + Σ_{e∈S} (cell_overhead + c_e) ≤ B_t,  |S| ≤ K_max

The modes of :meth:`Router.select` (``model.routing``): ``topk`` — ``s_i = max_j score_ij``, the
top-``k_max`` cells by a stable sort (equal scores keep index order), unit ``argmax_j`` (the first
maximal unit); ``greedy_budget`` — every (i, j) pair sorted by score (ties by ``i·M + j``), a
pair is taken while its cell is unchosen and it fits the budget, skipped otherwise, stop at
``k_max``;
``random`` — uniform cells (without replacement) and units from the generator; ``fixed`` — a
preset list of cells with their ``argmax`` units; ``dense`` — every cell with soft ``α``
(the P1.5 behaviour; consumers execute ``softmax(logits)``). The budget ``B_t`` (a fraction of
the dense step, shared path included) is enforced by every sparse mode (plan Track T: "the
budget cannot be exceeded"): ``topk`` and ``random`` keep the longest prefix of their order that
fits, ``greedy_budget`` enforces it in the walk, ``fixed`` refuses a list that may not fit, and
``dense`` is the one mode that reports ``cost = 1.0`` whatever the budget. ``k_min`` is a static
property of the table: :meth:`Router.select` requires ``shared + k_min · (cell_overhead + max_j
c_j) ≤ B_t`` up front, so a ``k_min`` prefix always fits whatever units are chosen and no batch can
fail at run time. Scores are computed in fp32 whatever the logits' dtype (bf16 cannot resolve
``λ·c`` at A.1 magnitudes). The scorer's own cost is the ``scorer`` profiler segment
(:meth:`Router.forward`) and :attr:`Selection.scorer_cost`, never part of :attr:`Selection.cost`
(global rule 5). Routing outputs are indices, weights and costs and nothing else: they never
authorise data access, tool use or state loading (global rule 7; the static check is
``test_p2_1_selection_carries_no_authority``). :meth:`Router.deadline_exceeded` is the separate
wall-clock check serving uses (P5.2); it decides nothing itself.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Self, get_args

import torch
from torch import Tensor, nn

from hsn.config import ModelConfig, Routing
from hsn.kernels.costs import CostTable, cost_table_from_config
from hsn.model.embeddings import init_weight_
from hsn.model.encoder import MLP2
from hsn.model.state import CellGraphState
from hsn.profiler import Profiler
from hsn.train.memory_plan import SCORER_SCALAR_FEATURES

__all__ = [
    "BUDGET_SLACK",
    "SCORER_FEATURE_NAMES",
    "SELECTION_MODES",
    "EMAFeatures",
    "Router",
    "Scorer",
    "Selection",
    "SelectionMode",
]

#: The selection modes, the same Literal as ``model.routing`` (one source of truth).
SelectionMode = Routing
SELECTION_MODES: tuple[str, ...] = tuple(get_args(Routing))
#: Relative tolerance of the budget comparison: costs are accumulated in fp32 on the device, so
#: the exact all-cells case (``cost = 1.0``, ``budget = 1.0``) must not lose a cell to an ulp.
BUDGET_SLACK = 1e-5

#: Feature blocks in layout order: three width-``d`` blocks, then the eight scalars (Appendix H.2).
SCORER_FEATURE_NAMES: tuple[str, ...] = (
    "u",
    "eps",
    "h",
    "age",
    "budget",
    "round",
    "context",
    "ema_utility",
    "ema_selected",
    "pos",
    "one",
)
_VECTOR_FEATURES = ("u", "eps", "h")
_SCALAR_FEATURES = SCORER_FEATURE_NAMES[len(_VECTOR_FEATURES) :]
if len(_SCALAR_FEATURES) != SCORER_SCALAR_FEATURES:  # pragma: no cover - layout and P1.1 agree
    raise RuntimeError(
        f"the P1.5 layout has {len(_SCALAR_FEATURES)} scalar features but memory_plan counts "
        f"{SCORER_SCALAR_FEATURES}"
    )
_INDEX_DTYPES = (torch.int64, torch.int32)


@dataclass(frozen=True)
class EMAFeatures:
    """The P2.3 running averages per cell: ``utility`` (EMA of the counterfactual utility) and
    ``selected`` (EMA of the selection indicator), each ``[B, N]`` or ``[N]`` (broadcast over the
    batch). Until P2.3 no caller has them and the scorer sees zeros."""

    utility: Tensor
    selected: Tensor


class Scorer(nn.Module):
    """Per-cell features ``[B, N, 3d+8]`` and the map to unit logits ``[B, N, M]``.

    ``cfg`` supplies every width and the normalisers (``d``, ``n_cells``, ``n_units``, ``age_max``,
    ``r_max``, ``n_context``, ``scorer``, ``scorer_hidden``, ``scorer_features.exclude``,
    ``activation``, ``init``); ``unroll`` is ``train.unroll``, the normaliser of the position
    feature, which lives outside ``model.*`` and is therefore a required argument.
    """

    _excluded: Tensor  # bool [3d+8]: True at zeroed feature slots (non-persistent)

    def __init__(self, cfg: ModelConfig, *, unroll: int) -> None:
        super().__init__()
        if isinstance(unroll, bool) or not isinstance(unroll, int) or unroll < 1:
            raise ValueError(f"Scorer: unroll (train.unroll) must be an int >= 1, got {unroll!r}")
        self.d = cfg.d
        self.n_cells = cfg.n_cells
        self.n_units = cfg.n_units
        self.age_max = cfg.age_max
        self.r_max = cfg.r_max
        self.n_context = cfg.n_context
        self.unroll = unroll
        self.kind = cfg.scorer
        self.in_features = 3 * cfg.d + SCORER_SCALAR_FEATURES

        unknown = [name for name in cfg.scorer_features.exclude if name not in SCORER_FEATURE_NAMES]
        if unknown:
            raise ValueError(
                f"model.scorer_features.exclude names unknown features {unknown}; known: "
                f"{list(SCORER_FEATURE_NAMES)}"
            )
        self.exclude: tuple[str, ...] = tuple(dict.fromkeys(cfg.scorer_features.exclude))

        self.feature_slices: dict[str, slice] = {}
        start = 0
        for name in SCORER_FEATURE_NAMES:
            width = cfg.d if name in _VECTOR_FEATURES else 1
            self.feature_slices[name] = slice(start, start + width)
            start += width
        assert start == self.in_features

        self.net: nn.Linear | MLP2
        if cfg.scorer == "linear":
            self.net = nn.Linear(self.in_features, cfg.n_units)
            init_weight_(self.net.weight, cfg.init)
            nn.init.zeros_(self.net.bias)
        elif cfg.scorer == "mlp":
            self.net = MLP2(
                self.in_features, cfg.scorer_hidden, cfg.n_units, cfg.activation, init=cfg.init
            )
        else:  # pragma: no cover - ModelConfig.scorer is a Literal
            raise ValueError(f"model.scorer={cfg.scorer!r} is not supported (linear | mlp)")

        excluded = torch.zeros(self.in_features, dtype=torch.bool)
        for name in self.exclude:
            excluded[self.feature_slices[name]] = True
        self.register_buffer("_excluded", excluded, persistent=False)

    # ----------------------------------------------------------------------------------------------
    # Features
    # ----------------------------------------------------------------------------------------------

    def features(
        self,
        u: Tensor,
        eps: Tensor,
        state: CellGraphState,
        ctx: Tensor,
        budget_frac: float,
        round_idx: int,
        ema: EMAFeatures | None = None,
    ) -> Tensor:
        """The ``[B, N, 3d+8]`` feature tensor in the module's layout, in ``u``'s dtype.

        ``u``, ``eps`` are ``[B, d]`` (the dtype and device of ``state.h``); ``ctx`` the context ids
        ``[B]`` (int64/int32); ``budget_frac`` a finite number >= 0; ``round_idx`` an int in
        ``[0, r_max)``; ``ema`` the P2.3 averages or ``None`` (zeros). Shape, dtype and device
        mismatches raise.
        """
        B, N, d = state.batch_size, state.n_cells, state.width
        if N != self.n_cells or d != self.d:
            raise ValueError(
                f"state has N = {N}, d = {d}; this Scorer expects N = {self.n_cells}, d = {self.d}"
            )
        self._check_vector(u, "u", B, state)
        self._check_vector(eps, "eps", B, state)
        if not isinstance(ctx, Tensor):
            raise TypeError(f"ctx must be a torch.Tensor of context ids, got {type(ctx).__name__}")
        if ctx.dtype not in _INDEX_DTYPES:
            raise TypeError(f"ctx must be an int64 or int32 tensor of ids, got dtype {ctx.dtype}")
        if ctx.shape != (B,):
            raise ValueError(f"ctx must have shape [B] = {(B,)}, got {tuple(ctx.shape)}")
        if ctx.device != state.device:
            raise ValueError(f"ctx must live on {state.device}, got {ctx.device}")
        if (
            isinstance(budget_frac, bool)
            or not isinstance(budget_frac, int | float)
            or not math.isfinite(budget_frac)
            or budget_frac < 0.0
        ):
            raise ValueError(f"budget_frac must be a finite number >= 0, got {budget_frac!r}")
        if isinstance(round_idx, bool) or not isinstance(round_idx, int):
            raise TypeError(f"round_idx must be an int, got {round_idx!r}")
        if not 0 <= round_idx < self.r_max:
            raise ValueError(
                f"round_idx={round_idx} must lie in [0, r_max) = [0, {self.r_max}) (0-based: "
                "rounds already completed at this position)"
            )

        dtype, device = u.dtype, u.device
        age = state.age.to(dtype) / self.age_max  # [B, N]
        budget = torch.full((B, N), float(budget_frac), dtype=dtype, device=device)
        rnd = torch.full((B, N), round_idx / self.r_max, dtype=dtype, device=device)
        context = (ctx.to(dtype) / self.n_context)[:, None].expand(B, N)
        if ema is None:
            ema_utility = torch.zeros(B, N, dtype=dtype, device=device)
            ema_selected = torch.zeros(B, N, dtype=dtype, device=device)
        else:
            ema_utility = self._check_ema(ema.utility, "utility", B, N, state).to(dtype)
            ema_selected = self._check_ema(ema.selected, "selected", B, N, state).to(dtype)
        pos = (state.pos.to(dtype) / self.unroll)[:, None].expand(B, N)
        one = torch.ones(B, N, dtype=dtype, device=device)
        scalars = torch.stack(
            [age, budget, rnd, context, ema_utility, ema_selected, pos, one], dim=-1
        )  # [B, N, 8] in _SCALAR_FEATURES order
        feats = torch.cat(
            [u[:, None, :].expand(B, N, d), eps[:, None, :].expand(B, N, d), state.h, scalars],
            dim=-1,
        )
        if self.exclude:
            feats = feats.masked_fill(self._excluded, 0)
        return feats

    def forward(self, feats: Tensor) -> Tensor:
        """``feats [B, N, 3d+8] -> logits [B, N, M]``, one score per (cell, unit)."""
        if not isinstance(feats, Tensor):
            raise TypeError(f"feats must be a torch.Tensor, got {type(feats).__name__}")
        if feats.ndim != 3 or feats.shape[-1] != self.in_features:
            raise ValueError(
                f"feats must have shape [B, N, {self.in_features}], got {tuple(feats.shape)}"
            )
        out: Tensor = self.net(feats)
        return out

    # ----------------------------------------------------------------------------------------------
    # Accounting and validation
    # ----------------------------------------------------------------------------------------------

    def param_count(self) -> int:
        """Measured parameter count (``(3d+8)·M + M`` linear; ``_mlp(3d+8, scorer_hidden, M)``)."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"kind={self.kind!r}, in_features={self.in_features}, n_units={self.n_units}, "
            f"n_cells={self.n_cells}, age_max={self.age_max}, r_max={self.r_max}, "
            f"n_context={self.n_context}, unroll={self.unroll}, exclude={list(self.exclude)}"
        )

    @staticmethod
    def _check_vector(x: Tensor, name: str, B: int, state: CellGraphState) -> None:
        if not isinstance(x, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
        if x.shape != (B, state.width):
            raise ValueError(
                f"{name} must have shape [B, d] = {(B, state.width)}, got {tuple(x.shape)}"
            )
        if x.dtype != state.dtype:
            raise ValueError(f"{name} must have dtype {state.dtype} like the state, got {x.dtype}")
        if x.device != state.device:
            raise ValueError(f"{name} must live on {state.device}, got {x.device}")

    @staticmethod
    def _check_ema(x: Tensor, name: str, B: int, N: int, state: CellGraphState) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"EMAFeatures.{name} must be a torch.Tensor, got {type(x).__name__}")
        if not x.is_floating_point():
            raise ValueError(f"EMAFeatures.{name} must be floating point, got {x.dtype}")
        if x.device != state.device:
            raise ValueError(f"EMAFeatures.{name} must live on {state.device}, got {x.device}")
        if x.shape == (N,):
            return x[None, :].expand(B, N)
        if x.shape == (B, N):
            return x
        raise ValueError(
            f"EMAFeatures.{name} must have shape [N] = {(N,)} or [B, N] = {(B, N)}, got "
            f"{tuple(x.shape)}"
        )


# --------------------------------------------------------------------------------------------------
# Selection (P2.1)
# --------------------------------------------------------------------------------------------------


@dataclass
class Selection:
    """The routing decision for one round: ``K`` slots per batch row.

    ``cell_idx`` and ``unit_idx`` are int64 ``[B, K]``; ``alpha [B, K]`` (fp32) is the weight of
    the slot's (cell, unit): 1.0 on every valid slot of a hard mode, 0.0 on a padded slot, and in
    ``dense`` mode the softmax weight of the cell's argmax unit (dense consumers execute the full
    ``softmax(logits)`` over units; there ``K = N``, not ``k_max``). ``valid [B, K]`` (bool) is
    False for padded slots. ``logits [B, N, M]`` are the scorer's logits as given. ``cost [B]``
    (fp32) is the executed cost of the row in dense-step units — ``shared + Σ_valid (cell_overhead
    + unit[unit_idx])`` — detached accounting (P2.2 adds the differentiable expected cost);
    ``scorer_cost`` is the scorer's own cost per position, reported beside it; ``n_rounds_used`` is
    1 until P2.7.

    Padded slots hold cell 0 / unit 0 so that every index is in range on every backend, and they
    **must be dropped by** ``valid`` by every consumer: a padded cell 0 collides with a genuinely
    selected cell 0 (``CellGraphState.with_delta`` rejects duplicate indices), ``alpha = 0`` is not
    a no-op through the gate, and unit statistics (balance, active units) must count valid slots
    only. A ``Selection`` carries indices, weights and costs and nothing that could authorise
    anything (global rule 7).
    """

    cell_idx: Tensor
    unit_idx: Tensor
    alpha: Tensor
    valid: Tensor
    logits: Tensor
    cost: Tensor
    scorer_cost: float
    n_rounds_used: int

    def __post_init__(self) -> None:
        for name in ("cell_idx", "unit_idx", "alpha", "valid", "logits", "cost"):
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"Selection.{name} must be a torch.Tensor, got {type(value).__name__}"
                )
        if self.logits.ndim != 3 or not self.logits.is_floating_point():
            raise ValueError(
                f"Selection.logits must be a floating [B, N, M] tensor, got shape "
                f"{tuple(self.logits.shape)} and dtype {self.logits.dtype}"
            )
        B = int(self.logits.shape[0])
        if self.cell_idx.ndim != 2 or self.cell_idx.shape[0] != B:
            raise ValueError(
                f"Selection.cell_idx must have shape [B, K] with B = {B}, got "
                f"{tuple(self.cell_idx.shape)}"
            )
        K = int(self.cell_idx.shape[1])
        for name, dtype_ok in (
            ("cell_idx", self.cell_idx.dtype == torch.int64),
            ("unit_idx", self.unit_idx.dtype == torch.int64),
            ("alpha", self.alpha.is_floating_point()),
            ("valid", self.valid.dtype == torch.bool),
        ):
            value = getattr(self, name)
            if value.shape != (B, K):
                raise ValueError(
                    f"Selection.{name} must have shape [B, K] = {(B, K)}, got {tuple(value.shape)}"
                )
            if not dtype_ok:
                raise ValueError(
                    f"Selection.{name} has dtype {value.dtype}; see the class docstring"
                )
        if self.cost.shape != (B,) or not self.cost.is_floating_point():
            raise ValueError(
                f"Selection.cost must be a floating [B] = {(B,)} tensor, got shape "
                f"{tuple(self.cost.shape)} and dtype {self.cost.dtype}"
            )
        for name in ("cell_idx", "unit_idx", "alpha", "valid", "cost"):
            value = getattr(self, name)
            if value.device != self.logits.device:
                raise ValueError(
                    f"Selection.{name} lives on {value.device} but logits on {self.logits.device}"
                )
        if (
            isinstance(self.scorer_cost, bool)
            or not isinstance(self.scorer_cost, int | float)
            or not math.isfinite(self.scorer_cost)
            or self.scorer_cost < 0.0
        ):
            raise ValueError(
                f"Selection.scorer_cost must be a finite number >= 0, got {self.scorer_cost!r}"
            )
        if (
            isinstance(self.n_rounds_used, bool)
            or not isinstance(self.n_rounds_used, int)
            or self.n_rounds_used < 1
        ):
            raise ValueError(
                f"Selection.n_rounds_used must be an int >= 1, got {self.n_rounds_used!r}"
            )

    @property
    def k(self) -> int:
        """The number of slots ``K`` (``k_max`` in the sparse modes, ``N`` in ``dense``)."""
        return int(self.cell_idx.shape[1])

    def n_valid(self) -> Tensor:
        """Selected (non-padded) slots per row, int64 ``[B]``."""
        return self.valid.sum(dim=1)


class Router(nn.Module):
    """The P1.5 :class:`Scorer` plus budgeted selection (module docstring).

    ``scorer`` becomes a submodule (P2.5 hands ``HSN.scorer`` over to ``HSN.router.scorer``);
    ``costs`` is the :class:`~hsn.kernels.costs.CostTable` of this model shape (its ``N`` and ``M``
    must match the scorer's, otherwise the profile is stale and must be regenerated);
    ``lambda_cost`` is ``model.lambda_cost``. The table lives in the non-persistent fp32 buffer
    ``cost_table`` (it is hardware-specific and derived from config and profile, so it never rides
    in a checkpoint); ``cell_overhead``, ``shared_cost`` and ``scorer_cost`` are plain floats,
    ``cost_table_hash`` and ``cost_source`` the provenance the run manifest records.
    """

    cost_table: (
        Tensor  # [M] buffer (non-persistent; fp32 unless the module is cast, accounting is fp32)
    )

    def __init__(self, scorer: Scorer, costs: CostTable, *, lambda_cost: float) -> None:
        super().__init__()
        if not isinstance(scorer, Scorer):
            raise TypeError(f"Router needs a Scorer, got {type(scorer).__name__}")
        if not isinstance(costs, CostTable):
            raise TypeError(f"Router needs a CostTable, got {type(costs).__name__}")
        if costs.n_units != scorer.n_units or costs.n_cells != scorer.n_cells:
            raise ValueError(
                f"the cost table was taken for N = {costs.n_cells}, M = {costs.n_units} but the "
                f"scorer has N = {scorer.n_cells}, M = {scorer.n_units}; regenerate the cost "
                "profile (hsn.kernels.costs.profile_dense_costs) for this model shape"
            )
        if (
            isinstance(lambda_cost, bool)
            or not isinstance(lambda_cost, int | float)
            or not math.isfinite(lambda_cost)
            or lambda_cost < 0.0
        ):
            raise ValueError(
                f"lambda_cost (model.lambda_cost) must be a finite number >= 0, got {lambda_cost!r}"
            )
        self.scorer = scorer
        self.n_cells = scorer.n_cells
        self.n_units = scorer.n_units
        self.lambda_cost = float(lambda_cost)
        self.costs = costs
        self.cell_overhead = costs.cell_overhead
        self.shared_cost = costs.shared
        self.scorer_cost = costs.scorer
        self.cost_table_hash = costs.hash
        self.cost_source = costs.source
        self.max_pair_cost = costs.max_pair_cost()
        self._unit_costs: list[float] = [float(v) for v in costs.unit.tolist()]
        self.register_buffer("cost_table", costs.unit.to(torch.float32), persistent=False)

    @classmethod
    def from_config(cls, cfg: ModelConfig, *, unroll: int, scorer: Scorer | None = None) -> Self:
        """A router for ``cfg``: the scorer (built here unless given), the table of
        ``model.cost_profile`` (the uniform placeholder when ``null``) and ``model.lambda_cost``;
        ``unroll`` is ``train.unroll``. A sparse ``model.routing`` whose ``model.budget`` cannot
        hold ``model.k_min`` cells (or the whole ``model.fixed_route``) raises here, at
        construction."""
        if not isinstance(cfg, ModelConfig):
            raise TypeError(f"from_config needs a ModelConfig, got {type(cfg).__name__}")
        if scorer is None:
            scorer = Scorer(cfg, unroll=unroll)
        router = cls(scorer, cost_table_from_config(cfg), lambda_cost=cfg.lambda_cost)
        if cfg.routing == "fixed":
            router.require_affordable(cfg.budget, len(cfg.fixed_route), "model.fixed_route")
        elif cfg.routing != "dense":
            router.require_affordable(cfg.budget, cfg.k_min, "model.k_min")
        return router

    # ----------------------------------------------------------------------------------------------
    # Scoring
    # ----------------------------------------------------------------------------------------------

    def forward(
        self, feats: Tensor, *, profiler: Profiler | None = None, tokens: int | None = None
    ) -> Tensor:
        """``feats [B, N, 3d+8] -> logits [B, N, M]`` through the scorer, timed as the ``scorer``
        segment of ``profiler`` when one is given (``tokens`` defaults to ``B``, one position per
        batch row)."""
        if profiler is None:
            plain: Tensor = self.scorer(feats)
            return plain
        if not isinstance(feats, Tensor) or feats.ndim != 3:
            raise ValueError(
                "feats must be a [B, N, 3d+8] tensor, got "
                f"{tuple(feats.shape) if isinstance(feats, Tensor) else type(feats).__name__}"
            )
        n_tokens = int(feats.shape[0]) if tokens is None else tokens
        with profiler.segment("scorer", tokens=n_tokens):
            logits: Tensor = self.scorer(feats)
        return logits

    def scores(self, logits: Tensor) -> Tensor:
        """``U − λ_cost · c`` per (cell, unit), fp32 ``[B, N, M]``."""
        self._check_logits(logits)
        scores = logits.float()
        if self.lambda_cost > 0.0:
            scores = scores - self.lambda_cost * self.cost_table.float()
        return scores

    def cell_scores(self, logits: Tensor) -> Tensor:
        """``s_i = max_j score_ij``, fp32 ``[B, N]`` (the top-K key; P2.2's straight-through
        threshold reads it)."""
        return self.scores(logits).amax(dim=-1)

    # ----------------------------------------------------------------------------------------------
    # Selection
    # ----------------------------------------------------------------------------------------------

    def select(
        self,
        logits: Tensor,
        budget: float,
        k_max: int,
        k_min: int,
        mode: SelectionMode,
        fixed: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Selection:
        """Select up to ``k_max`` (cell, unit) slots per row within ``budget`` (module docstring).

        ``logits`` are the scorer's ``[B, N, M]`` on the router's device; ``budget`` is ``B_t`` as
        a fraction of the dense step (finite, >= 0); ``0 <= k_min <= k_max <= N`` with
        ``k_max >= 1``; ``fixed`` (int64/int32 ``[L]``, distinct cells in ``[0, N)``, ``k_min <= L
        <= k_max``) only with ``mode="fixed"``, where it is required; ``generator`` only with
        ``mode="random"`` (the global RNG when ``None``). Every sparse mode raises ``ValueError``
        up front when ``budget`` cannot hold ``k_min`` cells at the dearest unit (``fixed``: the
        whole list), so the result always has ``k_min <= n_valid <= k_max`` and ``cost <= budget``.
        """
        B, N, M = self._check_logits(logits)
        budget_f = self._check_budget(budget)
        self._check_k(k_max, k_min, N)
        if mode not in SELECTION_MODES:
            raise ValueError(f"mode must be one of {SELECTION_MODES}, got {mode!r}")
        if mode == "fixed":
            if fixed is None:
                raise ValueError(
                    "mode='fixed' needs the preset cells in `fixed` (model.fixed_route)"
                )
            fixed_cells = self._check_fixed(fixed, N, k_max, k_min, logits.device)
        elif fixed is not None:
            raise ValueError(f"`fixed` is only used with mode='fixed', got mode={mode!r}")
        if generator is not None and mode != "random":
            raise ValueError(f"`generator` is only used with mode='random', got mode={mode!r}")
        if mode == "dense":
            return self._select_dense(logits)

        self.require_affordable(budget_f, k_min, "k_min")
        scores = self.scores(logits)
        if mode == "topk":
            cells, units = self._topk(scores, k_max)
            cells, units, valid = self._truncate(cells, units, budget_f)
        elif mode == "greedy_budget":
            cells, units, valid = self._greedy(scores, budget_f, k_max)
        elif mode == "random":
            cells, units = self._random(B, N, M, k_max, generator, logits.device)
            cells, units, valid = self._truncate(cells, units, budget_f)
        else:  # fixed
            self.require_affordable(budget_f, int(fixed_cells.shape[0]), "the fixed route")
            cells, units, valid = self._fixed(scores, fixed_cells, k_max)
        return Selection(
            cell_idx=cells,
            unit_idx=units,
            alpha=valid.to(torch.float32),
            valid=valid,
            logits=logits,
            cost=self._executed_cost(units, valid),
            scorer_cost=self.scorer_cost,
            n_rounds_used=1,
        )

    def require_affordable(self, budget: float, k: int, what: str) -> None:
        """Raise ``ValueError`` unless ``k`` cells fit in ``budget`` whatever units they use:
        ``shared + k · (cell_overhead + max_j c_j) <= budget``."""
        needed = self.shared_cost + k * self.max_pair_cost
        if needed > budget * (1.0 + BUDGET_SLACK):
            raise ValueError(
                f"budget {budget!r} (a fraction of the dense step) cannot hold {what} = {k} "
                f"cells: the shared path costs {self.shared_cost:.6g} and the dearest (cell, unit) "
                f"execution {self.max_pair_cost:.6g}, so {k} cells need {needed:.6g}"
            )

    def deadline_exceeded(self, start_ns: int, deadline_ms: float | None) -> bool:
        """Whether more than ``deadline_ms`` of host wall-clock time has passed since
        ``start_ns`` (``time.monotonic_ns()`` at the start of the request). ``None`` (the config's
        ``model.deadline_ms: null``) is "no deadline" and gives False. No device synchronisation:
        after asynchronous launches this measures issue time, not completion. A check only; what
        happens when it returns True is the caller's (P5.2: a cheaper validated path or an
        explicit failure)."""
        if isinstance(start_ns, bool) or not isinstance(start_ns, int):
            raise TypeError(f"start_ns must be an int from time.monotonic_ns(), got {start_ns!r}")
        if deadline_ms is None:
            return False
        if (
            isinstance(deadline_ms, bool)
            or not isinstance(deadline_ms, int | float)
            or not math.isfinite(deadline_ms)
            or deadline_ms <= 0.0
        ):
            raise ValueError(
                f"deadline_ms must be a finite number > 0 or None, got {deadline_ms!r}"
            )
        return (time.monotonic_ns() - start_ns) / 1e6 > deadline_ms

    # -- the modes --------------------------------------------------------------------------------

    def _select_dense(self, logits: Tensor) -> Selection:
        B, N = int(logits.shape[0]), self.n_cells
        device = logits.device
        probs = torch.softmax(logits.float(), dim=-1)  # [B, N, M]
        units = logits.float().argmax(dim=-1)  # [B, N], the first maximal unit
        alpha = probs.gather(2, units.unsqueeze(-1)).squeeze(-1)
        cells = torch.arange(N, device=device).unsqueeze(0).expand(B, N).clone()
        valid = torch.ones(B, N, dtype=torch.bool, device=device)
        cost = torch.full((B,), self.costs.dense_cost(), dtype=torch.float32, device=device)
        return Selection(
            cell_idx=cells,
            unit_idx=units,
            alpha=alpha,
            valid=valid,
            logits=logits,
            cost=cost,
            scorer_cost=self.scorer_cost,
            n_rounds_used=1,
        )

    @staticmethod
    def _topk(scores: Tensor, k_max: int) -> tuple[Tensor, Tensor]:
        cell_scores = scores.amax(dim=-1)  # [B, N]
        # stable: equal scores keep index order (never torch.topk, whose tie order is unspecified)
        order = torch.sort(cell_scores, dim=1, descending=True, stable=True).indices[:, :k_max]
        units = scores.argmax(dim=-1).gather(1, order)
        return order, units

    def _truncate(
        self, cells: Tensor, units: Tensor, budget: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Keep the longest prefix of the slots (in their order) that fits the budget."""
        per_slot = self.cell_overhead + self.cost_table.float()[units]  # fp32 [B, K]
        cumulative = self.shared_cost + per_slot.cumsum(dim=1)
        valid = cumulative <= budget * (1.0 + BUDGET_SLACK)  # monotone: already a prefix
        return cells.masked_fill(~valid, 0), units.masked_fill(~valid, 0), valid

    def _greedy(self, scores: Tensor, budget: float, k_max: int) -> tuple[Tensor, Tensor, Tensor]:
        """The chunk's pair walk, on the host after one device-to-host copy."""
        B, N, M = scores.shape
        order = torch.sort(scores.reshape(B, N * M), dim=1, descending=True, stable=True).indices
        rows: list[list[int]] = order.tolist()
        limit = budget * (1.0 + BUDGET_SLACK)
        cells = [[0] * k_max for _ in range(B)]
        units = [[0] * k_max for _ in range(B)]
        valid = [[False] * k_max for _ in range(B)]
        for b, row in enumerate(rows):
            chosen: set[int] = set()
            spent = self.shared_cost
            n_taken = 0
            for pair in row:
                i, j = divmod(pair, M)
                if i in chosen:
                    continue
                pair_cost = self.cell_overhead + self._unit_costs[j]
                if spent + pair_cost > limit:
                    continue  # skip and continue: a cheaper unit of this cell may fit later
                chosen.add(i)
                cells[b][n_taken], units[b][n_taken], valid[b][n_taken] = i, j, True
                spent += pair_cost
                n_taken += 1
                if n_taken == k_max:
                    break
        device = scores.device
        return (
            torch.tensor(cells, dtype=torch.int64, device=device),
            torch.tensor(units, dtype=torch.int64, device=device),
            torch.tensor(valid, dtype=torch.bool, device=device),
        )

    @staticmethod
    def _random(
        B: int, N: int, M: int, k_max: int, generator: torch.Generator | None, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        sample_device = generator.device if generator is not None else device
        keys = torch.rand(B, N, generator=generator, device=sample_device)
        cells = torch.argsort(keys, dim=1, stable=True)[:, :k_max]  # a uniform permutation prefix
        units = torch.randint(0, M, (B, k_max), generator=generator, device=sample_device)
        return cells.to(device), units.to(device)

    @staticmethod
    def _fixed(scores: Tensor, fixed: Tensor, k_max: int) -> tuple[Tensor, Tensor, Tensor]:
        B = int(scores.shape[0])
        L = int(fixed.shape[0])
        device = scores.device
        cells = torch.zeros(B, k_max, dtype=torch.int64, device=device)
        units = torch.zeros(B, k_max, dtype=torch.int64, device=device)
        valid = torch.zeros(B, k_max, dtype=torch.bool, device=device)
        cells[:, :L] = fixed.unsqueeze(0).expand(B, L)
        units[:, :L] = scores.argmax(dim=-1).gather(1, cells[:, :L])
        valid[:, :L] = True
        return cells, units, valid

    def _executed_cost(self, units: Tensor, valid: Tensor) -> Tensor:
        per_slot = (self.cell_overhead + self.cost_table.float()[units]) * valid.to(torch.float32)
        return self.shared_cost + per_slot.sum(dim=1)

    # -- validation and accounting ----------------------------------------------------------------

    def _check_logits(self, logits: Tensor) -> tuple[int, int, int]:
        if not isinstance(logits, Tensor):
            raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}")
        if logits.ndim != 3 or not logits.is_floating_point():
            raise ValueError(
                f"logits must be a floating [B, N, M] tensor, got shape {tuple(logits.shape)} and "
                f"dtype {logits.dtype}"
            )
        B, N, M = (int(s) for s in logits.shape)
        if N != self.n_cells or M != self.n_units:
            raise ValueError(
                f"logits have N = {N}, M = {M}; this Router expects N = {self.n_cells}, "
                f"M = {self.n_units}"
            )
        if logits.device != self.cost_table.device:
            raise ValueError(
                f"logits live on {logits.device} but the router on {self.cost_table.device}; move "
                "the router with .to(device)"
            )
        return B, N, M

    @staticmethod
    def _check_budget(budget: float) -> float:
        if (
            isinstance(budget, bool)
            or not isinstance(budget, int | float)
            or not math.isfinite(budget)
            or budget < 0.0
        ):
            raise ValueError(f"budget must be a finite number >= 0, got {budget!r}")
        return float(budget)

    @staticmethod
    def _check_k(k_max: int, k_min: int, N: int) -> None:
        for name, value in (("k_max", k_max), ("k_min", k_min)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {value!r}")
        if not 1 <= k_max <= N:
            raise ValueError(f"k_max = {k_max} must lie in [1, N] = [1, {N}]")
        if not 0 <= k_min <= k_max:
            raise ValueError(f"k_min = {k_min} must lie in [0, k_max] = [0, {k_max}]")

    @staticmethod
    def _check_fixed(fixed: Tensor, N: int, k_max: int, k_min: int, device: torch.device) -> Tensor:
        if not isinstance(fixed, Tensor):
            raise TypeError(f"fixed must be a torch.Tensor of cell ids, got {type(fixed).__name__}")
        if fixed.dtype not in _INDEX_DTYPES:
            raise TypeError(
                f"fixed must be an int64 or int32 tensor of cell ids, got {fixed.dtype}"
            )
        if fixed.ndim != 1:
            raise ValueError(f"fixed must have shape [L], got {tuple(fixed.shape)}")
        cells: list[int] = fixed.tolist()
        L = len(cells)
        if not k_min <= L <= k_max:
            raise ValueError(
                f"fixed lists {L} cells; mode='fixed' needs k_min <= L <= k_max = "
                f"[{k_min}, {k_max}]"
            )
        if L < 1:
            raise ValueError("fixed must list at least one cell")
        bad = [i for i in cells if i < 0 or i >= N]
        if bad:
            raise ValueError(f"fixed has cell ids outside [0, {N}): {bad}")
        if len(set(cells)) != L:
            raise ValueError(f"fixed must list distinct cells, got {cells}")
        return fixed.to(device=device, dtype=torch.int64)

    def param_count(self) -> int:
        """Measured parameter count (the scorer's; selection has no parameters)."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"n_cells={self.n_cells}, n_units={self.n_units}, lambda_cost={self.lambda_cost}, "
            f"shared_cost={self.shared_cost:.6g}, cell_overhead={self.cell_overhead:.6g}, "
            f"scorer_cost={self.scorer_cost:.6g}, cost_source={self.cost_source!r}, "
            f"cost_table_hash={self.cost_table_hash[:16]}..."
        )
