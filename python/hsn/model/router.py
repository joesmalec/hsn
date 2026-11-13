"""Router scorer and its feature builder (P1.5): the fixed ``3d + 8`` per-cell feature layout and
the scorer that maps it to ``M`` unit logits.

Plan §2.5 and Appendix H.2/H.5. Every candidate (cell, unit) is scored from information available at
position ``t`` before any cell is updated (global rule 2). The dense reference (P1.5) and the sparse
model (P2.1) share this one scorer: in dense mode ``α = softmax(logits)`` over the units of every
cell, so the scorer adds nothing beyond Appendix C's ``scorer`` row (``(3d + 8) × M + M``, 1,608
for A.1); P2.1 adds selection (top-K, budgets, deadlines) beside it in this module. The scorer
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

**Selection gradients (P2.2).** The decisions above are indices; nothing trains through them. The
keyword arguments of :meth:`Router.select` add the *gradient path* and the training controls of
plan §6 without changing a decision (``gate=None`` is exactly P2.1). The executor multiplies with
two tensors the selection carries, both fp32 and both exactly hard in the hard modes::

    cell_mask [B, N]        the weight of each cell's update       (1 selected / 0 not, forward)
    unit_weights [B, N, M]  the mixing weights over units per cell (one-hot of the chosen unit)
    alpha [B, K]            = cell_mask[cell] · unit_weights[cell, unit] on the valid slots

``gate="ste"`` (straight-through): ``mask = hard + (soft − soft.detach())`` with ``soft =
sigmoid((s − θ) / τ)``, ``s_i = max_j score_ij`` and θ the **detached** threshold; ``w = onehot +
(softmax(score / τ) − softmax(score / τ).detach())``. θ is the midpoint between the lowest selected
and the highest unselected cell score of the scored decision (before the probes below), so that
``hard == (soft > 0.5)`` exactly for ``topk`` (for ``greedy_budget`` the selected set need not be a
top set and the sigmoid is a surrogate) and K = N gives an all-ones mask (no unselected cell:
θ = −∞); the chunk's ``s_(K)`` itself would pin the K-th cell at 0.5 and make soft gating with
K = N differ from dense. ``gate="soft"`` (the warm-up, ``train.soft_until_step``): ``cell_mask =
soft`` and ``unit_weights = softmax(score / τ)``, every cell executes and ``cost`` is the dense
cost (soft training is charged to the training budget, paper §6); the controls stay hard.
``gate="reinforce"``: the cells are a
sample without replacement from ``softmax(s / τ)`` (Gumbel-top-k through the same stable sort and
budget prefix; ``topk`` only), ``log_prob`` is its Plackett–Luce log-probability (differentiable;
the trainer adds ``−log_prob · (R − baseline)``), the cell mask is hard (the cells train through
``log_prob`` alone) and the units keep the straight-through weights (the chunk's REINFORCE line
covers the cells; a hybrid, so an STE-versus-REINFORCE comparison is a comparison of the cell
estimators).
The unit softmax uses the *scores* ``U − λc`` so that its argmax is the hard unit (identical to the
chunk's ``softmax(logits / τ)`` at ``λ_cost = 0``). ``random`` takes no gradient (nothing it decides
depends on the scorer); ``fixed`` takes the unit gradient only. ``expected_cost`` is the
differentiable ``shared + Σ_i mask_i (overhead + Σ_j w_ij c_j)`` (the cost term of P2.4); it equals
``cost`` in the hard modes.

**Routing controls (P2.2)**, post-processing of a *scored* selection (``topk``, ``greedy_budget``;
``random`` and ``fixed`` are controls and take neither), recorded as flags so that counterfactual
probes (P2.3) and SAN statistics (P4.6) can exclude them: **refresh probes** force cells whose
``age > age_max`` (``model.age_max``) into the selection, stalest first, in place of the
lowest-scoring valid slots (``refreshed``); **exploration** replaces, with probability ε per row,
one valid non-refreshed slot (uniform) by a uniform unselected cell (``explored``); both take the
new cell's argmax unit. The row's cost is recomputed and a row that would exceed the budget reverts
to its scored selection (impossible with uniform per-unit costs, i.e. every profiled table; the
revert drops both probes of the row). The draws come from the CPU generator (``generator`` or the
global RNG, which checkpoints replay) as one ``[B, N + 2]`` tensor per position moved to the device:
no host loop and no device-to-host synchronisation (one host-to-device copy). There is no cap on
the probes: an ``age_max`` small against ``N / K`` turns every slot into a probe (round-robin
routing), which is the configuration's choice. **Dense fallback** (``dense_rows``, drawn
by :meth:`hsn.model.hsn.HSN.forward` with probability ``train.p_dense``): the row executes the P1.5
dense step (``cell_mask = 1``, ``unit_weights = softmax(logits)``), is charged ``cost =`` the dense
cost and flagged ``dense_fallback``; its ``cell_idx`` / ``unit_idx`` are the hard decision it did
not run. The decisions, the budget guarantee and the deterministic orders of P2.1 are unchanged.

**Counterfactual support (P2.3).** The scorer owns the running averages it reads: two persistent
buffers ``ema_utility [N]`` and ``ema_selected [N]``, zero at initialisation, read by
:meth:`Scorer.features` whenever no explicit :class:`EMAFeatures` is passed (so a model whose
averages were never updated sees exactly the zeros it saw before P2.3). :meth:`EMAFeatures.update`
is the pure EMA step and :meth:`Scorer.load_ema` writes its result back; only the P2.3 trainer does
that, once per window, when ``train.lambda_u > 0``. A state dict written before the buffers existed
(``Scorer`` version 1) loads with zeros for them; a version-2 dict without them is refused. The
counterfactual probes of :mod:`hsn.train.probes` (distinct from the refresh probes above) execute
an *edited* selection through the unchanged :meth:`hsn.model.hsn.HSN.step`:
:meth:`Router.intercept` arms a :class:`RouterTap` that records the features the scorer saw and the
selections :meth:`Router.select` returned, and applies an optional edit to each selection before it
is returned; :meth:`Router.with_candidate` is that edit: one candidate (cell, unit) per row added to
the selection (an unselected cell written with the given unit) or removed from it (a valid slot
dropped), with ``cell_mask``, ``unit_weights``, ``alpha`` and ``cost`` kept consistent so that the
sparse and the masked round execute the same counterfactual. An edited selection may exceed the
budget: it is a training probe, charged to the training budget, never a routing decision.

**Static K (P2.8).** Every sparse mode returns exactly ``K = k_max`` (``model.k_select``) slots
per row, whatever the budget, ``k_min`` or gate: slots that are not selected (a budget-truncated
suffix, a short fixed route, a greedy walk that stopped early) are padded with ``valid=False``,
cell 0, unit 0 and ``α = 0``. Cell 0 is the in-range sentinel: every executor ignores a padded
slot's indices (:meth:`hsn.model.cell.PDGCell.step_sparse` zeroes them before the gathers and
:func:`hsn.kernels.native.gated_delta_update` routes them to a discarded row), so the shapes of a
position never depend on the data and a compiled window or a CUDA Graph can capture it
(``model.static_k = true``, the only supported value: :meth:`Router.from_config` refuses
``false``). ``dense`` mode has ``K = N``, and only the P2.3 probes' :meth:`Router.with_candidate`
widens a selection by one column. The dense step's cost (:meth:`CostTable.dense_cost`) is a
Python float read once at construction, so a position never reduces the CPU cost table.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Self, get_args

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import GateMode, ModelConfig, Routing
from hsn.data.synthetic import BOS
from hsn.kernels.costs import CostTable, cost_table_from_config
from hsn.model.embeddings import init_weight_, parse_init
from hsn.model.encoder import MLP2
from hsn.model.state import CellGraphState
from hsn.profiler import Profiler
from hsn.train.memory_plan import SCORER_SCALAR_FEATURES

__all__ = [
    "ADDRESS_FEATURE_NAMES",
    "BUDGET_SLACK",
    "CANDIDATE_ADD",
    "CANDIDATE_NONE",
    "CANDIDATE_REMOVE",
    "GATE_MODES",
    "SCORED_MODES",
    "SCORER_FEATURE_NAMES",
    "SELECTION_MODES",
    "EMAFeatures",
    "Router",
    "RouterTap",
    "Scorer",
    "Selection",
    "SelectionMode",
]

#: The selection modes, the same Literal as ``model.routing`` (one source of truth).
SelectionMode = Routing
SELECTION_MODES: tuple[str, ...] = tuple(get_args(Routing))
#: The gate modes of ``train.gate_mode`` (P2.2); ``None`` in :meth:`Router.select` is "no gradient".
GATE_MODES: tuple[str, ...] = tuple(get_args(GateMode))
#: The modes whose decision follows the scores: the gate's cell mask, the refresh probes and the
#: exploration apply to these; ``random`` and ``fixed`` are controls and ``dense`` has no selection.
SCORED_MODES: tuple[str, ...] = ("topk", "greedy_budget")
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
#: The candidate kinds of :meth:`Router.with_candidate` (P2.3), one per row.
CANDIDATE_NONE, CANDIDATE_ADD, CANDIDATE_REMOVE = 0, 1, 2
_EMA_BUFFERS = ("ema_utility", "ema_selected")
ADDRESS_FEATURE_NAMES: tuple[str, str] = ("a_cur", "a_prev")
_KEY_ADDRESS_TAG = 0x48534E4135


def _check_decay(decay: float) -> float:
    if (
        isinstance(decay, bool)
        or not isinstance(decay, int | float)
        or not math.isfinite(decay)
        or not 0.0 <= decay <= 1.0
    ):
        raise ValueError(f"decay (train.ema_decay) must be a number in [0, 1], got {decay!r}")
    return float(decay)


@dataclass(frozen=True)
class EMAFeatures:
    """The P2.3 running averages per cell: ``utility`` (EMA of the counterfactual utility) and
    ``selected`` (EMA of the selection indicator), each ``[B, N]`` or ``[N]`` (broadcast over the
    batch). :attr:`Scorer.ema` is the ``[N]`` view of the scorer's own buffers; ``ema_utility`` and
    ``ema_selected`` are the chunk's names for the two fields."""

    utility: Tensor
    selected: Tensor

    @property
    def ema_utility(self) -> Tensor:
        return self.utility

    @property
    def ema_selected(self) -> Tensor:
        return self.selected

    def update(
        self,
        selection: Tensor,
        realised_utility: Tensor | None,
        decay: float,
        *,
        measured: Tensor | None = None,
    ) -> EMAFeatures:
        """One EMA step, ``ema ← decay · ema + (1 − decay) · x``, returned as new ``[N]`` tensors
        (this object is unchanged; :meth:`Scorer.load_ema` writes the result into the buffers).

        ``selection`` (bool ``[..., N]``, at least one leading row) marks the cells selected at each
        (row, position) of the window: ``x`` for ``selected`` is the fraction of rows in which the
        cell was selected. ``realised_utility`` (floating ``[N]``) is the window's measured utility
        per cell, or ``None`` (no measurement: ``utility`` is carried over); ``measured`` (bool
        ``[N]``, all True by default) names the cells it measured — the others keep their average.
        ``decay`` is ``train.ema_decay`` in ``[0, 1]``. Both fields must be ``[N]``."""
        decay = _check_decay(decay)
        utility, selected = self.utility, self.selected
        if utility.ndim != 1 or selected.shape != utility.shape:
            raise ValueError(
                "EMAFeatures.update needs [N] averages (the scorer's buffers), got utility "
                f"{tuple(utility.shape)} and selected {tuple(selected.shape)}"
            )
        N = int(utility.shape[0])
        if not isinstance(selection, Tensor) or selection.dtype != torch.bool:
            raise TypeError("selection must be a bool torch.Tensor [..., N]")
        if selection.ndim < 2 or selection.shape[-1] != N or selection.numel() == 0:
            raise ValueError(
                f"selection must be a non-empty bool [..., N = {N}] tensor, got "
                f"{tuple(selection.shape)}"
            )
        if selection.device != selected.device:
            raise ValueError(
                f"selection lives on {selection.device}, the averages on {selected.device}"
            )
        frequency = selection.reshape(-1, N).to(selected.dtype).mean(dim=0)
        new_selected = decay * selected.detach() + (1.0 - decay) * frequency
        new_utility = utility.detach().clone()
        if realised_utility is not None:
            if not isinstance(realised_utility, Tensor) or not realised_utility.is_floating_point():
                raise TypeError("realised_utility must be a floating torch.Tensor [N] or None")
            if realised_utility.shape != (N,) or realised_utility.device != utility.device:
                raise ValueError(
                    f"realised_utility must be [N] = {(N,)} on {utility.device}, got "
                    f"{tuple(realised_utility.shape)} on {realised_utility.device}"
                )
            if measured is None:
                measured = torch.ones(N, dtype=torch.bool, device=utility.device)
            elif (
                not isinstance(measured, Tensor)
                or measured.dtype != torch.bool
                or measured.shape != (N,)
                or measured.device != utility.device
            ):
                raise ValueError(f"measured must be a bool [N] = {(N,)} tensor on {utility.device}")
            stepped = decay * new_utility + (1.0 - decay) * realised_utility.to(utility.dtype)
            new_utility = torch.where(measured, stepped, new_utility)
        elif measured is not None:
            raise ValueError("measured names measured cells but realised_utility is None")
        return EMAFeatures(utility=new_utility, selected=new_selected)


class Scorer(nn.Module):
    """Per-cell features ``[B, N, 3d+8]`` and the map to unit logits ``[B, N, M]``.

    ``cfg`` supplies every width and the normalisers (``d``, ``n_cells``, ``n_units``, ``age_max``,
    ``r_max``, ``n_context``, ``scorer``, ``scorer_hidden``, ``scorer_features.exclude``,
    ``activation``, ``init``); ``unroll`` is ``train.unroll``, the normaliser of the position
    feature, which lives outside ``model.*`` and is therefore a required argument.

    P2.3: the persistent buffers ``ema_utility`` and ``ema_selected`` (``[N]``, zeros at
    initialisation) are the running averages :meth:`features` reads when no :class:`EMAFeatures`
    is given; :attr:`ema` views them and :meth:`load_ema` writes them (module docstring).

    A5: ``key_address`` appends ``a_cur`` and ``a_prev`` after ``one``, giving ``3d+10``
    features (``model.address_init``: the value their two columns start at, for every unit; the
    default is the random init of the other columns). ``seed`` is required then and seeds fixed
    byte codes and cell addresses without changing the global RNG or the existing scorer columns.
    The code buffers travel with the checkpoint; the previous byte comes from the caller and is
    never stored in the scorer.
    """

    #: State-dict version (``nn.Module`` convention): 2 since P2.3 added the EMA buffers.
    _version = 2

    _excluded: Tensor  # bool [3d+8]: True at zeroed feature slots (non-persistent)
    ema_utility: Tensor  # [N] P2.3 running average of the counterfactual utility (persistent)
    ema_selected: Tensor  # [N] P2.3 running average of the selection indicator (persistent)
    byte_codes: Tensor  # [V, d], A5 only: fixed code for each byte
    cell_addresses: Tensor  # [N, d], A5 only: fixed address for each cell
    ngram_codes: Tensor  # [n - 1, V, d], A9 only: the byte codes of key positions 2..n

    def __init__(self, cfg: ModelConfig, *, unroll: int, seed: int | None = None) -> None:
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
        self.key_address = cfg.key_address
        #: A9: the longest byte n-gram with an address (1: A5's a_cur and a_prev only)
        self.address_ngram = cfg.address_ngram if cfg.key_address else 1
        ngram_names = tuple(
            f"a_{which}{m}" for m in range(2, self.address_ngram + 1) for which in ("cur", "prev")
        )
        address_names = (ADDRESS_FEATURE_NAMES + ngram_names) if cfg.key_address else ()
        base_features = 3 * cfg.d + SCORER_SCALAR_FEATURES
        self.in_features = base_features + len(address_names)
        feature_names = SCORER_FEATURE_NAMES + address_names

        unknown = [name for name in cfg.scorer_features.exclude if name not in feature_names]
        if unknown:
            raise ValueError(
                f"model.scorer_features.exclude names unknown features {unknown}; known: "
                f"{list(feature_names)}"
            )
        self.exclude: tuple[str, ...] = tuple(dict.fromkeys(cfg.scorer_features.exclude))

        self.feature_slices: dict[str, slice] = {}
        start = 0
        for name in feature_names:
            width = cfg.d if name in _VECTOR_FEATURES else 1
            self.feature_slices[name] = slice(start, start + width)
            start += width
        assert start == self.in_features

        self.net: nn.Linear | MLP2
        if cfg.scorer == "linear":
            self.net = nn.Linear(base_features, cfg.n_units)
            init_weight_(self.net.weight, cfg.init)
            nn.init.zeros_(self.net.bias)
        elif cfg.scorer == "mlp":
            self.net = MLP2(
                base_features, cfg.scorer_hidden, cfg.n_units, cfg.activation, init=cfg.init
            )
        else:  # pragma: no cover - ModelConfig.scorer is a Literal
            raise ValueError(f"model.scorer={cfg.scorer!r} is not supported (linear | mlp)")

        if cfg.key_address:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("model.key_address requires run.seed as a non-negative int")
            # Independent RNG: adding codes must not perturb any existing parameter's init.
            generator = torch.Generator(device="cpu").manual_seed(
                (seed ^ _KEY_ADDRESS_TAG) % (2**63)
            )
            scale = math.sqrt(cfg.d)
            self.register_buffer(
                "byte_codes", torch.randn(cfg.vocab_size, cfg.d, generator=generator) / scale
            )
            self.register_buffer(
                "cell_addresses", torch.randn(cfg.n_cells, cfg.d, generator=generator) / scale
            )
            first = self.net if isinstance(self.net, nn.Linear) else self.net.fc1
            extra = first.weight.new_empty(first.out_features, 2)
            std = 1 / math.sqrt(self.in_features) if cfg.init == "fan_in" else parse_init(cfg.init)
            nn.init.trunc_normal_(extra, std=std, a=-2 * std, b=2 * std, generator=generator)
            if cfg.address_init is not None:
                # The address columns start at one value for every unit (model.address_init): a
                # route then follows the byte codes from the first step, and which feature a unit
                # follows at which byte class is left to training. The draw above is kept, so the
                # generator's later use is the same either way.
                extra = torch.full_like(extra, float(cfg.address_init))
            first.weight = nn.Parameter(torch.cat([first.weight.detach(), extra], dim=1))
            if self.address_ngram > 1:
                # A9, drawn after everything A5 draws: the codes of key positions 2..n and the
                # new columns, so the A5 buffers and columns are those of address_ngram = 1.
                tables = [
                    torch.randn(cfg.vocab_size, cfg.d, generator=generator) / scale
                    for _ in range(self.address_ngram - 1)
                ]
                self.register_buffer("ngram_codes", torch.stack(tables))
                more = first.weight.new_empty(first.out_features, len(ngram_names))
                nn.init.trunc_normal_(more, std=std, a=-2 * std, b=2 * std, generator=generator)
                if cfg.address_init is not None:
                    more = torch.full_like(more, float(cfg.address_init))
                first.weight = nn.Parameter(torch.cat([first.weight.detach(), more], dim=1))
            first.in_features = self.in_features
            if isinstance(self.net, MLP2):
                self.net.in_dim = self.in_features

        excluded = torch.zeros(self.in_features, dtype=torch.bool)
        for name in self.exclude:
            excluded[self.feature_slices[name]] = True
        self.register_buffer("_excluded", excluded, persistent=False)
        for name in _EMA_BUFFERS:  # P2.3: zeros until the trainer updates them
            self.register_buffer(name, torch.zeros(cfg.n_cells))

    # ----------------------------------------------------------------------------------------------
    # P2.3: the running averages
    # ----------------------------------------------------------------------------------------------

    @property
    def ema(self) -> EMAFeatures:
        """The scorer's own averages as an :class:`EMAFeatures` (views of the ``[N]`` buffers)."""
        return EMAFeatures(utility=self.ema_utility, selected=self.ema_selected)

    def load_ema(self, ema: EMAFeatures) -> None:
        """Write ``ema`` (``[N]`` floating tensors on this module's device) into the buffers, in
        place and without autograd (the result of :meth:`EMAFeatures.update`)."""
        if not isinstance(ema, EMAFeatures):
            raise TypeError(f"load_ema needs an EMAFeatures, got {type(ema).__name__}")
        for name, value in (("utility", ema.utility), ("selected", ema.selected)):
            buffer = getattr(self, f"ema_{name}")
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError(f"EMAFeatures.{name} must be a floating torch.Tensor")
            if value.shape != buffer.shape or value.device != buffer.device:
                raise ValueError(
                    f"EMAFeatures.{name} must be [N] = {tuple(buffer.shape)} on {buffer.device}, "
                    f"got {tuple(value.shape)} on {value.device}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"EMAFeatures.{name} holds a non-finite value")
        with torch.no_grad():
            self.ema_utility.copy_(ema.utility)
            self.ema_selected.copy_(ema.selected)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """A version-1 state dict (written before P2.3) has no EMA buffers; every model of that
        version had them at zero, so they load as zeros. Version 2 must carry them."""
        version = local_metadata.get("version")
        if version is None or version < 2:
            for name in _EMA_BUFFERS:
                key = prefix + name
                if key not in state_dict:
                    state_dict[key] = torch.zeros_like(getattr(self, name))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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
        *,
        tokens: Tensor | None = None,
        prev_tokens: Tensor | None = None,
        older: Tensor | None = None,
    ) -> Tensor:
        """The ``[B, N, 3d+8]`` feature tensor in the module's layout, in ``u``'s dtype.

        ``u``, ``eps`` are ``[B, d]`` (the dtype and device of ``state.h``); ``ctx`` the context ids
        ``[B]`` (int64/int32); ``budget_frac`` a finite number >= 0; ``round_idx`` an int in
        ``[0, r_max)``; ``ema`` explicit averages, or ``None`` for the scorer's own buffers
        (:attr:`ema`; zeros until P2.3 updates them). Shape, dtype and device mismatches raise.
        With A5, ``tokens`` and ``prev_tokens`` are ``[B]`` and the result has width ``3d+10``.
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
        self._check_budget_frac(budget_frac)
        self._check_round(round_idx)
        scalars = self._scalars(
            ctx[:, None], state.pos[:, None], budget_frac, round_idx, ema, u.dtype, u.device
        )[:, 0]
        if self.key_address:
            addresses = self.address_features(tokens, prev_tokens, older)
            if addresses.shape != (B, N, 2 * self.address_ngram):
                raise ValueError("key-address step tokens and prev_tokens must have shape [B]")
            scalars = torch.cat([scalars, addresses.to(u.dtype)], dim=-1)
        return self._assemble(u, eps, state.h, state.age, scalars)

    def window_scalars(
        self,
        ctx: Tensor,
        pos: Tensor,
        *,
        budget_frac: float,
        round_idx: int,
        dtype: torch.dtype,
        ema: EMAFeatures | None = None,
        tokens: Tensor | None = None,
        prev_tokens: Tensor | None = None,
        older: Tensor | None = None,
    ) -> Tensor:
        """The features of a window that do not depend on the step (P1.10a): ``[B, T, N, 7]`` in
        ``dtype``, the scalars after ``age`` in layout order (budget, round, context, the two P2.3
        averages, pos, one) for every position of the window. ``ctx`` are the window's context ids
        and ``pos`` the session position before each step (int64), both ``[B, T]`` on one device.
        :meth:`features` builds the same values from the same elementwise ops for one position.
        A5 requires current and previous ``tokens [B,T]`` and appends the two address features,
        so this tail has width 9 instead of 7."""
        if not isinstance(ctx, Tensor) or not isinstance(pos, Tensor):
            raise TypeError("ctx and pos must be torch.Tensors [B, T]")
        if ctx.dtype not in _INDEX_DTYPES:
            raise TypeError(f"ctx must be an int64 or int32 tensor of ids, got dtype {ctx.dtype}")
        if pos.dtype != torch.int64:
            raise TypeError(f"pos must be an int64 tensor, got dtype {pos.dtype}")
        if ctx.ndim != 2 or pos.shape != ctx.shape:
            raise ValueError(
                f"ctx and pos must both be [B, T], got {tuple(ctx.shape)} and {tuple(pos.shape)}"
            )
        if pos.device != ctx.device:
            raise ValueError(f"ctx lives on {ctx.device} but pos on {pos.device}")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError(f"dtype must be a floating torch.dtype, got {dtype!r}")
        self._check_budget_frac(budget_frac)
        self._check_round(round_idx)
        scalars = self._scalars(ctx, pos, budget_frac, round_idx, ema, dtype, ctx.device)
        if self.key_address:
            addresses = self.address_features(tokens, prev_tokens, older)
            if addresses.shape != (*ctx.shape, self.n_cells, 2 * self.address_ngram):
                raise ValueError("key-address window tokens and prev_tokens must have shape [B, T]")
            scalars = torch.cat([scalars, addresses.to(dtype)], dim=-1)
        return scalars

    def address_features(
        self, tokens: Tensor | None, prev_tokens: Tensor | None, older: Tensor | None = None
    ) -> Tensor:
        """A5's fixed similarities ``[..., N, 2]`` in ``a_cur, a_prev`` order.

        Both token tensors have the same shape and contain only current/past bytes. The caller
        supplies BOS for ``prev_tokens`` at document resets. Fixed codes are persistent buffers,
        not trainable parameters; the two matmuls run outside autocast so addresses agree at a
        fact's value and its query even when training uses mixed precision.

        A9 (``model.address_ngram = n > 1``): ``[..., N, 2n]``, the two features above and then
        ``a_cur<m>, a_prev<m>`` for ``m = 2 .. n``. ``older [..., n − 1]`` holds the bytes two to
        ``n`` positions back (``x_{t−2}`` first; BOS where the document has none; ``None``: all
        BOS). The code of the ``m`` bytes ``b_1 .. b_m`` is the elementwise product of
        ``codes_i[b_i]`` (``codes_1`` = A5's byte codes) scaled to A5's variance; ``a_cur<m>``
        reads the bytes ``x_{t−m+1} .. x_t`` and ``a_prev<m>`` the bytes ``x_{t−m} .. x_{t−1}``,
        so a key of ``m`` bytes has one address at its fact's value and at its query's last
        key byte.
        """
        if not self.key_address:
            raise ValueError("address_features requires model.key_address=true")
        for name, value in (("tokens", tokens), ("prev_tokens", prev_tokens)):
            if not isinstance(value, Tensor) or value.dtype not in _INDEX_DTYPES:
                raise TypeError(f"{name} must be an int64 or int32 token tensor for key_address")
            if value.device != self.byte_codes.device:
                raise ValueError(f"{name} must live on {self.byte_codes.device}")
        assert isinstance(tokens, Tensor) and isinstance(prev_tokens, Tensor)
        if tokens.shape != prev_tokens.shape:
            raise ValueError("key-address tokens and prev_tokens must have the same shape")
        n = self.address_ngram
        if n > 1:
            if older is None:
                older = torch.full(
                    (*tokens.shape, n - 1), BOS, dtype=tokens.dtype, device=tokens.device
                )
            elif (
                not isinstance(older, Tensor)
                or older.dtype not in _INDEX_DTYPES
                or older.shape != (*tokens.shape, n - 1)
                or older.device != tokens.device
            ):
                raise ValueError(
                    f"older must be an integer tensor {(*tokens.shape, n - 1)} on {tokens.device}"
                )
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            addresses = self.cell_addresses.transpose(0, 1)
            current = F.embedding(tokens, self.byte_codes) @ addresses
            previous = F.embedding(prev_tokens, self.byte_codes) @ addresses
            feats = [current, previous]
            if n > 1:
                assert older is not None
                lags = [tokens, prev_tokens, *older.unbind(dim=-1)]  # x_t, x_{t-1}, ..., x_{t-n}
                root = math.sqrt(self.d)
                for m in range(2, n + 1):
                    for back in (0, 1):  # a_cur<m>, a_prev<m>
                        code = F.embedding(lags[back + m - 1], self.byte_codes)  # the first byte
                        for i in range(2, m + 1):
                            table = self.ngram_codes[i - 2]
                            code = code * F.embedding(lags[back + m - i], table) * root
                        feats.append(code @ addresses)
            return torch.stack(feats, dim=-1) * math.sqrt(self.d)

    def _scalars(
        self,
        ctx: Tensor,
        pos: Tensor,
        budget_frac: float,
        round_idx: int,
        ema: EMAFeatures | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """``[B, T, N, 7]``: budget, round, context, ema_utility, ema_selected, pos, one (the
        layout after ``age``) from ``ctx`` and ``pos`` ``[B, T]``."""
        B, T = (int(n) for n in ctx.shape)
        N = self.n_cells
        budget = torch.full((B, T, N), float(budget_frac), dtype=dtype, device=device)
        rnd = torch.full((B, T, N), round_idx / self.r_max, dtype=dtype, device=device)
        context = (ctx.to(dtype) / self.n_context)[:, :, None].expand(B, T, N)
        if ema is None:
            ema = self.ema  # P2.3: the scorer's own averages (zeros until updated)
        ema_utility = self._check_ema(ema.utility, "utility", B, N, device).to(dtype)
        ema_selected = self._check_ema(ema.selected, "selected", B, N, device).to(dtype)
        position = (pos.to(dtype) / self.unroll)[:, :, None].expand(B, T, N)
        one = torch.ones(B, T, N, dtype=dtype, device=device)
        return torch.stack(
            [
                budget,
                rnd,
                context,
                ema_utility[:, None, :].expand(B, T, N),
                ema_selected[:, None, :].expand(B, T, N),
                position,
                one,
            ],
            dim=-1,
        )  # [B, T, N, 7] in _SCALAR_FEATURES[1:] order

    def _assemble(
        self, u: Tensor, eps: Tensor, h: Tensor, age: Tensor | None, scalars: Tensor
    ) -> Tensor:
        """``[B, N, 3d+8]`` from ``u``, ``eps`` ``[B, d]``, ``h [B, N, d]``, ``age [B, N]`` and the
        other scalars ``[B, N, 7]`` (:meth:`_scalars` at one position), unchecked: the callers
        validated their inputs (:meth:`features` per call, :meth:`window_scalars` per window).
        With ``age=None`` the scalars already hold the age feature in front, ``[B, N, 8]`` (the
        dense window of P1.10b, whose ages are known in advance)."""
        B, N, d = (int(n) for n in h.shape)
        parts = [u[:, None, :].expand(B, N, d), eps[:, None, :].expand(B, N, d), h]
        if age is not None:
            parts.append((age.to(u.dtype) / self.age_max)[:, :, None])  # [B, N, 1]
        parts.append(scalars)
        feats = torch.cat(parts, dim=-1)
        if self.exclude:
            feats = feats.masked_fill(self._excluded, 0)
        return feats

    def _check_round(self, round_idx: int) -> None:
        if isinstance(round_idx, bool) or not isinstance(round_idx, int):
            raise TypeError(f"round_idx must be an int, got {round_idx!r}")
        if not 0 <= round_idx < self.r_max:
            raise ValueError(
                f"round_idx={round_idx} must lie in [0, r_max) = [0, {self.r_max}) (0-based: "
                "rounds already completed at this position)"
            )

    @staticmethod
    def _check_budget_frac(budget_frac: float) -> None:
        if (
            isinstance(budget_frac, bool)
            or not isinstance(budget_frac, int | float)
            or not math.isfinite(budget_frac)
            or budget_frac < 0.0
        ):
            raise ValueError(f"budget_frac must be a finite number >= 0, got {budget_frac!r}")

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
    def _check_ema(x: Tensor, name: str, B: int, N: int, device: torch.device) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"EMAFeatures.{name} must be a torch.Tensor, got {type(x).__name__}")
        if not x.is_floating_point():
            raise ValueError(f"EMAFeatures.{name} must be floating point, got {x.dtype}")
        if x.device != device:
            raise ValueError(f"EMAFeatures.{name} must live on {device}, got {x.device}")
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

    P2.2 (module docstring): ``cell_mask [B, N]`` and ``unit_weights [B, N, M]`` (fp32) are what
    the executor multiplies the cell updates and the unit outputs with (exactly hard in the hard
    modes, with the gate's gradient attached; soft values in the warm-up; ``alpha`` is their
    product on the valid slots); ``expected_cost [B]`` (fp32) is the differentiable expected cost;
    ``log_prob [B]`` (fp32) the REINFORCE log-probability of the sampled cells (zeros otherwise);
    ``explored`` and ``refreshed`` (bool ``[B, K]``) flag the slots exploration and the refresh
    probes replaced; ``dense_fallback`` (bool ``[B]``) flags the rows that executed the dense step
    instead of their slots. In ``dense`` mode ``cell_mask`` is all ones and ``unit_weights`` the
    softmax over units.

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
    # P2.2: the gradient path and the control flags
    cell_mask: Tensor
    unit_weights: Tensor
    expected_cost: Tensor
    log_prob: Tensor
    explored: Tensor
    refreshed: Tensor
    dense_fallback: Tensor
    #: P2.10 (the split window's tape): the detached straight-through threshold ``θ [B]`` of the
    #: scored decision before the probes; ``None`` where no threshold exists (the unscored modes,
    #: ``dense``). Never read by the round itself.
    threshold: Tensor | None = None

    def __post_init__(self) -> None:
        for name in (
            "cell_idx",
            "unit_idx",
            "alpha",
            "valid",
            "logits",
            "cost",
            "cell_mask",
            "unit_weights",
            "expected_cost",
            "log_prob",
            "explored",
            "refreshed",
            "dense_fallback",
        ):
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
        N, M = int(self.logits.shape[1]), int(self.logits.shape[2])
        for name in ("cost", "expected_cost", "log_prob"):
            value = getattr(self, name)
            if value.shape != (B,) or not value.is_floating_point():
                raise ValueError(
                    f"Selection.{name} must be a floating [B] = {(B,)} tensor, got shape "
                    f"{tuple(value.shape)} and dtype {value.dtype}"
                )
        for name, shape, floating in (
            ("cell_mask", (B, N), True),
            ("unit_weights", (B, N, M), True),
            ("explored", (B, K), False),
            ("refreshed", (B, K), False),
            ("dense_fallback", (B,), False),
        ):
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(
                    f"Selection.{name} must have shape {shape}, got {tuple(value.shape)}"
                )
            if floating and not value.is_floating_point():
                raise ValueError(f"Selection.{name} must be floating, got {value.dtype}")
            if not floating and value.dtype != torch.bool:
                raise ValueError(f"Selection.{name} must be a bool tensor, got {value.dtype}")
        for name in (
            "cell_idx",
            "unit_idx",
            "alpha",
            "valid",
            "cost",
            "cell_mask",
            "unit_weights",
            "expected_cost",
            "log_prob",
            "explored",
            "refreshed",
            "dense_fallback",
        ):
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

    def selected_cells(self) -> Tensor:
        """The cells whose state this selection updates, bool ``[B, N]``: the valid slots
        scattered (padded slots contribute nothing, so a padded cell 0 never counts) and every cell
        of a dense-fallback row."""
        B, N = int(self.logits.shape[0]), int(self.logits.shape[1])
        counts = torch.zeros(B, N, dtype=torch.int64, device=self.logits.device)
        counts = counts.scatter_add(1, self.cell_idx, self.valid.to(torch.int64))
        return (counts > 0) | self.dense_fallback[:, None]


@dataclass
class RouterTap:
    """What :meth:`Router.intercept` records while it is armed (P2.3): the detached ``features``
    every :meth:`Router.forward` call scored (``[B, N, 3d+8]``) and the ``selections`` every
    :meth:`Router.select` call returned (after ``edit``), in call order. ``edit`` maps the
    selection :meth:`Router.select` made to the one it returns (``None``: unchanged)."""

    edit: Callable[[Selection], Selection] | None = None
    features: list[Tensor] = field(default_factory=list)
    selections: list[Selection] = field(default_factory=list)


class Router(nn.Module):
    """The P1.5 :class:`Scorer` plus budgeted selection (module docstring).

    ``scorer`` is a submodule (since P2.2 ``HSN.scorer`` is ``HSN.router.scorer``);
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
        self.age_max = scorer.age_max  # the refresh-probe threshold (model.age_max)
        self.lambda_cost = float(lambda_cost)
        #: ``train.unit_credit``: "softmax" (P2.2) or "executed" (:meth:`set_unit_credit`).
        self.unit_credit = "softmax"
        self.costs = costs
        self.cell_overhead = costs.cell_overhead
        self.shared_cost = costs.shared
        self.scorer_cost = costs.scorer
        self.cost_table_hash = costs.hash
        self.cost_source = costs.source
        self.max_pair_cost = costs.max_pair_cost()
        #: P2.8: the dense step's cost, once (CostTable.dense_cost reduces a CPU tensor, which a
        #: compiled or captured position must not do); the table never changes after construction.
        self._dense_cost = costs.dense_cost()
        self._unit_costs: list[float] = [float(v) for v in costs.unit.tolist()]
        self.register_buffer("cost_table", costs.unit.to(torch.float32), persistent=False)
        self._tap: RouterTap | None = None  # P2.3: armed by intercept()

    @classmethod
    def from_config(
        cls, cfg: ModelConfig, *, unroll: int, scorer: Scorer | None = None, seed: int | None = None
    ) -> Self:
        """A router for ``cfg``: the scorer (built here unless given), the table of
        ``model.cost_profile`` (the uniform placeholder when ``null``) and ``model.lambda_cost``;
        ``unroll`` is ``train.unroll``. A sparse ``model.routing`` whose ``model.budget`` cannot
        hold ``model.k_min`` cells (or the whole ``model.fixed_route``) raises here, at
        construction."""
        if not isinstance(cfg, ModelConfig):
            raise TypeError(f"from_config needs a ModelConfig, got {type(cfg).__name__}")
        if not cfg.static_k:
            raise NotImplementedError(
                "model.static_k=false (a selection trimmed to a data-dependent K) is not "
                "implemented: every mode pads to K = model.k_select with valid=False (P2.8), which "
                "is what graph capture and torch.compile need; set model.static_k=true"
            )
        if scorer is None:
            scorer = Scorer(cfg, unroll=unroll, seed=seed)
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
        batch row). While :meth:`intercept` is armed the features are recorded (P2.3)."""
        if self._tap is not None:
            self._tap.features.append(feats.detach())
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

    def set_unit_credit(self, mode: str) -> None:
        """``train.unit_credit`` (opt-in; the hard values never change): which scores the
        straight-through unit weight of a chosen unit differentiates. ``"softmax"``: the softmax
        over the cell's units (P2.2), whose gradient lowers every other unit's score where the
        chosen unit's weight is raised. ``"executed"``: the chosen unit's score only; the others
        enter that softmax as constants, so a unit that did not run gets no gradient from the
        slot. The soft warm-up and dense-fallback rows, where every unit runs, keep the softmax."""
        if mode not in ("softmax", "executed"):
            raise ValueError(f"unit_credit must be 'softmax' or 'executed', got {mode!r}")
        self.unit_credit = mode

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
        *,
        gate: GateMode | None = None,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        age: Tensor | None = None,
        dense_rows: Tensor | None = None,
        reserved: int = 0,
        reserved_units: bool = False,
    ) -> Selection:
        """Select up to ``k_max`` (cell, unit) slots per row within ``budget`` (module docstring).

        ``reserved`` (amendment A8, ``model.fast_cells``; ``topk`` without ``reinforce`` only):
        cells ``0 .. reserved − 1`` take the first ``reserved`` slots at every call, each at its
        best unit, and the scored decision fills the other ``k_max − reserved`` slots from the
        remaining cells. The straight-through threshold is taken among the remaining cells, the
        reserved cells' mask is the constant 1 (they have no selection gradient), and the probes
        never replace a reserved slot. ``0``: the decision as before, bit for bit.
        ``reserved_units`` (``model.fast_units = fixed``): reserved cell ``i`` runs unit ``i mod
        M`` instead of its best unit, with a constant one-hot unit weight in every gate mode.

        ``logits`` are the scorer's ``[B, N, M]`` on the router's device; ``budget`` is ``B_t`` as
        a fraction of the dense step (finite, >= 0); ``0 <= k_min <= k_max <= N`` with
        ``k_max >= 1``; ``fixed`` (int64/int32 ``[L]``, distinct cells in ``[0, N)``, ``k_min <= L
        <= k_max``) only with ``mode="fixed"``, where it is required; ``generator`` only where a
        draw happens (``mode="random"``, ``epsilon > 0`` or ``gate="reinforce"``; the global RNG
        when ``None``). Every sparse mode raises ``ValueError`` up front when ``budget`` cannot
        hold ``k_min`` cells at the dearest unit (``fixed``: the whole list), so the result always
        has ``k_min <= n_valid <= k_max`` and ``cost <= budget``.

        P2.2 keywords (module docstring): ``gate`` is the gradient path (``None``: hard, detached
        tensors; ``soft`` / ``ste`` / ``reinforce``: ``train.gate_mode`` at this step);
        ``temperature`` is τ (finite, > 0); ``epsilon`` the exploration rate in ``[0, 1]`` and
        ``age`` the state's ``age [B, N]`` for the refresh probes, both only with a scored mode
        (``topk``, ``greedy_budget``); ``dense_rows`` (bool ``[B]``) the rows that execute the
        dense step instead (not in ``dense`` mode, which is dense already). ``reinforce`` needs
        ``topk``; ``dense`` accepts ``gate`` ``None`` or ``soft`` only (its units are soft already).
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
        tau = self._check_gate_args(mode, gate, temperature, epsilon)
        if isinstance(reserved, bool) or not isinstance(reserved, int) or reserved < 0:
            raise ValueError(f"reserved must be an int >= 0, got {reserved!r}")
        if reserved:
            if mode != "topk" or gate == "reinforce":
                raise ValueError(
                    f"reserved={reserved} needs mode='topk' without gate='reinforce', got "
                    f"mode={mode!r}, gate={gate!r}"
                )
            if reserved >= k_max:
                raise ValueError(f"reserved={reserved} must be < k_max={k_max}")
        elif reserved_units:
            raise ValueError("reserved_units needs reserved > 0")
        age_t = self._check_age(age, mode, B, N, logits.device)
        dense = self._check_dense_rows(dense_rows, mode, B, logits.device)
        if generator is not None and not (mode == "random" or epsilon > 0.0 or gate == "reinforce"):
            raise ValueError(
                "`generator` is only used with mode='random', epsilon > 0 or gate='reinforce', "
                f"got mode={mode!r}, epsilon={epsilon!r}, gate={gate!r}"
            )
        if mode == "dense":
            return self._tapped(self._select_dense(logits))

        self.require_affordable(budget_f, k_min, "k_min")
        scores = self.scores(logits)  # fp32, autograd intact: the gradient path reads it
        scores_d = scores.detach()  # the decisions are indices and read a detached copy
        log_prob = torch.zeros(B, dtype=torch.float32, device=logits.device)
        if mode == "topk":
            if gate == "reinforce":
                cells, units, order = self._gumbel_topk(scores_d, k_max, tau, generator)
            elif reserved:
                cells, units = self._topk(scores_d, k_max, reserved)
                if reserved_units:  # fast cell i runs unit i mod M
                    fixed_units = torch.arange(reserved, device=units.device) % M
                    units = torch.cat(
                        [fixed_units[None, :].expand(B, reserved), units[:, reserved:]], dim=1
                    )
            else:
                cells, units = self._topk(scores_d, k_max)
            cells, units, valid = self._truncate(cells, units, budget_f)
            if gate == "reinforce":
                log_prob = self._plackett_luce(scores.amax(dim=-1) / tau, order, valid)
        elif mode == "greedy_budget":
            cells, units, valid = self._greedy(scores_d, budget_f, k_max)
        elif mode == "random":
            cells, units = self._random(B, N, M, k_max, generator, logits.device)
            cells, units, valid = self._truncate(cells, units, budget_f)
        else:  # fixed
            self.require_affordable(budget_f, int(fixed_cells.shape[0]), "the fixed route")
            cells, units, valid = self._fixed(scores_d, fixed_cells, k_max)
        scored = mode in SCORED_MODES
        # the straight-through threshold belongs to the scored decision, before the probes
        threshold = (
            self._ste_threshold(scores_d.amax(dim=-1), cells, valid, reserved) if scored else None
        )
        if scored and (age_t is not None or epsilon > 0.0):
            cells, units, explored, refreshed = self._post_process(
                cells, units, valid, scores_d, age_t, epsilon, budget_f, dense, generator, reserved
            )
        else:
            explored = torch.zeros(B, k_max, dtype=torch.bool, device=logits.device)
            refreshed = torch.zeros_like(explored)
        selection = self._finish(
            logits,
            scores,
            cells,
            units,
            valid,
            mode=mode,
            gate=gate,
            tau=tau,
            threshold=threshold,
            log_prob=log_prob,
            explored=explored,
            refreshed=refreshed,
            dense=dense,
            reserved=reserved,
            reserved_units=reserved_units,
        )
        return self._tapped(selection)

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

    # -- P2.3: counterfactual support -------------------------------------------------------------

    @contextlib.contextmanager
    def intercept(
        self, edit: Callable[[Selection], Selection] | None = None
    ) -> Iterator[RouterTap]:
        """Arm a :class:`RouterTap` for the duration of the ``with`` block: :meth:`forward`
        records its features and :meth:`select` returns ``edit(selection)`` (unchanged without
        ``edit``) and records it. One tap at a time: nesting raises. The tap is how the
        counterfactual probes (:mod:`hsn.train.probes`) read and edit the selection that
        :meth:`hsn.model.hsn.HSN.step` executes, without a hook in the model's step."""
        if self._tap is not None:
            raise RuntimeError("Router.intercept is already armed; taps do not nest")
        if edit is not None and not callable(edit):
            raise TypeError(f"edit must be callable or None, got {type(edit).__name__}")
        tap = RouterTap(edit=edit)
        self._tap = tap
        try:
            yield tap
        finally:
            self._tap = None

    def _tapped(self, selection: Selection) -> Selection:
        tap = self._tap
        if tap is None:
            return selection
        if tap.edit is not None:
            edited = tap.edit(selection)
            if not isinstance(edited, Selection):
                raise TypeError(f"the tap's edit must return a Selection, got {type(edited)}")
            if edited.logits.shape != selection.logits.shape:
                raise ValueError(
                    f"the tap's edit changed the selection's [B, N, M] from "
                    f"{tuple(selection.logits.shape)} to {tuple(edited.logits.shape)}"
                )
            # P1.10e: dense-fallback rows come from ``dense_rows`` only (the sparse round relies on
            # it inside a window, where it no longer re-checks); an edit may not add any. The usual
            # edit (with_candidate) keeps the tensor itself, so this costs nothing then.
            if edited.dense_fallback is not selection.dense_fallback and bool(
                (edited.dense_fallback & ~selection.dense_fallback).any()
            ):
                raise ValueError(
                    "the tap's edit added dense-fallback rows; they come from dense_rows only"
                )
            selection = edited
        tap.selections.append(selection)
        return selection

    def with_candidate(
        self, selection: Selection, cells: Tensor, units: Tensor, kind: Tensor
    ) -> Selection:
        """``selection`` with one counterfactual candidate per row (module docstring, P2.3).

        ``cells`` and ``units`` are int64 ``[B]``; ``kind`` (int64 ``[B]``) is
        :data:`CANDIDATE_NONE` (row unchanged), :data:`CANDIDATE_ADD` (``cells[b]`` must be
        unselected in row ``b``; it is written with unit ``units[b]`` in a free padded slot — one
        padded column is appended when ``K < N`` — with ``cell_mask = 1`` and one-hot
        ``unit_weights``) or :data:`CANDIDATE_REMOVE` (a valid slot holding ``(cells[b],
        units[b])`` becomes padded and the cell's ``cell_mask`` 0). ``cost`` and ``expected_cost``
        are recomputed on the edited rows. The selection must be hard (``cell_mask`` 0/1: not the
        soft warm-up)
        and an edited row must not be a dense-fallback row. Meant for no-grad probes: the edited
        tensors carry no gradient path of their own."""
        B, N, M = (int(s) for s in selection.logits.shape)
        device = selection.logits.device
        for name, value in (("cells", cells), ("units", units), ("kind", kind)):
            if not isinstance(value, Tensor) or value.dtype != torch.int64:
                raise TypeError(f"{name} must be an int64 torch.Tensor [B]")
            if value.shape != (B,) or value.device != device:
                raise ValueError(
                    f"{name} must be [B] = {(B,)} on {device}, got {tuple(value.shape)} on "
                    f"{value.device}"
                )
        add = kind == CANDIDATE_ADD
        remove = kind == CANDIDATE_REMOVE
        edited = add | remove
        safe_cells = cells.clamp(0, N - 1)
        safe_units = units.clamp(0, M - 1)
        selected = selection.selected_cells()
        match = (
            selection.valid
            & (selection.cell_idx == safe_cells[:, None])
            & (selection.unit_idx == safe_units[:, None])
        )  # [B, K]: at most one slot (cells are distinct within a row)
        mask = selection.cell_mask.detach()
        problems = torch.stack(
            [
                (~edited & (kind != CANDIDATE_NONE)).any(),
                (edited & ((cells < 0) | (cells >= N) | (units < 0) | (units >= M))).any(),
                (add & selected.gather(1, safe_cells[:, None]).squeeze(1)).any(),
                (remove & ~match.any(dim=1)).any(),
                (edited & selection.dense_fallback).any(),
                ((mask != 0) & (mask != 1)).any(),
            ]
        ).tolist()  # one device-to-host copy
        reasons = (
            "kind holds a value other than CANDIDATE_NONE / ADD / REMOVE",
            f"a candidate lies outside cells [0, {N}) or units [0, {M})",
            "an added candidate's cell is already selected in its row",
            "a removed candidate is not a valid slot (cell, unit) of its row",
            "a candidate row is a dense-fallback row (it does not execute its selection)",
            "the selection is soft (cell_mask not 0/1): a soft gate has no hard counterfactual",
        )
        bad = [reason for flag, reason in zip(problems, reasons, strict=True) if flag]
        if bad:
            raise ValueError("with_candidate: " + "; ".join(bad))

        def padded(x: Tensor, value: float | bool) -> Tensor:
            if selection.k >= N:  # no unselected cell without a padded slot: nothing to append
                return x
            return torch.cat([x, torch.full_like(x[:, :1], value)], dim=1)

        cell_idx = padded(selection.cell_idx, 0)
        unit_idx = padded(selection.unit_idx, 0)
        valid = padded(selection.valid, False)
        alpha = padded(selection.alpha.detach(), 0.0)
        explored = padded(selection.explored, False)
        refreshed = padded(selection.refreshed, False)
        slots = int(cell_idx.shape[1])
        match = padded(match, False)
        free = (~valid).to(torch.float32).argmax(dim=1)  # the first padded slot of each row
        slot_add = F.one_hot(free, slots).to(torch.bool) & add[:, None]
        slot_remove = match & remove[:, None]
        changed = slot_add | slot_remove
        cell_idx = torch.where(slot_add, safe_cells[:, None], cell_idx).masked_fill(slot_remove, 0)
        unit_idx = torch.where(slot_add, safe_units[:, None], unit_idx).masked_fill(slot_remove, 0)
        valid = (valid | slot_add) & ~slot_remove
        alpha = alpha.masked_fill(slot_add, 1.0).masked_fill(slot_remove, 0.0)
        explored = explored & ~changed
        refreshed = refreshed & ~changed
        cell_hot = F.one_hot(safe_cells, N).to(torch.bool)  # [B, N]
        cell_mask = mask.masked_fill(cell_hot & add[:, None], 1.0)
        cell_mask = cell_mask.masked_fill(cell_hot & remove[:, None], 0.0)
        unit_hot = F.one_hot(safe_units, M).to(selection.unit_weights.dtype)  # [B, M]
        unit_weights = torch.where(
            (cell_hot & add[:, None])[:, :, None],
            unit_hot[:, None, :],
            selection.unit_weights.detach(),
        )
        recomputed = self._executed_cost(unit_idx, valid)
        return Selection(
            cell_idx=cell_idx,
            unit_idx=unit_idx,
            alpha=alpha,
            valid=valid,
            logits=selection.logits,
            cost=torch.where(edited, recomputed, selection.cost),
            scorer_cost=selection.scorer_cost,
            n_rounds_used=selection.n_rounds_used,
            cell_mask=cell_mask,
            unit_weights=unit_weights,
            expected_cost=torch.where(edited, recomputed, selection.expected_cost.detach()),
            log_prob=selection.log_prob,
            explored=explored,
            refreshed=refreshed,
            dense_fallback=selection.dense_fallback,
        )

    # -- the modes --------------------------------------------------------------------------------

    def _select_dense(self, logits: Tensor) -> Selection:
        B, N = int(logits.shape[0]), self.n_cells
        device = logits.device
        probs = torch.softmax(logits.float(), dim=-1)  # [B, N, M]
        units = logits.float().argmax(dim=-1)  # [B, N], the first maximal unit
        alpha = probs.gather(2, units.unsqueeze(-1)).squeeze(-1)
        cells = torch.arange(N, device=device).unsqueeze(0).expand(B, N).clone()
        valid = torch.ones(B, N, dtype=torch.bool, device=device)
        cost = torch.full((B,), self._dense_cost, dtype=torch.float32, device=device)
        return Selection(
            cell_idx=cells,
            unit_idx=units,
            alpha=alpha,
            valid=valid,
            logits=logits,
            cost=cost,
            scorer_cost=self.scorer_cost,
            n_rounds_used=1,
            cell_mask=torch.ones(B, N, dtype=torch.float32, device=device),
            unit_weights=probs,
            expected_cost=cost.clone(),
            log_prob=torch.zeros(B, dtype=torch.float32, device=device),
            explored=torch.zeros(B, N, dtype=torch.bool, device=device),
            refreshed=torch.zeros(B, N, dtype=torch.bool, device=device),
            dense_fallback=torch.zeros(B, dtype=torch.bool, device=device),
        )

    @staticmethod
    def _topk(scores: Tensor, k_max: int, reserved: int = 0) -> tuple[Tensor, Tensor]:
        cell_scores = scores.amax(dim=-1)  # [B, N]
        if reserved:  # A8: the fast group sorts first, in index order (the sort is stable)
            head = torch.full_like(cell_scores[:, :reserved], float("inf"))
            cell_scores = torch.cat([head, cell_scores[:, reserved:]], dim=1)
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

    # -- P2.2: the gradient path, the probes, REINFORCE ------------------------------------------

    @staticmethod
    def _selected(cells: Tensor, valid: Tensor, N: int) -> Tensor:
        """Bool ``[B, N]``: the cells of the valid slots (padded slots add nothing)."""
        counts = torch.zeros(cells.shape[0], N, dtype=torch.int64, device=cells.device)
        return counts.scatter_add(1, cells, valid.to(torch.int64)) > 0

    @staticmethod
    def _draw(
        shape: tuple[int, ...], generator: torch.Generator | None, device: torch.device
    ) -> Tensor:
        """Uniform ``[0, 1)`` draws from the CPU generator (``generator`` or the global RNG) moved
        to ``device``: the same decisions on every backend for the same seed, and a checkpoint's
        RNG state replays them."""
        sample_device = generator.device if generator is not None else torch.device("cpu")
        return torch.rand(shape, generator=generator, device=sample_device).to(device)

    def _gumbel_topk(
        self, scores_d: Tensor, k_max: int, tau: float, generator: torch.Generator | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """A sample of ``k_max`` cells without replacement from ``softmax(s / τ)`` (Gumbel-top-k)
        through the top-k machinery; returns ``(cells, units, order)`` with ``order [B, N]`` the
        full sampled permutation (the Plackett–Luce log-probability reads it)."""
        B, N = int(scores_d.shape[0]), int(scores_d.shape[1])
        cell_scores = scores_d.amax(dim=-1)
        uniform = self._draw((B, N), generator, scores_d.device)
        tiny = torch.finfo(torch.float32).tiny
        gumbel = -torch.log(-torch.log(uniform.clamp(min=tiny, max=1.0 - 1e-7)))
        perturbed = cell_scores / tau + gumbel
        order = torch.sort(perturbed, dim=1, descending=True, stable=True).indices
        cells = order[:, :k_max]
        units = scores_d.argmax(dim=-1).gather(1, cells)
        return cells, units, order

    @staticmethod
    def _plackett_luce(s_tau: Tensor, order: Tensor, valid: Tensor) -> Tensor:
        """``log p`` of drawing the valid prefix of ``order`` without replacement from
        ``softmax(s_tau)``: ``Σ_k [s_(k) − logsumexp_{j ∉ earlier} s_j]``, fp32 ``[B]``,
        differentiable in ``s_tau``. The suffix log-sum-exps use an ``[N, N]`` mask (no
        ``logcumsumexp``, whose backend support varies)."""
        N = int(order.shape[1])
        sorted_s = s_tau.gather(1, order)  # [B, N] in sampled order
        later = torch.arange(N, device=order.device)
        suffix = later[None, :] >= later[:, None]  # [N(k), N(j)]: j >= k
        expanded = sorted_s[:, None, :].expand(-1, N, -1)
        masked = expanded.masked_fill(~suffix[None, :, :], float("-inf"))
        log_norm = torch.logsumexp(masked, dim=-1)  # [B, N]: logsumexp over positions k..N-1
        terms = sorted_s - log_norm
        K = int(valid.shape[1])
        return (terms[:, :K] * valid.to(terms.dtype)).sum(dim=1)

    def _ste_threshold(
        self, cell_scores_d: Tensor, cells: Tensor, valid: Tensor, reserved: int = 0
    ) -> Tensor:
        """θ ``[B]``: the midpoint between the lowest selected and the highest unselected cell
        score (``+inf`` when nothing is selected, ``−inf`` when everything is). With ``reserved``
        (A8) the lowest selected score is taken among the scored slots' cells only: the reserved
        cells are selected whatever their score."""
        selected = self._selected(cells, valid, int(cell_scores_d.shape[1]))
        inf = float("inf")
        lowest_in = torch.where(selected, cell_scores_d, torch.full_like(cell_scores_d, inf))
        if reserved:
            lowest_in = torch.cat(
                [torch.full_like(lowest_in[:, :reserved], inf), lowest_in[:, reserved:]], dim=1
            )
        highest_out = torch.where(selected, torch.full_like(cell_scores_d, -inf), cell_scores_d)
        return 0.5 * (lowest_in.amin(dim=1) + highest_out.amax(dim=1))

    def _post_process(
        self,
        cells: Tensor,
        units: Tensor,
        valid: Tensor,
        scores_d: Tensor,
        age: Tensor | None,
        epsilon: float,
        budget: float,
        dense: Tensor | None,
        generator: torch.Generator | None,
        reserved: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Refresh probes, then exploration, then the budget recheck (module docstring); returns
        ``(cells, units, explored, refreshed)``. ``valid`` never changes. The first ``reserved``
        slots (A8) are never replaced."""
        B, K = int(cells.shape[0]), int(cells.shape[1])
        N = self.n_cells
        device = cells.device
        unit_argmax = scores_d.argmax(dim=-1)  # [B, N]
        active = (
            ~dense if dense is not None else torch.ones(B, dtype=torch.bool, device=device)
        )  # dense-fallback rows keep their scored decision (it is not executed)
        cells0, units0 = cells, units
        n_valid = valid.sum(dim=1)
        slots = torch.arange(K, device=device)[None, :]
        refreshed = torch.zeros(B, K, dtype=torch.bool, device=device)
        explored = torch.zeros(B, K, dtype=torch.bool, device=device)
        if age is not None:
            stale = (age > self.age_max) & ~self._selected(cells, valid, N)
            n_force = torch.minimum(stale.sum(dim=1), (n_valid - reserved).clamp(min=0))
            key = torch.where(stale, age, torch.full_like(age, -1)).to(torch.float32)
            order = torch.sort(key, dim=1, descending=True, stable=True).indices  # stalest first
            j = n_valid[:, None] - 1 - slots  # the tail of the valid slots, lowest score first
            replace = (j >= 0) & (j < n_force[:, None]) & active[:, None]
            new_cells = order.gather(1, j.clamp(min=0, max=N - 1))
            cells = torch.where(replace, new_cells, cells)
            units = torch.where(replace, unit_argmax.gather(1, new_cells), units)
            refreshed = replace
        if epsilon > 0.0:
            draw = self._draw((B, N + 2), generator, device)  # one host-to-device copy
            u, r, keys = draw[:, 0], draw[:, 1], draw[:, 2:]
            candidate = valid & ~refreshed
            if reserved:
                candidate = candidate & (slots >= reserved)
            n_candidate = candidate.sum(dim=1)
            # never a cell the scorer chose, even one a refresh probe just evicted
            selected = self._selected(cells, valid, N) | self._selected(cells0, valid, N)
            n_unselected = N - selected.sum(dim=1)
            row = (u < epsilon) & (n_candidate > 0) & (n_unselected > 0) & active
            pick = (r * n_candidate.to(r.dtype)).floor().to(torch.int64)
            pick = torch.minimum(pick, (n_candidate - 1).clamp(min=0))
            rank = candidate.to(torch.int64).cumsum(dim=1) - 1
            slot = candidate & (rank == pick[:, None]) & row[:, None]  # one slot per row
            new_cell = torch.where(selected, torch.full_like(keys, -1.0), keys).argmax(dim=1)
            cells = torch.where(slot, new_cell[:, None], cells)
            units = torch.where(slot, unit_argmax.gather(1, new_cell[:, None]), units)
            explored = slot
        # the budget guarantee: a row the probes pushed over the budget keeps its scored selection
        over = self._executed_cost(units, valid) > budget * (1.0 + BUDGET_SLACK)
        cells = torch.where(over[:, None], cells0, cells)
        units = torch.where(over[:, None], units0, units)
        return cells, units, explored & ~over[:, None], refreshed & ~over[:, None]

    def _finish(
        self,
        logits: Tensor,
        scores: Tensor,
        cells: Tensor,
        units: Tensor,
        valid: Tensor,
        *,
        mode: SelectionMode,
        gate: GateMode | None,
        tau: float,
        threshold: Tensor | None,
        log_prob: Tensor,
        explored: Tensor,
        refreshed: Tensor,
        dense: Tensor | None,
        executed_scores: bool = False,
        reserved: int = 0,
        reserved_units: bool = False,
    ) -> Selection:
        """The gradient-path tensors of a sparse decision (module docstring) and the Selection.

        ``executed_scores`` (:meth:`finish_recorded`): a cell's score is read at the unit that
        ran it instead of at the largest of ``scores``. In :meth:`select` the two are one unit
        (the decision was taken from these ``scores``); for a recorded decision ``scores`` are
        recomputed and may round a near-tie the other way, which would hand the cell mask's
        gradient to a unit that did not run."""
        B, N, M = (int(s) for s in logits.shape)
        K = int(cells.shape[1])
        device = logits.device
        scored = mode in SCORED_MODES
        units_follow_scores = mode != "random"
        valid_f = valid.to(torch.float32)
        hard_mask = torch.zeros(B, N, dtype=torch.float32, device=device).scatter_add(
            1, cells, valid_f
        )
        unit_argmax = scores.detach().argmax(dim=-1)  # [B, N]
        delta = torch.where(valid, units - unit_argmax.gather(1, cells), torch.zeros_like(units))
        unit_full = unit_argmax.scatter_add(1, cells, delta)  # the chosen unit per cell
        unit_hard = F.one_hot(unit_full, M).to(torch.float32)  # [B, N, M]
        if executed_scores:  # the chosen unit's score: the executed unit on the selected cells
            cell_scores = scores.gather(2, unit_full[:, :, None]).squeeze(2)
        else:
            cell_scores = scores.amax(dim=-1)  # [B, N], autograd intact
        if gate is None:
            cell_mask, unit_weights = hard_mask, unit_hard
        else:
            if self.unit_credit == "executed" and gate != "soft":
                # the chosen unit's score with its graph, the other units' as constants
                live = unit_hard > 0.0
                soft_units = torch.softmax(torch.where(live, scores, scores.detach()) / tau, dim=-1)
            else:
                soft_units = torch.softmax(scores / tau, dim=-1)
            if scored:
                assert threshold is not None
                soft_mask = torch.sigmoid((cell_scores - threshold[:, None]) / tau)
                if reserved:  # A8: the fast group is always written; no selection gradient
                    soft_mask = torch.cat(
                        [torch.ones_like(soft_mask[:, :reserved]), soft_mask[:, reserved:]], dim=1
                    )
            if gate == "soft":  # the relaxation; the controls (random, fixed) stay hard
                unit_weights = soft_units if scored else unit_hard
                cell_mask = soft_mask if scored else hard_mask
            else:  # ste, reinforce: hard values; straight-through unit weights where the unit
                # follows the scores (every mode but random), the straight-through cell mask in
                # ste only (reinforce trains the cells through log_prob, not through the mask)
                unit_weights = (
                    unit_hard + (soft_units - soft_units.detach())
                    if units_follow_scores
                    else unit_hard
                )
                cell_mask = (
                    hard_mask + (soft_mask - soft_mask.detach())
                    if scored and gate == "ste"
                    else hard_mask
                )
            if reserved_units:  # the fast cells' unit is fixed: a constant one-hot weight
                unit_weights = torch.cat(
                    [unit_hard[:, :reserved], unit_weights[:, reserved:]], dim=1
                )
        cost = self._executed_cost(units, valid)
        dense_cost = self._dense_cost
        if gate == "soft" and scored:  # every cell executes in the warm-up
            cost = torch.full_like(cost, dense_cost)
        if dense is not None:
            cell_mask = torch.where(dense[:, None], torch.ones_like(cell_mask), cell_mask)
            unit_weights = torch.where(
                dense[:, None, None], torch.softmax(logits.float(), dim=-1), unit_weights
            )
            cost = torch.where(dense, torch.full_like(cost, dense_cost), cost)
            log_prob = torch.where(dense, torch.zeros_like(log_prob), log_prob)  # not executed
        per_cell = self.cell_overhead + (unit_weights * self.cost_table.float()).sum(dim=-1)
        expected = self.shared_cost + (cell_mask * per_cell).sum(dim=1)
        if dense is not None:
            expected = torch.where(dense, torch.full_like(expected, dense_cost), expected)
        slot_cell = cell_mask.gather(1, cells)
        slot_unit = (
            unit_weights.gather(1, cells[:, :, None].expand(B, K, M))
            .gather(2, units[:, :, None])
            .squeeze(-1)
        )
        alpha = slot_cell * slot_unit * valid_f
        return Selection(
            cell_idx=cells,
            unit_idx=units,
            alpha=alpha,
            valid=valid,
            logits=logits,
            cost=cost,
            scorer_cost=self.scorer_cost,
            n_rounds_used=1,
            cell_mask=cell_mask,
            unit_weights=unit_weights,
            expected_cost=expected,
            log_prob=log_prob,
            explored=explored,
            refreshed=refreshed,
            dense_fallback=(
                dense if dense is not None else torch.zeros(B, dtype=torch.bool, device=device)
            ),
            threshold=threshold,
        )

    def finish_recorded(
        self,
        logits: Tensor,
        cells: Tensor,
        units: Tensor,
        valid: Tensor,
        threshold: Tensor,
        *,
        mode: SelectionMode,
        gate: GateMode | None,
        temperature: float = 1.0,
        dense_rows: Tensor | None = None,
        executed_scores: bool = True,
    ) -> Selection:
        """The :class:`Selection` of a decision that was already taken (P2.10, the split window's
        replay): ``cells`` / ``units`` / ``valid`` (``[B, K]``) and the threshold ``θ [B]`` are the
        recorded values of :meth:`select` on these rows, ``logits [B, N, M]`` the scorer's logits
        recomputed with autograd. Nothing is decided again: the hard values (``alpha``, the masks,
        the cost) follow the recorded slots exactly, and ``logits`` only supply the gradient path
        of ``gate`` (:meth:`_finish`). No probes are flagged (``explored`` / ``refreshed`` are
        the recording's). ``dense_rows`` (bool ``[B]``) are the rows that ran the dense step
        instead (:meth:`select`'s ``dense_rows``)."""
        B, N, _ = self._check_logits(logits)
        dense = self._check_dense_rows(dense_rows, mode, B, logits.device)
        if mode not in SCORED_MODES:
            raise ValueError(f"finish_recorded needs a scored mode {SCORED_MODES}, got {mode!r}")
        tau = self._check_gate_args(mode, gate, temperature, 0.0)
        K = int(cells.shape[1])
        device = logits.device
        none = torch.zeros(B, K, dtype=torch.bool, device=device)
        return self._finish(
            logits,
            self.scores(logits),
            cells,
            units,
            valid,
            mode=mode,
            gate=gate,
            tau=tau,
            threshold=threshold,
            log_prob=torch.zeros(B, dtype=torch.float32, device=device),
            explored=none,
            refreshed=none,
            dense=dense,
            executed_scores=executed_scores,
        )

    @staticmethod
    def _check_gate_args(
        mode: SelectionMode, gate: GateMode | None, temperature: float, epsilon: float
    ) -> float:
        if gate is not None and gate not in GATE_MODES:
            raise ValueError(f"gate must be None or one of {GATE_MODES}, got {gate!r}")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, int | float)
            or not math.isfinite(temperature)
            or temperature <= 0.0
        ):
            raise ValueError(f"temperature must be a finite number > 0, got {temperature!r}")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, int | float)
            or not math.isfinite(epsilon)
            or not 0.0 <= epsilon <= 1.0
        ):
            raise ValueError(f"epsilon must be a number in [0, 1], got {epsilon!r}")
        if epsilon > 0.0 and mode not in SCORED_MODES:
            raise ValueError(
                f"exploration (epsilon={epsilon!r}) post-processes a scored selection "
                f"({SCORED_MODES}); mode={mode!r} takes none (random / fixed are controls, dense "
                "has no selection)"
            )
        if gate == "reinforce" and mode != "topk":
            raise ValueError(
                f"gate='reinforce' samples cells by Gumbel-top-k and needs mode='topk', got "
                f"mode={mode!r}"
            )
        if mode == "dense" and gate in ("ste", "reinforce"):
            raise ValueError(
                f"gate={gate!r} has no hard decision to pass gradients through in mode='dense' "
                "(its units are soft already); use gate=None or 'soft'"
            )
        return float(temperature)

    def _check_age(
        self, age: Tensor | None, mode: SelectionMode, B: int, N: int, device: torch.device
    ) -> Tensor | None:
        if age is None:
            return None
        if mode not in SCORED_MODES:
            raise ValueError(
                f"refresh probes (age=...) post-process a scored selection ({SCORED_MODES}); "
                f"mode={mode!r} takes none"
            )
        if not isinstance(age, Tensor):
            raise TypeError(f"age must be a torch.Tensor (state.age), got {type(age).__name__}")
        if age.dtype not in _INDEX_DTYPES:
            raise TypeError(f"age must be an int64 or int32 tensor, got {age.dtype}")
        if age.shape != (B, N):
            raise ValueError(f"age must have shape [B, N] = {(B, N)}, got {tuple(age.shape)}")
        if age.device != device:
            raise ValueError(f"age must live on {device}, got {age.device}")
        return age.to(torch.int64)

    @staticmethod
    def _check_dense_rows(
        dense_rows: Tensor | None, mode: SelectionMode, B: int, device: torch.device
    ) -> Tensor | None:
        if dense_rows is None:
            return None
        if mode == "dense":
            raise ValueError(
                "dense_rows marks rows of a sparse mode that run dense; mode='dense' is"
            )
        if not isinstance(dense_rows, Tensor) or dense_rows.dtype != torch.bool:
            got = dense_rows.dtype if isinstance(dense_rows, Tensor) else type(dense_rows).__name__
            raise TypeError(f"dense_rows must be a bool torch.Tensor, got {got}")
        if dense_rows.shape != (B,):
            raise ValueError(
                f"dense_rows must have shape [B] = {(B,)}, got {tuple(dense_rows.shape)}"
            )
        if dense_rows.device != device:
            raise ValueError(f"dense_rows must live on {device}, got {dense_rows.device}")
        return dense_rows

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
