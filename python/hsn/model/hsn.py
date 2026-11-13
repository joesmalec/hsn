"""The HSN module (P1.5): one complete input step in dense mode, wired in the order of plan §2.2,
and the loop over a window of positions with document resets.

For every position ``t`` (:meth:`HSN.step`; ``q = q_{t−1}`` is the carried summary)::

    reset (HSN.forward, before the step)
    ẑ_t   = P(q_{t−1})                         predict BEFORE encoding x_t (global rule 2)
    z_t   = E(emb(x_t))
    z*_t  = emb(x_t).detach()  |  E_ema(emb(x_t))   the prediction target (P1.6; detached)
    ε_t   = z_t − ẑ_t
    u_t   = S([z_t ; q_{t−1} ; ctx_t])
    feats = scorer.features(u_t, ε_t, state, ctx_t, budget, round 0)   [B, N, 3d+8]
    logits_u = router(feats)                                           [B, N, M]
    dense:  α   = softmax(logits_u) over units
            h_t = PDGCell.step_dense(u_t, ε_t, state, α)              every cell, one round
            q_t = mean_i h[i, t]      (CellGraphState.with_dense: exact, no incremental drift)
    sparse: sel = router.select(logits_u, budget, K, k_min, routing, gate, τ, ε, age)  (P2.1/P2.2)
            PDGCell.step_sparse(u_t, ε_t, state, sel)                  K cells, one unit each (P2.5)
            q_t = q_{t−1} + Σ_selected Δh / N    (incremental; the trainer resyncs periodically)
    masked: h_cand = PDGCell.step_dense(u_t, ε_t, state, sel.unit_weights)   every cell (P2.2)
            h_t = m ⊙ h_cand + (1 − m) ⊙ h[t−1],  m = sel.cell_mask;  q_t the exact mean
    logits = D([q_t ; u_t])     the UPDATED summary

P1.10a: :meth:`HSN.forward` runs the same position body, but does the work that depends on the
inputs and parameters only once per window — ``emb``, ``E`` and ``z*`` over ``[B, T]`` before the
loop, the position-independent scorer features and the edge weights once, ``D`` over the stacked
``[q_t ; u_t]`` after the loop, and the statistics from the stacks. Only ``P``, ``ε``, ``S``, the
scorer and the round run per position. :meth:`HSN.step` stays the per-position reference; the
window matches it to rounding (batched matmuls round differently from per-position ones), not bit
for bit. ``forward(..., capture=True)`` returns the per-position cell states and gates.

The dense reference uses the same linear scorer as the sparse model, with ``α = softmax(logits)``
over all units and every cell updated (Appendix H.5), so the measured parameter count is exactly
Appendix C.1: :meth:`HSN.param_report` lists the twelve C.1 rows plus amendment A1's ``cell_norm``
row (P1.5a), amendment A4's ``cand_norm`` row (P1.13; 0 unless ``model.cand_norm = rms``) and their
``total`` (1,820,617 for ``configs/ref_a1.yaml``; 1,820,361 before A1). The
learned edge weights of the P1.4 graph (``N·(D_g + random_edges)`` logits, 128 for A.1) are
trainable parameters that Appendix C does not list; the report gives them
as ``edges`` and ``total_trainable = total + edges``, which is what a run manifest's
``param_count`` must record (§0.5: measured, not assumed).

Two values the step needs live outside ``model.*``: the graph topology seed (``run.seed``, P1.4)
and the normaliser of the position feature (``train.unroll``). Both are required keyword
arguments of :class:`HSN`; :meth:`HSN.from_config` reads them from a :class:`~hsn.config.RunConfig`.

P1.6: ``StepOutput.z_target`` is the target of the prediction loss (:mod:`hsn.losses`), chosen by
``model.predictor_target``: the detached byte embedding of ``x_t`` (``fixed_embedding``) or the
output of the stop-gradient EMA copy of ``E`` (``ema_encoder``,
:class:`~hsn.model.encoder.EMATarget`, built only in that mode and updated by
:meth:`HSN.update_target` after each optimiser step with ``train.ema_tau``, a third keyword that
is required in that mode). The target is detached in both modes and never feeds the step; ``ẑ``
still comes from ``q_{t−1}`` alone.

P2.5: :meth:`HSN.step` chooses the path from ``model.routing`` (``dense`` → the dense reference,
unchanged; ``topk`` / ``greedy_budget`` / ``random`` / ``fixed`` → the sparse round) unless ``mode``
is given explicitly: ``mode="dense"`` always runs the dense reference, ``mode="sparse"`` and
``mode="masked"`` need a sparse ``model.routing``. The scorer lives inside the P2.1
:class:`~hsn.model.router.Router` (``HSN.router``; ``HSN.scorer`` is a property on it, so the
state dict keys are ``router.scorer.*``). ``StepOutput.cost`` is the executed cost as a fraction of
the dense step as a detached 0-dim tensor (1.0 dense; the batch mean of ``Selection.cost``
otherwise, so reading it costs no host synchronisation; the sparse round itself synchronises three
times per position in the index checks of ``messages_selected``, ``forward_selected`` and
``gated_delta_update``; P2.8's capturable round skips them, ``static=True`` below) and
``StepOutput.selection`` the executed
:class:`~hsn.model.router.Selection` (``None`` dense). A :class:`~hsn.profiler.Profiler` passed to
``step`` times the P0.6 segments ``shared_path``, ``scorer``, ``messages`` / ``selected_cells``
(sparse round only; the dense cell step is partitioned by
:func:`hsn.kernels.costs.profile_dense_costs`) and ``decode``.

P2.2 (training through hard selection): the selection carries its gradient path and the routing
controls (:mod:`hsn.model.router`), driven by the :class:`~hsn.train.schedules.SelectionSchedule` at
``HSN.train_step`` (the trainer sets it before every window). In training the gate of the step
(``train.gate_mode`` after ``train.soft_until_step`` steps, ``soft`` before) and the temperature
and exploration ramps reach :meth:`Router.select`; evaluation takes the hard, gradient-free
decision with no exploration and no dense fallback (refresh probes stay on: ``model.age_max`` is a
property of the model). The straight-through gradient reaches the sparse round through
``Selection.alpha`` (``cell_mask[cell] · unit_weights[cell, unit]`` on the valid slots, exactly
1.0 forward). Mode ``masked`` is the hard selection executed on the dense path — every cell computes
its candidate with the selection's per-cell unit weights, only the selected cells are written — the
reference the sparse round is checked against, and the path the **soft warm-up** runs on whatever
the mode (a soft ``cell_mask`` means every cell executes with a fractional weight; the selection is
charged the dense cost there). The **dense fallback** is drawn once per window per row in
:meth:`HSN.forward` with probability ``train.p_dense`` (training only, from the CPU RNG, so a
checkpoint replays it): a dense-fallback row executes the P1.5 dense step — inside the sparse round
by branching that row to ``step_dense``, inside the masked round through an all-ones mask — and is
charged the dense cost in ``Selection.cost``. The masked path computes every cell, so its
throughput is not a sparse-efficiency number.

P2.8 (launch-overhead reduction, :mod:`hsn.train.compile`): :meth:`HSN.forward` is three pieces —
:meth:`HSN._window_prologue` (the window-invariant work), :meth:`HSN._position` (the loop body:
reset, ``P``, ``ε``, ``S``, the scorer and the round) and :meth:`HSN._window_epilogue` (the decoder,
the statistics, ``info``) — which the compiled window reuses: it compiles blocks of
:meth:`HSN._position` with ``static=True`` (the capturable sparse round: no host synchronisation,
the unit bank's static pass) and the window's constants passed as tensors
(:class:`~hsn.model.cell.WindowConsts`). The eager path is unchanged, bit for bit.

A6 (``train.feedback_detach``, ``docs/amendments/A6_feedback_gradient.md``; opt-in): the feedback
reads of the recurrent state can be constants in the backward pass, with every forward value
unchanged. ``summary``: ``S`` reads ``q_{t−1}.detach()``; ``eps``: the cell and the scorer read
``z_t − ẑ_t.detach()`` (``StepOutput.eps`` and the prediction loss keep ``ẑ_t``'s gradient into
``P`` and ``q``); ``state``: the cell's units, gate and messages read ``h.detach()``
(:attr:`PDGCell.detach_state_read`); ``router``: the scorer's features are detached, so a
selection gradient trains the scorer's weights only. With all four the state's backward recurrence
is the gated carry ``(1 − b) ⊙ h`` and the linear summary update, whose Jacobian has no singular
value above 1: a gradient cannot grow on its way back through a window.

A6, the other half (``train.state_grad_clip``; opt-in): at the entry of every position the
gradient that the position passes back into the state is limited per batch row
(:func:`hsn.model.gradclip.clip_row_grad`: ``dL/dq`` to the limit, ``dL/dh`` to ``limit /
sqrt(N)``; ``train.state_grad_ratio``: both to that multiple of the batch's typical live row at
the position, a limit without a scale). The forward values are unchanged and a row under the
limit keeps its gradient unscaled, so a healthy window is full backpropagation through time (to
the rounding of autograd's sums). Tested on the A5 recipe, the limit did not remove the drops (exps
354, 355, 364); the cut above did.

``model.ablate_shared_u`` (opt-in; the "remove shared path" ablation of the P2.13 matrix): the
:class:`~hsn.model.cell.PDGCell` reads ``[ε_t ; h_i ; m_i]`` (width ``3d``) instead of
``[u_t ; ε_t ; h_i ; m_i]``, on every round and window (:mod:`hsn.model.cell`). This module is
unchanged by it:
``u_t`` is computed as before and still read by the scorer's features and the decoder, and is
handed to the cell, which does not read it. :meth:`HSN.param_report` then matches the closed form
of :func:`hsn.train.memory_plan.expected_param_count` at the ``3d`` width.

Amendment A8 (``model.fast_cells = F``, ``docs/amendments/A8_fast_cells.md``; opt-in, ``topk``
only): cells ``0 .. F − 1`` are selected at every position (:meth:`Router.select`'s ``reserved``)
and the router fills the other ``k_select − F`` slots from the remaining cells. The cell, the
units, the graph, the gate and the state layout are unchanged. With ``model.fast_readout`` the
fast group has its own read ``f = mean_{i < F} h[i]``: ``u_t = S([z_t ; q_{t−1} ; ctx_t ;
f_{t−1}])`` (a ``summary`` read under A6), ``D([q_t ; u_t ; r_t ; f_t])``, and the selected-cell
readout ``r_t`` averages the routed slots only. The split and fused windows refuse both keys.

Amendment A9 (``model.address_ngram = n > 1``, ``docs/amendments/A9_ngram_address.md``; opt-in,
needs ``model.key_address``): the scorer also gets the address of the last ``m`` bytes and of the
``m`` bytes before the current one, for ``m = 2 .. n`` (:meth:`Scorer.address_features`). A
window derives the older bytes from its own tokens; the trainer carries one byte across windows
(A5's ``prev_tokens``), so at the first ``n − 1`` positions of a window that continues a document
the bytes further back than that one read as BOS. A document that starts at a window boundary is
exact. :meth:`step` takes ``prev_tokens`` as ``[B, n]`` (``x_{t−1}`` first) for exact features;
``[B]`` means the older bytes are BOS.

Still out of scope: one round only (the :class:`~hsn.model.halting.HaltingHead` is owned but not
called until P2.7), no custom kernels (P2.9–P2.11); ``n_rounds`` is 1.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple, Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import FEEDBACK_PATHS, GateMode, ModelConfig, RunConfig
from hsn.data.synthetic import BOS
from hsn.model.cell import PDGCell, WindowConsts, profiler_segment
from hsn.model.embeddings import ByteEmbedding, ContextEmbedding
from hsn.model.encoder import Decoder, EMATarget, Encoder, Predictor, SharedTransform
from hsn.model.gradclip import clip_row_grad
from hsn.model.halting import HaltingHead
from hsn.model.router import SCORED_MODES, Router, Scorer, Selection
from hsn.model.state import CellGraphState
from hsn.profiler import Profiler
from hsn.train.schedules import SelectionSchedule

__all__ = ["PARAM_REPORT_ROWS", "HSN", "StepMode", "StepOutput"]

StepMode = Literal["dense", "masked", "sparse"]
_INDEX_DTYPES = (torch.int64, torch.int32)

#: Appendix C.1 rows in table order, keyed as :func:`hsn.train.memory_plan.expected_param_count`
#: keys them.
PARAM_REPORT_ROWS: tuple[str, ...] = (
    "byte_emb",
    "ctx_emb",
    "encoder",
    "predictor",
    "shared",
    "units",
    "w_out",
    "v_msg",
    "gate",
    "cell_norm",  # P1.5a (amendment A1): the RMSNorm gain over the 4d cell input (3d: ablation)
    "cand_norm",  # P1.13 (amendment A4): the RMSNorm gain before the candidate's tanh (0: none)
    "scorer",
    "halting",
    "decoder",
)


@dataclass
class StepOutput:
    """Everything one position produces: next-symbol ``logits [B, V]``, the encoded input ``z``,
    the prediction ``z_hat`` made before encoding, the detached prediction target ``z_target``
    (P1.6, by ``model.predictor_target``), the error ``eps = z − z_hat`` and the shared features
    ``u`` (all ``[B, d]``), the updated ``state``, the executed ``cost`` (a detached 0-dim tensor in
    the state dtype: the fraction of the dense step, 1.0 dense and the batch mean of
    ``Selection.cost`` otherwise), ``n_rounds``, ``aux`` statistics (detached 0-dim tensors:
    ``state_abs_max``, ``eps_abs_mean``, ``alpha_entropy`` — the entropy of the scorer's softmax
    over units in every mode) and, in a routed mode, the :class:`~hsn.model.router.Selection`
    that was executed (``None`` dense)."""

    logits: Tensor
    z: Tensor
    z_hat: Tensor
    z_target: Tensor
    eps: Tensor
    u: Tensor
    state: CellGraphState
    cost: Tensor
    n_rounds: int
    aux: dict[str, Tensor]
    selection: Selection | None = None


class _RoundResult(NamedTuple):
    """One position's round: the updated state and, when captured, the gate ``b`` of the written
    cells (``[B, N, d]``, NaN where not written; detached) and the written mask (bool
    ``[B, N]``)."""

    state: CellGraphState
    gate: Tensor | None = None
    written: Tensor | None = None


class _Controls(NamedTuple):
    """The P2.2 controls of one step or window (:meth:`HSN._controls`), read from the schedule at
    ``HSN.train_step`` once: the gate (``None`` in evaluation), the temperature τ, the exploration
    rate ε and whether the routing probes (exploration and refresh) run. A window reads them once
    (``train_step`` does not change inside it), so a compiled window sees them as arguments and
    never reads the step counter (P2.8)."""

    gate: GateMode | None
    tau: float
    epsilon: float
    probes: bool


class _PositionResult(NamedTuple):
    """One position of a window (:meth:`HSN._position`): the updated state, the prediction
    ``z_hat``, the error ``eps``, the shared features ``u`` (``[B, d]``), the scorer's
    ``unit_logits [B, N, M]`` (state dtype, autograd intact), the executed selection (``None``
    dense) and, with capture, the gate and written mask of :class:`_RoundResult`."""

    state: CellGraphState
    z_hat: Tensor
    eps: Tensor
    u: Tensor
    unit_logits: Tensor
    selection: Selection | None
    gate: Tensor | None
    written: Tensor | None


#: The per-position routed records a window stacks (:meth:`HSN._selection_record`), in order.
SELECTION_RECORD_KEYS: tuple[str, ...] = (
    "cell_idx",
    "unit_idx",
    "valid",
    "unit_logits",
    "cost_per_row",
    "expected_cost",
    "route_log_prob",
    "explored",
    "refreshed",
    "dense_fallback",
)


def _gate_of_written(b: Tensor, written: Tensor) -> Tensor:
    """The detached gate ``b [B, N, d]`` of the written cells (bool ``[B, N]``), NaN elsewhere."""
    return b.detach().masked_fill(~written[:, :, None], float("nan"))


def _gate_of_slots(b: Tensor, cell_idx: Tensor, valid: Tensor, n_cells: int) -> Tensor:
    """The sparse round's per-slot gate ``b [B, K, d]`` laid out per cell, ``[B, N, d]``: the
    valid slots' values at their cells, NaN at every cell no slot wrote. Padded slots go to a dummy
    row ``N`` that is cut off (valid cells are distinct within a row)."""
    B, K, d = (int(n) for n in b.shape)
    index = torch.where(valid, cell_idx, torch.full_like(cell_idx, n_cells))
    wide = b.new_full((B, n_cells + 1, d), float("nan"))
    return wide.scatter(1, index[:, :, None].expand(B, K, d), b.detach())[:, :n_cells]


def _window_positions(pos: Tensor, reset: Tensor) -> Tensor:
    """The session position before each step of a window, int64 ``[B, T]``, from the start
    positions ``pos [B]`` and the window's ``reset [B, T]`` (no host synchronisation): ``t − s``
    after the last reset ``s <= t`` of a row, ``pos + t`` in a row without one. Every round
    advances ``pos`` by exactly one, so this holds in every routing mode."""
    T = int(reset.shape[1])
    steps = torch.arange(T, dtype=torch.int64, device=reset.device)
    last = torch.where(reset, steps, torch.full_like(steps, -1)).cummax(dim=1).values
    return torch.where(last >= 0, steps - last, pos[:, None] + steps)


class _Prologue(NamedTuple):
    """The window-invariant work of a window (:meth:`HSN._window_prologue`): ``z``, ``z_target``
    (``[B, T, d]``), the context embeddings, the scorer's position-independent features, their
    per-position views and, in the P1.10b dense window, the counters ``(start_age, zero_age,
    positions, positions_after)`` (``age [B, N]``, ``pos [B, T]`` before and after each step)."""

    z: Tensor
    z_target: Tensor
    context: Tensor
    scalars: Tensor
    z_steps: tuple[Tensor, ...]
    ctx_steps: tuple[Tensor, ...]
    scalar_steps: tuple[Tensor, ...]
    counters: tuple[Tensor, Tensor, Tensor, Tensor] | None

    def dense_counters(self, t: int) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        """The P1.10b dense window's ``(age, pos)`` before and after position ``t`` (``None``
        outside the dense window): the start age at ``t = 0`` and 0 afterwards."""
        if self.counters is None:
            return None
        start_age, zero_age, positions, positions_after = self.counters
        age = start_age if t == 0 else zero_age
        return age, positions[:, t], zero_age, positions_after[:, t]


def _stack_record(values: Sequence[Tensor]) -> Tensor:
    """A per-position record over a window: per-row values (``[B, …]``) stacked at dim 1, batch
    means (0-dim) at dim 0 — the layout of :meth:`HSN.forward`'s routed ``info``."""
    return torch.stack(list(values), dim=1 if values[0].ndim > 0 else 0)


class HSN(nn.Module):
    """The predictive cell model: embeddings, ``E``, ``P``, ``S``, the :class:`PDGCell`, the
    :class:`Router` (the :class:`Scorer` plus P2.1 selection; ``model.cost_profile`` is read here
    and an unaffordable ``model.k_min`` / ``model.fixed_route`` raises at construction), the
    :class:`HaltingHead` and ``D``, built from ``cfg`` (``model.*``) plus ``seed`` (``run.seed``),
    ``unroll`` (``train.unroll``) and, when ``model.predictor_target = ema_encoder``, ``ema_tau``
    (``train.ema_tau``) for the :class:`~hsn.model.encoder.EMATarget` copy of ``E`` (``ema_target``;
    ``None`` otherwise). ``schedule`` is the P2.2 :class:`~hsn.train.schedules.SelectionSchedule`
    (``train.gate_mode`` and the ramps; the config defaults when ``None``: soft gates, τ = 1, no
    exploration, no dense fallback). ``feedback_detach`` is A6's ``train.feedback_detach`` (module
    docstring; empty: full backpropagation through time) and ``state_grad_clip`` its
    ``train.state_grad_clip`` (``None``: off)."""

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        seed: int,
        unroll: int,
        ema_tau: float | None = None,
        schedule: SelectionSchedule | None = None,
        feedback_detach: Collection[str] = (),
        state_grad_clip: float | None = None,
        state_grad_ratio: float | None = None,
        unit_credit: str = "softmax",
        ste_scope: str = "selected",
    ) -> None:
        super().__init__()
        if ste_scope not in ("selected", "all"):
            raise ValueError(f"ste_scope must be 'selected' or 'all', got {ste_scope!r}")
        #: ``train.ste_scope`` (opt-in): "all" trains the routed window on the masked round.
        self.ste_scope = ste_scope
        if state_grad_clip is not None and (
            isinstance(state_grad_clip, bool)
            or not isinstance(state_grad_clip, int | float)
            or not 0.0 < state_grad_clip < float("inf")
        ):
            raise ValueError(
                f"state_grad_clip must be None or a finite number > 0, got {state_grad_clip!r}"
            )
        if state_grad_ratio is not None and (
            isinstance(state_grad_ratio, bool)
            or not isinstance(state_grad_ratio, int | float)
            or not 1.0 < state_grad_ratio < float("inf")
        ):
            raise ValueError(
                f"state_grad_ratio must be None or a finite number > 1, got {state_grad_ratio!r}"
            )
        #: A6: the relative per-row limit (a multiple of the batch's typical live row); None: off.
        self.state_grad_ratio: float | None = (
            None if state_grad_ratio is None else float(state_grad_ratio)
        )
        #: A6: the per-row limit on dL/dq at every position (dL/dh: this / sqrt(N)); None: off.
        self.state_grad_clip: float | None = (
            None if state_grad_clip is None else float(state_grad_clip)
        )
        self._clip_h = (
            None if state_grad_clip is None else float(state_grad_clip) / cfg.n_cells**0.5
        )
        if isinstance(feedback_detach, str) or not isinstance(feedback_detach, Collection):
            raise TypeError(
                f"feedback_detach must be a collection of path names, got {feedback_detach!r}"
            )
        unknown = sorted(set(feedback_detach) - set(FEEDBACK_PATHS))
        if unknown:
            raise ValueError(
                f"feedback_detach names unknown paths {unknown}; known: {list(FEEDBACK_PATHS)}"
            )
        #: A6: the feedback reads that are constants in the backward pass, in FEEDBACK_PATHS order.
        self.feedback_detach: tuple[str, ...] = tuple(
            name for name in FEEDBACK_PATHS if name in feedback_detach
        )
        self._cut_summary = "summary" in self.feedback_detach
        self._cut_eps = "eps" in self.feedback_detach
        self._cut_router = "router" in self.feedback_detach
        if schedule is not None and not isinstance(schedule, SelectionSchedule):
            raise TypeError(
                f"schedule must be a SelectionSchedule or None, got {type(schedule).__name__}"
            )
        self.cfg = cfg
        self.seed = seed
        self.unroll = unroll
        self.ema_tau = ema_tau
        self.schedule = schedule if schedule is not None else SelectionSchedule.neutral()
        self.train_step = 0
        self.byte_emb = ByteEmbedding.from_config(cfg)
        self.ctx_emb = ContextEmbedding.from_config(cfg)
        self.encoder = Encoder(cfg)
        self.ema_target: EMATarget | None = None
        if cfg.predictor_target == "ema_encoder":
            if ema_tau is None:
                raise ValueError(
                    "model.predictor_target=ema_encoder needs train.ema_tau: pass ema_tau=... or "
                    "build the model with HSN.from_config(run)"
                )
            self.ema_target = EMATarget(self.encoder, ema_tau)
        elif cfg.predictor_target != "fixed_embedding":
            raise ValueError(
                f"model.predictor_target={cfg.predictor_target!r} is not implemented; known: "
                "fixed_embedding, ema_encoder"
            )
        self.predictor = Predictor(cfg)
        # A8: the seed only draws the extra input columns of model.fast_readout
        self.shared = SharedTransform(cfg, seed=seed) if cfg.fast_readout else SharedTransform(cfg)
        self.cell = PDGCell(cfg, seed=seed)
        self.cell.detach_state_read = "state" in self.feedback_detach  # A6
        # the scorer is built here, in the P1.5 position, so seeded initialisation is unchanged
        self.router = Router.from_config(cfg, unroll=unroll, seed=seed)
        self.router.set_unit_credit(unit_credit)  # train.unit_credit (opt-in)
        self.halting = HaltingHead(cfg)
        self.decoder = Decoder(cfg, seed=seed)
        # The fixed route stays a host tensor: Router._check_fixed reads it on the host anyway.
        self._fixed_route: Tensor | None = (
            torch.tensor(cfg.fixed_route, dtype=torch.int64) if cfg.routing == "fixed" else None
        )

    @property
    def scorer(self) -> Scorer:
        """The P1.5 :class:`Scorer`, owned by :attr:`router` since P2.5 (``router.scorer``)."""
        return self.router.scorer

    @classmethod
    def from_config(cls, run: RunConfig) -> Self:
        """``HSN(run.model, seed=run.seed, unroll=run.train.unroll, ema_tau=run.train.ema_tau,
        schedule=SelectionSchedule.from_config(run.train),
        feedback_detach=run.train.feedback_detach, state_grad_clip=run.train.state_grad_clip,
        state_grad_ratio=run.train.state_grad_ratio)``."""
        if not isinstance(run, RunConfig):
            raise TypeError(f"from_config needs a RunConfig, got {type(run).__name__}")
        return cls(
            run.model,
            seed=run.seed,
            unroll=run.train.unroll,
            ema_tau=run.train.ema_tau,
            schedule=SelectionSchedule.from_config(run.train),
            feedback_detach=run.train.feedback_detach,
            state_grad_clip=run.train.state_grad_clip,
            state_grad_ratio=run.train.state_grad_ratio,
            unit_credit=run.train.unit_credit,
            ste_scope=run.train.ste_scope,
        )

    def set_train_step(self, step: int) -> None:
        """The optimiser step the schedules read (the trainer calls this before every window)."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"step must be an int >= 0, got {step!r}")
        self.train_step = step

    # -- placement --------------------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.decoder.fc1.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.decoder.fc1.weight.dtype

    def init_state(self, B: int, device: torch.device | str | None = None) -> CellGraphState:
        """A fresh all-zero state for ``B`` sessions on ``device`` (the model's device by default),
        in the parameters' dtype."""
        dev = self.device if device is None else torch.device(device)
        return CellGraphState.zeros(B, self.cfg.n_cells, self.cfg.d, dev, self.dtype)

    def reset_state(self, state: CellGraphState, mask: Tensor) -> CellGraphState:
        """The P0.5 :class:`~hsn.train.loop.SequenceModel` protocol (P1.7): the sessions where
        ``mask [B]`` (bool) is True back to the zero state, the others untouched
        (:meth:`CellGraphState.reset`). :meth:`forward` applies its own ``reset`` argument position
        by position; the trainer never calls this itself."""
        self._check_state(state)
        return state.reset(mask)

    # -- one position -----------------------------------------------------------------------------

    def step(
        self,
        x_t: Tensor,
        ctx_t: Tensor,
        state: CellGraphState,
        mode: StepMode | None = None,
        *,
        profiler: Profiler | None = None,
        dense_rows: Tensor | None = None,
        prev_tokens: Tensor | None = None,
    ) -> StepOutput:
        """One input position in the §2.2 order (see the module docstring); returns a
        :class:`StepOutput` whose ``state`` is the functional update of ``state`` (autograd intact).

        ``x_t`` and ``ctx_t`` are int64/int32 ``[B]`` on the state's device; ``state`` must have
        this model's ``N`` and ``d`` and the parameters' dtype and device. ``mode=None`` follows
        ``model.routing`` (``dense`` → the dense reference, any selection mode → the sparse round);
        ``"dense"`` runs the dense reference whatever the routing; ``"sparse"`` and ``"masked"``
        (the selection applied on the dense path, P2.2) raise ``ValueError`` when ``model.routing``
        is ``dense``. In the soft warm-up a sparse round runs as the masked round (every cell
        executes). ``dense_rows`` (bool ``[B]``, routed modes only) marks the rows that execute the
        dense step instead. ``profiler`` (P0.6) times the segments named in the module docstring;
        ``None`` records nothing and never synchronises.
        """
        cfg = self.cfg
        if mode is None:
            mode = "dense" if cfg.routing == "dense" else "sparse"
        elif mode in ("sparse", "masked"):
            if cfg.routing == "dense":
                raise ValueError(
                    f"mode={mode!r} needs a selection mode in model.routing (topk, greedy_budget, "
                    "random, fixed); this model has model.routing=dense"
                )
        elif mode != "dense":
            raise ValueError(f"mode must be 'dense', 'masked', 'sparse' or None, got {mode!r}")
        if profiler is not None and not isinstance(profiler, Profiler):
            raise TypeError(f"profiler must be a hsn.profiler.Profiler or None, got {profiler!r}")
        self._check_step_inputs(x_t, ctx_t, state)
        dense_rows = self._check_dense_rows(dense_rows, mode, state)
        older: Tensor | None = None
        if self.cfg.key_address:
            if prev_tokens is not None and prev_tokens.ndim == 2:  # A9: [B, n], x_{t-1} first
                n = self.cfg.address_ngram
                if prev_tokens.shape != (state.batch_size, n):
                    raise ValueError(
                        f"prev_tokens must be [B] or [B, model.address_ngram] = "
                        f"{(state.batch_size, n)}, got {tuple(prev_tokens.shape)}"
                    )
                prev_tokens, older = prev_tokens[:, 0], prev_tokens[:, 1:]
            prev_tokens = self._previous_tokens(x_t, prev_tokens)
        return self._step_impl(
            x_t, ctx_t, state, mode, profiler, dense_rows, prev_tokens=prev_tokens, older=older
        )

    def _step_impl(
        self,
        x_t: Tensor,
        ctx_t: Tensor,
        state: CellGraphState,
        mode: StepMode,
        profiler: Profiler | None,
        dense_rows: Tensor | None,
        *,
        static: bool = False,
        prev_tokens: Tensor | None = None,
        older: Tensor | None = None,
    ) -> StepOutput:
        """:meth:`step` after its checks (``mode`` resolved). ``static`` (P2.8: the captured
        serving step of :func:`hsn.train.compile.compile_step`) runs the capturable sparse
        round."""
        cfg = self.cfg
        B = state.batch_size
        controls = self._controls()
        state = self._clip_state_grad(state)  # A6: what this position passes back is limited

        with profiler_segment(profiler, "shared_path", B):
            z_hat, z, z_target, eps, u = self._shared_path(x_t, ctx_t, state)
        # P1.7: under autocast E, P and S emit the autocast dtype (bf16 / fp16); the state path
        # (scorer features, the softmax over units, the cell update and q) runs in the state's
        # dtype, the fp32 master precision of §3.7. Without autocast these casts are no-ops.
        u_s, eps_s = u.to(state.dtype), self._eps_read(z, z_hat, eps).to(state.dtype)

        with self.cell.graph.step():
            with profiler_segment(profiler, "scorer", B):
                feats = self.scorer.features(
                    u_s.detach() if self._cut_router else u_s,
                    eps_s.detach() if self._cut_router else eps_s,
                    self._router_state(state),
                    ctx_t,
                    budget_frac=cfg.budget,
                    round_idx=0,
                    ema=None,
                    **(
                        {
                            "tokens": x_t,
                            "prev_tokens": self._reset_previous(prev_tokens, state.pos == 0, x_t),
                            **self._older_bytes(older, state.pos, x_t),
                        }
                        if cfg.key_address
                        else {}
                    ),
                )
                unit_logits, alpha, selection = self._score(
                    feats, state, mode, dense_rows, controls=controls
                )
            new_state = self._update(
                u_s,
                eps_s,
                state,
                mode,
                alpha,
                selection,
                unit_logits,
                profiler,
                dense_rows,
                controls=controls,
                static=static,
                dense_indices=self._dense_indices(dense_rows, bucket=static),
            ).state
        if selection is None:
            cost = torch.ones((), dtype=state.dtype, device=state.device)
        else:
            cost = selection.cost.detach().mean().to(state.dtype)

        with profiler_segment(profiler, "decode", B):
            readout = (
                self._selected_readout(new_state, selection, controls.gate, mode)
                if cfg.selected_readout
                else None
            )
            logits = self._decode(new_state.q, u, readout)

        with torch.no_grad():
            probs = alpha if alpha is not None else torch.softmax(unit_logits, dim=-1)
            log_alpha = F.log_softmax(unit_logits, dim=-1)
            aux = {
                "state_abs_max": new_state.h.abs().amax(),
                "eps_abs_mean": eps.abs().mean(),
                "alpha_entropy": -(probs * log_alpha).sum(dim=-1).mean(),
            }
        return StepOutput(
            logits=logits,
            z=z,
            z_hat=z_hat,
            z_target=z_target,
            eps=eps,
            u=u,
            state=new_state,
            cost=cost,
            n_rounds=1,
            aux=aux,
            selection=selection,
        )

    def _shared_path(
        self, x_t: Tensor, ctx_t: Tensor, state: CellGraphState
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """The shared path of §2.2 steps 2–3 from the carried summary: ``(z_hat, z, z_target, eps,
        u)``, each ``[B, d]``; the prediction is made before ``x_t`` is encoded (global rule 2)."""
        q_prev = state.q
        z_hat = self.predictor(q_prev)  # before x_t is encoded
        emb = self.byte_emb(x_t)
        z = self.encoder(emb)
        z_target = self._prediction_target(emb)  # detached; a function of x_t only
        eps = z - z_hat
        u = self.shared(self._shared_input(z, q_prev, self.ctx_emb(ctx_t), state.h))
        return z_hat, z, z_target, eps, u

    def _shared_input(self, z: Tensor, q_prev: Tensor, ctx: Tensor, h_prev: Tensor) -> Tensor:
        """The input of ``S``: ``[z_t ; q_{t−1} ; ctx_t]``, and with ``model.fast_readout`` (A8)
        the fast group's mean ``f_{t−1}`` after it. Both state reads are ``summary`` reads (A6)."""
        parts = [z, self._summary_read(q_prev), ctx]
        if self.cfg.fast_readout:
            parts.append(self._summary_read(self._fast_read(h_prev)))
        return torch.cat(parts, dim=-1)

    def _fast_read(self, h: Tensor) -> Tensor:
        """A8: ``f = mean_{i < F} h[i]``, ``[B, d]`` from ``h [B, N, d]``."""
        return h[:, : self.cfg.fast_cells].mean(dim=1)

    # -- A6: the feedback reads (values unchanged; see the module docstring) ----------------------

    def _summary_read(self, q_prev: Tensor) -> Tensor:
        """What ``S`` reads of the carried summary: a constant in the backward with ``summary``."""
        return q_prev.detach() if self._cut_summary else q_prev

    def _eps_read(self, z: Tensor, z_hat: Tensor, eps: Tensor) -> Tensor:
        """The error the cell and the scorer read: ``z − ẑ.detach()`` with ``eps`` (the same
        value; ``z`` keeps its gradient, the prediction does not), else ``eps`` itself."""
        return z - z_hat.detach() if self._cut_eps else eps

    def _clip_state_grad(self, state: CellGraphState) -> CellGraphState:
        """``state`` with the gradient into ``h`` and ``q`` limited per row
        (``train.state_grad_clip``, ``train.state_grad_ratio``); ``state`` itself when both are
        off or nothing is recorded (evaluation, serving)."""
        if (self._clip_h is None and self.state_grad_ratio is None) or not torch.is_grad_enabled():
            return state
        return CellGraphState._make(
            h=clip_row_grad(state.h, self._clip_h, ratio=self.state_grad_ratio),
            q=clip_row_grad(state.q, self.state_grad_clip, ratio=self.state_grad_ratio),
            age=state.age,
            pos=state.pos,
        )

    def _router_state(self, state: CellGraphState) -> CellGraphState:
        """The state whose ``h`` the scorer's features read (detached with ``router``)."""
        if not self._cut_router:
            return state
        return CellGraphState._make(h=state.h.detach(), q=state.q, age=state.age, pos=state.pos)

    def _gate(self) -> GateMode | None:
        """The gate of this step: the schedule's in training, ``None`` (hard, no gradient path) in
        evaluation."""
        return self.schedule.gate(self.train_step) if self.training else None

    def _controls(self) -> _Controls:
        """The P2.2 controls of this step (:class:`_Controls`): the schedule's gate, temperature
        and exploration rate in training (the probes — exploration and refresh — post-process a
        hard scored decision, so not in the soft warm-up and never for the controls), the hard
        decision in evaluation."""
        schedule, step, training = self.schedule, self.train_step, self.training
        scored = self.cfg.routing in SCORED_MODES
        gate = self._gate()
        tau = schedule.temperature(step) if training else 1.0
        probes = scored and gate != "soft"
        epsilon = schedule.epsilon(step) if training and probes else 0.0
        return _Controls(gate=gate, tau=tau, epsilon=epsilon, probes=probes)

    def _select(
        self,
        unit_logits: Tensor,
        state: CellGraphState,
        dense_rows: Tensor | None,
        controls: _Controls | None = None,
    ) -> Selection:
        """The selection of this model's ``model.routing`` for one round, with the P2.2 gate and
        controls of this step (:meth:`_controls` unless given)."""
        cfg = self.cfg
        gate, tau, epsilon, probes = self._controls() if controls is None else controls
        fast: dict[str, Any] = {}  # A8: only named when set, so wrappers of select keep working
        if cfg.fast_cells:
            fast["reserved"] = cfg.fast_cells
        if cfg.fast_units == "fixed":
            fast["reserved_units"] = True
        return self.router.select(
            unit_logits,
            budget=cfg.budget,
            k_max=cfg.k_select,
            k_min=cfg.k_min,
            mode=cfg.routing,
            fixed=self._fixed_route,
            generator=None,
            gate=gate,
            temperature=tau,
            epsilon=epsilon,
            age=state.age if probes else None,
            dense_rows=dense_rows,
            **fast,
        )

    def _score(
        self,
        feats: Tensor,
        state: CellGraphState,
        mode: StepMode,
        dense_rows: Tensor | None,
        *,
        controls: _Controls | None = None,
    ) -> tuple[Tensor, Tensor | None, Selection | None]:
        """The scorer logits ``[B, N, M]`` (state dtype) of the features, then either the dense
        mixing weights ``α = softmax(logits)`` (dense mode; no selection) or the routed
        :class:`~hsn.model.router.Selection` of this step (``α`` is ``None``)."""
        unit_logits = self.router(feats).to(state.dtype)  # [B, N, M]
        if mode == "dense":
            return unit_logits, torch.softmax(unit_logits, dim=-1), None
        return unit_logits, None, self._select(unit_logits, state, dense_rows, controls)

    def _update(
        self,
        u_s: Tensor,
        eps_s: Tensor,
        state: CellGraphState,
        mode: StepMode,
        alpha: Tensor | None,
        selection: Selection | None,
        unit_logits: Tensor,
        profiler: Profiler | None,
        dense_rows: Tensor | None,
        capture: bool = False,
        dense_counters: tuple[Tensor, Tensor] | None = None,
        *,
        controls: _Controls | None = None,
        static: bool = False,
        consts: WindowConsts | None = None,
        dense_indices: Tensor | None = None,
    ) -> _RoundResult:
        """The round of one position on ``state``: the dense reference, the masked round (mode
        ``masked`` or the soft warm-up) or the sparse round, shared by :meth:`step` and
        :meth:`forward` so that the two cannot drift apart. With ``capture`` the result also carries
        the gate of the written cells and the written mask (:meth:`forward`'s ``capture``).

        ``dense_counters`` (P1.10b, dense mode without disabled cells only) are the ``age`` and
        ``pos`` the written state carries, computed by :meth:`forward` for the whole window; the
        dense round writes them instead of zeroing the ages and ticking (the same values).

        P2.8: ``controls`` are the window's (:meth:`_controls` unless given); ``static`` runs the
        capturable sparse round and ``consts`` hands the window's edge weights and ``W1`` layout to
        the cell (:meth:`PDGCell.step_sparse`, :class:`~hsn.model.cell.WindowConsts`; a compiled
        window only)."""
        gate = (self._controls() if controls is None else controls).gate
        if mode == "dense":
            assert alpha is not None
            h_new, info = self._cell_dense(u_s, eps_s, state, alpha, consts)
            if dense_counters is None:
                new_state = self._write_dense(state, h_new)
            else:
                age, pos = dense_counters
                new_state = CellGraphState._make(h=h_new, q=h_new.mean(dim=1), age=age, pos=pos)
            if not capture:
                return _RoundResult(new_state)
            enabled = self._enabled_cells(state)
            return _RoundResult(new_state, _gate_of_written(info["b"], enabled), enabled)
        assert selection is not None
        if mode == "masked" or gate == "soft":
            # the soft warm-up executes every cell with a fractional weight: masked path
            return self._masked_round(u_s, eps_s, state, selection, capture, consts=consts)
        return self._sparse_round(
            u_s,
            eps_s,
            state,
            selection,
            unit_logits,
            profiler,
            dense_rows,
            capture,
            static=static,
            consts=consts,
            dense_indices=dense_indices,
        )

    def _cell_dense(
        self,
        u_s: Tensor,
        eps_s: Tensor,
        state: CellGraphState,
        alpha: Tensor,
        consts: WindowConsts | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """:meth:`PDGCell.step_dense`, called exactly as before P2.8 unless a compiled window
        hands it its ``consts`` (experiments and tests wrap the cell's methods with the P1.5
        signature)."""
        if consts is None:
            return self.cell.step_dense(u_s, eps_s, state, alpha)
        return self.cell.step_dense(u_s, eps_s, state, alpha, consts=consts)

    def _enabled_cells(self, state: CellGraphState) -> Tensor:
        """bool ``[B, N]``: every cell the dense round writes (all but ``model.disabled_cells``)."""
        B, N = state.batch_size, state.n_cells
        if not self.cell.disabled_cells:
            return torch.ones(B, N, dtype=torch.bool, device=state.device)
        return (~self.cell.disabled)[None, :].expand(B, N)

    def _sparse_round(
        self,
        u_s: Tensor,
        eps_s: Tensor,
        state: CellGraphState,
        selection: Selection,
        unit_logits: Tensor,
        profiler: Profiler | None,
        dense_rows: Tensor | None,
        capture: bool = False,
        *,
        static: bool = False,
        consts: WindowConsts | None = None,
        dense_indices: Tensor | None = None,
    ) -> _RoundResult:
        """The sparse round (P2.5): :meth:`PDGCell.step_sparse` on ``selection`` and the tick of
        the unselected cells. The rows of ``dense_rows`` (P2.2's dense fallback) run the P1.5 dense
        step instead — the same operations as ``mode="dense"`` — and replace the sparse result row
        by row; the selection they did not execute is handed to the cell with its fallback flags
        cleared. (The executed cost is the caller's: the detached batch mean of
        ``selection.cost``.)"""
        executed = selection
        if dense_rows is not None:
            executed = replace(selection, dense_fallback=torch.zeros_like(selection.dense_fallback))
        if static or consts is not None:  # P2.8's capturable round
            _, info = self.cell.step_sparse(
                u_s, eps_s, state, executed, profiler=profiler, static=static, consts=consts
            )
        else:  # the eager call, unchanged (experiments wrap these methods)
            _, info = self.cell.step_sparse(u_s, eps_s, state, executed, profiler=profiler)
        written = info["state"]
        selected = info["selected"]
        assert isinstance(written, CellGraphState) and isinstance(selected, Tensor)
        new_state = written.tick(selected)
        gate: Tensor | None = None
        if capture:
            b, valid = info["b"], info["valid"]
            assert isinstance(b, Tensor) and isinstance(valid, Tensor)
            gate = _gate_of_slots(b, executed.cell_idx, valid, state.n_cells)
        if dense_rows is not None and dense_indices is not None:
            # The row IDs were compacted once before the window. Compiled windows pad to a
            # power-of-two bucket with distinct non-fallback rows; only real fallback rows write.
            rows = dense_indices
            full = rows.numel() == state.h.shape[0]
            # The full bucket already has the original static batch shape; avoid gathering and
            # scattering a permutation of all rows (including a fully dense warm-up).
            subset = (
                state
                if full
                else CellGraphState._make(
                    h=state.h.index_select(0, rows),
                    q=state.q.index_select(0, rows),
                    age=state.age.index_select(0, rows),
                    pos=state.pos.index_select(0, rows),
                )
            )
            h_dense, dense_info = self._cell_dense(
                u_s if full else u_s.index_select(0, rows),
                eps_s if full else eps_s.index_select(0, rows),
                subset,
                torch.softmax(unit_logits if full else unit_logits.index_select(0, rows), dim=-1),
                consts,
            )
            dense_state = self._write_dense(subset, h_dense)
            valid_rows = dense_rows if full else dense_rows.index_select(0, rows)

            def merged(original: Tensor, dense: Tensor) -> Tensor:
                mask = valid_rows.reshape((-1,) + (1,) * (original.ndim - 1))
                if full:
                    return torch.where(mask, dense, original)
                values = torch.where(mask, dense, original.index_select(0, rows))
                return original.index_copy(0, rows, values)

            new_state = CellGraphState._make(
                h=merged(new_state.h, dense_state.h),
                q=merged(new_state.q, dense_state.q),
                age=merged(new_state.age, dense_state.age),
                pos=new_state.pos,  # both paths advanced it once
            )
            if capture:
                assert gate is not None
                enabled = self._enabled_cells(subset)
                dense_gate = _gate_of_written(dense_info["b"], enabled)
                gate = merged(gate, dense_gate)
                selected = merged(selected, enabled)
        if not capture:
            return _RoundResult(new_state)
        return _RoundResult(new_state, gate, selected)

    def _masked_round(
        self,
        u_s: Tensor,
        eps_s: Tensor,
        state: CellGraphState,
        selection: Selection,
        capture: bool = False,
        *,
        consts: WindowConsts | None = None,
    ) -> _RoundResult:
        """The P2.2 routed round on the dense path (module docstring): every cell computes its
        candidate with the selection's unit weights, ``h_new = m·h_cand + (1 − m)·h_prev`` (exactly
        ``h_cand`` where ``m = 1`` and ``h_prev`` where ``m = 0``), ``q`` the exact mean, the ages
        reset on the cells whose weight is non-zero (the hard decision and every cell of a
        dense-fallback row in the hard modes; every cell in the soft warm-up)."""
        alpha = selection.unit_weights.to(state.dtype)
        h_cand, info = self._cell_dense(u_s, eps_s, state, alpha, consts)
        mask = selection.cell_mask.to(state.dtype)[:, :, None]
        # P1.10c: m·h_cand + (1 − m)·h_prev in one op; exactly h_cand / h_prev at m = 1 / 0
        h_new = torch.lerp(state.h, h_cand, mask)
        if self.cell.disabled_cells:  # a soft m would move a disabled cell by an ulp
            h_new = torch.where(self.cell.disabled[None, :, None], state.h, h_new)
        updated = selection.cell_mask.detach() > 0
        if self.cell.disabled_cells:
            updated = updated & ~self.cell.disabled[None, :]
        written = state.with_dense(h_new)  # q: the exact mean (every age zeroed, restored below)
        new_state = CellGraphState._make(
            h=written.h, q=written.q, age=state.age.masked_fill(updated, 0), pos=written.pos
        ).tick(updated)
        if not capture:
            return _RoundResult(new_state)
        return _RoundResult(new_state, _gate_of_written(info["b"], updated), updated)

    def _prediction_target(self, emb: Tensor) -> Tensor:
        """``z_target [B, d]`` for the prediction loss, detached in both modes: the byte
        embedding ``emb`` itself (``fixed_embedding``) or the EMA copy of ``E`` applied to it
        (``ema_encoder``; the copy runs under ``no_grad``)."""
        if self.ema_target is None:
            return emb.detach()
        target: Tensor = self.ema_target(emb.detach())
        return target

    def update_target(self) -> None:
        """Move the EMA target encoder toward the online ``E`` (``train.ema_tau``); call it after
        each optimiser step. A no-op with the fixed-embedding target."""
        if self.ema_target is not None:
            self.ema_target.update()

    def _write_dense(self, state: CellGraphState, h_new: Tensor) -> CellGraphState:
        """Write the dense round into the state: every enabled cell replaced (``q`` the exact mean,
        which stays exact because disabled cells keep their old ``h``), updated cells at age 0,
        disabled cells keep counting, ``pos += 1``."""
        updated = state.with_dense(h_new)
        if not self.cell.disabled_cells:
            return updated.tick(torch.ones_like(state.age, dtype=torch.bool))
        enabled = (~self.cell.disabled)[None, :].expand_as(state.age)
        return CellGraphState._make(
            h=updated.h, q=updated.q, age=state.age.masked_fill(enabled, 0), pos=updated.pos
        ).tick(enabled)

    # -- a window of positions --------------------------------------------------------------------

    def forward(
        self,
        tokens: Tensor,
        ctx: Tensor,
        state: CellGraphState,
        reset: Tensor,
        *,
        capture: bool = False,
        reset_columns: Sequence[int] | None = None,
        prev_tokens: Tensor | None = None,
    ) -> tuple[Tensor, CellGraphState, dict[str, Tensor]]:
        """Run positions ``0 .. T−1`` of ``tokens [B, T]`` / ``ctx [B, T]`` from ``state``.

        ``reset [B, T]`` (bool) zeroes a session's state *before* position ``t`` is processed (a
        document boundary at ``t``). Returns ``logits [B, T, V]``, the final state and ``info``:
        the per-position stacks ``z``, ``z_hat``, ``z_target`` (detached), ``eps``, ``u``
        (``[B, T, d]``, autograd intact except for the target) and the per-position statistics
        ``state_abs_max``, ``eps_abs_mean``, ``alpha_entropy``, ``cost`` (the model dtype) and
        ``n_rounds`` (int64), each ``[T]``. With a sparse ``model.routing`` (every position runs
        the routed round of :meth:`step` with ``mode=None``) the window's selections are stacked
        too: detached ``cell_idx`` and ``unit_idx`` (int64 ``[B, T, K]``), ``valid`` (bool
        ``[B, T, K]``) and ``cost_per_row`` (fp32 ``[B, T]``), the router's decisions as charged
        (the one case where a valid slot is not written is a selected cell of the
        ``model.disabled_cells`` test hook; a dense-fallback row ran the dense step instead of its
        slots), plus P2.2's ``expected_cost [T]`` (the differentiable batch-mean expected cost,
        fp32), ``route_log_prob [B]`` (the REINFORCE log-probability summed over the positions;
        zeros unless the gate is ``reinforce``), ``explored [T]`` (fraction of rows with an explored
        slot), ``refreshed [T]`` (mean refreshed slots per row) and ``dense_fallback [T]`` (fraction
        of rows that ran dense), and P2.4's ``unit_logits [B, T, N, M]`` (the scorer's logits of
        every position in the state dtype, autograd intact: the balance term's input). In training
        the dense fallback is drawn here, once per row per window, with probability
        ``train.p_dense``.

        ``capture=True`` (off by default, and then free) adds the per-position cell states and
        gates, detached, for diagnostics:

        * ``cell_states [B, T, N, d]`` (state dtype): ``h`` after position ``t``;
        * ``cell_gates [B, T, N, d]`` (state dtype): the gate ``b`` with which each *written* cell
          was updated at ``t``, NaN at every cell that position did not write;
        * ``cell_written`` (bool ``[B, T, N]``): the cells position ``t`` wrote — in the dense
          round every enabled cell (all but ``model.disabled_cells``); in the masked round (mode
          ``masked`` or the soft warm-up) the cells with ``cell_mask > 0`` minus the disabled
          ones; in the sparse round the selected valid slots (``info["selected"]`` of
          :meth:`PDGCell.step_sparse`); in a dense-fallback row every enabled cell.

        The masked and the sparse round of the same selection agree on the written cells.

        ``reset_columns`` (P1.10b) are the positions at which the per-position reset runs, a
        strictly increasing sequence of ints in ``[0, T)`` that must contain every column where
        ``reset`` has a True (extra columns are allowed and change nothing). ``None`` derives them
        from ``reset``: free on the CPU, one device-to-host copy per window on an accelerator. A
        trainer passes them from the CPU batch it uploads, so no copy is needed. **Contract:** the
        list must cover every reset column. It is checked only when ``reset`` is on the CPU (a list
        that omits one raises ``ValueError``); on an accelerator it cannot be checked without the
        copy it saves, and an omitted column would silently skip that reset and carry the state
        across a document boundary. Derive it from the CPU copy of the same mask, as
        :class:`~hsn.train.tbptt.TBPTTTrainer` does, or pass ``None``.

        P1.10a: the work that depends on the inputs and parameters only runs once for the window —
        the byte and context embeddings, ``E`` and the prediction target over ``[B, T]``, the
        position-independent scorer features (:meth:`Scorer.window_scalars`), the edge weights
        (:meth:`PDGCell.window`), and after the loop the decoder over the stacked ``[q_t ; u_t]``
        and the statistics from the stacks. The loop runs, per position, the reset, ``ẑ_t =
        P(q_{t−1})``, ``ε_t``, ``u_t = S([z_t ; q_{t−1} ; ctx_t])``, the scorer and the round
        (:meth:`step`'s, shared code). ``E`` now runs before the loop in wall-clock order; global
        rule 2 is an information rule and still holds: ``ẑ_t`` reads ``q_{t−1}`` only (the P1.8
        leakage test). The batched matmuls round differently from :meth:`step`'s per-position ones
        (the GEMM row count changes), so ``forward`` matches stepping to rounding (fp64 1e-12,
        ``test_p1_10_forward_matches_stepwise_reference``), not bit for bit.

        P1.10b: the reset runs only at the reset columns, and the dense round (``model.routing``
        dense, no ``model.disabled_cells``) keeps no per-position ``age`` / ``pos`` bookkeeping: it
        writes every cell, so the age before position ``t`` is the start age (after a reset at 0)
        at ``t = 0`` and 0 afterwards, and ``pos`` follows from the reset mask. The states of every
        position still carry those values (views of the window's counters), and the reset there
        masks ``h`` and ``q`` only (``age`` and ``pos`` already account for it). Bit for bit the
        same as the per-position bookkeeping.
        """
        B, T = self._check_window_inputs(tokens, ctx, state, reset)
        if not isinstance(capture, bool):
            raise TypeError(f"capture must be a bool, got {capture!r}")
        resets_at = [False] * T
        for column in self._reset_columns(reset, reset_columns):
            resets_at[column] = True
        mode = self._window_mode()
        dense_rows = self._draw_dense_rows(B, state)
        dense_indices = self._dense_indices(dense_rows)
        controls = self._controls()
        pro = self._window_prologue(
            tokens, ctx, state, reset, mode, reset_first=resets_at[0], prev_tokens=prev_tokens
        )
        z_hats: list[Tensor] = []
        epss: list[Tensor] = []
        us: list[Tensor] = []
        qs: list[Tensor] = []
        hs: list[Tensor] = []
        state_maxima: list[Tensor] = []
        unit_logits_steps: list[Tensor] = []
        gates: list[Tensor] = []
        written: list[Tensor] = []
        records: list[dict[str, Tensor]] = []
        readouts: list[Tensor] = []
        # P1.10e: the bound method of the loop, looked up once per call (never at construction:
        # experiments replace modules, and functional_call swaps parameters)
        position = self._position
        with self.cell.window():
            for t in range(T):
                result = position(
                    state,
                    reset[:, t] if resets_at[t] else None,
                    pro.z_steps[t],
                    pro.ctx_steps[t],
                    pro.scalar_steps[t],
                    mode,
                    controls,
                    dense_rows,
                    capture,
                    pro.dense_counters(t),
                    dense_indices=dense_indices,
                )
                state = result.state
                if self.cfg.selected_readout:
                    readouts.append(
                        self._selected_readout(state, result.selection, controls.gate, mode)
                    )
                z_hats.append(result.z_hat)
                epss.append(result.eps)
                us.append(result.u)
                qs.append(state.q)
                state_maxima.append(state.h.detach().abs().amax())
                if capture:
                    hs.append(state.h.detach())
                unit_logits_steps.append(result.unit_logits.detach())
                if result.selection is not None:
                    records.append(self._selection_record(result.selection))
                if capture:
                    assert result.gate is not None and result.written is not None
                    gates.append(result.gate)
                    written.append(result.written)
        if records and len(records) != T:
            raise RuntimeError(
                f"{len(records)} of {T} positions ran a routed round; every position of a "
                "window runs the same mode (the dense fallback is a flag on the selection)"
            )
        routed = (
            {key: _stack_record([r[key] for r in records]) for key in SELECTION_RECORD_KEYS}
            if records
            else None
        )
        return self._window_epilogue(
            pro,
            state,
            z_hats=torch.stack(z_hats, dim=1),
            epss=torch.stack(epss, dim=1),
            us=torch.stack(us, dim=1),
            qs=torch.stack(qs, dim=1),
            hs=torch.stack(hs, dim=1) if capture else None,
            state_abs_max=torch.stack(state_maxima),
            unit_logits=torch.stack(unit_logits_steps, dim=1),
            routed=routed,
            gates=torch.stack(gates, dim=1) if capture else None,
            written=torch.stack(written, dim=1) if capture else None,
            readouts=torch.stack(readouts, dim=1) if readouts else None,
        )

    # -- the pieces of a window (P2.8: shared by forward and the compiled window) ----------------

    def _window_mode(self) -> StepMode:
        """The mode every position of a window runs: ``dense`` or the routed round, which is the
        sparse round, or in training with ``train.ste_scope=all`` the masked round (the same
        hard selection and the same written values; every cell and unit computed, so that the
        straight-through gradient reaches the ones that were not selected)."""
        if self.cfg.routing == "dense":
            return "dense"
        return "masked" if self.training and self.ste_scope == "all" else "sparse"

    def _draw_dense_rows(self, B: int, state: CellGraphState) -> Tensor | None:
        """P2.2's dense fallback, drawn once per row per window in training with probability
        ``train.p_dense`` from the CPU RNG (the same rows on every backend, replayed by a resume);
        ``None`` when no row is drawn or the routing is dense."""
        if self.cfg.routing == "dense" or not self.training:
            return None
        p_dense = self.schedule.p_dense(self.train_step)
        if p_dense <= 0.0:
            return None
        drawn = torch.rand(B) < p_dense
        if not bool(drawn.any()):  # a host-side check: no device synchronisation
            return None
        return drawn.to(state.device)

    @staticmethod
    def _dense_indices(dense_rows: Tensor | None, *, bucket: bool = False) -> Tensor | None:
        """Compact fallback row IDs once per window, before graph capture.

        Compiled execution uses power-of-two buckets, padded with distinct non-fallback rows.
        No duplicate index writes or data-dependent output shapes occur inside compiled blocks.
        A caller-supplied device mask costs one host read per window, never per position.
        """
        if dense_rows is None:
            return None
        cpu = dense_rows.detach().to("cpu")
        count = int(cpu.sum())
        if count == 0:
            return None
        size = min(cpu.numel(), 1 << (count - 1).bit_length()) if bucket else count
        order = torch.argsort(cpu.to(torch.int8), descending=True, stable=True)
        return order[:size].to(dense_rows.device)

    def _window_prologue(
        self,
        tokens: Tensor,
        ctx: Tensor,
        state: CellGraphState,
        reset: Tensor,
        mode: StepMode,
        *,
        reset_first: bool,
        prev_tokens: Tensor | None = None,
    ) -> _Prologue:
        """The window-invariant work of :meth:`forward` (P1.10a): the embeddings, ``E`` and the
        prediction target over ``[B, T]``, the context embeddings, the position-independent scorer
        features, and (P1.10b, dense routing without disabled cells) the window's ``age`` / ``pos``
        counters, with the age feature in front of the scalars. ``reset_first`` says whether column
        0 resets (it decides the start age)."""
        cfg = self.cfg
        B, T = (int(n) for n in tokens.shape)
        emb = self.byte_emb(tokens)  # [B, T, d]
        z = self.encoder(emb)
        z_target = self._prediction_target(emb)  # detached; a function of x_t only
        context = self.ctx_emb(ctx)  # [B, T, context_dim]
        positions = _window_positions(state.pos, reset)  # [B, T]: pos before each step
        previous_window: Tensor | None = None
        if cfg.key_address:
            previous = self._previous_tokens(tokens[:, 0], prev_tokens)
            shifted = torch.cat([previous[:, None], tokens[:, :-1]], dim=1)
            previous_window = shifted.masked_fill(reset, BOS)
        older_window: dict[str, Any] = {}
        if cfg.key_address and cfg.address_ngram > 1:
            # A9: x_{t-2} .. x_{t-n} from the window's own bytes; BOS where the document has
            # none, and before column 0 beyond the one carried byte (module docstring)
            assert previous_window is not None
            lag, lags = previous_window, []
            for back in range(2, cfg.address_ngram + 1):
                lag = torch.cat([torch.full_like(lag[:, :1], BOS), lag[:, :-1]], dim=1)
                lag = lag.masked_fill(positions < back, BOS)
                lags.append(lag)
            older_window = {"older": torch.stack(lags, dim=-1)}
        scalars = self.scorer.window_scalars(
            ctx,
            positions,
            budget_frac=cfg.budget,
            round_idx=0,
            dtype=state.dtype,
            tokens=tokens if cfg.key_address else None,
            prev_tokens=previous_window,
            **older_window,
        )  # [B, T, N, 7] (9 with A5; 7 + 2n with A9)
        # P1.10b: the dense round writes every cell, so its counters are known for the window
        dense_window = mode == "dense" and not self.cell.disabled_cells
        counters: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
        if dense_window:
            start_age = state.age.masked_fill(reset[:, 0, None], 0) if reset_first else state.age
            age_feature = torch.zeros(B, T, cfg.n_cells, dtype=state.dtype, device=state.device)
            age_feature[:, 0] = start_age.to(state.dtype) / self.scorer.age_max
            scalars = torch.cat([age_feature[..., None], scalars], dim=-1)  # [B, T, N, 8]
            counters = (start_age, torch.zeros_like(state.age), positions, positions + 1)
        # unbind: one backward node for the whole window instead of a slice backward per position
        return _Prologue(
            z=z,
            z_target=z_target,
            context=context,
            scalars=scalars,
            z_steps=z.unbind(1),
            ctx_steps=context.unbind(1),
            scalar_steps=scalars.unbind(1),
            counters=counters,
        )

    def _position(
        self,
        state: CellGraphState,
        reset_t: Tensor | None,
        z_t: Tensor,
        ctx_t: Tensor,
        scalars_t: Tensor,
        mode: StepMode,
        controls: _Controls,
        dense_rows: Tensor | None,
        capture: bool,
        dense_counters: tuple[Tensor, Tensor, Tensor, Tensor] | None,
        *,
        static: bool = False,
        consts: WindowConsts | None = None,
        dense_indices: Tensor | None = None,
    ) -> _PositionResult:
        """One position of a window (the body of :meth:`forward`'s loop): the reset (``reset_t``,
        bool ``[B]``, or ``None`` for a column without one), ``ẑ_t = P(q_{t−1})``, ``ε_t``,
        ``u_t = S([z_t ; q_{t−1} ; ctx_t])``, the scorer and the round (:meth:`step`'s, shared
        code). ``dense_counters`` (the P1.10b dense window) are the ``age`` / ``pos`` before and
        after the position. ``static`` / ``consts``: P2.8's capturable round (:meth:`_update`)."""
        if dense_counters is not None:
            h, q = state.h, state.q
            if reset_t is not None:  # age and pos already account for the reset
                h = h.masked_fill(reset_t[:, None, None], 0)
                q = q.masked_fill(reset_t[:, None], 0)
            state = CellGraphState._make(h=h, q=q, age=dense_counters[0], pos=dense_counters[1])
        elif reset_t is not None:
            state = state.reset(reset_t)
        state = self._clip_state_grad(state)  # A6: what this position passes back is limited
        q_prev = state.q
        z_hat = self.predictor(q_prev)  # from the carried summary only (global rule 2)
        eps = z_t - z_hat
        u = self.shared(self._shared_input(z_t, q_prev, ctx_t, state.h))
        # P1.7: see step. A6: the reads of the cell and the scorer (the same values)
        u_s, eps_s = u.to(state.dtype), self._eps_read(z_t, z_hat, eps).to(state.dtype)
        age = None if dense_counters is not None else state.age  # dense: a column of the scalars
        if self._cut_router:
            feats = self.scorer._assemble(
                u_s.detach(), eps_s.detach(), state.h.detach(), age, scalars_t
            )
        else:
            feats = self.scorer._assemble(u_s, eps_s, state.h, age, scalars_t)
        unit_logits, alpha, selection = self._score(
            feats, state, mode, dense_rows, controls=controls
        )
        result = self._update(
            u_s,
            eps_s,
            state,
            mode,
            alpha,
            selection,
            unit_logits,
            None,
            dense_rows,
            capture,
            dense_counters[2:] if dense_counters is not None else None,
            controls=controls,
            static=static,
            consts=consts,
            dense_indices=dense_indices,
        )
        return _PositionResult(
            result.state, z_hat, eps, u, unit_logits, selection, result.gate, result.written
        )

    @staticmethod
    def _selection_record(sel: Selection) -> dict[str, Tensor]:
        """The per-position values of a routed window's ``info`` (:data:`SELECTION_RECORD_KEYS`):
        detached indices and slot masks, the logits (autograd intact; P2.4's balance input), the
        per-row cost (detached), the batch-mean expected cost, the REINFORCE log-probability, and
        the exploration, refresh and dense-fallback rates of the position."""
        return {
            "cell_idx": sel.cell_idx,
            "unit_idx": sel.unit_idx,
            "valid": sel.valid,
            "unit_logits": sel.logits,
            "cost_per_row": sel.cost.detach(),
            "expected_cost": sel.expected_cost.mean(),
            "route_log_prob": sel.log_prob,
            "explored": sel.explored.any(dim=1).float().mean(),
            "refreshed": sel.refreshed.float().sum(dim=1).mean(),
            "dense_fallback": sel.dense_fallback.float().mean(),
        }

    def _window_epilogue(
        self,
        pro: _Prologue,
        state: CellGraphState,
        *,
        z_hats: Tensor,
        epss: Tensor,
        us: Tensor,
        qs: Tensor,
        hs: Tensor | None,
        state_abs_max: Tensor,
        unit_logits: Tensor,
        routed: dict[str, Tensor] | None,
        gates: Tensor | None,
        written: Tensor | None,
        readouts: Tensor | None = None,
    ) -> tuple[Tensor, CellGraphState, dict[str, Tensor]]:
        """The work after the loop (P1.10a): the decoder over the stacked ``[q_t ; u_t]``, the
        statistics from the stacks and ``info`` (:meth:`forward`). Every argument is stacked over
        the window's positions (``[B, T, …]``; ``routed`` the stacked
        :meth:`_selection_record`). ``state_abs_max`` is already reduced to ``[T]``;
        ``hs`` is materialized only for an explicit diagnostic capture."""
        T = int(z_hats.shape[1])
        if pro.counters is not None:
            positions_after = pro.counters[3]
            state = CellGraphState._make(
                h=state.h, q=state.q, age=state.age, pos=positions_after[:, -1].contiguous()
            )
        logits = self._decode(qs, us, readouts)
        info: dict[str, Tensor] = {
            "z": pro.z,
            "z_hat": z_hats,
            "z_target": pro.z_target,
            "eps": epss,
            "u": us,
        }
        with torch.no_grad():
            probs = torch.softmax(unit_logits, dim=-1)
            log_probs = F.log_softmax(unit_logits, dim=-1)
            info["state_abs_max"] = state_abs_max
            info["eps_abs_mean"] = epss.abs().mean(dim=(0, 2))
            info["alpha_entropy"] = -(probs * log_probs).sum(dim=-1).mean(dim=(0, 2))
        if routed is not None:
            cost_per_row = routed["cost_per_row"]
            info["cost"] = cost_per_row.mean(dim=0).to(state.dtype).to(self.dtype)
        else:
            info["cost"] = torch.ones(T, dtype=self.dtype, device=state.device)
        info["n_rounds"] = torch.ones(T, dtype=torch.int64, device=state.device)
        if routed is not None:
            info["cell_idx"] = routed["cell_idx"]
            info["unit_idx"] = routed["unit_idx"]
            info["valid"] = routed["valid"]
            info["unit_logits"] = routed["unit_logits"]  # P2.4
            info["cost_per_row"] = routed["cost_per_row"]
            info["expected_cost"] = routed["expected_cost"]
            info["route_log_prob"] = routed["route_log_prob"].sum(dim=1)
            info["explored"] = routed["explored"]
            info["refreshed"] = routed["refreshed"]
            info["dense_fallback"] = routed["dense_fallback"]
        if gates is not None and written is not None:
            assert hs is not None
            info["cell_states"] = hs
            info["cell_gates"] = gates
            info["cell_written"] = written
        return logits, state, info

    def _previous_tokens(self, tokens: Tensor, previous: Tensor | None) -> Tensor:
        """A5's external previous byte; it never changes CellGraphState's schema."""
        if previous is None:
            return torch.full_like(tokens, BOS)
        self._check_ids("prev_tokens", previous, tuple(tokens.shape), tokens.device)
        return previous

    def _older_bytes(self, older: Tensor | None, pos: Tensor, tokens: Tensor) -> dict[str, Tensor]:
        """A9's ``older`` keyword of one step: ``x_{t−2} .. x_{t−n}`` (``[B, n − 1]``), BOS where
        the document has no such byte (``pos < back``) and everywhere when the caller gave only
        one previous byte. ``{}`` without ``model.address_ngram > 1``."""
        n = self.cfg.address_ngram
        if n <= 1:
            return {}
        if older is None:
            older = torch.full(
                (*tokens.shape, n - 1), BOS, dtype=tokens.dtype, device=tokens.device
            )
        back = torch.arange(2, n + 1, device=pos.device)
        return {"older": older.masked_fill(pos[:, None] < back[None, :], BOS)}

    @staticmethod
    def _reset_previous(previous: Tensor | None, reset: Tensor, tokens: Tensor) -> Tensor:
        if previous is None:
            return torch.full_like(tokens, BOS)
        return previous.masked_fill(reset, BOS)

    def _decode(self, q: Tensor, u: Tensor, readout: Tensor | None) -> Tensor:
        probability = self.cfg.decoder_shortcut_dropout
        if self.training and probability:
            if probability == 1.0:
                q, u = q * 0.0, u * 0.0
            else:
                # One joint mask per position; no mask on the selected-cell path.
                keep = (torch.rand_like(q[..., :1]) >= probability).to(q.dtype)
                scale = keep / (1.0 - probability)
                q, u = q * scale, u * scale
        inputs = [q, u]
        if self.cfg.selected_readout:
            if readout is None:
                raise ValueError("A5 selected_readout needs the per-position readouts")
            inputs.append(readout)
        logits: Tensor = self.decoder(torch.cat(inputs, dim=-1))
        return logits

    def _selected_readout(
        self,
        state: CellGraphState,
        selection: Selection | None,
        gate: GateMode | None = None,
        mode: StepMode | None = None,
    ) -> Tensor:
        """A5: mean updated selected cells, with the existing gate's gradient estimator.

        STE keeps the exact hard mean and routes its gradient through the selected mask. Soft
        warm-up uses the normalised soft mask over all cells; fallback rows read all cells.
        No sequence attention, additional persistent state, or computation-path weight writes.
        """
        # A8 (model.fast_readout): r averages the routed slots only and f follows it, [B, 2d]
        fast = self.cfg.fast_cells if self.cfg.fast_readout else 0
        if selection is None:
            return torch.cat([state.q, self._fast_read(state.h)], dim=-1) if fast else state.q
        if mode == "masked" or (gate == "soft" and self.cfg.routing in SCORED_MODES):
            weights = selection.cell_mask.to(state.dtype)
            if fast:  # the fast group has its own read
                weights = torch.cat([torch.zeros_like(weights[:, :fast]), weights[:, fast:]], dim=1)
            readout = (state.h * weights[..., None]).sum(1) / weights.sum(
                1, keepdim=True
            ).clamp_min(1e-6)
        else:
            # Padded indices are unspecified, just as in the sparse cell kernels.
            safe_idx = selection.cell_idx.masked_fill(~selection.valid, 0)
            indices = safe_idx[..., None].expand(-1, -1, state.width)
            chosen = state.h.gather(1, indices)
            valid = selection.valid
            if fast:  # the reserved slots are the first `fast` slots (Router.select)
                valid = torch.cat([torch.zeros_like(valid[:, :fast]), valid[:, fast:]], dim=1)
            weights = valid.to(state.dtype)
            if gate == "ste":
                mask = selection.cell_mask.gather(1, safe_idx).to(state.dtype)
                weights = weights * (1.0 + mask - mask.detach())
            readout = (chosen * weights[..., None]).sum(1)
            readout = readout / valid.sum(1, keepdim=True).clamp_min(1).to(state.dtype)
        readout = torch.where(selection.dense_fallback[:, None], state.q, readout)
        if fast:
            readout = torch.cat([readout, self._fast_read(state.h)], dim=-1)
        return readout

    # -- accounting -------------------------------------------------------------------------------

    def param_report(self) -> dict[str, int]:
        """Measured parameter counts: the Appendix C.1 rows in table order and their ``total``
        (the closed form of :func:`hsn.train.memory_plan.expected_param_count`), then ``edges``
        (the P1.4 edge logits, outside Appendix C) and ``total_trainable`` (every parameter of
        this module; the run manifest's ``param_count``)."""
        modules: dict[str, nn.Module] = {
            "byte_emb": self.byte_emb,
            "ctx_emb": self.ctx_emb,
            "encoder": self.encoder,
            "predictor": self.predictor,
            "shared": self.shared,
            "units": self.cell.units,
            "w_out": self.cell.W_out,
            "v_msg": self.cell.V,
            "gate": self.cell.gate,
            "cell_norm": self.cell.norm,
            "cand_norm": self.cell.cand_norm,
            "scorer": self.scorer,
            "halting": self.halting,
            "decoder": self.decoder,
        }
        assert tuple(modules) == PARAM_REPORT_ROWS
        report = {name: sum(p.numel() for p in m.parameters()) for name, m in modules.items()}
        report["total"] = sum(report.values())
        report["edges"] = self.cell.graph.edge_logit.numel()
        report["total_trainable"] = sum(p.numel() for p in self.parameters())
        if report["total_trainable"] != report["total"] + report["edges"]:
            raise RuntimeError(
                f"param_report: {report['total_trainable']} trainable parameters, but the Appendix "
                f"C rows ({report['total']}) plus the edge logits ({report['edges']}) give "
                f"{report['total'] + report['edges']}; a submodule carries unaccounted parameters"
            )
        return report

    def extra_repr(self) -> str:
        text = f"seed={self.seed}, unroll={self.unroll}, routing={self.cfg.routing!r}"
        if self.ema_target is not None:
            text += f", ema_tau={self.ema_tau}"
        if self.feedback_detach:
            text += f", feedback_detach={list(self.feedback_detach)}"
        if self.state_grad_clip is not None:
            text += f", state_grad_clip={self.state_grad_clip}"
        if self.state_grad_ratio is not None:
            text += f", state_grad_ratio={self.state_grad_ratio}"
        return text

    # -- validation -------------------------------------------------------------------------------

    def _check_state(self, state: CellGraphState) -> None:
        if not isinstance(state, CellGraphState):
            raise TypeError(f"state must be a CellGraphState, got {type(state).__name__}")
        cfg = self.cfg
        if state.n_cells != cfg.n_cells or state.width != cfg.d:
            raise ValueError(
                f"state has N = {state.n_cells}, d = {state.width}; this model expects "
                f"N = {cfg.n_cells}, d = {cfg.d}"
            )
        if state.dtype != self.dtype:
            raise ValueError(
                f"state has dtype {state.dtype} but the parameters {self.dtype}; build it with "
                "init_state"
            )
        if state.device != self.device:
            raise ValueError(
                f"state lives on {state.device} but the parameters on {self.device}; move the "
                "model with .to(device) or build the state with init_state"
            )

    @staticmethod
    def _check_dense_rows(
        dense_rows: Tensor | None, mode: str, state: CellGraphState
    ) -> Tensor | None:
        if dense_rows is None:
            return None
        if mode == "dense":
            raise ValueError("dense_rows is only used with a routed mode ('masked' or 'sparse')")
        if not isinstance(dense_rows, Tensor) or dense_rows.dtype != torch.bool:
            raise TypeError("dense_rows must be a bool torch.Tensor [B]")
        if dense_rows.shape != (state.batch_size,) or dense_rows.device != state.device:
            raise ValueError(
                f"dense_rows must be a bool [B] = {(state.batch_size,)} tensor on "
                f"{state.device}, got {tuple(dense_rows.shape)} on {dense_rows.device}"
            )
        return dense_rows

    @staticmethod
    def _check_ids(name: str, ids: Tensor, shape: tuple[int, ...], device: torch.device) -> None:
        if not isinstance(ids, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor of ids, got {type(ids).__name__}")
        if ids.dtype not in _INDEX_DTYPES:
            raise TypeError(
                f"{name} must be an int64 or int32 tensor of ids, got dtype {ids.dtype}"
            )
        if ids.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(ids.shape)}")
        if ids.device != device:
            raise ValueError(f"{name} must live on {device}, got {ids.device}")

    def _check_step_inputs(self, x_t: Tensor, ctx_t: Tensor, state: CellGraphState) -> None:
        self._check_state(state)
        B = state.batch_size
        self._check_ids("x_t", x_t, (B,), state.device)
        self._check_ids("ctx_t", ctx_t, (B,), state.device)

    @staticmethod
    def _reset_columns(reset: Tensor, columns: Sequence[int] | None) -> list[int]:
        """The columns at which :meth:`forward` resets (``reset_columns``, module docstring):
        derived from ``reset`` when ``None`` (one device-to-host copy on an accelerator),
        validated otherwise."""
        T = int(reset.shape[1])
        if columns is None:
            derived: list[int] = reset.any(dim=0).nonzero().flatten().tolist()
            return derived
        if isinstance(columns, str | bytes) or not isinstance(columns, Sequence):
            raise TypeError(
                f"reset_columns must be a sequence of ints or None, got {type(columns).__name__}"
            )
        listed = list(columns)
        for column in listed:
            if isinstance(column, bool) or not isinstance(column, int):
                raise TypeError(f"reset_columns must hold ints, got {column!r}")
        if any(b <= a for a, b in zip(listed, listed[1:], strict=False)):
            raise ValueError(f"reset_columns must be strictly increasing, got {listed}")
        if listed and (listed[0] < 0 or listed[-1] >= T):
            raise ValueError(f"reset_columns must lie in [0, T) = [0, {T}), got {listed}")
        if reset.device.type == "cpu":  # the superset check is free here
            needed = reset.any(dim=0)
            needed[listed] = False
            missing = needed.nonzero().flatten().tolist()
            if missing:
                raise ValueError(
                    f"reset has resets at columns {missing} that reset_columns omits; it must "
                    "contain every column where some row resets"
                )
        return listed

    def _check_window_inputs(
        self, tokens: Tensor, ctx: Tensor, state: CellGraphState, reset: Tensor
    ) -> tuple[int, int]:
        self._check_state(state)
        B = state.batch_size
        if not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.shape[0] != B:
            raise ValueError(
                f"tokens must be a [B, T] tensor with B = {B}, got "
                f"{tuple(tokens.shape) if isinstance(tokens, Tensor) else type(tokens).__name__}"
            )
        T = int(tokens.shape[1])
        if T < 1:
            raise ValueError("tokens must hold at least one position (T >= 1)")
        self._check_ids("tokens", tokens, (B, T), state.device)
        self._check_ids("ctx", ctx, (B, T), state.device)
        if not isinstance(reset, Tensor):
            raise TypeError(f"reset must be a bool torch.Tensor, got {type(reset).__name__}")
        if reset.dtype != torch.bool:
            raise ValueError(f"reset must have dtype torch.bool, got {reset.dtype}")
        if reset.shape != (B, T):
            raise ValueError(f"reset must have shape [B, T] = {(B, T)}, got {tuple(reset.shape)}")
        if reset.device != state.device:
            raise ValueError(f"reset must live on {state.device}, got {reset.device}")
        return B, T
