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

The dense reference uses the same linear scorer as the sparse model, with ``α = softmax(logits)``
over all units and every cell updated (Appendix H.5), so the measured parameter count is exactly
Appendix C.1: :meth:`HSN.param_report` lists the twelve C.1 rows plus amendment A1's ``cell_norm``
row (P1.5a) and their ``total`` (1,820,617 for ``configs/ref_a1.yaml``; 1,820,361 before A1). The
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
``gated_delta_update``, which P2.8 gates) and ``StepOutput.selection`` the executed
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

Still out of scope: one round only (the :class:`~hsn.model.halting.HaltingHead` is owned but not
called until P2.7), no custom kernels (P2.9–P2.11), no ``torch.compile`` (P2.8); ``n_rounds`` is 1.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import GateMode, ModelConfig, RunConfig
from hsn.model.cell import PDGCell, profiler_segment
from hsn.model.embeddings import ByteEmbedding, ContextEmbedding
from hsn.model.encoder import Decoder, EMATarget, Encoder, Predictor, SharedTransform
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
    "cell_norm",  # P1.5a (amendment A1): the RMSNorm gain over the 4d cell input
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


class HSN(nn.Module):
    """The predictive cell model: embeddings, ``E``, ``P``, ``S``, the :class:`PDGCell`, the
    :class:`Router` (the :class:`Scorer` plus P2.1 selection; ``model.cost_profile`` is read here
    and an unaffordable ``model.k_min`` / ``model.fixed_route`` raises at construction), the
    :class:`HaltingHead` and ``D``, built from ``cfg`` (``model.*``) plus ``seed`` (``run.seed``),
    ``unroll`` (``train.unroll``) and, when ``model.predictor_target = ema_encoder``, ``ema_tau``
    (``train.ema_tau``) for the :class:`~hsn.model.encoder.EMATarget` copy of ``E`` (``ema_target``;
    ``None`` otherwise). ``schedule`` is the P2.2 :class:`~hsn.train.schedules.SelectionSchedule`
    (``train.gate_mode`` and the ramps; the config defaults when ``None``: soft gates, τ = 1, no
    exploration, no dense fallback)."""

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        seed: int,
        unroll: int,
        ema_tau: float | None = None,
        schedule: SelectionSchedule | None = None,
    ) -> None:
        super().__init__()
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
        self.shared = SharedTransform(cfg)
        self.cell = PDGCell(cfg, seed=seed)
        # the scorer is built here, in the P1.5 position, so seeded initialisation is unchanged
        self.router = Router.from_config(cfg, unroll=unroll)
        self.halting = HaltingHead(cfg)
        self.decoder = Decoder(cfg)
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
        schedule=SelectionSchedule.from_config(run.train))``."""
        if not isinstance(run, RunConfig):
            raise TypeError(f"from_config needs a RunConfig, got {type(run).__name__}")
        return cls(
            run.model,
            seed=run.seed,
            unroll=run.train.unroll,
            ema_tau=run.train.ema_tau,
            schedule=SelectionSchedule.from_config(run.train),
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
        B = state.batch_size

        with profiler_segment(profiler, "shared_path", B):
            z_hat, z, z_target, eps, u = self._shared_path(x_t, ctx_t, state)
        # P1.7: under autocast E, P and S emit the autocast dtype (bf16 / fp16); the state path
        # (scorer features, the softmax over units, the cell update and q) runs in the state's
        # dtype, the fp32 master precision of §3.7. Without autocast these casts are no-ops.
        u_s, eps_s = u.to(state.dtype), eps.to(state.dtype)

        selection: Selection | None = None
        with self.cell.graph.step():
            with profiler_segment(profiler, "scorer", B):
                feats = self.scorer.features(
                    u_s, eps_s, state, ctx_t, budget_frac=cfg.budget, round_idx=0, ema=None
                )
                unit_logits = self.router(feats).to(state.dtype)  # [B, N, M]
                if mode == "dense":
                    alpha = torch.softmax(unit_logits, dim=-1)
                else:
                    selection = self._select(unit_logits, state, dense_rows)
            if mode == "dense":
                h_new, _ = self.cell.step_dense(u_s, eps_s, state, alpha)
                new_state = self._write_dense(state, h_new)
                cost = torch.ones((), dtype=state.dtype, device=state.device)
            else:
                assert selection is not None
                if mode == "masked" or self._gate() == "soft":
                    # the soft warm-up executes every cell with a fractional weight: masked path
                    new_state, cost = self._masked_round(u_s, eps_s, state, selection)
                else:
                    new_state, cost = self._sparse_round(
                        u_s, eps_s, state, selection, unit_logits, profiler, dense_rows
                    )

        with profiler_segment(profiler, "decode", B):
            logits = self.decoder(torch.cat([new_state.q, u], dim=-1))  # the updated q_t

        with torch.no_grad():
            if mode != "dense":
                alpha = torch.softmax(unit_logits, dim=-1)
            log_alpha = F.log_softmax(unit_logits, dim=-1)
            aux = {
                "state_abs_max": new_state.h.abs().amax(),
                "eps_abs_mean": eps.abs().mean(),
                "alpha_entropy": -(alpha * log_alpha).sum(dim=-1).mean(),
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
        u = self.shared(torch.cat([z, q_prev, self.ctx_emb(ctx_t)], dim=-1))
        return z_hat, z, z_target, eps, u

    def _gate(self) -> GateMode | None:
        """The gate of this step: the schedule's in training, ``None`` (hard, no gradient path) in
        evaluation."""
        return self.schedule.gate(self.train_step) if self.training else None

    def _select(
        self, unit_logits: Tensor, state: CellGraphState, dense_rows: Tensor | None
    ) -> Selection:
        """The selection of this model's ``model.routing`` for one round, with the P2.2 gate and
        controls of this step: the schedule's gate, temperature and exploration rate in training
        (the probes — exploration and refresh — post-process a hard scored decision, so not in the
        soft warm-up and never for the controls), the hard decision in evaluation."""
        cfg = self.cfg
        schedule, step, training = self.schedule, self.train_step, self.training
        scored = cfg.routing in SCORED_MODES
        gate = self._gate()
        tau = schedule.temperature(step) if training else 1.0
        probes = scored and gate != "soft"
        epsilon = schedule.epsilon(step) if training and probes else 0.0
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
        )

    def _sparse_round(
        self,
        u_s: Tensor,
        eps_s: Tensor,
        state: CellGraphState,
        selection: Selection,
        unit_logits: Tensor,
        profiler: Profiler | None,
        dense_rows: Tensor | None,
    ) -> tuple[CellGraphState, Tensor]:
        """The sparse round (P2.5): :meth:`PDGCell.step_sparse` on ``selection``, the tick of the
        unselected cells, and the executed cost (the detached batch mean of ``selection.cost`` in
        the state dtype). The rows of ``dense_rows`` (P2.2's dense fallback) run the P1.5 dense
        step instead — the same operations as ``mode="dense"`` — and replace the sparse result row
        by row; the selection they did not execute is handed to the cell with its fallback flags
        cleared."""
        executed = selection
        if dense_rows is not None:
            executed = replace(selection, dense_fallback=torch.zeros_like(selection.dense_fallback))
        _, info = self.cell.step_sparse(u_s, eps_s, state, executed, profiler=profiler)
        written = info["state"]
        selected = info["selected"]
        assert isinstance(written, CellGraphState) and isinstance(selected, Tensor)
        new_state = written.tick(selected)
        if dense_rows is not None:
            h_dense, _ = self.cell.step_dense(u_s, eps_s, state, torch.softmax(unit_logits, dim=-1))
            dense_state = self._write_dense(state, h_dense)
            rows = dense_rows
            new_state = CellGraphState(
                h=torch.where(rows[:, None, None], dense_state.h, new_state.h),
                q=torch.where(rows[:, None], dense_state.q, new_state.q),
                age=torch.where(rows[:, None], dense_state.age, new_state.age),
                pos=new_state.pos,  # both paths advanced it once
            )
        return new_state, selection.cost.detach().mean().to(state.dtype)

    def _masked_round(
        self, u_s: Tensor, eps_s: Tensor, state: CellGraphState, selection: Selection
    ) -> tuple[CellGraphState, Tensor]:
        """The P2.2 routed round on the dense path (module docstring): every cell computes its
        candidate with the selection's unit weights, ``h_new = m·h_cand + (1 − m)·h_prev`` (exactly
        ``h_cand`` where ``m = 1`` and ``h_prev`` where ``m = 0``), ``q`` the exact mean, the ages
        reset on the cells whose weight is non-zero (the hard decision and every cell of a
        dense-fallback row in the hard modes; every cell in the soft warm-up). Returns the state
        and the executed cost."""
        alpha = selection.unit_weights.to(state.dtype)
        h_cand, _ = self.cell.step_dense(u_s, eps_s, state, alpha)
        mask = selection.cell_mask.to(state.dtype)[:, :, None]
        h_new = mask * h_cand + (1 - mask) * state.h
        if self.cell.disabled_cells:  # a soft m would move a disabled cell by an ulp
            h_new = torch.where(self.cell.disabled[None, :, None], state.h, h_new)
        updated = selection.cell_mask.detach() > 0
        if self.cell.disabled_cells:
            updated = updated & ~self.cell.disabled[None, :]
        written = state.with_dense(h_new)  # q: the exact mean (every age zeroed, restored below)
        new_state = replace(written, age=state.age.masked_fill(updated, 0)).tick(updated)
        return new_state, selection.cost.detach().mean().to(state.dtype)

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
        return replace(updated, age=state.age.masked_fill(enabled, 0)).tick(enabled)

    # -- a window of positions --------------------------------------------------------------------

    def forward(
        self, tokens: Tensor, ctx: Tensor, state: CellGraphState, reset: Tensor
    ) -> tuple[Tensor, CellGraphState, dict[str, Tensor]]:
        """Run positions ``0 .. T−1`` of ``tokens [B, T]`` / ``ctx [B, T]`` from ``state``.

        ``reset [B, T]`` (bool) zeroes a session's state *before* position ``t`` is processed (a
        document boundary at ``t``). Returns ``logits [B, T, V]``, the final state and ``info``:
        the per-position stacks ``z``, ``z_hat``, ``z_target`` (detached), ``eps``, ``u``
        (``[B, T, d]``, autograd intact except for the target) and the per-position statistics
        ``state_abs_max``, ``eps_abs_mean``, ``alpha_entropy``, ``cost`` (the model dtype) and
        ``n_rounds`` (int64), each ``[T]``. With a sparse ``model.routing`` (every position runs
        :meth:`step` with ``mode=None``) the window's selections are stacked too: detached
        ``cell_idx`` and ``unit_idx`` (int64 ``[B, T, K]``), ``valid`` (bool ``[B, T, K]``) and
        ``cost_per_row`` (fp32 ``[B, T]``), the router's decisions as charged (the one case where a
        valid slot is not written is a selected cell of the ``model.disabled_cells`` test hook; a
        dense-fallback row ran the dense step instead of its slots), plus P2.2's ``expected_cost
        [T]`` (the differentiable batch-mean expected cost, fp32), ``route_log_prob [B]`` (the
        REINFORCE log-probability summed over the positions; zeros unless the gate is
        ``reinforce``), ``explored [T]`` (fraction of rows with an explored slot), ``refreshed [T]``
        (mean refreshed slots per row) and ``dense_fallback [T]`` (fraction of rows that ran
        dense), and P2.4's ``unit_logits [B, T, N, M]`` (the scorer's logits of every position in
        the state dtype, autograd intact: the balance term's input). In training the dense
        fallback is drawn here, once per row per window, with probability ``train.p_dense``.
        """
        B, T = self._check_window_inputs(tokens, ctx, state, reset)
        dense_rows: Tensor | None = None
        if self.cfg.routing != "dense" and self.training:
            p_dense = self.schedule.p_dense(self.train_step)
            if p_dense > 0.0:  # a CPU draw: the same rows on every backend, replayed by a resume
                drawn = torch.rand(B) < p_dense
                if bool(drawn.any()):  # a host-side check: no device synchronisation
                    dense_rows = drawn.to(state.device)
        logits: list[Tensor] = []
        stacks: dict[str, list[Tensor]] = {
            "z": [],
            "z_hat": [],
            "z_target": [],
            "eps": [],
            "u": [],
        }
        stats: dict[str, list[Tensor]] = {
            "state_abs_max": [],
            "eps_abs_mean": [],
            "alpha_entropy": [],
        }
        costs: list[Tensor] = []
        rounds: list[int] = []
        selections: list[Selection] = []
        with self.cell.graph.step():
            for t in range(T):
                state = state.reset(reset[:, t])
                out = self.step(tokens[:, t], ctx[:, t], state, dense_rows=dense_rows)
                state = out.state
                logits.append(out.logits)
                for name in stacks:
                    stacks[name].append(getattr(out, name))
                for name in stats:
                    stats[name].append(out.aux[name])
                costs.append(out.cost)
                rounds.append(out.n_rounds)
                if out.selection is not None:
                    selections.append(out.selection)
        info: dict[str, Tensor] = {
            name: torch.stack(values, dim=1) for name, values in stacks.items()
        }
        info.update({name: torch.stack(values) for name, values in stats.items()})
        info["cost"] = torch.stack(costs).to(self.dtype)
        info["n_rounds"] = torch.tensor(rounds, dtype=torch.int64, device=state.device)
        if selections:
            if len(selections) != T:
                raise RuntimeError(
                    f"{len(selections)} of {T} positions ran a routed round; every position of a "
                    "window runs the same mode (the dense fallback is a flag on the selection)"
                )
            info["cell_idx"] = torch.stack([sel.cell_idx for sel in selections], dim=1)
            info["unit_idx"] = torch.stack([sel.unit_idx for sel in selections], dim=1)
            info["valid"] = torch.stack([sel.valid for sel in selections], dim=1)
            info["unit_logits"] = torch.stack([sel.logits for sel in selections], dim=1)  # P2.4
            info["cost_per_row"] = torch.stack([sel.cost.detach() for sel in selections], dim=1)
            info["expected_cost"] = torch.stack([sel.expected_cost.mean() for sel in selections])
            info["route_log_prob"] = torch.stack([sel.log_prob for sel in selections], dim=1).sum(
                dim=1
            )
            info["explored"] = torch.stack(
                [sel.explored.any(dim=1).float().mean() for sel in selections]
            )
            info["refreshed"] = torch.stack(
                [sel.refreshed.float().sum(dim=1).mean() for sel in selections]
            )
            info["dense_fallback"] = torch.stack(
                [sel.dense_fallback.float().mean() for sel in selections]
            )
        return torch.stack(logits, dim=1), state, info

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
