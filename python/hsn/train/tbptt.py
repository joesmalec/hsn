"""HSN training with truncated backpropagation through time (P1.7).

:class:`TBPTTTrainer` is the P0.5 :class:`~hsn.train.loop.Trainer` specialised for the
:class:`~hsn.model.hsn.HSN` module. It keeps the generic parts unchanged (the resumable document
streams of :class:`~hsn.train.loop.StreamLoader`, the optimiser ``train.optimizer`` names with the
P0.5 parameter groups, the learning-rate schedule, the AMP policy of :mod:`hsn.train.amp`,
evaluation, the step :class:`~hsn.train.loop.History` and the checkpoint files) and owns the HSN
window loop::

    per window:  state = state.detach()                          (every ``train.q_resync_every``
                 q     = state.recompute_q()                      windows, at the window start)
                 logits, state, info = model(tokens, ctx, state, reset)
                 extras = auxiliary_terms(info, cfg, model.cell.graph)       (P2.4)
                 terms  = assemble_terms({"logits": logits, **info}, batch, cfg, extras)
                 terms  = _routing_terms(terms, info)                         (P2.2 REINFORCE)
                 total, parts = total_and_parts(terms)
                 grad_norm_report(model, terms) on the first window of a stats step
                 backward (scaled under fp16)
    per step:    unscale; grad_norm per group; clip to train.grad_clip; optimiser step (with the
                 units the step's windows executed, P2.6); scheduler step; model.update_target();
                 log parts and statistics

* **Truncated BPTT.** Every window starts from ``state.detach()``: the carried
  :class:`~hsn.model.state.CellGraphState` keeps its values and loses its autograd history, so a
  window's gradient never reaches an earlier one (plan §2.6: fixed unroll, carried detached state;
  one optimiser step is ``train.grad_accum`` windows, P0.5 semantics). Document resets are the
  loader's ``reset`` mask (True at every document start) and are applied by the model itself,
  position by position, before the position is processed (P1.5). The trainer passes the columns
  that hold a reset, read from the CPU window, as ``reset_columns`` (P1.10b), so the model resets
  only there without a device-to-host copy.
* **Periodic ``q`` resync.** Every ``train.q_resync_every`` windows the summary is replaced by the
  exact mean over cells (:meth:`~hsn.model.state.CellGraphState.recompute_q`) at the window start,
  which removes the floating-point drift incremental sparse updates accumulate (P1.3 pitfall). The
  dense step of P1.5 writes the exact mean, so until P2.5 the resync is a numerical no-op; the
  counters (``windows``, ``windows_since_resync``, ``resyncs``) are part of every checkpoint.
* **The objective** is :func:`hsn.losses.assemble_terms`: every §2.6 term, with ``task`` and
  ``pred`` the only active ones in Phase 1. P2.4 adds the auxiliary terms: the raw values of the
  terms named in ``train.aux_enabled`` come from :func:`hsn.losses.auxiliary_terms` (``cost`` from
  the window's expected cost over the router's dense cost, ``balance`` from the stacked unit logits
  and routed slots, ``graph`` from the cell graph) and enter the assembly as ``extras``; the
  one-at-a-time rule is the configuration's and the assembly's.
* **Statistics.** Every :class:`TBPTTStepRecord` carries the λ-weighted loss ``parts`` (mean over
  the step's windows), the model's own statistics from ``info`` (``state_abs_max`` over the step,
  ``eps_mean`` = mean |ε| and ``alpha_entropy``, means over windows and over the positions each
  window computes: a window cut by P1.10d excludes its all-padding tail) and the global gradient
  norm before clipping. Every ``train.stats_every`` steps (and at the last step) the record
  also carries ``grad_norm/<group>``, the pre-clip norm per parameter group, one group per top-level
  submodule of the model (``byte_emb``, ``ctx_emb``, ``encoder``, ``predictor``, ``shared``,
  ``cell``, ``router``, ``decoder``; a group without gradients, such as ``halting`` while
  ``model.r_max = 1``, is not reported), and — P2.4 — ``term_grad_norm/<term>``, the gradient norm
  of every objective term on its own (:func:`hsn.losses.grad_norm_report` on the step's first
  window, before its backward, scaled by the fp16 loss scale when there is one; a term without a
  gradient path reports 0), and one log line prints all of it.
* **Checkpoints** are the P0.5 files (model, optimiser, scheduler, step, RNG state of Python,
  NumPy, torch, CUDA and MPS, loader position, loss scaler, config hash) plus the carried cell state
  and the resync counters in ``extra`` (:mod:`hsn.train.checkpoint`); :meth:`resume` restores all
  of it, so a resumed run continues bit for bit on the CPU profile.
* After every optimiser step :meth:`~hsn.model.hsn.HSN.update_target` moves the EMA prediction
  target when ``model.predictor_target = ema_encoder`` (P1.6; a no-op otherwise).
* **Routing (P2.2).** Before every window the trainer hands the model its step counter
  (:meth:`~hsn.model.hsn.HSN.set_train_step`), which the
  :class:`~hsn.train.schedules.SelectionSchedule` turns into the gate mode, the temperature, the
  exploration rate and the dense-fallback probability of the window. With ``train.gate_mode =
  reinforce`` the objective of a window gains the policy-gradient term ``−mean_b log p_b · (R −
  b̄)`` with ``R = −L_task`` of the window — ``R = −(L_task + λ_c·L_cost)`` when the P2.4 cost
  term is on, since the REINFORCE cell mask is hard and the cost penalty can reach the sampled
  cells only through the reward — and ``b̄`` an EMA baseline
  (``train.reinforce_baseline_decay``; initialised at the first reward so the first window's
  advantage is 0; a window without loss positions or with a non-finite loss, which the fp16
  scaler skips, leaves both the term and the baseline alone), logged as ``parts["reinforce"]``
  and carried in the checkpoint. The model's schedule must be the configuration's. Every step
  record also carries ``route_cost`` (the mean executed cost per position as a fraction of the
  dense step; 1.0 in dense mode) and, in a routed run, the ``explored``, ``refreshed`` and
  ``dense_fallback`` rates.
* **Active units (P2.6).** In a routed run every step record carries ``active_units``, the sorted
  ids of the units that executed in the step's windows: the union of the window ``unit_idx`` over
  the valid slots (:func:`~hsn.train.lazy_adam.active_units_from_indices`), every unit when the
  gate of the step is ``soft`` (the masked round runs the whole bank) or a row ran the dense
  fallback, ORed over the ``train.grad_accum`` windows of the step; ``stats["active_units"]`` is
  their count. With ``train.optimizer = lazy_adamw`` (the default of sparse runs) the mask is handed
  to :meth:`~hsn.train.lazy_adam.LazyAdamW.step`, so a unit that did not execute receives no
  optimiser update; in dense routing the record holds ``None`` and every unit updates.

* **Counterfactual utility (P2.3)**, on when ``train.lambda_u > 0`` (a routed configuration;
  the config layer refuses the rest). Each window is a probe window with probability
  ``train.probe_fraction``, drawn (with the probe position and the candidates) from a generator
  seeded by ``(run.seed, window counter)``, so the global RNG stream is untouched and a resume
  replays every probe. Before the window's forward pass, under the ``probe`` profiler segment
  (charged to the window's tokens) and the training autocast,
  :func:`hsn.train.probes.probe_window` probes the window from its start state (training split
  only, derived from the documents' meta) unless the step is in the soft warm-up of a scored
  routing. The objective gains ``utility`` = ``λ_u · utility_loss`` of the probe's pairs (zero on
  other windows, so the key is always there when the term is on) and, after the window's
  backward, the scorer's EMA features take one step (``train.ema_decay``): the selection frequency
  of every cell over the window's non-padding positions and, on a probe window, the mean target of
  the measured cells. Step records then carry ``probe_windows``, ``probe_pairs`` and
  ``probe_positions`` (rollout positions per row, summed).

* **Trailing padding (P1.10d).** Under ``data.align_docs`` every document starts at a window
  boundary, so a window whose rows have all ended their documents is padding in every row from some
  column ``c`` on, and the next window resets every row at ``t = 0``. The trainer then computes only
  the columns ``[0, c)`` (:func:`trailing_cut`, from the CPU window: no device synchronisation).
  This is exact in math under dense routing (since P1.10a's window-level ``E`` / ``D`` the floats
  agree to rounding, fp64 1e-12 and fp32 1e-5, because the GEMMs see ``[B, c]`` instead of
  ``[B, T]``): the columns before ``c`` run the same operations on the same inputs, every
  dense-routing term (task over ``loss_mask``, pred and the variance floor over
  ``doc >= 0``, graph over the parameters) gets exact zeros from the padding tail, and the carried
  state, which is the state at column ``c − 1`` instead of after the padding, is zeroed by the next
  window's reset (:meth:`~hsn.model.state.CellGraphState.reset` fills ``h``, ``q``, ``age`` and
  ``pos``). The trainer checks that reset: a window that follows a cut window and does not reset
  every row at ``t = 0`` raises before anything reads the carried state, and the check survives a
  checkpoint (``extra["cut_pending"]``). Routed runs (masked and sparse rounds) keep full windows,
  because there the padding tail is not inert: the cost and balance terms average over every
  position, the REINFORCE log-probability sums over them, :class:`~hsn.train.lazy_adam.LazyAdamW`
  steps the units any position executed, and the selection's random draws are taken per position.
  The probes of P2.3 run only in routed runs and always see the full window.
  ``stats["window_positions"]`` is the mean number of columns a step's windows computed
  (``train.unroll`` without a cut); ``state_abs_max``, ``eps_mean`` and ``alpha_entropy`` are over
  the computed positions.

Guards: a model that is not an :class:`HSN`, a configuration whose ``model_name`` is not ``hsn``, a
non-finite objective without a loss scaler, and a checkpoint of another configuration or without the
P1.7 entries raise (plan §4.3: errors over silence). ``torch.compile`` stays off.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from hsn.config import RunConfig
from hsn.data.dataset import ByteDataset
from hsn.device import DeviceContext, memory_stats, reset_peak, synchronize
from hsn.eval.evaluate import Metric
from hsn.losses import (
    assemble_terms,
    auxiliary_terms,
    grad_norm_report,
    total_and_parts,
    utility_loss,
)
from hsn.model.hsn import HSN
from hsn.model.router import SCORED_MODES
from hsn.model.state import CellGraphState
from hsn.train.amp import autocast_ctx, autocast_name
from hsn.train.checkpoint import (
    capture_rng_state,
    check_q_resync,
    checkpoint_path,
    load_checkpoint,
    pack_cell_state,
    q_resync_counter,
    restore_rng_state,
    save_checkpoint,
    unpack_cell_state,
)
from hsn.train.lazy_adam import LazyAdamW, active_units_from_indices
from hsn.train.loop import History, StepRecord, StreamLoader, Trainer
from hsn.train.probes import (
    ProbeResult,
    cell_utility,
    choose_probe_position,
    probe_window,
    utility_predictions,
    window_split,
)
from hsn.train.schedules import SelectionSchedule

__all__ = ["STAT_KEYS", "TBPTTStepRecord", "TBPTTTrainer", "trailing_cut"]

logger = logging.getLogger("hsn.train.tbptt")

Batch = dict[str, Tensor]
#: The model statistics every step records (from the ``info`` of :meth:`HSN.forward`).
STAT_KEYS: tuple[str, ...] = ("state_abs_max", "eps_mean", "alpha_entropy")


def trailing_cut(window: Mapping[str, Tensor]) -> int:
    """The number of leading columns of a loader ``window`` the trainer computes (P1.10d, module
    docstring): ``T``, or ``c = 1 +`` the last column where some row holds a document position
    (``doc >= 0``) when that is before ``T``, so that ``[c, T)`` is padding in every row.

    Reads the CPU window only (``doc``, ``loss_mask``, ``reset``, each ``[B, T]``), so the cut costs
    no device synchronisation. A window without any document position keeps ``T`` (there is no
    column to cut after). A loss position or a reset inside the padding tail is a malformed window
    and raises."""
    doc = window["doc"]
    if not isinstance(doc, Tensor) or doc.ndim != 2 or doc.dtype != torch.int64:
        raise TypeError("trailing_cut needs the window's int64 [B, T] document index 'doc'")
    if doc.device.type != "cpu":
        raise ValueError(
            "trailing_cut reads the CPU window (no device synchronisation), got doc on "
            f"{doc.device}"
        )
    T = int(doc.shape[1])
    columns = (doc >= 0).any(dim=0).nonzero().flatten()
    if columns.numel() == 0:
        return T
    cut = int(columns[-1]) + 1
    if cut < T:
        for key in ("loss_mask", "reset"):
            if bool(window[key][:, cut:].any()):
                raise ValueError(
                    f"the window holds a {key} position in its padding tail [{cut}, {T}), where "
                    "doc = -1 in every row: a malformed window"
                )
    return cut


def _columns(batch: Batch, cut: int) -> Batch:
    """The first ``cut`` columns of every ``[B, T]`` window tensor (views, no copy)."""
    return {key: value[:, :cut] for key, value in batch.items()}


@dataclass
class TBPTTStepRecord(StepRecord):
    """A P0.5 :class:`~hsn.train.loop.StepRecord` (``loss`` is the task cross-entropy over the
    step's loss positions) plus the P1.7 statistics: ``total`` (the assembled objective, mean over
    the step's windows), ``parts`` (every λ-weighted term, same mean), ``stats`` (the
    :data:`STAT_KEYS` every step — over the positions the windows computed, which exclude a
    trailing-padding tail cut by P1.10d — ``window_positions``, the mean number of columns
    computed per window, and ``grad_norm/<group>`` on stats steps), the resync counters after the
    step and, in a routed run, ``active_units`` (P2.6: the sorted ids of the units the step's
    windows executed; ``None`` in dense routing, where every unit runs)."""

    total: float = math.nan
    parts: dict[str, float] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)
    windows: int = 0
    resyncs: int = 0
    active_units: list[int] | None = None


class TBPTTTrainer(Trainer):
    """The TBPTT trainer (module docstring). Construction is the P0.5 trainer's: ``model`` must be
    an :class:`HSN` (``cfg.model_name == "hsn"``), ``ctx`` the resolved device context, ``profiler``
    an optional P0.6 profiler, ``run_dir`` where checkpoints go (``runs/<run_id>`` by default)."""

    def __init__(
        self,
        model: nn.Module,
        cfg: RunConfig,
        ctx: DeviceContext,
        profiler: Any | None = None,
        *,
        run_dir: str | Path | None = None,
    ) -> None:
        if not isinstance(model, HSN):
            raise TypeError(
                f"TBPTTTrainer trains the HSN module, got {type(model).__name__}; the baselines "
                "use hsn.train.loop.Trainer"
            )
        if cfg.model_name != "hsn":
            raise ValueError(
                f"model_name={cfg.model_name!r}: TBPTTTrainer needs model_name=hsn (the "
                "configuration and the model must agree)"
            )
        super().__init__(model, cfg, ctx, profiler, run_dir=run_dir)
        self.hsn: HSN = model
        expected = SelectionSchedule.from_config(cfg.train)
        if model.schedule != expected:  # P2.2: one source of truth for the gate and the ramps
            raise ValueError(
                f"the model's selection schedule {model.schedule} is not the configuration's "
                f"{expected}; build the model with HSN.from_config(cfg)"
            )
        self.windows = 0  # windows processed in this run (resumed runs continue the count)
        self.windows_since_resync = 0
        self.resyncs = 0
        self.reinforce_baseline: float | None = None  # P2.2: the EMA reward baseline
        self.counterfactual = cfg.train.lambda_u > 0.0  # P2.3: probes, utility term, EMA features
        # P1.10d: the trailing-padding cut (exact in math under dense routing only, to rounding in
        # floats since P1.10a; module docstring)
        self._cut_enabled = cfg.data.align_docs and cfg.model.routing == "dense"
        self._cut_pending = False  # the carried state comes from a cut window

    # P2.2 ------------------------------------------------------------------------------------

    def _routing_terms(
        self, terms: dict[str, Tensor], info: Mapping[str, Tensor], n_loss: int
    ) -> dict[str, Tensor]:
        """The window's terms plus the REINFORCE term (module docstring) when the gate of this step
        is ``reinforce``; otherwise ``terms`` unchanged. The reward is ``−task`` of the window, or
        ``−(task + cost)`` with the P2.4 cost term on (both λ-weighted parts). ``n_loss`` is the
        window's number of loss positions: without any there is no reward, and a non-finite task
        loss (an fp16 overflow the scaler skips) must not poison the baseline, so both leave the
        baseline alone and add a zero term."""
        if "route_log_prob" not in info or self.hsn.schedule.gate(self.step) != "reinforce":
            return terms
        task = terms["task"]
        reward = -float(task.detach())
        if "cost" in terms:  # P2.4: the hard REINFORCE mask sees the cost penalty only here
            reward -= float(terms["cost"].detach())
        if n_loss == 0 or not math.isfinite(reward):
            return {**terms, "reinforce": torch.zeros_like(task.detach())}
        baseline = self.reinforce_baseline if self.reinforce_baseline is not None else reward
        term = -info["route_log_prob"].float().mean() * (reward - baseline)
        decay = self.cfg.train.reinforce_baseline_decay
        self.reinforce_baseline = decay * baseline + (1.0 - decay) * reward
        return {**terms, "reinforce": term}

    # P2.3 ------------------------------------------------------------------------------------

    def _probe_generator(self) -> torch.Generator:
        """The window's probe generator, seeded by ``(run.seed, window counter)``: the probe draws
        never touch the global RNG and a resumed run replays them (the counter is checkpointed)."""
        seed = np.random.SeedSequence([self.cfg.seed, 0x50323033, self.windows]).generate_state(1)
        return torch.Generator(device="cpu").manual_seed(int(seed[0]))

    def _counterfactual_probe(
        self, train_ds: ByteDataset, window: Batch, batch: Batch, state: CellGraphState
    ) -> ProbeResult | None:
        """The window's probe from its start ``state`` (module docstring), or ``None`` when the
        window is not a probe window, has no loss position to probe, or the step is in the soft
        warm-up of a scored routing (a soft gate has no hard counterfactual)."""
        train = self.cfg.train
        if self.cfg.model.routing in SCORED_MODES and self.hsn.schedule.gate(self.step) == "soft":
            return None
        generator = self._probe_generator()
        if float(torch.rand((), generator=generator)) >= train.probe_fraction:
            return None
        t = choose_probe_position(batch["loss_mask"], train.probe_horizon, generator)
        if t is None:
            return None
        split = window_split(train_ds, window["doc"])
        B, T = batch["tokens"].shape
        with self._segment("probe", tokens=B * T), autocast_ctx(self.ctx):
            return probe_window(
                self.hsn,
                state,
                {**batch, "split": split},
                t,
                n_probe=train.n_probe,
                horizon=train.probe_horizon,
                generator=generator,
            )

    def _utility_term(self, probe: ProbeResult | None) -> Tensor:
        """``λ_u · utility_loss`` of the probe's pairs (fp32, gradient into the scorer only), or
        a zero term when the window has no probe or the probe no valid pair."""
        if probe is None or probe.n_pairs == 0:
            return torch.zeros((), dtype=torch.float32, device=self.ctx.device)
        u_pred, u_target = utility_predictions(self.hsn.router, probe)
        return self.cfg.train.lambda_u * utility_loss(u_pred, u_target)

    def _update_ema(
        self, info: Mapping[str, Tensor], batch: Batch, probe: ProbeResult | None
    ) -> None:
        """One step of the scorer's EMA features after a window (module docstring)."""
        nonpad = batch["doc"] >= 0
        if not bool(nonpad.any()):
            return  # a window of padding selected nothing worth counting
        cell_idx, valid = info["cell_idx"], info["valid"]  # [B, T, K]
        n_cells = self.hsn.cfg.n_cells
        counts = torch.zeros(
            *cell_idx.shape[:2], n_cells, dtype=torch.int64, device=cell_idx.device
        )
        selected = counts.scatter_add(2, cell_idx, valid.to(torch.int64)) > 0  # [B, T, N]
        utility: Tensor | None = None
        measured: Tensor | None = None
        if probe is not None and probe.n_pairs > 0:
            utility, measured = cell_utility(probe, n_cells)
        scorer = self.hsn.scorer
        scorer.load_ema(
            scorer.ema.update(
                selected[nonpad], utility, self.cfg.train.ema_decay, measured=measured
            )
        )

    # window helpers --------------------------------------------------------------------------

    def _window_cut(self, window: Batch) -> int:
        """The columns of the CPU ``window`` this window computes (P1.10d, module docstring):
        :func:`trailing_cut` under dense routing with ``data.align_docs``, ``T`` otherwise. Raises
        when the previous window was cut and this one does not reset every row at ``t = 0``: such a
        row would continue from the state at the cut instead of the state after the padding the
        uncut window computes, so the cut would no longer be exact."""
        T = int(window["doc"].shape[1])
        if self._cut_pending and not bool(window["reset"][:, 0].all()):
            rows = (~window["reset"][:, 0]).nonzero().flatten().tolist()
            raise RuntimeError(
                f"window {self.windows + 1} follows a window cut at its trailing padding "
                f"(P1.10d), but rows {rows} do not reset at t = 0: they would continue from the "
                "state at the cut, not from the state after the padding an uncut window computes. "
                "The cut relies on data.align_docs windows that start a document in every row "
                "after a window where every row ended one."
            )
        cut = trailing_cut(window) if self._cut_enabled else T
        self._cut_pending = cut < T
        return cut

    def _window_start(self, state: CellGraphState) -> CellGraphState:
        """The state a window starts from: detached, and with ``q`` recomputed exactly every
        ``train.q_resync_every`` windows."""
        state = state.detach()
        self.windows += 1
        self.windows_since_resync += 1
        if self.windows_since_resync >= self.cfg.train.q_resync_every:
            state = replace(state, q=state.recompute_q())
            self.resyncs += 1
            self.windows_since_resync = 0
        return state

    def _window_active_units(self, info: Mapping[str, Tensor]) -> Tensor | None:
        """The units a window executed (P2.6), bool ``[M]`` on the CPU; ``None`` in dense routing
        (``info`` has no selection stacks: every unit ran). Every unit when the gate of this step
        is ``soft`` — the masked round runs the whole bank — or when a row ran the dense fallback;
        otherwise the union of ``unit_idx`` over the valid slots."""
        if "unit_idx" not in info:
            return None
        n_units = self.hsn.cell.units.n_units
        if self.hsn.schedule.gate(self.step) == "soft":
            return torch.ones(n_units, dtype=torch.bool)
        return active_units_from_indices(
            info["unit_idx"], info["valid"], n_units, dense_rows=info["dense_fallback"] > 0
        )

    def _grad_groups(self) -> dict[str, list[nn.Parameter]]:
        """Trainable parameters per top-level submodule (plus ``root`` for direct parameters)."""
        groups: dict[str, list[nn.Parameter]] = {}
        direct = [p for p in self.model.parameters(recurse=False) if p.requires_grad]
        if direct:
            groups["root"] = direct
        for name, child in self.model.named_children():
            params = [p for p in child.parameters() if p.requires_grad]
            if params:
                groups[name] = params
        return groups

    @staticmethod
    def _group_grad_norms(groups: Mapping[str, Sequence[nn.Parameter]]) -> dict[str, float]:
        """``grad_norm/<group>``: the L2 norm of the group's gradients (after unscaling, before
        clipping); groups whose parameters carry no gradient are left out."""
        norms: dict[str, float] = {}
        for name, params in groups.items():
            grads = [p.grad.detach().float() for p in params if p.grad is not None]
            if not grads:
                continue
            stacked = torch.stack([torch.linalg.vector_norm(g) for g in grads])
            norms[f"grad_norm/{name}"] = float(torch.linalg.vector_norm(stacked))
        return norms

    # fit -------------------------------------------------------------------------------------

    def fit(
        self,
        train_ds: ByteDataset,
        dev_ds: ByteDataset,
        steps: int,
        *,
        metrics: Mapping[str, Metric] | None = None,
    ) -> History:
        """Train until ``self.step == steps`` (a resumed trainer continues where it stopped).

        ``train_ds`` and ``dev_ds`` are document datasets: the loader is built here so that its
        position can be checkpointed and restored (P0.5). Every record of the returned history is a
        :class:`TBPTTStepRecord`; ``measured`` holds the parameter count (``total_trainable`` of
        :meth:`HSN.param_report`), peak and reserved bytes, tokens/s and the window counters.
        """
        train = self.cfg.train
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if steps > train.steps:
            raise ValueError(
                f"steps={steps} exceeds the schedule horizon train.steps={train.steps}; raise "
                "train.steps instead of running past the learning-rate schedule"
            )
        if steps < self.step:
            raise ValueError(f"steps={steps} is before the current step {self.step}")
        self._refuse_test(train_ds, False)
        self._refuse_test(dev_ds, False)
        report = self.hsn.param_report()
        if "total_trainable" not in report:
            raise KeyError(f"param_report() lacks 'total_trainable': {sorted(report)}")
        history = History(
            param_report=dict(report),
            config_hash=self.config_hash,
            profile=self.ctx.profile,
            autocast=autocast_name(self.ctx),
        )
        logger.info(
            "run %s on %s (%s, autocast %s): params %s",
            self.cfg.run_id,
            self.ctx.device,
            self.ctx.profile,
            history.autocast,
            json.dumps(report, sort_keys=True),
        )
        if self._cut_enabled:
            logger.info("trailing-padding cut (P1.10d): on (dense routing, data.align_docs)")
        else:
            logger.info(
                "trailing-padding cut (P1.10d): off (%s): every window computes all its columns",
                "data.align_docs=false"
                if not self.cfg.data.align_docs
                else f"model.routing={self.cfg.model.routing}",
            )
        loader = self._loader(train_ds, shuffle=True, epochs=None)
        if self._loader_state is not None:
            loader.load_state_dict(self._loader_state)
            self._loader_state = None
        device = self.ctx.device
        state: CellGraphState
        if self._carried_state is None:
            state = self.hsn.init_state(train.batch, device)
        else:
            if not isinstance(self._carried_state, CellGraphState):
                raise TypeError(
                    f"carried state must be a CellGraphState, got {type(self._carried_state)}"
                )
            state = self._carried_state
            self._carried_state = None
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        groups = self._grad_groups()
        max_norm = train.grad_clip if train.grad_clip > 0.0 else math.inf
        n_windows = train.grad_accum
        reference_cost = self.hsn.router.costs.dense_cost()  # P2.4: the cost term's normaliser
        graph = self.hsn.cell.graph
        reset_peak(device)
        train_seconds = 0.0
        total_tokens = 0
        while self.step < steps:
            started = time.perf_counter()
            next_step = self.step + 1
            stats_step = next_step % train.stats_every == 0 or next_step == steps
            term_norms: dict[str, float] = {}
            self.optimizer.zero_grad(set_to_none=True)
            part_sums: dict[str, float] = {}
            total_sum = 0.0
            state_abs_max = 0.0
            eps_sum = 0.0
            entropy_sum = 0.0
            cost_sum = 0.0
            rate_sums: dict[str, float] = {}
            loss_sum = 0.0
            correct = torch.zeros((), dtype=torch.int64, device=device)
            n_loss = 0
            tokens_in_step = 0
            positions_sum = 0  # P1.10d: the columns the step's windows computed
            active: Tensor | None = None  # P2.6: the units the step's windows executed
            probe_counts = {"probe_windows": 0.0, "probe_pairs": 0.0, "probe_positions": 0.0}
            for window_index in range(n_windows):
                window = next(loader)
                cut = self._window_cut(window)  # P1.10d: from the CPU window, no device sync
                positions_sum += cut
                batch = self._to_device(window)
                # the columns the model computes: device views of the window (the probes of P2.3,
                # routed runs only, read the full window)
                fwd = batch if cut == window["doc"].shape[1] else _columns(batch, cut)
                state = self._window_start(state)
                self.hsn.set_train_step(self.step)  # P2.2: the schedules read it
                probe = (
                    self._counterfactual_probe(train_ds, window, batch, state)
                    if self.counterfactual
                    else None
                )  # P2.3: before the forward pass, from the window-start state
                # P1.10b: the columns where some row resets, from the CPU window (cut like the
                # device views), so that HSN.forward needs no device-to-host copy to find them
                reset_columns = window["reset"][:, :cut].any(dim=0).nonzero().flatten().tolist()
                logits, state, info = self._forward(fwd, state, reset_columns=reset_columns)
                window_active = self._window_active_units(info)
                if window_active is not None:
                    active = window_active if active is None else active | window_active
                extras = auxiliary_terms(info, self.cfg, graph, reference_cost=reference_cost)
                terms = assemble_terms({"logits": logits, **info}, fwd, self.cfg, extras)
                loss_mask = fwd["loss_mask"]
                count = int(loss_mask.sum())
                terms = self._routing_terms(terms, info, count)
                if self.counterfactual:  # P2.3: the utility regression
                    terms = {**terms, "utility": self._utility_term(probe)}
                    if probe is not None:
                        probe_counts["probe_windows"] += 1.0
                        probe_counts["probe_pairs"] += float(probe.n_pairs)
                        probe_counts["probe_positions"] += float(probe.positions)
                total, parts = total_and_parts(terms)
                if stats_step and window_index == 0:  # P2.4: per-term gradient norms, one window
                    scale = self.scaler.get_scale() if self.scaler is not None else 1.0
                    term_norms = grad_norm_report(self.model, terms, scale=scale)
                tokens_in_step += int((window["doc"] >= 0).sum())
                if count > 0:
                    flat_targets = fwd["targets"][loss_mask]
                    correct += (logits[loss_mask].float().argmax(dim=-1) == flat_targets).sum()
                    loss_sum += parts["task"] * count
                    n_loss += count
                total_sum += float(total.detach())
                for name, value in parts.items():
                    part_sums[name] = part_sums.get(name, 0.0) + value
                state_abs_max = max(state_abs_max, float(info["state_abs_max"].max()))
                eps_sum += float(info["eps_abs_mean"].float().mean())
                entropy_sum += float(info["alpha_entropy"].float().mean())
                cost_sum += float(info["cost"].float().mean())
                for key in ("explored", "refreshed", "dense_fallback"):
                    if key in info:
                        rate_sums[key] = rate_sums.get(key, 0.0) + float(info[key].float().mean())
                scaled = total / n_windows
                with self._segment("backward", tokens=tokens_in_step):
                    if self.scaler is not None:
                        scaled = self.scaler.scale(scaled)
                    scaled.backward()  # type: ignore[no-untyped-call]
                if self.counterfactual:  # P2.3: once per window, after its selections ran
                    self._update_ema(info, fwd, probe)
            with self._segment("optimizer_step"):
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                group_norms = self._group_grad_norms(groups) if stats_step else {}
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm))
                lr = float(self.optimizer.param_groups[0]["lr"])
                step_kwargs: dict[str, Any] = {}
                if isinstance(self.optimizer, LazyAdamW):  # P2.6: None = every unit (dense)
                    step_kwargs["active_units"] = active
                if self.scaler is not None:
                    self.scaler.step(self.optimizer, **step_kwargs)
                    self.scaler.update()
                else:
                    self.optimizer.step(**step_kwargs)
                self.scheduler.step()
                self.hsn.update_target()
            self.step = next_step
            active_ids = None if active is None else active.nonzero().flatten().tolist()
            mean_total = total_sum / n_windows
            mean_loss = loss_sum / n_loss if n_loss else float("nan")
            accuracy = float(correct) / n_loss if n_loss else float("nan")
            if self.scaler is None and not math.isfinite(mean_total):
                raise RuntimeError(f"the objective is {mean_total} at step {self.step}")
            seconds = time.perf_counter() - started
            train_seconds += seconds
            total_tokens += tokens_in_step
            stats = {
                "state_abs_max": state_abs_max,
                "eps_mean": eps_sum / n_windows,
                "alpha_entropy": entropy_sum / n_windows,
                "route_cost": cost_sum / n_windows,
                "window_positions": positions_sum / n_windows,
                **{key: value / n_windows for key, value in rate_sums.items()},
                **({"active_units": float(len(active_ids))} if active_ids is not None else {}),
                **(probe_counts if self.counterfactual else {}),
                **group_norms,
                **term_norms,
            }
            record = TBPTTStepRecord(
                step=self.step,
                loss=mean_loss,
                accuracy=accuracy,
                lr=lr,
                grad_norm=grad_norm,
                tokens=tokens_in_step,
                loss_positions=n_loss,
                seconds=seconds,
                total=mean_total,
                parts={name: value / n_windows for name, value in part_sums.items()},
                stats=stats,
                windows=self.windows,
                resyncs=self.resyncs,
                active_units=active_ids,
            )
            history.steps.append(record)
            if stats_step:
                logger.info(
                    "step %d/%d total %.4f %s acc %.4f lr %.2e grad_norm %.4f %s tokens/s %.0f%s",
                    self.step,
                    steps,
                    mean_total,
                    " ".join(f"{name} {value:.4f}" for name, value in record.parts.items()),
                    accuracy,
                    lr,
                    grad_norm,
                    " ".join(f"{name} {value:.4f}" for name, value in stats.items()),
                    tokens_in_step / seconds if seconds > 0 else 0.0,
                    "" if active_ids is None else f" active {active_ids}",
                )
            if self.step % train.eval_every == 0 or self.step == steps:
                result = self.evaluate(dev_ds, metrics)
                history.evals.append({"step": self.step, **result})
                logger.info(
                    "eval step %d: %s",
                    self.step,
                    " ".join(f"{k} {v:.4f}" for k, v in result.items() if isinstance(v, float)),
                )
            if self.step % train.ckpt_every == 0 or self.step == steps:
                path = self.save(self.run_dir, loader, state)
                history.checkpoints.append(str(path))
        self._carried_state = state.detach()
        synchronize(device)
        memory = memory_stats(device)
        history.measured = {
            "param_count": float(report["total_trainable"]),
            "active_param_count": float(report["total_trainable"]),  # dense: every parameter runs
            "peak_bytes": float(memory["peak"]),
            "reserved_bytes": float(memory["reserved"]),
            "tokens_per_s": total_tokens / train_seconds if train_seconds > 0 else 0.0,
            "train_seconds": train_seconds,
            "tokens": float(total_tokens),
            "steps": float(self.step),
            "windows": float(self.windows),
            "resyncs": float(self.resyncs),
        }
        return history

    # checkpoints -----------------------------------------------------------------------------

    def save(self, run_dir: str | Path, loader: StreamLoader, state: Any) -> Path:
        """Write ``step_<step>.pt`` into ``run_dir`` with everything a bit-exact resume needs:
        the P0.5 contents plus the carried cell state and the resync counters."""
        if not isinstance(state, CellGraphState):
            raise TypeError(f"save needs the carried CellGraphState, got {type(state).__name__}")
        path = checkpoint_path(run_dir, self.step)
        extra: dict[str, Any] = {
            "config_hash": self.config_hash,
            "model_name": self.cfg.model_name,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "carried_state": pack_cell_state(state),
            "q_resync": q_resync_counter(self.windows, self.windows_since_resync, self.resyncs),
            "reinforce_baseline": self.reinforce_baseline,  # P2.2 (None unless reinforce ran)
            "cut_pending": self._cut_pending,  # P1.10d: the next window must reset every row
        }
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.step,
            capture_rng_state(),
            loader.state_dict(),
            extra=extra,
        )
        logger.info("checkpoint %s", path)
        return path

    def resume(self, path: str | Path) -> int:
        """Load a checkpoint written by :meth:`save` for this configuration; returns its step.

        Restores the weights, the optimiser and scheduler, the loss scaler (required under fp16),
        every RNG state, the loader position, the carried cell state and the resync counters.
        """
        loaded = load_checkpoint(path, self.model, self.optimizer, self.scheduler)
        extra = loaded["extra"]
        if extra.get("config_hash") != self.config_hash:
            raise ValueError(
                f"{path} was written for configuration {extra.get('config_hash')}, this trainer "
                f"runs {self.config_hash}; load_checkpoint() loads weights across configurations"
            )
        if self.scaler is not None:
            if extra.get("scaler") is None:
                raise ValueError(f"{path} has no loss-scaler state but this profile uses fp16")
            self.scaler.load_state_dict(extra["scaler"])
        restore_rng_state(loaded["rng_state"])
        if loaded["loader_state"] is None:
            raise ValueError(f"{path} has no loader state")
        self._loader_state = dict(loaded["loader_state"])
        if extra.get("carried_state") is None:
            raise ValueError(f"{path} has no carried cell state (not a TBPTT checkpoint)")
        self._carried_state = unpack_cell_state(extra["carried_state"], self.ctx.device)
        if "q_resync" not in extra:
            raise ValueError(f"{path} has no q_resync counters (not a TBPTT checkpoint)")
        counters = check_q_resync(extra["q_resync"], str(path))
        self.windows = counters["windows"]
        self.windows_since_resync = counters["since_resync"]
        self.resyncs = counters["resyncs"]
        if "reinforce_baseline" in extra:
            baseline = extra["reinforce_baseline"]
            if baseline is not None and (
                isinstance(baseline, bool)
                or not isinstance(baseline, int | float)
                or not math.isfinite(baseline)
            ):
                raise ValueError(f"{path}: reinforce_baseline must be a finite number or None")
            self.reinforce_baseline = None if baseline is None else float(baseline)
        elif self.cfg.train.gate_mode == "reinforce":
            raise ValueError(
                f"{path} has no reinforce_baseline (written before P2.2) but train.gate_mode="
                "reinforce needs it to continue the same objective"
            )
        # P1.10d: a checkpoint written before the trailing cut never cut a window
        pending = extra.get("cut_pending", False)
        if not isinstance(pending, bool):
            raise ValueError(f"{path}: cut_pending must be a bool, got {pending!r}")
        self._cut_pending = pending
        self.step = int(loaded["step"])
        return self.step
