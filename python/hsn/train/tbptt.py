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
                 total, parts = assemble_loss({"logits": logits, **info}, batch, cfg)
                 backward (scaled under fp16)
    per step:    unscale; grad_norm per group; clip to train.grad_clip; optimiser step (with the
                 units the step's windows executed, P2.6); scheduler step; model.update_target();
                 log parts and statistics

* **Truncated BPTT.** Every window starts from ``state.detach()``: the carried
  :class:`~hsn.model.state.CellGraphState` keeps its values and loses its autograd history, so a
  window's gradient never reaches an earlier one (plan §2.6: fixed unroll, carried detached state;
  one optimiser step is ``train.grad_accum`` windows, P0.5 semantics). Document resets are the
  loader's ``reset`` mask (True at every document start) and are applied by the model itself,
  position by position, before the position is processed (P1.5).
* **Periodic ``q`` resync.** Every ``train.q_resync_every`` windows the summary is replaced by the
  exact mean over cells (:meth:`~hsn.model.state.CellGraphState.recompute_q`) at the window start,
  which removes the floating-point drift incremental sparse updates accumulate (P1.3 pitfall). The
  dense step of P1.5 writes the exact mean, so until P2.5 the resync is a numerical no-op; the
  counters (``windows``, ``windows_since_resync``, ``resyncs``) are part of every checkpoint.
* **The objective** is :func:`hsn.losses.assemble_loss`: every §2.6 term, with ``task`` and
  ``pred`` the only active ones in Phase 1.
* **Statistics.** Every :class:`TBPTTStepRecord` carries the λ-weighted loss ``parts`` (mean over
  the step's windows), the model's own statistics from ``info`` (``state_abs_max`` over the step,
  ``eps_mean`` = mean |ε| and ``alpha_entropy``, means over positions and windows) and the global
  gradient norm before clipping. Every ``train.stats_every`` steps (and at the last step) the record
  also carries ``grad_norm/<group>``, the pre-clip norm per parameter group, one group per top-level
  submodule of the model (``byte_emb``, ``ctx_emb``, ``encoder``, ``predictor``, ``shared``,
  ``cell``, ``router``, ``decoder``; a group without gradients, such as ``halting`` while
  ``model.r_max = 1``, is not reported), and one log line prints all of it.
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
  b̄)`` with ``R = −L_task`` of the window and ``b̄`` an EMA baseline
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

import torch
from torch import Tensor, nn

from hsn.config import RunConfig
from hsn.data.dataset import ByteDataset
from hsn.device import DeviceContext, memory_stats, reset_peak, synchronize
from hsn.eval.evaluate import Metric
from hsn.losses import assemble_loss
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from hsn.train.amp import autocast_name
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
from hsn.train.schedules import SelectionSchedule

__all__ = ["STAT_KEYS", "TBPTTStepRecord", "TBPTTTrainer"]

logger = logging.getLogger("hsn.train.tbptt")

Batch = dict[str, Tensor]
#: The model statistics every step records (from the ``info`` of :meth:`HSN.forward`).
STAT_KEYS: tuple[str, ...] = ("state_abs_max", "eps_mean", "alpha_entropy")


@dataclass
class TBPTTStepRecord(StepRecord):
    """A P0.5 :class:`~hsn.train.loop.StepRecord` (``loss`` is the task cross-entropy over the
    step's loss positions) plus the P1.7 statistics: ``total`` (the assembled objective, mean over
    the step's windows), ``parts`` (every λ-weighted term, same mean), ``stats`` (the
    :data:`STAT_KEYS` every step and ``grad_norm/<group>`` on stats steps), the resync counters
    after the step and, in a routed run, ``active_units`` (P2.6: the sorted ids of the units the
    step's windows executed; ``None`` in dense routing, where every unit runs)."""

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

    # P2.2 ------------------------------------------------------------------------------------

    def _routing_terms(
        self, total: Tensor, parts: dict[str, float], info: Mapping[str, Tensor], n_loss: int
    ) -> tuple[Tensor, dict[str, float]]:
        """The REINFORCE term of a window (module docstring), added when the gate of this step is
        ``reinforce``; otherwise ``(total, parts)`` unchanged. ``n_loss`` is the window's number
        of loss positions: without any there is no reward, and a non-finite task loss (an fp16
        overflow the scaler skips) must not poison the baseline, so both leave everything alone."""
        if "route_log_prob" not in info or self.hsn.schedule.gate(self.step) != "reinforce":
            return total, parts
        reward = -parts["task"]
        if n_loss == 0 or not math.isfinite(reward):
            return total, {**parts, "reinforce": 0.0}
        baseline = self.reinforce_baseline if self.reinforce_baseline is not None else reward
        term = -info["route_log_prob"].float().mean() * (reward - baseline)
        decay = self.cfg.train.reinforce_baseline_decay
        self.reinforce_baseline = decay * baseline + (1.0 - decay) * reward
        return total + term, {**parts, "reinforce": float(term.detach())}

    # window helpers --------------------------------------------------------------------------

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
        reset_peak(device)
        train_seconds = 0.0
        total_tokens = 0
        while self.step < steps:
            started = time.perf_counter()
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
            active: Tensor | None = None  # P2.6: the units the step's windows executed
            for _ in range(n_windows):
                window = next(loader)
                batch = self._to_device(window)
                state = self._window_start(state)
                self.hsn.set_train_step(self.step)  # P2.2: the schedules read it
                logits, state, info = self._forward(batch, state)
                window_active = self._window_active_units(info)
                if window_active is not None:
                    active = window_active if active is None else active | window_active
                total, parts = assemble_loss({"logits": logits, **info}, batch, self.cfg)
                loss_mask = batch["loss_mask"]
                count = int(loss_mask.sum())
                total, parts = self._routing_terms(total, parts, info, count)
                tokens_in_step += int((window["doc"] >= 0).sum())
                if count > 0:
                    flat_targets = batch["targets"][loss_mask]
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
            next_step = self.step + 1
            stats_step = next_step % train.stats_every == 0 or next_step == steps
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
                **{key: value / n_windows for key, value in rate_sums.items()},
                **({"active_units": float(len(active_ids))} if active_ids is not None else {}),
                **group_norms,
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
        self.step = int(loaded["step"])
        return self.step
