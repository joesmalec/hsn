"""Training diagnostics of the P1.10 efficiency audit (``scripts/eff_overfit.py``, ``f8884ec``).

Every ``every`` optimiser steps (P1.14: or at the steps of an explicit ``schedule``, the mixed
cadence of :func:`hsn.experiments.rules.diag_schedule`), per top-level module of the model:

* ``update_ratio/<module>`` = ``||theta_after - theta_before|| / ||theta_after||`` for that step;
* ``grad_share/<module>`` = ``||g_module|| / ||g_total||`` of the clipped (and unscaled) gradients,
  the plain norm ratio (not squared, so the shares do not sum to 1).

For an HSN, also the state statistics of one fixed window (:func:`window_state_stats`: the first
``n_docs`` documents, one window from a fresh state, evaluation mode, no gradients): cell
diversity, live cells, participation ratio ``rank_pr``, saturation, mean ``|h|`` and the cell spread
at answer and at valid positions, the write-gate quantiles with the write rate (P1.14: the share of
gates >= 0.5) and the mean gate, and the per-cell write shares of a routed model.

**Hooks (P1.11).** The audit replaced ``optimizer.step``; :class:`Diagnostics` registers
``torch.optim.Optimizer`` step pre- and post-hooks instead (agreed with the audit session, 7 Nov
2026). The record is still taken after ``optimizer.step`` and before ``scheduler.step`` /
``update_target``; the gradient norms are taken in the pre-hook (AdamW and LazyAdamW leave ``.grad``
alone) and the parameters are cloned on sampled steps only. Samples are keyed by the trainer's step
(the step being completed); the optimiser call count is recorded too, and without a loss scaler a
difference between the two raises (a GradScaler may skip ``optimizer.step`` on non-finite fp16
gradients, so with one the hook skips that sample). A step the P1.12 non-finite-gradient guard
skipped calls no optimiser step either: the count is compared net of the trainer's
``skipped_nonfinite`` (P1.15: and ``skipped_spike``), and that step has no sample. The state window
runs under ``no_grad`` with the training mode restored and keeps nothing from the training window
(P2.8: a compiled window is not disturbed). ``seconds`` is the wall time spent inside the hooks,
which the trainer counts as training time; the harness subtracts it.

With a post-optimizer projection (update limits or gradient-growth damping), those legacy
fields still describe the optimizer proposal. Additional ``applied_*`` measurements run after
both projections. ``applied_update_ratio`` uses the legacy after-weight denominator;
``applied_update_ratio_before`` uses the pre-step weights, matching update-limit bounds.
``applied_descent_cosine`` compares the final displacement to the negative clipped gradient;
it measures direction, not whether a finite step actually improves the objective. Additional
``applied_state/`` fields evaluate the final weights. They preserve RNG and do not replace any
frozen diagnostic or verdict input. Default runs retain their original records and hook order.

**Cost (10 Nov 2026).** The state window used to stop the card for about 1.3 s per pass on
``profile:rtx3060`` (two passes per sample with a post-optimizer projection): 0.85 s in the eager
``capture`` forward and 0.47 s in a batched GPU SVD. Under a compiled training window the pass now
runs the window's own compiled diagnostic variant (:class:`~hsn.train.compile.CompiledWindow` with
``capture=True``: same values to rounding, compiled once in :meth:`Diagnostics.attach`, before
training starts), and the
participation ratio is computed from the Gram matrix (:func:`participation_ratio`: the same
quantity as from the singular values, without a decomposition). An eager trainer keeps the eager
forward.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Collection
from typing import Any

import torch
from torch import Tensor, nn

from hsn.config import RunConfig
from hsn.data.dataset import ByteDataset, collate_windows
from hsn.device import DeviceContext
from hsn.model.hsn import HSN
from hsn.train.checkpoint import capture_rng_state, restore_rng_state

__all__ = [
    "Diagnostics",
    "cell_stats",
    "module_groups",
    "participation_ratio",
    "state_window",
    "trained_state",
    "window_state_stats",
]


def module_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Trainable parameters per top-level submodule (the audit's grouping)."""
    groups: dict[str, list[nn.Parameter]] = {}
    for name, child in model.named_children():
        params = [p for p in child.parameters() if p.requires_grad]
        if params:
            groups[name] = params
    return groups


def _norm(tensors: list[Tensor]) -> float:
    if not tensors:
        return 0.0
    return float(
        torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(t.float()) for t in tensors])
        )
    )


def state_window(
    cfg: RunConfig, ctx: DeviceContext, docs: ByteDataset, n_docs: int
) -> dict[str, Tensor]:
    """The first ``n_docs`` documents as one window of ``train.unroll`` positions on the device."""
    examples = [docs.examples[i] for i in range(min(n_docs, len(docs)))]
    if not examples:
        raise ValueError("state_window: no documents")
    batch = collate_windows(
        examples, cfg.train.unroll, batch=len(examples), align_docs=cfg.data.align_docs
    )
    T = cfg.train.unroll
    return {k: v[:, :T].to(ctx.device) for k, v in batch.items()}


class Diagnostics:
    """Per-module update ratios and gradient shares on sampled optimiser steps and (HSN) the state
    statistics of one fixed window (module docstring)."""

    def __init__(
        self,
        model: nn.Module,
        cfg: RunConfig,
        ctx: DeviceContext,
        docs: ByteDataset,
        every: int,
        n_docs: int = 4,
        *,
        schedule: Collection[int] | None = None,
        stability: bool = False,
        sensitivity_epsilon: float = 1e-3,
        sensitivity_position: int = 64,
        sensitivity_horizon: int = 8,
    ) -> None:
        if every < 1:
            raise ValueError(f"Diagnostics every must be >= 1, got {every}")
        self.model = model
        self.cfg = cfg
        self.ctx = ctx
        self.every = every
        # P1.14: an explicit set of steps (the mixed cadence) instead of the multiples of every
        self.schedule = frozenset(schedule) if schedule is not None else None
        self.groups = module_groups(model)
        self.records: list[dict[str, Any]] = []
        self.optimizer_calls = 0
        self.seconds = 0.0
        self.window: dict[str, Tensor] | None = (
            state_window(cfg, ctx, docs, n_docs) if isinstance(model, HSN) else None
        )
        self._trainer: Any = None
        self._scaled = False
        self._pending: dict[str, Any] | None = None
        self._applied_pending: tuple[dict[str, Any], dict[str, Any]] | None = None
        self._observe_applied = False
        self._handles: list[Any] = []
        #: The compiled diagnostic window of the attached trainer (None: the eager forward).
        self._capture_window: Callable[..., Any] | None = None
        #: Wall seconds :meth:`attach` spent compiling that variant (0.0: eager, nothing compiled).
        self.warm_seconds = 0.0
        #: ``[step, unix start, seconds]`` of every sample (hook time: the harness's ``phases``).
        self.intervals: list[list[float]] = []
        self._sample_wall = 0.0
        self._sample_seconds = 0.0
        from hsn.experiments.stability import StabilityObserver

        self.stability = (
            StabilityObserver(sensitivity_epsilon, sensitivity_position, sensitivity_horizon)
            if stability
            else None
        )

    def attach(self, trainer: Any) -> None:
        """Register the step hooks on ``trainer.optimizer`` (``trainer.step`` keys the samples)."""
        if self._handles:
            raise RuntimeError("Diagnostics are already attached")
        if trainer.model is not self.model:
            raise ValueError("the trainer trains another model than these diagnostics observe")
        self._trainer = trainer
        self._scaled = getattr(trainer, "scaler", None) is not None
        # P2.8: a compiled trainer's state window runs compiled too (its capture variant),
        # compiled and recorded here, before training, so that no sample waits for a compile
        self._capture_window = getattr(trainer, "compiled_window", None)
        if self._capture_window is not None and self.window is not None:
            self.warm_seconds = self._warm_capture()
        optimizer = trainer.optimizer
        self._handles.append(optimizer.register_step_pre_hook(self._pre))
        self._handles.append(optimizer.register_step_post_hook(self._post))
        self._observe_applied = (
            trainer.update_limiter is not None or trainer.gradient_growth is not None
        )
        if self._observe_applied:
            trainer.update_post_hooks.append(self._applied)

    def _warm_capture(self) -> float:
        """Compile (and, on CUDA, record) the diagnostic variant of the compiled window: two
        passes of the state window, whose results are dropped. Nothing of the run changes: no
        gradient, no random draw, no parameter or optimiser state. Returns the wall seconds."""
        started = time.perf_counter()
        for _ in range(2):  # CUDA Graph trees warm up once, then record
            self.state_stats()
        return time.perf_counter() - started

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._observe_applied:
            self._trainer.update_post_hooks.remove(self._applied)
            self._observe_applied = False
        self._pending = self._applied_pending = None
        self._capture_window = None

    def _pre(self, optimizer: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
        started = time.perf_counter()
        self.optimizer_calls += 1
        step = int(self._trainer.step) + 1  # the trainer increments its step after the update
        # P1.12: a step the non-finite-gradient guard skipped calls no optimiser step (P1.15: nor
        # does one the spike guard skipped; P1.16: nor a rollback's trigger step)
        skipped = int(getattr(self._trainer, "skipped_nonfinite", 0))
        skipped += int(getattr(self._trainer, "skipped_spike", 0))
        skipped += len(getattr(self._trainer, "rollbacks", ()))
        taken = step - skipped
        if not self._scaled and taken != self.optimizer_calls:
            raise RuntimeError(
                f"diagnostics: trainer step {step} ({taken} not skipped by the non-finite-gradient "
                f"or spike guard) != optimiser call {self.optimizer_calls} without a loss scaler"
            )
        self._pending = None
        sampled = step in self.schedule if self.schedule is not None else step % self.every == 0
        if sampled:
            with torch.no_grad():
                before = {n: [p.detach().clone() for p in ps] for n, ps in self.groups.items()}
                grads = {
                    n: [p.grad.detach() for p in ps if p.grad is not None]
                    for n, ps in self.groups.items()
                }
                g_total = _norm([g for gs in grads.values() for g in gs])
                g_group = {n: _norm(gs) for n, gs in grads.items()}
            self._pending = {"step": step, "before": before, "g_total": g_total, "g": g_group}
            self._sample_wall, self._sample_seconds = time.time(), time.perf_counter() - started
        self.seconds += time.perf_counter() - started

    def _post(self, optimizer: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
        pending, self._pending = self._pending, None
        if pending is None:
            return
        started = time.perf_counter()
        record: dict[str, Any] = {"step": pending["step"], "optimizer_calls": self.optimizer_calls}
        g_total = pending["g_total"]
        with torch.no_grad():
            for n, ps in self.groups.items():
                before = pending["before"][n]
                delta = _norm([p.detach() - b for p, b in zip(ps, before, strict=True)])
                weight = _norm([p.detach() for p in ps])
                record[f"update_ratio/{n}"] = delta / weight if weight > 0 else math.nan
                record[f"grad_share/{n}"] = pending["g"][n] / g_total if g_total > 0 else 0.0
        record.update(self.state_stats())
        self.records.append(record)
        if self._observe_applied:
            record["legacy_measurement_stage"] = "optimizer_proposal"
            self._applied_pending = (pending, record)
        else:
            self._stability_stats(record)
        self.seconds += time.perf_counter() - started
        self._sample_seconds += time.perf_counter() - started
        if not self._observe_applied:
            self.intervals.append([record["step"], self._sample_wall, self._sample_seconds])

    @torch.no_grad()
    def _applied(self, trainer: Any) -> None:
        saved, self._applied_pending = self._applied_pending, None
        if saved is None:
            return
        started = time.perf_counter()
        pending, record = saved
        for name, ps in self.groups.items():
            old = pending["before"][name]
            deltas = [p.detach() - b for p, b in zip(ps, old, strict=True)]
            delta_norm = _norm(deltas)
            before_norm, after_norm = _norm(old), _norm([p.detach() for p in ps])
            record[f"applied_update_ratio/{name}"] = (
                delta_norm / after_norm if after_norm else math.nan
            )
            record[f"applied_update_ratio_before/{name}"] = (
                delta_norm / before_norm if before_norm else math.nan
            )
            record[f"applied_update_norm/{name}"] = delta_norm
            record[f"applied_weight_norm/{name}"] = after_norm
            dot = sum(
                float((-d.double() * p.grad.detach().double()).sum())
                for d, p in zip(deltas, ps, strict=True)
                if p.grad is not None
            )
            denominator = delta_norm * pending["g"][name]
            record[f"applied_descent_cosine/{name}"] = dot / denominator if denominator else None
        rng = capture_rng_state()
        try:
            record.update({f"applied_state/{k}": v for k, v in self.state_stats().items()})
        finally:
            restore_rng_state(rng)
        self._stability_stats(record)
        self.seconds += time.perf_counter() - started
        self._sample_seconds += time.perf_counter() - started
        self.intervals.append([record["step"], self._sample_wall, self._sample_seconds])

    def _stability_stats(self, record: dict[str, Any]) -> None:
        if self.stability is not None and isinstance(self.model, HSN) and self.window is not None:
            record.update(self.stability.observe(self.model, self.window, int(record["step"])))

    @torch.no_grad()
    def state_stats(self) -> dict[str, float]:
        if not isinstance(self.model, HSN) or self.window is None:
            return {}
        return window_state_stats(self.model, self.window, window_fn=self._capture_window)


def participation_ratio(hs: Tensor) -> Tensor:
    """The participation ratio of each ``[N, d]`` matrix of ``hs [P, N, d]``, ``(Σ σ²)² / Σ σ⁴``
    over its singular values ``σ``, without a decomposition: ``Σ σ² = ‖H‖_F²`` and
    ``Σ σ⁴ = ‖H Hᵀ‖_F²``. One batched matmul on the device; a batched SVD of these small
    matrices took 0.47 s per window on ``profile:rtx3060`` (0.3 ms here) for the same number to
    fp32 rounding."""
    gram = hs @ hs.transpose(1, 2)  # [P, N, N]
    return hs.square().sum(dim=(1, 2)).square() / gram.square().sum(dim=(1, 2)).clamp_min(1e-24)


def cell_stats(hs: Tensor) -> dict[str, float]:
    """Statistics of the cell states ``hs [P, N, d]`` at ``P`` positions.

    ``diversity`` is the mean pairwise ``1 - cos(h_i, h_j)``; cells never written sit at exactly 0
    (cosine 0 with every cell) and inflate it, so ``live_cells`` (norm > 1e-6) and
    ``diversity_live`` (over pairs of live cells only) are reported too. ``rank_pr`` is the
    participation ratio of the ``[N, d]`` matrix (1 when every cell holds the same direction, up to
    ``min(N, d)``), its mean and 90th percentile over positions."""
    n = hs.shape[1]
    norms = hs.norm(dim=-1)
    unit = hs / norms.unsqueeze(-1).clamp_min(1e-12)
    cos = unit @ unit.transpose(1, 2)  # [P, N, N]
    off = (cos.sum(dim=(1, 2)) - cos.diagonal(dim1=1, dim2=2).sum(-1)) / (n * (n - 1))
    pr = participation_ratio(hs)
    live = norms > 1e-6  # [P, N]
    eye = torch.eye(n, dtype=torch.bool, device=hs.device)
    pair = live.unsqueeze(-1) & live.unsqueeze(-2) & ~eye
    n_pair = pair.sum(dim=(1, 2))
    div_live = ((1 - cos) * pair).sum(dim=(1, 2)) / n_pair.clamp_min(1)
    has_pair = n_pair > 0
    return {
        "diversity": float((1 - off).mean()),
        "diversity_live": float(div_live[has_pair].mean()) if bool(has_pair.any()) else math.nan,
        "live_cells": float(live.float().sum(-1).mean()),
        "rank_pr": float(pr.mean()),
        "rank_pr_p90": float(torch.quantile(pr, 0.9)),
        "sat99": float((hs.abs() >= 0.99).float().mean()),
        "abs_h": float(hs.abs().mean()),
        "cell_spread": float((hs - hs[:, :1]).abs().amax()),
    }


def _write_rate(values: Tensor) -> dict[str, float]:
    """P1.14, reported next to the gate quantiles (no rule reads them): ``gate/write_rate``, the
    share of the written cells' gate coordinates at or above 0.5 (strong writes), and
    ``gate/mean``, the mean gate (the mean share of a written coordinate replaced per position).
    Over every value (no subsample). A p50 near 0 with a p90 near 0.2 (exp 100) is "few but strong
    writes" when the write rate is above 0, and "no writes" when it is 0 with a small mean."""
    return {
        "gate/write_rate": float((values >= 0.5).float().mean()),
        "gate/mean": float(values.float().mean()),
    }


def _quantiles(values: Tensor) -> dict[str, float]:
    if values.numel() > 2_000_000:
        values = values[torch.randperm(values.numel(), device=values.device)[:2_000_000]]
    q = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], device=values.device))
    return {"gate/p10": float(q[0]), "gate/p50": float(q[1]), "gate/p90": float(q[2])}


@torch.no_grad()
def window_state_stats(
    model: HSN, w: dict[str, Tensor], *, window_fn: Callable[..., Any] | None = None
) -> dict[str, float]:
    """One window from a fresh state in evaluation mode: :func:`cell_stats` at answer and valid
    positions, the write gate ``b`` quantiles over the written cells at valid positions with the
    write rate and the mean gate (:func:`_write_rate`), and for a routed model the per-cell write
    shares.

    Reads ``HSN.forward(..., capture=True)``: ``cell_states [B, T, N, d]`` (h after each position),
    ``cell_gates`` (the gate of each written cell, NaN elsewhere) and ``cell_written [B, T, N]``,
    in every routing mode. A variant whose round has no gate (the GRU-cell variants report NaN)
    gives no gate quantiles. ``window_fn`` is the trainer's compiled window
    (:class:`~hsn.train.compile.CompiledWindow`), called in place of the eager forward with the
    same arguments: the same values to rounding, without the eager loop's host time."""
    was_training = model.training
    model.eval()
    try:
        B = int(w["tokens"].shape[0])
        forward = model if window_fn is None else window_fn
        _, _, info = forward(
            w["tokens"], w["context"], model.init_state(B), w["reset"], capture=True
        )
    finally:
        model.train(was_training)
    h = info["cell_states"].float()  # [B, T, N, d]
    valid = w["doc"] >= 0  # [B, T]
    out: dict[str, float] = {}
    for tag, mask in (("answer", w["loss_mask"].bool()), ("valid", valid)):
        if bool(mask.any()):
            out.update({f"{k}/{tag}": v for k, v in cell_stats(h[mask]).items()})
    written = info["cell_written"][valid]  # [P, N]
    gates = info["cell_gates"].float()[valid][written]  # [S, d]
    gates = gates[~torch.isnan(gates)]
    if gates.numel():
        out.update(_write_rate(gates))
        out.update(_quantiles(gates))
    if model.cfg.routing != "dense":
        writes = written.float().sum(0)  # [N]
        share = writes / writes.sum().clamp_min(1)
        out.update({"write_share/min": float(share.min()), "write_share/max": float(share.max())})
    return out


@torch.no_grad()
def trained_state(
    model: nn.Module, cfg: RunConfig, ctx: DeviceContext, docs: ByteDataset, n_docs: int
) -> dict[str, float]:
    """The audit's ``trained_state`` (:func:`window_state_stats` on the first ``n_docs``
    documents) plus, new in P1.11, the edge logits' absolute maximum and spread (``max - min``),
    which are exactly 0 while the dense cells stay symmetric. ``{}`` for a non-HSN model."""
    if not isinstance(model, HSN):
        return {}
    out = window_state_stats(model, state_window(cfg, ctx, docs, n_docs))
    edge = model.cell.graph.edge_logit.detach().float()
    out["edge_logit/absmax"] = float(edge.abs().max())
    out["edge_logit/spread"] = float(edge.max() - edge.min())
    return out
