"""Opt-in feedback on a fixed TRAIN monitor, with persistent, bounded rate scales.

Loss measures learning; raw gradients and sampled actual parameter changes measure stress.
Neither a large gradient nor a plateau proves signal usefulness. Trial rejection restores
rates, not weights. Emergency recovery remains the existing gradient guard/rollback's job.
"""

from __future__ import annotations

import copy
import json
import math
import statistics
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hsn.config import AdaptiveLRConfig


class _Trial(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scales: dict[str, float]
    loss: float = Field(ge=0)
    remaining: int = Field(ge=1)


class _State(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: int = 1
    scales: dict[str, float]
    monitor_fingerprint: str | None = None
    accepted_steps: int = Field(default=0, ge=0)
    norms: list[float] = []
    anchor_loss: float | None = Field(default=None, ge=0)
    best_loss: float | None = Field(default=None, ge=0)
    healthy_grad: float | None = Field(default=None, ge=0)
    bad_windows: int = Field(default=0, ge=0)
    plateau_windows: int = Field(default=0, ge=0)
    hot_windows: dict[str, int] = {}
    cooldown: int = Field(default=0, ge=0)
    trial: _Trial | None = None
    trial_blocked: bool = False
    observations: list[dict[str, Any]] = []


class AdaptiveLR:
    """State machine independent of tensors, scheduler clocks and validation/test data."""

    def __init__(self, cfg: AdaptiveLRConfig, groups: set[str]) -> None:
        if not groups or not groups <= {"shared", "router", "decoder"}:
            raise ValueError(f"invalid adaptive parameter groups: {groups}")
        self.cfg = cfg
        self.state = _State(scales={key: 1.0 for key in sorted(groups)})

    def scale(self, group: str) -> float:
        return self.state.scales[group]

    @property
    def due(self) -> bool:
        return len(self.state.norms) + 1 >= self.cfg.interval

    def bind_monitor(self, fingerprint: str) -> None:
        previous = self.state.monitor_fingerprint
        if previous is not None and fingerprint != previous:
            raise ValueError("adaptive_lr training monitor fingerprint differs")
        self.state.monitor_fingerprint = fingerprint

    def accepted(self, grad_norm: float) -> bool:
        if not math.isfinite(grad_norm) or grad_norm < 0:
            raise ValueError("adaptive_lr accepted gradient must be finite and nonnegative")
        self.state.accepted_steps += 1
        self.state.norms.append(grad_norm)
        return len(self.state.norms) >= self.cfg.interval

    def _limits(self) -> dict[str, float]:
        limits = {
            "shared": self.cfg.max_update_ratio,
            "router": self.cfg.router_max_update_ratio,
            "decoder": self.cfg.readout_max_update_ratio,
        }
        return {key: limits[key] for key in self.state.scales}

    def _cut(self, groups: set[str]) -> None:
        s = self.state
        if s.trial is not None:
            s.scales = dict(s.trial.scales)
            s.trial = None
        for key in groups:
            s.scales[key] = max(self.cfg.min_scale, s.scales[key] * self.cfg.reduce_factor)
        s.cooldown = self.cfg.cooldown
        s.bad_windows = s.plateau_windows = 0
        s.hot_windows.clear()
        s.trial_blocked = False

    def guard(self, step: int, reason: str) -> None:
        """Cuts survive model/optimizer rollback; skipped gradients never enter trend history."""
        before = dict(self.state.scales)
        self._cut(set(before))
        self.state.norms.clear()
        self.state.anchor_loss = self.state.healthy_grad = None
        self.state.observations.append(
            {
                "step": step,
                "decision": "guard_cut",
                "reason": reason,
                "before": before,
                "scales": dict(self.state.scales),
            }
        )

    def observe(
        self, step: int, loss: float, updates: dict[str, float], *, warmup: bool = False
    ) -> dict[str, Any]:
        """Call after a complete applied-update window; no monitor observation consumes RNG."""
        if (
            not math.isfinite(loss)
            or loss < 0
            or set(updates) != set(self.state.scales)
            or any(not math.isfinite(v) or v < 0 for v in updates.values())
            or len(self.state.norms) != self.cfg.interval
        ):
            raise ValueError("invalid adaptive_lr monitor loss, updates or window length")
        c, s = self.cfg, self.state
        grad = statistics.median(s.norms)
        s.norms.clear()
        before = dict(s.scales)
        record: dict[str, Any] = {
            "step": step,
            "accepted_steps": s.accepted_steps,
            "loss": loss,
            "grad_median": grad,
            "update_ratios": dict(updates),
            "before": before,
            "anchor_loss": s.anchor_loss,
            "healthy_grad": s.healthy_grad,
        }
        s.best_loss = loss if s.best_loss is None else min(loss, s.best_loss)
        hot = {key for key, limit in self._limits().items() if updates[key] > limit}
        if warmup or s.anchor_loss is None:
            s.anchor_loss, s.healthy_grad = loss, grad
            decision = "warmup" if warmup else "baseline"
        else:
            improved = loss < s.anchor_loss * (1 - c.threshold)
            worse = loss > s.anchor_loss * (1 + c.threshold)
            growing = grad > max(s.healthy_grad or 0, 1e-12) * c.grad_growth
            for key in s.scales:
                s.hot_windows[key] = s.hot_windows.get(key, 0) + 1 if key in hot else 0
            s.bad_windows = s.bad_windows + 1 if worse else 0
            # A plateau is eligible only while both gradient and update measurements are healthy.
            healthy = not hot and not growing
            s.plateau_windows = (
                s.plateau_windows + 1 if healthy and not improved and not worse else 0
            )
            overheated = {key for key, count in s.hot_windows.items() if count >= c.patience}
            if s.trial is not None:
                trial = s.trial
                if healthy and loss < trial.loss * (1 - c.threshold):
                    decision = "trial_accepted"
                    s.trial = None
                    s.anchor_loss, s.healthy_grad = loss, grad
                    s.cooldown = c.cooldown
                elif not healthy or loss > trial.loss * (1 + c.threshold) or trial.remaining == 1:
                    decision = "trial_rejected"
                    s.scales = dict(trial.scales)
                    s.trial = None
                    s.trial_blocked = True
                    s.cooldown = c.cooldown
                    s.plateau_windows = 0
                else:
                    trial.remaining -= 1
                    decision = "trial_pending"
            elif s.cooldown > 0:
                s.cooldown -= 1
                decision = "cooldown"
                s.bad_windows = s.plateau_windows = 0
                s.hot_windows.clear()
                if improved and healthy:
                    s.anchor_loss, s.healthy_grad = loss, grad
            elif s.bad_windows >= c.patience or (worse and growing):
                self._cut(set(s.scales))
                s.anchor_loss, s.healthy_grad = loss, grad
                decision = "loss_and_gradient_cut" if growing else "loss_cut"
            elif overheated:
                self._cut(overheated)
                s.anchor_loss, s.healthy_grad = loss, grad
                decision = "update_cut"
            elif improved:
                decision = "improving"
                if healthy:
                    s.anchor_loss, s.healthy_grad = loss, grad
                    s.trial_blocked = False
                s.plateau_windows = 0
            elif s.plateau_windows >= c.patience and not s.trial_blocked:
                raised = {
                    key: min(c.max_scale, value * c.increase_factor)
                    for key, value in s.scales.items()
                }
                if raised != s.scales:
                    s.trial = _Trial(scales=dict(s.scales), loss=loss, remaining=c.trial_windows)
                    s.scales = raised
                    decision = "trial_started"
                else:
                    decision = "at_ceiling"
                s.plateau_windows = 0
            else:
                decision = "hold"
        record.update(decision=decision, scales=dict(s.scales))
        s.observations.append(record)
        return record

    def state_dict(self) -> dict[str, Any]:
        return {"config": self.cfg.model_dump(mode="json"), **self.state.model_dump(mode="json")}

    def load_state_dict(self, raw: dict[str, Any]) -> None:
        raw = copy.deepcopy(raw)
        if raw.pop("config", None) != self.cfg.model_dump(mode="json"):
            raise ValueError("adaptive_lr checkpoint configuration differs or is missing")
        state = _State.model_validate(raw)
        scales = [state.scales] + ([] if state.trial is None else [state.trial.scales])
        if (
            state.version != 1
            or len(state.norms) >= self.cfg.interval
            or len(state.norms) > state.accepted_steps
            or state.cooldown > self.cfg.cooldown
            or (state.trial is not None and state.trial.remaining > self.cfg.trial_windows)
            or any(n < 0 for n in state.norms)
            or any(set(x) != set(self.state.scales) for x in scales)
            or any(
                not self.cfg.min_scale <= v <= self.cfg.max_scale
                for x in scales
                for v in x.values()
            )
            or not set(state.hot_windows) <= set(self.state.scales)
            or any(n < 0 for n in state.hot_windows.values())
        ):
            raise ValueError("malformed adaptive_lr checkpoint state")
        json.dumps(raw, allow_nan=False)
        self.state = state
