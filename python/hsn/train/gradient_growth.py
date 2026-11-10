"""Measured gradient growth controls selected groups' actual optimizer displacements.

This controller observes raw, unscaled, pre-clip norms. Windowed medians reject isolated
spikes. Global growth gates a response; group growth, share and share growth determine
which configured groups are damped. Each observation sets an absolute target scale, so
repeated warnings do not multiply the rate down to a permanent floor. Recovery is gradual.
All finite batches contribute; Adam moments advance once and EMA sees the final weights.
"""

from __future__ import annotations

import collections
import copy
import math
import statistics
from fnmatch import fnmatchcase
from typing import Any

import torch
from torch import Tensor, nn

from hsn.config import GradientGrowthConfig, GradNormPrecision
from hsn.train.grad_norm import norm_float64


class GradientGrowth:
    def __init__(self, model: nn.Module, config: GradientGrowthConfig) -> None:
        self.config = config
        self.groups: dict[str, list[tuple[str, nn.Parameter, Tensor]]] = {}
        named = {n: p for n, p in model.named_parameters() if p.requires_grad}
        used: set[str] = set()
        for group, selectors in sorted(config.groups.items()):
            if not group.strip() or not selectors:
                raise ValueError("gradient_growth requires named groups with selectors")
            selected: set[str] = set()
            for selector in selectors:
                matches = {n for n in named if fnmatchcase(n, selector)}
                if not matches:
                    raise ValueError(f"gradient_growth selector {selector!r} matches no parameters")
                selected.update(matches)
            if used & selected:
                raise ValueError(f"gradient_growth groups overlap on {sorted(used & selected)}")
            used.update(selected)
            self.groups[group] = [
                (n, p, torch.empty_like(p)) for n, p in named.items() if n in selected
            ]
        self.scales = dict.fromkeys(self.groups, 1.0)
        self.damped_steps = dict.fromkeys(self.groups, 0)
        self.history: collections.deque[tuple[float, ...]] = collections.deque(
            maxlen=config.reference_lag + config.window
        )
        self.samples = 0
        self.applied_steps = 0
        self.observations: list[dict[str, Any]] = []
        self.pending: dict[str, Any] | None = None

    @torch.no_grad()
    def raw_norms(self, *, precision: GradNormPrecision = "native") -> Tensor:
        """Call after unscaling and before global clipping; one deferred host copy."""
        norms = []
        for entries in self.groups.values():
            if precision == "float64":
                gradients = [p.grad for _, p, _ in entries if p.grad is not None]
                norms.append(norm_float64(gradients or [entries[0][1].new_zeros(())]))
                continue
            values = [p.grad.norm() for _, p, _ in entries if p.grad is not None]
            norms.append(torch.stack(values).norm() if values else entries[0][1].new_zeros(()))
        return torch.stack(norms)

    def observe(self, step: int, global_norm: float, group_norms: list[float]) -> None:
        c = self.config
        row = (global_norm, *group_norms)
        if len(group_norms) != len(self.groups) or any(not math.isfinite(v) or v < 0 for v in row):
            raise ValueError("gradient_growth observes finite, nonnegative norms for every group")
        self.pending = None
        self.samples += 1
        self.history.append(row)
        if len(self.history) < c.reference_lag + c.window or self.samples % c.interval:
            return
        values = list(self.history)
        old, recent = values[: c.window], values[-c.window :]
        old_global = statistics.median(r[0] for r in old)
        recent_global = statistics.median(r[0] for r in recent)
        global_growth = recent_global / max(old_global, c.norm_floor)
        groups: dict[str, Any] = {}
        for index, group in enumerate(self.groups, start=1):
            baseline = statistics.median(r[index] for r in old)
            current = statistics.median(r[index] for r in recent)
            share_old = statistics.median(r[index] / max(r[0], c.norm_floor) for r in old)
            share = statistics.median(r[index] / max(r[0], c.norm_floor) for r in recent)
            growth = current / max(baseline, c.norm_floor)
            share_growth = share / max(share_old, c.norm_floor)
            before = self.scales[group]
            hot = (
                global_growth >= c.global_growth_threshold
                and growth >= c.growth_threshold
                and share >= c.min_share
            )
            target = before
            action = "hold"
            if hot:
                effective = growth * max(1.0, share_growth) ** c.share_power
                target = max(
                    c.min_scale, min(1.0, (c.growth_threshold / effective) ** c.response_power)
                )
                self.scales[group] = min(target, before + c.recovery_rate)
                action = (
                    "damp"
                    if self.scales[group] < before
                    else "relax"
                    if self.scales[group] > before
                    else "hold_hot"
                )
            elif growth <= c.recovery_threshold:
                target = 1.0
                self.scales[group] = min(target, before + c.recovery_rate)
                action = "recover" if self.scales[group] > before else "stable"
            groups[group] = dict(
                reference_median=baseline,
                recent_median=current,
                growth=growth,
                reference_share=share_old,
                share=share,
                share_growth=share_growth,
                hot=hot,
                action=action,
                previous_scale=before,
                target_scale=target,
                scale=self.scales[group],
            )
        self.pending = dict(
            step=step,
            sample=self.samples,
            reference_global_median=old_global,
            recent_global_median=recent_global,
            global_growth=global_growth,
            groups=groups,
        )
        self.observations.append(self.pending)

    @torch.no_grad()
    def capture(self) -> None:
        for name, entries in self.groups.items():
            if self.scales[name] < 1.0 or self.pending is not None:
                torch._foreach_copy_([old for _, _, old in entries], [p for _, p, _ in entries])

    @torch.no_grad()
    def apply(self) -> dict[str, Any] | None:
        for name, entries in self.groups.items():
            scale = self.scales[name]
            if scale == 1.0 and self.pending is None:
                continue
            delta = [p - old for _, p, old in entries]
            if self.pending is not None:
                denom = (
                    torch.stack([old.square().sum() for _, _, old in entries])
                    .sum()
                    .sqrt()
                    .clamp_min(self.config.norm_floor)
                )
                proposed = torch.stack([d.square().sum() for d in delta]).sum().sqrt() / denom
            if scale < 1.0:
                for (_, p, old), d in zip(entries, delta, strict=True):
                    p.copy_(old + d * scale)
                self.damped_steps[name] += 1
            if self.pending is not None:
                applied = (
                    torch.stack([(p - old).square().sum() for _, p, old in entries]).sum().sqrt()
                    / denom
                )
                metrics = torch.stack((proposed, applied)).cpu().tolist()
                self.pending["groups"][name].update(
                    proposed_update_ratio=metrics[0], applied_update_ratio=metrics[1]
                )
        self.applied_steps += 1
        observation, self.pending = self.pending, None
        return observation

    def progress(self) -> dict[str, Any]:
        return dict(
            samples=self.samples,
            applied_steps=self.applied_steps,
            scales=dict(self.scales),
            damped_steps=dict(self.damped_steps),
            observation_count=len(self.observations),
            latest=copy.deepcopy(self.observations[-1]) if self.observations else None,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            **self.progress(),
            "history": list(self.history),
            "parameters": {
                name: [n for n, _, _ in entries] for name, entries in self.groups.items()
            },
            "observations": copy.deepcopy(self.observations),
        }

    def load_state_dict(self, raw: dict[str, Any]) -> None:
        expected = {name: [n for n, _, _ in entries] for name, entries in self.groups.items()}
        if raw.get("parameters") != expected:
            raise ValueError("gradient_growth checkpoint parameter groups differ")
        scales = raw["scales"]
        if set(scales) != set(self.groups) or any(
            not self.config.min_scale <= x <= 1 for x in scales.values()
        ):
            raise ValueError("gradient_growth checkpoint scales are invalid")
        history = [tuple(float(v) for v in row) for row in raw["history"]]
        if len(history) > self.config.reference_lag + self.config.window or any(
            len(row) != len(self.groups) + 1 or any(not math.isfinite(v) or v < 0 for v in row)
            for row in history
        ):
            raise ValueError("gradient_growth checkpoint norm history is invalid")
        samples, applied = raw["samples"], raw["applied_steps"]
        counts = raw["damped_steps"]
        if (
            any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (samples, applied))
            or applied > samples
            or len(history) != min(samples, self.config.reference_lag + self.config.window)
            or set(counts) != set(self.groups)
            or any(
                isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= applied
                for x in counts.values()
            )
        ):
            raise ValueError("gradient_growth checkpoint counters are invalid")
        self.scales = dict(scales)
        self.history.clear()
        self.history.extend(history)
        self.samples = samples
        self.applied_steps = applied
        self.damped_steps = dict(counts)
        self.observations = copy.deepcopy(raw["observations"])
        self.pending = None
