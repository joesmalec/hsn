"""Opt-in projection of an optimizer's proposed displacement onto configured L2 balls.

Each group has a radius ``max_ratio * max(||old weights||, norm_floor)``. Adam moments
advance normally once; only excessive parameter displacement is projected. No example,
gradient, or document is removed. Unconstrained proposals are left bitwise untouched.
This limits update size, not cumulative drift or the recurrent network's Jacobian.
An optional norm-growth ceiling anchors selected groups' radii to their initial weight norms;
growing weights then cannot increase their allowed absolute update without bound. It does not
project the weights themselves or constrain their distance from initialization.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any

import torch
from torch import Tensor, nn

from hsn.config import UpdateLimitConfig


class UpdateLimiter:
    def __init__(self, model: nn.Module, config: UpdateLimitConfig) -> None:
        self.config = config
        self.groups: dict[str, list[tuple[str, nn.Parameter, Tensor]]] = {}
        used: set[str] = set()
        named = dict((n, p) for n, p in model.named_parameters() if p.requires_grad)
        for group, settings in config.groups.items():
            if not group.strip():
                raise ValueError("update_limit group names must be nonempty")
            selected: set[str] = set()
            for selector in settings.selectors:
                matches = {n for n in named if fnmatchcase(n, selector)}
                if not matches:
                    raise ValueError(f"update_limit selector {selector!r} matches no parameters")
                selected.update(matches)
            if used & selected:
                raise ValueError(f"update_limit groups overlap on {sorted(used & selected)}")
            used.update(selected)
            self.groups[group] = [
                (n, p, torch.empty_like(p)) for n, p in named.items() if n in selected
            ]
        if not self.groups:
            raise ValueError("construct UpdateLimiter only with configured groups")
        first = next(iter(named.values()))
        # Device counters avoid a host synchronisation at each step. The log/checkpoint reads them.
        self.counts = torch.zeros(len(self.groups), device=first.device, dtype=torch.int64)
        self.max_proposed = torch.zeros(len(self.groups), device=first.device, dtype=first.dtype)
        self.max_applied = torch.zeros_like(self.max_proposed)
        self.last = torch.zeros(len(self.groups), 3, device=first.device, dtype=first.dtype)
        self.applied_steps = 0
        self.initial_norms = {
            name: torch.stack([p.detach().square().sum() for _, p, _ in self.groups[name]])
            .sum()
            .sqrt()
            for name in config.norm_growth_limits
        }

    @torch.no_grad()
    def capture(self) -> None:
        for entries in self.groups.values():
            torch._foreach_copy_([old for _, _, old in entries], [p for _, p, _ in entries])

    @torch.no_grad()
    def apply(self) -> None:
        for index, (group, entries) in enumerate(self.groups.items()):
            old_norm = torch.stack([old.square().sum() for _, _, old in entries]).sum().sqrt()
            denom = old_norm.clamp_min(self.config.norm_floor)
            deltas = [p - old for _, p, old in entries]
            norm = torch.stack([delta.square().sum() for delta in deltas]).sum().sqrt()
            reference = denom
            if group in self.initial_norms:
                ceiling = self.initial_norms[group] * self.config.norm_growth_limits[group]
                reference = torch.minimum(denom, ceiling.clamp_min(self.config.norm_floor))
            radius = reference * self.config.groups[group].max_ratio
            scale = radius / norm.clamp_min(radius)
            clipped = norm > radius
            for (_, p, old), delta in zip(entries, deltas, strict=True):
                # The where preserves the exact optimizer result when no projection is needed.
                p.copy_(torch.where(clipped, old + delta * scale, p))
            proposed = norm / denom
            actual = torch.stack([(p - old).square().sum() for _, p, old in entries]).sum().sqrt()
            applied = actual / denom
            self.counts[index].add_(clipped)
            self.max_proposed[index] = torch.maximum(self.max_proposed[index], proposed)
            self.max_applied[index] = torch.maximum(self.max_applied[index], applied)
            self.last[index] = torch.stack((proposed, applied, scale))
        self.applied_steps += 1

    def state_dict(self) -> dict[str, Any]:
        values = (
            torch.cat(
                (
                    self.counts[:, None],
                    self.max_proposed[:, None],
                    self.max_applied[:, None],
                    self.last,
                ),
                dim=1,
            )
            .detach()
            .cpu()
            .tolist()
        )
        return {
            "applied_steps": self.applied_steps,
            "groups": {
                name: {
                    "parameters": [n for n, _, _ in entries],
                    "clipped_steps": int(v[0]),
                    "max_proposed_ratio": v[1],
                    "max_applied_ratio": v[2],
                    "proposed_ratio": v[3],
                    "applied_ratio": v[4],
                    "scale": v[5],
                    **(
                        {
                            "initial_norm": float(self.initial_norms[name]),
                            "reference_norm_ceiling": max(
                                float(self.initial_norms[name])
                                * self.config.norm_growth_limits[name],
                                self.config.norm_floor,
                            ),
                        }
                        if name in self.initial_norms
                        else {}
                    ),
                }
                for (name, entries), v in zip(self.groups.items(), values, strict=True)
            },
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """Device copies for pre-step divergence snapshots; no host synchronization."""
        return {
            "applied_steps": self.applied_steps,
            "groups": {name: [n for n, _, _ in entries] for name, entries in self.groups.items()},
            "counts": self.counts.clone(),
            "max_proposed": self.max_proposed.clone(),
            "max_applied": self.max_applied.clone(),
            "last": self.last.clone(),
            **(
                {
                    "initial_norms": {
                        name: value.clone() for name, value in self.initial_norms.items()
                    }
                }
                if self.initial_norms
                else {}
            ),
        }

    def load_checkpoint_state(self, raw: dict[str, Any]) -> None:
        if set(raw["groups"]) != set(self.groups):
            raise ValueError("update_limit checkpoint groups differ")
        if set(raw.get("initial_norms", {})) != set(self.initial_norms):
            raise ValueError("update_limit checkpoint initial norm groups differ")
        for name, value in self.initial_norms.items():
            value.copy_(raw["initial_norms"][name])
        self.applied_steps = int(raw["applied_steps"])
        for name, entries in self.groups.items():
            if raw["groups"][name] != [n for n, _, _ in entries]:
                raise ValueError(f"update_limit checkpoint parameters differ for {name}")
        for name in ("counts", "max_proposed", "max_applied", "last"):
            # Mapping order is not part of the config hash. Match saved tensor rows by name
            # when an equivalent config lists its groups in a different order at resume.
            order = list(raw["groups"])
            for index, group in enumerate(self.groups):
                getattr(self, name)[index].copy_(raw[name][order.index(group)])
