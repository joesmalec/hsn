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
from hsn.kernels import trainer as trainer_kernels


def _root_sum_squares(tensors: list[Tensor]) -> Tensor:
    """``sqrt(sum_i sum(t_i ** 2))``: the squares through one multi-tensor product, one sum per
    tensor, then the sum of the sums (the order of the per-tensor loop)."""
    return torch.stack([sq.sum() for sq in torch._foreach_mul(tensors, tensors)]).sum().sqrt()


def _difference(left: list[Tensor], right: list[Tensor]) -> tuple[Tensor, ...]:
    """``left[i] - right[i]`` for every pair, as one multi-tensor op."""
    result: tuple[Tensor, ...] = torch._foreach_sub(left, right)
    return result


class UpdateLimiter:
    #: P2.10b: :meth:`capture` and :meth:`apply` run through the trainer kernels where they can
    #: (:meth:`_apply_fused`); ``False`` keeps the multi-tensor passes, which are the reference.
    fused: bool = True

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
        # P2.10b, the fused passes: every limited tensor in group order, each group's span in
        # that list, the groups' constants as vectors, and (made at the first fused apply) two
        # fp32 buffers per tensor for the squares the norms sum
        self._params: list[Tensor] = [p for entries in self.groups.values() for _, p, _ in entries]
        self._olds = [old for entries in self.groups.values() for _, _, old in entries]
        self._spans: list[tuple[int, int]] = []
        start = 0
        for entries in self.groups.values():
            self._spans.append((start, start + len(entries)))
            start += len(entries)
        self._ratios = torch.tensor(
            [config.groups[name].max_ratio for name in self.groups],
            device=first.device,
            dtype=first.dtype,
        )
        self._growth = torch.tensor(
            [config.norm_growth_limits.get(name, 1.0) for name in self.groups],
            device=first.device,
            dtype=first.dtype,
        )
        self._unbounded = torch.tensor(float("inf"), device=first.device, dtype=first.dtype)
        self._squares: tuple[list[Tensor], list[Tensor]] | None = None

    @torch.no_grad()
    def capture(self) -> None:
        if self.fused and trainer_kernels.copy(self._olds, self._params):
            return  # P2.10b: one launch for every group's tensors (fp32 on CUDA)
        for entries in self.groups.values():
            torch._foreach_copy_([old for _, _, old in entries], [p for _, p, _ in entries])

    @torch.no_grad()
    def apply(self) -> None:
        if not (self.fused and self._apply_fused()):
            self._apply_by_group()
        self.applied_steps += 1

    def _group_sums(self, squares: list[Tensor]) -> Tensor:
        """``[G]``: per group, the sum over its tensors of each tensor's own sum (the reduction
        order of :func:`_root_sum_squares`)."""
        sums = [square.sum() for square in squares]
        return torch.stack([torch.stack(sums[a:b]).sum() for a, b in self._spans])

    def _apply_fused(self) -> bool:
        """:meth:`_apply_by_group` with its elementwise passes as two kernel launches over
        every group's tensors (:func:`hsn.kernels.trainer.limit_squares`, ``limit_project``)
        and the groups' scalars as vectors ``[G]``. The reductions are the same ones (one sum
        per tensor of a buffer of the tensor's shape, then the sum of a group's sums), and every
        other step is the same operation per element or per group, so parameters and
        statistics are those of :meth:`_apply_by_group` bit for bit
        (``tests/test_p2_10b_trainer_kernels.py``). False, with nothing changed, when the
        kernels cannot take the tensors (not fp32 on a CUDA device, not built)."""
        params, olds = self._params, self._olds
        if not trainer_kernels.ready(params[0].device):
            return False
        if self._squares is None:
            self._squares = (
                [torch.empty_like(p) for p in params],
                [torch.empty_like(p) for p in params],
            )
        first, second = self._squares
        if not trainer_kernels.limit_squares(params, olds, first, second):
            return False
        floor = self.config.norm_floor
        denom = self._group_sums(first).sqrt().clamp_min(floor)
        norm = self._group_sums(second).sqrt()
        reference = denom
        if self.initial_norms:
            # a group without a growth limit: an unbounded ceiling leaves its reference alone
            initial = torch.stack(
                [self.initial_norms.get(name, self._unbounded) for name in self.groups]
            )
            reference = torch.minimum(denom, (initial * self._growth).clamp_min(floor))
        radius = reference * self._ratios
        scale = radius / norm.clamp_min(radius)
        clipped = norm > radius
        decision = torch.stack((scale, clipped.to(scale.dtype)), dim=1)  # [G, 2]
        rows = [decision[g] for g, (a, b) in enumerate(self._spans) for _ in range(a, b)]
        # the applied step's squares go where the old weights' were (their sums are queued)
        if not trainer_kernels.limit_project(params, olds, first, rows):
            return False
        proposed = norm / denom
        applied = self._group_sums(first).sqrt() / denom
        self.counts.add_(clipped)
        torch.maximum(self.max_proposed, proposed, out=self.max_proposed)
        torch.maximum(self.max_applied, applied, out=self.max_applied)
        self.last.copy_(torch.stack((proposed, applied, scale), dim=1))
        return True

    def _apply_by_group(self) -> None:
        # P2.10: the elementwise steps of a group go through torch's multi-tensor ops (the same
        # operation per element, in the same order; tests/test_p2_10_trainer_speed.py compares
        # them bit for bit with the per-tensor loop); the reductions stay one sum per tensor.
        for index, (group, entries) in enumerate(self.groups.items()):
            params: list[Tensor] = [p for _, p, _ in entries]
            olds = [old for _, _, old in entries]
            old_norm = _root_sum_squares(olds)
            denom = old_norm.clamp_min(self.config.norm_floor)
            deltas = list(_difference(params, olds))
            norm = _root_sum_squares(deltas)
            reference = denom
            if group in self.initial_norms:
                ceiling = self.initial_norms[group] * self.config.norm_growth_limits[group]
                reference = torch.minimum(denom, ceiling.clamp_min(self.config.norm_floor))
            radius = reference * self.config.groups[group].max_ratio
            scale = radius / norm.clamp_min(radius)
            clipped = norm > radius
            projected = torch._foreach_add(olds, list(torch._foreach_mul(deltas, scale)))
            # The where preserves the exact optimizer result when no projection is needed.
            torch._foreach_copy_(
                params,
                [torch.where(clipped, new, p) for new, p in zip(projected, params, strict=True)],
            )
            proposed = norm / denom
            actual = _root_sum_squares(list(_difference(params, olds)))
            applied = actual / denom
            self.counts[index].add_(clipped)
            self.max_proposed[index] = torch.maximum(self.max_proposed[index], proposed)
            self.max_applied[index] = torch.maximum(self.max_applied[index], applied)
            self.last[index] = torch.stack((proposed, applied, scale))

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
