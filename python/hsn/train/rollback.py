"""Rollback on a guard trigger (P1.16): in-memory snapshots of the training state, restored in
place.

P1.15's guards skip a step whose gradient is non-finite or spikes; on the stored runs that only
fails fast. A burst ramps over 5 to 40 steps under clipped updates, so by the trigger the weights
have already drifted: every later batch at those weights is large too and P1.12's stop rule ends
the run (exp 189 lost a model at dev 0.755 that way, 8 Nov 2026). PaLM's remedy (Chowdhery et al.
2022) is to restart from a checkpoint about 100 steps before the spike and skip the batches around
it. This module is that remedy without the disk: with ``train.rollback_every`` set the trainer
(:mod:`hsn.train.loop`) keeps a :class:`SnapshotRing` of the two newest snapshots, taken every
``rollback_every`` steps after an applied optimiser step, and a step the guards would skip restores
the **older** one, so the rollback reaches back between ``rollback_every`` and twice that many steps
(the newer one may already sit inside the ramp, and is dropped).

**A snapshot** holds copies of every tensor of ``model.state_dict(keep_vars=True)`` (the parameters
and the persistent buffers: P2.3's scorer averages, an EMA target encoder) and every tensor of the
optimiser state (``exp_avg``, ``exp_avg_sq``, the CPU ``step`` counters of AdamW and of
:class:`~hsn.train.lazy_adam.LazyAdamW`), plus the spike guard's history of applied norms. The copy
buffers are allocated at the first snapshot into a slot and reused afterwards; copies are
``torch._foreach_copy_`` per device, so taking a snapshot allocates nothing after the first and
synchronises nothing (device-to-device copies are queued on the stream; CPU tensors are copied by
the host).

**Restore** copies the snapshot back **into the same tensors** (no new parameter or buffer, so the
CUDA Graphs of P2.8's compiled window keep valid input addresses; the optimiser runs eagerly
outside the graphs). Optimiser state created after the snapshot is dropped and state the snapshot
holds for a parameter that has none now is re-created from it, so the optimiser is exactly the
snapshot's. The scheduler, the step clock and the random generators are not restored: the run's
clock keeps running (as with P1.12's skips), the windows between the snapshot and the trigger are
never replayed, and the trainer (:meth:`hsn.train.loop.Trainer._roll_back`) discards
``rollback_skip`` more steps' worth of windows, starts fresh documents in every row (so the carried
recurrent state resets) and trains at ``lr * rollback_lr_mult`` for ``rollback_lr_steps`` steps.

Memory: two copies of the parameters and both Adam moments, 2 x 3 x 7.3 MB = 44 MB for HSN at A.1.
Time: one copy of about 22 MB per ``rollback_every`` steps.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from hsn.config import TrainConfig

__all__ = ["RollbackPolicy", "Snapshot", "SnapshotRing"]


@dataclass(frozen=True)
class RollbackPolicy:
    """The rollback's settings (``train.rollback_*``, :mod:`hsn.config`)."""

    every: int  # snapshot cadence in steps
    skip: int = 0  # extra steps' worth of windows discarded after a rollback
    lr_mult: float = 1.0  # learning-rate factor after a rollback
    lr_steps: int = 0  # steps the factor lasts
    max_rollbacks: int = 5  # the run stops at its max_rollbacks + 1-th trigger

    def __post_init__(self) -> None:
        if isinstance(self.every, bool) or not isinstance(self.every, int) or self.every < 1:
            raise ValueError(f"rollback every must be an int >= 1, got {self.every!r}")
        if self.skip < 0 or self.lr_steps < 0:
            raise ValueError(f"rollback skip and lr_steps must be >= 0, got {self}")
        if not 0.0 < self.lr_mult <= 1.0:
            raise ValueError(f"rollback lr_mult must lie in (0, 1], got {self.lr_mult}")
        if self.max_rollbacks < 1:
            raise ValueError(f"rollback max_rollbacks must be >= 1, got {self.max_rollbacks}")

    @classmethod
    def from_config(cls, train: TrainConfig) -> RollbackPolicy | None:
        """The policy of ``train`` (None when ``train.rollback_every`` is not set)."""
        if train.rollback_every is None:
            return None
        return cls(
            every=train.rollback_every,
            skip=train.rollback_skip,
            lr_mult=train.rollback_lr_mult,
            lr_steps=train.rollback_lr_steps,
            max_rollbacks=train.rollback_max,
        )


def _model_tensors(model: nn.Module) -> list[Tensor]:
    """The live tensors of the model's state dict (parameters and persistent buffers), in order."""
    return [t for t in model.state_dict(keep_vars=True).values() if isinstance(t, Tensor)]


def _optimizer_entries(optimizer: torch.optim.Optimizer) -> list[tuple[int, str, Any]]:
    """``(parameter index, state key, value)`` over the optimiser's parameters in group order."""
    out: list[tuple[int, str, Any]] = []
    index = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if state:
                for key in sorted(state):
                    out.append((index, key, state[key]))
            index += 1
    return out


def _params(optimizer: torch.optim.Optimizer) -> list[Tensor]:
    return [p for group in optimizer.param_groups for p in group["params"]]


def _copy_into(dst: Sequence[Tensor], src: Sequence[Tensor]) -> None:
    """``dst[i].copy_(src[i])`` for every i, as one ``_foreach_copy_`` per device (no allocation,
    no synchronisation: device copies are queued; CPU copies run on the host)."""
    if len(dst) != len(src):
        raise ValueError(f"copy of {len(src)} tensors into {len(dst)}")
    groups: dict[tuple[torch.device, torch.device], tuple[list[Tensor], list[Tensor]]] = (
        defaultdict(lambda: ([], []))
    )
    for d, s in zip(dst, src, strict=True):
        if d.shape != s.shape or d.dtype != s.dtype:
            raise ValueError(
                f"snapshot tensor {tuple(s.shape)} {s.dtype} does not match the live tensor "
                f"{tuple(d.shape)} {d.dtype}"
            )
        dl, sl = groups[(d.device, s.device)]
        dl.append(d)
        sl.append(s)
    with torch.no_grad():
        for dl, sl in groups.values():
            torch._foreach_copy_(dl, sl)


@dataclass
class Snapshot:
    """One snapshot (module docstring): taken after optimiser step ``step``."""

    step: int
    model: list[Tensor]  # copies of the state-dict tensors, in their order
    keys: list[tuple[int, str]]  # the optimiser's tensor entries: (parameter index, state key)
    optim: list[Tensor]  # copies of those tensors
    other: dict[tuple[int, str], Any]  # the optimiser's non-tensor state values (copied)
    spike_history: list[float] = field(default_factory=list)

    def layout(self) -> tuple[Any, ...]:
        return (
            tuple((t.shape, t.dtype, t.device) for t in self.model),
            tuple(self.keys),
            tuple((t.shape, t.dtype, t.device) for t in self.optim),
        )


class SnapshotRing:
    """The two newest snapshots of a model and its optimiser (module docstring)."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.slots: list[Snapshot] = []  # oldest first, at most two

    @property
    def steps(self) -> list[int]:
        return [s.step for s in self.slots]

    def __len__(self) -> int:
        return len(self.slots)

    def _live(self) -> tuple[list[Tensor], list[tuple[int, str]], list[Tensor], dict[Any, Any]]:
        model = _model_tensors(self.model)
        keys: list[tuple[int, str]] = []
        tensors: list[Tensor] = []
        other: dict[tuple[int, str], Any] = {}
        for index, key, value in _optimizer_entries(self.optimizer):
            if isinstance(value, Tensor):
                keys.append((index, key))
                tensors.append(value)
            else:
                other[(index, key)] = copy.deepcopy(value)
        return model, keys, tensors, other

    def take(self, step: int, spike_history: Iterable[float] = ()) -> Snapshot:
        """Copy the live state into a slot: a new one while fewer than two exist, else the oldest
        one's buffers (reallocated only when the layout changed, e.g. optimiser state appeared)."""
        model, keys, tensors, other = self._live()
        reuse = self.slots.pop(0) if len(self.slots) == 2 else None
        layout = (
            tuple((t.shape, t.dtype, t.device) for t in model),
            tuple(keys),
            tuple((t.shape, t.dtype, t.device) for t in tensors),
        )
        if reuse is not None and reuse.layout() == layout:
            snap = reuse
            snap.step = int(step)
            snap.other = other
            _copy_into(snap.model, model)
            _copy_into(snap.optim, tensors)
        else:
            with torch.no_grad():
                snap = Snapshot(
                    step=int(step),
                    model=[t.detach().clone() for t in model],
                    keys=keys,
                    optim=[t.detach().clone() for t in tensors],
                    other=other,
                )
        snap.spike_history = [float(x) for x in spike_history]
        self.slots.append(snap)
        return snap

    def restore(self) -> Snapshot:
        """Restore the oldest snapshot (the older of two; the only one otherwise) in place and drop
        every newer one; returns it. Raises when the ring is empty."""
        if not self.slots:
            raise RuntimeError("no snapshot to roll back to")
        snap = self.slots[0]
        self.slots = [snap]
        model = _model_tensors(self.model)
        _copy_into(model, snap.model)
        params = _params(self.optimizer)
        wanted: dict[int, set[str]] = defaultdict(set)
        for index, key in [*snap.keys, *snap.other]:
            wanted[index].add(key)
        for index, p in enumerate(params):  # state created after the snapshot goes
            if index not in wanted:
                self.optimizer.state.pop(p, None)
                continue
            state = self.optimizer.state.get(p)
            for key in [k for k in (state or {}) if k not in wanted[index]]:
                del state[key]  # type: ignore[index]
        dst: list[Tensor] = []
        src: list[Tensor] = []
        for (index, key), saved in zip(snap.keys, snap.optim, strict=True):
            state = self.optimizer.state.setdefault(params[index], {})
            live = state.get(key)
            if isinstance(live, Tensor) and live.shape == saved.shape and live.dtype == saved.dtype:
                dst.append(live)
                src.append(saved)
            else:  # absent now (or reshaped): re-create it from the snapshot
                state[key] = saved.detach().clone()
        _copy_into(dst, src)
        for (index, key), value in snap.other.items():
            self.optimizer.state.setdefault(params[index], {})[key] = copy.deepcopy(value)
        return snap
