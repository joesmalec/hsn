"""The worker's launch decision (P1.11), a pure function so that it can be simulated on the CPU.

:func:`decide` looks at the queue (FIFO, by experiment id), the running experiments (of every
worker, from the database), the card and the idle state, and returns :class:`Launch` or
:class:`Wait` with the reason:

1. Nothing launches while the database is paused, or while a timing arm of ours is running.
2. The queue is walked in FIFO order and the first arm that fits launches. A CPU-profile arm fits
   when fewer than ``max_concurrent`` of ours run. A CUDA arm fits when:

   * the card was read (no nvidia-smi, or a failed reading, blocks every CUDA arm with the
     reason) and no foreign GPU process is on it (another session's run must not be disturbed,
     and its timing would be contaminated);
   * fewer than ``max_concurrent`` of ours run;
   * its memory estimate fits: ``total - reserve - foreign use - sum over our running arms of
     max(estimate, current use)``; an arm with no estimate yet runs alone, and its measured peak
     becomes the estimate of the arms with its memory key;
   * a timing arm fits only when nothing of ours runs and the machine has been idle (no foreign
     GPU process, no foreign process above 50 % CPU) at two observations at least ``idle_gap_s``
     apart (the worker passes the start of the current idle streak).
3. Starvation guard: once the oldest queued arm has been passed over :data:`MAX_BYPASS` times,
   nothing younger launches until it fits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["MAX_BYPASS", "Card", "Launch", "Limits", "QueueItem", "RunningItem", "Wait", "decide"]

MAX_BYPASS = 3


@dataclass(frozen=True)
class Limits:
    max_concurrent: int = 3
    reserve_mib: float = 1024.0
    idle_gap_s: float = 60.0


@dataclass(frozen=True)
class QueueItem:
    exp_id: int
    device: str  # "cuda" or "cpu" (the arm's hardware profile)
    timing: bool = False
    mem_est_mib: float | None = None  # None: unknown (runs alone)


@dataclass(frozen=True)
class RunningItem:
    exp_id: int
    device: str
    timing: bool = False
    mem_est_mib: float | None = None
    used_mib: float = 0.0  # its processes' current use on the card


@dataclass(frozen=True)
class Card:
    present: bool = False  # nvidia-smi exists
    ok: bool = True  # the last reading succeeded
    total_mib: float = 0.0
    used_mib: float = 0.0  # every process, ours included
    own_used_mib: float = 0.0  # our arms' processes
    foreign: tuple[str, ...] = ()  # counted foreign processes (descriptions)
    error: str | None = None


@dataclass(frozen=True)
class Launch:
    exp_id: int
    bypassed_head: bool = False


@dataclass(frozen=True)
class Wait:
    reason: str
    reasons: dict[int, str] = field(default_factory=dict)


def _available(card: Card, running: Sequence[RunningItem], limits: Limits) -> float:
    foreign = max(0.0, card.used_mib - card.own_used_mib)
    ours = sum(max(r.mem_est_mib or 0.0, r.used_mib) for r in running if r.device == "cuda")
    return card.total_mib - limits.reserve_mib - foreign - ours


def _blocked(
    item: QueueItem,
    running: Sequence[RunningItem],
    card: Card,
    limits: Limits,
    idle_since: float | None,
    now: float,
) -> str | None:
    """Why ``item`` cannot launch now (None: it can)."""
    if len(running) >= limits.max_concurrent:
        return f"{len(running)} running (max_concurrent {limits.max_concurrent})"
    if item.timing:
        if running:
            return "timing arm: waits until nothing of ours runs"
        if idle_since is None or now - idle_since < limits.idle_gap_s:
            gap = f"{limits.idle_gap_s:.0f} s"
            return f"timing arm: needs an idle card and CPU at two looks {gap} apart"
    if item.device != "cuda":
        return None
    if not card.present:
        return "no nvidia-smi on this host: cannot schedule a CUDA arm"
    if not card.ok:
        return f"nvidia-smi failed: {card.error}"
    if card.foreign:
        return "foreign GPU process: " + "; ".join(card.foreign)
    if item.mem_est_mib is None:
        if any(r.device == "cuda" for r in running):
            return "no memory estimate yet: runs alone"
        return None
    available = _available(card, running, limits)
    if item.mem_est_mib > available:
        return f"needs {item.mem_est_mib:.0f} MiB, {available:.0f} MiB available"
    return None


def decide(
    queue: Sequence[QueueItem],
    running: Sequence[RunningItem],
    card: Card,
    *,
    limits: Limits,
    paused: bool = False,
    idle_since: float | None = None,
    now: float = 0.0,
    head_bypassed: int = 0,
) -> Launch | Wait:
    """The next launch, or why there is none (module docstring)."""
    if paused:
        return Wait("paused")
    if not queue:
        return Wait("queue empty")
    if any(r.timing for r in running):
        return Wait("a timing arm is running")
    reasons: dict[int, str] = {}
    for position, item in enumerate(queue):
        if position > 0 and head_bypassed >= MAX_BYPASS:
            reasons[item.exp_id] = f"the oldest arm has waited {head_bypassed} launches"
            break
        why = _blocked(item, running, card, limits, idle_since, now)
        if why is None:
            return Launch(item.exp_id, bypassed_head=position > 0)
        reasons[item.exp_id] = why
    head = queue[0].exp_id
    return Wait(f"exp {head}: {reasons[head]}", reasons)
