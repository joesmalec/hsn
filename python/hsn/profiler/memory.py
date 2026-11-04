"""Peak and reserved memory per segment (P0.6) through the P0.3 helpers, with nesting.

A device has a single peak counter (CUDA) or a single sampled high-water mark (MPS, CPU; see
:mod:`hsn.device`). :class:`PeakTracker` resets it at every segment boundary and lets each open
segment keep the largest value of the sub-intervals it spans, so a parent's peak covers its
children and the gaps between them while a child's peak covers only its own interval. The
numbers are absolute bytes (what the run manifest's ``peak_bytes`` and ``reserved_bytes`` want):
``peak`` is exact on CUDA and the maximum of the boundary samples on MPS and the CPU, as P0.3
documents; ``reserved`` is the maximum of the boundary samples everywhere.

The tracker owns the device's peak counter while segments are open: other readers of
``memory_stats()["peak"]`` see it reset at every boundary. Callers synchronise the device before
``push`` and ``pop`` (the profiler does) so queued work is accounted to the right segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hsn.device import memory_stats, reset_peak

__all__ = ["MemoryPeak", "PeakTracker"]


@dataclass(frozen=True, kw_only=True)
class MemoryPeak:
    peak_bytes: int
    reserved_bytes: int


class PeakTracker:
    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._frames: list[list[int]] = []  # [peak, reserved] per open segment

    @property
    def depth(self) -> int:
        return len(self._frames)

    def push(self) -> None:
        """Open a segment: fold the running sub-interval into the open frames, then reset."""
        self._fold()
        self._frames.append([0, 0])
        reset_peak(self._device)

    def pop(self) -> MemoryPeak:
        """Close the innermost segment and return its peak and reserved bytes."""
        if not self._frames:
            raise RuntimeError("PeakTracker.pop() without a matching push()")
        frame = self._frames.pop()  # first, so a failing read never leaves the frame behind
        self._fold(frame)
        reset_peak(self._device)  # the parent's next sub-interval starts now
        return MemoryPeak(peak_bytes=frame[0], reserved_bytes=frame[1])

    def _fold(self, *extra: list[int]) -> None:
        frames = [*self._frames, *extra]
        if not frames:
            return
        stats = memory_stats(self._device)
        for frame in frames:
            frame[0] = max(frame[0], stats["peak"])
            frame[1] = max(frame[1], stats["reserved"])
