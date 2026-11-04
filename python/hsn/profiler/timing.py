"""Segment timing (P0.6): CUDA events on CUDA, synchronise-and-wall-clock on MPS and the CPU.

A :class:`Timer` measures one interval. ``start()`` first waits for the device queue to drain so
the interval does not inherit work queued before it, then records the start; ``stop()`` records
the end, synchronises again and only then reads the clocks (CUDA events must not be read before a
synchronisation). ``wall_ms`` is therefore host wall-clock *including* the final synchronisation
on every backend. ``device_ms`` is the CUDA-event elapsed time on CUDA; on MPS and the CPU the
plan's method is synchronise-and-wall-clock, so ``device_ms`` equals ``wall_ms`` there (the CPU
is the device, and PyTorch 2.15 gives MPS no separate device clock that the plan adopts).

These are latency-audit conditions (paper §12): every segment starts and ends with an empty
queue, so neighbouring segments never overlap. Throughput under overlap is a different
measurement and is not what a segment reports.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from hsn.device import synchronize

__all__ = ["CUDA_EVENTS", "SYNC_WALL_CLOCK", "Timer", "Timing", "percentile", "timing_method"]

CUDA_EVENTS = "cuda events"
SYNC_WALL_CLOCK = "synchronise + wall clock"


def timing_method(device: torch.device) -> str:
    """The plan's timing method for ``device`` as a report string."""
    return CUDA_EVENTS if device.type == "cuda" else SYNC_WALL_CLOCK


@dataclass(frozen=True, kw_only=True)
class Timing:
    wall_ms: float
    device_ms: float


class Timer:
    """One timed interval on ``device`` (see the module docstring for what each number means)."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._t0: float | None = None
        self._start_event: torch.cuda.Event | None = None

    def start(self) -> None:
        if self._t0 is not None:
            raise RuntimeError("Timer.start() called twice")
        synchronize(self._device)  # the interval starts with an empty queue
        if self._device.type == "cuda":
            self._start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            self._start_event.record()
        self._t0 = time.perf_counter()

    def stop(self) -> Timing:
        if self._t0 is None:
            raise RuntimeError("Timer.stop() called before start()")
        end_event: torch.cuda.Event | None = None
        if self._start_event is not None:
            end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end_event.record()
        synchronize(self._device)  # pitfall: synchronise before reading the events
        wall_ms = (time.perf_counter() - self._t0) * 1000.0
        if self._start_event is not None and end_event is not None:
            device_ms = float(self._start_event.elapsed_time(end_event))
        else:
            device_ms = wall_ms
        self._t0 = None
        self._start_event = None
        return Timing(wall_ms=wall_ms, device_ms=device_ms)


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (linear interpolation between order statistics) of ``values``."""
    if not values:
        raise ValueError("percentile of no values")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must lie in [0, 100], got {q!r}")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))
