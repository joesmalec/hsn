"""Kernel-launch counting (P0.6) through ``torch.profiler``.

At the 2M scale every path is launch-bound (paper A.2), so launches are a primary diagnostic.
On CUDA, Kineto records one device-side event per kernel, memcpy and memset; each is one
dispatch on the device queue and is counted as a launch. Synchronisation events, GPU-side copies
of user annotations and the host-side runtime calls are not launches and are left out by
activity type. PyTorch 2.15 has no profiler activity for MPS (its matmuls produce only CPU-side
events) and the CPU has no device queue, so those backends report :data:`UNSUPPORTED` (``-1``),
never a fake zero; so does a CUDA build whose profiler cannot collect CUDA activity (no CUPTI).

:func:`count_launches` counts one callable. :class:`LaunchCounter` serves the profiler: one
profiler session spans an outermost segment and every segment inside it is a ``record_function``
range. A device event is attributed by the host time of the PyTorch op that launched it (the
link torch itself uses: the event's ``linked_correlation_id`` names a frontend op, an unlinked
CPU event with a non-zero ``external_id``; runtime records can alias op ids and are not used),
falling back to the event's own timestamp, to every segment range containing that time. Ranges
are compared by time only: backward kernels are launched from the autograd worker thread while
the main thread sits inside the ``backward`` segment, and every range starts and ends with a
device synchronisation, so time containment is unambiguous. Parents include their children.
Timings taken with counting on include the profiler's overhead; count in a separate pass when
the latency numbers matter.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, nullcontext
from typing import Any

import torch
from torch.autograd import DeviceType  # type: ignore[attr-defined]
from torch.autograd.profiler_util import FunctionEvent
from torch.profiler import ProfilerActivity, profile, record_function, supported_activities

from hsn.device import DeviceContext, synchronize

__all__ = [
    "CUDA_METHOD",
    "HOST_LAUNCH_CALLS",
    "UNSUPPORTED",
    "LaunchCounter",
    "count_launches",
    "host_launches",
    "supported",
]

#: Report value for "this backend cannot count launches" (and for "not counted").
UNSUPPORTED = -1
CUDA_METHOD = "torch.profiler kernel events"
#: Kineto activity types that are one dispatch on the device queue each.
_LAUNCH_ACTIVITIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
_ACTIVITIES = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
_warmed_up = False
#: P2.8: the host-side CUDA runtime and driver calls that put work on the device queue (kernel
#: launches from PyTorch and from Triton, CUDA Graph launches, async copies and memsets). Under
#: CUDA Graphs the device still runs every kernel (the device events above), but the host issues
#: one graph launch for them: this is the count L1 reduces.
HOST_LAUNCH_CALLS = frozenset(
    {
        "cudaLaunchKernel",
        "cudaLaunchKernelExC",
        "cuLaunchKernel",
        "cuLaunchKernelEx",
        "cudaGraphLaunch",
        "cudaMemcpyAsync",
        "cudaMemsetAsync",
    }
)


def host_launches(events: list[FunctionEvent]) -> int:
    """The host launch calls (:data:`HOST_LAUNCH_CALLS`) among a profile's events (P2.8)."""
    return sum(1 for e in events if e.device_type == DeviceType.CPU and e.name in HOST_LAUNCH_CALLS)


def supported(device: torch.device) -> bool:
    """Whether launches can be counted on ``device`` (CUDA with CUDA profiler activity)."""
    return (
        device.type == "cuda"
        and torch.cuda.is_available()
        and ProfilerActivity.CUDA in supported_activities()  # type: ignore[no-untyped-call]
    )


def count_launches(fn: Callable[[], object], ctx: DeviceContext) -> int:
    """Device launches (kernels, memcpys, memsets) that ``fn()`` issues on ``ctx.device``.

    Returns :data:`UNSUPPORTED` (``-1``) on MPS and the CPU; ``fn`` still runs there. Call ``fn``
    once before counting when its first call initialises libraries (cuBLAS handles launch extra
    kernels the first time).
    """
    device = ctx.device
    if not supported(device):
        fn()
        return UNSUPPORTED
    _warm_up_profiler(device)
    synchronize(device)
    with profile(activities=_ACTIVITIES) as session:
        fn()
        synchronize(device)
    return sum(1 for event in session.events() if _is_launch(event))


class LaunchCounter:
    """One profiler session around an outermost segment with labelled ranges inside it."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._session: profile | None = None
        self._scope = ExitStack()
        self._labels: list[str] = []

    @property
    def active(self) -> bool:
        return self._session is not None

    def start(self) -> None:
        if self._session is not None:
            raise RuntimeError("LaunchCounter.start() while a session is active")
        _warm_up_profiler(self._device)
        self._labels = []
        session = profile(activities=_ACTIVITIES)
        self._scope.enter_context(session)  # may raise (another profiler on this thread)
        self._session = session

    def annotate(self, label: str) -> AbstractContextManager[object]:
        """A range for one segment entry; ``label`` must be unique within the session."""
        if self._session is None:
            return nullcontext()
        if label in self._labels:
            raise ValueError(f"launch-counter label {label!r} used twice in one session")
        self._labels.append(label)
        return record_function(label)

    def abandon(self) -> None:
        """End the session and discard it (used when a segment entry fails part-way)."""
        if self._session is None:
            return
        self._session = None
        self._labels = []
        self._scope.close()

    def stop(self) -> dict[str, int]:
        """End the session and return launches per label (every label gets an entry)."""
        session = self._session
        if session is None:
            raise RuntimeError("LaunchCounter.stop() without an active session")
        self._session = None
        self._scope.close()
        events: list[Any] = session.events()  # type: ignore[no-untyped-call]
        counts = dict.fromkeys(self._labels, 0)
        ranges = [
            e
            for e in events
            if e.is_user_annotation and e.device_type == DeviceType.CPU and e.name in counts
        ]
        frontend = {
            e.id: e
            for e in events
            if e.device_type == DeviceType.CPU
            and e.linked_correlation_id == 0
            and e.external_id != 0
        }
        for event in events:
            if not _is_launch(event):
                continue
            host = frontend.get(event.linked_correlation_id)
            start = host.time_range.start if host is not None else event.time_range.start
            hits = [s for s in ranges if s.time_range.start <= start <= s.time_range.end]
            if not hits:
                raise RuntimeError(
                    f"device event {event.name!r} at {start:.1f} us could not be attributed to "
                    "any open segment (launched outside every segment range)"
                )
            for span in hits:
                counts[span.name] += 1
        return counts


def _is_launch(event: FunctionEvent) -> bool:
    if event.device_type != DeviceType.CUDA:
        return False
    if event.activity_type is None:
        raise RuntimeError(
            f"profiler event {event.name!r} carries no activity type; torch >= 2.15 is expected"
        )
    return event.activity_type in _LAUNCH_ACTIVITIES


def _warm_up_profiler(device: torch.device) -> None:
    """One throw-away session with a one-kernel probe, once per process.

    CUPTI initialisation then stays out of the counted sessions, and a profiler that records no
    device event for a kernel it certainly launched is caught here instead of counting zero.
    """
    global _warmed_up
    if _warmed_up:
        return
    probe = torch.ones(1, device=device)
    synchronize(device)
    with profile(activities=_ACTIVITIES) as session:
        probe.add_(1)
        synchronize(device)
    if not any(_is_launch(event) for event in session.events()):
        raise RuntimeError(
            f"torch.profiler recorded no device event for a one-element add on {device}: CUPTI "
            "is not collecting CUDA activity (driver/toolkit pairing?), so launches cannot be "
            "counted"
        )
    _warmed_up = True
