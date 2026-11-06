"""Profiler (P0.6): wall time, device time, peak memory, launches and energy per named segment.

One :class:`Profiler` per run. Code wraps its phases in ``with profiler.segment(name, tokens=n)``;
segments nest, and the names are fixed project-wide so every report lines up::

    shared_path | scorer | selected_cells | messages | decode | backward | optimizer |
    compile | startup

The ``scorer`` segment is always its own line (plan §0.6 rule 5). ``compile`` and ``startup``
hold compilation and warm-up and are kept apart from the steady-state rows: they can neither
contain nor sit inside another segment, so the first call of a compiled function or a warm-up
iteration can never be folded into a steady-state number (plan P0.6 "Do not").

What a segment measures (details in :mod:`hsn.profiler.timing`, :mod:`hsn.profiler.memory`,
:mod:`hsn.profiler.launches` and :mod:`hsn.profiler.power`):

* every entry starts and ends with a device synchronisation, so ``wall_ms`` is end-to-end host
  time including device completion (latency-audit conditions, paper §12);
* ``device_ms`` is CUDA-event time on CUDA and the synchronised wall time on MPS and the CPU;
* ``peak_bytes`` and ``reserved_bytes`` are the peak allocated and reserved bytes during the
  entry through the P0.3 helpers;
* ``launches`` counts device dispatches on CUDA when ``count_launches=True`` and is ``-1``
  otherwise (MPS and the CPU cannot count; "not counted" is also ``-1``; the report's
  ``launches_method`` says which). Counting adds profiler overhead to the timings;
* ``energy_j`` comes from the power meter, which runs over **outermost** segments only: the
  meters resolve 100–200 ms at best (plan P0.6 pitfall: energy segments span seconds). A
  segment's ``energy_j`` is a number only when a real meter resolved every one of its entries;
  it is ``null`` for nested segments, with :class:`NullMeter`, and when a reading did not
  resolve (an energy counter that did not advance), so a zero is never a measurement.
  ``energy_components`` carries the meter's breakdown (GPU-only and whole-package joules,
  plan §3.8). Starting and stopping a meter leaves idle gaps between outermost segments in
  which the device clocks down, so take latency rows in a pass without a meter.

Nesting is inclusive: a parent's wall time, device time and launches include its children, so a
child's share of its parent is read directly from the two rows. ``report()`` aggregates per
segment: ``n`` entries, ``wall_ms_p50``/``wall_ms_p95`` over the entries, ``device_ms``,
``energy_j``, ``tokens`` and ``launches`` summed, ``peak_bytes``/``reserved_bytes`` the maximum.
The profiler is for single-threaded use: segments are opened and closed on one thread. If the
body of a segment raises, nothing is recorded for it; if the profiler's own bookkeeping raises
(a meter that fails, a launch that cannot be attributed), the error propagates after the segment
has been closed, so the profiler stays consistent and usable.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hsn.device import DeviceContext, synchronize
from hsn.profiler import launches as _launches
from hsn.profiler.launches import LaunchCounter, count_launches
from hsn.profiler.memory import MemoryPeak, PeakTracker
from hsn.profiler.power import (
    NULL_METHOD,
    NVML_METHOD,
    POWERMETRICS_METHOD,
    EnergyReading,
    NullMeter,
    NvmlEnergyMeter,
    PowerMeter,
    PowermetricsMeter,
)
from hsn.profiler.timing import Timer, Timing, percentile, timing_method

__all__ = [
    "LAUNCHES_DISABLED",
    "NULL_METHOD",
    "NVML_METHOD",
    "POWERMETRICS_METHOD",
    "SEGMENT_NAMES",
    "STANDALONE_SEGMENTS",
    "EnergyReading",
    "NullMeter",
    "NvmlEnergyMeter",
    "PowerMeter",
    "PowerSummary",
    "PowermetricsMeter",
    "ProfileReport",
    "Profiler",
    "SegmentReport",
    "count_launches",
]

#: The segment names used everywhere (plan P0.6 interfaces). Later chunks extend this tuple.
SEGMENT_NAMES: tuple[str, ...] = (
    "shared_path",
    "scorer",
    "selected_cells",
    "messages",
    "decode",
    "backward",
    "optimizer",
    "probe",  # P2.3: counterfactual utility probes (no-grad rollouts on a training window)
    "compile",
    "startup",
)
#: Segments that never nest with others: compilation and warm-up stay out of steady state.
STANDALONE_SEGMENTS: frozenset[str] = frozenset({"compile", "startup"})

LAUNCHES_DISABLED = "disabled"


@dataclass(frozen=True, kw_only=True)
class SegmentReport:
    """One row of the report: the per-segment fields of plan P0.6 plus reserved bytes and the
    energy breakdown."""

    n: int
    wall_ms_p50: float
    wall_ms_p95: float
    device_ms: float
    peak_bytes: int
    reserved_bytes: int
    launches: int
    energy_j: float | None  # None: not metered (nested segment, see the module docstring)
    energy_components: dict[str, float]
    tokens: int


@dataclass(frozen=True, kw_only=True)
class PowerSummary:
    """The meter as seen through its readings (``method`` is ``None`` before any reading)."""

    method: str | None
    interval_ms: int | None
    includes_host: bool | None
    n_readings: int
    n_samples: int
    n_unresolved: int  # readings that measured nothing: no meter, or a counter that stood still
    notes: str


@dataclass(frozen=True, kw_only=True)
class ProfileReport:
    profile: str
    device: str
    timing_method: str
    launches_method: str
    power: PowerSummary
    segments: dict[str, SegmentReport]  # in order of first entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "device": self.device,
            "timing_method": self.timing_method,
            "launches_method": self.launches_method,
            "power": asdict(self.power),
            "segments": {name: asdict(segment) for name, segment in self.segments.items()},
        }

    def to_json(self, path: Path | str) -> None:
        """Write the report as strict JSON (a NaN or infinity raises instead of being written)."""
        text = json.dumps(self.to_dict(), indent=2, allow_nan=False)
        Path(path).write_text(text + "\n", encoding="utf-8")


@dataclass
class _Entry:
    name: str
    wall_ms: float
    device_ms: float
    peak_bytes: int
    reserved_bytes: int
    tokens: int
    energy: EnergyReading | None
    label: str
    launches: int | None  # None until the outermost segment's launch session has closed


@dataclass
class _Frame:
    name: str
    tokens: int
    label: str
    timer: Timer
    annotation: AbstractContextManager[object]
    outermost: bool
    undo: list[Callable[[], object]] = field(default_factory=list)


class Profiler:
    def __init__(
        self,
        ctx: DeviceContext,
        power: PowerMeter | None = None,
        count_launches: bool = False,
    ) -> None:
        self.ctx = ctx
        self._device = ctx.device
        self._meter: PowerMeter = power if power is not None else NullMeter()
        self._meter_running = False
        self._count_launches = bool(count_launches)
        self._counter: LaunchCounter | None = (
            LaunchCounter(ctx.device)
            if self._count_launches and _launches.supported(ctx.device)
            else None
        )
        self._peaks = PeakTracker(ctx.device)
        self._stack: list[_Frame] = []
        self._entries: dict[str, list[_Entry]] = {}
        self._pending: list[_Entry] = []  # entries awaiting the session's launch counts
        self._readings: list[EnergyReading] = []
        self._n_unresolved = 0
        self._seq = 0

    # ----------------------------------------------------------------------------------------------
    # segments
    # ----------------------------------------------------------------------------------------------

    @contextmanager
    def segment(self, name: str, tokens: int = 0) -> Iterator[None]:
        """Time, measure and (optionally) count one entry of segment ``name``; nestable.

        ``tokens`` is the number of tokens the entry processed, summed into the report so
        per-token costs can be derived. Entering a name that is already open raises: a segment
        may contain other segments but never itself. ``compile`` and ``startup`` cannot nest
        with any segment. If the body raises, nothing is recorded for it and the exception
        propagates; the profiler stays usable.
        """
        frame = self._enter(name, tokens)
        try:
            yield
        except BaseException as exc:  # noqa: B036 - the body's own exception is re-raised
            try:
                self._exit(frame, ok=False)
            except Exception as cleanup:  # the body's error stays the one callers see
                exc.add_note(f"profiler cleanup also failed: {cleanup!r}")
            raise
        self._exit(frame, ok=True)

    def _enter(self, name: str, tokens: int) -> _Frame:
        if name not in SEGMENT_NAMES:
            raise ValueError(f"unknown segment {name!r}; the segment names are {SEGMENT_NAMES}")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError(f"tokens must be a non-negative int, got {tokens!r}")
        if any(frame.name == name for frame in self._stack):
            raise ValueError(f"segment {name!r} is already open; a segment cannot contain itself")
        if self._stack and name in STANDALONE_SEGMENTS:
            raise ValueError(
                f"segment {name!r} cannot run inside {self._stack[-1].name!r}: compilation and "
                "warm-up stay out of steady-state segments"
            )
        if self._stack and self._stack[0].name in STANDALONE_SEGMENTS:
            raise ValueError(
                f"segment {name!r} cannot run inside {self._stack[0].name!r}: a steady-state "
                "segment measured during warm-up or compilation would fold them into its rows"
            )
        self._seq += 1
        self._entries.setdefault(name, [])  # report order: first entry
        outermost = not self._stack
        frame = _Frame(
            name=name,
            tokens=tokens,
            label=f"hsn.profiler/{name}/{self._seq}",
            timer=Timer(self._device),
            annotation=nullcontext(),
            outermost=outermost,
        )
        synchronize(self._device)
        try:
            if outermost and self._counter is not None:
                self._counter.start()  # before the meter: profiler start-up is not metered
                frame.undo.append(self._counter.abandon)
            if self._counter is not None:
                frame.annotation = self._counter.annotate(frame.label)
            frame.annotation.__enter__()
            frame.undo.append(lambda: frame.annotation.__exit__(None, None, None))
            if outermost:
                self._meter.start()
                self._meter_running = True
                frame.undo.append(self._stop_meter_quietly)
            self._peaks.push()
            frame.undo.append(lambda: self._peaks.pop())
            frame.timer.start()
        except BaseException:
            for undo in reversed(frame.undo):
                with suppress(Exception):
                    undo()
            raise
        self._stack.append(frame)
        return frame

    def _exit(self, frame: _Frame, ok: bool) -> None:
        """Close ``frame`` whatever happens: every step runs, the first failure is re-raised."""
        if not self._stack or self._stack[-1] is not frame:
            open_names = [open_frame.name for open_frame in self._stack]
            raise RuntimeError(
                f"segment {frame.name!r} is closed out of order (open: {open_names}); segments "
                "must close in the reverse order of opening"
            )
        errors: list[BaseException] = []

        def attempt(step: Callable[[], Any]) -> Any:
            try:
                return step()
            except BaseException as exc:  # noqa: B036 - collected, then re-raised below
                errors.append(exc)
                return None

        timing: Timing | None = attempt(frame.timer.stop)  # synchronises before reading
        memory: MemoryPeak | None = attempt(self._peaks.pop)
        attempt(lambda: frame.annotation.__exit__(None, None, None))
        reading: EnergyReading | None = attempt(self._stop_meter) if frame.outermost else None
        counts: dict[str, int] | None = None
        if frame.outermost and self._counter is not None and self._counter.active:
            counts = attempt(self._counter.stop)
        self._stack.pop()
        if ok and not errors and timing is not None and memory is not None:
            entry = _Entry(
                name=frame.name,
                wall_ms=timing.wall_ms,
                device_ms=timing.device_ms,
                peak_bytes=memory.peak_bytes,
                reserved_bytes=memory.reserved_bytes,
                tokens=frame.tokens,
                energy=reading,
                label=frame.label,
                launches=None if self._counter is not None else _launches.UNSUPPORTED,
            )
            self._entries[frame.name].append(entry)
            if self._counter is not None:
                self._pending.append(entry)
        if frame.outermost and self._counter is not None:
            pending, self._pending = self._pending, []
            for entry in pending:
                if counts is None:  # the session failed: drop the entries it should have counted
                    self._entries[entry.name].remove(entry)
                else:
                    entry.launches = counts[entry.label]
        if errors:
            first = errors[0]
            for extra in errors[1:]:
                first.add_note(f"a later cleanup step also failed: {extra!r}")
            raise first

    def _stop_meter(self) -> EnergyReading | None:
        """Stop the meter; the reading is returned only when it resolved energy."""
        if not self._meter_running:
            return None
        self._meter_running = False
        reading = self._meter.stop()
        self._readings.append(reading)
        if reading.method == NULL_METHOD or reading.joules <= 0.0:
            self._n_unresolved += 1  # nothing measured: a zero is never reported as energy
            return None
        return reading

    def _stop_meter_quietly(self) -> None:
        self._stop_meter()

    # ----------------------------------------------------------------------------------------------
    # report
    # ----------------------------------------------------------------------------------------------

    def report(self) -> ProfileReport:
        """Aggregate every completed entry; raises while a segment is still open."""
        if self._stack:
            names = [frame.name for frame in self._stack]
            raise RuntimeError(f"report() called while segments are open: {names}")
        segments: dict[str, SegmentReport] = {}
        for name, entries in self._entries.items():
            if not entries:  # every entry of this segment raised
                continue
            walls = [entry.wall_ms for entry in entries]
            readings = [entry.energy for entry in entries if entry.energy is not None]
            if len(readings) != len(entries):  # some entry was nested or unresolved
                readings = []
            components: dict[str, float] = {}
            for reading in readings:
                for key, value in reading.components.items():
                    components[key] = components.get(key, 0.0) + value
            segments[name] = SegmentReport(
                n=len(entries),
                wall_ms_p50=percentile(walls, 50),
                wall_ms_p95=percentile(walls, 95),
                device_ms=float(sum(entry.device_ms for entry in entries)),
                peak_bytes=max(entry.peak_bytes for entry in entries),
                reserved_bytes=max(entry.reserved_bytes for entry in entries),
                launches=self._launches_of(name, entries),
                energy_j=float(sum(r.joules for r in readings)) if readings else None,
                energy_components=components,
                tokens=sum(entry.tokens for entry in entries),
            )
        return ProfileReport(
            profile=self.ctx.profile,
            device=str(self._device),
            timing_method=timing_method(self._device),
            launches_method=self._launches_method(),
            power=self._power_summary(),
            segments=segments,
        )

    def to_json(self, path: Path | str) -> None:
        self.report().to_json(path)

    def _launches_of(self, name: str, entries: list[_Entry]) -> int:
        if self._counter is None:
            return _launches.UNSUPPORTED
        total = 0
        for entry in entries:
            if entry.launches is None:
                raise RuntimeError(f"segment {name!r} has an entry without a launch count")
            total += entry.launches
        return total

    def _launches_method(self) -> str:
        if not self._count_launches:
            return LAUNCHES_DISABLED
        if self._counter is None:
            return f"unsupported on {self._device.type}"
        return _launches.CUDA_METHOD

    def _power_summary(self) -> PowerSummary:
        if not self._readings:
            return PowerSummary(
                method=None,
                interval_ms=None,
                includes_host=None,
                n_readings=0,
                n_samples=0,
                n_unresolved=0,
                notes="no reading taken yet",
            )
        first, last = self._readings[0], self._readings[-1]
        return PowerSummary(
            method=first.method,
            interval_ms=first.interval_ms,
            includes_host=first.includes_host,
            n_readings=len(self._readings),
            n_samples=sum(reading.n_samples for reading in self._readings),
            n_unresolved=self._n_unresolved,
            notes=last.notes,
        )
