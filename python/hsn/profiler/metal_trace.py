"""Metal System Trace (P2.8 on ``profile:m_series``): this process's GPU execution intervals on an
Apple GPU, from Xcode's ``xctrace``, on the host's monotonic clock.

PyTorch 2.15 has no profiler activity for MPS (:mod:`hsn.profiler.launches`): ``torch.profiler``
records host events only, so it can neither time nor count what an MPS step runs on the GPU.
Instruments' Metal System Trace records every command buffer the GPU executes, system-wide:

* :class:`Recording` runs ``xcrun xctrace record --template "Metal System Trace" --attach <pid>
  --no-prompt`` around a block. It waits for xctrace's ``Ctrl-C to stop the recording`` line before
  the block runs (the earlier ``Starting recording`` line comes before recording is live) and stops
  it with SIGINT afterwards; saving a trace takes seconds to a minute.
* :func:`gpu_intervals` exports the run's ``metal-gpu-intervals`` table (``xctrace export
  --xpath``) and keeps this process's rows: one GPU execution interval per command buffer, start
  in ns since the run started, duration in ns. The run's ``time-info`` table gives that start as a
  mach absolute time and the timebase, and on macOS :func:`time.monotonic_ns` is mach absolute
  time in ns, so an interval's host time is exact without markers (:func:`run_epoch_ns`).
* :class:`Serialised` is a ``TorchDispatchMode`` that waits for the device after every aten op and
  records the op's span on the monotonic clock (:class:`Dispatch`), with the caller's
  classification taken in the op's own context. Every GPU interval then lies inside the span of
  the dispatch that issued it, and :func:`assign` finds it. (No ``torch.profiler``: with Python
  stacks over the ~1.6e5 dispatches of a serialised step its events took tens of GB of memory.)
  ``torch.mps.profiler``'s ``wait_until_completed`` does not do this: the native Metal ops
  (gather, scatter, the index ops) do not commit under it and their work joins the next op's
  command buffer (measured while writing this module). The mode ignores compile internals:
  dynamo runs a frame eager while any other mode is active, whereas with this one compiled code
  stays compiled and its runtime aten calls (extern GEMMs, fallbacks) still pass through the mode;
  the inductor-generated Metal kernels are not aten calls and are serialised by the caller
  (``scripts/profile_step.py``).

* :class:`ChunkedTrace` records a serialised run in chunks of a bounded number of dispatches:
  a serialised training step (about 1.6e5 synchronised dispatches) costs about 36 KB of kernel
  trace per dispatch, 6-10 GB for one trace that takes over ten minutes to save. Each chunk is
  saved, exported and deleted before the next starts, and marker work at both ends of a chunk
  proves the recording covered it.

Measured pitfalls (macOS 26.5, Xcode 26.4.1, xctrace 16.0): a second ``xctrace record`` running at
the same time (an orphaned recorder, say) leaves every other recording without Metal data, so
:class:`Recording` refuses to start while one runs; xctrace writes a raw kernel-trace buffer per
recording (``instruments*.ktrace``) to the per-user temporary directory, ignores ``TMPDIR`` and
never deletes the buffer, so :class:`Recording` deletes the buffers its own recording created;
the instruments ``GPU`` plus ``Metal Application`` without the template recorded this process's
intervals only sometimes, so the template is used.

The export XML compresses repeated values: an element with ``id="n"`` defines a value and a later
``ref="n"`` element repeats it. :func:`iter_rows` resolves both into plain values.

What serialised numbers mean: an op's serialised GPU interval is its device time on an otherwise
idle GPU. It can exceed the time the same op takes inside a natural, coalesced command buffer
(the GPU clocks down between dispatches, and nothing overlaps), so a report should put the natural
step's GPU busy time (:func:`busy_ns`) next to the serialised sum.

Errors over silence: a missing ``xctrace``, a refused attach, an export that fails or a recording
without this process's intervals raise :class:`MetalTraceError` with the reason.
"""

from __future__ import annotations

import bisect
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, NamedTuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode

__all__ = [
    "LIVE_LINE",
    "TEMPLATE",
    "Assignment",
    "ChunkedTrace",
    "GPUInterval",
    "MetalTraceError",
    "Recording",
    "Dispatch",
    "Serialised",
    "Value",
    "assign",
    "available",
    "busy_ns",
    "encoders_per_command_buffer",
    "export_table",
    "gpu_intervals",
    "is_recorder",
    "iter_rows",
    "ktrace_dir",
    "mps_sync",
    "other_recordings",
    "run_epoch_ns",
    "version",
]

TEMPLATE = "Metal System Trace"
#: xctrace prints this once the recording is live (the first line, "Starting recording ...", comes
#: about two seconds earlier).
LIVE_LINE = "Ctrl-C to stop the recording"
_XCRUN = "/usr/bin/xcrun"


class MetalTraceError(RuntimeError):
    """xctrace is missing, refused, failed, or recorded nothing for this process."""


# --------------------------------------------------------------------------------------------------
# xctrace
# --------------------------------------------------------------------------------------------------


def available() -> tuple[bool, str]:
    """Whether ``xcrun xctrace`` can run here, and the reason when it cannot."""
    if sys.platform != "darwin":
        return False, f"Metal System Trace needs macOS (this is {sys.platform})"
    if not Path(_XCRUN).exists():
        return False, f"{_XCRUN} is missing (install Xcode)"
    try:
        found = subprocess.run(
            [_XCRUN, "--find", "xctrace"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"xcrun --find xctrace failed: {exc}"
    if found.returncode != 0:
        return False, f"xctrace not found (install Xcode): {found.stderr.strip()}"
    return True, ""


def _require() -> None:
    ok, reason = available()
    if not ok:
        raise MetalTraceError(reason)


def version() -> str:
    """``xctrace version``, e.g. ``xctrace version 16.0 (17E202)``."""
    _require()
    out = subprocess.run(
        [_XCRUN, "xctrace", "version"], capture_output=True, text=True, timeout=60, check=False
    )
    if out.returncode != 0:
        raise MetalTraceError(f"xctrace version failed: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout.strip() or out.stderr.strip()


def ktrace_dir() -> Path | None:
    """Where xctrace leaves its raw kernel-trace buffers (``instruments*.ktrace``): the per-user
    temporary directory (``getconf DARWIN_USER_TEMP_DIR``; xctrace ignores ``TMPDIR``)."""
    try:
        out = subprocess.run(
            ["getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return None
    path = out.stdout.strip()
    return Path(path) if out.returncode == 0 and path else None


def _ktraces(directory: Path | None) -> set[Path]:
    return set(directory.glob("instruments*.ktrace")) if directory is not None else set()


def other_recordings() -> list[str]:
    """``xctrace record`` processes running now (``pid command``). A second recording gets no Metal
    data (measured: an orphaned recorder left every later trace empty), so :class:`Recording`
    refuses to start while one runs."""
    try:
        out = subprocess.run(
            ["ps", "-Ao", "pid=,command="], capture_output=True, text=True, timeout=20, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if is_recorder(line)]


def is_recorder(ps_line: str) -> bool:
    """Whether a ``ps -Ao pid=,command=`` line is an ``xctrace record`` process: its executable is
    ``xctrace`` (``xcrun`` execs it) and its first argument ``record``; a shell whose command
    line merely mentions the words is not."""
    parts = ps_line.split()
    return len(parts) >= 3 and Path(parts[1]).name == "xctrace" and parts[2] == "record"


class Recording:
    """A Metal System Trace of ``pid`` (this process by default) around a ``with`` block, saved to
    ``out`` (which must not exist yet). The block starts once the recording is live; on exit (or at
    :meth:`stop`) xctrace is stopped with SIGINT and must save the trace. An exception from the
    block still stops the recording, and then propagates unchanged.

    xctrace leaves a raw kernel-trace buffer per recording in the user temporary directory and never
    deletes it (it is about as large as the trace, gigabytes for a serialised step); with
    ``clean_ktrace`` the buffers that appeared during this recording are deleted on exit
    (``ktrace_bytes`` says how much), and no other file is touched."""

    def __init__(
        self,
        out: Path,
        pid: int | None = None,
        *,
        start_timeout_s: float = 120.0,
        stop_timeout_s: float = 900.0,
        clean_ktrace: bool = True,
    ) -> None:
        self.out = Path(out)
        self.pid = os.getpid() if pid is None else pid
        self.start_timeout_s = start_timeout_s
        self.stop_timeout_s = stop_timeout_s
        self.clean_ktrace = clean_ktrace
        self.ktrace_bytes = 0
        self.lines: list[str] = []
        self._proc: subprocess.Popen[str] | None = None
        self._live = threading.Event()
        self._reader: threading.Thread | None = None
        self._stopped = False
        self._ktrace_dir: Path | None = None
        self._ktraces_before: set[Path] = set()

    def command(self) -> list[str]:
        return [
            _XCRUN,
            "xctrace",
            "record",
            "--template",
            TEMPLATE,
            "--attach",
            str(self.pid),
            "--output",
            str(self.out),
            "--no-prompt",
        ]

    def _read(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            self.lines.append(line.rstrip())
            if LIVE_LINE in line:
                self._live.set()

    def __enter__(self) -> Recording:
        _require()
        if self.out.exists():
            raise MetalTraceError(f"{self.out} exists; xctrace would not overwrite it")
        running = other_recordings()
        if running:
            raise MetalTraceError(
                "another xctrace recording is running (its trace session would leave this one "
                "without Metal data): " + " | ".join(running)
            )
        if self.clean_ktrace:
            self._ktrace_dir = ktrace_dir()
            self._ktraces_before = _ktraces(self._ktrace_dir)
        self._proc = subprocess.Popen(
            self.command(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        waited = 0.0
        while not self._live.wait(0.25):
            waited += 0.25
            if self._proc.poll() is not None or waited >= self.start_timeout_s:
                code = self._proc.returncode
                self._kill()
                why = f"exited {code}" if code is not None else "timed out"
                raise MetalTraceError(
                    f"xctrace did not start recording ({why} after {waited:.0f} s): "
                    + " | ".join(self.lines[-8:])
                )
        return self

    def _kill(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(30)
        self._clean()

    def _clean(self) -> None:
        """Delete the kernel-trace buffers that appeared since this recording started."""
        if not self.clean_ktrace or self._ktrace_dir is None:
            return
        for path in _ktraces(self._ktrace_dir) - self._ktraces_before:
            try:
                size = path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                continue
            self.ktrace_bytes += size

    def stop(self) -> None:
        """End the recording now (SIGINT); :meth:`__exit__` then waits for the trace to be saved."""
        if self._proc is not None and not self._stopped:
            self._proc.send_signal(signal.SIGINT)
            self._stopped = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        proc = self._proc
        assert proc is not None
        self.stop()
        try:
            code = proc.wait(self.stop_timeout_s)
        except subprocess.TimeoutExpired:
            self._kill()
            if exc_type is None:
                raise MetalTraceError(
                    f"xctrace did not save {self.out} within {self.stop_timeout_s:.0f} s"
                ) from None
            return
        if self._reader is not None:
            self._reader.join(10)
        self._clean()
        if exc_type is None and (code != 0 or not self.out.exists()):
            raise MetalTraceError(
                f"xctrace exited {code} without saving {self.out}: " + " | ".join(self.lines[-8:])
            )


def export_table(trace: Path, schema: str, out: Path, *, run: int = 1) -> Path:
    """Export the ``schema`` table of run ``run`` of ``trace`` as XML to ``out``."""
    _require()
    xpath = f'/trace-toc/run[@number="{run}"]/data/table[@schema="{schema}"]'
    with open(out, "w") as f:
        done = subprocess.run(
            [_XCRUN, "xctrace", "export", "--input", str(trace), "--xpath", xpath],
            stdout=f,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if done.returncode != 0:
        raise MetalTraceError(f"xctrace export of {schema} failed: {done.stderr.strip()}")
    return out


# --------------------------------------------------------------------------------------------------
# Export parsing
# --------------------------------------------------------------------------------------------------


class Value(NamedTuple):
    """One cell of an exported row: the element's tag and text, its ``fmt`` attribute, the pid
    found in it or its children (a ``process`` cell), and its children's texts."""

    tag: str
    text: str | None
    fmt: str | None
    pid: int | None
    fields: tuple[str, ...]


def _value(el: ET.Element, ids: dict[str, Value]) -> Value:
    ref = el.get("ref")
    if ref is not None:
        if ref not in ids:
            raise MetalTraceError(f"export row refers to undefined id {ref!r} (<{el.tag}>)")
        return ids[ref]
    pid: int | None = int(el.text) if el.tag == "pid" and el.text else None
    fields = []
    for child in el:
        v = _value(child, ids)
        if pid is None and v.pid is not None:
            pid = v.pid
        fields.append(v.text or "")
    value = Value(el.tag, el.text, el.get("fmt"), pid, tuple(fields))
    key = el.get("id")
    if key is not None:
        ids[key] = value
    return value


def iter_rows(path: Path) -> Iterator[dict[str, Value]]:
    """The rows of an exported table, ``{column mnemonic: Value}``, with ``ref`` resolved. Values
    are copied when first defined, so rows are cleared as they are read (exports are large)."""
    ids: dict[str, Value] = {}
    mnemonics: list[str] = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "mnemonic":
            mnemonics.append(el.text or "")
        elif el.tag == "row":
            cells = list(el)
            if len(cells) != len(mnemonics):
                raise MetalTraceError(
                    f"export row has {len(cells)} cells for {len(mnemonics)} columns in {path}"
                )
            yield {name: _value(cell, ids) for name, cell in zip(mnemonics, cells, strict=True)}
            el.clear()


def _int(value: Value, column: str) -> int:
    if value.text is None:
        raise MetalTraceError(f"empty {column!r} cell (<{value.tag}>) in a trace export")
    return int(value.text)


def run_epoch_ns(trace: Path, workdir: Path) -> int:
    """The trace run's start on the :func:`time.monotonic_ns` clock (``time-info``: the mach
    absolute time of the run's start times the mach timebase)."""
    rows = list(iter_rows(export_table(trace, "time-info", workdir / "time-info.xml")))
    if len(rows) != 1:
        raise MetalTraceError(f"time-info has {len(rows)} rows in {trace}; one expected")
    row = rows[0]
    numer, denom = (int(x) for x in row["timebase-info"].fields)
    return _int(row["mabs-epoch"], "mabs-epoch") * numer // denom


@dataclass(frozen=True, kw_only=True)
class GPUInterval:
    """One GPU execution interval (one command buffer) on the :func:`time.monotonic_ns` clock."""

    start_ns: int
    duration_ns: int
    channel: str
    command_buffer: int

    @property
    def end_ns(self) -> int:
        return self.start_ns + self.duration_ns


def gpu_intervals(trace: Path, pid: int, workdir: Path) -> list[GPUInterval]:
    """This process's GPU execution intervals in ``trace``, sorted by start; raises when the
    trace holds none (a recording that missed the work, or another pid)."""
    epoch = run_epoch_ns(trace, workdir)
    path = export_table(trace, "metal-gpu-intervals", workdir / "metal-gpu-intervals.xml")
    out = [
        GPUInterval(
            start_ns=_int(row["start"], "start") + epoch,
            duration_ns=_int(row["duration"], "duration"),
            channel=row["channel-name"].text or "",
            command_buffer=_int(row["cmdbuffer-id"], "cmdbuffer-id"),
        )
        for row in iter_rows(path)
        if row["process"].pid == pid
    ]
    if not out:
        raise MetalTraceError(f"{trace} holds no GPU interval of pid {pid}")
    return sorted(out, key=lambda g: g.start_ns)


def encoders_per_command_buffer(intervals: Sequence[GPUInterval]) -> dict[int, int]:
    """How many command buffers ran 1, 2, ... GPU intervals (one ``metal-gpu-intervals`` row per
    encoder execution). Under :class:`Serialised` every dispatch is its own command buffer with
    one encoder; more means work was coalesced into one command buffer."""
    per: Counter[int] = Counter(g.command_buffer for g in intervals)
    return dict(sorted(Counter(per.values()).items()))


def busy_ns(intervals: Sequence[GPUInterval], lo_ns: int, hi_ns: int) -> int:
    """GPU busy time inside ``[lo_ns, hi_ns)``: the length of the union of the intervals clipped
    to the window (command buffers on different channels may overlap)."""
    total = 0
    cursor = lo_ns
    for g in sorted(intervals, key=lambda g: g.start_ns):
        start, end = max(g.start_ns, cursor), min(g.end_ns, hi_ns)
        if end > start:
            total += end - start
            cursor = end
    return total


# --------------------------------------------------------------------------------------------------
# Clocks and assignment
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Assignment:
    """For each interval, the index of the host range it belongs to (``None``: none). ``contained``
    intervals lie wholly inside their range; ``partial`` ones only by their midpoint (they cross a
    range boundary); ``unassigned`` ones fall inside no range."""

    index: list[int | None]
    contained: int
    partial: int
    unassigned: int


def assign(
    intervals: Sequence[tuple[float, float]], ranges: Sequence[tuple[float, float]]
) -> Assignment:
    """Map each ``(start, end)`` interval to the shortest host range ``(start, end)`` containing
    it, or else to the shortest range containing its midpoint. Ranges may nest or overlap (the
    search walks back while an earlier range can still reach the point); one clock for both."""
    order = sorted(range(len(ranges)), key=lambda i: ranges[i][0])
    starts = [ranges[i][0] for i in order]
    reach: list[float] = []  # the largest end among ranges[order[0..k]]
    for i in order:
        reach.append(max(ranges[i][1], reach[-1]) if reach else ranges[i][1])

    def best(lo: float, hi: float) -> int | None:
        k = bisect.bisect_right(starts, lo) - 1
        found: int | None = None
        while k >= 0 and reach[k] >= hi:
            start, end = ranges[order[k]]
            if (
                start <= lo
                and end >= hi
                and (found is None or end - start < ranges[found][1] - ranges[found][0])
            ):
                found = order[k]
            k -= 1
        return found

    index: list[int | None] = []
    contained = partial = unassigned = 0
    for start, end in intervals:
        hit = best(start, end)
        if hit is not None:
            contained += 1
        else:
            mid = (start + end) / 2
            hit = best(mid, mid)
            if hit is None:
                unassigned += 1
            else:
                partial += 1
        index.append(hit)
    return Assignment(index=index, contained=contained, partial=partial, unassigned=unassigned)


# --------------------------------------------------------------------------------------------------
# Serialised dispatch
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Dispatch:
    """One serialised dispatch on the :func:`time.monotonic_ns` clock: from just before the op ran
    to just after the device finished it, so its GPU work lies inside; ``what`` is the caller's
    classification (``Serialised.classify(func)``, taken before the op ran; ``None`` without
    one)."""

    start_ns: int
    end_ns: int
    what: Any


class Serialised(TorchDispatchMode):  # type: ignore[no-untyped-call]
    """Every aten op waits for the device (``sync``) before it returns and is recorded as a
    :class:`Dispatch` in :attr:`records`; ``calls`` counts them. ``classify(func)`` (optional) runs
    first, in the op's own context: the Python stack that issued it, the autograd node being
    evaluated (``torch._C._current_autograd_node()``), the autograd sequence number. Compile
    internals are ignored (module docstring), so compiled frames stay compiled under the mode; the
    autograd engine's thread inherits it. With ``enabled`` false an op runs straight through (not
    waited for, recorded or counted). ``on_dispatch`` runs after each recorded op
    (:meth:`ChunkedTrace.tick`)."""

    def __init__(
        self, sync: Callable[[], None], classify: Callable[[Any], Any] | None = None
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.sync = sync
        self.classify = classify
        self.records: list[Dispatch] = []
        self.calls = 0
        self.enabled = True
        self.on_dispatch: Callable[[], None] | None = None

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if not self.enabled:
            return func(*args, **(kwargs or {}))
        what = self.classify(func) if self.classify is not None else None
        start = time.monotonic_ns()
        out = func(*args, **(kwargs or {}))
        self.sync()
        self.records.append(Dispatch(start_ns=start, end_ns=time.monotonic_ns(), what=what))
        self.calls += 1
        if self.on_dispatch is not None:
            self.on_dispatch()
        return out


def mps_sync() -> None:
    """``torch.mps.synchronize`` (the default ``sync`` of :class:`Serialised` on MPS)."""
    torch.mps.synchronize()


# --------------------------------------------------------------------------------------------------
# Chunked recording of a serialised run
# --------------------------------------------------------------------------------------------------


def _marker_work() -> None:
    """A short, recognisable piece of GPU work (one small matmul) for :class:`ChunkedTrace`."""
    x = torch.ones(64, 64, device="mps")
    torch.mm(x, x)


class ChunkedTrace:
    """Metal System Traces of a serialised run, in chunks of at most ``chunk`` dispatches.

    A trace of a whole serialised training step is gigabytes (about 36 KB of kernel trace per
    synchronised dispatch, 1.6e5 dispatches) and takes minutes to save, so the run pauses every
    ``chunk`` dispatches (:meth:`tick`, called after each dispatch, outside its range): the
    current recording is stopped and saved, its GPU intervals of ``pid`` are exported and kept,
    its trace, exports and kernel-trace buffers are deleted (unless ``keep``), and the next
    recording starts before the next dispatch. Pauses fall between dispatches, whose GPU work is
    complete, so serialised GPU intervals do not depend on them.

    Each chunk runs marker work (:func:`_marker_work`, with ``mode`` disabled so it is no dispatch
    of the run) right after its recording goes live and right before it stops; a chunk whose trace
    lacks either marker did not capture its whole span and raises :class:`MetalTraceError`. The
    markers' GPU intervals are left out of :attr:`intervals`."""

    def __init__(
        self,
        workdir: Path,
        pid: int | None = None,
        *,
        chunk: int = 20_000,
        keep: bool = False,
        mode: Serialised | None = None,
        sync: Callable[[], None] = mps_sync,
        marker: Callable[[], None] = _marker_work,
    ) -> None:
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}")
        self.workdir = Path(workdir)
        self.pid = os.getpid() if pid is None else pid
        self.chunk = chunk
        self.keep = keep
        self.mode = mode
        self.sync = sync
        self.marker = marker
        self.intervals: list[GPUInterval] = []
        self.chunks = 0
        self.pause_s = 0.0
        self.ktrace_bytes = 0
        self._count = 0
        self._recording: Recording | None = None
        self._start_marker: tuple[int, int] | None = None
        self._markers: list[tuple[int, int]] = []

    def _mark(self) -> tuple[int, int]:
        enabled = self.mode.enabled if self.mode is not None else False
        if self.mode is not None:
            self.mode.enabled = False
        try:
            self.sync()
            lo = time.monotonic_ns()
            self.marker()
            self.sync()
            hi = time.monotonic_ns()
        finally:
            if self.mode is not None:
                self.mode.enabled = enabled
        self._markers.append((lo, hi))
        return lo, hi

    def _start(self) -> None:
        recording = Recording(self.workdir / f"chunk{self.chunks:03d}.trace", self.pid)
        recording.__enter__()
        self._recording = recording
        self._start_marker = self._mark()

    def _finish(self) -> None:
        recording, start_marker = self._recording, self._start_marker
        if recording is None or start_marker is None:
            return
        end_marker = self._mark()
        self._recording = None
        recording.__exit__(None, None, None)
        self.ktrace_bytes += recording.ktrace_bytes
        exports = self.workdir / f"chunk{self.chunks:03d}_export"
        exports.mkdir(parents=True, exist_ok=True)
        got = gpu_intervals(recording.out, self.pid, exports)
        for name, (lo, hi) in (("start", start_marker), ("end", end_marker)):
            if not any(lo <= g.start_ns <= hi for g in got):
                raise MetalTraceError(
                    f"chunk {self.chunks} of the serialised trace holds no GPU interval of its "
                    f"{name} marker: the recording did not cover the chunk"
                )
        self.intervals.extend(
            g for g in got if not any(lo <= g.start_ns <= hi for lo, hi in self._markers)
        )
        if not self.keep:
            shutil.rmtree(recording.out, ignore_errors=True)
            shutil.rmtree(exports, ignore_errors=True)
        self.chunks += 1

    def tick(self) -> None:
        """One serialised dispatch done; at ``chunk`` dispatches, roll over to a new recording."""
        if self._recording is None:
            return
        self._count += 1
        if self._count >= self.chunk:
            started = time.perf_counter()
            self._finish()
            self._start()
            self._count = 0
            self.pause_s += time.perf_counter() - started

    def close(self) -> None:
        """Finish the last chunk now (idempotent)."""
        started = time.perf_counter()
        self._finish()
        self.pause_s += time.perf_counter() - started

    def __enter__(self) -> ChunkedTrace:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        elif self._recording is not None:
            recording, self._recording = self._recording, None
            recording.__exit__(exc_type, exc, tb)
