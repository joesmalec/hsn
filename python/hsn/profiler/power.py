"""Power meters (P0.6): joules per profiler segment, with the measurement method stated.

Paper §11 requires the meter method in every energy report, so every :class:`EnergyReading`
carries a ``method`` string that matches the plan's §3.8 table:

* :class:`NvmlEnergyMeter` — NVIDIA discrete GPUs through ``pynvml``: the delta of the board's
  total-energy counter (``nvmlDeviceGetTotalEnergyConsumption``, millijoules) is the measurement;
  10 Hz samples of ``NVML_FI_DEV_POWER_INSTANT`` taken by a background thread are the cross-check
  that §3.8 asks for (on GA106 ``nvmlDeviceGetPowerUsage`` is a 1 s average, so the counter is
  what short intervals rely on; some driver/card combinations report junk, so the notes carry the
  sampled estimate to compare with a wall meter). GPU board only: host power is excluded.
* :class:`PowermetricsMeter` — Apple silicon through ``sudo powermetrics --samplers
  gpu_power,cpu_power -i 200``: package power from the SoC counters (CPU, GPU, ANE and their
  combination), each 200 ms window integrated over its overlap with the metered interval. It
  needs root; when ``sudo`` cannot run it without a password, ``start()`` raises
  :class:`PermissionError` with instructions and never returns zero.
* :class:`NullMeter` — no meter (``power_meter: none`` in a hardware profile): 0 J with
  ``method="none"`` so a report always states that energy was not measured.

Every reading also carries ``components`` (joules by component, for example ``gpu_j`` and
``cpu_j``) so GPU-only and whole-package energy can be reported separately (plan §3.8).

``pynvml`` is an optional dependency (the ``cuda`` extra) and is imported lazily; the base install
stays free of GPU-only packages. Nothing here estimates energy from a TDP. The meters resolve
100–200 ms at best, so an energy segment should span seconds (plan P0.6 pitfall: 5 s or more on
GA106 without the counter); the profiler meters only outermost segments for that reason.
"""

from __future__ import annotations

import getpass
import importlib
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

__all__ = [
    "NULL_METHOD",
    "NVML_METHOD",
    "POWERMETRICS_METHOD",
    "EnergyReading",
    "NullMeter",
    "NvmlEnergyMeter",
    "PowerMeter",
    "PowermetricsMeter",
    "PowermetricsSample",
    "parse_powermetrics",
]

#: Method strings, verbatim from plan §3.8 (and the ``power_meter`` profile value for "none").
NVML_METHOD = "NVML total-energy counter plus 10 Hz samples of NVML_FI_DEV_POWER_INSTANT"
POWERMETRICS_METHOD = "sudo powermetrics --samplers gpu_power,cpu_power -i 200"
NULL_METHOD = "none"

_NVML_INTERVAL_MS = 100  # 10 Hz, as §3.8 states
_POWERMETRICS_INTERVAL_MS = 200  # ``-i 200``, as §3.8 states
_POWERMETRICS_SAMPLERS = "gpu_power,cpu_power"
_POWERMETRICS_STARTUP_S = 15.0  # sudo + process start + the first 200 ms window
_POWERMETRICS_EXECUTABLE = "/usr/bin/powermetrics"  # never resolved through PATH: it runs as root
_SUDO = "/usr/bin/sudo"
_SUPERUSER_MESSAGE = "must be invoked as the superuser"


@dataclass(frozen=True, kw_only=True)
class EnergyReading:
    """Energy over one ``start()``/``stop()`` interval and how it was measured."""

    joules: float
    n_samples: int  # power samples taken during the interval
    method: str  # plan §3.8 method string
    interval_ms: int  # sampling interval of the samples (0 when nothing is sampled)
    includes_host: bool  # whether host (CPU) power is part of ``joules``
    notes: str  # caveats and cross-checks, for the report
    components: dict[str, float] = field(default_factory=dict)  # joules by component


class PowerMeter(Protocol):
    """A meter is started before a segment and stopped after it; ``stop`` returns the reading."""

    def start(self) -> None: ...

    def stop(self) -> EnergyReading: ...


# --------------------------------------------------------------------------------------------------
# NullMeter
# --------------------------------------------------------------------------------------------------


class NullMeter:
    """No meter: every reading is 0 J with ``method="none"`` (the profile value for "no meter")."""

    def __init__(self) -> None:
        self._running = False

    def start(self) -> None:
        if self._running:
            raise RuntimeError("NullMeter.start() called while it is already running")
        self._running = True

    def stop(self) -> EnergyReading:
        if not self._running:
            raise RuntimeError("NullMeter.stop() called before start()")
        self._running = False
        return EnergyReading(
            joules=0.0,
            n_samples=0,
            method=NULL_METHOD,
            interval_ms=0,
            includes_host=False,
            notes="no power meter: energy was not measured",
        )


# --------------------------------------------------------------------------------------------------
# NvmlEnergyMeter
# --------------------------------------------------------------------------------------------------

# NVML value-type constant name -> attribute of the ``c_nvmlValue_t`` union holding the value.
_NVML_VALUE_ATTRS = (
    ("NVML_VALUE_TYPE_DOUBLE", "dVal"),
    ("NVML_VALUE_TYPE_UNSIGNED_INT", "uiVal"),
    ("NVML_VALUE_TYPE_UNSIGNED_LONG", "ulVal"),
    ("NVML_VALUE_TYPE_UNSIGNED_LONG_LONG", "ullVal"),
    ("NVML_VALUE_TYPE_SIGNED_LONG_LONG", "sllVal"),
    ("NVML_VALUE_TYPE_SIGNED_INT", "siVal"),
)


def _load_pynvml() -> Any:
    try:
        return importlib.import_module("pynvml")
    except ImportError as exc:
        raise ImportError(
            "NvmlEnergyMeter needs pynvml, an optional dependency of the cuda extra: install it "
            "with `uv sync --extra cuda` (hsn[cuda]); the base install stays free of GPU-only "
            "packages"
        ) from exc


@dataclass
class _NvmlRun:
    handle: Any
    name: str
    energy0_mj: int
    t0: float
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    samples_w: list[float] = field(default_factory=list)
    failed: int = 0
    last_error: str = ""


class NvmlEnergyMeter:
    """Energy of one NVIDIA GPU: total-energy counter delta plus 10 Hz instant-power samples.

    ``device_index`` is the NVML index (the board's position for the driver), which differs from
    the CUDA ordinal when ``CUDA_VISIBLE_DEVICES`` reorders devices. ``start()`` initialises NVML,
    reads the energy counter and starts the sampling thread; ``stop()`` stops the thread, reads
    the counter again and shuts NVML down. A device whose driver does not expose the energy
    counter raises at ``start()`` (energy would otherwise be a sampled estimate that needs
    intervals of 5 s or more on GA106); that card needs a wall meter.
    """

    interval_ms = _NVML_INTERVAL_MS

    def __init__(self, device_index: int = 0) -> None:
        if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
            raise ValueError(f"device_index must be a non-negative int, got {device_index!r}")
        self._nvml = _load_pynvml()
        self._index = device_index
        self._run: _NvmlRun | None = None

    def start(self) -> None:
        if self._run is not None:
            raise RuntimeError("NvmlEnergyMeter.start() called while it is already running")
        nvml = self._nvml
        try:
            nvml.nvmlInit()
        except nvml.NVMLError as exc:
            raise RuntimeError(
                f"NVML initialisation failed ({exc}); is the NVIDIA driver loaded?"
            ) from exc
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(self._index)
            name = _decode(nvml.nvmlDeviceGetName(handle))
        except nvml.NVMLError as exc:
            nvml.nvmlShutdown()
            raise RuntimeError(f"NVML cannot open GPU {self._index}: {exc}") from exc
        try:
            energy0 = int(nvml.nvmlDeviceGetTotalEnergyConsumption(handle))
        except nvml.NVMLError as exc:
            nvml.nvmlShutdown()
            raise RuntimeError(
                f"the NVML energy counter is unsupported on {name} ({exc}); its instant-power "
                "samples alone would need segments of 5 s or more, so energy on this card must "
                "come from a wall meter (plan §3.8)"
            ) from exc
        run = _NvmlRun(handle=handle, name=name, energy0_mj=energy0, t0=time.perf_counter())
        run.thread = threading.Thread(
            target=self._sample, args=(run,), name="hsn-nvml-sampler", daemon=True
        )
        self._run = run
        run.thread.start()

    def _sample(self, run: _NvmlRun) -> None:
        nvml = self._nvml
        field_ids = [nvml.NVML_FI_DEV_POWER_INSTANT]
        while not run.stop_event.wait(self.interval_ms / 1000.0):
            try:
                (value,) = nvml.nvmlDeviceGetFieldValues(run.handle, field_ids)
                if int(value.nvmlReturn) != int(nvml.NVML_SUCCESS):
                    raise RuntimeError(f"nvmlReturn={int(value.nvmlReturn)}")
                watts = _nvml_field_value(nvml, value) / 1000.0  # mW -> W
            except Exception as exc:  # the thread outlives any sampling error; stop() reports it
                run.failed += 1
                run.last_error = f"{type(exc).__name__}: {exc}"
                continue
            run.samples_w.append(watts)

    def stop(self) -> EnergyReading:
        run = self._run
        if run is None:
            raise RuntimeError("NvmlEnergyMeter.stop() called before start()")
        self._run = None
        run.stop_event.set()
        stuck = ""
        if run.thread is not None:
            run.thread.join(timeout=5.0)
            if run.thread.is_alive():
                stuck = "; the sampler thread did not stop within 5 s (stuck in an NVML call)"
        nvml = self._nvml
        try:
            energy1 = int(nvml.nvmlDeviceGetTotalEnergyConsumption(run.handle))
        except nvml.NVMLError as exc:
            raise RuntimeError(
                f"reading the NVML energy counter on {run.name} failed: {exc}"
            ) from exc
        finally:
            nvml.nvmlShutdown()
        elapsed_s = time.perf_counter() - run.t0
        joules = (energy1 - run.energy0_mj) / 1000.0
        if joules < 0:
            raise RuntimeError(
                f"the NVML energy counter on {run.name} went backwards ({run.energy0_mj} -> "
                f"{energy1} mJ); the reading is invalid"
            )
        components = {"gpu_j": joules}
        n = len(run.samples_w)
        if n:
            mean_w = statistics.fmean(run.samples_w)
            cross_check = (
                f"cross-check from {n} samples: mean {mean_w:.1f} W (min "
                f"{min(run.samples_w):.1f} W, max {max(run.samples_w):.1f} W) x {elapsed_s:.3f} s "
                f"= {mean_w * elapsed_s:.3f} J"
            )
        else:
            cross_check = f"no power samples (interval shorter than {self.interval_ms} ms)"
        failures = f"{run.failed} failed samples"
        if run.failed:
            failures += f" (last: {run.last_error})"
        resolution = ""
        if joules == 0.0:
            resolution = (
                "; the counter did not advance: the interval is shorter than its update period, "
                "so energy is not resolved at this length"
            )
        notes = (
            f"{run.name}: energy counter delta {joules:.3f} J over {elapsed_s:.3f} s{resolution}; "
            f"{cross_check}; {failures}{stuck}; GPU board only, host excluded; on GA106 "
            "nvmlDeviceGetPowerUsage averages over 1 s and some driver/card pairs report junk: "
            "validate against a wall meter"
        )
        return EnergyReading(
            joules=joules,
            n_samples=n,
            method=NVML_METHOD,
            interval_ms=self.interval_ms,
            includes_host=False,
            notes=notes,
            components=components,
        )


def _nvml_field_value(nvml: Any, value: Any) -> float:
    value_type = int(value.valueType)
    for constant, attr in _NVML_VALUE_ATTRS:
        if value_type == int(getattr(nvml, constant)):
            return float(getattr(value.value, attr))
    raise RuntimeError(f"NVML field value type {value_type} is not understood")


def _decode(name: object) -> str:
    return name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)


# --------------------------------------------------------------------------------------------------
# PowermetricsMeter
# --------------------------------------------------------------------------------------------------

_HEADER_RE = r"^\*\*\* Sampled system activity .*?\((\d+(?:\.\d+)?)ms elapsed\) \*\*\*"
_HEADER = re.compile(_HEADER_RE, re.MULTILINE)
_HEADER_LINE = re.compile(_HEADER_RE)
_CPU_POWER = re.compile(r"^CPU Power:\s*(\d+(?:\.\d+)?)\s*mW", re.MULTILINE)
_GPU_POWER = re.compile(r"^GPU Power:\s*(\d+(?:\.\d+)?)\s*mW", re.MULTILINE)
_ANE_POWER = re.compile(r"^ANE Power:\s*(\d+(?:\.\d+)?)\s*mW", re.MULTILINE)
_COMBINED_RE = r"^Combined Power \(CPU \+ GPU \+ ANE\):\s*(\d+(?:\.\d+)?)\s*mW"
_COMBINED_POWER = re.compile(_COMBINED_RE, re.MULTILINE)
_COMBINED_LINE = re.compile(_COMBINED_RE)


@dataclass(frozen=True, kw_only=True)
class PowermetricsSample:
    """One ``powermetrics`` sample block: average power over the ``elapsed_ms`` window."""

    index: int  # ordinal of the block's header in the output (truncated blocks keep their slot)
    elapsed_ms: float
    cpu_mw: float
    gpu_mw: float
    ane_mw: float
    combined_mw: float


def parse_powermetrics(text: str) -> list[PowermetricsSample]:
    """Samples from ``powermetrics --samplers gpu_power,cpu_power`` text output.

    Blocks that lack the CPU or GPU power line (the block cut short when the process is stopped)
    are left out; ``ANE Power`` defaults to 0 and a missing ``Combined Power`` line is the sum.
    """
    headers = list(_HEADER.finditer(text))
    samples: list[PowermetricsSample] = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        block = text[header.end() : end]
        cpu = _CPU_POWER.search(block)
        gpu = _GPU_POWER.search(block)
        if cpu is None or gpu is None:
            continue
        ane = _ANE_POWER.search(block)
        combined = _COMBINED_POWER.search(block)
        cpu_mw, gpu_mw = float(cpu.group(1)), float(gpu.group(1))
        ane_mw = float(ane.group(1)) if ane is not None else 0.0
        combined_mw = float(combined.group(1)) if combined is not None else cpu_mw + gpu_mw + ane_mw
        samples.append(
            PowermetricsSample(
                index=index,
                elapsed_ms=float(header.group(1)),
                cpu_mw=cpu_mw,
                gpu_mw=gpu_mw,
                ane_mw=ane_mw,
                combined_mw=combined_mw,
            )
        )
    return samples


def _powermetrics_command(executable: str, interval_ms: int) -> list[str]:
    # ``-b 1`` line-buffers the output so each sample arrives as it is taken; otherwise stdio
    # buffers whole blocks when stdout is a pipe.
    return [executable, "--samplers", _POWERMETRICS_SAMPLERS, "-i", str(interval_ms), "-b", "1"]


def _sudo_instructions(executable: str, interval_ms: int) -> str:
    user = getpass.getuser()
    pinned = " ".join(_powermetrics_command(executable, interval_ms)).replace(",", "\\,")
    return (
        "powermetrics must run as root and sudo cannot run it without a password here. Either "
        "run `sudo -v` in the terminal that launches this process (credentials are cached per "
        "terminal for a few minutes), or allow exactly this command without a password: `sudo "
        f"visudo -f /etc/sudoers.d/powermetrics` with the line `{user} ALL=(root) NOPASSWD: "
        f"{pinned}`. Energy is not measured otherwise; profile with NullMeter to skip energy "
        "explicitly."
    )


@dataclass
class _PowermetricsRun:
    process: subprocess.Popen[str]
    executable: str
    lock: threading.Condition = field(default_factory=threading.Condition)
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)
    header_times: list[float] = field(default_factory=list)  # perf_counter at each header
    ready: list[bool] = field(default_factory=list)  # per block: its power lines have arrived
    threads: list[threading.Thread] = field(default_factory=list)
    t_start: float = 0.0

    def closed_by(self, t: float) -> bool:
        """Whether a complete block whose window ends at or after ``t`` has arrived."""
        return any(h >= t and r for h, r in zip(self.header_times, self.ready, strict=True))


class PowermetricsMeter:
    """Apple-silicon package energy from ``sudo powermetrics --samplers gpu_power,cpu_power``.

    ``start()`` spawns the sampler and waits for its first sample so the metered interval is
    covered from its start; ``stop()`` waits for the sample that closes the interval, stops the
    process and integrates each 200 ms window over its overlap with the interval (a window's
    power is its average, so intervals shorter than a window take that average). ``scope`` picks
    what ``joules`` reports: ``"combined"`` (CPU + GPU + ANE, host included) or ``"gpu"``; the
    components carry all four so GPU-only and whole-package energy are both in every report
    (plan §3.8). When ``sudo -n`` cannot run powermetrics, ``start()`` raises
    :class:`PermissionError` with instructions; when no sample covers the interval, ``stop()``
    raises instead of reporting 0 J.
    """

    interval_ms = _POWERMETRICS_INTERVAL_MS

    def __init__(self, scope: Literal["combined", "gpu"] = "combined") -> None:
        if scope not in ("combined", "gpu"):
            raise ValueError(f"scope must be 'combined' or 'gpu', got {scope!r}")
        self._scope: Literal["combined", "gpu"] = scope
        self._run: _PowermetricsRun | None = None

    @property
    def scope(self) -> str:
        return self._scope

    def start(self) -> None:
        if self._run is not None:
            raise RuntimeError("PowermetricsMeter.start() called while it is already running")
        if sys.platform != "darwin":
            raise RuntimeError(
                f"powermetrics is a macOS tool; this is {sys.platform}. Use NvmlEnergyMeter on "
                "NVIDIA GPUs or NullMeter to profile without energy"
            )
        executable = shutil.which(_POWERMETRICS_EXECUTABLE)
        if executable is None:
            raise RuntimeError(f"{_POWERMETRICS_EXECUTABLE} is missing or not executable")
        if shutil.which(_SUDO) is None:
            raise RuntimeError(f"{_SUDO} is missing; powermetrics needs root")
        command = [_SUDO, "-n", *_powermetrics_command(executable, self.interval_ms)]
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        run = _PowermetricsRun(process=process, executable=executable)
        if process.stdout is None or process.stderr is None:  # pragma: no cover - PIPE above
            raise RuntimeError("powermetrics pipes were not created")
        for target, stream in ((_read_samples, process.stdout), (_read_lines, process.stderr)):
            thread = threading.Thread(
                target=target, args=(run, stream), name="hsn-powermetrics-reader", daemon=True
            )
            thread.start()
            run.threads.append(thread)
        # wait for the first complete sample: sudo + process start-up + one window, all outside
        # the interval, so the windows that follow cover it from its start
        if not _wait_for(run, lambda: bool(run.ready) and run.ready[0], _POWERMETRICS_STARTUP_S):
            _finish(run)
            stdout, stderr = "".join(run.stdout), "".join(run.stderr)
            _raise_refusal(stdout, stderr, executable, self.interval_ms)
            raise RuntimeError(
                f"powermetrics produced no sample within {_POWERMETRICS_STARTUP_S:.0f} s (exit "
                f"code {process.returncode}; stderr: {stderr.strip()[:300]!r}); energy cannot be "
                "measured"
            )
        run.t_start = time.perf_counter()
        self._run = run

    def stop(self) -> EnergyReading:
        run = self._run
        if run is None:
            raise RuntimeError("PowermetricsMeter.stop() called before start()")
        self._run = None
        t_stop = time.perf_counter()
        # wait for the complete block whose window closes after the interval (its power lines
        # follow its header by a few ms), so the interval is fully covered before SIGTERM
        _wait_for(run, lambda: run.closed_by(t_stop), 2 * self.interval_ms / 1000.0 + 1.0)
        _finish(run)
        return self._reading(
            "".join(run.stdout),
            "".join(run.stderr),
            list(run.header_times),
            run.t_start,
            t_stop,
            run.process.returncode,
            run.executable,
        )

    def _reading(
        self,
        stdout: str,
        stderr: str,
        header_times: list[float],
        t_start: float,
        t_stop: float,
        returncode: int | None,
        executable: str = "/usr/bin/powermetrics",
    ) -> EnergyReading:
        _raise_refusal(stdout, stderr, executable, self.interval_ms)
        samples = parse_powermetrics(stdout)
        wall_s = t_stop - t_start
        totals = {"cpu_j": 0.0, "gpu_j": 0.0, "ane_j": 0.0, "combined_j": 0.0}
        covered_s = 0.0
        n_samples = 0
        for sample in samples:
            if sample.index >= len(header_times):
                continue
            t_end = header_times[sample.index]
            t_begin = t_end - sample.elapsed_ms / 1000.0
            overlap = min(t_end, t_stop) - max(t_begin, t_start)
            if overlap <= 0.0:
                continue
            n_samples += 1
            covered_s += overlap
            totals["cpu_j"] += sample.cpu_mw / 1000.0 * overlap
            totals["gpu_j"] += sample.gpu_mw / 1000.0 * overlap
            totals["ane_j"] += sample.ane_mw / 1000.0 * overlap
            totals["combined_j"] += sample.combined_mw / 1000.0 * overlap
        if n_samples == 0:
            n_blocks = len(_HEADER.findall(stdout))
            raise RuntimeError(
                f"no powermetrics sample covers the {wall_s:.3f} s interval ({len(samples)} "
                f"complete samples, {n_blocks} blocks, exit code {returncode}; stderr: "
                f"{stderr.strip()[:300]!r}); energy was not measured"
            )
        coverage = covered_s / wall_s if wall_s > 0 else 0.0
        shortfall = wall_s - covered_s
        if shortfall > max(0.02, 0.05 * wall_s):  # windows tile the interval up to ms jitter
            raise RuntimeError(
                f"powermetrics windows cover only {covered_s:.3f} s of the {wall_s:.3f} s "
                f"interval ({coverage:.0%}; {n_samples} windows, exit code {returncode}); the "
                "sampler stalled or stopped early, so energy was not measured"
            )
        joules = totals["combined_j"] if self._scope == "combined" else totals["gpu_j"]
        notes = (
            f"{self._scope} scope; cpu {totals['cpu_j']:.3f} J, gpu {totals['gpu_j']:.3f} J, ane "
            f"{totals['ane_j']:.3f} J, combined {totals['combined_j']:.3f} J; {n_samples} windows "
            f"of {self.interval_ms} ms covering {covered_s:.3f} s of {wall_s:.3f} s ({coverage:.0%}"
            "); a window's power is its average, so sub-window intervals take that average; SoC "
            "package counters (CPU + GPU + ANE), no wall meter, DRAM and the rest of the system "
            "excluded"
        )
        return EnergyReading(
            joules=joules,
            n_samples=n_samples,
            method=POWERMETRICS_METHOD,
            interval_ms=self.interval_ms,
            includes_host=self._scope == "combined",
            notes=notes,
            components=totals,
        )


def _read_samples(run: _PowermetricsRun, stream: Any) -> None:
    # A block is complete once its "Combined Power" line (the last power line of the cpu_power
    # section) has arrived, or once the next header shows nothing more will be written to it.
    for line in stream:
        with run.lock:
            run.stdout.append(line)
            if _HEADER_LINE.match(line):
                if run.ready:
                    run.ready[-1] = True
                run.header_times.append(time.perf_counter())
                run.ready.append(False)
                run.lock.notify_all()
            elif run.ready and not run.ready[-1] and _COMBINED_LINE.match(line):
                run.ready[-1] = True
                run.lock.notify_all()
    stream.close()
    with run.lock:
        if run.ready:
            run.ready[-1] = True  # EOF: the parser drops the block if it was cut short
        run.lock.notify_all()


def _read_lines(run: _PowermetricsRun, stream: Any) -> None:
    for line in stream:
        with run.lock:
            run.stderr.append(line)
    stream.close()


def _wait_for(run: _PowermetricsRun, ready: Any, timeout_s: float) -> bool:
    deadline = time.perf_counter() + timeout_s
    with run.lock:
        while not ready():
            if run.process.poll() is not None:
                return bool(ready())
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                return False
            run.lock.wait(min(remaining, 0.1))
        return True


def _finish(run: _PowermetricsRun) -> None:
    process = run.process
    if process.poll() is None:
        process.terminate()  # sudo relays SIGTERM to powermetrics, which exits cleanly
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for thread in run.threads:
        thread.join(timeout=5.0)  # a root child holding the pipe open must not hang us


def _raise_refusal(stdout: str, stderr: str, executable: str, interval_ms: int) -> None:
    refused = _SUPERUSER_MESSAGE in stdout or _SUPERUSER_MESSAGE in stderr
    if refused or "password" in stderr or "sudo:" in stderr:
        detail = stderr.strip() or stdout.strip()
        raise PermissionError(
            f"`sudo -n {executable}` was refused ({detail[:200]}). "
            f"{_sudo_instructions(executable, interval_ms)}"
        )
