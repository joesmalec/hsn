"""P0.6 tests: profiler segments, report schema, power meters (mocked) and launch counting."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import load_hardware_profile
from hsn.device import DeviceContext, resolve, synchronize
from hsn.profiler import (
    NVML_METHOD,
    POWERMETRICS_METHOD,
    SEGMENT_NAMES,
    EnergyReading,
    NullMeter,
    NvmlEnergyMeter,
    PowermetricsMeter,
    Profiler,
    count_launches,
)
from hsn.profiler.launches import CUDA_METHOD, UNSUPPORTED
from hsn.profiler.power import parse_powermetrics

SEGMENT_FIELDS: dict[str, tuple[type, ...]] = {
    "n": (int,),
    "wall_ms_p50": (float,),
    "wall_ms_p95": (float,),
    "device_ms": (float,),
    "peak_bytes": (int,),
    "reserved_bytes": (int,),
    "launches": (int,),
    "energy_j": (float, type(None)),
    "energy_components": (dict,),
    "tokens": (int,),
}
TOP_LEVEL_FIELDS = {"profile", "device", "timing_method", "launches_method", "power", "segments"}
POWER_FIELDS = {
    "method",
    "interval_ms",
    "includes_host",
    "n_readings",
    "n_samples",
    "n_unresolved",
    "notes",
}
MIB = 1024 * 1024


def _cpu_ctx() -> DeviceContext:
    return resolve(load_hardware_profile("cpu"))


def _cuda_ctx() -> DeviceContext:
    if not torch.cuda.is_available():
        pytest.skip("no cuda device on this machine")
    try:
        return resolve(load_hardware_profile("rtx3060"))
    except (
        ValueError
    ):  # a pre-Ampere card cannot honour the bf16 policy; counting needs no autocast
        device = torch.device("cuda", torch.cuda.current_device())
        return DeviceContext(
            device=device,
            autocast_dtype=None,
            master_dtype=torch.float32,
            supports_bf16=False,
            unified_memory=False,
            vram_bytes=int(torch.cuda.mem_get_info(device)[1]),
            profile="cuda",
        )


# --------------------------------------------------------------------------------------------------
# segments
# --------------------------------------------------------------------------------------------------


def test_p0_6_segments_nest_and_sum() -> None:
    prof = Profiler(_cpu_ctx())
    x = torch.randn(64, 64)
    for _ in range(3):
        with prof.segment("shared_path", tokens=16):
            time.sleep(0.005)
            with prof.segment("scorer", tokens=16):
                time.sleep(0.004)
                y = x @ x
            with prof.segment("selected_cells"):
                time.sleep(0.003)
                y = y @ x
    report = prof.report()
    assert list(report.segments) == ["shared_path", "scorer", "selected_cells"]
    outer, scorer, cells = (report.segments[n] for n in ("shared_path", "scorer", "selected_cells"))
    assert outer.n == scorer.n == cells.n == 3
    assert (outer.tokens, scorer.tokens, cells.tokens) == (48, 48, 0)
    # the sleeps are lower bounds on the wall time of every entry
    assert outer.wall_ms_p50 >= 12.0 and scorer.wall_ms_p50 >= 4.0 and cells.wall_ms_p50 >= 3.0
    assert outer.wall_ms_p95 >= outer.wall_ms_p50
    # nesting is inclusive: the parent's time covers its children (device_ms sums the entries)
    assert outer.device_ms >= scorer.device_ms + cells.device_ms
    assert outer.device_ms >= 3 * 12.0
    assert outer.peak_bytes >= max(scorer.peak_bytes, cells.peak_bytes) > 0
    assert outer.reserved_bytes > 0
    assert outer.launches == scorer.launches == cells.launches == UNSUPPORTED  # not counted
    assert outer.energy_j is None  # NullMeter: nothing measured, never a zero
    assert scorer.energy_j is None and cells.energy_j is None  # nested: not metered
    assert outer.energy_components == scorer.energy_components == {}
    assert report.timing_method == "synchronise + wall clock"
    assert report.launches_method == "disabled"
    assert report.power.n_readings == report.power.n_unresolved == 3

    # names are fixed project-wide, a segment cannot contain itself, tokens are counts
    assert "scorer" in SEGMENT_NAMES
    with pytest.raises(ValueError, match="scorer"):
        with prof.segment("router"):
            pass
    with prof.segment("decode"):
        with pytest.raises(ValueError, match="already open"):
            with prof.segment("decode"):
                pass
        with pytest.raises(RuntimeError, match="open"):
            prof.report()
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="tokens"):
            with prof.segment("decode", tokens=bad):  # type: ignore[arg-type]
                pass
    # compilation and warm-up never nest with steady-state segments, in either direction
    for standalone in ("compile", "startup"):
        with prof.segment("shared_path"):
            with pytest.raises(ValueError, match="warm-up"):
                with prof.segment(standalone):
                    pass
        with prof.segment(standalone):
            with pytest.raises(ValueError, match="warm-up"):
                with prof.segment("scorer"):
                    pass
    assert prof.report().segments["compile"].n == 1
    # an exception inside a segment records nothing for it and leaves the profiler usable
    with pytest.raises(KeyError, match="boom"):
        with prof.segment("backward"):
            raise KeyError("boom")
    assert "backward" not in prof.report().segments
    with prof.segment("backward"):
        pass
    assert prof.report().segments["backward"].n == 1
    assert prof.report().segments["decode"].n == 1


class _FlakyMeter:
    """A meter whose ``start()`` or ``stop()`` fails on demand, to test the profiler's cleanup."""

    def __init__(self) -> None:
        self.fail_start = False
        self.fail_stop = False
        self.running = False
        self.starts = 0

    def start(self) -> None:
        assert not self.running
        if self.fail_start:
            raise RuntimeError("no power meter attached")
        self.running = True
        self.starts += 1

    def stop(self) -> EnergyReading:
        assert self.running
        self.running = False
        if self.fail_stop:
            raise RuntimeError("meter lost the device")
        return EnergyReading(
            joules=1.5, n_samples=3, method="fake", interval_ms=100, includes_host=False, notes=""
        )


def test_p0_6_cleanup_errors_keep_profiler_usable() -> None:
    meter = _FlakyMeter()
    prof = Profiler(_cpu_ctx(), power=meter)
    meter.fail_stop = True
    with pytest.raises(RuntimeError, match="lost the device"):
        with prof.segment("shared_path"):
            with prof.segment("scorer"):
                pass
    # the failed outermost entry is not recorded, the stack is empty, the meter is stopped
    assert not meter.running
    report = prof.report()
    assert "shared_path" not in report.segments
    assert report.segments["scorer"].n == 1 and report.segments["scorer"].energy_j is None
    # the profiler keeps working once the meter does
    meter.fail_stop = False
    with prof.segment("shared_path", tokens=3):
        pass
    report = prof.report()
    assert report.segments["shared_path"].energy_j == 1.5
    assert report.power.method == "fake" and report.power.n_readings == 1
    assert meter.starts == 2
    # the body's own exception stays the one callers see; the cleanup failure is a note
    meter.fail_stop = True
    with pytest.raises(KeyError, match="body") as info:
        with prof.segment("shared_path"):
            raise KeyError("body")
    assert any("cleanup also failed" in note for note in info.value.__notes__)
    assert not meter.running
    # a meter that cannot start fails the entry before anything is recorded
    meter.fail_stop = False
    meter.fail_start = True
    with pytest.raises(RuntimeError, match="no power meter"):
        with prof.segment("decode"):
            pass
    assert "decode" not in prof.report().segments
    meter.fail_start = False
    with prof.segment("decode"):
        pass
    assert prof.report().segments["decode"].energy_j == 1.5
    # a segment that is sometimes nested and sometimes outermost has no single energy
    with prof.segment("shared_path"):
        with prof.segment("decode"):
            pass
    assert prof.report().segments["decode"].n == 2
    assert prof.report().segments["decode"].energy_j is None


@pytest.mark.mps
def test_p0_6_mps_segments() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("no mps device on this machine")
    ctx = resolve(load_hardware_profile("m_series"))
    prof = Profiler(ctx, count_launches=True)
    x = torch.randn(256, 256, device=ctx.device)
    with prof.segment("startup"):
        x @ x  # first-call set-up stays out of the steady-state segment
    with prof.segment("shared_path", tokens=256):
        block = torch.ones(64 * MIB, dtype=torch.uint8, device=ctx.device)
        for _ in range(4):
            x = x @ x * 1e-3
        synchronize(ctx.device)
    del block
    report = prof.report()
    assert report.device == "mps" and report.timing_method == "synchronise + wall clock"
    assert report.launches_method == "unsupported on mps"
    steady = report.segments["shared_path"]
    assert steady.launches == UNSUPPORTED and steady.tokens == 256
    assert steady.device_ms > 0 and steady.wall_ms_p50 > 0
    assert steady.peak_bytes >= 64 * MIB  # the block was alive inside the segment
    assert steady.reserved_bytes > 0


# --------------------------------------------------------------------------------------------------
# meters
# --------------------------------------------------------------------------------------------------


def test_p0_6_null_meter() -> None:
    meter = NullMeter()
    with pytest.raises(RuntimeError, match="before start"):
        meter.stop()
    meter.start()
    with pytest.raises(RuntimeError, match="already running"):
        meter.start()
    reading = meter.stop()
    assert isinstance(reading, EnergyReading)
    assert reading.joules == 0.0 and reading.n_samples == 0
    assert reading.method == "none"  # the `power_meter: none` profile value
    assert reading.interval_ms == 0 and reading.includes_host is False
    assert reading.components == {} and "not measured" in reading.notes
    # the profiler without a meter uses the same explicit "none"
    for prof in (Profiler(_cpu_ctx()), Profiler(_cpu_ctx(), power=NullMeter())):
        assert prof.report().power.method is None  # nothing read yet
        with prof.segment("decode", tokens=4):
            pass
        report = prof.report()
        assert report.segments["decode"].energy_j is None  # not a measured zero
        assert report.power.method == "none" and report.power.includes_host is False
        assert report.power.n_readings == 1 and report.power.n_samples == 0
        assert report.power.n_unresolved == 1


def _fake_pynvml(
    energy_mj: list[int], *, power_mw: int = 150_000, counter_supported: bool = True
) -> types.ModuleType:
    """A stand-in for pynvml 13: the names NvmlEnergyMeter uses, no driver."""
    module = types.ModuleType("pynvml")
    state = {"init": 0, "shutdown": 0, "energy_calls": 0, "field_calls": 0}

    class NVMLError(Exception):
        pass

    class NVMLError_NotSupported(NVMLError):  # noqa: N801 - pynvml's own name
        pass

    def nvml_init() -> None:
        state["init"] += 1

    def nvml_shutdown() -> None:
        state["shutdown"] += 1

    def handle_by_index(index: int) -> str:
        assert index == 0
        return "handle-0"

    def device_name(handle: str) -> bytes:
        return b"Fake GA106"

    def total_energy(handle: str) -> int:
        if not counter_supported:
            raise NVMLError_NotSupported("Not Supported")
        value = energy_mj[min(state["energy_calls"], len(energy_mj) - 1)]
        state["energy_calls"] += 1
        return value

    def field_values(handle: str, field_ids: list[int]) -> list[Any]:
        assert field_ids == [186]
        state["field_calls"] += 1
        value = types.SimpleNamespace(uiVal=power_mw)
        return [types.SimpleNamespace(fieldId=186, valueType=1, nvmlReturn=0, value=value)]

    module.__dict__.update(
        NVMLError=NVMLError,
        NVMLError_NotSupported=NVMLError_NotSupported,
        nvmlInit=nvml_init,
        nvmlShutdown=nvml_shutdown,
        nvmlDeviceGetHandleByIndex=handle_by_index,
        nvmlDeviceGetName=device_name,
        nvmlDeviceGetTotalEnergyConsumption=total_energy,
        nvmlDeviceGetFieldValues=field_values,
        NVML_FI_DEV_POWER_INSTANT=186,
        NVML_SUCCESS=0,
        NVML_VALUE_TYPE_DOUBLE=0,
        NVML_VALUE_TYPE_UNSIGNED_INT=1,
        NVML_VALUE_TYPE_UNSIGNED_LONG=2,
        NVML_VALUE_TYPE_UNSIGNED_LONG_LONG=3,
        NVML_VALUE_TYPE_SIGNED_LONG_LONG=4,
        NVML_VALUE_TYPE_SIGNED_INT=5,
        state=state,
    )
    return module


def _sampler_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "hsn-nvml-sampler" and t.is_alive()]


def test_p0_6_nvml_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_pynvml([1_000, 6_000])
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    meter = NvmlEnergyMeter()
    with pytest.raises(RuntimeError, match="before start"):
        meter.stop()
    meter.start()
    with pytest.raises(RuntimeError, match="already running"):
        meter.start()
    assert len(_sampler_threads()) == 1
    time.sleep(0.35)
    reading = meter.stop()
    assert reading.joules == 5.0  # (6000 - 1000) mJ: the counter delta is the measurement
    assert reading.n_samples >= 1 and reading.n_samples == fake.state["field_calls"]
    assert reading.method == NVML_METHOD
    assert reading.method == (
        "NVML total-energy counter plus 10 Hz samples of NVML_FI_DEV_POWER_INSTANT"
    )
    assert reading.interval_ms == 100 and reading.includes_host is False
    assert reading.components == {"gpu_j": 5.0}  # the sampled cross-check stays in the notes
    assert "Fake GA106" in reading.notes and "mean 150.0 W" in reading.notes
    assert "wall meter" in reading.notes  # the GA106 caveat is stated
    assert fake.state["init"] == fake.state["shutdown"] == 1
    assert _sampler_threads() == []

    # through the profiler: the outermost segment is metered, nested rows are not
    fake = _fake_pynvml([0, 5_000])
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    prof = Profiler(_cpu_ctx(), power=NvmlEnergyMeter())
    with prof.segment("shared_path", tokens=2):
        with prof.segment("scorer", tokens=2):
            pass
    report = prof.report()
    assert report.segments["shared_path"].energy_j == 5.0
    assert report.segments["shared_path"].energy_components == {"gpu_j": 5.0}
    assert report.segments["scorer"].energy_j is None
    assert report.power.method == NVML_METHOD and report.power.includes_host is False
    assert report.power.n_readings == 1 and report.power.interval_ms == 100
    assert report.power.n_unresolved == 0
    assert fake.state["init"] == fake.state["shutdown"] == 1
    assert _sampler_threads() == []

    # a counter that did not advance is flagged and never summed as 0 J
    fake = _fake_pynvml([7_000, 7_000])
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    still = NvmlEnergyMeter()
    still.start()
    reading = still.stop()
    assert reading.joules == 0.0 and "did not advance" in reading.notes
    prof = Profiler(_cpu_ctx(), power=NvmlEnergyMeter())
    with prof.segment("decode"):
        pass
    report = prof.report()
    assert report.segments["decode"].energy_j is None
    assert report.power.n_readings == 1 and report.power.n_unresolved == 1

    # a card without the energy counter raises at start(): never a sampled-only estimate
    fake = _fake_pynvml([0], counter_supported=False)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    with pytest.raises(RuntimeError, match="energy counter"):
        NvmlEnergyMeter().start()
    assert fake.state["init"] == fake.state["shutdown"] == 1
    with pytest.raises(ValueError, match="device_index"):
        NvmlEnergyMeter(device_index=-1)
    # pynvml is optional: a missing module names the extra instead of failing later
    monkeypatch.setitem(sys.modules, "pynvml", None)
    with pytest.raises(ImportError, match="cuda"):
        NvmlEnergyMeter()


POWERMETRICS_TEXT = """\
Machine model: Mac16,8
OS version: 25F84
Boot arguments:
Boot time: Mon Nov  3 08:00:00 2026



*** Sampled system activity (Mon Nov  3 19:12:49 2026 -0800) (203.45ms elapsed) ***


**** Processor usage ****

E-Cluster Power: 23 mW
E-Cluster HW active frequency: 1044 MHz
E-Cluster HW active residency:  28.83% (600 MHz:  12% 912 MHz:  10% 1044 MHz: 6.8%)
E-Cluster idle residency:  71.17%
P0-Cluster Power: 15 mW
P0-Cluster HW active frequency: 1398 MHz
CPU 0 frequency: 1044 MHz
CPU 0 active residency:  12.34%
CPU Power: 1250 mW
GPU Power: 3400 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 4650 mW

**** GPU usage ****

GPU HW active frequency: 1398 MHz
GPU HW active residency:  87.65% (444 MHz:   0% 1398 MHz:  88%)
GPU SW requested state: (P1 :   0% P2 :   0% P3 :  88%)
GPU idle residency:  12.35%
GPU Power: 3400 mW

*** Sampled system activity (Mon Nov  3 19:12:49 2026 -0800) (200.12ms elapsed) ***


**** Processor usage ****

CPU Power: 750 mW
GPU Power: 1600 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 2350 mW

**** GPU usage ****

GPU HW active frequency: 1200 MHz
GPU idle residency:  40.00%
GPU Power: 1600 mW

*** Sampled system activity (Mon Nov  3 19:12:50 2026 -0800) (20.00ms elapsed) ***


**** Processor usage ****

E-Cluster Power: 20 mW
"""


def test_p0_6_powermetrics_parser() -> None:
    samples = parse_powermetrics(POWERMETRICS_TEXT)
    assert len(samples) == 2  # the block cut short when the process stops is left out
    first, second = samples
    assert (first.index, first.elapsed_ms, first.cpu_mw, first.gpu_mw) == (0, 203.45, 1250, 3400)
    assert first.ane_mw == 0 and first.combined_mw == 4650
    assert (second.index, second.elapsed_ms, second.combined_mw) == (1, 200.12, 2350)
    assert parse_powermetrics("") == []
    assert parse_powermetrics("GPU Power: 5 mW\n") == []  # no sample header: no sample
    # a block without a combined line sums the components
    without_total = (
        "*** Sampled system activity (x) (100ms elapsed) ***\nCPU Power: 1 mW\nGPU Power: 2 mW\n"
    )
    (only,) = parse_powermetrics(without_total)
    assert (only.cpu_mw, only.gpu_mw, only.ane_mw, only.combined_mw) == (1, 2, 0, 3)

    # the reading integrates each window over its overlap with the metered interval:
    # headers arrived at 10.0 s and 10.20345 s, so the windows are [9.79655, 10.0] and
    # [10.00333, 10.20345]; the third block is truncated and covers nothing
    header_times = [10.0, 10.20345, 10.40357]
    t_start, t_stop = 9.9, 10.2
    combined = PowermetricsMeter()._reading(POWERMETRICS_TEXT, "", header_times, t_start, t_stop, 0)
    gpu = PowermetricsMeter(scope="gpu")._reading(
        POWERMETRICS_TEXT, "", header_times, t_start, t_stop, 0
    )
    second = 10.2 - 10.00333  # the part of the second window inside the interval
    expected = {
        "cpu_j": 1.250 * 0.1 + 0.750 * second,
        "gpu_j": 3.400 * 0.1 + 1.600 * second,
        "ane_j": 0.0,
        "combined_j": 4.650 * 0.1 + 2.350 * second,
    }
    assert combined.joules == pytest.approx(expected["combined_j"])
    assert gpu.joules == pytest.approx(expected["gpu_j"])
    assert combined.components == pytest.approx(expected)
    assert combined.includes_host is True and gpu.includes_host is False
    assert combined.method == gpu.method == POWERMETRICS_METHOD
    assert combined.method == "sudo powermetrics --samplers gpu_power,cpu_power -i 200"
    assert combined.n_samples == 2 and combined.interval_ms == 200
    assert "99%" in combined.notes  # 0.29667 s of the 0.3 s interval; the 3 ms gap is jitter
    # never a silent zero: an interval no sample covers, one the windows cover only in part
    # (the truncated third block leaves 10.20345-10.3 uncovered), or the superuser refusal raise
    with pytest.raises(RuntimeError, match="not measured"):
        PowermetricsMeter()._reading(POWERMETRICS_TEXT, "", header_times, 11.0, 11.5, 0)
    with pytest.raises(RuntimeError, match="cover only"):
        PowermetricsMeter()._reading(POWERMETRICS_TEXT, "", header_times, 9.9, 10.3, 0)
    with pytest.raises(RuntimeError, match="not measured"):
        PowermetricsMeter()._reading("", "", [], 0.0, 0.1, 0)
    with pytest.raises(PermissionError, match="sudo"):
        PowermetricsMeter()._reading(
            "powermetrics must be invoked as the superuser\n", "", [], 0.0, 0.1, 0
        )
    with pytest.raises(ValueError, match="scope"):
        PowermetricsMeter(scope="cpu")  # type: ignore[arg-type]


class _FakeProcess:
    """``sudo -n powermetrics`` as seen by the meter when sudo refuses: exits at once."""

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.args = command
        self.stdout = io.StringIO("")
        self.stderr = io.StringIO("sudo: a password is required\n")
        self.returncode = 1

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


def test_p0_6_powermetrics_requires_sudo(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        spawned.append(command)
        return _FakeProcess(command, **kwargs)

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", lambda name: name)  # both paths are absolute
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    meter = PowermetricsMeter()
    with pytest.raises(PermissionError, match="sudo") as info:
        meter.start()
    message = str(info.value)
    assert "a password is required" in message and "NOPASSWD" in message and "sudo -v" in message
    assert "gpu_power\\,cpu_power -i 200" in message  # the sudoers line pins the exact command
    assert spawned == [
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/powermetrics",
            "--samplers",
            "gpu_power,cpu_power",
            "-i",
            "200",
            "-b",
            "1",
        ]
    ]
    with pytest.raises(RuntimeError, match="before start"):
        meter.stop()
    # not macOS
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="macOS"):
        PowermetricsMeter().start()


# --------------------------------------------------------------------------------------------------
# report schema
# --------------------------------------------------------------------------------------------------


def test_p0_6_report_schema(tmp_path: Path) -> None:
    prof = Profiler(_cpu_ctx())
    with prof.segment("startup"):
        pass
    with prof.segment("shared_path", tokens=8):
        with prof.segment("scorer", tokens=8):  # routing ran: the scorer is its own line
            torch.ones(4) @ torch.ones(4, 4)
    path = tmp_path / "profile.json"
    prof.to_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == TOP_LEVEL_FIELDS
    assert data["profile"] == "cpu" and data["device"] == "cpu"
    assert data["timing_method"] == "synchronise + wall clock"
    assert data["launches_method"] == "disabled"
    assert set(data["power"]) == POWER_FIELDS
    assert data["power"]["method"] == "none"
    assert data["power"]["n_readings"] == 2  # startup and shared_path: the outermost segments
    assert list(data["segments"]) == ["startup", "shared_path", "scorer"]
    for name, segment in data["segments"].items():
        assert set(segment) == set(SEGMENT_FIELDS), name
        for key, kinds in SEGMENT_FIELDS.items():
            assert type(segment[key]) in kinds, (name, key, segment[key])
    assert data["segments"]["scorer"]["tokens"] == 8
    assert data["segments"]["scorer"]["energy_j"] is None  # nested: not metered
    assert data["segments"]["shared_path"]["energy_j"] is None  # NullMeter: not measured
    assert data["power"]["n_unresolved"] == 2
    assert data["segments"]["startup"]["launches"] == UNSUPPORTED
    assert prof.report().to_dict() == data
    assert prof.report().to_dict() == prof.report().to_dict()  # reporting is repeatable
    # a counting profiler on a backend without kernel events says so in the report
    counting = Profiler(_cpu_ctx(), count_launches=True)
    with counting.segment("decode"):
        pass
    report = counting.report()
    assert report.launches_method == "unsupported on cpu"
    assert report.segments["decode"].launches == UNSUPPORTED
    # an empty report is still a complete document
    empty = Profiler(_cpu_ctx()).report().to_dict()
    assert set(empty) == TOP_LEVEL_FIELDS and empty["segments"] == {}
    assert empty["power"]["method"] is None and empty["power"]["n_readings"] == 0


# --------------------------------------------------------------------------------------------------
# launches
# --------------------------------------------------------------------------------------------------


def _count_on(ctx: DeviceContext) -> tuple[int, int]:
    ran: list[int] = []
    x = torch.randn(32, 32, device=ctx.device)

    def work() -> None:
        ran.append(1)
        x @ x

    return count_launches(work, ctx), len(ran)


def test_p0_6_launch_count_unsupported() -> None:
    profiles = ["cpu"]
    if torch.backends.mps.is_available():
        profiles.append("m_series")
    for profile in profiles:
        ctx = resolve(load_hardware_profile(profile))
        assert _count_on(ctx) == (UNSUPPORTED, 1), profile  # the callable still runs


@pytest.mark.cuda
def test_p0_6_launch_count_small_loop() -> None:
    ctx = _cuda_ctx()
    n = 7
    a = torch.randn(256, 256, device=ctx.device)
    b = torch.randn(256, 256, device=ctx.device)
    a @ b  # warm-up: cuBLAS initialisation launches extra kernels on the first call
    synchronize(ctx.device)

    def loop() -> None:
        for _ in range(n):
            a @ b

    assert count_launches(loop, ctx) == n
    assert count_launches(loop, ctx) == n  # repeatable
    prof = Profiler(ctx, count_launches=True)
    with prof.segment("shared_path", tokens=n):
        with prof.segment("scorer"):
            loop()
        loop()
    # backward kernels come from the autograd worker thread and still belong to `backward`
    weight = torch.randn(256, 256, device=ctx.device, requires_grad=True)
    (a @ weight).sum().backward()  # warm-up
    synchronize(ctx.device)
    with prof.segment("backward"):
        (a @ weight).sum().backward()
    report = prof.report()
    assert report.launches_method == CUDA_METHOD
    assert report.timing_method == "cuda events"
    assert report.segments["scorer"].launches == n
    assert report.segments["shared_path"].launches == 2 * n  # parents include their children
    assert report.segments["shared_path"].device_ms > 0
    assert report.segments["backward"].launches >= 2  # forward matmul + at least one grad kernel
