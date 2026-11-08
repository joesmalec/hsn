"""P2.8 on ``profile:m_series``: the Metal System Trace path of ``scripts/profile_step.py``.

PyTorch 2.15 has no MPS profiler activity, so the m_series profiles take GPU time from Xcode's
Metal System Trace (:mod:`hsn.profiler.metal_trace`) and attribute it to segments through the
dispatch range that contains each GPU interval. On the CPU these tests check the parts that do not
need a GPU: the export parser (id/ref compression, the pid filter, the mach-time epoch), the clock
fit, interval assignment, busy time, the serialising dispatch mode (it must leave compiled frames
compiled), the xctrace start/stop protocol against a stand-in, the Metal-kernel wrapper, the
per-graph kernel provenance, the MPS attribution with its decision line and the script's device
gate. The ``mps`` tests record a real trace (macOS with MPS and Xcode only).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import textwrap
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from hsn.profiler import metal_trace
from hsn.profiler.metal_trace import GPUInterval, MetalTraceError

REPO = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ps() -> ModuleType:
    return load_script("profile_step")


def _trace_ready() -> tuple[bool, str]:
    if not torch.backends.mps.is_available():
        return False, "no mps device on this machine"
    return metal_trace.available()


# --------------------------------------------------------------------------------------------------
# Export parsing
# --------------------------------------------------------------------------------------------------

GPU_XML = """<?xml version="1.0"?>
<trace-query-result>
<node xpath='//trace-toc[1]/run[1]/data[1]/table[109]'><schema name="metal-gpu-intervals">
<col><mnemonic>start</mnemonic></col><col><mnemonic>duration</mnemonic></col>
<col><mnemonic>channel-name</mnemonic></col><col><mnemonic>process</mnemonic></col>
<col><mnemonic>cmdbuffer-id</mnemonic></col></schema>
<row><start-time id="1" fmt="00:00.500">500</start-time><duration id="2" fmt="20 ns">20</duration>
<gpu-channel-name id="3" fmt="Compute">Compute</gpu-channel-name>
<process id="4" fmt="python (77)"><pid id="5" fmt="77">77</pid></process>
<metal-command-buffer-id id="6" fmt="0x1">11</metal-command-buffer-id></row>
<row><start-time id="7" fmt="00:00.100">100</start-time><duration ref="2"/>
<gpu-channel-name ref="3"/><process ref="4"/>
<metal-command-buffer-id id="8" fmt="0x2">12</metal-command-buffer-id></row>
<row><start-time id="9" fmt="00:00.300">300</start-time><duration id="10" fmt="5 ns">5</duration>
<gpu-channel-name id="11" fmt="Vertex">Vertex</gpu-channel-name>
<process id="12" fmt="WindowServer (9)"><pid id="13" fmt="9">9</pid></process>
<metal-command-buffer-id id="14" fmt="0x3">13</metal-command-buffer-id></row>
</node></trace-query-result>
"""

TIME_XML = """<?xml version="1.0"?>
<trace-query-result><node><schema name="time-info">
<col><mnemonic>update-time</mnemonic></col><col><mnemonic>mabs-epoch</mnemonic></col>
<col><mnemonic>timebase-info</mnemonic></col></schema>
<row><sample-time id="1" fmt="0">0</sample-time>
<mach-absolute-time id="2" fmt="3000">3000</mach-absolute-time>
<mach-timebase-info id="3" fmt="125/3">
<mach-timebase-info-field id="4" fmt="125">125</mach-timebase-info-field>
<mach-timebase-info-field id="5" fmt="3">3</mach-timebase-info-field></mach-timebase-info></row>
</node></trace-query-result>
"""


def _fake_export(monkeypatch: pytest.MonkeyPatch, tables: dict[str, str]) -> None:
    def export(trace: Path, schema: str, out: Path, *, run: int = 1) -> Path:
        out.write_text(tables[schema])
        return out

    monkeypatch.setattr(metal_trace, "export_table", export)


def test_p2_8_trace_rows_resolve_refs(tmp_path: Path) -> None:
    """``ref`` repeats the value an ``id`` defined (even inside a nested ``process`` cell), and the
    pid is read from the process's child."""
    path = tmp_path / "gpu.xml"
    path.write_text(GPU_XML)
    rows = list(metal_trace.iter_rows(path))
    assert [r["start"].text for r in rows] == ["500", "100", "300"]
    assert [r["duration"].text for r in rows] == ["20", "20", "5"]
    assert [r["process"].pid for r in rows] == [77, 77, 9]
    assert rows[1]["channel-name"].text == "Compute"


def test_p2_8_trace_undefined_ref_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.xml"
    path.write_text(GPU_XML.replace('<duration ref="2"/>', '<duration ref="99"/>'))
    with pytest.raises(MetalTraceError, match="undefined id"):
        list(metal_trace.iter_rows(path))


def test_p2_8_gpu_intervals_epoch_pid_and_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Intervals of the pid only, sorted, on the monotonic clock: run epoch 3000 mach ticks at
    125/3 ns per tick is 125,000 ns."""
    _fake_export(monkeypatch, {"time-info": TIME_XML, "metal-gpu-intervals": GPU_XML})
    got = metal_trace.gpu_intervals(tmp_path / "t.trace", 77, tmp_path)
    assert got == [
        GPUInterval(start_ns=125_100, duration_ns=20, channel="Compute", command_buffer=12),
        GPUInterval(start_ns=125_500, duration_ns=20, channel="Compute", command_buffer=11),
    ]
    with pytest.raises(MetalTraceError, match="no GPU interval of pid 5"):
        metal_trace.gpu_intervals(tmp_path / "t.trace", 5, tmp_path)


# --------------------------------------------------------------------------------------------------
# Clocks, assignment, busy time
# --------------------------------------------------------------------------------------------------


def test_p2_8_assign_innermost_partial_unassigned() -> None:
    """The shortest range containing the interval wins (nested and overlapping ranges included);
    an interval crossing a boundary goes by its midpoint; one outside every range is unassigned."""
    ranges = [(0.0, 100.0), (10.0, 20.0), (30.0, 60.0), (35.0, 45.0), (200.0, 210.0)]
    intervals = [
        (12.0, 18.0),
        (36.0, 40.0),
        (50.0, 55.0),
        (1.0, 2.0),
        (205.0, 215.0),
        (300.0, 301.0),
    ]
    got = metal_trace.assign(intervals, ranges)
    assert got.index == [1, 3, 2, 0, 4, None]  # (205, 215) crosses (200, 210): by its midpoint
    assert (got.contained, got.partial, got.unassigned) == (4, 1, 1)
    outside = metal_trace.assign([(150.0, 250.0)], ranges)  # midpoint 200: the range's start
    assert outside.index == [4] and outside.partial == 1


def test_p2_8_busy_ns_union_clipped() -> None:
    g = [
        GPUInterval(start_ns=0, duration_ns=10, channel="Compute", command_buffer=1),
        GPUInterval(start_ns=5, duration_ns=10, channel="Compute", command_buffer=2),  # overlaps
        GPUInterval(start_ns=30, duration_ns=10, channel="Compute", command_buffer=3),
        GPUInterval(start_ns=95, duration_ns=10, channel="Compute", command_buffer=4),  # clipped
    ]
    assert metal_trace.busy_ns(g, 0, 100) == 15 + 10 + 5
    assert metal_trace.busy_ns(g, 8, 32) == 7 + 2


# --------------------------------------------------------------------------------------------------
# The serialising dispatch mode
# --------------------------------------------------------------------------------------------------


def test_p2_8_serialised_mode_records_and_waits() -> None:
    """Each aten op, backward ones included, waits (here a counting stand-in for
    torch.mps.synchronize) and is recorded with its span and the caller's classification, taken
    in the op's context; disabled, ops pass straight through."""
    waits: list[int] = []
    seen: list[tuple[str, bool]] = []

    def classify(func: Any) -> str:
        seen.append((str(func), torch._C._current_autograd_node() is not None))
        return str(func)

    mode = metal_trace.Serialised(lambda: waits.append(time.monotonic_ns()), classify=classify)
    x = torch.randn(4, 4, requires_grad=True)
    with mode:
        (x @ x).tanh().sum().backward()  # type: ignore[no-untyped-call]
    assert mode.calls == len(waits) == len(mode.records) == len(seen) > 4
    assert all(r.start_ns <= w <= r.end_ns for r, w in zip(mode.records, waits, strict=True))
    whats = [r.what for r in mode.records]
    assert "aten.mm.default" in whats and "aten.tanh_backward.default" in whats
    assert ("aten.tanh_backward.default", True) in seen and ("aten.mm.default", False) in seen
    mode.enabled = False
    with mode:
        (x @ x).sum()
    assert mode.calls == len(waits)


def test_p2_8_serialised_mode_keeps_compiled_frames_compiled() -> None:
    """Dynamo runs a frame eager while a dispatch mode is active, unless the mode ignores compile
    internals: under :class:`Serialised` the function still compiles (the backend is called) and
    its runtime aten calls still pass through the mode."""
    compiled: list[int] = []

    def backend(gm: torch.fx.GraphModule, example_inputs: Any) -> Any:
        compiled.append(1)
        return gm.forward

    fn = torch.compile(lambda a: (a * 2).sin().sum(), backend=backend, fullgraph=True)
    mode = metal_trace.Serialised(lambda: None)
    with mode:
        out = fn(torch.randn(8))
    assert compiled == [1]
    assert mode.calls >= 2 and torch.isfinite(out)


# --------------------------------------------------------------------------------------------------
# xctrace protocol (a stand-in recorder)
# --------------------------------------------------------------------------------------------------

STAND_IN = textwrap.dedent(
    """
    import pathlib, signal, sys, time
    out, mode, ktrace = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    print("Starting recording with the Metal System Trace template.", flush=True)
    if mode == "fail":
        print("Cannot attach to process", flush=True)
        sys.exit(9)
    (ktrace / "instrumentsNEW1.ktrace").write_bytes(b"k" * 1000)  # xctrace's kernel buffer
    print("Ctrl-C to stop the recording", flush=True)
    stop = []
    signal.signal(signal.SIGINT, lambda *a: stop.append(1))
    while not stop:
        time.sleep(0.01)
    if mode == "ok":
        open(out, "w").write("trace")
    sys.exit(0)
    """
)


def _stand_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> metal_trace.Recording:
    script = tmp_path / "xctrace_stand_in.py"
    script.write_text(STAND_IN)
    ktrace = tmp_path / "user_temp"
    ktrace.mkdir(exist_ok=True)
    monkeypatch.setattr(metal_trace, "available", lambda: (True, ""))
    monkeypatch.setattr(metal_trace, "other_recordings", lambda: [])
    monkeypatch.setattr(metal_trace, "ktrace_dir", lambda: ktrace)
    recording = metal_trace.Recording(tmp_path / f"{mode}.trace", start_timeout_s=30)
    monkeypatch.setattr(
        recording,
        "command",
        lambda: [sys.executable, str(script), str(recording.out), mode, str(ktrace)],
    )
    return recording


def test_p2_8_recording_waits_for_live_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The block runs once the recording is live; on exit the trace is saved, and the kernel-trace
    buffer this recording created is deleted while an older one stays."""
    recording = _stand_in(tmp_path, monkeypatch, "ok")
    older = tmp_path / "user_temp" / "instrumentsOLD.ktrace"
    older.write_bytes(b"old")
    with recording:
        assert any(metal_trace.LIVE_LINE in line for line in recording.lines)
        assert (tmp_path / "user_temp" / "instrumentsNEW1.ktrace").exists()
    assert recording.out.read_text() == "trace"
    assert not (tmp_path / "user_temp" / "instrumentsNEW1.ktrace").exists()
    assert older.read_bytes() == b"old" and recording.ktrace_bytes == 1000


def test_p2_8_recording_stop_early(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recording = _stand_in(tmp_path, monkeypatch, "ok")
    with recording:
        recording.stop()
        recording.stop()  # idempotent
    assert recording.out.read_text() == "trace"


def test_p2_8_recording_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An xctrace that exits before recording, or one that saves nothing, raises with its output;
    a missing xctrace, or another recording running, raises before anything starts."""
    with pytest.raises(MetalTraceError, match="Cannot attach"):
        with _stand_in(tmp_path, monkeypatch, "fail"):
            pass
    with pytest.raises(MetalTraceError, match="without saving"):
        with _stand_in(tmp_path, monkeypatch, "nosave"):
            pass
    assert not (tmp_path / "user_temp" / "instrumentsNEW1.ktrace").exists()  # cleaned anyway
    busy = _stand_in(tmp_path, monkeypatch, "ok")
    monkeypatch.setattr(metal_trace, "other_recordings", lambda: ["123 xctrace record --attach 9"])
    with pytest.raises(MetalTraceError, match="another xctrace recording"):
        with busy:
            pass
    monkeypatch.setattr(metal_trace, "available", lambda: (False, "xctrace not found"))
    with pytest.raises(MetalTraceError, match="xctrace not found"):
        with metal_trace.Recording(tmp_path / "x.trace"):
            pass
    with pytest.raises(MetalTraceError, match="xctrace not found"):
        metal_trace.version()


def test_p2_8_is_recorder() -> None:
    xc = "/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace"
    assert metal_trace.is_recorder(f"123 {xc} record --template Metal System Trace --attach 9")
    assert not metal_trace.is_recorder(f"124 {xc} export --input a.trace --toc")
    assert not metal_trace.is_recorder("125 /bin/zsh -c ps | grep 'xctrace record'")
    assert not metal_trace.is_recorder("126")


def test_p2_8_encoders_per_command_buffer() -> None:
    g = [
        GPUInterval(start_ns=i, duration_ns=1, channel="Compute", command_buffer=cb)
        for i, cb in enumerate([1, 2, 2, 3])
    ]
    assert metal_trace.encoders_per_command_buffer(g) == {1: 2, 2: 1}


class FakeRecording:
    """A stand-in :class:`Recording`: the GPU work "recorded" is every event logged while live."""

    log: list[tuple[str, int]] = []
    live: list[tuple[int, int]] = []

    def __init__(self, out: Path, pid: int | None = None, **_: Any) -> None:
        self.out = Path(out)
        self.ktrace_bytes = 10

    def __enter__(self) -> FakeRecording:
        self._t0 = time.monotonic_ns()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.out.mkdir(parents=True)
        FakeRecording.live.append((self._t0, time.monotonic_ns()))


def test_p2_8_chunked_trace_rolls_over_and_drops_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every ``chunk`` dispatches a new recording starts; the intervals of all chunks are kept,
    the markers' are not; a chunk whose trace lacks a marker raises."""
    log: list[tuple[str, int]] = []
    FakeRecording.live = []
    monkeypatch.setattr(metal_trace, "Recording", FakeRecording)

    def intervals(trace: Path, pid: int, workdir: Path) -> list[GPUInterval]:
        lo, hi = FakeRecording.live[-1]
        return [
            GPUInterval(start_ns=t, duration_ns=1, channel=kind, command_buffer=i)
            for i, (kind, t) in enumerate(log)
            if lo <= t <= hi
        ]

    monkeypatch.setattr(metal_trace, "gpu_intervals", intervals)
    mode = metal_trace.Serialised(lambda: None)
    chunks = metal_trace.ChunkedTrace(
        tmp_path / "w",
        chunk=3,
        mode=mode,
        sync=lambda: None,
        marker=lambda: log.append(("marker", time.monotonic_ns())),
    )
    with chunks:
        for _ in range(7):
            log.append(("op", time.monotonic_ns()))
            chunks.tick()
    assert chunks.chunks == 3  # 3 + 3 + 1 dispatches
    assert [g.channel for g in chunks.intervals] == ["op"] * 7
    assert chunks.ktrace_bytes == 30 and mode.enabled
    assert not any((tmp_path / "w").glob("chunk*.trace"))  # deleted after export

    lost = metal_trace.ChunkedTrace(
        tmp_path / "w2", chunk=3, sync=lambda: None, marker=lambda: None
    )
    with pytest.raises(MetalTraceError, match="start marker"):
        with lost:
            pass


def test_p2_8_trace_unavailable_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metal_trace.sys, "platform", "linux")
    ok, reason = metal_trace.available()
    assert not ok and "macOS" in reason


# --------------------------------------------------------------------------------------------------
# profile_step: the Metal-kernel wrapper, provenance, attribution, device gate, machine status
# --------------------------------------------------------------------------------------------------


def test_p2_8_metal_kernel_wrapper(ps: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Off, a wrapped kernel is the kernel; on, it waits before and after, is counted and recorded
    with its span and direction (forward outside the autograd engine); non-callable attributes
    pass through."""
    import torch._inductor.runtime.runtime_utils as runtime_utils

    calls: list[tuple[Any, ...]] = []
    library = SimpleNamespace(generated_kernel_0=lambda *a, **k: calls.append(a) or "out", n=3)
    monkeypatch.setattr(runtime_utils, "compile_mps_shader", lambda source: library)
    waits: list[int] = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: waits.append(time.monotonic_ns()))
    gate = ps.serialise_mps_kernels()
    lib = runtime_utils.compile_mps_shader("source")
    assert lib.n == 3
    assert lib.generated_kernel_0(1, threads=[4]) == "out" and waits == [] and gate.calls == 0
    recorded: list[tuple[Any, ...]] = []
    gate.record = lambda *a: recorded.append(a)
    ticks: list[int] = []
    gate.on_dispatch = lambda: ticks.append(1)
    gate.on = True
    assert lib.generated_kernel_0(2) == "out"
    assert len(waits) == 2 and gate.calls == 1 and calls == [(1,), (2,)] and ticks == [1]
    ((name, start, end, backward),) = recorded
    assert name == "generated_kernel_0" and backward is False
    assert waits[0] <= start <= end <= waits[1] or start <= waits[1] <= end


def test_p2_8_mps_kernel_provenance_per_graph(
    ps: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the given graphs count; one kernel name with different origins in two of them raises."""
    import torch._inductor.debug as inductor_debug

    trace_a = 'File "/r/python/hsn/model/units.py", line 3, in forward'
    trace_b = 'File "/r/python/hsn/model/graph.py", line 9, in messages'
    jsons = {
        "('Torch-Compiled Region: 0/0', False)": {
            "generated_kernel_0:1": {"stack_traces": [trace_b]}
        },
        "('Torch-Compiled Region: 1/0', False)": {
            "generated_kernel_0:1": {"stack_traces": [trace_a]}
        },
        "('Torch-Compiled Region: 1/0', True)": {
            "generated_kernel_0:1": {"stack_traces": [trace_b]},
            "extern_kernels_mm_2": {"stack_traces": [trace_a]},
        },
    }
    monkeypatch.setattr(inductor_debug, "get_kernel_information_jsons", lambda: jsons)
    got = ps.mps_kernel_provenance(
        {"('Torch-Compiled Region: 1/0', False)", "('Torch-Compiled Region: 1/0', True)"}
    )
    assert got[(False, "generated_kernel_0")] == [[("/r/python/hsn/model/units.py", 3, "forward")]]
    assert got[(True, "generated_kernel_0")] == [[("/r/python/hsn/model/graph.py", 9, "messages")]]
    assert (True, "extern_kernels_mm_2") in got
    with pytest.raises(RuntimeError, match="different origins"):
        ps.mps_kernel_provenance(set(jsons))
    with pytest.raises(RuntimeError, match="no inductor graph"):
        ps.mps_kernel_provenance(set())


def test_p2_8_attribute_mps_segments_and_decision(ps: ModuleType) -> None:
    """GPU intervals take the segment shares of the dispatch whose span contains them; an interval
    inside no dispatch, or in a dispatch no rule classified, is ``other``; intervals outside the
    step window are left out; the decision line follows the CUDA definition."""
    D = metal_trace.Dispatch

    def gi(start_us: float, dur_us: float) -> GPUInterval:
        return GPUInterval(
            start_ns=int(start_us * 1e3),
            duration_ns=int(dur_us * 1e3),
            channel="Compute",
            command_buffer=int(start_us),
        )

    records = [
        D(start_ns=10_000, end_ns=20_000, what=({"selected_cells": 1.0}, "units mm")),
        D(start_ns=120_000, end_ns=180_000, what=({"scorer": 1.0}, "router addmm")),
        D(start_ns=310_000, end_ns=390_000, what=({"messages": 0.5, "scorer": 0.5}, "fused")),
        D(start_ns=500_000, end_ns=600_000, what=(None, "AddmmBackward0 / aten.mm")),
        D(start_ns=700_000, end_ns=705_000, what=({"decode": 1.0}, "no gpu work")),
    ]
    intervals = [gi(12, 4), gi(130, 30), gi(320, 40), gi(520, 60), gi(800, 10), gi(5000, 99)]
    got = ps.attribute_mps(records, intervals, 0, 1_000_000)
    assert got["device_us"] == pytest.approx(
        {"selected_cells": 4, "scorer": 50, "messages": 20, "other": 70}
    )
    assert got["kernels_total"] == 5  # the interval at 5000 µs is outside the step window
    assert got["dispatches"] == 4 and got["unassigned"] == 1 and got["contained"] == 4
    assert got["unattributed_kernels"] == 2
    assert got["assigned_fraction"] == pytest.approx(134 / 144)
    shares = {k: v / 400.0 for k, v in got["device_us"].items()}  # a 400 µs step
    assert ps.decision_line(shares, "L1") == (
        "L2 justified: no — messages is 5.0% of step time after L1"
    )


def test_p2_8_segment_table_cuda_and_proportional(ps: ModuleType) -> None:
    """CUDA: segment device time over the step p50, as it is. MPS: the serialised mix applied to
    the natural step's GPU busy time, so the shares sum to the GPU busy fraction; the raw
    serialised ms are kept."""
    attribution = {
        "device_us": {"messages": 300.0, "selected_cells": 600.0, "other": 100.0},
        "kernels": {"messages": 3.0, "selected_cells": 2.0, "other": 1.0},
        "gpu_us": 1000.0,
    }
    segments, shares = ps.segment_table(attribution, 2000.0)
    assert shares["selected_cells"] == pytest.approx(0.3) and shares["messages"] == 0.15
    assert segments["selected_cells"] == {
        "kernels_device": 2.0,
        "device_ms": 0.6,
        "share_of_step": 0.3,
    }
    segments, shares = ps.segment_table(attribution, 2000.0, natural_busy_us=500.0)
    assert shares["selected_cells"] == pytest.approx(0.15) and shares["messages"] == 0.075
    assert sum(shares.values()) == pytest.approx(500.0 / 2000.0)
    assert segments["messages"]["device_ms"] == 0.15
    assert segments["messages"]["device_ms_serialised"] == 0.3
    assert ps.decision_line(shares, "L1").startswith("L2 justified: no — selected_cells is 15.0%")


def test_p2_8_mps_classifier_rules_on_hsn(ps: ModuleType) -> None:
    """The MPS classifier on a real HSN forward, backward and optimizer step (CPU, ref_a1, dense):
    forward ops get the segment of their HSN frames, backward ops the segment of the forward op
    that created their node, gradient accumulation and the optimizer their own; the dense step
    reaches every model segment."""
    from hsn.config import load_config
    from hsn.runner import build_model

    cfg = load_config(REPO / "configs" / "ref_a1.yaml", overrides=["profile=cpu"])
    torch.manual_seed(0)
    model = build_model(cfg)
    B, T = 2, 4
    tokens = torch.randint(0, cfg.model.vocab_size, (B, T))
    ctx_ids = torch.randint(0, cfg.model.n_context, (B, T))
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    classifier = ps.MPSClassifier({})
    mode = metal_trace.Serialised(lambda: None, classify=classifier)
    with mode:
        logits, _, _ = model(tokens, ctx_ids, model.init_state(B), reset)
        logits.float().square().mean().backward()
        optimizer.step()
    segments = {seg for r in mode.records if r.what[0] for seg in r.what[0]}
    assert {"shared_path", "messages", "selected_cells", "decode"} <= segments
    assert {"grad_accumulation", "optimizer"} <= segments
    backward = [r for r in mode.records if "Backward" in r.what[1].split(" / ")[0]]
    assert backward and sum(r.what[0] is not None for r in backward) / len(backward) > 0.9
    assert classifier.forward  # sequence numbers of forward nodes were recorded


def test_p2_8_mps_classifier_input_buffer_sums(ps: ModuleType) -> None:
    """The engine's sum of two incoming gradients runs while the producing node is still current;
    it is ``grad_accumulation`` (as on CUDA), the node's own ops are not."""
    classifier = ps.MPSClassifier({})
    mode = metal_trace.Serialised(lambda: None, classify=classifier)
    x = torch.randn(16, requires_grad=True)
    y = x * 2
    with mode:
        (y.sin() + y.cos()).sum().backward()  # type: ignore[no-untyped-call]
    sums = [r for r in mode.records if r.what[1].startswith("input-buffer sum")]
    assert sums and all(r.what[0] == {"grad_accumulation": 1.0} for r in sums)
    own = [r for r in mode.records if r.what[1].startswith(("SinBackward0", "CosBackward0"))]
    assert own and not any(r.what[0] == {"grad_accumulation": 1.0} for r in own)


def test_p2_8_profile_step_refuses_cpu(ps: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CPU has no device queue: the script refuses it before any step runs."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_step.py",
            "--config",
            str(REPO / "configs" / "experiments" / "g1_ref_a1.yaml"),
            "--profile",
            "cpu",
            "--level",
            "L0",
        ],
    )
    monkeypatch.setattr(ps, "cpu_processes", lambda limit=8: [])
    cpu = SimpleNamespace(device=torch.device("cpu"))
    monkeypatch.setattr(ps, "build", lambda args: (None, cpu, None, None, None, None))
    with pytest.raises(RuntimeError, match="profiles a device queue"):
        ps.main()


def test_p2_8_machine_status_rule(ps: ModuleType) -> None:
    own = str(os.getpid())
    quiet = [{"pid": own, "cpu": "300.0", "rss_kib": "1", "name": "python"}]
    busy = quiet + [{"pid": "1", "cpu": "75.0", "rss_kib": "1", "name": "WindowServer"}]
    assert ps.machine_status([quiet, quiet]) == "idle"
    assert ps.machine_status([quiet, busy]) == "shared"
    ps_itself = quiet + [{"pid": "2", "cpu": "90.0", "rss_kib": "1", "name": "/bin/ps"}]
    assert ps.machine_status([ps_itself]) == "idle"


# --------------------------------------------------------------------------------------------------
# A real trace (macOS with MPS and Xcode)
# --------------------------------------------------------------------------------------------------


@pytest.mark.mps
def test_p2_8_metal_trace_serialised_ops_mps(tmp_path: Path, ps: ModuleType) -> None:
    """Under :class:`Serialised` every MPS op's GPU interval lies inside its dispatch range on the
    fitted clock, one encoder per command buffer, gather and scatter included (the native Metal
    ops that ``torch.mps.profiler``'s wait does not separate)."""
    ok, reason = _trace_ready()
    if not ok:
        pytest.skip(reason)
    dev = torch.device("mps")
    h = torch.randn(16, 64, 32, device=dev, requires_grad=True)
    idx = torch.randint(0, 64, (16, 8, 1), device=dev).expand(-1, -1, 32)
    w = torch.randn(32, 32, device=dev, requires_grad=True)

    def work() -> None:
        g = torch.gather(h, 1, idx)
        out = h.scatter(1, idx, torch.nn.functional.gelu(g @ w))
        (torch.softmax(out, -1) * out).sum().backward()  # type: ignore[no-untyped-call]

    work()
    torch.mps.synchronize()
    mode = metal_trace.Serialised(metal_trace.mps_sync, classify=str)
    chunks = metal_trace.ChunkedTrace(tmp_path / "ops", chunk=10, mode=mode)
    mode.on_dispatch = chunks.tick
    with chunks:
        lo = time.monotonic_ns()
        with mode:
            work()
        hi = time.monotonic_ns()
    assert chunks.chunks >= 2 and chunks.ktrace_bytes > 0
    intervals = [g for g in chunks.intervals if lo <= g.start_ns < hi]
    got = metal_trace.assign(
        [(g.start_ns, g.end_ns) for g in intervals], [(r.start_ns, r.end_ns) for r in mode.records]
    )
    assert len(intervals) >= 8 and got.contained == len(intervals)
    names = {mode.records[i].what for i in got.index if i is not None}
    assert any("gather" in n for n in names) and any("scatter" in n for n in names)
    assert set(metal_trace.encoders_per_command_buffer(intervals)) == {1}


@pytest.mark.mps
def test_p2_8_metal_kernel_wrapper_compiled_mps(
    ps: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the wrapper installed and the module cache cleared, an inductor-compiled function's
    Metal kernels run through it, and their names are in the new graphs' provenance."""
    if not torch.backends.mps.is_available():
        pytest.skip("no mps device on this machine")
    import torch._functorch.config as functorch_config
    import torch._inductor.config as inductor_config
    import torch._inductor.runtime.runtime_utils as runtime_utils
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.debug import get_kernel_information_jsons

    monkeypatch.setattr(runtime_utils, "compile_mps_shader", runtime_utils.compile_mps_shader)
    monkeypatch.setattr(inductor_config.trace, "provenance_tracking_to_timeline", True)
    monkeypatch.setattr(inductor_config, "fx_graph_cache", False)
    monkeypatch.setattr(functorch_config, "enable_autograd_cache", False)
    gate = ps.serialise_mps_kernels()
    PyCodeCache.cache_clear()
    before = set(get_kernel_information_jsons())
    x = torch.randn(64, 64, device="mps", requires_grad=True)
    fn = torch.compile(lambda a: (a.sin() * 3).relu().sum(), fullgraph=True, dynamic=False)
    fn(x).backward()  # type: ignore[no-untyped-call]
    recorded: list[tuple[Any, ...]] = []
    gate.record = lambda *a: recorded.append(a)
    gate.on = True
    fn(x).backward()  # type: ignore[no-untyped-call]
    gate.on = False
    names = {name for name, *_ in recorded}
    assert {backward for *_, backward in recorded} == {False, True}
    assert gate.calls > 0 and names
    provenance = ps.mps_kernel_provenance(set(get_kernel_information_jsons()) - before)
    assert names <= {name for _, name in provenance}
