#!/usr/bin/env python
"""P2.8: the step profile of one execution level and the L2 decision line.

    python scripts/profile_step.py --config configs/experiments/g1_ref_a1.yaml --profile rtx3060 \\
        --level L0|L1 [--override KEY=VALUE ...] [--tag NAME] [--steps 20] [--repeats 3]

writes ``profiles/<profile>_<level>[_<tag>].json`` and prints its last line. Levels (plan §3.2):
``L0`` is the native-op path (``hardware.compile_mode=none``), ``L1`` the launch-overhead layer of
:mod:`hsn.train.compile` (``hardware.compile_mode=reduce-overhead`` unless an override names
another compiled mode). The model trains through the real
:class:`~hsn.train.tbptt.TBPTTTrainer` on the configuration's data (evaluation and checkpoint
writes stubbed out; ``train.stats_every`` pushed past the run, so every measured step is a plain
step), after ``--warm`` unmeasured steps.

What the file holds:

* ``compile``: the wall time of the steps that compiled or recorded CUDA Graphs (L1; the trainer's
  ``measured["compile_seconds"]``) and of the first, warm-up step (``startup``). Never inside a
  steady-state number.
* ``train_step``: wall ms p50 / p95 of ``--repeats`` blocks of ``--steps`` plain steps (each
  block's p50 too), tokens per second, and from one profiled plain step: ``launches_host`` (the
  host's CUDA runtime and driver launch calls, :data:`hsn.profiler.launches.HOST_LAUNCH_CALLS`:
  what L1 removes), ``kernels_device`` (P0.6's launches: device kernels, memcpys, memsets, which
  CUDA Graphs replay rather than remove), ``device_ms`` (their summed device time) and the peak /
  reserved allocator bytes of the step.
* ``serving_step``: :func:`hsn.train.compile.compile_step` at this level, batch 1 and the
  configuration's batch: per-token latency p50 / p95 and host launches per token.
* ``segments``: per P0.6 segment (``shared_path``, ``scorer``, ``messages``, ``selected_cells``,
  ``decode``; forward and backward attributed to the forward code that created them; plus
  ``loss``, ``optimizer``, ``grad_accumulation`` (the leaf accumulation and the autograd
  engine's sums of incoming gradients) and ``other``): device kernels, device ms and the share
  of the L1 (or L0) step's p50. Attribution reads the source stack of each kernel: at L0 the
  Python frames ``torch.profiler(with_stack=True)`` records above the launching op (a backward
  op through the ``sequence_nr`` of the forward op that recorded it), at L1 inductor's kernel
  provenance (each compiled kernel's origin nodes and their stack traces), taken in an extra pass
  of the same compiled kernels without CUDA Graphs (a graph replay has no per-kernel host launch
  to link). The innermost HSN frame (a file of the package, :func:`in_package`; not torch's
  files under a ``.venv`` inside a checkout named ``hsn``) that names one gives the segment
  (:func:`classify_frames`:
  ``units.py`` → ``selected_cells``; the cell, graph, native ops and state updates →
  ``messages``, the P2.5 partition; the router → ``scorer``; E, P, S and the embeddings →
  ``shared_path``; the decoder → ``decode``). ``unattributed`` reports what no rule matched.
  ``bytes`` are the step's allocator peak and reserve (P0.6); they are not separable per segment
  inside one window, and DRAM traffic is not measured (Nsight Compute would be needed).
* ``decision`` (the last key, and the last line printed): ``L2 justified: yes|no — <segment> is
  <x>% of step time after L1`` for the larger of the L2 targets ``messages`` (csr_messages,
  gather_concat, gated_delta_update) and ``selected_cells`` (grouped_unit_mlp); ``yes`` iff its
  share of the L1 step is at least 20 % (the user's threshold, 7 Nov 2026: a segment below the G2
  gate's 20 % gain cannot deliver it alone, A.2). An L0 file states the same line for L0 shares,
  marked as not the decision.
* ``card``: the GPU compute processes seen before and after (``idle`` or ``shared``).

**MPS** (``profile:m_series``). PyTorch 2.15 records no MPS device events, so after the same timed
steps the GPU is measured through Xcode's Metal System Trace (:mod:`hsn.profiler.metal_trace`,
``xcrun xctrace record --attach`` on this process) in two traced steps:

* a *natural* plain step: ``train_step.device_ms`` is the GPU busy time of the step (the union of
  its command buffers' GPU intervals), ``gpu_busy_fraction`` that over the steady p50;
* a *serialised* plain step under :class:`~hsn.profiler.metal_trace.Serialised` (every aten op
  waits for the GPU and is recorded with its span on the monotonic clock; at L1 also every
  inductor-generated Metal kernel, through :func:`serialise_mps_kernels`, in a fresh compiled
  window like CUDA's attribution pass). Each GPU interval lies inside the span of the dispatch
  that issued it, and the dispatch is classified as it runs, by the CUDA rules
  (:class:`MPSClassifier`: the forward code's Python frames, a backward op through the autograd
  sequence number of the forward op that created its node, inductor provenance for compiled
  kernels and extern calls, the optimizer and the gradient accumulation); the segment's
  ``device_ms`` is its serialised GPU time. ``launches_host`` counts the dispatches that ran GPU
  work (aten ops at the dispatcher, compiled Metal kernel calls); ``kernels_device`` the step's
  command buffers (one per dispatch when serialised). The trace maps onto
  :func:`time.monotonic_ns` exactly, so no clock fit is needed, and ``torch.profiler`` is not
  used (its Python-stack events over ~1.6e5 serialised dispatches took tens of GB). The
  serialised step is recorded in chunks of ``--trace-chunk`` dispatches
  (:class:`~hsn.profiler.metal_trace.ChunkedTrace`: one trace of the whole step is gigabytes);
  traces go to ``--trace-dir`` and are deleted after export unless ``--keep-trace``.

Serialised GPU time runs 2.9-3.5 times the natural step's GPU busy time (measured on an M4 Pro:
the floor of a serialised dispatch is 2.6 µs, so it is not a fixed cost per command buffer; the
GPU likely clocks down between serialised dispatches). On MPS a segment's ``device_ms`` is
therefore *proportional*: its share of the serialised GPU time applied to the natural step's GPU
busy time; ``share_of_step`` and the decision line divide that by the steady step p50 as on CUDA,
so the shares sum to the GPU busy fraction. ``device_ms_serialised`` keeps the raw serialised
time, and ``metal_trace.decision_raw_cuda_formula`` the line from it (the user's choice of
7 Nov 2026). The serving step's host launches are ``-1`` on MPS (not traced).
``card`` on MPS lists the busiest CPU processes from ``ps`` before and after (``shared`` when
another process used half a core or more).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import hsn  # noqa: E402
from hsn.config import RunConfig, config_hash, load_config  # noqa: E402
from hsn.device import memory_stats, reset_peak, resolve, seed_everything, synchronize  # noqa: E402
from hsn.model.hsn import HSN  # noqa: E402
from hsn.profiler import metal_trace  # noqa: E402
from hsn.profiler.launches import HOST_LAUNCH_CALLS  # noqa: E402
from hsn.runner import build_model, load_datasets, make_trainer  # noqa: E402
from hsn.train import compile as compile_mod  # noqa: E402
from hsn.train.tbptt import TBPTTTrainer  # noqa: E402
from torch.autograd import DeviceType  # type: ignore[attr-defined]  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

DEVICE_ACTIVITIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
EVAL_PREFIX = "autograd::engine::evaluate_function: "
SEGMENTS = (
    "shared_path",
    "scorer",
    "messages",
    "selected_cells",
    "decode",
    "loss",
    "optimizer",
    "grad_accumulation",
    "other",
)
L2_TARGETS = ("messages", "selected_cells")
L2_THRESHOLD = 0.20
FRAME = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
PROFILER_FRAME = re.compile(r"^(.*?)\((\d+)\): (\S+)$")
#: MPS (module docstring)
MPS_BUSY_SHARE = 50.0  # % CPU of another process that marks the Mac as shared
MPS_LAUNCH_DEFINITIONS = {
    "launches_host": "dispatches of the serialised plain step that ran GPU work: aten ops at the "
    "dispatcher (a TorchDispatchMode) and inductor-generated Metal kernel calls",
    "kernels_device": "GPU command buffers of the serialised plain step (Metal System Trace "
    "metal-gpu-intervals; one per dispatch there)",
    "device_ms": "GPU busy time of a natural (unserialised) plain step: the union of its command "
    "buffers' GPU intervals",
}


# --------------------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------------------


def build(args: argparse.Namespace) -> tuple[RunConfig, Any, HSN, TBPTTTrainer, Any, Any]:
    mode = "none" if args.level == "L0" else "reduce-overhead"
    overrides = [
        f"profile={args.profile}",
        f"hardware.compile_mode={mode}",
        "train.stats_every=1000000",
        "train.eval_every=1000000",
        "train.ckpt_every=1000000",
        *args.override,
    ]
    cfg = load_config(Path(args.config), overrides=overrides)
    if args.level == "L1" and cfg.hardware.compile_mode == "none":
        raise ValueError("--level L1 needs a compiled hardware.compile_mode")
    if args.level == "L0" and cfg.hardware.compile_mode != "none":
        raise ValueError("--level L0 is hardware.compile_mode=none")
    ctx = resolve(cfg.hardware)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    if not isinstance(model, HSN):
        raise TypeError("profile_step.py profiles the HSN model")
    run_dir = Path(tempfile.mkdtemp(prefix="hsn_profile_step_"))
    trainer = make_trainer(model, cfg, ctx, run_dir=run_dir)
    assert isinstance(trainer, TBPTTTrainer)
    trainer.evaluate = lambda *a, **k: {}  # type: ignore[method-assign]
    trainer.save = lambda *a, **k: run_dir / "unused.pt"  # type: ignore[method-assign]
    train_ds, dev_ds = load_datasets(cfg)
    return cfg, ctx, model, trainer, train_ds, dev_ds


def gpu_processes() -> list[dict[str, str]]:
    """The compute processes ``nvidia-smi`` lists (pid, memory, name); empty without a GPU."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            rows.append({"pid": parts[0], "memory": parts[1], "name": parts[2]})
    return rows


def card_status(seen: list[list[dict[str, str]]]) -> str:
    """``shared`` when another process held a compute context (a Python process, or anything of
    200 MiB or more) in any listing; the audit tool's rule."""
    own = str(os.getpid())

    def counts(row: dict[str, str]) -> bool:
        if row["pid"] == own:
            return False
        if "python" in row["name"].lower():
            return True
        try:
            return int(row["memory"].split()[0]) >= 200
        except (ValueError, IndexError):
            return True

    return "shared" if any(counts(r) for listing in seen for r in listing) else "idle"


def parse_ps(out: str, min_cpu: float = 10.0) -> list[dict[str, str]]:
    """The processes of a ``ps -Ao pid=,pcpu=,rss=,comm=`` listing using at least ``min_cpu`` % CPU,
    busiest first: pid, % CPU, RSS KiB and the command's base name only (the profiles are
    published; full paths name the user's home and applications)."""
    rows = []
    for line in out.strip().splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and float(parts[1]) >= min_cpu:
            name = Path(parts[3].strip()).name
            rows.append({"pid": parts[0], "cpu": parts[1], "rss_kib": parts[2], "name": name})
    rows.sort(key=lambda r: float(r["cpu"]), reverse=True)
    return rows


def cpu_processes() -> list[dict[str, str]]:
    """The busy processes now (:func:`parse_ps`), for the MPS ``card``: an Apple GPU has no
    compute-process listing, and the CPU feeds a launch-bound step."""
    out = subprocess.run(
        ["ps", "-Ao", "pid=,pcpu=,rss=,comm="],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    ).stdout
    return parse_ps(out)


def machine_status(seen: list[list[dict[str, str]]]) -> str:
    """``shared`` when another process (not this one, not ``ps``) used ``MPS_BUSY_SHARE`` % of a
    core or more in any listing; ``idle`` otherwise."""
    own = str(os.getpid())

    def busy(row: dict[str, str]) -> bool:
        if row["pid"] == own or row["name"] == "ps":
            return False
        return float(row["cpu"]) >= MPS_BUSY_SHARE

    return "shared" if any(busy(r) for listing in seen for r in listing) else "idle"


def git_revision() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return rev + ("+dirty" if dirty else "")


# --------------------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def step_range(trainer: TBPTTTrainer) -> Iterator[list[Any]]:
    """A ``STEP::<k>`` profiler range from each ``optimizer.zero_grad`` (the first call of a step)
    to the next one (the audit tool's convention)."""
    opened: list[Any] = []
    original = trainer.optimizer.zero_grad
    count = [0]

    def zero_grad(*args: Any, **kwargs: Any) -> Any:
        if opened:
            opened[-1].__exit__(None, None, None)
        count[0] += 1
        rf = record_function(f"STEP::{count[0]}")
        rf.__enter__()
        opened.append(rf)
        return original(*args, **kwargs)

    trainer.optimizer.zero_grad = zero_grad  # type: ignore[method-assign]
    try:
        yield opened
    finally:
        synchronize(trainer.ctx.device)
        if opened:
            opened[-1].__exit__(None, None, None)
        trainer.optimizer.zero_grad = original  # type: ignore[method-assign]


def percentile(xs: list[float], q: float) -> float:
    ordered = sorted(xs)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


# --------------------------------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------------------------------


#: the hsn package's files as inductor's stack traces and live Python frames name them: the
#: directory of the package that was imported (``REPO/python/hsn`` when this script imports it
#: first; a worktree sharing another checkout's ``.venv`` gets that checkout's through ``hsn.pth``
#: when something imported ``hsn`` before)
PACKAGE_ROOT = str(Path(hsn.__path__[0])).replace("\\", "/") + "/"
#: and as ``torch.profiler(with_stack=True)`` names them: it strips each file's ``sys.path`` entry
#: (``torch.profiler.python_tracer._prefix_regex``), the package's parent
PACKAGE_RELATIVE = "hsn/"


def in_package(path: str) -> bool:
    """Whether a frame's file is in the hsn package. Not any path through a directory named
    ``hsn``: the repo's ``.venv`` lies under one when the checkout is named ``hsn``, and torch has
    files named like HSN's (``autograd/graph.py``, ``fx/graph.py``)."""
    norm = path.replace("\\", "/")
    return norm.startswith(PACKAGE_ROOT) or norm.startswith(PACKAGE_RELATIVE)


def _hsn_frames(frames: list[tuple[str, int, str]]) -> list[tuple[str, str]]:
    """``(file name, function)`` of the frames inside the ``hsn`` package, outermost first."""
    out = []
    for path, _, fn in frames:
        norm = path.replace("\\", "/")
        if in_package(norm):
            out.append((Path(norm).name, fn))
    return out


#: The functions of hsn.py by the segment of the work they do themselves (the model's own glue).
_HSN_FUNCTIONS = {
    "_position": "shared_path",
    "_window_prologue": "shared_path",
    "_shared_path": "shared_path",
    "_score": "scorer",
    "_select": "scorer",
    "_update": "messages",
    "_sparse_round": "messages",
    "_masked_round": "messages",
    "_write_dense": "messages",
    "_cell_dense": "messages",
}


def classify_frames(frames: list[tuple[str, int, str]]) -> str | None:
    """The segment of a kernel from the HSN frames of its source stack (outermost first; ``None``
    when there is none): the innermost HSN frame that names a segment decides (the unit bank
    called from the cell is ``selected_cells``, the cell's own ops ``messages``, the P2.5
    partition), except that ``E`` / ``P`` / ``S`` / ``D``'s MLP code serves the router too (a
    ``model.scorer=mlp`` scorer) and the decoder in the epilogue."""
    files = _hsn_frames(frames)
    if not files:
        return None
    names = {name for name, _ in files}
    for name, fn in reversed(files):  # innermost first
        match name:
            case "units.py":
                return "selected_cells"
            case "cell.py" | "graph.py" | "native.py" | "state.py":
                return "messages"
            case "router.py":
                return "scorer"
            case "losses.py":
                return "loss"
            case "encoder.py" | "embeddings.py":
                if "router.py" in names:
                    return "scorer"
                return "decode" if "_window_epilogue" in {f for _, f in files} else "shared_path"
            case "hsn.py" if fn in _HSN_FUNCTIONS:
                return _HSN_FUNCTIONS[fn]
    return "other"


def frames_of_stack_traces(stack_traces: list[str]) -> list[tuple[str, int, str]]:
    frames: list[tuple[str, int, str]] = []
    for trace in stack_traces:
        frames.extend((f, int(line), fn) for f, line, fn in FRAME.findall(trace))
    return frames


def chain(event: Any) -> list[Any]:
    out = []
    parent = event.cpu_parent
    while parent is not None:
        out.append(parent)
        parent = parent.cpu_parent
    return out


def frames_of_chain(event: Any) -> list[tuple[str, int, str]]:
    """The Python frames ``torch.profiler(with_stack=True)`` records as parent events of an op
    (named ``file(line): function``), outermost first."""
    frames = []
    for anc in [event, *chain(event)]:
        match = PROFILER_FRAME.match(anc.name)
        if match:
            frames.append((match.group(1), int(match.group(2)), match.group(3)))
    return frames[::-1]


Traces = list[list[tuple[str, int, str]]]  # one frame list (outermost first) per origin node


def alias_extern_provenance() -> None:
    """Make inductor's extern-kernel provenance reachable from the profiler (torch 2.15).

    With ``trace.provenance_tracking_to_timeline`` every extern call (cuBLAS ``mm`` / ``addmm`` /
    ``bmm`` / ``baddbmm``) runs through a profiling wrapper ``extern_kernels_<op>_<n>`` and
    inductor aliases the kernel's provenance to that name, but it looks the kernel up by its bare
    name while the provenance is recorded under ``<name>:<node handle>``, so the alias never
    happens. Within one graph the wrappers are generated in the order the handles were assigned,
    so the next not yet aliased ``<name>:<handle>`` of the same op is the wrapper's; this patch
    does that lookup (the per-graph tables are reset by inductor between graphs)."""
    import torch._inductor.debug as inductor_debug

    original = inductor_debug.alias_kernel_provenance
    aliased: set[tuple[int, str]] = set()

    def alias(kernel_name: str, alias_name: str) -> None:
        table = inductor_debug._inductor_kernel_stack_trace
        if kernel_name in table:
            original(kernel_name, alias_name)
            return
        for key in list(table):
            if key.startswith(kernel_name + ":") and (id(table), key) not in aliased:
                aliased.add((id(table), key))
                table[alias_name] = table[key]
                return

    inductor_debug.alias_kernel_provenance = alias


Provenance = dict[tuple[bool, str], Traces]  # (backward graph?, kernel or wrapper name)


def kernel_provenance() -> Provenance:
    """Inductor's kernel provenance (``trace.provenance_tracking_to_timeline``) over every
    compiled graph: ``(backward?, name)`` → the stack traces of the kernel's origin nodes, for a
    Triton kernel by its name and for an extern kernel by its profiling wrapper's name (made
    reachable by :func:`alias_extern_provenance`). Names restart per graph, so the forward and
    the backward graph are kept apart."""
    import ast

    from torch._inductor.debug import get_kernel_information_jsons

    out: Provenance = {}
    for graph_key, graph in get_kernel_information_jsons().items():
        try:
            backward = bool(ast.literal_eval(graph_key)[1])
        except (ValueError, SyntaxError, IndexError, TypeError):
            continue
        for key, info in (graph or {}).items():
            raw = info.get("stack_traces") if isinstance(info, dict) else info
            traces = [frames_of_stack_traces([trace]) for trace in (raw or [])]
            name = key.partition(":")[0]  # a Triton kernel's handle; wrappers have none
            out.setdefault((backward, name), []).extend(t for t in traces if t)
    return out


def innermost_label(frames: list[tuple[str, int, str]]) -> str:
    """``file:line function`` of the innermost HSN frame (for the ``other`` breakdown)."""
    for path, line, fn in reversed(frames):
        norm = path.replace("\\", "/")
        if in_package(norm):
            return f"{Path(norm).name}:{line} {fn}"
    return "(no hsn frame)"


def split(traces: Traces, *, extern: bool = False) -> dict[str, float]:
    """A kernel's share per segment: a fused Triton kernel's origin-node stacks classified one by
    one and the kernel split evenly over them (inductor fuses across segment boundaries); an
    extern kernel (a GEMM) by its own node, the first trace (the others are its inputs')."""
    if extern:
        traces = traces[:1]
    segments = [classify_frames(t) or "other" for t in traces]
    if not segments:
        return {}
    weight = 1.0 / len(segments)
    shares: dict[str, float] = collections.defaultdict(float)
    for segment in segments:
        shares[segment] += weight
    return dict(shares)


def attribute(events: list[Any], provenance: Provenance) -> dict[str, Any]:
    """Device kernels and device time of the first ``STEP::`` range, per segment."""
    cpu = [e for e in events if e.device_type == DeviceType.CPU]
    steps = sorted(
        (e for e in cpu if e.name.startswith("STEP::")), key=lambda e: e.time_range.start
    )
    if not steps:
        raise RuntimeError("no STEP:: range in the profile")
    first = steps[0]
    # the P0.6 link: a device event's linked_correlation_id names a frontend op
    frontend = {e.id: e for e in cpu if e.linked_correlation_id == 0 and e.external_id != 0}
    seq_frames: dict[int, list[tuple[str, int, str]]] = {}
    for e in cpu:
        seq = getattr(e, "sequence_nr", -1)
        if seq is None or seq < 0:
            continue
        if any(a.name.startswith(EVAL_PREFIX) for a in [e, *chain(e)]):
            continue  # a backward op
        frames = frames_of_chain(e)
        if frames:
            seq_frames.setdefault(int(seq), frames)

    def backward_segment(links: list[Any], i: int) -> dict[str, float] | None:
        """A kernel of the backward pass: the forward op that recorded its autograd node (the
        node's ``sequence_nr``) gives the segment; the leaf accumulation and the engine's sums
        of incoming gradients are ``grad_accumulation``."""
        node_name = links[i].name[len(EVAL_PREFIX) :]
        if node_name.endswith("AccumulateGrad"):
            return {"grad_accumulation": 1.0}
        node = next((c for c in links[:i] if c.name == node_name), None)
        if node is None:
            return {"grad_accumulation": 1.0}  # the engine's input-buffer sums
        seq = node.sequence_nr
        frames = seq_frames.get(int(seq)) if seq is not None and seq >= 0 else None
        return one(classify_frames(frames)) if frames else None

    def one(segment: str | None) -> dict[str, float] | None:
        return None if segment is None else {segment: 1.0}

    def triton_traces(device_name: str) -> Traces | None:
        key = device_name.split("(")[0].strip()
        found = [provenance[(bw, key)] for bw in (False, True) if (bw, key) in provenance]
        return [t for traces in found for t in traces] if found else None

    def extern_traces(links: list[Any]) -> Traces | None:
        backward = any(a.name == "CompiledFunctionBackward" for a in links)
        for anc in links:
            traces = provenance.get((backward, anc.name.rpartition(": ")[2]))
            if traces is not None:
                return traces
        return None

    def other_label(host: Any | None, device_name: str) -> str:
        traces = triton_traces(device_name)
        if traces is None and host is not None:
            traces = extern_traces([host, *chain(host)])
        if traces:
            labels = sorted({innermost_label(t) for t in traces if classify_frames(t) == "other"})
            return " | ".join(labels[:3]) or "(fused)"
        if host is not None:
            return f"eager {innermost_label(frames_of_chain(host))} / {host.name[:40]}"
        return f"(no host) {device_name[:50]}"

    def segment_of(host: Any | None, device_name: str) -> dict[str, float] | None:
        traces = triton_traces(device_name)
        if traces is not None:  # a compiled Triton kernel
            return split(traces)
        if host is None:
            return None
        links = [host, *chain(host)]
        traces = extern_traces(links)
        if traces is not None:  # an extern kernel (GEMM) of a compiled graph
            return split(traces, extern=True)
        for i, anc in enumerate(links):  # innermost first
            match anc.name:
                case name if name.startswith("Optimizer.") or "clip_grad" in name:
                    return {"optimizer": 1.0}
                case name if name.startswith(EVAL_PREFIX):
                    return backward_segment(links, i)
        return one(classify_frames(frames_of_chain(host)))

    kernels: collections.Counter[str] = collections.Counter()  # fractional where split
    device_us: collections.Counter[str] = collections.Counter()
    other_us: collections.Counter[str] = collections.Counter()  # what "other" holds
    unattributed: collections.Counter[str] = collections.Counter()
    unattributed_us = 0.0
    total = 0
    for ev in events:
        if ev.device_type != DeviceType.CUDA or str(ev.activity_type) not in DEVICE_ACTIVITIES:
            continue
        host = frontend.get(ev.linked_correlation_id)
        start = host.time_range.start if host is not None else ev.time_range.start
        if not (first.time_range.start <= start <= first.time_range.end):
            continue
        total += 1
        duration = ev.time_range.end - ev.time_range.start
        shares = segment_of(host, ev.name)
        if not shares:
            label = ev.name[:60] if host is None else f"{host.name[:50]} / {ev.name[:40]}"
            unattributed[label] += 1
            unattributed_us += duration
            shares = {"other": 1.0}
        for segment, weight in shares.items():
            kernels[segment] += weight
            device_us[segment] += weight * duration
        if "other" in shares:
            other_us[other_label(host, ev.name)] += shares["other"] * duration
    return {
        "kernels": dict(kernels),
        "device_us": dict(device_us),
        "kernels_total": total,
        "unattributed_kernels": sum(unattributed.values()),
        "unattributed_device_us": unattributed_us,
        "unattributed_top": dict(unattributed.most_common(10)),
        "other_top_ms": {k: round(v / 1e3, 3) for k, v in other_us.most_common(12)},
    }


def count_launches(events: list[Any]) -> tuple[int, int, float]:
    """(host launch calls, device events, device µs) inside the first ``STEP::`` range."""
    cpu = [e for e in events if e.device_type == DeviceType.CPU]
    first = min((e for e in cpu if e.name.startswith("STEP::")), key=lambda e: e.time_range.start)
    lo, hi = first.time_range.start, first.time_range.end
    by_id = {e.id: e for e in cpu}
    host = sum(1 for e in cpu if e.name in HOST_LAUNCH_CALLS and lo <= e.time_range.start <= hi)
    dev = 0
    dev_us = 0.0
    for ev in events:
        if ev.device_type != DeviceType.CUDA or str(ev.activity_type) not in DEVICE_ACTIVITIES:
            continue
        link = by_id.get(ev.linked_correlation_id)
        start = link.time_range.start if link is not None else ev.time_range.start
        if lo <= start <= hi:
            dev += 1
            dev_us += ev.time_range.end - ev.time_range.start
    return host, dev, dev_us


# --------------------------------------------------------------------------------------------------
# Counts and attribution: CUDA
# --------------------------------------------------------------------------------------------------


def profile_cuda(
    args: argparse.Namespace,
    trainer: TBPTTTrainer,
    model: HSN,
    ctx: Any,
    train_ds: Any,
    dev_ds: Any,
    horizon: int,
) -> tuple[int, int, float, dict[str, Any]]:
    """Host launches, device kernels and device µs of one profiled plain step, and the
    attribution pass (module docstring)."""
    # one profiled plain step for the launch counts (the next step; then a stats step closes)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with step_range(trainer):
            trainer.fit(train_ds, dev_ds, horizon + 2)
    host, dev, dev_us = count_launches(list(prof.events()))

    # attribution pass: the same kernels without CUDA Graphs (L1), with Python stacks
    provenance: Provenance = {}
    if args.level == "L1":
        window = trainer._compiled
        assert window is not None
        plain = compile_mod.CompiledWindow(model, ctx, window.mode, window.backend)
        plain.cuda_graphs = False
        plain._block = torch.compile(
            compile_mod._own_code(compile_mod._block),
            mode="default",
            fullgraph=True,
            dynamic=False,
        )
        trainer._compiled = plain
        # the graph window is done (timing and counts are taken): release its CUDA Graph pools,
        # or the plain window's activations come on top of them (B32 did not fit in 12 GB)
        from torch._inductor.cudagraph_trees import reset_cudagraph_trees

        del window
        reset_cudagraph_trees()
        torch.cuda.empty_cache()
        trainer.fit(train_ds, dev_ds, horizon + 4)  # compile the plain blocks
        provenance = kernel_provenance()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True
    ) as prof_stack:
        with step_range(trainer):
            trainer.fit(train_ds, dev_ds, horizon + (6 if args.level == "L1" else 4))
    attribution = attribute(list(prof_stack.events()), provenance)
    return host, dev, dev_us, attribution


# --------------------------------------------------------------------------------------------------
# Counts and attribution: MPS (Metal System Trace)
# --------------------------------------------------------------------------------------------------

#: Python files whose frames mark the optimizer step (on CUDA: the ``Optimizer.step#`` and
#: ``clip_grad`` profiler ranges); the fp16 GradScaler's unscale and step count there too
OPTIMIZER_FILES = (
    "torch/optim/",
    "torch/nn/utils/clip_grad.py",
    "torch/amp/grad_scaler.py",
    "hsn/train/lazy_adam.py",
)
#: inductor's profiling wrappers of extern calls (``trace.provenance_tracking_to_timeline``)
EXTERN_PREFIX = "extern_kernels_"
Shares = dict[str, float] | None


class MetalKernelGate:
    """The switch of :func:`serialise_mps_kernels`, the kernel calls it serialised, and the
    callbacks that record them (``record(name, start_ns, end_ns, backward)``) and roll the trace
    over (``on_dispatch()``)."""

    def __init__(self) -> None:
        self.on = False
        self.calls = 0
        self.record: Any = None
        self.on_dispatch: Any = None


def serialise_mps_kernels() -> MetalKernelGate:
    """Wrap every Metal library inductor compiles from now on (``runtime_utils.compile_mps_shader``,
    which its batched ``compile_mps_shaders`` and the generated modules call). While the gate is
    on, a kernel call waits for the device first (pending work is not the kernel's), runs, waits
    again and is recorded with its span on the monotonic clock and its graph direction (backward
    while the autograd engine evaluates a node, i.e. the compiled backward); off, a call goes
    straight through. Profile script only."""
    import torch._inductor.runtime.runtime_utils as runtime_utils

    original = runtime_utils.compile_mps_shader
    gate = MetalKernelGate()

    class Kernel:
        def __init__(self, name: str, fn: Any) -> None:
            self.name = name
            self.fn = fn

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if not gate.on:
                return self.fn(*args, **kwargs)
            torch.mps.synchronize()
            backward = torch._C._current_autograd_node() is not None
            start = time.monotonic_ns()
            out = self.fn(*args, **kwargs)
            torch.mps.synchronize()
            end = time.monotonic_ns()
            gate.calls += 1
            if gate.record is not None:
                gate.record(self.name, start, end, backward)
            if gate.on_dispatch is not None:
                gate.on_dispatch()
            return out

    class Library:
        def __init__(self, lib: Any) -> None:
            self._lib = lib

        def __getattr__(self, name: str) -> Any:
            attr = getattr(self._lib, name)
            return Kernel(name, attr) if callable(attr) else attr

    def compile_mps_shader(source: str) -> Any:
        return Library(original(source))

    runtime_utils.compile_mps_shader = compile_mps_shader
    return gate


def mps_kernel_provenance(graph_keys: set[str]) -> Provenance:
    """:func:`kernel_provenance` over the graphs ``graph_keys`` only. Inductor names its Metal
    kernels ``generated_kernel_<n>`` in every graph, so a name is unique only within one forward and
    one backward graph: two of the graphs naming one kernel with different origins raise."""
    import ast

    from torch._inductor.debug import get_kernel_information_jsons

    out: Provenance = {}
    owner: dict[tuple[bool, str], str] = {}
    for graph_key, graph in get_kernel_information_jsons().items():
        if graph_key not in graph_keys:
            continue
        backward = bool(ast.literal_eval(graph_key)[1])
        mine: Provenance = {}
        for key, info in (graph or {}).items():
            raw = info.get("stack_traces") if isinstance(info, dict) else info
            traces = [frames_of_stack_traces([trace]) for trace in (raw or [])]
            mine.setdefault((backward, key.partition(":")[0]), []).extend(t for t in traces if t)
        for name, traces in mine.items():
            if name in out and out[name] != traces:
                raise RuntimeError(
                    f"inductor graphs {owner[name]} and {graph_key} both name a "
                    f"{'backward' if name[0] else 'forward'} kernel {name[1]!r} with different "
                    "origins; the MPS attribution cannot tell them apart"
                )
            out.setdefault(name, traces)
            owner.setdefault(name, graph_key)
    if not out:
        raise RuntimeError("the attribution window compiled no inductor graph with provenance")
    return out


def python_frames(skip: int = 1) -> list[tuple[str, int, str]]:
    """The calling thread's Python stack as ``(file, line, function)``, outermost first, without
    the innermost ``skip`` frames."""
    frame: Any = sys._getframe(skip)
    out = []
    while frame is not None:
        out.append((frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name))
        frame = frame.f_back
    return out[::-1]


class MPSClassifier:
    """The segment of one serialised MPS dispatch by :func:`attribute`'s rules, applied in the op's
    own context (``Serialised.classify``) instead of from a recorded profile:

    * an extern call of a compiled graph (an ``extern_kernels_*`` frame on the stack): inductor
      provenance of that graph direction (backward while a node is being evaluated);
    * a backward op (``torch._C._current_autograd_node()``): ``AccumulateGrad`` is
      ``grad_accumulation``, and so are the engine's sums of incoming gradients (the input-buffer
      adds after a node's own computation, which run while that node is still current: a
      post-hook registered on the node the first time one of its ops is seen marks where its own
      computation ends); any other op of a node takes the segment of the forward op that created
      the node, by autograd sequence number (recorded here when the forward op ran:
      ``torch._C._autograd._get_sequence_nr() − 1`` names the node created just before the op
      reaches the dispatcher's Python key);
    * an op on another thread with no node: the engine's other gradient sums;
    * the optimizer step (:data:`OPTIMIZER_FILES` frames): ``optimizer``;
    * otherwise :func:`classify_frames` of the Python stack (its frames in the hsn package,
      :func:`in_package`).

    A rule that finds nothing leaves the dispatch unattributed (``other``), as on CUDA."""

    def __init__(self, provenance: Provenance) -> None:
        self.provenance = provenance
        self.main = threading.get_ident()
        self.forward: dict[int, str] = {}
        self._local = threading.local()  # per engine thread: the node in evaluation, and whether
        # its own computation is done (what follows is input-buffer sums)

    def _applied(self, key: tuple[str, int], grad_inputs: Any, grad_outputs: Any) -> None:
        if getattr(self._local, "node", None) == key:
            self._local.done = True

    def __call__(self, func: Any) -> tuple[Shares, str]:
        node = torch._C._current_autograd_node()
        frames = python_frames(3)  # from the frame that issued the op
        label = f"{innermost_label(frames)} / {func}"
        for _, _, fn in reversed(frames):
            if fn.startswith(EXTERN_PREFIX):
                traces = self.provenance.get((node is not None, fn))
                if traces is not None:
                    return split(traces, extern=True), label
        if node is not None:
            name = node.name()
            if name.endswith("AccumulateGrad"):
                return {"grad_accumulation": 1.0}, label
            key = (name, int(node._sequence_nr()))
            if getattr(self._local, "node", None) != key:
                self._local.node, self._local.done = key, False
                node.register_hook(functools.partial(self._applied, key))
            if self._local.done:
                return {"grad_accumulation": 1.0}, f"input-buffer sum after {name} / {func}"
            segment = self.forward.get(key[1])
            return (None if segment is None else {segment: 1.0}), f"{name} / {func}"
        if threading.get_ident() != self.main:
            return {"grad_accumulation": 1.0}, label
        paths = [path.replace("\\", "/") for path, _, _ in frames]
        if any(part in path for path in paths for part in OPTIMIZER_FILES):
            return {"optimizer": 1.0}, label
        found = classify_frames(frames)
        if found is None:
            return None, label
        self.forward.setdefault(int(torch._C._autograd._get_sequence_nr()) - 1, found)
        return {found: 1.0}, label

    def kernel(self, name: str, start_ns: int, end_ns: int, backward: bool) -> metal_trace.Dispatch:
        """The record of a compiled Metal kernel call (:func:`serialise_mps_kernels`)."""
        traces = self.provenance.get((backward, name))
        if traces is None:
            raise RuntimeError(
                f"no inductor provenance for the {'backward' if backward else 'forward'} Metal "
                f"kernel {name!r} of the attribution window"
            )
        return metal_trace.Dispatch(
            start_ns=start_ns, end_ns=end_ns, what=(split(traces), f"compiled {name}")
        )


@contextlib.contextmanager
def step_marks(
    trainer: TBPTTTrainer, *, on_mark: Any = None, sync: bool = True
) -> Iterator[list[int]]:
    """:func:`time.monotonic_ns` at each ``optimizer.zero_grad`` (a step's first call), after a
    device synchronisation when ``sync`` (so a step's GPU work lies between its two marks);
    ``on_mark(k)`` runs after mark ``k`` (1-based)."""
    marks: list[int] = []
    original = trainer.optimizer.zero_grad

    def zero_grad(*args: Any, **kwargs: Any) -> Any:
        if sync:
            synchronize(trainer.ctx.device)
        marks.append(time.monotonic_ns())
        if on_mark is not None:
            on_mark(len(marks))
        return original(*args, **kwargs)

    trainer.optimizer.zero_grad = zero_grad  # type: ignore[method-assign]
    try:
        yield marks
    finally:
        trainer.optimizer.zero_grad = original  # type: ignore[method-assign]


def attribute_mps(
    records: list[metal_trace.Dispatch],
    intervals: list[metal_trace.GPUInterval],
    lo_ns: int,
    hi_ns: int,
) -> dict[str, Any]:
    """:func:`attribute` for the serialised MPS step ``[lo_ns, hi_ns)`` (monotonic): each GPU
    interval goes to the dispatch whose span contains it (:func:`metal_trace.assign`), and takes
    that dispatch's segment shares (``what = (shares, label)``)."""
    window = sorted((r for r in records if lo_ns <= r.start_ns < hi_ns), key=lambda r: r.start_ns)
    step = [g for g in intervals if lo_ns <= g.start_ns < hi_ns]
    if not step:
        raise RuntimeError("the serialised step holds no GPU interval of this process")
    found = metal_trace.assign(
        [(g.start_ns, g.end_ns) for g in step], [(r.start_ns, r.end_ns) for r in window]
    )
    kernels: collections.defaultdict[str, float] = collections.defaultdict(float)
    device_us: collections.defaultdict[str, float] = collections.defaultdict(float)
    other_us: collections.Counter[str] = collections.Counter()
    unattributed: collections.Counter[str] = collections.Counter()
    unattributed_us = 0.0
    unassigned_us = 0.0
    dispatched: set[int] = set()
    for g, index in zip(step, found.index, strict=True):
        duration = g.duration_ns / 1e3
        shares: Shares = None
        label = "(no dispatch)"
        if index is None:
            unassigned_us += duration
        else:
            dispatched.add(index)
            shares, label = window[index].what
        if not shares:
            unattributed[label[:90]] += 1
            unattributed_us += duration
            shares = {"other": 1.0}
        for segment, weight in shares.items():
            kernels[segment] += weight
            device_us[segment] += weight * duration
        if "other" in shares:
            other_us[label[:90]] += round(shares["other"] * duration)  # µs
    total_us = sum(g.duration_ns for g in step) / 1e3
    return {
        "kernels": dict(kernels),
        "device_us": dict(device_us),
        "kernels_total": len(step),
        "unattributed_kernels": sum(unattributed.values()),
        "unattributed_device_us": unattributed_us,
        "unattributed_top": dict(unattributed.most_common(10)),
        "other_top_ms": {k: round(v / 1e3, 3) for k, v in other_us.most_common(12)},
        "dispatches": len(dispatched),
        "gpu_us": total_us,
        "contained": found.contained,
        "partial": found.partial,
        "unassigned": found.unassigned,
        "assigned_fraction": 1.0 - unassigned_us / total_us if total_us > 0 else 0.0,
    }


def combine_repeats(
    repeats: list[dict[str, Any]], natural_busy_us: list[float]
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """The MPS attribution of ``n`` serialised repeats (:func:`attribute_mps`) and ``n`` natural
    steps' GPU busy µs. A repeat's proportional segment µs are its serialised mix (segment over
    the repeat's serialised GPU total) applied to the median natural GPU busy time; the result's
    ``device_us`` is the median over repeats per segment, ``device_us_serialised`` the median of
    the raw serialised µs, ``gpu_us`` the median serialised total. Counts and the breakdowns
    (kernels, unattributed, other) are the first repeat's: the dispatches of every repeat are the
    same ops. Returns the combined attribution and each repeat's proportional µs."""
    if not repeats or len(repeats) != len(natural_busy_us):
        raise ValueError(f"{len(repeats)} serialised repeats for {len(natural_busy_us)} natural")
    busy = statistics.median(natural_busy_us)
    by_repeat = []
    for att in repeats:
        total = att["gpu_us"]
        factor = busy / total if total > 0 else 0.0
        by_repeat.append({seg: att["device_us"].get(seg, 0.0) * factor for seg in SEGMENTS})
    combined = dict(repeats[0])
    combined["device_us"] = {seg: statistics.median(r[seg] for r in by_repeat) for seg in SEGMENTS}
    combined["device_us_serialised"] = {
        seg: statistics.median(att["device_us"].get(seg, 0.0) for att in repeats)
        for seg in SEGMENTS
    }
    combined["gpu_us"] = statistics.median(att["gpu_us"] for att in repeats)
    return combined, by_repeat


def profile_mps(
    args: argparse.Namespace,
    trainer: TBPTTTrainer,
    model: HSN,
    ctx: Any,
    train_ds: Any,
    dev_ds: Any,
    horizon: int,
) -> tuple[int, int, float, dict[str, Any], dict[str, Any]]:
    """The MPS counterpart of :func:`profile_cuda` (module docstring): ``--trace-repeats`` natural
    traced plain steps, then (at L1 after compiling a fresh window whose Metal kernels are
    serialised) as many serialised ones in chunked traces, combined by :func:`combine_repeats`.
    Returns host dispatches, command buffers, the median natural GPU busy µs, the attribution and
    the trace record."""
    pid = os.getpid()
    n = args.trace_repeats
    if n < 1:
        raise ValueError(f"--trace-repeats must be >= 1, got {n}")
    keep = bool(args.keep_trace)
    base = Path(args.trace_dir) if args.trace_dir else Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="hsn_metal_trace_", dir=base))
    stem = f"{args.profile}_{args.level}" + (f"_{args.tag}" if args.tag else "")

    # n natural plain steps under one recording (it stops at the last one's end, before the
    # closing stats step)
    natural_rec = metal_trace.Recording(work / f"{stem}_natural.trace", pid)

    def stop_natural(k: int) -> None:
        if k == n + 1:
            natural_rec.stop()

    with natural_rec:
        with step_marks(trainer, on_mark=stop_natural) as marks:
            trainer.fit(train_ds, dev_ds, trainer.step + n + 1)
        synchronize(ctx.device)
    if len(marks) < n + 1:
        raise RuntimeError(f"the natural traced fit ran {len(marks)} steps; {n + 1} expected")
    natural_dir = work / f"{stem}_natural_export"
    natural_dir.mkdir(exist_ok=True)
    natural = metal_trace.gpu_intervals(natural_rec.out, pid, natural_dir)
    windows = list(zip(marks[:n], marks[1 : n + 1], strict=True))
    natural_busy_ns = [metal_trace.busy_ns(natural, lo, hi) for lo, hi in windows]
    natural_cbs = [sum(1 for g in natural if lo <= g.start_ns < hi) for lo, hi in windows]
    if not all(natural_cbs):
        raise RuntimeError(f"a natural traced step holds no GPU interval ({natural_cbs})")
    if not keep:
        shutil.rmtree(natural_rec.out, ignore_errors=True)
        shutil.rmtree(natural_dir, ignore_errors=True)

    # L1: the same kernels in a fresh window, compiled through the kernel wrapper (the CUDA
    # attribution pass's counterpart); the timed window never carried the wrapper
    gate: MetalKernelGate | None = None
    provenance: Provenance = {}
    if args.level == "L1":
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.debug import get_kernel_information_jsons

        window = trainer._compiled
        assert window is not None
        gate = serialise_mps_kernels()
        # generated modules are cached by path in memory: reload them so that their Metal
        # libraries compile through the wrapper
        PyCodeCache.cache_clear()
        before = set(get_kernel_information_jsons())
        # the timed window's own compiler (MPS: the fusion cap of compile.py; no CUDA Graphs
        # exist on MPS, so this is the same compile, through the wrapper)
        plain = compile_mod.CompiledWindow(model, ctx, window.mode, window.backend)
        trainer._compiled = plain
        del window
        trainer.fit(train_ds, dev_ds, trainer.step + 2)  # compile the plain blocks
        provenance = mps_kernel_provenance(set(get_kernel_information_jsons()) - before)

    # n serialised plain steps in chunked traces, each dispatch classified as it runs (the
    # closing stats step runs plain and untraced)
    classifier = MPSClassifier(provenance)
    mode = metal_trace.Serialised(metal_trace.mps_sync, classify=classifier)
    chunks = metal_trace.ChunkedTrace(
        work / f"{stem}_serialised", pid, chunk=args.trace_chunk, keep=keep, mode=mode
    )
    mode.on_dispatch = chunks.tick
    if gate is not None:
        gate.on_dispatch = chunks.tick
        gate.record = lambda name, start, end, backward: mode.records.append(
            classifier.kernel(name, start, end, backward)
        )

    def switch(k: int) -> None:
        if k == n + 1:  # the plain steps are done: the stats step runs unserialised, untraced
            mode.enabled = False
            if gate is not None:
                gate.on = False
            chunks.close()

    with chunks:
        if gate is not None:
            gate.on = True
        with step_marks(trainer, on_mark=switch) as serial_marks:
            with mode:
                trainer.fit(train_ds, dev_ds, trainer.step + n + 1)
        if gate is not None:
            gate.on = False
    if len(serial_marks) < n + 1:
        raise RuntimeError(f"the serialised fit ran {len(serial_marks)} steps; {n + 1} expected")
    if gate is not None and gate.calls == 0:
        raise RuntimeError(
            "the serialised L1 step ran no inductor Metal kernel through the wrapper (the window "
            "ran eager, or its libraries were compiled before serialise_mps_kernels)"
        )
    serial_windows = list(zip(serial_marks[:n], serial_marks[1 : n + 1], strict=True))
    repeats = [attribute_mps(mode.records, chunks.intervals, lo, hi) for lo, hi in serial_windows]
    attribution, by_repeat = combine_repeats(repeats, [b / 1e3 for b in natural_busy_ns])
    encoders = metal_trace.encoders_per_command_buffer(
        [g for g in chunks.intervals if serial_windows[0][0] <= g.start_ns < serial_windows[-1][1]]
    )
    natural_busy_ms = [b / 1e6 for b in natural_busy_ns]
    serial_ms = [att["gpu_us"] / 1e3 for att in repeats]

    def ms(values: list[float]) -> list[float]:
        return [round(v, 3) for v in values]

    trace_info: dict[str, Any] = {
        "xctrace": metal_trace.version(),
        "template": metal_trace.TEMPLATE,
        "repeats": n,
        "natural": {
            "step_wall_ms": ms([(hi - lo) / 1e6 for lo, hi in windows]),
            "command_buffers": natural_cbs,
            "gpu_busy_ms": ms(natural_busy_ms),
            "gpu_busy_ms_median": round(statistics.median(natural_busy_ms), 3),
        },
        "serialised": {
            "step_wall_ms": ms([(hi - lo) / 1e6 for lo, hi in serial_windows]),
            "command_buffers": [att["kernels_total"] for att in repeats],
            "gpu_ms": ms(serial_ms),
            "dispatches_with_gpu_work": [att["dispatches"] for att in repeats],
            "aten_ops": mode.calls,
            "compiled_kernel_calls": gate.calls if gate is not None else 0,
            "encoders_per_command_buffer": {str(k): v for k, v in encoders.items()},
            "trace_chunks": chunks.chunks,
            "dispatches_per_chunk": args.trace_chunk,
            "chunk_pause_s": round(chunks.pause_s, 1),
            "intervals_contained": [att["contained"] for att in repeats],
            "intervals_partial": [att["partial"] for att in repeats],
            "intervals_unassigned": [att["unassigned"] for att in repeats],
            "assigned_fraction": [round(att["assigned_fraction"], 4) for att in repeats],
        },
        "serialised_over_natural": [
            round(s / statistics.median(natural_busy_ms), 3) for s in serial_ms
        ],
        "device_us_by_repeat": by_repeat,  # main() turns these into shares and decision lines
        "traces_kept": str(work) if keep else None,
        "ktrace_temp_bytes_deleted": natural_rec.ktrace_bytes + chunks.ktrace_bytes,
    }
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return (
        int(attribution["dispatches"]),
        int(attribution["kernels_total"]),
        statistics.median(natural_busy_ns) / 1e3,
        attribution,
        trace_info,
    )


# --------------------------------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------------------------------


def serving(
    cfg: RunConfig, ctx: Any, model: HSN, level: str, batch: int, tokens: int
) -> dict[str, Any]:
    """Per-token latency of :func:`compile_step` at this level on a copy of the model."""
    import copy

    replica = copy.deepcopy(model).eval()
    mode = "none" if level == "L0" else cfg.hardware.compile_mode
    step = compile_mod.compile_step(replica, ctx, mode)
    state = replica.init_state(batch)
    generator = torch.Generator(device="cpu").manual_seed(0)
    xs = torch.randint(0, cfg.model.vocab_size, (tokens + 8, batch), generator=generator)
    cs = torch.randint(0, cfg.model.n_context, (tokens + 8, batch), generator=generator)
    xs, cs = xs.to(ctx.device), cs.to(ctx.device)
    autocast = (
        torch.autocast(ctx.device.type, ctx.autocast_dtype)
        if ctx.autocast_dtype is not None
        else contextlib.nullcontext()
    )
    times: list[float] = []
    with autocast:
        for t in range(8):  # warm-up: compile, record
            kwargs = {"prev_tokens": None if t == 0 else xs[t - 1]} if cfg.model.key_address else {}
            step(xs[t], cs[t], state, **kwargs)
        synchronize(ctx.device)
        for t in range(8, 8 + tokens):
            kwargs = {"prev_tokens": xs[t - 1]} if cfg.model.key_address else {}
            started = time.perf_counter()
            logits = step(xs[t], cs[t], state, **kwargs)
            float(logits[0, 0])  # the token is read: a synchronising latency
            times.append(time.perf_counter() - started)
        if ctx.device.type == "cuda":
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                kwargs = {"prev_tokens": xs[-1]} if cfg.model.key_address else {}
                step(xs[0], cs[0], state, **kwargs)
                synchronize(ctx.device)
    host = -1  # MPS: no host launch calls exist to count, and the serving step is not traced
    if ctx.device.type == "cuda":
        host = sum(
            1
            for e in prof.events()
            if e.device_type == DeviceType.CPU and e.name in HOST_LAUNCH_CALLS
        )
    return {
        "batch": batch,
        "tokens": tokens,
        "latency_ms_p50": round(1e3 * percentile(times, 0.5), 4),
        "latency_ms_p95": round(1e3 * percentile(times, 0.95), 4),
        "launches_host_per_token": host,
        "compile_s": round(getattr(step, "compile_seconds", 0.0), 3),
    }


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------


def segment_table(
    attribution: dict[str, Any], step_us: float
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Per segment: device kernels, device ms and the share of the step p50 ``step_us``. On MPS
    the attribution's device µs are the proportional medians of :func:`combine_repeats` and its
    raw serialised medians are kept as ``device_ms_serialised``; on CUDA the attributed device
    time as it is."""
    serialised = attribution.get("device_us_serialised")
    segments: dict[str, dict[str, float]] = {}
    shares: dict[str, float] = {}
    for name in SEGMENTS:
        us = attribution["device_us"].get(name, 0.0)
        shares[name] = us / step_us if step_us > 0 else 0.0
        segments[name] = {
            "kernels_device": round(attribution["kernels"].get(name, 0.0), 1),
            "device_ms": round(us / 1e3, 3),
            "share_of_step": round(shares[name], 4),
        }
        if serialised is not None:
            segments[name]["device_ms_serialised"] = round(serialised.get(name, 0.0) / 1e3, 3)
    return segments, shares


def decision_line(shares: dict[str, float], level: str) -> str:
    target = max(L2_TARGETS, key=lambda s: shares.get(s, 0.0))
    share = shares.get(target, 0.0)
    verdict = "yes" if share >= L2_THRESHOLD else "no"
    line = f"L2 justified: {verdict} — {target} is {100 * share:.1f}% of step time after L1"
    if level == "L0":
        line = f"(L0 shares, not the decision) {line.replace('after L1', 'at L0')}"
    return line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--level", choices=("L0", "L1"), required=True)
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--tag", default="")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--serve-tokens", type=int, default=200)
    ap.add_argument("--out", default=str(REPO / "profiles"))
    ap.add_argument(
        "--trace-dir",
        default="",
        metavar="DIR",
        help="MPS: where the Metal System Traces and their exports go (default: the temp dir)",
    )
    ap.add_argument(
        "--keep-trace",
        action="store_true",
        help="MPS: keep the traces (in --trace-dir) for Instruments instead of deleting them",
    )
    ap.add_argument(
        "--trace-repeats",
        type=int,
        default=3,
        metavar="N",
        help="MPS: natural and serialised traced steps; the shares are the median over them",
    )
    ap.add_argument(
        "--trace-chunk",
        type=int,
        default=20_000,
        metavar="N",
        help="MPS: serialised dispatches per Metal System Trace chunk",
    )
    args = ap.parse_args()
    if args.level == "L1":
        import torch._functorch.config as functorch_config
        import torch._inductor.config as inductor_config

        inductor_config.trace.provenance_tracking_to_timeline = True
        inductor_config.triton.unique_kernel_names = True
        # a cached graph carries no provenance, and the compile time is a cold compile's
        inductor_config.fx_graph_cache = False
        functorch_config.enable_autograd_cache = False
        alias_extern_provenance()
    seen = [gpu_processes()]
    cpu_seen = [cpu_processes()] if sys.platform == "darwin" else []
    cfg, ctx, model, trainer, train_ds, dev_ds = build(args)
    if ctx.device.type not in ("cuda", "mps"):
        raise RuntimeError(
            "profile_step.py profiles a device queue: CUDA (torch.profiler kernel events) or MPS "
            f"(Xcode's Metal System Trace); resolved {ctx.device}, which has neither"
        )
    if ctx.device.type == "mps":
        ok, reason = metal_trace.available()
        if not ok:  # before the timed steps, not after them
            raise RuntimeError(f"profile_step.py on MPS needs Metal System Trace: {reason}")
    T, B = cfg.train.unroll, cfg.train.batch

    # warm-up (compile and record at L1), then the timed plain steps
    started = time.perf_counter()
    reset_peak(ctx.device)  # the peak covers the warm-up too: L1's CUDA Graph pools are made there
    warm = trainer.fit(train_ds, dev_ds, args.warm)
    warm_s = time.perf_counter() - started
    horizon = args.warm + args.repeats * args.steps + 1  # the last step of a fit is a stats step
    history = trainer.fit(train_ds, dev_ds, horizon)
    memory = memory_stats(ctx.device)
    timed = [
        r for r in history.steps if args.warm < r.step <= args.warm + args.repeats * args.steps
    ]
    seconds = [r.seconds for r in timed]
    blocks = [seconds[i * args.steps : (i + 1) * args.steps] for i in range(args.repeats)]
    p50 = percentile(seconds, 0.5)
    tokens = sum(r.tokens for r in timed)

    trace_info: dict[str, Any] | None = None
    if ctx.device.type == "cuda":
        host, dev, dev_us, attribution = profile_cuda(
            args, trainer, model, ctx, train_ds, dev_ds, horizon
        )
    else:
        host, dev, dev_us, attribution, trace_info = profile_mps(
            args, trainer, model, ctx, train_ds, dev_ds, horizon
        )

    serve = [
        serving(cfg, ctx, model, args.level, 1, args.serve_tokens),
        serving(cfg, ctx, model, args.level, B, args.serve_tokens),
    ]
    seen.append(gpu_processes())
    if cpu_seen:
        cpu_seen.append(cpu_processes())
    mps = ctx.device.type == "mps"

    step_us = 1e6 * p50
    segments, shares = segment_table(attribution, step_us)
    if trace_info is not None:
        by_repeat = trace_info.pop("device_us_by_repeat")
        trace_info["shares_by_repeat"] = {
            seg: [round(r[seg] / step_us, 4) if step_us > 0 else 0.0 for r in by_repeat]
            for seg in SEGMENTS
        }
        trace_info["decision_by_repeat"] = [
            decision_line({seg: r[seg] / step_us for seg in SEGMENTS}, args.level)
            for r in by_repeat
        ]
        raw_shares = {
            k: attribution["device_us_serialised"].get(k, 0.0) / step_us if step_us > 0 else 0.0
            for k in SEGMENTS
        }
        trace_info["share_method"] = (
            "proportional, median of repeats (the user's choices, 7 Nov 2026): per repeat, "
            "serialised segment GPU ms x median natural step GPU busy / that repeat's "
            "serialised GPU total; device_ms the median over repeats; share = that / step p50"
        )
        trace_info["decision_raw_cuda_formula"] = decision_line(raw_shares, args.level)
    if mps:
        card: dict[str, Any] = {
            "status": machine_status(cpu_seen),
            "rule": f"shared when another process used >= {MPS_BUSY_SHARE:.0f}% CPU in a listing "
            "(processes at >= 10% CPU listed, base names only)",
            "cpu_processes": cpu_seen,
        }
        compile_note = (
            "compile (cold: inductor's graph caches off); MPS has no CUDA Graphs; never inside "
            "the steady-state numbers"
        )
        launch_definitions: dict[str, Any] = dict(MPS_LAUNCH_DEFINITIONS)
        method = (
            "MPS: Metal System Trace (xctrace) GPU intervals of a serialised plain step, each "
            "assigned to the dispatch range containing it (aten ops at the dispatcher"
            + (", inductor Metal kernels in a fresh compiled window" if args.level == "L1" else "")
            + "); the range classified as on CUDA (Python stacks, backward through sequence_nr"
            + (", inductor kernel provenance" if args.level == "L1" else "")
            + "); "
        )
    else:
        card = {"status": card_status(seen), "gpu_processes": seen}
        compile_note = (
            "compile (cold: inductor's graph caches off) and CUDA Graph recording; never "
            "inside the steady-state numbers"
        )
        launch_definitions = {
            "launches_host": sorted(HOST_LAUNCH_CALLS),
            "kernels_device": sorted(DEVICE_ACTIVITIES),
        }
        method = (
            "L0: torch.profiler Python stacks (backward through sequence_nr); "
            if args.level == "L0"
            else "L1: inductor kernel provenance (origin-node stack traces) in a pass of the "
            "same kernels without CUDA Graphs; eager parts by Python stacks; "
        )
    result: dict[str, Any] = {
        "level": args.level,
        "profile": args.profile,
        "tag": args.tag,
        "git_revision": git_revision(),
        "torch": torch.__version__,
        "config": args.config,
        "overrides": args.override,
        "config_hash": config_hash(cfg),
        "routing": cfg.model.routing,
        "gate_mode": cfg.train.gate_mode,
        "batch": B,
        "unroll": T,
        "precision": cfg.train.precision,
        "autocast": str(ctx.autocast_dtype),
        "compile_mode": cfg.hardware.compile_mode,
        "block": compile_mod.block_size(T) if args.level == "L1" else None,
        "card": card,
        "compile": {
            "warm_steps": args.warm,
            "warm_wall_s": round(warm_s, 3),
            "compile_seconds": round(float(warm.measured.get("compile_seconds", 0.0)), 3),
            "compile_steps": int(warm.measured.get("compile_steps", 0.0)),
            "startup_step_s": round(warm.steps[0].seconds, 3) if warm.steps else None,
            "note": compile_note,
        },
        "train_step": {
            "steps": len(seconds),
            "repeats": args.repeats,
            "wall_ms_p50": round(1e3 * p50, 3),
            "wall_ms_p95": round(1e3 * percentile(seconds, 0.95), 3),
            "wall_ms_p50_by_repeat": [round(1e3 * percentile(b, 0.5), 3) for b in blocks if b],
            "tokens_per_s": round(tokens / sum(seconds), 1) if seconds else None,
            "launches_host": host,
            "kernels_device": dev,
            "device_ms": round(dev_us / 1e3, 3),
            "gpu_busy_fraction": round(dev_us / step_us, 4) if step_us > 0 else None,
            "bytes": {"peak": int(memory["peak"]), "reserved": int(memory["reserved"])},
            "launch_definitions": launch_definitions,
        },
        "serving_step": serve,
        "segments": segments,
        "attribution": {
            "method": method + "segment from the HSN frames of the stack",
            "kernels_total": attribution["kernels_total"],
            "unattributed_kernels": attribution["unattributed_kernels"],
            "unattributed_device_ms": round(attribution["unattributed_device_us"] / 1e3, 3),
            "unattributed_top": attribution["unattributed_top"],
            "other_top_ms": attribution["other_top_ms"],
            "bytes_note": "allocator peak/reserved per step only; not separable per segment inside "
            "one window; DRAM traffic not measured"
            + (
                "; MPS: peak is a sampled high-water mark (hsn.device.memory_stats), reserved is "
                "torch.mps.driver_allocated_memory()"
                if mps
                else ""
            ),
        },
    }
    if trace_info is not None:
        result["metal_trace"] = trace_info
    result["decision"] = decision_line(shares, args.level)  # the last key
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.profile}_{args.level}" + (f"_{args.tag}" if args.tag else "") + ".json"
    path = out / name
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(path)
    print(json.dumps({k: v for k, v in result.items() if k != "card"}, indent=1)[:4000])
    print(result["decision"])


if __name__ == "__main__":
    main()
