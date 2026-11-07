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
  to link). The innermost HSN frame that names one gives the segment (:func:`classify_frames`:
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
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from hsn.config import RunConfig, config_hash, load_config  # noqa: E402
from hsn.device import memory_stats, reset_peak, resolve, seed_everything, synchronize  # noqa: E402
from hsn.model.hsn import HSN  # noqa: E402
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


def _hsn_frames(frames: list[tuple[str, int, str]]) -> list[tuple[str, str]]:
    """``(file name, function)`` of the frames inside the ``hsn`` package, outermost first."""
    out = []
    for path, _, fn in frames:
        norm = path.replace("\\", "/")
        if "hsn/" in norm and ("/hsn/" in norm or norm.startswith("hsn/")):
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
        if "/hsn/" in norm or norm.startswith("hsn/"):
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
            step(xs[t], cs[t], state)
        synchronize(ctx.device)
        for t in range(8, 8 + tokens):
            started = time.perf_counter()
            logits = step(xs[t], cs[t], state)
            float(logits[0, 0])  # the token is read: a synchronising latency
            times.append(time.perf_counter() - started)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step(xs[0], cs[0], state)
            synchronize(ctx.device)
    host = sum(
        1 for e in prof.events() if e.device_type == DeviceType.CPU and e.name in HOST_LAUNCH_CALLS
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
    cfg, ctx, model, trainer, train_ds, dev_ds = build(args)
    if ctx.device.type != "cuda":
        raise RuntimeError(f"profile_step.py profiles CUDA launches; resolved {ctx.device}")
    T, B = cfg.train.unroll, cfg.train.batch

    # warm-up (compile and record at L1), then the timed plain steps
    started = time.perf_counter()
    warm = trainer.fit(train_ds, dev_ds, args.warm)
    warm_s = time.perf_counter() - started
    horizon = args.warm + args.repeats * args.steps + 1  # the last step of a fit is a stats step
    reset_peak(ctx.device)
    history = trainer.fit(train_ds, dev_ds, horizon)
    memory = memory_stats(ctx.device)
    timed = [
        r for r in history.steps if args.warm < r.step <= args.warm + args.repeats * args.steps
    ]
    seconds = [r.seconds for r in timed]
    blocks = [seconds[i * args.steps : (i + 1) * args.steps] for i in range(args.repeats)]
    p50 = percentile(seconds, 0.5)
    tokens = sum(r.tokens for r in timed)

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

    serve = [
        serving(cfg, ctx, model, args.level, 1, args.serve_tokens),
        serving(cfg, ctx, model, args.level, B, args.serve_tokens),
    ]
    seen.append(gpu_processes())

    step_us = 1e6 * p50
    segments = {}
    shares: dict[str, float] = {}
    for name in SEGMENTS:
        us = attribution["device_us"].get(name, 0.0)
        shares[name] = us / step_us if step_us > 0 else 0.0
        segments[name] = {
            "kernels_device": round(attribution["kernels"].get(name, 0.0), 1),
            "device_ms": round(us / 1e3, 3),
            "share_of_step": round(shares[name], 4),
        }
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
        "card": {"status": card_status(seen), "gpu_processes": seen},
        "compile": {
            "warm_steps": args.warm,
            "warm_wall_s": round(warm_s, 3),
            "compile_seconds": round(float(warm.measured.get("compile_seconds", 0.0)), 3),
            "compile_steps": int(warm.measured.get("compile_steps", 0.0)),
            "startup_step_s": round(warm.steps[0].seconds, 3) if warm.steps else None,
            "note": "compile (cold: inductor's graph caches off) and CUDA Graph recording; never "
            "inside the steady-state numbers",
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
            "launch_definitions": {
                "launches_host": sorted(HOST_LAUNCH_CALLS),
                "kernels_device": sorted(DEVICE_ACTIVITIES),
            },
        },
        "serving_step": serve,
        "segments": segments,
        "attribution": {
            "method": (
                "L0: torch.profiler Python stacks (backward through sequence_nr); "
                if args.level == "L0"
                else "L1: inductor kernel provenance (origin-node stack traces) in a pass of the "
                "same kernels without CUDA Graphs; eager parts by Python stacks; "
            )
            + "segment from the HSN frames of the stack",
            "kernels_total": attribution["kernels_total"],
            "unattributed_kernels": attribution["unattributed_kernels"],
            "unattributed_device_ms": round(attribution["unattributed_device_us"] / 1e3, 3),
            "unattributed_top": attribution["unattributed_top"],
            "other_top_ms": attribution["other_top_ms"],
            "bytes_note": "allocator peak/reserved per step only; not separable per segment inside "
            "one window; DRAM traffic not measured",
        },
        "decision": decision_line(shares, args.level),
    }
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
