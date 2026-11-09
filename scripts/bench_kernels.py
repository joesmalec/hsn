#!/usr/bin/env python3
"""Compare CUDA adapters with compiled native L1, under scripts/run_when_idle.py.

Includes casts, CSR/padded-index adapters, fresh functional state outputs and autograd.
Forward and forward+backward are reported separately; paired samples alternate order.
This is a per-op benchmark, not integrated training evidence or an automatic enable policy.
Use --op and the hsn/<op>/<backend>/<mode> NVTX range for narrow Nsight runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import torch
from hsn.kernels import cuda_ops, native, spec, testvectors
from hsn.model.graph import CellGraph
from hsn.model.state import CellGraphState
from hsn.model.units import UnitBank
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]


def make_case(op: str, size: str, precision: torch.dtype):
    vector = testvectors.load_vector(op, size)
    dims = vector.dims
    t = {n: x.to("cuda").detach() for n, x in vector.role("input").items()}
    for value in t.values():
        if value.is_floating_point():
            value.requires_grad_(True)
    idx = t.get("cell_idx")
    if idx is not None:
        idx = idx.long()
    parameters: tuple[Tensor, ...]
    if op == "gather_concat":
        # Messages are in the autocast dtype; state/u/eps stay fp32.
        t["m"] = t["m"].to(precision).detach().requires_grad_(True)
        parameters = tuple(t[n] for n in ("u", "eps", "h", "m"))

        def invoke(backend):
            return (backend.gather_concat(*parameters, idx),)
    elif op == "grouped_unit_mlp":
        bank = UnitBank(dims["M"], dims["d"], dims["H"], "gelu", init="fan_in").cuda()
        with torch.no_grad():
            for name in ("W1", "b1", "W2", "b2"):
                getattr(bank, name).copy_(t[name])
        units = t["unit_idx"].long()
        parameters = (t["v"], bank.W1, bank.b1, bank.W2, bank.b2)

        def invoke(backend):
            return (backend.grouped_unit_mlp(t["v"], units, bank, static=True),)
    elif op == "csr_messages":
        cpu = vector.tensors
        degree = int((cpu["indptr"][1:] - cpu["indptr"][:-1]).max())
        graph = CellGraph(dims["N"], degree, 0, edge_norm="sigmoid")
        graph.set_topology(
            cpu["indptr"].long(), cpu["indices"].long(), torch.zeros(dims["E"], dtype=torch.bool)
        )
        graph = graph.cuda()
        projection = nn.Linear(dims["d"], dims["d"], bias=False).cuda()
        with torch.no_grad():
            projection.weight.copy_(t["V"])
        padded = torch.zeros_like(graph._pad_src, dtype=torch.float32)
        padded.flatten()[graph._edge_pos] = t["a"].detach()
        padded.requires_grad_(True)
        parameters = (t["h"], padded, projection.weight)

        def invoke(backend):
            return (
                backend.csr_messages(
                    t["h"], graph, projection, idx, weights=padded, validate=False
                ),
            )
    else:
        for name in ("c", "b"):
            t[name] = t[name].to(precision).detach().requires_grad_(True)
        state = CellGraphState(
            t["h"],
            t["q"],
            t["age"].long(),
            torch.zeros(dims["B"], device="cuda", dtype=torch.int64),
        )
        valid = t["valid"].bool()
        parameters = tuple(t[n] for n in ("h", "q", "c", "b"))

        def invoke(backend):
            result = backend.gated_delta_update(state, idx, t["c"], t["b"], valid, validate=False)
            return result.h, result.q

    def function(backend):
        def forward():
            with torch.autocast("cuda", dtype=precision, enabled=precision != torch.float32):
                return invoke(backend)

        return forward

    return function(native), function(cuda_ops), parameters


def measure(fn: Callable[[], object], name: str) -> dict[str, float]:
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    before = time.perf_counter()
    torch.cuda.nvtx.range_push(name)
    try:
        start.record()
        result = fn()
        stop.record()
        stop.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
    wall = 1000 * (time.perf_counter() - before)
    del result
    return {"device_ms": start.elapsed_time(stop), "wall_ms": wall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=tuple(spec.OPS))
    parser.add_argument("--size", choices=("tiny", "ref", "odd"), default="ref")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--out", type=Path, default=ROOT / "evidence/P2.10/bench.json")
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 3 or not args.out.resolve().is_relative_to(ROOT):
        parser.error("warmup >= 1, samples >= 3, and output inside this worktree required")
    precision = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.precision
    ]
    cuda_ops.prepare_cuda(0)
    torch.manual_seed(0)
    rows = []
    for op in (args.op,) if args.op else spec.OPS:
        native_fn, fused_fn, parameters = make_case(op, args.size, precision)
        compile_start = time.perf_counter()
        functions = {
            "L1": torch.compile(native_fn, fullgraph=True, mode="reduce-overhead"),
            "CUDA": torch.compile(fused_fn, fullgraph=True, mode="reduce-overhead"),
        }
        # Verify values/gradients before measuring. GPU suites remain the full correctness gate.
        expected = functions["L1"]()
        cotangents = tuple(torch.ones_like(t) for t in expected)
        expected_grad = torch.autograd.grad(expected, parameters, cotangents)
        expected = tuple(t.detach().clone() for t in expected)
        expected_grad = tuple(t.detach().clone() for t in expected_grad)
        actual = functions["CUDA"]()
        actual_grad = torch.autograd.grad(actual, parameters, cotangents)
        tol = spec.tolerance(precision)
        for got, ref in zip((*actual, *actual_grad), (*expected, *expected_grad), strict=True):
            torch.testing.assert_close(got, ref, atol=tol.atol, rtol=tol.rtol)
        torch.cuda.synchronize()
        startup = time.perf_counter() - compile_start
        del actual, expected, actual_grad, expected_grad
        for backward in (False, True):
            mode = "forward_backward" if backward else "forward"

            def step(
                name,
                functions=functions,
                backward=backward,
                parameters=parameters,
                cotangents=cotangents,
            ):
                torch.compiler.cudagraph_mark_step_begin()
                if not backward:
                    with torch.no_grad():
                        return functions[name]()
                result = functions[name]()
                return torch.autograd.grad(result, parameters, cotangents)

            warm_start = time.perf_counter()
            for _ in range(args.warmup):
                for name in functions:
                    step(name)
            torch.cuda.synchronize()
            warm_seconds = time.perf_counter() - warm_start
            torch.cuda.reset_peak_memory_stats()
            samples = {name: [] for name in functions}
            for i in range(args.samples):
                for name in ("L1", "CUDA") if i % 2 == 0 else ("CUDA", "L1"):
                    samples[name].append(
                        measure(lambda name=name: step(name), f"hsn/{op}/{name}/{mode}")
                    )
            medians = {
                name: {
                    metric: statistics.median(s[metric] for s in data)
                    for metric in ("device_ms", "wall_ms")
                }
                for name, data in samples.items()
            }
            rows.append(
                {
                    "op": op,
                    "mode": mode,
                    "samples": samples,
                    "median": medians,
                    "startup_seconds": startup,
                    "warmup_seconds": warm_seconds,
                    "combined_peak_bytes": torch.cuda.max_memory_allocated(),
                    "cuda_kernel_used": not (
                        op == "gated_delta_update" and precision != torch.float32
                    ),
                    "default_backend": "native",
                    "enable_decision": "pending review of repeated and integrated measurements",
                }
            )
    report = {
        "observed_at": datetime.now(UTC).isoformat(),
        "size": args.size,
        "precision": args.precision,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "driver_inventory": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "build": json.loads((ROOT / "build/cuda_build.json").read_text()),
        "scope": "per-op adapters versus compiled native, including casts and index conversion",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
