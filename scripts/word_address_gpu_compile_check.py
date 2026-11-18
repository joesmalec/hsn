#!/usr/bin/env python
"""The compiled event loop with the word address on the card: does it compile once and step?

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python scripts/word_address_gpu_compile_check.py \\
        --spec <specs>/mem_events2.yaml [--arm ts_memT_rd32_wev_wexact] [--steps 60] \\
        [--warm 10] [--max-recompiles 1] [--override KEY=VALUE ...]

Runs from the repository root (the data paths are relative to it). It trains one arm of the
spec exactly as the spec configures it (``profile=rtx3060``, ``hardware.compile_mode=
reduce-overhead``, TinyStories, B32, T256) through the real
:class:`~hsn.train.tbptt.TBPTTTrainer` for ``--steps`` steps, with evaluation and checkpoint
writes stubbed out and nothing written to the run database, and prints:

* the time to the first step (from ``fit``; the compile and the first CUDA-graph recordings);
* steps per second over the steps after ``--warm`` (the last step is left out: the trainer
  makes it a statistics step, with one extra compiled pass per loss term);
* dynamo's graph count (``torch._dynamo.utils.counters["stats"]["unique_graphs"]``) and the
  recompiles (graphs after the first), with their reasons on stderr (``TORCH_LOGS`` recompiles);
* the event slots per window (``E``, the batch's largest event count rounded up to the block)
  and the steps during which the compiled window compiled or recorded.

Exit status 1 when the recompiles exceed ``--max-recompiles`` (default 1), 2 without CUDA.
Before the fix of ``hsn.train.compile._block_columns`` every new ``E`` was a new graph, and the
mem_events2 arms compiled for 20 minutes on an RTX 3070 without a step (the CPU reproduction:
eight graphs in six steps, then ``FailOnRecompileLimitHit``). Run it alone on the card: the
step rate is a timing.
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--spec", required=True, help="the sweep spec (mem_events2.yaml)")
    p.add_argument("--arm", default="ts_memT_rd32_wev_wexact", help="the arm name in the spec")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--warm", type=int, default=10, help="steps left out of the step rate")
    p.add_argument("--max-recompiles", type=int, default=1)
    p.add_argument("--override", action="append", default=[], help="KEY=VALUE, after the arm's")
    return p


def _gpu_processes() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in out.strip().splitlines()[1:] if line.strip()]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.warm < 1 or args.steps < args.warm + 3:
        raise SystemExit("--steps must exceed --warm by at least 3 (the last step is left out)")
    if not torch.cuda.is_available():
        print("no CUDA device: this check runs on the card", file=sys.stderr)
        return 2
    os.chdir(REPO)
    torch._logging.set_logs(recompiles=True, graph_breaks=True)
    from hsn.config import load_config
    from hsn.device import resolve, seed_everything
    from hsn.experiments.spec import expand, load_spec
    from hsn.model.hsn import HSN
    from hsn.runner import build_model, load_datasets
    from hsn.train.tbptt import TBPTTTrainer
    from torch._dynamo.utils import counters

    logging.basicConfig(level=logging.WARNING)
    spec, _ = load_spec(args.spec)
    arms = {a.name: a for a in expand(spec, repo=REPO)}
    if args.arm not in arms:
        raise SystemExit(f"--arm {args.arm!r} is not in the spec: {sorted(arms)}")
    arm = arms[args.arm]
    overrides = [
        *arm.overrides,
        "train.eval_every=1000000",
        "train.ckpt_every=1000000",
        *args.override,
    ]
    cfg = load_config(REPO / arm.config_path, overrides)
    if cfg.hardware.device != "cuda" or cfg.hardware.compile_mode == "none":
        raise SystemExit("the arm must run compiled on CUDA (profile=rtx3060, reduce-overhead)")
    print(
        f"arm {arm.name} ({arm.arm_id}): write_events={cfg.model.write_events} "
        f"word_address={cfg.model.word_address} compile_mode={cfg.hardware.compile_mode} "
        f"B={cfg.train.batch} T={cfg.train.unroll} dynamo recompile_limit="
        f"{torch._dynamo.config.recompile_limit}",
        flush=True,
    )
    before = _gpu_processes()
    print(f"card before: {before or 'no compute processes'}", flush=True)

    ctx = resolve(cfg.hardware)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    if not isinstance(model, HSN):
        raise TypeError("the check trains an HSN model")
    run_dir = Path(tempfile.mkdtemp(prefix="hsn_word_address_check_"))
    trainer = TBPTTTrainer(model, cfg, ctx, run_dir=run_dir)
    trainer.evaluate = lambda *a, **k: {}  # type: ignore[method-assign]
    trainer.save = lambda *a, **k: run_dir / "unused.pt"  # type: ignore[method-assign]
    window = trainer.compiled_window
    if window is None:
        raise SystemExit("the trainer built no compiled window")
    train_ds, dev_ds = load_datasets(cfg)

    slots: collections.Counter[int] = collections.Counter()
    event_slots = type(model)._event_slots

    def counted(self: Any, pro: Any, reset: Any, block: int) -> Any:
        ev = event_slots(self, pro, reset, block)
        slots[int(ev.valid.shape[1])] += 1  # a host shape, no device read
        return ev

    type(model)._event_slots = counted  # type: ignore[method-assign]
    stamps: list[float] = []
    compiled_at: list[int] = []
    last_compile = [0.0]

    def callback(trainer: Any, history: Any) -> bool:
        torch.cuda.synchronize()
        stamps.append(time.perf_counter())
        if window.compile_seconds > last_compile[0]:
            compiled_at.append(int(trainer.step))
            last_compile[0] = window.compile_seconds
        if len(stamps) == 1:
            print(f"first step after {stamps[0] - started:.1f} s", flush=True)
        return False

    started = time.perf_counter()
    trainer.fit(train_ds, dev_ds, args.steps, callback=callback)
    torch.cuda.synchronize()
    total = time.perf_counter() - started
    graphs = int(counters["stats"]["unique_graphs"])
    recompiles = max(0, graphs - 1)
    # stamps[i] is the end of step i + 1; the rate over steps warm + 1 .. steps - 1
    rate_steps = len(stamps) - args.warm
    rate = rate_steps / (stamps[-1] - stamps[args.warm - 1]) if rate_steps > 0 else float("nan")
    print(f"time to first step: {stamps[0] - started:.1f} s")
    print(
        f"steps/s after warm-up: {rate:.2f} ({1000.0 / rate:.1f} ms/step over steps "
        f"{args.warm + 1}-{len(stamps)})"
    )
    print(f"total: {total:.1f} s for {args.steps} steps")
    print(
        f"compiled window: {window.calls} calls, {window.compile_seconds:.1f} s compiling or "
        f"recording, at steps {compiled_at}"
    )
    print(
        f"event slots per window (E: count): {sorted(slots.items())}; mean "
        f"{sum(k * v for k, v in slots.items()) / max(1, sum(slots.values())):.1f}"
    )
    print(f"peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    after = _gpu_processes()
    print(f"card after: {after or 'no compute processes'}")
    print(f"dynamo graphs: {graphs}, recompiles: {recompiles} (bound {args.max_recompiles})")
    if recompiles > args.max_recompiles:
        print("FAIL: the compiled event window recompiled", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
