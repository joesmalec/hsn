#!/usr/bin/env python
"""Post-A1 checks, item 1 of 5 Nov 2026 (second brief): the 5(b) recipe again, under bf16 with the
non-finite-gradient guard, recording the gradient norm of every trainable parameter at every step.

The recipe is the one of ``scripts/g1_harness_dense_supervision.py --tasks delayed --arms sparse
--lr 1e-3 --batch 16`` (delayed/in_range alone, the full training split, ``profile=rtx3060`` so bf16
autocast, the experiment configuration's schedule, warmup, weight decay and clip otherwise), 3,000
steps, dev evaluation every 500 steps. Run it with the interpreter of a checkout that carries the
guard (``Trainer._guard_nonfinite``); the script refuses one that does not.

Per-parameter norms are taken inside ``torch.nn.utils.clip_grad_norm_`` (wrapped for the trainer's
own call only): after unscaling, before clipping, i.e. the gradients the trainer's ``grad_norm``
measures. Writes ``<out>/run/`` (trainer run directory and checkpoint), ``history.json``,
``param_grad_norms.npy`` ``[steps, n_params]`` with ``param_names.json``, and ``summary.json``:
dev accuracy per evaluation, the guard's skip count, the global norm's p50 / p99 / max, the share
of clipped steps, the steps above 100, and for every skipped step or step above ``--flag-norm`` the
modules (parameter name minus its last component) with the largest gradient norm.

    python scripts/g1_guard_rerun.py --config configs/experiments/g1_ref_a1.yaml --out <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hsn.config import dump_resolved, load_config
from hsn.data.dataset import ByteDataset
from hsn.device import resolve, seed_everything
from hsn.runner import build_model, make_trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="rtx3060")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flag-norm", type=float, default=1000.0)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(
        args.config,
        overrides=[
            f"profile={args.profile}",
            f"train.batch={args.batch}",
            f"train.micro_batch={args.batch}",
            f"train.lr={args.lr}",
            f"train.steps={args.steps}",
            f"train.eval_every={args.eval_every}",
            f"train.ckpt_every={args.steps}",
            "train.stats_every=100",
            f"seed={args.seed}",
            "run_id=guard_bf16",
            "data.tasks=[delayed]",
            "data.variants=[in_range]",
        ],
    )
    ctx = resolve(cfg.hardware)
    if ctx.autocast_dtype is not torch.bfloat16:
        raise RuntimeError(f"profile {args.profile} autocasts to {ctx.autocast_dtype}, not bf16")
    manifest = Path(str(cfg.data.manifest))
    train = ByteDataset.load(
        manifest, "train", ["delayed"], ["in_range"], expected_unroll=cfg.train.unroll
    )
    dev = ByteDataset.load(
        manifest, "dev", ["delayed"], ["in_range"], expected_unroll=cfg.train.unroll
    )
    seed_everything(cfg.seed)
    model = build_model(cfg)
    run_dir = out / "run"
    trainer = make_trainer(model, cfg, ctx, run_dir=run_dir)
    if not hasattr(trainer, "_guard_nonfinite"):
        raise RuntimeError("this checkout's trainer has no non-finite-gradient guard")
    dump_resolved(cfg, run_dir / "config.resolved.yaml")

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, _ in named]
    plist = [p for _, p in named]
    records: list[np.ndarray] = []
    original = torch.nn.utils.clip_grad_norm_

    def wrapped(parameters: Any, max_norm: Any, *a: Any, **k: Any) -> Any:
        if (
            isinstance(parameters, list)
            and len(parameters) == len(plist)
            and all(x is y for x, y in zip(parameters, plist, strict=True))
        ):
            with torch.no_grad():
                zero = torch.zeros((), device=ctx.device)
                norms = torch.stack(
                    [
                        torch.linalg.vector_norm(p.grad.detach().float())
                        if p.grad is not None
                        else zero
                        for p in plist
                    ]
                )
            records.append(norms.cpu().numpy())
        return original(parameters, max_norm, *a, **k)

    torch.nn.utils.clip_grad_norm_ = wrapped  # type: ignore[assignment]
    started = time.perf_counter()
    try:
        history = trainer.fit(train, dev, args.steps)
    finally:
        torch.nn.utils.clip_grad_norm_ = original
    wall = time.perf_counter() - started

    steps = [vars(s) for s in history.steps]
    if len(records) != len(steps):
        raise RuntimeError(f"{len(records)} norm records for {len(steps)} steps")
    per_param = np.stack(records).astype(np.float32)
    np.save(out / "param_grad_norms.npy", per_param)
    with open(out / "param_names.json", "w") as fh:
        json.dump(names, fh)
    with open(out / "history.json", "w") as fh:
        json.dump(
            {"steps": steps, "evals": history.evals, "measured": history.measured}, fh, default=str
        )

    g = np.array([s["grad_norm"] for s in steps], dtype=np.float64)
    step_ids = [int(s["step"]) for s in steps]
    finite = np.isfinite(g)
    clip = float(cfg.train.grad_clip)
    modules = sorted({n.rsplit(".", 1)[0] if "." in n else n for n in names})
    module_of = [modules.index(n.rsplit(".", 1)[0] if "." in n else n) for n in names]

    def top_modules(row: np.ndarray) -> list[tuple[str, float]]:
        sq = np.zeros(len(modules))
        for j, m in enumerate(module_of):
            sq[m] += float(row[j]) ** 2 if math.isfinite(float(row[j])) else math.inf
        order = np.argsort(-sq)[: args.top]
        return [(modules[i], float(math.sqrt(sq[i]))) for i in order]

    flagged = [
        {
            "step": step_ids[i],
            "grad_norm": float(g[i]),
            "skipped": not bool(finite[i]),
            "loss": float(steps[i]["loss"]),
            "top_modules": top_modules(per_param[i]),
        }
        for i in range(len(g))
        if (not finite[i]) or g[i] > args.flag_norm
    ]
    summary: dict[str, Any] = {
        "config": args.config,
        "recipe": {
            "profile": args.profile,
            "autocast": str(ctx.autocast_dtype),
            "lr": cfg.train.lr,
            "batch": cfg.train.batch,
            "steps": args.steps,
            "schedule": cfg.train.schedule,
            "warmup_steps": cfg.train.warmup_steps,
            "weight_decay": cfg.train.weight_decay,
            "grad_clip": clip,
            "seed": cfg.seed,
        },
        "dev_accuracy": [
            {
                "step": int(e["step"]),
                "delayed/in_range": float(e["per_group"]["delayed/in_range"]["accuracy"]),
                "loss": float(e["per_group"]["delayed/in_range"]["loss"]),
            }
            for e in history.evals
        ],
        "skipped_nonfinite": history.measured.get("skipped_nonfinite_steps"),
        "grad_norm": {
            "n_steps": len(g),
            "n_finite": int(finite.sum()),
            "p50": float(np.percentile(g[finite], 50)),
            "p99": float(np.percentile(g[finite], 99)),
            "max": float(g[finite].max()),
            "argmax_step": step_ids[int(np.flatnonzero(finite)[g[finite].argmax()])],
            "clip_value": clip,
            "n_clipped": int((g > clip).sum()),
            "steps_above_100": [(step_ids[i], float(g[i])) for i in range(len(g)) if g[i] > 100],
        },
        "flag_norm": args.flag_norm,
        "flagged_steps": flagged,
        "final_train_loss": float(steps[-1]["loss"]),
        "wall_s": round(wall, 1),
        "measured": history.measured,
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "measured"}, indent=1))


if __name__ == "__main__":
    main()
