#!/usr/bin/env python
"""G1 diagnosis (first brief, item 6): rerun HSN seed 1 for the first N steps of its original
12,000-step schedule with the non-finite-gradient guard, in fp32 or bf16, logging every step.

    python scripts/g1_seed1_rerun.py --mode bf16 --out evidence/G1/diagnosis/reruns2/bf16

The configuration is the gate's seed-0 resolved configuration with seed=1 (the recipe of record:
lr 1e-3 cosine over train.steps=12,000, warmup 200, batch 16); the trainer is asked to fit only
--steps steps, so the learning-rate schedule is the original one truncated, not a shorter schedule.
--mode fp32 sets the profile's autocast dtype to fp32 (autocast off); bf16 keeps the profile's bf16.
Every step logs grad_norm, state_abs_max (the max |h| over cells and positions of the window) and
the guard's skip counter (train.stats_every=1); history.json holds every record. Writes
<out>/summary.json with the first non-finite gradient norm (if any), the skip count, and the loss
and gradient-norm trajectory around the original divergence step (3,140).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from hsn.config import load_config
from hsn.device import resolve, seed_everything
from hsn.runner import build_model, load_datasets, make_trainer

SOURCE = "/home/joe/hsn-p1.9/runs/g1_rtx3060/g1_ref_a1-rtx3060-s0/config.resolved.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("fp32", "bf16"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=SOURCE)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"seed={args.seed}",
        f"run_id=seed{args.seed}_rerun_{args.mode}",
        "train.stats_every=1",
        "train.eval_every=1000",
        "train.ckpt_every=4000",
    ]
    if args.mode == "fp32":
        overrides += ["hardware.autocast_dtype=fp32", "train.precision=fp32"]
    cfg = load_config(args.config, overrides=overrides)
    assert cfg.train.steps == 12000, cfg.train.steps  # the original schedule's horizon
    ctx = resolve(cfg.hardware)
    train, dev = load_datasets(cfg)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    trainer = make_trainer(model, cfg, ctx, run_dir=out / "run")
    t0 = time.perf_counter()
    history = trainer.fit(train, dev, args.steps)
    wall = time.perf_counter() - t0
    steps = history.steps
    first_nonfinite = next((s.step for s in steps if not math.isfinite(s.grad_norm)), None)
    max_norm = max(s.grad_norm for s in steps if math.isfinite(s.grad_norm))
    around = [
        {
            "step": s.step,
            "loss": s.loss,
            "grad_norm": s.grad_norm,
            "lr": s.lr,
            "state_abs_max": s.stats.get("state_abs_max"),
            "skipped": s.stats.get("skipped_nonfinite"),
        }
        for s in steps
        if 3100 <= s.step <= 3180
    ]
    summary = {
        "mode": args.mode,
        "autocast": history.autocast,
        "seed": cfg.seed,
        "schedule_horizon": cfg.train.steps,
        "steps_run": args.steps,
        "lr_at_3140": next((s.lr for s in steps if s.step == 3140), None),
        "first_nonfinite_grad_step": first_nonfinite,
        "skipped_nonfinite_steps": history.measured.get("skipped_nonfinite_steps"),
        "max_finite_grad_norm": max_norm,
        "n_norm_over_100": sum(
            1 for s in steps if math.isfinite(s.grad_norm) and s.grad_norm > 100
        ),
        "state_abs_max_max": max(s.stats.get("state_abs_max", 0.0) for s in steps),
        "final_loss": steps[-1].loss,
        "evals": history.evals,
        "around_3140": around,
        "wall_s": round(wall, 1),
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    with open(out / "history.json", "w") as fh:
        json.dump(
            {
                "steps": [vars(s) for s in steps],
                "evals": history.evals,
                "measured": history.measured,
            },
            fh,
            default=str,
        )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("around_3140", "evals")}, indent=1
        )
    )


if __name__ == "__main__":
    main()
