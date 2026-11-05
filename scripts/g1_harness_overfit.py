#!/usr/bin/env python
"""G1 harness diagnosis, item 5: overfit 64 delayed documents, fp32, weight decay 0, constant lr
3e-4 with a 200-step warmup, batch 4, up to 3,000 steps, train accuracy every 250 steps.

    python scripts/g1_harness_overfit.py --config configs/baselines/transformer_2m.yaml \
        --name transformer --out evidence/G1/diagnosis/overfit [--profile rtx3060] [--steps 3000]

One uninterrupted fit (a chunked fit restarts the loader at every chunk); the 64 documents are the
first 64 delayed/in_range documents of the train split, used both for training and as the "dev" set
of the trainer's periodic evaluation, so every evaluation is the train accuracy of those documents.
Writes <out>/<name>/ (the trainer's run directory, history.json with every evaluation) and
<out>/<name>_summary.json with the accuracy at every evaluation step and the first step at or above
0.99. The run is not stopped early: the summary names the step the target was reached at.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hsn.config import load_config
from hsn.data.dataset import ByteDataset
from hsn.device import resolve, seed_everything
from hsn.runner import build_model, make_trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="rtx3060")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--n-docs", type=int, default=64)
    ap.add_argument("--target", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(
        args.config,
        overrides=[
            f"profile={args.profile}",
            "hardware.autocast_dtype=fp32",
            "train.precision=fp32",
            "train.weight_decay=0",
            "train.lr=3e-4",
            "train.schedule=constant",
            "train.warmup_steps=200",
            "train.batch=4",
            "train.micro_batch=4",
            "train.grad_accum=1",
            f"train.steps={args.steps}",
            f"train.eval_every={args.eval_every}",
            f"train.ckpt_every={args.steps}",
            "train.stats_every=50",
            f"seed={args.seed}",
            f"run_id=overfit_{args.name}",
            "data.tasks=[delayed]",
            "data.variants=[in_range]",
        ],
    )
    ctx = resolve(cfg.hardware)
    manifest = Path(str(cfg.data.manifest))
    full = ByteDataset.load(
        manifest, "train", ["delayed"], ["in_range"], expected_unroll=cfg.train.unroll
    )
    docs = ByteDataset(list(full.examples[: args.n_docs]))
    seed_everything(cfg.seed)
    model = build_model(cfg)
    trainer = make_trainer(model, cfg, ctx, run_dir=out / args.name)
    started = time.perf_counter()
    history = trainer.fit(docs, docs, args.steps)
    wall = time.perf_counter() - started
    evals = [
        {"step": int(e["step"]), "train_accuracy": float(e["accuracy"]), "loss": float(e["loss"])}
        for e in history.evals
    ]
    reached = next((e["step"] for e in evals if e["train_accuracy"] >= args.target), None)
    summary = {
        "name": args.name,
        "config": args.config,
        "n_docs": len(docs),
        "doc_ids": [docs.meta(0).get("doc_id"), docs.meta(len(docs) - 1).get("doc_id")],
        "profile": args.profile,
        "autocast": history.autocast,
        "lr": cfg.train.lr,
        "schedule": cfg.train.schedule,
        "warmup_steps": cfg.train.warmup_steps,
        "weight_decay": cfg.train.weight_decay,
        "batch": cfg.train.batch,
        "steps": args.steps,
        "evals": evals,
        "reached_target_at_step": reached,
        "target": args.target,
        "final_train_accuracy": evals[-1]["train_accuracy"] if evals else None,
        "wall_s": round(wall, 1),
        "measured": history.measured,
        "param_count": history.measured.get("param_count"),
    }
    with open(out / f"{args.name}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(out / args.name / "history.json", "w") as fh:
        json.dump(
            {
                "steps": [vars(s) for s in history.steps],
                "evals": history.evals,
                "measured": history.measured,
            },
            fh,
            default=str,
        )
    print(json.dumps({k: v for k, v in summary.items() if k not in ("measured",)}, indent=1))


if __name__ == "__main__":
    main()
