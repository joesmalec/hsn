#!/usr/bin/env python
"""G1 harness diagnosis, follow-up to item 5: is the sparse supervision the reason the Transformer
never leaves the marginal predictor? Train the Transformer baseline on delayed/in_range with (a) the
task's own loss mask (answer positions only, about 3 % of the tokens) and (b) a dense next-symbol
mask over every non-padding position of the same documents, same recipe, same steps; evaluate both
on the untouched dev split (answer positions only). No model code and no data spec is changed: the
dense mask is a scratch copy of the training examples with loss_mask set over the document.

    python scripts/g1_harness_dense_supervision.py --config configs/baselines/transformer_2m.yaml \
        --out evidence/G1/diagnosis/dense [--steps 3000] [--batch 32] [--lr 1e-3]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from hsn.config import load_config
from hsn.data.dataset import ByteDataset
from hsn.device import resolve, seed_everything
from hsn.runner import build_model, make_trainer

PAD = 0


def dense_copy(ds: ByteDataset) -> ByteDataset:
    examples = []
    for ex in ds.examples:
        tokens = np.asarray(ex.tokens)
        targets = np.asarray(ex.targets)
        mask = (tokens != PAD) & (targets != PAD)
        meta = dict(ex.meta)
        meta["deps"] = [0] * int(mask.sum())
        examples.append(dataclasses.replace(ex, loss_mask=mask.astype(bool), meta=meta))
    return ByteDataset(examples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="rtx3060")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tasks", default="delayed")
    ap.add_argument("--arms", default="sparse,dense")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = args.tasks.split(",")
    results: dict[str, Any] = {
        "config": args.config,
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "tasks": tasks,
        "arms": {},
    }
    for arm in args.arms.split(","):
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
                f"run_id=dense_{arm}",
                f"data.tasks=[{','.join(tasks)}]",
                "data.variants=[in_range]",
            ],
        )
        ctx = resolve(cfg.hardware)
        manifest = Path(str(cfg.data.manifest))
        train = ByteDataset.load(
            manifest, "train", tasks, ["in_range"], expected_unroll=cfg.train.unroll
        )
        dev = ByteDataset.load(
            manifest, "dev", tasks, ["in_range"], expected_unroll=cfg.train.unroll
        )
        if arm == "dense":
            train = dense_copy(train)
        seed_everything(cfg.seed)
        model = build_model(cfg)
        trainer = make_trainer(model, cfg, ctx, run_dir=out / arm)
        t0 = time.perf_counter()
        history = trainer.fit(train, dev, args.steps)
        evals = [
            {
                "step": int(e["step"]),
                "dev_answer_accuracy": {g: float(v["accuracy"]) for g, v in e["per_group"].items()},
                "dev_answer_loss": {g: float(v["loss"]) for g, v in e["per_group"].items()},
            }
            for e in history.evals
        ]
        results["arms"][arm] = {
            "loss_positions_per_window_mean": float(
                np.mean([s.loss_positions for s in history.steps])
            ),
            "final_train_loss": float(history.steps[-1].loss),
            "evals": evals,
            "wall_s": round(time.perf_counter() - t0, 1),
            "autocast": history.autocast,
        }
        print(arm, json.dumps(results["arms"][arm]), flush=True)
        with open(out / "dense_supervision.json", "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
