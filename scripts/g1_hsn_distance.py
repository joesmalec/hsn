#!/usr/bin/env python
"""HSN gradient-flow diagnosis, item 4: can HSN learn delayed retrieval at a short dependency?

    python scripts/g1_hsn_distance.py --dep 3 8 --out evidence/G1/diagnosis/distance/d3-8

A temporary delayed variant generated in memory from the committed task's parameters with the
dependency range and document lengths overridden (NOT part of the gate, no data spec file changed):
train documents from the train split's key alphabet, evaluation on held-out train-split documents
(seen keys) and on dev-split documents (unseen keys). The experiment configuration of record
(configs/experiments/g1_ref_a1.yaml) trains delayed only for --steps at --lr with the batch given;
evaluation every --eval-every steps on both evaluation sets. Writes <out>/summary.json and the
trainer's run directory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hsn.config import load_config
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import TaskSpec, generate, load_dataset_spec
from hsn.device import resolve, seed_everything
from hsn.runner import build_model, make_trainer

REPO = Path(__file__).resolve().parents[1]


def variant(dep_lo: int, dep_hi: int, length: tuple[int, int], unroll: int) -> TaskSpec:
    spec = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
    task = spec.task("delayed", "in_range")
    return TaskSpec.model_validate(
        {
            **task.model_dump(),
            "variant": "in_range",
            "n_train": 20000,
            "n_dev": 1000,
            "n_test": 1000,
            "length_range": list(length),
            "dep_len_range": [dep_lo, dep_hi],
            "unroll": unroll,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dep", nargs=2, type=int, required=True, metavar=("LO", "HI"))
    ap.add_argument("--length", nargs=2, type=int, default=[16, 64])
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(REPO / "configs" / "experiments" / "g1_ref_a1.yaml"))
    ap.add_argument("--profile", default="rtx3060")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
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
            f"run_id=distance_{args.dep[0]}_{args.dep[1]}",
            "data.tasks=[delayed]",
            "data.variants=[in_range]",
            *args.override,
        ],
    )
    task = variant(args.dep[0], args.dep[1], (args.length[0], args.length[1]), cfg.train.unroll)
    train = ByteDataset([generate(task, "train", i) for i in range(args.n_train)])  # type: ignore[arg-type]
    held = ByteDataset([generate(task, "train", 10000 + i) for i in range(args.n_eval)])  # type: ignore[arg-type]
    dev = ByteDataset([generate(task, "dev", i) for i in range(args.n_eval)])  # type: ignore[arg-type]
    deps = [d for ex in train.examples for d in ex.meta["deps"]]
    ctx = resolve(cfg.hardware)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    trainer = make_trainer(model, cfg, ctx, run_dir=out / "run")
    t0 = time.perf_counter()
    history = trainer.fit(train, held, args.steps)
    wall = time.perf_counter() - t0
    final_dev = trainer.evaluate(dev)
    summary = {
        "dep_len_range": args.dep,
        "length_range": args.length,
        "n_train": len(train),
        "n_eval": args.n_eval,
        "train_deps_min_max_mean": [min(deps), max(deps), sum(deps) / len(deps)],
        "lr": cfg.train.lr,
        "batch": cfg.train.batch,
        "steps": args.steps,
        "autocast": history.autocast,
        "held_out_train_split_evals": [
            {
                "step": int(e["step"]),
                "answer_accuracy": float(e["accuracy"]),
                "loss": float(e["loss"]),
            }
            for e in history.evals
        ],
        "dev_split_final": {
            "answer_accuracy": float(final_dev["accuracy"]),
            "loss": float(final_dev["loss"]),
        },
        "final_train_loss": float(history.steps[-1].loss),
        "wall_s": round(wall, 1),
        "skipped_nonfinite_steps": history.measured.get("skipped_nonfinite_steps"),
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
