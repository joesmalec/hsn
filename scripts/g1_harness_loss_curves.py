#!/usr/bin/env python
"""G1 harness diagnosis, item 1: masked train loss versus step from the saved histories of the
gate's runs, against the floors a marginal predictor would reach.

    python scripts/g1_harness_loss_curves.py --runs <gate run dir> \
        --manifest data/synthetic_v1/manifest.json --out evidence/G1/diagnosis \
        [--names transformer_2m-rtx3060-s0 g1_ref_a1-rtx3060-s0 gru_2m-rtx3060-s0]

The per-step ``loss`` of history.json is the mean cross-entropy over the step's answer (loss-mask)
positions of both tasks. Floors: ln(48) = 3.871 for delayed (48 value symbols, the brief's
definition), the entropy of the empirical dev answer distribution for kv (NONE and the 48 values),
and their mixture weighted by the tasks' answer-position counts (what a model that only learns the
answer marginals scores on the mixed training stream). Dev loss per task at every evaluation step is
overlaid from history.json's evals. Writes loss_curves.png, loss_curves.csv and loss_curves.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from hsn.data.dataset import ByteDataset  # noqa: E402

NONE = 5


def answer_entropy(ds: ByteDataset) -> dict[str, dict[str, float]]:
    counts: dict[str, Counter[int]] = {}
    for i, ex in enumerate(ds.examples):
        task = str(ds.meta(i).get("task"))
        tg = np.asarray(ex.targets)[np.asarray(ex.loss_mask, dtype=bool)]
        counts.setdefault(task, Counter()).update(int(t) for t in tg.tolist())
    out = {}
    for task, c in counts.items():
        n = sum(c.values())
        h = -sum(k / n * math.log(k / n) for k in c.values())
        out[task] = {"entropy": h, "n": n, "majority_rate": max(c.values()) / n}
    return out


def smooth(values: list[float], window: int) -> list[float]:
    arr = np.asarray(values, dtype=float)
    out = []
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        seg = arr[lo : i + 1]
        seg = seg[np.isfinite(seg)]
        out.append(float(seg.mean()) if seg.size else float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--names",
        nargs="+",
        default=["transformer_2m-rtx3060-s0", "g1_ref_a1-rtx3060-s0", "gru_2m-rtx3060-s0"],
    )
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dev = ByteDataset.load(args.manifest, "dev", ["kv", "delayed"], ["in_range"], verify=False)
    ent = answer_entropy(dev)
    ln48 = math.log(48)
    kv_floor = ent["kv"]["entropy"]
    n_kv, n_del = ent["kv"]["n"], ent["delayed"]["n"]
    mixed_floor = (n_kv * kv_floor + n_del * ln48) / (n_kv + n_del)
    floors = {
        "delayed_ln48": ln48,
        "delayed_answer_entropy_dev": ent["delayed"]["entropy"],
        "kv_answer_entropy_dev": kv_floor,
        "mixed_floor_by_dev_answer_counts": mixed_floor,
        "dev_answer_positions": {"kv": n_kv, "delayed": n_del},
    }

    summary: dict[str, object] = {"floors": floors, "runs": {}}
    fig, axes = plt.subplots(1, len(args.names), figsize=(6 * len(args.names), 4.5), sharey=True)
    if len(args.names) == 1:
        axes = [axes]
    rows: list[list[object]] = []
    for ax, name in zip(axes, args.names, strict=True):
        hist_path = Path(args.runs) / name / "history.json"
        h = json.load(open(hist_path))
        steps = [int(s["step"]) for s in h["steps"]]
        loss = [float(s["loss"]) if s.get("loss") is not None else float("nan") for s in h["steps"]]
        sm = smooth(loss, args.window)
        ax.plot(steps, loss, color="0.8", lw=0.6, label="masked train loss (per step)")
        ax.plot(steps, sm, color="C0", lw=1.5, label=f"moving mean ({args.window} steps)")
        ax.axhline(ln48, color="C3", ls="--", lw=1, label="ln(48) = 3.871 (delayed floor)")
        ax.axhline(
            mixed_floor,
            color="C1",
            ls=":",
            lw=1.2,
            label=f"marginal floor, mixed stream = {mixed_floor:.3f}",
        )
        ev_steps = [int(e["step"]) for e in h.get("evals", [])]
        for task, color in (("kv/in_range", "C2"), ("delayed/in_range", "C4")):
            ys = [
                float(e["per_group"][task]["loss"])
                for e in h.get("evals", [])
                if task in e.get("per_group", {})
            ]
            if ys:
                ax.plot(
                    ev_steps[: len(ys)], ys, "o-", color=color, ms=4, lw=1, label=f"dev loss {task}"
                )
        finite = [v for v in loss if math.isfinite(v)]
        below = [
            s for s, v in zip(steps, sm, strict=True) if math.isfinite(v) and v < mixed_floor - 0.1
        ]
        run_summary = {
            "steps": len(steps),
            "final_loss": loss[-1],
            "min_loss": min(finite) if finite else None,
            "min_smoothed": float(np.nanmin(sm)) if sm else None,
            "first_step_smoothed_below_mixed_floor_minus_0.1": below[0] if below else None,
            "final_dev_loss": {
                t: float(h["evals"][-1]["per_group"][t]["loss"])
                for t in ("kv/in_range", "delayed/in_range")
                if h.get("evals")
            },
            "final_dev_accuracy": {
                t: float(h["evals"][-1]["per_group"][t]["accuracy"])
                for t in ("kv/in_range", "delayed/in_range")
                if h.get("evals")
            },
            "final_train_accuracy_moving_mean": float(
                np.nanmean([float(s["accuracy"]) for s in h["steps"][-args.window :]])
            ),
        }
        summary["runs"][name] = run_summary  # type: ignore[index]
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylim(0, 6)
        ax.grid(alpha=0.3)
        for s, v, m in zip(steps, loss, sm, strict=True):
            rows.append([name, s, v, m])
    axes[0].set_ylabel("cross-entropy over answer positions")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=130)
    with open(out / "loss_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "step", "train_loss", "train_loss_moving_mean"])
        w.writerows(rows)
    with open(out / "loss_curves.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
