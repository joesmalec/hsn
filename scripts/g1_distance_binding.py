#!/usr/bin/env python
"""Post-A1 checks, "distance vs binding" (5 Nov 2026): evaluation only, no training.

On every dev delayed/in_range document, one forward pass per document (fresh state, the run's
autocast; documents batched, each in its own row with a reset at position 0 and padding after its
end, so a row's positions see only their own document):

1. accuracy by dependency distance: every answer position with its distance from the document's
   ``meta["deps"]`` (one per loss position, in position order), bucketed 16-32, 33-64, 65-128,
   129-240 (other distances counted separately); also the first query alone;
2. (HSN only, ``--gate-docs`` documents) the update gate ``b`` of every cell at every position
   (``info["b"]`` of ``HSNCell.step_dense``, ``[B, N, d]``): its mean over cells and coordinates on
   filler, fact-key and fact-value tokens and on the rest; the largest spread of ``b`` across cells;
   and per document the survival of the first fact's write, prod over t in (value position, first
   answer position] of (1 - b_t), per cell and coordinate, summarised by its mean and its maximum
   over cells and coordinates; quartiles overall and per distance bucket.

    python scripts/g1_distance_binding.py --run <run dir with config.resolved.yaml + checkpoint> \
        --out <dir> --label <name> [--gate-docs 200]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hsn.config import load_config
from hsn.data.dataset import ByteDataset
from hsn.device import resolve, seed_everything
from hsn.runner import build_model
from hsn.train.amp import autocast_ctx
from hsn.train.checkpoint import latest_checkpoint, load_checkpoint
from torch import Tensor

QRY, SEP = 3, 4
KEYS, VALUES, FILLER = (64, 144), (144, 192), (192, 256)
BUCKETS = ((16, 32), (33, 64), (65, 128), (129, 240))


def bucket_of(dep: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= dep <= hi:
            return f"{lo}-{hi}"
    return "other"


def quartiles(xs: list[float]) -> dict[str, float | int]:
    if not xs:
        return {"n": 0}
    q = np.percentile(np.asarray(xs, dtype=np.float64), [25, 50, 75])
    return {"n": len(xs), "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--gate-docs", type=int, default=200)
    args = ap.parse_args()
    run_dir, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(run_dir / "config.resolved.yaml")
    ctx = resolve(cfg.hardware)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"{run_dir} has no checkpoint")
    load_checkpoint(checkpoint, model)
    model.to(device=ctx.device, dtype=ctx.master_dtype)
    model.eval()
    is_hsn = cfg.model_name == "hsn"
    gate_docs = args.gate_docs if is_hsn else 0
    dev = ByteDataset.load(
        Path(str(cfg.data.manifest)),
        "dev",
        ["delayed"],
        ["in_range"],
        expected_unroll=cfg.train.unroll,
    )
    T = cfg.train.unroll

    gates: list[Tensor] = []
    if is_hsn:
        orig = model.cell.step_dense

        def step_dense(u: Tensor, eps: Tensor, state: Any, alpha: Tensor) -> Any:
            h_new, info = orig(u, eps, state, alpha)
            if recording[0]:
                gates.append(info["b"].detach().float())
            return h_new, info

        model.cell.step_dense = step_dense  # type: ignore[method-assign]
    recording = [False]

    answers: list[dict[str, Any]] = []
    gate_sum = {k: 0.0 for k in ("filler", "fact_key", "fact_value", "other")}
    gate_cnt = {k: 0 for k in gate_sum}
    cell_spread_max = 0.0
    survival: list[dict[str, Any]] = []
    n = len(dev)
    for lo in range(0, n, args.batch):
        idx = list(range(lo, min(n, lo + args.batch)))
        B = len(idx)
        tok = np.zeros((B, T), dtype=np.int64)
        con = np.zeros((B, T), dtype=np.int64)
        lengths = []
        for r, i in enumerate(idx):
            ex = dev.examples[i]
            t = np.asarray(ex.tokens, dtype=np.int64)
            c = (
                np.asarray(ex.context, dtype=np.int64)
                if hasattr(ex, "context")
                else np.full_like(t, 2)
            )
            tok[r, : len(t)] = t
            con[r, : len(t)] = c
            con[r, len(t) :] = c[-1]
            lengths.append(len(t))
        tok_t = torch.as_tensor(tok, device=ctx.device)
        con_t = torch.as_tensor(con, device=ctx.device)
        reset = torch.zeros_like(tok_t, dtype=torch.bool)
        reset[:, 0] = True
        recording[0] = is_hsn and lo < gate_docs
        gates.clear()
        with torch.no_grad(), autocast_ctx(ctx):
            state = model.init_state(B, ctx.device)
            logits, _state, _info = model(tok_t, con_t, state, reset)
        pred = logits.float().argmax(dim=-1).cpu().numpy()
        b_all = None
        if recording[0]:
            if len(gates) != T:
                raise RuntimeError(f"{len(gates)} cell steps recorded for {T} positions")
            b_all = torch.stack(gates)  # [T, B, N, d]
            spread = (b_all.max(dim=2).values - b_all.min(dim=2).values).abs().max()
            cell_spread_max = max(cell_spread_max, float(spread))
        for r, i in enumerate(idx):
            ex = dev.examples[i]
            t = tok[r, : lengths[r]]
            targets = np.asarray(ex.targets, dtype=np.int64)
            lm = np.asarray(ex.loss_mask, dtype=bool)
            positions = [int(p) for p in np.flatnonzero(lm)]
            deps = [int(d) for d in ex.meta["deps"]]
            if len(deps) != len(positions):
                raise RuntimeError(f"doc {i}: {len(deps)} deps for {len(positions)} answers")
            for slot, (p, d) in enumerate(zip(positions, deps, strict=True)):
                answers.append(
                    {
                        "doc": i,
                        "slot": slot,
                        "dep": d,
                        "correct": bool(pred[r, p] == targets[p]),
                    }
                )
            if b_all is None or i >= gate_docs:
                continue
            sep = int(np.flatnonzero(t == SEP)[0])
            b_doc = b_all[: lengths[r], r]  # [L, N, d]
            mean_per_pos = b_doc.mean(dim=(1, 2)).cpu().numpy()
            for pos in range(lengths[r]):
                sym = int(t[pos])
                if pos < sep and FILLER[0] <= sym < FILLER[1]:
                    cls = "filler"
                elif pos < sep and KEYS[0] <= sym < KEYS[1]:
                    cls = "fact_key"
                elif pos < sep and VALUES[0] <= sym < VALUES[1]:
                    cls = "fact_value"
                else:
                    cls = "other"
                gate_sum[cls] += float(mean_per_pos[pos])
                gate_cnt[cls] += 1
            p0, d0 = positions[0], deps[0]
            vpos = p0 - d0
            if not (VALUES[0] <= int(t[vpos]) < VALUES[1] and int(t[vpos - 1]) == int(t[p0])):
                raise RuntimeError(f"doc {i}: position {vpos} is not the first fact's value")
            keep = torch.prod(1.0 - b_doc[vpos + 1 : p0 + 1], dim=0)  # [N, d]
            written = b_doc[vpos]  # [N, d]: the share of the value's candidate written at its step
            present = written * keep  # the value's write as a share of the state at the answer
            coord_gate = b_doc[:sep].mean(dim=0)  # [N, d]: each coordinate's mean gate before SEP
            survival.append(
                {
                    "doc": i,
                    "dep": d0,
                    "mean": float(keep.mean()),
                    "max": float(keep.max()),
                    "written_mean": float(written.mean()),
                    "present_mean": float(present.mean()),
                    "present_max": float(present.max()),
                    "coord_gate_below_0.01": float((coord_gate < 0.01).float().mean()),
                    "coord_gate_above_0.2": float((coord_gate > 0.2).float().mean()),
                    "first_correct": bool(pred[r, p0] == targets[p0]),
                }
            )

    def acc(rows: list[dict[str, Any]]) -> dict[str, Any]:
        k = sum(a["correct"] for a in rows)
        return {"n": len(rows), "correct": k, "accuracy": k / len(rows) if rows else None}

    names = [f"{lo}-{hi}" for lo, hi in BUCKETS] + ["other"]
    summary: dict[str, Any] = {
        "run": str(run_dir),
        "label": args.label,
        "checkpoint": checkpoint.name,
        "model": cfg.model_name,
        "autocast": str(ctx.autocast_dtype),
        "n_docs": n,
        "all_answers": acc(answers),
        "by_distance_all_answers": {
            b: acc([a for a in answers if bucket_of(a["dep"]) == b]) for b in names
        },
        "by_distance_first_query": {
            b: acc([a for a in answers if a["slot"] == 0 and bucket_of(a["dep"]) == b])
            for b in names
        },
        "dep_min_max": [min(a["dep"] for a in answers), max(a["dep"] for a in answers)],
    }
    if is_hsn:
        summary["gate"] = {
            "docs": min(gate_docs, n),
            "mean_by_token_class": {
                k: {
                    "positions": gate_cnt[k],
                    "mean_gate": gate_sum[k] / gate_cnt[k] if gate_cnt[k] else None,
                }
                for k in gate_sum
            },
            "cell_spread_max": cell_spread_max,
        }
        summary["survival_first_fact"] = {
            "mean_over_cells_and_coordinates": quartiles([s["mean"] for s in survival]),
            "max_over_cells_and_coordinates": quartiles([s["max"] for s in survival]),
            "by_distance_mean": {
                b: quartiles([s["mean"] for s in survival if bucket_of(s["dep"]) == b])
                for b in names
            },
            "by_distance_max": {
                b: quartiles([s["max"] for s in survival if bucket_of(s["dep"]) == b])
                for b in names
            },
            "written_at_value_mean": quartiles([s["written_mean"] for s in survival]),
            "present_at_answer_mean": quartiles([s["present_mean"] for s in survival]),
            "present_at_answer_max": quartiles([s["present_max"] for s in survival]),
            "by_distance_present_mean": {
                b: quartiles([s["present_mean"] for s in survival if bucket_of(s["dep"]) == b])
                for b in names
            },
            "coordinates_mean_gate_below_0.01": quartiles(
                [s["coord_gate_below_0.01"] for s in survival]
            ),
            "coordinates_mean_gate_above_0.2": quartiles(
                [s["coord_gate_above_0.2"] for s in survival]
            ),
        }
    with open(out / f"distance_binding_{args.label}.json", "w") as fh:
        json.dump({**summary, "survival_per_doc": survival}, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
