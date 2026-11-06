#!/usr/bin/env python
"""Post-A1 checks, item 1 of 5 Nov 2026: what does a delayed checkpoint's dev accuracy count?

The trainer's dev ``accuracy`` (``Trainer.evaluate``, ``per_group["delayed/in_range"]``) is the
fraction of *all* loss positions answered correctly: every query of every document (a delayed
document holds 1 to 4 queries; the first always asks the first fact). The sensitivity test of
``scripts/g1_harness_sensitivity.py`` probes one position per document: the answer to the query of
the first fact, which is the document's first query. This script measures both on one checkpoint,
over every dev delayed document, without training:

* ``trainer``: ``Trainer.evaluate`` on the dev split (the number the training history reports);
* ``probe_all``: the sensitivity script's forward (fresh state, one window, the run's autocast)
  scored over every loss position;
* ``probe_first``: the same forward scored at the first query's answer position only;
* ``by_query_slot``: the same forward scored per query slot (0 = first query).

    python scripts/g1_query_position_accuracy.py --run <run dir with resolved config + checkpoint> \
        --out <dir> --label <name>
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
from hsn.runner import build_model, make_trainer
from hsn.train.amp import autocast_ctx
from hsn.train.checkpoint import latest_checkpoint, load_checkpoint

QRY = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run)
    out = Path(args.out)
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
    dev = ByteDataset.load(
        Path(str(cfg.data.manifest)),
        "dev",
        ["delayed"],
        ["in_range"],
        expected_unroll=cfg.train.unroll,
    )

    trainer = make_trainer(model, cfg, ctx, run_dir=out / f"_eval_{args.label}")
    ev = trainer.evaluate(dev)
    trainer_acc = float(ev["per_group"]["delayed/in_range"]["accuracy"])

    n_all = c_all = n_first = c_first = 0
    slot_n: dict[int, int] = {}
    slot_c: dict[int, int] = {}
    queries_per_doc: dict[int, int] = {}
    for i in range(len(dev)):
        ex = dev.examples[i]
        tokens = np.asarray(ex.tokens, dtype=np.int64)
        targets = np.asarray(ex.targets, dtype=np.int64)
        lm = np.asarray(ex.loss_mask, dtype=bool)
        context = (
            np.asarray(ex.context, dtype=np.int64)
            if hasattr(ex, "context")
            else np.full_like(tokens, 2)
        )
        tok = torch.as_tensor(tokens, device=ctx.device)[None]
        ctxt = torch.as_tensor(context, device=ctx.device)[None]
        reset = torch.zeros_like(tok, dtype=torch.bool)
        reset[:, 0] = True
        with torch.no_grad(), autocast_ctx(ctx):
            state = model.init_state(1, ctx.device)
            logits, _state, _info = model(tok, ctxt, state, reset)
        pred = logits[0].float().argmax(dim=-1).cpu().numpy()
        positions = [int(p) for p in np.flatnonzero(lm)]
        queries_per_doc[len(positions)] = queries_per_doc.get(len(positions), 0) + 1
        for slot, p in enumerate(positions):
            if not (p > 0 and int(tokens[p - 1]) == QRY):
                raise RuntimeError(f"doc {i}: loss position {p} is not a query key position")
            ok = int(pred[p] == targets[p])
            n_all += 1
            c_all += ok
            slot_n[slot] = slot_n.get(slot, 0) + 1
            slot_c[slot] = slot_c.get(slot, 0) + ok
            if slot == 0:
                n_first += 1
                c_first += ok
    summary: dict[str, Any] = {
        "run": str(run_dir),
        "label": args.label,
        "checkpoint": checkpoint.name,
        "model": cfg.model_name,
        "autocast": str(ctx.autocast_dtype),
        "n_docs": len(dev),
        "queries_per_doc_histogram": {str(k): v for k, v in sorted(queries_per_doc.items())},
        "trainer_accuracy_all_queries": trainer_acc,
        "probe_all_queries": {"n": n_all, "correct": c_all, "accuracy": c_all / n_all},
        "probe_first_query": {"n": n_first, "correct": c_first, "accuracy": c_first / n_first},
        "by_query_slot": {
            str(s): {"n": slot_n[s], "correct": slot_c[s], "accuracy": slot_c[s] / slot_n[s]}
            for s in sorted(slot_n)
        },
    }
    with open(out / f"query_accuracy_{args.label}.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
