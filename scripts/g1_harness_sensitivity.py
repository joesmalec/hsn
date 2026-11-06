#!/usr/bin/env python
"""G1 harness diagnosis, item 4: does a trained checkpoint read the context at all? For 50 dev
delayed documents, change the value symbol of the first fact and report how often the argmax at the
query that asks for that fact's key changes.

    python scripts/g1_harness_sensitivity.py --run <run dir with resolved config and checkpoint> \
        --out evidence/G1/diagnosis --n-docs 50

Per document: the first fact is the first key symbol directly followed by a value symbol (not
preceded by QRY); the probed answer position is the first loss-mask position whose input symbol is
that key (preceded by QRY). The modified document replaces the fact's value by another value symbol
that does not occur in the document. Both versions run through the model in evaluation mode under
the profile's autocast (as the gate's evaluation does), from a fresh state, as one window. Reported:
the fraction of probed documents whose argmax at the answer position changed, the fraction where the
new argmax equals the new value, and the fraction where the original argmax equalled the original
value (the model's accuracy on exactly these positions).
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

QRY = 3


def first_fact_and_query(tokens: np.ndarray, loss_mask: np.ndarray) -> tuple[int, int, int] | None:
    """(value position of the first fact, its key, the answer position querying that key)."""
    for i in range(len(tokens) - 1):
        k, v = int(tokens[i]), int(tokens[i + 1])
        if 64 <= k < 144 and 144 <= v < 192 and (i == 0 or int(tokens[i - 1]) != QRY):
            for p in np.flatnonzero(loss_mask):
                p = int(p)
                if int(tokens[p]) == k and p > 0 and int(tokens[p - 1]) == QRY:
                    return i + 1, k, p
            return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-docs", type=int, default=50, help="0 = every document of the split")
    ap.add_argument("--label", default=None)
    ap.add_argument("--split", default="dev", help="dev (DIAGNOSIS.md) or train (overfit docs)")
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
        args.split,
        ["delayed"],
        ["in_range"],
        expected_unroll=cfg.train.unroll,
    )

    def argmax_at(tokens: np.ndarray, context: np.ndarray, p: int) -> int:
        tok = torch.as_tensor(tokens, dtype=torch.int64, device=ctx.device)[None]
        ctxt = torch.as_tensor(context, dtype=torch.int64, device=ctx.device)[None]
        reset = torch.zeros_like(tok, dtype=torch.bool)
        reset[:, 0] = True
        with torch.no_grad(), autocast_ctx(ctx):
            state = model.init_state(1, ctx.device)
            logits, _state, _info = model(tok, ctxt, state, reset)
        return int(logits[0, p].float().argmax())

    probes: list[dict[str, Any]] = []
    for i in range(len(dev)):
        if args.n_docs > 0 and len(probes) >= args.n_docs:
            break
        ex = dev.examples[i]
        tokens = np.asarray(ex.tokens, dtype=np.int64)
        lm = np.asarray(ex.loss_mask, dtype=bool)
        context = (
            np.asarray(ex.context, dtype=np.int64)
            if hasattr(ex, "context")
            else np.full_like(tokens, 2)
        )
        found = first_fact_and_query(tokens, lm)
        if found is None:
            continue
        vpos, key, p = found
        old_value = int(tokens[vpos])
        present = set(tokens.tolist())
        new_value = next(
            v
            for v in range(old_value + 1, old_value + 49)
            if 144 <= ((v - 144) % 48 + 144) < 192 and ((v - 144) % 48 + 144) not in present
        )
        new_value = (new_value - 144) % 48 + 144
        modified = tokens.copy()
        modified[vpos] = new_value
        a0 = argmax_at(tokens, context, p)
        a1 = argmax_at(modified, context, p)
        probes.append(
            {
                "doc": dev.meta(i).get("doc_id"),
                "fact_value_pos": vpos,
                "key": key,
                "answer_pos": p,
                "old_value": old_value,
                "new_value": new_value,
                "argmax_original": a0,
                "argmax_modified": a1,
                "changed": a0 != a1,
                "follows_new_value": a1 == new_value,
                "original_correct": a0 == old_value,
                "target": int(ex.targets[p]),
            }
        )
    n = len(probes)
    summary = {
        "run": str(run_dir),
        "label": args.label or run_dir.name,
        "checkpoint": checkpoint.name,
        "model": cfg.model_name,
        "autocast": str(ctx.autocast_dtype),
        "split": args.split,
        "n_probed": n,
        "n_changed_to_new_value": sum(p["follows_new_value"] for p in probes),
        "n_changed_to_other": sum(p["changed"] and not p["follows_new_value"] for p in probes),
        "n_unchanged": sum(not p["changed"] for p in probes),
        # post-A1 5(c): the same three counts over the documents answered correctly before the edit
        "n_correct_before_edit": sum(p["original_correct"] for p in probes),
        "correct_before_edit": {
            "n_changed_to_new_value": sum(
                p["follows_new_value"] for p in probes if p["original_correct"]
            ),
            "n_changed_to_other": sum(
                p["changed"] and not p["follows_new_value"] for p in probes if p["original_correct"]
            ),
            "n_unchanged": sum(not p["changed"] for p in probes if p["original_correct"]),
        },
        "changed_fraction": sum(p["changed"] for p in probes) / n if n else None,
        "follows_new_value_fraction": sum(p["follows_new_value"] for p in probes) / n
        if n
        else None,
        "original_correct_fraction": sum(p["original_correct"] for p in probes) / n if n else None,
        "argmax_original_histogram": {
            str(k): v
            for k, v in sorted(
                __import__("collections").Counter(p["argmax_original"] for p in probes).items()
            )
        },
        "probes": probes,
    }
    name = args.label or run_dir.name
    with open(out / f"sensitivity_{name}.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "probes"}, indent=1))


if __name__ == "__main__":
    main()
