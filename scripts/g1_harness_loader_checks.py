#!/usr/bin/env python
"""G1 harness diagnosis, items 2 and 3 (brief of 5 Nov 2026): the plumbing of one real training
batch and the fraction of answer positions whose dependency lies in an earlier unroll window.

    python scripts/g1_harness_loader_checks.py --config <config> --override profile=cpu \
        --out evidence/G1/diagnosis

Item 2: the first training batch the loader yields for the experiment configuration (shuffle as in
training, the run's seed), three delayed documents of it: tokens, targets, loss_mask, reset and dep
side by side around the first query, with the checks targets[t] == tokens[t+1] inside the document,
loss_mask True exactly at the answer positions the data defines (Example.loss_mask), and the answer
equal to the queried fact's value (a reference solve of the document: the value that followed the
queried key in a fact). Item 3: over one pass of the training loader, the fraction of loss-masked
positions whose dependency (t - dep[t]) lies before the start of the window that holds the answer,
per task. CPU only; nothing is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from hsn.config import load_config
from hsn.device import resolve, seed_everything
from hsn.runner import build_model, load_datasets, make_trainer

PAD, BOS, EOS, QRY, SEP, NONE = 0, 1, 2, 3, 4, 5
NAMES = {
    PAD: "PAD",
    BOS: "BOS",
    EOS: "EOS",
    QRY: "QRY",
    SEP: "SEP",
    NONE: "NONE",
    16: "INS",
    17: "OVR",
    18: "DEL",
}


def sym(v: int) -> str:
    if v in NAMES:
        return NAMES[v]
    if 64 <= v < 144:
        return f"k{v}"
    if 144 <= v < 192:
        return f"v{v}"
    if 192 <= v < 256:
        return f"f{v}"
    return f"?{v}"


def delayed_reference(tokens: np.ndarray) -> dict[int, int]:
    """key -> value of the facts of a delayed document: a key symbol directly followed by a value
    symbol and not preceded by QRY."""
    facts: dict[int, int] = {}
    for i in range(len(tokens) - 1):
        k, v = int(tokens[i]), int(tokens[i + 1])
        if 64 <= k < 144 and 144 <= v < 192 and (i == 0 or int(tokens[i - 1]) != QRY):
            facts[k] = v
    return facts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-docs", type=int, default=3)
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config, overrides=list(args.override))
    ctx = resolve(cfg.hardware)
    train_ds, _dev = load_datasets(cfg)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    trainer = make_trainer(model, cfg, ctx, run_dir=out / "_scratch_loader")
    report: dict[str, Any] = {
        "config": args.config,
        "seed": cfg.seed,
        "batch": cfg.train.batch,
        "unroll": cfg.train.unroll,
        "align_docs": cfg.data.align_docs,
    }

    # Item 2: the first real training batch (shuffled as in training).
    loader = trainer._loader(train_ds, shuffle=True, epochs=1)
    batch = next(iter(loader))
    tok, tgt, lm, rs, dep, doc = (
        np.asarray(batch[k]) for k in ("tokens", "targets", "loss_mask", "reset", "dep", "doc")
    )
    lines: list[str] = []
    checks: list[dict[str, Any]] = []
    shown = 0
    for r in range(tok.shape[0]):
        d = int(doc[r, 0])
        if d < 0:
            continue
        meta = train_ds.meta(d)
        if meta.get("task") != "delayed":
            continue
        ex = train_ds.examples[d]
        n = int(np.count_nonzero(tok[r] != PAD))
        row_tokens = tok[r, :n]
        # (a) next-symbol targets inside the document, (b) loss_mask == the example's own mask,
        # (c) the answer at every loss position equals the queried fact's value.
        a_ok = bool(np.array_equal(tgt[r, : n - 1], row_tokens[1:])) and int(tgt[r, n - 1]) == PAD
        b_ok = bool(np.array_equal(lm[r, :n], np.asarray(ex.loss_mask, dtype=bool)[:n]))
        facts = delayed_reference(row_tokens)
        loss_pos = np.flatnonzero(lm[r, :n])
        c_detail = []
        for p in loss_pos:
            key = int(row_tokens[p])
            prev = int(row_tokens[p - 1]) if p > 0 else -1
            c_detail.append(
                {
                    "pos": int(p),
                    "input": sym(key),
                    "preceded_by_QRY": prev == QRY,
                    "target": sym(int(tgt[r, p])),
                    "fact_value": sym(facts.get(key, -1)),
                    "ok": facts.get(key) == int(tgt[r, p]),
                    "dep": int(dep[r, p]),
                    "dependency_pos": int(p) - int(dep[r, p]),
                }
            )
        c_ok = all(x["ok"] and x["preceded_by_QRY"] for x in c_detail)
        checks.append(
            {
                "row": r,
                "doc": d,
                "doc_id": meta.get("doc_id"),
                "length": n,
                "reset_at_0": bool(rs[r, 0]),
                "resets_in_row": int(rs[r].sum()),
                "targets_are_next_symbol": a_ok,
                "loss_mask_equals_example": b_ok,
                "answers_equal_fact_values": c_ok,
                "n_answer_positions": int(len(loss_pos)),
                "answers": c_detail,
            }
        )
        if loss_pos.size:
            p0 = int(loss_pos[0])
            lo, hi = max(0, p0 - 8), min(n, p0 + 4)
            facts_s = ", ".join(f"{sym(k)}: {sym(v)}" for k, v in facts.items())
            lines.append(
                f"row {r} doc {meta.get('doc_id')} length {n}, first query at position {p0} "
                f"(dependency at {p0 - int(dep[r, p0])}, dep {int(dep[r, p0])}); facts {{{facts_s}}}"
            )
            lines.append("  t        : " + " ".join(f"{t:>5d}" for t in range(lo, hi)))
            lines.append(
                "  tokens   : " + " ".join(f"{sym(int(row_tokens[t])):>5s}" for t in range(lo, hi))
            )
            lines.append(
                "  targets  : " + " ".join(f"{sym(int(tgt[r, t])):>5s}" for t in range(lo, hi))
            )
            lines.append(
                "  loss_mask: " + " ".join(f"{'1' if lm[r, t] else '.':>5s}" for t in range(lo, hi))
            )
            lines.append(
                "  reset    : " + " ".join(f"{'R' if rs[r, t] else '.':>5s}" for t in range(lo, hi))
            )
            lines.append("  dep      : " + " ".join(f"{int(dep[r, t]):>5d}" for t in range(lo, hi)))
            lines.append("")
        shown += 1
        if shown >= args.n_docs:
            break
    report["item2_batch_rows"] = checks
    (out / "harness_item2_batch.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(
        "item 2 checks:",
        [
            (
                c["doc_id"],
                c["targets_are_next_symbol"],
                c["loss_mask_equals_example"],
                c["answers_equal_fact_values"],
            )
            for c in checks
        ],
    )

    # Item 3: window straddling over one pass of the training loader (as training sees it).
    loader = trainer._loader(train_ds, shuffle=True, epochs=1)
    counts: dict[str, dict[str, int]] = {}
    n_windows = 0
    for batch in loader:
        n_windows += 1
        lm = np.asarray(batch["loss_mask"])
        dep = np.asarray(batch["dep"])
        doc = np.asarray(batch["doc"])
        rows, ts = np.nonzero(lm)
        for r, t in zip(rows.tolist(), ts.tolist(), strict=True):
            d = int(doc[r, t])
            task = str(train_ds.meta(d).get("task")) if d >= 0 else "?"
            c = counts.setdefault(
                task,
                {"answers": 0, "dependency_in_earlier_window": 0, "dependency_before_doc_start": 0},
            )
            c["answers"] += 1
            if t - int(dep[r, t]) < 0:
                c["dependency_in_earlier_window"] += 1
    report["item3_windows"] = n_windows
    report["item3_straddling"] = {
        k: {
            **v,
            "fraction": v["dependency_in_earlier_window"] / v["answers"] if v["answers"] else None,
        }
        for k, v in counts.items()
    }
    print("item 3:", json.dumps(report["item3_straddling"]), "windows", n_windows)
    with open(out / "harness_items_2_3.json", "w") as fh:
        json.dump(report, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
