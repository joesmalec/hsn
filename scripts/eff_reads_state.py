#!/usr/bin/env python
"""P1.10 efficiency audit: does a variant that memorises faster also read state? The post-A1 5(b)
training recipe and the 5(c) edit probe, with an ``eff_overfit.py`` variant installed.

    python scripts/eff_reads_state.py --name <arm> --out evidence/EFF/reads_state \
        [--variant pdg|gru_cell|...] [--override KEY=VALUE ...] [--steps 3000] [--config <yaml>]

``--config`` defaults to ``configs/experiments/g1_ref_a1.yaml``; a baseline configuration (for
example ``configs/baselines/gru_2m.yaml`` with ``--override baseline.gru.d_model=64``) runs the same
recipe as a control (pass the HSN recipe's schedule keys as overrides so the two match).

5(b), as ``scripts/g1_harness_dense_supervision.py`` (branch ``g1-diagnosis``, ``5deec76``) runs its
``sparse`` arm: ``configs/experiments/g1_ref_a1.yaml`` (bf16 autocast, weight decay 0.01, cosine
schedule, 200 warm-up steps, ``grad_clip`` 1.0) with ``delayed/in_range`` alone on the full training
split, batch 16, lr 1e-3, 3,000 steps, dev answer accuracy every 500 steps, ``train.stats_every``
100, seed 0.

5(c), as ``scripts/g1_harness_sensitivity.py`` (``g1-diagnosis`` ``04782fb``) with ``--n-docs 0``,
run in this process on the trained model (so the variant needs no reconstruction from a checkpoint):
in every dev delayed document the value of the first fact is replaced by a value symbol absent from
the document, and the argmax at the answer position of the first query for that fact's key is read
before and after the edit, from a fresh state, one window, evaluation mode, the run's autocast.
Reported over all probed documents and over those answered correctly before the edit: answers that
change to the new value (follow the edit), change to another symbol, or stay unchanged (the three
partition the probes; the per-probe ``follows_new_value`` is the raw ``after == new value``).
The verdict uses the rule fixed in ``POST_A1_CHECKS.md`` before HSN's 5(c) ran: reads state when
>= 50 % of the correct answers follow the edit and <= 10 % are unchanged; does not read state when
< 20 % follow; partial otherwise.

For an HSN model the ``eff_overfit.py`` diagnostics are recorded on the first 4 dev documents every
``--diag-every`` steps, and on the first ``--diag-docs`` dev documents after training
(``trained_state``: cell diversity, cell spread, participation ratio of the cell states, saturation
at answer and valid positions): a variant that breaks bitwise cell identity but keeps the cells
parallel still has an effective state of width d.

Writes ``<out>/<name>_reads_state.json`` (the dev curve, the probe counts, the verdict, the
histogram of first-query answers, the diagnostics and every probe) and the trainer's run directory
``<out>/<name>/``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eff_overfit import (  # noqa: E402
    REPO,
    VARIANTS,
    Diagnostics,
    card_status,
    gpu_processes,
    install_variant,
)
from hsn.config import load_config  # noqa: E402
from hsn.data.dataset import ByteDataset  # noqa: E402
from hsn.device import resolve, seed_everything  # noqa: E402
from hsn.runner import build_model, make_trainer  # noqa: E402
from hsn.train.amp import autocast_ctx  # noqa: E402

QRY = 3
KEYS = range(64, 144)
VALUES = range(144, 192)


def first_fact_and_query(tokens: np.ndarray, loss_mask: np.ndarray) -> tuple[int, int, int] | None:
    """(value position of the first fact, its key, the answer position querying that key)."""
    for i in range(len(tokens) - 1):
        k, v = int(tokens[i]), int(tokens[i + 1])
        if k in KEYS and v in VALUES and (i == 0 or int(tokens[i - 1]) != QRY):
            for p in np.flatnonzero(loss_mask):
                p = int(p)
                if int(tokens[p]) == k and p > 0 and int(tokens[p - 1]) == QRY:
                    return i + 1, k, p
            return None
    return None


def absent_value(old: int, present: set[int]) -> int:
    """The next value symbol after ``old`` (cyclically over the 48) absent from the document."""
    for step in range(1, 49):
        v = (old - VALUES.start + step) % len(VALUES) + VALUES.start
        if v not in present:
            return v
    raise ValueError("every value symbol occurs in the document")


def outcome_counts(probes: list[dict[str, Any]]) -> dict[str, int]:
    """Partition probes into follow / other / unchanged. A probe follows only when its answer
    changed: a model that gives one constant answer would otherwise count every probe whose new
    value equals that answer as both following the edit and unchanged."""
    follow = sum(p["changed"] and p["follows_new_value"] for p in probes)
    other = sum(p["changed"] and not p["follows_new_value"] for p in probes)
    unchanged = sum(not p["changed"] for p in probes)
    if follow + other + unchanged != len(probes):
        raise AssertionError("outcome counts do not partition the probes")
    return {"follow": follow, "other": other, "unchanged": unchanged}


def verdict(follow: int, unchanged: int, correct: int) -> str:
    if correct == 0:
        return "no correct answers to probe"
    f, u = follow / correct, unchanged / correct
    if f >= 0.5 and u <= 0.1:
        return "reads state"
    if f < 0.2:
        return "does not read state"
    return "partial"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(REPO / "configs" / "experiments" / "g1_ref_a1.yaml"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=VARIANTS, default="pdg")
    ap.add_argument("--offset-std", type=float, default=0.5)
    ap.add_argument("--profile", default="rtx3060")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=0, help="0 = the full training split (smoke)")
    ap.add_argument("--n-dev", type=int, default=0, help="0 = every dev document (smoke)")
    ap.add_argument("--diag-every", type=int, default=500, help="0 = no diagnostics in training")
    ap.add_argument("--diag-docs", type=int, default=32, help="dev documents for trained_state")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen = [gpu_processes()]
    overrides = [
        f"profile={args.profile}",
        f"train.batch={args.batch}",
        f"train.micro_batch={args.batch}",
        f"train.lr={args.lr}",
        f"train.steps={args.steps}",
        f"train.eval_every={args.eval_every}",
        f"train.ckpt_every={args.steps}",
        "train.stats_every=100",
        f"seed={args.seed}",
        f"run_id=eff_rs_{args.name}",
        "data.tasks=[delayed]",
        "data.variants=[in_range]",
        *args.override,
    ]
    cfg = load_config(args.config, overrides=overrides)
    ctx = resolve(cfg.hardware)
    manifest = Path(str(cfg.data.manifest))
    train = ByteDataset.load(
        manifest, "train", ["delayed"], ["in_range"], expected_unroll=cfg.train.unroll
    )
    dev = ByteDataset.load(
        manifest, "dev", ["delayed"], ["in_range"], expected_unroll=cfg.train.unroll
    )
    if args.n_train:
        train = ByteDataset(list(train.examples[: args.n_train]))
    if args.n_dev:
        dev = ByteDataset(list(dev.examples[: args.n_dev]))
    seed_everything(cfg.seed)
    model = build_model(cfg)
    variant = install_variant(model, args.variant, args.seed, args.offset_std)
    trainer = make_trainer(model, cfg, ctx, run_dir=out / args.name)
    diag = Diagnostics(model, cfg, ctx, dev, max(args.diag_every, 1))
    if args.diag_every:
        diag.wrap(trainer.optimizer)
    t0 = time.perf_counter()
    history = trainer.fit(train, dev, args.steps)
    train_s = time.perf_counter() - t0
    evals = [
        {
            "step": int(e["step"]),
            "dev_answer_accuracy": {g: float(v["accuracy"]) for g, v in e["per_group"].items()},
            "dev_answer_loss": {g: float(v["loss"]) for g, v in e["per_group"].items()},
        }
        for e in history.evals
    ]

    trained_state = Diagnostics(model, cfg, ctx, dev, 1, n_docs=args.diag_docs).state_stats()

    # 5(c): the edit probe on every dev document (as g1_harness_sensitivity.py --n-docs 0)
    model.eval()

    def argmax_at(tokens: np.ndarray, context: np.ndarray, p: int) -> int:
        tok = torch.as_tensor(tokens, dtype=torch.int64, device=ctx.device)[None]
        ctxt = torch.as_tensor(context, dtype=torch.int64, device=ctx.device)[None]
        reset = torch.zeros_like(tok, dtype=torch.bool)
        reset[:, 0] = True
        with torch.no_grad(), autocast_ctx(ctx):
            logits, _state, _info = model(tok, ctxt, model.init_state(1, ctx.device), reset)
        return int(logits[0, p].float().argmax())

    probes: list[dict[str, Any]] = []
    for i in range(len(dev)):
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
        new_value = absent_value(old_value, set(tokens.tolist()))
        modified = tokens.copy()
        modified[vpos] = new_value
        a0 = argmax_at(tokens, context, p)
        a1 = argmax_at(modified, context, p)
        probes.append(
            {
                "doc": dev.meta(i).get("doc_id"),
                "key": key,
                "answer_pos": p,
                "old_value": old_value,
                "new_value": new_value,
                "argmax_original": a0,
                "argmax_modified": a1,
                "changed": a0 != a1,
                "follows_new_value": a1 == new_value,
                "original_correct": a0 == old_value,
            }
        )
    correct = [p for p in probes if p["original_correct"]]
    c_counts = outcome_counts(correct)
    seen.append(gpu_processes())
    result = {
        "name": args.name,
        "config": args.config,
        **variant,
        "overrides": overrides,
        "autocast": history.autocast,
        "param_count": history.measured.get("param_count"),
        "train_wall_s": round(train_s, 1),
        "final_train_loss": float(history.steps[-1].loss),
        "evals": evals,
        "n_probed": len(probes),
        "first_query_accuracy": len(correct) / len(probes) if probes else None,
        "all": outcome_counts(probes),
        "correct_before_edit": {"n": len(correct), **c_counts},
        "verdict": verdict(c_counts["follow"], c_counts["unchanged"], len(correct)),
        "first_query_answer_histogram": dict(
            collections.Counter(p["argmax_original"] for p in probes).most_common()
        ),
        "distinct_first_query_answers": len({p["argmax_original"] for p in probes}),
        "trained_state": trained_state,
        "diagnostics": diag.records,
        "card": card_status(os.getpid(), seen),
        "gpu_processes": seen,
        "probes": probes,
    }
    path = out / f"{args.name}_reads_state.json"
    path.write_text(json.dumps(result, indent=1, default=str))
    print(path)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("probes", "overrides", "diagnostics", "gpu_processes")
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
