#!/usr/bin/env python
"""HSN gradient-flow diagnosis, items 1 and 2: where does information from an earlier token fail to
reach the answer logit?

    python scripts/g1_hsn_gradflow.py --run <HSN run dir with config.resolved.yaml and checkpoint> \
        --out evidence/G1/diagnosis/gradflow [--n-docs 32]

Item 1 (gradient path), on --n-docs delayed documents of the train split, one document per forward
in fp32 (autocast off): the gradient of the answer-position logit of the correct class with respect
to the input embedding of (a) the first fact's value token, (b) that fact's key token, (c) the token
one position before the query (the QRY marker), (d) the answer position's own input (the queried
key); norms reported at the seeded initialisation and at the trained checkpoint. The embeddings are
captured per position with a forward hook on ``byte_emb`` (the model encodes one position per call).

Item 2 (write/read statistics), on one real training window (batch 16 x 256 from the training
loader) and the same documents: the update gate ``b`` of every cell and position (fractions below
0.01 and above 0.99, mean), mean |h_new - h_prev| per position, and the change of the summary ``q``
(the mean of the cell states, what the decoder reads) between the fact's value position and the
query position, relative to |q|; captured by wrapping ``cell.step_dense``. Everything measured goes
to <out>/gradflow.json.
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
from hsn.train.checkpoint import latest_checkpoint, load_checkpoint

QRY = 3


def first_fact_and_query(tokens: np.ndarray, loss_mask: np.ndarray) -> tuple[int, int, int] | None:
    for i in range(len(tokens) - 1):
        k, v = int(tokens[i]), int(tokens[i + 1])
        if 64 <= k < 144 and 144 <= v < 192 and (i == 0 or int(tokens[i - 1]) != QRY):
            for p in np.flatnonzero(loss_mask):
                p = int(p)
                if int(tokens[p]) == k and p > 0 and int(tokens[p - 1]) == QRY:
                    return i + 1, k, p
            return None
    return None


class Recorder:
    """Wraps cell.step_dense to record b, h_prev, h_new per call (one call per position)."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.calls: list[dict[str, torch.Tensor]] = []
        self._orig = model.cell.step_dense

    def __enter__(self) -> Recorder:
        orig = self._orig

        def wrapped(u: torch.Tensor, eps: torch.Tensor, state: Any, alpha: torch.Tensor) -> Any:
            h_new, info = orig(u, eps, state, alpha)
            self.calls.append(
                {
                    "b": info["b"].detach(),
                    "h_prev": info["h_prev"].detach(),
                    "h_new": h_new.detach(),
                }
            )
            return h_new, info

        self.model.cell.step_dense = wrapped
        return self

    def __exit__(self, *exc: object) -> None:
        self.model.cell.step_dense = self._orig


def gradient_probe(
    model: Any, ctx: Any, docs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> dict[str, Any]:
    device = ctx.device
    model.eval()
    rows = []
    for tokens, context, lm in docs:
        found = first_fact_and_query(tokens, lm)
        if found is None:
            continue
        vpos, key, p = found
        kpos, qpos = vpos - 1, p - 1
        captured: list[torch.Tensor] = []

        def hook(
            _m: Any, _i: Any, out: torch.Tensor, _captured: list[torch.Tensor] = captured
        ) -> torch.Tensor:
            out = out.clone().requires_grad_(True) if not out.requires_grad else out
            out.retain_grad()
            _captured.append(out)
            return out

        handle = model.byte_emb.register_forward_hook(hook)
        tok = torch.as_tensor(tokens, dtype=torch.int64, device=device)[None]
        ctxt = torch.as_tensor(context, dtype=torch.int64, device=device)[None]
        reset = torch.zeros_like(tok, dtype=torch.bool)
        reset[:, 0] = True
        model.zero_grad(set_to_none=True)
        state = model.init_state(1, device)
        logits, _state, _info = model(tok, ctxt, state, reset)
        target = int(tokens[p + 1]) if p + 1 < len(tokens) else None  # targets[p] == tokens[p+1]
        handle.remove()
        if target is None or len(captured) != len(tokens):
            continue
        logit = logits[0, p, target]
        logit.backward()

        def g(i: int, _captured: list[torch.Tensor] = captured) -> float:
            grad = _captured[i].grad
            return float(grad.norm()) if grad is not None else 0.0

        argmax = int(logits[0, p].argmax())
        rows.append(
            {
                "vpos": vpos,
                "kpos": kpos,
                "qpos": qpos,
                "p": p,
                "dependency": p - vpos,
                "target": target,
                "argmax": argmax,
                "grad_value_token": g(vpos),
                "grad_key_token": g(kpos),
                "grad_qry_token": g(qpos),
                "grad_answer_input": g(p),
                "grad_all_positions_max": max(g(i) for i in range(len(tokens))),
                "logit_value": float(logit),
            }
        )

    def agg(k: str) -> dict[str, float]:
        v = [r[k] for r in rows]
        return {
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        }

    return {
        "n_docs": len(rows),
        "dependency_mean": float(np.mean([r["dependency"] for r in rows])),
        "grad_value_token": agg("grad_value_token"),
        "grad_key_token": agg("grad_key_token"),
        "grad_qry_token": agg("grad_qry_token"),
        "grad_answer_input": agg("grad_answer_input"),
        "grad_all_positions_max": agg("grad_all_positions_max"),
        "fraction_value_grad_below_1e-8": float(
            np.mean([r["grad_value_token"] < 1e-8 for r in rows])
        ),
        "ratio_value_to_qry_median": float(
            np.median(
                [
                    r["grad_value_token"] / r["grad_qry_token"]
                    if r["grad_qry_token"] > 0
                    else float("nan")
                    for r in rows
                ]
            )
        ),
        "accuracy_on_probed_positions": float(np.mean([r["argmax"] == r["target"] for r in rows])),
        "rows": rows,
    }


def gate_stats(
    model: Any,
    ctx: Any,
    window: dict[str, torch.Tensor],
    docs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    device = ctx.device
    model.eval()
    out: dict[str, Any] = {}
    with torch.no_grad(), Recorder(model) as rec:
        tok = window["tokens"].to(device)
        ctxt = window["context"].to(device)
        reset = window["reset"].to(device)
        model(tok, ctxt, model.init_state(tok.shape[0], device), reset)
        b = torch.stack([c["b"] for c in rec.calls])  # [T, B, N, d]
        dh = torch.stack([(c["h_new"] - c["h_prev"]).abs().mean() for c in rec.calls])  # [T]
        nonpad = (tok != 0).T  # [T, B]
        bm = b.flatten(2)  # [T, B, N*d]
        sel = nonpad[:, :, None].expand_as(bm)
        bv = bm[sel]
        out["window"] = {
            "B": int(tok.shape[0]),
            "T": int(tok.shape[1]),
            "gate_mean": float(bv.mean()),
            "gate_fraction_below_0.01": float((bv < 0.01).float().mean()),
            "gate_fraction_above_0.99": float((bv > 0.99).float().mean()),
            "gate_fraction_below_0.1": float((bv < 0.1).float().mean()),
            "gate_p10": float(bv.quantile(0.1)),
            "gate_p50": float(bv.quantile(0.5)),
            "gate_p90": float(bv.quantile(0.9)),
            "mean_abs_delta_h_per_step": float(dh.mean()),
            "mean_abs_delta_h_first_10_steps": float(dh[:10].mean()),
            "mean_abs_delta_h_last_10_steps": float(dh[-10:].mean()),
            "cells_frozen_fraction_(mean gate over positions < 0.01)": float(
                (b.mean(dim=(0, 1, 3)) < 0.01).float().mean()
            ),
        }
    # q change between the fact's value position and the query position, per document
    qrows = []
    for tokens, context, lm in docs:
        found = first_fact_and_query(tokens, lm)
        if found is None:
            continue
        vpos, key, p = found
        with torch.no_grad(), Recorder(model) as rec:
            tok = torch.as_tensor(tokens, dtype=torch.int64, device=device)[None]
            ctxt = torch.as_tensor(context, dtype=torch.int64, device=device)[None]
            reset = torch.zeros_like(tok, dtype=torch.bool)
            reset[:, 0] = True
            model(tok, ctxt, model.init_state(1, device), reset)
            q = torch.stack(
                [c["h_new"].mean(dim=1)[0] for c in rec.calls]
            )  # [T, d]  (q = mean over cells)
            h = torch.stack([c["h_new"][0] for c in rec.calls])  # [T, N, d]
            dq = float((q[p] - q[vpos]).norm())
            qn = float(q[vpos].norm())
            qp = float(q[p].norm())
            dh_vp = float((h[vpos] - h[vpos - 1]).abs().mean()) if vpos > 0 else None
            qrows.append(
                {
                    "vpos": vpos,
                    "p": p,
                    "q_norm_at_value": qn,
                    "q_norm_at_query": qp,
                    "q_change_value_to_query": dq,
                    "q_change_relative": dq / qn if qn > 0 else None,
                    "q_norm_max_over_doc": float(q.norm(dim=1).max()),
                    "h_abs_max_over_doc": float(h.abs().max()),
                    "delta_h_at_value_write": dh_vp,
                    "q_cosine_value_to_query": float(
                        torch.nn.functional.cosine_similarity(q[p], q[vpos], dim=0)
                    ),
                }
            )

    def agg(k: str) -> dict[str, float]:
        v = [r[k] for r in qrows if r[k] is not None]
        return {
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        }

    out["summary_q"] = {
        "n_docs": len(qrows),
        **{
            k: agg(k)
            for k in (
                "q_norm_at_value",
                "q_norm_at_query",
                "q_change_value_to_query",
                "q_change_relative",
                "q_norm_max_over_doc",
                "h_abs_max_over_doc",
                "delta_h_at_value_write",
                "q_cosine_value_to_query",
            )
        },
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-docs", type=int, default=32)
    args = ap.parse_args()
    run_dir = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(
        run_dir / "config.resolved.yaml",
        overrides=["hardware.autocast_dtype=fp32", "train.precision=fp32"],
    )
    ctx = resolve(cfg.hardware)
    manifest = Path(str(cfg.data.manifest))
    train_ds = ByteDataset.load(
        manifest, "train", ["delayed"], ["in_range"], expected_unroll=cfg.train.unroll
    )
    docs = []
    for i in range(len(train_ds)):
        ex = train_ds.examples[i]
        docs.append(
            (
                np.asarray(ex.tokens, dtype=np.int64),
                np.asarray(ex.context, dtype=np.int64),
                np.asarray(ex.loss_mask, dtype=bool),
            )
        )
        if len(docs) >= args.n_docs:
            break
    results: dict[str, Any] = {
        "run": str(run_dir),
        "n_docs": args.n_docs,
        "autocast": str(ctx.autocast_dtype),
        "models": {},
    }
    for label in ("init", "trained"):
        seed_everything(cfg.seed)
        model = build_model(cfg)
        if label == "trained":
            ck = latest_checkpoint(run_dir)
            assert ck is not None
            load_checkpoint(ck, model)
            results["checkpoint"] = ck.name
        model.to(device=ctx.device, dtype=ctx.master_dtype)
        trainer = make_trainer(model, cfg, ctx, run_dir=out / f"_scratch_{label}")
        loader = trainer._loader(train_ds, shuffle=True, epochs=1)
        window = next(iter(loader))
        entry = {
            "gradient_path": gradient_probe(model, ctx, docs),
            "write_read": gate_stats(model, ctx, window, docs),
        }
        results["models"][label] = entry
        gp, wr = entry["gradient_path"], entry["write_read"]
        med = {
            k: gp[k]["median"]
            for k in ("grad_value_token", "grad_key_token", "grad_qry_token", "grad_answer_input")
        }
        print(
            f"[{label}] grad norms (median): value {med['grad_value_token']:.3e}  "
            f"key {med['grad_key_token']:.3e}  QRY {med['grad_qry_token']:.3e}  "
            f"answer-input {med['grad_answer_input']:.3e}  | value<1e-8: "
            f"{gp['fraction_value_grad_below_1e-8']:.2f}  acc {gp['accuracy_on_probed_positions']:.2f}",
            flush=True,
        )
        w = wr["window"]
        frozen = w["cells_frozen_fraction_(mean gate over positions < 0.01)"]
        print(
            f"[{label}] gate mean {w['gate_mean']:.3f}  <0.01: {w['gate_fraction_below_0.01']:.3f}  "
            f">0.99: {w['gate_fraction_above_0.99']:.3f}  p10/50/90 {w['gate_p10']:.3f}/"
            f"{w['gate_p50']:.3f}/{w['gate_p90']:.3f}  |dh| {w['mean_abs_delta_h_per_step']:.4f}  "
            f"frozen cells {frozen:.2f}",
            flush=True,
        )
        s = wr["summary_q"]
        print(
            f"[{label}] q: |q| at value {s['q_norm_at_value']['median']:.4f}, at query "
            f"{s['q_norm_at_query']['median']:.4f}, change {s['q_change_value_to_query']['median']:.4f} "
            f"(rel {s['q_change_relative']['median']:.3f}), cosine "
            f"{s['q_cosine_value_to_query']['median']:.3f}",
            flush=True,
        )
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    with open(out / "gradflow.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {out / 'gradflow.json'}")


if __name__ == "__main__":
    main()
