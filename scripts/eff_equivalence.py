#!/usr/bin/env python
"""P1.10 efficiency audit, E1: the dense HSN's cells are interchangeable (CPU).

    python scripts/eff_equivalence.py trajectory --out evidence/EFF/e1 [--steps 20]
    python scripts/eff_equivalence.py checkpoints <step_*.pt> ... [--json out.json]

``trajectory`` trains the overfit recipe of ``eff_overfit.py`` on the CPU (fp32) twice from the
same seed: once as configured (``n_cells=32``, ``in_degree=4``) and once with ``n_cells=5,
in_degree=4``, where every cell reads all four others. If the cells of the dense round stay
identical (the hypothesis), both runs compute the same function with the same parameters (the
Appendix C rows do not depend on ``N``; only the edge logits do, and they never receive a
gradient), so their per-step losses agree up to the rounding of the mean over cells. At every
dense round the script records the largest difference between any cell and cell 0, and at every
optimiser step the largest absolute edge-logit gradient. Writes ``<out>/trajectory.json``.

``checkpoints`` reads TBPTT checkpoints (no model is built): the largest absolute edge logit, the
largest difference between any cell and cell 0 in the carried state, and the fraction of carried
state coordinates with ``|h| >= 0.99``, ``>= 0.999`` and ``== 1``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eff_overfit import REPO, overfit_docs, recipe_overrides  # noqa: E402
from hsn.config import load_config  # noqa: E402
from hsn.device import resolve, seed_everything  # noqa: E402
from hsn.model.hsn import HSN  # noqa: E402
from hsn.runner import build_model, make_trainer  # noqa: E402
from torch import Tensor  # noqa: E402


def run_arm(name: str, extra: list[str], steps: int, n_docs: int, out: Path) -> dict[str, Any]:
    overrides = recipe_overrides(
        profile="cpu",
        lr=1e-3,
        steps=steps,
        eval_every=10 * steps,  # the final step evaluates anyway
        seed=0,
        run_id=f"eff_e1_{name}",
        extra=extra,
    )
    cfg = load_config(REPO / "configs" / "experiments" / "g1_ref_a1.yaml", overrides=overrides)
    ctx = resolve(cfg.hardware)
    docs = overfit_docs(cfg, n_docs)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    assert isinstance(model, HSN)
    spreads: list[float] = []
    original = model.cell.step_dense

    def step_dense(u: Tensor, eps: Tensor, state: Any, alpha: Tensor) -> Any:
        h_new, info = original(u, eps, state, alpha)
        with torch.no_grad():
            spreads.append(float((h_new - h_new[:, :1]).abs().amax()))
        return h_new, info

    model.cell.step_dense = step_dense  # type: ignore[method-assign]
    trainer = make_trainer(model, cfg, ctx, run_dir=out / name)
    edge_grads: list[float] = []
    opt_step = trainer.optimizer.step

    def step(*args: Any, **kwargs: Any) -> Any:
        g = model.cell.graph.edge_logit.grad
        edge_grads.append(float(g.abs().max()) if g is not None else math.nan)
        return opt_step(*args, **kwargs)

    trainer.optimizer.step = step  # type: ignore[method-assign]
    started = time.perf_counter()
    history = trainer.fit(docs, docs, steps)
    return {
        "name": name,
        "overrides": extra,
        "params": model.param_report(),
        "loss": [s.loss for s in history.steps],
        "grad_norm": [s.grad_norm for s in history.steps],
        "edge_grad_absmax_per_step": edge_grads,
        "cell_spread_max_over_rounds": max(spreads) if spreads else None,
        "rounds_checked": len(spreads),
        "edge_logit_absmax_final": float(model.cell.graph.edge_logit.detach().abs().max()),
        "final_eval": history.evals[-1] if history.evals else None,
        "wall_s": round(time.perf_counter() - started, 1),
    }


def trajectory(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [
        run_arm("n32", [], args.steps, args.n_docs, out),
        run_arm("n5", ["model.n_cells=5", "model.in_degree=4"], args.steps, args.n_docs, out),
    ]
    a, b = arms[0]["loss"], arms[1]["loss"]
    rel = [abs(x - y) / max(abs(x), 1e-12) for x, y in zip(a, b, strict=True)]
    ga, gb = arms[0]["grad_norm"], arms[1]["grad_norm"]
    grel = [abs(x - y) / max(abs(x), 1e-12) for x, y in zip(ga, gb, strict=True)]
    result = {
        "steps": args.steps,
        "arms": arms,
        "loss_rel_diff_per_step": rel,
        "loss_rel_diff_max": max(rel),
        "grad_norm_rel_diff_max": max(grel),
        "edge_grad_absmax_all_steps": max(max(arm["edge_grad_absmax_per_step"]) for arm in arms),
    }
    (out / "trajectory.json").write_text(json.dumps(result, indent=1, default=str))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "loss_rel_diff_max",
                    "grad_norm_rel_diff_max",
                    "edge_grad_absmax_all_steps",
                )
            }
            | {
                arm["name"]: {
                    "cell_spread_max": arm["cell_spread_max_over_rounds"],
                    "rounds": arm["rounds_checked"],
                    "loss_first_last": [arm["loss"][0], arm["loss"][-1]],
                    "wall_s": arm["wall_s"],
                }
                for arm in arms
            },
            indent=1,
        )
    )


def checkpoints(args: argparse.Namespace) -> None:
    rows = []
    for path in args.paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model"]
        h = ck["extra"]["carried_state"]["h"].float()
        a = h.abs()
        rows.append(
            {
                "checkpoint": str(path),
                "step": ck.get("step"),
                "edge_logit_absmax": float(sd["cell.graph.edge_logit"].abs().max()),
                "carried_cell_spread": float((h - h[:, :1]).abs().amax()),
                "sat_ge_0.99": float((a >= 0.99).float().mean()),
                "sat_ge_0.999": float((a >= 0.999).float().mean()),
                "sat_eq_1": float((a == 1.0).float().mean()),
                "abs_h_mean": float(a.mean()),
            }
        )
    text = json.dumps(rows, indent=1)
    if args.json:
        Path(args.json).write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trajectory")
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=20)
    t.add_argument("--n-docs", type=int, default=64)
    t.add_argument("--threads", type=int, default=8)
    c = sub.add_parser("checkpoints")
    c.add_argument("paths", nargs="+")
    c.add_argument("--json")
    args = ap.parse_args()
    if args.cmd == "trajectory":
        trajectory(args)
    else:
        checkpoints(args)


if __name__ == "__main__":
    main()
