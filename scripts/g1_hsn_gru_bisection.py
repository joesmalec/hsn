#!/usr/bin/env python
"""HSN gradient-flow diagnosis, item 5: bisection of the HSN module. The PDG cell update is replaced
by a standard GRU cell update and everything else is kept: the same byte and context embeddings,
encoder, predictor, shared path (u, eps), scorer, graph messages, summary q (the mean of the cell
states) and decoder. Each cell i keeps its own state h_i [d] and receives the same input vector
v_i = [u, eps, h_i, m_i] the PDG update receives; the update is nn.GRUCell(4d, d) with weights
shared across cells (r, z gates and the tanh candidate with the reset gate), instead of
h <- (1 - sigmoid(gate(v))) h + sigmoid(gate(v)) tanh(W_out(sum_j alpha_ij F_j(v))).

    python scripts/g1_hsn_gru_bisection.py --out evidence/G1/diagnosis/gru_bisection [--steps 3000]

No repository code is changed: the replacement is installed on the built model instance by this
script (the GRUCell is registered as a submodule so the trainer's optimizer owns it; the unit bank,
W_out and the gate are left in place but receive no gradient). Trains delayed/in_range alone with
the experiment configuration of record at --lr for --steps, evaluating on the dev split every
--eval-every steps; writes <out>/summary.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from hsn.config import load_config
from hsn.data.dataset import ByteDataset
from hsn.device import resolve, seed_everything
from hsn.runner import build_model, make_trainer
from torch import Tensor, nn

REPO = Path(__file__).resolve().parents[1]


def install_gru_update(model: Any, seed: int) -> nn.GRUCell:
    cell = model.cell
    d = cell.d
    g = torch.Generator().manual_seed(seed)
    gru = nn.GRUCell(4 * d, d)
    with torch.no_grad():  # the repository's init scale for the new weights
        for p in gru.parameters():
            if p.dim() > 1:
                p.copy_(torch.nn.init.trunc_normal_(torch.empty_like(p), std=0.02, generator=g))
            else:
                p.zero_()
    cell.gru_update = gru  # registered submodule: the optimizer sees it

    def step_dense(
        u: Tensor, eps: Tensor, state: Any, alpha: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        B, N, dd = cell._check_inputs(u, eps, state, alpha)
        with cell.graph.step():
            h_prev = state.h
            m = cell.graph.messages_dense(h_prev, cell.V)
            v = torch.cat(
                [u[:, None, :].expand(B, N, dd), eps[:, None, :].expand(B, N, dd), h_prev, m],
                dim=-1,
            )
            h_new = cell.gru_update(
                v.reshape(B * N, 4 * dd).to(h_prev.dtype), h_prev.reshape(B * N, dd)
            ).reshape(B, N, dd)
        return h_new, {"m": m, "v": v, "h_prev": h_prev}

    cell.step_dense = step_dense

    # HSN.param_report checks that every trainable parameter is accounted for by the Appendix C rows
    # plus the edge logits; the replacement adds its own row so the check stays honest.
    import types

    def param_report(self: Any) -> dict[str, int]:
        modules: dict[str, nn.Module] = {
            "byte_emb": self.byte_emb,
            "ctx_emb": self.ctx_emb,
            "encoder": self.encoder,
            "predictor": self.predictor,
            "shared": self.shared,
            "units": self.cell.units,
            "w_out": self.cell.W_out,
            "v_msg": self.cell.V,
            "gate": self.cell.gate,
            "scorer": self.scorer,
            "halting": self.halting,
            "decoder": self.decoder,
        }
        report = {name: sum(p.numel() for p in m.parameters()) for name, m in modules.items()}
        report["total"] = sum(report.values())
        report["edges"] = self.cell.graph.edge_logit.numel()
        report["gru_update"] = sum(p.numel() for p in self.cell.gru_update.parameters())
        report["total_trainable"] = sum(p.numel() for p in self.parameters())
        if report["total_trainable"] != report["total"] + report["edges"] + report["gru_update"]:
            raise RuntimeError("param_report (bisection): unaccounted parameters")
        return report

    model.param_report = types.MethodType(param_report, model)
    return gru


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(REPO / "configs" / "experiments" / "g1_ref_a1.yaml"))
    ap.add_argument("--profile", default="rtx3060")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
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
            "run_id=gru_bisection",
            "data.tasks=[delayed]",
            "data.variants=[in_range]",
            *args.override,
        ],
    )
    ctx = resolve(cfg.hardware)
    manifest = Path(str(cfg.data.manifest))
    train = ByteDataset.load(
        manifest, "train", ["delayed"], ["in_range"], expected_unroll=cfg.train.unroll
    )
    dev = ByteDataset.load(
        manifest, "dev", ["delayed"], ["in_range"], expected_unroll=cfg.train.unroll
    )
    seed_everything(cfg.seed)
    model = build_model(cfg)
    gru = install_gru_update(model, cfg.seed)
    n_gru = sum(p.numel() for p in gru.parameters())
    trainer = make_trainer(model, cfg, ctx, run_dir=out / "run")
    t0 = time.perf_counter()
    history = trainer.fit(train, dev, args.steps)
    summary = {
        "replacement": (
            "per-cell nn.GRUCell(4d, d) shared across cells on the PDG input "
            "v = [u, eps, h_i, m_i]; encoder, predictor, shared path, scorer, graph messages, "
            "summary q and decoder unchanged"
        ),
        "gru_params": n_gru,
        "lr": cfg.train.lr,
        "batch": cfg.train.batch,
        "steps": args.steps,
        "autocast": history.autocast,
        "evals": [
            {
                "step": int(e["step"]),
                "dev_answer_accuracy": float(e["per_group"]["delayed/in_range"]["accuracy"]),
                "dev_answer_loss": float(e["per_group"]["delayed/in_range"]["loss"]),
            }
            for e in history.evals
        ],
        "final_train_loss": float(history.steps[-1].loss),
        "wall_s": round(time.perf_counter() - t0, 1),
        "skipped_nonfinite_steps": history.measured.get("skipped_nonfinite_steps"),
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
