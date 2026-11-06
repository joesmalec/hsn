#!/usr/bin/env python
"""P1.10 efficiency audit, part A (steps to learn): the G1 overfit recipe with overrides, model
variants and per-module instrumentation.

    python scripts/eff_overfit.py --config configs/experiments/g1_ref_a1.yaml --name s0 \
        --out evidence/EFF/overfit [--lr 1e-3] [--steps 2500] [--eval-every 125] \
        [--variant pdg|gru_cell|cell_offset|cell_embed] [--override KEY=VALUE ...]

The recipe is ``scripts/g1_harness_overfit.py`` of branch ``g1-diagnosis`` (commit ``9703db5``):
the first ``--n-docs`` (64) ``delayed/in_range`` training documents, used for training and as the
"dev" set of the trainer's periodic evaluation (so every evaluation is the train accuracy of those
documents), fp32, weight decay 0, constant lr after 200 warmup steps, batch 4, one uninterrupted
fit. ``--override`` adds configuration overrides after the recipe's (for example
``train.lambda_p=0`` or ``baseline.gru.d_model=64``).

Variants (HSN only; installed on the built instance, no repository code changes, as
``g1_hsn_gru_bisection.py`` does):

* ``pdg``: the model as configured.
* ``gru_cell``: the PDG update replaced by one ``nn.GRUCell(4d, d)`` shared by every cell, on the
  normalised cell input ``norm([u; eps; h_i; m_i])`` (the A1 wrapper), with PyTorch's default GRU
  initialisation (the one ``gru_2m`` uses). The unit bank, ``W_out`` and the gate stay in place
  but receive no gradient.
* ``gru_cell_linear_head`` (wrapper bisection W1): ``gru_cell`` with HSN's MLP decoder
  ``D([q; u])`` replaced by the GRU baseline's readout ``Linear(LayerNorm(q))`` (weights
  ``N(0, 0.02^2)``, bias 0; ``u`` is not read).
* ``gru_cell_direct_input`` (W2): ``gru_cell`` whose update reads ``emb(x_t) + ctx_emb(ctx_t)``
  (width ``d``, the GRU baseline's input) instead of ``norm([u; eps; h_i; m_i])``; E, P, S and
  ``eps`` still run and feed the decoder and the prediction loss.
* ``linear_head`` (V1): the PDG model with ``D([q; u])`` replaced by ``Linear(LayerNorm(q))``
  as in ``gru_cell_linear_head``.
* ``decoder_fc1_fan_in`` (V2): the PDG model with the decoder's first layer at the A1 fan-in
  initialisation instead of ``model.init``; its output layer keeps ``model.init``.
* ``cell_offset``: a fixed per-cell vector ``o_i ~ N(0, offset_std^2)`` of width ``4d`` (a buffer
  from its own generator seeded with ``--seed``; no parameter) added to the normalised cell input
  of the dense round, so the cells are no longer interchangeable.
* ``cell_embed``: the same vector as a learned parameter (``N * 4d`` = 8,192 parameters for A.1),
  starting from the same values as ``cell_offset``.

The variants draw from their own generator, so every variant starts from the same model weights
and sees the same data order as ``pdg`` for one seed.

Instrumentation, every ``--diag-every`` optimiser steps (HSN and the baselines): per top-level
module, the update ratio ``||delta theta|| / ||theta||`` of that optimiser step and the module's
share of the (clipped) gradient norm, which clipping does not change. HSN only: one no-grad forward
of a fixed diagnostic window (the first four documents, one window, evaluation mode) gives the cell
diversity (mean pairwise ``1 - cos(h_i, h_j)`` over the cells), the fraction of state coordinates
with ``|h| >= 0.99``, the mean ``|h|`` (at answer positions and over every valid position) and the
gate quantiles ``b`` p10/p50/p90 over valid positions (not for ``gru_cell``).

Writes ``<out>/<name>/`` (the trainer's run directory with ``history.json``) and
``<out>/<name>_summary.json``: the evaluations, the first evaluation at or above 0.90 and 0.99,
the diagnostics, the measured numbers, the overrides and the GPU processes seen at the start and the
end of the run (a run with another process on the card is marked ``shared``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import types
from pathlib import Path
from typing import Any

import torch
from hsn.config import RunConfig, load_config
from hsn.data.dataset import ByteDataset, collate_windows
from hsn.device import DeviceContext, resolve, seed_everything
from hsn.model.embeddings import init_weight_
from hsn.model.hsn import HSN
from hsn.runner import build_model, make_trainer
from torch import Tensor, nn

REPO = Path(__file__).resolve().parents[1]
VARIANTS = (
    "pdg",
    "gru_cell",
    "gru_cell_linear_head",
    "gru_cell_direct_input",
    "cell_offset",
    "cell_embed",
    "cell_embed_linear_head",
    "linear_head",
    "decoder_fc1_fan_in",
)
THRESHOLDS = (0.90, 0.99)


# --------------------------------------------------------------------------------------------------
# GPU status (shared or idle)
# --------------------------------------------------------------------------------------------------


def gpu_processes() -> list[dict[str, str]]:
    """The compute processes ``nvidia-smi`` lists (pid, memory, name); empty without a GPU."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            rows.append({"pid": parts[0], "memory": parts[1], "name": parts[2]})
    return rows


def card_status(own_pid: int, seen: list[list[dict[str, str]]], min_mib: int = 200) -> str:
    """``shared`` when a process other than this one held a compute context in any listing,
    unless it is a small non-Python context (a desktop process such as a file manager holding
    less than ``min_mib`` MiB); any other Python process counts, whatever its memory."""

    def counts(row: dict[str, str]) -> bool:
        if row["pid"] == str(own_pid):
            return False
        if "python" in row["name"].lower():
            return True
        try:
            return int(row["memory"].split()[0]) >= min_mib
        except (ValueError, IndexError):
            return True  # unreadable: count it

    return "shared" if any(counts(row) for listing in seen for row in listing) else "idle"


# --------------------------------------------------------------------------------------------------
# The recipe (g1_harness_overfit.py @ 9703db5)
# --------------------------------------------------------------------------------------------------


def recipe_overrides(
    *,
    profile: str,
    lr: float,
    steps: int,
    eval_every: int,
    seed: int,
    run_id: str,
    extra: list[str],
) -> list[str]:
    """The overfit recipe's configuration overrides, then ``extra`` (later entries win)."""
    return [
        f"profile={profile}",
        "hardware.autocast_dtype=fp32",
        "train.precision=fp32",
        "train.weight_decay=0",
        f"train.lr={lr}",
        "train.schedule=constant",
        "train.warmup_steps=200",
        "train.batch=4",
        "train.micro_batch=4",
        "train.grad_accum=1",
        f"train.steps={steps}",
        f"train.eval_every={eval_every}",
        f"train.ckpt_every={steps}",
        "train.stats_every=50",
        f"seed={seed}",
        f"run_id={run_id}",
        "data.tasks=[delayed]",
        "data.variants=[in_range]",
        *extra,
    ]


def overfit_docs(cfg: RunConfig, n_docs: int) -> ByteDataset:
    """The first ``n_docs`` delayed/in_range documents of the train split."""
    manifest = Path(str(cfg.data.manifest))
    full = ByteDataset.load(
        manifest, "train", ["delayed"], ["in_range"], expected_unroll=cfg.train.unroll
    )
    return ByteDataset(list(full.examples[:n_docs]))


# --------------------------------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------------------------------


class CellIdentity(nn.Module):
    """``norm(v) + o[cell]`` on the dense round's ``[B, N, 4d]`` input; ``o`` fixed or learned."""

    def __init__(self, norm: nn.Module, offset: Tensor, learned: bool) -> None:
        super().__init__()
        self.norm = norm
        self.n_cells = int(offset.shape[0])
        if learned:
            self.offset = nn.Parameter(offset.clone())
        else:
            self.register_buffer("offset", offset.clone())

    def forward(self, v: Tensor) -> Tensor:
        out: Tensor = self.norm(v)
        if v.ndim != 3 or v.shape[1] != self.n_cells:
            raise ValueError(
                f"CellIdentity applies to the dense round's [B, N={self.n_cells}, 4d] input, got "
                f"{tuple(v.shape)}"
            )
        return out + self.offset.to(out.dtype)[None]


class LinearHead(nn.Module):
    """The GRU baseline's readout on HSN's summary: ``Linear(LayerNorm(q))`` applied to the decoder
    input ``[q; u]`` (``u`` is not read). The linear layer is named ``fc1`` because
    :attr:`HSN.device` and :attr:`HSN.dtype` read ``decoder.fc1.weight``."""

    def __init__(self, d: int, vocab: int) -> None:
        super().__init__()
        self.d = d
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, vocab)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.fc1(self.norm(x[..., : self.d]))
        return out


def _param_report_with(extra_rows: dict[str, nn.Module]) -> Any:
    """``HSN.param_report`` with the variant's extra rows (modules outside every Appendix C row)
    added to its honesty check (``total_trainable = total + edges + extras``)."""

    def param_report(self: HSN) -> dict[str, int]:
        rows: dict[str, nn.Module] = {
            "byte_emb": self.byte_emb,
            "ctx_emb": self.ctx_emb,
            "encoder": self.encoder,
            "predictor": self.predictor,
            "shared": self.shared,
            "units": self.cell.units,
            "w_out": self.cell.W_out,
            "v_msg": self.cell.V,
            "gate": self.cell.gate,
            "cell_norm": self.cell.norm,
            "scorer": self.scorer,
            "halting": self.halting,
            "decoder": self.decoder,
        }
        report = {name: sum(p.numel() for p in m.parameters()) for name, m in rows.items()}
        report["total"] = sum(report.values())
        report["edges"] = self.cell.graph.edge_logit.numel()
        extra = 0
        for name, module in extra_rows.items():
            report[name] = sum(p.numel() for p in module.parameters())
            extra += report[name]
        report["total_trainable"] = sum(p.numel() for p in self.parameters())
        if report["total_trainable"] != report["total"] + report["edges"] + extra:
            raise RuntimeError(f"param_report (variant): unaccounted parameters in {report}")
        return report

    return param_report


def install_variant(model: nn.Module, variant: str, seed: int, offset_std: float) -> dict[str, Any]:
    """Install ``variant`` on ``model`` (an :class:`HSN` unless ``pdg``); returns its record."""
    if variant == "pdg":
        return {"variant": "pdg"}
    if not isinstance(model, HSN):
        raise TypeError(f"--variant {variant} needs an HSN model, got {type(model).__name__}")
    cell = model.cell
    d, N = cell.d, cell.n_cells
    gen = torch.Generator().manual_seed(seed)
    if variant == "linear_head":
        # V1: the PDG model with HSN's MLP decoder D([q; u]) replaced by the GRU baseline's readout
        # Linear(LayerNorm(q)); the decoder row of HSN.param_report counts the new head
        model.decoder = LinearHead(d, model.cfg.vocab_size)
        return {"variant": variant, "decoder": "LayerNorm(q) -> Linear(d, V)"}
    if variant == "decoder_fc1_fan_in":
        # V2: the PDG model with D's first layer (the read of [q; u]) at the A1 fan-in init instead
        # of model.init (trunc_normal_0.02); D's output layer keeps 0.02
        init_weight_(model.decoder.fc1.weight, "fan_in")
        return {"variant": variant, "decoder_fc1_init": "fan_in"}
    if variant in ("cell_offset", "cell_embed", "cell_embed_linear_head"):
        learned = variant != "cell_offset"
        offset = torch.randn(N, 4 * d, generator=gen) * offset_std
        identity = CellIdentity(cell.norm, offset, learned=learned)
        # step_dense reads self.norm(v_raw); the cell_norm row of HSN.param_report now holds the
        # RMSNorm gain plus, for the learned variants, the per-cell vectors, so its check balances
        cell.norm = identity
        record = {
            "variant": variant,
            "offset_std": offset_std,
            "extra_params": N * 4 * d if learned else 0,
        }
        if variant == "cell_embed_linear_head":
            # S2 + V1: learned cell identity with the baseline readout (V1 removed the state
            # saturation that re-synchronised the S2 cells in E3); the decoder row counts the head
            model.decoder = LinearHead(d, model.cfg.vocab_size)
            record["decoder"] = "LayerNorm(q) -> Linear(d, V)"
        return record
    if variant in ("gru_cell", "gru_cell_linear_head", "gru_cell_direct_input"):
        direct = variant == "gru_cell_direct_input"
        # PyTorch's default initialisation (as nn.GRU in gru_2m); the direct variant reads width d
        gru = nn.GRUCell(d if direct else 4 * d, d)
        cell.gru_update = gru
        if direct:
            # W2: the cell reads emb(x_t) + ctx_emb(ctx_t), the GRU baseline's input, instead of
            # norm([u; eps; h_i; m_i]); E, P, S and eps still run and feed the decoder and L_pred
            original_shared_path = model._shared_path

            def shared_path(x_t: Tensor, ctx_t: Tensor, state: Any) -> Any:
                out = original_shared_path(x_t, ctx_t, state)
                cell._direct_input = model.byte_emb(x_t) + model.ctx_emb(ctx_t)
                return out

            model._shared_path = shared_path  # type: ignore[method-assign]

        def step_dense(
            u: Tensor, eps: Tensor, state: Any, alpha: Tensor
        ) -> tuple[Tensor, dict[str, Tensor]]:
            B, NN, dd = cell._check_inputs(u, eps, state, alpha)
            with cell.graph.step():
                h_prev = state.h
                if direct:
                    x_in = cell._direct_input.to(h_prev.dtype)  # [B, d]
                    v = x_in[:, None, :].expand(B, NN, dd)
                    info: dict[str, Tensor] = {"v": v, "h_prev": h_prev}
                    width = dd
                else:
                    m = cell.graph.messages_dense(h_prev, cell.V)
                    v_raw = torch.cat(
                        [
                            u[:, None, :].expand(B, NN, dd),
                            eps[:, None, :].expand(B, NN, dd),
                            h_prev,
                            m,
                        ],
                        dim=-1,
                    )
                    v = cell.norm(v_raw)
                    info = {"m": m, "v": v, "v_raw": v_raw, "h_prev": h_prev}
                    width = 4 * dd
                with torch.autocast(device_type=h_prev.device.type, enabled=False):
                    h_new = cell.gru_update(
                        v.reshape(B * NN, width).to(h_prev.dtype), h_prev.reshape(B * NN, dd)
                    ).reshape(B, NN, dd)
            return h_new, info

        cell.step_dense = step_dense
        extra: dict[str, nn.Module] = {"gru_update": gru}
        record: dict[str, Any] = {"variant": variant}
        if variant == "gru_cell_linear_head":
            # W1: the GRU baseline's readout (LayerNorm + linear, std 0.02, bias 0) on q only,
            # in place of HSN's MLP decoder D([q; u])
            model.decoder = LinearHead(d, model.cfg.vocab_size)
            record["decoder"] = "LayerNorm(q) -> Linear(d, V)"
        model.param_report = types.MethodType(_param_report_with(extra), model)
        record["extra_params"] = sum(p.numel() for p in gru.parameters())
        return record
    raise ValueError(f"unknown variant {variant!r}; known: {VARIANTS}")


# --------------------------------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------------------------------


def module_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    groups: dict[str, list[nn.Parameter]] = {}
    for name, child in model.named_children():
        params = [p for p in child.parameters() if p.requires_grad]
        if params:
            groups[name] = params
    return groups


def _norm(tensors: list[Tensor]) -> float:
    if not tensors:
        return 0.0
    return float(
        torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(t.float()) for t in tensors])
        )
    )


class Diagnostics:
    """Per-module update ratios and gradient shares on the optimiser steps it samples, and (HSN)
    the state statistics of one fixed window."""

    def __init__(
        self,
        model: nn.Module,
        cfg: RunConfig,
        ctx: DeviceContext,
        docs: ByteDataset,
        every: int,
        n_docs: int = 4,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.ctx = ctx
        self.every = every
        self.groups = module_groups(model)
        self.records: list[dict[str, Any]] = []
        self.calls = 0
        self.window: dict[str, Tensor] | None = None
        if isinstance(model, HSN):
            examples = [docs.examples[i] for i in range(min(n_docs, len(docs)))]
            batch = collate_windows(
                examples, cfg.train.unroll, batch=len(examples), align_docs=cfg.data.align_docs
            )
            T = cfg.train.unroll
            self.window = {k: v[:, :T].to(ctx.device) for k, v in batch.items()}

    def wrap(self, optimizer: torch.optim.Optimizer) -> None:
        original = optimizer.step

        def step(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls % self.every != 0:
                return original(*args, **kwargs)
            with torch.no_grad():
                before = {n: [p.detach().clone() for p in ps] for n, ps in self.groups.items()}
                grads = {
                    n: [p.grad.detach() for p in ps if p.grad is not None]
                    for n, ps in self.groups.items()
                }
            out = original(*args, **kwargs)
            record: dict[str, Any] = {"step": self.calls}
            with torch.no_grad():
                g_total = _norm([g for gs in grads.values() for g in gs])
                for n, ps in self.groups.items():
                    delta = _norm([p.detach() - b for p, b in zip(ps, before[n], strict=True)])
                    weight = _norm([p.detach() for p in ps])
                    record[f"update_ratio/{n}"] = delta / weight if weight > 0 else math.nan
                    record[f"grad_share/{n}"] = _norm(grads[n]) / g_total if g_total > 0 else 0.0
            record.update(self.state_stats())
            self.records.append(record)
            return out

        optimizer.step = step  # type: ignore[method-assign]

    @torch.no_grad()
    def state_stats(self) -> dict[str, float]:
        model = self.model
        if not isinstance(model, HSN) or self.window is None:
            return {}
        w = self.window
        captured: list[tuple[Tensor, Tensor | None]] = []
        cell = model.cell
        original = cell.step_dense

        def capture(u: Tensor, eps: Tensor, state: Any, alpha: Tensor) -> Any:
            h_new, info = original(u, eps, state, alpha)
            captured.append((h_new.detach(), info.get("b")))
            return h_new, info

        cell.step_dense = capture
        was_training = model.training
        model.eval()
        try:
            state = model.init_state(int(w["tokens"].shape[0]))
            model(w["tokens"], w["context"], state, w["reset"])
        finally:
            cell.step_dense = original
            model.train(was_training)
        h = torch.stack([c[0] for c in captured], dim=1).float()  # [B, T, N, d]
        valid = w["doc"] >= 0  # [B, T]
        answer = w["loss_mask"]
        out: dict[str, float] = {}
        for tag, mask in (("answer", answer), ("valid", valid)):
            if not bool(mask.any()):
                continue
            hs = h[mask]  # [P, N, d]
            unit = hs / hs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cos = unit @ unit.transpose(1, 2)  # [P, N, N]
            n = cos.shape[-1]
            off = (cos.sum(dim=(1, 2)) - cos.diagonal(dim1=1, dim2=2).sum(-1)) / (n * (n - 1))
            out[f"diversity/{tag}"] = float((1 - off).mean())
            out[f"sat99/{tag}"] = float((hs.abs() >= 0.99).float().mean())
            out[f"abs_h/{tag}"] = float(hs.abs().mean())
            out[f"cell_spread/{tag}"] = float((hs - hs[:, :1]).abs().amax())
            # participation ratio of the [N, d] cell-state matrix: 1 when every cell holds the
            # same direction, up to min(N, d) when the cells span independent directions
            sv2 = torch.linalg.svdvals(hs).square()
            pr = sv2.sum(-1).square() / sv2.square().sum(-1).clamp_min(1e-24)
            out[f"rank_pr/{tag}"] = float(pr.mean())
        if captured[0][1] is not None:
            b = torch.stack([c[1] for c in captured], dim=1).float()  # type: ignore[misc]
            bv = b[valid].reshape(-1)
            if bv.numel() > 2_000_000:
                bv = bv[torch.randperm(bv.numel(), device=bv.device)[:2_000_000]]
            q = torch.quantile(bv, torch.tensor([0.1, 0.5, 0.9], device=bv.device))
            out.update({"gate/p10": float(q[0]), "gate/p50": float(q[1]), "gate/p90": float(q[2])})
        return out


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(REPO / "configs" / "experiments" / "g1_ref_a1.yaml"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="rtx3060")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--eval-every", type=int, default=125)
    ap.add_argument("--n-docs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--variant", choices=VARIANTS, default="pdg")
    ap.add_argument("--offset-std", type=float, default=0.5)
    ap.add_argument("--diag-every", type=int, default=50)
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen = [gpu_processes()]
    overrides = recipe_overrides(
        profile=args.profile,
        lr=args.lr,
        steps=args.steps,
        eval_every=args.eval_every,
        seed=args.seed,
        run_id=f"eff_{args.name}",
        extra=args.override,
    )
    cfg = load_config(args.config, overrides=overrides)
    ctx = resolve(cfg.hardware)
    docs = overfit_docs(cfg, args.n_docs)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    variant = install_variant(model, args.variant, args.seed, args.offset_std)
    trainer = make_trainer(model, cfg, ctx, run_dir=out / args.name)
    diag = Diagnostics(model, cfg, ctx, docs, args.diag_every)
    diag.wrap(trainer.optimizer)
    started = time.perf_counter()
    history = trainer.fit(docs, docs, args.steps)
    wall = time.perf_counter() - started
    seen.append(gpu_processes())
    evals = [
        {"step": int(e["step"]), "train_accuracy": float(e["accuracy"]), "loss": float(e["loss"])}
        for e in history.evals
    ]
    first = {
        f"first_ge_{t:.2f}": next((e["step"] for e in evals if e["train_accuracy"] >= t), None)
        for t in THRESHOLDS
    }
    seconds = sorted(s.seconds for s in history.steps)
    summary = {
        "name": args.name,
        "config": args.config,
        "overrides": overrides,
        **variant,
        "n_docs": len(docs),
        "profile": args.profile,
        "autocast": history.autocast,
        "lr": cfg.train.lr,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "evals": evals,
        **first,
        "final_train_accuracy": evals[-1]["train_accuracy"] if evals else None,
        "wall_s": round(wall, 1),
        "step_seconds_p50": seconds[len(seconds) // 2] if seconds else None,
        "card": card_status(os.getpid(), seen),
        "gpu_processes": seen,
        "param_count": history.measured.get("param_count"),
        "measured": history.measured,
        "diagnostics": diag.records,
    }
    with open(out / f"{args.name}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    with open(out / args.name / "history.json", "w") as fh:
        json.dump(
            {
                "steps": [vars(s) for s in history.steps],
                "evals": history.evals,
                "measured": history.measured,
            },
            fh,
            default=str,
        )
    brief = {k: v for k, v in summary.items() if k not in ("measured", "diagnostics", "overrides")}
    print(json.dumps(brief, indent=1, default=str))


if __name__ == "__main__":
    main()
