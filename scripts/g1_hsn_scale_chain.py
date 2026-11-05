#!/usr/bin/env python
"""HSN gradient-flow diagnosis, supporting measurement: the magnitude of every stage of the write
path on real inputs, at initialisation and at the trained checkpoint.

    python scripts/g1_hsn_scale_chain.py --run <HSN run dir> --out evidence/G1/diagnosis/gradflow

One real training window (batch 16 x 256, fp32, CPU) through the model; `cell.step_dense` is wrapped
to record its intermediates per position: |v| (the cell input), |unit_out| (the unit MLPs' outputs),
|mixed| (their alpha-mixture), |c| = |tanh(W_out(mixed))| (the write candidate), b (the gate),
|h| (the state after the write) and |q| (the summary the decoder reads); plus |emb|, |z|, |u|,
|eps| from the shared path (hooks on byte_emb and encoder, the cell call for u and eps). RMS over
elements and non-padding positions. The window runs on the profile's device in fp32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from hsn.config import load_config
from hsn.device import resolve, seed_everything
from hsn.runner import build_model, load_datasets, make_trainer
from hsn.train.checkpoint import latest_checkpoint, load_checkpoint


def rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt())


def measure(model: Any, ctx: Any, window: dict[str, torch.Tensor]) -> dict[str, float]:
    device = ctx.device
    rec: dict[str, list[float]] = {
        k: [] for k in ("v", "unit_out", "mixed", "c", "b_mean", "h", "q", "dh")
    }
    shared: dict[str, list[float]] = {"emb": [], "z": [], "u": [], "eps": []}
    orig = model.cell.step_dense

    def wrapped(u: torch.Tensor, eps: torch.Tensor, state: Any, alpha: torch.Tensor) -> Any:
        h_new, info = orig(u, eps, state, alpha)
        rec["v"].append(rms(info["v"]))
        rec["unit_out"].append(rms(info["unit_out"]))
        rec["mixed"].append(rms(info["mixed"]))
        rec["c"].append(rms(info["c"]))
        rec["b_mean"].append(float(info["b"].float().mean()))
        rec["h"].append(rms(h_new))
        rec["q"].append(rms(h_new.mean(dim=1)))
        rec["dh"].append(rms(h_new - info["h_prev"]))
        shared["u"].append(rms(u))
        shared["eps"].append(rms(eps))
        return h_new, info

    hooks = [
        model.byte_emb.register_forward_hook(lambda m, i, o: shared["emb"].append(rms(o))),
        model.encoder.register_forward_hook(lambda m, i, o: shared["z"].append(rms(o))),
    ]
    model.cell.step_dense = wrapped
    model.eval()
    with torch.no_grad():
        tok = window["tokens"].to(device)
        model(
            tok,
            window["context"].to(device),
            model.init_state(tok.shape[0], device),
            window["reset"].to(device),
        )
    for h in hooks:
        h.remove()
    model.cell.step_dense = orig
    out = {}
    for k, v in {**shared, **rec}.items():
        if v:
            out[k + "_rms_mean_over_positions"] = sum(v) / len(v)
            out[k + "_rms_last_position"] = v[-1]
    weights = {
        "byte_emb": rms(model.byte_emb.weight),
        "W_out": rms(model.cell.W_out.weight),
        "gate_weight": rms(model.cell.gate.weight),
        "gate_bias": rms(model.cell.gate.bias),
        "units_W1": rms(model.cell.units.W1),
        "units_W2": rms(model.cell.units.W2),
    }
    out.update({"weight_rms_" + k: v for k, v in weights.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(
        run_dir / "config.resolved.yaml",
        overrides=["hardware.autocast_dtype=fp32", "train.precision=fp32"],
    )
    ctx = resolve(cfg.hardware)
    train_ds, _ = load_datasets(cfg)
    results: dict[str, Any] = {"run": str(run_dir)}
    for label in ("init", "trained"):
        seed_everything(cfg.seed)
        model = build_model(cfg)
        if label == "trained":
            ck = latest_checkpoint(run_dir)
            assert ck is not None
            load_checkpoint(ck, model)
        model.to(device=ctx.device, dtype=ctx.master_dtype)
        trainer = make_trainer(model, cfg, ctx, run_dir=out / f"_scratch_scale_{label}")
        window = next(iter(trainer._loader(train_ds, shuffle=True, epochs=1)))
        results[label] = measure(model, ctx, window)
        r = results[label]
        keys = ("emb", "z", "u", "eps", "v", "unit_out", "mixed", "c", "h", "q", "dh")
        print(
            f"[{label}] "
            + "  ".join(f"{k} {r[k + '_rms_mean_over_positions']:.2e}" for k in keys)
            + f"  b {r['b_mean_rms_mean_over_positions']:.3f}",
            flush=True,
        )
        print(
            f"[{label}] weights rms: "
            + "  ".join(f"{k} {v:.3e}" for k, v in r.items() if k.startswith("weight_rms_")),
            flush=True,
        )
        del model, trainer
    with open(out / "scale_chain.json", "w") as fh:
        json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
