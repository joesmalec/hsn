#!/usr/bin/env python3
"""P2.10: the step time of a training-window variant, and how its gradients compare.

``time`` trains through the real :class:`~hsn.train.tbptt.TBPTTTrainer` (statistics, evaluation
and checkpoints off, so every timed step is a plain training step) and reports the steady-state
step time of ``hardware.window`` = ``autograd`` (the compiled window of P2.8) or ``split``
(:mod:`hsn.train.split`)::

    python scripts/bench_window.py time --window split --warm 300 --steps 600 --out report.json

``grads`` compares the gradients of one window from the same parameters and start state:
the split window against the autograd window in fp32, and both bf16 variants against the fp32
autograd reference (relative L2 error and cosine per parameter group)::

    python scripts/bench_window.py grads --out report.json

One process per measurement; nothing else should run on the card meanwhile.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from hsn.config import RunConfig, config_hash, load_config  # noqa: E402
from hsn.device import resolve, seed_everything  # noqa: E402
from hsn.model.hsn import HSN  # noqa: E402
from hsn.runner import build_model, load_datasets, make_trainer  # noqa: E402
from hsn.train.amp import autocast_ctx  # noqa: E402
from hsn.train.split import SplitWindow  # noqa: E402
from hsn.train.tbptt import TBPTTTrainer  # noqa: E402

RECIPE = "configs/experiments/hsn_recipe_a6.yaml"
OFF = ("train.stats_every=1000000", "train.eval_every=1000000", "train.ckpt_every=1000000")


def _config(args: argparse.Namespace, *extra: str) -> RunConfig:
    return load_config(
        ROOT / args.config,
        overrides=[
            f"profile={args.profile}",
            "hardware.compile_mode=reduce-overhead",
            *OFF,
            *extra,
            *args.override,
        ],
    )


def _trainer(cfg: RunConfig) -> tuple[TBPTTTrainer, Any, Any]:
    ctx = resolve(cfg.hardware)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    if not isinstance(model, HSN):
        raise TypeError("bench_window.py times the HSN model")
    run_dir = Path(tempfile.mkdtemp(prefix="hsn_bench_window_"))
    trainer = make_trainer(model, cfg, ctx, run_dir=run_dir)
    assert isinstance(trainer, TBPTTTrainer)
    trainer.evaluate = lambda *a, **k: {}  # type: ignore[method-assign]
    trainer.save = lambda *a, **k: run_dir / "unused.pt"  # type: ignore[method-assign]
    train_ds, dev_ds = load_datasets(cfg)
    return trainer, train_ds, dev_ds


def _summary(seconds: list[float]) -> dict[str, float]:
    ordered = sorted(seconds)
    return {
        "steps": len(seconds),
        "mean_ms": round(1e3 * statistics.fmean(seconds), 3),
        "median_ms": round(1e3 * statistics.median(seconds), 3),
        "p95_ms": round(1e3 * ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
        "min_ms": round(1e3 * ordered[0], 3),
        "max_ms": round(1e3 * ordered[-1], 3),
    }


def time_window(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _config(args, f"hardware.window={args.window}")
    trainer, train_ds, dev_ds = _trainer(cfg)
    started = time.perf_counter()
    trainer.fit(train_ds, dev_ds, args.warm)  # compile, record, allocator warm-up
    warm_s = time.perf_counter() - started
    if not args.no_gc_freeze:  # as protocol.gc_freeze: no full collections in the timed steps
        gc.collect()
        gc.freeze()
    torch.cuda.reset_peak_memory_stats()
    # the last step of a fit is a statistics step (extra passes): one more, left out of the sample
    history = trainer.fit(train_ds, dev_ds, args.warm + args.steps + 1)
    timed = [r for r in history.steps if args.warm < r.step <= args.warm + args.steps]
    seconds = [r.seconds for r in timed]
    compiled = trainer.compiled_window
    report: dict[str, Any] = {
        "window": args.window,
        "config": args.config,
        "overrides": list(args.override),
        "config_hash": config_hash(cfg),
        "batch": cfg.train.batch,
        "unroll": cfg.train.unroll,
        "precision": cfg.train.precision,
        "torch": torch.__version__,
        "warm_steps": args.warm,
        "warm_wall_s": round(warm_s, 2),
        "compile_seconds": round(compiled.compile_seconds, 2) if compiled is not None else 0.0,
        "gc_freeze": not args.no_gc_freeze,
        "step_time": _summary(seconds),
        "blocks_mean_ms": [
            round(1e3 * statistics.fmean(seconds[i : i + 200]), 3)
            for i in range(0, len(seconds), 200)
        ],
        "loss_first": timed[0].loss,
        "loss_last": timed[-1].loss,
        "loss_mean_last_100": statistics.fmean(r.loss for r in timed[-100:]),
        "accuracy_mean_last_100": statistics.fmean(r.accuracy for r in timed[-100:]),
        "skipped_nonfinite_steps": int(getattr(trainer, "skipped_nonfinite_steps", 0)),
        "peak_bytes": int(torch.cuda.max_memory_allocated()),
    }
    split = getattr(trainer, "_split", None)
    if split is not None:
        report["last_phase_ms"] = {k: round(1e3 * v, 3) for k, v in split.last_phase_seconds.items()}
    return report


# --------------------------------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------------------------------


def _groups(name: str) -> str:
    head = name.split(".")[0]
    if head == "cell":
        return ".".join(name.split(".")[:2])
    return head


def _window_grads(
    trainer: TBPTTTrainer, window: Any, batch: dict[str, torch.Tensor], state: Any
) -> tuple[dict[str, torch.Tensor], float]:
    model = trainer.hsn
    model.zero_grad(set_to_none=True)
    with autocast_ctx(trainer.ctx):
        logits, _, info = window(batch["tokens"], batch["context"], state, batch["reset"])
    mask = batch["loss_mask"]
    task = torch.nn.functional.cross_entropy(logits[mask].float(), batch["targets"][mask])
    pred = (info["z_hat"].float() - info["z_target"].float()).square().mean()
    loss = task + pred
    loss.backward()  # type: ignore[no-untyped-call]
    grads = {
        n: p.grad.detach().double().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return grads, float(loss)


def _compare(a: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    names = sorted(ref)
    assert sorted(a) == names, (sorted(set(a) ^ set(ref)),)
    groups = sorted({_groups(n) for n in names})
    for group in [*groups, "all"]:
        members = [n for n in names if group == "all" or _groups(n) == group]
        x = torch.cat([a[n].flatten() for n in members])
        y = torch.cat([ref[n].flatten() for n in members])
        out[group] = {
            "rel_l2": float((x - y).norm() / y.norm().clamp_min(1e-300)),
            "cosine": float(torch.dot(x, y) / (x.norm() * y.norm()).clamp_min(1e-300)),
            "ref_norm": float(y.norm()),
        }
    return out


def grads(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {"config": args.config, "overrides": list(args.override)}
    results: dict[str, dict[str, torch.Tensor]] = {}
    saved: dict[str, Any] | None = None
    batch: dict[str, torch.Tensor] | None = None
    start: Any = None
    for precision in ("fp32", "bf16"):
        cfg = _config(args, f"train.precision={precision}", "hardware.window=autograd")
        trainer, train_ds, dev_ds = _trainer(cfg)
        model = trainer.hsn
        if saved is None:
            if args.train_steps:  # move the weights off their initial values first
                trainer.fit(train_ds, dev_ds, args.train_steps)
            saved = copy.deepcopy(model.state_dict())
            g = torch.Generator().manual_seed(args.seed)
            B, T = cfg.train.batch, cfg.train.unroll
            reset = torch.zeros(B, T, dtype=torch.bool)
            reset[:, 0] = True
            reset[0, 0] = False
            reset[1, T // 3] = True
            cpu = {
                "tokens": torch.randint(0, cfg.model.vocab_size, (B, T), generator=g),
                "context": torch.randint(0, cfg.model.n_context, (B, T), generator=g),
                "reset": reset,
                "targets": torch.randint(0, cfg.model.vocab_size, (B, T), generator=g),
                "loss_mask": torch.rand(B, T, generator=g) < 0.1,
            }
            batch = {k: v.to(trainer.ctx.device) for k, v in cpu.items()}
            # a realistic carried state: two windows of the model's own forward
            model.eval()
            with torch.no_grad(), autocast_ctx(trainer.ctx):
                state = model.init_state(B, trainer.ctx.device)
                for _ in range(2):
                    _, state, _ = model(batch["tokens"], batch["context"], state, batch["reset"])
            start = state.detach()
        else:
            model.load_state_dict(saved)
        model.train()
        model.set_train_step(args.train_steps)
        assert batch is not None
        compiled = trainer.compiled_window
        assert compiled is not None
        split = SplitWindow(model, cfg, compiled)
        for name, window in (("autograd", compiled), ("split", split)):
            for _ in range(3):  # compile, record, then the measured pass
                got, loss = _window_grads(trainer, window, batch, start)
            results[f"{name}_{precision}"] = got
            report[f"loss_{name}_{precision}"] = loss
    ref = results["autograd_fp32"]
    report["split_fp32_vs_autograd_fp32"] = _compare(results["split_fp32"], ref)
    report["autograd_bf16_vs_autograd_fp32"] = _compare(results["autograd_bf16"], ref)
    report["split_bf16_vs_autograd_fp32"] = _compare(results["split_bf16"], ref)
    report["split_bf16_vs_autograd_bf16"] = _compare(
        results["split_bf16"], results["autograd_bf16"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["time", "grads"])
    parser.add_argument("--config", default=RECIPE)
    parser.add_argument("--profile", default="rtx3060")
    parser.add_argument("--window", choices=["autograd", "split"], default="split")
    parser.add_argument("--warm", type=int, default=300)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-gc-freeze", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = time_window(args) if args.command == "time" else grads(args)
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
