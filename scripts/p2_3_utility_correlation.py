#!/usr/bin/env python
"""P2.3 definition of done: does the scorer's ``U`` rank the measured loss reductions?

    python scripts/p2_3_utility_correlation.py runs/p2_3/p2_3_kv_probes-cpu-s0 --horizon 16
    python scripts/p2_3_utility_correlation.py runs/p2_3/p2_3_kv_probes-cpu-s0 --horizon end
    python scripts/p2_3_utility_correlation.py runs/p2_3/p2_3_kv_probes-cpu-s0 --init --horizon 16

Loads the run's ``config.resolved.yaml`` and its last checkpoint (``--checkpoint`` to choose one;
``--init`` builds the run's step-0 weights from its seed instead), puts the model in evaluation
mode (hard decisions, no exploration: the policy that would be deployed) and runs
:func:`hsn.train.probes.measure_utility_correlation` over the dev split of the run's tasks — every
dev document once, in loader order, ``train.batch`` rows per window, the state carried as in
evaluation. Per window one position ``t`` is drawn among those whose horizon holds an answer
(``--horizon 16``: the chunk's ``train.probe_horizon``; ``end``: the pre-declared secondary
measurement, from ``t`` to the end of the window), with ``train.n_probe`` unselected alternatives
plus the selected slots; each valid pair gives the scorer's logit ``U`` at (row, cell, unit) and the
measured loss reduction. The primary number is the pooled Spearman correlation with its 95 %
bootstrap interval over windows. Probes on dev are an evaluation: nothing reaches a weight, the EMA
features or a training target. The JSON goes to ``<run_dir>/p2_3_utility_<which>_h<horizon>.json``
(``--out`` to choose); run directories are local (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from hsn.config import load_config  # noqa: E402
from hsn.device import resolve, seed_everything  # noqa: E402
from hsn.model.hsn import HSN  # noqa: E402
from hsn.runner import build_model, load_datasets  # noqa: E402
from hsn.train.checkpoint import latest_checkpoint, load_checkpoint  # noqa: E402
from hsn.train.loop import StreamLoader  # noqa: E402
from hsn.train.probes import measure_utility_correlation, window_split  # noqa: E402

WINDOW_FIELDS = ("tokens", "targets", "context", "loss_mask", "reset", "doc")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path, help="the run directory (config.resolved.yaml)")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--checkpoint", type=Path, help="a step_*.pt of the run (default: last)")
    which.add_argument("--init", action="store_true", help="the run's step-0 weights")
    parser.add_argument("--horizon", default="16", help="an int, or 'end' (to the window's end)")
    parser.add_argument("--seed", type=int, default=0, help="probe-position and bootstrap seed")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--profile", help="hardware profile override (default: the run's)")
    parser.add_argument("--out", type=Path, help="output JSON (default: inside the run dir)")
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    overrides = [f"profile={args.profile}"] if args.profile else []
    cfg = load_config(run_dir / "config.resolved.yaml", overrides=overrides)
    if cfg.model_name != "hsn":
        raise SystemExit(f"{run_dir} trained {cfg.model_name!r}; the probes need the HSN")
    horizon: int | None = None if args.horizon == "end" else int(args.horizon)
    ctx = resolve(cfg.hardware)
    seed_everything(cfg.seed)  # the runner's construction: the same step-0 weights
    model = build_model(cfg)
    assert isinstance(model, HSN)
    model.to(device=ctx.device, dtype=ctx.master_dtype)
    if args.init:
        source = "init"
    else:
        checkpoint = args.checkpoint or latest_checkpoint(run_dir)
        if checkpoint is None:
            raise SystemExit(f"{run_dir} has no checkpoint")
        load_checkpoint(checkpoint, model)
        source = checkpoint.name
    model.eval()

    _, dev_ds = load_datasets(cfg)
    loader = StreamLoader(
        dev_ds,
        batch=cfg.train.batch,
        unroll=cfg.train.unroll,
        seed=cfg.seed,
        shuffle=False,
        align_docs=cfg.data.align_docs,
        epochs=1,
    )

    def windows() -> Iterator[dict[str, Any]]:
        for window in loader:
            split = window_split(dev_ds, window["doc"])
            yield {
                **{key: window[key].to(ctx.device) for key in WINDOW_FIELDS},
                "split": split,
            }

    started = time.perf_counter()
    result = measure_utility_correlation(
        model,
        windows(),
        n_probe=cfg.train.n_probe,
        horizon=horizon,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )
    seconds = time.perf_counter() - started
    record = {
        **result.to_dict(),
        "run_dir": str(run_dir),
        "weights": source,
        "split": "dev",
        "tasks": list(cfg.data.tasks),
        "variants": list(cfg.data.variants),
        "n_docs": len(dev_ds),
        "mode": "eval",
        "device": str(ctx.device),
        "threads": torch.get_num_threads(),
        "seconds": seconds,
        "lambda_u": cfg.train.lambda_u,
        "threshold": 0.2,
        "passes": result.spearman > 0.2,
    }
    name = "init" if args.init else "trained"
    out = args.out or run_dir / f"p2_3_utility_{name}_h{args.horizon}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
