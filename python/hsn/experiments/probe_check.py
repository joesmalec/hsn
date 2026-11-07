"""``probe-check`` (P1.11): the batched edit probe against the audit's per-document probe on a
trained checkpoint of one experiment.

The model is rebuilt from the arm's stored configuration (seeded as the run, the variant installed
the same way), the run's last checkpoint is loaded, and both probes run on the dev documents under
the run's autocast. Batching changes GEMM row counts, so in bf16 a probe whose top-2 logit margin is
of the order of the rounding can flip; every flip is reported with both margins (a tie, not a bug,
when the margin is that small), together with both verdicts and both wall times.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from hsn.experiments.db import Database, loads

__all__ = ["probe_check"]


def probe_check(db: Database, exp_id: int, *, batch_rows: int = 64) -> dict[str, Any]:
    from hsn.config import RunConfig
    from hsn.data.dataset import ByteDataset
    from hsn.device import resolve, seed_everything
    from hsn.experiments.probe import edit_probe, edit_probe_reference, summarize
    from hsn.experiments.spec import protocol_params
    from hsn.experiments.variants import install_variant
    from hsn.runner import build_model, load_datasets
    from hsn.train.checkpoint import latest_checkpoint, load_checkpoint

    exp = db.experiment(exp_id)
    arm = db.arm(exp["arm_id"])
    cfg = RunConfig.model_validate(loads(arm["config_json"]))
    params = protocol_params(loads(arm["protocol_json"]))
    if params.name != "reads_state":
        raise ValueError(f"experiment {exp_id} ran {params.name}, not reads_state")
    if not exp["run_dir"]:
        raise ValueError(f"experiment {exp_id} has no run directory")
    checkpoint = latest_checkpoint(Path(exp["run_dir"]))
    if checkpoint is None:
        raise FileNotFoundError(f"no checkpoint in {exp['run_dir']}")
    ctx = resolve(cfg.hardware)
    _train, dev = load_datasets(cfg)
    n_dev = getattr(params, "n_dev", 0)
    if n_dev:
        dev = ByteDataset(list(dev.examples[:n_dev]))
    seed_everything(cfg.seed)
    model = build_model(cfg)
    install_variant(model, arm["variant"], cfg.seed, loads(arm["variant_json"]))
    model.to(device=ctx.device, dtype=ctx.master_dtype)
    load_checkpoint(checkpoint, model)
    t0 = time.perf_counter()
    batched = edit_probe(model, ctx, dev, batch_rows=batch_rows, margins=True)
    batched_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    reference = edit_probe_reference(model, ctx, dev, margins=True)
    reference_s = time.perf_counter() - t1
    flips = []
    for a, b in zip(reference, batched, strict=True):
        for which in ("original", "modified"):
            if a[f"argmax_{which}"] != b[f"argmax_{which}"]:
                flips.append(
                    {
                        "doc": a["doc"],
                        "which": which,
                        "reference": a[f"argmax_{which}"],
                        "batched": b[f"argmax_{which}"],
                        "margin_reference": a[f"margin_{which}"],
                        "margin_batched": b[f"margin_{which}"],
                    }
                )
    margins = [min(r["margin_original"], r["margin_modified"]) for r in reference]
    ref_summary, bat_summary = summarize(reference), summarize(batched)
    return {
        "exp_id": exp_id,
        "label": exp["label"],
        "checkpoint": str(checkpoint),
        "device": str(ctx.device),
        "autocast": str(ctx.autocast_dtype),
        "n_probed": len(reference),
        "batch_rows": batch_rows,
        "batched_s": round(batched_s, 2),
        "reference_s": round(reference_s, 2),
        "n_flips": len(flips),
        "max_flip_margin": max(
            (max(f["margin_reference"], f["margin_batched"]) for f in flips), default=None
        ),
        "min_margin_reference": min(margins) if margins else None,
        "verdict_reference": ref_summary["verdict"],
        "verdict_batched": bat_summary["verdict"],
        "correct_before_edit_reference": ref_summary["correct_before_edit"],
        "correct_before_edit_batched": bat_summary["correct_before_edit"],
        "flips": flips,
    }
