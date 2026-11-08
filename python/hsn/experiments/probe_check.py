"""``probe-check`` (P1.11): the batched edit probe against the audit's per-document probe on a
trained checkpoint of one experiment. ``probe-rerun`` (P1.14, :func:`probe_rerun`): v1 again on the
checkpoint against the stored per-document probes and summary, then probe v2 on it.

The model is rebuilt from the arm's stored configuration (seeded as the run, the variant installed
the same way), the run's last checkpoint is loaded, and both probes run on the dev documents under
the run's autocast. Batching changes GEMM row counts, so in bf16 a probe whose top-2 logit margin is
of the order of the rounding can flip; every flip is reported with both margins (a tie, not a bug,
when the margin is that small), together with both verdicts and both wall times.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from hsn.experiments.db import Database, loads

__all__ = ["probe_check", "probe_rerun"]


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
    cfg = RunConfig.model_validate(loads(exp.get("config_json") or arm["config_json"]))
    params = protocol_params(loads(exp.get("protocol_json") or arm["protocol_json"]))
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


def _restored(
    db: Database, exp_id: int, run_dir: Path, *, cpu: bool
) -> tuple[Any, Any, Any, Any, Path, Any]:
    """The experiment's model at the last checkpoint in ``run_dir`` (rebuilt as the run, the variant
    installed the same way), on the run's hardware or (``cpu``) on ``profile:cpu`` in fp32."""
    from hsn.config import RunConfig, load_hardware_profile
    from hsn.data.dataset import ByteDataset
    from hsn.device import resolve, seed_everything
    from hsn.experiments.spec import protocol_params
    from hsn.experiments.variants import install_variant
    from hsn.runner import build_model, load_datasets
    from hsn.train.checkpoint import latest_checkpoint, load_checkpoint

    exp = db.experiment(exp_id)
    arm = db.arm(exp["arm_id"])
    cfg = RunConfig.model_validate(loads(exp.get("config_json") or arm["config_json"]))
    params = protocol_params(loads(exp.get("protocol_json") or arm["protocol_json"]))
    if params.name != "reads_state":
        raise ValueError(f"experiment {exp_id} ran {params.name}, not reads_state")
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    ctx = resolve(load_hardware_profile("cpu") if cpu else cfg.hardware)
    _train, dev = load_datasets(cfg)
    if params.n_dev:
        dev = ByteDataset(list(dev.examples[: params.n_dev]))
    seed_everything(cfg.seed)
    model = build_model(cfg)
    install_variant(model, arm["variant"], cfg.seed, loads(arm["variant_json"]))
    model.to(device=ctx.device, dtype=ctx.master_dtype)
    load_checkpoint(checkpoint, model)  # loads on the CPU, copies into the model on its device
    return exp, params, ctx, dev, checkpoint, model


def probe_rerun(
    db: Database, exp_id: int, *, cpu: bool = False, batch_rows: int | None = None
) -> dict[str, Any]:
    """P1.14: v1 (batched, the run's ``probe_batch`` unless ``batch_rows``) on the experiment's last
    checkpoint, compared document by document with the per-document probes stored in its
    ``result.json`` (``probes``, or ``at_stop.probes`` after an early stop) and with its stored
    summary; then probe v2 on the same model. Every answer that differs from the stored one is
    listed with both top-2 margins (a near-tie under other rounding, as ``probe-check``)."""
    from hsn.experiments.probe import edit_probe, edit_probe_v2, summarize, summarize_v2

    result = db.row("SELECT json_path FROM results WHERE exp_id = ?", (exp_id,))
    if result is None or not result["json_path"]:
        raise ValueError(f"experiment {exp_id} has no stored result")
    json_path = Path(result["json_path"])
    run_dir = Path(db.experiment(exp_id)["run_dir"] or json_path.parent)
    exp, params, ctx, dev, checkpoint, model = _restored(db, exp_id, run_dir, cpu=cpu)
    stored_full = json.loads(json_path.read_text(encoding="utf-8"))
    block = stored_full if stored_full.get("probes") else (stored_full.get("at_stop") or {})
    stored, stored_summary = block.get("probes") or [], block.get("probe") or {}
    rows = batch_rows or params.probe_batch
    t0 = time.perf_counter()
    v1 = edit_probe(model, ctx, dev, batch_rows=rows, margins=True)
    v1_s = time.perf_counter() - t0
    if [p["doc"] for p in v1] != [p["doc"] for p in stored]:
        raise ValueError(f"experiment {exp_id}: the stored probes are of other documents")
    differ = []
    for now, then in zip(v1, stored, strict=True):
        for which in ("original", "modified"):
            if now[f"argmax_{which}"] != then[f"argmax_{which}"]:
                differ.append(
                    {
                        "doc": now["doc"],
                        "which": which,
                        "stored": then[f"argmax_{which}"],
                        "rerun": now[f"argmax_{which}"],
                        "margin_rerun": now[f"margin_{which}"],
                    }
                )
    summary = summarize(v1)
    keys = ("n_probed", "first_query_accuracy", "all", "correct_before_edit", "verdict")
    t1 = time.perf_counter()
    v2 = edit_probe_v2(model, ctx, dev, v1=v1, batch_rows=rows)
    v2_s = time.perf_counter() - t1
    summary_v2 = summarize_v2(v2, v1=v1)
    return {
        "exp_id": exp_id,
        "label": exp["label"],
        "checkpoint": str(checkpoint),
        "device": str(ctx.device),
        "autocast": str(ctx.autocast_dtype),
        "batch_rows": rows,
        "v1_s": round(v1_s, 2),
        "v2_s": round(v2_s, 2),
        "v1_answers_differing": len(differ),
        "v1_summary_equal": all(summary[k] == stored_summary.get(k) for k in keys),
        "v1_rerun": {k: summary[k] for k in keys},
        "v1_stored": {k: stored_summary.get(k) for k in keys},
        "v2": summary_v2,
        "differ": differ,
    }
