#!/usr/bin/env python
"""Attribute a run's GPU-utilisation samples to what the run was doing.

    python scripts/gpu_phase_report.py --exp 346 [--exp 347 ...] [--db PATH] [--low 50] [--json]

Reads (read-only) the worker's card time series of each experiment (``worker --telemetry-max-samples
N``; the ``training_gpu_samples`` view) and, when the run recorded them, its ``phases``: the
wall-clock intervals of every evaluation, diagnostic sample and checkpoint write and of the run's
stages (:func:`hsn.experiments.protocols._phases`). Each sample is given the phase whose interval
holds it, in this order: ``setup`` (model build and the first compiles, before the fit),
``diagnostics``, ``evaluate``, ``checkpoint``, ``probing`` (after training) and ``train`` (the
rest of the fit). A sample covers about one second, so a pause shorter than the sampling period is
attributed to whatever phase holds the sample's instant; ``margin`` widens the non-training
intervals by that much on each side.

A run without ``phases`` (recorded before 10 Nov 2026) is split by step instead: samples within
``--near`` steps of a multiple of the diagnostic cadence count as ``diagnostics``, of the
evaluation cadence as ``evaluate``; the worker's reported step is coarse, so that split is an
estimate and is labelled as one.

Device utilisation counts every process on the card: run an arm alone (a timing arm) for a clean
trace.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any

PHASES = ("setup", "diagnostics", "evaluate", "checkpoint", "probing", "train")


def _load(conn: sqlite3.Connection, exp: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = conn.execute(
        "SELECT e.card_json, e.protocol_json, r.record_json FROM experiments e "
        "LEFT JOIN results r USING(exp_id) WHERE e.exp_id = ?",
        (exp,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"experiment {exp} is not in the database")
    card = json.loads(row[0] or "{}")
    record = json.loads(row[2] or "{}")
    record["_protocol"] = json.loads(row[1] or "{}")
    samples = [s for s in card.get("telemetry", []) if s.get("utilization_pct") is not None]
    return samples, record


def _intervals(phases: dict[str, Any], margin: float) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {name: [] for name in PHASES}
    stages = phases.get("stages", {})
    for name in ("setup", "probing"):
        span = stages.get(name)
        if span and len(span) == 2:
            out[name].append((span[0], span[1]))
    for start, seconds in phases.get("evaluate", []):
        out["evaluate"].append((start - margin, start + seconds + margin))
    for _step, start, seconds in phases.get("diagnostics", []):
        out["diagnostics"].append((start - margin, start + seconds + margin))
    for start, seconds in phases.get("checkpoint", []):
        out["checkpoint"].append((start - margin, start + seconds + margin))
    return out


def _by_step(samples: list[dict[str, Any]], record: dict[str, Any], near: float) -> list[str]:
    """The estimate for a run without ``phases``: the step at each sample, interpolated between
    the worker's reported steps, against the cadences of the protocol."""
    protocol = record.get("_protocol", {})
    diag_every = int(protocol.get("diag_every") or 0)
    evals = [int(e["step"]) for e in record.get("evals", [])]
    eval_every = evals[1] - evals[0] if len(evals) > 1 else 0
    first: dict[int, float] = {}
    for s in samples:
        step = s.get("step")
        if step is not None and step not in first:
            first[int(step)] = float(s["time"])
    steps = sorted(first)
    times = [first[k] for k in steps]
    labels = []
    for s in samples:
        if "probing" in str(s.get("phase", "")):
            labels.append("probing")
            continue
        i = bisect.bisect_right(times, float(s["time"])) - 1
        if i < 0 or not steps or steps[i] == 0:
            labels.append("setup")
            continue
        step = float(steps[i])
        if i + 1 < len(steps):
            frac = (float(s["time"]) - times[i]) / max(times[i + 1] - times[i], 1e-9)
            step += (steps[i + 1] - steps[i]) * frac

        def close(every: int, step: float = step) -> bool:
            return every > 0 and min(step % every, every - step % every) <= near

        if close(eval_every):
            labels.append("evaluate")
        elif close(diag_every):
            labels.append("diagnostics")
        else:
            labels.append("train")
    return labels


def report(conn: sqlite3.Connection, exp: int, low: float, margin: float, near: float) -> dict:
    samples, record = _load(conn, exp)
    if not samples:
        return {"exp": exp, "error": "no card time series (worker --telemetry-max-samples)"}
    phases = record.get("phases")
    if phases:
        spans = _intervals(phases, margin)
        labels = []
        for s in samples:
            t = float(s["time"])
            label = "train"
            for name in PHASES[:-1]:
                if any(a <= t <= b for a, b in spans[name]):
                    label = name
                    break
            fit = phases.get("stages", {}).get("fit")
            if label == "train" and fit and not (fit[0] <= t <= (fit[1] if len(fit) > 1 else t)):
                label = "setup" if t < fit[0] else "probing"
            labels.append(label)
        source = "recorded phases"
    else:
        labels = _by_step(samples, record, near)
        source = "estimated from the reported step (no recorded phases)"
    out: dict[str, Any] = {"exp": exp, "source": source, "samples": len(samples), "phases": {}}
    for name in PHASES:
        values = [
            float(s["utilization_pct"])
            for s, lab in zip(samples, labels, strict=True)
            if lab == name
        ]
        if not values:
            continue
        out["phases"][name] = {
            "samples": len(values),
            "share": len(values) / len(samples),
            "mean_util": statistics.fmean(values),
            "median_util": statistics.median(values),
            "below_low": sum(v < low for v in values),
            "below_low_share": sum(v < low for v in values) / len(values),
        }
    fit = [
        float(s["utilization_pct"])
        for s, lab in zip(samples, labels, strict=True)
        if lab in ("train", "diagnostics", "evaluate", "checkpoint")
    ]
    if fit:
        out["fit"] = {
            "samples": len(fit),
            "mean_util": statistics.fmean(fit),
            "below_low": sum(v < low for v in fit),
            "below_low_share": sum(v < low for v in fit) / len(fit),
        }
    walls = record.get("walls") or {}
    out["walls"] = {k: round(float(v), 1) for k, v in walls.items() if isinstance(v, int | float)}
    out["median_step_s"] = (record.get("step_time") or {}).get("run_median_s")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--exp", type=int, action="append", required=True)
    parser.add_argument(
        "--db", type=Path, default=Path.home() / "hsn-experiments/experiments.sqlite"
    )
    parser.add_argument("--low", type=float, default=50.0, help="a low sample is below this %%")
    parser.add_argument("--margin", type=float, default=0.5, help="seconds around an interval")
    parser.add_argument("--near", type=float, default=30.0, help="steps (runs without phases)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    conn = sqlite3.connect(f"file:{args.db.expanduser()}?mode=ro", uri=True)
    reports = [report(conn, exp, args.low, args.margin, args.near) for exp in args.exp]
    if args.json:
        print(json.dumps(reports, indent=1))
        return
    for r in reports:
        if "error" in r:
            print(f"exp {r['exp']}: {r['error']}")
            continue
        print(
            f"exp {r['exp']}: {r['samples']} samples; phases {r['source']}; "
            f"median step {r['median_step_s']}"
        )
        print(
            f"  {'phase':12s} {'samples':>7s} {'share':>6s} {'mean%':>6s} {'median%':>8s} "
            f"{'<' + format(args.low, '.0f') + '%':>6s} {'share':>6s}"
        )
        for name, p in r["phases"].items():
            print(
                f"  {name:12s} {p['samples']:7d} {100 * p['share']:5.1f}% {p['mean_util']:6.1f} "
                f"{p['median_util']:8.1f} {p['below_low']:6d} {100 * p['below_low_share']:5.1f}%"
            )
        if "fit" in r:
            f = r["fit"]
            print(
                f"  during the fit: mean {f['mean_util']:.1f}%, {f['below_low']} of {f['samples']} "
                f"samples ({100 * f['below_low_share']:.1f}%) below {args.low:.0f}%"
            )
        print(f"  walls: {r['walls']}")


if __name__ == "__main__":
    main()
