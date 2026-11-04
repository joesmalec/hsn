#!/usr/bin/env python
"""Gate G0 (plan Phase 0 header; chunk P0.8): build the data, check leakage, train the baselines
over seeds, profile, rerun, decide and package.

    python scripts/gate_g0.py --profile cpu --seeds 0,1,2,3,4 --out evidence/G0_cpu

Gate statement (plan line 779): data protocol, baselines and profiler exist; a reproducible rerun
succeeds; leakage checks pass; primary metrics, quality tolerances and the G1 task set are frozen in
``configs/metrics_freeze.json``. Steps, in order:

1. Data protocol: build ``synthetic_v1`` into ``--data-out`` if its manifest is absent (the build
   runs the P0.4 leakage checks on every train/dev/test pair and compares with the golden manifest),
   otherwise regenerate and verify every shard against the manifest.
2. Leakage: every pair recorded in the manifest must be zero, and the train and dev shards of every
   task the configurations use are checked again live (``check_leakage``). ``<out>/leakage.json``.
3. Frozen metrics: ``configs/metrics_freeze.json`` must load (hash and schema) before any run.
4. Baselines: ``hsn run`` for every configuration (GRU and Transformer by default) over the seeds; a
   P0.6 profiling pass is part of every run.
5. Rerun: the first seed of every configuration is retrained from its resolved configuration;
   exact on ``profile:cpu`` (every metric and the model hash), within ``--rerun-tolerance`` on the
   primary metric (accuracy) on any other profile, where the tolerance is required.
6. Decision (PASS only if every check holds), written to ``<out>/decision.json`` and the result
   store, then the evidence package ``<out>/G0.zip``. Exit code 0 on PASS, 1 on FAIL.

Nothing here touches the test split (P0.8 "Do not"); every evaluation is on dev.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hsn.config import REPO_ROOT, load_config
from hsn.data.dataset import ByteDataset
from hsn.data.leakage import check_leakage
from hsn.data.splits import MANIFEST_NAME, verify_splits
from hsn.data.synthetic import load_dataset_spec
from hsn.data.synthetic import main as synthetic_main
from hsn.eval.results import (
    METRICS_FILE,
    STORE_FILE,
    ResultStore,
    evidence_package,
    load_metrics_freeze,
)
from hsn.runner import parse_seeds, rerun, run_seeds

DEFAULT_CONFIGS = (
    "configs/baselines/gru_2m.yaml",
    "configs/baselines/transformer_2m.yaml",
)
DEFAULT_SPEC = "configs/data/synthetic_v1.yaml"
DEFAULT_GOLDEN = "configs/data/synthetic_v1.manifest.json"
DEFAULT_DATA_OUT = "data/synthetic_v1"

logger = logging.getLogger("hsn.gate_g0")


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def ensure_data(spec: Path, data_out: Path, golden: Path | None) -> dict[str, Any]:
    """Build or verify the dataset; returns what was done and the manifest's own leakage record."""
    manifest = data_out / MANIFEST_NAME
    started = time.perf_counter()
    if manifest.is_file():
        dataset = load_dataset_spec(spec)
        problems = verify_splits(dataset, data_out)
        if problems:
            raise RuntimeError(f"{data_out} does not verify against {manifest}: {problems[:5]}")
        if golden is not None:
            if manifest.read_text(encoding="utf-8") != golden.read_text(encoding="utf-8"):
                raise RuntimeError(f"{manifest} differs from the golden manifest {golden}")
        action = "verified"
    else:
        argv = ["build", "--spec", str(spec), "--out", str(data_out)]
        if golden is not None:
            argv += ["--golden", str(golden)]
        code = synthetic_main(argv)
        if code != 0:
            raise RuntimeError(f"data build failed with exit code {code}")
        action = "built"
    loaded = json.loads(manifest.read_text(encoding="utf-8"))
    pairs = loaded.get("leakage", [])
    counts = [int(p["exact_duplicates"]) + int(p["overlapping_ngrams"]) for p in pairs]
    return {
        "action": action,
        "seconds": time.perf_counter() - started,
        "manifest": str(manifest),
        "name": loaded.get("name"),
        "spec_hash": loaded.get("spec_hash"),
        "split_hashes": loaded.get("split_hashes"),
        "golden": str(golden) if golden is not None else None,
        "manifest_pairs": len(pairs),
        "manifest_worst_count": max(counts) if counts else 0,
        "manifest_clean": all(c == 0 for c in counts),
    }


def live_leakage(
    manifest: Path, configs: Sequence[Path], overrides: Sequence[str]
) -> dict[str, Any]:
    """``check_leakage`` of train against dev for every task/variant the configurations use."""
    pairs: dict[str, Any] = {}
    seen: set[tuple[str, str]] = set()
    for config in configs:
        cfg = load_config(config, overrides=[*overrides])
        for task in cfg.data.tasks:
            variants = list(cfg.data.variants) or ["in_range", "out_of_range"]
            for variant in variants:
                if (task, variant) in seen:
                    continue
                seen.add((task, variant))
                try:
                    train = ByteDataset.load(manifest, "train", [task], [variant])
                    dev = ByteDataset.load(manifest, "dev", [task], [variant])
                except KeyError:
                    continue  # the manifest has no such variant of this task
                report = check_leakage(train.examples, dev.examples)
                pairs[f"{task}/{variant}"] = {
                    "train": f"{task}/{variant}/train",
                    "test": f"{task}/{variant}/dev",
                    **report.to_dict(),
                }
    return {"pairs": pairs, "clean": all(bool(p["is_clean"]) for p in pairs.values())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", required=True, help="hardware profile, e.g. cpu or m_series")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out", required=True, help="evidence directory, e.g. evidence/G0_cpu")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--data-spec", default=DEFAULT_SPEC)
    parser.add_argument("--data-out", default=DEFAULT_DATA_OUT)
    parser.add_argument("--golden", default=DEFAULT_GOLDEN, help="golden manifest to compare with")
    parser.add_argument("--no-golden", action="store_true", help="skip the golden comparison")
    parser.add_argument("--runs", help="parent of the run directories (default runs/g0_<profile>)")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--rerun-tolerance",
        type=float,
        help="metric tolerance for the rerun; 0 (exact) on cpu, required on other profiles",
    )
    parser.add_argument("--power", choices=("none", "profile"), default="none")
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument(
        "--freeze", default=None, help="freeze file (default configs/metrics_freeze.json)"
    )
    args = parser.parse_args(argv)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs) if args.runs else Path("runs") / f"g0_{args.profile}"
    seeds = parse_seeds(args.seeds)
    configs = [Path(c) for c in args.configs]
    if args.profile == "cpu":
        tolerance = 0.0 if args.rerun_tolerance is None else args.rerun_tolerance
    elif args.rerun_tolerance is None:
        parser.error(f"--rerun-tolerance is required on profile {args.profile} (not bit-exact)")
    else:
        tolerance = args.rerun_tolerance
    overrides = [f"profile={args.profile}", *args.override]
    checks: dict[str, bool] = {}
    timings: dict[str, float] = {}

    # 1. data protocol
    data = ensure_data(
        Path(args.data_spec), Path(args.data_out), None if args.no_golden else Path(args.golden)
    )
    timings["data"] = data["seconds"]
    checks["data_protocol"] = True
    logger.info("data %s in %.1f s (%s)", data["action"], data["seconds"], data["manifest"])

    # 2. leakage
    t0 = time.perf_counter()
    live = live_leakage(Path(data["manifest"]), configs, overrides)
    timings["leakage"] = time.perf_counter() - t0
    leakage: dict[str, Any] = {
        "manifest": data,
        "live": live,
        "clean": bool(data["manifest_clean"] and live["clean"]),
    }
    _write_json(out / "leakage.json", leakage)
    checks["leakage_checks"] = leakage["clean"]
    logger.info(
        "leakage: manifest %d pairs (worst %d), live %d pairs, clean %s",
        data["manifest_pairs"],
        data["manifest_worst_count"],
        len(live["pairs"]),
        leakage["clean"],
    )

    # 3. frozen metrics
    freeze_path = (
        Path(args.freeze) if args.freeze else REPO_ROOT / "configs" / "metrics_freeze.json"
    )
    frozen = load_metrics_freeze(freeze_path)
    checks["metrics_frozen"] = True
    logger.info(
        "freeze %s valid: frozen_at %s git %s", freeze_path, frozen.frozen_at, frozen.git_rev
    )

    # 4. baselines over seeds (each run carries its profiling pass)
    run_dirs: list[Path] = []
    first_dirs: list[Path] = []
    for config in configs:
        t0 = time.perf_counter()
        dirs = run_seeds(
            config,
            seeds,
            out=runs_dir,
            profile=args.profile,
            overrides=args.override,
            profile_steps=args.profile_steps,
            power=args.power,
        )
        timings[f"train:{config.stem}"] = time.perf_counter() - t0
        run_dirs += dirs
        first_dirs.append(dirs[0])
    checks["baselines_exist"] = True
    checks["profiler_exists"] = all(
        (d / "profile_train.json").is_file() and (d / "profile_serve.json").is_file()
        for d in run_dirs
    )

    # 5. reruns
    rerun_dirs: list[Path] = []
    verdicts: list[dict[str, Any]] = []
    for original in first_dirs:
        t0 = time.perf_counter()
        new_dir = rerun(
            original,
            tolerance=tolerance,
            out=runs_dir,
            profile_steps=args.profile_steps,
            power=args.power,
        )
        timings[f"rerun:{original.name}"] = time.perf_counter() - t0
        rerun_dirs.append(new_dir)
        verdicts.append(json.loads((new_dir / "rerun.json").read_text(encoding="utf-8")))
    checks["rerun_reproduces"] = all(bool(v["passed"]) for v in verdicts)

    # 6. decision, then the package
    passed = all(checks.values())
    decision = {
        "gate": "G0",
        "decision": "PASS" if passed else "FAIL",
        "statement": (
            "data protocol, baselines and profiler exist; a reproducible rerun succeeds; leakage "
            "checks pass; primary metrics, quality tolerances and the G1 task set are frozen in "
            "configs/metrics_freeze.json (plan Phase 0 header)."
        ),
        "checks": checks,
        "profile": args.profile,
        "seeds": seeds,
        "configs": [str(c) for c in configs],
        "runs": [str(d) for d in run_dirs],
        "reruns": [
            {
                "original": v["original_run_id"],
                "rerun": v["rerun_run_id"],
                "tolerance": v["tolerance"],
                "max_abs_diff": v["max_abs_diff"],
                "max_abs_diff_loss": v["max_abs_diff_loss"],
                "model_hash_equal": v["model_hash_equal"],
                "passed": v["passed"],
            }
            for v in verdicts
        ],
        "freeze": {
            "path": str(freeze_path),
            "frozen_at": frozen.frozen_at,
            "git_rev": frozen.git_rev,
            "sha256_of_body": frozen.sha256_of_body,
            "g1_tasks": frozen.g1_task_keys(),
        },
        "data": data,
        "leakage_clean": leakage["clean"],
        "timings_s": timings,
        "decided_at": _utc_now(),
    }
    _write_json(out / "decision.json", decision)
    ResultStore(runs_dir / STORE_FILE).append(
        "G0",
        {
            "profile": args.profile,
            "metric": "decision",
            "value": 1.0 if passed else 0.0,
            "text": decision["decision"],
            "git_rev": frozen.git_rev,
        },
    )
    package = evidence_package(
        "G0",
        [*run_dirs, *rerun_dirs],
        out / "G0.zip",
        freeze=freeze_path,
        extra_files={"decision.json": out / "decision.json", "leakage.json": out / "leakage.json"},
    )
    timings["total"] = time.perf_counter() - started
    decision["timings_s"] = timings
    decision["package"] = str(package)
    _write_json(out / "decision.json", decision)

    print(f"\nG0 {decision['decision']} ({args.profile}); package {package}")
    print("checks: " + ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items()))
    print("\nrun_id | model | seed | steps | dev loss | dev acc | per task | tokens/s | elapsed s")
    for run_dir in [*run_dirs, *rerun_dirs]:
        m = json.loads((run_dir / METRICS_FILE).read_text(encoding="utf-8"))
        per_task = " ".join(
            f"{g.split('/')[0]}={e['accuracy']:.3f}"
            for g, e in sorted(m["final"]["per_group"].items())
        )
        print(
            f"{m['run_id']} | {m['model_name']} | {m['seed']} | {m['steps']} | "
            f"{m['final']['loss']:.4f} | {m['final']['accuracy']:.4f} | {per_task} | "
            f"{m['measured']['tokens_per_s']:.0f} | {m['elapsed_s']:.0f}"
        )
    for v in verdicts:
        print(
            f"rerun {v['original_run_id']}: max |diff| accuracy {v['max_abs_diff']:.3g} "
            f"loss {v['max_abs_diff_loss']:.3g} (tolerance {v['tolerance']:g} on accuracy), "
            f"model hash equal {v['model_hash_equal']} -> "
            f"{'PASS' if v['passed'] else 'FAIL'}"
        )
    print("timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items()))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
