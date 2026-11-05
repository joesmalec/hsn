#!/usr/bin/env python
"""Gate G1 (plan Phase 1 header; chunk P1.9): the dense predictive cell on the frozen G1 task set.

    python scripts/gate_g1.py --profile rtx3060 --seeds 0,1,2,3,4 --out evidence/G1/ \\
        --power profile --rerun-tolerance 0.01 --jobs 3

Gate statement (plan line 1196): a dense predictive cell of about 2M parameters reaches >= 95 % on
the designated in-range state tasks, with finite gradients throughout and correct reset behaviour.
The task set and the threshold come from ``configs/metrics_freeze.json`` (P0.8; never edited):
kv/in_range and delayed/in_range at 0.95 per task per seed over five seeds. G1 is an implementation
sanity check; the comparison with the Phase 0 baselines is reported without a superiority claim.

Stages, in order; every stage leaves a JSON file in ``--out`` and a stage whose file exists is
reused when the script is invoked again (a crash resumes where it stopped; delete a file to redo its
stage):

1. Data protocol and leakage (the P0.8 code of ``gate_g0.py``): the dataset is verified against the
   golden manifest; ``check_leakage`` runs live on train vs dev *and* train vs test of every G1
   task, because this gate touches the test split once. ``leakage.json``.
2. Frozen metrics: the freeze file must load (hash and schema) before any run, and the experiment
   configuration must name exactly the frozen task set (chunk "Do not": change the G1 task set).
3. Invariant suite (P1.8, Track T): ``pytest tests/invariants`` in this environment with a junit
   report, or ``--invariants-junit`` to read one produced earlier in the same environment; the gate
   records which. ``correct_reset`` of the freeze is evidenced by the named reset invariant.
   ``invariants.json`` / ``invariants.xml``.
4. Overfit check (paper §18, chunk step 1): 64 training documents (half per task) must reach
   ``--overfit-target`` train accuracy within ``--overfit-steps``; the script stops with exit code 2
   otherwise (chunk: stop and diagnose rather than search). ``overfit.json``.
5. Recipe search on dev (chunk step 3; paper §12 equal search budgets): ``train.search_trials``
   trials, each the configuration plus one ``--trial`` override set, seed ``--search-seed`` for
   ``--search-steps`` steps; the GRU and Transformer configurations get the same trials, the same
   horizon and the matched budget keys of the experiment configuration. ``search.json`` records
   every trial, including the failed ones.
6. Five seeds of the winning recipe per method, each run carrying the P0.6 profiling pass (``--power
   profile`` reads the NVML meter); a rerun of the first HSN seed.
7. Test, exactly once per run (chunk step 3; pitfall: tuning on test): the final checkpoint is
   evaluated on the test split through ``Trainer.evaluate(..., allow_test=True)``; a run that
   already holds ``test_eval.json`` is never evaluated again. ``test_results.json`` and
   ``split=test`` rows in the result store.
8. Idle-GPU profiling pass of the first seed of every method (energy under ``--jobs`` concurrency is
   the whole card's), ``profiler_g1_<model>_{train,serve}.json``.
9. Decision (PASS only if every gating check holds), ``decision.json``, the result-store row, and
   the evidence package ``G1.zip`` (``hsn package --gate G1``). Exit code 0 on PASS, 1 on FAIL, 2
   when the overfit check failed and the gate stopped there.

Gating checks: data protocol, leakage, frozen metrics, invariant suite passed, correct reset
evidenced, environment consistent (the invariant run's git revision equals every training run's),
overfit, ``n_seeds`` of the freeze, every logged gradient norm finite (global and per group, every
step of every HSN seed run), test evaluated exactly once per run, and the threshold on the test
split for every seed and every frozen task. Reported but not gating: the baselines (matched
parameters, same budget), the profiler and energy files, and the rerun agreement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import platform
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

import torch
from hsn.config import REPO_ROOT, RunConfig, config_hash, load_config
from hsn.data.dataset import ByteDataset
from hsn.data.leakage import check_leakage
from hsn.device import resolve, seed_everything
from hsn.eval.results import (
    HISTORY_FILE,
    METRICS_FILE,
    RERUN_FILE,
    RESOLVED_CONFIG_FILE,
    STORE_FILE,
    MetricsFreeze,
    ResultStore,
    evidence_package,
    load_metrics_freeze,
)
from hsn.runner import (
    PowerChoice,
    build_model,
    load_datasets,
    make_power_meter,
    make_trainer,
    parse_seeds,
    profile_run,
    rerun,
    run_seeds,
)
from hsn.train.checkpoint import latest_checkpoint, load_checkpoint
from hsn.train.loop import SearchBudget

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_g0 import DEFAULT_DATA_OUT, DEFAULT_GOLDEN, DEFAULT_SPEC, ensure_data  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/g1_ref_a1.yaml"
DEFAULT_BASELINES = (
    "configs/baselines/gru_2m.yaml",
    "configs/baselines/transformer_2m.yaml",
)
#: The default search grid: trial 0 is the configuration itself; each further trial is one override
#: set on top of it (steps, learning rate and schedule, batch — chunk P1.9 "search the recipe").
DEFAULT_TRIALS: tuple[tuple[str, ...], ...] = (
    ("train.lr=3e-3",),
    ("train.batch=32", "train.micro_batch=32"),
    ("train.schedule=constant", "train.warmup_steps=0"),
)
#: Keys of the experiment configuration every baseline inherits so that the budgets match
#: (paper §12: equal search budgets; matched training compute).
MATCHED_TRAIN_KEYS = (
    "steps",
    "batch",
    "micro_batch",
    "unroll",
    "eval_every",
    "ckpt_every",
    "stats_every",
    "search_trials",
)
#: Invariant test(s) that evidence the freeze's ``correct_reset`` (P1.8, Track T).
RESET_INVARIANTS = ("test_p1_8_reset_removes_prior_state",)
TEST_EVAL_FILE = "test_eval.json"
SEARCH_METRIC = "g1_mean_dev_accuracy"
EXIT_PASS, EXIT_FAIL, EXIT_OVERFIT = 0, 1, 2

logger = logging.getLogger("hsn.gate_g1")


# --------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, check=False, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_state() -> tuple[str | None, bool]:
    """``(revision, dirty)`` of the repository the script runs from."""
    if shutil.which("git") is None:
        return None, False
    rev = _command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    status = _command(["git", "status", "--porcelain"], cwd=REPO_ROOT)
    return rev, bool(status)


def gpu_processes() -> list[dict[str, Any]] | None:
    """Every process on the GPU as ``nvidia-smi`` lists it (``None`` without the tool): the report
    says whether the card was shared while the measurements were taken."""
    if shutil.which("nvidia-smi") is None:
        return None
    out = _command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if out is None:
        return None
    rows: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            rows.append({"pid": int(parts[0]), "name": parts[1], "used_mib": parts[2]})
    return rows


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _tail(path: Path, n: int = 20) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


# --------------------------------------------------------------------------------------------------
# Stage 1-2: data, leakage, freeze
# --------------------------------------------------------------------------------------------------


def live_leakage_g1(manifest: Path, frozen: MetricsFreeze) -> dict[str, Any]:
    """``check_leakage`` of train vs dev and train vs test for every frozen G1 task (this gate reads
    the test split once, so the test pair is checked live as well)."""
    pairs: dict[str, Any] = {}
    for task in frozen.g1_tasks:
        train = ByteDataset.load(manifest, "train", [task.task], [task.variant])
        for split in ("dev", "test"):
            other = ByteDataset.load(manifest, split, [task.task], [task.variant])
            report = check_leakage(train.examples, other.examples)
            pairs[f"{task.key}:{split}"] = {
                "train": f"{task.key}/train",
                "test": f"{task.key}/{split}",
                **report.to_dict(),
            }
    return {"pairs": pairs, "clean": all(bool(p["is_clean"]) for p in pairs.values())}


def check_task_set(cfg: RunConfig, frozen: MetricsFreeze, where: str) -> None:
    """The configuration must name exactly the frozen G1 task set (chunk "Do not")."""
    tasks = sorted(cfg.data.tasks)
    variants = sorted(cfg.data.variants)
    want_tasks = sorted({t.task for t in frozen.g1_tasks})
    want_variants = sorted({t.variant for t in frozen.g1_tasks})
    if tasks != want_tasks or variants != want_variants:
        raise ValueError(
            f"{where}: data.tasks={cfg.data.tasks} data.variants={cfg.data.variants} is not the "
            f"frozen G1 task set {frozen.g1_task_keys()}; the task set is never changed here"
        )


# --------------------------------------------------------------------------------------------------
# Stage 3: the invariant suite
# --------------------------------------------------------------------------------------------------


def parse_junit(path: Path) -> dict[str, Any]:
    """Counts and per-testcase status of a pytest ``--junitxml`` report."""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    cases: dict[str, str] = {}
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.get(key, "0") or 0)
        for case in suite.iter("testcase"):
            name = f"{case.get('classname', '')}::{case.get('name', '')}"
            status = "passed"
            for child in case:
                if child.tag in ("failure", "error", "skipped"):
                    status = "failed" if child.tag == "failure" else child.tag
                    break
            cases[name] = status
    return {**counts, "cases": cases}


def invariants_verdict(parsed: Mapping[str, Any], returncode: int | None) -> dict[str, Any]:
    """``passed`` (the suite ran with at least one test and no failure or error; the pytest exit
    code must be 0 when known) and ``correct_reset`` (every reset invariant that ran passed)."""
    cases: Mapping[str, str] = parsed["cases"]
    passed = (
        int(parsed["tests"]) >= 1
        and int(parsed["failures"]) == 0
        and int(parsed["errors"]) == 0
        and (returncode is None or returncode == 0)
    )
    reset_cases = {
        name: status
        for name, status in cases.items()
        if any(name.split("::")[-1].startswith(test) for test in RESET_INVARIANTS)
    }
    correct_reset = bool(reset_cases) and all(s == "passed" for s in reset_cases.values())
    return {"passed": passed, "correct_reset": correct_reset, "reset_cases": reset_cases}


def run_invariant_suite(
    out: Path, junit_in: Path | None, junit_git_rev: str | None = None
) -> dict[str, Any]:
    """Run ``pytest tests/invariants`` with a junit report into ``out`` (mode ``ran``) or read the
    report ``junit_in`` (mode ``read``, with the git revision it was produced at given by the
    caller; without it the environment cannot be shown consistent); the record says which and in
    what environment."""
    out.mkdir(parents=True, exist_ok=True)
    junit = out / "invariants.xml"
    started = time.perf_counter()
    rev, dirty = git_state()
    record: dict[str, Any] = {
        "git_rev": rev,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "started_at": _utc_now(),
    }
    if junit_in is not None:
        if not junit_in.is_file():
            raise FileNotFoundError(f"--invariants-junit {junit_in} is not a file")
        if junit_in.resolve() != junit.resolve():
            shutil.copyfile(junit_in, junit)
        record.update(
            mode="read",
            junit=str(junit),
            source=str(junit_in),
            returncode=None,
            git_rev=junit_git_rev,
            git_dirty=None,
        )
        returncode: int | None = None
    else:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/invariants",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit}",
        ]
        proc = subprocess.run(
            command, cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=7200
        )
        (out / "invariants.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        returncode = proc.returncode
        record.update(mode="ran", command=command, junit=str(junit), returncode=returncode)
        if not junit.is_file():
            raise RuntimeError(
                f"pytest wrote no junit report (exit {returncode}); see {out / 'invariants.log'}"
            )
    parsed = parse_junit(junit)
    verdict = invariants_verdict(parsed, returncode)
    record.update(
        tests=parsed["tests"],
        failures=parsed["failures"],
        errors=parsed["errors"],
        skipped=parsed["skipped"],
        cases=parsed["cases"],
        seconds=time.perf_counter() - started,
        **verdict,
    )
    return record


# --------------------------------------------------------------------------------------------------
# Stage 4: the overfit check
# --------------------------------------------------------------------------------------------------


def select_overfit_docs(train: ByteDataset, frozen: MetricsFreeze, n_docs: int) -> ByteDataset:
    """The first ``n_docs // n_tasks`` training documents of every frozen task (manifest order)."""
    per_task = n_docs // len(frozen.g1_tasks)
    if per_task < 1:
        raise ValueError(f"--overfit-docs {n_docs} gives no document per task")
    picked = []
    for task in frozen.g1_tasks:
        found = [
            example
            for example in train.examples
            if example.meta.get("task") == task.task and example.meta.get("variant") == task.variant
        ][:per_task]
        if len(found) < per_task:
            raise ValueError(f"only {len(found)} training documents of {task.key}, need {per_task}")
        picked.extend(found)
    return ByteDataset(picked)


def overfit_check(
    config: Path,
    overrides: Sequence[str],
    frozen: MetricsFreeze,
    *,
    n_docs: int,
    steps: int,
    eval_every: int,
    target: float,
    run_dir: Path,
) -> dict[str, Any]:
    """Train the configuration on ``n_docs`` training documents and measure the accuracy over the
    answer positions of those same documents after every ``eval_every`` steps; stop early at
    ``target``. The data order restarts at every evaluation chunk (the loader is rebuilt), which
    does not matter for a fixed set of documents cycled many times."""
    if steps < 1 or eval_every < 1:
        raise ValueError("--overfit-steps and --overfit-eval-every must be >= 1")
    cfg = load_config(
        config,
        overrides=[
            *overrides,
            f"train.steps={steps}",
            f"train.eval_every={eval_every}",
            f"train.ckpt_every={steps}",
            "run_id=g1_overfit",
        ],
    )
    ctx = resolve(cfg.hardware)
    train_all, _dev = load_datasets(cfg)
    docs = select_overfit_docs(train_all, frozen, n_docs)
    n_loss = sum(int(example.loss_mask.sum()) for example in docs.examples)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    trainer = make_trainer(model, cfg, ctx, run_dir=run_dir)
    started = time.perf_counter()
    evals: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    tokens = 0.0
    train_seconds = 0.0
    reached: int | None = None
    horizon = 0
    while horizon < steps:
        horizon = min(horizon + eval_every, steps)
        history = trainer.fit(docs, docs, horizon)
        for record in history.steps:
            step_records.append(
                {
                    "step": record.step,
                    "loss": record.loss,
                    "accuracy": record.accuracy,
                    "grad_norm": record.grad_norm,
                    "lr": record.lr,
                    "group_norms": {
                        k: v
                        for k, v in getattr(record, "stats", {}).items()
                        if k.startswith("grad_norm/")
                    },
                }
            )
        tokens += float(history.measured.get("tokens", 0.0))
        train_seconds += float(history.measured.get("train_seconds", 0.0))
        for entry in history.evals:
            evals.append(
                {
                    "step": entry["step"],
                    "accuracy": float(entry["accuracy"]),
                    "loss": float(entry["loss"]),
                    "per_group": {
                        g: {"accuracy": float(e["accuracy"]), "loss": float(e["loss"])}
                        for g, e in entry["per_group"].items()
                    },
                }
            )
            if reached is None and float(entry["accuracy"]) >= target:
                reached = int(entry["step"])
        logger.info(
            "overfit step %d/%d: train-set accuracy %.4f loss %.4f",
            horizon,
            steps,
            evals[-1]["accuracy"],
            evals[-1]["loss"],
        )
        if reached is not None:
            break
    norms_finite = all(_is_finite_number(r["grad_norm"]) for r in step_records) and all(
        _is_finite_number(v) for r in step_records for v in r["group_norms"].values()
    )
    final = evals[-1]
    passed = reached is not None
    return {
        "n_docs": len(docs),
        "docs_per_task": {t.key: len(docs) // len(frozen.g1_tasks) for t in frozen.g1_tasks},
        "doc_ids": [str(example.meta.get("doc_id")) for example in docs.examples],
        "loss_positions": n_loss,
        "config_hash": config_hash(cfg),
        "config": str(config),
        "overrides": list(overrides),
        "profile": cfg.profile,
        "target": target,
        "steps_budget": steps,
        "eval_every": eval_every,
        "steps_run": step_records[-1]["step"] if step_records else 0,
        "reached_target_at_step": reached,
        "final_train_accuracy": final["accuracy"],
        "final_train_loss": final["loss"],
        "final_per_group": final["per_group"],
        "evals": evals,
        "steps": step_records,
        "grad_norms_finite": norms_finite,
        "tokens": tokens,
        "train_seconds": train_seconds,
        "seconds": time.perf_counter() - started,
        "passed": passed,
        "note": "accuracy over the answer positions of the same documents the model trains on; "
        "the data order restarts at every evaluation chunk",
    }


# --------------------------------------------------------------------------------------------------
# Running units (search trials, seeds) in process or as concurrent subprocesses
# --------------------------------------------------------------------------------------------------


class Unit(NamedTuple):
    """One invocation of the runner: ``hsn run`` of one seed (``kind="run"``: ``config`` plus
    ``overrides`` into ``out``) or ``hsn rerun`` of a finished run (``kind="rerun"``: ``source``
    is the original run directory and ``tolerance`` its tolerance), optionally only after the
    unit ``depends_on`` finished."""

    label: str
    model: str
    config: Path | None
    seed: int
    overrides: list[str]
    out: Path
    run_id: str
    profile: str
    kind: str = "run"
    source: Path | None = None
    tolerance: float = 0.0
    depends_on: str | None = None

    @property
    def run_dir(self) -> Path:
        if self.kind == "rerun":
            if self.source is None:
                raise ValueError(f"rerun unit {self.label!r} has no source run directory")
            return self.out / f"{self.source.name}-rerun"
        return self.out / f"{self.run_id}-{self.profile}-s{self.seed}"

    @property
    def done_marker(self) -> str:
        """The file whose presence means the unit finished (the runner writes it last)."""
        return RERUN_FILE if self.kind == "rerun" else METRICS_FILE

    def argv(self, *, profile_steps: int, power: str) -> list[str]:
        if self.kind == "rerun":
            return [
                sys.executable,
                "-m",
                "hsn.runner",
                "rerun",
                str(self.source),
                "--tolerance",
                str(self.tolerance),
                "--out",
                str(self.out),
                "--profile-steps",
                str(profile_steps),
                "--power",
                power,
            ]
        argv = [
            sys.executable,
            "-m",
            "hsn.runner",
            "run",
            "--config",
            str(self.config),
            "--seeds",
            str(self.seed),
            "--profile",
            self.profile,
            "--out",
            str(self.out),
            "--profile-steps",
            str(profile_steps),
            "--power",
            power,
        ]
        for item in [*self.overrides, f"run_id={self.run_id}"]:
            argv += ["--override", item]
        return argv


class UnitResult(NamedTuple):
    unit: Unit
    status: str  # "ok" | "reused" | "failed"
    run_dir: Path | None
    error: str | None = None
    seconds: float = 0.0
    gpu_processes: list[dict[str, Any]] | None = None


class Ledger:
    """Append-only record of every configuration that failed (Track E)."""

    def __init__(self, path: Path, entries: Sequence[Mapping[str, Any]] = ()) -> None:
        self.path = path
        self.entries: list[dict[str, Any]] = [dict(e) for e in entries]

    def add(self, entry: Mapping[str, Any]) -> None:
        record = {**entry, "recorded_at": _utc_now()}
        self.entries.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> Ledger:
        entries: list[dict[str, Any]] = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entries.append(json.loads(line))
        return cls(path, entries)


def _quarantine_crash(run_dir: Path, marker: str = METRICS_FILE) -> str | None:
    """A run directory without its completion ``marker`` is a crashed run: moved aside, its log
    tail kept."""
    if run_dir.is_dir() and not (run_dir / marker).is_file():
        tail = _tail(run_dir / "run.log")
        target = run_dir.with_name(f"{run_dir.name}.crashed-{_stamp()}")
        shutil.move(str(run_dir), str(target))
        return tail
    return None


def run_units(
    units: Sequence[Unit],
    *,
    jobs: int,
    profile_steps: int,
    power: str,
    logs_dir: Path,
    ledger: Ledger,
    poll_s: float = 5.0,
) -> list[UnitResult]:
    """Run every unit (reusing a finished run directory), ``jobs`` at a time as subprocesses when
    ``jobs > 1``, in this process otherwise; a unit with ``depends_on`` starts only after that
    unit finished. A unit that fails is recorded in the ledger and does not stop the others."""
    if jobs < 1:
        raise ValueError(f"--jobs must be >= 1, got {jobs}")
    results: dict[str, UnitResult] = {}
    pending: list[Unit] = []
    labels = {unit.label for unit in units}
    for unit in units:
        if unit.label in results or sum(1 for u in units if u.label == unit.label) > 1:
            raise ValueError(f"unit label {unit.label!r} repeats")
        if unit.depends_on is not None and unit.depends_on not in labels:
            raise ValueError(f"unit {unit.label!r} depends on unknown unit {unit.depends_on!r}")
        if (unit.run_dir / unit.done_marker).is_file():
            logger.info("reusing %s (%s)", unit.label, unit.run_dir)
            results[unit.label] = UnitResult(unit, "reused", unit.run_dir)
            continue
        tail = _quarantine_crash(unit.run_dir, unit.done_marker)
        if tail is not None:
            ledger.add(
                {
                    "kind": "crashed_run_dir",
                    "model": unit.model,
                    "label": unit.label,
                    "run_id": unit.run_dir.name,
                    "overrides": unit.overrides,
                    "status": f"crashed before {unit.done_marker}; directory moved aside",
                    "error": tail,
                }
            )
        pending.append(unit)

    def failed(unit: Unit, error: str, seconds: float) -> UnitResult:
        tail = _quarantine_crash(unit.run_dir, unit.done_marker)
        text = error if tail is None else f"{error}\n--- run.log tail ---\n{tail}"
        ledger.add(
            {
                "kind": "failed_run",
                "model": unit.model,
                "label": unit.label,
                "run_id": unit.run_dir.name,
                "overrides": unit.overrides,
                "status": "failed",
                "error": text[-4000:],
            }
        )
        logger.error("%s failed: %s", unit.label, error.splitlines()[-1] if error else "?")
        return UnitResult(unit, "failed", None, text[-4000:], seconds)

    def dependency_state(unit: Unit) -> str:
        """``ready`` (no dependency or it finished), ``waiting`` or ``failed``."""
        if unit.depends_on is None:
            return "ready"
        dep = results.get(unit.depends_on)
        if dep is None:
            return "waiting"
        return "ready" if dep.run_dir is not None else "failed"

    if jobs == 1:
        for unit in pending:
            state = dependency_state(unit)
            if state != "ready":
                results[unit.label] = failed(unit, f"dependency {unit.depends_on} {state}", 0.0)
                continue
            started = time.perf_counter()
            try:
                if unit.kind == "rerun":
                    if unit.source is None:
                        raise ValueError(f"rerun unit {unit.label!r} has no source")
                    run_dir = rerun(
                        unit.source,
                        tolerance=unit.tolerance,
                        out=unit.out,
                        profile_steps=profile_steps,
                        power=cast(PowerChoice, power),
                    )
                else:
                    if unit.config is None:
                        raise ValueError(f"run unit {unit.label!r} has no configuration")
                    (run_dir,) = run_seeds(
                        unit.config,
                        [unit.seed],
                        out=unit.out,
                        profile=unit.profile,
                        overrides=[*unit.overrides, f"run_id={unit.run_id}"],
                        profile_steps=profile_steps,
                        power=cast(PowerChoice, power),
                    )
            except Exception as exc:  # noqa: BLE001 - every failure is a recorded finding
                text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                results[unit.label] = failed(unit, text, time.perf_counter() - started)
                continue
            results[unit.label] = UnitResult(
                unit, "ok", run_dir, None, time.perf_counter() - started, gpu_processes()
            )
        return [results[u.label] for u in units]

    logs_dir.mkdir(parents=True, exist_ok=True)
    queue = list(pending)
    running: list[tuple[Unit, subprocess.Popen[bytes], float, Any]] = []
    while queue or running:
        for unit in [u for u in queue if dependency_state(u) == "failed"]:
            queue.remove(unit)
            results[unit.label] = failed(unit, f"dependency {unit.depends_on} failed", 0.0)
        ready = [u for u in queue if dependency_state(u) == "ready"]
        while ready and len(running) < jobs:
            unit = ready.pop(0)
            queue.remove(unit)
            log = (logs_dir / f"{unit.run_dir.name}.log").open("ab")
            proc = subprocess.Popen(
                unit.argv(profile_steps=profile_steps, power=power),
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            logger.info("started %s (pid %d) -> %s", unit.label, proc.pid, unit.run_dir)
            running.append((unit, proc, time.perf_counter(), log))
        if not running:
            if queue:  # every remaining unit waits on a unit that is not in this batch
                for unit in list(queue):
                    queue.remove(unit)
                    results[unit.label] = failed(
                        unit, f"dependency {unit.depends_on} never ran", 0.0
                    )
            break
        time.sleep(poll_s)
        still: list[tuple[Unit, subprocess.Popen[bytes], float, Any]] = []
        for unit, proc, started, log in running:
            code = proc.poll()
            if code is None:
                still.append((unit, proc, started, log))
                continue
            seen = gpu_processes()
            log.close()
            seconds = time.perf_counter() - started
            # ``hsn rerun`` exits 1 when the rerun is outside the tolerance: that is a recorded
            # verdict (rerun.json), not a failed unit
            if (unit.run_dir / unit.done_marker).is_file() and (code == 0 or unit.kind == "rerun"):
                results[unit.label] = UnitResult(unit, "ok", unit.run_dir, None, seconds, seen)
                logger.info("%s done in %.0f s (exit %d)", unit.label, seconds, code)
            else:
                tail = _tail(logs_dir / f"{unit.run_dir.name}.log", 30)
                results[unit.label] = failed(unit, f"exit code {code}\n{tail}", seconds)
        running = still
    return [results[u.label] for u in units]


# --------------------------------------------------------------------------------------------------
# Stage 5: the search
# --------------------------------------------------------------------------------------------------


def trial_sets(items: Sequence[Sequence[str]] | None) -> list[list[str]]:
    """Trial 0 is the configuration itself; every ``--trial`` adds one override set."""
    extra = [list(t) for t in (items if items is not None else DEFAULT_TRIALS)]
    for trial in extra:
        if not trial:
            raise ValueError("--trial needs at least one KEY=VALUE override")
    return [[], *extra]


def matched_overrides(cfg: RunConfig) -> list[str]:
    """The budget and data keys every baseline takes from the experiment configuration."""
    items = [f"train.{key}={getattr(cfg.train, key)}" for key in MATCHED_TRAIN_KEYS]
    items.append("data.tasks=[" + ",".join(cfg.data.tasks) + "]")
    items.append("data.variants=[" + ",".join(cfg.data.variants) + "]")
    return items


def g1_result(metrics: Mapping[str, Any], frozen: MetricsFreeze) -> dict[str, float]:
    """The search result of one run from its ``metrics.json``: mean and minimum final dev accuracy
    over the frozen tasks, the overall dev loss, and whether every gradient norm was finite."""
    per_group = metrics["final"]["per_group"]
    accs = []
    for task in frozen.g1_tasks:
        if task.key not in per_group:
            raise KeyError(f"run {metrics.get('run_id')} did not evaluate {task.key}")
        accs.append(float(per_group[task.key]["accuracy"]))
    return {
        SEARCH_METRIC: sum(accs) / len(accs),
        "g1_min_dev_accuracy": min(accs),
        "dev_loss": float(metrics["final"]["loss"]),
        "grad_norms_finite": 1.0 if metrics.get("grad_norms_finite") else 0.0,
    }


def pick_best(trials: Sequence[Mapping[str, Any]]) -> int | None:
    """Index of the best trial: highest mean dev accuracy over the G1 tasks, then highest minimum,
    then lowest dev loss, then the lowest index; failed trials never win."""
    ranked = [
        (
            -float(t["result"][SEARCH_METRIC]),
            -float(t["result"]["g1_min_dev_accuracy"]),
            float(t["result"]["dev_loss"]),
            int(t["index"]),
        )
        for t in trials
        if t.get("status") in ("ok", "reused")
        and _is_finite_number(t.get("result", {}).get(SEARCH_METRIC))
    ]
    if not ranked:
        return None
    return min(ranked)[3]


def search_method(
    *,
    model: str,
    config: Path,
    base_overrides: Sequence[str],
    trials: Sequence[Sequence[str]],
    frozen: MetricsFreeze,
    search_steps: int,
    search_seed: int,
    runs_dir: Path,
    profile: str,
    jobs: int,
    profile_steps: int,
    power: str,
    logs_dir: Path,
    ledger: Ledger,
    poll_s: float = 5.0,
) -> dict[str, Any]:
    """Every trial of one method within its ``train.search_trials`` budget; returns the record."""
    base = load_config(config, overrides=list(base_overrides))
    budget = SearchBudget.from_config(base)
    if len(trials) != budget.trials_allowed:
        raise ValueError(
            f"{config}: train.search_trials={budget.trials_allowed} but {len(trials)} trials were "
            "given (trial 0 is the configuration itself); the budget is the same for every method"
        )
    units: list[Unit] = []
    planned: list[dict[str, Any]] = []
    for index, trial in enumerate(trials):
        overrides = [*base_overrides, *trial, f"train.steps={search_steps}"]
        entry: dict[str, Any] = {
            "index": index,
            "model": model,
            "overrides": list(trial),
            "all_overrides": overrides,
            "status": "planned",
        }
        try:
            cfg = load_config(config, overrides=overrides)
            entry["config_hash"] = config_hash(cfg)
        except Exception as exc:  # noqa: BLE001 - an invalid trial is a recorded failure
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            budget.record(  # it still counts: every method gets the same number of trials
                {"model": model, "overrides": list(trial)}, {SEARCH_METRIC: math.nan, "failed": 1.0}
            )
            ledger.add(
                {
                    "kind": "invalid_trial",
                    "model": model,
                    "overrides": list(trial),
                    "status": "failed",
                    "error": entry["error"],
                }
            )
            planned.append(entry)
            continue
        run_id = f"{base.run_id}-trial{index}"
        units.append(
            Unit(
                f"{model}:trial{index}",
                model,
                config,
                search_seed,
                overrides,
                runs_dir,
                run_id,
                profile,
            )
        )
        entry["run_id"] = f"{run_id}-{profile}-s{search_seed}"
        planned.append(entry)
    hashes = [e["config_hash"] for e in planned if "config_hash" in e]
    if len(set(hashes)) != len(hashes):
        raise ValueError(
            f"{config}: two trials resolve to the same configuration; a repeated trial wastes "
            "budget"
        )
    results = {
        r.unit.label: r
        for r in run_units(
            units,
            jobs=jobs,
            profile_steps=profile_steps,
            power=power,
            logs_dir=logs_dir,
            ledger=ledger,
            poll_s=poll_s,
        )
    }
    for entry in planned:
        label = f"{model}:trial{entry['index']}"
        if label not in results:
            continue
        outcome = results[label]
        entry["seconds"] = outcome.seconds
        if outcome.run_dir is None:
            entry.update(status="failed", error=outcome.error)
            budget.record(
                {"model": model, "overrides": entry["overrides"]},
                {SEARCH_METRIC: math.nan, "failed": 1.0},
            )
            continue
        metrics = _read_json(outcome.run_dir / METRICS_FILE)
        try:
            result = g1_result(metrics, frozen)
        except KeyError as exc:
            entry.update(status="failed", error=str(exc))
            budget.record(
                {"model": model, "overrides": entry["overrides"]},
                {SEARCH_METRIC: math.nan, "failed": 1.0},
            )
            continue
        entry.update(
            status=outcome.status,
            run_dir=str(outcome.run_dir),
            result=result,
            steps=metrics["steps"],
            dev_per_group={
                g: float(e["accuracy"]) for g, e in metrics["final"]["per_group"].items()
            },
            tokens_per_s=metrics["measured"].get("tokens_per_s"),
        )
        budget.record(
            load_config(outcome.run_dir / RESOLVED_CONFIG_FILE), {**result, "failed": 0.0}
        )
    best = pick_best(planned)
    return {
        "model": model,
        "config": str(config),
        "search_steps": search_steps,
        "search_seed": search_seed,
        "trials_allowed": budget.trials_allowed,
        "trials_used": budget.trials_used,
        "trials": planned,
        "best_index": best,
        "best_overrides": planned[best]["overrides"] if best is not None else None,
        "selection": f"highest {SEARCH_METRIC}, then highest g1_min_dev_accuracy, then lowest "
        "dev_loss, then lowest index",
        # the budget keeps NaN for a failed trial (it never wins); strict JSON has no NaN
        "budget": json.loads(budget.to_json(), parse_constant=lambda _: None),
    }


# --------------------------------------------------------------------------------------------------
# Stage 7: test, once
# --------------------------------------------------------------------------------------------------


def evaluate_test(run_dir: Path, *, eval_batch: int, store: ResultStore) -> dict[str, Any]:
    """Evaluate the final checkpoint of ``run_dir`` on the test split exactly once (the file
    ``test_eval.json`` in the run directory is the witness and is never overwritten)."""
    witness = run_dir / TEST_EVAL_FILE
    if witness.is_file():
        logger.info("test already evaluated for %s; reusing", run_dir.name)
        return _read_json(witness)
    cfg = load_config(
        run_dir / RESOLVED_CONFIG_FILE,
        overrides=[f"train.batch={eval_batch}", f"train.micro_batch={eval_batch}"],
    )
    metrics = _read_json(run_dir / METRICS_FILE)
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"{run_dir} holds no checkpoint to evaluate")
    ctx = resolve(cfg.hardware)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    loaded = load_checkpoint(checkpoint, model)
    if int(loaded["step"]) != int(metrics["steps"]):
        raise ValueError(
            f"{checkpoint} is at step {loaded['step']}, the run finished at step {metrics['steps']}"
        )
    trainer = make_trainer(model, cfg, ctx, run_dir=run_dir / "_test_eval")
    manifest = Path(str(cfg.data.manifest))
    test_ds = ByteDataset.load(
        manifest,
        "test",
        list(cfg.data.tasks) or None,
        list(cfg.data.variants) or None,
        expected_unroll=cfg.train.unroll,
    )
    started = time.perf_counter()
    result = trainer.evaluate(test_ds, allow_test=True)
    record = {
        "run_id": metrics["run_id"],
        "model_name": metrics["model_name"],
        "seed": metrics["seed"],
        "config_hash": metrics["config_hash"],
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "step": int(loaded["step"]),
        "eval_batch": eval_batch,
        "split": "test",
        "accuracy": float(result["accuracy"]),
        "loss": float(result["loss"]),
        "n_positions": int(result["n_positions"]),
        "n_docs": int(result["n_docs"]),
        "per_group": {
            g: {
                "accuracy": float(e["accuracy"]),
                "loss": float(e["loss"]),
                "n_positions": int(e["n_positions"]),
            }
            for g, e in result["per_group"].items()
        },
        "dev_final": {
            "accuracy": float(metrics["final"]["accuracy"]),
            "loss": float(metrics["final"]["loss"]),
            "per_group": {
                g: float(e["accuracy"]) for g, e in metrics["final"]["per_group"].items()
            },
        },
        "seconds": time.perf_counter() - started,
        "evaluated_at": _utc_now(),
    }
    _write_json(witness, record)
    common = {
        "config_hash": metrics["config_hash"],
        "model_name": metrics["model_name"],
        "profile": metrics["profile"],
        "seed": metrics["seed"],
        "git_rev": metrics.get("git_rev"),
        "step": int(loaded["step"]),
        "is_rerun": bool(metrics.get("is_rerun")),
        "split": "test",
    }
    for metric in ("loss", "accuracy"):
        store.append(
            metrics["run_id"],
            {**common, "group": "all", "metric": metric, "value": float(record[metric])},
        )
        for group, entry in record["per_group"].items():
            store.append(
                metrics["run_id"],
                {**common, "group": group, "metric": metric, "value": float(entry[metric])},
            )
    return record


# --------------------------------------------------------------------------------------------------
# Stage 8: idle profiling pass
# --------------------------------------------------------------------------------------------------


def idle_profile(run_dir: Path, *, steps: int, power: str, out: Path, tag: str) -> dict[str, Any]:
    """The P0.6 profiling pass on the trained model of ``run_dir`` with nothing else of this gate
    running; writes ``profiler_g1_<tag>_{train,serve}.json`` into ``out``."""
    cfg = load_config(run_dir / RESOLVED_CONFIG_FILE)
    ctx = resolve(cfg.hardware)
    train_ds, dev_ds = load_datasets(cfg)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"{run_dir} holds no checkpoint to profile")
    load_checkpoint(checkpoint, model)
    model.to(device=ctx.device, dtype=ctx.master_dtype)
    train_report, serve_report = profile_run(
        model,
        cfg,
        ctx,
        train_ds,
        dev_ds,
        steps=steps,
        power=lambda: make_power_meter(cfg, cast(PowerChoice, power)),
    )
    files = {}
    for suffix, report in (("train", train_report), ("serve", serve_report)):
        name = f"profiler_g1_{tag}_{suffix}.json"
        _write_json(out / name, report.to_dict())
        files[suffix] = name
    train = train_report.to_dict()
    segments = train["segments"]
    step_ms = sum(
        float(segments[n]["wall_ms_p50"])
        for n in ("shared_path", "backward", "optimizer")
        if n in segments
    )
    launches = [segments[n].get("launches", -1) for n in ("shared_path", "backward", "optimizer")]
    energy = [segments[n].get("energy_j") for n in ("shared_path", "backward", "optimizer")]
    return {
        "run_id": run_dir.name,
        "steps": steps,
        "files": files,
        "power_method": train["power"]["method"],
        "step_wall_ms_p50": step_ms,
        "launches_per_step": (
            sum(launches) / steps if all(isinstance(v, int) and v >= 0 for v in launches) else -1
        ),
        "energy_j_per_step": (
            sum(float(e) for e in energy) / steps if all(e is not None for e in energy) else None
        ),
        "gpu_processes": gpu_processes(),
    }


# --------------------------------------------------------------------------------------------------
# Stage 9: the decision
# --------------------------------------------------------------------------------------------------


def grad_norms_finite(history: Mapping[str, Any]) -> dict[str, Any]:
    """Every step's global ``grad_norm`` and every ``grad_norm/<group>`` must be a finite number
    (``null`` is what the runner writes for a non-finite value)."""
    steps = history.get("steps", [])
    n_global = 0
    n_groups = 0
    bad: list[str] = []
    for record in steps:
        n_global += 1
        if not _is_finite_number(record.get("grad_norm")):
            bad.append(f"step {record.get('step')}: grad_norm={record.get('grad_norm')}")
        for key, value in record.get("stats", {}).items():
            if key.startswith("grad_norm/"):
                n_groups += 1
                if not _is_finite_number(value):
                    bad.append(f"step {record.get('step')}: {key}={value}")
    return {
        "finite": n_global > 0 and not bad,
        "n_steps": n_global,
        "n_group_norms": n_groups,
        "non_finite": bad[:20],
    }


def decide(
    frozen: MetricsFreeze,
    seed_results: Sequence[Mapping[str, Any]],
    *,
    invariants_passed: bool,
    correct_reset: bool,
    environment_consistent: bool,
    overfit_passed: bool,
    test_evaluated_once: bool,
    data_protocol: bool = True,
    leakage_clean: bool = True,
    metrics_frozen: bool = True,
) -> dict[str, Any]:
    """The G1 verdict from measured numbers. ``seed_results`` has one entry per HSN seed run:
    ``{"run_id", "seed", "test": {task_key: accuracy}, "dev": {...}, "grad_norms_finite": bool}``.
    Thresholds are inclusive within 1e-9 (the P0.7 convention)."""
    g1 = frozen.thresholds.G1
    tol = 1e-9
    reasons: list[str] = []
    seeds = sorted({int(r["seed"]) for r in seed_results})
    n_seeds_ok = len(seeds) >= g1.n_seeds and len(seeds) == len(seed_results)
    if not n_seeds_ok:
        reasons.append(
            f"{len(seed_results)} seed runs for {len(seeds)} seeds, the freeze needs {g1.n_seeds}"
        )
    threshold_ok = bool(seed_results)
    per_seed: dict[str, dict[str, Any]] = {}
    for record in seed_results:
        entry: dict[str, Any] = {"run_id": record["run_id"], "tasks": {}}
        for task in frozen.g1_tasks:
            test = record.get("test", {}).get(task.key)
            dev = record.get("dev", {}).get(task.key)
            ok = _is_finite_number(test) and float(test) + tol >= task.threshold
            if not ok:
                threshold_ok = False
                reasons.append(
                    f"seed {record['seed']} {task.key}: test accuracy {test} < {task.threshold}"
                )
            entry["tasks"][task.key] = {
                "test_accuracy": test,
                "dev_accuracy": dev,
                "threshold": task.threshold,
                "passed": ok,
            }
        entry["grad_norms_finite"] = bool(record.get("grad_norms_finite"))
        per_seed[str(record["seed"])] = entry
    grads_ok = bool(seed_results) and all(bool(r.get("grad_norms_finite")) for r in seed_results)
    if not grads_ok:
        reasons.append("a logged gradient norm is not finite (or there is no seed run)")
    if not invariants_passed:
        reasons.append("the P1.8 invariant suite did not pass in this environment")
    if not correct_reset:
        reasons.append("the reset invariant did not run and pass")
    if not environment_consistent:
        reasons.append(
            "the invariant suite and the training runs are not from the same git revision"
        )
    if not overfit_passed:
        reasons.append("the 64-document overfit check did not reach the target")
    if not test_evaluated_once:
        reasons.append("the test split was not evaluated exactly once per run")
    checks = {
        "data_protocol": bool(data_protocol),
        "leakage_checks": bool(leakage_clean),
        "metrics_frozen": bool(metrics_frozen),
        "invariants_passed": bool(invariants_passed),
        "correct_reset": bool(correct_reset),
        "environment_consistent": bool(environment_consistent),
        "overfit_64_examples": bool(overfit_passed),
        "n_seeds": n_seeds_ok,
        "gradients_finite": grads_ok,
        "test_evaluated_once": bool(test_evaluated_once),
        "threshold_per_seed_per_task": threshold_ok,
    }
    passed = all(checks.values())
    return {
        "decision": "PASS" if passed else "FAIL",
        "checks": checks,
        "reasons": reasons,
        "per_seed": per_seed,
        "threshold": {t.key: t.threshold for t in frozen.g1_tasks},
        "n_seeds_required": g1.n_seeds,
        "metric": "test accuracy (fraction) over answer positions, per frozen task, per seed",
    }


# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", required=True, help="hardware profile, e.g. rtx3060 or cpu")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out", required=True, help="evidence directory, e.g. evidence/G1")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="the ref_a1 experiment config")
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=list(DEFAULT_BASELINES),
        help="baseline configurations run at the matched budget (none with an empty list)",
    )
    parser.add_argument("--runs", help="parent of the run directories (default runs/g1_<profile>)")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--rerun-tolerance",
        type=float,
        help="rerun tolerance on the overall dev accuracy; 0 (exact) on cpu, required elsewhere",
    )
    parser.add_argument("--power", choices=("none", "profile"), default="none")
    parser.add_argument("--profile-steps", type=int, default=20, help="profiling pass of every run")
    parser.add_argument(
        "--idle-profile-steps", type=int, default=100, help="steps of the idle-GPU profiling pass"
    )
    parser.add_argument("--overfit-docs", type=int, default=64)
    parser.add_argument("--overfit-steps", type=int, default=1000)
    parser.add_argument("--overfit-eval-every", type=int, default=50)
    parser.add_argument("--overfit-target", type=float, default=0.99)
    parser.add_argument(
        "--after-overfit-failure",
        choices=("stop", "continue"),
        default="stop",
        help="stop (exit 2, diagnose by hand; the chunk's rule) or continue to the search",
    )
    parser.add_argument(
        "--trial",
        action="append",
        nargs="+",
        metavar="KEY=VALUE",
        help="one override set per --trial (trial 0 is the config itself); default grid otherwise",
    )
    parser.add_argument("--search-steps", type=int, default=2000)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--eval-batch", type=int, default=32, help="batch of the test evaluation")
    parser.add_argument(
        "--invariants-junit",
        help="junit report of `pytest tests/invariants` run earlier in this environment",
    )
    parser.add_argument(
        "--invariants-git-rev",
        help="git revision the --invariants-junit report was produced at (same environment)",
    )
    parser.add_argument("--jobs", type=int, default=1, help="concurrent `hsn run` subprocesses")
    parser.add_argument(
        "--poll-seconds", type=float, default=5.0, help="poll interval of the --jobs launcher"
    )
    parser.add_argument(
        "--stop-after",
        choices=("overfit", "search"),
        help="exit 0 after this stage's file is written (re-invoke to resume, e.g. with --jobs)",
    )
    parser.add_argument("--data-spec", default=DEFAULT_SPEC)
    parser.add_argument("--data-out", default=DEFAULT_DATA_OUT)
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--no-golden", action="store_true")
    parser.add_argument("--freeze", default=None)
    return parser


def _cached(out: Path, name: str) -> dict[str, Any] | None:
    path = out / name
    if path.is_file():
        logger.info("reusing %s", path)
        return _read_json(path)
    return None


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901 - one stage after another
    parser = _parser()
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(out / "gate_g1.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    try:
        return _main(args, parser, out)
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def _main(args: argparse.Namespace, parser: argparse.ArgumentParser, out: Path) -> int:  # noqa: C901
    started = time.perf_counter()
    runs_dir = Path(args.runs) if args.runs else Path("runs") / f"g1_{args.profile}"
    seeds = parse_seeds(args.seeds)
    config = Path(args.config)
    baselines = [Path(c) for c in args.baselines]
    if args.profile == "cpu":
        tolerance = 0.0 if args.rerun_tolerance is None else args.rerun_tolerance
    elif args.rerun_tolerance is None:
        parser.error(f"--rerun-tolerance is required on profile {args.profile} (not bit-exact)")
    else:
        tolerance = args.rerun_tolerance
    base_overrides = [f"profile={args.profile}", *args.override]
    timings: dict[str, float] = {}
    gpu: dict[str, Any] = {"start": gpu_processes()}
    ledger = Ledger.load(out / "failed_configurations.jsonl")
    store = ResultStore(runs_dir / STORE_FILE)
    logs_dir = out / "logs"

    # 1. data protocol and leakage -----------------------------------------------------------------
    freeze_path = (
        Path(args.freeze) if args.freeze else REPO_ROOT / "configs" / "metrics_freeze.json"
    )
    frozen = load_metrics_freeze(freeze_path)
    logger.info(
        "freeze %s valid: frozen_at %s git %s", freeze_path, frozen.frozen_at, frozen.git_rev
    )
    cfg = load_config(config, overrides=base_overrides)
    check_task_set(cfg, frozen, str(config))
    data = ensure_data(
        Path(args.data_spec), Path(args.data_out), None if args.no_golden else Path(args.golden)
    )
    timings["data"] = data["seconds"]
    manifest = Path(data["manifest"])
    if Path(str(cfg.data.manifest)).resolve() != manifest.resolve():
        raise ValueError(
            f"{config} names data.manifest {cfg.data.manifest}, the gate verified {manifest}"
        )
    leakage = _cached(out, "leakage.json")
    if leakage is None:
        t0 = time.perf_counter()
        live = live_leakage_g1(manifest, frozen)
        leakage = {
            "manifest": data,
            "live": live,
            "clean": bool(data["manifest_clean"] and live["clean"]),
        }
        leakage["seconds"] = time.perf_counter() - t0
        _write_json(out / "leakage.json", leakage)
    logger.info(
        "leakage clean: %s (%d live pairs)", leakage["clean"], len(leakage["live"]["pairs"])
    )

    # 3. invariant suite ---------------------------------------------------------------------------
    invariants = _cached(out, "invariants.json")
    if invariants is None:
        invariants = run_invariant_suite(
            out,
            Path(args.invariants_junit) if args.invariants_junit else None,
            args.invariants_git_rev,
        )
        _write_json(out / "invariants.json", invariants)
    timings["invariants"] = float(invariants.get("seconds", 0.0))
    logger.info(
        "invariants (%s): %d tests, %d failures, %d errors, %d skipped -> passed %s, reset %s",
        invariants["mode"],
        invariants["tests"],
        invariants["failures"],
        invariants["errors"],
        invariants["skipped"],
        invariants["passed"],
        invariants["correct_reset"],
    )

    # 4. overfit -----------------------------------------------------------------------------------
    overfit = _cached(out, "overfit.json")
    if overfit is None:
        overfit = overfit_check(
            config,
            base_overrides,
            frozen,
            n_docs=args.overfit_docs,
            steps=args.overfit_steps,
            eval_every=args.overfit_eval_every,
            target=args.overfit_target,
            run_dir=runs_dir / "overfit",
        )
        _write_json(out / "overfit.json", overfit)
    timings["overfit"] = float(overfit["seconds"])
    gpu["after_overfit"] = gpu_processes()
    logger.info(
        "overfit %d docs: accuracy %.4f after %d steps (target %.2f) -> %s",
        overfit["n_docs"],
        overfit["final_train_accuracy"],
        overfit["steps_run"],
        overfit["target"],
        "PASS" if overfit["passed"] else "FAIL",
    )
    if not overfit["passed"] and args.after_overfit_failure == "stop":
        verdict = decide(
            frozen,
            [],
            invariants_passed=bool(invariants["passed"]),
            correct_reset=bool(invariants["correct_reset"]),
            environment_consistent=False,
            overfit_passed=False,
            test_evaluated_once=False,
            leakage_clean=bool(leakage["clean"]),
        )
        decision = {
            "gate": "G1",
            **verdict,
            "stopped_at": "overfit",
            "statement": "the overfit check failed; the gate stopped before the search (chunk "
            "P1.9: stop and diagnose)",
            "overfit": {
                k: overfit[k]
                for k in (
                    "n_docs",
                    "steps_run",
                    "final_train_accuracy",
                    "final_train_loss",
                    "target",
                    "reached_target_at_step",
                )
            },
            "profile": args.profile,
            "config": str(config),
            "timings_s": timings,
            "gpu_sharing": gpu,
            "decided_at": _utc_now(),
        }
        _write_json(out / "decision.json", decision)
        store.append(
            "G1",
            {
                "profile": args.profile,
                "metric": "decision",
                "value": 0.0,
                "text": "FAIL",
                "git_rev": frozen.git_rev,
                "step": overfit["steps_run"],
            },
        )
        print(
            f"\nG1 FAIL ({args.profile}): overfit check {overfit['final_train_accuracy']:.4f} < "
            f"{overfit['target']} after {overfit['steps_run']} steps; stopped (exit 2)"
        )
        return EXIT_OVERFIT

    if args.stop_after == "overfit":
        print(f"\nstopped after the overfit check as requested ({out / 'overfit.json'})")
        return EXIT_PASS

    # 5. search ------------------------------------------------------------------------------------
    trials = trial_sets(args.trial)
    search = _cached(out, "search.json")
    if search is None:
        t0 = time.perf_counter()
        methods: dict[str, Any] = {}
        matched = matched_overrides(cfg)
        plan = [(cfg.model_name, config, [])]
        plan += [(load_config(b).model_name, b, matched) for b in baselines]
        for model, path, extra in plan:
            if model in methods:
                raise ValueError(f"two configurations name model {model!r}: {path}")
            methods[model] = search_method(
                model=model,
                config=path,
                base_overrides=[*base_overrides, *extra],
                trials=trials,
                frozen=frozen,
                search_steps=args.search_steps,
                search_seed=args.search_seed,
                runs_dir=runs_dir / "search",
                profile=args.profile,
                jobs=args.jobs,
                profile_steps=args.profile_steps,
                power=args.power,
                logs_dir=logs_dir,
                ledger=ledger,
                poll_s=args.poll_seconds,
            )
            if methods[model]["best_index"] is None:
                raise RuntimeError(f"no trial of {model} finished; see {ledger.path}")
        search = {
            "trials": [list(t) for t in trials],
            "matched_overrides": matched,
            "methods": methods,
            "seconds": time.perf_counter() - t0,
        }
        _write_json(out / "search.json", search)
    if [list(t) for t in trials] != search["trials"]:
        raise ValueError(
            f"{out / 'search.json'} was made with trials {search['trials']}, this invocation "
            f"gives {trials}; delete the file to search again"
        )
    timings["search"] = float(search["seconds"])
    gpu["after_search"] = gpu_processes()
    for model, record in search["methods"].items():
        logger.info(
            "search %s: best trial %s %s", model, record["best_index"], record["best_overrides"]
        )

    if args.stop_after == "search":
        print(f"\nstopped after the search as requested ({out / 'search.json'})")
        return EXIT_PASS

    # 6. seeds and the rerun --------------------------------------------------------------------
    t0 = time.perf_counter()
    units: list[Unit] = []
    method_cfgs: dict[str, tuple[Path, list[str]]] = {}
    for model, record in search["methods"].items():
        path = Path(record["config"])
        extra = [] if model == cfg.model_name else list(search["matched_overrides"])
        overrides = [*base_overrides, *extra, *record["best_overrides"]]
        method_cfgs[model] = (path, overrides)
        run_id = load_config(path, overrides=overrides).run_id
        for seed in seeds:
            units.append(
                Unit(
                    f"{model}:s{seed}", model, path, seed, overrides, runs_dir, run_id, args.profile
                )
            )
    first_hsn = next(u for u in units if u.model == cfg.model_name)
    rerun_unit = Unit(
        f"{cfg.model_name}:rerun-s{first_hsn.seed}",
        cfg.model_name,
        None,
        first_hsn.seed,
        [],
        runs_dir,
        "",
        args.profile,
        kind="rerun",
        source=first_hsn.run_dir,
        tolerance=tolerance,
        depends_on=first_hsn.label,
    )
    outcomes = run_units(
        [*units, rerun_unit],
        jobs=args.jobs,
        profile_steps=args.profile_steps,
        power=args.power,
        logs_dir=logs_dir,
        ledger=ledger,
        poll_s=args.poll_seconds,
    )
    timings["seeds_and_rerun"] = time.perf_counter() - t0
    gpu["after_seeds"] = gpu_processes()
    seed_runs = [r for r in outcomes if r.unit.kind == "run"]
    rerun_result = next(r for r in outcomes if r.unit.kind == "rerun")
    by_model: dict[str, list[UnitResult]] = {}
    for result in seed_runs:
        by_model.setdefault(result.unit.model, []).append(result)
    hsn_dirs = [r.run_dir for r in by_model.get(cfg.model_name, []) if r.run_dir is not None]
    if not hsn_dirs:
        raise RuntimeError(f"no HSN seed run finished; see {ledger.path}")
    if rerun_result.run_dir is None:
        raise RuntimeError(f"the rerun of {first_hsn.label} failed; see {ledger.path}")
    rerun_dir = rerun_result.run_dir
    verdict_rerun = _read_json(rerun_dir / RERUN_FILE)

    # 7. test, once --------------------------------------------------------------------------------
    t0 = time.perf_counter()
    all_dirs = [r.run_dir for r in seed_runs if r.run_dir is not None]
    test_results: dict[str, Any] = {}
    for run_dir in all_dirs:
        test_results[run_dir.name] = evaluate_test(run_dir, eval_batch=args.eval_batch, store=store)
    _write_json(
        out / "test_results.json",
        {
            "eval_batch": args.eval_batch,
            "runs": test_results,
            "note": "one evaluation of the test split per run, from its final checkpoint",
        },
    )
    timings["test"] = time.perf_counter() - t0
    test_once = all(
        store.query(run_id=d.name, split="test", group="all", metric="accuracy").num_rows == 1
        for d in all_dirs
    )

    # 8. idle profile ------------------------------------------------------------------------------
    t0 = time.perf_counter()
    profiles: dict[str, Any] = {}
    for model, results in by_model.items():
        first = next((r.run_dir for r in results if r.run_dir is not None), None)
        if first is None:
            continue
        name = f"profiler_g1_{model}_train.json"
        if (out / name).is_file() and (out / f"profiler_g1_{model}_serve.json").is_file():
            logger.info("reusing idle profile of %s", model)
            profiles[model] = {
                "run_id": first.name,
                "files": {"train": name, "serve": f"profiler_g1_{model}_serve.json"},
                "reused": True,
            }
            continue
        profiles[model] = idle_profile(
            first, steps=args.idle_profile_steps, power=args.power, out=out, tag=model
        )
    timings["idle_profile"] = time.perf_counter() - t0

    # 9. decision and package ----------------------------------------------------------------------
    seed_results: list[dict[str, Any]] = []
    grad_reports: dict[str, Any] = {}
    revs: set[str | None] = set()
    for result in by_model.get(cfg.model_name, []):
        if result.run_dir is None:
            continue
        metrics = _read_json(result.run_dir / METRICS_FILE)
        grads = grad_norms_finite(_read_json(result.run_dir / HISTORY_FILE))
        grads["metrics_grad_norms_finite"] = bool(metrics.get("grad_norms_finite"))
        grad_reports[result.run_dir.name] = grads
        revs.add(metrics.get("git_rev"))
        test = test_results[result.run_dir.name]
        seed_results.append(
            {
                "run_id": metrics["run_id"],
                "seed": int(metrics["seed"]),
                "test": {g: e["accuracy"] for g, e in test["per_group"].items()},
                "dev": {g: float(e["accuracy"]) for g, e in metrics["final"]["per_group"].items()},
                "grad_norms_finite": bool(grads["finite"] and grads["metrics_grad_norms_finite"]),
            }
        )
    env_consistent = invariants.get("git_rev") is not None and revs == {invariants.get("git_rev")}
    verdict = decide(
        frozen,
        seed_results,
        invariants_passed=bool(invariants["passed"]),
        correct_reset=bool(invariants["correct_reset"]),
        environment_consistent=env_consistent,
        overfit_passed=bool(overfit["passed"]),
        test_evaluated_once=test_once,
        leakage_clean=bool(leakage["clean"]),
    )
    method_table: dict[str, Any] = {}
    hsn_params = float(_read_json(hsn_dirs[0] / METRICS_FILE)["measured"]["param_count"])
    for model, results in by_model.items():
        rows = []
        for r in results:
            if r.run_dir is None:
                rows.append({"label": r.unit.label, "status": "failed"})
                continue
            m = _read_json(r.run_dir / METRICS_FILE)
            t = test_results[r.run_dir.name]
            rows.append(
                {
                    "run_id": m["run_id"],
                    "seed": m["seed"],
                    "status": r.status,
                    "steps": m["steps"],
                    "param_count": m["measured"]["param_count"],
                    "tokens_per_s": m["measured"]["tokens_per_s"],
                    "peak_bytes": m["measured"]["peak_bytes"],
                    "launches_per_step": m["measured"].get("launches_per_step"),
                    "dev": {g: float(e["accuracy"]) for g, e in m["final"]["per_group"].items()},
                    "test": {g: e["accuracy"] for g, e in t["per_group"].items()},
                    "grad_norms_finite": bool(m.get("grad_norms_finite")),
                    "elapsed_s": m["elapsed_s"],
                    "power_method": m["power"]["method"],
                    "gpu_processes_at_end": r.gpu_processes,
                }
            )
        params = [float(r["param_count"]) for r in rows if "param_count" in r]
        method_table[model] = {
            "runs": rows,
            "param_count": params[0] if params else None,
            "params_within_10pct_of_hsn": bool(params)
            and abs(params[0] - hsn_params) <= 0.1 * hsn_params,
            "best_overrides": search["methods"][model]["best_overrides"],
        }
    measured_profiles = [p for p in profiles.values() if not p.get("reused")]
    energy_metered = (
        all(p.get("power_method") not in (None, "none") for p in measured_profiles)
        if measured_profiles
        else None
    )
    reports = {
        "baselines_exist": all(
            any(r.run_dir is not None for r in by_model.get(m, []))
            for m in search["methods"]
            if m != cfg.model_name
        ),
        "baselines_matched_params": all(
            v["params_within_10pct_of_hsn"] for k, v in method_table.items() if k != cfg.model_name
        ),
        # by construction: the same trials, the same horizon and the matched keys
        "baselines_same_budget": True,
        "profiler_exists": all(
            (d / "profile_train.json").is_file() and (d / "profile_serve.json").is_file()
            for d in all_dirs
        ),
        "energy_metered": energy_metered,
        "rerun_reproduces": bool(verdict_rerun["passed"]),
        "all_seed_runs_finished": all(r.run_dir is not None for r in seed_runs),
    }
    failed_payload = {
        "failed_configurations": ledger.entries,
        "negative_findings": [] if verdict["decision"] == "PASS" else verdict["reasons"],
        "note": "every failed or crashed configuration of this gate (search trials, seed runs) "
        "and the reasons of a FAIL decision; search.json holds every trial, including the ones "
        "that ran",
    }
    _write_json(out / "failed_configurations.json", failed_payload)
    gpu["end"] = gpu_processes()
    timings["total"] = time.perf_counter() - started
    decision = {
        "gate": "G1",
        **verdict,
        "statement": "a dense predictive cell of about 2M parameters reaches >= 95 % on the "
        "designated in-range state tasks, with finite gradients throughout and correct reset "
        "behaviour (plan Phase 1 header); decided on the test split, evaluated once per run",
        "reports": reports,
        "profile": args.profile,
        "seeds": seeds,
        "config": str(config),
        "config_hash_of_record": config_hash(
            load_config(method_cfgs[cfg.model_name][0], overrides=method_cfgs[cfg.model_name][1])
        ),
        "recipe_overrides": method_cfgs[cfg.model_name][1],
        "methods": method_table,  # one table per model: ref_a1 (hsn) and the baselines
        "hsn_runs": [
            r.run_dir.name if r.run_dir else None for r in by_model.get(cfg.model_name, [])
        ],
        "gradient_reports": grad_reports,
        "rerun": {
            k: verdict_rerun[k]
            for k in (
                "original_run_id",
                "rerun_run_id",
                "tolerance",
                "gated_abs_diff",
                "max_abs_diff",
                "max_abs_diff_loss",
                "model_hash_equal",
                "passed",
            )
        },
        "overfit": {
            k: overfit[k]
            for k in (
                "n_docs",
                "steps_run",
                "final_train_accuracy",
                "final_train_loss",
                "target",
                "reached_target_at_step",
                "passed",
            )
        },
        "invariants": {
            k: invariants[k]
            for k in (
                "mode",
                "tests",
                "failures",
                "errors",
                "skipped",
                "passed",
                "correct_reset",
                "git_rev",
                "git_dirty",
            )
        },
        "search": {
            m: {
                "best_index": v["best_index"],
                "best_overrides": v["best_overrides"],
                "trials": [
                    {
                        "index": t["index"],
                        "overrides": t["overrides"],
                        "status": t["status"],
                        "result": t.get("result"),
                        "error": t.get("error"),
                    }
                    for t in v["trials"]
                ],
            }
            for m, v in search["methods"].items()
        },
        "idle_profiles": profiles,
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
        "gpu_sharing": gpu,
        "jobs": args.jobs,
        "decided_at": _utc_now(),
    }
    _write_json(out / "decision.json", decision)
    store.append(
        "G1",
        {
            "profile": args.profile,
            "metric": "decision",
            "value": 1.0 if verdict["decision"] == "PASS" else 0.0,
            "text": verdict["decision"],
            "git_rev": frozen.git_rev,
            "step": int(cfg.train.steps),
        },
    )
    extras = {
        name: out / name
        for name in (
            "decision.json",
            "leakage.json",
            "overfit.json",
            "search.json",
            "invariants.json",
            "test_results.json",
            "failed_configurations.json",
        )
    }
    for p in profiles.values():
        for name in p["files"].values():
            extras[name] = out / name
    if (out / "invariants.xml").is_file():
        extras["invariants.xml"] = out / "invariants.xml"
    package_path = out / "G1.zip"
    if package_path.exists():
        package_path = out / f"G1-{_stamp()}.zip"
    package = evidence_package(
        "G1", [*all_dirs, rerun_dir], package_path, freeze=freeze_path, extra_files=extras
    )
    decision["package"] = {
        "path": str(package),
        "sha256": _sha256(package),
        "bytes": package.stat().st_size,
    }
    decision["timings_s"]["total"] = time.perf_counter() - started
    _write_json(out / "decision.json", decision)

    print(
        f"\nG1 {verdict['decision']} ({args.profile}); package {package} sha256 "
        f"{decision['package']['sha256']}"
    )
    print(
        "checks: " + ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in verdict["checks"].items())
    )
    print("reports: " + ", ".join(f"{k}={v}" for k, v in reports.items()))
    print(
        "\nmodel | run_id | seed | steps | dev per task | test per task | tokens/s | peak MiB | "
        "elapsed s"
    )
    for model, table in method_table.items():
        for row in table["runs"]:
            if "run_id" not in row:
                print(f"{model} | {row['label']} | failed")
                continue
            dev = " ".join(f"{g.split('/')[0]}={a:.3f}" for g, a in sorted(row["dev"].items()))
            test = " ".join(f"{g.split('/')[0]}={a:.3f}" for g, a in sorted(row["test"].items()))
            print(
                f"{model} | {row['run_id']} | {row['seed']} | {row['steps']} | {dev} | {test} | "
                f"{row['tokens_per_s']:.0f} | {row['peak_bytes'] / 2**20:.0f} | "
                f"{row['elapsed_s']:.0f}"
            )
    for reason in verdict["reasons"]:
        print(f"- {reason}")
    print("timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items()))
    return EXIT_PASS if verdict["decision"] == "PASS" else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
