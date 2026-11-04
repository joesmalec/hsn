"""P0.8: append-only result store, the frozen metrics file and gate evidence packages.

Plan references: chunk P0.8 (interfaces), §4.3 (run artefacts), §8.1–8.2 (gate thresholds and stop
rules), §2.8–2.9 (metrics and gate statistics), §2.10 (cost arithmetic), Track E (what every gate
package contains).

Three durable outputs of experiment runs live here:

* :class:`ResultStore`: one JSON-lines file that is only ever appended to (there is no update or
  delete), read back as a :class:`pyarrow.Table`.
* ``configs/metrics_freeze.json``: the primary metric per suite, the quality tolerances, the G1 task
  set and every gate threshold. The pydantic models below are its JSON schema
  (:func:`metrics_freeze_schema`) and the file carries a SHA-256 of its body, so an in-place edit is
  detected (:class:`FreezeTamperedError`). A committed freeze file is never edited; a later change
  is a  new versioned file with a written justification (global rule 3, §8.2).
* :func:`evidence_package`: the ``hsn package --gate Gn`` zip whose ``SUMMARY.md`` lists every
  artefact Track E requires and where each one is inside the zip.

Run directories (written by :mod:`hsn.runner`) are read through :class:`RunRecord`; the file names
are the module constants so the writer and the reader cannot drift apart.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as pa_json
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from hsn import __version__
from hsn.config import REPO_ROOT
from hsn.eval.bootstrap import CI, PairedSamples, bootstrap_mean, paired_bootstrap

__all__ = [
    "FREEZE_PATH",
    "HISTORY_FILE",
    "MANIFEST_FILE",
    "METRICS_FILE",
    "PROFILE_SERVE_FILE",
    "PROFILE_TRAIN_FILE",
    "REQUIRED_ARTIFACTS",
    "RERUN_FILE",
    "RESOLVED_CONFIG_FILE",
    "STORE_FILE",
    "FreezeError",
    "FreezeSchemaError",
    "FreezeTamperedError",
    "MetricsFreeze",
    "MissingArtifactError",
    "ResultStore",
    "RunRecord",
    "evidence_package",
    "freeze_body_hash",
    "load_metrics_freeze",
    "metrics_freeze_schema",
    "write_metrics_freeze",
]

# File names inside a run directory (writer: hsn.runner; reader: RunRecord / evidence_package).
METRICS_FILE = "metrics.json"
MANIFEST_FILE = "run_manifest.json"  # written by hsn.config.write_run_manifest
RESOLVED_CONFIG_FILE = "config.resolved.yaml"  # written by hsn.config.write_run_manifest
HISTORY_FILE = "history.json"
PROFILE_TRAIN_FILE = "profile_train.json"
PROFILE_SERVE_FILE = "profile_serve.json"
RERUN_FILE = "rerun.json"
STORE_FILE = (
    "results.jsonl"  # the ResultStore next to the run directories (``<out>/results.jsonl``)
)

FREEZE_PATH = REPO_ROOT / "configs" / "metrics_freeze.json"

Scalar = str | int | float | bool | None
_RESERVED_COLUMNS = ("run_id", "appended_at")


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------------
# Result store
# --------------------------------------------------------------------------------------------------


class ResultStore:
    """Append-only store of flat result records, one JSON object per line.

    ``append`` only ever adds a line (the file is opened for appending and synced); nothing here
    rewrites or removes a record, which is what "append-only" (P0.8 pitfalls) means for results that
    gate decisions cite. A record is a flat mapping of scalars (``str``, ``int``, ``float``,
    ``bool``
    or ``None``); nested values and non-finite floats raise instead of being coerced. The row
    convention the runner uses (long format) is: ``config_hash, model_name, profile, seed, step,
    split, group ("<task>/<variant>" or "all"), metric, value, text, git_rev``.

    ``query(**filters)`` reads the whole file into a :class:`pyarrow.Table` and keeps the rows whose
    columns equal the given values (``None`` matches null). A filter on a column no record has
    raises ``KeyError``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, run_id: str, record: Mapping[str, Any]) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"run_id must be a non-empty string, got {run_id!r}")
        row: dict[str, Scalar] = {"run_id": run_id, "appended_at": _utc_now()}
        for key, value in record.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"record keys must be non-empty strings, got {key!r}")
            if key in _RESERVED_COLUMNS:
                raise ValueError(f"record may not set the reserved column {key!r}")
            if isinstance(value, bool | int | float | str) or value is None:
                row[key] = value
            else:
                raise TypeError(
                    f"record[{key!r}] must be a scalar (str, int, float, bool, None), "
                    f"got {type(value).__name__}"
                )
        line = json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def __len__(self) -> int:
        if not self.path.is_file():
            return 0
        with self.path.open("rb") as fh:
            return sum(1 for line in fh if line.strip())

    def query(self, **filters: Scalar) -> pa.Table:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            table = pa.table({"run_id": pa.array([], pa.string())})
        else:
            table = pa_json.read_json(self.path)
        for key, value in filters.items():
            if key not in table.column_names:
                raise KeyError(
                    f"no record in {self.path} has a column {key!r}: {table.column_names}"
                )
            column = table[key]
            mask = pc.is_null(column) if value is None else pc.equal(column, value)
            table = table.filter(mask)
        return table


# --------------------------------------------------------------------------------------------------
# Frozen metrics file
# --------------------------------------------------------------------------------------------------


class FreezeError(ValueError):
    """The freeze file cannot be used."""


class FreezeTamperedError(FreezeError):
    """The embedded ``sha256_of_body`` does not match the body (an in-place edit)."""


class FreezeSchemaError(FreezeError):
    """The body does not match the JSON schema."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _fraction(**extra: Any) -> Any:
    """A threshold stored as a fraction in (0, 1): 0.02 = 2 percentage points (P0.7 convention)."""
    return Field(gt=0.0, lt=1.0, **extra)


class G1Task(_Frozen):
    """One designated G1 task: ``<task>/<variant>`` must reach ``threshold`` accuracy per seed."""

    task: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    variant: Literal["in_range", "out_of_range"]
    threshold: float = Field(gt=0.0, le=1.0)

    @property
    def key(self) -> str:
        return f"{self.task}/{self.variant}"


class QualityTolerance(_Frozen):
    """§2.9: quality loss ≤ 1 pp accuracy or ≤ 2 % relative perplexity."""

    accuracy: float = _fraction()
    perplexity_relative: float = _fraction()


class G1Thresholds(_Frozen):
    accuracy_min: float = Field(gt=0.0, le=1.0)
    per_task: bool
    per_seed: bool
    n_seeds: int = Field(ge=1)
    finite_gradients: bool
    correct_reset: bool


class G2Thresholds(_Frozen):
    quality_tolerance: QualityTolerance
    improvement_min: float = _fraction()
    claims: list[Literal["latency", "energy"]] = Field(min_length=1)
    claim_both_only_if_both_pass: bool
    other_regression_max: float | None  # §2.9 names the budget without a number (null until set)
    n_seeds: int = Field(ge=1)
    ci_alpha: float = _fraction()
    optimisation_cycles_max: int = Field(ge=0)


class G3Thresholds(_Frozen):
    mean_forgetting_max: float = _fraction()
    task_loss_max: float = _fraction()
    new_task_gain_min: float = _fraction()
    macro_gap_max: float = _fraction()
    n_seeds: int = Field(ge=1)


class G4Thresholds(_Frozen):
    composition_gain_min: float = _fraction()
    ci_alpha: float = _fraction()
    ci_excludes_zero: bool
    growth_max: float = _fraction()
    n_seeds: int = Field(ge=1)


class G5Thresholds(_Frozen):
    restore_drill_max_s: int = Field(gt=0)
    load_test_hours: int = Field(gt=0)
    version_consistency: bool


class G6Thresholds(_Frozen):
    pilot_weeks: int = Field(gt=0)
    unresolved_critical_security_issues_max: int = Field(ge=0)
    quality_within_tolerance: bool
    cost_within_budget: bool


class GateThresholds(_Frozen):
    G1: G1Thresholds
    G2: G2Thresholds
    G3: G3Thresholds
    G4: G4Thresholds
    G5: G5Thresholds
    G6: G6Thresholds


class MetricsFreeze(_Frozen):
    """``configs/metrics_freeze.json`` (P0.8). Every number cites the plan; none is tuned."""

    frozen_at: str
    git_rev: str = Field(pattern=r"^[0-9a-f]{40}$")
    primary_metrics: dict[str, str] = Field(min_length=1)
    quality_tolerance: QualityTolerance
    g1_tasks: list[G1Task] = Field(min_length=1)
    thresholds: GateThresholds
    sha256_of_body: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("frozen_at")
    @classmethod
    def _aware_iso(cls, value: str) -> str:
        parsed = dt.datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("frozen_at must be an ISO 8601 timestamp with a timezone")
        return value

    @field_validator("primary_metrics")
    @classmethod
    def _metric_names(cls, value: dict[str, str]) -> dict[str, str]:
        for suite, metric in value.items():
            for text in (suite, metric):
                if not text or not text.replace("_", "a").isalnum() or not text[0].isalpha():
                    raise ValueError(f"suite and metric names are identifiers, got {text!r}")
        return value

    @field_validator("g1_tasks")
    @classmethod
    def _distinct_tasks(cls, value: list[G1Task]) -> list[G1Task]:
        keys = [task.key for task in value]
        if len(set(keys)) != len(keys):
            raise ValueError(f"g1_tasks repeats a task: {keys}")
        return value

    def body(self) -> dict[str, Any]:
        """The hashed part: every field except ``sha256_of_body``."""
        return self.model_dump(mode="json", exclude={"sha256_of_body"})

    def g1_task_keys(self) -> list[str]:
        return [task.key for task in self.g1_tasks]


def metrics_freeze_schema() -> dict[str, Any]:
    """The JSON schema the freeze file is validated against."""
    return MetricsFreeze.model_json_schema()


def freeze_body_hash(body: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON (sorted keys, compact, ASCII) of ``body`` minus the hash."""
    stripped = {key: value for key, value in body.items() if key != "sha256_of_body"}
    text = json.dumps(stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def load_metrics_freeze(path: str | Path = FREEZE_PATH) -> MetricsFreeze:
    """Load, hash-check and schema-validate a freeze file.

    The hash catches an in-place edit of a committed file (a changed threshold, a dropped task);
    an edit that also recomputes the hash is caught by review and git history, not here.
    """
    file = Path(path)
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FreezeError(f"freeze file {file} does not exist") from None
    except json.JSONDecodeError as exc:
        raise FreezeError(f"freeze file {file} is not valid JSON: {exc}") from None
    if not isinstance(raw, dict):
        raise FreezeSchemaError(f"freeze file {file} must hold a JSON object")
    recorded = raw.get("sha256_of_body")
    if not isinstance(recorded, str):
        raise FreezeTamperedError(f"freeze file {file} carries no sha256_of_body")
    actual = freeze_body_hash(raw)
    if recorded != actual:
        raise FreezeTamperedError(
            f"freeze file {file} was edited in place: body hash {actual} != recorded {recorded}. "
            "A committed freeze file is never edited; supersede it with a new versioned file and a "
            "written justification (plan global rule 3, §8.2)."
        )
    try:
        return MetricsFreeze.model_validate(raw)
    except ValidationError as exc:
        raise FreezeSchemaError(f"freeze file {file} fails the schema: {exc}") from None


def write_metrics_freeze(body: Mapping[str, Any], path: str | Path) -> MetricsFreeze:
    """Write a new freeze file (body + computed hash). Refuses to overwrite an existing one."""
    file = Path(path)
    if file.exists():
        raise FreezeError(
            f"{file} exists; a freeze file is never rewritten, write a new versioned file instead"
        )
    payload = {key: value for key, value in body.items() if key != "sha256_of_body"}
    payload["sha256_of_body"] = freeze_body_hash(payload)
    try:
        frozen = MetricsFreeze.model_validate(payload)
    except ValidationError as exc:
        raise FreezeSchemaError(f"freeze body fails the schema: {exc}") from None
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return frozen


# --------------------------------------------------------------------------------------------------
# Run directories
# --------------------------------------------------------------------------------------------------


class MissingArtifactError(FileNotFoundError):
    """A run directory lacks a file the evidence package requires."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MissingArtifactError(f"missing {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise MissingArtifactError(f"{path} must hold a JSON object")
    return loaded


@dataclass(frozen=True)
class RunRecord:
    """Everything the package reads from one run directory."""

    run_dir: Path
    metrics: dict[str, Any]
    manifest: dict[str, Any]
    history: dict[str, Any]
    profile_train: dict[str, Any]
    profile_serve: dict[str, Any]
    rerun: dict[str, Any] | None

    @classmethod
    def load(cls, run_dir: str | Path) -> RunRecord:
        root = Path(run_dir)
        if not root.is_dir():
            raise MissingArtifactError(f"run directory {root} does not exist")
        if not (root / RESOLVED_CONFIG_FILE).is_file():
            raise MissingArtifactError(f"missing {root / RESOLVED_CONFIG_FILE}")
        rerun_file = root / RERUN_FILE
        return cls(
            run_dir=root,
            metrics=_read_json(root / METRICS_FILE),
            manifest=_read_json(root / MANIFEST_FILE),
            history=_read_json(root / HISTORY_FILE),
            profile_train=_read_json(root / PROFILE_TRAIN_FILE),
            profile_serve=_read_json(root / PROFILE_SERVE_FILE),
            rerun=_read_json(rerun_file) if rerun_file.is_file() else None,
        )

    @property
    def run_id(self) -> str:
        return str(self.metrics["run_id"])

    @property
    def model_name(self) -> str:
        return str(self.metrics["model_name"])

    @property
    def seed(self) -> int:
        return int(self.metrics["seed"])

    @property
    def is_rerun(self) -> bool:
        return self.rerun is not None

    @property
    def final(self) -> dict[str, Any]:
        final = self.metrics["final"]
        if not isinstance(final, dict):
            raise MissingArtifactError(f"{self.run_dir}: metrics.json has no final evaluation")
        return final

    def group_accuracy(self) -> dict[str, float]:
        per_group = self.final["per_group"]
        return {str(group): float(entry["accuracy"]) for group, entry in per_group.items()}


# --------------------------------------------------------------------------------------------------
# Evidence package
# --------------------------------------------------------------------------------------------------

# (key, title) in the order of the chunk's context paragraph, then the Track E extras.
REQUIRED_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("environment", "Source and environment versions"),
    ("data_manifests", "Data manifests and split hashes"),
    ("configuration", "Exact resolved configurations"),
    ("seed_results", "Full per-seed results"),
    ("uncertainty", "Uncertainty estimates (paired bootstrap)"),
    ("per_task_regressions", "Per-task regressions"),
    ("profiler_traces", "Profiler traces and energy readings"),
    ("cost", "Total training and serving cost"),
    ("hashes", "Model and adapter hashes"),
    ("rerun", "Independent rerun"),
    ("frozen_metrics", "Frozen metrics file"),
    ("failed_configurations", "Failed configurations and negative findings"),
)

_PLANNING_RATES_USD_PER_HOUR = (1.0, 3.0)  # §8.5: G0–G2 "about 1,200 GPU-hours at $1–3/h"
_GATE_NAMES = tuple(f"G{n}" for n in range(7))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str]) -> str | None:
    if shutil.which("git") is None:
        return None
    try:
        proc = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:,.0f}" if abs(value) >= 1000.0 else f"{value:.4f}"
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def _ci_dict(ci: CI) -> dict[str, float | int]:
    return {
        "estimate": ci.estimate,
        "low": ci.low,
        "high": ci.high,
        "alpha": ci.alpha,
        "n_boot": ci.n_boot,
        "n_units": ci.n_units,
        "seed": ci.seed,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def _environment(records: Sequence[RunRecord]) -> dict[str, Any]:
    status = _git(["status", "--porcelain"])
    lock = REPO_ROOT / "uv.lock"
    manifests = {
        record.run_id: {
            key: record.manifest.get(key)
            for key in (
                "git_rev",
                "dirty",
                "lock_hash",
                "python",
                "torch",
                "cuda_toolkit",
                "driver",
                "profile",
                "config_hash",
            )
        }
        for record in records
    }
    return {
        "packaged_at": _utc_now(),
        "package_git_rev": _git(["rev-parse", "HEAD"]),
        "package_git_dirty": bool(status) if status is not None else None,
        "package_lock_hash": (
            hashlib.blake2b(lock.read_bytes(), digest_size=32).hexdigest()
            if lock.is_file()
            else None
        ),
        "hsn_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "run_manifests": manifests,
        "distinct_git_revs": sorted({str(m["git_rev"]) for m in manifests.values()}),
        "any_run_dirty": any(bool(m["dirty"]) for m in manifests.values()),
    }


def _data_section(records: Sequence[RunRecord], stage: Path) -> tuple[list[str], dict[str, Any]]:
    """Copy every distinct data manifest; summarise their leakage pairs."""
    files: list[str] = []
    leakage: dict[str, Any] = {"manifests": {}, "clean": True}
    seen: dict[str, str] = {}
    for record in records:
        data = record.metrics.get("data", {})
        manifest_path = Path(str(data.get("manifest", "")))
        if not manifest_path.is_file():
            raise MissingArtifactError(
                f"{record.run_id}: data manifest {manifest_path} (metrics.json data.manifest) "
                "is not a file"
            )
        key = str(manifest_path.resolve())
        if key in seen:
            continue
        loaded = _read_json(manifest_path)
        name = str(loaded.get("name", manifest_path.parent.name))
        target = stage / "data" / f"{name}.manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest_path, target)
        seen[key] = name
        files.append(f"data/{name}.manifest.json")
        pairs = loaded.get("leakage", [])
        counts = [
            int(pair.get("exact_duplicates", 0)) + int(pair.get("overlapping_ngrams", 0))
            for pair in pairs
        ]
        clean = all(count == 0 for count in counts)
        leakage["manifests"][name] = {
            "path": str(manifest_path),
            "spec_hash": loaded.get("spec_hash"),
            "split_hashes": loaded.get("split_hashes"),
            "n_pairs": len(pairs),
            "worst_count": max(counts) if counts else 0,
            "clean": clean,
            "pairs": pairs,
        }
        leakage["clean"] = bool(leakage["clean"] and clean)
    _write_json(stage / "data" / "leakage.json", leakage)
    files.append("data/leakage.json")
    return files, leakage


def _seed_rows(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        measured = record.metrics.get("measured", {})
        final = record.final
        rows.append(
            {
                "run_id": record.run_id,
                "model_name": record.model_name,
                "profile": record.metrics.get("profile"),
                "seed": record.seed,
                "is_rerun": record.is_rerun,
                "rerun_of": record.rerun.get("original_run_id") if record.rerun else None,
                "config_hash": record.metrics.get("config_hash"),
                "steps": record.metrics.get("steps"),
                "param_count": measured.get("param_count"),
                "dev_loss": final.get("loss"),
                "dev_accuracy": final.get("accuracy"),
                "dev_n_positions": final.get("n_positions"),
                "per_group": final.get("per_group"),
                "train_loss_final": record.metrics.get("train_loss_final"),
                "tokens_per_s": measured.get("tokens_per_s"),
                "peak_bytes": measured.get("peak_bytes"),
                "train_seconds": measured.get("train_seconds"),
                "elapsed_s": record.metrics.get("elapsed_s"),
                "model_hash": record.metrics.get("model_hash"),
            }
        )
    return rows


def _uncertainty(records: Sequence[RunRecord]) -> dict[str, Any]:
    """Bootstrap over task × seed units (never tokens, §2.8), paired by seed across models."""
    originals = [record for record in records if not record.is_rerun]
    per_model: dict[str, dict[str, float]] = {}
    for record in originals:
        units = per_model.setdefault(record.model_name, {})
        for group, accuracy in record.group_accuracy().items():
            units[f"{group}|seed{record.seed}"] = accuracy
    result: dict[str, Any] = {
        "unit": "task/variant x seed (final dev accuracy, fraction)",
        "per_model": {},
        "paired": {},
    }
    for model, units in sorted(per_model.items()):
        values = np.array(list(units.values()), dtype=np.float64)
        entry: dict[str, Any] = {"n_units": len(units), "mean": float(values.mean())}
        if len(units) >= 2:
            entry["ci"] = _ci_dict(bootstrap_mean(values))
        else:
            entry["ci"] = None
            entry["note"] = "bootstrap needs at least two units"
        result["per_model"][model] = entry
    for baseline, candidate in combinations(sorted(per_model), 2):
        samples = PairedSamples.from_mappings(per_model[baseline], per_model[candidate])
        entry = {"baseline": baseline, "candidate": candidate, "n_units": samples.n_units}
        if samples.n_units >= 2:
            ci = paired_bootstrap(samples.candidate, samples.baseline)
            entry["difference_ci"] = _ci_dict(ci)
            entry["ci_excludes_zero"] = bool(ci.low > 0.0 or ci.high < 0.0)
        else:
            entry["difference_ci"] = None
        result["paired"][f"{candidate} - {baseline}"] = entry
    return result


def _per_task(records: Sequence[RunRecord]) -> dict[str, Any]:
    originals = [record for record in records if not record.is_rerun]
    by_id = {record.run_id: record for record in records}
    table: dict[str, dict[str, Any]] = {}
    for record in originals:
        for group, accuracy in record.group_accuracy().items():
            entry = table.setdefault(group, {}).setdefault(record.model_name, {"seeds": {}})
            entry["seeds"][str(record.seed)] = accuracy
    for models in table.values():
        for entry in models.values():
            values = np.array(list(entry["seeds"].values()), dtype=np.float64)
            entry.update(
                mean=float(values.mean()),
                min=float(values.min()),
                max=float(values.max()),
                std=float(values.std(ddof=0)),
                n_seeds=int(values.size),
            )
    reruns: dict[str, Any] = {}
    for record in records:
        if record.rerun is None:
            continue
        original_id = str(record.rerun.get("original_run_id"))
        original = by_id.get(original_id)
        deltas: dict[str, float] = {}
        if original is not None:
            base = original.group_accuracy()
            deltas = {
                group: accuracy - base[group]
                for group, accuracy in record.group_accuracy().items()
                if group in base
            }
        reruns[record.run_id] = {
            "original_run_id": original_id,
            "original_in_package": original is not None,
            "accuracy_delta_per_group": deltas,
            "max_abs_delta": max((abs(v) for v in deltas.values()), default=None),
        }
    return {
        "metric": "final dev accuracy (fraction)",
        "reference": None,
        "note": (
            "No earlier gate package exists at G0, so there is no reference to regress against; "
            "the per-task deltas here are rerun minus original."
        ),
        "per_task": table,
        "reruns": reruns,
    }


def _serve_tokens_per_s(profile: Mapping[str, Any]) -> float | None:
    segment = profile.get("segments", {}).get("shared_path")
    if not segment:
        return None
    device_ms = float(segment.get("device_ms", 0.0))
    tokens = float(segment.get("tokens", 0.0))
    return tokens / (device_ms / 1000.0) if device_ms > 0 and tokens > 0 else None


def _cost(records: Sequence[RunRecord]) -> dict[str, Any]:
    """§2.10: wall_hours = tokens / (tokens_per_s × 3600); device_hours = wall_hours × devices."""
    runs: dict[str, Any] = {}
    total_tokens = 0.0
    total_train_hours = 0.0
    total_elapsed_hours = 0.0
    for record in records:
        measured = record.metrics.get("measured", {})
        tokens = float(measured.get("tokens", 0.0))
        tokens_per_s = float(measured.get("tokens_per_s", 0.0))
        train_seconds = float(measured.get("train_seconds", 0.0))
        elapsed = float(record.metrics.get("elapsed_s", train_seconds))
        wall_hours = tokens / (tokens_per_s * 3600.0) if tokens_per_s > 0 else 0.0
        elapsed_hours = elapsed / 3600.0
        serve_tps = _serve_tokens_per_s(record.profile_serve)
        runs[record.run_id] = {
            "processed_tokens": tokens,
            "measured_tokens_per_s": tokens_per_s,
            "train_seconds": train_seconds,
            "wall_hours_training": wall_hours,
            "elapsed_hours_total": elapsed_hours,
            "concurrent_devices": 1,
            "device_hours": elapsed_hours * 1,
            "serving_tokens_per_s": serve_tps,
            "serving_seconds_per_million_tokens": (1e6 / serve_tps) if serve_tps else None,
            "profile": record.metrics.get("profile"),
        }
        total_tokens += tokens
        total_train_hours += wall_hours
        total_elapsed_hours += elapsed_hours
    return {
        "formulas": {
            "wall_hours": "processed_tokens / (measured_tokens_per_s * 3600)",
            "device_hours": "wall_hours * concurrent_devices",
            "compute_cost": "device_hours * hourly_rate",
        },
        "measured": {
            "n_runs": len(records),
            "total_processed_tokens": total_tokens,
            "total_wall_hours_training": total_train_hours,
            "total_elapsed_hours": total_elapsed_hours,
            "total_device_hours": total_elapsed_hours,
            "runs": runs,
        },
        "forecast": {
            "note": "Planning figures at the §8.5 rates for G0–G2 ($1–3 per device-hour); "
            "a forecast, "
            "not a measured result.",
            "rate_usd_per_hour": list(_PLANNING_RATES_USD_PER_HOUR),
            "compute_cost_usd": [
                total_elapsed_hours * rate for rate in _PLANNING_RATES_USD_PER_HOUR
            ],
        },
    }


def _hashes(records: Sequence[RunRecord]) -> dict[str, Any]:
    return {
        "model_hash_method": "blake2b-256 over the sorted state_dict (name, dtype, shape, bytes)",
        "checkpoint_hash_method": "sha256 of the checkpoint file",
        "models": {
            record.run_id: {
                "model_hash": record.metrics.get("model_hash"),
                "config_hash": record.metrics.get("config_hash"),
                "checkpoints": record.metrics.get("checkpoints", {}),
            }
            for record in records
        },
        "adapter_hashes": {},
        "adapter_note": "No adapters exist before P3.4 (fast local weights).",
    }


def _summary(
    gate: str,
    records: Sequence[RunRecord],
    artifact_files: Mapping[str, list[str]],
    environment: Mapping[str, Any],
    leakage: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    per_task: Mapping[str, Any],
    cost: Mapping[str, Any],
    freeze: MetricsFreeze,
    decision: Mapping[str, Any] | None,
) -> str:
    lines: list[str] = [
        f"# Evidence package {gate}",
        "",
        f"Packaged {environment['packaged_at']} with hsn {environment['hsn_version']} at git "
        f"{environment['package_git_rev']} (dirty: {environment['package_git_dirty']}). "
        f"{len(records)} run directories, {sum(1 for r in records if r.is_rerun)} of them reruns.",
        "",
        "## Required artefacts (plan P0.8 context paragraph and Track E)",
        "",
        _md_table(
            ["#", "Artefact", "Files in this package", "Status"],
            [
                (i + 1, title, ", ".join(f"`{f}`" for f in artifact_files[key]), "present")
                for i, (key, title) in enumerate(REQUIRED_ARTIFACTS)
            ],
        ),
        "",
    ]
    if decision is not None:
        lines += [
            f"## {gate} decision",
            "",
            f"**{decision.get('decision', '?')}** — " + str(decision.get("statement", "")),
            "",
        ]
        checks = decision.get("checks")
        if isinstance(checks, dict):
            lines += [
                _md_table(
                    ["Check", "Result"],
                    [(name, "PASS" if ok else "FAIL") for name, ok in checks.items()],
                ),
                "",
            ]
    lines += ["## Runs (final dev evaluation)", ""]
    rows = []
    for record in sorted(records, key=lambda r: (r.model_name, r.is_rerun, r.seed)):
        measured = record.metrics.get("measured", {})
        final = record.final
        rows.append(
            (
                record.run_id,
                record.model_name,
                str(record.metrics.get("profile")),
                record.seed,
                int(record.metrics.get("steps", 0)),
                int(measured.get("param_count", 0)),
                float(final.get("loss", float("nan"))),
                float(final.get("accuracy", float("nan"))),
                float(measured.get("tokens_per_s", 0.0)),
                f"{float(measured.get('peak_bytes', 0.0)) / 2**20:.1f}",
                "rerun of " + str(record.rerun["original_run_id"]) if record.rerun else "",
            )
        )
    lines += [
        _md_table(
            [
                "run_id",
                "model",
                "profile",
                "seed",
                "steps",
                "params",
                "dev loss",
                "dev acc",
                "tokens/s",
                "peak MiB",
                "note",
            ],
            rows,
        ),
        "",
        "## Per-task final dev accuracy (originals; mean / min / max over seeds)",
        "",
    ]
    task_rows = []
    for group, models in sorted(per_task["per_task"].items()):
        for model, entry in sorted(models.items()):
            seeds = ", ".join(f"s{s}={v:.4f}" for s, v in sorted(entry["seeds"].items()))
            task_rows.append(
                (group, model, entry["mean"], entry["min"], entry["max"], entry["n_seeds"], seeds)
            )
    lines += [
        _md_table(["task/variant", "model", "mean", "min", "max", "n", "per seed"], task_rows),
        "",
        "## Uncertainty (bootstrap over task × seed units, paired by seed)",
        "",
    ]
    unc_rows = []
    for model, entry in sorted(uncertainty["per_model"].items()):
        ci = entry.get("ci")
        unc_rows.append(
            (
                model,
                entry["n_units"],
                entry["mean"],
                f"[{ci['low']:.4f}, {ci['high']:.4f}]" if ci else "n/a",
            )
        )
    lines += [_md_table(["model", "units", "mean acc", "95 % CI"], unc_rows), ""]
    pair_rows = []
    for name, entry in sorted(uncertainty["paired"].items()):
        ci = entry.get("difference_ci")
        pair_rows.append(
            (
                name,
                entry["n_units"],
                f"{ci['estimate']:+.4f} [{ci['low']:+.4f}, {ci['high']:+.4f}]" if ci else "n/a",
                str(entry.get("ci_excludes_zero", "n/a")),
            )
        )
    if pair_rows:
        lines += [
            _md_table(["difference", "units", "estimate [95 % CI]", "excludes 0"], pair_rows),
            "",
        ]
    lines += ["## Independent rerun", ""]
    rerun_rows = []
    for record in records:
        if record.rerun is None:
            continue
        info = record.rerun
        rerun_rows.append(
            (
                info.get("original_run_id"),
                record.run_id,
                float(info.get("tolerance", 0.0)),
                float(info.get("max_abs_diff", float("nan"))),
                float(info.get("max_abs_diff_loss", float("nan"))),
                str(info.get("model_hash_equal")),
                "PASS" if info.get("passed") else "FAIL",
            )
        )
    lines += [
        _md_table(
            [
                "original",
                "rerun",
                "tolerance (accuracy)",
                "max |Δaccuracy|",
                "max |Δloss| (nats)",
                "model hash equal",
                "verdict",
            ],
            rerun_rows,
        ),
        "",
        "## Cost (§2.10; measured, then forecast)",
        "",
    ]
    measured = cost["measured"]
    lines += [
        f"- Measured: {measured['n_runs']} runs, {measured['total_processed_tokens']:.0f} training "
        f"tokens, {measured['total_wall_hours_training']:.3f} wall-hours of optimisation, "
        f"{measured['total_elapsed_hours']:.3f} elapsed hours = "
        f"{measured['total_device_hours']:.3f} "
        "device-hours (one device per run).",
        f"- Forecast (not a result): {cost['forecast']['note']} Cost range "
        f"${cost['forecast']['compute_cost_usd'][0]:.2f}–${cost['forecast']['compute_cost_usd'][1]:.2f}.",
        "",
    ]
    serve_rows = [
        (
            run_id,
            entry["profile"],
            entry["measured_tokens_per_s"],
            entry["serving_tokens_per_s"] or 0.0,
        )
        for run_id, entry in sorted(measured["runs"].items())
    ]
    lines += [
        _md_table(["run_id", "profile", "train tokens/s", "serve tokens/s"], serve_rows),
        "",
        "## Data and leakage",
        "",
    ]
    for name, entry in leakage["manifests"].items():
        lines.append(
            f"- `{name}` (spec {entry['spec_hash']}): {entry['n_pairs']} leakage pairs, "
            f"worst count "
            f"{entry['worst_count']}, clean: {entry['clean']}."
        )
    lines += [
        "",
        "## Frozen metrics",
        "",
        f"- `freeze/metrics_freeze.json` frozen at {freeze.frozen_at}, git {freeze.git_rev}, body "
        f"sha256 {freeze.sha256_of_body}.",
        f"- G1 tasks: {', '.join(f'{t.key} ≥ {t.threshold}' for t in freeze.g1_tasks)}.",
        f"- Primary metrics: {json.dumps(freeze.primary_metrics, sort_keys=True)}.",
        "",
        "## Profiler and energy",
        "",
    ]
    mapping = records[0].metrics.get("profile_segments", {})
    lines.append(f"- Segment mapping of the baselines: {json.dumps(mapping, sort_keys=True)}.")
    for record in records:
        power = record.profile_train.get("power", {})
        segments = record.profile_train.get("segments", {})
        step_ms = sum(
            float(segments[name]["wall_ms_p50"])
            for name in ("shared_path", "backward", "optimizer")
            if name in segments
        )
        serve = record.profile_serve.get("segments", {}).get("shared_path", {})
        serve_ms = float(serve.get("wall_ms_p50", float("nan")))
        lines.append(
            f"- `{record.run_id}`: timing {record.profile_train.get('timing_method')!r}, launches "
            f"{record.profile_train.get('launches_method')!r}, power method "
            f"{power.get('method')!r} ({power.get('n_unresolved')} unresolved readings); "
            f"training step p50 {step_ms:.2f} ms (shared_path + backward + optimizer), serving "
            f"window p50 {serve_ms:.2f} ms."
        )
    lines += [
        "",
        "## Failed configurations and negative findings",
        "",
        "- None recorded for this gate (see `failed_configurations.json`).",
        "",
        "Test results above are measured on the dev split; nothing here touched the test split. "
        "Forecasts are confined to the cost section and labelled as such.",
    ]
    return "\n".join(lines)


def evidence_package(
    gate: str,
    run_dirs: Sequence[Path],
    out: Path,
    *,
    freeze: str | Path = FREEZE_PATH,
    extra_files: Mapping[str, Path] | None = None,
) -> Path:
    """Assemble the ``gate`` evidence zip from ``run_dirs``; returns ``out``.

    Every run directory must carry the files :mod:`hsn.runner` writes; at least one of them must be
    a rerun (``rerun.json``). ``extra_files`` maps a name inside the zip (for example
    ``decision.json`` from a gate script) to a local file. Missing inputs raise
    :class:`MissingArtifactError`; an existing ``out`` is never overwritten.
    """
    if gate not in _GATE_NAMES:
        raise ValueError(f"gate must be one of {_GATE_NAMES}, got {gate!r}")
    out_path = Path(out)
    if out_path.suffix != ".zip":
        raise ValueError(f"out must end with .zip, got {out_path}")
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; evidence packages are never overwritten")
    if not run_dirs:
        raise ValueError("evidence_package needs at least one run directory")
    records = [RunRecord.load(run_dir) for run_dir in run_dirs]
    ids = [record.run_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"run ids repeat: {ids}")
    if not any(record.is_rerun for record in records):
        raise MissingArtifactError(
            "no run directory holds rerun.json: an independent rerun is a required artefact"
        )
    frozen = load_metrics_freeze(freeze)
    extras = dict(extra_files or {})
    for name, source in extras.items():
        if "/" in name or "\\" in name or not Path(source).is_file():
            raise ValueError(f"extra file {name!r} -> {source} must be a plain name and a file")

    with tempfile.TemporaryDirectory(prefix=f"hsn-{gate}-") as tmp:
        stage = Path(tmp) / gate
        stage.mkdir()
        files: dict[str, list[str]] = {key: [] for key, _ in REQUIRED_ARTIFACTS}

        environment = _environment(records)
        _write_json(stage / "environment.json", environment)
        files["environment"].append("environment.json")
        for record in records:
            target = stage / "runs" / record.run_id / MANIFEST_FILE
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record.run_dir / MANIFEST_FILE, target)
            files["environment"].append(f"runs/{record.run_id}/{MANIFEST_FILE}")

        data_files, leakage = _data_section(records, stage)
        files["data_manifests"] += data_files

        for record in records:
            target = stage / "configs" / f"{record.run_id}.resolved.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record.run_dir / RESOLVED_CONFIG_FILE, target)
            files["configuration"].append(f"configs/{record.run_id}.resolved.yaml")

        seed_rows = _seed_rows(records)
        _write_json(stage / "results" / "per_seed.json", seed_rows)
        files["seed_results"].append("results/per_seed.json")
        for record in records:
            _write_json(stage / "results" / "history" / f"{record.run_id}.json", record.history)
            _write_json(stage / "results" / "metrics" / f"{record.run_id}.json", record.metrics)
            files["seed_results"].append(f"results/history/{record.run_id}.json")
            files["seed_results"].append(f"results/metrics/{record.run_id}.json")

        uncertainty = _uncertainty(records)
        _write_json(stage / "results" / "uncertainty.json", uncertainty)
        files["uncertainty"].append("results/uncertainty.json")

        per_task = _per_task(records)
        _write_json(stage / "results" / "per_task_regressions.json", per_task)
        files["per_task_regressions"].append("results/per_task_regressions.json")

        for record in records:
            for suffix, payload in (
                ("train", record.profile_train),
                ("serve", record.profile_serve),
            ):
                _write_json(stage / "profiles" / f"{record.run_id}.{suffix}.json", payload)
                files["profiler_traces"].append(f"profiles/{record.run_id}.{suffix}.json")

        cost = _cost(records)
        _write_json(stage / "cost.json", cost)
        files["cost"].append("cost.json")

        _write_json(stage / "hashes.json", _hashes(records))
        files["hashes"].append("hashes.json")

        for record in records:
            if record.rerun is not None:
                _write_json(stage / "rerun" / f"{record.run_id}.json", record.rerun)
                files["rerun"].append(f"rerun/{record.run_id}.json")

        (stage / "freeze").mkdir()
        shutil.copyfile(Path(freeze), stage / "freeze" / "metrics_freeze.json")
        _write_json(stage / "freeze" / "schema.json", metrics_freeze_schema())
        files["frozen_metrics"] += ["freeze/metrics_freeze.json", "freeze/schema.json"]

        _write_json(
            stage / "failed_configurations.json",
            {
                "failed_configurations": [],
                "negative_findings": [],
                "note": "Track E: every configuration that failed and every negative finding of "
                "the "
                "gate is listed here; none for this package.",
            },
        )
        files["failed_configurations"].append("failed_configurations.json")

        decision: dict[str, Any] | None = None
        for name, source in extras.items():
            shutil.copyfile(source, stage / name)
            if name == "decision.json":
                decision = _read_json(stage / name)

        summary = _summary(
            gate,
            records,
            files,
            environment,
            leakage,
            uncertainty,
            per_task,
            cost,
            frozen,
            decision,
        )
        _write_text(stage / "SUMMARY.md", summary)

        manifest = {
            "gate": gate,
            "packaged_at": environment["packaged_at"],
            "files": {
                str(path.relative_to(stage)): _sha256_file(path)
                for path in sorted(stage.rglob("*"))
                if path.is_file()
            },
            "artifacts": files,
        }
        _write_json(stage / "MANIFEST.json", manifest)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    zf.write(path, f"{gate}/{path.relative_to(stage)}")
    return out_path
