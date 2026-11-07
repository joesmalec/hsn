"""Backfill of the P1.10 audit's experiments into the database (P1.11), so that "what has already
been run" is answerable from the database from day one. Imports without torch.

Sources (under the audit's ``evidence/EFF/``): ``reads_state/*_reads_state.json`` (protocol
``reads_state``) and ``hsn/*_summary.json``, ``e2/*_summary.json`` (protocol ``overfit``). Each file
becomes one ``done`` experiment marked ``imported`` (never blocking a new enqueue of the same arm),
in a pseudo-sweep ``imported_audit``; importing is idempotent (a file already imported with the same
sha256 is skipped). Per file:

* **family:** from the config path (``baselines/gru`` -> gru, ``transformer`` -> transformer, else
  hsn); a GRU run's variant ``pdg`` is recorded as ``none`` (the audit's scripts wrote the default);
* **config:** ``.../configs/<rel>`` maps to this repository's ``configs/<rel>``; a file without one
  takes the script's default ``configs/experiments/g1_ref_a1.yaml``, recorded as inferred; the
  resolved configuration is today's ``load_config(config, overrides)`` (later overrides win), and
  when that fails the raw overrides are kept and the arm id hashes them instead;
* **protocol parameters:** the steps actually trained are the last evaluation's step (the resample
  run trained 600 steps of a 3,000-step schedule); ``diag_every`` is the spacing of the diagnostics
  steps (0 without diagnostics); ``diag_docs`` 32 where a ``trained_state`` exists;
* **commit:** the recorded ``{sha, modified_tracked_files}`` where present, else unknown (never
  guessed);
* **card:** as written (``idle`` or ``shared``: a shared run is fine for a verdict, not for a time);
* **reads-state counts** are recomputed from the stored probes with the partitioning
  :func:`hsn.experiments.probe.outcome_counts`; the counts as written are kept, and a difference
  (files written before the partition fix) is flagged; the verdict is recomputed and compared;
* **times:** ``ended_at`` is the file's modification time and ``started_at`` that minus the recorded
  wall time, marked approximate.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hsn.config import REPO_ROOT, load_config
from hsn.experiments import db as dbm
from hsn.experiments.db import Database, dumps
from hsn.experiments.rules import evaluate_verdicts, parse_rules
from hsn.experiments.spec import (
    OverfitParams,
    ReadsStateParams,
    arm_identity,
    data_hash,
    digest,
    memory_key,
    slug,
)
from hsn.experiments.variant_names import OFFSET_VARIANTS, RETIRED, variant_params

__all__ = ["IMPORT_SWEEP", "ImportReport", "convert", "import_audit_dir", "outcome_counts"]

IMPORT_SWEEP = "imported_audit"
DEFAULT_CONFIG = "configs/experiments/g1_ref_a1.yaml"
_RULES = {"verdicts": {"reads_state": {"preset": "reads_state"}}}


def outcome_counts(probes: list[dict[str, Any]]) -> dict[str, int]:
    """:func:`hsn.experiments.probe.outcome_counts` without importing torch."""
    follow = sum(bool(p["changed"] and p["follows_new_value"]) for p in probes)
    other = sum(bool(p["changed"] and not p["follows_new_value"]) for p in probes)
    unchanged = sum(not p["changed"] for p in probes)
    return {"follow": follow, "other": other, "unchanged": unchanged}


def _verdict(follow: int, unchanged: int, correct: int) -> str:
    if correct == 0:
        return "no correct answers to probe"
    f, u = follow / correct, unchanged / correct
    if f >= 0.5 and u <= 0.1:
        return "reads state"
    if f < 0.2:
        return "does not read state"
    return "partial"


def _summarize(probes: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [p for p in probes if p["original_correct"]]
    counts = outcome_counts(correct)
    n = len(correct)
    answers = [p["argmax_original"] for p in probes]
    hist: dict[str, int] = {}
    for a in answers:
        hist[str(a)] = hist.get(str(a), 0) + 1
    return {
        "n_probed": len(probes),
        "first_query_accuracy": n / len(probes) if probes else None,
        "all": outcome_counts(probes),
        "correct_before_edit": {"n": n, **counts},
        "verdict": _verdict(counts["follow"], counts["unchanged"], n),
        "first_query_answer_histogram": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
        "distinct_first_query_answers": len(set(answers)),
        "correct_n": n,
        "correct_follow": counts["follow"],
        "correct_other": counts["other"],
        "correct_unchanged": counts["unchanged"],
        "correct_follow_frac": counts["follow"] / n if n else None,
        "correct_unchanged_frac": counts["unchanged"] / n if n else None,
    }


@dataclass
class ImportReport:
    imported: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    flagged: list[dict[str, Any]] = field(default_factory=list)


def _config_rel(raw: str | None) -> tuple[str, bool]:
    if not raw:
        return DEFAULT_CONFIG, True
    text = str(raw)
    marker = "configs/"
    if marker in text:
        return text[text.rindex(marker) :], False
    return text, False


def _family(config_rel: str) -> str:
    if "baselines/gru" in config_rel:
        return "gru"
    if "transformer" in config_rel:
        return "transformer"
    return "hsn"


def _diag_every(diagnostics: list[dict[str, Any]]) -> int:
    steps = sorted({int(d["step"]) for d in diagnostics})
    if not steps:
        return 0
    if len(steps) == 1:
        return steps[0]
    return min(b - a for a, b in zip(steps, steps[1:], strict=False))


def convert(path: Path, data: dict[str, Any], *, repo: Path = REPO_ROOT) -> dict[str, Any]:
    """The arm, experiment and result of one audit JSON (module docstring)."""
    kind = "reads_state" if path.name.endswith("_reads_state.json") else "overfit"
    config_rel, inferred = _config_rel(data.get("config"))
    family = _family(config_rel)
    variant = "none" if family != "hsn" else str(data.get("variant") or "pdg")
    flags: list[str] = []
    given = (
        {"offset_std": data["offset_std"]}
        if variant in OFFSET_VARIANTS and "offset_std" in data
        else None
    )
    if variant in RETIRED:  # it ran when the variant existed; it cannot be enqueued again
        vparams: dict[str, Any] = {}
        flags.append(f"variant {variant} retired since: {RETIRED[variant]}")
    else:
        vparams = variant_params(variant, given)
    overrides = [str(o) for o in data.get("overrides") or []]
    raw_evals = data.get("evals") or []
    last_step = int(raw_evals[-1]["step"]) if raw_evals else None
    diagnostics = data.get("diagnostics") or []
    if inferred:
        flags.append(f"config inferred ({DEFAULT_CONFIG})")
    try:
        cfg = load_config(repo / config_rel, overrides)
        config_json: Any = cfg.model_dump(mode="json")
        resolved = True
    except (ValueError, OSError) as exc:
        cfg = None
        config_json = {"config": config_rel, "overrides": overrides, "error": str(exc)}
        resolved = False
        flags.append(f"config not resolvable today: {exc}")
    schedule = cfg.train.steps if cfg is not None else None
    steps = last_step if (last_step is not None and last_step != schedule) else None
    if kind == "reads_state":
        params: Any = ReadsStateParams(
            name="reads_state",
            steps=steps,
            diag_every=_diag_every(diagnostics),
            diag_docs=32
            if data.get("trained_state")
            else ReadsStateParams.model_fields["diag_docs"].default,
        )
    else:
        params = OverfitParams(
            name="overfit",
            steps=steps,
            n_docs=int(data.get("n_docs") or 64),
            diag_every=_diag_every(diagnostics),
        )
    pdump = params.model_dump(mode="json")
    if cfg is not None:
        data_id = data_hash(cfg, repo, required=False)
        arm_id, group_id = arm_identity(
            cfg,
            family=family,
            variant=variant,
            variant_params=vparams,
            protocol=pdump,
            early_stop=[],
            data=data_id,
        )
        mem = memory_key(cfg, variant, pdump)
        config_hash_value = digest(config_json, 64)
        seed = cfg.seed
    else:
        data_id = "unknown"
        arm_id = digest({"unresolved": config_json, "protocol": pdump, "variant": variant})
        group_id, mem, config_hash_value = arm_id, "unresolved", "unresolved"
        seed = 0
    if kind == "reads_state":
        group = "delayed/in_range"
        evals = [
            {
                "step": int(e["step"]),
                "accuracy": float(e["dev_answer_accuracy"][group]),
                "loss": float(e["dev_answer_loss"][group]),
                "per_group": {
                    g: {"accuracy": float(a), "loss": float(e["dev_answer_loss"][g])}
                    for g, a in e["dev_answer_accuracy"].items()
                },
                "dev_answer_accuracy": e["dev_answer_accuracy"],
                "dev_answer_loss": e["dev_answer_loss"],
            }
            for e in raw_evals
        ]
    else:
        evals = [
            {
                "step": int(e["step"]),
                "accuracy": float(e["train_accuracy"]),
                "loss": float(e["loss"]),
                "per_group": {},
            }
            for e in raw_evals
        ]
    final: dict[str, Any] = {}
    if evals:
        final = {
            "dev_accuracy": evals[-1]["accuracy"],
            "dev_loss": evals[-1]["loss"],
            "eval_step": evals[-1]["step"],
            "steps": evals[-1]["step"],
        }
        for g, v in evals[-1]["per_group"].items():
            final[f"{g}/accuracy"], final[f"{g}/loss"] = v["accuracy"], v["loss"]
    if data.get("final_train_loss") is not None:
        final["train_loss"] = float(data["final_train_loss"])
    wall = data.get("train_wall_s", data.get("wall_s"))
    record: dict[str, Any] = {
        "protocol": kind,
        "family": family,
        "variant": {"variant": variant, **vparams},
        "autocast": data.get("autocast"),
        "param_count": data.get("param_count"),
        "measured": data.get("measured"),
        "evals": evals,
        "diagnostics": diagnostics,
        "train": [],
        "final": final,
        "trained_state": data.get("trained_state") or None,
        "stopped": None,
        "walls": {"fit_s": wall} if wall is not None else {},
        "imported_from": str(path),
    }
    if kind == "reads_state":
        probes = data.get("probes") or []
        summary = _summarize(probes)
        written = {k: data.get(k) for k in ("all", "correct_before_edit", "verdict")}
        record["probe"] = summary
        record["as_written"] = written
        if written["all"] is not None and written["all"] != summary["all"]:
            flags.append(
                f"'all' counts recounted with the partition: {written['all']} -> {summary['all']}"
            )
        if (
            written["correct_before_edit"] is not None
            and written["correct_before_edit"] != summary["correct_before_edit"]
        ):
            flags.append("correct_before_edit counts differ from the stored probes")
        if written["verdict"] is not None and written["verdict"] != summary["verdict"]:
            flags.append(
                f"verdict differs: written {written['verdict']!r}, "
                f"recomputed {summary['verdict']!r}"
            )
    else:
        record["overfit"] = {
            "first_ge_0.90": data.get("first_ge_0.90"),
            "first_ge_0.99": data.get("first_ge_0.99"),
            "step_seconds_p50": data.get("step_seconds_p50"),
            "final_train_accuracy": data.get("final_train_accuracy"),
        }
    record["import_flags"] = flags
    commit = data.get("commit") if isinstance(data.get("commit"), dict) else None
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC)
    started = mtime - dt.timedelta(seconds=float(wall)) if wall is not None else None
    verdicts = (
        evaluate_verdicts(parse_rules(_RULES), record, {}, family) if kind == "reads_state" else {}
    )
    name = str(data.get("name") or path.stem)
    return {
        "arm": {
            "arm_id": arm_id,
            "group_id": group_id,
            "memory_key": mem,
            "family": family,
            "protocol": kind,
            "protocol_json": dumps(pdump),
            "variant": variant,
            "variant_json": dumps(vparams),
            "seed": seed,
            "config_path": config_rel,
            "overrides_json": dumps(overrides),
            "config_json": dumps(config_json),
            "config_hash": config_hash_value,
            "data_hash": data_id,
            "early_stop_json": dumps([]),
        },
        "experiment": {
            "label": name,
            "run_id": slug(name),
            "status": "done",
            "enqueued_at": (started or mtime).isoformat(timespec="microseconds"),
            "started_at": started.isoformat(timespec="microseconds") if started else None,
            "ended_at": mtime.isoformat(timespec="microseconds"),
            "commit_sha": commit.get("sha") if commit else None,
            "dirty_json": dumps(commit.get("modified_tracked_files")) if commit else None,
            "card_status": data.get("card"),
            "card_json": dumps({"gpu_processes": data.get("gpu_processes")}),
            "imported": 1,
            "import_source": str(path),
            "times_approx": 1,
            "run_dir": str(path.parent / name) if (path.parent / name).is_dir() else None,
        },
        "result": {
            "record": record,
            "verdicts": verdicts,
            "json_path": str(path),
            "final_dev_accuracy": final.get("dev_accuracy"),
            "final_dev_loss": final.get("dev_loss"),
            "train_s": float(wall) if wall is not None else None,
            "total_s": None,
        },
        "flags": flags,
        "resolved": resolved,
        "name": name,
    }


def audit_files(audit_dir: Path) -> list[Path]:
    root = Path(audit_dir)
    return [
        *sorted((root / "reads_state").glob("*_reads_state.json")),
        *sorted((root / "hsn").glob("*_summary.json")),
        *sorted((root / "e2").glob("*_summary.json")),
    ]


def import_audit_dir(
    db: Database,
    audit_dir: str | Path,
    *,
    dry_run: bool = False,
    repo: Path = REPO_ROOT,
    log: Callable[[str], None] = print,
) -> ImportReport:
    report = ImportReport()
    files = audit_files(Path(audit_dir))
    if not files:
        raise FileNotFoundError(
            f"no audit result JSONs under {audit_dir} (reads_state/, hsn/, e2/)"
        )
    for path in files:
        blob = path.read_bytes()
        sha = hashlib.sha256(blob).hexdigest()
        existing = db.row(
            "SELECT exp_id FROM experiments WHERE import_source = ? AND import_sha = ?",
            (str(path), sha),
        )
        if existing is not None:
            report.skipped.append({"file": str(path), "exp_id": existing["exp_id"]})
            continue
        item = convert(path, json.loads(blob), repo=repo)
        entry = {
            "file": str(path),
            "name": item["name"],
            "arm_id": item["arm"]["arm_id"],
            "flags": item["flags"],
        }
        if item["flags"]:
            report.flagged.append(entry)
        if dry_run:
            report.imported.append(entry)
            continue
        with db.write() as conn:
            sweep_id = dbm.register_sweep(
                conn, IMPORT_SWEEP, "audit", f"import of {Path(audit_dir).resolve()}", None, _RULES
            )
            dbm.insert_arm(conn, item["arm"])
            dbm.add_membership(conn, sweep_id, item["arm"]["arm_id"], item["name"], 0, {})
            exp_id = dbm.insert_experiment(
                conn,
                arm_id=item["arm"]["arm_id"],
                sweep_id=sweep_id,
                import_sha=sha,
                **item["experiment"],
            )
            dbm.write_result(conn, exp_id, item["result"])
        report.imported.append({**entry, "exp_id": exp_id})
    log(
        f"imported {len(report.imported)}, skipped {len(report.skipped)} (already imported), "
        f"flagged {len(report.flagged)}"
    )
    return report
