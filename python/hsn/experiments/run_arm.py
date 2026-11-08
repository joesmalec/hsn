"""The child process of one experiment (P1.11):
``python -m hsn.experiments run-arm --db DB --exp N --root ROOT``, started detached by the worker.

It owns its experiment row while that row is ``running`` with its PID (guarded writes,
:mod:`hsn.experiments.db`):

1. records its PID (unless the worker already did), start time, host, commit (sha plus the dirty
   tracked files; their diff goes to ``dirty.patch``), the path of the ``hsn`` package it imported
   and the versions;
2. re-validates the stored resolved configuration with its own code and recomputes the arm id
   (configuration, data manifest, family, variant, protocol, early-stop rules); a mismatch or a key
   this checkout does not know fails the run (the database is shared between worktrees);
3. runs the protocol (:mod:`hsn.experiments.protocols`) with progress at every evaluation;
4. evaluates the sweep's verdict rules on the record, writes the full JSON (every probe included)
   to ``<run dir>/result.json`` and, in one transaction, the results row and the final status
   (``done`` or ``stopped_early``).

An exception marks the run ``failed`` with the traceback's tail; SIGTERM (``cancel``) marks it
``cancelled``. **P1.14:** either way the child first keeps what exists: the protocol's partial
record (evaluations, diagnostics, every completed step's train record, the early-stop log, a
``failure`` block with the reason and the last completed step; :func:`hsn.experiments.protocols.
failure_info`) goes to ``result.json`` and, in the same guarded transaction as the final status, to
the results row, with every applicable verdict ``failed: <reason> @ <step>`` (never a pass). A
failure before the protocol starts keeps a minimal record. SIGTERM is ignored during that last
write. When a guarded write finds the row no longer running for this PID (recovery marked it
failed), the child logs it and stops without writing anything else. Card snapshots (nvidia-smi at
the start and the end) are taken here; the continuous card watch is the worker's.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

from hsn.config import REPO_ROOT, RunConfig, dump_resolved
from hsn.experiments import db as dbm
from hsn.experiments import machine
from hsn.experiments.db import Database, dumps, loads
from hsn.experiments.protocols import ArmRun, failure_info, run_protocol
from hsn.experiments.rules import RuleSet, evaluate_verdicts, parse_rules
from hsn.experiments.spec import arm_identity, data_hash, protocol_params

__all__ = ["main", "run_dir_for"]


class Cancelled(BaseException):
    """SIGTERM from ``cancel``."""


def _on_sigterm(signum: int, frame: FrameType | None) -> None:
    raise Cancelled()


@dataclass
class _Attempt:
    """What :func:`_run` built, kept for the failure path (P1.14)."""

    arm_run: ArmRun | None = None
    verdict_rules: RuleSet = field(default_factory=RuleSet)


def run_dir_for(root: Path, sweep: str | None, run_id: str, exp_id: int) -> Path:
    return Path(root) / (sweep or "_unswept") / run_id / f"exp{exp_id}"


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def _snapshot() -> dict[str, Any]:
    reading = machine.read_gpu()
    return {
        "time": reading.time,
        "present": reading.present,
        "ok": reading.ok,
        "error": reading.error,
        "used_mib": reading.used_mib,
        "processes": [p.to_json() for p in reading.processes],
    }


def _compact(record: dict[str, Any]) -> dict[str, Any]:
    """The record the database keeps: everything but the per-probe lists (v1's and v2's; P1.15:
    v3's and its per-document originals). ``result.json`` keeps them all."""
    lists = ("probes", "probes_v2", "probes_v3", "probes_v3_originals")
    out = {k: v for k, v in record.items() if k not in lists}
    if isinstance(out.get("at_stop"), dict):
        out["at_stop"] = {k: v for k, v in out["at_stop"].items() if k not in lists}
    return out


def _audit_fields(
    record: dict[str, Any], arm: dict[str, Any], commit: dict[str, Any]
) -> dict[str, Any]:
    """The top-level fields of the audit's result JSONs, for continuity of the evidence files."""
    out: dict[str, Any] = {
        "config": arm["config_path"],
        "overrides": loads(arm["overrides_json"]),
        "commit": {"sha": commit.get("sha"), "modified_tracked_files": commit.get("dirty")},
        "train_wall_s": round(float(record.get("walls", {}).get("fit_s", 0.0)), 1),
        "final_train_loss": (record.get("final") or {}).get("train_loss"),
    }
    probe = record.get("probe")
    if isinstance(probe, dict):
        for key in (
            "n_probed",
            "first_query_accuracy",
            "all",
            "correct_before_edit",
            "verdict",
            "first_query_answer_histogram",
            "distinct_first_query_answers",
        ):
            out[key] = probe.get(key)
    return out


def main(db_path: str, exp_id: int, root: str) -> int:
    db = Database(db_path)
    pid = os.getpid()
    exp = db.experiment(exp_id)
    arm = db.arm(exp["arm_id"])
    sweep = (
        db.row("SELECT * FROM sweeps WHERE id = ?", (exp["sweep_id"],))
        if exp["sweep_id"] is not None
        else None
    )
    member = (
        db.row(
            "SELECT constants_json FROM sweep_arms WHERE sweep_id = ? AND arm_id = ?",
            (exp["sweep_id"], exp["arm_id"]),
        )
        if sweep is not None
        else None
    )
    constants: dict[str, Any] = loads(member["constants_json"]) if member else {}
    git = machine.git_state(REPO_ROOT)
    import hsn

    started = dbm.child_start(
        db,
        exp_id,
        pid,
        host=machine.host(),
        commit_sha=git["sha"],
        dirty_json=dumps(git["dirty"]),
        hsn_path=str(Path(hsn.__file__).resolve().parent),
        versions_json=dumps(machine.versions()),
    )
    if not started:
        _log(f"experiment {exp_id} is not running for pid {pid}; not starting")
        return 3
    run_dir = Path(
        exp["run_dir"]
        or run_dir_for(Path(root), sweep["name"] if sweep else None, exp["run_id"], exp_id)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if git["dirty"]:
        (run_dir / "dirty.patch").write_text(git["diff"], encoding="utf-8")
        _log(
            f"WARNING: the checkout has modified tracked files {git['dirty']}; diff in dirty.patch"
        )
    signal.signal(signal.SIGTERM, _on_sigterm)
    _log(f"experiment {exp_id} ({exp['label']}, arm {exp['arm_id']}) pid {pid} commit {git['sha']}")
    attempt = _Attempt()
    try:
        record, verdicts, json_path = _run(
            db, exp, arm, sweep, constants, run_dir, pid, git, attempt
        )
    except BaseException as exc:  # noqa: BLE001 - every failure is recorded, then re-raised
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # the last write is short: let it finish
        cancelled = isinstance(exc, Cancelled)
        tail = "".join(traceback.format_exception(exc))[-4000:]
        if cancelled:
            _log(f"experiment {exp_id} cancelled (SIGTERM)")
        else:
            _log(f"experiment {exp_id} failed:\n{tail}")
        result = _failed_result(exc, attempt, exp, arm, constants, run_dir, git)
        ok = dbm.child_finish(
            db,
            exp_id,
            pid,
            status="cancelled" if cancelled else "failed",
            error="cancelled (SIGTERM)" if cancelled else tail,
            result=result,
        )
        if not ok:
            _log(f"experiment {exp_id} is no longer running for pid {pid}; record left on disk")
        if isinstance(exc, KeyboardInterrupt):
            raise
        return 143 if cancelled else 1
    status = "stopped_early" if record.get("stopped") else "done"
    final = record.get("final") or {}
    walls = record.get("walls") or {}
    ok = dbm.child_finish(
        db,
        exp_id,
        pid,
        status=status,
        result={
            "record": _compact(record),
            "verdicts": verdicts,
            "early_stop": record.get("stopped"),
            "json_path": str(json_path),
            "final_dev_accuracy": final.get("dev_accuracy"),
            "final_dev_loss": final.get("dev_loss"),
            "train_s": walls.get("train_s"),
            "total_s": walls.get("total_s"),
        },
    )
    if not ok:
        _log(
            f"experiment {exp_id} is no longer running for pid {pid} (recovered by a worker); "
            f"result left in {json_path}"
        )
        return 3
    labels = {k: v["label"] for k, v in verdicts.items()}
    _log(f"experiment {exp_id} {status}: {json.dumps(labels)}")
    return 0


def _failed_result(
    exc: BaseException,
    attempt: _Attempt,
    exp: dict[str, Any],
    arm: dict[str, Any],
    constants: dict[str, Any],
    run_dir: Path,
    git: dict[str, Any],
) -> dict[str, Any] | None:
    """P1.14: the partial record of a failed or cancelled attempt, written to ``result.json``, and
    the results row for it (``None`` when even that cannot be built: the status is written anyway,
    as before P1.14)."""
    try:
        record: dict[str, Any] = attempt.arm_run.record if attempt.arm_run is not None else {}
        record.setdefault("protocol", arm["protocol"])
        record.setdefault("family", arm["family"])
        if "failure" not in record:  # the failure came before or after the protocol
            last = int((record.get("final") or {}).get("steps") or 0)
            stage = attempt.arm_run.stage if attempt.arm_run is not None else "setup"
            record["failure"] = failure_info(exc, stage=stage, last_step=last)
        try:
            record.setdefault("card_snapshots", {})["end"] = _snapshot()
        except Exception:  # noqa: BLE001 - a snapshot is never worth losing the record
            pass
        record["arm"] = {
            "arm_id": arm["arm_id"],
            "group_id": arm["group_id"],
            "label": exp["label"],
            "exp_id": exp["exp_id"],
            "run_id": exp["run_id"],
            "seed": arm["seed"],
            "constants": constants,
        }
        verdicts = evaluate_verdicts(
            attempt.verdict_rules, record, constants, arm["family"], arm["protocol"]
        )
        full = {**_audit_fields(record, arm, git), **record, "verdicts": verdicts}
        json_path = run_dir / "result.json"
        json_path.write_text(json.dumps(full, indent=1, default=str), encoding="utf-8")
        walls = record.get("walls") or {}
        return {
            "record": _compact(record),
            "verdicts": verdicts,
            "early_stop": record.get("stopped"),
            "json_path": str(json_path),
            "final_dev_accuracy": None,  # not a final result: the last evaluation is in the record
            "final_dev_loss": None,
            "train_s": walls.get("train_s"),
            "total_s": walls.get("total_s"),
        }
    except Exception as err:  # noqa: BLE001 - the final status must still be written
        _log(f"could not keep the partial record: {type(err).__name__}: {err}")
        return None


def _run(
    db: Database,
    exp: dict[str, Any],
    arm: dict[str, Any],
    sweep: dict[str, Any] | None,
    constants: dict[str, Any],
    run_dir: Path,
    pid: int,
    git: dict[str, Any],
    attempt: _Attempt,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    # the attempt's own configuration and protocol (schema v2); older rows fall back to the arm's
    cfg = RunConfig.model_validate(loads(exp.get("config_json") or arm["config_json"]))
    protocol = loads(exp.get("protocol_json") or arm["protocol_json"])
    params = protocol_params(protocol)
    variant_params = loads(arm["variant_json"])
    early_stop = loads(arm["early_stop_json"])
    data = data_hash(cfg, REPO_ROOT, required=params.name != "dry_run")
    arm_id, _group = arm_identity(
        cfg,
        family=arm["family"],
        variant=arm["variant"],
        variant_params=variant_params,
        protocol=protocol,
        early_stop=early_stop,
        data=data,
    )
    if arm_id != arm["arm_id"]:
        raise RuntimeError(
            f"arm id mismatch: this checkout computes {arm_id} for arm {arm['arm_id']} (a changed "
            f"configuration schema or data: stored data hash {arm['data_hash']}, now {data}); "
            "enqueue the spec again from this checkout"
        )
    stop_rules = parse_rules({"early_stop": early_stop})
    verdict_rules = RuleSet()
    if sweep is not None:
        stored = loads(sweep["rules_json"]) or {}
        verdict_rules = parse_rules({"verdicts": stored.get("verdicts") or {}})
    attempt.verdict_rules = verdict_rules
    dump_resolved(cfg, run_dir / "config.resolved.yaml")
    start_card = _snapshot()

    def progress(p: dict[str, Any]) -> None:
        if not dbm.child_progress(db, exp["exp_id"], pid, p):
            raise RuntimeError(
                "the experiment row is no longer running for this process (recovered)"
            )

    attempt.arm_run = ArmRun(
        cfg=cfg,
        family=arm["family"],
        variant=arm["variant"],
        variant_params=variant_params,
        params=params,
        run_dir=run_dir,
        rules=stop_rules,
        constants=constants,
        on_progress=progress,
    )
    attempt.arm_run.record["card_snapshots"] = {"start": start_card}
    record = run_protocol(attempt.arm_run)
    record["card_snapshots"] = {"start": start_card, "end": _snapshot()}
    record["arm"] = {
        "arm_id": arm["arm_id"],
        "group_id": arm["group_id"],
        "label": exp["label"],
        "exp_id": exp["exp_id"],
        "run_id": exp["run_id"],
        "seed": arm["seed"],
        "constants": constants,
    }
    verdicts = evaluate_verdicts(verdict_rules, record, constants, arm["family"], arm["protocol"])
    full = {**_audit_fields(record, arm, git), **record, "verdicts": verdicts}
    json_path = run_dir / "result.json"
    json_path.write_text(json.dumps(full, indent=1, default=str), encoding="utf-8")
    return record, verdicts, json_path


def cli(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="hsn.experiments run-arm")
    ap.add_argument("--db", required=True)
    ap.add_argument("--exp", type=int, required=True)
    ap.add_argument("--root", required=True)
    args = ap.parse_args(argv)
    return main(args.db, args.exp, args.root)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli(sys.argv[1:]))
