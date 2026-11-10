"""The sweep command line (P1.11): ``python -m hsn.experiments <command>``, or
``hsn sweep <command>``.

::

    expand   <spec>                      arms, labels, ids (no database)
    enqueue  <spec> [--force] [--skip-imported] [--only GLOB] [--mem-estimate-mib N]
    worker   [--max-concurrent 3] [--reserve-mib 1024] [--requeue-orphans] [--require-clean]
             [--exit-when-empty] [--poll 5] [--idle-gap 60] [--omp-threads 4] [--inductor-cache DIR]
    status   [--sweep NAME] [--recent 10]
    cancel   EXP... | --arm ID | --sweep NAME      (queued: cancelled; running: SIGTERM)
    requeue  EXP... | --arm ID | --sweep NAME [--status failed,cancelled,stopped_early]
    pause | resume
    report   [--sweep NAME | --spec FILE] [--status ...] [--family F] [--protocol P]
             [--imported | --no-imported] [--per-seed] [--format md|csv] [--out FILE]
    import   --audit-dir DIR [--dry-run]
    probe-check --exp N [--batch-rows 64]
    probe-rerun --exp N [--cpu] [--batch-rows N] [--weights checkpoint|dump] [--key-alphabet train]
             [--out FILE]   v1 vs the stored probes, then v2, v3 and the split (P1.16: read-only)
    split-backfill [--exp N ...] [--recompute] [--cpu] --out FILE   (P1.15; read-only database)
    replay   --dump FILE | --exp N [--eager | --compiled] [--no-deterministic] [--repeats 2]
             [--cpu] [--out FILE]      (P1.15: the divergence dump's step again)
             [--continue-steps N [--spike-factor F] [--rollback-every N] [--rollback-skip N]
             [--rollback-lr-mult F] [--rollback-lr-steps N] [--rollback-max N]]   (P1.16)
    route-check --exp N [--split dev|train] [--n-docs 1000] [--cpu] [--batch-rows 64] [--out FILE]
             (P1.16: does routing follow the key; read-only)
    controller-report --exp N [--exp M ...] [--out FILE]  (read-only signals and matched ablations)
    run-arm  --db DB --exp N --root ROOT       (the worker's child; not for direct use)

Every command takes ``--db`` (default ``$HSN_EXPERIMENTS_DB`` or
``~/hsn-experiments/experiments.sqlite``) and ``--root`` for run outputs (default
``$HSN_EXPERIMENTS_ROOT`` or ``<db dir>/runs``). Usage: ``docs/sweeps.md``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hsn.experiments import db as dbm
from hsn.experiments.db import Database, default_db_path, default_root, loads

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hsn sweep", description="P1.11 experiment sweeps")
    ap.add_argument(
        "--db",
        default=None,
        help="database file (default $HSN_EXPERIMENTS_DB or ~/hsn-experiments/experiments.sqlite)",
    )
    ap.add_argument(
        "--root", default=None, help="run outputs (default $HSN_EXPERIMENTS_ROOT or <db dir>/runs)"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("expand", help="print a spec's arms and ids")
    p.add_argument("spec")

    p = sub.add_parser("enqueue", help="expand a spec and queue its arms")
    p.add_argument("spec")
    p.add_argument(
        "--force", action="store_true", help="queue arms that are already queued or done"
    )
    p.add_argument("--skip-imported", action="store_true", help="skip arms with an imported run")
    p.add_argument("--only", default=None, help="only labels matching this glob")
    p.add_argument("--mem-estimate-mib", type=float, default=None, help="GPU memory reservation")

    p = sub.add_parser("worker", help="run the queue")
    p.add_argument("--max-concurrent", type=int, default=3)
    p.add_argument("--reserve-mib", type=float, default=1024.0)
    p.add_argument("--requeue-orphans", action="store_true")
    p.add_argument("--require-clean", action="store_true")
    p.add_argument("--exit-when-empty", action="store_true")
    p.add_argument("--poll", type=float, default=5.0)
    p.add_argument("--telemetry-max-samples", type=int, default=0)
    p.add_argument("--idle-gap", type=float, default=60.0)
    p.add_argument("--omp-threads", type=int, default=4)
    p.add_argument("--inductor-cache", default=None, help="compiled-artifact cache of the arms")
    p.add_argument("--turn-file", default=None, help="require an explicitly claimed compute turn")
    p.add_argument("--turn-owner", default=None, help="owner name in --turn-file")

    p = sub.add_parser("status", help="queue, running and recent results")
    p.add_argument("--sweep", default=None)
    p.add_argument("--recent", type=int, default=10)

    p = sub.add_parser(
        "controller-report", help="controller signals and matched ablations (read-only)"
    )
    p.add_argument("--exp", type=int, action="append", required=True)
    p.add_argument("--out", default=None)

    for name in ("cancel", "requeue"):
        p = sub.add_parser(name)
        p.add_argument("exp", nargs="*", type=int)
        p.add_argument("--arm", default=None)
        p.add_argument("--sweep", default=None)
        if name == "requeue":
            p.add_argument("--status", default="failed,cancelled")

    sub.add_parser("pause", help="no new launches")
    sub.add_parser("resume", help="launches again")

    p = sub.add_parser("report", help="summary from the database")
    p.add_argument("--sweep", default=None)
    p.add_argument("--spec", default=None)
    p.add_argument("--status", default=None, help="comma-separated statuses")
    p.add_argument("--family", default=None)
    p.add_argument("--protocol", default=None)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--imported", dest="imported", action="store_true", default=None)
    group.add_argument("--no-imported", dest="imported", action="store_false")
    p.add_argument("--per-seed", action="store_true")
    p.add_argument("--format", choices=("md", "csv"), default="md")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--split-backfill",
        default=None,
        help="a split-backfill JSON: the first/later split of runs stored before P1.15",
    )

    p = sub.add_parser(
        "split-backfill", help="the first/later split of stored reads_state runs (read-only)"
    )
    p.add_argument("--exp", type=int, nargs="*", default=None)
    p.add_argument("--recompute", action="store_true", help="evaluate kept checkpoints again")
    p.add_argument("--cpu", action="store_true", help="recompute on profile:cpu in fp32")
    p.add_argument("--out", required=True, help="JSON output (a markdown table goes next to it)")

    p = sub.add_parser("import", help="backfill the P1.10 audit's result JSONs")
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("probe-check", help="batched vs per-document edit probe on a checkpoint")
    p.add_argument("--exp", type=int, required=True)
    p.add_argument("--batch-rows", type=int, default=64)
    p.add_argument("--out", default=None)

    p = sub.add_parser("probe-rerun", help="v1 again vs the stored probes, then probe v2 (P1.14)")
    p.add_argument("--exp", type=int, required=True)
    p.add_argument(
        "--checkpoint-step", type=int, help="probe this retained step instead of the last"
    )
    p.add_argument("--cpu", action="store_true", help="profile:cpu in fp32, not the run's hardware")
    p.add_argument("--batch-rows", type=int, default=None, help="default: the run's probe_batch")
    p.add_argument(
        "--weights",
        choices=("checkpoint", "dump"),
        default="checkpoint",
        help="P1.16: the last checkpoint, or the divergence dump (a run that failed without one)",
    )
    p.add_argument(
        "--key-alphabet",
        choices=("train",),
        default=None,
        help="P1.16: dev keys remapped to training keys 64-79 (a diagnostic, not a metric)",
    )
    p.add_argument("--out", default=None)

    p = sub.add_parser("route-check", help="does routing follow the key (P1.16; read-only)")
    p.add_argument("--exp", type=int, required=True)
    p.add_argument(
        "--checkpoint-step", type=int, help="probe this retained step instead of the last"
    )
    p.add_argument("--split", choices=("dev", "train"), default="dev")
    p.add_argument("--n-docs", type=int, default=1000)
    p.add_argument("--cpu", action="store_true", help="profile:cpu in fp32, not the run's hardware")
    p.add_argument("--batch-rows", type=int, default=64)
    p.add_argument("--out", default=None)

    p = sub.add_parser("replay", help="replay a divergence dump's step (P1.15)")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--dump", default=None, help="a divergence_<step>.pt file")
    where.add_argument("--exp", type=int, default=None, help="the experiment's first dump")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--eager", dest="eager", action="store_true", default=None)
    mode.add_argument("--compiled", dest="eager", action="store_false")
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--cpu", action="store_true", help="profile:cpu in fp32 (approximate)")
    p.add_argument("--out", default=None)
    # P1.16: keep training past the dumped step, with a remedy set on the resumed trainer
    p.add_argument("--continue-steps", type=int, default=0)
    p.add_argument("--spike-factor", type=float, default=None)
    p.add_argument("--rollback-every", type=int, default=None)
    p.add_argument("--rollback-skip", type=int, default=0)
    p.add_argument("--rollback-lr-mult", type=float, default=1.0)
    p.add_argument("--rollback-lr-steps", type=int, default=0)
    p.add_argument("--rollback-max", type=int, default=5)

    p = sub.add_parser("run-arm", help=argparse.SUPPRESS)
    p.add_argument("--exp", type=int, required=True)
    return ap


def _db(args: argparse.Namespace) -> Database:
    return Database(Path(args.db) if args.db else default_db_path())


def _root(args: argparse.Namespace, db: Database) -> Path:
    root = Path(args.root) if args.root else default_root(db.path)
    dbm.check_not_tmp(root, allow_tmp=False)
    return root


def _table(rows: Sequence[Sequence[Any]], header: Sequence[str]) -> str:
    cells = [[str(h) for h in header], *[["" if c is None else str(c) for c in r] for r in rows]]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    return "\n".join(
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in cells
    )


def cmd_expand(args: argparse.Namespace) -> int:
    from hsn.experiments.spec import expand, load_spec

    spec, _ = load_spec(args.spec)
    arms = expand(spec)
    rows = [
        (
            a.label,
            a.arm_id,
            a.family,
            a.variant,
            a.protocol.name,
            a.steps,
            "timing" if a.timing else "",
            a.run_id,
        )
        for a in arms
    ]
    print(
        _table(
            rows, ["label", "arm id", "family", "variant", "protocol", "steps", "timing", "run id"]
        )
    )
    print(f"{len(arms)} arms")
    return 0


def cmd_enqueue(args: argparse.Namespace) -> int:
    from hsn.experiments.rules import parse_rules
    from hsn.experiments.spec import digest, expand, load_spec

    spec, text = load_spec(args.spec)
    arms = expand(spec)
    if args.only:
        arms = [a for a in arms if fnmatch.fnmatch(a.label, args.only)]
        if not arms:
            print(f"no arm label matches {args.only!r}", file=sys.stderr)
            return 2
    rules = parse_rules(spec.rules)
    db = _db(args)
    report = dbm.enqueue_arms(
        db,
        name=spec.name,
        spec_hash=digest(spec.model_dump(mode="json")),
        spec_text=text,
        spec_path=str(Path(args.spec).resolve()),
        rules_json=rules.to_json(),
        arms=arms,
        force=args.force,
        skip_imported=args.skip_imported,
        mem_est_mib=args.mem_estimate_mib,
    )
    if report.added:
        print("added:")
        print(
            _table(
                [
                    (
                        a["exp_id"],
                        a["label"],
                        a["arm_id"],
                        f"{a['step_s']:.3g}" if a.get("step_s") is not None else "unknown",
                    )
                    for a in report.added
                ],
                ["exp", "label", "arm id", "expected s/step"],
            )
        )
    if report.skipped:
        print("skipped:")
        print(
            _table(
                [(s["label"], s["arm_id"], s["status"], s["exp_id"]) for s in report.skipped],
                ["label", "arm id", "existing", "exp"],
            )
        )
    row = db.row(
        "SELECT COUNT(*) AS n FROM sweeps WHERE name = ? AND id <= ?", (spec.name, report.sweep_id)
    )
    version = int(row["n"]) if row else 1
    for m in report.imported_matches:
        print(
            f"note: {m['label']} matches imported exp {m['exp_id']} "
            "(not skipped; --skip-imported skips it)"
        )
    print(
        f"sweep {spec.name} (spec version {version}): {len(report.added)} added, "
        f"{len(report.skipped)} skipped"
    )
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    from hsn.experiments.worker import Worker, WorkerOptions

    db = _db(args)
    options = WorkerOptions(
        max_concurrent=args.max_concurrent,
        reserve_mib=args.reserve_mib,
        idle_gap_s=args.idle_gap,
        requeue_orphans=args.requeue_orphans,
        require_clean=args.require_clean,
        exit_when_empty=args.exit_when_empty,
        poll_s=args.poll,
        omp_threads=args.omp_threads,
        telemetry_max_samples=args.telemetry_max_samples,
        inductor_cache=Path(args.inductor_cache) if args.inductor_cache else None,
        turn_file=Path(args.turn_file) if args.turn_file else None,
        turn_owner=args.turn_owner,
    )
    return Worker(db, _root(args, db), options).run()


def cmd_status(args: argparse.Namespace) -> int:
    db = _db(args)
    where = ""
    params: list[Any] = []
    if args.sweep:
        where = " AND e.sweep_id IN (SELECT id FROM sweeps WHERE name = ?)"
        params.append(args.sweep)
    with db.read() as conn:
        queued = conn.execute(
            "SELECT e.exp_id, e.label, e.timing, e.mem_est_mib, e.enqueued_at FROM experiments e "
            f"WHERE e.status = 'queued'{where} ORDER BY e.exp_id",
            params,
        ).fetchall()
        running = conn.execute(
            "SELECT e.exp_id, e.label, e.pid, e.worker, e.started_at, e.claimed_at, "
            "e.progress_json FROM experiments e "
            f"WHERE e.status = 'running'{where} ORDER BY e.exp_id",
            params,
        ).fetchall()
        recent = conn.execute(
            "SELECT e.exp_id, e.label, e.status, e.ended_at, e.card_status, r.verdicts_json, "
            "r.final_dev_accuracy, r.total_s, e.error, "
            "json_extract(r.record_json, '$.failure.reason') AS failure_reason, "
            "json_extract(r.record_json, '$.failure.last_step') AS failure_step, "
            # P1.15: the final evaluation's first/later split (absent before P1.15)
            "json_extract(r.record_json, '$.final.\"split/first/accuracy\"') AS first_acc, "
            "json_extract(r.record_json, '$.final.\"split/later/accuracy\"') AS later_acc "
            "FROM experiments e LEFT JOIN results r ON r.exp_id = e.exp_id "
            "WHERE e.status IN ('done', 'failed', 'stopped_early', 'cancelled') "
            f"AND e.imported = 0{where} ORDER BY e.ended_at DESC LIMIT ?",
            [*params, args.recent],
        ).fetchall()
    print(f"database {db.path}{' (PAUSED)' if db.paused else ''}")
    print(f"\nqueued ({len(queued)}):")
    print(
        _table(
            [
                (
                    r["exp_id"],
                    r["label"],
                    "timing" if r["timing"] else "",
                    r["mem_est_mib"],
                    r["enqueued_at"][:19],
                )
                for r in queued
            ],
            ["exp", "label", "timing", "mem est MiB", "enqueued"],
        )
    )
    print(f"\nrunning ({len(running)}):")
    rows = []
    controller_rows = []
    growth_rows = []
    for r in running:
        progress = loads(r["progress_json"]) or {}
        growth = progress.get("gradient_growth")
        if growth is not None:
            observation = growth.get("latest") or {}
            for name, scale in growth["scales"].items():
                group = observation.get("groups", {}).get(name, {})
                growth_rows.append(
                    (
                        r["exp_id"],
                        observation.get("step", ""),
                        name,
                        group.get("action", "warming up"),
                        group.get("recent_median", ""),
                        group.get("share", ""),
                        group.get("growth", ""),
                        scale,
                        growth["damped_steps"].get(name, 0),
                    )
                )
        controller = progress.get("adaptive_lr")
        if controller is not None:
            from hsn.experiments.controller import format_rates

            latest = controller.get("latest_observation") or {}
            change = controller.get("last_adjustment") or {}
            rates = controller.get("next_update_lrs") or {}
            controller_rows.append(
                (
                    r["exp_id"],
                    latest.get("step", ""),
                    latest.get("decision", "initializing"),
                    f"{change['decision']} @ {change['step']}" if change else "none",
                    latest.get("loss", ""),
                    latest.get("grad_median", ""),
                    format_rates(rates.get("shared", [])),
                    format_rates(rates.get("router", [])),
                    format_rates(rates.get("decoder", [])),
                    controller.get("cooldown_windows", ""),
                    "pending"
                    if controller.get("trial")
                    else ("blocked" if controller.get("trial_blocked") else "none"),
                )
            )
        rows.append(
            (
                r["exp_id"],
                r["label"],
                r["pid"],
                r["worker"],
                (r["started_at"] or r["claimed_at"] or "")[:19],
                progress.get("stage") or f"{progress.get('step', '')}/{progress.get('of', '')}",
                progress.get("dev_accuracy", ""),
                progress.get("dev_step", progress.get("step", ""))
                if "dev_accuracy" in progress
                else "",
            )
        )
    print(
        _table(rows, ["exp", "label", "pid", "worker", "started", "step", "dev acc", "dev @ step"])
    )
    if controller_rows:
        print("\ncontroller (training monitor; learning rates are for the next update):")
        print(
            _table(
                controller_rows,
                [
                    "exp",
                    "observed step",
                    "decision",
                    "last rate change",
                    "train monitor loss",
                    "grad median",
                    "shared lr",
                    "router lr",
                    "readout lr",
                    "cooldown",
                    "trial",
                ],
            )
        )
    if growth_rows:
        print("\ngradient growth (pre-clip training norms; update scales for selected groups):")
        print(
            _table(
                growth_rows,
                [
                    "exp",
                    "observed step",
                    "group",
                    "action",
                    "grad median",
                    "norm share",
                    "growth",
                    "update scale",
                    "damped steps",
                ],
            )
        )
    print(f"\nrecent ({len(recent)}):")
    recent_rows: list[tuple[Any, ...]] = []
    for r in recent:
        verdicts = loads(r["verdicts_json"]) or {}
        labels = "; ".join(
            f"{k}: {v['label'] if isinstance(v, dict) else v}" for k, v in verdicts.items()
        )
        err = (r["error"] or "").strip().splitlines()[-1:] if r["status"] == "failed" else []
        if r["failure_reason"] is not None:  # P1.14: the partial record's failure, once
            labels = f"{r['status']} @ {r['failure_step']}: {r['failure_reason']}"
        acc = f"{r['final_dev_accuracy']:.4f}" if r["final_dev_accuracy"] is not None else ""
        split = (
            f"{r['first_acc']:.3f}/{r['later_acc']:.3f}"
            if r["first_acc"] is not None and r["later_acc"] is not None
            else ""
        )
        recent_rows.append(
            (
                r["exp_id"],
                r["label"],
                r["status"],
                (r["ended_at"] or "")[:19],
                r["card_status"] or "",
                acc,
                split,
                labels or (err[0] if err else ""),
            )
        )
    print(
        _table(
            recent_rows,
            [
                "exp",
                "label",
                "status",
                "ended",
                "card",
                "dev acc",
                "first/later",
                "verdicts / error",
            ],
        )
    )
    return 0


def _select_ids(db: Database, args: argparse.Namespace, statuses: Sequence[str]) -> list[int]:
    ids = list(args.exp)
    marks = ",".join("?" * len(statuses))
    if args.arm:
        sql = f"SELECT exp_id FROM experiments WHERE arm_id = ? AND status IN ({marks})"
        ids += [r["exp_id"] for r in db.rows(sql, (args.arm, *statuses))]
    if args.sweep:
        sql = (
            "SELECT e.exp_id FROM experiments e JOIN sweeps s ON s.id = e.sweep_id "
            f"WHERE s.name = ? AND e.status IN ({marks})"
        )
        ids += [r["exp_id"] for r in db.rows(sql, (args.sweep, *statuses))]
    return sorted(set(ids))


def cmd_cancel(args: argparse.Namespace) -> int:
    from hsn.experiments.worker import kill_experiment

    db = _db(args)
    ids = _select_ids(db, args, ("queued", "running"))
    cancelled = dbm.cancel_queued(db, ids)
    signalled = [i for i in ids if i not in cancelled and kill_experiment(db, i)]
    print(f"cancelled (queued): {cancelled}; SIGTERM sent (running): {signalled}")
    other = [i for i in ids if i not in cancelled and i not in signalled]
    if other:
        print(f"not cancelled (not queued, or not a run-arm process of ours): {other}")
    return 0


def cmd_requeue(args: argparse.Namespace) -> int:
    db = _db(args)
    statuses = [s.strip() for s in args.status.split(",") if s.strip()]
    ids = _select_ids(db, args, statuses)
    new = dbm.requeue(db, ids, statuses)
    print(f"requeued {len(new)}: new experiments {new}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from hsn.experiments.report import build_report, format_csv, format_markdown

    db = _db(args)
    statuses = [s.strip() for s in args.status.split(",")] if args.status else None
    backfill = None
    if args.split_backfill:  # P1.15
        data = json.loads(Path(args.split_backfill).read_text(encoding="utf-8"))
        backfill = {int(r["exp_id"]): r for r in data["runs"] if "tier" in r}
    report = build_report(
        db,
        sweep=args.sweep,
        spec=args.spec,
        statuses=statuses,
        family=args.family,
        protocol=args.protocol,
        imported=args.imported,
        per_seed=args.per_seed,
        split_backfill=backfill,
    )
    text = format_csv(report) if args.format == "csv" else format_markdown(report)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(report.rows)} rows)")
    else:
        print(text, end="")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from hsn.experiments.importer import import_audit_dir

    db = _db(args)
    report = import_audit_dir(db, args.audit_dir, dry_run=args.dry_run)
    for item in report.flagged:
        print(f"flagged {item['name']}: " + "; ".join(item["flags"]))
    return 0


def cmd_probe_check(args: argparse.Namespace) -> int:
    from hsn.experiments.probe_check import probe_check

    db = _db(args)
    out = probe_check(db, args.exp, batch_rows=args.batch_rows)
    text = json.dumps(out, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "flips"}, indent=1, default=str))
    return 0


def cmd_probe_rerun(args: argparse.Namespace) -> int:
    from hsn.experiments.probe_check import probe_rerun

    db = Database(Path(args.db) if args.db else default_db_path(), read_only=True)  # P1.16
    out = probe_rerun(
        db,
        args.exp,
        cpu=args.cpu,
        batch_rows=args.batch_rows,
        weights=args.weights,
        key_alphabet=args.key_alphabet,
        checkpoint_step=args.checkpoint_step,
    )
    text = json.dumps(out, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    shown = {k: v for k, v in out.items() if k not in ("differ", "v2_differ")}
    print(json.dumps(shown, indent=1, default=str))
    return 0


def cmd_split_backfill(args: argparse.Namespace) -> int:
    """P1.15: the split of stored runs, from a read-only database (hsn.experiments.split)."""
    from hsn.experiments.split import backfill_split, backfill_table

    db = Database(Path(args.db) if args.db else default_db_path(), read_only=True)
    out = backfill_split(
        db, args.exp, recompute=args.recompute, cpu=args.cpu, log=lambda line: print(line)
    )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    table = path.with_suffix(".md")
    table.write_text(backfill_table(out), encoding="utf-8")
    done = [r for r in out["runs"] if "tier" in r]
    print(f"wrote {path} and {table}: {len(done)} runs with a split, {len(out['runs'])} read")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """P1.15: replay a divergence dump's step (hsn.train.divergence.replay)."""
    from hsn.train.divergence import dump_paths, replay

    if args.dump:
        path = Path(args.dump)
    else:
        db = Database(Path(args.db) if args.db else default_db_path(), read_only=True)
        exp = db.experiment(args.exp)
        result = db.row("SELECT json_path FROM results WHERE exp_id = ?", (args.exp,))
        run_dir = exp["run_dir"] or (
            str(Path(result["json_path"]).parent) if result and result["json_path"] else None
        )
        found = dump_paths(run_dir) if run_dir else []
        if not found:
            print(f"error: experiment {args.exp} has no divergence dump", file=sys.stderr)
            return 2
        path = found[0]
    remedy: dict[str, Any] = {}  # P1.16
    if args.spike_factor is not None:
        remedy["spike_factor"] = args.spike_factor
    if args.rollback_every is not None:
        remedy["rollback"] = {
            "every": args.rollback_every,
            "skip": args.rollback_skip,
            "lr_mult": args.rollback_lr_mult,
            "lr_steps": args.rollback_lr_steps,
            "max_rollbacks": args.rollback_max,
        }
    if remedy and not args.continue_steps:
        print("error: a remedy needs --continue-steps", file=sys.stderr)
        return 2
    out = replay(
        path,
        eager=args.eager,
        deterministic=args.deterministic,
        repeats=args.repeats,
        cpu=args.cpu,
        continue_steps=args.continue_steps,
        remedy=remedy or None,
    )
    text = json.dumps(out, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def cmd_route_check(args: argparse.Namespace) -> int:
    """P1.16: does routing follow the key (hsn.experiments.routing; read-only database)."""
    from hsn.experiments.routing import route_check

    db = Database(Path(args.db) if args.db else default_db_path(), read_only=True)
    out = route_check(
        db,
        args.exp,
        split=args.split,
        n_docs=args.n_docs,
        cpu=args.cpu,
        batch_rows=args.batch_rows,
        checkpoint_step=args.checkpoint_step,
    )
    text = json.dumps(out, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def cmd_controller_report(args: argparse.Namespace) -> int:
    from hsn.experiments.controller import controller_report

    db = Database(Path(args.db) if args.db else default_db_path(), read_only=True)
    report = controller_report(db, args.exp)
    body = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
    print(body, end="")
    return 0


def cmd_run_arm(args: argparse.Namespace) -> int:
    from hsn.experiments.run_arm import main as run_arm

    db_path = Path(args.db) if args.db else default_db_path()
    root = Path(args.root) if args.root else default_root(db_path)
    return run_arm(str(db_path), args.exp, str(root))


COMMANDS = {
    "expand": cmd_expand,
    "enqueue": cmd_enqueue,
    "worker": cmd_worker,
    "status": cmd_status,
    "controller-report": cmd_controller_report,
    "cancel": cmd_cancel,
    "requeue": cmd_requeue,
    "report": cmd_report,
    "import": cmd_import,
    "probe-check": cmd_probe_check,
    "probe-rerun": cmd_probe_rerun,
    "split-backfill": cmd_split_backfill,
    "replay": cmd_replay,
    "route-check": cmd_route_check,
    "run-arm": cmd_run_arm,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command in ("pause", "resume"):
        db = _db(args)
        db.set_paused(args.command == "pause")
        print(f"{db.path}: {'paused' if db.paused else 'running'} ({time.strftime('%H:%M:%S')})")
        return 0
    from hsn.experiments.rules import RuleError
    from hsn.experiments.spec import SpecError
    from hsn.experiments.steptime import StepTimeRefused

    try:
        return COMMANDS[args.command](args)
    except (SpecError, RuleError, StepTimeRefused, dbm.DatabaseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
