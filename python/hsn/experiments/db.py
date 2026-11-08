"""The experiment database (P1.11): one SQLite file that records every experiment, past, present
and future, and is the FIFO queue. Stdlib ``sqlite3``; imports without torch.

**Where.** ``--db``, else ``$HSN_EXPERIMENTS_DB``, else ``~/hsn-experiments/experiments.sqlite``: a
path outside every worktree, so that every session and every reboot sees the same file. Run outputs
go under ``$HSN_EXPERIMENTS_ROOT`` (default ``<db dir>/runs``). A path under the temporary directory
(``/tmp`` is wiped on reboot here) raises unless ``allow_tmp`` (tests) or
``HSN_EXPERIMENTS_ALLOW_TMP=1``.

**Concurrency.** WAL journal, ``busy_timeout`` 30 s, one connection per thread. Every write runs in
``with db.write():`` (``BEGIN IMMEDIATE``), which serialises writers (two or three workers, their
children, ``enqueue``) without lost updates; multi-statement reads use ``with db.read():`` for one
snapshot (a reporter never sees half a transaction).

**Schema** (version in ``PRAGMA user_version``; :data:`MIGRATIONS` applies in order, one transaction
per step, inside ``BEGIN IMMEDIATE`` so that concurrent first opens migrate once; a database newer
than the code raises):

* ``sweeps``: one row per (name, spec hash): the spec text and the expanded rules at enqueue time,
  the pre-registration record;
* ``arms``: one row per arm id (resolved config, family, protocol, variant, seed, data hash,
  early-stop rules, group id, memory key);
* ``sweep_arms``: membership, with the arm's label and constants in that sweep;
* ``experiments``: one row per attempt (a requeue adds a row, nothing is overwritten); ``exp_id``
  order is the FIFO order; status ``queued``, ``running``, ``done``, ``failed``,
  ``stopped_early`` or ``cancelled``; times, worker, host, PID, process group, commit and dirty
  files, card status, peak GPU memory, exit code, error, run directory, progress; imported rows are
  marked;
* ``results``: one row per finished experiment: the compact record (no per-probe list), the
  verdicts, the early-stop record, the path of the full JSON and a few scalars. P1.14: also one row
  per attempt that failed or was cancelled in its child, holding the partial record with its
  ``failure`` block (``final_dev_accuracy`` NULL: not a final result);
* ``meta``: ``paused`` and the like;
* v2 adds ``experiments.protocol_json`` and ``experiments.config_json``: each attempt's own protocol
  (execution parameters included) and resolved configuration (its own ``run_id``), which the child
  runs; rows from before v2 fall back to the arm's;
* the view ``queue``: queued experiments in FIFO order.

**Read-only (P1.15).** ``Database(path, read_only=True)`` opens an existing file through SQLite's
``mode=ro`` URI: no directory is created, nothing is migrated (a schema other than this code's
raises), and :meth:`Database.write` raises. The read-only commands (``split-backfill``, ``replay
--exp``) open the database this way, so that they can run beside a live worker.

**Guarded writes.** A child process changes its experiment only while that row is ``running`` with
its PID (:func:`child_start`, :func:`child_progress`, :func:`child_finish`); when recovery has
marked the row failed in between, the write changes nothing and returns False, and the child stops.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "BLOCKING",
    "FINISHED",
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "STATUSES",
    "Database",
    "DatabaseError",
    "EnqueueReport",
    "child_finish",
    "child_progress",
    "child_start",
    "claim_next",
    "default_db_path",
    "default_root",
    "enqueue_arms",
    "now",
]

STATUSES = ("queued", "running", "done", "failed", "stopped_early", "cancelled")
FINISHED = ("done", "stopped_early")
BLOCKING = ("queued", "running", "done", "stopped_early")  # enqueue skips an arm with one of these
DEFAULT_DB = Path("~/hsn-experiments/experiments.sqlite")


class DatabaseError(RuntimeError):
    """A database the code cannot use (a newer schema, a temporary path, a guarded write lost)."""


def now() -> str:
    """UTC time, ISO 8601 with microseconds (sortable)."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")


def parse_time(text: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(text) if text else None


def default_db_path() -> Path:
    env = os.environ.get("HSN_EXPERIMENTS_DB")
    return Path(env).expanduser() if env else DEFAULT_DB.expanduser()


def default_root(db_path: Path) -> Path:
    env = os.environ.get("HSN_EXPERIMENTS_ROOT")
    return Path(env).expanduser() if env else Path(db_path).expanduser().parent / "runs"


def _allow_tmp_env() -> bool:
    return os.environ.get("HSN_EXPERIMENTS_ALLOW_TMP", "") == "1"


def check_not_tmp(path: Path, *, allow_tmp: bool) -> None:
    """Refuse a path under the temporary directory (tmpfs here: wiped on reboot)."""
    if allow_tmp or _allow_tmp_env():
        return
    resolved = Path(path).expanduser().resolve()
    for tmp in {Path("/tmp"), Path(tempfile.gettempdir())}:
        base = tmp.resolve()
        if resolved == base or base in resolved.parents:
            raise DatabaseError(
                f"{path} is under the temporary directory {base}, which does not survive a reboot; "
                "use a path such as ~/hsn-experiments/ (or allow_tmp / HSN_EXPERIMENTS_ALLOW_TMP=1 "
                "in tests)"
            )


# --------------------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------------------

_STATUS_CHECK = "CHECK (status IN (" + ", ".join(f"'{s}'" for s in STATUSES) + "))"

MIGRATIONS: list[list[str]] = [
    [  # v1 (P1.11, 7 Nov 2026)
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        """CREATE TABLE sweeps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            spec_hash TEXT NOT NULL,
            spec_text TEXT NOT NULL,
            spec_path TEXT,
            rules_json TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            UNIQUE (name, spec_hash))""",
        """CREATE TABLE arms (
            arm_id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL,
            memory_key TEXT NOT NULL,
            family TEXT NOT NULL,
            protocol TEXT NOT NULL,
            protocol_json TEXT NOT NULL,
            variant TEXT NOT NULL,
            variant_json TEXT NOT NULL,
            seed INTEGER NOT NULL,
            config_path TEXT,
            overrides_json TEXT NOT NULL,
            config_json TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            data_hash TEXT NOT NULL,
            early_stop_json TEXT NOT NULL,
            created_at TEXT NOT NULL)""",
        """CREATE TABLE sweep_arms (
            sweep_id INTEGER NOT NULL REFERENCES sweeps (id),
            arm_id TEXT NOT NULL REFERENCES arms (arm_id),
            label TEXT NOT NULL,
            position INTEGER NOT NULL,
            constants_json TEXT NOT NULL,
            PRIMARY KEY (sweep_id, arm_id))""",
        f"""CREATE TABLE experiments (
            exp_id INTEGER PRIMARY KEY AUTOINCREMENT,
            arm_id TEXT NOT NULL REFERENCES arms (arm_id),
            sweep_id INTEGER REFERENCES sweeps (id),
            label TEXT NOT NULL,
            run_id TEXT NOT NULL,
            status TEXT NOT NULL {_STATUS_CHECK},
            timing INTEGER NOT NULL DEFAULT 0,
            mem_est_mib REAL,
            enqueued_at TEXT NOT NULL,
            claimed_at TEXT,
            started_at TEXT,
            ended_at TEXT,
            worker TEXT,
            host TEXT,
            pid INTEGER,
            pgid INTEGER,
            pid_start TEXT,
            commit_sha TEXT,
            dirty_json TEXT,
            hsn_path TEXT,
            versions_json TEXT,
            card_status TEXT,
            card_json TEXT,
            gpu_mem_peak_mib REAL,
            exit_code INTEGER,
            error TEXT,
            run_dir TEXT,
            log_path TEXT,
            progress_json TEXT,
            imported INTEGER NOT NULL DEFAULT 0,
            import_source TEXT,
            import_sha TEXT,
            times_approx INTEGER NOT NULL DEFAULT 0,
            requeue_of INTEGER REFERENCES experiments (exp_id))""",
        "CREATE INDEX experiments_status ON experiments (status, exp_id)",
        "CREATE INDEX experiments_arm ON experiments (arm_id)",
        """CREATE TABLE results (
            exp_id INTEGER PRIMARY KEY REFERENCES experiments (exp_id),
            record_json TEXT NOT NULL,
            verdicts_json TEXT NOT NULL,
            early_stop_json TEXT,
            json_path TEXT,
            final_dev_accuracy REAL,
            final_dev_loss REAL,
            train_s REAL,
            total_s REAL)""",
        "CREATE VIEW queue AS SELECT * FROM experiments WHERE status = 'queued' ORDER BY exp_id",
    ],
    [  # v2 (P1.11, 7 Nov 2026): the protocol (execution parameters included) and the resolved
        # configuration of each attempt. The arm id leaves the execution-only parameters out, so
        # two specs can enqueue one arm with different eval/probe batches; the arm row keeps the
        # first insert's (an imported run's, for example), so the attempt carries its own.
        "ALTER TABLE experiments ADD COLUMN protocol_json TEXT",
        "ALTER TABLE experiments ADD COLUMN config_json TEXT",
    ],
]
SCHEMA_VERSION = len(MIGRATIONS)


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, allow_nan=True)


def loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


class Database:
    """One experiment database (module docstring). Safe to share between threads (one connection
    per thread) and processes (SQLite's locking, WAL)."""

    def __init__(
        self, path: str | Path, *, allow_tmp: bool = False, read_only: bool = False
    ) -> None:
        self.path = Path(path).expanduser()
        check_not_tmp(self.path, allow_tmp=allow_tmp)
        self.read_only = read_only  # P1.15 (module docstring)
        self._local = threading.local()
        if read_only:
            if not self.path.is_file():
                raise DatabaseError(f"{self.path}: no database to open read-only")
            version = self.version()
            if version != SCHEMA_VERSION:
                raise DatabaseError(
                    f"{self.path} has schema version {version}, this code reads {SCHEMA_VERSION} "
                    "(read-only: it is not migrated)"
                )
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    @property
    def conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            if self.read_only:  # P1.15: SQLite refuses every write on this connection
                uri = f"{self.path.resolve().as_uri()}?mode=ro"
                conn = sqlite3.connect(uri, uri=True, timeout=30.0, isolation_level=None)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout = 30000")
                self._local.conn = conn
                return conn
            conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise DatabaseError(f"{self.path} is open read-only")
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        conn = self.conn
        conn.execute("BEGIN")
        try:
            yield conn
        finally:
            conn.execute("COMMIT")

    def version(self) -> int:
        return int(self.conn.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> None:
        with self.write() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise DatabaseError(
                    f"{self.path} has schema version {version}; this code knows up to "
                    f"{SCHEMA_VERSION} (update the checkout)"
                )
            for step in range(version, SCHEMA_VERSION):
                for statement in MIGRATIONS[step]:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {step + 1}")

    # small helpers ---------------------------------------------------------------------------

    def rows(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def row(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        r = self.conn.execute(sql, params).fetchone()
        return None if r is None else dict(r)

    def experiment(self, exp_id: int) -> dict[str, Any]:
        row = self.row("SELECT * FROM experiments WHERE exp_id = ?", (exp_id,))
        if row is None:
            raise KeyError(f"no experiment {exp_id}")
        return row

    def arm(self, arm_id: str) -> dict[str, Any]:
        row = self.row("SELECT * FROM arms WHERE arm_id = ?", (arm_id,))
        if row is None:
            raise KeyError(f"no arm {arm_id}")
        return row

    def meta(self, key: str) -> str | None:
        row = self.row("SELECT value FROM meta WHERE key = ?", (key,))
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        with self.write() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    @property
    def paused(self) -> bool:
        return self.meta("paused") == "1"

    def set_paused(self, paused: bool) -> None:
        self.set_meta("paused", "1" if paused else "0")

    def queued(self) -> list[dict[str, Any]]:
        return self.rows("SELECT * FROM queue")

    def running(self, host: str | None = None) -> list[dict[str, Any]]:
        if host is None:
            return self.rows("SELECT * FROM experiments WHERE status = 'running' ORDER BY exp_id")
        return self.rows(
            "SELECT * FROM experiments WHERE status = 'running' AND host = ? ORDER BY exp_id",
            (host,),
        )

    def step_time_estimate(self, memory_key: str) -> tuple[float, int] | None:
        """P1.14: the median recorded step time of this memory key's runs (not imported) and how
        many runs it comes from (:func:`hsn.experiments.steptime.step_time_estimate`)."""
        return _step_time_estimate(self.conn, memory_key)

    def memory_estimate(self, memory_key: str) -> float | None:
        """The largest peak GPU memory (MiB) recorded for an arm with this memory key."""
        row = self.row(
            "SELECT MAX(e.gpu_mem_peak_mib) AS peak FROM experiments e JOIN arms a "
            "ON a.arm_id = e.arm_id WHERE a.memory_key = ? AND e.gpu_mem_peak_mib IS NOT NULL "
            "AND e.imported = 0",
            (memory_key,),
        )
        return None if row is None or row["peak"] is None else float(row["peak"])


# --------------------------------------------------------------------------------------------------
# Enqueue
# --------------------------------------------------------------------------------------------------


def _step_time_estimate(conn: sqlite3.Connection, memory_key: str) -> tuple[float, int] | None:
    from hsn.experiments.steptime import step_time_estimate

    rows = conn.execute(
        "SELECT json_extract(r.record_json, '$.step_time.run_median_s') AS run_median_s, "
        "json_extract(r.record_json, '$.walls.train_s') AS train_s, "
        "json_extract(r.record_json, '$.final.steps') AS steps, "
        "json_extract(r.record_json, '$.measured.compile_steps') AS compile_steps, "
        "json_extract(r.record_json, '$.failure') IS NOT NULL AS failed "
        "FROM results r JOIN experiments e ON e.exp_id = r.exp_id JOIN arms a "
        "ON a.arm_id = e.arm_id WHERE a.memory_key = ? AND e.imported = 0",
        (memory_key,),
    ).fetchall()
    return step_time_estimate([dict(r) for r in rows])


@dataclass
class EnqueueReport:
    sweep_id: int
    added: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    imported_matches: list[dict[str, Any]] = field(default_factory=list)


def register_sweep(
    conn: sqlite3.Connection,
    name: str,
    spec_hash: str,
    spec_text: str,
    spec_path: str | None,
    rules_json: Any,
) -> int:
    row = conn.execute(
        "SELECT id FROM sweeps WHERE name = ? AND spec_hash = ?", (name, spec_hash)
    ).fetchone()
    if row is not None:
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO sweeps (name, spec_hash, spec_text, spec_path, rules_json, registered_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (name, spec_hash, spec_text, spec_path, dumps(rules_json), now()),
    )
    return int(cur.lastrowid or 0)


def insert_arm(conn: sqlite3.Connection, arm: Mapping[str, Any]) -> None:
    """Insert an arm row unless its id exists (same id = same experiment by construction)."""
    conn.execute(
        "INSERT OR IGNORE INTO arms (arm_id, group_id, memory_key, family, protocol, "
        "protocol_json, variant, variant_json, seed, config_path, overrides_json, config_json, "
        "config_hash, data_hash, early_stop_json, created_at) "
        "VALUES (:arm_id, :group_id, :memory_key, :family, :protocol, :protocol_json, :variant, "
        ":variant_json, :seed, :config_path, :overrides_json, :config_json, :config_hash, "
        ":data_hash, :early_stop_json, :created_at)",
        {"created_at": now(), **arm},
    )


def add_membership(
    conn: sqlite3.Connection, sweep_id: int, arm_id: str, label: str, position: int, constants: Any
) -> None:
    conn.execute(
        "INSERT INTO sweep_arms (sweep_id, arm_id, label, position, constants_json) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT (sweep_id, arm_id) DO UPDATE SET "
        "label = excluded.label, position = excluded.position, "
        "constants_json = excluded.constants_json",
        (sweep_id, arm_id, label, position, dumps(constants)),
    )


def insert_experiment(conn: sqlite3.Connection, **fields: Any) -> int:
    fields.setdefault("enqueued_at", now())
    keys = list(fields)
    cur = conn.execute(
        f"INSERT INTO experiments ({', '.join(keys)}) VALUES ({', '.join('?' for _ in keys)})",
        [fields[k] for k in keys],
    )
    return int(cur.lastrowid or 0)


def arm_row(arm: Any) -> dict[str, Any]:
    """The ``arms`` row of an expanded :class:`hsn.experiments.spec.Arm`."""
    from hsn.config import config_hash

    return {
        "arm_id": arm.arm_id,
        "group_id": arm.group_id,
        "memory_key": arm.memory_key,
        "family": arm.family,
        "protocol": arm.protocol.name,
        "protocol_json": dumps(arm.protocol.model_dump(mode="json")),
        "variant": arm.variant,
        "variant_json": dumps(arm.variant_params),
        "seed": arm.seed,
        "config_path": arm.config_path,
        "overrides_json": dumps(arm.overrides),
        "config_json": dumps(arm.config.model_dump(mode="json")),
        "config_hash": config_hash(arm.config),
        "data_hash": arm.data_hash,
        "early_stop_json": dumps(arm.early_stop),
    }


def enqueue_arms(
    db: Database,
    *,
    name: str,
    spec_hash: str,
    spec_text: str,
    spec_path: str | None,
    rules_json: Any,
    arms: Sequence[Any],
    force: bool = False,
    skip_imported: bool = False,
    mem_est_mib: float | None = None,
) -> EnqueueReport:
    """Register the sweep version and queue every arm that has no queued, running, done or
    stopped-early experiment (``force`` queues anyway; imported runs never block unless
    ``skip_imported``). One transaction. P1.14: every arm it would add passes the step-time rule
    (:func:`hsn.experiments.steptime.enqueue_refusal`); any refusal raises
    :class:`~hsn.experiments.steptime.StepTimeRefused` and queues nothing. Each added entry
    carries the arm's expected step time (``step_s``, None when no run of its memory key exists)."""
    from hsn.experiments.steptime import StepTimeRefused, enqueue_refusal

    refused: list[str] = []
    with db.write() as conn:
        sweep_id = register_sweep(conn, name, spec_hash, spec_text, spec_path, rules_json)
        report = EnqueueReport(sweep_id)
        for arm in arms:
            insert_arm(conn, arm_row(arm))
            add_membership(conn, sweep_id, arm.arm_id, arm.label, arm.position, arm.constants)
            existing = [
                dict(r)
                for r in conn.execute(
                    "SELECT exp_id, status, imported FROM experiments WHERE arm_id = ? "
                    "ORDER BY exp_id",
                    (arm.arm_id,),
                ).fetchall()
            ]
            own = [e for e in existing if not e["imported"] and e["status"] in BLOCKING]
            imported = [e for e in existing if e["imported"] and e["status"] in FINISHED]
            entry = {"label": arm.label, "arm_id": arm.arm_id}
            if imported:
                report.imported_matches.append({**entry, "exp_id": imported[-1]["exp_id"]})
            if own and not force:
                report.skipped.append(
                    {**entry, "exp_id": own[-1]["exp_id"], "status": own[-1]["status"]}
                )
                continue
            if imported and skip_imported and not force:
                report.skipped.append(
                    {**entry, "exp_id": imported[-1]["exp_id"], "status": "imported"}
                )
                continue
            estimate = _step_time_estimate(conn, arm.memory_key)
            why = enqueue_refusal(arm, estimate)
            if why is not None:
                refused.append(why)
                continue
            exp_id = insert_experiment(
                conn,
                arm_id=arm.arm_id,
                sweep_id=sweep_id,
                label=arm.label,
                run_id=arm.run_id,
                status="queued",
                timing=int(arm.timing),
                mem_est_mib=mem_est_mib,
                protocol_json=dumps(arm.protocol.model_dump(mode="json")),
                config_json=dumps(arm.config.model_dump(mode="json")),
            )
            report.added.append(
                {**entry, "exp_id": exp_id, "step_s": estimate[0] if estimate else None}
            )
        if refused:  # raising inside the transaction rolls every write of this enqueue back
            raise StepTimeRefused(
                f"{len(refused)} arm(s) refused under the 0.5 s step-time rule; nothing was "
                "queued:\n  " + "\n  ".join(refused)
            )
    return report


# --------------------------------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------------------------------


def claim_next(
    db: Database,
    choose: Callable[[list[dict[str, Any]], list[dict[str, Any]]], int | None],
    *,
    worker: str,
    host: str,
) -> dict[str, Any] | None:
    """In one ``BEGIN IMMEDIATE``: read the queue and every running experiment, let ``choose``
    pick an experiment id (or None), and mark it running for this worker."""
    with db.write() as conn:
        queue = [dict(r) for r in conn.execute("SELECT * FROM queue").fetchall()]
        running = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM experiments WHERE status = 'running' ORDER BY exp_id"
            ).fetchall()
        ]
        if not queue:
            return None
        exp_id = choose(queue, running)
        if exp_id is None:
            return None
        cur = conn.execute(
            "UPDATE experiments SET status = 'running', claimed_at = ?, worker = ?, host = ? "
            "WHERE exp_id = ? AND status = 'queued'",
            (now(), worker, host, exp_id),
        )
        if cur.rowcount != 1:
            raise DatabaseError(f"experiment {exp_id} was not queued when claimed")
        return dict(
            conn.execute("SELECT * FROM experiments WHERE exp_id = ?", (exp_id,)).fetchone()
        )


def set_launched(
    db: Database,
    exp_id: int,
    *,
    pid: int,
    pgid: int,
    pid_start: str | None,
    run_dir: str,
    log_path: str,
) -> None:
    with db.write() as conn:
        conn.execute(
            "UPDATE experiments SET pid = ?, pgid = ?, pid_start = ?, run_dir = ?, log_path = ? "
            "WHERE exp_id = ? AND status = 'running' AND pid IS NULL",
            (pid, pgid, pid_start, run_dir, log_path, exp_id),
        )


_GUARD = "exp_id = ? AND status = 'running' AND (pid IS NULL OR pid = ?)"


def child_start(db: Database, exp_id: int, pid: int, **fields: Any) -> bool:
    """The child's first write: its PID (if the worker has not written it yet), start time,
    commit, host, versions. False when the row is no longer running for this PID."""
    fields = {"started_at": now(), **fields}
    sets = ", ".join(f"{k} = ?" for k in fields)
    with db.write() as conn:
        cur = conn.execute(
            f"UPDATE experiments SET pid = COALESCE(pid, ?), {sets} WHERE {_GUARD}",
            (pid, *fields.values(), exp_id, pid),
        )
        return cur.rowcount == 1


def child_progress(db: Database, exp_id: int, pid: int, progress: Mapping[str, Any]) -> bool:
    with db.write() as conn:
        cur = conn.execute(
            f"UPDATE experiments SET progress_json = ? WHERE {_GUARD}",
            (dumps(progress), exp_id, pid),
        )
        return cur.rowcount == 1


def child_finish(
    db: Database,
    exp_id: int,
    pid: int,
    *,
    status: str,
    result: Mapping[str, Any] | None = None,
    **fields: Any,
) -> bool:
    """The child's last write, in one transaction: the results row (if any) and the final status.
    False (and nothing written) when the row is no longer running for this PID."""
    if status not in ("done", "stopped_early", "failed", "cancelled"):
        raise ValueError(f"not a final status: {status!r}")
    fields = {"status": status, "ended_at": now(), **fields}
    sets = ", ".join(f"{k} = ?" for k in fields)
    with db.write() as conn:
        cur = conn.execute(
            f"UPDATE experiments SET {sets} WHERE {_GUARD}", (*fields.values(), exp_id, pid)
        )
        if cur.rowcount != 1:
            return False
        if result is not None:
            write_result(conn, exp_id, result)
    return True


def write_result(conn: sqlite3.Connection, exp_id: int, result: Mapping[str, Any]) -> None:
    conn.execute(
        "INSERT INTO results (exp_id, record_json, verdicts_json, early_stop_json, json_path, "
        "final_dev_accuracy, final_dev_loss, train_s, total_s) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            exp_id,
            dumps(result["record"]),
            dumps(result.get("verdicts") or {}),
            dumps(result.get("early_stop")) if result.get("early_stop") is not None else None,
            result.get("json_path"),
            result.get("final_dev_accuracy"),
            result.get("final_dev_loss"),
            result.get("train_s"),
            result.get("total_s"),
        ),
    )


def mark_failed(
    db: Database,
    exp_id: int,
    error: str,
    *,
    exit_code: int | None = None,
    requeue: bool = False,
    expect_pid: int | None = None,
) -> int | None:
    """Mark a running experiment failed (only if it is still running, with ``expect_pid`` when
    given); with ``requeue`` also queue a new attempt. Returns the new attempt's id."""
    with db.write() as conn:
        sql = "UPDATE experiments SET status = 'failed', ended_at = ?, error = ?, exit_code = ? "
        sql += "WHERE exp_id = ? AND status = 'running'"
        params: list[Any] = [now(), error, exit_code, exp_id]
        if expect_pid is not None:
            sql += " AND pid = ?"
            params.append(expect_pid)
        cur = conn.execute(sql, params)
        if cur.rowcount != 1 or not requeue:
            return None
        return _requeue_in(conn, exp_id)


def _requeue_in(conn: sqlite3.Connection, exp_id: int) -> int:
    row = conn.execute("SELECT * FROM experiments WHERE exp_id = ?", (exp_id,)).fetchone()
    return insert_experiment(
        conn,
        arm_id=row["arm_id"],
        sweep_id=row["sweep_id"],
        label=row["label"],
        run_id=row["run_id"],
        status="queued",
        timing=row["timing"],
        mem_est_mib=row["mem_est_mib"],
        protocol_json=row["protocol_json"],
        config_json=row["config_json"],
        requeue_of=exp_id,
    )


def requeue(db: Database, exp_ids: Sequence[int], statuses: Sequence[str]) -> list[int]:
    """A new queued attempt (at the back of the queue) for every experiment in ``exp_ids`` whose
    status is in ``statuses``; returns the new ids."""
    out: list[int] = []
    with db.write() as conn:
        for exp_id in exp_ids:
            row = conn.execute(
                "SELECT status, imported FROM experiments WHERE exp_id = ?", (exp_id,)
            ).fetchone()
            if row is None or row["status"] not in statuses or row["imported"]:
                continue
            out.append(_requeue_in(conn, exp_id))
    return out


def cancel_queued(db: Database, exp_ids: Sequence[int]) -> list[int]:
    with db.write() as conn:
        done = []
        for exp_id in exp_ids:
            cur = conn.execute(
                "UPDATE experiments SET status = 'cancelled', ended_at = ? "
                "WHERE exp_id = ? AND status = 'queued'",
                (now(), exp_id),
            )
            if cur.rowcount == 1:
                done.append(exp_id)
        return done


def update_card(
    db: Database,
    exp_id: int,
    card: Mapping[str, Any],
    *,
    status: str | None,
    peak_mib: float | None,
) -> None:
    """The worker's card watch for one experiment (any status: the child may just have finished)."""
    with db.write() as conn:
        conn.execute(
            "UPDATE experiments SET card_json = ?, card_status = COALESCE(?, card_status), "
            "gpu_mem_peak_mib = CASE WHEN ? IS NULL THEN gpu_mem_peak_mib "
            "ELSE MAX(COALESCE(gpu_mem_peak_mib, 0), ?) END WHERE exp_id = ?",
            (dumps(card), status, peak_mib, peak_mib, exp_id),
        )
