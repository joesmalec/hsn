"""P1.11 tests: the experiment database (schema, migrations, FIFO order, enqueue dedup, guarded
writes) on a database under ``tmp_path``."""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
import tempfile
from pathlib import Path

import pytest
from hsn.experiments import db as dbm
from p1_11_support import db_path, dry_spec, enqueue_spec, isolated  # noqa: F401

pytestmark = pytest.mark.usefixtures("isolated")


def _open(path: str) -> int:
    return dbm.Database(path, allow_tmp=True).version()


def test_p1_11_db_schema_and_migrations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "a.sqlite"
    db = dbm.Database(path)
    assert db.version() == dbm.SCHEMA_VERSION == 1
    names = {
        r["name"] for r in db.rows("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
    }
    assert {"meta", "sweeps", "arms", "sweep_arms", "experiments", "results", "queue"} <= names
    assert db.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    db.set_paused(True)
    assert (
        dbm.Database(path).paused and dbm.Database(path).version() == 1
    )  # reopening is idempotent

    # concurrent first opens migrate exactly once
    fresh = tmp_path / "fresh.sqlite"
    with mp.get_context("spawn").Pool(4) as pool:
        assert pool.map(_open, [str(fresh)] * 8) == [1] * 8

    # a newer database raises
    newer = tmp_path / "newer.sqlite"
    dbm.Database(newer).conn.execute("PRAGMA user_version = 99")
    with pytest.raises(dbm.DatabaseError, match="schema version 99"):
        dbm.Database(newer)

    # a v2 migration upgrades a populated v1 database and keeps its rows
    db.set_meta("kept", "yes")
    db.close()
    monkeypatch.setattr(
        dbm, "MIGRATIONS", [*dbm.MIGRATIONS, ["ALTER TABLE meta ADD COLUMN note TEXT"]]
    )
    monkeypatch.setattr(dbm, "SCHEMA_VERSION", 2)
    upgraded = dbm.Database(path)
    assert upgraded.version() == 2 and upgraded.meta("kept") == "yes"
    assert "note" in {r["name"] for r in upgraded.rows("PRAGMA table_info(meta)")}


def test_p1_11_db_refuses_tmp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HSN_EXPERIMENTS_ALLOW_TMP")
    path = Path(tempfile.gettempdir()) / "p1_11_refused" / "x.sqlite"
    with pytest.raises(dbm.DatabaseError, match="temporary directory"):
        dbm.Database(path)
    assert not path.exists()
    monkeypatch.setenv("HSN_EXPERIMENTS_DB", "/home/someone/hsn-experiments/e.sqlite")
    assert dbm.default_db_path() == Path("/home/someone/hsn-experiments/e.sqlite")
    monkeypatch.delenv("HSN_EXPERIMENTS_DB")
    monkeypatch.delenv("HSN_EXPERIMENTS_ROOT")
    assert dbm.default_db_path() == Path("~/hsn-experiments/experiments.sqlite").expanduser()
    assert dbm.default_root(dbm.default_db_path()) == Path("~/hsn-experiments/runs").expanduser()


def test_p1_11_fifo_order(tmp_path: Path) -> None:
    db = dbm.Database(db_path())
    report = enqueue_spec(db, dry_spec(tmp_path, n=5))
    ids = [a["exp_id"] for a in report.added]
    assert ids == sorted(ids) and [r["exp_id"] for r in db.queued()] == ids
    assert [r["label"] for r in db.queued()] == [f"a/s{s}" for s in range(5)]

    first = dbm.claim_next(db, lambda q, r: q[0]["exp_id"], worker="w", host="h")
    assert first is not None and first["exp_id"] == ids[0] and first["status"] == "running"
    assert dbm.claim_next(db, lambda q, r: None, worker="w", host="h") is None
    assert dbm.cancel_queued(db, [ids[2]]) == [ids[2]]
    assert dbm.cancel_queued(db, [ids[0]]) == []  # running: not cancelled here
    assert dbm.mark_failed(db, ids[0], "boom", exit_code=1) is None
    (new,) = dbm.requeue(db, [ids[0]], ["failed"])
    assert [r["exp_id"] for r in db.queued()] == [ids[1], ids[3], ids[4], new]  # back of the queue
    assert (
        db.experiment(new)["requeue_of"] == ids[0] and db.experiment(ids[0])["status"] == "failed"
    )
    with pytest.raises(dbm.DatabaseError, match="not queued"):
        dbm.claim_next(db, lambda q, r: ids[2], worker="w", host="h")


def test_p1_11_enqueue_dedup(tmp_path: Path) -> None:
    db = dbm.Database(db_path())
    spec = dry_spec(tmp_path, n=6)
    first = enqueue_spec(db, spec)
    assert len(first.added) == 6 and not first.skipped
    again = enqueue_spec(db, spec)
    assert not again.added and [s["status"] for s in again.skipped] == ["queued"] * 6
    assert again.sweep_id == first.sweep_id, "the same spec is the same sweep version"
    ids = [a["exp_id"] for a in first.added]
    statuses = ["done", "running", "stopped_early", "failed", "cancelled", "queued"]
    with db.write() as conn:
        for exp_id, status in zip(ids, statuses, strict=True):
            conn.execute("UPDATE experiments SET status = ? WHERE exp_id = ?", (status, exp_id))
    third = enqueue_spec(db, spec)
    assert sorted(a["label"] for a in third.added) == ["a/s3", "a/s4"]  # failed, cancelled
    assert {s["label"]: s["status"] for s in third.skipped} == {
        "a/s0": "done",
        "a/s1": "running",
        "a/s2": "stopped_early",
        "a/s5": "queued",
    }
    forced = enqueue_spec(db, spec, force=True)
    assert len(forced.added) == 6
    # imported runs of the same arm never block, unless --skip-imported
    other = dry_spec(tmp_path, n=2, name="other", sleep_s=0.5)  # sleep_s: execution-only
    arm_ids = {a["arm_id"] for a in enqueue_spec(db, other).skipped}
    assert len(arm_ids) == 2, "an execution-only parameter keeps the arm ids"
    with db.write() as conn:
        conn.execute(
            "UPDATE experiments SET status = 'cancelled' WHERE status IN ('queued', 'running')"
        )
        conn.execute(
            "UPDATE experiments SET status = 'cancelled' WHERE status IN ('done', 'stopped_early')"
        )
        exp = dbm.insert_experiment(
            conn,
            arm_id=sorted(arm_ids)[0],
            sweep_id=None,
            label="imported",
            run_id="x",
            status="done",
            imported=1,
        )
    matched = enqueue_spec(db, other)
    assert len(matched.added) == 2 and [m["exp_id"] for m in matched.imported_matches] == [exp]
    with db.write() as conn:
        conn.execute("UPDATE experiments SET status = 'cancelled' WHERE status = 'queued'")
    skipped = enqueue_spec(db, other, skip_imported=True)
    assert len(skipped.added) == 1 and skipped.skipped[0]["status"] == "imported"
    # a changed spec is a new sweep version
    changed = dry_spec(tmp_path, n=2, name="other", sleep_s=1.0)
    assert enqueue_spec(db, changed).sweep_id != matched.sweep_id


def test_p1_11_guarded_child_writes(tmp_path: Path) -> None:
    db = dbm.Database(db_path())
    (added,) = enqueue_spec(db, dry_spec(tmp_path, n=1)).added
    exp_id = added["exp_id"]
    assert not dbm.child_start(db, exp_id, 123), "a queued row is not the child's"
    dbm.claim_next(db, lambda q, r: exp_id, worker="w", host="h")
    assert dbm.child_start(db, exp_id, 123, commit_sha="abc")  # pid still NULL: the child sets it
    assert db.experiment(exp_id)["pid"] == 123
    assert not dbm.child_progress(db, exp_id, 999, {"step": 1}), "another PID"
    assert dbm.child_progress(db, exp_id, 123, {"step": 2})
    dbm.mark_failed(db, exp_id, "process gone")  # recovery got there first
    assert not dbm.child_progress(db, exp_id, 123, {"step": 3})
    result = {"record": {"x": 1}, "verdicts": {}}
    assert not dbm.child_finish(db, exp_id, 123, status="done", result=result)
    assert db.row("SELECT * FROM results WHERE exp_id = ?", (exp_id,)) is None
    assert db.experiment(exp_id)["status"] == "failed"
    with pytest.raises(ValueError, match="final status"):
        dbm.child_finish(db, exp_id, 123, status="queued")
    with pytest.raises(sqlite3.IntegrityError):
        with db.write() as conn:
            conn.execute("UPDATE experiments SET status = 'bogus' WHERE exp_id = ?", (exp_id,))
