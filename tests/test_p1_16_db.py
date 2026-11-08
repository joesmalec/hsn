"""P1.16 tests: a first open's switch to WAL retries "database is locked" (the user's addition of
8 Nov 2026), and gives up on other errors and after the deadline."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from hsn.experiments import db as dbm


class _FlakyConn:
    """A connection stand-in whose ``PRAGMA journal_mode = WAL`` fails ``fails`` times."""

    def __init__(self, fails: int, message: str = "database is locked") -> None:
        self.fails = fails
        self.message = message
        self.calls = 0

    def execute(self, sql: str, *args: Any) -> None:
        if sql == "PRAGMA journal_mode = WAL":
            self.calls += 1
            if self.calls <= self.fails:
                raise sqlite3.OperationalError(self.message)


def test_p1_16_db_wal_switch_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(dbm.time, "sleep", sleeps.append)
    conn = _FlakyConn(fails=3)
    dbm._set_wal(conn)  # type: ignore[arg-type]
    assert conn.calls == 4
    assert sleeps == pytest.approx([dbm.WAL_RETRY_STEP_S * k for k in (1, 2, 3)])


def test_p1_16_db_wal_switch_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dbm.time, "sleep", lambda s: None)
    # any other operational error raises at once
    other = _FlakyConn(fails=1, message="disk I/O error")
    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        dbm._set_wal(other)  # type: ignore[arg-type]
    assert other.calls == 1
    # a lock that outlives the deadline raises
    clock = iter(range(0, 10_000, 10))  # seconds: the deadline passes after a few attempts
    monkeypatch.setattr(dbm.time, "monotonic", lambda: float(next(clock)))
    stuck = _FlakyConn(fails=10_000)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        dbm._set_wal(stuck)  # type: ignore[arg-type]
    assert 1 < stuck.calls < 10


def test_p1_16_db_open_uses_wal(tmp_path: Path) -> None:
    db = dbm.Database(tmp_path / "w.sqlite", allow_tmp=True)
    assert db.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
