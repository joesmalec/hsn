"""Live and completed gradient-growth evidence passes through the real SQLite child path."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from hsn.experiments import db as dbm
from hsn.experiments.cli import main as cli
from hsn.train import loop
from p1_11_support import db_path, isolated, restore_determinism, tiny_manifest  # noqa: F401
from p1_14_support import enqueue_and_claim, plant_inf_gradient, run_child, tiny_reads_state_spec

pytestmark = pytest.mark.usefixtures("isolated", "restore_determinism")


@pytest.mark.parametrize("failed", [False, True])
def test_growth_live_and_final_sqlite_records(
    tmp_path: Path,
    tiny_manifest: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failed: bool,
) -> None:
    db = dbm.Database(db_path())
    spec = tiny_reads_state_spec(
        tmp_path,
        tiny_manifest,
        name="gradient-growth",
        steps=8,
        eval_every=4,
        arms=[
            {
                "name": "growth",
                "family": "hsn",
                "overrides": [
                    "train.gradient_growth.enabled=true",
                    'train.gradient_growth.groups={chosen: ["cell.*"]}',
                    "train.gradient_growth.window=1",
                    "train.gradient_growth.reference_lag=2",
                    "train.gradient_growth.interval=1",
                    "train.stats_every=1000",
                ],
            }
        ],
    )
    (exp_id,) = enqueue_and_claim(db, spec)
    snapshots = []
    original = dbm.child_progress

    def capture(database: dbm.Database, exp: int, pid: int, progress: Mapping[str, Any]) -> bool:
        result = original(database, exp, pid, progress)
        snapshots.append(dict(progress))
        if progress["step"] == 3:
            assert cli(["status"]) == 0
        return result

    monkeypatch.setattr(dbm, "child_progress", capture)
    if failed:
        monkeypatch.setattr(loop, "NONFINITE_MAX_CONSECUTIVE", 3)
        plant_inf_gradient(monkeypatch, from_call=4)
    assert run_child(db, exp_id, tmp_path / "runs") == int(failed)
    row = db.row("SELECT record_json FROM results WHERE exp_id=?", (exp_id,))
    assert row is not None
    record = json.loads(row["record_json"])
    growth = record["gradient_growth"]
    assert growth["samples"] == growth["applied_steps"] == (3 if failed else 8)
    assert growth["observations"][0]["step"] == 3
    assert "proposed_update_ratio" in growth["observations"][0]["groups"]["chosen"]
    assert any(s["step"] == 3 for s in snapshots)  # observation without stats/evaluation
    assert all(
        "history" not in s["gradient_growth"] and "observations" not in s["gradient_growth"]
        for s in snapshots
    )
    if not failed:
        at_five = next(s for s in snapshots if s["step"] == 5)
        assert at_five["dev_step"] == 4
    output = capsys.readouterr().out
    assert "gradient growth (pre-clip" in output and "update scale" in output
    assert " gradient_growth {" in output
