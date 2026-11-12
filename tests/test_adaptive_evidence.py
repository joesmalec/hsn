"""Controller evidence reaches the harness's successful and failed SQLite records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from hsn.config import REPO_ROOT
from hsn.experiments import db as dbm
from hsn.experiments.cli import main as cli
from hsn.experiments.controller import controller_report
from hsn.experiments.spec import expand, load_spec
from hsn.train import loop
from p1_11_support import (  # noqa: F401
    db_path,
    fake_repo,
    isolated,
    restore_determinism,
    tiny_manifest,
)
from p1_14_support import enqueue_and_claim, plant_inf_gradient, run_child, tiny_reads_state_spec

pytestmark = pytest.mark.usefixtures("isolated", "restore_determinism")


@pytest.mark.parametrize("failed", [False, True])
def test_sqlite_retains_monitor_and_decisions(
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
        name="adaptive-record",
        steps=8,
        eval_every=4,
        arms=[
            {
                "name": "adaptive",
                "family": "hsn",
                "overrides": [
                    "train.adaptive_lr.enabled=true",
                    "train.adaptive_lr.interval=2",
                    "train.adaptive_lr.monitor_docs=4",
                    "train.adaptive_lr.patience=100",
                ],
            }
        ],
    )
    (exp_id,) = enqueue_and_claim(db, spec)
    snapshots = []
    real_progress = dbm.child_progress

    def capture(database: dbm.Database, exp: int, pid: int, progress: Mapping[str, Any]) -> bool:
        result = real_progress(database, exp, pid, progress)
        row = db.row("SELECT status,progress_json FROM experiments WHERE exp_id=?", (exp,))
        assert result and row is not None and row["status"] == "running"
        assert db.row("SELECT exp_id FROM results WHERE exp_id=?", (exp,)) is None
        snapshots.append(json.loads(row["progress_json"]))
        if progress["step"] == 2:
            assert cli(["status"]) == 0
        return result

    monkeypatch.setattr(dbm, "child_progress", capture)
    if failed:
        monkeypatch.setattr(loop, "NONFINITE_MAX_CONSECUTIVE", 3)
        plant_inf_gradient(monkeypatch, from_call=4)
    assert run_child(db, exp_id, tmp_path / "runs") == int(failed)
    row = db.row("SELECT record_json FROM results WHERE exp_id = ?", (exp_id,))
    assert row is not None
    record = json.loads(row["record_json"])
    block = record["adaptive_lr"]
    assert block["monitor_fingerprint"]
    assert block["observations"][0]["step"] == 2
    assert "loss" in block["observations"][0]
    assert set(block["observations"][0]["update_ratios"]) == set(block["scales"])
    assert block["accepted_steps"] == (3 if failed else 8)
    assert [p["step"] for p in snapshots] == ([1, 2, 4, 5] if failed else [1, 2, 4, 6, 8])
    assert (
        "dev_accuracy" not in snapshots[1]
    )  # controller evidence arrives before the first dev eval
    for snapshot in snapshots:
        live = snapshot["adaptive_lr"]
        assert live["monitor_source"] == "train"
        assert live["monitor_fingerprint"] == block["monitor_fingerprint"]
        assert "observations" not in live  # bounded live payload, full history only in final record
        assert live["next_update_lrs"]
    output = capsys.readouterr().out
    assert "controller (training monitor" in output and "readout lr" in output
    logged = [
        json.loads(line.split(" controller ", 1)[1])
        for line in output.splitlines()
        if " controller {" in line
    ]
    assert [e["observation_count"] for e in logged] == list(range(len(block["observations"]) + 1))
    saved = row["record_json"]
    report = controller_report(dbm.Database(db.path, read_only=True), [exp_id])
    assert report["runs"][0]["signal_summary"]["observations"] == len(block["observations"])
    assert (
        db.row("SELECT record_json FROM results WHERE exp_id=?", (exp_id,))["record_json"] == saved
    )
    if failed:
        assert [x["step"] for x in block["observations"] if x["decision"] == "guard_cut"] == [4, 5]
        assert set(block["scales"].values()) == {0.25}
        assert record["failure"]["kind"] == "nonfinite"
        assert snapshots[-1]["adaptive_lr"]["last_adjustment"]["decision"] == "guard_cut"
    else:
        at_six = next(p for p in snapshots if p["step"] == 6)
        assert at_six["dev_step"] == 4
        assert snapshots[-1]["phase"] == "trained, probing"
        assert snapshots[-1]["adaptive_lr"]["accepted_steps"] == 8


def test_adaptive_sweep_is_paired_and_preserves_existing_recipe(tmp_path: Path) -> None:
    spec, _ = load_spec(REPO_ROOT / "configs/sweeps/p1_17_a5_adaptive.yaml")
    arms = expand(spec, repo=fake_repo(tmp_path))  # arm ids where data/ is not built (CI)
    assert len(arms) == 8 and {a.config.seed for a in arms} == {0, 1}
    for arm in arms:
        assert arm.timing and arm.config.train.schedule == "wsd"
        if arm.config.train.adaptive_lr.enabled:
            assert arm.config.train.rollback_lr_steps == 0
            assert arm.config.train.rollback_lr_mult == 1
        assert arm.config.train.lambda_u == 0


def test_ablations_change_one_signal_and_share_the_full_controller_arm_identity(
    tmp_path: Path,
) -> None:
    repo = fake_repo(tmp_path)  # arm ids where data/ is not built (CI)
    base, _ = load_spec(REPO_ROOT / "configs/sweeps/p1_17_a5_adaptive.yaml")
    controls = {
        a.config.seed: a for a in expand(base, repo=repo) if a.label.startswith("a5_adaptive/")
    }
    spec, _ = load_spec(REPO_ROOT / "configs/sweeps/p1_17_a5_adaptive_ablations.yaml")
    arms = expand(spec, repo=repo)
    assert len(arms) == 10
    for arm in arms:
        reference = controls[arm.config.seed]
        actual, expected = arm.config.model_dump(), reference.config.model_dump()
        actual.pop("run_id")
        expected.pop("run_id")
        a, b = actual["train"]["adaptive_lr"], expected["train"]["adaptive_lr"]
        changed = [key for key in a if a[key] != b[key]]
        if arm.label.startswith("full/"):
            assert arm.arm_id == reference.arm_id and not changed
        else:
            assert len(changed) == 1 and a[changed[0]] is False
            a[changed[0]] = True
        assert actual == expected
        assert arm.protocol == reference.protocol and arm.timing
