"""Controller evidence reaches the harness's successful and failed SQLite records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hsn.config import REPO_ROOT
from hsn.experiments import db as dbm
from hsn.experiments.spec import expand, load_spec
from hsn.train import loop
from p1_11_support import (  # noqa: F401
    db_path,
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
    failed: bool,
) -> None:
    db = dbm.Database(db_path())
    spec = tiny_reads_state_spec(
        tmp_path,
        tiny_manifest,
        name="adaptive-record",
        steps=8,
        eval_every=2,
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
    if failed:
        assert [x["step"] for x in block["observations"] if x["decision"] == "guard_cut"] == [4, 5]
        assert set(block["scales"].values()) == {0.25}
        assert record["failure"]["kind"] == "nonfinite"


def test_adaptive_sweep_is_paired_and_preserves_existing_recipe() -> None:
    spec, _ = load_spec(REPO_ROOT / "configs/sweeps/p1_17_a5_adaptive.yaml")
    arms = expand(spec)
    assert len(arms) == 8 and {a.config.seed for a in arms} == {0, 1}
    for arm in arms:
        assert arm.timing and arm.config.train.schedule == "wsd"
        if arm.config.train.adaptive_lr.enabled:
            assert arm.config.train.rollback_lr_steps == 0
            assert arm.config.train.rollback_lr_mult == 1
        assert arm.config.train.lambda_u == 0
