"""P1.11 tests: ``import`` of the P1.10 audit's result JSONs.

On the committed fixtures (``tests/fixtures/p1_11``): an early GRU reads-state run (no commit, a
shared card, "all" counts from before the partition fix), the bridge run (commit, an absolute config
path, diagnostics, trained_state) and one overfit summary. Fields, recounts, the recomputed verdict,
idempotence, and that the imported bridge is recognised as the validation sweep's bridge arm (same
arm id) without blocking its enqueue. A second test imports the real audit directory when it exists
on this machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hsn.experiments import db as dbm
from hsn.experiments.importer import IMPORT_SWEEP, import_audit_dir
from hsn.experiments.report import build_report, format_markdown
from hsn.experiments.spec import expand, load_spec
from p1_11_support import FIXTURES, REPO, db_path, enqueue_spec, fake_repo, isolated  # noqa: F401

pytestmark = pytest.mark.usefixtures("isolated")
AUDIT = Path("/home/joe/hsn-eff/evidence/EFF")


def _by_name(db: dbm.Database) -> dict[str, dict[str, object]]:
    rows = db.rows(
        "SELECT e.*, a.family, a.variant, a.protocol, a.protocol_json AS arm_protocol_json, "
        "a.config_path, "
        "r.record_json, r.verdicts_json, r.final_dev_accuracy, r.train_s FROM experiments e "
        "JOIN arms a ON a.arm_id = e.arm_id JOIN results r ON r.exp_id = e.exp_id"
    )
    return {str(r["label"]): r for r in rows}


def test_p1_11_import_audit_jsons(tmp_path: Path) -> None:
    db = dbm.Database(db_path())
    repo = fake_repo(tmp_path)
    report = import_audit_dir(db, FIXTURES, repo=repo, log=lambda m: None)
    assert len(report.imported) == 3 and not report.skipped
    rows = _by_name(db)
    assert set(rows) == {"gru64x1", "hsn_pdg_b32_lr1e-3_bridge", "s2v1_cell_embed_linear_head"}

    gru = rows["gru64x1"]
    record = json.loads(str(gru["record_json"]))
    assert gru["family"] == "gru" and gru["variant"] == "none", (
        "a GRU's default 'pdg' is no variant"
    )
    assert gru["commit_sha"] is None and gru["card_status"] == "shared" and gru["imported"] == 1
    assert record["probe"]["all"] == {"follow": 0, "other": 0, "unchanged": 1000}
    assert record["as_written"]["all"]["follow"] == 25, "the pre-partition count is kept"
    assert any("recounted" in f for f in record["import_flags"])
    assert json.loads(str(gru["verdicts_json"]))["reads_state"]["label"] == "does not read state"
    assert gru["times_approx"] == 1 and gru["status"] == "done"

    bridge = rows["hsn_pdg_b32_lr1e-3_bridge"]
    record = json.loads(str(bridge["record_json"]))
    assert str(bridge["commit_sha"]).startswith("f8884ec3bf") and bridge["card_status"] == "idle"
    assert bridge["config_path"] == "configs/experiments/g1_ref_a1.yaml"
    protocol = json.loads(str(bridge["arm_protocol_json"]))
    assert bridge["protocol_json"] == bridge["arm_protocol_json"], "the attempt carries its own"
    assert (
        protocol["diag_every"] == 500 and protocol["steps"] is None and protocol["diag_docs"] == 32
    )
    assert len(record["diagnostics"]) == 6 and record["trained_state"]["rank_pr/answer"] == 1.0
    assert record["probe"]["correct_before_edit"] == {
        "n": 79,
        "follow": 0,
        "other": 61,
        "unchanged": 18,
    }
    assert bridge["final_dev_accuracy"] == pytest.approx(0.0696, abs=1e-4)
    assert bridge["train_s"] == pytest.approx(2236.7)
    assert not record["import_flags"], "the bridge was written after the partition fix"

    overfit = rows["s2v1_cell_embed_linear_head"]
    record = json.loads(str(overfit["record_json"]))
    assert overfit["protocol"] == "overfit" and overfit["variant"] == "cell_embed_linear_head"
    assert record["overfit"]["first_ge_0.99"] == 625 and len(record["evals"]) == 24
    assert json.loads(str(overfit["arm_protocol_json"]))["diag_every"] == 50

    # idempotent
    again = import_audit_dir(db, FIXTURES, repo=repo, log=lambda m: None)
    assert not again.imported and len(again.skipped) == 3

    # the imported bridge is the validation sweep's eager bridge arm: same arm id, reported as a
    # match by enqueue, never blocking it
    spec, _ = load_spec(REPO / "configs" / "sweeps" / "p1_11_validation.yaml")
    arms = {a.name: a for a in expand(spec, repo=repo)}
    assert arms["hsn_pdg_b32_lr1e-3_bridge"].arm_id == bridge["arm_id"]
    # report --imported answers "what has been run"
    table = format_markdown(build_report(db, sweep=IMPORT_SWEEP, per_seed=True))
    assert "hsn_pdg_b32_lr1e-3_bridge" in table and "does not read state" in table
    labels = {row["label"] for row in build_report(db, imported=True, per_seed=True).rows}
    assert labels == set(rows), "an imported run is reported under its own name"
    assert len(build_report(db, imported=False).rows) == 0


def test_p1_11_imported_runs_do_not_block_enqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = dbm.Database(db_path())
    repo = fake_repo(tmp_path)
    import_audit_dir(db, FIXTURES, repo=repo, log=lambda m: None)
    import hsn.experiments.spec as spec_module

    monkeypatch.setattr(spec_module, "REPO_ROOT", repo)
    original = spec_module.expand

    def expand(spec: Any, repo: Path = repo) -> list[Any]:
        # P1.14: the 0.5 s rule refuses an eager CUDA HSN arm without an approval at enqueue; the
        # validation sweep's eager bridge ran as exp 46 before the rule (the approval is
        # execution-only: the arm id is unchanged)
        arms = original(spec, repo=repo)
        for arm in arms:
            if arm.config.hardware.compile_mode == "none" and arm.family == "hsn":
                arm.protocol = arm.protocol.model_copy(
                    update={"max_step_s": 1.0, "step_time_approval": "test: exp 46's eager bridge"}
                )
        return arms

    monkeypatch.setattr(spec_module, "expand", expand)
    report = enqueue_spec(db, REPO / "configs" / "sweeps" / "p1_11_validation.yaml")
    assert len(report.added) == 5
    assert [m["label"] for m in report.imported_matches] == ["hsn_pdg_b32_lr1e-3_bridge/s0"]
    # the arm row is the import's (same arm id), but the queued attempt runs the spec's protocol
    # and configuration: evaluation at 128 streams (the bug exp 46 hit), the spec's run id
    (bridge,) = [a for a in report.added if a["label"] == "hsn_pdg_b32_lr1e-3_bridge/s0"]
    attempt = db.experiment(bridge["exp_id"])
    assert json.loads(db.arm(bridge["arm_id"])["protocol_json"])["eval_batch"] is None
    assert json.loads(attempt["protocol_json"])["eval_batch"] == 128
    assert json.loads(attempt["config_json"])["run_id"] == attempt["run_id"]
    skipped = enqueue_spec(
        db, REPO / "configs" / "sweeps" / "p1_11_validation.yaml", skip_imported=True
    )
    assert not skipped.added  # all queued already


@pytest.mark.skipif(
    not AUDIT.is_dir(), reason="the audit's evidence directory exists only on the audit machine"
)
def test_p1_11_import_real_audit_directory() -> None:
    db = dbm.Database(db_path())
    report = import_audit_dir(db, AUDIT, log=lambda m: None)
    files = [
        *AUDIT.glob("reads_state/*_reads_state.json"),
        *AUDIT.glob("hsn/*_summary.json"),
        *AUDIT.glob("e2/*_summary.json"),
    ]
    assert len(report.imported) == len(files) >= 40
    rows = _by_name(db)
    resample = rows["hsn_pdg_b32_lr1e-3_bridge_resample600"]
    protocol = json.loads(str(resample["arm_protocol_json"]))
    assert protocol["steps"] == 600 and protocol["diag_every"] == 25
    gru = rows["gru384x2_b32_lr3e-3"]
    assert json.loads(str(gru["verdicts_json"]))["reads_state"]["label"] == "reads state"
    # two imported runs of one arm (the bridge and queue8 (d): same recipe, two commits) keep
    # their own names in a per-seed report
    labels = [row["label"] for row in build_report(db, imported=True, per_seed=True).rows]
    assert "hsn_pdg_b32_lr1e-3_bridge" in labels and "hsn_pdg_b32_lr1e-3" in labels
    assert rows["hsn_pdg_b32_lr1e-3_bridge"]["arm_id"] == rows["hsn_pdg_b32_lr1e-3"]["arm_id"]
    unresolved = [f for f in report.flagged if any("not resolvable" in x for x in f["flags"])]
    assert not unresolved, unresolved
