"""P1.14 tests: a failed or cancelled arm keeps its partial record.

The planted divergence is P1.12's demo pattern on the CPU: from step 4 on, the logits' gradient is
infinite, the guard skips every step, and with the stop rule patched to 3 consecutive skips the
fit raises at step 6 (the last completed step is 5). The child runs in-process (so that the plant
reaches it) on a temporary database; ``result.json``, the results row, the verdicts, ``status``
and ``report`` must all show what existed up to the failure.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from hsn.experiments import db as dbm
from hsn.experiments.cli import main as cli
from hsn.experiments.protocols import failure_info
from hsn.experiments.report import build_report, format_markdown
from hsn.experiments.rules import evaluate_verdicts, parse_rules
from hsn.train import loop
from p1_11_support import (
    db_path,
    dry_spec,
    isolated,  # noqa: F401  (fixture)
    restore_determinism,  # noqa: F401  (fixture)
    tiny_manifest,  # noqa: F401  (fixture)
)
from p1_14_support import (
    claim,
    enqueue_and_claim,
    plant_inf_gradient,
    run_child,
    tiny_reads_state_spec,
)

pytestmark = pytest.mark.usefixtures("isolated", "restore_determinism")

RULES: dict[str, Any] = {
    "verdicts": {
        "reads_state": {"preset": "reads_state"},
        "open": {"preset": "memory_open", "from_step": 2, "every": 2, "to_step": 2},
        "gru_only": {"preset": "reads_state", "applies_to": {"family": ["gru"]}},
    },
    "early_stop": [
        {
            "name": "never",
            "at_steps": [2],
            "outcome": "never",
            "when": {"metric": "eval:accuracy", "steps": {"at": 2}, "op": "<", "value": -1.0},
        }
    ],
}


def _cli(*argv: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert cli(list(argv)) == 0
    return out.getvalue()


def _result(db: dbm.Database, exp_id: int) -> dict[str, Any]:
    row = db.row("SELECT * FROM results WHERE exp_id = ?", (exp_id,))
    assert row is not None, f"no results row for exp {exp_id}"
    return row


def test_p1_14_failed_arm_keeps_partial_record(
    tmp_path: Path,
    tiny_manifest: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = dbm.Database(db_path())
    spec = tiny_reads_state_spec(tmp_path, tiny_manifest, rules=RULES)
    (exp_id,) = enqueue_and_claim(db, spec)
    monkeypatch.setattr(loop, "NONFINITE_MAX_CONSECUTIVE", 3)
    plant_inf_gradient(monkeypatch, from_call=4)
    assert run_child(db, exp_id, tmp_path / "runs") == 1

    exp = db.experiment(exp_id)
    assert exp["status"] == "failed" and "P1.12 stop rule" in exp["error"]
    row = _result(db, exp_id)
    assert row["final_dev_accuracy"] is None, "a failed arm has no final result"
    record = json.loads(row["record_json"])
    failure = record["failure"]
    assert failure["kind"] == "nonfinite" and failure["stage"] == "train"
    assert failure["last_step"] == 5 and failure["skipped_nonfinite"] == 3
    assert failure["reason"] == "non-finite gradients (P1.12 stop)"
    assert "step 6: 3 consecutive optimizer steps skipped" in failure["error"]
    # every completed step's train record, the skipped ones with their kind of non-finite value
    assert [t["step"] for t in record["train"]] == [1, 2, 3, 4, 5]
    for t in record["train"]:
        skipped = t["step"] >= 4
        assert (t["grad_norm"] is None) == skipped
        assert ("nonfinite" in t) == skipped
        if skipped:
            # the planted infinite gradient reaches the global norm as inf or nan (inf * 0)
            assert set(t["nonfinite"]) == {"grad_norm"} and t["lr"] > 0
            assert t["nonfinite"]["grad_norm"] in ("inf", "nan")
    # evaluations and diagnostics up to the failure (a skipped step has no diagnostics sample)
    assert [e["step"] for e in record["evals"]] == [2, 4]
    assert [d["step"] for d in record["diagnostics"]] == [2]
    assert "gate/p50" in record["diagnostics"][0]
    assert [e["rule"] for e in record["early_stop_log"]] == ["never"]
    assert record["stopped"] is None and record["final"]["eval_step"] == 4
    assert record["walls"]["fit_s"] > 0 and "total_s" in record["walls"]
    # the verdicts: failed for every rule that applies, never a pass
    verdicts = {k: v["label"] for k, v in json.loads(row["verdicts_json"]).items()}
    label = "failed: non-finite gradients (P1.12 stop) @ 5"
    assert verdicts == {"reads_state": label, "open": label, "gru_only": "n/a (family hsn)"}
    # the run directory holds the full record
    full = json.loads(Path(row["json_path"]).read_text(encoding="utf-8"))
    assert full["failure"]["last_step"] == 5 and full["verdicts"]["open"]["label"] == label
    assert len(full["train"]) == 5 and full["arm"]["exp_id"] == exp_id
    # status and report show it
    status = _cli("status")
    assert "failed @ 5: non-finite gradients (P1.12 stop)" in status
    per_seed = build_report(db, sweep="p114", per_seed=True)
    (seed_row,) = per_seed.rows
    assert seed_row["status"] == "failed"
    assert seed_row["failure"] == "non-finite gradients (P1.12 stop) @ 5"
    assert seed_row["verdict:reads_state"] == label
    group = build_report(db, sweep="p114")
    (group_row,) = group.rows
    assert group_row["seeds"] == "0/1" and group_row["failure"] == seed_row["failure"]
    assert "(finding)" not in format_markdown(group)


def test_p1_14_failure_verdicts_never_pass(tmp_path: Path) -> None:
    """Every applicable rule reads ``failed: <reason> @ <step>`` whatever the partial record holds
    (here a probe that would pass); a rule outside ``applies_to`` still reads ``n/a``; a group of
    failed seeds is never a finding; the dry run's ``fail: true`` and a failure before the protocol
    starts both keep a record."""
    rules = parse_rules(
        {
            "verdicts": {
                "rs": {"preset": "reads_state", "group": {"min_agree": 1}},
                "gru": {"preset": "reads_state", "applies_to": {"family": ["gru"]}},
            }
        }
    )
    try:
        raise RuntimeError("step 9: boom")
    except RuntimeError as exc:
        failure = failure_info(exc, stage="train", last_step=8)
    passing_probe = {"correct_n": 10, "correct_follow_frac": 1.0, "correct_unchanged_frac": 0.0}
    record = {"probe": passing_probe, "failure": failure, "evals": [], "protocol": "reads_state"}
    labels = {k: v["label"] for k, v in evaluate_verdicts(rules, record, {}, "hsn").items()}
    assert labels == {"rs": "failed: RuntimeError: step 9: boom @ 8", "gru": "n/a (family hsn)"}
    assert evaluate_verdicts(rules, {**record, "failure": None}, {}, "hsn")["rs"]["label"] == (
        "reads state"
    )
    for stage, prefix in (("probe", "probe: "), ("train", "")):
        assert failure_info(ValueError("x"), stage=stage, last_step=0)["reason"] == (
            f"{prefix}ValueError: x"
        )

    db = dbm.Database(db_path())
    ids = enqueue_and_claim(db, dry_spec(tmp_path, n=3, name="dryfail", fail=True))
    for exp_id in ids:
        assert run_child(db, exp_id, tmp_path / "runs") == 1
        assert db.experiment(exp_id)["status"] == "failed"
        failed = json.loads(_result(db, exp_id)["record_json"])["failure"]
        assert failed["stage"] == "train" and failed["last_step"] == 0
        assert failed["reason"] == "RuntimeError: dry_run: failing as asked (fail: true)"
    report = build_report(db, sweep="dryfail")
    (row,) = report.rows
    assert row["seeds"] == "0/3" and row["status"] == "failed 3"
    assert row["failure"].count("dry_run: failing as asked") == 3
    assert "(finding)" not in format_markdown(report)

    # a failure before the protocol starts: the stored protocol no longer matches the arm id
    (exp_id,) = [int(a["exp_id"]) for a in _enqueue(db, tmp_path, "mismatch")]
    with db.write() as conn:
        protocol = json.loads(db.experiment(exp_id)["protocol_json"])
        conn.execute(
            "UPDATE experiments SET protocol_json = ? WHERE exp_id = ?",
            (json.dumps({**protocol, "accuracy": 0.25}), exp_id),
        )
    claim(db, exp_id)
    assert run_child(db, exp_id, tmp_path / "runs") == 1
    failed = json.loads(_result(db, exp_id)["record_json"])["failure"]
    assert failed["stage"] == "setup" and failed["reason"].startswith("RuntimeError: arm id")


def _enqueue(db: dbm.Database, tmp: Path, name: str) -> list[dict[str, Any]]:
    from p1_11_support import enqueue_spec

    return enqueue_spec(db, dry_spec(tmp, n=1, name=name)).added


def test_p1_14_cancel_keeps_partial_record(tmp_path: Path) -> None:
    """SIGTERM (``cancel``) to a running child: status ``cancelled`` as before, and now a results
    row and ``result.json`` with ``failure.kind == cancelled`` (the real signal path, in a
    subprocess)."""
    db = dbm.Database(db_path())
    (added,) = enqueue_and_claim(db, dry_spec(tmp_path, n=1, name="sleepy", sleep_s=60.0))
    log = tmp_path / "child.log"
    cmd = [
        sys.executable,
        "-m",
        "hsn.experiments",
        "--db",
        str(db.path),
        "--root",
        str(tmp_path / "runs"),
        "run-arm",
        "--exp",
        str(added),
    ]
    with open(log, "wb") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.time() + 60
        while f"experiment {added} (" not in log.read_text(errors="replace"):
            assert proc.poll() is None and time.time() < deadline, log.read_text()
            time.sleep(0.2)
        time.sleep(0.5)  # the dry run is sleeping now
        os.killpg(proc.pid, signal.SIGTERM)
        assert proc.wait(timeout=60) == 143
    finally:
        if proc.poll() is None:
            proc.kill()
    exp = db.experiment(added)
    assert exp["status"] == "cancelled" and exp["error"] == "cancelled (SIGTERM)"
    row = _result(db, added)
    failure = json.loads(row["record_json"])["failure"]
    assert failure["kind"] == "cancelled" and failure["reason"] == "cancelled (SIGTERM)"
    labels = json.loads(row["verdicts_json"])
    assert labels == {}  # the dry spec registers no verdicts
    assert json.loads(Path(row["json_path"]).read_text())["failure"]["kind"] == "cancelled"
    assert "cancelled @ 0: cancelled (SIGTERM)" in _cli("status")
