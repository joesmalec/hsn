"""P1.11 test: one tiny sweep end to end on the CPU through the command line.

Tiny kv + delayed data are built on disk here (never ``data/``). The spec has a tiny HSN and a tiny
GRU on the reads-state protocol (4 steps) and an HSN arm whose early stop fires at step 2; it is
expanded and enqueued, a worker runs it in a subprocess until the queue is empty (children detached
in their own process groups, CUDA hidden), and ``status`` and ``report`` read the result back.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from hsn.experiments import db as dbm
from hsn.experiments.cli import main as cli
from p1_11_support import (
    GRU_2M,
    REF_A1,
    REPO,
    TINY_GRU_OVERRIDES,
    TINY_HSN_OVERRIDES,
    check_hsn_reads_state_record,
    db_path,
    disk_overrides,
    isolated,  # noqa: F401  (fixture)
    tiny_manifest,  # noqa: F401  (fixture)
    write_yaml,
)

pytestmark = pytest.mark.usefixtures("isolated")


def _run(*argv: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert cli(list(argv)) == 0
    return out.getvalue()


def test_p1_11_end_to_end_cpu_sweep(tmp_path: Path, tiny_manifest: Path) -> None:  # noqa: F811
    spec = write_yaml(
        tmp_path / "tiny.yaml",
        {
            "name": "tiny",
            "base": {
                "config": str(REF_A1),
                "overrides": [
                    *disk_overrides(tiny_manifest, steps=4, eval_every=2),
                    *TINY_HSN_OVERRIDES,
                ],
            },
            "families": {"gru": {"config": str(GRU_2M), "overrides": TINY_GRU_OVERRIDES}},
            "protocol": {
                "name": "reads_state",
                "diag_every": 2,
                "diag_docs": 8,
                "eval_batch": 8,
                "probe_batch": 8,
            },
            "constants": {"chance": 0.0208},
            "rules": {
                "verdicts": {
                    "reads_state": {"preset": "reads_state", "group": {"min_agree": 1}},
                    "memory_open": {"preset": "memory_open", "from_step": 2, "every": 2},
                },
                "early_stop": [
                    {
                        "name": "fail_early",
                        "applies_to": {"family": ["hsn"]},
                        "at_steps": [2],
                        "outcome": "fail_early",
                        "when": {
                            "metric": "eval:delayed/in_range/accuracy",
                            "steps": {"at": 2},
                            "op": "<=",
                            "value": "stop_below",
                        },
                    }
                ],
            },
            "arms": [
                {"name": "hsn", "family": "hsn", "constants": {"stop_below": -1.0}},
                {"name": "gru", "family": "gru"},
                {"name": "hsn_stop", "family": "hsn", "constants": {"stop_below": 1.0}},
            ],
        },
    )
    expanded = _run("expand", str(spec))
    assert "3 arms" in expanded and "hsn_stop/s0" in expanded
    added = _run("enqueue", str(spec))
    assert "(spec version 1): 3 added, 0 skipped" in added
    assert "0 added, 3 skipped" in _run("enqueue", str(spec))

    root = tmp_path / "runs"
    worker = subprocess.run(
        [
            sys.executable,
            "-m",
            "hsn.experiments",
            "--db",
            str(db_path()),
            "--root",
            str(root),
            "worker",
            "--exit-when-empty",
            "--poll",
            "0.1",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert worker.returncode == 0, worker.stdout + worker.stderr
    assert "queue empty and no child running: exiting" in worker.stdout

    db = dbm.Database(db_path())
    rows = {
        r["label"]: r
        for r in db.rows(
            "SELECT e.*, r.json_path, r.verdicts_json "
            "FROM experiments e JOIN results r USING (exp_id)"
        )
    }
    assert {k: v["status"] for k, v in rows.items()} == {
        "hsn/s0": "done",
        "gru/s0": "done",
        "hsn_stop/s0": "stopped_early",
    }
    for label, row in rows.items():
        path = Path(row["json_path"])
        assert path.is_file() and path.parent == Path(row["run_dir"]), label
        full = json.loads(path.read_text())
        assert full["arm"]["label"] == label and full["commit"]["sha"] == row["commit_sha"]
        assert (path.parent / "config.resolved.yaml").is_file() and (
            path.parent / "run.log"
        ).is_file()
        assert row["card_status"] is not None
    verdicts = {
        k: {n: v["label"] for n, v in json.loads(r["verdicts_json"]).items()}
        for k, r in rows.items()
    }
    assert verdicts["gru/s0"]["memory_open"] == "n/a (family gru)"
    assert verdicts["hsn_stop/s0"]["reads_state"] == "stopped early: fail_early @ 2"
    assert verdicts["hsn/s0"]["reads_state"] in (
        "does not read state",
        "no correct answers to probe",
        "partial",
        "reads state",
    )
    # the database row of an HSN arm carries the full record (all but the per-probe list)
    stored = db.row("SELECT record_json FROM results WHERE exp_id = ?", (rows["hsn/s0"]["exp_id"],))
    assert stored is not None
    check_hsn_reads_state_record(
        json.loads(stored["record_json"]), eval_steps=[2, 4], diag_steps=[2, 4]
    )
    assert "probes" not in json.loads(stored["record_json"])
    stopped = json.loads(Path(rows["hsn_stop/s0"]["json_path"]).read_text())
    assert stopped["stopped"]["step"] == 2 and stopped["at_stop"]["probe"]["n_probed"] > 0

    status = _run("status")
    assert "queued (0)" in status and "running (0)" in status and "hsn_stop/s0" in status
    report = _run("report", "--sweep", "tiny")
    lines = [line for line in report.splitlines() if line.startswith("| ")]
    assert lines[0].startswith(
        "| label | family | variant | overrides | seeds | status | reads_state | memory_open"
    )
    assert len(lines) == 2 + 3
    assert "fail_early @ 2" in report and "n/a (family gru) 1/1" in report
    assert "(finding)" in report  # min_agree 1 on reads_state
    per_seed = _run("report", "--spec", str(spec), "--per-seed", "--format", "csv")
    assert per_seed.splitlines()[0].startswith("label,exp,family,variant,overrides,seed,status")
    assert len(per_seed.strip().splitlines()) == 4
