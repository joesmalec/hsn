"""P1.14 tests: the step-time guard (the user's 0.5 s rule, 7 Nov 2026).

At enqueue an arm that would be added is refused (and nothing is queued) when it is an eager HSN
arm on CUDA, when its ``max_step_s`` is above 0.5 s, both without a ``step_time_approval``, or when
the recorded step time of its memory key is above its limit (approved or not); GRU arms pass. At
run time the median of the eligible steps (no warm-up, compile, evaluation, after-evaluation,
diagnostics or statistics step) is compared with the limit once, and an arm above it fails early
with its partial record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hsn.experiments import db as dbm
from hsn.experiments import steptime
from hsn.experiments.rules import parse_rules
from hsn.experiments.spec import digest, expand, load_spec
from hsn.experiments.steptime import StepTimeExceeded, StepTimeGuard, StepTimeRefused
from p1_11_support import (
    db_path,
    fake_repo,
    isolated,  # noqa: F401  (fixture)
    restore_determinism,  # noqa: F401  (fixture)
    tiny_manifest,  # noqa: F401  (fixture)
    write_yaml,
)
from p1_14_support import enqueue_and_claim, run_child, tiny_reads_state_spec

pytestmark = pytest.mark.usefixtures("isolated", "restore_determinism")

CUDA = ["profile=rtx3060", "train.batch=8", "train.micro_batch=8", "train.steps=40"]


def _spec(tmp: Path, arms: list[dict[str, Any]], **protocol: Any) -> Path:
    raw = {
        "name": "steps",
        "base": {"config": "configs/experiments/g1_ref_a1.yaml", "overrides": list(CUDA)},
        "families": {"gru": {"config": "configs/baselines/gru_2m.yaml"}},
        "protocol": {"name": "reads_state", "diag_every": 10, **protocol},
        "arms": arms,
    }
    return write_yaml(tmp / f"spec_{len(list(tmp.glob('spec_*')))}.yaml", raw)


def _enqueue(db: dbm.Database, tmp: Path, path: Path, **kwargs: Any) -> dbm.EnqueueReport:
    spec, text = load_spec(path)
    arms = expand(spec, repo=fake_repo(tmp))
    return dbm.enqueue_arms(
        db,
        name=spec.name,
        spec_hash=digest(spec.model_dump(mode="json")),
        spec_text=text,
        spec_path=str(path),
        rules_json=parse_rules(spec.rules).to_json(),
        arms=arms,
        **kwargs,
    )


EAGER = {"name": "eager", "family": "hsn", "overrides": ["hardware.compile_mode=none"]}
COMPILED = {
    "name": "compiled",
    "family": "hsn",
    "overrides": ["hardware.compile_mode=reduce-overhead"],
}
GRU = {"name": "gru", "family": "gru", "overrides": ["hardware.compile_mode=none"]}


def _finish(db: dbm.Database, exp_id: int, record: dict[str, Any]) -> None:
    with db.write() as conn:
        conn.execute("UPDATE experiments SET status = 'done' WHERE exp_id = ?", (exp_id,))
        dbm.write_result(conn, exp_id, {"record": record, "verdicts": {}})


def test_p1_14_step_time_enqueue_guard(tmp_path: Path) -> None:
    db = dbm.Database(db_path())
    # an eager CUDA HSN arm without an approval: refused, and nothing of the spec is queued
    with pytest.raises(StepTimeRefused, match=r"eager/s0: an HSN arm on CUDA") as err:
        _enqueue(db, tmp_path, _spec(tmp_path, [EAGER, GRU]))
    assert "nothing was queued" in str(err.value)
    assert db.queued() == [] and db.rows("SELECT * FROM sweeps") == []
    # the eager GRU passes on its own; with an approval (and a limit above its step time) the HSN
    # arm passes too; neither key changes the arm id
    plain = _enqueue(db, tmp_path, _spec(tmp_path, [GRU]))
    assert [a["label"] for a in plain.added] == ["gru/s0"]
    approved = _spec(
        tmp_path, [EAGER], max_step_s=2.0, step_time_approval="user, 7 Nov 2026 (test)"
    )
    report = _enqueue(db, tmp_path, approved)
    assert [a["label"] for a in report.added] == ["eager/s0"]
    assert report.added[0]["step_s"] is None  # no run of its memory key yet
    unapproved = load_spec(_spec(tmp_path, [EAGER]))[0]
    assert expand(unapproved, repo=fake_repo(tmp_path))[0].arm_id == report.added[0]["arm_id"]
    protocol = json.loads(db.experiment(report.added[0]["exp_id"])["protocol_json"])
    assert protocol["max_step_s"] == 2.0 and protocol["step_time_approval"].startswith("user")
    # an already queued (or done) arm is skipped, never refused, without its approval
    again = _enqueue(db, tmp_path, _spec(tmp_path, [EAGER]))
    assert again.added == [] and [s["label"] for s in again.skipped] == ["eager/s0"]
    # a limit above the rule needs an approval, compiled or not
    with pytest.raises(StepTimeRefused, match=r"max_step_s 1 is above the 0.5 s rule"):
        _enqueue(db, tmp_path, _spec(tmp_path, [COMPILED], max_step_s=1.0))
    first = _enqueue(db, tmp_path, _spec(tmp_path, [COMPILED]))
    (compiled,) = first.added
    # a recorded step time above the limit refuses the memory key's next arm (another lr, same
    # shapes), approved or not; the estimate is the median over the key's runs
    _finish(db, compiled["exp_id"], {"step_time": {"run_median_s": 0.8}})
    lr = {**COMPILED, "name": "compiled_lr", "overrides": [*COMPILED["overrides"], "train.lr=2e-3"]}
    with pytest.raises(StepTimeRefused, match=r"expected step time is 0.8 s \(median of 1"):
        _enqueue(db, tmp_path, _spec(tmp_path, [lr], step_time_approval="user (test)"))
    added = _enqueue(
        db, tmp_path, _spec(tmp_path, [lr], max_step_s=1.0, step_time_approval="user (test)")
    ).added
    assert [round(a["step_s"], 3) for a in added] == [0.8]


def test_p1_14_step_time_estimate_rows() -> None:
    """The estimate reads a P1.14 record's run median, else (records before P1.14) the trainer's
    seconds over its non-compile steps; never a failed attempt's partial walls."""
    rows = [
        {"run_median_s": 0.25},
        {"train_s": 733.3, "steps": 3000, "compile_steps": 2.0},  # exp 48: 0.2446 s
        {"train_s": 2197.1, "steps": 3000, "compile_steps": None},  # exp 46: 0.732 s
        {"train_s": 50.0, "steps": 100, "compile_steps": 0, "failed": 1},
        {"train_s": None, "steps": None},
    ]
    median, n = steptime.step_time_estimate(rows) or (None, None)
    assert n == 3 and median == 0.25  # of 0.25, 733.3 / 2998 = 0.2446 and 2197.1 / 3000
    assert steptime.step_time_estimate([{"train_s": 1.0, "steps": 2, "compile_steps": 2}]) is None


def test_p1_14_step_time_runtime_guard() -> None:
    """The guard's exclusions and its single check on synthetic step streams."""
    guard = StepTimeGuard(limit_s=0.5, diag_steps={6}, stats_every=10, skip=2, window=4)
    stream = [
        (1, 9.0, False, False),  # warm-up
        (2, 9.0, False, False),  # warm-up
        (3, 9.0, True, False),  # compile
        (4, 0.2, False, True),  # evaluation
        (5, 9.0, False, False),  # after the evaluation
        (6, 9.0, False, False),  # diagnostics
        (7, 0.3, False, False),
        (8, 0.4, False, False),
        (9, 0.4, False, False),
        (10, 9.0, False, False),  # statistics
        (11, 0.45, False, False),  # the fourth eligible step: the check (median 0.4)
        (12, 0.9, False, False),  # after the check: counted in the run median only
    ]
    for step, seconds, compiled, evaluated in stream:
        guard.observe(step, seconds, compiled=compiled, evaluated=evaluated)
    summary = guard.summary()
    assert summary["checked"] and summary["check"] == {
        "median_s": 0.4,
        "n": 4,
        "first": 7,
        "last": 11,
        "passed": True,
    }
    assert summary["excluded"] == {
        "after_eval": 1,
        "compile": 1,
        "diag": 1,
        "eval": 1,
        "stats": 1,
        "warmup": 2,
    }
    assert summary["run_n"] == 5 and summary["run_median_s"] == 0.4
    slow = StepTimeGuard(limit_s=0.5, skip=0, window=3)
    slow.observe(1, 0.6, compiled=False, evaluated=False)
    slow.observe(2, 0.7, compiled=False, evaluated=False)
    with pytest.raises(
        StepTimeExceeded, match=r"step time 0.7 s > 0.5 s \(median of 3 steps, 1-3\)"
    ):
        slow.observe(3, 0.8, compiled=False, evaluated=False)
    short = StepTimeGuard(limit_s=0.5, skip=0, window=50)
    short.observe(1, 9.0, compiled=False, evaluated=False)
    assert not short.summary()["checked"] and short.summary()["run_median_s"] == 9.0


def test_p1_14_step_time_fails_a_slow_arm(
    tmp_path: Path,
    tiny_manifest: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny CPU arm whose limit no step can meet (``max_step_s`` 1e-6, window 2, no warm-up)
    fails at its second eligible step with ``failure.kind == step_time``, a partial record and a
    ``failed: step time ...`` verdict; the same arm at the default limit records its step times."""
    monkeypatch.setattr(steptime, "STEP_TIME_SKIP", 0)
    monkeypatch.setattr(steptime, "STEP_TIME_WINDOW", 2)
    db = dbm.Database(db_path())
    rules = {"verdicts": {"rs": {"preset": "reads_state"}}}
    overrides = ["train.stats_every=100"]
    spec = tiny_reads_state_spec(
        tmp_path,
        tiny_manifest,
        name="slow",
        eval_every=4,
        protocol={"diag_every": 4, "max_step_s": 1e-6},
        rules=rules,
        arms=[{"name": "a", "family": "hsn", "overrides": overrides}],
    )
    (exp_id,) = enqueue_and_claim(db, spec)
    assert run_child(db, exp_id, tmp_path / "runs") == 1
    exp = db.experiment(exp_id)
    assert exp["status"] == "failed" and "StepTimeExceeded" in exp["error"]
    row = db.row("SELECT * FROM results WHERE exp_id = ?", (exp_id,))
    assert row is not None
    record = json.loads(row["record_json"])
    failure = record["failure"]
    assert failure["kind"] == "step_time" and failure["last_step"] == 2
    assert failure["reason"].startswith("step time ")
    assert "(median of 2 steps, 1-2)" in failure["reason"]
    assert [t["step"] for t in record["train"]] == [1, 2]
    assert record["step_time"]["check"]["passed"] is False
    label = json.loads(row["verdicts_json"])["rs"]["label"]
    assert label == f"failed: {failure['reason']} @ 2"

    fine = tiny_reads_state_spec(
        tmp_path,
        tiny_manifest,
        name="fine",
        eval_every=4,
        protocol={"diag_every": 4},
        rules=rules,
        arms=[{"name": "a", "family": "hsn", "overrides": overrides}],
    )
    (exp_id,) = enqueue_and_claim(db, fine)
    assert run_child(db, exp_id, tmp_path / "runs") == 0
    row = db.row("SELECT record_json FROM results WHERE exp_id = ?", (exp_id,))
    assert row is not None
    block = json.loads(row["record_json"])["step_time"]
    assert block["limit_s"] == 0.5 and block["checked"] and block["check"]["passed"]
    # steps 1..7 seen by the callback (never the last); 4 is an evaluation (and diagnostics) step
    # and 5 follows it
    assert block["excluded"] == {"after_eval": 1, "eval": 1} and block["run_n"] == 5
