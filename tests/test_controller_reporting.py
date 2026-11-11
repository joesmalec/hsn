"""Live snapshots, honest warning summaries and provenance-controlled ablation comparisons."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hsn.config import AdaptiveLRConfig
from hsn.experiments import db as dbm
from hsn.experiments.cli import main as cli
from hsn.experiments.controller import controller_progress, controller_report, summarize_signals
from hsn.train.adaptive import AdaptiveLR
from p1_11_support import db_path, dry_spec, enqueue_spec, isolated  # noqa: F401

pytestmark = pytest.mark.usefixtures("isolated")


def test_snapshot_does_not_serialize_history_or_alias_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = AdaptiveLR(AdaptiveLRConfig(), {"shared"})
    c.guard(1, "test")
    trainer = SimpleNamespace(
        adaptive_lr=c,
        step=1,
        optimizer=SimpleNamespace(param_groups=[{"adaptive_group": "shared", "lr": 0.0003}]),
    )

    def no_history_copy() -> None:
        raise AssertionError("live progress must not serialize the full controller history")

    monkeypatch.setattr(c, "state_dict", no_history_copy)
    snapshot = controller_progress(trainer)["adaptive_lr"]
    assert snapshot["next_update_lrs"] == {"shared": [0.0003]}
    assert snapshot["cooldown_windows"] == 2
    assert snapshot["last_adjustment"]["reason"] == "test"
    c.guard(2, "another")
    assert snapshot["scales"] == {"shared": 0.5}
    snapshot["last_adjustment"]["scales"]["shared"] = 10
    assert c.state.observations[0]["scales"] == {"shared": 0.5}
    assert controller_progress(SimpleNamespace()) == {}


def test_signal_summary_excludes_actions_cooldowns_trials_and_guards() -> None:
    def event(
        step: int,
        loss: float,
        decision: str = "hold",
        *,
        grad: float = 1,
        before: float = 1,
        after: float = 1,
    ) -> dict[str, Any]:
        return {
            "step": step,
            "loss": loss,
            "decision": decision,
            "grad_median": grad,
            "anchor_loss": 1.0,
            "healthy_grad": 1.0,
            "update_ratios": {"router": 0.006, "decoder": 0.0001},
            "before": {"shared": before},
            "scales": {"shared": after},
        }

    raw = {
        "config": AdaptiveLRConfig().model_dump(),
        "observations": [
            event(0, 1, "baseline"),
            event(1, 1.02, grad=4),
            event(2, 1.05),
            event(3, 1.1, "update_cut", after=0.5),
            event(4, 1, "cooldown", before=0.5, after=0.5),
            event(5, 0.99),
            {"step": 6, "decision": "guard_cut"},
            event(7, 1, "baseline"),
            event(8, 1, "trial_started", before=0.5, after=0.55),
            event(9, 1, "trial_pending"),
            event(10, 0.95, "trial_accepted"),
            event(11, 0.94),
            event(12, 0.93),
        ],
    }
    before = copy.deepcopy(raw)
    report = summarize_signals(raw)
    assert raw == before
    loss = report["warning_follow_through"]["loss_regression"]
    assert loss["eligible_pairs"] == 3 and loss["alerts"] == 2
    assert loss["fraction_followed_by_regression"] == 1
    decoder = report["warning_follow_through"]["update/decoder"]
    assert decoder["fraction_followed_by_regression"] is None
    assert decoder["regressions_without_alert"] == 2
    assert [o["step"] for o in report["rate_change_outcomes"]] == [3, 8]
    assert report["rate_change_outcomes"][0]["next_monitor_loss_delta"] == pytest.approx(-0.1)
    assert report["excluded_pairs"]["guard_or_missing_loss"] == 2
    assert "not causal" in report["interpretation"]


def test_report_pairs_only_completed_clean_matching_experiments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Synthetic stored evidence tests pairing policy; these are not accuracy experiments.
    fixture = json.loads((Path(__file__).parent / "fixtures/adaptive_exp240.json").read_text())
    cases = [
        "full",
        "good",
        "seed",
        "source",
        "dirty",
        "data",
        "protocol",
        "horizon",
        "stale_eval",
        "two_flags",
        "failed",
        "monitor",
        "variant",
    ]
    db = dbm.Database(db_path())
    ids = [a["exp_id"] for a in enqueue_spec(db, dry_spec(tmp_path, n=len(cases))).added]
    for exp, case in zip(ids, cases, strict=True):
        cfg, protocol = copy.deepcopy(fixture["config"]), copy.deepcopy(fixture["protocol"])
        controller = copy.deepcopy(fixture["controller"])
        final = {
            "steps": 1000,
            "eval_step": 1000,
            "dev_accuracy": 0.3 if case == "full" else 0.2,
            "dev_loss": 1.0 if case == "full" else 1.2,
        }
        cfg["run_id"], cfg["notes"] = f"attempt-{exp}", f"notes {case}"
        if case != "full":
            cfg["train"]["adaptive_lr"]["update_cuts"] = False
        if case == "seed":
            cfg["seed"] += 1
        if case == "two_flags":
            cfg["train"]["adaptive_lr"]["loss_cuts"] = False
        if case == "protocol":
            protocol["eval_batch"] += 1
        if case == "horizon":
            final["steps"] = final["eval_step"] = 900
        if case == "stale_eval":
            final["eval_step"] = 900
        if case == "monitor":
            controller["monitor_fingerprint"] = "different"
        controller["config"] = copy.deepcopy(cfg["train"]["adaptive_lr"])
        record = {"adaptive_lr": controller, "final": final}
        with db.write() as conn:
            conn.execute(
                "UPDATE experiments SET status=?,commit_sha=?,dirty_json=?,config_json=?,"
                "protocol_json=? WHERE exp_id=?",
                (
                    "failed" if case == "failed" else "done",
                    "other" if case == "source" else "same",
                    '["M source.py"]' if case == "dirty" else "[]",
                    json.dumps(cfg),
                    json.dumps(protocol),
                    exp,
                ),
            )
            conn.execute(
                "UPDATE arms SET data_hash=?,variant=?,variant_json='{}',early_stop_json='[]' "
                "WHERE arm_id=(SELECT arm_id FROM experiments WHERE exp_id=?)",
                (
                    "different" if case == "data" else "same",
                    "other" if case == "variant" else fixture["variant_name"],
                    exp,
                ),
            )
            dbm.write_result(conn, exp, {"record": record})
    report = controller_report(dbm.Database(db.path, read_only=True), ids)
    (paired,) = report["matched_ablations"]
    assert paired["full_experiment"] == ids[0] and paired["ablated_experiment"] == ids[1]
    assert paired["signal"] == "update_cuts"
    assert paired["accuracy_gain_with_signal"] == pytest.approx(0.1)
    assert paired["loss_reduction_with_signal"] == pytest.approx(0.2)
    out = tmp_path / "controller.json"
    assert cli(["controller-report", "--exp", str(ids[0]), "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == json.loads(capsys.readouterr().out)
    with pytest.raises(dbm.DatabaseError, match="unknown experiment"):
        controller_report(db, [999999])
