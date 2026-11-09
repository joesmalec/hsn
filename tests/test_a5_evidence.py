"""Retain usable A5 weights after an unguarded collapse and probe exact checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hsn.config import REPO_ROOT
from hsn.experiments import db as dbm
from hsn.experiments.probe_check import _restored, probe_rerun
from hsn.experiments.protocols import _Keeper
from hsn.experiments.routing import route_check
from hsn.experiments.spec import expand, load_spec
from p1_11_support import (  # noqa: F401
    db_path,
    isolated,
    restore_determinism,
    tiny_manifest,
)
from p1_14_support import enqueue_and_claim, run_child, tiny_reads_state_spec

pytestmark = pytest.mark.usefixtures("isolated", "restore_determinism")


def test_retention_pins_peak_even_without_a_guard_event(tmp_path: Path) -> None:
    keeper = _Keeper(2, 1, 100, keep_best=True)
    history = SimpleNamespace(checkpoints=[], evals=[])
    trainer = SimpleNamespace(skipped_steps=[])
    for step, accuracy in ((2, 0.1), (4, 0.8), (6, 0.3), (8, 0.02), (10, 0.02)):
        path = tmp_path / f"step_{step:08d}.pt"
        path.touch()
        history.checkpoints.append(str(path))
        history.evals.append({"step": step, "accuracy": accuracy})
        keeper.prune(trainer, history)
    assert [s for s, _ in keeper.on_disk(history)] == [4, 10]
    assert keeper.best(history) == {
        "step": 4,
        "accuracy": 0.8,
        "path": str(tmp_path / "step_00000004.pt"),
    }
    # Unevaluated checkpoints do not inherit the latest evaluation's score.
    history.evals.append({"step": 9, "accuracy": 1.0})
    assert keeper.best(history)["step"] == 4  # type: ignore[index]


def test_retention_preserves_best_and_pre_guard_checkpoints(tmp_path: Path) -> None:
    keeper = _Keeper(2, 1, 100, keep_best=True)
    history = SimpleNamespace(checkpoints=[], evals=[])
    trainer = SimpleNamespace(skipped_steps=[])
    for step in (2, 4, 6, 8, 10):
        path = tmp_path / f"step_{step:08d}.pt"
        path.touch()
        history.checkpoints.append(str(path))
        history.evals.append({"step": step, "accuracy": 0.9 if step == 2 else 0.1})
        if step == 6:
            trainer.skipped_steps.append({"step": 5})
        keeper.prune(trainer, history)
    assert [s for s, _ in keeper.on_disk(history)] == [2, 4, 10]


def test_completed_run_retains_probeable_checkpoints(tmp_path: Path, tiny_manifest: Path) -> None:  # noqa: F811
    db = dbm.Database(db_path())
    spec = tiny_reads_state_spec(
        tmp_path,
        tiny_manifest,
        name="a5-retained",
        steps=8,
        eval_every=2,
        protocol={"version": 3, "checkpoint": "retained", "keep_every": 2, "keep_last": 4},
        arms=[
            {
                "name": "a5",
                "family": "hsn",
                "overrides": [
                    "model.routing=topk",
                    "train.gate_mode=ste",
                    "model.key_address=true",
                    "model.selected_readout=true",
                ],
            }
        ],
    )
    (exp_id,) = enqueue_and_claim(db, spec)
    assert run_child(db, exp_id, tmp_path / "runs") == 0
    row = db.row("SELECT json_path, record_json FROM results WHERE exp_id = ?", (exp_id,))
    assert row is not None
    run_dir = Path(row["json_path"]).parent
    record = json.loads(row["record_json"])
    assert [p["step"] for p in record["retained_checkpoints"]] == [2, 4, 6, 8]
    assert all(Path(p["path"]).is_file() for p in record["retained_checkpoints"])
    assert record["final"]["steps"] == 8
    assert record["best_checkpoint"]["step"] in (2, 4, 6, 8)
    before = Path(row["json_path"]).read_bytes()
    readonly = dbm.Database(db.path, read_only=True)
    probe = probe_rerun(readonly, exp_id, checkpoint_step=2, key_alphabet="train")
    assert Path(probe["checkpoint"]).name == "step_00000002.pt"
    assert probe["compared_with_stored"] is False
    # An explicit earlier step cannot be compared with the stored final probes, on dev keys too.
    assert not probe_rerun(readonly, exp_id, checkpoint_step=2)["compared_with_stored"]
    for split in ("train", "dev"):
        route = route_check(readonly, exp_id, checkpoint_step=2, split=split, n_docs=4)
        assert route["checkpoint"] == probe["checkpoint"]
    assert Path(row["json_path"]).read_bytes() == before
    with pytest.raises(FileNotFoundError, match="step 3"):
        _restored(readonly, exp_id, run_dir, cpu=True, checkpoint_step=3)
    with pytest.raises(ValueError, match="weights=checkpoint"):
        _restored(readonly, exp_id, run_dir, cpu=True, weights="dump", checkpoint_step=2)


def test_beta2_stage_is_matched_and_retention_does_not_change_arm_ids() -> None:
    original, _ = load_spec(REPO_ROOT / "configs/sweeps/p1_17_a5.yaml")
    proposed, _ = load_spec(REPO_ROOT / "configs/sweeps/p1_17_a5_b095.yaml")
    old = {a.label: a for a in expand(original)}
    new = expand(proposed)
    assert len(new) == 4
    for arm in new:
        reference = old[arm.label]
        expected = reference.config.model_dump(exclude={"run_id"})
        expected["train"]["betas"] = (0.9, 0.95)
        assert arm.config.model_dump(exclude={"run_id"}) == expected
    final_only = proposed.model_copy(
        update={"protocol": {**proposed.protocol, "checkpoint": "final", "keep_every": 0}}
    )
    assert [a.arm_id for a in expand(final_only)] == [a.arm_id for a in new]


def test_recovery_stage_has_independent_learning_rate_ablations() -> None:
    spec, _ = load_spec(REPO_ROOT / "configs/sweeps/p1_17_a5_recovery.yaml")
    arms = expand(spec)
    assert len(arms) == 8
    for arm in arms:
        cfg = arm.config
        assert cfg.train.slow_burst_factor == 10 and cfg.train.rollback_every == 50
        assert cfg.train.spike_factor == 100 and cfg.train.betas == (0.9, 0.95)
        assert not (cfg.train.router_lr_mult != 1 and cfg.train.readout_lr_mult != 1)
        assert cfg.train.lambda_u == 0  # no extra objective
