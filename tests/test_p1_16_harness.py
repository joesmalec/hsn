"""P1.16 tests: the sweep harness's kept checkpoints and partial results, the rollback in the
record, and the read-only tools (``probe-rerun --weights dump`` and ``--key-alphabet train``,
``route-check``) on tiny arms trained on the CPU through the real child process path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hsn.experiments import db as dbm
from hsn.experiments.probe import delayed_docs
from hsn.experiments.probe_check import DEV_KEYS, TRAIN_KEYS_REMAP, probe_rerun, train_key_alphabet
from hsn.experiments.report import build_report
from hsn.experiments.routing import doc_overlaps, route_check, summarize_route, verdict_route
from hsn.experiments.spec import expand, load_spec
from p1_11_support import (  # noqa: F401  (fixtures)
    db_path,
    fake_repo,
    isolated,
    restore_determinism,
    tiny_manifest,
)
from p1_14_support import enqueue_and_claim, plant_inf_gradient, run_child, tiny_reads_state_spec

pytestmark = pytest.mark.usefixtures("isolated", "restore_determinism")

RULES = {"verdicts": {"reads_state": {"preset": "reads_state"}}}


def _result(db: dbm.Database, exp_id: int) -> dict[str, Any]:
    row = db.row("SELECT * FROM results WHERE exp_id = ?", (exp_id,))
    assert row is not None
    return dict(row)


def _run_dir(db: dbm.Database, exp_id: int) -> Path:
    """The run directory (the stored result's; a test's child writes no run_dir column)."""
    return Path(_result(db, exp_id)["json_path"]).parent


# --------------------------------------------------------------------------------------------------
# Protocol fields
# --------------------------------------------------------------------------------------------------


def test_p1_16_keep_every_execution_only(tmp_path: Path, tiny_manifest: Path) -> None:  # noqa: F811
    """``keep_every`` / ``keep_last`` change no arm id (execution-only)."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain = tiny_reads_state_spec(tmp_path / "a", tiny_manifest, name="k0")
    kept = tiny_reads_state_spec(
        tmp_path / "b", tiny_manifest, name="k0", protocol={"keep_every": 2, "keep_last": 3}
    )
    repo = fake_repo(tmp_path)
    ids = [[a.arm_id for a in expand(load_spec(p)[0], repo=repo)] for p in (plain, kept)]
    assert ids[0] == ids[1]


# --------------------------------------------------------------------------------------------------
# A failed run re-scored from its kept checkpoint
# --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def failed_run(
    tmp_path_factory: pytest.TempPathFactory,
    tiny_manifest: Path,  # noqa: F811
) -> dict[str, Any]:
    """A tiny reads_state arm (version 3, keep_every 4, keep_last 1, 30 steps) whose gradients are
    infinite from step 9: P1.12's stop rule ends it after ten skips."""
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("failed")
    # the isolated fixture's settings, for this module-scoped run
    mp.setenv("HSN_EXPERIMENTS_DB", str(tmp / "experiments.sqlite"))
    mp.setenv("HSN_EXPERIMENTS_ROOT", str(tmp / "runs"))
    mp.setenv("HSN_EXPERIMENTS_ALLOW_TMP", "1")
    mp.setenv("HSN_EXPERIMENTS_FAKE_MACHINE", "1")
    mp.setenv("CUDA_VISIBLE_DEVICES", "")
    mp.setenv("OMP_NUM_THREADS", "2")
    try:
        db = dbm.Database(tmp / "experiments.sqlite", allow_tmp=True)
        spec = tiny_reads_state_spec(
            tmp,
            tiny_manifest,
            name="p116fail",
            steps=30,
            eval_every=4,
            protocol={"version": 3, "keep_every": 4, "keep_last": 1},
            rules=RULES,
        )
        (exp_id,) = enqueue_and_claim(db, spec)
        plant_inf_gradient(mp, from_call=9)
        code = run_child(db, exp_id, tmp / "runs")
    finally:
        mp.undo()
    return {"db": db, "exp_id": exp_id, "code": code, "tmp": tmp}


def test_p1_16_failed_run_partial_from_checkpoint(failed_run: dict[str, Any]) -> None:
    db, exp_id = failed_run["db"], failed_run["exp_id"]
    assert failed_run["code"] == 1 and db.experiment(exp_id)["status"] == "failed"
    record = json.loads(_result(db, exp_id)["record_json"])
    assert record["failure"]["kind"] == "nonfinite"
    partial = record["partial"]
    assert partial["from_step"] == 8 and Path(partial["checkpoint"]).name == "step_00000008.pt"
    assert "first guard event (step 9)" in partial["reason"] and "error" not in partial
    assert partial["final"]["eval_step"] == 8 and "split/later/accuracy" in partial["final"]
    assert set(partial["probes"]) == {"probe", "probe_v2", "probe_v3"}
    assert "gate/write_rate" in partial["trained"]
    # pruning: the pinned checkpoint before the first event (8) and the newest kept one (16) stay
    run_dir = _run_dir(db, exp_id)
    assert sorted(p.name for p in run_dir.glob("step_*.pt")) == [
        "step_00000008.pt",
        "step_00000016.pt",
    ]
    label = json.loads(_result(db, exp_id)["verdicts_json"])["reads_state"]["label"]
    assert label.startswith("failed: non-finite gradients (P1.12 stop) @ ")
    assert "(from step 8: " in label
    report = build_report(db, sweep="p116fail", per_seed=True)
    (row,) = report.rows
    assert "partial from step 8 (dev " in row["failure"]


def test_p1_16_probe_rerun_from_dump(failed_run: dict[str, Any]) -> None:
    """``probe-rerun --weights dump`` gives a failed run without stored probes its verdicts from
    the divergence dump (the state before the first skipped step), with the split."""
    db = dbm.Database(failed_run["db"].path, read_only=True)
    out = probe_rerun(db, failed_run["exp_id"], weights="dump")
    assert Path(out["checkpoint"]).name == "divergence_00000009.pt"
    assert out["weights"] == "dump" and out["compared_with_stored"] is False
    assert out["v1_summary_equal"] is None and out["v3"]["swap/verdict"]
    assert 0.0 <= out["eval"]["later_acc"] <= 1.0
    remapped = probe_rerun(db, failed_run["exp_id"], weights="dump", key_alphabet="train")
    assert remapped["key_alphabet"] == "train" and "not a metric" in remapped["diagnostic"]
    from hsn.experiments.probe_check import _restored

    empty = _run_dir(db, failed_run["exp_id"]).parent / "empty"
    empty.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError, match="no divergence dump"):
        _restored(db, failed_run["exp_id"], empty, cpu=False, weights="dump")


def test_p1_16_train_key_alphabet(failed_run: dict[str, Any]) -> None:
    from hsn.config import RunConfig
    from hsn.experiments.db import loads
    from hsn.runner import load_datasets

    db, exp_id = failed_run["db"], failed_run["exp_id"]
    exp = db.experiment(exp_id)
    cfg = RunConfig.model_validate(loads(exp["config_json"]))
    _train, dev = load_datasets(cfg)
    remapped = train_key_alphabet(dev)
    assert len(remapped) == len(dev)
    for before, after in zip(dev.examples, remapped.examples, strict=True):
        if before.meta["task"] != "delayed":
            assert after is before
            continue
        assert after.meta["key_alphabet"] == "train"
        assert not set(after.tokens.tolist()) & set(DEV_KEYS)
        back = after.tokens.copy()
        moved = (back >= TRAIN_KEYS_REMAP.start) & (back < TRAIN_KEYS_REMAP.stop)
        back[moved] += DEV_KEYS.start - TRAIN_KEYS_REMAP.start
        assert np.array_equal(back, before.tokens)  # a bijection on the document's bytes
        assert np.array_equal(after.loss_mask, before.loss_mask)
    docs = delayed_docs(remapped)  # every remapped document still parses
    assert docs


# --------------------------------------------------------------------------------------------------
# The rollback and kept checkpoints in finished runs
# --------------------------------------------------------------------------------------------------


def test_p1_16_finished_run_drops_kept(tmp_path: Path, tiny_manifest: Path) -> None:  # noqa: F811
    """A finished run deletes its kept checkpoints (the final one stays); no partial block."""
    db = dbm.Database(db_path())
    spec = tiny_reads_state_spec(
        tmp_path, tiny_manifest, name="p116ok", steps=8, protocol={"keep_every": 2}
    )
    (exp_id,) = enqueue_and_claim(db, spec)
    assert run_child(db, exp_id, tmp_path / "runs") == 0
    run_dir = _run_dir(db, exp_id)
    assert [p.name for p in run_dir.glob("step_*.pt")] == ["step_00000008.pt"]
    record = json.loads(_result(db, exp_id)["record_json"])
    assert "partial" not in record and record["final"]["rollbacks"] is None
    assert record["rollbacks"] == []


# --------------------------------------------------------------------------------------------------
# route-check
# --------------------------------------------------------------------------------------------------


def test_p1_16_route_overlap_exact() -> None:
    sets = [frozenset(s) for s in ({0, 1}, {0, 1}, {2, 3}, {4, 5}, {0, 2}, {6, 7}, {0, 1})]
    facts = [(64, 150, 2), (65, 151, 4)]  # keys at 1 and 3, values at 2 and 4
    queries = [(0, 0, 6)]  # the query at 6 asks fact 0
    (entry,) = doc_overlaps(sets, facts, queries, k=2)
    assert entry["o_key"] == 1.0 and entry["o_val"] == 0.0  # S(6) = S(1) = {0, 1}
    assert entry["o_other_key"] == 0.0 and entry["o_other_val"] == 0.5  # S(3), S(4) = {4,5}, {0,2}
    assert entry["d_key"] == 1.0 and entry["d_val"] == -0.5


def test_p1_16_route_verdicts_planted() -> None:
    """Routing that follows the key reads "follows the key"; key-independent routing reads "does
    not follow the key"; the swap control can veto a follow."""
    rng = np.random.default_rng(0)
    k, n_cells = 4, 32
    follow: list[list[dict[str, Any]]] = []
    blind: list[list[dict[str, Any]]] = []
    for _ in range(40):
        keys = rng.choice(np.arange(64, 112), size=3, replace=False)
        cells = {int(key): frozenset(rng.choice(n_cells, size=k, replace=False)) for key in keys}
        filler = [frozenset(rng.choice(n_cells, size=k, replace=False)) for _ in range(12)]
        facts = [(int(key), 150, 2 * i + 2) for i, key in enumerate(keys)]
        # follows: the fact's key position and the query of the same key route alike
        sets = list(filler)
        for key, _v, vpos in facts:
            sets[vpos - 1] = cells[key]
        sets[10] = cells[facts[1][0]]
        follow.append(doc_overlaps(sets, facts, [(1, 1, 10)], k))
        blind.append(doc_overlaps(list(filler), facts, [(1, 1, 10)], k))
    good = summarize_route(follow, [{"o_swap_new": 1.0, "o_swap_old": 0.1}] * 10, k)
    assert good["verdict"] == "follows the key" and good["all"]["d_key"]["low"] > 0.125
    assert summarize_route(blind, [], k)["verdict"] == "does not follow the key"
    vetoed = summarize_route(follow, [{"o_swap_new": 0.1, "o_swap_old": 0.1}] * 10, k)
    assert vetoed["verdict"] == "weakly key-dependent"
    assert verdict_route(good) == "follows the key"


def test_p1_16_route_check_topk_arm(tmp_path: Path, tiny_manifest: Path) -> None:  # noqa: F811
    """``route-check`` on a tiny top-K arm's checkpoint (read-only): a summary with a verdict, on
    the dev documents and on the training documents; a dense arm is refused."""
    db = dbm.Database(db_path())
    topk = [
        {
            "name": "t",
            "family": "hsn",
            "overrides": ["model.routing=topk", "train.gate_mode=ste", "model.k_select=2"],
        }
    ]
    spec = tiny_reads_state_spec(tmp_path, tiny_manifest, name="p116route", steps=4, arms=topk)
    (exp_id,) = enqueue_and_claim(db, spec)
    assert run_child(db, exp_id, tmp_path / "runs") == 0
    ro = dbm.Database(db.path, read_only=True)
    out = route_check(ro, exp_id, batch_rows=8)
    assert out["split"] == "dev" and out["k"] == 2 and out["n_docs"] > 0
    assert out["verdict"] in ("follows the key", "weakly key-dependent", "does not follow the key")
    assert out["all"]["n_queries"] > 0 and out["swap"]["n"] >= 0
    train = route_check(ro, exp_id, split="train", n_docs=8, batch_rows=8)
    assert train["n_docs"] <= 8 and "in-sample" in train["keys"]
    (tmp_path / "d").mkdir()
    dense = tiny_reads_state_spec(tmp_path / "d", tiny_manifest, name="p116dense", steps=2)
    (dense_id,) = enqueue_and_claim(db, dense)
    assert run_child(db, dense_id, tmp_path / "runs") == 0
    with pytest.raises(ValueError, match="routed model"):
        route_check(ro, dense_id)
