"""P1.15 tests: the first/later split of the delayed/in_range dev accuracy.

The query slots are parsed from the documents (each loss position follows ``QRY`` and is a key,
slot 0 asks the first fact); the evaluation labels its own per-position values by slot without
changing the frozen metric (the result without position groups is today's, and the slots sum
exactly to the group); the slots survive documents that span windows; every reads_state record
carries the split, which the rules, ``report`` and ``status`` read; the derived backfill reproduces
the audit's hand reconstruction and the recomputed one the run's own split; the read-only database
refuses writes.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hsn.data.dataset import ByteDataset
from hsn.device import seed_everything
from hsn.experiments import db as dbm
from hsn.experiments import split
from hsn.experiments.cli import main as cli_main
from hsn.experiments.protocols import ArmRun, run_protocol
from hsn.experiments.report import build_report
from hsn.experiments.rules import RecordView, evaluate_verdicts, parse_rules
from hsn.experiments.spec import protocol_params
from hsn.runner import build_model, make_trainer
from p1_11_support import (
    REF_A1,
    TINY_HSN_OVERRIDES,
    cpu_ctx,
    db_path,
    disk_overrides,
    docs,
    isolated,  # noqa: F401  (fixture)
    restore_determinism,  # noqa: F401  (fixture)
    tiny_gru_config,
    tiny_hsn_config,
    tiny_manifest,  # noqa: F401  (fixture)
)
from p1_14_support import enqueue_and_claim, run_child, tiny_reads_state_spec


def _dev() -> ByteDataset:
    return ByteDataset(docs("kv", "dev", 6) + docs("delayed", "dev", 30) + docs("kv", "dev", 6))


def test_p1_15_query_slots_parse() -> None:
    """Every delayed document's loss positions are its queries in order; slot 0 asks the first fact;
    kv documents are ``other``; a malformed document raises."""
    dev = _dev()
    slots = split.QuerySlots(dev)
    n_delayed = sum(1 for i in range(len(dev)) if dev.meta(i)["task"] == "delayed")
    assert slots.counts["first"] == n_delayed
    assert slots.counts["other"] == sum(
        int(np.asarray(dev.examples[i].loss_mask).sum())
        for i in range(len(dev))
        if dev.meta(i)["task"] != "delayed"
    )
    later = sum(slots.counts[n] for n in split.LATER_SLOTS)
    assert later > 0, "the tiny delayed documents ask later queries too"
    ex = docs("delayed", "dev", 1)[0]
    assert split.query_slots(ex.tokens, ex.loss_mask) == list(range(int(ex.loss_mask.sum())))
    bad = ex.loss_mask.copy()
    bad[int(np.flatnonzero(bad)[0]) - 1] = True  # the QRY position itself is not a key
    with pytest.raises(ValueError, match="not the key"):
        split.query_slots(ex.tokens, bad)
    with pytest.raises(ValueError, match="no delayed/in_range"):
        split.QuerySlots(ByteDataset(docs("kv", "dev", 4)))


def _trainer(cfg: Any) -> Any:
    seed_everything(cfg.seed)
    model = build_model(cfg)
    return make_trainer(model, cfg, cpu_ctx(), run_dir=None)


@pytest.mark.usefixtures("restore_determinism")
@pytest.mark.parametrize("family", ["hsn", "gru"])
def test_p1_15_split_matches_the_frozen_metric(family: str) -> None:
    """Evaluation with the query slots returns today's result plus ``per_position_group``, and the
    slots sum exactly to the delayed/in_range group (positions, correct answers, loss)."""
    cfg = tiny_hsn_config() if family == "hsn" else tiny_gru_config()
    dev = _dev()
    trainer = _trainer(cfg)
    plain = trainer.evaluate(dev)
    trainer.eval_position_groups = split.QuerySlots(dev)
    labelled = trainer.evaluate(dev)
    groups = labelled.pop("per_position_group")
    assert labelled == plain  # the frozen metric, bit for bit
    group = plain["per_group"]["delayed/in_range"]
    block = split.split_block(groups, group, total_positions=plain["n_positions"])
    assert block["first"]["n"] + block["later"]["n"] == group["n_positions"]
    hits = block["first"]["accuracy"] * block["first"]["n"]
    hits += block["later"]["accuracy"] * block["later"]["n"]
    assert round(hits) == round(group["accuracy"] * group["n_positions"])
    assert block["later"]["n"] == sum(block[n]["n"] for n in split.LATER_SLOTS)


def test_p1_15_split_block_refuses_a_mismatch() -> None:
    groups = {
        "first": {"accuracy": 1.0, "loss": 0.1, "n_positions": 2},
        "later/1": {"accuracy": 0.5, "loss": 0.7, "n_positions": 2},
    }
    good = {"accuracy": 0.75, "loss": 0.4, "n_positions": 4}
    block = split.split_block(groups, good)
    assert block["later"] == {"accuracy": 0.5, "loss": 0.7, "n": 2}
    assert block["later/2"] == {"accuracy": None, "loss": None, "n": 0}
    with pytest.raises(ValueError, match="correct answers"):
        split.split_block(groups, {**good, "accuracy": 0.5})
    with pytest.raises(ValueError, match="slot positions"):
        split.split_block(groups, {**good, "n_positions": 5})


@pytest.mark.usefixtures("restore_determinism")
def test_p1_15_split_counts_documents_across_windows() -> None:
    """With a window shorter than the documents, a document's queries arrive in two windows; the
    labeller counts them per document, so the slot counts are those of whole documents."""
    dev = _dev()
    trainer = _trainer(tiny_hsn_config("train.unroll=16"))
    trainer.eval_position_groups = split.QuerySlots(dev)
    result = trainer.evaluate(dev)
    counts = split.QuerySlots(dev).counts
    for name in split.NAMES:
        assert result["per_position_group"][name]["n_positions"] == counts[name], name


def _cfg(manifest: Path) -> Any:
    from hsn.config import load_config

    return load_config(REF_A1, [*TINY_HSN_OVERRIDES, *disk_overrides(manifest, steps=4)])


@pytest.mark.usefixtures("restore_determinism")
def test_p1_15_split_in_every_reads_state_record(tmp_path: Path, tiny_manifest: Path) -> None:  # noqa: F811
    """A reads_state record has the split in every evaluation and in ``final``; the first-query
    count is the probe's document count."""
    params = protocol_params({"name": "reads_state", "diag_every": 2, "probe_batch": 8})
    arm = ArmRun(
        cfg=_cfg(tiny_manifest),
        family="hsn",
        variant="pdg",
        variant_params={},
        params=params,
        run_dir=tmp_path / "run",
        rules=parse_rules(None),
        constants={},
    )
    record = run_protocol(arm)
    assert record["evals"] and all("split" in e for e in record["evals"])
    final = record["final"]
    for g in split.split_names():
        assert f"split/{g}/accuracy" in final and f"split/{g}/n" in final
    assert final["split/first/n"] == record["probe"]["n_probed"]
    view = RecordView(record)
    series = view.series("eval", "split/later/accuracy")
    assert set(series) == {e["step"] for e in record["evals"]}
    assert series[final["eval_step"]] == final["split/later/accuracy"]


def _record(later: float) -> dict[str, Any]:
    return {
        "protocol": "reads_state",
        "evals": [
            {"step": 4, "accuracy": 0.8, "loss": 1.0, "per_group": {}, "split": {"later": {}}},
            {
                "step": 8,
                "accuracy": 0.8,
                "loss": 1.0,
                "per_group": {},
                "split": {"later": {"accuracy": later, "loss": 1.0, "n": 10}},
            },
        ],
        "final": {"split/later/accuracy": later},
    }


def test_p1_15_split_rules() -> None:
    rules = parse_rules(
        {
            "verdicts": {
                "keyed": {
                    "cases": [
                        {
                            "label": "above the key-blind bound",
                            "when": {
                                "metric": "eval:split/later/accuracy",
                                "steps": {"at": 8},
                                "op": ">",
                                "value": 0.623,
                            },
                        }
                    ],
                    "default": "at or below",
                },
                "final_keyed": {
                    "cases": [
                        {
                            "label": "above",
                            "when": {
                                "metric": "final:split/later/accuracy",
                                "op": ">",
                                "value": 0.623,
                            },
                        }
                    ],
                    "default": "below",
                },
            }
        }
    )
    high = evaluate_verdicts(rules, _record(0.7), {}, "hsn")
    low = evaluate_verdicts(rules, _record(0.6), {}, "hsn")
    assert high["keyed"]["label"] == "above the key-blind bound"
    assert low["keyed"]["label"] == "at or below"
    assert high["final_keyed"]["label"] == "above" and low["final_keyed"]["label"] == "below"
    missing = _record(0.7)
    del missing["evals"][1]["split"]
    assert evaluate_verdicts(rules, missing, {}, "hsn")["keyed"]["label"].startswith("n/a")


def test_p1_15_derived_split_is_the_hand_reconstruction() -> None:
    """AUDIT.md (8 Nov 02:52): first / later of exps 136, 138 and 139 from the dev accuracy and the
    probe's first-query accuracy over 1,000 first and 1,126 later queries."""
    counts = {"first": 1000, "later": 1126}
    for answers, first, later in ((1682, 1.0, 0.606), (1623, 1.0, 0.553), (1184, 0.889, 0.262)):
        got = split.derived_split(answers / 2126, first, counts)
        assert got["first_acc"] == first and round(got["later_acc"], 3) == later
        assert got["later_1"] is None


@pytest.mark.usefixtures("isolated", "restore_determinism")
def test_p1_15_report_status_and_backfill(
    tmp_path: Path,
    tiny_manifest: Path,  # noqa: F811
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A stored reads_state run: ``report`` and ``status`` show its split; with the split stripped
    from its stored record (a run from before P1.15), the read-only backfill derives it and, from
    the kept checkpoint, recomputes exactly the run's own split; ``report --split-backfill`` fills
    the columns from the file."""
    db = dbm.Database(db_path())
    spec = tiny_reads_state_spec(tmp_path, tiny_manifest, name="split", steps=4)
    (exp_id,) = enqueue_and_claim(db, spec)
    assert run_child(db, exp_id, tmp_path / "runs") == 0
    row = db.row("SELECT record_json FROM results WHERE exp_id = ?", (exp_id,))
    assert row is not None
    record = json.loads(row["record_json"])
    final = record["final"]
    rows = build_report(db, per_seed=True).rows
    assert rows[0]["first_acc"] == final["split/first/accuracy"]
    assert rows[0]["split_src"] == "record"
    assert "first_acc" in build_report(db).columns
    assert cli_main(["status"]) == 0
    out = capsys.readouterr().out
    assert "first/later" in out
    assert f"{final['split/first/accuracy']:.3f}/{final['split/later/accuracy']:.3f}" in out

    # a run from before P1.15: no split in its stored record
    old = {**record, "final": {k: v for k, v in final.items() if not k.startswith("split/")}}
    old["evals"] = [{k: v for k, v in e.items() if k != "split"} for e in record["evals"]]
    with db.write() as conn:
        conn.execute(
            "UPDATE results SET record_json = ? WHERE exp_id = ?", (json.dumps(old), exp_id)
        )
    ro = dbm.Database(db_path(), read_only=True)
    out_file = tmp_path / "backfill.json"
    assert cli_main(["split-backfill", "--recompute", "--out", str(out_file)]) == 0
    backfill = json.loads(out_file.read_text())
    (entry,) = backfill["runs"]
    assert entry["tier"] == "recomputed" and out_file.with_suffix(".md").is_file()
    got = entry["recomputed"]
    assert got["split"] == record["evals"][-1]["split"]  # the run's own split, recomputed
    assert got["overall_minus_stored_answers"] == 0
    derived = entry["derived"]
    assert derived["first_acc"] == record["probe"]["first_query_accuracy"]
    assert math.isfinite(derived["later_acc"])
    assert cli_main(["report", "--per-seed", "--split-backfill", str(out_file)]) == 0
    assert "recomputed" in capsys.readouterr().out
    rows = build_report(ro, per_seed=True, split_backfill={exp_id: entry}).rows
    assert rows[0]["split_src"] == "recomputed" and rows[0]["later_1"] == got["later_1"]


def test_p1_15_database_read_only(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite"
    with pytest.raises(dbm.DatabaseError, match="no database"):
        dbm.Database(path, read_only=True, allow_tmp=True)
    assert not path.exists()
    dbm.Database(path, allow_tmp=True).close()
    ro = dbm.Database(path, read_only=True, allow_tmp=True)
    assert ro.version() == dbm.SCHEMA_VERSION
    with pytest.raises(dbm.DatabaseError, match="read-only"), ro.write():
        pass
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ro.conn.execute("INSERT INTO meta (key, value) VALUES ('x', 'y')")
