"""Regression coverage for synthetic_v2 reads-state and baseline probe checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hsn.config import load_config
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import DelayedParams, generate, load_dataset_spec
from hsn.experiments import split
from hsn.experiments.probe import edit_probe
from hsn.experiments.protocols import ArmRun, run_protocol
from hsn.experiments.report import SPLIT_COLUMNS, _split_cells
from hsn.experiments.rules import parse_rules
from hsn.experiments.spec import expand, load_spec, protocol_params
from hsn.runner import build_model, load_datasets, make_trainer
from hsn.train.checkpoint import load_checkpoint
from p1_11_support import (
    REPO,
    TINY_GRU_OVERRIDES,
    TINY_TRANSFORMER_OVERRIDES,
    cpu_ctx,
    disk_overrides,
    fake_repo,
    tiny_gru_config,
    tiny_manifest,  # noqa: F401
)


def _v2_data() -> tuple[ByteDataset, ByteDataset]:
    spec = load_dataset_spec(REPO / "configs/data/synthetic_v2.yaml")
    delayed = spec.task("delayed", "in_range")
    kv = spec.task("kv", "in_range")
    # Actual failing document and an actual eight-query document from the frozen v2 spec.
    dev = [generate(delayed, "dev", i) for i in (1, 457)]
    assert [int(ex.loss_mask.sum()) for ex in dev] == [5, 8]
    return (
        ByteDataset([generate(delayed, "train", i) for i in range(4)]),
        ByteDataset([generate(kv, "dev", 0), *dev, generate(kv, "dev", 1)]),
    )


def test_v2_slots_preserve_the_metric_across_windows() -> None:
    _, dev = _v2_data()
    assert DelayedParams().max_queries == 4, "do not change the frozen v1 generator"
    slots = split.QuerySlots(dev)
    assert slots.counts["first"] == 2 and slots.counts["later/7"] == 1
    assert slots.counts["other"] == sum(
        int(ex.loss_mask.sum()) for ex in dev.examples if ex.meta["task"] == "kv"
    )
    for ex in dev.examples[1:3]:
        assert split.query_slots(ex.tokens, ex.loss_mask) == list(range(int(ex.loss_mask.sum())))
    cfg = tiny_gru_config("train.unroll=16")
    trainer = make_trainer(build_model(cfg), cfg, cpu_ctx(), run_dir=None)
    plain = trainer.evaluate(dev)
    trainer.eval_position_groups = slots
    for _ in range(2):  # each evaluation gets a fresh counter across windows
        labelled = trainer.evaluate(dev)
        groups = labelled.pop("per_position_group")
        assert labelled == plain
        for name, count in slots.counts.items():
            assert groups[name]["n_positions"] == count
        block = split.split_block(
            groups, plain["per_group"]["delayed/in_range"], total_positions=plain["n_positions"]
        )
        assert block["first"]["n"] == 2 and block["later"]["n"] == 11
        assert block["later/7"]["n"] == 1
    assert split.split_counts(dev)["later"] == 11


def test_v2_reads_state_v3_finishes_training_evaluation_and_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, dev = _v2_data()
    # Tiny real v2 samples, preserving all trainer/evaluator/probe code paths.
    monkeypatch.setattr("hsn.runner.load_datasets", lambda cfg: (train, dev))
    cfg = tiny_gru_config("train.unroll=256", "data.tasks=[kv,delayed]", steps=2)
    params = protocol_params(
        {
            "name": "reads_state",
            "version": 3,
            "diagnostics": False,
            "diag_docs": 2,
            "eval_batch": 4,
            "probe_batch": 4,
        }
    )
    arm = ArmRun(
        cfg=cfg,
        family="gru",
        variant="none",
        variant_params={},
        params=params,
        run_dir=tmp_path,
        rules=parse_rules(None),
        constants={},
    )
    record = run_protocol(arm)
    assert arm.stage == "done" and record["final"]["steps"] == 2
    assert record["probe"]["n_probed"] == 2
    assert record["probe_v2"] and record["probe_v3"]
    final = record["final"]
    assert final["split/later/7/n"] == 1 and final["split/later/n"] == 11
    columns = _split_cells(record, 1, None)
    for q in range(1, 8):
        assert SPLIT_COLUMNS[f"later_{q}"] == f"split/later/{q}/accuracy"
        assert columns[f"later_{q}"] == final[f"split/later/{q}/accuracy"]


def test_new_slots_are_weighted_in_later_aggregation_and_backfill() -> None:
    groups = {"first": {"accuracy": 1.0, "loss": 0.0, "n_positions": 2}}
    for q in range(1, 8):
        groups[f"later/{q}"] = {"accuracy": 1 / q, "loss": float(q), "n_positions": q}
    total = {"n_positions": 30, "accuracy": 9 / 30, "loss": 140 / 30}
    block = split.split_block(groups, total, total_positions=30)
    assert block["later"] == {"accuracy": 7 / 28, "loss": 5.0, "n": 28}
    derived = split.derived_split(9 / 30, 1.0, {"first": 2, "later": 28})
    assert derived["later_acc"] == 7 / 28
    assert all(derived[f"later_{q}"] is None for q in range(1, 8))
    recomputed = {f"later_{q}": q / 10 for q in range(1, 8)}
    table = split.backfill_table(
        {"runs": [{"exp_id": 1, "label": "v2", "family": "gru", "recomputed": recomputed}]}
    )
    assert "0.400 / 0.500 / 0.600 / 0.700" in table


def test_invalid_extended_query_slots_still_raise() -> None:
    from hsn.data.synthetic import QRY

    tokens = np.asarray([112, 144] + [QRY, 112, 144] * 9, dtype=np.uint8)
    mask = np.zeros_like(tokens, dtype=bool)
    mask[3::3] = True
    with pytest.raises(ValueError, match="more than max_queries=8"):
        split.query_slots(tokens, mask)
    # Eight slots still validate query markers and the first fact, not just the count.
    tokens, mask = tokens[:-3].copy(), mask[:-3].copy()
    tokens[-3] = 0
    with pytest.raises(ValueError, match="not the key"):
        split.query_slots(tokens, mask)


@pytest.mark.parametrize("family", ["gru", "transformer"])
def test_baseline_sweep_checkpoints_survive_success_and_load_for_probes(
    family: str,
    tmp_path: Path,
    tiny_manifest: Path,  # noqa: F811
) -> None:
    spec, _ = load_spec(REPO / "configs/sweeps/baselines_27k.yaml")
    expanded = next(a for a in expand(spec, repo=fake_repo(tmp_path)) if a.family == family)
    assert expanded.protocol.checkpoint == "final"  # supported by the pinned old runtime
    scale = 4500
    tiny = TINY_GRU_OVERRIDES if family == "gru" else TINY_TRANSFORMER_OVERRIDES
    cfg = load_config(
        REPO / expanded.config_path,
        [
            *expanded.overrides,
            *tiny,
            *disk_overrides(tiny_manifest, steps=6),
            f"train.ckpt_every={expanded.config.train.ckpt_every // scale}",
        ],
    )
    params = expanded.protocol.model_copy(
        update={
            "steps": expanded.steps // scale,
            "keep_every": expanded.protocol.keep_every // scale,
            "diagnostics": False,
            "diag_docs": 2,
            "eval_batch": 4,
            "probe_batch": 4,
        }
    )
    arm = ArmRun(
        cfg=cfg,
        family=family,
        variant="none",
        variant_params={},
        params=params,
        run_dir=tmp_path / "run",
        rules=parse_rules(None),
        constants={},
    )
    record = run_protocol(arm)  # includes successful-run checkpoint cleanup
    assert arm.stage == "done" and record["final"]["steps"] == 6
    paths = [arm.run_dir / f"step_{i:08d}.pt" for i in range(1, 7)]
    assert all(p.is_file() for p in paths), "the earlier probe checkpoints were deleted"
    assert set(record["checkpoints"]) == {str(p) for p in paths}
    restored = build_model(cfg)
    load_checkpoint(paths[0], restored)
    _, dev = load_datasets(cfg)
    assert edit_probe(restored, cpu_ctx(), dev, batch_rows=4)
