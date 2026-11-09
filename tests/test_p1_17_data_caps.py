"""Capped experiment runs retain all task/variant groups and can reconstruct their dev set."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.data.dataset import ByteDataset
from hsn.experiments import db as dbm
from hsn.experiments.data_selection import PREFIX, STRATIFIED, cap_dataset, dataset_counts
from hsn.experiments.probe_check import _restored
from hsn.experiments.protocols import ArmRun, run_protocol
from hsn.experiments.rules import parse_rules
from hsn.experiments.spec import protocol_params
from hsn.runner import load_datasets
from hsn.train.divergence import load_dump, replay
from hsn.train.loop import StreamLoader
from p1_11_support import (
    db_path,
    disk_overrides,
    docs,
    isolated,  # noqa: F401
    restore_determinism,  # noqa: F401
    tiny_gru_config,
    tiny_manifest,  # noqa: F401
)
from p1_14_support import enqueue_and_claim, run_child, tiny_reads_state_spec


def _ids(dataset: ByteDataset) -> list[str]:
    return [str(example.meta["doc_id"]) for example in dataset.examples]


def test_p1_17_caps_cover_every_task_and_variant_without_duplicates() -> None:
    first = docs("kv", "train", 30)
    second = docs("delayed", "train", 30)
    # Only metadata matters to sampling; a second variant of the same task must be separate.
    third = [
        dataclasses.replace(
            ex,
            meta={**ex.meta, "variant": "out_of_range", "doc_id": str(ex.meta["doc_id"]) + ":long"},
        )
        for ex in docs("kv", "train", 30)
    ]
    dataset = ByteDataset([*first, *second, *third])
    selected = cap_dataset(dataset, 12, seed=19)
    groups = Counter((ex.meta["task"], ex.meta["variant"]) for ex in selected.examples)
    assert set(groups.values()) == {4} and len(groups) == 3
    assert len(_ids(selected)) == len(set(_ids(selected))) == 12
    original_order = {doc_id: index for index, doc_id in enumerate(_ids(dataset))}
    indices = [original_order[doc_id] for doc_id in _ids(selected)]
    assert indices == sorted(indices)
    assert _ids(cap_dataset(dataset, 12, seed=19)) == _ids(selected)
    assert _ids(cap_dataset(dataset, 12, seed=20)) != _ids(selected)
    assert cap_dataset(dataset, 0, seed=19) is dataset
    assert cap_dataset(dataset, 100, seed=19) is dataset
    with pytest.raises(ValueError, match="cannot cover all 3"):
        cap_dataset(dataset, 2, seed=19)


def test_p1_17_caps_preserve_small_groups_and_total_budget() -> None:
    dataset = ByteDataset([*docs("kv", "train", 1), *docs("delayed", "train", 40)])
    selected = cap_dataset(dataset, 8, seed=0)
    assert Counter(ex.meta["task"] for ex in selected.examples) == {"kv": 1, "delayed": 7}
    summary = dataset_counts(selected)
    assert summary["n_docs"] == 8
    assert summary["n_tokens"] == sum(ex.length for ex in selected.examples)
    assert summary["n_loss"] == sum(int(ex.loss_mask.sum()) for ex in selected.examples)
    assert summary["groups"]["kv/in_range"]["n_docs"] == 1


@pytest.mark.usefixtures("restore_determinism")
@pytest.mark.parametrize("overfit", [False, True])
def test_p1_17_protocol_records_full_and_effective_counts(
    overfit: bool,
    tiny_manifest: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    cfg = tiny_gru_config(*disk_overrides(tiny_manifest, steps=2, eval_every=2), steps=2)
    params = protocol_params(
        {"name": "overfit", "n_docs": 4, "diag_every": 0}
        if overfit
        else {"name": "train_eval", "n_train": 8, "n_dev": 4, "diag_every": 0}
    )
    arm = ArmRun(
        cfg=cfg,
        family="gru",
        variant="none",
        variant_params={},
        params=params,
        run_dir=tmp_path / "run",
        rules=parse_rules(None),
        constants={},
    )
    record = run_protocol(arm)
    selection = record["data_selection"]
    assert selection["policy"] == (PREFIX if overfit else STRATIFIED)
    assert selection["train"]["full"]["n_docs"] == 128
    assert selection["train"]["effective"]["n_docs"] == (4 if overfit else 8)
    assert selection["dev"]["effective"]["n_docs"] == 4
    if overfit:
        assert selection["dev"]["source"] == "train"
        assert set(record["evals"][-1]["per_group"]) == {"kv/in_range"}
    else:
        assert selection["dev"]["source"] == "dev"
        assert selection["dev"]["full"]["n_docs"] == 32
        assert set(record["evals"][-1]["per_group"]) == {"kv/in_range", "delayed/in_range"}
        assert {group["n_docs"] for group in selection["dev"]["effective"]["groups"].values()} == {
            2
        }


@pytest.mark.usefixtures("restore_determinism")
@pytest.mark.parametrize("legacy", [False, True])
def test_p1_17_capped_divergence_replay_reconstructs_training_subset(
    legacy: bool,
    tiny_manifest: Path,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hsn import runner

    cfg = tiny_gru_config(*disk_overrides(tiny_manifest, steps=2, eval_every=2), "seed=19", steps=2)
    params = protocol_params(
        {"name": "overfit", "n_docs": 8, "diag_every": 0}
        if legacy
        else {"name": "train_eval", "n_train": 8, "n_dev": 4, "diag_every": 0}
    )
    original = runner.make_trainer

    def planted(*args: Any, **kwargs: Any) -> Any:
        trainer = original(*args, **kwargs)

        def fail_second_step(t: Any) -> None:
            if t.step == 1:
                for parameter in t.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.fill_(float("inf"))

        trainer.grad_hook = fail_second_step
        return trainer

    monkeypatch.setattr(runner, "make_trainer", planted)
    record = run_protocol(
        ArmRun(
            cfg=cfg,
            family="gru",
            variant="none",
            variant_params={},
            params=params,
            run_dir=tmp_path / "run",
            rules=parse_rules(None),
            constants={},
        )
    )
    monkeypatch.setattr(runner, "make_trainer", original)
    path = Path(record["divergence"]["path"])
    dump = load_dump(path)
    assert dump["extra"]["divergence"]["notes"]["data_selection"] == {
        "policy": PREFIX if legacy else STRATIFIED,
        "seed": cfg.seed,
    }
    full, _dev = load_datasets(cfg)
    selected = cap_dataset(full, 8, seed=cfg.seed, policy=PREFIX if legacy else STRATIFIED)
    assert dump["loader_state"]["dataset_fingerprint"] == selected.fingerprint
    if legacy:
        # Old dumps had neither cap metadata nor dataset fingerprints.
        del dump["extra"]["divergence"]["notes"]["data_selection"]
        del dump["loader_state"]["dataset_fingerprint"]
        path = tmp_path / "legacy.pt"
        torch.save(dump, path)
    assert replay(path, repeats=1)["windows_match"]
    # Supplying the exact subset must not sample it a second time.
    assert replay(path, train_ds=selected, repeats=1)["windows_match"]


@pytest.mark.usefixtures("isolated", "restore_determinism")
def test_p1_17_probe_restore_uses_recorded_subset_and_preserves_legacy(
    tiny_manifest: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    db = dbm.Database(db_path())
    spec = tiny_reads_state_spec(
        tmp_path,
        tiny_manifest,
        name="caps",
        steps=2,
        eval_every=2,
        protocol={"n_train": 8, "n_dev": 4, "checkpoint": "final"},
    )
    (exp_id,) = enqueue_and_claim(db, spec)
    assert run_child(db, exp_id, tmp_path / "runs") == 0
    row = db.row("SELECT record_json, json_path FROM results WHERE exp_id = ?", (exp_id,))
    assert row is not None
    record = json.loads(row["record_json"])
    run_dir = Path(row["json_path"]).parent
    exp, params, _ctx, restored, _checkpoint, _model = _restored(db, exp_id, run_dir, cpu=True)
    from hsn.config import RunConfig

    cfg = RunConfig.model_validate(
        json.loads(exp["config_json"] or db.arm(exp["arm_id"])["config_json"])
    )
    _train, full_dev = load_datasets(cfg)
    expected = cap_dataset(full_dev, params.n_dev, seed=cfg.seed)
    assert _ids(restored) == _ids(expected)
    assert set(ex.meta["task"] for ex in restored.examples) == {"kv", "delayed"}
    # Split backfill must use exactly that same cap, including the cache's seed and policy.
    from hsn.experiments.split import backfill_split

    record["final"] = {
        key: value for key, value in record["final"].items() if not key.startswith("split/")
    }
    with db.write() as conn:
        conn.execute(
            "UPDATE results SET record_json = ? WHERE exp_id = ?", (json.dumps(record), exp_id)
        )
    backfill = backfill_split(db, [exp_id], recompute=True, cpu=True)
    assert backfill["runs"][0]["recomputed"]["split"] == record["evals"][-1]["split"]
    # A pre-fix record is still evaluated on its original first-N examples, never re-sampled.
    del record["data_selection"]
    with db.write() as conn:
        conn.execute(
            "UPDATE results SET record_json = ? WHERE exp_id = ?", (json.dumps(record), exp_id)
        )
    _exp, _params, _ctx, historical, _checkpoint, _model = _restored(db, exp_id, run_dir, cpu=True)
    assert _ids(historical) == _ids(full_dev)[: params.n_dev]


def _loader(dataset: ByteDataset) -> StreamLoader:
    return StreamLoader(
        dataset, batch=1, unroll=8, seed=0, shuffle=False, align_docs=True, epochs=None
    )


@pytest.mark.parametrize(
    "changed", ["order", "tokens", "targets", "context", "loss_mask", "deps", "doc_id"]
)
def test_p1_17_resume_refuses_same_count_but_changed_data(changed: str) -> None:
    examples = docs("kv", "train", 4)
    dataset = ByteDataset(examples)
    loader = _loader(dataset)
    next(loader)
    saved = loader.state_dict()
    assert saved["dataset_fingerprint"] == dataset.fingerprint
    if changed == "order":
        modified = list(reversed(examples))
    elif changed in ("deps", "doc_id"):
        meta = dict(examples[0].meta)
        if changed == "deps":
            meta["deps"] = tuple(int(dep) + 1 for dep in meta["deps"])
            meta["dep_len"] = max(meta["deps"])
        else:
            meta["doc_id"] += ":different"
        modified = [dataclasses.replace(examples[0], meta=meta), *examples[1:]]
    else:
        array = getattr(examples[0], changed).copy()
        if changed == "loss_mask":
            yes = next(i for i, value in enumerate(array) if value)
            no = next(i for i, value in enumerate(array) if not value)
            array[yes], array[no] = False, True  # preserve the number of dependency entries
        else:
            array[0] = (int(array[0]) + 1) % (16 if changed == "context" else 256)
        modified = [dataclasses.replace(examples[0], **{changed: array}), *examples[1:]]
    replacement = ByteDataset(modified)
    assert len(replacement) == len(dataset)
    with pytest.raises(ValueError, match="fingerprint|dataset|data"):
        _loader(replacement).load_state_dict(saved)
    legacy = {key: value for key, value in saved.items() if key != "dataset_fingerprint"}
    _loader(replacement).load_state_dict(legacy)


def test_p1_17_fingerprint_rebuilds_identically_and_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = docs("delayed", "train", 4)
    dataset = ByteDataset(examples)
    fingerprint = dataset.fingerprint
    rebuilt = ByteDataset(
        [
            dataclasses.replace(
                ex,
                tokens=ex.tokens.copy(),
                targets=ex.targets.copy(),
                loss_mask=ex.loss_mask.copy(),
                context=ex.context.copy(),
                meta=dict(reversed(list(ex.meta.items()))),
            )
            for ex in examples
        ]
    )
    assert rebuilt.fingerprint == fingerprint
    loader = _loader(rebuilt)
    next(loader)
    saved = loader.state_dict()
    _loader(dataset).load_state_dict(saved)

    def no_rehash(*args: object, **kwargs: object) -> None:
        raise AssertionError("checkpoint re-hashed an immutable dataset")

    monkeypatch.setattr("hsn.data.dataset.hashlib.blake2b", no_rehash)
    assert dataset.fingerprint == fingerprint
    assert loader.state_dict()["dataset_fingerprint"] == fingerprint
