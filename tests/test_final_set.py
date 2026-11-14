"""``synthetic_final`` (14 Nov 2026): the untouched final test set of the hold-out data. A key
alphabet that no task of the hold-out data uses and fresh documents, to be scored once per gate
claim (``python -m hsn.eval.final_test``). These tests check that the set is the hold-out tasks
apart from the test keys and the seeds, that its alphabet and documents are new, the manifest,
and that a scoring goes on the ledger and is refused a second time."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hsn.config import config_hash, dump_resolved, load_config
from hsn.data.splits import SplitManifest, write_shard
from hsn.data.synthetic import SPLITS, TaskSpec, check_example, generate, load_dataset_spec
from hsn.device import seed_everything
from hsn.eval import final_test
from hsn.model.hsn import HSN
from hsn.train.checkpoint import capture_rng_state, save_checkpoint

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "configs" / "data"
FINAL = load_dataset_spec(DATA / "synthetic_final.yaml")
HOLD = load_dataset_spec(DATA / "synthetic_hold.yaml")
FINAL_KEYS = (32, 48)
TRAIN_KEYS = (64, 112)
#: the seed offset of the development scoring scripts' fresh documents (hold_eval.py)
FRESH = 104_729


def _task(dataset, name: str) -> TaskSpec:
    return next(t for t in dataset.tasks if t.name == name)


def _used_bytes(spec: TaskSpec) -> set[int]:
    """Every data and marker byte a task's parameters name."""
    used: set[int] = set()
    for value in spec.params.values():
        ranges = list(value.values()) if isinstance(value, dict) else [value]
        for item in ranges:
            if (
                isinstance(item, list | tuple)
                and len(item) == 2
                and all(isinstance(x, int) for x in item)
            ):
                used.update(range(item[0], item[1]))
            elif isinstance(item, int) and 16 <= item < 32:
                used.add(item)
    return used


def test_the_tasks_are_the_hold_out_tasks_apart_from_test_keys_and_seeds() -> None:
    assert [t.name for t in FINAL.tasks] == ["kv", "delayed2h", "delayed2h_new"]
    for name, source in (("kv", "kv"), ("delayed2h", "delayed2h"), ("delayed2h_new", "delayed2h")):
        final, hold = _task(FINAL, name), _task(HOLD, source)
        assert final.family == hold.family and final.context_id == hold.context_id
        assert final.length_range == hold.length_range
        assert final.dep_len_range == hold.dep_len_range and final.unroll == hold.unroll
        ours, theirs = dict(final.params), dict(hold.params)
        assert ours.pop("keys") == {"train": [64, 112], "dev": [112, 128], "test": [32, 48]}
        theirs.pop("keys")
        if name == "delayed2h_new":
            assert ours.pop("hold_role") == "only" and theirs.pop("hold_role") == "exclude"
        assert ours == theirs
        assert final.n_dev == 0  # nothing here is for selection
    assert _task(FINAL, "delayed2h_new").n_test == 0


def test_the_final_alphabet_is_used_by_no_hold_out_task() -> None:
    final = set(range(*FINAL_KEYS))
    for dataset in (HOLD, load_dataset_spec(DATA / "synthetic_keys.yaml")):
        for task in dataset.tasks:
            assert not final & _used_bytes(task), task.key
    v1 = load_dataset_spec(DATA / "synthetic_v1.yaml")
    for task in v1.tasks:
        if task.family in ("kv", "delayed"):
            assert not final & _used_bytes(task), task.key


def test_seeds_are_new() -> None:
    taken: set[int] = set()
    for name in ("synthetic_hold", "synthetic_keys", "synthetic_v1", "synthetic_v2"):
        for task in load_dataset_spec(DATA / f"{name}.yaml").tasks:
            taken.update({task.seed, task.seed + FRESH})
    seeds = [task.seed for task in FINAL.tasks]
    assert len(set(seeds)) == len(seeds) and not taken & set(seeds)


def test_documents_use_the_keys_of_their_split() -> None:
    for name in ("kv", "delayed2h"):
        task = _task(FINAL, name)
        for split, (lo, hi), other in (
            ("test", FINAL_KEYS, TRAIN_KEYS),
            ("train", TRAIN_KEYS, FINAL_KEYS),
        ):
            seen: set[int] = set()
            for index in range(120):
                example = generate(task, split, index)
                assert not check_example(task, example)
                tokens = example.tokens
                seen.update(int(b) for b in tokens[(tokens >= lo) & (tokens < hi)])
                assert not bool(((tokens >= other[0]) & (tokens < other[1])).any())
            assert len(seen) == hi - lo  # the whole alphabet occurs


def test_new_pairs_are_held_pairs_of_training_bytes() -> None:
    task = _task(FINAL, "delayed2h_new")
    mod = task.params["hold_mod"]
    lo, hi = TRAIN_KEYS
    for index in range(120):
        t = generate(task, "train", index).tokens.tolist()
        i = 1
        while not lo <= t[i] < hi:
            i += 1
        while lo <= t[i] < hi:
            assert (t[i] + t[i + 1]) % mod == 0
            i += 3


def test_golden_manifest_matches(tmp_path: Path) -> None:
    golden = SplitManifest.load(DATA / "synthetic_final.manifest.json")
    assert golden.name == FINAL.name == "synthetic_final"
    assert golden.spec_hash == FINAL.spec_hash() and golden.is_clean
    for task in FINAL.tasks:
        for split in SPLITS:
            record = golden.shard(task.name, task.variant, split)
            assert record.n_docs == task.n_examples(split)
        split = "test" if task.n_test else "train"
        examples = [generate(task, split, i) for i in range(task.n_examples(split))]
        record = golden.shard(task.name, task.variant, split)
        assert write_shard(tmp_path, task, split, examples).hash == record.hash
        assert sum(e.length for e in examples) == record.n_tokens
        assert np.all([e.meta["in_range"] for e in examples])


# --------------------------------------------------------------------------------------------------
# Scoring: once per claim, on the record
# --------------------------------------------------------------------------------------------------


def _run_folder(tmp_path: Path) -> Path:
    """A run folder as the runner leaves it: the resolved configuration and one checkpoint of a
    tiny untrained HSN whose configuration names the hold-out tasks."""
    cfg = load_config(
        REPO / "configs" / "tiny_gradcheck.yaml",
        overrides=["model.vocab_size=256", "data.tasks=[kv, delayed2h]"],
    )
    seed_everything(0)
    model = HSN.from_config(cfg)
    run_dir = tmp_path / "run"
    dump_resolved(cfg, run_dir / "config.resolved.yaml")
    save_checkpoint(
        run_dir / "step_00000003.pt",
        model,
        None,
        None,
        3,
        capture_rng_state(),
        None,
        extra={"model_name": "hsn", "config_hash": config_hash(cfg)},
    )
    return run_dir


def test_scoring_goes_on_the_ledger_and_is_refused_a_second_time(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, ledger = _run_folder(tmp_path), tmp_path / "ledger.jsonl"
    args = [str(run_dir), "--reach", "63", "--docs", "6", "--ledger", str(ledger)]
    assert final_test.main([*args, "--claim", "a test claim"]) == 0
    out = capsys.readouterr().out
    assert "final keys" in out and "new pairs" in out and "beyond the reach" in out
    (record,) = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert record["claim"] == "a test claim" and record["again"] is None
    assert record["reach"] == 63 and record["checkpoint"] == "step_00000003.pt"
    assert set(record["results"]) == {"kv", "delayed2h"}
    assert set(record["results"]["delayed2h"]) == {"seen pairs", "new pairs", "final bytes"}
    for conditions in record["results"].values():
        for entry in conditions.values():
            total, beyond = entry["total"], entry["beyond_reach"]
            assert entry["docs"] == 6 and 0 < beyond["n"] <= total["n"]
            low, high = total["ci95"]
            assert 0.0 <= low <= total["accuracy"] <= high <= 1.0
    # the same checkpoint again: refused, and nothing is added
    assert final_test.main([*args, "--claim", "another claim"]) == 2
    assert "once per gate claim" in capsys.readouterr().err
    assert len(ledger.read_text().splitlines()) == 1
    # with a reason it is scored again, and the reason is on the record
    assert final_test.main([*args, "--claim", "another claim", "--again", "a reason"]) == 0
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(records) == 2 and records[1]["again"] == "a reason"
    assert records[1]["results"] == records[0]["results"]  # the same documents, the same score


def test_a_model_without_a_reader_has_every_answer_beyond_reach(tmp_path: Path) -> None:
    run_dir, ledger = _run_folder(tmp_path), tmp_path / "ledger.jsonl"
    code = final_test.main(
        [str(run_dir), "--reach", "0", "--docs", "4", "--ledger", str(ledger), "--claim", "c"]
    )
    assert code == 0
    record = json.loads(ledger.read_text().splitlines()[0])
    for conditions in record["results"].values():
        for entry in conditions.values():
            assert entry["total"] == entry["beyond_reach"]
