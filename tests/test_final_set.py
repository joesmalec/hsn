"""``synthetic_final`` (14 Nov 2026): the untouched final test set of the hold-out data. A key
alphabet that no task of the hold-out data uses and fresh documents, scored in one preregistered
campaign (``python -m hsn.eval.final_test``: freeze, then run). These tests check that the set
is the hold-out tasks apart from the test keys and the seeds, that its alphabet and documents
are new, the manifest, and the lock: a frozen plan, one campaign, the set spent afterwards, and
no candidate whose training stream held a final byte."""

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
# Scoring: one frozen plan, one campaign, on the record
# --------------------------------------------------------------------------------------------------

HOLD_OUT = ("data.dataset=synthetic_hold", "data.tasks=[kv, delayed2h]", "data.variants=[in_range]")


def _run_folder(tmp_path: Path, name: str = "run", data: tuple[str, ...] = HOLD_OUT) -> Path:
    """A run folder as the runner leaves it: the resolved configuration and one checkpoint of a
    tiny untrained HSN whose configuration names its training data."""
    cfg = load_config(
        REPO / "configs" / "tiny_gradcheck.yaml", overrides=["model.vocab_size=256", *data]
    )
    seed_everything(0)
    model = HSN.from_config(cfg)
    run_dir = tmp_path / name
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


def _draft(tmp_path: Path, *run_dirs: Path, reach: int = 63, **changes: object) -> Path:
    draft = {
        "name": "a test campaign",
        "claims": [{"id": "K", "text": "recall beyond the reach", "pass": "97 % or more"}],
        "candidates": [
            {"run_dir": str(run_dir), "checkpoint": "step_00000003.pt", "reach": reach}
            for run_dir in run_dirs
        ],
        "analysis": "accuracy per condition, total and beyond the reach",
        **changes,
    }
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(draft))
    return path


def _frozen(tmp_path: Path, *run_dirs: Path, reach: int = 63) -> tuple[Path, Path]:
    plan, ledger = tmp_path / "plan.json", tmp_path / "ledger.jsonl"
    draft = _draft(tmp_path, *run_dirs, reach=reach)
    assert final_test.main(["freeze", str(draft), "--out", str(plan), "--ledger", str(ledger)]) == 0
    return plan, ledger


def test_one_frozen_plan_one_campaign_then_the_set_is_spent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, second = _run_folder(tmp_path, "a"), _run_folder(tmp_path, "b")
    plan, ledger = _frozen(tmp_path, first, second)
    frozen = json.loads(plan.read_text())
    assert [c["run"] for c in frozen["candidates"]] == ["a", "b"]
    assert all(len(c["checkpoint_sha256"]) == 64 for c in frozen["candidates"])
    assert frozen["final_alphabet"] == [32, 48] and len(frozen["plan_sha256"]) == 64
    run = ["run", str(plan), "--ledger", str(ledger), "--docs", "5"]
    # a plan that is not committed in the repository is refused: it must be on record first
    assert final_test.main(run) == 2
    assert "not committed" in capsys.readouterr().err and not ledger.exists()
    assert final_test.main([*run, "--allow-uncommitted"]) == 0
    out = capsys.readouterr().out
    assert "final keys" in out and "new pairs" in out and "is spent" in out
    (record,) = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert record["plan"] == "a test campaign" and record["plan_committed"] is False
    assert record["plan_sha256"] == frozen["plan_sha256"]
    assert set(record["results"]) == {"a/step_00000003.pt", "b/step_00000003.pt"}
    for results in record["results"].values():
        assert set(results) == {"kv", "delayed2h"}
        assert set(results["delayed2h"]) == {"seen pairs", "new pairs", "final bytes"}
        assert results["kv"]["final keys"]["population"].startswith("key bytes absent")
        for conditions in results.values():
            for entry in conditions.values():
                total, beyond = entry["total"], entry["beyond_reach"]
                assert entry["docs"] == 5 and 0 < beyond["n"] <= total["n"]
                low, high = total["ci95"]
                assert 0.0 <= low <= total["accuracy"] <= high <= 1.0
    # the set is spent: neither the same plan nor a new one is scored or frozen again
    assert final_test.main([*run, "--allow-uncommitted"]) == 2
    assert "was spent" in capsys.readouterr().err
    other = _draft(tmp_path, first)
    again = ["freeze", str(other), "--out", str(tmp_path / "p2.json"), "--ledger", str(ledger)]
    assert final_test.main(again) == 2
    assert "needs a new final set" in capsys.readouterr().err
    assert len(ledger.read_text().splitlines()) == 1


def test_a_plan_or_checkpoint_changed_after_freezing_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _run_folder(tmp_path)
    plan, ledger = _frozen(tmp_path, run_dir)
    run = ["run", str(plan), "--ledger", str(ledger), "--docs", "4", "--allow-uncommitted"]
    edited = json.loads(plan.read_text())
    edited["candidates"][0]["reach"] = 0  # a look at the numbers must not move the mask
    plan.write_text(json.dumps(edited))
    assert final_test.main(run) == 2
    assert "changed after it was frozen" in capsys.readouterr().err
    plan, ledger = _frozen(tmp_path, run_dir)
    checkpoint = run_dir / "step_00000003.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
    assert final_test.main(run) == 2
    assert "is not the frozen file" in capsys.readouterr().err
    assert not ledger.exists()


def test_freezing_checks_the_plan_and_the_training_stream(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = _run_folder(tmp_path)
    ledger = ["--out", str(tmp_path / "plan.json"), "--ledger", str(tmp_path / "ledger.jsonl")]
    empty = _draft(tmp_path, good, claims=[], analysis="")
    assert final_test.main(["freeze", str(empty), *ledger]) == 2
    err = capsys.readouterr().err
    assert "at least one claim" in err and "'analysis'" in err
    # the five-task mix holds bytes 32 to 39 as bracket symbols
    mixed = _run_folder(
        tmp_path, "mixed", ("data.dataset=synthetic_v1", "data.tasks=[kv, brackets]")
    )
    assert final_test.main(["freeze", str(_draft(tmp_path, mixed)), *ledger]) == 2
    assert "training documents hold final bytes" in capsys.readouterr().err
    # text holds the final alphabet everywhere
    text = _run_folder(
        tmp_path, "text", ("data.dataset=tinystories_v1", "data.tasks=[tinystories]")
    )
    assert final_test.main(["freeze", str(_draft(tmp_path, text)), *ledger]) == 2
    assert "not a synthetic task set" in capsys.readouterr().err
    missing = _draft(tmp_path, tmp_path / "nowhere")
    assert final_test.main(["freeze", str(missing), *ledger]) == 2
    assert "does not exist" in capsys.readouterr().err
    assert not (tmp_path / "plan.json").exists()


def test_exposure_is_read_by_role() -> None:
    final = set(range(*FINAL_KEYS))
    tiny = REPO / "configs" / "tiny_gradcheck.yaml"
    hold_out = load_config(tiny, ["model.vocab_size=256", *HOLD_OUT])
    assert final_test.exposure(hold_out, final) == []
    five = load_config(
        tiny,
        [
            "model.vocab_size=256",
            "data.dataset=synthetic_v1",
            "data.tasks=[kv, delayed, brackets, arithmetic, algorithmic]",
        ],
    )
    problems = final_test.exposure(five, final)
    assert problems and all("brackets" in problem for problem in problems)
    # the dev and test keys of a task are not training inputs
    dev_as_final = set(range(112, 128))
    assert final_test.exposure(hold_out, dev_as_final) == []
    assert final_test.exposure(hold_out, set(range(64, 70)))  # training keys are


def test_a_model_without_a_reader_has_every_answer_beyond_reach(tmp_path: Path) -> None:
    plan, ledger = _frozen(tmp_path, _run_folder(tmp_path), reach=0)
    code = final_test.main(
        ["run", str(plan), "--ledger", str(ledger), "--docs", "4", "--allow-uncommitted"]
    )
    assert code == 0
    record = json.loads(ledger.read_text().splitlines()[0])
    for results in record["results"].values():
        for conditions in results.values():
            for entry in conditions.values():
                assert entry["total"] == entry["beyond_reach"]
