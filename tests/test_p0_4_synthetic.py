"""P0.4 tests: synthetic generators, reference solvers, splits, leakage checker, dataset."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from hsn.data.dataset import ByteDataset, collate_windows, iter_windows
from hsn.data.leakage import check_leakage
from hsn.data.splits import (
    SplitManifest,
    build_splits,
    load_shard,
    pack_examples,
    read_shard,
    shard_hash,
    write_shard,
)
from hsn.data.synthetic import (
    BOS,
    EOS,
    N_RESERVED,
    NONE,
    PAD,
    QRY,
    SEP,
    SPLITS,
    DatasetSpec,
    Example,
    TaskSpec,
    check_example,
    example_seed,
    generate,
    load_dataset_spec,
    main,
    reference_targets,
)

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "configs" / "data" / "synthetic_v1.yaml"
GOLDEN_PATH = REPO / "configs" / "data" / "synthetic_v1.manifest.json"
DATASET = load_dataset_spec(SPEC_PATH)
TASK_KEYS = [task.key for task in DATASET.tasks]
ALLOWED_CONTROLS = {BOS, EOS, QRY, SEP, NONE}


def _tiny(task: TaskSpec, n_train: int = 64, n_dev: int = 16, n_test: int = 16) -> TaskSpec:
    """The committed task with small split sizes; re-validated (model_copy would skip that)."""
    return TaskSpec.model_validate(
        {**task.model_dump(), "n_train": n_train, "n_dev": n_dev, "n_test": n_test}
    )


def _tiny_dataset() -> DatasetSpec:
    return DatasetSpec(
        name="tiny", unroll=DATASET.unroll, tasks=[_tiny(task) for task in DATASET.tasks]
    )


def _task(key: str) -> TaskSpec:
    name, variant = key.split("/")
    return DATASET.task(name, variant)  # type: ignore[arg-type]


def _all(task: TaskSpec, split: str) -> list[Example]:
    return [generate(task, split, i) for i in range(task.n_examples(split))]  # type: ignore[arg-type]


def _same(a: Example, b: Example) -> bool:
    return (
        np.array_equal(a.tokens, b.tokens)
        and np.array_equal(a.targets, b.targets)
        and np.array_equal(a.loss_mask, b.loss_mask)
        and np.array_equal(a.context, b.context)
        and a.meta == b.meta
    )


# --------------------------------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------------------------------


def test_p0_4_deterministic() -> None:
    for task in DATASET.tasks:
        tiny = _tiny(task)
        first = generate(tiny, "train", 3)
        assert _same(first, generate(tiny, "train", 3))  # same inputs -> identical bytes
        # a different index or split gives a different document
        assert not np.array_equal(first.tokens, generate(tiny, "train", 4).tokens)
        assert not np.array_equal(first.tokens, generate(tiny, "dev", 3).tokens)
        # nothing is shared between calls: interleaving other splits changes nothing
        generate(tiny, "dev", 0)
        generate(tiny, "test", 0)
        assert _same(first, generate(tiny, "train", 3))
        # the same (seed, split, index) seeds the same stream whatever the task object
        assert _same(first, generate(_tiny(task), "train", 3))
    assert example_seed(7, "train", 0) != example_seed(7, "dev", 0)
    assert example_seed(7, "train", 0) != example_seed(7, "train", 1)
    assert example_seed(7, "train", 0) != example_seed(8, "train", 0)
    with pytest.raises(IndexError):
        generate(_tiny(DATASET.tasks[0]), "dev", 16)
    with pytest.raises(ValueError, match="split"):
        generate(_tiny(DATASET.tasks[0]), "validation", 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("key", TASK_KEYS)
def test_p0_4_targets_match_reference_solver(key: str) -> None:
    task = _tiny(_task(key), n_train=96, n_dev=32, n_test=32)
    length_lo, length_hi = task.length_range
    dep_lo, dep_hi = task.dep_len_range
    seen_answers = 0
    for split in SPLITS:
        for example in _all(task, split):
            check_example(task, example)  # positions, targets and deps equal the solver's
            tokens = example.tokens
            positions, targets, deps = reference_targets(task, tokens)
            assert positions.size > 0  # every document has at least one task target
            assert np.array_equal(np.flatnonzero(example.loss_mask), positions)
            assert np.array_equal(example.targets[example.loss_mask], targets)
            assert np.array_equal(np.asarray(example.meta["deps"]), deps)
            seen_answers += int(positions.size)
            # framing, reserved bytes, next-symbol targets, constant context
            assert tokens[0] == BOS and tokens[-1] == EOS
            assert not example.loss_mask[-1]
            low = set(np.unique(tokens[tokens < N_RESERVED]).tolist())
            assert low <= ALLOWED_CONTROLS, low
            assert PAD not in low
            assert np.array_equal(example.targets[:-1], tokens[1:])
            assert example.targets[-1] == PAD
            assert np.all(example.context == task.context_id)
            # lengths and dependencies
            assert example.length <= length_hi
            if task.family in ("kv", "delayed"):
                assert example.length >= length_lo
            assert dep_lo <= example.meta["dep_len"] <= dep_hi
            assert example.meta["dep_len"] == max(example.meta["deps"])
            assert example.meta["in_range"] == (example.meta["dep_len"] <= task.unroll)
            assert (
                example.meta["doc_id"]
                == f"{task.name}:{task.variant}:{split}:{example.meta['index']}"
            )
            assert example.meta["split"] == split
            assert not tokens.flags.writeable
    assert seen_answers > 0
    # the solver rejects a malformed document
    bad = np.array([BOS, EOS, EOS], dtype=np.uint8)
    with pytest.raises(ValueError):
        reference_targets(task, bad)


def test_p0_4_out_of_range_lengths_exceed_unroll() -> None:
    for task in DATASET.tasks:
        tiny = _tiny(task)
        for split in SPLITS:
            for example in _all(tiny, split):
                if task.variant == "out_of_range":
                    assert example.meta["dep_len"] > tiny.unroll
                    assert example.meta["in_range"] is False
                else:
                    assert example.meta["dep_len"] <= tiny.unroll
                    assert example.meta["in_range"] is True
                    assert example.length <= tiny.unroll  # synthetic_v1: fits one window
    # a range that straddles the unroll window is rejected for either variant
    base = DATASET.tasks[0].model_dump()
    with pytest.raises(ValueError, match="in_range"):
        TaskSpec.model_validate({**base, "dep_len_range": [16, 257], "length_range": [32, 300]})
    with pytest.raises(ValueError, match="out_of_range"):
        TaskSpec.model_validate(
            {
                **base,
                "variant": "out_of_range",
                "dep_len_range": [200, 300],
                "length_range": [32, 400],
            }
        )


def test_p0_4_spec_validation_rejects() -> None:
    base = DATASET.tasks[0].model_dump()  # kv / in_range
    with pytest.raises(ValueError, match="too short"):
        TaskSpec.model_validate({**base, "length_range": [32, 200]})
    with pytest.raises(ValueError, match="both use byte"):
        TaskSpec.model_validate({**base, "params": {**base["params"], "values": [100, 160]}})
    with pytest.raises(ValueError, match="marker byte"):
        TaskSpec.model_validate(
            {**base, "params": {**base["params"], "markers": {"ins": 40, "ovr": 17, "del": 18}}}
        )
    with pytest.raises(ValueError, match="n_keys"):
        TaskSpec.model_validate({**base, "params": {**base["params"], "n_keys": 32}})
    with pytest.raises(ValueError, match="params"):
        TaskSpec.model_validate({**base, "params": {**base["params"], "unknown": 1}})
    with pytest.raises(ValueError, match="dep_len_range"):
        TaskSpec.model_validate({**base, "dep_len_range": [2, 240]})
    with pytest.raises(ValueError):
        TaskSpec.model_validate({**base, "context_id": 16})
    dataset = DATASET.model_dump()
    tasks = dataset["tasks"]
    with pytest.raises(ValueError, match="distinct seed"):
        DatasetSpec.model_validate(
            {**dataset, "tasks": [tasks[0], {**tasks[1], "seed": tasks[0]["seed"]}]}
        )
    with pytest.raises(ValueError, match="duplicate task"):
        DatasetSpec.model_validate({**dataset, "tasks": [tasks[0], {**tasks[0], "seed": 99}]})
    with pytest.raises(ValueError, match="unroll"):
        DatasetSpec.model_validate({**dataset, "unroll": 128})
    # the committed spec propagates its unroll and has one context id per family
    assert all(task.unroll == DATASET.unroll for task in DATASET.tasks)
    by_family = {task.family: task.context_id for task in DATASET.tasks}
    assert all(task.context_id == by_family[task.family] for task in DATASET.tasks)
    assert len(set(by_family.values())) == len(by_family)
    in_range = [task for task in DATASET.tasks if task.variant == "in_range"]
    for task in in_range:
        other = DATASET.task(task.name, "out_of_range")
        assert task.params == other.params  # the variants share every generator parameter


# --------------------------------------------------------------------------------------------------
# splits and leakage
# --------------------------------------------------------------------------------------------------


def _key_bytes(task: TaskSpec, split: str) -> np.ndarray:
    lo, hi = task.params["keys"][split]
    return np.arange(lo, hi, dtype=np.uint8)


def test_p0_4_splits_disjoint_and_leakage_zero(tmp_path: Path) -> None:
    dataset = _tiny_dataset()
    manifest = build_splits(dataset, tmp_path / "tiny")
    assert (tmp_path / "tiny" / "manifest.json").is_file()
    assert manifest.is_clean
    n_tasks = len(dataset.tasks)
    assert len(manifest.leakage) == n_tasks * (2 * n_tasks + 1)
    for pair in manifest.leakage:
        assert pair.exact_duplicates == 0, pair
        assert pair.overlapping_ngrams == 0, pair
        assert pair.test_docs_with_overlap == 0, pair
        assert pair.n_gram == 32
    checked = {(pair.train, pair.test) for pair in manifest.leakage}
    for task in dataset.tasks:
        for other in dataset.tasks:
            for split in ("dev", "test"):
                assert (f"{task.key}/train", f"{other.key}/{split}") in checked
        assert (f"{task.key}/dev", f"{task.key}/test") in checked
    # documents never repeat across splits of a task, and the key alphabets are disjoint:
    # a split's key bytes occur in no other split's documents
    for task in dataset.tasks:
        docs = {
            split: load_shard(tmp_path / "tiny" / f"{task.name}.{task.variant}.{split}.npz")
            for split in SPLITS
        }
        digests = {split: {ex.tokens.tobytes() for ex in docs[split]} for split in SPLITS}
        assert not (digests["train"] & digests["dev"])
        assert not (digests["train"] & digests["test"])
        assert not (digests["dev"] & digests["test"])
        if task.family in ("kv", "delayed"):
            for split in SPLITS:
                others = [s for s in SPLITS if s != split]
                foreign = np.concatenate([_key_bytes(task, s) for s in others])
                for example in docs[split]:
                    assert not np.isin(example.tokens, foreign).any()
                own = _key_bytes(task, split)
                assert any(np.isin(ex.tokens, own).any() for ex in docs[split])
    # positive controls: the checker does see leakage when it exists
    kv = dataset.tasks[0]
    train = _all(kv, "train")
    test = _all(kv, "test")
    self_report = check_leakage(train, train)
    assert self_report.exact_duplicates == len(train)
    assert self_report.overlapping_ngrams > 0
    assert self_report.test_docs_with_overlap == len(train)
    assert not self_report.is_clean
    assert self_report.examples and self_report.examples[0].test_doc == train[0].meta["doc_id"]
    clean = check_leakage(train, test)
    assert clean.is_clean and clean.n_train == len(train) and clean.n_test == len(test)
    # a planted 32-gram copied from a test document into a train document is found
    victim = test[0]
    gram = victim.tokens[5:37]
    host = train[0]
    # neighbours of the planted gram differ from the victim's, so exactly one 32-gram is shared
    before = np.array([255 if victim.tokens[4] != 255 else 254], dtype=np.uint8)
    after = np.array([255 if victim.tokens[37] != 255 else 254], dtype=np.uint8)
    planted_tokens = np.concatenate([host.tokens[:9], before, gram, after, host.tokens[11:]])
    targets = np.concatenate([planted_tokens[1:], np.array([PAD], dtype=np.uint8)])
    planted = Example(
        planted_tokens,
        targets,
        np.zeros(planted_tokens.shape[0], dtype=np.bool_),
        np.zeros(planted_tokens.shape[0], dtype=np.uint8),
        {**host.meta, "doc_id": "planted", "deps": (), "dep_len": 0},
    )
    report = check_leakage([planted, *train[1:]], test)
    assert report.exact_duplicates == 0
    assert report.overlapping_ngrams == 1
    assert report.test_docs_with_overlap == 1
    hit = report.examples[0]
    assert (hit.test_doc, hit.test_pos, hit.train_doc, hit.train_pos) == (
        victim.meta["doc_id"],
        5,
        "planted",
        10,
    )
    assert check_leakage([planted], test, n_gram=33).overlapping_ngrams == 0
    with pytest.raises(ValueError, match="n_gram"):
        check_leakage(train, test, n_gram=0)
    # the manifest round-trips through JSON and records the shards in task order
    reloaded = SplitManifest.load(tmp_path / "tiny" / "manifest.json")
    assert reloaded == manifest
    assert [s.key for s in reloaded.shards] == [
        f"{task.key}/{split}" for task in dataset.tasks for split in SPLITS
    ]
    assert manifest.spec_hash == dataset.spec_hash()


def test_p0_4_shard_hash_is_data_only(tmp_path: Path) -> None:
    task = _tiny(DATASET.tasks[2], n_train=8, n_dev=4, n_test=4)
    examples = _all(task, "dev")
    record = write_shard(tmp_path, task, "dev", examples)
    arrays, scalars = read_shard(tmp_path / record.file)
    assert shard_hash(arrays, scalars) == record.hash
    assert shard_hash(pack_examples(examples), scalars) == record.hash
    loaded = load_shard(tmp_path / record.file, record.hash)
    assert len(loaded) == len(examples)
    for a, b in zip(loaded, examples, strict=True):
        assert _same(a, b)
    with pytest.raises(ValueError, match="hash"):
        load_shard(tmp_path / record.file, "0" * 64)


# --------------------------------------------------------------------------------------------------
# dataset and collation
# --------------------------------------------------------------------------------------------------


def _expected_rows(lengths: list[int], batch: int, unroll: int, align: bool) -> list[list[int]]:
    rows: list[list[int]] = [[] for _ in range(batch)]
    used = [0] * batch
    for i, length in enumerate(lengths):
        r = min(range(batch), key=lambda r: (used[r], r))
        rows[r].append(i)
        used[r] += -(-length // unroll) * unroll if align else length
    return rows


@pytest.mark.parametrize("align_docs", [False, True])
def test_p0_4_collate_reset_flags_at_doc_starts(align_docs: bool) -> None:
    task = _tiny(DATASET.tasks[0], n_train=7)
    examples = _all(task, "train")
    dataset = ByteDataset(examples)
    assert len(dataset) == 7
    item = dataset[2]
    assert set(item) == {"tokens", "targets", "loss_mask", "context", "reset", "dep"}
    assert item["tokens"].dtype == torch.uint8 and item["reset"].dtype == torch.bool
    assert item["dep"].dtype == torch.int32
    assert bool(item["reset"][0]) and not bool(item["reset"][1:].any())
    assert torch.equal(
        item["dep"][item["loss_mask"]], torch.tensor(examples[2].meta["deps"], dtype=torch.int32)
    )
    assert dataset.meta(2)["doc_id"] == examples[2].meta["doc_id"]

    unroll, batch = 16, 2
    out = collate_windows(examples, unroll, batch=batch, align_docs=align_docs)
    assert set(out) == {"tokens", "targets", "context", "loss_mask", "reset", "dep", "doc"}
    width = out["tokens"].shape[1]
    assert out["tokens"].shape == (batch, width) and width % unroll == 0
    assert out["tokens"].dtype == torch.int64 and out["doc"].dtype == torch.int64
    assert out["loss_mask"].dtype == torch.bool and out["reset"].dtype == torch.bool
    lengths = [e.length for e in examples]
    rows = _expected_rows(lengths, batch, unroll, align_docs)
    expected_reset = torch.zeros(batch, width, dtype=torch.bool)
    for r, row in enumerate(rows):
        cursor = 0
        for i in row:
            end = cursor + lengths[i]
            expected_reset[r, cursor] = True
            if align_docs:
                assert cursor % unroll == 0
            assert torch.equal(
                out["tokens"][r, cursor:end], torch.from_numpy(examples[i].tokens.copy()).long()
            )
            assert torch.equal(
                out["targets"][r, cursor:end], torch.from_numpy(examples[i].targets.copy()).long()
            )
            assert torch.equal(
                out["loss_mask"][r, cursor:end], torch.from_numpy(examples[i].loss_mask.copy())
            )
            assert torch.all(out["doc"][r, cursor:end] == i)
            assert torch.all(out["context"][r, cursor:end] == task.context_id)
            deps = out["dep"][r, cursor:end][out["loss_mask"][r, cursor:end]]
            assert deps.tolist() == list(examples[i].meta["deps"])
            cursor = -(-end // unroll) * unroll if align_docs else end
    assert torch.equal(out["reset"], expected_reset)  # exactly at document starts
    assert int(out["reset"].sum()) == len(examples)
    padding = out["doc"] == -1
    assert torch.all(out["tokens"][padding] == PAD)
    assert torch.all(out["targets"][padding] == PAD)
    assert not out["loss_mask"][padding].any()
    assert not out["reset"][padding].any()
    assert torch.all(out["dep"][padding] == 0)
    assert int((~padding).sum()) == sum(lengths)
    windows = list(iter_windows(out, unroll))
    assert len(windows) == width // unroll
    assert all(w["tokens"].shape == (batch, unroll) for w in windows)
    assert torch.equal(torch.cat([w["tokens"] for w in windows], dim=1), out["tokens"])
    # dict items collate the same way as Example objects
    again = collate_windows(
        [dataset[i] for i in range(len(dataset))], unroll, batch=batch, align_docs=align_docs
    )
    for key in out:
        assert torch.equal(again[key], out[key]), key
    with pytest.raises(ValueError):
        collate_windows([], unroll)
    with pytest.raises(ValueError):
        collate_windows(examples, 0)


# --------------------------------------------------------------------------------------------------
# golden manifest and CLI
# --------------------------------------------------------------------------------------------------


def test_p0_4_golden_manifest_matches(tmp_path: Path) -> None:
    """The committed manifest hashes are reproduced here (CI: on both runners)."""
    golden = SplitManifest.load(GOLDEN_PATH)
    assert golden.name == DATASET.name
    assert golden.spec_hash == DATASET.spec_hash()
    assert golden.unroll == DATASET.unroll
    assert golden.is_clean
    assert len(golden.leakage) == len(DATASET.tasks) * (2 * len(DATASET.tasks) + 1)
    assert [s.key for s in golden.shards] == [
        f"{task.key}/{split}" for task in DATASET.tasks for split in SPLITS
    ]
    for task in DATASET.tasks:
        record = golden.shard(task.name, task.variant, "test")
        assert record.n_docs == task.n_test
        examples = _all(task, "test")
        assert write_shard(tmp_path, task, "test", examples).hash == record.hash
        assert sum(e.length for e in examples) == record.n_tokens
        for split in SPLITS:
            assert golden.shard(task.name, task.variant, split).n_docs == task.n_examples(split)


def _write_yaml(path: Path, dataset: DatasetSpec) -> Path:
    path.write_text(yaml.safe_dump(dataset.model_dump(mode="json"), sort_keys=False), "utf-8")
    return path


def test_p0_4_cli_build_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = DatasetSpec(
        name="tinycli",
        unroll=DATASET.unroll,
        tasks=[_tiny(task, 24, 8, 8) for task in DATASET.tasks[:4]],
    )
    spec = _write_yaml(tmp_path / "tiny.yaml", dataset)
    out = tmp_path / "out"
    golden = tmp_path / "golden.json"
    assert (
        main(["build", "--spec", str(spec), "--out", str(out), "--write-golden", str(golden)]) == 0
    )
    assert golden.is_file() and (out / "manifest.json").read_text() == golden.read_text()
    assert main(["build", "--spec", str(spec), "--out", str(out), "--golden", str(golden)]) == 0
    assert main(["verify", "--spec", str(spec), "--out", str(out)]) == 0
    assert main(["bench", "--spec", str(spec), "--n", "4"]) == 0
    assert "examples/s" in capsys.readouterr().out
    # loading through the manifest (hashes verified) and filtering by task and variant
    train = ByteDataset.load(out / "manifest.json", "train", tasks=["kv"])
    assert len(train) == 48 and {e.meta["task"] for e in train.examples} == {"kv"}
    dev = ByteDataset.load(out / "manifest.json", "dev", variants=["in_range"], expected_unroll=256)
    assert len(dev) == 16 and all(e.meta["in_range"] for e in dev.examples)
    with pytest.raises(ValueError, match="unroll"):
        ByteDataset.load(out / "manifest.json", "dev", expected_unroll=128)
    with pytest.raises(KeyError):
        ByteDataset.load(out / "manifest.json", "dev", tasks=["nope"])
    # a tampered shard fails verification and loading, and a stale golden fails the build
    shard = out / "kv.in_range.train.npz"
    with np.load(shard, allow_pickle=False) as loaded:
        arrays: dict[str, Any] = {key: loaded[key] for key in loaded.files}
    arrays["tokens"] = arrays["tokens"].copy()
    arrays["tokens"][5] = 200
    np.savez(shard, **arrays)
    assert main(["verify", "--spec", str(spec), "--out", str(out)]) == 1
    assert "file hash" in capsys.readouterr().err
    with pytest.raises(ValueError, match="hash"):
        ByteDataset.load(out / "manifest.json", "train", tasks=["kv"])
    assert len(ByteDataset.load(out / "manifest.json", "train", tasks=["kv"], verify=False)) == 48
    golden.write_text(golden.read_text().replace('"unroll": 256', '"unroll": 255'), "utf-8")
    assert main(["build", "--spec", str(spec), "--out", str(out), "--golden", str(golden)]) == 1
    assert "golden" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["build"])
