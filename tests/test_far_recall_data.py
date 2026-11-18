"""The far-recall test sets (18 Nov 2026): ``synthetic_far`` (``configs/data/synthetic_far.yaml``,
``synthetic_v1``'s kv and delayed tasks asked 272 to 3,072 bytes back) and
``synthetic_far_early`` (the same with every fact in the first 256 bytes), and the fact position
of an answer (``hsn.data.synthetic.fact_positions``, ``hsn.data.splits.shard_fact_positions``).

These tests check that the far set is ``synthetic_v1``'s tasks apart from the seeds, the sizes
and the ranges; that the early set's tasks are ``synthetic_v1``'s in-range tasks moved whole
windows further; that the seeds are new; that every document agrees with the reference solver,
uses the dev keys only and keeps its ranges; that the build is deterministic and reproduces
the committed manifests, which for the far set is the manifest of the set the analysis of
18 Nov 2026 scored; that the fact position points at the fact, read from the token stream
alone; that it is derived and not stored, so every shard has it and no shard changed; the grid
fact position x distance of the far set; and that the early set fills its first row."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest
from hsn.data.dataset import ByteDataset
from hsn.data.splits import (
    SplitManifest,
    pack_examples,
    read_shard,
    shard_fact_positions,
    write_shard,
)
from hsn.data.synthetic import (
    NONE,
    QRY,
    SEP,
    SPLITS,
    DatasetSpec,
    Example,
    TaskSpec,
    check_example,
    fact_positions,
    generate,
    load_dataset_spec,
    main,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "configs" / "data"
V1 = load_dataset_spec(DATA / "synthetic_v1.yaml")
FAR = load_dataset_spec(DATA / "synthetic_far.yaml")
EARLY = load_dataset_spec(DATA / "synthetic_far_early.yaml")
SETS = {"synthetic_far": FAR, "synthetic_far_early": EARLY}
WINDOW = 256
DEV_KEYS = range(112, 128)
OTHER_KEYS = [*range(64, 112), *range(128, 144)]  # the train and the test keys
VALUES = range(144, 192)
INS, OVR, DEL = 16, 17, 18
#: the far set that the analysis of 18 Nov 2026 built and scored
#: (``hsn-experiments/oor-recall/data/synthetic_far/manifest.json``)
FAR_SPEC_HASH = "c4b0640cb8eeaf404b7c0f82480fb0e8596ed9249671c68ba5444de8229af4c1"
FAR_DEV_HASH = "d64a28c21a70678217bb3c9322941f06425492e106d5de30c659f7f7bc54c268"
#: the grid bands of that analysis (``pos_analysis.py``): fact position and distance, inclusive
FACT_BANDS = [(0, 255), (256, 767), (768, 1535), (1536, 3200)]
DISTANCE_BANDS = [(256, 767), (768, 1535), (1536, 3072)]


@functools.cache
def _dev(dataset: str, key: str) -> tuple[Example, ...]:
    """Every dev document of one task (``key``: ``name/variant``), generated once per process."""
    task = next(t for t in SETS[dataset].tasks if t.key == key)
    return tuple(generate(task, "dev", index) for index in range(task.n_dev))


def _k(task: TaskSpec) -> int:
    """How many windows an early task's anchor is moved (the two digits of its name)."""
    return int(task.name[-2:])


def _grid(examples: tuple[Example, ...]) -> list[list[int]]:
    """Answers per fact-position band (rows) and distance band (columns)."""
    fact = np.concatenate([fact_positions(e) for e in examples])
    deps = np.concatenate([np.asarray(e.meta["deps"], dtype=np.int64) for e in examples])
    return [
        [
            int(((fact >= lo) & (fact <= hi) & (deps >= dlo) & (deps <= dhi)).sum())
            for dlo, dhi in DISTANCE_BANDS
        ]
        for lo, hi in FACT_BANDS
    ]


# --------------------------------------------------------------------------------------------------
# the specs
# --------------------------------------------------------------------------------------------------


def test_the_far_tasks_are_v1s_apart_from_seeds_sizes_and_ranges() -> None:
    assert FAR.name == "synthetic_far" and FAR.unroll == V1.unroll == WINDOW
    assert [(t.name, t.n_dev) for t in FAR.tasks] == [("kv", 1000), ("delayed", 2000)]
    for task in FAR.tasks:
        source = V1.task(task.name, "out_of_range")
        assert task.variant == "out_of_range" and task.family == source.family == task.name
        assert task.context_id == source.context_id and task.params == source.params
        assert task.seed != source.seed
        assert (task.n_train, task.n_test) == (0, 0)  # evaluation only, dev only
        assert task.dep_len_range == (272, 3072) and task.length_range == (288, 3200)
        assert task.dep_len_range[0] == source.dep_len_range[0]
        assert task.dep_len_range[1] == 4 * source.dep_len_range[1]


def test_the_early_tasks_are_v1s_in_range_tasks_moved_whole_windows() -> None:
    assert EARLY.name == "synthetic_far_early" and EARLY.unroll == WINDOW
    assert [t.name for t in EARLY.tasks] == [
        f"{family}_back{k:02d}" for family in ("kv", "delayed") for k in range(1, 12)
    ]
    for task in EARLY.tasks:
        source = V1.task(task.family, "in_range")
        shift = WINDOW * _k(task)
        assert task.dep_len_range == tuple(x + shift for x in source.dep_len_range)
        assert task.length_range == tuple(x + shift for x in source.length_range)
        assert task.params == source.params and task.context_id == source.context_id
        assert task.variant == "out_of_range"
        assert (task.n_train, task.n_dev, task.n_test) == (0, 200, 0)
    # the distances stay inside the far set's
    assert EARLY.tasks[-1].dep_len_range[1] <= FAR.tasks[0].dep_len_range[1]
    assert EARLY.tasks[0].dep_len_range[0] == FAR.tasks[0].dep_len_range[0]


def test_seeds_are_new() -> None:
    taken: set[int] = set()
    for path in sorted(DATA.glob("synthetic_*.yaml")):
        if path.stem not in SETS:
            taken.update(task.seed for task in load_dataset_spec(path).tasks)
    assert len(taken) > 20
    seeds = [task.seed for dataset in SETS.values() for task in dataset.tasks]
    assert len(set(seeds)) == len(seeds) == 24 and not taken & set(seeds)


# --------------------------------------------------------------------------------------------------
# the documents
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SETS))
def test_documents_agree_with_the_solver_and_keep_their_ranges(name: str) -> None:
    for task in SETS[name].tasks:
        dep_lo, dep_hi = task.dep_len_range
        length_lo, length_hi = task.length_range
        keys: set[int] = set()
        for example in _dev(name, task.key):
            check_example(task, example)
            tokens = example.tokens
            assert length_lo <= example.length <= length_hi
            deps = example.meta["deps"]
            assert dep_lo <= deps[0] == example.meta["dep_len"] == max(deps) <= dep_hi
            assert not example.meta["in_range"] and example.meta["split"] == "dev"
            assert not np.isin(tokens, OTHER_KEYS).any()  # no key that training or test uses
            keys.update(int(b) for b in tokens[(tokens >= 112) & (tokens < 128)])
        assert keys == set(DEV_KEYS)
        for split in ("train", "test"):
            with pytest.raises(IndexError):
                generate(task, split, 0)  # there is nothing to train on


def test_generation_is_deterministic() -> None:
    for name, dataset in SETS.items():
        for task in dataset.tasks:
            first = _dev(name, task.key)[3]
            again = generate(task, "dev", 3)
            assert np.array_equal(first.tokens, again.tokens) and first.meta == again.meta
            assert np.array_equal(first.loss_mask, again.loss_mask)


# --------------------------------------------------------------------------------------------------
# the fact position
# --------------------------------------------------------------------------------------------------


def _check_delayed(example: Example) -> None:
    """Read from the tokens alone: the fact of a query is the one ``k v`` before SEP whose key
    is the query's; the fact position is its value byte."""
    t = example.tokens.tolist()
    sep = t.index(SEP)
    fact = fact_positions(example).tolist()
    positions = np.flatnonzero(example.loss_mask).tolist()
    assert len(fact) == len(positions) == len(example.meta["deps"]) >= 1
    for at, s, dep in zip(positions, fact, example.meta["deps"], strict=True):
        key, answer = t[at], t[at + 1]
        assert t[at - 1] == QRY and key in DEV_KEYS and answer in VALUES
        assert [i for i in range(sep) if t[i] == key] == [s - 1]  # stated once, just before
        assert t[s] == answer and 0 < s < sep < at and at - s == dep


def _check_kv(example: Example) -> None:
    """Read from the tokens alone: the fact of a query is the last operation before SEP on the
    query's key; the fact position is its last symbol (the value of an INS or OVR, the key of a
    DEL)."""
    t = example.tokens.tolist()
    sep = t.index(SEP)
    fact = fact_positions(example).tolist()
    positions = np.flatnonzero(example.loss_mask).tolist()
    assert len(fact) == len(positions) == len(example.meta["deps"]) >= 1
    for at, s, dep in zip(positions, fact, example.meta["deps"], strict=True):
        key, answer = t[at], t[at + 1]
        assert t[at - 1] == QRY and key in DEV_KEYS
        last = max(i for i in range(sep) if t[i] == key)  # key bytes occur in operations only
        if answer == NONE:
            assert t[last - 1] == DEL and s == last
        else:
            assert t[last - 1] in (INS, OVR) and s == last + 1 and t[s] == answer
        assert 0 < s < sep < at and at - s == dep


def test_fact_positions_point_at_the_fact() -> None:
    """On the far sets and on synthetic_v1 (the function reads documents of any age)."""
    check = {"delayed": _check_delayed, "kv": _check_kv}
    seen = 0
    for name, dataset in SETS.items():
        for task in dataset.tasks:
            for example in _dev(name, task.key)[:150]:
                check[task.family](example)
                seen += 1
    for task in V1.tasks:
        if task.family in check:
            for index in range(150):
                check[task.family](generate(task, "dev", index))
                seen += 1
    assert seen == 2 * 150 + 22 * 150 + 4 * 150


def test_fact_positions_for_every_family() -> None:
    """Aligned with ``deps``, inside the document and before the answer, for all ten tasks of
    synthetic_v1 and for a document without answers."""
    for task in V1.tasks:
        for index in range(40):
            example = generate(task, "test", index)
            fact = fact_positions(example)
            positions = np.flatnonzero(example.loss_mask)
            assert fact.dtype == np.int64 and fact.shape == positions.shape
            assert np.array_equal(positions - fact, np.asarray(example.meta["deps"]))
            assert (fact >= 1).all() and (fact < positions).all()  # after BOS, before the answer
    empty = Example(
        np.asarray([1, 200, 2], dtype=np.uint8),
        np.asarray([200, 2, 0], dtype=np.uint8),
        np.zeros(3, dtype=np.bool_),
        np.zeros(3, dtype=np.uint8),
        {**generate(V1.tasks[0], "dev", 0).meta, "deps": (), "dep_len": 0},
    )
    assert fact_positions(empty).shape == (0,)


def test_shard_fact_positions_are_derived_not_stored(tmp_path: Path) -> None:
    """A shard's fact positions are its documents', in order and aligned with ``deps``, from
    the arrays in memory, from ``read_shard`` and from the file; the file holds what it held
    before (no shard of any dataset changed), and an empty shard gives an empty array."""
    task = FAR.task("delayed", "out_of_range")
    kv = FAR.task("kv", "out_of_range")
    for spec, n in ((task, 60), (kv, 40)):
        examples = list(_dev("synthetic_far", spec.key)[:n])
        expected = np.concatenate([fact_positions(e) for e in examples])
        arrays = pack_examples(examples)
        record = write_shard(tmp_path, spec, "dev", examples)
        from_memory = shard_fact_positions(arrays)
        assert from_memory.dtype == np.int32 and from_memory.shape == arrays["deps"].shape
        assert np.array_equal(from_memory, expected) and record.n_loss == expected.shape[0]
        on_disk, _ = read_shard(tmp_path / record.file)
        assert np.array_equal(shard_fact_positions(on_disk), expected)
        with np.load(tmp_path / record.file, allow_pickle=False) as file:
            assert np.array_equal(shard_fact_positions(file), expected)
            assert sorted(file.files) == sorted(
                ["tokens", "loss_mask", "deps", "offsets", "index"]
                + ["format", "task", "variant", "split", "family", "context_id", "unroll"]
            )
        # the position counts from the document's start, wherever the document lies in the shard
        answers = np.cumsum([0, *(len(e.meta["deps"]) for e in examples)])
        assert np.array_equal(from_memory[answers[-2] :], fact_positions(examples[-1]))
    empty = pack_examples([])
    assert shard_fact_positions(empty).shape == (0,)
    assert shard_fact_positions(empty).dtype == np.int32
    whole = pack_examples(list(_dev("synthetic_far", kv.key)[:5]))
    with pytest.raises(ValueError, match="loss positions"):
        shard_fact_positions({**whole, "deps": whole["deps"][:-1]})


def test_the_far_set_mixes_distance_and_fact_position() -> None:
    """The grid of the analysis of 18 Nov 2026 (its ``pos_analysis.txt``), answers 256 bytes or
    more back: rows are the fact's position, columns the distance. A fact asked a short way
    back sits late, so the first row is thin on the left; a late fact cannot be asked from
    far, so the last row is thin on the right."""
    delayed = _grid(_dev("synthetic_far", "delayed/out_of_range"))
    assert delayed == [[88, 136, 937], [170, 361, 971], [212, 457, 431], [281, 266, 14]]
    assert sum(map(sum, delayed)) == 4324  # every delayed answer is 256 or more back
    kv = _grid(_dev("synthetic_far", "kv/out_of_range"))
    assert kv == [[21, 37, 222], [37, 57, 227], [61, 108, 101], [155, 67, 0]]
    assert sum(map(sum, kv)) == 1093  # of 7,887; the others ask the middle from nearby


def test_the_early_set_fills_the_first_row() -> None:
    """Every delayed fact and every kv anchor fact lies in the first window and its answer in
    the last one, k windows later; the first row of the grid holds every delayed answer."""
    delayed: list[Example] = []
    kv: list[Example] = []
    for task in EARLY.tasks:
        k = _k(task)
        for example in _dev("synthetic_far_early", task.key):
            assert WINDOW * k < example.length <= WINDOW * (k + 1)  # k + 1 windows exactly
            fact = fact_positions(example)
            positions = np.flatnonzero(example.loss_mask)
            deps = np.asarray(example.meta["deps"])
            assert fact[0] < WINDOW and positions[0] // WINDOW == k
            assert 16 + WINDOW * k <= deps[0] <= 240 + WINDOW * k
            if task.family == "delayed":
                assert (fact < WINDOW).all() and (positions // WINDOW == k).all()
                assert (deps > WINDOW * (k - 1) + 240).all()
        (delayed if task.family == "delayed" else kv).extend(_dev("synthetic_far_early", task.key))
    grid = _grid(tuple(delayed))
    assert grid == [[830, 1272, 2412], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert sum(len(e.meta["deps"]) for e in delayed) == 830 + 1272 + 2412
    assert _grid(tuple(kv))[0] == [420, 600, 1200]  # the anchors: 200 documents per task


def test_an_early_document_is_an_in_range_document_with_a_longer_gap() -> None:
    """With synthetic_v1's in-range seed in place of its own, an early task draws the in-range
    document's distance and length plus 256 k; a delayed document is then the in-range
    document, symbol for symbol, with 256 k more filler between the facts and SEP."""
    for task in EARLY.tasks:
        source = V1.task(task.family, "in_range")
        shift = WINDOW * _k(task)
        twin = TaskSpec.model_validate({**task.model_dump(), "seed": source.seed, "n_dev": 20})
        for index in range(20):
            near, far = generate(source, "dev", index), generate(twin, "dev", index)
            assert far.length == near.length + shift
            assert far.meta["dep_len"] == near.meta["dep_len"] + shift
            if task.family != "delayed":
                continue
            a, b = near.tokens, far.tokens
            sep = int(np.flatnonzero(a == SEP)[0])
            assert np.array_equal(a[:sep], b[:sep])  # prefix, facts, the first filler bytes
            assert (b[sep : sep + shift] >= 192).all()  # the added bytes are filler
            assert np.array_equal(a[sep:], b[sep + shift :])  # SEP, the queries, EOS
            assert np.array_equal(fact_positions(near), fact_positions(far))


# --------------------------------------------------------------------------------------------------
# the build and the committed manifests
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SETS))
def test_the_build_reproduces_the_committed_manifest(
    name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The documented build command, run twice: the same manifest text as the committed one,
    then ``verify``; the built set loads through its manifest with every hash checked."""
    dataset: DatasetSpec = SETS[name]
    spec, golden_path = DATA / f"{name}.yaml", DATA / f"{name}.manifest.json"
    golden = SplitManifest.load(golden_path)
    assert golden.name == name and golden.spec_hash == dataset.spec_hash()
    assert golden.unroll == dataset.unroll and golden.leakage == [] and golden.is_clean
    assert [s.key for s in golden.shards] == [
        f"{task.key}/{split}" for task in dataset.tasks for split in SPLITS
    ]
    for record in golden.shards:
        task = dataset.task(record.task, record.variant)
        assert record.n_docs == task.n_examples(record.split)
        assert record.context_id == task.context_id and record.family == task.family
        if record.split == "dev":
            examples = _dev(name, task.key)
            assert record.n_tokens == sum(e.length for e in examples)
            assert record.n_loss == sum(len(e.meta["deps"]) for e in examples)
        else:
            assert (record.n_docs, record.n_tokens, record.n_loss) == (0, 0, 0)
    if name == "synthetic_far":  # the set that was scored on 18 Nov 2026
        assert golden.spec_hash == FAR_SPEC_HASH and golden.split_hashes["dev"] == FAR_DEV_HASH
        assert golden.shard("kv", "out_of_range", "dev").n_loss == 7887
        assert golden.shard("delayed", "out_of_range", "dev").n_loss == 4324
    command = ["build", "--spec", str(spec), "--no-leakage", "--golden", str(golden_path)]
    for out in (tmp_path / "a", tmp_path / "b"):
        assert main([*command, "--out", str(out)]) == 0
        assert (out / "manifest.json").read_text("utf-8") == golden_path.read_text("utf-8")
    assert "matches the golden file" in capsys.readouterr().out
    for record in golden.shards:  # the two builds wrote the same files, byte for byte
        first, second = (tmp_path / d / record.file for d in ("a", "b"))
        assert first.read_bytes() == second.read_bytes()
    assert main(["verify", "--spec", str(spec), "--out", str(tmp_path / "a")]) == 0
    manifest = tmp_path / "a" / "manifest.json"
    dev = ByteDataset.load(manifest, "dev", expected_unroll=WINDOW)
    assert len(dev) == sum(task.n_dev for task in dataset.tasks)
    assert len(ByteDataset.load(manifest, "train")) == 0
    first_task = dataset.tasks[0]
    some = ByteDataset.load(manifest, "dev", tasks=[first_task.name], variants=["out_of_range"])
    assert len(some) == first_task.n_dev
    loaded, made = some.examples[5], _dev(name, first_task.key)[5]
    assert np.array_equal(loaded.tokens, made.tokens) and loaded.meta == made.meta
    assert np.array_equal(fact_positions(loaded), fact_positions(made))
