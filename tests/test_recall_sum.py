"""The ``recall_sum`` task family (14 Nov 2026, "recall, then compute"; ``configs/data/
synthetic_sum.yaml``): digits stated as facts ``k d``, then sums ``QRY x PLUS y EQ s SEP`` whose
operands are named by key (recalled) or written as digits. These tests check the documents
against an independent reading of the token stream, the anchor dependency, the minimum distance
of every recalled operand, the three kinds of item, the splits' key ranges, the parameter
checks and the committed manifest."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from hsn.data.splits import SplitManifest, write_shard
from hsn.data.synthetic import (
    EOS,
    QRY,
    SEP,
    SPLITS,
    Example,
    TaskSpec,
    check_example,
    generate,
    load_dataset_spec,
    reference_targets,
)

REPO = Path(__file__).resolve().parents[1]
DATASET = load_dataset_spec(REPO / "configs" / "data" / "synthetic_sum.yaml")
GOLDEN_PATH = REPO / "configs" / "data" / "synthetic_sum.manifest.json"
IN_RANGE, OUT_OF_RANGE = DATASET.tasks
PLUS, EQ = 19, 20
D0 = 48  # the first digit symbol


def _with(spec: TaskSpec, top: dict[str, object] | None = None, **params: object) -> TaskSpec:
    dump = {**spec.model_dump(), **(top or {})}
    dump["params"] = {**dump["params"], **params}
    return TaskSpec.model_validate(dump)


def _read(example: Example, key_range: tuple[int, int]) -> tuple[dict[int, int], list[dict]]:
    """The facts (key -> value position) and the items of a document, read off its symbols:
    each item's ``EQ`` position, the positions of its operands' sources, how many operands are
    recalled, and its answer (the sum's digit symbols and ``SEP``)."""
    lo, hi = key_range
    t = example.tokens.tolist()
    i = 1
    while not lo <= t[i] < hi:
        i += 1
    value_pos: dict[int, int] = {}
    while lo <= t[i] < hi:
        value_pos[t[i]] = i + 1
        i += 2
    while t[i] != SEP:
        i += 1
    i += 1
    items = []
    while t[i] != EOS:
        assert (t[i], t[i + 2], t[i + 4]) == (QRY, PLUS, EQ)
        total, sources, recalled = 0, [], 0
        for at in (i + 1, i + 3):
            if t[at] in value_pos:
                total += t[value_pos[t[at]]] - D0
                sources.append(value_pos[t[at]])
                recalled += 1
            else:
                assert D0 <= t[at] < D0 + 10
                total += t[at] - D0
                sources.append(at)
        answer = [D0 + int(c) for c in reversed(str(total))] + [SEP]
        items.append({"eq": i + 4, "sources": sources, "recalled": recalled, "answer": answer})
        i += 5 + len(answer)
    assert i == len(t) - 1
    return value_pos, items


@pytest.mark.parametrize("task", DATASET.tasks, ids=lambda t: t.variant)
@pytest.mark.parametrize("split", ["dev", "test"])
def test_documents_agree_with_an_independent_reading(task: TaskSpec, split: str) -> None:
    lo, hi = task.params["keys"][split]
    for index in range(150):
        example = generate(task, split, index)
        assert not check_example(task, example)
        value_pos, items = _read(example, (lo, hi))
        positions = np.flatnonzero(example.loss_mask).tolist()
        expected_positions, expected_targets, expected_deps = [], [], []
        for item in items:
            for j, target in enumerate(item["answer"]):
                expected_positions.append(item["eq"] + j)
                expected_targets.append(target)
                expected_deps.append(item["eq"] + j - min(item["sources"]))
        assert positions == expected_positions
        assert example.targets[positions].tolist() == expected_targets
        assert list(example.meta["deps"]) == expected_deps
        solved = reference_targets(task, example.tokens)
        assert solved[0].tolist() == positions and solved[1].tolist() == expected_targets
        # the splits' keys are disjoint by range, and every stated key lies in this split's
        assert value_pos and all(lo <= key < hi for key in value_pos)


@pytest.mark.parametrize("task", DATASET.tasks, ids=lambda t: t.variant)
def test_anchor_gap_and_kinds(task: TaskSpec) -> None:
    """The longest dependency is the drawn anchor and belongs to the last item's ``SEP``; every
    recalled operand lies ``min_gap`` or more before its item's ``EQ``; all three kinds occur
    and most items recall."""
    gap = task.params["min_gap"]
    dep_lo, dep_hi = task.dep_len_range
    len_lo, len_hi = task.length_range
    kinds: Counter[int] = Counter()
    for index in range(300):
        example = generate(task, "dev", index)
        value_pos, items = _read(example, task.params["keys"]["dev"])
        deps = list(example.meta["deps"])
        assert dep_lo <= example.meta["dep_len"] == max(deps) == deps[-1] <= dep_hi
        assert len_lo <= example.length <= len_hi
        assert example.meta["in_range"] == (task.variant == "in_range")
        assert 1 <= len(items) <= task.params["max_items"]
        assert items[-1]["recalled"] >= 1 and min(items[-1]["sources"]) == min(value_pos.values())
        for item in items:
            kinds[item["recalled"]] += 1
            for source in item["sources"]:
                if source < item["eq"] - 3:  # a fact's value, not a digit of the item
                    assert item["eq"] - source >= gap
    total = sum(kinds.values())
    assert min(kinds[0], kinds[1], kinds[2]) > 0.05 * total
    assert kinds[1] + kinds[2] > 0.7 * total


def test_deterministic_and_split_dependent() -> None:
    a, b = generate(IN_RANGE, "train", 7), generate(IN_RANGE, "train", 7)
    assert np.array_equal(a.tokens, b.tokens) and a.meta == b.meta
    assert not np.array_equal(a.tokens, generate(IN_RANGE, "train", 8).tokens)
    train_keys = set(_read(a, IN_RANGE.params["keys"]["train"])[0])
    dev_keys = set(_read(generate(IN_RANGE, "dev", 7), IN_RANGE.params["keys"]["dev"])[0])
    assert train_keys and dev_keys and not train_keys & dev_keys


def test_weights_choose_the_kinds_and_no_gap_gives_short_documents() -> None:
    only_two = _with(IN_RANGE, recall_weights=[1, 0, 0])
    no_memory_but_the_last = _with(IN_RANGE, recall_weights=[0, 1, 0])
    for index in range(60):
        items = _read(generate(only_two, "train", index), (64, 112))[1]
        facts = len(_read(generate(only_two, "train", index), (64, 112))[0])
        assert all(item["recalled"] == min(2, facts) for item in items)
        items = _read(generate(no_memory_but_the_last, "train", index), (64, 112))[1]
        assert all(item["recalled"] == 1 for item in items)
    short = _with(
        IN_RANGE, {"length_range": [16, 64], "dep_len_range": [8, 48]}, min_gap=0, max_facts=3
    )
    for index in range(100):
        example = generate(short, "train", index)
        assert not check_example(short, example) and example.length <= 64


@pytest.mark.parametrize(
    ("top", "params", "message"),
    [
        ({}, {"digits": [48, 57]}, "exactly ten symbols"),
        ({}, {"recall_weights": [0, 0, 1]}, "positive weight"),
        ({}, {"recall_weights": [1, -1, 1]}, "non-negative"),
        ({}, {"markers": {"plus": 19}}, "missing"),
        ({}, {"keys": {"train": [64, 112], "dev": [112, 116], "test": [128, 144]}}, "max_facts"),
        ({}, {"filler": [50, 256]}, "both use byte"),
        ({"dep_len_range": [40, 240]}, {}, "must be >= 66"),  # 8 + (min_gap 64 - 6)
        ({"length_range": [96, 200]}, {}, "too short"),
    ],
)
def test_spec_validation_rejects(
    top: dict[str, object], params: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _with(IN_RANGE, top, **params)


def test_golden_manifest_matches(tmp_path: Path) -> None:
    """The committed manifest hashes are reproduced here (CI: on both runners)."""
    golden = SplitManifest.load(GOLDEN_PATH)
    assert golden.name == DATASET.name == "synthetic_sum"
    assert golden.spec_hash == DATASET.spec_hash()
    assert golden.unroll == DATASET.unroll and golden.is_clean
    assert [s.key for s in golden.shards] == [
        f"{task.key}/{split}" for task in DATASET.tasks for split in SPLITS
    ]
    for task in DATASET.tasks:
        for split in SPLITS:
            assert golden.shard(task.name, task.variant, split).n_docs == task.n_examples(split)
        record = golden.shard(task.name, task.variant, "test")
        examples = [generate(task, "test", index) for index in range(task.n_test)]
        assert write_shard(tmp_path, task, "test", examples).hash == record.hash
        assert sum(e.length for e in examples) == record.n_tokens
