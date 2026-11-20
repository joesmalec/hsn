"""The ``recall_sum_words`` task family (19 Nov 2026, ``configs/data/synthetic_sumwords.yaml``):
``recall_sum`` with keys that are words of letters, for the word events of HSN-T. These tests
check the documents against an independent reading of the token stream and the reference
solver, the item kinds, the anchor and the minimum distance of every recalled operand,
determinism, the split vocabularies (``synthetic_wordkeys``' words, disjoint between splits),
that under the word rule every key is one word and a fact pairs its key with its digit (the key's
code in ``a_prev`` at the digit's event, in ``a_cur`` where an item names it), the parameter
checks, that HSN-T with word events trains on the task and scores it under its own name, and the
committed manifest."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hsn.data.dataset import ByteDataset
from hsn.data.splits import SplitManifest, write_shard
from hsn.data.synthetic import (
    BOS,
    EOS,
    QRY,
    SEP,
    SPLITS,
    Example,
    RecallSumWordsParams,
    TaskSpec,
    check_example,
    generate,
    load_dataset_spec,
    recall_sum_kinds,
    reference_targets,
)
from hsn.device import resolve, seed_everything
from hsn.model.reader import word_ends
from hsn.model.word_address import word_address, word_tables
from hsn.runner import build_model
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import UNROLL, hsn_config
from test_a6_feedback_detach import A5
from test_hsn_t import CUT_C, READER

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "configs" / "data"
DATASET = load_dataset_spec(DATA / "synthetic_sumwords.yaml")
GOLDEN_PATH = DATA / "synthetic_sumwords.manifest.json"
IN_RANGE, OUT_OF_RANGE = DATASET.tasks
PLUS, EQ, SPACE = 19, 20, 32
D0 = 48  # the first digit symbol
LETTERS = set(range(97, 123))
DIGITS = set(range(48, 58))
FILLER = {*range(33, 39), *range(40, 48), *range(58, 65), *range(91, 97), *range(123, 127)}
CONTEXT = 9


def _with(spec: TaskSpec, top: dict[str, object] | None = None, **params: object) -> TaskSpec:
    dump = {**spec.model_dump(), **(top or {})}
    dump["params"] = {**dump["params"], **params}
    return TaskSpec.model_validate(dump)


def _read(example: Example) -> tuple[dict[tuple[int, ...], dict[str, Any]], list[dict[str, Any]]]:
    """The facts (key word -> its start, its digit's position and value) and the items of a
    document, read off its symbols: every maximal run of letters is a key word, the bytes
    around it say what it is; an item is the run between two ``SEP`` (each operand a word or
    one digit), with its ``EQ`` position, its operands' sources, how many it recalls, the
    position that closes each key operand, and its answer (the sum's digits and ``SEP``)."""
    t = example.tokens.tolist()
    sep = t.index(SEP)
    facts: dict[tuple[int, ...], dict[str, Any]] = {}
    i = 1
    while i < sep:
        if t[i] not in LETTERS:
            assert t[i] in FILLER, (i, t[i])
            i += 1
            continue
        end = i
        while t[end] in LETTERS:
            end += 1
        word = tuple(t[i:end])
        assert t[i - 1] in FILLER | {BOS, SPACE} and word not in facts
        assert t[end] == SPACE and t[end + 1] in DIGITS and t[end + 2] == SPACE
        facts[word] = {"start": i, "digit": end + 1, "value": t[end + 1] - D0}
        i = end + 3
    items = []
    at = sep + 1
    while t[at] != EOS:
        stop = t.index(SEP, at)
        item = t[at:stop]
        assert item[0] == QRY and item.count(PLUS) == 1 and item.count(EQ) == 1
        p, q = item.index(PLUS), item.index(EQ)
        total, sources, recalled, closes = 0, [], 0, []
        for first, operand, closer in ((1, item[1:p], p), (p + 1, item[p + 1 : q], q)):
            if len(operand) == 1 and operand[0] in DIGITS:
                total += operand[0] - D0
                sources.append(at + first)
            else:
                fact = facts[tuple(operand)]
                total += fact["value"]
                sources.append(fact["digit"])
                recalled += 1
                closes.append((tuple(operand), at + closer))
        answer = [D0 + int(c) for c in reversed(str(total))]
        assert item[q + 1 :] == answer
        items.append(
            {
                "plus": at + p,
                "eq": at + q,
                "sources": sources,
                "recalled": recalled,
                "answer": [*answer, SEP],
                "closes": closes,
                "sep": stop,
            }
        )
        at = stop + 1
    assert at == len(t) - 1
    return facts, items


@pytest.mark.parametrize("task", DATASET.tasks, ids=lambda t: t.variant)
@pytest.mark.parametrize("split", SPLITS)
def test_documents_agree_with_an_independent_reading(task: TaskSpec, split: str) -> None:
    if task.n_examples(split) == 0:
        pytest.skip("no documents")
    allowed = {BOS, EOS, QRY, SEP, PLUS, EQ, SPACE, *LETTERS, *DIGITS, *FILLER}
    for index in range(120):
        example = generate(task, split, index)
        check_example(task, example)
        assert set(example.tokens.tolist()) <= allowed and int(example.context[0]) == CONTEXT
        facts, items = _read(example)
        positions = np.flatnonzero(example.loss_mask).tolist()
        expected_positions, expected_targets, expected_deps, kinds = [], [], [], []
        for item in items:
            for j, target in enumerate(item["answer"]):
                expected_positions.append(item["eq"] + j)
                expected_targets.append(target)
                expected_deps.append(item["eq"] + j - min(item["sources"]))
                kinds.append(item["recalled"])
        assert positions == expected_positions
        assert example.targets[positions].tolist() == expected_targets
        assert list(example.meta["deps"]) == expected_deps
        solved = reference_targets(task, example.tokens)
        assert solved[0].tolist() == positions and solved[1].tolist() == expected_targets
        assert solved[2].tolist() == expected_deps
        assert recall_sum_kinds(task, example.tokens).tolist() == kinds
        assert facts and all(3 <= len(word) <= 8 for word in facts)


@pytest.mark.parametrize("task", DATASET.tasks, ids=lambda t: t.variant)
def test_anchor_gap_and_kinds(task: TaskSpec) -> None:
    """The longest dependency is the drawn anchor and belongs to the last item's ``SEP``; every
    recalled operand's digit lies ``min_gap`` or more before its item's ``EQ``; all three kinds
    occur and most items recall, about as in ``recall_sum``."""
    gap = task.params["min_gap"]
    dep_lo, dep_hi = task.dep_len_range
    len_lo, len_hi = task.length_range
    kinds: Counter[int] = Counter()
    n_facts = 0
    for index in range(300):
        example = generate(task, "dev", index)
        facts, items = _read(example)
        n_facts += len(facts)
        deps = list(example.meta["deps"])
        assert dep_lo <= example.meta["dep_len"] == max(deps) == deps[-1] <= dep_hi
        assert len_lo <= example.length <= len_hi
        assert example.meta["in_range"] == (task.variant == "in_range")
        assert 2 <= len(facts) <= task.params["max_facts"]
        assert 1 <= len(items) <= task.params["max_items"]
        first = min(fact["digit"] for fact in facts.values())
        assert items[-1]["recalled"] >= 1 and min(items[-1]["sources"]) == first
        recalled_digits = {fact["digit"] for fact in facts.values()}
        for item in items:
            kinds[item["recalled"]] += 1
            for source in item["sources"]:
                if source in recalled_digits:
                    assert item["eq"] - source >= gap
    total = sum(kinds.values())
    assert min(kinds[0], kinds[1], kinds[2]) > 0.05 * total
    assert kinds[1] + kinds[2] > 0.7 * total
    assert n_facts / 300 > 2.5


def test_deterministic_and_the_splits_use_disjoint_words() -> None:
    a, b = generate(IN_RANGE, "train", 7), generate(IN_RANGE, "train", 7)
    assert np.array_equal(a.tokens, b.tokens) and a.meta == b.meta
    assert not np.array_equal(a.tokens, generate(IN_RANGE, "train", 8).tokens)
    params = IN_RANGE.typed_params
    assert isinstance(params, RecallSumWordsParams)
    vocab = {split: {w for g in params.vocabulary[split] for w in g} for split in SPLITS}
    assert not vocab["train"] & vocab["dev"] and not vocab["train"] & vocab["test"]
    assert not vocab["dev"] & vocab["test"]
    # synthetic_wordkeys' words, word for word
    (kvw,) = load_dataset_spec(DATA / "synthetic_wordkeys.yaml").tasks
    assert params.vocabulary == kvw.typed_params.vocabulary  # type: ignore[attr-defined]
    for task in DATASET.tasks:
        for split in SPLITS:
            for index in range(min(100, task.n_examples(split))):
                facts, _ = _read(generate(task, split, index))
                assert set(facts) <= vocab[split]


@pytest.mark.parametrize("kind", ["exact", "bag"])
def test_every_key_is_one_word_and_a_fact_binds_under_the_word_address(kind: str) -> None:
    """With context 9 read as text the events of a document are BOS, per fact the SPACE after
    its word and the SPACE after its digit, per item PLUS, EQ and the SEP after the sum: ``1 +
    2 n + 3 m``. At a fact's digit event ``a_prev`` is its key's word code (and ``a_cur`` the
    digit's), the code ``a_cur`` has where an item's key operand closes; another key has
    another code. The reader's word ends (bytes >= 128 count as letters there; there are none)
    are the same positions."""
    g = torch.Generator().manual_seed(0)
    byte_codes = torch.randn(256, 64, generator=g)
    tables = word_tables(kind, 256, 64, g)
    for task in DATASET.tasks:
        for index in range(30):
            example = generate(task, "dev", index)
            tokens = torch.from_numpy(example.tokens.astype(np.int64))[None]
            reset = torch.zeros_like(tokens, dtype=torch.bool)
            reset[:, 0] = True
            text = torch.from_numpy(example.context == CONTEXT)[None]
            out = word_address(kind, tables, byte_codes, tokens, text, reset, None)
            facts, items = _read(example)
            expected = [0]
            for fact in facts.values():
                expected += [fact["digit"] - 1, fact["digit"] + 1]
            for item in items:
                expected += [item["plus"], item["eq"], item["sep"]]
            events = torch.nonzero(out.events[0]).flatten().tolist()
            assert events == sorted(expected)
            assert len(events) == 1 + 2 * len(facts) + 3 * len(items)
            ends = word_ends(tokens, reset, torch.zeros(1, dtype=torch.bool))
            assert torch.equal(ends[0], out.events[0] & ~reset[0])
            code = {}
            for word, fact in facts.items():
                closes_word = fact["digit"] - 1  # the SPACE after the word
                at = fact["digit"] + 1  # the SPACE after the digit: the digit's event
                code[word] = out.current[0, closes_word]
                assert torch.equal(out.previous[0, at], code[word])
            for item in items:
                for word, at in item["closes"]:
                    assert torch.equal(out.current[0, at], code[word])
                    for other, value in code.items():
                        if other != word:
                            assert not torch.allclose(out.current[0, at], value)


def test_a_digit_has_one_code_wherever_it_stands() -> None:
    """A fact's digit and a written digit operand are words of one byte: the same digit has
    the same code in ``a_cur`` at its event, different digits different codes."""
    g = torch.Generator().manual_seed(1)
    byte_codes = torch.randn(256, 64, generator=g)
    tables = word_tables("exact", 256, 64, g)
    seen: dict[int, torch.Tensor] = {}
    for index in range(40):
        example = generate(IN_RANGE, "train", index)
        t = example.tokens.tolist()
        tokens = torch.from_numpy(example.tokens.astype(np.int64))[None]
        reset = torch.zeros_like(tokens, dtype=torch.bool)
        reset[:, 0] = True
        out = word_address("exact", tables, byte_codes, tokens, torch.ones_like(reset), reset)
        for at in torch.nonzero(out.events[0]).flatten().tolist():
            if at and t[at - 1] in DIGITS and t[at - 2] not in DIGITS:
                digit = t[at - 1]
                if digit in seen:
                    assert torch.equal(out.current[0, at], seen[digit])
                seen[digit] = out.current[0, at]
    assert set(seen) == DIGITS
    codes = list(seen.values())
    assert all(not torch.equal(a, b) for i, a in enumerate(codes) for b in codes[i + 1 :])


@pytest.mark.parametrize(
    ("top", "params", "message"),
    [
        ({}, {"digits": [160, 170]}, "ASCII digits"),
        ({}, {"filler": [[33, 39], [65, 70]]}, "word byte"),  # capitals would join a word
        ({}, {"filler": [[33, 40]]}, "word byte"),  # the apostrophe
        ({}, {"filler": [[200, 256]]}, "below 128"),
        ({}, {"filler": [[30, 34]]}, "byte range"),
        ({}, {"filler": [[33, 39], [35, 37]]}, "both use byte"),
        ({}, {"filler": []}, "at least one"),
        ({}, {"space": 35}, "both use byte"),
        ({}, {"recall_weights": [0, 0, 1]}, "positive weight"),
        ({}, {"markers": {"plus": 19}}, "missing"),
        ({}, {"groups": {"train": 500, "dev": 1, "test": 125}}, "max_facts"),
        ({"dep_len_range": [90, 240]}, {}, "must be >= 91"),  # 3 x 8 + 10 + (64 - 7)
        ({"length_range": [96, 250]}, {}, "too short"),
    ],
)
def test_spec_validation_rejects(
    top: dict[str, object], params: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _with(IN_RANGE, top, **params)


def test_the_solver_rejects_malformed_documents() -> None:
    example = generate(IN_RANGE, "dev", 0)
    t = example.tokens.tolist()
    sep = t.index(SEP)
    first = next(i for i in range(1, sep) if t[i] in LETTERS)
    end = next(i for i in range(first, sep) if t[i] not in LETTERS)
    plus = t.index(PLUS, sep)
    broken = [list(t) for _ in range(5)]
    broken[0][end] = 40  # a fact without its SPACE
    broken[1][end + 1] = 97  # a fact whose digit is a letter: the word runs on
    broken[2][end + 2] = 50  # the digit glued to another digit
    broken[3][plus] = SPACE  # an item without PLUS
    broken[4] = t[:-1] + [QRY, EOS]  # a truncated item
    for tokens in broken:
        with pytest.raises(ValueError):
            reference_targets(IN_RANGE, np.asarray(tokens, dtype=np.uint8))


def test_kinds_need_a_recall_sum_family() -> None:
    (kvw,) = load_dataset_spec(DATA / "synthetic_wordkeys.yaml").tasks
    with pytest.raises(ValueError, match="recall_sum only"):
        recall_sum_kinds(kvw, generate(kvw, "dev", 0).tokens)


@pytest.mark.usefixtures("restore_determinism")
def test_hsn_t_with_word_events_trains_and_scores_the_task(tmp_path: Path) -> None:
    """Documents of the family go through the dataset, the window trainer of HSN-T with word
    events (the exact address, the task's context read as text) and the evaluation: finite
    losses, and a dev score reported for ``recall_sumw/in_range``."""
    tiny = _with(
        IN_RANGE,
        {
            "n_train": 32,
            "n_dev": 8,
            "n_test": 8,
            "length_range": [16, UNROLL],
            "dep_len_range": [22, 23],
            "unroll": UNROLL,
        },
        word_len=[3, 4],
        affix_len=[1, 2],
        min_gap=0,
    )
    train = ByteDataset([generate(tiny, "train", i) for i in range(32)])
    dev = ByteDataset([generate(tiny, "dev", i) for i in range(8)])
    steps = 6
    cfg = hsn_config(
        steps,
        *A5,
        CUT_C,
        "train.soft_until_step=0",
        *READER,
        "model.write_events=word",
        "model.word_address=exact",
        f"model.word_contexts=[0,{CONTEXT}]",
        "train.eval_every=3",
    )
    seed_everything(0)
    trainer = TBPTTTrainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    history = trainer.fit(train, dev, steps)
    assert len(history.steps) == steps and all(math.isfinite(s.loss) for s in history.steps)
    groups = history.evals[-1]["per_group"]
    assert set(groups) == {"recall_sumw/in_range"}
    assert 0.0 <= groups["recall_sumw/in_range"]["accuracy"] <= 1.0
    assert math.isfinite(groups["recall_sumw/in_range"]["loss"])


def test_golden_manifest_matches(tmp_path: Path) -> None:
    """The committed manifest hashes are reproduced here (CI: on both runners)."""
    golden = SplitManifest.load(GOLDEN_PATH)
    assert golden.name == DATASET.name == "synthetic_sumwords"
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
