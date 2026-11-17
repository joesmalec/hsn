"""The ``kv_words`` task family (16 Nov 2026, ``configs/data/synthetic_wordkeys.yaml``): key-value
with keys that are words of letters, for the word address of HSN-T's word events. These tests
check the documents against an independent reading of the token stream and the reference
solver, the anchor and the far queries, determinism, the split vocabularies (disjoint words, the
same letters), the shared beginnings and endings, the parameter checks, that a fact binds under
the word address (the key's code at the write and at the read), that the task trains and is
scored under its own name, and the committed manifest."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from hsn.data.dataset import ByteDataset
from hsn.data.splits import SplitManifest, write_shard
from hsn.data.synthetic import (
    BOS,
    EOS,
    NONE,
    QRY,
    SEP,
    SPLITS,
    Example,
    KVWordsParams,
    TaskSpec,
    check_example,
    generate,
    load_dataset_spec,
    reference_targets,
)
from hsn.model.word_address import word_address, word_tables
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import hsn_config, hsn_trainer

REPO = Path(__file__).resolve().parents[1]
DATASET = load_dataset_spec(REPO / "configs" / "data" / "synthetic_wordkeys.yaml")
GOLDEN_PATH = REPO / "configs" / "data" / "synthetic_wordkeys.manifest.json"
(TASK,) = DATASET.tasks
INS, OVR, DEL, SPACE = 16, 17, 18, 32
LETTERS = range(97, 123)


def _with(spec: TaskSpec, top: dict[str, object] | None = None, **params: object) -> TaskSpec:
    dump = {**spec.model_dump(), **(top or {})}
    dump["params"] = {**dump["params"], **params}
    return TaskSpec.model_validate(dump)


def _small(n: int = 200) -> TaskSpec:
    return _with(TASK, {"n_train": n, "n_dev": n, "n_test": n})


def _read(example: Example) -> tuple[list[dict], list[dict]]:
    """The operations and queries of a document, read word by word: every maximal run of
    letters is a key word, and the bytes around it say what it is."""
    t = example.tokens.tolist()
    sep = t.index(SEP)
    ops, queries = [], []
    i = 0
    while i < len(t):
        if t[i] not in LETTERS:
            i += 1
            continue
        end = i
        while t[end] in LETTERS:
            end += 1
        word, before, after = tuple(t[i:end]), t[i - 1], t[end]
        if i < sep:
            assert before in (INS, OVR, DEL), (i, before)
            if before == DEL:
                assert after == SPACE
                ops.append({"kind": "del", "word": word, "end": end, "value": None})
            else:
                assert 144 <= after < 192  # the value byte directly after the word
                ops.append({"kind": "set", "word": word, "end": end, "value": after})
        else:
            assert before == QRY and after == SPACE
            queries.append({"word": word, "space": end, "answer": t[end + 1]})
        i = end
    return ops, queries


@pytest.mark.parametrize("split", SPLITS)
def test_documents_agree_with_an_independent_reading(split: str) -> None:
    task = _small()
    lo, hi = TASK.length_range
    for index in range(150):
        example = generate(task, split, index)
        check_example(task, example)
        t = example.tokens
        assert t[0] == BOS and t[-1] == EOS and lo <= example.length <= hi <= TASK.unroll
        ops, queries = _read(example)
        # below the values (144-191) and distractors (192-255): controls, markers, SPACE, letters
        assert np.isin(t[t < 144], [BOS, EOS, QRY, SEP, NONE, INS, OVR, DEL, SPACE, *LETTERS]).all()
        state: dict[tuple[int, ...], tuple[int | None, int]] = {}
        for op in ops:
            assert 3 <= len(op["word"]) <= 8
            state[op["word"]] = (op["value"], op["end"])
        positions = np.flatnonzero(example.loss_mask).tolist()
        assert positions == [q["space"] for q in queries]
        targets, deps = [], []
        for q in queries:
            value, end = state[q["word"]]
            targets.append(NONE if value is None else value)
            deps.append(q["space"] - end)
            assert q["answer"] == targets[-1]
        assert example.targets[positions].tolist() == targets
        assert list(example.meta["deps"]) == deps
        solved = reference_targets(task, t)
        assert solved[0].tolist() == positions and solved[1].tolist() == targets
        assert solved[2].tolist() == deps


def test_anchor_lengths_and_far_queries() -> None:
    """The first query holds the drawn anchor distance, the document's largest; every document
    fits one window; far queries (>= 63) and every kind of answer occur."""
    task = _small(400)
    dep_lo, dep_hi = TASK.dep_len_range
    far = queries = 0
    answers: Counter[str] = Counter()
    for index in range(400):
        example = generate(task, "train", index)
        deps = list(example.meta["deps"])
        assert dep_lo <= deps[0] == example.meta["dep_len"] == max(deps) <= dep_hi
        assert example.meta["in_range"] and example.length <= TASK.unroll
        far += sum(d >= 63 for d in deps)
        queries += len(deps)
        for target in example.targets[example.loss_mask].tolist():
            answers["none" if target == NONE else "value"] += 1
    assert far / 400 > 1.5 and far / queries > 0.4
    assert answers["none"] > 0.05 * queries and answers["value"] > 0.7 * queries


def test_deterministic_and_split_dependent() -> None:
    a, b = generate(TASK, "train", 7), generate(TASK, "train", 7)
    assert np.array_equal(a.tokens, b.tokens) and a.meta == b.meta
    assert not np.array_equal(a.tokens, generate(TASK, "train", 8).tokens)
    # the same (seed, split, index) whatever the task object, and interleaving changes nothing
    generate(TASK, "dev", 3)
    assert np.array_equal(generate(_small(), "train", 7).tokens, a.tokens)
    # the vocabulary is a function of the parameters alone
    fresh = KVWordsParams.model_validate(TASK.params)
    assert fresh.vocabulary == TASK.typed_params.vocabulary  # type: ignore[attr-defined]
    other = KVWordsParams.model_validate({**TASK.params, "vocab_seed": 1})
    assert other.vocabulary["train"] != fresh.vocabulary["train"]


def test_vocabularies_are_disjoint_and_made_of_the_same_letters() -> None:
    params = TASK.typed_params
    assert isinstance(params, KVWordsParams)
    words = {split: [w for g in params.vocabulary[split] for w in g] for split in SPLITS}
    sets = {split: set(ws) for split, ws in words.items()}
    for split in SPLITS:
        assert len(sets[split]) == len(words[split]) == params.groups[split] * params.group_size
        assert all(3 <= len(w) <= 8 and set(w) <= set(LETTERS) for w in words[split])
        # every letter, also as a first and as a last letter, occurs in every split
        for at in (slice(None), slice(0, 1), slice(-1, None)):
            assert {c for w in words[split] for c in w[at]} == set(LETTERS)
    assert not sets["train"] & sets["dev"] and not sets["train"] & sets["test"]
    assert not sets["dev"] & sets["test"]
    # a group's words share a beginning or an ending of two letters or more
    for split in SPLITS:
        for group in params.vocabulary[split]:
            assert len({w[:2] for w in group}) == 1 or len({w[-2:] for w in group}) == 1
    # the documents of a split use its words only, and no other words
    for split in SPLITS:
        for index in range(100):
            ops, queries = _read(generate(_small(), split, index))
            assert {op["word"] for op in ops} | {q["word"] for q in queries} <= sets[split]


def test_keys_share_beginnings_and_endings() -> None:
    """A share of the queried keys shares its last two letters (the per-byte address of a
    key) or its first two with another key of the document; without p_share that is rare."""

    def shares(task: TaskSpec) -> tuple[float, float, float]:
        ending = beginning = total = 0
        docs_with_ending = 0
        for index in range(300):
            ops, queries = _read(generate(task, "dev", index))
            keys = {op["word"] for op in ops}
            docs_with_ending += any(a != b and a[-2:] == b[-2:] for a in keys for b in keys)
            for q in queries:
                w = q["word"]
                ending += any(k != w and k[-2:] == w[-2:] for k in keys)
                beginning += any(k != w and k[:2] == w[:2] for k in keys)
                total += 1
        return ending / total, beginning / total, docs_with_ending / 300

    ending, beginning, docs = shares(_small(300))
    assert 0.15 < ending < 0.5 and 0.15 < beginning < 0.5 and docs > 0.3
    ending, beginning, _ = shares(_with(_small(300), p_share=0.0))
    assert ending < 0.05 and beginning < 0.05


@pytest.mark.parametrize(
    ("top", "params", "message"),
    [
        ({}, {"values": [60, 90]}, "word byte"),  # values that are letters would join a word
        ({}, {"space": 65}, "word byte"),
        ({}, {"distractors": [192, 256], "space": 200}, "both use byte"),
        ({}, {"letters": [91, 97]}, "ASCII letters"),
        ({}, {"affix_len": [2, 8]}, "affix_len"),
        ({}, {"word_len": [1, 8]}, "word_len"),
        ({}, {"groups": {"train": 500, "dev": 1, "test": 125}}, "n_keys"),
        ({}, {"groups": {"train": 500, "dev": 125}}, "every split"),
        ({}, {"markers": {"ins": 16, "ovr": 17}}, "missing"),
        ({"dep_len_range": [8, 200]}, {}, "must be >= 11"),
        ({"length_range": [32, 250]}, {}, "too short"),
    ],
)
def test_spec_validation_rejects(
    top: dict[str, object], params: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _with(TASK, top, **params)


def test_the_solver_rejects_malformed_documents() -> None:
    example = generate(TASK, "dev", 0)
    t = example.tokens.tolist()
    space = t.index(SPACE, t.index(SEP))  # the first query's SPACE
    broken = [list(t), list(t), list(t)]
    broken[0][space] = 150  # a query word without its SPACE
    broken[1][space - 1] = 200  # the word's last letter a distractor: no SPACE after the word
    broken[2] = t[:-1] + [QRY, EOS]  # a truncated query
    for tokens in broken:
        with pytest.raises(ValueError):
            reference_targets(TASK, np.asarray(tokens, dtype=np.uint8))


def test_a_fact_binds_under_the_word_address() -> None:
    """With context 7 read as text, the events of a document are BOS, every operation's last
    symbol (the value, or a DEL's SPACE) and every query's SPACE; the event of a query has its
    key's word code in a_cur, as the event of the operation that defined the answer does, and
    a key that only shares an ending with it has another code."""
    g = torch.Generator().manual_seed(0)
    byte_codes = torch.randn(256, 64, generator=g)
    for kind in ("exact", "bag"):
        tables = word_tables(kind, 256, 64, g)
        for index in range(40):
            example = generate(TASK, "dev", index)
            tokens = torch.from_numpy(example.tokens.astype(np.int64))[None]
            reset = torch.zeros_like(tokens, dtype=torch.bool)
            reset[:, 0] = True
            out = word_address(
                kind, tables, byte_codes, tokens, torch.ones_like(reset), reset, None
            )
            ops, queries = _read(example)
            events = torch.nonzero(out.events[0]).flatten().tolist()
            assert events == sorted([0, *(op["end"] for op in ops), *(q["space"] for q in queries)])
            code = {op["word"]: out.current[0, op["end"]] for op in ops}
            for q in queries:
                at = q["space"]
                last = max(op["end"] for op in ops if op["word"] == q["word"])
                assert torch.equal(out.current[0, at], out.current[0, last])
                for word, other in code.items():
                    if word != q["word"]:
                        assert not torch.allclose(out.current[0, at], other)


@pytest.mark.usefixtures("restore_determinism")
def test_the_task_trains_and_is_scored_under_its_own_name(tmp_path: Path) -> None:
    """Documents of the family go through the dataset, the window trainer and the evaluation:
    finite losses, and a dev score reported for ``kvw/in_range``."""
    unroll = 32
    tiny = _with(
        TASK,
        {
            "n_train": 32,
            "n_dev": 8,
            "n_test": 8,
            "length_range": [16, unroll],
            "dep_len_range": [7, 16],
            "unroll": unroll,
        },
        word_len=[3, 4],
        n_keys=3,
    )
    train = ByteDataset([generate(tiny, "train", i) for i in range(32)])
    dev = ByteDataset([generate(tiny, "dev", i) for i in range(8)])
    steps = 6
    trainer = hsn_trainer(hsn_config(steps, "train.eval_every=3"), tmp_path / "run")
    history = trainer.fit(train, dev, steps)
    assert len(history.steps) == steps and all(np.isfinite(s.loss) for s in history.steps)
    groups = history.evals[-1]["per_group"]
    assert set(groups) == {"kvw/in_range"}
    assert 0.0 <= groups["kvw/in_range"]["accuracy"] <= 1.0
    assert np.isfinite(groups["kvw/in_range"]["loss"])


def test_golden_manifest_matches(tmp_path: Path) -> None:
    """The committed manifest hashes are reproduced here (CI: on both runners)."""
    golden = SplitManifest.load(GOLDEN_PATH)
    assert golden.name == DATASET.name == "synthetic_wordkeys"
    assert golden.spec_hash == DATASET.spec_hash()
    assert golden.unroll == DATASET.unroll and golden.is_clean
    assert [s.key for s in golden.shards] == [f"{TASK.key}/{split}" for split in SPLITS]
    for split in SPLITS:
        assert golden.shard(TASK.name, TASK.variant, split).n_docs == TASK.n_examples(split)
    record = golden.shard(TASK.name, TASK.variant, "test")
    examples = [generate(TASK, "test", index) for index in range(TASK.n_test)]
    assert write_shard(tmp_path, TASK, "test", examples).hash == record.hash
    assert sum(e.length for e in examples) == record.n_tokens
