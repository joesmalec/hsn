"""The ``delayed_words`` task family and ``synthetic_wordhold`` (16 Nov 2026,
``configs/data/synthetic_wordhold.yaml``): delayed recall with keys that are words of letters,
beside ``synthetic_wordkeys``' key-value task. These tests check the documents against an
independent reading of the token stream and the reference solver, the anchor and the far
queries, determinism, one vocabulary for both tasks (disjoint across splits, the same letters),
that the dataset's ``kvw`` is ``synthetic_wordkeys``' task document for document, the parameter
checks, that a fact binds under the word address (the key's code at the write and at the read),
that both tasks train and are scored under their own names, and the committed manifest."""

from __future__ import annotations

from pathlib import Path

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
    DelayedWordsParams,
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
DATA = REPO / "configs" / "data"
DATASET = load_dataset_spec(DATA / "synthetic_wordhold.yaml")
GOLDEN_PATH = DATA / "synthetic_wordhold.manifest.json"
TASK = DATASET.task("delayedw", "in_range")
KVW = DATASET.task("kvw", "in_range")
SPACE = 32
LETTERS = range(97, 123)
VALUES = range(144, 192)
FILLER = range(192, 256)


def _with(spec: TaskSpec, top: dict[str, object] | None = None, **params: object) -> TaskSpec:
    dump = {**spec.model_dump(), **(top or {})}
    dump["params"] = {**dump["params"], **params}
    return TaskSpec.model_validate(dump)


def _small(n: int = 200) -> TaskSpec:
    return _with(TASK, {"n_train": n, "n_dev": n, "n_test": n})


def _read(example: Example) -> tuple[list[dict], list[dict]]:
    """The facts and queries of a document, read word by word: every maximal run of letters is
    a key word, and the bytes around it say what it is."""
    t = example.tokens.tolist()
    sep = t.index(SEP)
    facts, queries = [], []
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
            assert before == BOS or before in VALUES or before in FILLER, (i, before)
            assert after in VALUES  # the value byte directly after the word
            facts.append({"word": word, "end": end, "value": after})
        else:
            assert before == QRY and after == SPACE
            queries.append({"word": word, "space": end, "answer": t[end + 1]})
        i = end
    return facts, queries


@pytest.mark.parametrize("split", SPLITS)
def test_documents_agree_with_an_independent_reading(split: str) -> None:
    task = _small()
    lo, hi = TASK.length_range
    for index in range(150):
        example = generate(task, split, index)
        check_example(task, example)
        t = example.tokens
        assert t[0] == BOS and t[-1] == EOS and lo <= example.length <= hi <= TASK.unroll
        facts, queries = _read(example)
        assert np.isin(t, [BOS, EOS, QRY, SEP, SPACE, *LETTERS, *VALUES, *FILLER]).all()
        # the facts are one block: w v w v ..., filler before and after it
        ends = [f["end"] for f in facts]
        for a, b in zip(facts, facts[1:], strict=False):
            assert b["end"] - len(b["word"]) == a["end"] + 1
        assert 1 <= len(facts) <= 6 and len({f["word"] for f in facts}) == len(facts)
        assert 1 <= len(queries) <= 4 and queries[0]["word"] == facts[0]["word"]
        assert len({q["word"] for q in queries}) == len(queries)
        state = {f["word"]: (f["value"], f["end"]) for f in facts}
        assert all(3 <= len(w) <= 8 for w in state)
        positions = np.flatnonzero(example.loss_mask).tolist()
        assert positions == [q["space"] for q in queries]
        targets, deps = [], []
        for q in queries:
            value, end = state[q["word"]]
            targets.append(value)
            deps.append(q["space"] - end)
            assert q["answer"] == value
        assert example.targets[positions].tolist() == targets
        assert list(example.meta["deps"]) == deps
        assert max(ends) < t.tolist().index(SEP)
        solved = reference_targets(task, t)
        assert solved[0].tolist() == positions and solved[1].tolist() == targets
        assert solved[2].tolist() == deps


def test_anchor_lengths_and_far_queries() -> None:
    """The first query holds the drawn anchor distance, the document's largest; every document
    fits one window; most queries are far (>= 63), and documents have several facts."""
    task = _small(400)
    dep_lo, dep_hi = TASK.dep_len_range
    far = queries = facts = 0
    for index in range(400):
        example = generate(task, "train", index)
        deps = list(example.meta["deps"])
        assert dep_lo <= deps[0] == example.meta["dep_len"] == max(deps) <= dep_hi
        assert example.meta["in_range"] and example.length <= TASK.unroll
        far += sum(d >= 63 for d in deps)
        queries += len(deps)
        facts += len(_read(example)[0])
    assert far / 400 > 1.2 and far / queries > 0.65
    assert 2.5 < facts / 400 < 4.5 and queries / 400 > 1.5


def test_deterministic_and_split_dependent() -> None:
    a, b = generate(TASK, "train", 7), generate(TASK, "train", 7)
    assert np.array_equal(a.tokens, b.tokens) and a.meta == b.meta
    assert not np.array_equal(a.tokens, generate(TASK, "train", 8).tokens)
    generate(TASK, "dev", 3)
    assert np.array_equal(generate(_small(), "train", 7).tokens, a.tokens)
    fresh = DelayedWordsParams.model_validate(TASK.params)
    assert fresh.vocabulary == TASK.typed_params.vocabulary  # type: ignore[attr-defined]


def test_one_vocabulary_for_both_tasks_disjoint_across_splits() -> None:
    """Both tasks draw from the same words (the same fields give the same vocabulary); the
    splits' words are disjoint and made of the same letters; a document of a split uses its
    split's words only, so no dev or test word is in either task's training documents."""
    params = TASK.typed_params
    assert isinstance(params, DelayedWordsParams)
    kvw = KVW.typed_params
    assert isinstance(kvw, KVWordsParams)
    assert params.vocabulary == kvw.vocabulary
    words = {split: {w for g in params.vocabulary[split] for w in g} for split in SPLITS}
    assert not words["train"] & words["dev"] and not words["train"] & words["test"]
    assert not words["dev"] & words["test"]
    for split in SPLITS:
        assert {c for w in words[split] for c in w} == set(LETTERS)
        for index in range(150):
            facts, queries = _read(generate(TASK, split, index))
            assert {f["word"] for f in facts} | {q["word"] for q in queries} <= words[split]
    # another vocab_seed is another vocabulary
    other = DelayedWordsParams.model_validate({**TASK.params, "vocab_seed": 1})
    assert other.vocabulary["train"] != params.vocabulary["train"]


def test_keys_share_beginnings_and_endings() -> None:
    def shares(task: TaskSpec) -> tuple[float, float]:
        ending = beginning = total = 0
        for index in range(400):
            facts, queries = _read(generate(task, "dev", index))
            keys = {f["word"] for f in facts}
            for q in queries:
                w = q["word"]
                ending += any(k != w and k[-2:] == w[-2:] for k in keys)
                beginning += any(k != w and k[:2] == w[:2] for k in keys)
                total += 1
        return ending / total, beginning / total

    ending, beginning = shares(_small(400))
    assert 0.15 < ending < 0.45 and 0.15 < beginning < 0.45
    ending, beginning = shares(_with(_small(400), p_share=0.0))
    assert ending < 0.05 and beginning < 0.05


def test_kvw_is_synthetic_wordkeys_task() -> None:
    """The dataset's kvw is synthetic_wordkeys' spec, so its shards are the same documents."""
    wordkeys = load_dataset_spec(DATA / "synthetic_wordkeys.yaml")
    (original,) = wordkeys.tasks
    assert KVW.model_dump() == original.model_dump()
    golden = SplitManifest.load(GOLDEN_PATH)
    before = SplitManifest.load(DATA / "synthetic_wordkeys.manifest.json")
    for split in SPLITS:
        assert golden.shard("kvw", "in_range", split) == before.shard("kvw", "in_range", split)
    seeds = {task.seed for task in DATASET.tasks}
    assert len(seeds) == 2 and len({task.context_id for task in DATASET.tasks}) == 2
    assert TASK.context_id == 8


@pytest.mark.parametrize(
    ("top", "params", "message"),
    [
        ({}, {"values": [60, 90]}, "word byte"),  # values that are letters would join a word
        ({}, {"filler": [192, 256], "space": 200}, "both use byte"),
        ({}, {"filler": [40, 60]}, "word byte"),  # digits are word bytes
        ({}, {"letters": [91, 97]}, "ASCII letters"),
        ({}, {"word_len": [1, 8]}, "word_len"),
        ({}, {"groups": {"train": 500, "dev": 1, "test": 125}}, "max_facts"),
        ({}, {"groups": {"train": 500, "dev": 125}}, "every split"),
        ({}, {"keys": {"train": [64, 112]}}, "Extra inputs"),
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
    facts, queries = _read(example)
    space = queries[0]["space"]
    first = facts[0]
    broken = [list(t) for _ in range(5)]
    broken[0][space] = 150  # a query word without its SPACE
    broken[1][first["end"]] = 200  # a fact's word followed by filler, not a value
    broken[2] = t[:-1] + [QRY, EOS]  # a truncated query
    after = facts[-1]["end"] + 1  # the first fact stated again after the last one
    broken[3] = [*t[:after], *first["word"], first["value"], *t[after:]]
    stranger = [122] * len(queries[0]["word"])  # a query that asks no fact's word
    assert tuple(stranger) not in {f["word"] for f in facts}
    broken[4][space - len(stranger) : space] = stranger
    for tokens in broken:
        with pytest.raises(ValueError):
            reference_targets(TASK, np.asarray(tokens, dtype=np.uint8))


@pytest.mark.parametrize("kind", ["exact", "bag"])
def test_a_fact_binds_under_the_word_address(kind: str) -> None:
    """With context 8 read as text, the events of a document are BOS, every fact's value and
    every query's SPACE; the event of a query has its key's word code in a_cur, as the event of
    the fact it asks does, a_prev at a fact is the code of the fact before (BOS for the first);
    another key of the document has another code. Not read as text, every byte is an event."""
    g = torch.Generator().manual_seed(0)
    byte_codes = torch.randn(256, 64, generator=g)
    tables = word_tables(kind, 256, 64, g)
    for index in range(40):
        example = generate(TASK, "dev", index)
        tokens = torch.from_numpy(example.tokens.astype(np.int64))[None]
        context = torch.from_numpy(example.context.astype(np.int64))[None]
        reset = torch.zeros_like(tokens, dtype=torch.bool)
        reset[:, 0] = True
        text = torch.isin(context, torch.tensor([0, 6, 7, 8]))
        assert bool(text.all())
        out = word_address(kind, tables, byte_codes, tokens, text, reset, None)
        facts, queries = _read(example)
        events = torch.nonzero(out.events[0]).flatten().tolist()
        assert events == sorted([0, *(f["end"] for f in facts), *(q["space"] for q in queries)])
        code = {f["word"]: out.current[0, f["end"]] for f in facts}
        assert torch.equal(out.previous[0, facts[0]["end"]], byte_codes[BOS])
        for a, b in zip(facts, facts[1:], strict=False):
            assert torch.equal(out.previous[0, b["end"]], code[a["word"]])
        for q in queries:
            assert torch.equal(out.current[0, q["space"]], code[q["word"]])
            for word, other in code.items():
                if word != q["word"]:
                    assert not torch.allclose(out.current[0, q["space"]], other)
        plain = word_address(kind, tables, byte_codes, tokens, ~text, reset, None)
        assert bool(plain.events.all())


@pytest.mark.usefixtures("restore_determinism")
def test_both_tasks_train_and_are_scored_under_their_own_names(tmp_path: Path) -> None:
    """Documents of both word-key families go through the dataset, the window trainer and the
    evaluation: finite losses, and a dev score for ``kvw/in_range`` and ``delayedw/in_range``."""
    unroll = 32
    top = {
        "n_train": 32,
        "n_dev": 8,
        "n_test": 8,
        "length_range": [20, unroll],
        "unroll": unroll,
    }
    delayed = _with(TASK, {**top, "dep_len_range": [7, 16]}, word_len=[3, 4], max_facts=3)
    kv = _with(KVW, {**top, "dep_len_range": [7, 16]}, word_len=[3, 4], n_keys=3)
    train = ByteDataset([generate(t, "train", i) for t in (delayed, kv) for i in range(32)])
    dev = ByteDataset([generate(t, "dev", i) for t in (delayed, kv) for i in range(8)])
    steps = 6
    trainer = hsn_trainer(hsn_config(steps, "train.eval_every=3"), tmp_path / "run")
    history = trainer.fit(train, dev, steps)
    assert len(history.steps) == steps and all(np.isfinite(s.loss) for s in history.steps)
    groups = history.evals[-1]["per_group"]
    assert set(groups) == {"kvw/in_range", "delayedw/in_range"}
    for name in groups:
        assert 0.0 <= groups[name]["accuracy"] <= 1.0 and np.isfinite(groups[name]["loss"])


def test_golden_manifest_matches(tmp_path: Path) -> None:
    """The committed manifest hashes are reproduced here (CI: on both runners)."""
    golden = SplitManifest.load(GOLDEN_PATH)
    assert golden.name == DATASET.name == "synthetic_wordhold"
    assert golden.spec_hash == DATASET.spec_hash()
    assert golden.unroll == DATASET.unroll and golden.is_clean
    assert [s.key for s in golden.shards] == [
        f"{task.key}/{split}" for task in DATASET.tasks for split in SPLITS
    ]
    for task in DATASET.tasks:
        for split in SPLITS:
            assert golden.shard(task.name, task.variant, split).n_docs == task.n_examples(split)
    record = golden.shard(TASK.name, TASK.variant, "test")
    examples = [generate(TASK, "test", index) for index in range(TASK.n_test)]
    assert write_shard(tmp_path, TASK, "test", examples).hash == record.hash
    assert sum(e.length for e in examples) == record.n_tokens
