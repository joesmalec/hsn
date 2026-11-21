"""``storyfacts`` (:mod:`hsn.data.storyfacts`, ``configs/data/storyfacts_v1.yaml``): stories with
stated facts, the corpus of the recall router.

The spec validates (the key word rule, the disjoint name pools); a build is a pure function of
the spec and the sources; plain documents are the source's stories unchanged and the other
story documents are those stories with whole sentences inserted; every answer is marked at the
positions that predict the asked value, with the distance back to the stated value, inside its
band; nothing but the text separates the kinds; the shards train a model and are scored by
distance; a fact binds under HSN-T's word address.
"""

from __future__ import annotations

import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from hsn.config import load_config
from hsn.data.dataset import ByteDataset
from hsn.data.splits import SplitManifest, read_shard
from hsn.data.storyfacts import (
    FAMILY,
    STORY_TASKS,
    TASKS,
    StoryFactsSpec,
    _assign,
    _characters,
    _stories,
    answers_of,
    build_storyfacts,
    describe,
    kv_document,
    load_storyfacts_spec,
    sentence_boundaries,
    story_document,
)
from hsn.data.storyfacts import main as storyfacts_main
from hsn.data.synthetic import BOS, EOS, SPLITS
from hsn.experiments.protocols import ArmRun, run_protocol
from hsn.experiments.rules import parse_rules
from hsn.experiments.spec import protocol_params
from hsn.model.word_address import word_address, word_tables
from p1_11_support import GRU_2M, TINY_GRU_OVERRIDES, restore_determinism, train_overrides  # noqa: F401
from storyfacts_support import (
    BANDS,
    GOLDEN,
    REPO,
    SPEC_PATH,
    UNROLL,
    build_tiny,
)

SPEC = load_storyfacts_spec(SPEC_PATH)
NEWLINE = bytes([16])


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[StoryFactsSpec, Path, SplitManifest]:
    return build_tiny(tmp_path_factory.mktemp("storyfacts"))


def _documents(manifest: Path, split: str, task: str) -> list[dict[str, Any]]:
    """The documents of one shard: ``tokens``, ``index`` and ``answers`` (read back)."""
    arrays, scalars = read_shard(manifest.parent / f"{task}.out_of_range.{split}.npz")
    assert (scalars["family"], scalars["context_id"]) == (FAMILY[task], 6)
    offsets, out, cursor = arrays["offsets"], [], 0
    for i in range(offsets.shape[0] - 1):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        mask = arrays["loss_mask"][lo:hi]
        deps = arrays["deps"][cursor : cursor + int(mask.sum())]
        cursor += int(mask.sum())
        assert mask[:-1].all() and not mask[-1]  # every byte is a target but the last
        out.append(
            {
                "tokens": arrays["tokens"][lo:hi],
                "index": int(arrays["index"][i]),
                "answers": answers_of(mask, deps),
                "deps": deps,
            }
        )
    return out


# --------------------------------------------------------------------------------------------------
# The spec
# --------------------------------------------------------------------------------------------------


def test_the_committed_spec_and_its_golden_manifest() -> None:
    names = SPEC.vocabulary()
    assert {split: len(names[split]) for split in SPLITS} == SPEC.names.n
    pools = [set(names[split]) for split in SPLITS]
    assert not (pools[0] & pools[1] or pools[0] & pools[2] or pools[1] & pools[2])
    taken = {o.lower() for owners in SPEC.owners.values() for o in owners}
    taken |= {word for words in SPEC.words.values() for word in words}
    assert not {name.lower() for pool in pools for name in pool} & taken
    assert all(name.isalpha() and name == name.capitalize() for pool in pools for name in pool)
    keys = [relation.key_word for relation in SPEC.relations]
    assert len(set(keys)) == len(keys) == 11 and "called" in keys and "{title}" in keys
    assert (SPEC.context_id, SPEC.unroll, SPEC.variant) == (6, 256, "out_of_range")
    golden = SplitManifest.load(GOLDEN)
    assert (golden.name, golden.unroll, golden.leakage) == ("storyfacts_v1", 256, [])
    assert golden.spec_hash == SPEC.spec_hash()
    assert [(s.task, s.split) for s in golden.shards] == [(t, s) for t in TASKS for s in SPLITS]
    assert all((s.family, s.context_id) == (FAMILY[s.task], 6) for s in golden.shards)
    docs = {split: sum(s.n_docs for s in golden.shards if s.split == split) for split in SPLITS}
    assert docs == {split: SPEC.stories.n[split] + SPEC.kv.n[split] for split in SPLITS}


def _raw() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    return raw


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"statements": ["{owner}'s {pet} is named {value}."]}, "key word rule"),
        ({"questions": ["What is it called? {value} is {value}."]}, "exactly once"),
        ({"questions": ["What is the {toy} called? {value}."]}, "no slot"),
        ({"statements": ["{owner}'s {pet} is called{value}."]}, "one space"),
        ({"slots": {"pet": "creatures"}}, "no word list"),
    ],
)
def test_spec_validation_rejects_a_bad_relation(change: dict[str, Any], message: str) -> None:
    raw = _raw()
    raw["relations"][0].update(change)
    with pytest.raises(ValueError, match=message):
        StoryFactsSpec.model_validate(raw)


def test_spec_validation_rejects_shared_keys_and_owners_and_a_small_pool() -> None:
    raw = _raw()
    raw["relations"][1]["statements"] = ["{owner} called {value} last {season}."]
    raw["relations"][1]["questions"] = [
        "Who did {owner} call last {season}? {owner} called {value}."
    ]
    with pytest.raises(ValueError, match="share a key word"):
        StoryFactsSpec.model_validate(raw)
    raw = _raw()
    raw["owners"]["dev"].append("lily")
    with pytest.raises(ValueError, match="own distinct names"):
        StoryFactsSpec.model_validate(raw)
    raw = _raw()
    raw["names"].update(onsets=["b"], vowels=["a"], finals=[""])
    with pytest.raises(ValueError, match="candidates"):
        StoryFactsSpec.model_validate(raw)


# --------------------------------------------------------------------------------------------------
# Sentences
# --------------------------------------------------------------------------------------------------


def test_sentence_boundaries_follow_the_rule() -> None:
    first = b'Tom ran. "Stop. Wait," said Mia. Mr. Bear came! Then he left'
    second = b'"We go."'
    text = first + NEWLINE + second + NEWLINE + "café one. Two.".encode()
    found = sentence_boundaries(text)
    # behind "ran.", "Mia." and "came!"; not inside the quotation, not behind "Mr.", not at the
    # end of a paragraph that closes no sentence; the end of the quoted paragraph; never inside
    # the paragraph that holds a byte above 127; the start and the end always
    inside = [first.index(piece) for piece in (b' "Stop', b" Mr.", b" Then")]
    assert found == [0, *inside, len(first) + 1 + len(second), len(text)]
    assert sentence_boundaries(b"") == [0]
    assert _characters(b"One day, Tom met Mia. Then Tom and Mia saw Ben, and Tom ran.") == [
        "Tom",
        "Mia",
    ]


# --------------------------------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------------------------------


def test_a_build_is_deterministic(
    tiny: tuple[StoryFactsSpec, Path, SplitManifest], tmp_path: Path
) -> None:
    spec, path, manifest = tiny
    again, counts = build_storyfacts(spec, path.parent.parent / "raw", tmp_path / "again")
    assert again.to_json() == manifest.to_json() == path.read_text(encoding="utf-8")
    for record in manifest.shards:
        first = hashlib.sha256((path.parent / record.file).read_bytes()).hexdigest()
        second = hashlib.sha256((tmp_path / "again" / record.file).read_bytes()).hexdigest()
        assert first == second
    assert counts["train_stories_used"] <= counts["train_stories"]
    # the CLI: the golden comparison passes on the same spec and fails on another seed
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(spec.model_dump(mode="json")), encoding="utf-8")
    build = ["build", "--raw", str(path.parent.parent / "raw"), "--out", str(tmp_path / "cli")]
    assert storyfacts_main([*build, "--spec", str(spec_file), "--golden", str(path)]) == 0
    other = spec.model_copy(update={"seed": spec.seed + 1})
    spec_file.write_text(yaml.safe_dump(other.model_dump(mode="json")), encoding="utf-8")
    assert storyfacts_main([*build, "--spec", str(spec_file), "--golden", str(path)]) == 1
    assert other.spec_hash() != spec.spec_hash()
    # a source with other bytes is refused
    (tmp_path / "raw").mkdir()
    for name in ("train.txt", "held.txt"):
        (tmp_path / "raw" / name).write_bytes(b"other")
    with pytest.raises(ValueError, match="sha256"):
        build_storyfacts(spec, tmp_path / "raw", tmp_path / "refused")


def test_documents_are_stories_with_whole_sentences_inserted(
    tiny: tuple[StoryFactsSpec, Path, SplitManifest],
) -> None:
    """Every story document is its stories, joined by a blank line: unchanged when plain, with
    bytes inserted and nothing else changed otherwise. Which stories a document gets does not
    depend on its kind, and a document is regenerated from its stories and its index."""
    spec, path, _ = tiny
    held = _stories(path.parent.parent / "raw", spec, spec.heldout)
    _, first = _assign(spec, "test", held, 0)
    taken, _ = _assign(spec, "dev", held, first)
    names = spec.vocabulary()["dev"]
    gap = NEWLINE * spec.stories.gap
    changed = 0
    for task in STORY_TASKS:
        for doc in _documents(path, "dev", task):
            stories = taken[doc["index"]]
            source, body = gap.join(stories), bytes(doc["tokens"][1:-1])
            assert doc["tokens"][0] == BOS and doc["tokens"][-1] == EOS
            assert all(b >= 32 or b == 16 for b in body)
            again = story_document(spec, "dev", doc["index"], stories, names)
            assert again.task == task and np.array_equal(again.tokens, doc["tokens"])
            assert np.array_equal(again.deps, doc["deps"])
            if task == "plain":
                assert body == source and again.n_facts == 0
                continue
            matcher = SequenceMatcher(None, source, body, autojunk=False)
            assert {op for op, *_ in matcher.get_opcodes()} == {"equal", "insert"}
            assert 1 <= again.n_facts <= len(spec.stories.fact_weights)
            changed += 1
    assert changed >= 10
    # the kind is drawn from the index alone: other stories, the same kind
    for index in range(12):
        kinds = {story_document(spec, "dev", index, [s], names).task for s in held[:3]}
        assert kinds == {story_document(spec, "dev", index, taken[index], names).task}


def test_answers_are_marked_at_the_asked_value_inside_their_band(
    tiny: tuple[StoryFactsSpec, Path, SplitManifest],
) -> None:
    spec, path, _ = tiny
    names = {split: set(spec.vocabulary()[split]) for split in SPLITS}
    lists = {word for relation in spec.relations for word in spec.words.get(relation.value, [])}
    bands = {name: 0 for name in BANDS}
    for split in ("train", "dev"):
        foreign = set().union(*(names[other] for other in SPLITS if other != split))
        assert all(
            not doc["answers"]
            for task in ("plain", "stated")
            for doc in _documents(path, split, task)
        )
        for task in ("facts", "kv"):
            for doc in _documents(path, split, task):
                tokens, answers = doc["tokens"], doc["answers"]
                assert answers, "a facts document and a fact sheet ask at least once"
                assert (doc["deps"] > 1).sum() == sum(n for _, n, _ in answers)
                text = bytes(tokens)
                assert not any(name.encode() in text for name in foreign)
                for position, length, distance in answers:
                    value = bytes(tokens[position + 1 : position + 1 + length])
                    stated = position - distance  # the fact position: answer position - distance
                    assert bytes(tokens[stated : stated + length]) == value
                    word = value.decode()
                    assert word.isdigit() or word in names[split] or word in lists
                    # the value is a whole word in the statement and in the answer
                    for at in (stated, position + 1):
                        assert not chr(tokens[at - 1]).isalnum()
                        assert not chr(tokens[at + length]).isalnum()
                    if task == "facts":
                        name = next(n for n, (lo, hi) in BANDS.items() if lo <= distance <= hi)
                        bands[name] += 1
                    else:
                        assert distance >= 2
    assert all(count > 5 for count in bands.values()), bands
    # the generator's own record agrees with what is read back from the shard
    sheet = kv_document(spec, "dev", 3)
    read = _documents(path, "dev", "kv")[3]
    assert np.array_equal(sheet.tokens, read["tokens"]) and np.array_equal(sheet.deps, read["deps"])
    assert [(a.position, a.length, a.distance) for a in sheet.answers] == sorted(read["answers"])
    assert sheet.n_facts >= len(sheet.answers) >= 1


def test_describe_counts_what_the_shards_hold(
    tiny: tuple[StoryFactsSpec, Path, SplitManifest], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, path, manifest = tiny
    table = describe(path.parent, edges=(8, 32, 128))
    for record in manifest.shards:
        row = table["splits"][record.split][record.task]
        docs = _documents(path, record.split, record.task)
        assert (
            (row["documents"], row["bytes"])
            == (record.n_docs, record.n_tokens)
            == (len(docs), sum(len(d["tokens"]) for d in docs))
        )
        assert row["answers"] == sum(len(d["answers"]) for d in docs)
        assert row["documents_with_answers"] == sum(bool(d["answers"]) for d in docs)
        assert sum(row["answers_by_distance"].values()) == row["answers"]
    kinds = {task: table["splits"]["train"][task]["documents"] for task in TASKS}
    assert sum(kinds[task] for task in STORY_TASKS) == spec.stories.n["train"]
    assert kinds["kv"] == spec.kv.n["train"] and min(kinds.values()) > 0
    assert storyfacts_main(["describe", "--out", str(path.parent)]) == 0
    assert "dev/facts: docs=" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------------
# Training and scoring
# --------------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("restore_determinism")
def test_the_shards_train_a_model_and_are_scored_by_distance(
    tiny: tuple[StoryFactsSpec, Path, SplitManifest], tmp_path: Path
) -> None:
    """All four tasks through the loader, the trainer and ``eval.by_distance``: the plain
    stories get the text tables, the others their answers by distance (the text bytes are the
    bucket under 64)."""
    spec, path, _ = tiny
    dev = ByteDataset.load(path, "dev", expected_unroll=UNROLL)  # every shard hash is checked
    assert len(dev) == spec.stories.n["dev"] + spec.kv.n["dev"]
    assert {e.meta["task"] for e in dev.examples} == set(TASKS)
    assert all(not e.meta["in_range"] or e.meta["dep_len"] <= UNROLL for e in dev.examples)
    cfg = load_config(
        GRU_2M,
        [
            *TINY_GRU_OVERRIDES,
            *train_overrides(steps=2, eval_every=2),
            f"data.manifest={path}",
            "data.dataset=storyfacts_tiny",
            f"data.tasks=[{', '.join(TASKS)}]",
            "data.variants=[out_of_range]",
            "eval.by_distance.enabled=true",
            "train.answer_weight=4.0",
        ],
    )
    arm = ArmRun(
        cfg=cfg,
        family="gru",
        variant="none",
        variant_params={},
        params=protocol_params({"name": "train_eval", "eval_batch": 16, "diagnostics": False}),
        run_dir=tmp_path / "run",
        rules=parse_rules(None),
        constants={},
    )
    record = run_protocol(arm)
    groups = record["by_distance"]["sets"]["dev"]["groups"]
    assert list(groups) == [f"{task}/out_of_range" for task in TASKS]
    assert groups["plain/out_of_range"]["kind"] == "text"
    facts = groups["facts/out_of_range"]
    assert facts["kind"] == "answers" and facts["family"] == "storyfacts"
    assert {"lt64", "64_255"} <= set(facts["by_distance"])
    answers = sum(
        len(d["answers"]) and sum(n for _, n, _ in d["answers"])
        for d in _documents(path, "dev", "facts")
    )
    far = sum(cell["n"] for key, cell in facts["by_distance"].items() if key != "lt64")
    assert 0 < far <= answers  # the answer bytes 64 or more back (the nearer ones share lt64)
    assert set(groups["stated/out_of_range"]["by_distance"]) == {"lt64"}
    json.dumps(record["by_distance"], allow_nan=False)


@pytest.mark.parametrize("kind", ["exact", "bag"])
def test_a_fact_binds_under_the_word_address(kind: str) -> None:
    """Read as text (context 6), the stated value is closed by an event whose previous key is
    the relation's key word, and the answer is predicted from the event that closes that word
    again: both read the same code, and no other relation of the sheet shares it."""
    g = torch.Generator().manual_seed(0)
    byte_codes = torch.randn(256, 64, generator=g)
    tables = word_tables(kind, 256, 64, g)
    names = SPEC.vocabulary()["dev"]
    checked = 0
    for index in range(30):
        sheet = kv_document(SPEC, "dev", index, names)
        tokens = torch.from_numpy(sheet.tokens.astype(np.int64))[None]
        reset = torch.zeros_like(tokens, dtype=torch.bool)
        reset[:, 0] = True
        out = word_address(kind, tables, byte_codes, tokens, torch.ones_like(reset), reset, None)
        events = out.events[0]
        codes = []
        for answer in sheet.answers:
            # the write: the first event behind the stated value (the byte that closes it)
            write = answer.position - answer.distance + answer.length
            assert bool(events[write]) and not bool(events[write - answer.length : write].any())
            # the read: the last event at or before the first answer position
            read = int(torch.nonzero(events[: answer.position + 1]).flatten()[-1])
            assert torch.equal(out.current[0, read], out.previous[0, write])
            codes.append(out.current[0, read])
            checked += 1
        for i, a in enumerate(codes):
            assert all(not torch.allclose(a, b) for b in codes[i + 1 :])
    assert checked > 60


def test_the_built_copy_is_the_golden_one() -> None:
    """Where the dataset is built (``data/storyfacts_v1``), its manifest is the committed one
    and its dev shards hash to it."""
    built = REPO / "data" / "storyfacts_v1" / "manifest.json"
    if not built.is_file():
        pytest.skip("data/storyfacts_v1 is not built here")
    assert built.read_text(encoding="utf-8") == GOLDEN.read_text(encoding="utf-8")
    dev = ByteDataset.load(built, "dev", ["kv", "stated"], expected_unroll=256)
    assert len(dev) > 0
