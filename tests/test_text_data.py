"""The text importer (hsn.data.text): shards the unchanged loader reads, bytes that round-trip."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import BOS, EOS, PAD
from hsn.data.text import SourceSpec, TextSpec, build_text, decode, encode
from hsn.train.loop import StreamLoader

SEP = "<|endoftext|>"


def _story(i: int, words: int = 40) -> str:
    return f"Story {i}.\n\n" + " ".join(f"w{i}x{j}" for j in range(words)) + "\tend é"


def _spec(tmp_path: Path, train: bytes, held: bytes, **changes: object) -> TextSpec:
    (tmp_path / "train.txt").write_bytes(train)
    (tmp_path / "held.txt").write_bytes(held)
    fields: dict[str, object] = {
        "name": "text_test",
        "unroll": 64,
        "task": "stories",
        "context_id": 6,
        "separator": SEP,
        "min_bytes": 8,
        "max_train_tokens": 10**6,
        "n_dev": 3,
        "n_test": 2,
        "n_gram": 16,
        "train": SourceSpec(
            file="train.txt", sha256=hashlib.sha256(train).hexdigest(), drop_last=True
        ),
        "heldout": SourceSpec(
            file="held.txt", sha256=hashlib.sha256(held).hexdigest(), drop_first=True
        ),
    }
    fields.update(changes)
    return TextSpec.model_validate(fields)


def _sources() -> tuple[bytes, bytes]:
    held = "tail of a cut story" + SEP + SEP.join(_story(i) for i in range(100, 106))
    # train: 12 stories, one repeated, one equal to a dev story, one too short, a cut tail
    stories = [_story(i) for i in range(12)]
    stories.insert(3, stories[1])
    stories.insert(5, _story(101))
    stories.insert(7, "tiny")
    train = "\r\n" + SEP.join(stories) + SEP + "a story cut by the byte ra"
    return train.encode(), held.encode()


def test_build_loads_through_the_unchanged_loader(tmp_path: Path) -> None:
    train, held = _sources()
    spec = _spec(tmp_path, train, held)
    out = tmp_path / "out"
    manifest, counts = build_text(spec, tmp_path, out)
    assert counts["train_documents"] == 12
    assert counts["train_equal_to_dev_or_test"] == 1
    assert [p.exact_duplicates for p in manifest.leakage] == [0, 0, 0]
    assert manifest.spec_hash == spec.spec_hash()

    sets = {
        split: ByteDataset.load(
            out / "manifest.json", split, ["stories"], ["out_of_range"], expected_unroll=64
        )
        for split in ("train", "dev", "test")
    }
    assert [len(sets[s]) for s in ("train", "dev", "test")] == [12, 3, 2]
    for i, example in enumerate(sets["train"].examples):
        tokens = example.tokens
        assert tokens[0] == BOS and tokens[-1] == EOS
        assert tokens[1:-1].min() >= 16  # no control symbol inside a document
        assert example.loss_mask[:-1].all() and not example.loss_mask[-1]
        assert (example.targets[:-1] == tokens[1:]).all() and example.targets[-1] == PAD
        assert (example.context == 6).all()
        assert decode(tokens, spec).decode() == _story(i)
    assert decode(sets["dev"].examples[1].tokens, spec).decode() == _story(101)
    assert decode(sets["test"].examples[0].tokens, spec).decode() == _story(103)


def test_build_is_reproducible_and_pins_its_sources(tmp_path: Path) -> None:
    train, held = _sources()
    spec = _spec(tmp_path, train, held)
    first, _ = build_text(spec, tmp_path, tmp_path / "a", leakage=False)
    second, _ = build_text(spec, tmp_path, tmp_path / "b", leakage=False)
    assert first.to_json() == second.to_json()
    (tmp_path / "train.txt").write_bytes(train + b"x")
    with pytest.raises(ValueError, match="sha256"):
        build_text(spec, tmp_path, tmp_path / "c", leakage=False)


def test_max_train_tokens_cuts_at_a_document(tmp_path: Path) -> None:
    train, held = _sources()
    one = len(encode(_story(0).encode(), _spec(tmp_path, train, held))) + 2
    spec = _spec(tmp_path, train, held, max_train_tokens=3 * one + 5)
    manifest, counts = build_text(spec, tmp_path, tmp_path / "out", leakage=False)
    assert counts["train_documents"] == 3
    assert manifest.shard("stories", "out_of_range", "train").n_tokens <= spec.max_train_tokens


def test_encode_maps_control_bytes_and_decode_inverts(tmp_path: Path) -> None:
    train, held = _sources()
    spec = _spec(tmp_path, train, held)
    body = encode(b"  a\r\nb\tc\x07d \xc3\xa9  ", spec)
    assert body == bytes([97, 16, 98, 17, 99, 32, 100, 32, 0xC3, 0xA9])
    assert decode(body, spec) == b"a\nb\tc d \xc3\xa9"


def test_a_long_document_continues_across_windows(tmp_path: Path) -> None:
    train, held = _sources()
    spec = _spec(tmp_path, train, held)
    build_text(spec, tmp_path, tmp_path / "out", leakage=False)
    data = ByteDataset.load(tmp_path / "out" / "manifest.json", "train", expected_unroll=64)
    loader = StreamLoader(
        data, batch=2, unroll=64, seed=0, shuffle=False, align_docs=True, epochs=1
    )
    windows = list(loader)
    resets = sum(int(w["reset"].sum()) for w in windows)
    assert resets == len(data)  # one reset per document, none inside it
    assert any(not bool(w["reset"][:, 0].all()) for w in windows)  # a window that continues
    supervised = sum(int(w["loss_mask"].sum()) for w in windows)
    assert supervised == sum(e.length - 1 for e in data.examples)
    first = np.concatenate([w["tokens"][0].numpy() for w in windows])
    length = data.examples[0].length
    assert (first[:length] == data.examples[0].tokens).all()
