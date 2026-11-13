"""Real text in the P0.4 shard format (12 Nov 2026): the importer behind ``tinystories_v1``.

The synthetic generator (:mod:`hsn.data.synthetic`) writes documents whose answers are known and
marks only those positions. Real text has no answers: every byte is a target. This module turns
raw text files into the same shards and manifest (:mod:`hsn.data.splits`), so the loader, the
trainer and the evaluation run unchanged; it adds no training code.

* **Documents** are the pieces of a source file between ``separator`` (TinyStories: one story),
  with carriage returns removed and surrounding white space stripped; a piece shorter than
  ``min_bytes`` is dropped. A document is ``BOS, bytes..., EOS``.
* **Bytes.** Byte values 0..31 are the program's control and marker symbols, so text may not use
  them: a newline becomes the marker ``newline`` (16), a tab the marker ``tab`` (17) and any other
  byte below 32 a space. Bytes 32..255 are kept, so UTF-8 passes through. :func:`decode` inverts
  the mapping.
* **Supervision.** ``loss_mask`` is True at every position but the last: the target of position
  ``t`` is byte ``t + 1``, the end symbol included; the last target is PAD. Dev accuracy is
  therefore next-byte accuracy and the dev loss divided by ``ln 2`` is bits per byte. ``deps`` is
  1 everywhere (the nearest byte): text has no annotated dependency.
* **Variant.** Documents are longer than a window, so the shards are labelled ``out_of_range``:
  the recurrent state is carried across a document's windows, the gradient is not.
* **Splits.** Train documents come from ``train`` in file order, up to ``max_train_tokens``; dev
  and test are the first ``n_dev`` and the next ``n_test`` distinct documents of ``heldout``.
  Duplicates inside a source are dropped (first kept), and so is every train document equal to a
  dev or test document, so the exact-duplicate count of the manifest is zero by construction. The
  n-gram overlap (:mod:`hsn.data.leakage`) is measured and recorded, not removed: natural text
  shares phrases ("Once upon a time, there was a little girl named Lily"), so a text manifest is
  never ``is_clean`` in the synthetic sense.

The spec names every source file by its SHA-256; the build refuses other bytes. The manifest has
no timestamps, so the same sources give byte-identical shards and manifest (``--golden``).

    python -I -m hsn.data.text build --spec configs/data/tinystories_v1.yaml \\
        --raw <folder with the source files> --out data/tinystories_v1 \\
        --golden configs/data/tinystories_v1.manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

import hsn
from hsn.config import Split
from hsn.data.leakage import NGramIndex
from hsn.data.splits import (
    MANIFEST_NAME,
    SHARD_FORMAT,
    LeakagePair,
    ShardRecord,
    SplitManifest,
    shard_file_name,
    shard_hash,
)
from hsn.data.synthetic import BOS, EOS, N_MARKERS, N_RESERVED, SPLITS, Variant

__all__ = ["SourceSpec", "TextSpec", "build_text", "decode", "encode", "load_text_spec"]

_SPACE = 32


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceSpec(_Strict):
    """One raw file, named relative to ``--raw`` and pinned by its SHA-256."""

    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    drop_first: bool = False  # the file starts inside a document
    drop_last: bool = False  # the file ends inside a document (a byte-range download)


class TextSpec(_Strict):
    """A text dataset: one task, three splits (see the module docstring)."""

    name: str
    unroll: int = Field(ge=1)
    task: str
    variant: Variant = "out_of_range"
    family: str = "text"
    context_id: int = Field(ge=0, le=15)
    separator: str = Field(min_length=1)
    min_bytes: int = Field(default=1, ge=1)
    newline: int = Field(default=16, ge=N_RESERVED, lt=N_MARKERS)
    tab: int = Field(default=17, ge=N_RESERVED, lt=N_MARKERS)
    max_train_tokens: int = Field(ge=1)
    n_dev: int = Field(ge=1)
    n_test: int = Field(ge=1)
    n_gram: int = Field(default=32, ge=1)
    train: SourceSpec
    heldout: SourceSpec

    @model_validator(mode="after")
    def _check(self) -> TextSpec:
        if self.newline == self.tab:
            raise ValueError("newline and tab need different marker symbols")
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def spec_hash(self) -> str:
        return hashlib.blake2b(self.canonical_json().encode(), digest_size=32).hexdigest()


def load_text_spec(path: str | Path) -> TextSpec:
    """Load and validate a text dataset spec YAML."""
    spec_path = Path(path)
    try:
        loaded = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{spec_path}: invalid YAML: {exc}") from exc
    try:
        return TextSpec.model_validate(loaded)
    except ValidationError as exc:
        raise ValueError(f"{spec_path}: {exc}") from exc


# --------------------------------------------------------------------------------------------------
# Bytes
# --------------------------------------------------------------------------------------------------


def _table(spec: TextSpec) -> bytes:
    table = bytearray(range(256))
    for value in range(N_MARKERS):
        table[value] = _SPACE
    table[ord("\n")] = spec.newline
    table[ord("\t")] = spec.tab
    return bytes(table)


def encode(text: bytes, spec: TextSpec) -> bytes:
    """The document body of ``text``: carriage returns removed, stripped, bytes below 32 mapped."""
    return text.replace(b"\r", b"").strip().translate(_table(spec))


def decode(tokens: npt.NDArray[np.uint8] | bytes, spec: TextSpec) -> bytes:
    """Text of a document's tokens (BOS, EOS and PAD dropped; the markers back to newline, tab)."""
    table = bytearray(range(256))
    table[spec.newline] = ord("\n")
    table[spec.tab] = ord("\t")
    body = bytes(tokens).translate(bytes(table), delete=bytes(range(N_RESERVED)))
    return body


def _documents(raw: bytes, spec: TextSpec, source: SourceSpec) -> tuple[list[bytes], int]:
    """Distinct document bodies of a source in file order, and how many pieces were dropped."""
    pieces = raw.split(spec.separator.encode())
    if source.drop_first:
        pieces = pieces[1:]
    if source.drop_last:
        pieces = pieces[:-1]
    table = _table(spec)
    seen: set[bytes] = set()
    docs: list[bytes] = []
    for piece in pieces:
        body = piece.replace(b"\r", b"").strip()
        if len(body) < spec.min_bytes:
            continue
        body = body.translate(table)
        digest = hashlib.blake2b(body, digest_size=16).digest()
        if digest in seen:
            continue
        seen.add(digest)
        docs.append(body)
    return docs, len(pieces) - len(docs)


# --------------------------------------------------------------------------------------------------
# Shards
# --------------------------------------------------------------------------------------------------


def _arrays(docs: Sequence[bytes]) -> dict[str, npt.NDArray[Any]]:
    """The flat arrays of a shard for ``BOS, body, EOS`` documents (the layout of ``splits``)."""
    n = len(docs)
    lengths = np.fromiter((len(doc) + 2 for doc in docs), dtype=np.int64, count=n)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    total = int(offsets[-1])
    tokens = np.empty(total, dtype=np.uint8)
    body = np.ones(total, dtype=np.bool_)
    body[offsets[:-1]] = False
    body[offsets[1:] - 1] = False
    tokens[body] = np.frombuffer(b"".join(docs), dtype=np.uint8)
    tokens[offsets[:-1]] = BOS
    tokens[offsets[1:] - 1] = EOS
    loss_mask = np.ones(total, dtype=np.bool_)
    loss_mask[offsets[1:] - 1] = False  # the last position's target is PAD
    return {
        "tokens": tokens,
        "loss_mask": loss_mask,
        "deps": np.ones(total - n, dtype=np.int32),
        "offsets": offsets,
        "index": np.arange(n, dtype=np.int64),
    }


def _write(out_dir: Path, spec: TextSpec, split: Split, docs: Sequence[bytes]) -> ShardRecord:
    arrays = _arrays(docs)
    scalars: dict[str, Any] = {
        "format": SHARD_FORMAT,
        "task": spec.task,
        "variant": spec.variant,
        "split": split,
        "family": spec.family,
        "context_id": spec.context_id,
        "unroll": spec.unroll,
    }
    file = shard_file_name(spec.task, spec.variant, split)
    payload: dict[str, Any] = {
        **arrays,
        **{key: np.asarray(value) for key, value in scalars.items()},
    }
    np.savez(out_dir / file, **payload)
    return ShardRecord(
        task=spec.task,
        variant=spec.variant,
        split=split,
        family=spec.family,
        context_id=spec.context_id,
        file=file,
        n_docs=len(docs),
        n_tokens=int(arrays["tokens"].shape[0]),
        n_loss=int(arrays["deps"].shape[0]),
        hash=shard_hash(arrays, scalars),
    )


@dataclass(frozen=True)
class _Doc:
    """What the leakage index reads of a document (no per-position metadata is built)."""

    tokens: npt.NDArray[np.uint8]
    meta: dict[str, str]

    @property
    def length(self) -> int:
        return int(self.tokens.shape[0])


def _leak_docs(spec: TextSpec, split: Split, docs: Sequence[bytes]) -> list[_Doc]:
    return [
        _Doc(
            np.frombuffer(bytes([BOS]) + doc + bytes([EOS]), dtype=np.uint8),
            {"doc_id": f"{spec.task}:{spec.variant}:{split}:{i}"},
        )
        for i, doc in enumerate(docs)
    ]


def _pair(train: str, test: str, index: NGramIndex, docs: Sequence[_Doc]) -> LeakagePair:
    report = index.check(docs)  # type: ignore[arg-type]  # reads tokens, length and doc_id only
    return LeakagePair(
        train=train,
        test=test,
        n_gram=report.n_gram,
        exact_duplicates=report.exact_duplicates,
        overlapping_ngrams=report.overlapping_ngrams,
        test_docs_with_overlap=report.test_docs_with_overlap,
    )


def _read(raw_dir: Path, source: SourceSpec) -> bytes:
    path = raw_dir / source.file
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != source.sha256:
        raise ValueError(f"{path}: sha256 {actual} != spec sha256 {source.sha256}")
    return data


def build_text(
    spec: TextSpec, raw_dir: Path, out_dir: Path, *, leakage: bool = True
) -> tuple[SplitManifest, dict[str, int]]:
    """Write the three shards and ``manifest.json`` of ``spec`` into ``out_dir``.

    Returns the manifest and the counts of what was kept and dropped (for the build's report).
    """
    held, held_dropped = _documents(_read(raw_dir, spec.heldout), spec, spec.heldout)
    if len(held) < spec.n_dev + spec.n_test:
        raise ValueError(
            f"{spec.heldout.file}: {len(held)} documents, need {spec.n_dev + spec.n_test}"
        )
    dev = held[: spec.n_dev]
    test = held[spec.n_dev : spec.n_dev + spec.n_test]
    held_out = set(dev) | set(test)
    pool, train_dropped = _documents(_read(raw_dir, spec.train), spec, spec.train)
    train: list[bytes] = []
    tokens = 0
    equal_to_held = 0
    for doc in pool:
        if doc in held_out:
            equal_to_held += 1
            continue
        if tokens + len(doc) + 2 > spec.max_train_tokens:
            break
        train.append(doc)
        tokens += len(doc) + 2
    if not train:
        raise ValueError(f"{spec.train.file}: no train document fits max_train_tokens")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[Split, list[bytes]] = {"train": train, "dev": dev, "test": test}
    records = [_write(out_dir, spec, split, by_split[split]) for split in SPLITS]
    pairs: list[LeakagePair] = []
    if leakage:
        key = f"{spec.task}/{spec.variant}"
        held_docs = {split: _leak_docs(spec, split, by_split[split]) for split in ("dev", "test")}
        index = NGramIndex(_leak_docs(spec, "train", train), spec.n_gram)  # type: ignore[arg-type]
        for split in ("dev", "test"):
            pairs.append(_pair(f"{key}/train", f"{key}/{split}", index, held_docs[split]))
        del index
        dev_index = NGramIndex(held_docs["dev"], spec.n_gram)  # type: ignore[arg-type]
        pairs.append(_pair(f"{key}/dev", f"{key}/test", dev_index, held_docs["test"]))
    split_hashes: dict[str, str] = {}
    for split in SPLITS:
        digest = hashlib.blake2b(digest_size=32)
        for record in records:
            if record.split == split:
                digest.update(bytes.fromhex(record.hash))
        split_hashes[split] = digest.hexdigest()
    manifest = SplitManifest(
        name=spec.name,
        spec_hash=spec.spec_hash(),
        unroll=spec.unroll,
        hsn_version=hsn.__version__,
        numpy_version=np.__version__,
        shards=records,
        split_hashes=split_hashes,
        leakage=pairs,
    )
    manifest.write(out_dir / MANIFEST_NAME)
    counts = {
        "heldout_documents": len(held),
        "heldout_dropped": held_dropped,
        "train_pool": len(pool),
        "train_dropped": train_dropped,
        "train_equal_to_dev_or_test": equal_to_held,
        "train_documents": len(train),
        "train_tokens": tokens,
    }
    return manifest, counts


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def _cmd_build(args: argparse.Namespace) -> int:
    spec = load_text_spec(args.spec)
    started = time.perf_counter()
    manifest, counts = build_text(spec, Path(args.raw), Path(args.out), leakage=not args.no_leakage)
    print(f"built {spec.name} in {time.perf_counter() - started:.1f} s")
    for name, value in counts.items():
        print(f"  {name}: {value}")
    for shard in manifest.shards:
        print(
            f"  {shard.task}/{shard.variant}/{shard.split}: "
            f"docs={shard.n_docs} tokens={shard.n_tokens} {shard.hash}"
        )
    status = 0
    for pair in manifest.leakage:
        print(
            f"  leakage {pair.train} -> {pair.test}: exact duplicates {pair.exact_duplicates}, "
            f"shared {pair.n_gram}-grams {pair.overlapping_ngrams} "
            f"in {pair.test_docs_with_overlap} documents"
        )
        if pair.exact_duplicates:
            print("ERROR: exact duplicates between splits", file=sys.stderr)
            status = 1
    if args.write_golden:
        golden = Path(args.write_golden)
        golden.write_text(manifest.to_json(), encoding="utf-8")
        print(f"wrote golden manifest {golden}")
    if args.golden:
        if Path(args.golden).read_text(encoding="utf-8") != manifest.to_json():
            print(f"ERROR: manifest differs from the golden file {args.golden}", file=sys.stderr)
            status = 1
        else:
            print(f"manifest matches the golden file {args.golden}")
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hsn.data.text", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write the shards and the manifest of a text dataset")
    build.add_argument("--spec", required=True)
    build.add_argument("--raw", required=True, help="folder holding the spec's source files")
    build.add_argument("--out", required=True)
    build.add_argument("--no-leakage", action="store_true", help="skip the n-gram overlap count")
    build.add_argument("--golden", default=None, help="manifest the build must reproduce")
    build.add_argument("--write-golden", default=None)
    build.set_defaults(func=_cmd_build)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
