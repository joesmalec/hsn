"""Split shards and the split manifest (P0.4).

:func:`build_splits` generates every split of every task of a
:class:`~hsn.data.synthetic.DatasetSpec` with :func:`~hsn.data.synthetic.generate`, writes one
``.npz`` shard per ``(task, variant, split)``
and a ``manifest.json`` next to them, and runs the leakage checks (:mod:`hsn.data.leakage`)
between every train split and every dev and test split of every task, plus dev against test
within a task.

**Shard layout** (``<task>.<variant>.<split>.npz``, uncompressed): flat ``tokens`` (uint8) and
``loss_mask`` (bool) of all documents, ``deps`` (int32, one entry per True ``loss_mask`` position),
``offsets`` (int64, ``n + 1``; document ``i`` spans ``offsets[i]:offsets[i + 1]``), ``index``
(int64, the example index of each document) and the scalars ``task``, ``variant``, ``split``,
``family``, ``context_id``, ``unroll`` and ``format``. ``targets`` (next symbol, PAD last),
``context`` (constant ``context_id``) and ``dep_len`` (largest ``deps`` entry) are derived on load.

**Hashes.** A shard's hash is blake2b-256 over a header naming the shard and the arrays' bytes in a
fixed order with explicit little-endian dtypes, so it depends only on the generated data, never
on how NumPy happens to serialise an ``.npz``. The manifest records every shard hash, one
aggregate hash per split (over the shard hashes in task order), the hash of the canonical spec,
the leakage counts of every pair checked and the ``hsn`` and NumPy versions (NumPy's
``Generator`` streams are stable across patch releases but may change in a feature release).
The manifest contains no timestamps, so an unchanged dataset produces byte-identical text; the
build CLI compares it with the committed golden copy ``configs/data/<name>.manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import hsn
from hsn.config import Split
from hsn.data.leakage import NGramIndex
from hsn.data.synthetic import (
    PAD,
    SPLITS,
    DatasetSpec,
    Example,
    TaskSpec,
    Variant,
    check_example,
    generate,
)

__all__ = [
    "SHARD_FORMAT",
    "LeakagePair",
    "ShardRecord",
    "SplitManifest",
    "build_splits",
    "load_shard",
    "pack_examples",
    "read_shard",
    "shard_file_name",
    "shard_hash",
    "unpack_examples",
    "verify_splits",
    "write_shard",
]

SHARD_FORMAT = 1
MANIFEST_NAME = "manifest.json"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ShardRecord(_Frozen):
    """One ``(task, variant, split)`` shard as recorded in the manifest."""

    task: str
    variant: Variant
    split: Split
    family: str
    context_id: int = Field(ge=0, le=15)
    file: str  # relative to the manifest directory
    n_docs: int = Field(ge=0)
    n_tokens: int = Field(ge=0)
    n_loss: int = Field(ge=0)
    hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def key(self) -> str:
        return f"{self.task}/{self.variant}/{self.split}"


class LeakagePair(_Frozen):
    """Leakage counts of one checked pair (shard keys ``task/variant/split``)."""

    train: str
    test: str
    n_gram: int = Field(ge=1)
    exact_duplicates: int = Field(ge=0)
    overlapping_ngrams: int = Field(ge=0)
    test_docs_with_overlap: int = Field(ge=0)

    @property
    def is_clean(self) -> bool:
        return self.exact_duplicates == 0 and self.overlapping_ngrams == 0


class SplitManifest(_Frozen):
    """``manifest.json``: what was generated, its hashes and the leakage results."""

    name: str
    spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    unroll: int = Field(ge=1)
    format: int = SHARD_FORMAT
    hsn_version: str
    numpy_version: str
    shards: list[ShardRecord]
    split_hashes: dict[str, str]  # split -> blake2b over the shard hashes in task order
    leakage: list[LeakagePair]

    @property
    def is_clean(self) -> bool:
        return all(pair.is_clean for pair in self.leakage)

    def shard(self, task: str, variant: Variant, split: Split) -> ShardRecord:
        for record in self.shards:
            if (record.task, record.variant, record.split) == (task, variant, split):
                return record
        raise KeyError(f"{self.name}: no shard for {task}/{variant}/{split}")

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

    def write(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        manifest_path = Path(path)
        try:
            return cls.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise ValueError(f"{manifest_path}: {exc}") from exc


# --------------------------------------------------------------------------------------------------
# Shards
# --------------------------------------------------------------------------------------------------


def shard_file_name(task: str, variant: str, split: str) -> str:
    return f"{task}.{variant}.{split}.npz"


def pack_examples(examples: Sequence[Example]) -> dict[str, npt.NDArray[Any]]:
    """Flat arrays of a shard (``tokens``, ``loss_mask``, ``deps``, ``offsets``, ``index``)."""
    lengths = [0, *(example.length for example in examples)]
    arrays: dict[str, npt.NDArray[Any]] = {
        "tokens": (
            np.concatenate([e.tokens for e in examples])
            if examples
            else np.zeros(0, dtype=np.uint8)
        ),
        "loss_mask": (
            np.concatenate([e.loss_mask for e in examples])
            if examples
            else np.zeros(0, dtype=np.bool_)
        ),
        "deps": np.asarray([d for e in examples for d in e.meta["deps"]], dtype=np.int32).reshape(
            -1
        ),
        "offsets": np.cumsum(np.asarray(lengths, dtype=np.int64), dtype=np.int64),
        "index": np.asarray([int(e.meta["index"]) for e in examples], dtype=np.int64),
    }
    return arrays


def shard_hash(arrays: dict[str, npt.NDArray[Any]], scalars: dict[str, Any]) -> str:
    """blake2b-256 over the shard header and arrays (fixed order, explicit little-endian dtypes)."""
    header = "\0".join(
        [
            f"hsn-shard/{scalars['format']}",
            str(scalars["task"]),
            str(scalars["variant"]),
            str(scalars["split"]),
            str(scalars["family"]),
            str(scalars["context_id"]),
            str(scalars["unroll"]),
            str(arrays["offsets"].shape[0] - 1),
        ]
    )
    digest = hashlib.blake2b(digest_size=32)
    digest.update(header.encode())
    digest.update(np.ascontiguousarray(arrays["tokens"], dtype=np.uint8).tobytes())
    digest.update(np.ascontiguousarray(arrays["loss_mask"], dtype=np.uint8).tobytes())
    digest.update(np.ascontiguousarray(arrays["deps"], dtype="<i4").tobytes())
    digest.update(np.ascontiguousarray(arrays["offsets"], dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(arrays["index"], dtype="<i8").tobytes())
    return digest.hexdigest()


def _scalars(spec: TaskSpec, split: Split) -> dict[str, Any]:
    return {
        "format": SHARD_FORMAT,
        "task": spec.name,
        "variant": spec.variant,
        "split": split,
        "family": spec.family,
        "context_id": spec.context_id,
        "unroll": spec.unroll,
    }


def _record(
    spec: TaskSpec, split: Split, file: str, arrays: dict[str, npt.NDArray[Any]]
) -> ShardRecord:
    return ShardRecord(
        task=spec.name,
        variant=spec.variant,
        split=split,
        family=spec.family,
        context_id=spec.context_id,
        file=file,
        n_docs=int(arrays["offsets"].shape[0] - 1),
        n_tokens=int(arrays["tokens"].shape[0]),
        n_loss=int(arrays["deps"].shape[0]),
        hash=shard_hash(arrays, _scalars(spec, split)),
    )


def write_shard(
    out_dir: Path, spec: TaskSpec, split: Split, examples: Sequence[Example]
) -> ShardRecord:
    """Write ``<task>.<variant>.<split>.npz`` into ``out_dir`` and return its manifest record."""
    arrays = pack_examples(examples)
    scalars = _scalars(spec, split)
    file = shard_file_name(spec.name, spec.variant, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        **arrays,
        **{key: np.asarray(value) for key, value in scalars.items()},
    }
    np.savez(out_dir / file, **payload)
    return _record(spec, split, file, arrays)


def read_shard(path: Path) -> tuple[dict[str, npt.NDArray[Any]], dict[str, Any]]:
    """The arrays and scalars of a shard file; raises on an unknown format."""
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {
            key: np.asarray(loaded[key])
            for key in ("tokens", "loss_mask", "deps", "offsets", "index")
        }
        scalars: dict[str, Any] = {
            "format": int(loaded["format"]),
            "task": str(loaded["task"]),
            "variant": str(loaded["variant"]),
            "split": str(loaded["split"]),
            "family": str(loaded["family"]),
            "context_id": int(loaded["context_id"]),
            "unroll": int(loaded["unroll"]),
        }
    if scalars["format"] != SHARD_FORMAT:
        raise ValueError(f"{path}: shard format {scalars['format']} != {SHARD_FORMAT}")
    return arrays, scalars


def unpack_examples(arrays: dict[str, npt.NDArray[Any]], scalars: dict[str, Any]) -> list[Example]:
    """Rebuild the documents of a shard (targets, context and dep_len are derived)."""
    tokens = np.ascontiguousarray(arrays["tokens"], dtype=np.uint8)
    loss_mask = np.ascontiguousarray(arrays["loss_mask"], dtype=np.bool_)
    deps = np.ascontiguousarray(arrays["deps"], dtype=np.int64)
    offsets = np.ascontiguousarray(arrays["offsets"], dtype=np.int64)
    index = np.ascontiguousarray(arrays["index"], dtype=np.int64)
    n = int(offsets.shape[0] - 1)
    if tokens.shape[0] != loss_mask.shape[0] or int(offsets[-1]) != tokens.shape[0]:
        raise ValueError("shard arrays disagree on the number of tokens")
    if index.shape[0] != n or int(loss_mask.sum()) != deps.shape[0]:
        raise ValueError("shard arrays disagree on the number of documents or loss positions")
    examples: list[Example] = []
    dep_cursor = 0
    task, variant, split = scalars["task"], scalars["variant"], scalars["split"]
    unroll = int(scalars["unroll"])
    for i in range(n):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        doc_tokens = tokens[lo:hi].copy()
        doc_mask = loss_mask[lo:hi].copy()
        n_loss = int(doc_mask.sum())
        doc_deps = deps[dep_cursor : dep_cursor + n_loss].tolist()
        dep_cursor += n_loss
        targets = np.empty_like(doc_tokens)
        targets[:-1] = doc_tokens[1:]
        targets[-1] = PAD
        dep_len = max(doc_deps) if doc_deps else 0
        examples.append(
            Example(
                doc_tokens,
                targets,
                doc_mask,
                np.full(hi - lo, int(scalars["context_id"]), dtype=np.uint8),
                {
                    "task": task,
                    "variant": variant,
                    "family": scalars["family"],
                    "split": split,
                    "index": int(index[i]),
                    "doc_id": f"{task}:{variant}:{split}:{int(index[i])}",
                    "dep_len": dep_len,
                    "in_range": dep_len <= unroll,
                    "deps": tuple(doc_deps),
                },
            )
        )
    return examples


def load_shard(path: Path, expected_hash: str | None = None) -> list[Example]:
    """Documents of a shard file; with ``expected_hash`` the recomputed hash must match."""
    arrays, scalars = read_shard(path)
    if expected_hash is not None:
        actual = shard_hash(arrays, scalars)
        if actual != expected_hash:
            raise ValueError(f"{path}: hash {actual} != manifest hash {expected_hash}")
    return unpack_examples(arrays, scalars)


# --------------------------------------------------------------------------------------------------
# build / verify
# --------------------------------------------------------------------------------------------------


def _dataset(specs: DatasetSpec | Sequence[TaskSpec], out_dir: Path) -> DatasetSpec:
    if isinstance(specs, DatasetSpec):
        return specs
    tasks = list(specs)
    if not tasks:
        raise ValueError("build_splits needs at least one TaskSpec")
    return DatasetSpec(name=out_dir.name, unroll=tasks[0].unroll, tasks=tasks)


def _examples(spec: TaskSpec, split: Split) -> list[Example]:
    return [generate(spec, split, index) for index in range(spec.n_examples(split))]


def _split_hashes(records: Sequence[ShardRecord]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for split in SPLITS:
        digest = hashlib.blake2b(digest_size=32)
        for record in records:
            if record.split == split:
                digest.update(bytes.fromhex(record.hash))
        hashes[split] = digest.hexdigest()
    return hashes


def _pair(train: str, test: str, index: NGramIndex, examples: Sequence[Example]) -> LeakagePair:
    report = index.check(examples)
    return LeakagePair(
        train=train,
        test=test,
        n_gram=report.n_gram,
        exact_duplicates=report.exact_duplicates,
        overlapping_ngrams=report.overlapping_ngrams,
        test_docs_with_overlap=report.test_docs_with_overlap,
    )


def build_splits(
    specs: DatasetSpec | Sequence[TaskSpec],
    out_dir: Path,
    *,
    n_gram: int = 32,
    leakage: bool = True,
) -> SplitManifest:
    """Generate every split of every task into ``out_dir`` and write ``manifest.json``.

    With ``leakage`` (the default) every train split is checked against every dev and test split
    of every task and each task's dev split against its test split; the counts are recorded in
    the manifest. The manifest is returned and written whether or not it is clean; callers decide
    (the CLI exits non-zero when it is not).
    """
    out_dir = Path(out_dir)
    dataset = _dataset(specs, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[ShardRecord] = []
    held: dict[str, list[Example]] = {}  # dev and test documents, kept for the leakage checks
    for task in dataset.tasks:
        for split in ("dev", "test"):
            examples = _examples(task, split)
            records.append(write_shard(out_dir, task, split, examples))
            held[f"{task.key}/{split}"] = examples
    pairs: list[LeakagePair] = []
    for task in dataset.tasks:
        train = _examples(task, "train")
        records.append(write_shard(out_dir, task, "train", train))
        if not leakage:
            continue
        index = NGramIndex(train, n_gram)
        del train
        train_key = f"{task.key}/train"
        for other in dataset.tasks:
            for split in ("dev", "test"):
                test_key = f"{other.key}/{split}"
                pairs.append(_pair(train_key, test_key, index, held[test_key]))
        dev_index = NGramIndex(held[f"{task.key}/dev"], n_gram)
        pairs.append(
            _pair(f"{task.key}/dev", f"{task.key}/test", dev_index, held[f"{task.key}/test"])
        )
    order = {split: i for i, split in enumerate(SPLITS)}
    task_order = {task.key: i for i, task in enumerate(dataset.tasks)}
    records.sort(key=lambda r: (task_order[f"{r.task}/{r.variant}"], order[r.split]))
    manifest = SplitManifest(
        name=dataset.name,
        spec_hash=dataset.spec_hash(),
        unroll=dataset.unroll,
        hsn_version=hsn.__version__,
        numpy_version=np.__version__,
        shards=records,
        split_hashes=_split_hashes(records),
        leakage=pairs,
    )
    manifest.write(out_dir / MANIFEST_NAME)
    return manifest


def verify_splits(dataset: DatasetSpec, out_dir: Path) -> list[str]:
    """Regenerate every shard and compare with ``out_dir/manifest.json`` and the files on disk.

    Returns a list of problems (empty when everything agrees). Every regenerated document is
    also checked against its reference solver (:func:`~hsn.data.synthetic.check_example`).
    """
    out_dir = Path(out_dir)
    problems: list[str] = []
    try:
        manifest = SplitManifest.load(out_dir / MANIFEST_NAME)
    except (OSError, ValueError) as exc:
        return [f"cannot read {out_dir / MANIFEST_NAME}: {exc}"]
    if manifest.spec_hash != dataset.spec_hash():
        problems.append(
            f"spec hash {dataset.spec_hash()} != manifest spec hash {manifest.spec_hash}"
        )
    if manifest.unroll != dataset.unroll:
        problems.append(f"unroll {dataset.unroll} != manifest unroll {manifest.unroll}")
    for task in dataset.tasks:
        for split in SPLITS:
            try:
                record = manifest.shard(task.name, task.variant, split)
            except KeyError as exc:
                problems.append(str(exc))
                continue
            examples = _examples(task, split)
            for example in examples:
                try:
                    check_example(task, example)
                except ValueError as exc:
                    problems.append(str(exc))
                    break
            regenerated = shard_hash(pack_examples(examples), _scalars(task, split))
            if regenerated != record.hash:
                problems.append(f"{record.key}: regenerated hash {regenerated} != {record.hash}")
            path = out_dir / record.file
            try:
                arrays, scalars = read_shard(path)
            except (OSError, ValueError, KeyError) as exc:
                problems.append(f"{record.key}: cannot read {path}: {exc}")
                continue
            on_disk = shard_hash(arrays, scalars)
            if on_disk != record.hash:
                problems.append(f"{record.key}: file hash {on_disk} != manifest hash {record.hash}")
    return problems
