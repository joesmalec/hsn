"""Byte dataset and window collation (P0.4).

:class:`ByteDataset` holds documents (:class:`~hsn.data.synthetic.Example`) in memory, either
given directly (tiny sets for tests and overfit checks) or loaded from the shards a manifest
names (:meth:`ByteDataset.load`, which recomputes every shard hash unless told not to). Each item
is a dict of tensors ``tokens``, ``targets``, ``context`` (uint8), ``loss_mask`` (bool), ``reset``
(bool, True at position 0 only: the recurrent state is reset at every document start) and
``dep`` (int32, the dependency distance at loss positions, 0 elsewhere).

:func:`collate_windows` turns a list of documents into ``batch`` parallel *streams*: documents are
assigned greedily to the currently shortest row and concatenated, rows are padded to a common
length ``T`` that is a multiple of ``unroll``, and ``reset`` is True exactly at document starts.
The trainer (P0.5, P1.7) iterates windows ``[:, w * unroll:(w + 1) * unroll]`` along ``T`` and
carries the recurrent state per row between consecutive windows, detaching it at every window
start (truncated BPTT). A dependency that straddles a window boundary therefore gets no gradient
through the window boundary, only a carried state: that is why ``synthetic_v1`` keeps in-range
documents no longer than ``unroll`` and why ``align_docs`` exists (every document starts at a
window boundary; the tail of its last window is padding). Padding is ``PAD`` with ``loss_mask``
False and ``reset`` False; ``doc`` maps every position back to the index of its document in the
input list (``-1`` on padding) so per-document metadata can be joined for evaluation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from torch.utils.data import Dataset

from hsn.config import Split
from hsn.data.splits import SplitManifest, load_shard
from hsn.data.synthetic import PAD, Example, Variant

__all__ = ["ByteDataset", "collate_windows", "example_tensors", "iter_windows"]

Batch = dict[str, torch.Tensor]
_FIELDS = ("tokens", "targets", "loss_mask", "context", "reset", "dep")


def example_tensors(example: Example) -> Batch:
    """The per-document tensors of :class:`ByteDataset` (arrays are copied: they are read-only)."""
    length = example.length
    reset = np.zeros(length, dtype=np.bool_)
    reset[0] = True
    dep = np.zeros(length, dtype=np.int32)
    dep[example.loss_mask] = np.asarray(example.meta["deps"], dtype=np.int32)
    return {
        "tokens": torch.from_numpy(example.tokens.copy()),
        "targets": torch.from_numpy(example.targets.copy()),
        "loss_mask": torch.from_numpy(example.loss_mask.copy()),
        "context": torch.from_numpy(example.context.copy()),
        "reset": torch.from_numpy(reset),
        "dep": torch.from_numpy(dep),
    }


class ByteDataset(Dataset[Batch]):
    """Documents as tensors; see the module docstring for the item layout."""

    def __init__(self, examples: Sequence[Example]) -> None:
        self._examples: tuple[Example, ...] = tuple(examples)
        for i, example in enumerate(self._examples):
            if not isinstance(example, Example):
                raise TypeError(f"examples[{i}] is {type(example).__name__}, not Example")

    @classmethod
    def load(
        cls,
        manifest: str | Path | SplitManifest,
        split: Split,
        tasks: Sequence[str] | None = None,
        variants: Sequence[Variant] | None = None,
        *,
        verify: bool = True,
        expected_unroll: int | None = None,
        root: str | Path | None = None,
    ) -> Self:
        """Documents of ``split`` from the shards a manifest names, in manifest order.

        ``tasks`` and ``variants`` filter by task name and variant (``None`` = all). Shard files
        are resolved relative to the manifest's directory (or ``root``). With ``verify`` every
        shard hash is recomputed and must match; ``expected_unroll`` (``train.unroll``) must equal
        the manifest's unroll when given. Naming a task or variant the manifest lacks raises.
        """
        if isinstance(manifest, SplitManifest):
            loaded = manifest
            if root is None:
                raise ValueError("root is required when a SplitManifest object is given")
            base = Path(root)
        else:
            path = Path(manifest)
            loaded = SplitManifest.load(path)
            base = Path(root) if root is not None else path.parent
        if expected_unroll is not None and loaded.unroll != expected_unroll:
            raise ValueError(
                f"{loaded.name}: manifest unroll {loaded.unroll} != expected {expected_unroll}"
            )
        known_tasks = {record.task for record in loaded.shards}
        known_variants = {record.variant for record in loaded.shards}
        if tasks is not None:
            unknown = [task for task in tasks if task not in known_tasks]
            if unknown:
                raise KeyError(f"{loaded.name}: unknown tasks {unknown}; has {sorted(known_tasks)}")
        if variants is not None:
            unknown_variants = [v for v in variants if v not in known_variants]
            if unknown_variants:
                raise KeyError(f"{loaded.name}: unknown variants {unknown_variants}")
        examples: list[Example] = []
        for record in loaded.shards:
            if record.split != split:
                continue
            if tasks is not None and record.task not in tasks:
                continue
            if variants is not None and record.variant not in variants:
                continue
            examples.extend(load_shard(base / record.file, record.hash if verify else None))
        return cls(examples)

    @property
    def examples(self) -> tuple[Example, ...]:
        return self._examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> Batch:
        return example_tensors(self._examples[index])

    def meta(self, index: int) -> dict[str, Any]:
        return dict(self._examples[index].meta)


def _as_tensors(item: Example | Mapping[str, torch.Tensor]) -> Batch:
    if isinstance(item, Example):
        return example_tensors(item)
    missing = [field for field in _FIELDS if field not in item]
    if missing:
        raise KeyError(f"collate_windows: document is missing {missing}")
    return {field: item[field] for field in _FIELDS}


def collate_windows(
    examples: Sequence[Example | Mapping[str, torch.Tensor]],
    unroll: int,
    batch: int = 1,
    align_docs: bool = False,
) -> Batch:
    """Concatenate documents into ``batch`` streams of ``T`` positions (a multiple of ``unroll``).

    Returns ``tokens``, ``targets``, ``context`` (int64 ``[B, T]``, ready for embeddings),
    ``loss_mask`` and ``reset`` (bool ``[B, T]``), ``dep`` (int32 ``[B, T]``) and ``doc`` (int64
    ``[B, T]``, index into ``examples`` or ``-1`` on padding). ``reset`` is True exactly at
    document starts, which includes position 0 of every non-empty row.
    """
    if unroll < 1:
        raise ValueError(f"unroll must be >= 1, got {unroll}")
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if not examples:
        raise ValueError("collate_windows needs at least one document")
    docs = [_as_tensors(item) for item in examples]
    lengths = [int(doc["tokens"].shape[0]) for doc in docs]
    if any(length == 0 for length in lengths):
        raise ValueError("collate_windows: empty document")
    rows: list[list[int]] = [[] for _ in range(batch)]
    row_len = [0] * batch
    for i, length in enumerate(lengths):
        shortest = min(range(batch), key=lambda r: (row_len[r], r))
        rows[shortest].append(i)
        padded = -(-length // unroll) * unroll if align_docs else length
        row_len[shortest] += padded
    width = -(-max(row_len) // unroll) * unroll
    tokens = np.full((batch, width), PAD, dtype=np.int64)
    targets = np.full((batch, width), PAD, dtype=np.int64)
    context = np.zeros((batch, width), dtype=np.int64)
    loss_mask = np.zeros((batch, width), dtype=np.bool_)
    reset = np.zeros((batch, width), dtype=np.bool_)
    dep = np.zeros((batch, width), dtype=np.int32)
    doc_index = np.full((batch, width), -1, dtype=np.int64)
    for r, row in enumerate(rows):
        cursor = 0
        for i in row:
            doc = docs[i]
            length = lengths[i]
            end = cursor + length
            tokens[r, cursor:end] = doc["tokens"].numpy()
            targets[r, cursor:end] = doc["targets"].numpy()
            context[r, cursor:end] = doc["context"].numpy()
            loss_mask[r, cursor:end] = doc["loss_mask"].numpy()
            dep[r, cursor:end] = doc["dep"].numpy()
            doc_index[r, cursor:end] = i
            reset[r, cursor] = True
            cursor = -(-end // unroll) * unroll if align_docs else end
    return {
        "tokens": torch.from_numpy(tokens),
        "targets": torch.from_numpy(targets),
        "context": torch.from_numpy(context),
        "loss_mask": torch.from_numpy(loss_mask),
        "reset": torch.from_numpy(reset),
        "dep": torch.from_numpy(dep),
        "doc": torch.from_numpy(doc_index),
    }


def iter_windows(batch: Mapping[str, torch.Tensor], unroll: int) -> Iterator[Batch]:
    """The consecutive ``[:, w * unroll:(w + 1) * unroll]`` windows of a collated batch."""
    width = int(batch["tokens"].shape[1])
    if width % unroll != 0:
        raise ValueError(f"batch width {width} is not a multiple of unroll {unroll}")
    for start in range(0, width, unroll):
        yield {key: value[:, start : start + unroll] for key, value in batch.items()}
