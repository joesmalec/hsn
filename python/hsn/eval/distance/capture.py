"""Every loss position of an evaluation pass, kept (``eval.by_distance``).

:class:`PositionCapture` is a window observer (:data:`hsn.eval.evaluate.WindowObserver`): for
every loss position of the pass it keeps the loss (nats, :func:`torch.nn.functional.
cross_entropy` on the float32 logits, as the ``loss`` metric) and the arg-max, placed by document
and position, and for the documents named in ``rows_for`` the whole row of log-probabilities.
Nothing it computes enters a metric.

A position's place comes from the window alone: ``doc`` names the document, and the position in
the document is the count since the row's last ``reset`` (True exactly at document starts), with
the count a row reached in the window before for a document that continues. As a guard the
window's token at every kept position must equal the document's token there, and the pass must
have visited every loss position of the dataset exactly once (:meth:`PositionCapture.result`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import Tensor

from hsn.data.dataset import ByteDataset

__all__ = ["Captured", "PositionCapture", "flat_documents"]


def flat_documents(ds: ByteDataset) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.uint8]]:
    """``(offsets [n_docs + 1], tokens [n_tokens])`` of the dataset's documents, in order."""
    lengths = np.fromiter((e.length for e in ds.examples), dtype=np.int64, count=len(ds))
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths)])
    tokens = (
        np.concatenate([e.tokens for e in ds.examples]) if len(ds) else np.zeros(0, dtype=np.uint8)
    )
    return offsets, tokens


@dataclass(frozen=True)
class Captured:
    """One pass, flat over the dataset's documents in order (document ``d`` is
    ``[offsets[d], offsets[d + 1])``)."""

    offsets: npt.NDArray[np.int64]
    tokens: npt.NDArray[np.uint8]
    nll: npt.NDArray[np.float32]  # nats; NaN where the position has no loss
    pred: npt.NDArray[np.int16]  # the arg-max symbol; -1 where the position has no loss
    row_at: npt.NDArray[np.int32]  # index into ``rows``; -1 where no row was kept
    rows: npt.NDArray[np.float32]  # [n_rows, V] log-probabilities

    @property
    def scored(self) -> npt.NDArray[np.bool_]:
        """The positions that have a loss."""
        return np.isfinite(self.nll)

    @property
    def hit(self) -> npt.NDArray[np.bool_]:
        """Whether the arg-max was the target (False where the position has no loss)."""
        targets = np.concatenate([self.tokens[1:], np.zeros(1, dtype=np.uint8)])
        hit: npt.NDArray[np.bool_] = self.pred == targets
        return hit

    def save(self, path: Path) -> None:
        """Write the per-position dump of the pass: ``nll`` (float32 nats, NaN without a
        target), ``hit`` and ``offsets``, flat in document order (the layout the grammar
        evaluation's dumps have, so the same tools read both)."""
        np.savez(path, nll=self.nll, hit=self.hit, offsets=self.offsets)

    @classmethod
    def from_dump(cls, ds: ByteDataset, path: Path) -> Captured:
        """A pass read back from :meth:`save`'s layout for the documents of ``ds``. A dump
        holds no predictions: ``pred`` is the target where the position was hit and -2 where
        it was not, and there are no rows, which is all a text table reads."""
        offsets, tokens = flat_documents(ds)
        with np.load(path) as dump:
            nll, hit = dump["nll"].astype(np.float32), dump["hit"].astype(np.bool_)
            if not np.array_equal(dump["offsets"], offsets):
                raise ValueError(f"{path}: the dump's documents are not this dataset's")
        expected = np.concatenate([e.loss_mask for e in ds.examples])
        if nll.shape != expected.shape or not np.array_equal(np.isfinite(nll), expected):
            raise ValueError(f"{path}: the dump's scored positions are not this dataset's")
        targets = np.concatenate([tokens[1:], np.zeros(1, dtype=np.uint8)]).astype(np.int16)
        pred = np.where(expected, np.where(hit, targets, -2), -1).astype(np.int16)
        return cls(
            offsets,
            tokens,
            nll,
            pred,
            np.full(nll.shape[0], -1, dtype=np.int32),
            np.zeros((0, 0), dtype=np.float32),
        )


class PositionCapture:
    """The observer of one pass over ``ds`` (module docstring). ``rows_for[d]`` True keeps the
    log-probability rows of document ``d``'s loss positions (about 1 KiB each at V = 256)."""

    def __init__(self, ds: ByteDataset, rows_for: Sequence[bool] | None = None) -> None:
        self._offsets, self._tokens = flat_documents(ds)
        n_tokens = int(self._offsets[-1])
        self._expected = np.zeros(n_tokens, dtype=np.bool_)
        for d, example in enumerate(ds.examples):
            self._expected[self._offsets[d] : self._offsets[d + 1]] = example.loss_mask
        self._offsets_t = torch.from_numpy(self._offsets)
        self._tokens_t = torch.from_numpy(self._tokens)
        self._rows_for: Tensor | None = None
        if rows_for is not None and any(rows_for):
            if len(rows_for) != len(ds):
                raise ValueError("rows_for must have one entry per document")
            self._rows_for = torch.tensor(list(rows_for), dtype=torch.bool)
        self._carry: Tensor | None = None
        self._flat: list[npt.NDArray[np.int64]] = []
        self._nll: list[npt.NDArray[np.float32]] = []
        self._pred: list[npt.NDArray[np.int16]] = []
        self._row_flat: list[npt.NDArray[np.int64]] = []
        self._rows: list[npt.NDArray[np.float32]] = []
        self._vocab = 0

    def __call__(self, window: Mapping[str, Tensor], logits: Tensor) -> None:
        doc, reset, mask = window["doc"], window["reset"], window["loss_mask"]
        B, T = doc.shape
        if self._carry is None:
            self._carry = torch.zeros(B, dtype=torch.int64)
        index = torch.arange(T, dtype=torch.int64).expand(B, T)
        start = torch.where(reset, index, torch.full_like(index, -1)).cummax(dim=1).values
        position = torch.where(start >= 0, index - start, self._carry[:, None] + index)
        self._carry = torch.where(doc[:, -1] >= 0, position[:, -1] + 1, 0)
        if not bool(mask.any()):
            return
        docs = doc[mask]
        flat = self._offsets_t[docs] + position[mask]
        if bool((docs < 0).any()) or not torch.equal(
            self._tokens_t[flat], window["tokens"][mask].to(torch.uint8)
        ):
            raise RuntimeError(
                "by-distance capture: a window's loss positions do not match their documents "
                "(the windows are not this dataset's, in order)"
            )
        selected = logits[mask.to(logits.device)].float()
        targets = window["targets"][mask].to(logits.device)
        self._vocab = int(selected.shape[-1])
        self._flat.append(flat.numpy())
        self._nll.append(F.cross_entropy(selected, targets, reduction="none").cpu().numpy())
        self._pred.append(selected.argmax(dim=-1).cpu().numpy().astype(np.int16))
        if self._rows_for is not None:
            want = self._rows_for[docs]
            if bool(want.any()):
                rows = torch.log_softmax(selected[want.to(selected.device)], dim=-1)
                self._row_flat.append(flat[want].numpy())
                self._rows.append(rows.cpu().numpy().astype(np.float32))

    def result(self) -> Captured:
        """The pass as flat arrays; raises unless every loss position was visited once."""
        n_tokens = int(self._offsets[-1])
        flat = np.concatenate(self._flat) if self._flat else np.zeros(0, dtype=np.int64)
        seen = np.bincount(flat, minlength=n_tokens)
        if not np.array_equal(seen, self._expected.astype(seen.dtype)):
            raise RuntimeError(
                f"by-distance capture: the pass visited {int(flat.shape[0])} loss positions, the "
                f"dataset has {int(self._expected.sum())} (every one must be visited once)"
            )
        nll = np.full(n_tokens, np.nan, dtype=np.float32)
        pred = np.full(n_tokens, -1, dtype=np.int16)
        row_at = np.full(n_tokens, -1, dtype=np.int32)
        if flat.shape[0]:
            nll[flat] = np.concatenate(self._nll)
            pred[flat] = np.concatenate(self._pred)
        rows: npt.NDArray[Any] = np.zeros((0, self._vocab), dtype=np.float32)
        if self._rows:
            rows = np.concatenate(self._rows)
            row_flat = np.concatenate(self._row_flat)
            row_at[row_flat] = np.arange(row_flat.shape[0], dtype=np.int32)
        return Captured(self._offsets, self._tokens, nll, pred, row_at, rows)
