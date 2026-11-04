"""Leakage checker (P0.4): exact duplicates and n-gram overlaps between two sets of documents.

:func:`check_leakage` compares a *train* set against a *test* set (dev or test split, another
task's split, an evaluation suite) and reports

* ``exact_duplicates``: test documents whose token bytes equal some train document, and
* ``overlapping_ngrams``: distinct test n-grams (``n_gram`` consecutive tokens inside one
  document; windows never cross a document boundary) that also occur in a train document,
  with ``test_docs_with_overlap`` the number of test documents containing at least one.

Implementation: every n-gram is hashed with a 64-bit polynomial hash computed vectorised over the
concatenated tokens (``n`` multiply-adds on ``uint64`` arrays, wrapping modulo 2**64; every
operand stays ``uint64`` because NumPy 2 promotes ``uint64 * int64`` to ``float64``). Train hashes
are sorted once (:class:`NGramIndex`), test hashes are looked up with ``searchsorted`` and every
candidate hit is verified byte for byte, so a hash collision never counts as leakage. A clean
report is zero in both counters; the build CLI exits non-zero otherwise (errors over silence).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from hsn.data.synthetic import Example

__all__ = ["LeakageHit", "LeakageReport", "NGramIndex", "check_leakage"]

_BASE = np.uint64(0x9E3779B97F4A7C15)  # odd multiplier of the polynomial hash (mod 2**64)
_DIGEST_SIZE = 16


@dataclass(frozen=True)
class LeakageHit:
    """One verified n-gram shared by a test document and a train document."""

    test_doc: str
    test_pos: int
    train_doc: str
    train_pos: int


@dataclass(frozen=True)
class LeakageReport:
    """Result of :func:`check_leakage`; ``is_clean`` means no duplicate and no shared n-gram."""

    n_train: int
    n_test: int
    n_gram: int
    exact_duplicates: int  # test documents equal to a train document
    overlapping_ngrams: int  # distinct test n-grams that also occur in train
    test_docs_with_overlap: int  # test documents with at least one shared n-gram
    examples: tuple[LeakageHit, ...]  # the first verified hits

    @property
    def is_clean(self) -> bool:
        return self.exact_duplicates == 0 and self.overlapping_ngrams == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_gram": self.n_gram,
            "exact_duplicates": self.exact_duplicates,
            "overlapping_ngrams": self.overlapping_ngrams,
            "test_docs_with_overlap": self.test_docs_with_overlap,
            "is_clean": self.is_clean,
            "examples": [
                {
                    "test_doc": hit.test_doc,
                    "test_pos": hit.test_pos,
                    "train_doc": hit.train_doc,
                    "train_pos": hit.train_pos,
                }
                for hit in self.examples
            ],
        }


@dataclass(frozen=True)
class _Corpus:
    tokens: npt.NDArray[np.uint8]  # every document concatenated
    offsets: npt.NDArray[np.int64]  # document i spans tokens[offsets[i]:offsets[i + 1]]
    doc_ids: tuple[str, ...]
    digests: tuple[bytes, ...]  # blake2b of each document's bytes


def _corpus(examples: Iterable[Example]) -> _Corpus:
    parts: list[npt.NDArray[np.uint8]] = []
    lengths: list[int] = [0]
    doc_ids: list[str] = []
    digests: list[bytes] = []
    for example in examples:
        parts.append(example.tokens)
        lengths.append(example.length)
        doc_ids.append(str(example.meta["doc_id"]))
        digests.append(hashlib.blake2b(example.tokens.tobytes(), digest_size=_DIGEST_SIZE).digest())
    tokens: npt.NDArray[np.uint8] = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)
    offsets: npt.NDArray[np.int64] = np.cumsum(np.asarray(lengths, dtype=np.int64), dtype=np.int64)
    return _Corpus(tokens, offsets, tuple(doc_ids), tuple(digests))


def _window_hashes(corpus: _Corpus, n: int) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
    """Hash of every n-gram that lies inside one document, with its start in the flat array."""
    total = int(corpus.tokens.shape[0])
    if total < n:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(corpus.tokens, n)
    hashes: npt.NDArray[np.uint64] = np.zeros(total - n + 1, dtype=np.uint64)
    for i in range(n):
        hashes = hashes * _BASE + windows[:, i].astype(np.uint64)
    starts: npt.NDArray[np.int64] = np.arange(total - n + 1, dtype=np.int64)
    doc = np.searchsorted(corpus.offsets, starts, side="right") - 1
    inside = starts + n <= corpus.offsets[doc + 1]
    return hashes[inside], starts[inside]


def _doc_of(corpus: _Corpus, position: int) -> int:
    return int(np.searchsorted(corpus.offsets, position, side="right")) - 1


class NGramIndex:
    """Sorted n-gram hashes of a train set; :meth:`check` compares test sets against it."""

    def __init__(self, train: Iterable[Example], n_gram: int = 32) -> None:
        if n_gram < 1:
            raise ValueError(f"n_gram must be >= 1, got {n_gram}")
        self.n_gram = n_gram
        self._train = _corpus(train)
        hashes, starts = _window_hashes(self._train, n_gram)
        order = np.argsort(hashes, kind="stable")
        self._hashes = hashes[order]
        self._starts = starts[order]
        self._doc_by_digest: dict[bytes, int] = {}
        for i, digest in enumerate(self._train.digests):
            self._doc_by_digest.setdefault(digest, i)

    @property
    def n_train(self) -> int:
        return len(self._train.doc_ids)

    def check(self, test: Iterable[Example], max_examples: int = 10) -> LeakageReport:
        corpus = _corpus(test)
        n = self.n_gram
        exact = sum(1 for digest in corpus.digests if digest in self._doc_by_digest)
        hashes, starts = _window_hashes(corpus, n)
        hits: list[LeakageHit] = []
        shared: set[bytes] = set()
        docs_with_overlap: set[int] = set()
        if self._hashes.shape[0] and hashes.shape[0]:
            first = np.searchsorted(self._hashes, hashes, side="left")
            clipped = np.minimum(first, self._hashes.shape[0] - 1)
            candidates = np.flatnonzero(self._hashes[clipped] == hashes)
            train_tokens = self._train.tokens
            for k in candidates.tolist():
                start = int(starts[k])
                window = corpus.tokens[start : start + n].tobytes()
                j = int(first[k])
                while j < self._hashes.shape[0] and self._hashes[j] == hashes[k]:
                    train_start = int(self._starts[j])
                    if train_tokens[train_start : train_start + n].tobytes() == window:
                        shared.add(window)
                        test_doc = _doc_of(corpus, start)
                        docs_with_overlap.add(test_doc)
                        if len(hits) < max_examples:
                            train_doc = _doc_of(self._train, train_start)
                            hits.append(
                                LeakageHit(
                                    test_doc=corpus.doc_ids[test_doc],
                                    test_pos=start - int(corpus.offsets[test_doc]),
                                    train_doc=self._train.doc_ids[train_doc],
                                    train_pos=train_start - int(self._train.offsets[train_doc]),
                                )
                            )
                        break
                    j += 1
        return LeakageReport(
            n_train=self.n_train,
            n_test=len(corpus.doc_ids),
            n_gram=n,
            exact_duplicates=exact,
            overlapping_ngrams=len(shared),
            test_docs_with_overlap=len(docs_with_overlap),
            examples=tuple(hits),
        )


def check_leakage(
    train: Iterable[Example], test: Iterable[Example], n_gram: int = 32
) -> LeakageReport:
    """Exact duplicates and shared n-grams between ``train`` and ``test`` (see the module doc)."""
    return NGramIndex(train, n_gram).check(test)
