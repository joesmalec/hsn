"""Accuracy by recall distance on the synthetic tasks (``eval.by_distance``).

Every loss position of a synthetic document is an answer with an annotated dependency distance
(``meta["deps"]``, :mod:`hsn.data.synthetic`: positions back to the earliest input symbol the
answer depends on). :func:`recall_table` groups one task/variant group's answers:

* ``by_distance``: by the distance (:data:`DISTANCE_KEYS`; the edges are those of the 18 Nov
  2026 out-of-range analysis: synthetic_v1's ranges end at 255 and 768, the far set's at 3,072);
* ``by_windows_back``: by how many windows before the answer's window the fact lies;
* for the recall families (``kv``, ``delayed`` and their word-key forms) also ``by_query`` (the
  document's first answer, its anchor, against the later ones) and ``by_fact_position``: the
  fact's position in the document against the distance (coarse buckets), which separates "far
  back" from "late in a long document";
* ``kv`` families: ``by_target`` (answers that are a value against answers that are ``NONE``),
  ``predicts_none`` and the distance table of the value answers alone;
* ``delayed`` families: ``own_values``, the accuracy when the prediction is restricted to the
  values stated in the answer's own document (does the model rank the right one first), with
  its chance level, the share of predictions that are one of them and the probability mass on
  them. A document's values are its bytes before the first ``SEP`` that lie in the value
  alphabet, and the alphabet is the byte range the group's answers span (:func:`value_range`,
  reported as ``alphabet``): the generators draw values from one range that keys and filler
  stay out of, so the range is exact once the smallest and the largest value were each asked
  for somewhere in the group.

The fact's position is the answer's position minus its distance (the definition of
``hsn.data.synthetic.fact_positions`` on ``claude/far-recall-data``), unless the document's
metadata carries the positions themselves (:data:`FACT_POSITION_KEY`: one entry per answer).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from hsn.data.synthetic import NONE, SEP, Example
from hsn.eval.distance.capture import Captured

__all__ = [
    "DISTANCE_KEYS",
    "FACT_POSITION_KEY",
    "RECALL_FAMILIES",
    "fact_positions",
    "recall_table",
    "value_range",
]

#: Families whose answers recall a stated fact (first/later query, fact position).
RECALL_FAMILIES: tuple[str, ...] = ("kv", "delayed", "kv_words", "delayed_words")
_NONE_FAMILIES: tuple[str, ...] = ("kv", "kv_words")
_OWN_VALUE_FAMILIES: tuple[str, ...] = ("delayed", "delayed_words")

#: Distance buckets: a distance ``d`` is in bucket ``i`` when ``_EDGES[i] <= d < _EDGES[i + 1]``.
_EDGES: tuple[int, ...] = (0, 64, 256, 512, 769, 1024, 1536, 2048, 3073)
DISTANCE_KEYS: tuple[str, ...] = (
    "lt64",
    "64_255",
    "256_511",
    "512_768",
    "769_1023",
    "1024_1535",
    "1536_2047",
    "2048_3072",
    "gt3072",
)
#: The coarse grid of ``by_fact_position``: fact position (rows) against distance (columns).
_COARSE_EDGES: tuple[int, ...] = (0, 256, 768, 1536)
_FACT_KEYS: tuple[str, ...] = ("0_255", "256_767", "768_1535", "ge1536")
_COARSE_DISTANCE_KEYS: tuple[str, ...] = ("lt256", "256_767", "768_1535", "ge1536")

#: The metadata key under which a document may carry its answers' fact positions.
FACT_POSITION_KEY = "fact_pos"

Float = npt.NDArray[np.float64]
Int = npt.NDArray[np.int64]
Bool = npt.NDArray[np.bool_]


def fact_positions(example: Example) -> Int:
    """The position in the document of the fact each answer depends on, one per loss position:
    the metadata's own array when the shard has one, else answer position minus distance."""
    positions = np.flatnonzero(example.loss_mask).astype(np.int64)
    stored = example.meta.get(FACT_POSITION_KEY)
    if stored is None:
        return positions - np.asarray(example.meta["deps"], dtype=np.int64)
    out = np.asarray(stored, dtype=np.int64)
    if out.shape != positions.shape:
        raise ValueError(
            f"{example.meta.get('doc_id')}: meta[{FACT_POSITION_KEY!r}] has {out.shape[0]} "
            f"entries for {positions.shape[0]} answers"
        )
    return out


def value_range(examples: Sequence[Example]) -> tuple[int, int] | None:
    """``(lowest, highest)`` byte among the answers of ``examples`` that are not ``NONE``
    (``None`` without such an answer): the value alphabet of a recall group."""
    low, high = 256, -1
    for example in examples:
        answers = example.targets[example.loss_mask]
        answers = answers[answers != NONE]
        if answers.shape[0]:
            low, high = min(low, int(answers.min())), max(high, int(answers.max()))
    return (low, high) if high >= 0 else None


def _cell(ok: Float, select: Bool) -> dict[str, Any]:
    n = int(select.sum())
    return {"n": n, "accuracy": float(ok[select].mean()) if n else None}


def _by_bucket(ok: Float, bucket: Int, keys: Sequence[str], select: Bool) -> dict[str, Any]:
    """``{key: cell}`` of the buckets that hold an answer of ``select``, in bucket order."""
    out: dict[str, Any] = {}
    for i, key in enumerate(keys):
        chosen = select & (bucket == i)
        if chosen.any():
            out[key] = _cell(ok, chosen)
    return out


def recall_table(
    examples: Sequence[Example],
    docs: Sequence[int],
    captured: Captured,
    *,
    window: int,
    values: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """The table of one task/variant group: ``examples`` are its documents and ``docs`` their
    indices in the captured dataset; ``window`` is the evaluation window (``train.unroll``);
    ``values`` is the group's value alphabet (:func:`value_range` of ``examples`` by default;
    give the whole group's when fewer documents are scored)."""
    family = str(examples[0].meta["family"])
    doc_of, slot, position, distance, fact = [], [], [], [], []
    for local, example in enumerate(examples):
        at = np.flatnonzero(example.loss_mask).astype(np.int64)
        doc_of.append(np.full(at.shape[0], local, dtype=np.int64))
        slot.append(np.arange(at.shape[0], dtype=np.int64))
        position.append(at)
        distance.append(np.asarray(example.meta["deps"], dtype=np.int64))
        fact.append(fact_positions(example))
    doc_a, slot_a, pos_a = np.concatenate(doc_of), np.concatenate(slot), np.concatenate(position)
    dist_a, fact_a = np.concatenate(distance), np.concatenate(fact)
    flat = captured.offsets[np.asarray(docs, dtype=np.int64)][doc_a] + pos_a
    target = captured.tokens[flat + 1].astype(np.int64)
    pred = captured.pred[flat].astype(np.int64)
    if (pred < 0).any():
        raise ValueError("recall_table: an answer position was not captured")
    ok = (pred == target).astype(np.float64)
    everything = np.ones(ok.shape[0], dtype=np.bool_)
    bucket = np.digitize(dist_a, _EDGES[1:]).astype(np.int64)
    back = pos_a // window - (pos_a - dist_a) // window
    out: dict[str, Any] = {
        "kind": "recall" if family in RECALL_FAMILIES else "answers",
        "family": family,
        "n_docs": len(examples),
        "n": int(ok.shape[0]),
        "accuracy": float(ok.mean()) if ok.shape[0] else None,
        "loss": float(captured.nll[flat].astype(np.float64).mean()) if ok.shape[0] else None,
        "by_distance": _by_bucket(ok, bucket, DISTANCE_KEYS, everything),
        "by_windows_back": {str(b): _cell(ok, back == b) for b in np.unique(back).tolist()},
    }
    if family not in RECALL_FAMILIES or not ok.shape[0]:
        return out
    out["by_query"] = {"first": _cell(ok, slot_a == 0), "later": _cell(ok, slot_a > 0)}
    fact_bucket = np.digitize(fact_a, _COARSE_EDGES[1:]).astype(np.int64)
    coarse = np.digitize(dist_a, _COARSE_EDGES[1:]).astype(np.int64)
    grid: dict[str, Any] = {}
    for i, key in enumerate(_FACT_KEYS):
        row = _by_bucket(ok, coarse, _COARSE_DISTANCE_KEYS, fact_bucket == i)
        if row:
            grid[key] = row
    out["by_fact_position"] = grid
    if family in _NONE_FAMILIES:
        is_none = target == NONE
        out["by_target"] = {"value": _cell(ok, ~is_none), "none": _cell(ok, is_none)}
        out["predicts_none"] = float((pred == NONE).mean())
        out["by_distance_value_targets"] = _by_bucket(ok, bucket, DISTANCE_KEYS, ~is_none)
    if family in _OWN_VALUE_FAMILIES:
        span = values if values is not None else value_range(examples)
        out["own_values"] = _own_values(examples, doc_a, flat, target, pred, bucket, captured, span)
    return out


def _own_values(
    examples: Sequence[Example],
    doc_a: Int,
    flat: Int,
    target: Int,
    pred: Int,
    bucket: Int,
    captured: Captured,
    values: tuple[int, int] | None,
) -> dict[str, Any]:
    """``delayed`` families: the prediction restricted to the values the document states (its
    bytes between ``BOS`` and the first ``SEP`` inside the value alphabet ``values``)."""
    row_at = captured.row_at[flat]
    if (row_at < 0).any() or values is None:
        return {"skipped": "the pass kept no log-probability rows for these answers"}
    alphabet = np.zeros(captured.rows.shape[1], dtype=np.bool_)
    alphabet[values[0] : values[1] + 1] = True
    right = np.zeros(target.shape[0], dtype=np.float64)
    chance = np.zeros_like(right)
    inside = np.zeros_like(right)
    mass = np.zeros_like(right)
    start = 0
    for local, example in enumerate(examples):
        n = int(example.loss_mask.sum())
        if n == 0:
            continue
        assert int(doc_a[start]) == local
        separators = np.flatnonzero(example.tokens == SEP)
        end = int(separators[0]) if separators.shape[0] else example.length
        before = example.tokens[1:end]
        stated = np.unique(before[alphabet[before]]).astype(np.int64)
        span = slice(start, start + n)
        start += n
        if stated.shape[0] == 0:
            continue
        rows = captured.rows[row_at[span]][:, stated].astype(np.float64)
        right[span] = stated[rows.argmax(axis=1)] == target[span]
        chance[span] = 1.0 / stated.shape[0]
        inside[span] = np.isin(pred[span], stated)
        mass[span] = np.exp(rows).sum(axis=1)
    everything = np.ones(right.shape[0], dtype=np.bool_)
    return {
        "alphabet": [int(values[0]), int(values[1])],
        "accuracy": float(right.mean()),
        "chance": float(chance.mean()),
        "predicted_share": float(inside.mean()),
        "mass": float(mass.mean()),
        "by_distance": {
            key: cell["accuracy"]
            for key, cell in _by_bucket(right, bucket, DISTANCE_KEYS, everything).items()
        },
    }
