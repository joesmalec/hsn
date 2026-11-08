"""The 5(c) edit probe of the reads-state check (P1.10 audit, ``scripts/eff_reads_state.py`` at
``f8884ec``), batched.

In every dev ``delayed/in_range`` document, the value of the first fact (the first ``k v`` pair with
``k`` a key and ``v`` a value that is not preceded by ``QRY``) is replaced by the next value symbol
after it, cyclically over the 48, that is absent from the document. The argmax at the answer
position ``p`` of the first query for that fact's key (the first loss position with
``tokens[p] == k`` and ``tokens[p - 1] == QRY``) is read before and after the edit, from a fresh
state, one window, evaluation mode, no gradients, the run's autocast; ``context`` is the example's
own. A document without that query is skipped, not counted. Only documents whose meta says
``task == delayed`` and ``variant == in_range`` are probed (kv documents use the same key and value
alphabets and would otherwise match).

Outcomes partition the probes (:func:`outcome_counts`): an answer *follows the edit* only when it
changed and equals the new value (the per-probe ``follows_new_value`` stays the raw ``after == new
value``). The verdict (:func:`verdict`, fixed in ``POST_A1_CHECKS.md`` before HSN's 5(c) ran) is
computed over the probes answered correctly before the edit.

**Batching (P1.11).** :func:`edit_probe` sorts the probed documents by ``p``, puts the original and
the edited document of each in the same batch, truncates every row after its own ``p`` (logits at
``p`` depend on tokens ``0..p`` only, and the edit sits before ``p``), pads after it, gives every
row its own fresh state with a reset in column 0 only, and reads the argmax at each row's own
``p``. :func:`edit_probe_reference` is the audit's loop (one full document per forward, B = 1),
kept for the tests and ``probe-check``. Batching changes GEMM row counts, so results agree with the
reference exactly in fp64 and to about one ulp in fp32 / bf16; a near-tie (top-2 margin of the
order of the rounding) can flip, which ``probe-check`` reports with its margin. Both probes raise
when no document is probed (P1.14).

**Probe v2, the vocabulary-matched edit (P1.14; protocol ``reads_state`` version 2).** At the old
recipe dense HSN answered with 10 to 18 of the 48 values, so v1's new value was in the model's
answer vocabulary for 0 to 11 % of the correct documents and "follows the edit" was near 0 by
construction. v2 writes a value the model can say:

* the *answer vocabulary* (:func:`answer_vocabulary`) is the set of value symbols among v1's
  first-query answers on the original documents (every probed dev document; a value counts once it
  is answered once); answers that are not value symbols are counted and left out;
* per document the candidates are the vocabulary minus the document's own symbols (the old value is
  in the document, so never a candidate); the new value is the candidate at index
  ``blake2b("hsn-probe-v2:<PROBE_V2_SEED>:<doc id>") mod len(candidates)`` (:func:`matched_value`):
  seeded, reproducible across machines and library versions, independent of the other documents.
  A document without a candidate is *unmatched* and counted;
* :func:`edit_probe_v2` runs after v1 with v1's items, order and chunking: row ``2r`` is the
  original, row ``2r + 1`` the v2 edit (an unmatched document keeps v1's edit there, its answer
  discarded), so every chunk's shape and the original rows' inputs are v1's, and their answers
  should equal v1's (``original_mismatches``, expected 0, is reported, never hidden);
* :func:`summarize_v2` reports follow / other / unchanged as v1 over the matched documents (and over
  the matched documents answered correctly before the edit), the vocabulary and its size, the
  share of documents with a matched edit and the v2 verdict (:func:`verdict_v2`: v1's thresholds,
  "no matched probes" at 0 correct matched documents, "too few matched probes" below
  :data:`PROBE_V2_MIN_CORRECT`).
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import QRY, DelayedParams
from hsn.device import DeviceContext
from hsn.train.amp import autocast_ctx

__all__ = [
    "KEYS",
    "PROBE_GROUP",
    "PROBE_V2_MIN_CORRECT",
    "PROBE_V2_SEED",
    "QRY",
    "VALUES",
    "ProbeItem",
    "absent_value",
    "answer_vocabulary",
    "edit_probe",
    "edit_probe_reference",
    "edit_probe_v2",
    "edit_probe_v2_reference",
    "first_fact_and_query",
    "matched_value",
    "outcome_counts",
    "probe_items",
    "summarize",
    "summarize_v2",
    "verdict",
    "verdict_v2",
]

_DELAYED = DelayedParams()
KEYS = range(
    min(lo for lo, _ in _DELAYED.keys.values()), max(hi for _, hi in _DELAYED.keys.values())
)
VALUES = range(*_DELAYED.values)
PROBE_GROUP = ("delayed", "in_range")
NO_CONTEXT = 2  # the audit's context when an example carries none
#: P1.14: the salt of probe v2's draw of the new value (module docstring)
PROBE_V2_SEED = 0
#: P1.14: below this many correct matched documents the v2 verdict is "too few matched probes"
PROBE_V2_MIN_CORRECT = 20


def first_fact_and_query(tokens: np.ndarray, loss_mask: np.ndarray) -> tuple[int, int, int] | None:
    """(value position of the first fact, its key, the answer position querying that key)."""
    for i in range(len(tokens) - 1):
        k, v = int(tokens[i]), int(tokens[i + 1])
        if k in KEYS and v in VALUES and (i == 0 or int(tokens[i - 1]) != QRY):
            for flat in np.flatnonzero(loss_mask):
                p = int(flat)
                if int(tokens[p]) == k and p > 0 and int(tokens[p - 1]) == QRY:
                    return i + 1, k, p
            return None
    return None


def absent_value(old: int, present: set[int]) -> int:
    """The next value symbol after ``old`` (cyclically over the 48) absent from the document."""
    for step in range(1, len(VALUES) + 1):
        v = (old - VALUES.start + step) % len(VALUES) + VALUES.start
        if v not in present:
            return v
    raise ValueError("every value symbol occurs in the document")


def outcome_counts(probes: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Partition probes into follow / other / unchanged. A probe follows only when its answer
    changed: a model that gives one constant answer would otherwise count every probe whose new
    value equals that answer as both following the edit and unchanged."""
    follow = sum(bool(p["changed"] and p["follows_new_value"]) for p in probes)
    other = sum(bool(p["changed"] and not p["follows_new_value"]) for p in probes)
    unchanged = sum(not p["changed"] for p in probes)
    if follow + other + unchanged != len(probes):
        raise AssertionError("outcome counts do not partition the probes")
    return {"follow": follow, "other": other, "unchanged": unchanged}


def verdict(follow: int, unchanged: int, correct: int) -> str:
    """The reads-state verdict over the ``correct`` probes answered correctly before the edit."""
    if correct == 0:
        return "no correct answers to probe"
    f, u = follow / correct, unchanged / correct
    if f >= 0.5 and u <= 0.1:
        return "reads state"
    if f < 0.2:
        return "does not read state"
    return "partial"


@dataclass(frozen=True)
class ProbeItem:
    """One probed document: its dev index and id, the fact and query found, both token rows."""

    index: int
    doc: Any
    key: int
    answer_pos: int
    old_value: int
    new_value: int
    tokens: np.ndarray
    modified: np.ndarray
    context: np.ndarray


def probe_items(dev: ByteDataset) -> list[ProbeItem]:
    """The probed documents of ``dev`` in dev order (module docstring)."""
    items: list[ProbeItem] = []
    for i in range(len(dev)):
        meta = dev.meta(i)
        if (meta.get("task"), meta.get("variant")) != PROBE_GROUP:
            continue
        ex = dev.examples[i]
        tokens = np.asarray(ex.tokens, dtype=np.int64)
        lm = np.asarray(ex.loss_mask, dtype=bool)
        context = (
            np.asarray(ex.context, dtype=np.int64)
            if getattr(ex, "context", None) is not None
            else np.full_like(tokens, NO_CONTEXT)
        )
        found = first_fact_and_query(tokens, lm)
        if found is None:
            continue
        vpos, key, p = found
        old_value = int(tokens[vpos])
        new_value = absent_value(old_value, set(tokens.tolist()))
        modified = tokens.copy()
        modified[vpos] = new_value
        items.append(
            ProbeItem(
                i, meta.get("doc_id"), key, p, old_value, new_value, tokens, modified, context
            )
        )
    return items


def _probed(dev: ByteDataset) -> list[ProbeItem]:
    """:func:`probe_items`, raising when there is none (P1.14: a probe of nothing is an error, not a
    verdict)."""
    items = probe_items(dev)
    if not items:
        raise ValueError(
            f"edit probe: no document probed among {len(dev)} dev documents (none is a "
            f"{PROBE_GROUP[0]}/{PROBE_GROUP[1]} document with a first-fact query)"
        )
    return items


def _record(item: ProbeItem, a0: int, a1: int) -> dict[str, Any]:
    return {
        "doc": item.doc,
        "key": item.key,
        "answer_pos": item.answer_pos,
        "old_value": item.old_value,
        "new_value": item.new_value,
        "argmax_original": a0,
        "argmax_modified": a1,
        "changed": a0 != a1,
        "follows_new_value": a1 == item.new_value,
        "original_correct": a0 == item.old_value,
    }


def _forward(model: nn.Module, tokens: Tensor, context: Tensor, reset: Tensor) -> Tensor:
    from hsn.model.hsn import HSN  # local: keeps this module importable for the baselines' tests

    B = int(tokens.shape[0])
    state = model.init_state(B, tokens.device)  # type: ignore[operator]
    kwargs: dict[str, Any] = {"reset_columns": [0]} if isinstance(model, HSN) else {}
    logits, _state, _info = model(tokens, context, state, reset, **kwargs)
    out: Tensor = logits
    return out


def _top(logits: Tensor) -> tuple[int, float]:
    """The argmax of a ``[V]`` row and its top-2 margin (fp32)."""
    row = logits.float()
    top = torch.topk(row, 2)
    return int(top.indices[0]), float(top.values[0] - top.values[1])


def _check_model(model: nn.Module, items: Sequence[ProbeItem]) -> None:
    from hsn.baselines.transformer import TransformerBaseline
    from hsn.model.hsn import HSN

    if isinstance(model, TransformerBaseline):
        too_long = [it.doc for it in items if it.answer_pos + 1 > model.window]
        if too_long:
            raise ValueError(
                f"edit probe: answer positions beyond the Transformer's {model.window}-position "
                f"window in documents {too_long[:5]}"
            )
    if isinstance(model, HSN) and model.cfg.routing == "random":
        raise ValueError(
            "edit probe: routing=random draws batch-shaped random numbers in evaluation, so a "
            "batched probe would not match the per-document one"
        )


@torch.no_grad()
def edit_probe(
    model: nn.Module,
    ctx: DeviceContext,
    dev: ByteDataset,
    *,
    batch_rows: int = 64,
    margins: bool = False,
) -> list[dict[str, Any]]:
    """The batched edit probe (module docstring); one record per probed document, in dev order.
    ``margins`` adds the top-2 logit margins before and after the edit."""
    if batch_rows < 2 or batch_rows % 2:
        raise ValueError(f"batch_rows must be an even number >= 2, got {batch_rows}")
    items = _probed(dev)
    _check_model(model, items)
    order = sorted(range(len(items)), key=lambda j: (items[j].answer_pos, j))
    per_chunk = batch_rows // 2
    found: dict[int, tuple[int, float, int, float]] = {}
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(order), per_chunk):
            chunk = order[start : start + per_chunk]
            T = max(items[j].answer_pos for j in chunk) + 1
            B = 2 * len(chunk)
            tok = np.zeros((B, T), dtype=np.int64)
            ctxt = np.full((B, T), NO_CONTEXT, dtype=np.int64)
            for r, j in enumerate(chunk):
                it = items[j]
                n = it.answer_pos + 1
                tok[2 * r, :n] = it.tokens[:n]
                tok[2 * r + 1, :n] = it.modified[:n]
                ctxt[2 * r, :n] = it.context[:n]
                ctxt[2 * r + 1, :n] = it.context[:n]
            tokens = torch.as_tensor(tok, device=ctx.device)
            context = torch.as_tensor(ctxt, device=ctx.device)
            reset = torch.zeros((B, T), dtype=torch.bool, device=ctx.device)
            reset[:, 0] = True
            with autocast_ctx(ctx):
                logits = _forward(model, tokens, context, reset)
            rows = torch.tensor([items[j].answer_pos for j in chunk], device=ctx.device)
            picked = logits[torch.arange(B, device=ctx.device), rows.repeat_interleave(2)]
            picked = picked.float().cpu()
            for r, j in enumerate(chunk):
                a0, m0 = _top(picked[2 * r])
                a1, m1 = _top(picked[2 * r + 1])
                found[j] = (a0, m0, a1, m1)
    finally:
        model.train(was_training)
    out: list[dict[str, Any]] = []
    for j, it in enumerate(items):
        a0, m0, a1, m1 = found[j]
        record = _record(it, a0, a1)
        if margins:
            record["margin_original"], record["margin_modified"] = m0, m1
        out.append(record)
    return out


@torch.no_grad()
def edit_probe_reference(
    model: nn.Module, ctx: DeviceContext, dev: ByteDataset, *, margins: bool = False
) -> list[dict[str, Any]]:
    """The audit's probe: for each probed document one forward of the whole document at B = 1,
    original then edited (``scripts/eff_reads_state.py`` at ``f8884ec``)."""
    items = _probed(dev)
    _check_model(model, items)
    was_training = model.training
    model.eval()

    def argmax_at(tokens: np.ndarray, context: np.ndarray, p: int) -> tuple[int, float]:
        tok = torch.as_tensor(tokens, dtype=torch.int64, device=ctx.device)[None]
        ctxt = torch.as_tensor(context, dtype=torch.int64, device=ctx.device)[None]
        reset = torch.zeros_like(tok, dtype=torch.bool)
        reset[:, 0] = True
        with autocast_ctx(ctx):
            state = model.init_state(1, ctx.device)  # type: ignore[operator]
            logits, _state, _info = model(tok, ctxt, state, reset)
        return _top(logits[0, p].cpu())

    out: list[dict[str, Any]] = []
    try:
        for it in items:
            a0, m0 = argmax_at(it.tokens, it.context, it.answer_pos)
            a1, m1 = argmax_at(it.modified, it.context, it.answer_pos)
            record = _record(it, a0, a1)
            if margins:
                record["margin_original"], record["margin_modified"] = m0, m1
            out.append(record)
    finally:
        model.train(was_training)
    return out


def summarize(probes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The probe fields of the audit's result JSON, plus flat counts for the rules."""
    correct = [p for p in probes if p["original_correct"]]
    c_counts = outcome_counts(correct)
    n = len(correct)
    return {
        "n_probed": len(probes),
        "first_query_accuracy": n / len(probes) if probes else None,
        "all": outcome_counts(probes),
        "correct_before_edit": {"n": n, **c_counts},
        "verdict": verdict(c_counts["follow"], c_counts["unchanged"], n),
        "first_query_answer_histogram": dict(
            collections.Counter(p["argmax_original"] for p in probes).most_common()
        ),
        "distinct_first_query_answers": len({p["argmax_original"] for p in probes}),
        "correct_n": n,
        "correct_follow": c_counts["follow"],
        "correct_other": c_counts["other"],
        "correct_unchanged": c_counts["unchanged"],
        "correct_follow_frac": c_counts["follow"] / n if n else None,
        "correct_unchanged_frac": c_counts["unchanged"] / n if n else None,
    }


# --------------------------------------------------------------------------------------------------
# Probe v2: the vocabulary-matched edit (P1.14, module docstring)
# --------------------------------------------------------------------------------------------------


def answer_vocabulary(v1: Sequence[dict[str, Any]]) -> tuple[list[int], int]:
    """The model's answer vocabulary from v1's probes: the value symbols among the original
    documents' first-query answers (sorted), and how many answers were not value symbols."""
    answers = [int(p["argmax_original"]) for p in v1]
    return sorted({a for a in answers if a in VALUES}), sum(a not in VALUES for a in answers)


def matched_value(
    item: ProbeItem, vocabulary: Sequence[int], *, seed: int = PROBE_V2_SEED
) -> tuple[int | None, int]:
    """``(new value, number of candidates)`` of v2's edit of ``item``: a value of the vocabulary
    absent from the document, drawn by a hash of the document id; ``(None, 0)`` when every value of
    the vocabulary occurs in the document."""
    present = {int(t) for t in item.tokens.tolist()}
    candidates = [int(v) for v in sorted(vocabulary) if int(v) not in present]
    if not candidates:
        return None, 0
    key = item.doc if item.doc is not None else f"index {item.index}"
    digest = hashlib.blake2b(f"hsn-probe-v2:{seed}:{key}".encode(), digest_size=8).digest()
    return candidates[int.from_bytes(digest, "big") % len(candidates)], len(candidates)


@dataclass(frozen=True)
class _V2Item:
    item: ProbeItem  # v1's item, with the v2 edit in ``modified`` and ``new_value`` when matched
    matched: bool
    candidates: int


def _v2_items(dev: ByteDataset, v1: Sequence[dict[str, Any]]) -> list[_V2Item]:
    items = _probed(dev)
    if [it.doc for it in items] != [p["doc"] for p in v1]:
        raise ValueError("edit probe v2: v1's probes are not this dev set's (other documents)")
    vocabulary, _outside = answer_vocabulary(v1)
    out = []
    for it in items:
        new, n = matched_value(it, vocabulary)
        if new is None:
            out.append(_V2Item(it, False, 0))
            continue
        vpos = int(np.flatnonzero(it.tokens != it.modified)[0])  # v1's edit: the first fact's value
        modified = it.tokens.copy()
        modified[vpos] = new
        out.append(_V2Item(dataclasses.replace(it, new_value=new, modified=modified), True, n))
    return out


def _v2_record(
    v2: _V2Item, a0: int, a1: int, v1: dict[str, Any], margins: tuple[float, float] | None
) -> dict[str, Any]:
    it = v2.item
    record: dict[str, Any] = {
        "doc": it.doc,
        "key": it.key,
        "answer_pos": it.answer_pos,
        "old_value": it.old_value,
        "matched": v2.matched,
        "candidates": v2.candidates,
        "new_value": it.new_value if v2.matched else None,
        "argmax_original": a0,
        "argmax_modified": a1 if v2.matched else None,
        "changed": (a0 != a1) if v2.matched else None,
        "follows_new_value": (a1 == it.new_value) if v2.matched else None,
        "original_correct": a0 == it.old_value,
        "original_matches_v1": a0 == int(v1["argmax_original"]),
    }
    if margins is not None:
        record["margin_original"], record["margin_modified"] = margins
    return record


@torch.no_grad()
def edit_probe_v2(
    model: nn.Module,
    ctx: DeviceContext,
    dev: ByteDataset,
    *,
    v1: Sequence[dict[str, Any]],
    batch_rows: int = 64,
    margins: bool = False,
) -> list[dict[str, Any]]:
    """The batched v2 probe (module docstring) after v1's probes ``v1`` of the same model and dev
    set; one record per probed document, in dev order. Its batching is :func:`edit_probe`'s, row
    for row."""
    if batch_rows < 2 or batch_rows % 2:
        raise ValueError(f"batch_rows must be an even number >= 2, got {batch_rows}")
    v2 = _v2_items(dev, v1)
    items = [x.item for x in v2]
    _check_model(model, items)
    order = sorted(range(len(items)), key=lambda j: (items[j].answer_pos, j))
    per_chunk = batch_rows // 2
    found: dict[int, tuple[int, float, int, float]] = {}
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(order), per_chunk):
            chunk = order[start : start + per_chunk]
            T = max(items[j].answer_pos for j in chunk) + 1
            B = 2 * len(chunk)
            tok = np.zeros((B, T), dtype=np.int64)
            ctxt = np.full((B, T), NO_CONTEXT, dtype=np.int64)
            for r, j in enumerate(chunk):
                it = items[j]
                n = it.answer_pos + 1
                tok[2 * r, :n] = it.tokens[:n]
                tok[2 * r + 1, :n] = it.modified[:n]
                ctxt[2 * r, :n] = it.context[:n]
                ctxt[2 * r + 1, :n] = it.context[:n]
            tokens = torch.as_tensor(tok, device=ctx.device)
            context = torch.as_tensor(ctxt, device=ctx.device)
            reset = torch.zeros((B, T), dtype=torch.bool, device=ctx.device)
            reset[:, 0] = True
            with autocast_ctx(ctx):
                logits = _forward(model, tokens, context, reset)
            rows = torch.tensor([items[j].answer_pos for j in chunk], device=ctx.device)
            picked = logits[torch.arange(B, device=ctx.device), rows.repeat_interleave(2)]
            picked = picked.float().cpu()
            for r, j in enumerate(chunk):
                a0, m0 = _top(picked[2 * r])
                a1, m1 = _top(picked[2 * r + 1])
                found[j] = (a0, m0, a1, m1)
    finally:
        model.train(was_training)
    out = []
    for j, x in enumerate(v2):
        a0, m0, a1, m1 = found[j]
        out.append(_v2_record(x, a0, a1, v1[j], (m0, m1) if margins else None))
    return out


@torch.no_grad()
def edit_probe_v2_reference(
    model: nn.Module,
    ctx: DeviceContext,
    dev: ByteDataset,
    *,
    v1: Sequence[dict[str, Any]],
    margins: bool = False,
) -> list[dict[str, Any]]:
    """v2 as the audit's loop: one full document per forward at B = 1, original then v2 edit."""
    v2 = _v2_items(dev, v1)
    _check_model(model, [x.item for x in v2])
    was_training = model.training
    model.eval()

    def argmax_at(tokens: np.ndarray, context: np.ndarray, p: int) -> tuple[int, float]:
        tok = torch.as_tensor(tokens, dtype=torch.int64, device=ctx.device)[None]
        ctxt = torch.as_tensor(context, dtype=torch.int64, device=ctx.device)[None]
        reset = torch.zeros_like(tok, dtype=torch.bool)
        reset[:, 0] = True
        with autocast_ctx(ctx):
            state = model.init_state(1, ctx.device)  # type: ignore[operator]
            logits, _state, _info = model(tok, ctxt, state, reset)
        return _top(logits[0, p].cpu())

    out = []
    try:
        for j, x in enumerate(v2):
            it = x.item
            a0, m0 = argmax_at(it.tokens, it.context, it.answer_pos)
            a1, m1 = argmax_at(it.modified, it.context, it.answer_pos)
            out.append(_v2_record(x, a0, a1, v1[j], (m0, m1) if margins else None))
    finally:
        model.train(was_training)
    return out


def verdict_v2(
    follow: int, unchanged: int, correct: int, *, min_correct: int = PROBE_V2_MIN_CORRECT
) -> str:
    """The v2 verdict over the ``correct`` matched documents answered correctly before the edit:
    v1's thresholds (:func:`verdict`), after "no matched probes" (0) and "too few matched probes"
    (below ``min_correct``)."""
    if correct == 0:
        return "no matched probes"
    if correct < min_correct:
        return "too few matched probes"
    return verdict(follow, unchanged, correct)


def summarize_v2(
    probes: Sequence[dict[str, Any]], *, v1: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """The ``probe_v2`` block: v1's counts over the matched documents, the vocabulary, the share of
    documents with a matched edit, the mismatches with v1's original answers and the verdict."""
    vocabulary, outside = answer_vocabulary(v1)
    matched = [p for p in probes if p["matched"]]
    correct = [p for p in matched if p["original_correct"]]
    c_counts = outcome_counts(correct)
    n = len(correct)
    return {
        "n_probed": len(probes),
        "vocab": vocabulary,
        "vocab_size": len(vocabulary),
        "answers_outside_values": outside,
        "matched_n": len(matched),
        "matched_frac": len(matched) / len(probes) if probes else None,
        "correct_unmatched_n": sum(
            bool(p["original_correct"] and not p["matched"]) for p in probes
        ),
        "all": outcome_counts(matched),
        "correct_before_edit": {"n": n, **c_counts},
        "original_mismatches": sum(not p["original_matches_v1"] for p in probes),
        "verdict": verdict_v2(c_counts["follow"], c_counts["unchanged"], n),
        "correct_n": n,
        "correct_follow": c_counts["follow"],
        "correct_other": c_counts["other"],
        "correct_unchanged": c_counts["unchanged"],
        "correct_follow_frac": c_counts["follow"] / n if n else None,
        "correct_unchanged_frac": c_counts["unchanged"] / n if n else None,
    }
