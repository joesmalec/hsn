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

**Probe v3, key-addressed edits (P1.15; protocol ``reads_state`` version 3).** v1 and v2 edit and
score the first query, which always asks the first fact: answerable by position alone, so "reads
state" there confirms positional recall, not key-addressed retrieval. v3 probes every *later*
query ``QRY k v`` (slots 1 to 3) of every dev delayed/in_range document (:func:`delayed_docs`
parses and checks them), with two edits, each on its own copy of the document:

* **the value edit** (the user's specification): the value of fact ``k`` becomes a value of the
  model's answer vocabulary (the value symbols it answers at any query of the original documents,
  in v3's own pass) absent from the document, drawn by ``blake2b("hsn-probe-v3:value:<seed>:<doc
  id>:<slot>")``; it *follows* when the answer at that query changes to the new value. It says
  whether the later answer is read from the state;
* **the key swap** (decision D1 of the P1.15 plan): the keys of fact ``k`` and of a partner fact
  ``j`` (not ``k``, not asked before this slot, another value; drawn the same way with tag
  ``swap``) trade places; it *follows* when the answer changes to ``v_j``. A reader that binds keys
  follows; a key-blind reader (positions and exclusion only) is unchanged, since no position or
  value moved. It says whether the read is addressed by key.

Every probe records its slot and its key-blind ambiguity: the number of facts a key-blind reader may
assign to the slot (:func:`eligible_facts`, the generator's rule and the audit's
``ceiling_check.py``; 1 means position alone answers it). :func:`summarize_v3` reports follow /
other / unchanged over the later queries answered correctly before the edit, overall, by slot and by
ambiguity, for both edits, with the vocabulary and the probe pass's own first/later accuracy;
:func:`verdict_v3_value` ("no correct later queries", "vocabulary too small" below
:data:`PROBE_V3_MIN_VOCAB`, "too few correct later queries" below :data:`PROBE_V3_MIN_CORRECT`, then
v1's thresholds labelled "... (later queries)") and :func:`verdict_v3_swap` ("no swap probes", "too
few swap probes", "binds keys", "does not bind keys", "partial", with v1's thresholds). Batching as
v1 (rows truncated after their last scored position, fresh states, chunks by length;
:func:`edit_probe_v3` with ``reference=True`` is the per-row B = 1 loop). The probe never changes
the dev documents.
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
    "PROBE_V3_MIN_CORRECT",
    "PROBE_V3_MIN_VOCAB",
    "PROBE_V3_SEED",
    "DelayedDoc",
    "V3Item",
    "delayed_docs",
    "edit_probe_v3",
    "eligible_facts",
    "parse_delayed",
    "summarize_v3",
    "v3_items",
    "verdict_v3_swap",
    "verdict_v3_value",
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


# --------------------------------------------------------------------------------------------------
# Probe v3: key-addressed edits of the later queries (P1.15, module docstring)
# --------------------------------------------------------------------------------------------------

#: P1.15: the salt of probe v3's draws (the new value and the swap partner)
PROBE_V3_SEED = 0
#: P1.15: below this many correct probes a v3 verdict is "too few ..."
PROBE_V3_MIN_CORRECT = 20
#: P1.15: the value-edit verdict's threshold labels end with this, so that "reads state" over the
#: later queries is never read as v1's first-query verdict or as key binding (the swap's)
LATER_SUFFIX = " (later queries)"
#: P1.15: below this answer vocabulary the value-edit verdict is "vocabulary too small" (the
#: overnight reading note of 8 Nov: a 4- or 5-value vocabulary makes "follows the edit" cheap)
PROBE_V3_MIN_VOCAB = 20


@dataclass(frozen=True)
class DelayedDoc:
    """A parsed delayed/in_range dev document: its facts ``(key, value, value position)`` in order
    and its queries ``(slot, fact index, answer position)`` in order (slot 0 asks fact 0)."""

    index: int
    doc: Any
    tokens: np.ndarray
    context: np.ndarray
    facts: tuple[tuple[int, int, int], ...]
    queries: tuple[tuple[int, int, int], ...]


def parse_delayed(tokens: np.ndarray, loss_mask: np.ndarray) -> tuple[list[Any], list[Any]]:
    """``(facts, queries)`` of a delayed document (:class:`DelayedDoc`). Raises when a loss position
    is not a ``QRY k`` key, a queried key is not a fact's, a key is stated twice, or the first query
    does not ask the first fact."""
    positions = [int(p) for p in np.flatnonzero(loss_mask)]
    if not positions:
        raise ValueError("no query")
    first_qry = positions[0] - 1
    facts: list[tuple[int, int, int]] = []
    for i in range(first_qry):
        k, v = int(tokens[i]), int(tokens[i + 1])
        if k in KEYS and v in VALUES and (i == 0 or int(tokens[i - 1]) != QRY):
            facts.append((k, v, i + 1))
    keys = [k for k, _, _ in facts]
    if len(set(keys)) != len(keys):
        raise ValueError(f"a key is stated twice: {keys}")
    queries: list[tuple[int, int, int]] = []
    for q, p in enumerate(positions):
        if p < 1 or int(tokens[p - 1]) != QRY or int(tokens[p]) not in keys:
            raise ValueError(f"loss position {p} does not ask a fact's key")
        queries.append((q, keys.index(int(tokens[p])), p))
    if queries[0][1] != 0:
        raise ValueError("the first query does not ask the first fact")
    return facts, queries


def eligible_facts(n_facts: int, slot: int, asked: Sequence[int]) -> list[int]:
    """The facts a key-blind reader may assign to later slot ``slot`` (the generator's rule,
    ``_Delayed.emit``; the audit's ``ceiling_check.py``): index ``w >= 1`` with ``2w >= 3 slot``,
    not asked in an earlier slot."""
    return [w for w in range(1, n_facts) if 2 * w >= 3 * slot and w not in asked]


def delayed_docs(dev: ByteDataset) -> list[DelayedDoc]:
    """The probed documents of ``dev`` (delayed/in_range, parsed), in dev order."""
    out: list[DelayedDoc] = []
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
        try:
            facts, queries = parse_delayed(tokens, lm)
        except ValueError as exc:
            raise ValueError(f"probe v3: dev document {meta.get('doc_id', i)}: {exc}") from exc
        out.append(DelayedDoc(i, meta.get("doc_id"), tokens, context, tuple(facts), tuple(queries)))
    if not out:
        raise ValueError(
            f"probe v3: no {PROBE_GROUP[0]}/{PROBE_GROUP[1]} document among {len(dev)}"
        )
    return out


def _draw(tag: str, doc: Any, slot: int, n: int, *, seed: int = PROBE_V3_SEED) -> int:
    digest = hashlib.blake2b(f"hsn-probe-v3:{tag}:{seed}:{doc}:{slot}".encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big") % n


@dataclass(frozen=True)
class V3Item:
    """One later query of a document and its two edits (module docstring, probe v3)."""

    doc: DelayedDoc
    slot: int
    which: int  # the asked fact's index
    answer_pos: int
    eligible: int  # facts a key-blind reader may assign to this slot (>= 1)
    value_new: int | None  # the value edit's new value (None: no vocabulary value absent)
    value_candidates: int
    value_tokens: np.ndarray | None
    swap_with: int | None  # the swap partner's fact index (None: no partner)
    swap_candidates: int
    swap_tokens: np.ndarray | None

    @property
    def key(self) -> int:
        return self.doc.facts[self.which][0]

    @property
    def old_value(self) -> int:
        return self.doc.facts[self.which][1]

    @property
    def swap_value(self) -> int | None:
        return None if self.swap_with is None else self.doc.facts[self.swap_with][1]


def v3_items(docs: Sequence[DelayedDoc], vocabulary: Sequence[int]) -> list[V3Item]:
    """Every later query of ``docs`` with its value edit (a vocabulary value absent from the
    document, drawn by hash) and its key swap (a partner fact other than the asked one, not asked
    before this slot, with another value, drawn by hash). The edits are copies."""
    items: list[V3Item] = []
    for d in docs:
        present = {int(t) for t in d.tokens.tolist()}
        candidates = [int(v) for v in sorted(vocabulary) if int(v) not in present]
        asked: list[int] = [d.queries[0][1]]
        for slot, which, p in d.queries[1:]:
            eligible = eligible_facts(len(d.facts), slot, asked)
            if which not in eligible:
                raise ValueError(
                    f"probe v3: document {d.doc} slot {slot} asks fact {which}, outside the "
                    f"generator's eligible facts {eligible}"
                )
            new: int | None = None
            value_tokens = None
            if candidates:
                new = candidates[_draw("value", d.doc, slot, len(candidates))]
                value_tokens = d.tokens.copy()
                value_tokens[d.facts[which][2]] = new
            old_value = d.facts[which][1]
            partners = [
                w
                for w in range(len(d.facts))
                if w != which and w not in asked and d.facts[w][1] != old_value
            ]
            swap_with: int | None = None
            swap_tokens = None
            if partners:
                swap_with = partners[_draw("swap", d.doc, slot, len(partners))]
                swap_tokens = d.tokens.copy()
                k_pos, j_pos = d.facts[which][2] - 1, d.facts[swap_with][2] - 1
                swap_tokens[k_pos], swap_tokens[j_pos] = d.tokens[j_pos], d.tokens[k_pos]
            items.append(
                V3Item(
                    d,
                    slot,
                    which,
                    p,
                    len(eligible),
                    new,
                    len(candidates),
                    value_tokens,
                    swap_with,
                    len(partners),
                    swap_tokens,
                )
            )
            asked.append(which)
    return items


def _rows_answers(
    model: nn.Module,
    ctx: DeviceContext,
    rows: Sequence[tuple[np.ndarray, np.ndarray, Sequence[int]]],
    *,
    batch_rows: int,
) -> list[list[tuple[int, float]]]:
    """``(argmax, top-2 margin)`` at each requested position of each row ``(tokens, context,
    positions)``: rows sorted by their last position, chunks of ``batch_rows``, every row truncated
    after its last position with its own fresh state (v1's batching)."""
    order = sorted(range(len(rows)), key=lambda j: (max(rows[j][2]), j))
    found: dict[int, list[tuple[int, float]]] = {}
    for start in range(0, len(order), batch_rows):
        chunk = order[start : start + batch_rows]
        T = max(max(rows[j][2]) for j in chunk) + 1
        B = len(chunk)
        tok = np.zeros((B, T), dtype=np.int64)
        ctxt = np.full((B, T), NO_CONTEXT, dtype=np.int64)
        for r, j in enumerate(chunk):
            tokens, context, positions = rows[j]
            n = max(positions) + 1
            tok[r, :n], ctxt[r, :n] = tokens[:n], context[:n]
        tokens_t = torch.as_tensor(tok, device=ctx.device)
        context_t = torch.as_tensor(ctxt, device=ctx.device)
        reset = torch.zeros((B, T), dtype=torch.bool, device=ctx.device)
        reset[:, 0] = True
        with autocast_ctx(ctx):
            logits = _forward(model, tokens_t, context_t, reset)
        picked = logits.float().cpu()
        for r, j in enumerate(chunk):
            found[j] = [_top(picked[r, p]) for p in rows[j][2]]
    return [found[j] for j in range(len(rows))]


def _rows_answers_reference(
    model: nn.Module,
    ctx: DeviceContext,
    rows: Sequence[tuple[np.ndarray, np.ndarray, Sequence[int]]],
) -> list[list[tuple[int, float]]]:
    """:func:`_rows_answers` as the audit's loop: one full row per forward at B = 1."""
    out = []
    for tokens, context, positions in rows:
        tok = torch.as_tensor(tokens, dtype=torch.int64, device=ctx.device)[None]
        ctxt = torch.as_tensor(context, dtype=torch.int64, device=ctx.device)[None]
        reset = torch.zeros_like(tok, dtype=torch.bool)
        reset[:, 0] = True
        with autocast_ctx(ctx):
            state = model.init_state(1, ctx.device)  # type: ignore[operator]
            logits, _state, _info = model(tok, ctxt, state, reset)
        row = logits[0].float().cpu()
        out.append([_top(row[p]) for p in positions])
    return out


def _v3_outcome(original: int, edited: int | None, target: int | None) -> dict[str, Any]:
    if edited is None or target is None:
        return {"argmax": None, "changed": None, "follows": None}
    return {"argmax": edited, "changed": original != edited, "follows": edited == target}


@torch.no_grad()
def edit_probe_v3(
    model: nn.Module,
    ctx: DeviceContext,
    dev: ByteDataset,
    *,
    batch_rows: int = 64,
    reference: bool = False,
    margins: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Probe v3 (module docstring): the original documents (every query answered from one row per
    document), then one value-edit row and one key-swap row per later query. ``reference`` runs
    the per-row B = 1 loop instead of the batched one. Returns ``(probes, originals)``: one record
    per later query and one per document (its answers and true values at every slot), in dev
    order."""
    if batch_rows < 1:
        raise ValueError(f"batch_rows must be >= 1, got {batch_rows}")
    docs = delayed_docs(dev)
    last = [
        ProbeItem(d.index, d.doc, 0, d.queries[-1][2], 0, 0, d.tokens, d.tokens, d.context)
        for d in docs
    ]
    _check_model(model, last)

    def answer(rows: Sequence[tuple[np.ndarray, np.ndarray, Sequence[int]]]) -> list[Any]:
        if reference:
            return _rows_answers_reference(model, ctx, rows)
        return _rows_answers(model, ctx, rows, batch_rows=batch_rows)

    was_training = model.training
    model.eval()
    try:
        originals = answer([(d.tokens, d.context, [p for _, _, p in d.queries]) for d in docs])
        answered = [a for row in originals for a, _m in row]
        vocabulary = sorted({a for a in answered if a in VALUES})
        items = v3_items(docs, vocabulary)
        by_doc = {d.index: originals[i] for i, d in enumerate(docs)}
        rows: list[tuple[np.ndarray, np.ndarray, Sequence[int]]] = []
        slots: list[tuple[int, str]] = []
        for n, it in enumerate(items):
            if it.value_tokens is not None:
                rows.append((it.value_tokens, it.doc.context, [it.answer_pos]))
                slots.append((n, "value"))
            if it.swap_tokens is not None:
                rows.append((it.swap_tokens, it.doc.context, [it.answer_pos]))
                slots.append((n, "swap"))
        edited = answer(rows) if rows else []
    finally:
        model.train(was_training)
    got: dict[tuple[int, str], tuple[int, float]] = {
        key: row[0] for key, row in zip(slots, edited, strict=True)
    }
    out: list[dict[str, Any]] = []
    for n, it in enumerate(items):
        a0, m0 = by_doc[it.doc.index][it.slot]
        value = got.get((n, "value"))
        swap = got.get((n, "swap"))
        record: dict[str, Any] = {
            "doc": it.doc.doc,
            "slot": it.slot,
            "key": it.key,
            "answer_pos": it.answer_pos,
            "old_value": it.old_value,
            "n_facts": len(it.doc.facts),
            "eligible": it.eligible,
            "argmax_original": a0,
            "original_correct": a0 == it.old_value,
            "vocab_size": len(vocabulary),
            "value_new": it.value_new,
            "value_candidates": it.value_candidates,
            **{
                f"value_{k}": v
                for k, v in _v3_outcome(
                    a0, None if value is None else value[0], it.value_new
                ).items()
            },
            "swap_key": None if it.swap_with is None else it.doc.facts[it.swap_with][0],
            "swap_value": it.swap_value,
            "swap_candidates": it.swap_candidates,
            **{
                f"swap_{k}": v
                for k, v in _v3_outcome(
                    a0, None if swap is None else swap[0], it.swap_value
                ).items()
            },
        }
        if margins:
            record["margin_original"] = m0
            record["margin_value"] = value[1] if value else None
            record["margin_swap"] = swap[1] if swap else None
        out.append(record)
    if not out:
        raise ValueError("probe v3: no later query among the probed documents")
    per_doc = [
        {
            "doc": d.doc,
            "answers": [a for a, _ in originals[i]],
            "values": [d.facts[w][1] for _, w, _ in d.queries],
        }
        for i, d in enumerate(docs)
    ]
    return out, per_doc


def _v3_counts(probes: Sequence[dict[str, Any]], edit: str) -> dict[str, Any]:
    """Follow / other / unchanged of one edit over ``probes`` (those with that edit)."""
    done = [p for p in probes if p[f"{edit}_argmax"] is not None]
    follow = sum(bool(p[f"{edit}_changed"] and p[f"{edit}_follows"]) for p in done)
    other = sum(bool(p[f"{edit}_changed"] and not p[f"{edit}_follows"]) for p in done)
    unchanged = sum(not p[f"{edit}_changed"] for p in done)
    n = len(done)
    return {
        "n": n,
        "follow": follow,
        "other": other,
        "unchanged": unchanged,
        "follow_frac": follow / n if n else None,
        "unchanged_frac": unchanged / n if n else None,
    }


def verdict_v3_value(
    follow: int,
    unchanged: int,
    correct: int,
    vocab_size: int,
    *,
    correct_later: int | None = None,
    min_correct: int = PROBE_V3_MIN_CORRECT,
    min_vocab: int = PROBE_V3_MIN_VOCAB,
) -> str:
    """The value-edit verdict over the ``correct`` later queries answered correctly before the
    edit and edited: "no correct later queries" (``correct_later``, the correct later queries with
    or without an edit, default ``correct``, is 0), "vocabulary too small", "too few correct later
    queries", then v1's thresholds, labelled for the later queries ("reads state (later queries)",
    "does not read state (later queries)", "partial (later queries)"): a later answer read from the
    state, which says nothing of key binding (that is the swap's verdict)."""
    if (correct if correct_later is None else correct_later) == 0:
        return "no correct later queries"
    if vocab_size < min_vocab:
        return "vocabulary too small"
    if correct < min_correct:
        return "too few correct later queries"
    return f"{verdict(follow, unchanged, correct)}{LATER_SUFFIX}"


def verdict_v3_swap(
    follow: int, unchanged: int, correct: int, *, min_correct: int = PROBE_V3_MIN_CORRECT
) -> str:
    """The key-swap verdict over the correct later queries with a partner: "no swap probes", "too
    few swap probes", then "binds keys" (follow >= 0.5, unchanged <= 0.1), "does not bind keys"
    (follow < 0.2), else "partial"."""
    if correct == 0:
        return "no swap probes"
    if correct < min_correct:
        return "too few swap probes"
    f, u = follow / correct, unchanged / correct
    if f >= 0.5 and u <= 0.1:
        return "binds keys"
    if f < 0.2:
        return "does not bind keys"
    return "partial"


def summarize_v3(
    probes: Sequence[dict[str, Any]], originals: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """The ``probe_v3`` block (module docstring) from :func:`edit_probe_v3`'s two lists: both
    edits' outcomes over the correct later queries, overall, by slot and by key-blind ambiguity;
    the vocabulary; the probe pass's own first/later accuracy; the verdicts; flat keys for the
    rules."""
    answers = [a for o in originals for a in o["answers"]]
    vocabulary = sorted({a for a in answers if a in VALUES})
    correct = [p for p in probes if p["original_correct"]]
    out: dict[str, Any] = {
        "n_documents": len(originals),
        "n_later_queries": len(probes),
        "vocab": vocabulary,
        "vocab_size": len(vocabulary),
        "answers_outside_values": sum(a not in VALUES for a in answers),
        "correct_later_n": len(correct),
    }
    slots: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for o in originals:
        for q, (a, v) in enumerate(zip(o["answers"], o["values"], strict=True)):
            name = "first" if q == 0 else f"later/{q}"
            slots[name][0] += int(a == v)
            slots[name][1] += 1
            if q:
                slots["later"][0] += int(a == v)
                slots["later"][1] += 1
    out["pass_accuracy"] = {k: (c / n if n else None) for k, (c, n) in sorted(slots.items())}
    for edit in ("value", "swap"):
        block = _v3_counts(correct, edit)
        block["by_slot"] = {
            str(q): _v3_counts([p for p in correct if p["slot"] == q], edit) for q in (1, 2, 3)
        }
        block["by_ambiguity"] = {
            "single": _v3_counts([p for p in correct if p["eligible"] == 1], edit),
            "multiple": _v3_counts([p for p in correct if p["eligible"] > 1], edit),
        }
        block["available_n"] = sum(p[f"{edit}_argmax"] is not None for p in probes)
        out[edit] = block
    v, s = out["value"], out["swap"]
    out["value"]["verdict"] = verdict_v3_value(
        v["follow"], v["unchanged"], v["n"], out["vocab_size"], correct_later=len(correct)
    )
    out["swap"]["verdict"] = verdict_v3_swap(s["follow"], s["unchanged"], s["n"])
    for edit in ("value", "swap"):  # flat keys for the rules (probe_v3:<edit>/<key>)
        b = out[edit]
        for key in ("follow_frac", "unchanged_frac", "verdict", "available_n"):
            out[f"{edit}/{key}"] = b[key]
        out[f"{edit}/correct_n"] = b["n"]
    return out
