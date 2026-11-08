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
"""

from __future__ import annotations

import collections
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
    "QRY",
    "VALUES",
    "ProbeItem",
    "absent_value",
    "edit_probe",
    "edit_probe_reference",
    "first_fact_and_query",
    "outcome_counts",
    "probe_items",
    "summarize",
    "verdict",
]

_DELAYED = DelayedParams()
KEYS = range(
    min(lo for lo, _ in _DELAYED.keys.values()), max(hi for _, hi in _DELAYED.keys.values())
)
VALUES = range(*_DELAYED.values)
PROBE_GROUP = ("delayed", "in_range")
NO_CONTEXT = 2  # the audit's context when an example carries none


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
