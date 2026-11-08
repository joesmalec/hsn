"""``route-check`` (P1.16): does a routed model's routing follow the key?

The delayed task asks a fact's value after a filler: ``... k v ... QRY k ?``. A model that binds
keys by routing would send a query to the cells its fact was written to: the cells selected at the
query key's position would overlap those selected at the asked fact's key position (where the key
is read) or value position (where the value is written) more than those of the document's other
facts. This module measures that on a stored checkpoint, read-only.

**The pass.** The arm's model at its latest checkpoint
(:func:`hsn.experiments.probe_check._restored`) runs every parsed delayed document (probe v3's
parser, :func:`hsn.experiments.probe.delayed_docs`)
from a fresh state, in evaluation mode (hard top-K, no exploration; refresh probes stay on, as in
every evaluation, and their rate is reported), batched ``batch_rows`` documents at a time, each row
padded after its end (the recurrence is causal, so padding never changes an earlier position). Per
position it reads the selected cells (``info["cell_idx"]`` with ``info["valid"]``, ``[B, T, K]``).

**Statistics**, per query (first queries and later queries reported apart, and together), with
``S(t)`` the set of cells selected at position ``t`` and ``K`` the selection size:

* ``o_key = |S(query) ∩ S(asked fact's key position)| / K``;
* ``o_val = |S(query) ∩ S(asked fact's value position)| / K``;
* ``o_other_key`` / ``o_other_val``: the same against the document's other facts, averaged;
* ``d_key = o_key − o_other_key`` and ``d_val = o_val − o_other_val`` (queries of a document with a
  single fact have no comparison and are left out), each with a 95 % bootstrap CI over documents
  (the unit; :func:`hsn.eval.bootstrap.bootstrap_mean` of the per-document means);
* **the key-swap control** (later queries with a swap partner, probe v3's swap document: the asked
  key and the partner's exchanged): ``o_swap_new`` against the key's **new** position and
  ``o_swap_old`` against its old one (which now holds the partner's key); ``d_swap = o_swap_new −
  o_swap_old``. Routing that follows the key follows it to the new position.

Under key-independent routing every ``d`` is 0 in expectation.

**Verdict** (diagnostic thresholds, not frozen metrics and not a gate; ``follow = 0.5 / K``, half a
cell): ``follows the key`` when the CI's lower bound of ``d_key`` or of ``d_val`` is at least
``follow`` and the swap control's mean ``d_swap`` is at least ``follow`` (or there are no swap
probes, which the output notes); ``weakly key-dependent`` when either CI excludes 0; else ``does
not follow the key``. A dense model selects every cell and is refused.

``split="train"`` runs the first ``n_docs`` delayed documents of the training split instead (keys
seen in training, in-sample; labelled), so that "routing follows trained keys" and "routing follows
keys never seen" can be told apart. Writes nothing; the CLI prints or saves the JSON.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hsn.experiments.db import Database

__all__ = ["doc_overlaps", "route_check", "route_follow", "summarize_route", "verdict_route"]

#: the CI level of the per-document bootstrap
ROUTE_ALPHA = 0.05
ROUTE_BOOT = 10_000


def _sets(cell_idx: np.ndarray, valid: np.ndarray) -> list[frozenset[int]]:
    """Per position, the set of valid selected cells (``cell_idx``, ``valid`` ``[T, K]``)."""
    return [
        frozenset(int(c) for c, v in zip(row, ok, strict=True) if v)
        for row, ok in zip(cell_idx, valid, strict=True)
    ]


def _overlap(a: frozenset[int], b: frozenset[int], k: int) -> float:
    return len(a & b) / k


def doc_overlaps(
    sets: Sequence[frozenset[int]],
    facts: Sequence[tuple[int, int, int]],
    queries: Sequence[tuple[int, int, int]],
    k: int,
) -> list[dict[str, Any]]:
    """The per-query overlaps (module docstring) of one document: ``sets[t]`` the cells selected at
    position ``t``, ``facts`` ``(key, value, value position)``, ``queries`` ``(slot, fact index,
    answer position)`` as :func:`hsn.experiments.probe.parse_delayed` gives them."""
    out: list[dict[str, Any]] = []
    for slot, which, p in queries:
        here = sets[p]
        key_pos, val_pos = facts[which][2] - 1, facts[which][2]
        others = [w for w in range(len(facts)) if w != which]
        entry: dict[str, Any] = {
            "slot": slot,
            "which": which,
            "o_key": _overlap(here, sets[key_pos], k),
            "o_val": _overlap(here, sets[val_pos], k),
        }
        if others:
            entry["o_other_key"] = float(
                np.mean([_overlap(here, sets[facts[w][2] - 1], k) for w in others])
            )
            entry["o_other_val"] = float(
                np.mean([_overlap(here, sets[facts[w][2]], k) for w in others])
            )
            entry["d_key"] = entry["o_key"] - entry["o_other_key"]
            entry["d_val"] = entry["o_val"] - entry["o_other_val"]
        out.append(entry)
    return out


def _ci(per_doc: Sequence[float]) -> dict[str, Any]:
    from hsn.eval.bootstrap import bootstrap_mean

    if len(per_doc) < 2:
        return {"n_docs": len(per_doc), "mean": float(np.mean(per_doc)) if per_doc else None}
    ci = bootstrap_mean(np.asarray(per_doc, dtype=np.float64), n_boot=ROUTE_BOOT, alpha=ROUTE_ALPHA)
    return {"n_docs": ci.n_units, "mean": ci.estimate, "low": ci.low, "high": ci.high}


def summarize_route(
    per_doc: Sequence[Sequence[dict[str, Any]]],
    swaps: Sequence[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    """The summary of :func:`route_follow`'s per-document query entries and swap entries: means of
    the overlaps, the per-document bootstrap of ``d_key`` and ``d_val`` (all, first, later
    queries), the swap control and :func:`verdict_route`."""
    out: dict[str, Any] = {"k": k, "follow": 0.5 / k}
    for tag, keep in (("all", None), ("first", True), ("later", False)):
        block: dict[str, Any] = {}
        entries = [e for doc in per_doc for e in doc if keep is None or (e["slot"] == 0) == keep]
        block["n_queries"] = len(entries)
        for name in ("o_key", "o_val", "o_other_key", "o_other_val"):
            values = [e[name] for e in entries if name in e]
            block[name] = float(np.mean(values)) if values else None
        for name in ("d_key", "d_val"):
            doc_means = []
            for doc in per_doc:
                values = [
                    e[name] for e in doc if name in e and (keep is None or (e["slot"] == 0) == keep)
                ]
                if values:
                    doc_means.append(float(np.mean(values)))
            block[name] = _ci(doc_means)
        out[tag] = block
    if swaps:
        out["swap"] = {
            "n": len(swaps),
            "o_swap_new": float(np.mean([s["o_swap_new"] for s in swaps])),
            "o_swap_old": float(np.mean([s["o_swap_old"] for s in swaps])),
            "d_swap": float(np.mean([s["o_swap_new"] - s["o_swap_old"] for s in swaps])),
        }
    else:
        out["swap"] = {"n": 0}
    out["verdict"] = verdict_route(out)
    return out


def verdict_route(summary: dict[str, Any]) -> str:
    """The diagnostic's verdict (module docstring) from a :func:`summarize_route` summary."""
    follow = float(summary["follow"])
    lows = [
        summary["all"][name].get("low")
        for name in ("d_key", "d_val")
        if summary["all"][name].get("low") is not None
    ]
    swap = summary.get("swap") or {}
    swap_ok = swap.get("n", 0) == 0 or float(swap["d_swap"]) >= follow
    if lows and max(lows) >= follow and swap_ok:
        return "follows the key"
    if lows and max(lows) > 0.0:
        return "weakly key-dependent"
    return "does not follow the key"


def _selected(
    model: Any, ctx: Any, rows: Sequence[tuple[np.ndarray, np.ndarray]], batch_rows: int
) -> tuple[list[list[frozenset[int]]], list[float]]:
    """Per row ``(tokens, context)``, the selected cell sets per position, and the refresh rates of
    the batches."""
    import torch

    from hsn.train.amp import autocast_ctx

    out: list[list[frozenset[int]]] = [[] for _ in rows]
    refresh: list[float] = []
    order = sorted(range(len(rows)), key=lambda j: (len(rows[j][0]), j))
    model.eval()
    for start in range(0, len(order), batch_rows):
        chunk = order[start : start + batch_rows]
        T = max(len(rows[j][0]) for j in chunk)
        tokens = np.zeros((len(chunk), T), dtype=np.int64)
        context = np.zeros((len(chunk), T), dtype=np.int64)
        for r, j in enumerate(chunk):
            n = len(rows[j][0])
            tokens[r, :n] = rows[j][0]
            context[r, :n] = rows[j][1]
            context[r, n:] = rows[j][1][-1]
        reset = np.zeros((len(chunk), T), dtype=np.bool_)
        reset[:, 0] = True
        device = ctx.device
        with torch.no_grad(), autocast_ctx(ctx):
            state = model.init_state(len(chunk), device)
            _logits, _state, info = model(
                torch.from_numpy(tokens).to(device),
                torch.from_numpy(context).to(device),
                state,
                torch.from_numpy(reset).to(device),
                reset_columns=[0],
            )
        cells = info["cell_idx"].cpu().numpy()
        valid = info["valid"].cpu().numpy()
        if "refreshed" in info:
            refresh.append(float(info["refreshed"].float().mean()))
        for r, j in enumerate(chunk):
            n = len(rows[j][0])
            out[j] = _sets(cells[r, :n], valid[r, :n])
    return out, refresh


def route_follow(
    model: Any, ctx: Any, docs: Sequence[Any], *, batch_rows: int = 64
) -> dict[str, Any]:
    """The diagnostic (module docstring) of ``model`` on parsed delayed documents ``docs``
    (:class:`hsn.experiments.probe.DelayedDoc`)."""
    from hsn.experiments.probe import v3_items
    from hsn.model.router import SCORED_MODES

    routing = getattr(getattr(model, "cfg", None), "routing", None)
    if routing not in SCORED_MODES:
        raise ValueError(
            f"route-check needs a routed model ({SCORED_MODES}), got routing {routing!r}: dense "
            "routing selects every cell"
        )
    k = int(model.cfg.k_select)
    t0 = time.perf_counter()
    sets, refresh = _selected(model, ctx, [(d.tokens, d.context) for d in docs], batch_rows)
    per_doc = [doc_overlaps(s, d.facts, d.queries, k) for s, d in zip(sets, docs, strict=True)]
    items = [it for it in v3_items(docs, []) if it.swap_tokens is not None]
    swap_sets, _ = _selected(
        model, ctx, [(it.swap_tokens, it.doc.context) for it in items], batch_rows
    )
    swaps = []
    for it, s in zip(items, swap_sets, strict=True):
        assert it.swap_with is not None
        p = it.answer_pos
        new_pos = it.doc.facts[it.swap_with][2] - 1  # where the asked key now sits
        old_pos = it.doc.facts[it.which][2] - 1  # now the partner's key
        swaps.append(
            {
                "o_swap_new": _overlap(s[p], s[new_pos], k),
                "o_swap_old": _overlap(s[p], s[old_pos], k),
            }
        )
    summary = summarize_route(per_doc, swaps, k)
    summary["n_docs"] = len(docs)
    summary["refresh_rate"] = float(np.mean(refresh)) if refresh else None
    summary["seconds"] = round(time.perf_counter() - t0, 2)
    return summary


def route_check(
    db: Database,
    exp_id: int,
    *,
    split: str = "dev",
    n_docs: int = 1000,
    cpu: bool = False,
    batch_rows: int = 64,
) -> dict[str, Any]:
    """``route-check --exp N``: :func:`route_follow` on the experiment's latest checkpoint, on its
    dev documents (``split="dev"``) or the first ``n_docs`` delayed training documents
    (``split="train"``, in-sample). ``db`` is only read."""
    from hsn.config import RunConfig
    from hsn.data.dataset import ByteDataset
    from hsn.experiments.db import loads
    from hsn.experiments.probe import delayed_docs
    from hsn.experiments.probe_check import _restored
    from hsn.runner import load_datasets

    if split not in ("dev", "train"):
        raise ValueError(f"split must be 'dev' or 'train', got {split!r}")
    exp = db.experiment(exp_id)
    result = db.row("SELECT json_path FROM results WHERE exp_id = ?", (exp_id,))
    # the run directory, else the stored result's (as probe-rerun and split-backfill find it)
    run_dir = exp["run_dir"] or (
        str(Path(result["json_path"]).parent) if result and result["json_path"] else None
    )
    if not run_dir:
        raise ValueError(f"experiment {exp_id} has no run directory and no stored result")
    _exp, _params, ctx, dev, checkpoint, model = _restored(db, exp_id, Path(run_dir), cpu=cpu)
    if split == "train":
        arm = db.arm(exp["arm_id"])
        cfg = RunConfig.model_validate(loads(exp.get("config_json") or arm["config_json"]))
        train, _dev = load_datasets(cfg)
        picked = [
            ex
            for ex in train.examples
            if (ex.meta.get("task"), ex.meta.get("variant")) == ("delayed", "in_range")
        ][:n_docs]
        docs = delayed_docs(ByteDataset(picked))
    else:
        docs = delayed_docs(dev)[:n_docs]
    summary = route_follow(model, ctx, docs, batch_rows=batch_rows)
    return {
        "exp_id": exp_id,
        "label": exp["label"],
        "checkpoint": str(checkpoint),
        "split": split,
        "keys": "seen in training (in-sample)" if split == "train" else "never seen in training",
        "device": str(ctx.device),
        "autocast": str(ctx.autocast_dtype),
        "batch_rows": batch_rows,
        "diagnostic": "thresholds of the diagnostic only, not frozen metrics",
        **summary,
    }
