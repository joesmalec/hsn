"""The first/later split of the delayed/in_range dev accuracy (P1.15).

In a ``delayed`` document (:class:`hsn.data.synthetic._Delayed`) the loss positions are the key
positions of the queries ``QRY k v``, in order: slot 0 always asks the first fact (answerable by
position alone), slots 1 to 3 ask later facts by key. The audit's analysis of 8 Nov 2026 found
every strong model at about 1.0 on slot 0 and about 0.60 on the later slots, just under the
key-blind bound, so one dev accuracy hides the task's real test. This module splits the
evaluation's own per-position values (:func:`hsn.eval.evaluate.evaluate_windows` with
``position_groups``) by query slot:

* :class:`QuerySlots` labels every loss position of the dev set: ``first``, ``later/1``,
  ``later/2``, ``later/3`` in delayed/in_range documents (parsed and checked: each loss position
  follows ``QRY`` and is a key, slot 0 asks the first fact), ``other`` everywhere else;
* :func:`split_block` turns the evaluation's ``per_position_group`` into the record's ``split``
  block (``first``, ``later`` over slots 1 to 3, and each later slot: accuracy, loss and ``n``)
  and checks it against the frozen per-group metric (errors over silence): the counts and the
  correct answers of the slots sum to the group's, and the losses to its total.

The primary metric is computed exactly as before (the split is a second accumulator over the same
values); without ``position_groups`` the evaluation is today's code.

**Backfill** (:func:`backfill_split`, ``python -m hsn.experiments split-backfill``): runs stored
before P1.15 have no split. Every one gets the audit's derived split (first = the probe's
first-query accuracy, later by subtraction); a run whose checkpoint is kept can be evaluated again
with the slots (``--recompute``), which gives the exact split and the later slots. It reads the
database read-only and writes only its output file; ``report --split-backfill FILE`` shows it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import QRY, DelayedParams

__all__ = [
    "LATER_SLOTS",
    "SLOT_NAMES",
    "SPLIT_GROUP",
    "QuerySlots",
    "backfill_split",
    "backfill_table",
    "derived_split",
    "query_slots",
    "split_block",
    "split_counts",
    "split_names",
]

_DELAYED = DelayedParams()
_KEYS = range(
    min(lo for lo, _ in _DELAYED.keys.values()), max(hi for _, hi in _DELAYED.keys.values())
)
_VALUES = range(*_DELAYED.values)
#: the documents the split covers (the reads-state probe's group)
SPLIT_GROUP = ("delayed", "in_range")
#: slot 0 and the later slots (the generator asks at most ``max_queries`` = 4)
SLOT_NAMES: tuple[str, ...] = ("first", *(f"later/{q}" for q in range(1, _DELAYED.max_queries)))
LATER_SLOTS: tuple[str, ...] = SLOT_NAMES[1:]
OTHER = "other"
NAMES: tuple[str, ...] = (*SLOT_NAMES, OTHER)
_OTHER_ID = NAMES.index(OTHER)


def query_slots(tokens: np.ndarray, loss_mask: np.ndarray) -> list[int]:
    """The query slot (0, 1, ...) of every loss position of a delayed document, in order. Raises
    when a loss position does not follow ``QRY`` or is not a key, when slot 0 does not ask the first
    fact, or when there are more queries than the generator asks."""
    positions = [int(p) for p in np.flatnonzero(loss_mask)]
    if not positions:
        raise ValueError("delayed document without a loss position")
    if len(positions) > _DELAYED.max_queries:
        raise ValueError(f"{len(positions)} queries, more than max_queries={_DELAYED.max_queries}")
    for p in positions:
        if p < 1 or int(tokens[p - 1]) != QRY or int(tokens[p]) not in _KEYS:
            raise ValueError(f"loss position {p} is not the key of a QRY k v query")
    first_key = None
    for i in range(len(tokens) - 1):
        k, v = int(tokens[i]), int(tokens[i + 1])
        if k in _KEYS and v in _VALUES and (i == 0 or int(tokens[i - 1]) != QRY):
            first_key = k
            break
    if first_key is None or int(tokens[positions[0]]) != first_key:
        raise ValueError("the first query does not ask the first fact")
    return list(range(len(positions)))


class QuerySlots:
    """The ``position_groups`` of :func:`hsn.eval.evaluate.evaluate_windows` for one dev set: the
    group of every loss position (module docstring). :meth:`start` gives a fresh labeller for one
    evaluation pass; it counts each document's loss positions across windows, so a document may span
    windows, and raises when a document yields more loss positions than it has."""

    names: tuple[str, ...] = NAMES

    def __init__(self, dev: ByteDataset) -> None:
        self._labels: list[np.ndarray] = []
        covered = 0
        for i in range(len(dev)):
            meta = dev.meta(i)
            ex = dev.examples[i]
            lm = np.asarray(ex.loss_mask, dtype=bool)
            n = int(lm.sum())
            if (meta.get("task"), meta.get("variant")) == SPLIT_GROUP:
                try:
                    slots = query_slots(np.asarray(ex.tokens), lm)
                except ValueError as exc:
                    raise ValueError(f"dev document {meta.get('doc_id', i)}: {exc}") from exc
                self._labels.append(np.asarray(slots, dtype=np.int64))
                covered += 1
            else:
                self._labels.append(np.full(n, _OTHER_ID, dtype=np.int64))
        if not covered:
            raise ValueError(
                f"query slots: no {SPLIT_GROUP[0]}/{SPLIT_GROUP[1]} document among {len(dev)} dev "
                "documents"
            )
        self.n_docs = len(dev)
        counts = np.bincount(np.concatenate(self._labels), minlength=len(NAMES))
        #: loss positions per group over the whole dev set
        self.counts: dict[str, int] = {name: int(c) for name, c in zip(NAMES, counts, strict=True)}

    def start(self) -> Callable[[Mapping[str, Tensor]], Tensor]:
        seen = np.zeros(self.n_docs, dtype=np.int64)
        labels = self._labels

        def label(window: Mapping[str, Tensor]) -> Tensor:
            doc = window["doc"].numpy()
            mask = window["loss_mask"].numpy()
            out = np.full(doc.shape, _OTHER_ID, dtype=np.int64)
            for r in range(doc.shape[0]):
                cols = np.flatnonzero(mask[r])
                if cols.size == 0:
                    continue
                docs = doc[r, cols]
                for d in np.unique(docs):
                    if d < 0 or d >= len(labels):
                        raise ValueError(f"query slots: window document {d} is not a dev document")
                    sel = cols[docs == d]
                    lo = int(seen[d])
                    slots = labels[d][lo : lo + sel.size]
                    if slots.size != sel.size:
                        raise ValueError(
                            f"query slots: document {d} yields more than its "
                            f"{labels[d].size} loss positions"
                        )
                    out[r, sel] = slots
                    seen[d] += sel.size
            return torch.from_numpy(out)

        return label


def split_counts(dev: ByteDataset) -> dict[str, int]:
    """Loss positions per slot of ``dev``'s delayed/in_range documents (``first`` and the later
    slots) and ``later``, their sum."""
    counts = QuerySlots(dev).counts
    out = {name: counts[name] for name in SLOT_NAMES}
    out["later"] = sum(counts[name] for name in LATER_SLOTS)
    return out


def _entry(accuracy: float, loss: float, n: int) -> dict[str, Any]:
    if n == 0:
        return {"accuracy": None, "loss": None, "n": 0}
    return {"accuracy": float(accuracy), "loss": float(loss), "n": int(n)}


def split_block(
    per_position_group: Mapping[str, Mapping[str, Any]],
    group: Mapping[str, Any],
    *,
    total_positions: int | None = None,
) -> dict[str, dict[str, Any]]:
    """The record's ``split`` block from an evaluation's ``per_position_group`` (module
    docstring), checked against ``group``, the evaluation's ``per_group`` entry of delayed/in_range
    (and, with ``total_positions``, against every loss position of the evaluation). Raises when
    they disagree."""
    out: dict[str, dict[str, Any]] = {}
    correct: dict[str, int] = {}
    loss_sum: dict[str, float] = {}
    for name in SLOT_NAMES:
        entry = per_position_group.get(name) or {}
        n = int(entry.get("n_positions") or 0)
        acc = float(entry["accuracy"]) if n else 0.0
        loss = float(entry["loss"]) if n else 0.0
        hits = acc * n
        if abs(hits - round(hits)) > 1e-6:
            raise ValueError(f"split: {name} accuracy {acc} x {n} is not a count of answers")
        correct[name], loss_sum[name] = round(hits), loss * n
        out[name] = _entry(acc, loss, n)
    n_later = sum(int(out[name]["n"]) for name in LATER_SLOTS)
    later_correct = sum(correct[name] for name in LATER_SLOTS)
    later_loss = sum(loss_sum[name] for name in LATER_SLOTS)
    out["later"] = _entry(
        later_correct / n_later if n_later else math.nan,
        later_loss / n_later if n_later else math.nan,
        n_later,
    )
    n_group = int(group.get("n_positions") or 0)
    n_split = int(out["first"]["n"]) + n_later
    if n_group and n_split != n_group:
        raise ValueError(f"split: {n_split} slot positions but the group has {n_group}")
    group_hits = float(group["accuracy"]) * n_group
    if n_group and round(group_hits) != correct["first"] + later_correct:
        raise ValueError(
            f"split: {correct['first']} + {later_correct} correct answers but the group has "
            f"{group_hits:.3f}"
        )
    if n_group:
        total_loss = float(group["loss"]) * n_group
        if not math.isclose(loss_sum["first"] + later_loss, total_loss, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"split: slot losses do not sum to the group's {total_loss}")
    if total_positions is not None:
        other = int((per_position_group.get(OTHER) or {}).get("n_positions") or 0)
        if n_split + other != total_positions:
            raise ValueError(
                f"split: {n_split} + {other} labelled positions, the evaluation has "
                f"{total_positions}"
            )
    return out


def split_names() -> Sequence[str]:
    """The split groups of a record, in report order."""
    return ("first", "later", *LATER_SLOTS)


# --------------------------------------------------------------------------------------------------
# Read-only backfill of stored runs (``split-backfill``)
# --------------------------------------------------------------------------------------------------


def derived_split(
    dev_accuracy: float, first_query_accuracy: float, counts: Mapping[str, int]
) -> dict[str, Any]:
    """The audit's hand reconstruction for a run stored before P1.15: first = the probe's
    first-query accuracy (every dev document's first query, from a fresh state), later =
    (dev x N - first x N1) / N2 over the dev set's counts. The probe and the evaluation agree to
    rounding, so ``later`` is approximate; the later slots are not available."""
    n1, n2 = int(counts["first"]), int(counts["later"])
    total = dev_accuracy * (n1 + n2)
    later = (total - first_query_accuracy * n1) / n2 if n2 else None
    return {
        "first_acc": float(first_query_accuracy),
        "later_acc": later,
        "later_1": None,
        "later_2": None,
        "later_3": None,
    }


def backfill_split(
    db: Any,
    exp_ids: Sequence[int] | None = None,
    *,
    recompute: bool = False,
    cpu: bool = False,
    log: Callable[[str], None] = lambda line: None,
) -> dict[str, Any]:
    """P1.15's read-only backfill of the split for stored reads_state runs (``split-backfill``).

    Every stored record gets the derived split (:func:`derived_split`); with ``recompute``, a run
    whose checkpoint is kept is also evaluated again, eager, on its hardware (``cpu``: profile:cpu
    in fp32) with :class:`QuerySlots`, which gives the exact split, the later slots included, and
    its overall accuracy is compared with the stored one. ``db`` is only read (open it with
    ``read_only=True``); nothing is written beside the returned dict."""
    import dataclasses
    import tempfile
    from pathlib import Path

    from hsn.config import RunConfig
    from hsn.experiments.db import loads
    from hsn.experiments.probe_check import _restored
    from hsn.experiments.spec import protocol_params
    from hsn.runner import load_datasets, make_trainer
    from hsn.train.checkpoint import latest_checkpoint

    where = "a.protocol = 'reads_state' AND r.record_json IS NOT NULL"
    params: list[Any] = []
    if exp_ids:
        where += f" AND e.exp_id IN ({', '.join('?' for _ in exp_ids)})"
        params.extend(int(x) for x in exp_ids)
    sql = (
        "SELECT e.exp_id, e.label, e.status, e.run_dir, e.commit_sha, e.imported, a.family, "
        "r.record_json, r.json_path FROM experiments e JOIN arms a ON a.arm_id = e.arm_id "
        f"JOIN results r ON r.exp_id = e.exp_id WHERE {where} ORDER BY e.exp_id"
    )
    with db.read() as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    dev_cache: dict[str, tuple[Any, dict[str, int]]] = {}
    runs: list[dict[str, Any]] = []
    group = "/".join(SPLIT_GROUP)
    for row in rows:
        exp_id = int(row["exp_id"])
        entry: dict[str, Any] = {
            "exp_id": exp_id,
            "label": row["label"],
            "family": row["family"],
            "status": row["status"],
            "commit": (row["commit_sha"] or "")[:8],
        }
        record = loads(row["record_json"]) or {}
        final = record.get("final") or {}
        if final.get("split/first/accuracy") is not None:
            entry["skipped"] = "the record has the split"
            runs.append(entry)
            continue
        exp = db.experiment(exp_id)
        arm = db.arm(exp["arm_id"])
        cfg = RunConfig.model_validate(loads(exp.get("config_json") or arm["config_json"]))
        proto = protocol_params(loads(exp.get("protocol_json") or arm["protocol_json"]))
        n_dev = int(getattr(proto, "n_dev", 0) or 0)
        key = f"{cfg.data.manifest}|{cfg.data.tasks}|{cfg.data.variants}|{n_dev}"
        if key not in dev_cache:
            from hsn.data.dataset import ByteDataset

            _train, dev = load_datasets(cfg)
            if n_dev:
                dev = ByteDataset(list(dev.examples[:n_dev]))
            try:
                dev_cache[key] = (dev, split_counts(dev))
            except ValueError as exc:
                dev_cache[key] = (dev, {"error": str(exc)})  # type: ignore[dict-item]
        dev, counts = dev_cache[key]
        if "error" in counts:
            entry["skipped"] = f"dev set: {counts['error']}"
            runs.append(entry)
            continue
        entry["counts"] = dict(counts)
        dev_acc = final.get(f"{group}/accuracy")
        probe = record.get("probe") or (record.get("at_stop") or {}).get("probe") or {}
        fqa = probe.get("first_query_accuracy")
        entry["stored_dev_accuracy"] = dev_acc
        entry["probe_first_query_accuracy"] = fqa
        entry["eval_step"] = final.get("eval_step")
        if dev_acc is not None and fqa is not None:
            if int(probe.get("n_probed") or 0) != counts["first"]:
                entry["note"] = (
                    f"probe n_probed {probe.get('n_probed')} != first queries {counts['first']}"
                )
            entry["derived"] = derived_split(float(dev_acc), float(fqa), counts)
        # the run directory, else the stored result's (as probe-rerun finds it)
        run_dir: Path | None = (
            Path(row["run_dir"])
            if row["run_dir"]
            else (Path(row["json_path"]).parent if row["json_path"] else None)
        )
        checkpoint = latest_checkpoint(run_dir) if run_dir is not None else None
        entry["checkpoint"] = str(checkpoint) if checkpoint is not None else None
        if recompute and checkpoint is not None and run_dir is not None:
            log(f"exp {exp_id}: evaluating {checkpoint}")
            _exp, rparams, ctx, rdev, ckpt, model = _restored(db, exp_id, run_dir, cpu=cpu)
            eager = dataclasses.replace(ctx, compile_enabled=False)
            with tempfile.TemporaryDirectory() as tmp:
                trainer = make_trainer(model, cfg, eager, run_dir=Path(tmp))
                trainer.eval_batch = getattr(rparams, "eval_batch", None)
                trainer.eval_position_groups = QuerySlots(rdev)
                result = trainer.evaluate(rdev)
            block = split_block(
                result["per_position_group"],
                result["per_group"][group],
                total_positions=result["n_positions"],
            )
            n_all = int(result["per_group"][group]["n_positions"])
            overall = float(result["per_group"][group]["accuracy"])
            entry["recomputed"] = {
                "first_acc": block["first"]["accuracy"],
                "later_acc": block["later"]["accuracy"],
                **{f"later_{q}": block[f"later/{q}"]["accuracy"] for q in (1, 2, 3)},
                "overall": overall,
                "overall_minus_stored_answers": (
                    round((overall - float(dev_acc)) * n_all) if dev_acc is not None else None
                ),
                "checkpoint_step": int(ckpt.stem.split("_")[-1]),
                "device": str(eager.device),
                "autocast": str(eager.autocast_dtype),
                "split": block,
            }
        best = entry.get("recomputed") or entry.get("derived")
        if best is not None:
            entry["tier"] = "recomputed" if "recomputed" in entry else "derived"
            for col in ("first_acc", "later_acc", "later_1", "later_2", "later_3"):
                entry[col] = best.get(col)
        runs.append(entry)
    return {"runs": runs, "recompute": recompute, "cpu": cpu}


def backfill_table(result: Mapping[str, Any]) -> str:
    """A markdown table of :func:`backfill_split`'s result."""

    def f(x: Any, d: int = 4) -> str:
        return "" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))

    lines = [
        "| exp | label | family | dev | probe first | derived later | recomputed first | "
        "recomputed later | later 1 / 2 / 3 | recomputed - stored (answers) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["runs"]:
        if "skipped" in r:
            continue
        d, c = r.get("derived") or {}, r.get("recomputed") or {}
        slots = " / ".join(f(c.get(f"later_{q}"), 3) for q in (1, 2, 3)) if c else ""
        lines.append(
            f"| {r['exp_id']} | {r['label']} | {r['family']} | {f(r.get('stored_dev_accuracy'))} | "
            f"{f(r.get('probe_first_query_accuracy'), 3)} | {f(d.get('later_acc'))} | "
            f"{f(c.get('first_acc'), 3)} | {f(c.get('later_acc'))} | {slots} | "
            f"{f(c.get('overall_minus_stored_answers'))} |"
        )
    return "\n".join(lines) + "\n"
