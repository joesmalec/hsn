"""The by-distance passes of a run and of a checkpoint (``eval.by_distance``).

:class:`SetScorer` is one dataset: its task/variant groups, which documents need their
log-probability rows, and the static text index, built once. :class:`ByDistance` is the
trainer's evaluation observer (:attr:`hsn.train.loop.Trainer.eval_observer`): it captures the
pass (:class:`~hsn.eval.distance.capture.PositionCapture`) and puts the groups' tables into the
evaluation's result under ``by_distance``. :class:`RunScorer` is what a run does with the key
on: the sets are loaded when the run is set up (a wrong manifest fails before training), and
:meth:`RunScorer.final` scores each with one :meth:`~hsn.train.loop.Trainer.evaluate` pass, so
the windows, the carried state, the autocast and the weights are the run's own evaluations'.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor

from hsn.config import RunConfig
from hsn.data.dataset import ByteDataset
from hsn.data.splits import SplitManifest
from hsn.eval.distance.capture import Captured, PositionCapture
from hsn.eval.distance.recall import _OWN_VALUE_FAMILIES, recall_table, value_range
from hsn.eval.distance.text import TextIndex, find_frequent_words, name_list, text_table

__all__ = ["KEY", "SCHEMA", "ByDistance", "RunScorer", "SetScorer", "cap_documents", "render"]

#: Version of the ``by_distance`` block (``docs/distance_eval.md``); raised when a key changes
#: meaning or goes away, not when one is added.
SCHEMA = 1
#: The block's key in an evaluation result and in the run record.
KEY = "by_distance"

logger = logging.getLogger("hsn.eval.distance")


def _group(meta: Mapping[str, Any]) -> str:
    return f"{meta.get('task', '?')}/{meta.get('variant', '?')}"


def cap_documents(ds: ByteDataset, max_docs: int | None) -> ByteDataset:
    """The first ``max_docs`` documents of every task/variant group, in the dataset's order
    (``None``: the dataset itself). The same documents for every seed and run."""
    if max_docs is None:
        return ds
    taken: dict[str, int] = {}
    kept = []
    for example in ds.examples:
        group = _group(example.meta)
        if taken.get(group, 0) < max_docs:
            taken[group] = taken.get(group, 0) + 1
            kept.append(example)
    return ds if len(kept) == len(ds) else ByteDataset(kept)


class SetScorer:
    """One dataset to score: ``ds`` (capped to ``max_docs`` per group), evaluated in windows
    of ``window`` positions; ``dataset`` is its manifest's name (the frequent-word list of a
    text set is looked up by it)."""

    def __init__(
        self,
        ds: ByteDataset,
        *,
        window: int,
        max_docs: int | None = None,
        dataset: str | None = None,
    ) -> None:
        if len(ds) == 0:
            raise ValueError("by-distance scoring needs at least one document")
        self._full = ds
        self.ds = cap_documents(ds, max_docs)
        self.window = window
        self.dataset = dataset
        self.groups: dict[str, list[int]] = {}
        for i, example in enumerate(self.ds.examples):
            self.groups.setdefault(_group(example.meta), []).append(i)
        self._rows_for = [
            str(example.meta["family"]) in _OWN_VALUE_FAMILIES for example in self.ds.examples
        ]
        self._text: dict[str, tuple[TextIndex, np.ndarray[Any, np.dtype[np.int64]]]] = {}
        # the value alphabet of every own-value group, read from the whole (uncapped) group
        self._values = {
            group: value_range([e for e in ds.examples if _group(e.meta) == group])
            for group, docs in self.groups.items()
            if self._rows_for[docs[0]]
        }

    def capture(self) -> PositionCapture:
        """The observer of one pass over :attr:`ds`."""
        return PositionCapture(self.ds, self._rows_for)

    def _text_index(
        self, group: str, docs: list[int], captured: Captured
    ) -> tuple[TextIndex, np.ndarray[Any, np.dtype[np.int64]]]:
        if group not in self._text:
            names = name_list(
                [bytes(e.tokens) for e in self._full.examples if _group(e.meta) == group]
            )
            index = TextIndex(
                [bytes(self.ds.examples[i].tokens) for i in docs],
                names,
                window=self.window,
                frequent=find_frequent_words(self.dataset),
            )
            offsets = captured.offsets
            flat = np.concatenate([np.arange(offsets[d], offsets[d + 1]) for d in docs])
            self._text[group] = (index, flat)
        return self._text[group]

    def tables(self, captured: Captured) -> dict[str, Any]:
        """``{"<task>/<variant>": table}`` of one pass: :func:`~hsn.eval.distance.text.
        text_table` for a text group, :func:`~hsn.eval.distance.recall.recall_table` for
        every other."""
        out: dict[str, Any] = {}
        for group, docs in self.groups.items():
            examples = [self.ds.examples[i] for i in docs]
            if str(examples[0].meta["family"]) == "text":
                index, flat = self._text_index(group, docs, captured)
                out[group] = text_table(index, captured.nll[flat], captured.hit[flat])
            else:
                out[group] = recall_table(
                    examples, docs, captured, window=self.window, values=self._values.get(group)
                )
        return out


class ByDistance:
    """The evaluation observer (:class:`hsn.train.loop.EvalObserver`): every evaluated dataset
    gets a :class:`SetScorer` (kept, so a dataset evaluated again reuses its static index), the
    pass is captured, and :meth:`finish` gives its groups' tables. ``captured`` is the last
    pass (for a dump).

    It never fails the evaluation it watches (which may be one of a run's own, mid-training):
    an error while capturing or tabling is logged, the rest of the pass is not captured, and
    :meth:`finish` gives ``{"error": "<type>: <message>"}`` in place of the tables."""

    key = KEY

    def __init__(self, *, window: int, dataset: str | None = None) -> None:
        self.window = window
        self.dataset = dataset
        self.captured: Captured | None = None
        self._scorers: dict[int, SetScorer] = {}
        self._active: tuple[SetScorer, PositionCapture] | None = None
        self._error: str | None = None

    def add(self, scorer: SetScorer) -> None:
        """Use ``scorer`` for its own dataset (its cap and dataset name)."""
        self._scorers[id(scorer.ds)] = scorer

    def _fail(self, exc: Exception) -> None:
        logger.exception("by-distance scoring failed; the evaluation goes on without it")
        self._error = f"{type(exc).__name__}: {exc}"[:500]

    def start(self, ds: ByteDataset) -> Callable[[Mapping[str, Tensor], Tensor], None]:
        self._error, self._active, self.captured = None, None, None
        try:
            scorer = self._scorers.get(id(ds))
            if scorer is None or scorer.ds is not ds:
                scorer = SetScorer(ds, window=self.window, dataset=self.dataset)
                self._scorers[id(ds)] = scorer
            self._active = (scorer, scorer.capture())
        except Exception as exc:  # noqa: BLE001 - reported by finish()
            self._fail(exc)
        return self._observe

    def _observe(self, window: Mapping[str, Tensor], logits: Tensor) -> None:
        if self._error is not None or self._active is None:
            return
        try:
            self._active[1](window, logits)
        except Exception as exc:  # noqa: BLE001 - reported by finish()
            self._fail(exc)

    def finish(self) -> dict[str, Any]:
        active, self._active = self._active, None
        if self._error is None and active is not None:
            try:
                self.captured = active[1].result()
                return active[0].tables(self.captured)
            except Exception as exc:  # noqa: BLE001 - reported in the result
                self._fail(exc)
        return {"error": self._error or "no pass was captured"}


def _manifest_name(path: str | Path | None) -> str | None:
    return SplitManifest.load(Path(path)).name if path is not None else None


class RunScorer:
    """``eval.by_distance`` of one run (module docstring). ``dev_ds`` is the run's dev set as
    it evaluates it (``None``: only the sets of ``eval.by_distance.sets``)."""

    def __init__(self, cfg: RunConfig, dev_ds: ByteDataset | None) -> None:
        by = cfg.eval.by_distance
        self.cfg = cfg
        self.window = int(cfg.train.unroll)
        self._dev_name = _manifest_name(cfg.data.manifest) if dev_ds is not None else None
        self.sets: dict[str, tuple[SetScorer, dict[str, Any]]] = {}

        def add(name: str, ds: ByteDataset, dataset: str | None, source: dict[str, Any]) -> None:
            scorer = SetScorer(ds, window=self.window, max_docs=by.max_docs, dataset=dataset)
            source = {
                **source,
                "dataset": dataset,
                "split": "dev",
                "n_docs": len(scorer.ds),
                "max_docs": by.max_docs,
            }
            self.sets[name] = (scorer, source)

        if dev_ds is not None:
            add(
                "dev",
                dev_ds,
                self._dev_name,
                {
                    "manifest": str(cfg.data.manifest),
                    "tasks": list(cfg.data.tasks),
                    "variants": list(cfg.data.variants),
                },
            )
        for entry in by.sets:
            ds = ByteDataset.load(
                entry.manifest,
                "dev",
                list(entry.tasks) or None,
                list(entry.variants) or None,
                expected_unroll=cfg.train.unroll,
            )
            add(
                entry.name,
                ds,
                _manifest_name(entry.manifest),
                {
                    "manifest": str(entry.manifest),
                    "tasks": list(entry.tasks),
                    "variants": list(entry.variants),
                },
            )
        if not self.sets:
            raise ValueError("eval.by_distance: no set to score")

    def observer(self) -> ByDistance:
        """The observer of the run's own evaluations (``eval.by_distance.every_eval``): the
        dev set as the run evaluates it, every document."""
        return ByDistance(window=self.window, dataset=self._dev_name)

    def final(
        self, trainer: Any, *, step: int | None = None, dump_dir: Path | None = None
    ) -> dict[str, Any]:
        """The ``by_distance`` block: one :meth:`~hsn.train.loop.Trainer.evaluate` pass per
        set with the weights the trainer's evaluations score. A set that fails keeps its
        ``source`` and an ``error`` (and the pass's own numbers when the pass itself ran) and
        does not stop the others: the run's record is worth more than its report. With
        ``dump_dir`` every set's per-position dump is written to ``<dump_dir>/<set>.npz``."""
        started = time.perf_counter()
        out: dict[str, Any] = {
            "schema": SCHEMA,
            "step": int(trainer.step if step is None else step),
            # train.weight_ema: the evaluations (and this pass) score the weight average
            "weights": "averaged"
            if getattr(trainer, "weight_ema", None) is not None
            else "trained",
            "window": self.window,
            "align_docs": bool(self.cfg.data.align_docs),
            "sets": {},
        }
        previous = trainer.eval_observer
        try:
            for name, (scorer, source) in self.sets.items():
                observer = ByDistance(window=self.window, dataset=scorer.dataset)
                observer.add(scorer)
                trainer.eval_observer = observer
                began = time.perf_counter()
                try:
                    result = trainer.evaluate(scorer.ds)
                    block: dict[str, Any] = {
                        "accuracy": float(result["accuracy"]),
                        "loss": float(result["loss"]),
                        "n_positions": int(result["n_positions"]),
                        "n_windows": int(result["n_windows"]),
                        "seconds": round(time.perf_counter() - began, 3),
                    }
                    groups = result[KEY]
                    if set(groups) == {"error"}:  # the pass ran, its tables failed
                        block["error"] = groups["error"]
                    else:
                        block["groups"] = groups
                    if dump_dir is not None and observer.captured is not None:
                        Path(dump_dir).mkdir(parents=True, exist_ok=True)
                        observer.captured.save(Path(dump_dir) / f"{name}.npz")
                except Exception as exc:  # noqa: BLE001 - kept in the block, the run goes on
                    logger.exception("by-distance scoring of set %s failed", name)
                    block = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                out["sets"][name] = {"source": source, **block}
        finally:
            trainer.eval_observer = previous
        out["seconds"] = round(time.perf_counter() - started, 3)
        return out


# --------------------------------------------------------------------------------------------------
# A plain-text rendering (the CLI and the run log)
# --------------------------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.1f}"


def _cells(cells: Mapping[str, Mapping[str, Any]]) -> str:
    return "  ".join(f"{key} {_pct(c['accuracy'])} ({c['n']})" for key, c in cells.items())


def _render_group(name: str, table: Mapping[str, Any]) -> list[str]:
    if table["kind"] == "text":
        lines = [
            f"  {name}: {table['bits_per_byte']:.4f} bits per byte, accuracy "
            f"{_pct(table['accuracy'])} ({table['n']} bytes, {table['n_docs']} documents)",
            "    names inside a sentence, by the previous mention (bits per letter after the "
            "capital | bits of the capital; 95 % over stories)",
        ]
        for key, row in table["names"]["rows"].items():
            parts = []
            for which in ("rest", "first"):
                value, ci = row[f"bits_{which}"], row[f"ci95_{which}"]
                parts.append("-" if value is None else f"{value:.2f} ({ci[0]:.2f}-{ci[1]:.2f})")
            lines.append(f"      {key:24s} n {row['n']:5d}   {parts[0]:18s} | {parts[1]}")
        words = table["words"]
        if "rows" not in words:
            lines.append(f"    words: skipped ({words['skipped']})")
            return lines
        lines.append("    letters of content words, by the word's last use (bits per letter)")
        for key, row in words["rows"].items():
            bits = "-" if row["bits"] is None else f"{row['bits']:.3f}"
            lines.append(f"      {key:24s} n {row['n']:7d}   share {row['share']:.3f}   {bits}")
        return lines
    lines = [
        f"  {name}: accuracy {_pct(table['accuracy'])} ({table['n']} answers, "
        f"{table['n_docs']} documents)",
        f"    by distance: {_cells(table['by_distance'])}",
        f"    by windows back: {_cells(table['by_windows_back'])}",
    ]
    if "by_query" in table:
        lines.append(f"    by query: {_cells(table['by_query'])}")
        for fact, row in table["by_fact_position"].items():
            lines.append(f"    fact at {fact}, by distance: {_cells(row)}")
    if "by_target" in table:
        lines.append(
            f"    by target: {_cells(table['by_target'])}  predicts NONE "
            f"{_pct(table['predicts_none'])}"
        )
        lines.append(f"    value targets by distance: {_cells(table['by_distance_value_targets'])}")
    own = table.get("own_values")
    if own is not None and "accuracy" in own:
        lines.append(
            f"    restricted to the document's values: {_pct(own['accuracy'])} (chance "
            f"{_pct(own['chance'])}); predicts one of them {_pct(own['predicted_share'])}"
        )
    return lines


def render(block: Mapping[str, Any]) -> str:
    """The ``by_distance`` block as text: one paragraph per set and group."""
    lines: list[str] = []
    for name, entry in block["sets"].items():
        source = entry["source"]
        head = f"{name} ({source['dataset']}, {source['n_docs']} documents)"
        if "error" in entry:
            lines.append(f"{head}: failed: {entry['error']}")
            continue
        lines.append(
            f"{head}: accuracy {_pct(entry['accuracy'])}, loss {entry['loss']:.4f} nats, "
            f"{entry['seconds']:.1f} s"
        )
        for group, table in entry["groups"].items():
            lines.extend(_render_group(group, table))
    return "\n".join(lines)
