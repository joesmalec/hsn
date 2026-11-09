"""Generic windowed trainer (P0.5): the loop, evaluation and search budget baselines share with HSN.

**Protocol.** Every model the trainer runs satisfies :class:`SequenceModel`: ``init_state``,
``reset_state``, ``forward(tokens, context, state, reset) -> (logits [B, T, V], state, info)`` and
``param_report``. The GRU and Transformer baselines (:mod:`hsn.baselines`) implement it; the HSN
module (P1.5) and the TBPTT trainer (P1.7, a subclass of :class:`Trainer`) build on the same loop.

**Streams and windows.** Training data are *documents* (:class:`~hsn.data.dataset.ByteDataset`).
:class:`StreamLoader` concatenates them into ``train.batch`` parallel streams, exactly as
:func:`hsn.data.dataset.collate_windows` does, and slices the streams into consecutive windows of
``train.unroll`` positions, lazily and without end: a row that runs out of positions takes the
next document of a per-epoch permutation seeded from ``seed`` (plan §4.3: data order is
deterministic given the seed). With ``data.align_docs`` every document starts at a window boundary
and the tail of its last window is padding, which is what makes P0.4's ``in_range`` ("the
dependency fits one window") true. The loader's state is a few integers per row, so a checkpoint
resumes at the same window (:func:`hsn.train.checkpoint`).

**One step** is ``train.grad_accum`` windows: the recurrent state is detached at every window
start (truncated BPTT), the model applies the ``reset`` flags itself (a document never sees the
state of the one before it), the loss is next-symbol cross-entropy over the positions where
``loss_mask`` is True (every supervised position in the accumulated step has equal weight),
gradients are clipped to ``train.grad_clip`` (global norm), AMP follows the
device context (:mod:`hsn.train.amp`), the learning rate follows ``train.schedule`` with
``train.warmup_steps`` over the horizon ``train.steps``. Every ``train.eval_every`` steps and at
the end the dev set is evaluated (:func:`hsn.eval.evaluate.evaluate_windows`); every
``train.ckpt_every`` steps and at the end a checkpoint is written to ``run_dir``
(``runs/<run_id>`` unless given). ``fit``
returns a :class:`History` with one record per step, every evaluation, the model's parameter
report and the *measured* numbers the run manifest needs (plan §0.5: measured, not assumed). P1.16:
:attr:`Trainer.extra_ckpt_every` (a harness setting, ``None`` by default) writes a checkpoint at its
multiples too. P1.17: repeated calls to ``fit`` retain the loader cursor and carried state, just
like checkpoint resume. Loader checkpoints fingerprint the ordered dataset; legacy checkpoints
without the fingerprint remain readable.

**Schedules.** The learning-rate factor of step ``s`` (``LambdaLR``'s count of scheduler steps, so
optimiser step ``k`` trains at the factor of ``k - 1``) is ``(s + 1) / warmup_steps`` during the
warm-up, then ``constant`` 1, ``cosine`` ``0.5 (1 + cos(pi p))`` or ``linear`` ``1 - p`` with ``p``
the progress from the warm-up's end to ``train.steps``. **P1.13:** ``wsd`` (warmup-stable-decay,
Hägele et al. 2024) holds 1 after the warm-up and decays over the last
``D = round(train.decay_fraction * train.steps)`` steps as ``1 - sqrt((s - (steps - D)) / D)``,
reaching 0 at ``train.steps`` like the other decays (the 1 - sqrt shape gave a lower final loss
than a linear decay there; a 20 % decay matched cosine). Its value is a function of the step and
``train.steps`` only, so a run that stops before the decay (P1.11's protocol ``steps`` below
``train.steps``) trains at the peak throughout. Every parameter group follows the same factor of
its own base rate.

**Hooks (P1.11).** ``fit(..., callback=f)`` calls ``f(trainer, history)`` once per step, after the
step's evaluation and checkpoint; when it returns True before the last step, the trainer writes a
checkpoint (unless that step already wrote one) and stops there (the sweep harness's early-stop
rules, :mod:`hsn.experiments`). The step it stops at is not a statistics step of the TBPTT trainer
(those key on the last step of ``fit``). :attr:`Trainer.eval_batch` (``None`` = ``train.batch``)
sets the number of parallel streams of :meth:`Trainer.evaluate` only; training is unaffected and
the per-document metrics change by rounding at most. **P1.15:** :attr:`Trainer.eval_position_groups`
(``None`` by default) labels the evaluation's loss positions
(:func:`hsn.eval.evaluate.evaluate_windows`, ``position_groups``); the result gains
``per_position_group``, everything else is computed as before.

**Search budget.** :class:`SearchBudget` counts hyperparameter-search trials against
``train.search_trials`` so that every method in the baseline ladder gets the same number of
trials (paper §12); recording a trial beyond the budget raises.

**Guards.** The test split is refused unless a caller says ``allow_test=True`` (gate scripts only;
chunk "Do not"). Shape mismatches, unknown schedules, a non-finite loss under bf16/fp32 (unless its
step was skipped, below), a checkpoint from another configuration and features that belong to later
chunks (micro-batching, P4.3) raise instead of being reinterpreted (plan §4.3: errors over silence).

**Non-finite gradients (P1.12).** Without a loss scaler (bf16 and fp32), a step whose global
gradient norm (before clipping) is not finite is skipped (:meth:`Trainer._guard_nonfinite`): the
gradients are zeroed (``None``), the optimiser and its state are left alone, the scheduler still
advances (the schedule keeps its clock), the skip is counted (``skipped_nonfinite``) and logged.
Clipping such a gradient would write non-finite values into the weights and kill the run a few
steps later (seed 1 of the G1 gate run under bf16 on ``profile:rtx3060``: an infinite norm at step
3,140, a NaN objective at 3,143). Under fp16 the ``GradScaler`` skips such steps itself and the
guard is inactive. The count is in ``History.measured["skipped_nonfinite_steps"]`` and in every
checkpoint (with the current run of consecutive skips; a checkpoint written before the guard reads
0, a malformed count raises); the TBPTT trainer also puts it in every step record. A skipped step
keeps its non-finite ``grad_norm`` in its record. The guard protects the weights and the optimiser
state only: what the forward pass carries (the recurrent state, a scorer's running statistics) is
not rolled back, and a carried state that went non-finite clears at its rows' next reset.
**Stop rule:** a run that keeps skipping is not training, so after a skip :class:`RuntimeError` is
raised when :data:`NONFINITE_MAX_CONSECUTIVE` steps in a row were skipped, or when at least
:data:`NONFINITE_MIN_SKIPS` steps and more than :data:`NONFINITE_MAX_FRACTION` of the steps so far
were skipped. The guard reads the norm the step already copied to the host for its record, so it
adds no device synchronisation (P2.8's compiled window included: the norm, the clipping and the
optimiser run eagerly, after the compiled backward).

**Gradient spikes (P1.15).** A divergence can be a single finite jump: exp 171 (8 Nov 2026) ran with
gradient norms of 0.3 to 0.56, then one batch gave 2.2e10, which clipping applied, and the model
broke. With ``train.spike_factor`` set (``None``, the default, keeps today's training), a step whose
finite pre-clip norm exceeds the factor times the median of the last ``train.spike_window`` (default
100) applied norms is skipped like a non-finite one (:meth:`Trainer._guard_spike`, after
:meth:`Trainer._guard_nonfinite`): gradients zeroed, no optimiser step, the scheduler advances, the
skip counted (``skipped_spike``) and logged. The guard decides once ``min(SPIKE_MIN_HISTORY,
window)`` norms are known; skipped and non-finite norms never enter the history. Spike skips count
in the P1.12 stop rule with the non-finite ones (one run of consecutive skips, one fraction;
:meth:`Trainer._stop_rule`), whose message then names both. The count and the history go into the
checkpoint (only with the guard on) and ``History.measured["skipped_spike_steps"]``; every skipped
step, with its reason, is in :attr:`Trainer.skipped_steps`. The guard reads the same host float as
P1.12's, so it adds no device synchronisation and leaves P2.8's captured graphs alone; under an fp16
``GradScaler`` it is refused. :func:`spike_decision` is the pure rule (evidence scripts use it). The
factor: 100 sits at the edge of ordinary variation. On the stored runs it would skip 55 steps in 17
of the 108 runs that did not fail, learners among them (steps of 180 to 572 times the median in exps
65, 154, 155 and 161); exp 171's jump (7e10 times) and the peak of exp 170's burst (5e14 times) are
caught at any factor up to about 1e10, so 1e3 to 1e4 is the safer choice for experiments (only 3
runs skip anything at 1e4). The default stays ``None`` (off).

**Divergence dumps (P1.15).** With ``train.divergence_dump`` (on by default) each step starts with
a snapshot of what it starts from (:meth:`Trainer._take_snapshot`: the loader position, the
generators, copies of the carried state and the buffers, the counters; host-side work and device
copies, no synchronisation), and the first step a guard skips writes it, with the parameters,
optimiser and scheduler as they still are, as a checkpoint-format dump in ``run_dir``
(:mod:`hsn.train.divergence`, :meth:`Trainer._on_skip`) for a deterministic replay. Nothing in the
training changes: the snapshot only reads. :attr:`Trainer.grad_hook` (``None``; the replay's) is
called after a step's backward, before the clipping.

**Rollback (P1.16).** With ``train.rollback_every`` set (``None``, the default, keeps today's
training) a step the guards above would skip rolls back instead, once a snapshot exists
(:mod:`hsn.train.rollback`): every ``rollback_every`` steps, after an applied optimiser step, the
trainer copies the parameters, buffers, optimiser state and spike history into a ring of two
in-memory snapshots; a trigger (P1.15's dump written first, as for a skip) restores the **older**
snapshot in place (the CUDA Graphs of P2.8 keep their input addresses), discards
``train.rollback_skip`` steps' worth of windows, makes every loader row start a fresh document (the
carried state resets at the next window's ``reset``) and trains at ``lr * train.rollback_lr_mult``
for ``train.rollback_lr_steps`` steps. The step clock, the scheduler and the generators keep
running; the trigger step is listed in :attr:`Trainer.skipped_steps` with reason ``rollback`` and
in :attr:`Trainer.rollbacks` with what was restored. A rollback is not a skip (decision Q2 = A of
8 Nov 2026): it does not count in P1.12's stop rule and resets its run of consecutive skips; its
own limit is ``train.rollback_max`` (the run raises a :class:`RuntimeError` naming
:data:`ROLLBACK_LIMIT` at the ``rollback_max + 1``-th trigger). A trigger before the first snapshot
is P1.15's skip and counts as one. The count is ``History.measured["rollbacks"]`` and, with the
learning-rate cut, part of the checkpoint (the snapshots are not: a resumed run re-arms at its next
multiple of ``rollback_every``). Refused under an fp16 ``GradScaler``.

**Optimiser (P2.6).** ``train.optimizer=adamw`` builds ``torch.optim.AdamW`` with weight decay on
the tensors of ``ndim >= 2`` and none on the rest; ``train.optimizer=lazy_adamw`` builds the same
groups as a :class:`~hsn.train.lazy_adam.LazyAdamW` with the unit bank's stacked tensors in unit
groups, and needs a model with a :class:`~hsn.model.units.UnitBank` (the baselines have none and
raise). The TBPTT trainer hands the lazy optimiser the units each step executed.

**The cell group (P1.13).** With ``train.cell_lr_mult != 1`` or ``train.cell_weight_decay`` set,
the HSN cell's parameters (:func:`~hsn.train.lazy_adam.cell_parameters`: ``V``, ``W_out``, the
gate, the input and candidate norms, the unit bank, the edge logits; the ``cell`` group of the
P1.11 diagnostics) train in groups of their own at ``train.lr * cell_lr_mult``, with
``cell_weight_decay`` (``None``: ``train.weight_decay``) on their ``ndim >= 2`` tensors and none on
the rest. The other parameters' groups come first, so the logged learning rate
(``param_groups[0]``) stays the global one. ``LambdaLR`` scales each group's own base rate by the
same factor, so the ratio holds through the warm-up and the decay. Gradient clipping and the
non-finite guard stay global (over every parameter). With the defaults the groups are exactly the
ones above. A model without an HSN cell refuses the keys (the configuration already does).
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import logging
import math
import statistics
import time
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import RunConfig, config_hash, wsd_decay_steps
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import PAD
from hsn.device import (
    DeviceContext,
    memory_stats,
    reset_peak,
    seed_everything,
    set_deterministic,
    synchronize,
)
from hsn.eval.evaluate import DEFAULT_METRICS, Metric, PositionGroups, evaluate_windows
from hsn.train.adaptive import AdaptiveLR
from hsn.train.amp import autocast_ctx, autocast_name, make_grad_scaler
from hsn.train.checkpoint import (
    capture_rng_state,
    checkpoint_path,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from hsn.train.divergence import StepSnapshot, persistent_buffers, write_dump
from hsn.train.lazy_adam import build_lazy_adamw, cell_parameters
from hsn.train.rollback import RollbackPolicy, SnapshotRing

__all__ = [
    "NONFINITE_MAX_CONSECUTIVE",
    "NONFINITE_MAX_FRACTION",
    "NONFINITE_MIN_SKIPS",
    "ROLLBACK_LIMIT",
    "SCHEDULES",
    "SPIKE_MIN_HISTORY",
    "SPIKE_WINDOW",
    "BudgetExhaustedError",
    "FitCallback",
    "History",
    "SearchBudget",
    "SequenceModel",
    "StepRecord",
    "StreamLoader",
    "Trainer",
    "Trial",
    "detach_state",
    "spike_decision",
]

logger = logging.getLogger("hsn.train")

Batch = dict[str, Tensor]
SCHEDULES: tuple[str, ...] = ("constant", "cosine", "linear", "wsd")
# P1.11: called once per step after evaluation and checkpoint; True stops the fit there
FitCallback = Callable[["Trainer", "History"], bool]
_WINDOW_FIELDS = ("tokens", "targets", "context", "loss_mask", "reset", "dep", "doc")

# P1.12: the non-finite-gradient guard's stop rule (module docstring). Module constants, not config
# keys: a new key would change every configuration's hash (old checkpoints would not resume), and
# the rule is a safety stop, not an experimental parameter. Calibration: the guarded post-A1 bf16
# rerun of 5 Nov 2026 skipped 0 of 3,000 steps; G1 lost two of five runs to one bad step each.
#: A run raises at this many consecutive skipped steps.
NONFINITE_MAX_CONSECUTIVE = 10
#: A run raises when it skipped at least NONFINITE_MIN_SKIPS steps and more than this fraction of
#: its steps so far (30 of a 3,000-step run, 120 of 12,000).
NONFINITE_MAX_FRACTION = 0.01
NONFINITE_MIN_SKIPS = 10

# P1.15: the gradient-spike guard (module docstring). train.spike_factor turns it on and
# train.spike_window sets its window (default SPIKE_WINDOW; the stored runs' spike ratios barely
# change between windows of 25 and 200 steps); the minimum history is a module constant, as P1.12's.
#: The default window: the running median reads the last 100 applied (finite, not skipped) norms.
SPIKE_WINDOW = 100
#: The guard decides once it has seen min(SPIKE_MIN_HISTORY, window) applied norms (the warm-up's
#: first steps fall fast: exp 171 went from 4.2 to 0.4 in ten steps).
SPIKE_MIN_HISTORY = 20

#: P1.16: the rollback limit's message names this (the harness's failure kind ``rollback``).
ROLLBACK_LIMIT = "P1.16 rollback limit"


def spike_decision(
    grad_norm: float, history: Collection[float], factor: float, window: int = SPIKE_WINDOW
) -> tuple[bool, float | None]:
    """P1.15: ``(skip, median)`` for a finite ``grad_norm`` against ``history``, the applied norms
    so far (the last ``window`` of them are read): skip when there are at least
    ``min(SPIKE_MIN_HISTORY, window)`` and ``grad_norm > factor * median``. ``median`` is None below
    the minimum history. A pure function: the trainer and the evidence scripts share it."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    recent = list(history)[-window:]
    if len(recent) < min(SPIKE_MIN_HISTORY, window):
        return False, None
    median = statistics.median(recent)
    return bool(grad_norm > factor * median), float(median)


def slow_burst_decision(
    grad_norm: float, history: Collection[float], factor: float, window: int, lag: int
) -> tuple[bool, float | None]:
    """Detect sustained growth against an older median that cannot follow the same ramp.

    Include the candidate in the recent window, but never in the applied history until accepted.
    Requiring the candidate itself to exceed the old reference lets a healthy step recover after
    a skip. Both windows must be full; ``lag`` counts applied steps between their endpoints.
    """
    if window < 1 or lag < 1:
        raise ValueError("slow-burst window and lag must be positive")
    values = [*history, grad_norm]
    if len(values) < window + lag:
        return False, None
    baseline = float(statistics.median(values[-(window + lag) : -lag]))
    recent = float(statistics.median(values[-window:]))
    threshold = factor * baseline
    return bool(recent > threshold and grad_norm > threshold), baseline


def _skipped_counter(extra: Mapping[str, Any], key: str, path: str) -> int:
    """A count of the non-finite-gradient guard from a checkpoint's ``extra``: absent in
    checkpoints written before the guard (P1.12), then 0; otherwise a non-negative int."""
    value = extra.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}: {key} must be a non-negative int, got {value!r}")
    return value


# --------------------------------------------------------------------------------------------------
# Protocol and state helpers
# --------------------------------------------------------------------------------------------------


@runtime_checkable
class SequenceModel(Protocol):
    """What the trainer needs from a model (chunk P0.5 interfaces)."""

    def init_state(self, B: int, device: torch.device | str) -> Any:
        """The recurrent state for ``B`` fresh streams (``None`` for a stateless model)."""

    def reset_state(self, state: Any, mask: Tensor) -> Any:
        """``state`` with the rows where ``mask`` (``[B]`` bool) is True reset to initial."""

    def forward(
        self, tokens: Tensor, context: Tensor, state: Any, reset: Tensor
    ) -> tuple[Tensor, Any, dict[str, Any]]:
        """``tokens``, ``context``, ``reset`` are ``[B, T]``; ``reset[b, t]`` resets row ``b``
        before position ``t``. Returns logits ``[B, T, V]``, the new state and an info dict."""

    def param_report(self) -> dict[str, int]:
        """Measured parameter counts with at least ``active``, ``stored`` and ``trainable``."""


def detach_state(state: Any) -> Any:
    """``state`` with every tensor detached (nested tuples, lists and dicts are walked)."""
    return _map_state(state, lambda t: t.detach())


def _map_state(state: Any, fn: Any) -> Any:
    if state is None:
        return None
    if isinstance(state, Tensor):
        return fn(state)
    if isinstance(state, tuple):
        return tuple(_map_state(s, fn) for s in state)
    if isinstance(state, list):
        return [_map_state(s, fn) for s in state]
    if isinstance(state, dict):
        return {k: _map_state(v, fn) for k, v in state.items()}
    raise TypeError(f"recurrent state must be tensors in tuples/lists/dicts, got {type(state)}")


# --------------------------------------------------------------------------------------------------
# Stream loader
# --------------------------------------------------------------------------------------------------


@dataclass
class _Row:
    doc: int = -1  # index of the document being consumed, -1 when idle
    offset: int = 0  # positions of it already emitted
    arrays: dict[str, np.ndarray[Any, Any]] | None = None


class StreamLoader:
    """``batch`` document streams sliced into ``[B, unroll]`` windows; deterministic, resumable.

    Infinite when ``epochs`` is ``None`` (training: epochs follow each other with new
    permutations); finite otherwise (evaluation: ``shuffle=False`` and ``epochs=1`` visit every
    document once, idle rows pad and the loader stops when every row is idle). Window fields are
    those of :func:`~hsn.data.dataset.collate_windows`: ``tokens``, ``targets``, ``context``
    (int64), ``loss_mask``, ``reset`` (bool), ``dep`` (int32) and ``doc`` (int64, ``-1`` on
    padding).
    """

    def __init__(
        self,
        dataset: ByteDataset,
        *,
        batch: int,
        unroll: int,
        seed: int,
        shuffle: bool,
        align_docs: bool,
        epochs: int | None,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("StreamLoader needs at least one document")
        if batch < 1 or unroll < 1:
            raise ValueError(f"batch and unroll must be >= 1, got {batch} and {unroll}")
        if epochs is not None and epochs < 1:
            raise ValueError(f"epochs must be >= 1 or None, got {epochs}")
        self.dataset = dataset
        self.batch = batch
        self.unroll = unroll
        self.seed = seed
        self.shuffle = shuffle
        self.align_docs = align_docs
        self.epochs = epochs
        self.n_docs = len(dataset)
        self.epoch = 0
        self.cursor = 0  # next position in the permutation of the current epoch
        self.windows = 0  # windows emitted in the current epoch
        self.windows_total = 0
        self._rows = [_Row() for _ in range(batch)]
        self._perm = self._permutation(0)

    def _permutation(self, epoch: int) -> np.ndarray[Any, np.dtype[np.int64]]:
        if not self.shuffle:
            return np.arange(self.n_docs, dtype=np.int64)
        return np.random.default_rng([self.seed, epoch]).permutation(self.n_docs).astype(np.int64)

    def _exhausted(self) -> bool:
        return (
            self.epochs is not None and self.epoch + 1 >= self.epochs and self.cursor >= self.n_docs
        )

    def _next_doc(self) -> int | None:
        if self.cursor >= self.n_docs:
            if self._exhausted():
                return None
            self.epoch += 1
            self.cursor = 0
            self.windows = 0
            self._perm = self._permutation(self.epoch)
        index = int(self._perm[self.cursor])
        self.cursor += 1
        return index

    def _arrays(self, index: int) -> dict[str, np.ndarray[Any, Any]]:
        item = self.dataset[index]
        return {
            key: item[key].numpy() for key in ("tokens", "targets", "context", "loss_mask", "dep")
        }

    def state_dict(self) -> dict[str, Any]:
        """Enough to resume at the next window: epoch, cursor and ``(doc, offset)`` per row."""
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "windows": self.windows,
            "windows_total": self.windows_total,
            "n_docs": self.n_docs,
            "dataset_fingerprint": self.dataset.fingerprint,
            "rows": [[row.doc, row.offset] for row in self._rows],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["n_docs"]) != self.n_docs:
            raise ValueError(
                f"loader state is for {state['n_docs']} documents, dataset has {self.n_docs}"
            )
        fingerprint = state.get("dataset_fingerprint")
        if fingerprint is not None and fingerprint != self.dataset.fingerprint:
            raise ValueError(
                "loader dataset fingerprint differs: document contents or order changed"
            )
        rows = list(state["rows"])
        if len(rows) != self.batch:
            raise ValueError(f"loader state has {len(rows)} rows, loader has batch={self.batch}")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self.windows = int(state["windows"])
        self.windows_total = int(state["windows_total"])
        self._perm = self._permutation(self.epoch)
        self._rows = []
        for doc, offset in rows:
            row = _Row(int(doc), int(offset))
            if row.doc >= 0:
                row.arrays = self._arrays(row.doc)
                if not 0 <= row.offset < int(row.arrays["tokens"].shape[0]):
                    raise ValueError(
                        f"loader state offset {row.offset} is outside document {row.doc}"
                    )
            self._rows.append(row)

    def drop_documents(self) -> None:
        """P1.16: every row drops the rest of its current document, so the next window starts a
        fresh document in every row at ``t = 0`` (with its ``reset``); the epoch's permutation and
        cursor are unchanged (the dropped documents' remainders are never emitted)."""
        self._rows = [_Row() for _ in range(self.batch)]

    def __iter__(self) -> Iterator[Batch]:
        return self

    def __next__(self) -> Batch:
        if all(row.doc < 0 for row in self._rows) and self._exhausted():
            raise StopIteration
        B, T = self.batch, self.unroll
        tokens = np.full((B, T), PAD, dtype=np.int64)
        targets = np.full((B, T), PAD, dtype=np.int64)
        context = np.zeros((B, T), dtype=np.int64)
        loss_mask = np.zeros((B, T), dtype=np.bool_)
        reset = np.zeros((B, T), dtype=np.bool_)
        dep = np.zeros((B, T), dtype=np.int32)
        doc = np.full((B, T), -1, dtype=np.int64)
        for r, row in enumerate(self._rows):
            pos = 0
            while pos < T:
                if row.doc < 0:
                    if self.align_docs and pos > 0:
                        break  # the next document starts at the next window boundary
                    index = self._next_doc()
                    if index is None:
                        break  # finite loader, no documents left: pad the rest of the row
                    row.doc, row.offset, row.arrays = index, 0, self._arrays(index)
                assert row.arrays is not None
                length = int(row.arrays["tokens"].shape[0])
                take = min(T - pos, length - row.offset)
                lo, hi = row.offset, row.offset + take
                tokens[r, pos : pos + take] = row.arrays["tokens"][lo:hi]
                targets[r, pos : pos + take] = row.arrays["targets"][lo:hi]
                context[r, pos : pos + take] = row.arrays["context"][lo:hi]
                loss_mask[r, pos : pos + take] = row.arrays["loss_mask"][lo:hi]
                dep[r, pos : pos + take] = row.arrays["dep"][lo:hi]
                doc[r, pos : pos + take] = row.doc
                if row.offset == 0:
                    reset[r, pos] = True
                row.offset += take
                pos += take
                if row.offset >= length:
                    row.doc, row.offset, row.arrays = -1, 0, None
        self.windows += 1
        self.windows_total += 1
        return {
            "tokens": torch.from_numpy(tokens),
            "targets": torch.from_numpy(targets),
            "context": torch.from_numpy(context),
            "loss_mask": torch.from_numpy(loss_mask),
            "reset": torch.from_numpy(reset),
            "dep": torch.from_numpy(dep),
            "doc": torch.from_numpy(doc),
        }


# --------------------------------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------------------------------


@dataclass
class StepRecord:
    """What one optimiser step measured."""

    step: int
    loss: float  # mean cross-entropy over the step's loss positions (NaN when there were none)
    accuracy: float  # fraction of loss positions predicted correctly (NaN when there were none)
    lr: float
    grad_norm: float  # global gradient norm before clipping
    tokens: int  # non-padding positions in the step
    loss_positions: int
    seconds: float


@dataclass
class History:
    """The result of :meth:`Trainer.fit`."""

    param_report: dict[str, int]
    config_hash: str
    profile: str
    autocast: str
    steps: list[StepRecord] = field(default_factory=list)
    evals: list[dict[str, Any]] = field(default_factory=list)  # {"step": n, **evaluate(...)}
    checkpoints: list[str] = field(default_factory=list)
    measured: dict[str, float] = field(default_factory=dict)  # param_count, peak_bytes, ...

    @property
    def last_eval(self) -> dict[str, Any] | None:
        return self.evals[-1] if self.evals else None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


# --------------------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------------------


class Trainer:
    """The windowed training loop (module docstring) for any :class:`SequenceModel`.

    ``profiler`` is optional and duck-typed: an object with ``segment(name, tokens=0)`` returning a
    context manager (P0.6). ``run_dir`` receives checkpoints; it defaults to ``runs/<run_id>``.
    The trainer seeds every generator from ``cfg.seed`` when constructed, so dropout and any
    sampling inside the loop are reproducible; weight initialisation is the caller's (seed before
    building the model, as the runner does). ``train.deterministic`` is applied through
    :func:`hsn.device.set_deterministic`.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: RunConfig,
        ctx: DeviceContext,
        profiler: Any | None = None,
        *,
        run_dir: str | Path | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        if not isinstance(model, SequenceModel):
            raise TypeError(
                f"{type(model).__name__} does not implement the SequenceModel protocol "
                "(init_state, reset_state, forward, param_report)"
            )
        if profiler is not None and not callable(getattr(profiler, "segment", None)):
            raise TypeError("profiler must provide segment(name, tokens=0)")
        train = cfg.train
        if train.micro_batch != train.batch:
            raise NotImplementedError(
                f"train.micro_batch={train.micro_batch} != train.batch={train.batch}: "
                "micro-batching arrives with P4.3"
            )
        if train.optimizer not in ("adamw", "lazy_adamw"):
            raise ValueError(f"train.optimizer={train.optimizer!r} is not adamw or lazy_adamw")
        if train.schedule not in SCHEDULES:
            raise ValueError(f"train.schedule={train.schedule!r} is not one of {SCHEDULES}")
        self.model = model
        self.cfg = cfg
        self.ctx = ctx
        self.profiler = profiler
        self.run_dir = Path(run_dir) if run_dir is not None else Path("runs") / cfg.run_id
        self.config_hash = config_hash(cfg)
        set_deterministic(train.deterministic)
        seed_everything(cfg.seed)
        self.model.to(device=ctx.device, dtype=ctx.master_dtype)
        if ctx.compile_enabled and type(self) is Trainer:
            # P2.8: hardware.compile_mode compiles the HSN window (TBPTTTrainer); a baseline runs
            # eager and its history says so (measured["compiled_windows"] = 0)
            logger.warning(
                "hardware.compile_mode=%s compiles the HSN training window only; %s trains eager",
                cfg.hardware.compile_mode,
                type(self.model).__name__,
            )
        self.optimizer = self._make_optimizer()
        # P1.16: the rollback (train.rollback_*; None: off) and its learning-rate cut, which the
        # scheduler's factor reads (no cut until a rollback sets one)
        self.rollback: RollbackPolicy | None = RollbackPolicy.from_config(train)
        self._lr_cut_until = 0
        self._lr_cut_mult = 1.0
        self.adaptive_lr = (
            AdaptiveLR(
                train.adaptive_lr, {g["adaptive_group"] for g in self.optimizer.param_groups}
            )
            if train.adaptive_lr.enabled
            else None
        )
        self._adaptive_monitor: ByteDataset | None = None
        self.scheduler = self._make_scheduler()
        self.scaler = make_grad_scaler(ctx)
        if self.rollback is not None and self.scaler is not None:
            raise NotImplementedError(
                f"train.rollback_every={self.rollback.every} under an fp16 GradScaler: the scaler "
                "skips steps itself and the rollback's guards are not defined there"
            )
        self._ring: SnapshotRing | None = None  # P1.16: built at the first snapshot
        self.rollbacks: list[dict[str, Any]] = []  # P1.16: every rollback, in order
        self._rollback_windows: int | None = None  # P1.16: windows to discard after a rollback
        self.step = 0
        # P1.12: optimiser steps skipped by the non-finite-gradient guard, and the current run of
        # consecutive ones (the stop rule's)
        self.skipped_nonfinite = 0
        self._skip_streak = 0
        # P1.15: the gradient-spike guard (train.spike_factor; None: off), its skips and the history
        # of applied finite norms its median reads; every skipped step with its reason
        self.spike_factor: float | None = train.spike_factor
        self.spike_window = train.spike_window
        self.slow_burst_factor = train.slow_burst_factor
        self.slow_burst_window = train.slow_burst_window
        self._spike_history_limit = self.spike_window + (
            self.slow_burst_window if self.slow_burst_factor is not None else 0
        )
        if self.spike_factor is not None and self.scaler is not None:
            raise NotImplementedError(
                f"train.spike_factor={self.spike_factor} under an fp16 GradScaler: the scaler "
                "skips steps itself and the spike guard is not defined there"
            )
        self.skipped_spike = 0
        self._spike_history: collections.deque[float] = collections.deque(
            maxlen=self._spike_history_limit
        )
        self.skipped_steps: list[dict[str, Any]] = []
        # P1.15: the divergence dump (train.divergence_dump; hsn.train.divergence): the current
        # step's snapshot, the written dump's summary, notes the harness adds to it, and a hook
        # called with the trainer after a step's backward and before the clipping (the replay's;
        # None by default)
        self.divergence_dump: bool = train.divergence_dump
        self._step_snapshot: StepSnapshot | None = None
        self.divergence: dict[str, Any] | None = None
        self.dump_notes: dict[str, Any] = {}
        self.grad_hook: Callable[[Trainer], None] | None = None
        self._loader_state: dict[str, Any] | None = None
        self._carried_state: Any = None
        # P1.11: parallel streams of evaluate() (None = train.batch); a harness setting, not config
        self.eval_batch: int | None = None
        # P1.15: labels of evaluate()'s loss positions (the reads-state harness's query slots,
        # hsn.experiments.split); a harness setting, not config. None: today's evaluation
        self.eval_position_groups: PositionGroups | None = None
        # P1.16: an extra checkpoint every this many steps (the harness's protocol keep_every); a
        # harness setting, not config (train.ckpt_every is unchanged). None: none
        self.extra_ckpt_every: int | None = None

    # construction ----------------------------------------------------------------------------

    def _make_optimizer(self) -> torch.optim.Optimizer:
        optimizer = self._make_base_optimizer()
        train = self.cfg.train
        if train.router_lr_mult == train.readout_lr_mult == 1.0 and not train.adaptive_lr.enabled:
            return optimizer
        # Split the existing groups before any state or scheduler exists. This preserves the
        # cell's overrides, lazy unit metadata and ndim-based decay rules for both optimisers.
        by_id: dict[int, str] = {}
        multipliers = {"router": train.router_lr_mult, "decoder": train.readout_lr_mult}
        for name, mult in multipliers.items():
            if mult != 1.0 or train.adaptive_lr.enabled:
                module = getattr(self.model, name, None)
                if not isinstance(module, nn.Module):
                    if mult != 1.0:
                        raise ValueError(f"learning-rate multiplier requires HSN.{name}")
                    continue
                by_id.update((id(p), name) for p in module.parameters())
        grouped: dict[str, list[dict[str, Any]]] = {"shared": [], "router": [], "decoder": []}
        for group in optimizer.param_groups:
            buckets: dict[str, list[nn.Parameter]] = {}
            for p in group["params"]:
                buckets.setdefault(by_id.get(id(p), "shared"), []).append(p)
            for name, params in buckets.items():
                grouped[name].append(
                    {
                        **group,
                        "params": params,
                        "lr": group["lr"] * multipliers.get(name, 1.0),
                        **({"adaptive_group": name} if train.adaptive_lr.enabled else {}),
                    }
                )
        optimizer.param_groups.clear()
        for groups in grouped.values():
            for group in groups:
                optimizer.add_param_group(group)
        return optimizer

    def _make_base_optimizer(self) -> torch.optim.Optimizer:
        train = self.cfg.train
        if train.optimizer == "lazy_adamw":  # P2.6: the unit bank's tensors in unit groups
            return build_lazy_adamw(
                self.model,
                lr=train.lr,
                betas=train.betas,
                weight_decay=train.weight_decay,
                cell_lr_mult=train.cell_lr_mult,
                cell_weight_decay=train.cell_weight_decay,
            )
        if train.cell_lr_mult != 1.0 or train.cell_weight_decay is not None:
            return self._make_cell_grouped_adamw()  # P1.13
        decay = [p for p in self.model.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in self.model.parameters() if p.requires_grad and p.ndim < 2]
        if not decay and not no_decay:
            raise ValueError("the model has no trainable parameters")
        groups: list[dict[str, Any]] = []
        if decay:
            groups.append({"params": decay, "weight_decay": train.weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return torch.optim.AdamW(groups, lr=train.lr, betas=train.betas)

    def _make_cell_grouped_adamw(self) -> torch.optim.AdamW:
        """P1.13: AdamW with the HSN cell in groups of its own (module docstring, "Optimiser"):
        the other parameters first (so ``param_groups[0]`` keeps the global learning rate the
        step records log), decay before no-decay, empty groups left out."""
        train = self.cfg.train
        cell_ids = {id(p) for p in cell_parameters(self.model)}
        cell_decay = (
            train.weight_decay if train.cell_weight_decay is None else train.cell_weight_decay
        )
        groups: list[dict[str, Any]] = []
        for in_cell, lr, weight_decay in (
            (False, train.lr, train.weight_decay),
            (True, train.lr * train.cell_lr_mult, cell_decay),
        ):
            params = [
                p
                for p in self.model.parameters()
                if p.requires_grad and (id(p) in cell_ids) == in_cell
            ]
            decay = [p for p in params if p.ndim >= 2]
            no_decay = [p for p in params if p.ndim < 2]
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return torch.optim.AdamW(groups, lr=train.lr, betas=train.betas)

    def _make_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        train = self.cfg.train
        warmup, total, schedule = train.warmup_steps, train.steps, train.schedule
        # P1.13: the start of the WSD decay (the config validated the fraction against the warm-up)
        decay = 0 if train.decay_fraction is None else wsd_decay_steps(train.decay_fraction, total)
        decay_start = total - decay

        def factor(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            if schedule == "wsd":  # P1.13: constant, then 1 - sqrt over the last `decay` steps
                if step < decay_start:
                    return 1.0
                return 1.0 - math.sqrt(min(1.0, (step - decay_start) / decay))
            progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
            if schedule == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            if schedule == "linear":
                return 1.0 - progress
            return 1.0

        def cut(step: int) -> float:  # P1.16: the rollback's learning-rate cut (none by default)
            value = factor(step)
            if step < self._lr_cut_until:
                value *= self._lr_cut_mult
            return value

        if self.adaptive_lr is None:
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, cut)

        def adaptive_factor(group: str) -> Callable[[int], float]:
            def rate(step: int) -> float:
                assert self.adaptive_lr is not None
                return cut(step) * self.adaptive_lr.scale(group)

            return rate

        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            [adaptive_factor(g["adaptive_group"]) for g in self.optimizer.param_groups],
        )

    def _prepare_adaptive_monitor(self, train_ds: ByteDataset) -> None:
        if self.adaptive_lr is None:
            return
        # A dev monitor would make the controller itself a validation-trained component.
        if not len(train_ds) or any(e.meta.get("split") != "train" for e in train_ds.examples):
            raise ValueError("adaptive_lr requires a nonempty, explicitly TRAIN-labelled dataset")
        from hsn.experiments.data_selection import cap_dataset

        groups = {(e.meta["task"], e.meta["variant"]) for e in train_ds.examples}
        monitor = cap_dataset(
            train_ds, max(len(groups), self.cfg.train.adaptive_lr.monitor_docs), seed=self.cfg.seed
        )
        if not any(e.loss_mask.any() for e in monitor.examples):
            raise ValueError("adaptive_lr training monitor has no supervised positions")
        self.adaptive_lr.bind_monitor(monitor.fingerprint)
        self._adaptive_monitor = monitor

    def _adaptive_monitor_loss(self) -> float:
        assert self._adaptive_monitor is not None
        # Fresh loader/state and restored RNG: monitoring must not change the training stream,
        # carried state, previous byte, target encoder or subsequent stochastic updates.
        rng = capture_rng_state()
        try:
            seed_everything((self.cfg.seed ^ 0xA5AD) % (2**32))
            result = evaluate_windows(
                self._eval_model(),
                self._loader(self._adaptive_monitor, shuffle=False, epochs=1),
                DEFAULT_METRICS,
                device=self.ctx.device,
                autocast=autocast_ctx(self.ctx),
            )
            return float(result["loss"])
        finally:
            restore_rng_state(rng)

    def _adaptive_before_step(self, skipped: bool) -> list[tuple[Tensor, Tensor, str]]:
        if self.adaptive_lr is None or skipped or not self.adaptive_lr.due:
            return []
        return [
            (p, p.detach().clone(), group["adaptive_group"])
            for group in self.optimizer.param_groups
            for p in group["params"]
        ]

    def _adaptive_after_step(
        self, applied: bool, grad_norm: float, before: list[tuple[Tensor, Tensor, str]]
    ) -> None:
        controller = self.adaptive_lr
        if controller is None:
            return
        step = self.step + 1
        if not applied:
            controller.guard(step, "optimizer_step_skipped")
            logger.info("adaptive_lr %s", json.dumps(controller.state.observations[-1]))
            return
        if not controller.accepted(grad_norm):
            return
        delta: dict[str, list[Tensor]] = {}
        weight: dict[str, list[Tensor]] = {}
        with torch.no_grad():
            for p, previous, group in before:
                delta.setdefault(group, []).append(
                    (p.detach().float() - previous.float()).square().sum()
                )
                weight.setdefault(group, []).append(p.detach().float().square().sum())
            updates = {
                group: float(
                    (
                        torch.stack(delta[group]).sum()
                        / torch.stack(weight[group]).sum().clamp_min(1e-24)
                    ).sqrt()
                )
                for group in delta
            }
        observation = controller.observe(
            step, self._adaptive_monitor_loss(), updates, warmup=step <= self.cfg.train.warmup_steps
        )
        logger.info("adaptive_lr %s", json.dumps(observation))

    def _segment(self, name: str, tokens: int = 0) -> AbstractContextManager[object]:
        if self.profiler is None:
            return contextlib.nullcontext()
        segment: AbstractContextManager[object] = self.profiler.segment(name, tokens=tokens)
        return segment

    def _guard_nonfinite(self, grad_norm: float, step: int) -> bool:
        """The non-finite-gradient guard (P1.12, module docstring) for ``step`` with the global
        gradient norm ``grad_norm`` (before clipping). True means the step is skipped: the
        gradients are zeroed, the skip is counted and logged, and the caller leaves the optimiser
        alone (it still advances the scheduler). Raises (the stop rule) when the run keeps
        skipping. Under fp16 the ``GradScaler`` skips such steps itself: the guard returns False
        and touches nothing. P1.15: with the spike guard on, a finite norm leaves the run of
        consecutive skips to :meth:`_guard_spike`, which may still skip the step."""
        if self.scaler is not None:
            return False
        if math.isfinite(grad_norm):
            if self.spike_factor is None:
                self._skip_streak = 0
            return False
        self._on_skip(step, "nonfinite", grad_norm)  # P1.15: the dump, before the zeroing
        self.optimizer.zero_grad(set_to_none=True)
        if self._can_roll_back():  # P1.16: a rollback, not a skip
            self._roll_back(step, "nonfinite", grad_norm, None)
            return True
        self.skipped_nonfinite += 1
        self._skip_streak += 1
        self.skipped_steps.append(
            {"step": step, "reason": "nonfinite", "grad_norm": repr(grad_norm)}
        )
        logger.warning(
            "step %d: gradient norm %s is not finite; optimizer step skipped (%d skipped so far)",
            step,
            grad_norm,
            self.skipped_nonfinite,
        )
        self._stop_rule(step)
        return True

    def _guard_spike(self, grad_norm: float, step: int) -> bool:
        """P1.15's gradient-spike guard (module docstring) for ``step`` with the finite global
        gradient norm ``grad_norm`` (before clipping), when ``train.spike_factor`` is set. True
        means the step is skipped exactly as a non-finite one (:meth:`_guard_nonfinite`);
        otherwise the norm joins the history of applied norms. The skip counts in the P1.12 stop
        rule (:meth:`_stop_rule`)."""
        factor = self.spike_factor
        if factor is None:
            return False
        skip, median = spike_decision(grad_norm, self._spike_history, factor, self.spike_window)
        guard = "spike"
        if not skip and self.slow_burst_factor is not None:
            skip, median = slow_burst_decision(
                grad_norm,
                self._spike_history,
                self.slow_burst_factor,
                self.spike_window,
                self.slow_burst_window,
            )
            if skip:
                guard, factor = "slow_burst", self.slow_burst_factor
        if not skip:
            self._skip_streak = 0
            self._spike_history.append(float(grad_norm))
            return False
        self._on_skip(step, "spike", grad_norm)  # the dump, before the zeroing
        self.optimizer.zero_grad(set_to_none=True)
        if self._can_roll_back():  # P1.16: a rollback, not a skip
            self._roll_back(step, "spike", grad_norm, median)
            if guard == "slow_burst":
                self.rollbacks[-1]["guard"] = guard
                self.skipped_steps[-1]["guard"] = guard
            return True
        self.skipped_spike += 1
        self._skip_streak += 1
        self.skipped_steps.append(
            {"step": step, "reason": "spike", "grad_norm": float(grad_norm), "median": median}
        )
        if guard == "slow_burst":
            self.skipped_steps[-1]["guard"] = guard
        logger.warning(
            "step %d: gradient norm %.4g is %.3g times the median %.4g "
            "(%s factor %g); optimizer step skipped (%d spikes skipped so far)",
            step,
            grad_norm,
            grad_norm / median if median else math.inf,
            median,
            guard,
            factor,
            self.skipped_spike,
        )
        self._stop_rule(step)
        return True

    # P1.16: rollback -------------------------------------------------------------------------

    def _can_roll_back(self) -> bool:
        """A trigger rolls back when the rollback is on and a snapshot exists."""
        return self.rollback is not None and self._ring is not None and len(self._ring) > 0

    def take_rollback_snapshot(self) -> None:
        """Copy the current training state into the rollback ring (:mod:`hsn.train.rollback`);
        the fit loop calls it every ``rollback.every`` applied steps, a replay after a resume."""
        if self.rollback is None:
            raise RuntimeError("take_rollback_snapshot without a rollback policy")
        if self._ring is None:
            # A two-slot ring can already be inside a slow ramp by the time it is detected.
            # Retain enough snapshots to reach before the detector's older reference window.
            capacity = 2
            if self.slow_burst_factor is not None:
                capacity += math.ceil(self._spike_history_limit / self.rollback.every)
            self._ring = SnapshotRing(self.model, self.optimizer, capacity=capacity)
        self._ring.take(self.step, self._spike_history)

    def _roll_back(self, step: int, kind: str, grad_norm: float, median: float | None) -> None:
        """The rollback of ``step`` (module docstring, "Rollback"): raises at the limit, else
        restores the older snapshot in place, sets the learning-rate cut and the windows the loader
        discards before the next step, and records the event."""
        policy = self.rollback
        assert policy is not None and self._ring is not None
        what = "non-finite gradient norm" if kind == "nonfinite" else "gradient spike"
        if len(self.rollbacks) >= policy.max_rollbacks:
            raise RuntimeError(
                f"step {step}: a {what} would be rollback {len(self.rollbacks) + 1}, above "
                f"train.rollback_max={policy.max_rollbacks}. The run keeps diverging after "
                "rolling back: stopping "
                f"({ROLLBACK_LIMIT})"
            )
        snap = self._ring.restore()
        self._spike_history = collections.deque(
            snap.spike_history, maxlen=self._spike_history_limit
        )
        self._skip_streak = 0
        if policy.lr_steps:
            self._lr_cut_until = step + policy.lr_steps
            self._lr_cut_mult = policy.lr_mult
        windows = policy.skip * self.cfg.train.grad_accum
        self._rollback_windows = windows
        norm: float | str = float(grad_norm) if math.isfinite(grad_norm) else repr(grad_norm)
        event = {
            "step": int(step),
            "kind": kind,
            "grad_norm": norm,
            "median": median,
            "restored_step": int(snap.step),
            "lost_steps": int(step - 1 - snap.step),
            "windows_skipped": int(windows),
        }
        self.rollbacks.append(event)
        self.skipped_steps.append(
            {
                "step": int(step),
                "reason": "rollback",
                "trigger": kind,
                "grad_norm": norm,
                "median": median,
                "restored_step": int(snap.step),
            }
        )
        logger.warning(
            "step %d: %s (gradient norm %s); rolled back to the snapshot of step %d (%d steps "
            "undone), %d windows discarded, fresh documents%s (rollback %d of at most %d)",
            step,
            what,
            grad_norm,
            snap.step,
            step - 1 - snap.step,
            windows,
            f", lr x {policy.lr_mult:g} for {policy.lr_steps} steps" if policy.lr_steps else "",
            len(self.rollbacks),
            policy.max_rollbacks,
        )

    def _rollback_after_step(self, loader: StreamLoader, skipped: bool) -> None:
        """P1.16, after every step: a pending rollback's loader work (discard windows, fresh
        documents), else the snapshot of an applied step at the cadence."""
        if self.rollback is None:
            return
        if self._rollback_windows is not None:
            for _ in range(self._rollback_windows):
                next(loader)
            loader.drop_documents()
            self._rollback_windows = None
            return
        if not skipped and self.step % self.rollback.every == 0:
            self.take_rollback_snapshot()

    def _rollback_measured(self) -> dict[str, float]:
        """``rollbacks`` for ``History.measured``, only with the rollback on."""
        if self.rollback is None:
            return {}
        return {"rollbacks": float(len(self.rollbacks))}

    def _stop_rule(self, step: int) -> None:
        """P1.12's stop rule over every skipped step (P1.15: non-finite and spike skips together,
        decision D4): raises after :data:`NONFINITE_MAX_CONSECUTIVE` consecutive skips, or at
        :data:`NONFINITE_MIN_SKIPS` and more than :data:`NONFINITE_MAX_FRACTION` of the steps.
        Without spike skips the messages are P1.12's."""
        skipped = self.skipped_nonfinite + self.skipped_spike
        spikes = self.skipped_spike
        what = (
            "a non-finite gradient norm"
            if spikes == 0
            else f"a non-finite gradient norm or a gradient spike ({self.skipped_nonfinite} "
            f"non-finite, {spikes} gradient spikes)"
        )
        if self._skip_streak >= NONFINITE_MAX_CONSECUTIVE:
            raise RuntimeError(
                f"step {step}: {self._skip_streak} consecutive optimizer steps skipped for "
                f"{what} (limit {NONFINITE_MAX_CONSECUTIVE}); {skipped} of "
                f"{step} steps skipped so far. The run is not training: stopping (P1.12 stop rule)"
            )
        if skipped >= NONFINITE_MIN_SKIPS and skipped > NONFINITE_MAX_FRACTION * step:
            raise RuntimeError(
                f"step {step}: {skipped} of {step} steps skipped for {what} "
                f"(more than {NONFINITE_MAX_FRACTION:.0%} of the steps, and at least "
                f"{NONFINITE_MIN_SKIPS}). The run is not training reliably: stopping (P1.12 stop "
                "rule)"
            )

    def _guard_extra(self, counters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The guard's counts for a checkpoint's ``extra`` (P1.12; P1.15: with the spike guard on,
        its count and the history of applied norms, so that a resumed run continues it; from
        ``counters`` when given, the step snapshot's)."""
        c = self._counters() if counters is None else counters
        out: dict[str, Any] = {
            "skipped_nonfinite": int(c["skipped_nonfinite"]),
            "skipped_nonfinite_streak": int(c["skip_streak"]),
        }
        if self.spike_factor is not None:
            out["skipped_spike"] = int(c["skipped_spike"])
            out["spike_history"] = [float(x) for x in c["spike_history"]]
        if self.rollback is not None:  # P1.16: the events and the learning-rate cut
            out["rollbacks"] = [dict(r) for r in c["rollbacks"]]
            out["rollback_lr_cut"] = [int(c["lr_cut"][0]), float(c["lr_cut"][1])]
        if self.adaptive_lr is not None:
            out["adaptive_lr"] = c["adaptive_lr"]
        return out

    def _restore_guard(self, extra: Mapping[str, Any], path: str) -> None:
        """The guard's counts from a checkpoint's ``extra``: 0 when written before the guard;
        a malformed count, or more consecutive skips than skips, raises. P1.15: the spike guard's
        count and history (absent: 0 and empty; a malformed history raises)."""
        if self.adaptive_lr is not None:
            raw = extra.get("adaptive_lr")
            if not isinstance(raw, dict):
                raise ValueError(f"{path}: adaptive_lr checkpoint state is missing")
            self.adaptive_lr.load_state_dict(raw)
        skipped = _skipped_counter(extra, "skipped_nonfinite", path)
        streak = _skipped_counter(extra, "skipped_nonfinite_streak", path)
        spikes = 0
        history: list[float] = []
        if self.spike_factor is not None:
            spikes = _skipped_counter(extra, "skipped_spike", path)
            raw = extra.get("spike_history", [])
            if (
                not isinstance(raw, list)
                or len(raw) > self._spike_history_limit
                or not all(isinstance(x, float) and math.isfinite(x) and x >= 0.0 for x in raw)
            ):
                raise ValueError(
                    f"{path}: spike_history must be a list of at most "
                    f"{self._spike_history_limit} finite "
                    "non-negative floats"
                )
            history = list(raw)
        if streak > skipped + spikes:
            raise ValueError(
                f"{path}: skipped_nonfinite_streak={streak} exceeds skipped_nonfinite={skipped}"
                + (f" + skipped_spike={spikes}" if spikes else "")
            )
        self.skipped_nonfinite = skipped
        self.skipped_spike = spikes
        self._spike_history = collections.deque(history, maxlen=self._spike_history_limit)
        self._skip_streak = streak
        if self.rollback is not None:  # P1.16 (absent: written without the rollback)
            events = extra.get("rollbacks", [])
            cut = extra.get("rollback_lr_cut", [0, 1.0])
            if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
                raise ValueError(f"{path}: rollbacks must be a list of event dicts")
            if (
                not isinstance(cut, list | tuple)
                or len(cut) != 2
                or isinstance(cut[0], bool)
                or not isinstance(cut[0], int)
                or not isinstance(cut[1], float | int)
            ):
                raise ValueError(f"{path}: rollback_lr_cut must be [until_step, mult]")
            self.rollbacks = [dict(e) for e in events]
            self._lr_cut_until, self._lr_cut_mult = int(cut[0]), float(cut[1])

    def _loader(
        self, ds: ByteDataset, *, shuffle: bool, epochs: int | None, batch: int | None = None
    ) -> StreamLoader:
        if batch is not None and batch < 1:
            raise ValueError(f"loader batch must be >= 1, got {batch}")
        return StreamLoader(
            ds,
            batch=self.cfg.train.batch if batch is None else batch,
            unroll=self.cfg.train.unroll,
            seed=self.cfg.seed,
            shuffle=shuffle,
            align_docs=self.cfg.data.align_docs,
            epochs=epochs,
        )

    def _to_device(self, window: Batch) -> Batch:
        pin = self.cfg.hardware.pinned_memory and self.ctx.device.type == "cuda"
        out: Batch = {}
        for key in _WINDOW_FIELDS:
            value = window[key]
            out[key] = (
                value.pin_memory().to(self.ctx.device, non_blocking=True)
                if pin
                else value.to(self.ctx.device)
            )
        return out

    def _refuse_test(self, ds: ByteDataset, allow_test: bool) -> None:
        if allow_test:
            return
        for i in range(len(ds)):
            if ds.meta(i).get("split") == "test":
                raise ValueError(
                    "the test split is reserved for gate scripts (chunk P0.5 'Do not'); pass "
                    "allow_test=True only from one"
                )

    def _forward(
        self,
        batch: Batch,
        state: Any,
        *,
        window: Callable[..., Any] | None = None,
        **model_kwargs: Any,
    ) -> tuple[Tensor, Any, dict[str, Any]]:
        """The model over one device window; ``model_kwargs`` go to its forward. This trainer's
        own :meth:`fit` passes none, so a baseline's forward is called exactly as before; only the
        HSN trainer (:class:`~hsn.train.tbptt.TBPTTTrainer`, P1.10b) passes ``reset_columns`` from
        the CPU window, and (P2.8) ``window``, a compiled window called in place of the model's
        forward with the same arguments."""
        tokens = batch["tokens"]
        B, T = tokens.shape
        fn = self.model if window is None else window
        with self._segment("forward", tokens=B * T), autocast_ctx(self.ctx):
            logits, new_state, info = fn(
                tokens, batch["context"], state, batch["reset"], **model_kwargs
            )
        V = self.cfg.model.vocab_size
        if not isinstance(logits, Tensor) or tuple(logits.shape) != (B, T, V):
            got = tuple(logits.shape) if isinstance(logits, Tensor) else type(logits).__name__
            raise ValueError(f"model returned logits {got}, expected [B, T, V] = {(B, T, V)}")
        if not isinstance(info, dict):
            raise TypeError(f"model info must be a dict, got {type(info).__name__}")
        return logits, new_state, info

    # fit -------------------------------------------------------------------------------------

    def fit(
        self,
        train_ds: ByteDataset,
        dev_ds: ByteDataset,
        steps: int,
        *,
        metrics: Mapping[str, Metric] | None = None,
        callback: FitCallback | None = None,
    ) -> History:
        """Train until ``self.step == steps`` (so a resumed trainer continues where it stopped), or
        until ``callback`` (module docstring, "Hooks") returns True."""
        train = self.cfg.train
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if steps > train.steps:
            raise ValueError(
                f"steps={steps} exceeds the schedule horizon train.steps={train.steps}; raise "
                "train.steps instead of running past the learning-rate schedule"
            )
        if steps < self.step:
            raise ValueError(f"steps={steps} is before the current step {self.step}")
        self._refuse_test(train_ds, False)
        self._refuse_test(dev_ds, False)
        self._prepare_adaptive_monitor(train_ds)
        report = self.model.param_report()
        for key in ("active", "stored", "trainable"):
            if key not in report:
                raise KeyError(f"param_report() lacks {key!r}: {sorted(report)}")
        history = History(
            param_report=dict(report),
            config_hash=self.config_hash,
            profile=self.ctx.profile,
            autocast=autocast_name(self.ctx),
        )
        logger.info(
            "run %s on %s (%s, autocast %s): params %s",
            self.cfg.run_id,
            self.ctx.device,
            self.ctx.profile,
            history.autocast,
            json.dumps(report, sort_keys=True),
        )
        loader = self._loader(train_ds, shuffle=True, epochs=None)
        if self._loader_state is not None:
            loader.load_state_dict(self._loader_state)
            self._loader_state = None
        device = self.ctx.device
        batch_size = train.batch
        state: Any = self._carried_state
        self._carried_state = None
        if state is None:
            state = self.model.init_state(batch_size, device)
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        max_norm = train.grad_clip if train.grad_clip > 0.0 else math.inf
        reset_peak(device)
        train_seconds = 0.0
        total_tokens = 0
        while self.step < steps:
            started = time.perf_counter()
            snapshot = self._step_snapshot = self._take_snapshot(loader, state)  # P1.15
            windows = [next(loader) for _ in range(train.grad_accum)]
            counts = [int(window["loss_mask"].sum()) for window in windows]
            step_loss_positions = sum(counts)
            if snapshot is not None:
                snapshot.windows.extend(windows)
            self.optimizer.zero_grad(set_to_none=True)
            # host-side float64 accumulator: MPS has no float64 (P0.8 fix for profile:m_series)
            loss_sum = torch.zeros((), dtype=torch.float64)
            correct = torch.zeros((), dtype=torch.int64, device=device)
            n_loss = 0
            tokens_in_step = 0
            for window, count in zip(windows, counts, strict=True):
                batch = self._to_device(window)
                state = detach_state(state)
                logits, state, _info = self._forward(batch, state)
                loss_mask = batch["loss_mask"]
                tokens_in_step += int((window["doc"] >= 0).sum())
                if count > 0:
                    flat_logits = logits[loss_mask].float()
                    flat_targets = batch["targets"][loss_mask]
                    loss = F.cross_entropy(flat_logits, flat_targets)
                    correct += (flat_logits.argmax(dim=-1) == flat_targets).sum()
                    loss_sum += loss.detach().cpu().double() * count
                    n_loss += count
                else:  # no loss positions in this window (padding only): zero gradient
                    loss = logits.float().sum() * 0.0
                scaled = (
                    loss / train.grad_accum
                    if train.grad_accum == 1
                    else loss * (count / step_loss_positions if step_loss_positions else 0.0)
                )
                with self._segment("backward", tokens=tokens_in_step):
                    if self.scaler is not None:
                        scaled = self.scaler.scale(scaled)
                    scaled.backward()  # type: ignore[no-untyped-call]
            if snapshot is not None:  # P1.15: the dump records the step's loss
                snapshot.loss = float(loss_sum) / n_loss if n_loss else math.nan
            with self._segment("optimizer_step"):
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                if self.grad_hook is not None:  # P1.15: the replay's raw gradients
                    self.grad_hook(self)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm))
                lr = float(self.optimizer.param_groups[0]["lr"])
                skipped = self._guard_nonfinite(grad_norm, self.step + 1)  # P1.12
                if not skipped and self.spike_factor is not None:  # P1.15
                    skipped = self._guard_spike(grad_norm, self.step + 1)
                before = self._adaptive_before_step(skipped)
                applied = not skipped
                if self.scaler is not None:
                    scale_before = self.scaler.get_scale() if self.adaptive_lr is not None else None
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if scale_before is not None:
                        applied = applied and self.scaler.get_scale() >= scale_before
                elif not skipped:
                    self.optimizer.step()
                self._adaptive_after_step(applied, grad_norm, before)
                self.scheduler.step()
            self.step += 1
            self._rollback_after_step(loader, skipped)  # P1.16
            mean_loss = float(loss_sum) / n_loss if n_loss else float("nan")
            accuracy = float(correct) / n_loss if n_loss else float("nan")
            if self.scaler is None and n_loss and not math.isfinite(mean_loss) and not skipped:
                raise RuntimeError(f"loss is {mean_loss} at step {self.step}")
            seconds = time.perf_counter() - started
            train_seconds += seconds
            total_tokens += tokens_in_step
            history.steps.append(
                StepRecord(
                    step=self.step,
                    loss=mean_loss,
                    accuracy=accuracy,
                    lr=lr,
                    grad_norm=grad_norm,
                    tokens=tokens_in_step,
                    loss_positions=n_loss,
                    seconds=seconds,
                )
            )
            if self.step % train.stats_every == 0 or self.step == steps:
                logger.info(
                    "step %d/%d loss %.4f acc %.4f lr %.2e grad_norm %.3f tokens/s %.0f",
                    self.step,
                    steps,
                    mean_loss,
                    accuracy,
                    lr,
                    grad_norm,
                    tokens_in_step / seconds if seconds > 0 else 0.0,
                )
            if self.step % train.eval_every == 0 or self.step == steps:
                result = self.evaluate(dev_ds, metrics)
                history.evals.append({"step": self.step, **result})
                logger.info(
                    "eval step %d: %s",
                    self.step,
                    " ".join(f"{k} {v:.4f}" for k, v in result.items() if isinstance(v, float)),
                )
            saved = False
            if self.step % train.ckpt_every == 0 or self.step == steps or self._kept_step():
                path = self.save(self.run_dir, loader, state)
                history.checkpoints.append(str(path))
                saved = True
            if callback is not None and self.step < steps and callback(self, history):
                if not saved:
                    history.checkpoints.append(str(self.save(self.run_dir, loader, state)))
                logger.info("stopped by callback at step %d of %d", self.step, steps)
                break
        self._loader_state = loader.state_dict()
        self._carried_state = detach_state(state)
        synchronize(device)
        stats = memory_stats(device)
        history.measured = {
            "param_count": float(report["stored"]),
            "active_param_count": float(report["active"]),
            "peak_bytes": float(stats["peak"]),
            "reserved_bytes": float(stats["reserved"]),
            "tokens_per_s": total_tokens / train_seconds if train_seconds > 0 else 0.0,
            "train_seconds": train_seconds,
            "tokens": float(total_tokens),
            "steps": float(self.step),
            "compiled_windows": 0.0,  # P2.8: the generic trainer never compiles
            "skipped_nonfinite_steps": float(self.skipped_nonfinite),  # P1.12
            **self._spike_measured(),  # P1.15
            **self._rollback_measured(),  # P1.16
        }
        return history

    def _kept_step(self) -> bool:
        """P1.16: whether this step writes a kept checkpoint (:attr:`extra_ckpt_every`)."""
        every = self.extra_ckpt_every
        return every is not None and every > 0 and self.step % every == 0

    def _spike_measured(self) -> dict[str, float]:
        """P1.15: ``skipped_spike_steps`` for ``History.measured``, only with the guard on (a rule
        on it reads missing, never a false 0, when the guard did not run)."""
        if self.spike_factor is None:
            return {}
        return {"skipped_spike_steps": float(self.skipped_spike)}

    # evaluation ------------------------------------------------------------------------------

    def evaluate(
        self,
        ds: ByteDataset,
        metrics: Mapping[str, Metric] | None = None,
        *,
        allow_test: bool = False,
    ) -> dict[str, Any]:
        """Metrics over every document of ``ds`` (one pass, no gradients), overall and per task.

        Per-task entries live under ``"per_group"`` keyed ``"<task>/<variant>"``. The test split
        raises unless ``allow_test`` (gate scripts only).
        """
        self._refuse_test(ds, allow_test)
        if len(ds) == 0:
            raise ValueError("evaluate: the dataset is empty")
        names: list[str] = []
        ids: list[int] = []
        for i in range(len(ds)):
            meta = ds.meta(i)
            group = f"{meta.get('task', '?')}/{meta.get('variant', '?')}"
            if group not in names:
                names.append(group)
            ids.append(names.index(group))
        loader = self._loader(ds, shuffle=False, epochs=1, batch=self.eval_batch)
        with self._segment("evaluate"):
            result = evaluate_windows(
                self._eval_model(),
                loader,
                metrics if metrics is not None else DEFAULT_METRICS,
                device=self.ctx.device,
                autocast=autocast_ctx(self.ctx),
                group_of_doc=torch.tensor(ids, dtype=torch.int64),
                group_names=names,
                position_groups=self.eval_position_groups,  # P1.15 (None: no extra groups)
            )
        result["n_docs"] = len(ds)
        return result

    def _eval_model(self) -> Any:
        """What :meth:`evaluate` runs over the windows: the model itself (P2.8: the HSN trainer
        hands its compiled window when it has one)."""
        return self.model

    # checkpoints -----------------------------------------------------------------------------

    def save(self, run_dir: str | Path, loader: StreamLoader, state: Any) -> Path:
        """Write ``step_<step>.pt`` into ``run_dir`` with everything a bit-exact resume needs."""
        path = checkpoint_path(run_dir, self.step)
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.step,
            capture_rng_state(),
            loader.state_dict(),
            extra=self._checkpoint_extra(state),
        )
        logger.info("checkpoint %s", path)
        return path

    def _checkpoint_extra(
        self, state: Any, counters: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """A checkpoint's ``extra``: the configuration's identity, the scaler, the carried state
        and the guards' counts (P1.15: ``counters``, a :meth:`_counters` dict, replaces the
        current ones; the divergence dump writes the step's start)."""
        return {
            "config_hash": self.config_hash,
            "model_name": self.cfg.model_name,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "carried_state": _map_state(detach_state(state), lambda t: t.cpu()),
            **self._guard_extra(counters),  # P1.12
        }

    def _counters(self) -> dict[str, Any]:
        """P1.15: the trainer's counters a checkpoint carries (copies), for the step snapshot."""
        return {
            "skipped_nonfinite": self.skipped_nonfinite,
            "skip_streak": self._skip_streak,
            "skipped_spike": self.skipped_spike,
            "spike_history": [float(x) for x in self._spike_history],
            "rollbacks": [dict(r) for r in self.rollbacks],  # P1.16
            "lr_cut": (self._lr_cut_until, self._lr_cut_mult),
            **(
                {"adaptive_lr": self.adaptive_lr.state_dict()}
                if self.adaptive_lr is not None
                else {}
            ),
        }

    def _snapshot_state(self, state: Any) -> Any:
        """P1.15: a device copy of the carried state (the step snapshot's)."""
        return _map_state(detach_state(state), lambda t: t.clone())

    def _take_snapshot(self, loader: StreamLoader, state: Any) -> StepSnapshot | None:
        """P1.15: the step snapshot (:mod:`hsn.train.divergence`) while a dump may still be
        written, else None (the dump off, already written, or an fp16 scaler: no guard)."""
        if not self.divergence_dump or self.divergence is not None or self.scaler is not None:
            return None
        return StepSnapshot(
            step=self.step,
            loader_state=loader.state_dict(),
            rng_state=capture_rng_state(),
            state=self._snapshot_state(state),
            buffers=persistent_buffers(self.model),
            counters=self._counters(),
        )

    def _on_skip(self, step: int, kind: str, grad_norm: float) -> None:
        """P1.15: called by a guard that skips ``step``, before it zeroes the gradients: the
        divergence dump of the first such step (:func:`hsn.train.divergence.write_dump`)."""
        snapshot = self._step_snapshot
        if snapshot is None or self.divergence is not None:
            return
        write_dump(self, snapshot, step=step, kind=kind, grad_norm=grad_norm)

    def resume(self, path: str | Path) -> int:
        """Load a checkpoint written by :meth:`save` for this configuration; returns its step."""
        loaded = load_checkpoint(path, self.model, self.optimizer, self.scheduler)
        extra = loaded["extra"]
        if extra.get("config_hash") != self.config_hash:
            raise ValueError(
                f"{path} was written for configuration {extra.get('config_hash')}, this trainer "
                f"runs {self.config_hash}; load_checkpoint() loads weights across configurations"
            )
        if self.scaler is not None:
            if extra.get("scaler") is None:
                raise ValueError(f"{path} has no loss-scaler state but this profile uses fp16")
            self.scaler.load_state_dict(extra["scaler"])
        restore_rng_state(loaded["rng_state"])
        if loaded["loader_state"] is None:
            raise ValueError(f"{path} has no loader state")
        self._loader_state = dict(loaded["loader_state"])
        self._restore_guard(extra, str(path))  # P1.12
        device = self.ctx.device
        self._carried_state = _map_state(extra.get("carried_state"), lambda t: t.to(device))
        self.step = int(loaded["step"])
        return self.step


# --------------------------------------------------------------------------------------------------
# Search budget
# --------------------------------------------------------------------------------------------------


class BudgetExhaustedError(RuntimeError):
    """Raised when a trial would exceed ``trials_allowed``."""


@dataclass
class Trial:
    key: str  # config hash
    config: dict[str, Any] | None  # the trial's configuration (resolved dump), when it was one
    result: dict[str, float]


class SearchBudget:
    """Counts hyperparameter-search trials so every method gets the same number (paper §12).

    ``record(trial_cfg, result)`` stores one trial; ``trial_cfg`` is a :class:`RunConfig` (keyed
    by its config hash) or a mapping of the hyperparameters tried (keyed by its canonical JSON
    hash). The same configuration twice is a wasted trial and raises; a trial beyond the budget
    raises :class:`BudgetExhaustedError`. ``exhausted()`` tells a search loop to stop.
    """

    def __init__(self, trials_allowed: int) -> None:
        if isinstance(trials_allowed, bool) or trials_allowed < 1:
            raise ValueError(f"trials_allowed must be an int >= 1, got {trials_allowed!r}")
        self.trials_allowed = int(trials_allowed)
        self.trials: list[Trial] = []

    @classmethod
    def from_config(cls, cfg: RunConfig) -> SearchBudget:
        return cls(cfg.train.search_trials)

    @property
    def trials_used(self) -> int:
        return len(self.trials)

    def remaining(self) -> int:
        return self.trials_allowed - len(self.trials)

    def exhausted(self) -> bool:
        return len(self.trials) >= self.trials_allowed

    def record(
        self, trial_cfg: RunConfig | Mapping[str, Any], result: Mapping[str, float]
    ) -> Trial:
        if self.exhausted():
            raise BudgetExhaustedError(
                f"search budget of {self.trials_allowed} trials is exhausted; a further trial "
                "would give this method more trials than the others"
            )
        if isinstance(trial_cfg, RunConfig):
            key = config_hash(trial_cfg)
            config: dict[str, Any] | None = trial_cfg.model_dump(mode="json")
        elif isinstance(trial_cfg, Mapping):
            config = dict(trial_cfg)
            key = hashlib.blake2b(
                json.dumps(config, sort_keys=True, default=str).encode(), digest_size=16
            ).hexdigest()
        else:
            raise TypeError(f"trial_cfg must be a RunConfig or a mapping, got {type(trial_cfg)}")
        if any(trial.key == key for trial in self.trials):
            raise ValueError(f"trial {key} was already recorded; repeating it wastes a trial")
        checked: dict[str, float] = {}
        for name, value in result.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"result[{name!r}] must be a number, got {type(value).__name__}")
            checked[name] = float(value)
        trial = Trial(key=key, config=config, result=checked)
        self.trials.append(trial)
        return trial

    def best(self, metric: str, *, maximize: bool) -> Trial:
        """The recorded trial with the best ``metric`` (NaN results never win)."""
        if not self.trials:
            raise ValueError("no trials recorded")
        candidates = [
            t for t in self.trials if metric in t.result and math.isfinite(t.result[metric])
        ]
        if not candidates:
            raise KeyError(f"no trial has a finite result for {metric!r}")
        return max(candidates, key=lambda t: t.result[metric] if maximize else -t.result[metric])

    def to_json(self) -> str:
        return json.dumps(
            {
                "trials_allowed": self.trials_allowed,
                "trials": [asdict(trial) for trial in self.trials],
            },
            indent=2,
            sort_keys=True,
        )
