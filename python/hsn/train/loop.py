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
``loss_mask`` is True, gradients are clipped to ``train.grad_clip`` (global norm), AMP follows the
device context (:mod:`hsn.train.amp`), the learning rate follows ``train.schedule`` with
``train.warmup_steps`` over the horizon ``train.steps``. Every ``train.eval_every`` steps and at
the end the dev set is evaluated (:func:`hsn.eval.evaluate.evaluate_windows`); every
``train.ckpt_every`` steps and at the end a checkpoint is written to ``run_dir``
(``runs/<run_id>`` unless given). ``fit``
returns a :class:`History` with one record per step, every evaluation, the model's parameter
report and the *measured* numbers the run manifest needs (plan §0.5: measured, not assumed).

**Search budget.** :class:`SearchBudget` counts hyperparameter-search trials against
``train.search_trials`` so that every method in the baseline ladder gets the same number of
trials (paper §12); recording a trial beyond the budget raises.

**Guards.** The test split is refused unless a caller says ``allow_test=True`` (gate scripts only;
chunk "Do not"). Shape mismatches, unknown schedules, a non-finite loss under bf16/fp32, a
checkpoint from another configuration and features that belong to later chunks (micro-batching,
P4.3) raise instead of being reinterpreted (plan §4.3: errors over silence).

**Optimiser (P2.6).** ``train.optimizer=adamw`` builds ``torch.optim.AdamW`` with weight decay on
the tensors of ``ndim >= 2`` and none on the rest; ``train.optimizer=lazy_adamw`` builds the same
groups as a :class:`~hsn.train.lazy_adam.LazyAdamW` with the unit bank's stacked tensors in unit
groups, and needs a model with a :class:`~hsn.model.units.UnitBank` (the baselines have none and
raise). The TBPTT trainer hands the lazy optimiser the units each step executed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import RunConfig, config_hash
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
from hsn.eval.evaluate import DEFAULT_METRICS, Metric, evaluate_windows
from hsn.train.amp import autocast_ctx, autocast_name, make_grad_scaler
from hsn.train.checkpoint import (
    capture_rng_state,
    checkpoint_path,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from hsn.train.lazy_adam import build_lazy_adamw

__all__ = [
    "SCHEDULES",
    "BudgetExhaustedError",
    "History",
    "SearchBudget",
    "SequenceModel",
    "StepRecord",
    "StreamLoader",
    "Trainer",
    "Trial",
    "detach_state",
]

logger = logging.getLogger("hsn.train")

Batch = dict[str, Tensor]
SCHEDULES: tuple[str, ...] = ("constant", "cosine", "linear")
_WINDOW_FIELDS = ("tokens", "targets", "context", "loss_mask", "reset", "dep", "doc")


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
            "rows": [[row.doc, row.offset] for row in self._rows],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["n_docs"]) != self.n_docs:
            raise ValueError(
                f"loader state is for {state['n_docs']} documents, dataset has {self.n_docs}"
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
        self.optimizer = self._make_optimizer()
        self.scheduler = self._make_scheduler()
        self.scaler = make_grad_scaler(ctx)
        self.step = 0
        self._loader_state: dict[str, Any] | None = None
        self._carried_state: Any = None

    # construction ----------------------------------------------------------------------------

    def _make_optimizer(self) -> torch.optim.Optimizer:
        train = self.cfg.train
        if train.optimizer == "lazy_adamw":  # P2.6: the unit bank's tensors in unit groups
            return build_lazy_adamw(
                self.model, lr=train.lr, betas=train.betas, weight_decay=train.weight_decay
            )
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

    def _make_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        train = self.cfg.train
        warmup, total, schedule = train.warmup_steps, train.steps, train.schedule

        def factor(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
            if schedule == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            if schedule == "linear":
                return 1.0 - progress
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)

    def _segment(self, name: str, tokens: int = 0) -> AbstractContextManager[object]:
        if self.profiler is None:
            return contextlib.nullcontext()
        segment: AbstractContextManager[object] = self.profiler.segment(name, tokens=tokens)
        return segment

    def _loader(self, ds: ByteDataset, *, shuffle: bool, epochs: int | None) -> StreamLoader:
        return StreamLoader(
            ds,
            batch=self.cfg.train.batch,
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
    ) -> History:
        """Train until ``self.step == steps`` (so a resumed trainer continues where it stopped)."""
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
            self.optimizer.zero_grad(set_to_none=True)
            # host-side float64 accumulator: MPS has no float64 (P0.8 fix for profile:m_series)
            loss_sum = torch.zeros((), dtype=torch.float64)
            correct = torch.zeros((), dtype=torch.int64, device=device)
            n_loss = 0
            tokens_in_step = 0
            for _ in range(train.grad_accum):
                window = next(loader)
                batch = self._to_device(window)
                state = detach_state(state)
                logits, state, _info = self._forward(batch, state)
                loss_mask = batch["loss_mask"]
                count = int(loss_mask.sum())
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
                scaled = loss / train.grad_accum
                with self._segment("backward", tokens=tokens_in_step):
                    if self.scaler is not None:
                        scaled = self.scaler.scale(scaled)
                    scaled.backward()  # type: ignore[no-untyped-call]
            with self._segment("optimizer_step"):
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm))
                lr = float(self.optimizer.param_groups[0]["lr"])
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
            self.step += 1
            mean_loss = float(loss_sum) / n_loss if n_loss else float("nan")
            accuracy = float(correct) / n_loss if n_loss else float("nan")
            if self.scaler is None and n_loss and not math.isfinite(mean_loss):
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
            if self.step % train.ckpt_every == 0 or self.step == steps:
                path = self.save(self.run_dir, loader, state)
                history.checkpoints.append(str(path))
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
        }
        return history

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
        loader = self._loader(ds, shuffle=False, epochs=1)
        with self._segment("evaluate"):
            result = evaluate_windows(
                self.model,
                loader,
                metrics if metrics is not None else DEFAULT_METRICS,
                device=self.ctx.device,
                autocast=autocast_ctx(self.ctx),
                group_of_doc=torch.tensor(ids, dtype=torch.int64),
                group_names=names,
            )
        result["n_docs"] = len(ds)
        return result

    # checkpoints -----------------------------------------------------------------------------

    def save(self, run_dir: str | Path, loader: StreamLoader, state: Any) -> Path:
        """Write ``step_<step>.pt`` into ``run_dir`` with everything a bit-exact resume needs."""
        path = checkpoint_path(run_dir, self.step)
        extra: dict[str, Any] = {
            "config_hash": self.config_hash,
            "model_name": self.cfg.model_name,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "carried_state": _map_state(detach_state(state), lambda t: t.cpu()),
        }
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.step,
            capture_rng_state(),
            loader.state_dict(),
            extra=extra,
        )
        logger.info("checkpoint %s", path)
        return path

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
