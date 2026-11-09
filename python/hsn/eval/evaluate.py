"""Token-level evaluation (P0.5): metrics over loss positions, overall and per task/variant.

A *metric* maps the logits and targets of the positions where the loss mask is True —
``(logits [N, V] float32, targets [N] int64) -> values [N] float32`` — and is averaged over those
positions. :func:`nll` (next-symbol cross-entropy in nats) and :func:`token_accuracy` (1.0 where the
arg-max equals the target) are the defaults; :func:`bits_per_byte` is ``nll / ln 2``. Only loss
positions count: P0.4 marks exactly the positions whose next symbol is a task answer, so that the
free-choice symbols of a generator never inflate accuracy.

:func:`evaluate_windows` runs a :class:`~hsn.train.loop.SequenceModel` over a finite stream of
windows (the ``[B, T]`` batches :func:`hsn.data.dataset.collate_windows` and the trainer's loader
produce), carrying the recurrent state between consecutive windows exactly as training does, and
returns the averages overall and per *group* (``"<task>/<variant>"`` by default, taken from the
``doc`` index of every position through ``group_of_doc``). Every accuracy is a fraction in
``[0, 1]`` (the unit :mod:`hsn.eval.metrics` expects); counts are integers.

**Position groups (P1.15).** ``position_groups`` (optional, :class:`PositionGroups`) labels every
loss position of a pass, for example by query slot (:class:`hsn.experiments.split.QuerySlots`); a
second accumulator averages the same per-position values by those labels, reported under
``"per_position_group"``. The overall and per-group results are computed exactly as without it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "DEFAULT_METRICS",
    "Metric",
    "MetricAccumulator",
    "PositionGroups",
    "bits_per_byte",
    "evaluate_windows",
    "nll",
    "token_accuracy",
]

Metric = Callable[[Tensor, Tensor], Tensor]
Batch = dict[str, Tensor]


class PositionGroups(Protocol):
    """P1.15: labels of loss positions. ``names`` are the groups; :meth:`start` gives the labeller
    of one pass, which maps a CPU window (the loader's fields) to ``[B, T]`` int64 indices into
    ``names`` (every loss position labelled; values elsewhere are ignored)."""

    @property
    def names(self) -> Sequence[str]: ...

    def start(self) -> Callable[[Mapping[str, Tensor]], Tensor]: ...


def nll(logits: Tensor, targets: Tensor) -> Tensor:
    """Per-position next-symbol cross-entropy in nats (float32)."""
    return F.cross_entropy(logits.float(), targets, reduction="none")


def bits_per_byte(logits: Tensor, targets: Tensor) -> Tensor:
    """Per-position cross-entropy in bits (byte-level models: bits per byte)."""
    return nll(logits, targets) / math.log(2.0)


def token_accuracy(logits: Tensor, targets: Tensor) -> Tensor:
    """1.0 where the arg-max prediction equals the target, else 0.0 (float32)."""
    return (logits.argmax(dim=-1) == targets).to(torch.float32)


DEFAULT_METRICS: dict[str, Metric] = {"loss": nll, "accuracy": token_accuracy}


class MetricAccumulator:
    """Running sums of every metric over loss positions, overall and per group."""

    def __init__(self, metrics: Mapping[str, Metric], group_names: Sequence[str] = ()) -> None:
        if not metrics:
            raise ValueError("at least one metric is required")
        for name, fn in metrics.items():
            if not callable(fn):
                raise TypeError(f"metric {name!r} is not callable")
        self.metrics = dict(metrics)
        self.group_names = list(group_names)
        self._sums: dict[str, float] = dict.fromkeys(self.metrics, 0.0)
        self._group_sums: dict[str, list[float]] = {
            name: [0.0] * len(self.group_names) for name in self.metrics
        }
        self._n = 0
        self._group_n = [0] * len(self.group_names)

    @property
    def n_positions(self) -> int:
        return self._n

    def update(
        self,
        logits: Tensor,
        targets: Tensor,
        loss_mask: Tensor,
        group_ids: Tensor | None = None,
    ) -> None:
        """Add one window: ``logits [B, T, V]``, ``targets``/``loss_mask``/``group_ids`` ``[B, T]``.

        ``group_ids`` indexes ``group_names`` at loss positions (values elsewhere are ignored).
        """
        if logits.dim() != 3:
            raise ValueError(f"logits must be [B, T, V], got shape {tuple(logits.shape)}")
        shape = tuple(logits.shape[:2])
        for name, value in (("targets", targets), ("loss_mask", loss_mask)):
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        if loss_mask.dtype is not torch.bool:
            raise TypeError(f"loss_mask must be a bool tensor, got {loss_mask.dtype}")
        if self.group_names and group_ids is None:
            raise ValueError("group_ids are required when group_names were given")
        n = int(loss_mask.sum())
        if n == 0:
            return
        selected_logits = logits[loss_mask]
        selected_targets = targets[loss_mask]
        groups: Tensor | None = None
        if self.group_names and group_ids is not None:
            if tuple(group_ids.shape) != shape:
                raise ValueError(f"group_ids must have shape {shape}, got {tuple(group_ids.shape)}")
            groups = group_ids[loss_mask].to(torch.int64)
            if bool((groups < 0).any()) or bool((groups >= len(self.group_names)).any()):
                raise ValueError("group_ids at loss positions must index group_names")
            self._group_n = [
                a + b
                for a, b in zip(
                    self._group_n,
                    torch.bincount(groups, minlength=len(self.group_names)).tolist(),
                    strict=True,
                )
            ]
        self._n += n
        for name, fn in self.metrics.items():
            values = fn(selected_logits, selected_targets)
            if tuple(values.shape) != (n,):
                raise ValueError(
                    f"metric {name!r} must return one value per position ({n}), "
                    f"got shape {tuple(values.shape)}"
                )
            # host-side float64 sums: MPS has no float64 (P0.8 fix for profile:m_series)
            values = values.detach().cpu().to(torch.float64)  # move first: MPS cannot cast
            self._sums[name] += float(values.sum())
            if groups is not None:
                per_group = torch.bincount(
                    groups.cpu(), weights=values, minlength=len(self.group_names)
                )
                self._group_sums[name] = [
                    a + b for a, b in zip(self._group_sums[name], per_group.tolist(), strict=True)
                ]

    def result(self) -> dict[str, Any]:
        """Means over loss positions: ``{metric: mean, "n_positions": n, "per_group": {...}}``."""
        out: dict[str, Any] = {
            name: (self._sums[name] / self._n if self._n else float("nan")) for name in self.metrics
        }
        out["n_positions"] = self._n
        per_group: dict[str, dict[str, Any]] = {}
        for g, group in enumerate(self.group_names):
            count = self._group_n[g]
            entry: dict[str, Any] = {
                name: (self._group_sums[name][g] / count if count else float("nan"))
                for name in self.metrics
            }
            entry["n_positions"] = count
            per_group[group] = entry
        out["per_group"] = per_group
        return out


def evaluate_windows(
    model: Any,
    windows: Iterable[Batch],
    metrics: Mapping[str, Metric],
    *,
    device: torch.device,
    autocast: AbstractContextManager[object] | None = None,
    group_of_doc: Tensor | None = None,
    group_names: Sequence[str] = (),
    position_groups: PositionGroups | None = None,
) -> dict[str, Any]:
    """Run ``model`` over consecutive ``windows`` without gradients and average the metrics.

    ``windows`` is a finite sequence of ``[B, T]`` batches with the P0.4 keys (``tokens``,
    ``targets``, ``context``, ``loss_mask``, ``reset``, ``doc``); all must share ``B``. The state
    is initialised once and carried from window to window; the model applies ``reset`` itself.
    ``group_of_doc[d]`` (int64, on any device) is the index into ``group_names`` of document
    ``d``; when given, per-group results are reported under ``"per_group"``. The result also
    carries ``n_windows`` and ``n_tokens`` (non-padding positions seen). ``position_groups``
    (P1.15, module docstring) adds ``per_position_group``.
    """
    accumulator = MetricAccumulator(metrics, group_names)
    by_position: MetricAccumulator | None = None
    labeller: Callable[[Mapping[str, Tensor]], Tensor] | None = None
    if position_groups is not None:
        by_position = MetricAccumulator(metrics, list(position_groups.names))
        labeller = position_groups.start()
    if group_names and group_of_doc is None:
        raise ValueError("group_of_doc is required when group_names are given")
    groups = group_of_doc.to(device) if group_of_doc is not None else None
    was_training = bool(getattr(model, "training", False))
    model.eval()
    n_windows = 0
    n_tokens = 0
    state: Any = None
    prev_tokens: Tensor | None = None
    underlying = getattr(model, "model", model)
    key_address = bool(getattr(getattr(underlying, "cfg", None), "key_address", False))
    batch_size: int | None = None
    ctx = autocast if autocast is not None else nullcontext()
    try:
        with torch.no_grad():
            for window in windows:
                tokens = window["tokens"].to(device)
                B, _T = tokens.shape
                if batch_size is None:
                    batch_size = B
                    state = model.init_state(B, device)
                elif B != batch_size:
                    raise ValueError(f"window batch size changed from {batch_size} to {B}")
                context = window["context"].to(device)
                reset = window["reset"].to(device)
                loss_mask = window["loss_mask"].to(device)
                targets = window["targets"].to(device)
                with ctx:
                    kwargs = {"prev_tokens": prev_tokens} if key_address else {}
                    logits, state, _info = model(tokens, context, state, reset, **kwargs)
                if key_address:
                    prev_tokens = tokens[:, -1].clone()
                group_ids: Tensor | None = None
                if groups is not None:
                    doc = window["doc"].to(device)
                    group_ids = groups[doc.clamp_min(0)]
                accumulator.update(logits.float(), targets, loss_mask, group_ids)
                if by_position is not None and labeller is not None:  # P1.15
                    labels = labeller(window).to(device)
                    by_position.update(logits.float(), targets, loss_mask, labels)
                n_windows += 1
                n_tokens += int((window["doc"] >= 0).sum())
    finally:
        if was_training:
            model.train()
    out = accumulator.result()
    if by_position is not None:
        out["per_position_group"] = by_position.result()["per_group"]
    out["n_windows"] = n_windows
    out["n_tokens"] = n_tokens
    return out
