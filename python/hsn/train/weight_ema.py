"""``train.weight_ema``: an exponential moving average of the weights (opt-in; default off).

Training is unchanged: the optimiser, the gradients, the carried state and every checkpoint's
``model`` hold the *training* weights, and a run with the average on takes the same steps, bit
for bit, as one without it. What changes is which weights are *scored*: the trainer's
``evaluate`` runs the averaged weights, and when a run ends the harness installs them in the
model, so that the final evaluation, the trained-state diagnostics and the probes describe the
averaged model.

The average, after the ``n``-th applied optimiser step (``n`` counted from 0)::

    avg <- d_n * avg + (1 - d_n) * w,    d_n = min(decay, (1 + n) / (10 + n))

The warm-up ``d_n`` makes the average follow the weights closely at the start of a run (the first
step replaces 90 % of the initial weights) and reach ``decay`` after ``(10 * decay - 1) /
(1 - decay)`` steps (4,490 at 0.998; 890 at 0.99). A skipped step (a guard, an fp16 overflow)
moves no weight and is not averaged.

Every floating-point parameter is averaged; buffers (the scorer's feature averages, an EMA target
encoder) are not: they stay the training run's.

A checkpoint carries the average in ``extra["weight_ema"]`` (``decay``, ``updates`` and the
averaged tensors by parameter name), so a resumed run continues it exactly.
:func:`hsn.train.checkpoint.load_checkpoint` with ``averaged=True`` is the evaluation-only
reader: it loads the training weights and then replaces them by the checkpoint's average, when it
has one.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from hsn.kernels import trainer as trainer_kernels

__all__ = ["WARMUP", "WeightEMA", "install_averaged", "warmup_decay"]

#: The warm-up's constant: ``d_n = min(decay, (1 + n) / (WARMUP + n))``.
WARMUP = 10


def warmup_decay(decay: float, updates: int) -> float:
    """The rate of update number ``updates`` (counted from 0)."""
    return min(decay, (1.0 + updates) / (WARMUP + updates))


def _averaged_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters() if p.is_floating_point()]


class WeightEMA:
    """The moving average of ``model``'s floating-point parameters (module docstring)."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0 or not math.isfinite(decay):
            raise ValueError(f"train.weight_ema={decay} must be in (0, 1)")
        self.decay = float(decay)
        self.updates = 0
        named = _averaged_parameters(model)
        if not named:
            raise ValueError("train.weight_ema: the model has no floating-point parameter")
        self.names = [name for name, _ in named]
        self._params = [p for _, p in named]
        self._shadow = [p.detach().clone() for p in self._params]
        self._backup: list[Tensor] | None = None  # the training weights while the average is in
        self.installed = False  # install(): the model holds the average for good

    # training --------------------------------------------------------------------------------

    @torch.no_grad()
    def update(self) -> None:
        """One step of the average towards the model's current weights."""
        if self.installed or self._backup is not None:
            raise RuntimeError("train.weight_ema: the model holds the averaged weights")
        weight = 1.0 - warmup_decay(self.decay, self.updates)
        params = [p.detach() for p in self._params]
        # P2.10b: one kernel launch for fp32 weights on CUDA (the same lerp, bit for bit)
        if not trainer_kernels.lerp(self._shadow, params, weight):
            torch._foreach_lerp_(self._shadow, params, weight)
        self.updates += 1

    # scoring ---------------------------------------------------------------------------------

    @contextlib.contextmanager
    def averaged(self) -> Iterator[None]:
        """The model holds the averaged weights inside the block and the training weights, bit
        for bit, after it. In place, so a compiled window sees them. Nested or after
        :meth:`install`: the model is left as it is."""
        if self.installed or self._backup is not None:
            yield
            return
        with torch.no_grad():
            self._backup = [p.detach().clone() for p in self._params]
            for p, avg in zip(self._params, self._shadow, strict=True):
                p.copy_(avg)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, keep in zip(self._params, self._backup, strict=True):
                    p.copy_(keep)
            self._backup = None

    @torch.no_grad()
    def install(self) -> None:
        """The averaged weights become the model's, for good (the end of a run: everything
        scored after it reads the averaged model). The average no longer updates."""
        if self._backup is not None:
            raise RuntimeError("train.weight_ema: install() inside averaged()")
        if self.installed:
            return
        for p, avg in zip(self._params, self._shadow, strict=True):
            p.copy_(avg)
        self.installed = True

    # checkpoints -----------------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {
                name: avg.detach().to(device="cpu", copy=True)
                for name, avg in zip(self.names, self._shadow, strict=True)
            },
        }

    @torch.no_grad()
    def load_state_dict(self, state: Any, path: str = "checkpoint") -> None:
        """Restore :meth:`state_dict`; a missing, foreign or malformed state raises."""
        shadow = _checked(state, path)
        if float(state["decay"]) != self.decay:
            raise ValueError(
                f"{path}: weight_ema was written with decay {state['decay']}, this trainer "
                f"runs {self.decay}"
            )
        if set(shadow) != set(self.names):
            raise ValueError(f"{path}: weight_ema holds other parameters than the model")
        for name, avg in zip(self.names, self._shadow, strict=True):
            value = shadow[name]
            if tuple(value.shape) != tuple(avg.shape):
                raise ValueError(
                    f"{path}: weight_ema[{name}] has shape {tuple(value.shape)}, the model's "
                    f"is {tuple(avg.shape)}"
                )
            avg.copy_(value.to(device=avg.device, dtype=avg.dtype))
        self.updates = int(state["updates"])
        self.installed = False


def _checked(state: Any, path: str) -> Mapping[str, Tensor]:
    if not isinstance(state, Mapping):
        raise ValueError(f"{path}: weight_ema checkpoint state is missing")
    updates = state.get("updates")
    decay = state.get("decay")
    shadow = state.get("shadow")
    if (
        isinstance(updates, bool)
        or not isinstance(updates, int)
        or updates < 0
        or not isinstance(decay, float)
        or not isinstance(shadow, Mapping)
        or not all(isinstance(k, str) and isinstance(v, Tensor) for k, v in shadow.items())
    ):
        raise ValueError(f"{path}: weight_ema must hold decay, updates and the averaged tensors")
    return shadow


@torch.no_grad()
def install_averaged(model: nn.Module, extra: Mapping[str, Any], path: str = "checkpoint") -> bool:
    """Replace ``model``'s weights by the average a checkpoint's ``extra`` carries; False (and the
    model untouched) when it carries none. Evaluation only: the training weights are gone."""
    state = extra.get("weight_ema")
    if state is None:
        return False
    shadow = _checked(state, path)
    named = dict(_averaged_parameters(model))
    if set(shadow) != set(named):
        raise ValueError(f"{path}: weight_ema holds other parameters than the model")
    for name, p in named.items():
        value = shadow[name]
        if tuple(value.shape) != tuple(p.shape):
            raise ValueError(
                f"{path}: weight_ema[{name}] has shape {tuple(value.shape)}, the model's is "
                f"{tuple(p.shape)}"
            )
        p.copy_(value.to(device=p.device, dtype=p.dtype))
    return True
