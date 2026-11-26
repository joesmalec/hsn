"""Opt-in compiled reader for HSN-T with its memory (``hardware.reader_compile``,
``docs/wcells_speed.md``).

An HSN-T with its memory runs its reader in the window's eager prologue, on the eager, the
compiled and the fused event loop alike: four blocks over the window and the carried keys,
about 200 kernels forward and 300 backward a window. With the fused event block
(``hardware.event_kernel=fused``) the reader is 41 % of the card's time of a training step, and
no compiled graph runs in a training window at all. With ``hardware.reader_compile=default`` the
carried reader's read of a training window (:meth:`hsn.model.reader.Reader.read`) is one
``torch.compile`` graph in inductor's default mode; its backward is compiled with it.

* **What is compiled:** ``Reader.read(tokens, reset, cache)``, the same Python the eager read
  runs, with ``fullgraph`` and static shapes. The cache goes in and comes out of the graph
  (fixed shapes). No CUDA graphs: the graph's outputs keep their autograd edges and nothing is
  overwritten by a later call, so ``train.grad_accum`` and ``train.tbptt_windows`` are what
  they are without the key (a segment's windows are three calls of one compiled function).
* **Which windows:** a training window with a gradient (the statistics steps' term passes
  among them). Evaluation and every call without a gradient keep the eager reader, so a run's
  evaluations are computed as without the key.
* **How the model gets it:** the compiled window hands :meth:`hsn.model.hsn.HSN._window_prologue`
  the read to call (``read=``); nothing is installed on the model, which stays the module it
  was (a copy, a checkpoint and the evaluation see no difference).
* **Numerics:** inductor fuses the reader's elementwise chains into kernels of its own, which
  round differently from the eager kernels: equal to rounding, not bit for bit. Hence a key,
  off by default; at ``none`` nothing of this module is built.

The scope is checked when the configuration loads
(:func:`hsn.config.reader_compile_scope_error`) and again here on the built model.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor

from hsn.device import DeviceContext
from hsn.model.hsn import HSN
from hsn.model.reader import Cache, Reader
from hsn.train.compile import Backend, _compiler

__all__ = ["READER_COMPILE_MODES", "CompiledReader", "reader_of"]

#: The values of ``hardware.reader_compile`` that compile (``none`` builds nothing).
READER_COMPILE_MODES: tuple[str, ...] = ("default",)


def _read(reader: Reader, tokens: Tensor, reset: Tensor, *cache: Tensor) -> tuple[Tensor, Cache]:
    """The compiled function: the carried reader's read of one window (the class's own
    method: nothing is looked up on the instance)."""
    return Reader.read(reader, tokens, reset, cache)


class CompiledReader:
    """``hardware.reader_compile``: :meth:`read` is called like the carried reader's
    :meth:`hsn.model.reader.Reader.read` and runs it compiled in a training window with a
    gradient, eager otherwise (module docstring). ``calls`` counts the compiled reads,
    ``compile_seconds`` accumulates the wall time of the reads that compiled (the first of
    each variant: batch shape, autocast state)."""

    def __init__(
        self, reader: Reader, ctx: DeviceContext, mode: str, backend: Backend = "inductor"
    ) -> None:
        if mode not in READER_COMPILE_MODES:
            raise ValueError(
                f"hardware.reader_compile must be none or one of {READER_COMPILE_MODES}, got "
                f"{mode!r}"
            )
        if not reader.carry:
            raise ValueError(
                "hardware.reader_compile compiles the carried reader's read "
                "(model.reader_carry > 0); this reader carries nothing"
            )
        self.reader = reader
        self.mode = mode
        self._fn = _compiler(mode, backend, ctx.device.type)(_read)
        self.calls = 0
        self.compile_seconds = 0.0
        self._variants: set[tuple[object, ...]] = set()

    def read(
        self,
        tokens: Tensor,
        reset: Tensor,
        cache: Cache,
        short: tuple[Tensor, int] | None = None,
        *,
        blank: Tensor | None = None,
        unknown: Tensor | None = None,
    ) -> tuple[Tensor, Cache]:
        reader = self.reader
        if not (reader.training and torch.is_grad_enabled()):
            # evaluation, and every read without a gradient: the eager reader, as without the key
            return reader.read(tokens, reset, cache, short, blank=blank, unknown=unknown)
        if short is not None or blank is not None or unknown is not None:
            raise ValueError(
                "hardware.reader_compile: the compiled read takes no short rows "
                "(train.reach_dropout) and no byte-content masks; the configuration refuses "
                "them with the key"
            )
        cache = reader.check_cache(cache, int(tokens.shape[0]))
        kind = tokens.device.type
        key = (tuple(tokens.shape), torch.is_autocast_enabled(kind), len(cache))
        new = key not in self._variants
        started = time.perf_counter()
        out, kept = self._fn(reader, tokens, reset, *cache)
        self.calls += 1
        if new:
            self.compile_seconds += time.perf_counter() - started
            self._variants.add(key)
        return out, tuple(kept)


def reader_of(
    model: HSN, ctx: DeviceContext, mode: str, *, backend: Backend = "inductor"
) -> CompiledReader | None:
    """``hardware.reader_compile`` as what :func:`hsn.train.compile.compile_window` takes:
    ``None`` for ``none``, otherwise the :class:`CompiledReader` of ``model``'s reader. Raises
    with the reason for a model without a carried reader."""
    if mode == "none":
        return None
    reader: Any = getattr(model, "reader", None)
    if not isinstance(reader, Reader):
        raise ValueError(
            f"hardware.reader_compile={mode} needs an HSN-T with its memory "
            "(model.reader=transformer); this model has no reader"
        )
    return CompiledReader(reader, ctx, mode, backend)
