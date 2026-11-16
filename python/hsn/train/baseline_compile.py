"""Opt-in compiled windows for the baselines (``hardware.baseline_compile``,
``docs/baseline_compile.md``).

P2.8 compiles the HSN's training window (:mod:`hsn.train.compile`, ``hardware.compile_mode``);
the baselines of the generic trainer (:class:`hsn.train.loop.Trainer`) ran eager, so their step
times stood against the HSN's compiled ones. With ``hardware.baseline_compile`` set to a
``torch.compile`` mode (``default`` | ``reduce-overhead`` | ``max-autotune`` |
``max-autotune-no-cudagraphs``) the trainer runs a baseline's windows, training and evaluation,
through a :class:`CompiledBaseline`. ``none`` (the default) builds nothing and the trainer is the
one it was. ``hardware.compile_mode`` keeps its meaning: the generic trainer still ignores it.

**What is compiled**, by family (``fullgraph``, static shapes; the same Python the eager forward
runs, so nothing of a baseline's architecture changes; the backward of a compiled graph is
compiled with it, and the loss, the clipping, the guards and the optimiser stay eager, as under
the HSN's compiled window):

* ``transformer``, ``reader`` and ``hsn_t`` without its memory (``model.memory = false``,
  :class:`hsn.model.hsn_t.HSNTNoMemory`): the whole forward of a window as one graph. With
  ``model.reader_carry`` the reader's cache goes in and out of that graph (fixed shapes);
  under CUDA graphs the cache returned is copied out of graph memory, as the next window reads
  it after the next replay began.
* ``gru``: the embeddings before the recurrence and the read-out after it
  (:meth:`~hsn.baselines.gru.GRUBaseline._embed`, :meth:`~hsn.baselines.gru.GRUBaseline._readout`),
  as two graphs. The recurrence between them is ``torch.nn.GRU``, which ``torch.compile`` does not
  take (dynamo breaks the graph at RNN modules, and tracing through one would replace cuDNN's
  kernel by an unrolled loop of small ones); it stays the call it was, cuDNN on CUDA.
* ``selective_ssm``: the whole forward of a window as one graph, with the scan of every block
  run as :func:`hsn.baselines.mamba._plain_scan`: the same recurrence as plain tensor
  operations, with autograd's backward and the resets applied at every position. The eager
  scan (``_SelectiveScan``) writes every position's state into one window-sized tensor in
  place and has a hand-written backward that does the same; a compiled graph has no in-place
  writes, and that loop compiled costs a pass over the tensor per position, quadratic in the
  window (the document has the measurements). This is the one place where the compiled window
  does not run the eager forward's own code; its values equal the eager ones to rounding, and
  it reads no reset columns from the device (the eager forward's one host sync per window).

**Variants.** A compiled function is built per batch shape, grad mode, training mode and autocast
state, each with its own code object, so the evaluation (``no_grad``, evaluation mode, its own
batch size) is a function of its own and nothing recompiles once warm. A compiled graph is also
specialised to the memory layout of its inputs: the window's inputs are made contiguous, and the
GRU's read-out has a function per layout of the recurrence's output (a transposed view after one
cuDNN call, a contiguous tensor after several, which a reset inside a window causes). The first
call of a function compiles; under CUDA graphs the second records.

**CUDA graphs** (``reduce-overhead`` and ``max-autotune`` on CUDA): every window starts a new
graph step (``torch.compiler.cudagraph_mark_step_begin``). The logits a window returns are graph
memory, which the next window's replay overwrites: read them before the next window, as the
trainer and the evaluation do. ``train.grad_accum > 1`` is refused (a gradient held in ``.grad``
across windows would be overwritten too), as for the HSN. The state and the info tensor of a
``selective_ssm`` window, and a carried reader cache (``model.reader_carry``), come out of a
compiled graph and are cloned out of graph memory.

**Scope** (errors over silence, :func:`check_baseline_scope`): the five families above; no
dropout (a compiled random draw differs from the eager one); ``train.grad_accum = 1`` under CUDA
graphs. On MPS a backend that cannot compile falls back to the eager model with a logged notice.

The compiled window equals the eager forward to rounding (fused kernels round differently), not
bit for bit.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from hsn.baselines.gru import GRUBaseline
from hsn.baselines.mamba import MambaBaseline
from hsn.baselines.reader_lm import ReaderLM
from hsn.baselines.transformer import TransformerBaseline
from hsn.config import RunConfig
from hsn.device import DeviceContext
from hsn.model.hsn_t import HSNTNoMemory
from hsn.train.compile import Backend, _compiler, _mps_supported

__all__ = [
    "CUDA_GRAPH_MODES",
    "CompiledBaseline",
    "check_baseline_scope",
    "compile_baseline",
]

log = logging.getLogger(__name__)

#: The ``torch.compile`` modes that replay the compiled graphs as CUDA graphs on CUDA.
CUDA_GRAPH_MODES: tuple[str, ...] = ("reduce-overhead", "max-autotune")

#: What runs compiled and what stays eager, by family (the run record's ``scope``).
_SCOPE: dict[type[nn.Module], str] = {
    GRUBaseline: "the embeddings and the read-out, forward and backward; the nn.GRU recurrence "
    "between them is the eager call (cuDNN on CUDA)",
    TransformerBaseline: "the whole forward of a window and its backward",
    ReaderLM: "the whole forward of a window and its backward",
    MambaBaseline: "the whole forward of a window and its backward, with the selective scan "
    "run as plain tensor operations under autograd (_plain_scan), not the eager run's "
    "in-place loop with its hand-written backward",
    HSNTNoMemory: "the whole forward of a window and its backward",
}

WindowOut = tuple[Tensor, Any, dict[str, Any]]


def _whole_forward(
    model: nn.Module, tokens: Tensor, context: Tensor, reset: Tensor
) -> tuple[Tensor, None, dict[str, Any]]:
    """A stateless model's forward (``transformer``, ``reader``, ``hsn_t`` without memory)."""
    out: tuple[Tensor, None, dict[str, Any]] = model(tokens, context, None, reset)
    return out


def _carried_forward(
    model: nn.Module, tokens: Tensor, context: Tensor, state: Any, reset: Tensor
) -> tuple[Tensor, Any, dict[str, Any]]:
    """The forward of a reader that carries its cache (``model.reader_carry``: ``reader`` and
    ``hsn_t`` without memory): the window and the cache in, the logits and the next cache out,
    one graph (the cache has the same shapes at every window)."""
    out: tuple[Tensor, Any, dict[str, Any]] = model(tokens, context, state, reset)
    return out


def _carries(model: nn.Module) -> bool:
    """A reader model with ``model.reader_carry`` (its state is the reader's cache)."""
    return isinstance(model, ReaderLM | HSNTNoMemory) and bool(model.reader.carry)


def _gru_embed(model: GRUBaseline, tokens: Tensor, context: Tensor, reset: Tensor) -> Tensor:
    return model._embed(tokens, context, reset)


def _gru_readout(model: GRUBaseline, y: Tensor) -> Tensor:
    return model._readout(y)


def _mamba_window(
    model: MambaBaseline, tokens: Tensor, context: Tensor, h: Tensor, conv: Tensor, reset: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return model._window(tokens, context, h, conv, reset, None)  # None: the plain scan


def _uses_cuda_graphs(ctx: DeviceContext, mode: str, backend: Backend) -> bool:
    return ctx.device.type == "cuda" and backend == "inductor" and mode in CUDA_GRAPH_MODES


def check_baseline_scope(
    model: nn.Module, cfg: RunConfig, ctx: DeviceContext, *, backend: Backend = "inductor"
) -> None:
    """Raise ``NotImplementedError`` (with the reason) unless ``model`` under ``cfg`` is in the
    compiled scope (module docstring)."""
    mode = cfg.hardware.baseline_compile
    if type(model) not in _SCOPE:
        raise NotImplementedError(
            f"hardware.baseline_compile={mode}: {type(model).__name__} is not a baseline in the "
            f"compiled scope ({', '.join(sorted(c.__name__ for c in _SCOPE))}); the HSN's "
            "compiled window is hardware.compile_mode"
        )
    dropout = {
        GRUBaseline: cfg.baseline.gru.dropout,
        TransformerBaseline: cfg.baseline.transformer.dropout,
    }.get(type(model), 0.0)
    if dropout > 0.0:
        raise NotImplementedError(
            f"hardware.baseline_compile={mode} with dropout {dropout}: a compiled random draw "
            "differs from the eager one, so the compiled run could not be checked against the "
            "eager one; train with dropout 0 or hardware.baseline_compile=none"
        )
    if cfg.train.grad_accum != 1 and _uses_cuda_graphs(ctx, mode, backend):
        raise NotImplementedError(
            f"hardware.baseline_compile={mode} with train.grad_accum={cfg.train.grad_accum}: a "
            "CUDA-graph backward output held in .grad across windows would be overwritten by "
            "the next window's replay; use train.grad_accum=1 or a mode without CUDA graphs "
            "(default, max-autotune-no-cudagraphs)"
        )


class CompiledBaseline:
    """A baseline's windows through ``torch.compile`` (module docstring): called like the
    model's forward, ``(tokens, context, state, reset) -> (logits, state, info)``, and usable
    where :func:`hsn.eval.evaluate.evaluate_windows` takes a model (``training``, ``train``,
    ``eval``, ``init_state``). ``calls`` counts windows; ``compile_seconds`` accumulates the wall
    time of the windows during which a function compiled or recorded; ``graphs`` counts the
    graphs dynamo compiled during this object's windows (one per compiled function: a larger
    number means something recompiled)."""

    def __init__(self, model: nn.Module, ctx: DeviceContext, mode: str, backend: Backend) -> None:
        self.model = model
        self.ctx = ctx
        self.mode = mode
        self.backend = backend
        self.cuda_graphs = _uses_cuda_graphs(ctx, mode, backend)
        self.calls = 0
        self.compile_seconds = 0.0
        self.graphs = 0
        self._skips_before = _cudagraph_skips()
        self._compile = _compiler(mode, backend, ctx.device.type)
        #: (function, variant) -> its compiled function, and how often it ran
        self._fns: dict[tuple[object, ...], Callable[..., Any]] = {}
        self._runs: dict[tuple[object, ...], int] = {}
        self._new = False  # the current window ran a function that compiles or records

    # the model, as evaluate_windows sees it ----------------------------------------------------

    @property
    def training(self) -> bool:
        return self.model.training

    def train(self, mode: bool = True) -> CompiledBaseline:
        self.model.train(mode)
        return self

    def eval(self) -> CompiledBaseline:
        self.model.eval()
        return self

    def init_state(self, B: int, device: torch.device | str) -> Any:
        return self.model.init_state(B, device)  # type: ignore[operator]

    # the window --------------------------------------------------------------------------------

    def variant(self, tokens: Tensor) -> tuple[object, ...]:
        """What selects a compiled function, apart from what its own inputs add: the batch
        shape, grad mode, training mode and autocast state."""
        return (
            tuple(tokens.shape),
            torch.is_grad_enabled(),
            self.model.training,
            torch.is_autocast_enabled(self.ctx.device.type),
        )

    def _fn(self, fn: Callable[..., Any], *key: object) -> Callable[..., Any]:
        """The compiled ``fn`` of variant ``key``, built on first use with a code object of its
        own (one graph each, so nothing recompiles); marks the window when this call of it
        compiles or, under CUDA graphs (warm-up, then recording), records."""
        name = (fn.__name__, *key)
        if name not in self._fns:
            self._fns[name] = self._compile(fn)
        runs = self._runs.get(name, 0)
        self._runs[name] = runs + 1
        if runs < (2 if self.cuda_graphs else 1):
            self._new = True
        return self._fns[name]

    def __call__(self, tokens: Tensor, context: Tensor, state: Any, reset: Tensor) -> WindowOut:
        model = self.model
        # one memory layout for the inputs: a compiled graph is specialised to its inputs'
        # strides (the loader's windows are contiguous already, and this returns them as is)
        tokens, context, reset = tokens.contiguous(), context.contiguous(), reset.contiguous()
        key = self.variant(tokens)
        self._new = False
        graphs = _unique_graphs()
        started = time.perf_counter()
        if self.cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()  # type: ignore[no-untyped-call]
        out: WindowOut
        if isinstance(model, GRUBaseline):

            def embed(tokens: Tensor, context: Tensor, reset: Tensor) -> Tensor:
                out: Tensor = self._fn(_gru_embed, *key)(model, tokens, context, reset)
                return out

            def readout(y: Tensor) -> Tensor:
                # the recurrence's output is a transposed view after one cuDNN call and a
                # contiguous tensor after several (a reset inside the window): a graph each
                out: Tensor = self._fn(_gru_readout, *key, tuple(y.stride()))(model, y)
                return out

            out = model._window(tokens, context, state, reset, embed, readout)
        elif _carries(model):  # model.reader_carry: the cache in and out of the one graph
            out = self._fn(_carried_forward, *key)(model, tokens, context, state, reset)
            if self.cuda_graphs:  # graph memory, which the next window's replay overwrites
                out = (out[0], tuple(t.clone() for t in out[1]), out[2])
        elif isinstance(model, MambaBaseline):
            h, conv = model._check_inputs(tokens, context, state, reset)
            logits, h, conv, top = self._fn(_mamba_window, *key)(
                model, tokens, context, h.contiguous(), conv.contiguous(), reset
            )
            if self.cuda_graphs:  # out of graph memory: the next replay overwrites it
                h, conv, top = h.clone(), conv.clone(), top.clone()
            out = logits, (h, conv), {"state_abs_max": top}
        else:
            if state is not None:
                raise ValueError(f"{type(model).__name__} carries no state, got a state")
            out = self._fn(_whole_forward, *key)(model, tokens, context, reset)
        self.calls += 1
        compiled = _unique_graphs() - graphs
        self.graphs += compiled
        if self._new or compiled:
            if self.cuda_graphs:
                torch.cuda.synchronize(self.ctx.device)
            self.compile_seconds += time.perf_counter() - started
        return out

    def report(self) -> dict[str, Any]:
        """What ran, for the run record: the mode, whether the graphs replay as CUDA graphs
        (and how many graphs PyTorch refused to replay that way since this object was built,
        process-wide: 0 unless something in a graph rules CUDA graphs out), what of the family
        is compiled, the windows, compiled functions and graphs so far and their compile
        time."""
        return {
            "mode": self.mode,
            "backend": self.backend if isinstance(self.backend, str) else repr(self.backend),
            "cuda_graphs": self.cuda_graphs,
            "cudagraph_skips": _cudagraph_skips() - self._skips_before,
            "scope": _SCOPE[type(self.model)],
            "evaluation": "compiled",
            "windows": self.calls,
            "functions": len(self._fns),
            "graphs": self.graphs,
            "compile_seconds": self.compile_seconds,
        }


def _cudagraph_skips() -> int:
    """How many compiled graphs of this process inductor did not replay as CUDA graphs although
    the mode asked for it (its ``cudagraph_skips`` counter; each skip is also logged by PyTorch
    with its reason)."""
    from torch._dynamo.utils import counters

    return int(counters["inductor"]["cudagraph_skips"])


def _unique_graphs() -> int:
    """The graphs dynamo compiled in this process so far (its ``unique_graphs`` counter)."""
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])


def compile_baseline(
    model: nn.Module, cfg: RunConfig, ctx: DeviceContext, *, backend: Backend = "inductor"
) -> CompiledBaseline | None:
    """The compiled windows of the baseline ``model`` at ``cfg.hardware.baseline_compile``
    (module docstring), after :func:`check_baseline_scope`; ``None`` when the key is ``none``,
    or on MPS when ``backend`` cannot compile there (a logged notice; the windows then run
    eager). ``backend`` is ``inductor`` (what the key means); tests pass ``aot_eager`` where
    inductor's C++ toolchain is absent or its compile time is not the point."""
    mode = cfg.hardware.baseline_compile
    if mode == "none":
        return None
    check_baseline_scope(model, cfg, ctx, backend=backend)
    if ctx.device.type == "mps":
        ok, reason = _mps_supported(backend, ctx.device)
        if not ok:
            log.warning(
                "torch.compile (%s, mode=%s) is not supported on %s (%s); the %s windows run eager",
                backend,
                mode,
                ctx.device,
                reason,
                type(model).__name__,
            )
            return None
    return CompiledBaseline(model, ctx, mode, backend)
