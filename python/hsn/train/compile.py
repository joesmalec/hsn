"""Launch-overhead reduction, layer L1 (P2.8): a compiled training window and a captured serving
step, without custom kernels.

At the reference scale HSN is launch-bound (plan A.2): a training window is ``T`` positions of
small kernels issued from Python, so the card waits on the host between launches. This module
compiles the work of a window with ``torch.compile`` (``hardware.compile_mode``: ``none`` |
``reduce-overhead`` | ``max-autotune``, passed through as the compile mode) and, on CUDA, replays
it as CUDA Graphs. What it compiles is the same code the eager path runs
(:meth:`hsn.model.hsn.HSN._position`), with three changes that make the shapes and the control flow
static (plan P2.8, "static shapes are required for graph capture"):

* **Blocks of positions.** :func:`compile_window` compiles a block of ``block_size(T)``
  consecutive positions (at most :data:`MAX_BLOCK`, a divisor of ``T``) as one graph and calls it
  ``T / block`` times per window; the window-invariant work before and after the loop
  (:meth:`HSN._window_prologue`, :meth:`HSN._window_epilogue`: embeddings, ``E``, the scorer's
  window features, the decoder, the statistics) stays eager. The block is the compile unit chosen
  by the P2.8 spike on ``profile:rtx3060`` (g1 recipe, bf16 B16 T256): a whole window as one graph
  did not finish compiling in 10 minutes, one compiled position per call left the per-position
  gradient accumulation and copies on the host (11.7k host launches per step, 0.229 s), and blocks
  of 16 to 32 positions gave 0.136 to 0.138 s per step against 0.703 s eager with 500 to 830 host
  launches. With one graph per run, blocks of 16 compile in about half the time of 32 (cold
  cache: dense 67 s against 120 s, topk 139 s against 195 s) for a step 1 to 1.4 % slower (dense
  0.148 against 0.147 s, topk 0.091 against 0.090 s), which is faster end to end for any run up to
  about 30,000 steps (every configured run is 3,000 to 12,000), so blocks of 16 it is. A warm
  inductor cache (a later run of the same configuration) halves the compile again (dense 30 s).
* **Resets as tensor ops.** A block applies the reset of every column (``masked_fill`` with the
  column's mask, the identity where it is False), so a block never depends on which columns reset
  and one graph serves every window.
* **The capturable round.** The routed path runs :meth:`PDGCell.step_sparse` with ``static=True``
  (no host synchronisation; the unit bank's static pass) and every round reads the window's edge
  weights and ``W1`` layout as tensors (:class:`~hsn.model.cell.WindowConsts`), computed eagerly
  once per window, so the edge-softmax backward stays the eager one (the dense cells' exact zero
  edge gradient, plan finding 1 of the efficiency audit).

Dense-fallback row IDs are compacted before the window and padded to a power-of-two batch
(capped at B). Only that subset runs the dense cell pass; distinct padding rows keep their sparse
result. Each bucket has its own compiled function and compile-time accounting, so changing row
IDs within a bucket never changes the graph. At B32 there are six nonempty buckets. The sparse
pass still runs on all rows. State maxima are reduced within each position; only ``capture=True``
on the eager path materializes the full state history.

The compiled window matches eager ``HSN.forward`` to rounding (fused kernels round differently),
not bit for bit; ``hardware.compile_mode = none`` leaves every path eager and unchanged.
``profile:rtx3060`` compiles by default (``reduce-overhead``, the user's decision of 7 Nov 2026);
every other profile stays ``none`` until it is measured there. A run that must stay eager on
rtx3060 overrides ``hardware.compile_mode=none``. The generic trainer of the baselines never
compiles (a logged notice, ``measured["compiled_windows"] = 0``).

Scope (errors over silence: anything else raises ``NotImplementedError`` when the window is
built, and runs eager with ``compile_mode: none``):

* ``model.routing`` ``dense`` or ``topk``; gates ``soft`` (the masked round), ``ste`` and
  evaluation; refresh probes and the dense fallback (``train.p_dense``) included;
* not ``greedy_budget`` (a host-side walk), ``random`` (device random numbers: a compiled draw
  differs from the eager one), ``fixed``, exploration (``train.epsilon_*`` > 0) or ``reinforce``
  (both draw from the CPU generator at every position), nor a temperature ramp
  (``train.temperature_start != temperature_end``: the router checks τ as a Python number, so
  each value would be a new graph; a constant τ is one variant);
* not the fp64 master dtype (the gradchecks of ``configs/tiny_gradcheck.yaml`` stay eager: the
  chunk's "Do not").

CUDA Graphs and the trainer (:class:`~hsn.train.tbptt.TBPTTTrainer`): the trainer marks a new graph
step per window, takes the carried state out of graph memory (a replay overwrites it), runs full
windows (no trailing cut: one fixed ``T``), takes a statistics step's per-term gradient norms from
one extra compiled pass per term (several retained backwards of one recorded graph are refused by
CUDA Graph trees) and refuses ``train.grad_accum > 1`` (a gradient held in ``.grad`` from one
window would live in graph memory the next window's replay overwrites). Compile time is reported
apart from the steady state (the ``compile`` segment, never inside step timings).

:func:`compile_step` is the serving step (P5.2's inference path in miniature): one position under
``no_grad`` on a preallocated state updated in place, so that its buffers keep their addresses and
a CUDA Graph can capture the step.

MPS: ``torch.compile`` is tried there (inductor's Metal backend) and the module falls back to
the eager path with one logged notice when compilation fails; on CPU and CUDA a failure raises.
MPS compiles do not fuse and cap a kernel's buffers (:data:`MPS_INDUCTOR_OPTIONS`): fused, the g1
top-K STE block's backward exceeds Metal's 31 kernel buffers and, under the cap, computes wrong
gradients there (an inductor Metal fusion bug in torch 2.15).
"""

from __future__ import annotations

import copy
import itertools
import logging
import time
import types
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from hsn.device import DeviceContext
from hsn.kernels import window as window_kernels
from hsn.model.cell import WindowConsts
from hsn.model.hsn import (
    HSN,
    SELECTION_RECORD_KEYS,
    StepMode,
    _Controls,
    _Prologue,
    _stack_record,
)
from hsn.model.router import BUDGET_SLACK
from hsn.model.state import CellGraphState

__all__ = [
    "MAX_BLOCK",
    "MPS_INDUCTOR_OPTIONS",
    "CompileMode",
    "CompiledWindow",
    "ServingStep",
    "block_size",
    "check_compile_scope",
    "compile_step",
    "compile_window",
    "mps_inductor_options",
]

log = logging.getLogger(__name__)

CompileMode = Literal["none", "reduce-overhead", "max-autotune"]
_MODES: tuple[CompileMode, ...] = ("none", "reduce-overhead", "max-autotune")

#: The largest block of positions compiled as one graph: 16 (module docstring: half the compile of
#: 32 for a step about 1 % slower; larger blocks only add compile time).
MAX_BLOCK = 16

#: MPS inductor options, added to the compile mode's own (measured on an M4 Pro, torch 2.15,
#: 7 Nov 2026, the g1 top-K STE block of 16 positions):
#:
#: * ``max_fusion_size = 1`` (no fusion): with any fusion (sizes 2 to 64) inductor's Metal backend
#:   compiles the block's backward wrong (parameter gradients up to 1.3e3 times eager's, relative;
#:   MPS eager matches the CPU to 9e-5); unfused it equals eager to 2e-5. Dense and the soft
#:   warm-up compiled right fused, but one rule for every MPS compile keeps correctness first.
#: * ``max_fusion_unique_io_buffers = 24``: a Metal kernel takes at most 31 buffer arguments, and
#:   fused the backward did not compile ("number of constant buffers exceeds maximum supported
#:   (31)"); kept as a guard, since one node (a wide ``cat``) can read many buffers.
MPS_INDUCTOR_OPTIONS: dict[str, Any] = {"max_fusion_size": 1, "max_fusion_unique_io_buffers": 24}

WindowOut = tuple[Tensor, CellGraphState, dict[str, Tensor]]


def block_size(T: int) -> int:
    """The positions per compiled block for a window of ``T``: the largest divisor of ``T`` that
    is at most :data:`MAX_BLOCK` (16 for the A.1 unroll of 256)."""
    if isinstance(T, bool) or not isinstance(T, int) or T < 1:
        raise ValueError(f"T must be an int >= 1, got {T!r}")
    return max(b for b in range(1, min(T, MAX_BLOCK) + 1) if T % b == 0)


# --------------------------------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------------------------------


def check_compile_scope(model: HSN, *, grad_accum: int = 1) -> None:
    """Raise ``NotImplementedError`` (with the reason) unless ``model`` is in the compiled scope
    (module docstring): routing, gate, exploration, master dtype and ``train.grad_accum``."""
    if not isinstance(model, HSN):
        raise TypeError(f"compile needs an HSN model, got {type(model).__name__}")
    cfg = model.cfg
    if cfg.routing not in ("dense", "topk"):
        raise NotImplementedError(
            f"model.routing={cfg.routing!r} is not compiled (P2.8 compiles dense and topk): "
            + {
                "greedy_budget": "its budget walk runs on the host",
                "random": "a compiled random draw differs from the eager one",
                "fixed": "the fixed route is not in the compiled scope",
            }.get(cfg.routing, "unsupported mode")
            + "; run it with hardware.compile_mode=none"
        )
    schedule = model.schedule
    if schedule.gate_mode == "reinforce":
        raise NotImplementedError(
            "train.gate_mode=reinforce is not compiled: its Gumbel samples come from the CPU "
            "generator at every position; run it with hardware.compile_mode=none"
        )
    if schedule.temperature_start != schedule.temperature_end:
        raise NotImplementedError(
            "a temperature ramp (train.temperature_start != temperature_end) is not compiled: "
            "the router checks τ as a Python number, so every new value would compile a new "
            "graph; use a constant temperature or hardware.compile_mode=none"
        )
    if schedule.epsilon_start > 0.0 or schedule.epsilon_end > 0.0:
        raise NotImplementedError(
            "exploration (train.epsilon_start / epsilon_end > 0) is not compiled: it draws from "
            "the CPU generator at every position; run it with hardware.compile_mode=none"
        )
    if model.dtype == torch.float64:
        raise NotImplementedError(
            "the fp64 master dtype is not compiled: the gradchecks (configs/tiny_gradcheck.yaml) "
            "stay eager (P2.8 'Do not')"
        )
    if isinstance(grad_accum, bool) or not isinstance(grad_accum, int) or grad_accum != 1:
        raise NotImplementedError(
            f"train.grad_accum={grad_accum!r} is not compiled: a CUDA-Graph backward output "
            "held in .grad across windows would be overwritten by the next window's replay; use "
            "train.grad_accum=1 or hardware.compile_mode=none"
        )


def _check_mode(mode: str) -> CompileMode:
    if mode not in _MODES:
        raise ValueError(f"compile mode must be one of {_MODES}, got {mode!r}")
    return mode


Backend = str | Callable[..., Any]
_instances = itertools.count()


def _own_code(fn: Callable[..., Any]) -> Callable[..., Any]:
    """A copy of the module-level function ``fn`` with its own code object. Dynamo keeps its
    compiled graphs (and counts recompiles against ``torch._dynamo.config.recompile_limit``) per
    code object, so every compiled window or step gets its own: one model's variants never count
    against another's (a process that builds several models, such as a run followed by its
    profile pass on a replica, would otherwise reach the limit; with ``fullgraph`` that raises)."""
    assert isinstance(fn, types.FunctionType) and fn.__closure__ is None
    code = fn.__code__.replace(co_name=f"{fn.__name__}_{next(_instances)}")
    return types.FunctionType(code, fn.__globals__, code.co_name, fn.__defaults__)


def _compiler(
    mode: CompileMode, backend: Backend, device_type: str = "cuda"
) -> Callable[[Callable[..., Any]], Any]:
    """``torch.compile`` with ``fullgraph`` (a graph break, or reaching the recompile limit,
    raises) and static shapes, on a copy of the function with its own code object
    (:func:`_own_code`). The mode goes to inductor; other backends (tests) take none. On MPS the
    mode's inductor options are passed with :data:`MPS_INDUCTOR_OPTIONS` added
    (``torch.compile`` takes ``mode`` or ``options``, not both); elsewhere the call is unchanged."""

    def wrap(fn: Callable[..., Any]) -> Any:
        own = _own_code(fn)
        if backend == "inductor" and device_type == "mps":
            return torch.compile(
                own, options=mps_inductor_options(mode), fullgraph=True, dynamic=False
            )
        if backend == "inductor":
            return torch.compile(own, mode=mode, fullgraph=True, dynamic=False)
        return torch.compile(own, backend=backend, fullgraph=True, dynamic=False)

    return wrap


def mps_inductor_options(mode: str) -> dict[str, Any]:
    """The inductor options of compile ``mode`` (``torch._inductor.list_mode_options``) with
    :data:`MPS_INDUCTOR_OPTIONS` added."""
    import torch._inductor

    return {**torch._inductor.list_mode_options(mode), **MPS_INDUCTOR_OPTIONS}


def _uses_cuda_graphs(ctx: DeviceContext, mode: CompileMode, backend: Backend) -> bool:
    return ctx.device.type == "cuda" and backend == "inductor" and mode != "none"


def _mps_supported(backend: Backend, device: torch.device) -> tuple[bool, str]:
    """Whether ``torch.compile`` with ``backend`` compiles and runs a small forward and backward
    on ``device`` (the MPS probe of :func:`compile_window`); the reason when it does not."""
    try:
        x = torch.randn(4, 8, device=device, requires_grad=True)
        w = torch.randn(8, 8, device=device, requires_grad=True)

        def probe(a: Tensor, b: Tensor) -> Tensor:
            return torch.tanh(a @ b).sigmoid().sum()

        out = torch.compile(probe, backend=backend, fullgraph=True)(x, w)
        out.backward()  # type: ignore[no-untyped-call]
    except Exception as exc:  # noqa: BLE001 - any failure means "fall back", logged with its reason
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


# --------------------------------------------------------------------------------------------------
# The compiled training window
# --------------------------------------------------------------------------------------------------


def _block(
    model: HSN,
    h: Tensor,
    q: Tensor,
    age: Tensor,
    pos: Tensor,
    z_b: Tensor,
    c_b: Tensor,
    s_b: Tensor,
    r_b: Tensor,
    dense_in: tuple[Tensor, Tensor] | None,
    dense_rows: Tensor | None,
    dense_indices: Tensor | None,
    consts: WindowConsts,
    mode: StepMode,
    controls: _Controls,
    capture: bool = False,
) -> dict[str, Tensor]:
    """A block of consecutive positions (the compile unit): :meth:`HSN._position` at every column
    of ``z_b`` / ``c_b`` / ``s_b`` (``[B, n, …]``) with the column's reset ``r_b[:, i]``, the
    capturable round and the window's constants. ``dense_in`` are the P1.10b dense window's
    ``(pos [B, n], pos_after [B, n])``: its ages are ``age`` at the block's first position (the
    window's start age in the first block, zeros after) and zeros afterwards, made here, so that
    no tensor is passed twice (dynamo specialises on aliased inputs). Returns the final state and
    the block's per-position stacks; with ``capture`` (a diagnostic, no-grad window) also the
    per-position cell states, gates and written masks of :meth:`HSN.forward`'s ``capture``."""
    state = CellGraphState._make(h=h, q=q, age=age, pos=pos)
    n = int(z_b.shape[1])
    z_hats: list[Tensor] = []
    epss: list[Tensor] = []
    us: list[Tensor] = []
    qs: list[Tensor] = []
    state_maxima: list[Tensor] = []
    logit_steps: list[Tensor] = []
    records: list[dict[str, Tensor]] = []
    readouts: list[Tensor] = []
    hs: list[Tensor] = []
    gates: list[Tensor] = []
    written: list[Tensor] = []
    zero_age = torch.zeros_like(age) if dense_in is not None else None
    for i in range(n):
        counters: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
        if dense_in is not None:
            assert zero_age is not None
            pos_b, after_b = dense_in
            counters = (age if i == 0 else zero_age, pos_b[:, i], zero_age, after_b[:, i])
        result = model._position(
            state,
            r_b[:, i],
            z_b[:, i],
            c_b[:, i],
            s_b[:, i],
            mode,
            controls,
            dense_rows,
            capture,
            counters,
            static=True,
            consts=consts,
            dense_indices=dense_indices,
        )
        state = result.state
        if capture:
            assert result.gate is not None and result.written is not None
            hs.append(state.h.detach())
            gates.append(result.gate)
            written.append(result.written)
        if model.cfg.selected_readout:
            readouts.append(model._selected_readout(state, result.selection, controls.gate, mode))
        z_hats.append(result.z_hat)
        epss.append(result.eps)
        us.append(result.u)
        qs.append(state.q)
        state_maxima.append(state.h.detach().abs().amax())
        logit_steps.append(result.unit_logits.detach())
        if result.selection is not None:
            records.append(model._selection_record(result.selection))
    out = {
        "h": state.h,
        "q": state.q,
        "age": state.age.contiguous(),
        "pos": state.pos.contiguous(),  # one stride for every block (dense: a view of [B, T])
        "z_hat": torch.stack(z_hats, dim=1),
        "eps": torch.stack(epss, dim=1),
        "u": torch.stack(us, dim=1),
        "qs": torch.stack(qs, dim=1),
        "state_abs_max": torch.stack(state_maxima),
        "unit_logits_d": torch.stack(logit_steps, dim=1),
    }
    if records:
        for key in SELECTION_RECORD_KEYS:
            out["sel_" + key] = _stack_record([r[key] for r in records])
    if readouts:
        out["readouts"] = torch.stack(readouts, dim=1)
    if capture:
        out["cap_h"] = torch.stack(hs, dim=1)
        out["cap_gate"] = torch.stack(gates, dim=1)
        out["cap_written"] = torch.stack(written, dim=1)
    return out


#: The parameters autocast casts at every use (the linears of P, S, the scorer, V, the gate,
#: W_out and the unit bank): the ones a :class:`SweepTwin` holds pre-cast.
SWEEP_CAST_PREFIXES: tuple[str, ...] = (
    "predictor.",
    "shared.",
    "router.",
    "cell.V.",
    "cell.gate.",
    "cell.W_out.",
    "cell.units.",
)


class SweepTwin:
    """The model the split window's forward sweep runs (P2.10): a structural copy of the HSN
    whose autocast-cast weights (:data:`SWEEP_CAST_PREFIXES`) are held in the autocast dtype and
    refreshed once per window, and whose other parameters and buffers are the model's own
    tensors. A compiled block of the model itself casts every such weight at every position
    (under autograd it has to: the gradients of a window are summed in the fp32 masters) and
    receives the window's ``W1`` layout as a call argument, which a CUDA graph copies into its
    input buffer at every block; the sweep needs neither. The twin's values are the model's:
    the same casts, made once (``tests/kernels/test_p2_10_window.py``). ``sweep_edge`` and
    ``sweep_w1_flat`` are the window constants as buffers of the twin, so a compiled block reads
    them at static addresses.

    ``fused`` (``hardware.window=fused``) adds the layouts of the fused sweep
    (:func:`_fused_tape_block`): ``sweep_w2_flat``, ``W2`` laid out for its second layer
    (:meth:`~hsn.model.units.UnitBank.forward_static_chosen`), and, when the model's shape
    allows (:func:`sweep_merge_error`), the merged matrix products: maps that read the same
    input share one matmul, with the same operands (a map that reads part of the input has
    zeros elsewhere, which add nothing):

    * ``sweep_front_w1/b1/w2/b2``: ``S`` and ``P`` as one two-layer map of ``[z; q; ctx]``
      (``P`` reads ``q`` only; the second layer is block-diagonal);
    * ``sweep_score_w/b``: the scorer's linear with ``V`` as ``d`` more outputs (``V`` reads
      the state columns of the scorer's features);
    * ``sweep_w1_gate`` / ``sweep_b1_gate``: the unit bank's first layer with the gate's linear
      as ``d`` more outputs (:class:`~hsn.model.cell.WindowConsts`).

    Four matmuls per position fewer; the values agree with the separate products to rounding
    (other matmul shapes)."""

    def __init__(self, model: HSN, dtype: torch.dtype, *, fused: bool = False) -> None:
        self.model = model
        self.dtype = dtype
        self.fused = fused
        self.merged = fused and sweep_merge_error(model) is None
        twin = copy.deepcopy(model)
        twin.requires_grad_(False)
        self.twin = twin
        self._cast: list[tuple[Tensor, Tensor]] = []
        self._shared: list[tuple[nn.Module, str, str]] = []  # (twin owner, leaf, dotted name)
        masters = dict(model.named_parameters())
        for name in [n for n, _ in twin.named_parameters()]:
            owner_name, _, leaf = name.rpartition(".")
            owner = twin.get_submodule(owner_name)
            source = masters[name]
            if source.dtype != dtype and name.startswith(SWEEP_CAST_PREFIXES):
                target = nn.Parameter(source.detach().to(dtype), requires_grad=False)
                owner._parameters[leaf] = target
                self._cast.append((source, target))
            else:
                owner._parameters[leaf] = source
                self._shared.append((owner, leaf, name))
        self._buffers: list[tuple[nn.Module, str, str]] = []
        for name, buffer in model.named_buffers():
            owner_name, _, leaf = name.rpartition(".")
            owner = twin.get_submodule(owner_name)
            owner._buffers[leaf] = buffer
            self._buffers.append((owner, leaf, name))
        cell = model.cell
        units = cell.units
        device = next(model.parameters()).device
        twin.cell.register_buffer(
            "sweep_edge",
            torch.zeros(cell.n_cells, cell.graph.max_in_degree, dtype=dtype, device=device),
            persistent=False,
        )
        twin.cell.units.register_buffer(
            "sweep_w1_flat",
            torch.zeros(units.in_dim, units.n_units * units.hidden, dtype=dtype, device=device),
            persistent=False,
        )
        if fused:
            twin.cell.units.register_buffer(
                "sweep_w2_flat",
                torch.zeros(units.hidden, units.n_units * units.d, dtype=dtype, device=device),
                persistent=False,
            )
        if self.merged:
            pred, shared = model.predictor, model.shared
            net = cast(nn.Linear, model.router.scorer.net)
            hidden = shared.hidden + pred.hidden
            wide = units.n_units * units.hidden + cell.d

            def zeros(*shape: int) -> Tensor:
                return torch.zeros(*shape, dtype=dtype, device=device)

            twin.register_buffer("sweep_front_w1", zeros(shared.in_dim, hidden), persistent=False)
            twin.register_buffer("sweep_front_b1", zeros(hidden), persistent=False)
            twin.register_buffer(
                "sweep_front_w2", zeros(hidden, shared.out_dim + pred.out_dim), persistent=False
            )
            twin.register_buffer(
                "sweep_front_b2", zeros(shared.out_dim + pred.out_dim), persistent=False
            )
            twin.register_buffer(
                "sweep_score_w", zeros(net.in_features, net.out_features + cell.d), persistent=False
            )
            twin.register_buffer(
                "sweep_score_b", zeros(net.out_features + cell.d), persistent=False
            )
            twin.cell.units.register_buffer(
                "sweep_w1_gate", zeros(units.in_dim, wide), persistent=False
            )
            twin.cell.units.register_buffer("sweep_b1_gate", zeros(wide), persistent=False)

    def refresh(self) -> None:
        """Bring the twin up to the model: the cast weights and the window constants are copied
        (inside the model's ``cell.window()``, without autograd), the shared tensors re-bound if
        the model replaced one, the mode followed."""
        model, twin = self.model, self.twin
        if twin.training != model.training:
            twin.train(model.training)
        for source, target in self._cast:
            target.copy_(source)
        masters = dict(model.named_parameters())
        for owner, leaf, name in self._shared:
            if owner._parameters[leaf] is not masters[name]:
                owner._parameters[leaf] = masters[name]
        buffers = dict(model.named_buffers())
        for owner, leaf, name in self._buffers:
            if owner._buffers[leaf] is not buffers[name]:
                owner._buffers[leaf] = buffers[name]
        consts = model.cell.window_consts()
        twin.cell.get_buffer("sweep_edge").copy_(consts.edge_weights)
        cell, units = model.cell, model.cell.units
        if self.fused:
            twin.cell.units.get_buffer("sweep_w2_flat").copy_(units._w2_flat())
        if not self.merged:
            twin.cell.units.get_buffer("sweep_w1_flat").copy_(consts.w1_flat)
            return
        # the merged layouts (class docstring): the blocks that are zero stay zero
        pred, shared = model.predictor, model.shared
        net = cast(nn.Linear, model.router.scorer.net)
        d, hs, du = pred.in_dim, shared.hidden, shared.out_dim
        w1 = twin.get_buffer("sweep_front_w1")
        w1[:, :hs].copy_(shared.fc1.weight.t())
        w1[d : 2 * d, hs:].copy_(pred.fc1.weight.t())  # P reads q, the middle block of [z; q; ctx]
        b1 = twin.get_buffer("sweep_front_b1")
        b1[:hs].copy_(shared.fc1.bias)
        b1[hs:].copy_(pred.fc1.bias)
        w2 = twin.get_buffer("sweep_front_w2")
        w2[:hs, :du].copy_(shared.fc2.weight.t())
        w2[hs:, du:].copy_(pred.fc2.weight.t())
        b2 = twin.get_buffer("sweep_front_b2")
        b2[:du].copy_(shared.fc2.bias)
        b2[du:].copy_(pred.fc2.bias)
        m = int(net.out_features)
        score_w = twin.get_buffer("sweep_score_w")
        score_w[:, :m].copy_(net.weight.t())
        score_w[2 * d : 3 * d, m:].copy_(cell.V.weight.t())  # the features are [u; eps; h; ...]
        twin.get_buffer("sweep_score_b")[:m].copy_(net.bias)
        mh = units.n_units * units.hidden
        w1_gate = twin.cell.units.get_buffer("sweep_w1_gate")
        w1_gate[:, :mh].copy_(consts.w1_flat)
        w1_gate[:, mh:].copy_(cell.gate.weight.t())
        b1_gate = twin.cell.units.get_buffer("sweep_b1_gate")
        b1_gate[:mh].copy_(units.b1.reshape(mh))
        b1_gate[mh:].copy_(cell.gate.bias)


def sweep_merge_error(model: HSN) -> str | None:
    """Why the fused sweep keeps its matrix products separate for this model (``None``: it
    merges them, :class:`SweepTwin`). The merged layouts assume the reference shapes: ``S`` and
    ``P`` two-layer maps with one activation, ``P`` reading the ``q`` block of ``S``'s input, a
    linear scorer over ``[u; eps; h; ...]`` without excluded features, ``V`` without a bias and
    a gate that reads the unit bank's input."""
    pred, shared, cell = model.predictor, model.shared, model.cell
    scorer = model.router.scorer
    d = int(model.cfg.d)
    if type(pred.act) is not type(shared.act) or pred.activation_name != shared.activation_name:
        return "the predictor and the shared transform use different activations"
    if pred.in_dim != d or shared.in_dim < 2 * d:
        return "the predictor does not read the q block of the shared transform's input"
    if scorer.kind != "linear" or not isinstance(scorer.net, nn.Linear) or scorer.exclude:
        return "the scorer is not a plain linear map of its features"
    if scorer.net.bias is None or scorer.net.in_features < 3 * d:
        return "the scorer's features do not hold the cell states at [2d, 3d)"
    if cell.V.bias is not None or cell.V.in_features != d:
        return "the message projection has a bias"
    if not isinstance(cell.gate, nn.Linear) or cell.gate.in_features != cell.units.in_dim:
        return "the gate does not read the unit bank's input"
    if cell.gate.bias is None:
        return "the gate has no bias"
    return None


def _fused_tape_block(
    model: HSN,
    h: Tensor,
    q: Tensor,
    age: Tensor,
    pos: Tensor,
    z_b: Tensor,
    c_b: Tensor,
    s_b: Tensor,
    r_b: Tensor,
    consts: WindowConsts,
    controls: _Controls,
) -> dict[str, Tensor]:
    """:func:`_tape_block` with the fused glue (P2.10, ``hardware.window=fused``). The matrix
    products of a position are :meth:`HSN._position`'s own (the shared path, the scorer, and the
    round's through :meth:`~hsn.model.cell.PDGCell.sparse_products`); what lies between them is
    two kernels that write the block's tape in place:

    * the decision (:func:`hsn.kernels.window.select_tape`: the router's top-K, budget prefix,
      threshold and refresh probes with its tie rules) into record ``i`` of the decision tape;
    * the end of the round (:func:`hsn.kernels.window.update_tape`: candidate, gated update,
      summary, ages) from record ``i`` of the state tape into record ``i + 1``, with the next
      position's reset applied, so that record ``i`` is always the pre-round state a position
      reads and the block's last record is the state it returns.

    No tensor is made per position for the state or the tape, and nothing is stacked. On a
    :class:`SweepTwin` with merged layouts, maps that read the same input share a matmul: ``S``
    with ``P``, the scorer with ``V``, the gate with the unit bank's first layer. No autograd,
    no exploration, no dense-fallback rows (the caller's scope)."""
    router = model.router
    B, n = int(z_b.shape[0]), int(z_b.shape[1])
    N, d = int(h.shape[1]), int(h.shape[2])
    K = int(model.cfg.k_select)
    front_w1 = model._buffers.get("sweep_front_w1")
    front_b1 = model._buffers.get("sweep_front_b1")
    front_w2 = model._buffers.get("sweep_front_w2")
    front_b2 = model._buffers.get("sweep_front_b2")
    score_w = model._buffers.get("sweep_score_w")
    score_b = model._buffers.get("sweep_score_b")
    d_u = int(model.shared.out_dim)
    n_units = int(model.cell.units.n_units)
    reset = r_b.contiguous()
    first = reset[:, 0]
    tape_h = h.new_empty((B, n + 1, N, d))
    tape_q = q.new_empty((B, n + 1, d))
    tape_age = age.new_empty((B, n + 1, N))
    tape_h[:, 0] = h.masked_fill(first[:, None, None], 0)
    tape_q[:, 0] = q.masked_fill(first[:, None], 0)
    tape_age[:, 0] = age.masked_fill(first[:, None], 0)
    cells = age.new_empty((B, n, K))
    units = age.new_empty((B, n, K))
    valid = age.new_empty((B, n, K), dtype=torch.bool)
    refreshed = age.new_empty((B, n, K), dtype=torch.bool)
    theta = h.new_empty((B, n))
    h_sel = h.new_empty((B, n, K, d))
    h_max = h.new_empty((B, n))
    qs = q.new_empty((B, n, d))
    limit = float(model.cfg.budget) * (1.0 + BUDGET_SLACK)
    for i in range(n):
        h_i, q_i, age_i = tape_h[:, i], tape_q[:, i], tape_age[:, i]
        z_t = z_b[:, i]
        x = torch.cat([z_t, q_i, c_b[:, i]], dim=-1)
        if front_w1 is not None and front_b1 is not None:
            assert front_w2 is not None and front_b2 is not None
            hidden = model.shared.act(torch.addmm(front_b1, x, front_w1))
            both = torch.addmm(front_b2, hidden, front_w2)
            u, z_hat = both[:, :d_u], both[:, d_u:]
        else:
            z_hat = model.predictor(q_i)
            u = model.shared(x)
        u_s, eps_s = u.to(h.dtype), (z_t - z_hat).to(h.dtype)
        feats = model.scorer._assemble(u_s, eps_s, h_i, age_i, s_b[:, i])
        vh: Tensor | None = None
        if score_w is not None and score_b is not None:
            scored = torch.addmm(score_b, feats.reshape(B * N, -1), score_w)
            logits = scored[:, :n_units].reshape(B, N, n_units).to(h.dtype)
            vh = scored[:, n_units:].reshape(B, N, d)
        else:
            logits = router(feats).to(h.dtype)
        window_kernels.select_tape(
            logits,
            tape_age,
            router.cost_table,
            cells,
            units,
            valid,
            theta,
            refreshed,
            t=i,
            age_max=int(router.age_max),
            probes=controls.probes,
            lambda_cost=float(router.lambda_cost),
            cell_overhead=float(router.cell_overhead),
            shared_cost=float(router.shared_cost),
            limit=limit,
        )
        state_i = CellGraphState._make(h=h_i, q=q_i, age=age_i, pos=pos)
        gate, y = model.cell.sparse_products(
            u_s, eps_s, state_i, cells[:, i], units[:, i], valid[:, i], consts, vh=vh
        )
        window_kernels.update_tape(
            model.cell, y, gate, tape_h, tape_q, tape_age, cells, valid, reset, h_sel, h_max, qs,
            t=i,
        )  # fmt: skip
    # the position counter: positions since the row's last reset in the block, or pos + n
    steps = torch.arange(1, n + 1, device=pos.device, dtype=pos.dtype)
    last = (reset.to(pos.dtype) * steps).amax(dim=1)
    return {
        "h": tape_h[:, n].contiguous(),
        "q": tape_q[:, n].contiguous(),
        "age": tape_age[:, n].contiguous(),
        "pos": torch.where(last > 0, n + 1 - last, pos + n),
        "tape_h": tape_h[:, :n],
        "tape_q": tape_q[:, :n],
        "tape_age": tape_age[:, :n],
        "tape_hsel": h_sel,
        "tape_theta": theta,
        "qs": qs,
        "sel_cell_idx": cells,
        "sel_unit_idx": units,
        "sel_valid": valid,
        "explored": theta.new_zeros(n),
        "refreshed": refreshed.float().sum(dim=2).mean(dim=0),
        "state_abs_max": h_max.amax(dim=0),
    }


#: The per-position stacks of :func:`_tape_block` that are joined over a window (the final state
#: aside): per-row values ``[B, n, …]`` and per-position batch statistics ``[n]``.
TAPE_KEYS: tuple[str, ...] = (
    "tape_h",
    "tape_q",
    "tape_age",
    "tape_hsel",
    "tape_theta",
    "qs",
    "sel_cell_idx",
    "sel_unit_idx",
    "sel_valid",
    "explored",
    "refreshed",
    "state_abs_max",
)


def _tape_block(
    model: HSN,
    h: Tensor,
    q: Tensor,
    age: Tensor,
    pos: Tensor,
    z_b: Tensor,
    c_b: Tensor,
    s_b: Tensor,
    r_b: Tensor,
    consts: WindowConsts | None,
    mode: StepMode,
    controls: _Controls,
    fused: bool = False,
    dense_rows: Tensor | None = None,
    dense_indices: Tensor | None = None,
) -> dict[str, Tensor]:
    """A block of consecutive positions of the split window's forward sweep (P2.10,
    :mod:`hsn.train.split`): :func:`_block` without autograd and with a tape. Per position it
    applies the column's reset itself, records the pre-round state (``tape_h [B, n, N, d]``,
    ``tape_q [B, n, d]``, ``tape_age [B, n, N]``: what the position's scorer, cell and predictor
    read), runs :meth:`HSN._position` (the capturable routed round) and records the decision
    (``sel_cell_idx`` / ``sel_unit_idx`` / ``sel_valid`` ``[B, n, K]``, the straight-through
    threshold ``tape_theta [B, n]``), the summary after the position (``qs``) and the updated
    states of the selected slots (``tape_hsel [B, n, K, d]``, cell 0 on padded slots). The replay
    recomputes everything else with autograd from these. Sparse hard-gate rounds
    (:func:`hsn.train.split.scope_error`); the rows of ``dense_rows`` (P2.2's dense fallback,
    with ``dense_indices`` as in :func:`_block`) run the dense round, as in any window. ``fused``
    (``hardware.window=fused``, no dense rows) is :func:`_fused_tape_block`: the same tape with
    the glue between the matrix products as two kernels that write it in place.
    ``consts=None``: ``model`` is a :class:`SweepTwin`'s twin and the window constants are its
    buffers."""
    if consts is None:
        bank = model.cell.units
        w1_gate = bank._buffers.get("sweep_w1_gate") if fused else None
        if w1_gate is not None:  # a fused twin with merged products: W1 is its first columns
            consts = WindowConsts(
                edge_weights=model.cell.get_buffer("sweep_edge"),
                w1_flat=w1_gate[:, : bank.n_units * bank.hidden],
                w2_flat=bank.get_buffer("sweep_w2_flat"),
                w1_gate=w1_gate,
                b1_gate=bank.get_buffer("sweep_b1_gate"),
            )
        else:
            consts = WindowConsts(
                edge_weights=model.cell.get_buffer("sweep_edge"),
                w1_flat=bank.get_buffer("sweep_w1_flat"),
                w2_flat=bank._buffers.get("sweep_w2_flat") if fused else None,
            )
    if fused:
        return _fused_tape_block(model, h, q, age, pos, z_b, c_b, s_b, r_b, consts, controls)
    state = CellGraphState._make(h=h, q=q, age=age, pos=pos)
    n = int(z_b.shape[1])
    d = int(h.shape[-1])
    h_pre: list[Tensor] = []
    q_pre: list[Tensor] = []
    age_pre: list[Tensor] = []
    h_sel: list[Tensor] = []
    thetas: list[Tensor] = []
    qs: list[Tensor] = []
    cells: list[Tensor] = []
    units: list[Tensor] = []
    valids: list[Tensor] = []
    explored: list[Tensor] = []
    refreshed: list[Tensor] = []
    state_maxima: list[Tensor] = []
    for i in range(n):
        state = state.reset(r_b[:, i])
        h_pre.append(state.h)
        q_pre.append(state.q)
        age_pre.append(state.age)
        result = model._position(
            state,
            None,
            z_b[:, i],
            c_b[:, i],
            s_b[:, i],
            mode,
            controls,
            dense_rows,
            False,
            None,
            static=True,
            consts=consts,
            dense_indices=dense_indices,
        )
        state = result.state
        assert result.selection is not None
        sel = result.selection
        safe = sel.cell_idx.masked_fill(~sel.valid, 0)
        h_sel.append(state.h.gather(1, safe[:, :, None].expand(-1, -1, d)))
        assert sel.threshold is not None
        thetas.append(sel.threshold)
        qs.append(state.q)
        cells.append(sel.cell_idx)
        units.append(sel.unit_idx)
        valids.append(sel.valid)
        explored.append(sel.explored.any(dim=1).float().mean())
        refreshed.append(sel.refreshed.float().sum(dim=1).mean())
        state_maxima.append(state.h.abs().amax())
    return {
        "h": state.h,
        "q": state.q,
        "age": state.age.contiguous(),
        "pos": state.pos.contiguous(),
        "tape_h": torch.stack(h_pre, dim=1),
        "tape_q": torch.stack(q_pre, dim=1),
        "tape_age": torch.stack(age_pre, dim=1),
        "tape_hsel": torch.stack(h_sel, dim=1),
        "tape_theta": torch.stack(thetas, dim=1),
        "qs": torch.stack(qs, dim=1),
        "sel_cell_idx": torch.stack(cells, dim=1),
        "sel_unit_idx": torch.stack(units, dim=1),
        "sel_valid": torch.stack(valids, dim=1),
        "explored": torch.stack(explored),
        "refreshed": torch.stack(refreshed),
        "state_abs_max": torch.stack(state_maxima),
    }


def _cat_record(parts: Sequence[Tensor]) -> Tensor:
    """Blocks of a :func:`hsn.model.hsn._stack_record` record joined over the window."""
    return torch.cat(list(parts), dim=1 if parts[0].ndim > 1 else 0)


class CompiledWindow:
    """The compiled training window (:func:`compile_window`): called like
    :meth:`HSN.forward` (``tokens``, ``ctx``, ``state``, ``reset``; ``reset_columns`` validated
    and otherwise unused, since every block resets as tensor ops) and returning the same
    ``(logits, state, info)``. ``capture=True`` is the diagnostic window (the per-position cell
    states, gates and written masks of :meth:`HSN.forward`): its own compiled variant, without
    gradients only (a training window never captures). ``compile_seconds`` accumulates the wall
    time of the calls that compiled or recorded (the first call of each new variant), ``calls``
    counts calls."""

    def __init__(self, model: HSN, ctx: DeviceContext, mode: CompileMode, backend: Backend) -> None:
        self.model = model
        self.ctx = ctx
        self.mode = mode
        self.backend = backend
        self.cuda_graphs = _uses_cuda_graphs(ctx, mode, backend)
        self._block = _compiler(mode, backend, ctx.device.type)(_block)
        self._bucket_blocks: dict[int, Any] = {0: self._block}
        #: The capture variant's compiled blocks (their own code objects and recompile budgets).
        self._capture_blocks: dict[int, Any] = {}
        #: The split window's forward sweep (P2.10): compiled on first use (:meth:`tape`), one
        #: compiled function for the native round and one for the fused round.
        self._tape_blocks: dict[tuple[bool, bool, int], Any] = {}
        self._sweep_twins: dict[tuple[torch.dtype, bool], SweepTwin] = {}
        self.calls = 0
        self.compile_seconds = 0.0
        self._variants: set[tuple[object, ...]] = set()

    def variant(
        self, B: int, T: int, dense_bucket: int, capture: bool = False
    ) -> tuple[object, ...]:
        """What selects a compiled graph: batch, block, grad mode, training mode, gate, autocast
        and the padded dense-fallback batch size (and the diagnostic capture)."""
        model = self.model
        key: tuple[object, ...] = (
            B,
            block_size(T),
            torch.is_grad_enabled(),
            model.training,
            model._controls().gate,
            torch.is_autocast_enabled(self.ctx.device.type),
            dense_bucket,
        )
        return (*key, "capture") if capture else key

    def is_new_variant(self, B: int, T: int, dense_bucket: int, capture: bool = False) -> bool:
        """True when the next call with these properties compiles (or records) a graph."""
        key = self.variant(B, T, dense_bucket, capture)
        warm = (*key, "warm")
        if self.cuda_graphs:  # CUDA Graph trees warm up once, then record on the second call
            return key not in self._variants or warm not in self._variants
        return key not in self._variants

    def __call__(
        self,
        tokens: Tensor,
        ctx: Tensor,
        state: CellGraphState,
        reset: Tensor,
        *,
        capture: bool = False,
        reset_columns: Sequence[int] | None = None,
        prev_tokens: Tensor | None = None,
    ) -> WindowOut:
        model = self.model
        if not isinstance(capture, bool):
            raise TypeError(f"capture must be a bool, got {capture!r}")
        if capture and torch.is_grad_enabled():
            raise ValueError(
                "capture=True is the diagnostic window: call it under torch.no_grad() (a "
                "training window returns no per-position states), or call the model's forward"
            )
        B, T = model._check_window_inputs(tokens, ctx, state, reset)
        if reset_columns is not None:
            model._reset_columns(reset, reset_columns)  # validated; every block resets anyway
        mode = model._window_mode()
        dense_rows = model._draw_dense_rows(B, state)
        dense_indices = model._dense_indices(dense_rows, bucket=True)
        dense_bucket = 0 if dense_indices is None else int(dense_indices.numel())
        controls = model._controls()
        key = self.variant(B, T, dense_bucket, capture)
        new = self.is_new_variant(B, T, dense_bucket, capture)
        started = time.perf_counter()
        out = self._run(
            tokens,
            ctx,
            state,
            reset,
            mode,
            controls,
            dense_rows,
            prev_tokens,
            dense_indices,
            capture=capture,
        )
        self.calls += 1
        if new:
            if self.cuda_graphs:
                torch.cuda.synchronize(self.ctx.device)
            self.compile_seconds += time.perf_counter() - started
            self._variants.add((*key, "warm") if key in self._variants else key)
        return out

    def tape(
        self,
        pro: _Prologue,
        state: CellGraphState,
        reset: Tensor,
        *,
        fused: bool = False,
        twin: bool = True,
        dense_rows: Tensor | None = None,
        dense_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """The split window's forward sweep (P2.10, :mod:`hsn.train.split`): the compiled
        :func:`_tape_block` over the window, under ``torch.no_grad()`` in training mode. Returns
        :func:`run_tape`'s tape, cloned out of CUDA-graph memory (the next replay overwrites
        it). Counted in ``calls`` and, on the calls that compile or record, ``compile_seconds``
        like any other variant. The sweep runs a :class:`SweepTwin` (``twin=False``: the model
        itself, for the tests; the fused sweep then keeps its matrix products separate)."""
        model = self.model
        B, T = (int(n) for n in reset.shape)
        use_twin = twin
        bucket = 0 if dense_indices is None else int(dense_indices.numel())
        key: tuple[object, ...] = (
            B,
            block_size(T),
            model.training,
            model._controls().gate,
            torch.is_autocast_enabled(self.ctx.device.type),
            "tape-fused" if fused else "tape",
            use_twin,
            bucket,
        )
        warm = (*key, "warm")
        new = key not in self._variants or (self.cuda_graphs and warm not in self._variants)
        started = time.perf_counter()
        # as the autograd window: one compiled function per padded dense-fallback batch size
        slot = (fused, use_twin, bucket)
        if slot not in self._tape_blocks:
            self._tape_blocks[slot] = _compiler(self.mode, self.backend, self.ctx.device.type)(
                _tape_block
            )
        sweep: SweepTwin | None = None
        if use_twin:
            # the dtype autocast casts to in this call (the masters' own without autocast)
            device_type = self.ctx.device.type
            dtype = (
                torch.get_autocast_dtype(device_type)
                if torch.is_autocast_enabled(device_type)
                else next(model.parameters()).dtype
            )
            if (dtype, fused) not in self._sweep_twins:
                self._sweep_twins[dtype, fused] = SweepTwin(model, dtype, fused=fused)
            sweep = self._sweep_twins[dtype, fused]
        if self.cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()  # type: ignore[no-untyped-call]
        tape = run_tape(
            self._tape_blocks[slot],
            model,
            pro,
            state,
            reset,
            block_size(T),
            fused=fused,
            twin=sweep,
            dense_rows=dense_rows,
            dense_indices=dense_indices,
        )
        if self.cuda_graphs:  # the final state would be overwritten by the next replay
            for name in ("h", "q", "age", "pos"):
                tape[name] = tape[name].clone()
        self.calls += 1
        if new:
            if self.cuda_graphs:
                torch.cuda.synchronize(self.ctx.device)
            self.compile_seconds += time.perf_counter() - started
            self._variants.add(warm if key in self._variants else key)
        return tape

    def _run(
        self,
        tokens: Tensor,
        ctx: Tensor,
        state: CellGraphState,
        reset: Tensor,
        mode: StepMode,
        controls: _Controls,
        dense_rows: Tensor | None,
        prev_tokens: Tensor | None = None,
        dense_indices: Tensor | None = None,
        *,
        capture: bool = False,
    ) -> WindowOut:
        model = self.model
        T = int(tokens.shape[1])
        blk = block_size(T)
        pro = model._window_prologue(
            tokens, ctx, state, reset, mode, reset_first=True, prev_tokens=prev_tokens
        )
        if self.cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()  # type: ignore[no-untyped-call]
        h, q, age, pos = state.h, state.q, state.age, state.pos
        if pro.counters is not None:  # the dense window: the first block starts at the start age
            age = pro.counters[0]
        if torch.is_grad_enabled() and not (h.requires_grad or q.requires_grad):
            # One graph for every block: the later blocks receive a state that requires grad, so
            # the first block's detached carried state becomes a leaf that does too (its gradient
            # is computed and dropped). Otherwise the first block is a second graph to compile.
            h = h.detach().requires_grad_(True)
            q = q.detach().requires_grad_(True)
        blocks: list[dict[str, Tensor]] = []
        bucket = 0 if dense_indices is None else int(dense_indices.numel())
        table = self._capture_blocks if capture else self._bucket_blocks
        if bucket not in table:
            # Each shape bucket has its own Dynamo code object/recompile budget. A B32 window
            # needs at most six fallback shapes, independent of which row IDs were drawn.
            table[bucket] = _compiler(self.mode, self.backend, self.ctx.device.type)(_block)
        block = table[bucket]
        with model.cell.window():
            consts = model.cell.window_consts()
            for start in range(0, T, blk):
                cols = slice(start, start + blk)
                dense_in: tuple[Tensor, Tensor] | None = None
                if pro.counters is not None:
                    _, _, positions, positions_after = pro.counters
                    dense_in = (positions[:, cols], positions_after[:, cols])
                out = block(
                    model,
                    h,
                    q,
                    age,
                    pos,
                    pro.z[:, cols],
                    pro.context[:, cols],
                    pro.scalars[:, cols],
                    reset[:, cols],
                    dense_in,
                    dense_rows,
                    dense_indices,
                    consts,
                    mode,
                    controls,
                    capture,
                )
                h, q, age, pos = out["h"], out["q"], out["age"], out["pos"]
                blocks.append(out)
        if self.cuda_graphs:  # out of graph memory: the next replay overwrites the outputs
            h, q, age, pos = h.clone(), q.clone(), age.clone(), pos.clone()
        final = CellGraphState._make(h=h, q=q, age=age, pos=pos)

        def joined(key: str) -> Tensor:
            return torch.cat([b[key] for b in blocks], dim=1)

        routed: dict[str, Tensor] | None = None
        if "sel_cell_idx" in blocks[0]:
            routed = {
                key: _cat_record([b["sel_" + key] for b in blocks]) for key in SELECTION_RECORD_KEYS
            }
        return model._window_epilogue(
            pro,
            final,
            z_hats=joined("z_hat"),
            epss=joined("eps"),
            us=joined("u"),
            qs=joined("qs"),
            hs=joined("cap_h") if capture else None,
            state_abs_max=torch.cat([b["state_abs_max"] for b in blocks]),
            unit_logits=joined("unit_logits_d"),
            routed=routed,
            gates=joined("cap_gate") if capture else None,
            written=joined("cap_written") if capture else None,
            readouts=joined("readouts") if model.cfg.selected_readout else None,
        )


def run_tape(
    block: Callable[..., dict[str, Tensor]],
    model: HSN,
    pro: _Prologue,
    state: CellGraphState,
    reset: Tensor,
    blk: int,
    *,
    fused: bool = False,
    twin: SweepTwin | None = None,
    dense_rows: Tensor | None = None,
    dense_indices: Tensor | None = None,
) -> dict[str, Tensor]:
    """The forward sweep of a split window (:func:`_tape_block` as ``block``, compiled or not)
    over blocks of ``blk`` positions, without autograd: the joined tape (:data:`TAPE_KEYS`) and
    the final state (``h``, ``q``, ``age``, ``pos``). The prologue's stacks are read detached.
    With a ``twin`` the blocks run the twin (refreshed here) and read its window constants."""
    if torch.is_grad_enabled():
        raise RuntimeError("the split window's forward sweep runs under torch.no_grad()")
    T = int(reset.shape[1])
    mode = model._window_mode()
    # The sweep takes the hard decision only (gate=None): the same slots, threshold and probes
    # as the step's gate, whose gradient path (the soft unit weights and cell mask) the replay
    # rebuilds from the recorded slots. Nothing here would read it.
    controls = model._controls()._replace(gate=None)
    z, context, scalars = pro.z.detach(), pro.context.detach(), pro.scalars.detach()
    h, q, age, pos = state.h, state.q, state.age, state.pos
    blocks: list[dict[str, Tensor]] = []
    with model.cell.window():
        consts: WindowConsts | None = None
        runner = model
        if twin is not None:
            twin.refresh()
            runner = twin.twin
        else:
            consts = model.cell.window_consts()
        for start in range(0, T, blk):
            cols = slice(start, start + blk)
            out = block(
                runner,
                h,
                q,
                age,
                pos,
                z[:, cols],
                context[:, cols],
                scalars[:, cols],
                reset[:, cols],
                consts,
                mode,
                controls,
                fused,
                dense_rows,
                dense_indices,
            )
            h, q, age, pos = out["h"], out["q"], out["age"], out["pos"]
            blocks.append(out)
    tape = {key: _cat_record([b[key] for b in blocks]) for key in TAPE_KEYS}
    tape.update(h=h, q=q, age=age, pos=pos)
    return tape


def _eager_window(model: HSN) -> Callable[..., WindowOut]:
    def window(
        tokens: Tensor,
        ctx: Tensor,
        state: CellGraphState,
        reset: Tensor,
        *,
        capture: bool = False,
        reset_columns: Sequence[int] | None = None,
        prev_tokens: Tensor | None = None,
    ) -> WindowOut:
        result: WindowOut = model(
            tokens,
            ctx,
            state,
            reset,
            capture=capture,
            reset_columns=reset_columns,
            prev_tokens=prev_tokens,
        )
        return result

    return window


def compile_window(
    model: HSN,
    ctx: DeviceContext,
    mode: CompileMode,
    *,
    backend: Backend = "inductor",
    grad_accum: int = 1,
) -> Callable[..., WindowOut]:
    """The training window of ``model`` at compile level ``mode`` (module docstring): ``none``
    returns the eager ``model.forward``; otherwise a :class:`CompiledWindow` (after
    :func:`check_compile_scope`). On MPS a backend that cannot compile falls back to eager with a
    logged notice. ``backend`` is ``inductor`` (what ``hardware.compile_mode`` means); tests pass
    ``aot_eager`` where inductor's C++ toolchain is absent."""
    mode = _check_mode(mode)
    if mode == "none":
        return _eager_window(model)
    check_compile_scope(model, grad_accum=grad_accum)
    if ctx.device.type == "mps":
        ok, reason = _mps_supported(backend, ctx.device)
        if not ok:
            log.warning(
                "torch.compile (%s, mode=%s) is not supported on %s (%s); the training window "
                "runs eager (L0)",
                backend,
                mode,
                ctx.device,
                reason,
            )
            return _eager_window(model)
    return CompiledWindow(model, ctx, mode, backend)


# --------------------------------------------------------------------------------------------------
# The serving step
# --------------------------------------------------------------------------------------------------


def _serve(
    model: HSN,
    x_t: Tensor,
    ctx_t: Tensor,
    h: Tensor,
    q: Tensor,
    age: Tensor,
    pos: Tensor,
    mode: StepMode,
    static: bool,
    prev_tokens: Tensor | None = None,
) -> Tensor:
    """One serving position: :meth:`HSN._step_impl` on the state ``(h, q, age, pos)``, written
    back into those tensors in place; returns the logits ``[B, V]``."""
    state = CellGraphState._make(h=h, q=q, age=age, pos=pos)
    out = model._step_impl(
        x_t, ctx_t, state, mode, None, None, static=static, prev_tokens=prev_tokens
    )
    h.copy_(out.state.h)
    q.copy_(out.state.q)
    age.copy_(out.state.age)
    pos.copy_(out.state.pos)
    return out.logits


class ServingStep:
    """The serving step of :func:`compile_step`: ``step(x_t [B], ctx_t [B], state) -> logits
    [B, V]`` under ``no_grad``, updating the preallocated ``state`` in place (build it with
    ``model.init_state(B)`` and pass the same object every call: CUDA Graphs replay on its
    buffers, so a new state object costs one recording). The model must be in evaluation mode
    (serving never trains). Under CUDA Graphs the logits are graph memory: read them before the
    next call. ``compile_seconds`` accumulates the calls that compiled or recorded."""

    def __init__(self, model: HSN, ctx: DeviceContext, mode: CompileMode, backend: Backend) -> None:
        self.model = model
        self.ctx = ctx
        self.mode = mode
        self.cuda_graphs = _uses_cuda_graphs(ctx, mode, backend)
        self.static = mode != "none"
        self._fn: Callable[..., Tensor] = (
            _serve if mode == "none" else _compiler(mode, backend, ctx.device.type)(_serve)
        )
        self.calls = 0
        self.compile_seconds = 0.0
        self._marked: set[int] = set()
        self._seen: dict[tuple[object, ...], int] = {}

    def __call__(
        self,
        x_t: Tensor,
        ctx_t: Tensor,
        state: CellGraphState,
        *,
        prev_tokens: Tensor | None = None,
    ) -> Tensor:
        model = self.model
        if model.training:
            raise RuntimeError("the serving step runs in evaluation mode: call model.eval() first")
        model._check_step_inputs(x_t, ctx_t, state)
        if model.cfg.key_address:
            prev_tokens = model._previous_tokens(x_t, prev_tokens)
        for t in (state.h, state.q, state.age, state.pos):
            if t.requires_grad:
                raise RuntimeError("the serving state must not require grad (detach it)")
        if self.static and id(state) not in self._marked:
            for t in (state.h, state.q, state.age, state.pos):
                torch._dynamo.mark_static_address(t)
            self._marked.add(id(state))
        key = (id(state), state.batch_size, torch.is_autocast_enabled(self.ctx.device.type))
        seen = self._seen.get(key, 0)
        new = self.mode != "none" and seen < (2 if self.cuda_graphs else 1)
        started = time.perf_counter()
        with torch.no_grad(), model.cell.graph.step():
            if self.cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()  # type: ignore[no-untyped-call]
            logits = self._fn(
                model,
                x_t,
                ctx_t,
                state.h,
                state.q,
                state.age,
                state.pos,
                model._window_mode(),
                self.static,
                prev_tokens,
            )
        self.calls += 1
        self._seen[key] = seen + 1
        if new:
            if self.cuda_graphs:
                torch.cuda.synchronize(self.ctx.device)
            self.compile_seconds += time.perf_counter() - started
        return logits


def compile_step(
    model: HSN, ctx: DeviceContext, mode: CompileMode, *, backend: Backend = "inductor"
) -> Callable[..., Tensor]:
    """The serving step of ``model`` at compile level ``mode`` (:class:`ServingStep`; module
    docstring): ``none`` is the eager in-place step (L0); otherwise the position is compiled and,
    on CUDA, captured as a CUDA Graph on the state's preallocated buffers. The compiled scope
    (:func:`check_compile_scope`) applies; on MPS a backend that cannot compile falls back to the
    eager step with a logged notice."""
    mode = _check_mode(mode)
    if mode != "none":
        check_compile_scope(model)
        if ctx.device.type == "mps":
            ok, reason = _mps_supported(backend, ctx.device)
            if not ok:
                log.warning(
                    "torch.compile (%s, mode=%s) is not supported on %s (%s); the serving step "
                    "runs eager (L0)",
                    backend,
                    mode,
                    ctx.device,
                    reason,
                )
                return ServingStep(model, ctx, "none", backend)
    return ServingStep(model, ctx, mode, backend)
