"""The fused word-event block on CUDA (``docs/wev_fused_event.md``): the forward sweep's
kernels (``kernels/cuda/hsn_event.cu``, in the window fatbin of
``scripts/build_window_kernels.py``) with a slot's matrix products between them, the reverse
sweep of P2.10c on the same buffers, and the kernel of the local backward.

A slot's forward pass is ``hsn_ev_finish`` (the end of the slot before, the reset, the front's
first layer), two small products (the front's second layer; the state's: ``V h`` and the
state's term of the logits), ``hsn_ev_decide`` (the scorer, the decision, the messages, the
cell input), the unit bank's first product, ``hsn_ev_pick``, the unit bank's second product.

:class:`EventKernels` holds the buffers of one block shape. Its :meth:`EventKernels.forward` is
:func:`hsn.train.event_fused.forward_reference` (the same arguments and tape), its
:meth:`EventKernels.sweep` is :func:`hsn.train.event_fused.sweep_reference` and its
:meth:`EventKernels.pairs` the cell norm's backward of :func:`hsn.train.event_fused.local_backward`;
:class:`EventKernelSet` keeps one per number of event slots and is what
:func:`hsn.train.event_fused.fused_event_block` is handed. fp32 state; the products run in fp32
or bf16.

The tape is the buffers themselves: a block's tape is valid until the next forward sweep of a
block of the same shape (one block per window and one backward pass per forward, as a training
step runs them).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from hsn.kernels import window as window_kernels
from hsn.model.router import BUDGET_SLACK
from hsn.train.state_gradient import CellConstants
from hsn.train.state_sweep import ACTIVATION_CODES, ReverseSweep, SweepShape, out_adjacency

if TYPE_CHECKING:
    from hsn.model.hsn import HSN
    from hsn.train.event_fused import EventSpec, EventWeights

__all__ = [
    "HEAD_THREADS",
    "PICK_THREADS",
    "EventKernelSet",
    "EventKernels",
    "decide_shared_words",
    "finish_shared_words",
    "unsupported",
]

#: Threads of a row's block in ``hsn_ev_finish`` and ``hsn_ev_decide`` (one per logit of the
#: row) and in ``hsn_ev_pick``.
HEAD_THREADS = 512
PICK_THREADS = 512


def finish_shared_words(spec: EventSpec, threads: int = HEAD_THREADS) -> int:
    """The dynamic shared memory of ``hsn_ev_finish`` in 4-byte words (its header's layout)."""
    N, K, d = spec.N, spec.K, spec.d
    return N * (d + 1) + d + N + 4 * K + 3 * K * d + threads // 32


def decide_shared_words(spec: EventSpec, n_scalars: int) -> int:
    """The dynamic shared memory of ``hsn_ev_decide`` in 4-byte words (its header's layout)."""
    N, M, K, d = spec.N, spec.M, spec.K, spec.d
    warps = (N + 31) // 32
    features = 3 * d + 1 + n_scalars
    return (
        M * features + M * spec.A * d + N * M + 3 * d + M + M * spec.A + 5 * N + 2 * M + 4 * K
        + 1 + 3 * warps + 2 * K + 2 * K * spec.D + 2 * K * d + N * n_scalars
    )  # fmt: skip


def unsupported(spec: EventSpec, n_scalars: int, threads: int = HEAD_THREADS) -> str | None:
    """Why the kernels do not take this shape, or ``None``."""
    N, M, K, d = spec.N, spec.M, spec.K, spec.d
    if M & (M - 1) or M > 32:
        return f"n_units={M} must be a power of two <= 32"
    if N * M > threads or 2 * d > threads or N > threads or K * spec.D > threads:
        return f"n_cells*n_units={N * M} and 2d={2 * d} must fit {threads} threads"
    if spec.D > 8:
        return "the message graph's in-degree must be at most 8"
    if M * 16 + M * spec.A * 4 > threads or K * 16 > threads:
        return "the scorer's row terms need 16 lanes a unit and 4 a gate row"
    words = max(finish_shared_words(spec, threads), decide_shared_words(spec, n_scalars))
    if 4 * words > window_kernels._MAX_SHARED_BYTES:
        return "a kernel's shared memory exceeds 48 KiB at this shape"
    if spec.unit_act not in ACTIVATION_CODES or spec.front_act not in ACTIVATION_CODES:
        return "unknown activation"
    return None


class EventKernels:
    """The buffers and launches of the fused event block for one shape (module docstring).

    ``bias_in_product``: the two products add their bias (``addmm``; one rounding, as the
    compiled round's) instead of the kernels adding it to the columns they read (``mm``; under
    bf16 one more rounding). ``graph``: capture the reverse sweep's loop in a CUDA graph at
    its first block. ``forward_graph``: capture the forward sweep's launches in a CUDA graph
    at the first block.

    ``share``: kernels of the same sizes and at least as many slots whose memory these use
    (every buffer is the leading part of the other's): the blocks of fewer slots
    (:meth:`at`) need no buffers of their own, only their own CUDA graphs."""

    B: int
    E: int
    S: int
    back: ReverseSweep
    pad_src: Tensor
    # the buffers (``sizes`` in ``__init__``)
    h: Tensor
    q: Tensor
    age: Tensor
    q_pre: Tensor
    reset: Tensor
    live: Tensor
    z: Tensor
    static: Tensor
    scal: Tensor
    u: Tensor
    z_hat: Tensor
    eps: Tensor
    logits: Tensor
    sel_cells: Tensor
    sel_units: Tensor
    theta: Tensor
    soft: Tensor
    prob: Tensor
    ties: Tensor
    cost: Tensor
    refreshed: Tensor
    m: Tensor
    v: Tensor
    readout: Tensor
    abs_max: Tensor
    first: Tensor
    out: Tensor
    act_f: Tensor
    uz: Tensor
    h_op: Tensor
    vh: Tensor
    act: Tensor
    d_hid: Tensor
    front_static: Tensor
    front_b1: Tensor
    front_q: Tensor
    front_w2_p: Tensor
    front_b2_p: Tensor
    v_p: Tensor
    score_w: Tensor
    score_b: Tensor
    gate_w: Tensor
    edge: Tensor
    norm_gain: Tensor
    cand_gain: Tensor
    cost_table: Tensor
    w1: Tensor
    b1: Tensor
    b1_f: Tensor
    w2: Tensor
    b2: Tensor
    b2_f: Tensor

    def __init__(
        self,
        spec: EventSpec,
        weights: EventWeights,
        consts: CellConstants,
        *,
        B: int,
        E: int,
        n_scalars: int,
        device: torch.device,
        product_dtype: torch.dtype = torch.float32,
        bias_in_product: bool = True,
        graph: bool = True,
        forward_graph: bool = False,
        share: EventKernels | None = None,
    ) -> None:
        front_hidden = int(weights.front_q.shape[0])
        reason = unsupported(spec, n_scalars)
        if reason is not None:
            raise ValueError(f"the event kernels do not take this shape: {reason}")
        if not window_kernels.available():
            raise RuntimeError("window kernels missing; run scripts/build_window_kernels.py")
        if not window_kernels.prepared(device):
            window_kernels.prepare(device.index or 0)
        if product_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("the products run in fp32 or bf16")
        if share is not None and (share.B != B or share.E < E or share.S != n_scalars):
            raise ValueError("the kernels to share with are of another shape or fewer slots")
        self.spec = spec
        self.B, self.E, self.S = B, E, n_scalars
        self.device = device
        self.product_dtype = product_dtype
        self.half = product_dtype == torch.bfloat16
        self.bias_in_product = bias_in_product
        self.graph = graph
        self.forward_graph = forward_graph
        N, M, K, d, H, n, D = spec.N, spec.M, spec.K, spec.d, spec.H, spec.in_dim, spec.D
        self.Hf = front_hidden
        self.Hs = spec.front_s  # S's block of the front's hidden layer
        mask = consts.pad_mask
        if mask is None:
            raise ValueError("the reverse sweep needs the graph's padding mask (pad_mask)")
        d_out = int(out_adjacency(consts.pad_src, mask)[0].shape[1])
        shape = SweepShape(
            B=B, T=E, K=K, N=N, d=d, H=H, M=M, n_fast=0, n_read=0, in_dim=n, d_out=d_out
        )
        if window_kernels.back_unsupported(shape) is not None:
            raise ValueError("the reverse sweep's kernels do not take this shape")
        #: The reverse sweep; its record buffers are the forward sweep's tape.
        self.back = ReverseSweep(
            shape, consts, device=device, state_dtype=torch.float32,
            product_dtype=product_dtype, kernels=True,
            share=None if share is None else share.back,
        )  # fmt: skip
        Hf, S, A = self.Hf, n_scalars, max(M * spec.A, 1)
        static_in = int(weights.front_static.shape[1])
        width = int(weights.score_w.shape[1])
        sizes: dict[str, tuple[str, tuple[int, ...]]] = {
            # the state, slot by slot (record t: what slot t reads; record E: after the block)
            # (h slot-major: the block's states are one contiguous matrix for the scorer's
            # gradient)
            "h": ("f32", (E + 1, B, N, d)),
            "q": ("f32", (B, E + 1, d)),
            "age": ("i64", (B, E + 1, N)),
            "q_pre": ("f32", (B, E, d)),
            "reset": ("u8", (B, E)),
            "live": ("u8", (B, E)),
            "z": ("f32", (B, E, d)),
            "static": ("prod", (B * E, Hf)),
            "scal": ("f32", (B, E, N, S)),
            # the tape beside the reverse sweep's record
            "u": ("f32", (B, E, d)),
            "z_hat": ("f32", (B, E, d)),
            "eps": ("f32", (B, E, d)),
            "logits": ("f32", (B, E, N, M)),
            "sel_cells": ("i64", (B, E, K)),
            "sel_units": ("i64", (B, E, K)),
            "theta": ("f32", (B, E)),
            "soft": ("f32", (B, E, K)),
            "prob": ("f32", (B, E, K)),
            "ties": ("u8", (B, E, K, M)),
            "cost": ("f32", (B, E)),
            "refreshed": ("f32", (B, E)),
            "m": ("f32", (B, E, K, d)),
            "v": ("prod", (E, B, K, n)),
            "readout": ("f32", (B, E, d)),
            "abs_max": ("f32", (B, E)),
            # the products' results and the second one's operand
            "first": ("prod", (B * K, M * H + d)),
            "out": ("prod", (B * K, M * d)),
            # the operands and results of a slot's two small products (the front's second
            # layer, V h)
            "act_f": ("prod", (B, Hf)),
            "uz": ("prod", (B, 2 * d)),
            # (the state's product runs in fp32 on values rounded as the products read them:
            # its V h is rounded where it is read, and the state's term of the logits stays a
            # partial sum of the scorer's one product)
            "h_op": ("f32", (B * N, d)),
            "vh": ("f32", (B * N, d + M)),
            "act": ("prod", (E, B, K, H)),  # the second product's operands, slot by slot
            "d_hid": ("prod", (B, E, K, H)),  # the reverse sweep's dhid of every slot
            # the weights, as the kernels and the products read them
            "front_static": ("prod", (Hf, static_in)),
            "front_b1": ("prod", (Hf,)),
            # (transposed where a kernel's lane index is the weight's row: hsn_event.cu)
            "front_q": ("f32", (d, Hf)),
            "front_w2_p": ("prod", (Hf, 2 * d)),
            "front_b2_p": ("prod", (2 * d,)),
            "v_p": ("f32", (d, d + M)),  # [V' | W_h']: the state's product
            "score_w": ("f32", (M, width)),
            "score_b": ("f32", (M,)),
            "gate_w": ("f32", (d, A)),
            "edge": ("f32", (N, D)),
            "norm_gain": ("f32", (n,)),
            "cand_gain": ("f32", (d,)),
            "cost_table": ("f32", (M,)),
            "w1": ("prod", (n, M * H + d)),
            "b1": ("prod", (M * H + d,)),
            "b1_f": ("f32", (M * H + d,)),
            "w2": ("prod", (H, M * d)),
            "b2": ("prod", (M * d,)),
            "b2_f": ("f32", (M * d,)),
        }
        kinds = {
            "f32": torch.float32, "prod": product_dtype, "i64": torch.int64, "u8": torch.bool
        }  # fmt: skip
        for name, (kind, size) in sizes.items():
            if share is None:
                buffer = torch.zeros(*size, dtype=kinds[kind], device=device)
            else:
                whole = getattr(share, name).reshape(-1)
                count = 1
                for extent in size:
                    count *= extent
                buffer = whole[:count].view(*size)
            setattr(self, name, buffer)
        self.back.d_hid = self.d_hid
        self.pad_src = (
            spec.pad_src.to(device=device, dtype=torch.int32).contiguous()
            if share is None
            else share.pad_src
        )
        self._consts = consts
        self._weights = weights
        self._children: dict[int, EventKernels] = {}
        self.calls = 0
        #: Measurements only: ``hsn_ev_decide`` returns after this phase (0: the whole kernel).
        self.stop_after = 0
        self._forward_graph: torch.cuda.CUDAGraph | None = None

    def at(self, slots: int) -> EventKernels:
        """The kernels of a block of ``slots`` slots on these buffers (``slots`` at most this
        shape's): the same memory laid out for fewer slots, with its own CUDA graphs. A
        block's tape is valid until the next forward sweep of any of them."""
        if slots == self.E:
            return self
        if not 0 < slots < self.E:
            raise ValueError(f"these buffers hold {self.E} slots, not {slots}")
        child = self._children.get(slots)
        if child is None:
            child = EventKernels(
                self.spec, self._weights, self._consts, B=self.B, E=slots, n_scalars=self.S,
                device=self.device, product_dtype=self.product_dtype,
                bias_in_product=self.bias_in_product, graph=self.graph,
                forward_graph=self.forward_graph, share=self,
            )  # fmt: skip
            self._children[slots] = child
        return child

    def blocks(self) -> list[EventKernels]:
        """These kernels and those of fewer slots on their buffers."""
        return [self, *self._children.values()]

    # -- the weights of a window ------------------------------------------------------------------

    def load_weights(self, model: HSN, w: EventWeights) -> None:
        """The window's weights into the buffers. ``w`` already holds the values the products
        read (:func:`hsn.train.event_fused.event_weights` with the products' dtype)."""
        self.front_static.copy_(w.front_static)
        self.front_b1.copy_(w.front_b1)
        self.front_q.copy_(w.front_q.t())
        self.front_w2_p.copy_(w.front_w2.t())
        self.front_b2_p.copy_(w.front_b2)
        self.score_w.copy_(w.score_w)
        self.score_b.copy_(w.score_b)
        if w.gate_w is not None:
            self.gate_w.copy_(w.gate_w.t())
        d = self.spec.d
        self.v_p[:, :d].copy_(w.v.t())
        self.v_p[:, d:].copy_(w.score_w[:, 2 * d : 3 * d].t())
        self.edge.copy_(w.edge)
        self.norm_gain.copy_(w.norm_gain)
        self.cand_gain.copy_(w.cand_gain)
        self.w1.copy_(w.w1)
        self.b1.copy_(w.b1)
        self.b1_f.copy_(w.b1)
        self.w2.copy_(w.w2)
        self.b2.copy_(w.b2)
        self.b2_f.copy_(w.b2)
        self.cost_table.copy_(model.router.cost_table.float())
        router = model.router
        self._costs = (
            float(router.lambda_cost), float(router.cell_overhead), float(router.shared_cost)
        )  # fmt: skip

    # -- the forward sweep ------------------------------------------------------------------------

    def _finish(self, t: int) -> None:
        """``hsn_ev_finish``: finish slot ``t − 1`` (``t > 0``) and start slot ``t`` (``t < E``)."""
        spec, back = self.spec, self.back
        E = self.E
        flags = (1 if t < E else 0) | (2 if t > 0 else 0) | (4 if self.half else 0)
        flags |= 64 if self.bias_in_product else 0
        tensors = [
            self.h, self.q, self.q_pre, self.age, self.reset, self.live, self.static,
            self.front_q, back.cells, back.units, back.valid, self.out, self.b2_f,
            self.cand_gain, back.b, back.h_old, back.y, back.c, self.readout, self.abs_max,
            self.act_f, self.h_op,
        ]  # fmt: skip
        ints = [
            self.B, E, spec.N, spec.M, spec.K, spec.d, self.Hf, t, flags,
            ACTIVATION_CODES[spec.front_act],
        ]  # fmt: skip
        window_kernels._extension().launch(
            "hsn_ev_finish", self.B, HEAD_THREADS, 4 * finish_shared_words(spec), tensors, ints,
            [spec.cand_eps],
        )  # fmt: skip

    def _decide(self, t: int) -> None:
        """``hsn_ev_decide``: the scorer, the decision, the messages and the cell input."""
        spec, back = self.spec, self.back
        flags = (4 if self.half else 0) | (8 if spec.probes else 0)
        flags |= 16 if spec.gate_input == "summary" else 0
        flags |= 32 if spec.gate_input != "none" else 0
        lam, overhead, shared_cost = self._costs
        tensors = [
            self.h, self.q_pre, self.age, self.live, self.z, self.uz, self.scal, self.score_w,
            self.score_b, self.gate_w, self.vh, self.edge, self.pad_src, self.norm_gain,
            self.cost_table, self.h_op, self.u, self.z_hat, self.eps, self.logits, back.cells,
            back.units, back.valid, self.sel_cells, self.sel_units, self.theta, self.soft,
            self.prob, self.ties, self.cost, self.refreshed, back.h_old, self.m, back.rms,
            self.v,
        ]  # fmt: skip
        ints = [
            self.B, self.E, spec.N, spec.M, spec.K, spec.d, spec.in_dim, spec.D, spec.A, self.S,
            t, flags, spec.age_max, self.stop_after,
        ]  # fmt: skip
        floats = [
            spec.tau, spec.norm_eps, lam, overhead, shared_cost,
            spec.budget * (1.0 + BUDGET_SLACK),
        ]  # fmt: skip
        window_kernels._extension().launch(
            "hsn_ev_decide", self.B, HEAD_THREADS, 4 * decide_shared_words(spec, self.S),
            tensors, ints, floats,
        )  # fmt: skip

    def _front(self) -> None:
        """A slot's two small products: ``[u; ẑ] = act_f · W2'ᵀ + b2'``, and the state's:
        ``h · [Vᵀ | W_hᵀ]``, ``V h`` of every cell and the state's term of every logit."""
        torch.addmm(self.front_b2_p, self.act_f, self.front_w2_p, out=self.uz)
        torch.mm(self.h_op, self.v_p, out=self.vh)

    def _pick(self, t: int) -> None:
        spec, back = self.spec, self.back
        flags = (1 if self.half else 0) | (2 if self.bias_in_product else 0)
        window_kernels._extension().launch(
            "hsn_ev_pick", self.B, PICK_THREADS, 4 * spec.K,
            [self.first, self.b1_f, back.units, back.hidden, self.act, back.gate_pre, back.b],
            [self.B, self.E, spec.K, spec.d, spec.H, spec.M, t, flags,
             ACTIVATION_CODES[spec.unit_act]],
            [],
        )  # fmt: skip

    def _slots(self) -> None:
        """The launches of a block's forward sweep: per slot the finish of the slot before and
        the start of this one, the front's second layer and ``V h``, the decision, the first
        product, the pick, the second product; then the finish of the last slot."""
        B, K, n, H = self.B, self.spec.K, self.spec.in_dim, self.spec.H
        for t in range(self.E):
            self._finish(t)
            self._front()
            self._decide(t)
            operand = self.v[t].view(B * K, n)
            if self.bias_in_product:
                torch.addmm(self.b1, operand, self.w1, out=self.first)
            else:
                torch.mm(operand, self.w1, out=self.first)
            self._pick(t)
            act = self.act[t].view(B * K, H)
            if self.bias_in_product:
                torch.addmm(self.b2, act, self.w2, out=self.out)
            else:
                torch.mm(act, self.w2, out=self.out)
        self._finish(self.E)

    def capture_forward(self, warm_up: int = 2) -> None:
        """Capture the forward sweep's launches in a CUDA graph (the buffers must hold a
        block's inputs and weights). Later sweeps replay it."""
        for _ in range(warm_up):
            self._slots()
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._slots()
        self._forward_graph = graph

    def load_inputs(
        self,
        w: EventWeights,
        h: Tensor,
        q: Tensor,
        age: Tensor,
        z: Tensor,
        ctx: Tensor,
        scalars: Tensor,
        reset: Tensor,
        live: Tensor,
    ) -> None:
        """A block's inputs into the buffers (the first ``E`` slots of what is given), and the
        front's static part: ``[z; ctx] · W1'ᵀ + b1'`` for all slots as one product."""
        B, E = self.B, self.E
        self.h[0].copy_(h)
        self.q[:, 0].copy_(q)
        self.age[:, 0].copy_(age)
        self.z.copy_(z[:, :E])
        self.scal.copy_(scalars[:, :E])
        self.reset.copy_(reset[:, :E])
        self.live.copy_(live[:, :E])
        pd = self.product_dtype
        x = torch.cat([z[:, :E].to(pd), ctx[:, :E].to(pd)], dim=-1).reshape(B * E, -1)
        torch.addmm(self.front_b1, x, self.front_static.t(), out=self.static)

    def forward(
        self,
        model: HSN,
        w: EventWeights,
        spec: EventSpec,
        h: Tensor,
        q: Tensor,
        age: Tensor,
        z: Tensor,
        ctx: Tensor,
        scalars: Tensor,
        reset: Tensor,
        live: Tensor,
        *,
        product_dtype: torch.dtype | None = None,
        cell_consts: CellConstants | None = None,
        slots: int | None = None,
    ) -> dict[str, Tensor]:
        """:func:`hsn.train.event_fused.forward_reference` on the card: the tape of the first
        ``E`` slots of the block (``slots``, when given, must be these kernels' ``E``:
        :meth:`at`), as views of the buffers."""
        if int(z.shape[0]) != self.B or int(z.shape[1]) < self.E:
            raise ValueError("the block's shape is not the one these buffers were built for")
        if int(scalars.shape[-1]) != self.S or (slots is not None and slots != self.E):
            raise ValueError("the block's shape is not the one these buffers were built for")
        if (product_dtype or torch.float32) != self.product_dtype:
            raise ValueError("the products' dtype is not the one these buffers were built for")
        self.load_weights(model, w)
        self.load_inputs(w, h, q, age, z, ctx, scalars, reset, live)
        if self.forward_graph and self._forward_graph is None:
            self.capture_forward()  # the block's inputs are loaded: record the launches
        if self._forward_graph is not None:
            self._forward_graph.replay()
        else:
            self._slots()
        self.calls += 1
        return self.tape()

    def tape(self) -> dict[str, Tensor]:
        back, E = self.back, self.E
        return {
            "c": back.c,
            "b": back.b,
            "h_old": back.h_old,
            "y": back.y,
            "gate_pre": back.gate_pre,
            "m": self.m,
            "rms": back.rms,
            "hidden": back.hidden,
            "cells": back.cells,
            "units": back.units,
            "valid": back.valid,
            "v": self.v.permute(1, 0, 2, 3),
            "v_eb": self.v,
            "act_eb": self.act,
            "d_hid": self.d_hid,
            "u": self.u,
            "eps": self.eps,
            "z_hat": self.z_hat,
            "h_pre": self.h[:E].permute(1, 0, 2, 3),
            "h_pre_eb": self.h[:E],
            "age_pre": self.age[:, :E],
            "soft": self.soft,
            "prob": self.prob,
            "ties": self.ties,
            "theta": self.theta,
            "q_pre": self.q_pre,
            "q_post": self.q[:, 1:],
            "readout": self.readout,
            "logits": self.logits,
            "sel_cells": self.sel_cells,
            "sel_units": self.sel_units,
            "sel_valid": back.valid,
            "cost": self.cost,
            "refreshed": self.refreshed.mean(dim=0),
            "abs_max": self.abs_max.amax(dim=0),
            "h": self.h[E],
            "q": self.q[:, E],
            # (its own tensor: the state leaves the block, and the next block's buffers may
            # be these laid out for another number of slots)
            "age": self.age[:, E].clone(),
            "reset": self.reset,
        }

    # -- the reverse sweep ------------------------------------------------------------------------

    def sweep(
        self,
        tape: dict[str, Tensor],
        consts: CellConstants,
        n_cells: int,
        d_q_pre: Tensor | None,
        d_q_post: Tensor | None,
        d_h_sel: Tensor | None,
        initial: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """:func:`hsn.train.event_fused.sweep_reference` on the card
        (:class:`hsn.train.state_sweep.ReverseSweep` with its kernels). The record is already
        in the sweep's buffers (the forward sweep wrote it there); this loads the cotangents
        and the weights and runs it."""
        back = self.back
        if tape["c"] is not back.c:
            raise ValueError("this tape is not the one in these buffers")
        back.reset.copy_(self.reset)
        for buffer, value in (
            (back.d_q_pre, d_q_pre),
            (back.d_q_post, d_q_post),
            (back.d_h_sel, d_h_sel),
        ):
            if value is None:
                buffer.zero_()
            else:
                buffer.copy_(value)
        back.d_h_fast.zero_()
        torch.matmul(self.m, consts.v, out=back.mv)
        back.load_weights(consts)
        if initial is None:
            back.A.zero_()
            back.R.zero_()
        else:
            back.A.copy_(initial[0])
            back.R.copy_(initial[1])
        back.op1.zero_()
        back.op2.zero_()
        if self.graph and back._graph is None:
            back.capture()
        dc, db = back.run()
        return dc, db, back.A.clone(), back.R.clone()

    def pairs(
        self, dv: Tensor, row_of: Tensor, tape: dict[str, Tensor], gain: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        """The cell norm's backward of every slot of the block as one kernel
        (``hsn_ev_pairs``; the PyTorch form is in
        :func:`hsn.train.event_fused.local_backward`): ``(d_u, d_eps [B, E, d], dm [B, E, K,
        d], g_norm [n])`` from ``dv [P, n]`` (a slot's row is ``row_of [P]``), or ``None``
        for what the kernel does not take."""
        spec = self.spec
        B, E, K, d, n = self.B, self.E, spec.K, spec.d, spec.in_dim
        threads = -(-n // 32) * 32
        if K > 16 or threads > 1024 or dv.dtype != self.product_dtype or not dv.is_contiguous():
            return None
        if tape["h_old"] is not self.back.h_old:
            return None
        d_u, d_eps = torch.empty_like(self.u), torch.empty_like(self.eps)
        dm = torch.empty_like(self.m)
        g_part = torch.empty(B * E, n, dtype=torch.float32, device=self.device)
        window_kernels._extension().launch(
            "hsn_ev_pairs", B * E, threads, 4 * K * (1 + threads // 32),
            [dv, row_of.to(torch.int32).contiguous(), self.u, self.eps, self.back.h_old, self.m,
             self.back.rms, gain.contiguous(), d_u, d_eps, dm, g_part],
            [K, d, n, 1 if self.half else 0],
            [],
        )  # fmt: skip
        return d_u, d_eps, dm, g_part.sum(dim=0)

    def describe(self) -> dict[str, Any]:
        return {
            "B": self.B,
            "E": self.E,
            "product_dtype": str(self.product_dtype),
            "bias_in_product": self.bias_in_product,
            "shared_bytes": 4
            * max(finish_shared_words(self.spec), decide_shared_words(self.spec, self.S)),
            "calls": self.calls,
        }


class EventKernelSet:
    """The event kernels of a run: one :class:`EventKernels` per block shape (the number of
    event slots follows the batch), built at a shape's first block, and on its buffers the
    kernels of every smaller number of slots a block ran (:meth:`EventKernels.at`: the fused
    block's ``trim``). ``forward``, ``sweep`` and ``pairs`` are what
    :func:`hsn.train.event_fused.fused_event_block` takes. ``forward_graph``: also capture
    every shape's forward sweep in a CUDA graph at its first block."""

    def __init__(
        self, *, bias_in_product: bool = True, graph: bool = True, forward_graph: bool = True
    ) -> None:
        self.bias_in_product = bias_in_product
        self.graph = graph
        self.forward_graph = forward_graph
        self.kernels: dict[tuple[object, ...], EventKernels] = {}
        self.last: EventKernels | None = None

    def forward(
        self,
        model: HSN,
        w: EventWeights,
        spec: EventSpec,
        h: Tensor,
        q: Tensor,
        age: Tensor,
        z: Tensor,
        ctx: Tensor,
        scalars: Tensor,
        reset: Tensor,
        live: Tensor,
        *,
        product_dtype: torch.dtype | None = None,
        cell_consts: CellConstants | None = None,
        slots: int | None = None,
    ) -> dict[str, Tensor]:
        product = h.dtype if product_dtype is None else product_dtype
        key = (int(z.shape[0]), int(z.shape[1]), int(scalars.shape[-1]), product, h.device, spec)
        kern = self.kernels.get(key)
        if kern is None:
            if cell_consts is None:
                raise ValueError("the first block of a shape needs the cell's constants")
            kern = EventKernels(
                spec,
                w,
                cell_consts,
                B=int(z.shape[0]),
                E=int(z.shape[1]),
                n_scalars=int(scalars.shape[-1]),
                device=h.device,
                product_dtype=product,
                bias_in_product=self.bias_in_product,
                graph=self.graph,
                forward_graph=self.forward_graph,
            )
            self.kernels[key] = kern
        if slots is not None:
            kern = kern.at(slots)
        self.last = kern
        return kern.forward(
            model, w, spec, h, q, age, z, ctx, scalars, reset, live, product_dtype=product_dtype
        )

    def _of(self, tape: dict[str, Tensor]) -> EventKernels | None:
        """The kernels whose buffers hold ``tape``."""
        if self.last is not None and tape["c"] is self.last.back.c:
            return self.last
        for shape in self.kernels.values():
            for kern in shape.blocks():
                if tape["c"] is kern.back.c:
                    return kern
        return None

    def sweep(
        self,
        tape: dict[str, Tensor],
        consts: CellConstants,
        n_cells: int,
        d_q_pre: Tensor | None,
        d_q_post: Tensor | None,
        d_h_sel: Tensor | None,
        initial: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        kern = self._of(tape)
        if kern is None:
            raise ValueError("this tape is not in any of the set's buffers")
        return kern.sweep(tape, consts, n_cells, d_q_pre, d_q_post, d_h_sel, initial)

    def pairs(
        self, dv: Tensor, row_of: Tensor, tape: dict[str, Tensor], gain: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        kern = self._of(tape)
        return None if kern is None else kern.pairs(dv, row_of, tape, gain)
