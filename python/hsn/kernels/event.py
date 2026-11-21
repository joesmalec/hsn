"""The fused word-event block on CUDA (``docs/wev_fused_event.md``): the forward sweep's two
kernels (``kernels/cuda/hsn_event.cu``, in the window fatbin of
``scripts/build_window_kernels.py``) with the two matrix products of a slot between them, and
the reverse sweep of P2.10c on the same buffers.

:class:`EventKernels` holds the buffers of one block shape. Its :meth:`EventKernels.forward` is
:func:`hsn.train.event_fused.forward_reference` (the same arguments and tape) and its
:meth:`EventKernels.sweep` is :func:`hsn.train.event_fused.sweep_reference`; both are handed to
:func:`hsn.train.event_fused.fused_event_block`. fp32 state; the products run in fp32 or bf16.

The tape is the buffers themselves: a block's tape is valid until the next block's forward
sweep (one block per window and one backward pass per forward, as a training step runs them).
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

__all__ = ["HEAD_THREADS", "PICK_THREADS", "EventKernels", "head_shared_words", "unsupported"]

#: Threads of a row's block in ``hsn_ev_head`` (one per logit of the row) and ``hsn_ev_pick``.
HEAD_THREADS = 512
PICK_THREADS = 512


def head_shared_words(spec: EventSpec, front_hidden: int, threads: int = HEAD_THREADS) -> int:
    """The dynamic shared memory of ``hsn_ev_head`` in 4-byte words (its header's layout)."""
    N, M, K, d = spec.N, spec.M, spec.K, spec.d
    warps = (N + 31) // 32
    return (
        front_hidden + threads + 3 * d + M + M * spec.A + N * M + 5 * N + M + 4 * K + 1
        + 3 * warps + 2 * K + 3 * K * d + K * spec.D * d
    )  # fmt: skip


def unsupported(spec: EventSpec, front_hidden: int, threads: int = HEAD_THREADS) -> str | None:
    """Why the kernels do not take this shape, or ``None``."""
    N, M, d = spec.N, spec.M, spec.d
    if M & (M - 1) or M > 32:
        return f"n_units={M} must be a power of two <= 32"
    if N * M > threads or 2 * d > threads or N > threads:
        return f"n_cells*n_units={N * M} and 2d={2 * d} must fit {threads} threads"
    if 4 * head_shared_words(spec, front_hidden, threads) > window_kernels._MAX_SHARED_BYTES:
        return "the head kernel's shared memory exceeds 48 KiB at this shape"
    if spec.unit_act not in ACTIVATION_CODES or spec.front_act not in ACTIVATION_CODES:
        return "unknown activation"
    return None


class EventKernels:
    """The buffers and launches of the fused event block for one shape (module docstring).

    ``bias_in_product``: the two products add their bias (``addmm``; one rounding, as the
    compiled round's) instead of the kernels adding it to the columns they read (``mm``; under
    bf16 one more rounding). ``graph``: capture the reverse sweep's loop in a CUDA graph at
    its first block."""

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
    ) -> None:
        front_hidden = int(weights.front_q.shape[0])
        reason = unsupported(spec, front_hidden)
        if reason is not None:
            raise ValueError(f"the event kernels do not take this shape: {reason}")
        if not window_kernels.available():
            raise RuntimeError("window kernels missing; run scripts/build_window_kernels.py")
        if not window_kernels.prepared(device):
            window_kernels.prepare(device.index or 0)
        if product_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("the products run in fp32 or bf16")
        self.spec = spec
        self.B, self.E, self.S = B, E, n_scalars
        self.device = device
        self.product_dtype = product_dtype
        self.half = product_dtype == torch.bfloat16
        self.bias_in_product = bias_in_product
        self.graph = graph
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
        )  # fmt: skip

        def f32(*size: int) -> Tensor:
            return torch.zeros(*size, dtype=torch.float32, device=device)

        def prod(*size: int) -> Tensor:
            return torch.zeros(*size, dtype=product_dtype, device=device)

        def i64(*size: int) -> Tensor:
            return torch.zeros(*size, dtype=torch.int64, device=device)

        def u8(*size: int) -> Tensor:
            return torch.zeros(*size, dtype=torch.bool, device=device)

        # the state, slot by slot (record t: what slot t reads; record E: after the block)
        self.h, self.q, self.age = f32(B, E + 1, N, d), f32(B, E + 1, d), i64(B, E + 1, N)
        self.q_pre = f32(B, E, d)
        self.reset, self.live = u8(B, E), u8(B, E)
        self.z, self.static, self.scal = f32(B, E, d), prod(B * E, self.Hf), f32(B, E, N, n_scalars)
        # the tape beside the reverse sweep's record
        self.u, self.z_hat, self.eps = f32(B, E, d), f32(B, E, d), f32(B, E, d)
        self.logits = f32(B, E, N, M)
        self.sel_cells, self.sel_units = i64(B, E, K), i64(B, E, K)
        self.theta, self.soft, self.prob = f32(B, E), f32(B, E, K), f32(B, E, K)
        self.ties = u8(B, E, K, M)
        self.cost, self.refreshed = f32(B, E), f32(B, E)
        self.m = f32(B, E, K, d)
        self.v = prod(E, B, K, n)
        self.readout, self.abs_max = f32(B, E, d), f32(B, E)
        # the products' results and the second one's operand
        self.first, self.act, self.out = prod(B * K, M * H + d), prod(B * K, H), prod(B * K, M * d)
        # the weights, as the kernels and the products read them
        self.front_static = prod(self.Hf, int(weights.front_static.shape[1]))
        self.front_b1 = prod(self.Hf)
        self.front_q, self.front_w2, self.front_b2 = (
            f32(self.Hf, d),
            f32(2 * d, self.Hf),
            f32(2 * d),
        )
        self.score_w, self.score_b = f32(M, int(weights.score_w.shape[1])), f32(M)
        self.gate_w = f32(max(M * spec.A, 1), d)
        self.v_w, self.edge = f32(d, d), f32(N, D)
        self.pad_src = spec.pad_src.to(device=device, dtype=torch.int32).contiguous()
        self.norm_gain, self.cand_gain = f32(n), f32(d)
        self.cost_table = f32(M)
        self.w1, self.b1, self.b1_f = prod(n, M * H + d), prod(M * H + d), f32(M * H + d)
        self.w2, self.b2, self.b2_f = prod(H, M * d), prod(M * d), f32(M * d)
        self.calls = 0
        self._forward_graph: torch.cuda.CUDAGraph | None = None

    # -- the weights of a window ------------------------------------------------------------------

    def load_weights(self, model: HSN, w: EventWeights) -> None:
        """The window's weights into the buffers. ``w`` already holds the values the products
        read (:func:`hsn.train.event_fused.event_weights` with the products' dtype)."""
        self.front_static.copy_(w.front_static)
        self.front_b1.copy_(w.front_b1)
        self.front_q.copy_(w.front_q)
        self.front_w2.copy_(w.front_w2)
        self.front_b2.copy_(w.front_b2)
        self.score_w.copy_(w.score_w)
        self.score_b.copy_(w.score_b)
        if w.gate_w is not None:
            self.gate_w.copy_(w.gate_w)
        self.v_w.copy_(w.v)
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

    def _head(self, t: int) -> None:
        spec, back = self.spec, self.back
        E = self.E
        flags = (1 if t < E else 0) | (2 if t > 0 else 0) | (4 if self.half else 0)
        flags |= (8 if spec.probes else 0) | (16 if spec.gate_input == "summary" else 0)
        flags |= (32 if spec.gate_input != "none" else 0) | (64 if self.bias_in_product else 0)
        lam, overhead, shared_cost = self._costs
        tensors = [
            self.h, self.q, self.q_pre, self.age, self.reset, self.live, self.z, self.static,
            self.scal, self.front_q, self.front_w2, self.front_b2, self.score_w, self.score_b,
            self.gate_w, self.v_w, self.edge, self.pad_src, self.norm_gain, self.cand_gain,
            self.cost_table, self.b2_f, self.u, self.z_hat, self.eps, self.logits, back.cells,
            back.units, back.valid, self.sel_cells, self.sel_units, self.theta, self.soft,
            self.prob, self.ties, self.cost, self.refreshed, back.h_old, self.m, back.rms,
            self.v, self.out, back.b, back.y, back.c, self.readout, self.abs_max,
        ]  # fmt: skip
        ints = [
            self.B, E, spec.N, spec.M, spec.K, spec.d, spec.H, self.Hf, self.Hs, spec.in_dim,
            spec.D, spec.A, self.S, 0, t, flags, spec.age_max, ACTIVATION_CODES[spec.front_act],
        ]  # fmt: skip
        floats = [
            spec.tau, spec.norm_eps, spec.cand_eps, lam, overhead, shared_cost,
            spec.budget * (1.0 + BUDGET_SLACK),
        ]  # fmt: skip
        window_kernels._extension().launch(
            "hsn_ev_head", self.B, HEAD_THREADS, 4 * head_shared_words(spec, self.Hf),
            tensors, ints, floats,
        )  # fmt: skip

    def _pick(self, t: int) -> None:
        spec, back = self.spec, self.back
        flags = (1 if self.half else 0) | (2 if self.bias_in_product else 0)
        window_kernels._extension().launch(
            "hsn_ev_pick", self.B, PICK_THREADS, 0,
            [self.first, self.b1_f, back.units, back.hidden, self.act, back.gate_pre, back.b],
            [self.B, self.E, spec.K, spec.d, spec.H, spec.M, t, flags,
             ACTIVATION_CODES[spec.unit_act]],
            [],
        )  # fmt: skip

    def _slots(self) -> None:
        """The launches of a block's forward sweep: per slot the head, the first product, the
        pick, the second product; then the head once more to finish the last slot."""
        B, K, n = self.B, self.spec.K, self.spec.in_dim
        for t in range(self.E):
            self._head(t)
            operand = self.v[t].view(B * K, n)
            if self.bias_in_product:
                torch.addmm(self.b1, operand, self.w1, out=self.first)
            else:
                torch.mm(operand, self.w1, out=self.first)
            self._pick(t)
            if self.bias_in_product:
                torch.addmm(self.b2, self.act, self.w2, out=self.out)
            else:
                torch.mm(self.act, self.w2, out=self.out)
        self._head(self.E)

    def capture_forward(self, warm_up: int = 2) -> None:
        """Capture the forward sweep's launches in a CUDA graph (the buffers must hold a
        block's inputs and weights: call after one :meth:`forward`). Later sweeps replay it."""
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
        """A block's inputs into the buffers, and the front's static part: ``[z; ctx] · W1'ᵀ +
        b1'`` for all slots as one product."""
        B, E = self.B, self.E
        self.h[:, 0].copy_(h)
        self.q[:, 0].copy_(q)
        self.age[:, 0].copy_(age)
        self.z.copy_(z)
        self.scal.copy_(scalars)
        self.reset.copy_(reset)
        self.live.copy_(live)
        pd = self.product_dtype
        x = torch.cat([z.to(pd), ctx.to(pd)], dim=-1).reshape(B * E, -1)
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
    ) -> dict[str, Tensor]:
        """:func:`hsn.train.event_fused.forward_reference` on the card: the block's tape, as
        views of the buffers."""
        if tuple(z.shape[:2]) != (self.B, self.E) or int(scalars.shape[-1]) != self.S:
            raise ValueError("the block's shape is not the one these buffers were built for")
        if (product_dtype or torch.float32) != self.product_dtype:
            raise ValueError("the products' dtype is not the one these buffers were built for")
        self.load_weights(model, w)
        self.load_inputs(w, h, q, age, z, ctx, scalars, reset, live)
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
            "u": self.u,
            "eps": self.eps,
            "z_hat": self.z_hat,
            "h_pre": self.h[:, :E],
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
            "h": self.h[:, E],
            "q": self.q[:, E],
            "age": self.age[:, E],
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

    def describe(self) -> dict[str, Any]:
        return {
            "B": self.B,
            "E": self.E,
            "product_dtype": str(self.product_dtype),
            "bias_in_product": self.bias_in_product,
            "head_shared_bytes": 4 * head_shared_words(self.spec, self.Hf),
            "calls": self.calls,
        }
