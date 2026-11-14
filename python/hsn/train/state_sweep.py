"""P2.10c, phase 2: the reverse sweep of the state-gradient window on static buffers.

:func:`hsn.train.state_gradient.state_gradient_backward` is the reference: plain PyTorch, new
tensors at every position. This module is the same sweep in the form the card runs it
(``docs/p2_10c_state_gradient_window.md``, "Phase 2 log"): every value a position reads or
writes lives in a buffer that is allocated once per shape, and a position is three glue steps
with the cell's two transposed unit layers between them::

    head     the carry of P2.10 (R, the gather of A, dL/dc and dL/db of the position, the
             carried part of A), the candidate's backward, the gate's backward; writes the
             second layer's operand and the gate's columns of the first's
    product  W_out and the second layer, transposed (the routed slots in the all-units layout,
             the slots with a fixed unit as one batched product)
    middle   the activation's derivative and the cell norm's inner product; writes the first
             layer's operand
    product  the first layer's state rows with the gate, transposed, with the cell norm's gain
             and V folded in
    tail     the cell norm's backward, the scatter into the cell and into the sources of its
             messages, R, the reset
    limit    only with amendment A6's per-row limit (train.state_grad_clip / _ratio): the tail
             then leaves the reset to this step and hands it the rows' norms; a row over its
             bound is scaled, then the reset

No glue step multiplies a matrix. ``W_out`` is folded into the second layer's transposed
weights (per unit ``W_out · W2_jᵀ``), so the head hands the product the candidate's cotangent
itself. The cell norm's backward is linear in the first layer's product, so its gain and ``V``
are folded into that product's weights: it delivers ``[g_h ⊙ dv_h ; (g_m ⊙ dv_m) V]`` and the
tail needs only ``dh = r · prod_h − (r² / n) σ · h_old`` and ``Vᵀ dm = r · prod_m − (r² / n) σ ·
(m V)``, with ``m V`` taken once per window. (The first version did both small products inside
the kernels and took 100 µs per position for the head and for the tail at 20 slots.)

The operands of the all-units products are zero outside one unit block per slot. A glue step
writes the slot's block and clears the block the position before it (``t + 1``) used, so the
operand buffers are zeroed once per window, not rewritten at every position.

A source's cotangent is gathered, not scattered: every cell sums, in a fixed order, the
contributions of the selected cells it is a source of (:func:`out_adjacency`). The result does
not depend on the order in which threads run, on the card as here.

Each glue step has two implementations with one contract: the PyTorch one of this module (any
device and dtype; the reference a kernel is tested against, step by step) and, on CUDA with the
window kernels built, a kernel (``hsn_win_back_head`` / ``_mid`` / ``_tail``,
:mod:`hsn.kernels.window`). On CUDA the loop over the positions can be captured once per shape
in a CUDA graph (:meth:`ReverseSweep.capture`) and replayed for every window.

The limit (:class:`hsn.train.state_gradient.RowLimits`) is a property of the run, fixed when
the sweep is built. Its step has the PyTorch implementation only: a sweep with a limit does
not take the kernels yet, and :class:`SweepRunner` leaves such a window to the PyTorch sweep.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from hsn.model.gradclip import limit_row_grad, row_norm
from hsn.train.state_gradient import CellConstants, RowLimits, activation_grad, sweep_layouts

__all__ = ["ReverseSweep", "SweepRunner", "SweepShape", "out_adjacency", "sweep_shape"]

#: ``model.activation`` as the kernels' code.
ACTIVATION_CODES: dict[str, int] = {"gelu": 0, "relu": 1, "silu": 2, "tanh": 3}


class SweepShape(NamedTuple):
    """The sizes a :class:`ReverseSweep` is built for. ``n_fast`` leading slots have a fixed
    unit and their own operands (0: every slot takes the all-units layout); ``n_read`` cells
    ``0 .. n_read − 1`` are read after every position (amendment A8's fast read of the decoder;
    0: none); ``d_out`` is the largest number of cells one cell is a message source of."""

    B: int
    T: int
    K: int
    N: int
    d: int
    H: int
    M: int
    n_fast: int
    n_read: int
    in_dim: int
    d_out: int


def out_adjacency(pad_src: Tensor, pad_mask: Tensor) -> tuple[Tensor, Tensor]:
    """The message graph by source: ``(out_tgt [N, D_out], out_edge [N, D_out])`` (int64) from
    the padded sources ``pad_src [N, D]`` of every cell and the mask of its real slots. Row
    ``n`` lists the edges out of cell ``n`` in the order of their target cell, then of the
    target's slot: ``out_tgt`` is the target (−1 on padding) and ``out_edge`` the edge's index
    ``target · D + slot`` in the flattened padded view, where its weight is (0 on padding)."""
    N, D = (int(n) for n in pad_src.shape)
    target, slot = pad_mask.nonzero(as_tuple=True)  # row-major: by target, then by slot
    source = pad_src[target, slot]
    order = torch.argsort(source, stable=True)
    source, target, flat = source[order], target[order], (target * D + slot)[order]
    counts = torch.bincount(source, minlength=N)
    d_out = max(int(counts.max()) if counts.numel() else 0, 1)
    position = (
        torch.arange(source.numel(), device=pad_src.device) - (counts.cumsum(0) - counts)[source]
    )
    out_tgt = torch.full((N, d_out), -1, dtype=torch.int64, device=pad_src.device)
    out_edge = torch.zeros((N, d_out), dtype=torch.int64, device=pad_src.device)
    out_tgt[source, position] = target
    out_edge[source, position] = flat
    return out_tgt, out_edge


class ReverseSweep:
    """The reverse sweep of one window shape on static buffers (module docstring).

    ``load`` copies a window's record, cotangents and weights into the buffers; ``run`` walks
    the positions ``T − 1 .. 0`` and returns ``(dL/dc, dL/db)`` (the buffers ``dc`` and ``db``:
    the next ``load`` overwrites them). ``kernels`` chooses the glue steps' implementation
    (``None``: the kernels when the buffers are on a prepared CUDA device and the shape fits
    them, else PyTorch). ``state_dtype`` is the state's dtype, ``product_dtype`` the dtype of
    the two products and of their operands (bf16 under autocast, else the state's).
    ``limits``: amendment A6's per-row limit on what a position passes back (``None``: none)."""

    def __init__(
        self,
        shape: SweepShape,
        consts: CellConstants,
        *,
        device: torch.device,
        state_dtype: torch.dtype,
        product_dtype: torch.dtype,
        kernels: bool | None = None,
        limits: RowLimits | None = None,
    ) -> None:
        self.shape = shape
        self.limits = limits
        self.device = device
        self.state_dtype = state_dtype
        self.product_dtype = product_dtype
        self.activation = consts.activation
        self.cand_norm = consts.cand_gain is not None
        self.cand_eps = float(consts.cand_eps)
        self.cell_norm = consts.norm_gain_state is not None
        B, T, K, N, d, H, M, F, n_read, _, d_out = shape
        if not 0 <= F < K:
            raise ValueError(f"n_fast={F} must leave a routed slot of the {K} slots")
        self.routed = K - F
        rows = B * self.routed

        def state(*size: int) -> Tensor:
            return torch.zeros(*size, dtype=state_dtype, device=device)

        def product(*size: int) -> Tensor:
            return torch.zeros(*size, dtype=product_dtype, device=device)

        def index(*size: int) -> Tensor:
            return torch.zeros(*size, dtype=torch.int64, device=device)

        # the state's cotangent, and its rows' norms for the limit (of A's rows, of R's rows)
        self.A, self.R = state(B, N, d), state(B, d)
        self.norms = state(2, B)
        # the window's record (the replay's values, the tape's decisions)
        self.c, self.b, self.h_old = state(B, T, K, d), state(B, T, K, d), state(B, T, K, d)
        self.y, self.gate_pre = state(B, T, K, d), state(B, T, K, d)
        self.mv = state(B, T, K, d)  # m V: the messages through V, once per window
        self.rms = state(B, T, K)
        self.hidden = product(B, T, K, H)
        self.cells, self.units = index(B, T, K), index(B, T, K)
        self.valid = torch.zeros(B, T, K, dtype=torch.bool, device=device)
        self.reset = torch.zeros(B, T, dtype=torch.bool, device=device)
        # the cotangents of the node's outputs (zeros for a read that carried none)
        self.d_q_pre, self.d_q_post = state(B, T, d), state(B, T, d)
        self.d_h_sel = state(B, T, K, d)
        self.d_h_fast = state(B, T, max(n_read, 1), d)
        # the cell's weights: W_out folded into the second layer's transposed weights, the
        # cell norm's gain and V into the first layer's (module docstring)
        self.cand_gain, self.gate_bias = state(d), state(d)
        self.b1 = state(M, H)
        self.w2_t, self.w1_state_t = product(M * d, H), product(M * H + d, 2 * d)
        self.fast_w2_t = product(max(F, 1), d, H)
        self.fast_w1_state_t = product(max(F, 1), H + d, 2 * d)
        self.out_tgt = torch.full((N, d_out), -1, dtype=torch.int32, device=device)
        self.out_edge = index(N, d_out)
        self.out_w = state(N, d_out)
        self._graph_of: Tensor | None = None  # the pad_src the out-adjacency was built for
        # the products' operands and results
        self.op1, self.op2 = product(rows, M * d), product(rows, M * H + d)
        self.op1_fast, self.op2_fast = product(max(F, 1), B, d), product(max(F, 1), B, H + d)
        self.act_r, self.dv_r = product(rows, H), product(rows, 2 * d)
        # torch.mm and torch.addmm reach different cuBLAS paths for one shape. Measured on an
        # RTX 3060 (bf16, [rows, 512] x [512, 512]): 384 rows 20.0 against 11.7 us, 128 rows
        # 6.5 against 7.3, 640 rows 20.1 against 24.5. A fixed rule, so that a run's arithmetic
        # does not depend on a timing: addmm (a zero bias) between 256 and 512 rows.
        self._second_addmm = 256 < rows < 512
        self._no_bias = product(H)
        self.act_f, self.dv_f = product(max(F, 1), B, H), product(max(F, 1), B, 2 * d)
        self.sigma = state(B, K)
        # what the sweep returns
        self.dc, self.db = state(B, T, K, d), state(B, T, K, d)
        self._graph: torch.cuda.CUDAGraph | None = None
        self.kernels = self._kernels_fit() if kernels is None else kernels
        if self.kernels and not self._kernels_fit():
            raise ValueError("the reverse sweep's kernels do not take this shape or device")

    # -- the kernels ------------------------------------------------------------------------------

    def _kernels_fit(self) -> bool:
        from hsn.kernels import window as window_kernels

        if self.device.type != "cuda" or self.state_dtype != torch.float32:
            return False
        if self.limits is not None:  # the limit's step is not a kernel yet
            return False
        if self.product_dtype not in (torch.float32, torch.bfloat16):
            return False
        if not window_kernels.available() or not window_kernels.prepared(self.device):
            return False
        return window_kernels.back_unsupported(self.shape) is None

    # -- loading a window -------------------------------------------------------------------------

    def load_weights(self, consts: CellConstants) -> None:
        """The cell's weights of this window (they change at every optimiser step)."""
        s = self.shape
        d, M = s.d, s.M
        layouts = sweep_layouts(consts, s.n_fast)
        if consts.cand_gain is not None:
            self.cand_gain.copy_(consts.cand_gain)
        self.gate_bias.copy_(consts.gate_bias)
        self.b1.copy_(consts.b1)
        # dF = dy W_out, then the unit's block of the second layer: one product with W_out W2_jᵀ
        second = torch.matmul(consts.w_out, layouts.w2_t.reshape(M, d, s.H))
        self.w2_t.copy_(second.reshape(M * d, s.H))
        # prod = dv · blockdiag(diag(g_h), diag(g_m) V): the norm's gain and V after the product
        fold = consts.w_out.new_zeros(2 * d, 2 * d)
        gain = consts.norm_gain_state
        fold[:d, :d] = torch.eye(d, dtype=fold.dtype, device=fold.device)
        fold[d:, d:] = consts.v
        if gain is not None:
            fold[:d, :d] = torch.diag(gain[:d])
            fold[d:, d:] = gain[d:, None] * consts.v
        self.w1_state_t.copy_(layouts.w1_state_t @ fold)
        if s.n_fast:
            assert layouts.fast_w2_t is not None and layouts.fast_w1_state_t is not None
            self.fast_w2_t.copy_(torch.matmul(consts.w_out, layouts.fast_w2_t))
            self.fast_w1_state_t.copy_(layouts.fast_w1_state_t @ fold)
        if self._graph_of is not consts.pad_src:  # the topology: once (it is fixed between windows)
            mask = consts.pad_mask
            if mask is None:
                raise ValueError("the reverse sweep needs the graph's padding mask (pad_mask)")
            out_tgt, out_edge = out_adjacency(consts.pad_src, mask)
            if tuple(out_tgt.shape) != tuple(self.out_tgt.shape):
                raise ValueError(
                    f"the message graph's out-degree {int(out_tgt.shape[1])} is not the "
                    f"shape's d_out = {s.d_out}"
                )
            self.out_tgt.copy_(out_tgt)
            self.out_edge.copy_(out_edge)
            self._graph_of = consts.pad_src
        weights = consts.edge_weights.reshape(-1)[self.out_edge]
        self.out_w.copy_(torch.where(self.out_tgt >= 0, weights, torch.zeros_like(weights)))

    def load(
        self,
        c: Tensor,
        b: Tensor,
        h_old: Tensor,
        cells: Tensor,
        units: Tensor,
        valid: Tensor,
        reset: Tensor,
        d_q_pre: Tensor | None,
        d_q_post: Tensor | None,
        d_h_sel: Tensor | None,
        d_h_fast: Tensor | None,
        record: dict[str, Tensor],
        consts: CellConstants,
        initial: tuple[Tensor, Tensor] | None = None,
    ) -> None:
        """A window into the buffers: :func:`state_gradient_backward`'s arguments (``c``, ``b``,
        ``h_old`` ``[B, T, K, d]``; ``cells``, ``units``, ``valid`` ``[B, T, K]``; ``reset
        [B, T]``; the cotangents, ``None`` for a read that carried none; the replay's record;
        the cell's weights). ``initial``: the cotangent of the state after the last position
        (``None``: zero)."""
        self.c.copy_(c)
        self.b.copy_(b)
        self.h_old.copy_(h_old)
        self.cells.copy_(torch.where(valid, cells, torch.zeros_like(cells)))
        self.units.copy_(torch.where(valid, units, torch.zeros_like(units)))
        self.valid.copy_(valid)
        self.reset.copy_(reset)
        for buffer, value in (
            (self.d_q_pre, d_q_pre),
            (self.d_q_post, d_q_post),
            (self.d_h_sel, d_h_sel),
        ):
            if value is None:
                buffer.zero_()
            else:
                buffer.copy_(value)
        if self.shape.n_read and d_h_fast is not None:
            self.d_h_fast.copy_(d_h_fast)
        else:
            self.d_h_fast.zero_()
        self.y.copy_(record["y"])
        self.gate_pre.copy_(record["gate_pre"])
        torch.matmul(record["m"], consts.v, out=self.mv)
        self.hidden.copy_(record["hidden"])
        if self.cell_norm:
            self.rms.copy_(record["rms"])
        self.load_weights(consts)
        if initial is None:
            self.A.zero_()
            self.R.zero_()
        else:
            self.A.copy_(initial[0])
            self.R.copy_(initial[1])
        self.op1.zero_()
        self.op2.zero_()

    # -- the sweep --------------------------------------------------------------------------------

    def run(self) -> tuple[Tensor, Tensor]:
        """The loaded window's sweep: ``(dc, db)``, the buffers. Replays the captured graph
        when there is one."""
        if self._graph is not None:
            self._graph.replay()
        else:
            for t in range(self.shape.T - 1, -1, -1):
                self.step(t)
        return self.dc, self.db

    def capture(self, warm_up: int = 2) -> None:
        """Capture the loop over the positions in a CUDA graph (CUDA only). The buffers must
        hold a window (:meth:`load`): its sweep runs ``warm_up`` times eagerly, then the loop
        is recorded (a capture records the work, it does not run it), and the buffers are put
        back as loaded. Later calls of :meth:`run` replay the graph."""
        if self.device.type != "cuda":
            raise RuntimeError("a CUDA graph needs the buffers on a CUDA device")
        keep = (self.A.clone(), self.R.clone())
        for _ in range(warm_up):
            self._restart(keep)
            for t in range(self.shape.T - 1, -1, -1):
                self.step(t)
        self._restart(keep)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for t in range(self.shape.T - 1, -1, -1):
                self.step(t)
        self._graph = graph
        self._restart(keep)

    def _restart(self, initial: tuple[Tensor, Tensor]) -> None:
        self.A.copy_(initial[0])
        self.R.copy_(initial[1])
        self.op1.zero_()
        self.op2.zero_()

    def step(self, t: int) -> None:
        """Position ``t``: the three glue steps and the two transposed products between them,
        and with a limit its step."""
        self.head(t)
        self.products_second()
        self.middle(t)
        self.products_first()
        self.tail(t)
        if self.limits is not None:
            self.limit(t)

    def products_second(self) -> None:
        """``W_out`` and the second layer, transposed: ``act = op1 · (W_out W2ᵀ)``."""
        if self._second_addmm:
            torch.addmm(self._no_bias, self.op1, self.w2_t, out=self.act_r)
        else:
            torch.mm(self.op1, self.w2_t, out=self.act_r)
        if self.shape.n_fast:
            torch.bmm(self.op1_fast, self.fast_w2_t, out=self.act_f)

    def products_first(self) -> None:
        """The first layer's state rows with the gate, transposed, with the cell norm's gain
        and ``V`` folded in: ``prod = op2 · [W1_s; G_s]ᵀ · blockdiag(g_h, g_m V)``."""
        torch.mm(self.op2, self.w1_state_t, out=self.dv_r)
        if self.shape.n_fast:
            torch.bmm(self.op2_fast, self.fast_w1_state_t, out=self.dv_f)

    def head(self, t: int) -> None:
        if self.kernels:
            from hsn.kernels import window as window_kernels

            window_kernels.back_head(self, t)
        else:
            self.head_reference(t)

    def middle(self, t: int) -> None:
        if self.kernels:
            from hsn.kernels import window as window_kernels

            window_kernels.back_mid(self, t)
        else:
            self.middle_reference(t)

    def tail(self, t: int) -> None:
        if self.kernels:
            from hsn.kernels import window as window_kernels

            window_kernels.back_tail(self, t)
        else:
            self.tail_reference(t)

    def limit(self, t: int) -> None:
        self.limit_reference(t)

    # -- the glue steps in PyTorch (the kernels' reference) ---------------------------------------

    def _slots(self, t: int) -> tuple[Tensor, Tensor, Tensor]:
        """``(ok [B, K, 1], index [B, K, d], units [B, K])`` of position ``t``: the valid
        slots, their cells as a scatter index (the dummy row ``N`` on padded slots) and their
        units."""
        s = self.shape
        valid = self.valid[:, t]
        index = torch.where(valid, self.cells[:, t], torch.full_like(self.cells[:, t], s.N))
        return valid[:, :, None], index[:, :, None].expand(s.B, s.K, s.d), self.units[:, t]

    def _write_blocks(
        self, operand: Tensor, fast: Tensor, values: Tensor, t: int, width: int
    ) -> None:
        """``values [B, K, width]`` into the operands of a product: a routed slot's values into
        its unit's block of its row of ``operand`` (after the block position ``t + 1`` used is
        cleared), a fixed slot's into ``fast [F, B, ·]``."""
        s = self.shape
        F = s.n_fast
        columns = torch.arange(width, device=self.device)[None, :]
        if t + 1 < s.T:
            before = self.units[:, t + 1, F:].reshape(-1, 1) * width + columns
            operand.scatter_(1, before, torch.zeros_like(before, dtype=operand.dtype))
        now = self.units[:, t, F:].reshape(-1, 1) * width + columns
        operand.scatter_(1, now, values[:, F:].reshape(-1, width).to(operand.dtype))
        if F:
            fast[:, :, :width] = values[:, :F].transpose(0, 1).to(fast.dtype)

    def head_reference(self, t: int) -> None:
        s = self.shape
        N, d, F = s.N, s.d, s.n_fast
        ok, index, _ = self._slots(t)
        zero = self.A.new_zeros(())
        R = self.R + self.d_q_post[:, t]
        self.R.copy_(R)
        A = torch.cat([self.A, self.A.new_zeros(s.B, 1, d)], dim=1)  # the padded slots' dummy row
        if s.n_read:
            A = torch.cat([A[:, : s.n_read] + self.d_h_fast[:, t], A[:, s.n_read :]], dim=1)
        A = A.scatter_add(1, index, torch.where(ok, self.d_h_sel[:, t], zero))
        carried = A.gather(1, index)
        share = (R / N)[:, None, :]
        g = carried + share
        b, c = self.b[:, t], self.c[:, t]
        dc = torch.where(ok, b * g, zero)
        db = torch.where(ok, (c - self.h_old[:, t]) * g, zero)
        self.dc[:, t] = dc
        self.db[:, t] = db
        A = A.scatter(1, index, torch.where(ok, (1 - b) * carried - b * share, zero))
        self.A.copy_(A[:, :N])
        # the candidate, down to the unit's output (alpha is 1 on the valid slots)
        dz = dc * (1.0 - c * c)
        if self.cand_norm:
            y = self.y[:, t]
            scaled = self.cand_gain * dz
            rho = torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + self.cand_eps)
            dz = rho * scaled - rho.pow(3) * y * (y * scaled).mean(dim=-1, keepdim=True)
        d_unit = torch.where(ok, dz, zero)  # W_out is in the product's weights
        # the gate, and its part of the cell norm's inner product
        d_gate = db * b * (1.0 - b)
        self.sigma.copy_((d_gate * (self.gate_pre[:, t] - self.gate_bias)).sum(dim=-1))
        self._write_blocks(self.op1, self.op1_fast, d_unit, t, d)
        gate_columns = s.M * s.H
        self.op2[:, gate_columns:] = d_gate[:, F:].reshape(-1, d).to(self.op2.dtype)
        if F:
            self.op2_fast[:, :, s.H :] = d_gate[:, :F].transpose(0, 1).to(self.op2_fast.dtype)

    def _per_slot(self, routed: Tensor, fast: Tensor) -> Tensor:
        """A product's result per slot, ``[B, K, ·]`` in the state's dtype, from its two parts."""
        s = self.shape
        out = routed.reshape(s.B, self.routed, -1)
        if s.n_fast:
            out = torch.cat([fast.transpose(0, 1), out], dim=1)
        return out.to(self.state_dtype)

    def middle_reference(self, t: int) -> None:
        s = self.shape
        ok, _, units = self._slots(t)
        zero = self.A.new_zeros(())
        hidden = self.hidden[:, t].to(self.state_dtype)
        d_act = self._per_slot(self.act_r, self.act_f)
        d_hid = torch.where(ok, d_act * activation_grad(self.activation, hidden), zero)
        self.sigma.add_((d_hid * (hidden - self.b1[units])).sum(dim=-1))
        self._write_blocks(self.op2, self.op2_fast, d_hid, t, s.H)

    def tail_reference(self, t: int) -> None:
        s = self.shape
        B, K, N, d = s.B, s.K, s.N, s.d
        ok, index, _ = self._slots(t)
        zero = self.A.new_zeros(())
        prod = self._per_slot(self.dv_r, self.dv_f)  # [g_h dv_h ; (g_m dv_m) V]
        if self.cell_norm:
            r = self.rms[:, t][..., None]
            back = (r * r / s.in_dim) * self.sigma[..., None]
            ds = r * prod - back * torch.cat([self.h_old[:, t], self.mv[:, t]], dim=-1)
        else:
            ds = prod
        ds = torch.where(ok, ds, zero)
        dvm = ds[..., d:]  # [B, K, d]: what a source of the slot's messages receives
        A = torch.cat([self.A, self.A.new_zeros(B, 1, d)], dim=1)
        A = A.scatter_add(1, index, ds[..., :d])[:, :N]
        # every cell gathers from the selected cells it is a source of, in out_tgt's order
        slot_of = torch.full((B, N + 1), -1, dtype=torch.int64, device=self.device)
        slot_of.scatter_(1, index[:, :, 0], torch.arange(K, device=self.device).expand(B, K))
        for j in range(s.d_out):
            target = self.out_tgt[:, j].to(torch.int64)  # [N], -1 on padding
            slot = torch.where(target >= 0, slot_of[:, target.clamp(min=0)], -1)  # [B, N]
            taken = dvm.gather(1, slot.clamp(min=0)[:, :, None].expand(B, N, d))
            A = A + torch.where(
                (slot >= 0)[:, :, None], self.out_w[:, j][None, :, None] * taken, zero
            )
        R = self.R + self.d_q_pre[:, t]
        if self.limits is not None:  # the limit's step takes the rows' norms and the reset
            self.A.copy_(A)
            self.R.copy_(R)
            self.norms[0] = row_norm(A)[:, 0]
            self.norms[1] = row_norm(R)[:, 0]
            return
        row = self.reset[:, t]
        self.A.copy_(A.masked_fill(row[:, None, None], 0))
        self.R.copy_(R.masked_fill(row[:, None], 0))

    def limit_reference(self, t: int) -> None:
        """Amendment A6's limit on what position ``t`` passes back
        (:func:`hsn.model.gradclip.limit_row_grad` with the norms the tail took), then the
        reset: the rows that reset are limited and counted like the others, and zeroed."""
        limits = self.limits
        assert limits is not None
        A = limit_row_grad(self.A, limits.h, limits.ratio, norm=self.norms[0][:, None])
        R = limit_row_grad(self.R, limits.q, limits.ratio, norm=self.norms[1][:, None])
        row = self.reset[:, t]
        self.A.copy_(A.masked_fill(row[:, None, None], 0))
        self.R.copy_(R.masked_fill(row[:, None], 0))


def sweep_shape(
    c: Tensor,
    record: dict[str, Tensor],
    consts: CellConstants,
    n_cells: int,
    n_fast: int,
    n_read: int,
) -> SweepShape:
    """The :class:`SweepShape` of a window's sweep arguments: ``c [B, T, K, d]``, the replay's
    record, the cell's weights; ``n_fast`` leading slots with a fixed unit, ``n_read`` fast
    cells the decoder reads."""
    B, T, K, d = (int(n) for n in c.shape)
    if consts.pad_mask is None:
        raise ValueError("the reverse sweep needs the graph's padding mask (pad_mask)")
    d_out = int(out_adjacency(consts.pad_src, consts.pad_mask)[0].shape[1])
    return SweepShape(
        B=B,
        T=T,
        K=K,
        N=int(n_cells),
        d=d,
        H=int(record["hidden"].shape[-1]),
        M=int(consts.w2.shape[0]),
        n_fast=int(n_fast),
        n_read=int(n_read),
        in_dim=int(consts.in_dim),
        d_out=d_out,
    )


class SweepRunner:
    """The reverse sweep of a training run on the card: one :class:`ReverseSweep` per window
    shape, loaded with every window's values and run through its kernels, by default as a
    CUDA graph captured at the shape's first window. Called with
    :func:`hsn.train.state_gradient.state_gradient_backward`'s arguments (installed with
    :func:`hsn.train.state_gradient.set_fast_sweep`); returns ``None`` for what it does not
    take (another device, a dtype or shape the kernels do not handle), and the caller keeps
    the PyTorch sweep. ``limits`` (amendment A6's per-row limit) belongs to the run: a sweep
    and its graph are built for one."""

    def __init__(self, *, graph: bool = True) -> None:
        self.graph = graph
        self.sweeps: dict[tuple[object, ...], ReverseSweep] = {}
        self._d_out: dict[int, int] = {}
        #: Windows swept, and how many of them captured a graph (the measurements).
        self.calls = 0
        self.captures = 0

    def __call__(
        self,
        c: Tensor,
        b: Tensor,
        h_old: Tensor,
        cells: Tensor,
        units: Tensor,
        valid: Tensor,
        reset: Tensor,
        n_cells: int,
        d_q_pre: Tensor | None,
        d_q_post: Tensor | None,
        d_h_sel: Tensor | None,
        d_h_fast: Tensor | None,
        record: dict[str, Tensor],
        consts: CellConstants,
        *,
        n_fast: int = 0,
        n_read: int = 0,
        limits: RowLimits | None = None,
    ) -> tuple[Tensor, Tensor] | None:
        from hsn.kernels import window as window_kernels

        device = c.device
        if device.type != "cuda" or c.dtype != torch.float32:
            return None
        if limits is not None:  # the limit's step is not a kernel yet: the PyTorch sweep
            return None
        if not window_kernels.available() or not window_kernels.prepared(device):
            return None
        shape = sweep_shape(c, record, consts, n_cells, n_fast, n_read)
        if window_kernels.back_unsupported(shape) is not None:
            return None
        product_dtype = record["hidden"].dtype
        if product_dtype not in (torch.float32, torch.bfloat16):
            return None
        key: tuple[object, ...] = (
            shape,
            product_dtype,
            device,
            consts.activation,
            consts.cand_gain is not None,
            consts.norm_gain_state is not None,
            limits,
        )
        sweep = self.sweeps.get(key)
        if sweep is None:
            sweep = ReverseSweep(
                shape,
                consts,
                device=device,
                state_dtype=c.dtype,
                product_dtype=product_dtype,
                kernels=True,
                limits=limits,
            )
            self.sweeps[key] = sweep
        sweep.load(
            c, b, h_old, cells, units, valid, reset, d_q_pre, d_q_post, d_h_sel, d_h_fast,
            record, consts,
        )  # fmt: skip
        if self.graph and sweep._graph is None:
            sweep.capture()
            self.captures += 1
        dc, db = sweep.run()
        self.calls += 1
        # the buffers are the next window's: autograd gets its own tensors
        return dc.clone(), db.clone()
