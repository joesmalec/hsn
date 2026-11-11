"""P2.10: the split window — a training window as a forward sweep without autograd, one batched
replay with autograd and a linear backward recurrence over the state (``hardware.window=split``).

Why. A compiled training window issues about 124 small GPU kernels per position, forward and
backward (about 31,700 per step at B16/T256): the step is bound by the number of kernels, not by
arithmetic. Amendment A6's feedback cut (``train.feedback_detach`` with all four paths) makes
everything heavy *local to its position* in the backward pass: the units, the gate and the
messages read ``h.detach()``, ``S`` reads ``q.detach()``, the cell and the scorer read
``z − ẑ.detach()`` and the scorer's features are detached. The only gradient that crosses
positions is the gated carry ``(1 − b) ⊙ h``, the linear summary update, and the reads of ``q``
and of the selected cells by the predictor, the decoder and the selected readout. So:

1. **Forward sweep** (:func:`hsn.train.compile.run_tape`): the positions in order, no autograd,
   recording a tape — the pre-round state of every position (after its reset), the decisions, the
   summary after every position and the updated states of the selected slots.
2. **Replay** (:meth:`SplitWindow._replay`): the per-position local computations for all ``B·T``
   rows at once, with autograd, from the taped state and the *recorded* decisions (nothing is
   decided again): ``S``, the scorer and :meth:`~hsn.model.router.Router.finish_recorded`,
   :meth:`~hsn.model.cell.PDGCell.replay_sparse` (``ĉ``, ``b̂``), then ``P``, the selected readout
   and the window epilogue (decoder, ``info``) of :class:`~hsn.model.hsn.HSN`. The 256 small
   matmuls of every layer become one large one, and the unit bank runs each pair's unit only.
3. **State recurrence** (:class:`StateRecurrence`, one autograd node between the two halves of
   the replay): forward returns the taped ``q`` (before and after every position) and the taped
   selected states; backward turns their cotangents into ``dL/dĉ`` and ``dL/db̂`` of every
   position with the linear sweep below. The trainer's ``backward()`` is unchanged.

With ``A [B, N, d]`` and ``R [B, d]`` the cotangents of the state after position ``t``, and
``D_t``, ``P_t``, ``S_t`` those of ``q`` after ``t`` (the decoder's read), ``q`` before ``t``
(the predictor's read) and the selected updated states (the readout's read)::

    A = 0; R = 0
    for t = T−1 .. 0:
        R += D_t;  A[cell_k] += S_{t,k}                     (valid slots)
        g_k = A[cell_k] + R / N
        dĉ_k = b_k ⊙ g_k;  db̂_k = (c_k − h_old_k) ⊙ g_k     (exact zeros on padded slots)
        A[cell_k] = (1 − b_k) ⊙ A[cell_k] − b_k ⊙ R / N
        R += P_t
        A, R = 0 on the rows that reset at t

which is the vector-Jacobian product of :func:`hsn.kernels.native.gated_delta_update` and
:meth:`~hsn.model.state.CellGraphState.reset` over the window.

The gradients are the model's own (``tests/test_p2_10_split_window.py``: equal to
:meth:`HSN.forward`'s to 1e-9 in float64). Under bf16 autocast they agree with the compiled
window's to bf16 rounding, not bit for bit: a batched matmul rounds differently from 256 small
ones, and its weight gradient is rounded to bf16 once per window (as the encoder's and the
decoder's already are) where the per-position gradients were summed in fp32.

Dense-fallback rows (``train.p_dense``). A row that runs the dense round for a window is replayed
through :meth:`~hsn.model.cell.PDGCell.step_dense` (every cell, the softmax mix of the units) and
has its own recurrence (:func:`dense_recurrence_backward`): every cell is written,
``q' = mean(h')`` does not read ``q``, so with ``g = A + R / N`` the sweep is ``dĉ = b ⊙ g``,
``db̂ = (c − h) ⊙ g``, ``A = (1 − b) ⊙ g`` and ``R`` restarts at ``P_t``. Its readout is ``q``.

Scope (:func:`scope_error`; anything else keeps the autograd window): routed ``topk`` with the
``ste`` gate at every step, all four feedback paths cut, no exploration, no counterfactual
probes, no state-gradient clip, no disabled cells, one round, no loss scaler.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd.function import FunctionCtx, once_differentiable

from hsn.config import RunConfig, split_window_scope_error
from hsn.kernels import window as window_kernels
from hsn.model.hsn import HSN, _Prologue
from hsn.model.router import Scorer, Selection
from hsn.model.state import CellGraphState
from hsn.train.compile import CompiledWindow, WindowOut, _tape_block, run_tape

__all__ = [
    "SplitWindow",
    "StateRecurrence",
    "dense_recurrence_backward",
    "recurrence_backward",
    "scope_error",
]

#: The most memory a window's dense-fallback replay may take (:meth:`SplitWindow.__call__`).
DENSE_BYTES = 4 << 30


def scope_error(cfg: RunConfig) -> str | None:
    """Why ``cfg`` cannot train with the split window, or ``None`` when it can (module
    docstring, "Scope"): :func:`hsn.config.split_window_scope_error`, which the configuration
    itself applies when ``hardware.window`` asks for the split window."""
    return split_window_scope_error(cfg)


# --------------------------------------------------------------------------------------------------
# The state recurrence
# --------------------------------------------------------------------------------------------------


def recurrence_backward(
    c: Tensor,
    b: Tensor,
    h_old: Tensor,
    cells: Tensor,
    valid: Tensor,
    reset: Tensor,
    n_cells: int,
    d_q_pre: Tensor | None,
    d_q_post: Tensor | None,
    d_h_sel: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """The reference sweep of the module docstring in plain torch (any device and dtype):
    ``(dL/dc, dL/db)``, each ``[B, T, K, d]``. ``c``, ``b``, ``h_old`` are ``[B, T, K, d]``,
    ``cells`` (int64, cell 0 on padded slots) and ``valid`` ``[B, T, K]``, ``reset`` bool
    ``[B, T]``; the cotangents are ``[B, T, d]`` (``q`` before / after each position) and
    ``[B, T, K, d]`` (the selected updated states), ``None`` for a read that carried none.
    Padded slots get exact zeros and never touch the cell cotangent (``where``, never a product
    with a mask: a non-finite value in a padded slot stays out, as in autograd)."""
    B, T, K, d = (int(n) for n in c.shape)
    N = int(n_cells)
    A = c.new_zeros(B, N + 1, d)  # row N: the dummy cell of the padded slots
    R = c.new_zeros(B, d)
    dc = torch.zeros_like(c)
    db = torch.zeros_like(b)
    zero = c.new_zeros(())
    for t in range(T - 1, -1, -1):
        ok = valid[:, t, :, None]
        index = torch.where(valid[:, t], cells[:, t], torch.full_like(cells[:, t], N))
        index = index[:, :, None].expand(B, K, d)
        if d_q_post is not None:
            R = R + d_q_post[:, t]
        if d_h_sel is not None:
            A = A.scatter_add(1, index, torch.where(ok, d_h_sel[:, t], zero))
        carried = A.gather(1, index)
        share = (R / N)[:, None, :]
        g = carried + share
        b_t = b[:, t]
        dc[:, t] = torch.where(ok, b_t * g, zero)
        db[:, t] = torch.where(ok, (c[:, t] - h_old[:, t]) * g, zero)
        A = A.scatter(1, index, torch.where(ok, (1 - b_t) * carried - b_t * share, zero))
        if d_q_pre is not None:
            R = R + d_q_pre[:, t]
        row = reset[:, t]
        A = A.masked_fill(row[:, None, None], 0)
        R = R.masked_fill(row[:, None], 0)
    return dc, db


def dense_recurrence_backward(
    c: Tensor,
    b: Tensor,
    h_old: Tensor,
    reset: Tensor,
    n_cells: int,
    d_q_pre: Tensor | None,
    d_q_post: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """The sweep of the rows that ran the dense round (module docstring, "Dense-fallback rows"):
    ``(dL/dc, dL/db)``, each ``[B', T, N, d]``, from ``c``, ``b``, ``h_old`` ``[B', T, N, d]``
    (every cell of every position), ``reset [B', T]`` and the cotangents of ``q`` before and
    after each position (``[B', T, d]`` or ``None``). The vector-Jacobian product of
    ``h' = (1 − b) h + b c``, ``q' = mean(h')`` and the reset over the window."""
    Bd, T, N, d = (int(n) for n in c.shape)
    assert N == int(n_cells)
    A = c.new_zeros(Bd, N, d)
    R = c.new_zeros(Bd, d)
    dc = torch.zeros_like(c)
    db = torch.zeros_like(b)
    for t in range(T - 1, -1, -1):
        if d_q_post is not None:
            R = R + d_q_post[:, t]
        g = A + (R / N)[:, None, :]
        b_t = b[:, t]
        dc[:, t] = b_t * g
        db[:, t] = (c[:, t] - h_old[:, t]) * g
        A = (1 - b_t) * g
        R = torch.zeros_like(R) if d_q_pre is None else d_q_pre[:, t]
        row = reset[:, t]
        A = A.masked_fill(row[:, None, None], 0)
        R = R.masked_fill(row[:, None], 0)
    return dc, db


#: The sweep :class:`StateRecurrence` runs: the reference above unless a faster implementation of
#: the same contract was installed (the CUDA kernel of :mod:`hsn.kernels.window`, on CUDA tensors).
RecurrenceFn = Callable[..., tuple[Tensor, Tensor] | None]
_fast_recurrence: RecurrenceFn | None = None


_fast_dense_recurrence: RecurrenceFn | None = None


def set_fast_recurrence(fn: RecurrenceFn | None, dense: RecurrenceFn | None = None) -> None:
    """Install (or, with ``None``, remove) a faster :func:`recurrence_backward` (and
    :func:`dense_recurrence_backward`). Each is called with the reference's arguments and must
    return ``None`` for inputs it does not handle."""
    global _fast_recurrence, _fast_dense_recurrence
    _fast_recurrence = fn
    _fast_dense_recurrence = dense


class StateRecurrence(torch.autograd.Function):
    """The state recurrence of a window as one autograd node (module docstring): forward returns
    the taped ``q`` before and after every position and the taped selected states; backward is
    the linear sweep that yields ``dL/dĉ`` and ``dL/db̂``. The outputs are *recorded* values, not
    recomputed from ``ĉ`` / ``b̂`` (they agree to the rounding of the replay's batched matmuls),
    so the node is not a forward/backward pair in the ``gradcheck`` sense; its backward is exact
    for the recorded trajectory. ``c_dense`` / ``b_dense`` (``[B', T, N, d]`` or ``None``) are
    the dense-fallback rows' candidates and gates (``tape["dense_ids"]`` names the rows)."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        c: Tensor,
        b: Tensor,
        c_dense: Tensor | None,
        b_dense: Tensor | None,
        tape: dict[str, Tensor],
        n_cells: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if c_dense is None or b_dense is None:
            ctx.save_for_backward(c, b)
        else:
            ctx.save_for_backward(c, b, c_dense, b_dense)
        ctx.tape = tape  # type: ignore[attr-defined]
        ctx.n_cells = n_cells  # type: ignore[attr-defined]
        # fresh tensors: the tape's own are read as constants elsewhere in the replay
        return tape["q_pre"].clone(), tape["q_post"].clone(), tape["h_sel"].clone()

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx, d_q_pre: Tensor | None, d_q_post: Tensor | None, d_h_sel: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, None, None]:
        saved = ctx.saved_tensors  # type: ignore[attr-defined]
        c, b = saved[0], saved[1]
        c_dense, b_dense = (saved[2], saved[3]) if len(saved) == 4 else (None, None)
        tape: dict[str, Tensor] = ctx.tape  # type: ignore[attr-defined]
        n_cells: int = ctx.n_cells  # type: ignore[attr-defined]
        args = (
            c,
            b,
            tape["h_old"],
            tape["cells"],
            tape["valid"],
            tape["reset"],
            n_cells,
            d_q_pre,
            d_q_post,
            d_h_sel,
        )
        result = _fast_recurrence(*args) if _fast_recurrence is not None else None
        if result is None:
            result = recurrence_backward(*args)
        dc, db = result
        if c_dense is None or b_dense is None:
            return dc, db, None, None, None, None
        # the dense-fallback rows: their slots were not executed (zeros), their cells all were
        rows = tape["dense_ids"]
        dc = dc.index_fill(0, rows, 0)
        db = db.index_fill(0, rows, 0)
        dense_args = (
            c_dense,
            b_dense,
            tape["h_dense_old"],
            tape["reset"].index_select(0, rows),
            n_cells,
            None if d_q_pre is None else d_q_pre.index_select(0, rows),
            None if d_q_post is None else d_q_post.index_select(0, rows),
        )
        dense = _fast_dense_recurrence(*dense_args) if _fast_dense_recurrence is not None else None
        if dense is None:
            dense = dense_recurrence_backward(*dense_args)
        return dc, db, dense[0], dense[1], None, None


def _scorer_logits(
    scorer: Scorer, u: Tensor, eps: Tensor, h: Tensor, age: Tensor, scalars: Tensor
) -> Tensor:
    """The scorer's logits ``[R, N, M]`` of :meth:`Scorer._assemble`'s features without building
    them: for the linear scorer the ``[R, N, 3d+8]`` feature tensor (``u`` and ``eps`` repeated
    for every cell: 200 MB for a B32/T256 window) is only ever multiplied by one weight matrix,
    so the product is taken per feature block and the row terms are added once per row. The same
    sum in another order, in the state's dtype (as the caller's cast makes the assembled
    product). Scorers this
    does not cover (the MLP scorer, excluded features) take the assembled path."""
    net = scorer.net
    if not isinstance(net, nn.Linear) or scorer.exclude:
        out: Tensor = scorer(scorer._assemble(u, eps, h, age, scalars))
        return out
    d = int(h.shape[-1])
    weight = net.weight
    row = F.linear(torch.cat([u, eps], dim=-1), weight[:, : 2 * d])  # [R, M]
    tail = torch.cat([(age.to(u.dtype) / scorer.age_max)[:, :, None], scalars], dim=-1)
    dtype = h.dtype  # the state's dtype: the partial products are summed in it, not in bf16
    cells = F.linear(h, weight[:, 2 * d : 3 * d]).to(dtype)  # [R, N, M]
    cells = cells + F.linear(tail, weight[:, 3 * d :], net.bias).to(dtype)
    return cells + row.to(dtype)[:, None, :]


def _readout_of_slots(chosen: Tensor, sel: Selection, gate: str | None, q: Tensor) -> Tensor:
    """:meth:`HSN._selected_readout`'s hard branch from the selected slots' updated states
    ``chosen [R, K, d]`` (the same operations; ``q [R, d]`` serves the dense-fallback rows, of
    which the split window has none)."""
    dtype = chosen.dtype
    safe_idx = sel.cell_idx.masked_fill(~sel.valid, 0)
    weights = sel.valid.to(dtype)
    if gate == "ste":
        mask = sel.cell_mask.gather(1, safe_idx).to(dtype)
        weights = weights * (1.0 + mask - mask.detach())
    readout = (chosen * weights[..., None]).sum(1)
    readout = readout / sel.valid.sum(1, keepdim=True).clamp_min(1).to(dtype)
    return torch.where(sel.dense_fallback[:, None], q, readout)


# --------------------------------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------------------------------


class SplitWindow:
    """The split training window (module docstring): called like :meth:`HSN.forward` /
    :class:`~hsn.train.compile.CompiledWindow` and returning the same ``(logits, state, info)``,
    with ``logits`` and ``info`` on one autograd graph whose backward is the model's.

    ``compiled`` is the run's compiled window (its ``tape`` variant runs the forward sweep and
    its counters account for it) or ``None`` (the sweep then runs eagerly: the CPU, tests).
    Calls outside the split window's use — evaluation, ``torch.no_grad()``, a diagnostic
    ``capture`` — go to ``fallback`` (the compiled window, or the model's forward)."""

    def __init__(self, model: HSN, cfg: RunConfig, compiled: CompiledWindow | None) -> None:
        reason = scope_error(cfg)
        if reason is not None:
            raise ValueError(f"hardware.window=split is not available for this run: {reason}")
        self.model = model
        self.cfg = cfg
        self.compiled = compiled
        self.fallback: Callable[..., WindowOut] = compiled if compiled is not None else model
        self.calls = 0
        #: ``hardware.window=fused``: the sweep's cell round is the fused CUDA op.
        self.fused = cfg.hardware.window == "fused"
        #: ``hardware.sweep_twin``: the compiled sweep runs the pre-cast twin of the model.
        self.twin = cfg.hardware.sweep_twin
        device = next(model.parameters()).device
        if self.fused:
            reason = (
                "it needs a compiled window on CUDA (hardware.compile_mode)"
                if compiled is None or device.type != "cuda"
                else window_kernels.cell_round_error(model.cell)
            )
            if reason is not None:
                raise ValueError(f"hardware.window=fused is not available for this run: {reason}")
        if device.type == "cuda" and window_kernels.available():
            # the recurrence's backward as one kernel launch per window (fp32 state)
            window_kernels.prepare(device.index or 0)
            set_fast_recurrence(
                window_kernels.recurrence_backward, window_kernels.dense_recurrence_backward
            )
        #: Wall seconds of the last call's phases (forward sweep, replay), for the measurements.
        self.last_phase_seconds: dict[str, float] = {}

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
        if capture or not model.training or not torch.is_grad_enabled():
            return self.fallback(
                tokens,
                ctx,
                state,
                reset,
                capture=capture,
                reset_columns=reset_columns,
                prev_tokens=prev_tokens,
            )
        B, T = model._check_window_inputs(tokens, ctx, state, reset)
        if reset_columns is not None:
            model._reset_columns(reset, reset_columns)  # validated; every position resets anyway
        controls = model._controls()
        if controls.gate != "ste" or controls.epsilon > 0.0:
            raise RuntimeError(
                f"the split window needs the ste gate without exploration at every step, got "
                f"gate={controls.gate!r}, epsilon={controls.epsilon!r}"
            )
        mode = model._window_mode()
        # P2.2's dense fallback: drawn here, once per row per window, as the autograd window does
        rng = torch.get_rng_state()
        dense_rows = model._draw_dense_rows(B, state)
        dense_ids: Tensor | None = None
        dense_indices: Tensor | None = None
        if dense_rows is not None:
            dense_ids = dense_rows.nonzero().flatten()
            if self.fused or self._dense_replay_bytes(int(dense_ids.numel()), T) > DENSE_BYTES:
                # the fused round has no dense rows; a window with too many would not fit the
                # batched replay: the autograd window runs it, from the same draw
                torch.set_rng_state(rng)
                return self.fallback(
                    tokens,
                    ctx,
                    state,
                    reset,
                    reset_columns=reset_columns,
                    prev_tokens=prev_tokens,
                )
            dense_indices = model._dense_indices(dense_rows, bucket=self.compiled is not None)
        pro = model._window_prologue(
            tokens, ctx, state, reset, mode, reset_first=True, prev_tokens=prev_tokens
        )
        started = time.perf_counter()
        with torch.no_grad():
            if self.compiled is not None:
                tape = self.compiled.tape(
                    pro,
                    state,
                    reset,
                    fused=self.fused,
                    twin=self.twin,
                    dense_rows=dense_rows,
                    dense_indices=dense_indices,
                )
            else:
                tape = run_tape(
                    _tape_block,
                    model,
                    pro,
                    state,
                    reset,
                    T,
                    dense_rows=dense_rows,
                    dense_indices=dense_indices,
                )
        swept = time.perf_counter()
        out = self._replay(pro, tape, reset, B, T, dense_ids)
        self.calls += 1
        self.last_phase_seconds = {"sweep": swept - started, "replay": time.perf_counter() - swept}
        return out

    def _dense_replay_bytes(self, rows: int, T: int) -> int:
        """A bound on the memory of replaying ``rows`` dense-fallback rows: every unit's hidden
        layer for every cell of every position, about three times over (the activation, what
        its backward keeps, its gradient)."""
        cfg = self.model.cfg
        width = 2 if torch.is_autocast_enabled(next(self.model.parameters()).device.type) else 4
        return 3 * rows * T * cfg.n_cells * cfg.n_units * cfg.hidden * width

    def _replay(
        self,
        pro: _Prologue,
        tape: dict[str, Tensor],
        reset: Tensor,
        B: int,
        T: int,
        dense_ids: Tensor | None = None,
    ) -> WindowOut:
        model = self.model
        cfg = model.cfg
        cell = model.cell
        controls = model._controls()
        N, M, d = cfg.n_cells, cfg.n_units, cfg.d
        R = B * T
        K = int(tape["sel_cell_idx"].shape[-1])
        h_pre = tape["tape_h"].reshape(R, N, d)
        q_pre = tape["tape_q"].reshape(R, d)
        age_pre = tape["tape_age"].reshape(R, N)
        dtype = h_pre.dtype  # the state's (fp32 master) dtype
        cells = tape["sel_cell_idx"].reshape(R, K)
        units = tape["sel_unit_idx"].reshape(R, K)
        valid = tape["sel_valid"].reshape(R, K)
        z = pro.z.reshape(R, d)
        context = pro.context.reshape(R, -1)
        scalars = pro.scalars.reshape(R, N, -1)

        # the shared path's local half: S, and what the cell and the scorer read of the error
        with torch.no_grad():
            z_hat_value = model.predictor(q_pre)
        u = model.shared(torch.cat([z, q_pre, context], dim=-1))
        u_s = u.to(dtype)
        eps_s = (z - z_hat_value).to(dtype)  # A6 `eps`: z keeps its gradient, the prediction none
        # the scorer on detached features (A6 `router`), and the recorded decision's gradient path
        unit_logits = _scorer_logits(
            model.scorer, u_s.detach(), eps_s.detach(), h_pre, age_pre, scalars
        ).to(dtype)  # [R, N, M]
        dense_r: Tensor | None = None
        if dense_ids is not None:
            dense_b = torch.zeros(B, dtype=torch.bool, device=reset.device)
            dense_r = dense_b.index_fill(0, dense_ids, True)[:, None].expand(B, T).reshape(R)
        selection = model.router.finish_recorded(
            unit_logits,
            cells,
            units,
            valid,
            tape["tape_theta"].reshape(R),
            mode=cfg.routing,
            gate=controls.gate,
            temperature=controls.tau,
            dense_rows=dense_r,
        )
        c_dense: Tensor | None = None
        b_dense: Tensor | None = None
        h_dense_old: Tensor | None = None
        with cell.window():
            c, b = cell.replay_sparse(u_s, eps_s, h_pre, selection)  # [R, K, d]
            if dense_ids is not None:
                # the rows that ran the dense round: every cell, the softmax mix of the units
                Bd = int(dense_ids.numel())

                def rows(x: Tensor) -> Tensor:
                    picked = x.reshape(B, T, *x.shape[1:]).index_select(0, dense_ids)
                    return picked.reshape(Bd * T, *x.shape[1:])

                h_dense_old = rows(h_pre)
                dense_state = CellGraphState._make(
                    h=h_dense_old,
                    q=rows(q_pre),
                    age=rows(age_pre),
                    pos=torch.zeros(Bd * T, dtype=torch.int64, device=reset.device),
                )
                alpha = torch.softmax(rows(unit_logits), dim=-1)
                _, dense_info = cell.step_dense(rows(u_s), rows(eps_s), dense_state, alpha)
                c_dense = dense_info["c"].reshape(Bd, T, N, d)
                b_dense = dense_info["b"].reshape(Bd, T, N, d)

        safe = cells.masked_fill(~valid, 0)
        recurrence_tape = {
            "h_old": h_pre.gather(1, safe[:, :, None].expand(R, K, d)).reshape(B, T, K, d),
            "cells": safe.reshape(B, T, K),
            "valid": valid.reshape(B, T, K),
            "reset": reset,
            "q_pre": tape["tape_q"],
            "q_post": tape["qs"],
            "h_sel": tape["tape_hsel"],
        }
        if dense_ids is not None:
            assert h_dense_old is not None
            recurrence_tape["dense_ids"] = dense_ids
            recurrence_tape["h_dense_old"] = h_dense_old.reshape(-1, T, N, d)
        q_pre_n, q_post_n, h_sel_n = StateRecurrence.apply(
            c.reshape(B, T, K, d), b.reshape(B, T, K, d), c_dense, b_dense, recurrence_tape, N
        )

        # the reads of the state that carry gradient: P, the selected readout, the decoder
        z_hat = model.predictor(q_pre_n.reshape(R, d))
        eps = z - z_hat
        readouts: Tensor | None = None
        if cfg.selected_readout:
            readouts = _readout_of_slots(
                h_sel_n.reshape(R, K, d), selection, controls.gate, q_post_n.reshape(R, d)
            ).reshape(B, T, d)
        routed = {
            "cell_idx": tape["sel_cell_idx"],
            "unit_idx": tape["sel_unit_idx"],
            "valid": tape["sel_valid"],
            "unit_logits": unit_logits.reshape(B, T, N, M),
            "cost_per_row": selection.cost.detach().reshape(B, T),
            "expected_cost": selection.expected_cost.reshape(B, T).mean(dim=0),
            "route_log_prob": selection.log_prob.reshape(B, T),
            "explored": tape["explored"],
            "refreshed": tape["refreshed"],
            "dense_fallback": (
                torch.zeros(T, dtype=torch.float32, device=reset.device)
                if dense_r is None
                else dense_r.reshape(B, T).float().mean(dim=0)
            ),
        }
        final = CellGraphState._make(h=tape["h"], q=tape["q"], age=tape["age"], pos=tape["pos"])
        return model._window_epilogue(
            pro,
            final,
            z_hats=z_hat.reshape(B, T, d),
            epss=eps.reshape(B, T, d),
            us=u.reshape(B, T, d),
            qs=q_post_n,
            hs=None,
            state_abs_max=tape["state_abs_max"],
            unit_logits=unit_logits.detach().reshape(B, T, N, M),
            routed=routed,
            gates=None,
            written=None,
            readouts=readouts,
        )

    # -- what the trainer reads of a compiled window ----------------------------------------------

    @property
    def compile_seconds(self) -> float:
        return self.compiled.compile_seconds if self.compiled is not None else 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "window": "fused" if self.fused else "split",
            "compiled_sweep": self.compiled is not None,
        }
