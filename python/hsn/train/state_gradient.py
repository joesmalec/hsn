"""P2.10c: the state gradient in the split window (``docs/p2_10c_state_gradient_window.md``).

The split window of P2.10 (:mod:`hsn.train.split`) needs amendment A6's full feedback cut: with
it the units, the gate and the messages read the state as a constant, so their backward is
local to a position and runs for every position at once, and only the gated carry crosses
positions. Without the ``state`` cut (``train.feedback_detach = [summary, eps, router]``, "the
state gradient") the cell's own read of the state carries gradient: the candidate ``c`` and the
gate ``b`` of a selected cell depend on the pre-round state of that cell and of the sources of
its messages, and the cotangent of the state at position ``t`` depends on the cell's backward
at every later position. That part of the backward is a recurrence over positions.

This module is that recurrence, in plain PyTorch, as the reference every faster implementation
is tested against. The split window keeps its three parts:

1. the forward sweep without autograd (unchanged: the same tape);
2. the batched replay with autograd (unchanged operations; it also hands the sweep the values
   it computed, :meth:`hsn.model.cell.PDGCell.replay_sparse`'s ``record``). The replay still
   reads the recorded state as a constant, so its graph is the *local* half of the gradient:
   the partial derivatives of every position's ``(c, b)`` with respect to the parameters, ``u``,
   ``eps`` and the selection's weights at a fixed state;
3. the state recurrence as one autograd node (:class:`StateGradientRecurrence`), whose backward
   is :func:`state_gradient_backward`: the sweep of P2.10 plus, per position, the
   vector-Jacobian product of the selected cells' ``(c, b)`` with respect to the pre-round
   state (:func:`cell_state_vjp`). It returns the *total* ``dL/dc`` and ``dL/db`` of every
   position, and autograd carries them through the replay's graph as before.

So the parameter gradients are still autograd's; what is written by hand is the path of the
gradient through the state, and it is exact for the recorded trajectory.

The sweep, with ``A [B, N, d]`` and ``R [B, d]`` the cotangents of the state after position
``t``, ``D_t``, ``P_t``, ``S_t`` those of ``q`` after ``t``, ``q`` before ``t`` and the selected
updated states (P2.10), and ``G_t`` that of the fast cells' states after ``t`` (A8's fast read
of the decoder)::

    A = 0; R = 0
    for t = T−1 .. 0:
        R += D_t;  A[i] += G_{t,i} (i < F);  A[cell_k] += S_{t,k}            (valid slots)
        g_k = A[cell_k] + R / N
        dc_k = b_k ⊙ g_k;  db_k = (c_k − h_old_k) ⊙ g_k
        (dh_k, dm_k) = cell_state_vjp(dc_k, db_k)        the cell's read of h[cell_k] and of m_k
        A[cell_k] = (1 − b_k) ⊙ A[cell_k] − b_k ⊙ R / N + dh_k
        A[src_{k,e}] += a_{cell_k,e} · Vᵀ dm_k           every source e of cell_k's messages
        R += P_t
        A, R = limit(A), limit(R)                        row by row (A6's limit; off by default)
        A, R = 0 on the rows that reset at t

The cell at a selected slot (``x = [u; ε; h_i; m_i]`` of width ``n``, unit ``j``, slot weight
``α``; :meth:`PDGCell.step_sparse`)::

    m = Σ_e a_e V h[src_e];   v = g_n ⊙ x · r,  r = 1 / sqrt(mean(x²) + eps)
    hid = v W1_j + b1_j;  F = act(hid) W2_j + b2_j;  gate_pre = G v + b_g;  b = σ(gate_pre)
    y = W_out F;  c = tanh(α · (g_c ⊙ y · ρ)),  ρ = 1 / sqrt(mean(y²) + eps)

and its backward into the state columns ``s = [h_i; m_i]`` of ``x`` (the columns of ``u`` and
``ε`` are the replay's)::

    dz = dc ⊙ (1 − c²);  dy = ρ (g_c ⊙ α dz) − ρ³ y · mean(y ⊙ g_c ⊙ α dz);  dF = W_outᵀ dy
    dhid = (W2_j dF) ⊙ act'(hid);  dg = db ⊙ b ⊙ (1 − b)
    dv_s = (W1_j)_s dhid + (Gᵀ)_s dg
    σ = ⟨hid − b1_j, dhid⟩ + ⟨gate_pre − b_g, dg⟩          = ⟨v, dv⟩ over all n columns
    ds = r (g_n)_s ⊙ dv_s − (r² / n) · s · σ               the norm's backward, state columns
    dh = ds[:d];  dm = ds[d:]

The norm's backward needs the inner product of ``v`` with the whole ``dv``; because ``v W1_j =
hid − b1_j`` and ``G v = gate_pre − b_g``, that product is ``σ`` and the sweep never forms the
``u`` and ``ε`` columns of ``dv``: the transposed first layer runs on ``2d`` of its ``n``
input columns. Without the cell norm ``ds = dv_s``; without the candidate norm ``y = W_out(α
F)`` and ``dF = α W_outᵀ dz``.

The limit (``train.state_grad_clip`` / ``train.state_grad_ratio``, :class:`RowLimits`): the
model limits the gradient of ``h`` and of ``q`` at the entry of every position, after the
position's reset and before any of its reads. At that point of the sweep ``A`` and ``R`` are the
whole cotangent of the state position ``t`` reads, so the sweep applies
:func:`hsn.model.gradclip.limit_row_grad` to each there. The rows that reset at ``t`` are
limited like the others and count in the batch's typical row (they carry what the position
passes back to the zeroed state), and are zeroed after it, as the reset's backward does. What
the decoder and the losses read of the state *after* position ``t − 1`` (``D``, ``G``, ``S``)
joins after the limit of position ``t``, as in the model.

This is the reference only: no kernel applies the limit, and the split window's scope offers
it to no training run (``row_limit=True`` admits it, for tests).

Scope (:func:`hsn.config.split_window_scope_error` with ``state_gradient=True``): the split
window's, with ``summary``, ``eps`` and ``router`` cut and ``state`` not; amendment A8's fast
cells and A9's address n-grams are inside it. A window with dense-fallback rows
(``train.p_dense``) runs the autograd window from the same draw.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.autograd.function import FunctionCtx, once_differentiable

from hsn.model.cell import PDGCell
from hsn.model.gradclip import limit_row_grad

__all__ = [
    "CellConstants",
    "RowLimits",
    "StateGradientRecurrence",
    "SweepLayouts",
    "activation_grad",
    "cell_constants",
    "cell_state_vjp",
    "set_fast_sweep",
    "state_gradient_backward",
    "sweep_layouts",
]

#: What :meth:`hsn.model.cell.PDGCell.replay_sparse` records for the sweep (``rms`` only with
#: the cell norm), each ``[B, T, K, …]`` when the sweep reads it.
RECORD_KEYS: tuple[str, ...] = ("y", "alpha", "hidden", "gate_pre", "m", "rms")


class RowLimits(NamedTuple):
    """Amendment A6's per-row limit on what a position passes back into the state
    (``train.state_grad_clip``, ``train.state_grad_ratio``; :mod:`hsn.model.gradclip`), as the
    model applies it at the entry of every position: the absolute limits of a row of ``dL/dh``
    and of ``dL/dq`` (``None``: none) and the multiple of the batch's typical live row that
    bounds both (``None``: none)."""

    h: float | None
    q: float | None
    ratio: float | None


class CellConstants(NamedTuple):
    """What the reverse sweep reads of the cell's parameters (:func:`cell_constants`): detached
    values in the state's dtype, constant over a window. ``w1_state``, ``gate_state`` and
    ``norm_gain_state`` are the slices of the unit bank's first layer, the gate and the cell
    norm's gain at the state columns ``[h_i; m_i]`` of the cell input, its last ``2d``."""

    w_out: Tensor  # [d, d]: nn.Linear's weight, y = F @ w_out.T
    cand_gain: Tensor | None  # [d]: the candidate norm's gain (None: model.cand_norm = none)
    cand_eps: float
    w2: Tensor  # [M, H, d]
    w1_state: Tensor  # [M, 2d, H]
    b1: Tensor  # [M, H]
    gate_state: Tensor  # [d, 2d]
    gate_bias: Tensor  # [d]
    norm_gain_state: Tensor | None  # [2d]: the cell norm's gain (None: model.cell_norm = none)
    in_dim: int  # n: the width the cell norm averages over (4d; 3d with model.ablate_shared_u)
    v: Tensor  # [d, d]: the message projection's weight, V h = h @ v.T
    edge_weights: Tensor  # [N, D_max]: a_ij in the padded view, 0 on padding
    pad_src: Tensor  # int64 [N, D_max]: the source of every padded slot
    activation: str
    pad_mask: Tensor | None = None  # bool [N, D_max]: the slots that hold an edge


def cell_constants(cell: PDGCell, dtype: torch.dtype) -> CellConstants:
    """The :class:`CellConstants` of ``cell`` in ``dtype`` (the state's). Inside
    :meth:`PDGCell.window` the edge weights are the window's."""
    d, n = cell.d, cell.in_dim
    units = cell.units

    def const(x: Tensor) -> Tensor:
        return x.detach().to(dtype)

    cand_gain: Tensor | None = None
    cand_eps = 0.0
    if isinstance(cell.cand_norm, nn.RMSNorm):
        cand_gain = const(cell.cand_norm.weight)
        eps = cell.cand_norm.eps
        cand_eps = float(torch.finfo(dtype).eps if eps is None else eps)
    norm_gain: Tensor | None = None
    if isinstance(cell.norm, nn.RMSNorm):
        norm_gain = const(cell.norm.weight)[n - 2 * d :]
    return CellConstants(
        w_out=const(cell.W_out.weight),
        cand_gain=cand_gain,
        cand_eps=cand_eps,
        w2=const(units.W2),
        w1_state=const(units.W1)[:, n - 2 * d :, :],
        b1=const(units.b1),
        gate_state=const(cell.gate.weight)[:, n - 2 * d :],
        gate_bias=const(cell.gate.bias),
        norm_gain_state=norm_gain,
        in_dim=n,
        v=const(cell.V.weight),
        edge_weights=cell.graph._padded_weights_as(dtype).detach(),
        pad_src=cell.graph._pad_src,
        activation=units.activation_name,
        pad_mask=cell.graph._pad_mask,
    )


def activation_grad(name: str, x: Tensor) -> Tensor:
    """The derivative of ``model.activation`` at ``x`` (:data:`hsn.model.encoder.ACTIVATIONS`;
    ``gelu`` is the exact, erf one)."""
    if name == "gelu":
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return cdf + x * torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    if name == "relu":
        return (x > 0).to(x.dtype)
    if name == "silu":
        s = torch.sigmoid(x)
        return s * (1.0 + x * (1.0 - s))
    if name == "tanh":
        return 1.0 - torch.tanh(x).square()
    raise ValueError(f"no derivative for model.activation={name!r}")


def _by_unit(x: Tensor, units: Tensor, weight: Tensor) -> Tensor:
    """``out[p] = x[p] @ weight[units[p]]`` for ``x […, K, a]``, ``units […, K]`` and ``weight
    [M, a, b]``: one product per unit on that unit's slots (the grouped pass of the unit bank,
    transposed), without a per-slot copy of the weights."""
    flat = x.reshape(-1, int(x.shape[-1]))
    index = units.reshape(-1)
    out = flat.new_zeros(flat.shape[0], int(weight.shape[-1]))
    for j in range(int(weight.shape[0])):
        rows = (index == j).nonzero().flatten()
        if rows.numel():
            out = out.index_copy(0, rows, flat.index_select(0, rows) @ weight[j])
    return out.reshape(*x.shape[:-1], int(weight.shape[-1]))


class SweepLayouts(NamedTuple):
    """The cell's two transposed unit layers in the layouts a compiled reverse sweep multiplies
    with (:func:`sweep_layouts`): static shapes, one product per layer for the slots whose unit
    the router chose and one batched product per layer for the slots whose unit is fixed.

    The routed slots use the all-units layout of the forward sweep, transposed: a slot's
    cotangent is written into its unit's block of an operand that is zero elsewhere, so one
    matmul serves every unit (``M`` times the arithmetic of the slot's own unit, on static
    shapes). The first ``n_fast`` slots (amendment A8's fast cells with ``model.fast_units =
    fixed``: slot ``i`` at unit ``i mod M`` at every position) multiply with their own unit's
    weights, stacked per slot."""

    w2_t: Tensor  # [M·d, H]: row j·d + k is unit j's W2[:, k]
    w1_state_t: Tensor  # [M·H + d, 2d]: row j·H + h is unit j's W1[state rows, h]; then the gate
    n_fast: int
    fast_w2_t: Tensor | None  # [F, d, H]
    fast_w1_state_t: Tensor | None  # [F, H + d, 2d]: slot i's W1[state rows, :]ᵀ; then the gate


def sweep_layouts(consts: CellConstants, n_fast: int = 0) -> SweepLayouts:
    """The :class:`SweepLayouts` of a cell's weights; ``n_fast`` leading slots have a fixed unit."""
    n_units, hidden, d = (int(n) for n in consts.w2.shape)
    w2_t = consts.w2.permute(0, 2, 1).reshape(n_units * d, hidden)
    w1_t = consts.w1_state.permute(0, 2, 1).reshape(n_units * hidden, 2 * d)
    w1_state_t = torch.cat([w1_t, consts.gate_state], dim=0)
    if not n_fast:
        return SweepLayouts(w2_t, w1_state_t, 0, None, None)
    slots = torch.arange(n_fast, device=consts.w2.device) % n_units
    fast_w1 = consts.w1_state.index_select(0, slots).transpose(1, 2)  # [F, H, 2d]
    gate = consts.gate_state[None].expand(n_fast, d, 2 * d)
    return SweepLayouts(
        w2_t,
        w1_state_t,
        n_fast,
        consts.w2.index_select(0, slots).transpose(1, 2).contiguous(),
        torch.cat([fast_w1, gate], dim=1),
    )


def _static_product(
    x: Tensor,
    units: Tensor,
    flat_t: Tensor,
    fast_t: Tensor | None,
    n_fast: int,
    extra: Tensor | None = None,
) -> Tensor:
    """``out[b, k] = x[b, k] @ W[units[b, k]] (+ extra[b, k] @ G)`` in the static layouts of
    :class:`SweepLayouts`: ``x [B, K, a]``, ``flat_t [M·a (+ r), o]`` (the units' blocks, then
    ``G``), ``fast_t [F, a (+ r), o]``, ``extra [B, K, r]`` the rider's operand."""
    B, K, a = (int(n) for n in x.shape)
    width = int(flat_t.shape[0]) - (0 if extra is None else int(extra.shape[-1]))
    routed = x[:, n_fast:].reshape(-1, a)
    index = units[:, n_fast:].reshape(-1)
    columns = index[:, None] * a + torch.arange(a, device=x.device)[None, :]
    operand = routed.new_zeros(routed.shape[0], width).scatter(1, columns, routed)
    if extra is not None:
        operand = torch.cat([operand, extra[:, n_fast:].reshape(routed.shape[0], -1)], dim=1)
    out = (operand @ flat_t).reshape(B, K - n_fast, -1)
    if not n_fast:
        return out
    assert fast_t is not None
    fixed = x[:, :n_fast] if extra is None else torch.cat([x, extra], dim=-1)[:, :n_fast]
    return torch.cat([torch.bmm(fixed.transpose(0, 1), fast_t).transpose(0, 1), out], dim=1)


def cell_state_vjp(
    dc: Tensor,
    db: Tensor,
    *,
    c: Tensor,
    b: Tensor,
    h_old: Tensor,
    units: Tensor,
    y: Tensor,
    alpha: Tensor,
    hidden: Tensor,
    gate_pre: Tensor,
    m: Tensor,
    rms: Tensor | None,
    consts: CellConstants,
    layouts: SweepLayouts | None = None,
) -> tuple[Tensor, Tensor]:
    """The vector-Jacobian product of the selected cells' candidate and gate with respect to
    what they read of the state (module docstring): ``(dh, dvm)``, each ``[…, K, d]``. ``dh``
    is the cotangent of the slot's own pre-round state ``h[cell_k]`` through the cell input
    (not the carry); ``dvm = Vᵀ dm`` is the cotangent of the slot's weighted source sum, so a
    source ``e`` of the cell receives ``a_e · dvm``. ``dc``, ``db`` (``[…, K, d]``) are the
    cotangents of ``c`` and ``b``; the other arguments are the recorded forward values of the
    same slots (``units`` int64 ``[…, K]``, the unit that ran; ``y``, ``gate_pre``, ``m`` ``[…,
    K, d]``; ``alpha``, ``rms`` ``[…, K]``; ``hidden [… , K, H]``). Nothing here depends on a
    slot being valid: the caller masks the padded slots' results.

    ``layouts`` (``[B, K, …]`` arguments only) takes the two transposed unit layers in the
    static layouts of a compiled sweep (:class:`SweepLayouts`) instead of one product per unit
    on that unit's slots: the same values to rounding, in the arithmetic the kernels run."""
    dtype = c.dtype
    d = int(c.shape[-1])
    dz = dc * (1.0 - c * c)
    weight = alpha.to(dtype)[..., None]
    if consts.cand_gain is not None:  # c = tanh(alpha * RMSNorm(y)), y = W_out F
        scaled = consts.cand_gain * (weight * dz)
        rho = torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + consts.cand_eps)
        dy = rho * scaled - rho.pow(3) * y * (y * scaled).mean(dim=-1, keepdim=True)
        d_unit = dy @ consts.w_out
    else:  # c = tanh(W_out(alpha * F))
        d_unit = weight * (dz @ consts.w_out)
    hid = hidden.to(dtype)
    # the executed unit's two layers, transposed: F = act(hid) @ W2_j + b2_j, hid = v @ W1_j + b1_j
    d_gate = db * b * (1.0 - b)
    if layouts is None:
        d_act = _by_unit(d_unit, units, consts.w2.transpose(1, 2))  # […, K, H]
        d_hid = d_act * activation_grad(consts.activation, hid)
        dv = _by_unit(d_hid, units, consts.w1_state.transpose(1, 2))  # […, K, 2d]
        dv = dv + d_gate @ consts.gate_state
    else:
        fast = layouts.n_fast
        d_act = _static_product(d_unit, units, layouts.w2_t, layouts.fast_w2_t, fast)
        d_hid = d_act * activation_grad(consts.activation, hid)
        dv = _static_product(
            d_hid, units, layouts.w1_state_t, layouts.fast_w1_state_t, fast, extra=d_gate
        )
    if consts.norm_gain_state is not None:
        if rms is None:
            raise ValueError("the cell norm's backward needs the recorded rms")
        sigma = (d_hid * (hid - consts.b1[units])).sum(dim=-1)
        sigma = sigma + (d_gate * (gate_pre - consts.gate_bias)).sum(dim=-1)
        state = torch.cat([h_old, m], dim=-1)
        r = rms.to(dtype)[..., None]
        ds = r * (consts.norm_gain_state * dv) - (r * r / consts.in_dim) * state * sigma[..., None]
    else:
        ds = dv
    return ds[..., :d], ds[..., d:] @ consts.v


def state_gradient_backward(
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
    trace: dict[str, Tensor] | None = None,
    initial: tuple[Tensor, Tensor] | None = None,
    layouts: SweepLayouts | None = None,
    limits: RowLimits | None = None,
) -> tuple[Tensor, Tensor]:
    """The reverse sweep of the module docstring in plain torch (any device and dtype): the
    total ``(dL/dc, dL/db)``, each ``[B, T, K, d]``.

    ``c``, ``b``, ``h_old`` are ``[B, T, K, d]`` (the replay's candidate and gate, the recorded
    pre-round states of the slots); ``cells`` and ``units`` int64 ``[B, T, K]`` (cell 0 and unit
    0 on padded slots), ``valid [B, T, K]``, ``reset`` bool ``[B, T]``. The cotangents: ``d_q_pre``
    / ``d_q_post`` ``[B, T, d]`` (``q`` before / after each position), ``d_h_sel [B, T, K, d]``
    (the selected updated states) and ``d_h_fast [B, T, F, d]`` (the states of cells ``0 .. F−1``
    after each position, A8's fast read), ``None`` for a read that carried none. ``record`` holds
    :data:`RECORD_KEYS` as ``[B, T, K, …]`` and ``consts`` the cell's weights.

    Padded slots get exact zeros and never touch the cell cotangent (``where``, never a product
    with a mask), as in :func:`hsn.train.split.recurrence_backward`, of which this is the
    extension: with ``cell_state_vjp`` returning zeros it is that sweep.

    ``trace`` (tests, and the oracle of a kernel's intermediate values) receives, stacked over
    the positions at dim 1: ``A [B, T, N, d]`` and ``R [B, T, d]``, the cotangents of the state
    that position ``t`` reads (after its reset; before they are zeroed on the rows that reset;
    with ``limits``, as limited: what the position passes back), and ``dh`` / ``dvm``
    ``[B, T, K, d]``, :func:`cell_state_vjp`'s results on the valid slots; with ``limits`` also
    ``limited_h`` / ``limited_q``, bool ``[B, T]``: the rows the limit changed.

    ``limits``: amendment A6's per-row limit (:class:`RowLimits`; module docstring), ``None``
    without one.

    ``initial`` is the cotangent ``(A [B, N, d], R [B, d])`` of the state after the last
    position. A window's final state is detached, so the window passes ``None`` (zeros); a
    test gives a single position the cotangent that autograd found for the state after it,
    which checks one step of the sweep with nothing accumulated. ``layouts``:
    :func:`cell_state_vjp`'s (the transposed unit layers in a compiled sweep's layouts)."""
    B, T, K, d = (int(n) for n in c.shape)
    N = int(n_cells)
    D = int(consts.pad_src.shape[1])
    n_fast = 0 if d_h_fast is None else int(d_h_fast.shape[2])
    A = c.new_zeros(B, N + 1, d)  # row N: the dummy cell of the padded slots
    R = c.new_zeros(B, d)
    if initial is not None:
        A = torch.cat([initial[0].to(c.dtype), A[:, N:]], dim=1)
        R = initial[1].to(c.dtype)
    dc = torch.zeros_like(c)
    db = torch.zeros_like(b)
    zero = c.new_zeros(())
    rms = record.get("rms")
    steps: dict[str, list[Tensor]] = {"A": [], "R": [], "dh": [], "dvm": []}
    if limits is not None:
        steps.update(limited_h=[], limited_q=[])
    for t in range(T - 1, -1, -1):
        ok = valid[:, t, :, None]
        safe = torch.where(valid[:, t], cells[:, t], torch.zeros_like(cells[:, t]))
        index = torch.where(valid[:, t], cells[:, t], torch.full_like(cells[:, t], N))
        index = index[:, :, None].expand(B, K, d)
        if d_q_post is not None:
            R = R + d_q_post[:, t]
        if d_h_fast is not None and n_fast:
            A = torch.cat([A[:, :n_fast] + d_h_fast[:, t], A[:, n_fast:]], dim=1)
        if d_h_sel is not None:
            A = A.scatter_add(1, index, torch.where(ok, d_h_sel[:, t], zero))
        carried = A.gather(1, index)
        share = (R / N)[:, None, :]
        g = carried + share
        b_t, c_t, h_t = b[:, t], c[:, t], h_old[:, t]
        dc_t = torch.where(ok, b_t * g, zero)
        db_t = torch.where(ok, (c_t - h_t) * g, zero)
        dc[:, t] = dc_t
        db[:, t] = db_t
        # the cell's read of the state: its own cell and the sources of its messages
        dh, dvm = cell_state_vjp(
            dc_t,
            db_t,
            c=c_t,
            b=b_t,
            h_old=h_t,
            units=units[:, t],
            y=record["y"][:, t],
            alpha=record["alpha"][:, t],
            hidden=record["hidden"][:, t],
            gate_pre=record["gate_pre"][:, t],
            m=record["m"][:, t],
            rms=None if rms is None else rms[:, t],
            consts=consts,
            layouts=layouts,
        )
        dh = torch.where(ok, dh, zero)
        dvm = torch.where(ok, dvm, zero)
        A = A.scatter(1, index, torch.where(ok, (1 - b_t) * carried - b_t * share + dh, zero))
        source = consts.pad_src[safe].reshape(B, K * D, 1).expand(B, K * D, d)
        weighted = consts.edge_weights[safe][..., None] * dvm[:, :, None, :]  # [B, K, D, d]
        A = A.scatter_add(1, source, weighted.reshape(B, K * D, d))
        if d_q_pre is not None:
            R = R + d_q_pre[:, t]
        if limits is not None:  # A6: what position t passes back, row by row
            free_h, free_q = A[:, :N], R
            A = torch.cat([limit_row_grad(free_h, limits.h, limits.ratio), A[:, N:]], dim=1)
            R = limit_row_grad(free_q, limits.q, limits.ratio)
            if trace is not None:
                steps["limited_h"].append((A[:, :N] != free_h).flatten(1).any(dim=1))
                steps["limited_q"].append((R != free_q).any(dim=1))
        if trace is not None:
            steps["A"].append(A[:, :N])
            steps["R"].append(R)
            steps["dh"].append(dh)
            steps["dvm"].append(dvm)
        row = reset[:, t]
        A = A.masked_fill(row[:, None, None], 0)
        R = R.masked_fill(row[:, None], 0)
    if trace is not None:
        for key, values in steps.items():
            trace[key] = torch.stack(values[::-1], dim=1)
    return dc, db


#: A faster :func:`state_gradient_backward` (:class:`hsn.train.state_sweep.SweepRunner`: the
#: kernels on the card), called with its arguments and ``n_fast`` / ``n_read``; it returns
#: ``None`` for inputs it does not take.
FastSweep = Callable[..., "tuple[Tensor, Tensor] | None"]
_fast_sweep: FastSweep | None = None


def set_fast_sweep(fn: FastSweep | None) -> None:
    """Install (or, with ``None``, remove) a faster implementation of
    :func:`state_gradient_backward` for :class:`StateGradientRecurrence`."""
    global _fast_sweep
    _fast_sweep = fn


class StateGradientRecurrence(torch.autograd.Function):
    """The state recurrence of a window with the state gradient, as one autograd node (module
    docstring): :class:`hsn.train.split.StateRecurrence` with :func:`state_gradient_backward`
    as its backward and a fourth output. Forward returns the taped ``q`` before and after every
    position, the taped selected states and the taped states of the fast cells after every
    position (``[B, T, F, d]``; ``F = 0`` without A8's fast read); backward returns the total
    ``dL/dĉ`` and ``dL/db̂``. As there, the outputs are recorded values, and the backward is
    exact for the recorded trajectory. ``tape`` holds that node's entries and ``h_fast``, the
    recorded ``units`` and the replay's record (:data:`RECORD_KEYS`, ``[B, T, K, …]``);
    ``consts`` are the cell's weights (:func:`cell_constants`); ``n_fast`` leading slots have a
    fixed unit (what a faster sweep, :func:`set_fast_sweep`, lays out apart); ``limits`` is
    amendment A6's per-row limit (:class:`RowLimits`), ``None`` without one."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        c: Tensor,
        b: Tensor,
        tape: dict[str, Tensor],
        consts: CellConstants,
        n_cells: int,
        n_fast: int = 0,
        limits: RowLimits | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ctx.save_for_backward(c, b)
        ctx.tape = tape  # type: ignore[attr-defined]
        ctx.consts = consts  # type: ignore[attr-defined]
        ctx.n_cells = n_cells  # type: ignore[attr-defined]
        ctx.n_fast = n_fast  # type: ignore[attr-defined]
        ctx.limits = limits  # type: ignore[attr-defined]
        ctx.set_materialize_grads(False)  # a read without a gradient arrives as None
        # fresh tensors: the tape's own are read as constants elsewhere in the replay
        return (
            tape["q_pre"].clone(),
            tape["q_post"].clone(),
            tape["h_sel"].clone(),
            tape["h_fast"].clone(),
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        d_q_pre: Tensor | None,
        d_q_post: Tensor | None,
        d_h_sel: Tensor | None,
        d_h_fast: Tensor | None,
    ) -> tuple[Tensor, Tensor, None, None, None, None, None]:
        c, b = ctx.saved_tensors  # type: ignore[attr-defined]
        tape: dict[str, Tensor] = ctx.tape  # type: ignore[attr-defined]
        consts: CellConstants = ctx.consts  # type: ignore[attr-defined]
        n_cells: int = ctx.n_cells  # type: ignore[attr-defined]
        limits: RowLimits | None = ctx.limits  # type: ignore[attr-defined]
        args = (
            c,
            b,
            tape["h_old"],
            tape["cells"],
            tape["units"],
            tape["valid"],
            tape["reset"],
            n_cells,
            d_q_pre,
            d_q_post,
            d_h_sel,
            d_h_fast,
            {key: tape[key] for key in RECORD_KEYS if key in tape},
            consts,
        )
        result = None
        if _fast_sweep is not None:
            n_fast: int = ctx.n_fast  # type: ignore[attr-defined]
            result = _fast_sweep(
                *args, n_fast=n_fast, n_read=int(tape["h_fast"].shape[2]), limits=limits
            )
        if result is None:
            result = state_gradient_backward(*args, limits=limits)
        return result[0], result[1], None, None, None, None, None
