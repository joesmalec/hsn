"""The fused word-event block, phase 1 (``docs/wev_fused_event.md``): the event loop of
``model.write_events`` as one forward sweep without autograd, one autograd node, and a backward
pass written by hand.

The compiled event loop (:meth:`hsn.model.hsn.HSN._event_block` through
:mod:`hsn.train.compile`) runs :meth:`HSN._position` at every event slot with autograd: about 86
small kernels per slot, forward and backward. Here an event slot ("round") is

* **forward**, no autograd: the front (``P`` and ``S`` on the carried summary), the scorer's
  logits, the top-K decision with its refresh probes, the messages, the cell input and its
  norm, the unit bank's two products, the candidate, the gate and the masked update. Every
  value the backward pass reads is written to a tape (:data:`TAPE_KEYS`).
* **backward, the state's part**: the reverse sweep of P2.10c
  (:func:`hsn.train.state_gradient.state_gradient_backward`, on the card
  :class:`hsn.train.state_sweep.ReverseSweep` with its kernels), unchanged. It carries the
  cotangent of ``h`` and ``q`` slot by slot, through the cell's read of the state, and returns
  the total ``dL/dc`` and ``dL/db`` of every slot and the cotangent of the block's start state.
* **backward, the parameters' part** (:func:`local_backward`): from ``dL/dc`` and ``dL/db`` of
  all slots at once, the cell's local backward and the scorer's, with one matrix product per
  unit and layer over the whole block instead of products at every slot. No autograd.

The node (:class:`EventRecurrence`) returns the taped summaries, read-outs and logits; ``P``,
the decoder, the balance term and ``S`` (replayed once for all slots on the taped summary, a
constant under the ``summary`` cut) stay autograd's, outside it. This is the layout of the
split window (:mod:`hsn.train.split`); what is new is that the cell's and the scorer's
parameter gradients no longer come from a batched autograd replay but from the tape.

No equation changes: the same cells, graph, units, ε, gate, router and ``q``. The scope is the
word-event recipe's round (:func:`scope_error`): top-K straight-through routing with executed
credit, the ``summary`` / ``eps`` / ``router`` cuts with the state gradient, both norms, the
linear scorer with A5/A9 addresses, A11 and A12/A13's gate on the summary or on ``u``.

This module is plain PyTorch: :func:`forward_reference` and :func:`local_backward` run on any
device and dtype and are the references the kernels (:mod:`hsn.kernels.event`) are tested
against; in float64 the block equals the autograd event loop to rounding
(``tests/test_wev_fused_event.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd.function import FunctionCtx, once_differentiable

from hsn.model.cell import WindowConsts
from hsn.model.hsn import HSN, _Controls
from hsn.train.state_gradient import (
    CellConstants,
    activation_grad,
    cell_constants,
    state_gradient_backward,
)

__all__ = [
    "TAPE_KEYS",
    "EventRecurrence",
    "EventRun",
    "EventSpec",
    "EventWeights",
    "event_spec",
    "event_weights",
    "forward_reference",
    "fused_event_block",
    "local_backward",
    "round_reference",
    "scope_error",
    "sweep_reference",
]

#: What a forward sweep writes for a block of ``E`` slots, ``[B, E, …]`` (``P`` is the products'
#: dtype, bf16 under autocast; the others the state's dtype unless said):
#: the reverse sweep's record ``c``, ``b``, ``h_old``, ``y``, ``gate_pre``, ``m`` ``[B, E, K, d]``,
#: ``rms [B, E, K]``, ``hidden [B, E, K, H]`` (P), ``cells`` / ``units`` (int64) and ``valid``
#: (bool: a slot that was written, so never one of a padding row) ``[B, E, K]``;
#: the local backward's ``v [B, E, K, n]`` (P), ``u``, ``eps``, ``z_hat`` ``[B, E, d]``,
#: ``h_pre [B, E, N, d]`` and ``age_pre [B, E, N]`` (the state the slot read, after its reset),
#: ``soft`` / ``prob [B, E, K]`` (fp32: the sigmoid of a slot's cell score against the
#: threshold and the softmax weight of its unit, whose derivatives are the straight-through
#: path), ``ties [B, E, K, M]`` (bool: the units whose score is the cell's largest; one unless
#: scores are equal) and ``theta [B, E]``;
#: the node's outputs ``q_pre``, ``q_post``, ``readout`` ``[B, E, d]``, ``logits [B, E, N, M]``;
#: the records ``sel_cells`` / ``sel_units`` / ``sel_valid`` ``[B, E, K]`` (the decision as the
#: router returned it, padding rows included), ``cost [B, E]``, ``refreshed [E]``,
#: ``abs_max [E]``; and the state after the block, ``h``, ``q``, ``age``.
TAPE_KEYS: tuple[str, ...] = (
    "c", "b", "h_old", "y", "gate_pre", "m", "rms", "hidden", "cells", "units", "valid",
    "v", "u", "eps", "z_hat", "h_pre", "age_pre", "soft", "prob", "ties", "theta",
    "q_pre", "q_post", "readout", "logits",
    "sel_cells", "sel_units", "sel_valid", "cost", "refreshed", "abs_max",
)  # fmt: skip


class EventSpec(NamedTuple):
    """The sizes and constants of a model's event round (:func:`event_spec`)."""

    N: int
    K: int
    M: int
    d: int
    H: int
    in_dim: int  # the cell input's width n (4d)
    D: int  # the padded in-degree of the message graph
    A: int  # address features of the scorer (0 without model.key_address)
    age_max: int
    tau: float
    norm_eps: float
    cand_eps: float
    unit_act: str
    front_act: str
    front_s: int  # S's hidden units: the first block of the merged front's hidden layer
    gate_input: str  # "none" | "summary" | "u": what A12's address gate reads
    probes: bool
    budget: float
    k_min: int
    lambda_cost: float
    pad_src: Tensor  # int64 [N, D]


class EventWeights(NamedTuple):
    """The weights an event round reads, detached, in the state's dtype
    (:func:`event_weights`), in the layouts the forward sweep multiplies with.

    The front is ``hsn.model.hsn.HSN.front_layout``'s two-layer map of ``[z; q; ctx; q]`` with
    its first layer split by what changes inside a block: ``hid = [z; ctx] · front_staticᵀ +
    q · front_qᵀ + front_b1`` (``S``'s and ``P``'s read of the summary are one value, so their
    columns are added), ``[u; ẑ] = act(hid) · front_w2ᵀ + front_b2``. The unit bank is
    :meth:`hsn.model.units.UnitBank.merged_layout`'s (the gate riding the first layer, ``W_out``
    folded into the second)."""

    front_static: Tensor  # [Hf, d + d_ctx]
    front_q: Tensor  # [Hf, d]
    front_b1: Tensor  # [Hf]
    front_w2: Tensor  # [2d, Hf]
    front_b2: Tensor  # [2d]
    score_w: Tensor  # [M, 3d + 8 + A]
    score_b: Tensor  # [M]
    gate_w: Tensor | None  # [M·A, d]
    v: Tensor  # [d, d]: V h = h @ v.T
    edge: Tensor  # [N, D]
    norm_gain: Tensor  # [n]
    w1: Tensor  # [n, M·H + d]
    b1: Tensor  # [M·H + d]
    w2: Tensor  # [H, M·d]
    b2: Tensor  # [M·d]
    cand_gain: Tensor  # [d]


def scope_error(model: HSN, controls: _Controls | None = None) -> str | None:
    """Why the fused event block cannot run ``model``'s event round, or ``None`` when it can.
    ``controls``: the step's routing controls (``None``: the model's own)."""
    cfg = model.cfg
    cell, scorer, router = model.cell, model.scorer, model.router
    controls = model._controls() if controls is None else controls
    if cfg.write_events == "byte":
        return "model.write_events=byte: there is no event loop"
    if cfg.routing != "topk":
        return f"model.routing={cfg.routing!r}: the fused event block needs topk"
    if controls.gate != "ste":
        return f"the gate of this step is {controls.gate!r}: the fused event block needs ste"
    if controls.epsilon > 0.0:
        return "exploration (train.epsilon_*) is not built into the fused event block"
    if router.unit_credit != "executed":
        return f"train.unit_credit={router.unit_credit!r}: the fused event block needs executed"
    if not (model._cut_summary and model._cut_eps and model._cut_router):
        return "train.feedback_detach must hold summary, eps and router"
    if cell.detach_state_read:
        return "train.feedback_detach holds state: the full cut is the split window's scope"
    if model._clip_h is not None or model.state_grad_ratio is not None:
        return "train.state_grad_clip / state_grad_ratio: the per-row limit is not built"
    if not isinstance(cell.norm, nn.RMSNorm) or not isinstance(cell.cand_norm, nn.RMSNorm):
        return "model.cell_norm and model.cand_norm must both be rms"
    if cell.ablate_shared_u or cell.disabled_cells:
        return "model.ablate_shared_u and model.disabled_cells are not built"
    if cfg.r_max != 1 or cfg.fast_cells or cfg.fast_readout or cfg.write_gate:
        return "model.r_max > 1, fast cells (A8) and the write gate (A10) are not built"
    if not cfg.selected_readout:
        return "model.selected_readout=false is not built"
    if model.cell_read is not None:
        return "model.cell_attention_heads: the attention read of the cells is not built"
    if scorer.kind != "linear" or scorer.exclude:
        return "the fused event block needs the linear scorer without excluded features"
    if scorer.address_gate is not None and scorer.gate_input not in ("u", "summary"):
        return f"model.address_gate_input={scorer.gate_input!r}: only u and summary are built"
    if router.closed_units is not None:
        return "model.cell_units=fixed is not built"
    if model.front_layout() is None:
        return "P and S do not share an activation: the merged front is not available"
    if model.training and model.schedule.p_dense(model.train_step) > 0.0:
        return "train.p_dense > 0: dense-fallback rows are not built"
    return None


def event_spec(model: HSN, controls: _Controls | None = None) -> EventSpec:
    """The :class:`EventSpec` of ``model`` at this step's ``controls``."""
    cfg, cell, scorer, router = model.cfg, model.cell, model.scorer, model.router
    controls = model._controls() if controls is None else controls
    assert isinstance(cell.norm, nn.RMSNorm) and isinstance(cell.cand_norm, nn.RMSNorm)
    dtype = model.dtype
    norm_eps = cell.norm.eps if cell.norm.eps is not None else torch.finfo(dtype).eps
    cand_eps = cell.cand_norm.eps if cell.cand_norm.eps is not None else torch.finfo(dtype).eps
    gate_input = "none" if scorer.address_gate is None else scorer.gate_input
    return EventSpec(
        N=cfg.n_cells,
        K=cfg.k_select,
        M=cfg.n_units,
        d=cfg.d,
        H=cell.units.hidden,
        in_dim=cell.in_dim,
        D=cell.graph.max_in_degree,
        A=scorer.in_features - scorer._address_start,
        age_max=scorer.age_max,
        tau=float(controls.tau),
        norm_eps=float(norm_eps),
        cand_eps=float(cand_eps),
        unit_act=cell.units.activation_name,
        front_act=model.shared.activation_name,
        front_s=model.shared.hidden,
        gate_input=gate_input,
        probes=bool(controls.probes),
        budget=float(cfg.budget),
        k_min=cfg.k_min,
        lambda_cost=float(router.lambda_cost),
        pad_src=cell.graph._pad_src,
    )


def event_weights(
    model: HSN,
    consts: WindowConsts,
    dtype: torch.dtype,
    product_dtype: torch.dtype | None = None,
) -> EventWeights:
    """The :class:`EventWeights` of this window. ``consts`` are the merged window constants
    (:meth:`HSN.merged_consts`, inside the cell's window). ``product_dtype``: the weights of
    the matrix products hold the values the products read, rounded to that dtype as autocast
    casts them at every use (the norms' gains and the edge weights are read in the state's
    dtype and stay as they are); ``None``: no rounding."""
    if consts.w1_gate is None or consts.b1_gate is None:
        raise ValueError("the fused event block needs the merged constants (HSN.merged_consts)")
    if consts.fold_w2 is None or consts.fold_b2 is None:
        raise ValueError("the fused event block needs the merged constants (HSN.merged_consts)")
    layout = model.front_layout()
    if layout is None:
        raise ValueError("P and S do not share an activation: no merged front")
    cfg, cell, scorer = model.cfg, model.cell, model.scorer
    d = cfg.d
    in_s = model.shared.in_dim

    rp = _rounder(dtype, product_dtype)

    def const(x: Tensor) -> Tensor:
        return x.detach().to(dtype)

    def cast(x: Tensor) -> Tensor:
        return rp(const(x))

    w1, b1, w2, b2 = (cast(x) for x in layout)
    assert isinstance(scorer.net, nn.Linear)
    assert isinstance(cell.norm, nn.RMSNorm) and isinstance(cell.cand_norm, nn.RMSNorm)
    gate = scorer.address_gate
    return EventWeights(
        front_static=torch.cat([w1[:, :d], w1[:, 2 * d : in_s]], dim=1).contiguous(),
        front_q=(w1[:, d : 2 * d] + w1[:, in_s:]).contiguous(),
        front_b1=b1,
        front_w2=w2,
        front_b2=b2,
        score_w=cast(scorer.net.weight),
        score_b=cast(scorer.net.bias),
        gate_w=None if gate is None else cast(gate.weight),
        v=cast(cell.V.weight),
        edge=const(consts.edge_weights),
        norm_gain=const(cell.norm.weight),
        w1=cast(consts.w1_gate),
        b1=cast(consts.b1_gate),
        w2=cast(consts.fold_w2),
        b2=cast(consts.fold_b2),
        cand_gain=const(cell.cand_norm.weight),
    )


def _act(name: str, x: Tensor) -> Tensor:
    if name == "gelu":
        return F.gelu(x)
    if name == "relu":
        return F.relu(x)
    if name == "silu":
        return F.silu(x)
    if name == "tanh":
        return torch.tanh(x)
    raise ValueError(f"no activation {name!r}")


def _rounder(dtype: torch.dtype, product_dtype: torch.dtype | None) -> Callable[[Tensor], Tensor]:
    """``x`` as it comes back from the products' dtype: the identity when the products run in
    the state's dtype, else the value rounded to ``product_dtype`` (bf16 under autocast)."""
    if product_dtype is None or product_dtype == dtype:
        return lambda x: x
    return lambda x: x.to(product_dtype).to(dtype)


def round_reference(
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
    """One event slot without autograd, in plain PyTorch: the reference of the fused round's
    forward kernels. ``h [B, N, d]``, ``q [B, d]``, ``age [B, N]`` are the state before the slot;
    ``z [B, d]``, ``ctx [B, d_ctx]``, ``scalars [B, N, 7 + A]`` the slot's inputs, ``reset`` and
    ``live`` bool ``[B]``. Returns the slot's tape (:data:`TAPE_KEYS` without the leading ``E``)
    and the state after it, ``h``, ``q``, ``age`` (a padding row's is the state it had).

    ``product_dtype``: where the compiled round rounds to the products' dtype (the outputs of
    its matrix products and what they read) the values are rounded here too, so that under
    bf16 the reference and the kernels round at the same places; ``None``: no rounding."""
    dtype = h.dtype
    rp = _rounder(dtype, product_dtype)
    B, N, d = (int(n) for n in h.shape)
    K, M, H, A = spec.K, spec.M, spec.H, spec.A
    z = z.to(dtype)
    # the reset of the slot, then the front on the carried summary
    h_r = h.masked_fill(reset[:, None, None], 0)
    q_r = q.masked_fill(reset[:, None], 0)
    age_r = age.masked_fill(reset[:, None], 0)
    # (the first layer's part of [z; ctx] is one product for all slots on the card, so it is
    # rounded on its own)
    static = rp(rp(torch.cat([z, ctx.to(dtype)], dim=-1)) @ w.front_static.t() + w.front_b1)
    hid_f = rp(static + rp(q_r) @ w.front_q.t())
    out_f = rp(rp(_act(spec.front_act, hid_f)) @ w.front_w2.t() + w.front_b2)
    u, z_hat = out_f[:, :d], out_f[:, d:]
    eps = rp(z - z_hat)
    # the scorer on the features [u; eps; h_i; age_i / age_max; scalars_i], one sum per logit
    scal = scalars.to(dtype)
    row = u @ w.score_w[:, :d].t() + eps @ w.score_w[:, d : 2 * d].t() + w.score_b  # [B, M]
    own = rp(h_r) @ w.score_w[:, 2 * d : 3 * d].t()  # [B, N, M]
    age_f = rp(age_r.to(dtype) / spec.age_max)
    rest = age_f[:, :, None] * w.score_w[:, 3 * d] + rp(scal) @ w.score_w[:, 3 * d + 1 :].t()
    logits = rp(row[:, None, :] + own + rest)
    if w.gate_w is not None:
        read = q_r if spec.gate_input == "summary" else u
        gate = rp(rp(read) @ w.gate_w.t()).view(B, M, A)
        address = rp(scal[:, :, scal.shape[-1] - A :])
        logits = rp(logits + rp(torch.einsum("bna,bma->bnm", address, gate)))
    # the decision: the router's own code on these logits (no gradient is recorded here)
    sel = model.router.select(
        logits,
        budget=spec.budget,
        k_max=K,
        k_min=spec.k_min,
        mode="topk",
        gate="ste",
        temperature=spec.tau,
        epsilon=0.0,
        age=age_r if spec.probes else None,
    )
    assert sel.threshold is not None
    valid = sel.valid & live[:, None]
    cells = torch.where(valid, sel.cell_idx, torch.zeros_like(sel.cell_idx))
    units = torch.where(valid, sel.unit_idx, torch.zeros_like(sel.unit_idx))
    # the straight-through derivatives of the slot's weights (the router's scores are fp32)
    scores = logits.float()
    if spec.lambda_cost > 0.0:
        scores = scores - spec.lambda_cost * model.router.cost_table.float()
    # (the router's own expressions on its own shapes, then the slots' entries: a vectorised
    # exp may round an element by its place in the tensor)
    top_all = scores.amax(dim=-1)  # [B, N]
    soft_all = torch.sigmoid((top_all - sel.threshold[:, None]) / spec.tau)
    prob_all = torch.softmax(scores / spec.tau, dim=-1)  # [B, N, M]
    cell_scores = scores.gather(1, cells[:, :, None].expand(B, K, M))  # [B, K, M]
    ties = (cell_scores == top_all.gather(1, cells)[:, :, None]) & valid[:, :, None]
    soft = soft_all.gather(1, cells)
    prob = prob_all.gather(1, cells[:, :, None].expand(B, K, M)).gather(2, units[:, :, None])
    prob = prob.squeeze(2)
    # the messages, the cell input and its norm
    idx = cells[:, :, None].expand(B, K, d)
    vh = rp(rp(h_r) @ w.v.t())  # [B, N, d]
    src = spec.pad_src[cells].reshape(B, K * spec.D, 1).expand(B, K * spec.D, d)
    sources = vh.gather(1, src).reshape(B, K, spec.D, d)
    m = rp(rp(sources * rp(w.edge[cells])[..., None]).sum(dim=2))
    h_old = h_r.gather(1, idx)
    x = torch.cat([u[:, None].expand(B, K, d), eps[:, None].expand(B, K, d), h_old, m], dim=-1)
    rms = torch.rsqrt(x.square().mean(dim=-1) + spec.norm_eps)
    v = rp(x * rms[..., None] * w.norm_gain)
    # the unit bank: every unit's first layer and the gate in one product, the chosen unit's
    # second layer with W_out folded in
    first = rp(v.reshape(B * K, -1) @ w.w1 + w.b1)
    pick = units.reshape(B * K, 1)
    columns = pick * H + torch.arange(H, device=h.device)[None, :]
    hidden = first.gather(1, columns)
    gate_pre = first[:, M * H :]
    out = rp(rp(_act(spec.unit_act, hidden)) @ w.w2 + w.b2)
    y = out.gather(1, pick * d + torch.arange(d, device=h.device)[None, :]).reshape(B, K, d)
    b = torch.sigmoid(gate_pre.reshape(B, K, d))
    rho = torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + spec.cand_eps)
    c = torch.tanh(y * rho * w.cand_gain)
    ok = valid[:, :, None]
    h_new = torch.where(ok, (1 - b) * h_old + b * c, h_old)
    # the update: the written cells, the summary, the ages (a padding row keeps its state)
    safe = torch.where(valid, cells, torch.full_like(cells, N))[:, :, None].expand(B, K, d)
    wide = torch.cat([h_r, h_r.new_zeros(B, 1, d)], dim=1)
    h_out = wide.scatter(1, safe, h_new)[:, :N]
    q_out = q_r + (h_new - h_old).sum(dim=1) / N
    selected = torch.zeros(B, N + 1, dtype=torch.bool, device=h.device)
    selected = selected.scatter(1, safe[:, :, 0], True)[:, :N]
    age_out = age_r.masked_fill(selected, 0) + (~selected).to(age.dtype)
    n_valid = valid.sum(dim=1, keepdim=True)
    readout = torch.where(ok, h_new, torch.zeros_like(h_new)).sum(dim=1)
    readout = readout / n_valid.clamp_min(1).to(dtype)
    h_next = torch.where(live[:, None, None], h_out, h)
    q_next = torch.where(live[:, None], q_out, q)
    age_next = torch.where(live[:, None], age_out, age)
    return {
        "c": c,
        "b": b,
        "h_old": h_old,
        "y": y,
        "gate_pre": gate_pre.reshape(B, K, d),
        "m": m,
        "rms": rms,
        "hidden": hidden.reshape(B, K, H)
        if product_dtype is None
        else hidden.reshape(B, K, H).to(product_dtype),
        "cells": cells,
        "units": units,
        "valid": valid,
        "v": v if product_dtype is None else v.to(product_dtype),
        "u": u,
        "eps": eps,
        "z_hat": z_hat,
        "h_pre": h_r,
        "age_pre": age_r,
        "soft": soft,
        "prob": prob,
        "ties": ties,
        "theta": sel.threshold,
        "q_pre": q_r,
        "q_post": q_next,
        "readout": readout,
        "logits": logits,
        "sel_cells": sel.cell_idx,
        "sel_units": sel.unit_idx,
        "sel_valid": valid,
        "cost": sel.cost.detach(),
        "refreshed": sel.refreshed.float().sum(dim=1).mean(),
        "abs_max": h_next.abs().amax(),
        "h": h_next,
        "q": q_next,
        "age": age_next,
    }


@torch.no_grad()
def forward_reference(
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
    """The forward sweep of a block of ``E`` event slots in plain PyTorch
    (:func:`round_reference` at every slot): the tape (:data:`TAPE_KEYS`, stacked ``[B, E, …]``;
    the batch statistics ``[E]``) and the state after the block (``h``, ``q``, ``age``).
    ``z [B, E, d]``, ``ctx [B, E, d_ctx]``, ``scalars [B, E, N, 7 + A]``, ``reset`` and ``live``
    bool ``[B, E]``."""
    slots: list[dict[str, Tensor]] = []
    for t in range(int(z.shape[1])):
        slot = round_reference(
            model, w, spec, h, q, age, z[:, t], ctx[:, t], scalars[:, t], reset[:, t], live[:, t],
            product_dtype=product_dtype,
        )  # fmt: skip
        h, q, age = slot["h"], slot["q"], slot["age"]
        slots.append(slot)
    tape = {
        key: torch.stack([s[key] for s in slots], dim=1 if slots[0][key].ndim > 0 else 0)
        for key in TAPE_KEYS
    }
    tape.update(h=h, q=q, age=age, reset=reset)
    return tape


def _by_unit(units: Tensor, n_units: int) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
    """The rows of a block grouped by unit: ``(order, inverse, segments)`` with ``order`` the
    rows sorted by their unit (stable), ``inverse`` its inverse permutation and ``segments[j]``
    the range of unit ``j``'s rows in that order. One read of ``M`` counts from the device."""
    order = torch.argsort(units, stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    counts = torch.bincount(units, minlength=n_units).tolist()
    segments: list[tuple[int, int]] = []
    start = 0
    for count in counts:
        segments.append((start, start + int(count)))
        start += int(count)
    return order, inverse, segments


@torch.no_grad()
def local_backward(
    w: EventWeights,
    spec: EventSpec,
    tape: dict[str, Tensor],
    dc: Tensor,
    db: Tensor,
    d_readout: Tensor | None,
    d_logits: Tensor | None,
    scalars: Tensor,
    *,
    product_dtype: torch.dtype | None = None,
) -> dict[str, Tensor | None]:
    """The parameters' part of the block's backward pass, for all its slots at once and
    without autograd: from the total ``dL/dc`` and ``dL/db`` of every slot (``[B, E, K, d]``,
    the reverse sweep's result; exact zeros on padded slots), the cotangent of the read-outs
    (``[B, E, d]`` or ``None``) and of the logits (``[B, E, N, M]`` or ``None``: the balance
    term's) to the gradients of

    * the unit bank's merged layouts ``w1 [n, M·H + d]``, ``b1``, ``w2 [H, M·d]``, ``b2``
      (:meth:`hsn.model.units.UnitBank.merged_layout`: the gate rides the first, ``W_out`` is
      folded into the second; autograd takes them to the masters outside);
    * the two norms' gains, ``V`` and the padded edge weights ``[N, D]``;
    * the scorer's weight, bias and the address gate's weight (the straight-through path of
      every written slot: its cell mask and, with executed credit, its unit weight, and the
      read-out's weight on the cell mask);
    * ``u`` and the cell's error read ``eps`` of every slot, ``[B, E, d]``.

    The unit bank's products are taken per unit over that unit's rows of the whole block
    (``M`` products per layer and direction), not at every slot. ``product_dtype``: the dtype
    those products run in (bf16 on the card under autocast; ``None``: the state's)."""
    dtype = dc.dtype
    pd = dtype if product_dtype is None else product_dtype
    B, E, K, d = (int(n) for n in dc.shape)
    N, M, H, n, A = spec.N, spec.M, spec.H, spec.in_dim, spec.A
    P = B * E * K
    c, b, y = tape["c"], tape["b"], tape["y"]
    valid = tape["valid"]
    # the candidate: c = tanh(alpha · g_c ⊙ y · rho), alpha = 1 on the written slots
    dz = dc * (1.0 - c * c)
    rho = torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + spec.cand_eps)
    normed = y * rho
    d_alpha = (dz * (normed * w.cand_gain)).sum(dim=-1)  # [B, E, K]
    g_cand = (dz * normed).reshape(P, d).sum(dim=0)
    scaled = w.cand_gain * dz
    dy = (rho * scaled - rho.pow(3) * y * (y * scaled).mean(dim=-1, keepdim=True)).reshape(P, d)
    d_gate = (db * b * (1.0 - b)).reshape(P, d)
    # the unit bank, one unit at a time over its rows of the whole block
    order, inverse, segments = _by_unit(tape["units"].reshape(P), M)
    hid_s = tape["hidden"].reshape(P, H).index_select(0, order)
    act_s = _act(spec.unit_act, hid_s.to(dtype)).to(pd)
    dy_s = dy.index_select(0, order).to(pd)
    v_all = tape["v"].reshape(P, n)
    v_s = v_all.index_select(0, order).to(pd)
    w1_p, w2_p = w.w1.to(pd), w.w2.to(pd)
    g_w1 = w.w1.new_zeros(n, M * H + d)
    g_b1 = w.b1.new_zeros(M * H + d)
    g_w2 = w.w2.new_zeros(H, M * d)
    g_b2 = w.b2.new_zeros(M * d)
    d_act = act_s.new_zeros(P, H)
    for j, (lo, hi) in enumerate(segments):
        if hi > lo:
            g_w2[:, j * d : (j + 1) * d] = (act_s[lo:hi].t() @ dy_s[lo:hi]).to(dtype)
            g_b2[j * d : (j + 1) * d] = dy_s[lo:hi].to(dtype).sum(dim=0)
            d_act[lo:hi] = dy_s[lo:hi] @ w2_p[:, j * d : (j + 1) * d].t()
    d_hid = (d_act.to(dtype) * activation_grad(spec.unit_act, hid_s.to(dtype))).to(pd)
    dv_s = d_hid.new_zeros(P, n)
    for j, (lo, hi) in enumerate(segments):
        if hi > lo:
            g_w1[:, j * H : (j + 1) * H] = (v_s[lo:hi].t() @ d_hid[lo:hi]).to(dtype)
            g_b1[j * H : (j + 1) * H] = d_hid[lo:hi].to(dtype).sum(dim=0)
            dv_s[lo:hi] = d_hid[lo:hi] @ w1_p[:, j * H : (j + 1) * H].t()
    d_gate_p = d_gate.to(pd)
    v_p = v_all.to(pd)
    g_w1[:, M * H :] = (v_p.t() @ d_gate_p).to(dtype)
    g_b1[M * H :] = d_gate.sum(dim=0)
    dv = (dv_s.index_select(0, inverse) + d_gate_p @ w1_p[:, M * H :].t()).to(dtype)
    # the cell norm: v = g_n ⊙ x · r
    u, eps = tape["u"], tape["eps"]
    x = torch.cat(
        [
            u[:, :, None].expand(B, E, K, d),
            eps[:, :, None].expand(B, E, K, d),
            tape["h_old"],
            tape["m"],
        ],
        dim=-1,
    ).reshape(P, n)
    r = tape["rms"].reshape(P, 1)
    scaled_dv = w.norm_gain * dv
    inner = (x * scaled_dv).sum(dim=-1, keepdim=True)
    dx = r * scaled_dv - (r.pow(3) / n) * x * inner
    g_norm = (dv * x * r).sum(dim=0)
    dx = dx.reshape(B, E, K, n)
    d_u = dx[..., :d].sum(dim=2)
    d_eps = dx[..., d : 2 * d].sum(dim=2)
    dm = dx[..., 3 * d :].reshape(P, d)
    # the messages: m = Σ_e a_e · V h[src_e]
    cells = tape["cells"]
    h_pre = tape["h_pre"]  # [B, E, N, d]
    D = spec.D
    src = spec.pad_src[cells]  # [B, E, K, D]
    rows = torch.arange(B * E, device=dc.device).reshape(B, E, 1, 1) * N
    h_src = h_pre.reshape(B * E * N, d).index_select(0, (rows + src).reshape(-1)).reshape(P, D, d)
    a_sel = w.edge[cells].reshape(P, D)
    g_v = dm.t() @ (h_src * a_sel[:, :, None]).sum(dim=1)
    through = dm @ w.v  # [P, d]: Vᵀ dm
    per_edge = (h_src * through[:, None, :]).sum(dim=-1)  # [P, D]
    g_edge = (
        w.edge.new_zeros(N * D)
        .index_add(
            0,
            (cells.reshape(P, 1) * D + torch.arange(D, device=dc.device)[None, :]).reshape(-1),
            per_edge.reshape(-1),
        )
        .reshape(N, D)
    )
    # the straight-through path into the scores of the written slots, in fp32 and in autograd's
    # own order of operations (the router's scores are fp32 whatever the logits' dtype): the
    # cell mask from alpha and from the read-out's weight, through the sigmoid, to the cell's
    # largest score (shared evenly by equal scores, as amax's backward does); the unit weight
    # from alpha, through the softmax, to the executed unit's score alone (executed credit)
    soft, prob, ties = tape["soft"], tape["prob"], tape["ties"]
    g_alpha = d_alpha.float()
    d_mask = g_alpha
    if d_readout is not None:
        h_new = (1 - b) * tape["h_old"] + b * c
        n_valid = valid.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype)
        d_mask = d_mask + ((d_readout / n_valid)[:, :, None, :] * h_new).sum(dim=-1).float()
    d_cell = (d_mask * (1 - soft) * soft) / spec.tau  # sigmoid's backward: g · (1 − y) · y
    d_unit = (prob * (g_alpha - g_alpha * prob)) / spec.tau  # softmax's: y · (g − Σ g y)
    n_ties = ties.sum(dim=-1, keepdim=True).clamp_min(1)
    d_score = (d_cell[..., None] / n_ties) * ties
    d_score = d_score + F.one_hot(tape["units"], M) * d_unit[..., None]
    d_score = torch.where(valid[..., None], d_score, torch.zeros_like(d_score))  # [B, E, K, M]
    flat = (cells[..., None] * M + torch.arange(M, device=dc.device)).reshape(B * E, K * M)
    d_log = dc.new_zeros(B * E, N * M).scatter_add(1, flat, d_score.reshape(B * E, K * M).to(dtype))
    d_log = d_log.reshape(B, E, N, M)
    if d_logits is not None:
        d_log = d_log + d_logits.to(dtype)
    # the scorer: logits = feats · Wᵀ + bias (+ A12's gate), the features constants (router cut)
    g_sw = w.score_w.new_zeros(w.score_w.shape)
    per_row = d_log.sum(dim=2).reshape(B * E, M)  # [B·E, M]
    g_sw[:, :d] = per_row.t() @ u.reshape(B * E, d)
    g_sw[:, d : 2 * d] = per_row.t() @ eps.reshape(B * E, d)
    flat_log = d_log.reshape(B * E * N, M)
    g_sw[:, 2 * d : 3 * d] = flat_log.t() @ h_pre.reshape(B * E * N, d)
    age_f = (tape["age_pre"].to(dtype) / spec.age_max).reshape(B * E * N)
    g_sw[:, 3 * d] = flat_log.t() @ age_f
    scal = scalars.to(dtype).reshape(B * E * N, -1)
    g_sw[:, 3 * d + 1 :] = flat_log.t() @ scal
    g_sb = flat_log.sum(dim=0)
    g_gate: Tensor | None = None
    if w.gate_w is not None:
        read = tape["q_pre"] if spec.gate_input == "summary" else u
        address = scal[:, scal.shape[-1] - A :].reshape(B * E, N, A)
        d_gate_rows = torch.einsum("rnm,rna->rma", d_log.reshape(B * E, N, M), address)
        g_gate = d_gate_rows.reshape(B * E, M * A).t() @ read.reshape(B * E, d)
    return {
        "w1": g_w1,
        "b1": g_b1,
        "w2": g_w2,
        "b2": g_b2,
        "edge": g_edge,
        "norm_gain": g_norm,
        "cand_gain": g_cand,
        "v": g_v,
        "score_w": g_sw,
        "score_b": g_sb,
        "gate_w": g_gate,
        "u": d_u,
        "eps": d_eps,
    }


#: A block's reverse sweep: ``(tape, consts, d_q_pre, d_q_post, d_h_sel, initial) -> (dc, db,
#: dh0, dq0)``; the cotangents may be ``None``.
Sweep = Callable[..., tuple[Tensor, Tensor, Tensor, Tensor]]


def sweep_reference(
    tape: dict[str, Tensor],
    consts: CellConstants,
    n_cells: int,
    d_q_pre: Tensor | None,
    d_q_post: Tensor | None,
    d_h_sel: Tensor | None,
    initial: tuple[Tensor, Tensor] | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The block's reverse sweep in plain PyTorch
    (:func:`hsn.train.state_gradient.state_gradient_backward` on the tape): ``(dc, db, dh0,
    dq0)``, the total cotangents of every slot's candidate and gate and the cotangent of the
    state the block started from. ``initial``: the cotangent of the state after the block."""
    c = tape["c"]
    dtype = c.dtype
    record = {
        "y": tape["y"],
        "alpha": tape["valid"].to(dtype),
        "hidden": tape["hidden"].to(dtype),
        "gate_pre": tape["gate_pre"],
        "m": tape["m"],
        "rms": tape["rms"],
    }
    trace: dict[str, Tensor] = {}
    reset = tape["reset"]
    dc, db = state_gradient_backward(
        c, tape["b"], tape["h_old"], tape["cells"], tape["units"], tape["valid"], reset,
        n_cells, d_q_pre, d_q_post, d_h_sel, None, record, consts, trace=trace, initial=initial,
    )  # fmt: skip
    # the cotangent of the state slot 0 read, after its reset: zero where slot 0 reset
    first = reset[:, 0]
    dh0 = trace["A"][:, 0].masked_fill(first[:, None, None], 0)
    dq0 = trace["R"][:, 0].masked_fill(first[:, None], 0)
    return dc, db, dh0, dq0


class EventRun:
    """What the node of one block keeps for its backward pass: the tape, the weights and sizes
    of the window, the slots' scalars, the cell's constants for the reverse sweep, and the
    sweep and the products' dtype to run it with."""

    def __init__(
        self,
        tape: dict[str, Tensor],
        weights: EventWeights,
        spec: EventSpec,
        scalars: Tensor,
        consts: CellConstants,
        *,
        sweep: Sweep | None = None,
        product_dtype: torch.dtype | None = None,
    ) -> None:
        self.tape = tape
        self.weights = weights
        self.spec = spec
        self.scalars = scalars
        self.consts = consts
        self.sweep = sweep
        self.product_dtype = product_dtype
        #: Filled by the backward pass (tests and measurements read them).
        self.dc: Tensor | None = None
        self.db: Tensor | None = None


class EventRecurrence(torch.autograd.Function):
    """A block of event slots as one autograd node (module docstring). Forward returns the
    taped values: the summary before and after every slot (``[B, E, d]``; before: after the
    slot's reset, what ``P`` reads), the selected read-outs, the logits, and the state after
    the block. Backward runs the reverse sweep and :func:`local_backward` and returns the
    gradients of ``u`` and of the cell's error read (``[B, E, d]``), of the start state ``h``
    and ``q``, and of the eleven parameter tensors in the order of the arguments."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        u: Tensor,
        eps: Tensor,
        h: Tensor,
        q: Tensor,
        w1: Tensor,
        b1: Tensor,
        w2: Tensor,
        b2: Tensor,
        edge: Tensor,
        norm_gain: Tensor,
        cand_gain: Tensor,
        v: Tensor,
        score_w: Tensor,
        score_b: Tensor,
        gate_w: Tensor | None,
        run: EventRun,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        ctx.run = run  # type: ignore[attr-defined]
        ctx.has_gate = gate_w is not None  # type: ignore[attr-defined]
        ctx.set_materialize_grads(False)  # a read without a gradient arrives as None
        tape = run.tape
        return (
            tape["q_pre"].clone(),
            tape["q_post"].clone(),
            tape["readout"].clone(),
            tape["logits"].clone(),
            tape["h"].clone(),
            tape["q"].clone(),
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        d_q_pre: Tensor | None,
        d_q_post: Tensor | None,
        d_readout: Tensor | None,
        d_logits: Tensor | None,
        d_h: Tensor | None,
        d_q: Tensor | None,
    ) -> tuple[Any, ...]:
        run: EventRun = ctx.run  # type: ignore[attr-defined]
        tape, spec = run.tape, run.spec
        valid = tape["valid"]
        dtype = tape["c"].dtype
        d_h_sel: Tensor | None = None
        if d_readout is not None:
            n_valid = valid.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype)
            share = (d_readout / n_valid)[:, :, None, :].expand(*valid.shape, spec.d)
            d_h_sel = torch.where(valid[..., None], share, torch.zeros_like(share))
        initial: tuple[Tensor, Tensor] | None = None
        if d_h is not None or d_q is not None:
            initial = (
                torch.zeros_like(tape["h"]) if d_h is None else d_h,
                torch.zeros_like(tape["q"]) if d_q is None else d_q,
            )
        sweep = sweep_reference if run.sweep is None else run.sweep
        dc, db, dh0, dq0 = sweep(tape, run.consts, spec.N, d_q_pre, d_q_post, d_h_sel, initial)
        run.dc, run.db = dc, db
        grads = local_backward(
            run.weights, spec, tape, dc, db, d_readout, d_logits, run.scalars,
            product_dtype=run.product_dtype,
        )  # fmt: skip
        return (
            grads["u"],
            grads["eps"],
            dh0,
            dq0,
            grads["w1"],
            grads["b1"],
            grads["w2"],
            grads["b2"],
            grads["edge"],
            grads["norm_gain"],
            grads["cand_gain"],
            grads["v"],
            grads["score_w"],
            grads["score_b"],
            grads["gate_w"] if ctx.has_gate else None,  # type: ignore[attr-defined]
            None,
        )


#: A block's forward sweep: :func:`forward_reference`'s arguments and result.
Forward = Callable[..., dict[str, Tensor]]


def fused_event_block(
    model: HSN,
    h: Tensor,
    q: Tensor,
    age: Tensor,
    z_b: Tensor,
    c_b: Tensor,
    s_b: Tensor,
    r_b: Tensor,
    v_b: Tensor,
    consts: WindowConsts,
    controls: _Controls | None = None,
    *,
    forward: Forward | None = None,
    sweep: Sweep | None = None,
    product_dtype: torch.dtype | None = None,
    runs: list[EventRun] | None = None,
) -> dict[str, Tensor]:
    """:meth:`hsn.model.hsn.HSN._event_block` for a block of event slots, fused: the same
    arguments (the state before the block, the slots' inputs ``z_b`` / ``c_b`` / ``s_b`` ``[B,
    E, …]``, their resets ``r_b`` and live rows ``v_b``) and the same result (the state after
    the block and the per-slot stacks with their gradient), from the forward sweep, the node
    and the replay of ``S`` and ``P`` around it (module docstring). ``consts``: the merged
    window constants (:meth:`HSN.merged_consts`), inside the cell's window.

    ``forward`` / ``sweep``: the forward and the reverse sweep to run (``None``: the PyTorch
    references; on the card :class:`hsn.kernels.event.EventKernels`). ``product_dtype``: the
    dtype of the matrix products (``None``: the state's). ``runs`` receives the block's
    :class:`EventRun` (tests and measurements)."""
    controls = model._controls() if controls is None else controls
    reason = scope_error(model, controls)
    if reason is not None:
        raise ValueError(f"the fused event block does not take this model: {reason}")
    cell, scorer = model.cell, model.scorer
    dtype = h.dtype
    spec = event_spec(model, controls)
    weights = event_weights(model, consts, dtype, product_dtype)
    run_forward = forward_reference if forward is None else forward
    with torch.no_grad():
        tape = run_forward(
            model, weights, spec, h.detach(), q.detach(), age, z_b.detach(), c_b.detach(), s_b,
            r_b, v_b, product_dtype=product_dtype,
        )  # fmt: skip
    sweep_consts = cell_constants(cell, dtype)  # the window's edge weights: in its grad mode
    run = EventRun(tape, weights, spec, s_b, sweep_consts, sweep=sweep, product_dtype=product_dtype)
    # S on the taped summary (a constant under the summary cut): the local half of u, with
    # its gradient into S, the encoder and the reader
    u = model.shared(model._shared_input(z_b, tape["q_pre"], c_b, h))
    # the cell's and the scorer's read of the error: z keeps its gradient, the prediction none
    eps_read = (z_b - tape["z_hat"].to(z_b.dtype)).to(dtype)
    assert isinstance(scorer.net, nn.Linear)
    gate = scorer.address_gate
    q_pre, q_post, readouts, logits, h_end, q_end = EventRecurrence.apply(
        u.to(dtype),
        eps_read,
        h,
        q,
        consts.w1_gate,
        consts.b1_gate,
        consts.fold_w2,
        consts.fold_b2,
        consts.edge_weights,
        cell.norm.weight,
        cell.cand_norm.weight,
        cell.V.weight,
        scorer.net.weight,
        scorer.net.bias,
        None if gate is None else gate.weight,
        run,
    )
    # P reads the carried summary with its gradient, behind the node
    z_hat = model.predictor(q_pre)
    E = int(z_b.shape[1])
    zeros = tape["cost"].new_zeros(E)
    if runs is not None:
        runs.append(run)
    return {
        "h": h_end,
        "q": q_end,
        "age": tape["age"],
        "z_hat": z_hat,
        "eps": z_b - z_hat,
        "u": u,
        "qs": q_post,
        "state_abs_max": tape["abs_max"],
        "unit_logits_d": tape["logits"],
        "readouts": readouts,
        "sel_cell_idx": tape["sel_cells"],
        "sel_unit_idx": tape["sel_units"],
        "sel_valid": tape["sel_valid"],
        "sel_unit_logits": logits,
        "sel_cost_per_row": tape["cost"],
        "sel_expected_cost": tape["cost"].mean(dim=0),
        "sel_route_log_prob": torch.zeros_like(tape["cost"]),
        "sel_explored": zeros,
        "sel_refreshed": tape["refreshed"],
        "sel_dense_fallback": zeros,
    }


def forward_events(
    model: HSN,
    tokens: Tensor,
    ctx: Tensor,
    state: Any,
    reset: Tensor,
    prev_tokens: Any = None,
    *,
    forward: Forward | None = None,
    sweep: Sweep | None = None,
    product_dtype: torch.dtype | None = None,
    runs: list[EventRun] | None = None,
) -> tuple[Tensor, Any, dict[str, Tensor]]:
    """:meth:`hsn.model.hsn.HSN.forward` for a window of ``model.write_events`` with the event
    loop fused: the model's own prologue, event slots and epilogue around one
    :func:`fused_event_block` over all the window's slots. Returns ``(logits, state, info)`` as
    the model does."""
    from hsn.model.hsn import event_block

    mode = model._window_mode()
    controls = model._controls()
    pro = model._window_prologue(
        tokens, ctx, state, reset, mode, reset_first=True, prev_tokens=prev_tokens
    )
    ev = model._event_slots(pro, reset, event_block(int(tokens.shape[1])))
    with model.cell.window():
        consts = model.merged_consts()
        out = fused_event_block(
            model, state.h, state.q, state.age, ev.z, ev.context, ev.scalars, ev.reset, ev.valid,
            consts, controls, forward=forward, sweep=sweep, product_dtype=product_dtype,
            runs=runs,
        )  # fmt: skip
    return model._event_epilogue(pro, ev, state, [out], out["h"], out["q"], out["age"])
