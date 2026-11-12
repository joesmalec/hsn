"""The L0 reference implementations of the four HSN ops (P2.5): plain PyTorch, cpu / cuda / mps.

These four functions are the exact boundaries at which the fused kernels of P2.9–P2.11 replace the
native path (plan P2.5 "Context you need"), so each stays a separate function with the chunk's
signature and :meth:`hsn.model.cell.PDGCell.step_sparse` composes them through
:func:`hsn.kernels.ops`. For ``K`` selected cells per batch row (``P = B·K`` pairs) and the previous
round's states ``h [B, N, d]`` (plan §2.3)::

    m_i  = Σ_{j ∈ N(i)} a_ij · V · h_j                  csr_messages       (P1.4 messages_selected)
    v_i  = [u_t ; ε_t ; h_i ; m_i]                      gather_concat
    F_j(v_i)  for the one unit j selected for cell i    grouped_unit_mlp   (P1.2 forward_selected)
    h_i' = (1 − b_i) ⊙ h_i + b_i ⊙ c_i   for selected i gated_delta_update
    q    = q + Σ_selected (h_i' − h_i) / N              (incremental summary, plan §2.4)

Every index operation is a ``gather`` or an out-of-place ``scatter`` with a tensor source: no
in-place write touches a tensor that may require grad (the chunk's pitfall), no
``scatter_add``/``index_put`` accumulation (order-dependent with duplicate indices, and refused by
MPS under ``torch.use_deterministic_algorithms(True)``), and the same ops run on every backend.
Padded selection slots (``valid=False``; the P2.1 router pads with cell 0, unit 0, α 0) are ignored
by :func:`gated_delta_update`: they are redirected to a dummy row that is sliced away, so a padded
cell 0 never collides with a genuinely selected cell 0, contributes exactly zero to ``q`` and
receives zero gradient. ``gather``'s backward is a scatter-add on cpu and cuda, which torch's
deterministic mode supports; on MPS the deterministic flag refuses it (``scatter_reduce_mps``), as
it already refuses the embedding backward, so MPS with ``train.deterministic=true`` stays
unsupported.

Dtypes follow §3.7: the state is the fp32 master and ``u``, ``ε``, ``h`` share its dtype; under
autocast the gate ``b``, the candidate ``c`` and the unit outputs come back in the autocast dtype
(fp16 on MPS, bf16 on CUDA), exactly as on the dense path. Messages follow the reduction policy:
CUDA autocast promotes their final sum to fp32. :func:`gather_concat`
casts the messages to the state dtype explicitly (the promotion ``torch.cat`` applies on the dense
path) and :func:`gated_delta_update` evaluates the gated expression with the dense path's operand
dtypes and casts the result to the state dtype before the scatter, so ``h``, ``q`` and ``v`` always
hold the state dtype. The P2.9 ABI's single tensor dtype ``T`` per call is the state dtype without
autocast; a fused kernel under autocast receives the autocast-dtype operands these signatures
document.

Host synchronisations: :func:`gated_delta_update` validates the selection once per call (range and
distinctness among valid slots, as :meth:`CellGraphState.with_delta` does) unless
``validate=False``; :func:`csr_messages` inherits the range check of ``messages_selected`` and
:func:`grouped_unit_mlp` the unit histogram of ``forward_selected``. :func:`gather_concat` never
synchronises: it relies on the backend's own bounds check of ``gather`` (cpu and cuda raise; MPS
does not), which is why the composed round runs ``csr_messages`` on the same indices first. P2.8's
capturable round turns them off (``csr_messages(..., validate=False)``,
``grouped_unit_mlp(..., static=True)``, ``gated_delta_update(..., validate=False)``), since its
indices come from :meth:`hsn.model.router.Router.select`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hsn.model.graph import CellGraph
from hsn.model.state import CellGraphState
from hsn.model.units import UnitBank

__all__ = [
    "csr_messages",
    "gated_delta_update",
    "gather_concat",
    "grouped_unit_mlp",
    "selection_mask",
]

_INDEX_DTYPE = torch.int64


# --------------------------------------------------------------------------------------------------
# Validation helpers (shape, dtype and device only; no device synchronisation)
# --------------------------------------------------------------------------------------------------


def _check_tensor(name: str, x: object) -> Tensor:
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    return x


def _check_shape(name: str, x: Tensor, shape: tuple[int, ...]) -> None:
    if tuple(x.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}, got {list(x.shape)}")


def _check_device(name: str, x: Tensor, device: torch.device) -> None:
    if x.device != device:
        raise ValueError(f"{name} must live on {device}, got {x.device}")


def _check_floating(name: str, x: object, shape: tuple[int, ...], device: torch.device) -> Tensor:
    tensor = _check_tensor(name, x)
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be floating point, got {tensor.dtype}")
    _check_shape(name, tensor, shape)
    _check_device(name, tensor, device)
    return tensor


def _check_bool(name: str, x: object, shape: tuple[int, ...], device: torch.device) -> Tensor:
    tensor = _check_tensor(name, x)
    if tensor.dtype != torch.bool:
        raise ValueError(f"{name} must have dtype torch.bool, got {tensor.dtype}")
    _check_shape(name, tensor, shape)
    _check_device(name, tensor, device)
    return tensor


# --------------------------------------------------------------------------------------------------
# The four ops
# --------------------------------------------------------------------------------------------------


def gather_concat(u: Tensor | None, eps: Tensor, h: Tensor, m: Tensor, cell_idx: Tensor) -> Tensor:
    """``v [B, K, 4d] = [u ; ε ; h[cell_idx] ; m]`` for the selected cells.

    ``u``, ``eps`` are ``[B, d]`` in the dtype of ``h [B, N, d]`` (the state dtype); ``m [B, K, d]``
    is any floating dtype on the same device (the autocast dtype under autocast) and is cast to
    ``h``'s; ``cell_idx [B, K]`` is int64 with ``K >= 1``. The selected rows of ``h`` come from one
    ``gather`` (deterministic on every backend; its backward is an index scatter-add). No device
    synchronisation: out-of-range indices are the backend's error (cpu and cuda raise), see the
    module docstring.

    ``u=None`` is the cell input of ``model.ablate_shared_u`` (:mod:`hsn.model.cell`): ``v
    [B, K, 3d] = [ε ; h[cell_idx] ; m]``, the same operations without the first block.
    """
    h_t = _check_tensor("h", h)
    if h_t.dim() != 3 or min(h_t.shape) < 1 or not h_t.is_floating_point():
        raise ValueError(
            f"h must be a floating [B, N, d] tensor with every dimension >= 1, got shape "
            f"{list(h_t.shape)} and dtype {h_t.dtype}"
        )
    B, _, d = (int(n) for n in h_t.shape)
    device = h_t.device
    for name, x in (("eps", eps),) if u is None else (("u", u), ("eps", eps)):
        vec = _check_floating(name, x, (B, d), device)
        if vec.dtype != h_t.dtype:
            raise ValueError(f"{name} must have dtype {h_t.dtype} like h, got {vec.dtype}")
    idx = _check_tensor("cell_idx", cell_idx)
    if idx.dtype != _INDEX_DTYPE:
        raise ValueError(f"cell_idx must have dtype torch.int64, got {idx.dtype}")
    if idx.dim() != 2 or idx.shape[0] != B or idx.shape[1] < 1:
        raise ValueError(f"cell_idx must have shape [B, K] = [{B}, >= 1], got {list(idx.shape)}")
    _check_device("cell_idx", idx, device)
    K = int(idx.shape[1])
    msg = _check_floating("m", m, (B, K, d), device)
    h_sel = h_t.gather(1, idx.unsqueeze(-1).expand(B, K, d))
    if u is None:  # model.ablate_shared_u: the cell input without the shared path's u
        return torch.cat([eps.unsqueeze(1).expand(B, K, d), h_sel, msg.to(h_t.dtype)], dim=-1)
    return torch.cat(
        [
            u.unsqueeze(1).expand(B, K, d),
            eps.unsqueeze(1).expand(B, K, d),
            h_sel,
            msg.to(h_t.dtype),
        ],
        dim=-1,
    )


def grouped_unit_mlp(
    v: Tensor,
    unit_idx: Tensor,
    bank: UnitBank,
    *,
    static: bool = False,
    w1_flat: Tensor | None = None,
) -> Tensor:
    """``out [P, d]``: ``out[p] = F_{unit_idx[p]}(v[p])`` through the P1.2 grouped pass
    (:meth:`UnitBank.forward_selected`, which sorts the pairs by unit, runs one matmul pair per
    present unit and restores the order; values outside ``[0, M)`` raise ``IndexError`` there).
    ``v`` is ``[P, 4d]`` (the bank's ``in_dim``); ``unit_idx [P]`` int64 (or int32). The output
    has the autocast dtype under
    autocast and ``v``'s dtype otherwise. ``static=True`` (P2.8, the capturable round) runs
    :meth:`UnitBank.forward_static` instead (every unit on the pairs, then a gather: static
    shapes, no host-side histogram, indices not checked), with the window's ``w1_flat`` when
    given; it agrees with the grouped pass to rounding. CUDA autocast rounds the activation
    and intermediate gradients in reduced precision; the CUDA adapter retains this native
    path because the raw single-dtype ABI's fp32 intermediates do not match that contract."""
    if not isinstance(bank, UnitBank):
        raise TypeError(f"bank must be a UnitBank, got {type(bank).__name__}")
    v_t = _check_tensor("v", v)
    idx = _check_tensor("unit_idx", unit_idx)
    if v_t.dim() != 2:
        raise ValueError(f"v must have shape [P, 4d], got {list(v_t.shape)}")
    if idx.dim() != 1 or idx.shape[0] != v_t.shape[0]:
        raise ValueError(f"unit_idx must have shape [P] = [{v_t.shape[0]}], got {list(idx.shape)}")
    _check_device("unit_idx", idx, v_t.device)
    if bank.W1.device != v_t.device:
        raise ValueError(
            f"v lives on {v_t.device} but the unit bank on {bank.W1.device}; move the module"
        )
    if static:
        return bank.forward_static(v_t, idx, w1_flat=w1_flat)
    if w1_flat is not None:
        raise ValueError("w1_flat is only used by the static pass (static=True)")
    return bank.forward_selected(v_t, idx)


def csr_messages(
    h: Tensor,
    graph: CellGraph,
    V: nn.Linear,
    cell_idx: Tensor,
    *,
    weights: Tensor | None = None,
    validate: bool = True,
) -> Tensor:
    """``m [B, K, d]``: the neighbour messages of the selected cells from the **pre-round** states
    ``h [B, N, d]`` (the chunk's pitfall: never the states already updated in this round), through
    the P1.4 CSR path (:meth:`CellGraph.messages_selected`: ``V`` once over all cells, a padded
    gather of each selected cell's sources, weights ``a_ij``). ``cell_idx [B, K]`` is int64 in
    ``[0, N)`` (checked there, one device synchronisation, unless ``validate=False``: P2.8's
    capturable round). ``weights`` are a compiled window's padded edge weights (P2.8). The
    messages have ``V(h)``'s dtype except that CUDA autocast promotes their sum to fp32."""
    if not isinstance(graph, CellGraph):
        raise TypeError(f"graph must be a CellGraph, got {type(graph).__name__}")
    if not isinstance(V, nn.Linear):
        raise TypeError(
            f"V must be an nn.Linear (the d × d message projection), got {type(V).__name__}"
        )
    if weights is None and validate:
        return graph.messages_selected(h, V, cell_idx)
    return graph.messages_selected(h, V, cell_idx, weights=weights, validate=validate)


def selection_mask(cell_idx: Tensor, valid: Tensor, n_cells: int) -> Tensor:
    """``bool [B, N]``: True at the cells named by the valid slots of ``cell_idx [B, K]``
    (``valid [B, K]`` bool). Padded slots and indices outside ``[0, N)`` mark nothing. A comparison
    against ``arange(N)`` (``O(B·K·N)``, trivial at these sizes): deterministic on every backend, no
    scatter and no device synchronisation."""
    idx = _check_tensor("cell_idx", cell_idx)
    if idx.dtype != _INDEX_DTYPE:
        raise ValueError(f"cell_idx must have dtype torch.int64, got {idx.dtype}")
    if idx.dim() != 2:
        raise ValueError(f"cell_idx must have shape [B, K], got {list(idx.shape)}")
    mask = _check_bool("valid", valid, tuple(int(n) for n in idx.shape), idx.device)
    if isinstance(n_cells, bool) or not isinstance(n_cells, int) or n_cells < 1:
        raise ValueError(f"n_cells must be an int >= 1, got {n_cells!r}")
    cells = torch.arange(n_cells, device=idx.device)
    hits = (idx.unsqueeze(-1) == cells) & mask.unsqueeze(-1)  # [B, K, N]
    return hits.any(dim=1)


def gated_delta_update(
    state: CellGraphState,
    cell_idx: Tensor,
    c: Tensor,
    b: Tensor,
    valid: Tensor,
    *,
    validate: bool = True,
) -> CellGraphState:
    """The functional gated update of the selected cells (plan §2.3, §2.4)::

        h_new_k = (1 − b_k) ⊙ h[cell_k] + b_k ⊙ c_k          on valid slots
        h'[cell_k] = h_new_k ;  q' = q + Σ_valid (h_new_k − h[cell_k]) / N ;  age'[cell_k] = 0

    ``cell_idx [B, K]`` int64, ``c`` and ``b`` ``[B, K, d]`` floating (any dtype; the result is cast
    to the state dtype), ``valid [B, K]`` bool. Unselected cells keep their ``h`` bit for bit,
    selected cells hold exactly the dense formula's value, ``pos`` is untouched (the caller ticks).
    Padded slots (``valid=False``) are ignored: they are routed to a dummy row ``N`` of a widened
    copy that is sliced off, so they never collide with a selected cell 0, add exactly zero to ``q``
    and receive zero gradient. Autograd is intact with respect to ``h``, ``q``, ``c`` and ``b`` (an
    out-of-place ``scatter`` with a tensor source: deterministic on cpu and cuda under
    ``torch.use_deterministic_algorithms``).

    ``validate=True`` checks, with one device synchronisation, that every valid index lies in
    ``[0, N)`` and that the valid slots of a row are distinct (a duplicate would count twice in
    ``q`` and make the scatter order-dependent); ``False`` skips it for callers whose indices come
    from :meth:`hsn.model.router.Router.select`, which guarantees both.
    """
    if not isinstance(state, CellGraphState):
        raise TypeError(f"state must be a CellGraphState, got {type(state).__name__}")
    B, N, d = state.batch_size, state.n_cells, state.width
    device = state.device
    idx = _check_tensor("cell_idx", cell_idx)
    if idx.dtype != _INDEX_DTYPE:
        raise ValueError(f"cell_idx must have dtype torch.int64, got {idx.dtype}")
    if idx.dim() != 2 or idx.shape[0] != B or idx.shape[1] < 1:
        raise ValueError(f"cell_idx must have shape [B, K] = [{B}, >= 1], got {list(idx.shape)}")
    _check_device("cell_idx", idx, device)
    K = int(idx.shape[1])
    cand = _check_floating("c", c, (B, K, d), device)
    gate = _check_floating("b", b, (B, K, d), device)
    ok = _check_bool("valid", valid, (B, K), device)
    if not isinstance(validate, bool):
        raise TypeError(f"validate must be a bool, got {validate!r}")
    if validate:
        _validate_selection(idx, ok, N)

    # Padded slots gather cell 0 (in range on every backend, whatever the router padded with).
    index = torch.where(ok, idx, torch.zeros_like(idx)).unsqueeze(-1).expand(B, K, d)
    h_old = state.h.gather(1, index)
    gated = (1 - gate) * h_old + gate * cand
    h_new = torch.where(ok.unsqueeze(-1), gated, h_old).to(state.dtype)
    # Padded slots go to the dummy row N; valid slots are distinct, so no two slots write one row.
    safe = torch.where(ok, idx, torch.full_like(idx, N))
    wide = torch.cat([state.h, state.h.new_zeros(B, 1, d)], dim=1)
    h_next = wide.scatter(1, safe.unsqueeze(-1).expand(B, K, d), h_new).narrow(1, 0, N).contiguous()
    mask = selection_mask(idx, ok, N)
    q_next = state.q + (h_new - h_old).sum(dim=1) / N
    age_next = state.age.masked_fill(mask, 0)
    return CellGraphState(h=h_next, q=q_next, age=age_next, pos=state.pos)


def _validate_selection(cell_idx: Tensor, valid: Tensor, n_cells: int) -> None:
    """Range and distinctness of the valid slots, one device synchronisation."""
    B, K = (int(n) for n in cell_idx.shape)
    # Invalid slots get distinct sentinels >= N, so only valid duplicates are duplicates.
    sentinel = n_cells + torch.arange(K, device=cell_idx.device).unsqueeze(0).expand(B, K)
    keyed = torch.where(valid, cell_idx, sentinel)
    ordered = keyed.sort(dim=1).values
    duplicates = (ordered[:, 1:] == ordered[:, :-1]).sum()
    lowest = torch.where(valid, cell_idx, torch.full_like(cell_idx, n_cells)).min()
    highest = torch.where(valid, cell_idx, torch.full_like(cell_idx, -1)).max()
    lo, hi, n_dup = (int(v) for v in torch.stack([lowest, highest, duplicates]).tolist())
    if lo < 0 or hi >= n_cells:
        raise IndexError(
            f"valid cell_idx entries must lie in [0, {n_cells}), got values in [{lo}, {hi}]"
        )
    if n_dup:
        raise ValueError(
            f"valid cell_idx entries must be distinct within each row ({n_dup} duplicate "
            f"{'entry' if n_dup == 1 else 'entries'})"
        )
