"""The capability-unit bank (P1.2): ``M`` two-layer MLPs ``F_j: 4d -> H -> d`` stored as stacked
parameters, with a dense all-units pass and a grouped selected-pairs pass.

Plan §2.1 / §2.3: each unit maps the cell input ``v_i = [u_t; ε_t; h_i; m_i]`` (width ``4d``) to a
candidate contribution of width ``d``; the cell update mixes the outputs of the units selected for
a cell. The weights are stacked so that every unit lives in one tensor each: ``W1 [M, 4d, H]``,
``b1 [M, H]``, ``W2 [M, H, d]``, ``b2 [M, d]`` (Appendix C.1 row ``units``: ``M × (4d·H + H + H·d
+ d)``, 1,315,328 for A.1). A unit is applied as ``x @ W1[j] + b1[j]``, so ``W1`` is stored
``[in, out]`` per unit (not ``nn.Linear``'s ``[out, in]``).

Two execution paths that compute the same function, in the same dtype (under autocast both run
the matmuls, bias adds and activation in the autocast dtype):

* :meth:`UnitBank.forward_all` is the dense reference: every unit on every cell, output
  ``[B, N, M, d]``. It materialises ``B·N·M·H`` hidden values, fine for the A.1 dense reference
  (P1.5) and unusable at scale (the chunk's pitfall). P1.10c: the first layer is one ``addmm`` of
  the ``P = B·N`` inputs against ``W1`` laid out as ``[4d, M·H]`` (built once per
  :meth:`UnitBank.window`, per call outside one), the second one ``baddbmm`` over the units with
  the hidden values as ``[M, P, H]``; the output is a ``[B, N, M, d]`` view of ``[M, P, d]``, with
  no per-call copy of the weights.
* :meth:`UnitBank.forward_selected` is the grouped pass that the sparse path (P2.5) and the fused
  ``grouped_unit_mlp`` kernel (P2.9) replace: ``P`` (cell, unit) pairs with one unit each. Pairs
  are sorted by unit (stable), one matmul pair runs per *present* unit, and the outputs are
  restored to the input order. Nothing is padded. The unit histogram is read on the host, which
  is one device-to-host copy per call and rules out graph capture of this reference path.

``unit_ids`` and ``versions`` are the versioned-interface metadata of paper §4 that the P5.1
manifests will carry. P1.2 holds them and carries them in the ``state_dict`` as extra state:
loading a checkpoint restores the versions and refuses a bank whose unit ids differ.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Self

import torch
from torch import Tensor, nn

from hsn.config import ModelConfig
from hsn.model.embeddings import init_weight_
from hsn.model.encoder import make_activation

__all__ = ["UnitBank"]

_INDEX_DTYPES = (torch.int64, torch.int32)


def _check_unit_ids(unit_ids: Sequence[str] | None, n_units: int) -> list[str]:
    if unit_ids is None:
        return [f"unit{j}" for j in range(n_units)]
    ids = [] if isinstance(unit_ids, str) else list(unit_ids)
    if len(ids) != n_units or not all(isinstance(u, str) for u in ids) or len(set(ids)) != n_units:
        raise ValueError(f"unit_ids must be {n_units} distinct strings, got {unit_ids!r}")
    return ids


def _check_versions(versions: Sequence[int] | None, n_units: int) -> list[int]:
    if versions is None:
        return [1] * n_units
    vs = list(versions)
    if len(vs) != n_units or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in vs
    ):
        raise ValueError(f"versions must be {n_units} integers >= 1, got {versions!r}")
    return vs


class _PermuteRows(torch.autograd.Function):
    """:func:`permute_rows`."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, perm: Tensor, inverse: Tensor) -> Tensor:
        ctx.save_for_backward(inverse)
        return x.index_select(0, perm)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None, None]:
        (inverse,) = ctx.saved_tensors
        return grad.index_select(0, inverse), None, None


def permute_rows(x: Tensor, perm: Tensor, inverse: Tensor) -> Tensor:
    """``x.index_select(0, perm)`` for a permutation ``perm`` of ``range(len(x))`` with its
    inverse ``inverse`` (``inverse[perm] = arange``; neither is checked). The value is
    ``index_select``'s. The backward takes the gradient's rows in the inverse order, a gather:
    ``index_select``'s own backward adds every row into a tensor of zeros (``index_add``), which
    for a permutation writes each row exactly once, so the gradients are equal bit for bit and
    the scatter and the zero tensor are saved (P2.10: two ``[B·T·K, ·]`` reorderings per
    window in the replay's grouped unit pass)."""
    out: Tensor = _PermuteRows.apply(x, perm, inverse)
    return out


class UnitBank(nn.Module):
    """``M`` capability units ``F_j: 4d -> H -> d`` with stacked weights.

    ``n_units`` is ``M``, ``d`` the width (the input width is ``4d``), ``hidden`` is ``H``;
    ``activation`` and ``init`` are ``model.activation`` / ``model.init``. ``unit_ids`` default to
    ``unit0 .. unit{M-1}`` and ``versions`` to ``1`` for every unit.
    """

    W1: nn.Parameter  # [M, 4d, H]
    b1: nn.Parameter  # [M, H]
    W2: nn.Parameter  # [M, H, d]
    b2: nn.Parameter  # [M, d]

    def __init__(
        self,
        n_units: int,
        d: int,
        hidden: int,
        activation: str,
        *,
        init: str,
        unit_ids: Sequence[str] | None = None,
        versions: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        for name, value in (("n_units", n_units), ("d", d), ("hidden", hidden)):
            if value < 1:
                raise ValueError(f"UnitBank needs {name} >= 1, got {value}")
        self.n_units = n_units
        self.d = d
        self.hidden = hidden
        self.in_dim = 4 * d
        self.activation_name = activation
        self.init = init
        self.act = make_activation(activation)
        self.W1 = nn.Parameter(torch.empty(n_units, self.in_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(n_units, hidden))
        self.W2 = nn.Parameter(torch.empty(n_units, hidden, d))
        self.b2 = nn.Parameter(torch.zeros(n_units, d))
        init_weight_(self.W1, init)
        init_weight_(self.W2, init)
        self.unit_ids: list[str] = _check_unit_ids(unit_ids, n_units)
        self.versions: list[int] = _check_versions(versions, n_units)
        #: P1.10c: ``W1`` as ``[4d, M·H]`` for the window of :meth:`window` (``None`` outside),
        #: bound to the ``W1`` object, its version and the grad mode it was built under.
        self._window_w1: tuple[tuple[int, int, bool], Tensor] | None = None
        self._in_window = False

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> Self:
        return cls(cfg.n_units, cfg.d, cfg.hidden, cfg.activation, init=cfg.state_init)

    # ----------------------------------------------------------------------------------------------
    # Execution
    # ----------------------------------------------------------------------------------------------

    @contextmanager
    def window(self) -> Iterator[None]:
        """A window of :meth:`forward_all` calls (P1.10c; :meth:`hsn.model.cell.PDGCell.window`):
        ``W1``'s ``[4d, M·H]`` layout is built once, on first use, and dropped at the end. A use
        that sees ``W1`` replaced, updated in place or the grad mode switched raises. Windows do
        not nest."""
        if self._in_window:
            raise RuntimeError("UnitBank.window is already active; windows do not nest")
        self._in_window = True
        self._window_w1 = None
        try:
            yield
        finally:
            self._in_window = False
            self._window_w1 = None

    def _w1_flat(self) -> Tensor:
        """``W1 [M, 4d, H]`` laid out as ``[4d, M·H]`` (column ``j·H + k`` is unit ``j``'s hidden
        ``k``): once per :meth:`window`, per call outside one."""
        w1 = self.W1
        if not self._in_window:
            return w1.permute(1, 0, 2).reshape(self.in_dim, self.n_units * self.hidden)
        stamp = (id(w1), w1._version, torch.is_grad_enabled())
        if self._window_w1 is None:
            flat = w1.permute(1, 0, 2).reshape(self.in_dim, self.n_units * self.hidden)
            self._window_w1 = (stamp, flat)
            return flat
        if self._window_w1[0] != stamp:
            raise RuntimeError(
                "UnitBank.window: W1 must not change inside a window (an in-place update, a new "
                "parameter, or a grad-mode switch between positions)"
            )
        return self._window_w1[1]

    def forward_all(self, v: Tensor, *, w1_flat: Tensor | None = None) -> Tensor:
        """Dense reference: every unit on every cell, ``v [B, N, 4d] -> [B, N, M, d]``.

        ``out[b, n, j] = F_j(v[b, n])``. Materialises ``[M, B·N, H]`` hidden values: the first
        layer is one ``addmm`` against ``W1`` laid out ``[4d, M·H]``, the second one ``baddbmm``
        per unit (P1.10c, module docstring). The result is a ``[B, N, M, d]`` view of the
        ``[M, B·N, d]`` output. Under autocast the matmuls and bias adds run in the autocast dtype,
        as :meth:`forward_selected`'s ``addmm`` does. ``w1_flat`` (P2.8, a compiled window) is the
        window's ``[4d, M·H]`` layout (:class:`~hsn.model.cell.WindowConsts`) instead of the window
        cache, the same values.
        """
        if v.ndim != 3 or v.shape[-1] != self.in_dim:
            raise ValueError(
                f"forward_all expects v [B, N, {self.in_dim}], got shape {tuple(v.shape)}"
            )
        B, N = int(v.shape[0]), int(v.shape[1])
        M, H, P = self.n_units, self.hidden, B * N
        w1 = self._w1_flat() if w1_flat is None else w1_flat
        hidden = torch.addmm(self.b1.reshape(M * H), v.reshape(P, self.in_dim), w1)
        hidden = self.act(hidden).view(P, M, H).transpose(0, 1)  # [M, P, H]
        out = torch.baddbmm(self.b2[:, None, :], hidden, self.W2)  # [M, P, d]
        return out.view(M, B, N, self.d).permute(1, 2, 0, 3)

    def forward_selected(self, v: Tensor, unit_idx: Tensor) -> Tensor:
        """Grouped pass: ``out[p] = F_{unit_idx[p]}(v[p])`` for ``P`` pairs, ``[P, 4d] -> [P, d]``.

        ``unit_idx`` is int64 or int32 with values in ``[0, M)``; anything else raises
        (``IndexError`` for values outside the range, checked on the host copy before any device
        indexing, so CUDA never reaches a device-side assert). Pairs are sorted by unit (stable),
        each present unit runs one ``addmm`` per layer on its contiguous block, and the outputs
        are put back in the input order through ``index_select`` (its backward is an
        ``index_add``, which deterministic mode allows on every backend). The per-unit weights
        come from one ``unbind`` per parameter, so backward produces one stacked gradient per
        parameter rather than one full-size gradient per present unit.

        ``P = 0`` returns an empty ``[0, d]`` tensor of ``v``'s dtype with no graph; the A.1
        selection (``P = B·K``, ``k_min >= 1``) never produces it.
        """
        if v.ndim != 2 or v.shape[1] != self.in_dim:
            raise ValueError(
                f"forward_selected expects v [P, {self.in_dim}], got shape {tuple(v.shape)}"
            )
        if unit_idx.dtype not in _INDEX_DTYPES:
            raise TypeError(f"unit_idx must be int64 or int32, got dtype {unit_idx.dtype}")
        if unit_idx.ndim != 1 or unit_idx.shape[0] != v.shape[0]:
            raise ValueError(
                f"unit_idx must be [P] with P = {v.shape[0]} rows of v, got shape "
                f"{tuple(unit_idx.shape)}"
            )
        n_pairs = v.shape[0]
        if n_pairs == 0:
            return v.new_zeros((0, self.d))

        idx = unit_idx.to(torch.int64)
        order = torch.argsort(idx, stable=True)
        sorted_host = idx.index_select(0, order).cpu()
        lo, hi = int(sorted_host[0]), int(sorted_host[-1])
        if lo < 0 or hi >= self.n_units:
            raise IndexError(
                f"unit_idx values must lie in [0, {self.n_units}), got min {lo} and max {hi}"
            )
        counts: list[int] = torch.bincount(sorted_host, minlength=self.n_units).tolist()

        v_sorted = v.index_select(0, order)
        w1s = self.W1.unbind(0)
        b1s = self.b1.unbind(0)
        w2s = self.W2.unbind(0)
        b2s = self.b2.unbind(0)
        outputs: list[Tensor] = []
        start = 0
        for j, count in enumerate(counts):
            if count == 0:
                continue
            block = v_sorted[start : start + count]
            hidden = self.act(torch.addmm(b1s[j], block, w1s[j]))
            outputs.append(torch.addmm(b2s[j], hidden, w2s[j]))
            start += count
        out_sorted = torch.cat(outputs, dim=0)
        inverse = torch.empty_like(order).scatter_(
            0, order, torch.arange(n_pairs, device=order.device)
        )
        return out_sorted.index_select(0, inverse)

    def forward_grouped(self, v: Tensor, unit_idx: Tensor) -> Tensor:
        """:meth:`forward_selected` for a large batch of pairs (P2.10, the split window's
        replay: every pair of a window at once): the same grouped pass — pairs sorted by unit,
        one ``addmm`` per layer and present unit, the order restored — with the sorted pairs cut
        by one ``split`` instead of one slice per unit. The values are :meth:`forward_selected`'s;
        the backward concatenates the units' input gradients once, where a slice per unit
        allocates and adds a full ``[P, 4d]`` tensor per present unit (eight 33 MB tensors for a
        B32/T256 window). One host read (the unit histogram); ``unit_idx`` must lie in
        ``[0, M)`` (the caller's recorded decisions). The two reorderings are
        :func:`permute_rows`: their backward gathers, where ``index_select``'s scatters each
        ``[P, ·]`` gradient into a tensor of zeros."""
        if v.ndim != 2 or v.shape[1] != self.in_dim:
            raise ValueError(
                f"forward_grouped expects v [P, {self.in_dim}], got shape {tuple(v.shape)}"
            )
        if unit_idx.ndim != 1 or unit_idx.shape[0] != v.shape[0]:
            raise ValueError(
                f"unit_idx must be [P] with P = {v.shape[0]} rows of v, got shape "
                f"{tuple(unit_idx.shape)}"
            )
        n_pairs = int(v.shape[0])
        if n_pairs == 0:
            return v.new_zeros((0, self.d))
        idx = unit_idx.to(torch.int64)
        counts: list[int] = torch.bincount(idx, minlength=self.n_units).tolist()
        if len(counts) != self.n_units:
            raise IndexError(f"unit_idx values must lie in [0, {self.n_units})")
        order = torch.argsort(idx, stable=True)
        inverse = torch.empty_like(order).scatter_(
            0, order, torch.arange(n_pairs, device=order.device)
        )
        blocks = torch.split(permute_rows(v, order, inverse), counts)
        w1s = self.W1.unbind(0)
        b1s = self.b1.unbind(0)
        w2s = self.W2.unbind(0)
        b2s = self.b2.unbind(0)
        outputs: list[Tensor] = []
        for j, count in enumerate(counts):
            if count == 0:
                continue
            hidden = self.act(torch.addmm(b1s[j], blocks[j], w1s[j]))
            outputs.append(torch.addmm(b2s[j], hidden, w2s[j]))
        out_sorted = torch.cat(outputs, dim=0)
        return permute_rows(out_sorted, inverse, order)

    def forward_static(
        self, v: Tensor, unit_idx: Tensor, *, w1_flat: Tensor | None = None
    ) -> Tensor:
        """The capturable grouped pass (P2.8): ``out[p] = F_{unit_idx[p]}(v[p])``, ``[P, 4d] ->
        [P, d]``, with static shapes and no host synchronisation. Every unit runs on every pair
        (:meth:`forward_all` on the ``P`` pairs, ``M``× the FLOPs of :meth:`forward_selected` on
        them), then a ``gather`` picks each pair's unit, so the shapes do not depend on which units
        are present. The unselected units' outputs get exact-zero gradients. ``unit_idx`` must lie
        in ``[0, M)`` (not checked: the indices come from :meth:`Router.select`, and checking
        their values would synchronise). Agrees with :meth:`forward_selected` to rounding (the
        matmuls have other shapes). ``w1_flat`` as in :meth:`forward_all`."""
        if v.ndim != 2 or v.shape[1] != self.in_dim:
            raise ValueError(
                f"forward_static expects v [P, {self.in_dim}], got shape {tuple(v.shape)}"
            )
        if unit_idx.dtype not in _INDEX_DTYPES:
            raise TypeError(f"unit_idx must be int64 or int32, got dtype {unit_idx.dtype}")
        if unit_idx.ndim != 1 or unit_idx.shape[0] != v.shape[0]:
            raise ValueError(
                f"unit_idx must be [P] with P = {v.shape[0]} rows of v, got shape "
                f"{tuple(unit_idx.shape)}"
            )
        P = int(v.shape[0])
        every = self.forward_all(v.reshape(1, P, self.in_dim), w1_flat=w1_flat)[0]  # [P, M, d]
        index = unit_idx.to(torch.int64)[:, None, None].expand(P, 1, self.d)
        return every.gather(1, index).squeeze(1)

    def _w2_flat(self) -> Tensor:
        """``W2 [M, H, d]`` laid out as ``[H, M·d]`` (column ``j·d + k`` is unit ``j``'s output
        ``k``), the second layer's counterpart of :meth:`_w1_flat`."""
        return self.W2.permute(1, 0, 2).reshape(self.hidden, self.n_units * self.d)

    def forward_static_chosen(
        self,
        v: Tensor,
        unit_idx: Tensor,
        *,
        w1_flat: Tensor,
        w2_flat: Tensor,
        b1_flat: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """:meth:`forward_static` without the work nobody reads (P2.10, the forward sweep):
        ``out[p] = F_{unit_idx[p]}(v[p])``, ``[P, 4d] -> [P, d]``, static shapes, no host
        synchronisation. The first layer is :meth:`forward_all`'s ``addmm`` (every unit: the
        pairs' units differ, and one matmul over all of them is what keeps the shapes static);
        the chosen unit's hidden row is gathered *before* the activation, so the activation and
        the second layer run on ``[P, H]`` instead of ``[M, P, H]``: one ``addmm`` against ``W2``
        laid out ``[H, M·d]`` (:meth:`_w2_flat`, ``w2_flat``) and a gather of the chosen unit's
        ``d`` outputs. Each value is the same dot product as :meth:`forward_static`'s (the
        chosen unit's row of ``W1``, its activation, its ``W2`` and biases); it agrees with it to
        rounding (the second matmul has another shape). Forward only: the unselected units'
        second-layer weights get no gradient path here, so a caller that needs gradients keeps
        :meth:`forward_static`.

        Returns ``(out, rider)``. With ``b1_flat`` the first layer carries riders: ``w1_flat`` is
        ``[4d, M·H + r]`` and ``b1_flat`` ``[M·H + r]``, the bank's own layout followed by ``r``
        columns of another linear map of the same input (the cell's gate: one matmul instead
        of two); ``rider [P, r]`` is their output, ``None`` without ``b1_flat``."""
        if v.ndim != 2 or v.shape[1] != self.in_dim:
            raise ValueError(
                f"forward_static_chosen expects v [P, {self.in_dim}], got shape {tuple(v.shape)}"
            )
        if unit_idx.ndim != 1 or unit_idx.shape[0] != v.shape[0]:
            raise ValueError(
                f"unit_idx must be [P] with P = {v.shape[0]} rows of v, got shape "
                f"{tuple(unit_idx.shape)}"
            )
        P, M, H, d = int(v.shape[0]), self.n_units, self.hidden, self.d
        index = unit_idx.to(torch.int64)[:, None, None]
        first = torch.addmm(self.b1.reshape(M * H) if b1_flat is None else b1_flat, v, w1_flat)
        rider = None if b1_flat is None else first[:, M * H :]
        hidden = first[:, : M * H].view(P, M, H)
        chosen = self.act(hidden.gather(1, index.expand(P, 1, H)).squeeze(1))  # [P, H]
        out = torch.addmm(self.b2.reshape(M * d), chosen, w2_flat).view(P, M, d)
        return out.gather(1, index.expand(P, 1, d)).squeeze(1), rider

    # ----------------------------------------------------------------------------------------------
    # Accounting and metadata
    # ----------------------------------------------------------------------------------------------

    def param_count(self) -> int:
        """Measured parameter count of the whole bank (``M × (4d·H + H + H·d + d)``)."""
        return sum(p.numel() for p in self.parameters())

    def get_extra_state(self) -> dict[str, list[str] | list[int]]:
        """``unit_ids`` and ``versions`` travel with the weights in the ``state_dict``."""
        return {"unit_ids": list(self.unit_ids), "versions": list(self.versions)}

    def set_extra_state(self, state: object) -> None:
        """Restore ``versions`` from a checkpoint; refuse one whose ``unit_ids`` differ."""
        if not isinstance(state, dict) or set(state) != {"unit_ids", "versions"}:
            raise ValueError(
                f"UnitBank extra state must have keys unit_ids and versions, got {state!r}"
            )
        unit_ids = _check_unit_ids(state["unit_ids"], self.n_units)
        if unit_ids != self.unit_ids:
            raise ValueError(
                f"checkpoint unit_ids {unit_ids!r} do not match this bank's {self.unit_ids!r}"
            )
        self.versions = _check_versions(state["versions"], self.n_units)

    def extra_repr(self) -> str:
        return (
            f"n_units={self.n_units}, d={self.d}, hidden={self.hidden}, "
            f"activation={self.activation_name!r}, init={self.init!r}"
        )
