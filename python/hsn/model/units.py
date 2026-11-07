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
from typing import Self

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

    def forward_all(self, v: Tensor) -> Tensor:
        """Dense reference: every unit on every cell, ``v [B, N, 4d] -> [B, N, M, d]``.

        ``out[b, n, j] = F_j(v[b, n])``. Materialises ``[M, B·N, H]`` hidden values: the first
        layer is one ``addmm`` against ``W1`` laid out ``[4d, M·H]``, the second one ``baddbmm``
        per unit (P1.10c, module docstring). The result is a ``[B, N, M, d]`` view of the
        ``[M, B·N, d]`` output. Under autocast the matmuls and bias adds run in the autocast dtype,
        as :meth:`forward_selected`'s ``addmm`` does.
        """
        if v.ndim != 3 or v.shape[-1] != self.in_dim:
            raise ValueError(
                f"forward_all expects v [B, N, {self.in_dim}], got shape {tuple(v.shape)}"
            )
        B, N = int(v.shape[0]), int(v.shape[1])
        M, H, P = self.n_units, self.hidden, B * N
        hidden = torch.addmm(self.b1.reshape(M * H), v.reshape(P, self.in_dim), self._w1_flat())
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
