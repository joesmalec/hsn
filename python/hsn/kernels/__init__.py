"""Kernel backends (P2.5): which implementation of the four HSN ops the sparse path runs.

The sparse round of :meth:`hsn.model.cell.PDGCell.step_sparse` is composed of exactly four functions
(plan P2.5 interfaces, the boundaries the fused kernels of P2.9–P2.11 replace)::

    gather_concat(u, eps, h, m, cell_idx)              -> v [B, K, 4d]
    grouped_unit_mlp(v, unit_idx, bank)                -> out [P, d]
    csr_messages(h, graph, V, cell_idx)                -> m [B, K, d]
    gated_delta_update(state, cell_idx, c, b, valid)   -> CellGraphState

:mod:`hsn.kernels.native` holds the L0 reference implementations in plain PyTorch ops (cpu, cuda,
mps). :func:`set_backend` chooses the backend by name and :func:`ops` returns the module whose four
functions the cell calls, so a later backend swaps in at exactly these boundaries. In P2.5 only
``native`` exists: ``cuda`` is the P2.10 extension and ``metal`` the P2.11 one, and asking for
either raises until that chunk lands (an error, never a silent fall-back to the native path).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, get_args

if TYPE_CHECKING:
    from torch import Tensor, nn

    from hsn.model.graph import CellGraph
    from hsn.model.state import CellGraphState
    from hsn.model.units import UnitBank

__all__ = ["BACKENDS", "Backend", "KernelOps", "current_backend", "ops", "set_backend"]

Backend = Literal["native", "cuda", "metal"]
#: The backend names, in the order the plan lists them (P2.5 native, P2.10 cuda, P2.11 metal).
BACKENDS: tuple[str, ...] = tuple(get_args(Backend))
_PENDING: dict[str, str] = {
    "cuda": "P2.10 (CUDA fused kernels)",
    "metal": "P2.11 (Metal fused kernels)",
}

_backend: Backend = "native"


class KernelOps(Protocol):
    """The four op signatures every backend provides (a module or an object with these
    attributes)."""

    def gather_concat(
        self, u: Tensor, eps: Tensor, h: Tensor, m: Tensor, cell_idx: Tensor
    ) -> Tensor: ...

    def grouped_unit_mlp(self, v: Tensor, unit_idx: Tensor, bank: UnitBank) -> Tensor: ...

    def csr_messages(
        self, h: Tensor, graph: CellGraph, V: nn.Linear, cell_idx: Tensor
    ) -> Tensor: ...

    def gated_delta_update(
        self,
        state: CellGraphState,
        cell_idx: Tensor,
        c: Tensor,
        b: Tensor,
        valid: Tensor,
        *,
        validate: bool = True,
    ) -> CellGraphState: ...


def set_backend(name: Backend) -> None:
    """Select the kernel backend for every later sparse round.

    ``"native"`` is the P2.5 reference and the only backend that exists; ``"cuda"`` and ``"metal"``
    raise ``NotImplementedError`` naming the chunk that adds them (P2.10, P2.11); any other name
    raises ``ValueError``.
    """
    global _backend
    if not isinstance(name, str) or name not in BACKENDS:
        raise ValueError(f"unknown kernel backend {name!r}; the backends are {BACKENDS}")
    if name in _PENDING:
        raise NotImplementedError(
            f"kernel backend {name!r} is chunk {_PENDING[name]}, which has not landed; only "
            "'native' (P2.5) exists"
        )
    _backend = "native"


def current_backend() -> Backend:
    """The name of the backend :func:`ops` returns."""
    return _backend


def ops() -> KernelOps:
    """The implementation module of the current backend (``hsn.kernels.native`` in P2.5)."""
    if _backend != "native":  # pragma: no cover - set_backend admits only "native"
        raise RuntimeError(f"kernel backend {_backend!r} has no implementation")
    from hsn.kernels import native

    return native
