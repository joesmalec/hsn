"""Kernel backends (P2.5): which implementation of the four HSN ops the sparse path runs.

The sparse round of :meth:`hsn.model.cell.PDGCell.step_sparse` is composed of exactly four functions
(plan P2.5 interfaces, the boundaries the fused kernels of P2.9–P2.11 replace)::

    gather_concat(u, eps, h, m, cell_idx)              -> v [B, K, 4d]
    grouped_unit_mlp(v, unit_idx, bank)                -> out [P, d]
    csr_messages(h, graph, V, cell_idx)                -> m [B, K, d]
    gated_delta_update(state, cell_idx, c, b, valid)   -> CellGraphState

:mod:`hsn.kernels.native` holds the L0 reference implementations in plain PyTorch ops (cpu, cuda,
mps). :func:`set_backend` chooses the backend by name and :func:`ops` returns the module whose four
functions the cell calls. Native remains the default until CUDA performance is measured.
The opt-in CUDA backend requires prebuilt artifacts and explicit prepare_cuda before capture.
Non-CUDA tensors and unsupported dtype/activation combinations retain native semantics.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Literal, Protocol, cast, get_args

if TYPE_CHECKING:
    from torch import Tensor, nn

    from hsn.model.graph import CellGraph
    from hsn.model.state import CellGraphState
    from hsn.model.units import UnitBank

__all__ = [
    "BACKENDS",
    "Backend",
    "KernelOps",
    "current_backend",
    "ops",
    "prepare_cuda",
    "set_backend",
]

# Import before defining ops(): Python otherwise replaces the package's dispatch function
# with its same-named submodule on the first CUDA import. Registration never initializes CUDA.
cuda_ops = import_module("hsn.kernels.ops")

prepare_cuda = cuda_ops.prepare_cuda

Backend = Literal["native", "cuda", "metal"]
#: The backend names, in the order the plan lists them (P2.5 native, P2.10 cuda, P2.11 metal).
BACKENDS: tuple[str, ...] = tuple(get_args(Backend))
_PENDING: dict[str, str] = {
    "metal": "P2.11 (Metal fused kernels)",
}

_backend: Backend = "native"


class KernelOps(Protocol):
    """The four op signatures every backend provides (a module or an object with these
    attributes)."""

    def gather_concat(
        self, u: Tensor, eps: Tensor, h: Tensor, m: Tensor, cell_idx: Tensor
    ) -> Tensor: ...

    def grouped_unit_mlp(
        self,
        v: Tensor,
        unit_idx: Tensor,
        bank: UnitBank,
        *,
        static: bool = False,
        w1_flat: Tensor | None = None,
    ) -> Tensor: ...

    def csr_messages(
        self,
        h: Tensor,
        graph: CellGraph,
        V: nn.Linear,
        cell_idx: Tensor,
        *,
        weights: Tensor | None = None,
        validate: bool = True,
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

    CUDA requires prebuilt artifacts; prepare_cuda loads them on the chosen device before
    execution. Selecting a backend does not create a context or run a workload.
    """
    global _backend
    if not isinstance(name, str) or name not in BACKENDS:
        raise ValueError(f"unknown kernel backend {name!r}; the backends are {BACKENDS}")
    if name in _PENDING:
        raise NotImplementedError(
            f"kernel backend {name!r} is chunk {_PENDING[name]}, which has not landed"
        )
    if name == "cuda" and not cuda_ops.available():
        raise RuntimeError(
            "P2.10 CUDA artifacts missing; build in this worktree before selecting cuda"
        )
    _backend = name


def current_backend() -> Backend:
    """The name of the backend :func:`ops` returns."""
    return _backend


def ops() -> KernelOps:
    """The selected implementation module."""
    if _backend == "cuda":
        return cast(KernelOps, cuda_ops)
    from hsn.kernels import native

    return native
