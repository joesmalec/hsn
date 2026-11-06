"""The specification of the four HSN kernel ops (P2.9) on the Python side.

The normative text is the C header ``kernels/include/hsn_ops.h``; this module and
``rust/hsn-kernels/src/spec.rs`` mirror it, and tests check that the constants of all three agree.
The numerical reference is the pure-Rust CPU oracle (``hsn_kernels::CpuBackend``); its outputs are
committed as test vectors in ``kernels/testvectors/`` (see :mod:`hsn.kernels.testvectors`), and
every implementation (the P2.5 native ops here, the fused kernels of P2.10/P2.11, the Rust
runtime of P5.2) must reproduce them within :func:`tolerance`.

The ops (``r`` indexes batch rows so that ``b`` always names the gate; ``P = B·K``)::

    (a) gather_concat     v[r,k] = [u[r]; eps[r]; h[r, cell_idx[r,k]]; m[r,k]]            [B,K,4d]
    (b) grouped_unit_mlp  j = unit_idx[p];  z1 = v[p] W1[j] + b1[j]  (hidden_saved, pre-GELU)
                          out[p] = gelu(z1) W2[j] + b2[j];  gelu(x) = x/2 (1 + erf(x/√2))  [P,d]
    (c) csr_messages      i = cell_idx[r,k];  m[r,k] = Σ_{e ∈ row i} a[e] · V h[r, indices[e]]
                                                                                         [B,K,d]
    (d) gated_delta_update, in place; for valid slots, n = cell_idx[r,k]:
                          h_old = h[r,n];  h[r,n] ← (1−b)⊙h_old + b⊙c
                          q[r] ← q[r] + (Σ_valid (h_new − h_old)) / N;  age[r,n] ← 0

Backward (every gradient output is written in full, never accumulated into)::

    (a) grad_u[r] = Σ_k dv[r,k,0:d];  grad_eps[r] = Σ_k dv[r,k,d:2d]
        grad_h[r,n] = Σ_{k: cell_idx[r,k]=n} dv[r,k,2d:3d];  grad_m[r,k] = dv[r,k,3d:4d]
    (b) a = gelu(z1);  dz1 = (dout W2[j]ᵀ) ⊙ gelu'(z1),  gelu'(x) = Φ(x) + x φ(x)
        grad_v = dz1 W1[j]ᵀ;  grad_W1[j] = Σ_{p∈j} vᵀ dz1;  grad_b1[j] = Σ_{p∈j} dz1
        grad_W2[j] = Σ_{p∈j} aᵀ dout;  grad_b2[j] = Σ_{p∈j} dout;  units without pairs → 0
    (c) for every slot (r,k), edge e of row cell_idx[r,k], source j = indices[e]:
        grad_h[r,j] += a[e] Vᵀ dm[r,k];  grad_V += a[e] dm[r,k] ⊗ h[r,j]
        grad_a[e] += dm[r,k]·(V h[r,j])
    (d) G = ∂L/∂h_out, g = ∂L/∂q_out; valid slot k, n = cell_idx[r,k]:
        grad_h[r,n] = (1−b)⊙G[r,n] − b⊙g[r]/N      (grad_h = G on cells no valid slot selects)
        grad_q = g;  grad_c = b⊙(G[r,n] + g[r]/N);  grad_b = (c − h_old)⊙(G[r,n] + g[r]/N)
        padded slots: grad_c = grad_b = 0

The ``g/N`` terms of (d) are the summary coupling: ``q`` is maintained incrementally as
``q + Σ Δh / N`` (plan §2.4), so its gradient reaches every selected ``h``, ``c`` and ``b``.

Layout: row-major contiguous tensors, ``int32`` indices, one float dtype per call (f32 everywhere,
f16 on CUDA and Metal, bf16 on CUDA only). ``V`` is ``[d_out, d_in]`` like ``nn.Linear.weight``;
``W1`` is ``[M, 4d, H]`` (``[in, out]`` per unit) like :class:`hsn.model.units.UnitBank`; ``a``
holds the *normalised* edge weights ``a_ij`` in CSR order. Padded slots of the gated update
(``valid = 0``) are never read, whatever ``cell_idx`` holds there; ``gather_concat`` and
``csr_messages`` take no ``valid`` and accept repeated indices (the cell maps padded slots to
cell 0).

Where the ops sit in the round (:meth:`hsn.model.cell.PDGCell.step_sparse`): ``v_raw =
gather_concat(…)``, then the host ops ``v = RMSNorm(v_raw)`` (amendment A1), ``b = σ(gate(v))``,
``c = tanh(W_out(α · grouped_unit_mlp(v, …)))``, then ``gated_delta_update(h, …, c, b)`` on the
raw ``h``. The RMSNorm, the gate, ``W_out``, ``tanh`` and the α mix are not part of the four ops,
and the grouped-MLP vectors take ``v`` as given. Changing what an op computes (for example a cell
input of another width) is a new :data:`SPEC_VERSION` in the header, ``spec.rs`` and this module
and a regeneration of the vectors (``HSN_WRITE_TESTVECTORS=1 cargo test -p hsn-kernels --test
oracle``); the manifest records the version and the loader refuses a mismatch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

__all__ = [
    "DTYPE_CODES",
    "HEADER_CONSTANTS",
    "INT32_MAX",
    "OPS",
    "OP_NAMES",
    "SIZES",
    "SPEC_VERSION",
    "STATUS_CODES",
    "Agreement",
    "OpSpec",
    "TensorSpec",
    "Tolerance",
    "VectorSize",
    "allclose_report",
    "check_int32_extent",
    "tolerance",
]

#: ``HSN_OPS_SPEC_VERSION``.
SPEC_VERSION = 1
#: ``HSN_DTYPE_*``: the float dtype codes of the C ABI.
DTYPE_CODES: dict[str, int] = {"f32": 0, "f16": 1, "bf16": 2}
#: ``HSN_OK`` and ``HSN_ERR_*``: the return codes of the C ABI.
STATUS_CODES: dict[str, int] = {
    "OK": 0,
    "ERR_INVALID_ARGUMENT": 1,
    "ERR_UNSUPPORTED_DTYPE": 2,
    "ERR_INDEX_OUT_OF_RANGE": 3,
    "ERR_DUPLICATE_INDEX": 4,
    "ERR_SIZE_OVERFLOW": 5,
    "ERR_BAD_GRAPH": 6,
    "ERR_BACKEND": 7,
}
#: Every ``#define`` of the header with a value, by its C name.
HEADER_CONSTANTS: dict[str, int] = {
    "HSN_OPS_SPEC_VERSION": SPEC_VERSION,
    **{f"HSN_DTYPE_{name.upper()}": code for name, code in DTYPE_CODES.items()},
    **{f"HSN_{name}": code for name, code in STATUS_CODES.items()},
}
#: Every dimension and element count of an op call must stay at or below this (int32 indexing).
INT32_MAX = 2**31 - 1


# --------------------------------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerance:
    """Elementwise agreement ``|x − ref| ≤ atol + rtol·|ref|`` (the ``torch.allclose`` rule)."""

    atol: float
    rtol: float


_TOLERANCES: dict[torch.dtype, Tolerance] = {
    torch.float32: Tolerance(atol=1e-5, rtol=1e-5),
    torch.float16: Tolerance(atol=1e-2, rtol=1e-2),
    torch.bfloat16: Tolerance(atol=1e-2, rtol=1e-2),
}


def tolerance(dtype: torch.dtype) -> Tolerance:
    """The tolerance of an implementation computing in ``dtype`` against the fp32 vectors
    (f32 ``1e-5``; f16 and bf16 ``1e-2``). Other dtypes raise ``ValueError``."""
    try:
        return _TOLERANCES[dtype]
    except KeyError:
        raise ValueError(
            f"no P2.9 tolerance for {dtype}; the ABI dtypes are {sorted(map(str, _TOLERANCES))}"
        ) from None


@dataclass(frozen=True)
class Agreement:
    """The outcome of :func:`allclose_report`: ``ok`` when every element agrees; ``max_abs`` the
    largest ``|x − ref|``; ``max_ratio`` the largest ``|x − ref| / (atol + rtol·|ref|)`` (agreement
    is a ratio ≤ 1); ``n_bad`` the number of disagreeing elements (NaN counts as one)."""

    ok: bool
    max_abs: float
    max_ratio: float
    n_bad: int


def allclose_report(actual: Tensor, expected: Tensor, dtype: torch.dtype) -> Agreement:
    """Compare ``actual`` with the vector tensor ``expected`` under ``tolerance(dtype)``, in
    float64 on the CPU. Integer tensors must match exactly. Shapes must be equal (``ValueError``
    otherwise: nothing is broadcast)."""
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"shape mismatch: actual {list(actual.shape)}, expected {list(expected.shape)}"
        )
    if not expected.is_floating_point():
        equal = actual.detach().cpu().to(torch.int64) == expected.cpu().to(torch.int64)
        n_bad = int((~equal).sum())
        diff = (actual.detach().cpu().to(torch.int64) - expected.cpu().to(torch.int64)).abs()
        max_abs = float(diff.max()) if diff.numel() else 0.0
        ratio = 0.0 if n_bad == 0 else math.inf
        return Agreement(ok=n_bad == 0, max_abs=max_abs, max_ratio=ratio, n_bad=n_bad)
    tol = tolerance(dtype)
    if expected.numel() == 0:
        return Agreement(ok=True, max_abs=0.0, max_ratio=0.0, n_bad=0)
    a = actual.detach().cpu().to(torch.float64)
    e = expected.cpu().to(torch.float64)
    diff = (a - e).abs()
    bound = tol.atol + tol.rtol * e.abs()
    bad = ~(diff <= bound)  # NaN compares False, so it is bad
    finite = torch.nan_to_num(diff, nan=math.inf)
    return Agreement(
        ok=not bool(bad.any()),
        max_abs=float(finite.max()),
        max_ratio=float((finite / bound).max()),
        n_bad=int(bad.sum()),
    )


# --------------------------------------------------------------------------------------------------
# Sizes and the tensors of every op
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorSize:
    """One of the chunk's three test-vector sizes."""

    name: str
    B: int
    N: int
    d: int
    K: int
    M: int
    H: int

    @property
    def P(self) -> int:  # noqa: N802 - the plan's symbol
        return self.B * self.K


SIZES: dict[str, VectorSize] = {
    "tiny": VectorSize("tiny", B=2, N=3, d=4, K=2, M=2, H=8),
    "ref": VectorSize("ref", B=4, N=32, d=64, K=4, M=8, H=512),
    "odd": VectorSize("odd", B=3, N=37, d=96, K=5, M=5, H=256),
}

Role = Literal["input", "output", "cotangent", "grad"]
Kind = Literal["float", "int32"]


@dataclass(frozen=True)
class TensorSpec:
    """One tensor of an op's vectors: its name (the header's argument name), its role (forward
    ``input`` or ``output`` including saved tensors, backward seed ``cotangent``, backward output
    ``grad`` of the input ``name[5:]``), its kind (float, stored as f32, or int32) and its symbolic
    shape (``"B"``, ``"4d"``, ``"N+1"``, …)."""

    name: str
    role: Role
    kind: Kind
    shape: tuple[str, ...]

    def resolve(self, dims: Mapping[str, int]) -> tuple[int, ...]:
        """The concrete shape for the dimension values ``dims``; an unknown symbol raises."""
        out: list[int] = []
        for symbol in self.shape:
            key = {"4d": "d", "N+1": "N"}.get(symbol, symbol)
            if key not in dims:
                raise ValueError(f"tensor {self.name} needs dimension {key}, not in {dict(dims)}")
            value = dims[key]
            out.append(4 * value if symbol == "4d" else value + 1 if symbol == "N+1" else value)
        return tuple(out)


@dataclass(frozen=True)
class OpSpec:
    """An op: its name, the dimension symbols of its vectors and every tensor they hold."""

    name: str
    dims: tuple[str, ...]
    tensors: tuple[TensorSpec, ...]

    def tensor(self, name: str) -> TensorSpec:
        for spec in self.tensors:
            if spec.name == name:
                return spec
        raise KeyError(f"op {self.name} has no tensor {name!r}")

    def by_role(self, role: Role) -> tuple[TensorSpec, ...]:
        return tuple(spec for spec in self.tensors if spec.role == role)


def _t(name: str, role: Role, kind: Kind, *shape: str) -> TensorSpec:
    return TensorSpec(name, role, kind, shape)


OPS: dict[str, OpSpec] = {
    "gather_concat": OpSpec(
        "gather_concat",
        ("B", "N", "d", "K"),
        (
            _t("u", "input", "float", "B", "d"),
            _t("eps", "input", "float", "B", "d"),
            _t("h", "input", "float", "B", "N", "d"),
            _t("m", "input", "float", "B", "K", "d"),
            _t("cell_idx", "input", "int32", "B", "K"),
            _t("v", "output", "float", "B", "K", "4d"),
            _t("grad_v", "cotangent", "float", "B", "K", "4d"),
            _t("grad_u", "grad", "float", "B", "d"),
            _t("grad_eps", "grad", "float", "B", "d"),
            _t("grad_h", "grad", "float", "B", "N", "d"),
            _t("grad_m", "grad", "float", "B", "K", "d"),
        ),
    ),
    "grouped_unit_mlp": OpSpec(
        "grouped_unit_mlp",
        ("P", "M", "d", "H"),
        (
            _t("v", "input", "float", "P", "4d"),
            _t("unit_idx", "input", "int32", "P"),
            _t("W1", "input", "float", "M", "4d", "H"),
            _t("b1", "input", "float", "M", "H"),
            _t("W2", "input", "float", "M", "H", "d"),
            _t("b2", "input", "float", "M", "d"),
            _t("hidden_saved", "output", "float", "P", "H"),
            _t("out", "output", "float", "P", "d"),
            _t("grad_out", "cotangent", "float", "P", "d"),
            _t("grad_v", "grad", "float", "P", "4d"),
            _t("grad_W1", "grad", "float", "M", "4d", "H"),
            _t("grad_b1", "grad", "float", "M", "H"),
            _t("grad_W2", "grad", "float", "M", "H", "d"),
            _t("grad_b2", "grad", "float", "M", "d"),
        ),
    ),
    "csr_messages": OpSpec(
        "csr_messages",
        ("B", "N", "d", "K", "E"),
        (
            _t("h", "input", "float", "B", "N", "d"),
            _t("indptr", "input", "int32", "N+1"),
            _t("indices", "input", "int32", "E"),
            _t("a", "input", "float", "E"),
            _t("V", "input", "float", "d", "d"),
            _t("cell_idx", "input", "int32", "B", "K"),
            _t("m", "output", "float", "B", "K", "d"),
            _t("grad_m", "cotangent", "float", "B", "K", "d"),
            _t("grad_h", "grad", "float", "B", "N", "d"),
            _t("grad_a", "grad", "float", "E"),
            _t("grad_V", "grad", "float", "d", "d"),
        ),
    ),
    "gated_delta_update": OpSpec(
        "gated_delta_update",
        ("B", "N", "d", "K"),
        (
            _t("h", "input", "float", "B", "N", "d"),
            _t("q", "input", "float", "B", "d"),
            _t("age", "input", "int32", "B", "N"),
            _t("cell_idx", "input", "int32", "B", "K"),
            _t("valid", "input", "int32", "B", "K"),
            _t("c", "input", "float", "B", "K", "d"),
            _t("b", "input", "float", "B", "K", "d"),
            _t("h_out", "output", "float", "B", "N", "d"),
            _t("q_out", "output", "float", "B", "d"),
            _t("age_out", "output", "int32", "B", "N"),
            _t("h_old_saved", "output", "float", "B", "K", "d"),
            _t("grad_h_out", "cotangent", "float", "B", "N", "d"),
            _t("grad_q_out", "cotangent", "float", "B", "d"),
            _t("grad_h", "grad", "float", "B", "N", "d"),
            _t("grad_q", "grad", "float", "B", "d"),
            _t("grad_c", "grad", "float", "B", "K", "d"),
            _t("grad_b", "grad", "float", "B", "K", "d"),
        ),
    ),
}
#: The op names in the header's order (a, b, c, d).
OP_NAMES: tuple[str, ...] = tuple(OPS)


def check_int32_extent(name: str, shape: Sequence[int]) -> int:
    """The element count of a tensor of ``shape``, refused with ``OverflowError`` when it or any
    dimension exceeds :data:`INT32_MAX` (the ABI indexes with int32; impossible at the reference
    sizes, checked anyway). For the hosts of the fused kernels (P2.10, P2.11)."""
    count = 1
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError(f"{name}: dimensions must be ints >= 0, got {list(shape)}")
        if dim > INT32_MAX:
            raise OverflowError(f"{name}: dimension {dim} exceeds INT32_MAX ({INT32_MAX})")
        count *= dim
    if count > INT32_MAX:
        raise OverflowError(
            f"{name}: {list(shape)} has {count} elements, more than INT32_MAX ({INT32_MAX})"
        )
    return count
