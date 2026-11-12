"""P2.10 CUDA custom operators and sparse-model adapters.

Import registers schemas/fakes/autograd only. Call prepare_cuda before execution or
capture, after the resource gate. Builds are explicit; there is no import-time JIT.
The raw operators use P2.9's contiguous, single-dtype tensors and int32 indices.
Their index/CSR preconditions are the caller's responsibility (hsn_ops.h).
Only first derivatives are registered. Hidden implementation outputs are nondifferentiable.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from hsn.model.graph import CellGraph
    from hsn.model.state import CellGraphState
    from hsn.model.units import UnitBank

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "build"


def available() -> bool:
    """Artifact presence only: never initializes CUDA or loads a library."""
    return (
        (BUILD / "hsn_ops.fatbin").is_file()
        and (BUILD / "cuda_build.json").is_file()
        and any(BUILD.glob("hsn_cuda_py*.so"))
    )


@lru_cache(maxsize=1)
def _extension() -> ModuleType:
    if not available():
        raise RuntimeError(
            "P2.10 CUDA artifacts missing; run scripts/build_kernels.py --target cuda "
            "in this worktree's environment through the idle guard"
        )
    manifest = json.loads((BUILD / "cuda_build.json").read_text())
    if not manifest["extension_built"] or manifest["torch"] != torch.__version__:
        raise RuntimeError("HSN CUDA extension must be rebuilt for this PyTorch version")
    if (
        hashlib.sha256((BUILD / "hsn_ops.fatbin").read_bytes()).hexdigest()
        != manifest["fatbin_sha256"]
    ):
        raise RuntimeError("HSN CUDA fatbin differs from its build manifest")
    for name, digest in manifest["sources"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"HSN CUDA source changed since build: {name}; rebuild first")
    libraries = list(BUILD.glob("hsn_cuda_py*.so"))
    if len(libraries) != 1:
        raise RuntimeError(
            "expected exactly one hsn_cuda_py extension; clean stale build artifacts"
        )
    spec = importlib.util.spec_from_file_location("hsn_cuda_py", libraries[0])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_cuda(device: int = 0) -> None:
    """Load every fatbin entry point on this device before warmup/graph capture."""
    _extension().prepare(str(BUILD / "hsn_ops.fatbin"), device)


def _run(op: int, inputs: list[Tensor], dims: list[int]) -> list[Tensor]:
    return cast(list[Tensor], _extension().run(op, inputs, dims))


def _dims(h: Tensor, idx: Tensor, *, edges: int = 0) -> list[int]:
    B, N, d = h.shape
    K = idx.shape[1]
    return [B, N, d, K, B * K, 1, 1, edges]


def _unit_dims(v: Tensor, w1: Tensor, w2: Tensor) -> list[int]:
    P, M, H, d = v.shape[0], w1.shape[0], w1.shape[2], w2.shape[2]
    return [P, 1, d, 1, P, M, H, 0]


@torch.library.custom_op("hsn::gather_concat", mutates_args=(), device_types="cuda")
def gather_op(u: Tensor, eps: Tensor, h: Tensor, m: Tensor, idx: Tensor) -> Tensor:
    return _run(0, [u, eps, h, m, idx], _dims(h, idx))[0]


@gather_op.register_fake
def _gather_fake(u: Tensor, eps: Tensor, h: Tensor, m: Tensor, idx: Tensor) -> Tensor:
    return h.new_empty((*idx.shape, 4 * h.shape[2]))


@torch.library.custom_op("hsn::gather_concat_backward", mutates_args=(), device_types="cuda")
def gather_backward(dv: Tensor, idx: Tensor, n: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    B, K, width = dv.shape
    d = width // 4
    du, de, dh, dm = _run(1, [dv, idx], [B, n, d, K, B * K, 1, 1, 0])
    return du, de, dh, dm


@gather_backward.register_fake
def _gather_bwd_fake(dv: Tensor, idx: Tensor, n: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    B, K, width = dv.shape
    d = width // 4
    return (
        dv.new_empty((B, d)),
        dv.new_empty((B, d)),
        dv.new_empty((B, n, d)),
        dv.new_empty((B, K, d)),
    )


def _gather_setup(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
    ctx.save_for_backward(inputs[4])
    ctx.n = inputs[2].shape[1]


def _gather_grad(ctx: Any, dv: Tensor) -> tuple[Any, ...]:
    return (*gather_backward(dv.contiguous(), ctx.saved_tensors[0], ctx.n), None)


gather_op.register_autograd(_gather_grad, setup_context=_gather_setup)


@torch.library.custom_op("hsn::grouped_unit_mlp", mutates_args=(), device_types="cuda")
def mlp_op(
    v: Tensor, idx: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor
) -> tuple[Tensor, Tensor]:
    out, hidden = _run(2, [v, idx, w1, b1, w2, b2], _unit_dims(v, w1, w2))
    return out, hidden


@mlp_op.register_fake
def _mlp_fake(
    v: Tensor, idx: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor
) -> tuple[Tensor, Tensor]:
    return v.new_empty((v.shape[0], w2.shape[2])), v.new_empty((v.shape[0], w1.shape[2]))


@torch.library.custom_op("hsn::grouped_unit_mlp_backward", mutates_args=(), device_types="cuda")
def mlp_backward(
    v: Tensor, idx: Tensor, w1: Tensor, w2: Tensor, hidden: Tensor, dout: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    dv, dw1, db1, dw2, db2 = _run(3, [v, idx, w1, w2, hidden, dout], _unit_dims(v, w1, w2))
    return dv, dw1, db1, dw2, db2


@mlp_backward.register_fake
def _mlp_bwd_fake(
    v: Tensor, idx: Tensor, w1: Tensor, w2: Tensor, hidden: Tensor, dout: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(v),
        torch.empty_like(w1),
        w1.new_empty((w1.shape[0], w1.shape[2])),
        torch.empty_like(w2),
        w2.new_empty((w2.shape[0], w2.shape[2])),
    )


def _mlp_setup(ctx: Any, inputs: tuple[Any, ...], output: tuple[Tensor, Tensor]) -> None:
    v, idx, w1, _, w2, _ = inputs
    ctx.save_for_backward(v, idx, w1, w2, output[1])
    ctx.mark_non_differentiable(output[1])


def _mlp_grad(ctx: Any, dout: Tensor, dhidden: Tensor) -> tuple[Any, ...]:
    dv, dw1, db1, dw2, db2 = mlp_backward(*ctx.saved_tensors, dout.contiguous())
    return dv, None, dw1, db1, dw2, db2


mlp_op.register_autograd(_mlp_grad, setup_context=_mlp_setup)


@torch.library.custom_op("hsn::csr_messages", mutates_args=(), device_types="cuda")
def messages_op(
    h: Tensor, indptr: Tensor, indices: Tensor, a: Tensor, V: Tensor, idx: Tensor
) -> Tensor:
    return _run(4, [h, indptr, indices, a, V, idx], _dims(h, idx, edges=a.numel()))[0]


@messages_op.register_fake
def _messages_fake(
    h: Tensor, indptr: Tensor, indices: Tensor, a: Tensor, V: Tensor, idx: Tensor
) -> Tensor:
    return h.new_empty((*idx.shape, h.shape[2]))


@torch.library.custom_op("hsn::csr_messages_backward", mutates_args=(), device_types="cuda")
def messages_backward(
    h: Tensor, indptr: Tensor, indices: Tensor, a: Tensor, V: Tensor, idx: Tensor, dm: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    dh, da, dV = _run(5, [h, indptr, indices, a, V, idx, dm], _dims(h, idx, edges=a.numel()))
    return dh, da, dV


@messages_backward.register_fake
def _messages_bwd_fake(
    h: Tensor, indptr: Tensor, indices: Tensor, a: Tensor, V: Tensor, idx: Tensor, dm: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    return torch.empty_like(h), torch.empty_like(a), torch.empty_like(V)


def _messages_setup(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
    ctx.save_for_backward(*inputs)


def _messages_grad(ctx: Any, dm: Tensor) -> tuple[Any, ...]:
    dh, da, dV = messages_backward(*ctx.saved_tensors, dm.contiguous())
    return dh, None, None, da, dV, None


messages_op.register_autograd(_messages_grad, setup_context=_messages_setup)


@torch.library.custom_op(
    "hsn::gated_delta_update", mutates_args=("h", "q", "age"), device_types="cuda"
)
def update_inplace(
    h: Tensor, q: Tensor, age: Tensor, idx: Tensor, valid: Tensor, c: Tensor, b: Tensor
) -> Tensor:
    if any(t.requires_grad for t in (h, q, c, b)) and torch.is_grad_enabled():
        raise RuntimeError("use the functional update for training")
    return _run(6, [h, q, age, idx, valid, c, b], _dims(h, idx))[0]


@update_inplace.register_fake
def _update_inplace_fake(
    h: Tensor, q: Tensor, age: Tensor, idx: Tensor, valid: Tensor, c: Tensor, b: Tensor
) -> Tensor:
    return torch.empty_like(c)


@torch.library.custom_op("hsn::gated_delta_update_functional", mutates_args=(), device_types="cuda")
def update_op(
    h: Tensor, q: Tensor, age: Tensor, idx: Tensor, valid: Tensor, c: Tensor, b: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    hn, qn, an = h.clone(), q.clone(), age.clone()
    old = _run(6, [hn, qn, an, idx, valid, c, b], _dims(h, idx))[0]
    return hn, qn, an, old


@update_op.register_fake
def _update_fake(
    h: Tensor, q: Tensor, age: Tensor, idx: Tensor, valid: Tensor, c: Tensor, b: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return torch.empty_like(h), torch.empty_like(q), torch.empty_like(age), torch.empty_like(c)


@torch.library.custom_op("hsn::gated_delta_update_state", mutates_args=(), device_types="cuda")
def state_update_op(
    h: Tensor, q: Tensor, age: Tensor, idx: Tensor, valid: Tensor, c: Tensor, b: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    hn, qn, an, old = _run(8, [h, q, age, idx, valid, c, b], _dims(h, idx))
    return hn, qn, an, old


@state_update_op.register_fake
def _state_update_fake(
    h: Tensor, q: Tensor, age: Tensor, idx: Tensor, valid: Tensor, c: Tensor, b: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return torch.empty_like(h), torch.empty_like(q), torch.empty_like(age), torch.empty_like(c)


@torch.library.custom_op("hsn::gated_delta_update_backward", mutates_args=(), device_types="cuda")
def update_backward(
    idx: Tensor, valid: Tensor, c: Tensor, b: Tensor, old: Tensor, G: Tensor, g: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dh, dq, dc, db = _run(7, [idx, valid, c, b, old, G, g], _dims(G, idx))
    return dh, dq, dc, db


@update_backward.register_fake
def _update_bwd_fake(
    idx: Tensor, valid: Tensor, c: Tensor, b: Tensor, old: Tensor, G: Tensor, g: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return torch.empty_like(G), torch.empty_like(g), torch.empty_like(c), torch.empty_like(b)


def _update_setup(ctx: Any, inputs: tuple[Any, ...], output: tuple[Tensor, ...]) -> None:
    _, _, _, idx, valid, c, b = inputs
    ctx.save_for_backward(idx, valid, c, b, output[3])
    ctx.mark_non_differentiable(output[2], output[3])


def _update_grad(ctx: Any, G: Tensor, g: Tensor, dage: Tensor, dold: Tensor) -> tuple[Any, ...]:
    dh, dq, dc, db = update_backward(*ctx.saved_tensors, G.contiguous(), g.contiguous())
    return dh, dq, None, None, None, dc, db


update_op.register_autograd(_update_grad, setup_context=_update_setup)
state_update_op.register_autograd(_update_grad, setup_context=_update_setup)


def _supported(t: Tensor) -> bool:
    return t.is_cuda and t.dtype in (torch.float32, torch.float16, torch.bfloat16)


def _i32(t: Tensor) -> Tensor:
    if t.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be int32 or int64")
    # Clamping prevents a huge int64 from wrapping into a valid int32 index.
    return t.clamp(-1, 2**31 - 1).to(torch.int32).contiguous()


def _compute_dtype(t: Tensor) -> torch.dtype:
    if torch.is_autocast_enabled("cuda") and t.dtype == torch.float32:
        return torch.get_autocast_dtype("cuda")
    return t.dtype


def gather_concat(u: Tensor | None, eps: Tensor, h: Tensor, m: Tensor, cell_idx: Tensor) -> Tensor:
    from hsn.kernels import native

    # u=None (model.ablate_shared_u) is the 3d cell input [eps; h; m]: the P2.9 gather kernel's
    # ABI is [B, K, 4d] with u, so that input keeps the native path.
    if u is None or not _supported(h):
        return native.gather_concat(u, eps, h, m, cell_idx)
    if u.dtype != h.dtype or eps.dtype != h.dtype:
        raise ValueError("u, eps and h must have the same state dtype")
    result = gather_op(
        u.contiguous(), eps.contiguous(), h.contiguous(), m.to(h.dtype).contiguous(), _i32(cell_idx)
    )
    return cast(Tensor, result)


def grouped_unit_mlp(
    v: Tensor,
    unit_idx: Tensor,
    bank: UnitBank,
    *,
    static: bool = False,
    w1_flat: Tensor | None = None,
) -> Tensor:
    from hsn.kernels import native

    # Native autocast rounds both activations and intermediate gradients in
    # reduced precision. The raw ABI keeps fp32 intermediates, which can exceed
    # native's gradient tolerance at model-sized pair counts. Preserve native's
    # mixed-precision contract until a matching autocast kernel is specified.
    # The raw ABI fixes the unit input at [P, 4d] and W1 at [M, 4d, H]: a bank of another input
    # width (model.ablate_shared_u: 3d) keeps the native path.
    if (
        not _supported(v)
        or bank.activation_name != "gelu"
        or torch.is_autocast_enabled("cuda")
        or bank.in_dim != 4 * bank.d
    ):
        return native.grouped_unit_mlp(v, unit_idx, bank, static=static, w1_flat=w1_flat)
    if not static:
        lo, hi = int(unit_idx.min()), int(unit_idx.max())
        if lo < 0 or hi >= bank.n_units:
            raise IndexError("unit_idx outside the unit bank")
    dtype = _compute_dtype(v)
    # Cast master parameters per call. Never cache reduced-precision leaves across a window:
    # their backward would accumulate all steps in reduced precision before reaching the masters.
    w1 = (
        bank.W1
        if w1_flat is None
        else w1_flat.reshape(bank.in_dim, bank.n_units, bank.hidden).permute(1, 0, 2)
    )
    out, _ = mlp_op(
        v.to(dtype).contiguous(),
        _i32(unit_idx),
        w1.to(dtype).contiguous(),
        bank.b1.to(dtype).contiguous(),
        bank.W2.to(dtype).contiguous(),
        bank.b2.to(dtype).contiguous(),
    )
    return cast(Tensor, out)


def csr_messages(
    h: Tensor,
    graph: CellGraph,
    V: nn.Linear,
    cell_idx: Tensor,
    *,
    weights: Tensor | None = None,
    validate: bool = True,
) -> Tensor:
    from hsn.kernels import native

    # Native autocast rounds the projection before its fp32 reduction. The
    # single-dtype ABI cannot preserve that mixed-precision output/gradient
    # contract (notably bf16 edge-weight gradients), so retain native here.
    if not _supported(h) or torch.is_autocast_enabled("cuda"):
        return native.csr_messages(h, graph, V, cell_idx, weights=weights, validate=validate)
    B, _, _ = graph._check_h(h)
    graph._check_cell_idx(cell_idx, B, check_values=validate)
    if V.bias is not None:
        raise ValueError("CSR messages require a bias-free projection")
    a = graph.edge_weights() if weights is None else weights.reshape(-1)[graph._edge_pos]
    dtype = _compute_dtype(h)
    result = messages_op(
        h.to(dtype).contiguous(),
        _i32(graph.indptr),
        _i32(graph.indices),
        a.to(dtype).contiguous(),
        V.weight.to(dtype).contiguous(),
        _i32(cell_idx),
    )
    return cast(Tensor, result)


def gated_delta_update(
    state: CellGraphState,
    cell_idx: Tensor,
    c: Tensor,
    b: Tensor,
    valid: Tensor,
    *,
    validate: bool = True,
) -> CellGraphState:
    from hsn.kernels import native
    from hsn.model.state import CellGraphState

    # The single-dtype ABI cannot express native's mixed-dtype (1-b) rounding. Preserve that
    # path until a measured, separately specified mixed-dtype kernel is available.
    if not _supported(state.h) or c.dtype != state.dtype or b.dtype != state.dtype:
        return native.gated_delta_update(state, cell_idx, c, b, valid, validate=validate)
    native._check_bool("valid", valid, tuple(cell_idx.shape), state.device)
    if validate:
        native._validate_selection(cell_idx, valid, state.n_cells)
    # The private functional kernel preserves int64 ages and writes fresh state
    # directly, including unchanged cells; the public in-place ABI is unchanged.
    hn, qn, age, _ = state_update_op(
        state.h.contiguous(),
        state.q.contiguous(),
        state.age.contiguous(),
        _i32(cell_idx),
        valid.to(torch.int32).contiguous(),
        c.contiguous(),
        b.contiguous(),
    )
    return CellGraphState._make(hn, qn, age, state.pos)
