"""P2.10: the window kernels of the split window (:mod:`hsn.train.split`) on CUDA.

``build/hsn_window.fatbin`` (``kernels/cuda/hsn_window.cu``) is loaded through a small generic
launcher (``build/hsn_window_py*.so``, ``python/hsn/kernels/window_ext.cpp``); both are built by
``scripts/build_window_kernels.py`` and are independent of the four P2.9 ops (their fatbin,
extension, ABI and tests are untouched). Import never initialises CUDA; :func:`prepare` loads the
fatbin on a device. fp32 state only: every wrapper returns ``None`` for inputs it does not
handle, and the caller keeps its PyTorch path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from hsn.model.cell import PDGCell
    from hsn.model.state import CellGraphState

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "build"

_prepared: set[int] = set()
#: Dynamic shared memory a block may ask for without opting in to more (48 KiB on every
#: supported architecture).
_MAX_SHARED_BYTES = 48 * 1024


def available() -> bool:
    """Artifact presence only: never initialises CUDA or loads a library."""
    return (
        (BUILD / "hsn_window.fatbin").is_file()
        and (BUILD / "window_build.json").is_file()
        and any(BUILD.glob("hsn_window_py*.so"))
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _extension() -> ModuleType:
    if not available():
        raise RuntimeError(
            "P2.10 window kernels missing; run scripts/build_window_kernels.py in this worktree"
        )
    manifest = json.loads((BUILD / "window_build.json").read_text())
    if manifest.get("torch") != torch.__version__:
        raise RuntimeError("the window-kernel extension must be rebuilt for this PyTorch version")
    if _sha(BUILD / "hsn_window.fatbin") != manifest["fatbin_sha256"]:
        raise RuntimeError("the window-kernel fatbin differs from its build manifest")
    for key in ("device_sources", "host_sources"):
        for name, digest in manifest[key].items():
            if _sha(ROOT / name) != digest:
                raise RuntimeError(f"window-kernel source changed since the build: {name}")
    libraries = list(BUILD.glob("hsn_window_py*.so"))
    if len(libraries) != 1:
        raise RuntimeError("expected exactly one hsn_window_py extension; clean stale artifacts")
    spec = importlib.util.spec_from_file_location("hsn_window_py", libraries[0])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(device: int = 0) -> None:
    """Load the window fatbin on ``device`` (idempotent)."""
    _extension().prepare(str(BUILD / "hsn_window.fatbin"), device)
    _prepared.add(device)


def prepared(device: torch.device) -> bool:
    return device.type == "cuda" and (device.index or 0) in _prepared


def _f32(t: Tensor) -> Tensor:
    return t.contiguous()


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
) -> tuple[Tensor, Tensor] | None:
    """:func:`hsn.train.split.recurrence_backward` as one kernel launch (one block per batch
    row), or ``None`` when the inputs are not fp32 tensors on a prepared CUDA device."""
    if not prepared(c.device) or c.dtype != torch.float32 or cells.dtype != torch.int64:
        return None
    B, T, K, d = (int(n) for n in c.shape)
    N = int(n_cells)
    shared_bytes = 4 * (N + 1) * d  # the row's cell and summary cotangents
    if shared_bytes > _MAX_SHARED_BYTES:
        return None
    dc = torch.empty_like(c, memory_format=torch.contiguous_format)
    db = torch.empty_like(c, memory_format=torch.contiguous_format)
    absent = dc  # an unread placeholder address for the cotangents a window did not produce
    tensors = [
        _f32(c),
        _f32(b),
        _f32(h_old),
        cells.contiguous(),
        valid.contiguous(),
        reset.contiguous(),
        absent if d_q_pre is None else _f32(d_q_pre),
        absent if d_q_post is None else _f32(d_q_post),
        absent if d_h_sel is None else _f32(d_h_sel),
        dc,
        db,
    ]
    for cot in (d_q_pre, d_q_post, d_h_sel):
        if cot is not None and cot.dtype != torch.float32:
            return None
    ints = [B, T, K, d, N, d_q_pre is not None, d_q_post is not None, d_h_sel is not None]
    threads = 256
    _extension().launch(
        "hsn_win_recurrence", B, threads, shared_bytes, tensors, [int(v) for v in ints], []
    )
    return dc, db


# --------------------------------------------------------------------------------------------------
# The fused cell round of the forward sweep
# --------------------------------------------------------------------------------------------------

_THREADS = 256


def cell_round_error(cell: PDGCell) -> str | None:
    """Why the fused cell round cannot run this cell, or ``None`` when it can."""
    d = cell.d
    if not available():
        return "the window kernels are not built (scripts/build_window_kernels.py)"
    if cell.cell_norm != "rms":
        return f"model.cell_norm={cell.cell_norm!r}: the fused round implements the RMS norm"
    if cell.units.activation_name != "gelu":
        return f"model.activation={cell.units.activation_name!r}: the fused round implements GELU"
    if cell.disabled_cells:
        return "model.disabled_cells is a test hook of the native round"
    if cell.V.weight.dtype != torch.float32:
        return f"fp32 master weights only, got {cell.V.weight.dtype}"
    H, K = cell.units.hidden, 1
    largest = max(
        4 * 16 * (H + _THREADS),  # output layer: sixteen activations and the partial sums
        4 * (d * d + 16 * d + 64 * d + 16),  # cell front: V, sixteen inputs
        4 * (d * d + 4 * K * d + K),  # cell back: W_out (K is checked per call)
    )
    if d > 128:
        return f"model.d={d}: the gate's rows are read sixteen warps' worth at a time (d <= 128)"
    if largest > _MAX_SHARED_BYTES:
        return f"model.d={d}, model.hidden={H}: a weight tile does not fit a block's shared memory"
    return None


@torch.library.custom_op("hsn::window_cell_round", mutates_args=(), device_types="cuda")
def cell_round_op(
    u: Tensor,
    eps: Tensor,
    h: Tensor,
    q: Tensor,
    age: Tensor,
    cells: Tensor,
    units: Tensor,
    valid: Tensor,
    alpha: Tensor,
    V: Tensor,
    edge_w: Tensor,
    pad_src: Tensor,
    norm_w: Tensor,
    gate_w: Tensor,
    gate_b: Tensor,
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    W_out: Tensor,
    cand_w: Tensor,
    cand_norm: bool,
    norm_eps: float,
    cand_eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The sparse round and the tick, forward only: ``(h_new, q_new, age_new, h_sel)``
    (``kernels/cuda/hsn_window.cu``, "The fused cell round"). Inputs are contiguous fp32 /
    int64 / bool tensors on one prepared CUDA device; not differentiable (the split window's
    sweep runs it under ``torch.no_grad()``)."""
    B, N, d = (int(n) for n in h.shape)
    K = int(cells.shape[1])
    M, _, H = (int(n) for n in W1.shape)
    D = int(edge_w.shape[1])
    P = B * K
    v = h.new_empty((P, 4 * d))
    gate = h.new_empty((P, d))
    act = h.new_empty((P, H))
    pairs = torch.empty(P, dtype=torch.int32, device=h.device)
    offsets = torch.empty(M + 1, dtype=torch.int32, device=h.device)
    h_new = torch.empty_like(h)
    q_new = torch.empty_like(q)
    age_new = torch.empty_like(age)
    h_sel = h.new_empty((B, K, d))
    out = h.new_empty((P, d))
    tensors = [
        u, eps, h, q, age, cells, units, valid, alpha, V, edge_w, pad_src, norm_w, gate_w,
        gate_b, W1, b1, W2, b2, W_out, cand_w, v, gate, act, pairs, offsets, h_new, q_new,
        age_new, h_sel, out,
    ]  # fmt: skip
    ints = [B, N, d, K, M, H, D, int(cand_norm)]
    floats = [norm_eps, cand_eps]
    ext = _extension()
    segments = (P + 15) // 16  # a unit's pair list is walked sixteen pairs per block
    in_dim = 4 * d
    ext.launch(
        "hsn_win_cell_front",
        segments,
        _THREADS,
        4 * (d * d + 16 * d + 16 * in_dim + 16),
        tensors,
        ints,
        floats,
    )
    ext.launch("hsn_win_routes", (M + 7) // 8, _THREADS, 0, tensors, ints, floats)
    ext.launch(
        "hsn_win_mlp_hidden",
        M * ((H + 255) // 256) * segments,
        _THREADS,
        4 * 16 * in_dim,
        tensors,
        ints,
        floats,
    )
    ext.launch(
        "hsn_win_mlp_out",
        M * segments,
        _THREADS,
        4 * 16 * (H + _THREADS),
        tensors,
        ints,
        floats,
    )
    ext.launch("hsn_win_cell_back", B, _THREADS, 4 * (d * d + 4 * K * d + K), tensors, ints, floats)
    return h_new, q_new, age_new, h_sel


@cell_round_op.register_fake
def _cell_round_fake(
    u: Tensor,
    eps: Tensor,
    h: Tensor,
    q: Tensor,
    age: Tensor,
    cells: Tensor,
    units: Tensor,
    valid: Tensor,
    alpha: Tensor,
    V: Tensor,
    edge_w: Tensor,
    pad_src: Tensor,
    norm_w: Tensor,
    gate_w: Tensor,
    gate_b: Tensor,
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    W_out: Tensor,
    cand_w: Tensor,
    cand_norm: bool,
    norm_eps: float,
    cand_eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(h),
        torch.empty_like(q),
        torch.empty_like(age),
        h.new_empty((*cells.shape, h.shape[2])),
    )


def cell_round(
    cell: PDGCell,
    u: Tensor,
    eps: Tensor,
    state: CellGraphState,
    cells: Tensor,
    units: Tensor,
    valid: Tensor,
    edge_weights: Tensor,
) -> tuple[CellGraphState, Tensor]:
    """:meth:`PDGCell.step_sparse` and the tick of one position as the fused op: the state
    after the position and the selected slots' updated states ``[B, K, d]`` (cell 0 on padded
    slots). ``cells`` / ``units`` / ``valid`` are a hard decision (``alpha`` is 1 on its valid
    slots), ``edge_weights`` the window's padded edge weights. fp32 state, no autograd."""
    from hsn.model.state import CellGraphState

    cand_norm = cell.cand_norm_kind != "none"
    norm_w = cell.norm.weight
    h_new, q_new, age_new, h_sel = cell_round_op(
        u.contiguous(),
        eps.contiguous(),
        state.h.contiguous(),
        state.q.contiguous(),
        state.age.contiguous(),
        cells.contiguous(),
        units.contiguous(),
        valid.contiguous(),
        valid.to(torch.float32).contiguous(),
        cell.V.weight,
        edge_weights.to(torch.float32).contiguous(),
        cell.graph._pad_src,
        norm_w,
        cell.gate.weight,
        cell.gate.bias,
        cell.units.W1,
        cell.units.b1,
        cell.units.W2,
        cell.units.b2,
        cell.W_out.weight,
        cell.cand_norm.weight if cand_norm else norm_w,
        cand_norm,
        float(cast(nn.RMSNorm, cell.norm).eps or 0.0),
        float(cast(nn.RMSNorm, cell.cand_norm).eps or 0.0) if cand_norm else 0.0,
    )
    return CellGraphState._make(h=h_new, q=q_new, age=age_new, pos=state.pos + 1), h_sel
