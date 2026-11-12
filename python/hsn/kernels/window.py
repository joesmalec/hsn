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
#: Threads of a row's block in the selection kernel (one per logit of the row when they fit:
#: the kernel then reduces a cell's units by warp shuffles; never fewer than one per cell).
_SELECT_THREADS = 256
#: Threads of a row's block in the end-of-round kernel (a multiple of 32; one per slot element
#: when they fit, and the row's ``N * d`` states in ``N * d / threads`` steps).
_UPDATE_THREADS = 512


def _select_shared(n_cells: int, n_units: int, k: int) -> int:
    return 4 * (4 * n_cells + n_units + 4 * k + 1 + 3 * ((n_cells + 31) // 32))


def _update_shared(n_cells: int, d: int, k: int) -> int:
    return 4 * (2 * k + 1 + n_cells + k * ((d + 31) // 32) + 2 * k * d)


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


def dense_recurrence_backward(
    c: Tensor,
    b: Tensor,
    h_old: Tensor,
    reset: Tensor,
    n_cells: int,
    d_q_pre: Tensor | None,
    d_q_post: Tensor | None,
) -> tuple[Tensor, Tensor] | None:
    """:func:`hsn.train.split.dense_recurrence_backward` as one kernel launch (one block per
    dense-fallback row), or ``None`` when the inputs are not fp32 tensors on a prepared CUDA
    device."""
    if not prepared(c.device) or c.dtype != torch.float32:
        return None
    Bd, T, N, d = (int(n) for n in c.shape)
    shared_bytes = 4 * (N + 1) * d
    if shared_bytes > _MAX_SHARED_BYTES or N != int(n_cells):
        return None
    for cot in (d_q_pre, d_q_post):
        if cot is not None and cot.dtype != torch.float32:
            return None
    dc = torch.empty_like(c, memory_format=torch.contiguous_format)
    db = torch.empty_like(c, memory_format=torch.contiguous_format)
    absent = dc
    tensors = [
        _f32(c),
        _f32(b),
        _f32(h_old),
        reset.contiguous(),
        absent if d_q_pre is None else _f32(d_q_pre),
        absent if d_q_post is None else _f32(d_q_post),
        dc,
        db,
    ]
    ints = [Bd, T, N, d, int(d_q_pre is not None), int(d_q_post is not None)]
    _extension().launch("hsn_win_recurrence_dense", Bd, 256, shared_bytes, tensors, ints, [])
    return dc, db


# --------------------------------------------------------------------------------------------------
# The fused cell round of the forward sweep
# --------------------------------------------------------------------------------------------------

_THREADS = 256


def cell_round_error(cell: PDGCell) -> str | None:
    """Why the fused cell round cannot run this cell, or ``None`` when it can."""
    d = cell.d
    if cell.ablate_shared_u:
        return (
            "model.ablate_shared_u: the fused round builds the 4d cell input [u; eps; h; m] in "
            "its kernel (the fused window does not use it: its products are PDGCell's own)"
        )
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


@torch.library.custom_op("hsn::window_select", mutates_args=(), device_types="cuda")
def select_op(
    logits: Tensor,
    age: Tensor,
    cost: Tensor,
    k: int,
    age_max: int,
    probes: bool,
    lambda_cost: float,
    cell_overhead: float,
    shared_cost: float,
    limit: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The hard decision of one position, forward only: ``(cells, units, valid, theta,
    refreshed)`` (``kernels/cuda/hsn_window.cu``, ``hsn_win_select``). ``logits [B, N, M]`` fp32,
    ``age [B, N]`` int64 and ``cost [M]`` fp32 are contiguous tensors on one prepared CUDA
    device."""
    B, N, M = (int(n) for n in logits.shape)
    device = logits.device
    cells = torch.empty((B, k), dtype=torch.int64, device=device)
    units = torch.empty((B, k), dtype=torch.int64, device=device)
    valid = torch.empty((B, k), dtype=torch.bool, device=device)
    theta = torch.empty((B,), dtype=torch.float32, device=device)
    refreshed = torch.empty((B, k), dtype=torch.bool, device=device)
    _extension().launch(
        "hsn_win_select",
        B,
        _SELECT_THREADS,
        _select_shared(N, M, k),
        [logits, age, cost, cells, units, valid, theta, refreshed],
        [B, N, M, k, age_max, int(probes), 0, 1, 1],
        [lambda_cost, cell_overhead, shared_cost, limit],
    )
    return cells, units, valid, theta, refreshed


@select_op.register_fake
def _select_fake(
    logits: Tensor,
    age: Tensor,
    cost: Tensor,
    k: int,
    age_max: int,
    probes: bool,
    lambda_cost: float,
    cell_overhead: float,
    shared_cost: float,
    limit: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    B = logits.shape[0]
    return (
        logits.new_empty((B, k), dtype=torch.int64),
        logits.new_empty((B, k), dtype=torch.int64),
        logits.new_empty((B, k), dtype=torch.bool),
        logits.new_empty((B,), dtype=torch.float32),
        logits.new_empty((B, k), dtype=torch.bool),
    )


@torch.library.custom_op(
    "hsn::window_select_tape",
    mutates_args=("cells", "units", "valid", "theta", "refreshed"),
    device_types="cuda",
)
def select_tape_op(
    logits: Tensor,
    age: Tensor,
    cost: Tensor,
    cells: Tensor,
    units: Tensor,
    valid: Tensor,
    theta: Tensor,
    refreshed: Tensor,
    t: int,
    age_max: int,
    probes: bool,
    lambda_cost: float,
    cell_overhead: float,
    shared_cost: float,
    limit: float,
) -> None:
    """:func:`select_op` written into a tape: the decision of position ``t`` from ``logits
    [B, N, M]`` and the ages ``age[:, t]`` (``[B, n + 1, N]``) into record ``t`` of ``cells`` /
    ``units`` ``[B, n, K]`` int64, ``valid`` / ``refreshed`` ``[B, n, K]`` bool and ``theta
    [B, n]`` (the same kernel; no other record is touched). Contiguous tensors on one prepared
    CUDA device."""
    B, N, M = (int(n) for n in logits.shape)
    n, k = int(cells.shape[1]), int(cells.shape[2])
    _extension().launch(
        "hsn_win_select",
        B,
        _SELECT_THREADS,
        _select_shared(N, M, k),
        [logits, age, cost, cells, units, valid, theta, refreshed],
        [B, N, M, k, age_max, int(probes), t, n, int(age.shape[1])],
        [lambda_cost, cell_overhead, shared_cost, limit],
    )


@select_tape_op.register_fake
def _select_tape_fake(
    logits: Tensor,
    age: Tensor,
    cost: Tensor,
    cells: Tensor,
    units: Tensor,
    valid: Tensor,
    theta: Tensor,
    refreshed: Tensor,
    t: int,
    age_max: int,
    probes: bool,
    lambda_cost: float,
    cell_overhead: float,
    shared_cost: float,
    limit: float,
) -> None:
    return None


@torch.library.custom_op(
    "hsn::window_select_fold",
    mutates_args=("cells", "units", "valid", "theta", "refreshed"),
    device_types="cuda",
)
def select_fold_op(
    scored: Tensor,
    age: Tensor,
    cost: Tensor,
    cells: Tensor,
    units: Tensor,
    valid: Tensor,
    theta: Tensor,
    refreshed: Tensor,
    t: int,
    n_units: int,
    age_max: int,
    probes: bool,
    lambda_cost: float,
    cell_overhead: float,
    shared_cost: float,
    limit: float,
) -> None:
    """:func:`select_tape_op` on the scorer's matrix product as it is (P2.10b, the folded
    sweep): ``scored [B * N, W]`` in fp32 or bf16 holds the logits in its first ``n_units``
    columns, which the kernel reads in place: no slice copy, no cast. Contiguous tensors on one
    prepared CUDA device."""
    rows, width = int(scored.shape[0]), int(scored.shape[1])
    B = int(cells.shape[0])
    N = rows // B
    n, k = int(cells.shape[1]), int(cells.shape[2])
    half = scored.dtype == torch.bfloat16
    _extension().launch(
        "hsn_win_select",
        B,
        _SELECT_THREADS,
        _select_shared(N, n_units, k),
        [scored, age, cost, cells, units, valid, theta, refreshed],
        [B, N, n_units, k, age_max, int(probes), t, n, int(age.shape[1]), width, int(half)],
        [lambda_cost, cell_overhead, shared_cost, limit],
    )


@select_fold_op.register_fake
def _select_fold_fake(
    scored: Tensor,
    age: Tensor,
    cost: Tensor,
    cells: Tensor,
    units: Tensor,
    valid: Tensor,
    theta: Tensor,
    refreshed: Tensor,
    t: int,
    n_units: int,
    age_max: int,
    probes: bool,
    lambda_cost: float,
    cell_overhead: float,
    shared_cost: float,
    limit: float,
) -> None:
    return None


def select_unsupported(n_cells: int, k: int, n_units: int = 1) -> str | None:
    """Why :func:`select` cannot take a model of this shape (``None``: it can)."""
    if n_cells > _SELECT_THREADS:
        return f"model.n_cells={n_cells}: one thread per cell of a row ({_SELECT_THREADS})"
    if k > n_cells:
        return f"k_select={k} exceeds model.n_cells={n_cells}"
    if _select_shared(n_cells, n_units, k) > _MAX_SHARED_BYTES:
        return f"model.n_cells={n_cells}: the row's scores do not fit a block's shared memory"
    return None


def select(
    logits: Tensor,
    age: Tensor,
    cost_table: Tensor,
    *,
    k: int,
    age_max: int,
    probes: bool,
    lambda_cost: float,
    cell_overhead: float,
    shared_cost: float,
    limit: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """:meth:`hsn.model.router.Router.select` for the forward sweep (mode ``topk``, the hard
    decision, no exploration, the refresh probes when ``probes``) as one kernel: ``(cells,
    units, valid, theta, refreshed)`` with the router's values, tie rules included (equal
    scores and equal ages in index order). ``limit`` is ``budget * (1 + BUDGET_SLACK)``."""
    decision: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] = select_op(
        logits.to(torch.float32).contiguous(),
        age.contiguous(),
        cost_table.to(torch.float32).contiguous(),
        k,
        age_max,
        probes,
        lambda_cost,
        cell_overhead,
        shared_cost,
        limit,
    )
    return decision


@torch.library.custom_op("hsn::window_cell_update", mutates_args=(), device_types="cuda")
def cell_update_op(
    y: Tensor,
    gate: Tensor,
    h: Tensor,
    q: Tensor,
    age: Tensor,
    pos: Tensor,
    cells: Tensor,
    valid: Tensor,
    cand_w: Tensor,
    cand_norm: bool,
    cand_eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The end of one position's round, forward only: ``(h, q, age, pos, h_sel, h_max)``
    (``kernels/cuda/hsn_window.cu``, ``hsn_win_cell_update``). Contiguous fp32 / int64 / bool
    tensors on one prepared CUDA device."""
    B, N, d = (int(n) for n in h.shape)
    K = int(cells.shape[1])
    h_next = torch.empty_like(h)
    q_next = torch.empty_like(q)
    age_next = torch.empty_like(age)
    pos_next = torch.empty_like(pos)
    h_sel = h.new_empty((B, K, d))
    h_max = h.new_empty((B,))
    _extension().launch(
        "hsn_win_cell_update",
        B,
        _UPDATE_THREADS,
        _update_shared(N, d, K),
        [
            y,
            gate,
            h,
            q,
            age,
            pos,
            cells,
            valid,
            cand_w,
            h_next,
            q_next,
            age_next,
            pos_next,
            h_sel,
            h_max,
        ],  # fmt: skip
        [B, N, d, K, int(cand_norm), 0, 1, 1, 0, 0, 1, 0],
        [cand_eps],
    )
    return h_next, q_next, age_next, pos_next, h_sel, h_max


@cell_update_op.register_fake
def _cell_update_fake(
    y: Tensor,
    gate: Tensor,
    h: Tensor,
    q: Tensor,
    age: Tensor,
    pos: Tensor,
    cells: Tensor,
    valid: Tensor,
    cand_w: Tensor,
    cand_norm: bool,
    cand_eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(h),
        torch.empty_like(q),
        torch.empty_like(age),
        torch.empty_like(pos),
        h.new_empty((*cells.shape, h.shape[2])),
        h.new_empty((h.shape[0],)),
    )


@torch.library.custom_op(
    "hsn::window_update_tape",
    mutates_args=("tape_h", "tape_q", "tape_age", "tape_hsel", "h_max", "qs"),
    device_types="cuda",
)
def update_tape_op(
    y: Tensor,
    gate: Tensor,
    tape_h: Tensor,
    tape_q: Tensor,
    tape_age: Tensor,
    cells: Tensor,
    valid: Tensor,
    cand_w: Tensor,
    reset: Tensor,
    tape_hsel: Tensor,
    h_max: Tensor,
    qs: Tensor,
    t: int,
    cand_norm: bool,
    cand_eps: float,
) -> None:
    """:func:`cell_update_op` on a tape: the round of position ``t`` reads record ``t`` of the
    state tape (``tape_h [B, n + 1, N, d]``, ``tape_q [B, n + 1, d]``, ``tape_age
    [B, n + 1, N]``) and of the decision (``cells`` / ``valid`` ``[B, n, K]``) and writes record
    ``t + 1`` of the state tape: the state after the position, zeroed on the rows whose
    ``reset[:, t + 1]`` is set (the pre-round state of the next position; the last record,
    ``t + 1 == n``, is never reset). Record ``t`` of ``tape_hsel [B, n, K, d]``, ``h_max
    [B, n]`` and ``qs [B, n, d]`` (the summary after the position) are the values before that
    reset. The same kernel as :func:`cell_update_op`; contiguous tensors on one prepared CUDA
    device, ``reset [B, n]`` bool."""
    B, n_state, N, d = (int(n) for n in tape_h.shape)
    n, K = int(cells.shape[1]), int(cells.shape[2])
    _extension().launch(
        "hsn_win_cell_update",
        B,
        _UPDATE_THREADS,
        _update_shared(N, d, K),
        [
            y,
            gate,
            tape_h,
            tape_q,
            tape_age,
            tape_age,
            cells,
            valid,
            cand_w,
            tape_h,
            tape_q,
            tape_age,
            tape_age,
            tape_hsel,
            h_max,
            reset,
            qs,
        ],  # fmt: skip
        [B, N, d, K, int(cand_norm), t, n_state, n, t + 1, int(t + 1 < n), 0, 1],
        [cand_eps],
    )


@update_tape_op.register_fake
def _update_tape_fake(
    y: Tensor,
    gate: Tensor,
    tape_h: Tensor,
    tape_q: Tensor,
    tape_age: Tensor,
    cells: Tensor,
    valid: Tensor,
    cand_w: Tensor,
    reset: Tensor,
    tape_hsel: Tensor,
    h_max: Tensor,
    qs: Tensor,
    t: int,
    cand_norm: bool,
    cand_eps: float,
) -> None:
    return None


@torch.library.custom_op(
    "hsn::window_update_fold",
    mutates_args=("tape_h", "tape_q", "tape_age", "tape_hsel", "h_max", "qs", "x"),
    device_types="cuda",
)
def update_fold_op(
    y: Tensor,
    gate: Tensor,
    tape_h: Tensor,
    tape_q: Tensor,
    tape_age: Tensor,
    cells: Tensor,
    valid: Tensor,
    cand_w: Tensor,
    reset: Tensor,
    tape_hsel: Tensor,
    h_max: Tensor,
    qs: Tensor,
    x: Tensor,
    z: Tensor,
    context: Tensor,
    t: int,
    cand_norm: bool,
    cand_eps: float,
) -> None:
    """:func:`update_tape_op` on the round's matrix products as they are (P2.10b, the folded
    sweep): ``y [B, K, d]`` (``W_out``'s output) and ``gate [B * K, d]`` (the gate's
    pre-activation: the rider columns of the first layer's output, a strided view of the wider
    product) in fp32 or bf16, read without a cast or a copy. The kernel also writes the next
    position's input of the first product: ``x [n, B, d_z + d + d_c]`` (fp32 or bf16) holds
    that input for every position of the block, and record ``t + 1`` (when there is one)
    becomes ``[z; q; context]`` of position ``t + 1`` from ``z [B, n, d_z]``, ``context
    [B, n, d_c]`` and ``q`` after the round and the next reset, each value rounded to ``x``'s
    dtype (the cast of the assembled row). Tensors on one prepared CUDA device, contiguous
    except ``gate``."""
    B, n_state, N, d = (int(n) for n in tape_h.shape)
    n, K = int(cells.shape[1]), int(cells.shape[2])
    d_z, d_c = int(z.shape[2]), int(context.shape[2])
    if tuple(x.shape) != (n, B, d_z + d + d_c):
        raise ValueError(f"x must be [n, B, d_z + d + d_c], got {tuple(x.shape)}")
    ints = [B, N, d, K, int(cand_norm), t, n_state, n, t + 1, int(t + 1 < n), 0, 1]
    ints += [
        int(y.dtype == torch.bfloat16),
        int(gate.dtype == torch.bfloat16),
        int(gate.stride(0)),
        1,
        int(x.dtype == torch.bfloat16),
        d_z,
        int(z.dtype == torch.bfloat16),
        d_c,
        int(context.dtype == torch.bfloat16),
    ]
    _extension().launch_views(
        "hsn_win_cell_update",
        B,
        _UPDATE_THREADS,
        _update_shared(N, d, K),
        [
            y,
            gate,
            tape_h,
            tape_q,
            tape_age,
            tape_age,
            cells,
            valid,
            cand_w,
            tape_h,
            tape_q,
            tape_age,
            tape_age,
            tape_hsel,
            h_max,
            reset,
            qs,
            x,
            z,
            context,
        ],  # fmt: skip
        [1],  # the gate is a view
        ints,
        [cand_eps],
    )


@update_fold_op.register_fake
def _update_fold_fake(
    y: Tensor,
    gate: Tensor,
    tape_h: Tensor,
    tape_q: Tensor,
    tape_age: Tensor,
    cells: Tensor,
    valid: Tensor,
    cand_w: Tensor,
    reset: Tensor,
    tape_hsel: Tensor,
    h_max: Tensor,
    qs: Tensor,
    x: Tensor,
    z: Tensor,
    context: Tensor,
    t: int,
    cand_norm: bool,
    cand_eps: float,
) -> None:
    return None


def cell_update_unsupported(n_cells: int, d: int, k: int) -> str | None:
    """Why :func:`cell_update` cannot take a model of this shape (``None``: it can)."""
    if k > _UPDATE_THREADS:
        return f"k_select={k}: one thread per slot computes its candidate norm"
    if _update_shared(n_cells, d, k) > _MAX_SHARED_BYTES:
        return f"k_select={k}, model.d={d}: the slots' states do not fit a block's shared memory"
    return None


def cell_update(
    cell: PDGCell,
    y: Tensor,
    gate: Tensor,
    state: CellGraphState,
    cells: Tensor,
    valid: Tensor,
) -> tuple[CellGraphState, Tensor, Tensor]:
    """The round of one position from its matrix products on
    (:meth:`hsn.model.cell.PDGCell.sparse_products`): the candidate (its norm, ``tanh``), the
    gate's sigmoid, :func:`hsn.kernels.native.gated_delta_update` and the tick as one kernel.
    Returns the state after the position, the selected slots' cells after the update
    ``[B, K, d]`` (cell 0 on padded slots) and each row's ``max |h|`` ``[B]``. fp32 state, a hard
    decision (``alpha`` 1 on the valid slots), no autograd."""
    from hsn.model.state import CellGraphState

    cand_norm = cell.cand_norm_kind != "none"
    h, q, age, pos, h_sel, h_max = cell_update_op(
        y.contiguous(),
        gate.contiguous(),
        state.h.contiguous(),
        state.q.contiguous(),
        state.age.contiguous(),
        state.pos.contiguous(),
        cells.contiguous(),
        valid.contiguous(),
        cell.cand_norm.weight if cand_norm else cell.norm.weight,
        cand_norm,
        float(cast(nn.RMSNorm, cell.cand_norm).eps or 0.0) if cand_norm else 0.0,
    )
    return CellGraphState._make(h=h, q=q, age=age, pos=pos), h_sel, h_max


def select_tape(
    logits: Tensor,
    tape_age: Tensor,
    cost_table: Tensor,
    cells: Tensor,
    units: Tensor,
    valid: Tensor,
    theta: Tensor,
    refreshed: Tensor,
    *,
    t: int,
    age_max: int,
    probes: bool,
    lambda_cost: float,
    cell_overhead: float,
    shared_cost: float,
    limit: float,
) -> None:
    """:func:`select` for position ``t`` of a block, written into record ``t`` of the block's
    decision tape (``cells`` / ``units`` / ``valid`` / ``refreshed`` ``[B, n, K]``, ``theta
    [B, n]``) from the ages ``tape_age[:, t]`` (``[B, n + 1, N]``): no tensor is made, no copy
    follows."""
    select_tape_op(
        logits.to(torch.float32).contiguous(),
        tape_age,
        cost_table.to(torch.float32).contiguous(),
        cells,
        units,
        valid,
        theta,
        refreshed,
        t,
        age_max,
        probes,
        lambda_cost,
        cell_overhead,
        shared_cost,
        limit,
    )


def update_tape(
    cell: PDGCell,
    y: Tensor,
    gate: Tensor,
    tape_h: Tensor,
    tape_q: Tensor,
    tape_age: Tensor,
    cells: Tensor,
    valid: Tensor,
    reset: Tensor,
    tape_hsel: Tensor,
    h_max: Tensor,
    qs: Tensor,
    *,
    t: int,
) -> None:
    """:func:`cell_update` for position ``t`` of a block on its tapes (:func:`update_tape_op`):
    reads the pre-round state at record ``t`` and writes the next position's pre-round state
    (its reset applied) at record ``t + 1``, in place."""
    cand_norm = cell.cand_norm_kind != "none"
    update_tape_op(
        y.contiguous(),
        gate.contiguous(),
        tape_h,
        tape_q,
        tape_age,
        cells,
        valid,
        cell.cand_norm.weight if cand_norm else cell.norm.weight,
        reset,
        tape_hsel,
        h_max,
        qs,
        t,
        cand_norm,
        float(cast(nn.RMSNorm, cell.cand_norm).eps or 0.0) if cand_norm else 0.0,
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


def select_fold(
    scored: Tensor,
    tape_age: Tensor,
    cost_table: Tensor,
    cells: Tensor,
    units: Tensor,
    valid: Tensor,
    theta: Tensor,
    refreshed: Tensor,
    *,
    t: int,
    n_units: int,
    age_max: int,
    probes: bool,
    lambda_cost: float,
    cell_overhead: float,
    shared_cost: float,
    limit: float,
) -> None:
    """:func:`select_tape` from the scorer's matrix product ``scored [B * N, W]`` as it is
    (:func:`select_fold_op`)."""
    select_fold_op(
        scored,
        tape_age,
        cost_table.to(torch.float32).contiguous(),
        cells,
        units,
        valid,
        theta,
        refreshed,
        t,
        n_units,
        age_max,
        probes,
        lambda_cost,
        cell_overhead,
        shared_cost,
        limit,
    )


def update_fold(
    cell: PDGCell,
    y: Tensor,
    gate: Tensor,
    tape_h: Tensor,
    tape_q: Tensor,
    tape_age: Tensor,
    cells: Tensor,
    valid: Tensor,
    reset: Tensor,
    tape_hsel: Tensor,
    h_max: Tensor,
    qs: Tensor,
    x: Tensor,
    z: Tensor,
    context: Tensor,
    *,
    t: int,
) -> None:
    """:func:`update_tape` from the round's matrix products as they are, with the next
    position's input of the first product written into ``x`` (:func:`update_fold_op`)."""
    cand_norm = cell.cand_norm_kind != "none"
    update_fold_op(
        y,
        gate,
        tape_h,
        tape_q,
        tape_age,
        cells,
        valid,
        cell.cand_norm.weight if cand_norm else cell.norm.weight,
        reset,
        tape_hsel,
        h_max,
        qs,
        x,
        z,
        context,
        t,
        cand_norm,
        float(cast(nn.RMSNorm, cell.cand_norm).eps or 0.0) if cand_norm else 0.0,
    )
