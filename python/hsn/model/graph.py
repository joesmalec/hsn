"""Fixed sparse cell graph and message passing (P1.4): a CSR adjacency over the ``N`` cells with
in-degree ``D_g`` (plus optional random candidate edges), one learnable weight per edge, and the
neighbour messages ``m_i = Σ_{j ∈ N(i)} a_ij · V · h[j, r−1]`` (plan §2.3) for every cell or for a
selected subset.

Topology (plan §2.1, paper §6). Edges are directed ``j → i``: cell ``i`` *reads* its sources ``j``.
The adjacency is stored as CSR by target row: ``indptr [N+1]``, ``indices [E]`` (the sources of row
``i`` are ``indices[indptr[i]:indptr[i+1]]``, strictly increasing within the row, never ``i``
itself) and ``is_random [E]`` marking the random candidate edges. The constructor draws, for every
cell and from a ``torch.Generator`` seeded with ``seed`` alone, ``in_degree`` distinct fixed sources
and ``random_edges`` further distinct candidate sources, so ``E = N · (in_degree + random_edges)``
and the topology is a function of the seed only (the global RNG is neither read nor advanced).
``random_edges`` counts per cell, which keeps every row the same length (static shapes, P2.8) and
makes "exactly ``in_degree`` fixed sources per cell" well defined. ``in_degree = 0`` (the
``model.in_degree=0`` ablation) is allowed and gives an edgeless graph whose messages are zero.
:meth:`CellGraph.set_topology` replaces the CSR between windows (SAN, P4.6) and is refused while
:attr:`CellGraph.in_step` is set; it keeps the logits of edges that persist and starts new ones at
0. Candidate edges are *present*, never scored here (P4.6 owns scoring).

Edge weights. ``edge_logit [E]`` is the one parameter. :meth:`CellGraph.edge_weights` normalises it
over each cell's incoming edges: ``softmax`` per row (weights of a row sum to one) or ``sigmoid``
per edge (each weight in ``(0, 1)``, independent). Logits start at zero, so softmax rows start
uniform. :meth:`CellGraph.l1` is ``Σ |a_ij|`` for ``L_graph`` (plan §2.6); under ``softmax`` it is
the constant number of non-empty rows and carries no gradient, so the sparsity penalty acts with
``sigmoid``.

Messages. ``Vh = V(h_prev)`` is computed once for all ``N`` cells (``V`` before the weighting, as
the equation reads). The CSR is mirrored into a padded view ``[N, D_max]`` (source per slot, edge
id per slot, slot mask; non-persistent buffers rebuilt whenever the topology changes or a state
dict is loaded). The selected-cell path gathers the sources from ``Vh`` and sums them (gather →
multiply → sum): deterministic, no atomics, the same ops on CPU, CUDA and MPS. The all-cells path
sums the same way, each row's sources in slot order, which keeps identical cells bitwise identical
(a dense ``A · Vh`` contraction would not; :meth:`messages_dense`). :meth:`dense_adjacency` builds
``A [N, N]`` for tests and analysis, once per window. The int64 CSR buffers are what the later
``csr_messages`` kernel reads (P2.9 narrows them to the C ABI's int32). Both message functions read
only the ``h_prev`` they are given — the previous round's states — and never touch a
:class:`~hsn.model.state.CellGraphState`.

State dict. ``indptr``, ``indices``, ``is_random`` and ``edge_logit`` are saved; a checkpoint with
another edge count is refused with a message that says to call :meth:`CellGraph.set_topology` with
the checkpoint's topology first (never reinterpreted by shape). Replacing the topology creates a new
``edge_logit`` parameter: an optimiser built before must have its parameter group refreshed (P4.6).

Windows (P1.10a). The edge weights depend on the parameters only, so a window of positions
(:meth:`hsn.model.hsn.HSN.forward`) computes them once: inside :meth:`CellGraph.window` the padded
weights are built on first use and reused by every message call of the window, and dropped when the
window ends (a cast to the message dtype under autocast still runs per call, so that the window's
gradient contributions are summed in the weights' dtype). A cached value is bound to the
``edge_logit`` object, its in-place version counter and the grad mode it was built under; a use
that sees any of them changed raises ``RuntimeError`` rather than serving a stale weight. Outside a
window every call recomputes, as before. The cache never outlives a window, so
``torch.func.functional_call`` and replaced parameters are always read fresh.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn

from hsn.config import EdgeNorm

__all__ = ["CellGraph"]

_INDEX_DTYPE = torch.int64
_SEED_LIMIT = 2**32  # torch.Generator.manual_seed accepts more, but one convention (hsn.device)
_EDGE_NORMS: tuple[EdgeNorm, ...] = ("softmax", "sigmoid")
_CSR_FIELDS = ("indptr", "indices", "is_random")


class CellGraph(nn.Module):
    """The fixed sparse cell graph: CSR adjacency, learnable edge weights and message passing.

    See the module docstring for the layout. Buffers move with the module (``graph.to(device)``);
    ``h_prev`` and ``cell_idx`` must live on the graph's device.
    """

    indptr: Tensor  # int64 [N+1]
    indices: Tensor  # int64 [E], strictly increasing within each row, never the row itself
    is_random: Tensor  # bool [E]
    edge_logit: nn.Parameter  # [E]
    _row: Tensor  # int64 [E]: the target cell of every edge
    _edge_pos: Tensor  # int64 [E]: the edge's position in the flattened padded view
    _pad_src: Tensor  # int64 [N, D_max]: source per slot (0 on padding)
    _pad_edge: Tensor  # int64 [N, D_max]: edge id per slot (E on padding)
    _pad_mask: Tensor  # bool [N, D_max]: True where a slot holds an edge

    def __init__(
        self,
        n_cells: int,
        in_degree: int,
        seed: int,
        random_edges: int = 0,
        edge_norm: EdgeNorm = "softmax",
    ) -> None:
        super().__init__()
        for name, value in (
            ("n_cells", n_cells),
            ("in_degree", in_degree),
            ("random_edges", random_edges),
            ("seed", seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"CellGraph: {name} must be an int, got {value!r}")
        if n_cells < 1:
            raise ValueError(f"CellGraph: n_cells must be >= 1, got {n_cells}")
        if in_degree < 0 or random_edges < 0:
            raise ValueError(
                f"CellGraph: in_degree and random_edges must be >= 0, got {in_degree} and "
                f"{random_edges}"
            )
        if in_degree + random_edges > n_cells - 1:
            raise ValueError(
                f"CellGraph: in_degree + random_edges = {in_degree + random_edges} sources per "
                f"cell exceed the {n_cells - 1} other cells (model.in_degree, model.random_edges "
                f"must sum to < model.n_cells)"
            )
        if not 0 <= seed < _SEED_LIMIT:
            raise ValueError(f"CellGraph: seed must lie in [0, {_SEED_LIMIT}), got {seed}")
        if edge_norm not in _EDGE_NORMS:
            raise ValueError(
                f"CellGraph: edge_norm must be one of {list(_EDGE_NORMS)}, got {edge_norm!r}"
            )
        self.n_cells = n_cells
        self.in_degree = in_degree
        self.random_edges = random_edges
        self.seed = seed
        self.edge_norm: EdgeNorm = edge_norm
        #: True while a step (round) is running; :meth:`set_topology` is refused meanwhile.
        self.in_step = False
        #: The per-window cache of :meth:`window` (``None`` outside a window): name ->
        #: ((edge_logit id, its version, grad mode), tensor).
        self._window_cache: dict[str, tuple[tuple[int, int, bool], Tensor]] | None = None

        indptr, indices, is_random = _build_topology(n_cells, in_degree, random_edges, seed)
        _check_topology(indptr, indices, is_random, n_cells, in_degree)
        self.register_buffer("indptr", indptr)
        self.register_buffer("indices", indices)
        self.register_buffer("is_random", is_random)
        self.edge_logit = nn.Parameter(torch.zeros(indices.numel()))
        for name in ("_row", "_edge_pos", "_pad_src", "_pad_edge", "_pad_mask"):
            self.register_buffer(name, torch.empty(0, dtype=_INDEX_DTYPE), persistent=False)
        self._max_in_degree = 0
        self._rebuild_derived()
        # torch 2.15 ships this method without annotations.
        self.register_load_state_dict_post_hook(_rebuild_after_load)  # type: ignore[no-untyped-call]

    # -- shape ------------------------------------------------------------------------------------

    @property
    def n_edges(self) -> int:
        return int(self.indices.numel())

    @property
    def max_in_degree(self) -> int:
        """The longest CSR row (the padded view's width ``D_max``)."""
        return self._max_in_degree

    def extra_repr(self) -> str:
        return (
            f"n_cells={self.n_cells}, in_degree={self.in_degree}, "
            f"random_edges={self.random_edges}, edge_norm={self.edge_norm!r}, "
            f"n_edges={self.n_edges}"
        )

    # -- the step guard ---------------------------------------------------------------------------

    @contextmanager
    def step(self) -> Iterator[None]:
        """Mark a step (round) as running: :meth:`set_topology` raises inside. Re-entrant, so a
        window may wrap its steps; the previous flag value is restored on exit."""
        previous = self.in_step
        self.in_step = True
        try:
            yield
        finally:
            self.in_step = previous

    @contextmanager
    def window(self) -> Iterator[None]:
        """A window of steps (P1.10a, module docstring): the step guard is held, and the edge
        weights are computed once and reused until the window ends. Windows do not nest."""
        if self._window_cache is not None:
            raise RuntimeError("CellGraph.window is already active; windows do not nest")
        with self.step():
            self._window_cache = {}
            try:
                yield
            finally:
                self._window_cache = None

    def _cached(self, name: str, build: Callable[[], Tensor]) -> Tensor:
        """``build()``, once per window under ``name`` (every call outside a window)."""
        cache = self._window_cache
        if cache is None:
            return build()
        logit = self.edge_logit
        stamp = (id(logit), logit._version, torch.is_grad_enabled())
        hit = cache.get(name)
        if hit is None:
            value = build()
            cache[name] = (stamp, value)
            return value
        if hit[0] != stamp:
            raise RuntimeError(
                f"CellGraph.window: the cached {name!r} edge weights were built for another "
                "edge_logit, version or grad mode; the edge logits must not change inside a window "
                "(an in-place update, a new parameter, or a grad-mode switch between positions)"
            )
        return hit[1]

    # -- edge weights -----------------------------------------------------------------------------

    def _padded_weights(self) -> Tensor:
        """``a_ij`` in the padded view ``[N, D_max]``; exactly zero on padding slots. Computed once
        per :meth:`window`."""
        return self._cached("padded", self._build_padded_weights)

    def _padded_weights_as(self, dtype: torch.dtype) -> Tensor:
        """:meth:`_padded_weights` in ``dtype`` (the message dtype under autocast). The cast runs
        per call, never cached: a cached low-precision copy used at every position would sum the
        window's gradient contributions in that precision (bf16), where the per-call cast sums
        them in the weights' own dtype, as before P1.10a. A no-op when the dtypes agree."""
        return self._padded_weights().to(dtype)

    def _build_padded_weights(self) -> Tensor:
        logit = self.edge_logit
        extended = torch.cat([logit, logit.new_zeros(1)])  # slot E: the padding sentinel
        pad = extended[self._pad_edge]
        mask = self._pad_mask
        if self.edge_norm == "sigmoid":
            return torch.where(mask, torch.sigmoid(pad), torch.zeros_like(pad))
        pad = pad.masked_fill(~mask, float("-inf"))
        # A row without edges would be all -inf and softmax to NaN; its slots are masked anyway.
        empty = ~mask.any(dim=1, keepdim=True)
        pad = torch.where(empty, torch.zeros_like(pad), pad)
        return torch.where(mask, torch.softmax(pad, dim=1), torch.zeros_like(pad))

    def edge_weights(self) -> Tensor:
        """``a_ij [E]`` in CSR order: ``edge_logit`` normalised over each cell's incoming edges
        (``softmax`` per row, or ``sigmoid`` per edge)."""
        return self._padded_weights().reshape(-1)[self._edge_pos]

    def dense_adjacency(self) -> Tensor:
        """``A [N, N]`` with ``A[i, j] = a_ij`` for every edge ``j → i`` and 0 elsewhere (rows are
        targets, columns sources): ``messages_dense`` is ``einsum("ij,bjd->bid", A, V(h))``.
        Built once per :meth:`window` (P1.10c), per call outside one."""
        return self._cached("adjacency", self._build_dense_adjacency)

    def _build_dense_adjacency(self) -> Tensor:
        weights = self.edge_weights()
        zeros = weights.new_zeros(self.n_cells, self.n_cells)
        return zeros.index_put((self._row, self.indices), weights)

    def l1(self) -> Tensor:
        """``Σ |a_ij|`` over the edge weights, the ``L_graph`` term (constant under ``softmax``)."""
        return self.edge_weights().abs().sum()

    # -- messages ---------------------------------------------------------------------------------

    def messages_dense(self, h_prev: Tensor, V: nn.Linear) -> Tensor:
        """``m [B, N, d]`` for every cell from the previous round's states ``h_prev [B, N, d]``:
        ``V`` is applied once to all cells, then each cell sums its weighted sources in slot order
        over the padded view. (P1.10c measured a dense-adjacency contraction ``A · V(h)``
        instead and kept this one: the contraction sums each row over all ``N`` columns, so rows
        with their edges at different columns round differently. From the zero state that gives
        the edge logits rounding-noise gradients, 1.6e-22 in fp64, where the slot-order sum keeps
        them exactly 0 and the cells exactly symmetric.)"""
        if self._window_cache is None:  # P1.10e: inside a window the caller validated h_prev
            self._check_h(h_prev)
        Vh: Tensor = V(h_prev)  # [B, N, d]
        weights = self._padded_weights_as(Vh.dtype)  # [N, D_max]
        sources = Vh[:, self._pad_src]  # [B, N, D_max, d]
        return (sources * weights[None, :, :, None]).sum(dim=2)

    def messages_selected(self, h_prev: Tensor, V: nn.Linear, cell_idx: Tensor) -> Tensor:
        """``m [B, K, d]`` for the cells ``cell_idx [B, K]`` only (int64, in ``[0, N)``; repeats
        are allowed). ``V`` is still applied once to all ``N`` cells — any cell may be a source —
        and only the selected rows gather. The range check synchronises with the device once per
        call (P2.8 may gate it, as for :meth:`CellGraphState.with_delta`)."""
        B, _, d = self._check_h(h_prev)
        K = self._check_cell_idx(cell_idx, B)
        D = self._max_in_degree
        Vh: Tensor = V(h_prev)  # [B, N, d]
        weights = self._padded_weights_as(Vh.dtype)[cell_idx]  # [B, K, D_max]
        source_idx = self._pad_src[cell_idx].reshape(B, K * D, 1).expand(B, K * D, d)
        sources = Vh.gather(1, source_idx).reshape(B, K, D, d)
        return (sources * weights[..., None]).sum(dim=2)

    # -- topology ---------------------------------------------------------------------------------

    def set_topology(self, indptr: Tensor, indices: Tensor, is_random: Tensor) -> None:
        """Replace the CSR adjacency. Allowed only between windows: raises ``RuntimeError`` while
        :attr:`in_step` is set. The new topology is validated like the constructor's (int64
        ``indptr [N+1]`` and ``indices [E]``, bool ``is_random [E]``, rows strictly increasing, no
        self loops, at most ``in_degree`` non-random sources per cell; ``ValueError`` otherwise).
        Edges present in both topologies keep their logits, new edges start at 0, and
        ``edge_logit`` becomes a new parameter (refresh the optimiser's parameter group)."""
        if self.in_step:
            raise RuntimeError(
                "CellGraph.set_topology is allowed only between windows: a step is running "
                "(in_step=True). Topology changes mid-step are a P4.6 'do not'."
            )
        _check_topology(indptr, indices, is_random, self.n_cells, self.in_degree)
        device = self.indices.device
        new_indptr = indptr.to(device=device, copy=True)
        new_indices = indices.to(device=device, copy=True)
        new_is_random = is_random.to(device=device, copy=True)
        new_row = _rows_of(new_indptr)
        # Keys row·N + source are strictly increasing in a valid CSR, so a merge is a searchsorted.
        old_key = self._row * self.n_cells + self.indices
        new_key = new_row * self.n_cells + new_indices
        old_logit = self.edge_logit.detach()
        n_old = old_key.numel()
        if n_old > 0:
            position = torch.searchsorted(old_key, new_key)
            clamped = position.clamp(max=n_old - 1)
            found = (position < n_old) & (old_key[clamped] == new_key)
            logit = torch.where(found, old_logit[clamped], torch.zeros_like(old_logit[clamped]))
        else:
            logit = old_logit.new_zeros(new_key.numel())
        self.indptr = new_indptr
        self.indices = new_indices
        self.is_random = new_is_random
        self.edge_logit = nn.Parameter(logit.contiguous())
        self._rebuild_derived()

    def _rebuild_derived(self) -> None:
        """Recompute the padded view and the per-edge helpers from the CSR buffers."""
        N, E = self.n_cells, self.n_edges
        device = self.indices.device
        counts = self.indptr[1:] - self.indptr[:-1]
        D = int(counts.max().item())
        row = _rows_of(self.indptr)
        slot = torch.arange(E, device=device) - self.indptr[:-1].repeat_interleave(counts)
        edge_ids = torch.arange(E, device=device)
        pad_edge = torch.full((N, D), E, dtype=_INDEX_DTYPE, device=device)
        pad_edge[row, slot] = edge_ids
        pad_src = torch.zeros((N, D), dtype=_INDEX_DTYPE, device=device)
        pad_src[row, slot] = self.indices
        self._max_in_degree = D
        self._row = row
        self._edge_pos = row * D + slot
        self._pad_src = pad_src
        self._pad_edge = pad_edge
        self._pad_mask = pad_edge < E

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Refuse a checkpoint with another edge count before anything is copied, so the module
        never ends up with a loaded ``indptr`` over the old ``indices``."""
        key = prefix + "indices"
        loaded = state_dict.get(key)
        if isinstance(loaded, Tensor) and loaded.numel() != self.n_edges:
            error_msgs.append(
                f"size mismatch for {key}: the checkpoint has {loaded.numel()} edges, this "
                f"CellGraph has {self.n_edges}; call set_topology(indptr, indices, is_random) "
                "with the checkpoint's topology before load_state_dict"
            )
            return
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    # -- validation -------------------------------------------------------------------------------

    def _check_h(self, h_prev: Tensor) -> tuple[int, int, int]:
        if not isinstance(h_prev, Tensor):
            raise TypeError(f"h_prev must be a torch.Tensor, got {type(h_prev).__name__}")
        if h_prev.dim() != 3 or h_prev.shape[1] != self.n_cells or min(h_prev.shape) < 1:
            raise ValueError(
                f"h_prev must have shape [B, N, d] with N = {self.n_cells}, got "
                f"{tuple(h_prev.shape)}"
            )
        if not h_prev.is_floating_point():
            raise ValueError(f"h_prev must be floating point, got {h_prev.dtype}")
        if h_prev.device != self.indices.device:
            raise ValueError(
                f"h_prev must live on the graph's device {self.indices.device}, got "
                f"{h_prev.device}; move the graph with .to(device)"
            )
        B, _, d = (int(n) for n in h_prev.shape)
        return B, self.n_cells, d

    def _check_cell_idx(self, cell_idx: Tensor, B: int) -> int:
        if not isinstance(cell_idx, Tensor):
            raise TypeError(f"cell_idx must be a torch.Tensor, got {type(cell_idx).__name__}")
        if cell_idx.dtype != _INDEX_DTYPE:
            raise ValueError(f"cell_idx must have dtype torch.int64, got {cell_idx.dtype}")
        if cell_idx.dim() != 2 or cell_idx.shape[0] != B or cell_idx.shape[1] < 1:
            raise ValueError(
                f"cell_idx must have shape [B, K] = [{B}, >= 1], got {tuple(cell_idx.shape)}"
            )
        if cell_idx.device != self.indices.device:
            raise ValueError(
                f"cell_idx must live on the graph's device {self.indices.device}, got "
                f"{cell_idx.device}"
            )
        lowest, highest = (int(v) for v in torch.stack([cell_idx.min(), cell_idx.max()]).tolist())
        if lowest < 0 or highest >= self.n_cells:
            raise IndexError(
                f"cell_idx must lie in [0, {self.n_cells}), got values in [{lowest}, {highest}]"
            )
        return int(cell_idx.shape[1])


# --------------------------------------------------------------------------------------------------
# Topology construction and validation
# --------------------------------------------------------------------------------------------------


def _build_topology(
    n_cells: int, in_degree: int, random_edges: int, seed: int
) -> tuple[Tensor, Tensor, Tensor]:
    """CSR ``(indptr, indices, is_random)`` on the CPU from the seed alone: every cell draws
    ``in_degree + random_edges`` distinct sources among the other cells (one random permutation
    per row); the first ``in_degree`` are fixed, the rest random candidates; rows are sorted."""
    N, per_row = n_cells, in_degree + random_edges
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # A random permutation of the N−1 other cells per row, as the stable argsort of random keys.
    keys = torch.rand(N, max(N - 1, 0), generator=generator)
    perm = torch.argsort(keys, dim=1, stable=True)[:, :per_row]
    sources = perm + (perm >= torch.arange(N)[:, None]).to(_INDEX_DTYPE)  # skip self
    random_slot = torch.arange(per_row)[None, :].expand(N, per_row) >= in_degree
    order = torch.argsort(sources, dim=1, stable=True)
    indices = sources.gather(1, order).reshape(-1).contiguous()
    is_random = random_slot.gather(1, order).reshape(-1).contiguous()
    indptr = torch.arange(N + 1, dtype=_INDEX_DTYPE) * per_row
    return indptr, indices, is_random


def _rows_of(indptr: Tensor) -> Tensor:
    """The target row of every edge, ``[E]``, from ``indptr [N+1]``."""
    counts = indptr[1:] - indptr[:-1]
    return torch.arange(counts.numel(), device=indptr.device).repeat_interleave(counts)


def _check_topology(
    indptr: Tensor, indices: Tensor, is_random: Tensor, n_cells: int, in_degree: int
) -> None:
    """Raise ``TypeError``/``ValueError`` unless the triple is a valid CSR for this graph."""
    for name, tensor in zip(_CSR_FIELDS, (indptr, indices, is_random), strict=True):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.dim() != 1:
            raise ValueError(f"{name} must be one-dimensional, got shape {tuple(tensor.shape)}")
    if indptr.dtype != _INDEX_DTYPE or indices.dtype != _INDEX_DTYPE:
        raise ValueError(
            f"indptr and indices must have dtype torch.int64, got {indptr.dtype} and "
            f"{indices.dtype}"
        )
    if is_random.dtype != torch.bool:
        raise ValueError(f"is_random must have dtype torch.bool, got {is_random.dtype}")
    if indptr.numel() != n_cells + 1:
        raise ValueError(f"indptr must have shape [N+1] = [{n_cells + 1}], got [{indptr.numel()}]")
    E = indices.numel()
    if is_random.numel() != E:
        raise ValueError(f"is_random must have shape [E] = [{E}], got [{is_random.numel()}]")
    devices = {str(t.device) for t in (indptr, indices, is_random)}
    if len(devices) != 1:
        raise ValueError(f"indptr, indices and is_random must share one device, got {devices}")
    counts = indptr[1:] - indptr[:-1]
    first, last, negative = (
        int(v) for v in torch.stack([indptr[0], indptr[-1], (counts < 0).sum()]).tolist()
    )
    if first != 0 or negative or last != E:
        raise ValueError(
            f"indptr must start at 0, be non-decreasing and end at E = {E}; got indptr[0] = "
            f"{first}, {negative} decreasing step(s), indptr[-1] = {last}"
        )
    if E == 0:
        return
    row = _rows_of(indptr)
    same_row = row[1:] == row[:-1]
    lowest, highest, self_loops, unsorted, over_cap = (
        int(v)
        for v in torch.stack(
            [
                indices.min(),
                indices.max(),
                (indices == row).sum(),
                (same_row & (indices[1:] <= indices[:-1])).sum(),
                (
                    torch.zeros(n_cells, dtype=_INDEX_DTYPE, device=indptr.device).index_add(
                        0, row, (~is_random).to(_INDEX_DTYPE)
                    )
                    > in_degree
                ).sum(),
            ]
        ).tolist()
    )
    if lowest < 0 or highest >= n_cells:
        raise ValueError(f"indices must lie in [0, {n_cells}), got values in [{lowest}, {highest}]")
    if self_loops:
        raise ValueError(f"indices has {self_loops} self loop(s); a cell never reads itself")
    if unsorted:
        raise ValueError(
            f"indices must be strictly increasing within each CSR row ({unsorted} violation(s): "
            "unsorted or duplicate sources)"
        )
    if over_cap:
        raise ValueError(
            f"{over_cap} cell(s) have more than in_degree = {in_degree} non-random sources"
        )


def _rebuild_after_load(module: nn.Module, incompatible_keys: Any) -> None:
    """``load_state_dict`` post-hook: validate the loaded CSR and rebuild the padded view."""
    assert isinstance(module, CellGraph)
    _check_topology(
        module.indptr, module.indices, module.is_random, module.n_cells, module.in_degree
    )
    module._rebuild_derived()
