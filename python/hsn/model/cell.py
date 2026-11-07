"""The Predictive Delta Graph cell (P1.5): one dense round of the cell update of plan §2.3 for
every cell of the graph.

:class:`PDGCell` owns the per-cell parameters of Appendix C.1 that are not units: the message
projection ``V`` (``d × d``, no bias; row ``v_msg``), the output projection ``W_out`` (``d × d``, no
bias; row ``w_out``) and the gate ``B`` (``4d → d`` with bias; row ``gate``), plus the P1.2
:class:`~hsn.model.units.UnitBank` (``units``) and the P1.4 :class:`~hsn.model.graph.CellGraph`
(the topology and the learned edge weights ``a_ij``). For every cell ``i`` and the previous round's
states ``h[·, t, r−1]``::

    m_i = Σ_{j ∈ N(i)} a_ij · V · h[j, t, r−1]             CellGraph.messages_dense
    v_i = [u_t ; ε_t ; h[i, t, r−1] ; m_i]                 ∈ R^{4d}
    c_i = tanh( W_out · Σ_j α_ij · F_j(v_i) )              tanh on the *mixed* unit output
    b_i = σ( B · v_i )                                     ∈ (0, 1)^d
    h[i, t, r] = (1 − b_i) ⊙ h[i, t, r−1] + b_i ⊙ c_i

In dense mode every cell is updated and ``α_i`` is a distribution over all ``M`` units (a softmax
of the scorer logits, computed by :class:`~hsn.model.hsn.HSN`). The sparse round (P2.5,
:meth:`PDGCell.step_sparse`) runs the same equations for the ``K`` cells of a P2.1
:class:`~hsn.model.router.Selection` only, one unit each, composed of the four kernel ops of
:mod:`hsn.kernels` at the boundaries P2.9–P2.11 replace::

    m   = csr_messages(h, graph, V, cell_idx)            [B, K, d]   (reads the pre-round h)
    v   = gather_concat(u, ε, h, m, cell_idx)            [B, K, 4d]
    F   = grouped_unit_mlp(v, unit_idx, units)           [B·K, d]    (UnitBank.forward_selected)
    c   = tanh( W_out · α ⊙ F ),  b = σ( B · v )         α = Selection.alpha (1.0 on hard slots)
    state' = gated_delta_update(state, cell_idx, c, b, valid)   h, q (incremental), age; no tick

``Selection.alpha`` multiplies the selected unit's output before ``W_out`` (§2.3's ``Σ_j α_ij
F_j(v_i)`` with one term) and is the only place a selection gradient enters the round: P2.2's
straight-through estimators (the cell mask and the unit weights) must be folded into ``alpha``.
Padded slots (``valid=False``) are ignored end to end: their cell and unit indices are replaced by
0 before the gathers (so nothing depends on what the router pads with, in range or not), they are
not written, add nothing to ``q`` and get no gradient. When every cell is selected with one-hot
``α`` the sparse round equals the dense one to fp32 rounding
(``test_p2_5_sparse_equals_dense_when_all_selected``). With ``h = 0`` initially every coordinate of
``h`` stays in ``[−1, 1]``: each update is a coordinatewise convex combination of ``h`` and a
``tanh`` output.

``model.disabled_cells`` is the P1.8 test hook: those cells are never updated (their row of ``h``
is carried through unchanged, with no gradient into the discarded candidate). In sparse mode a
selected disabled cell is dropped from the valid slots: it is not written and keeps ageing, while
``Selection.cost`` still charges it (the router chose it).
``model.ablate_shared_u`` (remove ``u`` from ``v``) is not implemented here: the P1.1 closed forms
fix ``v`` at width ``4d`` and
refuse to count the ablation, so this module refuses to build it rather than guess its shapes.

The dense round reads only the ``state`` it is given and returns the new ``h``; writing the state
(the summary ``q``, ages, position) is :class:`~hsn.model.hsn.HSN`'s job through the P1.3
functional updates. The sparse round returns the per-slot new values and, in ``info["state"]``, the
functionally updated state (``h``, incremental ``q``, ages of the written cells at 0) because the
scatter and the ``q`` update are one kernel boundary; the tick (``age += 1`` elsewhere,
``pos += 1``) is still :class:`~hsn.model.hsn.HSN`'s.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextlib import AbstractContextManager

import torch
from torch import Tensor, nn

from hsn import kernels
from hsn.config import ModelConfig
from hsn.kernels.native import selection_mask
from hsn.model.embeddings import init_weight_
from hsn.model.graph import CellGraph
from hsn.model.router import Selection
from hsn.model.state import CellGraphState
from hsn.model.units import UnitBank
from hsn.profiler import Profiler

__all__ = ["PDGCell", "profiler_segment"]


def _mix(alpha: Tensor, unit_out: Tensor) -> Tensor:
    """``Σ_j α_ij F_j(v_i)``: ``alpha [B, N, M]`` and ``unit_out [B, N, M, d]`` to ``[B, N, d]``.
    One batched contraction (P1.10c) when both share a dtype; under autocast the unit outputs are
    bf16 and the mixing weights keep the state dtype, so the promoting product and sum stay (a
    contraction would round the weights to bf16)."""
    if alpha.dtype != unit_out.dtype:
        return (alpha.unsqueeze(-1) * unit_out).sum(dim=2)
    B, N, M, d = (int(n) for n in unit_out.shape)
    mixed = torch.matmul(alpha.reshape(B * N, 1, M), unit_out.reshape(B * N, M, d))
    return mixed.reshape(B, N, d)


def profiler_segment(
    profiler: Profiler | None, name: str, tokens: int
) -> AbstractContextManager[object]:
    """``profiler.segment(name, tokens=tokens)`` or a null context when no profiler is attached
    (no synchronisation and no bookkeeping unless a caller asks for a profile)."""
    if profiler is None:
        return contextlib.nullcontext()
    segment: AbstractContextManager[object] = profiler.segment(name, tokens=tokens)
    return segment


class PDGCell(nn.Module):
    """``V``, ``W_out``, the gate ``B`` (attribute ``gate``), the unit bank and the cell graph, and
    the dense round :meth:`step_dense`.

    ``seed`` is ``run.seed``: it fixes the graph topology (P1.4), which lives outside ``model.*``
    and is therefore a required argument.
    """

    disabled: Tensor  # bool [N]: True for model.disabled_cells (non-persistent, config-derived)

    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        super().__init__()
        if cfg.ablate_shared_u:
            raise NotImplementedError(
                "model.ablate_shared_u=true is not implemented in P1.5: the P1.1 closed forms fix "
                "the "
                "cell input v = [u; eps; h_i; m_i] at width 4d and refuse to count the ablation "
                "(hsn.train.memory_plan); the ablation's shapes are defined with the P2.13 matrix"
            )
        d = cfg.d
        self.d = d
        self.n_cells = cfg.n_cells
        self.n_units = cfg.n_units
        self.seed = seed
        self.V = nn.Linear(d, d, bias=False)
        self.W_out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(4 * d, d)
        for linear in (self.V, self.W_out, self.gate):
            init_weight_(linear.weight, cfg.state_init)
        # P1.5a (amendment A1): the gate starts near retention, sigmoid(-3) = 0.047 by default,
        # instead of 0.5; a bias of 0 reproduces the P1.5 cell.
        nn.init.constant_(self.gate.bias, cfg.gate_bias_init)
        # P1.5a: RMSNorm over the 4d cell input before the unit MLPs and the gate (a learned gain
        # of width 4d, the Appendix C.1 row ``cell_norm``); Identity when model.cell_norm = none.
        self.cell_norm = cfg.cell_norm
        self.norm: nn.Module = (
            nn.RMSNorm(4 * d, eps=1e-6) if cfg.cell_norm == "rms" else nn.Identity()
        )
        self.units = UnitBank.from_config(cfg)
        self.graph = CellGraph(
            cfg.n_cells,
            cfg.in_degree,
            seed,
            random_edges=cfg.random_edges,
            edge_norm=cfg.edge_norm,
        )
        self.disabled_cells: tuple[int, ...] = tuple(sorted(set(cfg.disabled_cells)))
        disabled = torch.zeros(cfg.n_cells, dtype=torch.bool)
        if self.disabled_cells:
            disabled[list(self.disabled_cells)] = True
        self.register_buffer("disabled", disabled, persistent=False)
        #: P1.10e: True inside :meth:`window`, whose caller (``HSN.forward``) validated the window
        #: once: the rounds then skip their per-call shape / dtype / device checks.
        self._in_window = False

    # ----------------------------------------------------------------------------------------------
    # A window of rounds (P1.10a)
    # ----------------------------------------------------------------------------------------------

    @contextlib.contextmanager
    def window(self) -> Iterator[None]:
        """A window of positions (:meth:`hsn.model.hsn.HSN.forward` holds it around its loop): the
        graph's step guard is held and the work that depends on the parameters only — the edge
        weights and the dense adjacency (:meth:`CellGraph.window`), the unit bank's ``W1`` layout
        (:meth:`UnitBank.window`, P1.10c) — is done once for the window. The rounds themselves are
        unchanged, and ``step_dense`` / ``step_sparse`` are still called once per position (tests
        and diagnostics wrap them). Nothing is cached beyond the window."""
        previous = self._in_window
        with self.graph.window(), self.units.window():
            self._in_window = True
            try:
                yield
            finally:
                self._in_window = previous

    # ----------------------------------------------------------------------------------------------
    # The dense round
    # ----------------------------------------------------------------------------------------------

    def step_dense(
        self, u: Tensor, eps: Tensor, state: CellGraphState, alpha: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """One synchronous round for every cell: ``(h_new [B, N, d], info)``.

        ``u``, ``eps`` are ``[B, d]`` and ``alpha`` is ``[B, N, M]`` (the mixing weights over units
        per cell; a softmax in dense mode), all in the dtype and on the device of ``state.h``.
        Messages read ``state.h`` only (the previous round). ``info`` carries the intermediate
        tensors of the equations above (``m``, ``v``, ``unit_out [B, N, M, d]``, ``mixed``, ``c``,
        ``b``, ``h_prev``) for tests and the P1.5 reference check. The graph's step guard is held
        for the duration (``set_topology`` raises meanwhile).
        """
        if self._in_window:  # P1.10e: validated once by the window's caller
            B, N, d = (int(n) for n in state.h.shape)
        else:
            B, N, d = self._check_inputs(u, eps, state, alpha)
        with self.graph.step():
            h_prev = state.h
            m = self.graph.messages_dense(h_prev, self.V)  # [B, N, d]
            v_raw = torch.cat(
                [u[:, None, :].expand(B, N, d), eps[:, None, :].expand(B, N, d), h_prev, m], dim=-1
            )  # [B, N, 4d]
            v = self.norm(v_raw)  # P1.5a: what the units and the gate read
            unit_out = self.units.forward_all(v)  # [B, N, M, d] (a view of [M, B·N, d])
            mixed = _mix(alpha, unit_out)  # Σ_j α_ij F_j(v_i)
            # P1.9: tanh, the gate and the convex combination run in the state's dtype. Under
            # bf16 autocast W_out and the gate emit bf16; a bf16 ``1 - b`` is rounded, the weights
            # no longer sum to one and |h| compounds past the §2.3 bound over a window (measured
            # 1.33 after 256 positions on rtx3060). Without autocast the casts are no-ops.
            c = torch.tanh(self.W_out(mixed).to(h_prev.dtype))
            b = torch.sigmoid(self.gate(v).to(h_prev.dtype))
            h_new = torch.lerp(h_prev, c, b)  # (1 − b) ⊙ h + b ⊙ c in one op (P1.10c)
            if self.disabled_cells:
                h_new = torch.where(self.disabled[None, :, None], h_prev, h_new)
        info = {
            "m": m,
            "v": v,  # the normalised cell input (P1.5a); ``v_raw`` is the concatenation
            "v_raw": v_raw,
            "unit_out": unit_out,
            "mixed": mixed,
            "c": c,
            "b": b,
            "h_prev": h_prev,
        }
        return h_new, info

    # ----------------------------------------------------------------------------------------------
    # The sparse round (P2.5)
    # ----------------------------------------------------------------------------------------------

    def step_sparse(
        self,
        u: Tensor,
        eps: Tensor,
        state: CellGraphState,
        sel: Selection,
        *,
        profiler: Profiler | None = None,
    ) -> tuple[Tensor, dict[str, Tensor | CellGraphState]]:
        """One round for the ``K`` selected cells of ``sel``: ``(h_new [B, K, d], info)``.

        ``u``, ``eps`` are ``[B, d]`` in the dtype and on the device of ``state.h``; ``sel`` is a
        P2.1 :class:`~hsn.model.router.Selection` on the same device (``cell_idx``/``unit_idx``
        ``[B, K]``, ``alpha`` and ``valid``). The four ops of :func:`hsn.kernels.ops` run in the
        order of the module docstring: messages and inputs read ``state.h`` (the previous round),
        the unit bank runs once per present unit on the ``B·K`` pairs, and
        ``gated_delta_update`` writes the valid slots functionally. ``h_new[b, k]`` is the slot's
        updated state (``h[cell]`` unchanged on padded or disabled slots). ``info`` carries ``m``,
        ``v``, ``unit_out [B, K, d]``, ``mixed`` (``α ⊙ unit_out``), ``c``, ``b``, ``h_old``,
        ``valid`` (the slots actually written: ``sel.valid`` minus disabled cells), ``selected``
        (bool ``[B, N]``, the cells written — the tick mask) and ``state`` (the updated
        :class:`~hsn.model.state.CellGraphState` before the tick). With a ``profiler`` the round is
        timed as the P0.6 segments ``messages`` (messages, inputs, gate; then mixing, ``W_out``,
        ``tanh`` and the update: two entries), and ``selected_cells`` (the unit bank), the
        partition of :func:`hsn.kernels.costs.profile_dense_costs`. The graph's step guard is held
        for the duration.
        """
        if self._in_window:  # P1.10e: validated once by the window's caller
            B, N, d = (int(n) for n in state.h.shape)
        else:
            B, N, d = self._check_vectors(u, eps, state)
            self._check_selection(sel, B, state)
        K = sel.k
        native = kernels.ops()
        # Padded (and disabled) slots gather cell 0 and unit 0, valid indices on every backend, and
        # are never written: nothing below reads what the router put in a padded slot.
        valid = sel.valid
        cells = torch.where(valid, sel.cell_idx, torch.zeros_like(sel.cell_idx))
        if self.disabled_cells:
            valid = valid & ~self.disabled[cells]
            cells = torch.where(valid, cells, torch.zeros_like(cells))
        units = torch.where(valid, sel.unit_idx, torch.zeros_like(sel.unit_idx))
        with self.graph.step():
            with profiler_segment(profiler, "messages", B):
                h_prev = state.h
                m = native.csr_messages(h_prev, self.graph, self.V, cells)  # [B, K, d]
                v_raw = native.gather_concat(u, eps, h_prev, m, cells)  # [B, K, 4d]
                v = self.norm(v_raw)  # P1.5a: the same normalisation as the dense round
                b = torch.sigmoid(self.gate(v).to(state.dtype))  # P1.9: see step_dense
            with profiler_segment(profiler, "selected_cells", B):
                unit_out = native.grouped_unit_mlp(
                    v.reshape(B * K, 4 * d), units.reshape(B * K), self.units
                ).reshape(B, K, d)
            # The second per-cell entry of a position carries no tokens (profile_dense_costs does
            # the same), so the segment's tokens stay one count per position.
            with profiler_segment(profiler, "messages", 0):
                # α (fp32, the selection's gradient entry) promotes the unit output like the dense
                # path's mixing does; W_out recasts under autocast.
                mixed = sel.alpha.unsqueeze(-1) * unit_out  # α_i F_j(v_i)
                c = torch.tanh(self.W_out(mixed).to(state.dtype))  # P1.9: see step_dense
                new_state = native.gated_delta_update(state, cells, c, b, valid)
        h_old = v_raw[..., 2 * d : 3 * d]  # the gathered h[cell] slice of the raw v
        h_new = torch.where(valid.unsqueeze(-1), (1 - b) * h_old + b * c, h_old).to(state.dtype)
        info: dict[str, Tensor | CellGraphState] = {
            "m": m,
            "v": v,
            "unit_out": unit_out,
            "mixed": mixed,
            "c": c,
            "b": b,
            "h_old": h_old,
            "valid": valid,
            "selected": selection_mask(cells, valid, N),
            "state": new_state,
        }
        return h_new, info

    # ----------------------------------------------------------------------------------------------
    # Accounting and validation
    # ----------------------------------------------------------------------------------------------

    def param_count(self) -> int:
        """Measured parameter count of ``V``, ``W_out``, the gate, the units and the edge logits."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"d={self.d}, n_cells={self.n_cells}, n_units={self.n_units}, seed={self.seed}, "
            f"disabled_cells={list(self.disabled_cells)}"
        )

    def _check_inputs(
        self, u: Tensor, eps: Tensor, state: CellGraphState, alpha: Tensor
    ) -> tuple[int, int, int]:
        B, N, d = self._check_vectors(u, eps, state)
        self._check_like_state("alpha", alpha, (B, N, self.n_units), state)
        return B, N, d

    def _check_vectors(self, u: Tensor, eps: Tensor, state: CellGraphState) -> tuple[int, int, int]:
        """The checks every round shares: the state's shape against this cell, ``u``/``eps``
        ``[B, d]`` in the state's dtype and device, the parameters on the state's device."""
        if not isinstance(state, CellGraphState):
            raise TypeError(f"state must be a CellGraphState, got {type(state).__name__}")
        B, N, d = state.batch_size, state.n_cells, state.width
        if N != self.n_cells or d != self.d:
            raise ValueError(
                f"state has N = {N}, d = {d}; this PDGCell expects N = {self.n_cells}, d = {self.d}"
            )
        for name, x, shape in (("u", u, (B, d)), ("eps", eps, (B, d))):
            self._check_like_state(name, x, shape, state)
        if self.V.weight.device != state.device:
            raise ValueError(
                f"state lives on {state.device} but the cell's parameters on "
                f"{self.V.weight.device}; move the module with .to(device)"
            )
        if self.V.weight.dtype != state.dtype:
            raise ValueError(
                f"state has dtype {state.dtype} but the cell's parameters {self.V.weight.dtype}"
            )
        return B, N, d

    @staticmethod
    def _check_like_state(
        name: str, x: Tensor, shape: tuple[int, ...], state: CellGraphState
    ) -> None:
        if not isinstance(x, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
        if x.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(x.shape)}")
        if x.dtype != state.dtype:
            raise ValueError(f"{name} must have dtype {state.dtype} like the state, got {x.dtype}")
        if x.device != state.device:
            raise ValueError(f"{name} must live on {state.device}, got {x.device}")

    def _check_selection(self, sel: Selection, B: int, state: CellGraphState) -> None:
        """A P2.1 ``Selection`` for ``B`` rows on the state's device with ``1 <= K <= N`` (its own
        ``__post_init__`` validated the fields against each other)."""
        if not isinstance(sel, Selection):
            raise TypeError(f"sel must be a hsn.model.router.Selection, got {type(sel).__name__}")
        if sel.cell_idx.shape[0] != B:
            raise ValueError(f"sel has {sel.cell_idx.shape[0]} rows but the state has B = {B}")
        if not 1 <= sel.k <= self.n_cells:
            raise ValueError(
                f"sel has K = {sel.k} slots; the sparse round needs 1 <= K <= N = {self.n_cells}"
            )
        if sel.logits.shape[1:] != (self.n_cells, self.n_units):
            raise ValueError(
                f"sel.logits must be [B, N, M] = [{B}, {self.n_cells}, {self.n_units}], got "
                f"{tuple(sel.logits.shape)}"
            )
        if sel.cell_idx.device != state.device:
            raise ValueError(f"sel must live on {state.device}, got {sel.cell_idx.device}")
        # P2.2 adds dense-fallback rows (charged the dense cost, every cell written) and soft gates
        # to Selection; the sparse round executes K slots only, so it refuses such a selection
        # rather than run it as if it were hard (errors over silence; the merge branches instead).
        fallback = getattr(sel, "dense_fallback", None)
        if isinstance(fallback, Tensor) and bool(fallback.any()):
            raise NotImplementedError(
                "step_sparse cannot execute dense-fallback rows (P2.2); run them through step_dense"
            )
