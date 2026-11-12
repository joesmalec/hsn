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

P1.13 (amendment A4, ``docs/amendments/A4_cand_norm.md``; opt-in): with ``model.cand_norm = rms``
the candidate's pre-activation is RMS-normalised over its width ``d`` with a learned gain ``g``
(:attr:`PDGCell.cand_norm`, the Appendix C.1 row ``cand_norm``), ``c_i = tanh( RMSNorm_g( W_out ·
Σ_j α_ij F_j(v_i) ) )``: the tanh, the convex gate and with them ``h ∈ [−1, 1]`` are unchanged, and
the pre-activation's RMS is ``|g|`` instead of whatever the weights grew to. The sparse round
applies its per-slot ``α`` after the norm, ``c = tanh( α ⊙ RMSNorm_g( W_out · F ) )``: ``α`` is
exactly 1.0 forward on the valid slots, so the values equal the normalised ``W_out(α ⊙ F)``, but a
scale-invariant norm after ``α`` would zero P2.2's straight-through gradient, which enters the round
only through ``α``. ``none`` (default) is the P1.5 candidate, bit for bit.

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

``model.ablate_shared_u`` (opt-in; the "remove shared path" ablation of paper §12 and of the P2.13
matrix, Appendix A.1: "remove u from v"): the cell input is ``v_i = [ε_t ; h[i, t, r−1] ; m_i]`` in
``R^{3d}`` (:attr:`PDGCell.in_dim`), so the cell-input norm (gain of width ``3d``), every unit MLP
(``3d → H → d``) and the gate ``B`` (``3d → d``) are built at that width and nothing a cell computes
depends on ``u_t``. The rounds keep their signatures (``u`` is still passed and checked, and not
read); ``m``, ``W_out``, the candidate norm, the gated update and the state are unchanged, and so is
everything outside the cell that reads ``u`` (the scorer's features, the decoder ``D([q; u; r])``,
the halting head): the ablation removes the shared path from the *cells*, one change, not the
shared transform ``S`` from the model. It holds on every path: the dense and the masked round
(:meth:`PDGCell.step_dense`), the sparse round eager and static (:meth:`PDGCell.step_sparse`,
``gather_concat`` with ``u=None``), and the split and fused windows (:meth:`PDGCell.replay_sparse`,
:meth:`PDGCell.sparse_products`). With the key off (default) every shape and every operation is the
4d cell's, bit for bit.

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
from typing import NamedTuple

import torch
from torch import Tensor, nn

from hsn import kernels
from hsn.config import ModelConfig
from hsn.kernels import native as native_ops
from hsn.kernels.native import selection_mask
from hsn.model.embeddings import init_weight_
from hsn.model.graph import CellGraph
from hsn.model.router import Selection
from hsn.model.state import CellGraphState
from hsn.model.units import UnitBank
from hsn.profiler import Profiler

__all__ = ["PDGCell", "WindowConsts", "profiler_segment"]


class WindowConsts(NamedTuple):
    """The per-window constants of the cell (P2.8): the padded edge weights ``[N, D_max]``
    (:meth:`CellGraph._padded_weights`) and ``W1`` laid out ``[4d, M·H]``
    (:meth:`UnitBank._w1_flat`), built eagerly once per window by :meth:`PDGCell.window_consts`
    and handed to a compiled window as tensors. The rounds read them instead of the window caches,
    whose stamps (object id, version counter) a compiler would specialise on, and the softmax of
    the edge weights keeps its eager backward (the exact zero edge gradient of identical cells)."""

    edge_weights: Tensor
    w1_flat: Tensor
    #: ``W2`` laid out ``[H, M·d]`` (:meth:`UnitBank._w2_flat`), for the forward sweep's
    #: :meth:`PDGCell.sparse_products` (P2.10); ``None``: built where it is read.
    w2_flat: Tensor | None = None
    #: The forward sweep's merged first layer (P2.10): ``w1_flat`` followed by the gate's weight
    #: as ``d`` more columns, ``[4d, M·H + d]``, and the matching bias ``[M·H + d]``. The gate's
    #: linear and the unit bank's first layer read the same normalised input, so one matmul
    #: serves both (:meth:`UnitBank.forward_static_chosen`'s rider). ``None``: two matmuls.
    w1_gate: Tensor | None = None
    b1_gate: Tensor | None = None


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
        d = cfg.d
        self.d = d
        self.n_cells = cfg.n_cells
        self.n_units = cfg.n_units
        self.seed = seed
        #: ``model.ablate_shared_u`` (module docstring): the cell input is ``[eps; h_i; m_i]``.
        self.ablate_shared_u = cfg.ablate_shared_u
        #: Width of the cell input ``v`` (the norm, the units and the gate): ``4d``, or ``3d``
        #: without ``u`` (``model.ablate_shared_u``).
        self.in_dim = cfg.cell_input_width
        self.V = nn.Linear(d, d, bias=False)
        self.W_out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(self.in_dim, d)
        for linear in (self.V, self.W_out, self.gate):
            init_weight_(linear.weight, cfg.state_init)
        # P1.5a (amendment A1): the gate starts near retention, sigmoid(-3) = 0.047 by default,
        # instead of 0.5; a bias of 0 reproduces the P1.5 cell.
        nn.init.constant_(self.gate.bias, cfg.gate_bias_init)
        # P1.5a: RMSNorm over the 4d cell input before the unit MLPs and the gate (a learned gain
        # of width 4d, the Appendix C.1 row ``cell_norm``); Identity when model.cell_norm = none.
        self.cell_norm = cfg.cell_norm
        self.norm: nn.Module = (
            nn.RMSNorm(self.in_dim, eps=1e-6) if cfg.cell_norm == "rms" else nn.Identity()
        )
        # P1.13 (amendment A4): RMSNorm over W_out(.) before the candidate's tanh (a learned gain of
        # width d, the Appendix C.1 row ``cand_norm``; its ones init draws no random numbers);
        # Identity when model.cand_norm = none.
        self.cand_norm_kind = cfg.cand_norm
        self.cand_norm: nn.Module = (
            nn.RMSNorm(d, eps=1e-6) if cfg.cand_norm == "rms" else nn.Identity()
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
        #: A6 (``train.feedback_detach`` holds ``state``; set by :class:`~hsn.model.hsn.HSN`): the
        #: units, the gate and the messages read a detached copy of the pre-round ``h``, so the
        #: only gradient a round passes back into ``h`` is the carry ``(1 - b) * h``. The values
        #: are unchanged.
        self.detach_state_read = False
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

    def window_consts(self) -> WindowConsts:
        """The :class:`WindowConsts` of this window (P2.8): the same tensors the rounds of the
        window read from the caches. Only inside :meth:`window`."""
        if not self._in_window:
            raise RuntimeError("PDGCell.window_consts is only available inside PDGCell.window()")
        return WindowConsts(
            edge_weights=self.graph._padded_weights(), w1_flat=self.units._w1_flat()
        )

    # ----------------------------------------------------------------------------------------------
    # The dense round
    # ----------------------------------------------------------------------------------------------

    def step_dense(
        self,
        u: Tensor,
        eps: Tensor,
        state: CellGraphState,
        alpha: Tensor,
        *,
        consts: WindowConsts | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """One synchronous round for every cell: ``(h_new [B, N, d], info)``.

        ``u``, ``eps`` are ``[B, d]`` and ``alpha`` is ``[B, N, M]`` (the mixing weights over units
        per cell; a softmax in dense mode), all in the dtype and on the device of ``state.h``.
        Messages read ``state.h`` only (the previous round). ``info`` carries the intermediate
        tensors of the equations above (``m``, ``v``, ``unit_out [B, N, M, d]``, ``mixed``, ``c``,
        ``b``, ``h_prev``) for tests and the P1.5 reference check. The graph's step guard is held
        for the duration (``set_topology`` raises meanwhile).

        ``consts`` (P2.8, a compiled window only) are the window's edge weights and ``W1`` layout,
        read instead of the window caches (the same values); the inputs are not checked then (the
        window validated them).
        """
        if self._in_window or consts is not None:  # P1.10e: validated once by the window
            B, N, d = (int(n) for n in state.h.shape)
        else:
            B, N, d = self._check_inputs(u, eps, state, alpha)
        weights = None if consts is None else consts.edge_weights
        w1_flat = None if consts is None else consts.w1_flat
        with self.graph.step():
            h_prev = state.h
            # A6: what the round reads of the state (the carry below keeps the state itself)
            h_read = h_prev.detach() if self.detach_state_read else h_prev
            m = self.graph.messages_dense(h_read, self.V, weights=weights)  # [B, N, d]
            if self.ablate_shared_u:  # the cell input without the shared path's u: [B, N, 3d]
                v_raw = torch.cat([eps[:, None, :].expand(B, N, d), h_read, m], dim=-1)
            else:
                v_raw = torch.cat(
                    [u[:, None, :].expand(B, N, d), eps[:, None, :].expand(B, N, d), h_read, m],
                    dim=-1,
                )  # [B, N, 4d]
            v = self.norm(v_raw)  # P1.5a: what the units and the gate read
            unit_out = self.units.forward_all(v, w1_flat=w1_flat)  # [B, N, M, d] ([M, B·N, d])
            mixed = _mix(alpha, unit_out)  # Σ_j α_ij F_j(v_i)
            # P1.9: tanh, the gate and the convex combination run in the state's dtype. Under
            # bf16 autocast W_out and the gate emit bf16; a bf16 ``1 - b`` is rounded, the weights
            # no longer sum to one and |h| compounds past the §2.3 bound over a window (measured
            # 1.33 after 256 positions on rtx3060). Without autocast the casts are no-ops.
            # P1.13: cand_norm (Identity by default) normalises the state-dtype pre-activation.
            c = torch.tanh(self.cand_norm(self.W_out(mixed).to(h_prev.dtype)))
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
        static: bool = False,
        consts: WindowConsts | None = None,
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

        ``static=True`` (P2.8) is the capturable round of a compiled window: no host
        synchronisation (the index checks of the messages and of the update are skipped, since
        :meth:`Router.select` guarantees in-range, distinct valid slots) and the unit bank's static
        pass (:meth:`UnitBank.forward_static`: every unit on the ``B·K`` pairs, then a gather)
        instead of the grouped pass with its host-side histogram. The messages, the gate and the
        update are the same ops; the unit outputs agree to rounding. ``consts`` as in
        :meth:`step_dense`.
        """
        if self._in_window or consts is not None or static:  # validated by the caller (P1.10e)
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
        weights = None if consts is None else consts.edge_weights
        w1_flat = None if consts is None else consts.w1_flat
        in_dim = self.in_dim
        u_read = None if self.ablate_shared_u else u  # model.ablate_shared_u: v without u
        with self.graph.step():
            with profiler_segment(profiler, "messages", B):
                h_prev = state.h
                # A6: what the round reads of the state (the update below keeps the state itself)
                h_read = h_prev.detach() if self.detach_state_read else h_prev
                if static or weights is not None:  # P2.8: no range check, window weights
                    m = native.csr_messages(
                        h_read, self.graph, self.V, cells, weights=weights, validate=not static
                    )
                else:
                    m = native.csr_messages(h_read, self.graph, self.V, cells)  # [B, K, d]
                v_raw = native.gather_concat(u_read, eps, h_read, m, cells)  # [B, K, 4d]
                v = self.norm(v_raw)  # P1.5a: the same normalisation as the dense round
                b = torch.sigmoid(self.gate(v).to(state.dtype))  # P1.9: see step_dense
            with profiler_segment(profiler, "selected_cells", B):
                if static:  # P2.8: every unit on the pairs, then a gather (no histogram)
                    unit_out = native.grouped_unit_mlp(
                        v.reshape(B * K, in_dim),
                        units.reshape(B * K),
                        self.units,
                        static=True,
                        w1_flat=w1_flat,
                    ).reshape(B, K, d)
                else:
                    unit_out = native.grouped_unit_mlp(
                        v.reshape(B * K, in_dim), units.reshape(B * K), self.units
                    ).reshape(B, K, d)
            # The second per-cell entry of a position carries no tokens (profile_dense_costs does
            # the same), so the segment's tokens stay one count per position.
            with profiler_segment(profiler, "messages", 0):
                # α (fp32, the selection's gradient entry) promotes the unit output like the dense
                # path's mixing does; W_out recasts under autocast.
                mixed = sel.alpha.unsqueeze(-1) * unit_out  # α_i F_j(v_i)
                if self.cand_norm_kind == "none":
                    c = torch.tanh(self.W_out(mixed).to(state.dtype))  # P1.9: see step_dense
                else:  # P1.13: α after the norm keeps the selection gradient (module docstring)
                    pre = self.cand_norm(self.W_out(unit_out).to(state.dtype))
                    c = torch.tanh(sel.alpha.unsqueeze(-1).to(state.dtype) * pre)
                new_state = native.gated_delta_update(
                    state, cells, c, b, valid, validate=not static
                )
        # the gathered h[cell] slice of the raw v, its last block but one ([2d, 3d) of the 4d v)
        h_old = v_raw[..., in_dim - 2 * d : in_dim - d]
        if self.detach_state_read:  # A6: the returned slots keep the carry's gradient
            h_old = h_prev.gather(1, cells.unsqueeze(-1).expand(B, K, d))
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

    def sparse_products(
        self,
        u: Tensor,
        eps: Tensor,
        state: CellGraphState,
        cells: Tensor,
        units: Tensor,
        valid: Tensor,
        consts: WindowConsts,
        *,
        vh: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """The matrix products of :meth:`step_sparse`'s capturable round and nothing after them
        (P2.10, the fused forward sweep): ``(gate, y)``, each ``[B, K, d]`` in the state's dtype,
        the gate's pre-activation and ``W_out`` of the chosen unit's output. Messages, inputs,
        norm, the gate's linear and ``W_out`` are :meth:`step_sparse`'s own calls in its order
        (``static=True``); the unit bank runs :meth:`UnitBank.forward_static_chosen` (the static
        pass's first layer, then only the chosen unit's activation and second layer: the same
        values to rounding); the candidate norm, ``tanh``, the sigmoid and the state update are
        left to the caller (:func:`hsn.kernels.window.cell_update`). No disabled cells, no
        autograd. Two of the products can ride in a neighbour's matmul, with the same operands:
        ``vh`` is ``V(h)`` ``[B, N, d]`` from the caller (the sweep computes it with the scorer,
        which reads the same states), and ``consts.w1_gate`` / ``b1_gate`` put the gate's linear
        into the unit bank's first layer."""
        if self.disabled_cells:
            raise ValueError("PDGCell.sparse_products does not support model.disabled_cells")
        B, _, d = (int(n) for n in state.h.shape)
        K = int(cells.shape[1])
        native = kernels.ops()
        cells = torch.where(valid, cells, torch.zeros_like(cells))
        units = torch.where(valid, units, torch.zeros_like(units))
        with self.graph.step():
            h_read = state.h
            if vh is None:
                m = native.csr_messages(
                    h_read, self.graph, self.V, cells, weights=consts.edge_weights, validate=False
                )
            else:  # V(h) came with the scorer's matmul: the native path's gather of the sources
                m = self.graph.messages_selected(
                    h_read, self.V, cells, weights=consts.edge_weights, validate=False, vh=vh
                )
            u_read = None if self.ablate_shared_u else u  # model.ablate_shared_u: v without u
            v = self.norm(native.gather_concat(u_read, eps, h_read, m, cells))
            flat = v.reshape(B * K, self.in_dim)
            w2_flat = consts.w2_flat if consts.w2_flat is not None else self.units._w2_flat()
            if consts.w1_gate is not None and consts.b1_gate is not None:
                unit_out, rider = self.units.forward_static_chosen(
                    flat,
                    units.reshape(B * K),
                    w1_flat=consts.w1_gate,
                    w2_flat=w2_flat,
                    b1_flat=consts.b1_gate,
                )
                assert rider is not None
                gate = rider.reshape(B, K, d).to(state.dtype)
            else:
                gate = self.gate(v).to(state.dtype)
                unit_out, _ = self.units.forward_static_chosen(
                    flat, units.reshape(B * K), w1_flat=consts.w1_flat, w2_flat=w2_flat
                )
            y = self.W_out(unit_out.reshape(B, K, d)).to(state.dtype)
        return gate, y

    def replay_sparse(
        self, u: Tensor, eps: Tensor, h_read: Tensor, sel: Selection
    ) -> tuple[Tensor, Tensor]:
        """The candidate and the gate of :meth:`step_sparse` without the state update (P2.10, the
        split window's replay): ``(c, b)``, each ``[R, K, d]`` in ``h_read``'s dtype, for ``R``
        independent rows (every position of a window at once). ``h_read [R, N, d]`` is the recorded
        pre-round state, read as a constant (A6's ``state`` cut: the replay is only defined with
        it); ``sel`` is the recorded decision with its gradient path
        (:meth:`hsn.model.router.Router.finish_recorded`). The same operations as
        :meth:`step_sparse` in the same order: messages, inputs, norm, gate, the unit bank's
        grouped pass (one unit per pair, :meth:`UnitBank.forward_grouped`), ``W_out``, the
        candidate norm, ``tanh``. Only inside :meth:`window` (the edge weights and ``W1`` layout
        are the window's)."""
        if not self._in_window:
            raise RuntimeError("PDGCell.replay_sparse is only available inside PDGCell.window()")
        if self.disabled_cells:
            raise ValueError("PDGCell.replay_sparse does not support model.disabled_cells")
        R, _, d = (int(n) for n in h_read.shape)
        K = sel.k
        dtype = h_read.dtype
        valid = sel.valid
        cells = torch.where(valid, sel.cell_idx, torch.zeros_like(sel.cell_idx))
        units = torch.where(valid, sel.unit_idx, torch.zeros_like(sel.unit_idx))
        h_const = h_read.detach()
        graph = self.graph
        D = graph.max_in_degree
        with graph.step():
            # the messages of CellGraph.messages_selected with V applied to the gathered sources
            # only (the same rows of the same product): the state is a constant here, so nothing
            # has to be scattered back over all N cells in the backward
            index = graph._pad_src[cells].reshape(R, K * D, 1).expand(R, K * D, d)
            sources = self.V(h_const.gather(1, index)).reshape(R, K, D, d)
            edge_weights = graph._padded_weights_as(sources.dtype)[cells]  # [R, K, D_max]
            m = (sources * edge_weights[..., None]).sum(dim=2)
            u_read = None if self.ablate_shared_u else u  # model.ablate_shared_u: v without u
            v_raw = native_ops.gather_concat(u_read, eps, h_const, m, cells)  # [R, K, 4d]
            v = self.norm(v_raw)
            b = torch.sigmoid(self.gate(v).to(dtype))
            unit_out = self.units.forward_grouped(
                v.reshape(R * K, self.in_dim), units.reshape(R * K)
            ).reshape(R, K, d)
            if self.cand_norm_kind == "none":
                mixed = sel.alpha.unsqueeze(-1) * unit_out
                c = torch.tanh(self.W_out(mixed).to(dtype))
            else:
                pre = self.cand_norm(self.W_out(unit_out).to(dtype))
                c = torch.tanh(sel.alpha.unsqueeze(-1).to(dtype) * pre)
        return c, b

    # ----------------------------------------------------------------------------------------------
    # Accounting and validation
    # ----------------------------------------------------------------------------------------------

    def param_count(self) -> int:
        """Measured parameter count of ``V``, ``W_out``, the gate, the norms, the units and the
        edge logits."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        text = (
            f"d={self.d}, n_cells={self.n_cells}, n_units={self.n_units}, seed={self.seed}, "
            f"disabled_cells={list(self.disabled_cells)}"
        )
        if self.ablate_shared_u:  # only when set: the default repr is the P1.5 one
            text += ", ablate_shared_u=True"
        return text

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
