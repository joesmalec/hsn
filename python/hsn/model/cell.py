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
of the scorer logits, computed by :class:`~hsn.model.hsn.HSN`); the sparse path (P2.5) updates the
selected cells with one unit each through ``UnitBank.forward_selected`` and
``CellGraph.messages_selected``. With ``h = 0`` initially every coordinate of ``h`` stays in
``[−1, 1]``: each update is a coordinatewise convex combination of ``h`` and a ``tanh`` output.

``model.disabled_cells`` is the P1.8 test hook: those cells are never updated (their row of ``h``
is carried through unchanged, with no gradient into the discarded candidate).
``model.ablate_shared_u`` (remove ``u`` from ``v``) is not implemented here: the P1.1 closed forms
fix ``v`` at width ``4d`` and
refuse to count the ablation, so this module refuses to build it rather than guess its shapes.

The cell reads only the ``state`` it is given and returns the new ``h``; writing the state (the
summary ``q``, ages, position) is :class:`~hsn.model.hsn.HSN`'s job through the P1.3 functional
updates.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hsn.config import ModelConfig
from hsn.model.embeddings import init_weight_
from hsn.model.graph import CellGraph
from hsn.model.state import CellGraphState
from hsn.model.units import UnitBank

__all__ = ["PDGCell"]


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
            init_weight_(linear.weight, cfg.init)
        nn.init.zeros_(self.gate.bias)
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
        B, N, d = self._check_inputs(u, eps, state, alpha)
        with self.graph.step():
            h_prev = state.h
            m = self.graph.messages_dense(h_prev, self.V)  # [B, N, d]
            v = torch.cat(
                [u[:, None, :].expand(B, N, d), eps[:, None, :].expand(B, N, d), h_prev, m], dim=-1
            )  # [B, N, 4d]
            unit_out = self.units.forward_all(v)  # [B, N, M, d]
            mixed = (alpha.unsqueeze(-1) * unit_out).sum(dim=2)  # Σ_j α_ij F_j(v_i)
            c = torch.tanh(self.W_out(mixed))
            b = torch.sigmoid(self.gate(v))
            h_new = (1 - b) * h_prev + b * c
            if self.disabled_cells:
                h_new = torch.where(self.disabled[None, :, None], h_prev, h_new)
        info = {
            "m": m,
            "v": v,
            "unit_out": unit_out,
            "mixed": mixed,
            "c": c,
            "b": b,
            "h_prev": h_prev,
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
        if not isinstance(state, CellGraphState):
            raise TypeError(f"state must be a CellGraphState, got {type(state).__name__}")
        B, N, d = state.batch_size, state.n_cells, state.width
        if N != self.n_cells or d != self.d:
            raise ValueError(
                f"state has N = {N}, d = {d}; this PDGCell expects N = {self.n_cells}, d = {self.d}"
            )
        for name, x, shape in (
            ("u", u, (B, d)),
            ("eps", eps, (B, d)),
            ("alpha", alpha, (B, N, self.n_units)),
        ):
            if not isinstance(x, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
            if x.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(x.shape)}")
            if x.dtype != state.dtype:
                raise ValueError(
                    f"{name} must have dtype {state.dtype} like the state, got {x.dtype}"
                )
            if x.device != state.device:
                raise ValueError(f"{name} must live on {state.device}, got {x.device}")
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
