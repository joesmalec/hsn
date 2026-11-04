"""The HSN module (P1.5): one complete input step in dense mode, wired in the order of plan §2.2,
and the loop over a window of positions with document resets.

For every position ``t`` (:meth:`HSN.step`; ``q = q_{t−1}`` is the carried summary)::

    reset (HSN.forward, before the step)
    ẑ_t   = P(q_{t−1})                         predict BEFORE encoding x_t (global rule 2)
    z_t   = E(emb(x_t))
    z*_t  = emb(x_t).detach()  |  E_ema(emb(x_t))   the prediction target (P1.6; detached)
    ε_t   = z_t − ẑ_t
    u_t   = S([z_t ; q_{t−1} ; ctx_t])
    feats = scorer.features(u_t, ε_t, state, ctx_t, budget, round 0)   [B, N, 3d+8]
    α     = softmax(scorer(feats)) over units                         [B, N, M]
    h_t   = PDGCell.step_dense(u_t, ε_t, state, α)                    every cell, one round
    q_t   = mean_i h[i, t]      (CellGraphState.with_dense: exact, no incremental drift)
    logits = D([q_t ; u_t])     the UPDATED summary

The dense reference uses the same linear scorer as the sparse model, with ``α = softmax(logits)``
over all units and every cell updated (Appendix H.5), so the measured parameter count is exactly
Appendix C.1: :meth:`HSN.param_report` lists the twelve C.1 rows and their ``total`` (1,820,361 for
``configs/ref_a1.yaml``). The learned edge weights of the P1.4 graph (``N·(D_g + random_edges)``
logits, 128 for A.1) are trainable parameters that Appendix C does not list; the report gives them
as ``edges`` and ``total_trainable = total + edges``, which is what a run manifest's
``param_count`` must record (§0.5: measured, not assumed).

Two values the step needs live outside ``model.*``: the graph topology seed (``run.seed``, P1.4)
and the normaliser of the position feature (``train.unroll``). Both are required keyword
arguments of :class:`HSN`; :meth:`HSN.from_config` reads them from a :class:`~hsn.config.RunConfig`.

P1.6: ``StepOutput.z_target`` is the target of the prediction loss (:mod:`hsn.losses`), chosen by
``model.predictor_target``: the detached byte embedding of ``x_t`` (``fixed_embedding``) or the
output of the stop-gradient EMA copy of ``E`` (``ema_encoder``,
:class:`~hsn.model.encoder.EMATarget`, built only in that mode and updated by
:meth:`HSN.update_target` after each optimiser step with ``train.ema_tau``, a third keyword that
is required in that mode). The target is detached in both modes and never feeds the step; ``ẑ``
still comes from ``q_{t−1}`` alone.

Scope of P1.5: dense mode only (``mode="sparse"`` raises until P2.5), one round (the
:class:`~hsn.model.halting.HaltingHead` is owned but not called until P2.7), no custom kernels,
no ``torch.compile``. ``StepOutput.cost`` is the executed cost as a fraction of the dense step
(1.0 here; P2.1 replaces it with profiled costs) and ``n_rounds`` is 1.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import ModelConfig, RunConfig
from hsn.model.cell import PDGCell
from hsn.model.embeddings import ByteEmbedding, ContextEmbedding
from hsn.model.encoder import Decoder, EMATarget, Encoder, Predictor, SharedTransform
from hsn.model.halting import HaltingHead
from hsn.model.router import Scorer
from hsn.model.state import CellGraphState

__all__ = ["PARAM_REPORT_ROWS", "HSN", "StepMode", "StepOutput"]

StepMode = Literal["dense", "sparse"]
_INDEX_DTYPES = (torch.int64, torch.int32)

#: Appendix C.1 rows in table order, keyed as :func:`hsn.train.memory_plan.expected_param_count`
#: keys them.
PARAM_REPORT_ROWS: tuple[str, ...] = (
    "byte_emb",
    "ctx_emb",
    "encoder",
    "predictor",
    "shared",
    "units",
    "w_out",
    "v_msg",
    "gate",
    "scorer",
    "halting",
    "decoder",
)


@dataclass
class StepOutput:
    """Everything one position produces: next-symbol ``logits [B, V]``, the encoded input ``z``,
    the prediction ``z_hat`` made before encoding, the detached prediction target ``z_target``
    (P1.6, by ``model.predictor_target``), the error ``eps = z − z_hat`` and the shared features
    ``u`` (all ``[B, d]``), the updated ``state``, the executed ``cost`` (fraction of the dense
    step), ``n_rounds`` and ``aux`` statistics (detached 0-dim tensors: ``state_abs_max``,
    ``eps_abs_mean``, ``alpha_entropy``)."""

    logits: Tensor
    z: Tensor
    z_hat: Tensor
    z_target: Tensor
    eps: Tensor
    u: Tensor
    state: CellGraphState
    cost: float
    n_rounds: int
    aux: dict[str, Tensor]


class HSN(nn.Module):
    """The dense predictive cell model: embeddings, ``E``, ``P``, ``S``, the :class:`PDGCell`, the
    :class:`Scorer`, the :class:`HaltingHead` and ``D``, built from ``cfg`` (``model.*``) plus
    ``seed`` (``run.seed``), ``unroll`` (``train.unroll``) and, when
    ``model.predictor_target = ema_encoder``, ``ema_tau`` (``train.ema_tau``) for the
    :class:`~hsn.model.encoder.EMATarget` copy of ``E`` (``ema_target``; ``None`` otherwise)."""

    def __init__(
        self, cfg: ModelConfig, *, seed: int, unroll: int, ema_tau: float | None = None
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.seed = seed
        self.unroll = unroll
        self.ema_tau = ema_tau
        self.byte_emb = ByteEmbedding.from_config(cfg)
        self.ctx_emb = ContextEmbedding.from_config(cfg)
        self.encoder = Encoder(cfg)
        self.ema_target: EMATarget | None = None
        if cfg.predictor_target == "ema_encoder":
            if ema_tau is None:
                raise ValueError(
                    "model.predictor_target=ema_encoder needs train.ema_tau: pass ema_tau=... or "
                    "build the model with HSN.from_config(run)"
                )
            self.ema_target = EMATarget(self.encoder, ema_tau)
        elif cfg.predictor_target != "fixed_embedding":
            raise ValueError(
                f"model.predictor_target={cfg.predictor_target!r} is not implemented; known: "
                "fixed_embedding, ema_encoder"
            )
        self.predictor = Predictor(cfg)
        self.shared = SharedTransform(cfg)
        self.cell = PDGCell(cfg, seed=seed)
        self.scorer = Scorer(cfg, unroll=unroll)
        self.halting = HaltingHead(cfg)
        self.decoder = Decoder(cfg)

    @classmethod
    def from_config(cls, run: RunConfig) -> Self:
        """``HSN(run.model, seed=run.seed, unroll=run.train.unroll, ema_tau=run.train.ema_tau)``."""
        if not isinstance(run, RunConfig):
            raise TypeError(f"from_config needs a RunConfig, got {type(run).__name__}")
        return cls(run.model, seed=run.seed, unroll=run.train.unroll, ema_tau=run.train.ema_tau)

    # -- placement --------------------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.decoder.fc1.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.decoder.fc1.weight.dtype

    def init_state(self, B: int, device: torch.device | str | None = None) -> CellGraphState:
        """A fresh all-zero state for ``B`` sessions on ``device`` (the model's device by default),
        in the parameters' dtype."""
        dev = self.device if device is None else torch.device(device)
        return CellGraphState.zeros(B, self.cfg.n_cells, self.cfg.d, dev, self.dtype)

    # -- one position -----------------------------------------------------------------------------

    def step(
        self, x_t: Tensor, ctx_t: Tensor, state: CellGraphState, mode: StepMode = "dense"
    ) -> StepOutput:
        """One input position in the §2.2 order (see the module docstring); returns a
        :class:`StepOutput` whose ``state`` is the functional update of ``state`` (autograd intact).

        ``x_t`` and ``ctx_t`` are int64/int32 ``[B]`` on the state's device; ``state`` must have
        this model's ``N`` and ``d`` and the parameters' dtype and device. Only ``mode="dense"``
        exists in P1.5; ``"sparse"`` raises ``NotImplementedError`` (P2.5).
        """
        if mode == "sparse":
            raise NotImplementedError(
                "mode='sparse' (selected cells, one unit each) is P2.5; P1.5 implements the dense "
                "reference only"
            )
        if mode != "dense":
            raise ValueError(f"mode must be 'dense' or 'sparse', got {mode!r}")
        self._check_step_inputs(x_t, ctx_t, state)
        cfg = self.cfg
        q_prev = state.q

        z_hat = self.predictor(q_prev)  # before x_t is encoded
        emb = self.byte_emb(x_t)
        z = self.encoder(emb)
        z_target = self._prediction_target(emb)  # detached; a function of x_t only
        eps = z - z_hat
        u = self.shared(torch.cat([z, q_prev, self.ctx_emb(ctx_t)], dim=-1))

        with self.cell.graph.step():
            feats = self.scorer.features(
                u, eps, state, ctx_t, budget_frac=cfg.budget, round_idx=0, ema=None
            )
            unit_logits = self.scorer(feats)  # [B, N, M]
            alpha = torch.softmax(unit_logits, dim=-1)
            h_new, _ = self.cell.step_dense(u, eps, state, alpha)

        new_state = self._write_dense(state, h_new)
        logits = self.decoder(torch.cat([new_state.q, u], dim=-1))  # the updated q_t

        with torch.no_grad():
            log_alpha = F.log_softmax(unit_logits, dim=-1)
            aux = {
                "state_abs_max": new_state.h.abs().amax(),
                "eps_abs_mean": eps.abs().mean(),
                "alpha_entropy": -(alpha * log_alpha).sum(dim=-1).mean(),
            }
        return StepOutput(
            logits=logits,
            z=z,
            z_hat=z_hat,
            z_target=z_target,
            eps=eps,
            u=u,
            state=new_state,
            cost=1.0,
            n_rounds=1,
            aux=aux,
        )

    def _prediction_target(self, emb: Tensor) -> Tensor:
        """``z_target [B, d]`` for the prediction loss, detached in both modes: the byte
        embedding ``emb`` itself (``fixed_embedding``) or the EMA copy of ``E`` applied to it
        (``ema_encoder``; the copy runs under ``no_grad``)."""
        if self.ema_target is None:
            return emb.detach()
        target: Tensor = self.ema_target(emb.detach())
        return target

    def update_target(self) -> None:
        """Move the EMA target encoder toward the online ``E`` (``train.ema_tau``); call it after
        each optimiser step. A no-op with the fixed-embedding target."""
        if self.ema_target is not None:
            self.ema_target.update()

    def _write_dense(self, state: CellGraphState, h_new: Tensor) -> CellGraphState:
        """Write the dense round into the state: every enabled cell replaced (``q`` the exact mean,
        which stays exact because disabled cells keep their old ``h``), updated cells at age 0,
        disabled cells keep counting, ``pos += 1``."""
        updated = state.with_dense(h_new)
        if not self.cell.disabled_cells:
            return updated.tick(torch.ones_like(state.age, dtype=torch.bool))
        enabled = (~self.cell.disabled)[None, :].expand_as(state.age)
        return replace(updated, age=state.age.masked_fill(enabled, 0)).tick(enabled)

    # -- a window of positions --------------------------------------------------------------------

    def forward(
        self, tokens: Tensor, ctx: Tensor, state: CellGraphState, reset: Tensor
    ) -> tuple[Tensor, CellGraphState, dict[str, Tensor]]:
        """Run positions ``0 .. T−1`` of ``tokens [B, T]`` / ``ctx [B, T]`` from ``state``.

        ``reset [B, T]`` (bool) zeroes a session's state *before* position ``t`` is processed (a
        document boundary at ``t``). Returns ``logits [B, T, V]``, the final state and ``info``:
        the per-position stacks ``z``, ``z_hat``, ``z_target`` (detached), ``eps``, ``u``
        (``[B, T, d]``, autograd intact except for the target) and the per-position statistics
        ``state_abs_max``, ``eps_abs_mean``, ``alpha_entropy``, ``cost`` (float) and ``n_rounds``
        (int64), each ``[T]``.
        """
        _, T = self._check_window_inputs(tokens, ctx, state, reset)
        logits: list[Tensor] = []
        stacks: dict[str, list[Tensor]] = {
            "z": [],
            "z_hat": [],
            "z_target": [],
            "eps": [],
            "u": [],
        }
        stats: dict[str, list[Tensor]] = {
            "state_abs_max": [],
            "eps_abs_mean": [],
            "alpha_entropy": [],
        }
        costs: list[float] = []
        rounds: list[int] = []
        with self.cell.graph.step():
            for t in range(T):
                state = state.reset(reset[:, t])
                out = self.step(tokens[:, t], ctx[:, t], state)
                state = out.state
                logits.append(out.logits)
                for name in stacks:
                    stacks[name].append(getattr(out, name))
                for name in stats:
                    stats[name].append(out.aux[name])
                costs.append(out.cost)
                rounds.append(out.n_rounds)
        info: dict[str, Tensor] = {
            name: torch.stack(values, dim=1) for name, values in stacks.items()
        }
        info.update({name: torch.stack(values) for name, values in stats.items()})
        info["cost"] = torch.tensor(costs, dtype=self.dtype, device=state.device)
        info["n_rounds"] = torch.tensor(rounds, dtype=torch.int64, device=state.device)
        return torch.stack(logits, dim=1), state, info

    # -- accounting -------------------------------------------------------------------------------

    def param_report(self) -> dict[str, int]:
        """Measured parameter counts: the Appendix C.1 rows in table order and their ``total``
        (the closed form of :func:`hsn.train.memory_plan.expected_param_count`), then ``edges``
        (the P1.4 edge logits, outside Appendix C) and ``total_trainable`` (every parameter of
        this module; the run manifest's ``param_count``)."""
        modules: dict[str, nn.Module] = {
            "byte_emb": self.byte_emb,
            "ctx_emb": self.ctx_emb,
            "encoder": self.encoder,
            "predictor": self.predictor,
            "shared": self.shared,
            "units": self.cell.units,
            "w_out": self.cell.W_out,
            "v_msg": self.cell.V,
            "gate": self.cell.gate,
            "scorer": self.scorer,
            "halting": self.halting,
            "decoder": self.decoder,
        }
        assert tuple(modules) == PARAM_REPORT_ROWS
        report = {name: sum(p.numel() for p in m.parameters()) for name, m in modules.items()}
        report["total"] = sum(report.values())
        report["edges"] = self.cell.graph.edge_logit.numel()
        report["total_trainable"] = sum(p.numel() for p in self.parameters())
        if report["total_trainable"] != report["total"] + report["edges"]:
            raise RuntimeError(
                f"param_report: {report['total_trainable']} trainable parameters, but the Appendix "
                f"C rows ({report['total']}) plus the edge logits ({report['edges']}) give "
                f"{report['total'] + report['edges']}; a submodule carries unaccounted parameters"
            )
        return report

    def extra_repr(self) -> str:
        text = f"seed={self.seed}, unroll={self.unroll}"
        if self.ema_target is not None:
            text += f", ema_tau={self.ema_tau}"
        return text

    # -- validation -------------------------------------------------------------------------------

    def _check_state(self, state: CellGraphState) -> None:
        if not isinstance(state, CellGraphState):
            raise TypeError(f"state must be a CellGraphState, got {type(state).__name__}")
        cfg = self.cfg
        if state.n_cells != cfg.n_cells or state.width != cfg.d:
            raise ValueError(
                f"state has N = {state.n_cells}, d = {state.width}; this model expects "
                f"N = {cfg.n_cells}, d = {cfg.d}"
            )
        if state.dtype != self.dtype:
            raise ValueError(
                f"state has dtype {state.dtype} but the parameters {self.dtype}; build it with "
                "init_state"
            )
        if state.device != self.device:
            raise ValueError(
                f"state lives on {state.device} but the parameters on {self.device}; move the "
                "model with .to(device) or build the state with init_state"
            )

    @staticmethod
    def _check_ids(name: str, ids: Tensor, shape: tuple[int, ...], device: torch.device) -> None:
        if not isinstance(ids, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor of ids, got {type(ids).__name__}")
        if ids.dtype not in _INDEX_DTYPES:
            raise TypeError(
                f"{name} must be an int64 or int32 tensor of ids, got dtype {ids.dtype}"
            )
        if ids.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(ids.shape)}")
        if ids.device != device:
            raise ValueError(f"{name} must live on {device}, got {ids.device}")

    def _check_step_inputs(self, x_t: Tensor, ctx_t: Tensor, state: CellGraphState) -> None:
        self._check_state(state)
        B = state.batch_size
        self._check_ids("x_t", x_t, (B,), state.device)
        self._check_ids("ctx_t", ctx_t, (B,), state.device)

    def _check_window_inputs(
        self, tokens: Tensor, ctx: Tensor, state: CellGraphState, reset: Tensor
    ) -> tuple[int, int]:
        self._check_state(state)
        B = state.batch_size
        if not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.shape[0] != B:
            raise ValueError(
                f"tokens must be a [B, T] tensor with B = {B}, got "
                f"{tuple(tokens.shape) if isinstance(tokens, Tensor) else type(tokens).__name__}"
            )
        T = int(tokens.shape[1])
        if T < 1:
            raise ValueError("tokens must hold at least one position (T >= 1)")
        self._check_ids("tokens", tokens, (B, T), state.device)
        self._check_ids("ctx", ctx, (B, T), state.device)
        if not isinstance(reset, Tensor):
            raise TypeError(f"reset must be a bool torch.Tensor, got {type(reset).__name__}")
        if reset.dtype != torch.bool:
            raise ValueError(f"reset must have dtype torch.bool, got {reset.dtype}")
        if reset.shape != (B, T):
            raise ValueError(f"reset must have shape [B, T] = {(B, T)}, got {tuple(reset.shape)}")
        if reset.device != state.device:
            raise ValueError(f"reset must live on {state.device}, got {reset.device}")
        return B, T
