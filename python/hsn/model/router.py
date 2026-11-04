"""Router scorer and its feature builder (P1.5): the fixed ``3d + 8`` per-cell feature layout and
the scorer that maps it to ``M`` unit logits.

Plan §2.5 and Appendix H.2/H.5. Every candidate (cell, unit) is scored from information available at
position ``t`` before any cell is updated (global rule 2). The dense reference (P1.5) and the sparse
model (P2.1) share this one scorer: in dense mode ``α = softmax(logits)`` over the units of every
cell, so the parameter count stays at Appendix C's 1,820,361 (``scorer`` row: ``(3d + 8) × M + M``,
1,608 for A.1); P2.1 adds selection (top-K, budgets, deadlines) beside it in this module. The scorer
is linear by default; ``model.scorer = mlp`` swaps in a two-layer :class:`~hsn.model.encoder.MLP2`
with hidden width ``model.scorer_hidden`` (the C.2 "two-layer scorer" row), which costs more per
position than the four selected cells execute (plan §2.5).

Feature layout (:data:`SCORER_FEATURE_NAMES`, width ``3d + 8``; 200 for A.1), per cell ``i``::

    [ u_t (d) ; ε_t (d) ; h[i, t, r−1] (d) ;
      age_i / age_max ; budget_remaining_fraction ; round_index / R_max ; context_id / n_ctx ;
      ema_utility_i ; ema_selected_i ; pos / unroll ; 1.0 ]

``u_t`` and ``ε_t`` are shared by every cell of a batch row; ``h`` is the cell's state from the
previous round (``state.h``); ``age``/``pos`` come from the
:class:`~hsn.model.state.CellGraphState`.
``budget_remaining_fraction`` is ``B_t`` minus what earlier rounds spent, as a fraction of the dense
step cost (``model.budget`` before round 1); ``round_index`` is 0-based (rounds already completed at
this position; always 0 while ``R_max = 1``), so the feature is ``round_idx / r_max``;
``context_id / n_ctx`` uses the raw context symbol id. ``ema_utility`` and ``ema_selected`` are the
P2.3 running averages and are **zero** until P2.3 passes an :class:`EMAFeatures`; ``pos / unroll``
normalises the session position by ``train.unroll``; the last feature is the constant 1.
``model.scorer_features.exclude`` zeroes the named blocks (an ablation) without changing the width,
so no parameter count changes; a name outside :data:`SCORER_FEATURE_NAMES` raises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hsn.config import ModelConfig
from hsn.model.embeddings import init_weight_
from hsn.model.encoder import MLP2
from hsn.model.state import CellGraphState
from hsn.train.memory_plan import SCORER_SCALAR_FEATURES

__all__ = ["SCORER_FEATURE_NAMES", "EMAFeatures", "Scorer"]

#: Feature blocks in layout order: three width-``d`` blocks, then the eight scalars (Appendix H.2).
SCORER_FEATURE_NAMES: tuple[str, ...] = (
    "u",
    "eps",
    "h",
    "age",
    "budget",
    "round",
    "context",
    "ema_utility",
    "ema_selected",
    "pos",
    "one",
)
_VECTOR_FEATURES = ("u", "eps", "h")
_SCALAR_FEATURES = SCORER_FEATURE_NAMES[len(_VECTOR_FEATURES) :]
if len(_SCALAR_FEATURES) != SCORER_SCALAR_FEATURES:  # pragma: no cover - layout and P1.1 agree
    raise RuntimeError(
        f"the P1.5 layout has {len(_SCALAR_FEATURES)} scalar features but memory_plan counts "
        f"{SCORER_SCALAR_FEATURES}"
    )
_INDEX_DTYPES = (torch.int64, torch.int32)


@dataclass(frozen=True)
class EMAFeatures:
    """The P2.3 running averages per cell: ``utility`` (EMA of the counterfactual utility) and
    ``selected`` (EMA of the selection indicator), each ``[B, N]`` or ``[N]`` (broadcast over the
    batch). Until P2.3 no caller has them and the scorer sees zeros."""

    utility: Tensor
    selected: Tensor


class Scorer(nn.Module):
    """Per-cell features ``[B, N, 3d+8]`` and the map to unit logits ``[B, N, M]``.

    ``cfg`` supplies every width and the normalisers (``d``, ``n_cells``, ``n_units``, ``age_max``,
    ``r_max``, ``n_context``, ``scorer``, ``scorer_hidden``, ``scorer_features.exclude``,
    ``activation``, ``init``); ``unroll`` is ``train.unroll``, the normaliser of the position
    feature, which lives outside ``model.*`` and is therefore a required argument.
    """

    _excluded: Tensor  # bool [3d+8]: True at zeroed feature slots (non-persistent)

    def __init__(self, cfg: ModelConfig, *, unroll: int) -> None:
        super().__init__()
        if isinstance(unroll, bool) or not isinstance(unroll, int) or unroll < 1:
            raise ValueError(f"Scorer: unroll (train.unroll) must be an int >= 1, got {unroll!r}")
        self.d = cfg.d
        self.n_cells = cfg.n_cells
        self.n_units = cfg.n_units
        self.age_max = cfg.age_max
        self.r_max = cfg.r_max
        self.n_context = cfg.n_context
        self.unroll = unroll
        self.kind = cfg.scorer
        self.in_features = 3 * cfg.d + SCORER_SCALAR_FEATURES

        unknown = [name for name in cfg.scorer_features.exclude if name not in SCORER_FEATURE_NAMES]
        if unknown:
            raise ValueError(
                f"model.scorer_features.exclude names unknown features {unknown}; known: "
                f"{list(SCORER_FEATURE_NAMES)}"
            )
        self.exclude: tuple[str, ...] = tuple(dict.fromkeys(cfg.scorer_features.exclude))

        self.feature_slices: dict[str, slice] = {}
        start = 0
        for name in SCORER_FEATURE_NAMES:
            width = cfg.d if name in _VECTOR_FEATURES else 1
            self.feature_slices[name] = slice(start, start + width)
            start += width
        assert start == self.in_features

        self.net: nn.Linear | MLP2
        if cfg.scorer == "linear":
            self.net = nn.Linear(self.in_features, cfg.n_units)
            init_weight_(self.net.weight, cfg.init)
            nn.init.zeros_(self.net.bias)
        elif cfg.scorer == "mlp":
            self.net = MLP2(
                self.in_features, cfg.scorer_hidden, cfg.n_units, cfg.activation, init=cfg.init
            )
        else:  # pragma: no cover - ModelConfig.scorer is a Literal
            raise ValueError(f"model.scorer={cfg.scorer!r} is not supported (linear | mlp)")

        excluded = torch.zeros(self.in_features, dtype=torch.bool)
        for name in self.exclude:
            excluded[self.feature_slices[name]] = True
        self.register_buffer("_excluded", excluded, persistent=False)

    # ----------------------------------------------------------------------------------------------
    # Features
    # ----------------------------------------------------------------------------------------------

    def features(
        self,
        u: Tensor,
        eps: Tensor,
        state: CellGraphState,
        ctx: Tensor,
        budget_frac: float,
        round_idx: int,
        ema: EMAFeatures | None = None,
    ) -> Tensor:
        """The ``[B, N, 3d+8]`` feature tensor in the module's layout, in ``u``'s dtype.

        ``u``, ``eps`` are ``[B, d]`` (the dtype and device of ``state.h``); ``ctx`` the context ids
        ``[B]`` (int64/int32); ``budget_frac`` a finite number >= 0; ``round_idx`` an int in
        ``[0, r_max)``; ``ema`` the P2.3 averages or ``None`` (zeros). Shape, dtype and device
        mismatches raise.
        """
        B, N, d = state.batch_size, state.n_cells, state.width
        if N != self.n_cells or d != self.d:
            raise ValueError(
                f"state has N = {N}, d = {d}; this Scorer expects N = {self.n_cells}, d = {self.d}"
            )
        self._check_vector(u, "u", B, state)
        self._check_vector(eps, "eps", B, state)
        if not isinstance(ctx, Tensor):
            raise TypeError(f"ctx must be a torch.Tensor of context ids, got {type(ctx).__name__}")
        if ctx.dtype not in _INDEX_DTYPES:
            raise TypeError(f"ctx must be an int64 or int32 tensor of ids, got dtype {ctx.dtype}")
        if ctx.shape != (B,):
            raise ValueError(f"ctx must have shape [B] = {(B,)}, got {tuple(ctx.shape)}")
        if ctx.device != state.device:
            raise ValueError(f"ctx must live on {state.device}, got {ctx.device}")
        if (
            isinstance(budget_frac, bool)
            or not isinstance(budget_frac, int | float)
            or not math.isfinite(budget_frac)
            or budget_frac < 0.0
        ):
            raise ValueError(f"budget_frac must be a finite number >= 0, got {budget_frac!r}")
        if isinstance(round_idx, bool) or not isinstance(round_idx, int):
            raise TypeError(f"round_idx must be an int, got {round_idx!r}")
        if not 0 <= round_idx < self.r_max:
            raise ValueError(
                f"round_idx={round_idx} must lie in [0, r_max) = [0, {self.r_max}) (0-based: "
                "rounds already completed at this position)"
            )

        dtype, device = u.dtype, u.device
        age = state.age.to(dtype) / self.age_max  # [B, N]
        budget = torch.full((B, N), float(budget_frac), dtype=dtype, device=device)
        rnd = torch.full((B, N), round_idx / self.r_max, dtype=dtype, device=device)
        context = (ctx.to(dtype) / self.n_context)[:, None].expand(B, N)
        if ema is None:
            ema_utility = torch.zeros(B, N, dtype=dtype, device=device)
            ema_selected = torch.zeros(B, N, dtype=dtype, device=device)
        else:
            ema_utility = self._check_ema(ema.utility, "utility", B, N, state).to(dtype)
            ema_selected = self._check_ema(ema.selected, "selected", B, N, state).to(dtype)
        pos = (state.pos.to(dtype) / self.unroll)[:, None].expand(B, N)
        one = torch.ones(B, N, dtype=dtype, device=device)
        scalars = torch.stack(
            [age, budget, rnd, context, ema_utility, ema_selected, pos, one], dim=-1
        )  # [B, N, 8] in _SCALAR_FEATURES order
        feats = torch.cat(
            [u[:, None, :].expand(B, N, d), eps[:, None, :].expand(B, N, d), state.h, scalars],
            dim=-1,
        )
        if self.exclude:
            feats = feats.masked_fill(self._excluded, 0)
        return feats

    def forward(self, feats: Tensor) -> Tensor:
        """``feats [B, N, 3d+8] -> logits [B, N, M]``, one score per (cell, unit)."""
        if not isinstance(feats, Tensor):
            raise TypeError(f"feats must be a torch.Tensor, got {type(feats).__name__}")
        if feats.ndim != 3 or feats.shape[-1] != self.in_features:
            raise ValueError(
                f"feats must have shape [B, N, {self.in_features}], got {tuple(feats.shape)}"
            )
        out: Tensor = self.net(feats)
        return out

    # ----------------------------------------------------------------------------------------------
    # Accounting and validation
    # ----------------------------------------------------------------------------------------------

    def param_count(self) -> int:
        """Measured parameter count (``(3d+8)·M + M`` linear; ``_mlp(3d+8, scorer_hidden, M)``)."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"kind={self.kind!r}, in_features={self.in_features}, n_units={self.n_units}, "
            f"n_cells={self.n_cells}, age_max={self.age_max}, r_max={self.r_max}, "
            f"n_context={self.n_context}, unroll={self.unroll}, exclude={list(self.exclude)}"
        )

    @staticmethod
    def _check_vector(x: Tensor, name: str, B: int, state: CellGraphState) -> None:
        if not isinstance(x, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
        if x.shape != (B, state.width):
            raise ValueError(
                f"{name} must have shape [B, d] = {(B, state.width)}, got {tuple(x.shape)}"
            )
        if x.dtype != state.dtype:
            raise ValueError(f"{name} must have dtype {state.dtype} like the state, got {x.dtype}")
        if x.device != state.device:
            raise ValueError(f"{name} must live on {state.device}, got {x.device}")

    @staticmethod
    def _check_ema(x: Tensor, name: str, B: int, N: int, state: CellGraphState) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"EMAFeatures.{name} must be a torch.Tensor, got {type(x).__name__}")
        if not x.is_floating_point():
            raise ValueError(f"EMAFeatures.{name} must be floating point, got {x.dtype}")
        if x.device != state.device:
            raise ValueError(f"EMAFeatures.{name} must live on {state.device}, got {x.device}")
        if x.shape == (N,):
            return x[None, :].expand(B, N)
        if x.shape == (B, N):
            return x
        raise ValueError(
            f"EMAFeatures.{name} must have shape [N] = {(N,)} or [B, N] = {(B, N)}, got "
            f"{tuple(x.shape)}"
        )
