"""Halting head (P1.5 defines it, P2.7 uses it): ``Linear(2d → 1)`` over ``[q; u]`` giving the
probability of stopping after the current internal round.

Plan P2.7 and Appendix C.1 (row ``halting``: ``2d + 1`` parameters, 129 for A.1). While
``model.r_max = 1`` there is only one round and nothing to halt, so :class:`~hsn.model.hsn.HSN`
owns a head (it is part of the reference parameter count) but never calls it. P2.7 restructures
the step into a round loop that reads ``p_halt`` after every round, stops at
``model.halt_threshold``
at inference (never below one round, never above ``r_max``) and trains it through a ponder loss.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hsn.config import ModelConfig
from hsn.model.embeddings import init_weight_

__all__ = ["HaltingHead"]


class HaltingHead(nn.Module):
    """``forward(q [B, d], u [B, d]) -> p_halt [B]`` = ``σ(Linear([q; u]))``."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.d = cfg.d
        self.threshold = cfg.halt_threshold
        self.linear = nn.Linear(2 * cfg.d, 1)
        init_weight_(self.linear.weight, cfg.init)
        nn.init.zeros_(self.linear.bias)

    def forward(self, q: Tensor, u: Tensor) -> Tensor:
        for name, x in (("q", q), ("u", u)):
            if not isinstance(x, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
            if x.ndim != 2 or x.shape[-1] != self.d:
                raise ValueError(f"{name} must have shape [B, {self.d}], got {tuple(x.shape)}")
        if q.shape[0] != u.shape[0]:
            raise ValueError(
                f"q and u must share the batch size, got {q.shape[0]} and {u.shape[0]}"
            )
        out: Tensor = torch.sigmoid(self.linear(torch.cat([q, u], dim=-1))).squeeze(-1)
        return out

    def param_count(self) -> int:
        """Measured parameter count (``2d + 1``)."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return f"d={self.d}, threshold={self.threshold}"
