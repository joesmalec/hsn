"""GRU byte-level language model (P0.5): the recurrent control of the baseline ladder.

``GRUBaseline`` is a stacked :class:`torch.nn.GRU` between a byte embedding (plus the context
symbol embedding every model in this program reads, ``model.n_context`` symbols) and a linear
output head. It implements the :class:`hsn.train.loop.SequenceModel` protocol: the recurrent state
is the hidden state of every layer, ``[n_layers, B, d_model]``, carried between windows by the
trainer (which detaches it at every window start, truncated BPTT) and reset to zeros wherever
``reset[b, t]`` is True *before* position ``t`` is processed, so a document never sees the state
of the one before it. Resets inside a window are handled by running cuDNN on the maximal
segments between columns where any row resets and masking the state in between, so the number of
kernel launches per window is the number of distinct reset columns plus one, not ``T``.

Sizes come from ``baseline.gru.*`` (``d_model``, ``n_layers``, ``dropout``); the vocabulary and
context symbol count from ``model.vocab_size`` and ``model.n_context``. With the defaults
(d_model 384, 2 layers, V=256, 16 context symbols) the model has 1,977,856 parameters as
measured by :meth:`GRUBaseline.param_report`: the documented ≈2M-parameter baseline. Every
parameter is read at every position, so *active* equals *stored* for this model; the two are
reported separately because HSN and the sparse-expert baseline differ in them (paper §12).

Under CUDA autocast the input and the carried state are cast to the autocast dtype explicitly:
cuDNN's RNN kernels require the input, the hidden state and the (autocast-copied) weights to share
one dtype, and the carried state otherwise keeps whatever dtype the previous window produced. MPS
autocast does not cover ``nn.GRU`` (its weights stay fp32), so there the GRU runs in the master
dtype and only the embeddings, LayerNorm and head follow autocast (P0.8, profile:m_series).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from hsn.config import RunConfig

__all__ = ["GRUBaseline"]


class GRUBaseline(nn.Module):
    """Byte embedding + context embedding → stacked GRU → LayerNorm → linear head."""

    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        spec = cfg.baseline.gru
        self.vocab_size = cfg.model.vocab_size
        self.n_context = cfg.model.n_context
        self.d_model = spec.d_model
        self.n_layers = spec.n_layers
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.ctx_emb = nn.Embedding(self.n_context, self.d_model)
        self.drop = nn.Dropout(spec.dropout)
        self.gru = nn.GRU(
            self.d_model,
            self.d_model,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=spec.dropout if self.n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.ctx_emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # SequenceModel protocol -----------------------------------------------------------------

    def init_state(self, B: int, device: torch.device | str) -> Tensor:
        """Zeros ``[n_layers, B, d_model]`` in the parameters' dtype."""
        if B < 1:
            raise ValueError(f"B must be >= 1, got {B}")
        weight = self.tok_emb.weight
        return torch.zeros(self.n_layers, B, self.d_model, device=device, dtype=weight.dtype)

    def reset_state(self, state: Tensor, mask: Tensor) -> Tensor:
        """Zero the rows of ``state`` where ``mask`` (``[B]`` bool) is True."""
        self._check_state(state, mask.shape[0])
        if mask.shape != (state.shape[1],) or mask.dtype is not torch.bool:
            raise ValueError(
                f"mask must be a bool [B={state.shape[1]}] tensor, got {tuple(mask.shape)}"
            )
        return state.masked_fill(mask.view(1, -1, 1).to(state.device), 0.0)

    def forward(
        self, tokens: Tensor, context: Tensor, state: Tensor, reset: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """``tokens``/``context``/``reset`` are ``[B, T]``; returns logits ``[B, T, V]``, state."""
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [B, T], got shape {tuple(tokens.shape)}")
        B, T = tokens.shape
        for name, value in (("context", context), ("reset", reset)):
            if value.shape != (B, T):
                raise ValueError(f"{name} must have shape {(B, T)}, got {tuple(value.shape)}")
        if reset.dtype is not torch.bool:
            raise TypeError(f"reset must be a bool tensor, got {reset.dtype}")
        self._check_state(state, B)
        x = self.drop(self.tok_emb(tokens) + self.ctx_emb(context))
        h = state
        if x.device.type == "cuda" and torch.is_autocast_enabled(x.device.type):
            # cuDNN: autocast copies the RNN weights to the autocast dtype; input and state match it
            dtype = torch.get_autocast_dtype(x.device.type)
            x = x.to(dtype)
            h = h.to(dtype)
        else:
            # P0.8: MPS (and CPU) autocast does not cover nn.GRU, whose weights stay in the master
            # dtype, so the input and the carried state must be that dtype
            dtype = next(self.gru.parameters()).dtype
            if x.dtype != dtype:
                x = x.to(dtype)
            if h.dtype != dtype:
                h = h.to(dtype)
        # Maximal segments between the columns where any row resets; cuDNN runs on each segment
        # and the state is masked at the segment start (one host sync per window for `cols`).
        cols = reset.any(dim=0).nonzero().flatten().tolist()
        outputs: list[Tensor] = []
        start = 0
        for col in [*cols, T]:
            if col > start:
                out, h = self.gru(x[:, start:col], h)
                outputs.append(out)
                start = col
            if col < T:
                h = h.masked_fill(reset[:, col].view(1, B, 1), 0.0)
        y = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
        logits = self.head(self.drop(self.norm(y)))
        return logits, h, {"state_abs_max": h.detach().abs().max()}  # a tensor: no host sync

    def param_report(self) -> dict[str, int]:
        """Measured parameter counts; ``active`` == ``stored`` for this dense model."""
        parts = {
            "byte_emb": _count(self.tok_emb),
            "ctx_emb": _count(self.ctx_emb),
            "gru": _count(self.gru),
            "norm": _count(self.norm),
            "head": _count(self.head),
        }
        stored = _count(self)
        if sum(parts.values()) != stored:
            raise RuntimeError("GRUBaseline.param_report: component counts do not sum to the total")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {**parts, "stored": stored, "active": stored, "trainable": trainable}

    # helpers ---------------------------------------------------------------------------------

    def _check_state(self, state: Tensor, B: int) -> None:
        expected = (self.n_layers, B, self.d_model)
        if not isinstance(state, Tensor) or tuple(state.shape) != expected:
            got = tuple(state.shape) if isinstance(state, Tensor) else type(state).__name__
            raise ValueError(f"GRUBaseline state must be a tensor of shape {expected}, got {got}")


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
