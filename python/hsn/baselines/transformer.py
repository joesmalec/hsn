"""Small causal Transformer byte-level language model (P0.5): the attention control.

``TransformerBaseline`` is a pre-norm Transformer (GPT-2 layout: LayerNorm → multi-head causal
self-attention → residual, LayerNorm → GELU MLP → residual) over the byte embedding plus the
context-symbol embedding plus a learned position embedding. Its context window is the TBPTT
window ``train.unroll``: every window is processed on its own, with positions ``0..T-1`` relative
to the window start, and there is no key/value cache, so **it never attends across window
boundaries** (chunk pitfall). It is stateless under the :class:`hsn.train.loop.SequenceModel`
protocol: ``init_state`` and ``reset_state`` return ``None``. Inside a window the ``reset`` flags
split each row into documents and the attention mask is block-diagonal over those segments (and
causal), so a document never attends to the one packed before it; that is the transformer's
counterpart of the GRU's state reset.

Sizes come from ``baseline.transformer.*`` (``d_model``, ``n_layers``, ``n_heads``, ``ffn_hidden``,
``dropout``); the vocabulary and context symbol count from ``model.vocab_size`` and
``model.n_context``; the window from ``train.unroll``. With the defaults (d_model 192, 4 layers,
4 heads, MLP 768, V=256, 16 context symbols, unroll 256) the model has 1,930,624 parameters as
measured by :meth:`TransformerBaseline.param_report`: the documented ≈2M-parameter baseline.
Every parameter is read at every position, so *active* equals *stored* for this model.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import RunConfig

__all__ = ["TransformerBaseline"]


class _Block(nn.Module):
    """One pre-norm block: causal multi-head self-attention and a GELU MLP, both residual."""

    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.norm_attn = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.fc_in = nn.Linear(d_model, ffn_hidden)
        self.fc_out = nn.Linear(ffn_hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(self.norm_attn(x)).split(d, dim=-1)
        q = q.view(B, T, self.n_heads, -1).transpose(1, 2)  # [B, H, T, d_head]
        k = k.view(B, T, self.n_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.n_heads, -1).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.drop(self.proj(attn.transpose(1, 2).reshape(B, T, d)))
        x = x + self.drop(self.fc_out(F.gelu(self.fc_in(self.norm_mlp(x)))))
        return x


class TransformerBaseline(nn.Module):
    """Byte + context + position embeddings → ``n_layers`` pre-norm blocks → LayerNorm → head."""

    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        spec = cfg.baseline.transformer
        self.vocab_size = cfg.model.vocab_size
        self.n_context = cfg.model.n_context
        self.d_model = spec.d_model
        self.n_layers = spec.n_layers
        self.n_heads = spec.n_heads
        self.window = cfg.train.unroll  # the context window: never attends across windows
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.ctx_emb = nn.Embedding(self.n_context, self.d_model)
        self.pos_emb = nn.Embedding(self.window, self.d_model)
        self.drop = nn.Dropout(spec.dropout)
        self.blocks = nn.ModuleList(
            _Block(self.d_model, self.n_heads, spec.ffn_hidden, spec.dropout)
            for _ in range(self.n_layers)
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    # SequenceModel protocol -----------------------------------------------------------------

    def init_state(self, B: int, device: torch.device | str) -> None:
        """Stateless: there is nothing to carry between windows."""
        if B < 1:
            raise ValueError(f"B must be >= 1, got {B}")
        return None

    def reset_state(self, state: None, mask: Tensor) -> None:
        if state is not None:
            raise ValueError(f"TransformerBaseline carries no state, got {type(state).__name__}")
        if mask.dtype is not torch.bool or mask.dim() != 1:
            raise ValueError(f"mask must be a bool [B] tensor, got {tuple(mask.shape)}")
        return None

    def forward(
        self, tokens: Tensor, context: Tensor, state: None, reset: Tensor
    ) -> tuple[Tensor, None, dict[str, Any]]:
        """``tokens``/``context``/``reset`` are ``[B, T]`` with ``T <= train.unroll``."""
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [B, T], got shape {tuple(tokens.shape)}")
        B, T = tokens.shape
        if T > self.window:
            raise ValueError(
                f"window of {T} positions exceeds the context window train.unroll={self.window}"
            )
        for name, value in (("context", context), ("reset", reset)):
            if value.shape != (B, T):
                raise ValueError(f"{name} must have shape {(B, T)}, got {tuple(value.shape)}")
        if reset.dtype is not torch.bool:
            raise TypeError(f"reset must be a bool tensor, got {reset.dtype}")
        if state is not None:
            raise ValueError(f"TransformerBaseline carries no state, got {type(state).__name__}")
        positions = torch.arange(T, device=tokens.device)
        x = self.drop(self.tok_emb(tokens) + self.ctx_emb(context) + self.pos_emb(positions))
        mask = self.attention_mask(reset)
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(self.norm(x))
        return logits, None, {}

    def param_report(self) -> dict[str, int]:
        """Measured parameter counts; ``active`` == ``stored`` for this dense model."""
        parts = {
            "byte_emb": _count(self.tok_emb),
            "ctx_emb": _count(self.ctx_emb),
            "pos_emb": _count(self.pos_emb),
            "blocks": _count(self.blocks),
            "norm": _count(self.norm),
            "head": _count(self.head),
        }
        stored = _count(self)
        if sum(parts.values()) != stored:
            raise RuntimeError(
                "TransformerBaseline.param_report: component counts do not sum to the total"
            )
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {**parts, "stored": stored, "active": stored, "trainable": trainable}

    # helpers ---------------------------------------------------------------------------------

    @staticmethod
    def attention_mask(reset: Tensor) -> Tensor:
        """Bool ``[B, 1, T, T]``: query ``i`` may attend key ``j`` iff ``j <= i`` and no reset
        lies in ``(j, i]`` (same document). Integer cumsum: deterministic on every backend."""
        B, T = reset.shape
        segment = reset.to(torch.int64).cumsum(dim=1)  # document index of every position
        same = segment.unsqueeze(2) == segment.unsqueeze(1)  # [B, T, T]
        causal = torch.ones(T, T, dtype=torch.bool, device=reset.device).tril()
        return (same & causal).unsqueeze(1)


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
