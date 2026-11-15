"""The HSN-T reader alone with a next-byte head (``model_name: reader``): the exact-reader
diagnostic.

Asked for by the user's review amendments to the HSN-T white paper (14 Nov 2026, item 04a-3):
before a text or arithmetic result of HSN-T is read as a failure of the hybrid, the reader it
contains has to be measured by itself. The stock Transformer baseline of the same shape is not
that reader: it has a position table and a context input, the reader has rotary positions and
no context input. This model is the reader module itself
(:class:`hsn.model.reader.Reader`: the same embedding, blocks, span, normalisation, document
mask and positions, built from the same ``model.reader_*`` keys and the same seed, so its
initial weights are those of the reader inside an HSN-T of that seed) and one linear layer from
its output ``c_t`` to the next byte.

Stateless under the :class:`hsn.train.loop.SequenceModel` protocol, like the Transformer
baseline: every window is read on its own and the context symbol is not an input. What it can
reach is what the reader can reach: the document's bytes up to ``t`` inside the window, the
last ``1 + L (n − 1)`` of them with a span ``n`` and ``L`` blocks.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from hsn.config import RunConfig
from hsn.model.reader import Reader

__all__ = ["ReaderLM"]


class ReaderLM(nn.Module):
    """``logits_t = W c_t + b`` with ``c = Reader(tokens, reset)`` (module docstring)."""

    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        if cfg.model.reader == "none":
            raise ValueError("ReaderLM needs model.reader=transformer (the reader it stands for)")
        self.vocab_size = cfg.model.vocab_size
        self.reader = Reader(cfg.model, seed=cfg.seed)
        with torch.random.fork_rng(devices=[]):  # the head has its own generator, as the reader
            generator = torch.Generator(device="cpu").manual_seed(
                (cfg.seed + 0x4EAD1E) % (2**63 - 1)
            )
            torch.set_rng_state(generator.get_state())
            self.head = nn.Linear(self.reader.width, self.vocab_size)
            nn.init.normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)

    # SequenceModel protocol -----------------------------------------------------------------

    def init_state(self, B: int, device: torch.device | str) -> None:
        """Stateless: there is nothing to carry between windows."""
        if B < 1:
            raise ValueError(f"B must be >= 1, got {B}")
        return None

    def reset_state(self, state: None, mask: Tensor) -> None:
        if state is not None:
            raise ValueError(f"ReaderLM carries no state, got {type(state).__name__}")
        if mask.dtype is not torch.bool or mask.dim() != 1:
            raise ValueError(f"mask must be a bool [B] tensor, got {tuple(mask.shape)}")
        return None

    def forward(
        self, tokens: Tensor, context: Tensor, state: None, reset: Tensor
    ) -> tuple[Tensor, None, dict[str, Any]]:
        """``tokens`` / ``reset`` are ``[B, T]``; ``context`` is accepted and not read (the
        reader has no context input)."""
        if state is not None:
            raise ValueError(f"ReaderLM carries no state, got {type(state).__name__}")
        if context.shape != tokens.shape:
            raise ValueError(
                f"context must have shape {tuple(tokens.shape)}, got {tuple(context.shape)}"
            )
        logits: Tensor = self.head(self.reader(tokens, reset))
        return logits, None, {}

    def param_report(self) -> dict[str, int]:
        """Measured parameter counts; ``active`` == ``stored`` for this dense model."""
        reader = sum(p.numel() for p in self.reader.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "reader": reader,
            "head": head,
            "stored": reader + head,
            "active": reader + head,
            "trainable": trainable,
        }
