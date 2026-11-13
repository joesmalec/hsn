"""The HSN's fixed byte codes as an input of the baselines (``baseline.address_input``).

Amendment A5 gives the HSN router two fixed features per cell: the similarity of the cell's
address to a frozen random code of the current byte and of the previous byte
(:meth:`hsn.model.router.Router.address_features`). They are what lets the HSN route a key it
never trained on: the code of an unseen byte is as good as the code of a seen one, while a learned
embedding row of an unseen byte never receives a gradient.

``AddressInput`` gives a baseline the same information at its input: the frozen code of the
current byte and of the previous byte, concatenated and mapped to the model width by one learned
linear layer whose output is added to the embeddings. The code table is drawn exactly as the
router draws its own (same generator seed and order), so with ``baseline.address_dim`` equal to
the HSN's ``model.d`` and the same run seed the two models read the same table. It is a control,
off by default: a baseline that binds unseen keys with it shows how much of the HSN's score on the
held-out-key tasks is the fixed code rather than the cell graph.

The previous byte of a position is the byte before it in the window, BOS where the position
starts a document (the HSN's rule). A window's first position that continues a document has no
previous byte inside the window; its previous code is zero there. That case does not occur with
aligned in-range documents, where every window starts a document.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import RunConfig
from hsn.data.synthetic import BOS
from hsn.model.router import _KEY_ADDRESS_TAG

__all__ = ["AddressInput"]


class AddressInput(nn.Module):
    """Frozen byte codes of the current and previous byte → one learned projection."""

    byte_codes: Tensor  # [V, address_dim], a persistent buffer, never trained

    def __init__(self, cfg: RunConfig, d_model: int) -> None:
        super().__init__()
        seed = cfg.seed
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("baseline.address_input requires seed as a non-negative int")
        dim = cfg.baseline.address_dim
        generator = torch.Generator(device="cpu").manual_seed((seed ^ _KEY_ADDRESS_TAG) % (2**63))
        self.dim = dim
        self.register_buffer(
            "byte_codes",
            torch.randn(cfg.model.vocab_size, dim, generator=generator) / math.sqrt(dim),
        )
        self.proj = nn.Linear(2 * dim, d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, tokens: Tensor, reset: Tensor) -> Tensor:
        """``[B, T, d_model]`` to add to the embeddings of ``tokens`` (``reset`` as the model's)."""
        previous = torch.roll(tokens, shifts=1, dims=1).masked_fill(reset, BOS)
        known = reset.clone()
        known[:, 1:] = True  # only a window's first position can lack its previous byte
        current_code = F.embedding(tokens, self.byte_codes)
        previous_code = F.embedding(previous, self.byte_codes) * known.unsqueeze(-1)
        codes = torch.cat([current_code, previous_code], dim=-1)
        out: Tensor = self.proj(codes.to(self.proj.weight.dtype))
        return out
