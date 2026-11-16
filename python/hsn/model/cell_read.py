"""The attention read of the cells (``model.cell_attention_heads``; HSN-T only, opt-in candidate;
``docs/amendments/cell_attention_read.md``).

The decoder of an HSN-T reads the memory through the summary ``q_t`` and through ``r_t``, the
mean of the cells the router selected. This module adds a third read beside them: at every
position the reader's output ``c_t`` queries **all** cells by content and by address, and the
result ``m_t`` is one more input of the decoder, ``D([q_t ; u_t ; r_t ; c_t ; m_t])``.

With ``H`` heads of width ``e = d / H``, the cell states ``h_{t,i}`` after position ``t`` (the
state ``r_t`` reads) and the fixed address code ``a_i`` of cell ``i`` (A5's
``Scorer.cell_addresses``, a buffer)::

    qry_t        = W_Q c_t + b_Q                          [B, d]     → H heads of width e
    key_{t,i}    = W_K [h_{t,i} ; a_i] + b_K              [B, N, d]  → H heads of width e
    val_{t,i}    = W_V h_{t,i} + b_V                      [B, N, d]  → H heads of width e
    α_{t,i}^(j)  = softmax_i( qry_t^(j) · key_{t,i}^(j) / sqrt(e) )   over every cell, no top-k
    m_t          = W_O [ Σ_i α_{t,i}^(1) val_{t,i}^(1) ; … ; Σ_i α_{t,i}^(H) val_{t,i}^(H) ] + b_O

Nothing is written: the read holds no state, changes no cell and does not enter the router,
the units or the gates. It is a read of the memory, not sequence attention: the keys are the
``N`` cells of one position, never other positions of the window.

Parameters (``d (w + 4d + 4)`` with ``w`` the reader's width): ``query`` (``w → d``), ``state``
(``d → 2d``: the state half of ``W_K`` and ``W_V`` stacked, so one matrix product over the
cells serves the keys and the values), ``address`` (``d → d``, no bias: the address half of
``W_K``) and ``out`` (``d → d``). The weights are drawn as ``model.state_init`` asks (the
cell's own maps) from a generator of their own, so no other parameter's initialisation
changes; the biases start at zero.

The forward pass has fixed shapes and no data-dependent control flow, so it traces into the
compiled block of :mod:`hsn.train.compile` as it stands.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hsn.config import ModelConfig
from hsn.model.embeddings import init_weight_

__all__ = ["CellAttentionRead"]

_SEED_TAG = 0xCA77EAD


class CellAttentionRead(nn.Module):
    """``m = read(c, h, addresses)``: ``c [B, w]`` (the reader's output of the position),
    ``h [B, N, d]`` (the cell states), ``addresses [N, d]`` (the cells' fixed codes) →
    ``m [B, d]`` (module docstring)."""

    def __init__(self, cfg: ModelConfig, *, seed: int = 0) -> None:
        super().__init__()
        heads = cfg.cell_attention_heads
        if heads < 1:
            raise ValueError(
                f"CellAttentionRead needs model.cell_attention_heads >= 1, got {heads}"
            )
        if cfg.reader == "none" or not cfg.key_address:
            raise ValueError(
                "the attention read of the cells needs model.reader (its queries) and "
                "model.key_address (the cells' address codes)"
            )
        if cfg.d % heads:
            raise ValueError(f"model.d={cfg.d} must be a multiple of the {heads} heads")
        d = cfg.d
        self.heads = heads
        self.head_dim = d // heads
        self.d = d
        self.scale = float(self.head_dim) ** -0.5
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed((seed + _SEED_TAG) % (2**63 - 1))
            torch.set_rng_state(generator.get_state())
            self.query = nn.Linear(cfg.reader_width, d)
            self.state = nn.Linear(d, 2 * d)  # rows [:d]: W_K's state half; rows [d:]: W_V
            self.address = nn.Linear(d, d, bias=False)  # W_K's address half
            self.out = nn.Linear(d, d)
            for layer in (self.query, self.state, self.address, self.out):
                init_weight_(layer.weight, cfg.state_init)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, c: Tensor, h: Tensor, addresses: Tensor) -> Tensor:
        return self.read(c, h, addresses)[0]

    def read(self, c: Tensor, h: Tensor, addresses: Tensor) -> tuple[Tensor, Tensor]:
        """``(m [B, d], α [B, H, N])``: the read and the attention weights of every head over
        the cells (each row sums to 1)."""
        both = self.state(h)  # [B, N, 2d]: one matrix product for the keys and the values
        return self._attend(c, both, addresses)

    def read_latest(self, c: Tensor, h: Tensor, index: Tensor, addresses: Tensor) -> Tensor:
        """The read of a window whose cells change only at events (``model.write_events``):
        ``c [B, T, w]`` (every byte's query), ``h [B, S, N, d]`` (the cell states of ``S``
        stages: the window's start and after each event slot), ``index [B, T]`` (the stage each
        byte reads) → ``m [B, T, d]``. The same read as :meth:`read` at every byte; the map of
        the cell states runs once per stage and its result is gathered per byte."""
        B, T = (int(n) for n in index.shape)
        N = int(h.shape[2])
        both = self.state(h)  # [B, S, N, 2d]
        both = both.gather(1, index.view(B, T, 1, 1).expand(-1, -1, N, 2 * self.d))
        return self._attend(c, both, addresses)[0]

    def _attend(self, c: Tensor, both: Tensor, addresses: Tensor) -> tuple[Tensor, Tensor]:
        """The attention of :meth:`read` from the queries' input ``c [..., w]`` and the mapped
        cell states ``both [..., N, 2d]`` (keys' state half, then values)."""
        *lead, N, _ = both.shape
        d, H, e = self.d, self.heads, self.head_dim
        query = self.query(c).view(*lead, H, e, 1)
        keys = both[..., :d] + self.address(addresses)  # the address half, [N, d] for all rows
        keys = keys.view(*lead, N, H, e).transpose(-3, -2)  # [..., H, N, e]
        values = both[..., d:].view(*lead, N, H, e).transpose(-3, -2)  # [..., H, N, e]
        scores = (keys @ query.to(keys.dtype)).squeeze(-1) * self.scale  # [..., H, N]
        alpha = torch.softmax(scores, dim=-1)
        mixed = (alpha.unsqueeze(-2).to(values.dtype) @ values).squeeze(-2)  # [..., H, e]
        m: Tensor = self.out(mixed.reshape(*lead, d))
        return m, alpha

    def extra_repr(self) -> str:
        return f"heads={self.heads}, head_dim={self.head_dim}"
