"""HSN-T (``model_name: hsn_t``, ``model.reader = transformer``, ``docs/hsn_t.md``): the
window reader.

A small causal Transformer over the byte ids of one window, all positions at once. Its output at
position ``t``, ``c_t`` (width ``model.reader_width``), takes the place of the fast cells' read of
amendment A8 in the two places the model reads it: ``u_t = S([z_t ; q_{t−1} ; ctx_t ; c_t])`` and
``D([q_t ; u_t ; r_t ; c_t])``. The cells and their carried state, the fixed addresses, the
router, the units, the convex tanh gate, the graph and the summary ``q`` are unchanged: the memory
is the HSN's, the reading of the current window is the reader's. This is the one module of the
model with sequence attention, which A5's readout text excludes; the key is off by default.

What ``c_t`` may depend on (tested in ``tests/test_hsn_t.py``):

* the bytes ``x_s`` with ``s ≤ t`` of the same window and the same document (``reset`` splits a
  row into documents, as it resets the cells; the attention mask is block-diagonal over them);
* with ``model.reader_span = n`` only the last ``n`` of them (``t − n < s ≤ t``): the
  short-sighted reader, which has to leave everything further back to the memory;
* nothing of an earlier window, no cell state, no context symbol.

Layout: byte embedding → ``model.reader_layers`` pre-norm blocks (LayerNorm → causal multi-head
self-attention → residual; LayerNorm → GELU MLP of width ``4 w`` → residual) → LayerNorm. The
position enters as a rotation of the attention's queries and keys by the position in the window
(rotary embedding), so only the distance ``t − s`` is read and there is no position parameter:
with a span the output is a function of the last ``n`` bytes alone, wherever they sit in the
window.

The rotation's cosines and sines are read from a table of the first ``_ROTARY_TABLE`` window
positions (``tests/test_reader_rotary_table.py``). It holds what the forward pass computed on
every call until 15 Nov 2026: the float64 angles, computed on the device of the parameters and
cast to their dtype. It is made when the reader is built and made again, not moved or cast,
whenever the module is (``.to()``, ``.cuda()``, ``.double()``), so the outputs are the same bit
for bit. Under ``torch.compile`` the computation in the forward pass was inlined into the
kernels that rotate the queries and keys, as float64 trigonometry for each of their elements
(78 ms of a 111 ms training step on an RTX 3060 at width 224). The table is a buffer outside
the state dict, so checkpoints are unchanged. The forward pass still computes the angles
itself for a window longer than the table, on MPS (which has no float64; its float32 angles
are kept as they were) and for parameters that were moved or cast without the module.

The parameters are drawn from the module's own generator (``seed``) inside a forked RNG, so
building a reader neither changes another parameter's initialisation nor consumes the global
random stream.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import ModelConfig

__all__ = ["Reader"]

_ROTARY_BASE = 10_000.0
#: Window positions in the rotary table: four windows of the default ``train.unroll`` (the
#: reader is built from ``ModelConfig``, which has no window length). 2 · 1024 · half values.
_ROTARY_TABLE = 1024
_INIT_STD = 0.02


def _rotary_angles(T: int, half: int, device: torch.device) -> Tensor:
    """The rotary angles ``[T, half]``: position ``t`` times ``10000^(−i / half)``. In float64
    where the device has it (float32 on MPS), so that a rotation by ``t`` against one by ``s``
    is the rotation by ``t − s`` to the working precision at every window position."""
    wide = torch.float32 if device.type == "mps" else torch.float64
    exponent = torch.arange(half, device=device, dtype=wide) / half
    return torch.arange(T, device=device, dtype=wide)[:, None] * (_ROTARY_BASE**-exponent)


def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the pairs ``(x[..., i], x[..., i + half])`` of ``x [B, H, T, 2·half]`` by the
    position's angles ``cos`` / ``sin [T, half]``."""
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


class _ReaderBlock(nn.Module):
    """One pre-norm block: causal multi-head self-attention with rotary positions and a GELU
    MLP, both residual."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm_attn = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.fc_in = nn.Linear(width, 4 * width)
        self.fc_out = nn.Linear(4 * width, width)

    def forward(self, x: Tensor, mask: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, w = x.shape
        q, k, v = self.qkv(self.norm_attn(x)).split(w, dim=-1)
        q = q.view(B, T, self.heads, -1).transpose(1, 2)  # [B, H, T, w / H]
        k = k.view(B, T, self.heads, -1).transpose(1, 2)
        v = v.view(B, T, self.heads, -1).transpose(1, 2)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        attn = F.scaled_dot_product_attention(
            _rotate(q, cos, sin), _rotate(k, cos, sin), v, attn_mask=mask
        )
        x = x + self.proj(attn.transpose(1, 2).reshape(B, T, w))
        out: Tensor = x + self.fc_out(F.gelu(self.fc_in(self.norm_mlp(x))))
        return out


class Reader(nn.Module):
    """``c = Reader(tokens, reset)``: ``[B, T] → [B, T, model.reader_width]`` (module docstring)."""

    #: The rotary table (module docstring): cos / sin ``[_ROTARY_TABLE, half]`` in the dtype
    #: and on the device of the parameters; None where the forward pass computes them (MPS).
    _rot_cos: Tensor | None
    _rot_sin: Tensor | None

    def __init__(self, cfg: ModelConfig, *, seed: int = 0) -> None:
        super().__init__()
        if cfg.reader != "transformer":
            raise ValueError(f"Reader needs model.reader=transformer, got {cfg.reader!r}")
        width, heads = cfg.reader_width, cfg.reader_heads
        self.width = width
        self.heads = heads
        self.span: int | None = cfg.reader_span
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed((seed + 0xA14EAD) % (2**63 - 1))
            torch.set_rng_state(generator.get_state())
            self.emb = nn.Embedding(cfg.vocab_size, width)
            self.blocks = nn.ModuleList(
                _ReaderBlock(width, heads) for _ in range(cfg.reader_layers)
            )
            self.norm = nn.LayerNorm(width)
            for module in self.modules():
                if isinstance(module, nn.Linear | nn.Embedding):
                    nn.init.normal_(module.weight, std=_INIT_STD)
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
        self.register_buffer("_rot_cos", None, persistent=False)
        self.register_buffer("_rot_sin", None, persistent=False)
        self._fill_rotary_table()

    def _fill_rotary_table(self) -> None:
        """Compute the rotary table where the parameters are: the cos / sin the forward pass
        would compute there for a window of ``_ROTARY_TABLE`` positions."""
        weight = self.emb.weight
        if weight.device.type == "mps":
            self._rot_cos = self._rot_sin = None
            return
        angles = _rotary_angles(_ROTARY_TABLE, self.width // self.heads // 2, weight.device)
        self._rot_cos = angles.cos().to(weight.dtype)
        self._rot_sin = angles.sin().to(weight.dtype)

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        """``.to()`` / ``.cuda()`` / ``.double()``: the table is computed again for the new
        device and dtype. Moved, a CPU table could differ from the card's own in a rounding;
        cast, a float32 table would not be the float64 angles."""
        self._rot_cos = self._rot_sin = None
        super()._apply(fn, recurse)  # type: ignore[no-untyped-call]  # torch has no types here
        self._fill_rotary_table()
        return self

    def _rotation(self, T: int, x: Tensor) -> tuple[Tensor, Tensor]:
        """``cos`` / ``sin [T, half]`` of the rotary angles in the dtype of ``x``: read from
        the table when it holds them for ``x``, computed otherwise (module docstring)."""
        cos, sin = self._rot_cos, self._rot_sin
        if (
            cos is not None
            and sin is not None
            and T <= cos.shape[0]
            and cos.dtype == x.dtype
            and cos.device == x.device
        ):
            return cos[:T], sin[:T]
        angles = _rotary_angles(T, self.width // self.heads // 2, x.device)
        return angles.cos().to(x.dtype), angles.sin().to(x.dtype)

    def attention_mask(self, reset: Tensor) -> Tensor:
        """Bool ``[B, 1, T, T]``, True where position ``t`` (row) may read position ``s``
        (column): ``s ≤ t``, the same document of the row (``reset [B, T]`` starts a document at
        its True columns), and with a span ``t − s < span``. Every row holds its own position."""
        T = int(reset.shape[1])
        segment = reset.to(torch.int64).cumsum(dim=1)  # the document index of every position
        same = segment[:, :, None] == segment[:, None, :]
        index = torch.arange(T, device=reset.device)
        distance = index[:, None] - index[None, :]  # t − s
        near = distance >= 0
        if self.span is not None:
            near = near & (distance < self.span)
        return (same & near)[:, None]

    def forward(self, tokens: Tensor, reset: Tensor, *, blank: Tensor | None = None) -> Tensor:
        """``blank`` (bool ``[B, T]``; the byte-content options of ``docs/key_content.md``):
        the positions at which the reader reads a zero row in place of the byte's row of its
        embedding table. The attention mask and the rotary positions are those of the bytes as
        they are. ``None`` (default): every row is read, the lookup unchanged."""
        if tokens.ndim != 2 or reset.shape != tokens.shape or reset.dtype is not torch.bool:
            raise ValueError(
                f"Reader needs tokens [B, T] and a bool reset of the same shape, got "
                f"{tuple(tokens.shape)} and {reset.dtype} {tuple(reset.shape)}"
            )
        x = self.emb(tokens)
        if blank is not None:
            if blank.shape != tokens.shape or blank.dtype is not torch.bool:
                raise ValueError(
                    f"blank must be a bool tensor of the tokens' shape {tuple(tokens.shape)}, "
                    f"got {blank.dtype} {tuple(blank.shape)}"
                )
            x = x.masked_fill(blank[..., None], 0.0)
        cos, sin = self._rotation(int(tokens.shape[1]), x)
        mask = self.attention_mask(reset)
        for block in self.blocks:
            x = block(x, mask, cos, sin)
        out: Tensor = self.norm(x)
        return out

    def extra_repr(self) -> str:
        return f"width={self.width}, heads={self.heads}, span={self.span}"
