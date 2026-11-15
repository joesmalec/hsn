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

Four opt-in keys change how the position enters (``docs/hsn_t.md``, "The reader's positions";
15 Nov 2026). At their defaults the reader is the one above: the same parameters, state dict and
outputs, bit for bit (``tests/test_reader_positions.py``).

* ``model.reader_rotary_pairs = n``: only the first ``n`` pairs of each head (the ``n`` highest
  frequencies of the same ladder) are rotated. The other pairs get the angle zero, so their part
  of a query is compared with a key as it is, at every distance. Still distances only. A list
  gives ``n`` for each block: the Transformer baseline adds its positions once, before the first
  block, and the reader rotates in every block; ``[all, …, 0]`` leaves the last block's attention
  without a rotation: it compares queries and keys as they are (what it reads of the order is
  what the blocks before it wrote into their outputs).
* ``model.reader_rotary_dtype = parameters``: the rotation is computed in the dtype of the
  parameters (float32 master weights) and the rotated queries and keys are cast back to the
  working dtype for the attention. Under bf16 autocast the default rotates in bf16: the cos /
  sin and every product are rounded to 8 bits.
* ``model.reader_rotary_base = b`` (default 10000): the base of the frequency ladder, so pair
  ``i`` of a head turns by ``t · b^(−i / half)`` at distance ``t``, in the table and in the
  angles the forward pass computes itself alike. A larger base gives the slower pairs longer
  wavelengths; pair 0 keeps one radian a position. Still distances only, no parameter; it acts
  on the pairs ``model.reader_rotary_pairs`` rotates, and ``model.reader_position = learned``
  refuses it.
* ``model.reader_position = learned``: no rotation; a table ``pos`` of
  ``model.reader_positions`` rows is added to the byte embedding by window position, as the
  Transformer baseline's. The reader then reads the position in the window: its output is no
  longer a function of distances alone, and with a span not of the last ``n`` bytes alone. The
  table is drawn after the other parameters, which therefore start as the rotary reader's.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import ModelConfig

__all__ = ["Reader"]

#: The default of ``model.reader_rotary_base``: the base of the rotary frequency ladder.
_ROTARY_BASE = 10_000.0
#: Window positions in the rotary table: four windows of the default ``train.unroll`` (the
#: reader is built from ``ModelConfig``, which has no window length). 2 · 1024 · half values.
_ROTARY_TABLE = 1024
_INIT_STD = 0.02


def _rotary_angles(
    T: int,
    half: int,
    device: torch.device,
    pairs: int | None = None,
    base: float = _ROTARY_BASE,
) -> Tensor:
    """The rotary angles ``[T, half]``: position ``t`` times ``base^(−i / half)``
    (``model.reader_rotary_base``, 10000 by default). In float64 where the device has it
    (float32 on MPS), so that a rotation by ``t`` against one by ``s`` is the rotation by
    ``t − s`` to the working precision at every window position. With ``pairs``
    (``model.reader_rotary_pairs``) the pairs ``i ≥ pairs`` have the angle zero: their cos is 1
    and their sin 0, and the rotation returns them as they are, bit for bit."""
    wide = torch.float32 if device.type == "mps" else torch.float64
    exponent = torch.arange(half, device=device, dtype=wide) / half
    angles = torch.arange(T, device=device, dtype=wide)[:, None] * (base**-exponent)
    if pairs is not None:
        angles[:, pairs:] = 0.0
    return angles


def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the pairs ``(x[..., i], x[..., i + half])`` of ``x [B, H, T, 2·half]`` by the
    position's angles ``cos`` / ``sin [T, half]``."""
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


class _ReaderBlock(nn.Module):
    """One pre-norm block: causal multi-head self-attention with rotary positions and a GELU
    MLP, both residual."""

    def __init__(self, width: int, heads: int, *, rotate_in: str = "working") -> None:
        super().__init__()
        self.heads = heads
        #: ``model.reader_rotary_dtype``: "working" rotates in the dtype of the queries and keys,
        #: "parameters" in the dtype of the cos / sin it is given (the parameters')
        self.rotate_in = rotate_in
        self.norm_attn = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.fc_in = nn.Linear(width, 4 * width)
        self.fc_out = nn.Linear(4 * width, width)

    def forward(self, x: Tensor, mask: Tensor, cos: Tensor | None, sin: Tensor | None) -> Tensor:
        """``cos`` / ``sin [T, half]``: the rotary angles' in the dtype of the parameters, or
        ``None`` for no rotation (``model.reader_position = learned``)."""
        B, T, w = x.shape
        q, k, v = self.qkv(self.norm_attn(x)).split(w, dim=-1)
        q = q.view(B, T, self.heads, -1).transpose(1, 2)  # [B, H, T, w / H]
        k = k.view(B, T, self.heads, -1).transpose(1, 2)
        v = v.view(B, T, self.heads, -1).transpose(1, 2)
        if cos is not None and sin is not None:
            if self.rotate_in == "parameters":  # in the cos / sin's dtype, then cast back
                working = q.dtype
                q = _rotate(q.to(cos.dtype), cos, sin).to(working)
                k = _rotate(k.to(cos.dtype), cos, sin).to(working)
            else:
                cos, sin = cos.to(q.dtype), sin.to(q.dtype)
                q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(attn.transpose(1, 2).reshape(B, T, w))
        out: Tensor = x + self.fc_out(F.gelu(self.fc_in(self.norm_mlp(x))))
        return out


class Reader(nn.Module):
    """``c = Reader(tokens, reset)``: ``[B, T] → [B, T, model.reader_width]`` (module docstring)."""

    #: The rotary table (module docstring): cos / sin ``[_ROTARY_TABLE, half]`` in the dtype
    #: and on the device of the parameters; None where the forward pass computes them (MPS)
    #: and without a rotation (``model.reader_position = learned``).
    _rot_cos: Tensor | None
    _rot_sin: Tensor | None
    #: ``model.reader_position = learned``: the position table ``[model.reader_positions, w]``;
    #: None with rotary positions (no entry in the state dict).
    pos: nn.Parameter | None

    def __init__(self, cfg: ModelConfig, *, seed: int = 0) -> None:
        super().__init__()
        if cfg.reader != "transformer":
            raise ValueError(f"Reader needs model.reader=transformer, got {cfg.reader!r}")
        width, heads = cfg.reader_width, cfg.reader_heads
        self.width = width
        self.heads = heads
        self.span: int | None = cfg.reader_span
        self.position = cfg.reader_position
        #: ``model.reader_rotary_base``: the base of the frequency ladder of the table and of
        #: the angles the forward pass computes itself
        self.rotary_base: float = float(cfg.reader_rotary_base)
        #: ``model.reader_rotary_pairs`` as configured: one count, one for each block, or None
        self.rotary_pairs: int | list[int] | None = cfg.reader_rotary_pairs
        half = width // heads // 2
        counts = (
            list(self.rotary_pairs)
            if isinstance(self.rotary_pairs, list)
            else [self.rotary_pairs] * cfg.reader_layers
        )
        #: The rotated pairs of each block's heads; None: all of them (the default)
        self._block_pairs: list[int | None] = [
            None if n is None or n >= half else n for n in counts
        ]
        #: The pairs the rotary table itself keeps: the blocks' count when they share one (the
        #: table is then handed to every block as it is), None when it holds every pair
        self._table_pairs: int | None = (
            self._block_pairs[0] if len(set(self._block_pairs)) == 1 else None
        )
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed((seed + 0xA14EAD) % (2**63 - 1))
            torch.set_rng_state(generator.get_state())
            self.emb = nn.Embedding(cfg.vocab_size, width)
            self.blocks = nn.ModuleList(
                _ReaderBlock(width, heads, rotate_in=cfg.reader_rotary_dtype)
                for _ in range(cfg.reader_layers)
            )
            self.norm = nn.LayerNorm(width)
            for module in self.modules():
                if isinstance(module, nn.Linear | nn.Embedding):
                    nn.init.normal_(module.weight, std=_INIT_STD)
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
            if cfg.reader_position == "learned":  # drawn last: the other draws are unchanged
                self.pos = nn.Parameter(torch.empty(cfg.reader_positions, width))
                nn.init.normal_(self.pos, std=_INIT_STD)
            else:
                self.register_parameter("pos", None)
        self.register_buffer("_rot_cos", None, persistent=False)
        self.register_buffer("_rot_sin", None, persistent=False)
        self._fill_rotary_table()

    def _fill_rotary_table(self) -> None:
        """Compute the rotary table where the parameters are: the cos / sin the forward pass
        would compute there for a window of ``_ROTARY_TABLE`` positions."""
        weight = self.emb.weight
        if weight.device.type == "mps" or self.position != "rotary":
            self._rot_cos = self._rot_sin = None
            return
        angles = _rotary_angles(
            _ROTARY_TABLE,
            self.width // self.heads // 2,
            weight.device,
            self._table_pairs,
            self.rotary_base,
        )
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
        half = self.width // self.heads // 2
        angles = _rotary_angles(T, half, x.device, self._table_pairs, self.rotary_base)
        return angles.cos().to(x.dtype), angles.sin().to(x.dtype)

    def _block_rotation(
        self, pairs: int | None, cos: Tensor | None, sin: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None]:
        """The cos / sin one block is handed: the table's when it already holds the block's
        pairs (the default, and one count for all blocks), none for a block that rotates
        nothing, and otherwise the table's with the angle zero in the pairs ``i ≥ pairs``."""
        if cos is None or sin is None or pairs == 0:
            return None, None
        if pairs is None or pairs == self._table_pairs:
            return cos, sin
        keep = torch.arange(cos.shape[-1], device=cos.device) < pairs
        return torch.where(keep, cos, 1.0), torch.where(keep, sin, 0.0)

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

    def forward(self, tokens: Tensor, reset: Tensor) -> Tensor:
        if tokens.ndim != 2 or reset.shape != tokens.shape or reset.dtype is not torch.bool:
            raise ValueError(
                f"Reader needs tokens [B, T] and a bool reset of the same shape, got "
                f"{tuple(tokens.shape)} and {reset.dtype} {tuple(reset.shape)}"
            )
        x = self.emb(tokens)
        T = int(tokens.shape[1])
        cos: Tensor | None
        sin: Tensor | None
        if self.pos is not None:  # model.reader_position = learned: the table, no rotation
            if T > self.pos.shape[0]:
                raise ValueError(
                    f"a window of {T} positions exceeds the reader's position table "
                    f"(model.reader_positions={self.pos.shape[0]})"
                )
            x = x + self.pos[:T]
            cos = sin = None
        else:
            cos, sin = self._rotation(T, x)
        mask = self.attention_mask(reset)
        for i, block in enumerate(self.blocks):
            block_cos, block_sin = self._block_rotation(self._block_pairs[i], cos, sin)
            x = block(x, mask, block_cos, block_sin)
        out: Tensor = self.norm(x)
        return out

    def extra_repr(self) -> str:
        text = f"width={self.width}, heads={self.heads}, span={self.span}"
        if self.position != "rotary":
            text += f", position={self.position}"
        if self.rotary_pairs is not None:
            text += f", rotary_pairs={self.rotary_pairs}"
        if self.rotary_base != _ROTARY_BASE:
            text += f", rotary_base={self.rotary_base:g}"
        return text
