"""Small causal Transformer byte-level language model (P0.5): the attention control.

``TransformerBaseline`` is a pre-norm Transformer (GPT-2 layout: LayerNorm → multi-head causal
self-attention → residual, LayerNorm → GELU MLP → residual) over the byte embedding plus the
context-symbol embedding plus a learned position embedding. Its context window is the TBPTT
window ``train.unroll``: every window is processed on its own, with positions ``0..T-1`` relative
to the window start, and there is no key/value cache, so **it never attends across window
boundaries** (chunk pitfall; the opt-in ``baseline.transformer.carry`` below is the exception).
It is stateless under the :class:`hsn.train.loop.SequenceModel` protocol: ``init_state`` and
``reset_state`` return ``None``. Inside a window the ``reset`` flags
split each row into documents and the attention mask is block-diagonal over those segments (and
causal), so a document never attends to the one packed before it; that is the transformer's
counterpart of the GRU's state reset.

Sizes come from ``baseline.transformer.*`` (``d_model``, ``n_layers``, ``n_heads``, ``ffn_hidden``,
``dropout``); the vocabulary and context symbol count from ``model.vocab_size`` and
``model.n_context``; the window from ``train.unroll``. With the defaults (d_model 192, 4 layers,
4 heads, MLP 768, V=256, 16 context symbols, unroll 256) the model has 1,930,624 parameters as
measured by :meth:`TransformerBaseline.param_report`: the documented ≈2M-parameter baseline.
Every parameter is read at every position, so *active* equals *stored* for this model.

**Transformer-XL** (opt-in, 17 Nov 2026; ``docs/transformer_xl_baseline.md``). Two keys of
``baseline.transformer`` give the model segment-level recurrence (Dai et al. 2019); at their
defaults the model above is unchanged, bit for bit.

* ``position: rotary``: no position table. Every block rotates its queries and keys by their
  position (rotary embedding, base 10000, every pair of each head), so a score depends on the
  distance ``t − s`` alone; the role of Transformer-XL's relative position encoding. The
  rotation's cosines and sines are a table made when the model is built (and again when it is
  moved or cast), for the positions ``−carry … train.unroll − 1``; a buffer outside the state
  dict.
* ``carry: M`` (needs ``rotary``): Transformer-XL's memory of length ``M``. Every block keeps the
  keys and values of the last ``M`` positions of the stream before the window, detached (no
  gradient crosses a window), in the state ``(keys, values, valid)``: ``keys`` / ``values [L,
  B, H, M, d_model / H]`` (block, row, head, slot, channel; slot ``j`` is the position ``j − M``
  before the window, the keys un-rotated) and ``valid [B, M]`` (bool: the slot holds a position
  of the document going on). Window position ``t`` attends every valid cached slot and the
  window positions ``s ≤ t`` of its own document: the causal mask over ``[cache; window]``. The
  cached keys are rotated by their positions ``−M … −1`` and the window's queries and keys by
  ``0 … T − 1``, so every score sees the distance in the stream. ``reset`` acts as before: a
  position reads nothing of a document before its own, cached or not, and the slots of a
  document that ended are invalid. The trainer carries the state from window to window in
  training and evaluation, detaches it at every window and writes it into checkpoints, as the
  GRU's. The state has fixed shapes, so a window compiles as one graph
  (``hardware.baseline_compile``). ``carry: 0`` with ``rotary`` is the same model reading every
  window alone (stateless): the cacheless twin.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.baselines.address import AddressInput
from hsn.config import RunConfig
from hsn.model.reader import _attention_bias, _keep_tail, _rotate

__all__ = ["Cache", "TransformerBaseline"]

#: ``baseline.transformer.carry``: the state ``(keys, values, valid)`` (module docstring). A
#: plain tuple: the trainer walks it as it walks a state.
Cache = tuple[Tensor, ...]
#: The base of the rotary frequency ladder (``baseline.transformer.position = rotary``).
_ROTARY_BASE = 10_000.0


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

    def forward(
        self, x: Tensor, mask: Tensor, cos: Tensor | None = None, sin: Tensor | None = None
    ) -> Tensor:
        """``cos`` / ``sin [T, d_head / 2]``: the rotary angles of the window's positions
        (``position = rotary``); ``None``: no rotation (the learned position table)."""
        q, k, v = self._heads(x)
        if cos is not None and sin is not None:  # rotary: in the working dtype
            cos, sin = cos.to(q.dtype), sin.to(q.dtype)
            q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self._residual(x, attn)

    def forward_carry(
        self, x: Tensor, mask: Tensor, cos: Tensor, sin: Tensor, keys: Tensor, values: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """:meth:`forward` over the cached keys and values too (``carry``). ``keys`` /
        ``values [B, H, M, d_head]``: the un-rotated keys and the values of the ``M`` positions
        before the window; ``cos`` / ``sin [M + T, d_head / 2]``: the angles of the positions
        ``−M … T − 1``; ``mask [B, 1, T, M + T]``, the cached columns first. Returns the output
        and the window's un-rotated keys and its values ``[B, H, T, d_head]`` (the next
        cache's)."""
        q, k, v = self._heads(x)
        M = keys.shape[2]
        all_k = torch.cat([keys.to(k.dtype), k], dim=2)  # [B, H, M + T, d_head]
        all_v = torch.cat([values.to(v.dtype), v], dim=2)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q, all_k = _rotate(q, cos[M:], sin[M:]), _rotate(all_k, cos, sin)
        attn = F.scaled_dot_product_attention(
            q, all_k, all_v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self._residual(x, attn), k, v

    def _heads(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Queries, keys and values ``[B, H, T, d_head]``."""
        B, T, d = x.shape
        q, k, v = self.qkv(self.norm_attn(x)).split(d, dim=-1)
        q = q.view(B, T, self.n_heads, -1).transpose(1, 2)  # [B, H, T, d_head]
        k = k.view(B, T, self.n_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.n_heads, -1).transpose(1, 2)
        return q, k, v

    def _residual(self, x: Tensor, attn: Tensor) -> Tensor:
        """The attention's projection and the MLP, both residual."""
        B, T, d = x.shape
        x = x + self.drop(self.proj(attn.transpose(1, 2).reshape(B, T, d)))
        x = x + self.drop(self.fc_out(F.gelu(self.fc_in(self.norm_mlp(x)))))
        return x


class TransformerBaseline(nn.Module):
    """Byte + context + position embeddings → ``n_layers`` pre-norm blocks → LayerNorm → head.
    With ``baseline.transformer.position = rotary`` the position enters the blocks as a
    rotation instead, and with ``baseline.transformer.carry`` the blocks attend a cache of the
    windows before (module docstring)."""

    #: The learned position table; None with rotary positions (no entry in the state dict).
    pos_emb: nn.Embedding | None
    #: Rotary positions: cos / sin ``[carry + train.unroll, d_head / 2]`` of the positions
    #: ``−carry … train.unroll − 1``, in the dtype and on the device of the parameters (buffers
    #: outside the state dict); not registered with the learned table.
    _rot_cos: Tensor | None
    _rot_sin: Tensor | None

    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        spec = cfg.baseline.transformer
        self.vocab_size = cfg.model.vocab_size
        self.n_context = cfg.model.n_context
        self.d_model = spec.d_model
        self.n_layers = spec.n_layers
        self.n_heads = spec.n_heads
        self.window = cfg.train.unroll  # the context window (and with carry, its cache's)
        #: ``baseline.transformer.position``: learned (the table) | rotary
        self.position = spec.position
        #: ``baseline.transformer.carry``: Transformer-XL's memory length (0: none)
        self.carry: int = spec.carry
        if self.carry and self.position != "rotary":
            raise ValueError(
                "baseline.transformer.carry needs baseline.transformer.position=rotary"
            )
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.ctx_emb = nn.Embedding(self.n_context, self.d_model)
        if self.position == "learned":
            self.pos_emb = nn.Embedding(self.window, self.d_model)
        else:
            self.pos_emb = None
        # baseline.address_input: the HSN's fixed byte codes as an input (a control, off by default)
        self.address = AddressInput(cfg, self.d_model) if cfg.baseline.address_input else None
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
        if self.position == "rotary":
            self.register_buffer("_rot_cos", None, persistent=False)
            self.register_buffer("_rot_sin", None, persistent=False)
            self._fill_rotary_table()

    # SequenceModel protocol -----------------------------------------------------------------

    def init_state(self, B: int, device: torch.device | str) -> Cache | None:
        """Stateless (None): nothing to carry between windows. With ``carry`` the empty cache
        ``(keys, values, valid)`` of ``B`` fresh streams (module docstring), keys and values in
        the dtype of the parameters whatever the working dtype (a bf16 value is exact in
        float32), so the state has the same dtype and shapes at every window."""
        if B < 1:
            raise ValueError(f"B must be >= 1, got {B}")
        if not self.carry:
            return None
        shape = self._cache_shape(B)
        keys = torch.zeros(shape, device=device, dtype=self.tok_emb.weight.dtype)
        valid = torch.zeros(B, self.carry, device=device, dtype=torch.bool)
        return keys, torch.zeros_like(keys), valid

    def reset_state(self, state: Cache | None, mask: Tensor) -> Cache | None:
        """With ``carry``: ``state`` with nothing valid in the rows where ``mask`` (``[B]``
        bool) is True."""
        if not self.carry:
            if state is not None:
                raise ValueError(
                    f"TransformerBaseline carries no state, got {type(state).__name__}"
                )
            if mask.dtype is not torch.bool or mask.dim() != 1:
                raise ValueError(f"mask must be a bool [B] tensor, got {tuple(mask.shape)}")
            return None
        if mask.dtype is not torch.bool or mask.dim() != 1:
            raise ValueError(f"mask must be a bool [B] tensor, got {tuple(mask.shape)}")
        keys, values, valid = self.check_cache(state, int(mask.shape[0]))
        return keys, values, valid & ~mask.to(valid.device)[:, None]

    def forward(
        self, tokens: Tensor, context: Tensor, state: Cache | None, reset: Tensor
    ) -> tuple[Tensor, Cache | None, dict[str, Any]]:
        """``tokens``/``context``/``reset`` are ``[B, T]`` with ``T <= train.unroll``. With
        ``carry`` ``state`` is the cache of the windows before and the next one is returned in
        its place; otherwise both are None."""
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
        cache = self.check_cache(state, B) if self.carry else None
        if not self.carry and state is not None:
            raise ValueError(f"TransformerBaseline carries no state, got {type(state).__name__}")
        x = self.tok_emb(tokens) + self.ctx_emb(context)
        if self.pos_emb is not None:  # the learned table (the default)
            positions = torch.arange(T, device=tokens.device)
            x = x + self.pos_emb(positions)
        if self.address is not None:
            x = x + self.address(tokens, reset)
        x = self.drop(x)
        if self.pos_emb is not None:
            mask = self.attention_mask(reset)
            for block in self.blocks:
                x = block(x, mask)
            return self.head(self.norm(x)), None, {}
        assert self._rot_cos is not None and self._rot_sin is not None
        if cache is None:  # rotary, every window alone (the table starts at position 0)
            cos, sin = self._rot_cos[:T], self._rot_sin[:T]
            mask = _attention_bias(self.attention_mask(reset), x)
            for block in self.blocks:
                x = block(x, mask, cos, sin)
            return self.head(self.norm(x)), None, {}
        x, cache = self._read_with_cache(x, reset, cache)
        return self.head(self.norm(x)), cache, {}

    def param_report(self) -> dict[str, int]:
        """Measured parameter counts; ``active`` == ``stored`` for this dense model."""
        parts = {
            "byte_emb": _count(self.tok_emb),
            "ctx_emb": _count(self.ctx_emb),
            **({"pos_emb": _count(self.pos_emb)} if self.pos_emb is not None else {}),
            **({"address": _count(self.address)} if self.address is not None else {}),
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

    # baseline.transformer.carry ---------------------------------------------------------------

    def check_cache(self, cache: object, B: int) -> Cache:
        """``cache`` as :meth:`init_state` makes it for ``B`` streams, or a ``ValueError``."""
        if not self.carry:
            raise ValueError(
                "a TransformerBaseline without baseline.transformer.carry has no cache"
            )
        shape = self._cache_shape(B)
        layout = [(shape, "float"), (shape, "float"), ((B, self.carry), "bool")]

        def kind(t: Tensor) -> str:
            return "bool" if t.dtype is torch.bool else "float" if t.is_floating_point() else "int"

        got: object = type(cache).__name__
        if isinstance(cache, tuple) and all(isinstance(t, Tensor) for t in cache):
            got = [(tuple(t.shape), kind(t)) for t in cache]
            if got == layout:
                return cache
        raise ValueError(
            f"baseline.transformer.carry={self.carry}: the state must be {layout} (init_state), "
            f"got {got}"
        )

    def carry_mask(self, reset: Tensor, valid: Tensor) -> Tensor:
        """Bool ``[B, 1, T, M + T]``: True where window position ``t`` (row) attends a cached
        slot (the first ``M`` columns) or a window position ``s`` (the last ``T``). A cached
        slot needs ``valid`` (``[B, M]``: its document is the one going on) and no reset at or
        before ``t``; a window position needs ``s ≤ t`` and the document of ``t``. Every row
        holds its own position."""
        segment = reset.to(torch.int64).cumsum(dim=1)  # the document index of every position
        going_on = segment == 0  # [B, T]: no reset at or before t, the cached document goes on
        cached = valid[:, None, :] & going_on[:, :, None]  # [B, T, M]
        return torch.cat([cached, self.attention_mask(reset)[:, 0]], dim=-1)[:, None]

    def _read_with_cache(self, x: Tensor, reset: Tensor, cache: Cache) -> tuple[Tensor, Cache]:
        """The blocks over the window ``x [B, T, d]`` after the windows ``cache`` holds; the
        output and the next cache: the last ``M`` positions of the stream after this window,
        detached."""
        assert self._rot_cos is not None and self._rot_sin is not None
        T = int(x.shape[1])
        M = self.carry
        keys, values, valid = cache
        cos, sin = self._rot_cos[: M + T], self._rot_sin[: M + T]  # positions −M … T − 1
        mask = _attention_bias(self.carry_mask(reset, valid), x)
        kept_keys: list[Tensor] = []
        kept_values: list[Tensor] = []
        for i, module in enumerate(self.blocks):
            block = cast(_Block, module)
            x, k, v = block.forward_carry(x, mask, cos, sin, keys[i], values[i])
            kept_keys.append(_keep_tail(keys[i], k, M, dim=2))
            kept_values.append(_keep_tail(values[i], v, M, dim=2))
        segment = reset.to(torch.int64).cumsum(dim=1)
        last = segment[:, -1:]  # the document the stream is in after the window
        still = valid & (last == 0)  # cached positions of a document that goes on
        # contiguous: the state keeps one memory layout from window to window (a compiled
        # window is specialised to its inputs' strides)
        valid_next = _keep_tail(still, segment == last, M, dim=1).contiguous()
        kept: Cache = (
            torch.stack(kept_keys).detach(),
            torch.stack(kept_values).detach(),
            valid_next,
        )
        return x, kept

    def _cache_shape(self, B: int) -> tuple[int, ...]:
        return (self.n_layers, B, self.n_heads, self.carry, self.d_model // self.n_heads)

    def _fill_rotary_table(self) -> None:
        """The rotary cos / sin of the positions ``−carry … train.unroll − 1`` where the
        parameters are: position ``p`` of pair ``i`` turns by ``p · 10000^(−i / half)``, the
        angles in float64 (float32 on MPS, which has none), cast to the parameters' dtype."""
        weight = self.tok_emb.weight
        wide = torch.float32 if weight.device.type == "mps" else torch.float64
        half = self.d_model // self.n_heads // 2
        exponent = torch.arange(half, device=weight.device, dtype=wide) / half
        positions = torch.arange(-self.carry, self.window, device=weight.device, dtype=wide)
        angles = positions[:, None] * (_ROTARY_BASE**-exponent)
        self._rot_cos = angles.cos().to(weight.dtype)
        self._rot_sin = angles.sin().to(weight.dtype)

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        """``.to()`` / ``.cuda()`` / ``.double()``: with rotary positions the table is computed
        again for the new device and dtype (cast, a float32 table would not be the float64
        angles)."""
        if self.position != "rotary":
            super()._apply(fn, recurse)  # type: ignore[no-untyped-call]  # torch has no types
            return self
        self._rot_cos = self._rot_sin = None
        super()._apply(fn, recurse)  # type: ignore[no-untyped-call]
        self._fill_rotary_table()
        return self

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

    def extra_repr(self) -> str:
        if self.position == "learned":
            return ""
        return f"position={self.position}, carry={self.carry}"


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
