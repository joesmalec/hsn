"""Frozen changed methods from reader.py at 8f9f184 (16 Nov 2026).

The untouched block equations and helpers come from Reader. These old methods
preserve the original table bounds, bool masks, and concatenate-then-slice cache
updates, independently of the optimized implementations under test.
"""

from typing import cast

import torch
from hsn.model.reader import (
    _NO_POSITION,
    _ROTARY_TABLE,
    Cache,
    Reader,
    _ReaderBlock,
    _rotary_angles,
    word_bytes,
    word_ends,
)
from torch import Tensor


class BeforeCarrySpeed(Reader):
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

    def forward(self, tokens: Tensor, reset: Tensor) -> Tensor:
        if tokens.ndim != 2 or reset.shape != tokens.shape or reset.dtype is not torch.bool:
            raise ValueError(
                f"Reader needs tokens [B, T] and a bool reset of the same shape, got "
                f"{tuple(tokens.shape)} and {reset.dtype} {tuple(reset.shape)}"
            )
        if self.carry:
            raise ValueError(
                f"a reader with model.reader_carry={self.carry} reads with "
                "read(tokens, reset, cache), the cache from init_cache"
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

    def read(self, tokens: Tensor, reset: Tensor, cache: Cache) -> tuple[Tensor, Cache]:
        """``model.reader_carry``: ``(c, cache)`` for the window ``tokens`` after the windows
        ``cache`` holds (module docstring, "Reading past the window"). ``c [B, T, w]`` is the
        reader's output, as :meth:`forward`'s; the cache returned holds the last ``C``
        positions of the stream after this window, and with ``model.reader_top_words`` its
        last ``N`` word ends, detached."""
        if tokens.ndim != 2 or reset.shape != tokens.shape or reset.dtype is not torch.bool:
            raise ValueError(
                f"Reader needs tokens [B, T] and a bool reset of the same shape, got "
                f"{tuple(tokens.shape)} and {reset.dtype} {tuple(reset.shape)}"
            )
        if not self.carry:
            raise ValueError("read() is the reader with model.reader_carry; call the reader")
        B, T = (int(n) for n in tokens.shape)
        cache = self.check_cache(cache, B)
        keys, values, valid = cache[:3]
        C, N = self.carry, self.top_words
        x = self.emb(tokens)
        # the angles of the positions −C … T − 1: the window's as forward's, the cached ones
        # those of 1 … C with the sine's sign turned (cos is even, sin odd)
        cos_all, sin_all = self._rotation(max(T, C + 1), x)
        cos = torch.cat([cos_all[1 : C + 1].flip(0), cos_all[:T]])
        sin = torch.cat([-sin_all[1 : C + 1].flip(0), sin_all[:T]])
        mask = self.carry_mask(reset, valid)
        word_keys = word_values = None
        top_mask = mask
        if N:  # the top block also reads word ends beyond the span: the window's and the slots
            word_keys, word_values, word_valid, word_pos, last_letter = cache[3:]
            ends = word_ends(tokens, reset, last_letter)
            window_words, cached_words = self.word_masks(reset, ends, word_valid, word_pos)
            spans = mask[:, 0]
            window_reads = spans[:, :, C:] | window_words
            top_mask = torch.cat([spans[:, :, :C], window_reads, cached_words], dim=-1)[:, None]
        kept_keys: list[Tensor] = []
        kept_values: list[Tensor] = []
        top = len(self.blocks) - 1
        for i, module in enumerate(self.blocks):
            block = cast(_ReaderBlock, module)
            block_cos, block_sin = self._block_rotation(self._block_pairs[i], cos, sin)
            words = N > 0 and i == top
            x, k, v = block.forward_carry(
                x,
                top_mask if words else mask,
                block_cos,
                block_sin,
                keys[i],
                values[i],
                word_keys if words else None,
                word_values if words else None,
            )
            kept_keys.append(torch.cat([keys[i], k.to(keys.dtype)], dim=2)[:, :, T:])
            kept_values.append(torch.cat([values[i], v.to(values.dtype)], dim=2)[:, :, T:])
        segment = reset.to(torch.int64).cumsum(dim=1)
        last = segment[:, -1:]  # the document the stream is in after the window
        still = valid & (last == 0)  # cached positions of a document that goes on
        # contiguous: the cache keeps one memory layout from window to window (a compiled
        # window is specialised to its inputs' strides)
        valid_next = torch.cat([still, segment == last], dim=1)[:, T:].contiguous()
        out: Tensor = self.norm(x)
        kept: Cache = (
            torch.stack(kept_keys).detach(),
            torch.stack(kept_values).detach(),
            valid_next,
        )
        if N:  # the next word slots: the last N word ends of the document going on
            assert word_keys is not None and word_values is not None
            index = torch.arange(T, device=tokens.device)
            position = torch.cat([word_pos, index.expand(B, T)], dim=1)  # [B, N + T]
            ok = torch.cat([word_valid & (last == 0), ends & (segment == last)], dim=1)
            pick = torch.where(ok, position, _NO_POSITION).topk(N, dim=1).indices
            ok_next = ok.gather(1, pick)  # the most recent first, the empty slots last
            pos_next = torch.where(ok_next, position.gather(1, pick) - T, _NO_POSITION)
            rows = pick[:, None, :, None].expand(-1, self.heads, -1, word_keys.shape[-1])
            all_keys = torch.cat([word_keys, k.to(word_keys.dtype)], dim=2)  # k: the top's
            all_values = torch.cat([word_values, v.to(word_values.dtype)], dim=2)
            kept += (
                all_keys.gather(2, rows).detach(),
                all_values.gather(2, rows).detach(),
                ok_next,
                pos_next,
                word_bytes(tokens[:, -1]),
            )
        return out, kept
