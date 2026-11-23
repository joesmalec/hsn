"""Keyed word cells (``model.word_cells = hash``; HSN-T, opt-in; ``docs/amendments/word_cells.md``).

Stage 1 of the memory redesign of 22 Nov 2026: the keyed cells with the **by-rule table value**
(``model.word_cells_value = bytes``), the control arm. Nothing in the store is trained and no
gradient enters it; what is learned is how the decoder reads it (``U`` here and the decoder's new
input columns). The document has the rule in full; in short:

- **Words** are the word address's (:mod:`hsn.model.word_address`): letters and digits under a
  text context, an apostrophe directly between two letters, within one document. The rule is
  written out again here with a carry of its own, because the keys read the bytes as written
  (the address folds the case) and need the word's first bytes.
- **Keys.** A word ``w`` of ``n >= min_len`` bytes has the keys ``(l, w_1 .. w_l)``,
  ``l = 1 .. min(L, n)``, as integers ``l * 256^L + sum_m w_m * 256^(L - m)``. A key's candidate
  cells are ``((a_i * k + b_i) mod (2^31 - 1)) mod N_s`` for ``A`` pairs drawn from the run seed.
- **The value** of ``(l, ..)``: ``s_l(w) = (n - l)^(-1/2) * sum_{m > l} P[min(m, 16)] * T[w_m]``
  with fixed tables of +-1 (``T`` per byte as written, ``P`` per byte position); 0 when ``n = l``.
- **The write** is deferred to the end of the window (:meth:`WordCells.write`), one batched
  update of fixed shape (``T * L`` slots a row; gathers and ``amax`` scatters only). Of a key's
  word ends in the window the latest counts. (1) A key that a candidate cell holds writes
  there. (2) Every other key asks for its candidate with the oldest write position among the
  cells no write of this window has taken (an empty cell first; ties to the lower way); where
  several ask for one cell the latest word end takes it (at one word end the longer prefix).
  (3) A key that lost in 2 asks once more; what loses again is not written in this window.
  The value replaces the cell, which is the clear on a new key.
- **The read** (:meth:`WordCells.window`, :meth:`WordCells.read`). Byte ``t`` is inside a word
  when it is a letter or digit, or an apostrophe directly after a letter (it cannot yet know
  whether that one joins; the prefix counts it). With ``j`` bytes of the word so far it reads the
  candidate cell that holds ``(min(j, L), the first min(j, L) bytes)`` **as the window
  started**, so only entries of earlier windows of its document:
  ``p_t = [U_min(j,16) h ; 1[age >= 256] ; log(1 + age / 256)]``, zeros without a match.
- **The carry** (:class:`WordCellsCarry`) rides with :class:`hsn.model.hsn.WindowCarry`. A
  document start clears a row.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import ModelConfig
from hsn.model.word_address import APOSTROPHE, WORD_TABLES

__all__ = [
    "AGE_SCALE",
    "EMPTY",
    "NO_KEY",
    "OFFSETS",
    "WORD_CELLS_KEYS",
    "WRITE_ROUNDS",
    "WordCells",
    "WordCellsCarry",
    "WordCellsWindow",
]

#: The read maps ``U_j``: one per number of bytes of the word so far, ``j = 1 .. 16``; longer
#: prefixes read with the 16th.
OFFSETS = 16
#: The age (in bytes) from which an entry counts as beyond the reader's sight, and the scale of
#: the age feature.
AGE_SCALE = 256
#: The write position of a cell that holds nothing: older than any write.
EMPTY = -(1 << 40)
#: The key of a cell that holds nothing.
NO_KEY = -1
#: The deferred write: how often a key that holds no cell asks for one (steps 2 and 3).
WRITE_ROUNDS = 2
#: The keys under which a window's ``info`` holds the next :class:`WordCellsCarry`.
WORD_CELLS_KEYS: tuple[str, str, str, str] = (
    "word_cells_word",
    "word_cells_table",
    "word_cells_key",
    "word_cells_when",
)

_PRIME = (1 << 31) - 1
_SEED_TAG = 0x57434C4C  # b"WCLL"
_HUGE = 1 << 60


class WordCellsCarry(NamedTuple):
    """What a window hands to the next window of the same rows: ``word`` (int64
    ``[B, d + L + 3]``: the code sum of the word in progress, its first ``L`` bytes, its length,
    whether its last byte is a letter, whether an apostrophe is waiting), the cells ``table
    [B, N_s, d]``, their keys ``key [B, N_s]`` (int64; :data:`NO_KEY`: empty) and the byte
    positions of their last writes ``when [B, N_s]`` (int64; :data:`EMPTY`)."""

    word: Tensor
    table: Tensor
    key: Tensor
    when: Tensor


class WordCellsWindow(NamedTuple):
    """A window's by-rule record (:meth:`WordCells.window`), without a gradient. The read, per
    byte: ``value [B, T, d]`` (the matched cell; zeros without a match), ``offset``
    (``min(j, 16) - 1``), ``age_features [B, T, 2]`` and the masks ``inside``, ``hit``, ``old``
    (a match :data:`AGE_SCALE` bytes old or more). The write plan, per slot ``t * L + l - 1``:
    ``slot_key``, ``slot_active``, ``slot_when`` (``[B, T * L]``) and ``slot_value``; ``cleared
    [B]`` (a document started in the window), the ``carry`` the window started from and the
    next window's ``word`` state."""

    value: Tensor
    offset: Tensor
    age_features: Tensor
    inside: Tensor
    hit: Tensor
    old: Tensor
    slot_key: Tensor
    slot_active: Tensor
    slot_when: Tensor
    slot_value: Tensor
    cleared: Tensor
    carry: WordCellsCarry
    word: Tensor


class WordCells(nn.Module):
    """The keyed word cells of an HSN-T (module docstring). Built from a generator of its own,
    without a draw from the global stream, so every other parameter and buffer starts as it
    does without the key. ``U`` is the only parameter."""

    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        super().__init__()
        if cfg.word_cells != "hash" or cfg.word_cells_value != "bytes":
            raise ValueError(
                f"model.word_cells={cfg.word_cells!r} with model.word_cells_value="
                f"{cfg.word_cells_value!r} is not built (stage 1: hash with bytes)"
            )
        self.d = cfg.d
        self.vocab = cfg.vocab_size
        self.n_cells = cfg.word_cells_n
        self.ways = cfg.word_cells_ways
        self.levels = cfg.word_cells_levels
        self.min_len = cfg.word_cells_min_len
        self.contexts = tuple(cfg.word_contexts)
        #: The age from which an entry is "old" and the scale of the age feature.
        self.sight = AGE_SCALE
        state = np.random.SeedSequence([seed, _SEED_TAG]).generate_state(1)[0]
        generator = torch.Generator(device="cpu").manual_seed(int(state))

        def signs(*shape: int) -> Tensor:
            return torch.randint(0, 2, shape, generator=generator, dtype=torch.int8) * 2 - 1

        self.byte_code: Tensor
        self.place_code: Tensor
        self.hash_a: Tensor
        self.hash_b: Tensor
        self.register_buffer("byte_code", signs(cfg.vocab_size, cfg.d))
        self.register_buffer("place_code", signs(WORD_TABLES, cfg.d))
        self.register_buffer(
            "hash_a", torch.randint(1, _PRIME, (self.ways,), generator=generator, dtype=torch.int64)
        )
        self.register_buffer(
            "hash_b", torch.randint(0, _PRIME, (self.ways,), generator=generator, dtype=torch.int64)
        )
        self.U = nn.Parameter(torch.zeros(OFFSETS, cfg.d, cfg.d))

    # -- the carry --------------------------------------------------------------------------------

    @property
    def word_width(self) -> int:
        return self.d + self.levels + 3

    def init_carry(self, B: int, device: torch.device | str) -> WordCellsCarry:
        """The carry of a stream's first window: no word in progress, every cell empty."""
        return WordCellsCarry(
            word=torch.zeros(B, self.word_width, dtype=torch.int64, device=device),
            table=torch.zeros(B, self.n_cells, self.d, dtype=self.U.dtype, device=device),
            key=torch.full((B, self.n_cells), NO_KEY, dtype=torch.int64, device=device),
            when=torch.full((B, self.n_cells), EMPTY, dtype=torch.int64, device=device),
        )

    def check_carry(self, carry: object, B: int) -> WordCellsCarry:
        """``carry`` as a :class:`WordCellsCarry` of ``B`` rows of this module's shapes."""
        N, d = self.n_cells, self.d
        shapes = ((B, self.word_width), (B, N, d), (B, N), (B, N))
        parts = tuple(carry) if isinstance(carry, list | tuple) else ()
        ok = len(parts) == 4 and all(
            isinstance(t, Tensor) and tuple(t.shape) == shape
            for t, shape in zip(parts, shapes, strict=False)
        )
        if ok:
            word, table, key, when = parts
            ok = (
                word.dtype == key.dtype == when.dtype == torch.int64
                and table.is_floating_point()
                and not table.requires_grad
            )
        if not ok:
            raise ValueError(
                "model.word_cells: the carried cells must be the WordCellsCarry of the window "
                f"before (word int64 {shapes[0]}, table {shapes[1]}, key and when int64 "
                f"{shapes[2]}), as address_history builds it from that window's info"
            )
        return WordCellsCarry(*parts)

    # -- the fixed rule ---------------------------------------------------------------------------

    def candidates(self, keys: Tensor) -> Tensor:
        """The candidate cells of ``keys`` (int64, any shape): ``[..., A]``."""
        mixed = (keys[..., None] * self.hash_a + self.hash_b) % _PRIME
        return mixed % self.n_cells

    @torch.no_grad()
    def window(
        self,
        tokens: Tensor,
        ctx: Tensor,
        reset: Tensor,
        positions: Tensor,
        carry: WordCellsCarry,
    ) -> WordCellsWindow:
        """A window's keys, its read of the carried cells and its write plan, by rule.
        ``tokens``, ``ctx``, ``reset`` and ``positions`` (every column's byte position in its
        document) are ``[B, T]``. No column reads a later byte."""
        B, T = (int(n) for n in tokens.shape)
        d, L = self.d, self.levels
        device = tokens.device
        x = tokens.to(torch.int64)
        text = torch.zeros_like(reset)
        for context_id in self.contexts:
            text = text | (ctx == context_id)
        sums0, first0 = carry.word[:, :d], carry.word[:, d : d + L]
        len0, letter0, pending0 = carry.word[:, d + L :].unbind(dim=1)
        held = pending0 > 0  # the window before ended on an apostrophe directly after a letter
        rows = torch.arange(B, device=device)[:, None]
        steps = torch.arange(T, device=device)

        # the word rule (hsn.model.word_address, on the bytes as written)
        letter = text & (((x >= 65) & (x <= 90)) | ((x >= 97) & (x <= 122)))
        base = letter | (text & (x >= 48) & (x <= 57))
        same = ~reset

        def after(first: Tensor, row: Tensor) -> Tensor:
            return torch.cat([first[:, None], row[:, :-1]], dim=1) & same

        going = (len0 > 0) & ~held
        hangs = text & (x == APOSTROPHE) & after(going & (letter0 > 0), letter)
        hung = after(held, hangs)
        joins = hung & letter
        word = base | torch.cat([joins[:, 1:], torch.zeros_like(joins[:, :1])], dim=1)
        close_late = hung & ~letter
        closes = (after(going, base) & ~base & ~hangs) | close_late

        def place(index: Tensor, byte: Tensor) -> Tensor:
            """What the ``index``-th byte (from 1) of a word adds to its code sum, ``[..., d]``."""
            return (self.place_code[index.clamp(1, WORD_TABLES) - 1] * self.byte_code[byte]).to(
                torch.int64
            )

        # a carried apostrophe that column 0 makes a word byte: the carried word takes it first
        takes = joins[:, 0]
        quote = torch.full_like(len0, APOSTROPHE)
        sums0 = sums0 + place(len0 + 1, quote) * takes[:, None]
        slot = torch.arange(L, device=device)[None, :] == len0[:, None]
        first0 = torch.where(slot & takes[:, None], quote[:, None], first0)
        len0 = len0 + takes

        # each byte's place in its word, the word's first bytes and its code sum up to the byte
        before = torch.cat([((len0 > 0) & (~held | takes))[:, None], word[:, :-1]], dim=1) & same
        start = torch.where(word & ~before, steps, -1).cummax(dim=1).values
        carried = start < 0  # the word began in a window before
        index = torch.where(carried, len0[:, None] + steps + 1, steps - start + 1)
        lead = torch.cat([first0, x], dim=1)  # [B, L + T]
        places = torch.arange(L, device=device)
        column = ((steps - index + 1)[..., None] + places).clamp(max=T - 1)  # [B, T, L]
        source = torch.where(column >= 0, column + L, places.expand_as(column))
        known = index[..., None] > places
        prefix = lead.gather(1, source.reshape(B, T * L)).view(B, T, L) * known
        added = place(index, x) * word[..., None]
        total = torch.cat([torch.zeros_like(added[:, :1]), added.cumsum(dim=1)], dim=1)
        sums = total[:, 1:] - total[rows, start.clamp_min(0)] + sums0[:, None] * carried[..., None]
        weights = 256 ** torch.arange(L - 1, -1, -1, device=device)  # [L]

        # the read: the key of the prefix so far against the cells as the window started
        inside = base | hangs
        level = index.clamp(max=L)
        key_now = level * 256**L + (prefix * weights).sum(dim=-1)
        cells = self.candidates(key_now)  # [B, T, A]
        flat = cells.reshape(B, T * self.ways)
        tags = carry.key.gather(1, flat).view(B, T, self.ways)
        match = tags == key_now[..., None]
        way = match.to(torch.int8).argmax(dim=-1, keepdim=True)
        cell = cells.gather(2, way).squeeze(2)
        fresh = reset.cumsum(dim=1) > 0  # a document started in the window: nothing to read
        hit = inside & ~fresh & match.any(dim=-1)
        age = (positions - carry.when.gather(1, cell)).clamp_min(0) * hit
        value = carry.table[rows, cell] * hit[..., None]
        old = hit & (age >= self.sight)
        scaled = age.to(value.dtype) / self.sight
        age_features = torch.stack([old.to(value.dtype), torch.log1p(scaled)], dim=-1)

        # the write plan: the word a column closes, as it stood after its last byte
        at = torch.where(close_late, (steps - 1).clamp_min(0), steps)
        length = torch.cat([len0[:, None], index * word], dim=1)[rows, at]
        closed_sum = torch.cat([sums0[:, None], sums], dim=1)[rows, at]  # [B, T, d]
        closed_first = torch.cat([first0[:, None], prefix], dim=1)[rows, at]  # [B, T, L]
        levels = torch.arange(1, L + 1, device=device)
        head = self.place_code[:L] * self.byte_code[closed_first]  # [B, T, L, d]
        rest = closed_sum[:, :, None] - head.to(torch.int64).cumsum(dim=2)
        count = (length[..., None] - levels).clamp_min(1)
        slot_value = rest.to(value.dtype) / count.to(value.dtype).sqrt()[..., None]
        slot_key = levels * 256**L + (closed_first * weights).cumsum(dim=-1)
        last_reset = torch.where(reset, steps, -1).amax(dim=1, keepdim=True)
        live = closes & (steps > last_reset) & (length >= self.min_len)
        slot_active = live[..., None] & (length[..., None] >= levels)

        # the word in progress at the window's end
        hanging = hangs[:, -1]
        alive = hanging | base[:, -1]
        end = torch.where(hanging, T - 1, T)[:, None]
        tail = [
            torch.cat([len0[:, None], index * word], dim=1)[rows, end][:, 0] * alive,
            torch.cat([letter0[:, None] > 0, letter], dim=1)[rows, end][:, 0] & alive,
            hanging,
        ]
        next_word = torch.cat(
            [
                torch.cat([sums0[:, None], sums], dim=1)[rows, end][:, 0] * alive[:, None],
                torch.cat([first0[:, None], prefix], dim=1)[rows, end][:, 0] * alive[:, None],
                torch.stack([t.to(torch.int64) for t in tail], dim=1),
            ],
            dim=1,
        )
        return WordCellsWindow(
            value=value,
            offset=index.clamp(1, OFFSETS) - 1,
            age_features=age_features,
            inside=inside,
            hit=hit,
            old=old,
            slot_key=slot_key.reshape(B, T * L),
            slot_active=slot_active.reshape(B, T * L),
            slot_when=positions[..., None].expand(B, T, L).reshape(B, T * L),
            slot_value=slot_value.reshape(B, T * L, d),
            cleared=reset.any(dim=1),
            carry=carry,
            word=next_word,
        )

    @torch.no_grad()
    def write(self, win: WordCellsWindow) -> WordCellsCarry:
        """The deferred write of a window (module docstring): the next window's carry."""
        N, d, A = self.n_cells, self.d, self.ways
        B, W = (int(n) for n in win.slot_key.shape)
        device = win.slot_key.device
        cleared = win.cleared
        key0 = win.carry.key.masked_fill(cleared[:, None], NO_KEY)
        when0 = win.carry.when.masked_fill(cleared[:, None], EMPTY)
        table0 = win.carry.table.masked_fill(cleared[:, None, None], 0.0)
        keys = win.slot_key
        cells = self.candidates(keys)  # [B, W, A]
        flat = cells.reshape(B, W * A)
        ways = torch.arange(A, device=device)
        holds = key0.gather(1, flat).view(B, W, A) == keys[..., None]
        holder = holds.any(dim=-1)
        stale = when0.gather(1, flat).view(B, W, A) * A + ways  # oldest first, then the lower way
        order = torch.arange(W, device=device).expand(B, W)
        pending = win.slot_active
        taken = torch.zeros(B, N + 1, dtype=torch.bool, device=device)
        winner = torch.full((B, N + 1), -1, dtype=torch.int64, device=device)
        for turn in range(1 + WRITE_ROUNDS):
            if turn == 0:  # a key that holds a cell writes there
                way = holds.to(torch.int8).argmax(dim=-1, keepdim=True)
                asks = pending & holder
            else:  # the others: the oldest candidate that no write of this window has taken
                barred = taken.gather(1, flat).view(B, W, A)
                way = torch.where(barred, ways + _HUGE, stale).argmin(dim=-1, keepdim=True)
                asks = pending & ~barred.gather(2, way).squeeze(2)
            target = torch.where(asks, cells.gather(2, way).squeeze(2), N)
            best = torch.full_like(winner, -1).scatter_reduce(
                1, target, torch.where(asks, order, -1), "amax", include_self=True
            )  # the latest slot that asks for each cell (cell N: the slots that do not ask)
            got = best >= 0
            got[:, N] = False
            winner = torch.where(got, best, winner)
            taken = taken | got
            if turn == 0:
                pending = pending & ~holder
            else:  # whoever lost the cell to another key asks once more
                won = keys.gather(1, best.clamp_min(0)).masked_fill(~got, NO_KEY - 1)
                pending = asks & (won.gather(1, target) != keys)
        has = winner[:, :N] >= 0
        slot = winner[:, :N].clamp_min(0)
        table = torch.where(
            has[..., None], win.slot_value.gather(1, slot[..., None].expand(B, N, d)), table0
        )
        return WordCellsCarry(
            word=win.word,
            table=table,
            key=torch.where(has, keys.gather(1, slot), key0),
            when=torch.where(has, win.slot_when.gather(1, slot), when0),
        )

    # -- the learned read -------------------------------------------------------------------------

    def read(self, win: WordCellsWindow) -> Tensor:
        """``p_t = [U_min(j,16) h ; 1[age >= 256] ; log(1 + age / 256)]``, ``[B, T, d + 2]``:
        zeros without a match. The gradient reaches ``U`` only (the cells are by rule)."""
        B, T, d = (int(n) for n in win.value.shape)
        spread = win.value.new_zeros(B, T, OFFSETS, d)
        spread.scatter_(2, win.offset[:, :, None, None].expand(B, T, 1, d), win.value[:, :, None])
        weight = self.U.permute(1, 0, 2).reshape(d, OFFSETS * d)
        mapped = F.linear(spread.view(B, T, OFFSETS * d), weight)
        return torch.cat([mapped, win.age_features.to(mapped.dtype)], dim=-1)

    @torch.no_grad()
    def rates(self, win: WordCellsWindow) -> tuple[Tensor, Tensor]:
        """A window's two statistics (0-dim): the share of its bytes inside a word that read
        an entry, and the share of those entries that are :attr:`sight` bytes old or more."""
        inside = win.inside.sum()
        hit = win.hit.sum()
        hit_rate = hit.to(torch.float32) / inside.clamp_min(1)
        old_rate = win.old.sum().to(torch.float32) / hit.clamp_min(1)
        return hit_rate, old_rate

    def extra_repr(self) -> str:
        return f"cells={self.n_cells}, ways={self.ways}, levels={self.levels}"
