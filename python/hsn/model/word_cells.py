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

Stage 2 (built 22 Nov 2026, untrained): the HSN's own write and candidate 2.

- **``model.word_cells_value = word | cell``** (:meth:`WordCells.write_learned`). The write is
  still deferred and once per key per window, with the model's values at the key's last word
  end ``e`` (:class:`KeyedFeed`). (1) as above. (2), (3) as above with the choice by the
  router's scorer: its best candidate (``max_j`` of its logits on ``[u_e ; eps_e ; h^K_i ;
  age_i ; the event's scalars]``, the address features zero) among the cells no write of the
  window has taken. A cell taken by a new key is cleared. Then the cell's update, with the
  cell's own modules: ``m`` from ``model.in_degree`` fixed in-edges from the routed cells as
  event ``e`` left them (``V``, learned edge weights; read as values, see the document), ``v =
  RMSNorm([u_e + V_s s_l(w) ; eps_e ; h^K_i ; m])`` (``cell``: without ``V_s s_l(w)``), the unit
  the scorer's logits choose, ``W_out``, the candidate's norm and tanh, the gate ``sigmoid(B v
  + beta + beta_K)`` and the convex update. The straight-through weight of the choice (the cell
  among its free candidates, the unit among the units) multiplies the candidate's
  pre-activation, as in a routed slot. The cells keep their gradient to the next window when
  the caller carries them attached (``train.tbptt_windows > 1``).
- **``model.word_cells = address``** (candidate 2): no hash and no tag. Every cell has a fixed
  address ``A_i``; ``W_l(k)`` is the exact word code (:mod:`hsn.model.word_address`) of the
  prefix. A write takes the scorer's best cell of all, the cell's rank being its best logit
  plus ``kappa * sqrt(d) * <W_l(k), A_i>``; nothing is cleared. A byte inside a word reads the
  :data:`READ_CELLS` best-ranked cells by the same logits at the byte and the code of its
  prefix so far, without a write: ``p_t = [U_j mean h ; the age inputs' means]``
  (:meth:`WordCells.read_learned`).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import ModelConfig
from hsn.model.cell import PDGCell
from hsn.model.embeddings import init_weight_
from hsn.model.router import Router
from hsn.model.word_address import APOSTROPHE, WORD_TABLES

__all__ = [
    "AGE_CAP",
    "AGE_SCALE",
    "EMPTY",
    "GATE_BIAS",
    "NO_KEY",
    "OFFSETS",
    "READ_CELLS",
    "WORD_CELLS_KEYS",
    "WRITE_ROUNDS",
    "KeyedFeed",
    "WordCells",
    "WordCellsCarry",
    "WordCellsWindow",
    "WriteRequests",
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
#: Stage 2. The design's start of the keyed gate's own bias ``beta_K`` (added to the cell's
#: gate bias): the default of ``model.word_cells_gate_bias``.
GATE_BIAS = 3.0
#: The scorer's age feature of a keyed cell is ``log(1 + min(age, AGE_CAP) / AGE_SCALE)``, the
#: age in bytes since its last write; a cell that holds nothing reads as :data:`AGE_CAP` old.
AGE_CAP = 16 * AGE_SCALE
#: ``model.word_cells = address``: the cells a byte inside a word reads (``K_r``).
READ_CELLS = 2
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


class WriteRequests(NamedTuple):
    """The learned write's requests of a window (:meth:`WordCells.requests`), by rule: one per
    key of a row, at the key's last word end in the window. Int64: ``row [P]`` (the batch row),
    ``slot [P]`` (the write plan's slot ``t * L + l - 1``), ``key [P]``, ``when [P]`` (the word
    end's byte position), ``cand [P, C]`` (the candidate cells: ``A`` with the hash, all ``N_s``
    with ``model.word_cells=address``) and ``flat [P, C]`` (``row * N_s + cand``: a candidate
    in the rows' cells laid end to end)."""

    row: Tensor
    slot: Tensor
    key: Tensor
    when: Tensor
    cand: Tensor
    flat: Tensor


class WordCellsWindow(NamedTuple):
    """A window's by-rule record (:meth:`WordCells.window`), without a gradient. The read, per
    byte: ``value [B, T, d]`` (the matched cell; zeros without a match), ``offset``
    (``min(j, 16) - 1``), ``age_features [B, T, 2]`` and the masks ``inside``, ``hit``, ``old``
    (a match :data:`AGE_SCALE` bytes old or more). The write plan, per slot ``t * L + l - 1``:
    ``slot_key``, ``slot_active``, ``slot_when`` (``[B, T * L]``) and ``slot_value``; ``cleared
    [B]`` (a document started in the window), the ``carry`` the window started from and the
    next window's ``word`` state. Stage 2: ``cell [B, T]``, the matched cell of each byte (the
    learned arms read it again, with the carried gradient); with ``model.word_cells=address``
    there is no match (``hit``: the bytes that read, ``value`` and the ages are the model's
    read) and ``read_code [B, T, d]`` / ``slot_code [B, T * L, d]`` are the exact word codes
    of each byte's prefix so far and of each slot's key. ``requests``: the learned write's
    requests (:class:`WriteRequests`), taken with the plan so that the write itself needs no
    host read before its turns are decided (``None``: the write takes them)."""

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
    cell: Tensor | None = None
    read_code: Tensor | None = None
    slot_code: Tensor | None = None
    requests: WriteRequests | None = None


class KeyedFeed(NamedTuple):
    """What the model hands the learned arms after its event loop
    (:meth:`hsn.model.hsn.HSN._keyed_feed`). Per byte, in the state's dtype: ``u [B, T, d]``
    (at an event the event slot's own), ``eps [B, T, d]`` (the error as the cell reads it),
    ``scalars [B, T, 7]`` (the scorer's scalars behind the age, the two per-cell averages
    zero) and ``slot [B, T]`` (the number of the row's events at or before the byte: event
    ``slot - 1`` of ``routed``). ``routed [B, E, N, d]``: the routed cells as every event slot
    left them, without a gradient. The cell and the router whose modules the write runs, and
    the step's controls: ``gate`` (``None`` in evaluation: no straight-through terms), ``tau``
    and ``cut`` (``train.feedback_detach`` holds ``router``: the scorer's features are
    constants in the backward pass)."""

    u: Tensor
    eps: Tensor
    scalars: Tensor
    routed: Tensor
    slot: Tensor
    cell: PDGCell
    router: Router
    gate: str | None
    tau: float
    cut: bool


class WordCells(nn.Module):
    """The keyed word cells of an HSN-T (module docstring). Built from a generator of its own,
    without a draw from the global stream, so every other parameter and buffer starts as it
    does without the key. With the by-rule value ``U`` is the only parameter; the learned
    arms add the keyed in-edges' logits ``edge_logit [N_s, D_g]`` (zero: uniform weights),
    the gate bias ``beta_K [d]`` and, for ``word``, ``V_s [d, d]`` (``model.state_init``),
    all drawn after stage 1's tables."""

    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        super().__init__()
        self.addressed = cfg.word_cells == "address"
        #: The value arm (``bytes | word | cell``) and whether the HSN writes the cells.
        self.kind = cfg.word_cells_value
        self.learned = self.kind != "bytes"
        if cfg.word_cells == "off" or (self.addressed and not self.learned):
            raise ValueError(
                f"model.word_cells={cfg.word_cells!r} with model.word_cells_value="
                f"{self.kind!r} is not built (hash with any value, address with word | cell)"
            )
        self.d = cfg.d
        self.n_cells = cfg.word_cells_n
        self.ways = cfg.word_cells_ways
        self.levels = cfg.word_cells_levels
        self.min_len = cfg.word_cells_min_len
        self.contexts = tuple(cfg.word_contexts)
        self.kappa = cfg.word_cells_kappa
        self.edge_norm = cfg.edge_norm
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
        self.sources: Tensor
        self.addresses: Tensor
        self.register_buffer("byte_code", signs(cfg.vocab_size, cfg.d))
        self.register_buffer("place_code", signs(WORD_TABLES, cfg.d))
        for name, low in (("hash_a", 1), ("hash_b", 0)):
            drawn = torch.randint(low, _PRIME, (self.ways,), generator=generator, dtype=torch.int64)
            if not self.addressed:  # candidate 2 has no hash
                self.register_buffer(name, drawn)
        self.U = nn.Parameter(torch.zeros(OFFSETS, cfg.d, cfg.d))
        self.V_s: nn.Parameter | None
        self.register_parameter("V_s", None)
        if self.learned:
            # stage 2, drawn behind stage 1's tables: each keyed cell's fixed in-edges from the
            # routed cells (distinct sources), their logits, the gate bias and the word's map
            ranks = torch.rand(self.n_cells, cfg.n_cells, generator=generator)
            self.register_buffer("sources", ranks.argsort(dim=1)[:, : cfg.in_degree])
            self.edge_logit = nn.Parameter(torch.zeros(self.n_cells, cfg.in_degree))
            self.beta_K = nn.Parameter(torch.full((cfg.d,), float(cfg.word_cells_gate_bias)))
            if self.kind == "word":
                weight = torch.empty(cfg.d, cfg.d)
                self.V_s = nn.Parameter(init_weight_(weight, cfg.state_init, generator=generator))
        if self.addressed:  # candidate 2: a fixed address per cell, scaled as A5's
            address = torch.randn(self.n_cells, cfg.d, generator=generator) / math.sqrt(cfg.d)
            self.register_buffer("addresses", address)

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
                # the learned arms' cells keep their gradient inside a gradient segment
                and (self.learned or not table.requires_grad)
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
        tables: tuple[Tensor, Tensor] | None = None,
    ) -> WordCellsWindow:
        """A window's keys, its read of the carried cells and its write plan, by rule.
        ``tokens``, ``ctx``, ``reset`` and ``positions`` (every column's byte position in its
        document) are ``[B, T]``. No column reads a later byte. ``tables``
        (``model.word_cells=address``): the scorer's sign bits of the exact word code
        ``[16, V, d]`` and A5's byte codes ``[V, d]``."""
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
        fresh = reset.cumsum(dim=1) > 0  # a document started in the window: nothing to read
        read_code: Tensor | None = None
        if self.addressed:
            # candidate 2: no tag to match; the model ranks the cells (read_learned)
            assert tables is not None
            cell, hit = torch.zeros_like(index), inside & ~fresh
            value = carry.table.new_zeros(B, T, d)
            old, age_features = torch.zeros_like(hit), value[..., :2]
            pick = (level - 1).clamp_min(0)[..., None, None].expand(B, T, 1, d)
            read_code = self._codes(prefix, tables).gather(2, pick).squeeze(2)
        else:
            key_now = level * 256**L + (prefix * weights).sum(dim=-1)
            cells = self.candidates(key_now)  # [B, T, A]
            flat = cells.reshape(B, T * self.ways)
            tags = carry.key.gather(1, flat).view(B, T, self.ways)
            match = tags == key_now[..., None]
            way = match.to(torch.int8).argmax(dim=-1, keepdim=True)
            cell = cells.gather(2, way).squeeze(2)
            hit = inside & ~fresh & match.any(dim=-1)
            age = (positions - carry.when.gather(1, cell)).clamp_min(0) * hit
            value = carry.table[rows, cell] * hit[..., None]
            old = hit & (age >= self.sight)
            scaled = age.to(value.dtype) / self.sight
            age_features = torch.stack([old.to(value.dtype), torch.log1p(scaled)], dim=-1)

        # the word as it stands after byte t at index t + 1; index 0: as the window before left it
        len_x = torch.cat([len0[:, None], index * word], dim=1)
        sum_x = torch.cat([sums0[:, None], sums], dim=1)
        first_x = torch.cat([first0[:, None], prefix], dim=1)

        # the write plan: the word a column closes, as it stood after its last byte
        at = torch.where(close_late, (steps - 1).clamp_min(0), steps)
        length, closed_sum, closed_first = len_x[rows, at], sum_x[rows, at], first_x[rows, at]
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
            len_x[rows, end][:, 0] * alive,
            torch.cat([letter0[:, None] > 0, letter], dim=1)[rows, end][:, 0] & alive,
            hanging,
        ]
        next_word = torch.cat(
            [
                sum_x[rows, end][:, 0] * alive[:, None],
                first_x[rows, end][:, 0] * alive[:, None],
                torch.stack([t.to(torch.int64) for t in tail], dim=1),
            ],
            dim=1,
        )
        plan_key = slot_key.reshape(B, T * L)
        plan_active = slot_active.reshape(B, T * L)
        plan_when = positions[..., None].expand(B, T, L).reshape(B, T * L)
        return WordCellsWindow(
            value=value,
            offset=index.clamp(1, OFFSETS) - 1,
            age_features=age_features,
            inside=inside,
            hit=hit,
            old=old,
            slot_key=plan_key,
            slot_active=plan_active,
            slot_when=plan_when,
            slot_value=slot_value.reshape(B, T * L, d),
            cleared=reset.any(dim=1),
            carry=carry,
            word=next_word,
            cell=cell,
            read_code=read_code,
            slot_code=(
                None if tables is None else self._codes(closed_first, tables).reshape(B, T * L, d)
            ),
            # the learned write's requests, here: their count is a host read, and here it waits
            # beside the event slots' own (docs/wcells_speed.md)
            requests=self.requests(plan_key, plan_active, plan_when) if self.learned else None,
        )

    @torch.no_grad()
    def requests(self, slot_key: Tensor, slot_active: Tensor, slot_when: Tensor) -> WriteRequests:
        """The learned write's requests from a window's write plan (``[B, T * L]`` each): one
        per key of a row, the key's last word end in the window. One host read (how many)."""
        B, W = (int(n) for n in slot_key.shape)
        N = self.n_cells
        device = slot_key.device
        order = torch.arange(W, device=device)
        ranked = torch.where(slot_active, slot_key * W + order, _HUGE).sort(dim=1).values
        after = torch.cat([ranked[:, 1:], ranked.new_full((B, 1), -W)], dim=1)
        last = (ranked // W != after // W) & (ranked < _HUGE)
        row, place = last.nonzero(as_tuple=True)
        slot = ranked[row, place] % W
        key = slot_key[row, slot]
        if self.addressed:
            cand = torch.arange(N, device=device).expand(int(row.numel()), N)
        else:
            cand = self.candidates(key)  # [P, A]
        flat = row[:, None] * N + cand
        return WriteRequests(row, slot, key, slot_when[row, slot], cand, flat)

    def _codes(self, first: Tensor, tables: tuple[Tensor, Tensor]) -> Tensor:
        """The exact word codes (:mod:`hsn.model.word_address`) of the prefixes of ``first
        [..., L]`` (bytes as written; A-Z are folded as in the address), ``[..., L, d]``: one
        byte reads A5's byte code, ``l`` bytes ``d^(-1/2)`` times the product of their signs."""
        bits, byte_codes = tables
        folded = torch.where((first >= 65) & (first <= 90), first + 32, first)
        places = torch.arange(self.levels, device=first.device)
        flips = bits[places, folded].to(torch.int64).cumsum(dim=-2) & 1  # [..., L, d]
        code = (1 - 2 * flips).to(byte_codes.dtype) / math.sqrt(self.d)
        return torch.cat([byte_codes[folded[..., :1]], code[..., 1:, :]], dim=-2)

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
        return self._mapped(win.value, win.offset, win.age_features)

    # -- stage 2: the HSN's write (word | cell) and candidate 2 (address) --------------------------

    def _logits(self, feed: KeyedFeed, shared: Tensor, part: Tensor, age: Tensor) -> Tensor:
        """The scorer's logits of keyed cells, fp32 ``[..., C, M]`` (less the router's cost
        term): the scorer's linear map split by feature. ``shared [..., 1, M]`` holds what
        the cells of a row share (:meth:`_shared`), ``part [..., C, M]`` each cell's own term
        (:meth:`_cell_part`) and ``age [..., C]`` the bytes since its last write. The two
        running averages and the address features of a keyed cell are zero."""
        weight = self._weights(feed)[:, 3 * self.d]
        scaled = torch.log1p(age.clamp(0, AGE_CAP).to(weight.dtype) / AGE_SCALE)
        logits = (shared + part + scaled[..., None] * weight).float()
        router = feed.router
        if router.lambda_cost > 0.0:
            logits = logits - router.lambda_cost * router.cost_table.float()
        return logits

    def _weights(self, feed: KeyedFeed) -> Tensor:
        scorer = feed.router.scorer
        assert isinstance(scorer.net, nn.Linear)
        weight = scorer.net.weight
        return weight.masked_fill(scorer._excluded, 0.0) if scorer.exclude else weight

    def _shared(self, feed: KeyedFeed, u: Tensor, eps: Tensor, scalars: Tensor) -> Tensor:
        """The scorer's terms of ``u``, ``eps`` and the scalars behind the age, ``[..., M]``."""
        d = self.d
        weight = self._weights(feed)
        columns = torch.cat([weight[:, : 2 * d], weight[:, 3 * d + 1 : 3 * d + 8]], dim=1)
        if feed.cut:
            u, eps = u.detach(), eps.detach()
        assert isinstance(feed.router.scorer.net, nn.Linear)
        return F.linear(torch.cat([u, eps, scalars], dim=-1), columns, feed.router.scorer.net.bias)

    def _cell_part(self, feed: KeyedFeed, table: Tensor) -> Tensor:
        """The scorer's term of every keyed cell's own state, ``[B, N_s, M]``."""
        d = self.d
        return F.linear(
            table.detach() if feed.cut else table, self._weights(feed)[:, 2 * d : 3 * d]
        )

    def _prior(self, code: Tensor) -> Tensor:
        """Candidate 2's address prior ``kappa * sqrt(d) * <code, A_i>``, fp32 ``[..., N_s]``,
        outside autocast (as A5's address features), so that a key's write and its reads rank
        the cells alike."""
        with torch.autocast(device_type=code.device.type, enabled=False):
            return (self.kappa * math.sqrt(self.d)) * (code @ self.addresses.T).float()

    def _mapped(self, value: Tensor, offset: Tensor, age_features: Tensor) -> Tensor:
        """``[U_offset value ; age_features]``, ``[B, T, d + 2]`` (the read's learned map)."""
        B, T, d = (int(n) for n in value.shape)
        spread = value.new_zeros(B, T, OFFSETS, d)
        spread.scatter_(2, offset[:, :, None, None].expand(B, T, 1, d), value[:, :, None])
        weight = self.U.permute(1, 0, 2).reshape(d, OFFSETS * d)
        mapped = F.linear(spread.view(B, T, OFFSETS * d), weight)
        return torch.cat([mapped, age_features.to(mapped.dtype)], dim=-1)

    def read_learned(self, win: WordCellsWindow, feed: KeyedFeed) -> tuple[Tensor, Tensor, Tensor]:
        """``p_t [B, T, d + 2]`` of the learned arms and the read's two statistics
        (:meth:`rates`). ``hash``: the tag match of :meth:`window`, read once more so that the
        value keeps the carried cells' gradient. ``address``: at every byte inside a word the
        :data:`READ_CELLS` cells the scorer's logits at the byte plus the address prior of
        the prefix so far rank first, with the selection's straight-through weight; ``p_t =
        [U_j mean h ; the means of the two age inputs over the read cells that hold a
        write]``."""
        B, T = (int(n) for n in win.offset.shape)
        table, when = win.carry.table, win.carry.when
        rows = torch.arange(B, device=table.device)[:, None]
        if not self.addressed:
            assert win.cell is not None
            value = table[rows, win.cell] * win.hit[..., None]
            return self._mapped(value, win.offset, win.age_features), *self.rates(win)
        assert win.read_code is not None
        K, L = min(READ_CELLS, self.n_cells), self.levels
        positions = win.slot_when.view(B, T, L)[..., 0]
        shared = self._shared(feed, feed.u, feed.eps, feed.scalars)[:, :, None]
        age = positions[..., None] - when[:, None]  # [B, T, N]
        scores = self._logits(feed, shared, self._cell_part(feed, table)[:, None], age)
        ranks = scores.amax(dim=-1) + self._prior(win.read_code)
        ranked = ranks.topk(min(K + 1, self.n_cells), dim=-1)
        cells = ranked.indices[..., :K]  # [B, T, K]
        held = (when[rows[..., None], cells] > EMPTY) & win.hit[..., None]
        weight = held.to(table.dtype)
        if feed.gate is not None and self.n_cells > K:
            # the straight-through weight of a read cell: its rank against the midpoint
            # between the last cell read and the first one left out
            theta = 0.5 * (ranked.values[..., K - 1] + ranked.values[..., K]).detach()
            soft = torch.sigmoid((ranked.values[..., :K] - theta[..., None]) / feed.tau)
            weight = weight * (1.0 + (soft - soft.detach())).to(table.dtype)
        value = (table[rows[..., None], cells] * weight[..., None]).sum(dim=2) / K
        age = (positions[..., None] - when[rows[..., None], cells]).clamp_min(0)
        count = held.sum(dim=-1).clamp_min(1)
        old = held & (age >= self.sight)
        logs = torch.log1p(age.to(table.dtype) / self.sight) * held
        ages = torch.stack([old.sum(dim=-1) / count, logs.sum(dim=-1) / count], dim=-1)
        hits = held.any(dim=-1).sum()
        hit_rate = hits.to(torch.float32) / win.inside.sum().clamp_min(1)
        old_rate = old.any(dim=-1).sum().to(torch.float32) / hits.clamp_min(1)
        return self._mapped(value, win.offset, ages), hit_rate, old_rate

    def write_learned(
        self, win: WordCellsWindow, feed: KeyedFeed
    ) -> tuple[WordCellsCarry, dict[str, Tensor]]:
        """The deferred write of the learned arms (module docstring): the next window's carry
        (the cells with their gradient) and the write's records: ``word_cells_units`` (bool
        ``[M]``: the units the writes ran), ``word_cells_gate`` (the mean gate of the writes)
        and, with ``address``, ``word_cells_prior`` (the share of the writes that took the
        cell the address prior alone ranks first)."""
        N, L = self.n_cells, self.levels
        B = int(win.slot_key.shape[0])
        cell, router = feed.cell, feed.router
        M = cell.n_units
        device = win.slot_key.device
        cleared = win.cleared
        key0 = win.carry.key.masked_fill(cleared[:, None], NO_KEY)
        when0 = win.carry.when.masked_fill(cleared[:, None], EMPTY)
        table0 = win.carry.table.masked_fill(cleared[:, None, None], 0.0)
        dtype = table0.dtype
        # one request per key of a row: the key's last word end in the window (by rule: taken
        # with the window's plan, WordCells.window)
        req = win.requests
        if req is None:
            req = self.requests(win.slot_key, win.slot_active, win.slot_when)
        r, slot, key, when, cand, flat = req
        P, column = int(r.numel()), slot // L
        u, eps = feed.u[r, column], feed.eps[r, column]
        mine = r[:, None]
        part = self._cell_part(feed, table0)[mine, cand]  # [P, C, M]
        shared = self._shared(feed, u, eps, feed.scalars[r, column])[:, None]
        scores = self._logits(feed, shared, part, when[:, None] - torch.take(when0, flat))
        rank = scores.amax(dim=-1)  # [P, C]: a candidate's score, as a routed cell's
        prior: Tensor | None = None
        if self.addressed:
            assert win.slot_code is not None
            prior = self._prior(win.slot_code[r, slot])
            rank = rank + prior
        # the turns: (1) the holder, (2) and (3) the scorer's best free candidate; where
        # several keys ask for one cell the latest word end takes it (the longer prefix).
        # No host read in the turns: a request that does not ask, or does not win, is sent to
        # a spare entry behind the rows' cells in place of being left out by a mask (a mask's
        # row count is a host read, and with it the write waits for the event loop before
        # it has issued its own work; docs/wcells_speed.md)
        spare = B * N
        taken = torch.zeros(spare + 1, dtype=torch.bool, device=device)
        way = torch.full((P,), -1, dtype=torch.int64, device=device)
        runner = torch.full((P,), -math.inf, device=device)  # the best free candidate left out
        pending = torch.ones(P, dtype=torch.bool, device=device)
        rank_d = rank.detach()
        for turn in range(1 + WRITE_ROUNDS):
            if turn == 0:
                if self.addressed:
                    continue
                holds = torch.take(key0, flat) == key[:, None]
                choice, asks = holds.to(torch.int8).argmax(dim=1), holds.any(dim=1)
                second = runner
            else:
                free = ~torch.take(taken, flat)
                open_ = rank_d.masked_fill(~free, -math.inf)
                choice, asks = open_.argmax(dim=1), pending & free.any(dim=1)  # ties: the first
                same = cand == cand.gather(1, choice[:, None])  # (two ways may be one cell)
                second = open_.masked_fill(same, -math.inf).amax(dim=1)
            target = flat.gather(1, choice[:, None]).squeeze(1)
            latest = torch.full((spare + 1,), -1, dtype=torch.int64, device=device)
            latest.scatter_reduce_(0, torch.where(asks, target, spare), slot, "amax")
            won = asks & (torch.take(latest, target) == slot)
            way = torch.where(won, choice, way)
            runner = torch.where(won, second, runner)
            taken.index_fill_(0, torch.where(won, target, spare), True)
            pending = (pending & ~asks) if turn == 0 else (asks & ~won)
        # the write's host read: which requests write (the cell update runs on those alone)
        q = (way >= 0).nonzero().flatten()
        rq, cq, picked = r[q], cand[q, way[q]], scores[q, way[q]]  # picked [Q, M]
        unit = picked.argmax(dim=-1)
        alpha = picked.new_ones(q.shape)
        if feed.gate is not None:
            # the straight-through weights of a routed slot: the unit among the units, the cell
            # against the midpoint to the best free candidate it was preferred to
            if router.unit_credit == "executed":
                hard = F.one_hot(unit, M) > 0
                soft_units = torch.softmax(torch.where(hard, picked, picked.detach()) / feed.tau, 1)
            else:
                soft_units = torch.softmax(picked / feed.tau, dim=1)
            chosen = soft_units.gather(1, unit[:, None]).squeeze(1)
            own, rest = rank[q, way[q]], runner[q]
            choice_made = rest > -math.inf
            theta = torch.where(choice_made, 0.5 * (own.detach() + rest), own.detach())
            soft = torch.sigmoid((own - theta) / feed.tau)
            mask = torch.where(choice_made, 1.0 + (soft - soft.detach()), 1.0)
            alpha = (1.0 + (chosen - chosen.detach())) * mask  # exactly 1 forward
        h_old = table0[rq, cq]
        if not self.addressed:  # a cell taken by a new key is cleared
            h_old = h_old.masked_fill((key0[rq, cq] != key[q])[:, None], 0.0)
        # the messages of the keyed cell's fixed in-edges, from the routed cells as its event
        # left them
        event = (feed.slot[rq, column[q]] - 1)[:, None]
        seen = cell.V(feed.routed[rq[:, None], event, self.sources[cq]])  # [Q, D_g, d]
        logit = self.edge_logit[cq]
        edges = torch.sigmoid(logit) if self.edge_norm == "sigmoid" else torch.softmax(logit, 1)
        m = (seen * edges[..., None].to(seen.dtype)).sum(dim=1)
        x = u[q]
        if self.V_s is not None:  # word: the code of the word's bytes behind the key
            x = x + F.linear(win.slot_value[rq, slot[q]].to(dtype), self.V_s).to(dtype)
        v = cell.norm(torch.cat([x, eps[q], h_old, m.to(dtype)], dim=-1))
        out = cell.units.forward_selected(v, unit)
        b = torch.sigmoid(cell.gate(v).to(dtype) + self.beta_K)
        if cell.cand_norm_kind == "none":
            c = torch.tanh(cell.W_out(alpha[:, None] * out).to(dtype))
        else:  # the weight behind the norm, as in the routed round
            c = torch.tanh(alpha[:, None].to(dtype) * cell.cand_norm(cell.W_out(out).to(dtype)))
        table = table0.index_put((rq, cq), torch.lerp(h_old, c, b))
        ran = torch.zeros(M, dtype=torch.bool, device=device).index_fill(0, unit, True)
        none = b.new_zeros(()).float()  # a window without a keyed write
        records = {
            "word_cells_units": ran,
            "word_cells_gate": b.detach().float().mean() if len(q) else none,
        }
        if prior is not None:
            follows = (cq == prior[q].argmax(dim=1)).float()
            records["word_cells_prior"] = follows.mean() if len(q) else none
        return (
            WordCellsCarry(
                word=win.word,
                table=table,
                key=key0 if self.addressed else key0.index_put((rq, cq), key[q]),
                when=when0.index_put((rq, cq), when[q]),
            ),
            records,
        )

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
        how = "address" if self.addressed else f"ways={self.ways}"
        return f"cells={self.n_cells}, {how}, levels={self.levels}, value={self.kind}"
