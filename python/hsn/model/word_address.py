"""The word address of the event loop (``model.word_address``; HSN-T, opt-in candidate;
``docs/amendments/write_events.md``, "The word address").

With ``model.write_events = word`` the memory runs at word ends. This module gives those
events an address that reads the whole word: A5's two features with "byte" replaced by "word".

**Words** (a fixed rule, nothing learned). Under a text context (``model.word_contexts``) a word
byte is an ASCII letter or digit, or an apostrophe directly between two letters; a word is a
maximal run of word bytes, and every other byte is a word of one byte. Under every other
context each byte is its own word. A document start (``reset``) ends whatever came before it.

**Events.** Every position where a word is known to be complete: the byte after a word of
letters (for a word that is followed by an apostrophe and then no letter, the byte after that
apostrophe: one byte late, because the rule may not read ahead), every byte under a context
that is not text, and every document start.

**The key of an event** is the word it closes: the word of letters that ended just before it,
or else the byte at the event itself (a document start, a byte of a context that is not text).
``a_cur`` reads the code of the event's key and ``a_prev`` the code of the key of the row's
event before it (BOS at a document start), so with every byte its own word and an event both
are A5's byte features bit for bit.

**Codes** (fixed tables of ±1 drawn from the run seed after everything A5 and A9 draw;
``f`` are the word's bytes with A-Z folded to a-z, under text contexts only):

- ``exact``: ``d^(−1/2) · s_1[f_1] ⊙ s_2[f_2] ⊙ … ⊙ s_j[f_j]`` with one table per byte position
  (positions past :data:`WORD_TABLES` reuse the last table).
- ``bag``: the unit vector of the sum, over the word's 2- and 3-grams with a start and an end
  mark, of ``t_1[a] ⊙ t_2[b]`` and ``u_1[a] ⊙ u_2[b] ⊙ u_3[c]``.
- A word of one byte keeps A5's byte code ``C[f_1]`` under both.

Both are running sums over the bytes of a word (of sign bits modulo 2, or of gram codes), so
the whole window is one cumulative sum before the loop. What a window hands to the next one
is :func:`state_width` integers per row, carried outside the cell state behind A5's bytes:
the running sum, length and last two bytes of the word in progress, whether the window ended
on an apostrophe that the next byte has yet to place, and the key of the last event.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

from hsn.data.synthetic import BOS

__all__ = ["APOSTROPHE", "WORD_TABLES", "WordAddress", "state_width", "word_address", "word_tables"]

#: ``exact``: the byte positions of a word with a sign table of their own.
WORD_TABLES = 16
APOSTROPHE = 39


class WordAddress(NamedTuple):
    """:func:`word_address` of a window: the ``events`` (bool ``[B, T]``), the codes ``a_cur``
    and ``a_prev`` read at each position (``current``, ``previous``: ``[B, T, d]``, meaningful
    at the events) and the ``state`` for the next window (int64 ``[B, state_width(d)]``)."""

    events: Tensor
    current: Tensor
    previous: Tensor
    state: Tensor


def state_width(d: int) -> int:
    """Integers per row carried from window to window: the running sum of the word in progress
    (``d``), the last event's key as a sum (``d``), then the word's length, its last two folded
    bytes, the pending-apostrophe flag, and the last key's kind (1: the sum) and byte."""
    return 2 * d + 6


def word_tables(kind: str, vocab: int, d: int, generator: torch.Generator) -> dict[str, Tensor]:
    """The fixed tables of ``model.word_address = kind``, drawn from ``generator``: for
    ``exact`` the sign bits ``word_exact [WORD_TABLES, V, d]`` (1: the factor −1), for ``bag``
    the ±1 tables of the 2-grams ``word_bag2 [2, V + 2, d]`` and of the 3-grams ``word_bag3
    [3, V + 2, d]`` (rows ``V`` and ``V + 1``: the start and the end mark)."""

    def bits(*shape: int) -> Tensor:
        return torch.randint(0, 2, shape, generator=generator, dtype=torch.int8)

    if kind == "exact":
        return {"word_exact": bits(WORD_TABLES, vocab, d)}
    if kind == "bag":
        return {
            "word_bag2": bits(2, vocab + 2, d) * 2 - 1,
            "word_bag3": bits(3, vocab + 2, d) * 2 - 1,
        }
    raise ValueError(f"model.word_address={kind!r} has no tables (exact | bag)")


def word_address(
    kind: str,
    tables: dict[str, Tensor],
    byte_codes: Tensor,
    tokens: Tensor,
    text: Tensor,
    reset: Tensor,
    state: Tensor | None = None,
) -> WordAddress:
    """The events and the word codes of a window (module docstring). ``tokens`` (ids), ``text``
    (bool: the position's context is a text context) and ``reset`` are ``[B, T]``;
    ``byte_codes [V, d]`` are A5's; ``state`` is the :class:`WordAddress` state of the window
    before (``None``: nothing carried, the key before the window reads as BOS). No position
    reads a later byte, and nothing here is trained or depends on trained weights."""
    B, T = (int(n) for n in tokens.shape)
    V, d = (int(n) for n in byte_codes.shape)
    device = tokens.device
    start_mark, end_mark = V, V + 1
    x = tokens.to(torch.int64)
    if state is None:
        state = torch.zeros(B, state_width(d), dtype=torch.int64, device=device)
        state[:, -1] = BOS
    run0, key0 = state[:, :d], state[:, d : 2 * d]
    len0, last1, last2, pending0, wide0, byte0 = state[:, 2 * d :].unbind(dim=1)
    held = pending0 > 0  # the window before ended on an apostrophe after a letter
    rows = torch.arange(B, device=device)[:, None]
    steps = torch.arange(T, device=device)

    # the byte classes, and each byte's fold (text contexts only)
    upper = (x >= 65) & (x <= 90)
    letter = text & (upper | ((x >= 97) & (x <= 122)))
    base = letter | (text & (x >= 48) & (x <= 57))
    f = torch.where(text & upper, x + 32, x)
    same = ~reset  # the byte before belongs to the same document

    def after(first: Tensor, row: Tensor) -> Tensor:
        """``row`` of the byte before each position (``first`` before column 0)."""
        return torch.cat([first[:, None], row[:, :-1]], dim=1) & same

    going = (len0 > 0) & ~held  # the byte before the window was a letter or digit of a word
    letter_before = after(going & (last1 >= 97) & (last1 <= 122), letter)
    base_before = after(going, base)
    hangs = text & (x == APOSTROPHE) & letter_before  # an apostrophe directly after a letter
    hung = after(held, hangs)
    joins = hung & letter  # the apostrophe before this letter is a byte of the word
    joined = torch.cat([joins[:, 1:], torch.zeros_like(joins[:, :1])], dim=1)
    word = base | joined
    close_now = base_before & ~base & ~hangs  # the word ended at the byte before
    close_late = hung & ~letter  # it ended before the apostrophe before
    closes = close_now | close_late
    events = reset | ~text | closes

    def contribution(index: Tensor, byte: Tensor, one: Tensor, two: Tensor) -> Tensor:
        """What the ``index``-th byte of a word (from 1) adds to its running sum, int64
        ``[…, d]``; ``one`` and ``two`` are the bytes one and two before it."""
        if kind == "exact":
            table = tables["word_exact"]
            flat = (index.clamp(1, WORD_TABLES) - 1) * V + byte
            return table.reshape(-1, d)[flat].to(torch.int64)
        pairs, triples = tables["word_bag2"], tables["word_bag3"]
        one = torch.where(index >= 2, one, start_mark)
        two = torch.where(index >= 3, two, start_mark)
        pair = pairs[0][one] * pairs[1][byte]
        triple = triples[0][two] * triples[1][one] * triples[2][byte]
        return pair.to(torch.int64) + (triple * (index >= 2)[..., None]).to(torch.int64)

    # a held apostrophe that column 0 makes a word byte: the carried word takes it first
    takes = joins[:, 0]
    quote = torch.full_like(last1, APOSTROPHE)
    run0 = run0 + contribution(len0 + 1, quote, last1, last2) * takes[:, None]
    len0 = len0 + takes
    last2 = torch.where(takes, last1, last2)
    last1 = torch.where(takes, quote, last1)

    # each word byte's place in its word and the running sum of its word up to it
    word_before = torch.cat([((len0 > 0) & (~held | takes))[:, None], word[:, :-1]], dim=1) & same
    first = torch.where(word & ~word_before, steps, -1).cummax(dim=1).values  # the word's start
    carried = first < 0  # the word began in a window before
    index = torch.where(carried, len0[:, None] + steps + 1, steps - first + 1)
    lead = torch.cat([last2[:, None], last1[:, None], f], dim=1)  # [B, T + 2]
    one, two = lead[:, 1 : T + 1], lead[:, :T]
    added = contribution(index, f, one, two) * word[..., None]
    total = torch.cat([torch.zeros_like(added[:, :1]), added.cumsum(dim=1)], dim=1)
    run = total[:, 1:] - total[rows, first.clamp_min(0)] + run0[:, None] * carried[..., None]
    if kind == "exact":
        run, run0 = run & 1, run0 & 1

    # the word as it stands after byte j at index j + 1; index 0: as the window before left it
    run_x = torch.cat([run0[:, None], run], dim=1)
    len_x = torch.cat([len0[:, None], index * word], dim=1)
    last1_x = lead[:, 1:]
    last2_x = torch.cat([last2[:, None], torch.where(index >= 2, one, start_mark)], dim=1)

    # the key of an event: the word it closes, or the byte at the event
    at = torch.where(close_late, (steps - 1).clamp_min(0), steps)
    wide = closes & (len_x[rows, at] >= 2)  # a word of two bytes or more: its own code
    key_byte = torch.where(closes, last1_x[rows, at], f)
    key = run_x[rows, at]
    if kind == "bag":
        pairs, triples = tables["word_bag2"], tables["word_bag3"]
        end1, end2 = last1_x[rows, at], last2_x[rows, at]
        marks = pairs[0][end1] * pairs[1][end_mark]
        marks = marks + triples[0][end2] * triples[1][end1] * triples[2][end_mark]
        key = key + marks.to(torch.int64)
    key = key * wide[..., None]

    def code(total: Tensor, wide: Tensor, byte: Tensor) -> Tensor:
        if kind == "exact":
            vector = (1 - 2 * total).to(byte_codes.dtype) / math.sqrt(d)
        else:
            vector = total.to(byte_codes.dtype)
            vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.where(wide[..., None], vector, byte_codes[byte])

    current = code(key, wide, key_byte)
    last = torch.where(events, steps, -1).cummax(dim=1).values  # the latest event up to t
    before = torch.cat([torch.full_like(last[:, :1], -1), last[:, :-1]], dim=1) + 1
    keys = torch.cat([code(key0, wide0 > 0, byte0)[:, None], current], dim=1)
    previous = torch.where(reset[..., None], byte_codes[BOS], keys[rows, before])

    # what the next window needs: the word in progress and the last event's key
    hanging = hangs[:, -1]
    end = torch.where(hanging, T - 1, T)[:, None]
    alive = hanging | base[:, -1]
    had = last[:, -1:] >= 0
    final = last[:, -1:].clamp_min(0)
    tail = [
        len_x[rows, end][:, 0] * alive,
        last1_x[rows, end][:, 0] * alive,
        last2_x[rows, end][:, 0] * alive,
        hanging,
        torch.where(had, wide[rows, final], wide0[:, None] > 0)[:, 0],
        torch.where(had, key_byte[rows, final], byte0[:, None])[:, 0],
    ]
    out = torch.cat(
        [
            run_x[rows, end][:, 0] * alive[:, None],
            torch.where(had, key[rows, final][:, 0], key0),
            torch.stack([t.to(torch.int64) for t in tail], dim=1),
        ],
        dim=1,
    )
    return WordAddress(events=events, current=current, previous=previous, state=out)
