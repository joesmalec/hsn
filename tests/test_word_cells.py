"""The keyed word cells (``model.word_cells``, ``docs/amendments/word_cells.md``; stage 1: the
by-rule table, the control arm).

The default leaves every configuration and a model's outputs as main (30bb802) has them; the
rule (keys, deferred write, read) equals a per-row reference in plain Python that sees a whole
row at once; a write is readable from the next window on, never from a later byte, an earlier
document or its own window; the carry reaches the callers and a checkpoint; the compiled window
and the fused block's reference equal the eager window; the gradients; the statistics; the
refusals. The CUDA tests train a few bf16 steps of the recipe's CUDA arm with the key.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import test_p2_8_compile as p28
import test_wev_fused_trainer as fused_tests
import torch
from hsn.config import HASH_OMIT_AT_DEFAULT, load_config
from hsn.config import config_hash as chash
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import config_view, digest, memory_key
from hsn.model.hsn import (
    WindowCarry,
    _window_positions,
    address_history,
    clone_carry,
    pack_carry,
    split_packed_carry,
)
from hsn.model.hsn_t import HSNT
from hsn.model.word_cells import EMPTY, NO_KEY, OFFSETS, WORD_CELLS_KEYS, WordCells
from hsn.runner import build_model
from hsn.train.memory_plan import expected_param_count, word_cells_param_count
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import restore_determinism  # noqa: F401
from test_hsn_t import CUT_C, _gap, _model, _reader_cfg
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from test_reader_carry_memory import CARRY, WORD
from test_tbptt_windows import _assert_equal, _trainer
from test_tbptt_windows import _cfg as _trainer_cfg
from test_write_events import NGRAM, STE
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
RECIPE = REPO / "configs" / "experiments" / "hsn_t_memory_words_carry_bal_qk.yaml"
PROTOCOL = {"name": "train_eval", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
TEXT = 6
ON = "model.word_cells=hash"
#: the tests' tiny word-event carry HSN-T (window 12, carry 5), and with 16 keyed cells
BASE = (CARRY, STE, CUT_C, *NGRAM, *WORD)
CELLS = (*BASE, ON, "model.word_cells_n=16")
#: stage 2's arms beside the by-rule table: the overrides behind a configuration with the key
ARM = {
    "bytes": (),
    "word": ("model.word_cells_value=word",),
    "address": ("model.word_cells=address", "model.word_cells_value=cell"),
}
Stream = tuple[Tensor, Tensor, Tensor]


def _ids(text: str) -> list[int]:
    return list(text.encode("latin-1"))


def _prose(B: int, T: int, seed: int, resets: float = 0.01) -> Stream:
    """Rows of words that come back (shared prefixes, capitals, digits, apostrophes inside, a
    word longer than 16 bytes) between separators (apostrophes at word ends and doubled, a byte
    >= 128); document starts anywhere; row 1 partly under another context."""
    g = torch.Generator().manual_seed(seed)
    words = "the them then cat Cat cats don't o'clock x ab abc 42 a1b2 quiet quite it's 7's"
    pool = [*words.split(), "internationalisations"]
    gaps = [" ", " ", ", ", ". ", "' ", "'' ", "'", "\x10", " \xc8 "]
    rows = []
    for _ in range(B):
        row = ""
        while len(row) < T:
            a, b = (int(i) for i in torch.randint(0, 1 << 20, (2,), generator=g))
            row += pool[a % len(pool)] + gaps[b % len(gaps)]
        rows.append(_ids(row[:T]))
    ctx = torch.full((B, T), TEXT)
    reset = torch.rand(B, T, generator=g) < resets
    reset[:, 0] = True
    if B > 1:
        ctx[1, T // 2 : T // 2 + 9] = 3
    return torch.tensor(rows), ctx, reset


def _module(*extra: str) -> WordCells:
    return WordCells(_reader_cfg(*CELLS, *extra).model, seed=3).double()


def _by_windows(mod: WordCells, stream: Stream, cuts: list[int]) -> list[Any]:
    """The module over a stream cut at ``cuts``: every window's record and the carry after it."""
    tokens, ctx, reset = stream
    carry = mod.init_carry(tokens.shape[0], "cpu")
    pos = torch.zeros(tokens.shape[0], dtype=torch.int64)
    out = []
    for a, b in zip([0, *cuts], [*cuts, tokens.shape[1]], strict=True):
        positions = _window_positions(pos, reset[:, a:b])
        win = mod.window(tokens[:, a:b], ctx[:, a:b], reset[:, a:b], positions, carry)
        carry, pos = mod.write(win), positions[:, -1] + 1
        out.append((win, carry))
    return out


# --------------------------------------------------------------------------------------------------
# The rule as written, one row at a time (it sees the whole row: nothing is carried or waited for)
# --------------------------------------------------------------------------------------------------


def _row_words(row: list[int], text: list[bool], reset: list[bool]) -> tuple[list[Any], dict]:
    """Per byte the word so far it is inside (a tuple of bytes; ``None`` outside a word), and
    per column the word that closes there."""
    T = len(row)

    def letter(t: int) -> bool:
        return text[t] and (65 <= row[t] <= 90 or 97 <= row[t] <= 122)

    def base(t: int) -> bool:
        return letter(t) or (text[t] and 48 <= row[t] <= 57)

    def hangs(t: int) -> bool:  # an apostrophe directly after a letter of its document
        return text[t] and row[t] == 39 and t > 0 and not reset[t] and letter(t - 1)

    def joined(t: int) -> bool:
        return hangs(t) and t + 1 < T and not reset[t + 1] and letter(t + 1)

    so_far: list[Any] = [None] * T
    closed: dict[int, tuple[int, ...]] = {}
    t = 0
    while t < T:
        if not base(t):
            t += 1
            continue
        end = t
        while end + 1 < T and not reset[end + 1] and (base(end + 1) or joined(end + 1)):
            end += 1
        for s in range(t, end + 1):
            so_far[s] = tuple(row[t : s + 1])
        after = end + 1
        if after < T and not reset[after]:
            if not hangs(after):
                closed[after] = tuple(row[t:after])
            else:  # the apostrophe did not join: it still counts in the prefix of its byte
                so_far[after] = tuple(row[t : after + 1])
                if after + 1 < T and not reset[after + 1]:
                    closed[after + 1] = tuple(row[t:after])
        t = end + 1
    return so_far, closed


class _RowStore:
    """One row's keyed cells and the rule of ``docs/amendments/word_cells.md``."""

    def __init__(self, mod: WordCells) -> None:
        self.n, self.L, self.min_len, self.d = mod.n_cells, mod.levels, mod.min_len, mod.d
        self.hashes = list(zip(mod.hash_a.tolist(), mod.hash_b.tolist(), strict=True))
        self.P, self.T = mod.place_code.tolist(), mod.byte_code.tolist()
        self.counts = {"held": 0, "second": 0, "dropped": 0, "evicted": 0}
        self.clear()

    def clear(self) -> None:
        self.key: list[int | None] = [None] * self.n
        self.when = [EMPTY] * self.n
        self.value = [[0.0] * self.d for _ in range(self.n)]

    def key_of(self, prefix: tuple[int, ...]) -> int:
        head = prefix[: self.L]
        places = sum(b * 256 ** (self.L - 1 - m) for m, b in enumerate(head))
        return len(head) * 256**self.L + places

    def cells(self, key: int) -> list[int]:
        return [((a * key + b) % ((1 << 31) - 1)) % self.n for a, b in self.hashes]

    def holder(self, key: int) -> int | None:
        return next((c for c in self.cells(key) if self.key[c] == key), None)

    def code(self, word: tuple[int, ...], level: int) -> list[float]:
        total = [0] * self.d
        for m in range(level, len(word)):  # the bytes after the key's
            place = self.P[min(m, 15)]
            total = [s + p * c for s, p, c in zip(total, place, self.T[word[m]], strict=True)]
        return [s / math.sqrt(max(len(word) - level, 1)) for s in total]

    def write(self, events: list[tuple[int, int, tuple[int, ...]]]) -> None:
        """A window's word ends ``(column, byte position, word)``, in order."""
        asks: dict[int, tuple[tuple[int, int], list[float], int]] = {}
        for column, position, word in events:
            levels = min(self.L, len(word)) if len(word) >= self.min_len else 0
            for level in range(1, levels + 1):
                key = self.key_of(word[:level])
                asks[key] = ((column, level), self.code(word, level), position)
        taken = {self.holder(key): key for key in asks if self.holder(key) is not None}
        pending = [key for key in asks if self.holder(key) is None]
        self.counts["held"] += len(taken)
        for turn in range(2):
            want: dict[int, list[int]] = {}
            for key in pending:
                cells = self.cells(key)
                free = [(self.when[c], i, c) for i, c in enumerate(cells) if c not in taken]
                if free:
                    want.setdefault(min(free)[2], []).append(key)
                else:
                    self.counts["dropped"] += 1
            pending = []
            for cell, keys in want.items():
                taken[cell] = max(keys, key=lambda k: asks[k][0])
                pending += [k for k in keys if k != taken[cell]]
                self.counts["second"] += turn
                self.counts["evicted"] += self.key[cell] is not None
        self.counts["dropped"] += len(pending)
        for cell, key in taken.items():
            self.key[cell], (_, self.value[cell], self.when[cell]) = key, asks[key]


def _reference(mod: WordCells, stream: Stream, cuts: list[int]) -> tuple[list[Any], dict]:
    """Per window and row: every byte's read (``None`` outside a word; else its offset and the
    matched cell's value and age, ``()`` without a match) and the cells after the window."""
    tokens, ctx, reset = stream
    T = tokens.shape[1]
    out: list[Any] = [[] for _ in range(len(cuts) + 1)]
    counts: dict[str, int] = {}
    for b in range(tokens.shape[0]):
        row, flags = tokens[b].tolist(), reset[b].tolist()
        so_far, closed = _row_words(row, (ctx[b] == TEXT).tolist(), flags)
        position, pos = [], 0
        for t in range(T):
            pos = 0 if flags[t] else pos
            position.append(pos)
            pos += 1
        store = _RowStore(mod)
        for w, (lo, hi) in enumerate(zip([0, *cuts], [*cuts, T], strict=True)):
            reads: list[Any] = []
            for t in range(lo, hi):
                word = so_far[t]
                if word is None:
                    reads.append(None)
                    continue
                cell = None if any(flags[lo : t + 1]) else store.holder(store.key_of(word))
                hit = () if cell is None else (store.value[cell], position[t] - store.when[cell])
                reads.append((min(len(word), OFFSETS) - 1, hit))
            starts = [t for t in range(lo, hi) if flags[t]]
            if starts:
                store.clear()
            first = starts[-1] if starts else lo - 1
            ends = [(t, position[t], closed[t]) for t in range(lo, hi) if t in closed and t > first]
            store.write(ends)
            out[w].append((reads, list(store.key), list(store.when), list(store.value)))
        counts = {k: counts.get(k, 0) + v for k, v in store.counts.items()}
    return out, counts


SMALL = ("model.word_cells_n=8", "model.word_cells_ways=1", "model.word_cells_levels=2")
THREE = ("model.word_cells_n=24", "model.word_cells_ways=3", "model.word_cells_min_len=3")


@pytest.mark.parametrize(
    ("extra", "window", "branches"),
    [
        ((), 16, "held second dropped evicted"),
        ((), 1, "held evicted"),
        (SMALL, 16, "held dropped evicted"),
        (THREE, 11, "held second evicted"),
        (("model.word_cells_n=1024",), 32, "held"),
    ],
)
def test_the_rule_against_a_per_row_reference(extra: tuple, window: int, branches: str) -> None:
    """Keys, deferred write and read over three rows of several documents each, window by
    window (also one byte a window), equal the per-row reference exactly: the cells, their keys
    and write positions after every window, and every byte's match, value, offset and age. The
    streams make the write's branches happen (a key that holds its cell, a second ask, a key
    not written, a cell taken from another key)."""
    mod = _module(*extra)
    mod.sight = 24
    stream = _prose(3, 400, 5)
    cuts = list(range(window, 400, window))
    reference, counts = _reference(mod, stream, cuts)
    assert counts["held"] > 20 and all(counts[k] > 0 for k in branches.split()), counts
    hits = old = 0
    for (win, carry), rows in zip(_by_windows(mod, stream, cuts), reference, strict=True):
        for b, (reads, keys, when, values) in enumerate(rows):
            assert carry.key[b].tolist() == [NO_KEY if k is None else k for k in keys]
            assert carry.when[b].tolist() == when
            assert torch.equal(carry.table[b], torch.tensor(values, dtype=torch.float64))
            for t, read in enumerate(reads):
                assert bool(win.inside[b, t]) == (read is not None), (b, t)
                if read is None:
                    continue
                offset, hit = read
                assert int(win.offset[b, t]) == offset and bool(win.hit[b, t]) == bool(hit)
                value, age = hit if hit else ([0.0] * mod.d, 0)
                assert win.value[b, t].tolist() == value, (b, t)
                want = [float(age >= 24), math.log1p(age / 24)] if hit else [0.0, 0.0]
                assert win.age_features[b, t].tolist() == pytest.approx(want, abs=1e-15)
                assert bool(win.old[b, t]) == (bool(hit) and age >= 24)
                hits, old = hits + bool(hit), old + (bool(hit) and age >= 24)
    assert hits > 100 and old > 10, (hits, old)


def test_keys_and_word_ends_do_not_depend_on_the_cuts() -> None:
    """Whatever the cuts (down to one byte a window): the same bytes inside a word, offsets,
    word ends with their keys, values and positions, and the same word in progress at the
    end. (What a byte reads does depend on them, through where its own window starts: a write
    is readable from the next window on; the reference above takes the cuts as they are.)"""
    mod = _module()
    stream = _prose(3, 240, 8, resets=0.0)
    ((whole, end),) = _by_windows(mod, stream, [])
    g = torch.Generator().manual_seed(2)
    for trial in range(4):
        cuts = sorted({int(c) for c in torch.randint(1, 240, (3 + 9 * trial,), generator=g)})
        parts = _by_windows(mod, stream, list(range(1, 240)) if trial == 3 else cuts)
        for name in ("inside", "offset", "slot_active", "slot_key", "slot_when", "slot_value"):
            got = torch.cat([getattr(win, name) for win, _ in parts], dim=1)
            live = whole.slot_active if name.startswith("slot_") else whole.inside
            assert torch.equal(got[live], getattr(whole, name)[live]), (name, cuts)
            assert name not in ("inside", "slot_active") or torch.equal(got, live)
        assert torch.equal(parts[-1][1].word, end.word)


def test_a_write_is_readable_from_the_next_window_on() -> None:
    """Never in its own window, never from a later byte, never from an earlier document."""
    mod = _module("model.word_cells_n=256")
    rows = ["alpha beta  ", "gamma gamma ", "alpha gamma ", "alpha gamma "]
    tokens = torch.tensor([_ids("".join(rows))])
    ctx = torch.full_like(tokens, TEXT)
    reset = torch.zeros_like(tokens, dtype=torch.bool)
    reset[0, 0] = reset[0, 36] = True  # the fourth window is another document
    parts = _by_windows(mod, (tokens, ctx, reset), [12, 24, 36])
    hits = ["".join("x" if h else "." for h in win.hit[0].tolist()) for win, _ in parts]
    assert hits == ["............", "............", "xxxxx.xxxxx.", "............"]
    assert parts[2][0].age_features[0, 0, 1] == pytest.approx(math.log1p(19 / 256))  # 24 - 5
    later = tokens.clone()
    later[0, 30:] = torch.tensor(_ids("gamma alpha gamma "))
    again = _by_windows(mod, (later, ctx, reset), [12, 24, 36])
    for name in ("hit", "value", "offset", "age_features"):
        assert torch.equal(getattr(again[2][0], name)[:, :6], getattr(parts[2][0], name)[:, :6])


# --------------------------------------------------------------------------------------------------
# The default is main's; the configuration
# --------------------------------------------------------------------------------------------------


def test_the_default_keeps_every_configuration_of_main() -> None:
    """Every configuration file's hash, arm view and memory key are those main (30bb802) gives
    (recorded by its code); the keys are out of all three at their defaults and in them on."""
    recorded = json.loads((FIXTURES / "word_cells_ids_30bb802.json").read_text())
    assert len(recorded) == 23
    for name, want in recorded.items():
        cfg = load_config(REPO / name)
        got = [chash(cfg), digest(config_view(cfg)), memory_key(cfg, "none", PROTOCOL)]
        assert got == want, name
    for folder in ("experiments", "baselines"):  # no configuration file is left out
        names = {f"configs/{folder}/{p.name}" for p in (REPO / "configs" / folder).glob("*.yaml")}
        assert names <= set(recorded)
    keys = [k for k in HASH_OMIT_AT_DEFAULT if k.startswith("model.word_cells")]
    assert len(keys) == 8 and all(k.split(".")[1] in type(cfg.model).model_fields for k in keys)
    base, named = load_config(RECIPE), load_config(RECIPE, ["model.word_cells=off"])
    on = load_config(RECIPE, [ON])
    assert base.model.word_cells == "off" and chash(named) == chash(base) != chash(on)
    assert not any("word_cells" in k for k in config_view(base)["model"])
    assert config_view(on)["model"] == {**config_view(base)["model"], "word_cells": "hash"}
    assert memory_key(on, "none", PROTOCOL) != memory_key(base, "none", PROTOCOL)
    assert type(base.model).model_validate({"word_cells": False}).word_cells == "off"  # YAML's


def test_the_default_model_is_mains_bit_for_bit() -> None:
    """The tests' word-event carry HSN-T without the key against main's code (the fixture,
    written at 30bb802 in float64 on x86-64 Linux): the same parameter and buffer names and
    drawn weights, and on those weights, over three carried windows, main's logits, final
    state, carry and gradients (each as its sum and its sum of magnitudes). Bit for bit where
    the fixture was written; the tolerances cover another machine's rounding."""
    want = torch.load(FIXTURES / "word_cells_off_30bb802.pt", weights_only=True)
    tokens, ctx, reset = want["stream"]
    for extra in ((), ("model.word_cells=off",)):
        cfg = _reader_cfg(*BASE, *extra)
        model = _model(cfg)
        drawn = model.state_dict()
        assert list(drawn) == list(want["weights"]) and model.word_cells is None
        for name, value in want["weights"].items():
            if isinstance(value, Tensor):  # (float32 draws differ in the last bits by machine)
                assert _gap(drawn[name].double(), value.double()) <= 1e-5, name
        model.load_state_dict(want["weights"])
        T = cfg.train.unroll
        state, prev = model.init_state(3), None
        for lo in range(0, tokens.shape[1], T):
            window = (tokens[:, lo : lo + T], ctx[:, lo : lo + T], state, reset[:, lo : lo + T])
            logits, state, info = model(*window, prev_tokens=prev)
            assert not any("word_cells" in key for key in info)
            prev = address_history(cfg.model, window[0], prev, info)
            state = state.detach()
        logits.square().mean().backward()
        assert isinstance(prev, WindowCarry) and prev.cells is None
        assert set(pack_carry(prev)) == {"address", "reader"}
        found = [logits, state.h, state.q, *prev.reader[:2]]
        refs = [want["logits"], *want["final"][:2], *want["reader"][:2]]
        for got, ref in zip(found, refs, strict=True):
            assert got.shape == ref.shape and _gap(got, ref) <= 1e-12
        exact = (state.age, state.pos, prev.address, prev.reader[2])
        _assert_equal(exact, (*want["final"][2:], want["address"], want["reader"][2]), "integers")
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        assert list(grads) == list(want["grads"])
        for name, ref in want["grads"].items():
            got = torch.stack([grads[name].sum(), grads[name].abs().sum()])
            assert _gap(got, ref) <= 1e-12 * max(1.0, float(ref[1])), name


def test_the_bytes_arm_is_stage_ones_bit_for_bit() -> None:
    """``model.word_cells=hash`` with the by-rule value (``bytes``, named or not) against stage
    1's code (the fixture, written at d3537f4 in float64 on x86-64 Linux): the same parameter
    and buffer names, and on those weights (main's fixture and the key's own), over four carried
    windows of prose, its logits, statistics, final state, cells, address carry and gradients
    (each as its sum and its sum of magnitudes); and the recipe's hash, arm view and memory
    key with the key on. Bit for bit where the fixture was written."""
    want = torch.load(FIXTURES / "word_cells_bytes_d3537f4.pt", weights_only=True)
    base = torch.load(FIXTURES / "word_cells_off_30bb802.pt", weights_only=True)["weights"]
    tokens, ctx, reset = want["stream"]
    for extra in ((), ("model.word_cells_value=bytes",)):
        cfg = _reader_cfg(*CELLS, *extra)
        model = _model(cfg)
        assert list(model.state_dict()) == want["names"]
        model.load_state_dict(
            {name: want["weights"].get(name, base.get(name)) for name in want["names"]}
        )
        T = cfg.train.unroll
        state, prev, sums = model.init_state(3), None, []
        for lo in range(0, tokens.shape[1], T):
            window = (tokens[:, lo : lo + T], ctx[:, lo : lo + T], state, reset[:, lo : lo + T])
            logits, state, info = model(*window, prev_tokens=prev)
            prev = address_history(cfg.model, window[0], prev, info)
            state = state.detach()
            rates = (info["word_cells_hit"].double(), info["word_cells_old"].double())
            sums.append(torch.stack([logits.detach().sum(), logits.detach().abs().sum(), *rates]))
        logits.square().mean().backward()
        assert isinstance(prev, WindowCarry) and prev.cells is not None
        found = [logits, torch.stack(sums), state.h, state.q, prev.cells.table]
        refs = [want["logits"], want["sums"], *want["final"][:2], want["cells"][1]]
        for got, ref in zip(found, refs, strict=True):
            assert got.shape == ref.shape and _gap(got, ref) <= 1e-12
        exact = (state.age, state.pos, prev.address, *prev.cells[0::2], prev.cells.when)
        refs = (*want["final"][2:], want["address"], *want["cells"][0::2], want["cells"][3])
        _assert_equal(exact, refs, "integers")
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        assert list(grads) == list(want["grads"])
        for name, ref in want["grads"].items():
            got = torch.stack([grads[name].sum(), grads[name].abs().sum()])
            assert _gap(got, ref) <= 1e-12 * max(1.0, float(ref[1])), name
    arms = {"on": [ON], "named": [ON, "model.word_cells_value=bytes"]}
    arms["sized"] = [ON, "model.word_cells_n=256", "model.word_cells_ways=1"]
    for name, overrides in arms.items():
        cfg = load_config(RECIPE, overrides)
        got = [chash(cfg), digest(config_view(cfg)), memory_key(cfg, "none", PROTOCOL)]
        assert got == want["ids"][name], name


def test_the_key_adds_the_read_maps_and_the_decoder_columns_only() -> None:
    cfg, on = _reader_cfg(*BASE), _reader_cfg(*CELLS)
    seed_everything(0)
    plain = HSNT.from_config(cfg)
    seed_everything(0)
    model = HSNT.from_config(on)
    d, width = cfg.model.d, plain.decoder.fc1.weight.shape[1]
    assert model.word_cells is not None and not bool(model.word_cells.U.any())
    for name, p in plain.named_parameters():
        q = model.get_parameter(name)
        assert torch.equal(p, q[:, :width] if name == "decoder.fc1.weight" else q), name
    assert bool(model.decoder.fc1.weight[:, width:].abs().sum() > 0)
    new = {n for n, _ in model.named_parameters()} - {n for n, _ in plain.named_parameters()}
    assert new == {"word_cells.U"} and model.decoder.fc1.weight.shape[1] == width + d + 2
    report = model.param_report()
    assert report["word_cells"] == word_cells_param_count(on.model) == OFFSETS * d * d
    assert report["decoder"] == expected_param_count(on.model)["decoder"]
    full, base = load_config(RECIPE, [ON]).model, load_config(RECIPE).model
    grown = expected_param_count(full)["decoder"] - expected_param_count(base)["decoder"]
    assert (word_cells_param_count(full), grown) == (65_536, 66 * 512)


#: what the key refuses: a configuration file ("": the recipe), overrides, a word of the reason
REFUSED: list[tuple[str, list[str], str]] = [
    ("", ["model.memory=false"], "model.memory=false has none"),
    ("", ["model.write_events=byte", "model.word_address=none"], "needs model.write_events=word"),
    ("", ["model.word_address=none"], "needs model.write_events=word"),
    ("", ["model.reader_carry=0"], "needs model.reader_carry > 0"),
    ("", ["model.cell_attention_heads=4"], "not built beside the keyed cells"),
    ("", ["model.word_contexts=[]"], "word_contexts is empty"),
    ("", ["model.word_cells_ways=3", "model.word_cells_n=2"], "candidate cells per key"),
    ("", ["model.word_cells=off", "model.word_cells_n=64"], "model.word_cells is off"),
    ("", ["model.word_cells_value=sum"], "word_cells_value"),
    ("", ["model.word_cells=address"], "word_cells_value=word"),
    ("", ["model.word_cells_levels=4"], "word_cells_levels"),
    # the baselines, the reader alone, the HSN, HSN-T without its memory and with per-byte writes
    ("baselines/gru_2m.yaml", [], "word_cells"),
    ("baselines/transformer_2m.yaml", [], "word_cells"),
    ("baselines/reader_w64l2.yaml", [], "word_cells"),
    ("experiments/hsn_recipe_a6.yaml", [], "word_cells"),
    ("experiments/hsn_t_nomem.yaml", [], "word_cells"),
    ("experiments/hsn_t_memory.yaml", [], "word_cells"),
]


@pytest.mark.parametrize(("name", "overrides", "message"), REFUSED)
def test_refusals(name: str, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(REPO / "configs" / name if name else RECIPE, [ON, *overrides])


# --------------------------------------------------------------------------------------------------
# The model: the carry, what a byte may read, the gradients, the compiled window
# --------------------------------------------------------------------------------------------------


def _read(model: HSNT, stream: Stream, T: int) -> tuple[Tensor, list[Any]]:
    """The stream read window by window as a caller reads it: the logits and every carry."""
    tokens, ctx, reset = stream
    state, prev = model.init_state(tokens.shape[0]), None
    logits, carried = [], []
    for lo in range(0, tokens.shape[1], T):
        sl = slice(lo, lo + T)
        out, state, info = model(tokens[:, sl], ctx[:, sl], state, reset[:, sl], prev_tokens=prev)
        prev = address_history(model.cfg, tokens[:, sl], prev, info)
        logits.append(out)
        carried.append(prev)
    return torch.cat(logits, dim=1), carried


def _cells_model(read_maps: bool = True) -> HSNT:
    """The tests' model with the key; ``read_maps``: ``U`` away from its zero start (the
    cells' values reach the logits), else at it."""
    model = _model(_reader_cfg(*CELLS))
    assert model.word_cells is not None
    with torch.no_grad():
        model.word_cells.U.normal_(0.0, 0.5) if read_maps else model.word_cells.U.zero_()
    return model


def test_the_model_reads_the_cells_and_the_carry_must_be_carried() -> None:
    """The model's cells are the module's on the same bytes; the logits read them (with empty
    cells they differ); a carry without the cells, cells of another shape and a missing
    ``info`` are refused, so no caller (the evaluation's loop among them) can drop the cells
    unnoticed; the copies and the checkpoint form hold them."""
    model = _cells_model().eval()
    cfg, cells = model.cfg, model.word_cells
    assert cells is not None
    T, W = 12, 8
    stream = _prose(3, W * T, 4)
    with torch.no_grad():
        _, carried = _read(model, stream, T)
        parts = _by_windows(cells, stream, list(range(T, W * T, T)))
        for carry, (_, want) in zip(carried, parts, strict=True):
            assert isinstance(carry, WindowCarry) and carry.cells is not None
            _assert_equal(tuple(carry.cells), tuple(want), "the cells")
        assert sum(int(win.hit.sum()) for win, _ in parts) > 10
        last = carried[W - 2]
        args = (*(x[:, -T:] for x in stream[:2]), model.init_state(3), stream[2][:, -T:])
        full, _, info = model(*args, prev_tokens=last)
        assert set(WORD_CELLS_KEYS) <= set(info) and not info["word_cells_table"].requires_grad
        blind, _, _ = model(*args, prev_tokens=last._replace(cells=cells.init_carry(3, "cpu")))
        assert _gap(full, blind) > 1e-3
        with pytest.raises(ValueError, match="with its keyed word cells"):
            model(*args, prev_tokens=WindowCarry(last.address, last.reader))
        short = last.cells._replace(key=last.cells.key[:, :-1])
        with pytest.raises(ValueError, match="carried cells must be"):
            model(*args, prev_tokens=last._replace(cells=short))
        partial = {k: v for k, v in info.items() if "table" not in k}
        with pytest.raises(ValueError, match="pass the info"):
            address_history(cfg, args[0], last, partial)
        plain = _model(_reader_cfg(*BASE)).eval()
        with pytest.raises(ValueError, match="the model has none"):
            plain(*args, prev_tokens=last)
    clone = clone_carry(last)
    assert isinstance(clone, WindowCarry) and clone.cells.table is not last.cells.table
    _assert_equal(tuple(clone.cells), tuple(last.cells), "clone")
    packed = pack_carry(last)
    assert set(packed) == {"address", "reader", "word_cells"} and len(packed["word_cells"]) == 4
    with pytest.raises(ValueError, match="must also hold the keyed word cells"):
        split_packed_carry(cfg, {k: packed[k] for k in ("address", "reader")}, "ckpt")


def test_the_model_reads_no_later_byte_and_no_earlier_document() -> None:
    model = _cells_model().eval()
    T = 12
    tokens, ctx, reset = _prose(3, 4 * T, 6, resets=0.0)
    with torch.no_grad():
        _, carried = _read(model, (tokens[:, : 3 * T], ctx[:, : 3 * T], reset[:, : 3 * T]), T)
        _, others = _read(model, _prose(3, 3 * T, 9), T)
        window = (tokens[:, 3 * T :], ctx[:, 3 * T :])
        state, t0 = model.init_state(3), 7
        flags = torch.zeros(3, T, dtype=torch.bool)
        logits, _, info = model(*window, state, flags, prev_tokens=carried[-1])
        assert float(info["word_cells_hit"]) > 0.2
        later = window[0].clone()
        later[:, t0:] = torch.tensor(_ids("then "))
        changed, _, _ = model(later, window[1], state, flags, prev_tokens=carried[-1])
        assert _gap(changed[:, :t0], logits[:, :t0]) <= 1e-12
        assert _gap(changed[:, t0:], logits[:, t0:]) > 1e-6
        # a document start in row 0: from there on nothing of the cells, the address or the
        # reader's cache before it is read; before it, and in the other rows, they show
        flags[0, 4] = True
        ours, _, _ = model(*window, state, flags, prev_tokens=carried[-1])
        theirs, _, _ = model(*window, state, flags, prev_tokens=others[-1])
        assert _gap(ours[0, 4:], theirs[0, 4:]) <= 1e-12 < _gap(ours[0, :4], theirs[0, :4])
        assert _gap(ours[1:], theirs[1:]) > 1e-6


def test_gradients_reach_the_read_maps_and_the_decoder_columns_only() -> None:
    """No gradient enters the cells (they are by rule and carried detached). At the start
    (``U`` zero) the gradient reaches ``U``, at the offsets some byte read a value with and no
    other, and the decoder's two age columns; its ``d`` value columns get theirs once ``U``
    moved."""
    T, stream = 12, _prose(3, 48, 4, resets=0.0)
    for moved in (False, True):
        model = _cells_model(read_maps=moved)
        cells, d = model.word_cells, model.cfg.d
        assert cells is not None
        cells.sight = 8  # some entries are old in a stream this short
        read = torch.zeros(OFFSETS, dtype=torch.bool)
        parts = _by_windows(cells, stream, [T, 2 * T, 3 * T])
        for win, _ in parts:  # (the entry of a word no longer than its key is a zero value)
            read[win.offset[win.hit & win.value.any(dim=-1)]] = True
        assert bool(read[:3].all()) and not bool(read.all())
        assert any(bool(win.old.any()) for win, _ in parts)
        logits, carried = _read(model, stream, T)
        kept = carried[-1].cells
        assert not any(t.requires_grad for t in kept) and all(t.grad_fn is None for t in kept)
        logits.square().mean().backward()
        columns = model.decoder.fc1.weight.grad[:, -(d + 2) :]
        assert torch.equal(cells.U.grad.abs().sum(dim=(1, 2)) > 0, read)
        assert bool((columns[:, d:].abs().sum(dim=0) > 0).all())
        assert bool(columns[:, :d].abs().sum() > 0) == moved


@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("arm", list(ARM))
def test_compiled_window_equals_eager_and_compiles_once(
    arm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The keys, the read and the write run in the eager prologue and epilogue, so the key
    changes no graph: over six consecutive windows the compiled window equals the eager one
    (values, the next cells and their tags exactly, the gradients, ``U``'s among them),
    compiles one graph and keeps it. Stage 2's arms: the compiled blocks hand on the cells
    of every slot, and the HSN's write on them equals the eager window's."""
    from torch._dynamo.utils import counters

    monkeypatch.setattr("hsn.model.hsn.EVENT_BLOCK", p28.compile_mod.MAX_BLOCK)  # small_blocks
    T = 20
    reader = ("model_name=hsn_t", "model.reader=transformer", "model.reader_width=8")
    routed = ("model.routing=topk", "model.key_address=true", "model.selected_readout=true")
    gate = ("model.address_gate=true", "model.address_gate_input=summary")
    cfg = p28._cfg(*routed, *reader, "model.reader_heads=2", *gate, *CELLS, *ARM[arm])
    seed_everything(0)
    model = HSNT.from_config(cfg)
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.3)
    model.train(True)
    window = p28._window(model, p28._ctx())
    tol = p28.TOL[("cpu", "fp32")]
    state = p28._inputs(model, b=3, t=T)[2]
    tokens, ctx, _ = _prose(3, 6 * T, 12, resets=0.0)
    flags = torch.zeros(3, T, dtype=torch.bool)
    carry, graphs, hits = None, -1, 0.0
    index = [n for n, p in model.named_parameters() if p.requires_grad].index("word_cells.U")
    for i in range(6):
        args = (tokens[:, i * T : (i + 1) * T], ctx[:, i * T : (i + 1) * T], state, flags)
        ref = p28._run(functools.partial(model, *args, prev_tokens=carry))
        with torch.compiler.set_stance("fail_on_recompile" if i else "default"):
            got = p28._run(functools.partial(window, *args, prev_tokens=carry))
        graphs = counters["stats"]["unique_graphs"] if i == 0 else graphs
        diff = p28._max_diff(got, ref)
        assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], (i, diff)
        grads, wanted = p28._grads(model, got), p28._grads(model, ref)
        assert p28._grad_diff(grads, wanted) <= tol["grads"], i
        hits += float(ref[2]["word_cells_hit"])
        assert i == 0 or float(wanted[index].abs().sum()) > 0
        carry = address_history(cfg.model, args[0], carry, ref[2])
        state = got[1].detach()
    assert counters["stats"]["unique_graphs"] == graphs and window.calls == 6 and hits > 1.0


# --------------------------------------------------------------------------------------------------
# The trainer
# --------------------------------------------------------------------------------------------------


def _text_datasets(n: int = 24, length: int = 150) -> tuple[ByteDataset, ByteDataset]:
    """Documents of words that come back, under the text context, longer than four windows."""

    def docs(split: str, count: int, seed: int) -> ByteDataset:
        examples = []
        for i, row in enumerate(_prose(count, length, seed)[0].numpy().astype(np.uint8)):
            doc = {"task": "kv", "variant": "in_range", "family": "kv", "split": split, "index": i}
            lengths = {"dep_len": 1, "in_range": True, "deps": (1,) * (length - 1)}
            meta = {**doc, "doc_id": f"kv:in_range:{split}:{i}", **lengths}
            mask = np.arange(length) < length - 1
            context = np.full(length, TEXT, dtype=np.uint8)
            examples.append(Example(row, np.roll(row, -1), mask, context, meta))
        return ByteDataset(examples)

    return docs("train", n, 21), docs("dev", 8, 22)


reference_block = fused_tests.reference_block  # the fused block on its PyTorch reference
#: the trainer tests' tiny HSN-T with word events (lazy AdamW, balance), the carry, 32 cells
KEYS = ("model.word_contexts=[0,6]", "model.reader_carry=8", ON, "model.word_cells_n=32")
TRAINED = (*fused_tests.EVENTS, *KEYS)
#: (windows a gradient segment, arm): stage 2's arms with the cells attached inside a segment
SEGMENTS = [(1, "bytes"), (2, "bytes"), (1, "word"), (2, "word"), (2, "address")]


@pytest.mark.parametrize(("segments", "arm"), SEGMENTS)
def test_training_carries_logs_and_resumes_bit_for_bit(
    segments: int, arm: str, tmp_path: Path
) -> None:
    """Four steps of the tiny recipe on text: the statistics hold the read's hit rate and the
    share of old entries; ``U`` trains; the checkpoint holds the cells on the CPU; and a run
    resumed from step 2 (other initial weights) continues bit for bit, the cells among what
    it carries (also with two windows a gradient segment, and with the HSN's write)."""
    accumulate = (f"train.grad_accum={2 * segments}", f"train.tbptt_windows={segments}")
    keys = (*TRAINED, *ARM[arm], *accumulate)
    cfg = _trainer_cfg(4, *keys, "train.ckpt_every=2", "train.stats_every=1")
    train_ds, dev_ds = _text_datasets()
    trainer_a = _trainer(cfg, tmp_path / "a")
    start = trainer_a.model.word_cells.U.detach().clone()
    history_a = trainer_a.fit(train_ds, dev_ds, 4)
    assert all(math.isfinite(s.loss) for s in history_a.steps)
    stats = history_a.steps[-1].stats
    assert 0.2 < stats["word_cells_hit"] <= 1.0 and 0.0 <= stats["word_cells_old"] <= 1.0
    assert "grad_norm/word_cells" in stats
    assert not torch.equal(trainer_a.model.word_cells.U, start)
    checkpoint = tmp_path / "a" / "step_00000002.pt"
    blob = torch.load(checkpoint, weights_only=True)
    payload = blob["extra"]["prev_tokens"]
    assert set(payload) == {"address", "reader", "word_cells"}
    _, table, key, when = payload["word_cells"]
    assert table.device.type == "cpu" and table.shape == (cfg.train.batch, 32, cfg.model.d)
    assert bool((key >= 0).any()) == (arm != "address") and bool((when[key >= 0] >= 0).all())
    assert bool((when > EMPTY).any()) and float(table.abs().max()) > 0
    trainer_b = _trainer(cfg, tmp_path / "b", seed=1)
    assert trainer_b.resume(checkpoint) == 2
    history_b = trainer_b.fit(train_ds, dev_ds, 4)
    for a, b in zip(history_a.steps[2:], history_b.steps, strict=True):
        assert (a.step, a.loss, a.total, a.parts) == (b.step, b.loss, b.total, b.parts)
        assert (a.grad_norm, a.stats["word_cells_hit"]) == (b.grad_norm, b.stats["word_cells_hit"])
    _assert_equal(trainer_a.model.state_dict(), trainer_b.model.state_dict(), "weights")
    carry_a, carry_b = trainer_a._prev_tokens, trainer_b._prev_tokens
    assert isinstance(carry_a, WindowCarry) and carry_a.cells is not None
    _assert_equal(tuple(carry_a), tuple(carry_b), "carry")
    del payload["word_cells"]  # a checkpoint without the cells is refused by name
    torch.save(blob, tmp_path / "without.pt")
    with pytest.raises(ValueError, match="written without the key"):
        _trainer(cfg, tmp_path / "c").resume(tmp_path / "without.pt")


@pytest.mark.usefixtures("reference_block")
@pytest.mark.parametrize(("segments", "arm"), [SEGMENTS[0], *SEGMENTS[2:]])
def test_the_fused_block_reference_equals_the_eager_trainer(
    segments: int, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``hardware.event_kernel=fused`` beside the key (the block's PyTorch reference in
    float64, as the fused trainer tests run it on the CPU): four optimiser steps equal the
    eager trainer's, the read's statistics included. The by-rule arm does not touch the
    block. Stage 2's arms read the routed cells of every slot from the block's tape; their
    write then equals the eager loop's, and so do the gradients it gets through the cells
    inside a segment of two windows (the write's own parameters among them)."""
    monkeypatch.setattr(fused_tests, "_long_datasets", _text_datasets)
    accumulate = (f"train.grad_accum={segments}", f"train.tbptt_windows={segments}")
    runs = {}
    for mode in ("none", "default"):
        keys = (*KEYS, *ARM[arm], *accumulate, *fused_tests.FP64)
        cfg = fused_tests._small(*keys, f"hardware.compile_mode={mode}")
        runs[mode] = fused_tests._fit(cfg, tmp_path / mode, 4)
    got, ref = runs["default"], runs["none"]
    fused = got["trainer"]._fused_events
    assert fused is not None and fused.calls == 4 * segments
    assert ref["trainer"]._fused_events is None
    fused_tests._assert_same_run(got, ref, loss=1e-9, grad=1e-8, param=1e-9, stats=1e-8)
    assert got["history"].steps[-1].stats["word_cells_hit"] > 0.2
    names = [n for n, _ in got["trainer"].model.named_parameters()]
    own = [i for i, n in enumerate(names) if n in ("word_cells.beta_K", "word_cells.edge_logit")]
    grads = got["grads"][-1]
    assert len(own) == 2 * (arm != "bytes")
    assert all((grads[i] is not None and bool(grads[i].any())) == (segments > 1) for i in own)


# --------------------------------------------------------------------------------------------------
# On the card
# --------------------------------------------------------------------------------------------------

#: the recipe's CUDA arm (its header) and a few bf16 steps of it
CARD_ARM = (
    "hardware.compile_mode=reduce-overhead",
    "hardware.graph_alias_index=true",
    "hardware.message_gradients=state",
    "hardware.round_products=merged",
)
CARD_STEPS = ("train.steps=4", "train.batch=8", "train.micro_batch=8", "train.warmup_steps=0")
QUIET = ("train.eval_every=1000", "train.ckpt_every=1000", "train.stats_every=1")


#: (the event loop, the arm, windows a gradient segment) of the card test below
ON_THE_CARD = [(kernel, "bytes", 1) for kernel in ("compiled", "fused")]
ON_THE_CARD += [(kernel, "word", n) for n in (1, 3) for kernel in ("compiled", "fused")]
ON_THE_CARD += [("fused", "address", 3)]


@pytest.mark.cuda
@pytest.mark.parametrize(("kernel", "arm", "segments"), ON_THE_CARD)
def test_bf16_steps_of_the_recipe_with_the_key_on_the_card(
    kernel: str, arm: str, segments: int, tmp_path: Path
) -> None:
    """hsn_t_memory_words_carry_bal_qk.yaml's CUDA arm with ``model.word_cells=hash``, on the
    compiled event blocks and on the fused event block: four bf16 steps under CUDA graphs on
    text; finite losses, ``U`` trained, a hit rate in the statistics, and the cells on the
    card (float32) equal to the rule run on the CPU over the same windows (keys, write
    positions and the word in progress exactly, the values to float32 rounding).

    Stage 2's arms (``word``; ``address`` with the memo's 256 cells): the same steps, and with
    three windows a gradient segment (``hardware.compile_mode=default``, as a segment needs)
    the write's own parameters train; with every window detached they stay. The cells are
    the HSN's, in ``[-1, 1]``; the word in progress is still the rule's."""
    from cards import CUDA_PROFILE
    from hsn.kernels import window as window_kernels

    p28._cuda_ctx()
    if kernel == "fused" and not window_kernels.available():
        pytest.skip("needs the window kernels (scripts/build_window_kernels.py)")
    card = [f"profile={CUDA_PROFILE}", *CARD_ARM, f"hardware.event_kernel={kernel}"]
    card += ["model.word_cells_n=256"] * (arm == "address")
    if segments > 1:
        card += ["hardware.compile_mode=default", f"train.grad_accum={segments}"]
        card += [f"train.tbptt_windows={segments}"]
    cfg = load_config(RECIPE, [ON, *ARM[arm], *card, *CARD_STEPS, *QUIET])
    seed_everything(0)
    model = build_model(cfg)
    trainer = TBPTTTrainer(model, cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    seen = fused_tests._watch(trainer)  # the training windows, not a statistics step's passes
    windows: list[dict[str, Tensor]] = []
    forward = trainer._forward

    def spy(batch: dict[str, Tensor], state: Any, **kwargs: Any) -> Any:
        before = len(seen)
        out = forward(batch, state, **kwargs)
        if len(seen) > before:
            windows.append({k: batch[k].cpu() for k in ("tokens", "context", "reset")})
        return out

    trainer._forward = spy  # type: ignore[method-assign]
    start = {n: p.detach().clone() for n, p in model.word_cells.named_parameters()}
    history = trainer.fit(*_text_datasets(64, 900), cfg.train.steps)
    assert len(history.steps) == 4 and all(math.isfinite(s.loss) for s in history.steps)
    assert history.steps[-1].stats["word_cells_hit"] > 0.3
    assert bool(model.word_cells.U.detach().abs().sum() > 0)
    carry = trainer._prev_tokens
    assert isinstance(carry, WindowCarry) and carry.cells is not None
    assert carry.cells.table.device.type == "cuda" and carry.cells.table.dtype == torch.float32
    mod = WordCells(cfg.model.model_copy(update={"word_cells": "hash"}), seed=cfg.seed)
    cells, pos = mod.init_carry(8, "cpu"), torch.zeros(8, dtype=torch.int64)
    assert len(windows) == 4 * segments
    for batch in windows:
        positions = _window_positions(pos, batch["reset"])
        inputs = (batch["tokens"], batch["context"], batch["reset"], positions, cells)
        win = mod.window(*inputs)
        cells = cells._replace(word=win.word) if mod.learned else mod.write(win)
        pos = positions[:, -1] + 1
    word, table, key, when = (t.cpu() for t in carry.cells)
    if arm == "bytes":
        assert bool((cells.key >= 0).sum() > 100)
        compared = (cells.word, cells.key, cells.when)
        _assert_equal((word, key, when), compared, "the cells on the card")
        assert _gap(table, cells.table) <= 1e-6  # (the card rounds a value's last bit otherwise)
        return
    assert torch.equal(word, cells.word) and bool((when > EMPTY).sum() > 100)
    assert bool(torch.isfinite(table).all()) and 0.1 < float(table.abs().max()) <= 1.0
    stats = history.steps[-1].stats
    assert 0.0 < stats["word_cells_gate"] < 1.0 and ("word_cells_prior" in stats) == (arm != "word")
    moved = {n: not torch.equal(p, start[n]) for n, p in model.word_cells.named_parameters()}
    assert moved.pop("U") and set(moved.values()) == {segments > 1}, moved
    fused = trainer._fused_events
    assert (fused is not None) == (kernel == "fused")
    if fused is not None:
        assert fused.calls == 4 * segments and fused.describe()["buffer_sets"] == segments


eager_kernels = fused_tests.eager_kernels  # the compiled window without inductor (aot_eager)


@fused_tests.needs_card
@pytest.mark.cuda
@pytest.mark.usefixtures("eager_kernels")
@pytest.mark.parametrize(("segments", "arm"), SEGMENTS[2:])
def test_the_hsn_write_with_the_kernels_equals_without_fp32_cuda(
    segments: int, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fp32 on the card, the fused trainer tests' tiny model with stage 2's arms: the trainer
    on the block's kernels (the routed cells of every slot from the tape on the card) against
    the trainer on the compiled blocks, two steps from a start with a margin at every routed
    decision: the same losses, raw gradients, parameters and statistics to the compiled
    window's CUDA tolerance, with two windows a segment the gradients of the write's own
    parameters among them."""
    monkeypatch.setattr(fused_tests, "_long_datasets", _text_datasets)
    accumulate = (f"train.grad_accum={2 * segments}", f"train.tbptt_windows={segments}")
    keys = (*KEYS, *ARM[arm], *accumulate, *fused_tests.FP32)
    ref, got = fused_tests._card_pair(tmp_path, 2, *keys)
    fused = got["trainer"]._fused_events
    assert fused is not None and fused.kernels and fused.calls == 4 * segments
    assert ref["trainer"]._fused_events is None
    fused_tests._assert_same_run(got, ref, loss=1e-5, grad=2e-4, param=1e-3, stats=2e-4)
    names = [n for n, _ in got["trainer"].model.named_parameters()]
    own = [i for i, n in enumerate(names) if n in ("word_cells.beta_K", "word_cells.edge_logit")]
    grads = got["grads"][-1]
    assert len(own) == 2
    assert all((grads[i] is not None and bool(grads[i].any())) == (segments > 1) for i in own)
