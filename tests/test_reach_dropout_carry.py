"""Reach dropout on the carried reader (``train.reach_dropout`` with ``model.reader_carry > 0`` on
HSN-T with its memory; ``docs/reach_dropout.md``, "With the carried reader").

With the carry the reader sees ``C`` bytes across windows, so no loss inside a window needs the
memory; reach dropout shortens some rows' reach in training so that the memory is asked. A short
row narrows the carried reader's sliding span to distances under ``S``, over the cached and the
window positions alike: the mask of ``model.reader_carry = S − 1``. The cache then holds what was
read under the short span, so the trainer holds a row's draw until its next document start: a
document is short or whole from its first byte to its last.

These tests check that the pair loads and keeps #95's hash rules and the other refusals; that
without short rows the carried reader is the one it was, bit for bit (outputs, gradients, the
next cache); that a short row, and a short document inside a row, equals the reader with
``model.reader_carry = S − 1`` (outputs and cache) and every other row the carried reader, at the
reader and in the model (per byte and word events, with and without C-read), in training only;
that ``p = 0`` with the carry hands the model nothing and trains as without the keys; that
``p = 1`` trains as the ``model.reader_carry = S − 1`` configuration and evaluates with the
whole carry; that a row's draw changes at a document start only, with and without
``data.align_docs``, and a resumed run continues bit for bit; and that the compiled window equals
the eager one and compiles nothing new while the rows change (the CPU, and two tests for the
card).
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import (
    RunConfig,
    canonical_json,
    config_hash,
    load_config,
    reach_dropout_scope_error,
)
from hsn.device import seed_everything
from hsn.experiments.spec import memory_key
from hsn.model.hsn import READER_CACHE_KEYS, WindowCarry, address_history
from hsn.model.hsn_t import HSNT
from hsn.model.reader import Cache, Reader
from hsn.train import tbptt
from hsn.train.compile import compile_window
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import restore_determinism  # noqa: F401
from test_hsn_t import CUT_C, READER, _gap, _model, _reader_cfg
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from test_reach_dropout import E1, RECIPE, TOP0, WORDS, _grads_hook
from test_reader_carry_memory import CARRY, FORMS, C, _stream
from test_tbptt_windows import STREAM, _assert_equal, _long_datasets, _trainer
from test_tbptt_windows import _cfg as _trainer_cfg
from test_word_address import _compiled_windows
from test_write_events import CATTN, NGRAM, STE
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

S = 3  # the short span of the small models: the reader with model.reader_carry = S - 1 = 2
SHORT = f"model.reader_carry={S - 1}"
ON = ("train.reach_dropout=0.5", f"train.reach_dropout_span={S}")
ROWS = (True, False, True)  # the short rows of a batch of three
#: float64: a row's outputs against the same row in another batch, or behind a cache of
#: another length (the attention sums over other masked columns)
ROUNDING = 1e-12
#: the carried top0 reader of the names arms (the memo's experiment 2, arm b)
CARRIED = [*TOP0, "model.reader_carry=256"]


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_the_pair_loads_and_keeps_the_hash_rules() -> None:
    """``train.reach_dropout`` beside ``model.reader_carry``: accepted on both memory recipes;
    the keys stay out of the hash at their defaults (#95's rule), change it when set, and leave
    the model and the memory key alone."""
    base = load_config(RECIPE, CARRIED)
    named = load_config(
        RECIPE, [*CARRIED, "train.reach_dropout=0.0", "train.reach_dropout_span=16"]
    )
    assert config_hash(named) == config_hash(base)
    assert "reach_dropout" not in canonical_json(base)
    on = load_config(RECIPE, [*CARRIED, *E1])
    assert reach_dropout_scope_error(on) is None
    eight = load_config(RECIPE, [*CARRIED, "train.reach_dropout=0.5", "train.reach_dropout_span=8"])
    plain = load_config(RECIPE, [*TOP0, *E1])  # reach dropout without the carry
    hashes = {config_hash(cfg) for cfg in (base, on, eight, plain)}
    assert len(hashes) == 4
    protocol = {"name": "train_eval", "eval_batch": 128}
    assert memory_key(on, "pdg", protocol) == memory_key(base, "pdg", protocol)
    assert on.model == base.model
    words = load_config(WORDS, [*CARRIED, *E1])
    assert (words.model.write_events, words.model.reader_carry) == ("word", 256)
    # the span may reach the carry's own (S = C reads one position less than the carry)
    edge = [*TOP0, "model.reader_carry=16", "train.reach_dropout=0.5"]
    assert load_config(RECIPE, [*edge, "train.reach_dropout_span=16"]).train.reach_dropout == 0.5
    assert load_config(RECIPE, [*CARRIED, *E1, STREAM]).data.align_docs is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # a short span that is not shorter than the carry's own span of C + 1 positions
        (
            [
                *TOP0,
                "model.reader_carry=8",
                "train.reach_dropout=0.5",
                "train.reach_dropout_span=9",
            ],
            "not shorter than the carried reader's own span",
        ),
        ([*CARRIED, *E1, "train.reach_dropout_span=256"], "not shorter than the window"),
        # the refusals of the carry and of reach dropout that stay
        ([*CARRIED, *E1, "model.reader_span=16"], "sliding span"),
        ([*CARRIED, *E1, "model.reader_mask=causal"], "reader_mask=causal"),
        ([*TOP0, *E1, "model.reader_mask=causal"], "short rows' span mask could not act"),
        ([*CARRIED, *E1, "train.lambda_u=0.1"], "lambda_u"),
        ([*CARRIED, *E1, "model.memory=false"], "model.memory=false"),
        ([*CARRIED, *E1, "model.key_address=false"], "key_address"),
    ],
)
def test_configuration_refusals(overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(RECIPE, overrides)


# --------------------------------------------------------------------------------------------------
# The reader
# --------------------------------------------------------------------------------------------------


def _readers() -> tuple[Reader, Reader, RunConfig]:
    """The reader with the carry ``C`` and the same weights with the carry ``S − 1``."""
    cfg = _reader_cfg(CARRY)
    whole = Reader(cfg.model, seed=0).double()
    short = Reader(_reader_cfg(SHORT).model, seed=0).double()
    short.load_state_dict(whole.state_dict())
    return whole, short, cfg


def _bytes(cfg: RunConfig, windows: int) -> tuple[Tensor, Tensor]:
    """``windows`` consecutive windows of three rows of bytes as one stream (``tokens``,
    ``reset``). Row 0: one document over the first two windows, the next starts inside the
    third; row 1: a document that starts with the second window; row 2: one that starts two
    bytes before the first boundary, inside the short span's and the carry's reach of it."""
    T = cfg.train.unroll
    generator = torch.Generator().manual_seed(11)
    tokens = torch.randint(0, cfg.model.vocab_size, (3, windows * T), generator=generator)
    reset = torch.zeros(3, windows * T, dtype=torch.bool)
    reset[:, 0] = True
    reset[2, T - 2] = True
    if windows > 1:
        reset[1, T] = True
    if windows > 2:
        reset[0, 2 * T + 3] = True
    return tokens, reset


def _documents(reset: Tensor, first: tuple[bool, ...]) -> Tensor:
    """Bool ``[B, stream]``, one flag per document, as the trainer hands them to the model
    window by window: the documents of row ``b`` are short and whole in turn, the first one
    short where ``first[b]``."""
    index = reset.to(torch.int64).cumsum(dim=1) - 1  # the document of every position
    return (index % 2 == 0) == torch.tensor(first)[:, None]


def _read_windows(
    reader: Reader, tokens: Tensor, reset: Tensor, T: int, flags: Tensor | None = None
) -> tuple[Tensor, list[Cache]]:
    """The stream read window by window with the cache; ``flags``: the short rows (``[B]``) or
    positions (``[B, stream]``), handed in with every window."""
    cache = reader.init_cache(int(tokens.shape[0]), "cpu")
    assert cache is not None
    outs, caches = [], []
    for lo in range(0, int(tokens.shape[1]), T):
        sl = slice(lo, lo + T)
        if flags is None:
            out, cache = reader.read(tokens[:, sl], reset[:, sl], cache)
        else:
            rows = flags if flags.dim() == 1 else flags[:, sl]
            out, cache = reader.read(tokens[:, sl], reset[:, sl], cache, (rows, S))
        outs.append(out)
        caches.append(cache)
    return torch.cat(outs, dim=1), caches


def test_no_short_row_is_the_carried_reader_bit_for_bit() -> None:
    """Without short rows the carried reader is the code it was; with rows of which none is
    short it gives the same outputs, parameter gradients and next cache, bit for bit."""
    whole, _, cfg = _readers()
    T = cfg.train.unroll
    tokens, reset = _bytes(cfg, 2)
    first = whole.read(tokens[:, :T], reset[:, :T], whole.init_cache(3, "cpu"))[1]  # type: ignore[arg-type]
    window = (tokens[:, T:], reset[:, T:])
    params = list(whole.parameters())

    def run(short: tuple[Tensor, int] | None) -> tuple[Tensor, Cache, list[Tensor]]:
        out, cache = (
            whole.read(*window, first) if short is None else whole.read(*window, first, short)
        )
        grads = torch.autograd.grad(out.square().sum(), params, allow_unused=True)
        return out, cache, [g for g in grads if g is not None]

    out, cache, grads = run(None)
    for none in (torch.zeros(3, dtype=torch.bool), torch.zeros(3, T, dtype=torch.bool)):
        got, got_cache, got_grads = run((none, S))
        assert torch.equal(got, out)
        _assert_equal(got_cache, cache, "cache")
        _assert_equal(got_grads, grads, "gradients")


def test_short_rows_read_as_the_short_carry() -> None:
    """Four windows with the same short rows: a short row's mask, output and cache are those of
    the reader with ``model.reader_carry = S − 1`` (documents that start with a window, inside
    one and two bytes before a boundary), and every other row's are the carried reader's, bit
    for bit."""
    whole, short, cfg = _readers()
    T = cfg.train.unroll
    tokens, reset = _bytes(cfg, 4)
    rows = torch.tensor(ROWS)
    with torch.no_grad():
        got, caches = _read_windows(whole, tokens, reset, T, rows)
        plain, plain_caches = _read_windows(whole, tokens, reset, T)
        ref, ref_caches = _read_windows(short, tokens, reset, T)
    assert _gap(got[rows], ref[rows]) <= ROUNDING
    assert torch.equal(got[~rows], plain[~rows])
    assert _gap(ref[rows], plain[rows]) > 1e-3  # the span changes those rows
    tail = slice(C - (S - 1), C)  # the cache's last S - 1 positions: the short carry's cache
    for cache, plain_cache, ref_cache in zip(caches, plain_caches, ref_caches, strict=True):
        for i in (0, 1):  # keys, values [L, B, H, C, w / H]
            assert _gap(cache[i][:, rows][:, :, :, tail], ref_cache[i][:, rows]) <= ROUNDING
            assert torch.equal(cache[i][:, ~rows], plain_cache[i][:, ~rows])
        assert torch.equal(cache[2], plain_cache[2])  # valid: the documents, not the span
        assert torch.equal(cache[2][:, tail], ref_cache[2])
    # the mask itself: a short row's is the short carry's behind the cached columns it drops
    valid = caches[0][2]
    window = reset[:, T : 2 * T]
    mask = whole.carry_mask(window, valid, (rows, S))[:, 0]
    assert not bool(mask[rows][:, :, : C - (S - 1)].any())
    assert torch.equal(
        mask[rows][:, :, C - (S - 1) :], short.carry_mask(window, valid[:, tail])[rows, 0]
    )
    assert torch.equal(mask[~rows], whole.carry_mask(window, valid)[~rows, 0])
    assert bool(mask.diagonal(offset=C, dim1=1, dim2=2).all())  # every position holds itself


def test_short_documents_read_as_the_short_carry() -> None:
    """Flags per position: a document that is short from its start to its end is read as the
    reader with ``model.reader_carry = S − 1`` reads it, wherever it starts (with a window,
    inside one, two bytes before a boundary) and whatever the row's other documents are; a
    whole document is read as the carried reader reads it."""
    whole, short, cfg = _readers()
    T = cfg.train.unroll
    tokens, reset = _bytes(cfg, 4)
    flags = _documents(reset, (False, True, False))
    mixed_rows = flags[:, :T].any(dim=1) & ~flags[:, :T].all(dim=1)
    assert bool(mixed_rows.any())  # a window with a short and a whole document in one row
    with torch.no_grad():
        got, _ = _read_windows(whole, tokens, reset, T, flags)
        plain, _ = _read_windows(whole, tokens, reset, T)
        ref, _ = _read_windows(short, tokens, reset, T)
    assert _gap(got[flags], ref[flags]) <= ROUNDING
    assert _gap(got[~flags], plain[~flags]) <= ROUNDING
    assert _gap(ref[flags], plain[flags]) > 1e-3
    # a row that changes its span inside a document is neither reader: the mismatch the held
    # draw avoids (the cache of a short window read by a whole one, and the reverse)
    switch = torch.zeros_like(flags)
    switch[:, T : 2 * T] = True  # every row short in the second window only
    with torch.no_grad():
        mixed, _ = _read_windows(whole, tokens, reset, T, switch)
    assert not bool(reset[0, 1 : 2 * T + 3].any())  # row 0: one document over these windows
    assert _gap(mixed[0, T : 2 * T], ref[0, T : 2 * T]) > 1e-6  # short, whole entries cached
    assert _gap(mixed[0, 2 * T : 2 * T + 3], plain[0, 2 * T : 2 * T + 3]) > 1e-6  # the reverse


def test_reader_refuses_bad_short_rows() -> None:
    whole, _, cfg = _readers()
    T = cfg.train.unroll
    tokens, reset = _bytes(cfg, 1)
    cache = whole.init_cache(3, "cpu")
    assert cache is not None
    rows = torch.tensor(ROWS)
    for bad, message in (
        ((rows.long(), S), "bool"),
        ((rows[:2], S), "bool"),
        ((torch.zeros(3, T + 1, dtype=torch.bool), S), "bool"),
        ((rows, 0), "int >= 1"),
        ((rows, True), "int >= 1"),
    ):
        with pytest.raises(ValueError, match=message):
            whole.read(tokens, reset, cache, bad)  # type: ignore[arg-type]
    words = cfg.model.model_copy(update={"reader_top_words": 4, "reader_rotary_pairs": 0})
    top = Reader(words, seed=0).double()
    with pytest.raises(ValueError, match="word slots"):
        top.read(tokens, reset, top.init_cache(3, "cpu"), (rows, S))  # type: ignore[arg-type]
    # flags per position are the carried reader's: a reader without a cache takes rows only
    plain = _model(_reader_cfg())
    with pytest.raises(ValueError, match="carried reader's"):
        plain.set_reach_rows(torch.zeros(3, T, dtype=torch.bool), S)
    with pytest.raises(ValueError, match="bool"):
        plain.set_reach_rows(torch.zeros(3, T, 1, dtype=torch.bool), S)


# --------------------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------------------


def _models(*extra: str) -> tuple[HSNT, HSNT, RunConfig]:
    """HSN-T with the carry ``C`` and the same weights with the carry ``S − 1``, in training
    mode (the carry adds no parameter)."""
    cfg = _reader_cfg(CARRY, *extra)
    whole = _model(cfg)
    short = _model(_reader_cfg(SHORT, *extra))
    short.load_state_dict(whole.state_dict())
    return whole, short, cfg


def _run(
    model: HSNT, stream: tuple[Tensor, Tensor, Tensor], T: int, flags: Tensor | None = None
) -> tuple[Tensor, Tensor, list[WindowCarry]]:
    """The stream through the model window by window, as the trainer runs it: the short rows
    (``[B]``) or positions (``[B, stream]``) set before every window and cleared after it.
    Returns the logits, the final cell state and what was carried out of every window."""
    tokens, ctx, reset = stream
    state = model.init_state(int(tokens.shape[0]))
    prev: Tensor | WindowCarry | None = None
    logits, carried = [], []
    with torch.no_grad():
        for lo in range(0, int(tokens.shape[1]), T):
            sl = slice(lo, lo + T)
            if flags is not None:
                model.set_reach_rows(flags if flags.dim() == 1 else flags[:, sl], S)
            try:
                out, state, info = model(
                    tokens[:, sl], ctx[:, sl], state, reset[:, sl], prev_tokens=prev
                )
            finally:
                model.set_reach_rows(None)
            prev = address_history(model.cfg, tokens[:, sl], prev, info)
            assert isinstance(prev, WindowCarry)
            logits.append(out)
            carried.append(prev)
    return torch.cat(logits, dim=1), state.h, carried


@pytest.mark.parametrize("form", list(FORMS))
def test_short_rows_are_the_short_carry_model_in_training_only(form: str) -> None:
    """Three training windows with the same short rows: a short row's logits and final state
    are those of the model with ``model.reader_carry = S − 1``, every other row's the carried
    model's (per byte and word events, with and without C-read), in float64 to rounding. In
    evaluation the rows change nothing, bit for bit: logits, state and the next cache."""
    whole, short, cfg = _models(STE, CUT_C, *NGRAM, *FORMS[form])
    T = cfg.train.unroll
    stream = _stream(cfg, 3)
    rows = torch.tensor(ROWS)
    mixed, plain, ref = _run(whole, stream, T, rows), _run(whole, stream, T), _run(short, stream, T)
    for got, want, base in zip(mixed[:2], ref[:2], plain[:2], strict=True):
        assert _gap(got[rows], want[rows]) <= ROUNDING
        assert _gap(got[~rows], base[~rows]) <= ROUNDING
    assert _gap(ref[0][rows][:, T:], plain[0][rows][:, T:]) > 1e-3
    for carry, want in zip(mixed[2], ref[2], strict=True):  # the cache a short row leaves
        assert _gap(carry.reader[0][:, rows][:, :, :, C - (S - 1) :], want.reader[0][:, rows]) <= (
            ROUNDING
        )
        assert torch.equal(carry.address, want.address)
    whole.eval()
    evaluated = _run(whole, stream, T)
    with_rows = _run(whole, stream, T, rows)
    assert torch.equal(with_rows[0], evaluated[0]) and torch.equal(with_rows[1], evaluated[1])
    _assert_equal([tuple(c) for c in with_rows[2]], [tuple(c) for c in evaluated[2]], "carry")
    assert whole._reach is None


@pytest.mark.parametrize("form", ["per_byte_cread", "word_events"])
def test_short_documents_are_the_short_carry_model(form: str) -> None:
    """Flags per position, as the trainer hands them without ``data.align_docs``: the logits of
    a short document are those of the model with ``model.reader_carry = S − 1`` and the logits
    of a whole document the carried model's (a document start resets the cells, so a document's
    logits depend on no other document of its row)."""
    whole, short, cfg = _models(STE, CUT_C, *NGRAM, *FORMS[form])
    T = cfg.train.unroll
    stream = _stream(cfg, 3)
    flags = _documents(stream[2], (True, False, True))
    assert bool((flags.any(dim=1) & ~flags.all(dim=1)).any())  # rows of short and whole documents
    mixed, plain, ref = (
        _run(whole, stream, T, flags),
        _run(whole, stream, T),
        _run(short, stream, T),
    )
    assert _gap(mixed[0][flags], ref[0][flags]) <= ROUNDING
    assert _gap(mixed[0][~flags], plain[0][~flags]) <= ROUNDING
    assert _gap(ref[0][flags], plain[0][flags]) > 1e-3


# --------------------------------------------------------------------------------------------------
# The trainer
# --------------------------------------------------------------------------------------------------


def _spy_draws(trainer: TBPTTTrainer) -> list[tuple[Tensor, Tensor]]:
    """The window's reset and the flags the trainer drew for it, window by window."""
    seen: list[tuple[Tensor, Tensor]] = []
    original = trainer._reach_rows

    def spy(B: int, reset: Tensor | None = None) -> Tensor | None:
        rows = original(B, reset)
        if rows is not None:
            assert reset is not None
            seen.append((reset.clone(), rows.clone()))
        return rows

    trainer._reach_rows = spy  # type: ignore[method-assign]
    return seen


def _fit(
    cfg: RunConfig, run_dir: Path, steps: int
) -> tuple[TBPTTTrainer, Any, list[list[Tensor | None]], list[tuple[Tensor, Tensor]]]:
    """``steps`` steps on documents of two to four windows: the trainer, its history, the raw
    gradients of every step and the flags of every window."""
    trainer = _trainer(cfg, run_dir)
    grads: list[list[Tensor | None]] = []
    trainer.grad_hook = _grads_hook(grads)
    draws = _spy_draws(trainer)
    history = trainer.fit(*_long_datasets(), steps)
    return trainer, history, grads, draws


def _records(history: Any) -> list[tuple[Any, ...]]:
    """What every step measured, comparable bit for bit (a step without a loss position has
    the loss NaN: its text stands for it)."""
    return [(s.step, repr(s.loss), s.total, s.parts, s.grad_norm, s.windows) for s in history.steps]


def _same_grads(got: list[list[Tensor | None]], want: list[list[Tensor | None]]) -> bool:
    return len(got) == len(want) and all(
        (g is None and w is None) or (g is not None and w is not None and torch.equal(g, w))
        for a, b in zip(got, want, strict=True)
        for g, w in zip(a, b, strict=True)
    )


def test_p0_with_the_carry_trains_as_without_the_keys(tmp_path: Path) -> None:
    """``train.reach_dropout = 0`` beside the carry: the trainer hands the model nothing, holds
    no draw, writes no ``reach_held`` into its checkpoints, and the run is the run without the
    keys bit for bit: losses, raw gradients, weights and the carried cache."""
    steps = 4
    base = _trainer_cfg(steps, CARRY, STREAM, "train.stats_every=1", "train.ckpt_every=2")
    named = _trainer_cfg(
        steps,
        CARRY,
        STREAM,
        "train.stats_every=1",
        "train.ckpt_every=2",
        "train.reach_dropout=0.0",
        "train.reach_dropout_span=16",
    )
    assert config_hash(named) == config_hash(base)
    a, history_a, grads_a, draws_a = _fit(base, tmp_path / "a", steps)
    b, history_b, grads_b, draws_b = _fit(named, tmp_path / "b", steps)
    assert draws_a == draws_b == [] and b._reach_held is None and not b._reach_hold
    assert _records(history_a) == _records(history_b)
    assert _same_grads(grads_a, grads_b)
    _assert_equal(a.model.state_dict(), b.model.state_dict(), "weights")
    assert isinstance(a._prev_tokens, WindowCarry) and isinstance(b._prev_tokens, WindowCarry)
    _assert_equal(tuple(a._prev_tokens), tuple(b._prev_tokens), "carry")
    extra = torch.load(tmp_path / "b" / "step_00000002.pt", weights_only=True)["extra"]
    assert "reach_held" not in extra
    # reach dropout without the carry keeps its rows per window and holds nothing
    plain = _trainer(_trainer_cfg(steps, *ON), tmp_path / "plain")
    drawn = plain._reach_rows(4)
    assert drawn is not None and drawn.shape == (4,) and plain._reach_held is None


def test_p1_trains_as_the_short_carry_and_evaluates_the_whole_carry(tmp_path: Path) -> None:
    """``p = 1``: every document is short, so the training windows are those of the
    ``model.reader_carry = S − 1`` configuration (losses and raw gradients to fp32 rounding:
    the two attentions sum over caches of other lengths), while evaluation reads with the whole
    carry: it is the carried configuration's evaluation of the same weights."""
    steps = 3
    dev = _long_datasets()[1]
    common = (STREAM, "train.stats_every=1")
    dropout = _trainer_cfg(steps, CARRY, *common, "train.reach_dropout=1.0", ON[1])
    a, history_a, grads_a, draws = _fit(dropout, tmp_path / "dropout", steps)
    b, history_b, grads_b, _ = _fit(_trainer_cfg(steps, SHORT, *common), tmp_path / "short", steps)
    assert len(draws) == steps and all(bool(rows.all()) for _, rows in draws)
    for x, y in zip(history_a.steps, history_b.steps, strict=True):
        assert abs(x.total - y.total) <= 1e-5 * max(1.0, abs(y.total)), (x.total, y.total)
    tol = p28.TOL[("cpu", "fp32")]["grads"]
    for got, want in zip(grads_a, grads_b, strict=True):
        assert p28._grad_diff(got, want) <= tol
    assert a.hsn._reach is None  # cleared after every window
    loss = a.evaluate(dev)["loss"]
    for name, carry, same in (("whole", CARRY, True), ("short_eval", SHORT, False)):
        other = _trainer(_trainer_cfg(steps, carry, *common), tmp_path / name)
        other.model.load_state_dict(a.model.state_dict())
        assert (other.evaluate(dev)["loss"] == loss) is same, name


@pytest.mark.parametrize("aligned", [True, False])
def test_a_rows_draw_is_held_until_its_next_document_start(aligned: bool, tmp_path: Path) -> None:
    """Documents of two to four windows, with ``data.align_docs`` (a document starts at a
    window's first column) and without it (anywhere): the flags the trainer hands the model are
    per position, a document keeps one flag from its start to its end, across windows, the
    flag of a document that starts in a window is that window's draw for the row, and the flags
    do change between documents. A rerun repeats them; the global random stream is untouched."""
    steps = 12
    cfg = _trainer_cfg(steps, CARRY, *ON, *(() if aligned else (STREAM,)))
    assert cfg.data.align_docs is aligned
    trainer, history, _, draws = _fit(cfg, tmp_path / "a", steps)
    assert len(draws) == steps and all(math.isfinite(s.total) for s in history.steps)
    B, T = cfg.train.batch, cfg.train.unroll
    held: Tensor | None = None
    changes = inside = 0
    for index, (reset, flags) in enumerate(draws):
        assert reset.shape == flags.shape == (B, T) and flags.dtype is torch.bool
        # the window's own draw: the generator of (run seed, window counter), as without the carry
        seed = np.random.SeedSequence([cfg.seed, 0x52454143, index + 1]).generate_state(1)
        fresh = torch.rand(B, generator=torch.Generator().manual_seed(int(seed[0]))) < 0.5
        going_on = reset.cumsum(dim=1) == 0
        before = fresh if held is None else held
        want = torch.where(going_on, before[:, None], fresh[:, None])
        assert torch.equal(flags, want), index
        if aligned:
            assert not bool(reset[:, 1:].any())
        inside += int((flags.any(dim=1) & ~flags.all(dim=1)).sum())  # two documents, two flags
        if held is not None:
            changes += int((flags[:, -1] != held).sum())
            assert torch.equal(flags[:, 0][~reset[:, 0]], held[~reset[:, 0]])  # held over the edge
        held = flags[:, -1]
    assert trainer._reach_held is not None and torch.equal(trainer._reach_held, held)  # type: ignore[arg-type]
    assert changes > 0 and (inside > 0) is (not aligned)
    stacked = torch.stack([flags for _, flags in draws])
    assert bool(stacked.any()) and not bool(stacked.all())
    again = _fit(cfg, tmp_path / "b", steps)
    assert _records(again[1]) == _records(history)
    _assert_equal([f for _, f in again[3]], [f for _, f in draws], "flags")
    before_rng = torch.get_rng_state()
    drawn = trainer._reach_rows(B, draws[-1][0])
    assert drawn is not None and drawn.shape == (B, T)
    assert torch.equal(torch.get_rng_state(), before_rng)  # the global random stream is untouched
    with pytest.raises(ValueError, match="needs the window's reset"):
        trainer._reach_rows(B)


@pytest.mark.parametrize("segments", [1, 2])
def test_resume_continues_bit_for_bit(segments: int, tmp_path: Path) -> None:
    """A run resumed from its step-2 checkpoint (other initial weights) continues the
    uninterrupted run bit for bit: records, weights, the carry and the flags of every later
    window, on documents that run over several windows; also with a gradient segment of two
    windows. The checkpoint holds the rows' held draws, and one without them is refused."""
    cfg = _trainer_cfg(
        4,
        STREAM,
        CARRY,
        *ON,
        f"train.grad_accum={2 * segments}",
        f"train.tbptt_windows={segments}",
        "train.ckpt_every=2",
        "train.stats_every=2",
    )
    train_ds, dev_ds = _long_datasets()
    trainer_a = _trainer(cfg, tmp_path / "a")
    draws_a = _spy_draws(trainer_a)
    history_a = trainer_a.fit(train_ds, dev_ds, 4)
    checkpoint = tmp_path / "a" / "step_00000002.pt"
    held = torch.load(checkpoint, weights_only=True)["extra"]["reach_held"]
    assert held.dtype is torch.bool and held.shape == (cfg.train.batch,)
    per_step = 2 * segments
    assert torch.equal(held, draws_a[2 * per_step - 1][1][:, -1])
    trainer_b = _trainer(cfg, tmp_path / "b", seed=1)
    assert trainer_b.resume(checkpoint) == 2
    assert trainer_b._reach_held is not None and torch.equal(trainer_b._reach_held, held)
    draws_b = _spy_draws(trainer_b)
    history_b = trainer_b.fit(train_ds, dev_ds, 4)
    assert _records(history_a)[2:] == _records(history_b)
    _assert_equal([f for _, f in draws_a[2 * per_step :]], [f for _, f in draws_b], "flags")
    _assert_equal(trainer_a.model.state_dict(), trainer_b.model.state_dict(), "weights")
    carry_a, carry_b = trainer_a._prev_tokens, trainer_b._prev_tokens
    assert isinstance(carry_a, WindowCarry) and isinstance(carry_b, WindowCarry)
    _assert_equal(tuple(carry_a), tuple(carry_b), "carry")
    # a checkpoint of the carry alone (no held draws) is not this configuration's
    payload = torch.load(checkpoint, weights_only=True)
    del payload["extra"]["reach_held"]
    stripped = tmp_path / "stripped.pt"
    torch.save(payload, stripped)
    with pytest.raises(ValueError, match="written without them"):
        _trainer(cfg, tmp_path / "c").resume(stripped)


# --------------------------------------------------------------------------------------------------
# The compiled window
# --------------------------------------------------------------------------------------------------


def _compiled_model(form: str, T: int) -> tuple[HSNT, RunConfig]:
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        STE,
        CUT_C,
        *READER,
        *NGRAM,
        "model.address_gate=true",
        "model.address_gate_input=summary",
        CARRY,
        *FORMS[form],
    )
    seed_everything(0)
    model = HSNT.from_config(cfg)
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():  # as p28._model(cfg, training=True, scale=0.3)
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.3)
    return model.train(True), cfg


def _window_rows(T: int) -> list[Tensor | None]:
    """The short rows of five consecutive windows of three rows: rows, other rows, flags per
    position, none, rows again (a window-level input; a trainer holds a document's draw)."""
    position = torch.zeros(3, T, dtype=torch.bool)
    position[0], position[2, T // 2 :] = True, True
    return [
        torch.tensor(ROWS),
        torch.tensor((False, True, True)),
        position,
        None,
        torch.tensor((True, True, False)),
    ]


@pytest.mark.parametrize("form", ["per_byte_cread", "word_events_cread"])
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_equals_eager_and_new_rows_compile_nothing(
    form: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader runs in the eager prologue, so short rows on the carried reader change no
    graph: over five consecutive windows with other short rows in every window (rows, flags per
    position, none) the compiled window equals the eager one (values, the next cache, the
    carried address, the parameter gradients), compiles one graph and keeps it."""
    from torch._dynamo.utils import counters

    monkeypatch.setattr("hsn.model.hsn.EVENT_BLOCK", p28.compile_mod.MAX_BLOCK)  # small_blocks
    T = 20
    model, cfg = _compiled_model(form, T)
    window = p28._window(model, p28._ctx())
    tol = p28.TOL[("cpu", "fp32")]
    state = p28._inputs(model, b=3, t=T)[2]
    carry: Tensor | WindowCarry | None = None
    shorts = _window_rows(T)
    graphs = -1
    for i, (tokens, ctx, reset) in enumerate(_compiled_windows(T)):
        model.set_reach_rows(shorts[i], S)
        ref = p28._run(functools.partial(model, tokens, ctx, state, reset, prev_tokens=carry))
        compiled = functools.partial(window, tokens, ctx, state, reset, prev_tokens=carry)
        if i == 0:
            got = p28._run(compiled)
            graphs = counters["stats"]["unique_graphs"]
        else:
            with torch.compiler.set_stance("fail_on_recompile"):
                got = p28._run(compiled)
        model.set_reach_rows(None)
        assert set(READER_CACHE_KEYS) <= set(got[2])
        diff = p28._max_diff(got, ref)  # the cache's valid, the carried address: exact
        assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], (i, diff)
        assert p28._grad_diff(p28._grads(model, got), p28._grads(model, ref)) <= tol["grads"], i
        if shorts[i] is not None:  # the rows did change this window's values
            plain = p28._run(functools.partial(model, tokens, ctx, state, reset, prev_tokens=carry))
            assert float((plain[0] - ref[0]).abs().max()) > 1e-4, i
        carry = address_history(cfg.model, tokens, carry, ref[2])
        state = got[1].detach()
    assert counters["stats"]["unique_graphs"] == graphs
    assert window.calls == 5


def _compiled_trainer_cfg(steps: int, mode: str, *extra: str) -> RunConfig:
    return _trainer_cfg(
        steps,
        STREAM,
        CARRY,
        *ON,
        CATTN,
        "train.stats_every=1",
        f"hardware.compile_mode={mode}",
        *extra,
    )


@pytest.mark.parametrize("events", [False, True])
@pytest.mark.usefixtures("small_blocks")
def test_compiled_trainer_equals_eager_without_new_graphs(
    events: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four optimiser steps on documents of several windows with ``hardware.compile_mode=
    reduce-overhead`` (the HSN-T arms' mode; ``aot_eager`` on the CPU, where there are no CUDA
    graphs) and with ``none``: the carry, held draws, C-read, every step a statistics step. The
    flags are the same in both runs and change from step to step, the losses and raw gradients
    agree to the compiled-vs-eager tolerance, and the steps after the first compile nothing
    new. ``events``: the loop over events with the exact word address."""
    from torch._dynamo.utils import counters

    monkeypatch.setattr(
        tbptt, "compile_window", functools.partial(compile_window, backend="aot_eager")
    )
    steps = 4
    words = ("model.write_events=word", "model.word_address=exact") if events else ()
    runs: dict[str, tuple[Any, list[list[Tensor | None]], list[tuple[Tensor, Tensor]]]] = {}
    for mode in ("none", "reduce-overhead"):
        trainer = _trainer(_compiled_trainer_cfg(steps, mode, *words), tmp_path / mode)
        draws = _spy_draws(trainer)
        grads: list[list[Tensor | None]] = []
        graphs: list[int] = []
        grad_hook = _grads_hook(grads)

        def hook(t: Any, grad_hook: Any = grad_hook, graphs: list[int] = graphs) -> None:
            grad_hook(t)
            graphs.append(int(counters["stats"]["unique_graphs"]))

        trainer.grad_hook = hook
        history = trainer.fit(*_long_datasets(), steps)
        runs[mode] = (history, grads, draws)
        if mode != "none":
            compiled = trainer.compiled_window
            assert compiled is not None and not compiled.cuda_graphs
            assert history.measured["compiled_windows"] >= steps
            assert graphs[1:] == [graphs[0]] * (steps - 1), graphs
    (eager, eager_grads, eager_draws), (comp, comp_grads, comp_draws) = runs.values()
    assert len(comp_draws) == steps
    assert len({tuple(flags.flatten().tolist()) for _, flags in comp_draws}) > 1
    _assert_equal([f for _, f in comp_draws], [f for _, f in eager_draws], "flags")
    for a, b in zip(eager.steps, comp.steps, strict=True):
        assert abs(a.total - b.total) <= 1e-5 * max(1.0, abs(a.total)), (a.total, b.total)
    for got, ref in zip(comp_grads, eager_grads, strict=True):
        assert p28._grad_diff(got, ref) <= p28.TOL[("cpu", "fp32")]["grads"]


# --------------------------------------------------------------------------------------------------
# The card (not run where the card is shared: docs/reach_dropout.md, "To run on the card")
# --------------------------------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("form", ["per_byte_cread", "word_events_cread"])
def test_compiled_window_with_short_rows_on_the_carry_cuda(form: str) -> None:
    """On the card (inductor, CUDA Graphs, fp32): five consecutive windows with other short
    rows in every window (rows, flags per position, none) run on the graph of the first (no
    recompile), give finite logits and the cache on the device, and the rows change the
    window's values. (The values against the eager window: the CPU test above.)"""
    from torch._dynamo.utils import counters

    ctx = p28._cuda_ctx()
    T = 20
    model, cfg = _compiled_model(form, T)
    model = model.to(ctx.device)
    window = p28._window(model, ctx, backend="inductor")
    state = p28._inputs(model, b=3, t=T)[2]
    carry: Tensor | WindowCarry | None = None
    graphs = -1
    for i, (tokens, cx, reset) in enumerate(_compiled_windows(T)):
        tokens, cx, reset = tokens.to(ctx.device), cx.to(ctx.device), reset.to(ctx.device)
        rows = _window_rows(T)[i]
        model.set_reach_rows(None if rows is None else rows.to(ctx.device), S)
        compiled = functools.partial(window, tokens, cx, state, reset, prev_tokens=carry)
        if i == 0:
            got = p28._run(compiled)
            graphs = counters["stats"]["unique_graphs"]
        else:
            with torch.compiler.set_stance("fail_on_recompile"):
                got = p28._run(compiled)
        model.set_reach_rows(None)
        assert bool(torch.isfinite(got[0]).all()), i
        if rows is not None:
            with torch.no_grad():
                plain = model(tokens, cx, state, reset, prev_tokens=carry)
            assert float((plain[0] - got[0]).abs().max()) > 1e-4, i
        carry = address_history(cfg.model, tokens, carry, got[2])
        assert isinstance(carry, WindowCarry) and carry.reader[0].device.type == "cuda"
        state = got[1].detach()
    assert counters["stats"]["unique_graphs"] == graphs
    assert window.calls == 5


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("events", [False, True])
def test_compiled_training_with_reach_dropout_on_the_carry_cuda(
    events: bool, precision: str, tmp_path: Path
) -> None:
    """On the card: twelve steps on documents of several windows through the compiled window
    with CUDA Graphs (fp32 and bf16 autocast), the carry, held draws and C-read on: finite
    losses, flags that change from step to step and are the eager CPU run's, the carry on the
    device, and no new graph from the second step to the last but one (the last is the
    statistics step). (Whether the full configuration compiles once and steps at the expected
    rate is ``scripts/word_address_gpu_compile_check.py``: ``docs/reach_dropout.md``, "To run
    on the card".)"""
    from cards import CUDA_PROFILE
    from torch._dynamo.utils import counters

    p28._cuda_ctx()
    steps = 12
    words = ("model.write_events=word", "model.word_address=exact") if events else ()
    cfg = _trainer_cfg(
        steps,
        STREAM,
        CARRY,
        *ON,
        CATTN,
        *words,
        f"profile={CUDA_PROFILE}",
        "hardware.compile_mode=reduce-overhead",
        f"hardware.autocast_dtype={precision}",
        f"train.precision={precision}",
    )
    trainer = _trainer(cfg, tmp_path / "run")
    draws = _spy_draws(trainer)
    graphs: list[int] = []
    trainer.grad_hook = lambda _t: graphs.append(int(counters["stats"]["unique_graphs"]))
    history = trainer.fit(*_long_datasets(), steps)
    assert len(history.steps) == steps and all(math.isfinite(s.total) for s in history.steps)
    assert trainer.compiled_window is not None and trainer.compiled_window.cuda_graphs
    # the steps after the second and before the last (the statistics step) compile nothing new
    assert graphs[2:-1] == [graphs[1]] * (steps - 3), graphs
    assert len(draws) == steps and all(flags.device.type == "cuda" for _, flags in draws)
    assert len({tuple(flags.flatten().tolist()) for _, flags in draws}) > 1
    cpu = _trainer(_trainer_cfg(steps, STREAM, CARRY, *ON, CATTN, *words), tmp_path / "cpu")
    reference = _spy_draws(cpu)
    cpu.fit(*_long_datasets(), steps)
    _assert_equal([f.cpu() for _, f in draws], [f for _, f in reference], "flags")
    carry = trainer._prev_tokens
    assert isinstance(carry, WindowCarry)
    keys, values, valid = carry.reader
    assert keys.device.type == "cuda" and keys.dtype == values.dtype == torch.float32
    assert bool(valid.any()) and trainer.hsn._reach is None
