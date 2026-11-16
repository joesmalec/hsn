"""The top block's word ends (``model.reader_top_words``, ``docs/hsn_t_no_memory.md``, "Reading
past the window"; opt-in, 15 Nov 2026).

With ``model.reader_top_words = N`` (and ``model.reader_carry = C``, the top block unrotated)
the top block also reads the keys and values at the last ``N`` word ends of its document before
the window, carried and detached; an entry inside the local span is not read twice. A word end
is a byte that is not a letter or digit directly after one (ASCII letters and digits, and bytes
of 128 and above, count as letters). These tests check the word ends on hand-made bytes; that
the default and the key at 0 change nothing; that a stream read window by window equals a
direct computation whose word reads are chosen by loops; that the word entries inside the
local span are masked; that a document reads no word of the one before it; that the cache is
detached; the refusals; the trainer and the compiled window.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from hsn.config import HASH_OMIT_AT_DEFAULT, canonical_json, config_hash, load_config
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import config_view, memory_key
from hsn.model.reader import Reader, word_bytes, word_ends
from hsn.runner import build_model, make_trainer
from hsn.train.loop import Trainer
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, train_overrides
from test_reader_carry import (
    NOMEM,
    PROTOCOL,
    READER_LM,
    RECIPE_A6,
    TINY,
    _by_windows,
    _direct,
    _gap,
    _reader,
    _sliding_visible,
)
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

NEWLINE, TAB = 16, 17  # the text importer's markers (hsn.data.text)
TOP0 = [4, 2, 0]  # three blocks, the top one unrotated


def _bytes(text: str) -> Tensor:
    """``text`` as the importer writes it: UTF-8, newline and tab as their markers."""
    raw = text.encode("utf-8")
    return torch.tensor([[NEWLINE if b == 10 else TAB if b == 9 else b for b in raw]])


def _ends_at(text: str) -> list[int]:
    tokens = _bytes(text)
    reset = torch.zeros_like(tokens, dtype=torch.bool)
    reset[0, 0] = True
    return (
        word_ends(tokens, reset, torch.zeros(1, dtype=torch.bool))[0].nonzero().flatten().tolist()
    )


def _word_visible(tokens: Tensor, reset: Tensor, C: int, N: int, T: int) -> Tensor:
    """Bool ``[B, S, S]``: the word entries query ``a`` reads in the windows of ``T``, by loops.
    The carried set is the last ``N`` word ends of ``a``'s document before ``a``'s window; the
    window's own word ends before ``a − C`` follow them; of the two, the last ``N`` are read,
    those at ``s < a − C`` (outside the local span) only."""
    B, S = tokens.shape
    letters = [[bool(word_bytes(tokens[b, s])) for s in range(S)] for b in range(B)]
    visible = torch.zeros(B, S, S, dtype=torch.bool)
    for b in range(B):
        start = 0
        ends: list[int] = []
        for a in range(S):
            if bool(reset[b, a]):
                start, ends = a, []
            elif a > start and letters[b][a - 1] and not letters[b][a]:
                ends.append(a)
            first = (a // T) * T  # a's window starts here
            carried = [s for s in ends if s < first][-N:]
            own = [s for s in ends if first <= s < a - C]
            for s in (carried + own)[-N:]:
                if s < a - C:
                    visible[b, a, s] = True
    return visible


def _direct_words(reader: Reader, tokens: Tensor, reset: Tensor, T: int) -> Tensor:
    """``_direct`` with the top block's reads widened by the word entries."""
    C, N = reader.carry, reader.top_words
    near = _sliding_visible(reset, C)
    return _direct(reader, tokens, near, top=near | _word_visible(tokens, reset, C, N, T))


def _text_stream(B: int, S: int, seed: int, resets: float = 0.03) -> tuple[Tensor, Tensor]:
    """Bytes with short words: letters, digits, a byte >= 128, separated by spaces, commas,
    newlines and tabs."""
    g = torch.Generator().manual_seed(seed)
    alphabet = torch.tensor([*b"abcdeXYZ019", 0xC3, 0xA9, *b"  ,.", NEWLINE, TAB])
    tokens = alphabet[torch.randint(0, len(alphabet), (B, S), generator=g)]
    reset = torch.rand(B, S, generator=g) < resets
    reset[:, 0] = True
    return tokens, reset


# --------------------------------------------------------------------------------------------------
# Word ends
# --------------------------------------------------------------------------------------------------


def test_word_ends_on_hand_made_bytes() -> None:
    #            0123456789012345678
    assert _ends_at("Hi, you2 are.\nOK") == [2, 8, 12]
    assert _ends_at("one  two\t3") == [3, 8]  # spaces after a space end nothing
    assert _ends_at("café latte.") == [5, 11]  # é is two bytes >= 128: letters
    assert _ends_at("a") == [] and _ends_at(" a b ") == [2, 4]
    assert _ends_at("x\n\ny") == [1]  # the newline marker (16) after a letter
    # a word that runs to the window's end ends at the next window's first byte
    tokens = _bytes("ab cd")
    reset = torch.zeros_like(tokens, dtype=torch.bool)
    assert word_ends(tokens[:, :1], reset[:, :1], torch.tensor([True])).tolist() == [[False]]
    assert word_ends(tokens[:, 2:], reset[:, 2:], torch.tensor([True])).tolist() == [
        [True, False, False]
    ]
    assert word_ends(tokens[:, 2:], reset[:, 2:], torch.tensor([False])).tolist() == [
        [False, False, False]
    ]
    # a document's first byte ends nothing, whatever the byte before it
    reset[0, 2] = True
    assert word_ends(tokens[:, 2:], reset[:, 2:], torch.tensor([True])).tolist() == [
        [False, False, False]
    ]
    letters = word_bytes(torch.arange(256))
    assert letters.nonzero().flatten().tolist() == [
        *range(48, 58),
        *range(65, 91),
        *range(97, 123),
        *range(128, 256),
    ]


# --------------------------------------------------------------------------------------------------
# The key: opt-in, refusals
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "overrides"),
    [
        (RECIPE_A6, ()),
        (READER_LM, ()),
        (
            NOMEM,
            (
                "model.reader_width=160",
                "model.reader_layers=4",
                "model.reader_rotary_pairs=[20,20,20,0]",
                "model.reader_carry=256",
            ),
        ),
    ],
)
def test_the_key_is_opt_in_and_its_default_keeps_hashes_memory_keys_and_arm_ids(
    path: Path, overrides: tuple[str, ...]
) -> None:
    base = load_config(path, list(overrides))
    assert base.model.reader_top_words == 0
    assert "reader_top_words" not in json.loads(canonical_json(base))["model"]
    assert "model.reader_top_words" in HASH_OMIT_AT_DEFAULT
    assert "reader_top_words" not in config_view(base).get("model", {})
    if base.model.reader == "none":
        return
    named = load_config(path, [*overrides, "model.reader_top_words=0"])
    assert config_hash(named) == config_hash(base)
    assert memory_key(named, "none", PROTOCOL) == memory_key(base, "none", PROTOCOL)
    assert config_view(named) == config_view(base)
    if not base.model.reader_carry:
        return
    on = load_config(path, [*overrides, "model.reader_top_words=256"])
    assert config_hash(on) != config_hash(base)
    assert memory_key(on, "none", PROTOCOL) != memory_key(base, "none", PROTOCOL)
    assert config_view(on)["model"]["reader_top_words"] == 256


TOP = ["model.reader_layers=3", "model.reader_rotary_pairs=[8,8,0]"]


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        (RECIPE_A6, ["model.reader_top_words=8"], "carries the keys of model.reader"),
        (READER_LM, [*TOP, "model.reader_top_words=8"], "set model.reader_carry > 0"),
        (READER_LM, ["model.reader_carry=8", "model.reader_top_words=8"], "top block unrotated"),
        (
            READER_LM,
            [
                *TOP[:1],
                "model.reader_rotary_pairs=[8,0,8]",
                "model.reader_carry=8",
                "model.reader_top_words=8",
            ],
            "top block unrotated",
        ),
        (
            READER_LM,
            ["model.reader_rotary_pairs=4", "model.reader_carry=8", "model.reader_top_words=8"],
            "top block unrotated",
        ),
        (READER_LM, [*TOP, "model.reader_carry=8", "model.reader_top_words=-1"], "top_words"),
    ],
)
def test_refusals(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(path, overrides)


def test_accepted_settings() -> None:
    cfg = load_config(READER_LM, [*TOP, "model.reader_carry=8", "model.reader_top_words=4"])
    assert build_model(cfg).reader.top_words == 4
    every = ["model.reader_rotary_pairs=0", "model.reader_carry=8", "model.reader_top_words=4"]
    assert load_config(READER_LM, every).model.reader_top_words == 4  # no block rotated
    nomem = [
        *TINY,
        "model.reader_rotary_pairs=[4,0]",
        "model.reader_carry=8",
        "model.reader_top_words=4",
    ]
    assert build_model(load_config(NOMEM, nomem)).reader.top_words == 4


# --------------------------------------------------------------------------------------------------
# The reads
# --------------------------------------------------------------------------------------------------


def test_words_zero_is_the_carrying_reader() -> None:
    plain = _reader(reader_carry=8, reader_rotary_pairs=TOP0)
    named = _reader(reader_carry=8, reader_rotary_pairs=TOP0, reader_top_words=0)
    assert len(plain.init_cache(2, "cpu")) == len(named.init_cache(2, "cpu")) == 3  # type: ignore[arg-type]
    tokens, reset = _text_stream(2, 30, 1)
    with torch.no_grad():
        assert torch.equal(
            _by_windows(plain, tokens, reset, 6), _by_windows(named, tokens, reset, 6)
        )


@pytest.mark.parametrize(("T", "C", "N"), [(6, 6, 3), (6, 4, 2), (4, 8, 5), (12, 3, 2), (5, 2, 40)])
def test_windows_with_words_are_the_direct_computation(T: int, C: int, N: int) -> None:
    """The direct computation over the whole stream: every block reads its sliding span, the
    top block also the word entries ``_word_visible`` chooses by loops. Each key is read once
    there, so the windows count no word entry twice."""
    reader = _reader(reader_carry=C, reader_rotary_pairs=TOP0, reader_top_words=N)
    tokens, reset = _text_stream(3, 60, 4)
    reset[1, 24] = reset[2, 30] = True
    with torch.no_grad():
        got = _by_windows(reader, tokens, reset, T)
        want = _direct_words(reader, tokens, reset, T)
        plain = _reader(reader_carry=C, reader_rotary_pairs=TOP0)
        without = _by_windows(plain, tokens, reset, T)
    assert _gap(got, want) <= 1e-11
    assert _gap(got, without) > 1e-3  # the word entries are read


def test_word_entries_inside_the_local_span_are_masked() -> None:
    """Two windows of 8 with a span of 16: every word end of the first window lies inside the
    span of every position of the second, so the word entries add nothing; and the mask of
    the word entries is False wherever the entry is within ``C`` of the query."""
    C, N, T = 16, 8, 8
    words = _reader(reader_carry=C, reader_rotary_pairs=TOP0, reader_top_words=N)
    plain = _reader(reader_carry=C, reader_rotary_pairs=TOP0)
    tokens = _bytes("ab cd e,fg hi j k")[:, :16]
    reset = torch.zeros_like(tokens, dtype=torch.bool)
    reset[0, 0] = True
    with torch.no_grad():
        a = _by_windows(words, tokens, reset, T)
        b = _by_windows(plain, tokens, reset, T)
    assert _gap(a, b) <= 1e-12
    # the masks themselves, on a carried set placed by hand
    wpos = torch.tensor([[-1, -5, -16, -17, -30, -40, -41, -50]])
    wvalid = torch.tensor([[True] * 7 + [False]])
    no_ends = torch.zeros(1, T, dtype=torch.bool)
    no_reset = torch.zeros(1, T, dtype=torch.bool)
    _, cached = words.word_masks(no_reset, no_ends, wvalid, wpos)
    t = torch.arange(T)[:, None]
    assert torch.equal(cached[0], (wpos[0][None, :] < t - C) & wvalid[0][None, :])
    assert not bool(cached[0][(t - wpos[0][None, :]) <= C].any())


@pytest.mark.parametrize("T", [5, 8])
def test_a_document_reads_no_word_of_the_one_before_it(T: int) -> None:
    reader = _reader(reader_carry=4, reader_rotary_pairs=TOP0, reader_top_words=6)
    tokens, reset = _text_stream(2, 40, 3, 0.0)
    reset[0, 24] = True
    reset[1, 21] = True
    with torch.no_grad():
        out = _by_windows(reader, tokens, reset, T)
        before = tokens.clone()
        before[0, :24] = torch.flip(before[0, :24], dims=[0])
        before[1, :21] = (before[1, :21] + 1) % 256
        changed = _by_windows(reader, before, reset, T)
    assert torch.equal(changed[0, 24:], out[0, 24:]) and torch.equal(changed[1, 21:], out[1, 21:])
    assert _gap(changed[1, :21], out[1, :21]) > 0.0


def test_the_word_cache_is_detached_and_keeps_the_last_N_word_ends() -> None:
    C, N, T = 3, 4, 6
    reader = _reader(reader_carry=C, reader_rotary_pairs=TOP0, reader_top_words=N).train()
    text = "aa bb cc dd ee ff gg hh"  # word ends at 2, 5, 8, 11, 14, 17, 20
    tokens = _bytes(text)[:, :18]
    reset = torch.zeros_like(tokens, dtype=torch.bool)
    reset[0, 0] = True
    cache = reader.init_cache(1, "cpu")
    for lo in range(0, 18, T):
        out, cache = reader.read(tokens[:, lo : lo + T], reset[:, lo : lo + T], cache)  # type: ignore[arg-type]
        assert all(not x.requires_grad and x.grad_fn is None for x in cache)
    _, _, _, wk, wv, wvalid, wpos, last = cache
    # after the stream (18 bytes): the ends 17, 14, 11, 8, most recent first, relative to 18
    assert wvalid.tolist() == [[True] * 4] and wpos.tolist() == [[-1, -4, -7, -10]]
    assert last.tolist() == [False]  # byte 17 is the space that ends "ff"
    out.square().mean().backward()
    assert all(p.grad is not None for p in reader.parameters())


# --------------------------------------------------------------------------------------------------
# The trainer and the compiled window
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [NOMEM, READER_LM])
def test_the_generic_trainer_trains_and_evaluates_with_words(path: Path, tmp_path: Path) -> None:
    """As the carry's trainer test: windows of 8, a step without a supervised position (the
    kv documents supervise their answers only) has no loss and is left out."""
    steps = 40
    sizes = list(TINY) if path == NOMEM else ["model.reader_width=16", "model.reader_heads=2"]
    cfg = load_config(
        path,
        overrides=[
            *train_overrides(steps),
            *sizes,
            "train.lr=3e-3",
            "train.unroll=8",
            "model.reader_rotary_pairs=[4,0]",
            "model.reader_carry=4",
            "model.reader_top_words=4",
        ],
    )
    seed_everything(0)
    trainer = make_trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    assert type(trainer) is Trainer
    history = trainer.fit(*datasets(), steps)
    assert len(history.steps) == steps
    assert all(math.isfinite(s.loss) == (s.loss_positions > 0) for s in history.steps)
    losses = [s.loss for s in history.steps if s.loss_positions > 0]
    assert len(losses) >= 15
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.05
    assert len(history.evals) == 1 and math.isfinite(history.evals[0]["loss"])
    payload = torch.load(history.checkpoints[-1], map_location="cpu", weights_only=False)
    carried = payload["extra"]["carried_state"]
    assert len(carried) == 8 and carried[6].dtype == torch.int64


def test_the_model_with_words_compiles_as_one_graph_and_matches_eager() -> None:
    from torch._dynamo.utils import counters

    keys = ["model.reader_carry=4", "model.reader_rotary_pairs=[4,0]", "model.reader_top_words=3"]
    cfg = load_config(NOMEM, [*TINY, *keys])
    seed_everything(0)
    model = build_model(cfg).train()
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    try:
        graphs = None
        states = [model.init_state(3, "cpu"), model.init_state(3, "cpu")]
        for i in range(3):
            tokens, reset = _text_stream(3, 8, 10 + i, 0.05)
            reset[:, 0] = i == 0  # the documents go on across the windows
            ctx = torch.zeros_like(tokens)
            results = []
            for j, forward in enumerate((model, compiled)):
                model.zero_grad(set_to_none=True)
                logits, states[j], _ = forward(tokens, ctx, states[j], reset)
                logits.square().mean().backward()
                grads = {n: p.grad.clone() for n, p in model.named_parameters()}
                results.append((logits.detach(), grads))
            (ref, ref_grads), (got, got_grads) = results
            assert _gap(got, ref) <= 1e-6
            assert all(_gap(got_grads[n], ref_grads[n]) <= 1e-6 for n in ref_grads)
            assert all(_gap(a.double(), b.double()) <= 1e-6 for a, b in zip(*states, strict=True))
            if graphs is not None:
                assert counters["stats"]["unique_graphs"] == graphs, "the next window recompiled"
            graphs = counters["stats"]["unique_graphs"]
    finally:
        torch._dynamo.reset()
