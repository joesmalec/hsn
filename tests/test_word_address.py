"""The word address of the event loop (``model.word_address``; HSN-T, opt-in candidate;
``docs/amendments/write_events.md``, "The word address").

At an event A5's two features read word codes: ``a_cur`` the word the event closes, ``a_prev``
the word of the event before. These tests check the rule against a plain per-row reading of
it, that ``none`` is the first build (the outputs written at 1096531), that a context that is
not text keeps A5's byte address bit for bit, the case fold, the carry (two windows equal one
long window), that nothing reads a later byte, the configuration, and that a run's audit
completes.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pytest
import torch
from hsn.config import HASH_OMIT_AT_DEFAULT, RunConfig, canonical_json, config_hash, load_config
from hsn.data.synthetic import BOS, EOS
from hsn.device import resolve, seed_everything
from hsn.experiments.protocols import ArmRun, run_protocol
from hsn.experiments.rules import parse_rules
from hsn.experiments.spec import memory_key, protocol_params
from hsn.model.hsn import address_history
from hsn.model.hsn_t import HSNT
from hsn.model.word_address import WORD_TABLES, state_width
from hsn.runner import build_model
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import (
    REF_A1,
    TINY_HSN_OVERRIDES,
    disk_overrides,
    restore_determinism,  # noqa: F401
    tiny_manifest,  # noqa: F401
)
from p1_15_support import datasets, hsn_config
from test_a6_feedback_detach import A5, _start
from test_hsn_t import CUT_C, READER, _gap, _model
from test_write_events import CATTN, HYBRID, NGRAM, RECIPE_A6, STE, _ev, _ev_cfg, _window_of
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "write_events_1096531.pt"
TEXT = 6  # the default text context (TinyStories)
KINDS = ["exact", "bag"]


def _wa(kind: str) -> str:
    return f"model.word_address={kind}"


def _word_cfg(kind: str, *extra: str) -> RunConfig:
    return _ev_cfg("word", *extra, _wa(kind))


def _ids(text: str) -> list[int]:
    return list(text.encode("latin-1"))


# --------------------------------------------------------------------------------------------------
# The rule as written, one row at a time
# --------------------------------------------------------------------------------------------------


def _letter(x: int) -> bool:
    return 65 <= x <= 90 or 97 <= x <= 122


def _fold(x: int) -> int:
    return x + 32 if 65 <= x <= 90 else x


def _row_keys(
    row: list[int], text: list[bool], reset: list[bool]
) -> dict[int, tuple[tuple[int, ...], tuple[int, ...] | None]]:
    """The events of a row and, per event, its key and the key of the event before (``None``:
    BOS, at a document start or when nothing came before), by the rule as written: it reads
    the whole row at once, so an apostrophe's place is looked up, not waited for."""
    T = len(row)

    def base(t: int) -> bool:
        return text[t] and (_letter(row[t]) or 48 <= row[t] <= 57)

    def joined(t: int) -> bool:
        return (
            text[t]
            and row[t] == 39
            and 0 < t < T - 1
            and not reset[t]
            and not reset[t + 1]
            and text[t - 1]
            and _letter(row[t - 1])
            and text[t + 1]
            and _letter(row[t + 1])
        )

    in_word = [base(t) or joined(t) for t in range(T)]
    closed: dict[int, tuple[int, ...]] = {}
    t = 0
    while t < T:
        if not in_word[t]:
            t += 1
            continue
        end = t
        while end + 1 < T and in_word[end + 1] and not reset[end + 1]:
            end += 1
        word = tuple(_fold(x) for x in row[t : end + 1])
        after = end + 1
        if after < T and not reset[after]:
            hangs = text[after] and row[after] == 39 and _letter(row[end])
            if not hangs:
                closed[after] = word
            elif after + 1 < T and not reset[after + 1]:
                closed[after + 1] = word  # one byte late: the apostrophe had to be placed
        t = end + 1
    out: dict[int, tuple[tuple[int, ...], tuple[int, ...] | None]] = {}
    before: tuple[int, ...] | None = None
    for t in range(T):
        if reset[t]:
            before = None
        if not (reset[t] or not text[t] or t in closed):
            continue
        key = closed.get(t, (_fold(row[t]) if text[t] else row[t],))
        out[t] = (key, before)
        before = key
    return out


def _code(scorer: torch.nn.Module, kind: str, key: tuple[int, ...] | None) -> Tensor:
    """The code of a key from the scorer's tables, by the formula."""
    codes: Tensor = scorer.byte_codes  # type: ignore[assignment]
    V, d = codes.shape
    if key is None:
        return codes[BOS]
    if len(key) == 1:
        return codes[key[0]]
    if kind == "exact":
        signs = torch.ones(d, dtype=codes.dtype)
        for i, x in enumerate(key):
            signs = signs * (1 - 2 * scorer.word_exact[min(i, WORD_TABLES - 1), x]).to(codes.dtype)
        return signs / math.sqrt(d)
    marked = [V, *key, V + 1]
    total = torch.zeros(d, dtype=torch.int64)
    for a, b in zip(marked, marked[1:], strict=False):
        total += (scorer.word_bag2[0, a] * scorer.word_bag2[1, b]).to(torch.int64)
    for a, b, c in zip(marked, marked[1:], marked[2:], strict=False):
        three = scorer.word_bag3
        total += (three[0, a] * three[1, b] * three[2, c]).to(torch.int64)
    vector = total.to(codes.dtype)
    return vector / vector.norm()


def _story(B: int, T: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    """Text-like rows: words of letters of both cases and digits, apostrophes between
    letters, after digits, doubled and at word ends, separators, bytes >= 128, BOS and EOS;
    document starts anywhere; row 1 under another context, row 2 text after another context."""
    g = torch.Generator().manual_seed(seed)

    def pick(pool: list[int], n: int) -> list[int]:
        return [pool[int(i)] for i in torch.randint(0, len(pool), (n,), generator=g)]

    wordy = [*_ids("abzLYQ"), *_ids("ae"), *_ids("07"), 39, 39]
    separators = [*_ids(" ,.\"'"), 32, 32, 16, 200, EOS, BOS]
    tokens = torch.empty(B, T, dtype=torch.int64)
    for b in range(B):
        row: list[int] = []
        while len(row) < T:
            row += pick(wordy, int(torch.randint(1, 8, (1,), generator=g)))
            row += pick(separators, int(torch.randint(1, 3, (1,), generator=g)))
        tokens[b] = torch.tensor(row[:T])
    ctx = torch.full((B, T), TEXT)
    reset = torch.rand(B, T, generator=g) < 0.06
    if B > 1:
        ctx[1] = 3
    if B > 2:
        ctx[2, : T // 3] = 2
        reset[2, T // 3] = True
    return tokens, ctx, reset


def _scorer(kind: str) -> torch.nn.Module:
    return _model(_word_cfg(kind)).scorer


def _words(scorer: torch.nn.Module, tokens: Tensor, ctx: Tensor, reset: Tensor, state=None):  # type: ignore[no-untyped-def]
    return scorer.word_codes(tokens, ctx == TEXT, reset, state)  # type: ignore[operator]


@pytest.mark.parametrize("kind", KINDS)
def test_events_and_codes_follow_the_word_rule(kind: str) -> None:
    scorer = _scorer(kind)
    assert scorer.byte_codes.dtype == torch.float64
    for seed in range(4):
        tokens, ctx, reset = _story(4, 96, seed)
        out = _words(scorer, tokens, ctx, reset)
        wide = 0
        for b in range(tokens.shape[0]):
            keys = _row_keys(tokens[b].tolist(), (ctx[b] == TEXT).tolist(), reset[b].tolist())
            assert out.events[b].nonzero().flatten().tolist() == sorted(keys), (seed, b)
            for t, (key, before) in keys.items():
                wide += len(key) > 1
                assert _gap(out.current[b, t], _code(scorer, kind, key)) <= 1e-12, (seed, b, t)
                assert _gap(out.previous[b, t], _code(scorer, kind, before)) <= 1e-12, (b, t)
        assert wide > 20
    assert bool(out.events[1].all())  # another context: every byte is its own word


def test_the_rule_on_a_sentence() -> None:
    """The binding, read off a sentence: a fact ``named lily`` is written at lily's event under
    ``named`` in ``a_prev``, and at the later event of ``named`` the same code is in ``a_cur``."""
    scorer = _scorer("exact")
    text = "named Lily. don't 7's 'a' lily said"
    tokens = torch.tensor([_ids(text)])
    ctx, reset = torch.full_like(tokens, TEXT), torch.zeros_like(tokens, dtype=torch.bool)
    reset[0, 0] = True
    out = _words(scorer, tokens, ctx, reset)
    keys = _row_keys(tokens[0].tolist(), [True] * len(text), reset[0].tolist())
    words = {t: bytes(key).decode() for t, (key, _) in keys.items()}
    # the document start, then one event per word at the byte after it; `'a'` closes one byte
    # late (the quote after it might have joined a letter), `7's` is 7 and then s
    assert words == {
        0: "n", 5: "named", 10: "lily", 17: "don't", 19: "7", 21: "s", 25: "a", 30: "lily"
    }  # fmt: skip
    assert out.events[0].nonzero().flatten().tolist() == sorted(words)
    code = {word: _code(scorer, "exact", tuple(_ids(word))) for word in ("named", "lily")}
    assert torch.equal(out.current[0, 5], out.previous[0, 10])  # named: read, and lily's write
    assert _gap(out.current[0, 5], code["named"]) <= 1e-12
    assert torch.equal(out.current[0, 10], out.current[0, 30])  # Lily and lily: one address
    assert _gap(out.current[0, 30], code["lily"]) <= 1e-12
    assert _gap(out.current[0, 10].norm(), torch.tensor(1.0)) <= 1e-12
    assert _gap(out.current[0, 10], out.current[0, 5]) > 0.1
    # after the last byte the word `said` is still open; the next window closes it
    more = _words(scorer, torch.tensor([_ids(" Sam")]), ctx[:, :4], reset[:, 1:5], out.state)
    assert more.events[0].tolist() == [True, False, False, False]
    assert _gap(more.current[0, 0], _code(scorer, "exact", tuple(_ids("said")))) <= 1e-12
    assert torch.equal(more.previous[0, 0], out.current[0, 30])


@pytest.mark.parametrize("kind", KINDS)
def test_one_address_across_case_and_for_an_unseen_word(kind: str) -> None:
    """The address reads fixed tables and the folded bytes only: ``Lily``, ``LILY`` and
    ``lily`` have one code, a word no run has seen has a unit code like any other, and no
    parameter stands behind either (the content path still reads the capital)."""
    cfg = _word_cfg(kind)
    model = _model(cfg)
    scorer = model.scorer
    rows = [_ids("lily qzxv "), _ids("Lily qzxv "), _ids("LILY Qzxv ")]
    tokens = torch.tensor(rows)
    ctx, reset = torch.full_like(tokens, TEXT), torch.zeros_like(tokens, dtype=torch.bool)
    out = _words(scorer, tokens, ctx, reset)
    assert out.events.tolist() == [[False] * 4 + [True] + [False] * 4 + [True]] * 3
    for t in (4, 9):
        assert torch.equal(out.current[0, t], out.current[1, t])
        assert torch.equal(out.current[0, t], out.current[2, t])
        assert _gap(out.current[0, t].norm(), torch.tensor(1.0)) <= 1e-12
        assert out.current.grad_fn is None and not out.current.requires_grad
    tables = {"exact": ["word_exact"], "bag": ["word_bag2", "word_bag3"]}[kind]
    buffers = dict(model.named_buffers())
    mine = {name.rsplit(".", 1)[-1]: value for name, value in buffers.items() if "word_" in name}
    assert sorted(mine) == tables and not any(v.is_floating_point() for v in mine.values())
    assert not any("word_" in name for name, _ in model.named_parameters())
    # the scorer's features of the three rows agree at the events, the content path does not
    state = _start(model, 3)
    pro = model._window_prologue(tokens, ctx, state, reset, "sparse", reset_first=False)
    address = slice(-2, None)  # a_cur, a_prev
    assert torch.equal(pro.scalars[0, 4, :, address], pro.scalars[1, 4, :, address])
    assert torch.equal(pro.scalars[0, 9, :, address], pro.scalars[2, 9, :, address])
    assert _gap(pro.z[0, 0], pro.z[1, 0]) > 1e-3
    # without the key the same model builds no table
    plain = _model(_ev_cfg("word"))
    assert not any("word_" in name for name, _ in plain.named_buffers())


# --------------------------------------------------------------------------------------------------
# none is the first build; another context keeps A5's byte address
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["soft", "ngram_cattn"])
def test_none_reproduces_the_first_build(name: str) -> None:
    """The event loop's outputs as commit 1096531 computed them (before the key existed), for
    two of the anchor configurations with ``write_events=word`` in evaluation mode."""
    extra = {"soft": (), "ngram_cattn": (STE, CUT_C, CATTN, *NGRAM)}[name]
    golden = torch.load(GOLDEN)[name]
    cfg = _ev_cfg("word", *extra, _wa("none"))
    model = _model(cfg).eval()
    assert sorted(k for k, _ in model.named_buffers()) == golden["buffers"]
    tokens, ctx, state, reset, carry = _window_of(cfg, model)
    with torch.no_grad():
        logits, final, info = model(tokens, ctx, state, reset, prev_tokens=carry)
    assert "address_carry" not in info
    assert _gap(logits[:, :, ::8], golden["logits"]) <= 1e-12
    assert _gap(final.h, golden["h"]) <= 1e-12 and _gap(final.q, golden["q"]) <= 1e-12
    assert torch.equal(info["event_valid"], golden["event_valid"])
    assert torch.equal(info["event_pos"], golden["event_pos"])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("extra", [(STE, CUT_C, CATTN), (STE, CUT_C, CATTN, *NGRAM)])
def test_another_context_keeps_the_byte_address(kind: str, extra: tuple[str, ...]) -> None:
    """Under a context that is not text every byte is its own word and an event: the address
    features are A5's bit for bit, and the model is ``write_events=all`` without the key
    (which is the loop over bytes), over two windows with the carry."""
    cfg = _word_cfg(kind, *extra)
    words, every = _model(cfg).eval(), _model(_ev_cfg("all", *extra)).eval()
    theirs = dict(every.named_parameters())
    for key, value in words.named_parameters():
        assert torch.equal(value, theirs[key]), key
    assert torch.equal(words.scorer.byte_codes, every.scorer.byte_codes)
    tokens, ctx, state, reset, _ = _window_of(cfg, words)
    ctx = ctx.masked_fill(ctx == TEXT, 0)  # no text context in the window
    carry = byte_carry = None
    with torch.no_grad():
        for window in (tokens, tokens.flip(1)):
            args = (window, ctx, state, reset, "sparse")
            pro = words._window_prologue(*args, reset_first=True, prev_tokens=carry)
            ref = every._window_prologue(*args, reset_first=True, prev_tokens=byte_carry)
            assert bool(pro.events.all()) and torch.equal(pro.scalars, ref.scalars)
            logits, after, info = words(window, ctx, state, reset, prev_tokens=carry)
            expected, ref_after, ref_info = every(window, ctx, state, reset, prev_tokens=byte_carry)
            assert torch.equal(logits, expected)
            assert torch.equal(after.h, ref_after.h) and torch.equal(after.q, ref_after.q)
            carry = address_history(cfg.model, window, carry, info)
            byte_carry = address_history(every.cfg, window, byte_carry, ref_info)
            n = byte_carry.reshape(len(window), -1).shape[1]
            assert carry.shape == (len(window), n + state_width(cfg.model.d))
            assert torch.equal(carry[:, :n], byte_carry.reshape(len(window), -1))
            state = after
    with pytest.raises(ValueError, match="pass the info"):
        address_history(cfg.model, tokens, None)
    with pytest.raises(ValueError, match="address carry"):
        words(tokens, ctx, state, reset, prev_tokens=byte_carry)


# --------------------------------------------------------------------------------------------------
# The carry, and what a position may read
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_two_windows_equal_one_long_window(kind: str) -> None:
    """Any cut of a window into windows gives the same events and the same codes at them:
    inside a word, before and after an apostrophe, in windows of a single byte."""
    scorer = _scorer(kind)
    g = torch.Generator().manual_seed(11)
    for seed in range(3):
        tokens, ctx, reset = _story(4, 80, 20 + seed)
        T = tokens.shape[1]
        whole = _words(scorer, tokens, ctx, reset)
        for trial in range(6):
            cuts = sorted({int(c) for c in torch.randint(1, T, (1 + 2 * trial,), generator=g)})
            if trial == 5:
                cuts = list(range(1, T))  # one byte per window
            state, events, current, previous = None, [], [], []
            for a, b in zip([0, *cuts], [*cuts, T], strict=True):
                part = _words(scorer, tokens[:, a:b], ctx[:, a:b], reset[:, a:b], state)
                state = part.state
                events.append(part.events), current.append(part.current)
                previous.append(part.previous)
            at = whole.events
            assert torch.equal(torch.cat(events, 1), at), (seed, cuts)
            assert torch.equal(torch.cat(current, 1)[at], whole.current[at]), (seed, cuts)
            assert torch.equal(torch.cat(previous, 1)[at], whole.previous[at]), (seed, cuts)
            assert torch.equal(state, whole.state), (seed, cuts)


@pytest.mark.parametrize("kind", KINDS)
def test_the_address_reads_no_later_byte(kind: str) -> None:
    scorer = _scorer(kind)
    tokens, ctx, reset = _story(4, 64, 31)
    other, _, _ = _story(4, 64, 32)
    whole = _words(scorer, tokens, ctx, reset)
    for t0 in range(1, 64):
        later = torch.cat([tokens[:, :t0], other[:, t0:]], dim=1)
        out = _words(scorer, later, ctx, reset)
        at = whole.events[:, :t0]
        assert torch.equal(out.events[:, :t0], at), t0
        assert torch.equal(out.current[:, :t0][at], whole.current[:, :t0][at]), t0
        assert torch.equal(out.previous[:, :t0][at], whole.previous[:, :t0][at]), t0


@pytest.mark.parametrize("kind", KINDS)
def test_the_model_reads_no_later_byte_and_no_earlier_document(kind: str) -> None:
    cfg = _word_cfg(kind, STE, CUT_C, CATTN, *NGRAM)
    model = _model(cfg).eval()
    tokens, _, reset = _story(3, cfg.train.unroll, 41)
    tokens[0, :2], tokens[0, 2] = torch.tensor(_ids("ab")), 32
    ctx = torch.full_like(tokens, TEXT)
    reset[:] = False
    reset[0, 6] = True
    state, t0 = _start(model, 3), 5
    with torch.no_grad():
        first, _, first_info = model(tokens[:, :4], ctx[:, :4], state, reset[:, :4])
        carry = address_history(cfg.model, tokens[:, :4], None, first_info)
        logits, _, info = model(tokens, ctx, state, reset, prev_tokens=carry)
        assert not bool(info["event_valid"].all())
        later = tokens.clone()
        later[:, t0::2], later[:, t0 + 1 :: 2] = 98, 32
        changed, _, _ = model(later, ctx, state, reset, prev_tokens=carry)
        assert _gap(changed[:, :t0], logits[:, :t0]) <= 1e-12
        assert _gap(changed[:, t0:], logits[:, t0:]) > 1e-6
        # another document, state and carried address before the reset of row 0
        before = tokens.clone()
        before[0, :6] = torch.tensor(_ids("Zq 7'x"))
        other = address_history(cfg.model, tokens[:, :4], None, {"address_carry": carry.flip(0)})
        changed, _, _ = model(before, ctx, _start(model, 3, seed=5), reset, prev_tokens=other)
        assert _gap(changed[0, :6], logits[0, :6]) > 1e-6
        assert _gap(changed[0, 6:], logits[0, 6:]) <= 1e-12


def test_trainer_carries_and_checkpoints_the_word_state(tmp_path: Path) -> None:
    steps = 4
    cfg = hsn_config(
        steps,
        *A5,
        CUT_C,
        "train.soft_until_step=0",
        *READER,
        *NGRAM,
        _ev("word"),
        _wa("bag"),
        "model.word_contexts=[1,2,3,4,5]",
        "train.ckpt_every=2",
    )

    def trainer_in(run_dir: Path) -> TBPTTTrainer:
        seed_everything(0)
        return TBPTTTrainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=run_dir)

    trainer = trainer_in(tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    assert len(history.steps) == steps and all(math.isfinite(s.loss) for s in history.steps)
    carry = trainer._prev_tokens
    assert carry is not None and carry.dtype == torch.int64
    assert tuple(carry.shape) == (cfg.train.batch, 2 + state_width(cfg.model.d))
    again = trainer_in(tmp_path / "again")
    again.resume(sorted((tmp_path / "run").glob("step_*.pt"))[-1])
    assert again._prev_tokens is not None and torch.equal(again._prev_tokens.cpu(), carry.cpu())


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_key_is_opt_in_and_the_default_keeps_its_hash() -> None:
    word = load_config(RECIPE_A6, [*HYBRID, _ev("word")])
    assert word.model.word_address == "none" and word.model.word_contexts == [TEXT]
    dumped = json.loads(canonical_json(word))["model"]
    assert "word_address" not in dumped and "word_contexts" not in dumped
    assert HASH_OMIT_AT_DEFAULT[-2:] == ("model.word_address", "model.word_contexts")
    named = load_config(RECIPE_A6, [*HYBRID, _ev("word"), _wa("none")])
    exact = load_config(RECIPE_A6, [*HYBRID, _ev("word"), _wa("exact")])
    bag = load_config(RECIPE_A6, [*HYBRID, _ev("word"), _wa("bag")])
    assert config_hash(named) == config_hash(word)
    assert len({config_hash(word), config_hash(exact), config_hash(bag)}) == 3
    protocol = {"name": "reads_state", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
    assert memory_key(named, "pdg", protocol) == memory_key(word, "pdg", protocol)
    assert memory_key(exact, "pdg", protocol) != memory_key(word, "pdg", protocol)
    for kind in ("byte", "all"):
        with pytest.raises(ValueError, match="model.write_events=word"):
            load_config(RECIPE_A6, [*HYBRID, _ev(kind), _wa("exact")])
    with pytest.raises(ValueError, match="word_contexts"):
        load_config(RECIPE_A6, [*HYBRID, _ev("word"), _wa("bag"), "model.word_contexts=[6,99]"])


# --------------------------------------------------------------------------------------------------
# A run's audit completes
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["none", *KINDS])
def test_a_run_with_word_events_finishes_its_audit(
    kind: str,
    tiny_manifest: Path,  # noqa: F811
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``train_eval`` end to end on a tiny CPU run (the first build failed here, in the stage
    ``trained_state``): the state statistics, which need ``capture=True``, are skipped with a
    logged note and a flag in the record; the run trains, evaluates and carries the address."""
    cfg = load_config(
        REF_A1,
        [
            *TINY_HSN_OVERRIDES,
            *disk_overrides(tiny_manifest, steps=6),
            *A5,
            CUT_C,
            "train.soft_until_step=0",
            *READER,
            _ev("word"),
            _wa(kind),
            "model.word_contexts=[1,2]",  # the tiny tasks' contexts read as text
        ],
    )
    arm = ArmRun(
        cfg=cfg,
        family="hsn_t",
        variant="pdg",
        variant_params={},
        params=protocol_params({"name": "train_eval", "diagnostics": False, "diag_docs": 8}),
        run_dir=tmp_path / "run",
        rules=parse_rules(None),
        constants={"chance": 0.02},
        on_progress=lambda _: None,
    )
    with caplog.at_level(logging.WARNING, logger="hsn.experiments.diagnostics"):
        record = run_protocol(arm)
    assert "failure" not in record and record["final"]["steps"] == 6
    assert math.isfinite(record["final"]["dev_loss"])
    trained = record["trained_state"]
    assert trained["state_stats_skipped"] == 1.0 and "edge_logit/absmax" in trained
    assert any("state statistics are skipped" in r.getMessage() for r in caplog.records)
    assert isinstance(HSNT.from_config(cfg), HSNT)
