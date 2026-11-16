"""The loop over events (``model.write_events``; HSN-T only, opt-in candidate;
``docs/amendments/write_events.md``).

With ``word`` the memory runs only at word ends and at document starts, and the position loop
runs over those events instead of over every byte; the decoder still reads every byte. These
tests check the event rule, the slots, and the anchor of the whole build: ``all`` (every byte
an event, through the same event-slot path) must equal the loop over bytes in float64, in the
logits, the final state and the gradients. Then what ``word`` may depend on (no later byte,
no earlier document), that padding slots leave the state alone, the losses, the refusals, the
compiled window and the trainer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import (
    HASH_OMIT_AT_DEFAULT,
    RunConfig,
    canonical_json,
    config_hash,
    load_config,
    split_window_scope_error,
)
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import memory_key
from hsn.losses import assemble_terms, auxiliary_terms, balance_loss
from hsn.model.hsn import BOS, EVENT_BLOCK, HSN, event_block, event_positions
from hsn.model.hsn_t import HSNT
from hsn.model.state import CellGraphState
from hsn.runner import build_model
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, hsn_config
from test_a6_feedback_detach import A5, _cfg, _start
from test_a8_fast_cells import COMPILED
from test_hsn_t import CUT_C, FAMILY, PREV, READER, SHORT, _gap, _model
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"

HYBRID = [FAMILY, "model.reader=transformer", "model.k_select=12"]
CATTN = "model.cell_attention_heads=2"
NGRAM = ("model.address_ngram=2", "model.address_history=ngram")
NOMEM = REPO / "configs" / "experiments" / "hsn_t_nomem.yaml"
STE = "train.gate_mode=ste"
T_LONG = "train.unroll=40"  # more than one block of event slots


def _ev(kind: str) -> str:
    return f"model.write_events={kind}"


def _ev_cfg(kind: str, *extra: str) -> RunConfig:
    return _cfg(*A5, *READER, *extra, _ev(kind))


def _text(cfg: RunConfig, seed: int = 3) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """A window of text-like bytes: words of one to five letters or digits, single and double
    separators (space, comma, the newline marker 16), some bytes >= 128; a document start in
    the middle of rows 0 and 1; ``prev`` is the byte carried into the window (a letter in row
    0, so its first byte can end a word). ``(tokens, ctx, reset, prev)``."""
    g = torch.Generator().manual_seed(seed)
    B, T = cfg.train.batch, cfg.train.unroll
    letters = torch.tensor([97, 98, 122, 65, 90, 48, 57, 200, 255])
    separators = torch.tensor([32, 44, 16, 46, 17])
    tokens = torch.empty(B, T, dtype=torch.int64)
    for b in range(B):
        row: list[int] = []
        while len(row) < T:
            n = int(torch.randint(1, 6, (1,), generator=g))
            row += letters[torch.randint(0, len(letters), (n,), generator=g)].tolist()
            n = int(torch.randint(1, 3, (1,), generator=g))
            row += separators[torch.randint(0, len(separators), (n,), generator=g)].tolist()
        tokens[b] = torch.tensor(row[:T])
    tokens[0, 0] = 32  # row 0 starts with a separator after the carried letter
    ctx = torch.randint(0, cfg.model.n_context, (B, T), generator=g)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[0, T // 2] = True
    reset[1, 5] = True
    reset[2, 0] = True
    prev = torch.tensor([97, 32, 98][:B] + [32] * max(0, B - 3))
    return tokens, ctx, reset, prev


def _carry(cfg: RunConfig, prev: Tensor) -> Tensor:
    """``prev`` as the window's ``prev_tokens``: ``[B]``, or ``[B, n]`` with A9's history."""
    if cfg.model.address_history == "byte":
        return prev
    older = torch.full((prev.shape[0], cfg.model.address_ngram - 1), 101)
    return torch.cat([prev[:, None], older], dim=1)


def _pair(*extra: str) -> tuple[RunConfig, HSNT, HSNT]:
    """The same HSN-T twice: the loop over bytes and ``all`` (every byte an event slot)."""
    cfg = _ev_cfg("byte", *extra)
    byte, every = _model(cfg), _model(_ev_cfg("all", *extra))
    for (name, a), (_, b) in zip(byte.named_parameters(), every.named_parameters(), strict=True):
        assert torch.equal(a, b), name
    return cfg, byte, every


# --------------------------------------------------------------------------------------------------
# The event rule and the slots
# --------------------------------------------------------------------------------------------------


def test_events_are_word_ends_and_document_starts() -> None:
    #          a   b   ' ' ' ' c   ,   1   é    \n  \t  Z  ' '
    row = [97, 98, 32, 32, 99, 44, 49, 233, 16, 17, 90, 32]
    tokens = torch.tensor([row, row])
    previous = torch.tensor([[BOS, *row[:-1]], [122, *row[:-1]]])
    reset = torch.zeros(2, len(row), dtype=torch.bool)
    reset[1, 4] = True
    previous = previous.masked_fill(reset, BOS)
    events = event_positions("word", tokens, previous, reset)
    #                  a      b      sp    sp     c      ,     1      é      \n    \t     Z      sp
    assert events[0].tolist() == [
        False, False, True, False, False, True, False, False, True, False, False, True
    ]  # fmt: skip
    # row 1: the same, plus the document start at column 4 (a letter, not a word end)
    assert events[1].tolist() == [
        False, False, True, False, True, True, False, False, True, False, False, True
    ]  # fmt: skip
    # the byte carried into the window: a first byte that is a separator ends its word
    first = torch.tensor([[32], [32], [97]])
    carried = torch.tensor([[97], [32], [97]])
    none = torch.zeros(3, 1, dtype=torch.bool)
    assert event_positions("word", first, carried, none).flatten().tolist() == [True, False, False]
    assert bool(event_positions("all", tokens, previous, reset).all())


def test_slots_hold_each_rows_events_in_order() -> None:
    cfg = _ev_cfg("word", T_LONG)
    model = _model(cfg)
    tokens, ctx, reset, prev = _text(cfg)
    B, T = tokens.shape
    state = _start(model, B)
    pro = model._window_prologue(
        tokens, ctx, state, reset, model._window_mode(), reset_first=True, prev_tokens=prev
    )
    assert pro.events is not None and pro.positions is not None and pro.counters is None
    shifted = torch.cat([prev[:, None], tokens[:, :-1]], dim=1).masked_fill(reset, BOS)
    assert torch.equal(pro.events, event_positions("word", tokens, shifted, reset))
    assert bool(pro.events[0, 0]) and bool(pro.events[reset].all())
    block = event_block(T)
    assert block == 10 and event_block(256) == EVENT_BLOCK == 16 and event_block(12) == 12
    ev = model._event_slots(pro, reset, block)
    count = pro.events.sum(dim=1)
    E = ev.valid.shape[1]
    assert E % block == 0 and E - block < int(count.max()) <= E <= T
    for b in range(B):
        n = int(count[b])
        columns = pro.events[b].nonzero().flatten()
        assert ev.valid[b].tolist() == [True] * n + [False] * (E - n)
        assert torch.equal(ev.where[b, :n], columns)
        assert not bool(pro.events[b, ev.where[b, n:]].any())  # padding: columns without event
        assert not bool(ev.reset[b, n:].any())
        assert torch.equal(ev.z[b, :n], pro.z[b, columns])
        assert torch.equal(ev.context[b, :n], pro.context[b, columns])
        assert torch.equal(ev.scalars[b, :n], pro.scalars[b, columns])
        assert torch.equal(ev.pos[b, :n], pro.positions[b, columns])
        for t in range(T):
            at = int(pro.events[b, : t + 1].sum())
            assert int(ev.at[b, t]) == at
            assert int(ev.before[b, t]) == at - int(pro.events[b, t])
    # a window without any event still has one block of (padding) slots
    quiet = torch.full_like(tokens, 97)
    none = torch.zeros_like(reset)
    pro = model._window_prologue(
        quiet, ctx, state, none, model._window_mode(), reset_first=True, prev_tokens=prev
    )
    ev = model._event_slots(pro, none, block)
    assert ev.valid.shape[1] == block and not bool(ev.valid.any())


# --------------------------------------------------------------------------------------------------
# The anchor: every byte an event equals the loop over bytes
# --------------------------------------------------------------------------------------------------

ANCHOR = {
    "soft": (),
    "ste": (STE,),
    "ste_cut": (STE, CUT_C),
    "ste_cattn": (STE, CUT_C, CATTN),
    "soft_cattn": (CATTN,),
    "ngram_cattn": (STE, CUT_C, CATTN, *NGRAM),
    "short": (STE, *SHORT[len(READER) :], CATTN),
    "reader_prev": (STE, *PREV[len(READER) :]),
    "clip": (STE, CATTN, "train.state_grad_clip=1e-3"),
    "ratio": (STE, CUT_C, CATTN, "train.state_grad_ratio=2.0"),
    "long": (STE, CUT_C, CATTN, T_LONG),
}


def _window_of(
    cfg: RunConfig, model: HSNT
) -> tuple[Tensor, Tensor, CellGraphState, Tensor, Tensor]:
    tokens, ctx, reset, prev = _text(cfg)
    return tokens, ctx, _start(model, tokens.shape[0]), reset, _carry(cfg, prev)


@pytest.mark.parametrize("name", list(ANCHOR))
@pytest.mark.parametrize("training", [True, False])
def test_all_equals_the_loop_over_bytes_forward(name: str, training: bool) -> None:
    cfg, byte, every = _pair(*ANCHOR[name])
    byte.train(training), every.train(training)
    tokens, ctx, state, reset, carry = _window_of(cfg, byte)
    T = tokens.shape[1]
    with torch.no_grad():
        ref_logits, ref_state, ref = byte(tokens, ctx, state, reset, prev_tokens=carry)
        logits, new_state, info = every(tokens, ctx, state, reset, prev_tokens=carry)
    assert logits.dtype == torch.float64
    assert _gap(logits, ref_logits) <= 1e-12
    assert _gap(new_state.h, ref_state.h) <= 1e-12 and _gap(new_state.q, ref_state.q) <= 1e-12
    assert torch.equal(new_state.age, ref_state.age) and torch.equal(new_state.pos, ref_state.pos)
    assert bool(info["event_valid"].all()) and info["event_valid"].shape[1] == T
    assert torch.equal(info["event_pos"], torch.arange(T).expand_as(info["event_pos"]))
    for key in ("z", "z_hat", "z_target", "eps", "u", "eps_abs_mean", "alpha_entropy"):
        assert _gap(info[key], ref[key]) <= 1e-12, key
    assert _gap(info["cost"], ref["cost"]) <= 1e-6  # reduced in fp32
    assert _gap(info["state_abs_max"], ref["state_abs_max"]) <= 1e-12
    for key in ("cell_idx", "unit_idx", "valid", "explored", "refreshed", "dense_fallback"):
        assert (key in info) == (key in ref)
        if key in ref:
            assert torch.equal(info[key], ref[key]), key
    if "unit_logits" in ref:
        assert _gap(info["unit_logits"], ref["unit_logits"]) <= 1e-12


def _all_grads(
    model: HSNT, window: tuple[Tensor, Tensor, CellGraphState, Tensor, Tensor]
) -> dict[str, Tensor | None]:
    """The gradients of a fixed functional of a window (logits, the predictions, the final
    state) into every parameter and into the carried state."""
    tokens, ctx, state, reset, carry = window
    h = state.h.clone().requires_grad_(True)
    q = state.q.clone().requires_grad_(True)
    start = CellGraphState._make(h=h, q=q, age=state.age, pos=state.pos)
    model.zero_grad(set_to_none=True)
    logits, final, info = model(tokens, ctx, start, reset, prev_tokens=carry)
    loss = logits.square().mean() + info["z_hat"].square().mean() + info["u"].square().mean()
    (loss + final.h.square().sum() + final.q.square().sum()).backward()
    grads: dict[str, Tensor | None] = {n: p.grad for n, p in model.named_parameters()}
    return {**grads, "start.h": h.grad, "start.q": q.grad}


@pytest.mark.parametrize("name", list(ANCHOR))
def test_all_equals_the_loop_over_bytes_gradients(name: str) -> None:
    cfg, byte, every = _pair(*ANCHOR[name])
    window = _window_of(cfg, byte)
    ref, got = _all_grads(byte, window), _all_grads(every, window)
    assert set(ref) == set(got)
    reached = 0
    for key, r in ref.items():
        g = got[key]
        assert (r is None) == (g is None), key
        if r is not None and g is not None:
            assert _gap(g, r) <= 1e-12 * max(1.0, float(r.abs().max())), key
            reached += int(float(r.abs().sum()) > 0.0)
    assert reached > 10


# --------------------------------------------------------------------------------------------------
# word: what a byte may depend on, and the slots that are padding
# --------------------------------------------------------------------------------------------------

WORD = {
    "ste": (STE, CUT_C),
    "cattn": (STE, CUT_C, CATTN),
    "soft_cattn": (CATTN,),
    "short": (STE, CUT_C, CATTN, *SHORT[len(READER) :]),
    "long_ngram": (STE, CUT_C, CATTN, T_LONG, *NGRAM),
}


def _swap(tokens: Tensor) -> Tensor:
    """Other bytes with other events: every letter becomes a space, every separator a letter."""
    wordy = ((tokens >= 48) & (tokens <= 57)) | (tokens >= 65)
    return torch.where(wordy, torch.full_like(tokens, 32), torch.full_like(tokens, 98))


@pytest.mark.parametrize("name", list(WORD))
@pytest.mark.parametrize("quiet", [False, True])
def test_word_depends_on_no_later_byte_and_no_earlier_document(name: str, quiet: bool) -> None:
    """``quiet``: no word ends after the reset of row 0, so the reset is the row's last event."""
    cfg = _ev_cfg("word", *WORD[name])
    model = _model(cfg)
    tokens, ctx, state, reset, carry = _window_of(cfg, model)
    B, T = tokens.shape
    k, t0 = T // 2, 8
    assert bool(reset[0, k])
    if quiet:
        tokens[0, k:] = 97
    with torch.no_grad():
        logits, final, info = model(tokens, ctx, state, reset, prev_tokens=carry)
        assert not bool(info["event_valid"].all())  # some slots are padding
        later = tokens.clone()
        later[:, t0::2], later[:, t0 + 1 :: 2] = 98, 32  # other bytes, a word end every second
        changed, _, other = model(later, ctx, state, reset, prev_tokens=carry)
        assert not torch.equal(other["event_valid"].sum(1), info["event_valid"].sum(1))
        assert _gap(changed[:, :t0], logits[:, :t0]) <= 1e-12
        assert _gap(changed[:, t0:], logits[:, t0:]) > 1e-6
        # another document before the reset of row 0: other bytes, state and carried bytes
        before = tokens.clone()
        before[0, :k] = _swap(before[0, :k])
        changed, moved, _ = model(
            before, ctx, _start(model, B, seed=5), reset, prev_tokens=_swap(carry)
        )
        assert _gap(changed[0, :k], logits[0, :k]) > 1e-6
        assert _gap(changed[0, k:], logits[0, k:]) <= 1e-12
        assert _gap(moved.h[0], final.h[0]) <= 1e-12 and _gap(moved.q[0], final.q[0]) <= 1e-12
        assert torch.equal(moved.age[0], final.age[0]) and int(moved.pos[0]) == T - k
        # the carried state is read where the document continues (row 1 before its reset)
        assert _gap(changed[1, :5], logits[1, :5]) > 1e-6


def _rows(state: CellGraphState, rows: slice) -> CellGraphState:
    return CellGraphState._make(
        h=state.h[rows], q=state.q[rows], age=state.age[rows], pos=state.pos[rows]
    )


@pytest.mark.parametrize("name", ["cattn", "soft_cattn"])
def test_padding_slots_leave_the_state_unchanged(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _ev_cfg("word", *WORD[name], T_LONG)
    model = _model(cfg)
    tokens, ctx, state, reset, carry = _window_of(cfg, model)
    B, T = tokens.shape
    tokens[2], reset[2] = 97, False  # row 2: one long word, no event in the window
    state = CellGraphState._make(h=state.h, q=state.q, age=state.age + 3, pos=state.pos + 7)
    with torch.no_grad():
        logits, final, info = model(tokens, ctx, state, reset, prev_tokens=carry)
    count = info["event_valid"].sum(dim=1)
    E = info["event_valid"].shape[1]
    assert int(count[2]) == 0 and 0 < int(count.max()) < E
    # a row without an event: h, q and age bit for bit as they came; pos counts its bytes
    assert torch.equal(final.h[2], state.h[2]) and torch.equal(final.q[2], state.q[2])
    assert torch.equal(final.age[2], state.age[2]) and int(final.pos[2]) == 7 + T
    assert int(final.pos[0]) == T - T // 2 and int(final.pos[1]) == T - 5
    # age counts events: after its reset row 0 saw `after` events, and no cell is older
    late = info["event_valid"][0] & (info["event_pos"][0] >= T // 2)
    after = int(late.sum())
    assert int(final.age[0].max()) <= after < T - T // 2
    if name == "cattn":  # hard routing: some cell was not selected by the last event
        assert int(final.age[0].max()) > 0
    # the records of padding slots hold no valid selection
    assert not bool(info["valid"][~info["event_valid"]].any())
    assert bool(info["valid"][info["event_valid"]].any())
    with torch.no_grad():
        # more padding slots (a block as long as the window) change nothing
        monkeypatch.setattr("hsn.model.hsn.EVENT_BLOCK", T)
        wide, wide_final, wide_info = model(tokens, ctx, state, reset, prev_tokens=carry)
        assert wide_info["event_valid"].shape[1] == T > E
        assert _gap(wide, logits) <= 1e-12 and _gap(wide_final.h, final.h) <= 1e-12
        assert torch.equal(wide_final.age, final.age) and torch.equal(wide_final.pos, final.pos)
        monkeypatch.setattr("hsn.model.hsn.EVENT_BLOCK", EVENT_BLOCK)
        # and a row computes the same alone as beside rows with more events
        for rows in (slice(2, 3), slice(0, 2)):
            alone, alone_final, _ = model(
                tokens[rows], ctx[rows], _rows(state, rows), reset[rows], prev_tokens=carry[rows]
            )
            assert _gap(alone, logits[rows]) <= 1e-12
            assert _gap(alone_final.h, final.h[rows]) <= 1e-12


def test_bytes_before_the_first_event_read_the_starting_state() -> None:
    """Row 2 has no event: every byte reads the window's starting ``q`` and cells, and zeros
    where ``r`` stands, so another start state changes its logits and other bytes of *other*
    rows do not."""
    cfg = _ev_cfg("word", STE, CUT_C, CATTN)
    model = _model(cfg)
    tokens, ctx, state, reset, carry = _window_of(cfg, model)
    tokens[2], reset[2] = 97, False
    seen: list[Tensor] = []
    hook = model.decoder.register_forward_hook(lambda _m, inputs, _o: seen.append(inputs[0]))
    with torch.no_grad():
        model(tokens, ctx, state, reset, prev_tokens=carry)
    hook.remove()
    d, w = cfg.model.d, cfg.model.reader_width
    read = seen[0]  # [B, T, q ; u ; r ; c ; m]
    assert read.shape[-1] == 4 * d + w
    assert torch.equal(read[2, :, :d], state.q[2].expand(tokens.shape[1], d))
    assert float(read[2, :, 2 * d : 3 * d].abs().max()) == 0.0
    assert float(read[2, :, -d:].abs().min()) > 0.0  # m_t reads the starting cells
    assert float(read[0, :, 2 * d : 3 * d].abs().max()) > 0.0


# --------------------------------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------------------------------


def _batch(tokens: Tensor) -> dict[str, Tensor]:
    B, T = tokens.shape
    doc = torch.zeros(B, T, dtype=torch.int64)
    doc[1, T // 2 :] = -1  # padding in the second half of row 1
    return {
        "tokens": tokens,
        "targets": tokens.roll(-1, dims=1).contiguous(),
        "loss_mask": doc >= 0,
        "doc": doc,
    }


def test_prediction_loss_is_taken_at_the_events() -> None:
    cfg = _ev_cfg("word", STE, CUT_C, "train.lambda_p=0.5", T_LONG)
    model = _model(cfg)
    tokens, ctx, state, reset, carry = _window_of(cfg, model)
    logits, _, info = model(tokens, ctx, state, reset, prev_tokens=carry)
    batch = _batch(tokens)
    terms = assemble_terms({"logits": logits, **info}, batch, cfg)
    live, where = info["event_valid"], info["event_pos"]
    assert info["z_hat"].shape[:2] == live.shape and info["z"].shape[:2] == tokens.shape
    errors = []
    for b in range(tokens.shape[0]):
        for e in range(live.shape[1]):
            t = int(where[b, e])
            if bool(live[b, e]) and int(batch["doc"][b, t]) >= 0:
                errors.append((info["z_hat"][b, e] - info["z_target"][b, t]).float().square())
    assert 0 < len(errors) < int(live.sum())  # padding slots and padding positions are left out
    expected = 0.5 * torch.stack(errors).mean()
    assert float((terms["pred"] - expected).detach().abs()) <= 1e-6 * float(expected.detach())
    assert terms["pred"].requires_grad
    # the next-byte loss stays per byte
    mask = batch["loss_mask"]
    task = torch.nn.functional.cross_entropy(logits[mask].float(), batch["targets"][mask])
    assert float((terms["task"] - task).detach().abs()) <= 1e-6
    # a window without an event: a zero term with a gradient path, not an error
    quiet = torch.full_like(tokens, 97)
    none = torch.zeros_like(reset)
    logits, _, info = model(quiet, ctx, state, none, prev_tokens=carry)
    assert not bool(info["event_valid"].any())
    terms = assemble_terms({"logits": logits, **info}, _batch(quiet), cfg)
    assert float(terms["pred"].detach()) == 0.0 and terms["pred"].requires_grad


def test_terms_of_all_equal_the_loop_over_bytes() -> None:
    extra = (STE, CUT_C, "train.lambda_p=0.5", "train.aux_enabled=[balance]", "train.lambda_b=0.1")
    cfg, byte, every = _pair(*extra)
    tokens, ctx, state, reset, carry = _window_of(cfg, byte)
    batch = _batch(tokens)
    terms = []
    for model in (byte, every):
        logits, _, info = model(tokens, ctx, state, reset, prev_tokens=carry)
        extras = auxiliary_terms(info, cfg, model.cell.graph)
        terms.append(assemble_terms({"logits": logits, **info}, batch, cfg, extras))
    assert list(terms[0]) == list(terms[1]) == ["task", "pred", "balance"]
    for key, value in terms[0].items():
        gap = float((terms[1][key] - value).detach().abs())
        assert gap <= 1e-6 * max(1.0, float(value.detach().abs())), key


def test_balance_is_taken_over_the_valid_slots() -> None:
    cfg = _ev_cfg("word", STE, CUT_C, "train.aux_enabled=[balance]", "train.lambda_b=0.1", T_LONG)
    model = _model(cfg)
    tokens, ctx, state, reset, carry = _window_of(cfg, model)
    _, _, info = model(tokens, ctx, state, reset, prev_tokens=carry)
    live = info["event_valid"]
    assert not bool(live.all())
    expected = balance_loss(info["unit_logits"][live], info["unit_idx"][live], info["valid"][live])
    got = auxiliary_terms(info, cfg, model.cell.graph)["balance"]
    assert float((got - expected).detach().abs()) == 0.0
    everywhere = balance_loss(info["unit_logits"], info["unit_idx"], info["valid"])
    assert float((got - everywhere).detach().abs()) > 0.0


# --------------------------------------------------------------------------------------------------
# Configuration and refusals
# --------------------------------------------------------------------------------------------------


def test_key_is_opt_in_and_the_default_keeps_its_hash() -> None:
    base = load_config(RECIPE_A6)
    assert base.model.write_events == "byte"
    assert "write_events" not in json.loads(canonical_json(base))["model"]
    assert HASH_OMIT_AT_DEFAULT[-1] == "model.write_events"
    hybrid = load_config(RECIPE_A6, HYBRID)
    named = load_config(RECIPE_A6, [*HYBRID, _ev("byte")])
    assert config_hash(named) == config_hash(hybrid)
    assert "write_events" not in hybrid.model.model_dump(exclude_defaults=True)
    word = load_config(RECIPE_A6, [*HYBRID, _ev("word")])
    every = load_config(RECIPE_A6, [*HYBRID, _ev("all")])
    assert len({config_hash(hybrid), config_hash(word), config_hash(every)}) == 3
    assert json.loads(canonical_json(word))["model"]["write_events"] == "word"
    # the memory keys of the experiment database, as main computed them before the key existed
    protocol = {"name": "reads_state", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
    assert memory_key(base, "pdg", protocol) == "c33ec88833506533"
    assert memory_key(hybrid, "pdg", protocol) == "c8bcf4024c9285fe"
    assert memory_key(word, "pdg", protocol) != memory_key(hybrid, "pdg", protocol)


WORD_ON = [*HYBRID, _ev("word")]


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        (RECIPE_A6, [_ev("word")], "set model.reader=transformer"),  # the HSN: every byte
        (RECIPE_A6, [*HYBRID, "model.write_events=sentence"], "'byte', 'word' or 'all'"),
        (NOMEM, [_ev("word")], "model.memory=false has none"),
        (RECIPE_A6, [*WORD_ON, "model.selected_readout=false"], "needs model.key_address=true"),
        (RECIPE_A6, [*WORD_ON, "model.fast_cells=2"], r"model.fast_readout \(A8\) are not built"),
        (RECIPE_A6, [*WORD_ON, "model.write_gate=true"], r"model.write_gate \(A10\) is not built"),
        (RECIPE_A6, [*WORD_ON, "train.gate_mode=reinforce"], "train.gate_mode=reinforce"),
        (RECIPE_A6, [*WORD_ON, "train.aux_enabled=[cost]", "train.lambda_c=0.1"], "the cost term"),
        (RECIPE_A6, [*WORD_ON, "hardware.window=split"], "the event loop is built into"),
        (RECIPE_A6, [*WORD_ON, "hardware.window=fused"], "the event loop is built into"),
    ],
)
def test_configuration_refusals(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(path, overrides)


def test_other_model_names_and_the_split_window_refuse_by_name() -> None:
    cfg = load_config(RECIPE_A6, WORD_ON)
    assert "model.write_events=word" in (split_window_scope_error(cfg) or "")
    with pytest.raises(ValueError, match="is HSN-T's event loop; model_name='reader'"):
        load_config(RECIPE_A6, [*WORD_ON, "model_name=reader"])
    with pytest.raises(ValueError):  # the HSN class takes no reader, so no event loop either
        HSN.from_config(_ev_cfg("word"))


def test_capture_is_refused() -> None:
    cfg = _ev_cfg("word", STE)
    model = _model(cfg)
    tokens, ctx, state, reset, carry = _window_of(cfg, model)
    with torch.no_grad(), pytest.raises(NotImplementedError, match="not built into the event loop"):
        model(tokens, ctx, state, reset, capture=True, prev_tokens=carry)


# --------------------------------------------------------------------------------------------------
# The compiled autograd window and training
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(COMPILED))
@pytest.mark.parametrize("kind", ["word", "all"])
@pytest.mark.usefixtures("small_blocks")
def test_compiled_event_window_equals_eager(
    name: str, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The event block is the compile unit (the window the TinyStories HSN-T arms train on):
    the compiled window equals the eager event loop, and a second window with another number
    of event blocks runs on the same graph."""
    monkeypatch.setattr("hsn.model.hsn.EVENT_BLOCK", p28.compile_mod.MAX_BLOCK)  # small_blocks
    overrides, training = COMPILED[name]
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        *overrides,
        *READER,
        CATTN,
        _ev(kind),
    )
    seed_everything(0)
    model = HSNT.from_config(cfg)
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():  # as p28._model(cfg, training=training, scale=0.3)
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.3)
    model.train(training)
    ctx = p28._ctx()
    window = p28._window(model, ctx)
    tokens, cx, state, reset = p28._case_inputs(model, True)
    tokens[:, 0::3], tokens[:, 1::3], tokens[:, 2::3] = 97, 120, 32  # a word end every third byte
    tokens[0] = 97  # and a row without one
    tol = p28.TOL[(ctx.device.type, "fp32")]
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    got = p28._run(lambda: window(tokens, cx, state, reset))
    slots = int(ref[2]["event_valid"].shape[1])
    assert slots == (tokens.shape[1] if kind == "all" else 2 * ((tokens.shape[1] // 3 + 1) // 2))
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    if training:
        grads, ref_grads = p28._grads(model, got), p28._grads(model, ref)
        assert p28._grad_diff(grads, ref_grads) <= tol["grads"]
        assert sum(g is not None and float(g.abs().sum()) > 0.0 for g in grads) > 10
    if kind == "word":
        calls, seconds = window.calls, window.compile_seconds
        tokens = tokens.clone()
        tokens[1, 0::2], tokens[1, 1::2] = 97, 32  # more events: another block of slots
        ref = p28._run(lambda: model(tokens, cx, state, reset))
        got = p28._run(lambda: window(tokens, cx, state, reset))
        assert int(ref[2]["event_valid"].shape[1]) > slots
        diff = p28._max_diff(got, ref)
        assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
        assert window.calls == calls + 1 and window.compile_seconds == seconds  # no new graph
        with torch.no_grad(), pytest.raises(NotImplementedError, match="event loop"):
            window(tokens, cx, state, reset, capture=True)


@pytest.mark.parametrize("extra", [(), (CATTN,)])
def test_recipe_with_word_events_trains_and_evaluates(
    extra: tuple[str, ...], tmp_path: Path
) -> None:
    """Thirty TBPTT steps of the small A6 recipe with the candidate's cut on the tiny CPU
    configuration of the trainer tests, the memory writing at word ends only: finite losses
    that fall, the route guard and the statistics fed per event slot, and two evaluations."""
    steps = 30
    cfg = hsn_config(
        steps,
        *A5,
        CUT_C,
        "model.cand_norm=rms",
        "model.k_select=3",
        "train.soft_until_step=0",
        "train.gate_mode=ste",
        "train.lr=3e-3",
        "train.eval_every=15",
        "train.lambda_p=1.0",
        "train.route_guard.enabled=true",
        *READER,
        *extra,
        _ev("word"),
    )
    seed_everything(0)
    model = build_model(cfg)
    assert isinstance(model, HSNT) and model.cfg.write_events == "word"
    trainer = TBPTTTrainer(model, cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first - 0.05, (first, last)
    assert [e["step"] for e in history.evals] == [15, 30]
    assert all(math.isfinite(e["loss"]) for e in history.evals)
