"""The keyed word cells, stage 2 (``docs/amendments/word_cells.md``): the HSN's own write
(``model.word_cells_value = word | cell``) and candidate 2 (``model.word_cells = address``).

The new keys are opt-in and leave stage 1's arms their identities; the learned arms add their
parameters and nothing else; the deferred write and the addressed read equal a reference in
plain Python that takes one request (one byte) at a time and scores it with the scorer's own
map on the assembled features; the cells stay in ``[-1, 1]``; the write's parameters, the units
and the scorer get a gradient through the carried cells exactly when the cells are carried
attached, and the trainer carries them so inside a gradient segment and nowhere else. The
compiled window, the fused block's reference, the resume and the card are tested with stage
1's tests (``tests/test_word_cells.py``), which take the arm as a parameter.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import HASH_OMIT_AT_DEFAULT, load_config
from hsn.config import config_hash as chash
from hsn.device import seed_everything
from hsn.experiments.spec import config_view, digest, memory_key
from hsn.model.hsn import WindowCarry, _window_positions, address_history
from hsn.model.hsn_t import HSNT
from hsn.model.word_cells import (
    AGE_CAP,
    AGE_SCALE,
    EMPTY,
    GATE_BIAS,
    NO_KEY,
    READ_CELLS,
    WRITE_ROUNDS,
    KeyedFeed,
    WordCells,
    WordCellsWindow,
)
from hsn.train.memory_plan import word_cells_param_count
from p1_11_support import restore_determinism  # noqa: F401
from test_hsn_t import _gap, _model, _reader_cfg
from test_tbptt_windows import _cfg as _trainer_cfg
from test_tbptt_windows import _trainer
from test_word_cells import (
    BASE,
    ON,
    PROTOCOL,
    RECIPE,
    TRAINED,
    _prose,
    _text_datasets,
)
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

WORD, CELL = "model.word_cells_value=word", "model.word_cells_value=cell"
ADDRESS = "model.word_cells=address"
#: the learned arms on the tests' tiny model: (name, overrides behind BASE)
ARMS = {
    "word": (ON, "model.word_cells_n=16", WORD),
    "cell": (ON, "model.word_cells_n=16", CELL),
    "one_way": (ON, "model.word_cells_n=6", "model.word_cells_ways=1", WORD),
    "address": (ADDRESS, "model.word_cells_n=12", WORD),
    "address_cell": (ADDRESS, "model.word_cells_n=5", CELL, "model.word_cells_kappa=0.5"),
}


def _arm(name: str, *extra: str) -> HSNT:
    """The tests' model with a learned arm, ``U`` away from its zero start."""
    model = _model(_reader_cfg(*BASE, *ARMS[name], *extra))
    assert model.word_cells is not None
    with torch.no_grad():
        model.word_cells.U.normal_(0.0, 0.5)
    return model


# --------------------------------------------------------------------------------------------------
# The configuration and the parameters
# --------------------------------------------------------------------------------------------------


def test_the_new_keys_are_opt_in_and_leave_the_hash_arms_alone() -> None:
    assert "model.word_cells_kappa" in HASH_OMIT_AT_DEFAULT
    bytes_arm = load_config(RECIPE, [ON])
    assert "word_cells_kappa" not in config_view(bytes_arm)["model"]
    seen = {(chash(bytes_arm), memory_key(bytes_arm, "none", PROTOCOL))}
    arms = ([ON, WORD], [ON, CELL], [ADDRESS, WORD, "model.word_cells_n=256"])
    for overrides in (*arms, [*arms[2], "model.word_cells_kappa=2"]):
        cfg = load_config(RECIPE, overrides)
        seen.add((chash(cfg), memory_key(cfg, "none", PROTOCOL)))
        assert digest(config_view(cfg)) != digest(config_view(bytes_arm))
    assert len(seen) == 5 and len({h for h, _ in seen}) == 5 == len({m for _, m in seen})
    view = config_view(load_config(RECIPE, arms[2]))["model"]
    assert view["word_cells"] == "address" and "word_cells_kappa" not in view
    assert load_config(RECIPE, [ON, WORD, "train.tbptt_windows=3", "train.grad_accum=3"])
    opened = load_config(RECIPE, [ON, WORD, "model.word_cells_gate_bias=6"])
    assert config_view(opened)["model"]["word_cells_gate_bias"] == 6.0
    assert (chash(opened), memory_key(opened, "none", PROTOCOL)) not in seen
    assert "word_cells_gate_bias" not in config_view(load_config(RECIPE, arms[0]))["model"]


NO_GATE = ("model.address_gate=false", "model.address_gate_input=u")
REFUSED = [
    ([ADDRESS], "set model.word_cells_value=word"),
    ([ADDRESS, WORD, "model.word_address=bag"], "needs model.word_address=exact"),
    ([ADDRESS, WORD, "model.word_cells_ways=3"], "the addressed cells have none"),
    ([ON, WORD, "model.word_cells_kappa=2"], "hash has none"),
    ([ON, WORD, "model.scorer=mlp", *NO_GATE], "needs model.scorer=linear"),
    ([ON, CELL, "model.routing=random"], "needs model.scorer=linear and model.routing=topk"),
    ([ON, CELL, "model.ablate_shared_u=true"], "model.ablate_shared_u removes"),
    (["model.word_cells_kappa=2"], "model.word_cells is off"),
    ([ON, "model.word_cells_value=sum"], "word_cells_value"),
    ([ON, "model.word_cells_gate_bias=6"], "the by-rule value has no gate"),
]


@pytest.mark.parametrize(("overrides", "message"), REFUSED)
def test_refusals(overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(RECIPE, overrides)


def test_the_learned_arms_add_their_parameters_only() -> None:
    """Against the by-rule arm of the same sizes: every shared parameter and buffer starts as
    it does there (the arms draw from the cells' own generator, behind stage 1's tables); the
    new names are the in-edges, their logits (zero), the gate bias, ``V_s`` for ``word`` and
    the addresses of candidate 2, which has no hash; the counts are the closed form's."""
    expected = {
        "word": {"sources", "edge_logit", "beta_K", "V_s"},
        "cell": {"sources", "edge_logit", "beta_K"},
        "address": {"sources", "edge_logit", "beta_K", "V_s", "addresses"},
    }
    for name, new in expected.items():
        cfg = _reader_cfg(*BASE, *ARMS[name])
        sizes = [k for k in ARMS[name] if "word_cells_n" in k]
        seed_everything(0)
        plain = HSNT.from_config(_reader_cfg(*BASE, ON, *sizes)).state_dict()
        seed_everything(0)
        model = HSNT.from_config(cfg)
        cells, state = model.word_cells, model.state_dict()
        assert cells is not None
        gone = {"word_cells.hash_a", "word_cells.hash_b"} if name == "address" else set()
        assert set(state) - set(plain) == {f"word_cells.{n}" for n in new}
        assert set(plain) - set(state) == gone
        for key, value in plain.items():
            assert key in gone or not isinstance(value, Tensor) or torch.equal(value, state[key])
        d, n, degree = cfg.model.d, cfg.model.word_cells_n, cfg.model.in_degree
        assert cells.sources.shape == (n, degree) and int(cells.sources.max()) < cfg.model.n_cells
        assert all(len(set(row)) == degree for row in cells.sources.tolist())
        assert not bool(cells.edge_logit.any()) and bool((cells.beta_K == GATE_BIAS).all())
        assert GATE_BIAS == cfg.model.word_cells_gate_bias == 3.0
        count = 16 * d * d + n * degree + d + (d * d if "V_s" in new else 0)
        assert model.param_report()["word_cells"] == word_cells_param_count(cfg.model) == count
    recipe = load_config(RECIPE, [ON, WORD]).model
    assert word_cells_param_count(recipe) == 65_536 + 4_096 + 64 + 4_096
    with pytest.raises(ValueError, match="is not built"):
        WordCells(_reader_cfg(*BASE, ON).model.model_copy(update={"word_cells": "address"}), seed=0)


# --------------------------------------------------------------------------------------------------
# The rule of the learned arms, one request (one byte) at a time
# --------------------------------------------------------------------------------------------------


def _feed(model: HSNT, B: int, T: int, seed: int, gate: str | None = "ste") -> KeyedFeed:
    """Values as the model hands them, drawn at random; every byte counts as an event."""
    g = torch.Generator().manual_seed(seed)
    d, N = model.cfg.d, model.cfg.n_cells

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=g, dtype=torch.float64)

    scalars = rnd(B, T, 7)
    scalars[..., 3:5] = 0.0
    slot = torch.arange(1, T + 1).expand(B, T)
    routed = torch.tanh(rnd(B, T, N, d))
    return KeyedFeed(
        rnd(B, T, d),
        rnd(B, T, d),
        scalars,
        routed,
        slot,
        model.cell,
        model.router,
        gate,
        1.0,
        False,
    )


def _scorer_rank(
    cells: WordCells, feed: KeyedFeed, at: tuple[int, int], state: Tensor, age: int
) -> Tensor:
    """The scorer's logits ``[M]`` of a keyed cell at byte ``at``: its own map on the assembled
    features ``[u ; eps ; h ; age ; scalars ; address features (zero)]``."""
    scorer = feed.router.scorer
    feature = math.log1p(min(max(age, 0), AGE_CAP) / AGE_SCALE)
    parts = [feed.u[at], feed.eps[at], state, state.new_tensor([feature]), feed.scalars[at]]
    width = scorer.in_features - sum(int(p.numel()) for p in parts)
    logits: Tensor = scorer.net(torch.cat([*parts, state.new_zeros(width)]))
    return logits.float()


def _prefix_code(model: HSNT, prefix: list[int]) -> Tensor:
    """The exact word code of ``prefix`` (bytes as written), from its definition."""
    folded = [b + 32 if 65 <= b <= 90 else b for b in prefix]
    if len(folded) == 1:
        return model.scorer.byte_codes[folded[0]]
    bits = model.scorer.get_buffer("word_exact")
    flips = sum(bits[m, b].to(torch.int64) for m, b in enumerate(folded)) % 2
    return (1 - 2 * flips).to(torch.float64) / math.sqrt(model.cfg.d)


def _reference_write(
    model: HSNT, win: WordCellsWindow, feed: KeyedFeed, b: int
) -> tuple[Tensor, Tensor, Tensor, dict[str, int]]:
    """Row ``b``'s deferred write as the document states it."""
    cells, cell = model.word_cells, model.cell
    assert cells is not None
    L, N, d = cells.levels, cells.n_cells, cells.d
    key0, when0, table0 = win.carry.key[b].clone(), win.carry.when[b].clone(), win.carry.table[b]
    if bool(win.cleared[b]):
        key0[:], when0[:], table0 = NO_KEY, EMPTY, torch.zeros_like(table0)
    requests: dict[int, int] = {}  # a key's last word end in the window
    for w in range(win.slot_key.shape[1]):
        if bool(win.slot_active[b, w]):
            requests[int(win.slot_key[b, w])] = w
    seen = {"held": 0, "second": 0, "dropped": 0, "new": 0}

    def candidates(key: int) -> list[int]:
        return list(range(N)) if cells.addressed else cells.candidates(torch.tensor(key)).tolist()

    def rank(w: int, c: int) -> float:
        age = int(win.slot_when[b, w]) - int(when0[c])
        best = float(_scorer_rank(cells, feed, (b, w // L), table0[c], age).max())
        if cells.addressed:
            assert win.slot_code is not None
            best += cells.kappa * math.sqrt(d) * float(win.slot_code[b, w] @ cells.addresses[c])
        return best

    taken: dict[int, tuple[int, int]] = {}  # cell -> (key, slot)
    pending = []
    for key, w in requests.items():
        holders = [] if cells.addressed else [c for c in candidates(key) if int(key0[c]) == key]
        if holders:
            taken[holders[0]] = (key, w)
            seen["held"] += 1
        else:
            pending.append((key, w))
    for turn in range(WRITE_ROUNDS):
        asks: dict[int, list[tuple[int, int]]] = {}
        for key, w in pending:
            free = [c for c in dict.fromkeys(candidates(key)) if c not in taken]
            if not free:
                seen["dropped"] += 1
                continue
            ranks = [rank(w, c) for c in free]
            asks.setdefault(free[ranks.index(max(ranks))], []).append((w, key))
        pending = []
        for c, group in asks.items():
            w, key = max(group)  # the latest word end, at one word end the longer prefix
            taken[c] = (key, w)
            seen["second"] += turn
            pending += [(k, slot) for slot, k in group if slot != w]
    seen["dropped"] += len(pending)
    table, keys, whens = table0.clone(), key0.clone(), when0.clone()
    for c, (key, w) in taken.items():
        t = w // L
        age = int(win.slot_when[b, w]) - int(when0[c])
        unit = int(_scorer_rank(cells, feed, (b, t), table0[c], age).argmax())
        fresh = not cells.addressed and int(key0[c]) != key
        seen["new"] += fresh
        h = torch.zeros(d, dtype=table0.dtype) if fresh else table0[c]
        edges = torch.softmax(cells.edge_logit[c], dim=0)
        event = int(feed.slot[b, t]) - 1
        m = sum(
            (
                a * cell.V(feed.routed[b, event, int(j)])
                for a, j in zip(edges, cells.sources[c], strict=True)
            ),
            torch.zeros(d, dtype=table0.dtype),
        )
        x = feed.u[b, t]
        if cells.V_s is not None:
            x = x + cells.V_s @ win.slot_value[b, w]
        v = cell.norm(torch.cat([x, feed.eps[b, t], h, m]))
        out = cell.units.forward_all(v[None, None])[0, 0, unit]
        cand = torch.tanh(cell.cand_norm(cell.W_out(out)))
        gate = torch.sigmoid(cell.gate(v) + cells.beta_K)
        table[c] = (1 - gate) * h + gate * cand
        whens[c] = win.slot_when[b, w]
        if not cells.addressed:
            keys[c] = key
    return table, keys, whens, seen


def _reference_read(model: HSNT, win: WordCellsWindow, feed: KeyedFeed) -> Tensor:
    """Candidate 2's read ``p_t``, one byte at a time."""
    cells = model.word_cells
    assert cells is not None and win.read_code is not None
    B, T = win.offset.shape
    d, L = cells.d, cells.levels
    out = torch.zeros(B, T, d + 2, dtype=torch.float64)
    table, when = win.carry.table, win.carry.when
    for b in range(B):
        for t in range(T):
            if not bool(win.hit[b, t]):
                continue
            position = int(win.slot_when[b, t * L])
            ranks = []
            for c in range(cells.n_cells):
                age = position - int(when[b, c])
                best = float(_scorer_rank(cells, feed, (b, t), table[b, c], age).max())
                prior = cells.kappa * math.sqrt(d) * float(win.read_code[b, t] @ cells.addresses[c])
                ranks.append(best + prior)
            order = sorted(range(cells.n_cells), key=lambda c: -ranks[c])[:READ_CELLS]
            held = [c for c in order if int(when[b, c]) > EMPTY]
            value = sum((table[b, c] for c in held), torch.zeros(d, dtype=torch.float64))
            ages = [position - int(when[b, c]) for c in held]
            n = max(len(held), 1)
            old = sum(a >= cells.sight for a in ages) / n
            logs = sum(math.log1p(a / cells.sight) for a in ages) / n
            mapped = cells.U[int(win.offset[b, t])] @ (value / READ_CELLS)
            out[b, t] = torch.cat([mapped, torch.tensor([old, logs], dtype=torch.float64)])
    return out


@pytest.mark.parametrize("name", list(ARMS))
def test_the_learned_write_and_read_against_a_reference(name: str) -> None:
    """Six windows of prose (document starts among them) through the module with drawn values:
    after every window the cells, their keys and write positions equal the per-row reference
    (which scores with the scorer's own map on the assembled features), every branch of the
    rule is met, every cell stays in ``[-1, 1]``, a write is one of the model's events, and
    candidate 2's read equals its per-byte reference, its codes those of the word address."""
    model = _arm(name).eval()
    cells = model.word_cells
    assert cells is not None
    B, T = 3, 16
    tokens, ctx, reset = _prose(B, 6 * T, 5, resets=0.02)
    tables = (model.scorer.get_buffer("word_exact"), model.scorer.byte_codes)
    carry, state = cells.init_carry(B, "cpu"), None
    pos, total = torch.zeros(B, dtype=torch.int64), {"held": 0, "second": 0, "dropped": 0, "new": 0}
    with torch.no_grad():
        for i, lo in enumerate(range(0, 6 * T, T)):
            sl = slice(lo, lo + T)
            positions = _window_positions(pos, reset[:, sl])
            args = (tokens[:, sl], ctx[:, sl], reset[:, sl], positions, carry)
            win = cells.window(*args, tables if cells.addressed else None)
            feed = _feed(model, B, T, 40 + i)
            words = model.scorer.word_codes(tokens[:, sl], ctx[:, sl] == 6, reset[:, sl], state)
            state = words.state
            active = win.slot_active.view(B, T, -1).any(dim=-1)
            assert not bool((active & ~words.events).any())
            read, hit, _ = cells.read_learned(win, feed)
            if cells.addressed:
                assert _gap(read, _reference_read(model, win, feed)) <= 1e-9
                for b, t in win.hit.nonzero().tolist():
                    start = t
                    while (
                        start > 0 and bool(win.hit[b, start - 1]) and not bool(reset[b, lo + start])
                    ):
                        start -= 1
                    prefix = tokens[b, lo + start : lo + t + 1].tolist()[: cells.levels]
                    if start > 0 or bool(reset[b, lo]):  # the word began inside the window
                        assert _gap(win.read_code[b, t], _prefix_code(model, prefix)) <= 1e-12
            else:
                assert _gap(read, cells.read(win)) == 0.0 and float(hit) == float(
                    cells.rates(win)[0]
                )
            after, records = cells.write_learned(win, feed)
            for b in range(B):
                table, keys, whens, seen = _reference_write(model, win, feed, b)
                assert _gap(after.table[b], table) <= 1e-9, (i, b)
                assert torch.equal(after.key[b], keys) and torch.equal(after.when[b], whens)
                total = {k: total[k] + seen[k] for k in total}
            assert float(after.table.abs().max()) <= 1.0 and torch.equal(after.word, win.word)
            assert 0.0 < float(records["word_cells_gate"]) < 1.0
            assert ("word_cells_prior" in records) == cells.addressed
            carry, pos = after, positions[:, -1] + 1
    wanted = ("second", "dropped") if cells.addressed else ("held", "new", "second")
    if name == "one_way":  # a key that loses its one cell has no other to ask for
        wanted = ("held", "new", "dropped")
    assert all(total[k] > 0 for k in wanted), total


def test_the_straight_through_weights_change_no_value_and_train_the_scorer() -> None:
    """Without a gate (evaluation) and with the straight-through gate the write gives the same
    cells; the scorer's weights get a gradient from the write with the gate only, and the
    credit of the executed unit alone is another gradient than the softmax's."""
    model = _arm("word")
    cells = model.word_cells
    assert cells is not None
    tokens, ctx, reset = _prose(3, 16, 5, resets=0.0)
    positions = _window_positions(torch.zeros(3, dtype=torch.int64), reset)
    win = cells.window(tokens, ctx, reset, positions, cells.init_carry(3, "cpu"))
    weight = model.scorer.net.weight
    grads, tables = {}, []
    for gate, credit in ((None, "softmax"), ("ste", "softmax"), ("ste", "executed")):
        model.router.set_unit_credit(credit)
        after, _ = cells.write_learned(win, _feed(model, 3, 16, 9, gate))
        grads[gate, credit] = torch.autograd.grad(
            after.table.square().sum(), weight, allow_unused=True
        )[0]
        tables.append(after.table.detach())
        assert _gap(tables[-1], tables[0]) == 0.0
    assert grads[None, "softmax"] is None
    assert float(grads["ste", "softmax"].abs().sum()) > 0
    assert _gap(grads["ste", "softmax"], grads["ste", "executed"]) > 1e-9


# --------------------------------------------------------------------------------------------------
# The model and the trainer: the gradient through the carried cells
# --------------------------------------------------------------------------------------------------


def _carried(model: HSNT, stream: Any, T: int, attached: bool) -> tuple[Tensor, WindowCarry]:
    """Three windows with the state detached at every window and the cells carried attached
    or not: the last window's logits and the carry it started from."""
    tokens, ctx, reset = stream
    state, prev = model.init_state(tokens.shape[0]), None
    for lo in range(0, 3 * T, T):
        sl = slice(lo, lo + T)
        before = prev
        logits, state, info = model(
            tokens[:, sl], ctx[:, sl], state, reset[:, sl], prev_tokens=prev
        )
        prev = address_history(model.cfg, tokens[:, sl], prev, info, attached=attached)
        state = state.detach()
    assert isinstance(before, WindowCarry)
    return logits, before


@pytest.mark.parametrize("name", ["word", "cell", "address"])
def test_the_write_trains_through_the_carried_cells(name: str) -> None:
    """The loss of a third window, with the routed state detached at every window: carried
    attached, the cells pass it into the writes of the windows before, so ``V_s``, ``beta_K``
    and the in-edge logits get a gradient and the units', the scorer's, ``V``'s and ``S``'s
    change; carried detached (the default of ``address_history``) the write's own parameters
    get none. The values are the same either way."""
    T, stream = 12, _prose(3, 36, 4, resets=0.0)
    found: dict[bool, dict[str, Tensor | None]] = {}
    logits = {}
    for attached in (False, True):
        model = _arm(name)
        cells = model.word_cells
        assert cells is not None
        logits[attached], before = _carried(model, stream, T, attached)
        assert before.cells is not None and before.cells.table.requires_grad == attached
        assert float(before.cells.table.abs().max()) <= 1.0
        logits[attached].square().mean().backward()
        found[attached] = {n: p.grad for n, p in model.named_parameters()}
    assert _gap(logits[True], logits[False]) == 0.0
    own = ["word_cells.edge_logit", "word_cells.beta_K"] + ["word_cells.V_s"] * (name != "cell")
    for key in own:
        assert found[False][key] is None and float(found[True][key].abs().sum()) > 0, key
    shared = ("cell.units.W1", "cell.units.W2", "cell.gate.weight", "cell.W_out.weight")
    for key in (*shared, "cell.V.weight", "router.scorer.net.weight", "shared.fc1.weight"):
        assert _gap(found[True][key], found[False][key]) > 1e-9, key


@pytest.mark.parametrize("seed", [2, 4])
def test_the_model_writes_no_later_byte_into_an_earlier_event(seed: int) -> None:
    """A byte changed inside a window leaves the logits before it as they were, and the cells
    that a key of a word end before it wrote in both readings; it changes the logits behind
    it. (A cell may go to another key of the same word end, or to none, when later word ends
    ask for other cells: the deferred write settles a window's keys together.)"""
    model = _arm("word").eval()
    T, t0 = 12, 9
    tokens, ctx, reset = _prose(3, 3 * T, seed, resets=0.0)
    with torch.no_grad():
        state, prev = model.init_state(3), None
        for lo in (0, T):
            sl = slice(lo, lo + T)
            _, state, info = model(tokens[:, sl], ctx[:, sl], state, reset[:, sl], prev_tokens=prev)
            prev = address_history(model.cfg, tokens[:, sl], prev, info)
        window = (tokens[:, 2 * T :], ctx[:, 2 * T :])
        flags = torch.zeros(3, T, dtype=torch.bool)
        logits, _, info = model(*window, state, flags, prev_tokens=prev)
        later = window[0].clone()
        later[:, t0:] = torch.tensor(list(b"zz "))
        changed, _, other = model(later, window[1], state, flags, prev_tokens=prev)
        assert (
            _gap(changed[:, :t0], logits[:, :t0]) <= 1e-12 < _gap(changed[:, t0:], logits[:, t0:])
        )
        when, key = info["word_cells_when"], info["word_cells_key"]
        early = (when >= 2 * T) & (when < 2 * T + t0)
        early &= (other["word_cells_when"] == when) & (other["word_cells_key"] == key)
        assert int(early.sum()) >= 4
        assert _gap(info["word_cells_table"][early], other["word_cells_table"][early]) <= 1e-12


@pytest.mark.parametrize(
    ("segments", "name"), [(1, "word"), (3, "word"), (3, "cell"), (2, "address")]
)
def test_the_trainer_carries_the_cells_attached_inside_a_segment(
    segments: int, name: str, tmp_path: Path
) -> None:
    """Two steps of the tiny recipe on text with a learned arm: with ``train.tbptt_windows >
    1`` the write's own parameters move (their gradient comes through the cells of the
    segment's earlier windows) and with every window detached they stay; the keyed writes'
    units count as active exactly in the windows whose cells stay attached; the statistics
    hold the write's records."""
    arm = tuple(k for k in ARMS[name] if "word_cells_n" not in k)
    keys = tuple(k for k in TRAINED if k != ON and "word_cells_n" not in k)
    accumulate = (f"train.grad_accum={max(segments, 2)}", f"train.tbptt_windows={segments}")
    cfg = _trainer_cfg(2, *keys, *arm, "model.word_cells_n=32", *accumulate, "train.stats_every=1")
    trainer = _trainer(cfg, tmp_path / "run")
    cells = trainer.model.word_cells
    start = {n: p.detach().clone() for n, p in cells.named_parameters()}
    held, statistics = [], trainer._window_statistics

    def spy(info: dict[str, Tensor], *args: Any) -> Any:
        held.append("word_cells_units" in info)
        if held[-1]:
            info["word_cells_units"] = torch.ones_like(info["word_cells_units"])
        return statistics(info, *args)

    trainer._window_statistics = spy  # type: ignore[method-assign]
    history = trainer.fit(*_text_datasets(), 2)
    assert all(math.isfinite(s.loss) for s in history.steps)
    stats = history.steps[-1].stats
    assert 0.0 < stats["word_cells_gate"] < 1.0 and ("word_cells_prior" in stats) == (
        name == "address"
    )
    moved = {n: not torch.equal(p, start[n]) for n, p in cells.named_parameters()}
    assert moved.pop("U")
    assert set(moved.values()) == {segments > 1}, moved
    per_step = [(i + 1) % segments != 0 for i in range(max(segments, 2))]
    assert held == per_step * 2
    units = list(range(cfg.model.n_units))
    assert all((s.active_units == units) for s in history.steps) or segments == 1
    carry = trainer._prev_tokens
    assert isinstance(carry, WindowCarry) and carry.cells is not None
    assert not carry.cells.table.requires_grad and float(carry.cells.table.abs().max()) <= 1.0
