"""The keyed word cells' speed work (``docs/wcells_speed.md``): what changed is how the work is
issued, not what is computed.

- The learned write (:meth:`hsn.model.word_cells.WordCells.write_learned`) takes its requests
  with the window's plan and decides its turns without a host read. It is compared here with the
  write as it stood on main ``846ecd3`` (:func:`_write_at_846ecd3`, the code as it was), on the
  same windows and values: the cells, the keys, the write positions and the records are equal bit
  for bit, and so is every gradient, the carried cells' included, over a chain of windows.
- On the card (``-m cuda``) the same comparison in fp32 with the products under bf16 autocast
  and the router's cut, as the recipe trains.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from hsn.model.hsn import _window_positions
from hsn.model.hsn_t import HSNT
from hsn.model.word_cells import (
    EMPTY,
    NO_KEY,
    WRITE_ROUNDS,
    KeyedFeed,
    WordCells,
    WordCellsCarry,
    WordCellsWindow,
)
from p1_11_support import restore_determinism  # noqa: F401
from test_word_cells import _prose
from test_word_cells_write import ARMS, _arm
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

_HUGE = 1 << 60


def _write_at_846ecd3(
    self: WordCells, win: WordCellsWindow, feed: KeyedFeed
) -> tuple[WordCellsCarry, dict[str, Tensor]]:
    """``WordCells.write_learned`` as it stood on main ``846ecd3`` (the reference of this file):
    the requests taken inside the write, the turns with masks (a host read each)."""
    N, L = self.n_cells, self.levels
    B, W = (int(n) for n in win.slot_key.shape)
    cell, router = feed.cell, feed.router
    M = cell.n_units
    device = win.slot_key.device
    cleared = win.cleared
    key0 = win.carry.key.masked_fill(cleared[:, None], NO_KEY)
    when0 = win.carry.when.masked_fill(cleared[:, None], EMPTY)
    table0 = win.carry.table.masked_fill(cleared[:, None, None], 0.0)
    dtype = table0.dtype
    # one request per key of a row: the key's last word end in the window
    order = torch.arange(W, device=device)
    ranked = torch.where(win.slot_active, win.slot_key * W + order, _HUGE).sort(dim=1).values
    after = torch.cat([ranked[:, 1:], ranked.new_full((B, 1), -W)], dim=1)
    last = (ranked // W != after // W) & (ranked < _HUGE)
    r, place = last.nonzero(as_tuple=True)
    slot = ranked[r, place] % W
    P, column = int(r.numel()), slot // L
    key, when = win.slot_key[r, slot], win.slot_when[r, slot]
    u, eps = feed.u[r, column], feed.eps[r, column]
    if self.addressed:
        cand = torch.arange(N, device=device).expand(P, N)
    else:
        cand = self.candidates(key)  # [P, A]
    mine = r[:, None]
    part = self._cell_part(feed, table0)[mine, cand]  # [P, C, M]
    shared = self._shared(feed, u, eps, feed.scalars[r, column])[:, None]
    scores = self._logits(feed, shared, part, when[:, None] - when0[mine, cand])
    rank = scores.amax(dim=-1)  # [P, C]: a candidate's score, as a routed cell's
    prior: Tensor | None = None
    if self.addressed:
        assert win.slot_code is not None
        prior = self._prior(win.slot_code[r, slot])
        rank = rank + prior
    # the turns: (1) the holder, (2) and (3) the scorer's best free candidate; where
    # several keys ask for one cell the latest word end takes it (the longer prefix)
    flat = mine * N + cand
    index = torch.arange(P, device=device)
    taken = torch.zeros(B * N, dtype=torch.bool, device=device)
    way = torch.full((P,), -1, dtype=torch.int64, device=device)
    runner = torch.full((P,), -math.inf, device=device)  # the best free candidate left out
    pending = torch.ones(P, dtype=torch.bool, device=device)
    rank_d = rank.detach()
    for turn in range(1 + WRITE_ROUNDS):
        if turn == 0:
            if self.addressed:
                continue
            holds = key0[mine, cand] == key[:, None]
            choice, asks = holds.to(torch.int8).argmax(dim=1), holds.any(dim=1)
            second = runner
        else:
            free = ~taken[flat]
            open_ = rank_d.masked_fill(~free, -math.inf)
            choice, asks = open_.argmax(dim=1), pending & free.any(dim=1)  # ties: the first
            same = cand == cand.gather(1, choice[:, None])  # (two ways may be one cell)
            second = open_.masked_fill(same, -math.inf).amax(dim=1)
        target = flat[index, choice]
        latest = torch.full((B * N,), -1, dtype=torch.int64, device=device)
        latest.scatter_reduce_(0, target[asks], slot[asks], "amax")
        won = asks & (latest[target] == slot)
        way = torch.where(won, choice, way)
        runner = torch.where(won, second, runner)
        taken[target[won]] = True
        pending = (pending & ~asks) if turn == 0 else (asks & ~won)
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


def _feed(
    model: HSNT, B: int, T: int, seed: int, *, gate: str | None, cut: bool, like: Tensor
) -> KeyedFeed:
    """Values as the model hands them, drawn at random; every byte counts as an event."""
    g = torch.Generator().manual_seed(seed)
    d, N = model.cfg.d, model.cfg.n_cells

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=g, dtype=torch.float64).to(like)

    scalars = rnd(B, T, 7)
    scalars[..., 3:5] = 0.0
    slot = torch.arange(1, T + 1, device=like.device).expand(B, T)
    return KeyedFeed(
        rnd(B, T, d).requires_grad_(True),
        rnd(B, T, d).requires_grad_(True),
        scalars,
        torch.tanh(rnd(B, T, N, d)),
        slot,
        model.cell,
        model.router,
        gate,
        1.0,
        cut,
    )


def _bits(x: Tensor) -> Tensor:
    """A floating tensor's bit patterns (so that ``-0.0`` and ``0.0`` differ)."""
    wide = x.detach().double() if x.dtype != torch.float64 else x.detach()
    return wide.contiguous().view(torch.int64)


def _same(a: Tensor | None, b: Tensor | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if not a.is_floating_point():
        return bool(torch.equal(a, b))
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(_bits(a), _bits(b)))


def _chain(
    model: HSNT,
    write: Any,
    *,
    windows: int,
    B: int,
    T: int,
    gate: str | None,
    cut: bool,
    autocast: bool = False,
) -> tuple[list[WordCellsCarry], list[dict[str, Tensor]], list[Tensor | None], int]:
    """``windows`` windows of prose through the module with the cells carried attached and
    ``write`` as the learned write: every window's carry and records, the gradients of the sum
    of what the windows read and of the last cells (every parameter, the first carry's cells
    and every window's ``u`` and ``eps``), and the number of writes."""
    cells = model.word_cells
    assert cells is not None
    like = cells.U
    device = like.device
    tokens, ctx, reset = (x.to(device) for x in _prose(B, windows * T, 5, resets=0.02))
    tables = (model.scorer.get_buffer("word_exact"), model.scorer.byte_codes)
    first = cells.init_carry(B, device)
    first = first._replace(table=first.table.clone().requires_grad_(True))
    carry, pos = first, torch.zeros(B, dtype=torch.int64, device=device)
    carries, records, leaves = [], [], [first.table]
    total: Tensor | None = None
    written = 0
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
        for i, lo in enumerate(range(0, windows * T, T)):
            sl = slice(lo, lo + T)
            positions = _window_positions(pos, reset[:, sl])
            args = (tokens[:, sl], ctx[:, sl], reset[:, sl], positions, carry)
            win = cells.window(*args, tables if cells.addressed else None)
            feed = _feed(model, B, T, 40 + i, gate=gate, cut=cut, like=like)
            leaves += [feed.u, feed.eps]
            read = cells.read_learned(win, feed)[0]
            after, record = write(cells, win, feed)
            written += int(
                (after.when != carry.when.masked_fill(win.cleared[:, None], EMPTY)).sum()
            )
            part = read.float().square().sum()
            total = part if total is None else total + part
            carries.append(after)
            records.append(record)
            carry, pos = after, positions[:, -1] + 1
    assert total is not None
    total = total + carry.table.float().square().sum()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(total, [*params, *leaves], allow_unused=True)
    return carries, records, list(grads), written


def _compare(model: HSNT, **how: Any) -> int:
    new = _chain(model, WordCells.write_learned, **how)
    old = _chain(model, _write_at_846ecd3, **how)
    for step, (a, b) in enumerate(zip(new[0], old[0], strict=True)):
        for name, x, y in zip(a._fields, a, b, strict=True):
            assert _same(x, y), (step, name)
    for step, (a, b) in enumerate(zip(new[1], old[1], strict=True)):
        assert a.keys() == b.keys()
        for name in a:
            assert _same(a[name], b[name]), (step, name)
    assert len(new[2]) == len(old[2])
    for index, (x, y) in enumerate(zip(new[2], old[2], strict=True)):
        assert _same(x, y), index
    assert any(g is not None and float(g.abs().sum()) > 0 for g in new[2])
    assert new[3] == old[3]
    return new[3]


@pytest.mark.parametrize("name", list(ARMS))
@pytest.mark.parametrize("gate", [None, "ste"])
@pytest.mark.parametrize("cut", [False, True])
def test_the_learned_write_equals_the_write_of_846ecd3_bit_for_bit(
    name: str, gate: str | None, cut: bool
) -> None:
    """Six windows with document starts among them, the cells carried attached: every carry,
    record and gradient of the write equals main's, bit for bit (float64, the CPU), with and
    without the straight-through gate and the router's cut, on every learned arm."""
    model = _arm(name)
    if gate is None:
        model.eval()
    assert _compare(model, windows=6, B=3, T=16, gate=gate, cut=cut) > 0


def test_the_window_hands_the_write_its_requests() -> None:
    """A learned arm's window carries the write's requests (one per key of a row, at the key's
    last word end), the by-rule table's window none; a window without them (a caller that
    built its own) gives the same write."""
    model = _arm("word")
    cells = model.word_cells
    assert cells is not None
    B, T = 3, 16
    tokens, ctx, reset = _prose(B, T, 5, resets=0.02)
    positions = _window_positions(torch.zeros(B, dtype=torch.int64), reset)
    win = cells.window(tokens, ctx, reset, positions, cells.init_carry(B, "cpu"))
    req = win.requests
    assert req is not None and req.row.numel() > 0
    pairs = set(zip(req.row.tolist(), req.key.tolist(), strict=True))
    assert len(pairs) == req.row.numel()  # one request per key of a row
    for r, s, k, w in zip(
        req.row.tolist(), req.slot.tolist(), req.key.tolist(), req.when.tolist(), strict=True
    ):
        same_key = (win.slot_key[r] == k) & win.slot_active[r]
        assert bool(win.slot_active[r, s]) and int(win.slot_key[r, s]) == k
        assert s == int(same_key.nonzero().max()) and w == int(win.slot_when[r, s])
    assert torch.equal(req.flat, req.row[:, None] * cells.n_cells + req.cand)
    assert torch.equal(req.cand, cells.candidates(req.key))
    feed = _feed(model, B, T, 7, gate="ste", cut=False, like=cells.U)
    with_requests, _ = cells.write_learned(win, feed)
    without, _ = cells.write_learned(win._replace(requests=None), feed)
    for x, y in zip(with_requests, without, strict=True):
        assert _same(x, y)
    table = _arm("word", "model.word_cells_value=bytes").word_cells
    assert table is not None and not table.learned
    assert table.window(tokens, ctx, reset, positions, table.init_carry(B, "cpu")).requests is None


@pytest.mark.cuda
@pytest.mark.parametrize("name", ["word", "cell", "address"])
@pytest.mark.parametrize("autocast", [False, True])
def test_the_learned_write_equals_the_write_of_846ecd3_bit_for_bit_cuda(
    name: str, autocast: bool
) -> None:
    """The same on the card, as the recipe trains: an fp32 state, the router's cut, the
    straight-through gate, with and without the products under bf16 autocast; a batch of 32
    rows over windows of 64 bytes."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    model = _arm(name).float().cuda()
    assert _compare(model, windows=5, B=32, T=64, gate="ste", cut=True, autocast=autocast) > 0
