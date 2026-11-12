"""P2.10: the window kernels (``kernels/cuda/hsn_window.cu``) against their PyTorch references.
CUDA-marked; skipped without a card or without the built artifacts unless ``HSN_REQUIRE_CUDA=1``
(then missing artifacts fail)."""

from __future__ import annotations

import os

import pytest
import torch
from hsn.kernels import window
from hsn.train.split import recurrence_backward

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def device() -> torch.device:
    required = os.environ.get("HSN_REQUIRE_CUDA") == "1"
    if not torch.cuda.is_available():
        if required:
            pytest.fail("HSN_REQUIRE_CUDA=1 but no CUDA device is available")
        pytest.skip("no CUDA device")
    if not window.available():
        if required:
            pytest.fail("HSN_REQUIRE_CUDA=1 but the window kernels are not built")
        pytest.skip("window kernels not built (scripts/build_window_kernels.py)")
    window.prepare(0)
    return torch.device("cuda", 0)


def _case(
    device: torch.device, B: int, T: int, N: int, K: int, d: int, seed: int
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    cells = torch.stack(
        [torch.stack([torch.randperm(N, generator=g)[:K] for _ in range(T)]) for _ in range(B)]
    )
    valid = torch.rand(B, T, K, generator=g) > 0.25
    reset = torch.rand(B, T, generator=g) < 0.05
    case = {
        "c": torch.rand(B, T, K, d, generator=g) * 2 - 1,
        "b": torch.rand(B, T, K, d, generator=g),
        "h_old": torch.rand(B, T, K, d, generator=g) * 2 - 1,
        "cells": torch.where(valid, cells, torch.zeros_like(cells)),
        "valid": valid,
        "reset": reset,
        "pre": torch.randn(B, T, d, generator=g),
        "post": torch.randn(B, T, d, generator=g),
        "sel": torch.randn(B, T, K, d, generator=g),
    }
    return {k: v.to(device) for k, v in case.items()}


@pytest.mark.parametrize(
    "shape",
    [
        (32, 256, 32, 4, 64),
        (3, 7, 5, 3, 4),
        (1, 1, 2, 1, 1),
        (5, 33, 9, 9, 37),
        (2, 12, 40, 6, 200),
    ],
)
@pytest.mark.parametrize("present", [(True, True, True), (False, True, False), (True, False, True)])
def test_recurrence_kernel_equals_the_reference(
    device: torch.device, shape: tuple[int, ...], present: tuple[bool, bool, bool]
) -> None:
    B, T, N, K, d = shape
    case = _case(device, B, T, N, K, d, seed=sum(shape))
    cots = [
        case[name] if on else None for name, on in zip(("pre", "post", "sel"), present, strict=True)
    ]
    base = (case["c"], case["b"], case["h_old"], case["cells"], case["valid"], case["reset"], N)
    got = window.recurrence_backward(*base, *cots)
    assert got is not None
    want = recurrence_backward(
        *(t.double() if t.is_floating_point() else t for t in base[:6]),
        N,
        *(None if c is None else c.double() for c in cots),
    )
    for g, w in zip(got, want, strict=True):
        scale = float(w.abs().max().clamp_min(1.0))
        assert float((g.double() - w).abs().max()) <= 2e-5 * scale
        assert bool((g[~case["valid"]] == 0).all())


def test_recurrence_kernel_keeps_a_padded_nan_out(device: torch.device) -> None:
    case = _case(device, 4, 16, 8, 3, 8, seed=3)
    for name in ("c", "b", "h_old", "sel"):
        case[name][~case["valid"]] = float("nan")
    got = window.recurrence_backward(
        case["c"],
        case["b"],
        case["h_old"],
        case["cells"],
        case["valid"],
        case["reset"],
        8,
        case["pre"],
        case["post"],
        case["sel"],
    )
    assert got is not None
    assert bool(torch.isfinite(got[0]).all()) and bool(torch.isfinite(got[1]).all())


def test_recurrence_kernel_declines_what_it_does_not_handle(device: torch.device) -> None:
    case = _case(device, 2, 4, 4, 2, 4, seed=1)
    args = (case["cells"], case["valid"], case["reset"], 4, None, case["post"].double(), None)
    assert (
        window.recurrence_backward(*(case[k].double() for k in ("c", "b", "h_old")), *args) is None
    )
    big = _case(device, 2, 3, 40, 2, 400, seed=2)  # more shared memory than a block may take
    assert (
        window.recurrence_backward(
            big["c"],
            big["b"],
            big["h_old"],
            big["cells"],
            big["valid"],
            big["reset"],
            40,
            None,
            big["post"],
            None,
        )
        is None
    )
    cpu = {k: v.cpu() for k, v in case.items()}
    assert (
        window.recurrence_backward(
            cpu["c"],
            cpu["b"],
            cpu["h_old"],
            cpu["cells"],
            cpu["valid"],
            cpu["reset"],
            4,
            None,
            cpu["post"],
            None,
        )
        is None
    )


@pytest.mark.parametrize("shape", [(8, 256, 32, 64), (3, 7, 5, 4), (1, 1, 1, 1), (2, 33, 9, 37)])
@pytest.mark.parametrize("present", [(True, True), (False, True), (True, False)])
def test_dense_recurrence_kernel_equals_the_reference(
    device: torch.device, shape: tuple[int, ...], present: tuple[bool, bool]
) -> None:
    from hsn.train.split import dense_recurrence_backward

    Bd, T, N, d = shape
    g = torch.Generator().manual_seed(sum(shape))
    c = (torch.rand(Bd, T, N, d, generator=g) * 2 - 1).to(device)
    b = torch.rand(Bd, T, N, d, generator=g).to(device)
    h_old = (torch.rand(Bd, T, N, d, generator=g) * 2 - 1).to(device)
    reset = (torch.rand(Bd, T, generator=g) < 0.05).to(device)
    cots = [torch.randn(Bd, T, d, generator=g).to(device) if on else None for on in present]
    got = window.dense_recurrence_backward(c, b, h_old, reset, N, *cots)
    assert got is not None
    want = dense_recurrence_backward(
        c.double(),
        b.double(),
        h_old.double(),
        reset,
        N,
        *(None if x is None else x.double() for x in cots),
    )
    for a, w in zip(got, want, strict=True):
        scale = float(w.abs().max().clamp_min(1.0))
        assert float((a.double() - w).abs().max()) <= 2e-5 * scale


# --------------------------------------------------------------------------------------------------
# The fused cell round against PDGCell.step_sparse and the tick
# --------------------------------------------------------------------------------------------------


def _round_case(
    device: torch.device, overrides: tuple[str, ...], batch: int, budget: float, seed: int
) -> tuple:
    from hsn.config import load_config
    from hsn.device import seed_everything
    from hsn.model.hsn import HSN
    from hsn.model.state import CellGraphState
    from test_a6_feedback_detach import REF

    cfg = load_config(
        REF,
        overrides=[
            "profile=cpu",
            "model.routing=topk",
            "train.gate_mode=ste",
            "model.key_address=true",
            "model.selected_readout=true",
            "train.feedback_detach=[summary,eps,state,router]",
            f"model.budget={budget}",
            *overrides,
        ],
    )
    seed_everything(seed)
    model = HSN.from_config(cfg).to(device).train()
    with torch.no_grad():  # away from the initial values: open gates, live norms and biases
        for p in model.parameters():
            p.add_(0.2 * torch.randn_like(p))
    g = torch.Generator().manual_seed(seed)
    N, M, d = cfg.model.n_cells, cfg.model.n_units, cfg.model.d
    h = (torch.rand(batch, N, d, generator=g) * 1.6 - 0.8).to(device)
    state = CellGraphState._make(
        h=h,
        q=h.mean(dim=1) + 0.01,
        age=torch.randint(0, 9, (batch, N), generator=g).to(device),
        pos=torch.randint(0, 50, (batch,), generator=g).to(device),
    )
    u = torch.randn(batch, d, generator=g).to(device)
    eps = torch.randn(batch, d, generator=g).to(device)
    logits = torch.randn(batch, N, M, generator=g).to(device)
    sel = model.router.select(
        logits, cfg.model.budget, cfg.model.k_select, cfg.model.k_min, "topk", gate="ste"
    )
    return cfg, model, state, u, eps, sel


ODD = (
    "model.cand_norm=rms",
    "model.n_units=3",
    "model.hidden=77",
    "model.d=24",
    "model.context_dim=24",
    "model.n_cells=11",
    "model.k_select=5",
    "model.in_degree=3",
)
ONE = ("model.cand_norm=rms", "model.n_units=1", "model.hidden=33", "model.k_select=1")


@pytest.mark.parametrize(
    ("overrides", "batch", "budget"),
    [
        (("model.cand_norm=rms",), 32, 1.0),  # the A6 recipe's sizes
        (("model.cand_norm=rms",), 32, 0.01),  # a budget that pads slots
        (("model.cand_norm=none",), 5, 1.0),
        (ODD, 7, 1.0),
        (ONE, 1, 1.0),
    ],
)
def test_fused_cell_round_equals_the_native_round(
    device: torch.device, overrides: tuple[str, ...], batch: int, budget: float
) -> None:
    cfg, model, state, u, eps, sel = _round_case(device, overrides, batch, budget, seed=11)
    cell = model.cell
    assert window.cell_round_error(cell) is None
    d = cfg.model.d
    with torch.no_grad(), cell.window():
        consts = cell.window_consts()
        _, info = cell.step_sparse(u, eps, state, sel, static=True, consts=consts)
        want = info["state"].tick(info["selected"])
        got, h_sel = window.cell_round(
            cell, u, eps, state, sel.cell_idx, sel.unit_idx, sel.valid, consts.edge_weights
        )
        safe = sel.cell_idx.masked_fill(~sel.valid, 0)
        want_sel = want.h.gather(1, safe[:, :, None].expand(-1, -1, d))
    if budget < 1.0:
        assert not bool(sel.valid.all()) and bool(sel.valid.any())
    assert torch.equal(got.age, want.age) and torch.equal(got.pos, want.pos)
    for name, a, b in (("h", got.h, want.h), ("q", got.q, want.q), ("h_sel", h_sel, want_sel)):
        assert float((a - b).abs().max()) <= 2e-5, name
    written = info["selected"]
    assert torch.equal(got.h[~written], state.h[~written])  # unselected cells bit for bit


def test_fused_cell_round_refuses_the_shared_u_ablation(device: torch.device) -> None:
    """The in-kernel round builds the 4d input ``[u; eps; h; m]``: it names the ablation as the
    reason it cannot run that cell (the fused window does not use this op)."""
    overrides = ("model.cand_norm=rms", "model.ablate_shared_u=true")
    _, model, *_ = _round_case(device, overrides, 2, 1.0, seed=11)
    assert "ablate_shared_u" in (window.cell_round_error(model.cell) or "")


def test_fused_cell_round_over_a_window_stays_close(device: torch.device) -> None:
    """256 positions of fused rounds against 256 native ones from the same start (the same
    decisions each position, taken from the native trajectory's scores)."""
    cfg, model, state, _, _, _ = _round_case(device, ("model.cand_norm=rms",), 8, 1.0, seed=5)
    cell = model.cell
    g = torch.Generator().manual_seed(9)
    native, fused = state, state
    worst = 0.0
    with torch.no_grad(), cell.window():
        consts = cell.window_consts()
        for _ in range(256):
            u = torch.randn(8, cfg.model.d, generator=g).to(device)
            eps = torch.randn(8, cfg.model.d, generator=g).to(device)
            logits = torch.randn(8, cfg.model.n_cells, cfg.model.n_units, generator=g).to(device)
            sel = model.router.select(
                logits, cfg.model.budget, cfg.model.k_select, cfg.model.k_min, "topk", gate="ste"
            )
            _, info = cell.step_sparse(u, eps, native, sel, static=True, consts=consts)
            native = info["state"].tick(info["selected"])
            fused, _ = window.cell_round(
                cell, u, eps, fused, sel.cell_idx, sel.unit_idx, sel.valid, consts.edge_weights
            )
            worst = max(worst, float((fused.h - native.h).abs().max()))
    assert worst <= 1e-4
    assert float((fused.q - native.q).abs().max()) <= 1e-4
    assert torch.equal(fused.age, native.age)


# --------------------------------------------------------------------------------------------------
# The selection kernel against Router.select
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        (),
        ("model.age_max=2",),  # the refresh probes fire
        ("model.budget=0.1",),  # the budget cuts the prefix: padded slots
        ("model.age_max=2", "model.budget=0.1"),
        ("model.lambda_cost=0.05", "model.age_max=3"),
    ],
)
def test_select_kernel_takes_the_routers_decision(
    device: torch.device, overrides: tuple[str, ...]
) -> None:
    """``hsn_win_select`` returns :meth:`Router.select`'s hard decision: cells, units, valid
    slots, the straight-through threshold and the refreshed slots, on random scores, on scores
    with many exact ties (index order decides) and with stale cells (equal ages too)."""
    from hsn.model.router import BUDGET_SLACK
    from test_a6_feedback_detach import A5, ALL, _cfg, _model

    cfg = _cfg(*A5, ALL, "model.cand_norm=rms", *overrides)
    router = _model(cfg).router.to(device=device, dtype=torch.float32)
    N, M, K = cfg.model.n_cells, cfg.model.n_units, cfg.model.k_select
    g = torch.Generator().manual_seed(7)
    fired = 0
    for trial in range(40):
        B = 1 + trial % 5
        logits = torch.randn(B, N, M, generator=g)
        if trial % 2:
            logits = (logits * 2).round() / 2  # exact ties between cells and between units
        age = torch.randint(0, 2 * int(router.age_max) + 2, (B, N), generator=g)
        if trial % 3 == 0:
            age = (age // 3) * 3  # equal ages among the stale cells
        logits, age = logits.to(device), age.to(device)
        for probes in (True, False):
            want = router.select(
                logits,
                cfg.model.budget,
                K,
                cfg.model.k_min,
                "topk",
                gate=None,
                age=age if probes else None,
            )
            cells, units, valid, theta, refreshed = window.select(
                logits,
                age,
                router.cost_table,
                k=K,
                age_max=int(router.age_max),
                probes=probes,
                lambda_cost=float(router.lambda_cost),
                cell_overhead=float(router.cell_overhead),
                shared_cost=float(router.shared_cost),
                limit=float(cfg.model.budget) * (1.0 + BUDGET_SLACK),
            )
            assert torch.equal(valid, want.valid)
            assert torch.equal(cells, want.cell_idx)
            assert torch.equal(units, want.unit_idx)
            assert torch.equal(refreshed, want.refreshed)
            assert want.threshold is not None
            assert torch.equal(theta, want.threshold)
            fired += int(refreshed.sum())
    if any("age_max" in o for o in overrides):
        assert fired > 0  # the probes were exercised


# --------------------------------------------------------------------------------------------------
# The end-of-round kernel against PDGCell.step_sparse and the tick
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        (),
        ("model.cand_norm=none",),
        ("model.budget=0.1",),
        # the 3d cell input [eps; h; m] (the products' width), without and with the candidate
        # norm and padded slots
        ("model.ablate_shared_u=true",),
        ("model.budget=0.1", "model.ablate_shared_u=true"),
    ],
)
def test_cell_update_kernel_equals_the_native_round(
    device: torch.device, extra: tuple[str, ...]
) -> None:
    """``sparse_products`` + ``hsn_win_cell_update`` against ``step_sparse`` (static) and the
    tick on the same decision: the same state, summary, ages and position, the slots' cells
    after the update and the state's largest magnitude, with and without the candidate norm
    and with padded slots; and for the ablated cell input (the kernel reads the products'
    outputs, whose width does not change)."""
    from hsn.kernels import native
    from test_a6_feedback_detach import A5, ALL, _cfg, _model

    cfg = _cfg(
        *A5, ALL, *(("model.cand_norm=rms",) if not extra or "budget" in extra[0] else ()), *extra
    )
    model = _model(cfg).to(device=device, dtype=torch.float32)
    cell, router = model.cell, model.router
    N, d, K = cfg.model.n_cells, cfg.model.d, cfg.model.k_select
    g = torch.Generator().manual_seed(11)
    B = 5
    state = model.init_state(B, device)
    padded = 0
    with torch.no_grad(), cell.window():
        consts = cell.window_consts()
        for _ in range(12):
            u = torch.randn(B, d, generator=g).to(device)
            eps = torch.randn(B, d, generator=g).to(device)
            logits = torch.randn(B, N, cfg.model.n_units, generator=g).to(device)
            sel = router.select(logits, cfg.model.budget, K, cfg.model.k_min, "topk", gate=None)
            padded += int((~sel.valid).sum())
            _, info = cell.step_sparse(u, eps, state, sel, static=True, consts=consts)
            want = info["state"].tick(info["selected"])
            gate, y = cell.sparse_products(
                u, eps, state, sel.cell_idx, sel.unit_idx, sel.valid, consts
            )
            got, h_sel, h_max = window.cell_update(cell, y, gate, state, sel.cell_idx, sel.valid)
            assert float((got.h - want.h).abs().max()) <= 2e-6
            assert float((got.q - want.q).abs().max()) <= 2e-6
            assert torch.equal(got.age, want.age) and torch.equal(got.pos, want.pos)
            safe = sel.cell_idx.masked_fill(~sel.valid, 0)
            slots = want.h.gather(1, safe[:, :, None].expand(-1, -1, d))
            assert float((h_sel - slots).abs().max()) <= 2e-6
            assert float((h_max - got.h.abs().amax(dim=(1, 2))).abs().max()) == 0.0
            unwritten = ~native.selection_mask(sel.cell_idx, sel.valid, N)
            assert torch.equal(got.h[unwritten], state.h[unwritten])  # untouched cells: bit for bit
            state = want
    if extra and "budget" in extra[0]:
        assert padded > 0


# --------------------------------------------------------------------------------------------------
# The tape forms of the two glue kernels against their plain forms
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cand_norm", ["rms", "none"])
def test_tape_ops_write_what_the_plain_ops_return(device: torch.device, cand_norm: str) -> None:
    """``select_tape`` / ``update_tape`` over a block of positions against ``select`` /
    ``cell_update`` and :meth:`CellGraphState.reset` position by position (the same kernels, so
    bit for bit): record ``t`` of the decision tape; record ``t + 1`` of the state tape with the
    next position's reset applied (resets at the block's first, interior and last positions, and
    two in a row); the summary, slots and maxima before that reset; a last record that is never
    reset; and no record written but the position's own (the tapes start as a sentinel)."""
    from hsn.model.router import BUDGET_SLACK
    from test_a6_feedback_detach import A5, ALL, _cfg, _model

    cfg = _cfg(*A5, ALL, f"model.cand_norm={cand_norm}", "model.age_max=2", "model.budget=0.1")
    model = _model(cfg).to(device=device, dtype=torch.float32)
    cell, router = model.cell, model.router
    N, M, d, K = cfg.model.n_cells, cfg.model.n_units, cfg.model.d, cfg.model.k_select
    g = torch.Generator().manual_seed(23)
    B, n = 5, 6
    state = model.init_state(B, device).with_dense(
        (torch.rand(B, N, d, generator=g) * 1.6 - 0.8).to(device)
    )
    for _ in range(4):
        state = state.tick((torch.rand(B, N, generator=g) < 0.5).to(device))
    reset = torch.zeros(B, n, dtype=torch.bool)
    reset[0, 0] = True
    reset[1, 3] = True
    reset[2, n - 1] = True
    reset[3, 1] = reset[3, 2] = True
    reset = reset.to(device)
    sentinel = 7.0
    tape_h = torch.full((B, n + 1, N, d), sentinel, device=device)
    tape_q = torch.full((B, n + 1, d), sentinel, device=device)
    tape_age = torch.full((B, n + 1, N), 7, dtype=torch.int64, device=device)
    cells = torch.full((B, n, K), 7, dtype=torch.int64, device=device)
    units = torch.full((B, n, K), 7, dtype=torch.int64, device=device)
    valid = torch.zeros((B, n, K), dtype=torch.bool, device=device)
    refreshed = torch.zeros((B, n, K), dtype=torch.bool, device=device)
    theta = torch.full((B, n), sentinel, device=device)
    h_sel = torch.full((B, n, K, d), sentinel, device=device)
    h_max = torch.full((B, n), sentinel, device=device)
    qs = torch.full((B, n, d), sentinel, device=device)
    first = state.reset(reset[:, 0])
    tape_h[:, 0], tape_q[:, 0], tape_age[:, 0] = first.h, first.q, first.age
    keys = {
        "age_max": int(router.age_max),
        "probes": True,
        "lambda_cost": float(router.lambda_cost),
        "cell_overhead": float(router.cell_overhead),
        "shared_cost": float(router.shared_cost),
        "limit": float(cfg.model.budget) * (1.0 + BUDGET_SLACK),
    }
    fired = padded = 0
    with torch.no_grad():
        for t in range(n):
            pre = state.reset(reset[:, t])
            assert torch.equal(tape_h[:, t], pre.h) and torch.equal(tape_q[:, t], pre.q)
            assert torch.equal(tape_age[:, t], pre.age)
            logits = torch.randn(B, N, M, generator=g).to(device)
            want = window.select(logits, pre.age, router.cost_table, k=K, **keys)
            window.select_tape(
                logits, tape_age, router.cost_table, cells, units, valid, theta, refreshed,
                t=t, **keys,
            )  # fmt: skip
            got = (cells[:, t], units[:, t], valid[:, t], theta[:, t], refreshed[:, t])
            for a, b in zip(got, want, strict=True):
                assert torch.equal(a, b)
            assert bool((cells[:, t + 1 :] == 7).all()) and bool((theta[:, t + 1 :] == 7).all())
            fired += int(refreshed[:, t].sum())
            padded += int((~valid[:, t]).sum())
            y = torch.randn(B, K, d, generator=g).to(device)
            gate = torch.randn(B, K, d, generator=g).to(device)
            after, slots, top = window.cell_update(cell, y, gate, pre, want[0], want[2])
            window.update_tape(
                cell, y, gate, tape_h, tape_q, tape_age, cells, valid, reset, h_sel, h_max, qs,
                t=t,
            )  # fmt: skip
            nxt = after.reset(reset[:, t + 1]) if t + 1 < n else after
            assert torch.equal(tape_h[:, t + 1], nxt.h) and torch.equal(tape_q[:, t + 1], nxt.q)
            assert torch.equal(tape_age[:, t + 1], nxt.age)
            assert torch.equal(qs[:, t], after.q) and torch.equal(h_sel[:, t], slots)
            assert torch.equal(h_max[:, t], top)
            assert bool((tape_h[:, t + 2 :] == sentinel).all())
            assert bool((qs[:, t + 1 :] == sentinel).all())
            assert torch.equal(tape_h[:, t], pre.h)  # the record it read is untouched
            state = after
    assert fired > 0 and padded > 0  # probes and padded slots were exercised


def test_cell_update_kernel_reports_a_nan_state(device: torch.device) -> None:
    """The row maximum of the end-of-round kernel is ``amax``'s: a NaN anywhere in the row's
    state is a NaN maximum (the non-finite check of the window reads it)."""
    from test_a6_feedback_detach import A5, ALL, _cfg, _model

    cfg = _cfg(*A5, ALL, "model.cand_norm=rms")
    model = _model(cfg).to(device=device, dtype=torch.float32)
    N, d, K = cfg.model.n_cells, cfg.model.d, cfg.model.k_select
    g = torch.Generator().manual_seed(5)
    B = 3
    h = (torch.rand(B, N, d, generator=g) * 1.6 - 0.8).to(device)
    h[1, N - 1, d - 1] = float("nan")  # a cell no slot writes
    state = model.init_state(B, device).with_dense(h)
    cells = torch.arange(K, device=device).expand(B, K).contiguous()
    valid = torch.ones(B, K, dtype=torch.bool, device=device)
    y = torch.randn(B, K, d, generator=g).to(device)
    gate = torch.randn(B, K, d, generator=g).to(device)
    with torch.no_grad():
        after, _, top = window.cell_update(model.cell, y, gate, state, cells, valid)
    assert bool(torch.isnan(top[1])) and bool(torch.isfinite(top[[0, 2]]).all())
    assert torch.equal(top[[0, 2]], after.h.abs().amax(dim=(1, 2))[[0, 2]])
