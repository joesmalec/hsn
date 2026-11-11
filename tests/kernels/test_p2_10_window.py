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
