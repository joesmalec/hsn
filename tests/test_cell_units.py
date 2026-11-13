"""``model.cell_units = fixed`` (opt-in control for amendment A8, 13 Nov 2026): every cell ``i``
runs unit ``i mod M``; the router still chooses the cells. It separates the two things the
fixed-unit fast cells change at once (cells written at every position; units that are all in
use). No parameter is added and the default is the model as before, bit for bit."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import RunConfig, config_hash, load_config, split_window_scope_error
from hsn.model.hsn import HSN
from hsn.model.router import CLOSED_UNIT_LOGIT, Router
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, hsn_config, hsn_trainer
from test_a6_feedback_detach import A5, ALL
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
FIXED = ("model.k_select=4", "model.cell_units=fixed")


def _cfg(*extra: str) -> RunConfig:
    return load_config(RECIPE_A6, [*FIXED, *extra])


def _router(cfg: RunConfig) -> Router:
    return HSN.from_config(cfg).router


def _logits(B: int, N: int, M: int, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, N, M, generator=g)


def test_cell_units_is_opt_in() -> None:
    base = load_config(RECIPE_A6)
    assert base.model.cell_units == "routed"
    assert config_hash(load_config(RECIPE_A6, ["model.cell_units=routed"])) == config_hash(base)
    assert config_hash(load_config(RECIPE_A6, ["model.cell_units=fixed"])) != config_hash(base)
    with pytest.raises(ValueError, match="topk"):
        load_config(RECIPE_A6, ["model.cell_units=fixed", "model.routing=random"])


def test_split_and_fused_windows_refuse_fixed_units() -> None:
    recipe = ["model.k_select=8", "train.gate_mode=ste"]
    assert split_window_scope_error(load_config(RECIPE_A6, recipe)) is None
    reason = split_window_scope_error(load_config(RECIPE_A6, [*recipe, "model.cell_units=fixed"]))
    assert reason is not None and "cell_units" in reason


def test_no_parameter_is_added_and_the_default_router_is_untouched() -> None:
    routed, fixed = HSN.from_config(load_config(RECIPE_A6)), HSN.from_config(_cfg())
    assert routed.router.closed_units is None
    assert set(routed.state_dict()) == set(fixed.state_dict())
    assert sum(p.numel() for p in routed.parameters()) == sum(p.numel() for p in fixed.parameters())
    feats = torch.randn(2, routed.cfg.n_cells, routed.router.scorer.in_features)
    torch.testing.assert_close(routed.router(feats), routed.router.scorer(feats), rtol=0, atol=0)


def test_logits_close_every_unit_but_the_cells_own() -> None:
    cfg = _cfg()
    router = _router(cfg)
    N, M = cfg.model.n_cells, cfg.model.n_units
    feats = torch.randn(3, N, router.scorer.in_features)
    logits, raw = router(feats), router.scorer(feats)
    own = torch.arange(N) % M
    assert torch.equal(logits.argmax(dim=-1), own[None].expand(3, N))
    picked = logits.gather(2, own[None, :, None].expand(3, N, 1))
    assert torch.equal(picked, raw.gather(2, own[None, :, None].expand(3, N, 1)))
    closed = torch.ones(N, M, dtype=torch.bool)
    closed[torch.arange(N), own] = False
    assert bool((logits[:, closed] == CLOSED_UNIT_LOGIT).all())


@pytest.mark.parametrize("gate", [None, "ste", "soft"])
def test_selection_runs_each_cell_at_its_own_unit(gate: str | None) -> None:
    cfg = _cfg()
    router = _router(cfg)
    N, M, K = cfg.model.n_cells, cfg.model.n_units, cfg.model.k_select
    logits = router._close_units(_logits(4, N, M).requires_grad_(True))
    sel = router.select(logits, cfg.model.budget, K, cfg.model.k_min, "topk", gate=gate)
    assert bool(sel.valid.all())
    assert torch.equal(sel.unit_idx, sel.cell_idx % M)
    # the cells are the top K by each cell's score at its own unit
    own = (torch.arange(N) % M)[None, :, None].expand(4, N, 1)
    cell_scores = logits.detach().gather(2, own).squeeze(2)
    want = torch.sort(cell_scores, dim=1, descending=True, stable=True).indices[:, :K]
    assert torch.equal(sel.cell_idx, want)
    # the unit weights are exactly one-hot at the cell's own unit, in every gate mode
    onehot = torch.nn.functional.one_hot(torch.arange(N) % M, M).float()[None].expand(4, N, M)
    assert torch.equal(sel.unit_weights.detach(), onehot)


def test_no_gradient_reaches_a_closed_pair() -> None:
    cfg = _cfg()
    router = _router(cfg)
    N, M, K = cfg.model.n_cells, cfg.model.n_units, cfg.model.k_select
    raw = _logits(2, N, M).requires_grad_(True)
    sel = router.select(
        router._close_units(raw), cfg.model.budget, K, cfg.model.k_min, "topk", gate="ste"
    )
    weights = torch.randn(2, K, generator=torch.Generator().manual_seed(3))
    (sel.alpha * weights).sum().backward()
    assert raw.grad is not None
    closed = torch.ones(N, M, dtype=torch.bool)
    closed[torch.arange(N), torch.arange(N) % M] = False
    assert bool((raw.grad[:, closed] == 0).all())
    assert float(raw.grad[:, ~closed].abs().sum()) > 0.0


COMPILED = {
    "ste": (("train.gate_mode=ste",), True),
    "soft": ((), True),  # the soft warm-up: the masked round
    "eval": ((), False),
}


@pytest.mark.parametrize("name", list(COMPILED))
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_equals_eager(name: str) -> None:
    overrides, training = COMPILED[name]
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        *overrides,
        *FIXED,
    )
    model = p28._model(cfg, training=training, scale=0.3)
    ctx = p28._ctx()
    window = p28._window(model, ctx)
    tokens, cx, state, reset = p28._case_inputs(model, True)
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    got = p28._run(lambda: window(tokens, cx, state, reset))
    tol = p28.TOL[(ctx.device.type, "fp32")]
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    if training:
        assert p28._grad_diff(p28._grads(model, got), p28._grads(model, ref)) <= tol["grads"]


def test_recipe_with_fixed_units_trains(tmp_path: Path) -> None:
    steps = 30
    cfg = hsn_config(
        steps,
        *A5,
        ALL,
        "model.cand_norm=rms",
        "model.n_cells=6",
        "model.k_select=4",
        "train.soft_until_step=5",
        "train.lr=3e-3",
        "model.cell_units=fixed",
    )
    trainer = hsn_trainer(cfg, tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.05
