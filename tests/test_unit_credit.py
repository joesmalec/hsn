"""``train.unit_credit`` (opt-in, docs/a6_takeover_followup.md): with ``executed`` the
straight-through unit weight of a slot differentiates the executed unit's score only. The hard
values of a selection never change, the executed unit's own derivative is the softmax's, and no
other unit receives a gradient from the slot."""

from __future__ import annotations

import pytest
import torch
from hsn.config import RunConfig, config_hash, load_config
from p1_11_support import restore_determinism  # noqa: F401
from test_a6_feedback_detach import A5, ALL, _cfg, _model
from test_p2_10_split_window import _assert_same_grads, _both
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

RECIPE = (*A5, ALL, "model.cand_norm=rms")


def _select(cfg: RunConfig, logits: Tensor):  # type: ignore[no-untyped-def]
    model = _model(cfg)
    m = cfg.model
    return model.router.select(
        logits, m.budget, m.k_select, m.k_min, m.routing, gate="ste", temperature=1.0
    )


def test_executed_credit_keeps_the_values_and_the_executed_units_derivative() -> None:
    soft_cfg, exec_cfg = _cfg(*RECIPE), _cfg(*RECIPE, "train.unit_credit=executed")
    n_cells, n_units = soft_cfg.model.n_cells, soft_cfg.model.n_units
    logits = torch.randn(3, n_cells, n_units, generator=torch.Generator().manual_seed(0))

    def unit_grad(cfg: RunConfig) -> tuple[Tensor, object]:
        leaf = logits.clone().requires_grad_(True)
        sel = _select(cfg, leaf)
        rows = torch.arange(leaf.shape[0])[:, None]
        slot_unit = sel.unit_weights[rows, sel.cell_idx, sel.unit_idx]  # [B, K]
        (grad,) = torch.autograd.grad((slot_unit * sel.valid).sum(), leaf)
        return grad, sel

    g_soft, s_soft = unit_grad(soft_cfg)
    g_exec, s_exec = unit_grad(exec_cfg)
    # the decision and every hard value are the same
    for name in ("cell_idx", "unit_idx", "valid", "alpha", "cell_mask", "unit_weights"):
        assert torch.equal(getattr(s_exec, name), getattr(s_soft, name)), name
    rows = torch.arange(logits.shape[0])[:, None]
    executed = torch.zeros_like(logits, dtype=torch.bool)
    executed[rows, s_soft.cell_idx, s_soft.unit_idx] = s_soft.valid
    # the executed unit's own derivative is the softmax's diagonal term
    torch.testing.assert_close(g_exec[executed], g_soft[executed], rtol=1e-6, atol=1e-9)
    assert bool((g_exec[executed] != 0).any())
    # no other unit is touched; under the softmax the other units of a selected cell are
    assert bool((g_exec[~executed] == 0).all())
    selected_cells = executed.any(dim=-1, keepdim=True).expand_as(executed)
    assert bool((g_soft[selected_cells & ~executed] != 0).any())


def test_split_window_equals_autograd_with_executed_credit() -> None:
    cfg = _cfg(*RECIPE, "train.unit_credit=executed")
    ref, got, ref_info, s_info = _both(cfg)
    _assert_same_grads(ref, got)
    for key in ("cell_idx", "unit_idx", "valid"):
        assert torch.equal(s_info[key], ref_info[key])


def test_unit_credit_is_opt_in_and_keeps_the_hash() -> None:
    base = _cfg(*RECIPE)
    assert base.train.unit_credit == "softmax"
    assert config_hash(_cfg(*RECIPE, "train.unit_credit=softmax")) == config_hash(base)
    assert config_hash(_cfg(*RECIPE, "train.unit_credit=executed")) != config_hash(base)
    with pytest.raises(ValueError, match="unit_credit"):
        _cfg(*RECIPE, "train.unit_credit=majority")


def test_unit_credit_is_hsn_only() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    gru = next(iter(sorted((root / "configs").rglob("*gru*.yaml"))), None)
    if gru is None:
        pytest.skip("no GRU configuration in this checkout")
    with pytest.raises(ValueError, match="HSN-only"):
        load_config(gru, overrides=["train.unit_credit=executed"])
