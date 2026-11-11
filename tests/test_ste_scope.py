"""``train.ste_scope=all`` (opt-in, docs/a6_takeover_followup.md): the routed training window runs
the masked round. The decision and every forward value are the sparse round's; the gradient of
every parameter outside the router is the sparse round's too (an unselected cell is not written,
a unit that was not chosen has weight 0); only the scorer's gradient changes, because the cell
mask's and the unit weights' straight-through terms now exist for every cell and unit."""

from __future__ import annotations

import pytest
import torch
from hsn.config import config_hash
from hsn.train.split import scope_error
from p1_11_support import restore_determinism  # noqa: F401
from test_a6_feedback_detach import A5, ALL, _cfg, _model, _start, _window

pytestmark = pytest.mark.usefixtures("restore_determinism")

RECIPE = (*A5, ALL, "model.cand_norm=rms")


def _window_grads(cfg):  # type: ignore[no-untyped-def]
    model = _model(cfg)
    model.train()
    tokens, ctx, reset = _window(cfg)
    reset[0, 0] = False
    reset[1, 5] = True
    state = _start(model, tokens.shape[0])
    mask = torch.zeros_like(reset)
    mask[:, 3::4] = True
    logits, new_state, info = model(tokens, ctx, state, reset)
    loss = logits[mask].square().sum() + (info["z_hat"] - info["z_target"]).square().mean()
    names = [n for n, _ in model.named_parameters()]
    grads = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    return model, logits, new_state, info, dict(zip(names, grads, strict=True))


def test_masked_training_round_keeps_the_values_and_widens_the_scorers_gradient() -> None:
    _, logits, state, info, sparse = _window_grads(_cfg(*RECIPE))
    wide_model, w_logits, w_state, w_info, wide = _window_grads(
        _cfg(*RECIPE, "train.ste_scope=all")
    )
    for key in ("cell_idx", "unit_idx", "valid"):
        assert torch.equal(w_info[key], info[key])
    torch.testing.assert_close(w_logits, logits, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(w_state.h, state.h, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(w_state.q, state.q, rtol=1e-9, atol=1e-11)
    assert {n for n, g in wide.items() if g is None} == {n for n, g in sparse.items() if g is None}
    changed = []
    for name, want in sparse.items():
        if want is None:
            continue
        if name.startswith("router."):
            assert bool(torch.isfinite(wide[name]).all())
            changed.append(float((wide[name] - want).abs().max()))
        else:
            torch.testing.assert_close(
                wide[name], want, rtol=1e-7, atol=1e-10, msg=lambda m, n=name: n + m
            )
    assert max(changed) > 0.0  # the unselected cells and units now reach the scorer
    wide_model.eval()
    assert wide_model._window_mode() == "sparse"  # evaluation is the sparse round, as before


def test_ste_scope_is_opt_in_keeps_the_hash_and_is_outside_the_split_window() -> None:
    base = _cfg(*RECIPE)
    assert base.train.ste_scope == "selected"
    assert config_hash(_cfg(*RECIPE, "train.ste_scope=selected")) == config_hash(base)
    wide = _cfg(*RECIPE, "train.ste_scope=all")
    assert config_hash(wide) != config_hash(base)
    assert "ste_scope" in (scope_error(wide) or "")
    with pytest.raises(ValueError, match="ste_scope"):
        _cfg(*RECIPE, "train.ste_scope=all", "hardware.window=split")
