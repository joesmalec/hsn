"""P2.10: the split window (``hardware.window=split``, :mod:`hsn.train.split`) computes the
model's own gradients: a forward sweep without autograd, one batched replay and the linear state
recurrence equal :meth:`HSN.forward`'s autograd to rounding (float64 on the CPU)."""

from __future__ import annotations

import pytest
import torch
from hsn.config import RunConfig, canonical_json, config_hash
from hsn.kernels import native
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from hsn.train.split import SplitWindow, recurrence_backward, scope_error
from p1_11_support import restore_determinism  # noqa: F401
from test_a6_feedback_detach import A5, ALL, ROUTED, _cfg, _model, _start, _window
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

RECIPE = (*A5, ALL, "model.cand_norm=rms")


# --------------------------------------------------------------------------------------------------
# The state recurrence against autograd of the real update and reset
# --------------------------------------------------------------------------------------------------


def test_recurrence_equals_autograd_of_the_update_chain() -> None:
    g = torch.Generator().manual_seed(7)
    B, T, N, K, d = 3, 9, 5, 3, 4
    f64 = torch.float64
    c = torch.rand(B, T, K, d, generator=g, dtype=f64, requires_grad=True)
    b = torch.rand(B, T, K, d, generator=g, dtype=f64, requires_grad=True)
    # distinct cells per row and position; padded slots hold cell 0, which another slot of the
    # same row may genuinely select (the collision the dummy row exists for)
    cells = torch.stack(
        [torch.stack([torch.randperm(N, generator=g)[:K] for _ in range(T)]) for _ in range(B)]
    )
    valid = torch.rand(B, T, K, generator=g) > 0.3
    cells = torch.where(valid, cells, torch.zeros_like(cells))
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[0, 0] = reset[1, 4] = reset[2, 7] = reset[2, 8] = True
    w_pre = torch.randn(B, T, d, generator=g, dtype=f64)
    w_post = torch.randn(B, T, d, generator=g, dtype=f64)
    w_sel = torch.randn(B, T, K, d, generator=g, dtype=f64)

    h0 = torch.rand(B, N, d, generator=g, dtype=f64)
    state = CellGraphState._make(
        h=h0,
        q=h0.mean(dim=1),
        age=torch.zeros(B, N, dtype=torch.int64),
        pos=torch.zeros(B, dtype=torch.int64),
    )
    total = torch.zeros((), dtype=f64)
    h_old = []
    for t in range(T):
        state = state.reset(reset[:, t])
        total = total + (state.q * w_pre[:, t]).sum()
        index = cells[:, t, :, None].expand(B, K, d)
        h_old.append(state.h.gather(1, index).detach())
        state = native.gated_delta_update(state, cells[:, t], c[:, t], b[:, t], valid[:, t])
        total = total + (state.q * w_post[:, t]).sum()
        chosen = state.h.gather(1, index)
        total = total + (torch.where(valid[:, t, :, None], chosen, 0.0) * w_sel[:, t]).sum()
    want_c, want_b = torch.autograd.grad(total, [c, b])

    got_c, got_b = recurrence_backward(
        c.detach(),
        b.detach(),
        torch.stack(h_old, dim=1),
        cells,
        valid,
        reset,
        N,
        w_pre,
        w_post,
        torch.where(valid[..., None], w_sel, 0.0),
    )
    torch.testing.assert_close(got_c, want_c, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(got_b, want_b, rtol=1e-12, atol=1e-14)
    assert bool((got_c[~valid] == 0).all()) and bool((got_b[~valid] == 0).all())


def test_recurrence_keeps_a_padded_nan_out() -> None:
    """Autograd writes zeros on padded slots whatever their values; so does the sweep."""
    B, T, N, K, d = 1, 2, 3, 2, 2
    c = torch.full((B, T, K, d), 0.5, dtype=torch.float64)
    c[0, 0, 1] = float("nan")
    b = torch.full((B, T, K, d), 0.25, dtype=torch.float64)
    cells = torch.tensor([[[1, 0], [2, 0]]])
    valid = torch.tensor([[[True, False], [True, False]]])
    reset = torch.zeros(B, T, dtype=torch.bool)
    ones = torch.ones(B, T, d, dtype=torch.float64)
    dc, db = recurrence_backward(
        c, b, torch.zeros_like(c), cells, valid, reset, N, None, ones, None
    )
    assert bool(torch.isfinite(dc).all()) and bool(torch.isfinite(db).all())


# --------------------------------------------------------------------------------------------------
# End to end: the split window's gradients are HSN.forward's
# --------------------------------------------------------------------------------------------------


def _objective(logits: Tensor, info: dict[str, Tensor], mask: Tensor) -> Tensor:
    """Every gradient-carrying output of a window: the logits at the loss positions, the
    prediction, the expected cost and the logits the balance term reads."""
    task = logits[mask].square().sum() if bool(mask.any()) else logits.sum() * 0.0
    pred = (info["z_hat"] - info["z_target"]).square().mean() + 0.1 * info["z"].square().mean()
    cost = info["expected_cost"].square().sum()
    balance = info["unit_logits"].square().mean()
    return task + 0.7 * pred + 0.3 * cost + 0.2 * balance


def _both(cfg: RunConfig, *, loss_positions: bool = True) -> tuple[dict, dict, dict, dict]:
    """(reference grads, split grads, reference info, split info) on one window with interior
    resets and a carried, non-zero start state in one row."""
    model = _model(cfg)
    tokens, ctx, reset = _window(cfg)
    reset[0, 0] = False  # row 0 carries its start state into the window
    reset[1, 5] = True
    reset[2, 9] = True
    reset[2, 10] = True
    state = _start(model, tokens.shape[0])
    mask = torch.zeros_like(reset)
    if loss_positions:
        mask[:, 3::4] = True
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]

    logits, ref_state, ref_info = model(tokens, ctx, state, reset)
    ref = torch.autograd.grad(_objective(logits, ref_info, mask), params, allow_unused=True)

    window = SplitWindow(model, cfg, None)
    s_logits, s_state, s_info = window(tokens, ctx, state, reset)
    got = torch.autograd.grad(_objective(s_logits, s_info, mask), params, allow_unused=True)

    torch.testing.assert_close(s_logits, logits, rtol=1e-10, atol=1e-12)
    for field in ("h", "q"):
        torch.testing.assert_close(
            getattr(s_state, field), getattr(ref_state, field), rtol=1e-10, atol=1e-12
        )
    assert torch.equal(s_state.age, ref_state.age) and torch.equal(s_state.pos, ref_state.pos)
    return dict(zip(names, ref, strict=True)), dict(zip(names, got, strict=True)), ref_info, s_info


def _assert_same_grads(ref: dict, got: dict) -> None:
    assert {n for n, g in ref.items() if g is None} == {n for n, g in got.items() if g is None}
    live = 0
    for name, want in ref.items():
        if want is None:
            continue
        torch.testing.assert_close(
            got[name], want, rtol=1e-9, atol=1e-12, msg=lambda m, n=name: n + m
        )
        live += int(bool(want.abs().max() > 0))
    assert live >= 20  # the paths are live: this compares real gradients, not zeros


@pytest.mark.parametrize("extra", [(), ("model.cand_norm=none",), ("model.lambda_cost=0.05",)])
def test_split_gradients_equal_autograd(extra: tuple[str, ...]) -> None:
    cfg = _cfg(*RECIPE, *extra)
    ref, got, ref_info, s_info = _both(cfg)
    _assert_same_grads(ref, got)
    for key in ("cell_idx", "unit_idx", "valid"):
        assert torch.equal(s_info[key], ref_info[key])
    for key in ("cost", "explored", "refreshed", "dense_fallback", "state_abs_max"):
        torch.testing.assert_close(s_info[key], ref_info[key], rtol=1e-10, atol=1e-12)
    for key in ("eps_abs_mean", "alpha_entropy", "expected_cost", "route_log_prob"):
        torch.testing.assert_close(s_info[key], ref_info[key], rtol=1e-9, atol=1e-12)
    assert set(s_info) == set(ref_info)


def test_split_gradients_with_refresh_probes() -> None:
    """A small ``age_max`` makes the refresh probes replace slots: the replay follows the
    recorded slots and the threshold of the scored decision before the probes."""
    cfg = _cfg(*RECIPE, "model.age_max=2")
    ref, got, ref_info, s_info = _both(cfg)
    assert float(ref_info["refreshed"].sum()) > 0
    _assert_same_grads(ref, got)
    torch.testing.assert_close(s_info["refreshed"], ref_info["refreshed"])


def test_split_gradients_with_padded_slots() -> None:
    """A budget that cannot hold ``k_select`` cells pads the selection."""
    cfg = _cfg(*RECIPE, "model.budget=0.1")
    ref, got, ref_info, _ = _both(cfg)
    assert not bool(ref_info["valid"].all()) and bool(ref_info["valid"].any())
    _assert_same_grads(ref, got)


def test_split_gradients_without_selected_readout() -> None:
    cfg = _cfg(*ROUTED, ALL)
    ref, got, _, _ = _both(cfg)
    assert {n for n, g in ref.items() if g is None} == {n for n, g in got.items() if g is None}
    for name, want in ref.items():
        if want is not None:
            torch.testing.assert_close(got[name], want, rtol=1e-9, atol=1e-12)


def test_split_window_without_loss_positions_gives_the_same_none_set() -> None:
    cfg = _cfg(*RECIPE)
    ref, got, _, _ = _both(cfg, loss_positions=False)
    assert {n for n, g in ref.items() if g is None} == {n for n, g in got.items() if g is None}
    for name, want in ref.items():
        if want is not None:
            torch.testing.assert_close(got[name], want, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# Scope and configuration
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        (*A5, "train.feedback_detach=[summary,eps,state]"),
        (*A5, "train.feedback_detach=[eps,state,router]"),
        (*A5,),
        ("model.routing=topk", "train.gate_mode=soft", ALL),
        (*A5, ALL, "train.p_dense=0.25"),
        (*A5, ALL, "train.epsilon_start=0.1", "train.epsilon_end=0.1"),
        (*A5, ALL, "train.state_grad_clip=1.0"),
        (*A5, ALL, "train.lambda_u=0.1", "train.probe_horizon=4"),
        (*A5, ALL, "train.soft_until_step=2"),
        (ALL,),
    ],
)
def test_out_of_scope_configurations_are_refused(overrides: tuple[str, ...]) -> None:
    cfg = _cfg(*overrides)
    assert scope_error(cfg) is not None
    model = HSN.from_config(cfg)
    with pytest.raises(ValueError, match="hardware.window=split"):
        SplitWindow(model, cfg, None)


def test_the_key_is_opt_in_and_keeps_the_hash() -> None:
    base = _cfg(*RECIPE)
    assert base.hardware.window == "autograd"
    assert '"window"' not in canonical_json(base)
    split = _cfg(*RECIPE, "hardware.window=split")
    assert scope_error(split) is None
    assert config_hash(split) != config_hash(base)


def test_split_window_falls_back_outside_training() -> None:
    cfg = _cfg(*RECIPE)
    model = _model(cfg)
    tokens, ctx, reset = _window(cfg)
    state = _start(model, tokens.shape[0])
    window = SplitWindow(model, cfg, None)
    model.eval()
    with torch.no_grad():
        logits, _, info = window(tokens, ctx, state, reset)
        want, _, _ = model(tokens, ctx, state, reset)
    assert torch.equal(logits, want) and window.calls == 0
