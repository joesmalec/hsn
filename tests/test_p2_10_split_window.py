"""P2.10: the split window (``hardware.window=split``, :mod:`hsn.train.split`) computes the
model's own gradients: a forward sweep without autograd, one batched replay and the linear state
recurrence equal :meth:`HSN.forward`'s autograd to rounding (float64 on the CPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hsn.config import RunConfig, canonical_json, config_hash
from hsn.kernels import native
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from hsn.train.split import SplitWindow, recurrence_backward, scope_error
from p1_11_support import restore_determinism  # noqa: F401
from test_a6_feedback_detach import A5, ALL, ROUTED, _cfg, _model, _start, _window
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
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


def _both(
    cfg: RunConfig, *, loss_positions: bool = True, draw_seed: int = 0, train_step: int = 0
) -> tuple[dict, dict, dict, dict]:
    """(reference grads, split grads, reference info, split info) on one window with interior
    resets and a carried, non-zero start state in one row, at training step ``train_step``."""
    model = _model(cfg)
    model.set_train_step(train_step)
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

    torch.manual_seed(draw_seed)  # the dense-fallback draw of the window (train.p_dense)
    logits, ref_state, ref_info = model(tokens, ctx, state, reset)
    ref = torch.autograd.grad(_objective(logits, ref_info, mask), params, allow_unused=True)

    window = SplitWindow(model, cfg, None)
    torch.manual_seed(draw_seed)
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


@pytest.mark.parametrize("draw_seed", [0, 1, 2, 3])
def test_split_gradients_with_dense_fallback_rows(draw_seed: int) -> None:
    """``train.p_dense``: the rows drawn for the dense round are replayed through the dense
    step and their own recurrence; the other rows are untouched by them."""
    cfg = _cfg(*RECIPE, "train.p_dense=0.5", "model.lambda_cost=0.05")
    ref, got, ref_info, s_info = _both(cfg, draw_seed=draw_seed)
    _assert_same_grads(ref, got)
    torch.testing.assert_close(s_info["dense_fallback"], ref_info["dense_fallback"])
    for key in ("cost", "expected_cost", "eps_abs_mean", "alpha_entropy"):
        torch.testing.assert_close(s_info[key], ref_info[key], rtol=1e-9, atol=1e-12)


def test_dense_fallback_draws_cover_mixed_and_full_windows() -> None:
    """The four draws above are not all of one kind: some rows dense, and all or none."""
    cfg = _cfg(*RECIPE, "train.p_dense=0.5")
    model = _model(cfg)
    state = _start(model, cfg.train.batch)
    counts = set()
    for seed in range(4):
        torch.manual_seed(seed)
        rows = model._draw_dense_rows(cfg.train.batch, state)
        counts.add(0 if rows is None else int(rows.sum()))
    assert len(counts) >= 2 and any(0 < n < cfg.train.batch for n in counts)


def test_dense_recurrence_equals_autograd() -> None:
    from hsn.train.split import dense_recurrence_backward

    g = torch.Generator().manual_seed(5)
    Bd, T, N, d = 2, 7, 4, 3
    f64 = torch.float64
    c = torch.rand(Bd, T, N, d, generator=g, dtype=f64, requires_grad=True)
    b = torch.rand(Bd, T, N, d, generator=g, dtype=f64, requires_grad=True)
    reset = torch.zeros(Bd, T, dtype=torch.bool)
    reset[0, 0] = reset[1, 3] = True
    w_pre = torch.randn(Bd, T, d, generator=g, dtype=f64)
    w_post = torch.randn(Bd, T, d, generator=g, dtype=f64)
    h = torch.rand(Bd, N, d, generator=g, dtype=f64)
    q = h.mean(dim=1)
    total = torch.zeros((), dtype=f64)
    olds = []
    for t in range(T):
        h = h.masked_fill(reset[:, t, None, None], 0)
        q = q.masked_fill(reset[:, t, None], 0)
        total = total + (q * w_pre[:, t]).sum()
        olds.append(h.detach())
        h = (1 - b[:, t]) * h + b[:, t] * c[:, t]
        q = h.mean(dim=1)
        total = total + (q * w_post[:, t]).sum()
    want_c, want_b = torch.autograd.grad(total, [c, b])
    got_c, got_b = dense_recurrence_backward(
        c.detach(), b.detach(), torch.stack(olds, dim=1), reset, N, w_pre, w_post
    )
    torch.testing.assert_close(got_c, want_c, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(got_b, want_b, rtol=1e-12, atol=1e-14)


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


def test_recorded_decision_credits_the_unit_that_ran() -> None:
    """The replay recomputes the scorer's logits, and under bf16 they can round a near-tie
    between two units the other way than the sweep's did. :meth:`Router.select` reads a cell's
    score at the unit it runs (its ``amax`` and its decision are one tensor), so the cell
    mask's gradient must reach the unit that ran in the replay too: with the first version's
    ``amax`` of the recomputed scores it went to the other unit (on the A6 recipe 0.01 to 1.3 %
    of the slots, and up to half of one unit's scorer-row gradient while a route was being
    taken over)."""
    model = _model(_cfg(*RECIPE))
    router = model.router
    n_cells, n_units = model.cell.n_cells, model.cell.units.n_units
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, n_cells, n_units, generator=g)
    cells = torch.stack([torch.randperm(n_cells, generator=g)[:3] for _ in range(2)])
    units = logits.argmax(-1).gather(1, cells)
    valid = torch.ones_like(cells, dtype=torch.bool)
    theta = torch.zeros(2)
    rows = torch.arange(2)[:, None]

    def mask_grad(values: Tensor, executed: bool) -> Tensor:
        leaf = values.clone().requires_grad_(True)
        sel = router.finish_recorded(
            leaf, cells, units, valid, theta, mode="topk", gate="ste", executed_scores=executed
        )
        (grad,) = torch.autograd.grad(sel.cell_mask.gather(1, cells).sum(), leaf)
        return grad

    # the decision's own scores: the executed unit is the largest, the two readings are one
    assert torch.equal(mask_grad(logits, True), mask_grad(logits, False))
    # recomputed scores that put another unit a hair above the one that ran
    other = (units + 1) % n_units
    bumped = logits.clone()
    bumped[rows, cells, other] = bumped[rows, cells, units] + 1e-3
    executed, largest = mask_grad(bumped, True), mask_grad(bumped, False)
    assert bool((executed[rows, cells, units] != 0).all())
    assert bool((executed[rows, cells, other] == 0).all())
    assert bool((largest[rows, cells, other] != 0).all())
    assert bool((largest[rows, cells, units] == 0).all())
    # the unit weights' path follows the recorded unit in both
    sel = router.finish_recorded(bumped, cells, units, valid, theta, mode="topk", gate="ste")
    assert torch.equal(sel.unit_idx, units)
    assert torch.equal(sel.unit_weights[rows, cells].argmax(-1), units)


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
        (*A5, ALL, "train.epsilon_start=0.1", "train.epsilon_end=0.1"),
        (*A5, ALL, "train.state_grad_clip=1.0"),
        (*A5, ALL, "train.lambda_u=0.1", "train.probe_horizon=4"),
        (ALL,),
    ],
)
def test_out_of_scope_configurations_are_refused(overrides: tuple[str, ...]) -> None:
    cfg = _cfg(*overrides)
    assert scope_error(cfg) is not None
    model = HSN.from_config(cfg)
    with pytest.raises(ValueError, match="hardware.window=split"):
        SplitWindow(model, cfg, None)


@pytest.mark.parametrize(
    "overrides",
    [
        (*A5, "train.feedback_detach=[summary,eps,state]"),
        ("model.routing=topk", "train.gate_mode=soft", ALL),
        (*A5, ALL, "train.epsilon_start=0.1", "train.epsilon_end=0.1"),
        (ALL,),
    ],
)
@pytest.mark.parametrize("window", ["split", "fused"])
def test_the_configuration_refuses_the_key_outside_its_scope(
    overrides: tuple[str, ...], window: str
) -> None:
    """Loading is where a spec is expanded and queued: the refusal comes there."""
    with pytest.raises(ValueError, match="hardware.window"):
        _cfg(*overrides, f"hardware.window={window}")
    assert _cfg(*RECIPE, "train.p_dense=0.25", f"hardware.window={window}").hardware.window


def test_the_key_is_opt_in_and_keeps_the_hash() -> None:
    base = _cfg(*RECIPE)
    assert base.hardware.window == "autograd"
    assert '"window"' not in canonical_json(base)
    split = _cfg(*RECIPE, "hardware.window=split")
    assert scope_error(split) is None
    assert config_hash(split) != config_hash(base)


def test_soft_warm_up_runs_the_fallback_then_the_split_window() -> None:
    """``train.soft_until_step``: the warm-up's windows (the soft gate: every cell runs, with
    autograd) go to the fallback, bit for bit the model's own forward on its own graph; from
    the first hard-gate step the split window runs and returns the model's gradients."""
    cfg = _cfg(*RECIPE, "train.soft_until_step=3", "hardware.window=split")
    assert scope_error(cfg) is None
    ref, got, _, _ = _both(cfg, train_step=3)
    _assert_same_grads(ref, got)
    model = _model(cfg)
    tokens, ctx, reset = _window(cfg)
    state = _start(model, tokens.shape[0])
    window = SplitWindow(model, cfg, None)
    model.set_train_step(2)
    assert not window.splits_now()
    logits, _, _ = window(tokens, ctx, state, reset)
    want, _, _ = model(tokens, ctx, state, reset)
    assert torch.equal(logits, want) and logits.requires_grad and window.calls == 0
    model.set_train_step(3)
    assert window.splits_now()
    window(tokens, ctx, state, reset)
    assert window.calls == 1


def test_training_switches_from_the_soft_warm_up_to_the_split_window(tmp_path: Path) -> None:
    """A split run with ``train.soft_until_step`` trains: its warm-up steps are the autograd
    run's (the same losses, bit for bit), the split window runs every step after them
    (statistics steps included) and the losses stay with the autograd run's to rounding."""
    from p1_15_support import datasets, hsn_config, hsn_trainer

    recipe = (*RECIPE, "train.soft_until_step=3", "train.stats_every=2")
    native = hsn_trainer(hsn_config(8, *recipe), tmp_path / "native")
    want = native.fit(*datasets(), 8)
    split = hsn_trainer(hsn_config(8, *recipe, "hardware.window=split"), tmp_path / "split")
    got = split.fit(*datasets(), 8)
    assert split._split is not None and split._split.calls == 5  # steps 3 to 7
    assert [s.loss for s in got.steps[:3]] == [s.loss for s in want.steps[:3]]
    for mine, theirs in zip(got.steps[3:], want.steps[3:], strict=True):
        assert mine.loss == pytest.approx(theirs.loss, rel=1e-4)
        assert mine.grad_norm == pytest.approx(theirs.grad_norm, rel=1e-3)


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


# --------------------------------------------------------------------------------------------------
# CUDA: the compiled forward sweep (CUDA Graphs, the sweep twin) and the recurrence kernels
# --------------------------------------------------------------------------------------------------

CUDA_RECIPE = (
    "model.routing=topk",
    "train.gate_mode=ste",
    "model.key_address=true",
    "model.selected_readout=true",
    "model.cand_norm=rms",
    ALL,
)


def _cuda_case(*extra: str) -> tuple:
    import test_p2_8_compile as p28

    ctx = p28._cuda_ctx()
    cfg = p28._cfg(*CUDA_RECIPE, *extra)
    model = p28._model(cfg, training=True, scale=0.3).to(ctx.device)
    compiled = p28._window(model, ctx, backend="inductor")
    tokens, cx, state, reset = p28._case_inputs(model, True)
    return p28, cfg, model, compiled, (tokens, cx, state, reset)


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks", "_release_compiled_state")
@pytest.mark.parametrize("p_dense", [0.0, 0.5])
def test_split_window_compiled_cuda_equals_eager(p_dense: float) -> None:
    """fp32 on CUDA: the split window over a compiled, CUDA-graph sweep (warm-up, recording,
    replay) takes the eager window's decisions and returns its logits and gradients to the
    tolerance of the compiled autograd window (P2.8), with and without dense-fallback rows."""
    p28, cfg, model, compiled, inputs = _cuda_case(f"train.p_dense={p_dense}")
    tokens, cx, state, reset = inputs
    mask = torch.zeros_like(reset)
    mask[:, 1::3] = True
    params = [p for p in model.parameters()]
    torch.manual_seed(5)
    logits, final, info = model(tokens, cx, state, reset)
    if p_dense:
        assert 0.0 < float(info["dense_fallback"].mean()) < 1.0
    assert p28._margin(info, model.cfg.k_select) > p28.MIN_MARGIN
    want = torch.autograd.grad(_objective(logits, info, mask), params, allow_unused=True)
    split = SplitWindow(model, cfg, compiled)
    for _ in range(3):  # warm-up, recording, replay
        torch.manual_seed(5)
        s_logits, s_final, s_info = split(tokens, cx, state, reset)
        got = torch.autograd.grad(_objective(s_logits, s_info, mask), params, allow_unused=True)
    tol = p28.TOL[("cuda", "fp32")]
    for key in ("cell_idx", "unit_idx", "valid"):
        assert torch.equal(s_info[key], info[key]), key
    assert float((s_logits - logits).abs().max()) <= tol["values"]
    for name in ("h", "q"):
        gap = (getattr(s_final, name) - getattr(final, name)).abs().max()
        assert float(gap) <= tol["values"], name
    assert torch.equal(s_final.age, final.age) and torch.equal(s_final.pos, final.pos)
    # this objective's gradients are large (a sum of squared logits): relative to each tensor
    for g, w in zip(got, want, strict=True):
        assert (g is None) == (w is None)
        if w is not None:
            assert float((g - w).abs().max()) <= 2e-5 * float(w.abs().max()) + 1e-6
    assert split.calls == 3 and compiled.compile_seconds > 0.0


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks", "_release_compiled_state")
def test_sweep_twin_tape_is_the_models_in_fp32() -> None:
    """Without autocast the twin's weights are plain copies: its tape is the model's bit for
    bit, also after the weights change (the twin is refreshed every window)."""
    from hsn.train.compile import TAPE_KEYS

    p28, cfg, model, compiled, (tokens, cx, state, reset) = _cuda_case()
    for round_ in range(2):
        tapes = {}
        with torch.no_grad():
            pro = model._window_prologue(
                tokens, cx, state, reset, model._window_mode(), reset_first=True
            )
            for twin in (True, False):
                for _ in range(3):
                    tape = compiled.tape(pro, state, reset, twin=twin)
                tapes[twin] = {k: v.clone() for k, v in tape.items()}
            for key in (*TAPE_KEYS, "h", "q", "age", "pos"):
                assert torch.equal(tapes[True][key], tapes[False][key]), (round_, key)
            for p in model.parameters():  # an optimiser step's worth of change
                p.add_(0.01 * torch.randn_like(p))


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks", "_release_compiled_state")
@pytest.mark.parametrize("extra", [(), ("model.age_max=2",)])
@pytest.mark.parametrize("resets", [False, True])
def test_fused_sweep_tape_is_the_split_sweeps_in_fp32(extra: tuple[str, ...], resets: bool) -> None:
    """``hardware.window=fused``: the sweep whose decision is the selection kernel and whose
    tape is written in place records the tape of the sweep that calls :meth:`Router.select`:
    every decision, age and position bit for bit (the kernel takes the router's decision), the
    state values to the rounding of two compilations of the same arithmetic (P2.8's tolerance),
    with and without refresh probes, and with resets at a block's first, interior and last
    positions (the end-of-round kernel applies the next position's reset)."""
    from hsn.kernels import window as window_kernels
    from hsn.train.compile import TAPE_KEYS

    p28, cfg, model, compiled, (tokens, cx, state, reset) = _cuda_case(*extra)
    if not window_kernels.available():
        pytest.skip("window kernels not built (scripts/build_window_kernels.py)")
    window_kernels.prepare(0)
    if resets:  # blocks of 4 positions: the first of a block, an interior one, the last
        reset = reset.clone()
        reset[0, 0] = True
        reset[1, 2] = True
        reset[0, 3] = True
        reset[1, 4] = True
        reset[0, int(reset.shape[1]) - 1] = True
    tapes = {}
    with torch.no_grad():
        pro = model._window_prologue(
            tokens, cx, state, reset, model._window_mode(), reset_first=True
        )
        for fused in (False, True):
            for _ in range(3):  # warm-up, recording, replay
                tape = compiled.tape(pro, state, reset, fused=fused)
            tapes[fused] = {k: v.clone() for k, v in tape.items()}
    tol = p28.TOL[("cuda", "fp32")]["values"]
    for key in (*TAPE_KEYS, "h", "q", "age", "pos"):
        got, want = tapes[True][key], tapes[False][key]
        if got.dtype.is_floating_point:
            assert float((got - want).abs().max()) <= tol, key
        else:
            assert torch.equal(got, want), key
    for key in ("explored", "refreshed"):
        assert torch.equal(tapes[True][key], tapes[False][key]), key
    if extra:
        assert float(tapes[True]["refreshed"].sum()) > 0.0
    fused_twins = [twin for (_, is_fused), twin in compiled._sweep_twins.items() if is_fused]
    assert fused_twins and all(twin.merged for twin in fused_twins)  # the merged products ran


def test_forward_static_chosen_is_the_static_pass() -> None:
    """The fused sweep's unit pass (:meth:`UnitBank.forward_static_chosen`: every unit's first
    layer, then the chosen unit's activation and second layer only) returns the static pass's
    values in float64, and its rider columns are the extra linear map of the same input."""
    from hsn.model.units import UnitBank

    torch.manual_seed(3)
    bank = UnitBank(5, 6, 16, "gelu", init="fan_in").double()
    with torch.no_grad():
        bank.b1.normal_()
        bank.b2.normal_()
    P = 11
    v = torch.randn(P, bank.in_dim, dtype=torch.float64)
    index = torch.randint(0, 5, (P,))
    with torch.no_grad():
        want = bank.forward_static(v, index)
        got, rider = bank.forward_static_chosen(
            v, index, w1_flat=bank._w1_flat(), w2_flat=bank._w2_flat()
        )
        assert rider is None
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-13)
        gate = torch.nn.Linear(bank.in_dim, 6).double()
        got, rider = bank.forward_static_chosen(
            v,
            index,
            w1_flat=torch.cat([bank._w1_flat(), gate.weight.t()], dim=1),
            w2_flat=bank._w2_flat(),
            b1_flat=torch.cat([bank.b1.reshape(-1), gate.bias]),
        )
        assert rider is not None
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(rider, gate(v), rtol=1e-12, atol=1e-13)


def test_sweep_twin_merged_layouts_are_the_separate_maps() -> None:
    """The merged layouts of a fused :class:`SweepTwin` (float64, CPU): ``S`` with ``P``, the
    scorer with ``V``, and the unit bank's first layer with the gate each return the separate
    maps' values on random inputs, before and after the weights change (the zero blocks stay
    zero across refreshes)."""
    from hsn.train.compile import SweepTwin, sweep_merge_error

    model = _model(_cfg(*RECIPE))
    assert sweep_merge_error(model) is None
    sweep = SweepTwin(model, torch.float64, fused=True)
    assert sweep.merged
    twin, cell, bank = sweep.twin, model.cell, model.cell.units
    d, m_units = model.cfg.d, bank.n_units
    g = torch.Generator().manual_seed(9)
    f64 = torch.float64
    for round_ in range(2):
        with torch.no_grad(), cell.window():
            sweep.refresh()
            x = torch.randn(7, model.shared.in_dim, generator=g, dtype=f64)
            hidden = model.shared.act(
                torch.addmm(twin.get_buffer("sweep_front_b1"), x, twin.get_buffer("sweep_front_w1"))
            )
            both = torch.addmm(
                twin.get_buffer("sweep_front_b2"), hidden, twin.get_buffer("sweep_front_w2")
            )
            du = model.shared.out_dim
            torch.testing.assert_close(both[:, :du], model.shared(x), rtol=1e-11, atol=1e-12)
            torch.testing.assert_close(
                both[:, du:], model.predictor(x[:, d : 2 * d]), rtol=1e-11, atol=1e-12
            )
            net = model.router.scorer.net
            feats = torch.randn(13, net.in_features, generator=g, dtype=f64)
            scored = torch.addmm(
                twin.get_buffer("sweep_score_b"), feats, twin.get_buffer("sweep_score_w")
            )
            n_out = net.out_features
            torch.testing.assert_close(scored[:, :n_out], net(feats), rtol=1e-11, atol=1e-12)
            torch.testing.assert_close(
                scored[:, n_out:], cell.V(feats[:, 2 * d : 3 * d]), rtol=1e-11, atol=1e-12
            )
            v = torch.randn(5, bank.in_dim, generator=g, dtype=f64)
            first = torch.addmm(
                twin.cell.units.get_buffer("sweep_b1_gate"),
                v,
                twin.cell.units.get_buffer("sweep_w1_gate"),
            )
            mh = m_units * bank.hidden
            torch.testing.assert_close(
                first[:, :mh],
                torch.addmm(bank.b1.reshape(mh), v, bank._w1_flat()),
                rtol=1e-11,
                atol=1e-12,
            )
            torch.testing.assert_close(first[:, mh:], cell.gate(v), rtol=1e-11, atol=1e-12)
            for p_ in model.parameters():  # an optimiser step's worth of change, biases included
                p_.add_(0.05 * torch.randn(p_.shape, generator=g, dtype=p_.dtype))
        assert round_ in (0, 1)
