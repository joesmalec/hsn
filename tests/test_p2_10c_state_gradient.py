"""P2.10c: the split window with the state gradient (``docs/p2_10c_state_gradient_window.md``,
:mod:`hsn.train.state_gradient`): without the ``state`` cut the recurrence's reverse sweep also
carries the cell's read of the state, and the window's gradients are :meth:`HSN.forward`'s
(float64 on the CPU), without and with amendment A8's fast cells."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hsn.config import RunConfig, config_hash, load_config, split_window_scope_error
from hsn.device import seed_everything
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from hsn.train import state_gradient
from hsn.train.compile import fixed_fast_slots
from hsn.train.split import SplitWindow, recurrence_backward, scope_error
from hsn.train.state_gradient import (
    CellConstants,
    activation_grad,
    cell_constants,
    cell_state_vjp,
    state_gradient_backward,
)
from p1_11_support import restore_determinism  # noqa: F401
from test_a6_feedback_detach import A5, ALL, ROUTED, _cfg, _model, _start, _window
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"

#: the state gradient: every feedback path cut but the cell's read of the state
CUT = "train.feedback_detach=[summary,eps,router]"
BASE = (*A5, CUT, "model.cand_norm=rms")
#: two fast cells and two routed slots on the six-cell model (tests/test_a8_fast_cells.py)
FAST = ("model.k_select=4", "model.fast_cells=2")
READ = (*FAST, "model.fast_readout=true")
PINNED = (*READ, "model.fast_units=fixed")


# --------------------------------------------------------------------------------------------------
# The pieces: the activation's derivative, the cell's backward into the state, the sweep
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["gelu", "relu", "silu", "tanh"])
def test_activation_grad_is_the_modules_derivative(name: str) -> None:
    from hsn.model.encoder import make_activation

    g = torch.Generator().manual_seed(3)
    x = (3.0 * torch.randn(257, generator=g, dtype=torch.float64)).requires_grad_(True)
    (want,) = torch.autograd.grad(make_activation(name)(x).sum(), x)
    torch.testing.assert_close(activation_grad(name, x.detach()), want, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match="no derivative"):
        activation_grad("swish", x.detach())


def _one_position(cfg: RunConfig, seed: int = 4) -> tuple:
    """A model, one position's inputs and its executed selection (hard, as a window's sweep
    takes it), with a state that carries gradient."""
    model = _model(cfg)
    B, d = cfg.train.batch, cfg.model.d
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(B, d, generator=g, dtype=torch.float64)
    eps = torch.randn(B, d, generator=g, dtype=torch.float64)
    start = _start(model, B)
    h = start.h.clone().requires_grad_(True)
    state = CellGraphState._make(h=h, q=start.q, age=start.age, pos=start.pos)
    logits = torch.randn(B, cfg.model.n_cells, cfg.model.n_units, generator=g)
    fast = {}
    if cfg.model.fast_cells:
        fast = {"reserved": cfg.model.fast_cells}
        if cfg.model.fast_units == "fixed":
            fast["reserved_units"] = True
    sel = model.router.select(
        logits,
        budget=cfg.model.budget,
        k_max=cfg.model.k_select,
        k_min=cfg.model.k_min,
        mode="topk",
        gate="ste",
        **fast,
    )
    return model, u, eps, state, sel


@pytest.mark.parametrize(
    "extra",
    [
        (),
        ("model.cand_norm=none",),
        ("model.cell_norm=none",),
        ("model.cell_norm=none", "model.cand_norm=none"),
        ("model.ablate_shared_u=true",),
        ("model.activation=relu",),
        ("model.activation=silu",),
        ("model.activation=tanh",),
        ("model.edge_norm=sigmoid",),
        ("model.random_edges=2",),
        ("model.budget=0.1",),
        PINNED,
    ],
)
def test_cell_state_vjp_equals_autograd_of_the_sparse_round(extra: tuple[str, ...]) -> None:
    """The hand-written backward of one position's cells into the state they read (their own
    cell and the sources of their messages) against autograd of
    :meth:`PDGCell.step_sparse`'s candidate and gate, from the values the replay records."""
    cfg = _cfg(*BASE, *extra)
    model, u, eps, state, sel = _one_position(cfg)
    cell = model.cell
    assert not cell.detach_state_read
    B, N, d = state.h.shape
    K = sel.k
    g = torch.Generator().manual_seed(8)
    w_c = torch.randn(B, K, d, generator=g, dtype=torch.float64)
    w_b = torch.randn(B, K, d, generator=g, dtype=torch.float64)

    _, info = cell.step_sparse(u, eps, state, sel)
    c, b, valid = info["c"], info["b"], info["valid"]
    assert isinstance(c, Tensor) and isinstance(b, Tensor) and isinstance(valid, Tensor)
    if "model.budget=0.1" in extra:
        assert not bool(valid.all()) and bool(valid.any())  # padded slots
    ok = valid[..., None]
    (want,) = torch.autograd.grad((torch.where(ok, c * w_c + b * w_b, 0.0)).sum(), state.h)

    record: dict[str, Tensor] = {}
    with torch.no_grad(), cell.window():
        c_r, b_r = cell.replay_sparse(u, eps, state.h.detach(), sel, record=record)
        consts = cell_constants(cell, torch.float64)
    torch.testing.assert_close(c_r, c, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(b_r, b, rtol=1e-12, atol=1e-14)
    safe = sel.cell_idx.masked_fill(~valid, 0)
    units = sel.unit_idx.masked_fill(~valid, 0)
    h_old = state.h.detach().gather(1, safe[:, :, None].expand(B, K, d))
    dh, dvm = cell_state_vjp(
        torch.where(ok, w_c, 0.0),
        torch.where(ok, w_b, 0.0),
        c=c_r,
        b=b_r,
        h_old=h_old,
        units=units,
        y=record["y"],
        alpha=record["alpha"],
        hidden=record["hidden"],
        gate_pre=record["gate_pre"],
        m=record["m"],
        rms=record.get("rms"),
        consts=consts,
    )
    dh, dvm = torch.where(ok, dh, 0.0), torch.where(ok, dvm, 0.0)
    D = consts.pad_src.shape[1]
    got = torch.zeros(B, N, d, dtype=torch.float64)
    got = got.scatter_add(1, safe[:, :, None].expand(B, K, d), dh)
    source = consts.pad_src[safe].reshape(B, K * D, 1).expand(B, K * D, d)
    weighted = consts.edge_weights[safe][..., None] * dvm[:, :, None, :]
    got = got.scatter_add(1, source, weighted.reshape(B, K * D, d))
    torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-13)
    assert float(want.abs().max()) > 1e-3  # a live path, not zeros


def test_the_replays_record_leaves_its_values_alone() -> None:
    """``record`` only keeps what the replay computed: the candidate and the gate are the
    same bits with and without it, and the recorded values are the pass's own."""
    cfg = _cfg(*BASE)
    model, u, eps, state, sel = _one_position(cfg)
    cell = model.cell
    h = state.h.detach()
    record: dict[str, Tensor] = {}
    with torch.no_grad(), cell.window():
        plain = cell.replay_sparse(u, eps, h, sel)
        kept = cell.replay_sparse(u, eps, h, sel, record=record)
    assert torch.equal(plain[0], kept[0]) and torch.equal(plain[1], kept[1])
    assert set(record) == set(state_gradient.RECORD_KEYS)
    B, K, d = kept[0].shape
    assert record["hidden"].shape == (B, K, cfg.model.hidden)
    assert record["alpha"].shape == (B, K) and record["rms"].shape == (B, K)
    torch.testing.assert_close(torch.sigmoid(record["gate_pre"]), kept[1], rtol=0, atol=0)
    # the hidden pre-activation of every pair's own unit, in the pairs' order
    bank = cell.units
    v = torch.randn(11, bank.in_dim, dtype=torch.float64)
    index = torch.randint(0, bank.n_units, (11,))
    seen: dict[str, Tensor] = {}
    with torch.no_grad():
        out = bank.forward_grouped(v, index, record=seen)
        want = torch.einsum("pi,pih->ph", v, bank.W1[index]) + bank.b1[index]
        assert torch.equal(out, bank.forward_grouped(v, index))
    torch.testing.assert_close(seen["hidden"], want, rtol=1e-12, atol=1e-13)
    assert not any(value.requires_grad for value in (*record.values(), *seen.values()))


def _silent(consts: CellConstants) -> CellConstants:
    """The constants of a cell whose candidate and gate do not read the state."""
    return consts._replace(
        w_out=torch.zeros_like(consts.w_out),
        w1_state=torch.zeros_like(consts.w1_state),
        gate_state=torch.zeros_like(consts.gate_state),
        norm_gain_state=None,
    )


def test_sweep_without_the_state_read_is_the_split_windows_recurrence() -> None:
    """With a cell that does not read the state the sweep is P2.10's, value for value: the
    carry, the summary, the resets and the padded slots are the same code path."""
    g = torch.Generator().manual_seed(7)
    B, T, N, K, d, H, M, D = 3, 9, 5, 3, 4, 6, 2, 2
    f64 = torch.float64
    c = torch.rand(B, T, K, d, generator=g, dtype=f64)
    b = torch.rand(B, T, K, d, generator=g, dtype=f64)
    h_old = torch.rand(B, T, K, d, generator=g, dtype=f64)
    cells = torch.stack(
        [torch.stack([torch.randperm(N, generator=g)[:K] for _ in range(T)]) for _ in range(B)]
    )
    valid = torch.rand(B, T, K, generator=g) > 0.3
    cells = torch.where(valid, cells, torch.zeros_like(cells))
    units = torch.randint(0, M, (B, T, K), generator=g)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[0, 0] = reset[1, 4] = reset[2, 7] = reset[2, 8] = True
    w_pre = torch.randn(B, T, d, generator=g, dtype=f64)
    w_post = torch.randn(B, T, d, generator=g, dtype=f64)
    w_sel = torch.where(valid[..., None], torch.randn(B, T, K, d, generator=g, dtype=f64), 0.0)
    record = {
        "y": torch.randn(B, T, K, d, generator=g, dtype=f64),
        "alpha": valid.to(f64),
        "hidden": torch.randn(B, T, K, H, generator=g, dtype=f64),
        "gate_pre": torch.randn(B, T, K, d, generator=g, dtype=f64),
        "m": torch.randn(B, T, K, d, generator=g, dtype=f64),
    }
    consts = _silent(
        CellConstants(
            w_out=torch.randn(d, d, dtype=f64),
            cand_gain=torch.ones(d, dtype=f64),
            cand_eps=1e-6,
            w2=torch.randn(M, H, d, dtype=f64),
            w1_state=torch.randn(M, 2 * d, H, dtype=f64),
            b1=torch.randn(M, H, dtype=f64),
            gate_state=torch.randn(d, 2 * d, dtype=f64),
            gate_bias=torch.randn(d, dtype=f64),
            norm_gain_state=None,
            in_dim=4 * d,
            v=torch.randn(d, d, dtype=f64),
            edge_weights=torch.rand(N, D, dtype=f64),
            pad_src=torch.randint(0, N, (N, D), generator=g),
            activation="gelu",
        )
    )
    want = recurrence_backward(c, b, h_old, cells, valid, reset, N, w_pre, w_post, w_sel)
    got = state_gradient_backward(
        c, b, h_old, cells, units, valid, reset, N, w_pre, w_post, w_sel, None, record, consts
    )
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    # absent cotangents: the same None handling
    want = recurrence_backward(c, b, h_old, cells, valid, reset, N, None, w_post, None)
    got = state_gradient_backward(
        c, b, h_old, cells, units, valid, reset, N, None, w_post, None, None, record, consts
    )
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


# --------------------------------------------------------------------------------------------------
# End to end: the window's gradients are HSN.forward's
# --------------------------------------------------------------------------------------------------


def _objective(logits: Tensor, info: dict[str, Tensor], mask: Tensor) -> Tensor:
    """Every gradient-carrying output of a window (tests/test_p2_10_split_window.py)."""
    task = logits[mask].square().sum() if bool(mask.any()) else logits.sum() * 0.0
    pred = (info["z_hat"] - info["z_target"]).square().mean() + 0.1 * info["z"].square().mean()
    cost = info["expected_cost"].square().sum()
    balance = info["unit_logits"].square().mean()
    return task + 0.7 * pred + 0.3 * cost + 0.2 * balance


def _case(
    cfg: RunConfig, model: HSN | None = None
) -> tuple[HSN, tuple[Tensor, Tensor, CellGraphState, Tensor], Tensor]:
    """A model (the perturbed one of the A6 tests unless given) and one window with interior
    resets and a carried, non-zero start state in one row, and the loss positions."""
    if model is None:
        model = _model(cfg)
    tokens, ctx, reset = _window(cfg)
    reset[0, 0] = False  # row 0 carries its start state into the window
    reset[1, 5] = True
    reset[2, 9] = True
    reset[2, 10] = True
    state = _start(model, tokens.shape[0])
    mask = torch.zeros_like(reset)
    mask[:, 3::4] = True
    return model, (tokens, ctx, state, reset), mask


def _both(
    cfg: RunConfig,
    *,
    loss_positions: bool = True,
    train_step: int = 0,
    model: HSN | None = None,
) -> tuple[dict, dict, dict, dict]:
    """(reference grads, window grads, reference info, window info) at ``train_step``."""
    model, (tokens, ctx, state, reset), mask = _case(cfg, model)
    model.set_train_step(train_step)
    if not loss_positions:
        mask = torch.zeros_like(mask)
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]

    logits, ref_state, ref_info = model(tokens, ctx, state, reset)
    ref = torch.autograd.grad(_objective(logits, ref_info, mask), params, allow_unused=True)

    window = SplitWindow(model, cfg, None, state_gradient=True)
    s_logits, s_state, s_info = window(tokens, ctx, state, reset)
    got = torch.autograd.grad(_objective(s_logits, s_info, mask), params, allow_unused=True)
    assert window.calls == 1

    torch.testing.assert_close(s_logits, logits, rtol=1e-10, atol=1e-12)
    for field in ("h", "q"):
        torch.testing.assert_close(
            getattr(s_state, field), getattr(ref_state, field), rtol=1e-10, atol=1e-12
        )
    assert torch.equal(s_state.age, ref_state.age) and torch.equal(s_state.pos, ref_state.pos)
    for key in ("cell_idx", "unit_idx", "valid"):
        assert torch.equal(s_info[key], ref_info[key])
    assert set(s_info) == set(ref_info)
    return dict(zip(names, ref, strict=True)), dict(zip(names, got, strict=True)), ref_info, s_info


def _assert_same_grads(ref: dict, got: dict, *, live: int = 20, scaled: bool = False) -> None:
    """Every parameter gradient equal at rtol 1e-9 and the same set of absent ones. ``scaled``
    (a window without loss positions, as in the P2.10 test): entries near zero by cancellation
    are held to 1e-8 of the tensor's largest entry."""
    assert {n for n, g in ref.items() if g is None} == {n for n, g in got.items() if g is None}
    count = 0
    for name, want in ref.items():
        if want is None:
            continue
        atol = max(1e-12, 1e-8 * float(want.abs().max())) if scaled else 1e-12
        torch.testing.assert_close(
            got[name], want, rtol=1e-9, atol=atol, msg=lambda m, n=name: n + m
        )
        count += int(bool(want.abs().max() > 0))
    assert count >= live  # the paths are live: this compares real gradients, not zeros


@pytest.mark.parametrize(
    "extra",
    [
        (),
        ("model.cand_norm=none",),
        ("model.cell_norm=none",),
        ("model.lambda_cost=0.05",),
        ("train.unit_credit=executed",),
        ("model.ablate_shared_u=true",),
        ("model.activation=silu",),
        ("model.edge_norm=sigmoid", "model.random_edges=2"),
        ("model.address_ngram=2",),
    ],
)
def test_window_gradients_equal_autograd(extra: tuple[str, ...]) -> None:
    cfg = _cfg(*BASE, *extra)
    ref, got, ref_info, s_info = _both(cfg)
    _assert_same_grads(ref, got)
    for key in ("cost", "explored", "refreshed", "dense_fallback", "state_abs_max"):
        torch.testing.assert_close(s_info[key], ref_info[key], rtol=1e-10, atol=1e-12)
    for key in ("eps_abs_mean", "alpha_entropy", "expected_cost", "route_log_prob"):
        torch.testing.assert_close(s_info[key], ref_info[key], rtol=1e-9, atol=1e-12)


def test_window_gradients_with_refresh_probes() -> None:
    cfg = _cfg(*BASE, "model.age_max=2")
    ref, got, ref_info, s_info = _both(cfg)
    assert float(ref_info["refreshed"].sum()) > 0
    _assert_same_grads(ref, got)
    torch.testing.assert_close(s_info["refreshed"], ref_info["refreshed"])


def test_window_gradients_with_padded_slots() -> None:
    cfg = _cfg(*BASE, "model.budget=0.1")
    ref, got, ref_info, _ = _both(cfg)
    assert not bool(ref_info["valid"].all()) and bool(ref_info["valid"].any())
    _assert_same_grads(ref, got)


def test_window_gradients_without_selected_readout() -> None:
    cfg = _cfg(*ROUTED, CUT)
    ref, got, _, _ = _both(cfg)
    _assert_same_grads(ref, got, live=15)


def test_window_without_loss_positions_gives_the_same_none_set() -> None:
    cfg = _cfg(*BASE)
    ref, got, _, _ = _both(cfg, loss_positions=False)
    _assert_same_grads(ref, got, live=10, scaled=True)


def test_window_after_a_soft_warm_up() -> None:
    """``train.soft_until_step``: the warm-up's windows go to the fallback (the model's own
    forward, bit for bit), the hard-gate steps to the window."""
    cfg = _cfg(*BASE, "train.soft_until_step=3")
    ref, got, _, _ = _both(cfg, train_step=3)
    _assert_same_grads(ref, got)
    model, (tokens, ctx, state, reset), _ = _case(cfg)
    window = SplitWindow(model, cfg, None, state_gradient=True)
    model.set_train_step(2)
    assert not window.splits_now()
    logits, _, _ = window(tokens, ctx, state, reset)
    want, _, _ = model(tokens, ctx, state, reset)
    assert torch.equal(logits, want) and window.calls == 0


def test_a_window_with_dense_fallback_rows_runs_the_autograd_window() -> None:
    """``train.p_dense``: the sweep has no dense recurrence, so a window that drew dense rows
    is the model's own forward from the same draw; a window that drew none is swept."""
    cfg = _cfg(*BASE, "train.p_dense=0.2")
    model, (tokens, ctx, state, reset), _ = _case(cfg)
    window = SplitWindow(model, cfg, None, state_gradient=True)
    swept = fallen = 0
    for seed in range(8):
        torch.manual_seed(seed)
        want, _, info = model(tokens, ctx, state, reset)
        calls = window.calls
        torch.manual_seed(seed)
        logits, _, s_info = window(tokens, ctx, state, reset)
        torch.testing.assert_close(s_info["dense_fallback"], info["dense_fallback"])
        if float(info["dense_fallback"].sum()) > 0:
            assert torch.equal(logits, want) and window.calls == calls
            fallen += 1
        else:
            torch.testing.assert_close(logits, want, rtol=1e-10, atol=1e-12)
            assert window.calls == calls + 1
            swept += 1
    assert fallen >= 1 and swept >= 1


# --------------------------------------------------------------------------------------------------
# Amendment A8: the fast cells
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        FAST,
        READ,
        PINNED,
        (*PINNED, "train.unit_credit=executed"),
        (*PINNED, "model.age_max=2"),
        (*PINNED, "model.address_ngram=2"),
    ],
)
def test_window_gradients_with_fast_cells(extra: tuple[str, ...]) -> None:
    """The reserved slots (a constant mask; with ``fast_units=fixed`` a constant unit), the
    fast read of ``S`` (a constant under the summary cut) and of the decoder (the fast cells'
    states after the position, with gradient), and the read-out over the routed slots only."""
    cfg = _cfg(*BASE, *extra)
    ref, got, ref_info, _ = _both(cfg)
    _assert_same_grads(ref, got)
    F = cfg.model.fast_cells
    B, T = cfg.train.batch, cfg.train.unroll
    assert torch.equal(ref_info["cell_idx"][:, :, :F], torch.arange(F).expand(B, T, F))
    if cfg.model.fast_readout:  # the new columns of S and of the decoder carry gradient
        d = cfg.model.d
        for name in ("shared.fc1.weight", "decoder.fc1.weight"):
            assert float(ref[name][:, -d:].abs().sum()) > 0.0
            torch.testing.assert_close(got[name][:, -d:], ref[name][:, -d:], rtol=1e-9, atol=1e-12)


def test_fast_cells_the_budget_cuts_keep_their_state_in_the_fast_read() -> None:
    """A budget that holds fewer slots than the fast group leaves a fast cell unwritten at a
    position: the decoder's fast read then reads its carried state, not a slot's."""
    cfg = _cfg(*BASE, *PINNED, "model.budget=0.1")
    ref, got, ref_info, _ = _both(cfg)
    F = cfg.model.fast_cells
    assert not bool(ref_info["valid"][:, :, :F].all()) and bool(ref_info["valid"].any())
    _assert_same_grads(ref, got)


# --------------------------------------------------------------------------------------------------
# The state's cotangent itself, and that the comparison above can fail
# --------------------------------------------------------------------------------------------------


def _autograd_by_position(
    model: HSN,
    inputs: tuple[Tensor, Tensor, CellGraphState, Tensor],
    mask: Tensor,
    *,
    static: bool,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """The model's own window with autograd, every position reading a copy of its entry state:
    ``(logits, dL/dh, dL/dq)`` with the gradients of the copies, which are what each position
    alone passes back (the model resets inside the position, so a row that resets there passes
    nothing). ``static`` runs the capturable round (:meth:`PDGCell.step_sparse` with
    ``static=True``: the round of the compiled window and of the split window's sweep) instead
    of the eager one: the same function, with the unit bank's products in another order."""
    tokens, ctx, state, reset = inputs
    entries: list[CellGraphState] = []
    position = model._position

    def recording(entry: CellGraphState, *args: object, **kwargs: object) -> object:
        copy = CellGraphState._make(
            h=entry.h.clone(), q=entry.q.clone(), age=entry.age, pos=entry.pos
        )
        entries.append(copy)
        if static:  # the sweep's constants: its round, product for product
            consts = model.cell.window_consts(fast_slots=fixed_fast_slots(model))
            kwargs = {**kwargs, "static": True, "consts": consts}
        return position(copy, *args, **kwargs)  # type: ignore[arg-type]

    start = CellGraphState._make(
        h=state.h.clone().requires_grad_(True),
        q=state.q.clone().requires_grad_(True),
        age=state.age,
        pos=state.pos,
    )
    model._position = recording  # type: ignore[method-assign]
    try:
        logits, _, info = model(tokens, ctx, start, reset)
    finally:
        del model._position
    grads = torch.autograd.grad(
        _objective(logits, info, mask), [x for entry in entries for x in (entry.h, entry.q)]
    )
    return logits.detach(), list(grads[0::2]), list(grads[1::2])


@pytest.mark.parametrize("static", [False, True], ids=["eager_round", "sweeps_round"])
@pytest.mark.parametrize("extra", [(), PINNED], ids=["routed", "fast_cells"])
def test_sweeps_state_cotangent_is_autograds(
    extra: tuple[str, ...], static: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep's ``A`` and ``R`` (its ``trace``) are the gradient of the objective with
    respect to the state each position reads, at every position; and a single step of the
    sweep, started from autograd's cotangent of the state after a position, gives autograd's
    cotangent of the state before it. The single step accumulates nothing, so it stays a sharp
    test where rounding is amplified along a window (a trained model: see the write-up).

    Against the model's capturable round the window's forward sweep is the same arithmetic
    position by position, so the two passes follow one trajectory (the logits then differ only
    by the replay's batched ``S`` and decoder)."""
    _cotangents_by_position(_cfg(*BASE, *extra), static, monkeypatch)


def _cotangents_by_position(
    cfg: RunConfig, static: bool, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Tensor], Tensor]:
    """The comparison of :func:`test_sweeps_state_cotangent_is_autograds` for ``cfg``; returns
    the sweep's trace and the window's resets."""
    model, inputs, mask = _case(cfg)
    tokens, ctx, state, reset = inputs
    logits, d_h, d_q = _autograd_by_position(model, inputs, mask, static=static)
    B, T, N, d = cfg.train.batch, cfg.train.unroll, cfg.model.n_cells, cfg.model.d

    trace: dict[str, Tensor] = {}
    given: list[tuple] = []
    sweep = state_gradient.state_gradient_backward

    def traced(*args: object, **kwargs: object) -> tuple[Tensor, Tensor]:
        given.append(args)
        limits.append(kwargs.get("limits"))
        return sweep(*args, **kwargs, trace=trace)  # type: ignore[arg-type]

    limits: list[object] = []
    monkeypatch.setattr(state_gradient, "state_gradient_backward", traced)
    window = SplitWindow(model, cfg, None, state_gradient=True)
    s_logits, _, s_info = window(tokens, ctx, state, reset)
    _objective(s_logits, s_info, mask).backward()
    monkeypatch.undo()
    if static:
        torch.testing.assert_close(s_logits.detach(), logits, rtol=1e-12, atol=1e-13)
    assert trace["A"].shape == (B, T, N, d) and len(given) == 1
    tol = {"rtol": 1e-9, "atol": 1e-12}
    largest = 0.0
    for t in range(T):
        keep = ~reset[:, t]
        got_h = torch.where(keep[:, None, None], trace["A"][:, t], 0.0)
        got_q = torch.where(keep[:, None], trace["R"][:, t], 0.0)
        torch.testing.assert_close(got_h, d_h[t], **tol, msg=lambda m, t=t: f"h {t}{m}")
        torch.testing.assert_close(got_q, d_q[t], **tol, msg=lambda m, t=t: f"q {t}{m}")
        largest = max(largest, float(d_h[t].abs().max()))
    assert largest > 1e-3
    assert float(trace["dh"].abs().max()) > 0 and float(trace["dvm"].abs().max()) > 0

    # one step at a time, from autograd's cotangent of the state after the position
    c, b, h_old, cells, units, valid, rst, n_cells, *cots, record, consts = given[0]
    for t in range(T):
        at = slice(t, t + 1)
        after = (d_h[t + 1], d_q[t + 1]) if t + 1 < T else (torch.zeros(B, N, d), torch.zeros(B, d))
        step: dict[str, Tensor] = {}
        sweep(
            c[:, at],
            b[:, at],
            h_old[:, at],
            cells[:, at],
            units[:, at],
            valid[:, at],
            rst[:, at],
            n_cells,
            *(None if x is None else x[:, at] for x in cots),
            {key: value[:, at] for key, value in record.items()},
            consts,
            trace=step,
            initial=(after[0].to(torch.float64), after[1].to(torch.float64)),
            limits=limits[0],  # type: ignore[arg-type]
        )
        keep = ~reset[:, t]
        got_h = torch.where(keep[:, None, None], step["A"][:, 0], 0.0)
        got_q = torch.where(keep[:, None], step["R"][:, 0], 0.0)
        torch.testing.assert_close(got_h, d_h[t], **tol, msg=lambda m, t=t: f"step h {t}{m}")
        torch.testing.assert_close(got_q, d_q[t], **tol, msg=lambda m, t=t: f"step q {t}{m}")
    return trace, reset


def test_without_the_cells_backward_the_gradients_are_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The comparison is sensitive to what this work adds: with the cell's read of the state
    left out of the sweep (P2.10's recurrence on a model without the ``state`` cut) the cell's
    gradients are far from autograd's."""
    cfg = _cfg(*BASE)

    def nothing(dc: Tensor, db: Tensor, **_: object) -> tuple[Tensor, Tensor]:
        return torch.zeros_like(dc), torch.zeros_like(dc)

    monkeypatch.setattr(state_gradient, "cell_state_vjp", nothing)
    ref, got, _, _ = _both(cfg)
    apart = {
        name: float((got[name] - want).abs().max() / want.abs().max())
        for name, want in ref.items()
        if want is not None and name.startswith("cell.")
    }
    assert max(apart.values()) > 1e-2, apart
    assert sum(gap > 1e-4 for gap in apart.values()) >= 6, apart


# --------------------------------------------------------------------------------------------------
# Amendment A6's per-row limit on what a position passes back (train.state_grad_clip / _ratio)
# --------------------------------------------------------------------------------------------------

#: sixteen rows: with fewer than ten live rows the batch's typical row is its largest and a
#: ratio never acts (hsn.model.gradclip.typical_row). (Not ten to twelve: in this window's
#: rows 9 to 11 the scorer's own gradient is ill-conditioned, and the window and the model
#: then agree on it to 1e-9 only, with or without a limit; every other parameter to 1e-14.)
ROWS = ("train.batch=16", "train.micro_batch=16")
#: limits that act on this model's window: a ratio just over 1, an absolute limit under the
#: rows' norms, and both (the smaller applies)
LIMITS = {
    "ratio": ("train.state_grad_ratio=1.02",),
    "clip": ("train.state_grad_clip=0.02",),
    "ratio_and_clip": ("train.state_grad_ratio=1.5", "train.state_grad_clip=0.05"),
}


@pytest.mark.parametrize("extra", [(), PINNED], ids=["routed", "fast_cells"])
@pytest.mark.parametrize("name", list(LIMITS))
def test_window_gradients_with_the_row_limit(name: str, extra: tuple[str, ...]) -> None:
    """The window with the limit gives the autograd window's gradients with the limit, and
    they are not the gradients without it."""
    cfg = _cfg(*BASE, *ROWS, *extra, *LIMITS[name])
    ref, got, _, _ = _both(cfg)
    _assert_same_grads(ref, got)
    free, _, _, _ = _both(_cfg(*BASE, *ROWS, *extra))
    apart = {
        n: float((ref[n] - g).abs().max() / g.abs().max())
        for n, g in free.items()
        if g is not None and float(g.abs().max()) > 0
    }
    assert max(apart.values()) > 1e-3, apart


@pytest.mark.parametrize("static", [False, True], ids=["eager_round", "sweeps_round"])
@pytest.mark.parametrize("extra", [(), PINNED], ids=["routed", "fast_cells"])
@pytest.mark.parametrize("name", list(LIMITS))
def test_sweeps_limited_state_cotangent_is_autograds(
    name: str, extra: tuple[str, ...], static: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the limit the sweep's ``A`` and ``R`` are what each position passes back in the
    autograd window, at every position and one step at a time. The limit acts on rows of both
    (``limited_h``, ``limited_q``) and, with a ratio, at positions where a row resets: such a
    row is limited and counted in the batch's typical row before it is zeroed, as in the model
    (reset, then limit, then the position's reads); counted otherwise, the typical row and
    every limited row of that position would differ."""
    cfg = _cfg(*BASE, *ROWS, *extra, *LIMITS[name])
    trace, reset = _cotangents_by_position(cfg, static, monkeypatch)
    limited_h, limited_q = trace["limited_h"], trace["limited_q"]
    assert limited_h.shape == reset.shape and limited_h.dtype == torch.bool
    assert int(limited_h.sum()) >= 5 and int(limited_q.sum()) >= 5
    assert not bool(limited_h.all()) and not bool(limited_q.all())
    interior = reset[:, 1:].any(dim=0)  # positions after the first where some row resets
    assert int(interior.sum()) == 3
    acted = (limited_h | limited_q)[:, 1:] & ~reset[:, 1:]
    assert bool(acted[:, interior].any(dim=0).all())


def test_the_row_limit_is_the_models() -> None:
    """The window hands the sweep the model's limits: the absolute limit of ``dL/dq``, that of
    ``dL/dh`` (``limit / sqrt(n_cells)``) and the ratio; none without the keys, and none under
    the full cut (whose window the scope refuses with them)."""
    cfg = _cfg(*BASE, "train.state_grad_clip=0.5", "train.state_grad_ratio=3.0")
    limits = SplitWindow(HSN.from_config(cfg), cfg, None).row_limits
    assert limits is not None
    assert limits.q == 0.5 and limits.ratio == 3.0
    assert limits.h == pytest.approx(0.5 / cfg.model.n_cells**0.5, rel=1e-15)
    ratio = _cfg(*BASE, "train.state_grad_ratio=3.0")
    only = SplitWindow(HSN.from_config(ratio), ratio, None).row_limits
    assert only is not None and (only.h, only.q, only.ratio) == (None, None, 3.0)
    plain = _cfg(*BASE)
    assert SplitWindow(HSN.from_config(plain), plain, None).row_limits is None
    with pytest.raises(ValueError, match="state_grad_clip"):
        full = _cfg(*A5, ALL, "train.state_grad_ratio=3.0")
        SplitWindow(HSN.from_config(full), full, None)


# --------------------------------------------------------------------------------------------------
# The candidate base's shape
# --------------------------------------------------------------------------------------------------

#: the candidate base of 13 Nov 2026 on the standard recipe: twelve slots, 64 cells, the state
#: gradient, executed credit, no soft warm-up (float64, a short window: the widths are the
#: recipe's, the batch and the window are not)
CANDIDATE = (
    "profile=cpu",
    CUT,
    "model.k_select=12",
    "model.n_cells=64",
    "train.unit_credit=executed",
    "train.soft_until_step=0",
    "train.precision=fp64",
    "train.batch=3",
    "train.micro_batch=3",
    "train.unroll=12",
)
#: the same with amendment A8's fast group: eight fast cells at fixed units, their own read
CANDIDATE_FAST = (
    "model.k_select=20",
    "model.fast_cells=8",
    "model.fast_units=fixed",
    "model.fast_readout=true",
)


@pytest.mark.parametrize("extra", [(), CANDIDATE_FAST], ids=["routed", "fast_cells"])
def test_window_gradients_at_the_recipes_shape(extra: tuple[str, ...]) -> None:
    """``configs/experiments/hsn_recipe_a6.yaml`` with the candidate base's overrides: the
    recipe's widths (d 64, H 512, eight units, 64 cells, twelve slots or eight fast and twelve
    routed), every parameter gradient equal to autograd's."""
    cfg = load_config(RECIPE_A6, [*CANDIDATE, *extra])
    assert cfg.model.d == 64 and cfg.model.hidden == 512 and cfg.model.n_units == 8
    assert scope_error(cfg, state_gradient=True) is None
    # the recipe's own initialisation, moved a little off it: at these widths the A6 tests'
    # perturbation of 0.3 gives gradients of 1e8, whose small entries are float64 noise
    seed_everything(0)
    model = HSN.from_config(cfg).to(torch.float64)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ref, got, ref_info, _ = _both(cfg, model=model.train())
    _assert_same_grads(ref, got)
    assert ref_info["cell_idx"].shape[-1] == cfg.model.k_select


# --------------------------------------------------------------------------------------------------
# Scope and configuration
# --------------------------------------------------------------------------------------------------


def test_the_scope_with_the_state_gradient() -> None:
    assert scope_error(_cfg(*BASE), state_gradient=True) is None
    assert scope_error(_cfg(*BASE, *PINNED), state_gradient=True) is None
    assert scope_error(_cfg(*BASE, "model.address_ngram=2"), state_gradient=True) is None
    assert scope_error(_cfg(*BASE, "train.p_dense=0.25"), state_gradient=True) is None
    # A6's per-row limit: the state-gradient sweep applies it, the full cut's recurrence not
    for limit in ("train.state_grad_clip=1.0", "train.state_grad_ratio=3.0"):
        assert scope_error(_cfg(*BASE, limit), state_gradient=True) is None
        reason = scope_error(_cfg(*A5, ALL, limit))
        assert reason is not None and "state_grad_clip" in reason
    for overrides, message in (
        ((*A5, ALL), "holds every path"),
        ((*A5, "train.feedback_detach=[summary,eps]"), "lacks"),
        ((*A5, "train.feedback_detach=[eps,router]"), "lacks"),
        ((*A5,), "lacks"),
        ((*BASE, "train.epsilon_start=0.1", "train.epsilon_end=0.1"), "epsilon"),
        ((*BASE, "model.cell_units=fixed"), "cell_units"),
        (("model.routing=topk", "train.gate_mode=soft", CUT), "ste"),
    ):
        reason = scope_error(_cfg(*overrides), state_gradient=True)
        assert reason is not None and message in reason, overrides


def test_the_window_follows_the_configurations_cut() -> None:
    """The configuration decides which window is built: without the ``state`` cut the one
    with the state gradient, with it P2.10's. ``state_gradient`` only asserts the caller's
    expectation, and a model built with another cut than the configuration's is refused."""
    cut = _cfg(*BASE)
    assert SplitWindow(HSN.from_config(cut), cut, None).state_gradient is True
    assert SplitWindow(HSN.from_config(cut), cut, None, state_gradient=True).state_gradient
    with pytest.raises(ValueError, match="lacks \\['state'\\]"):
        SplitWindow(HSN.from_config(cut), cut, None, state_gradient=False)
    full = _cfg(*A5, ALL, "model.cand_norm=rms")
    assert SplitWindow(HSN.from_config(full), full, None).state_gradient is False
    with pytest.raises(ValueError, match="holds every path"):
        SplitWindow(HSN.from_config(full), full, None, state_gradient=True)
    with pytest.raises(ValueError, match="the model does not"):
        SplitWindow(HSN.from_config(full), cut, None)


@pytest.mark.parametrize("window", ["split", "fused"])
def test_the_configuration_key_selects_it(window: str) -> None:
    """``hardware.window`` = ``split`` / ``fused`` accepts the state gradient, and with it the
    fast cells; the full cut with fast cells stays refused (its recurrence has no fast read);
    A9's address n-grams are accepted under both cuts. The key is opt-in as before: a
    configuration without it keeps its hash."""
    base = _cfg(*BASE)
    keyed = _cfg(*BASE, f"hardware.window={window}")
    assert keyed.hardware.window == window and config_hash(keyed) != config_hash(base)
    assert _cfg(*BASE, *PINNED, f"hardware.window={window}").model.fast_cells == 2
    assert _cfg(*BASE, "model.address_ngram=2", f"hardware.window={window}").hardware.window
    assert _cfg(*A5, ALL, "model.address_ngram=2", f"hardware.window={window}").hardware.window
    with pytest.raises(ValueError, match="fast_cells"):
        _cfg(*A5, ALL, *FAST, f"hardware.window={window}")
    with pytest.raises(ValueError, match="hardware.window"):
        _cfg(*A5, "train.feedback_detach=[summary,eps]", f"hardware.window={window}")
    assert split_window_scope_error(_cfg(*BASE)) is None
    assert split_window_scope_error(_cfg(*A5, ALL)) is None
    recipe = load_config(RECIPE_A6)
    assert split_window_scope_error(recipe) is None
    assert config_hash(recipe) == config_hash(load_config(RECIPE_A6, ["hardware.window=autograd"]))


def test_full_cut_window_with_address_ngrams() -> None:
    """A9's ``model.address_ngram`` under the full cut (P2.10's window, which refused it as
    untested): the scorer's wider scalars pass through the replay as they are, and the
    window's gradients are :meth:`HSN.forward`'s."""
    import test_p2_10_split_window as p210

    cfg = _cfg(*p210.RECIPE, "model.address_ngram=2")
    assert scope_error(cfg) is None and scope_error(cfg, state_gradient=False) is None
    ref, got, ref_info, s_info = p210._both(cfg)
    p210._assert_same_grads(ref, got)
    for key in ("cell_idx", "unit_idx", "valid"):
        assert torch.equal(s_info[key], ref_info[key])
