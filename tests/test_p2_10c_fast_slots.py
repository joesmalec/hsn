"""P2.10c, step 1 of phase 2: the unit products of the slots whose unit is fixed.

Amendment A8's fast cells with ``model.fast_units = fixed`` hold the first ``F`` slots at every
position, slot ``i`` at unit ``i mod M``. The static rounds multiplied every pair with every
unit's weights; for those slots one unit's weights are enough. These tests hold the cheaper
products to the ones they replace: the same values to rounding in the forward sweep (the split
window's static round and the fused sweep's products) and in the reverse sweep's layouts, the
same decisions, and nothing at all for a model without such slots."""

from __future__ import annotations

import pytest
import torch
from hsn.config import RunConfig
from hsn.model.cell import WindowConsts
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from hsn.model.units import UnitBank
from hsn.train import compile as compile_module
from hsn.train import state_gradient
from hsn.train.compile import (
    TAPE_KEYS,
    SweepTwin,
    _tape_block,
    fixed_fast_slots,
    run_tape,
    sweep_merge_error,
)
from hsn.train.split import SplitWindow
from hsn.train.state_gradient import cell_constants, cell_state_vjp, sweep_layouts
from p1_11_support import restore_determinism  # noqa: F401
from test_a6_feedback_detach import _cfg, _model
from test_p2_10c_state_gradient import BASE, FAST, PINNED, READ, _case, _objective, _one_position
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

F64 = torch.float64
TIGHT = {"rtol": 1e-12, "atol": 1e-13}


def _bank(n_units: int = 3, d: int = 4, hidden: int = 7) -> UnitBank:
    torch.manual_seed(3)
    bank = UnitBank(n_units, d, hidden, "gelu", init="fan_in").double()
    with torch.no_grad():
        bank.b1.normal_()
        bank.b2.normal_()
    return bank


def _pairs(bank: UnitBank, B: int, K: int, n_fast: int, seed: int = 5) -> tuple[Tensor, Tensor]:
    """``v [B, K, in]`` and the units of a recorded decision whose first ``n_fast`` slots are
    fixed (slot ``i`` at unit ``i mod M``) and whose other slots are anything."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(B, K, bank.in_dim, generator=g, dtype=F64)
    units = torch.randint(0, bank.n_units, (B, K), generator=g)
    units[:, :n_fast] = torch.arange(n_fast) % bank.n_units
    return v, units


# --------------------------------------------------------------------------------------------------
# The unit bank
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_fast", [1, 2, 3, 5])
def test_slot_layout_holds_each_slots_own_unit(n_fast: int) -> None:
    bank = _bank()
    w1, b1, w2, b2 = bank.slot_layout(n_fast)
    assert w1.shape == (n_fast, bank.in_dim, bank.hidden) and b1.shape == (n_fast, bank.hidden)
    assert w2.shape == (n_fast, bank.hidden, bank.d) and b2.shape == (n_fast, bank.d)
    for slot in range(n_fast):
        unit = slot % bank.n_units
        assert torch.equal(w1[slot], bank.W1[unit]) and torch.equal(b1[slot], bank.b1[unit])
        assert torch.equal(w2[slot], bank.W2[unit]) and torch.equal(b2[slot], bank.b2[unit])
    gate = torch.nn.Linear(bank.in_dim, 5).double()
    w1_gate, b1_gate, _, _ = bank.slot_layout(n_fast, rider=gate)
    assert w1_gate.shape == (n_fast, bank.in_dim, bank.hidden + 5)
    assert torch.equal(w1_gate[:, :, : bank.hidden], w1)
    assert torch.equal(w1_gate[:, :, bank.hidden :], gate.weight.t().expand(n_fast, -1, -1))
    assert torch.equal(b1_gate[:, bank.hidden :], gate.bias.expand(n_fast, -1))
    with pytest.raises(ValueError, match="n_slots"):
        bank.slot_layout(0)


@pytest.mark.parametrize("n_fast", [1, 3, 5])
def test_forward_static_slots_is_the_static_pass(n_fast: int) -> None:
    """Values and gradients (the pairs' inputs and every weight of the bank) of the static pass
    with fixed slots against :meth:`UnitBank.forward_static` and the grouped pass."""
    bank = _bank()
    B, K = 4, n_fast + 2
    v, units = _pairs(bank, B, K, n_fast)
    v.requires_grad_(True)
    weight = torch.randn(B, K, bank.d, generator=torch.Generator().manual_seed(1), dtype=F64)
    params = [v, bank.W1, bank.b1, bank.W2, bank.b2]

    flat, index = v.reshape(B * K, -1), units.reshape(B * K)
    want = bank.forward_static(flat, index).reshape(B, K, -1)
    want_grads = torch.autograd.grad((want * weight).sum(), params)
    got = bank.forward_static_slots(v, units, *bank.slot_layout(n_fast))
    got_grads = torch.autograd.grad((got * weight).sum(), params)
    torch.testing.assert_close(got, want, **TIGHT)
    torch.testing.assert_close(got, bank.forward_selected(flat, index).reshape(B, K, -1), **TIGHT)
    for mine, theirs in zip(got_grads, want_grads, strict=True):
        torch.testing.assert_close(mine, theirs, **TIGHT)
    with pytest.raises(ValueError, match="proper prefix"):
        bank.forward_static_slots(v[:, :n_fast], units[:, :n_fast], *bank.slot_layout(n_fast))


@pytest.mark.parametrize("rider", [False, True])
@pytest.mark.parametrize("n_fast", [2, 5])
def test_forward_static_chosen_slots_is_the_chosen_pass(n_fast: int, rider: bool) -> None:
    """The fused sweep's pass (the chosen unit's second layer only, the gate riding in the
    first layer) with fixed slots against the pass over all slots."""
    bank = _bank()
    B, K = 3, n_fast + 3
    v, units = _pairs(bank, B, K, n_fast)
    gate = torch.nn.Linear(bank.in_dim, 6).double()
    with torch.no_grad():
        w1_flat, w2_flat, b1_flat = bank._w1_flat(), bank._w2_flat(), None
        if rider:
            w1_flat = torch.cat([w1_flat, gate.weight.t()], dim=1)
            b1_flat = torch.cat([bank.b1.reshape(-1), gate.bias])
        want, want_rider = bank.forward_static_chosen(
            v.reshape(B * K, -1),
            units.reshape(B * K),
            w1_flat=w1_flat,
            w2_flat=w2_flat,
            b1_flat=b1_flat,
        )
        got, got_rider = bank.forward_static_chosen_slots(
            v,
            units,
            *bank.slot_layout(n_fast, rider=gate if rider else None),
            w1_flat=w1_flat,
            w2_flat=w2_flat,
            b1_flat=b1_flat,
        )
    torch.testing.assert_close(got, want.reshape(B, K, -1), **TIGHT)
    assert (got_rider is None) == (want_rider is None) == (not rider)
    if got_rider is not None and want_rider is not None:
        torch.testing.assert_close(got_rider, want_rider.reshape(B, K, -1), **TIGHT)
        torch.testing.assert_close(got_rider, gate(v), **TIGHT)


# --------------------------------------------------------------------------------------------------
# The cell's static rounds
# --------------------------------------------------------------------------------------------------


def _fused_consts(model: HSN, n_fast: int, *, merged: bool) -> WindowConsts:
    """The constants a fused sweep hands the cell (inside ``cell.window()``): the all-units
    layouts, merged with the gate or not, and the fixed slots' when ``n_fast``."""
    cell, bank = model.cell, model.cell.units
    consts = cell.window_consts()._replace(w2_flat=bank._w2_flat())
    if merged:
        consts = consts._replace(
            w1_gate=torch.cat([consts.w1_flat, cell.gate.weight.t()], dim=1),
            b1_gate=torch.cat([bank.b1.reshape(-1), cell.gate.bias]),
        )
    if n_fast:
        layout = bank.slot_layout(n_fast, rider=cell.gate if merged else None)
        consts = consts._replace(
            fast_w1=layout[0], fast_b1=layout[1], fast_w2=layout[2], fast_b2=layout[3]
        )
    return consts


def test_static_round_with_fixed_slots_is_the_static_round() -> None:
    """:meth:`PDGCell.step_sparse`'s capturable round: the updated state with the fixed slots'
    products equals the round without them, and so does the gradient into the state."""
    cfg = _cfg(*BASE, *PINNED)
    model, u, eps, state, sel = _one_position(cfg)
    cell = model.cell
    F = fixed_fast_slots(model)
    assert F == cfg.model.fast_cells == 2
    fixed_units = (torch.arange(F) % cfg.model.n_units).expand(cfg.train.batch, F)
    assert torch.equal(sel.unit_idx[:, :F], fixed_units)
    weight = torch.randn(state.h.shape, generator=torch.Generator().manual_seed(2), dtype=F64)
    results = []
    with cell.window():
        for fast in (0, F):
            consts = cell.window_consts(fast_slots=fast)
            assert (consts.fast_w1 is None) == (fast == 0)
            _, info = cell.step_sparse(u, eps, state, sel, static=True, consts=consts)
            new = info["state"]
            assert isinstance(new, CellGraphState)
            (grad,) = torch.autograd.grad((new.h * weight).sum(), state.h)
            results.append((new.h.detach(), new.q.detach(), grad))
    for mine, theirs in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(mine, theirs, **TIGHT)
    assert float(results[0][2].abs().max()) > 1e-3


@pytest.mark.parametrize("merged", [False, True])
def test_fused_products_with_fixed_slots_are_the_fused_products(merged: bool) -> None:
    """:meth:`PDGCell.sparse_products` (and, merged, the folded sweep's
    :meth:`~PDGCell.sparse_products_raw`): the gate's pre-activation and ``W_out``'s output of
    every slot with the fixed slots' products against the all-units products."""
    cfg = _cfg(*BASE, *PINNED)
    model, u, eps, state, sel = _one_position(cfg)
    cell = model.cell
    F = fixed_fast_slots(model)
    B, K, d = cfg.train.batch, cfg.model.k_select, cfg.model.d
    args = (u, eps, state, sel.cell_idx, sel.unit_idx, sel.valid)
    with torch.no_grad(), cell.window():
        plain = _fused_consts(model, 0, merged=merged)
        fast = _fused_consts(model, F, merged=merged)
        vh = cell.V(state.h)
        want = cell.sparse_products(*args, plain, vh=vh)
        got = cell.sparse_products(*args, fast, vh=vh)
        for mine, theirs in zip(got, want, strict=True):
            assert mine.shape == (B, K, d)
            torch.testing.assert_close(mine, theirs, **TIGHT)
        if merged:
            want_raw = cell.sparse_products_raw(*args, plain, vh=vh)
            got_raw = cell.sparse_products_raw(*args, fast, vh=vh)
            assert got_raw[0].shape == (B * K, d) and got_raw[1].shape == (B, K, d)
            assert got_raw[0].is_contiguous()  # the end-of-round kernel reads it by its stride
            for mine, theirs in zip(got_raw, want_raw, strict=True):
                torch.testing.assert_close(mine, theirs, **TIGHT)


# --------------------------------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------------------------------


def _tape(model: HSN, cfg: RunConfig, inputs: tuple) -> dict[str, Tensor]:
    tokens, ctx, state, reset = inputs
    with torch.no_grad():
        pro = model._window_prologue(
            tokens, ctx, state, reset, model._window_mode(), reset_first=True
        )
        return run_tape(_tape_block, model, pro, state, reset, cfg.train.unroll)


@pytest.mark.parametrize("extra", [(), ("model.age_max=2",), ("model.budget=0.1",)])
def test_sweep_with_fixed_slots_records_the_same_tape(
    extra: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The split window's sweep of a model with fixed fast units: the per-slot products give
    the tape of the all-units products, every decision identical and every value to rounding
    (also with refresh probes, and with a budget that cuts fast slots)."""
    cfg = _cfg(*BASE, *PINNED, *extra)
    model, inputs, _ = _case(cfg)
    assert fixed_fast_slots(model) == 2
    got = _tape(model, cfg, inputs)
    monkeypatch.setattr(compile_module, "fixed_fast_slots", lambda model: 0)
    want = _tape(model, cfg, inputs)
    for key in (*TAPE_KEYS, "h", "q", "age", "pos"):
        if got[key].dtype.is_floating_point:
            torch.testing.assert_close(got[key], want[key], **TIGHT, msg=lambda m, k=key: k + m)
        else:
            assert torch.equal(got[key], want[key]), key
    assert not torch.equal(got["tape_hsel"], want["tape_hsel"])  # the other products did run


@pytest.mark.parametrize("extra", [(), FAST, READ], ids=["routed", "fast_routed", "fast_read"])
def test_a_model_without_fixed_slots_takes_the_sweep_as_it_was(
    extra: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No fast cells, or fast cells whose unit the scorer chooses: the sweep's constants carry
    no slot layouts and the static round never enters the new pass."""
    cfg = _cfg(*BASE, *extra)
    model, inputs, _ = _case(cfg)
    assert fixed_fast_slots(model) == 0
    with model.cell.window():
        consts = model.cell.window_consts(fast_slots=fixed_fast_slots(model))
    assert consts.fast_w1 is None and consts.fast_w2 is None

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the fixed-slot pass ran for a model without fixed slots")

    monkeypatch.setattr(UnitBank, "forward_static_slots", refuse)
    monkeypatch.setattr(UnitBank, "forward_static_chosen_slots", refuse)
    monkeypatch.setattr(UnitBank, "slot_layout", refuse)
    tape = _tape(model, cfg, inputs)
    assert tape["sel_cell_idx"].shape[-1] == cfg.model.k_select


def test_the_autograd_window_keeps_its_round(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the split window's sweep asks for the slot layouts: the autograd window of a model
    with fixed fast units (eager, and the capturable round with the window's own constants)
    runs the round it ran, and never the fixed-slot pass."""
    cfg = _cfg(*BASE, *PINNED)
    model, (tokens, ctx, state, reset), _ = _case(cfg)
    with model.cell.window():
        assert model.cell.window_consts().fast_w1 is None

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the fixed-slot pass ran in the autograd window")

    monkeypatch.setattr(UnitBank, "forward_static_slots", refuse)
    monkeypatch.setattr(UnitBank, "slot_layout", refuse)
    logits, _, _ = model(tokens, ctx, state, reset)
    position = model._position

    def static(*args: object, **kwargs: object) -> object:
        consts = model.cell.window_consts()
        return position(*args, **{**kwargs, "static": True, "consts": consts})  # type: ignore[arg-type]

    monkeypatch.setattr(model, "_position", static)
    compiled_round, _, _ = model(tokens, ctx, state, reset)
    torch.testing.assert_close(compiled_round, logits, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("fused", [False, True])
def test_sweep_twin_holds_the_fixed_slots_layouts(fused: bool) -> None:
    """The sweep twin (the compiled sweep's model): its per-slot buffers are the bank's slot
    layout, with the gate's columns on a merged twin, before and after the weights change; a
    block of the twin records the model's tape."""
    cfg = _cfg(*BASE, *PINNED)
    model, inputs, _ = _case(cfg)
    tokens, ctx, state, reset = inputs
    cell, bank = model.cell, model.cell.units
    assert sweep_merge_error(model) is None
    sweep = SweepTwin(model, F64, fused=fused)
    assert sweep.fast_slots == 2 and sweep.merged == fused
    g = torch.Generator().manual_seed(9)
    for _ in range(2):
        with torch.no_grad(), cell.window():
            sweep.refresh()
            want = bank.slot_layout(2, rider=cell.gate if fused else None)
            for name, value in zip(("w1", "b1", "w2", "b2"), want, strict=True):
                assert torch.equal(sweep.twin.cell.units.get_buffer(f"sweep_fast_{name}"), value)
            for p in model.parameters():  # an optimiser step's worth of change
                p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    if fused:
        return  # the fused blocks' glue is CUDA kernels; their products are tested above
    T = cfg.train.unroll
    with torch.no_grad():
        pro = model._window_prologue(
            tokens, ctx, state, reset, model._window_mode(), reset_first=True
        )
        want_tape = run_tape(_tape_block, model, pro, state, reset, T)
        got_tape = run_tape(_tape_block, model, pro, state, reset, T, twin=sweep)
    for key in (*TAPE_KEYS, "h", "q", "age", "pos"):
        assert torch.equal(got_tape[key], want_tape[key]), key


def test_a_twin_without_fixed_slots_has_no_slot_buffers() -> None:
    model = _model(_cfg(*BASE))
    sweep = SweepTwin(model, F64, fused=True)
    assert sweep.fast_slots == 0
    assert "sweep_fast_w1" not in sweep.twin.cell.units._buffers


# --------------------------------------------------------------------------------------------------
# The reverse sweep's layouts
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        (),
        ("model.cand_norm=none",),
        ("model.cell_norm=none",),
        PINNED,
        (*PINNED, "model.budget=0.1"),
    ],
)
def test_cells_backward_in_the_static_layouts_is_the_per_unit_backward(
    extra: tuple[str, ...],
) -> None:
    """:func:`cell_state_vjp` with :class:`SweepLayouts` (the all-units operands for the
    routed slots, one batched product per layer for the fixed slots) against the product per
    unit, one position."""
    cfg = _cfg(*BASE, *extra)
    model, u, eps, state, sel = _one_position(cfg)
    cell = model.cell
    B, K, d = cfg.train.batch, cfg.model.k_select, cfg.model.d
    g = torch.Generator().manual_seed(8)
    dc = torch.randn(B, K, d, generator=g, dtype=F64)
    db = torch.randn(B, K, d, generator=g, dtype=F64)
    record: dict[str, Tensor] = {}
    with torch.no_grad(), cell.window():
        c, b = cell.replay_sparse(u, eps, state.h.detach(), sel, record=record)
        consts = cell_constants(cell, F64)
    n_fast = fixed_fast_slots(model)
    layouts = sweep_layouts(consts, n_fast)
    M, H = cfg.model.n_units, cfg.model.hidden
    assert layouts.w2_t.shape == (M * d, H) and layouts.w1_state_t.shape == (M * H + d, 2 * d)
    assert layouts.n_fast == n_fast and (layouts.fast_w2_t is None) == (n_fast == 0)
    safe = sel.cell_idx.masked_fill(~sel.valid, 0)
    values = {
        "c": c,
        "b": b,
        "h_old": state.h.detach().gather(1, safe[:, :, None].expand(B, K, d)),
        "units": sel.unit_idx.masked_fill(~sel.valid, 0),
        "y": record["y"],
        "alpha": record["alpha"],
        "hidden": record["hidden"],
        "gate_pre": record["gate_pre"],
        "m": record["m"],
        "rms": record.get("rms"),
        "consts": consts,
    }
    want = cell_state_vjp(dc, db, **values)
    got = cell_state_vjp(dc, db, **values, layouts=layouts)
    ok = sel.valid[..., None]  # a padded slot's result is masked by the sweep
    for mine, theirs in zip(got, want, strict=True):
        torch.testing.assert_close(
            torch.where(ok, mine, 0.0), torch.where(ok, theirs, 0.0), **TIGHT
        )
    assert float(want[0].abs().max()) > 1e-3 and float(want[1].abs().max()) > 1e-3


@pytest.mark.parametrize("extra", [(), PINNED], ids=["routed", "fast_cells"])
def test_sweep_in_the_static_layouts_is_the_reference_sweep(
    extra: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whole window's reverse sweep in the layouts a compiled sweep multiplies with: the
    same ``dL/dc``, ``dL/db`` and state cotangents as the reference sweep, on one record."""
    cfg = _cfg(*BASE, *extra)
    model, (tokens, ctx, state, reset), mask = _case(cfg)
    given: list[tuple] = []
    sweep = state_gradient.state_gradient_backward

    def keeping(*args: object, **kwargs: object) -> tuple[Tensor, Tensor]:
        given.append(args)
        return sweep(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(state_gradient, "state_gradient_backward", keeping)
    window = SplitWindow(model, cfg, None, state_gradient=True)
    logits, _, info = window(tokens, ctx, state, reset)
    _objective(logits, info, mask).backward()
    monkeypatch.undo()
    args = given[0]
    consts = args[-1]
    want_trace: dict[str, Tensor] = {}
    got_trace: dict[str, Tensor] = {}
    layouts = sweep_layouts(consts, fixed_fast_slots(model))
    with torch.no_grad():
        want = sweep(*args, trace=want_trace)
        got = sweep(*args, trace=got_trace, layouts=layouts)
    for mine, theirs in zip(got, want, strict=True):
        torch.testing.assert_close(mine, theirs, rtol=1e-11, atol=1e-13)
    for key in ("A", "R", "dh", "dvm"):
        torch.testing.assert_close(got_trace[key], want_trace[key], rtol=1e-11, atol=1e-13)
    assert float(want_trace["A"].abs().max()) > 1e-3
