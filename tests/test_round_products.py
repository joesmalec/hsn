"""``hardware.round_products=merged`` (opt-in, ``docs/word_event_speed.md``): the compiled
autograd window's sparse round with fewer and smaller matrix products.

``separate`` (the default, the round since P2.8) runs every unit's activation and second layer
on every selected (cell, unit) pair and gathers one output in ``M``, and gives the gate,
``W_out``, ``P`` and ``S`` a product each. ``merged`` runs the chosen unit's second layer only
(:meth:`hsn.model.units.UnitBank.forward_static_merged`), lets the gate ride in the unit bank's
first product, folds ``W_out`` into its second and gives ``P`` and ``S`` one product per layer
(:meth:`hsn.model.hsn.HSN.front_layout`). These tests check that it is the same function:

* the unit pass against the grouped pass in float64 (values and every gradient, for every
  split of the pairs over the units), the exact zero gradient of a unit no pair chose, the
  rider and the fold;
* the merged front against ``P`` and ``S`` in float64;
* the whole window: the capturable round with the merged products against the same round
  with the separate ones in float64, run eagerly (the loop over bytes and the loop over
  events; values, final state and every parameter gradient), and the compiled window with the
  key against the eager window and the compiled window without it, to the tolerance of the
  compile tests;
* the key's place in the configuration (off by default, out of the hash and of the memory key
  at its default) and its refusals.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import (
    HASH_OMIT_AT_DEFAULT,
    REPO_ROOT,
    RunConfig,
    config_hash,
    load_config,
    round_products_scope_error,
)
from hsn.device import seed_everything
from hsn.experiments.spec import memory_key
from hsn.model.hsn import HSN
from hsn.model.hsn_t import HSNT
from hsn.model.units import UnitBank, pick_hidden, pick_unit
from hsn.train import compile as compile_mod
from hsn.train.compile import CompiledWindow, compile_window
from test_a6_feedback_detach import A5
from test_hsn_t import CUT_C, READER
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from torch import Tensor, nn

RECIPE = REPO_ROOT / "configs" / "experiments" / "hsn_t_memory_words.yaml"
CARD = ("profile=rtx3060", "hardware.compile_mode=reduce-overhead")
MERGED = "hardware.round_products=merged"


def _bank(seed: int = 0) -> UnitBank:
    torch.manual_seed(seed)
    bank = UnitBank(8, 6, 10, "gelu", init="trunc_normal_0.02")
    with torch.no_grad():
        for p in bank.parameters():
            p.copy_(torch.randn(p.shape))
    return bank.double()


def _splits(P: int, M: int, generator: torch.Generator) -> dict[str, Tensor]:
    """Splits of ``P`` pairs over ``M`` units: uniform, every pair on one unit, two units,
    one pair on each unit and the rest on one, round robin, and a unit nobody chose."""
    rare = torch.cat([torch.arange(M), torch.full((max(0, P - M),), 3)])[:P]
    return {
        "uniform": torch.randint(0, M, (P,), generator=generator),
        "single": torch.full((P,), M - 1),
        "two": torch.randint(0, 2, (P,), generator=generator) * (M - 1),
        "rare": rare[torch.randperm(P, generator=generator)],
        "round_robin": torch.arange(P) % M,
        "one_unused": torch.randint(0, M - 1, (P,), generator=generator),
    }


# --------------------------------------------------------------------------------------------------
# The unit pass on the merged layouts
# --------------------------------------------------------------------------------------------------


def _maps(bank: UnitBank, seed: int = 3) -> tuple[nn.Linear, nn.Linear]:
    """A gate (a linear map of the bank's input with a bias) and a ``W_out`` (a linear map of
    its output without one), as the cell has them."""
    torch.manual_seed(seed)
    gate = nn.Linear(bank.in_dim, 5).to(bank.W1.dtype)
    out_map = nn.Linear(bank.d, bank.d, bias=False).to(bank.W1.dtype)
    return gate, out_map


@pytest.mark.parametrize("P", [1, 7, 36, 384])
def test_merged_pass_is_the_grouped_pass(P: int) -> None:
    """float64: the merged unit pass returns ``W_out(F_j(v))`` and the gate's linear of ``v``,
    for every split of the pairs over the units, with the gradients of the grouped pass (and
    of the all-units static pass) followed by the two linears: the input, the four stacked
    parameters, the gate's weight and bias and ``W_out``."""
    bank = _bank()
    gate, out_map = _maps(bank)
    generator = torch.Generator().manual_seed(P)
    params = [*bank.parameters(), gate.weight, gate.bias, out_map.weight]
    for name, units in _splits(P, bank.n_units, generator).items():
        v = torch.randn(P, bank.in_dim, dtype=torch.float64, generator=generator)
        v.requires_grad_(True)
        weights = torch.randn(2, P, bank.d, dtype=torch.float64, generator=generator)[:, :, :5]

        def loss(out: Tensor, rider: Tensor, weights: Tensor = weights) -> Tensor:
            return (out[:, :5] * weights[0]).sum() + (rider * weights[1]).sum()

        want, want_rider = out_map(bank.forward_selected(v, units)), gate(v)
        want_grads = torch.autograd.grad(loss(want, want_rider), [v, *params])
        got, rider = bank.forward_static_merged(v, units, bank.merged_layout(gate, out_map))
        assert got.shape == want.shape and rider.shape == want_rider.shape
        grads = torch.autograd.grad(loss(got, rider), [v, *params])
        torch.testing.assert_close(got, want, rtol=0.0, atol=1e-12, msg=name)
        torch.testing.assert_close(rider, want_rider, rtol=0.0, atol=1e-12, msg=name)
        static = out_map(bank.forward_static(v, units))
        torch.testing.assert_close(got, static, rtol=0.0, atol=1e-12, msg=name)
        for g, w in zip(grads, want_grads, strict=True):
            torch.testing.assert_close(g, w, rtol=0.0, atol=1e-11, msg=name)
        # a unit no pair chose gets an exact zero gradient, as in the all-units pass
        unused = [j for j in range(bank.n_units) if not bool((units == j).any())]
        for g in grads[1:5]:
            assert float(g[unused].abs().sum()) == 0.0


def test_picks_are_gathers_with_the_gather_gradient() -> None:
    """``pick_unit`` and ``pick_hidden`` return what ``gather`` and a slice return, and their
    own backward (one masked copy, one scatter into zeros) gives autograd's gradient of those,
    bit for bit: every place is written at most once."""
    generator = torch.Generator().manual_seed(5)
    P, M, H, r = 23, 8, 10, 5
    unit = torch.randint(0, M, (P,), generator=generator)
    x = torch.randn(P, M, H, dtype=torch.float64, generator=generator, requires_grad=True)
    weight = torch.randn(P, H, dtype=torch.float64, generator=generator)
    want = x.gather(1, unit[:, None, None].expand(P, 1, H)).squeeze(1)
    got = pick_unit(x, unit)
    assert torch.equal(got, want)
    (grad,) = torch.autograd.grad((got * weight).sum(), x)
    (want_grad,) = torch.autograd.grad((want * weight).sum(), x)
    assert torch.equal(grad, want_grad)
    first = torch.randn(P, M * H + r, dtype=torch.float64, generator=generator)
    first.requires_grad_(True)
    rider_weight = torch.randn(P, r, dtype=torch.float64, generator=generator)
    chosen, rider = pick_hidden(first, unit, M, H)
    want_chosen = first[:, : M * H].view(P, M, H).gather(1, unit[:, None, None].expand(P, 1, H))
    want_chosen, want_rider = want_chosen.squeeze(1), first[:, M * H :]
    assert torch.equal(chosen, want_chosen) and torch.equal(rider, want_rider)
    (grad,) = torch.autograd.grad((chosen * weight).sum() + (rider * rider_weight).sum(), first)
    (want_grad,) = torch.autograd.grad(
        (want_chosen * weight).sum() + (want_rider * rider_weight).sum(), first
    )
    assert torch.equal(grad, want_grad)
    # one output unused: its gradient reads as zeros
    (grad,) = torch.autograd.grad((pick_hidden(first, unit, M, H)[0] * weight).sum(), first)
    assert float(grad[:, M * H :].abs().sum()) == 0.0


def test_merged_layout_and_its_checks() -> None:
    bank = _bank()
    gate, out_map = _maps(bank)
    M, H, d = bank.n_units, bank.hidden, bank.d
    w1, b1, w2, b2 = bank.merged_layout(gate, out_map)
    assert w1.shape == (bank.in_dim, M * H + 5) and b1.shape == (M * H + 5,)
    assert w2.shape == (H, M * d) and b2.shape == (M * d,)
    assert torch.equal(w1[:, : M * H], bank._w1_flat()) and torch.equal(
        w1[:, M * H :], gate.weight.t()
    )
    for j in range(M):  # unit j's second layer with W_out folded in
        block = w2[:, j * d : (j + 1) * d]
        torch.testing.assert_close(block, bank.W2[j] @ out_map.weight.t(), rtol=0.0, atol=1e-14)
    with pytest.raises(ValueError, match="rider"):
        bank.merged_layout(nn.Linear(3, 5).double(), out_map)
    with pytest.raises(ValueError, match="without a bias"):
        bank.merged_layout(gate, nn.Linear(d, d).double())
    v = torch.randn(4, bank.in_dim, dtype=torch.float64)
    units = torch.zeros(4, dtype=torch.int64)
    layout = (w1, b1, w2, b2)
    with pytest.raises(ValueError, match="expects v"):
        bank.forward_static_merged(v[:, :-1], units, layout)
    with pytest.raises(ValueError, match="unit_idx must be"):
        bank.forward_static_merged(v, units[:3], layout)
    with pytest.raises(TypeError, match="int64 or int32"):
        bank.forward_static_merged(v, units.float(), layout)
    with pytest.raises(ValueError, match="layout"):
        bank.forward_static_merged(v, units, (w1[:, : M * H], b1, w2, b2))
    got, _ = bank.forward_static_merged(v, units.to(torch.int32), layout)
    want = out_map(bank.forward_selected(v, units))
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-12)


def test_merged_pass_under_autocast_keeps_the_masters_dtype() -> None:
    """Under autocast the layout is built in the masters' dtype (outside autocast), the
    products run in the autocast dtype, as the all-units pass's do, and the gradients arrive
    in the masters' dtype."""
    bank = _bank().float()
    gate, out_map = _maps(bank)
    v = torch.randn(40, bank.in_dim, requires_grad=True)
    units = torch.randint(0, bank.n_units, (40,))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        layout = bank.merged_layout(gate, out_map)
        assert all(t.dtype == torch.float32 for t in layout)
        got, rider = bank.forward_static_merged(v, units, layout)
        want = out_map(bank.forward_static(v, units))
    assert got.dtype == rider.dtype == want.dtype == torch.bfloat16
    error = float((got.float() - want.float()).abs().max().detach())
    assert error < 0.1 * float(want.float().abs().max().detach())
    grads = torch.autograd.grad(got.float().sum(), [v, bank.W1, out_map.weight])
    assert all(g.dtype == torch.float32 for g in grads)


# --------------------------------------------------------------------------------------------------
# The merged front
# --------------------------------------------------------------------------------------------------


def _hsn_t(*extra: str, training: bool = True, dtype: torch.dtype = torch.float32) -> HSNT:
    cfg = p28._cfg(
        "model.routing=topk",
        "train.gate_mode=ste",
        *A5,
        *READER,
        *extra,
    )
    seed_everything(0)
    model = HSNT.from_config(cfg)
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.3)
    return model.to(dtype).train(training)


@pytest.mark.parametrize("cut", [(), (CUT_C,)], ids=["no_cut", "cut_c"])
def test_merged_front_is_p_and_s(cut: tuple[str, ...]) -> None:
    """float64: ``(ẑ, u)`` of the merged front equal ``P(q)`` and ``S([z; q; ctx; c])``, and
    so do the gradients into ``q`` (through ``P`` always, through ``S`` unless the summary read
    is cut), into ``S``'s other inputs and into the four linears."""
    model = _hsn_t(*cut, dtype=torch.float64)
    cfg = model.cfg
    B = 5
    generator = torch.Generator().manual_seed(1)
    width = cfg.context_dim + cfg.reader_width
    z = torch.randn(B, cfg.d, dtype=torch.float64, generator=generator, requires_grad=True)
    q = torch.randn(B, cfg.d, dtype=torch.float64, generator=generator, requires_grad=True)
    ctx = torch.randn(B, width, dtype=torch.float64, generator=generator, requires_grad=True)
    h = torch.zeros(B, cfg.n_cells, cfg.d, dtype=torch.float64)
    weights = torch.randn(2, B, cfg.d, dtype=torch.float64, generator=generator)
    params = [z, q, ctx, *model.predictor.parameters(), *model.shared.parameters()]

    def grads(z_hat: Tensor, u: Tensor) -> tuple[Tensor, ...]:
        return torch.autograd.grad((z_hat * weights[0]).sum() + (u * weights[1]).sum(), params)

    want = (model.predictor(q), model.shared(model._shared_input(z, q, ctx, h)))
    with model.cell.window():
        consts = model.merged_consts(units=False)
        assert consts.front_w1 is not None and consts.fold_w2 is None
        assert consts.front_w1.shape == (2 * cfg.hidden, model.shared.in_dim + cfg.d)
        got = model._front(z, q, ctx, h, consts)
        for g, w in zip(got, want, strict=True):
            torch.testing.assert_close(g, w, rtol=0.0, atol=1e-12)
        for g, w in zip(grads(*got), grads(*want), strict=True):
            torch.testing.assert_close(g, w, rtol=0.0, atol=1e-11)
        assert model.merged_consts(front=False).front_w1 is None


def test_front_keeps_its_own_products_without_a_shared_activation() -> None:
    model = _hsn_t()
    model.predictor.activation_name = "other"
    assert model.front_layout() is None
    with model.cell.window():
        consts = model.merged_consts()
    assert consts.front_w1 is None and consts.fold_w2 is not None and consts.w1_gate is not None


# --------------------------------------------------------------------------------------------------
# The window: the capturable round with the merged products
# --------------------------------------------------------------------------------------------------


def _eager_static(model: HSN, *, merged: bool) -> CompiledWindow:
    """The compiled window's code without a compiler: the block functions themselves, so the
    capturable round (``static=True``) runs eagerly, in any dtype."""
    window = CompiledWindow(model, p28._ctx(), "default", "eager", merged=merged)
    window._block = compile_mod._block
    window._bucket_blocks = {0: compile_mod._block}
    window._event_blocks = {0: compile_mod._event_block}
    return window


def _text_inputs(model: HSN) -> tuple[Tensor, Tensor, p28.CellGraphState, Tensor]:
    tokens, cx, state, reset = p28._case_inputs(model, True)
    tokens[:, 0::3], tokens[:, 1::3], tokens[:, 2::3] = 97, 120, 32  # a word end every third byte
    tokens[0] = 97  # and a row without one
    return tokens, cx, state, reset


@pytest.mark.parametrize("events", ["byte", "word"])
@pytest.mark.parametrize("extra", [(), (CUT_C,), ("model.cand_norm=none",)], ids=str)
@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
@pytest.mark.usefixtures("small_blocks")
def test_merged_round_is_the_separate_round_in_float64(
    events: str, extra: tuple[str, ...], training: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The capturable round with the merged products against the same round with the separate
    ones, both run eagerly in float64 over four blocks: logits, final state and ``info`` to
    1e-10, the selections exactly, and every parameter gradient."""
    monkeypatch.setattr("hsn.model.hsn.EVENT_BLOCK", compile_mod.MAX_BLOCK)
    model = _hsn_t(*extra, f"model.write_events={events}", training=training, dtype=torch.float64)
    tokens, cx, state, reset = _text_inputs(model) if events == "word" else p28._inputs(model)
    state = p28.CellGraphState._make(
        h=state.h.double(), q=state.q.double(), age=state.age, pos=state.pos
    )
    ref = p28._run(lambda: _eager_static(model, merged=False)(tokens, cx, state, reset))
    got = p28._run(lambda: _eager_static(model, merged=True)(tokens, cx, state, reset))
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= 1e-10, diff
    if training:
        grads, ref_grads = p28._grads(model, got), p28._grads(model, ref)
        assert p28._grad_diff(grads, ref_grads) <= 1e-9
        assert sum(g is not None and float(g.abs().sum()) > 0.0 for g in grads) > 10


@pytest.mark.parametrize("name", ["topk_ste", "topk_eval"])
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_with_merged_products_equals_eager(name: str) -> None:
    """The loop over bytes (the HSN): the compiled window with the key equals the eager
    window, and the compiled window without the key, to the tolerance of the compile tests
    (values and parameter gradients; the selections exactly)."""
    overrides, training = p28.MODES[name]
    model = p28._model(p28._cfg(*overrides), training=training, scale=0.3)
    tokens, cx, state, reset = p28._case_inputs(model, True)
    tol = p28.TOL[("cpu", "fp32")]
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    assert p28._margin(ref[2], model.cfg.k_select) > p28.MIN_MARGIN
    plain = p28._run(lambda: p28._window(model, p28._ctx())(tokens, cx, state, reset))
    window = compile_window(
        model, p28._ctx(), "reduce-overhead", backend=p28.CPU_BACKEND, merged=True
    )
    assert isinstance(window, CompiledWindow) and window.merged
    got = p28._run(lambda: window(tokens, cx, state, reset))
    for other in (ref, plain):
        diff = p28._max_diff(got, other)
        assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    if training:
        grads = p28._grads(model, got)
        assert p28._grad_diff(grads, p28._grads(model, ref)) <= tol["grads"]
        assert p28._grad_diff(grads, p28._grads(model, plain)) <= tol["grads"]
        assert sum(g is not None and float(g.abs().sum()) > 0.0 for g in grads) > 10


@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
@pytest.mark.usefixtures("small_blocks")
def test_compiled_event_window_with_merged_products_equals_eager(
    training: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop over events (``model.write_events=word``, the window the word-event arms train
    on): the compiled window with the key equals the eager event loop and the compiled window
    without it, and a second window with more event slots runs on the same graph."""
    from torch._dynamo.utils import counters

    monkeypatch.setattr("hsn.model.hsn.EVENT_BLOCK", compile_mod.MAX_BLOCK)
    model = _hsn_t("model.write_events=word", training=training)
    tokens, cx, state, reset = _text_inputs(model)
    tol = p28.TOL[("cpu", "fp32")]
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    plain = p28._run(lambda: p28._window(model, p28._ctx())(tokens, cx, state, reset))
    window = compile_window(
        model, p28._ctx(), "reduce-overhead", backend=p28.CPU_BACKEND, merged=True
    )
    assert isinstance(window, CompiledWindow)
    got = p28._run(lambda: window(tokens, cx, state, reset))
    for other in (ref, plain):
        diff = p28._max_diff(got, other)
        assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    if training:
        grads = p28._grads(model, got)
        assert p28._grad_diff(grads, p28._grads(model, ref)) <= tol["grads"]
        assert p28._grad_diff(grads, p28._grads(model, plain)) <= tol["grads"]
    graphs = counters["stats"]["unique_graphs"]
    tokens = tokens.clone()
    tokens[1, 0::2], tokens[1, 1::2] = 97, 32  # more events: another block of slots
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    with torch.compiler.set_stance("fail_on_recompile"):
        got = p28._run(lambda: window(tokens, cx, state, reset))
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    assert counters["stats"]["unique_graphs"] == graphs


@pytest.mark.usefixtures("small_blocks")
def test_compiled_recipe_window_with_merged_products_compiles_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window of the word-event recipe in small (top-K with straight-through gates, the
    state gradient's cut, A9's two-byte features, the address gate, word events with the
    exact word address) with the key: one graph for consecutive windows of 2 to ``T`` event
    slots, each equal to the eager event loop (values, integer outputs, the carried address
    and the parameter gradients)."""
    import functools

    from hsn.model.hsn import address_history
    from test_word_address import NGRAM, STE, _compiled_windows, _wa
    from torch._dynamo.utils import counters

    monkeypatch.setattr("hsn.model.hsn.EVENT_BLOCK", compile_mod.MAX_BLOCK)
    T = 20
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        STE,
        CUT_C,
        *READER,
        *NGRAM,
        "model.address_gate=true",
        "model.address_gate_input=summary",
        "model.write_events=word",
        _wa("exact"),
    )
    seed_everything(0)
    model = HSNT.from_config(cfg)
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.3)
    model.train(True)
    window = compile_window(
        model, p28._ctx(), "reduce-overhead", backend=p28.CPU_BACKEND, merged=True
    )
    assert isinstance(window, CompiledWindow)
    tol = p28.TOL[("cpu", "fp32")]
    state = p28._inputs(model, b=3, t=T)[2]
    carry: Tensor | None = None
    graphs, slots = -1, []
    for i, (tokens, ctx, reset) in enumerate(_compiled_windows(T)):
        ref = p28._run(functools.partial(model, tokens, ctx, state, reset, prev_tokens=carry))
        compiled = functools.partial(window, tokens, ctx, state, reset, prev_tokens=carry)
        if i == 0:
            got = p28._run(compiled)
            graphs = counters["stats"]["unique_graphs"]
        else:
            with torch.compiler.set_stance("fail_on_recompile"):
                got = p28._run(compiled)
        diff = p28._max_diff(got, ref)
        assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], (i, diff)
        assert p28._grad_diff(p28._grads(model, got), p28._grads(model, ref)) <= tol["grads"], i
        slots.append(int(ref[2]["event_valid"].shape[1]))
        carry = address_history(cfg.model, tokens, carry, ref[2])
        state = got[1].detach()
    assert counters["stats"]["unique_graphs"] == graphs
    assert len(set(slots)) == len(slots) and T in slots, slots


@pytest.mark.cuda
@pytest.mark.parametrize("case", ["topk_ste", "topk_eval", "events_train", "events_eval"])
@pytest.mark.usefixtures("small_blocks")
def test_merged_products_equal_eager_cuda(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """On the card, with CUDA graphs (warm-up, recording, replay): the compiled window with
    the key equals the eager window in fp32 to the measured tolerance of the compile tests,
    for the loop over bytes (the HSN) and the loop over events (HSN-T with word events)."""
    ctx = p28._cuda_ctx()
    if case.startswith("events"):
        monkeypatch.setattr("hsn.model.hsn.EVENT_BLOCK", compile_mod.MAX_BLOCK)
        training = case == "events_train"
        model: HSN = _hsn_t("model.write_events=word", CUT_C, training=training).to(ctx.device)
        tokens, cx, state, reset = _text_inputs(model)
    else:
        overrides, training = p28.MODES[case]
        model = p28._model(p28._cfg(*overrides), training=training, scale=0.3).to(ctx.device)
        tokens, cx, state, reset = p28._case_inputs(model, True)
    window = compile_window(model, ctx, "reduce-overhead", backend="inductor", merged=True)
    assert isinstance(window, CompiledWindow) and window.cuda_graphs
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    for _ in range(3):  # warm-up, recording, replay
        got = p28._run(lambda: window(tokens, cx, state, reset))
    tol = p28.TOL[("cuda", "fp32")]
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    if training:
        assert p28._grad_diff(p28._grads(model, got), p28._grads(model, ref)) <= tol["grads"]


@pytest.mark.cuda
def test_merged_pass_under_bf16_on_the_card() -> None:
    """On the card under bf16 autocast, at the recipe's sizes (384 pairs, 8 units, 256 -> 512
    -> 64): the merged unit pass against the all-units pass, the gate's and ``W_out``'s own
    products. Both round to bf16; they agree to that rounding, in the outputs and in every
    gradient."""
    ctx = p28._cuda_ctx()
    torch.manual_seed(0)
    bank = UnitBank(8, 64, 512, "gelu", init="fan_in").to(ctx.device)
    gate = nn.Linear(bank.in_dim, 64).to(ctx.device)
    out_map = nn.Linear(64, 64, bias=False).to(ctx.device)
    P = 384
    v = torch.randn(P, bank.in_dim, device=ctx.device, requires_grad=True)
    params = [v, *bank.parameters(), gate.weight, gate.bias, out_map.weight]
    weights = torch.randn(2, P, 64, device=ctx.device)
    for name, units in _splits(P, bank.n_units, torch.Generator().manual_seed(1)).items():
        units = units.to(ctx.device)

        def run(merged: bool, units: Tensor = units) -> tuple[Tensor, Tensor, list[Tensor]]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if merged:
                    layout = bank.merged_layout(gate, out_map)
                    out, rider = bank.forward_static_merged(v, units, layout)
                else:
                    out, rider = out_map(bank.forward_static(v, units)), gate(v)
            loss = (out.float() * weights[0]).sum() + (rider.float() * weights[1]).sum()
            return out.float(), rider.float(), list(torch.autograd.grad(loss, params))

        out, rider, grads = run(True)
        want, want_rider, want_grads = run(False)
        scale = float(want.abs().max())
        assert float((out - want).abs().max()) <= 0.02 * scale, name
        assert float((rider - want_rider).abs().max()) <= 0.02 * float(want_rider.abs().max())
        for g, w in zip(grads, want_grads, strict=True):
            assert g.dtype == w.dtype == torch.float32
            assert float((g - w).abs().max()) <= 0.03 * float(w.abs().max()) + 1e-6, name


@pytest.mark.usefixtures("small_blocks")
def test_merged_products_with_the_causal_reader_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """``model.reader_mask=causal`` (``docs/reader_causal_flash.md``) on a memory run with a
    whole-window reader, together with the key: the reader runs in the window's prologue and
    the merged round after it, so the compiled event window with both equals the eager one
    (and the eager one with the document mask, on a window without a reset after column 0)."""
    monkeypatch.setattr("hsn.model.hsn.EVENT_BLOCK", compile_mod.MAX_BLOCK)
    words = ("model.write_events=word", CUT_C, "data.align_docs=true")
    document = _hsn_t(*words)
    causal = _hsn_t(*words, "model.reader_mask=causal")
    assert causal.reader is not None and causal.reader.causal
    for a, b in zip(document.parameters(), causal.parameters(), strict=True):
        assert torch.equal(a, b)
    tokens, cx, state, reset = _text_inputs(causal)
    assert not bool(reset[:, 1:].any())
    tol = p28.TOL[("cpu", "fp32")]
    ref = p28._run(lambda: causal(tokens, cx, state, reset))
    same = p28._max_diff(ref, p28._run(lambda: document(tokens, cx, state, reset)))
    assert same["exact"] == 0.0 and same["values"] <= tol["values"], same
    window = compile_window(
        causal, p28._ctx(), "reduce-overhead", backend=p28.CPU_BACKEND, merged=True
    )
    got = p28._run(lambda: window(tokens, cx, state, reset))
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    assert p28._grad_diff(p28._grads(causal, got), p28._grads(causal, ref)) <= tol["grads"]


# --------------------------------------------------------------------------------------------------
# The configuration key
# --------------------------------------------------------------------------------------------------


def test_key_is_off_by_default_and_left_out_of_the_hash() -> None:
    base = load_config(RECIPE, list(CARD))
    assert base.hardware.round_products == "separate"
    assert "hardware.round_products" in HASH_OMIT_AT_DEFAULT
    off = load_config(RECIPE, [*CARD, "hardware.round_products=separate"])
    assert config_hash(off) == config_hash(base)
    assert "round_products" not in base.hardware.model_dump(exclude_defaults=True)
    on = load_config(RECIPE, [*CARD, MERGED])
    assert on.hardware.round_products == "merged" and round_products_scope_error(on) is None
    assert config_hash(on) != config_hash(base)
    assert on.model == base.model and on.train == base.train  # the model and its training
    # the experiment database: the key at its default leaves the memory key (the stored step
    # time and memory estimates) as it was, and an arm with the key has its own
    protocol = {"name": "train_eval", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
    assert memory_key(off, "pdg", protocol) == memory_key(base, "pdg", protocol)
    assert memory_key(on, "pdg", protocol) != memory_key(base, "pdg", protocol)


@pytest.mark.parametrize(
    ("path", "overrides", "reason"),
    [
        (RECIPE, (MERGED,), "hardware.compile_mode=none"),
        (p28.REF_A1, (*CARD, MERGED), "model.routing='dense'"),
        (RECIPE, (*CARD, "hardware.round_products=fused"), "'separate' or 'merged'"),
        (
            REPO_ROOT / "configs" / "experiments" / "hsn_t_nomem.yaml",
            (*CARD, MERGED),
            "has no sparse round",
        ),
    ],
)
def test_key_is_refused_where_no_compiled_sparse_round_runs(
    path: Path, overrides: tuple[str, ...], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        load_config(path, list(overrides))


def test_key_stands_beside_the_reader_mask_and_the_host_sync_refusals() -> None:
    """With main's keys of 17 Nov 2026: the key loads together with ``model.reader_mask=causal``
    on the recipe with a whole-window reader (the top0 reader), and the generic trainer's
    ``train.host_sync`` / ``train.fused_adamw`` stay refused for a memory run, with the key
    as without it."""
    top0 = (
        "model.reader_width=160",
        "model.reader_layers=4",
        "model.reader_rotary_pairs=[20,20,20,0]",
        "model.reader_span=null",
    )
    both = load_config(RECIPE, [*CARD, *top0, "model.reader_mask=causal", MERGED])
    assert both.model.reader_mask == "causal" and both.hardware.round_products == "merged"
    assert round_products_scope_error(both) is None
    with pytest.raises(ValueError, match="model.reader_span=32"):  # the recipe's 32-byte reader
        load_config(RECIPE, [*CARD, "model.reader_mask=causal", MERGED])
    for extra in ((), (MERGED,)):
        for key in ("train.host_sync=stats", "train.fused_adamw=true"):
            with pytest.raises(ValueError, match="trains with the HSN trainer"):
                load_config(RECIPE, [*CARD, *top0, *extra, key])


def test_trainer_builds_the_window_the_key_names(tmp_path: Path) -> None:
    from hsn.device import resolve
    from hsn.runner import build_model
    from hsn.train.tbptt import TBPTTTrainer
    from p1_15_support import hsn_config

    def window(*extra: str) -> CompiledWindow:
        cfg: RunConfig = hsn_config(
            4, *A5, "train.gate_mode=ste", "train.soft_until_step=0", *extra
        )
        seed_everything(0)
        trainer = TBPTTTrainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path)
        assert trainer.compiled_window is not None
        return trainer.compiled_window

    assert not window("hardware.compile_mode=default").merged
    assert window("hardware.compile_mode=default", MERGED).merged
