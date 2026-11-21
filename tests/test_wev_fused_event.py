"""The fused word-event block, phase 1 (``docs/wev_fused_event.md``, :mod:`hsn.train.event_fused`):
the event loop of ``model.write_events`` as a forward sweep without autograd, one autograd node
and a hand-written backward pass.

* In float64 the block equals the model's own autograd event loop (:meth:`HSN.forward`): the
  logits, the final state, every value of ``info``, the gradient of every parameter and of the
  window's start state, for the word-event recipe's keys and for each of them varied.
* The gradient crosses windows (``train.tbptt_windows > 1``: the start state is not detached).
* The scope's refusals.
* Under the products' rounding (bf16 on the CPU) the reference stays close to fp32.
* On the card: the kernels against the PyTorch reference (the tape, slot by slot) and against
  the autograd event loop, in fp32 and under bf16 autocast.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from hsn.config import RunConfig
from hsn.device import seed_everything
from hsn.kernels import window as window_kernels
from hsn.losses import balance_loss
from hsn.model.hsn import HSN, event_block
from hsn.model.hsn_t import HSNT
from hsn.model.state import CellGraphState
from hsn.train import event_fused as ef
from test_a6_feedback_detach import A5, _cfg
from test_hsn_t import CUT_C, READER
from torch import Tensor

#: The word-event recipe's memory keys on the six-cell model of the A6 tests (d = 8): the
#: state gradient, executed credit, both norms, A9's two address bytes carried, A11's wrapped
#: position, A12's gate on the carried summary (A13), word events with the exact word address,
#: refresh probes after three slots.
RECIPE: tuple[str, ...] = (
    *A5,
    *READER,
    CUT_C,
    "train.unit_credit=executed",
    "train.p_dense=0.0",
    "model.cell_norm=rms",
    "model.cand_norm=rms",
    "model.address_ngram=2",
    "model.address_history=ngram",
    "model.address_gate=true",
    "model.address_gate_input=summary",
    "model.scorer_position=wrap",
    "model.write_events=word",
    "model.word_address=exact",
    "model.word_contexts=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]",
    "model.age_max=3",
)
B, T = 4, 12

VARIANTS: dict[str, tuple[str, ...]] = {
    "recipe": (),
    "gate_on_u": ("model.address_gate_input=u",),
    "no_gate": ("model.address_gate=false", "model.address_gate_input=u"),
    "one_address_byte": ("model.address_ngram=1", "model.address_history=byte"),
    "no_probes": ("model.age_max=1000",),
    "every_cell": ("model.k_select=6",),
    "one_cell": ("model.k_select=1",),
    "sigmoid_edges": ("model.edge_norm=sigmoid",),
    "silu": ("model.activation=silu",),
    "relu": ("model.activation=relu",),
    "tanh": ("model.activation=tanh",),
    "temperature": ("train.temperature_start=1.7", "train.temperature_end=1.7"),
    "byte_events": ("model.write_events=word", "model.word_address=none"),
    "more_units": ("model.n_units=5", "model.k_select=3"),
}


def _model(*extra: str, dtype: torch.dtype = torch.float64, scale: float = 0.3) -> HSNT:
    cfg: RunConfig = _cfg(*RECIPE, *extra)
    seed_everything(0)
    model = HSNT.from_config(cfg).to(dtype)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():  # away from the symmetric start, so every path carries a gradient
        for p in model.parameters():
            p.add_((scale * torch.randn(p.shape, generator=generator)).to(p.dtype))
    return model.train()


def _inputs(
    model: HSN, seed: int = 3, b: int = B, t: int = T
) -> tuple[Tensor, Tensor, CellGraphState, Tensor]:
    """Text with a word end every third byte, a row without one (a padding row), a row that
    starts a document at column 0 and one that starts one mid-window; a start state away from
    zero with ages past the probes' threshold."""
    generator = torch.Generator().manual_seed(seed)
    m = model.cfg
    tokens = torch.randint(97, 123, (b, t), generator=generator)
    tokens[:, 2::3] = 32
    tokens[0] = 97
    ctx = torch.randint(0, m.n_context, (b, t), generator=generator)
    h = (torch.rand(b, m.n_cells, m.d, generator=generator) * 1.6 - 0.8).to(model.dtype)
    state = model.init_state(b).with_dense(h)
    for _ in range(5):
        state = state.tick(torch.rand(b, m.n_cells, generator=generator) < 0.4)
    reset = torch.zeros(b, t, dtype=torch.bool)
    reset[1, 5] = True
    reset[2, 0] = True
    return tokens, ctx, state, reset


def _objective(logits: Tensor, final: CellGraphState, info: dict[str, Tensor]) -> Tensor:
    """Everything a training step reads with a gradient: the logits, the prediction at the
    event slots, the balance term's logits, and the state after the window."""
    loss = logits.sin().sum()
    index = info["event_pos"][:, :, None].expand(-1, -1, info["z_hat"].shape[-1])
    live = info["event_valid"][:, :, None].to(logits.dtype)
    loss = loss + ((info["z_hat"] - info["z_target"].gather(1, index)).square() * live).sum()
    loss = loss + 5.0 * balance_loss(info["unit_logits"], info["unit_idx"], info["valid"])
    return loss + 0.3 * final.h.cos().sum() + 0.2 * final.q.sin().sum()


Window = Callable[..., tuple[Tensor, CellGraphState, dict[str, Tensor]]]


def _run(model: HSN, window: Window, inputs: tuple[Any, ...]) -> dict[str, Any]:
    tokens, ctx, state, reset = inputs
    model.zero_grad(set_to_none=True)
    start = CellGraphState._make(
        h=state.h.clone().requires_grad_(True),
        q=state.q.clone().requires_grad_(True),
        age=state.age,
        pos=state.pos,
    )
    logits, final, info = window(tokens, ctx, start, reset)
    loss = _objective(logits, final, info)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert start.h.grad is not None and start.q.grad is not None
    grads["<start h>"], grads["<start q>"] = start.h.grad, start.q.grad
    return {"loss": loss.detach(), "logits": logits.detach(), "final": final, "info": info,
            "grads": grads}  # fmt: skip


def _assert_same(ref: dict[str, Any], got: dict[str, Any], *, value: float, grad: float) -> None:
    torch.testing.assert_close(got["logits"], ref["logits"], rtol=0.0, atol=value)
    for name in ("h", "q"):
        torch.testing.assert_close(
            getattr(got["final"], name), getattr(ref["final"], name), rtol=0.0, atol=value
        )
    assert torch.equal(got["final"].age, ref["final"].age)
    assert torch.equal(got["final"].pos, ref["final"].pos)
    assert set(got["info"]) == set(ref["info"])
    for key, want in ref["info"].items():
        have = got["info"][key]
        if want.dtype.is_floating_point:
            torch.testing.assert_close(
                have.detach().to(want.dtype), want.detach(), rtol=0.0, atol=value, msg=key
            )
        else:
            assert torch.equal(have, want), key
    assert set(got["grads"]) == set(ref["grads"])
    for name, want in ref["grads"].items():
        scale = float(want.abs().max())
        torch.testing.assert_close(
            got["grads"][name], want, rtol=0.0, atol=grad * max(scale, 1e-30), msg=name
        )


# --------------------------------------------------------------------------------------------------
# float64: the fused block is the autograd event loop
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(VARIANTS))
def test_fused_block_equals_the_autograd_event_loop(name: str) -> None:
    """The window through the fused block equals :meth:`HSN.forward`'s event loop in float64:
    values to 1e-11, every parameter gradient and the start state's to 1e-9 of its largest
    entry (the scorer's included: the straight-through path is fp32 in both, operation for
    operation), integer and boolean records exact."""
    model = _model(*VARIANTS[name])
    assert ef.scope_error(model) is None
    inputs = _inputs(model)
    ref = _run(model, model, inputs)
    runs: list[ef.EventRun] = []
    got = _run(model, lambda *a: ef.forward_events(model, *a, runs=runs), inputs)
    _assert_same(ref, got, value=1e-11, grad=1e-9)
    tape = runs[0].tape
    assert int(tape["valid"].sum()) > 0
    live = got["info"]["event_valid"]
    assert not bool(tape["valid"][~live].any())  # a padding row writes nothing
    assert sum(float(g.abs().sum()) > 0.0 for g in got["grads"].values()) > 20


def test_the_cases_cover_probes_resets_and_padding_rows() -> None:
    """What the equality above rests on: the recipe case has refresh probes, a reset inside
    the window, a row that starts a document at column 0 and padding rows."""
    model = _model()
    tokens, ctx, state, reset = _inputs(model)
    runs: list[ef.EventRun] = []
    _, _, info = ef.forward_events(model, tokens, ctx, state, reset, runs=runs)
    tape = runs[0].tape
    assert float(tape["refreshed"].sum()) > 0.0
    assert bool(tape["reset"][:, 1:].any()) and bool(tape["reset"][:, 0].any())
    live = info["event_valid"]
    assert bool((~live).any()) and bool(live.any())
    assert int(tape["valid"].shape[1]) == event_block(T)


@pytest.mark.parametrize("name", ["recipe", "no_gate", "every_cell"])
def test_gradient_crosses_windows(name: str) -> None:
    """``train.tbptt_windows > 1``: two windows of a stream with the state carried attached and
    one backward pass. The second window's loss reaches the first window's weights through the
    carried state, in the fused block as in the autograd loop."""
    model = _model(*VARIANTS[name])
    tokens, ctx, state, reset = _inputs(model, t=2 * T)
    later = torch.zeros_like(reset[:, T:])  # the second window continues every document
    later[3, 4] = True

    def two(window: Window) -> dict[str, Tensor]:
        model.zero_grad(set_to_none=True)
        logits_a, mid, info_a = window(tokens[:, :T], ctx[:, :T], state, reset[:, :T])
        logits_b, final, info_b = window(
            tokens[:, T:], ctx[:, T:], mid, later, info_a.get("address_carry")
        )
        loss = _objective(logits_b, final, info_b) + 0.1 * logits_a.cos().sum()
        loss.backward()
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    ref = two(lambda *a: model(*a[:4], prev_tokens=a[4] if len(a) > 4 else None))
    got = two(lambda *a: ef.forward_events(model, *a))
    assert set(got) == set(ref)
    for key, want in ref.items():
        scale = float(want.abs().max())
        torch.testing.assert_close(got[key], want, rtol=0.0, atol=1e-9 * max(scale, 1e-30), msg=key)


def test_carried_gradient_is_not_zero() -> None:
    """The check above can fail: with the start state of the second window detached the
    gradients differ."""
    model = _model()
    tokens, ctx, state, reset = _inputs(model, t=2 * T)
    later = torch.zeros_like(reset[:, T:])

    def two(detach: bool) -> Tensor:
        model.zero_grad(set_to_none=True)
        _, mid, info_a = ef.forward_events(model, tokens[:, :T], ctx[:, :T], state, reset[:, :T])
        if detach:
            mid = mid.detach()
        logits, final, info = ef.forward_events(
            model, tokens[:, T:], ctx[:, T:], mid, later, info_a.get("address_carry")
        )
        _objective(logits, final, info).backward()
        grad = model.cell.units.W1.grad
        assert grad is not None
        return grad.clone()

    carried, cut = two(False), two(True)
    assert float((carried - cut).abs().max()) > 1e-6 * float(carried.abs().max())


# --------------------------------------------------------------------------------------------------
# The pieces
# --------------------------------------------------------------------------------------------------


def test_a_unit_no_slot_chose_gets_an_exact_zero() -> None:
    """The unit bank's gradients are sums over a unit's own slots: a unit that no slot of the
    block ran has an exact zero gradient, as in the grouped autograd pass."""
    model = _model("model.n_units=5", "model.k_select=1")
    inputs = _inputs(model)
    runs: list[ef.EventRun] = []
    got = _run(model, lambda *a: ef.forward_events(model, *a, runs=runs), inputs)
    tape = runs[0].tape
    used = set(tape["units"][tape["valid"]].tolist())
    unused = [j for j in range(5) if j not in used]
    assert unused, "the case needs a unit nobody chose"
    for j in unused:
        assert float(got["grads"]["cell.units.W1"][j].abs().max()) == 0.0
        assert float(got["grads"]["cell.units.W2"][j].abs().max()) == 0.0


def test_equal_scores_share_the_cell_mask_gradient() -> None:
    """Two units of a cell with one score (bf16 logits tie often): the cell mask's gradient
    is shared evenly among them, as the backward of the router's ``amax`` shares it, and the
    block still equals the autograd loop."""
    model = _model()
    with torch.no_grad():  # every unit's score equal: the scorer's weights and the gate at zero
        model.scorer.net.weight.zero_()
        model.scorer.net.bias.zero_()
        assert model.scorer.address_gate is not None
        model.scorer.address_gate.weight.zero_()
    inputs = _inputs(model)
    ref = _run(model, model, inputs)
    runs: list[ef.EventRun] = []
    got = _run(model, lambda *a: ef.forward_events(model, *a, runs=runs), inputs)
    tape = runs[0].tape
    assert int(tape["ties"].sum(dim=-1).max()) == model.cfg.n_units
    _assert_same(ref, got, value=1e-11, grad=1e-9)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (("train.unit_credit=softmax",), "unit_credit"),
        (("train.feedback_detach=[summary,eps,state,router]",), "state"),
        (("train.feedback_detach=[summary,router]",), "feedback_detach"),
        (("model.cand_norm=none",), "norm"),
        (("model.cell_norm=none",), "norm"),
        (
            ("model.scorer=mlp", "model.address_gate=false", "model.address_gate_input=u"),
            "scorer",
        ),
        (("model.address_gate_input=reader_prev",), "address_gate_input"),
        (("train.p_dense=0.5",), "p_dense"),
        (("model.cell_attention_heads=2",), "attention"),
    ],
)
def test_scope_refusals(overrides: tuple[str, ...], reason: str) -> None:
    model = _model(*overrides)
    message = ef.scope_error(model)
    assert message is not None and reason in message
    tokens, ctx, state, reset = _inputs(model)
    with pytest.raises(ValueError, match="does not take this model"):
        ef.forward_events(model, tokens, ctx, state, reset)


def test_evaluation_is_outside_the_scope() -> None:
    """The block is a training window: without the straight-through gate (evaluation) the
    model's own loop runs."""
    model = _model().eval()
    message = ef.scope_error(model)
    assert message is not None and "gate" in message


def test_products_rounding_stays_close_to_fp32() -> None:
    """The reference with the products rounded to bf16 where the compiled round under autocast
    holds bf16 (the kernels round at the same places): the same decisions here, the logits and
    the gradients within a few percent of fp32's."""
    model = _model(dtype=torch.float32, scale=0.1)
    inputs = _inputs(model)
    plain = _run(model, lambda *a: ef.forward_events(model, *a), inputs)
    rounded = _run(
        model, lambda *a: ef.forward_events(model, *a, product_dtype=torch.bfloat16), inputs
    )
    assert torch.equal(rounded["info"]["cell_idx"], plain["info"]["cell_idx"])
    assert torch.equal(rounded["info"]["unit_idx"], plain["info"]["unit_idx"])
    scale = float(plain["logits"].abs().max())
    assert float((rounded["logits"] - plain["logits"]).abs().max()) < 0.03 * scale
    num = sum(float((rounded["grads"][n] - g).square().sum()) for n, g in plain["grads"].items())
    den = sum(float(g.square().sum()) for g in plain["grads"].values())
    assert (num / den) ** 0.5 < 0.05


def test_fp32_reference_keeps_the_decisions() -> None:
    """fp32 on the CPU: identical decisions, values and gradients to fp32 rounding (the
    tolerances of the compiled window's tests)."""
    model = _model(dtype=torch.float32, scale=0.1)
    inputs = _inputs(model)
    ref = _run(model, model, inputs)
    got = _run(model, lambda *a: ef.forward_events(model, *a), inputs)
    _assert_same(ref, got, value=8e-6 * max(1.0, float(ref["logits"].abs().max())), grad=1.2e-4)


# --------------------------------------------------------------------------------------------------
# On the card: the kernels
# --------------------------------------------------------------------------------------------------

needs_card = pytest.mark.skipif(
    not (torch.cuda.is_available() and window_kernels.available()),
    reason="needs a CUDA device and the window kernels (scripts/build_window_kernels.py)",
)

#: A shape the kernels take (a power-of-two number of units, a thread per logit).
CARD_SHAPE: tuple[str, ...] = (
    "model.d=16",
    "model.context_dim=16",
    "model.n_cells=16",
    "model.in_degree=3",
    "model.n_units=4",
    "model.k_select=5",
    "model.hidden=32",
    "model.reader_width=16",
    "model.age_max=4",
    "train.unroll=32",
)


def _card_case(
    product: torch.dtype,
) -> tuple[HSNT, tuple[Tensor, Tensor, CellGraphState, Tensor], Any]:
    from hsn.kernels import event as ek

    model = _model(*CARD_SHAPE, dtype=torch.float32, scale=0.15).to("cuda")
    tokens, ctx, state, reset = _inputs(model, b=6, t=32)
    dev = torch.device("cuda")
    state = CellGraphState._make(
        h=state.h.to(dev), q=state.q.to(dev), age=state.age.to(dev), pos=state.pos.to(dev)
    )
    inputs = (tokens.to(dev), ctx.to(dev), state, reset.to(dev))
    pd = None if product == torch.float32 else product
    holder: dict[str, Any] = {}

    def forward(m: HSN, w: Any, spec: Any, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        if "kern" not in holder:
            z, scalars = args[3], args[5]
            holder["kern"] = ek.EventKernels(
                spec, w, ef.cell_constants(m.cell, torch.float32), B=int(z.shape[0]),
                E=int(z.shape[1]), n_scalars=int(scalars.shape[-1]), device=dev,
                product_dtype=product,
            )  # fmt: skip
        return holder["kern"].forward(m, w, spec, *args, **kwargs)

    def sweep(*args: Any) -> Any:
        return holder["kern"].sweep(*args)

    def window(*a: Any) -> Any:
        return ef.forward_events(model, *a, forward=forward, sweep=sweep, product_dtype=pd)

    return model, inputs, window


@needs_card
def test_kernels_equal_the_reference_fp32_cuda() -> None:
    """fp32 products: the kernels' window equals the PyTorch reference's and the autograd
    event loop's to the compiled window's CUDA tolerance, with identical decisions."""
    model, inputs, window = _card_case(torch.float32)
    ref = _run(model, model, inputs)
    plain = _run(model, lambda *a: ef.forward_events(model, *a), inputs)
    got = _run(model, window, inputs)
    scale = max(1.0, float(ref["logits"].abs().max()))
    for other in (ref, plain):
        _assert_same(other, got, value=1.5e-5 * scale, grad=2e-4)


@needs_card
def test_kernels_under_bf16_cuda() -> None:
    """bf16 products under autocast: against the PyTorch reference with the same rounding the
    logits stay within 2 % and the gradients within 5 % (relative L2) where the decisions
    agree; against the autograd loop under autocast the same bounds."""
    model, inputs, window = _card_case(torch.bfloat16)

    def cast(fn: Window) -> Window:
        def run(*a: Any) -> Any:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, final, info = fn(*a)
            return logits.float(), final, info

        return run

    ref = _run(model, cast(model), inputs)
    plain = _run(
        model,
        cast(lambda *a: ef.forward_events(model, *a, product_dtype=torch.bfloat16)),
        inputs,
    )
    got = _run(model, cast(window), inputs)
    for other in (plain, ref):
        same = (got["info"]["cell_idx"] == other["info"]["cell_idx"]).float().mean()
        assert float(same) > 0.9
        if float(same) == 1.0:
            scale = float(other["logits"].abs().max())
            assert float((got["logits"] - other["logits"]).abs().max()) < 0.02 * scale
            num = sum(
                float((got["grads"][n] - g).square().sum()) for n, g in other["grads"].items()
            )
            den = sum(float(g.square().sum()) for g in other["grads"].values())
            assert (num / den) ** 0.5 < 0.05
