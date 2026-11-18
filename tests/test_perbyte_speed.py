"""The per-byte step-time work of 18 Nov 2026 (``docs/perbyte_speed.md``).

**The block inputs.** The compiled autograd window hands every block of the loop over bytes
its inputs (the encoded bytes, the context with the reader's output, the scorer's scalars, the
resets) as dense tensors (:func:`hsn.train.compile._window_blocks`): one copy of the window,
laid out block by block, instead of a slice of the window per block. A change of layout only:

* the blocks hold the slices' values, are dense, and one block is the tensor itself;
* the gradient that reaches the window's tensor is the slices' bit for bit, a ``-0.0`` in a
  block's gradient included (the slices' sum turned it into ``+0.0``; so does the hook);
* the compiled window on the blocks is the compiled window on the slices bit for bit over
  consecutive windows of HSN-T with its per-byte memory and the carried reader: values, the
  final state, the next cache, every parameter gradient (and so it equals the eager window as
  before, which the compile tests keep checking);
* the trainer over several optimiser steps: the same records, gradients and weights.

**The prologue's per-position views** are not made for the compiled window, which never
read them.

**The chosen unit's gather** in the all-units pass through
:func:`hsn.model.units.pick_unit` (``as_gather``): the value and the gradient of the gather
it replaces, bit for bit.

**``hardware.message_gradients=state``** (opt-in): the messages of the sparse round add up the
gradients of their two gathers in the state's dtype. The messages are the same bit for bit
under bf16 autocast in the eager round; without autocast the whole computation is; under
autocast the two gradients are the fp32 sums of what the default adds up in bf16; the key's
place in the configuration and its refusals.

The tests marked ``cuda`` repeat the eager comparisons on the card (fp32 and bf16) and train
a few steps through the compiled window with CUDA Graphs with the key.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import (
    HASH_OMIT_AT_DEFAULT,
    REPO_ROOT,
    config_hash,
    load_config,
    message_gradients_scope_error,
)
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import memory_key
from hsn.model.graph import CellGraph
from hsn.model.hsn import WindowCarry, address_history
from hsn.model.hsn_t import HSNT
from hsn.model.units import UnitBank, pick_unit
from hsn.runner import build_model
from hsn.train import compile as compile_mod
from hsn.train import tbptt as tbptt_mod
from hsn.train.compile import _window_blocks, apply_message_gradients
from hsn.train.tbptt import TBPTTTrainer
from p1_15_support import datasets, hsn_config
from test_a6_feedback_detach import A5
from test_hsn_t import CUT_C, READER
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from test_word_address import _compiled_windows
from test_write_events import CATTN, NGRAM, STE
from torch import Tensor

CARRY = "model.reader_carry=5"
RECIPE = REPO_ROOT / "configs" / "experiments" / "hsn_t_memory_carry.yaml"
CARD = ("profile=rtx3060", "hardware.compile_mode=reduce-overhead")
STATE = "hardware.message_gradients=state"


def _slices(x: Tensor, blk: int) -> tuple[Tensor, ...]:
    """The block inputs as they were: a slice of the window per block."""
    return tuple(x[:, start : start + blk] for start in range(0, int(x.shape[1]), blk))


def _same_bits(a: Tensor, b: Tensor) -> bool:
    """Equal bit for bit (``torch.equal`` takes ``-0.0`` for ``+0.0``)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if not a.is_floating_point():
        return bool(torch.equal(a, b))
    width = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return bool(torch.equal(a.contiguous().view(width), b.contiguous().view(width)))


# --------------------------------------------------------------------------------------------------
# The blocks
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64, torch.bool])
def test_blocks_hold_the_slices_values_densely(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randint(0, 2, (3, 12, 5, 2), generator=generator).to(dtype)
    if dtype.is_floating_point:
        x = x + torch.randn(3, 12, 5, 2, generator=generator).to(dtype)
    for shape in ((3, 12, 5, 2), (3, 12, 10), (3, 12)):
        y = x.reshape(shape) if len(shape) > 2 else x[:, :, 0, 0]
        blocks = _window_blocks(y, 4)
        assert len(blocks) == 3
        for got, want in zip(blocks, _slices(y, 4), strict=True):
            assert got.is_contiguous() and got.shape == want.shape and got.dtype == dtype
            assert _same_bits(got, want.contiguous())
            # a dense tensor of its own shape: the strides of [B, blk, …], whatever T is
            assert got.stride() == torch.empty(got.shape).stride()


def test_one_block_is_the_tensor_itself() -> None:
    x = torch.randn(2, 8, 3, requires_grad=True)
    (block,) = _window_blocks(x, 8)
    assert block is x


def test_blocks_of_a_strided_window() -> None:
    """A window that is itself a view (other strides) gives the same dense blocks."""
    base = torch.randn(12, 3, 4)
    x = base.transpose(0, 1)  # [3, 12, 4], not contiguous
    for got, want in zip(_window_blocks(x, 4), _slices(x, 4), strict=True):
        assert got.is_contiguous() and _same_bits(got, want.contiguous())


@pytest.mark.parametrize("blk", [2, 4, 12])
def test_gradient_is_the_slices_gradient_bit_for_bit(blk: int) -> None:
    """Every block multiplied by its own weights, some of them ``-0.0`` and some of them
    negative on a zero: the gradient of the window holds ``-0.0`` in a block's gradient. The
    slices' sum over the blocks (the block's gradient plus exact zeros) turned it into
    ``+0.0`` when there was a sum; the blocks do the same."""
    generator = torch.Generator().manual_seed(blk)
    T = 12
    weights = torch.randn(2, T, 3, generator=generator)
    weights[0, ::3, 0] = -0.0
    weights[1, 1::4, 2] = 0.0

    def gradient(parts_of: object) -> Tensor:
        x = torch.randn(2, T, 3, generator=torch.Generator().manual_seed(1), requires_grad=True)
        parts = parts_of(x, blk)  # type: ignore[operator]
        total = x.new_zeros(())
        for index, part in enumerate(parts):
            cols = slice(index * blk, (index + 1) * blk)
            total = total + (part * weights[:, cols]).sum()
        (grad,) = torch.autograd.grad(total, x)
        return grad

    want, got = gradient(_slices), gradient(_window_blocks)
    assert _same_bits(got, want)
    negative_zero = (want == 0) & torch.signbit(want)
    if blk == T:  # one block: no sum, the gradient as the block gave it
        assert bool(negative_zero.any())
    else:
        assert not bool(negative_zero.any())
        assert bool(((weights == 0) & torch.signbit(weights)).any())  # the case was there


def test_gradient_with_another_reader_of_the_window() -> None:
    """The window's tensor is also read outside the blocks (the losses read ``z``): the two
    gradients add up as they did."""
    generator = torch.Generator().manual_seed(5)
    weights = torch.randn(2, 12, 3, generator=generator)
    other = torch.randn(2, 12, 3, generator=generator)

    def gradient(parts_of: object) -> Tensor:
        x = torch.randn(2, 12, 3, generator=torch.Generator().manual_seed(2), requires_grad=True)
        parts = parts_of(x, 4)  # type: ignore[operator]
        total = sum((p * weights[:, 4 * i : 4 * i + 4]).sum() for i, p in enumerate(parts))
        (grad,) = torch.autograd.grad(total + (x * other).sum(), x)
        return grad

    assert _same_bits(gradient(_window_blocks), gradient(_slices))


# --------------------------------------------------------------------------------------------------
# The compiled window
# --------------------------------------------------------------------------------------------------


def _carry_model(*extra: str) -> HSNT:
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
        CARRY,
        *extra,
    )
    seed_everything(0)
    model = HSNT.from_config(cfg)
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():  # as p28._model(cfg, training=True, scale=0.3)
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.3)
    model.train(True)
    return model


def _run_windows(
    backend: str, extra: tuple[str, ...], blocks: object | None, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[p28.Window, list[Tensor | None]]]:
    """Five consecutive compiled windows (20 bytes: several blocks) of HSN-T with its
    per-byte memory and the carried reader; each window's outputs and parameter gradients."""
    torch._dynamo.reset()
    if blocks is not None:
        monkeypatch.setattr(compile_mod, "_window_blocks", blocks)
    T = 20
    model = _carry_model(*extra)
    window = p28._window(model, p28._ctx(), backend)
    state = p28._inputs(model, b=3, t=T)[2]
    carry: Tensor | WindowCarry | None = None
    out = []
    for tokens, ctx, reset in _compiled_windows(T):
        got = p28._run(functools.partial(window, tokens, ctx, state, reset, prev_tokens=carry))
        out.append((got, p28._grads(model, got)))
        carry = address_history(model.cfg, tokens, carry, got[2])
        state = got[1].detach()
    return out


@pytest.mark.parametrize("extra", [(), (CATTN,)], ids=["per_byte", "per_byte_cread"])
@pytest.mark.parametrize("backend", ["aot_eager", p28.CPU_BACKEND])
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_on_blocks_is_the_window_on_slices(
    backend: str, extra: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    new = _run_windows(backend, extra, None, monkeypatch)
    old = _run_windows(backend, extra, _slices, monkeypatch)
    assert len(new) == len(old) == 5
    for i, ((got, grads), (ref, ref_grads)) in enumerate(zip(new, old, strict=True)):
        assert _same_bits(got[0], ref[0]), i
        for name in ("h", "q", "age", "pos"):
            assert _same_bits(getattr(got[1], name), getattr(ref[1], name)), (i, name)
        assert set(got[2]) == set(ref[2])
        for key, value in ref[2].items():
            assert _same_bits(got[2][key], value), (i, key)
        assert any(g is not None for g in ref_grads)
        for g, r in zip(grads, ref_grads, strict=True):
            assert (g is None) == (r is None)
            if g is not None and r is not None:
                assert _same_bits(g, r), i


@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_makes_no_per_position_views(monkeypatch: pytest.MonkeyPatch) -> None:
    """The eager loop reads the prologue's per-position views; the compiled window reads
    blocks and asks for none (they were made and dropped, 768 a window at T 256)."""
    model = _carry_model()
    seen: list[tuple[int, int, int]] = []
    prologue = type(model)._window_prologue

    def spy(self: HSNT, *args: object, **kwargs: object) -> object:
        pro = prologue(self, *args, **kwargs)  # type: ignore[arg-type]
        seen.append((len(pro.z_steps), len(pro.ctx_steps), len(pro.scalar_steps)))
        return pro

    monkeypatch.setattr(type(model), "_window_prologue", spy)
    T = 20
    tokens, ctx, reset = _compiled_windows(T)[0]
    state = p28._inputs(model, b=3, t=T)[2]
    model(tokens, ctx, state, reset)
    p28._window(model, p28._ctx(), "aot_eager")(tokens, ctx, state, reset)
    assert seen == [(T, T, T), (0, 0, 0)]


# --------------------------------------------------------------------------------------------------
# The trainer
# --------------------------------------------------------------------------------------------------


def _train(
    tmp_path: Path, blocks: object | None, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[tuple[float, float]], list[dict[str, Tensor]], dict[str, Tensor]]:
    torch._dynamo.reset()
    monkeypatch.setattr(compile_mod, "MAX_BLOCK", 4)  # the tiny trainer window: several blocks
    if blocks is not None:
        monkeypatch.setattr(compile_mod, "_window_blocks", blocks)
    monkeypatch.setattr(  # the CPU's compile backend of the trainer tests
        tbptt_mod,
        "compile_window",
        functools.partial(compile_mod.compile_window, backend="aot_eager"),
    )
    steps = 6
    cfg = hsn_config(
        steps,
        *A5,
        CUT_C,
        "model.cand_norm=rms",
        "model.k_select=3",
        "train.soft_until_step=0",
        STE,
        "train.lr=3e-3",
        *READER,
        *NGRAM,
        "model.address_gate=true",
        "model.address_gate_input=summary",
        CARRY,
        "hardware.compile_mode=default",
    )
    seed_everything(0)
    trainer = TBPTTTrainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path)
    assert trainer.compiled_window is not None
    raw: list[dict[str, Tensor]] = []

    def hook(tr: TBPTTTrainer) -> None:
        raw.append(
            {n: p.grad.clone() for n, p in tr.model.named_parameters() if p.grad is not None}
        )

    trainer.grad_hook = hook
    history = trainer.fit(*datasets(), steps)
    assert trainer.compiled_window.calls >= steps
    records = [(s.loss, s.grad_norm) for s in history.steps]
    weights = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    return records, raw, weights


def test_trainer_steps_are_the_same_bit_for_bit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Six optimiser steps of the small recipe (HSN-T, per-byte memory, carried reader, the
    candidate's cut, straight-through gates) on the compiled window: with the dense blocks
    and with the slices the step records, every raw gradient of every step and the weights
    after the last step are the same bits."""
    with monkeypatch.context() as patch:
        new = _train(tmp_path / "blocks", None, patch)
    with monkeypatch.context() as patch:
        old = _train(tmp_path / "slices", _slices, patch)
    assert new[0] == old[0] and len(new[0]) == 6
    assert len(new[1]) == len(old[1]) == 6
    for got, ref in zip(new[1], old[1], strict=True):
        assert got.keys() == ref.keys() and len(ref) > 20
        for name, value in ref.items():
            assert _same_bits(got[name], value), name
    for name, value in old[2].items():
        assert _same_bits(new[2][name], value), name


# --------------------------------------------------------------------------------------------------
# The chosen unit's gather, bit for bit
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_pick_unit_as_gather_has_the_gathers_gradient_bit_for_bit(dtype: torch.dtype) -> None:
    """The value is the gather's; the gradient is the gather's own (a zero tensor with each
    row added at its unit) bit for bit, where the incoming gradient holds ``-0.0`` too."""
    generator = torch.Generator().manual_seed(0)
    P, M, W = 37, 5, 6
    unit = torch.randint(0, M, (P,), generator=generator)
    weights = torch.randn(P, W, generator=generator).to(dtype)
    weights[::4, 0] = -0.0
    index = unit[:, None, None].expand(P, 1, W)

    def run(pick: object) -> tuple[Tensor, Tensor]:
        x = torch.randn(P, M, W, generator=torch.Generator().manual_seed(1)).to(dtype)
        x.requires_grad_(True)
        out = pick(x)  # type: ignore[operator]
        (grad,) = torch.autograd.grad((out * weights).sum(), x)
        return out.detach(), grad

    want, want_grad = run(lambda x: x.gather(1, index).squeeze(1))
    got, got_grad = run(lambda x: pick_unit(x, unit, as_gather=True))
    assert _same_bits(got, want) and _same_bits(got_grad, want_grad)
    assert not bool(((want_grad == 0) & torch.signbit(want_grad)).any())
    # the merged round's pick (as_gather=False, as it was) writes the row as it comes
    plain, plain_grad = run(lambda x: pick_unit(x, unit))
    assert _same_bits(plain, want) and torch.equal(plain_grad, want_grad)
    assert bool(((plain_grad == 0) & torch.signbit(plain_grad)).any())


def test_static_unit_pass_is_the_gather_of_the_all_units_pass() -> None:
    """:meth:`UnitBank.forward_static` against every unit's output with a gather: the value
    and every gradient (the input, the four stacked parameters) bit for bit."""
    torch.manual_seed(0)
    bank = UnitBank(8, 6, 10, "gelu", init="trunc_normal_0.02")
    generator = torch.Generator().manual_seed(3)
    P = 23
    unit = torch.randint(0, bank.n_units, (P,), generator=generator)
    weights = torch.randn(P, bank.d, generator=generator)
    v0 = torch.randn(P, bank.in_dim, generator=generator)

    def run(static: bool) -> list[Tensor]:
        v = v0.clone().requires_grad_(True)
        if static:
            out = bank.forward_static(v, unit)
        else:
            every = bank.forward_all(v.reshape(1, P, bank.in_dim))[0]
            out = every.gather(1, unit[:, None, None].expand(P, 1, bank.d)).squeeze(1)
        grads = torch.autograd.grad((out * weights).sum(), [v, *bank.parameters()])
        return [out.detach(), *grads]

    for got, want in zip(run(True), run(False), strict=True):
        assert _same_bits(got, want)


# --------------------------------------------------------------------------------------------------
# hardware.message_gradients=state (opt-in)
# --------------------------------------------------------------------------------------------------


def _graph_case(
    dtype: torch.dtype = torch.float32,
) -> tuple[CellGraph, torch.nn.Linear, Tensor, Tensor, Tensor]:
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(4)
    N, d, B, K = 12, 8, 6, 5
    graph = CellGraph(N, 3, 0).to(dtype)
    with torch.no_grad():
        graph.edge_logit.copy_(torch.randn(graph.edge_logit.shape, generator=generator))
    V = torch.nn.Linear(d, d, bias=False).to(dtype)
    h = (torch.rand(B, N, d, generator=generator) * 1.6 - 0.8).to(dtype)
    cells = torch.stack([torch.randperm(N, generator=generator)[:K] for _ in range(B)])
    weights = torch.randn(B, K, d, generator=generator).to(dtype)
    return graph, V, h, cells, weights


def _messages(
    graph: CellGraph, V: torch.nn.Linear, h: Tensor, cells: Tensor, weights: Tensor, *, bf16: bool
) -> tuple[Tensor, list[Tensor]]:
    """The messages of the selected cells and the gradients of a fixed functional of them
    with respect to the state, ``V`` and the edge logits."""
    h = h.clone().requires_grad_(True)
    with torch.autocast(h.device.type, dtype=torch.bfloat16, enabled=bf16):
        m = graph.messages_selected(h, V, cells, validate=False)
        loss = (m.float() * weights.float()).sum()
    grads = torch.autograd.grad(loss, [h, V.weight, graph.edge_logit])
    return m.detach(), list(grads)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_state_gradients_without_autocast_are_the_same_computation(dtype: torch.dtype) -> None:
    graph, V, h, cells, weights = _graph_case(dtype)
    want, want_grads = _messages(graph, V, h, cells, weights, bf16=False)
    graph.state_gradients = True
    got, grads = _messages(graph, V, h, cells, weights, bf16=False)
    assert got.dtype == dtype and _same_bits(got, want)
    for g, w in zip(grads, want_grads, strict=True):
        assert _same_bits(g, w)


def test_state_gradients_under_autocast_keep_the_messages_and_sum_in_fp32() -> None:
    """bf16 autocast: the messages are the same bits. The gradients of the state and of
    ``V`` and the edge logits differ from the default's by its bf16 sums only: they agree
    with it to bf16 rounding, and they are at least as close to the gradients of the same
    messages computed in float64 (the same bf16 forward values held in float64, no rounding
    in the backward sums) as the default's are."""
    graph, V, h, cells, weights = _graph_case()
    want, want_grads = _messages(graph, V, h, cells, weights, bf16=True)
    graph.state_gradients = True
    got, grads = _messages(graph, V, h, cells, weights, bf16=True)
    assert got.dtype == torch.bfloat16 and _same_bits(got, want)
    changed = 0
    for g, w in zip(grads, want_grads, strict=True):
        scale = float(w.abs().max())
        assert float((g - w).abs().max()) <= 0.05 * scale  # bf16: 8 bits, a few sums deep
        changed += int(not torch.equal(g, w))
    assert changed > 0  # the case does hold rows gathered more than once
    # the gradient of the edge weights: the default adds each selected row's gradient in bf16,
    # the key in fp32. Against the exact sum of the same bf16 addends:
    graph.state_gradients = False
    padded = graph._padded_weights().detach()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        Vh = V(h).detach()
    D, B, K, d = graph.max_in_degree, *cells.shape, h.shape[-1]
    source_idx = graph._pad_src[cells].reshape(B, K * D, 1).expand(B, K * D, d)
    sources = Vh.gather(1, source_idx).reshape(B, K, D, d)
    # d loss / d weights_sel [B, K, D], each addend as the bf16 backward forms it
    addends = (sources * weights.to(torch.bfloat16)[:, :, None, :]).sum(dim=3)
    exact = torch.zeros(padded.shape, dtype=torch.float64).index_put(
        (cells,), addends.double(), accumulate=True
    )

    def edge_weight_grad(state: bool) -> Tensor:
        graph.state_gradients = state
        own = graph._padded_weights().detach().requires_grad_(True)
        hh = h.clone()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            m = graph.messages_selected(hh, V, cells, weights=own, validate=False)
            loss = (m.float() * weights.float()).sum()
        (grad,) = torch.autograd.grad(loss, own)
        return grad.double()

    error_default = float((edge_weight_grad(False) - exact).abs().max())
    error_state = float((edge_weight_grad(True) - exact).abs().max())
    assert error_state <= error_default
    assert error_state <= 1e-5 * float(exact.abs().max())  # fp32 sums of the same addends


def test_message_gradients_key_is_off_by_default_and_left_out_of_the_hash() -> None:
    base = load_config(RECIPE, list(CARD))
    assert base.hardware.message_gradients == "autocast"
    assert "hardware.message_gradients" in HASH_OMIT_AT_DEFAULT
    off = load_config(RECIPE, [*CARD, "hardware.message_gradients=autocast"])
    assert config_hash(off) == config_hash(base)
    assert "message_gradients" not in base.hardware.model_dump(exclude_defaults=True)
    on = load_config(RECIPE, [*CARD, STATE])
    assert on.hardware.message_gradients == "state"
    assert message_gradients_scope_error(on) is None
    assert config_hash(on) != config_hash(base)
    assert on.model == base.model and on.train == base.train
    protocol = {"name": "train_eval", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
    assert memory_key(off, "pdg", protocol) == memory_key(base, "pdg", protocol)
    assert memory_key(on, "pdg", protocol) != memory_key(base, "pdg", protocol)
    # with the merged products, on the CPU and eager: the key is the round's, not the card's
    both = load_config(RECIPE, [*CARD, STATE, "hardware.round_products=merged"])
    assert both.hardware.round_products == "merged"
    assert load_config(RECIPE, [STATE]).hardware.compile_mode == "none"
    words = REPO_ROOT / "configs" / "experiments" / "hsn_t_memory_words_carry.yaml"
    assert load_config(words, [*CARD, STATE]).hardware.message_gradients == "state"


@pytest.mark.parametrize(
    ("path", "overrides", "reason"),
    [
        (p28.REF_A1, (STATE,), "model.routing=dense"),
        (RECIPE, (*CARD, "hardware.message_gradients=fp32"), "'autocast' or 'state'"),
        (
            REPO_ROOT / "configs" / "experiments" / "hsn_t_nomem.yaml",
            (*CARD, STATE),
            "has no sparse round",
        ),
        (
            REPO_ROOT / "configs" / "experiments" / "hsn_recipe_a6.yaml",
            (*CARD, "hardware.window=fused", STATE),
            "hardware.window=fused",
        ),
    ],
)
def test_message_gradients_key_is_refused_outside_its_scope(
    path: Path, overrides: tuple[str, ...], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        load_config(path, list(overrides))


def test_trainer_sets_the_graphs_gradient_dtype(tmp_path: Path) -> None:
    def graph_flag(*extra: str) -> bool:
        cfg = hsn_config(4, *A5, STE, "train.soft_until_step=0", *extra)
        seed_everything(0)
        model = build_model(cfg)
        trainer = TBPTTTrainer(model, cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
        return bool(trainer.hsn.cell.graph.state_gradients)

    assert graph_flag() is False
    assert graph_flag(STATE) is True
    model = _carry_model()
    apply_message_gradients(model, "state")
    assert model.cell.graph.state_gradients is True
    apply_message_gradients(model, "autocast")
    assert model.cell.graph.state_gradients is False
    with pytest.raises(ValueError, match="autocast or state"):
        apply_message_gradients(model, "fp32")


@pytest.mark.parametrize("backend", ["aot_eager", p28.CPU_BACKEND])
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_with_state_gradients_without_autocast(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fp32, no autocast: the compiled window with the key is the compiled window without
    it bit for bit (the casts are no-ops: the same graph)."""
    old = _run_windows(backend, (), None, monkeypatch)
    torch._dynamo.reset()
    model = _carry_model()
    apply_message_gradients(model, "state")
    window = p28._window(model, p28._ctx(), backend)
    T = 20
    state = p28._inputs(model, b=3, t=T)[2]
    carry: Tensor | WindowCarry | None = None
    for i, (tokens, ctx, reset) in enumerate(_compiled_windows(T)):
        got = p28._run(functools.partial(window, tokens, ctx, state, reset, prev_tokens=carry))
        ref, ref_grads = old[i]
        assert _same_bits(got[0], ref[0]) and _same_bits(got[1].h, ref[1].h)
        for g, r in zip(p28._grads(model, got), ref_grads, strict=True):
            assert (g is None) == (r is None)
            if g is not None and r is not None:
                assert _same_bits(g, r), i
        carry = address_history(model.cfg, tokens, carry, got[2])
        state = got[1].detach()


# --------------------------------------------------------------------------------------------------
# On the card
# --------------------------------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_blocks_and_pick_on_the_card(dtype: torch.dtype) -> None:
    """The CUDA kernels of the eager ops: the blocks against the slices (values, layout, the
    gradient with a ``-0.0`` in it) and ``pick_unit(as_gather=True)`` against ``gather``
    (value and gradient), bit for bit, in fp32 and bf16."""
    device = p28._cuda_ctx().device
    generator = torch.Generator().manual_seed(0)
    T, blk = 12, 4
    weights = torch.randn(2, T, 3, generator=generator)
    weights[0, ::3, 0] = -0.0
    weights = weights.to(device, dtype)
    x0 = torch.randn(2, T, 3, generator=generator).to(device, dtype)
    for got, want in zip(_window_blocks(x0, blk), _slices(x0, blk), strict=True):
        assert got.is_contiguous() and _same_bits(got, want.contiguous())

    def gradient(parts_of: object) -> Tensor:
        x = x0.clone().requires_grad_(True)
        parts = parts_of(x, blk)  # type: ignore[operator]
        total = x.new_zeros(())
        for index, part in enumerate(parts):
            total = total + (part * weights[:, index * blk : (index + 1) * blk]).sum()
        (grad,) = torch.autograd.grad(total, x)
        return grad

    assert _same_bits(gradient(_window_blocks), gradient(_slices))
    P, M, W = 37, 5, 6
    unit = torch.randint(0, M, (P,), generator=generator).to(device)
    pick_weights = torch.randn(P, W, generator=generator)
    pick_weights[::4, 0] = -0.0
    pick_weights = pick_weights.to(device, dtype)
    x1 = torch.randn(P, M, W, generator=generator).to(device, dtype)
    index = unit[:, None, None].expand(P, 1, W)

    def run(pick: object) -> tuple[Tensor, Tensor]:
        x = x1.clone().requires_grad_(True)
        out = pick(x)  # type: ignore[operator]
        (grad,) = torch.autograd.grad((out * pick_weights).sum(), x)
        return out.detach(), grad

    want_out, want_grad = run(lambda x: x.gather(1, index).squeeze(1))
    got_out, got_grad = run(lambda x: pick_unit(x, unit, as_gather=True))
    assert _same_bits(got_out, want_out) and _same_bits(got_grad, want_grad)


@pytest.mark.cuda
def test_state_gradients_on_the_card_keep_the_eager_messages() -> None:
    """bf16 autocast on the card, the eager round: the messages are the same bits with the
    key; the three gradients agree with the default's to bf16 rounding."""
    device = p28._cuda_ctx().device
    graph, V, h, cells, weights = _graph_case()
    graph, V = graph.to(device), V.to(device)
    h, cells, weights = h.to(device), cells.to(device), weights.to(device)
    want, want_grads = _messages(graph, V, h, cells, weights, bf16=True)
    graph.state_gradients = True
    got, grads = _messages(graph, V, h, cells, weights, bf16=True)
    assert got.dtype == torch.bfloat16 and _same_bits(got, want)
    for g, w in zip(grads, want_grads, strict=True):
        assert bool(torch.isfinite(g).all())
        assert float((g - w).abs().max()) <= 0.05 * float(w.abs().max())


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("extra", [(STATE,), (STATE, "hardware.round_products=merged")])
def test_compiled_training_on_the_card_with_the_key(extra: tuple[str, ...], tmp_path: Path) -> None:
    """Twelve steps of the small recipe (HSN-T, per-byte memory, carried reader) under bf16
    autocast through the compiled window with CUDA Graphs, on the dense blocks, with
    ``hardware.message_gradients=state`` alone and beside the merged products: finite
    losses and the graph flag set. (Without autocast the key changes nothing: the CPU tests.
    The step time and the losses of the full configuration are in
    ``docs/perbyte_speed.md``.)"""
    from cards import CUDA_PROFILE

    p28._cuda_ctx()
    steps = 12
    precision = "bf16"
    cfg = hsn_config(
        steps,
        *A5,
        CUT_C,
        "model.cand_norm=rms",
        "model.k_select=3",
        "train.soft_until_step=0",
        STE,
        *READER,
        *NGRAM,
        CARRY,
        f"profile={CUDA_PROFILE}",
        "hardware.compile_mode=reduce-overhead",
        f"hardware.autocast_dtype={precision}",
        f"train.precision={precision}",
        *extra,
    )
    seed_everything(0)
    trainer = TBPTTTrainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    assert trainer.hsn.cell.graph.state_gradients is True
    history = trainer.fit(*datasets(), steps)
    assert len(history.steps) == steps
    assert all(torch.isfinite(torch.tensor(s.loss)) for s in history.steps)
    assert trainer.compiled_window is not None and trainer.compiled_window.calls >= steps
