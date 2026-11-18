"""The per-byte loop's block inputs (``docs/perbyte_speed.md``, 18 Nov 2026).

The compiled autograd window hands every block of the loop over bytes its inputs (the encoded
bytes, the context with the reader's output, the scorer's scalars, the resets) as dense
tensors (:func:`hsn.train.compile._window_blocks`): one copy of the window, laid out block by
block, instead of a slice of the window per block. It is a change of layout only. These tests
check that nothing else changed:

* the blocks hold the slices' values, are dense, and one block is the tensor itself;
* the gradient that reaches the window's tensor is the slices' bit for bit, a ``-0.0`` in a
  block's gradient included (the slices' sum turned it into ``+0.0``; so does the hook);
* the compiled window on the blocks is the compiled window on the slices bit for bit over
  consecutive windows of HSN-T with its per-byte memory and the carried reader: values, the
  final state, the next cache, every parameter gradient (and so it equals the eager window as
  before, which the compile tests keep checking);
* the trainer over several optimiser steps: the same records, gradients and weights.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.device import resolve, seed_everything
from hsn.model.hsn import WindowCarry, address_history
from hsn.model.hsn_t import HSNT
from hsn.runner import build_model
from hsn.train import compile as compile_mod
from hsn.train import tbptt as tbptt_mod
from hsn.train.compile import _window_blocks
from hsn.train.tbptt import TBPTTTrainer
from p1_15_support import datasets, hsn_config
from test_a6_feedback_detach import A5
from test_hsn_t import CUT_C, READER
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from test_word_address import _compiled_windows
from test_write_events import CATTN, NGRAM, STE
from torch import Tensor

CARRY = "model.reader_carry=5"


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
