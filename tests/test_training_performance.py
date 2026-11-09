"""Regressions for wasted dense-fallback work and diagnostic state-history allocation."""

from copy import deepcopy
from typing import Any

import pytest
import torch
from hsn.config import REPO_ROOT, load_config
from hsn.device import resolve
from hsn.model.hsn import HSN
from hsn.train import compile as compile_mod


def _model() -> HSN:
    cfg = load_config(
        REPO_ROOT / "configs/tiny_gradcheck.yaml",
        ["model.routing=topk", "train.gate_mode=ste", "train.precision=fp32"],
    )
    torch.manual_seed(9)
    return HSN.from_config(cfg)


def _inputs(model: HSN, batch: int = 8) -> tuple[Any, ...]:
    generator = torch.Generator().manual_seed(15)
    tokens = torch.randint(model.cfg.vocab_size, (batch, 4), generator=generator)
    ctx = torch.randint(model.cfg.n_context, tokens.shape, generator=generator)
    reset = torch.zeros_like(tokens, dtype=torch.bool)
    reset[:, 0] = True
    return tokens, ctx, model.init_state(batch), reset


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("count", [1, 3, 5, 8])
def test_fallback_executes_only_its_row_bucket(
    monkeypatch: pytest.MonkeyPatch, static: bool, count: int
) -> None:
    """Compare full-batch fallback values/gradients while observing dense kernel batch size.

    The static path uses the real compiled-window/block code without tracing so the kernel
    observer can run. Actual Dynamo forward/backward and recompile parity live in P2.8 tests.
    """
    model = _model()
    reference = deepcopy(model)
    inputs = _inputs(model)
    rows = torch.arange(8) < count
    monkeypatch.setattr(model, "_draw_dense_rows", lambda *args: rows)
    monkeypatch.setattr(reference, "_draw_dense_rows", lambda *args: rows)
    monkeypatch.setattr(reference, "_dense_indices", lambda *args, **kwargs: torch.arange(8))
    seen: list[int] = []
    dense = model._cell_dense

    def observe(u: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        seen.append(u.shape[0])
        return dense(u, *args, **kwargs)

    monkeypatch.setattr(model, "_cell_dense", observe)
    if static:
        monkeypatch.setattr(compile_mod, "_compiler", lambda *args: lambda fn: fn)
        ctx = resolve(load_config(REPO_ROOT / "configs/tiny_gradcheck.yaml").hardware)
        run = compile_mod.CompiledWindow(model, ctx, "reduce-overhead", "aot_eager")
        ref = compile_mod.CompiledWindow(reference, ctx, "reduce-overhead", "aot_eager")
    else:
        run, ref = model, reference
    out, state, info = run(*inputs)
    expected, expected_state, expected_info = ref(*inputs)
    bucket = min(8, 1 << (count - 1).bit_length()) if static else count
    assert seen == [bucket] * inputs[0].shape[1]
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state.h, expected_state.h, rtol=1e-5, atol=1e-6)
    assert torch.equal(state.age, expected_state.age)
    assert torch.equal(info["dense_fallback"], expected_info["dense_fallback"])
    # Distinct padding IDs never overwrite sparse rows or double their gradients.
    assert torch.equal(out[~rows], expected[~rows])
    (out.square().mean() + state.h.square().mean()).backward()
    (expected.square().mean() + expected_state.h.square().mean()).backward()
    for (name, param), (_, original) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert (param.grad is None) == (original.grad is None), name
        if param.grad is not None:
            torch.testing.assert_close(param.grad, original.grad, rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.parametrize("static", [False, True])
def test_window_statistics_do_not_materialize_state_history(
    monkeypatch: pytest.MonkeyPatch, static: bool
) -> None:
    model = _model().eval()
    inputs = _inputs(model)
    capture = model(*inputs, capture=True)
    seen: list[tuple[Any, torch.Size]] = []
    epilogue = model._window_epilogue

    def observe(*args: Any, **kwargs: Any) -> Any:
        seen.append((kwargs["hs"], kwargs["state_abs_max"].shape))
        return epilogue(*args, **kwargs)

    monkeypatch.setattr(model, "_window_epilogue", observe)
    if static:
        monkeypatch.setattr(compile_mod, "_compiler", lambda *args: lambda fn: fn)
        ctx = resolve(load_config(REPO_ROOT / "configs/tiny_gradcheck.yaml").hardware)
        run = compile_mod.CompiledWindow(model, ctx, "reduce-overhead", "aot_eager")
    else:
        run = model
    out, state, info = run(*inputs)
    assert seen == [(None, torch.Size([4]))]
    assert "cell_states" not in info
    torch.testing.assert_close(out, capture[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state.h, capture[1].h, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        info["state_abs_max"],
        capture[2]["cell_states"].abs().amax(dim=(0, 2, 3)),
        rtol=1e-5,
        atol=1e-6,
    )
