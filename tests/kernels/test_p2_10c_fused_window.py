"""P2.10c: the fused window with the state gradient, end to end on the card
(``hardware.window=fused`` with ``train.feedback_detach=[summary, eps, router]``): the compiled
forward sweep, the batched replay and the reverse sweep as kernels in a CUDA graph.

The comparisons are on one trajectory. The window's parameter gradients through the kernels'
sweep against the same window through the PyTorch sweep (one tape, one replay: only the
sweep's implementation differs), in fp32 and under bf16 autocast; and, in fp32 on an initial
model whose decisions have margins, against the compiled autograd window. CUDA-marked; skipped
without a card or the built artifacts unless ``HSN_REQUIRE_CUDA=1``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from hsn.config import RunConfig, load_config
from hsn.device import resolve, seed_everything
from hsn.kernels import window
from hsn.model.hsn import HSN
from hsn.train import compile as compile_mod
from hsn.train import state_gradient
from hsn.train.amp import autocast_ctx
from hsn.train.compile import CompiledWindow, compile_window
from hsn.train.split import SplitWindow
from torch import Tensor

pytestmark = pytest.mark.cuda

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/experiments/hsn_recipe_a6.yaml"
CUT = "train.feedback_detach=[summary,eps,router]"
SHAPES = {
    "candidate_routed": ("model.n_cells=64", "model.k_select=12"),
    "approved_shape": (
        "model.n_cells=64", "model.k_select=20", "model.fast_cells=8", "model.fast_units=fixed",
        "model.fast_readout=true",
    ),
    "approved_shape_ngram2": (
        "model.n_cells=64", "model.k_select=20", "model.fast_cells=8", "model.fast_units=fixed",
        "model.fast_readout=true", "model.address_ngram=2",
    ),
}  # fmt: skip


@pytest.fixture(scope="module")
def device() -> torch.device:
    required = os.environ.get("HSN_REQUIRE_CUDA") == "1"
    if not torch.cuda.is_available():
        if required:
            pytest.fail("HSN_REQUIRE_CUDA=1 but no CUDA device is available")
        pytest.skip("no CUDA device")
    if not window.available():
        if required:
            pytest.fail("HSN_REQUIRE_CUDA=1 but the window kernels are not built")
        pytest.skip("window kernels not built (scripts/build_window_kernels.py)")
    window.prepare(0)
    return torch.device("cuda", 0)


@pytest.fixture
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compile_mod, "MAX_BLOCK", 4)


@pytest.fixture(autouse=True)
def _release_compiled_state() -> object:
    yield
    state_gradient.set_fast_sweep(None)
    torch._dynamo.reset()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        import gc

        from torch._inductor.cudagraph_trees import reset_cudagraph_trees

        reset_cudagraph_trees()
        gc.collect()
        torch.cuda.empty_cache()


def _case(
    precision: str, *overrides: str, scale: float = 0.05
) -> tuple[RunConfig, HSN, CompiledWindow, object]:
    cfg = load_config(
        RECIPE,
        overrides=[
            "profile=rtx3060",
            "hardware.compile_mode=reduce-overhead",
            "hardware.window=fused",
            f"train.precision={precision}",
            f"hardware.autocast_dtype={precision}",
            "train.unroll=16",
            CUT,
            "train.unit_credit=executed",
            "train.soft_until_step=0",
            *overrides,
        ],
    )
    ctx = resolve(cfg.hardware)
    seed_everything(0)
    model = HSN.from_config(cfg).to(ctx.device)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g).to(ctx.device))
    model.train()
    compiled = compile_window(model, ctx, "reduce-overhead", backend="inductor")
    assert isinstance(compiled, CompiledWindow)
    return cfg, model, compiled, ctx


def _inputs(cfg: RunConfig, device: torch.device, B: int = 6, T: int = 16) -> tuple[Tensor, ...]:
    g = torch.Generator().manual_seed(11)
    reset = torch.rand(B, T, generator=g) < 0.05
    reset[:, 0] = True
    reset[1, 3] = reset[2, 4] = reset[3, 6] = reset[4, T - 1] = True
    tokens = torch.randint(0, cfg.model.vocab_size, (B, T), generator=g)
    context = torch.randint(0, cfg.model.n_context, (B, T), generator=g)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 1::3] = True
    return tuple(x.to(device) for x in (tokens, context, reset, mask))


def _loss(logits: Tensor, info: dict[str, Tensor], mask: Tensor) -> Tensor:
    task = torch.nn.functional.cross_entropy(logits[mask].float(), mask.nonzero()[:, 1] % 7)
    return task + (info["z_hat"].float() - info["z_target"].float()).square().mean()


def _grads(
    model: HSN, call: object, ctx: object, inputs: tuple[Tensor, ...], rounds: int = 3
) -> tuple[list[Tensor | None], dict[str, Tensor]]:
    """A window's parameter gradients (the last of ``rounds`` calls: a compiled window warms
    up, records and replays) and its routing decisions."""
    tokens, context, reset, mask = inputs
    params = list(model.parameters())
    for _ in range(rounds):
        state = model.init_state(int(tokens.shape[0]))
        with autocast_ctx(ctx):  # type: ignore[arg-type]
            logits, _, info = call(tokens, context, state, reset)  # type: ignore[operator]
        grads = torch.autograd.grad(_loss(logits, info, mask), params, allow_unused=True)
    slots = {k: info[k].clone() for k in ("cell_idx", "unit_idx", "valid")}
    return [None if g is None else g.detach().clone() for g in grads], slots


def _compare(got: list[Tensor | None], want: list[Tensor | None]) -> tuple[float, float]:
    """``(relative L2 distance, cosine)`` of two gradients over all parameters."""
    assert [g is None for g in got] == [g is None for g in want]
    x = torch.cat([g.double().flatten() for g in got if g is not None])
    y = torch.cat([g.double().flatten() for g in want if g is not None])
    return float((x - y).norm() / y.norm()), float(torch.dot(x, y) / (x.norm() * y.norm()))


@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("name", list(SHAPES))
def test_fused_window_through_the_kernels_sweep_and_through_the_reference_sweep(
    device: torch.device, name: str, precision: str
) -> None:
    """One window, one tape, one replay: the reverse sweep as kernels in a CUDA graph against
    the PyTorch sweep. The decisions are the same slots by construction, and the parameter
    gradients agree to the rounding of the sweep's own arithmetic."""
    cfg, model, compiled, ctx = _case(precision, *SHAPES[name])
    inputs = _inputs(cfg, device)
    split = SplitWindow(model, cfg, compiled)
    assert split.state_gradient and split.fused and split.sweep_runner is not None
    got, slots = _grads(model, split, ctx, inputs)
    assert split.sweep_runner.calls == 3 and split.sweep_runner.captures == 1
    state_gradient.set_fast_sweep(None)  # the PyTorch sweep on the same window
    want, slots_ref = _grads(model, split, ctx, inputs, rounds=1)
    assert split.sweep_runner.calls == 3
    for key in slots:
        assert torch.equal(slots[key], slots_ref[key]), key
    distance, cosine = _compare(got, want)
    print(f"{name} {precision}: the kernels' sweep against the PyTorch sweep")
    print(f"    gradients: relative distance {distance:.2e}, cosine {cosine:.8f}")
    limit = 2e-4 if precision == "fp32" else 5e-2
    assert distance <= limit and cosine >= 1.0 - limit


@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("name", ["candidate_routed", "approved_shape"])
def test_fused_window_against_the_compiled_autograd_window_in_fp32(
    device: torch.device, name: str
) -> None:
    """fp32, an initial model (the recipe's initialisation, moved a little): where the fused
    window and the compiled autograd window take the same routing decisions, their parameter
    gradients agree to the tolerance of two compilations of one arithmetic. (On a trained
    model whose forward pass amplifies rounding the two cannot be held equal end to end:
    ``docs/p2_10c_state_gradient_window.md``.)"""
    cfg, model, compiled, ctx = _case("fp32", *SHAPES[name], scale=0.02)
    inputs = _inputs(cfg, device)
    want, slots_ref = _grads(model, compiled, ctx, inputs)
    split = SplitWindow(model, cfg, compiled)
    got, slots = _grads(model, split, ctx, inputs)
    same = float((slots["cell_idx"] == slots_ref["cell_idx"]).float().mean())
    distance, cosine = _compare(got, want)
    print(f"{name}: fused against autograd, fp32: same cells {same:.4f}")
    print(f"    gradients: relative distance {distance:.2e}, cosine {cosine:.8f}")
    assert same == 1.0, "the two windows took different decisions: no end-to-end comparison"
    assert distance <= 1e-3 and cosine >= 0.999999
