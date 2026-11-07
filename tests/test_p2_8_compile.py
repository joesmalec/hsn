"""P2.8 tests: the L1 launch-overhead layer (:mod:`hsn.train.compile`).

The compiled training window and the serving step run the eager code (``HSN._position``) as
compiled blocks of positions: they must equal eager to rounding (by dtype, from measurement), keep
the P1.8 invariants and the exact dense-cell symmetry, never recompile once warm, and (CUDA) cut
the host launches at least in half. The static-K selection and the capturable sparse round are
checked against the L0 round.

Backends on the CPU: inductor (what ``hardware.compile_mode`` means) on Linux; ``aot_eager``
(dynamo and AOTAutograd, eager kernels) for the tests that check tracing, recompiles and the
trainer, and wherever inductor's C++ toolchain is not available (the hosted macOS runner).
"""

from __future__ import annotations

import functools
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import RunConfig, load_config, load_hardware_profile
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example, TaskSpec, generate, load_dataset_spec
from hsn.device import DeviceContext, resolve, seed_everything
from hsn.kernels import native
from hsn.model.hsn import HSN
from hsn.model.router import Router
from hsn.model.state import CellGraphState
from hsn.profiler.launches import HOST_LAUNCH_CALLS
from hsn.train import compile as compile_mod
from hsn.train import tbptt
from hsn.train.compile import (
    CompiledWindow,
    ServingStep,
    block_size,
    check_compile_scope,
    compile_step,
    compile_window,
)
from hsn.train.tbptt import TBPTTTrainer
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
#: ref_a1 shrunk (the P2.2 test recipe): N 6, M 3, K 2.
SMALL = (
    "model.d=8",
    "model.context_dim=8",
    "model.n_cells=6",
    "model.in_degree=2",
    "model.n_units=3",
    "model.k_select=2",
    "model.k_min=1",
    "model.hidden=16",
    "model.scorer_hidden=8",
    "train.unroll=16",
)
#: inductor's CPU backend needs a C++ compiler with OpenMP (Linux runners have it).
CPU_BACKEND = "inductor" if sys.platform.startswith("linux") else "aot_eager"
B, T = 3, 8
#: How many seeds of the CPU generator to search for a dense-fallback draw that picks some row,
#: and one that picks none (at p_dense 0.5 and B = 3 a draw picks no row with probability 1/8, so
#: 64 seeds hold both kinds many times over; the test asserts it found them).
SEED_SEARCH = 64

#: Compiled vs eager in fp32: 10x the largest difference measured (the efficiency audit's rule),
#: rounded up. 7 Nov 2026, every case of MODES plus top-K STE with the dense fallback, three seeds
#: each: CPU (inductor) values (logits, state, info) 7.2e-7, parameter gradients 1.14e-5; CUDA
#: (inductor, CUDA Graphs) values 1.45e-6, gradients 1.9e-5.
TOL = {
    ("cpu", "fp32"): {"values": 8e-6, "grads": 1.2e-4},
    ("cuda", "fp32"): {"values": 1.5e-5, "grads": 2e-4},
}

MODES: dict[str, tuple[tuple[str, ...], bool]] = {
    # name: (overrides, training)
    "dense": ((), True),
    "dense_eval": ((), False),
    "topk_ste": (("model.routing=topk", "train.gate_mode=ste"), True),
    "topk_soft": (("model.routing=topk",), True),  # the soft warm-up: the masked round
    "topk_eval": (("model.routing=topk",), False),
}


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _cfg(*overrides: str, small: bool = True) -> RunConfig:
    return load_config(REF_A1, overrides=[*(SMALL if small else ()), *overrides])


def _model(cfg: RunConfig, *, training: bool, scale: float = 0.1, seed: int = 0) -> HSN:
    seed_everything(seed)
    model = HSN.from_config(cfg)
    generator = torch.Generator(device="cpu").manual_seed(seed + 7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * scale)
    model.train(training)
    return model


def _ctx(profile: str = "cpu") -> DeviceContext:
    return resolve(load_hardware_profile(profile))


def _inputs(
    model: HSN, seed: int = 3, b: int = B, t: int = T
) -> tuple[Tensor, Tensor, CellGraphState, Tensor]:
    """Tokens, contexts, a random start state with ages and positions, and resets at ``t = 0``
    (row 0) and mid-window (row 1), on the model's device."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    m, dev = model.cfg, model.device
    tokens = torch.randint(0, m.vocab_size, (b, t), generator=generator).to(dev)
    ctx = torch.randint(0, m.n_context, (b, t), generator=generator).to(dev)
    h = (torch.rand(b, m.n_cells, m.d, generator=generator) * 1.6 - 0.8).to(dev, model.dtype)
    state = model.init_state(b).with_dense(h)
    for _ in range(3):
        state = state.tick((torch.rand(b, m.n_cells, generator=generator) < 0.5).to(dev))
    reset = torch.zeros(b, t, dtype=torch.bool)
    reset[0, 0] = True
    if b > 1:
        reset[1, min(3, t - 1)] = True
    return tokens, ctx, state, reset.to(dev)


Window = tuple[Tensor, CellGraphState, dict[str, Tensor]]


def _loss(window: Window) -> Tensor:
    """A fixed functional of every differentiable output of a window."""
    logits, final, info = window
    total = logits.float().square().mean() + final.h.float().sum() + final.q.float().sum()
    total = total + info["eps"].float().square().mean() + info["z_hat"].float().mean()
    if "unit_logits" in info:
        total = total + info["unit_logits"].float().square().mean()
    return total


def _grads(model: HSN, window: Window) -> list[Tensor | None]:
    params = [p for p in model.parameters() if p.requires_grad]
    if not torch.is_grad_enabled() or not _loss(window).requires_grad:
        return [None for _ in params]
    return list(torch.autograd.grad(_loss(window), params, allow_unused=True))


def _max_diff(window: Window, ref: Window) -> dict[str, float]:
    """The largest absolute differences of the values (logits, state, float info) and the
    integer/bool outputs' agreement (0 when equal)."""
    (logits, final, info), (logits_r, final_r, info_r) = window, ref
    assert set(info) == set(info_r), sorted(set(info) ^ set(info_r))
    values = [(logits - logits_r).abs().max()]
    exact = 0.0
    for name in ("h", "q"):
        values.append((getattr(final, name) - getattr(final_r, name)).abs().max())
    for name in ("age", "pos"):
        exact += float(not torch.equal(getattr(final, name), getattr(final_r, name)))
    for key, value in info_r.items():
        if value.is_floating_point():
            values.append((info[key].float() - value.float()).abs().max())
        else:
            exact += float(not torch.equal(info[key], value))
    return {"values": float(torch.stack(values).max()), "exact": exact}


def _grad_diff(got: list[Tensor | None], ref: list[Tensor | None]) -> float:
    worst = 0.0
    for g, r in zip(got, ref, strict=True):
        if g is None or r is None:
            other = g if g is not None else r
            worst = max(worst, 0.0 if other is None else float(other.abs().max()))
        else:
            worst = max(worst, float((g - r).abs().max()))
    return worst


def _window(model: HSN, ctx: DeviceContext, backend: str = CPU_BACKEND) -> CompiledWindow:
    window = compile_window(model, ctx, "reduce-overhead", backend=backend)
    assert isinstance(window, CompiledWindow)
    return window


@pytest.fixture
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocks of at most 4 positions on the CPU: an 8-position window runs two blocks (the block
    boundary is covered) and each compile stays short."""
    monkeypatch.setattr(compile_mod, "MAX_BLOCK", 4)


#: The smallest top-K score margin (the K-th minus the (K+1)-th cell score) of a routed
#: comparison: far above the compiled-vs-eager rounding (~1e-7), so the two take the same
#: discrete decisions. Rounding can flip a near tie (a margin at the rounding level), which is
#: not a defect; routed comparisons start from distinct cell states (no resets: a reset makes
#: every cell identical, and the cells' later near ties are rounding-sensitive) and check it.
MIN_MARGIN = 1e-4


def _margin(info: dict[str, Tensor], k: int) -> float:
    scores = info["unit_logits"].float().amax(dim=-1).sort(dim=-1, descending=True).values
    return float((scores[..., k - 1] - scores[..., k]).min())


def _case_inputs(model: HSN, routed: bool) -> tuple[Tensor, Tensor, CellGraphState, Tensor]:
    tokens, cx, state, reset = _inputs(model)
    if routed:
        reset = torch.zeros_like(reset)
    return tokens, cx, state, reset


def _run(fn: Callable[[], Window]) -> Window:
    torch.manual_seed(1234)  # the dense-fallback draw (CPU generator) is the same in both
    return fn()


# --------------------------------------------------------------------------------------------------
# Compiled equals eager
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(MODES))
@pytest.mark.usefixtures("small_blocks")
def test_p2_8_compiled_equals_eager(name: str) -> None:
    """The compiled window equals ``HSN.forward`` to the fp32 rounding tolerance (logits, final
    state, every ``info`` value, parameter gradients of a fixed functional), with integer and
    boolean outputs exact, in every compiled mode (dense training and evaluation, top-K with
    straight-through gates, the soft warm-up's masked round, top-K evaluation), from a random
    state over two blocks: dense with a reset at ``t = 0`` and one mid-window; routed from
    distinct cell states with every top-K margin above :data:`MIN_MARGIN` (checked)."""
    overrides, training = MODES[name]
    routed = "model.routing=topk" in overrides
    model = _model(_cfg(*overrides), training=training, scale=0.3 if routed else 0.1)
    ctx = _ctx()
    window = _window(model, ctx)
    tokens, cx, state, reset = _case_inputs(model, routed)
    assert block_size(T) == T // 2  # two blocks
    ref = _run(lambda: model(tokens, cx, state, reset))
    got = _run(lambda: window(tokens, cx, state, reset))
    if routed:
        assert _margin(ref[2], model.cfg.k_select) > MIN_MARGIN
    tol = TOL[("cpu", "fp32")]
    diff = _max_diff(got, ref)
    assert diff["exact"] == 0.0
    assert diff["values"] <= tol["values"], diff
    if training:
        assert _grad_diff(_grads(model, got), _grads(model, ref)) <= tol["grads"]


@pytest.mark.usefixtures("small_blocks")
def test_p2_8_compiled_dense_fallback_equals_eager() -> None:
    """Top-K STE with P2.2's dense fallback: the rows drawn dense run the dense step inside the
    compiled block, as in the eager window (the same CPU draw)."""
    model = _model(
        _cfg("model.routing=topk", "train.gate_mode=ste", "train.p_dense=0.5"),
        training=True,
        scale=0.3,
    )
    window = _window(model, _ctx())
    tokens, cx, state, reset = _case_inputs(model, True)
    ref = _run(lambda: model(tokens, cx, state, reset))
    got = _run(lambda: window(tokens, cx, state, reset))
    assert _margin(ref[2], model.cfg.k_select) > MIN_MARGIN
    assert bool(ref[2]["dense_fallback"].max() > 0)  # some row ran dense
    diff = _max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= TOL[("cpu", "fp32")]["values"], diff
    assert _grad_diff(_grads(model, got), _grads(model, ref)) <= TOL[("cpu", "fp32")]["grads"]


@pytest.mark.parametrize("name", ["dense_eval", "topk_eval"])
def test_p2_8_serving_step_equals_eager(name: str) -> None:
    """The serving step (:func:`compile_step`) on a preallocated state, updated in place, equals
    ``HSN.step`` position by position (the eager reference), compiled and at ``none``."""
    overrides, _ = MODES[name]
    model = _model(_cfg(*overrides), training=False)
    ctx = _ctx()
    tokens, cx, state0, _ = _inputs(model)
    for mode in ("none", "reduce-overhead"):
        step = compile_step(model, ctx, mode, backend=CPU_BACKEND)  # type: ignore[arg-type]
        assert isinstance(step, ServingStep)
        served = CellGraphState(
            h=state0.h.clone(), q=state0.q.clone(), age=state0.age.clone(), pos=state0.pos.clone()
        )
        ref_state = state0
        for t in range(T):
            with torch.no_grad():
                out = model.step(tokens[:, t], cx[:, t], ref_state)
            ref_state = out.state
            logits = step(tokens[:, t], cx[:, t], served)
            tol = 0.0 if mode == "none" else TOL[("cpu", "fp32")]["values"]
            assert float((logits - out.logits).abs().max()) <= tol
            assert float((served.h - ref_state.h).abs().max()) <= tol
            assert torch.equal(served.age, ref_state.age) and torch.equal(served.pos, ref_state.pos)


# --------------------------------------------------------------------------------------------------
# Recompiles, scope, capture
# --------------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("small_blocks")
def test_p2_8_no_recompiles_after_warmup() -> None:
    """Once the variants in use are warm (training with and without dense-fallback rows,
    evaluation), new data, new reset patterns, new dense-fallback rows and a new step counter
    never recompile: dynamo's ``fail_on_recompile`` stance stays silent and the graph count does
    not move. (A temperature ramp is outside the compiled scope:
    ``test_p2_8_unsupported_modes_refused``.)"""
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    cfg = _cfg(
        "model.routing=topk",
        "train.gate_mode=ste",
        "train.p_dense=0.5",
        "train.temperature_start=0.7",
        "train.temperature_end=0.7",
        "model.age_max=2",
    )
    model = _model(cfg, training=True)
    window = _window(model, _ctx(), backend="aot_eager")

    def run(seed: int, step: int, training: bool, dense_seed: int) -> None:
        model.train(training)
        model.set_train_step(step)
        tokens, cx, state, reset = _inputs(model, seed=seed)
        reset = torch.rand(reset.shape, generator=torch.Generator().manual_seed(seed)) < 0.2
        torch.manual_seed(dense_seed)
        window(tokens, cx, state, reset)

    def draws_rows(seed: int) -> bool:  # the trainer-side CPU draw of the dense fallback
        torch.manual_seed(seed)
        return bool((torch.rand(B) < 0.5).any())

    with_rows = [s for s in range(SEED_SEARCH) if draws_rows(s)][:3]
    without_rows = [s for s in range(SEED_SEARCH) if not draws_rows(s)][:3]
    assert len(with_rows) == 3 and len(without_rows) == 3
    # warm: training with and without fallback rows, then evaluation
    run(0, 0, True, with_rows[0])
    run(0, 0, True, without_rows[0])
    run(0, 0, False, 0)
    graphs = counters["stats"]["unique_graphs"]
    with torch.compiler.set_stance("fail_on_recompile"):
        for i, seed in enumerate(range(1, 5)):
            run(seed, seed * 3, True, (with_rows + without_rows)[i % 6])
            run(seed, seed * 3, False, seed)
    assert counters["stats"]["unique_graphs"] == graphs


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (("model.routing=greedy_budget",), "greedy_budget"),
        (("model.routing=random",), "random"),
        (("model.routing=fixed", "model.fixed_route=[1, 4]"), "fixed"),
        (("model.routing=topk", "train.gate_mode=reinforce"), "reinforce"),
        (
            (
                "model.routing=topk",
                "train.gate_mode=ste",
                "train.epsilon_start=0.1",
                "train.epsilon_end=0.1",
            ),
            "epsilon",
        ),
        (
            (
                "model.routing=topk",
                "train.gate_mode=ste",
                "train.temperature_start=1.0",
                "train.temperature_end=0.5",
                "train.temperature_steps=10",
            ),
            "temperature ramp",
        ),
    ],
)
def test_p2_8_unsupported_modes_refused(overrides: tuple[str, ...], reason: str) -> None:
    """Every mode outside the compiled scope raises ``NotImplementedError`` naming its reason
    when the window or the serving step is built (they run with ``compile_mode: none``)."""
    model = _model(_cfg(*overrides), training=True)
    with pytest.raises(NotImplementedError, match=reason):
        compile_window(model, _ctx(), "reduce-overhead", backend="aot_eager")
    with pytest.raises(NotImplementedError, match=reason):
        compile_step(model, _ctx(), "reduce-overhead", backend="aot_eager")


def test_p2_8_fp64_and_grad_accum_refused() -> None:
    """The fp64 master dtype (the gradchecks stay eager: the chunk's "Do not") and
    ``train.grad_accum > 1`` are refused; ``none`` always returns the eager forward."""
    model = _model(_cfg(), training=True).double()
    with pytest.raises(NotImplementedError, match="fp64"):
        compile_window(model, _ctx(), "reduce-overhead", backend="aot_eager")
    model = _model(_cfg(), training=True)
    with pytest.raises(NotImplementedError, match="grad_accum"):
        check_compile_scope(model, grad_accum=2)
    eager = compile_window(model, _ctx(), "none")
    assert not isinstance(eager, CompiledWindow)
    tokens, cx, state, reset = _inputs(model)
    got, ref = eager(tokens, cx, state, reset), model(tokens, cx, state, reset)
    assert torch.equal(got[0], ref[0])
    with pytest.raises(ValueError, match="compile mode"):
        compile_window(model, _ctx(), "fast")  # type: ignore[arg-type]


@pytest.mark.usefixtures("small_blocks")
def test_p2_8_capture_runs_eager() -> None:
    """``capture=True`` is a diagnostic of the eager forward: the compiled window refuses it and
    the model's own forward still captures after a compiled window ran."""
    model = _model(_cfg(), training=False)
    window = _window(model, _ctx(), backend="aot_eager")
    tokens, cx, state, reset = _inputs(model)
    window(tokens, cx, state, reset)
    with pytest.raises(ValueError, match="capture"):
        window(tokens, cx, state, reset, capture=True)
    _, _, info = model(tokens, cx, state, reset, capture=True)
    assert info["cell_states"].shape == (B, T, model.cfg.n_cells, model.cfg.d)


def test_p2_8_mps_falls_back_with_notice(caplog: pytest.LogCaptureFixture) -> None:
    """On MPS a backend that cannot compile falls back to the eager window and step with one
    logged notice (simulated on the CPU: an MPS device context and a backend that raises); on the
    CPU the same failure raises at the first call (errors over silence)."""

    def failing(gm: Any, example_inputs: Any) -> Any:
        raise RuntimeError("this backend cannot compile")

    model = _model(_cfg(), training=False)
    cpu = _ctx()
    mps = DeviceContext(
        device=torch.device("mps"),
        autocast_dtype=None,
        master_dtype=torch.float32,
        supports_bf16=False,
        unified_memory=True,
        vram_bytes=0,
        compile_enabled=True,
        profile="m_series",
    )
    with caplog.at_level(logging.WARNING, logger=compile_mod.__name__):
        window = compile_window(model, mps, "reduce-overhead", backend=failing)  # type: ignore[arg-type]
        step = compile_step(model, mps, "reduce-overhead", backend=failing)  # type: ignore[arg-type]
    assert not isinstance(window, CompiledWindow)
    assert isinstance(step, ServingStep) and step.mode == "none"
    notices = [r for r in caplog.records if "runs eager" in r.getMessage()]
    assert len(notices) == 2
    tokens, cx, state, reset = _inputs(model)
    assert torch.equal(window(tokens, cx, state, reset)[0], model(tokens, cx, state, reset)[0])
    compiled = compile_window(model, cpu, "reduce-overhead", backend=failing)  # type: ignore[arg-type]
    with pytest.raises(Exception, match="cannot compile"):
        compiled(tokens, cx, state, reset)


# --------------------------------------------------------------------------------------------------
# Invariants with the compiled window (P1.8 subset) and the dense symmetry
# --------------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("routing", [(), ("model.routing=topk", "train.gate_mode=ste")])
def test_p2_8_invariant_subset_with_compile(routing: tuple[str, ...]) -> None:
    """P1.8 invariants on the compiled window (CPU), dense and top-K STE: the state stays in
    ``[-1, 1]`` over many windows from the zero state; a reset at ``t = 0`` removes the prior
    state (two start states, the same outputs bit for bit); no leakage (changing the inputs from
    position ``t0`` on leaves every earlier output bit for bit); a disabled cell keeps its state;
    and two models built from one seed give the same outputs bit for bit."""
    cfg = _cfg("model.disabled_cells=[1]", *routing)
    model = _model(cfg, training=True, scale=0.3)
    window = _window(model, _ctx())
    tokens, cx, state, _ = _inputs(model)
    no_reset = torch.zeros(B, T, dtype=torch.bool)
    # state bound over 12 windows from the zero state
    carried = model.init_state(B)
    for seed in range(12):
        toks, cxs, _, _ = _inputs(model, seed=seed)
        _, carried, info = window(toks, cxs, carried, no_reset)
        carried = carried.detach()
        assert float(info["state_abs_max"].max()) <= 1.0
    # disabled cell 1: unchanged from its start value
    _, final, _ = window(tokens, cx, state, no_reset)
    assert torch.equal(final.h[:, 1], state.h[:, 1])
    # reset at t = 0 in every row removes the start state
    reset0 = no_reset.clone()
    reset0[:, 0] = True
    other = model.init_state(B).with_dense(torch.rand_like(state.h) * 2 - 1)
    a, b = window(tokens, cx, state, reset0), window(tokens, cx, other, reset0)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1].h, b[1].h)
    # leakage: inputs from t0 on cannot change outputs before t0
    t0 = 5
    changed = tokens.clone()
    changed[:, t0:] = (changed[:, t0:] + 17) % model.cfg.vocab_size
    base, edited = window(tokens, cx, state, no_reset), window(changed, cx, state, no_reset)
    assert torch.equal(base[0][:, :t0], edited[0][:, :t0])
    assert torch.equal(base[2]["z_hat"][:, : t0 + 1], edited[2]["z_hat"][:, : t0 + 1])
    # determinism: a second model from the same seed
    twin = _model(cfg, training=True, scale=0.3)
    again = _window(twin, _ctx())(tokens, cx, state, no_reset)
    assert torch.equal(again[0], base[0]) and torch.equal(again[1].h, base[1].h)


def _symmetry(window: Callable[..., Window], model: HSN, b: int, t: int) -> tuple[float, float]:
    """Cell spread (``max_i |h_i − h_0|`` of the final state) and ``max |edge_logit.grad|`` of one
    window of the unperturbed dense model from the zero state."""
    generator = torch.Generator(device="cpu").manual_seed(5)
    dev = model.device
    tokens = torch.randint(0, 256, (b, t), generator=generator).to(dev)
    cx = torch.randint(0, 16, (b, t), generator=generator).to(dev)
    reset = torch.zeros(b, t, dtype=torch.bool, device=dev)
    reset[:, 0] = True
    model.zero_grad(set_to_none=True)
    logits, final, info = window(tokens, cx, model.init_state(b), reset)
    h = final.h.detach().clone()
    (logits.float().logsumexp(-1).mean() + info["z_hat"].float().square().mean()).backward()
    grad = model.cell.graph.edge_logit.grad
    assert grad is not None
    return float((h - h[:, :1]).abs().max()), float(grad.abs().max())


@pytest.mark.usefixtures("small_blocks")
def test_p2_8_dense_symmetry_exact_compiled() -> None:
    """The dense reference's 32 cells stay bitwise identical from the zero state and the edge
    logits get exactly zero gradient through the compiled window (the efficiency audit's finding
    1; the running experiments depend on it): ``ref_a1``, CPU, fp32."""
    cfg = load_config(REF_A1, overrides=["train.unroll=16"])
    seed_everything(0)
    model = HSN.from_config(cfg)
    model.train()
    spread, edge = _symmetry(_window(model, _ctx()), model, 2, 16)
    assert spread == 0.0 and edge == 0.0
    spread, edge = _symmetry(model, model, 2, 16)  # the eager reference, for the record
    assert spread == 0.0 and edge == 0.0


# --------------------------------------------------------------------------------------------------
# Static K and the capturable round
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["topk", "greedy_budget", "random", "fixed"])
@pytest.mark.parametrize("budget", [1.0, 0.2])
def test_p2_8_static_k_every_mode(mode: str, budget: float) -> None:
    """Every sparse mode returns exactly ``K = k_select`` slots whatever the budget; padded
    slots are ``valid=False`` with cell 0, unit 0 and ``α = 0`` (the in-range sentinel)."""
    overrides = [f"model.routing={mode}", f"model.budget={budget}"]
    if mode == "fixed":
        overrides.append("model.fixed_route=[1]")
    model = _model(_cfg(*overrides), training=False)
    logits = torch.randn(B, model.cfg.n_cells, model.cfg.n_units)
    fixed = torch.tensor([1]) if mode == "fixed" else None
    sel = model.router.select(
        logits,
        budget=budget,
        k_max=model.cfg.k_select,
        k_min=1,
        mode=mode,  # type: ignore[arg-type]
        fixed=fixed,
    )
    assert sel.cell_idx.shape == sel.unit_idx.shape == sel.valid.shape == (B, model.cfg.k_select)
    padded = ~sel.valid
    assert bool((sel.cell_idx[padded] == 0).all()) and bool((sel.unit_idx[padded] == 0).all())
    assert bool((sel.alpha[padded] == 0).all())


def test_p2_8_static_k_false_refused() -> None:
    """``model.static_k=false`` (a data-dependent K) is refused at construction."""
    with pytest.raises(NotImplementedError, match="static_k"):
        HSN.from_config(_cfg("model.routing=topk", "model.static_k=false"))
    with pytest.raises(NotImplementedError, match="static_k"):
        Router.from_config(_cfg("model.static_k=false").model, unroll=16)


def test_p2_8_static_round_matches_l0_round() -> None:
    """The capturable round (``static=True``: no host synchronisation, every unit on the pairs
    then a gather) equals the L0 sparse round to rounding in fp64, with padded slots ignored:
    the updated state, the slot values and the gradients; and :meth:`UnitBank.forward_static`
    equals the grouped pass."""
    probe = _model(_cfg("model.routing=topk"), training=False).router
    budget = probe.shared_cost + 1.5 * probe.max_pair_cost  # one cell fits, two do not
    cfg = _cfg("model.routing=topk", f"model.budget={budget}")
    model = _model(cfg, training=False, scale=0.3).double()
    tokens, cx, state, _ = _inputs(model)
    bank = model.cell.units
    v = torch.randn(7, bank.in_dim, dtype=torch.float64, requires_grad=True)
    idx = torch.tensor([2, 0, 0, 1, 2, 1, 0])
    a = native.grouped_unit_mlp(v, idx, bank)
    b_ = native.grouped_unit_mlp(v, idx, bank, static=True)
    torch.testing.assert_close(b_, a, atol=1e-12, rtol=0.0)
    ga = torch.autograd.grad(a.square().sum(), [v, bank.W1, bank.W2])
    gb = torch.autograd.grad(b_.square().sum(), [v, bank.W1, bank.W2])
    for x, y in zip(ga, gb, strict=True):
        torch.testing.assert_close(y, x, atol=1e-12, rtol=0.0)
    # the round on a selection with padded slots
    logits = model.router(
        model.scorer.features(
            torch.randn(B, cfg.model.d, dtype=torch.float64),
            torch.randn(B, cfg.model.d, dtype=torch.float64),
            state,
            cx[:, 0],
            budget_frac=budget,
            round_idx=0,
        )
    )
    sel = model.router.select(logits, budget=budget, k_max=cfg.model.k_select, k_min=1, mode="topk")
    assert not bool(sel.valid.all())  # the tight budget pads
    u = torch.randn(B, cfg.model.d, dtype=torch.float64)
    eps = torch.randn(B, cfg.model.d, dtype=torch.float64)
    h0, info0 = model.cell.step_sparse(u, eps, state, sel)
    h1, info1 = model.cell.step_sparse(u, eps, state, sel, static=True)
    torch.testing.assert_close(h1, h0, atol=1e-12, rtol=0.0)
    s0, s1 = info0["state"], info1["state"]
    assert isinstance(s0, CellGraphState) and isinstance(s1, CellGraphState)
    torch.testing.assert_close(s1.h, s0.h, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(s1.q, s0.q, atol=1e-12, rtol=0.0)
    assert torch.equal(s1.age, s0.age)


def test_p2_8_block_size() -> None:
    """Blocks are the largest divisor of ``T`` up to 16: 16 for the A.1 unroll."""
    assert block_size(256) == 16
    assert block_size(8) == 8
    assert block_size(48) == 16
    assert block_size(36) == 12
    assert block_size(37) == 1
    with pytest.raises(ValueError):
        block_size(0)


# --------------------------------------------------------------------------------------------------
# The trainer
# --------------------------------------------------------------------------------------------------


def _docs(n: int, split: str = "train") -> list[Example]:
    task = SPEC.task("kv", "in_range")
    spec = TaskSpec.model_validate(
        {
            **task.model_dump(),
            "n_train": n,
            "n_dev": n,
            "n_test": n,
            "length_range": [16, 28],
            "dep_len_range": [4, 10],
            "unroll": 32,
        }
    )
    return [generate(spec, split, i) for i in range(n)]  # type: ignore[arg-type]


def _trainer_cfg(compile_mode: str, *overrides: str) -> RunConfig:
    return load_config(
        REF_A1,
        overrides=[
            "model.d=8",
            "model.context_dim=8",
            "model.n_cells=4",
            "model.in_degree=2",
            "model.n_units=2",
            "model.k_select=2",
            "model.hidden=16",
            "model.scorer_hidden=8",
            "train.unroll=32",
            "train.batch=4",
            "train.micro_batch=4",
            "train.steps=5",
            "train.eval_every=100",
            "train.ckpt_every=100",
            "train.stats_every=3",
            f"hardware.compile_mode={compile_mode}",
            *overrides,
        ],
    )


@pytest.mark.usefixtures("small_blocks")
def test_p2_8_trainer_compile_time_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compiled run: every window runs the compiled window (evaluation too: the same metrics as
    the eager evaluation to rounding), and a statistics step's per-term
    gradient norms come from one extra compiled pass per term with a gradient path, equal to the
    eager run's retained-backward norms to rounding; the trailing cut is off (full windows); the
    steps that compiled are reported as ``compile_seconds`` / ``compile_steps`` and kept out of
    ``train_seconds``; the losses follow the eager run's to rounding."""
    monkeypatch.setattr(
        tbptt, "compile_window", functools.partial(compile_window, backend="aot_eager")
    )
    train, dev = ByteDataset(_docs(16)), ByteDataset(_docs(4, "dev"))
    histories = {}
    for mode in ("none", "reduce-overhead"):
        cfg = _trainer_cfg(mode)
        ctx = resolve(cfg.hardware)
        assert ctx.compile_enabled == (mode != "none")
        seed_everything(0)
        trainer = TBPTTTrainer(HSN.from_config(cfg), cfg, ctx, run_dir=tmp_path / mode)
        histories[mode] = trainer.fit(train, dev, 5)
        if mode != "none":
            compiled = trainer._compiled
            assert compiled is not None
            # five windows, plus one pass per gradient term on statistics steps 3 and 5
            live = sum(
                1
                for key, value in histories[mode].steps[2].stats.items()
                if key.startswith("term_grad_norm/") and value > 0.0
            )
            assert live >= 2  # task and pred at least
            # and the evaluation at the last step runs its windows compiled (an eval variant)
            assert compiled.calls > 5 + 2 * live
            assert trainer.compiled_window is compiled
            assert not trainer._cut_enabled
    eager, comp = histories["none"], histories["reduce-overhead"]
    assert comp.measured["compile_steps"] >= 1.0 and comp.measured["compile_seconds"] > 0.0
    assert eager.measured["compile_steps"] == 0.0 and eager.measured["compile_seconds"] == 0.0
    assert eager.measured["compiled_windows"] == 0.0 and comp.measured["compiled_windows"] >= 9.0
    stamped = sum(r.seconds for r in comp.steps)
    total = comp.measured["train_seconds"] + comp.measured["compile_seconds"]
    assert abs(stamped - total) < 1e-6
    assert all(r.stats["window_positions"] == 32.0 for r in comp.steps)  # no cut when compiled
    for a, b in zip(eager.steps, comp.steps, strict=True):
        assert abs(a.total - b.total) <= 1e-4 * max(1.0, abs(a.total))
        norms = {k: v for k, v in a.stats.items() if k.startswith("term_grad_norm/")}
        assert list(norms) == [k for k in b.stats if k.startswith("term_grad_norm/")]
        for key, value in norms.items():
            assert abs(b.stats[key] - value) <= 1e-4 * max(1.0, abs(value)), key
    assert len(eager.evals) == len(comp.evals) == 1  # the last step evaluates
    for key, value in eager.evals[0].items():
        if isinstance(value, float):
            assert abs(comp.evals[0][key] - value) <= 1e-4 * max(1.0, abs(value)), key


# --------------------------------------------------------------------------------------------------
# CUDA
# --------------------------------------------------------------------------------------------------


def _cuda_ctx() -> DeviceContext:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    return _ctx("rtx3060")


CUDA_CASES = [
    pytest.param(name, precision, id=f"{name}-{precision}")
    for name in ("dense", "dense_eval", "topk_ste", "topk_soft", "topk_eval")
    for precision in ("fp32", "bf16")
    # bf16 rounding (~1e-2) reaches the top-K margins of a routed window: its decisions are
    # compared in fp32 only
    if precision == "fp32" or not MODES[name][0]
]


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize(("name", "precision"), CUDA_CASES)
def test_p2_8_compiled_equals_eager_cuda(name: str, precision: str) -> None:
    """The compiled window with CUDA Graphs (warm-up, recording, replay) equals the eager
    window on CUDA over two blocks: fp32 to the measured tolerance (routed windows from distinct
    cell states, every top-K margin above :data:`MIN_MARGIN`); bf16 autocast (dense; one bf16
    rounding at an early position spreads through the window) by comparison: the compiled
    window is no further from the fp32 eager reference than the eager bf16 window is, up to a
    factor 2."""
    ctx = _cuda_ctx()
    overrides, training = MODES[name]
    routed = "model.routing=topk" in overrides
    model = _model(_cfg(*overrides), training=training, scale=0.3 if routed else 0.1)
    model = model.to(ctx.device)
    window = _window(model, ctx, backend="inductor")
    tokens, cx, state, reset = _case_inputs(model, routed)
    ref32 = _run(lambda: model(tokens, cx, state, reset))
    if routed:
        assert _margin(ref32[2], model.cfg.k_select) > MIN_MARGIN
    if precision == "fp32":
        for _ in range(3):  # warm-up, recording, replay
            got = _run(lambda: window(tokens, cx, state, reset))
        diff = _max_diff(got, ref32)
        tol = TOL[("cuda", "fp32")]
        assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
        if training:
            assert _grad_diff(_grads(model, got), _grads(model, ref32)) <= tol["grads"]
        return
    with torch.autocast("cuda", torch.bfloat16):
        ref16 = _run(lambda: model(tokens, cx, state, reset))
        for _ in range(3):
            got = _run(lambda: window(tokens, cx, state, reset))
    eager_err = _max_diff(ref16, ref32)["values"]
    got_err = _max_diff(got, ref32)["values"]
    assert got_err <= 2.0 * eager_err + 1e-6, (got_err, eager_err)


def _launches(fn: Callable[[], Any]) -> tuple[int, int]:
    """(host launch calls, device events) of ``fn`` under ``torch.profiler``."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    host = dev = 0
    for event in prof.events():
        if event.device_type.name == "CPU" and event.name in HOST_LAUNCH_CALLS:
            host += 1
        elif event.device_type.name == "CUDA" and str(event.activity_type) in (
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        ):
            dev += 1
    return host, dev


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("name", ["dense", "topk_ste"])
def test_p2_8_launches_drop_at_least_2x(name: str) -> None:
    """One training window (forward and backward) on CUDA issues at least 2x fewer host
    launch calls at L1 (compiled, CUDA Graphs) than at L0 (eager): ref_a1 at T = 32, in blocks of
    4 positions to keep the compile short (the A.1 window runs blocks of 16; the profiles of
    ``scripts/profile_step.py`` measure that)."""
    ctx = _cuda_ctx()
    overrides, _ = MODES[name]
    cfg = load_config(REF_A1, overrides=["train.unroll=32", *overrides])
    seed_everything(0)
    model = HSN.from_config(cfg).to(ctx.device)
    model.train()
    window = _window(model, ctx, backend="inductor")
    tokens, cx, state, reset = _inputs(model, b=4, t=32)

    def step(fn: Callable[..., Window]) -> None:
        model.zero_grad(set_to_none=True)
        _loss(fn(tokens, cx, state, reset)).backward()

    for _ in range(3):
        step(window)
    step(model)
    eager = _launches(lambda: step(model))
    compiled = _launches(lambda: step(window))
    print(f"host launches {eager[0]} -> {compiled[0]}; device events {eager[1]} -> {compiled[1]}")
    assert eager[0] >= 2 * compiled[0], (eager, compiled)


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_p2_8_dense_symmetry_exact_compiled_cuda(precision: str, tmp_path: Path) -> None:
    """On CUDA the compiled window keeps the dense cells bitwise identical and the edge gradient
    exactly zero from the zero state (fp32 and bf16 autocast), and a short compiled training run
    (30 steps of the TBPTT trainer, CUDA Graphs) leaves ``edge_logit`` exactly uniform and the
    carried cells bitwise identical."""
    ctx = _cuda_ctx()
    cfg = load_config(REF_A1, overrides=["train.unroll=16"])
    seed_everything(0)
    model = HSN.from_config(cfg).to(ctx.device)
    model.train()
    window = _window(model, ctx, backend="inductor")
    autocast = torch.autocast("cuda", torch.bfloat16, enabled=precision == "bf16")
    with autocast:
        for _ in range(3):
            spread, edge = _symmetry(window, model, 4, 16)
            assert spread == 0.0 and edge == 0.0
    # a short compiled training run
    overrides = [
        "train.unroll=32",
        "train.batch=4",
        "train.micro_batch=4",
        "train.steps=30",
        "train.eval_every=1000",
        "train.ckpt_every=1000",
        "train.stats_every=10",
        "train.lr=1e-3",
        "profile=rtx3060",
        "hardware.compile_mode=reduce-overhead",
        f"hardware.autocast_dtype={precision}",
    ]
    cfg = load_config(REF_A1, overrides=overrides)
    seed_everything(0)
    hsn_model = HSN.from_config(cfg)
    trainer = TBPTTTrainer(hsn_model, cfg, resolve(cfg.hardware), run_dir=tmp_path)
    trainer.fit(ByteDataset(_docs(32)), ByteDataset(_docs(4, "dev")), 30)
    edge_logit = hsn_model.cell.graph.edge_logit.detach()
    assert bool((edge_logit == edge_logit[0]).all())
    carried = trainer._carried_state
    assert carried is not None
    assert float((carried.h - carried.h[:, :1]).abs().max()) == 0.0


@pytest.mark.cuda
@pytest.mark.usefixtures("small_blocks")
def test_p2_8_compiled_statistics_step_cuda(tmp_path: Path) -> None:
    """On CUDA (CUDA Graphs), a statistics step's per-term gradient norms come from extra compiled
    passes and equal the eager run's retained-backward norms to fp32 rounding; the losses and the
    evaluation agree too."""
    _cuda_ctx()
    histories = {}
    for mode in ("none", "reduce-overhead"):
        cfg = _trainer_cfg(
            mode,
            "profile=rtx3060",
            "hardware.autocast_dtype=fp32",
            "train.precision=fp32",
            "train.steps=6",
            "train.stats_every=3",
        )
        seed_everything(0)
        trainer = TBPTTTrainer(HSN.from_config(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path)
        histories[mode] = trainer.fit(ByteDataset(_docs(16)), ByteDataset(_docs(4, "dev")), 6)
    eager, comp = histories["none"], histories["reduce-overhead"]
    assert comp.measured["compiled_windows"] > 6.0
    for a, b in zip(eager.steps, comp.steps, strict=True):
        assert abs(a.total - b.total) <= 1e-3 * max(1.0, abs(a.total))
        for key, value in a.stats.items():
            if key.startswith("term_grad_norm/"):
                assert abs(b.stats[key] - value) <= 1e-3 * max(1.0, abs(value)), key
    for key, value in eager.evals[0].items():
        if isinstance(value, float):
            assert abs(comp.evals[0][key] - value) <= 1e-3 * max(1.0, abs(value)), key
