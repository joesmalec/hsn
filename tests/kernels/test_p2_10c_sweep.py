"""P2.10c: the reverse sweep's three kernels (``hsn_win_back_head`` / ``_mid`` / ``_tail``)
against their PyTorch steps (:class:`hsn.train.state_sweep.ReverseSweep`), on the card.

Every comparison is on one record: two sweeps are loaded with the same window, one runs the
kernels and one the PyTorch steps. Each kernel is tested alone from identical buffers at every
position (nothing accumulates: the tolerance is rounding), then whole windows, the captured
CUDA graph, and the runner that the recurrence node calls. fp32, and bf16 products as under
autocast. CUDA-marked; skipped without a card or the built artifacts unless
``HSN_REQUIRE_CUDA=1``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from hsn.config import RunConfig, load_config
from hsn.device import seed_everything
from hsn.kernels import window
from hsn.model.hsn import HSN
from hsn.train import state_gradient
from hsn.train.compile import fixed_fast_slots
from hsn.train.split import SplitWindow
from hsn.train.state_sweep import ReverseSweep, SweepRunner, sweep_shape
from torch import Tensor

pytestmark = pytest.mark.cuda

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/experiments/hsn_recipe_a6.yaml"
CUT = "train.feedback_detach=[summary,eps,router]"

#: what a kernel and its PyTorch step both write
MUTABLE = (
    "A", "R", "dc", "db", "op1", "op1_fast", "op2", "op2_fast", "act_r", "act_f", "dv_r", "dv_f",
    "sigma",
)  # fmt: skip


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


@pytest.fixture(autouse=True)
def _no_fast_sweep() -> object:
    yield
    state_gradient.set_fast_sweep(None)


# --------------------------------------------------------------------------------------------------
# Cases: a model on the card, one window, and what its recurrence node hands the sweep
# --------------------------------------------------------------------------------------------------

SMALL = {
    "routed": (),
    "no_cand_norm": ("model.cand_norm=none",),
    "no_cell_norm": ("model.cell_norm=none",),
    "padded": ("model.budget=0.1",),
    "silu": ("model.activation=silu",),
    "relu": ("model.activation=relu",),
    "fast_read_routed_units": ("model.k_select=4", "model.fast_cells=2", "model.fast_readout=true"),
    "fast_fixed_units": (
        "model.k_select=4", "model.fast_cells=2", "model.fast_readout=true",
        "model.fast_units=fixed",
    ),
    "fast_fixed_units_cut": (
        "model.k_select=4", "model.fast_cells=2", "model.fast_readout=true",
        "model.fast_units=fixed", "model.budget=0.1",
    ),
}  # fmt: skip

#: the recipe's widths: the routed candidate and the approved shape, short windows
WIDE = {
    "candidate_routed": ("model.n_cells=64", "model.k_select=12"),
    "approved_shape": (
        "model.n_cells=64", "model.k_select=20", "model.fast_cells=8", "model.fast_units=fixed",
        "model.fast_readout=true",
    ),
}  # fmt: skip


def _small_cfg(*extra: str) -> RunConfig:
    from test_a6_feedback_detach import A5, _cfg

    return _cfg(*A5, CUT, "model.cand_norm=rms", "train.precision=fp32", *extra)


def _wide_cfg(*extra: str) -> RunConfig:
    return load_config(
        RECIPE,
        [
            "profile=cpu", CUT, "train.unit_credit=executed", "train.soft_until_step=0",
            "train.precision=fp32", "train.batch=5", "train.micro_batch=5", "train.unroll=10",
            *extra,
        ],
    )  # fmt: skip


def _arguments(
    cfg: RunConfig, device: torch.device, *, autocast: bool, scale: float
) -> tuple[HSN, tuple, tuple]:
    """A model on the card, a window, and the arguments its recurrence node gives the sweep
    (captured from the PyTorch sweep); with ``autocast`` the replay ran under bf16, so the
    record's hidden values are bf16."""
    seed_everything(0)
    model = HSN.from_config(cfg).to(device)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g).to(device))
    model.train()
    B, T = cfg.train.batch, cfg.train.unroll
    tokens = torch.randint(0, cfg.model.vocab_size, (B, T), generator=g).to(device)
    ctx = torch.randint(0, cfg.model.n_context, (B, T), generator=g).to(device)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    reset[0, 0] = False
    reset[1, T // 2] = True
    reset[2, T - 1] = True
    reset = reset.to(device)
    state = model.init_state(B)
    with torch.no_grad():
        state.h.copy_(0.5 * torch.rand(state.h.shape, generator=g).to(device) - 0.25)
        state.q.copy_(state.h.mean(dim=1))
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    mask[:, 1::3] = True
    given: list[tuple] = []
    sweep = state_gradient.state_gradient_backward

    def keeping(*args: object, **kwargs: object) -> tuple[Tensor, Tensor]:
        given.append(args)
        return sweep(*args, **kwargs)  # type: ignore[arg-type]

    state_gradient.state_gradient_backward = keeping  # type: ignore[assignment]
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            split = SplitWindow(model, cfg, None, state_gradient=True)
            state_gradient.set_fast_sweep(None)  # the PyTorch sweep: its arguments are kept
            logits, _, info = split(tokens, ctx, state, reset)
        _loss(logits, info, mask).backward()
    finally:
        state_gradient.state_gradient_backward = sweep
    assert len(given) == 1
    return model, given[0], (tokens, ctx, state, reset, mask)


def _loss(logits: Tensor, info: dict[str, Tensor], mask: Tensor) -> Tensor:
    task = logits.float()[mask].square().mean()
    pred = (info["z_hat"].float() - info["z_target"].float()).square().mean()
    return task + 0.7 * pred + 0.3 * info["expected_cost"].square().sum()


def _pair(model: HSN, args: tuple, cfg: RunConfig) -> tuple[ReverseSweep, ReverseSweep]:
    """Two sweeps for one window: the kernels, and the PyTorch steps on the card."""
    c, _, _, _, _, _, _, n_cells, *_, record, consts = args
    n_read = cfg.model.fast_cells if cfg.model.fast_readout else 0
    shape = sweep_shape(c, record, consts, n_cells, fixed_fast_slots(model), n_read)
    assert window.back_unsupported(shape) is None
    make = lambda kernels: ReverseSweep(  # noqa: E731
        shape, consts, device=c.device, state_dtype=c.dtype,
        product_dtype=record["hidden"].dtype, kernels=kernels,
    )  # fmt: skip
    return make(True), make(False)


def _load(sweep: ReverseSweep, args: tuple) -> None:
    c, b, h_old, cells, units, valid, reset, _, *cots, record, consts = args
    sweep.load(c, b, h_old, cells, units, valid, reset, *cots, record, consts)


def _copy_state(dst: ReverseSweep, src: ReverseSweep) -> None:
    for name in MUTABLE:
        getattr(dst, name).copy_(getattr(src, name))


def _gap(got: Tensor, want: Tensor) -> float:
    """The largest difference of two buffers, relative to the reference's largest entry."""
    a, b = got.float(), want.float()
    return float((a - b).abs().max()) / max(float(b.abs().max()), 1e-30)


def _tolerance(t: Tensor) -> float:
    return 3e-5 if t.dtype == torch.float32 else 1.5e-2  # fp32 rounding; one bf16 step


# --------------------------------------------------------------------------------------------------
# Each kernel alone, from identical buffers, at every position
# --------------------------------------------------------------------------------------------------


def _check_kernels_step_by_step(model: HSN, args: tuple, cfg: RunConfig) -> dict[str, float]:
    fast, ref = _pair(model, args, cfg)
    _load(fast, args)
    _load(ref, args)
    worst: dict[str, float] = {}

    def compare(step: str, names: tuple[str, ...], t: int) -> None:
        for name in names:
            got, want = getattr(fast, name), getattr(ref, name)
            if name in ("dc", "db"):
                got, want = got[:, t], want[:, t]
            gap = _gap(got, want)
            worst[f"{step}.{name}"] = max(worst.get(f"{step}.{name}", 0.0), gap)
            assert gap <= _tolerance(got), (step, name, t, gap)

    with torch.no_grad():
        for t in range(fast.shape.T - 1, -1, -1):
            _copy_state(fast, ref)  # the kernel starts from the reference's buffers
            window.back_head(fast, t)
            ref.head_reference(t)
            compare(
                "head", ("A", "R", "dc", "db", "op1", "op1_fast", "op2", "op2_fast", "sigma"), t
            )
            _copy_state(fast, ref)
            fast.products_second()
            ref.products_second()
            window.back_mid(fast, t)
            ref.middle_reference(t)
            compare("mid", ("op2", "op2_fast", "sigma"), t)
            _copy_state(fast, ref)
            fast.products_first()
            ref.products_first()
            window.back_tail(fast, t)
            ref.tail_reference(t)
            compare("tail", ("A", "R"), t)
    assert float(ref.dc.abs().max()) > 0 and float(ref.A.abs().max()) >= 0
    return worst


@pytest.mark.parametrize("name", list(SMALL))
def test_kernels_step_by_step_small_model(device: torch.device, name: str) -> None:
    """fp32, the small model (three units, d 8: the kernels' general paths): each kernel
    against its PyTorch step from the same buffers, at every position of a window."""
    cfg = _small_cfg(*SMALL[name])
    model, args, _ = _arguments(cfg, device, autocast=False, scale=0.3)
    _check_kernels_step_by_step(model, args, cfg)


@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("name", list(WIDE))
def test_kernels_step_by_step_recipe_widths(
    device: torch.device, name: str, autocast: bool
) -> None:
    """The recipe's widths (d 64, H 512, eight units, 64 cells): the routed candidate and the
    approved shape (eight fast cells at fixed units with the fast read, twelve routed slots),
    in fp32 and with bf16 products and operands as under autocast."""
    cfg = _wide_cfg(*WIDE[name])
    model, args, _ = _arguments(cfg, device, autocast=autocast, scale=0.05)
    assert args[-2]["hidden"].dtype == (torch.bfloat16 if autocast else torch.float32)
    worst = _check_kernels_step_by_step(model, args, cfg)
    print(name, "bf16" if autocast else "fp32", {k: f"{v:.1e}" for k, v in worst.items()})


# --------------------------------------------------------------------------------------------------
# Whole windows: the kernels' sweep, its CUDA graph, the reference sweep
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("name", ["routed", "fast_fixed_units", *WIDE])
def test_kernel_sweep_over_a_window(device: torch.device, name: str, autocast: bool) -> None:
    """A whole window on one record: the kernels' sweep against the PyTorch steps on the card
    and against the reference sweep of phase 1; the captured CUDA graph replays the eager
    kernels' result bit for bit, also on a second window's values."""
    wide = name in WIDE
    if autocast and not wide:
        pytest.skip("bf16 products: the recipe's widths")
    cfg = _wide_cfg(*WIDE[name]) if wide else _small_cfg(*SMALL[name])
    model, args, _ = _arguments(cfg, device, autocast=autocast, scale=0.05 if wide else 0.3)
    fast, ref = _pair(model, args, cfg)
    with torch.no_grad():
        _load(fast, args)
        _load(ref, args)
        dc, db = (x.clone() for x in fast.run())
        want_dc, want_db = (x.clone() for x in ref.run())
        reference = state_gradient.state_gradient_backward(*args)
    limit = 2e-4 if not autocast else 6e-2
    for got, want, what in ((dc, want_dc, "dc"), (db, want_db, "db")):
        assert _gap(got, want) <= limit, (what, _gap(got, want))
    if not autocast:  # the phase-1 reference multiplies in fp32 per unit: the same values
        assert _gap(dc, reference[0]) <= limit and _gap(db, reference[1]) <= limit
        assert _gap(want_dc, reference[0]) <= limit
    # the graph: captured on this window, replayed, then replayed on other cotangents
    with torch.no_grad():
        _load(fast, args)
        fast.capture()
        graph_dc, graph_db = (x.clone() for x in fast.run())
        assert torch.equal(graph_dc, dc) and torch.equal(graph_db, db)
        c, b, h_old, cells, units, valid, reset, n_cells, pre, post, sel, fst, record, consts = args
        other = (c, b, h_old, cells, units, valid, reset, n_cells,
                 None if pre is None else 0.5 * pre, post, None if sel is None else -sel, fst,
                 record, consts)  # fmt: skip
        _load(fast, other)
        again_dc, again_db = (x.clone() for x in fast.run())
        _load(ref, other)
        other_dc, other_db = ref.run()
        assert _gap(again_dc, other_dc) <= limit and _gap(again_db, other_db) <= limit
        assert not torch.equal(again_dc, dc)


def test_runner_gives_the_window_its_gradients(device: torch.device) -> None:
    """The runner the recurrence node calls (:class:`SweepRunner`): a window's parameter
    gradients through the kernels' sweep (a captured graph from the first call on) against the
    same window through the PyTorch sweep, fp32, the approved shape."""
    cfg = _wide_cfg(*WIDE["approved_shape"])
    model, _, (tokens, ctx, state, reset, mask) = _arguments(
        cfg, device, autocast=False, scale=0.05
    )
    params = list(model.parameters())
    split = SplitWindow(model, cfg, None, state_gradient=True)  # installs its runner
    runner = split.sweep_runner
    assert isinstance(runner, SweepRunner)

    def grads() -> list[Tensor | None]:
        logits, _, info = split(tokens, ctx, state, reset)
        return list(torch.autograd.grad(_loss(logits, info, mask), params, allow_unused=True))

    state_gradient.set_fast_sweep(None)
    want = grads()  # through the PyTorch sweep
    assert runner.calls == 0
    state_gradient.set_fast_sweep(runner)
    for round_ in range(3):
        got = grads()
        assert runner.calls == round_ + 1 and runner.captures == 1
        for mine, theirs in zip(got, want, strict=True):
            assert (mine is None) == (theirs is None)
            if theirs is not None:
                assert _gap(mine, theirs) <= 5e-4, round_  # type: ignore[arg-type]
    assert len(runner.sweeps) == 1


def test_kernels_decline_what_they_do_not_take(device: torch.device) -> None:
    cfg = _small_cfg()
    model, args, _ = _arguments(cfg, device, autocast=False, scale=0.3)
    fast, _ = _pair(model, args, cfg)
    assert window.back_unsupported(fast.shape._replace(K=33)) is not None
    assert window.back_unsupported(fast.shape._replace(d=256, K=32)) is not None
    runner = SweepRunner()
    c, b, *rest = args
    assert runner(c.double(), b.double(), *rest) is None  # another dtype: the PyTorch sweep
    assert runner.calls == 0
