"""P2.10b: the folded forward sweep (``hsn.train.compile._folded_tape_block``) records the tape
of the fused sweep of P2.10 bit for bit, in fp32 and under bf16 autocast, and the two glue
kernels' folded forms (``hsn.kernels.window.select_fold``, ``update_fold``) compute what the
kernels and casts they replace compute. CUDA-marked; skipped without a card or without the built
artifacts unless ``HSN_REQUIRE_CUDA=1`` (then missing artifacts fail)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from hsn.config import load_config
from hsn.device import resolve, seed_everything
from hsn.kernels import window
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from hsn.train import compile as compile_mod
from hsn.train.amp import autocast_ctx
from hsn.train.compile import (
    FOLD_ROUND,
    FOLD_SCORES,
    SWEEP_FOLD,
    TAPE_KEYS,
    CompiledWindow,
    compile_window,
    sweep_merge_error,
)
from torch import Tensor

pytestmark = pytest.mark.cuda

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs/experiments/hsn_recipe_a6.yaml"


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


def _bits(x: Tensor) -> Tensor:
    view = {torch.float32: torch.int32, torch.bfloat16: torch.int16}.get(x.dtype)
    return x.contiguous() if view is None else x.contiguous().view(view)


def _same(a: Tensor, b: Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(_bits(a), _bits(b)))


# --------------------------------------------------------------------------------------------------
# The folded forms of the two glue kernels
# --------------------------------------------------------------------------------------------------


def _recipe_model(device: torch.device, *overrides: str) -> HSN:
    cfg = load_config(RECIPE, overrides=list(overrides))
    seed_everything(0)
    model = HSN.from_config(cfg).to(device)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.2 * torch.randn(p.shape, generator=g).to(device))
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("extra", [(), ("model.age_max=2",), ("model.n_units=3",)])
def test_select_fold_takes_the_decision_of_select_tape(
    device: torch.device, dtype: torch.dtype, extra: tuple[str, ...]
) -> None:
    """``select_fold`` on the scorer's wide product (the logits in its first columns, fp32 or
    bf16): the decision of ``select_tape`` on the logits sliced out and cast to fp32, with and
    without refresh probes, and with a unit count that is not a power of two (the kernel then
    loops over a cell's units instead of reducing them by shuffles)."""
    model = _recipe_model(device, *extra)
    cfg = model.cfg
    router = model.router
    B, n, N, M, K, d = 5, 3, cfg.n_cells, cfg.n_units, cfg.k_select, cfg.d
    g = torch.Generator().manual_seed(3)
    age = torch.randint(0, 5, (B, n + 1, N), generator=g).to(device)
    common = {
        "age_max": int(router.age_max),
        "probes": True,
        "lambda_cost": float(router.lambda_cost),
        "cell_overhead": float(router.cell_overhead),
        "shared_cost": float(router.shared_cost),
        "limit": float(cfg.budget) * 1.00001,
    }

    def outputs() -> list[Tensor]:
        return [
            torch.zeros(B, n, K, dtype=torch.int64, device=device),
            torch.zeros(B, n, K, dtype=torch.int64, device=device),
            torch.zeros(B, n, K, dtype=torch.bool, device=device),
            torch.zeros(B, n, device=device),
            torch.zeros(B, n, K, dtype=torch.bool, device=device),
        ]

    want, got = outputs(), outputs()
    for t in range(n):
        scored = torch.randn(B * N, M + d, generator=g).to(device=device, dtype=dtype)
        if t == 1:  # ties between cells and between a cell's units
            scored[:, :M] = scored[0, 0]
        logits = scored[:, :M].reshape(B, N, M).to(torch.float32)
        window.select_tape(logits, age, router.cost_table, *want, t=t, **common)
        window.select_fold(scored, age, router.cost_table, *got, t=t, n_units=M, **common)
        for x, y in zip(want, got, strict=True):
            assert _same(x, y), t
    if "model.age_max=2" in extra:
        assert int(got[4].sum()) > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cand_norm", ["rms", "none"])
def test_update_fold_writes_what_update_tape_writes(
    device: torch.device, dtype: torch.dtype, cand_norm: str
) -> None:
    """``update_fold`` on the products as they are (``y`` in the products' dtype, the gate as a
    strided view of a wider product): the tape of ``update_tape`` on their casts to fp32, bit
    for bit, and the next position's row ``[z; q; context]`` in the first product's input,
    rounded to its dtype (``z`` in the products' dtype, the context in fp32, as in the sweep);
    the first row, which the block assembles, is not touched."""
    model = _recipe_model(device, f"model.cand_norm={cand_norm}")
    cfg = model.cfg
    cell = model.cell
    B, n, N, K, d = 5, 3, cfg.n_cells, cfg.k_select, cfg.d
    d_z, d_c = d, 48  # another width for the context
    g = torch.Generator().manual_seed(9)

    def tapes() -> list[Tensor]:
        gen = torch.Generator().manual_seed(2)
        tape_h = (torch.rand(B, n + 1, N, d, generator=gen) * 1.6 - 0.8).to(device)
        tape_q = torch.randn(B, n + 1, d, generator=gen).to(device)
        tape_age = torch.randint(0, 9, (B, n + 1, N), generator=gen).to(device)
        return [tape_h, tape_q, tape_age]

    cells = torch.stack(
        [torch.stack([torch.randperm(N, generator=g)[:K] for _ in range(n)]) for _ in range(B)]
    ).to(device)
    valid = (torch.rand(B, n, K, generator=g) > 0.2).to(device)
    reset = (torch.rand(B, n, generator=g) < 0.4).to(device)
    want, got = tapes(), tapes()
    extra_want = [torch.zeros(B, n, K, d, device=device), torch.zeros(B, n, device=device)]
    extra_want.append(torch.zeros(B, n, d, device=device))
    extra_got = [torch.zeros_like(x) for x in extra_want]
    z = torch.randn(B, n, d_z, generator=g).to(device=device, dtype=dtype)
    context = torch.randn(B, n, d_c, generator=g).to(device)
    x = torch.randn(n, B, d_z + d + d_c, generator=g).to(device=device, dtype=dtype)
    before = x.clone()
    for t in range(n):
        y = torch.randn(B, K, d, generator=g).to(device=device, dtype=dtype)
        wide = torch.randn(B * K, 5 * d, generator=g).to(device=device, dtype=dtype)
        gate = wide[:, 4 * d :]  # the rider columns: a strided view
        assert not gate.is_contiguous()
        window.update_tape(
            cell, y.float(), gate.float().reshape(B, K, d), *want, cells, valid, reset,
            *extra_want, t=t,
        )  # fmt: skip
        window.update_fold(cell, y, gate, *got, cells, valid, reset, *extra_got, x, z, context, t=t)
        for a, b in zip([*want, *extra_want], [*got, *extra_got], strict=True):
            assert _same(a, b), t
    for t in range(1, n):  # the rows the kernel wrote: the summary after the next reset
        row = torch.cat([z[:, t].float(), got[1][:, t], context[:, t]], dim=-1).to(dtype)
        assert _same(x[t], row), t
    assert _same(x[0], before[0])


def test_update_fold_rounds_to_bf16_as_torch_does(device: torch.device) -> None:
    """The kernel's fp32 -> bf16 cast (round to nearest, ties to even) against torch's, on every
    upper half of an fp32 with lower halves on, next to and far from the ties: a round with no
    valid slot leaves the summary as it is, so the summaries written are the casts of the
    inputs."""
    model = _recipe_model(device)
    cfg = model.cfg
    N, K, d = cfg.n_cells, cfg.k_select, cfg.d
    lows = torch.tensor(
        [0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF, 0x4000, 0xC000], dtype=torch.int64
    )
    highs = torch.arange(1 << 16, dtype=torch.int64)
    patterns = ((highs[:, None] << 16) | lows[None, :]).reshape(-1)  # 2^19 bit patterns
    patterns = torch.where(patterns >= 1 << 31, patterns - (1 << 32), patterns).to(torch.int32)
    values = patterns.view(torch.float32)
    B = values.numel() // d
    q = values.reshape(B, d).to(device)
    finite = ~torch.isnan(q)
    tape_h = torch.zeros(B, 2, N, d, device=device)
    tape_q = torch.stack([q, torch.zeros_like(q)], dim=1).contiguous()
    tape_age = torch.zeros(B, 2, N, dtype=torch.int64, device=device)
    cells = torch.zeros(B, 1, K, dtype=torch.int64, device=device)
    valid = torch.zeros(B, 1, K, dtype=torch.bool, device=device)
    reset = torch.zeros(B, 1, dtype=torch.bool, device=device)
    y = torch.zeros(B, K, d, device=device)
    x = torch.zeros(2, B, 3 * d, dtype=torch.bfloat16, device=device)
    blank = torch.zeros(B, 2, d, device=device)
    with torch.no_grad():
        # one position of a two-position block: record 1 of x gets the summary after position 0
        window.update_fold_op(
            y, y.reshape(B * K, d), tape_h, tape_q, tape_age,
            cells.expand(B, 2, K).contiguous(), valid.expand(B, 2, K).contiguous(),
            model.cell.cand_norm.weight, reset.expand(B, 2).contiguous(),
            torch.zeros(B, 2, K, d, device=device), torch.zeros(B, 2, device=device),
            torch.zeros(B, 2, d, device=device), x, blank, blank, 0, True, 1e-6,
        )  # fmt: skip
    written = x[1, :, d : 2 * d]
    carried = tape_q[:, 1]  # q + 0: the inputs themselves, but for -0.0, which the sum makes 0.0
    assert bool(torch.equal(_bits(carried)[finite], _bits(q + 0.0)[finite]))
    assert _same(written, carried.to(torch.bfloat16))  # NaN patterns included
    assert int(((_bits(q) != _bits(carried)) & finite).sum()) == 1


# --------------------------------------------------------------------------------------------------
# The compiled sweep: folded against the fused sweep of P2.10
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocks of 4 positions: a 16-position window runs four blocks and each compile is short."""
    monkeypatch.setattr(compile_mod, "MAX_BLOCK", 4)


@pytest.fixture(autouse=True)
def _release_compiled_state() -> object:
    yield
    torch._dynamo.reset()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        import gc

        from torch._inductor.cudagraph_trees import reset_cudagraph_trees

        reset_cudagraph_trees()
        gc.collect()
        torch.cuda.empty_cache()


def _sweep_case(
    device: torch.device, precision: str, *overrides: str
) -> tuple[HSN, CompiledWindow, object]:
    cfg = load_config(
        RECIPE,
        overrides=[
            "profile=rtx3060",
            "hardware.compile_mode=reduce-overhead",
            f"train.precision={precision}",
            f"hardware.autocast_dtype={precision}",
            "train.unroll=16",
            *overrides,
        ],
    )
    ctx = resolve(cfg.hardware)
    seed_everything(0)
    model = HSN.from_config(cfg).to(ctx.device)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.2 * torch.randn(p.shape, generator=g).to(ctx.device))
    model.train()
    compiled = compile_window(model, ctx, "reduce-overhead", backend="inductor")
    assert isinstance(compiled, CompiledWindow)
    return model, compiled, ctx


@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("precision", ["bf16", "fp32"])
@pytest.mark.parametrize("extra", [(), ("model.age_max=2",)])
def test_folded_sweep_records_the_fused_sweeps_tape_bit_for_bit(
    device: torch.device, precision: str, extra: tuple[str, ...]
) -> None:
    """On the recipe's model: every fold level records the tape of the fused sweep of P2.10 bit
    for bit over three carried windows, with resets at a block's first, interior and last
    positions, with and without refresh probes, in fp32 and under bf16 autocast, and again
    after the weights moved."""
    model, compiled, ctx = _sweep_case(device, precision, *extra)
    assert sweep_merge_error(model) is None and SWEEP_FOLD == FOLD_ROUND | FOLD_SCORES
    cfg = model.cfg
    B, T = 6, 16
    g = torch.Generator().manual_seed(11)
    windows = []
    for index in range(3):
        reset = torch.rand(B, T, generator=g) < 0.05
        reset[:, 0] = index == 0
        reset[1, 3] = reset[2, 4] = reset[3, 6] = reset[4, T - 1] = True
        windows.append(
            (
                torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(device),
                torch.randint(0, cfg.n_context, (B, T), generator=g).to(device),
                reset.to(device),
            )
        )

    def sweep(fold: int | None) -> list[dict[str, Tensor]]:
        tapes = []
        state = model.init_state(B, device)
        with torch.no_grad(), autocast_ctx(ctx):  # type: ignore[arg-type]
            for index, (tokens, context, reset) in enumerate(windows):
                pro = model._window_prologue(
                    tokens, context, state, reset, model._window_mode(), reset_first=True
                )
                for _ in range(3 if index == 0 else 1):  # warm-up, recording, replay
                    tape = compiled.tape(pro, state, reset, fused=True, fold=fold)
                tapes.append({key: value.clone() for key, value in tape.items()})
                state = CellGraphState._make(
                    h=tape["h"].clone(),
                    q=tape["q"].clone(),
                    age=tape["age"].clone(),
                    pos=tape["pos"].clone(),
                )
        return tapes

    levels = (FOLD_ROUND, FOLD_SCORES, None)  # each fold alone, and the default: both
    for round_ in range(2):
        reference = sweep(0)
        for tape in reference:  # no weight the twin leaves unrefreshed was read
            assert bool(torch.isfinite(tape["tape_h"]).all() and torch.isfinite(tape["qs"]).all())
        if extra:
            assert sum(float(tape["refreshed"].sum()) for tape in reference) > 0.0
        for fold in levels:
            for got, want in zip(sweep(fold), reference, strict=True):
                for key in (*TAPE_KEYS, "h", "q", "age", "pos"):
                    assert _same(got[key], want[key]), (round_, fold, key)
        with torch.no_grad():
            for p in model.parameters():  # an optimiser step's worth of change
                p.add_(0.01 * torch.randn_like(p))
    twins = [twin for (_, fused), twin in compiled._sweep_twins.items() if fused]
    assert twins and all(twin.merged for twin in twins)
    folds = {slot[3] for slot in compiled._tape_blocks}
    assert folds == {0, 1, 2, 3}  # each level ran its own compiled block


@pytest.mark.usefixtures("small_blocks")
def test_folded_sweep_on_another_shape_in_fp32(device: torch.device) -> None:
    """A small model (three units, so the selection kernel loops over a cell's units instead of
    reducing them by shuffles; two sources per cell; a width that is not a warp's multiple) in
    fp32: the fused sweep's decisions, ages and positions bit for bit, and its values to fp32
    rounding. Not bit for bit here: at this width the compiler chooses among several layouts
    for the round's norm (a sum over 32 values) by timing them, for every compiled block anew,
    so the last place of that sum differs between two compilations of the same sweep, with or
    without the folds (at the recipe's width, a sum over 256 values, it has one layout: the
    test above)."""
    small = (
        "model.d=8",
        "model.context_dim=8",
        "model.n_cells=6",
        "model.in_degree=2",
        "model.n_units=3",
        "model.k_select=2",
        "model.hidden=16",
    )
    model, compiled, ctx = _sweep_case(device, "fp32", *small)
    assert sweep_merge_error(model) is None
    cfg = model.cfg
    B, T = 5, 16
    g = torch.Generator().manual_seed(13)
    tokens = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(device)
    context = torch.randint(0, cfg.n_context, (B, T), generator=g).to(device)
    reset = torch.rand(B, T, generator=g) < 0.1
    reset[:, 0] = True
    reset = reset.to(device)
    state = model.init_state(B, device)
    tapes = {}
    with torch.no_grad(), autocast_ctx(ctx):  # type: ignore[arg-type]
        pro = model._window_prologue(
            tokens, context, state, reset, model._window_mode(), reset_first=True
        )
        for fold in (0, None):
            for _ in range(3):  # warm-up, recording, replay
                tape = compiled.tape(pro, state, reset, fused=True, fold=fold)
            tapes[fold] = {key: value.clone() for key, value in tape.items()}
    for key in (*TAPE_KEYS, "h", "q", "age", "pos"):
        got, want = tapes[None][key], tapes[0][key]
        if got.dtype.is_floating_point:
            assert float((got - want).abs().max()) <= 2e-6, key
        else:
            assert torch.equal(got, want), key
    assert bool(torch.isfinite(tapes[0]["tape_h"]).all())
    assert {slot[3] for slot in compiled._tape_blocks} == {0, SWEEP_FOLD}
