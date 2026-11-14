"""P2.10c: the window kernels with amendment A8's fast cells against their PyTorch references:
the selection kernel's reserved slots (:meth:`Router.select`'s ``reserved`` and
``reserved_units``), the fast read in the folded end-of-round kernel's row, and the fused and
folded sweeps' tapes against the split sweep's for models with fast cells, with fixed units,
with the fast read and with A9's address n-grams. CUDA-marked; skipped without a card or without
the built artifacts unless ``HSN_REQUIRE_CUDA=1`` (then missing artifacts fail)."""

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
from hsn.train.compile import CompiledWindow, compile_window, sweep_merge_error
from torch import Tensor

pytestmark = pytest.mark.cuda

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/experiments/hsn_recipe_a6.yaml"

#: the approved shape of 13 Nov 2026, and its parts
FAST = ("model.n_cells=64", "model.k_select=20", "model.fast_cells=8")
PINNED = (*FAST, "model.fast_units=fixed")
READ = (*PINNED, "model.fast_readout=true")


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
# The selection kernel's reserved slots against Router.select
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("reserved", [1, 2, 3])
@pytest.mark.parametrize(
    "overrides",
    [
        (),
        ("model.age_max=2",),  # the refresh probes fire: they never take a reserved slot
        ("model.budget=0.1",),  # the budget cuts the prefix, reserved slots included
        ("model.age_max=2", "model.budget=0.35"),
        ("model.lambda_cost=0.05", "model.age_max=3"),
    ],
)
def test_select_kernel_reserves_the_fast_slots(
    device: torch.device, overrides: tuple[str, ...], reserved: int, fixed: bool
) -> None:
    """``hsn_win_select`` with reserved cells returns :meth:`Router.select`'s decision with
    ``reserved`` (and ``reserved_units``): the reserved cells in the first slots, the scored
    cells behind them, the threshold among the scored slots, probes that leave the reserved
    slots alone; on random scores, exact ties and stale cells."""
    from hsn.model.router import BUDGET_SLACK
    from test_a6_feedback_detach import A5, ALL, _cfg, _model

    cfg = _cfg(*A5, ALL, "model.cand_norm=rms", "model.k_select=4", *overrides)
    router = _model(cfg).router.to(device=device, dtype=torch.float32)
    N, M, K = cfg.model.n_cells, cfg.model.n_units, cfg.model.k_select
    g = torch.Generator().manual_seed(7)
    fired = cut = 0
    for trial in range(40):
        B = 1 + trial % 5
        logits = torch.randn(B, N, M, generator=g)
        if trial % 2:
            logits = (logits * 2).round() / 2  # exact ties between cells and between units
        age = torch.randint(0, 2 * int(router.age_max) + 2, (B, N), generator=g)
        if trial % 3 == 0:
            age = (age // 3) * 3  # equal ages among the stale cells
        logits, age = logits.to(device), age.to(device)
        for probes in (True, False):
            want = router.select(
                logits,
                cfg.model.budget,
                K,
                cfg.model.k_min,
                "topk",
                gate=None,
                age=age if probes else None,
                reserved=reserved,
                reserved_units=fixed,
            )
            cells, units, valid, theta, refreshed = window.select(
                logits,
                age,
                router.cost_table,
                k=K,
                age_max=int(router.age_max),
                probes=probes,
                lambda_cost=float(router.lambda_cost),
                cell_overhead=float(router.cell_overhead),
                shared_cost=float(router.shared_cost),
                limit=float(cfg.model.budget) * (1.0 + BUDGET_SLACK),
                reserved=reserved,
                reserved_units=fixed,
            )
            assert torch.equal(valid, want.valid)
            assert torch.equal(cells, want.cell_idx)
            assert torch.equal(units, want.unit_idx)
            assert torch.equal(refreshed, want.refreshed)
            assert want.threshold is not None
            assert torch.equal(theta, want.threshold)
            assert not bool(refreshed[:, :reserved].any())
            fired += int(refreshed.sum())
            cut += int((~valid[:, :reserved]).sum())
    if any("age_max" in o for o in overrides):
        assert fired > 0  # the probes were exercised
    if "model.budget=0.1" in overrides and reserved > 1:
        assert cut > 0  # the budget cut a reserved slot


def test_select_kernel_without_reserved_cells_is_the_kernel_of_p2_10(device: torch.device) -> None:
    """``reserved = 0`` named explicitly is the call without it."""
    g = torch.Generator().manual_seed(3)
    B, N, M, K = 4, 32, 8, 4
    logits = torch.randn(B, N, M, generator=g).to(device)
    age = torch.randint(0, 9, (B, N), generator=g).to(device)
    cost = torch.rand(M, generator=g).to(device)
    common = {
        "k": K,
        "age_max": 4,
        "probes": True,
        "lambda_cost": 0.0,
        "cell_overhead": 0.01,
        "shared_cost": 0.02,
        "limit": 1.0,
    }
    want = window.select(logits, age, cost, **common)
    got = window.select(logits, age, cost, **common, reserved=0, reserved_units=False)
    for a, b in zip(got, want, strict=True):
        assert _same(a, b)


# --------------------------------------------------------------------------------------------------
# The fast read in the folded end-of-round kernel's row
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
@pytest.mark.parametrize("n_fast", [1, 4, 8])
def test_update_fold_writes_the_fast_read(
    device: torch.device, dtype: torch.dtype, n_fast: int
) -> None:
    """``update_fold`` with ``n_fast``: the tape is the one without it, the row's ``[z; q;
    context]`` is the one without it, and the row ends with the mean of the first ``n_fast``
    cells of the state the next position reads (zero after a reset), in the row's dtype."""
    model = _recipe_model(device, "model.n_cells=64", "model.k_select=12")
    cfg = model.cfg
    cell = model.cell
    B, n, N, K, d = 5, 4, cfg.n_cells, cfg.k_select, cfg.d
    d_z, d_c = d, 48
    g = torch.Generator().manual_seed(9)

    def tapes() -> list[Tensor]:
        gen = torch.Generator().manual_seed(2)
        tape_h = (torch.rand(B, n + 1, N, d, generator=gen) * 1.6 - 0.8).to(device)
        tape_q = torch.randn(B, n + 1, d, generator=gen).to(device)
        tape_age = torch.randint(0, 9, (B, n + 1, N), generator=gen).to(device)
        return [tape_h, tape_q, tape_age]

    # cells 0 and 1 in the first two slots at every position (written fast cells), the other
    # slots among the remaining cells: a row's valid slots name distinct cells
    cells = torch.stack(
        [
            torch.stack(
                [
                    torch.cat([torch.arange(2), 2 + torch.randperm(N - 2, generator=g)[: K - 2]])
                    for _ in range(n)
                ]
            )
            for _ in range(B)
        ]
    )
    cells[0, 1, 0], cells[0, 1, 1] = 1, 0  # in any slot order
    cells = cells.to(device)
    valid = torch.rand(B, n, K, generator=g) > 0.2
    valid[:, :, :2] = True
    valid[1, 2, 0] = False  # an unwritten fast cell: the row reads its carried state
    valid = valid.to(device)
    reset = (torch.rand(B, n, generator=g) < 0.4).to(device)
    want, got = tapes(), tapes()
    extra_want = [torch.zeros(B, n, K, d, device=device), torch.zeros(B, n, device=device)]
    extra_want.append(torch.zeros(B, n, d, device=device))
    extra_got = [torch.zeros_like(x) for x in extra_want]
    z = torch.randn(B, n, d_z, generator=g).to(device=device, dtype=dtype)
    context = torch.randn(B, n, d_c, generator=g).to(device)
    x_plain = torch.randn(n, B, d_z + d + d_c, generator=g).to(device=device, dtype=dtype)
    x_fast = torch.randn(n, B, d_z + d + d_c + d, generator=g).to(device=device, dtype=dtype)
    before = x_fast.clone()
    for t in range(n):
        y = torch.randn(B, K, d, generator=g).to(device=device, dtype=dtype)
        gate = torch.randn(B * K, d, generator=g).to(device=device, dtype=dtype)
        window.update_fold(
            cell, y, gate, *want, cells, valid, reset, *extra_want, x_plain, z, context, t=t
        )
        window.update_fold(
            cell, y, gate, *got, cells, valid, reset, *extra_got, x_fast, z, context,
            t=t, n_fast=n_fast,
        )  # fmt: skip
        for a, b in zip([*want, *extra_want], [*got, *extra_got], strict=True):
            assert _same(a, b), t
    tape_h = got[0]
    for t in range(1, n):
        assert _same(x_fast[t][:, : d_z + d + d_c], x_plain[t]), t
        mean = tape_h[:, t, :n_fast].sum(dim=1) / n_fast  # the state position t reads
        wrote = x_fast[t][:, d_z + d + d_c :].float()
        tolerance = 2e-6 if dtype == torch.float32 else 1e-2
        assert float((wrote - mean).abs().max()) <= tolerance, t
        cleared = reset[:, t]
        assert bool((wrote[cleared] == 0).all())
    assert _same(x_fast[0], before[0])
    with pytest.raises(ValueError, match="x must be"):
        window.update_fold(
            cell, y, gate, *got, cells, valid, reset, *extra_got, x_plain, z, context,
            t=0, n_fast=n_fast,
        )  # fmt: skip


# --------------------------------------------------------------------------------------------------
# The compiled sweeps: fused and folded against the split sweep, with fast cells
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


def _windows(model: HSN, device: torch.device, B: int, T: int) -> list[tuple[Tensor, ...]]:
    cfg = model.cfg
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
    return windows


def _sweep(
    model: HSN,
    compiled: CompiledWindow,
    ctx: object,
    windows: list[tuple[Tensor, ...]],
    *,
    fused: bool,
    fold: int | None = None,
) -> list[dict[str, Tensor]]:
    """Three carried windows through one sweep (warm-up, recording and replay on the first)."""
    tapes = []
    B = int(windows[0][0].shape[0])
    state = model.init_state(B, windows[0][0].device)
    with torch.no_grad(), autocast_ctx(ctx):  # type: ignore[arg-type]
        for index, (tokens, context, reset) in enumerate(windows):
            pro = model._window_prologue(
                tokens, context, state, reset, model._window_mode(), reset_first=True
            )
            for _ in range(3 if index == 0 else 1):
                tape = compiled.tape(pro, state, reset, fused=fused, fold=fold)
            tapes.append({key: value.clone() for key, value in tape.items()})
            state = CellGraphState._make(
                h=tape["h"].clone(), q=tape["q"].clone(), age=tape["age"].clone(),
                pos=tape["pos"].clone(),
            )  # fmt: skip
    return tapes


SHAPES = {
    "fast_routed_units": FAST,
    "fast_fixed_units": PINNED,
    "fast_read": READ,
    "fast_read_probes": (*READ, "model.age_max=2"),
    "fast_read_ngram2": (*READ, "model.address_ngram=2"),
    "routed_ngram2": ("model.n_cells=64", "model.k_select=12", "model.address_ngram=2"),
}

#: A decision counts as settled when its scores are this far apart. Two sweeps compute one
#: score through different matrix products and agree to rounding (a few units in the last
#: place: about 1e-5 at these scores of about 18), so a decision between closer scores may go
#: either way, after which the row's two trajectories differ until its next reset. Measured on
#: the first of these cases: two cells with different states whose scores are one fp32 value
#: in the split sweep and one unit in the last place apart in the fused sweep, at the cut.
MARGIN = 1e-4


def _settled(
    model: HSN, windows: list[tuple[Tensor, ...]], tapes: list[dict[str, Tensor]]
) -> list[tuple[Tensor, Tensor]]:
    """Per window, ``(before [B, T], at [B, T])``: whether a row's state before position ``t``,
    and its decision at ``t``, are settled, that is, reached through decisions with a margin
    only (the scores are recomputed from the reference tape's pre-round states). A row that
    met an unsettled decision counts again from its next reset, which zeroes its state."""
    from hsn.train.split import _scorer_logits

    cfg = model.cfg
    F = cfg.fast_cells
    routed = cfg.k_select - F
    fixed = cfg.fast_units == "fixed"
    device = windows[0][0].device
    B = int(windows[0][0].shape[0])
    state = model.init_state(B, device)
    diverged = torch.zeros(B, dtype=torch.bool, device=device)
    out = []
    for (tokens, context, reset), tape in zip(windows, tapes, strict=True):
        with torch.no_grad():
            pro = model._window_prologue(
                tokens, context, state, reset, model._window_mode(), reset_first=True
            )
            _, T, N, d = (int(n) for n in tape["tape_h"].shape)
            h = tape["tape_h"].reshape(B * T, N, d)
            q = tape["tape_q"].reshape(B * T, d)
            z = pro.z.reshape(B * T, d)
            u = model.shared(model._shared_input(z, q, pro.context.reshape(B * T, -1), h))
            eps = z - model.predictor(q)
            logits = _scorer_logits(
                model.scorer, u, eps, h, tape["tape_age"].reshape(B * T, N),
                pro.scalars.reshape(B * T, N, -1),
            )  # fmt: skip
            best = model.router.scores(logits).topk(2, dim=-1).values  # [R, N, 2]
            slow = best[:, F:, 0].sort(dim=1, descending=True).values
            cut = (
                slow[:, routed - 1] - slow[:, routed]
            )  # the last scored slot against the next cell
            cells = tape["sel_cell_idx"].reshape(B * T, -1)
            chooses = torch.zeros(B * T, N, dtype=torch.bool, device=device)
            chooses.scatter_(1, cells, tape["sel_valid"].reshape(B * T, -1))
            if fixed:  # a fixed slot's unit is not a decision
                chooses[:, :F] = False
            gap = best[..., 0] - best[..., 1]
            unit = torch.where(chooses, gap, torch.full_like(gap, float("inf"))).amin(dim=1)
            risky = ~((cut > MARGIN) & (unit > MARGIN)).reshape(B, T)
        before = torch.zeros(B, T, dtype=torch.bool, device=device)
        at = torch.zeros(B, T, dtype=torch.bool, device=device)
        for t in range(T):
            diverged = diverged & ~reset[:, t]
            before[:, t] = ~diverged
            diverged = diverged | risky[:, t]
            at[:, t] = ~diverged
        out.append((before, at))
        state = CellGraphState._make(h=tape["h"], q=tape["q"], age=tape["age"], pos=tape["pos"])
    return out


#: The tape's records by what they depend on: the state before a position, or its decision.
BEFORE = ("tape_h", "tape_q", "tape_age")
AT = ("tape_hsel", "tape_theta", "qs", "sel_cell_idx", "sel_unit_idx", "sel_valid")


@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("name", list(SHAPES))
def test_fused_sweeps_record_the_split_sweeps_tape_in_fp32(device: torch.device, name: str) -> None:
    """fp32, the recipe's widths at 64 cells: the fused sweep (the selection kernel with its
    reserved slots, the per-slot products, the fast read in the first product's input) and
    the folded sweep (the fast read written by the end-of-round kernel) record the tape of the
    split sweep, which runs :meth:`HSN._position`: wherever the decisions so far had a margin
    (:func:`_settled`), every decision and age identical and the values to the rounding of two
    compilations of one arithmetic. Three carried windows with resets at a block's first,
    interior and last positions."""
    import test_p2_8_compile as p28

    model, compiled, ctx = _sweep_case(device, "fp32", *SHAPES[name])
    assert sweep_merge_error(model) is None
    cfg = model.cfg
    windows = _windows(model, device, 6, 16)
    reference = _sweep(model, compiled, ctx, windows, fused=False)
    settled = _settled(model, windows, reference)
    share = float(torch.stack([at.float().mean() for _, at in settled]).mean())
    assert share > 0.5, share  # the comparison covers most of the windows
    tol = p28.TOL[("cuda", "fp32")]["values"]
    F = cfg.fast_cells
    for tape in reference:
        assert bool(torch.isfinite(tape["tape_h"]).all())
        if F:
            fast = torch.arange(F, device=device).expand(6, 16, F)
            assert torch.equal(tape["sel_cell_idx"][:, :, :F], fast)
            if cfg.fast_units == "fixed":
                assert torch.equal(tape["sel_unit_idx"][:, :, :F], fast % cfg.n_units)
    if "model.age_max=2" in SHAPES[name]:
        assert sum(float(tape["refreshed"].sum()) for tape in reference) > 0.0
    for fold in (0, None):  # the fused sweep of P2.10, and the folded sweep
        tapes = _sweep(model, compiled, ctx, windows, fused=True, fold=fold)
        for got, want, (before, at) in zip(tapes, reference, settled, strict=True):
            for key, where in [(k, before) for k in BEFORE] + [(k, at) for k in AT]:
                mine, theirs = got[key][where], want[key][where]
                if mine.dtype.is_floating_point:
                    scale = max(1.0, float(theirs.abs().max())) if key == "tape_theta" else 1.0
                    gap = float((mine - theirs).abs().max())
                    assert gap <= tol * scale, (name, fold, key, gap)
                else:
                    assert torch.equal(mine, theirs), (name, fold, key)
    twins = [twin for (_, fused), twin in compiled._sweep_twins.items() if fused]
    assert twins and all(twin.merged for twin in twins)
    expected = F if cfg.fast_units == "fixed" else 0
    assert all(twin.fast_slots == expected for twin in compiled._sweep_twins.values())
    print(f"{name}: {share:.0%} of the positions settled and compared")


@pytest.mark.usefixtures("small_blocks")
def test_folded_sweep_with_the_fast_read_under_bf16(device: torch.device) -> None:
    """bf16 autocast, the approved shape: the folded sweep against the fused sweep of P2.10.
    The two differ in one thing here, who computes the fast read ``f`` (the block's compiled
    mean or the end-of-round kernel), so their tapes agree to that value's rounding: finite,
    the fast slots in place, and the first position of the first window, where no ``f`` of the
    kernel's has been read yet, identical."""
    model, compiled, ctx = _sweep_case(device, "bf16", *READ)
    cfg = model.cfg
    windows = _windows(model, device, 6, 16)
    fused = _sweep(model, compiled, ctx, windows, fused=True, fold=0)
    folded = _sweep(model, compiled, ctx, windows, fused=True, fold=None)
    F = cfg.fast_cells
    fast = torch.arange(F, device=device).expand(6, 16, F)
    same_cells = []
    for got, want in zip(folded, fused, strict=True):
        assert bool(torch.isfinite(got["tape_h"]).all() and torch.isfinite(got["qs"]).all())
        assert torch.equal(got["sel_cell_idx"][:, :, :F], fast)
        assert torch.equal(got["sel_unit_idx"][:, :, :F], fast % cfg.n_units)
        same_cells.append(float((got["sel_cell_idx"] == want["sel_cell_idx"]).float().mean()))
    for key in ("sel_cell_idx", "sel_unit_idx", "sel_valid"):
        assert torch.equal(folded[0][key][:, 0], fused[0][key][:, 0]), key
    assert _same(folded[0]["tape_hsel"][:, 0], fused[0]["tape_hsel"][:, 0])
    print(f"folded against fused under bf16, share of identical cells per window: {same_cells}")
