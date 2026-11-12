"""P2.1 tests: the cost table from a P0.6 cost profile, the ``Router``'s five selection modes
(budget, bounds, determinism, randomness), the ``scorer`` profiler segment, the deadline helper,
the config fields, and the static check that routing outputs carry no authority."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, get_type_hints

import pytest
import torch
from hsn.config import (
    REPO_ROOT,
    ModelConfig,
    RunConfig,
    dump_resolved,
    load_config,
    load_hardware_profile,
)
from hsn.device import DeviceContext, resolve
from hsn.kernels.costs import (
    COST_PROFILE_KEY,
    COST_PROFILE_SEGMENTS,
    UNIFORM_SOURCE,
    CostTable,
    cost_table_from_config,
    load_cost_table,
    profile_dense_costs,
    write_cost_profile,
)
from hsn.model.hsn import HSN
from hsn.model.router import BUDGET_SLACK, SELECTION_MODES, Router, Scorer, Selection
from hsn.profiler import Profiler
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
TINY = REPO / "configs" / "tiny_gradcheck.yaml"

cuda_param = pytest.param("cuda", marks=pytest.mark.cuda)
mps_param = pytest.param("mps", marks=pytest.mark.mps)
DEVICES = ["cpu", cuda_param, mps_param]

SPARSE_MODES = ("topk", "greedy_budget", "random", "fixed")
TOLERANCE = 1.0 + 2 * BUDGET_SLACK  # fp32 accounting on the device plus the documented slack


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _device(device_type: str) -> torch.device:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device on this machine")
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no mps device on this machine")
    return torch.device(device_type)


def _cpu_ctx() -> DeviceContext:
    return resolve(load_hardware_profile("cpu"))


def _config(path: Path, overrides: Sequence[str] = ()) -> RunConfig:
    return load_config(path, overrides=list(overrides))


def _scorer(cfg: RunConfig, seed: int = 0) -> Scorer:
    torch.manual_seed(seed)
    scorer = Scorer(cfg.model, unroll=cfg.train.unroll)
    return scorer.double() if cfg.train.precision == "fp64" else scorer


def _model(path: Path) -> tuple[RunConfig, HSN]:
    cfg = _config(path)
    torch.manual_seed(cfg.seed)
    model = HSN.from_config(cfg)
    if cfg.train.precision == "fp64":
        model = model.double()
    return cfg, model


def _table(
    n_units: int,
    n_cells: int,
    seed: int,
    *,
    shared: float | None = None,
    overhead: float | None = None,
    unit: Sequence[float] | None = None,
    scorer: float = 0.01,
) -> CostTable:
    """A hand-built, non-uniform table (the P0.6 report cannot produce one; tests need them)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if unit is None:
        unit_costs = (
            0.01 + 0.09 * torch.rand(n_units, generator=generator, dtype=torch.float64)
        ).tolist()
    else:
        unit_costs = list(unit)
    shared_cost = float(torch.rand(1, generator=generator)) * 0.3 if shared is None else shared
    cell_overhead = (
        float(torch.rand(1, generator=generator)) * 0.05 if overhead is None else overhead
    )
    record = json.dumps([unit_costs, shared_cost, cell_overhead, scorer, n_units, n_cells])
    return CostTable(
        unit=torch.tensor(unit_costs, dtype=torch.float64),
        cell_overhead=cell_overhead,
        shared=shared_cost,
        scorer=scorer,
        n_units=n_units,
        n_cells=n_cells,
        dense_ms_per_position=None,
        source=f"test table {seed}",
        hash=hashlib.blake2b(record.encode(), digest_size=32).hexdigest(),
    )


def _logits(B: int, N: int, M: int, seed: int, dtype: torch.dtype = torch.float32) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(B, N, M, generator=generator).to(dtype)


def _exact_cost(sel: Selection, table: CostTable) -> list[float]:
    """The executed cost of every row recomputed in Python floats from the table."""
    costs = []
    for b in range(sel.cell_idx.shape[0]):
        total = table.shared
        for k in range(sel.k):
            if bool(sel.valid[b, k]):
                total += table.cell_overhead + float(table.unit[int(sel.unit_idx[b, k])])
        costs.append(total)
    return costs


def _check_selection(
    sel: Selection, table: CostTable, *, k_max: int, k_min: int, budget: float | None
) -> None:
    """The invariants every sparse selection satisfies."""
    B = sel.logits.shape[0]
    N, M = table.n_cells, table.n_units
    assert sel.k == k_max
    assert sel.cell_idx.dtype == sel.unit_idx.dtype == torch.int64
    assert sel.valid.dtype == torch.bool and sel.alpha.dtype == torch.float32
    assert sel.cost.dtype == torch.float32 and sel.cost.shape == (B,)
    assert sel.n_rounds_used == 1 and sel.scorer_cost == table.scorer
    assert int(sel.cell_idx.min()) >= 0 and int(sel.cell_idx.max()) < N
    assert int(sel.unit_idx.min()) >= 0 and int(sel.unit_idx.max()) < M
    n_valid = sel.n_valid()
    assert bool((n_valid >= k_min).all()) and bool((n_valid <= k_max).all()), (
        n_valid,
        k_min,
        k_max,
    )
    for b in range(B):
        cells = sel.cell_idx[b][sel.valid[b]].tolist()
        assert len(set(cells)) == len(cells), "a cell was selected twice"
        padded = ~sel.valid[b]
        assert bool((sel.cell_idx[b][padded] == 0).all()) and bool(
            (sel.unit_idx[b][padded] == 0).all()
        )
        assert bool((sel.alpha[b][padded] == 0).all()) and bool(
            (sel.alpha[b][sel.valid[b]] == 1).all()
        )
        # the valid slots come first (a prefix) in every mode
        first_pad = int(padded.int().argmax()) if bool(padded.any()) else k_max
        assert not bool(sel.valid[b][first_pad:].any())
    exact = _exact_cost(sel, table)
    assert torch.allclose(
        sel.cost.cpu().double(), torch.tensor(exact, dtype=torch.float64), rtol=1e-5, atol=1e-6
    )
    if budget is not None:
        assert all(c <= budget * TOLERANCE for c in exact), (exact, budget)


def _report(
    *,
    n_cells: int,
    n_units: int,
    batch: int,
    positions: int,
    ms: dict[str, float],
    block_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A hand-written cost profile in the P0.6 report shape."""
    tokens = batch * positions
    segments: dict[str, Any] = {}
    for name in COST_PROFILE_SEGMENTS:
        entries = 2 * positions if name == "messages" else positions
        segments[name] = {
            "n": entries,
            "wall_ms_p50": ms[name] / entries,
            "wall_ms_p95": ms[name] / entries,
            "device_ms": ms[name],
            "peak_bytes": 1024,
            "reserved_bytes": 2048,
            "launches": -1,
            "energy_j": None,
            "energy_components": {},
            "tokens": tokens,
        }
    block: dict[str, Any] = {
        "schema": 1,
        "producer": "hsn.kernels.costs.profile_dense_costs",
        "n_cells": n_cells,
        "n_units": n_units,
        "d": 4,
        "hidden": 8,
        "batch": batch,
        "positions": positions,
        "profile": "cpu",
        "device": "cpu",
        "dtype": "torch.float32",
    }
    block.update(block_extra or {})
    return {
        "profile": "cpu",
        "device": "cpu",
        "timing_method": "synchronise + wall clock",
        "launches_method": "disabled",
        "power": {
            "method": "none",
            "interval_ms": None,
            "includes_host": None,
            "n_readings": 5 * positions,
            "n_samples": 0,
            "n_unresolved": 5 * positions,
            "notes": "",
        },
        "segments": segments,
        COST_PROFILE_KEY: block,
    }


def _write(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------------------
# The chunk's tests
# --------------------------------------------------------------------------------------------------


def test_p2_1_budget_never_exceeded() -> None:
    cfg = _config(REF_A1)
    scorer = _scorer(cfg)
    N, M = cfg.model.n_cells, cfg.model.n_units
    generator = torch.Generator(device="cpu").manual_seed(2021)
    n_cases = 0
    for trial in range(60):
        table = _table(M, N, seed=trial)
        router = Router(scorer, table, lambda_cost=2.0 * float(torch.rand(1, generator=generator)))
        B = int(torch.randint(1, 5, (1,), generator=generator))
        logits = torch.randn(B, N, M, generator=generator)
        k_max = int(torch.randint(1, N + 1, (1,), generator=generator))
        k_min = int(torch.randint(0, k_max + 1, (1,), generator=generator))
        floor = table.shared + k_min * table.max_pair_cost()
        ceiling = max(floor, table.dense_cost())
        budget = floor + float(torch.rand(1, generator=generator)) * (ceiling - floor)
        for mode in ("topk", "greedy_budget", "random"):
            draw = torch.Generator(device="cpu").manual_seed(trial) if mode == "random" else None
            sel = router.select(logits, budget, k_max, k_min, mode, generator=draw)
            _check_selection(sel, table, k_max=k_max, k_min=k_min, budget=budget)
            n_cases += 1
        affordable = int((budget * (1 + BUDGET_SLACK) - table.shared) // table.max_pair_cost())
        length = min(k_max, affordable)
        if length >= max(k_min, 1):
            fixed = torch.randperm(N, generator=generator)[:length]
            sel = router.select(logits, budget, k_max, k_min, "fixed", fixed=fixed)
            _check_selection(sel, table, k_max=k_max, k_min=k_min, budget=budget)
            assert bool((sel.cell_idx[:, :length] == fixed[None, :]).all())
            assert int(sel.n_valid().min()) == length
            n_cases += 1
    assert n_cases >= 180

    # The statically infeasible cases raise before any selection: k_min cells at the dearest
    # unit do not fit, or the fixed route does not fit.
    table = _table(M, N, seed=7, shared=0.2, overhead=0.01, unit=[0.02] * (M - 1) + [0.08])
    router = Router(scorer, table, lambda_cost=0.0)
    logits = _logits(2, N, M, seed=1)
    with pytest.raises(ValueError, match="cannot hold k_min = 3"):
        router.select(logits, 0.2 + 2 * 0.09, 4, 3, "topk")
    with pytest.raises(ValueError, match="cannot hold the fixed route = 3"):
        router.select(logits, 0.2 + 2 * 0.09, 4, 1, "fixed", fixed=torch.tensor([0, 1, 2]))
    # ... while k_min cells that fit whatever their unit are always selected
    sel = router.select(logits, 0.2 + 3 * 0.09, 4, 3, "greedy_budget")
    assert int(sel.n_valid().min()) >= 3

    # A budget that holds k_max cells at the dearest unit makes greedy_budget and topk agree
    # exactly: the pair walk takes every cell's best unit in cell order.
    for seed in range(5):
        logits = _logits(3, N, M, seed=seed)
        top = router.select(logits, 10.0, 6, 0, "topk")
        greedy = router.select(logits, 10.0, 6, 0, "greedy_budget")
        assert torch.equal(top.cell_idx, greedy.cell_idx)
        assert torch.equal(top.unit_idx, greedy.unit_idx)
        assert bool(top.valid.all()) and bool(greedy.valid.all())
        assert torch.equal(top.cost, greedy.cost)

    # greedy_budget skips a pair that does not fit and continues, so a cell whose best unit is
    # too dear can still enter with a cheaper unit (the chunk's pair walk), and topk stops at the
    # first cell of its order that does not fit.
    unit = [0.01] * M
    unit[1] = 0.30
    table = _table(M, N, seed=8, shared=0.0, overhead=0.0, unit=unit)
    router = Router(scorer, table, lambda_cost=0.0)
    logits = torch.zeros(1, N, M)
    logits[0, 5, 1] = 3.0  # cell 5 prefers the dear unit 1 ...
    logits[0, 5, 0] = 2.5  # ... but its unit 0 is the next best pair of all
    logits[0, 9, 2] = 2.0
    budget = 0.05
    greedy = router.select(logits, budget, 2, 0, "greedy_budget")
    assert greedy.cell_idx[0].tolist() == [5, 9] and greedy.unit_idx[0].tolist() == [0, 2]
    assert greedy.valid[0].tolist() == [True, True]
    top = router.select(logits, budget, 2, 0, "topk")
    assert top.cell_idx[0].tolist() == [0, 0] and top.valid[0].tolist() == [False, False]
    assert float(top.cost[0]) == 0.0

    # dense reports the dense cost whatever the budget (the one mode that may exceed it)
    dense = router.select(logits, 0.01, 2, 0, "dense")
    assert dense.k == N and bool(dense.valid.all())
    assert torch.allclose(dense.cost, torch.full((1,), table.dense_cost()))

    # all N cells at one unit each cost 1/M with the uniform table (dense runs all M units)
    uniform = CostTable.uniform(M, N)
    assert uniform.dense_cost() == pytest.approx(1.0, abs=1e-12)
    router = Router(scorer, uniform, lambda_cost=0.0)
    sel = router.select(_logits(2, N, M, seed=3), 1.0, N, N, "topk")
    assert bool(sel.valid.all()) and torch.allclose(sel.cost, torch.full((2,), 1.0 / M))
    # an exact fit: N cells whose costs sum to exactly the budget all stay valid (fp32
    # accumulation on the device never drops the last one to an ulp)
    exact_fit = _table(M, N, seed=9, shared=0.0, overhead=0.0, unit=[1.0 / N] * M)
    router = Router(scorer, exact_fit, lambda_cost=0.0)
    for mode in ("topk", "greedy_budget"):
        sel = router.select(_logits(2, N, M, seed=3), 1.0, N, N, mode)
        assert bool(sel.valid.all()) and torch.allclose(sel.cost, torch.ones(2))


def test_p2_1_k_bounds() -> None:
    for path in (TINY, REF_A1):
        cfg = _config(path)
        router = Router.from_config(cfg.model, unroll=cfg.train.unroll)
        if cfg.train.precision == "fp64":
            router = router.double()
        N, M = cfg.model.n_cells, cfg.model.n_units
        dtype = torch.float64 if cfg.train.precision == "fp64" else torch.float32
        table = router.costs
        assert table.source == UNIFORM_SOURCE
        for B in (1, 3):
            logits = _logits(B, N, M, seed=B, dtype=dtype)
            for k_max in sorted({1, 2, N // 2, N}):
                for k_min in sorted({0, 1, k_max}):
                    for mode in SPARSE_MODES:
                        kwargs: dict[str, Any] = {}
                        if mode == "fixed":
                            kwargs["fixed"] = torch.arange(max(k_min, 1)) * (N // max(k_min, 1))
                            kwargs["fixed"] = kwargs["fixed"][: max(k_min, 1)]
                        if mode == "random":
                            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(k_max)
                        sel = router.select(logits, 1.0, k_max, k_min, mode, **kwargs)
                        _check_selection(sel, table, k_max=k_max, k_min=k_min, budget=1.0)
                        assert sel.logits is logits
                        if mode != "fixed":  # budget 1.0 holds every cell: k_max selected
                            assert bool((sel.n_valid() == k_max).all())
                    dense = router.select(logits, 1.0, k_max, k_min, "dense")
                    assert dense.k == N and dense.cell_idx.shape == (B, N)
                    assert bool((dense.cell_idx == torch.arange(N)[None, :]).all())
                    assert bool(dense.valid.all()) and dense.n_rounds_used == 1
                    probs = torch.softmax(logits.float(), dim=-1)
                    assert torch.equal(dense.unit_idx, logits.float().argmax(dim=-1))
                    assert torch.allclose(dense.alpha, probs.amax(dim=-1))
                    assert torch.allclose(dense.cost, torch.ones(B))

    cfg = _config(TINY)
    router = Router.from_config(cfg.model, unroll=cfg.train.unroll)
    N, M = cfg.model.n_cells, cfg.model.n_units
    logits = _logits(2, N, M, seed=0)
    with pytest.raises(ValueError, match="k_max"):
        router.select(logits, 1.0, N + 1, 1, "topk")
    with pytest.raises(ValueError, match="k_max"):
        router.select(logits, 1.0, 0, 0, "topk")
    with pytest.raises(ValueError, match="k_min"):
        router.select(logits, 1.0, 2, 3, "topk")
    with pytest.raises(ValueError, match="k_min"):
        router.select(logits, 1.0, 2, -1, "topk")
    with pytest.raises(TypeError, match="k_max"):
        router.select(logits, 1.0, True, 1, "topk")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="budget"):
        router.select(logits, -0.1, 2, 1, "topk")
    with pytest.raises(ValueError, match="budget"):
        router.select(logits, math.nan, 2, 1, "topk")
    with pytest.raises(ValueError, match="mode"):
        router.select(logits, 1.0, 2, 1, "sparse")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fixed"):
        router.select(logits, 1.0, 2, 1, "fixed")
    with pytest.raises(ValueError, match="only used with mode='fixed'"):
        router.select(logits, 1.0, 2, 1, "topk", fixed=torch.tensor([0]))
    with pytest.raises(ValueError, match="only used with mode='random'"):
        router.select(logits, 1.0, 2, 1, "topk", generator=torch.Generator(device="cpu"))
    for bad in (
        torch.tensor([0, 0]),
        torch.tensor([0, N]),
        torch.tensor([-1]),
        torch.tensor([0, 1, 2]),
    ):
        with pytest.raises(ValueError):
            router.select(logits, 1.0, 2, 1, "fixed", fixed=bad)
    with pytest.raises(ValueError, match="k_min <= L <= k_max"):
        router.select(logits, 1.0, 2, 2, "fixed", fixed=torch.tensor([0]))
    with pytest.raises(TypeError):
        router.select(logits, 1.0, 2, 1, "fixed", fixed=torch.tensor([0.0]))
    with pytest.raises(ValueError, match="shape"):
        router.select(logits, 1.0, 2, 1, "fixed", fixed=torch.tensor([[0]]))
    with pytest.raises(ValueError, match="N = "):
        router.select(_logits(2, N + 1, M, seed=0), 1.0, 2, 1, "topk")
    with pytest.raises(ValueError, match="floating"):
        router.select(torch.zeros(2, N, M, dtype=torch.int64), 1.0, 2, 1, "topk")
    with pytest.raises(TypeError):
        router.select([[0.0]], 1.0, 2, 1, "topk")  # type: ignore[arg-type]


def test_p2_1_topk_deterministic() -> None:
    cfg = _config(REF_A1)
    scorer = _scorer(cfg)
    N, M = cfg.model.n_cells, cfg.model.n_units
    uniform = CostTable.uniform(M, N)
    router = Router(scorer, uniform, lambda_cost=0.0)
    logits = _logits(4, N, M, seed=11)
    first = router.select(logits, 1.0, 4, 1, "topk")
    second = router.select(logits.clone(), 1.0, 4, 1, "topk")
    for name in ("cell_idx", "unit_idx", "alpha", "valid", "cost"):
        assert torch.equal(getattr(first, name), getattr(second, name)), name

    # Ties break by index: all-equal logits select cells 0..K-1 in order with unit 0.
    sel = router.select(torch.zeros(2, N, M), 1.0, 5, 1, "topk")
    assert sel.cell_idx.tolist() == [[0, 1, 2, 3, 4]] * 2
    assert sel.unit_idx.tolist() == [[0] * 5] * 2
    # A two-way tie between cells 9 and 2 puts the lower index first; a unit tie picks unit 0.
    tied = torch.zeros(1, N, M)
    tied[0, 9, 3] = 1.0
    tied[0, 2, 0] = 1.0
    tied[0, 2, 1] = 1.0
    sel = router.select(tied, 1.0, 3, 1, "topk")
    assert sel.cell_idx[0].tolist() == [2, 9, 0] and sel.unit_idx[0].tolist() == [0, 3, 0]
    assert torch.equal(router.cell_scores(tied), tied.amax(dim=-1))

    # score = U − λ·c: the dear unit loses to the cheap one once λ·Δc exceeds its edge, and the
    # subtraction happens in fp32 whatever the logits' dtype (bf16 could not resolve λ·c here).
    unit = [0.0] * M
    unit[0] = 0.002
    table = _table(M, N, seed=5, shared=0.0, overhead=0.0, unit=unit)
    costly = Router(scorer, table, lambda_cost=1.0)
    even = torch.ones(2, N, M, dtype=torch.bfloat16)  # every unit ties in bf16; unit 0 is dearer
    sel = costly.select(even, 1.0, 4, 1, "topk")
    assert sel.unit_idx.tolist() == [[1] * 4] * 2
    assert costly.scores(even).dtype == torch.float32
    free = Router(scorer, table, lambda_cost=0.0)
    assert free.select(even, 1.0, 4, 1, "topk").unit_idx.tolist() == [[0] * 4] * 2
    for mode in ("topk", "greedy_budget"):
        strong = torch.zeros(1, N, M)
        strong[0, 7, 0] = 1.0  # unit 0 of cell 7 leads by 0.5 ...
        strong[0, 7, 1] = 0.5
        with_lambda = Router(scorer, table, lambda_cost=400.0)  # ... λ·Δc = 0.8 overturns it
        assert with_lambda.select(strong, 1.0, 1, 1, mode).unit_idx[0].tolist() == [1]
        assert free.select(strong, 1.0, 1, 1, mode).unit_idx[0].tolist() == [0]

    # Rows are independent: permuting the batch permutes the selection.
    perm = torch.tensor([3, 0, 2, 1])
    permuted = router.select(logits[perm], 1.0, 4, 1, "topk")
    assert torch.equal(permuted.cell_idx, first.cell_idx[perm])
    assert torch.equal(permuted.unit_idx, first.unit_idx[perm])
    # and greedy_budget is deterministic too
    g1 = router.select(logits, 0.5, 8, 1, "greedy_budget")
    g2 = router.select(logits, 0.5, 8, 1, "greedy_budget")
    assert torch.equal(g1.cell_idx, g2.cell_idx) and torch.equal(g1.unit_idx, g2.unit_idx)


def test_p2_1_random_differs_by_seed() -> None:
    cfg = _config(REF_A1)
    router = Router.from_config(cfg.model, unroll=cfg.train.unroll)
    N, M = cfg.model.n_cells, cfg.model.n_units
    logits = _logits(4, N, M, seed=0)

    def draw(seed: int, budget: float = 1.0, k_max: int = 4) -> Selection:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return router.select(logits, budget, k_max, 1, "random", generator=generator)

    one, two, again = draw(1), draw(2), draw(1)
    assert torch.equal(one.cell_idx, again.cell_idx) and torch.equal(one.unit_idx, again.unit_idx)
    assert not (torch.equal(one.cell_idx, two.cell_idx) and torch.equal(one.unit_idx, two.unit_idx))
    assert torch.equal(one.logits, logits)  # the logits are carried, not consulted
    _check_selection(one, router.costs, k_max=4, k_min=1, budget=1.0)

    seen_cells: set[int] = set()
    seen_units: set[int] = set()
    for seed in range(300):
        sel = draw(seed)
        seen_cells.update(sel.cell_idx.flatten().tolist())
        seen_units.update(sel.unit_idx.flatten().tolist())
        rows = [tuple(row) for row in sel.cell_idx.tolist()]
        assert all(len(set(row)) == 4 for row in rows)
    assert seen_cells == set(range(N)) and seen_units == set(range(M))

    # the global RNG when no generator is given; seeding it makes the draw repeatable
    torch.manual_seed(5)
    a = router.select(logits, 1.0, 4, 1, "random")
    torch.manual_seed(5)
    b = router.select(logits, 1.0, 4, 1, "random")
    assert torch.equal(a.cell_idx, b.cell_idx) and torch.equal(a.unit_idx, b.unit_idx)

    # the budget truncates a random draw too, in sampled order
    pair = router.max_pair_cost
    sel = draw(3, budget=router.shared_cost + 2.5 * pair, k_max=4)
    assert sel.n_valid().tolist() == [2, 2, 2, 2]
    assert torch.equal(sel.cell_idx[:, :2], draw(3).cell_idx[:, :2])


def test_p2_1_scorer_cost_reported(tmp_path: Path) -> None:
    cfg, model = _model(TINY)
    ctx = _cpu_ctx()
    profile = write_cost_profile(
        profile_dense_costs(model, ctx, batch=2, positions=3), tmp_path / "tiny_costs.json"
    )
    table = CostTable.from_profile(profile, cfg.model.n_units)
    assert table.scorer > 0.0
    router = Router(model.scorer, table, lambda_cost=0.0)
    B, T = 2, 5
    state = model.init_state(B)
    u = torch.randn(B, cfg.model.d, dtype=torch.float64)
    eps = torch.randn(B, cfg.model.d, dtype=torch.float64)
    ctx_ids = torch.zeros(B, dtype=torch.int64)
    feats = router.scorer.features(u, eps, state, ctx_ids, budget_frac=1.0, round_idx=0)
    profiler = Profiler(ctx)
    for _ in range(T):
        logits = router(feats, profiler=profiler)  # the scorer is its own segment on every step
        sel = router.select(logits, cfg.model.budget, cfg.model.k_select, cfg.model.k_min, "topk")
        assert sel.scorer_cost == router.scorer_cost == table.scorer
    report = profiler.report()
    assert list(report.segments) == ["scorer"]
    scorer_row = report.segments["scorer"]
    assert scorer_row.n == T and scorer_row.tokens == T * B and scorer_row.device_ms > 0.0
    # the reported fraction is the profile's scorer line over the dense step, recomputed here
    data = json.loads(profile.read_text(encoding="utf-8"))
    per = {
        name: data["segments"][name]["device_ms"] / data["segments"][name]["tokens"]
        for name in COST_PROFILE_SEGMENTS
    }
    dense = per["shared_path"] + per["decode"] + per["messages"] + per["selected_cells"]
    assert table.scorer == pytest.approx(per["scorer"] / dense, rel=1e-12)
    assert table.dense_ms_per_position == pytest.approx(dense, rel=1e-12)
    # tokens can be given explicitly; without a profiler nothing is recorded and the logits agree
    with_tokens = Profiler(ctx)
    assert torch.equal(router(feats, profiler=with_tokens, tokens=7), router(feats))
    assert with_tokens.report().segments["scorer"].tokens == 7
    with pytest.raises(ValueError):
        router(feats, profiler=with_tokens, tokens=-1)
    # the uniform placeholder reports a zero scorer cost and says it is not a measurement
    placeholder = Router.from_config(cfg.model, unroll=cfg.train.unroll)
    assert placeholder.scorer_cost == 0.0 and placeholder.cost_source == UNIFORM_SOURCE
    assert (
        "uniform" in repr(placeholder) and placeholder.param_count() == model.scorer.param_count()
    )


def test_p2_1_cost_table_from_profile(tmp_path: Path) -> None:
    ms = {
        "shared_path": 6.0,
        "scorer": 3.0,
        "messages": 12.0,
        "selected_cells": 24.0,
        "decode": 6.0,
    }
    path = _write(
        tmp_path / "costs.json", _report(n_cells=4, n_units=2, batch=2, positions=3, ms=ms)
    )
    unit, overhead, shared = load_cost_table(path, 2)
    # per position: shared 1, scorer 0.5, messages 2, units 4, decode 1 -> dense 8 (scorer out)
    assert unit.tolist() == [0.0625, 0.0625] and unit.dtype == torch.float64
    assert overhead == 0.0625 and shared == 0.25
    table = CostTable.from_profile(path, 2)
    assert table.scorer == 0.0625
    assert table.dense_ms_per_position == 8.0  # 1 + 1 + 2 + 4 ms per position, scorer out
    assert table.dense_cost() == pytest.approx(1.0, abs=1e-12)
    assert table.n_cells == 4 and table.n_units == 2 and table.source == f"profile:{path}"
    assert table.max_pair_cost() == 0.125
    assert table.hash == CostTable.from_profile(path, 2).hash  # stable across loads
    assert len(table.hash) == 64
    record = table.to_record()
    assert record["hash"] == table.hash and record["unit"] == [0.0625, 0.0625]

    # the hash follows the values and the provenance (a shape change with the same timings)
    other = _write(
        tmp_path / "other.json",
        _report(n_cells=4, n_units=2, batch=2, positions=3, ms={**ms, "decode": 7.0}),
    )
    assert CostTable.from_profile(other, 2).hash != table.hash
    wider = _write(
        tmp_path / "wider.json",
        _report(n_cells=4, n_units=2, batch=2, positions=3, ms=ms, block_extra={"hidden": 16}),
    )
    wide_table = CostTable.from_profile(wider, 2)
    assert wide_table.hash != table.hash and torch.equal(wide_table.unit, table.unit)
    assert CostTable.uniform(2, 4).hash != table.hash
    assert (
        CostTable.uniform(2, 4).hash == CostTable.uniform(2, 4).hash != CostTable.uniform(2, 5).hash
    )

    # refusals, each naming the problem
    base = _report(n_cells=4, n_units=2, batch=2, positions=3, ms=ms)
    with pytest.raises(ValueError, match="M = 2 units, this model has M = 8"):
        load_cost_table(path, 8)
    broken: list[tuple[str, dict[str, Any]]] = []
    without_block = json.loads(json.dumps(base))
    del without_block[COST_PROFILE_KEY]
    broken.append(("not a cost profile", without_block))
    bad_schema = json.loads(json.dumps(base))
    bad_schema[COST_PROFILE_KEY]["schema"] = 2
    broken.append(("schema", bad_schema))
    no_segment = json.loads(json.dumps(base))
    del no_segment["segments"]["messages"]
    broken.append(("'messages' is missing", no_segment))
    no_tokens = json.loads(json.dumps(base))
    no_tokens["segments"]["scorer"]["tokens"] = 0
    broken.append(("tokens = 0", no_tokens))
    uneven = json.loads(json.dumps(base))
    uneven["segments"]["decode"]["tokens"] = 5
    broken.append(("same positions", uneven))
    zero_ms = json.loads(json.dumps(base))
    zero_ms["segments"]["selected_cells"]["device_ms"] = 0.0
    broken.append(("never a measurement", zero_ms))
    nan_ms = json.loads(json.dumps(base))
    nan_ms["segments"]["shared_path"]["device_ms"] = math.nan
    broken.append(("finite", nan_ms))
    no_n = json.loads(json.dumps(base))
    no_n["segments"]["decode"]["n"] = 0
    broken.append(("n = 0", no_n))
    bad_block = json.loads(json.dumps(base))
    bad_block[COST_PROFILE_KEY]["n_cells"] = "32"
    broken.append(("n_cells", bad_block))
    no_profile = json.loads(json.dumps(base))
    no_profile[COST_PROFILE_KEY]["profile"] = ""
    broken.append(("profile", no_profile))
    for i, (message, data) in enumerate(broken):
        file = tmp_path / f"broken{i}.json"
        file.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            CostTable.from_profile(file, 2)
    (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        CostTable.from_profile(tmp_path / "list.json", 2)
    (tmp_path / "junk.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        CostTable.from_profile(tmp_path / "junk.json", 2)
    with pytest.raises(ValueError, match="n_units"):
        CostTable.from_profile(path, 0)
    with pytest.raises(ValueError):
        write_cost_profile(without_block, tmp_path / "x.json")

    # the live path: a real profile of the tiny model, read back and bound to a router
    cfg, model = _model(TINY)
    report = profile_dense_costs(model, _cpu_ctx(), batch=2, positions=4, warmup=1)
    assert set(report[COST_PROFILE_KEY]) >= {
        "n_cells",
        "n_units",
        "d",
        "hidden",
        "batch",
        "positions",
    }
    assert report[COST_PROFILE_KEY]["n_cells"] == cfg.model.n_cells
    assert list(report["segments"]) == ["startup", *COST_PROFILE_SEGMENTS]
    assert report["segments"]["messages"]["n"] == 8 and report["segments"]["scorer"]["n"] == 4
    live = write_cost_profile(report, tmp_path / "live" / "tiny.json")
    live_table = CostTable.from_profile(live, cfg.model.n_units)
    assert (
        bool((live_table.unit > 0).all()) and live_table.shared > 0 and live_table.cell_overhead > 0
    )
    assert live_table.dense_cost() == pytest.approx(1.0, abs=1e-9)
    run = _config(TINY, overrides=[f"model.cost_profile={live}"])
    assert run.model.cost_profile == live
    router = Router.from_config(run.model, unroll=run.train.unroll)
    assert router.cost_table_hash == live_table.hash and router.cost_source == live_table.source
    assert torch.equal(router.cost_table, live_table.unit.float())
    assert "cost_table" not in router.state_dict()  # derived, never checkpointed
    assert set(router.state_dict()) == {f"scorer.{k}" for k in model.scorer.state_dict()}
    assert cost_table_from_config(run.model).hash == live_table.hash
    # a profile of another shape is refused for this config
    ref = _config(REF_A1, overrides=[f"model.cost_profile={live}"])
    with pytest.raises(ValueError, match="regenerate"):
        cost_table_from_config(ref.model)
    with pytest.raises(ValueError, match="regenerate"):
        Router(_scorer(_config(REF_A1)), live_table, lambda_cost=0.0)
    with pytest.raises(ValueError, match="batch"):
        profile_dense_costs(model, _cpu_ctx(), batch=0, positions=1)
    with pytest.raises(TypeError):
        profile_dense_costs(model.scorer, _cpu_ctx(), batch=1, positions=1)  # type: ignore[arg-type]


def test_p2_1_selection_carries_no_authority() -> None:
    """Track T: routing outputs carry no authority (static check). A Selection is indices,
    weights and costs; nothing in it names a permission, a tenant, a key or a scope."""
    forbidden = (
        "auth",
        "permission",
        "permit",
        "access",
        "allow",
        "grant",
        "role",
        "privilege",
        "credential",
        "acl",
        "scope",
        "token",
        "tenant",
        "key",
        "secret",
        "policy",
    )
    fields = dataclasses.fields(Selection)
    assert [f.name for f in fields] == [
        "cell_idx",
        "unit_idx",
        "alpha",
        "valid",
        "logits",
        "cost",
        "scorer_cost",
        "n_rounds_used",
        # P2.2: the gradient path and the control flags (indices, weights, costs and flags only)
        "cell_mask",
        "unit_weights",
        "expected_cost",
        "log_prob",
        "explored",
        "refreshed",
        "dense_fallback",
        # P2.10: the detached straight-through threshold of the decision (a number per row)
        "threshold",
    ]
    hints = get_type_hints(Selection)
    for field in fields:
        name = field.name.lower()
        assert not any(word in name for word in forbidden), field.name
        # tensors and plain numbers only (the P2.10 threshold is a tensor or absent)
        allowed = (Tensor, float, int, Tensor | None)
        assert hints[field.name] in allowed, (field.name, hints[field.name])
    # Router.forward(tokens=...) is the profiler's token count, so "token" is not scanned there
    api_words = tuple(word for word in forbidden if word != "token")
    for name in ("select", "forward", "scores", "cell_scores", "deadline_exceeded", "from_config"):
        parameters = inspect.signature(getattr(Router, name)).parameters
        for parameter in parameters:
            assert not any(word in parameter.lower() for word in api_words), (name, parameter)
    public = [name for name in dir(Router) if not name.startswith("_")]
    assert not any(any(word in name.lower() for word in api_words) for name in public), public
    doc = (Selection.__doc__ or "") + (Router.__doc__ or "")
    assert "authorise" in doc  # the contract is stated where the type is defined


# --------------------------------------------------------------------------------------------------
# Further P2.1 behaviour
# --------------------------------------------------------------------------------------------------


def test_p2_1_fixed_mode() -> None:
    cfg = _config(REF_A1)
    router = Router.from_config(cfg.model, unroll=cfg.train.unroll)
    N, M = cfg.model.n_cells, cfg.model.n_units
    logits = _logits(3, N, M, seed=4)
    fixed = torch.tensor([7, 2, 30], dtype=torch.int32)
    sel = router.select(logits, 1.0, 5, 1, "fixed", fixed=fixed)
    assert sel.cell_idx.tolist() == [[7, 2, 30, 0, 0]] * 3
    assert sel.valid.tolist() == [[True, True, True, False, False]] * 3
    expected_units = logits.argmax(dim=-1)[:, [7, 2, 30]]
    assert torch.equal(sel.unit_idx[:, :3], expected_units)
    assert bool((sel.unit_idx[:, 3:] == 0).all()) and sel.cell_idx.dtype == torch.int64
    _check_selection(sel, router.costs, k_max=5, k_min=1, budget=1.0)
    # exactly the list, never truncated: a budget that cannot hold it raises instead
    with pytest.raises(ValueError, match="fixed route"):
        router.select(logits, 2.5 * router.max_pair_cost, 5, 1, "fixed", fixed=fixed)
    # the config's fixed route goes through from_config and the k bounds
    run = _config(REF_A1, overrides=["model.routing=fixed", "model.fixed_route=[1, 2]"])
    assert Router.from_config(run.model, unroll=run.train.unroll).costs.n_cells == N
    with pytest.raises(ValueError, match="model.fixed_route"):
        Router.from_config(
            run.model.model_copy(update={"budget": 1.0 / (N * M)}), unroll=run.train.unroll
        )


def test_p2_1_deadline_exceeded() -> None:
    cfg = _config(TINY)
    router = Router.from_config(cfg.model, unroll=cfg.train.unroll)
    now = time.monotonic_ns()
    assert router.deadline_exceeded(now, 1000.0) is False
    assert router.deadline_exceeded(now - 2_000_000_000, 1000.0) is True
    assert router.deadline_exceeded(now, None) is False  # model.deadline_ms: null
    start = time.monotonic_ns()
    time.sleep(0.02)
    assert router.deadline_exceeded(start, 5.0) is True
    assert router.deadline_exceeded(start, 60_000) is False  # an int deadline is fine
    for bad in (0.0, -1.0, math.inf, math.nan, True):
        with pytest.raises(ValueError):
            router.deadline_exceeded(now, bad)
    for bad_start in (1.5, "0", True, None):
        with pytest.raises(TypeError):
            router.deadline_exceeded(bad_start, 1.0)  # type: ignore[arg-type]
    assert cfg.model.deadline_ms is None  # the default: no deadline


@pytest.mark.parametrize("device_type", DEVICES)
def test_p2_1_runs_on_device(device_type: str) -> None:
    device = _device(device_type)
    cfg = _config(REF_A1)
    N, M = cfg.model.n_cells, cfg.model.n_units
    table = _table(M, N, seed=1, shared=0.1, overhead=0.005)
    router = Router(_scorer(cfg), table, lambda_cost=0.5).to(device)
    assert router.cost_table.device.type == device.type
    logits = _logits(3, N, M, seed=2).to(device)
    generator = torch.Generator(device="cpu").manual_seed(0)  # a host generator; indices move
    budget = table.shared + 3.5 * table.max_pair_cost()
    for mode in SELECTION_MODES:
        kwargs: dict[str, Any] = {}
        if mode == "fixed":
            kwargs["fixed"] = torch.tensor([3, 1, 4])
        if mode == "random":
            kwargs["generator"] = generator
        sel = router.select(logits, budget, 4, 1, mode, **kwargs)
        for name in ("cell_idx", "unit_idx", "alpha", "valid", "cost"):
            assert getattr(sel, name).device.type == device.type, (mode, name)
        if mode != "dense":
            _check_selection(sel, table, k_max=4, k_min=1, budget=budget)
            assert bool((sel.n_valid() >= 3).all())  # 3 cells fit at the dearest unit
    # the device-side selection agrees with the CPU oracle for the deterministic modes
    cpu_router = Router(_scorer(cfg), table, lambda_cost=0.5)
    for mode in ("topk", "greedy_budget", "dense"):
        on_device = router.select(logits, budget, 4, 1, mode)
        on_cpu = cpu_router.select(logits.cpu(), budget, 4, 1, mode)
        assert torch.equal(on_device.cell_idx.cpu(), on_cpu.cell_idx)
        assert torch.equal(on_device.unit_idx.cpu(), on_cpu.unit_idx)
        assert torch.equal(on_device.valid.cpu(), on_cpu.valid)
        assert torch.allclose(on_device.cost.cpu(), on_cpu.cost)
    if device.type != "cpu":
        with pytest.raises(ValueError, match="move the router"):
            cpu_router.select(logits, budget, 4, 1, "topk")


def test_p2_1_config_fields(tmp_path: Path) -> None:
    cfg = _config(REF_A1)
    assert cfg.model.cost_profile is None
    assert ModelConfig().cost_profile is None
    for key in (
        "routing",
        "k_select",
        "k_min",
        "budget",
        "lambda_cost",
        "deadline_ms",
        "cost_profile",
    ):
        assert key in ModelConfig.model_fields
    assert ModelConfig.model_fields["k_min"].default == 1
    with_profile = _config(REF_A1, overrides=["model.cost_profile=profiles/cpu_costs.json"])
    assert with_profile.model.cost_profile == Path("profiles/cpu_costs.json")
    dump_resolved(with_profile, tmp_path / "resolved.yaml")
    assert load_config(tmp_path / "resolved.yaml") == with_profile
    # a relative path resolves against the repository root and a missing file says so
    with pytest.raises(FileNotFoundError) as info:
        cost_table_from_config(with_profile.model)
    assert str(REPO_ROOT / "profiles" / "cpu_costs.json") in str(info.value)

    def build(**model: Any) -> RunConfig:
        return _config(REF_A1, overrides=[f"model.{k}={v}" for k, v in model.items()])

    with pytest.raises(ValueError, match="distinct"):
        build(fixed_route="[1, 1]")
    with pytest.raises(ValueError, match=r"\[model.k_min, model.k_select\]"):
        build(routing="fixed", fixed_route="[0, 1, 2, 3, 4]")  # k_select = 4
    with pytest.raises(ValueError, match=r"\[model.k_min, model.k_select\]"):
        build(routing="fixed", fixed_route="[0]", k_min=2)
    assert build(routing="fixed", fixed_route="[0, 1, 2, 3]").model.fixed_route == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="greedy_budget needs model.budget > 0"):
        build(routing="greedy_budget", budget=0.0)
    assert build(routing="greedy_budget", budget=0.5).model.budget == 0.5
    assert build(routing="topk", budget=0.0).model.budget == 0.0  # the config allows it ...
    # ... and the router refuses it at construction, not on a batch
    cheap = build(routing="topk", budget=0.0)
    with pytest.raises(ValueError, match="cannot hold model.k_min = 1"):
        Router.from_config(cheap.model, unroll=cheap.train.unroll)
    fine = build(routing="topk", budget=0.25)
    assert Router.from_config(fine.model, unroll=fine.train.unroll).lambda_cost == 0.0
    with pytest.raises(ValueError, match="lambda_cost"):
        Router(_scorer(cfg), CostTable.uniform(8, 32), lambda_cost=-1.0)
    with pytest.raises(TypeError):
        Router.from_config(cfg, unroll=cfg.train.unroll)  # type: ignore[arg-type]
    # the example config lists the new key at its default
    example = load_config(REPO / "configs" / "_example.yaml")
    assert example.model.cost_profile is None
