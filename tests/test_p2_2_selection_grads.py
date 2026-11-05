"""P2.2 tests: the gradient path through hard selection (soft warm-up, straight-through,
REINFORCE), the selection schedules, exploration, refresh probes, dense fallback, the budget after
post-processing, the trainer plumbing, and the random / fixed controls training end to end."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import RunConfig, TrainConfig, load_config, load_hardware_profile
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example, TaskSpec, generate, load_dataset_spec
from hsn.device import DeviceContext, resolve, seed_everything
from hsn.kernels.costs import CostTable
from hsn.model.hsn import HSN, StepOutput
from hsn.model.router import BUDGET_SLACK, GATE_MODES, SCORED_MODES, Router, Scorer, Selection
from hsn.model.state import CellGraphState
from hsn.train.schedules import SelectionSchedule
from hsn.train.tbptt import TBPTTStepRecord, TBPTTTrainer
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
EXAMPLE = REPO / "configs" / "_example.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
UNROLL = 32  # the kv fixture's window (documents fit one window)
N_DOCS = 64

cuda_param = pytest.param("cuda", marks=pytest.mark.cuda)
mps_param = pytest.param("mps", marks=pytest.mark.mps)
DEVICES = ["cpu", cuda_param, mps_param]

#: The tiny model every test uses: N = 6 cells, M = 3 units, K = 2.
TINY_OVERRIDES = (
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
    "train.batch=4",
    "train.micro_batch=4",
)
N, M, K = 6, 3, 2


def AGE_BOUND(age_max: int) -> int:  # noqa: N802 - a named constant of the fixture shapes
    """The oldest a cell can get: a probe fires when ``age > age_max`` but only ``K`` slots a
    position, so the last of ``N − K`` cells that went stale together waits ``ceil((N − K) / K)``
    further positions."""
    return age_max + math.ceil((N - K) / K)


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _config(*overrides: str) -> RunConfig:
    return load_config(REF_A1, overrides=[*TINY_OVERRIDES, *overrides])


def _build(cfg: RunConfig, seed: int = 0, perturb: int | None = None) -> HSN:
    seed_everything(seed)
    model = HSN.from_config(cfg)
    if perturb is not None:
        _perturb(model, perturb)
    return model


def _perturb(model: HSN, seed: int, scale: float = 0.3) -> None:
    """Seeded noise on every parameter (the P1.5 recipe): a freshly initialised model's outputs
    barely depend on the state, so every gradient is ~1e-13 whatever the path."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn(p.shape, generator=generator, dtype=torch.float64) * scale
            p.add_(noise.to(dtype=p.dtype, device=p.device))


def _window(
    cfg: RunConfig, B: int = 4, T: int = 16, seed: int = 1
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(0, cfg.model.vocab_size, (B, T), generator=generator)
    ctx = torch.randint(0, cfg.model.n_context, (B, T), generator=generator)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    return tokens, ctx, reset


def _device(device_type: str) -> torch.device:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device on this machine")
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no mps device on this machine")
    return torch.empty(0, device=device_type).device


def _cpu() -> DeviceContext:
    return resolve(load_hardware_profile("cpu"))


def _scorer(cfg: RunConfig, seed: int = 0) -> Scorer:
    torch.manual_seed(seed)
    return Scorer(cfg.model, unroll=cfg.train.unroll)


def _router(
    cfg: RunConfig | None = None, table: CostTable | None = None, lambda_cost: float = 0.0
) -> Router:
    cfg = cfg if cfg is not None else _config()
    if table is None:
        return Router.from_config(cfg.model, unroll=cfg.train.unroll)
    return Router(_scorer(cfg), table, lambda_cost=lambda_cost)


def _table(
    unit: Sequence[float], *, shared: float = 0.0, overhead: float = 0.0, n_cells: int = N
) -> CostTable:
    """A hand-built, non-uniform table (every profiled table is uniform per unit)."""
    record = json.dumps([list(unit), shared, overhead, n_cells])
    return CostTable(
        unit=torch.tensor(list(unit), dtype=torch.float64),
        cell_overhead=overhead,
        shared=shared,
        scorer=0.0,
        n_units=len(unit),
        n_cells=n_cells,
        dense_ms_per_position=None,
        source="test table",
        hash=hashlib.blake2b(record.encode(), digest_size=32).hexdigest(),
    )


def _logits(B: int, seed: int, n: int = N, m: int = M) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(B, n, m, generator=generator)


def _hard_mask(sel: Selection) -> Tensor:
    """The cells of the valid slots as a fp32 ``[B, N]`` mask, from the indices alone."""
    B, n_cells = int(sel.logits.shape[0]), int(sel.logits.shape[1])
    mask = torch.zeros(B, n_cells, dtype=torch.float32, device=sel.cell_idx.device)
    return mask.scatter_add(1, sel.cell_idx, sel.valid.to(torch.float32))


def _check_invariants(
    sel: Selection, table: CostTable, *, k_max: int, k_min: int, budget: float
) -> None:
    """The P2.1 invariants every sparse selection keeps after post-processing."""
    B = sel.logits.shape[0]
    assert sel.k == k_max
    n_valid = sel.n_valid()
    assert bool((n_valid >= k_min).all()) and bool((n_valid <= k_max).all())
    for b in range(B):
        cells = sel.cell_idx[b][sel.valid[b]].tolist()
        assert len(set(cells)) == len(cells), "a cell was selected twice"
        total = table.shared
        for k in range(k_max):
            if bool(sel.valid[b, k]):
                total += table.cell_overhead + float(table.unit[int(sel.unit_idx[b, k])])
        assert float(sel.cost[b]) == pytest.approx(total, rel=1e-5, abs=1e-6)
        if not bool(sel.dense_fallback[b]):
            assert total <= budget * (1.0 + 2 * BUDGET_SLACK), (b, total, budget)
    assert bool(
        sel.valid[~sel.explored.logical_or(sel.refreshed).logical_not()].all()
    )  # flags only on valid slots


# --------------------------------------------------------------------------------------------------
# kv in-range documents for the training tests (the P1.7 fixture recipe)
# --------------------------------------------------------------------------------------------------


def _tiny_task() -> TaskSpec:
    task = SPEC.task("kv", "in_range")
    return TaskSpec.model_validate(
        {
            **task.model_dump(),
            "n_train": 64,
            "n_dev": 16,
            "n_test": 16,
            "length_range": [16, UNROLL],
            "dep_len_range": [4, 20],
            "unroll": UNROLL,
        }
    )


def _docs(task: TaskSpec, split: str, n: int) -> list[Example]:
    return [generate(task, split, i) for i in range(n)]  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def train_docs() -> list[Example]:
    return _docs(_tiny_task(), "train", N_DOCS)


@pytest.fixture(scope="module")
def dev_docs() -> list[Example]:
    return _docs(_tiny_task(), "dev", 16)


def _train_config(*overrides: str) -> RunConfig:
    return _config(
        f"train.unroll={UNROLL}",
        "train.steps=40",
        "train.eval_every=40",
        "train.ckpt_every=40",
        "train.stats_every=10",
        *overrides,
    )


# --------------------------------------------------------------------------------------------------
# The chunk's tests
# --------------------------------------------------------------------------------------------------


def test_p2_2_ste_grad_reaches_scorer(tmp_path: Path) -> None:
    """Straight-through: the forward tensors are exactly hard, the mask's Jacobian is diagonal
    (the threshold is detached) with the sigmoid's slope on the diagonal, the unit weights carry
    the softmax Jacobian, and a task loss reaches the scorer through a hard-routed window."""
    cfg = _config()
    router = _router(cfg)
    tau = 0.7
    logits = _logits(2, seed=3).requires_grad_(True)
    sel = router.select(logits, 1.0, K, 1, "topk", gate="ste", temperature=tau)
    hard = _hard_mask(sel)
    assert torch.equal(sel.cell_mask, hard) and torch.equal(sel.alpha, sel.valid.float())
    assert set(sel.cell_mask.unique().tolist()) <= {0.0, 1.0}
    one_hot = torch.nn.functional.one_hot(logits.detach().argmax(-1), M).float()
    assert torch.equal(sel.unit_weights, one_hot)  # topk executes the argmax unit of every cell
    assert torch.allclose(sel.expected_cost, sel.cost)
    assert (
        sel.cell_mask.requires_grad and sel.unit_weights.requires_grad and sel.alpha.requires_grad
    )

    def mask_of(x: Tensor) -> Tensor:
        return router.select(x, 1.0, K, 1, "topk", gate="ste", temperature=tau).cell_mask

    jac = torch.autograd.functional.jacobian(mask_of, logits)  # [B, N, B, N, M]
    s = logits.detach().amax(dim=-1)  # λ = 0: the cell score is the max logit
    unit = logits.detach().argmax(dim=-1)
    for b in range(2):
        selected = hard[b] > 0
        theta = 0.5 * (s[b][selected].min() + s[b][~selected].max())
        for i in range(N):
            expected = torch.zeros(2, N, M)
            sig = torch.sigmoid((s[b, i] - theta) / tau)
            expected[b, i, unit[b, i]] = sig * (1 - sig) / tau
            torch.testing.assert_close(jac[b, i], expected, atol=1e-6, rtol=1e-5)

    def units_of(x: Tensor) -> Tensor:
        return router.select(x, 1.0, K, 1, "topk", gate="ste", temperature=tau).unit_weights

    jac_u = torch.autograd.functional.jacobian(units_of, logits)  # [B, N, M, B, N, M]
    probs = torch.softmax(logits.detach() / tau, dim=-1)
    for b in range(2):
        for i in range(N):
            p = probs[b, i]
            expected = (torch.diag(p) - p[:, None] * p[None, :]) / tau  # ∂softmax_m / ∂logit_m'
            torch.testing.assert_close(jac_u[b, i, :, b, i, :], expected, atol=1e-6, rtol=1e-5)
            others = jac_u[b, i].clone()  # [M, B, N, M]: no other cell or row contributes
            others[:, b, i, :] = 0
            assert float(others.abs().max()) == 0.0

    # gate=None (P2.1 callers, evaluation): the same decision, no gradient path
    plain = router.select(logits, 1.0, K, 1, "topk")
    assert torch.equal(plain.cell_idx, sel.cell_idx) and torch.equal(plain.unit_idx, sel.unit_idx)
    assert torch.equal(plain.cell_mask, sel.cell_mask.detach())
    assert torch.equal(plain.unit_weights, sel.unit_weights.detach())
    assert not plain.cell_mask.requires_grad and not plain.unit_weights.requires_grad
    assert not plain.alpha.requires_grad and not plain.expected_cost.requires_grad
    assert not plain.explored.any() and not plain.refreshed.any() and not plain.dense_fallback.any()
    assert torch.equal(plain.log_prob, torch.zeros(2))
    # random takes no gradient, fixed the unit gradient only
    rnd = router.select(
        logits, 1.0, K, 1, "random", gate="ste", generator=torch.Generator().manual_seed(0)
    )
    assert not rnd.cell_mask.requires_grad and not rnd.unit_weights.requires_grad
    fix = router.select(logits, 1.0, K, 1, "fixed", fixed=torch.tensor([1, 4]), gate="ste")
    assert not fix.cell_mask.requires_grad and fix.unit_weights.requires_grad
    assert torch.equal(fix.cell_mask[:, [1, 4]], torch.ones(2, 2))

    # through the model: a hard-routed window's task loss reaches the scorer
    hard_cfg = _config("model.routing=topk", "train.gate_mode=ste")
    model = _build(hard_cfg, perturb=3)
    tokens, ctx, reset = _window(hard_cfg)
    logits_w, state, info = model(tokens, ctx, model.init_state(4), reset)
    targets = tokens.roll(-1, dims=1)
    loss = torch.nn.functional.cross_entropy(
        logits_w.reshape(-1, logits_w.shape[-1]), targets.reshape(-1)
    )
    loss.backward()
    grad = model.router.scorer.net.weight.grad
    assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.norm()) > 1e-6
    assert float(info["cost"].mean()) == pytest.approx(K / (N * M), abs=1e-6)  # the hard cost ran
    assert torch.equal(model.router.scorer.net.weight, model.scorer.net.weight)  # one scorer
    assert "scorer" not in dict(model.named_children()) and "router" in dict(model.named_children())
    assert all(not key.startswith("scorer.") for key in model.state_dict())


def test_p2_2_soft_with_k_equal_n_equals_dense() -> None:
    """Soft gates with K = N are the dense model: an all-ones cell mask, the softmax over units,
    and a whole window's logits and state equal bit for bit on cpu."""
    cfg = _config()
    router = _router(cfg)
    logits = _logits(3, seed=5)
    sel = router.select(logits, 1.0, N, N, "topk", gate="soft", temperature=1.0)
    assert torch.equal(sel.cell_mask, torch.ones(3, N))
    assert torch.equal(sel.unit_weights, torch.softmax(logits, dim=-1))
    assert bool(sel.valid.all()) and torch.allclose(
        sel.cost, torch.ones(3)
    )  # the warm-up executes every cell
    assert torch.allclose(
        sel.alpha, torch.softmax(logits, dim=-1).amax(dim=-1).gather(1, sel.cell_idx)
    )
    # a hard K = N selection too: the mask is all ones and the unit weights one-hot
    ste = router.select(logits, 1.0, N, N, "topk", gate="ste")
    assert torch.equal(ste.cell_mask, torch.ones(3, N))

    soft_cfg = _config(
        "model.routing=topk", f"model.k_select={N}", f"model.k_min={N}", "train.gate_mode=soft"
    )
    dense_cfg = _config("model.routing=dense")
    soft, dense = _build(soft_cfg, perturb=7), _build(dense_cfg)
    dense.load_state_dict(soft.state_dict())  # the same weights, only model.routing differs
    tokens, ctx, reset = _window(soft_cfg)
    logits_s, state_s, info_s = soft(tokens, ctx, soft.init_state(4), reset)
    logits_d, state_d, info_d = dense(tokens, ctx, dense.init_state(4), reset)
    assert torch.equal(logits_s, logits_d)
    for name in ("h", "q", "age", "pos"):
        assert torch.equal(getattr(state_s, name), getattr(state_d, name)), name
    assert bool((info_s["cost"] == 1.0).all()) and bool((info_d["cost"] == 1.0).all())
    torch.testing.assert_close(
        info_s["alpha_entropy"], info_d["alpha_entropy"]
    )  # log vs log_softmax
    assert bool((info_s["expected_cost"] < 1.0).all())  # the relaxed selection's expected cost
    assert not info_s["explored"].any() and not info_s["dense_fallback"].any()
    assert set(info_d) == {
        "z",
        "z_hat",
        "z_target",
        "eps",
        "u",
        "state_abs_max",
        "eps_abs_mean",
        "alpha_entropy",
        "cost",
        "n_rounds",
    }  # the dense output is P1.5's exactly
    # with K < N the soft path differs from dense (a relaxation, not the dense step)
    partial_cfg = _config("model.routing=topk", "train.gate_mode=soft")
    partial = _build(partial_cfg)
    partial.load_state_dict(soft.state_dict())
    logits_p, _, info_p = partial(tokens, ctx, partial.init_state(4), reset)
    assert not torch.equal(logits_p, logits_d) and bool((info_p["cost"] == 1.0).all())
    # ... and the warm-up runs no probe (nothing is stale when every cell executes): the ages
    # are all reset even with a tiny age_max and an exploration rate
    warm_cfg = _config(
        "model.routing=topk",
        "train.gate_mode=soft",
        "train.epsilon_start=0.3",
        "train.epsilon_end=0.3",
        "model.age_max=2",
    )
    warm = _build(warm_cfg, perturb=7)
    _, state_w, info_w = warm(tokens, ctx, warm.init_state(4), reset)
    assert not state_w.age.any() and float(info_w["refreshed"].sum()) == 0.0
    assert float(info_w["explored"].sum()) == 0.0


def test_p2_2_exploration_rate_measured() -> None:
    """ε replaces exactly one valid slot of a row by an unselected cell with probability ε, flags
    it, keeps the budget and the distinctness, and leaves the other rows untouched."""
    cfg = _config()
    router = _router(cfg)
    B, eps = 4000, 0.3
    logits = _logits(B, seed=11)
    plain = router.select(logits, 1.0, K, 1, "topk")
    sel = router.select(
        logits,
        1.0,
        K,
        1,
        "topk",
        generator=torch.Generator(device="cpu").manual_seed(21),
        epsilon=eps,
    )
    explored_rows = sel.explored.any(dim=1)
    rate = float(explored_rows.float().mean())
    assert abs(rate - eps) < 4 * math.sqrt(eps * (1 - eps) / B), rate  # four binomial sigmas
    assert torch.equal(sel.valid, plain.valid) and torch.equal(sel.cost, plain.cost)
    differs = sel.cell_idx != plain.cell_idx
    assert torch.equal(differs, sel.explored)  # the flag marks exactly the replaced slot
    assert bool((sel.explored.sum(dim=1)[explored_rows] == 1).all())
    assert torch.equal(sel.cell_idx[~explored_rows], plain.cell_idx[~explored_rows])
    assert torch.equal(sel.unit_idx[~explored_rows], plain.unit_idx[~explored_rows])
    unit_argmax = logits.argmax(dim=-1)
    for b in explored_rows.nonzero().flatten()[:200].tolist():
        k = int(sel.explored[b].nonzero())
        new_cell = int(sel.cell_idx[b, k])
        assert new_cell not in plain.cell_idx[b].tolist()  # a cell that was not selected
        assert int(sel.unit_idx[b, k]) == int(unit_argmax[b, new_cell])  # its argmax unit
        assert len(set(sel.cell_idx[b].tolist())) == K
    # both slots are replaced about equally often (uniform over the valid slots)
    per_slot = sel.explored.float().sum(dim=0)
    assert abs(float(per_slot[0] - per_slot[1])) < 4 * math.sqrt(B * eps / 4)
    # the new cell is uniform over the unselected ones: every cell appears
    assert set(sel.cell_idx[sel.explored].tolist()) == set(range(N))
    _check_invariants(sel, router.costs, k_max=K, k_min=1, budget=1.0)
    # the explored cell is executed: hard mask 1, alpha 1, and under STE still hard
    assert torch.equal(sel.cell_mask, _hard_mask(sel)) and bool(
        (sel.alpha[sel.explored] == 1).all()
    )
    ste = router.select(
        logits,
        1.0,
        K,
        1,
        "topk",
        generator=torch.Generator(device="cpu").manual_seed(21),
        epsilon=eps,
        gate="ste",
    )
    assert torch.equal(ste.explored, sel.explored) and torch.equal(
        ste.cell_mask.detach(), sel.cell_mask
    )
    # ε = 0: nothing happens; the same generator gives the same draw; the global RNG works too
    none = router.select(logits, 1.0, K, 1, "topk", epsilon=0.0)
    assert not none.explored.any() and torch.equal(none.cell_idx, plain.cell_idx)
    again = router.select(
        logits,
        1.0,
        K,
        1,
        "topk",
        generator=torch.Generator(device="cpu").manual_seed(21),
        epsilon=eps,
    )
    assert torch.equal(again.cell_idx, sel.cell_idx) and torch.equal(again.explored, sel.explored)
    torch.manual_seed(1)
    glob = router.select(logits[:100], 1.0, K, 1, "topk", epsilon=1.0)
    assert bool(glob.explored.any(dim=1).all())  # ε = 1: every row explores
    # the controls take no exploration; a generator without a draw is refused (P2.1)
    for mode, kwargs in (("random", {}), ("fixed", {"fixed": torch.tensor([0, 1])}), ("dense", {})):
        with pytest.raises(ValueError, match="exploration"):
            router.select(logits[:2], 1.0, K, 1, mode, epsilon=0.1, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only used with mode='random'"):
        router.select(logits[:2], 1.0, K, 1, "topk", generator=torch.Generator(device="cpu"))
    for bad in (-0.1, 1.5, math.nan, True):
        with pytest.raises(ValueError, match="epsilon"):
            router.select(logits[:2], 1.0, K, 1, "topk", epsilon=bad)  # type: ignore[arg-type]
    # with every cell selected there is nothing to explore
    full = router.select(logits[:50], 1.0, N, N, "topk", epsilon=1.0)
    assert not full.explored.any()


def test_p2_2_refresh_probe_forces_stale_cells() -> None:
    """Cells older than ``age_max`` that are not selected replace the lowest-scoring slots, stalest
    first, with their argmax unit and a ``refreshed`` flag; a window keeps every age within
    ``age_max + 1``; a row the probe would push over the budget keeps its scored selection."""
    cfg = _config("model.age_max=4")
    router = _router(cfg)
    assert router.age_max == 4
    logits = torch.zeros(4, N, M)
    for b in range(4):  # cell scores 5 > 4 > 3 > 2 > 1 > 0 for cells 0..5 (units differ per cell)
        for i in range(N):
            logits[b, i, i % M] = float(N - i)
    age = torch.zeros(4, N, dtype=torch.int64)
    age[0, 3], age[0, 5] = 10, 20  # row 0: two stale cells -> both slots replaced, stalest first
    age[1, 2] = 7  # row 1: one stale cell -> the lowest-scoring slot (slot 1) is replaced
    age[2] = 4  # row 2: at the threshold (not > age_max) -> nothing
    age[3, 0] = 9  # row 3: the stale cell is already selected -> nothing to force
    plain = router.select(logits, 1.0, K, 1, "topk")
    assert plain.cell_idx.tolist() == [[0, 1]] * 4
    sel = router.select(logits, 1.0, K, 1, "topk", age=age)
    assert sel.cell_idx.tolist() == [[3, 5], [0, 2], [0, 1], [0, 1]]
    assert sel.refreshed.tolist() == [[True, True], [False, True], [False, False], [False, False]]
    assert not sel.explored.any() and torch.equal(sel.valid, plain.valid)
    assert sel.unit_idx[0].tolist() == [3 % M, 5 % M] and int(sel.unit_idx[1, 1]) == 2 % M
    assert torch.equal(sel.cell_mask[0], torch.tensor([0.0, 0, 0, 1, 0, 1]))
    _check_invariants(sel, router.costs, k_max=K, k_min=1, budget=1.0)
    # a stale cell and an exploration on the same row never touch the same slot
    probed = router.select(
        logits,
        1.0,
        K,
        1,
        "topk",
        age=age,
        epsilon=1.0,
        generator=torch.Generator(device="cpu").manual_seed(3),
    )
    assert not (probed.explored & probed.refreshed).any()
    assert probed.explored.tolist()[0] == [False, False]  # row 0 has no non-refreshed slot left
    assert bool(probed.explored[1:].any(dim=1).all())
    for b in range(4):
        assert len(set(probed.cell_idx[b].tolist())) == K
        for k in probed.explored[b].nonzero().flatten().tolist():
            assert (
                int(probed.cell_idx[b, k]) not in plain.cell_idx[b].tolist()
            )  # never an evicted cell
    # int32 ages are accepted; the controls, wrong shapes and dtypes are refused
    assert torch.equal(
        router.select(logits, 1.0, K, 1, "topk", age=age.int()).cell_idx, sel.cell_idx
    )
    with pytest.raises(ValueError, match="refresh probes"):
        router.select(logits, 1.0, K, 1, "random", age=age)
    with pytest.raises(ValueError, match="refresh probes"):
        router.select(logits, 1.0, K, 1, "fixed", fixed=torch.tensor([0]), age=age)
    with pytest.raises(ValueError, match="shape"):
        router.select(logits, 1.0, K, 1, "topk", age=age[:, :3])
    with pytest.raises(TypeError):
        router.select(logits, 1.0, K, 1, "topk", age=age.float())

    # the budget guarantee: forcing a cell whose argmax unit is dear reverts that row only
    table = _table([0.01, 0.01, 0.30])
    costly = Router(_scorer(cfg), table, lambda_cost=0.0)
    budget = 0.05  # holds two cheap cells, never the dear unit
    dear = logits.clone()
    dear[:, 3] = 0.0
    dear[:, 3, 2] = 2.0  # cell 3 prefers the dear unit 2 ...
    dear[:, 5] = 0.0
    dear[:, 5, 0] = 1.5  # ... cell 5 a cheap one
    stale = torch.zeros(4, N, dtype=torch.int64)
    stale[0, 3] = 9  # row 0: forcing cell 3 would cost 0.31 > budget -> reverted
    stale[1, 5] = 9  # row 1: forcing cell 5 fits -> refreshed
    forced = costly.select(
        dear, budget, K, 0, "topk", age=stale
    )  # k_min 0: the dear unit never fits
    base = costly.select(dear, budget, K, 0, "topk")
    assert torch.equal(forced.cell_idx[0], base.cell_idx[0]) and not forced.refreshed[0].any()
    assert int(forced.cell_idx[1, 1]) == 5 and bool(forced.refreshed[1, 1])
    _check_invariants(forced, table, k_max=K, k_min=0, budget=budget)

    # through the model: no cell goes staler than age_max + 1 in a window, and evaluation keeps
    # the probes (they are a property of the model); a high age_max forces nothing
    short = _config("model.routing=topk", "train.gate_mode=ste", "model.age_max=3")
    model = _build(short, perturb=5)
    tokens, ctx, reset = _window(short, B=8, T=16)
    for training in (True, False):
        model.train(training)
        with torch.set_grad_enabled(training):
            _, state, info = model(tokens, ctx, model.init_state(8), reset)
        assert int(state.age.max()) <= AGE_BOUND(3), training
        assert float(info["refreshed"].sum()) > 0, training
    tall = _config("model.routing=topk", "train.gate_mode=ste", "model.age_max=100")
    lazy = _build(tall, perturb=5)
    _, state, info = lazy(tokens, ctx, lazy.init_state(8), reset)
    assert int(state.age.max()) > AGE_BOUND(3) and float(info["refreshed"].sum()) == 0.0


def test_p2_2_reinforce_loss_finite(
    train_docs: list[Example], dev_docs: list[Example], tmp_path: Path
) -> None:
    """REINFORCE: the sampled cells' Plackett–Luce log-probability is finite, matches a hand
    computation, differentiates to the scorer; the trainer's term is finite, zero on the first
    window (the baseline starts at the first reward) and the baseline survives a resume."""
    cfg = _config("model.n_cells=4", "model.n_units=2", "model.in_degree=2")
    router = Router.from_config(cfg.model, unroll=cfg.train.unroll)
    logits = torch.tensor([[[0.5, 1.5], [2.0, -1.0], [0.0, 0.3], [-0.5, 0.1]]], requires_grad=True)
    tau = 0.5
    generator = torch.Generator(device="cpu").manual_seed(4)
    sel = router.select(
        logits, 1.0, 2, 1, "topk", gate="reinforce", temperature=tau, generator=generator
    )
    assert sel.log_prob.shape == (1,) and bool(torch.isfinite(sel.log_prob).all())
    assert float(sel.log_prob.detach()) <= 0

    s = logits.detach().amax(dim=-1)[0] / tau  # the cell scores over τ
    c0, c1 = sel.cell_idx[0].tolist()
    remaining = [i for i in range(4) if i != c0]
    expected = (s[c0] - torch.logsumexp(s, 0)) + (s[c1] - torch.logsumexp(s[remaining], 0))
    assert float(sel.log_prob.detach()) == pytest.approx(float(expected), rel=1e-6)
    assert sel.log_prob.requires_grad
    sel.log_prob.sum().backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0
    # hard forward values with the straight-through unit weights; the sample varies with the seed
    assert torch.equal(sel.cell_mask.detach(), _hard_mask(sel))
    assert set(sel.cell_mask.detach().unique().tolist()) <= {0.0, 1.0}
    assert not sel.cell_mask.requires_grad  # the cells train through log_prob alone
    assert sel.unit_weights.requires_grad and set(sel.unit_weights.detach().unique().tolist()) <= {
        0.0,
        1.0,
    }
    draws = {
        tuple(
            router.select(
                logits,
                1.0,
                2,
                1,
                "topk",
                gate="reinforce",
                generator=torch.Generator(device="cpu").manual_seed(seed),
            )
            .cell_idx[0]
            .tolist()
        )
        for seed in range(40)
    }
    assert len(draws) > 1
    # with a very low temperature the sample is the top-k; budget truncation shortens the prefix
    cold = router.select(
        logits, 1.0, 2, 1, "topk", gate="reinforce", temperature=1e-3, generator=generator
    )
    assert cold.cell_idx[0].tolist() == [1, 0]
    # reinforce needs topk; the P2.1 modes refuse it
    for mode, kwargs in (
        ("greedy_budget", {}),
        ("random", {}),
        ("fixed", {"fixed": torch.tensor([0])}),
    ):
        with pytest.raises(ValueError, match="reinforce"):
            router.select(logits, 1.0, 2, 1, mode, gate="reinforce", **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gate"):
        router.select(logits, 1.0, 2, 1, "topk", gate="hard")  # type: ignore[arg-type]
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="temperature"):
            router.select(logits, 1.0, 2, 1, "topk", gate="ste", temperature=bad)

    # the trainer: finite objective, the first window's term is exactly zero, the baseline moves
    run_cfg = _train_config(
        "model.routing=topk",
        "train.gate_mode=reinforce",
        "train.steps=6",
        "train.eval_every=6",
        "train.ckpt_every=3",
        "train.stats_every=3",
        "train.reinforce_baseline_decay=0.5",
    )
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)
    model = _build(run_cfg, perturb=9)
    trainer = TBPTTTrainer(model, run_cfg, _cpu(), run_dir=tmp_path / "a")
    history = trainer.fit(train_ds, dev_ds, 6)
    records = [r for r in history.steps if isinstance(r, TBPTTStepRecord)]
    assert len(records) == 6
    assert all(math.isfinite(r.total) and math.isfinite(r.parts["reinforce"]) for r in records)
    assert records[0].parts["reinforce"] == 0.0
    assert any(r.parts["reinforce"] != 0.0 for r in records[1:])
    assert list(records[0].parts) == ["task", "pred", "reinforce"]
    assert trainer.reinforce_baseline is not None and math.isfinite(trainer.reinforce_baseline)
    baseline = trainer.reinforce_baseline
    info = {"route_log_prob": torch.tensor([-2.0, -3.0])}
    for task, n_loss in ((math.nan, 5), (math.inf, 5), (1.0, 0)):  # P2.4: terms in, terms out
        kept = trainer._routing_terms({"task": torch.tensor(task)}, info, n_loss)
        assert list(kept) == ["task", "reinforce"] and float(kept["reinforce"]) == 0.0
        assert trainer.reinforce_baseline == baseline  # a skipped window never moves it
    terms = {"task": torch.tensor(1.0), "pred": torch.tensor(0.5)}
    moved = trainer._routing_terms(terms, info, 5)
    assert float(sum(moved.values())) == pytest.approx(1.5 + 2.5 * (-1.0 - baseline))
    assert float(moved["reinforce"]) == pytest.approx(2.5 * (-1.0 - baseline))
    assert trainer.reinforce_baseline == pytest.approx(0.5 * baseline + 0.5 * -1.0)
    assert records[-1].stats["route_cost"] == pytest.approx(K / (N * M), abs=1e-6)
    # the baseline is checkpointed and restored; a pre-P2.2 checkpoint is refused in this mode
    again = TBPTTTrainer(_build(run_cfg, seed=1), run_cfg, _cpu(), run_dir=tmp_path / "b")
    assert again.resume(tmp_path / "a" / "step_00000003.pt") == 3
    payload = torch.load(tmp_path / "a" / "step_00000003.pt", weights_only=True)
    assert again.reinforce_baseline == payload["extra"]["reinforce_baseline"]
    history_b = again.fit(train_ds, dev_ds, 6)
    for rec_a, rec_b in zip(history.steps[3:], history_b.steps, strict=True):
        assert isinstance(rec_a, TBPTTStepRecord) and isinstance(rec_b, TBPTTStepRecord)
        assert rec_a.parts == rec_b.parts and rec_a.total == rec_b.total
    payload["extra"]["reinforce_baseline"] = math.nan
    torch.save(payload, tmp_path / "nan.pt")
    with pytest.raises(ValueError, match="finite"):
        TBPTTTrainer(_build(run_cfg), run_cfg, _cpu(), run_dir=tmp_path / "n").resume(
            tmp_path / "nan.pt"
        )
    del payload["extra"]["reinforce_baseline"]
    torch.save(payload, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="reinforce_baseline"):
        TBPTTTrainer(_build(run_cfg), run_cfg, _cpu(), run_dir=tmp_path / "c").resume(
            tmp_path / "old.pt"
        )
    # a soft warm-up keeps the term at zero until the gate switches
    warm_cfg = _train_config(
        "model.routing=topk",
        "train.gate_mode=reinforce",
        "train.soft_until_step=2",
        "train.steps=4",
        "train.eval_every=4",
        "train.ckpt_every=4",
        "train.stats_every=4",
    )
    warm = TBPTTTrainer(_build(warm_cfg), warm_cfg, _cpu(), run_dir=tmp_path / "warm")
    hist_w = warm.fit(train_ds, dev_ds, 4)
    recs = [r for r in hist_w.steps if isinstance(r, TBPTTStepRecord)]
    assert "reinforce" not in recs[0].parts and "reinforce" not in recs[1].parts
    assert recs[0].stats["route_cost"] == 1.0 and recs[2].stats["route_cost"] == pytest.approx(
        K / (N * M), abs=1e-6
    )
    assert "reinforce" in recs[2].parts and "reinforce" in recs[3].parts


def test_p2_2_random_and_fixed_modes_train(
    train_docs: list[Example],
    dev_docs: list[Example],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The controls (random, fixed) and the learned router (topk + STE) train end to end on kv
    in-range: the task loss decreases, every objective is finite, the routing statistics are
    logged, and the trainer hands every window its step."""
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)
    runs = {
        "random": ["model.routing=random"],
        "fixed": ["model.routing=fixed", "model.fixed_route=[1, 4]", "train.gate_mode=ste"],
        "topk_ste": [
            "model.routing=topk",
            "train.gate_mode=ste",
            "train.epsilon_start=0.1",
            "train.epsilon_end=0.1",
        ],
    }
    seen_steps: list[int] = []
    for name, overrides in runs.items():
        cfg = _train_config(*overrides)
        model = _build(cfg)
        trainer = TBPTTTrainer(model, cfg, _cpu(), run_dir=tmp_path / name)
        if name == "topk_ste":

            def spy(step: int, original: Any = model.set_train_step) -> None:
                seen_steps.append(step)
                original(step)

            monkeypatch.setattr(model, "set_train_step", spy)
        history = trainer.fit(train_ds, dev_ds, 40)
        records = [r for r in history.steps if isinstance(r, TBPTTStepRecord)]
        assert len(records) == 40
        assert all(math.isfinite(r.total) for r in records), name
        first = sum(r.parts["task"] for r in records[:5]) / 5
        last = sum(r.parts["task"] for r in records[-5:]) / 5
        assert last < first, (name, first, last)
        assert all(
            r.stats["route_cost"] == pytest.approx(K / (N * M), abs=1e-6) for r in records
        ), name
        for key in ("explored", "refreshed", "dense_fallback"):
            assert key in records[-1].stats, (name, key)
        if name == "topk_ste":
            assert sum(r.stats["explored"] for r in records) > 0
        else:  # the controls take no probes
            assert all(
                r.stats["explored"] == 0.0 and r.stats["refreshed"] == 0.0 for r in records
            ), name
        assert history.evals[-1]["accuracy"] >= 0.0 and math.isfinite(history.evals[-1]["loss"])
        groups = {k[len("grad_norm/") :] for k in records[-1].stats if k.startswith("grad_norm/")}
        if name == "random":
            assert "router" not in groups  # random routing gives the scorer no gradient
        else:  # topk: mask and units; fixed under ste: the unit gradient
            assert "router" in groups, name
    assert seen_steps == list(range(40))


# --------------------------------------------------------------------------------------------------
# Further P2.2 behaviour
# --------------------------------------------------------------------------------------------------


def test_p2_2_schedule_values() -> None:
    schedule = SelectionSchedule(
        gate_mode="ste",
        soft_until=10,
        temperature_start=1.0,
        temperature_end=0.5,
        temperature_steps=100,
        epsilon_start=0.2,
        epsilon_end=0.0,
        epsilon_steps=50,
        p_dense_value=0.1,
    )
    assert schedule.gate(0) == "soft" and schedule.gate(9) == "soft" and schedule.gate(10) == "ste"
    assert schedule.temperature(0) == 1.0 and schedule.temperature(50) == pytest.approx(0.75)
    assert schedule.temperature(100) == 0.5 and schedule.temperature(10_000) == 0.5
    assert schedule.epsilon(0) == 0.2 and schedule.epsilon(25) == pytest.approx(0.1)
    assert schedule.epsilon(50) == 0.0 and schedule.epsilon(51) == 0.0
    assert schedule.p_dense(0) == schedule.p_dense(500) == 0.1 and not schedule.is_neutral()
    neutral = SelectionSchedule.neutral()
    assert neutral.gate(0) == "soft" and neutral.temperature(3) == 1.0 and neutral.epsilon(3) == 0.0
    assert neutral.p_dense(3) == 0.0 and neutral.is_neutral() and neutral.soft_until == 0
    cfg = _config(
        "model.routing=topk",
        "train.gate_mode=reinforce",
        "train.soft_until_step=3",
        "train.temperature_start=2.0",
        "train.temperature_end=1.0",
        "train.temperature_steps=4",
        "train.p_dense=0.25",
    )
    from_cfg = SelectionSchedule.from_config(cfg.train)
    assert from_cfg.gate(2) == "soft" and from_cfg.gate(3) == "reinforce"
    assert from_cfg.temperature(2) == 1.5 and from_cfg.p_dense(0) == 0.25
    assert HSN.from_config(cfg).schedule == from_cfg
    with pytest.raises(TypeError):
        SelectionSchedule.from_config(cfg)  # type: ignore[arg-type]
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="step"):
            schedule.temperature(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="never be reached"):
        SelectionSchedule(temperature_start=1.0, temperature_end=0.5, temperature_steps=0)
    with pytest.raises(ValueError, match="never be reached"):
        SelectionSchedule(epsilon_start=0.1, epsilon_end=0.0, epsilon_steps=0)
    with pytest.raises(ValueError, match="gate_mode"):
        SelectionSchedule(gate_mode="hard")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="temperature_start"):
        SelectionSchedule(temperature_start=0.0, temperature_end=0.0)
    with pytest.raises(ValueError, match="p_dense"):
        SelectionSchedule(p_dense_value=1.5)
    model = HSN.from_config(_config())
    assert model.train_step == 0
    model.set_train_step(7)
    assert model.train_step == 7
    for bad in (-1, 2.0, True):
        with pytest.raises(ValueError):
            model.set_train_step(bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="schedule"):
        HSN(cfg.model, seed=0, unroll=16, schedule=cfg.train)  # type: ignore[arg-type]
    bare = HSN(cfg.model, seed=0, unroll=cfg.train.unroll)  # the neutral schedule, not cfg's
    with pytest.raises(ValueError, match="selection schedule"):
        TBPTTTrainer(bare, cfg, _cpu())


def test_p2_2_dense_fallback_charged_in_cost() -> None:
    """A dense-fallback row executes the dense step (all cells, the softmax over units), is
    charged the dense cost and flagged; the draw follows ``train.p_dense`` and is training-only."""
    cfg = _config()
    router = _router(cfg)
    logits = _logits(3, seed=8)
    rows = torch.tensor([True, False, True])
    sel = router.select(logits, 1.0, K, 1, "topk", gate="ste", dense_rows=rows)
    plain = router.select(logits, 1.0, K, 1, "topk", gate="ste")
    assert torch.equal(sel.dense_fallback, rows)
    assert torch.equal(sel.cell_mask[rows], torch.ones(2, N))
    assert torch.equal(sel.unit_weights[rows], torch.softmax(logits[rows], dim=-1))
    assert torch.equal(sel.cost[rows], torch.ones(2)) and torch.equal(
        sel.expected_cost[rows], torch.ones(2)
    )
    assert torch.equal(sel.cell_mask[1], plain.cell_mask[1]) and torch.equal(
        sel.cost[1], plain.cost[1]
    )
    assert torch.equal(sel.cell_idx, plain.cell_idx)  # the hard decision is kept for the record
    policy = router.select(
        logits,
        1.0,
        K,
        1,
        "topk",
        gate="reinforce",
        dense_rows=rows,
        generator=torch.Generator(device="cpu").manual_seed(2),
    )
    assert bool((policy.log_prob[rows] == 0).all()) and bool((policy.log_prob[~rows] < 0).all())
    assert torch.equal(sel.selected_cells()[rows], torch.ones(2, N, dtype=torch.bool))
    assert torch.equal(sel.selected_cells()[1], _hard_mask(plain)[1] > 0)
    assert not sel.cell_mask[rows].requires_grad or float(sel.cell_mask[rows].sum()) == 2 * N
    with pytest.raises(ValueError, match="dense_rows"):
        router.select(logits, 1.0, K, 1, "dense", dense_rows=rows)
    with pytest.raises(ValueError, match="shape"):
        router.select(logits, 1.0, K, 1, "topk", dense_rows=rows[:2])
    with pytest.raises(TypeError):
        router.select(logits, 1.0, K, 1, "topk", dense_rows=rows.float())

    # p_dense = 1: every row of every position runs dense and the window equals the dense model
    always = _config("model.routing=topk", "train.gate_mode=ste", "train.p_dense=1.0")
    dense_cfg = _config("model.routing=dense")
    model, dense = _build(always, perturb=4), _build(dense_cfg)
    dense.load_state_dict(model.state_dict())
    tokens, ctx, reset = _window(always)
    logits_m, state_m, info_m = model(tokens, ctx, model.init_state(4), reset)
    logits_d, state_d, _ = dense(tokens, ctx, dense.init_state(4), reset)
    assert torch.equal(logits_m, logits_d) and torch.equal(state_m.h, state_d.h)
    assert torch.equal(state_m.age, state_d.age)
    assert bool((info_m["dense_fallback"] == 1.0).all()) and bool((info_m["cost"] == 1.0).all())
    # p_dense = 0.25: the rate is measured per window; evaluation never falls back
    some = _config("model.routing=topk", "train.gate_mode=ste", "train.p_dense=0.25")
    sampled = _build(some)
    B = 400
    tokens, ctx, reset = _window(some, B=B, T=4)
    torch.manual_seed(0)
    _, _, info_s = sampled(tokens, ctx, sampled.init_state(B), reset)
    rate = float(info_s["dense_fallback"].mean())
    assert abs(rate - 0.25) < 4 * math.sqrt(0.25 * 0.75 / B), rate
    assert bool((info_s["dense_fallback"] == rate).all())  # one draw per row per window
    expected_cost = rate * 1.0 + (1 - rate) * K / (N * M)
    assert float(info_s["cost"].mean()) == pytest.approx(expected_cost, abs=1e-5)
    sampled.eval()
    with torch.no_grad():
        _, _, info_e = sampled(tokens, ctx, sampled.init_state(B), reset)
    assert float(info_e["dense_fallback"].sum()) == 0.0
    with pytest.raises(ValueError, match="dense_rows"):
        sampled.step(
            tokens[:, 0],
            ctx[:, 0],
            sampled.init_state(B),
            "dense",
            dense_rows=torch.ones(B, dtype=torch.bool),
        )


def test_p2_2_budget_kept_after_post_processing() -> None:
    """Random tables, budgets, ages and ε = 1: every selection keeps the P2.1 invariants (cost
    within budget, distinct cells, k bounds), and the probes apply exactly where the budget
    allows."""
    cfg = _config()
    scorer = _scorer(cfg)
    generator = torch.Generator(device="cpu").manual_seed(2022)
    n_refreshed = n_explored = n_reverted = 0
    for _trial in range(60):
        unit = (0.01 + 0.09 * torch.rand(M, generator=generator, dtype=torch.float64)).tolist()
        shared = float(torch.rand(1, generator=generator)) * 0.2
        overhead = float(torch.rand(1, generator=generator)) * 0.03
        table = _table(unit, shared=shared, overhead=overhead)
        router = Router(scorer, table, lambda_cost=float(torch.rand(1, generator=generator)))
        B = int(torch.randint(1, 6, (1,), generator=generator))
        k_max = int(torch.randint(1, N + 1, (1,), generator=generator))
        k_min = int(torch.randint(0, k_max + 1, (1,), generator=generator))
        floor = table.shared + k_min * table.max_pair_cost()
        ceiling = max(floor, table.dense_cost())
        budget = floor + float(torch.rand(1, generator=generator)) * (ceiling - floor)
        logits = torch.randn(B, N, M, generator=generator)
        age = torch.randint(0, 2 * cfg.model.age_max, (B, N), generator=generator)
        for mode in SCORED_MODES:
            for gate in (None, "soft", "ste"):
                plain = router.select(logits, budget, k_max, k_min, mode)  # type: ignore[arg-type]
                sel = router.select(
                    logits,
                    budget,
                    k_max,
                    k_min,
                    mode,
                    generator=generator,  # type: ignore[arg-type]
                    gate=gate,
                    temperature=0.5,
                    epsilon=1.0,
                    age=age,  # type: ignore[arg-type]
                )
                if gate != "soft":  # the warm-up executes every cell and reports the dense cost
                    _check_invariants(sel, table, k_max=k_max, k_min=k_min, budget=budget)
                assert torch.equal(sel.valid, plain.valid)
                changed = (sel.cell_idx != plain.cell_idx).any(dim=1)
                flagged = (sel.explored | sel.refreshed).any(dim=1)
                assert torch.equal(changed, flagged)
                n_refreshed += int(sel.refreshed.sum())
                n_explored += int(sel.explored.sum())
                # a row without flags kept its scored selection: either nothing was stale and
                # nothing could be explored, or the probe would have exceeded the budget
                for b in (~flagged).nonzero().flatten().tolist():
                    stale = ((age[b] > cfg.model.age_max) & ~(_hard_mask(plain)[b] > 0)).any()
                    explorable = int(plain.n_valid()[b]) > 0 and int(plain.n_valid()[b]) < N
                    if stale or explorable:
                        n_reverted += 1
                if gate == "soft":
                    assert torch.allclose(sel.cost, torch.full_like(sel.cost, table.dense_cost()))
                else:
                    assert torch.allclose(sel.expected_cost, sel.cost, rtol=1e-5, atol=1e-6)
    assert n_refreshed > 0 and n_explored > 0 and n_reverted > 0


def test_p2_2_expected_cost_matches_cost_in_hard_modes() -> None:
    """``expected_cost`` is the differentiable twin of ``cost`` in every hard mode and the relaxed
    selection's expectation in the soft warm-up; the P2.1 ``dense`` mode carries the new fields."""
    cfg = _config()
    table = _table([0.02, 0.05, 0.11], shared=0.1, overhead=0.01)
    router = Router(_scorer(cfg), table, lambda_cost=0.3)
    logits = _logits(3, seed=9).requires_grad_(True)
    budget = table.dense_cost()
    for mode, kwargs in (
        ("topk", {}),
        ("greedy_budget", {}),
        ("random", {"generator": torch.Generator(device="cpu").manual_seed(1)}),
        ("fixed", {"fixed": torch.tensor([2, 5])}),
    ):
        for gate in (None, "ste"):
            sel = router.select(logits, budget, K, 1, mode, gate=gate, **kwargs)  # type: ignore[arg-type]
            assert torch.allclose(sel.expected_cost, sel.cost, rtol=1e-5, atol=1e-6), (mode, gate)
            if gate == "ste" and mode != "random":
                assert sel.expected_cost.requires_grad
                sel.expected_cost.sum().backward()
                assert logits.grad is not None
                logits.grad = None
    soft = router.select(logits, budget, K, 1, "topk", gate="soft", temperature=2.0)
    by_hand = router.shared_cost + (
        soft.cell_mask * (router.cell_overhead + (soft.unit_weights * router.cost_table).sum(-1))
    ).sum(1)
    assert torch.allclose(soft.expected_cost, by_hand)
    assert bool((soft.cell_mask > 0).all()) and bool((soft.cell_mask < 1).all())
    assert torch.allclose(soft.cost, torch.full((3,), table.dense_cost()))
    dense = router.select(logits, budget, K, 1, "dense")
    assert torch.equal(dense.cell_mask, torch.ones(3, N))
    assert torch.allclose(dense.unit_weights, torch.softmax(logits, dim=-1))
    assert torch.allclose(dense.expected_cost, dense.cost) and not dense.dense_fallback.any()
    assert dense.explored.shape == (3, N) and not dense.explored.any()
    assert torch.equal(dense.selected_cells(), torch.ones(3, N, dtype=torch.bool))
    for mode, kwargs in (
        ("fixed", {"fixed": torch.tensor([2, 5])}),
        ("random", {"generator": torch.Generator(device="cpu").manual_seed(1)}),
    ):  # the controls stay hard in the warm-up
        control = router.select(logits, budget, K, 1, mode, gate="soft", **kwargs)  # type: ignore[arg-type]
        assert set(control.unit_weights.detach().unique().tolist()) <= {0.0, 1.0}
        assert set(control.cell_mask.unique().tolist()) <= {0.0, 1.0}
        assert torch.allclose(control.expected_cost, control.cost)
        assert not control.cell_mask.requires_grad and not control.unit_weights.requires_grad
    soft_dense = router.select(logits, budget, K, 1, "dense", gate="soft")
    assert torch.equal(soft_dense.unit_weights, dense.unit_weights)
    with pytest.raises(ValueError, match="dense"):
        router.select(logits, budget, K, 1, "dense", gate="ste")


def test_p2_2_eval_mode_is_hard_and_quiet() -> None:
    """In evaluation the routed model takes the hard, gradient-free decision: no exploration, no
    dense fallback, the hard cost, deterministic repeats; training with the same weights explores
    and falls back."""
    cfg = _config(
        "model.routing=topk",
        "train.gate_mode=ste",
        "train.epsilon_start=0.5",
        "train.epsilon_end=0.5",
        "train.p_dense=0.5",
        "model.age_max=100",
    )
    model = _build(cfg, perturb=2)
    tokens, ctx, reset = _window(cfg, B=32, T=8)
    model.eval()
    with torch.no_grad():
        logits_a, state_a, info_a = model(tokens, ctx, model.init_state(32), reset)
        logits_b, state_b, info_b = model(tokens, ctx, model.init_state(32), reset)
        masked = model.step(tokens[:, 0], ctx[:, 0], model.init_state(32), "masked")
        sparse = model.step(tokens[:, 0], ctx[:, 0], model.init_state(32), "sparse")
    assert torch.equal(logits_a, logits_b) and torch.equal(state_a.h, state_b.h)
    # the masked path is the sparse round's reference (P2.5): the same decision, the same state
    assert masked.selection is not None and sparse.selection is not None
    assert torch.equal(masked.selection.cell_idx, sparse.selection.cell_idx)
    torch.testing.assert_close(masked.logits, sparse.logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(masked.state.h, sparse.state.h, atol=1e-5, rtol=1e-5)
    assert torch.equal(masked.state.age, sparse.state.age) and torch.equal(masked.cost, sparse.cost)
    assert float(info_a["explored"].sum()) == 0.0 and float(info_a["dense_fallback"].sum()) == 0.0
    assert torch.allclose(info_a["cost"], torch.full_like(info_a["cost"], K / (N * M)))
    assert not logits_a.requires_grad and not info_a["expected_cost"].requires_grad
    model.train()
    torch.manual_seed(0)
    _, _, info_t = model(tokens, ctx, model.init_state(32), reset)
    assert float(info_t["explored"].mean()) > 0.0 and float(info_t["dense_fallback"].mean()) > 0.0
    assert info_t["expected_cost"].requires_grad
    out = model.step(tokens[:, 0], ctx[:, 0], model.init_state(32), "masked")
    assert isinstance(out, StepOutput) and out.selection is not None and out.cost.shape == ()
    with pytest.raises(ValueError, match="mode"):
        model.step(tokens[:, 0], ctx[:, 0], model.init_state(32), "routed")  # type: ignore[arg-type]
    dense_model = _build(_config("model.routing=dense"))
    with pytest.raises(ValueError, match="model.routing=dense"):
        dense_model.step(tokens[:, 0], ctx[:, 0], dense_model.init_state(32), "masked")


def test_p2_2_sparse_round_takes_the_gradient() -> None:
    """P2.2 on P2.5's sparse round: the same hard decision and state as the masked reference, a
    finite non-zero scorer gradient through ``Selection.alpha`` (STE), and dense-fallback rows that
    are the dense step bit for bit while the other rows stay sparse."""
    cfg = _config("model.routing=topk", "train.gate_mode=ste")
    model = _build(cfg, perturb=11)
    tokens, ctx, reset = _window(cfg, B=4, T=1)
    x, c = tokens[:, 0], ctx[:, 0]
    state = model.init_state(4)
    masked = model.step(x, c, state, "masked")
    sparse = model.step(x, c, state, "sparse")
    assert masked.selection is not None and sparse.selection is not None
    assert torch.equal(masked.selection.cell_idx, sparse.selection.cell_idx)
    assert torch.equal(masked.selection.unit_idx, sparse.selection.unit_idx)
    torch.testing.assert_close(sparse.logits, masked.logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sparse.state.h, masked.state.h, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sparse.state.q, masked.state.q, atol=1e-5, rtol=1e-5)
    assert torch.equal(sparse.state.age, masked.state.age) and torch.equal(sparse.cost, masked.cost)
    assert float(sparse.cost) == pytest.approx(K / (N * M), abs=1e-6)
    sparse.logits.square().mean().backward()
    grad = model.router.scorer.net.weight.grad
    assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.norm()) > 1e-8
    assert sparse.selection.alpha.requires_grad  # the gradient entry of the sparse round
    # dense-fallback rows: the dense step bit for bit, the other rows the sparse round
    rows = torch.tensor([True, False, True, False])
    mixed = model.step(x, c, state, "sparse", dense_rows=rows)
    dense = model.step(x, c, state, "dense")
    plain = model.step(x, c, state, "sparse")
    assert mixed.selection is not None and torch.equal(mixed.selection.dense_fallback, rows)
    assert torch.equal(mixed.logits[rows], dense.logits[rows])
    assert torch.equal(mixed.state.h[rows], dense.state.h[rows])
    assert torch.equal(mixed.state.age[rows], dense.state.age[rows])
    assert torch.equal(mixed.logits[~rows], plain.logits[~rows])
    assert torch.equal(mixed.state.h[~rows], plain.state.h[~rows])
    assert torch.equal(mixed.state.age[~rows], plain.state.age[~rows])
    assert torch.equal(mixed.state.pos, dense.state.pos)
    assert float(mixed.cost) == pytest.approx(0.5 * 1.0 + 0.5 * K / (N * M), abs=1e-6)
    with pytest.raises(ValueError, match="dense_rows"):
        model.step(x, c, state, "dense", dense_rows=rows)
    # a window in training: every position stacks the P2.5 decisions and the P2.2 statistics
    logits_w, _, info = model(tokens, ctx, model.init_state(4), reset)
    assert {
        "cell_idx",
        "unit_idx",
        "valid",
        "cost_per_row",
        "expected_cost",
        "route_log_prob",
    } <= set(info)
    assert info["cell_idx"].shape == (4, 1, K) and info["cost_per_row"].shape == (4, 1)


@pytest.mark.parametrize("device_type", DEVICES)
def test_p2_2_runs_on_device(device_type: str) -> None:
    device = _device(device_type)
    cfg = _config(
        "model.routing=topk",
        "train.gate_mode=ste",
        "train.epsilon_start=0.3",
        "train.epsilon_end=0.3",
        "train.p_dense=0.2",
        "model.age_max=3",
        "train.temperature_start=1.0",
        "train.temperature_end=0.5",
        "train.temperature_steps=10",
    )
    model = _build(cfg, perturb=6).to(device)
    model.set_train_step(5)
    tokens, ctx, reset = (t.to(device) for t in _window(cfg, B=8, T=12))
    torch.manual_seed(0)
    logits, state, info = model(tokens, ctx, model.init_state(8), reset)
    targets = tokens.roll(-1, dims=1)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )
    loss.backward()
    assert bool(torch.isfinite(loss)) and all(t.device.type == device.type for t in info.values())
    grad = model.router.scorer.net.weight.grad
    assert grad is not None and bool(torch.isfinite(grad).all()) and grad.device.type == device.type
    assert int(state.age.max()) <= AGE_BOUND(3)
    out = model.step(tokens[:, 0], ctx[:, 0], model.init_state(8), "masked")
    assert out.selection is not None and out.selection.explored.device.type == device.type
    assert out.cost.device.type == device.type
    for gate in ("soft", "reinforce"):
        gated = _build(_config("model.routing=topk", f"train.gate_mode={gate}"), perturb=6).to(
            device
        )
        logits_g, _, info_g = gated(tokens, ctx, gated.init_state(8), reset)
        assert bool(torch.isfinite(logits_g).all()) and bool(
            torch.isfinite(info_g["route_log_prob"]).all()
        )
        if gate == "reinforce":
            assert bool((info_g["route_log_prob"] < 0).all())
    # the cpu oracle: the same decisions as on the device for the deterministic parts
    if device.type != "cpu":
        cpu = _build(cfg, perturb=6)
        cpu.eval()
        model.eval()
        with torch.no_grad():
            _, state_c, info_c = cpu(tokens.cpu(), ctx.cpu(), cpu.init_state(8), reset.cpu())
            _, state_g, info_g = model(tokens, ctx, model.init_state(8), reset)
        torch.testing.assert_close(info_c["cost"], info_g["cost"].cpu())
        assert (
            int(state_g.age.max()) <= AGE_BOUND(3) and float(info_g["dense_fallback"].sum()) == 0.0
        )


def test_p2_2_config_fields(tmp_path: Path) -> None:
    cfg = load_config(REF_A1)
    assert (
        cfg.train.reinforce_baseline_decay == 0.9 and TrainConfig().reinforce_baseline_decay == 0.9
    )
    assert load_config(EXAMPLE).train.reinforce_baseline_decay == 0.9
    for key in (
        "gate_mode",
        "soft_until_step",
        "temperature_start",
        "temperature_end",
        "temperature_steps",
        "epsilon_start",
        "epsilon_end",
        "epsilon_steps",
        "p_dense",
        "reinforce_baseline_decay",
    ):
        assert key in TrainConfig.model_fields
    assert cfg.train.gate_mode == "soft" and cfg.train.p_dense == 0.0 and cfg.model.age_max == 64
    with pytest.raises(ValueError, match="temperature_end"):
        load_config(REF_A1, overrides=["train.temperature_end=0.5"])
    with pytest.raises(ValueError, match="epsilon_end"):
        load_config(REF_A1, overrides=["train.epsilon_start=0.1"])
    ramp = load_config(
        REF_A1, overrides=["train.temperature_end=0.5", "train.temperature_steps=100"]
    )
    assert ramp.train.temperature_steps == 100
    for bad in ("1.0", "-0.1"):
        with pytest.raises(ValueError, match="reinforce_baseline_decay"):
            load_config(REF_A1, overrides=[f"train.reinforce_baseline_decay={bad}"])
    with pytest.raises(ValueError, match="gate_mode"):
        load_config(REF_A1, overrides=["train.gate_mode=hard"])
    with pytest.raises(ValueError, match="needs model.routing=topk"):
        load_config(REF_A1, overrides=["model.routing=random", "train.gate_mode=reinforce"])
    with pytest.raises(ValueError, match="no selection to explore"):
        load_config(REF_A1, overrides=["train.p_dense=0.1"])
    with pytest.raises(ValueError, match="no selection to explore"):
        load_config(REF_A1, overrides=["train.epsilon_start=0.1", "train.epsilon_end=0.1"])
    with pytest.raises(ValueError, match="control"):
        load_config(
            REF_A1,
            overrides=[
                "model.routing=fixed",
                "model.fixed_route=[0]",
                "train.epsilon_start=0.1",
                "train.epsilon_end=0.1",
            ],
        )
    sampled = load_config(REF_A1, overrides=["model.routing=random", "train.p_dense=0.1"])
    assert sampled.train.p_dense == 0.1  # the dense fallback applies to the controls
    assert GATE_MODES == ("soft", "ste", "reinforce")
    # ref_a1 unchanged: the P2.2 keys sit at their defaults and the resolved config is the same
    resolved = tmp_path / "resolved.yaml"
    from hsn.config import dump_resolved

    dump_resolved(cfg, resolved)
    assert load_config(resolved) == cfg
    assert cfg == load_config(EXAMPLE, overrides=["run_id=ref_a1", "notes=" + cfg.notes])
    # the model keeps its P1.5 count and report rows (the router owns the scorer, no extra params)
    model = HSN.from_config(cfg)
    report = model.param_report()
    assert report["total"] == 1_820_617 and report["scorer"] == 1_608  # the A1 total
    assert model.router.param_count() == 1_608 and "_fixed_route" not in model.state_dict()
    fixed = HSN.from_config(
        load_config(REF_A1, overrides=["model.routing=fixed", "model.fixed_route=[3, 1]"])
    )
    assert fixed._fixed_route is not None and fixed._fixed_route.tolist() == [3, 1]
    assert model._fixed_route is None
    # a config the budget cannot hold is refused when the model is built
    with pytest.raises(ValueError, match="cannot hold"):
        HSN.from_config(load_config(REF_A1, overrides=["model.routing=topk", "model.budget=0.0"]))
    state = CellGraphState.zeros(2, cfg.model.n_cells, cfg.model.d, "cpu", torch.float32)
    assert model.reset_state(state, torch.tensor([True, False])).pos.tolist() == [0, 0]
