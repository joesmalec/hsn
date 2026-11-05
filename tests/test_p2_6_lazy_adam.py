"""P2.6 tests: ``LazyAdamW`` equals AdamW when every unit is active, inactive units receive no
update, the per-unit bias correction matches independent Adams, the state (with its per-unit
steps) round-trips through a checkpoint, the active-unit mask from selections, the trainer's unit
groups and mask hand-off, a bit-exact resume of a sparse run and the routing-dependent optimiser
default."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from hsn.config import RunConfig, canonical_json, load_config, load_hardware_profile
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example, TaskSpec, generate, load_dataset_spec
from hsn.device import DeviceContext, resolve, seed_everything, set_deterministic
from hsn.model.hsn import HSN
from hsn.model.router import Selection
from hsn.model.units import UnitBank
from hsn.train.lazy_adam import (
    LazyAdamW,
    active_units_from_indices,
    build_lazy_adamw,
    find_unit_bank,
)
from hsn.train.tbptt import TBPTTStepRecord, TBPTTTrainer
from torch import Tensor, nn

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
EXAMPLE = REPO / "configs" / "_example.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
UNROLL = 32
#: The chunk's agreement tolerance against torch.optim.AdamW (fp32, 50 steps).
ATOL = 1e-7

cuda_param = pytest.param("cuda", marks=pytest.mark.cuda)
HYPER = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8}


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _restore_determinism() -> Iterator[None]:
    """The resume test sets ``train.deterministic``, which the trainer applies process-wide."""
    before = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        set_deterministic(before)


def _device(device_type: str) -> torch.device:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device on this machine")
    return torch.empty(0, device=device_type).device


def _cpu_ctx() -> DeviceContext:
    return resolve(load_hardware_profile("cpu"))


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


@pytest.fixture(scope="module")
def train_docs() -> list[Example]:
    task = _tiny_task()
    return [generate(task, "train", i) for i in range(64)]  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def dev_docs() -> list[Example]:
    task = _tiny_task()
    return [generate(task, "dev", i) for i in range(16)]  # type: ignore[arg-type]


def _train_config(*overrides: str) -> RunConfig:
    """The P2.5 trainer recipe: ref_a1 shrunk, ``routing=topk`` with the straight-through gate."""
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
            "model.routing=topk",
            "train.gate_mode=ste",
            f"train.unroll={UNROLL}",
            "train.batch=4",
            "train.micro_batch=4",
            "train.steps=100",
            "train.eval_every=50",
            "train.ckpt_every=100",
            "train.stats_every=1",
            *overrides,
        ],
    )


def _lazy_config(*overrides: str) -> RunConfig:
    """A routed run small enough that units stay inactive: one slot per position, one stream,
    sixteen positions per window, eight units."""
    return _train_config(
        "model.n_units=8",
        "model.k_select=1",
        "train.unroll=16",
        "train.batch=1",
        "train.micro_batch=1",
        *overrides,
    )


def _selection(
    unit_idx: list[list[int]],
    valid: list[list[bool]],
    dense: list[bool],
    *,
    n_cells: int = 4,
    n_units: int = 8,
) -> Selection:
    units = torch.tensor(unit_idx, dtype=torch.int64)
    B, K = units.shape
    return Selection(
        cell_idx=torch.arange(K).repeat(B, 1),
        unit_idx=units,
        alpha=torch.ones(B, K),
        valid=torch.tensor(valid),
        logits=torch.zeros(B, n_cells, n_units),
        cost=torch.ones(B),
        scorer_cost=0.0,
        n_rounds_used=1,
        cell_mask=torch.ones(B, n_cells),
        unit_weights=torch.zeros(B, n_cells, n_units),
        expected_cost=torch.ones(B),
        log_prob=torch.zeros(B),
        explored=torch.zeros(B, K, dtype=torch.bool),
        refreshed=torch.zeros(B, K, dtype=torch.bool),
        dense_fallback=torch.tensor(dense),
    )


def _params(device: torch.device, dtype: torch.dtype = torch.float32) -> list[Tensor]:
    """Two stacked unit tensors and two ordinary ones, |p| < 0.5 so one fp32 ulp is < 1e-7."""
    M, a, b = 8, 16, 32
    shapes = [(M, a, b), (M, a), (a, b), (b,)]
    return [
        (torch.randn(*shape, device=device, dtype=dtype) * 0.05).requires_grad_()
        for shape in shapes
    ]


def _groups(params: list[Tensor], *, lazy: bool) -> list[dict[str, object]]:
    unit: dict[str, object] = {"unit_dim": 0, "n_units": 8} if lazy else {}
    return [
        {"params": [params[2]], "weight_decay": 0.01},
        {"params": [params[3]], "weight_decay": 0.0},
        {"params": [params[0]], "weight_decay": 0.01, **unit},
        {"params": [params[1]], "weight_decay": 0.0, **unit},
    ]


def _max_diff(xs: list[Tensor], ys: list[Tensor]) -> float:
    return max(float((x.detach() - y.detach()).abs().max()) for x, y in zip(xs, ys, strict=True))


def _same(a: float, b: float) -> bool:
    """Equality that holds for two NaNs (a window without loss positions records a NaN loss)."""
    return a == b or (math.isnan(a) and math.isnan(b))


# --------------------------------------------------------------------------------------------------
# The optimiser
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", ["cpu", cuda_param])
def test_p2_6_equals_adamw_when_all_active(device_type: str) -> None:
    """Fifty steps with the same gradients, a learning-rate change half way, the mask ``None`` on
    even steps and all-True on odd ones: every parameter and both moments agree with
    ``torch.optim.AdamW`` within 1e-7; unit steps are ``[M]`` int64 on the CPU, ordinary ones
    0-dim."""
    device = _device(device_type)
    torch.manual_seed(0)
    ref = _params(device)
    lazy = [p.detach().clone().requires_grad_() for p in ref]
    opt_ref = torch.optim.AdamW(_groups(ref, lazy=False), foreach=False, **HYPER)
    opt = LazyAdamW(_groups(lazy, lazy=True), **HYPER)
    assert opt.n_units == 8
    worst = 0.0
    for step in range(50):
        if step == 25:
            for group in (*opt_ref.param_groups, *opt.param_groups):
                group["lr"] *= 0.5
        grads = [torch.randn_like(p) * 0.1 for p in ref]
        for p, g in zip(ref, grads, strict=True):
            p.grad = g.clone()
        for p, g in zip(lazy, grads, strict=True):
            p.grad = g.clone()
        opt_ref.step()
        mask = None if step % 2 == 0 else torch.ones(8, dtype=torch.bool, device=device)
        opt.step(mask)
        worst = max(worst, _max_diff(ref, lazy))
    assert worst < ATOL, worst
    for p, q in zip(ref, lazy, strict=True):
        for key in ("exp_avg", "exp_avg_sq"):
            assert float((opt_ref.state[p][key] - opt.state[q][key]).abs().max()) < ATOL, key
    for index in (0, 1):
        step_t = opt.state[lazy[index]]["step"]
        assert step_t.dtype is torch.int64 and step_t.device.type == "cpu"
        assert torch.equal(step_t, torch.full((8,), 50, dtype=torch.int64))
    for index in (2, 3):
        step_t = opt.state[lazy[index]]["step"]
        assert step_t.dtype is torch.int64 and step_t.ndim == 0 and int(step_t) == 50
    assert all(torch.isfinite(p).all() for p in lazy)


def test_p2_6_inactive_units_unchanged() -> None:
    """Random masks, non-zero (once non-finite) gradients on every unit: the weights, both moments
    and the step of an inactive unit are unchanged by the step, active units move, a unit that is
    never active stays at its initial values with step 0, and nothing becomes non-finite."""
    torch.manual_seed(1)
    M = 8
    params = _params(torch.device("cpu"))
    init = [p.detach().clone() for p in params]
    opt = LazyAdamW(_groups(params, lazy=True), weight_decay=0.01, **HYPER)
    for step in range(30):
        mask = torch.rand(M) < 0.5
        mask[M - 1] = False  # never active
        mask[step % (M - 1)] = True  # at least one active unit
        for p in params:
            p.grad = torch.randn_like(p) * 0.1
        if step == 5:  # a non-finite gradient on an inactive unit is ignored, not propagated
            params[0].grad[M - 1] = float("nan")
        before = [p.detach().clone() for p in params]
        states = [
            {key: value.clone() for key, value in opt.state[p].items()} if p in opt.state else {}
            for p in params
        ]
        opt.step(mask)
        for index in (0, 1):
            p, old, state = params[index], before[index], states[index]
            for j in range(M):
                if mask[j]:
                    assert not torch.equal(p[j], old[j]), (step, index, j)
                    continue
                assert torch.equal(p[j], old[j]), (step, index, j)
                if state:
                    for key in ("exp_avg", "exp_avg_sq"):
                        assert torch.equal(opt.state[p][key][j], state[key][j]), (step, key, j)
                    assert int(opt.state[p]["step"][j]) == int(state["step"][j])
        for index in (2, 3):  # ordinary tensors step every time
            assert not torch.equal(params[index], before[index])
        assert all(torch.isfinite(p).all() for p in params), step
    for index in (0, 1):
        assert torch.equal(params[index][M - 1], init[index][M - 1])
        assert int(opt.state[params[index]]["step"][M - 1]) == 0
        assert torch.equal(
            opt.state[params[index]]["exp_avg"][M - 1], torch.zeros_like(init[index][M - 1])
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_p2_6_per_unit_bias_correction_matches_separate_adams(dtype: torch.dtype) -> None:
    """One stacked tensor under ``LazyAdamW`` with random masks equals ``M`` independent
    ``torch.optim.AdamW`` instances, each stepped only when its unit is active: parameters and
    moments within 1e-7 after 40 steps, the per-unit step counters equal to the counts of active
    steps; unit 0 is inactive at step 0 (the 0/0 guard) and the last unit never active."""
    torch.manual_seed(2)
    M, a, b = 8, 16, 32
    hyper = {**HYPER, "weight_decay": 0.01}
    stacked = (torch.randn(M, a, b, dtype=dtype) * 0.05).requires_grad_()
    opt = LazyAdamW([{"params": [stacked], "unit_dim": 0, "n_units": M}], **hyper)
    singles = [stacked[j].detach().clone().requires_grad_() for j in range(M)]
    opts = [torch.optim.AdamW([p], foreach=False, **hyper) for p in singles]
    counts = torch.zeros(M, dtype=torch.int64)
    for step in range(40):
        mask = torch.rand(M) < 0.6
        mask[M - 1] = False
        mask[0] = step != 0 and bool(mask[0])
        mask[1 + step % (M - 2)] = True
        grad = torch.randn(M, a, b, dtype=dtype) * 0.1
        stacked.grad = grad.clone()
        opt.step(mask)
        for j in range(M):
            if mask[j]:
                singles[j].grad = grad[j].clone()
                opts[j].step()
                counts[j] += 1
    together = torch.stack([p.detach() for p in singles])
    assert float((stacked.detach() - together).abs().max()) < ATOL
    state = opt.state[stacked]
    assert torch.equal(state["step"], counts) and int(counts[M - 1]) == 0 and int(counts[0]) < 40
    for key in ("exp_avg", "exp_avg_sq"):
        for j in range(M - 1):
            assert float((state[key][j] - opts[j].state[singles[j]][key]).abs().max()) < ATOL, key
        assert torch.equal(state[key][M - 1], torch.zeros(a, b, dtype=dtype))
    assert torch.equal(stacked[M - 1].detach(), singles[M - 1].detach())


@pytest.mark.parametrize("device_type", ["cpu", cuda_param])
def test_p2_6_state_checkpoint_roundtrip(device_type: str, tmp_path: Path) -> None:
    """``state_dict`` → ``torch.save`` → ``torch.load(map_location="cpu")`` → a fresh optimiser:
    the continuation equals the original step for step; the per-unit steps come back as int64
    ``[M]`` on the CPU and the moments on the parameters' device; a plain AdamW state and a state
    whose ``n_units`` disagrees with the parameters are refused."""
    device = _device(device_type)
    torch.manual_seed(3)
    M = 8
    params = _params(device)
    opt = LazyAdamW(_groups(params, lazy=True), **HYPER)
    masks = [torch.rand(M) < 0.5 for _ in range(20)]
    for mask in masks:
        mask[0] = True
    grads = [[torch.randn_like(p) * 0.1 for p in params] for _ in range(20)]

    def run(optimizer: LazyAdamW, ps: list[Tensor], steps: range) -> None:
        for step in steps:
            for p, g in zip(ps, grads[step], strict=True):
                p.grad = g.clone()
            optimizer.step(masks[step].to(device))

    run(opt, params, range(10))
    torch.save(opt.state_dict(), tmp_path / "opt.pt")
    loaded = torch.load(tmp_path / "opt.pt", map_location="cpu", weights_only=True)
    fresh_params = [p.detach().clone().requires_grad_() for p in params]
    fresh = LazyAdamW(_groups(fresh_params, lazy=True), **HYPER)
    assert not fresh.state
    fresh.load_state_dict(loaded)
    for p, q in zip(params, fresh_params, strict=True):
        step_t = fresh.state[q]["step"]
        assert step_t.dtype is torch.int64 and step_t.device.type == "cpu"
        assert torch.equal(step_t, opt.state[p]["step"])
        for key in ("exp_avg", "exp_avg_sq"):
            assert fresh.state[q][key].device == q.device
            assert torch.equal(fresh.state[q][key], opt.state[p][key])
    assert tuple(fresh.state[fresh_params[0]]["step"].shape) == (M,)
    run(opt, params, range(10, 20))
    run(fresh, fresh_params, range(10, 20))
    for p, q in zip(params, fresh_params, strict=True):
        assert torch.equal(p, q)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(opt.state[p][key], fresh.state[q][key]), key
    # refusals
    adamw_params = [p.detach().clone().requires_grad_() for p in params]
    adamw = torch.optim.AdamW(_groups(adamw_params, lazy=False), foreach=False, **HYPER)
    for p in adamw_params:
        p.grad = torch.zeros_like(p)
    adamw.step()
    with pytest.raises(ValueError, match="step state"):
        LazyAdamW(_groups(fresh_params, lazy=True), **HYPER).load_state_dict(adamw.state_dict())
    wrong = torch.load(tmp_path / "opt.pt", map_location="cpu", weights_only=True)
    wrong["param_groups"][2]["n_units"] = M + 1
    with pytest.raises(ValueError, match="n_units"):
        LazyAdamW(_groups(fresh_params, lazy=True), **HYPER).load_state_dict(wrong)


def test_p2_6_group_validation() -> None:
    """A unit group needs both keys, ``unit_dim=0``, tensors ``[n_units, ...]`` and one ``M``;
    hyperparameters and the mask handed to ``step`` are checked; the base API still works."""
    w = torch.zeros(4, 3, requires_grad=True)
    v = torch.zeros(5, 3, requires_grad=True)
    with pytest.raises(ValueError, match="both"):
        LazyAdamW([{"params": [w], "n_units": 4}])
    with pytest.raises(ValueError, match="unit_dim"):
        LazyAdamW([{"params": [w], "unit_dim": 1, "n_units": 3}])
    with pytest.raises(ValueError, match=r"\[n_units=5"):
        LazyAdamW([{"params": [w], "unit_dim": 0, "n_units": 5}])
    with pytest.raises(ValueError, match="disagree"):
        LazyAdamW(
            [
                {"params": [w], "unit_dim": 0, "n_units": 4},
                {"params": [v], "unit_dim": 0, "n_units": 5},
            ]
        )
    with pytest.raises(ValueError, match="betas"):
        LazyAdamW([w], betas=(0.9, 1.0))
    with pytest.raises(ValueError, match="lr"):
        LazyAdamW([w], lr=-1.0)
    opt = LazyAdamW([{"params": [w], "unit_dim": 0, "n_units": 4}])
    w.grad = torch.ones_like(w)
    with pytest.raises(TypeError, match="bool"):
        opt.step(torch.ones(4))
    with pytest.raises(ValueError, match=r"\[M\]"):
        opt.step(torch.ones(3, dtype=torch.bool))
    opt.step(torch.tensor([True, False, True, False]))
    assert torch.equal(opt.state[w]["step"], torch.tensor([1, 0, 1, 0]))
    assert torch.equal(w[1], torch.zeros(3)) and not torch.equal(w[0], torch.zeros(3))
    plain = LazyAdamW([v])  # no unit group: a mask has no meaning
    assert plain.n_units is None
    v.grad = torch.ones_like(v)
    with pytest.raises(ValueError, match="no unit group"):
        plain.step(torch.ones(5, dtype=torch.bool))
    plain.step()
    assert int(plain.state[v]["step"]) == 1
    opt.zero_grad(set_to_none=True)
    assert w.grad is None


# --------------------------------------------------------------------------------------------------
# The active-unit mask
# --------------------------------------------------------------------------------------------------


def test_p2_6_active_mask_from_selection() -> None:
    """The union over valid slots only (padded slots hold unit 0 and never count), every unit for a
    dense-fallback row, the union over a window of selections, the range check, and the shape and
    dtype checks of the free function."""
    mask = active_units_from_indices(
        torch.tensor([[0, 3], [5, 0]]), torch.tensor([[False, True], [True, False]]), 8
    )
    assert mask.dtype is torch.bool and mask.device.type == "cpu"
    assert mask.tolist() == [False, False, False, True, False, True, False, False]
    window = active_units_from_indices(
        torch.tensor([[[1, 1], [2, 0]], [[1, 7], [0, 0]]]),  # [B=2, T=2, K=2]
        torch.tensor([[[True, True], [True, False]], [[False, True], [False, False]]]),
        8,
        dense_rows=torch.zeros(2, 2, dtype=torch.bool),
    )
    assert window.nonzero().flatten().tolist() == [1, 2, 7]
    dense = active_units_from_indices(
        torch.tensor([[3]]), torch.tensor([[True]]), 8, dense_rows=torch.tensor([True])
    )
    assert bool(dense.all())
    with pytest.raises(IndexError, match=r"\[0, 8\)"):
        active_units_from_indices(torch.tensor([[8]]), torch.tensor([[True]]), 8)
    ignored = active_units_from_indices(torch.tensor([[8]]), torch.tensor([[False]]), 8)
    assert not bool(ignored.any())  # an out-of-range padded slot is a padded slot
    with pytest.raises(ValueError, match="shape"):
        active_units_from_indices(torch.tensor([[1, 2]]), torch.tensor([[True]]), 8)
    with pytest.raises(TypeError, match="int64"):
        active_units_from_indices(torch.tensor([[1.0]]), torch.tensor([[True]]), 8)
    with pytest.raises(TypeError, match="bool"):
        active_units_from_indices(torch.tensor([[1]]), torch.tensor([[1]]), 8)
    with pytest.raises(TypeError, match="dense_rows"):
        active_units_from_indices(
            torch.tensor([[1]]), torch.tensor([[True]]), 8, dense_rows=torch.tensor([1.0])
        )
    # the optimiser's methods
    w = torch.zeros(8, 2, requires_grad=True)
    opt = LazyAdamW([{"params": [w], "unit_dim": 0, "n_units": 8}])
    first = _selection([[0, 3], [5, 0]], [[False, True], [True, False]], [False, False])
    assert opt.active_mask_from_selection(first).nonzero().flatten().tolist() == [3, 5]
    second = _selection([[6, 6, 6]], [[True, True, False]], [False])
    assert opt.active_mask_from_selections([first, second]).nonzero().flatten().tolist() == [
        3,
        5,
        6,
    ]
    fallback = _selection([[2, 0]], [[True, False]], [True])
    assert bool(opt.active_mask_from_selection(fallback).all())
    with pytest.raises(ValueError, match="at least one"):
        opt.active_mask_from_selections([])
    with pytest.raises(TypeError, match="Selection"):
        opt.active_mask_from_selection(torch.zeros(2))  # type: ignore[arg-type]
    plain = LazyAdamW([torch.zeros(3, requires_grad=True)])
    with pytest.raises(ValueError, match="no unit group"):
        plain.active_mask_from_selection(first)


# --------------------------------------------------------------------------------------------------
# The trainer
# --------------------------------------------------------------------------------------------------


def test_p2_6_trainer_builds_unit_groups(tmp_path: Path) -> None:
    """With ``routing=topk`` the optimiser defaults to ``LazyAdamW``: the unit bank's four tensors
    are the unit groups (``unit_dim=0, n_units=M``), every shared parameter (E, P, S, D, V, W_out,
    the gate, the scorer, the edge logits, the embeddings) sits in an ordinary group, the decay
    split is the P0.5 rule, an explicit ``adamw`` still builds AdamW, and models without exactly
    one trainable unit bank are refused."""
    cfg = _train_config()
    assert cfg.train.optimizer == "lazy_adamw"
    seed_everything(0)
    model = HSN.from_config(cfg)
    trainer = TBPTTTrainer(model, cfg, _cpu_ctx(), run_dir=tmp_path)
    opt = trainer.optimizer
    assert isinstance(opt, LazyAdamW) and opt.n_units == cfg.model.n_units == 2
    groups = opt.param_groups
    assert [("n_units" in g, g["weight_decay"]) for g in groups] == [
        (False, cfg.train.weight_decay),
        (False, 0.0),
        (True, cfg.train.weight_decay),
    ]
    unit_ids = {id(p) for p in model.cell.units.parameters()}
    lazy_ids = {id(p) for g in groups if "n_units" in g for p in g["params"]}
    assert lazy_ids == unit_ids and len(unit_ids) == 4
    shared = {name: p for name, p in model.named_parameters() if not name.startswith("cell.units.")}
    plain_ids = {id(p) for g in groups if "n_units" not in g for p in g["params"]}
    assert {id(p) for p in shared.values()} == plain_ids
    for prefix in (
        "encoder",
        "predictor",
        "shared",
        "decoder",
        "cell.V",
        "cell.W_out",
        "cell.gate",
    ):
        assert any(name.startswith(prefix) for name in shared), prefix
    assert "router.scorer.net.weight" in shared and "cell.graph.edge_logit" in shared
    for g in groups:
        decays = all(p.ndim >= 2 for p in g["params"])
        assert g["weight_decay"] == (cfg.train.weight_decay if decays else 0.0)
    assert find_unit_bank(model) is model.cell.units
    # the explicit choice
    plain_cfg = _train_config("train.optimizer=adamw")
    plain = TBPTTTrainer(HSN.from_config(plain_cfg), plain_cfg, _cpu_ctx(), run_dir=tmp_path / "p")
    assert type(plain.optimizer) is torch.optim.AdamW
    # refusals
    hyper = {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.01}
    with pytest.raises(ValueError, match="UnitBank"):
        build_lazy_adamw(nn.Linear(2, 2), **hyper)
    two = nn.ModuleList([model.cell.units, UnitBank(2, 8, 16, "gelu", init="trunc_normal_0.02")])
    with pytest.raises(ValueError, match="has 2"):
        build_lazy_adamw(two, **hyper)
    frozen = HSN.from_config(cfg)
    for p in frozen.cell.units.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable"):
        build_lazy_adamw(frozen, **hyper)


def _check_lazy_run(
    trainer: TBPTTTrainer, train_ds: ByteDataset, dev_ds: ByteDataset, steps: int
) -> Tensor:
    """Fit one step at a time with the bank snapshotted before each: units absent from
    ``active_units`` are bit-identical afterwards; a present unit with a non-zero gradient moved
    (one whose gradient is exactly zero — it executed only at the window's last position, say —
    takes the AdamW step of a zero gradient, which leaves a zero-initialised bias in place); returns
    the per-unit counts of active steps, which must equal the optimiser's step counters."""
    model = trainer.hsn
    bank = model.cell.units
    M = bank.n_units
    names = ("W1", "b1", "W2", "b2")
    counts = torch.zeros(M, dtype=torch.int64)
    moved = 0
    for k in range(1, steps + 1):
        before = {name: getattr(bank, name).detach().clone() for name in names}
        history = trainer.fit(train_ds, dev_ds, k)
        record = history.steps[-1]
        assert isinstance(record, TBPTTStepRecord) and record.step == k
        ids = record.active_units
        assert ids is not None and ids == sorted(set(ids)) and 1 <= len(ids) <= M, ids
        assert record.stats["active_units"] == float(len(ids))
        active = torch.zeros(M, dtype=torch.bool)
        active[ids] = True
        counts += active.to(torch.int64)
        for name in names:
            param = getattr(bank, name)
            after, grad = param.detach(), param.grad  # the step's (clipped) gradient is still here
            assert grad is not None
            for j in range(M):
                changed = not torch.equal(after[j], before[name][j])
                if not active[j]:
                    assert not changed, (k, name, j)
                    assert not bool(grad[j].any()), (k, name, j)  # inactive: a zero gradient row
                elif bool(grad[j].any()):
                    assert changed, (k, name, j)
                    moved += 1
    assert moved > 0
    opt = trainer.optimizer
    assert isinstance(opt, LazyAdamW)
    for name in names:
        assert torch.equal(opt.state[getattr(bank, name)]["step"], counts), name
    return counts


def test_p2_6_tbptt_inactive_units_do_not_move(
    train_docs: list[Example], dev_docs: list[Example], tmp_path: Path
) -> None:
    """A routed TBPTT run on the cpu: the step records name the units that executed, their weights
    move, the others' do not, the per-unit steps count the active steps, and some unit sat out at
    least one step; under the soft gate every unit is active."""
    cfg = _lazy_config(
        "train.steps=6", "train.eval_every=6", "train.ckpt_every=6", "train.stats_every=1"
    )
    assert cfg.train.optimizer == "lazy_adamw"
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)
    seed_everything(0)
    trainer = TBPTTTrainer(HSN.from_config(cfg), cfg, _cpu_ctx(), run_dir=tmp_path)
    counts = _check_lazy_run(trainer, train_ds, dev_ds, 6)
    assert int((counts < 6).sum()) >= 1, counts.tolist()
    soft_cfg = _lazy_config(
        "train.gate_mode=soft", "train.steps=2", "train.eval_every=2", "train.ckpt_every=2"
    )
    seed_everything(0)
    soft = TBPTTTrainer(HSN.from_config(soft_cfg), soft_cfg, _cpu_ctx(), run_dir=tmp_path / "soft")
    history = soft.fit(train_ds, dev_ds, 2)
    for record in history.steps:
        assert isinstance(record, TBPTTStepRecord)
        assert record.active_units == list(range(8)) and record.stats["active_units"] == 8.0
    # dense routing: no selection, every unit updates, the record holds None
    dense_cfg = _train_config(
        "model.routing=dense",
        "train.optimizer=lazy_adamw",
        "train.steps=1",
        "train.eval_every=1",
        "train.ckpt_every=1",
    )
    seed_everything(0)
    dense = TBPTTTrainer(HSN.from_config(dense_cfg), dense_cfg, _cpu_ctx(), run_dir=tmp_path / "d")
    record = dense.fit(train_ds, dev_ds, 1).steps[-1]
    assert isinstance(record, TBPTTStepRecord) and record.active_units is None
    assert "active_units" not in record.stats
    opt = dense.optimizer
    assert isinstance(opt, LazyAdamW)
    assert torch.equal(opt.state[dense.hsn.cell.units.W1]["step"], torch.full((2,), 1))


def test_p2_6_tbptt_resume_bit_exact(
    train_docs: list[Example], dev_docs: list[Example], tmp_path: Path
) -> None:
    """Six lazy sparse steps versus a resume from the step-3 checkpoint into a differently
    initialised trainer: records (with the active units), weights and the optimiser state
    (including the per-unit steps) agree bit for bit on the cpu."""
    cfg = _lazy_config(
        "train.steps=6",
        "train.eval_every=3",
        "train.ckpt_every=3",
        "train.stats_every=3",
        "train.deterministic=true",
        "train.warmup_steps=2",
        "train.schedule=cosine",
    )
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)
    seed_everything(0)
    trainer_a = TBPTTTrainer(HSN.from_config(cfg), cfg, _cpu_ctx(), run_dir=tmp_path / "a")
    history_a = trainer_a.fit(train_ds, dev_ds, 6)
    payload = torch.load(tmp_path / "a" / "step_00000003.pt", weights_only=True)
    steps = [entry["step"] for entry in payload["optimizer"]["state"].values()]
    assert sum(tuple(s.shape) == (8,) and s.dtype is torch.int64 for s in steps) == 4
    seed_everything(1)
    trainer_b = TBPTTTrainer(HSN.from_config(cfg), cfg, _cpu_ctx(), run_dir=tmp_path / "b")
    assert trainer_b.resume(tmp_path / "a" / "step_00000003.pt") == 3
    history_b = trainer_b.fit(train_ds, dev_ds, 6)
    for rec_a, rec_b in zip(history_a.steps[3:], history_b.steps, strict=True):
        assert isinstance(rec_a, TBPTTStepRecord) and isinstance(rec_b, TBPTTStepRecord)
        assert rec_a.active_units == rec_b.active_units and rec_a.active_units is not None
        assert _same(rec_a.loss, rec_b.loss) and _same(rec_a.total, rec_b.total)
        assert rec_a.parts == rec_b.parts and rec_a.stats == rec_b.stats
        assert rec_a.grad_norm == rec_b.grad_norm and rec_a.lr == rec_b.lr
    for key, value in trainer_a.model.state_dict().items():
        other = trainer_b.model.state_dict()[key]
        assert torch.equal(value, other) if isinstance(value, Tensor) else value == other, key
    state_a, state_b = trainer_a.optimizer.state_dict(), trainer_b.optimizer.state_dict()
    for index, entry in state_a["state"].items():
        for key, value in entry.items():
            other = state_b["state"][index][key]
            assert torch.equal(value, other) if isinstance(value, Tensor) else value == other


@pytest.mark.parametrize("profile", [pytest.param("rtx3060", marks=pytest.mark.cuda)])
def test_p2_6_amp_sparse_lazy_runs(
    profile: str, train_docs: list[Example], dev_docs: list[Example], tmp_path: Path
) -> None:
    """Four lazy sparse steps under the profile's bf16 autocast: the moments live on the device,
    the steps on the cpu, inactive units' fp32 master weights do not move."""
    hardware = load_hardware_profile(profile)
    if not torch.cuda.is_available():
        pytest.skip("no cuda device on this machine")
    ctx = resolve(hardware)
    cfg = _lazy_config(
        f"profile={profile}",
        "train.steps=4",
        "train.eval_every=4",
        "train.ckpt_every=4",
        "train.stats_every=1",
    )
    seed_everything(0)
    trainer = TBPTTTrainer(HSN.from_config(cfg), cfg, ctx, run_dir=tmp_path)
    _check_lazy_run(trainer, ByteDataset(train_docs), ByteDataset(dev_docs), 4)
    opt = trainer.optimizer
    assert isinstance(opt, LazyAdamW)
    w1 = trainer.hsn.cell.units.W1
    assert w1.dtype is torch.float32 and w1.device.type == "cuda"
    assert opt.state[w1]["exp_avg"].device.type == "cuda"
    assert opt.state[w1]["step"].device.type == "cpu"


# --------------------------------------------------------------------------------------------------
# The configuration default
# --------------------------------------------------------------------------------------------------


def test_p2_6_config_default_by_routing() -> None:
    """``train.optimizer`` unset: ``adamw`` for dense routing (``ref_a1`` and ``_example`` keep
    their resolved values), ``lazy_adamw`` for every selection mode; an explicit value is kept
    either way; a partial ``RunConfig`` dict resolves too."""
    dense = load_config(REF_A1)
    assert dense.train.optimizer == "adamw"
    assert canonical_json(dense) == canonical_json(load_config(REF_A1, ["train.optimizer=adamw"]))
    assert load_config(EXAMPLE).train.optimizer == "adamw"
    for routing in ("topk", "greedy_budget", "random", "fixed"):
        overrides = [f"model.routing={routing}"]
        if routing == "fixed":
            overrides.append("model.fixed_route=[1, 4]")
        assert load_config(REF_A1, overrides).train.optimizer == "lazy_adamw", routing
        assert load_config(REF_A1, [*overrides, "train.optimizer=adamw"]).train.optimizer == "adamw"
    assert load_config(REF_A1, ["train.optimizer=lazy_adamw"]).train.optimizer == "lazy_adamw"
    hardware = load_hardware_profile("cpu").model_dump()
    assert RunConfig.model_validate({"hardware": hardware}).train.optimizer == "adamw"
    sparse = RunConfig.model_validate({"hardware": hardware, "model": {"routing": "random"}})
    assert sparse.train.optimizer == "lazy_adamw"
    kept = RunConfig.model_validate(
        {"hardware": hardware, "model": {"routing": "random"}, "train": {"optimizer": "adamw"}}
    )
    assert kept.train.optimizer == "adamw"
    assert RunConfig.model_validate(
        {"hardware": hardware, "model": sparse.model}
    ).train.optimizer == ("lazy_adamw")
