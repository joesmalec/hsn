"""P2.10b: the trainer kernels (``kernels/cuda/hsn_trainer.cu``, :mod:`hsn.kernels.trainer`) change
no value. Each pass of the trainer after ``backward()`` that runs as one kernel launch (the
optimiser step, the float64 clip's scaling, the update caps, the weight average) equals its
PyTorch path bit for bit on the card, over many steps; what the kernels cannot take runs the
PyTorch path. CUDA-marked tests are skipped without a card or without the built artifacts unless
``HSN_REQUIRE_CUDA=1`` (then missing artifacts fail)."""

from __future__ import annotations

import copy
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from hsn.config import UpdateLimitConfig, load_config
from hsn.device import set_deterministic
from hsn.kernels import trainer as trainer_kernels
from hsn.kernels import window
from hsn.losses import assemble_terms, auxiliary_terms, terms_total, total_and_parts
from hsn.runner import build_model
from hsn.train.grad_norm import clip_grad_norm_
from hsn.train.lazy_adam import LazyAdamW, build_lazy_adamw
from hsn.train.route_guard import RouteGuard
from hsn.train.tbptt import TBPTTTrainer, _StepStatistics
from hsn.train.update_limit import UpdateLimiter
from hsn.train.weight_ema import WeightEMA
from p1_15_support import datasets, hsn_config, hsn_trainer
from torch import Tensor, nn

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


@pytest.fixture(autouse=True)
def _kernels_on() -> Iterator[None]:
    before = trainer_kernels.set_enabled(True)
    deterministic = torch.are_deterministic_algorithms_enabled()
    yield
    trainer_kernels.set_enabled(before)
    set_deterministic(deterministic)  # a trainer of the tiny configuration switches it on


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The names of the trainer kernels launched during the test."""
    seen: list[str] = []
    inner = trainer_kernels._launch

    def spy(name: str, *args: object, **kwargs: object) -> bool:
        launched = inner(name, *args, **kwargs)  # type: ignore[arg-type]
        if launched:
            seen.append(name)
        return launched

    monkeypatch.setattr(trainer_kernels, "_launch", spy)
    return seen


def _same(a: Tensor, b: Tensor) -> bool:
    """Equal bit patterns (``torch.equal`` would let 0.0 pass for -0.0)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {torch.float32: torch.int32, torch.float64: torch.int64}.get(a.dtype)
    if view is None:
        return bool(torch.equal(a, b))
    return bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))


SHAPES: list[tuple[int, ...]] = [
    (256, 64),
    (512, 192),
    (8, 202),
    (1, 128),
    (512,),
    (8,),
    (1,),
    (3, 5, 7),
    (129,),
    (70001,),
]


def _values(shape: tuple[int, ...], scale: float, g: torch.Generator) -> Tensor:
    """Normal values with zeros of both signs, subnormals and large magnitudes planted."""
    x = torch.randn(*shape, generator=g) * scale
    flat = x.view(-1)
    if flat.numel() >= 8:
        flat[:6] = torch.tensor([0.0, -0.0, 1e-42, -1e-40, 1e30, -3e-30])
    return x


def _clone_model(model: nn.Module) -> nn.Module:
    twin = copy.deepcopy(model)
    for p, q in zip(model.parameters(), twin.parameters(), strict=True):
        assert _same(p.detach(), q.detach())
    return twin


# --------------------------------------------------------------------------------------------------
# The optimiser step
# --------------------------------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
@pytest.mark.parametrize("betas", [(0.9, 0.95), (0.9, 0.999), (0.0, 0.5), (0.5, 0.0)])
def test_dense_step_kernel_equals_the_multi_tensor_step(
    device: torch.device, launches: list[str], weight_decay: float, betas: tuple[float, float]
) -> None:
    g = torch.Generator().manual_seed(0)
    reference = [nn.Parameter(_values(s, 0.1, g).to(device)) for s in SHAPES]
    fast = [nn.Parameter(p.detach().clone()) for p in reference]

    def groups(params: list[nn.Parameter]) -> list[dict[str, object]]:
        return [{"params": params[:4], "lr": 2.5e-3}, {"params": params[4:], "lr": 5e-4}]

    a = LazyAdamW(groups(reference), betas=betas, weight_decay=weight_decay)
    b = LazyAdamW(groups(fast), betas=betas, weight_decay=weight_decay)
    a.fused = False
    for step in range(40):
        scale = (1e-3, 1.0, 30.0, 1e-9)[step % 4]
        for index, (p, q) in enumerate(zip(reference, fast, strict=True)):
            if step == 3 and index == 2:  # a tensor without a gradient is skipped by both,
                p.grad = q.grad = None  # and is one step behind the others from then on
                continue
            grad = _values(tuple(p.shape), scale, g).to(device)
            p.grad, q.grad = grad.clone(), grad.clone()
        a.step()
        b.step()
        for p, q in zip(reference, fast, strict=True):
            assert _same(p.detach(), q.detach()), step
    for p, q in zip(reference, fast, strict=True):
        assert _same(a.state[p]["exp_avg"], b.state[q]["exp_avg"])
        assert _same(a.state[p]["exp_avg_sq"], b.state[q]["exp_avg_sq"])
        assert int(a.state[p]["step"]) == int(b.state[q]["step"])
    assert set(launches) == {"hsn_tr_adamw"}
    assert len(launches) >= 2 * 40  # one launch per group and step count


@pytest.mark.cuda
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
@pytest.mark.parametrize("betas", [(0.9, 0.95), (0.9, 0.999)])
def test_unit_step_kernel_equals_the_masked_step(
    device: torch.device, launches: list[str], weight_decay: float, betas: tuple[float, float]
) -> None:
    M = 8
    shapes = [(M, 64, 96), (M, 96), (M, 96, 16), (M, 16), (M, 3), (M,)]
    g = torch.Generator().manual_seed(1)
    reference = [nn.Parameter(_values(s, 0.1, g).to(device)) for s in shapes]
    fast = [nn.Parameter(p.detach().clone()) for p in reference]

    def groups(params: list[nn.Parameter]) -> list[dict[str, object]]:
        return [
            {"params": params[:3], "unit_dim": 0, "n_units": M},
            {"params": params[3:], "unit_dim": 0, "n_units": M, "lr": 1e-3, "weight_decay": 0.0},
        ]

    a = LazyAdamW(groups(reference), lr=2.5e-3, betas=betas, weight_decay=weight_decay)
    b = LazyAdamW(groups(fast), lr=2.5e-3, betas=betas, weight_decay=weight_decay)
    a.fused = False
    for step in range(40):
        scale = (1e-3, 1.0, 30.0, 1e-9)[step % 4]
        mask: Tensor | None = torch.rand(M, generator=g) < 0.6
        if step == 5:
            mask = torch.ones(M, dtype=torch.bool)
        elif step == 6:
            mask = torch.zeros(M, dtype=torch.bool)
        elif step % 7 == 0:
            mask = None  # every unit
        for index, (p, q) in enumerate(zip(reference, fast, strict=True)):
            if step == 9 and index == 1:  # its step counts part from the group's other tensors
                p.grad = q.grad = None
                continue
            grad = _values(tuple(p.shape), scale, g).to(device)
            p.grad, q.grad = grad.clone(), grad.clone()
        a.step(active_units=mask)
        b.step(active_units=mask)
        for p, q in zip(reference, fast, strict=True):
            assert _same(p.detach(), q.detach()), step
            assert _same(a.state[p]["exp_avg"], b.state[q]["exp_avg"]), step
            assert _same(a.state[p]["exp_avg_sq"], b.state[q]["exp_avg_sq"]), step
            assert torch.equal(a.state[p]["step"], b.state[q]["step"]), step
    assert set(launches) == {"hsn_tr_adamw_units"}
    assert len(launches) > 2 * 40  # per group, and one more once a tensor's counts differ


@pytest.mark.cuda
def test_optimiser_keeps_its_pytorch_path_for_what_the_kernels_cannot_take(
    device: torch.device, launches: list[str]
) -> None:
    g = torch.Generator().manual_seed(2)
    for kind in ("float64", "strided-gradient", "switched-off"):
        dtype = torch.float64 if kind == "float64" else torch.float32
        start = torch.randn(6, 10, generator=g).to(device=device, dtype=dtype)
        p, q = nn.Parameter(start.clone()), nn.Parameter(start.clone())
        a, b = LazyAdamW([p], lr=1e-2), LazyAdamW([q], lr=1e-2)
        a.fused = False
        if kind == "switched-off":
            trainer_kernels.set_enabled(False)
        for _ in range(3):
            grad = torch.randn(10, 6, generator=g).to(device=device, dtype=dtype)
            # a transposed view: the same values, not contiguous
            p.grad = grad.t().contiguous()
            q.grad = grad.t() if kind == "strided-gradient" else grad.t().contiguous()
            a.step()
            b.step()
            assert _same(p.detach(), q.detach()), kind
    assert launches == []


# --------------------------------------------------------------------------------------------------
# The weight average, the clip, the update caps
# --------------------------------------------------------------------------------------------------


def _small_model(device: torch.device, seed: int = 3) -> nn.Module:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(37, 53), nn.GELU(), nn.Linear(53, 11), nn.LayerNorm(11))
    return model.to(device)


@pytest.mark.cuda
def test_weight_average_kernel_equals_the_multi_tensor_lerp(
    device: torch.device, launches: list[str]
) -> None:
    model = _small_model(device)
    fast = WeightEMA(model, 0.99)
    trainer_kernels.set_enabled(False)
    reference = WeightEMA(model, 0.99)
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for _ in range(120):  # the warm-up's weights fall from 0.9 through 0.5 to 0.01
            for p in model.parameters():
                p.add_(1e-2 * torch.randn(p.shape, generator=g).to(device))
            trainer_kernels.set_enabled(False)
            reference.update()
            trainer_kernels.set_enabled(True)
            fast.update()
            for x, y in zip(reference._shadow, fast._shadow, strict=True):
                assert _same(x, y)
    assert launches == ["hsn_tr_lerp"] * 120


@pytest.mark.cuda
def test_weight_average_kernel_takes_more_tensors_than_one_launch_holds(
    device: torch.device, launches: list[str]
) -> None:
    g = torch.Generator().manual_seed(5)
    sizes = [1 + (7 * i) % 23 for i in range(103)]
    selfs = [torch.randn(n, generator=g).to(device) for n in sizes]
    ends = [torch.randn(n, generator=g).to(device) for n in sizes]
    want = [x.clone() for x in selfs]
    torch._foreach_lerp_(want, ends, 0.3)
    assert trainer_kernels.lerp(selfs, ends, 0.3)
    for x, y in zip(want, selfs, strict=True):
        assert _same(x, y)
    assert launches == ["hsn_tr_lerp"]


@pytest.mark.cuda
@pytest.mark.parametrize("scale", [1e-3, 1.0, 40.0, 1e6])
def test_float64_clip_kernel_equals_the_float64_scaling(
    device: torch.device, launches: list[str], scale: float
) -> None:
    model = _small_model(device)
    x = torch.randn(19, 37, generator=torch.Generator().manual_seed(1)).to(device)
    (model(x).square().mean() * scale).backward()
    twin = _clone_model(model)
    for p, q in zip(model.parameters(), twin.parameters(), strict=True):
        q.grad = p.grad.clone()
    trainer_kernels.set_enabled(False)
    want = clip_grad_norm_(list(twin.parameters()), 1.0, precision="float64")
    trainer_kernels.set_enabled(True)
    got = clip_grad_norm_(list(model.parameters()), 1.0, precision="float64")
    assert _same(got, want)
    for p, q in zip(model.parameters(), twin.parameters(), strict=True):
        assert _same(p.grad, q.grad)
    assert launches == (["hsn_tr_scale64"] if float(want) > 1.0 else [])


def _per_tensor_apply(limiter: UpdateLimiter) -> None:
    """UpdateLimiter.apply one tensor at a time, as before P2.10 (PR #46)."""
    for index, (group, entries) in enumerate(limiter.groups.items()):
        old_norm = torch.stack([old.square().sum() for _, _, old in entries]).sum().sqrt()
        denom = old_norm.clamp_min(limiter.config.norm_floor)
        deltas = [p - old for _, p, old in entries]
        norm = torch.stack([delta.square().sum() for delta in deltas]).sum().sqrt()
        reference = denom
        if group in limiter.initial_norms:
            ceiling = limiter.initial_norms[group] * limiter.config.norm_growth_limits[group]
            reference = torch.minimum(denom, ceiling.clamp_min(limiter.config.norm_floor))
        radius = reference * limiter.config.groups[group].max_ratio
        scale = radius / norm.clamp_min(radius)
        clipped = norm > radius
        for (_, p, old), delta in zip(entries, deltas, strict=True):
            p.copy_(torch.where(clipped, old + delta * scale, p))
        proposed = norm / denom
        actual = torch.stack([(p - old).square().sum() for _, p, old in entries]).sum().sqrt()
        applied = actual / denom
        limiter.counts[index].add_(clipped)
        limiter.max_proposed[index] = torch.maximum(limiter.max_proposed[index], proposed)
        limiter.max_applied[index] = torch.maximum(limiter.max_applied[index], applied)
        limiter.last[index] = torch.stack((proposed, applied, scale))
    limiter.applied_steps += 1


@pytest.mark.cuda
@pytest.mark.parametrize("reference_kind", ["multi-tensor", "per-tensor"])
def test_update_limit_kernels_equal_the_pytorch_passes(
    device: torch.device, launches: list[str], reference_kind: str
) -> None:
    config = UpdateLimitConfig.model_validate(
        {
            "groups": {
                "first": {"selectors": ["0.*"], "max_ratio": 0.003},
                "second": {"selectors": ["2.weight"], "max_ratio": 0.5},
                "third": {"selectors": ["2.bias", "3.*"], "max_ratio": 0.0125},
            },
            "norm_growth_limits": {"first": 1.0, "third": 1.37},
        }
    )
    reference, fast = _small_model(device), _small_model(device)
    a, b = UpdateLimiter(reference, config), UpdateLimiter(fast, config)
    a.fused = False
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for step in range(24):
            a.capture()
            b.capture()
            for p, q in zip(reference.parameters(), fast.parameters(), strict=True):
                # small and large proposed steps: the untouched and the projected branch, and
                # weights that grow past the anchored groups' ceilings
                size = (1e-5, 0.02, 0.3)[step % 3]
                move = size * torch.randn(p.shape, generator=g).to(device)
                p.add_(move)
                q.add_(move)
            if reference_kind == "per-tensor":
                _per_tensor_apply(a)
            else:
                a.apply()
            b.apply()
            for p, q in zip(reference.parameters(), fast.parameters(), strict=True):
                assert _same(p.detach(), q.detach()), step
            for (_, _, old), (_, _, new) in zip(
                [e for entries in a.groups.values() for e in entries],
                [e for entries in b.groups.values() for e in entries],
                strict=True,
            ):
                assert _same(old, new), step
            for name in ("counts", "max_proposed", "max_applied", "last"):
                assert _same(getattr(a, name), getattr(b, name)), (name, step)
    assert a.applied_steps == b.applied_steps == 24
    assert a.state_dict() == b.state_dict()
    counts = a.counts.tolist()
    assert max(counts) > 0 and min(counts) < 24  # both branches were taken
    assert launches == ["hsn_tr_copy", "hsn_tr_limit_squares", "hsn_tr_limit_project"] * 24


@pytest.mark.cuda
def test_the_step_after_backward_equals_its_pytorch_passes_on_the_recipe_model(
    device: torch.device, launches: list[str]
) -> None:
    """The recipe's model, optimiser groups, caps and weight average through sixty steps of
    clip, capture, optimiser step, caps and average on random gradients and unit masks: every
    weight, moment, statistic and average equal with the kernels on and off."""
    cfg = load_config(RECIPE)
    torch.manual_seed(0)
    reference = build_model(cfg).to(device)
    fast = _clone_model(reference)
    train = cfg.train

    def parts(model: nn.Module) -> tuple[LazyAdamW, UpdateLimiter, WeightEMA]:
        optimizer = build_lazy_adamw(
            model, lr=train.lr, betas=tuple(train.betas), weight_decay=train.weight_decay
        )
        return optimizer, UpdateLimiter(model, train.update_limit), WeightEMA(model, 0.99)

    trainer_kernels.set_enabled(False)
    opt_a, limit_a, ema_a = parts(reference)
    trainer_kernels.set_enabled(True)
    opt_b, limit_b, ema_b = parts(fast)
    g = torch.Generator().manual_seed(6)
    M = cfg.model.n_units
    for step in range(60):
        scale = (3e-3, 0.5, 20.0)[step % 3]
        mask = torch.rand(M, generator=g) < 0.7
        for p, q in zip(reference.parameters(), fast.parameters(), strict=True):
            if p.ndim == 2 and p.shape[0] == 1:  # the halting head: no gradient, as in training
                continue
            grad = (scale * torch.randn(p.shape, generator=g)).to(device)
            p.grad, q.grad = grad.clone(), grad.clone()
        norms: list[Tensor] = []
        for on, model, optimizer, limiter, ema in (
            (False, reference, opt_a, limit_a, ema_a),
            (True, fast, opt_b, limit_b, ema_b),
        ):
            trainer_kernels.set_enabled(on)
            params = [p for p in model.parameters() if p.requires_grad]
            norms.append(clip_grad_norm_(params, train.grad_clip, precision="float64"))
            limiter.capture()
            optimizer.step(active_units=mask)
            limiter.apply()
            ema.update()
        assert _same(norms[0], norms[1])
        for (name, p), q in zip(reference.named_parameters(), fast.parameters(), strict=True):
            assert _same(p.detach(), q.detach()), (name, step)
            if p.grad is not None:
                assert _same(p.grad, q.grad), (name, step)
        for x, y in zip(ema_a._shadow, ema_b._shadow, strict=True):
            assert _same(x, y), step
        for name in ("counts", "max_proposed", "max_applied", "last"):
            assert _same(getattr(limit_a, name), getattr(limit_b, name)), (name, step)
    for p, q in zip(reference.parameters(), fast.parameters(), strict=True):
        if p in opt_a.state:
            for key in ("exp_avg", "exp_avg_sq"):
                assert _same(opt_a.state[p][key], opt_b.state[q][key])
            assert torch.equal(opt_a.state[p]["step"], opt_b.state[q]["step"])
    assert limit_a.state_dict() == limit_b.state_dict()
    assert min(limit_a.counts.tolist()) > 0  # the caps were binding
    names = set(launches)
    assert names == {
        "hsn_tr_scale64",
        "hsn_tr_copy",
        "hsn_tr_adamw",
        "hsn_tr_adamw_units",
        "hsn_tr_limit_squares",
        "hsn_tr_limit_project",
        "hsn_tr_lerp",
    }


# --------------------------------------------------------------------------------------------------
# Off the card: the PyTorch paths, and the route guard's flag
# --------------------------------------------------------------------------------------------------


def test_kernels_decline_cpu_tensors() -> None:
    x, y = torch.ones(3), torch.zeros(3)
    assert not trainer_kernels.ready(torch.device("cpu"))
    assert not trainer_kernels.lerp([x], [y], 0.5)
    assert not trainer_kernels.copy([x], [y])
    assert not trainer_kernels.scale64([x], torch.tensor(0.5, dtype=torch.float64))
    assert torch.equal(x, torch.ones(3))


def test_route_guard_reads_its_scale_only_when_it_changed() -> None:
    cfg = load_config(RECIPE)
    router = nn.Module()
    router.scorer = nn.Module()
    router.scorer.net = nn.Linear(5, 4)
    guard = RouteGuard(router, cfg.train.route_guard, n_units=4, vocab=8)
    assert guard._scaled() is False
    guard.capture()
    assert guard._old is None  # nothing is scaled: no copy of the rows
    guard.scale[2] = 0.5
    assert guard._scaled() is True
    guard.capture()
    assert guard._old is not None
    with torch.no_grad():
        before = [p.detach().clone() for p in guard.params]
        for p in guard.params:
            p.add_(1.0)
        moved = [p.detach().clone() for p in guard.params]
        guard.apply()
    for p, old, new in zip(guard.params, before, moved, strict=True):
        assert torch.equal(p[2], old[2] + 0.5 * (new[2] - old[2]))  # the scaled unit's rows
        assert torch.equal(p[0], old[0] + (new[0] - old[0]))
    guard.scale.fill_(1.0)
    assert guard._scaled() is False
    guard.load_state_dict({**guard.state_dict(), "scale": torch.tensor([1.0, 0.25, 1.0, 1.0])})
    assert guard._scaled() is True


# --------------------------------------------------------------------------------------------------
# The window's statistics: issued after the backward pass, read once
# --------------------------------------------------------------------------------------------------


def _reads_before_p2_10b(
    trainer: TBPTTTrainer,
    info: dict[str, Tensor],
    fwd: dict[str, Tensor],
    logits: Tensor,
    terms: dict[str, Tensor],
    count: int,
    tokens: int,
    report: dict[str, int],
) -> dict[str, object]:
    """The host reads of one window as TBPTTTrainer.fit made them before P2.10b, one by one."""
    active = trainer._window_active_units(info)
    total, parts = total_and_parts(terms)
    loss_mask = fwd["loss_mask"]
    correct = 0
    if count > 0:
        flat_targets = fwd["targets"][loss_mask]
        correct = int((logits[loss_mask].float().argmax(dim=-1) == flat_targets).sum())
    return {
        "active": None if active is None else active.tolist(),
        "parts": parts,
        "total": float(total.detach()),
        "parameter_sum": trainer._window_parameter_sum(info, fwd, tokens, report),
        "correct": correct,
        "loss_sum": parts["task"] * count if count > 0 else 0.0,
        "state_abs_max": float(info["state_abs_max"].max()),
        "eps": float(info["eps_abs_mean"].float().mean()),
        "entropy": float(info["alpha_entropy"].float().mean()),
        "cost": float(info["cost"].float().mean()),
        "rates": {
            key: float(info[key].float().mean())
            for key in ("explored", "refreshed", "dense_fallback")
            if key in info
        },
    }


@pytest.mark.parametrize(
    "overrides",
    [
        ("model.routing=topk", "train.gate_mode=ste"),
        ("model.routing=topk", "train.gate_mode=ste", "train.soft_until_step=3"),
        ("model.routing=topk", "train.gate_mode=ste", "train.p_dense=0.5"),
        (),  # the dense reference: no selection in info
    ],
)
def test_window_statistics_equal_the_reads_they_replace(
    tmp_path: Path, overrides: tuple[str, ...]
) -> None:
    cfg = hsn_config(4, *overrides)
    trainer = hsn_trainer(cfg, tmp_path)
    train_ds, _ = datasets()
    loader = trainer._loader(train_ds, shuffle=True, epochs=None)
    model = trainer.hsn
    report = model.param_report()
    state = model.init_state(cfg.train.batch, trainer.ctx.device)
    reference_cost = model.router.costs.dense_cost()
    seen_loss = seen_ties = False
    for step in range(4):
        window = next(loader)
        fwd = trainer._to_device(window)
        model.set_train_step(step)
        trainer.step = step
        logits, state, info = trainer._forward(fwd, state.detach())
        if step == 2:  # ties: the first maximum is the prediction either way
            logits = logits.detach().clone()
            logits[..., 1:] = 0.0
            logits[..., 0] = torch.where(fwd["targets"] % 2 == 0, 0.0, -1.0)
            seen_ties = True
        extras = auxiliary_terms(info, cfg, model.cell.graph, reference_cost=reference_cost)
        terms = assemble_terms({"logits": logits, **info}, fwd, cfg, extras)
        count = int(window["loss_mask"].sum())
        tokens = int((window["doc"] >= 0).sum())
        seen_loss = seen_loss or count > 0
        want = _reads_before_p2_10b(trainer, info, fwd, logits, terms, count, tokens, report)
        total = terms_total(terms)
        total.backward()  # the statistics are read after the backward pass
        statistics = trainer._window_statistics(
            info, fwd, logits, terms, total, count, None, tokens
        )
        totals = _StepStatistics()
        trainer._read_window_statistics(statistics, totals, report)
        got = {
            "active": None if totals.active is None else totals.active.tolist(),
            "parts": totals.part_sums,
            "total": totals.total_sum,
            "parameter_sum": totals.parameter_sum,
            "correct": totals.correct,
            "loss_sum": totals.loss_sum,
            "state_abs_max": totals.state_abs_max,
            "eps": totals.eps_sum,
            "entropy": totals.entropy_sum,
            "cost": totals.cost_sum,
            "rates": totals.rate_sums,
        }
        assert got == want, step
        assert totals.n_loss == count
        model.zero_grad(set_to_none=True)
    assert seen_loss and seen_ties


def test_window_statistics_refuse_a_unit_outside_the_bank(tmp_path: Path) -> None:
    trainer = hsn_trainer(hsn_config(1, "model.routing=topk", "train.gate_mode=ste"), tmp_path)
    info = {
        "unit_idx": torch.tensor([[[0, 5]]]),
        "valid": torch.ones(1, 1, 2, dtype=torch.bool),
        "dense_fallback": torch.zeros(1, 1),
        "state_abs_max": torch.ones(1),
        "eps_abs_mean": torch.ones(1),
        "alpha_entropy": torch.ones(1),
        "cost": torch.ones(1),
    }
    fwd = {
        "doc": torch.zeros(1, 1, dtype=torch.int64),
        "targets": torch.zeros(1, 1, dtype=torch.int64),
        "loss_mask": torch.zeros(1, 1, dtype=torch.bool),
    }
    task = torch.zeros(())
    statistics = trainer._window_statistics(
        info, fwd, torch.zeros(1, 1, 4), {"task": task}, task, 0, None, 1
    )
    with pytest.raises(IndexError, match="outside"):
        trainer._read_window_statistics(statistics, _StepStatistics(), trainer.hsn.param_report())
