"""P2.10: the trainer's per-step shortcuts change no value: the multi-tensor optimiser step and
the float64 clip equal their per-tensor references bit for bit."""

from __future__ import annotations

import copy

import pytest
import torch
from hsn.config import UpdateLimitConfig
from hsn.train.grad_norm import clip_grad_norm_, norm_float64
from hsn.train.lazy_adam import LazyAdamW
from hsn.train.update_limit import UpdateLimiter
from torch import nn


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _model(device: str) -> nn.Module:
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(37, 53), nn.GELU(), nn.Linear(53, 11), nn.LayerNorm(11))
    return model.to(device)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
def test_multi_tensor_step_equals_the_per_tensor_step(device: str, weight_decay: float) -> None:
    reference, fast = _model(device), _model(device)
    groups = lambda m: [  # noqa: E731 - two groups with their own rates
        {"params": list(m[0].parameters()), "lr": 2e-3},
        {"params": [*m[2].parameters(), *m[3].parameters()], "lr": 5e-4},
    ]
    a = LazyAdamW(groups(reference), betas=(0.9, 0.95), weight_decay=weight_decay)
    b = LazyAdamW(groups(fast), betas=(0.9, 0.95), weight_decay=weight_decay)
    a.foreach = False
    assert b.foreach
    g = torch.Generator().manual_seed(0)
    for step in range(7):
        x = torch.randn(19, 37, generator=g).to(device)
        for model, opt in ((reference, a), (fast, b)):
            opt.zero_grad(set_to_none=True)
            model(x).square().mean().backward()
            if step == 3:  # a tensor without a gradient is skipped by both
                next(model[2].parameters()).grad = None
            opt.step()
        for p, q in zip(reference.parameters(), fast.parameters(), strict=True):
            assert torch.equal(p, q)
    for p, q in zip(reference.parameters(), fast.parameters(), strict=True):
        assert torch.equal(a.state[p]["exp_avg"], b.state[q]["exp_avg"])
        assert torch.equal(a.state[p]["exp_avg_sq"], b.state[q]["exp_avg_sq"])
        assert int(a.state[p]["step"]) == int(b.state[q]["step"])


def _reference_clip(parameters: list[nn.Parameter], max_norm: float) -> torch.Tensor:
    """The float64 clip as it was before P2.10 (hsn.train.grad_norm, PR #46)."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    total = norm_float64(gradients)
    if not bool(torch.isfinite(total)):
        return total
    coefficient = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for gradient in gradients:
        wide = gradient.detach().to(dtype=torch.float64)
        gradient.copy_((wide * coefficient).to(dtype=gradient.dtype))
    return total


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("scale", [1e-3, 1.0, 40.0, float("inf")])
def test_float64_clip_equals_its_reference(device: str, scale: float) -> None:
    model = _model(device)
    x = torch.randn(19, 37, generator=torch.Generator().manual_seed(1)).to(device)
    (model(x).square().mean() * (1.0 if scale == float("inf") else scale)).backward()
    if scale == float("inf"):
        next(model.parameters()).grad[0, 0] = float("inf")
    twin = copy.deepcopy(model)
    for p, q in zip(model.parameters(), twin.parameters(), strict=True):
        q.grad = p.grad.clone()
    want = _reference_clip(list(twin.parameters()), 1.0)
    got = clip_grad_norm_(list(model.parameters()), 1.0, precision="float64")
    assert torch.equal(got.cpu(), want.cpu()) or (
        bool(torch.isnan(got)) and bool(torch.isnan(want))
    )
    for p, q in zip(model.parameters(), twin.parameters(), strict=True):
        assert torch.equal(
            p.grad,
            q.grad,
        ) or bool(torch.isnan(p.grad).any())


def _reference_apply(limiter: UpdateLimiter) -> None:
    """UpdateLimiter.apply as it was before P2.10 (PR #46): one tensor at a time."""
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


@pytest.mark.parametrize("device", _devices())
def test_update_limit_equals_its_per_tensor_reference(device: str) -> None:
    config = UpdateLimitConfig.model_validate(
        {
            "groups": {
                "first": {"selectors": ["0.*"], "max_ratio": 0.003},
                "rest": {"selectors": ["2.*", "3.*"], "max_ratio": 0.5},
            },
            "norm_growth_limits": {"first": 1.0},
        }
    )
    reference, fast = _model(device), _model(device)
    a, b = UpdateLimiter(reference, config), UpdateLimiter(fast, config)
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for step in range(5):
            a.capture()
            b.capture()
            for p, q in zip(reference.parameters(), fast.parameters(), strict=True):
                # small and large proposed steps: both the projected and the untouched branch
                move = (0.02 if step % 2 else 1e-5) * torch.randn(p.shape, generator=g).to(device)
                p.add_(move)
                q.add_(move)
            _reference_apply(a)
            b.apply()
            for p, q in zip(reference.parameters(), fast.parameters(), strict=True):
                assert torch.equal(p, q)
            for name in ("counts", "max_proposed", "max_applied", "last"):
                assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert int(a.counts.sum()) > 0 and int(a.counts.min()) < 5  # both branches were taken
