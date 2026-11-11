"""Finite norm overflow, clipping direction, nonfinite protection and trainer continuity."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from cards import CUDA_PROFILE
from hsn.config import GradientGrowthConfig, canonical_json, config_hash
from hsn.losses import grad_norm_report
from hsn.train.grad_norm import clip_grad_norm_, norm_float64
from hsn.train.gradient_growth import GradientGrowth
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import StepProbe, datasets, gru_config, gru_trainer, hsn_config, hsn_trainer
from test_a5_stability import A5

pytestmark = pytest.mark.usefixtures("restore_determinism")


def parameter(values: list[float], dtype: torch.dtype = torch.float32, device: str = "cpu") -> Any:
    p = torch.nn.Parameter(torch.zeros(len(values), dtype=dtype, device=device))
    p.grad = torch.tensor(values, dtype=dtype, device=device)
    return p


def test_native_clipping_is_bitwise_pytorch() -> None:
    for values in ([3.0, 4.0], [3e20, 4e20], [0.0, 0.0]):
        left, right = parameter(values), parameter(values)
        expected = torch.nn.utils.clip_grad_norm_([left], 0.7)
        actual = clip_grad_norm_([right], 0.7)
        assert torch.equal(actual, expected)
        assert torch.equal(left.grad, right.grad)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (torch.float32, [3e20, -4e20]),
        (torch.bfloat16, [3e20, -4e20]),
        (torch.float16, [60000.0, -60000.0]),
        (torch.float64, [3e100, -4e100]),
    ],
)
def test_wide_norm_and_clipping_preserve_direction(dtype: torch.dtype, values: list[float]) -> None:
    p = parameter(values, dtype)
    original = p.grad.clone()
    address = p.grad.data_ptr()
    expected = math.hypot(*original.double().tolist())
    raw = clip_grad_norm_([p], 0.7, precision="float64")
    assert raw.dtype == torch.float64 and float(raw) == pytest.approx(expected, rel=1e-14)
    assert p.grad.data_ptr() == address
    torch.testing.assert_close(p.grad, (original.double() * (0.7 / (expected + 1e-6))).to(dtype))
    assert math.isfinite(float(norm_float64([p.grad])))
    assert float(norm_float64([p.grad])) == pytest.approx(0.7, rel=0.01)


def test_coefficient_is_not_narrowed_before_multiplication() -> None:
    p = parameter([3e30, -4e30])
    raw = clip_grad_norm_([p], 1e-20, precision="float64")
    assert float(raw) == pytest.approx(5e30)
    # The coefficient is ~2e-51, below even a float32 subnormal. The products are representable.
    torch.testing.assert_close(p.grad, torch.tensor([6e-21, -8e-21]), rtol=1e-6, atol=0)


def test_missing_zero_mixed_dtype_and_disabled_clipping() -> None:
    missing = torch.nn.Parameter(torch.ones(1))
    assert float(clip_grad_norm_([missing], 1.0, precision="float64")) == 0
    assert missing.grad is None
    zero = parameter([0.0])
    assert float(clip_grad_norm_([zero], 1.0, precision="float64")) == 0
    assert torch.equal(zero.grad, torch.zeros(1))
    left, right = parameter([3e20]), parameter([4e20], torch.float64)
    before = [left.grad.clone(), right.grad.clone()]
    raw = clip_grad_norm_([left, right], math.inf, precision="float64")
    assert float(raw) == pytest.approx(5e20)
    assert all(torch.equal(p.grad, g) for p, g in zip((left, right), before, strict=True))


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_real_nonfinite_gradients_remain_visible_to_guard(bad: float) -> None:
    p = parameter([1.0, bad])
    raw = clip_grad_norm_([p], 1.0, precision="float64")
    assert not math.isfinite(float(raw))
    assert p.grad[0] == 1.0
    assert math.isnan(float(p.grad[1])) if math.isnan(bad) else float(p.grad[1]) == bad


def test_config_default_hash_and_rejection() -> None:
    original = hsn_config(4)
    native = hsn_config(4, "train.grad_norm_precision=native")
    wide = hsn_config(4, "train.grad_norm_precision=float64")
    assert config_hash(original) == config_hash(native)
    assert "grad_norm_precision" not in json.loads(canonical_json(native))["train"]
    assert config_hash(wide) != config_hash(native)
    with pytest.raises(ValueError, match="grad_norm_precision"):
        hsn_config(4, "train.grad_norm_precision=float32")


def test_group_controller_and_term_reports_use_wide_norms() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3e20, -4e20]])
    cfg = GradientGrowthConfig.model_validate(dict(enabled=True, groups={"all": ["*"]}))
    controller = GradientGrowth(model, cfg)
    assert math.isinf(float(controller.raw_norms()[0]))
    assert float(controller.raw_norms(precision="float64")[0]) == pytest.approx(5e20)
    groups = TBPTTTrainer._group_grad_norms({"all": list(model.parameters())}, precision="float64")
    assert groups["grad_norm/all"] == pytest.approx(5e20)
    term = (model.weight * torch.tensor([[3e20, -4e20]])).sum()
    report = grad_norm_report(model, {"task": term}, precision="float64")
    assert report["term_grad_norm/task"] == pytest.approx(5e20)
    assert torch.equal(model.weight.grad, torch.tensor([[3e20, -4e20]]))


def plant_large(t: Any) -> None:
    if t.step in (1, 4):
        for p in t.model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        gradient = next(p.grad for p in t.model.parameters() if p.grad is not None)
        gradient.view(-1)[:2] = gradient.new_tensor([3e20, -4e20])


@pytest.mark.parametrize("family", ["hsn", "gru"])
def test_trainers_apply_finite_overflow_and_still_skip_true_inf(
    tmp_path: Path, family: str
) -> None:
    overrides = (
        "train.grad_norm_precision=float64",
        "train.gradient_growth.enabled=true",
        'train.gradient_growth.groups={all: ["*"]}',
    )
    trainer = (
        hsn_trainer(hsn_config(4, *A5, *overrides), tmp_path)
        if family == "hsn"
        else gru_trainer(gru_config(4, *overrides), tmp_path)
    )

    def hook(t: Any) -> None:
        plant_large(t)
        if t.step == 2:
            next(p.grad for p in t.model.parameters() if p.grad is not None).fill_(math.inf)

    trainer.grad_hook = hook
    with StepProbe(trainer) as probe:
        history = trainer.fit(*datasets(), 4)
    assert history.steps[1].grad_norm == pytest.approx(5e20)
    assert trainer.skipped_nonfinite == 1
    assert probe.optimizer_steps[2] == 2
    assert probe.optimizer_steps[3] == 2 and probe.unchanged(3)
    assert probe.optimizer_steps[4] == 3
    assert trainer.gradient_growth is not None
    assert trainer.gradient_growth.samples == trainer.gradient_growth.applied_steps == 3


@pytest.mark.parametrize("optimizer", ["adamw", "lazy_adamw"])
def test_checkpoint_resume_with_wide_norms_keeps_optimizer_and_stream(
    tmp_path: Path, optimizer: str
) -> None:
    cfg = hsn_config(
        6,
        *A5,
        "train.grad_norm_precision=float64",
        f"train.optimizer={optimizer}",
        "train.ckpt_every=3",
    )
    full = hsn_trainer(cfg, tmp_path / "full")
    full.grad_hook = plant_large
    expected = full.fit(*datasets(), 6)
    resumed = hsn_trainer(cfg, tmp_path / "resumed")
    resumed.resume(tmp_path / "full" / "step_00000003.pt")
    resumed.grad_hook = plant_large
    actual = resumed.fit(*datasets(), 6)
    assert [s.loss for s in actual.steps] == [s.loss for s in expected.steps[3:]]
    assert [s.grad_norm for s in actual.steps] == [s.grad_norm for s in expected.steps[3:]]
    for name, value in full.model.state_dict().items():
        other = resumed.model.state_dict()[name]
        assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    for p, q in zip(full.model.parameters(), resumed.model.parameters(), strict=True):
        for key, value in full.optimizer.state[p].items():
            other = resumed.optimizer.state[q][key]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other


@pytest.mark.cuda
def test_compiled_wide_clipping_preserves_storage_and_finite_updates(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cfg = hsn_config(
        4,
        *A5,
        "train.grad_norm_precision=float64",
        f"profile={CUDA_PROFILE}",
        "train.precision=bf16",
        "train.deterministic=false",
        "hardware.compile_mode=reduce-overhead",
    )
    trainer = hsn_trainer(cfg, tmp_path)
    trainer.grad_hook = plant_large
    addresses = [p.data_ptr() for p in trainer.model.parameters()]
    history = trainer.fit(*datasets(), 4)
    assert trainer.compiled_window is not None
    assert history.steps[1].grad_norm == pytest.approx(5e20)
    assert trainer.skipped_nonfinite == 0
    assert addresses == [p.data_ptr() for p in trainer.model.parameters()]
    assert all(bool(torch.isfinite(p).all()) for p in trainer.model.parameters())


@pytest.mark.mps
def test_mps_wide_norm_uses_cpu_and_returns_gradient_in_place() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    p = parameter([3e20, -4e20], device="mps")
    address = p.grad.data_ptr()
    norm = clip_grad_norm_([p], 1.0, precision="float64")
    assert norm.device.type == "cpu" and float(norm) == pytest.approx(5e20)
    assert p.grad.device.type == "mps" and p.grad.data_ptr() == address
    torch.testing.assert_close(p.grad.cpu(), torch.tensor([0.6, -0.8]))


@pytest.mark.cuda
def test_grad_scaler_accepts_large_finite_and_rejects_real_inf() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    p = torch.nn.Parameter(torch.ones(2, device="cuda"))
    optimizer = torch.optim.SGD([p], lr=0.1)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    scaler.scale(p.sum()).backward()
    assert p.grad is not None
    p.grad.fill_(1e30)  # finite scaled gradients; inject BEFORE the scaler's nonfinite check
    scaler.unscale_(optimizer)
    norm = clip_grad_norm_([p], 1.0, precision="float64")
    assert math.isfinite(float(norm))
    before = p.detach().clone()
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(p, before) and bool(torch.isfinite(p).all())
    assert scaler.get_scale() == scale
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(p.sum()).backward()
    assert p.grad is not None
    p.grad[0] = math.inf
    scaler.unscale_(optimizer)
    norm = clip_grad_norm_([p], 1.0, precision="float64")
    assert math.isinf(float(norm))
    before = p.detach().clone()
    scaler.step(optimizer)
    scaler.update()
    assert torch.equal(p, before) and scaler.get_scale() < scale
