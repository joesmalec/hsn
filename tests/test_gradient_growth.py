"""Sustained raw-gradient growth, selective damping, recovery and reproducible resume."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from cards import CUDA_PROFILE
from hsn.config import GradientGrowthConfig, config_hash
from hsn.train.gradient_growth import GradientGrowth
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, gru_config, gru_trainer, hsn_config, hsn_trainer
from test_a5_stability import A5

pytestmark = pytest.mark.usefixtures("restore_determinism")


def controller(**kw: Any) -> tuple[torch.nn.Module, GradientGrowth]:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    cfg = GradientGrowthConfig.model_validate(
        dict(
            enabled=True,
            groups={"chosen": ["0.*"]},
            window=3,
            reference_lag=30,
            interval=1,
            growth_threshold=1.5,
            global_growth_threshold=1.25,
            min_share=0.2,
            min_scale=0.1,
            share_power=0,
            recovery_rate=0.2,
            **kw,
        )
    )
    return model, GradientGrowth(model, cfg)


def test_sustained_growth_damps_without_ratchet_and_recovers() -> None:
    _, c = controller()
    for step in range(1, 34):
        c.observe(step, 2.0, [1.0])
    assert c.scales["chosen"] == 1.0
    for step in range(34, 40):
        c.observe(step, 4.0, [3.0])
    assert c.scales["chosen"] == pytest.approx(0.5)
    assert sum(e["groups"]["chosen"]["hot"] for e in c.observations) > 1
    # A repeated warning sets the same target; it does not compound to the floor.
    assert min(e["groups"]["chosen"]["scale"] for e in c.observations) == 0.5
    for step in range(40, 80):
        c.observe(step, 4.0, [3.0])
    assert c.scales["chosen"] == 1.0
    assert any(e["groups"]["chosen"]["action"] == "recover" for e in c.observations)


def test_isolated_spike_and_small_share_do_not_damp() -> None:
    _, c = controller()
    for step in range(1, 34):
        c.observe(step, 2.0, [1.0])
    c.observe(34, 200.0, [100.0])
    assert c.scales["chosen"] == 1.0
    for step in range(35, 39):
        c.observe(step, 2.0, [1.0])
    assert c.scales["chosen"] == 1.0
    for step in range(39, 44):
        c.observe(step, 200.0, [3.0])
    assert c.scales["chosen"] == 1.0  # grows, but the selected group does not drive the global norm


def test_projection_changes_only_selected_weights_and_preserves_adam_state() -> None:
    model, c = controller()
    for step in range(1, 34):
        c.observe(step, 2.0, [1.0])
    for step in range(34, 37):
        c.observe(step, 4.0, [3.0])
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    c.capture()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ptrs = [p.data_ptr() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    proposal = {n: p.detach().clone() for n, p in model.named_parameters()}
    moments = {p: {k: v.clone() for k, v in values.items()} for p, values in opt.state.items()}
    event = c.apply()
    for name, p in model.named_parameters():
        if name.startswith("0."):
            torch.testing.assert_close(
                p, before[name] + (proposal[name] - before[name]) * 0.5, rtol=0, atol=0
            )
        else:
            assert torch.equal(p, proposal[name])
        for key, value in opt.state[p].items():
            assert torch.equal(value, moments[p][key])
    assert ptrs == [p.data_ptr() for p in model.parameters()]
    assert event is not None
    metrics = event["groups"]["chosen"]
    assert metrics["applied_update_ratio"] / metrics["proposed_update_ratio"] == pytest.approx(
        0.5, rel=1e-5
    )


def test_raw_norm_is_measured_before_clipping() -> None:
    model, c = controller()
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 100
    raw = c.raw_norms()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert raw.item() > 100 and c.raw_norms().item() < 1


GROWTH = (
    "train.gradient_growth.enabled=true",
    'train.gradient_growth.groups={chosen: ["cell.*"]}',
    "train.gradient_growth.window=1",
    "train.gradient_growth.reference_lag=2",
    "train.gradient_growth.interval=1",
    "train.gradient_growth.share_power=0",
    "train.gradient_growth.global_growth_threshold=1.1",
)


def rising_gradients(t: Any) -> None:
    for p in t.model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    next(
        p.grad
        for n, p in t.model.named_parameters()
        if n.startswith("cell.") and p.grad is not None
    ).view(-1)[0] = 1.0 if t.step < 2 else 8.0


@pytest.mark.parametrize("optimizer", ["adamw", "lazy_adamw"])
def test_growth_resume_preserves_history_scales_optimizer_and_stream(
    tmp_path: Path, optimizer: str
) -> None:
    cfg = hsn_config(8, *A5, *GROWTH, f"train.optimizer={optimizer}", "train.ckpt_every=4")
    full = hsn_trainer(cfg, tmp_path / "full")
    full.grad_hook = rising_gradients
    expected = full.fit(*datasets(), 8)
    resumed = hsn_trainer(cfg, tmp_path / "resumed")
    resumed.resume(tmp_path / "full" / "step_00000004.pt")
    resumed.grad_hook = rising_gradients
    actual = resumed.fit(*datasets(), 8)
    assert [s.loss for s in actual.steps] == [s.loss for s in expected.steps[4:]]
    for name, value in full.model.state_dict().items():
        other = resumed.model.state_dict()[name]
        assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    assert full.gradient_growth is not None and resumed.gradient_growth is not None
    assert full.gradient_growth.state_dict() == resumed.gradient_growth.state_dict()
    assert full.gradient_growth.damped_steps["chosen"] > 0
    for p, q in zip(full.model.parameters(), resumed.model.parameters(), strict=True):
        for key, value in full.optimizer.state[p].items():
            other = resumed.optimizer.state[q][key]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other


@pytest.mark.parametrize("family", ["hsn", "gru"])
def test_nonfinite_guard_does_not_observe_or_apply_growth(tmp_path: Path, family: str) -> None:
    options = ("train.gradient_growth.enabled=true", 'train.gradient_growth.groups={all: ["*"]}')
    trainer = (
        hsn_trainer(hsn_config(4, *A5, *options), tmp_path)
        if family == "hsn"
        else gru_trainer(gru_config(4, *options), tmp_path)
    )

    def poison(t: Any) -> None:
        if t.step == 1:
            next(p.grad for p in t.model.parameters() if p.grad is not None).fill_(float("inf"))

    trainer.grad_hook = poison
    trainer.fit(*datasets(), 4)
    c = trainer.gradient_growth
    assert c is not None and c.samples == c.applied_steps == 3
    assert trainer.skipped_nonfinite == 1


def test_hash_defaults_validation_and_corrupt_state() -> None:
    assert config_hash(hsn_config(4, *A5)) == config_hash(
        hsn_config(4, *A5, "train.gradient_growth.enabled=false")
    )
    assert config_hash(hsn_config(4, *A5)) != config_hash(hsn_config(4, *A5, *GROWTH))
    for extra in (
        "train.spike_factor=100",
        "train.rollback_every=1",
        "train.adaptive_lr.enabled=true",
    ):
        with pytest.raises(ValueError, match="requires adaptive_lr, rollback"):
            hsn_config(4, *A5, *GROWTH, extra)
    _, c = controller()
    c.observe(1, 2.0, [1.0])
    raw = copy.deepcopy(c.state_dict())
    raw["history"][0] = (float("nan"), 1.0)
    with pytest.raises(ValueError, match="norm history"):
        c.load_state_dict(raw)


@pytest.mark.cuda
def test_compiled_growth_changes_updates_and_preserves_storage(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cfg = hsn_config(
        6,
        *A5,
        *GROWTH,
        f"profile={CUDA_PROFILE}",
        "train.precision=bf16",
        "train.deterministic=false",
        "hardware.compile_mode=reduce-overhead",
    )
    trainer = hsn_trainer(cfg, tmp_path)
    trainer.grad_hook = rising_gradients
    ptrs = [p.data_ptr() for p in trainer.model.parameters()]
    history = trainer.fit(*datasets(), 6)
    assert trainer.compiled_window is not None
    assert ptrs == [p.data_ptr() for p in trainer.model.parameters()]
    assert (
        trainer.gradient_growth is not None and trainer.gradient_growth.damped_steps["chosen"] > 0
    )
    assert torch.isfinite(torch.tensor([s.loss for s in history.steps])).all()
    torch._dynamo.reset()
