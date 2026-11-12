"""A5 stability: slow ramps, optimiser groups, rollback and exact resume."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from cards import CUDA_PROFILE
from hsn.config import config_hash
from hsn.train.loop import slow_burst_decision, spike_decision
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, gru_config, gru_trainer, hsn_config, hsn_trainer

pytestmark = pytest.mark.usefixtures("restore_determinism")

A5 = (
    "model.routing=topk",
    "train.gate_mode=ste",
    "model.key_address=true",
    "model.selected_readout=true",
)
SLOW = (
    "train.spike_factor=10000",
    "train.spike_window=4",
    "train.slow_burst_factor=2",
    "train.slow_burst_window=12",
)


def test_slow_guard_catches_ramp_that_running_median_follows() -> None:
    history: list[float] = []
    triggered = False
    for norm in [1.0] * 40 + [1.25**i for i in range(1, 25)]:
        assert not spike_decision(norm, history, 20, 4)[0]
        skip, baseline = slow_burst_decision(norm, history, 2, 4, 12)
        if skip:
            assert baseline == 1.0
            # A normal candidate is accepted even while the recent window is still elevated.
            assert not slow_burst_decision(1.0, history, 2, 4, 12)[0]
            triggered = True
            break
        history.append(norm)
    assert triggered


def test_slow_guard_needs_full_windows_and_sustained_growth() -> None:
    assert slow_burst_decision(100.0, [1.0] * 14, 10, 4, 12) == (False, None)
    # An isolated outlier is the sharp guard's job, not a median rise.
    assert slow_burst_decision(100.0, [1.0] * 16, 10, 4, 12) == (False, 1.0)
    assert slow_burst_decision(1.0, [1.0] * 40, 10, 4, 12) == (False, 1.0)
    # Strictly above the factor, just like the existing spike guard.
    assert slow_burst_decision(10.0, [1.0] * 16 + [10.0] * 3, 10, 4, 12) == (False, 1.0)


@pytest.mark.parametrize(
    "overrides",
    [
        ["train.slow_burst_factor=10"],
        ["train.slow_burst_window=50"],
        ["train.router_lr_mult=0"],
        ["train.readout_lr_mult=nan"],
        ["train.router_lr_mult=0.3"],  # dense router is unused
    ],
)
def test_invalid_stability_settings_are_rejected(overrides: list[str]) -> None:
    with pytest.raises(ValueError):
        hsn_config(4, *overrides)


@pytest.mark.parametrize("key", ["router_lr_mult", "readout_lr_mult"])
def test_module_rates_refused_for_baselines(key: str) -> None:
    with pytest.raises(ValueError, match="HSN-only"):
        gru_config(4, f"train.{key}=0.3")


def test_stability_keys_are_hashed_only_when_active() -> None:
    base = hsn_config(4, *A5)
    explicit = hsn_config(
        4,
        *A5,
        "train.router_lr_mult=1",
        "train.readout_lr_mult=1",
        "train.slow_burst_factor=null",
        "train.slow_burst_window=250",
    )
    assert config_hash(base) == config_hash(explicit)
    for override in ("train.router_lr_mult=0.3", "train.readout_lr_mult=0.3"):
        assert config_hash(hsn_config(4, *A5, override)) != config_hash(base)
    assert config_hash(hsn_config(4, *A5, *SLOW)) != config_hash(base)


@pytest.mark.parametrize("optimizer", ["adamw", "lazy_adamw"])
@pytest.mark.parametrize("cell_mult", [1.0, 0.5])
def test_module_rates_scale_updates_without_changing_cell_groups(
    tmp_path: Path, optimizer: str, cell_mult: float
) -> None:
    extra = (
        *A5,
        f"train.optimizer={optimizer}",
        f"train.cell_lr_mult={cell_mult}",
        "train.weight_decay=0",
        "train.schedule=constant",
    )
    base = hsn_trainer(hsn_config(8, *extra), tmp_path / "base")
    slow = hsn_trainer(
        hsn_config(8, *extra, "train.router_lr_mult=0.3", "train.readout_lr_mult=0.2"),
        tmp_path / "slow",
    )
    start = {n: p.detach().clone() for n, p in base.model.named_parameters()}
    all_ids = [id(p) for g in slow.optimizer.param_groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == {id(p) for p in slow.model.parameters() if p.requires_grad}
    units = {id(p) for p in slow.model.cell.units.parameters()}
    if optimizer == "lazy_adamw":
        assert {
            id(p) for g in slow.optimizer.param_groups if "n_units" in g for p in g["params"]
        } == units
    for trainer in (base, slow):
        for p in trainer.model.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        trainer.optimizer.step()
    for name, p in slow.model.named_parameters():
        if not p.requires_grad:
            continue
        rate = 0.3 if name.startswith("router.") else 0.2 if name.startswith("decoder.") else 1.0
        reference = dict(base.model.named_parameters())[name]
        torch.testing.assert_close(
            start[name] - p, (start[name] - reference) * rate, atol=6e-8, rtol=1e-4
        )


@pytest.mark.parametrize("optimizer", ["adamw", "lazy_adamw"])
def test_module_rates_survive_schedule_cuts_and_resume(tmp_path: Path, optimizer: str) -> None:
    cfg = hsn_config(
        8,
        *A5,
        *SLOW,
        f"train.optimizer={optimizer}",
        "train.router_lr_mult=0.3",
        "train.readout_lr_mult=0.2",
        "train.cell_lr_mult=0.5",
        "train.schedule=wsd",
        "train.decay_fraction=0.25",
        "train.warmup_steps=2",
        "train.rollback_every=2",
        "train.rollback_lr_mult=0.5",
        "train.rollback_lr_steps=3",
        "train.ckpt_every=4",
    )
    full = hsn_trainer(cfg, tmp_path / "full")
    # Exercise the same scheduler state as a rollback, without changing the training examples.
    full._lr_cut_until, full._lr_cut_mult = 6, 0.5
    expected = full.fit(*datasets(), 8)
    resumed = hsn_trainer(cfg, tmp_path / "resumed")
    resumed.resume(tmp_path / "full" / "step_00000004.pt")
    assert list(resumed._spike_history) == list(
        torch.load(tmp_path / "full" / "step_00000004.pt", weights_only=True)["extra"][
            "spike_history"
        ]
    )
    groups = resumed.optimizer.param_groups
    for g in groups:
        assert g["lr"] / groups[0]["lr"] == pytest.approx(g["initial_lr"] / cfg.train.lr)
    actual = resumed.fit(*datasets(), 8)
    assert [s.lr for s in actual.steps] == [s.lr for s in expected.steps[4:]]
    for name, value in full.model.state_dict().items():
        other = resumed.model.state_dict()[name]
        assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other, (
            name
        )


@pytest.mark.parametrize("family", ["hsn", "gru", pytest.param("hsn_cuda", marks=pytest.mark.cuda)])
def test_slow_ramp_rolls_back_before_rise_and_recovers(tmp_path: Path, family: str) -> None:
    extra = (
        *SLOW,
        "train.rollback_every=2",
        "train.rollback_lr_mult=0.5",
        "train.rollback_lr_steps=3",
    )
    if family == "hsn_cuda":
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device on this machine")
        extra += (
            f"profile={CUDA_PROFILE}",
            "train.precision=bf16",
            "train.deterministic=false",
            "hardware.compile_mode=reduce-overhead",
        )
    trainer = (
        hsn_trainer(hsn_config(52, *A5, *extra), tmp_path)
        if family.startswith("hsn")
        else gru_trainer(gru_config(52, *extra), tmp_path)
    )
    saved: dict[int, dict[str, torch.Tensor]] = {}

    def ramp(t: Any) -> None:
        norm = 1.0 if t.step < 40 or t.rollbacks else 1.25 ** (t.step + 1 - 40)
        grads = [p.grad for p in t.model.parameters() if p.grad is not None]
        for grad in grads:
            grad.zero_()
        grads[0].view(-1)[0] = norm

    def remember(t: Any, _history: Any) -> bool:
        saved[t.step] = {n: p.detach().clone() for n, p in t.model.named_parameters()}
        return False

    trainer.grad_hook = ramp
    history = trainer.fit(*datasets(), 52, callback=remember)
    assert len(trainer.rollbacks) == 1 and trainer.skipped_spike == 0
    event = trainer.rollbacks[0]
    assert event["guard"] == "slow_burst"
    assert event["step"] > 44 and event["restored_step"] < 40
    for name, value in saved[event["restored_step"]].items():
        assert torch.equal(saved[event["step"]][name], value), name
    assert math.isfinite(history.evals[-1]["loss"])
    assert trainer.step == 52
    assert trainer._ring is not None and trainer._ring.capacity == 10


def test_extended_guard_history_roundtrips_and_rejects_corruption(tmp_path: Path) -> None:
    trainer = hsn_trainer(hsn_config(1, *A5, *SLOW), tmp_path / "source")
    history = [1.0] * 12 + [2.0, 3.0, 4.0, 5.0]
    trainer._spike_history.extend(history)
    extra = trainer._guard_extra()
    resumed = hsn_trainer(trainer.cfg, tmp_path / "resumed")
    resumed._restore_guard(extra, "checkpoint")
    assert list(resumed._spike_history) == history
    assert resumed._guard_spike(6.0, 20)
    assert resumed.skipped_steps[-1]["guard"] == "slow_burst"
    with pytest.raises(ValueError, match="spike_history"):
        resumed._restore_guard({**extra, "spike_history": [1.0] * 17}, "bad")
