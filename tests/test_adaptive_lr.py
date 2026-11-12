"""Learning-rate feedback, real optimiser integration and exact continuation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from cards import CUDA_PROFILE
from hsn.config import AdaptiveLRConfig, RunConfig, config_hash
from hsn.experiments.controller import controller_progress
from hsn.experiments.protocols import _adaptive_record
from hsn.experiments.spec import arm_identity
from hsn.train.adaptive import AdaptiveLR
from hsn.train.checkpoint import capture_rng_state
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import (
    SPIKE,
    datasets,
    gru_config,
    gru_trainer,
    hsn_config,
    hsn_trainer,
    plant,
)

pytestmark = pytest.mark.usefixtures("restore_determinism")

ON = (
    "train.adaptive_lr.enabled=true",
    "train.adaptive_lr.interval=2",
    "train.adaptive_lr.monitor_docs=4",
    "train.adaptive_lr.patience=2",
    "train.adaptive_lr.cooldown=0",
    "train.adaptive_lr.grad_growth=1000000",
    "train.adaptive_lr.max_update_ratio=10",
    "train.adaptive_lr.router_max_update_ratio=10",
    "train.adaptive_lr.readout_max_update_ratio=10",
    "train.schedule=constant",
    "train.divergence_dump=false",
)
A5 = (
    "model.routing=topk",
    "train.gate_mode=ste",
    "model.key_address=true",
    "model.selected_readout=true",
)


def controller(**kwargs: Any) -> AdaptiveLR:
    return AdaptiveLR(
        AdaptiveLRConfig(interval=2, cooldown=0, **kwargs), {"shared", "router", "decoder"}
    )


def observe(c: AdaptiveLR, loss: float, grad: float = 1.0, **updates: float) -> str:
    for _ in range(c.cfg.interval):
        c.accepted(grad)
    return str(
        c.observe(
            c.state.accepted_steps, loss, {key: updates.get(key, 1e-4) for key in c.state.scales}
        )["decision"]
    )


def test_sustained_slide_cuts_cumulatively_and_stops_at_floor() -> None:
    c = controller(min_scale=0.125)
    assert observe(c, 1.0) == "baseline"
    assert observe(c, 1.02) == "hold"  # one noisy observation cannot cause a loss cut
    assert observe(c, 0.99) == "hold"
    for baseline in (1.0, 1.1, 1.2, 1.3):
        observe(c, baseline + 0.04)
        assert observe(c, baseline + 0.06) == "loss_cut"
    assert set(c.state.scales.values()) == {0.125}


def test_gradual_gradient_growth_uses_healthy_anchor_not_rolling_baseline() -> None:
    c = controller(patience=10)
    observe(c, 1.0)
    for grad in (1.5, 2.0, 2.5):
        assert observe(c, 1.02, grad) == "hold"
    assert observe(c, 1.03, 3.1) == "loss_and_gradient_cut"
    # Growth without degraded learning is not enough to label the model too hot.
    c = controller(patience=10)
    observe(c, 1.0)
    assert observe(c, 0.9, 10) == "improving"


def test_update_budget_cuts_only_offending_group() -> None:
    c = controller()
    observe(c, 1.0)
    observe(c, 0.9, router=0.02)
    assert observe(c, 0.8, router=0.02) == "update_cut"
    assert c.state.scales == {"shared": 1, "router": 0.5, "decoder": 1}


@pytest.mark.parametrize("flag", ["loss_cuts", "gradient_cuts", "update_cuts", "trial_increases"])
def test_independent_signal_ablations_and_emergency_cuts(flag: str) -> None:
    def trigger(c: AdaptiveLR) -> str:
        if flag == "trial_increases":
            c.guard(0, "earlier cut")
        observe(c, 1)
        if flag == "gradient_cuts":
            return observe(c, 1.02, 4)
        if flag == "update_cuts":
            observe(c, 0.9, router=0.02)
            return observe(c, 0.8, router=0.02)
        loss = 1.1 if flag == "loss_cuts" else 1
        observe(c, loss)
        return observe(c, loss)

    full, ablated = controller(), controller(**{flag: False})
    decision = trigger(full)
    assert decision in {"loss_cut", "loss_and_gradient_cut", "update_cut", "trial_started"}
    assert trigger(ablated) not in {
        "loss_cut",
        "loss_and_gradient_cut",
        "update_cut",
        "trial_started",
    }
    before = dict(ablated.state.scales)
    ablated.guard(100, "overflow")
    assert ablated.state.scales == {key: value * 0.5 for key, value in before.items()}
    # Raw measurements are still recorded for the ablated signal.
    assert ablated.state.observations[-2]["loss"] > 0


def test_experiment_240_config_identity_and_legacy_controller_resume() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures/adaptive_exp240.json").read_text())
    cfg = RunConfig.model_validate(fixture["config"])
    assert config_hash(cfg) == fixture["config_hash"]
    arm, _ = arm_identity(
        cfg,
        family="hsn",
        variant=fixture["variant_name"],
        variant_params=fixture["variant"],
        protocol=fixture["protocol"],
        early_stop=fixture["early_stop"],
        data=fixture["data_hash"],
    )
    assert arm == fixture["arm_id"]
    c = AdaptiveLR(cfg.train.adaptive_lr, {"shared", "router", "decoder"})
    c.load_state_dict(fixture["controller"])
    assert c.state.accepted_steps == 1000
    assert c.state.scales == {"shared": 1, "router": 0.5, "decoder": 1}
    assert c.state.observations == fixture["controller"]["observations"]
    changed_raw = cfg.model_dump()
    changed_raw["train"]["adaptive_lr"]["update_cuts"] = False
    changed = RunConfig.model_validate(changed_raw)
    assert config_hash(changed) != fixture["config_hash"]
    with pytest.raises(ValueError, match="configuration differs"):
        AdaptiveLR(changed.train.adaptive_lr, set(c.state.scales)).load_state_dict(
            fixture["controller"]
        )


@pytest.mark.parametrize("flag", ["gradient_cuts", "update_cuts"])
def test_disabled_signal_does_not_veto_plateau_trial(flag: str) -> None:
    c = controller(**{flag: False})
    c.guard(0, "make room for trial")
    observe(c, 1)
    stress = {"grad": 4.0} if flag == "gradient_cuts" else {"router": 0.02}
    observe(c, 1, **stress)
    assert observe(c, 1, **stress) == "trial_started"
    assert observe(c, 0.95, **stress) == "trial_accepted"


@pytest.mark.parametrize("improves", [False, True])
def test_bounded_trial_accept_or_reject_and_checkpoint(improves: bool) -> None:
    c = controller()
    c.guard(1, "test")
    observe(c, 1.0)
    observe(c, 1.0)
    assert observe(c, 1.0) == "trial_started"
    assert c.scale("shared") == 0.55
    resumed = controller()
    resumed.load_state_dict(c.state_dict())
    loss = 0.95 if improves else 1.0
    for _ in range(2):
        assert observe(c, loss) == observe(resumed, loss)
    assert c.state_dict() == resumed.state_dict()
    assert c.scale("shared") == (0.55 if improves else 0.5)
    if not improves:
        for _ in range(5):
            assert observe(c, 1.0) != "trial_started"


def test_guard_aborts_trial_and_cut_survives_cooldown() -> None:
    c = controller()
    c.guard(1, "test")
    for _ in range(3):
        observe(c, 1.0)
    assert c.state.trial is not None
    c.guard(10, "overflow")
    assert c.state.trial is None
    assert c.scale("shared") == 0.25  # cuts previous stable .5, not the trial .55
    c = AdaptiveLR(AdaptiveLRConfig(interval=2, cooldown=2), {"shared"})
    observe(c, 1.0)
    observe(c, 1.1)
    assert observe(c, 1.1) == "loss_cut"
    assert observe(c, 1.3) == "cooldown"
    assert observe(c, 1.4) == "cooldown"
    assert c.scale("shared") == 0.5


def test_ceiling_warmup_and_malformed_resume() -> None:
    c = controller()
    observe(c, 1.0)
    observe(c, 1.0)
    assert observe(c, 1.0) == "at_ceiling"
    raw = c.state_dict()
    for bad in ({}, {**raw, "scales": {"shared": 100}}, {**raw, "norms": [float("nan")]}):
        with pytest.raises(ValueError):
            c.load_state_dict(bad)
    c.bind_monitor("one")
    with pytest.raises(ValueError, match="fingerprint"):
        c.bind_monitor("two")
    c.accepted(1)
    c.accepted(1)
    assert c.observe(6, 100, dict.fromkeys(c.state.scales, 0), warmup=True)["decision"] == "warmup"
    assert c.scale("shared") == 1


@pytest.mark.parametrize("model", ["gru", "a5"])
def test_real_monitor_and_midwindow_resume_are_exact(tmp_path: Path, model: str) -> None:
    cfg_fn, make = (gru_config, gru_trainer) if model == "gru" else (hsn_config, hsn_trainer)
    cfg = cfg_fn(8, *ON, *(A5 if model == "a5" else ()), "train.ckpt_every=3")
    full = make(cfg, tmp_path / "full")
    expected = full.fit(*datasets(), 8)
    resumed = make(cfg, tmp_path / "resumed")
    resumed.resume(tmp_path / "full" / "step_00000003.pt")
    assert resumed.adaptive_lr is not None
    assert len(resumed.adaptive_lr.state.norms) == 1
    actual = resumed.fit(*datasets(), 8)
    assert [s.loss for s in actual.steps] == [s.loss for s in expected.steps[3:]]
    assert resumed.adaptive_lr.state_dict() == full.adaptive_lr.state_dict()
    for name, tensor in full.model.state_dict().items():
        other = resumed.model.state_dict()[name]
        assert torch.equal(tensor, other) if isinstance(tensor, torch.Tensor) else tensor == other
    assert _adaptive_record(resumed)["adaptive_lr"] == resumed.adaptive_lr.state_dict()


@pytest.mark.parametrize("optimizer", ["adamw", "lazy_adamw"])
def test_control_changes_real_group_lrs_and_preserves_multipliers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, optimizer: str
) -> None:
    cfg = hsn_config(
        8,
        *ON,
        *A5,
        f"train.optimizer={optimizer}",
        "train.router_lr_mult=0.3",
        "train.readout_lr_mult=0.2",
        "train.cell_lr_mult=0.4",
    )
    t = hsn_trainer(cfg, tmp_path)
    initial = [g["lr"] for g in t.optimizer.param_groups]
    losses = iter([1.0, 1.1, 1.2, 1.3])
    monkeypatch.setattr(t, "_adaptive_monitor_loss", lambda: next(losses))
    t.fit(*datasets(), 8)
    assert [g["lr"] for g in t.optimizer.param_groups] == [lr * 0.5 for lr in initial]
    assert t.adaptive_lr.state.observations[2]["decision"] == "loss_cut"
    # Reporting reads the optimizer, preserving both component and cell multipliers.
    live = controller_progress(t)["adaptive_lr"]
    assert live["last_adjustment"]["step"] == 6
    expected: dict[str, set[float]] = {}
    for group, initial_lr in zip(t.optimizer.param_groups, initial, strict=True):
        expected.setdefault(group["adaptive_group"], set()).add(initial_lr * 0.5)
    assert live["next_update_lrs"] == {key: sorted(values) for key, values in expected.items()}
    assert len(live["next_update_lrs"]["shared"]) == 2


def test_monitor_is_train_only_and_restores_rng_and_state(tmp_path: Path) -> None:
    t = hsn_trainer(hsn_config(4, *ON, *A5), tmp_path)
    train, dev = datasets()
    with pytest.raises(ValueError, match="TRAIN"):
        t._prepare_adaptive_monitor(dev)
    t._prepare_adaptive_monitor(train)
    t.model.train()
    before = copy.deepcopy(t.model.state_dict())
    rng = capture_rng_state()
    t._adaptive_monitor_loss()
    assert t.model.training
    assert torch.equal(rng["torch"], capture_rng_state()["torch"])
    assert t._carried_state is None and t._prev_tokens is None
    for name, tensor in before.items():
        other = t.model.state_dict()[name]
        assert torch.equal(tensor, other) if isinstance(tensor, torch.Tensor) else tensor == other


def test_disabled_hash_compatibility_and_conflicting_recovery_refused() -> None:
    assert config_hash(gru_config(4)) == config_hash(
        gru_config(4, "train.adaptive_lr.enabled=false")
    )
    assert config_hash(gru_config(4)) != config_hash(gru_config(4, *ON))
    with pytest.raises(ValueError, match="persistent recovery"):
        hsn_config(
            4,
            *ON,
            "train.rollback_every=1",
            "train.rollback_lr_steps=200",
            "train.rollback_lr_mult=0.5",
        )


def test_rollback_cuts_persist_and_skips_do_not_enter_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = hsn_config(
        12,
        *ON,
        *A5,
        "train.spike_factor=100",
        "train.spike_window=2",
        "train.rollback_every=2",
        "train.rollback_skip=1",
        "train.adaptive_lr.interval=20",
        "train.ckpt_every=8",
    )
    t = hsn_trainer(cfg, tmp_path / "full")
    plant(monkeypatch, t, {5: SPIKE, 8: SPIKE})
    t.fit(*datasets(), 12)
    assert [r["step"] for r in t.rollbacks] == [5, 8]
    assert t.adaptive_lr.scale("shared") == 0.25
    assert t.adaptive_lr.state.accepted_steps == 10
    assert len(t.adaptive_lr.state.norms) == 4
    resumed = hsn_trainer(cfg, tmp_path / "resume")
    resumed.resume(tmp_path / "full" / "step_00000008.pt")
    assert resumed.adaptive_lr.scale("shared") == 0.25
    assert resumed.adaptive_lr.state.norms == []
    assert all(g["lr"] == cfg.train.lr * 0.25 for g in resumed.optimizer.param_groups)


def test_monitor_does_not_change_training_trajectory_without_rate_decision(tmp_path: Path) -> None:
    base = gru_config(6, "train.schedule=constant", "train.divergence_dump=false")
    monitored = gru_config(
        6, *ON, "train.adaptive_lr.patience=100", "train.adaptive_lr.grad_growth=1000000"
    )
    reference = gru_trainer(base, tmp_path / "base")
    a = reference.fit(*datasets(), 6)
    checked = gru_trainer(monitored, tmp_path / "monitored")
    b = checked.fit(*datasets(), 6)
    assert [s.loss for s in a.steps] == [s.loss for s in b.steps]
    for name, value in reference.model.state_dict().items():
        assert torch.equal(value, checked.model.state_dict()[name])


def test_live_trial_and_repeated_fit_match_checkpoint_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = gru_config(8, *ON, "train.adaptive_lr.interval=1", "train.ckpt_every=3")
    t = gru_trainer(cfg, tmp_path / "full")
    # An earlier persistent cut makes room below the original-rate ceiling.
    t.adaptive_lr.guard(0, "test")
    for group in t.optimizer.param_groups:
        group["lr"] *= 0.5
    monkeypatch.setattr(t, "_adaptive_monitor_loss", lambda: 1.0 if t.step < 3 else 0.9)
    t.fit(*datasets(), 3)
    assert t.adaptive_lr.state.trial is not None
    live = controller_progress(t)["adaptive_lr"]
    assert live["trial"]["remaining"] == cfg.train.adaptive_lr.trial_windows
    assert live["latest_observation"]["decision"] == "trial_started"
    assert live["scales"] == {"shared": 0.55}
    resumed = gru_trainer(cfg, tmp_path / "resumed")
    resumed.resume(tmp_path / "full" / "step_00000003.pt")
    monkeypatch.setattr(resumed, "_adaptive_monitor_loss", lambda: 0.9)
    a = t.fit(*datasets(), 8)
    b = resumed.fit(*datasets(), 8)
    assert [s.loss for s in a.steps] == [s.loss for s in b.steps]
    assert t.adaptive_lr.state_dict() == resumed.adaptive_lr.state_dict()


@pytest.mark.cuda
def test_compiled_cuda_monitor_survives_rollback_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    cfg = hsn_config(
        10,
        *ON,
        *A5,
        f"profile={CUDA_PROFILE}",
        "train.precision=bf16",
        "hardware.compile_mode=reduce-overhead",
        "train.deterministic=false",
        "train.adaptive_lr.patience=100",
        "train.adaptive_lr.grad_growth=1000000",
        "train.spike_factor=100",
        "train.spike_window=2",
        "train.rollback_every=2",
        "train.rollback_skip=1",
        "train.ckpt_every=8",
    )
    t = hsn_trainer(cfg, tmp_path / "full")
    plant(monkeypatch, t, {5: SPIKE})
    t.fit(*datasets(), 8)
    assert len(t.rollbacks) == 1
    assert t.adaptive_lr.scale("shared") == 0.5
    assert t.adaptive_lr.state.accepted_steps == 7
    resumed = hsn_trainer(cfg, tmp_path / "resumed")
    resumed.resume(tmp_path / "full" / "step_00000008.pt")
    assert resumed.adaptive_lr.state_dict() == t.adaptive_lr.state_dict()
    plant(monkeypatch, resumed, {})
    resumed.fit(*datasets(), 10)
    assert resumed.adaptive_lr.scale("shared") == 0.5
    assert resumed.adaptive_lr.state.accepted_steps == 9
    assert resumed.adaptive_lr.state.observations[-1]["step"] == 9
