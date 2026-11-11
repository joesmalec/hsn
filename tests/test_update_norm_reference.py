"""Initial-reference update budgets and measurements after all parameter adjustments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from cards import CUDA_PROFILE
from hsn.config import UpdateLimitConfig, canonical_json, config_hash
from hsn.experiments.diagnostics import Diagnostics, _norm
from hsn.train.update_limit import UpdateLimiter
from p1_11_support import cpu_ctx, restore_determinism  # noqa: F401
from p1_15_support import datasets, gru_config, gru_trainer, hsn_config, hsn_trainer
from test_a5_stability import A5

LIMIT = 'train.update_limit.groups={all: {selectors: ["*"], max_ratio: 0.0001}}'
ANCHOR = "train.update_limit.norm_growth_limits={all: 1.0}"
pytestmark = pytest.mark.usefixtures("restore_determinism")


def test_radius_cannot_expand_with_growing_weights_and_current_cap_still_holds() -> None:
    model = torch.nn.Linear(2, 2, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(1)
    limiter = UpdateLimiter(
        model,
        UpdateLimitConfig.model_validate(
            {
                "groups": {"all": {"selectors": ["*"], "max_ratio": 0.01}},
                "norm_growth_limits": {"all": 1.0},
            }
        ),
    )
    for magnitude in (10.0, 0.1):
        with torch.no_grad():
            model.weight.fill_(magnitude)
            limiter.capture()
            before = model.weight.clone()
            model.weight.add_(1)
            limiter.apply()
        delta = float((model.weight - before).detach().norm())
        assert delta == pytest.approx(0.01 * min(float(before.norm()), 2.0))
    record = limiter.state_dict()["groups"]["all"]
    assert record["initial_norm"] == record["reference_norm_ceiling"] == 2


def test_zero_initial_norm_uses_configured_floor() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    limiter = UpdateLimiter(
        model,
        UpdateLimitConfig.model_validate(
            {
                "groups": {"all": {"selectors": ["*"], "max_ratio": 0.01}},
                "norm_growth_limits": {"all": 2.0},
                "norm_floor": 1.0,
            }
        ),
    )
    limiter.capture()
    with torch.no_grad():
        model.weight.fill_(1)
    limiter.apply()
    assert float(model.weight.detach().norm()) == pytest.approx(0.01)


@pytest.mark.parametrize("invalid", [{"absent": 1}, {"all": 0.5}, {"all": float("inf")}])
def test_invalid_anchors_fail(invalid: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        UpdateLimitConfig.model_validate(
            {
                "groups": {"all": {"selectors": ["*"], "max_ratio": 0.01}},
                "norm_growth_limits": invalid,
            }
        )


@pytest.mark.parametrize("family", ["hsn", "gru"])
def test_anchor_resume_uses_saved_initial_norm_and_training_is_exact(
    tmp_path: Path, family: str
) -> None:
    cfg_fn, trainer_fn = (hsn_config, hsn_trainer) if family == "hsn" else (gru_config, gru_trainer)
    cfg = cfg_fn(6, LIMIT, ANCHOR, "train.ckpt_every=3")
    full = trainer_fn(cfg, tmp_path / "full")
    result = full.fit(*datasets(), 6)
    resumed = trainer_fn(cfg, tmp_path / "resumed")
    assert resumed.update_limiter is not None
    resumed.update_limiter.initial_norms["all"].fill_(99)
    resumed.resume(tmp_path / "full" / "step_00000003.pt")
    other = resumed.fit(*datasets(), 6)
    assert [s.loss for s in other.steps] == [s.loss for s in result.steps[3:]]
    for a, b in zip(full.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.equal(a, b)
    assert full.update_limiter is not None
    assert full.update_limiter.state_dict() == resumed.update_limiter.state_dict()


def test_default_anchor_is_hash_omitted_for_existing_caps() -> None:
    cfg = hsn_config(4, *A5, LIMIT)
    assert "norm_growth_limits" not in canonical_json(cfg)
    assert config_hash(cfg) == config_hash(
        hsn_config(4, *A5, LIMIT, "train.update_limit.norm_growth_limits={}")
    )
    assert config_hash(cfg) != config_hash(hsn_config(4, *A5, LIMIT, ANCHOR))


@pytest.mark.parametrize("family", ["hsn", "gru"])
@pytest.mark.parametrize("growth", [False, True])
def test_applied_diagnostics_measure_final_weights_and_do_not_change_training(
    tmp_path: Path, family: str, growth: bool
) -> None:
    cfg_fn, trainer_fn = (hsn_config, hsn_trainer) if family == "hsn" else (gru_config, gru_trainer)
    options = (
        (
            "train.gradient_growth.enabled=true",
            'train.gradient_growth.groups={chosen: ["*"]}',
            "train.gradient_growth.window=1",
            "train.gradient_growth.reference_lag=1",
            "train.gradient_growth.interval=1",
        )
        if growth
        else ()
    )
    cfg = cfg_fn(4, LIMIT, ANCHOR, *options)

    def rising(t: Any) -> None:
        for p in t.model.parameters():
            if p.grad is not None:
                p.grad.fill_(10.0**t.step)

    reference = trainer_fn(cfg, tmp_path / "reference")
    reference.grad_hook = rising
    expected = reference.fit(*datasets(), 4)
    observed = trainer_fn(cfg, tmp_path / "observed")
    observed.grad_hook = rising
    train, dev = datasets()
    diag = Diagnostics(observed.model, cfg, cpu_ctx(), dev, 1)
    diag.attach(observed)
    before: dict[str, list[torch.Tensor]] = {}
    direct = []

    def save_before(opt: Any, args: Any, kwargs: Any) -> None:
        before.update({n: [p.detach().clone() for p in ps] for n, ps in diag.groups.items()})

    handle = observed.optimizer.register_step_pre_hook(save_before)

    def after(t: Any) -> None:
        direct.append(
            {
                n: _norm([p.detach() - b for p, b in zip(ps, before[n], strict=True)])
                / _norm(before[n])
                for n, ps in diag.groups.items()
            }
        )

    observed.update_post_hooks.append(after)
    actual = observed.fit(train, dev, 4)
    observed.update_post_hooks.remove(after)
    handle.remove()
    diag.detach()
    assert not observed.update_post_hooks
    assert [s.loss for s in actual.steps] == [s.loss for s in expected.steps]
    assert all(
        torch.equal(a, b)
        for a, b in zip(reference.model.parameters(), observed.model.parameters(), strict=True)
    )
    assert len(diag.records) == 4
    if growth:
        assert observed.gradient_growth is not None
        assert observed.gradient_growth.damped_steps["chosen"] > 0
    for record, measured in zip(diag.records, direct, strict=True):
        assert record["legacy_measurement_stage"] == "optimizer_proposal"
        for name, ratio in measured.items():
            assert record[f"applied_update_ratio_before/{name}"] == ratio
        if family == "hsn":
            assert "applied_state/gate/mean" in record
    assert (
        any(r["update_ratio/decoder"] > r["applied_update_ratio/decoder"] for r in diag.records)
        if family == "hsn"
        else True
    )


def test_applied_diagnostics_have_no_sample_for_skipped_step(tmp_path: Path) -> None:
    cfg = hsn_config(4, LIMIT, ANCHOR)
    trainer = hsn_trainer(cfg, tmp_path)
    train, dev = datasets()
    diag = Diagnostics(trainer.model, cfg, cpu_ctx(), dev, 1)
    diag.attach(trainer)

    def poison(t: Any) -> None:
        if t.step == 1:
            next(p.grad for p in t.model.parameters() if p.grad is not None).fill_(float("inf"))

    trainer.grad_hook = poison
    trainer.fit(train, dev, 4)
    diag.detach()
    assert [r["step"] for r in diag.records] == [1, 3, 4]
    assert all("applied_update_norm/cell" in r for r in diag.records)


@pytest.mark.cuda
def test_compiled_anchored_cap_and_final_diagnostics_preserve_storage(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cfg = hsn_config(
        4,
        *A5,
        LIMIT,
        ANCHOR,
        f"profile={CUDA_PROFILE}",
        "train.precision=bf16",
        "train.grad_norm_precision=float64",
        "train.deterministic=false",
        "hardware.compile_mode=reduce-overhead",
    )
    trainer = hsn_trainer(cfg, tmp_path)
    pointers = [p.data_ptr() for p in trainer.model.parameters()]
    train, dev = datasets()
    diag = Diagnostics(trainer.model, cfg, trainer.ctx, dev, 1)
    diag.attach(trainer)
    history = trainer.fit(train, dev, 4)
    diag.detach()
    assert trainer.compiled_window is not None
    assert pointers == [p.data_ptr() for p in trainer.model.parameters()]
    assert trainer.update_limiter is not None
    cap = trainer.update_limiter.state_dict()["groups"]["all"]
    assert cap["clipped_steps"] > 0
    assert cap["max_applied_ratio"] <= 0.000101
    for record in diag.records:
        delta = sum(v * v for k, v in record.items() if k.startswith("applied_update_norm/")) ** 0.5
        assert delta <= cap["reference_norm_ceiling"] * 0.000101
    assert torch.isfinite(torch.tensor([s.loss for s in history.steps])).all()
    torch._dynamo.reset()
