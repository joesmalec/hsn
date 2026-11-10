"""A5 preventive update projection and independent decoder shortcut dropout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from cards import CUDA_PROFILE
from hsn.config import UpdateLimitConfig, config_hash
from hsn.train.rollback import SnapshotRing
from hsn.train.update_limit import UpdateLimiter
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, gru_config, gru_trainer, hsn_config, hsn_trainer
from test_a5_stability import A5, SLOW

pytestmark = pytest.mark.usefixtures("restore_determinism")

LIMIT = 'train.update_limit.groups={memory: {selectors: ["cell.*"], max_ratio: 0.0001}}'


def test_projection_bounds_actual_updates_keeps_moments_and_unlimited_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1)).double()
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    limiter = UpdateLimiter(
        model,
        UpdateLimitConfig.model_validate(
            {"groups": {"first": {"selectors": ["0.*"], "max_ratio": 0.001}}}
        ),
    )
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    ptrs = [p.data_ptr() for p in model.parameters()]
    limiter.capture()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    proposed = {n: p.detach().clone() for n, p in model.named_parameters()}
    moments = {p: {k: v.clone() for k, v in s.items()} for p, s in opt.state.items()}
    limiter.apply()
    ratio = torch.cat(
        [(p - initial[n]).flatten() for n, p in model.named_parameters() if n.startswith("0.")]
    ).norm()
    ratio /= torch.cat([p.flatten() for n, p in initial.items() if n.startswith("0.")]).norm()
    assert float(ratio.detach()) == pytest.approx(0.001)
    assert ptrs == [p.data_ptr() for p in model.parameters()]
    for n, p in model.named_parameters():
        if n.startswith("1."):
            assert torch.equal(p, proposed[n])
        for key, v in opt.state[p].items():
            assert torch.equal(v, moments[p][key])
    assert limiter.state_dict()["groups"]["first"]["clipped_steps"] == 1


def test_unclipped_proposal_is_bitwise_and_zero_norm_is_bounded() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    limiter = UpdateLimiter(
        model,
        UpdateLimitConfig.model_validate(
            {"norm_floor": 1.0, "groups": {"all": {"selectors": ["*"], "max_ratio": 0.01}}}
        ),
    )
    with torch.no_grad():
        model.weight.zero_()
        limiter.capture()
        model.weight.fill_(1.0)
        limiter.apply()
        assert model.weight.norm().item() == pytest.approx(0.01)
        limiter.capture()
        model.weight.add_(0.00001)
        expected = model.weight.clone()
        limiter.apply()
        assert torch.equal(model.weight, expected)


@pytest.mark.parametrize(
    "groups",
    [
        {"x": {"selectors": ["absent.*"], "max_ratio": 0.1}},
        {
            "x": {"selectors": ["*"], "max_ratio": 0.1},
            "y": {"selectors": ["weight"], "max_ratio": 0.1},
        },
    ],
)
def test_invalid_selectors_fail_loudly(groups: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="matches no parameters|overlap"):
        UpdateLimiter(torch.nn.Linear(2, 2), UpdateLimitConfig.model_validate({"groups": groups}))


@pytest.mark.parametrize("optimizer", ["adamw", "lazy_adamw"])
def test_limiter_and_dropout_resume_exactly(tmp_path: Path, optimizer: str) -> None:
    cfg = hsn_config(
        6,
        *A5,
        LIMIT,
        f"train.optimizer={optimizer}",
        "model.decoder_shortcut_dropout=0.3",
        "train.ckpt_every=3",
    )
    full = hsn_trainer(cfg, tmp_path / "full")
    expected = full.fit(*datasets(), 6)
    resumed = hsn_trainer(cfg, tmp_path / "resumed")
    resumed.resume(tmp_path / "full" / "step_00000003.pt")
    got = resumed.fit(*datasets(), 6)
    assert [s.loss for s in got.steps] == [s.loss for s in expected.steps[3:]]
    for name, value in full.model.state_dict().items():
        other = resumed.model.state_dict()[name]
        assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    assert full.update_limiter is not None and resumed.update_limiter is not None
    assert full.update_limiter.state_dict() == resumed.update_limiter.state_dict()
    assert full.update_limiter.applied_steps == 6


@pytest.mark.parametrize("family", ["hsn", "gru"])
def test_skipped_step_does_not_apply_limiter(tmp_path: Path, family: str) -> None:
    extra = ('train.update_limit.groups={all: {selectors: ["*"], max_ratio: 0.0001}}',)
    trainer = (
        hsn_trainer(hsn_config(4, *A5, *extra), tmp_path)
        if family == "hsn"
        else gru_trainer(gru_config(4, *extra), tmp_path)
    )

    def poison(t: Any) -> None:
        if t.step == 1:
            next(p.grad for p in t.model.parameters() if p.grad is not None).fill_(float("inf"))

    trainer.grad_hook = poison
    trainer.fit(*datasets(), 4)
    assert trainer.skipped_nonfinite == 1
    assert trainer.update_limiter is not None and trainer.update_limiter.applied_steps == 3


def test_dropout_leaves_readout_and_eval_intact_and_zero_consumes_no_rng(tmp_path: Path) -> None:
    model = hsn_trainer(hsn_config(1, *A5, "model.decoder_shortcut_dropout=1"), tmp_path).hsn
    q, u, r = [torch.randn(2, 3, model.cfg.d, requires_grad=True) for _ in range(3)]
    model.train()
    logits = model._decode(q, u, r)
    expected = model.decoder(torch.cat((q * 0, u * 0, r), dim=-1))
    assert torch.equal(logits, expected)
    logits.sum().backward()
    assert q.grad is not None and not q.grad.any()
    assert u.grad is not None and not u.grad.any()
    assert r.grad is not None and r.grad.any()
    model.eval()
    assert torch.equal(model._decode(q, u, r), model.decoder(torch.cat((q, u, r), dim=-1)))
    model = hsn_trainer(hsn_config(1, *A5), tmp_path / "off").hsn
    model.train()
    rng = torch.get_rng_state()
    assert torch.equal(model._decode(q, u, r), model.decoder(torch.cat((q, u, r), dim=-1)))
    assert torch.equal(rng, torch.get_rng_state())


def test_new_settings_validate_and_preserve_default_hashes() -> None:
    cfg = hsn_config(4, *A5)
    assert config_hash(cfg) == config_hash(
        hsn_config(
            4,
            *A5,
            "train.update_limit.groups={}",
            "train.rollback_selection=oldest",
            "model.decoder_shortcut_dropout=0",
        )
    )
    assert config_hash(cfg) != config_hash(hsn_config(4, *A5, LIMIT))
    for invalid in (
        "model.decoder_shortcut_dropout=1.1",
        "model.decoder_shortcut_dropout=nan",
        "train.rollback_selection=guard_horizon",
    ):
        with pytest.raises(ValueError):
            hsn_config(4, *A5, invalid)
    with pytest.raises(ValueError, match="requires model.selected_readout"):
        hsn_config(4, "model.decoder_shortcut_dropout=0.2")


def test_limiter_checkpoint_follows_group_names_when_mapping_order_changes() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    groups = {
        "first": {"selectors": ["0.*"], "max_ratio": 0.001},
        "second": {"selectors": ["1.*"], "max_ratio": 1.0},
    }
    source = UpdateLimiter(model, UpdateLimitConfig.model_validate({"groups": groups}))
    source.capture()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    source.apply()
    restored = UpdateLimiter(
        model, UpdateLimitConfig.model_validate({"groups": dict(reversed(groups.items()))})
    )
    restored.load_checkpoint_state(source.checkpoint_state())
    assert restored.state_dict() == source.state_dict()


def test_guard_horizon_selects_by_trigger_and_advances_anchor(tmp_path: Path) -> None:
    trainer = hsn_trainer(
        hsn_config(
            80, *A5, *SLOW, "train.rollback_every=2", "train.rollback_selection=guard_horizon"
        ),
        tmp_path,
    )
    for step in range(2, 22, 2):
        trainer.step = step
        trainer.take_rollback_snapshot()
    trainer._roll_back(22, "spike", 100, 1, guard="spike")
    assert trainer.rollbacks[-1]["restored_step"] == 20
    for step in range(24, 42, 2):
        trainer.step = step
        trainer.take_rollback_snapshot()
    trainer._roll_back(42, "spike", 100, 1, guard="slow_burst")
    assert trainer.rollbacks[-1]["restored_step"] == 26
    assert trainer.rollbacks[-1]["guard"] == "slow_burst"
    for step in range(44, 60, 2):
        trainer.step = step
        trainer.take_rollback_snapshot()
    trainer._roll_back(60, "spike", 100, 1, guard="slow_burst")
    assert trainer.rollbacks[-1]["restored_step"] == 44
    assert trainer.rollbacks[-1]["horizon_available"]
    ring = SnapshotRing(trainer.model, trainer.optimizer)
    ring.take(10)
    assert ring.restore(at_or_before=5).step == 10


@pytest.mark.cuda
def test_compiled_limiter_and_dropout_keep_parameter_storage(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cfg = hsn_config(
        4,
        *A5,
        LIMIT,
        "model.decoder_shortcut_dropout=0.3",
        f"profile={CUDA_PROFILE}",
        "train.precision=bf16",
        "train.deterministic=false",
        "hardware.compile_mode=reduce-overhead",
    )
    trainer = hsn_trainer(cfg, tmp_path)
    pointers = [p.data_ptr() for p in trainer.model.parameters()]
    history = trainer.fit(*datasets(), 4)
    assert trainer.compiled_window is not None
    assert pointers == [p.data_ptr() for p in trainer.model.parameters()]
    assert trainer.update_limiter is not None
    record = trainer.update_limiter.state_dict()["groups"]["memory"]
    assert record["clipped_steps"] > 0
    assert record["max_applied_ratio"] <= 0.000101
    assert torch.isfinite(torch.tensor([s.loss for s in history.steps])).all()
    torch._dynamo.reset()
