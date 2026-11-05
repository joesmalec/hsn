"""The non-finite-gradient guard (G1 diagnosis of 5 Nov 2026): without a loss scaler, a step whose
global gradient norm is not finite leaves the weights untouched, zeroes the gradients, counts the
skip, logs it and lets training continue; the counter is in every TBPTT step record, in the run's
measured block and in the checkpoint."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest
import torch
from hsn.config import RunConfig, load_config, load_hardware_profile
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example, TaskSpec, generate, load_dataset_spec
from hsn.device import DeviceContext, resolve, seed_everything
from hsn.model.hsn import HSN
from hsn.train.loop import Trainer
from hsn.train.tbptt import TBPTTStepRecord, TBPTTTrainer
from torch import Tensor, nn

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
GRU_2M = REPO / "configs" / "baselines" / "gru_2m.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
UNROLL = 32


def _tiny_task() -> TaskSpec:
    task = SPEC.task("kv", "in_range")
    return TaskSpec.model_validate(
        {
            **task.model_dump(),
            "n_train": 32,
            "n_dev": 8,
            "n_test": 8,
            "length_range": [16, UNROLL],
            "dep_len_range": [4, 20],
            "unroll": UNROLL,
        }
    )


def _docs(split: str, n: int) -> list[Example]:
    return [generate(_tiny_task(), split, i) for i in range(n)]  # type: ignore[arg-type]


def _cpu() -> DeviceContext:
    return resolve(load_hardware_profile("cpu"))


def _hsn_config() -> RunConfig:
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
            f"train.unroll={UNROLL}",
            "train.batch=4",
            "train.micro_batch=4",
            "train.steps=3",
            "train.eval_every=100",
            "train.ckpt_every=3",
            "train.stats_every=1",
            "train.warmup_steps=0",
            "train.schedule=constant",
        ],
    )


def _gru_config() -> RunConfig:
    return load_config(
        GRU_2M,
        overrides=[
            "baseline.gru.d_model=16",
            "baseline.gru.n_layers=1",
            f"train.unroll={UNROLL}",
            "train.batch=4",
            "train.micro_batch=4",
            "train.steps=3",
            "train.eval_every=100",
            "train.ckpt_every=3",
            "train.warmup_steps=0",
            "train.schedule=constant",
        ],
    )


def _snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _changed(before: dict[str, Tensor], model: nn.Module) -> list[str]:
    return [name for name, p in model.named_parameters() if not torch.equal(before[name], p)]


def _infinite_grad(_: Tensor) -> Tensor:
    return torch.full_like(_, math.inf)


def test_p1_9_nonfinite_gradient_step_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """TBPTT trainer: an infinite gradient on one parameter at step 2 skips that optimizer step
    (weights unchanged, gradients zeroed), counts and logs it, the record carries the count, and
    step 3 trains normally; the counter survives a checkpoint round trip."""
    cfg = _hsn_config()
    train_ds, dev_ds = ByteDataset(_docs("train", 32)), ByteDataset(_docs("dev", 8))
    seed_everything(0)
    model = HSN.from_config(cfg)
    trainer = TBPTTTrainer(model, cfg, _cpu(), run_dir=tmp_path / "run")
    history = trainer.fit(train_ds, dev_ds, 1)
    assert trainer.skipped_nonfinite == 0
    before = _snapshot(model)
    target = next(p for p in model.parameters() if p.requires_grad)
    handle = target.register_hook(_infinite_grad)
    with caplog.at_level(logging.WARNING, logger="hsn.train"):
        history = trainer.fit(train_ds, dev_ds, 2)
    handle.remove()
    record = history.steps[-1]
    assert isinstance(record, TBPTTStepRecord) and record.step == 2
    assert not math.isfinite(record.grad_norm)
    assert trainer.skipped_nonfinite == 1
    assert record.stats["skipped_nonfinite"] == 1.0
    assert _changed(before, model) == [], "a skipped step must leave every weight untouched"
    assert all(p.grad is None for p in model.parameters()), "the gradients are zeroed (set to None)"
    assert any("optimizer step skipped" in m and "step 2" in m for m in caplog.messages)
    history = trainer.fit(train_ds, dev_ds, 3)
    assert history.steps[-1].step == 3 and math.isfinite(history.steps[-1].grad_norm)
    assert trainer.skipped_nonfinite == 1 and history.measured["skipped_nonfinite_steps"] == 1.0
    assert _changed(before, model), "a finite step after the skip trains normally"
    # the counter is checkpointed and restored
    seed_everything(1)
    other = TBPTTTrainer(HSN.from_config(cfg), cfg, _cpu(), run_dir=tmp_path / "other")
    assert other.resume(tmp_path / "run" / "step_00000003.pt") == 3
    assert other.skipped_nonfinite == 1


def test_p1_9_nonfinite_gradient_step_skipped_baseline(tmp_path: Path) -> None:
    """The generic P0.5 trainer (baselines) applies the same guard."""
    cfg = _gru_config()
    train_ds, dev_ds = ByteDataset(_docs("train", 32)), ByteDataset(_docs("dev", 8))
    seed_everything(0)
    from hsn.runner import build_model

    model = build_model(cfg)
    trainer = Trainer(model, cfg, _cpu(), run_dir=tmp_path / "run")
    trainer.fit(train_ds, dev_ds, 1)
    before = _snapshot(model)
    target = next(p for p in model.parameters() if p.requires_grad)
    handle = target.register_hook(_infinite_grad)
    history = trainer.fit(train_ds, dev_ds, 2)
    handle.remove()
    assert not math.isfinite(history.steps[-1].grad_norm)
    assert trainer.skipped_nonfinite == 1 and _changed(before, model) == []
    history = trainer.fit(train_ds, dev_ds, 3)
    assert history.measured["skipped_nonfinite_steps"] == 1.0 and _changed(before, model)
    seed_everything(1)
    other = Trainer(build_model(cfg), cfg, _cpu(), run_dir=tmp_path / "other")
    assert other.resume(tmp_path / "run" / "step_00000003.pt") == 3
    assert other.skipped_nonfinite == 1
