"""P1.11 tests: the trainer hooks the sweep harness needs (``fit(..., callback=...)`` and
``Trainer.eval_batch``), on both trainers, on ``profile:cpu``.

A callback that never stops leaves the run exactly as it was (bitwise on the CPU, apart from wall
times); a stop writes a checkpoint at the stop step (once) and the measured numbers count the steps
that ran; the last step is never a stop. The evaluation batch changes only the number of windows
of an evaluation, and the metrics by rounding at most.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import RunConfig
from hsn.data.dataset import ByteDataset
from hsn.device import seed_everything
from hsn.runner import build_model, make_trainer
from hsn.train.loop import History, Trainer
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import (
    cpu_ctx,
    docs,
    restore_determinism,  # noqa: F401  (fixture)
    tiny_gru_config,
    tiny_hsn_config,
)

pytestmark = pytest.mark.usefixtures("restore_determinism")
# wall times and host memory readings differ between two identical runs
TIMING_KEYS = {"seconds", "tokens_per_s", "train_seconds", "peak_bytes", "reserved_bytes"}


def _data() -> tuple[ByteDataset, ByteDataset]:
    return ByteDataset(docs("kv", "train", 32)), ByteDataset(docs("kv", "dev", 12))


def _fit(
    cfg: RunConfig, run_dir: Path, steps: int, callback: Any = None
) -> tuple[Trainer, History]:
    seed_everything(cfg.seed)
    model = build_model(cfg)
    trainer = make_trainer(model, cfg, cpu_ctx(), run_dir=run_dir)
    train, dev = _data()
    return trainer, trainer.fit(train, dev, steps, callback=callback)


def _comparable(history: History) -> dict[str, Any]:
    raw = asdict(history)
    for record in raw["steps"]:
        record.pop("seconds")
    raw["measured"] = {k: v for k, v in raw["measured"].items() if k not in TIMING_KEYS}
    raw.pop("checkpoints")  # paths differ (run directories)
    return raw


def _nan_safe_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_nan_safe_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_nan_safe_equal(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


@pytest.mark.parametrize("family", ["hsn", "gru"])
def test_p1_11_trainer_callback_none_is_unchanged(family: str, tmp_path: Path) -> None:
    cfg = tiny_hsn_config() if family == "hsn" else tiny_gru_config()
    calls: list[int] = []

    def never(trainer: Trainer, history: History) -> bool:
        calls.append(trainer.step)
        assert history.steps[-1].step == trainer.step
        return False

    trainer_a, plain = _fit(cfg, tmp_path / "a", 6)
    trainer_b, hooked = _fit(cfg, tmp_path / "b", 6, callback=never)
    assert isinstance(trainer_a, TBPTTTrainer) == (family == "hsn")
    assert calls == [1, 2, 3, 4, 5], "once per step, never at the last step"
    assert _nan_safe_equal(_comparable(plain), _comparable(hooked))
    other = trainer_b.model.state_dict()
    for key, value in trainer_a.model.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other[key]), key
        else:
            assert value == other[key], key


@pytest.mark.parametrize("family", ["hsn", "gru"])
def test_p1_11_trainer_callback_stops(family: str, tmp_path: Path) -> None:
    cfg = tiny_hsn_config(steps=8) if family == "hsn" else tiny_gru_config(steps=8)

    def stop_at_3(trainer: Trainer, history: History) -> bool:
        if trainer.step == 3:
            # the step's evaluation (eval_every 3) is already in the history
            assert history.evals[-1]["step"] == 3
            return True
        return False

    trainer, history = _fit(cfg, tmp_path / "stop", 8, callback=stop_at_3)
    assert trainer.step == 3 and [r.step for r in history.steps] == [1, 2, 3]
    assert history.measured["steps"] == 3.0
    assert [Path(p).name for p in history.checkpoints] == ["step_00000003.pt"]
    assert (tmp_path / "stop" / "step_00000003.pt").exists()
    if family == "hsn":
        assert isinstance(trainer, TBPTTTrainer) and trainer._carried_state is not None

    # a stop on a checkpoint step does not write it twice; the last step never stops
    cfg2 = cfg.model_copy(update={"train": cfg.train.model_copy(update={"ckpt_every": 2})})
    _, history2 = _fit(cfg2, tmp_path / "dup", 8, callback=lambda t, h: t.step == 2)
    assert [Path(p).name for p in history2.checkpoints] == ["step_00000002.pt"]
    _, history3 = _fit(cfg, tmp_path / "last", 2, callback=lambda t, h: t.step == 2)
    assert [r.step for r in history3.steps] == [1, 2] and history3.evals[-1]["step"] == 2


@pytest.mark.parametrize("family", ["hsn", "gru"])
def test_p1_11_eval_batch_same_metrics(family: str, tmp_path: Path) -> None:
    cfg = tiny_hsn_config() if family == "hsn" else tiny_gru_config()
    trainer, _ = _fit(cfg, tmp_path / "run", 2)
    _, dev = _data()  # 12 documents
    base = trainer.evaluate(dev)
    results = {}
    for batch in (3, 12, 40):  # 40 > the number of documents: padded rows
        trainer.eval_batch = batch
        results[batch] = trainer.evaluate(dev)
    trainer.eval_batch = None
    again = trainer.evaluate(dev)
    assert again == base, "eval_batch None is the configured batch"
    assert base["n_windows"] == 3  # 12 documents in 4 streams, one window each
    assert {b: r["n_windows"] for b, r in results.items()} == {3: 4, 12: 1, 40: 1}
    for r in results.values():
        assert r["n_positions"] == base["n_positions"] and r["n_tokens"] == base["n_tokens"]
        assert r["accuracy"] == pytest.approx(base["accuracy"], abs=1e-12)
        assert r["loss"] == pytest.approx(base["loss"], rel=1e-5)
        for group, values in base["per_group"].items():
            assert r["per_group"][group]["loss"] == pytest.approx(values["loss"], rel=1e-5)
    with pytest.raises(ValueError, match="batch must be >= 1"):
        trainer.eval_batch = 0
        trainer.evaluate(dev)
