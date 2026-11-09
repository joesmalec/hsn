"""Long-delayed storage decisions must be eligible for a utility target."""

from pathlib import Path

import torch
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import generate
from hsn.device import resolve
from hsn.train.probes import choose_probe_position, dependency_probe_coverage
from hsn.train.tbptt import TBPTTTrainer
from test_p2_3_probes import _build, _tiny_task, _train_config


def test_p1_17_long_horizon_reaches_writes_without_collapsing_to_bos() -> None:
    mask = torch.zeros(1, 256, dtype=torch.bool)
    deps = torch.zeros(1, 256, dtype=torch.int32)
    mask[0, 120] = True
    deps[0, 120] = 118  # the value was written at position 2
    assert dependency_probe_coverage(mask, deps, 16) == (0, 1)
    assert dependency_probe_coverage(mask, deps, 256) == (0, 1)
    assert dependency_probe_coverage(mask, deps, None) == (1, 1)
    rng = torch.Generator().manual_seed(11)
    starts = {choose_probe_position(mask, None, rng) for _ in range(1000)}
    assert 2 in starts and len(starts) > 100
    assert choose_probe_position(mask, 256, rng) == 0


def test_p1_17_coverage_refuses_credit_across_tbptt_boundary() -> None:
    mask = torch.zeros(2, 32, dtype=torch.bool)
    deps = torch.zeros(2, 32, dtype=torch.int32)
    mask[:, 20] = True
    deps[:, 20] = torch.tensor([30, 10], dtype=torch.int32)
    assert dependency_probe_coverage(mask, deps, None) == (1, 2)


def test_p1_17_trainer_runs_to_end_utility_probes(tmp_path: Path) -> None:
    cfg = _train_config("train.probe_horizon=null", "train.steps=2", "train.eval_every=2")
    train = ByteDataset([generate(_tiny_task(), "train", i) for i in range(8)])
    dev = ByteDataset([generate(_tiny_task(), "dev", i) for i in range(4)])
    trainer = TBPTTTrainer(_build(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path)
    history = trainer.fit(train, dev, 2)
    assert sum(s.stats["probe_pairs"] for s in history.steps) > 0
    assert all(torch.isfinite(torch.tensor(s.total)) for s in history.steps)
    assert history.measured["probe_dependency_fraction"] == 1.0
