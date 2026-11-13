"""``train.answer_weight`` (opt-in diagnostic, 13 Nov 2026): the task loss as a weighted mean over
the loss positions, the answers (dependency longer than one symbol) weighted against the others.
Training only: no model value changes, and 1.0 is the plain mean, bit for bit."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from hsn.config import config_hash, load_config, split_window_scope_error
from hsn.data.dataset import ByteDataset
from hsn.device import seed_everything
from hsn.losses import assemble_terms, task_cross_entropy
from hsn.runner import build_model, make_trainer
from p1_11_support import cpu_ctx, docs, tiny_gru_config, tiny_hsn_config

RECIPE_A6 = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "hsn_recipe_a6.yaml"


def _positions(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(9, 16, generator=g)
    targets = torch.randint(0, 16, (9,), generator=g)
    dep = torch.tensor([1, 1, 40, 1, 200, 1, 1, 17, 1])  # three answers, six text positions
    return logits, targets, dep


def test_default_is_the_plain_mean_bit_for_bit() -> None:
    logits, targets, dep = _positions()
    plain = F.cross_entropy(logits, targets)
    assert torch.equal(task_cross_entropy(logits, targets, dep, 1.0), plain)
    assert torch.equal(task_cross_entropy(logits, targets, None, 1.0), plain)


def test_weighted_mean_against_a_direct_computation() -> None:
    logits, targets, dep = _positions()
    ce = F.cross_entropy(logits, targets, reduction="none")
    answers = dep > 1
    w = 8.0
    want = (w * ce[answers].sum() + ce[~answers].sum()) / (w * answers.sum() + (~answers).sum())
    got = task_cross_entropy(logits, targets, dep, w)
    assert got.item() == pytest.approx(want.item(), rel=1e-6)
    # a large weight tends to the answers' mean, a small one to the other positions' mean
    assert task_cross_entropy(logits, targets, dep, 1e6).item() == pytest.approx(
        ce[answers].mean().item(), rel=1e-4
    )
    assert task_cross_entropy(logits, targets, dep, 1e-6).item() == pytest.approx(
        ce[~answers].mean().item(), rel=1e-4
    )


def test_answers_only_window_is_unchanged_by_the_weight() -> None:
    logits, targets, _ = _positions()
    dep = torch.full((9,), 64)
    plain = F.cross_entropy(logits, targets)
    assert task_cross_entropy(logits, targets, dep, 8.0).item() == pytest.approx(plain.item())


def test_weight_needs_the_dependencies() -> None:
    logits, targets, dep = _positions()
    with pytest.raises(ValueError, match="dep"):
        task_cross_entropy(logits, targets, None, 4.0)
    with pytest.raises(ValueError, match="shape"):
        task_cross_entropy(logits, targets, dep[:-1], 4.0)


def test_assemble_terms_reads_the_window_dependencies() -> None:
    plain_cfg = tiny_hsn_config()
    cfg = tiny_hsn_config("train.answer_weight=8.0", "train.lambda_p=0.0")
    B, T, V = 2, 6, plain_cfg.model.vocab_size
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(B, T, V, generator=g)
    batch = {
        "targets": torch.randint(0, V, (B, T), generator=g),
        "loss_mask": torch.tensor([[0, 1, 0, 1, 1, 0], [1, 0, 0, 0, 1, 1]], dtype=torch.bool),
        "dep": torch.tensor([[0, 1, 0, 30, 1, 0], [1, 0, 0, 0, 90, 1]]),
        "doc": torch.zeros(B, T, dtype=torch.int64),
    }
    got = assemble_terms({"logits": logits}, batch, cfg)["task"]
    mask = batch["loss_mask"]
    want = task_cross_entropy(logits[mask], batch["targets"][mask], batch["dep"][mask], 8.0)
    assert torch.equal(got, want)
    assert got.item() != pytest.approx(F.cross_entropy(logits[mask], batch["targets"][mask]).item())


def test_config_hash_and_window_scope() -> None:
    base = tiny_hsn_config()
    assert base.train.answer_weight == 1.0
    assert config_hash(tiny_hsn_config("train.answer_weight=1.0")) == config_hash(base)
    assert config_hash(tiny_hsn_config("train.answer_weight=8.0")) != config_hash(base)
    with pytest.raises(ValueError):
        tiny_hsn_config("train.answer_weight=0.0")
    recipe = ["model.k_select=8", "train.gate_mode=ste"]
    assert split_window_scope_error(load_config(RECIPE_A6, recipe)) is None
    reason = split_window_scope_error(load_config(RECIPE_A6, [*recipe, "train.answer_weight=8.0"]))
    assert reason is not None and "answer_weight" in reason


@pytest.mark.parametrize("family", ["hsn", "gru"])
def test_training_runs_with_the_weight(family: str, tmp_path: Path) -> None:
    """Both trainers (the HSN's TBPTT trainer and the baselines' loop) take the weighted task
    loss. On a task whose loss positions are all answers the weight leaves the loss unchanged."""
    losses = []
    for weight in ("1.0", "8.0"):
        make = tiny_hsn_config if family == "hsn" else tiny_gru_config
        cfg = make(f"train.answer_weight={weight}", steps=2)
        seed_everything(0)
        model = build_model(cfg)
        trainer = make_trainer(model, cfg, cpu_ctx(), run_dir=tmp_path / f"{family}-{weight}")
        train, dev = (
            ByteDataset(docs("delayed", "train", 8)),
            ByteDataset(docs("delayed", "dev", 4)),
        )
        result = trainer.fit(train, dev, 2)
        losses.append(float(result.steps[-1].loss))
        assert math.isfinite(losses[-1])
    assert losses[0] == pytest.approx(losses[1], rel=1e-5)
