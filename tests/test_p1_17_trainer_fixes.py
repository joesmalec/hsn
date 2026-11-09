"""Training continuity and EMA clocks follow applied updates rather than fit boundaries."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hsn.config import load_config
from hsn.data.dataset import ByteDataset
from hsn.device import resolve, seed_everything, set_deterministic
from hsn.losses import assemble_terms
from hsn.runner import build_model
from hsn.train.loop import History, Trainer, detach_state
from hsn.train.probes import dependency_probe_coverage
from p1_15_support import (
    INF_GRADIENT,
    datasets,
    gru_config,
    gru_trainer,
    hsn_config,
    hsn_trainer,
    plant,
    train_overrides,
)


@pytest.fixture(autouse=True)
def _restore_determinism() -> Iterator[None]:
    before = torch.are_deterministic_algorithms_enabled()
    yield
    set_deterministic(before)


def _trainer(kind: str, root: Path) -> Trainer:
    # Windows cut inside documents, so continuing only the loader or only the state is wrong.
    extra = ("train.unroll=8", "train.divergence_dump=false")
    if kind == "hsn":
        return hsn_trainer(hsn_config(6, *extra), root)
    if kind == "gru":
        return gru_trainer(gru_config(6, *extra), root)
    cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs/baselines/transformer_2m.yaml",
        overrides=[
            *train_overrides(6),
            *extra,
            "baseline.transformer.d_model=16",
            "baseline.transformer.n_layers=1",
            "baseline.transformer.n_heads=2",
            "baseline.transformer.ffn_hidden=32",
        ],
    )
    seed_everything(0)
    return Trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=root)


@pytest.mark.parametrize("kind", ["hsn", "gru", "transformer"])
def test_repeated_fit_continues_loader_state_and_checkpoint(tmp_path: Path, kind: str) -> None:
    train, dev = datasets()
    whole = _trainer(kind, tmp_path / "whole")
    uninterrupted = whole.fit(train, dev, 6)
    split = _trainer(kind, tmp_path / "split")
    first = split.fit(train, dev, 2)
    second = split.fit(train, dev, 6)
    for got, expected in zip([*first.steps, *second.steps], uninterrupted.steps, strict=True):
        torch.testing.assert_close(
            torch.tensor([got.loss, got.grad_norm]),
            torch.tensor([expected.loss, expected.grad_norm]),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    for name, value in split.model.state_dict().items():
        expected_value = whole.model.state_dict()[name]
        assert (
            torch.equal(value, expected_value)
            if isinstance(value, torch.Tensor)
            else value == expected_value
        ), name
    full_checkpoint = torch.load(uninterrupted.checkpoints[-1], weights_only=True)
    split_checkpoint = torch.load(second.checkpoints[-1], weights_only=True)
    assert split_checkpoint["loader_state"] == full_checkpoint["loader_state"]
    assert split._loader_state == full_checkpoint["loader_state"]
    # Disk resume and in-memory continuation agree as well.
    resumed = _trainer(kind, tmp_path / "resumed")
    resumed.resume(first.checkpoints[-1])
    resumed.fit(train, dev, 6)
    for name, value in resumed.model.state_dict().items():
        expected_value = whole.model.state_dict()[name]
        assert (
            torch.equal(value, expected_value)
            if isinstance(value, torch.Tensor)
            else value == expected_value
        ), name


@pytest.mark.parametrize("kind", ["guard", "rollback", "scaler"])
def test_ema_target_moves_only_on_applied_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    extra = ("train.rollback_every=1",) if kind == "rollback" else ()
    trainer = hsn_trainer(
        hsn_config(
            4,
            "model.predictor_target=ema_encoder",
            "train.ema_tau=0.5",
            "train.divergence_dump=false",
            *extra,
        ),
        tmp_path,
    )
    if kind == "scaler":
        trainer.scaler = torch.amp.GradScaler("cpu", init_scale=2.0**10)
    plant(monkeypatch, trainer, {3: INF_GRADIENT})
    targets: dict[int, dict[str, torch.Tensor]] = {}

    def remember(t: Trainer, history: History) -> bool:
        del t, history
        assert trainer.hsn.ema_target is not None
        targets[trainer.step] = {
            name: value.clone() for name, value in trainer.hsn.ema_target.named_buffers()
        }
        return False

    trainer.fit(*datasets(), 4, callback=remember)
    prior = 1 if kind == "rollback" else 2
    assert all(torch.equal(targets[3][name], value) for name, value in targets[prior].items())
    assert any(not torch.equal(targets[1][name], value) for name, value in targets[2].items())
    if kind == "scaler":
        assert trainer.scaler is not None
        assert trainer.scaler.get_scale() == 2.0**9
    elif kind == "rollback":
        assert trainer.rollbacks[0]["restored_step"] == 1
    else:
        assert trainer.skipped_nonfinite == 1


@pytest.mark.parametrize("kind", ["hsn", "gru"])
def test_accumulation_weights_answers_and_preserves_auxiliary_average(
    tmp_path: Path, kind: str
) -> None:
    # Three windows have 1, 8 and 0 supervised positions. Every answer should have weight 1/9,
    # while HSN's prediction loss retains its existing 1/3-per-window convention.
    original = datasets()[0].examples[0]
    mask = np.zeros(24, dtype=np.bool_)
    mask[1] = True
    mask[8:16] = True
    example = replace(
        original,
        tokens=np.arange(50, 74, dtype=np.uint8),
        targets=np.arange(51, 75, dtype=np.uint8),
        context=np.zeros(24, dtype=np.uint8),
        loss_mask=mask,
        meta={**original.meta, "deps": [1] * 9, "dep_len": 1},
    )
    dataset = ByteDataset([example])
    changes = (
        "train.unroll=8",
        "train.batch=1",
        "train.micro_batch=1",
        "train.grad_accum=3",
        "train.grad_clip=0",
        "train.divergence_dump=false",
    )
    cfg = hsn_config(1, *changes) if kind == "hsn" else gru_config(1, *changes)
    factory = hsn_trainer if kind == "hsn" else gru_trainer
    reference = factory(cfg, tmp_path / "reference")
    loader = reference._loader(dataset, shuffle=True, epochs=None)
    state = reference.model.init_state(1, reference.ctx.device)
    objective: torch.Tensor | None = None
    for _ in range(3):
        batch = next(loader)
        detached = state.detach() if kind == "hsn" else detach_state(state)
        logits, state, info = reference._forward(batch, detached)
        mask_t = batch["loss_mask"]
        if kind == "hsn":
            terms = assemble_terms({"logits": logits, **info}, batch, cfg)
            term = terms["task"] * (int(mask_t.sum()) / 9)
            term = term + sum(value / 3 for name, value in terms.items() if name != "task")
        elif bool(mask_t.any()):
            term = (
                F.cross_entropy(logits[mask_t].float(), batch["targets"][mask_t], reduction="sum")
                / 9
            )
        else:
            term = logits.sum() * 0
        objective = term if objective is None else objective + term
    assert objective is not None
    objective.backward()
    expected = {
        name: p.grad.clone() for name, p in reference.model.named_parameters() if p.grad is not None
    }
    trainer = factory(cfg, tmp_path / "actual")
    actual: dict[str, torch.Tensor] = {}

    def capture(t: Trainer) -> None:
        actual.update(
            {name: p.grad.clone() for name, p in t.model.named_parameters() if p.grad is not None}
        )

    trainer.grad_hook = capture
    history = trainer.fit(dataset, dataset, 1)
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=3e-5, atol=1e-7, msg=name)
    assert history.steps[0].loss_positions == 9
    if kind == "hsn":
        assert history.steps[0].total == pytest.approx(float(objective.detach()), rel=1e-6)
        assert history.steps[0].parts["task"] == pytest.approx(history.steps[0].loss, rel=1e-6)


def test_active_parameters_count_distinct_units_per_nonpadding_position(tmp_path: Path) -> None:
    trainer = hsn_trainer(
        hsn_config(
            1,
            "model.n_units=3",
            "model.routing=topk",
            "train.gate_mode=ste",
        ),
        tmp_path,
    )
    report = trainer.hsn.param_report()
    shared = report["total_trainable"] - report["units"] - report["halting"]
    per_unit = report["units"] // 3
    info = {
        "unit_idx": torch.tensor([[[0, 0], [1, 2], [0, 1]]]),
        "valid": torch.ones(1, 3, 2, dtype=torch.bool),
        "dense_fallback": torch.tensor([[0, 0, 1]]),
    }
    batch = {"doc": torch.tensor([[0, 0, -1]])}
    assert trainer._window_parameter_sum(info, batch, 2, report) / 2 == shared + 1.5 * per_unit
    info["dense_fallback"][0, 0] = 1
    assert trainer._window_parameter_sum(info, batch, 2, report) / 2 == shared + 2.5 * per_unit


def test_sparse_history_reports_logical_active_parameters(tmp_path: Path) -> None:
    trainer = hsn_trainer(
        hsn_config(1, "model.n_units=3", "model.routing=topk", "train.gate_mode=ste"),
        tmp_path,
    )
    assert isinstance(trainer.hsn.scorer.net, torch.nn.Linear)
    with torch.no_grad():
        trainer.hsn.scorer.net.weight.zero_()
        trainer.hsn.scorer.net.bias.copy_(torch.tensor([10.0, -10.0, -10.0]))
    report = trainer.hsn.param_report()
    expected = report["total_trainable"] - report["halting"] - report["units"] * 2 / 3
    history = trainer.fit(*datasets(), 1)
    assert history.steps[0].active_units == [0]
    assert history.steps[0].stats["active_param_count"] == expected
    assert history.measured["active_param_count"] == expected
    assert history.measured["active_param_count"] < history.measured["param_count"]


@pytest.mark.parametrize("horizon", [4, None])
def test_probe_coverage_is_reported_for_training_windows(
    tmp_path: Path, horizon: int | None
) -> None:
    trainer = hsn_trainer(
        hsn_config(
            1,
            "model.routing=topk",
            "train.gate_mode=ste",
            "train.lambda_u=0.1",
            "train.probe_fraction=1",
            "train.n_probe=1",
            f"train.probe_horizon={horizon if horizon is not None else 'null'}",
        ),
        tmp_path,
    )
    train, dev = datasets()
    window = next(trainer._loader(train, shuffle=True, epochs=None))
    eligible, total = dependency_probe_coverage(window["loss_mask"], window["dep"], horizon)
    history = trainer.fit(train, dev, 1)
    assert total > 0
    assert history.measured["probe_dependency_eligible"] == eligible
    assert history.measured["probe_dependency_total"] == total
    assert history.measured["probe_dependency_fraction"] == eligible / total
    assert history.steps[0].stats["probe_dependency_eligible"] == eligible
