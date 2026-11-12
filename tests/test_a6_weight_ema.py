"""A6: ``train.weight_ema`` averages the weights without changing a training step, and the
evaluations score the average (hsn.train.weight_ema; docs/amendments/A6_feedback_gradient.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from cards import CUDA_PROFILE
from hsn.config import canonical_json, config_hash
from hsn.train.checkpoint import load_checkpoint
from hsn.train.weight_ema import WARMUP, WeightEMA, install_averaged, warmup_decay
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, gru_config, gru_trainer, hsn_config, hsn_trainer
from torch import nn

pytestmark = pytest.mark.usefixtures("restore_determinism")

EMA = "train.weight_ema=0.9"
A5 = (
    "model.routing=topk",
    "train.gate_mode=ste",
    "model.key_address=true",
    "model.selected_readout=true",
)
EVERY = ("train.eval_every=2", "train.ckpt_every=4")


def _weights(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _trainer(kind: str, run_dir: Path, *overrides: str, steps: int = 8) -> Any:
    if kind == "hsn":
        return hsn_trainer(hsn_config(steps, *A5, *EVERY, *overrides), run_dir)
    return gru_trainer(gru_config(steps, *EVERY, *overrides), run_dir)


# --------------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------------


def test_default_is_off_and_keeps_the_hash() -> None:
    base = hsn_config(4)
    assert base.train.weight_ema is None
    assert "weight_ema" not in canonical_json(base)
    on = hsn_config(4, EMA)
    assert on.train.weight_ema == 0.9
    assert config_hash(on) != config_hash(base)
    assert config_hash(hsn_config(4, "train.weight_ema=0.99")) != config_hash(on)
    assert gru_config(4, EMA).train.weight_ema == 0.9  # any model


@pytest.mark.parametrize("value", ["0", "1", "1.5", "-0.1", "nan"])
def test_decay_outside_the_open_unit_interval_is_refused(value: str) -> None:
    with pytest.raises(ValueError, match="weight_ema"):
        hsn_config(4, f"train.weight_ema={value}")


def test_rollback_and_the_average_are_not_combined() -> None:
    with pytest.raises(ValueError, match="rollback"):
        hsn_config(4, EMA, "train.rollback_every=2")


# --------------------------------------------------------------------------------------------------
# the average
# --------------------------------------------------------------------------------------------------


def test_warmup_follows_the_weights_first_and_reaches_the_decay() -> None:
    assert warmup_decay(0.998, 0) == pytest.approx(1 / WARMUP)
    assert warmup_decay(0.998, 90) == pytest.approx(0.91)
    assert warmup_decay(0.998, 4489) < 0.998 and warmup_decay(0.998, 4490) == 0.998
    assert warmup_decay(0.99, 889) < 0.99 and warmup_decay(0.99, 890) == 0.99


def test_update_is_the_documented_recurrence() -> None:
    torch.manual_seed(0)
    model = nn.Linear(3, 2).double()
    ema = WeightEMA(model, 0.5)
    expected = _weights(model)
    for n in range(12):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        ema.update()
        d = min(0.5, (1 + n) / (WARMUP + n))
        for name, p in model.named_parameters():
            expected[name] = d * expected[name] + (1 - d) * p.detach()
    assert ema.updates == 12
    state = ema.state_dict()
    assert state["decay"] == 0.5 and state["updates"] == 12
    for name, value in expected.items():
        torch.testing.assert_close(state["shadow"][name], value, rtol=1e-12, atol=1e-12)


def test_averaged_block_swaps_in_place_and_restores_bit_for_bit() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    ema = WeightEMA(model, 0.5)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update()
    live = _weights(model)
    storage = model.weight.data_ptr()
    with ema.averaged():
        inside = _weights(model)
        assert model.weight.data_ptr() == storage  # in place: a compiled window sees it
        assert not _same(inside, live)
        assert _same(inside, dict(ema.state_dict()["shadow"]))
        with ema.averaged():  # nested: left as it is
            assert _same(_weights(model), inside)
        assert _same(_weights(model), inside)
        with pytest.raises(RuntimeError, match="averaged"):
            ema.update()
    assert _same(_weights(model), live)
    with pytest.raises(ZeroDivisionError), ema.averaged():
        raise ZeroDivisionError
    assert _same(_weights(model), live)  # restored when the block raises


def test_install_is_one_way() -> None:
    model = nn.Linear(2, 2)
    ema = WeightEMA(model, 0.5)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update()
    average = dict(ema.state_dict()["shadow"])
    ema.install()
    assert ema.installed and _same(_weights(model), average)
    with ema.averaged():
        assert _same(_weights(model), average)
    assert _same(_weights(model), average)
    with pytest.raises(RuntimeError, match="averaged"):
        ema.update()


def test_malformed_or_foreign_state_is_refused() -> None:
    model = nn.Linear(2, 2)
    ema = WeightEMA(model, 0.5)
    good = ema.state_dict()
    with pytest.raises(ValueError, match="missing"):
        ema.load_state_dict(None, "x.pt")
    with pytest.raises(ValueError, match="decay 0.25"):
        ema.load_state_dict({**good, "decay": 0.25}, "x.pt")
    with pytest.raises(ValueError, match="other parameters"):
        ema.load_state_dict({**good, "shadow": {"weight": good["shadow"]["weight"]}}, "x.pt")
    with pytest.raises(ValueError, match="shape"):
        ema.load_state_dict({**good, "shadow": {**good["shadow"], "bias": torch.zeros(3)}}, "x.pt")
    with pytest.raises(ValueError, match="must hold"):
        ema.load_state_dict({**good, "updates": -1}, "x.pt")
    with pytest.raises(ValueError, match="other parameters"):
        install_averaged(nn.Linear(2, 2, bias=False), {"weight_ema": good}, "x.pt")
    assert install_averaged(model, {}, "x.pt") is False


# --------------------------------------------------------------------------------------------------
# the trainers
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["hsn", "gru"])
def test_training_is_unchanged_and_the_evaluations_score_the_average(
    kind: str, tmp_path: Path
) -> None:
    train_ds, dev_ds = datasets()
    plain = _trainer(kind, tmp_path / "plain")
    expected = plain.fit(train_ds, dev_ds, 8)
    averaged = _trainer(kind, tmp_path / "averaged", EMA)
    history = averaged.fit(train_ds, dev_ds, 8)
    # the same steps, bit for bit: losses, gradient norms and the weights training ended on
    assert [s.loss for s in history.steps] == [s.loss for s in expected.steps]
    assert [s.grad_norm for s in history.steps] == [s.grad_norm for s in expected.steps]
    assert _same(_weights(averaged.model), _weights(plain.model))
    assert averaged.weight_ema is not None and averaged.weight_ema.updates == 8
    assert plain.weight_ema is None
    # the evaluations differ: they scored the average
    assert [e["loss"] for e in history.evals] != [e["loss"] for e in expected.evals]
    last = averaged.evaluate(dev_ds)
    assert last["loss"] == history.evals[-1]["loss"]
    assert averaged.evaluate(dev_ds, averaged=False)["loss"] == expected.evals[-1]["loss"]
    assert plain.evaluate(dev_ds, averaged=False)["loss"] == expected.evals[-1]["loss"]
    assert _same(_weights(averaged.model), _weights(plain.model))  # training weights are back
    # the same number from a second model that holds the checkpoint's average
    other = _trainer(kind, tmp_path / "other")
    loaded = load_checkpoint(tmp_path / "averaged" / "step_00000008.pt", other.model)
    assert _same(_weights(other.model), _weights(plain.model))  # "model" = the training weights
    assert other.evaluate(dev_ds)["loss"] == expected.evals[-1]["loss"]
    assert install_averaged(other.model, loaded["extra"])
    assert other.evaluate(dev_ds)["loss"] == last["loss"]
    reader = _trainer(kind, tmp_path / "reader")
    load_checkpoint(tmp_path / "averaged" / "step_00000008.pt", reader.model, averaged=True)
    assert _same(_weights(reader.model), _weights(other.model))
    # a checkpoint without an average loads as usual with averaged=True
    load_checkpoint(tmp_path / "plain" / "step_00000008.pt", reader.model, averaged=True)
    assert _same(_weights(reader.model), _weights(plain.model))
    with pytest.raises(ValueError, match="evaluation only"):
        load_checkpoint(
            tmp_path / "plain" / "step_00000008.pt", plain.model, plain.optimizer, averaged=True
        )


@pytest.mark.parametrize("kind", ["hsn", "gru"])
def test_resume_continues_the_average_exactly(kind: str, tmp_path: Path) -> None:
    train_ds, dev_ds = datasets()
    full = _trainer(kind, tmp_path / "full", EMA)
    expected = full.fit(train_ds, dev_ds, 8)
    resumed = _trainer(kind, tmp_path / "resumed", EMA)
    assert resumed.resume(tmp_path / "full" / "step_00000004.pt") == 4
    assert resumed.weight_ema is not None and resumed.weight_ema.updates == 4
    actual = resumed.fit(train_ds, dev_ds, 8)
    assert [s.loss for s in actual.steps] == [s.loss for s in expected.steps[4:]]
    assert [e["loss"] for e in actual.evals] == [e["loss"] for e in expected.evals[2:]]
    assert full.weight_ema is not None
    assert _same(
        dict(resumed.weight_ema.state_dict()["shadow"]),
        dict(full.weight_ema.state_dict()["shadow"]),
    )
    # a checkpoint of a run without the average is another configuration
    plain = _trainer(kind, tmp_path / "plain")
    with pytest.raises(ValueError, match="configuration"):
        plain.resume(tmp_path / "full" / "step_00000004.pt")


def test_installed_average_is_scored_as_is_and_refuses_more_training(tmp_path: Path) -> None:
    train_ds, dev_ds = datasets()
    trainer = _trainer("hsn", tmp_path / "run", EMA)
    history = trainer.fit(train_ds, dev_ds, 4)
    live = _weights(trainer.model)
    assert trainer.install_averaged_weights() is True
    assert trainer.weight_ema is not None
    assert _same(_weights(trainer.model), dict(trainer.weight_ema.state_dict()["shadow"]))
    assert not _same(_weights(trainer.model), live)
    assert trainer.evaluate(dev_ds)["loss"] == history.evals[-1]["loss"]
    with pytest.raises(RuntimeError, match="averaged weights"):
        trainer.fit(train_ds, dev_ds, 8)
    # resuming a checkpoint brings the training weights back, and training continues
    assert trainer.resume(tmp_path / "run" / "step_00000004.pt") == 4
    assert _same(_weights(trainer.model), live)
    trainer.fit(train_ds, dev_ds, 8)
    # without the key there is nothing to install
    assert _trainer("hsn", tmp_path / "plain").install_averaged_weights() is False


def test_load_for_evaluation_scores_a_checkpoints_average(tmp_path: Path) -> None:
    train_ds, dev_ds = datasets()
    trainer = _trainer("hsn", tmp_path / "run", EMA)
    history = trainer.fit(train_ds, dev_ds, 8)
    at_4 = next(e["loss"] for e in history.evals if e["step"] == 4)
    assert at_4 != history.evals[-1]["loss"]
    trainer.load_for_evaluation(tmp_path / "run" / "step_00000004.pt")
    assert trainer.evaluate(dev_ds)["loss"] == at_4  # the checkpoint's average, not the trainer's
    plain = _trainer("hsn", tmp_path / "plain")
    expected = plain.fit(train_ds, dev_ds, 8)
    plain.load_for_evaluation(tmp_path / "plain" / "step_00000004.pt")
    assert plain.evaluate(dev_ds)["loss"] == next(
        e["loss"] for e in expected.evals if e["step"] == 4
    )
    with pytest.raises(ValueError, match="no averaged weights"):
        trainer.load_for_evaluation(tmp_path / "plain" / "step_00000004.pt")


def test_a_skipped_step_is_not_averaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from p1_15_support import INF_GRADIENT, plant

    train_ds, dev_ds = datasets()
    trainer = _trainer("hsn", tmp_path / "run", EMA)
    plant(monkeypatch, trainer, {3: INF_GRADIENT})
    trainer.fit(train_ds, dev_ds, 6)
    assert trainer.skipped_nonfinite == 1
    assert trainer.weight_ema is not None and trainer.weight_ema.updates == 5


# --------------------------------------------------------------------------------------------------
# CUDA: the compiled evaluation window sees the weights swapped in place
# --------------------------------------------------------------------------------------------------


@pytest.mark.cuda
def test_compiled_evaluation_scores_the_average_on_cuda(tmp_path: Path) -> None:
    """With CUDA Graphs (reduce-overhead) the compiled evaluation window scores the averaged
    weights (the in-place swap reaches it) and, with ``averaged=False``, the training weights:
    each equals an eager model holding those weights, and the two differ."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    train_ds, dev_ds = datasets()
    cuda = (
        f"profile={CUDA_PROFILE}",
        "hardware.autocast_dtype=fp32",
        "train.deterministic=false",
        "train.lr=2e-2",  # weights that move, so the average is not the last step's
        "train.eval_every=10",
        "train.ckpt_every=20",
    )
    compiled = hsn_trainer(
        hsn_config(20, *A5, *cuda, EMA, "hardware.compile_mode=reduce-overhead"),
        tmp_path / "compiled",
    )
    assert compiled.compiled_window is not None
    history = compiled.fit(train_ds, dev_ds, 20)
    averaged = compiled.evaluate(dev_ds)["loss"]
    training = compiled.evaluate(dev_ds, averaged=False)["loss"]
    assert averaged == history.evals[-1]["loss"]
    assert compiled.evaluate(dev_ds)["loss"] == averaged  # the swap leaves no trace
    eager_cfg = hsn_config(20, *A5, *cuda, EMA, "hardware.compile_mode=none")
    path = tmp_path / "compiled" / "step_00000020.pt"
    eager = hsn_trainer(eager_cfg, tmp_path / "eager")
    assert eager.compiled_window is None
    load_checkpoint(path, eager.model)
    eager_training = eager.evaluate(dev_ds, averaged=False)["loss"]
    eager.load_for_evaluation(path)
    eager_averaged = eager.evaluate(dev_ds)["loss"]
    assert averaged == pytest.approx(eager_averaged, rel=1e-4)
    assert training == pytest.approx(eager_training, rel=1e-4)
    assert abs(averaged - training) > 100 * abs(averaged - eager_averaged) + 1e-4
