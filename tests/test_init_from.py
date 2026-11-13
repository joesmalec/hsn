"""``train.init_from`` (opt-in warm start, 13 Nov 2026): a run starts from the weights of another
run's checkpoint at step 0 with a fresh optimiser. Default off and left out of the hash."""

import math
from pathlib import Path

import pytest
import torch
from hsn.config import canonical_json, config_hash
from hsn.experiments.protocols import run_protocol
from hsn.model.hsn import HSN
from hsn.runner import warm_start
from hsn.train.checkpoint import latest_checkpoint
from p1_11_support import tiny_manifest  # noqa: F401
from p1_15_support import datasets, hsn_config, hsn_trainer
from test_a6_feedback_detach import A5, ALL
from test_p1_15_spike_guard import _arm

SMALL = (*A5, ALL, "model.cand_norm=rms", "model.n_cells=6", "model.k_select=3", "train.lr=3e-3")


def test_key_is_opt_in_and_hashed_only_when_set(tmp_path: Path) -> None:
    off = hsn_config(4, *SMALL)
    on = hsn_config(4, *SMALL, f"train.init_from={tmp_path / 'step_00000004.pt'}")
    assert off.train.init_from is None
    assert "init_from" not in canonical_json(off)
    assert config_hash(on) != config_hash(off)
    assert on.train.init_from == tmp_path / "step_00000004.pt"


@pytest.mark.parametrize("ema", [False, True])
def test_warm_start_loads_the_weights_and_the_new_run_trains(ema: bool, tmp_path: Path) -> None:
    extra = ("train.weight_ema=0.9",) if ema else ()
    steps = 6
    first = hsn_trainer(hsn_config(steps, *SMALL, *extra), tmp_path / "first")
    first.fit(*datasets(), steps)
    ckpt = latest_checkpoint(tmp_path / "first")
    assert ckpt is not None
    raw = {n: v.detach().clone() for n, v in first.model.named_parameters()}
    want = raw
    if ema:  # the average replaces the weights: the model the run's evaluations scored
        assert first.install_averaged_weights()
        want = {n: v.detach().clone() for n, v in first.model.named_parameters()}
        assert any(not torch.equal(want[n], raw[n]) for n in raw)

    # a second run of the same parameters, another seed
    cfg = hsn_config(steps, *SMALL, *extra, "seed=3", f"train.init_from={ckpt}")
    torch.manual_seed(3)
    model = HSN.from_config(cfg)
    before = {n: v.detach().clone() for n, v in model.named_parameters()}
    report = warm_start(model, cfg.train.init_from)
    assert report["step"] == steps and report["averaged"] is ema
    after = dict(model.named_parameters())
    assert any(not torch.equal(before[n], after[n]) for n in after)
    for name, value in want.items():
        assert torch.equal(after[name], value), name

    from hsn.device import resolve
    from hsn.train.tbptt import TBPTTTrainer

    trainer = TBPTTTrainer(model, cfg, resolve(cfg.hardware), run_dir=tmp_path / "second")
    assert trainer.step == 0
    history = trainer.fit(*datasets(), steps)
    assert len(history.steps) == steps and all(math.isfinite(s.loss) for s in history.steps)


def test_warm_start_refuses_a_missing_file_and_other_parameters(tmp_path: Path) -> None:
    cfg = hsn_config(2, *SMALL)
    model = HSN.from_config(cfg)
    with pytest.raises(FileNotFoundError):
        warm_start(model, tmp_path / "nothing.pt")
    first = hsn_trainer(cfg, tmp_path / "first")
    first.fit(*datasets(), 2)
    ckpt = latest_checkpoint(tmp_path / "first")
    other = HSN.from_config(hsn_config(2, *SMALL, "model.n_cells=8"))
    with pytest.raises((RuntimeError, ValueError)):
        warm_start(other, ckpt)


def test_the_run_record_names_the_warm_start(
    tmp_path: Path,
    tiny_manifest: Path,  # noqa: F811
) -> None:
    """Through the experiment protocol: the record of a warm-started arm carries ``init_from``
    (path, step, the source's configuration hash); an ordinary arm's record has no such key."""
    first = run_protocol(_arm(tmp_path / "first", tiny_manifest))
    assert "init_from" not in first
    ckpt = latest_checkpoint(tmp_path / "first")
    assert ckpt is not None, "the first arm left its last checkpoint"
    record = run_protocol(_arm(tmp_path / "second", tiny_manifest, f"train.init_from={ckpt}"))
    assert record["init_from"]["path"] == str(ckpt)
    assert record["init_from"]["step"] == 8
    assert record["final"]["steps"] == 8  # the warm-started run counts its own steps from 0
