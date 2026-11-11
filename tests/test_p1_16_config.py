"""P1.16 tests: the rollback keys (``train.rollback_*``) and the hash registry that keeps every
configuration written before them at its hash (old checkpoints resume, old dumps replay)."""

from __future__ import annotations

import pytest
from hsn.config import (
    HASH_OMIT_AT_DEFAULT,
    REPO_ROOT,
    ROLLBACK_KEYS,
    TrainConfig,
    canonical_json,
    config_hash,
    load_config,
)

#: config_hash at main 80b893d (before P1.16), computed with that commit's code.
HASHES_80B893D = {
    ("configs/ref_a1.yaml", ()): "0831924010f9c5ecf6f1794c811f0c61ac96cdb5e3fbcc5a3c99b97a8001aa70",
    (
        "configs/experiments/g1_ref_a1.yaml",
        (),
    ): "ff3b6a42c8827327269edeeaf8704781b2f96d8c89205b3ea02fb73044c2f4a8",
    (
        "configs/baselines/gru_2m.yaml",
        (),
    ): "c8dcbaa48e3b27ecdde3d078359ab95f7482f49935d076bed34f521aeb88cf60",
    (
        "configs/baselines/transformer_2m.yaml",
        (),
    ): "65066a8915b79a328801b42780ae621f7185f23b553247a70c4f1b4dbd75071f",
    (
        "configs/experiments/g1_ref_a1.yaml",
        ("profile=rtx3060", "train.betas=[0.9,0.95]", "train.spike_factor=10000"),
    ): "1876e38d61301b6b7a4fae0ca5718f59dcec83d411dfc7e716dc2ffd5aa5dc66",
}


@pytest.mark.parametrize(("path", "overrides"), list(HASHES_80B893D))
def test_p1_16_config_hash_unchanged_at_defaults(path: str, overrides: tuple[str, ...]) -> None:
    cfg = load_config(REPO_ROOT / path, list(overrides))
    assert config_hash(cfg) == HASHES_80B893D[(path, overrides)]
    for dotted in HASH_OMIT_AT_DEFAULT:
        assert f'"{dotted.split(".")[1]}"' not in canonical_json(cfg)


def test_p1_16_rollback_keys_defaults_off() -> None:
    train = TrainConfig()
    assert train.rollback_every is None
    assert (train.rollback_skip, train.rollback_lr_mult, train.rollback_lr_steps) == (0, 1.0, 0)
    assert train.rollback_max == 5
    assert HASH_OMIT_AT_DEFAULT == (
        *tuple(f"train.{k}" for k in ROLLBACK_KEYS),
        "model.key_address",
        "model.selected_readout",
        "train.router_lr_mult",
        "train.readout_lr_mult",
        "train.slow_burst_factor",
        "train.slow_burst_window",
        "train.adaptive_lr",
        "train.adaptive_lr.loss_cuts",
        "train.adaptive_lr.gradient_cuts",
        "train.adaptive_lr.update_cuts",
        "train.adaptive_lr.trial_increases",
        "train.divergence_max_dumps",
        "train.divergence_healthy_every",
        "train.divergence_healthy_keep",
        "train.rollback_selection",
        "train.update_limit",
        "train.update_limit.norm_growth_limits",
        "model.decoder_shortcut_dropout",
        "train.gradient_growth",
        "train.grad_norm_precision",
        "train.feedback_detach",
        "train.state_grad_clip",
        "model.address_init",
        "train.state_grad_ratio",
        "train.weight_ema",
    )


def test_p1_16_rollback_key_set_changes_hash() -> None:
    base = load_config(REPO_ROOT / "configs/ref_a1.yaml")
    on = load_config(REPO_ROOT / "configs/ref_a1.yaml", ["train.rollback_every=50"])
    assert config_hash(on) != config_hash(base)
    assert '"rollback_every":50' in canonical_json(on)
    tuned = load_config(
        REPO_ROOT / "configs/ref_a1.yaml",
        ["train.rollback_every=50", "train.rollback_skip=10", "train.rollback_lr_mult=0.5"],
    )
    assert config_hash(tuned) != config_hash(on)


@pytest.mark.parametrize(
    "override",
    [
        "train.rollback_skip=10",
        "train.rollback_lr_mult=0.5",
        "train.rollback_lr_steps=200",
        "train.rollback_max=3",
        "train.rollback_selection=guard_horizon",
    ],
)
def test_p1_16_rollback_keys_refused_without_every(override: str) -> None:
    with pytest.raises(ValueError, match="rollback_every is not set"):
        load_config(REPO_ROOT / "configs/ref_a1.yaml", [override])


@pytest.mark.parametrize(
    "override",
    [
        "train.rollback_every=0",
        "train.rollback_lr_mult=0",
        "train.rollback_lr_mult=1.5",
        "train.rollback_max=0",
        "train.rollback_skip=-1",
    ],
)
def test_p1_16_rollback_keys_validated(override: str) -> None:
    with pytest.raises(ValueError):
        load_config(REPO_ROOT / "configs/ref_a1.yaml", ["train.rollback_every=50", override])
