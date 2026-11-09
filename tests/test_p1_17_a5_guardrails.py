"""A5 refuses stale cost profiles and variants that would silently erase its readout."""

from pathlib import Path

import pytest
import torch
from hsn.config import REPO_ROOT, load_config
from hsn.device import resolve
from hsn.experiments.spec import SpecError, SweepSpec, expand
from hsn.experiments.variants import install_variant
from hsn.kernels.costs import cost_table_from_config, profile_dense_costs, write_cost_profile
from hsn.model.hsn import HSN


@pytest.mark.parametrize("scorer", ["linear", "mlp"])
def test_a5_preserves_fan_in_initialization(scorer: str) -> None:
    cfg = load_config(
        REPO_ROOT / "configs/tiny_gradcheck.yaml",
        ["model.routing=topk", "model.init=fan_in", f"model.scorer={scorer}"],
    )
    torch.manual_seed(12)
    base = HSN.from_config(cfg)
    after_base = torch.random.get_rng_state()
    cfg = cfg.model_copy(
        update={
            "model": cfg.model.model_copy(update={"key_address": True, "selected_readout": True})
        }
    )
    torch.manual_seed(12)
    model = HSN.from_config(cfg)
    assert torch.equal(torch.random.get_rng_state(), after_base)
    for name, param in model.named_parameters():
        original = dict(base.named_parameters())[name]
        prefix = param[:, : original.shape[1]] if param.ndim == 2 else param
        assert torch.equal(prefix, original), name
        assert bool(torch.isfinite(param).all()), name


@pytest.mark.parametrize("flag", ["key_address", "selected_readout"])
def test_p1_17_cost_profiles_bind_a5_options(tmp_path: Path, flag: str) -> None:
    cfg = load_config(
        REPO_ROOT / "configs/tiny_gradcheck.yaml",
        ["model.routing=topk", "train.gate_mode=ste"],
    )
    report = profile_dense_costs(HSN.from_config(cfg), resolve(cfg.hardware), batch=1, positions=1)
    assert report["cost_profile"][flag] is False
    # Historical profiles with no A5 fields remain valid for historical configurations.
    for key in ("key_address", "selected_readout"):
        report["cost_profile"].pop(key)
    path = write_cost_profile(report, tmp_path / "legacy.json")
    legacy = cfg.model.model_copy(update={"cost_profile": path})
    assert cost_table_from_config(legacy).n_cells == cfg.model.n_cells
    with pytest.raises(ValueError, match=rf"{flag}.*regenerate"):
        cost_table_from_config(legacy.model_copy(update={flag: True}))

    enabled = cfg.model_copy(update={"model": cfg.model.model_copy(update={flag: True})})
    report = profile_dense_costs(
        HSN.from_config(enabled), resolve(enabled.hardware), batch=1, positions=1
    )
    assert report["cost_profile"][flag] is True
    path = write_cost_profile(report, tmp_path / "a5.json")
    measured = enabled.model.model_copy(update={"cost_profile": path})
    assert cost_table_from_config(measured).n_cells == cfg.model.n_cells
    with pytest.raises(ValueError, match=rf"{flag}.*regenerate"):
        cost_table_from_config(measured.model_copy(update={flag: False}))


@pytest.mark.parametrize("flag", ["key_address", "selected_readout"])
def test_p1_17_a5_rejects_variants_before_training(flag: str) -> None:
    overrides = ["model.routing=topk", "train.gate_mode=ste", f"model.{flag}=true"]
    cfg = load_config(REPO_ROOT / "configs/tiny_gradcheck.yaml", overrides)
    model = HSN.from_config(cfg)
    with pytest.raises(ValueError, match="A5.*unsupported"):
        install_variant(model, "linear_head", seed=0)
    assert install_variant(model, "pdg", seed=0) == {"variant": "pdg"}

    spec = SweepSpec.model_validate(
        {
            "name": "a5-unsupported-variant",
            "base": {"config": "configs/tiny_gradcheck.yaml", "overrides": overrides},
            "protocol": {"name": "dry_run"},
            "arms": [{"name": "a5", "family": "hsn", "variant": "linear_head"}],
        }
    )
    with pytest.raises(SpecError, match="A5 requires variant pdg"):
        expand(spec)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("d", 8),
        ("hidden", 16),
        ("scorer", "mlp"),
        ("cell_norm", "none"),
        ("cand_norm", "rms"),
        ("in_degree", 1),
        ("vocab_size", 32),
        ("activation", "relu"),
    ],
)
def test_cost_profile_rejects_changed_compute(tmp_path: Path, field: str, value: str | int) -> None:
    cfg = load_config(REPO_ROOT / "configs/tiny_gradcheck.yaml")
    report = profile_dense_costs(HSN.from_config(cfg), resolve(cfg.hardware), batch=1, positions=1)
    path = write_cost_profile(report, tmp_path / "shape.json")
    cfg = cfg.model.model_copy(update={"cost_profile": path})
    assert cost_table_from_config(cfg).n_cells == cfg.n_cells
    with pytest.raises(ValueError, match=rf"{field}.*regenerate"):
        cost_table_from_config(cfg.model_copy(update={field: value}))


def test_legacy_profile_does_not_assume_new_norms(tmp_path: Path) -> None:
    cfg = load_config(REPO_ROOT / "configs/tiny_gradcheck.yaml")
    report = profile_dense_costs(HSN.from_config(cfg), resolve(cfg.hardware), batch=1, positions=1)
    for name in ("cell_norm", "cand_norm", "scorer", "scorer_hidden"):
        report["cost_profile"].pop(name)
    path = write_cost_profile(report, tmp_path / "legacy.json")
    cfg = cfg.model.model_copy(update={"cost_profile": path})
    assert cost_table_from_config(cfg).n_cells == cfg.n_cells
    with pytest.raises(ValueError, match="unrecorded; legacy default.*regenerate"):
        cost_table_from_config(cfg.model_copy(update={"cand_norm": "rms"}))
