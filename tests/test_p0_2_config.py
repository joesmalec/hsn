"""P0.2 tests: configuration schema, dotted overrides, hashing, hardware profiles, run manifest."""

import json
import platform
from pathlib import Path
from typing import Any

import pytest
import yaml
from hsn.config import (
    MEASURED_KEYS,
    PROFILES,
    RunConfig,
    config_hash,
    dump_resolved,
    load_config,
    load_hardware_profile,
    write_run_manifest,
)
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "configs" / "_example.yaml"
PROFILE_DIR = REPO / "configs" / "hardware"
PROFILE_FILES = ("rtx3060", "m_series", "dgx_spark", "a100", "cpu")  # the P0.2 file list


def _write(path: Path, mapping: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return path


def _cpu_profile() -> dict[str, Any]:
    loaded = yaml.safe_load((PROFILE_DIR / "cpu.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _schema_leaves(model: type[BaseModel], prefix: str = "") -> set[str]:
    """Dotted leaf keys of a pydantic model, descending into nested models."""
    leaves: set[str] = set()
    for name, field in model.model_fields.items():
        annotation = field.annotation
        dotted = f"{prefix}{name}"
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            leaves |= _schema_leaves(annotation, f"{dotted}.")
        else:
            leaves.add(dotted)
    return leaves


def _yaml_leaves(mapping: dict[str, Any], prefix: str = "") -> set[str]:
    leaves: set[str] = set()
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            leaves |= _yaml_leaves(value, f"{dotted}.")
        else:
            leaves.add(dotted)
    return leaves


def _lookup(cfg: BaseModel, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        node = getattr(node, part)
    return node


# --------------------------------------------------------------------------------------------------
# Tests named by the chunk
# --------------------------------------------------------------------------------------------------


def test_p0_2_roundtrip(tmp_path: Path) -> None:
    cfg = load_config(EXAMPLE)
    resolved = tmp_path / "config.resolved.yaml"
    dump_resolved(cfg, resolved)
    again = load_config(resolved)
    assert again == cfg
    assert config_hash(again) == config_hash(cfg)

    text = resolved.read_text(encoding="utf-8")
    assert text.startswith("# ") and config_hash(cfg) in text.splitlines()[0]
    assert "\nhardware:\n" in text  # the resolved file carries the hardware block inline
    assert "!!python" not in text

    # With a profile-level stream override, a run-level stream key and a CLI stream override in
    # play: precedence is defaults < profile < run YAML < CLI, and the round trip stays exact.
    profile = {**_cpu_profile(), "stream": {"hot_windows": 2, "prefetch_top": "3K"}}
    src = _write(
        tmp_path / "run.yaml",
        {"profile": "cpu", "stream": {"hot_windows": 6}, "hardware": profile},
    )
    cfg2 = load_config(src, overrides=["stream.device_budget_bytes=4096"])
    assert cfg2.stream.hot_windows == 6  # run YAML beats the profile
    assert cfg2.stream.prefetch_top == "3K"  # profile beats the schema default
    assert cfg2.stream.device_budget_bytes == 4096  # CLI beats both
    assert cfg2.hardware.stream == {"hot_windows": 2, "prefetch_top": "3K"}
    dump_resolved(cfg2, tmp_path / "resolved2.yaml")
    assert load_config(tmp_path / "resolved2.yaml") == cfg2

    # A hardware block may also be a path relative to the config file.
    src3 = _write(tmp_path / "run3.yaml", {"profile": "cpu", "hardware": "hw.yaml"})
    _write(tmp_path / "hw.yaml", _cpu_profile())
    assert load_config(src3).hardware == cfg.hardware


def test_p0_2_overrides() -> None:
    cfg = load_config(
        EXAMPLE,
        overrides=[
            "model.k_select=8",
            "train.lr=5e-4",
            "train.deterministic=true",
            "model.routing=topk",
            "model.fixed_route=[0, 1]",
            "train.betas=[0.8, 0.95]",
            "model.deadline_ms=null",
            "replay.quotas={php: 10, go: 5}",
            "notes=123",
            "run_id=2024",
            "data.manifest=data/manifest_01.json",
            "model.scorer_features.exclude=[eps]",
            "hardware.vram_cap_fraction=0.5",
            "hardware.stream.hot_windows=9",
            "train.aux_enabled=[cost]",
            "stream.prefetch_top=12",
        ],
    )
    assert cfg.model.k_select == 8 and type(cfg.model.k_select) is int
    assert cfg.train.lr == 5e-4 and type(cfg.train.lr) is float
    assert cfg.train.deterministic is True
    assert cfg.model.routing == "topk"
    assert cfg.model.fixed_route == [0, 1]
    assert cfg.train.betas == (0.8, 0.95)
    assert cfg.model.deadline_ms is None
    assert cfg.replay.quotas == {"php": 10, "go": 5}
    assert cfg.notes == "123" and cfg.run_id == "2024"  # str fields take the text verbatim
    assert cfg.data.manifest == Path("data/manifest_01.json")
    assert cfg.model.scorer_features.exclude == ["eps"]
    assert cfg.hardware.vram_cap_fraction == 0.5
    assert cfg.hardware.stream == {"hot_windows": 9}
    assert (
        cfg.stream.hot_windows == 4
    )  # the YAML states stream.hot_windows, so it beats the profile
    assert cfg.train.aux_enabled == ["cost"]
    assert cfg.stream.prefetch_top == 12

    # `profile=` selects another profile file before it is loaded.
    m_series = load_config(EXAMPLE, overrides=["profile=m_series"])
    assert m_series.profile == "m_series" and m_series.hardware.device == "mps"

    for bad in (
        "model.k_select",  # no '='
        "=3",  # no key
        "model.nope=1",  # unknown key
        "nope.k_select=1",  # unknown section
        "model.k_select=abc",  # wrong type
        "model.d.x=1",  # a scalar has no sub-keys
        "model.k select=1",  # not an identifier
        "profile=tpu",  # unknown profile
    ):
        with pytest.raises(ValueError):
            load_config(EXAMPLE, overrides=[bad])


def test_p0_2_validators_reject() -> None:
    hardware = _cpu_profile()

    def build(**sections: Any) -> RunConfig:
        return RunConfig.model_validate({"hardware": hardware, **sections})

    with pytest.raises(ValueError, match="k_select"):
        build(model={"k_select": 33, "n_cells": 32})
    with pytest.raises(ValueError, match="k_min"):
        build(model={"k_min": 5, "k_select": 4})
    with pytest.raises(ValueError, match="in_degree"):
        build(model={"in_degree": 32, "n_cells": 32})
    with pytest.raises(ValueError, match="r_max"):
        build(model={"r_max": 0})
    with pytest.raises(ValueError, match="r_max"):
        build(model={"r_max": 4})
    assert build(model={"r_max": 3, "in_degree": 0}).model.r_max == 3
    with pytest.raises(ValueError, match="fixed_route"):
        build(model={"routing": "fixed"})
    with pytest.raises(ValueError, match="disabled_cells"):
        build(model={"disabled_cells": [32]})
    with pytest.raises(ValueError, match="deadline_ms"):
        build(model={"deadline_ms": 0})

    with pytest.raises(ValueError, match="one at a time"):
        build(train={"aux_enabled": ["cost", "balance"]})
    with pytest.raises(ValueError, match="one at a time"):
        build(train={"aux_enabled": ["cost", "balance"], "allow_multiple_aux": "   "})
    justified = build(
        model={"routing": "topk"},  # P2.4: cost / balance need a selection to penalise
        train={
            "aux_enabled": ["cost", "balance"],
            "allow_multiple_aux": "ablation 3: cost + balance",
        },
    )
    assert justified.train.aux_enabled == ["cost", "balance"]
    with pytest.raises(ValueError, match="twice"):
        build(train={"aux_enabled": ["cost", "cost"], "allow_multiple_aux": "x"})
    with pytest.raises(ValueError, match="unknown terms"):
        build(train={"aux_enabled": ["ponder"]})
    with pytest.raises(ValueError, match="betas"):
        build(train={"betas": [0.9, 1.0]})

    with pytest.raises(ValueError, match="[Ee]xtra"):
        build(model={"n_cell": 32})  # misspelt key: errors over silence
    with pytest.raises(ValueError, match="[Ee]xtra"):
        build(unknown_section={"a": 1})
    with pytest.raises(ValueError, match="profile"):
        build(profile="a100")
    with pytest.raises(ValueError, match="run_id"):
        build(run_id="../escape")
    with pytest.raises(ValueError, match="prefetch_top"):
        build(stream={"prefetch_top": "2k"})
    with pytest.raises(ValueError, match="nvcc_arch"):
        RunConfig.model_validate({"hardware": {**hardware, "device": "cuda"}})
    with pytest.raises(ValueError, match="nvcc_arch"):
        RunConfig.model_validate({"hardware": {**hardware, "nvcc_arch": ["sm_86"]}})
    with pytest.raises(ValueError, match="vram_cap_fraction"):
        RunConfig.model_validate({"hardware": {**hardware, "vram_cap_fraction": 0.0}})
    with pytest.raises(ValueError, match="[Ee]xtra"):
        RunConfig.model_validate({"hardware": {**hardware, "stream": {"nope": 1}}})
    with pytest.raises(ValueError):
        RunConfig.model_validate(
            {}
        )  # hardware is required: the profile file is the source of truth

    # Models are frozen: a config never changes after it has been hashed.
    cfg = build()
    with pytest.raises(ValueError):
        cfg.model.k_select = 2  # type: ignore[misc]


def test_p0_2_hash_stable_under_key_order(tmp_path: Path) -> None:
    ordered = {
        "run_id": "r",
        "seed": 3,
        "profile": "cpu",
        "model": {"d": 32, "hidden": 128, "n_cells": 8, "k_select": 2, "in_degree": 2},
        "train": {"unroll": 16, "batch": 2},
    }
    shuffled = {
        "train": {"batch": 2, "unroll": 16},
        "model": {"in_degree": 2, "k_select": 2, "n_cells": 8, "hidden": 128, "d": 32},
        "profile": "cpu",
        "seed": 3,
        "run_id": "r",
    }
    text_a = _write(tmp_path / "a.yaml", ordered).read_text(encoding="utf-8")
    text_b = _write(tmp_path / "b.yaml", shuffled).read_text(encoding="utf-8")
    assert text_a != text_b  # the files really differ in key order
    hash_a = config_hash(load_config(tmp_path / "a.yaml"))
    hash_b = config_hash(load_config(tmp_path / "b.yaml"))
    assert hash_a == hash_b
    assert len(hash_a) == 64 and set(hash_a) <= set("0123456789abcdef")

    # The hash covers every value and is the same whether the hardware came from the profile file,
    # an inline block or code.
    assert config_hash(load_config(tmp_path / "a.yaml", overrides=["seed=4"])) != hash_a
    inline = RunConfig.model_validate({**ordered, "hardware": _cpu_profile()})
    assert config_hash(inline) == hash_a
    assert config_hash(load_config(EXAMPLE)) != hash_a


def test_p0_2_manifest_requires_measured_keys(tmp_path: Path) -> None:
    cfg = load_config(EXAMPLE)
    full = {
        "param_count": 1_820_361,
        "peak_bytes": 29_360_128,
        "reserved_bytes": 33_554_432,
        "tokens_per_s": 1234.5,
        "launches_per_step": 412,
    }
    assert set(full) == set(MEASURED_KEYS)
    run_dir = tmp_path / "runs" / cfg.run_id

    for key in full:
        partial = {k: v for k, v in full.items() if k != key}
        with pytest.raises(ValueError, match=key):
            write_run_manifest(run_dir, cfg, partial)
    with pytest.raises(ValueError, match="param_count"):
        write_run_manifest(run_dir, cfg, {**full, "param_count": "many"})
    with pytest.raises(ValueError, match="launches_per_step"):
        write_run_manifest(run_dir, cfg, {**full, "launches_per_step": True})
    assert not run_dir.exists()  # nothing is written when the measured dict is rejected

    out = write_run_manifest(run_dir, cfg, {**full, "energy_j_per_token": 0.01})
    assert out == run_dir / "run_manifest.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {
        "git_rev",
        "dirty",
        "lock_hash",
        "python",
        "torch",
        "cuda_toolkit",
        "driver",
        "profile",
        "measured",
        "config_hash",
    }
    assert data["measured"] == {**full, "energy_j_per_token": 0.01}
    assert data["profile"] == cfg.profile == "cpu"
    assert data["config_hash"] == config_hash(cfg)
    assert data["python"] == platform.python_version()
    assert isinstance(data["dirty"], bool)
    assert isinstance(data["git_rev"], str) and len(data["git_rev"]) == 40  # this repository
    assert isinstance(data["lock_hash"], str) and len(data["lock_hash"]) == 64  # uv.lock
    assert data["torch"] is None or isinstance(data["torch"], str)
    for key in ("cuda_toolkit", "driver"):
        assert data[key] is None or isinstance(data[key], str)
    assert load_config(run_dir / "config.resolved.yaml") == cfg


# --------------------------------------------------------------------------------------------------
# Definition of done: every Appendix A key exists with its name and default; profiles load.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROFILE_FILES)
def test_p0_2_profiles_load(name: str) -> None:
    hardware = load_hardware_profile(name)
    assert hardware.name == name
    cfg = load_config(EXAMPLE, overrides=[f"profile={name}"])
    assert cfg.profile == name and cfg.hardware == hardware
    # §3.7 dtype policy
    assert (
        hardware.autocast_dtype == {"cuda": "bf16", "mps": "fp16", "cpu": "fp32"}[hardware.device]
    )
    assert hardware.master_dtype == "fp32"
    assert hardware.unified_memory == (name in {"dgx_spark", "m_series", "cpu"})
    assert (hardware.device == "cuda") == bool(hardware.nvcc_arch)


def test_p0_2_profile_files_match_schema_names() -> None:
    on_disk = sorted(path.stem for path in PROFILE_DIR.glob("*.yaml"))
    assert on_disk == sorted(PROFILE_FILES)
    assert set(on_disk) <= set(PROFILES)
    with pytest.raises(FileNotFoundError):
        load_hardware_profile("gpu8gb")  # listed in A.5; its file arrives with P4.3
    with pytest.raises(ValueError):
        load_hardware_profile("tpu")


# Every Appendix A key that states a default, with that default (the Definition of done).
APPENDIX_A_DEFAULTS: dict[str, Any] = {
    "model.vocab_size": 256,
    "model.d": 64,
    "model.n_cells": 32,
    "model.n_units": 8,
    "model.k_select": 4,
    "model.k_min": 1,
    "model.r_max": 1,
    "model.in_degree": 4,
    "model.random_edges": 0,
    "model.hidden": 512,
    "model.n_context": 16,
    "model.context_dim": 64,
    "model.activation": "gelu",
    "model.init": "trunc_normal_0.02",
    "model.scorer_hidden": 128,
    "model.scorer_features.exclude": [],
    "model.fixed_route": [],
    "model.budget": 1.0,
    "model.lambda_cost": 0.0,
    "model.deadline_ms": None,
    "model.static_k": True,
    "model.age_max": 64,
    "model.halt_threshold": 0.5,
    "model.disabled_cells": [],
    "model.ablate_shared_u": False,
    "model.state_schema_version": 1,
    "data.dataset": "synthetic_v1",
    "data.tokenizer": "bytes",
    "data.context_symbols": Path("configs/data/context_symbols.yaml"),
    "train.unroll": 256,
    "train.batch": 4,
    "train.micro_batch": 4,
    "train.grad_accum": 1,
    "train.grad_clip": 1.0,
    "train.q_resync_every": 64,
    "train.aux_enabled": [],
    "train.allow_multiple_aux": "",
    "train.probe_fraction": 0.05,
    "train.n_probe": 4,
    "train.probe_horizon": 16,
    "train.ema_decay": 0.99,
    "train.replay_fraction": 0.25,
    "train.distill_temperature": 2.0,
    "train.obsolete_label_versions": [],
    "train.ewc_cap": 1.0,
    "train.ewc_batches": 64,
    "train.pcgrad": False,
    "train.pcgrad_eps": 1e-8,
    "train.interference_window": 32,
    "train.shared_lr_scale_on_conflict": 0.5,
    "train.deterministic": False,
    "replay.near_dup_threshold": 0.9,
    "fast.rank": 8,
    "fast.retention_max_loss_pp": 1.0,
    "growth.windows": 3,
    "growth.cap": 0.25,
    "stream.enabled": False,
    "stream.policy": "utility_lru",
    "stream.hot_windows": 4,
    "stream.prefetch_top": "2K",
    "serve.persistence_enabled": False,
    "serve.encryption": "none",
    "serve.capture_policy": "disabled",
}

# Appendix A keys without a stated default: they must exist under exactly these names.
APPENDIX_A_KEYS_WITHOUT_DEFAULT: tuple[str, ...] = (
    "model.edge_norm",
    "model.scorer",
    "model.routing",
    "model.predictor_target",
    "data.manifest",
    "data.tasks",
    "data.split",
    "train.optimizer",
    "train.lr",
    "train.betas",
    "train.weight_decay",
    "train.warmup_steps",
    "train.schedule",
    "train.precision",
    "train.steps",
    "train.eval_every",
    "train.ckpt_every",
    "train.stats_every",
    "train.lambda_p",
    "train.lambda_r",
    "train.lambda_d",
    "train.lambda_e",
    "train.lambda_c",
    "train.lambda_b",
    "train.lambda_g",
    "train.lambda_u",
    "train.lambda_ponder",
    "train.gate_mode",
    "train.soft_until_step",
    "train.temperature_start",
    "train.temperature_end",
    "train.temperature_steps",
    "train.epsilon_start",
    "train.epsilon_end",
    "train.epsilon_steps",
    "train.p_dense",
    "train.ewc_groups",
    "train.activation_checkpoint",
    "replay.path",
    "replay.capacity",
    "replay.quotas",
    "replay.expiry_days",
    "fast.rho",
    "fast.eta",
    "fast.steps_max",
    "growth.threshold_per_task",
    "growth.budget_per_stage",
    "growth.quality_tolerance_pp",
    "growth.starting_params",
    "san.update_every_windows",
    "san.validation_windows",
    "san.max_edges",
    "san.k_random",
    "san.quality_tolerance_pp",
    "stream.device_budget_bytes",
    "stream.host_budget_bytes",
    "serve.state_ttl_s",
    "serve.limits.max_input_bytes",
    "serve.limits.max_tokens",
    "serve.limits.max_budget",
    "serve.limits.max_deadline_ms",
    "serve.limits.rps_per_tenant",
    "hardware.name",
    "hardware.device",
    "hardware.autocast_dtype",
    "hardware.master_dtype",
    "hardware.vram_bytes",
    "hardware.vram_cap_fraction",
    "hardware.unified_memory",
    "hardware.pinned_memory",
    "hardware.compile_mode",
    "hardware.nvcc_arch",
    "hardware.power_meter",
    "hardware.recipe.batch",
    "hardware.recipe.micro_batch",
    "hardware.recipe.unroll",
    "hardware.recipe.activation_checkpoint",
    "hardware.stream",
    "run_id",
    "seed",
    "notes",
    "profile",
    "model_name",
)


def test_p0_2_appendix_a_defaults() -> None:
    cfg = RunConfig.model_validate({"hardware": _cpu_profile()})
    for dotted, expected in APPENDIX_A_DEFAULTS.items():
        assert _lookup(cfg, dotted) == expected, dotted
    leaves = _schema_leaves(RunConfig)
    missing = [
        key for key in (*APPENDIX_A_DEFAULTS, *APPENDIX_A_KEYS_WITHOUT_DEFAULT) if key not in leaves
    ]
    assert missing == []
    # Defaults that are not derived quantities: no parameter count lives in the schema.
    assert "param_count" not in " ".join(leaves)


def test_p0_2_example_covers_every_key() -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert "hardware" not in raw  # the example refers to a profile instead of inlining hardware
    example_leaves = _yaml_leaves(raw)
    schema_leaves = {leaf for leaf in _schema_leaves(RunConfig) if not leaf.startswith("hardware.")}
    assert example_leaves == schema_leaves
    # ... and it states the defaults: loading it equals the schema defaults except run_id and notes.
    cfg = load_config(EXAMPLE)
    defaults = RunConfig.model_validate({"hardware": _cpu_profile()})
    assert cfg.model_dump(exclude={"run_id", "notes"}) == defaults.model_dump(
        exclude={"run_id", "notes"}
    )
