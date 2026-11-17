"""The memory-off 160 x 3 text recipe with PR #90's speed settings
(``configs/experiments/hsn_t_nomem_top0_w1024_l3_fast.yaml``; the user, 17 Nov 2026, asked that
runs use them).

The recipe is ``hsn_t_nomem_top0_w1024_l3.yaml`` with exactly four keys set:
``model.reader_mask=causal`` (``docs/reader_causal_flash.md``), ``train.host_sync=stats``,
``train.fused_adamw=true`` and ``train.divergence_dump=false`` (``docs/trainer_host_sync.md``):
the same configuration view and hash once the run's name and notes are aligned, on the CPU
profile and with the card's lines, and a spec on it expands to the arm ids of a spec on the
parent with those four lines. It loads with neither key refused: none of the per-step features
``stats`` refuses is on.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from hsn.config import REPO_ROOT, RunConfig, config_hash, host_sync_refusals, load_config
from hsn.experiments.spec import config_view, expand, family_of, load_spec, memory_key
from hsn.model.hsn_t import HSNTNoMemory
from p1_11_support import fake_repo, write_yaml
from test_hsn_t_nomem_w1024_l3_recipe import CARD, PARAMS

EXPERIMENTS = REPO_ROOT / "configs" / "experiments"
DATA = REPO_ROOT / "configs" / "data"
RECIPE = EXPERIMENTS / "hsn_t_nomem_top0_w1024_l3_fast.yaml"
PARENT = EXPERIMENTS / "hsn_t_nomem_top0_w1024_l3.yaml"
#: the four keys this recipe sets on the parent (PR #90)
FAST = (
    "model.reader_mask=causal",
    "train.host_sync=stats",
    "train.fused_adamw=true",
    "train.divergence_dump=false",
)
#: the dotted keys those overrides set, as a configuration dump has them
FAST_KEYS = {
    "model.reader_mask",
    "train.host_sync",
    "train.fused_adamw",
    "train.divergence_dump",
}


def _named_as(cfg: RunConfig, path: Path) -> RunConfig:
    """``cfg`` with the run name and notes of the recipe at ``path``, the two keys the
    configuration hash reads that are not part of the run (the arm id leaves them out)."""
    other = load_config(path)
    return cfg.model_copy(update={"run_id": other.run_id, "notes": other.notes})


def _flat(tree: Any, prefix: str = "") -> dict[str, Any]:
    """A configuration dump as ``{dotted key: leaf}``."""
    if not isinstance(tree, dict):
        return {prefix: tree}
    out: dict[str, Any] = {}
    for key, value in tree.items():
        out.update(_flat(value, f"{prefix}.{key}" if prefix else key))
    return out


@pytest.mark.parametrize("card", [(), CARD], ids=["cpu", "card"])
def test_recipe_loads_with_nothing_refused(card: tuple[str, ...]) -> None:
    cfg = load_config(RECIPE, list(card))  # RunConfig refuses at load: no ValueError
    assert host_sync_refusals(cfg) == []
    assert cfg.model_name == "hsn_t" and family_of(cfg) == "hsn_t_nomem"
    assert not cfg.model.memory and cfg.model.reader_mask == "causal"
    assert cfg.model.reader_span is None and cfg.model.reader_carry == 0 and cfg.data.align_docs
    train = cfg.train
    assert (train.host_sync, train.fused_adamw, train.divergence_dump) == ("stats", True, False)
    # the per-step features stats refuses, every one off
    assert train.spike_factor is None and train.slow_burst_factor is None
    assert train.rollback_every is None
    assert not train.adaptive_lr.enabled and not train.gradient_growth.enabled
    assert not train.update_limit.groups and train.weight_ema is None
    assert train.answer_weight == 1.0 and train.grad_norm_precision == "native"
    assert train.precision != "fp16" and train.optimizer == "adamw"
    hardware = (cfg.hardware.name, cfg.hardware.device)
    assert hardware == (("rtx3060", "cuda") if card else ("cpu", "cpu"))  # not MPS (refused)
    report = HSNTNoMemory.from_config(cfg).param_report()
    assert report["trainable"] == report["stored"] == PARAMS  # the parent's model


@pytest.mark.parametrize("card", [(), CARD], ids=["cpu", "card"])
def test_recipe_is_the_parent_with_the_four_keys(card: tuple[str, ...]) -> None:
    recipe = load_config(RECIPE, list(card))
    parent = load_config(PARENT, list(card))
    spec = load_config(PARENT, [*card, *FAST])
    assert config_view(recipe) == config_view(spec)  # what the arm id reads
    assert config_hash(_named_as(recipe, PARENT)) == config_hash(spec)
    protocol = {"name": "train_eval", "eval_batch": 32, "probe_batch": 64, "diag_docs": 32}
    assert memory_key(recipe, "none", protocol) == memory_key(spec, "none", protocol)
    # on the parent: exactly the four keys and the name and notes differ
    mine = _flat(recipe.model_dump(mode="json"))
    theirs = _flat(parent.model_dump(mode="json"))
    assert mine.keys() == theirs.keys()
    assert {k for k in mine if mine[k] != theirs[k]} == {*FAST_KEYS, "run_id", "notes"}
    assert config_hash(_named_as(recipe, PARENT)) != config_hash(parent)
    assert memory_key(recipe, "none", protocol) != memory_key(parent, "none", protocol)


def _repo(tmp: Path) -> Path:
    repo = fake_repo(tmp)
    built = repo / "data" / "tinystories_w1024"
    built.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / "tinystories_w1024.manifest.json", built / "manifest.json")
    return repo


def _arm_ids(tmp: Path, name: str, config: Path, arm: list[str]) -> dict[str, str]:
    """A 27k arm on ``config`` as the parent's runs were specified (seeds 0 and 1): the card's
    base, families block and protocol, the arm's overrides ``arm``."""
    raw = {
        "name": name,
        "timing": False,
        "base": {"config": str(config), "overrides": ["profile=rtx3060"]},
        "families": {"hsn_t_nomem": {"config": str(config), "overrides": list(CARD[1:4])}},
        "protocol": {
            "name": "train_eval",
            "steps": 27000,
            "diagnostics": False,
            "diag_every": 250,
            "diag_docs": 32,
            "eval_batch": 32,
            "checkpoint": "retained",
            "keep_every": 27000,
            "keep_last": 1,
        },
        "seeds": [0, 1],
        "arms": [{"name": "ts27_fast", "family": "hsn_t_nomem", "overrides": [*arm, *CARD[4:]]}],
    }
    path = write_yaml(tmp / f"{name}.yaml", raw)
    return {a.label: a.arm_id for a in expand(load_spec(path)[0], repo=_repo(tmp))}


def test_recipe_arms_are_the_parents_with_the_four_keys(tmp_path: Path) -> None:
    """A spec on the recipe and one on the parent with the four lines expand to the same arm ids
    (the base file's path is not part of an arm's identity); the parent's own differ."""
    from_parent = _arm_ids(tmp_path, "from_parent", PARENT, list(FAST))
    from_recipe = _arm_ids(tmp_path, "from_recipe", RECIPE, [])
    plain = _arm_ids(tmp_path, "plain", PARENT, [])
    assert from_recipe == from_parent
    assert len(from_recipe) == 2
    assert set(from_recipe.values()).isdisjoint(plain.values())
