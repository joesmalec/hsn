"""The text recipe of HSN-T without its memory with three reader blocks
(``configs/experiments/hsn_t_nomem_top0_w1024_l3.yaml``, the user's decision of 17 Nov 2026: "I
agree that 160X3 is the best middle ground").

The recipe loads and validates at 1,578,272 parameters; it resolves to the 16 Nov standard
``hsn_t_nomem_top0_w1024.yaml`` with ``model.reader_layers=3`` and
``model.reader_rotary_pairs=[20,20,0]`` (the same configuration hash once the run's name and notes
are aligned), and to ``hsn_t_nomem.yaml`` with the overrides of the runs behind the decision
(spec ``fast_readers``, arm ``ts27_nomem_w160l3_top0_w1024``, and its seed 1 in spec
``fast_160x3_s1``, run on main at 5c54832), on the CPU profile and with the card's lines; a spec
on it expands to the arm ids those runs recorded.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from hsn.config import REPO_ROOT, RunConfig, config_hash, load_config
from hsn.experiments.spec import config_view, expand, family_of, load_spec
from hsn.model.hsn_t import HSNTNoMemory
from hsn.train.memory_plan import memory_off_param_count
from p1_11_support import fake_repo, write_yaml
from test_hsn_t_nomem_w1024_recipe import DATA_HASH
from test_hsn_t_nomem_w1024_recipe import PARAMS as STANDARD_PARAMS

EXPERIMENTS = REPO_ROOT / "configs" / "experiments"
DATA = REPO_ROOT / "configs" / "data"
RECIPE = EXPERIMENTS / "hsn_t_nomem_top0_w1024_l3.yaml"
STANDARD = EXPERIMENTS / "hsn_t_nomem_top0_w1024.yaml"
NOMEM = EXPERIMENTS / "hsn_t_nomem.yaml"
#: the two keys this recipe changes on the standard
THREE_BLOCKS = ("model.reader_layers=3", "model.reader_rotary_pairs=[20,20,0]")
#: the overrides of arm ts27_nomem_w160l3_top0_w1024 (specs fast_readers and fast_160x3_s1) on
#: hsn_t_nomem.yaml, without its two hardware lines (in CARD)
SPEC_ARM = (
    "data.dataset=tinystories_w1024",
    "data.manifest=data/tinystories_w1024/manifest.json",
    "data.tasks=[tinystories]",
    "data.variants=[out_of_range]",
    "train.unroll=1024",
    "train.batch=8",
    "train.micro_batch=8",
    "model.reader_width=160",
    "model.reader_layers=3",
    "model.reader_rotary_pairs=[20,20,0]",
)
#: the card's lines of those specs (base, families block and arm), left to a spec by the recipe
CARD = (
    "profile=rtx3060",
    "train.steps=27000",
    "train.ckpt_every=27000",
    "train.eval_every=1500",
    "hardware.compile_mode=none",
    "hardware.baseline_compile=reduce-overhead",
)
#: arm ids of those runs (their result.json, 17 Nov 2026); seed 1 shares seed 0's group
RECORDED = {
    "ts27_nomem_w160l3_top0_w1024/s0": "cbf69170a5493817",
    "ts27_nomem_w160l3_top0_w1024/s1": "0473099ae665f4ca",
}
PARAMS = 1_578_272
#: one reader block of width 160 (the standard's fourth)
BLOCK = 309_280


def _named_as(cfg: RunConfig, path: Path) -> RunConfig:
    """``cfg`` with the run name and notes of the recipe at ``path``, the two keys the
    configuration hash reads that are not part of the run (the arm id leaves them out)."""
    other = load_config(path)
    return cfg.model_copy(update={"run_id": other.run_id, "notes": other.notes})


def test_recipe_loads_at_the_decided_shape() -> None:
    cfg = load_config(RECIPE)
    assert cfg.model_name == "hsn_t" and family_of(cfg) == "hsn_t_nomem"
    m = cfg.model
    assert not m.memory and m.reader == "transformer"
    assert (m.reader_width, m.reader_layers, m.reader_heads) == (160, 3, 4)
    assert m.reader_rotary_pairs == [20, 20, 0]  # top0
    assert m.reader_span is None and m.reader_carry == 0 and m.reader_top_words == 0
    assert cfg.data.dataset == "tinystories_w1024" and cfg.data.tasks == ["tinystories"]
    assert cfg.data.align_docs
    assert (cfg.train.unroll, cfg.train.batch, cfg.train.micro_batch) == (1024, 8, 8)
    assert cfg.train.optimizer == "adamw" and cfg.train.lr == 1.0e-3 and cfg.train.lambda_p == 0
    assert cfg.train.warmup_steps == 100 and cfg.train.schedule == "cosine"
    assert cfg.hardware.name == "cpu"  # the card is a spec's override
    report = HSNTNoMemory.from_config(cfg).param_report()
    assert report["trainable"] == report["stored"] == PARAMS  # as the runs' result.json
    assert memory_off_param_count(m)["total"] == PARAMS
    assert STANDARD_PARAMS - PARAMS == BLOCK


@pytest.mark.parametrize("card", [(), CARD], ids=["cpu", "card"])
def test_recipe_is_the_standard_with_three_blocks(card: tuple[str, ...]) -> None:
    recipe = load_config(RECIPE, list(card))
    standard = load_config(STANDARD, [*card, *THREE_BLOCKS])
    assert config_view(recipe) == config_view(standard)  # what the arm id reads
    assert config_hash(_named_as(recipe, STANDARD)) == config_hash(standard)
    assert recipe.run_id != standard.run_id  # the only differences: the name and the notes


@pytest.mark.parametrize("card", [(), CARD], ids=["cpu", "card"])
def test_recipe_is_nomem_with_the_overrides_of_the_runs(card: tuple[str, ...]) -> None:
    recipe = load_config(RECIPE, list(card))
    spec = load_config(NOMEM, [*card, *SPEC_ARM])
    assert config_view(recipe) == config_view(spec)
    assert config_hash(_named_as(recipe, NOMEM)) == config_hash(spec)


def _repo(tmp: Path) -> Path:
    repo = fake_repo(tmp)
    built = repo / "data" / "tinystories_w1024"
    built.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / "tinystories_w1024.manifest.json", built / "manifest.json")
    return repo


def _arm_ids(tmp: Path, name: str, config: Path, arm: list[str]) -> Any:
    """The arm ts27_nomem_w160l3_top0_w1024 of spec fast_readers (seeds 0 and 1) on ``config``:
    the spec's base, families block and protocol, the arm's overrides ``arm``."""
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
        "arms": [
            {
                "name": "ts27_nomem_w160l3_top0_w1024",
                "family": "hsn_t_nomem",
                "overrides": [*arm, *CARD[4:]],
            }
        ],
    }
    path = write_yaml(tmp / f"{name}.yaml", raw)
    arms = expand(load_spec(path)[0], repo=_repo(tmp))
    assert {a.data_hash for a in arms} == {DATA_HASH}
    assert {a.group_id for a in arms} == {RECORDED["ts27_nomem_w160l3_top0_w1024/s0"]}
    return {a.label: a.arm_id for a in arms}


def test_recipe_arms_are_the_recorded_runs(tmp_path: Path) -> None:
    """A spec on the recipe and the runs' spec on ``hsn_t_nomem.yaml`` expand to the same arm ids
    (the base file's path is not part of an arm's identity), the ones the runs recorded."""
    from_nomem = _arm_ids(tmp_path, "from_nomem", NOMEM, list(SPEC_ARM))
    from_recipe = _arm_ids(tmp_path, "from_recipe", RECIPE, [])
    assert from_recipe == from_nomem == RECORDED
