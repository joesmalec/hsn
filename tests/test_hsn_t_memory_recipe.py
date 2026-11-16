"""The recipe of HSN-T with its memory (``configs/experiments/hsn_t_memory.yaml``, the user's
decision of 16 Nov 2026): main's HSN-T form, which the memory specs of 15-16 Nov 2026 pass as
overrides on top of ``hsn_recipe_a6.yaml``, with the attention read of the cells on
(``model.cell_attention_heads: 4``, ``docs/amendments/cell_attention_read.md``).

The recipe loads and validates; with the read set back to 0 it resolves to ``hsn_recipe_a6.yaml``
with those overrides exactly (the same configuration hash once the run's name and notes are
aligned, the same arm ids), so the recipe is "main's form + C-read"; and the code default of the
key stays 0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hsn.config import REPO_ROOT, ModelConfig, RunConfig, config_hash, load_config
from hsn.experiments.spec import config_view, expand, family_of, load_spec
from hsn.model.hsn_t import HSNT
from p1_11_support import fake_repo, write_yaml

EXPERIMENTS = REPO_ROOT / "configs" / "experiments"
RECIPE = EXPERIMENTS / "hsn_t_memory.yaml"
RECIPE_A6 = EXPERIMENTS / "hsn_recipe_a6.yaml"
#: the overrides of main's HSN-T memory specs (mem_read.yaml, mem_read_s1.yaml: base.overrides)
#: without the data, steps, profile and compile lines
MAIN_FORM = (
    "model_name=hsn_t",
    "model.k_select=12",
    "model.reader=transformer",
    "model.reader_span=32",
    "model.address_gate=true",
    "model.address_gate_input=summary",
    "model.n_cells=64",
    "train.feedback_detach=[summary, eps, router]",
    "model.scorer_position=wrap",
    "model.address_ngram=2",
    "model.address_history=ngram",
    "train.unit_credit=executed",
    "train.soft_until_step=0",
)
#: the hardware lines of those specs; graph_alias_index cannot stand in a configuration file
#: without replacing the whole profile, so the recipe leaves it to the spec as well
CARD = (
    "profile=rtx3060",
    "hardware.compile_mode=reduce-overhead",
    "hardware.graph_alias_index=true",
)
OFF = "model.cell_attention_heads=0"
ON = "model.cell_attention_heads=4"
HARDWARE = ((), CARD)


def _named_as_a6(cfg: RunConfig) -> RunConfig:
    """``cfg`` with ``hsn_recipe_a6``'s run name and notes, the two keys the configuration hash
    reads that are not part of the run (the arm id leaves them out)."""
    a6 = load_config(RECIPE_A6)
    return cfg.model_copy(update={"run_id": a6.run_id, "notes": a6.notes})


def test_recipe_loads_with_the_read_on() -> None:
    cfg = load_config(RECIPE)
    assert cfg.model_name == "hsn_t" and family_of(cfg) == "hsn_t"
    assert cfg.model.cell_attention_heads == 4
    assert cfg.model.memory and cfg.model.reader == "transformer" and cfg.model.reader_span == 32
    assert cfg.model.k_select == 12 and cfg.model.n_cells == 64
    assert cfg.train.feedback_detach == ["summary", "eps", "router"]
    assert cfg.train.unit_credit == "executed" and cfg.train.soft_until_step == 0
    assert not cfg.hardware.graph_alias_index  # an override of the spec (module docstring)
    # main's HSN-T model is 2,037,801 trainable parameters; the read adds 53,504 (its own
    # 20,736 and d * hidden = 32,768 decoder columns)
    on = HSNT.from_config(cfg).param_report()
    off = HSNT.from_config(load_config(RECIPE, [OFF])).param_report()
    assert (on["total_trainable"], off["total_trainable"]) == (2_091_305, 2_037_801)
    assert on["cell_read"] == 20_736 and "cell_read" not in off
    # the code default stays off
    assert ModelConfig().cell_attention_heads == 0
    assert load_config(RECIPE_A6).model.cell_attention_heads == 0


@pytest.mark.parametrize("hardware", HARDWARE, ids=["cpu", "card"])
def test_recipe_without_the_read_is_mains_form(hardware: tuple[str, ...]) -> None:
    recipe = load_config(RECIPE, [*hardware, OFF])
    main = load_config(RECIPE_A6, [*hardware, *MAIN_FORM])
    assert config_view(recipe) == config_view(main)  # what the arm id reads
    assert config_hash(_named_as_a6(recipe)) == config_hash(main)
    assert recipe.run_id != main.run_id  # the only differences: the name and the notes
    # with the read on it is main's form with the key set, and nothing else
    on = load_config(RECIPE, list(hardware))
    main_on = load_config(RECIPE_A6, [*hardware, *MAIN_FORM, ON])
    assert config_view(on) == config_view(main_on)
    assert config_hash(_named_as_a6(on)) == config_hash(main_on)
    assert config_hash(on) != config_hash(recipe)


def _arm_ids(tmp: Path, name: str, base: dict[str, Any], arms: list[dict[str, Any]]) -> list[str]:
    raw = {
        "name": name,
        "base": base,
        "protocol": {"name": "train_eval", "steps": 9000},
        "seeds": [0, 1],
        "arms": arms,
    }
    path = write_yaml(tmp / f"{name}.yaml", raw)
    return [arm.arm_id for arm in expand(load_spec(path)[0], repo=fake_repo(tmp))]


def test_recipe_arms_share_the_arm_ids_of_mains_form(tmp_path: Path) -> None:
    """A spec on the recipe and a spec on ``hsn_recipe_a6`` with main's overrides expand to the
    same arms: the read off and on, two seeds."""
    from_a6 = _arm_ids(
        tmp_path,
        "from_a6",
        {"config": str(RECIPE_A6), "overrides": [*CARD, *MAIN_FORM]},
        [
            {"name": "rd32", "family": "hsn_t", "overrides": []},
            {"name": "rd32_cattn4", "family": "hsn_t", "overrides": [ON]},
        ],
    )
    from_recipe = _arm_ids(
        tmp_path,
        "from_recipe",
        {"config": str(RECIPE), "overrides": list(CARD)},
        [
            {"name": "rd32", "family": "hsn_t", "overrides": [OFF]},
            {"name": "rd32_cattn4", "family": "hsn_t", "overrides": []},
        ],
    )
    assert len(from_a6) == 4 and len(set(from_a6)) == 4
    assert from_recipe == from_a6
