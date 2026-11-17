"""The recipe of HSN-T with its memory writing at word ends
(``configs/experiments/hsn_t_memory_words.yaml``, the user's decision of 17 Nov 2026: learning
rate 1.25e-3 for HSN-T runs with word events, ``model.write_events=word``, and only for those).

The recipe loads and validates at the memory recipe's parameter count (the word address adds
buffers, no parameters); it resolves to ``hsn_t_memory.yaml`` with word events, the exact word
address over contexts [0, 6] and lr 1.25e-3 exactly (the same configuration hash once the run's
name and notes are aligned, the same arm ids), on the CPU profile and with the card's lines; a
task's text contexts set on top are those overrides with the key changed, and the memory recipe
keeps 2.5e-3. A spec on it with the top0 reader alone expands to the arm ids the TinyStories runs
at 1.25e-3 recorded (specs ``wev_top0_lr``, ``wev_top0_lr2`` and ``wev27k_top0_lr125`` of 16-17
Nov 2026, run on main at 5c54832).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hsn.config import REPO_ROOT, ModelConfig, RunConfig, config_hash, load_config
from hsn.experiments.spec import config_view, expand, family_of, load_spec
from hsn.model.hsn_t import HSNT
from p1_11_support import fake_repo, write_yaml
from test_hsn_t_memory_recipe import CARD, MAIN_FORM

EXPERIMENTS = REPO_ROOT / "configs" / "experiments"
DATA = REPO_ROOT / "configs" / "data"
RECIPE = EXPERIMENTS / "hsn_t_memory_words.yaml"
MEMORY = EXPERIMENTS / "hsn_t_memory.yaml"
RECIPE_A6 = EXPERIMENTS / "hsn_recipe_a6.yaml"
#: the four keys this recipe changes on hsn_t_memory.yaml (the arm overrides of the TinyStories
#: runs at lr 1.25e-3, in their order)
WORDS = (
    "model.write_events=word",
    "model.word_address=exact",
    "model.word_contexts=[0, 6]",
    "train.lr=0.00125",
)
HARDWARE = ((), CARD)
#: main's HSN-T memory model, 2,037,801 trainable parameters (tests/test_hsn_t_memory_recipe.py)
PARAMS = 2_037_801
#: the task specs' text contexts, supersets of the recipe's [0, 6] (the recipe's header)
TASK_CONTEXTS = {
    "wordkeys": "model.word_contexts=[0, 6, 7]",
    "wordhold": "model.word_contexts=[0, 6, 7, 8]",
}
#: the top0 reader of the TinyStories runs; on the recipe the 32-byte span must be cleared
TOP0 = (
    "model.reader_width=160",
    "model.reader_layers=4",
    "model.reader_rotary_pairs=[20,20,20,0]",
    "model.reader_span=null",
)
#: the data lines of those specs
TINYSTORIES = (
    "data.dataset=tinystories_v1",
    "data.manifest=data/tinystories_v1/manifest.json",
    "data.tasks=[tinystories]",
    "data.variants=[out_of_range]",
)
#: what the specs of those runs passed on hsn_recipe_a6.yaml besides the card and step lines:
#: main's HSN-T overrides with the top0 reader in place of the 32-byte span, and TinyStories
RUN_BASE = (*(o for o in MAIN_FORM if o != "model.reader_span=32"), *TOP0[:3], *TINYSTORIES)


@dataclass(frozen=True)
class Round:
    """One spec of the TinyStories runs at lr 1.25e-3 and the arm ids its runs recorded
    (their result.json)."""

    arm: str
    steps: int
    ckpt_every: int
    seeds: tuple[int, ...]
    recorded: dict[str, str]


ROUNDS = (
    Round(  # specs wev_top0_lr (s0) and wev_top0_lr2 (s1): 0.8842 / 0.8808 dev bits per byte
        "ts_memT_top0_wev_wexact_lr125",
        9000,
        3000,
        (0, 1),
        {
            "ts_memT_top0_wev_wexact_lr125/s0": "1888ced675259be1",
            "ts_memT_top0_wev_wexact_lr125/s1": "9e1fb450cada613b",
        },
    ),
    Round(  # spec wev27k_top0_lr125: 0.7972
        "ts27_memT_top0_wev_wexact_lr125",
        27000,
        9000,
        (0,),
        {"ts27_memT_top0_wev_wexact_lr125/s0": "f0c22b32ea9bad19"},
    ),
)


def _named_as(cfg: RunConfig, path: Path) -> RunConfig:
    """``cfg`` with the run name and notes of the recipe at ``path``, the two keys the
    configuration hash reads that are not part of the run (the arm id leaves them out)."""
    other = load_config(path)
    return cfg.model_copy(update={"run_id": other.run_id, "notes": other.notes})


def test_recipe_loads_with_word_events() -> None:
    cfg = load_config(RECIPE)
    assert cfg.model_name == "hsn_t" and family_of(cfg) == "hsn_t"
    m = cfg.model
    assert (m.write_events, m.word_address) == ("word", "exact")
    # TinyStories' text and its padding, as every run behind the decision; not the code default
    assert m.word_contexts == [0, 6] and ModelConfig().word_contexts == [6]
    assert m.memory and m.reader == "transformer" and m.reader_span == 32
    assert m.cell_attention_heads == 0
    assert cfg.train.lr == 1.25e-3
    assert load_config(MEMORY).train.lr == 2.5e-3  # the per-byte memory keeps its rate
    assert not cfg.hardware.graph_alias_index  # an override of the spec, as in hsn_t_memory.yaml
    # the word address adds buffers (its fixed tables), no parameters
    words = HSNT.from_config(cfg).param_report()["total_trainable"]
    memory = HSNT.from_config(load_config(MEMORY)).param_report()["total_trainable"]
    assert words == memory == PARAMS


@pytest.mark.parametrize("hardware", HARDWARE, ids=["cpu", "card"])
def test_recipe_is_the_memory_recipe_with_word_events(hardware: tuple[str, ...]) -> None:
    recipe = load_config(RECIPE, list(hardware))
    memory = load_config(MEMORY, [*hardware, *WORDS])
    assert config_view(recipe) == config_view(memory)  # what the arm id reads
    assert config_hash(_named_as(recipe, MEMORY)) == config_hash(memory)
    assert recipe.run_id != memory.run_id  # the only differences: the name and the notes
    # and so hsn_recipe_a6 with main's HSN-T overrides and these four
    a6 = load_config(RECIPE_A6, [*hardware, *MAIN_FORM, *WORDS])
    assert config_view(recipe) == config_view(a6)
    assert config_hash(_named_as(recipe, RECIPE_A6)) == config_hash(a6)
    # the explicit [0, 6] set again changes nothing
    again = load_config(RECIPE, [*hardware, WORDS[2]])
    assert config_hash(again) == config_hash(recipe)
    # a task's text contexts on top: the memory recipe with the four keys, that one changed
    for contexts in TASK_CONTEXTS.values():
        on = load_config(RECIPE, [*hardware, contexts])
        memory_on = load_config(MEMORY, [*hardware, *WORDS, contexts])
        assert config_hash(_named_as(on, MEMORY)) == config_hash(memory_on)
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


def test_recipe_arms_share_the_arm_ids_of_the_memory_recipe(tmp_path: Path) -> None:
    """A spec on the recipe and a spec on ``hsn_t_memory.yaml`` with the four overrides expand
    to the same arms: the recipe's text contexts and word keys', two seeds."""
    wordkeys = TASK_CONTEXTS["wordkeys"]
    from_memory = _arm_ids(
        tmp_path,
        "from_memory",
        {"config": str(MEMORY), "overrides": [*CARD, *WORDS]},
        [
            {"name": "wev", "family": "hsn_t", "overrides": []},
            {"name": "wev_wc067", "family": "hsn_t", "overrides": [wordkeys]},
        ],
    )
    from_recipe = _arm_ids(
        tmp_path,
        "from_recipe",
        {"config": str(RECIPE), "overrides": list(CARD)},
        [
            {"name": "wev", "family": "hsn_t", "overrides": []},
            {"name": "wev_wc067", "family": "hsn_t", "overrides": [wordkeys]},
        ],
    )
    assert len(from_memory) == 4 and len(set(from_memory)) == 4
    assert from_recipe == from_memory


def _tinystories_repo(tmp: Path) -> Path:
    repo = fake_repo(tmp)
    built = repo / "data" / "tinystories_v1"
    built.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / "tinystories_v1.manifest.json", built / "manifest.json")
    return repo


def _run_ids(
    tmp: Path, name: str, round_: Round, config: Path, base: list[str], arm: list[str]
) -> dict[str, str]:
    """The arms of one TinyStories round at lr 1.25e-3 on ``config``, with the protocol, the
    card and the step lines of its spec."""
    raw = {
        "name": name,
        "timing": False,
        "base": {
            "config": str(config),
            "overrides": [
                *CARD,
                *base,
                f"train.steps={round_.steps}",
                f"train.ckpt_every={round_.ckpt_every}",
                "train.eval_every=1500",
            ],
        },
        "protocol": {
            "name": "train_eval",
            "steps": round_.steps,
            "diagnostics": False,
            "diag_every": 250,
            "diag_docs": 32,
            "eval_batch": 128,
            "checkpoint": "retained",
            "keep_every": round_.steps,
            "keep_last": 1,
        },
        "seeds": [0],
        "arms": [
            {"name": round_.arm, "family": "hsn_t", "overrides": arm, "seeds": list(round_.seeds)}
        ],
    }
    path = write_yaml(tmp / f"{name}.yaml", raw)
    return {a.label: a.arm_id for a in expand(load_spec(path)[0], repo=_tinystories_repo(tmp))}


@pytest.mark.parametrize("round_", ROUNDS, ids=["9k", "27k"])
def test_recipe_arms_are_the_tinystories_runs(tmp_path: Path, round_: Round) -> None:
    """A spec on the recipe with the top0 reader alone (no word key) expands to the arm ids of
    the runs behind the decision, and so do the specs that ran them (on ``hsn_recipe_a6.yaml``;
    the base file's path is not part of an arm's identity)."""
    as_run = _run_ids(tmp_path, "as_run", round_, RECIPE_A6, list(RUN_BASE), list(WORDS))
    from_recipe = _run_ids(tmp_path, "from_recipe", round_, RECIPE, list(TINYSTORIES), list(TOP0))
    assert from_recipe == as_run == round_.recorded
