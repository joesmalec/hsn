"""The carry recipes of HSN-T with its memory (the user's decision of 18 Nov 2026: the reader
carry, ``model.reader_carry``, ``docs/amendments/reader_carry_memory.md``, adopted as recipes of
their own): ``configs/experiments/hsn_t_memory_carry.yaml`` (the per-byte memory) and
``configs/experiments/hsn_t_memory_words_carry.yaml`` (word events).

Each is its base recipe (``hsn_t_memory.yaml``, ``hsn_t_memory_words.yaml``; both unchanged)
with five keys changed and nothing else: the top0 reader (width 160, four blocks, rotary pairs
[20, 20, 20, 0], no span) and ``model.reader_carry=256``. Each loads and validates at the top0
reader's parameter count (the carry adds none); it resolves to its base with those five
overrides exactly (the same configuration hash once the run's name and notes are aligned, the
same arm ids), on the CPU profile and with the card's lines. A spec on it with the TinyStories
data lines and no arm override expands to the arm ids the measured runs recorded (specs
``text_carry_e1``, ``text_carry_e1_s1``, ``text_carry_e1_27k`` and ``text_carry_e1_27k_s1`` of
17 Nov 2026, run at 6d2bb88 on ``hsn_t_memory.yaml`` with the keys as arm overrides; the base
file's path is not part of an arm's identity). The specs live outside the repository
(``~/hsn-experiments/remote/specs``); what they pass is written out here: the card, data and
step lines and the protocol are those of ``test_hsn_t_memory_words_recipe._run_ids``, the arm
overrides are ``WORDS`` (word events only) and ``CARRY``. Without the carry the same specs
expand to the recorded ids of the runs each carry run is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hsn.config import REPO_ROOT, ModelConfig, RunConfig, config_hash, load_config
from hsn.experiments.spec import config_view, family_of
from hsn.model.hsn import reader_carried
from hsn.model.hsn_t import HSNT
from test_hsn_t_memory_recipe import CARD
from test_hsn_t_memory_words_recipe import (
    TINYSTORIES,
    TOP0,
    WORDS,
    Round,
    _named_as,
    _run_ids,
)

EXPERIMENTS = REPO_ROOT / "configs" / "experiments"
MEMORY = EXPERIMENTS / "hsn_t_memory.yaml"
#: the five keys a carry recipe changes on its base, as the arm overrides of the measured runs
#: (in their order): the top0 reader, whose span must be cleared, and the carry
CARRY = (*TOP0, "model.reader_carry=256")
#: the same five as configuration keys, and what a base recipe holds there
CARRY_KEYS: dict[str, tuple[Any, Any]] = {  # key: (the carry recipe, its base)
    "model.reader_width": (160, 64),
    "model.reader_layers": (4, 2),
    "model.reader_rotary_pairs": ([20, 20, 20, 0], None),
    "model.reader_span": (None, 32),
    "model.reader_carry": (256, 0),
}
HARDWARE = ((), CARD)
#: HSN-T with its memory and the top0 reader (the measured runs' param_count); main's form with
#: the 64 x 2 reader is 2,037,801 (tests/test_hsn_t_memory_recipe.py)
PARAMS, BASE_PARAMS = 3_298_025, 2_037_801


@dataclass(frozen=True)
class Recipe:
    """A carry recipe, the recipe it is built on, and what the measured arms passed on
    ``hsn_t_memory.yaml`` before the five keys."""

    path: Path
    base: Path
    words: tuple[str, ...]
    #: the measured TinyStories rounds and the arm ids their runs recorded (their result.json)
    rounds: tuple[Round, ...]


RECIPES = {
    "per_byte": Recipe(
        EXPERIMENTS / "hsn_t_memory_carry.yaml",
        MEMORY,
        (),
        (
            Round(  # spec text_carry_e1: 0.8272 dev bits per byte (no carry 0.8665)
                "ts_memT_top0_c256",
                9000,
                3000,
                (0,),
                {"ts_memT_top0_c256/s0": "4815baad3d694a25"},
            ),
            Round(  # spec text_carry_e1_27k: 0.7337 (no carry 0.7783)
                "ts_memT_top0_c256_27k",
                27000,
                9000,
                (0,),
                {"ts_memT_top0_c256_27k/s0": "2c02ed8cd1677e50"},
            ),
        ),
    ),
    "word_events": Recipe(
        EXPERIMENTS / "hsn_t_memory_words_carry.yaml",
        EXPERIMENTS / "hsn_t_memory_words.yaml",
        WORDS,
        (
            Round(  # specs text_carry_e1 (s0) and text_carry_e1_s1: 0.8321 / 0.8338
                "ts_memT_top0_wev_c256",
                9000,
                3000,
                (0, 1),
                {
                    "ts_memT_top0_wev_c256/s0": "d0a23b9c39b08f8e",
                    "ts_memT_top0_wev_c256/s1": "fd12960892ab00a6",
                },
            ),
            Round(  # specs text_carry_e1_27k (s0) and text_carry_e1_27k_s1: 0.7413 / 0.7381
                "ts_memT_top0_wev_c256_27k",
                27000,
                9000,
                (0, 1),
                {
                    "ts_memT_top0_wev_c256_27k/s0": "ba51973f9d0e9c09",
                    "ts_memT_top0_wev_c256_27k/s1": "b446636ec6a18a95",
                },
            ),
        ),
    ),
}
FORMS = sorted(RECIPES)
#: the measured references of those runs, by the carry arm's label: the same arm without the
#: carry, which is the base recipe with the top0 reader alone (recorded arm ids; the 27k
#: word-event reference has seed 0 only)
REFERENCES = {
    "ts_memT_top0_c256/s0": "81065bb8a8770ebd",  # ts_memT_top0_c0, 0.8665
    "ts_memT_top0_c256_27k/s0": "6f012092033241c2",  # ts27_memT_top0, 0.7783
    "ts_memT_top0_wev_c256/s0": "1888ced675259be1",  # ts_memT_top0_wev_c0, 0.8842
    "ts_memT_top0_wev_c256/s1": "9e1fb450cada613b",  # 0.8845
    "ts_memT_top0_wev_c256_27k/s0": "f0c22b32ea9bad19",  # ts27_memT_top0_wev_wexact_lr125, 0.7972
}


def _flat(value: Any, prefix: str = "") -> dict[str, Any]:
    """A configuration dump as dotted keys (a list is a value)."""
    if not isinstance(value, dict):
        return {prefix: value}
    flat: dict[str, Any] = {}
    for key, item in value.items():
        flat.update(_flat(item, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def _changed(a: RunConfig, b: RunConfig) -> dict[str, tuple[Any, Any]]:
    """Every key of the whole resolved configurations (defaults included) at which ``a`` and
    ``b`` differ, with the two values."""
    fa, fb = _flat(a.model_dump(mode="json")), _flat(b.model_dump(mode="json"))
    assert fa.keys() == fb.keys()
    return {key: (fa[key], fb[key]) for key in fa if fa[key] != fb[key]}


@pytest.mark.parametrize("form", FORMS)
def test_recipe_loads_with_the_carry(form: str) -> None:
    recipe = RECIPES[form]
    cfg, base = load_config(recipe.path), load_config(recipe.base)
    assert cfg.model_name == "hsn_t" and family_of(cfg) == "hsn_t"
    m = cfg.model
    assert m.memory and m.reader == "transformer" and m.reader_heads == 4
    assert (m.reader_width, m.reader_layers, m.reader_rotary_pairs) == (160, 4, [20, 20, 20, 0])
    assert m.reader_span is None and m.reader_carry == 256
    assert reader_carried(m) and not reader_carried(base.model)
    # everything else is the base's: the writes, the rate, C-read off
    assert (m.write_events, m.word_address, m.word_contexts) == (
        base.model.write_events,
        base.model.word_address,
        base.model.word_contexts,
    )
    assert m.write_events == ("word" if recipe.words else "byte")
    assert cfg.train.lr == base.train.lr == (1.25e-3 if recipe.words else 2.5e-3)
    assert m.cell_attention_heads == 0
    assert not cfg.hardware.graph_alias_index  # an override of the spec, as in the base
    # the code defaults stay where they were: no carry, the 64 x 2 reader
    d = ModelConfig()
    assert (d.reader_carry, d.reader_width, d.reader_layers) == (0, 64, 2)
    assert d.reader_rotary_pairs is None and d.reader_span is None
    # the top0 reader's parameter count: the carry adds none (a cache, no parameter)
    carried = HSNT.from_config(cfg).param_report()["total_trainable"]
    top0 = HSNT.from_config(load_config(recipe.base, list(TOP0))).param_report()
    assert carried == top0["total_trainable"] == PARAMS
    assert HSNT.from_config(base).param_report()["total_trainable"] == BASE_PARAMS


@pytest.mark.parametrize("hardware", HARDWARE, ids=["cpu", "card"])
@pytest.mark.parametrize("form", FORMS)
def test_recipe_is_its_base_with_the_five_keys(form: str, hardware: tuple[str, ...]) -> None:
    recipe = RECIPES[form]
    cfg = load_config(recipe.path, list(hardware))
    base = load_config(recipe.base, list(hardware))
    # the five keys, the run's name and its notes, and nothing else in the whole configuration
    changed = _changed(cfg, base)
    assert {k: v for k, v in changed.items() if k not in ("run_id", "notes")} == CARRY_KEYS
    assert {"run_id", "notes"} <= changed.keys()
    # so it is the base with the five overrides
    on_base = load_config(recipe.base, [*hardware, *CARRY])
    assert config_view(cfg) == config_view(on_base)  # what the arm id reads
    assert config_hash(_named_as(cfg, recipe.base)) == config_hash(on_base)
    assert config_hash(cfg) != config_hash(base)
    # and hsn_t_memory.yaml with what the measured arms passed
    as_run = load_config(MEMORY, [*hardware, *recipe.words, *CARRY])
    assert config_view(cfg) == config_view(as_run)
    assert config_hash(_named_as(cfg, MEMORY)) == config_hash(as_run)
    # the five set again change nothing
    assert config_hash(load_config(recipe.path, [*hardware, *CARRY])) == config_hash(cfg)


def test_word_recipe_is_the_per_byte_recipe_with_word_events() -> None:
    """The two carry recipes differ as their bases do: by the four word keys."""
    words = load_config(RECIPES["word_events"].path)
    per_byte = load_config(RECIPES["per_byte"].path, list(WORDS))
    assert config_view(words) == config_view(per_byte)
    assert config_hash(_named_as(words, RECIPES["per_byte"].path)) == config_hash(per_byte)


@pytest.mark.parametrize("steps", [0, 1], ids=["9k", "27k"])
@pytest.mark.parametrize("form", FORMS)
def test_recipe_arms_are_the_measured_runs(tmp_path: Path, form: str, steps: int) -> None:
    """A spec on the recipe with the TinyStories data lines and no arm override expands to the
    arm ids of the measured carry runs, and so does the spec that ran them (on
    ``hsn_t_memory.yaml``, the word keys and the five keys as arm overrides)."""
    recipe = RECIPES[form]
    round_ = recipe.rounds[steps]
    data = list(TINYSTORIES)
    as_run = _run_ids(tmp_path, "as_run", round_, MEMORY, data, [*recipe.words, *CARRY])
    from_recipe = _run_ids(tmp_path, "from_recipe", round_, recipe.path, data, [])
    assert from_recipe == as_run == round_.recorded
    # the carry is part of the identity: the base recipe with the top0 reader is another arm,
    # the run each carry run is measured against
    no_carry = _run_ids(tmp_path, "no_carry", round_, recipe.base, data, list(TOP0))
    assert no_carry.keys() == from_recipe.keys()
    assert not set(no_carry.values()) & set(from_recipe.values())
    assert all(REFERENCES.get(label, arm_id) == arm_id for label, arm_id in no_carry.items())
    assert REFERENCES.keys() & no_carry.keys()
