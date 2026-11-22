"""The word-event carry recipe with the paper's balance loss
(``configs/experiments/hsn_t_memory_words_carry_bal.yaml``, the user's decision of 21 Nov 2026:
the balance loss of Appendix A.3 adopted for the word-event recipe, as a recipe of its own).

It is ``hsn_t_memory_words_carry.yaml`` (unchanged) with two keys changed,
``train.aux_enabled=[balance]`` and ``train.lambda_b=0.02``, and nothing else: the same
configuration and so the same configuration hash once the run's name and notes are aligned, on
the CPU profile and with the card's lines. It loads, validates and builds on the CPU at the
parent's parameter count (the balance loss adds none), and the objective it trains has the
balance term.

A spec on it with the code-native data lines expands to the arm id the measured run recorded
(spec ``fields_27k_fix``, run at 3c85c2a on ``hsn_t_memory_words_carry.yaml`` with the two keys
as arm overrides: 1.172 dev bits per byte, 77.38 % next-byte accuracy, no route-guard event).
The spec lives outside the repository (``~/hsn-experiments/remote/specs``); what it passes is
written out here.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import REPO_ROOT, config_hash, load_config
from hsn.experiments.spec import config_view, expand, family_of, load_spec
from hsn.losses import assemble_terms
from hsn.model.hsn import reader_carried
from hsn.model.hsn_t import HSNT
from p1_11_support import fake_repo, write_yaml
from test_hsn_t_memory_carry_recipes import PARAMS, SPEED, _changed
from test_hsn_t_memory_recipe import CARD
from test_hsn_t_memory_words_recipe import _named_as

EXPERIMENTS = REPO_ROOT / "configs" / "experiments"
DATA = REPO_ROOT / "configs" / "data"
RECIPE = EXPERIMENTS / "hsn_t_memory_words_carry_bal.yaml"
PARENT = EXPERIMENTS / "hsn_t_memory_words_carry.yaml"
#: the two keys this recipe changes on its parent, as arm overrides (the measured run's order)
BALANCE = ("train.aux_enabled=[balance]", "train.lambda_b=0.02")
#: the same two as configuration keys: (this recipe, its parent)
BALANCE_KEYS: dict[str, tuple[Any, Any]] = {
    "train.aux_enabled": (["balance"], []),
    "train.lambda_b": (0.02, 0.0),
}
HARDWARE = ((), CARD)
#: the measured run with the balance loss (spec fields_27k_fix, code native, 27k, seed 0)
MEASURED = {"f27_code_native_wevbal/s0": "e5e8d3bbb89bc354"}
CODE_NATIVE = (
    "data.dataset=fields_code_native",
    "data.manifest=data/fields_code_native/manifest.json",
    "data.tasks=[fields_code_native]",
    "data.variants=[out_of_range]",
    "model.word_contexts=[0,6,15]",
)


def test_recipe_loads_with_the_balance_loss() -> None:
    cfg, parent = load_config(RECIPE), load_config(PARENT)
    assert cfg.model_name == "hsn_t" and family_of(cfg) == "hsn_t"
    assert cfg.train.aux_enabled == ["balance"] and cfg.train.lambda_b == 0.02
    assert parent.train.aux_enabled == [] and parent.train.lambda_b == 0.0
    # the parent's reader, carry and word events, and the rate of word events
    m = cfg.model
    assert (m.reader_width, m.reader_layers, m.reader_rotary_pairs) == (160, 4, [20, 20, 20, 0])
    assert m.reader_span is None and m.reader_carry == 256 and reader_carried(m)
    assert (m.write_events, m.word_address, m.word_contexts) == ("word", "exact", [0, 6])
    assert cfg.train.lr == 1.25e-3
    # the CUDA arm's lines stay overrides of the spec, as in the parent
    assert (cfg.hardware.message_gradients, cfg.hardware.round_products) == ("autocast", "separate")


@pytest.mark.parametrize("hardware", HARDWARE, ids=["cpu", "card"])
def test_recipe_is_its_parent_with_the_balance_loss(hardware: tuple[str, ...]) -> None:
    cfg = load_config(RECIPE, list(hardware))
    parent = load_config(PARENT, list(hardware))
    # the two keys, the run's name and its notes, and nothing else in the whole configuration
    changed = _changed(cfg, parent)
    assert {k: v for k, v in changed.items() if k not in ("run_id", "notes")} == BALANCE_KEYS
    assert {"run_id", "notes"} <= changed.keys()
    # so it is the parent with those overrides
    on_parent = load_config(PARENT, [*hardware, *BALANCE])
    assert config_view(cfg) == config_view(on_parent)  # what the arm id reads
    assert config_hash(_named_as(cfg, PARENT)) == config_hash(on_parent)
    assert config_hash(cfg) != config_hash(parent)
    # the keys set again change nothing
    assert config_hash(load_config(RECIPE, [*hardware, *BALANCE])) == config_hash(cfg)


def test_recipe_takes_the_speed_keys_on_the_card() -> None:
    card = load_config(RECIPE, list(CARD))
    fast = load_config(RECIPE, [*CARD, *SPEED])
    assert _changed(fast, card) == {
        "hardware.message_gradients": ("state", "autocast"),
        "hardware.round_products": ("merged", "separate"),
    }


def test_recipe_builds_on_the_cpu_with_the_balance_term() -> None:
    """The model builds on the CPU at the parent's parameter count, and the objective names the
    balance term with its weight (the parent's has none)."""
    cfg, parent = load_config(RECIPE), load_config(PARENT)
    model = HSNT.from_config(cfg)
    assert model.param_report()["total_trainable"] == PARAMS
    assert HSNT.from_config(parent).param_report()["total_trainable"] == PARAMS
    out = {
        "logits": torch.zeros(1, 2, 256),
        **{key: torch.zeros(1, 2, cfg.model.d) for key in ("z_hat", "z_target", "z")},
    }
    batch = {
        "targets": torch.zeros(1, 2, dtype=torch.int64),
        "loss_mask": torch.ones(1, 2, dtype=torch.bool),
        "doc": torch.zeros(1, 2, dtype=torch.int64),
    }
    raw = torch.tensor(1.5)
    terms = assemble_terms(out, batch, cfg, extras={"balance": raw})
    assert list(terms) == ["task", "pred", "balance"]
    assert float(terms["balance"]) == pytest.approx(0.02 * 1.5)
    assert "balance" not in assemble_terms(out, batch, parent)
    with pytest.raises(ValueError, match="not enabled"):  # the parent does not train the term
        assemble_terms(out, batch, parent, extras={"balance": raw})


def test_recipe_arm_is_the_measured_run(tmp_path: Path) -> None:
    """A spec on the recipe with the code-native data lines expands to the arm id of the run
    with the balance loss (spec fields_27k_fix, on the parent with the two keys as arm
    overrides), and so does the spec that ran it; the parent alone is another arm."""
    repo = fake_repo(tmp_path)
    built = repo / "data" / "fields_code_native"
    built.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / "fields_code_native.manifest.json", built / "manifest.json")

    def arms(name: str, config: Path, arm: list[str]) -> dict[str, str]:
        raw = {
            "name": name,
            "timing": False,
            "base": {
                "config": "configs/experiments/hsn_t_nomem.yaml",
                "overrides": [
                    "profile=rtx3060",
                    "train.batch=32",
                    "train.micro_batch=32",
                    "train.steps=27000",
                    "train.ckpt_every=9000",
                    "train.eval_every=9000",
                ],
            },
            "families": {
                "hsn_t": {
                    "config": "configs/experiments/hsn_t_memory_carry.yaml",
                    "overrides": [
                        "hardware.compile_mode=reduce-overhead",
                        "hardware.graph_alias_index=true",
                        *SPEED,
                    ],
                }
            },
            "protocol": {
                "name": "train_eval",
                "steps": 27000,
                "diagnostics": False,
                "diag_every": 250,
                "diag_docs": 32,
                "eval_batch": 32,
                "checkpoint": "retained",
                "keep_every": 9000,
                "keep_last": 1,
            },
            "seeds": [0],
            "arms": [
                {
                    "name": "f27_code_native_wevbal",
                    "family": "hsn_t",
                    "config": str(config.relative_to(REPO_ROOT)),
                    "overrides": [*CODE_NATIVE, *arm],
                }
            ],
        }
        path = write_yaml(tmp_path / f"{name}.yaml", raw)
        return {a.label: a.arm_id for a in expand(load_spec(path)[0], repo=repo)}

    as_run = arms("as_run", PARENT, list(BALANCE))
    from_recipe = arms("from_recipe", RECIPE, [])
    assert from_recipe == as_run == MEASURED
    assert not set(arms("parent", PARENT, []).values()) & set(MEASURED.values())
