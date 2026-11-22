"""The word-event carry recipe with the balance loss and the reader's unrotated top block
normalised (``configs/experiments/hsn_t_memory_words_carry_bal_qk.yaml``, the user's decision
of 22 Nov 2026).

It is ``hsn_t_memory_words_carry_bal.yaml`` (unchanged) with one key changed,
``model.reader_qk_norm=[false,false,false,true]`` (``docs/amendments/reader_qk_norm.md``, the
opt-in key of PR #109), and nothing else: the same configuration and so the same configuration
hash once the run's name and notes are aligned, on the CPU profile and with the card's lines.
It loads, validates and builds on the CPU with the two norm weights of the top block (80
parameters more than its parent), and only the top block has them. The CUDA arm's lines take
the two speed keys and ``hardware.event_kernel=fused`` (``docs/wev_fused_event.md``) inside
the fused block's scope; the CPU profile refuses the fused block.

A spec on it with the code-native data lines expands to the arm id the measured run recorded
(spec ``fields_27k_qk``, run at 9aeb660 on ``hsn_t_memory_words_carry.yaml`` with the balance
keys, the top0 rotary pairs and the key as arm overrides: 1.023 dev bits per byte, 80.00 %
next-byte accuracy), and with ``hardware.event_kernel=fused`` added to the arm the id of the
fused run (spec ``fields_27k_fused``, run at f582f07: 1.020, 80.03 %). The specs live outside
the repository (``~/hsn-experiments/remote/specs``); what they pass is written out here.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import REPO_ROOT, config_hash, event_kernel_scope_error, load_config
from hsn.experiments.spec import config_view, expand, family_of, load_spec
from hsn.model.hsn import reader_carried
from hsn.model.hsn_t import HSNT
from hsn.train.memory_plan import reader_param_count
from p1_11_support import fake_repo, write_yaml
from test_hsn_t_memory_carry_recipes import PARAMS, SPEED, _changed
from test_hsn_t_memory_recipe import CARD
from test_hsn_t_memory_words_carry_bal_recipe import BALANCE, CODE_NATIVE
from test_hsn_t_memory_words_recipe import _named_as

EXPERIMENTS = REPO_ROOT / "configs" / "experiments"
DATA = REPO_ROOT / "configs" / "data"
RECIPE = EXPERIMENTS / "hsn_t_memory_words_carry_bal_qk.yaml"
PARENT = EXPERIMENTS / "hsn_t_memory_words_carry_bal.yaml"
#: the recipe the measured arms set their overrides on (the parent's parent)
WORDS_CARRY = EXPERIMENTS / "hsn_t_memory_words_carry.yaml"
#: the one key this recipe changes on its parent, as an arm override
QK = "model.reader_qk_norm=[false,false,false,true]"
#: the same key as a configuration key: (this recipe, its parent)
QK_KEYS: dict[str, tuple[Any, Any]] = {"model.reader_qk_norm": ([False, False, False, True], False)}
#: what the measured arms passed on hsn_t_memory_words_carry.yaml after the data lines
AS_RUN = (*BALANCE, "model.reader_rotary_pairs=[20,20,20,0]", QK)
#: the fused word-event block, an override of the CUDA arm (PRs #110/#111)
FUSED = "hardware.event_kernel=fused"
HARDWARE = ((), CARD)
#: the top block's two norm weights, one per query and key head channel (160 / 4 heads)
HEAD = 40
#: the measured runs, code native, 27k, seed 0: spec fields_27k_qk, and spec fields_27k_fused
#: (the same arm with the fused block)
MEASURED = {"f27_code_native_wevbal_top0qk/s0": "0dba6e332a02eaed"}
MEASURED_FUSED = {"f27_code_native_wevbal_top0qk/s0": "eaaf5c4945b8ccf1"}


def test_recipe_loads_with_the_top_block_normalised() -> None:
    cfg, parent = load_config(RECIPE), load_config(PARENT)
    assert cfg.model_name == "hsn_t" and family_of(cfg) == "hsn_t"
    assert cfg.model.reader_qk_norm == [False, False, False, True]
    assert parent.model.reader_qk_norm is False
    # the parent's reader (the key is on its one unrotated block), carry, word events and loss
    m = cfg.model
    assert (m.reader_width, m.reader_layers, m.reader_rotary_pairs) == (160, 4, [20, 20, 20, 0])
    assert m.reader_width // m.reader_heads == HEAD
    assert m.reader_span is None and m.reader_carry == 256 and reader_carried(m)
    assert (m.write_events, m.word_address, m.word_contexts) == ("word", "exact", [0, 6])
    assert cfg.train.aux_enabled == ["balance"] and cfg.train.lambda_b == 0.02
    assert cfg.train.lr == 1.25e-3
    # the CUDA arm's lines stay overrides of the spec, as in the parent
    assert (cfg.hardware.message_gradients, cfg.hardware.round_products) == ("autocast", "separate")
    assert cfg.hardware.event_kernel == "compiled"


@pytest.mark.parametrize("hardware", HARDWARE, ids=["cpu", "card"])
def test_recipe_is_its_parent_with_the_key(hardware: tuple[str, ...]) -> None:
    cfg = load_config(RECIPE, list(hardware))
    parent = load_config(PARENT, list(hardware))
    # the one key, the run's name and its notes, and nothing else in the whole configuration
    changed = _changed(cfg, parent)
    assert {k: v for k, v in changed.items() if k not in ("run_id", "notes")} == QK_KEYS
    assert {"run_id", "notes"} <= changed.keys()
    # so it is the parent with that override
    on_parent = load_config(PARENT, [*hardware, QK])
    assert config_view(cfg) == config_view(on_parent)  # what the arm id reads
    assert config_hash(_named_as(cfg, PARENT)) == config_hash(on_parent)
    assert config_hash(cfg) != config_hash(parent)
    # the key set again changes nothing
    assert config_hash(load_config(RECIPE, [*hardware, QK])) == config_hash(cfg)


def test_recipe_takes_the_speed_keys_and_the_fused_block_on_the_card() -> None:
    """The CUDA arm's lines of the header: the two speed keys and the fused word-event block
    change those three keys only, and the block's scope takes the recipe; the CPU profile of
    the file refuses the block."""
    card = load_config(RECIPE, list(CARD))
    fast = load_config(RECIPE, [*CARD, *SPEED, FUSED])
    assert _changed(fast, card) == {
        "hardware.message_gradients": ("state", "autocast"),
        "hardware.round_products": ("merged", "separate"),
        "hardware.event_kernel": ("fused", "compiled"),
    }
    assert event_kernel_scope_error(fast) is None
    with pytest.raises(ValueError, match="CUDA only"):
        load_config(RECIPE, [FUSED])


def test_recipe_builds_on_the_cpu_with_the_top_block_normalised() -> None:
    """The model builds on the CPU with the parent's parameters and the top block's two norm
    weights (``2 x 40``, at 1); the three rotated blocks have none."""
    cfg, parent = load_config(RECIPE), load_config(PARENT)
    model = HSNT.from_config(cfg)
    assert model.param_report()["total_trainable"] == PARAMS + 2 * HEAD
    assert HSNT.from_config(parent).param_report()["total_trainable"] == PARAMS
    assert model.param_report()["reader"] == reader_param_count(cfg.model)
    assert reader_param_count(cfg.model) - reader_param_count(parent.model) == 2 * HEAD
    reader = model.reader
    assert reader.qk_norm == [False, False, False, True]
    *rotated, top = reader.blocks
    assert all(block.q_norm is None and block.k_norm is None for block in rotated)
    for norm in (top.q_norm, top.k_norm):
        assert isinstance(norm, torch.nn.RMSNorm) and norm.normalized_shape == (HEAD,)
        assert torch.equal(norm.weight, torch.ones(HEAD))
    named = sorted(
        n for n, _ in model.named_parameters() if n.endswith(("q_norm.weight", "k_norm.weight"))
    )
    assert [n.split("blocks.", 1)[1] for n in named] == ["3.k_norm.weight", "3.q_norm.weight"]


def _arms(tmp_path: Path, name: str, config: Path, arm: list[str]) -> dict[str, str]:
    """The arm ids of a spec with ``fields_27k_qk``'s base and family lines and one code-native
    arm on ``config`` with ``arm`` after the data lines."""
    repo = fake_repo(tmp_path)
    built = repo / "data" / "fields_code_native"
    built.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / "fields_code_native.manifest.json", built / "manifest.json")
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
                "name": "f27_code_native_wevbal_top0qk",
                "family": "hsn_t",
                "config": str(config.relative_to(REPO_ROOT)),
                "overrides": [*CODE_NATIVE, *arm],
            }
        ],
    }
    path = write_yaml(tmp_path / f"{name}.yaml", raw)
    return {a.label: a.arm_id for a in expand(load_spec(path)[0], repo=repo)}


def test_recipe_arm_is_the_measured_run(tmp_path: Path) -> None:
    """A spec on the recipe with the code-native data lines expands to the arm id of the run
    with the key (spec fields_27k_qk, on hsn_t_memory_words_carry.yaml with the balance keys,
    the top0 rotary pairs and the key as arm overrides), and so does the spec that ran it; with
    the fused block in the arm, the fused run's (spec fields_27k_fused). The parent alone is
    another arm."""
    as_run = _arms(tmp_path, "as_run", WORDS_CARRY, list(AS_RUN))
    from_recipe = _arms(tmp_path, "from_recipe", RECIPE, [])
    assert from_recipe == as_run == MEASURED
    fused_as_run = _arms(tmp_path, "fused_as_run", WORDS_CARRY, [*AS_RUN, FUSED])
    fused = _arms(tmp_path, "fused", RECIPE, [FUSED])
    assert fused == fused_as_run == MEASURED_FUSED
    parent = _arms(tmp_path, "parent", PARENT, [])
    assert not set(parent.values()) & {*MEASURED.values(), *MEASURED_FUSED.values()}
