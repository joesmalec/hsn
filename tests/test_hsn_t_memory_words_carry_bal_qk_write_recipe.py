"""The QK-norm recipe with the keyed word cells written by the HSN, three windows a gradient
segment and, in its CUDA arm, the carried reader compiled
(``configs/experiments/hsn_t_memory_words_carry_bal_qk_write.yaml``, the user's decision of 26 Nov
2026).

It is ``hsn_t_memory_words_carry_bal_qk.yaml`` (unchanged) with four keys changed,
``model.word_cells=hash`` and ``model.word_cells_value=word`` (``docs/amendments/word_cells.md``,
stage 2) and ``train.tbptt_windows=3`` with ``train.grad_accum=3``, and nothing else: the same
configuration and so the same configuration hash once the run's name and notes are aligned, on
the CPU profile and with the card's lines. It loads, validates and builds on the CPU with the
measured runs' 3,405,689 trainable parameters. The CUDA arm's lines take
``hardware.compile_mode=default`` (a gradient segment refuses CUDA graphs), the two speed keys,
the fused event block and ``hardware.reader_compile=default`` (PR #116,
``docs/wcells_speed.md``); the CPU profile refuses the block and the reader compile.

A spec on it with the code-native data lines and the CUDA arm's lines expands to the arm ids the
measured runs recorded: the compiled reader's (spec ``fields_27k_wcells_rcomp``, run at 115cbaf:
0.9711 dev bits per byte, 80.99 % next-byte accuracy) and, without ``hardware.reader_compile``,
the HSN write's (spec ``fields_27k_wcells_word``, run at 89835a4: 0.9708, 80.98 %). Both specs set
the keys as arm overrides on ``hsn_t_memory_words_carry_bal_qk.yaml``. They live outside the
repository (``~/hsn-experiments/remote/specs``); what they pass is written out here.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from hsn.config import (
    REPO_ROOT,
    config_hash,
    event_kernel_scope_error,
    load_config,
    reader_compile_scope_error,
)
from hsn.experiments.spec import SpecError, config_view, expand, family_of, load_spec
from hsn.model.hsn import reader_carried
from hsn.model.hsn_t import HSNT
from hsn.train.memory_plan import word_cells_param_count
from p1_11_support import fake_repo, write_yaml
from test_hsn_t_memory_carry_recipes import PARAMS, SPEED, _changed
from test_hsn_t_memory_words_carry_bal_qk_recipe import FUSED, HEAD
from test_hsn_t_memory_words_carry_bal_recipe import CODE_NATIVE
from test_hsn_t_memory_words_recipe import _named_as

EXPERIMENTS = REPO_ROOT / "configs" / "experiments"
DATA = REPO_ROOT / "configs" / "data"
RECIPE = EXPERIMENTS / "hsn_t_memory_words_carry_bal_qk_write.yaml"
PARENT = EXPERIMENTS / "hsn_t_memory_words_carry_bal_qk.yaml"
#: the four keys this recipe changes on its parent, as arm overrides (the measured arms' order)
WRITE = (
    "model.word_cells=hash",
    "model.word_cells_value=word",
    "train.grad_accum=3",
    "train.tbptt_windows=3",
)
#: the same keys as configuration keys: (this recipe, its parent)
WRITE_KEYS: dict[str, tuple[Any, Any]] = {
    "model.word_cells": ("hash", "off"),
    "model.word_cells_value": ("word", "bytes"),
    "train.grad_accum": (3, 1),
    "train.tbptt_windows": (3, 1),
}
#: the compile mode of the CUDA arm: compiled, without CUDA graphs (a segment refuses them)
MODE = "hardware.compile_mode=default"
#: the opt-in compiled reader of PR #116
RC = "hardware.reader_compile=default"
#: the card's lines with that compile mode (the parents' card lines take reduce-overhead)
CARD = ("profile=rtx3060", MODE, "hardware.graph_alias_index=true")
#: the header's CUDA arm: its lines after the profile, and with it
ARM_LINES = (*CARD[1:], *SPEED, FUSED, RC)
CUDA_ARM = ("profile=rtx3060", *ARM_LINES)
HARDWARE = ((), CARD, CUDA_ARM)
#: the measured runs' param_count
WRITE_PARAMS = 3_405_689
#: the parent's: hsn_t_memory_words_carry_bal.yaml's and the top block's two norm weights
PARENT_PARAMS = PARAMS + 2 * HEAD
#: the decoder's new input columns, ``d + 2`` of them into its ``H`` hidden units
DECODER_COLUMNS = (64 + 2) * 512
#: what the measured arms passed on hsn_t_memory_words_carry_bal_qk.yaml after the data lines
#: (the family lines take reduce-overhead; the arm sets the compile mode the segments take)
AS_RUN = (
    "model.word_cells=hash",
    "model.word_cells_value=word",
    MODE,
    "train.grad_accum=3",
    "train.tbptt_windows=3",
)
#: the family lines of fields_27k_wcells_word and fields_27k_wcells_rcomp
FAMILY = ("hardware.compile_mode=reduce-overhead", "hardware.graph_alias_index=true", *SPEED, FUSED)
#: the measured runs, code native, 9,000 steps x 3 windows, seed 0: spec fields_27k_wcells_word
#: (the HSN write) and spec fields_27k_wcells_rcomp (the same arm with the compiled reader)
MEASURED = {"f27_code_native_word_xw3/s0": "69e767e8700385e8"}
MEASURED_RC = {"f27_code_native_word_xw3_rc/s0": "5a3d0954ca98e44c"}


def test_recipe_loads_with_the_hsn_write() -> None:
    cfg, parent = load_config(RECIPE), load_config(PARENT)
    assert cfg.model_name == "hsn_t" and family_of(cfg) == "hsn_t"
    m, t = cfg.model, cfg.train
    assert (m.word_cells, m.word_cells_value) == ("hash", "word")
    assert (parent.model.word_cells, parent.model.word_cells_value) == ("off", "bytes")
    assert (t.tbptt_windows, t.grad_accum) == (3, 3)
    assert (parent.train.tbptt_windows, parent.train.grad_accum) == (1, 1)
    # the keyed cells' sizes at their code defaults (the header)
    sizes = (m.word_cells_n, m.word_cells_ways, m.word_cells_levels, m.word_cells_min_len)
    assert sizes == (1024, 2, 3, 1) and m.word_cells_gate_bias == 3.0
    # the parent's reader, QK norm, carry, word events and loss
    assert (m.reader_width, m.reader_layers, m.reader_rotary_pairs) == (160, 4, [20, 20, 20, 0])
    assert m.reader_qk_norm == [False, False, False, True]
    assert m.reader_span is None and m.reader_carry == 256 and reader_carried(m)
    assert (m.write_events, m.word_address, m.word_contexts) == ("word", "exact", [0, 6])
    assert m.cell_attention_heads == 0
    assert t.aux_enabled == ["balance"] and t.lambda_b == 0.02 and t.lr == 1.25e-3
    assert (t.batch, t.micro_batch, t.unroll) == (32, 32, 256)
    # the CUDA arm's lines stay overrides of the spec, as in the parent
    hw = cfg.hardware
    assert (hw.compile_mode, hw.message_gradients, hw.round_products) == (
        "none",
        "autocast",
        "separate",
    )
    assert (hw.event_kernel, hw.reader_compile) == ("compiled", "none")


@pytest.mark.parametrize("hardware", HARDWARE, ids=["cpu", "card", "cuda_arm"])
def test_recipe_is_its_parent_with_the_write(hardware: tuple[str, ...]) -> None:
    cfg = load_config(RECIPE, list(hardware))
    parent = load_config(PARENT, list(hardware))
    # the four keys, the run's name and its notes, and nothing else in the whole configuration
    changed = _changed(cfg, parent)
    assert {k: v for k, v in changed.items() if k not in ("run_id", "notes")} == WRITE_KEYS
    assert {"run_id", "notes"} <= changed.keys()
    # so it is the parent with those overrides
    on_parent = load_config(PARENT, [*hardware, *WRITE])
    assert config_view(cfg) == config_view(on_parent)  # what the arm id reads
    assert config_hash(_named_as(cfg, PARENT)) == config_hash(on_parent)
    assert config_hash(cfg) != config_hash(parent)
    # the keys set again change nothing
    assert config_hash(load_config(RECIPE, [*hardware, *WRITE])) == config_hash(cfg)


def test_recipe_takes_the_cuda_arm_lines_on_the_card() -> None:
    """The CUDA arm's lines of the header: on the card's lines the two speed keys, the fused
    event block and the compiled reader change those four keys only, and both scopes take the
    recipe. The CPU profile of the file refuses the block and the reader compile, and the
    family lines' reduce-overhead refuses the segments."""
    card = load_config(RECIPE, list(CARD))
    fast = load_config(RECIPE, list(CUDA_ARM))
    assert _changed(fast, card) == {
        "hardware.message_gradients": ("state", "autocast"),
        "hardware.round_products": ("merged", "separate"),
        "hardware.event_kernel": ("fused", "compiled"),
        "hardware.reader_compile": ("default", "none"),
    }
    assert event_kernel_scope_error(fast) is None
    assert reader_compile_scope_error(fast) is None
    with pytest.raises(ValueError, match="CUDA only"):
        load_config(RECIPE, [FUSED])
    with pytest.raises(ValueError, match="compile_mode=none"):
        load_config(RECIPE, [RC])
    with pytest.raises(ValueError, match="CUDA graphs"):
        load_config(RECIPE, ["profile=rtx3060", "hardware.compile_mode=reduce-overhead"])


def test_recipe_builds_on_the_cpu_with_the_keyed_cells() -> None:
    """The model builds on the CPU with the measured runs' parameter count: the parent's, the
    keyed cells' module (the read maps ``U`` and the write's in-edge logits, ``beta_K`` and
    ``V_s``) and the decoder's ``d + 2`` columns that read them; the parent has no module."""
    cfg, parent = load_config(RECIPE), load_config(PARENT)
    model = HSNT.from_config(cfg)
    report = model.param_report()
    assert report["total_trainable"] == WRITE_PARAMS
    assert HSNT.from_config(parent).param_report()["total_trainable"] == PARENT_PARAMS
    assert report["word_cells"] == word_cells_param_count(cfg.model) == 65_536 + 8_256
    assert WRITE_PARAMS - PARENT_PARAMS - report["word_cells"] == DECODER_COLUMNS
    assert HSNT.from_config(parent).word_cells is None
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters() if n.startswith("word_cells")}
    assert shapes == {
        "word_cells.U": (16, 64, 64),
        "word_cells.V_s": (64, 64),
        "word_cells.edge_logit": (1024, 4),
        "word_cells.beta_K": (64,),
    }


def _arms(
    tmp_path: Path, name: str, config: Path, arm: list[str], *, label: str, family: tuple[str, ...]
) -> dict[str, str]:
    """The arm ids of a spec with ``fields_27k_wcells_word``'s base lines and protocol, the
    ``family`` lines of hsn_t and one code-native arm ``label`` on ``config`` with ``arm`` after
    the data lines."""
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
                "train.steps=9000",
                "train.ckpt_every=9000",
                "train.eval_every=3000",
            ],
        },
        "families": {
            "hsn_t": {
                "config": str(PARENT.relative_to(REPO_ROOT)),
                "overrides": list(family),
            }
        },
        "protocol": {
            "name": "train_eval",
            "steps": 9000,
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
                "name": label,
                "family": "hsn_t",
                "config": str(config.relative_to(REPO_ROOT)),
                "overrides": [*CODE_NATIVE, *arm],
            }
        ],
    }
    path = write_yaml(tmp_path / f"{name}.yaml", raw)
    return {a.label: a.arm_id for a in expand(load_spec(path)[0], repo=repo)}


def test_recipe_arms_are_the_measured_runs(tmp_path: Path) -> None:
    """A spec on the recipe with the code-native data lines and the CUDA arm's lines expands to
    the arm id of the run with the compiled reader (spec fields_27k_wcells_rcomp), and without
    ``hardware.reader_compile`` to the HSN write's (spec fields_27k_wcells_word), as the specs
    that ran them do (the keys as arm overrides on the parent); so does a spec whose family
    lines are the header's CUDA arm. Without the arm's compile mode the family lines'
    reduce-overhead refuses the segments, and the parent alone is another arm."""
    write, rc = "f27_code_native_word_xw3", "f27_code_native_word_xw3_rc"
    as_run = _arms(tmp_path, "as_run", PARENT, list(AS_RUN), label=write, family=FAMILY)
    from_recipe = _arms(tmp_path, "from_recipe", RECIPE, [MODE], label=write, family=FAMILY)
    assert from_recipe == as_run == MEASURED
    rc_as_run = _arms(tmp_path, "rc_as_run", PARENT, [*AS_RUN, RC], label=rc, family=FAMILY)
    rc_recipe = _arms(tmp_path, "rc", RECIPE, [MODE, RC], label=rc, family=FAMILY)
    assert rc_recipe == rc_as_run == MEASURED_RC
    # the header's CUDA arm as the family lines (its profile line is the base's)
    header = _arms(tmp_path, "header", RECIPE, [], label=rc, family=ARM_LINES)
    assert header == MEASURED_RC
    with pytest.raises(SpecError, match="CUDA graphs"):
        _arms(tmp_path, "no_mode", RECIPE, [], label=write, family=FAMILY)
    parent = _arms(tmp_path, "parent", PARENT, [], label=write, family=FAMILY)
    assert not set(parent.values()) & {*MEASURED.values(), *MEASURED_RC.values()}
