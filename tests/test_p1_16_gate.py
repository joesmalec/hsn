"""P1.16 tests: amendment A2 in ``scripts/gate_g1.py`` (``docs/amendments/A2_gate_protocol.md``).

Every method searches the same grid of absolute override sets (the learning rate {3e-4, 5e-4, 1e-3,
2e-3, 3e-3} on WSD 0.2) on its own recipe: the baselines take only the budget keys of the
experiment configuration, every configuration's ``train.search_trials`` is the grid size, the
horizon defaults to 9,000 steps, and every trial records its resolved recipe keys."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from hsn.config import config_hash, load_config

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO / "configs" / "experiments" / "g1_ref_a1.yaml"
GRU = REPO / "configs" / "baselines" / "gru_2m.yaml"
TRANSFORMER = REPO / "configs" / "baselines" / "transformer_2m.yaml"


def _gate() -> ModuleType:
    sys.path.insert(0, str(REPO / "scripts"))  # gate_g1 imports gate_g0 from its directory
    spec = importlib.util.spec_from_file_location("gate_g1", REPO / "scripts" / "gate_g1.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p1_16_gate_grid_absolute_and_distinct() -> None:
    """The default grid resolves for every method at the default horizon, five distinct
    configurations each, at the grid's learning rates on WSD 0.2; there is no trial 0."""
    gate = _gate()
    trials = gate.trial_sets(None)
    assert len(trials) == 5 and [] not in trials
    assert [t[0] for t in trials] == [f"train.lr={lr}" for lr in gate.SEARCH_LRS]
    hsn = load_config(EXPERIMENT)
    matched = gate.matched_overrides(hsn)
    budget = gate.search_overrides(trials)
    for path, extra in ((EXPERIMENT, []), (GRU, matched), (TRANSFORMER, matched)):
        hashes = set()
        for trial in trials:
            cfg = load_config(
                path, overrides=[*extra, *budget, *trial, f"train.steps={gate.SEARCH_STEPS}"]
            )
            assert cfg.train.schedule == "wsd" and cfg.train.decay_fraction == 0.2
            assert cfg.train.steps == 9000 and cfg.train.search_trials == 5
            hashes.add(config_hash(cfg))
        assert len(hashes) == 5, path
    assert {float(lr) for lr in gate.SEARCH_LRS} == {3e-4, 5e-4, 1e-3, 2e-3, 3e-3}


def test_p1_16_gate_no_recipe_inheritance() -> None:
    """The baselines take the budget keys only: their warm-up, weight decay, betas and gradient
    clip are their own, not the experiment configuration's."""
    gate = _gate()
    hsn = load_config(EXPERIMENT)
    matched = gate.matched_overrides(hsn)
    recipe = {"lr", "betas", "weight_decay", "warmup_steps", "schedule", "grad_clip"}
    assert not {o.split("=")[0].removeprefix("train.") for o in matched} & recipe
    assert set(gate.MATCHED_TRAIN_KEYS) == {
        "steps",
        "batch",
        "micro_batch",
        "unroll",
        "eval_every",
        "ckpt_every",
        "stats_every",
    }
    own = load_config(GRU)
    gru = load_config(GRU, overrides=[*matched, *gate.trial_sets(None)[0]])
    assert hsn.train.warmup_steps == 200 and gru.train.warmup_steps == own.train.warmup_steps == 100
    assert gru.train.weight_decay == own.train.weight_decay
    assert gru.train.betas == own.train.betas
    assert (gru.train.steps, gru.train.batch) == (hsn.train.steps, hsn.train.batch)
    assert gru.train.lr == 3e-4  # the grid's, absolute


def test_p1_16_gate_search_trials_equal_grid() -> None:
    gate = _gate()
    assert gate.search_overrides([["a=1"], ["b=2"], ["c=3"]]) == ["train.search_trials=3"]
    for path in (EXPERIMENT, GRU, TRANSFORMER):
        cfg = load_config(path, overrides=gate.search_overrides(gate.trial_sets(None)))
        assert cfg.train.search_trials == 5


def test_p1_16_gate_search_steps_default() -> None:
    gate = _gate()
    args = gate._parser().parse_args(["--profile", "cpu", "--out", "x"])
    assert args.search_steps == gate.SEARCH_STEPS == 9000
    custom = gate.trial_sets([["train.lr=1e-3"], ["train.lr=2e-3"]])
    assert custom == [["train.lr=1e-3"], ["train.lr=2e-3"]]  # the --trial sets are the grid


def test_p1_16_gate_trial_recipe_recorded() -> None:
    gate = _gate()
    cfg = load_config(GRU, overrides=["train.betas=[0.9,0.95]"])
    recipe = gate.trial_recipe(cfg)
    assert set(recipe) == set(gate.RECIPE_KEYS)
    assert recipe["betas"] == [0.9, 0.95] and recipe["warmup_steps"] == 100
    table = gate.search_table(
        {
            "model": "gru",
            "config": str(GRU),
            "trials": [
                {
                    "index": 0,
                    "status": "ok",
                    "recipe": recipe,
                    "result": {gate.SEARCH_METRIC: 0.5},
                },
                {"index": 1, "status": "failed", "error": "x"},
            ],
        }
    )
    assert "trial 0: ok 0.5000 lr=0.001" in table and "betas=[0.9, 0.95]" in table
    assert "trial 1: failed -" in table
