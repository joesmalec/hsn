"""P1.18 tests: the standard HSN recipe file of 11 Nov 2026 (amendment A6 with the route guard)
resolves to exp 421's configuration (so its arms share the arm ids of exps 421-426), the defaults
and the earlier recipe stay as they were, and the example spec expands to the arms of the
nine-run comparison."""

from __future__ import annotations

from pathlib import Path

from hsn.config import REPO_ROOT, ModelConfig, TrainConfig, load_config
from hsn.experiments.spec import config_view, expand, load_spec
from p1_11_support import fake_repo

EXPERIMENTS = REPO_ROOT / "configs" / "experiments"
RECIPE = EXPERIMENTS / "hsn_recipe_a6.yaml"
PROFILE = ("profile=rtx3060", "hardware.compile_mode=reduce-overhead")
#: exps 421-426's spec (p1_18_a6_final_seeds / _final_guard_seeds): hsn_recipe_b095 plus these
EXP421 = (
    *PROFILE,
    "model.routing=topk",
    "model.cand_norm=rms",
    "model.key_address=true",
    "model.selected_readout=true",
    "train.optimizer=lazy_adamw",
    "train.gate_mode=ste",
    "train.lr=2.5e-3",
    "train.betas=[0.9,0.95]",
    "train.probe_horizon=null",
    "train.spike_factor=null",
    "train.rollback_every=null",
    "train.grad_norm_precision=float64",
    "train.feedback_detach=[summary,eps,state,router]",
    'train.update_limit.groups={memory: {selectors: ["cell.*"], max_ratio: 0.003}, '
    'routing: {selectors: ["router.*"], max_ratio: 0.003}}',
    "train.update_limit.norm_floor=1e-12",
    "train.update_limit.norm_growth_limits={routing: 1.0}",
    "train.weight_ema=0.99",
    "train.route_guard.enabled=true",
)
#: the stored arm ids of the nine-run comparison (docs/p1_18_a6_results.md), seeds 0 to 5
GUARDED = (  # exps 421-426 (and 430-432 on seeds 0-2)
    "ad8e72c5d039096a",
    "a9945b6f87bb0ee6",
    "10d1bbc7a15ec0cf",
    "a498b16a5f731782",
    "5cde3036b9512932",
    "3928875e2f5a39df",
)
UNGUARDED = (  # exps 375-377 and 418-420 (and 433-435 on seeds 0-2)
    "ae1b745a59b9bafb",
    "2331121b0629b66c",
    "8955604a3c788d05",
    "633e4e09fc891245",
    "0bea153fc87834ff",
    "1be72e354d064fdc",
)


def test_p1_18_recipe_resolves_to_exp421() -> None:
    recipe = load_config(RECIPE, list(PROFILE))
    exp421 = load_config(EXPERIMENTS / "hsn_recipe_b095.yaml", list(EXP421))
    assert config_view(recipe) == config_view(exp421)
    assert load_config(RECIPE, ["profile=rtx3060"]) == recipe  # profile:rtx3060 compiles (P2.8)
    train, model = recipe.train, recipe.model
    assert model.routing == "topk" and model.cand_norm == "rms"
    assert model.key_address and model.selected_readout
    assert train.feedback_detach == ["summary", "eps", "state", "router"]
    assert train.weight_ema == 0.99 and train.route_guard.enabled
    assert set(train.update_limit.groups) == {"memory", "routing"}
    assert train.update_limit.norm_growth_limits == {"routing": 1.0}
    # unguarded: no skipped step and no rollback
    assert train.spike_factor is None and train.rollback_every is None


def test_p1_18_recipe_leaves_defaults() -> None:
    train = TrainConfig()
    assert train.feedback_detach == [] and train.weight_ema is None
    assert not train.route_guard.enabled and train.update_limit.groups == {}
    assert ModelConfig().routing == "dense" and not ModelConfig().key_address
    for name in ("configs/ref_a1.yaml", "configs/experiments/hsn_recipe_b095.yaml"):
        earlier = load_config(REPO_ROOT / name)
        assert earlier.model.routing == "dense" and earlier.train.weight_ema is None
        assert earlier.train.feedback_detach == [] and not earlier.train.route_guard.enabled
    # the earlier recipe keeps its own settings (its arm ids are exp 194's, test_p1_16_recipe)
    b095 = load_config(EXPERIMENTS / "hsn_recipe_b095.yaml")
    assert b095.train.lr == 2e-3 and b095.train.spike_factor == 10000.0


def test_p1_18_example_spec_expands(tmp_path: Path) -> None:
    spec, _ = load_spec(REPO_ROOT / "configs" / "sweeps" / "example_p1_18.yaml")
    arms = expand(spec, repo=fake_repo(tmp_path))
    by_label = {a.label: a.arm_id for a in arms}
    assert len(by_label) == 12
    # the recipe with the comparison's protocol is the stored arms, with the guard and without
    assert tuple(by_label[f"hsn_recipe/s{seed}"] for seed in range(6)) == GUARDED
    assert tuple(by_label[f"no_guard/s{seed}"] for seed in range(6)) == UNGUARDED
