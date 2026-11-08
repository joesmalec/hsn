"""P1.16 tests: the standard HSN recipe file (AdamW beta2 0.95) resolves to exp 194's configuration
(so its arms share exp 194's arm id), the defaults stay as they were, and the example spec of
P1.16's additions expands."""

from __future__ import annotations

from pathlib import Path

from hsn.config import REPO_ROOT, TrainConfig, load_config
from hsn.experiments.spec import config_view, expand, load_spec
from p1_11_support import fake_repo

RECIPE = REPO_ROOT / "configs" / "experiments" / "hsn_recipe_b095.yaml"
#: exp 194's spec (s3_beta2_hsn.yaml on branch hsn-efficiency, 8 Nov 2026): g1_ref_a1 plus these
EXP194 = (
    "profile=rtx3060",
    "hardware.compile_mode=reduce-overhead",
    "train.batch=32",
    "train.micro_batch=32",
    "train.lr=0.002",
    "train.steps=9000",
    "train.schedule=wsd",
    "train.decay_fraction=0.2",
    "train.eval_every=500",
    "train.ckpt_every=9000",
    "train.stats_every=100",
    "data.tasks=[delayed]",
    "data.variants=[in_range]",
    "model.cand_norm=rms",
    "train.betas=[0.9,0.95]",
    "train.spike_factor=10000",
)


def test_p1_16_recipe_resolves_to_exp194() -> None:
    recipe = load_config(RECIPE, ["profile=rtx3060"])
    exp194 = load_config(REPO_ROOT / "configs" / "experiments" / "g1_ref_a1.yaml", list(EXP194))
    assert config_view(recipe) == config_view(exp194)
    assert recipe.train.betas == (0.9, 0.95) and recipe.model.cand_norm == "rms"
    assert recipe.hardware.compile_mode == "reduce-overhead"  # profile:rtx3060 compiles (P2.8)


def test_p1_16_recipe_leaves_defaults() -> None:
    assert TrainConfig().betas == (0.9, 0.999)
    for name in ("configs/ref_a1.yaml", "configs/experiments/g1_ref_a1.yaml"):
        assert load_config(REPO_ROOT / name).train.betas == (0.9, 0.999)
    assert load_config(REPO_ROOT / "configs/baselines/gru_2m.yaml").train.betas == (0.9, 0.999)


def test_p1_16_example_spec_expands(tmp_path: Path) -> None:
    spec, _ = load_spec(REPO_ROOT / "configs" / "sweeps" / "example_p1_16.yaml")
    arms = expand(spec, repo=fake_repo(tmp_path))
    by_label = {a.label: a for a in arms}
    assert set(by_label) == {"hsn_recipe/s0", "hsn_rollback/s0"}
    # the recipe with exp 194's protocol is exp 194's arm (s3_beta2_hsn, run dir ...-612f783264ab)
    assert by_label["hsn_recipe/s0"].arm_id == "612f783264abbd77"
    protocol = by_label["hsn_recipe/s0"].protocol
    assert protocol.keep_every == 1000 and protocol.keep_last == 2  # type: ignore[union-attr]
    rollback = load_config(RECIPE, ["profile=rtx3060", "train.rollback_every=50"])
    assert rollback.train.rollback_every == 50
