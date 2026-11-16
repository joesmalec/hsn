"""The standard of HSN-T without its memory on text (``configs/experiments/
hsn_t_nomem_top0_w1024.yaml``, the user's decision of 16 Nov 2026: "HSN-T top0, 1024 windows,
no carry") and its data (``configs/data/tinystories_w1024.yaml``).

The recipe loads and validates at 1,887,552 parameters; it resolves to ``hsn_t_nomem.yaml`` with
the overrides of the 27k spec that measured it (spec ``w1024_27k``, arm
``ts27_nomem_w160l4_top0_w1024``), on the CPU profile and with the card's lines; a spec on it
expands to the arm ids that round recorded. The data spec is ``tinystories_v1`` with the window
changed, and its golden manifest is the one those runs read.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pytest
from hsn.config import REPO_ROOT, RunConfig, config_hash, load_config
from hsn.data.splits import SplitManifest
from hsn.data.text import load_text_spec
from hsn.experiments.spec import config_view, expand, family_of, load_spec
from hsn.model.hsn_t import HSNTNoMemory
from hsn.train.memory_plan import memory_off_param_count
from p1_11_support import fake_repo, write_yaml

EXPERIMENTS = REPO_ROOT / "configs" / "experiments"
DATA = REPO_ROOT / "configs" / "data"
RECIPE = EXPERIMENTS / "hsn_t_nomem_top0_w1024.yaml"
NOMEM = EXPERIMENTS / "hsn_t_nomem.yaml"
#: the overrides of arm ts27_nomem_w160l4_top0_w1024 of the 27k spec, on hsn_t_nomem.yaml
SPEC_ARM = (
    "model.reader_width=160",
    "model.reader_layers=4",
    "model.reader_rotary_pairs=[20,20,20,0]",
    "data.dataset=tinystories_w1024",
    "data.manifest=data/tinystories_w1024/manifest.json",
    "data.tasks=[tinystories]",
    "data.variants=[out_of_range]",
    "train.unroll=1024",
    "train.batch=8",
    "train.micro_batch=8",
)
#: what that spec's families block and base added for the card, left to a spec by the recipe
CARD = (
    "profile=rtx3060",
    "hardware.compile_mode=none",
    "train.steps=27000",
    "train.ckpt_every=27000",
    "train.eval_every=1500",
)
#: the lines of that families block the arm's own overrides replaced (tinystories_v1, batch 32)
V1_LINES = (
    "data.dataset=tinystories_v1",
    "data.manifest=data/tinystories_v1/manifest.json",
    "data.tasks=[tinystories]",
    "data.variants=[out_of_range]",
    "train.batch=32",
    "train.micro_batch=32",
)
#: arm ids of the 27k round (its experiment database, 16 Nov 2026); the data hash in them is the
#: blake2b of the golden manifest below
RECORDED = {
    "ts27_nomem_w160l4_top0_w1024/s0": "7e30db1226dcddff",
    "ts27_nomem_w160l4_top0_w1024/s1": "c7f5653cf942b69d",
    "ts27_nomem_w160l4_top0_w1024_c1024/s0": "b376947d05d0f05c",
}
DATA_HASH = "d5467e784af6c272"
PARAMS = 1_887_552


def _named_as_nomem(cfg: RunConfig) -> RunConfig:
    """``cfg`` with ``hsn_t_nomem``'s run name and notes, the two keys the configuration hash
    reads that are not part of the run (the arm id leaves them out)."""
    base = load_config(NOMEM)
    return cfg.model_copy(update={"run_id": base.run_id, "notes": base.notes})


def test_recipe_loads_at_the_decided_shape() -> None:
    cfg = load_config(RECIPE)
    assert cfg.model_name == "hsn_t" and family_of(cfg) == "hsn_t_nomem"
    m = cfg.model
    assert not m.memory and m.reader == "transformer"
    assert (m.reader_width, m.reader_layers, m.reader_heads) == (160, 4, 4)
    assert m.reader_rotary_pairs == [20, 20, 20, 0]  # top0
    assert m.reader_span is None and m.reader_carry == 0 and m.reader_top_words == 0
    assert cfg.data.dataset == "tinystories_w1024" and cfg.data.tasks == ["tinystories"]
    assert (cfg.train.unroll, cfg.train.batch, cfg.train.micro_batch) == (1024, 8, 8)
    assert cfg.train.batch * cfg.train.unroll == 32 * 256  # the positions of a 256 step
    assert cfg.train.optimizer == "adamw" and cfg.train.lr == 1.0e-3 and cfg.train.lambda_p == 0
    assert cfg.hardware.name == "cpu"  # the card is a spec's override
    report = HSNTNoMemory.from_config(cfg).param_report()
    assert report["trainable"] == report["stored"] == PARAMS
    assert memory_off_param_count(m)["total"] == PARAMS
    # the carry the decision left out would add no parameters
    carry = load_config(RECIPE, ["model.reader_carry=1024"])
    assert HSNTNoMemory.from_config(carry).param_report()["trainable"] == PARAMS


@pytest.mark.parametrize("card", [(), CARD], ids=["cpu", "card"])
def test_recipe_is_nomem_with_the_overrides_of_the_27k_arm(card: tuple[str, ...]) -> None:
    recipe = load_config(RECIPE, list(card))
    spec = load_config(NOMEM, [*card, *SPEC_ARM])
    assert config_view(recipe) == config_view(spec)  # what the arm id reads
    assert config_hash(_named_as_nomem(recipe)) == config_hash(spec)
    assert recipe.run_id != spec.run_id  # the only differences: the name and the notes


def _repo(tmp: Path) -> Path:
    repo = fake_repo(tmp)
    built = repo / "data" / "tinystories_w1024"
    built.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / "tinystories_w1024.manifest.json", built / "manifest.json")
    return repo


def _arm_ids(tmp: Path, name: str, config: Path, family: list[str], arm: list[str]) -> Any:
    """The 27k spec's arms of this recipe (seeds 0 and 1, and the carry arm) on ``config``."""
    raw = {
        "name": name,
        "timing": False,
        "base": {"config": str(config), "overrides": ["profile=rtx3060"]},
        "families": {"hsn_t_nomem": {"config": str(config), "overrides": family}},
        "protocol": {
            "name": "train_eval",
            "steps": 27000,
            "diagnostics": False,
            "diag_every": 250,
            "diag_docs": 32,
            "eval_batch": 128,
            "checkpoint": "retained",
            "keep_every": 27000,
            "keep_last": 1,
        },
        "seeds": [0],
        "arms": [
            {
                "name": "ts27_nomem_w160l4_top0_w1024",
                "family": "hsn_t_nomem",
                "overrides": arm,
                "seeds": [0, 1],
                "protocol": {"eval_batch": 32},
            },
            {
                "name": "ts27_nomem_w160l4_top0_w1024_c1024",
                "family": "hsn_t_nomem",
                "overrides": [*arm, "model.reader_carry=1024"],
                "protocol": {"eval_batch": 32},
            },
        ],
    }
    path = write_yaml(tmp / f"{name}.yaml", raw)
    arms = expand(load_spec(path)[0], repo=_repo(tmp))
    assert {a.data_hash for a in arms} == {DATA_HASH}
    return {a.label: a.arm_id for a in arms}


def test_recipe_arms_are_the_27k_rounds(tmp_path: Path) -> None:
    """A spec on the recipe and the 27k spec on ``hsn_t_nomem.yaml`` expand to the same arm ids
    (the base file's path is not part of an arm's identity), the ones that round recorded."""
    card = list(CARD[1:])
    from_nomem = _arm_ids(tmp_path, "from_nomem", NOMEM, [*card, *V1_LINES], list(SPEC_ARM))
    from_recipe = _arm_ids(tmp_path, "from_recipe", RECIPE, card, [])
    assert from_recipe == from_nomem == RECORDED


def test_data_spec_is_v1_at_windows_of_1024() -> None:
    w1024 = load_text_spec(DATA / "tinystories_w1024.yaml")
    v1 = load_text_spec(DATA / "tinystories_v1.yaml")
    assert (w1024.name, w1024.unroll, v1.unroll) == ("tinystories_w1024", 1024, 256)
    assert w1024.model_dump(exclude={"name", "unroll"}) == v1.model_dump(exclude={"name", "unroll"})
    golden_path = DATA / "tinystories_w1024.manifest.json"
    golden = SplitManifest.load(golden_path)
    assert golden.name == "tinystories_w1024" and golden.unroll == 1024
    assert golden.spec_hash == w1024.spec_hash()
    assert golden.leakage == []  # built with --no-leakage (the header of the data spec)
    assert hashlib.blake2b(golden_path.read_bytes(), digest_size=32).hexdigest()[:16] == DATA_HASH

    # the same stories as v1: per split the same documents, positions and targets
    def counts(m: SplitManifest) -> dict[str, tuple[int, int, int]]:
        return {s.split: (s.n_docs, s.n_tokens, s.n_loss) for s in m.shards}

    assert counts(golden) == counts(SplitManifest.load(DATA / "tinystories_v1.manifest.json"))
