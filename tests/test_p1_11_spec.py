"""P1.11 tests: sweep specs expand deterministically into arms with stable ids.

An arm id hashes what the arm computes: the resolved configuration's non-default values (not
``run_id``), the data, the family, the variant, the protocol's science parameters and the
early-stop rules. It does not move with YAML spelling or key order, the arm's name, execution-only
parameters (evaluation and probe batch) or verdict rules, and it moves with any override, the seed,
the variant, a protocol science parameter, an early-stop rule or the data. The golden table pins the
validation sweep's ids so that a change of the hash shows up in review.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from hsn.experiments.spec import SpecError, SweepSpec, expand, load_spec
from p1_11_support import REPO, fake_repo, write_yaml

VALIDATION = REPO / "configs" / "sweeps" / "p1_11_validation.yaml"
# expanded on 7 Nov 2026 from configs/sweeps/p1_11_validation.yaml and the synthetic_v1 manifest;
# a change here changes which arms count as already run (enqueue dedup): explain it in the PR
GOLDEN = {
    "gru384x2_b32_lr3e-3/s0": "af02e23493b600c0",
    "gru128x1_b32_lr3e-3/s0": "9dc4466c5460dab4",
    "gru64x1_b32_lr3e-3/s0": "1fd2b47d0846b270",
    "hsn_pdg_b32_lr1e-3_bridge/s0": "79b251da811d9f63",
    "hsn_pdg_b32_lr1e-3_bridge_compiled/s0": "f6ce1da2777804a8",
}
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _spec(**changes: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "name": "unit",
        "base": {
            "config": "configs/experiments/g1_ref_a1.yaml",
            "overrides": [
                "train.batch=8",
                "train.micro_batch=8",
                "train.steps=40",
                "train.eval_every=10",
            ],
        },
        "families": {
            "gru": {"config": "configs/baselines/gru_2m.yaml", "overrides": ["train.lr=0.003"]}
        },
        "protocol": {"name": "reads_state", "diag_every": 10, "eval_batch": 16, "probe_batch": 8},
        "constants": {"chance": 0.0208333},
        "seeds": [0, 1],
        "rules": {
            "verdicts": {
                "reads_state": {"preset": "reads_state"},
                "open": {"preset": "memory_open", "from_step": 10, "every": 10},
            },
            "early_stop": [
                {
                    "name": "fail_early",
                    "at_steps": [20],
                    "outcome": "fail_early",
                    "when": {
                        "metric": "eval:accuracy",
                        "steps": {"at": 20},
                        "op": "<=",
                        "value": "chance + 0.01",
                    },
                }
            ],
        },
        "arms": [
            {"name": "hsn", "family": "hsn", "overrides": ["train.lr=1e-3"]},
            {"name": "gru", "family": "gru", "grid": {"baseline.gru.d_model": [64, 128]}},
        ],
    }
    for key, value in changes.items():
        raw[key] = value
    return raw


def _ids(tmp: Path, raw: dict[str, Any], *, repo: Path | None = None) -> dict[str, str]:
    path = write_yaml(tmp / f"spec_{len(list(tmp.glob('spec_*')))}.yaml", raw)
    spec, _ = load_spec(path)
    return {a.label: a.arm_id for a in expand(spec, repo=repo or fake_repo(tmp))}


def test_p1_11_spec_expansion_is_deterministic_and_ids_stable(tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    base = _ids(tmp_path, _spec(), repo=repo)
    assert list(base) == [
        "hsn/s0",
        "hsn/s1",
        "gru[d_model=64]/s0",
        "gru[d_model=64]/s1",
        "gru[d_model=128]/s0",
        "gru[d_model=128]/s1",
    ]
    assert len(set(base.values())) == len(base)
    assert _ids(tmp_path, _spec(), repo=repo) == base

    # unchanged: YAML spelling and key order, the arm's name, execution-only parameters, verdict
    # rules, and an override that sets a key to its default
    same = _spec()
    same["arms"][0]["overrides"] = ["train.lr=0.001"]
    same["arms"][0] = dict(reversed(list(same["arms"][0].items())))
    same["protocol"] = {
        "probe_batch": 32,
        "eval_batch": 64,
        "diag_every": 10,
        "name": "reads_state",
    }
    same["rules"]["verdicts"] = {"other": {"preset": "cells_apart"}}
    same["base"]["overrides"] = [*same["base"]["overrides"], "train.q_resync_every=64"]  # default
    renamed = copy.deepcopy(same)
    renamed["arms"][0]["name"] = "hsn_renamed"
    ids = _ids(tmp_path, same, repo=repo)
    assert ids == base
    assert list(_ids(tmp_path, renamed, repo=repo).values()) == list(base.values())

    # changed: an override, the seed, the variant, a protocol science parameter, an early-stop rule
    def changed(raw: dict[str, Any], label: str = "hsn/s0", repo_: Path = repo) -> bool:
        return _ids(tmp_path, raw, repo=repo_)[label] != base[label]

    lr = _spec()
    lr["arms"][0]["overrides"] = ["train.lr=2e-3"]
    assert changed(lr)
    variant = _spec()
    variant["arms"][0]["variant"] = "cell_offset"
    assert changed(variant)
    offset = _spec()
    offset["arms"][0]["variant"] = {"name": "cell_offset", "offset_std": 0.25}
    assert (
        _ids(tmp_path, offset, repo=repo)["hsn/s0"] != _ids(tmp_path, variant, repo=repo)["hsn/s0"]
    )
    diag = _spec()
    diag["protocol"]["diag_every"] = 5
    assert changed(diag)
    stop = _spec()
    stop["rules"]["early_stop"][0]["at_steps"] = [30]
    stop["rules"]["early_stop"][0]["when"]["steps"] = {"at": 30}
    assert changed(stop)
    assert base["hsn/s0"] != base["hsn/s1"]
    # an early stop that differs only by an arm constant is a different arm (the resolved value is
    # hashed, not the constant's name)
    by_constant = _spec()
    by_constant["rules"]["early_stop"][0]["when"]["value"] = "cut"
    by_constant["arms"] = [
        {"name": "a", "family": "hsn", "constants": {"cut": 0.1}},
        {"name": "b", "family": "hsn", "constants": {"cut": 0.2}},
    ]
    ids_c = _ids(tmp_path, by_constant, repo=repo)
    assert ids_c["a/s0"] != ids_c["b/s0"]
    other_data = fake_repo(tmp_path / "other", manifest=b'{"rebuilt": true}')
    assert changed(_spec(), repo_=other_data)


def test_p1_11_spec_golden_ids_of_the_validation_sweep(tmp_path: Path) -> None:
    spec, text = load_spec(VALIDATION)
    arms = expand(spec, repo=fake_repo(tmp_path))
    assert {a.label: a.arm_id for a in arms} == GOLDEN
    for arm in arms:
        assert RUN_ID.match(arm.run_id) and arm.config.run_id == arm.run_id
        assert arm.steps == 3000 and arm.config.train.batch == 32
        assert arm.config.data.tasks == ["delayed"] and arm.protocol.name == "reads_state"
    bridge = next(a for a in arms if a.name == "hsn_pdg_b32_lr1e-3_bridge")
    assert bridge.timing and bridge.variant == "pdg" and bridge.config.train.lr == 1e-3
    gru = next(a for a in arms if a.name == "gru64x1_b32_lr3e-3")
    assert gru.variant == "none" and gru.config.train.warmup_steps == 200
    assert gru.config.baseline.gru.d_model == 64 and gru.config.train.lr == 3e-3
    assert "name: p1_11_validation" in text


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"unknown": 1}, "Extra inputs"),
        ({"arms": [{"name": "a", "overrides": ["train.no_such_key=1"]}]}, "unknown config key"),
        (
            {"arms": [{"name": "a", "family": "gru", "variant": "cell_offset"}]},
            "needs model family hsn",
        ),
        (
            {
                "arms": [
                    {
                        "name": "a",
                        "variant": "cell_offset",
                        "overrides": ["hardware.compile_mode=reduce-overhead"],
                    }
                ]
            },
            "untested",
        ),
        (
            {
                "arms": [
                    {"name": "a", "family": "gru", "config": "configs/experiments/g1_ref_a1.yaml"}
                ]
            },
            "model_name",
        ),
        ({"arms": [{"name": "a"}, {"name": "a"}]}, "duplicate arm label"),
        ({"arms": [{"name": "a", "overrides": ["model.routing=random"]}]}, "routing=random"),
        ({"arms": [{"name": "a", "protocol": {"steps": 50}}]}, "protocol steps"),
        ({"protocol": {"name": "no_such_protocol"}}, "protocol name"),
        ({"arms": [{"name": "a", "variant": "gru_cell_direct_input"}]}, "retired"),
        (
            {
                "arms": [
                    {"name": "a", "variant": "pdg", "family": "hsn", "protocol": {"diag_every": 7}}
                ]
            },
            "does not record",
        ),
        ({"arms": [{"name": "a b"}]}, "String should match"),
        ({"families": {"ssm": {"config": "x.yaml"}}}, "unknown model families"),
    ],
)
def test_p1_11_spec_rejects_bad_input(tmp_path: Path, change: dict[str, Any], match: str) -> None:
    with pytest.raises(SpecError, match=match):
        _ids(tmp_path, _spec(**change))


def test_p1_11_spec_requires_built_data(tmp_path: Path) -> None:
    raw = _spec()
    repo = tmp_path / "bare"
    (repo).mkdir()
    (repo / "configs").symlink_to(REPO / "configs", target_is_directory=True)
    with pytest.raises(SpecError, match="does not exist"):
        _ids(tmp_path, raw, repo=repo)
    raw["protocol"] = {"name": "dry_run"}
    raw["rules"] = None
    ids = _ids(tmp_path, raw, repo=repo)  # a dry run needs no data
    assert len(ids) == 6
    assert SweepSpec.model_validate(yaml.safe_load(yaml.safe_dump(raw))).name == "unit"
