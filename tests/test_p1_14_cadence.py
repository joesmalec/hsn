"""P1.14 tests: the mixed diagnostic cadence and ``min_samples``.

With ``diag_dense_every`` a spec's arm samples its diagnostics densely inside the windows its rules
read (derived at expansion, stored in the protocol, part of the arm id) and every ``diag_every``
steps elsewhere; without it nothing changes (schedule and arm id). ``min_samples`` is set per rule
(D1, A): the static check refuses a window the cadence fills with fewer steps, and fewer recorded
samples make the atom missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hsn.config import load_config
from hsn.experiments.protocols import ArmRun, run_protocol
from hsn.experiments.rules import (
    Cadence,
    RuleError,
    check_rules,
    diag_schedule,
    evaluate_verdicts,
    merge_windows,
    parse_rules,
    rule_windows,
)
from hsn.experiments.spec import SpecError, expand, load_spec, protocol_params
from p1_11_support import (
    REF_A1,
    TINY_HSN_OVERRIDES,
    disk_overrides,
    fake_repo,
    restore_determinism,  # noqa: F401  (fixture)
    tiny_manifest,  # noqa: F401  (fixture)
    write_yaml,
)

WINDOWED = {"preset": "memory_open", "window": 100, "to_step": 1000, "from_step": 500}


def _spec(tmp: Path, *, rules: dict[str, Any] | None, **protocol: Any) -> dict[str, Any]:
    raw = {
        "name": "cadence",
        "base": {
            "config": "configs/experiments/g1_ref_a1.yaml",
            "overrides": ["train.batch=8", "train.micro_batch=8", "train.steps=1000"],
        },
        "protocol": {"name": "reads_state", "diag_every": 500, **protocol},
        "rules": rules,
        "arms": [{"name": "a", "family": "hsn"}],
    }
    path = write_yaml(tmp / f"spec_{len(list(tmp.glob('spec_*')))}.yaml", raw)
    spec, _ = load_spec(path)
    return {a.label: a for a in expand(spec, repo=fake_repo(tmp))}


def test_p1_14_diag_schedule_and_windows() -> None:
    assert diag_schedule(10, 4) == (4, 8)
    assert diag_schedule(10, 0) == ()
    assert diag_schedule(100, 50, 10, [(15, 32), (90, 200)]) == (20, 30, 50, 90, 100)
    assert diag_schedule(10, 0, 1, [(3, 4)]) == (3, 4)
    assert merge_windows([[90, 110], [10, 20], [100, 130], [15, 18]]) == [[10, 20], [90, 130]]
    rules = parse_rules(
        {
            "verdicts": {
                "w": WINDOWED,
                "at": {
                    "cases": [
                        {
                            "label": "x",
                            "when": {
                                "metric": "diag:gate/p90",
                                "steps": {"at": 750},
                                "op": ">",
                                "value": 0.1,
                            },
                        }
                    ],
                    "default": "y",
                },
                "gru": {
                    **WINDOWED,
                    "from_step": 100,
                    "to_step": 100,
                    "applies_to": {"family": "gru"},
                },
                "eval": {
                    "cases": [
                        {
                            "label": "x",
                            "when": {
                                "metric": "eval:accuracy",
                                "steps": {"from": 1, "to": 9},
                                "agg": "max",
                                "op": ">",
                                "value": 0,
                            },
                        }
                    ],
                    "default": "y",
                },
            },
            "early_stop": [
                {
                    "name": "s",
                    "at_steps": [300],
                    "outcome": "o",
                    "when": {
                        "metric": "diag:grad_share/cell",
                        "steps": {"from": 200, "to": 300, "every": 25},
                        "agg": "median",
                        "op": "<",
                        "value": 0.01,
                    },
                }
            ],
        }
    )
    # memory_open's windows around 500 and 1000 (clipped at the run's end), the one-step window,
    # the early stop's span; the gru-only rule and the eval atom read no hsn diag: window
    assert rule_windows(rules, "hsn", "reads_state", 1000) == [
        [200, 300],
        [400, 600],
        [750, 750],
        [900, 1000],
    ]
    everything = parse_rules(
        {
            "verdicts": {
                "all": {
                    "cases": [
                        {
                            "label": "x",
                            "when": {
                                "metric": "diag:gate/p50",
                                "agg": "median",
                                "op": ">",
                                "value": 0,
                            },
                        }
                    ],
                    "default": "y",
                }
            }
        }
    )
    with pytest.raises(RuleError, match="mix of dense and sparse"):
        rule_windows(everything, "hsn", "reads_state", 1000)


def test_p1_14_mixed_cadence_in_specs(tmp_path: Path) -> None:
    rules = {"verdicts": {"w": {**WINDOWED, "min_samples": 5}}}
    single = _spec(tmp_path, rules={"verdicts": {"w": WINDOWED}})["a/s0"]
    # a single cadence of 500 gives each window one sample: fine without min_samples, refused with
    assert single.protocol.diag_windows == () and single.protocol.diag_dense_every == 0
    with pytest.raises(
        SpecError, match=r"holds 1 of this arm's diag steps, fewer than min_samples 5"
    ):
        _spec(tmp_path, rules=rules)
    mixed = _spec(tmp_path, rules=rules, diag_dense_every=25)["a/s0"]
    assert mixed.protocol.diag_windows == ((400, 600), (900, 1000))
    steps = diag_schedule(1000, 500, 25, mixed.protocol.diag_windows)
    assert steps[:3] == (400, 425, 450) and steps[-1] == 1000 and len(steps) == 9 + 5
    assert mixed.arm_id != single.arm_id  # the schedule is part of what the arm records
    # explicit windows are merged with the derived ones; windows without the dense cadence refuse
    extra = _spec(tmp_path, rules=rules, diag_dense_every=25, diag_windows=[[100, 150], [580, 650]])
    assert extra["a/s0"].protocol.diag_windows == ((100, 150), (400, 650), (900, 1000))
    with pytest.raises(SpecError, match="diag_windows needs diag_dense_every"):
        _spec(tmp_path, rules=rules, diag_windows=[[100, 150]])
    # a rule step off the dense multiples is a static error, as with a single cadence
    off = {
        "cases": [
            {
                "label": "x",
                "when": {"metric": "diag:gate/p50", "steps": {"at": 410}, "op": ">", "value": 0},
            }
        ],
        "default": "y",
    }
    with pytest.raises(SpecError, match=r"reads steps \[410\] this arm does not record"):
        _spec(tmp_path, rules={"verdicts": {"off": off}}, diag_dense_every=25)
    with pytest.raises(SpecError, match="no rule reads a diag: window"):
        _spec(tmp_path, rules=None, diag_dense_every=25)


def test_p1_14_mixed_cadence_ids_unchanged_at_default(tmp_path: Path) -> None:
    """Writing the new keys at their defaults (0, []) gives the arm id of a spec without them."""
    rules = {"verdicts": {"w": WINDOWED}}
    plain = _spec(tmp_path, rules=rules)["a/s0"].arm_id
    defaults = _spec(tmp_path, rules=rules, diag_dense_every=0, diag_windows=[])["a/s0"].arm_id
    assert plain == defaults


@pytest.mark.usefixtures("restore_determinism")
def test_p1_14_mixed_cadence_run_records_the_schedule(
    tmp_path: Path,
    tiny_manifest: Path,  # noqa: F811
) -> None:
    """A tiny arm (8 steps, diag_every 4, dense every step in [5, 6]) records exactly the
    schedule."""
    cfg = load_config(REF_A1, [*TINY_HSN_OVERRIDES, *disk_overrides(tiny_manifest, steps=8)])
    params = protocol_params(
        {
            "name": "train_eval",
            "diag_every": 4,
            "diag_dense_every": 1,
            "diag_windows": [[5, 6]],
            "diag_docs": 4,
        }
    )
    arm = ArmRun(
        cfg=cfg,
        family="hsn",
        variant="pdg",
        variant_params={},
        params=params,
        run_dir=tmp_path / "arm",
        rules=parse_rules(None),
        constants={},
    )
    record = run_protocol(arm)
    assert [d["step"] for d in record["diagnostics"]] == [4, 5, 6, 8]
    assert record["step_time"]["excluded"] == {"warmup": 7}  # a run shorter than the warm-up


def test_p1_14_min_samples() -> None:
    def atom(**extra: Any) -> dict[str, Any]:
        return {
            "metric": "diag:gate/p50",
            "steps": {"from": 10, "to": 30},
            "agg": "median",
            "op": ">=",
            "value": 0.01,
            **extra,
        }

    def rule(**extra: Any) -> Any:
        return parse_rules(
            {
                "verdicts": {
                    "r": {"cases": [{"label": "open", "when": atom(**extra)}], "default": "shut"}
                }
            }
        )

    # serialised only when set, so stored rules (and early stops in arm ids) stay as they were
    assert "min_samples" not in rule().verdicts_json()["r"]["cases"][0]["when"]
    assert rule(min_samples=3).verdicts_json()["r"]["cases"][0]["when"]["min_samples"] == 3
    for bad, match in (
        ({"min_samples": 0}, ">= 1"),
        ({"min_samples": 2.5}, "integer"),
        ({"min_samples": 2, "steps": {"at": 10}}, "needs a window"),
        ({"min_samples": 2, "steps": None}, "needs a window"),
    ):
        raw = {k: v for k, v in atom(**bad).items() if v is not None}
        with pytest.raises(RuleError, match=match):
            parse_rules(
                {"verdicts": {"r": {"cases": [{"label": "x", "when": raw}], "default": "y"}}}
            )
    with pytest.raises(RuleError, match="needs a window"):
        parse_rules(
            {
                "verdicts": {
                    "r": {
                        "cases": [
                            {
                                "label": "x",
                                "when": {
                                    "metric": "trained:abs_h",
                                    "op": ">",
                                    "value": 0,
                                    "min_samples": 2,
                                },
                            }
                        ],
                        "default": "y",
                    }
                }
            }
        )
    # the static check counts the cadence's steps in the window
    every10 = Cadence.of(40, 10, 10, ("trained", "probe", "final"), "reads_state")
    check_rules(rule(min_samples=3), every10, {}, "hsn")
    with pytest.raises(
        RuleError, match="holds 3 of this arm's diag steps, fewer than min_samples 4"
    ):
        check_rules(rule(min_samples=4), every10, {}, "hsn")
    # at evaluation, fewer recorded samples read missing (n/a), with the count
    record = {"diagnostics": [{"step": 10, "gate/p50": 0.5}, {"step": 30, "gate/p50": 0.5}]}
    assert evaluate_verdicts(rule(), record, {}, "hsn")["r"]["label"] == "open"
    label = evaluate_verdicts(rule(min_samples=3), record, {}, "hsn")["r"]["label"]
    assert label == "n/a: diag:gate/p50 2 samples < min_samples 3"
    # the memory_open preset passes it to every window atom
    preset = parse_rules({"verdicts": {"m": {**WINDOWED, "min_samples": 5}}})
    atoms = preset.verdicts_json()["m"]["cases"][0]["when"]["all"]
    assert len(atoms) == 4 and all(a["min_samples"] == 5 for a in atoms)
