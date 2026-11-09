"""P1.16 tests: a failed arm's re-scored kept checkpoint in the rules (the ``partial:`` source and
the ``(from step X: <label>)`` suffix, never a pass), the report's failure cell, and the
``memory_writes_run`` preset (any sliding window of the run)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from hsn.experiments.report import _failure_cell
from hsn.experiments.rules import (
    MEMORY_WRITES_RUN_GATE,
    MEMORY_WRITES_RUN_RATIO,
    RuleError,
    evaluate_verdicts,
    parse_rules,
    partial_record,
)

FAILURE = {"reason": "gradient spikes or non-finite gradients (P1.12 stop)", "last_step": 5243}


def _record(partial: dict[str, Any] | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "protocol": "reads_state",
        "failure": FAILURE,
        "evals": [{"step": s, "accuracy": a} for s, a in ((4000, 0.70), (5000, 0.755))],
        "train": [{"step": s, "grad_norm": 0.5} for s in range(1, 5244)],
        "diagnostics": [{"step": s} for s in range(250, 5250, 250)],
        "final": {"accuracy": 0.10},  # the last evaluation before the failure: not a final result
    }
    if partial is not None:
        record["partial"] = partial
    return record


RULES = parse_rules(
    {
        "verdicts": {
            "good": {
                "cases": [
                    {
                        "label": "above 0.7",
                        "when": {"metric": "final:accuracy", "op": ">", "value": 0.7},
                    }
                ],
                "default": "at or below 0.7",
            },
            "late": {
                "cases": [
                    {
                        "label": "high at 5000",
                        "when": {
                            "metric": "eval:accuracy",
                            "steps": {"at": 5000},
                            "op": ">",
                            "value": 0.75,
                        },
                    }
                ],
                "default": "not",
            },
            "after": {
                "cases": [
                    {
                        "label": "evaluated at 6000",
                        "when": {
                            "metric": "eval:accuracy",
                            "steps": {"at": 6000},
                            "op": ">=",
                            "value": 0.0,
                        },
                    }
                ],
                "default": "no",
            },
            "src": {
                "cases": [
                    {
                        "label": "from 5000",
                        "when": {"metric": "partial:from_step", "op": "==", "value": 5000},
                    }
                ],
                "default": "other",
            },
            "ref": {
                "cases": [
                    {
                        "label": "follows good",
                        "when": {"metric": "verdict:good", "op": "==", "value": "above 0.7"},
                    }
                ],
                "default": "does not",
            },
        }
    }
)


def test_p1_16_failed_verdicts_from_partial() -> None:
    """Every rule of a failed record reads ``failed: …`` and, with a re-scored kept checkpoint,
    ``(from step X: <the rule on that checkpoint>)``; verdict references see the partial labels."""
    partial = {"from_step": 5000, "final": {"accuracy": 0.755}, "trained": {}, "probes": {}}
    out = evaluate_verdicts(RULES, _record(partial), {}, "hsn")
    base = "failed: gradient spikes or non-finite gradients (P1.12 stop) @ 5243"
    labels = {k: v["label"] for k, v in out.items()}
    assert labels["good"] == f"{base} (from step 5000: above 0.7)"
    assert labels["late"] == f"{base} (from step 5000: high at 5000)"
    assert labels["src"] == f"{base} (from step 5000: from 5000)"
    assert labels["ref"] == f"{base} (from step 5000: follows good)"
    assert out["after"]["label"].startswith(f"{base} (from step 5000: n/a")
    assert all(v["label"].startswith("failed: ") for v in out.values())  # never a pass
    assert out["good"]["partial"]["from_step"] == 5000
    # without a kept checkpoint: P1.14's label alone
    plain = evaluate_verdicts(RULES, _record(None), {}, "hsn")
    assert {v["label"] for v in plain.values()} == {base}


def test_p1_16_partial_record_view() -> None:
    partial = {
        "from_step": 5000,
        "final": {"accuracy": 0.755},
        "trained": {"gate/write_rate": 0.1},
        "probes": {"probe_v3": {"swap/follow_frac": 0.0}},
    }
    view = partial_record(_record(partial))
    assert view is not None
    assert "failure" not in view and view["partial"]["from_step"] == 5000
    assert view["final"] == {"accuracy": 0.755} and view["trained_state"]["gate/write_rate"] == 0.1
    assert view["probe_v3"] == {"swap/follow_frac": 0.0} and view["probe"] is None
    assert max(e["step"] for e in view["train"]) == 5000
    assert max(e["step"] for e in view["diagnostics"]) == 5000
    assert partial_record(_record(None)) is None


def test_p1_16_partial_source_static_check() -> None:
    """``partial:`` is a scalar source of the training protocols (the static checks accept it)."""
    from hsn.experiments.rules import Cadence, check_rules

    cadence = Cadence.of(9000, 500, 250, ("trained", "probe", "final", "partial"), "reads_state")
    rules = parse_rules(
        {
            "verdicts": {
                "p": {
                    "cases": [
                        {
                            "label": "x",
                            "when": {"metric": "partial:accuracy", "op": ">", "value": 0},
                        }
                    ],
                    "default": "y",
                }
            }
        }
    )
    check_rules(rules, cadence, {}, "hsn")


def test_p1_16_spec_protocols_write_partial() -> None:
    from hsn.experiments.spec import protocol_params, protocol_sources

    for name in ("reads_state", "train_eval"):
        assert "partial" in protocol_sources(protocol_params({"name": name}))
    assert "partial" not in protocol_sources(protocol_params({"name": "overfit"}))


def test_p1_16_report_failure_cell() -> None:
    assert _failure_cell(None, None) == ""
    assert _failure_cell(FAILURE, None) == f"{FAILURE['reason']} @ 5243"
    cell = _failure_cell(FAILURE, {"from_step": 5000, "final": {"dev_accuracy": 0.7553}})
    assert cell == f"{FAILURE['reason']} @ 5243; partial from step 5000 (dev 0.7553)"


# --------------------------------------------------------------------------------------------------
# memory_writes_run
# --------------------------------------------------------------------------------------------------


def _writes(rates: dict[int, tuple[float, float]]) -> dict[str, Any]:
    return {
        "protocol": "reads_state",
        "diagnostics": [
            {"step": s, "gate/p90": g, "update_ratio/cell": u} for s, (g, u) in rates.items()
        ],
    }


def test_p1_16_memory_writes_run_reads_late_writers() -> None:
    """Strong gates in one part of the run and a moving cell in another read "writes" (the
    mlp-scorer runs open their gates early and update later); one noisy sample does not; a run
    whose gates never open reads "no writes"; the GRU reads n/a."""
    rules = parse_rules({"verdicts": {"w": {"preset": "memory_writes_run", "to_step": 8750}}})
    (case,) = rules.verdicts["w"].cases
    assert case.label == "writes" and case.when.kind == "all"
    gates, moves = case.when.parts
    assert gates.kind == moves.kind == "any"
    assert len(gates.parts) == len(moves.parts) == len(range(500, 8751, 500))
    steps = range(250, 9001, 250)
    g, u = MEMORY_WRITES_RUN_GATE * 2, MEMORY_WRITES_RUN_RATIO * 2
    late = _writes({s: (g if s >= 6000 else 0.0, u if s >= 6000 else 0.0) for s in steps})
    split = _writes({s: (g if s <= 1000 else 0.0, u if s >= 5000 else 0.0) for s in steps})
    never = _writes({s: (0.0, u) for s in steps})
    one = _writes({s: (g if s == 4000 else 0.0, u) for s in steps})
    labels = {
        name: evaluate_verdicts(rules, rec, {}, "hsn")["w"]["label"]
        for name, rec in (("late", late), ("split", split), ("never", never), ("one", one))
    }
    assert labels == {"late": "writes", "split": "writes", "never": "no writes", "one": "no writes"}
    assert evaluate_verdicts(rules, late, {}, "gru")["w"]["label"].startswith("n/a")


def test_p1_16_memory_writes_run_parameters() -> None:
    with pytest.raises(RuleError, match="to_step"):
        parse_rules({"verdicts": {"w": {"preset": "memory_writes_run"}}})
    with pytest.raises(RuleError):
        parse_rules({"verdicts": {"w": {"preset": "memory_writes_run", "to_step": 100}}})
    with pytest.raises(RuleError, match="gate_key"):
        parse_rules(
            {"verdicts": {"w": {"preset": "memory_writes_run", "to_step": 900, "gate_key": "x"}}}
        )
    assert (MEMORY_WRITES_RUN_GATE, MEMORY_WRITES_RUN_RATIO) == (0.095, 0.0027)


def test_p1_16_preset_expansions_pinned() -> None:
    """memory_writes_run expands as calibrated, and memory_writes as on 80b893d."""
    from pathlib import Path

    data = json.loads(
        (Path(__file__).parent / "fixtures" / "p1_16" / "preset_expansions.json").read_text()
    )
    for name, raw in data["cases"].items():
        got = parse_rules({"verdicts": {name: raw}}).to_json()["verdicts"][name]
        assert json.loads(json.dumps(got)) == data["expansions"][name], name
