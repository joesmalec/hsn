"""P1.11 tests: the rule language (verdicts, presets, early stops, static checks).

The presets are checked on the real bridge run (fixture ``hsn_pdg_b32_lr1e-3_bridge_reads_state``):
it does not read state; the pre-registered "memory open" (every diagnostic at multiples of 500 from
step 500) fails on the one-step sample at step 500 (grad share of the cell 0.0084) and holds from
step 1,000, as recorded in the audit log; its dense cells are not apart. The reads-state preset
gives exactly the audit's verdict function on every count.
"""

from __future__ import annotations

import itertools
import json
import math
from typing import Any

import pytest
from hsn.experiments import probe
from hsn.experiments.rules import (
    Cadence,
    RuleError,
    check_early_stop,
    check_rules,
    evaluate_verdicts,
    parse_rules,
)
from p1_11_support import FIXTURES


def _record() -> dict[str, Any]:
    return {
        "evals": [
            {
                "step": s,
                "accuracy": a,
                "loss": 3.0 - a,
                "per_group": {"delayed/in_range": {"accuracy": a, "loss": 3.0 - a}},
            }
            for s, a in ((10, 0.1), (20, 0.2), (30, 0.3))
        ],
        "diagnostics": [{"step": s, "x": s / 100.0, "gate/p50": 0.05} for s in range(5, 31, 5)],
        "train": [
            {"step": s, "loss": 4.0 - s / 10, "accuracy": s / 30, "grad_norm": 1.0}
            for s in range(1, 31)
        ],
        "trained_state": {"rank_pr/answer": 1.5, "diversity_live/answer": 0.3},
        "probe": {"correct_n": 10, "correct_follow_frac": 0.6, "correct_unchanged_frac": 0.0},
        "final": {"train_loss": 1.0},
    }


def _verdict(
    rule: dict[str, Any],
    record: dict[str, Any] | None = None,
    family: str = "hsn",
    **constants: Any,
) -> str:
    rules = parse_rules({"verdicts": {"r": rule}})
    return str(evaluate_verdicts(rules, record or _record(), constants, family)["r"]["label"])


def _when(**atom: Any) -> dict[str, Any]:
    return {"cases": [{"label": "yes", "when": atom}], "default": "no"}


def test_p1_11_rule_evaluation() -> None:
    assert _verdict(_when(metric="eval:accuracy", steps={"at": 20}, op=">=", value=0.2)) == "yes"
    assert (
        _verdict(_when(metric="eval:delayed/in_range/loss", steps={"at": 30}, op="<", value=2.71))
        == "yes"
    )
    assert (
        _verdict(
            _when(metric="diag:x", steps={"from": 10, "to": 20}, agg="median", op="==", value=0.15)
        )
        == "yes"
    )
    assert (
        _verdict(
            _when(metric="diag:x", steps={"from": 10, "every": 10}, agg="all", op=">=", value=0.1)
        )
        == "yes"
    )
    assert _verdict(_when(metric="diag:x", agg="any", op=">", value=0.29)) == "yes"
    assert _verdict(_when(metric="diag:x", agg="all", op=">", value=0.06)) == "no"
    assert (
        _verdict(_when(metric="train:loss", steps={"from": 21}, agg="max", op="<=", value=1.9))
        == "yes"
    )
    assert (
        _verdict(_when(metric="eval:accuracy", steps={"at": 30}, op="between", value=[0.29, 0.31]))
        == "yes"
    )
    assert (
        _verdict(
            _when(metric="eval:accuracy", steps={"at": 10}, op="<=", value="chance + 0.08"),
            chance=0.0208,
        )
        == "yes"
    )
    assert (
        _verdict(
            _when(metric="eval:accuracy", steps={"at": 10}, op="<=", value="chance - 0.01"),
            chance=0.0208,
        )
        == "no"
    )
    assert _verdict(_when(metric="trained:rank_pr/answer", op=">=", value="lo"), lo=1.0) == "yes"
    assert _verdict(_when(metric="probe:correct_follow_frac", op="!=", value=0.6)) == "no"
    # combinators and verdict references (evaluated in dependency order)
    rules = parse_rules(
        {
            "verdicts": {
                "b": {
                    "cases": [
                        {
                            "label": "pass",
                            "when": {
                                "all": [
                                    {"metric": "verdict:a", "op": "==", "value": "expect"},
                                    {
                                        "not": {
                                            "metric": "eval:accuracy",
                                            "steps": {"at": 10},
                                            "op": ">",
                                            "value": 0.5,
                                        }
                                    },
                                ]
                            },
                        }
                    ],
                    "default": "fail",
                },
                "a": _when(metric="probe:correct_n", op=">", value=5),
            }
        }
    )
    out = evaluate_verdicts(rules, _record(), {"expect": "yes"}, "hsn")
    assert (out["a"]["label"], out["b"]["label"]) == ("yes", "pass")
    assert evaluate_verdicts(rules, _record(), {"expect": "no"}, "hsn")["b"]["label"] == "fail"
    # missing data, NaN, families, early-stopped arms
    missing = _verdict(
        _when(metric="diag:grad_share/cell", steps={"from": 10}, agg="all", op=">=", value=0.0)
    )
    assert missing.startswith("n/a: diag:grad_share/cell")
    nan = _record()
    nan["diagnostics"][2]["x"] = math.nan  # step 15
    assert _verdict(
        _when(metric="diag:x", steps={"from": 10, "to": 20}, agg="median", op=">", value=0.0), nan
    ).startswith("n/a")
    gated = {
        **_when(metric="trained:rank_pr/answer", op=">=", value=2.0),
        "applies_to": {"family": ["hsn"]},
    }
    assert _verdict(gated, family="gru") == "n/a (family gru)"
    stopped = {**_record(), "stopped": {"rule": "fail_early", "outcome": "fail_early", "step": 20}}
    assert (
        _verdict(_when(metric="probe:correct_n", op=">", value=5), stopped)
        == "stopped early: fail_early @ 20"
    )
    detail = evaluate_verdicts(
        parse_rules(
            {
                "verdicts": {
                    "r": _when(
                        metric="diag:x",
                        steps={"from": 10, "to": 20},
                        agg="median",
                        op=">=",
                        value=0.1,
                    )
                }
            }
        ),
        _record(),
        {},
        "hsn",
    )["r"]["detail"]
    assert (
        detail[0]["measured"] == pytest.approx(0.15)
        and detail[0]["steps"] == [10, 15, 20]
        and detail[0]["n"] == 3
    )


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"verdicts": {"r": _when(metric="nope:x", op=">", value=1)}}, "unknown metric source"),
        ({"verdicts": {"r": _when(metric="diag:x", op=">", value=1)}}, "needs agg"),
        ({"verdicts": {"r": _when(metric="diag:x", agg="mode", op=">", value=1)}}, "unknown agg"),
        ({"verdicts": {"r": _when(metric="probe:x", agg="all", op=">", value=1)}}, "take no steps"),
        (
            {"verdicts": {"r": _when(metric="eval:f1", steps={"at": 1}, op=">", value=1)}},
            "eval metric",
        ),
        ({"verdicts": {"r": _when(metric="probe:x", op="~", value=1)}}, "unknown op"),
        ({"verdicts": {"r": _when(metric="probe:x", op="between", value=1)}}, "between"),
        ({"verdicts": {"r": _when(metric="probe:x", op=">", value=None)}}, "number or a string"),
        (
            {
                "verdicts": {
                    "r": _when(
                        metric="diag:x", steps={"at": 1, "every": 2}, agg="all", op=">", value=1
                    )
                }
            },
            "steps.at",
        ),
        ({"verdicts": {"r": {"cases": [], "default": "x"}}}, "non-empty list of cases"),
        (
            {
                "verdicts": {
                    "a": _when(metric="verdict:b", op="==", value="x"),
                    "b": _when(metric="verdict:a", op="==", value="x"),
                }
            },
            "cycle",
        ),
        ({"verdicts": {"a": _when(metric="verdict:zz", op="==", value="x")}}, "unknown verdicts"),
        ({"verdicts": {"a": {"preset": "nope"}}}, "unknown preset"),
        ({"verdicts": {"a": {"preset": "memory_open", "colour": 1}}}, "takes no parameters"),
        (
            {
                "early_stop": [
                    {
                        "name": "s",
                        "at_steps": [],
                        "outcome": "x",
                        "when": {"metric": "probe:x", "op": ">", "value": 1},
                    }
                ]
            },
            "non-empty",
        ),
        ({"surprise": {}}, "unknown rules keys"),
    ],
)
def test_p1_11_rule_parse_errors(raw: dict[str, Any], match: str) -> None:
    with pytest.raises(RuleError, match=match):
        parse_rules(raw)


def _bridge_record() -> dict[str, Any]:
    data = json.loads(
        (FIXTURES / "reads_state" / "hsn_pdg_b32_lr1e-3_bridge_reads_state.json").read_text()
    )
    evals = [
        {
            "step": e["step"],
            "accuracy": e["dev_answer_accuracy"]["delayed/in_range"],
            "loss": e["dev_answer_loss"]["delayed/in_range"],
            "per_group": {
                "delayed/in_range": {
                    "accuracy": e["dev_answer_accuracy"]["delayed/in_range"],
                    "loss": e["dev_answer_loss"]["delayed/in_range"],
                }
            },
        }
        for e in data["evals"]
    ]
    return {
        "evals": evals,
        "diagnostics": data["diagnostics"],
        "trained_state": data["trained_state"],
        "probe": probe.summarize(data["probes"]),
    }


def test_p1_11_rule_presets_on_the_bridge() -> None:
    record = _bridge_record()
    rules = parse_rules(
        {
            "verdicts": {
                "reads_state": {"preset": "reads_state"},
                "memory_open": {"preset": "memory_open"},
                "memory_open_from_1000": {"preset": "memory_open", "from_step": 1000},
                "memory_open_median": {"preset": "memory_open", "agg": "median"},
                "cells_apart": {"preset": "cells_apart"},
            }
        }
    )
    out = {k: v["label"] for k, v in evaluate_verdicts(rules, record, {}, "hsn").items()}
    assert out == {
        "reads_state": "does not read state",
        "memory_open": "shut",  # grad_share/cell 0.0084 at step 500
        "memory_open_from_1000": "open",
        "memory_open_median": "open",
        "cells_apart": "not apart",  # rank_pr 1.0: the dense cells are identical
    }
    detail = evaluate_verdicts(rules, record, {}, "hsn")["memory_open"]["detail"]
    assert detail[0]["steps"] == [500, 1000, 1500, 2000, 2500, 3000]
    assert min(detail[1]["samples"]) == pytest.approx(0.0084, abs=5e-5)
    gru = {k: v["label"] for k, v in evaluate_verdicts(rules, record, {}, "gru").items()}
    assert gru["memory_open"] == "n/a (family gru)" and gru["cells_apart"] == "n/a (family gru)"
    assert gru["reads_state"] == "does not read state"
    expanded = rules.verdicts_json()["memory_open"]
    assert expanded["applies_to"] == {"family": ["hsn"]}
    assert expanded["cases"][0]["when"]["all"][0]["steps"] == {"from": 500, "every": 500}


def test_p1_11_reads_state_preset_is_the_audit_verdict() -> None:
    rules = parse_rules({"verdicts": {"v": {"preset": "reads_state"}}})
    for n, follow, unchanged in itertools.product(range(0, 11), range(0, 11), range(0, 11)):
        if follow + unchanged > n:
            continue
        record = {
            "probe": {
                "correct_n": n,
                "correct_follow_frac": follow / n if n else None,
                "correct_unchanged_frac": unchanged / n if n else None,
            }
        }
        label = evaluate_verdicts(rules, record, {}, "hsn")["v"]["label"]
        assert label == probe.verdict(follow, unchanged, n), (n, follow, unchanged)


def test_p1_11_rule_static_checks() -> None:
    cadence = Cadence.of(3000, 500, 500, ("trained", "probe", "final"))
    assert cadence.eval_steps == (500, 1000, 1500, 2000, 2500, 3000)

    def check(raw: dict[str, Any], c: Cadence = cadence, **constants: Any) -> None:
        check_rules(parse_rules(raw), c, constants, "hsn")

    stop = {"name": "s", "at_steps": [1500], "outcome": "fail_early"}
    ok = {
        "early_stop": [
            {
                **stop,
                "when": {
                    "all": [
                        {
                            "metric": "eval:accuracy",
                            "steps": {"at": 1500},
                            "op": "<=",
                            "value": "chance + 0.01",
                        },
                        {
                            "metric": "diag:gate/p50",
                            "steps": {"from": 1000, "to": 1500},
                            "agg": "median",
                            "op": "<",
                            "value": 0.01,
                        },
                    ]
                },
            }
        ]
    }
    check(ok, chance=0.02)
    cases = [
        (
            {
                "early_stop": [
                    {
                        **stop,
                        "when": {
                            "metric": "eval:accuracy",
                            "steps": {"at": 2000},
                            "op": "<",
                            "value": 1,
                        },
                    }
                ]
            },
            "after the rule's step",
        ),
        (
            {
                "early_stop": [
                    {
                        **stop,
                        "at_steps": [1400],
                        "when": {
                            "metric": "eval:accuracy",
                            "steps": {"at": 1000},
                            "op": "<",
                            "value": 1,
                        },
                    }
                ]
            },
            "not an evaluation step",
        ),
        (
            {
                "early_stop": [
                    {
                        **stop,
                        "at_steps": [3000],
                        "when": {
                            "metric": "eval:accuracy",
                            "steps": {"at": 3000},
                            "op": "<",
                            "value": 1,
                        },
                    }
                ]
            },
            "not before the last step",
        ),
        (
            {
                "early_stop": [
                    {**stop, "when": {"metric": "probe:correct_n", "op": ">", "value": 1}}
                ]
            },
            "step series only",
        ),
        (
            {
                "verdicts": {
                    "r": _when(
                        metric="diag:gate/p50",
                        steps={"from": 250, "every": 250},
                        agg="all",
                        op=">",
                        value=0,
                    )
                }
            },
            "does not record",
        ),
        (
            {"verdicts": {"r": _when(metric="eval:accuracy", steps={"at": 700}, op=">", value=0)}},
            "does not record",
        ),
        (
            {"verdicts": {"r": _when(metric="overfit:first_ge_0.99", op="<", value=10)}},
            "writes no overfit",
        ),
        (
            {
                "verdicts": {
                    "r": _when(metric="eval:accuracy", steps={"at": 500}, op=">", value="label")
                }
            },
            "not a number",
        ),
        (
            {
                "verdicts": {
                    "r": _when(metric="eval:accuracy", steps={"at": 500}, op=">", value="nope + 1")
                }
            },
            "unknown constant",
        ),
        (
            {
                "verdicts": {
                    "r": _when(metric="diag:x", steps={"from": 3500}, agg="all", op=">", value=0)
                }
            },
            "selects no step",
        ),
    ]
    for raw, match in cases:
        with pytest.raises(RuleError, match=match):
            check(raw)
    no_diag = Cadence.of(3000, 500, 0, ("final",))
    with pytest.raises(RuleError, match="records none"):
        check({"verdicts": {"r": _when(metric="diag:x", agg="all", op=">", value=0)}}, no_diag)
    # rules outside the arm's family are not checked against it
    gated = {
        "verdicts": {
            "r": {
                **_when(metric="trained:rank_pr/answer", op=">", value=1),
                "applies_to": {"family": ["gru"]},
            }
        }
    }
    check_rules(parse_rules(gated), no_diag, {}, "hsn")


def test_p1_11_early_stop_evaluation() -> None:
    rules = parse_rules(
        {
            "early_stop": [
                {
                    "name": "low",
                    "at_steps": [20],
                    "outcome": "fail_early",
                    "when": {
                        "all": [
                            {
                                "metric": "eval:accuracy",
                                "steps": {"at": 20},
                                "op": "<=",
                                "value": "chance + 0.2",
                            },
                            {
                                "metric": "diag:x",
                                "steps": {"from": 10, "to": 20},
                                "agg": "median",
                                "op": "<",
                                "value": 0.2,
                            },
                        ]
                    },
                },
                {
                    "name": "gate",
                    "at_steps": [10, 20],
                    "outcome": "shut",
                    "when": {
                        "metric": "diag:gate/p50",
                        "steps": {"at": 10},
                        "op": "<",
                        "value": 0.01,
                    },
                },
            ]
        }
    )
    record = _record()
    assert check_early_stop(rules, record, 15, {"chance": 0.0}, "hsn") == (None, [])
    stop, log = check_early_stop(rules, record, 20, {"chance": 0.0}, "hsn")
    assert (
        stop is not None
        and stop["rule"] == "low"
        and stop["outcome"] == "fail_early"
        and stop["step"] == 20
    )
    assert [v["result"] for v in stop["values"]] == [True, True]
    assert stop["values"][1]["measured"] == pytest.approx(0.15)
    # data past the stop step are never read
    future = _record()
    future["evals"][1]["accuracy"] = 0.9
    stop2, log2 = check_early_stop(rules, future, 20, {"chance": 0.0}, "hsn")
    assert stop2 is None and [e["rule"] for e in log2] == ["low", "gate"]
    # a missing atom does not stop the run, and is logged
    gone = _record()
    gone["diagnostics"] = []
    stop3, log3 = check_early_stop(rules, gone, 20, {"chance": 0.0}, "hsn")
    assert stop3 is None and log3[0]["result"] is None
