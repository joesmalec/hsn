"""P1.15 tests: the rule presets (``memory_writes``, ``reads_state_v3``, ``binds_keys``, and
``min_vocab`` on ``reads_state_v2``).

The existing presets expand exactly as on the base commit (``tests/fixtures/p1_15/
preset_expansions.json``, captured with 6e609b8's library); ``min_vocab`` absent leaves
``reads_state_v2`` as it was; the v3 presets give the probe's own verdicts; ``memory_writes`` on the
stored runs' medians (steps 1,000 to 1,500) calls every run that read state "writes" and every dead
one "no writes"; its window must hold enough diagnostics at enqueue.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import pytest
from hsn.experiments import probe
from hsn.experiments.rules import Cadence, RuleError, check_rules, evaluate_verdicts, parse_rules
from p1_15_support import FIXTURES


def test_p1_15_existing_presets_unchanged() -> None:
    data = json.loads((FIXTURES / "preset_expansions.json").read_text())
    for name, raw in data["cases"].items():
        got = parse_rules({"verdicts": {name: raw}}).to_json()["verdicts"][name]
        assert json.loads(json.dumps(got)) == data["expansions"][name], name


def _label(raw: dict[str, Any], record: dict[str, Any], family: str = "hsn") -> str:
    rules = parse_rules({"verdicts": {"v": raw}})
    return str(
        evaluate_verdicts(rules, {"protocol": "reads_state", **record}, {}, family)["v"]["label"]
    )


def test_p1_15_min_vocab_on_reads_state_v2() -> None:
    plain = parse_rules({"verdicts": {"v": {"preset": "reads_state_v2"}}}).to_json()
    same = parse_rules({"verdicts": {"v": {"preset": "reads_state_v2"}}}).to_json()
    assert plain == same
    raw = {"preset": "reads_state_v2", "min_vocab": 20}
    good = {"correct_n": 40, "correct_follow_frac": 0.7, "correct_unchanged_frac": 0.05}
    assert _label(raw, {"probe_v2": {**good, "vocab_size": 5}}) == "vocabulary too small"
    assert _label(raw, {"probe_v2": {**good, "vocab_size": 48}}) == "reads state"
    assert _label(raw, {"probe_v2": {**good, "vocab_size": 5, "correct_n": 0}}) == (
        "no matched probes"
    )
    # without min_vocab exps 132 / 133's small vocabularies still read as before
    assert _label({"preset": "reads_state_v2"}, {"probe_v2": {**good, "vocab_size": 5}}) == (
        "reads state"
    )


def test_p1_15_reads_state_v3_preset_is_the_probe_verdict() -> None:
    raw = {"preset": "reads_state_v3"}
    grid = itertools.product(
        (0, 4, 25), (0, 10, 25), (1, 19, 48), (0.1, 0.3, 0.8), (0.0, 0.05, 0.5)
    )
    for correct_later, correct, vocab, follow, unchanged in grid:
        correct = min(correct, correct_later)
        want = probe.verdict_v3_value(
            round(follow * correct),
            round(unchanged * correct),
            correct,
            vocab,
            correct_later=correct_later,
        )
        fields = {
            "correct_later_n": correct_later,
            "vocab_size": vocab,
            "value/correct_n": correct,
            "value/follow_frac": round(follow * correct) / correct if correct else None,
            "value/unchanged_frac": round(unchanged * correct) / correct if correct else None,
        }
        assert _label(raw, {"probe_v3": fields}) == want, fields


def test_p1_15_binds_keys_preset_is_the_probe_verdict() -> None:
    raw = {"preset": "binds_keys"}
    for correct, follow, unchanged in itertools.product(
        (0, 10, 25), (0.1, 0.3, 0.8), (0.0, 0.05, 0.5)
    ):
        want = probe.verdict_v3_swap(round(follow * correct), round(unchanged * correct), correct)
        fields = {
            "swap/correct_n": correct,
            "swap/follow_frac": round(follow * correct) / correct if correct else None,
            "swap/unchanged_frac": round(unchanged * correct) / correct if correct else None,
        }
        assert _label(raw, {"probe_v3": fields}) == want, fields


#: medians of gate/write_rate and update_ratio/cell over steps 1,000 to 1,500 of the stored runs
#: with a write rate (exps 128 to 174, read-only scan of 8 Nov 2026); True = the run wrote (it read
#: state, or learned slowly, or writes in the regime of seeds that read state later)
STORED = {
    128: (0.0340, 0.0024, True),  # d=128 at R*: learning slowly (reads state at 9k, exp 173)
    129: (0.0151, 0.0007, False),  # at chance throughout; memory_open said "open"
    130: (0.0761, 0.0040, True),
    132: (0.1418, 0.0014, True),  # the sparse-write regime; its 9k seed read state
    133: (0.1173, 0.0025, True),
    135: (0.2899, 0.0027, True),
    136: (0.1424, 0.0024, True),
    138: (0.0844, 0.0015, True),
    139: (0.1792, 0.0017, True),
    155: (0.1290, 0.0015, True),
    156: (0.0609, 0.0038, True),
    170: (0.0002, 0.0017, False),  # the starved cell
    171: (0.0022, 0.0021, False),  # before its divergence
    172: (0.1536, 0.0030, True),
    173: (0.0449, 0.0023, True),
    174: (0.0275, 0.0007, False),  # memory_open said "open"
}


def test_p1_15_memory_writes_on_the_stored_medians() -> None:
    raw = {"preset": "memory_writes"}
    for exp, (rate, ratio, writes) in STORED.items():
        diagnostics = [
            {"step": s, "gate/write_rate": rate, "update_ratio/cell": ratio}
            for s in (1000, 1250, 1500)
        ]
        label = _label(raw, {"diagnostics": diagnostics})
        assert label == ("writes" if writes else "no writes"), exp
    assert _label(raw, {"diagnostics": []}, family="gru") == "n/a (family gru)"
    short = [{"step": 1000, "gate/write_rate": 0.2, "update_ratio/cell": 0.003}]
    assert _label(raw, {"diagnostics": short}).startswith("n/a")  # fewer than min_samples


def test_p1_15_memory_writes_static_checks() -> None:
    rules = parse_rules({"verdicts": {"w": {"preset": "memory_writes"}}})
    sources = ("trained", "probe", "final")
    check_rules(rules, Cadence.of(9000, 500, 250, sources, "reads_state"), {}, "hsn")
    with pytest.raises(RuleError, match="min_samples"):
        check_rules(rules, Cadence.of(9000, 500, 500, sources, "reads_state"), {}, "hsn")
    with pytest.raises(RuleError, match="selects no step"):
        check_rules(rules, Cadence.of(600, 100, 25, sources, "reads_state"), {}, "hsn")
    check_rules(rules, Cadence.of(600, 100, 25, sources, "reads_state"), {}, "gru")  # n/a there
    with pytest.raises(RuleError, match="to_step"):
        parse_rules(
            {"verdicts": {"w": {"preset": "memory_writes", "from_step": 900, "to_step": 800}}}
        )
