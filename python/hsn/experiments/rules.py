"""Pre-registered rules of a sweep: verdicts and early stops (P1.11).

The rules are part of the sweep spec, stored in the database at ``enqueue`` (before any run) in
their expanded form. They are experiment rules registered before the run, not gate thresholds
(those live in ``configs/metrics_freeze.json``).

**Condition.** An *atom* compares one measured number (or label) with a value::

    {metric: "diag:grad_share/cell", steps: {from: 400, to: 600, every: 25}, agg: median,
     op: ">=", value: 0.01}

* ``metric`` is ``<source>:<key>``. Step series: ``eval:`` (``accuracy``, ``loss`` or
  ``<task>/<variant>/accuracy|loss`` of the dev evaluations; P1.15: ``split/<g>/accuracy|loss``
  with ``g`` one of ``first``, ``later``, ``later/1`` to ``later/3``, the first/later split of
  reads_state runs, :mod:`hsn.experiments.split`), ``diag:<key>`` (the diagnostics
  records, :mod:`hsn.experiments.diagnostics`), ``train:loss|accuracy|grad_norm`` (every step).
  Scalars: ``trained:<key>`` (the trained model's state statistics), ``probe:<key>`` (the edit
  probe, :func:`hsn.experiments.probe.summarize`), ``probe_v2:<key>`` (P1.14: the
  vocabulary-matched edit of reads_state version 2, :func:`hsn.experiments.probe.summarize_v2`),
  ``probe_v3:<key>`` (P1.15: the key-addressed edits of reads_state version 3,
  :func:`hsn.experiments.probe.summarize_v3`: ``value/...`` and ``swap/...`` keys),
  ``overfit:<key>``, ``final:<key>`` (the last
  evaluation flattened, ``train_loss``, ``steps``) and ``verdict:<rule>`` (another rule's label).
* ``steps`` (series only) selects samples: ``{at: n}`` or ``{from: a, to: b, every: k}`` (every
  bound optional; ``every`` keeps the multiples of ``k``). Without ``steps`` every sample counts.
* ``agg`` (series only, required unless ``steps.at``): ``all``/``any`` apply the comparison to every
  sample; ``median``, ``mean``, ``min``, ``max``, ``first``, ``last`` compare one aggregate.
* ``min_samples`` (P1.14; series atoms whose ``steps`` has ``from`` and/or ``to``, optional): the
  window must hold at least that many samples. The static check refuses a cadence that puts fewer
  steps in it; at evaluation fewer recorded samples (a step P1.12's guard skipped, a short run) make
  the atom missing (``n/a: <metric> 3 samples < min_samples 5``). Unset: no minimum (as before).
* ``op``: ``>=``, ``>``, ``<=``, ``<``, ``==``, ``!=``, or ``between`` with ``value: [lo, hi]``
  (inclusive).
* ``value``: a number, a string, a constant name, or ``"<constant> + <number>"`` /
  ``"<constant> - <number>"``. Constants come from the sweep and the arm (the arm's win). A string
  that names no constant is a literal (for comparing labels with ``==`` / ``!=``); an ordered
  comparison with a non-number raises.

Atoms combine with ``{all: [...]}``, ``{any: [...]}`` and ``{not: ...}``. Evaluation is
three-valued: a selected sample that is absent or NaN makes the atom *missing*; ``all`` is False if
any part is False, else missing if any part is missing; ``any`` likewise with True.

**Verdict rule.** ``{cases: [{label, when}], default, applies_to: {family: [...], protocol: [...]},
group: {min_agree: n}}``: the first case whose condition holds gives the label, the default
otherwise. An arm outside ``applies_to`` reads ``n/a (family <f>)`` or ``n/a (protocol <p>)`` (never
a pass or a fail); a missing atom in a case reached reads ``n/a: <metric> missing ...``; an arm that
stopped early reads ``stopped early: <outcome> @ <step>`` for every verdict, and an arm that failed
(P1.14: a partial record with a ``failure`` block) ``failed: <reason> @ <last step>``.
``group.min_agree`` is read by ``report`` at seed-group level ("counts as a finding" when that many
seeds share a label).
Presets (:data:`PRESETS`) expand into this form at enqueue: ``reads_state``, ``reads_state_v2``
(P1.14), ``memory_open``, ``cells_apart``.

**Early stop.** ``{name, at_steps: [n, ...], outcome, when, applies_to}``: evaluated by the trainer
callback right after step ``n``'s evaluation and diagnostics; when ``when`` holds the run stops with
``outcome``. Early-stop conditions read step series only, never past ``n``. A missing atom does not
stop the run; the evaluation is logged either way.

**Static checks** (:func:`check_rules`, at enqueue, per arm) raise on an unknown source, a source
the protocol does not produce, an early stop that reads the future or a scalar source, an
``at_steps`` entry that is not an evaluation step or not before the last step, a selection the
cadence does not produce, a window with fewer steps than its ``min_samples``, a missing constant and
an ordered comparison with a non-number.

**Mixed diagnostic cadence (P1.14).** With a protocol's ``diag_dense_every`` set,
:func:`rule_windows` gives the windows the arm's rules read (every ``diag:`` atom of the verdicts
and early stops that apply to it: ``steps.at`` a one-step window, a span its ``[from, to]``, clipped
to the run); :func:`diag_schedule` samples every ``diag_dense_every`` steps inside them and every
``diag_every`` steps elsewhere. An atom with no ``steps`` would read a mix of dense and sparse
samples, and is refused in that mode.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AGGS",
    "MIN_SAMPLES_DOC",
    "OPS",
    "PRESETS",
    "SERIES_SOURCES",
    "SOURCES",
    "Atom",
    "Cadence",
    "EarlyStopRule",
    "RuleError",
    "RuleSet",
    "VerdictRule",
    "check_early_stop",
    "check_rules",
    "diag_schedule",
    "evaluate_verdicts",
    "parse_rules",
    "resolved_early_stop",
    "rule_windows",
]

SERIES_SOURCES = ("eval", "diag", "train")
SCALAR_SOURCES = ("trained", "probe", "probe_v2", "probe_v3", "overfit", "final", "verdict")
SOURCES = SERIES_SOURCES + SCALAR_SOURCES
AGGS = ("all", "any", "median", "mean", "min", "max", "first", "last")
OPS = (">=", ">", "<=", "<", "==", "!=", "between")
_ORDERED = (">=", ">", "<=", "<", "between")
_CONST_EXPR = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:([+-])\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?))?\s*$"
)
EVAL_METRICS = ("accuracy", "loss")
#: P1.15: ``eval:split/<group>/<metric>`` reads an evaluation's first/later split
#: (:mod:`hsn.experiments.split`); no task is named ``split``
SPLIT_PREFIX = "split/"
#: P1.14: the minimum the docs recommend for a windowed rule (``min_samples`` is set per rule)
MIN_SAMPLES_DOC = 5
TRAIN_METRICS = ("loss", "accuracy", "grad_norm")


class RuleError(ValueError):
    """A rule that cannot be parsed, or that does not fit the arm it is registered for."""


# --------------------------------------------------------------------------------------------------
# Parsed form
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Steps:
    at: int | None = None
    start: int | None = None
    stop: int | None = None
    every: int | None = None

    def to_json(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, value in (("at", self.at), ("from", self.start), ("to", self.stop)):
            if value is not None:
                out[key] = value
        if self.every is not None:
            out["every"] = self.every
        return out

    def select(self, available: Iterable[int]) -> list[int]:
        steps = sorted(set(available))
        if self.at is not None:
            return [self.at]
        out = []
        for s in steps:
            if self.start is not None and s < self.start:
                continue
            if self.stop is not None and s > self.stop:
                continue
            if self.every is not None and s % self.every != 0:
                continue
            out.append(s)
        return out

    def wanted(self, available: Iterable[int], last: int) -> list[int]:
        """The steps this selection needs. ``at`` and ``every`` name their steps whatever the run
        recorded (every multiple of ``every`` from ``from`` to ``to``, or to ``last``), so that a
        cadence that misses one is detected; a plain window takes the recorded steps in it."""
        if self.at is not None:
            return [self.at]
        if self.every is not None:
            lo = self.every if self.start is None else max(self.start, self.every)
            first = -(-lo // self.every) * self.every
            hi = last if self.stop is None else min(self.stop, last)
            return list(range(first, hi + 1, self.every))
        return self.select(available)


@dataclass(frozen=True)
class Atom:
    source: str
    key: str
    op: str
    value: Any
    steps: Steps | None = None
    agg: str | None = None
    min_samples: int | None = None  # P1.14

    @property
    def metric(self) -> str:
        return f"{self.source}:{self.key}"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"metric": self.metric}
        if self.steps is not None:
            out["steps"] = self.steps.to_json()
        if self.agg is not None:
            out["agg"] = self.agg
        out["op"] = self.op
        out["value"] = list(self.value) if isinstance(self.value, tuple) else self.value
        # only when set (P1.14): stored rules and arm ids stay as they were
        if self.min_samples is not None:
            out["min_samples"] = self.min_samples
        return out


@dataclass(frozen=True)
class Combo:
    kind: str  # all | any | not
    parts: tuple[Cond, ...]

    def to_json(self) -> dict[str, Any]:
        if self.kind == "not":
            return {"not": self.parts[0].to_json()}
        return {self.kind: [p.to_json() for p in self.parts]}


Cond = Atom | Combo


@dataclass(frozen=True)
class Scope:
    """``applies_to``: the model families and protocols a rule is for (None: every one)."""

    families: tuple[str, ...] | None = None
    protocols: tuple[str, ...] | None = None

    def excludes(self, family: str, protocol: str | None) -> str | None:
        """Why the rule does not apply to this arm (None: it applies)."""
        if self.families is not None and family not in self.families:
            return f"family {family}"
        if self.protocols is not None and protocol is not None and protocol not in self.protocols:
            return f"protocol {protocol}"
        return None

    def to_json(self) -> dict[str, list[str]] | None:
        out: dict[str, list[str]] = {}
        if self.families is not None:
            out["family"] = list(self.families)
        if self.protocols is not None:
            out["protocol"] = list(self.protocols)
        return out or None


@dataclass(frozen=True)
class Case:
    label: str
    when: Cond


@dataclass(frozen=True)
class VerdictRule:
    cases: tuple[Case, ...]
    default: str
    scope: Scope = Scope()
    min_agree: int | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cases": [{"label": c.label, "when": c.when.to_json()} for c in self.cases],
            "default": self.default,
        }
        if self.scope.to_json() is not None:
            out["applies_to"] = self.scope.to_json()
        if self.min_agree is not None:
            out["group"] = {"min_agree": self.min_agree}
        return out


@dataclass(frozen=True)
class EarlyStopRule:
    name: str
    at_steps: tuple[int, ...]
    outcome: str
    when: Cond
    scope: Scope = Scope()

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "at_steps": list(self.at_steps),
            "outcome": self.outcome,
            "when": self.when.to_json(),
        }
        if self.scope.to_json() is not None:
            out["applies_to"] = self.scope.to_json()
        return out


@dataclass(frozen=True)
class RuleSet:
    verdicts: dict[str, VerdictRule] = field(default_factory=dict)
    early_stop: tuple[EarlyStopRule, ...] = ()

    @property
    def order(self) -> list[str]:
        """Verdict names in an order where every ``verdict:`` reference comes first."""
        return _topological(self.verdicts)

    def verdicts_json(self) -> dict[str, Any]:
        return {name: rule.to_json() for name, rule in self.verdicts.items()}  # declared order

    def early_stop_json(self) -> list[dict[str, Any]]:
        return [r.to_json() for r in self.early_stop]

    def to_json(self) -> dict[str, Any]:
        """The expanded rules as stored at enqueue (the database sorts keys, so the declared verdict
        order is kept in ``verdict_order``)."""
        return {
            "verdicts": self.verdicts_json(),
            "verdict_order": list(self.verdicts),
            "early_stop": self.early_stop_json(),
        }


# --------------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------------


def _int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuleError(f"{what} must be an integer, got {value!r}")
    return value


def _parse_steps(raw: Any) -> Steps:
    if not isinstance(raw, Mapping) or not raw:
        raise RuleError(f"steps must be a mapping with at / from / to / every, got {raw!r}")
    unknown = set(raw) - {"at", "from", "to", "every"}
    if unknown:
        raise RuleError(f"unknown steps keys {sorted(unknown)}")
    if "at" in raw and len(raw) > 1:
        raise RuleError("steps.at selects one step; it takes no from / to / every")
    steps = Steps(
        at=_int(raw["at"], "steps.at") if "at" in raw else None,
        start=_int(raw["from"], "steps.from") if "from" in raw else None,
        stop=_int(raw["to"], "steps.to") if "to" in raw else None,
        every=_int(raw["every"], "steps.every") if "every" in raw else None,
    )
    if steps.every is not None and steps.every < 1:
        raise RuleError("steps.every must be >= 1")
    if steps.start is not None and steps.stop is not None and steps.start > steps.stop:
        raise RuleError(f"steps.from {steps.start} > steps.to {steps.stop}")
    return steps


def _parse_value(raw: Any, op: str) -> Any:
    if op == "between":
        if not isinstance(raw, Sequence) or isinstance(raw, str) or len(raw) != 2:
            raise RuleError(f"op between takes value [lo, hi], got {raw!r}")
        return tuple(_parse_value(v, "==") for v in raw)
    if isinstance(raw, bool) or raw is None or isinstance(raw, (list, dict)):
        raise RuleError(f"value must be a number or a string, got {raw!r}")
    if isinstance(raw, (int, float)):
        if not math.isfinite(float(raw)):
            raise RuleError(f"value must be finite, got {raw!r}")
        return float(raw)
    if isinstance(raw, str):
        return raw
    raise RuleError(f"value must be a number or a string, got {raw!r}")


def _parse_atom(raw: Mapping[str, Any]) -> Atom:
    unknown = set(raw) - {"metric", "steps", "agg", "op", "value", "min_samples"}
    if unknown:
        raise RuleError(f"unknown condition keys {sorted(unknown)}")
    for key in ("metric", "op", "value"):
        if key not in raw:
            raise RuleError(f"condition {dict(raw)} lacks {key!r}")
    metric = raw["metric"]
    if not isinstance(metric, str) or ":" not in metric:
        raise RuleError(f"metric must be '<source>:<key>', got {metric!r}")
    source, key = metric.split(":", 1)
    if source not in SOURCES:
        raise RuleError(f"unknown metric source {source!r} in {metric!r}; known: {SOURCES}")
    if not key:
        raise RuleError(f"metric {metric!r} has no key")
    if source == "eval":
        _eval_key(key)
    if source == "train" and key not in TRAIN_METRICS:
        raise RuleError(f"train metrics are {TRAIN_METRICS}, got {key!r}")
    op = raw["op"]
    if op not in OPS:
        raise RuleError(f"unknown op {op!r}; known: {OPS}")
    steps = _parse_steps(raw["steps"]) if "steps" in raw else None
    agg = raw.get("agg")
    if source in SERIES_SOURCES:
        if agg is None and not (steps is not None and steps.at is not None):
            raise RuleError(f"{metric}: a step series needs agg (one of {AGGS}) unless steps.at")
        if agg is not None and agg not in AGGS:
            raise RuleError(f"unknown agg {agg!r}; known: {AGGS}")
    elif steps is not None or agg is not None:
        raise RuleError(f"{metric}: scalar sources take no steps / agg")
    min_samples = None
    if "min_samples" in raw:
        min_samples = _int(raw["min_samples"], "min_samples")
        if min_samples < 1:
            raise RuleError(f"{metric}: min_samples must be >= 1, got {min_samples}")
        if steps is None or steps.at is not None or (steps.start is None and steps.stop is None):
            raise RuleError(
                f"{metric}: min_samples needs a window (steps with from and/or to, not at)"
            )
    return Atom(source, key, op, _parse_value(raw["value"], op), steps, agg, min_samples)


def parse_cond(raw: Any) -> Cond:
    if not isinstance(raw, Mapping) or not raw:
        raise RuleError(f"a condition must be a mapping, got {raw!r}")
    keys = set(raw)
    if keys == {"all"} or keys == {"any"}:
        kind = next(iter(keys))
        parts = raw[kind]
        if not isinstance(parts, Sequence) or isinstance(parts, str) or not parts:
            raise RuleError(f"{kind} takes a non-empty list of conditions")
        return Combo(kind, tuple(parse_cond(p) for p in parts))
    if keys == {"not"}:
        return Combo("not", (parse_cond(raw["not"]),))
    if "metric" in keys:
        return _parse_atom(raw)
    raise RuleError(
        f"a condition is an atom (metric, op, value) or all / any / not, got {dict(raw)}"
    )


def _names(raw: Any, what: str) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Sequence) or not raw or not all(isinstance(f, str) for f in raw):
        raise RuleError(f"applies_to.{what} must be a non-empty list of names, got {raw!r}")
    return tuple(sorted(set(raw)))


def _scope(raw: Any) -> Scope:
    if raw is None:
        return Scope()
    if not isinstance(raw, Mapping) or not raw or set(raw) - {"family", "protocol"}:
        raise RuleError(f"applies_to must be {{family: [...], protocol: [...]}}, got {raw!r}")
    return Scope(_names(raw.get("family"), "family"), _names(raw.get("protocol"), "protocol"))


def _verdict(name: str, raw: Any) -> VerdictRule:
    if not isinstance(raw, Mapping):
        raise RuleError(f"verdict {name!r} must be a mapping")
    if "preset" in raw:
        return _preset(name, raw)
    unknown = set(raw) - {"cases", "default", "applies_to", "group"}
    if unknown:
        raise RuleError(f"verdict {name!r}: unknown keys {sorted(unknown)}")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, Sequence) or isinstance(cases_raw, str) or not cases_raw:
        raise RuleError(f"verdict {name!r} needs a non-empty list of cases")
    cases = []
    for c in cases_raw:
        if not isinstance(c, Mapping) or set(c) != {"label", "when"}:
            raise RuleError(f"verdict {name!r}: a case is {{label, when}}, got {c!r}")
        if not isinstance(c["label"], str) or not c["label"]:
            raise RuleError(f"verdict {name!r}: a case label must be a non-empty string")
        cases.append(Case(c["label"], parse_cond(c["when"])))
    default = raw.get("default")
    if not isinstance(default, str) or not default:
        raise RuleError(f"verdict {name!r} needs a default label")
    return VerdictRule(tuple(cases), default, _scope(raw.get("applies_to")), _group(name, raw))


def _group(name: str, raw: Mapping[str, Any]) -> int | None:
    group = raw.get("group")
    if group is None:
        return None
    if not isinstance(group, Mapping) or set(group) != {"min_agree"}:
        raise RuleError(f"verdict {name!r}: group must be {{min_agree: n}}")
    n = _int(group["min_agree"], "group.min_agree")
    if n < 1:
        raise RuleError("group.min_agree must be >= 1")
    return n


# presets ------------------------------------------------------------------------------------------


def _atom(
    metric: str,
    op: str,
    value: Any,
    steps: dict[str, int] | None = None,
    agg: str | None = None,
    min_samples: int | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"metric": metric, "op": op, "value": value}
    if steps is not None:
        out["steps"] = steps
    if agg is not None:
        out["agg"] = agg
    if min_samples is not None:
        out["min_samples"] = min_samples
    return out


def _preset_reads_state(p: dict[str, Any]) -> dict[str, Any]:
    follow_min, unchanged_max, follow_below = (
        p.pop("follow_min", 0.5),
        p.pop("unchanged_max", 0.1),
        p.pop("follow_below", 0.2),
    )
    return {
        "cases": [
            {"label": "no correct answers to probe", "when": _atom("probe:correct_n", "==", 0)},
            {
                "label": "reads state",
                "when": {
                    "all": [
                        _atom("probe:correct_follow_frac", ">=", follow_min),
                        _atom("probe:correct_unchanged_frac", "<=", unchanged_max),
                    ]
                },
            },
            {
                "label": "does not read state",
                "when": _atom("probe:correct_follow_frac", "<", follow_below),
            },
        ],
        "default": "partial",
    }


def _preset_reads_state_v2(p: dict[str, Any]) -> dict[str, Any]:
    """P1.14: the v2 verdict (decision D4, A) over the matched documents answered correctly before
    the edit: v1's thresholds, after "no matched probes" (0) and "too few matched probes" (below
    ``min_correct``, default 20). Needs reads_state version 2 (or a dry run)."""
    follow_min, unchanged_max, follow_below = (
        p.pop("follow_min", 0.5),
        p.pop("unchanged_max", 0.1),
        p.pop("follow_below", 0.2),
    )
    min_correct = p.pop("min_correct", 20)
    return {
        "cases": [
            {"label": "no matched probes", "when": _atom("probe_v2:correct_n", "==", 0)},
            {
                "label": "too few matched probes",
                "when": _atom("probe_v2:correct_n", "<", min_correct),
            },
            {
                "label": "reads state",
                "when": {
                    "all": [
                        _atom("probe_v2:correct_follow_frac", ">=", follow_min),
                        _atom("probe_v2:correct_unchanged_frac", "<=", unchanged_max),
                    ]
                },
            },
            {
                "label": "does not read state",
                "when": _atom("probe_v2:correct_follow_frac", "<", follow_below),
            },
        ],
        "default": "partial",
    }


def _preset_memory_open(p: dict[str, Any]) -> dict[str, Any]:
    """Gate p50 and the cell's gradient share at or above their thresholds at the multiples of
    ``every`` from ``from_step``. Without ``window``: the one sample at each such step (``agg``
    all, as registered for queue9 at a cadence of 500), or one aggregate over all of them. With
    ``window: w`` (and ``to_step``): at each such step s, the ``agg`` (default median) over the
    diagnostics in [s - w, s + w] (the user's windowed rule of 7 Nov, at a cadence of 25).
    ``min_samples`` (P1.14) goes to every atom (each window must hold that many samples)."""
    start, every = p.pop("from_step", 500), p.pop("every", 500)
    to_step = p.pop("to_step", None)
    window = p.pop("window", None)
    gate, share = p.pop("gate_p50_min", 0.01), p.pop("cell_share_min", 0.01)
    least = p.pop("min_samples", None)
    if window is None:
        steps: dict[str, int] = {"from": start, "every": every}
        if to_step is not None:
            steps["to"] = to_step
        agg = p.pop("agg", "all")
        atoms = [
            _atom("diag:gate/p50", ">=", gate, steps, agg, least),
            _atom("diag:grad_share/cell", ">=", share, dict(steps), agg, least),
        ]
    else:
        if to_step is None:
            raise RuleError("preset memory_open: window needs to_step (the last checkpoint)")
        agg = p.pop("agg", "median")
        atoms = []
        for s in range(start, int(to_step) + 1, every):
            span = {"from": s - int(window), "to": s + int(window)}
            atoms.append(_atom("diag:gate/p50", ">=", gate, dict(span), agg, least))
            atoms.append(_atom("diag:grad_share/cell", ">=", share, dict(span), agg, least))
    return {
        "cases": [{"label": "open", "when": {"all": atoms}}],
        "default": "shut",
        "applies_to": {"family": ["hsn"]},
    }


def _preset_cells_apart(p: dict[str, Any]) -> dict[str, Any]:
    rank, div = p.pop("rank_pr_min", 2.0), p.pop("diversity_live_min", 0.2)
    return {
        "cases": [
            {
                "label": "apart",
                "when": {
                    "all": [
                        _atom("trained:rank_pr/answer", ">=", rank),
                        _atom("trained:diversity_live/answer", ">=", div),
                    ]
                },
            }
        ],
        "default": "not apart",
        "applies_to": {"family": ["hsn"]},
    }


PRESETS = {
    "reads_state": _preset_reads_state,  # the audit's verdict (probe counts, correct answers)
    "reads_state_v2": _preset_reads_state_v2,  # P1.14: the same over probe v2's matched edits
    "memory_open": _preset_memory_open,  # queue9 rule: gate p50 and the cell's grad share, x500
    "cells_apart": _preset_cells_apart,  # queue9 rule: rank_pr and diversity_live, trained model
}


def _preset(name: str, raw: Mapping[str, Any]) -> VerdictRule:
    params = dict(raw)
    preset = params.pop("preset")
    if preset not in PRESETS:
        raise RuleError(f"verdict {name!r}: unknown preset {preset!r}; known: {sorted(PRESETS)}")
    applies_to = params.pop("applies_to", None)
    group = params.pop("group", None)
    expanded = PRESETS[preset](params)
    if params:
        raise RuleError(f"verdict {name!r}: preset {preset} takes no parameters {sorted(params)}")
    if applies_to is not None:
        expanded["applies_to"] = applies_to
    if group is not None:
        expanded["group"] = group
    return _verdict(name, expanded)


def _early_stop(raw: Any) -> EarlyStopRule:
    if not isinstance(raw, Mapping):
        raise RuleError(f"an early-stop rule must be a mapping, got {raw!r}")
    unknown = set(raw) - {"name", "at_steps", "outcome", "when", "applies_to"}
    missing = {"name", "at_steps", "outcome", "when"} - set(raw)
    if unknown or missing:
        raise RuleError(
            f"early-stop rule {raw.get('name')!r}: unknown keys {sorted(unknown)}, "
            f"missing {sorted(missing)}"
        )
    at = raw["at_steps"]
    if not isinstance(at, Sequence) or isinstance(at, str) or not at:
        raise RuleError("early-stop at_steps must be a non-empty list of steps")
    steps = tuple(sorted({_int(s, "at_steps") for s in at}))
    for key in ("name", "outcome"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise RuleError(f"early-stop {key} must be a non-empty string")
    return EarlyStopRule(
        raw["name"],
        steps,
        raw["outcome"],
        parse_cond(raw["when"]),
        _scope(raw.get("applies_to")),
    )


def parse_rules(raw: Any) -> RuleSet:
    """``{verdicts: {name: rule}, early_stop: [rule, ...]}`` (both optional)."""
    if raw is None:
        return RuleSet()
    if not isinstance(raw, Mapping):
        raise RuleError(f"rules must be a mapping, got {raw!r}")
    unknown = set(raw) - {"verdicts", "early_stop"}
    if unknown:
        raise RuleError(f"unknown rules keys {sorted(unknown)}")
    verdicts_raw = raw.get("verdicts") or {}
    if not isinstance(verdicts_raw, Mapping):
        raise RuleError("rules.verdicts must be a mapping of name -> rule")
    verdicts = {str(name): _verdict(str(name), rule) for name, rule in verdicts_raw.items()}
    stops_raw = raw.get("early_stop") or []
    if not isinstance(stops_raw, Sequence) or isinstance(stops_raw, str):
        raise RuleError("rules.early_stop must be a list")
    stops = tuple(_early_stop(r) for r in stops_raw)
    names = [s.name for s in stops]
    if len(set(names)) != len(names):
        raise RuleError(f"duplicate early-stop names {names}")
    ruleset = RuleSet(verdicts, stops)
    ruleset.order  # noqa: B018  (raises on an unknown reference or a cycle)
    return ruleset


def _atoms(cond: Cond) -> list[Atom]:
    if isinstance(cond, Atom):
        return [cond]
    return [a for part in cond.parts for a in _atoms(part)]


def _topological(verdicts: Mapping[str, VerdictRule]) -> list[str]:
    deps = {
        name: {a.key for c in rule.cases for a in _atoms(c.when) if a.source == "verdict"}
        for name, rule in verdicts.items()
    }
    for name, refs in deps.items():
        unknown = refs - set(verdicts)
        if unknown:
            raise RuleError(f"verdict {name!r} references unknown verdicts {sorted(unknown)}")
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(name: str, path: tuple[str, ...]) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise RuleError(f"verdict references form a cycle: {' -> '.join((*path, name))}")
        state[name] = 1
        for ref in sorted(deps[name]):
            visit(ref, (*path, name))
        state[name] = 2
        order.append(name)

    for name in sorted(verdicts):
        visit(name, ())
    return order


# --------------------------------------------------------------------------------------------------
# Values and the record view
# --------------------------------------------------------------------------------------------------


def resolve_value(value: Any, constants: Mapping[str, Any]) -> Any:
    """A number, a constant (``name`` or ``name ± number``) or a string literal."""
    if isinstance(value, tuple):
        return tuple(resolve_value(v, constants) for v in value)
    if not isinstance(value, str):
        return value
    m = _CONST_EXPR.match(value)
    if m is None:
        return value
    name, sign, number = m.groups()
    if name not in constants:
        if sign is not None:
            raise RuleError(f"value {value!r} uses the unknown constant {name!r}")
        return value  # a literal label
    base = constants[name]
    if sign is None:
        return (
            float(base) if isinstance(base, (int, float)) and not isinstance(base, bool) else base
        )
    if isinstance(base, bool) or not isinstance(base, (int, float)):
        raise RuleError(f"value {value!r}: constant {name!r} is not a number")
    return float(base) + (float(number) if sign == "+" else -float(number))


def _eval_key(key: str) -> tuple[str | None, str]:
    parts = key.split("/")
    metric = parts[-1]
    if metric not in EVAL_METRICS:
        raise RuleError(f"eval metric must end in one of {EVAL_METRICS}, got {key!r}")
    return ("/".join(parts[:-1]) or None), metric


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    return None


class RecordView:
    """Indexed access to a run record (the dict a protocol writes; see :func:`series`)."""

    def __init__(self, record: Mapping[str, Any]) -> None:
        self.record = record
        self.evals = {int(e["step"]): e for e in record.get("evals") or []}
        self.diag = {int(d["step"]): d for d in record.get("diagnostics") or []}
        self.train = {int(t["step"]): t for t in record.get("train") or []}

    def series(self, source: str, key: str) -> dict[int, float | None]:
        if source == "eval":
            group, metric = _eval_key(key)
            out: dict[int, float | None] = {}
            for step, e in self.evals.items():
                if group is None:
                    out[step] = _num(e.get(metric))
                elif group.startswith(SPLIT_PREFIX):  # P1.15: the first/later split
                    split = (e.get("split") or {}).get(group[len(SPLIT_PREFIX) :]) or {}
                    out[step] = _num(split.get(metric))
                else:
                    out[step] = _num((e.get("per_group") or {}).get(group, {}).get(metric))
            return out
        rows = self.diag if source == "diag" else self.train
        return {step: _num(row.get(key)) for step, row in rows.items()}

    def scalar(self, source: str, key: str, verdicts: Mapping[str, str]) -> Any:
        if source == "verdict":
            return verdicts.get(key)
        block = {
            "trained": self.record.get("trained_state"),
            "probe": self.record.get("probe"),
            "probe_v2": self.record.get("probe_v2"),
            "probe_v3": self.record.get("probe_v3"),  # P1.15
            "overfit": self.record.get("overfit"),
            "final": self.record.get("final"),
        }[source]
        if not isinstance(block, Mapping):
            return None
        value = block.get(key)
        if isinstance(value, str):
            return value
        return _num(value)


# --------------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------------


def _compare(op: str, x: Any, value: Any) -> bool:
    if op == "between":
        lo, hi = value
        return bool(lo <= x <= hi)
    if op == "==":
        return bool(x == value)
    if op == "!=":
        return bool(x != value)
    if not isinstance(x, float) or not isinstance(value, float):
        raise RuleError(f"ordered comparison {op} needs numbers, got {x!r} and {value!r}")
    return bool({">=": x >= value, ">": x > value, "<=": x <= value, "<": x < value}[op])


_AGG: dict[str, Callable[[list[float]], float]] = {
    "median": statistics.median,
    "mean": statistics.fmean,
    "min": min,
    "max": max,
    "first": lambda v: v[0],
    "last": lambda v: v[-1],
}


def _eval_atom(
    atom: Atom,
    view: RecordView,
    constants: Mapping[str, Any],
    verdicts: Mapping[str, str],
    upto: int | None,
) -> tuple[bool | None, dict[str, Any]]:
    value = resolve_value(atom.value, constants)
    detail: dict[str, Any] = {
        "metric": atom.metric,
        "op": atom.op,
        "value": list(value) if isinstance(value, tuple) else value,
    }
    if atom.op in _ORDERED:
        vals = value if isinstance(value, tuple) else (value,)
        if not all(isinstance(v, float) for v in vals):
            raise RuleError(f"{atom.metric} {atom.op}: the value {value!r} is not a number")
    if atom.source in SERIES_SOURCES:
        series = view.series(atom.source, atom.key)
        available = [s for s in series if upto is None or s <= upto]
        last = upto if upto is not None else max(available, default=0)
        wanted = (atom.steps or Steps()).wanted(available, last)
        if upto is not None:
            wanted = [s for s in wanted if s <= upto]
        samples = [series.get(s) for s in wanted]
        detail["steps"] = wanted
        missing = [s for s, x in zip(wanted, samples, strict=True) if x is None]
        if not wanted or missing:
            detail["missing"] = missing if wanted else "no sample selected"
            return None, detail
        xs = [float(x) for x in samples if x is not None]
        detail["n"] = len(xs)
        if atom.min_samples is not None and len(xs) < atom.min_samples:
            detail["missing"] = f"{len(xs)} samples < min_samples {atom.min_samples}"
            return None, detail
        agg = atom.agg or "last"
        detail["agg"] = agg
        if agg in ("all", "any"):
            hits = [_compare(atom.op, x, value) for x in xs]
            detail["samples"] = xs if len(xs) <= 32 else {"min": min(xs), "max": max(xs)}
            return (all(hits) if agg == "all" else any(hits)), detail
        x = float(_AGG[agg](xs))
        detail["measured"] = x
        return _compare(atom.op, x, value), detail
    x = view.scalar(atom.source, atom.key, verdicts)
    detail["measured"] = x
    if x is None:
        detail["missing"] = atom.metric
        return None, detail
    if atom.op in _ORDERED and not isinstance(x, float):
        return None, {**detail, "missing": f"{atom.metric} is not a number ({x!r})"}
    return _compare(atom.op, x, value), detail


def _eval_cond(
    cond: Cond,
    view: RecordView,
    constants: Mapping[str, Any],
    verdicts: Mapping[str, str],
    upto: int | None,
    log: list[dict[str, Any]],
) -> bool | None:
    if isinstance(cond, Atom):
        result, detail = _eval_atom(cond, view, constants, verdicts, upto)
        log.append({**detail, "result": result})
        return result
    results = [_eval_cond(p, view, constants, verdicts, upto, log) for p in cond.parts]
    if cond.kind == "not":
        return None if results[0] is None else not results[0]
    if cond.kind == "all":
        if any(r is False for r in results):
            return False
        return None if any(r is None for r in results) else True
    if any(r is True for r in results):
        return True
    return None if any(r is None for r in results) else False


def evaluate_verdicts(
    rules: RuleSet,
    record: Mapping[str, Any],
    constants: Mapping[str, Any],
    family: str,
    protocol: str | None = None,
) -> dict[str, dict[str, Any]]:
    """``{name: {label, detail}}`` for every verdict rule on one run record (``protocol`` defaults
    to the record's)."""
    protocol = protocol if protocol is not None else record.get("protocol")
    view = RecordView(record)
    stopped = record.get("stopped")
    failure = record.get("failure")
    labels: dict[str, str] = {}
    out: dict[str, dict[str, Any]] = {}
    for name in rules.order:
        rule = rules.verdicts[name]
        outside = rule.scope.excludes(family, protocol)
        if outside is not None:
            labels[name] = f"n/a ({outside})"
            out[name] = {"label": labels[name], "detail": []}
            continue
        if isinstance(failure, Mapping):  # P1.14: a failed arm's partial record never passes
            labels[name] = f"failed: {failure.get('reason')} @ {failure.get('last_step')}"
            out[name] = {"label": labels[name], "detail": []}
            continue
        if stopped:
            labels[name] = f"stopped early: {stopped['outcome']} @ {stopped['step']}"
            out[name] = {"label": labels[name], "detail": []}
            continue
        label: str | None = None
        log: list[dict[str, Any]] = []
        for case in rule.cases:
            result = _eval_cond(case.when, view, constants, labels, None, log)
            if result is True:
                label = case.label
                break
            if result is None:
                missing = [d for d in log if d.get("result") is None]
                what = ", ".join(f"{d['metric']} {d.get('missing')}" for d in missing) or "missing"
                label = f"n/a: {what}"
                break
        labels[name] = label if label is not None else rule.default
        out[name] = {"label": labels[name], "detail": log}
    return out


def check_early_stop(
    rules: RuleSet,
    record: Mapping[str, Any],
    step: int,
    constants: Mapping[str, Any],
    family: str,
    protocol: str | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Evaluate the early-stop rules due at ``step`` on the record so far: ``(stop, log)``, where
    ``stop`` is ``{rule, step, outcome, values}`` for the first rule that holds."""
    view = RecordView(record)
    log: list[dict[str, Any]] = []
    for rule in rules.early_stop:
        if step not in rule.at_steps:
            continue
        if rule.scope.excludes(family, protocol) is not None:
            continue
        atoms: list[dict[str, Any]] = []
        result = _eval_cond(rule.when, view, constants, {}, step, atoms)
        entry = {"rule": rule.name, "step": step, "result": result, "values": atoms}
        log.append(entry)
        if result is True:
            return {"rule": rule.name, "step": step, "outcome": rule.outcome, "values": atoms}, log
    return None, log


# --------------------------------------------------------------------------------------------------
# Static checks
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cadence:
    """What a run of one arm will produce: its last step, evaluation and diagnostic steps, and the
    scalar sources its protocol writes."""

    steps: int
    eval_steps: tuple[int, ...]
    diag_steps: tuple[int, ...]
    scalar_sources: tuple[str, ...]
    protocol: str | None = None

    @classmethod
    def of(
        cls,
        steps: int,
        eval_every: int,
        diag_every: int,
        scalar_sources: Iterable[str],
        protocol: str | None = None,
        *,
        dense_every: int = 0,
        windows: Iterable[Sequence[int]] = (),
    ) -> Cadence:
        evals = sorted({s for s in range(eval_every, steps + 1, eval_every)} | {steps})
        diags = diag_schedule(steps, diag_every, dense_every, windows)
        return cls(steps, tuple(evals), diags, tuple(scalar_sources), protocol)


def diag_schedule(
    steps: int, diag_every: int, dense_every: int = 0, windows: Iterable[Sequence[int]] = ()
) -> tuple[int, ...]:
    """The diagnostics steps of a run of ``steps`` steps: every multiple of ``diag_every`` (0: none)
    and, with ``dense_every`` (P1.14's mixed cadence), every multiple of it inside the windows
    ``[lo, hi]``."""
    out = set(range(diag_every, steps + 1, diag_every)) if diag_every > 0 else set()
    if dense_every > 0:
        for lo, hi in windows:
            first = -(-max(int(lo), 1) // dense_every) * dense_every
            out.update(range(first, min(int(hi), steps) + 1, dense_every))
    return tuple(sorted(out))


def merge_windows(windows: Iterable[Sequence[int]]) -> list[list[int]]:
    """Windows ``[lo, hi]`` sorted, with overlapping ones merged."""
    out: list[list[int]] = []
    for lo, hi in sorted((int(a), int(b)) for a, b in windows):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def rule_windows(rules: RuleSet, family: str, protocol: str | None, steps: int) -> list[list[int]]:
    """P1.14: the windows the ``diag:`` atoms of the verdicts and early stops that apply to this arm
    read (``steps.at`` a one-step window, a span its ``[from, to]``, clipped to ``[1, steps]``),
    merged. An atom with no ``steps`` raises: it would read a mix of dense and sparse samples."""
    conds: list[tuple[str, Cond]] = []
    for name, rule in rules.verdicts.items():
        if rule.scope.excludes(family, protocol) is None:
            conds += [(f"verdict {name!r}", case.when) for case in rule.cases]
    for stop in rules.early_stop:
        if stop.scope.excludes(family, protocol) is None:
            conds.append((f"early stop {stop.name!r}", stop.when))
    windows = []
    for where, cond in conds:
        for atom in _atoms(cond):
            if atom.source != "diag":
                continue
            sel = atom.steps
            if sel is None:
                raise RuleError(
                    f"{where}: {atom.metric} reads every diagnostics sample; with diag_dense_every "
                    "it would read a mix of dense and sparse ones: give it steps"
                )
            if sel.at is not None:
                lo = hi = sel.at
            else:
                lo, hi = sel.start or 1, sel.stop if sel.stop is not None else steps
            lo, hi = max(1, lo), min(steps, hi)
            if lo <= hi:
                windows.append((lo, hi))
    return merge_windows(windows)


def _check_atom(
    atom: Atom, cadence: Cadence, constants: Mapping[str, Any], where: str, upto: int | None
) -> None:
    value = resolve_value(atom.value, constants)
    if atom.op in _ORDERED:
        vals = value if isinstance(value, tuple) else (value,)
        if not all(isinstance(v, float) for v in vals):
            raise RuleError(
                f"{where}: {atom.metric} {atom.op} {atom.value!r} is not a number "
                f"(constants {dict(constants)})"
            )
    if atom.source in SERIES_SOURCES:
        available = {
            "eval": cadence.eval_steps,
            "diag": cadence.diag_steps,
            "train": tuple(range(1, cadence.steps + 1)),
        }[atom.source]
        if not available:
            raise RuleError(
                f"{where}: {atom.metric} needs {atom.source} samples, and this arm records none"
            )
        steps = atom.steps or Steps()
        wanted = steps.wanted(available, upto if upto is not None else cadence.steps)
        if not wanted:
            raise RuleError(
                f"{where}: {atom.metric} {steps.to_json()} selects no step of "
                f"{list(available)[:12]}"
            )
        absent = [s for s in wanted if s not in available]
        if absent:
            raise RuleError(
                f"{where}: {atom.metric} reads steps {absent} this arm does not record "
                f"({atom.source} steps {list(available)[:12]}...)"
            )
        if atom.min_samples is not None and len(wanted) < atom.min_samples:
            raise RuleError(
                f"{where}: {atom.metric} {steps.to_json()} holds {len(wanted)} of this arm's "
                f"{atom.source} steps, fewer than min_samples {atom.min_samples}"
            )
        if upto is not None and max(wanted) > upto:
            raise RuleError(
                f"{where}: {atom.metric} reads step {max(wanted)}, after the rule's step {upto}"
            )
    else:
        if upto is not None:
            raise RuleError(f"{where}: an early stop reads step series only, not {atom.metric}")
        if atom.source != "verdict" and atom.source not in cadence.scalar_sources:
            hint = {
                "probe_v2": " (probe_v2 needs reads_state version 2)",
                "probe_v3": " (probe_v3 needs reads_state version 3)",
            }.get(atom.source, "")
            raise RuleError(
                f"{where}: protocol {cadence.protocol} writes no {atom.source}: values "
                f"({atom.metric}){hint}; restrict the rule with applies_to: {{protocol: [...]}}"
            )


def check_rules(
    rules: RuleSet, cadence: Cadence, constants: Mapping[str, Any], family: str
) -> None:
    """The static checks of one arm (module docstring); raises :class:`RuleError`."""
    for name, rule in rules.verdicts.items():
        if rule.scope.excludes(family, cadence.protocol) is not None:
            continue
        for case in rule.cases:
            for atom in _atoms(case.when):
                _check_atom(atom, cadence, constants, f"verdict {name!r}", None)
    for stop in rules.early_stop:
        if stop.scope.excludes(family, cadence.protocol) is not None:
            continue
        for at in stop.at_steps:
            if at not in cadence.eval_steps:
                raise RuleError(
                    f"early stop {stop.name!r}: step {at} is not an evaluation step "
                    f"{list(cadence.eval_steps)}"
                )
            if at >= cadence.steps:
                raise RuleError(
                    f"early stop {stop.name!r}: step {at} is not before the last step "
                    f"{cadence.steps}"
                )
            for atom in _atoms(stop.when):
                _check_atom(atom, cadence, constants, f"early stop {stop.name!r} @ {at}", at)


def _resolved_cond(cond: Cond, constants: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(cond, Atom):
        out = cond.to_json()
        value = resolve_value(cond.value, constants)
        out["value"] = list(value) if isinstance(value, tuple) else value
        return out
    if cond.kind == "not":
        return {"not": _resolved_cond(cond.parts[0], constants)}
    return {cond.kind: [_resolved_cond(p, constants) for p in cond.parts]}


def resolved_early_stop(
    rules: RuleSet, family: str, protocol: str | None, constants: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The early-stop rules that apply to one arm, with the arm's constants substituted: what the
    arm's id hashes (two arms whose stops differ only by a constant are different arms) and what its
    child evaluates."""
    out = []
    for rule in rules.early_stop:
        if rule.scope.excludes(family, protocol) is not None:
            continue
        entry = rule.to_json()
        entry["when"] = _resolved_cond(rule.when, constants)
        out.append(entry)
    return out
