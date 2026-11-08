"""Reports of the experiment database (P1.11). Reads only the database; imports without torch.

``report`` gives one row per seed group (the arms of one recipe that differ only in their seed),
or with ``per_seed`` one row per arm, as markdown or CSV. Per arm it uses the latest finished
attempt (``done`` or ``stopped_early``), else the latest attempt. Columns:

* label, family, variant, the overrides that set the arm apart from the others in the report;
  (``per_seed`` lists every finished attempt of an arm, for example two imported runs of one recipe
  on two commits; a group counts each arm once, by its latest finished attempt);
* seeds finished / total, and the statuses;
* each verdict rule: the most common label and how many seeds share it (``reads state 2/3``),
  marked ``(finding)`` when the rule's ``group.min_agree`` is met (computed at group level, never
  per arm);
* final dev accuracy and loss (mean [min, max] over the finished seeds), first-query accuracy and
  the follow / unchanged fractions of the edit probe (P1.14: and of probe v2, with its vocabulary
  size and the share of documents with a matched edit); the trained model's write rate (P1.14: the
  share of gates >= 0.5) and the median step time (P1.14: ``step_time.run_median_s``);
* early stops; failures (P1.14: ``<reason> @ <last step>`` of an attempt that failed or was
  cancelled, from its partial record); median train and total wall seconds and compile seconds;
* card status, commits, imported.

A failed attempt's verdicts read ``failed: ...`` and are never counted as a finding.

With a spec (``--spec``), that spec's verdict rules are evaluated again on the stored records of its
arms, whichever sweep ran them, with the spec's constants.
"""

from __future__ import annotations

import csv
import io
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hsn.experiments.db import FINISHED, Database, loads
from hsn.experiments.rules import RuleSet, evaluate_verdicts, parse_rules

__all__ = ["Report", "build_report", "format_csv", "format_markdown"]

_SELECT = """
SELECT e.*, a.family, a.protocol, a.variant, a.variant_json, a.seed, a.group_id,
       a.overrides_json, a.config_path, r.verdicts_json, r.record_json, r.final_dev_accuracy,
       r.final_dev_loss, r.train_s, r.total_s, r.json_path, r.early_stop_json,
       s.name AS sweep_name
FROM experiments e
JOIN arms a ON a.arm_id = e.arm_id
LEFT JOIN results r ON r.exp_id = e.exp_id
LEFT JOIN sweeps s ON s.id = e.sweep_id
"""


@dataclass
class Report:
    columns: list[str]
    rows: list[dict[str, Any]]
    verdict_names: list[str]


def _latest(rows: Sequence[dict[str, Any]], *, all_finished: bool = False) -> list[dict[str, Any]]:
    """Per arm, the latest finished attempt (``all_finished``: every finished attempt), else the
    latest attempt."""
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_arm.setdefault(r["arm_id"], []).append(r)
    out = []
    for attempts in by_arm.values():
        finished = [a for a in attempts if a["status"] in FINISHED]
        if all_finished and finished:
            out.extend(finished)
        else:
            out.append(max(finished or attempts, key=lambda a: a["exp_id"]))
    return sorted(out, key=lambda r: r["exp_id"])


def _select(
    db: Database,
    *,
    sweep: str | None,
    arm_ids: Sequence[str] | None,
    statuses: Sequence[str] | None,
    family: str | None,
    protocol: str | None,
    imported: bool | None,
    all_finished: bool = False,
) -> list[dict[str, Any]]:
    where: list[str] = []
    params: list[Any] = []
    if sweep is not None:
        where.append(
            "e.arm_id IN (SELECT sa.arm_id FROM sweep_arms sa "
            "JOIN sweeps s2 ON s2.id = sa.sweep_id WHERE s2.name = ?)"
        )
        params.append(sweep)
    if arm_ids is not None:
        where.append(f"e.arm_id IN ({', '.join('?' for _ in arm_ids)})")
        params.extend(arm_ids)
    if family is not None:
        where.append("a.family = ?")
        params.append(family)
    if protocol is not None:
        where.append("a.protocol = ?")
        params.append(protocol)
    if imported is not None:
        where.append("e.imported = ?")
        params.append(int(imported))
    sql = _SELECT + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY e.exp_id"
    with db.read() as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        labels = {
            (r["sweep_id"], r["arm_id"]): r["label"]
            for r in conn.execute("SELECT sweep_id, arm_id, label FROM sweep_arms").fetchall()
        }
    rows = _latest(rows, all_finished=all_finished)
    if statuses is not None:
        rows = [r for r in rows if r["status"] in statuses]
    for r in rows:
        if not r["imported"]:  # an imported run keeps its own name (one arm can hold several)
            r["label"] = labels.get((r["sweep_id"], r["arm_id"]), r["label"])
    return rows


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _spread(values: Sequence[float], digits: int = 4) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    mean, lo, hi = statistics.fmean(values), min(values), max(values)
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _distinct_overrides(rows: Sequence[dict[str, Any]]) -> dict[int, str]:
    lists = {
        r["exp_id"]: [
            o for o in (loads(r["overrides_json"]) or []) if not o.startswith(("seed=", "run_id="))
        ]
        for r in rows
    }
    common = set.intersection(*(set(v) for v in lists.values())) if lists else set()
    return {k: " ".join(o for o in v if o not in common) for k, v in lists.items()}


def _verdict_labels(
    row: dict[str, Any], rules: RuleSet | None, constants: Mapping[str, Any] | None
) -> dict[str, str]:
    stored = loads(row["verdicts_json"]) or {}
    labels = {k: (v["label"] if isinstance(v, Mapping) else str(v)) for k, v in stored.items()}
    if rules is None or row["record_json"] is None:
        return labels
    record = loads(row["record_json"])
    fresh = evaluate_verdicts(rules, record, dict(constants or {}), row["family"], row["protocol"])
    return {k: v["label"] for k, v in fresh.items()}


def _per_seed_row(row: dict[str, Any], labels: dict[str, str], overrides: str) -> dict[str, Any]:
    record = loads(row["record_json"]) or {}
    probe = record.get("probe") or {}
    walls = record.get("walls") or {}
    stopped = record.get("stopped")
    failure = record.get("failure")
    trained = record.get("trained_state") or (record.get("at_stop") or {}).get("trained_state")
    v2 = record.get("probe_v2") or {}
    step_time = record.get("step_time") or {}
    return {
        "label": row["label"],
        "exp": row["exp_id"],
        "family": row["family"],
        "variant": row["variant"],
        "overrides": overrides,
        "seed": row["seed"],
        "status": row["status"],
        **{f"verdict:{k}": v for k, v in labels.items()},
        "dev_acc": row["final_dev_accuracy"],
        "dev_loss": row["final_dev_loss"],
        "first_query_acc": probe.get("first_query_accuracy"),
        "follow": probe.get("correct_follow_frac"),
        "unchanged": probe.get("correct_unchanged_frac"),
        "v2_follow": v2.get("correct_follow_frac"),
        "v2_unchanged": v2.get("correct_unchanged_frac"),
        "v2_vocab": v2.get("vocab_size"),
        "v2_matched": v2.get("matched_frac"),
        "write_rate": (trained or {}).get("gate/write_rate"),
        "step_s": step_time.get("run_median_s"),
        "early_stop": f"{stopped['outcome']} @ {stopped['step']}" if stopped else "",
        "failure": (
            f"{failure.get('reason')} @ {failure.get('last_step')}"
            if isinstance(failure, Mapping)
            else ""
        ),
        "train_s": row["train_s"] if row["train_s"] is not None else walls.get("fit_s"),
        "total_s": row["total_s"],
        "compile_s": walls.get("compile_s"),
        "card": row["card_status"] or "",
        "commit": (row["commit_sha"] or "")[:8],
        "imported": bool(row["imported"]),
        "group_id": row["group_id"],
    }


def build_report(
    db: Database,
    *,
    sweep: str | None = None,
    spec: str | Path | None = None,
    statuses: Sequence[str] | None = None,
    family: str | None = None,
    protocol: str | None = None,
    imported: bool | None = None,
    per_seed: bool = False,
) -> Report:
    rules: RuleSet | None = None
    constants_by_arm: dict[str, dict[str, Any]] = {}
    label_by_arm: dict[str, str] = {}
    arm_ids: list[str] | None = None
    min_agree: dict[str, int] = {}
    if spec is not None:
        from hsn.experiments.spec import expand, load_spec

        parsed, _text = load_spec(spec)
        rules = parse_rules(parsed.rules)
        arms = expand(parsed)
        arm_ids = [a.arm_id for a in arms]
        constants_by_arm = {a.arm_id: a.constants for a in arms}
        label_by_arm = {a.arm_id: a.label for a in arms}
        min_agree = {k: v.min_agree for k, v in rules.verdicts.items() if v.min_agree}
        verdict_names = list(rules.verdicts)
    elif sweep is not None:
        latest = db.row(
            "SELECT rules_json FROM sweeps WHERE name = ? ORDER BY id DESC LIMIT 1", (sweep,)
        )
        stored = loads(latest["rules_json"]) if latest else {}
        verdicts = (stored or {}).get("verdicts") or {}
        min_agree = {
            k: v["group"]["min_agree"]
            for k, v in verdicts.items()
            if isinstance(v, Mapping) and v.get("group")
        }
        verdict_names = list((stored or {}).get("verdict_order") or verdicts)
    else:
        verdict_names = []
    rows = _select(
        db,
        sweep=sweep,
        arm_ids=arm_ids,
        statuses=statuses,
        family=family,
        protocol=protocol,
        imported=imported,
        all_finished=per_seed,
    )
    overrides = _distinct_overrides(rows)
    per = []
    for r in rows:
        if r["arm_id"] in label_by_arm and not r["imported"]:
            r["label"] = label_by_arm[r["arm_id"]]
        labels = _verdict_labels(r, rules, constants_by_arm.get(r["arm_id"]))
        for name in labels:
            if name not in verdict_names:
                verdict_names.append(name)
        per.append(_per_seed_row(r, labels, overrides[r["exp_id"]]))
    vcols = [f"verdict:{v}" for v in verdict_names]
    if per_seed:
        columns = [
            "label",
            "exp",
            "family",
            "variant",
            "overrides",
            "seed",
            "status",
            *vcols,
            "dev_acc",
            "dev_loss",
            "first_query_acc",
            "follow",
            "unchanged",
            "v2_follow",
            "v2_unchanged",
            "v2_vocab",
            "v2_matched",
            "write_rate",
            "step_s",
            "early_stop",
            "failure",
            "train_s",
            "total_s",
            "compile_s",
            "card",
            "commit",
            "imported",
        ]
        return Report(columns, per, verdict_names)
    groups: dict[str, list[dict[str, Any]]] = {}
    for p in per:
        groups.setdefault(p["group_id"], []).append(p)
    out = []
    for members in groups.values():
        finished = [m for m in members if m["status"] in FINISHED]
        first = members[0]
        row: dict[str, Any] = {
            "label": first["label"].rsplit("/s", 1)[0],
            "family": first["family"],
            "variant": first["variant"],
            "overrides": first["overrides"],
            "seeds": f"{len(finished)}/{len(members)}",
            "status": ", ".join(
                f"{s} {n}" for s, n in sorted(Counter(m["status"] for m in members).items())
            ),
        }
        for name, col in zip(verdict_names, vcols, strict=True):
            seen = [str(m[col]) for m in finished if m.get(col)]
            if not seen:
                row[col] = ""
                continue
            label, count = Counter(seen).most_common(1)[0]
            cell = f"{label} {count}/{len(finished)}"
            need = min_agree.get(name)
            if (
                need is not None
                and count >= need
                and not label.startswith(("n/a", "stopped early", "failed"))
            ):
                cell += " (finding)"
            row[col] = cell
        row["dev_acc"] = _spread([m["dev_acc"] for m in finished if m["dev_acc"] is not None])
        row["dev_loss"] = _spread([m["dev_loss"] for m in finished if m["dev_loss"] is not None])
        row["first_query_acc"] = _spread(
            [m["first_query_acc"] for m in finished if m["first_query_acc"] is not None], 3
        )
        row["follow"] = _spread([m["follow"] for m in finished if m["follow"] is not None], 3)
        row["unchanged"] = _spread(
            [m["unchanged"] for m in finished if m["unchanged"] is not None], 3
        )
        for key in ("v2_follow", "v2_unchanged", "v2_vocab", "v2_matched"):
            row[key] = _spread([float(m[key]) for m in finished if m[key] is not None], 3)
        row["write_rate"] = _spread(
            [m["write_rate"] for m in finished if m["write_rate"] is not None], 3
        )
        times = [float(m["step_s"]) for m in members if m["step_s"] is not None]
        row["step_s"] = f"{statistics.median(times):.3f}" if times else ""
        row["early_stop"] = ", ".join(m["early_stop"] for m in members if m["early_stop"])
        row["failure"] = ", ".join(m["failure"] for m in members if m["failure"])
        for key in ("train_s", "total_s", "compile_s"):
            values = [float(m[key]) for m in finished if m[key] is not None]
            row[key] = f"{statistics.median(values):.0f}" if values else ""
        row["card"] = ", ".join(sorted({m["card"] for m in members if m["card"]}))
        row["commit"] = ", ".join(sorted({m["commit"] for m in members if m["commit"]}))
        row["imported"] = "yes" if any(m["imported"] for m in members) else ""
        out.append(row)
    columns = [
        "label",
        "family",
        "variant",
        "overrides",
        "seeds",
        "status",
        *vcols,
        "dev_acc",
        "dev_loss",
        "first_query_acc",
        "follow",
        "unchanged",
        "v2_follow",
        "v2_unchanged",
        "v2_vocab",
        "v2_matched",
        "write_rate",
        "step_s",
        "early_stop",
        "failure",
        "train_s",
        "total_s",
        "compile_s",
        "card",
        "commit",
        "imported",
    ]
    return Report(columns, out, verdict_names)


def _cell(value: Any) -> str:
    text = _fmt(value)
    return text.replace("|", "\\|").replace("\n", " ")


def format_markdown(report: Report) -> str:
    header = [c.removeprefix("verdict:") for c in report.columns]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in report.rows:
        lines.append("| " + " | ".join(_cell(row.get(c)) for c in report.columns) + " |")
    return "\n".join(lines) + "\n"


def format_csv(report: Report) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=report.columns, extrasaction="ignore")
    writer.writeheader()
    for row in report.rows:
        writer.writerow({c: _fmt(row.get(c), 6) for c in report.columns})
    return buf.getvalue()
