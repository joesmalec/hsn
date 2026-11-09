"""Compact live controller reporting and read-only evidence summaries (no torch import)."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, TypeGuard

from hsn.experiments.db import Database, DatabaseError

ABLATIONS = ("loss_cuts", "gradient_cuts", "update_cuts", "trial_increases")


def controller_progress(trainer: Any) -> dict[str, Any]:
    """Snapshot only the latest observation/change, never copy the growing event history.

    Optimizer group rates already include warmup/decay and component multipliers. A component
    can have several rates (e.g. a separate cell rate), so retain all distinct rates. These are
    the rates scheduled for the NEXT update, since the trainer has advanced its scheduler.
    """
    controller = getattr(trainer, "adaptive_lr", None)
    if controller is None:
        return {}
    state = controller.state
    observations = state.observations
    latest = observations[-1] if observations else None
    adjustment = next((o for o in reversed(observations) if o["before"] != o["scales"]), None)
    rates: dict[str, set[float]] = {}
    for group in trainer.optimizer.param_groups:
        rates.setdefault(group["adaptive_group"], set()).add(float(group["lr"]))
    return {
        "adaptive_lr": {
            "report_step": int(trainer.step),
            "accepted_steps": state.accepted_steps,
            "observation_count": len(observations),
            "monitor_source": "train",
            "monitor_fingerprint": state.monitor_fingerprint,
            "scales": dict(state.scales),
            "next_update_lrs": {key: sorted(values) for key, values in rates.items()},
            "cooldown_windows": state.cooldown,
            "trial": None if state.trial is None else state.trial.model_dump(mode="json"),
            "trial_blocked": state.trial_blocked,
            "enabled_signals": {name: bool(getattr(controller.cfg, name)) for name in ABLATIONS},
            "latest_observation": copy.deepcopy(latest),
            "last_adjustment": copy.deepcopy(adjustment),
        }
    }


def format_rates(values: list[float]) -> str:
    return "/".join(f"{value:.3g}" for value in values)


def _finite(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def signal_alerts(observation: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, bool]:
    """Raw warning thresholds, independently of whether that signal's actions are enabled."""
    loss, anchor = observation.get("loss"), observation.get("anchor_loss")
    grad, healthy = observation.get("grad_median"), observation.get("healthy_grad")
    worse = bool(_finite(loss) and _finite(anchor) and loss > anchor * (1 + config["threshold"]))
    growing = bool(
        _finite(grad) and _finite(healthy) and grad > max(healthy, 1e-12) * config["grad_growth"]
    )
    updates = observation.get("update_ratios") or {}
    alerts = {"loss_regression": worse, "loss_and_gradient_growth": worse and growing}
    for group, key in (
        ("shared", "max_update_ratio"),
        ("router", "router_max_update_ratio"),
        ("decoder", "readout_max_update_ratio"),
    ):
        if group in updates:
            alerts[f"update/{group}"] = bool(
                _finite(updates[group]) and updates[group] > config[key]
            )
    return alerts


def summarize_signals(controller: Mapping[str, Any]) -> dict[str, Any]:
    """Describe subsequent monitor losses, without calling correlation causal usefulness.

    An action can prevent the very regression a warning predicts. Exclude intervals beginning
    with a rate change, cooldown, warmup, or active trial from warning follow-through counts.
    A guard between measurements breaks the pair. Report action outcomes separately.
    """
    config = controller["config"]
    observations = controller.get("observations", [])
    predictive: dict[str, dict[str, int | float | None]] = {}
    outcomes = []
    excluded: Counter[str] = Counter()
    for current, following in pairwise(observations):
        if not _finite(current.get("loss")) or not _finite(following.get("loss")):
            excluded["guard_or_missing_loss"] += 1
            continue
        changed = current["before"] != current["scales"]
        delta = float(following["loss"] - current["loss"])
        if changed:
            outcomes.append(
                {
                    "step": current["step"],
                    "decision": current["decision"],
                    "next_step": following["step"],
                    "next_monitor_loss_delta": delta,
                }
            )
            excluded["rate_changed"] += 1
            continue
        if current["decision"] in {"baseline", "warmup", "cooldown"} or current[
            "decision"
        ].startswith("trial_"):
            excluded["warmup_cooldown_or_trial"] += 1
            continue
        regressed = following["loss"] > current["loss"] * (1 + config["threshold"])
        for name, alert in signal_alerts(current, config).items():
            count = predictive.setdefault(
                name,
                {
                    "eligible_pairs": 0,
                    "alerts": 0,
                    "alerts_followed_by_regression": 0,
                    "alerts_without_next_regression": 0,
                    "regressions_without_alert": 0,
                },
            )
            count["eligible_pairs"] = int(count["eligible_pairs"] or 0) + 1
            count["alerts"] = int(count["alerts"] or 0) + int(alert)
            key = "alerts_followed_by_regression" if regressed else "alerts_without_next_regression"
            count[key] = int(count[key] or 0) + int(alert)
            count["regressions_without_alert"] = int(count["regressions_without_alert"] or 0) + int(
                regressed and not alert
            )
    for counts in predictive.values():
        n = int(counts["alerts"] or 0)
        counts["fraction_followed_by_regression"] = (
            int(counts["alerts_followed_by_regression"] or 0) / n if n else None
        )
    return {
        "observations": len(observations),
        "decisions": dict(Counter(o["decision"] for o in observations)),
        "enabled_signals": {name: bool(config.get(name, True)) for name in ABLATIONS},
        "warning_follow_through": predictive,
        "excluded_pairs": dict(excluded),
        "rate_change_outcomes": outcomes,
        "interpretation": "Observational training-monitor evidence, not causal usefulness. "
        "A warning may prevent deterioration when acted on. Counts are correlated measurements; "
        "use matched-seed ablations and final quality to assess benefit.",
    }


def matched_ablations(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair full-controller runs with one-switch ablations only under matching provenance."""
    pairs = []
    for full in runs:
        if not all(full["flags"].values()) or full["pair_key"] is None:
            continue
        for ablated in runs:
            off = [key for key, enabled in ablated["flags"].items() if not enabled]
            if len(off) != 1 or ablated["pair_key"] != full["pair_key"]:
                continue
            a, b = full["final"], ablated["final"]
            if a.get("steps") != b.get("steps") or not all(
                _finite(x.get("dev_accuracy"))
                and _finite(x.get("dev_loss"))
                and x.get("eval_step") == x.get("steps")
                for x in (a, b)
            ):
                continue
            pairs.append(
                {
                    "signal": off[0],
                    "seed": full["seed"],
                    "full_experiment": full["exp_id"],
                    "ablated_experiment": ablated["exp_id"],
                    "steps": a["steps"],
                    "accuracy_gain_with_signal": a["dev_accuracy"] - b["dev_accuracy"],
                    "loss_reduction_with_signal": b["dev_loss"] - a["dev_loss"],
                }
            )
    return pairs


def controller_report(db: Database, exp_ids: Sequence[int]) -> dict[str, Any]:
    """Read completed evidence or a compact running snapshot; never mutate historical runs."""
    runs = []
    for exp_id in dict.fromkeys(exp_ids):
        row = db.row(
            "SELECT e.exp_id,e.status,e.commit_sha,e.dirty_json,e.progress_json,"
            "COALESCE(e.config_json,a.config_json) AS config_json,"
            "COALESCE(e.protocol_json,a.protocol_json) AS protocol_json,"
            "a.data_hash,a.family,a.variant,a.variant_json,a.early_stop_json,r.record_json "
            "FROM experiments e JOIN arms a USING(arm_id) "
            "LEFT JOIN results r USING(exp_id) WHERE e.exp_id=?",
            (exp_id,),
        )
        if row is None:
            raise DatabaseError(f"unknown experiment {exp_id}")
        config = json.loads(row["config_json"])
        progress = json.loads(row["progress_json"] or "{}")
        record = json.loads(row["record_json"] or "{}")
        controller = record.get("adaptive_lr")
        if controller is None and not progress.get("adaptive_lr"):
            raise DatabaseError(f"experiment {exp_id} has no recorded adaptive controller evidence")
        flags = {key: bool(config["train"]["adaptive_lr"].get(key, True)) for key in ABLATIONS}
        normalized = copy.deepcopy(config)
        normalized.pop("run_id", None)
        normalized.pop("notes", None)
        for key in ABLATIONS:
            normalized["train"]["adaptive_lr"].pop(key, None)
        # No pairing across code, dirty source, data, protocol or non-ablation configuration.
        pair_key = None
        protocol = json.loads(row["protocol_json"])
        final = record.get("final", {})
        horizon = protocol.get("steps") or config["train"]["steps"]
        if (
            row["status"] == "done"
            and row["commit_sha"]
            and row["dirty_json"] == "[]"
            and controller is not None
            and final.get("steps") == horizon
        ):
            pair_key = json.dumps(
                [
                    normalized,
                    row["commit_sha"],
                    row["data_hash"],
                    protocol,
                    row["family"],
                    row["variant"],
                    json.loads(row["variant_json"]),
                    json.loads(row["early_stop_json"]),
                    record.get("data_selection"),
                    controller.get("monitor_fingerprint"),
                ],
                sort_keys=True,
            )
        runs.append(
            {
                "exp_id": exp_id,
                "status": row["status"],
                "seed": config["seed"],
                "flags": flags,
                "pair_key": pair_key,
                "final": record.get("final", {}),
                "live": progress.get("adaptive_lr"),
                "signal_summary": None if controller is None else summarize_signals(controller),
            }
        )
    paired = matched_ablations(runs)
    return {
        "runs": [{k: v for k, v in run.items() if k != "pair_key"} for run in runs],
        "matched_ablations": paired,
        "comparison_policy": "Completed clean runs only; identical source commit, seed, data, "
        "protocol, training horizon and configuration except one controller switch. "
        "Positive deltas favor the full controller. Repeat across seeds; a single pair "
        "does not establish an advantage.",
    }
