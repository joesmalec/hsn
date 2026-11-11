"""Optional analytical views over existing records; never change the worker schema version.

Install with ``python -m hsn.experiments.measurements --db PATH``. Views read historical and
future records without backfilling or rewriting results. Missing measurements remain NULL.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from hsn.experiments.db import default_db_path

VIEWS = {
    "training_evaluations": """
        SELECT r.exp_id, CAST(j.key AS INTEGER) AS evaluation_index,
            json_extract(j.value, '$.step') AS step,
            json_extract(j.value, '$.accuracy') AS accuracy,
            json_extract(j.value, '$.loss') AS loss,
            json_extract(j.value, '$.per_group."delayed/in_range".accuracy') AS delayed_accuracy,
            json_extract(j.value, '$.split.later.accuracy') AS later_accuracy
        FROM results r, json_each(CASE WHEN json_valid(r.record_json)
            THEN r.record_json ELSE '{}' END, '$.evals') j
    """,
    "training_measurements": """
        WITH history AS (
            SELECT *, LAG(accuracy) OVER w AS previous_accuracy,
                MAX(accuracy) OVER w AS peak_so_far,
                MAX(delayed_accuracy) OVER w AS peak_delayed_so_far,
                MAX(later_accuracy) OVER w AS peak_later_so_far
            FROM training_evaluations
            WINDOW w AS (PARTITION BY exp_id ORDER BY evaluation_index
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        ), curves AS (
            SELECT exp_id, COUNT(*) AS evaluation_count, MAX(accuracy) AS peak_accuracy,
                CASE WHEN COUNT(*) > 1 THEN
                    MAX(MAX(0, 100 * (previous_accuracy - accuracy))) END AS max_adjacent_drop_pp,
                CASE WHEN COUNT(*) > 1 THEN
                    MAX(100 * (peak_so_far - accuracy)) END AS max_drawdown_pp,
                CASE WHEN COUNT(delayed_accuracy) > 1 THEN
                    MAX(100 * (peak_delayed_so_far - delayed_accuracy))
                    END AS max_delayed_drawdown_pp,
                CASE WHEN COUNT(later_accuracy) > 1 THEN
                    MAX(100 * (peak_later_so_far - later_accuracy)) END AS max_later_drawdown_pp
            FROM history GROUP BY exp_id
        ), records AS (
            SELECT exp_id, CASE WHEN json_valid(record_json) THEN record_json ELSE '{}' END AS raw,
                json_valid(record_json) AS record_json_valid, final_dev_accuracy
            FROM results
        )
        SELECT e.exp_id, e.label, e.status, e.commit_sha,
            json_extract(e.config_json, '$.seed') AS seed,
            json_extract(e.config_json, '$.model_name') AS model_name,
            json_extract(e.config_json, '$.data.tasks') AS tasks_json,
            a.data_hash, e.gpu_mem_peak_mib, r.record_json_valid,
            r.final_dev_accuracy,
            CASE WHEN r.final_dev_accuracy IS NOT NULL THEN
                json_extract(r.raw, '$.final."delayed/in_range/accuracy"')
                END AS final_delayed_accuracy,
            CASE WHEN r.final_dev_accuracy IS NOT NULL THEN
                json_extract(r.raw, '$.final."split/later/accuracy"') END AS final_later_accuracy,
            json_extract(r.raw, '$.final.eval_step') AS reported_eval_step,
            json_extract(r.raw, '$.final.dev_accuracy') AS reported_eval_accuracy,
            json_extract(r.raw, '$.failure') AS failure_json,
            json_array_length(r.raw, '$.skipped_steps') AS skipped_steps,
            json_array_length(r.raw, '$.rollbacks') AS rollback_count,
            json_extract(r.raw, '$.probe_v3."swap/follow_frac"') AS swap_follow,
            json_extract(r.raw, '$.probe_v3."swap/unchanged_frac"') AS swap_unchanged,
            json_extract(r.raw, '$.probe_v3."swap/correct_n"') AS swap_eligible,
            json_extract(r.raw, '$.probe_v3."swap/verdict"') AS binding_verdict,
            json_extract(r.raw, '$.step_time.run_median_s') AS median_step_s,
            c.evaluation_count, c.peak_accuracy, c.max_adjacent_drop_pp, c.max_drawdown_pp,
            c.max_delayed_drawdown_pp, c.max_later_drawdown_pp
        FROM experiments e LEFT JOIN records r USING(exp_id)
        LEFT JOIN curves c USING(exp_id) LEFT JOIN arms a USING(arm_id)
    """,
    "training_diagnostics": """
        SELECT r.exp_id, json_extract(j.value, '$.step') AS step,
            json_extract(j.value, '$."applied_weight_norm/router"') AS router_weight_norm,
            json_extract(j.value, '$."applied_update_norm/router"') AS router_update_norm,
            json_extract(j.value, '$."applied_descent_cosine/router"') AS router_descent_cosine,
            json_extract(j.value, '$."applied_update_norm/cell"') AS memory_update_norm,
            json_extract(j.value, '$."applied_state/gate/write_rate"') AS applied_write_rate,
            json_extract(j.value, '$."stability/route_churn"') AS route_churn,
            json_extract(j.value, '$."stability/reference_step"') AS route_reference_step,
            json_extract(j.value, '$."stability/state_gain_max"') AS state_gain_max,
            json_extract(j.value, '$."stability/logit_gain_max"') AS logit_gain_max,
            json_extract(j.value, '$."stability/eligible_rows"') AS sensitivity_eligible_rows,
            j.value AS diagnostics_json
        FROM results r, json_each(CASE WHEN json_valid(r.record_json)
            THEN r.record_json ELSE '{}' END, '$.diagnostics') j
    """,
    "training_gpu_samples": """
        SELECT e.exp_id, json_extract(j.value, '$.time') AS sampled_at_unix,
            json_extract(j.value, '$.step') AS last_reported_step,
            json_extract(j.value, '$.phase') AS phase,
            json_extract(j.value, '$.utilization_pct') AS device_utilization_pct,
            json_extract(j.value, '$.device_memory_mib') AS device_memory_mib,
            json_extract(j.value, '$.process_memory_mib') AS process_memory_mib
        FROM experiments e, json_each(CASE WHEN json_valid(e.card_json)
            THEN e.card_json ELSE '{}' END, '$.telemetry') j
    """,
    "training_steps": """
        SELECT r.exp_id, json_extract(j.value, '$.step') AS step,
            json_extract(j.value, '$.loss') AS loss,
            json_extract(j.value, '$.accuracy') AS accuracy,
            json_extract(j.value, '$.lr') AS learning_rate,
            CASE WHEN json_type(j.value, '$.grad_norm') IN ('real','integer')
                THEN json_extract(j.value, '$.grad_norm') END AS finite_grad_norm,
            json_extract(j.value, '$.grad_norm') AS grad_norm_recorded
        FROM results r, json_each(CASE WHEN json_valid(r.record_json)
            THEN r.record_json ELSE '{}' END, '$.train') j
    """,
}


def install(conn: sqlite3.Connection) -> None:
    """Atomically install/update only our named views; no result data or version changes."""
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        for name, select in VIEWS.items():
            conn.execute(f"DROP VIEW IF EXISTS {name}")
            conn.execute(f"CREATE VIEW {name} AS {select}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=default_db_path())
    args = parser.parse_args()
    with sqlite3.connect(args.db.expanduser().resolve().as_uri() + "?mode=rw", uri=True) as conn:
        install(conn)
    print("Installed: " + ", ".join(VIEWS))


if __name__ == "__main__":
    main()
