#!/usr/bin/env python3
"""Validate and package the completed legacy-checkpoint Gate M campaign.

This uses only the standard library. It never loads a checkpoint or accesses a GPU.
The evaluated scripts are preserved byte-for-byte as .py.txt evidence; the model
runtime is the detached reader checkout at 76c2aec, not the current main branch.

    python scripts/gate_m_bundle.py --source /path/to/gate_m \
        --reference /path/to/hold_matrix.py --out docs/gate_m
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

BUCKETS = ("near", "mid", "far", "far-strict", "all")
GROUPS = (
    "kv/training keys",
    "kv/dev keys",
    "kv/test keys",
    "delayed2h/seen pairs",
    "delayed2h/new pairs",
    "delayed2h/never-seen bytes",
)
RUNTIME = "76c2aec878af68f945040e909ffdfc03b71b4bd1"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def campaign_labels(source: bytes) -> tuple[str, ...]:
    """Read the declared campaign without importing the evaluated script."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RUNS" for target in node.targets
        ):
            runs = ast.literal_eval(node.value)
            require(isinstance(runs, dict) and len(runs) == 13, "Expected 13 frozen checkpoints")
            return tuple(runs)
    raise ValueError("Evaluator does not declare RUNS")


def check_metric(metric: dict[str, Any], rows: list[list[int]], col: int = 0) -> None:
    require(len(rows) == 600, "Expected 600 document count rows")
    for row in rows:
        require(all(type(v) is int and v >= 0 for v in row), "Invalid document counts")
        require(row[col] <= row[-1], "Numerator exceeds denominator")
    right, total = sum(row[col] for row in rows), sum(row[-1] for row in rows)
    require(metric["n"] == total, "Answer count does not match document counts")
    require(metric["docs"] == sum(row[-1] > 0 for row in rows), "Document count mismatch")
    if not total:
        require(metric["accuracy"] is None and metric["ci95"] is None, "Nonempty empty metric")
    else:
        require(metric["accuracy"] == right / total, "Accuracy does not match document counts")
        low, high = metric["ci95"]
        require(0 <= low <= high <= 1, "Invalid confidence interval")


def validate_result(result: dict[str, Any], label: str) -> None:
    require(result["label"] == label, "Checkpoint label mismatch")
    require(result["status"] == "complete", f"{label}: campaign is incomplete")
    require(result["version"] == 1 and result["docs"] == 600, "Unexpected evaluation protocol")
    require(result["span"] == 32 and result["reach"] == 63, "Unexpected distance masks")
    require(result["averaged"] is True and result["device"] == "cpu", "Unexpected execution")
    require(result["threads"] <= 8, "CPU thread limit exceeded")
    require(set(result["groups"]) == set(GROUPS), "Missing or extra task/key groups")
    n3 = ["N3-fast-kept", "N3-fast-zero"] if result["fast_cells"] else ["N3"]
    conditions = {"N0", "N1", "N2a", "N2b-answer", "N2b-all", *n3}
    if result["reader"]:
        conditions.add("N4")
    for condition in ("N1", *n3):
        audit = result["state_removal_audit"][condition]
        require(audit["routed_state_zero"] is True, "State-removal audit failed")
        if condition.startswith("N3"):
            require(audit["positions_checked"] > 0, "State-input audit did not run")
        if condition == "N3-fast-kept":
            require(audit["fast_read_nonzero"] is True, "Fast read was not retained")
    for key, group in result["groups"].items():
        necessity = group["necessity"]
        require(set(necessity) == conditions, f"{key}: missing necessity conditions")
        baseline = necessity["N0"]
        for flag in ("reference_match", "off_logits_bitwise_equal"):
            require(baseline[flag] is True, f"{key}: baseline invariant failed: {flag}")
        if "archived_reference_match" in baseline:
            require(baseline["archived_reference_match"] is True, "Archived baseline mismatch")
        for measured in necessity.values():
            require(len(measured["predictions"]) == 600, "Missing document predictions")
            require(
                [len(p) for p in measured["predictions"]]
                == [r[-1] for r in baseline["counts"]["all"]],
                "Prediction counts disagree with answer counts",
            )
            for bucket in BUCKETS:
                rows = measured["counts"][bucket]
                check_metric(measured["matrix"][bucket], rows)
                require(
                    [r[-1] for r in rows] == [r[-1] for r in baseline["counts"][bucket]],
                    "Intervention changed the scoring population",
                )
        require(
            necessity["N2b-answer"]["predictions"] == necessity["N2b-all"]["predictions"],
            "N2b answer/all predictions differ",
        )
        require(necessity["N2b-all"]["N2b_answer_predictions_identical"], "N2b check missing")
        if not result["fast_cells"]:
            require(
                necessity["N1"]["predictions"] == necessity["N3"]["predictions"],
                "N1/N3 predictions differ without fast cells",
            )
            require(necessity["N3"]["N1_answers_identical"], "N1/N3 check missing")
        binding = group["binding"]
        require(
            binding["query_fixed"] and binding["reference_solver_checked"], "Edit checks failed"
        )
        delayed = key.startswith("delayed2h/")
        expected = {"B1", "B2", "B3"} if delayed else {"B1"}
        require(set(binding["conditions"]) == expected, "Missing binding conditions")
        require(
            len(binding["probes"]) == baseline["matrix"]["all"]["n"],
            "Missing binding query records",
        )
        if delayed:
            require(binding["identity_swap_logits_exact"] is True, "Identity state swap failed")
        for probe in binding["probes"]:
            require(
                probe["baseline_prediction"]
                == baseline["predictions"][probe["doc"]][probe["slot"]],
                "Probe baseline mismatch",
            )
            for kind, edit in probe["edits"].items():
                require(edit["new"] != probe["old"], "Counterfactual answer did not change")
                require(
                    len(set(edit["changed"])) == (1 if kind == "B1" else 4),
                    "Wrong number of edited byte positions",
                )
        for condition, measured in binding["conditions"].items():
            reconstructed = {b: [[0, 0, 0, 0] for _ in range(600)] for b in BUCKETS}
            kind = "B1" if condition == "B3" else condition
            for probe in binding["probes"]:
                if kind not in probe["edits"] or probe["baseline_prediction"] != probe["old"]:
                    continue
                pred = probe["predictions"][condition]
                new = probe["edits"][kind]["new"]
                outcome = 0 if pred == new else 1 if pred == probe["old"] else 2
                for bucket in probe["buckets"]:
                    row = reconstructed[bucket][probe["doc"]]
                    row[outcome] += 1
                    row[3] += 1
            require(
                reconstructed == measured["counts"], "Binding counts disagree with query records"
            )
            for bucket in BUCKETS:
                rows = measured["counts"][bucket]
                for col, outcome in enumerate(("follows", "unchanged", "other")):
                    check_metric(measured["outcomes"][outcome][bucket], rows, col)


def export_bundle(source: Path, reference: Path, out: Path) -> dict[str, Any]:
    evaluator = (source / "gate_m_interventions.py").read_bytes()
    reference_bytes = reference.read_bytes()
    labels = campaign_labels(evaluator)
    report = (source / "RESULTS.md").read_text()
    records: list[tuple[str, bytes, dict[str, Any]]] = []
    document_hashes = None
    for label in labels:
        raw = (source / f"{label}_600.json").read_bytes()
        result = json.loads(raw)
        validate_result(result, label)
        require(result["script_sha256"] == digest(evaluator), "Evaluated source changed")
        require(result["hold_matrix_sha256"] == digest(reference_bytes), "Reference source changed")
        hashes = {key: group["documents_sha256"] for key, group in result["groups"].items()}
        require(
            document_hashes is None or document_hashes == hashes, "Checkpoints used different docs"
        )
        document_hashes = hashes
        require(f"- {label}: complete;" in report, "Report is incomplete or stale")
        records.append((label, raw, result))
    # Nothing is exported unless the entire campaign passes validation.
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "sources").mkdir(exist_ok=True)
    manifest: dict[str, Any] = {
        "runtime_commit": RUNTIME,
        "evaluator_sha256": digest(evaluator),
        "reference_sha256": digest(reference_bytes),
        "documents_sha256": document_hashes,
        "checkpoints": {},
    }
    for label, raw, result in records:
        filename = f"{label}_600.json.gz"
        packed = gzip.compress(raw, compresslevel=9, mtime=0)
        (out / "raw" / filename).write_bytes(packed)
        manifest["checkpoints"][label] = {
            "file": f"raw/{filename}",
            "raw_sha256": digest(raw),
            "gzip_sha256": digest(packed),
            "checkpoint_sha256": result["sha256"],
            "config_sha256": result["config_sha256"],
        }
        report = report.replace(f"]({label}_600.json)", f"](raw/{filename})")
    (out / "sources/gate_m_interventions.py.txt").write_bytes(evaluator)
    (out / "sources/hold_matrix.py.txt").write_bytes(reference_bytes)
    (out / "RESULTS.md").write_text(report)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_bundle(args.source, args.reference, args.out)
    print(f"Validated and exported {len(manifest['checkpoints'])} checkpoints to {args.out}")


if __name__ == "__main__":
    main()
