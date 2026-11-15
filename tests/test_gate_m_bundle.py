"""Evidence integrity checks; no checkpoints, archived code execution or device access."""

from __future__ import annotations

import copy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts/gate_m_bundle.py"
_SPEC = importlib.util.spec_from_file_location("gate_m_bundle", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
bundle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bundle)


def example(label="test", fast=False):
    """A complete-shaped campaign record with one scored query per key group."""
    zero = {"accuracy": None, "n": 0, "docs": 0, "ci95": None}
    one = {"accuracy": 1.0, "n": 1, "docs": 1, "ci95": [1.0, 1.0]}
    matrix = {b: copy.deepcopy(one if b in ("near", "all") else zero) for b in bundle.BUCKETS}
    counts = {
        b: [[1, 1] if i == 0 and b in ("near", "all") else [0, 0] for i in range(600)]
        for b in bundle.BUCKETS
    }
    measured = {
        "matrix": matrix,
        "counts": counts,
        "predictions": [[5]] + [[] for _ in range(599)],
        "reference_match": True,
        "off_logits_bitwise_equal": True,
        "N1_answers_identical": True,
        "N2b_answer_predictions_identical": True,
    }
    n3 = ["N3-fast-kept", "N3-fast-zero"] if fast else ["N3"]
    groups = {}
    for key in bundle.GROUPS:
        delayed = key.startswith("delayed2h/")
        conditions = ("B1", "B2", "B3") if delayed else ("B1",)
        outcomes = {
            name: {
                b: copy.deepcopy(zero)
                if b not in ("near", "all")
                else {
                    **one,
                    "accuracy": float(name == "follows"),
                    "ci95": [float(name == "follows"), float(name == "follows")],
                }
                for b in bundle.BUCKETS
            }
            for name in ("follows", "unchanged", "other")
        }
        binding_counts = {
            b: [
                [1, 0, 0, 1] if i == 0 and b in ("near", "all") else [0, 0, 0, 0]
                for i in range(600)
            ]
            for b in bundle.BUCKETS
        }
        edits = {"B1": {"new": 6, "changed": [2]}}
        if delayed:
            edits["B2"] = {"new": 6, "changed": [1, 2, 4, 5]}
        groups[key] = {
            "documents_sha256": "same-documents-for-every-checkpoint",
            "necessity": {
                c: copy.deepcopy(measured)
                for c in ("N0", "N1", "N2a", "N2b-answer", "N2b-all", *n3, "N4")
            },
            "binding": {
                "query_fixed": True,
                "reference_solver_checked": True,
                "identity_swap_logits_exact": delayed,
                "conditions": {
                    c: {
                        "counts": copy.deepcopy(binding_counts),
                        "outcomes": copy.deepcopy(outcomes),
                    }
                    for c in conditions
                },
                "probes": [
                    {
                        "doc": 0,
                        "slot": 0,
                        "old": 5,
                        "baseline_prediction": 5,
                        "edits": edits,
                        "predictions": dict.fromkeys(conditions, 6),
                        "buckets": ["near", "all"],
                    }
                ],
            },
        }
    return {
        "label": label,
        "status": "complete",
        "version": 1,
        "docs": 600,
        "span": 32,
        "reach": 63,
        "averaged": True,
        "device": "cpu",
        "threads": 8,
        "fast_cells": 8 if fast else 0,
        "reader": True,
        "groups": groups,
        "sha256": "checkpoint-hash",
        "config_sha256": "config-hash",
        "state_removal_audit": {
            c: {"routed_state_zero": True, "positions_checked": 1, "fast_read_nonzero": fast}
            for c in ("N1", *n3)
        },
    }


@pytest.mark.parametrize("fast", [False, True])
def test_accepts_complete_record_with_separate_fast_conditions(fast):
    result = example(fast=fast)
    bundle.validate_result(result, "test")


def test_refuses_incomplete_or_missing_conditions():
    result = example()
    result["status"] = "running"
    with pytest.raises(ValueError, match="incomplete"):
        bundle.validate_result(result, "test")
    result["status"] = "complete"
    del result["groups"]["kv/training keys"]["necessity"]["N1"]
    with pytest.raises(ValueError, match="missing necessity"):
        bundle.validate_result(result, "test")


def test_refuses_failed_identity_swap_and_wrong_byte_count():
    result = example()
    binding = result["groups"]["delayed2h/seen pairs"]["binding"]
    binding["identity_swap_logits_exact"] = False
    with pytest.raises(ValueError, match="Identity state swap"):
        bundle.validate_result(result, "test")
    binding["identity_swap_logits_exact"] = True
    binding["probes"][0]["edits"]["B2"]["changed"] = [1, 2]
    with pytest.raises(ValueError, match="edited byte positions"):
        bundle.validate_result(result, "test")


def test_recomputes_binding_outcomes_from_queries():
    result = example()
    result["groups"]["delayed2h/seen pairs"]["binding"]["probes"][0]["predictions"]["B3"] = 5
    with pytest.raises(ValueError, match="Binding counts disagree"):
        bundle.validate_result(result, "test")


def test_refuses_wrong_metric_and_nonidentical_answers():
    result = example()
    necessity = result["groups"]["kv/training keys"]["necessity"]
    necessity["N0"]["matrix"]["near"]["n"] = 2
    with pytest.raises(ValueError, match="Answer count"):
        bundle.validate_result(result, "test")
    necessity["N0"]["matrix"]["near"]["n"] = 1
    necessity["N3"]["predictions"][0] = [9]
    with pytest.raises(ValueError, match="N1/N3 predictions"):
        bundle.validate_result(result, "test")


def test_bundle_is_lossless_and_requires_every_checkpoint(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    labels = [f"model-{i}" for i in range(13)]
    evaluator = ("RUNS = " + repr(dict.fromkeys(labels, ("run", "name", 18000))) + "\n").encode()
    reference = tmp_path / "hold_matrix.py"
    reference.write_bytes(b"# Unchanged reference\n")
    (source / "gate_m_interventions.py").write_bytes(evaluator)
    (source / "RESULTS.md").write_text(
        "\n".join(f"- {label}: complete; [raw]({label}_600.json)" for label in labels) + "\n"
    )
    result = example()
    result["script_sha256"] = bundle.digest(evaluator)
    result["hold_matrix_sha256"] = bundle.digest(reference.read_bytes())
    for label in labels[:-1]:
        result["label"] = label
        (source / f"{label}_600.json").write_text(json.dumps(result))
    with pytest.raises(FileNotFoundError):
        bundle.export_bundle(source, reference, out)
    assert not out.exists()
    result["label"] = labels[-1]
    result["script_sha256"] = "changed"
    last = source / f"{labels[-1]}_600.json"
    last.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Evaluated source changed"):
        bundle.export_bundle(source, reference, out)
    assert not out.exists()
    result["script_sha256"] = bundle.digest(evaluator)
    last.write_text(json.dumps(result))
    manifest = bundle.export_bundle(source, reference, out)
    assert len(manifest["checkpoints"]) == 13
    assert (out / "sources/gate_m_interventions.py.txt").read_bytes() == evaluator
    assert (out / "sources/hold_matrix.py.txt").read_bytes() == reference.read_bytes()
    for label, entry in manifest["checkpoints"].items():
        packed = (out / entry["file"]).read_bytes()
        raw = gzip.decompress(packed)
        assert raw == (source / f"{label}_600.json").read_bytes()
        assert bundle.digest(raw) == entry["raw_sha256"]
        assert bundle.digest(packed) == entry["gzip_sha256"]
        assert f"](raw/{label}_600.json.gz)" in (out / "RESULTS.md").read_text()
