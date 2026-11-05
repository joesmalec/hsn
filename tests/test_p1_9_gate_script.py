"""P1.9 tests: the G1 gate script (decision, package, test-once, resumability), the runner's HSN
registration and trainer selection, and the experiment configuration.

Everything runs on ``profile:cpu`` with a tiny dataset (kv and delayed in_range, 64 training
documents of at most 32 symbols), a tiny HSN shrunk from ``ref_a1`` with the P1.7 overrides and tiny
baselines for a handful of steps, so the whole file takes well under a minute on a hosted runner.
The one tiny gate run is shared by the tests through a module-scoped fixture; its decision is FAIL
(two seeds, a tiny model far below the threshold), which is a valid outcome of the gate and what
``test_p1_9_gate_script_fails_below_threshold`` asserts on measured, not mocked, numbers before it
drives the pure decision function with mocked results.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
import yaml
from hsn.config import load_config, load_hardware_profile
from hsn.data.splits import build_splits
from hsn.data.synthetic import DatasetSpec, TaskSpec, load_dataset_spec
from hsn.device import resolve, set_deterministic
from hsn.eval.results import METRICS_FILE, STORE_FILE, ResultStore, load_metrics_freeze
from hsn.model.hsn import HSN
from hsn.runner import MODEL_REGISTRY, build_model, make_trainer, run_seeds
from hsn.train.loop import Trainer
from hsn.train.tbptt import TBPTTTrainer

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
EXPERIMENT = REPO / "configs" / "experiments" / "g1_ref_a1.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
UNROLL = 32
TINY_MODEL = {
    "d": 8,
    "context_dim": 8,
    "n_cells": 4,
    "in_degree": 2,
    "n_units": 2,
    "k_select": 2,
    "hidden": 16,
    "scorer_hidden": 8,
}


@pytest.fixture(autouse=True, scope="module")
def _restore_determinism() -> Iterator[None]:
    """The tiny configs set ``train.deterministic``, which the trainer applies process-wide."""
    before = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        set_deterministic(before)


# --------------------------------------------------------------------------------------------------
# Fixtures: tiny data, tiny configurations, a fabricated junit report, one gate run
# --------------------------------------------------------------------------------------------------


def tiny_spec() -> DatasetSpec:
    tasks = []
    for family in ("kv", "delayed"):
        task = SPEC.task(family, "in_range")  # type: ignore[arg-type]
        tasks.append(
            TaskSpec.model_validate(
                {
                    **task.model_dump(),
                    "n_train": 64,
                    "n_dev": 16,
                    "n_test": 16,
                    "length_range": [16, UNROLL],
                    "dep_len_range": [4, 20],
                    "unroll": UNROLL,
                }
            )
        )
    return DatasetSpec(name="tiny_g1", unroll=UNROLL, tasks=tasks)


def load_script(name: str) -> ModuleType:
    """Load ``scripts/<name>.py`` as a module (registered in ``sys.modules`` so that dataclasses
    and ``from gate_g0 import ...`` inside the scripts resolve)."""
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_hsn_config(path: Path, manifest: Path, *, steps: int = 4) -> Path:
    """The experiment configuration shrunk to the P1.7 tiny model on the tiny data."""
    raw = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    raw["run_id"] = "g1_tiny"
    raw["model"] = {**raw["model"], **TINY_MODEL}
    raw["data"] = {**raw["data"], "dataset": "tiny_g1", "manifest": str(manifest)}
    raw["train"] = {
        **raw["train"],
        "unroll": UNROLL,
        "batch": 4,
        "micro_batch": 4,
        "warmup_steps": 1,
        "steps": steps,
        "eval_every": steps,
        "ckpt_every": steps,
        "stats_every": 2,
        "search_trials": 2,
        "deterministic": True,
    }
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def write_baseline_config(path: Path, manifest: Path, model_name: str, *, steps: int = 4) -> Path:
    baseline: dict[str, Any] = (
        {"gru": {"d_model": 32, "n_layers": 1, "dropout": 0.0}}
        if model_name == "gru"
        else {"transformer": {"d_model": 32, "n_layers": 1, "n_heads": 2, "ffn_hidden": 64}}
    )
    cfg = {
        "run_id": f"{model_name}_tiny",
        "seed": 0,
        "profile": "cpu",
        "model_name": model_name,
        "data": {
            "dataset": "tiny_g1",
            "manifest": str(manifest),
            "tasks": ["kv", "delayed"],
            "variants": ["in_range"],
        },
        "train": {
            "lr": 1.0e-3,
            "warmup_steps": 1,
            "schedule": "cosine",
            "unroll": UNROLL,
            "batch": 4,
            "micro_batch": 4,
            "steps": steps,
            "eval_every": steps,
            "ckpt_every": steps,
            "stats_every": 2,
            "search_trials": 2,
            "deterministic": True,
        },
        "baseline": baseline,
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def write_junit(path: Path, *, reset: str = "passed", extra_failure: bool = False) -> Path:
    """A pytest-style junit report with the P1.8 invariants named (``reset`` is the status of the
    reset invariant: passed | failed | skipped)."""
    cases = {
        "test_p1_8_reset_removes_prior_state[cpu]": reset,
        "test_p1_8_reset_removes_prior_state[mps]": "skipped",  # absent device: no evidence
        "test_p1_8_prediction_no_leakage": "passed",
        "test_p1_8_gradcheck_tiny_fp64": "passed",
        "test_p1_8_checkpoint_exact": "failed" if extra_failure else "passed",
        "test_p1_8_state_in_unit_interval[mps]": "skipped",
    }
    failures = sum(1 for s in cases.values() if s == "failed")
    skipped = sum(1 for s in cases.values() if s == "skipped")
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<testsuites>",
        f'<testsuite name="pytest" errors="0" failures="{failures}" skipped="{skipped}" '
        f'tests="{len(cases)}" time="1.0">',
    ]
    for name, status in cases.items():
        lines.append(f'<testcase classname="tests.invariants.test_p1_8" name="{name}" time="0.1">')
        if status == "failed":
            lines.append('<failure message="assert False">boom</failure>')
        elif status == "skipped":
            lines.append('<skipped type="pytest.skip" message="no mps">skip</skipped>')
        lines.append("</testcase>")
    lines += ["</testsuite>", "</testsuites>"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("data")
    built = build_splits(tiny_spec(), out)
    assert built.is_clean
    spec_path = out.parent / "tiny_g1.yaml"
    spec_path.write_text(
        yaml.safe_dump(json.loads(tiny_spec().model_dump_json())), encoding="utf-8"
    )
    return out


@pytest.fixture(scope="module")
def manifest(data_dir: Path) -> Path:
    return data_dir / "manifest.json"


def gate_argv(
    root: Path, data_dir: Path, configs: dict[str, Path], junit: Path, git_rev: str | None
) -> list[str]:
    return [
        "--profile",
        "cpu",
        "--seeds",
        "0,1",
        "--out",
        str(root / "evidence"),
        "--config",
        str(configs["hsn"]),
        "--baselines",
        str(configs["gru"]),
        str(configs["transformer"]),
        "--runs",
        str(root / "runs"),
        "--data-spec",
        str(data_dir.parent / "tiny_g1.yaml"),
        "--data-out",
        str(data_dir),
        "--no-golden",
        "--overfit-docs",
        "8",
        "--overfit-steps",
        "4",
        "--overfit-eval-every",
        "2",
        "--after-overfit-failure",
        "continue",
        "--trial",
        "model.k_select=9",  # invalid for the tiny HSN (k_select > n_cells): a recorded failure
        "--search-steps",
        "2",
        "--profile-steps",
        "1",
        "--idle-profile-steps",
        "1",
        "--eval-batch",
        "4",
        "--invariants-junit",
        str(junit),
        "--invariants-git-rev",
        str(git_rev),
        "--jobs",
        "1",
    ]


@pytest.fixture(scope="module")
def gate_run(
    tmp_path_factory: pytest.TempPathFactory, data_dir: Path, manifest: Path
) -> dict[str, Any]:
    """One tiny gate run on cpu, shared by the tests; returns its paths and exit code."""
    root = tmp_path_factory.mktemp("gate")
    configs = {
        "hsn": write_hsn_config(root / "hsn.yaml", manifest),
        "gru": write_baseline_config(root / "gru.yaml", manifest, "gru"),
        "transformer": write_baseline_config(root / "transformer.yaml", manifest, "transformer"),
    }
    junit = write_junit(root / "invariants_ok.xml")
    gate = load_script("gate_g1")
    rev, _dirty = gate.git_state()
    argv = gate_argv(root, data_dir, configs, junit, rev)
    code = gate.main(argv)
    return {
        "gate": gate,
        "root": root,
        "argv": argv,
        "code": code,
        "out": root / "evidence",
        "runs": root / "runs",
        "configs": configs,
        "decision": read_json(root / "evidence" / "decision.json"),
    }


# --------------------------------------------------------------------------------------------------
# Tests named by the chunk
# --------------------------------------------------------------------------------------------------


def test_p1_9_gate_script_fails_below_threshold(gate_run: dict[str, Any], tmp_path: Path) -> None:
    gate, out, runs = gate_run["gate"], gate_run["out"], gate_run["runs"]
    decision = gate_run["decision"]
    # the tiny run: measured numbers far below the threshold, two seeds instead of five -> FAIL
    assert gate_run["code"] == 1
    assert decision["decision"] == "FAIL" and decision["gate"] == "G1"
    checks = decision["checks"]
    assert checks["threshold_per_seed_per_task"] is False and checks["n_seeds"] is False
    assert (
        checks["invariants_passed"]
        and checks["correct_reset"]
        and checks["overfit_64_examples"] is False
    )
    assert checks["gradients_finite"] and checks["test_evaluated_once"] and checks["leakage_checks"]
    assert checks["environment_consistent"] is True  # --invariants-git-rev is this revision
    assert decision["threshold"] == {"kv/in_range": 0.95, "delayed/in_range": 0.95}
    assert sorted(decision["per_seed"]) == ["0", "1"]
    for entry in decision["per_seed"].values():
        for task in ("kv/in_range", "delayed/in_range"):
            cell = entry["tasks"][task]
            assert 0.0 <= cell["test_accuracy"] < 0.95 and cell["passed"] is False
            assert isinstance(cell["dev_accuracy"], float)
    assert any("test accuracy" in reason for reason in decision["reasons"])
    # the package exists and the store carries the decision and one test row per run
    package = Path(decision["package"]["path"])
    assert package == out / "G1.zip" and package.is_file()
    assert len(decision["package"]["sha256"]) == 64
    store = ResultStore(runs / STORE_FILE)
    assert store.query(run_id="G1", metric="decision").column("text").to_pylist() == ["FAIL"]
    test_results = read_json(out / "test_results.json")["runs"]
    hsn_runs = decision["hsn_runs"]
    assert len(hsn_runs) == 2 and all(r in test_results for r in hsn_runs)
    for run_id, record in test_results.items():
        assert set(record["per_group"]) == {"kv/in_range", "delayed/in_range"}
        assert (
            store.query(run_id=run_id, split="test", group="all", metric="accuracy").num_rows == 1
        )
        assert (runs / run_id / "test_eval.json").is_file()
        assert record["step"] == 4 and record["checkpoint"] == "step_00000004.pt"
    assert len(test_results) == 6  # 3 methods x 2 seeds; the rerun is not test-evaluated
    # every configuration tried is recorded: the invalid trial failed, the others ran
    search = read_json(out / "search.json")
    assert set(search["methods"]) == {"hsn", "gru", "transformer"}
    hsn_trials = search["methods"]["hsn"]["trials"]
    assert [t["index"] for t in hsn_trials] == [0, 1]
    assert (
        hsn_trials[0]["status"] == "ok" and hsn_trials[0]["result"]["g1_mean_dev_accuracy"] >= 0.0
    )
    assert hsn_trials[1]["status"] == "failed" and "k_select" in hsn_trials[1]["error"]
    assert (
        search["methods"]["hsn"]["best_index"] == 0 and search["methods"]["hsn"]["trials_used"] == 2
    )
    assert search["methods"]["gru"]["trials_used"] == 2
    failed = read_json(out / "failed_configurations.json")
    assert any(
        e["kind"] == "invalid_trial" and e["model"] == "hsn"
        for e in failed["failed_configurations"]
    )
    assert failed["negative_findings"] == decision["reasons"]
    overfit = read_json(out / "overfit.json")
    assert overfit["n_docs"] == 8 and overfit["steps_run"] == 4 and overfit["passed"] is False
    assert [e["step"] for e in overfit["evals"]] == [2, 4] and overfit["grad_norms_finite"]
    assert decision["rerun"]["passed"] is True and decision["reports"]["rerun_reproduces"]
    assert decision["reports"]["profiler_exists"] and decision["reports"]["baselines_exist"]
    # the pure decision function with mocked results
    frozen = load_metrics_freeze()

    def seeds(n: int, **edits: float) -> list[dict[str, Any]]:
        rows = []
        for seed in range(n):
            tasks = {"kv/in_range": 0.97, "delayed/in_range": 0.96}
            rows.append(
                {
                    "run_id": f"r{seed}",
                    "seed": seed,
                    "test": dict(tasks),
                    "dev": dict(tasks),
                    "grad_norms_finite": True,
                }
            )
        for key, value in edits.items():
            seed, _, task = key.partition("_")
            rows[int(seed[1:])]["test"][task.replace("__", "/")] = value
        return rows

    ok = dict(
        invariants_passed=True,
        correct_reset=True,
        environment_consistent=True,
        overfit_passed=True,
        test_evaluated_once=True,
    )
    verdict = gate.decide(frozen, seeds(5), **ok)
    assert verdict["decision"] == "PASS" and all(verdict["checks"].values())
    assert gate.decide(frozen, seeds(5, s2_kv__in_range=0.95), **ok)["decision"] == "PASS"
    below = gate.decide(frozen, seeds(5, s3_delayed__in_range=0.94), **ok)
    assert below["decision"] == "FAIL" and below["checks"]["threshold_per_seed_per_task"] is False
    assert any("seed 3 delayed/in_range" in r for r in below["reasons"])
    assert below["per_seed"]["3"]["tasks"]["delayed/in_range"]["passed"] is False
    assert gate.decide(frozen, seeds(4), **ok)["checks"]["n_seeds"] is False
    rows = seeds(5)
    rows[1]["grad_norms_finite"] = False
    assert gate.decide(frozen, rows, **ok)["checks"]["gradients_finite"] is False
    for flag in ok:
        assert gate.decide(frozen, seeds(5), **{**ok, flag: False})["decision"] == "FAIL"
    assert gate.decide(frozen, seeds(5), **ok, leakage_clean=False)["decision"] == "FAIL"
    # gradient norms: a null (the runner's spelling of a non-finite value) fails the check
    history = {
        "steps": [
            {"step": 1, "grad_norm": 0.5, "stats": {"grad_norm/cell": 0.1, "eps_mean": 0.0}},
            {"step": 2, "grad_norm": 0.4, "stats": {"grad_norm/cell": None}},
        ]
    }
    report = gate.grad_norms_finite(history)
    assert report["finite"] is False and report["n_steps"] == 2 and report["n_group_norms"] == 2
    history["steps"][1]["stats"]["grad_norm/cell"] = 0.2
    assert gate.grad_norms_finite(history)["finite"] is True
    assert gate.grad_norms_finite({"steps": []})["finite"] is False
    # the invariant report: a failure or a missing / failed reset invariant is not evidence
    failing = write_junit(tmp_path / "failing.xml", extra_failure=True)
    parsed = gate.parse_junit(failing)
    assert parsed["tests"] == 6 and parsed["failures"] == 1 and parsed["skipped"] == 2
    verdict_inv = gate.invariants_verdict(parsed, None)
    assert verdict_inv["passed"] is False and verdict_inv["correct_reset"] is True
    reset_failed = gate.parse_junit(write_junit(tmp_path / "reset.xml", reset="failed"))
    assert gate.invariants_verdict(reset_failed, 1)["correct_reset"] is False
    reset_skipped = gate.parse_junit(write_junit(tmp_path / "skipped.xml", reset="skipped"))
    assert gate.invariants_verdict(reset_skipped, 0)["correct_reset"] is False  # no variant ran
    good = gate.invariants_verdict(gate.parse_junit(write_junit(tmp_path / "ok.xml")), 0)
    assert good["passed"] and good["correct_reset"]
    assert (
        gate.invariants_verdict(gate.parse_junit(write_junit(tmp_path / "rc.xml")), 2)["passed"]
        is False
    )
    # a second invocation with the same arguments retrains nothing and decides again
    before = {p.name: p.stat().st_mtime_ns for p in runs.rglob(METRICS_FILE)}
    n_rows = len(store)
    assert gate.main(gate_run["argv"]) == 1
    after = {p.name: p.stat().st_mtime_ns for p in runs.rglob(METRICS_FILE)}
    assert after == before
    assert len(store) == n_rows + 1  # the decision row only: test evaluations are not repeated
    again = read_json(out / "decision.json")
    assert (
        Path(again["package"]["path"]).name.startswith("G1-")
        and Path(again["package"]["path"]).is_file()
    )
    assert again["per_seed"] == decision["per_seed"]
    # a GPU profile needs an explicit rerun tolerance; another task set is refused
    with pytest.raises(SystemExit):
        gate.main(["--profile", "m_series", "--out", str(tmp_path / "x")])
    bad = yaml.safe_load(gate_run["configs"]["hsn"].read_text(encoding="utf-8"))
    bad["data"]["tasks"] = ["kv"]
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen G1 task set"):
        gate.main(
            [
                "--profile",
                "cpu",
                "--out",
                str(tmp_path / "y"),
                "--config",
                str(tmp_path / "bad.yaml"),
            ]
        )
    # a read junit without its revision cannot show the environment consistent
    no_rev = gate.run_invariant_suite(tmp_path / "inv", write_junit(tmp_path / "j.xml"), None)
    assert no_rev["mode"] == "read" and no_rev["git_rev"] is None and no_rev["passed"]


def test_p1_9_package_contains_profiler_trace(gate_run: dict[str, Any]) -> None:
    decision, out = gate_run["decision"], gate_run["out"]
    with zipfile.ZipFile(out / "G1.zip") as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("G1/MANIFEST.json"))
        summary = zf.read("G1/SUMMARY.md").decode("utf-8")
        run_ids = [decision["rerun"]["rerun_run_id"]]
        for table in decision["methods"].values():
            run_ids += [row["run_id"] for row in table["runs"] if "run_id" in row]
        assert set(decision["hsn_runs"]) <= set(run_ids)
        assert len(run_ids) == 7
        traces = set(manifest["artifacts"]["profiler_traces"])
        for run_id in run_ids:
            for suffix in ("train", "serve"):
                name = f"profiles/{run_id}.{suffix}.json"
                assert name in traces and f"G1/{name}" in names
            profile = json.loads(zf.read(f"G1/profiles/{run_id}.train.json"))
            assert set(profile["segments"]) == {"startup", "shared_path", "backward", "optimizer"}
            assert profile["segments"]["shared_path"]["n"] == 1
            assert profile["power"]["method"] == "none"
            assert profile["segments"]["shared_path"]["launches"] == -1  # not counted on cpu
        assert "Profiler traces and energy readings" in summary
        # the idle-GPU profiling pass of every method and the gate's own files are at the root
        for model in ("hsn", "gru", "transformer"):
            for suffix in ("train", "serve"):
                assert f"G1/profiler_g1_{model}_{suffix}.json" in names
            assert decision["idle_profiles"][model]["step_wall_ms_p50"] > 0.0
        for name in (
            "decision.json",
            "leakage.json",
            "overfit.json",
            "search.json",
            "invariants.json",
            "invariants.xml",
            "test_results.json",
            "failed_configurations.json",
        ):
            assert f"G1/{name}" in names, name
        assert json.loads(zf.read("G1/failed_configurations.json")) == read_json(
            out / "failed_configurations.json"
        )
        assert (
            "None recorded" not in summary and "nothing here touched the test split" not in summary
        )
        assert "invalid_trial" not in summary or "failed configuration" in summary
        assert "`test_results.json`" in summary
        assert "**FAIL**" in summary
        uncertainty = json.loads(zf.read("G1/results/uncertainty.json"))
        assert set(uncertainty["per_model"]) == {"hsn", "gru", "transformer"}
        assert uncertainty["per_model"]["hsn"]["n_units"] == 4  # 2 tasks x 2 seeds, no trial runs
        assert "hsn - gru" in uncertainty["paired"]


# --------------------------------------------------------------------------------------------------
# Further tests
# --------------------------------------------------------------------------------------------------


def test_p1_9_gate_script_jobs_parallel(gate_run: dict[str, Any], tmp_path: Path) -> None:
    """``--jobs 2`` runs trials, seeds and the dependent rerun as concurrent subprocesses and
    reaches the same measured numbers as the in-process run; ``--stop-after`` resumes."""
    gate = gate_run["gate"]
    argv = list(gate_run["argv"])
    argv[argv.index("--out") + 1] = str(tmp_path / "evidence")
    argv[argv.index("--runs") + 1] = str(tmp_path / "runs")
    argv[argv.index("--jobs") + 1] = "2"
    argv += ["--poll-seconds", "0.2"]
    assert gate.main([*argv, "--stop-after", "search"]) == 0
    assert (tmp_path / "evidence" / "search.json").is_file()
    assert not (tmp_path / "evidence" / "decision.json").exists()
    assert not any(p.name.startswith("g1_tiny-cpu-s") for p in (tmp_path / "runs").iterdir())
    assert gate.main(argv) == 1
    decision = read_json(tmp_path / "evidence" / "decision.json")
    assert decision["jobs"] == 2 and decision["decision"] == "FAIL"
    assert decision["per_seed"] == gate_run["decision"]["per_seed"]  # bit-exact on cpu
    assert decision["rerun"]["passed"] and decision["rerun"]["model_hash_equal"]
    assert (
        decision["timings_s"]["search"]
        == read_json(tmp_path / "evidence" / "search.json")["seconds"]
    )
    logs = {p.name for p in (tmp_path / "evidence" / "logs").iterdir()}
    assert "g1_tiny-cpu-s0.log" in logs and "g1_tiny-cpu-s0-rerun.log" in logs
    assert "g1_tiny-trial0-cpu-s0.log" in logs
    # the dependency logic of the launcher itself
    ledger = gate.Ledger(tmp_path / "ledger.jsonl")
    units = [
        gate.Unit(
            "a", "m", None, 0, [], tmp_path, "x", "cpu", kind="rerun", source=tmp_path / "nope"
        ),
        gate.Unit(
            "b",
            "m",
            None,
            0,
            [],
            tmp_path,
            "y",
            "cpu",
            kind="rerun",
            source=tmp_path / "nope2",
            depends_on="a",
        ),
    ]
    results = gate.run_units(
        units, jobs=1, profile_steps=1, power="none", logs_dir=tmp_path / "l", ledger=ledger
    )
    assert [r.status for r in results] == ["failed", "failed"]
    assert "dependency a failed" in str(results[1].error)
    assert len(ledger.entries) == 2 and all(e["kind"] == "failed_run" for e in ledger.entries)
    with pytest.raises(ValueError, match="unknown unit"):
        gate.run_units(
            [units[1]],
            jobs=1,
            profile_steps=1,
            power="none",
            logs_dir=tmp_path / "l",
            ledger=ledger,
        )


def test_p1_9_runner_trains_hsn(manifest: Path, tmp_path: Path) -> None:
    """``hsn`` is registered, the runner picks the TBPTT trainer for it, and a run writes every
    artefact (the profiling pass detaches the HSN cell state)."""
    assert MODEL_REGISTRY["hsn"] == HSN.from_config
    config = write_hsn_config(tmp_path / "hsn.yaml", manifest, steps=2)
    cfg = load_config(config)
    ctx = resolve(load_hardware_profile("cpu"))
    model = build_model(cfg)
    assert isinstance(model, HSN)
    trainer = make_trainer(model, cfg, ctx, run_dir=tmp_path / "t")
    assert isinstance(trainer, TBPTTTrainer)
    gru_cfg = load_config(write_baseline_config(tmp_path / "gru.yaml", manifest, "gru", steps=2))
    gru_trainer = make_trainer(build_model(gru_cfg), gru_cfg, ctx, run_dir=tmp_path / "g")
    assert type(gru_trainer) is Trainer
    (run_dir,) = run_seeds(config, [3], out=tmp_path / "runs", profile="cpu", profile_steps=2)
    assert run_dir == tmp_path / "runs" / "g1_tiny-cpu-s3"
    expected = {
        "config.resolved.yaml",
        "run_manifest.json",
        "history.json",
        "metrics.json",
        "profile_train.json",
        "profile_serve.json",
        "run.log",
        "step_00000002.pt",
    }
    assert {p.name for p in run_dir.iterdir()} == expected
    metrics = read_json(run_dir / METRICS_FILE)
    assert metrics["model_name"] == "hsn" and metrics["steps"] == 2 and metrics["seed"] == 3
    assert metrics["measured"]["param_count"] == metrics["param_report"]["total_trainable"]
    assert metrics["measured"]["windows"] == 2.0 and metrics["grad_norms_finite"] is True
    assert set(metrics["final"]["per_group"]) == {"kv/in_range", "delayed/in_range"}
    history = read_json(run_dir / "history.json")
    assert [s["step"] for s in history["steps"]] == [1, 2]
    assert set(history["steps"][0]["parts"]) == {"task", "pred"}
    assert "grad_norm/cell" in history["steps"][1]["stats"]  # stats_every = 2
    profile = read_json(run_dir / "profile_train.json")
    assert profile["segments"]["shared_path"]["n"] == 2 and profile["segments"]["startup"]["n"] == 1
    assert read_json(run_dir / "profile_serve.json")["segments"]["shared_path"]["n"] >= 1


def test_p1_9_experiment_config_matches_ref_a1() -> None:
    """The committed experiment configuration: ref_a1's model block, dense routing, the frozen task
    set, and a search budget."""
    cfg = load_config(EXPERIMENT)
    ref = load_config(REF_A1)
    assert cfg.model == ref.model and cfg.model.routing == "dense"
    assert cfg.model_name == "hsn" and cfg.run_id == "g1_ref_a1"
    raw = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    assert raw["model"] == yaml.safe_load(REF_A1.read_text(encoding="utf-8"))["model"]
    frozen = load_metrics_freeze()
    assert sorted(cfg.data.tasks) == sorted({t.task for t in frozen.g1_tasks})
    assert cfg.data.variants == ["in_range"] and cfg.data.align_docs
    assert cfg.train.search_trials >= 1 and cfg.train.steps >= cfg.train.eval_every
    assert cfg.train.unroll == 256 and cfg.train.lambda_p == 1.0
    assert cfg.train.aux_enabled == [] and cfg.train.precision == "bf16"
    gate = load_script("gate_g1")
    gate.check_task_set(cfg, frozen, "experiment")
    assert len(gate.trial_sets(None)) == cfg.train.search_trials
    matched = gate.matched_overrides(cfg)
    assert f"train.steps={cfg.train.steps}" in matched and "data.tasks=[kv,delayed]" in matched
    gru = load_config(REPO / "configs" / "baselines" / "gru_2m.yaml", overrides=matched)
    assert gru.train.steps == cfg.train.steps and gru.train.batch == cfg.train.batch
    assert (
        gru.data.tasks == ["kv", "delayed"] and gru.train.search_trials == cfg.train.search_trials
    )


def test_p1_9_pick_best_excludes_memory_infeasible() -> None:
    """The search winner is the best trial that fits the seed stage's per-run memory budget; a
    better trial above it is marked excluded with the numbers (measured on rtx3060: the batch-32
    trial peaks at 5.69 GiB and three or two copies do not fit the 12 GB card)."""
    gate = load_script("gate_g1")

    def trial(index: int, mean: float, peak: float) -> dict[str, Any]:
        return {
            "index": index,
            "status": "ok",
            "peak_bytes": peak,
            "result": {
                gate.SEARCH_METRIC: mean,
                "g1_min_dev_accuracy": mean / 2,
                "dev_loss": 4.0 - mean,
            },
        }

    trials = [trial(0, 0.088, 3.3e9), trial(1, 0.089, 3.3e9), trial(2, 0.104, 6.1e9)]
    assert gate.pick_best(trials) == 2
    assert "excluded" not in trials[2]
    assert gate.pick_best(trials, max_peak_bytes=12.9e9 * 0.85 / 3) == 1
    assert "exceeds" in trials[2]["excluded"] and "excluded" not in trials[1]
    failed = [{"index": 0, "status": "failed", "error": "boom"}]
    assert gate.pick_best(failed) is None
    assert gate.pick_best([trial(0, 0.5, 9e9)], max_peak_bytes=1e9) is None
