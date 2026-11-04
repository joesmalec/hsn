"""P0.8 tests: runner, result store, exact rerun, evidence package, freeze file, gate script.

Everything runs on ``profile:cpu`` with a tiny dataset (two in-range families, 64 training
documents of at most 32 symbols) and tiny baselines for a handful of steps, so the whole file takes
seconds on a hosted CI runner.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pytest
import torch
import yaml
from hsn.config import load_config
from hsn.data.splits import build_splits
from hsn.data.synthetic import DatasetSpec, TaskSpec, load_dataset_spec
from hsn.device import set_deterministic
from hsn.eval.results import (
    FREEZE_PATH,
    METRICS_FILE,
    REQUIRED_ARTIFACTS,
    RERUN_FILE,
    STORE_FILE,
    FreezeError,
    FreezeSchemaError,
    FreezeTamperedError,
    MissingArtifactError,
    ResultStore,
    RunRecord,
    evidence_package,
    freeze_body_hash,
    load_metrics_freeze,
    metrics_freeze_schema,
    write_metrics_freeze,
)
from hsn.runner import (
    MODEL_REGISTRY,
    build_model,
    main,
    model_hash,
    parse_seeds,
    register_model,
    rerun,
    run_seeds,
)

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "configs" / "_example.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
UNROLL = 32


@pytest.fixture(autouse=True, scope="module")
def _restore_determinism() -> Iterator[None]:
    """The tiny configs set ``train.deterministic``, which the trainer applies process-wide
    (``torch.use_deterministic_algorithms``); MPS tests of other chunks must not inherit it.
    Module scope, so it wraps the module-scoped ``runs`` fixture as well."""
    before = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        set_deterministic(before)


# --------------------------------------------------------------------------------------------------
# Fixtures: a tiny dataset on disk and tiny baseline configurations
# --------------------------------------------------------------------------------------------------


def tiny_spec(families: tuple[str, ...] = ("algorithmic", "kv")) -> DatasetSpec:
    tasks = []
    for family in families:
        task = SPEC.task(family, "in_range")  # type: ignore[arg-type]
        params = dict(task.params)
        if family == "algorithmic":
            params["n_max"] = 4
        tasks.append(
            TaskSpec.model_validate(
                {
                    **task.model_dump(),
                    "n_train": 64,
                    "n_dev": 16,
                    "n_test": 16,
                    "length_range": [16, UNROLL],
                    "dep_len_range": [8, 24] if family == "algorithmic" else [4, 20],
                    "unroll": UNROLL,
                    "params": params,
                }
            )
        )
    return DatasetSpec(name="tiny_v1", unroll=UNROLL, tasks=tasks)


@pytest.fixture(scope="module")
def manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("data")
    built = build_splits(tiny_spec(), out)
    assert built.is_clean
    return out / "manifest.json"


def write_config(
    path: Path, manifest: Path, model_name: str, *, steps: int = 6, run_id: str | None = None
) -> Path:
    baseline: dict[str, Any] = (
        {"gru": {"d_model": 32, "n_layers": 1, "dropout": 0.0}}
        if model_name == "gru"
        else {"transformer": {"d_model": 32, "n_layers": 1, "n_heads": 2, "ffn_hidden": 64}}
    )
    cfg = {
        "run_id": run_id or f"{model_name}_tiny",
        "seed": 0,
        "profile": "cpu",
        "model_name": model_name,
        "data": {
            "dataset": "tiny_v1",
            "manifest": str(manifest),
            "tasks": ["algorithmic", "kv"],
            "variants": ["in_range"],
        },
        "train": {
            "lr": 1.0e-3,
            "warmup_steps": 2,
            "schedule": "cosine",
            "unroll": UNROLL,
            "batch": 4,
            "micro_batch": 4,
            "steps": steps,
            "eval_every": steps,
            "ckpt_every": steps,
            "stats_every": steps,
            "deterministic": True,
        },
        "baseline": baseline,
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory, manifest: Path) -> dict[str, Path]:
    """Two seeds of each baseline and an exact rerun of GRU seed 0, shared by the tests."""
    root = tmp_path_factory.mktemp("runs")
    gru = write_config(root / "gru.yaml", manifest, "gru")
    transformer = write_config(root / "transformer.yaml", manifest, "transformer")
    out = root / "runs"
    gru_dirs = run_seeds(gru, [0, 1], out=out, profile="cpu", profile_steps=2)
    tf_dirs = run_seeds(transformer, [0, 1], out=out, profile="cpu", profile_steps=2)
    again = rerun(gru_dirs[0], tolerance=0.0, profile_steps=2)
    return {
        "out": out,
        "gru0": gru_dirs[0],
        "gru1": gru_dirs[1],
        "tf0": tf_dirs[0],
        "tf1": tf_dirs[1],
        "gru0_rerun": again,
        "gru_config": gru,
    }


def read_json(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


# --------------------------------------------------------------------------------------------------
# Tests named by the chunk
# --------------------------------------------------------------------------------------------------


def test_p0_8_rerun_exact_on_cpu(runs: dict[str, Path], tmp_path: Path) -> None:
    """``hsn rerun`` with tolerance 0 reproduces every final metric and the model hash on cpu."""
    original, again = runs["gru0"], runs["gru0_rerun"]
    assert again == runs["out"] / "gru_tiny-cpu-s0-rerun"
    verdict = read_json(again / RERUN_FILE)
    assert verdict["passed"] is True and verdict["exact"] is True
    assert verdict["max_abs_diff"] == 0.0 and verdict["model_hash_equal"] is True
    assert verdict["original_run_id"] == "gru_tiny-cpu-s0"
    assert set(verdict["diffs"]) >= {"all/loss", "all/accuracy", "algorithmic/in_range/accuracy"}
    assert all(value == 0.0 for value in verdict["diffs"].values())
    a, b = read_json(original / METRICS_FILE), read_json(again / METRICS_FILE)
    assert a["final"] == b["final"] and a["model_hash"] == b["model_hash"]
    assert a["config_hash"] == b["config_hash"]  # byte-identical resolved configuration
    assert b["is_rerun"] is True and b["rerun_of"] == a["run_id"] and a["is_rerun"] is False
    assert a["evals"] == b["evals"]
    # the per-step training losses agree bit for bit as well
    steps_a = read_json(original / "history.json")["steps"]
    steps_b = read_json(again / "history.json")["steps"]
    assert [s["loss"] for s in steps_a] == [s["loss"] for s in steps_b]
    # the store carries the verdict
    table = ResultStore(runs["out"] / STORE_FILE).query(metric="rerun_max_abs_diff")
    assert table.num_rows == 1 and table.column("text").to_pylist() == ["PASS"]
    # a run whose recorded metrics were edited no longer reproduces: exit code 1 through the CLI
    copy = tmp_path / "edited" / original.name
    shutil.copytree(original, copy)
    edited = read_json(copy / METRICS_FILE)
    edited["final"]["accuracy"] += 0.1
    (copy / METRICS_FILE).write_text(json.dumps(edited), encoding="utf-8")
    assert main(["rerun", str(copy), "--tolerance", "0", "--profile-steps", "1"]) == 1
    verdict = read_json(copy.parent / f"{original.name}-rerun" / RERUN_FILE)
    assert verdict["passed"] is False and verdict["max_abs_diff"] == pytest.approx(0.1)
    assert verdict["max_abs_diff_loss"] == 0.0 and verdict["metric"] == "accuracy"
    # a loss-only difference is reported but gated only at tolerance 0 (accuracy is the frozen
    # primary metric of the synthetic suites; the loss is in nats, another unit)
    loss_copy = tmp_path / "loss_edited" / original.name
    shutil.copytree(original, loss_copy)
    edited = read_json(loss_copy / METRICS_FILE)
    edited["final"]["loss"] += 0.05
    (loss_copy / METRICS_FILE).write_text(json.dumps(edited), encoding="utf-8")
    assert main(["rerun", str(loss_copy), "--tolerance", "0.01", "--profile-steps", "1"]) == 0
    verdict = read_json(loss_copy.parent / f"{original.name}-rerun" / RERUN_FILE)
    assert verdict["max_abs_diff"] == 0.0 and verdict["max_abs_diff_loss"] == pytest.approx(0.05)
    assert main(["rerun", str(loss_copy), "--tolerance", "0", "--out", str(tmp_path / "l0")]) == 1
    # a tolerance covers it; a rerun of a rerun and a repeated rerun are refused
    assert main(["rerun", str(copy), "--tolerance", "0.2", "--out", str(tmp_path / "tol")]) == 0
    with pytest.raises(ValueError, match="itself a rerun"):
        rerun(again)
    with pytest.raises(FileExistsError):
        rerun(original)
    with pytest.raises(ValueError, match="tolerance"):
        rerun(original, tolerance=-1.0, out=tmp_path / "neg")


def test_p0_8_store_roundtrip(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "store" / "results.jsonl")
    assert store.path == tmp_path / "store" / "results.jsonl" and len(store) == 0
    empty = store.query()
    assert isinstance(empty, pa.Table) and empty.num_rows == 0 and "run_id" in empty.column_names
    store.append("run-a", {"metric": "accuracy", "value": 0.5, "seed": 0, "text": None})
    store.append("run-a", {"metric": "loss", "value": 1.25, "seed": 0, "text": None})
    store.append("run-b", {"metric": "accuracy", "value": 0.75, "seed": 1, "text": "note"})
    size = store.path.stat().st_size
    table = store.query()
    assert table.num_rows == 3 and len(store) == 3
    assert set(table.column_names) >= {"run_id", "appended_at", "metric", "value", "seed", "text"}
    assert table.column("run_id").to_pylist() == ["run-a", "run-a", "run-b"]
    assert store.query(run_id="run-a").num_rows == 2
    picked = store.query(run_id="run-a", metric="loss")
    assert picked.num_rows == 1 and picked.column("value").to_pylist() == [1.25]
    assert store.query(seed=1).column("run_id").to_pylist() == ["run-b"]
    assert store.query(text=None).num_rows == 2
    assert store.query(metric="missing").num_rows == 0
    with pytest.raises(KeyError, match="no record"):
        store.query(nope=1)
    # append-only: a second handle sees everything; the file only ever grows
    other = ResultStore(store.path)
    other.append("run-c", {"metric": "accuracy", "value": 0.0, "seed": 2})
    assert store.query().num_rows == 4 and store.path.stat().st_size > size
    assert not hasattr(store, "delete") and not hasattr(store, "update")
    # records are flat scalars; nested values, reserved columns and bad ids raise before writing
    with pytest.raises(TypeError, match="scalar"):
        store.append("run-d", {"nested": {"a": 1}})
    with pytest.raises(ValueError, match="reserved"):
        store.append("run-d", {"run_id": "x"})
    with pytest.raises(ValueError, match="run_id"):
        store.append("", {"metric": "x"})
    with pytest.raises(ValueError):
        store.append("run-d", {"value": float("nan")})
    assert store.query().num_rows == 4


def test_p0_8_package_has_all_artifacts(runs: dict[str, Path], tmp_path: Path) -> None:
    run_dirs = [runs["gru0"], runs["gru1"], runs["tf0"], runs["tf1"], runs["gru0_rerun"]]
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps({"decision": "PASS", "statement": "test", "checks": {"rerun": True}}),
        encoding="utf-8",
    )
    out = evidence_package(
        "G0", run_dirs, tmp_path / "evidence" / "G0.zip", extra_files={"decision.json": decision}
    )
    assert out == tmp_path / "evidence" / "G0.zip" and out.is_file()
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        summary = zf.read("G0/SUMMARY.md").decode("utf-8")
        manifest = json.loads(zf.read("G0/MANIFEST.json"))
        # every required artefact is listed in the summary with at least one file that exists
        for key, title in REQUIRED_ARTIFACTS:
            assert title in summary, title
            files = manifest["artifacts"][key]
            assert files, key
            for name in files:
                assert f"G0/{name}" in names, name
        # the manifest hashes every file in the zip (but itself)
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(zf.read(f"G0/{name}")).hexdigest() == digest, name
        assert set(manifest["files"]) == {n[3:] for n in names if n != "G0/MANIFEST.json"}
        assert "G0/decision.json" in names and "**PASS**" in summary
        uncertainty = json.loads(zf.read("G0/results/uncertainty.json"))
        assert set(uncertainty["per_model"]) == {"gru", "transformer"}
        assert uncertainty["per_model"]["gru"]["n_units"] == 4  # 2 tasks x 2 seeds
        assert uncertainty["per_model"]["gru"]["ci"]["n_boot"] == 10_000
        paired = uncertainty["paired"]["transformer - gru"]
        assert paired["n_units"] == 4 and set(paired["difference_ci"]) >= {"low", "high"}
        per_task = json.loads(zf.read("G0/results/per_task_regressions.json"))
        assert per_task["reference"] is None
        assert per_task["reruns"]["gru_tiny-cpu-s0-rerun"]["max_abs_delta"] == 0.0
        assert set(per_task["per_task"]) == {"algorithmic/in_range", "kv/in_range"}
        cost = json.loads(zf.read("G0/cost.json"))
        assert cost["measured"]["n_runs"] == 5 and cost["forecast"]["rate_usd_per_hour"] == [1, 3]
        assert cost["measured"]["runs"]["gru_tiny-cpu-s0"]["serving_tokens_per_s"] > 0
        hashes = json.loads(zf.read("G0/hashes.json"))
        assert (
            hashes["models"]["gru_tiny-cpu-s0"]["model_hash"]
            == read_json(runs["gru0"] / METRICS_FILE)["model_hash"]
        )
        assert "step_00000006.pt" in hashes["models"]["gru_tiny-cpu-s0"]["checkpoints"]
        leakage = json.loads(zf.read("G0/data/leakage.json"))
        assert leakage["clean"] is True and "tiny_v1" in leakage["manifests"]
        frozen = json.loads(zf.read("G0/freeze/metrics_freeze.json"))
        assert frozen == json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        assert json.loads(zf.read("G0/freeze/schema.json")) == metrics_freeze_schema()
        profile = json.loads(zf.read("G0/profiles/gru_tiny-cpu-s0.train.json"))
        assert set(profile["segments"]) == {"startup", "shared_path", "backward", "optimizer"}
        assert profile["segments"]["shared_path"]["n"] == 2
    # the package refuses to overwrite, needs a rerun and needs every run file
    with pytest.raises(FileExistsError):
        evidence_package("G0", run_dirs, out)
    with pytest.raises(MissingArtifactError, match="rerun"):
        evidence_package("G0", [runs["gru0"], runs["tf0"]], tmp_path / "no_rerun.zip")
    broken = tmp_path / "broken" / runs["tf0"].name
    shutil.copytree(runs["tf0"], broken)
    (broken / "profile_serve.json").unlink()
    with pytest.raises(MissingArtifactError, match="profile_serve"):
        evidence_package("G0", [broken, runs["gru0_rerun"]], tmp_path / "broken.zip")
    with pytest.raises(ValueError, match="gate"):
        evidence_package("G9", run_dirs, tmp_path / "g9.zip")
    # the CLI and the script wrapper reach the same function
    assert (
        main(
            [
                "package",
                "--gate",
                "G0",
                "--runs",
                *map(str, run_dirs),
                "--out",
                str(tmp_path / "cli.zip"),
            ]
        )
        == 0
    )
    assert zipfile.ZipFile(tmp_path / "cli.zip").testzip() is None


def test_p0_8_freeze_tamper_detected(tmp_path: Path) -> None:
    frozen = load_metrics_freeze()  # the committed file: hash and schema both hold
    raw = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    assert raw["sha256_of_body"] == freeze_body_hash(raw) == frozen.sha256_of_body
    assert frozen.body() == {k: v for k, v in raw.items() if k != "sha256_of_body"}
    # the numbers the plan fixes (§8.1, §2.9, the gate statements) and the user's G1 designation
    assert frozen.g1_task_keys() == ["kv/in_range", "delayed/in_range"]
    assert all(task.threshold == 0.95 for task in frozen.g1_tasks)
    assert frozen.quality_tolerance.accuracy == 0.01
    assert frozen.quality_tolerance.perplexity_relative == 0.02
    t = frozen.thresholds
    assert t.G1.accuracy_min == 0.95 and t.G1.n_seeds == 5
    assert t.G2.improvement_min == 0.20 and t.G2.n_seeds == 5 and t.G2.ci_alpha == 0.05
    assert t.G2.other_regression_max is None and t.G2.optimisation_cycles_max == 2
    assert (t.G3.mean_forgetting_max, t.G3.task_loss_max, t.G3.new_task_gain_min) == (
        0.02,
        0.05,
        0.03,
    )
    assert t.G3.macro_gap_max == 0.01
    assert t.G4.composition_gain_min == 0.03 and t.G4.growth_max == 0.25 and t.G4.n_seeds == 3
    assert t.G5.restore_drill_max_s == 900 and t.G5.load_test_hours == 72
    assert t.G6.pilot_weeks == 4 and t.G6.unresolved_critical_security_issues_max == 0
    assert frozen.primary_metrics["state_update"] == "accuracy"
    assert len(frozen.git_rev) == 40
    schema = metrics_freeze_schema()
    assert set(schema["required"]) == set(raw)
    # an in-place edit of one threshold is detected by the embedded hash
    tampered = dict(raw)
    tampered["g1_tasks"] = [dict(raw["g1_tasks"][0], threshold=0.5)]
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(FreezeTamperedError, match="edited in place"):
        load_metrics_freeze(path)
    # so is a dropped hash
    missing = {k: v for k, v in raw.items() if k != "sha256_of_body"}
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(FreezeTamperedError, match="no sha256_of_body"):
        load_metrics_freeze(path)
    # a recomputed hash over a body that breaks the schema fails validation
    for bad in (
        {**missing, "thresholds": {**missing["thresholds"], "G7": {}}},
        {**missing, "bonus": 1},
        {**missing, "quality_tolerance": {"accuracy": 1.0, "perplexity_relative": 0.02}},
        {**missing, "g1_tasks": []},
        {**missing, "frozen_at": "2026-11-04T00:00:00"},
    ):
        bad["sha256_of_body"] = freeze_body_hash(bad)
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(FreezeSchemaError):
            load_metrics_freeze(path)
    with pytest.raises(FreezeError):
        load_metrics_freeze(tmp_path / "absent.json")
    # writing: never over an existing file; the written file loads back equal
    with pytest.raises(FreezeError, match="never rewritten"):
        write_metrics_freeze(missing, FREEZE_PATH)
    written = write_metrics_freeze(missing, tmp_path / "v2" / "metrics_freeze.json")
    assert written == load_metrics_freeze(tmp_path / "v2" / "metrics_freeze.json")
    with pytest.raises(FreezeSchemaError):
        write_metrics_freeze({**missing, "bonus": 1}, tmp_path / "bad.json")


# --------------------------------------------------------------------------------------------------
# Further tests
# --------------------------------------------------------------------------------------------------


def test_p0_8_run_artifacts_and_registry(runs: dict[str, Path], tmp_path: Path) -> None:
    run_dir = runs["gru0"]
    expected = {
        "config.resolved.yaml",
        "run_manifest.json",
        "history.json",
        "metrics.json",
        "profile_train.json",
        "profile_serve.json",
        "run.log",
        "step_00000006.pt",
    }
    assert {p.name for p in run_dir.iterdir()} == expected
    metrics = read_json(run_dir / METRICS_FILE)
    manifest = read_json(run_dir / "run_manifest.json")
    assert metrics["run_id"] == "gru_tiny-cpu-s0" == metrics["config_run_id"]
    assert metrics["config_hash"] == manifest["config_hash"]
    assert metrics["steps"] == 6 and metrics["seed"] == 0 and metrics["profile"] == "cpu"
    assert set(manifest["measured"]) >= {
        "param_count",
        "peak_bytes",
        "reserved_bytes",
        "tokens_per_s",
        "launches_per_step",
    }
    assert manifest["measured"]["launches_per_step"] == -1.0  # not counted on cpu (P0.6)
    assert metrics["measured"]["param_count"] == metrics["param_report"]["stored"]
    assert metrics["grad_norms_finite"] is True
    assert metrics["data"]["n_train_docs"] == 128 and metrics["data"]["variants"] == ["in_range"]
    assert set(metrics["final"]["per_group"]) == {"algorithmic/in_range", "kv/in_range"}
    assert metrics["power"] == {"requested": "none", "method": "none"}
    assert len(metrics["model_hash"]) == 64 and set(metrics["checkpoints"]) == {"step_00000006.pt"}
    resolved = load_config(run_dir / "config.resolved.yaml")
    assert resolved.seed == 0 and resolved.run_id == "gru_tiny-cpu-s0"
    assert resolved.data.variants == ["in_range"]
    record = RunRecord.load(run_dir)
    assert record.run_id == "gru_tiny-cpu-s0" and not record.is_rerun
    # the store holds the long-format rows of every run
    store = ResultStore(runs["out"] / STORE_FILE)
    table = store.query(run_id="gru_tiny-cpu-s0", split="dev", metric="accuracy")
    assert table.num_rows == 3  # all + two groups
    assert store.query(run_id="gru_tiny-cpu-s1").num_rows > 0
    # model hashes: the two seeds differ, a rerun matches, a hash is stable across calls
    second = read_json(runs["gru1"] / METRICS_FILE)
    assert second["model_hash"] != metrics["model_hash"]
    model = build_model(resolved)
    assert model_hash(model) == model_hash(model)
    # registry
    assert set(MODEL_REGISTRY) >= {"gru", "transformer"}
    with pytest.raises(KeyError, match="registered"):
        build_model(load_config(EXAMPLE, overrides=["model_name=hsn"]))
    with pytest.raises(ValueError, match="already registered"):
        register_model("gru", MODEL_REGISTRY["gru"])
    with pytest.raises(ValueError):
        register_model("not valid", MODEL_REGISTRY["gru"])
    # the data.variants key validates like every other config key
    assert load_config(EXAMPLE, overrides=["data.variants=[out_of_range]"]).data.variants == [
        "out_of_range"
    ]
    with pytest.raises(ValueError):
        load_config(EXAMPLE, overrides=["data.variants=[bogus]"])
    # seeds and run directories
    assert parse_seeds("0,1,2") == [0, 1, 2]
    for bad_seeds in ("", "0,,1", "1,1", "-1"):
        with pytest.raises(ValueError):
            parse_seeds(bad_seeds)
    with pytest.raises(FileExistsError, match="never overwritten"):
        run_seeds(runs["gru_config"], [0], out=runs["out"], profile="cpu")
    assert (
        main(
            [
                "run",
                "--config",
                str(runs["gru_config"]),
                "--seeds",
                "7",
                "--profile",
                "cpu",
                "--out",
                str(tmp_path / "cli_runs"),
                "--override",
                "train.steps=2",
                "--override",
                "train.eval_every=2",
                "--override",
                "train.ckpt_every=2",
                "--profile-steps",
                "1",
            ]
        )
        == 0
    )
    cli_metrics = read_json(tmp_path / "cli_runs" / "gru_tiny-cpu-s7" / METRICS_FILE)
    assert cli_metrics["seed"] == 7 and cli_metrics["steps"] == 2


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p0_8_gate_g0_smoke(tmp_path: Path) -> None:
    """The gate script end to end on tiny data: build, check, train, rerun, decide, package."""
    spec_path = tmp_path / "tiny.yaml"
    spec_path.write_text(
        yaml.safe_dump(json.loads(tiny_spec(("algorithmic",)).model_dump_json())), encoding="utf-8"
    )
    data_out = tmp_path / "data"
    manifest = data_out / "manifest.json"
    gru = write_config(tmp_path / "gru.yaml", manifest, "gru", steps=4)
    transformer = write_config(tmp_path / "transformer.yaml", manifest, "transformer", steps=4)
    for path in (gru, transformer):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg["data"]["tasks"] = ["algorithmic"]
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    gate = load_script("gate_g0")
    argv = [
        "--profile",
        "cpu",
        "--seeds",
        "0,1",
        "--out",
        str(tmp_path / "evidence"),
        "--configs",
        str(gru),
        str(transformer),
        "--data-spec",
        str(spec_path),
        "--data-out",
        str(data_out),
        "--no-golden",
        "--runs",
        str(tmp_path / "runs"),
        "--profile-steps",
        "1",
    ]
    assert gate.main(argv) == 0
    decision = read_json(tmp_path / "evidence" / "decision.json")
    assert decision["decision"] == "PASS" and all(decision["checks"].values())
    assert set(decision["checks"]) == {
        "data_protocol",
        "leakage_checks",
        "metrics_frozen",
        "baselines_exist",
        "profiler_exists",
        "rerun_reproduces",
    }
    assert decision["data"]["action"] == "built" and decision["leakage_clean"] is True
    assert len(decision["runs"]) == 4 and len(decision["reruns"]) == 2
    assert all(r["max_abs_diff"] == 0.0 and r["model_hash_equal"] for r in decision["reruns"])
    assert all(r["max_abs_diff_loss"] == 0.0 for r in decision["reruns"])
    assert decision["freeze"]["g1_tasks"] == ["kv/in_range", "delayed/in_range"]
    leakage = read_json(tmp_path / "evidence" / "leakage.json")
    assert leakage["clean"] and set(leakage["live"]["pairs"]) == {"algorithmic/in_range"}
    with zipfile.ZipFile(tmp_path / "evidence" / "G0.zip") as zf:
        assert "G0/decision.json" in zf.namelist() and "G0/leakage.json" in zf.namelist()
        assert "**PASS**" in zf.read("G0/SUMMARY.md").decode("utf-8")
    store = ResultStore(tmp_path / "runs" / STORE_FILE)
    assert store.query(run_id="G0", metric="decision").column("text").to_pylist() == ["PASS"]
    # a second invocation verifies the existing data instead of rebuilding, then refuses the
    # existing run directories (runs are never overwritten)
    with pytest.raises(FileExistsError):
        gate.main(argv)
    # a GPU profile needs an explicit tolerance
    with pytest.raises(SystemExit):
        gate.main(["--profile", "m_series", "--out", str(tmp_path / "x"), "--configs", str(gru)])
    # the package script wrapper
    package = load_script("evidence_package")
    run_dirs = [str(Path(d)) for d in decision["runs"]] + [
        str(tmp_path / "runs" / r["rerun"]) for r in decision["reruns"]
    ]
    assert (
        package.main(["--gate", "G0", "--runs", *run_dirs, "--out", str(tmp_path / "again.zip")])
        == 0
    )
