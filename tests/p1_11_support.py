"""Shared fixtures of the P1.11 tests (the sweep harness): tiny kv + delayed documents, tiny HSN and
baseline configurations on ``profile:cpu``, tiny on-disk data, and the isolation that keeps every
test away from the real experiment database (``~/hsn-experiments``) and from the GPU.

Not a test module (no ``test_`` prefix); the P1.11 test files import it from ``tests/``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from hsn.config import RunConfig, load_config, load_hardware_profile
from hsn.data.splits import build_splits
from hsn.data.synthetic import DatasetSpec, Example, TaskSpec, generate, load_dataset_spec
from hsn.device import DeviceContext, resolve, set_deterministic

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
EXPERIMENT = REPO / "configs" / "experiments" / "g1_ref_a1.yaml"
GRU_2M = REPO / "configs" / "baselines" / "gru_2m.yaml"
TRANSFORMER_2M = REPO / "configs" / "baselines" / "transformer_2m.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "p1_11"
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
TINY_HSN_OVERRIDES = [f"model.{k}={v}" for k, v in TINY_MODEL.items()]
TINY_GRU_OVERRIDES = ["baseline.gru.d_model=16", "baseline.gru.n_layers=1"]
TINY_TRANSFORMER_OVERRIDES = [
    "baseline.transformer.d_model=16",
    "baseline.transformer.n_layers=1",
    "baseline.transformer.n_heads=2",
    "baseline.transformer.ffn_hidden=32",
]


def tiny_task(family: str, *, n_train: int = 64, n_dev: int = 16) -> TaskSpec:
    """The committed ``<family>/in_range`` task with documents of at most ``UNROLL`` symbols."""
    task = SPEC.task(family, "in_range")  # type: ignore[arg-type]
    return TaskSpec.model_validate(
        {
            **task.model_dump(),
            "n_train": n_train,
            "n_dev": n_dev,
            "n_test": 16,
            "length_range": [16, UNROLL],
            "dep_len_range": [4, 20],
            "unroll": UNROLL,
        }
    )


def tiny_spec(families: tuple[str, ...] = ("kv", "delayed")) -> DatasetSpec:
    return DatasetSpec(name="tiny_p1_11", unroll=UNROLL, tasks=[tiny_task(f) for f in families])


def docs(family: str, split: str, n: int) -> list[Example]:
    task = tiny_task(family, n_train=max(n, 16), n_dev=max(n, 16))
    return [generate(task, split, i) for i in range(n)]  # type: ignore[arg-type]


def build_tiny_data(out: Path) -> Path:
    """Write the tiny kv + delayed dataset to ``out`` and return its manifest path."""
    build_splits(tiny_spec(), out)
    return out / "manifest.json"


def train_overrides(*, steps: int = 6, batch: int = 4, eval_every: int = 3) -> list[str]:
    return [
        "profile=cpu",
        f"train.unroll={UNROLL}",
        f"train.batch={batch}",
        f"train.micro_batch={batch}",
        f"train.steps={steps}",
        f"train.eval_every={eval_every}",
        f"train.ckpt_every={steps}",
        "train.stats_every=2",
        "train.warmup_steps=1",
        "train.deterministic=true",
    ]


def tiny_hsn_config(*overrides: str, steps: int = 6) -> RunConfig:
    return load_config(
        REF_A1, overrides=[*TINY_HSN_OVERRIDES, *train_overrides(steps=steps), *overrides]
    )


def tiny_gru_config(*overrides: str, steps: int = 6) -> RunConfig:
    return load_config(
        GRU_2M, overrides=[*TINY_GRU_OVERRIDES, *train_overrides(steps=steps), *overrides]
    )


def tiny_transformer_config(*overrides: str, steps: int = 6) -> RunConfig:
    return load_config(
        TRANSFORMER_2M,
        overrides=[*TINY_TRANSFORMER_OVERRIDES, *train_overrides(steps=steps), *overrides],
    )


def cpu_ctx() -> DeviceContext:
    return resolve(load_hardware_profile("cpu"))


def write_yaml(path: Path, payload: Any) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def restore_determinism() -> Iterator[None]:
    """The tiny configurations set ``train.deterministic``, which the trainer applies globally."""
    before = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        set_deterministic(before)


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the harness at a database under ``tmp_path`` (never ``~/hsn-experiments``), allow a
    temporary path for it, hide CUDA from every child process and keep the real machine probes
    (nvidia-smi, top) out of the tests. Yields the database path."""
    db = tmp_path / "experiments.sqlite"
    monkeypatch.setenv("HSN_EXPERIMENTS_DB", str(db))
    monkeypatch.setenv("HSN_EXPERIMENTS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HSN_EXPERIMENTS_ALLOW_TMP", "1")
    monkeypatch.setenv("HSN_EXPERIMENTS_FAKE_MACHINE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    yield db


GOLDEN_MANIFEST = REPO / "configs" / "data" / "synthetic_v1.manifest.json"


def fake_repo(root: Path, *, manifest: bytes | None = None) -> Path:
    """A repository view for spec expansion: ``configs/`` linked to the real one and
    ``data/synthetic_v1/manifest.json`` a copy of the committed golden manifest (the built data's
    manifest is byte-identical to it), so arm ids can be computed where ``data/`` is not built."""
    repo = root / "repo"
    (repo / "data" / "synthetic_v1").mkdir(parents=True, exist_ok=True)
    link = repo / "configs"
    if not link.exists():
        link.symlink_to(REPO / "configs", target_is_directory=True)
    target = repo / "data" / "synthetic_v1" / "manifest.json"
    target.write_bytes(GOLDEN_MANIFEST.read_bytes() if manifest is None else manifest)
    return repo


def dry_spec(root: Path, *, n: int = 4, name: str = "dry", **protocol: Any) -> Path:
    """A spec of ``n`` dry-run arms (no training, no data) of the tiny HSN on ``profile:cpu``."""
    raw = {
        "name": name,
        "base": {
            "config": str(REF_A1),
            "overrides": [*TINY_HSN_OVERRIDES, *train_overrides(steps=4, eval_every=2)],
        },
        "protocol": {"name": "dry_run", **protocol},
        "seeds": list(range(n)),
        "arms": [{"name": "a", "family": "hsn"}],
    }
    return write_yaml(root / f"{name}.yaml", raw)


def enqueue_spec(db: Any, path: Path, **kwargs: Any) -> Any:
    """Expand ``path`` and enqueue its arms into ``db`` (what the CLI's ``enqueue`` does)."""
    from hsn.experiments.db import enqueue_arms
    from hsn.experiments.rules import parse_rules
    from hsn.experiments.spec import digest, expand, load_spec

    spec, text = load_spec(path)
    arms = expand(spec)
    rules = parse_rules(spec.rules)
    return enqueue_arms(
        db,
        name=spec.name,
        spec_hash=digest(spec.model_dump(mode="json")),
        spec_text=text,
        spec_path=str(path),
        rules_json=rules.to_json(),
        arms=arms,
        **kwargs,
    )


def db_path() -> Path:
    """The test database the ``isolated`` fixture set up."""
    import os

    return Path(os.environ["HSN_EXPERIMENTS_DB"])


def disk_overrides(manifest: Path, *, steps: int = 4, eval_every: int = 2) -> list[str]:
    """Overrides that train on the tiny on-disk data of :func:`build_tiny_data`."""
    return [
        *train_overrides(steps=steps, eval_every=eval_every),
        f"data.manifest={manifest}",
        "data.dataset=tiny_p1_11",
        "data.tasks=[kv,delayed]",
        "data.variants=[in_range]",
    ]


@pytest.fixture(scope="module")
def tiny_manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_data(tmp_path_factory.mktemp("tiny_data"))


# The fields an HSN reads-state record carries for every diagnostic and for the trained model (the
# audit's JSONs; the compiled bridge of the validation sweep is the control the first real sweep
# reads its compiled arms against)
STATE_KEYS = [
    f"{stat}/{where}"
    for where in ("answer", "valid")
    for stat in (
        "diversity",
        "diversity_live",
        "live_cells",
        "rank_pr",
        "rank_pr_p90",
        "sat99",
        "abs_h",
        "cell_spread",
    )
]
GATE_KEYS = ["gate/p10", "gate/p50", "gate/p90"]
HSN_GROUPS = ["byte_emb", "ctx_emb", "encoder", "predictor", "shared", "cell", "router", "decoder"]


def check_hsn_reads_state_record(
    record: dict[str, Any], *, eval_steps: list[int], diag_steps: list[int]
) -> None:
    """Every field the control role needs, at every evaluation and every diagnostic."""
    assert [e["step"] for e in record["evals"]] == eval_steps
    for e in record["evals"]:
        assert {"accuracy", "loss"} <= set(e) and e["per_group"]
        for g, v in e["per_group"].items():
            assert {"accuracy", "loss"} <= set(v)
            assert e["dev_answer_accuracy"][g] == v["accuracy"]
            assert e["dev_answer_loss"][g] == v["loss"]
    assert [d["step"] for d in record["diagnostics"]] == diag_steps
    for d in record["diagnostics"]:
        for group in HSN_GROUPS:
            assert f"grad_share/{group}" in d and f"update_ratio/{group}" in d, (group, sorted(d))
        missing = [k for k in [*STATE_KEYS, *GATE_KEYS] if k not in d]
        assert not missing, missing
    trained = record["trained_state"]
    missing = [
        k
        for k in [*STATE_KEYS, *GATE_KEYS, "edge_logit/absmax", "edge_logit/spread"]
        if k not in trained
    ]
    assert not missing, missing
    probe = record["probe"]
    for key in (
        "n_probed",
        "first_query_accuracy",
        "all",
        "correct_before_edit",
        "verdict",
        "correct_follow_frac",
    ):
        assert key in probe, key
    assert record["final"]["dev_loss"] == record["evals"][-1]["loss"]
