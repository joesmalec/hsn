"""Burst capture must reconstruct the recorded trajectory without changing its source."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.data.dataset import ByteDataset
from hsn.experiments.capture import _equal, _training_state
from hsn.experiments.protocols import ArmRun
from hsn.experiments.rules import RuleSet
from hsn.experiments.spec import CheckpointCaptureParams, SpikeDiagnosticParams, arm_identity
from hsn.experiments.spike import Case, run_diagnostic, sha256
from p1_15_support import datasets, hsn_config, hsn_trainer


def source_case(
    tmp_path: Path, optimizer: str = "adamw", cuda: bool = False, dropout: float = 0.0
) -> tuple[Any, ...]:
    overrides = [
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        "train.gate_mode=ste",
        f"train.optimizer={optimizer}",
        f"model.decoder_shortcut_dropout={dropout}",
        'train.update_limit.groups={chosen: {selectors: ["cell.*"], max_ratio: 0.001}}',
    ]
    if cuda:
        overrides += [
            "profile=rtx3060",
            "train.precision=bf16",
            "train.deterministic=false",
            "hardware.compile_mode=reduce-overhead",
        ]
    cfg = hsn_config(6, *overrides)
    data, dev = datasets()
    trainer = hsn_trainer(cfg, tmp_path / "source")
    trainer.extra_ckpt_every = 2
    history = trainer.fit(data, dev, 6)
    trace = tmp_path / "source" / "result.json"
    trace.write_text(
        json.dumps(
            {
                "arm": {"run_id": cfg.run_id},
                "train": [
                    {"step": s.step, "loss": s.loss, "grad_norm": s.grad_norm}
                    for s in history.steps
                ],
            }
        )
    )
    source = tmp_path / "source" / "step_00000002.pt"
    params = SpikeDiagnosticParams(
        name="spike_diagnostic",
        parent_exp=1,
        dump=str(source),
        dump_sha256=sha256(source),
        stage="capture",
        modes=("compiled" if cuda else "eager",),
        deterministic=not cuda,
        capture=CheckpointCaptureParams(steps=(3, 5), trace=str(trace), trace_sha256=sha256(trace)),
    )
    directory = tmp_path / "capture"
    directory.mkdir()
    diagnostic_cfg = cfg.model_copy(update={"run_id": "separate-capture-arm"})
    arm = ArmRun(diagnostic_cfg, "hsn", "pdg", {}, params, directory, RuleSet(), {})
    return cfg, data, source, params, arm


@pytest.mark.parametrize("optimizer,dropout", [("adamw", 0.5), ("lazy_adamw", 0.0)])
def test_capture_matches_uninterrupted_state_and_preserves_source(
    tmp_path: Path, optimizer: str, dropout: float
) -> None:
    cfg, data, source, params, arm = source_case(tmp_path, optimizer, dropout=dropout)
    checksum = sha256(source)
    record = run_diagnostic(arm, params, train_ds=data)["diagnostic"]
    assert record["completed"] and record["prefix_matched"]
    assert record["source_run_id"] == cfg.run_id != arm.cfg.run_id
    assert record["production_updates"] == 0
    assert [r["disposable_updates"] for r in record["replay_runs"]] == [2, 2]
    assert all(c["passed"] for c in record["checks"])
    captured = torch.load(arm.run_dir / "divergence_00000005.pt", weights_only=False)
    original = torch.load(source.parent / "step_00000004.pt", weights_only=False)
    assert _equal(_training_state(captured), _training_state(original))
    assert sha256(source) == checksum == record["reference_sha256"]
    assert not list(arm.run_dir.glob("step_00000005.pt"))
    case = Case(arm.run_dir / "divergence_00000005.pt", cfg, train_ds=data)
    assert case.step == 5 and case.info["kind"] == "capture"
    assert captured["extra"].get("divergences", []) == []
    assert all(item["gradient_stage"] == "after_clip" for item in record["captures"])


def test_capture_fails_at_first_trace_disagreement(tmp_path: Path) -> None:
    _, data, source, params, arm = source_case(tmp_path)
    assert params.capture is not None
    path = Path(params.capture.trace)
    trace = json.loads(path.read_text())
    trace["train"][3]["loss"] += 1
    path.write_text(json.dumps(trace))
    params = params.model_copy(
        update={"capture": params.capture.model_copy(update={"trace_sha256": sha256(path)})}
    )
    checksum = sha256(source)
    with pytest.raises(ValueError, match="differs from source at step 4"):
        run_diagnostic(arm, params, train_ds=data)
    prefix = arm.record["diagnostic"]["replay_runs"][0]["prefix"]
    assert [row["passed"] for row in prefix] == [True, False]
    assert not (arm.run_dir / "divergence_00000005.pt").exists()
    assert sha256(source) == checksum


def test_capture_rejects_changed_trace_or_dataset(tmp_path: Path) -> None:
    _, data, _, params, arm = source_case(tmp_path)
    assert params.capture is not None
    changed = params.model_copy(
        update={"capture": params.capture.model_copy(update={"trace_sha256": "0" * 64})}
    )
    with pytest.raises(ValueError, match="trace checksum"):
        run_diagnostic(arm, changed, train_ds=data)
    reordered = ByteDataset(list(reversed(data.examples)))
    with pytest.raises(ValueError, match="fingerprint"):
        run_diagnostic(arm, params, train_ds=reordered)


def test_capture_rejects_source_output_alias(tmp_path: Path) -> None:
    _, data, source, params, arm = source_case(tmp_path)
    arm.run_dir = source.parent
    checksum = sha256(source)
    with pytest.raises(ValueError, match="separate"):
        run_diagnostic(arm, params, train_ds=data)
    assert sha256(source) == checksum


def test_capture_still_rejects_changed_training_configuration(tmp_path: Path) -> None:
    cfg, data, source, params, arm = source_case(tmp_path)
    arm.cfg = arm.cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"lr": cfg.train.lr * 2})}
    )
    checksum = sha256(source)
    with pytest.raises(ValueError, match="was written for configuration"):
        run_diagnostic(arm, params, train_ds=data)
    assert sha256(source) == checksum


def test_capture_schema_and_legacy_identity() -> None:
    fields = dict(
        name="spike_diagnostic",
        parent_exp=1,
        dump="source.pt",
        dump_sha256="0" * 64,
        stage="reproduce",
    )
    params = SpikeDiagnosticParams.model_validate(fields)
    cfg = hsn_config(6)
    raw = params.model_dump(mode="json")
    kwargs = dict(family="hsn", variant="pdg", variant_params={}, early_stop=[], data="test")
    identity = arm_identity(cfg, protocol=raw, **kwargs)
    raw.pop("capture")
    raw.pop("reference_selectors")
    assert arm_identity(cfg, protocol=raw, **kwargs) == identity
    with pytest.raises(ValueError, match="required only"):
        SpikeDiagnosticParams.model_validate({**fields, "stage": "capture"})
    for steps in ((), (0,), (3, 3), (4, 3)):
        with pytest.raises(ValueError, match="positive, unique and increasing"):
            CheckpointCaptureParams(steps=steps, trace="trace.json", trace_sha256="0" * 64)


@pytest.mark.cuda
def test_compiled_capture_reconstructs_original_prefix(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    _, data, source, params, arm = source_case(tmp_path, "lazy_adamw", cuda=True)
    checksum = sha256(source)
    record = run_diagnostic(arm, params, train_ds=data)["diagnostic"]
    assert record["completed"] and all(c["passed"] for c in record["checks"])
    assert sha256(source) == checksum
    torch._dynamo.reset()
