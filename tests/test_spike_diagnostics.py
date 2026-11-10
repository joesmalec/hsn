"""Backward diagnostics preserve their source state and identify controlled gradient sources."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.experiments.protocols import ArmRun, failure_info, run_protocol
from hsn.experiments.rules import RuleSet
from hsn.experiments.spec import SpikeDiagnosticParams
from hsn.experiments.spike import Case, compare, run_diagnostic, sha256
from hsn.experiments.worker import turn_remaining
from hsn.train.divergence import dump_paths
from p1_15_support import GUARD, datasets, hsn_config, hsn_trainer, plant


@pytest.fixture
def saved_case(tmp_path: Path) -> tuple[Any, Any, Path]:
    cfg = hsn_config(
        4,
        "train.divergence_healthy_every=1",
        "model.routing=topk",
        "train.gate_mode=ste",
        "model.key_address=true",
        "model.selected_readout=true",
    )
    data, dev = datasets()
    trainer = hsn_trainer(cfg, tmp_path / "source")
    trainer.fit(data, dev, 2)
    return cfg, data, tmp_path / "source" / "healthy_00000002.pt"


def test_backward_only_is_repeatable_and_instrumentation_is_inert(
    saved_case: Any, tmp_path: Path
) -> None:
    cfg, data, path = saved_case
    before = sha256(path)
    case = Case(path, cfg, train_ds=data)
    a = case.run(tmp_path / "a", compiled=False, deterministic=True)
    b = case.run(tmp_path / "b", compiled=False, deterministic=True, trace=True)
    assert compare(a, b)["passed"]
    assert a.report["optimizer_updates"] == 0 and a.trainer.step == 1
    assert a.report["raw_grad_norm"] == pytest.approx(case.info["grad_norm"], rel=1e-6)
    assert any(r.get("local_input_gradient") for r in b.report["trace"])
    assert sha256(path) == before
    assert not list((tmp_path / "a").glob("*.pt"))


@pytest.mark.parametrize("stage", ["isolate", "update", "trace"])
def test_diagnostic_protocol_preserves_source_and_full_batch(
    saved_case: Any, tmp_path: Path, stage: Any
) -> None:
    cfg, data, path = saved_case
    checksum = sha256(path)
    case = Case(path, cfg, train_ds=data)
    params = SpikeDiagnosticParams(
        name="spike_diagnostic",
        parent_exp=1,
        dump=str(path),
        dump_sha256=checksum,
        stage=stage,
        target_doc=case.docs[0],
        monitor_docs=4,
    )
    directory = tmp_path / stage
    directory.mkdir()
    arm = ArmRun(cfg, "hsn", "pdg", {}, params, directory, RuleSet(), {})
    record = run_diagnostic(arm, params, train_ds=data)["diagnostic"]
    assert record["completed"] and record["production_updates"] == 0
    assert sha256(path) == checksum
    if stage == "isolate":
        assert record["document_partition_relative_error"] < 1e-5
        assert record["term_partition_relative_error"] < 1e-5
    if stage == "update":
        update = record["shadow_updates"]["eager"]
        assert update["optimizer_updates"] == 1 and update["monitor_fingerprint"]
        assert any(v["norm"] > 0 for v in update["parameter_updates"].values())


def test_bad_source_and_checksum_fail_before_backward(saved_case: Any, tmp_path: Path) -> None:
    cfg, data, path = saved_case
    wrong = cfg.model_copy(update={"seed": cfg.seed + 1})
    with pytest.raises(ValueError, match="configuration differs"):
        Case(path, wrong, train_ds=data)
    params = SpikeDiagnosticParams(
        name="spike_diagnostic",
        parent_exp=1,
        dump=str(path),
        dump_sha256="0" * 64,
        stage="reproduce",
    )
    arm = ArmRun(cfg, "hsn", "pdg", {}, params, tmp_path, RuleSet(), {})
    with pytest.raises(ValueError, match="checksum"):
        run_protocol(arm)
    assert arm.record["failure"]["kind"] == "error"


def test_retained_reference_is_reported_at_its_own_step(saved_case: Any, tmp_path: Path) -> None:
    cfg, data, path = saved_case
    reference = path.parent / "step_00000002.pt"
    params = SpikeDiagnosticParams(
        name="spike_diagnostic",
        parent_exp=1,
        dump=str(path),
        dump_sha256=sha256(path),
        stage="reproduce",
        reference_checkpoint=str(reference),
        reference_sha256=sha256(reference),
    )
    directory = tmp_path / "reference"
    directory.mkdir()
    arm = ArmRun(cfg, "hsn", "pdg", {}, params, directory, RuleSet(), {})
    result = run_diagnostic(arm, params, train_ds=data)["diagnostic"]
    assert result["reference"]["step"] == 3 and result["step"] == 2
    assert result["reference"]["report"]["optimizer_updates"] == 0
    assert result["reference"]["mode"] == "next_window"
    assert result["reference"]["window_sha256"] != result["window_sha256"]


def test_earlier_weights_use_exact_source_window_and_do_not_modify_source(
    saved_case: Any, tmp_path: Path
) -> None:
    cfg, data, path = saved_case
    reference = path.parent / "healthy_00000001.pt"
    checksum = sha256(path)
    params = SpikeDiagnosticParams(
        name="spike_diagnostic",
        parent_exp=1,
        dump=str(path),
        dump_sha256=checksum,
        stage="reproduce",
        reference_checkpoint=str(reference),
        reference_sha256=sha256(reference),
        reference_mode="same_window_weights",
    )
    directory = tmp_path / "matched"
    directory.mkdir()
    arm = ArmRun(cfg, "hsn", "pdg", {}, params, directory, RuleSet(), {})
    result = run_diagnostic(arm, params, train_ds=data)["diagnostic"]
    report = result["reference"]["by_mode"]["eager"]["report"]
    assert result["reference"]["same_windows"]
    assert report["window_sha256"] == result["window_sha256"]
    assert report["weights_step"] == 0
    assert report["stream_checkpoint"] == str(path)
    assert report["weights_checkpoint"] == str(reference)
    assert report["optimizer_updates"] == 0 and result["step"] == 2
    assert all(check["passed"] for check in result["checks"])
    assert sha256(path) == checksum
    case = Case(path, cfg, train_ds=data)
    baseline = case.run(tmp_path / "baseline", compiled=False, deterministic=True)
    identity = case.run(
        tmp_path / "identity", compiled=False, deterministic=True, weight_override=path
    )
    assert compare(baseline, identity)["passed"]


def test_selected_reference_weights_preserve_other_parameters(
    saved_case: Any, tmp_path: Path
) -> None:
    cfg, data, source = saved_case
    reference = source.parent / "healthy_00000001.pt"
    case = Case(source, cfg, train_ds=data)
    base = case.run(tmp_path / "selected_base", compiled=False, deterministic=True)
    changed = case.run(
        tmp_path / "selected",
        compiled=False,
        deterministic=True,
        weight_override=reference,
        weight_selectors=("cell.*",),
    )
    early = torch.load(reference, weights_only=False)["model"]
    late = dict(base.trainer.model.named_parameters())
    assert changed.report["window_sha256"] == base.report["window_sha256"]
    assert not changed.report["reference_buffers_loaded"]
    selected = set(changed.report["selected_parameters"])
    assert selected and all(name.startswith("cell.") for name in selected)
    for name, value in changed.trainer.model.named_parameters():
        assert torch.equal(value.cpu(), early[name] if name in selected else late[name].cpu())
    identity = case.run(
        tmp_path / "selected_identity",
        compiled=False,
        deterministic=True,
        weight_override=source,
        weight_selectors=("cell.*",),
    )
    assert compare(base, identity)["passed"]
    with pytest.raises(ValueError, match="matches no parameters"):
        case.run(
            tmp_path / "bad_selector",
            compiled=False,
            deterministic=True,
            weight_override=reference,
            weight_selectors=("missing.*",),
        )


def test_reference_weights_reject_other_configuration(saved_case: Any, tmp_path: Path) -> None:
    cfg, data, path = saved_case
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["extra"]["config_hash"] = "another-configuration"
    wrong = tmp_path / "wrong.pt"
    torch.save(payload, wrong)
    with pytest.raises(ValueError, match="configuration hash"):
        Case(path, cfg, train_ds=data).run(
            tmp_path / "wrong", compiled=False, deterministic=True, weight_override=wrong
        )


def test_multiple_dumps_have_bounded_healthy_references_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = hsn_config(
        8,
        *GUARD,
        "train.divergence_max_dumps=2",
        "train.divergence_healthy_every=1",
        "train.divergence_healthy_keep=2",
    )
    trainer = hsn_trainer(cfg, tmp_path)
    plant(monkeypatch, trainer, {4: "inf_gradient", 6: "inf_gradient", 8: "inf_gradient"})
    trainer.fit(*datasets(), 8)
    assert len(dump_paths(tmp_path)) == len(trainer.divergences) == 2
    assert len(list(tmp_path.glob("healthy_*.pt"))) == 2
    assert len(list(tmp_path.glob("reference_*.pt"))) == 4
    assert trainer.divergence == trainer.divergences[0]
    for source in tmp_path.glob("healthy_*.json"):
        report = json.loads(source.read_text())
        assert report["optimizer_update_applied"] and report["parameter_updates"]
        assert report["clipped_grad_norm"] <= cfg.train.grad_clip + 1e-5
    resumed = hsn_trainer(cfg, tmp_path / "resumed")
    resumed.resume(tmp_path / "step_00000008.pt")
    assert len(resumed.divergences) == 2


def test_capture_does_not_change_learning(tmp_path: Path) -> None:
    cfg = hsn_config(3, "train.divergence_healthy_every=1")
    on = hsn_trainer(cfg, tmp_path / "on")
    off = hsn_trainer(hsn_config(3, "train.divergence_dump=false"), tmp_path / "off")
    a = on.fit(*datasets(), 3)
    b = off.fit(*datasets(), 3)
    assert [(r.loss, r.grad_norm) for r in a.steps] == [(r.loss, r.grad_norm) for r in b.steps]
    assert all(
        torch.equal(p, q)
        for p, q in zip(on.model.parameters(), off.model.parameters(), strict=True)
    )


def test_turn_claim_must_be_explicit_and_unexpired(tmp_path: Path) -> None:
    path = tmp_path / "turn.json"
    assert turn_remaining(path, "A5", 100) == 0
    base = {
        "owner": "A5",
        "status": "active",
        "start": "1970-01-01T00:00:00Z",
        "end": "1970-01-01T00:10:00Z",
        "released_at": None,
    }
    for change in ({"owner": "P2.10"}, {"status": "released"}, {"released_at": "now"}):
        path.write_text(json.dumps({**base, **change}))
        assert turn_remaining(path, "A5", 100) == 0
    path.write_text(json.dumps(base))
    assert turn_remaining(path, "A5", 100) == 500
    assert turn_remaining(path, "A5", 600) == 0
    assert failure_info(TimeoutError("budget"), stage="replay", last_step=0)["kind"] == "timeout"


@pytest.mark.cuda
def test_compiled_a5_backward_stops_before_update_and_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from cards import CUDA_PROFILE
    from hsn.train import compile as compile_mod

    monkeypatch.setattr(compile_mod, "MAX_BLOCK", 4)

    cfg = hsn_config(
        4,
        f"profile={CUDA_PROFILE}",
        "hardware.compile_mode=reduce-overhead",
        "hardware.autocast_dtype=bf16",
        "train.precision=bf16",
        "model.routing=topk",
        "train.gate_mode=ste",
        "model.key_address=true",
        "model.selected_readout=true",
        "train.divergence_healthy_every=1",
    )
    data, dev = datasets()
    trainer = hsn_trainer(cfg, tmp_path / "source")
    trainer.fit(data, dev, 2)
    path = tmp_path / "source" / "healthy_00000002.pt"
    case = Case(path, cfg, train_ds=data)
    a = case.run(tmp_path / "a", compiled=True, deterministic=True)
    b = case.run(tmp_path / "b", compiled=True, deterministic=True)
    assert compare(a, b)["passed"] and a.trainer.step == 1
    assert a.report["optimizer_updates"] == 0
    params = SpikeDiagnosticParams(
        name="spike_diagnostic",
        parent_exp=1,
        dump=str(path),
        dump_sha256=sha256(path),
        stage="isolate",
        modes=("compiled",),
        target_doc=case.docs[0],
        reference_checkpoint=str(path.parent / "healthy_00000001.pt"),
        reference_sha256=sha256(path.parent / "healthy_00000001.pt"),
        reference_mode="same_window_weights",
    )
    directory = tmp_path / "compiled_isolation"
    directory.mkdir()
    arm = ArmRun(cfg, "hsn", "pdg", {}, params, directory, RuleSet(), {})
    result = run_diagnostic(arm, params, train_ds=data)["diagnostic"]
    assert result["completed"] and all(p["compiled"] for p in result["passes"])
    partition = result["attribution_by_mode"]["compiled"]
    assert partition["term_partition_relative_error"] <= 0.02
    assert partition["document_partition_relative_error"] <= 0.02
    assert result["reference"]["same_windows"]
    assert all(p["window_sha256"] == result["window_sha256"] for p in result["passes"])
