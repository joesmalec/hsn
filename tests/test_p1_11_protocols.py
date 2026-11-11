"""P1.11 tests: the protocols on tiny CPU runs (HSN and GRU): early-stop rules stop training at
their step and are recorded, the probe and the trained state still run after a stop (under
``at_stop``, never feeding a verdict), and the reads-state and overfit records carry the audit's
fields plus the per-phase wall times."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hsn.config import load_config
from hsn.experiments.protocols import ArmRun, run_protocol
from hsn.experiments.rules import evaluate_verdicts, parse_rules
from hsn.experiments.spec import protocol_params
from p1_11_support import (
    GRU_2M,
    REF_A1,
    TINY_GRU_OVERRIDES,
    TINY_HSN_OVERRIDES,
    check_hsn_reads_state_record,
    disk_overrides,
    restore_determinism,  # noqa: F401  (fixture)
    tiny_manifest,  # noqa: F401  (fixture)
)

pytestmark = pytest.mark.usefixtures("restore_determinism")


def _arm(
    family: str,
    manifest: Path,
    run_dir: Path,
    *,
    rules: dict[str, Any] | None = None,
    **protocol: Any,
) -> ArmRun:
    if family == "hsn":
        cfg = load_config(REF_A1, [*TINY_HSN_OVERRIDES, *disk_overrides(manifest, steps=6)])
    else:
        cfg = load_config(GRU_2M, [*TINY_GRU_OVERRIDES, *disk_overrides(manifest, steps=6)])
    params = protocol_params(
        {"name": "reads_state", "diag_every": 2, "diag_docs": 8, "probe_batch": 8, **protocol}
    )
    progress: list[dict[str, Any]] = []
    return ArmRun(
        cfg=cfg,
        family=family,
        variant="pdg" if family == "hsn" else "none",
        variant_params={},
        params=params,
        run_dir=run_dir,
        rules=parse_rules(rules),
        constants={"chance": 0.02},
        on_progress=progress.append,
    )


def _stop_rule(op: str, value: float) -> dict[str, Any]:
    return {
        "early_stop": [
            {
                "name": "low",
                "at_steps": [2, 4],
                "outcome": "fail_early",
                "when": {
                    "all": [
                        {
                            "metric": "eval:delayed/in_range/accuracy",
                            "steps": {"at": 2},
                            "op": op,
                            "value": value,
                        },
                        {
                            "metric": "diag:gate/p50",
                            "steps": {"from": 2, "to": 2},
                            "agg": "median",
                            "op": ">=",
                            "value": 0.0,
                        },
                    ]
                },
                "applies_to": {"family": ["hsn"]},
            },
            {
                "name": "gru_low",
                "at_steps": [2],
                "outcome": "fail_early",
                "when": {
                    "metric": "train:loss",
                    "steps": {"from": 1, "to": 2},
                    "agg": "max",
                    "op": op,
                    "value": 100.0,
                },
                "applies_to": {"family": ["gru"]},
            },
        ]
    }


@pytest.mark.parametrize("family", ["hsn", "gru"])
def test_p1_11_early_stop_rules(family: str, tiny_manifest: Path, tmp_path: Path) -> None:  # noqa: F811
    arm = _arm(family, tiny_manifest, tmp_path / "stop", rules=_stop_rule("<=", 1.0))
    record = run_protocol(arm)
    stop = record["stopped"]
    assert stop is not None and stop["step"] == 2 and stop["outcome"] == "fail_early"
    assert stop["rule"] == ("low" if family == "hsn" else "gru_low")
    assert (
        all(v["result"] is True for v in stop["values"])
        and stop["values"][0]["measured"] is not None
    )
    assert [t["step"] for t in record["train"]] == [1, 2] and record["final"]["steps"] == 2
    assert record["measured"]["steps"] == 2.0
    assert [Path(p).name for p in record["checkpoints"]] == ["step_00000002.pt"]
    # the probe and the trained state still ran, under at_stop, and feed no verdict
    assert record["probe"] is None and record["trained_state"] is None
    assert record["at_stop"]["step"] == 2 and record["at_stop"]["probe"]["n_probed"] > 0
    assert len(record["at_stop"]["probes"]) == record["at_stop"]["probe"]["n_probed"]
    verdicts = evaluate_verdicts(
        parse_rules({"verdicts": {"rs": {"preset": "reads_state"}}}), record, {}, family
    )
    assert verdicts["rs"]["label"] == "stopped early: fail_early @ 2"

    arm2 = _arm(family, tiny_manifest, tmp_path / "go", rules=_stop_rule(">", 1.0))
    record2 = run_protocol(arm2)
    assert record2["stopped"] is None and record2["final"]["steps"] == 6
    assert [e["result"] for e in record2["early_stop_log"]] == (
        [False, False] if family == "hsn" else [False]
    )
    assert record2["probe"]["n_probed"] > 0 and "at_stop" not in record2
    assert [p["step"] for p in arm2_progress(arm2)] == [2, 4, 6]  # 6: reported after the fit


def arm2_progress(arm: ArmRun) -> list[dict[str, Any]]:
    on_progress: Any = arm.on_progress
    return list(on_progress.__self__)  # the list the arm's progress callback appends to


def test_p1_11_reads_state_record(tiny_manifest: Path, tmp_path: Path) -> None:  # noqa: F811
    record = run_protocol(_arm("hsn", tiny_manifest, tmp_path, eval_batch=12))
    assert set(record["walls"]) >= {
        "setup_s",
        "fit_s",
        "train_s",
        "compile_s",
        "eval_s",
        "diag_s",
        "trained_s",
        "probe_s",
        "total_s",
    }
    assert record["walls"]["eval_s"] > 0 and record["eval_count"] == 3
    assert "skipped_nonfinite_steps" in record["final"]  # P1.12's guard (None without it)
    compile_info = record["compile"]  # P2.8 records zeros on an eager run
    assert compile_info["mode"] == "none" and compile_info["compiled_window"] is False
    for key in ("compiled_windows", "compile_seconds", "compile_steps", "window_compile_seconds"):
        assert compile_info[key] in (None, 0.0), (key, compile_info[key])
    assert [d["step"] for d in record["diagnostics"]] == [2, 4, 6]
    check_hsn_reads_state_record(record, eval_steps=[2, 4, 6], diag_steps=[2, 4, 6])
    first = record["evals"][0]
    assert (
        set(first["dev_answer_accuracy"]) == {"kv/in_range", "delayed/in_range"}
        and first["n_windows"] == 3
    )
    probe = record["probe"]
    assert probe["n_probed"] == len(record["probes"]) and probe["verdict"] in (
        "does not read state",
        "no correct answers to probe",
        "partial",
        "reads state",
    )
    assert sum(probe["all"].values()) == probe["n_probed"]
    assert record["trained_state"]["edge_logit/absmax"] == 0.0  # dense symmetry from the zero state
    assert record["variant"] == {"variant": "pdg"} and record["param_count"] > 0


def test_p1_11_overfit_protocol(tiny_manifest: Path, tmp_path: Path) -> None:  # noqa: F811
    cfg = load_config(REF_A1, [*TINY_HSN_OVERRIDES, *disk_overrides(tiny_manifest, steps=6)])
    arm = ArmRun(
        cfg=cfg,
        family="hsn",
        variant="cell_offset",
        variant_params={"offset_std": 0.5},
        params=protocol_params(
            {"name": "overfit", "n_docs": 8, "diag_every": 3, "thresholds": [0.0, 0.99]}
        ),
        run_dir=tmp_path,
        rules=parse_rules(None),
        constants={},
    )
    record = run_protocol(arm)
    overfit = record["overfit"]
    assert (
        overfit["first_ge_0.00"] == 2
        and overfit["final_train_accuracy"] == record["evals"][-1]["accuracy"]
    )
    assert overfit["step_seconds_p50"] > 0 and [d["step"] for d in record["diagnostics"]] == [3, 6]
    assert record["variant"]["variant"] == "cell_offset" and record["variant"]["extra_params"] == 0
    assert "probe" not in record and "trained_state" not in record


def test_p1_11_dry_run_protocol(tmp_path: Path) -> None:
    cfg = load_config(REF_A1, [*TINY_HSN_OVERRIDES, "train.steps=6", "train.eval_every=2"])
    rules = parse_rules(
        {
            "early_stop": [
                {
                    "name": "s",
                    "at_steps": [4],
                    "outcome": "x",
                    "when": {
                        "metric": "eval:accuracy",
                        "steps": {"at": 4},
                        "op": "<",
                        "value": 0.9,
                    },
                }
            ]
        }
    )
    arm = ArmRun(
        cfg=cfg,
        family="hsn",
        variant="pdg",
        variant_params={},
        params=protocol_params(
            {"name": "dry_run", "accuracy": 0.5, "record": {"probe": {"correct_n": 3}}}
        ),
        run_dir=tmp_path,
        rules=rules,
        constants={},
    )
    record = run_protocol(arm)
    assert record["stopped"]["step"] == 4 and [e["step"] for e in record["evals"]] == [2, 4]
    assert record["at_stop"]["probe"] == {"correct_n": 3} and record["probe"] is None
    with pytest.raises(RuntimeError, match="failing as asked"):
        run_protocol(
            ArmRun(
                cfg=cfg,
                family="hsn",
                variant="pdg",
                variant_params={},
                params=protocol_params({"name": "dry_run", "fail": True}),
                run_dir=tmp_path,
                rules=parse_rules(None),
                constants={},
            )
        )


def test_diagnostics_switch_takes_no_sample_and_changes_no_training(
    tiny_manifest: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    """``protocol.diagnostics: false`` (10 Nov 2026): no diagnostic sample during training
    whatever the cadence says, the same training (losses, evaluations) step for step, and the
    end-of-run state statistics and probes still there."""
    on = run_protocol(_arm("hsn", tiny_manifest, tmp_path / "on", diag_stability=True))
    off = run_protocol(
        _arm("hsn", tiny_manifest, tmp_path / "off", diagnostics=False, diag_stability=True)
    )
    assert on["diagnostics_enabled"] is True and [d["step"] for d in on["diagnostics"]] == [2, 4, 6]
    assert off["diagnostics_enabled"] is False and off["diagnostics"] == []
    assert off["phases"]["diagnostics"] == [] and off["walls"]["diag_s"] == 0.0
    assert [t["loss"] for t in off["train"]] == [t["loss"] for t in on["train"]]
    assert [e["accuracy"] for e in off["evals"]] == [e["accuracy"] for e in on["evals"]]
    assert off["final"] == on["final"]
    assert off["trained_state"] == on["trained_state"] and off["probe"] == on["probe"]
    assert len(on["phases"]["diagnostics"]) == 3 and len(on["phases"]["evaluate"]) == 3


def test_diagnostics_switch_is_execution_only_and_rules_see_it(
    tiny_manifest: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    """The switch keeps the arm's id (the training is the same), ``expand(diagnostics=...)``
    (``enqueue --diagnostics on|off``) overrides the spec either way, and a rule that reads a
    ``diag:`` value is refused when nothing will be sampled."""
    from hsn.experiments.spec import SpecError, expand, load_spec
    from p1_14_support import tiny_reads_state_spec

    plain, _ = load_spec(tiny_reads_state_spec(tmp_path, tiny_manifest, name="plain"))
    off, _ = load_spec(
        tiny_reads_state_spec(tmp_path, tiny_manifest, name="off", protocol={"diagnostics": False})
    )
    arm_on, arm_off = expand(plain)[0], expand(off)[0]
    assert arm_on.protocol.diagnostics is True and arm_off.protocol.diagnostics is False
    assert arm_on.arm_id == arm_off.arm_id
    assert expand(plain, diagnostics=False)[0].protocol.diagnostics is False
    assert expand(off, diagnostics=True)[0].protocol.diagnostics is True
    assert expand(plain, diagnostics=False)[0].arm_id == arm_on.arm_id
    needs = {
        "verdicts": {
            "writes": {
                "cases": [
                    {
                        "label": "writes",
                        "when": {
                            "metric": "diag:gate/p50",
                            "steps": {"at": 2},
                            "op": ">=",
                            "value": 0.0,
                        },
                    }
                ],
                "default": "no",
            }
        }
    }
    with_rule, _ = load_spec(
        tiny_reads_state_spec(tmp_path, tiny_manifest, name="ruled", rules=needs)
    )
    assert expand(with_rule)  # sampled: the rule has its step
    with pytest.raises(SpecError):
        expand(with_rule, diagnostics=False)


def test_collector_is_frozen_only_for_compiled_runs_and_released(
    tiny_manifest: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    """``protocol.gc_freeze``: an eager run never freezes; a compiled one (here: a trainer that
    reports a compiled window) freezes after each evaluation and unfreezes when the fit ends."""
    import gc

    from hsn.experiments import protocols

    before = gc.get_freeze_count()  # the interpreter may hold a few frozen objects of its own
    eager = run_protocol(_arm("hsn", tiny_manifest, tmp_path / "eager"))
    assert eager["gc_freezes"] == 0 and gc.get_freeze_count() == before

    arm = _arm("hsn", tiny_manifest, tmp_path / "frozen")
    callback = protocols._StepCallback(arm, None, audit_keys=False)
    callback.gc_freeze = True

    class Compiled:
        compiled_window = object()

    class Eager:
        compiled_window = None

    callback.freeze_collector(Eager())
    assert callback.freezes == 0 and gc.get_freeze_count() == before
    try:
        callback.freeze_collector(Compiled())
        assert callback.freezes == 1 and gc.get_freeze_count() > before + 1000
    finally:
        callback.release_collector()
    assert gc.get_freeze_count() == 0
    callback.gc_freeze = False
    callback.freeze_collector(Compiled())
    assert callback.freezes == 1 and gc.get_freeze_count() == 0
