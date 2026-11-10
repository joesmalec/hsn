"""Verified replay of a recorded prefix on disposable weights, without discarding samples.

Capture occurs after clipping and before the selected step's optimizer update. Every replayed
loss and raw norm must agree with the immutable source trace; the final captured step is never
applied. Repeated captures additionally compare weights, moments, RNG, stream and carried state.
This creates inputs for the existing backward-only diagnostics, not a new training result.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from hsn.device import resolve, seed_everything
from hsn.experiments.data_selection import cap_dataset, recorded_policy
from hsn.experiments.spec import SpikeDiagnosticParams
from hsn.experiments.spike import _window_hash, sha256
from hsn.runner import build_model, load_datasets, make_trainer
from hsn.train.divergence import write_dump


class _CaptureComplete(Exception):
    """Stop the disposable trainer before the final requested optimizer update."""


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(_equal(x, y) for x, y in zip(a, b, strict=True))
        )
    return bool(a == b)


def _training_state(payload: dict[str, Any]) -> dict[str, Any]:
    # Capture metadata contains paths; the complete training state must match independently.
    extra = {k: v for k, v in payload["extra"].items() if k != "divergence"}
    return {
        **{
            k: payload[k]
            for k in ("step", "model", "optimizer", "scheduler", "rng_state", "loader_state")
        },
        "extra": extra,
    }


def capture_prefix(
    arm: Any,
    params: SpikeDiagnosticParams,
    result: dict[str, Any],
    *,
    train_ds: Any = None,
) -> dict[str, Any]:
    capture = params.capture
    assert capture is not None
    cfg = arm.cfg
    ctx = resolve(cfg.hardware)
    if (params.modes[0] == "compiled") != ctx.compile_enabled:
        raise ValueError("capture must use the original execution mode")
    if params.deterministic != cfg.train.deterministic:
        raise ValueError("capture must preserve the original deterministic setting")
    if cfg.train.spike_factor is not None or cfg.train.rollback_every is not None:
        raise ValueError("capture currently requires an unguarded source without rollback")
    source = Path(params.dump)
    if arm.run_dir.resolve() == source.parent.resolve():
        raise ValueError("capture output must be separate from the source experiment")
    trace_path = Path(capture.trace)
    if sha256(trace_path) != capture.trace_sha256:
        raise ValueError("capture trace checksum changed")
    trace = json.loads(trace_path.read_text())
    source_run_id = (trace.get("arm") or {}).get("run_id")
    if not isinstance(source_run_id, str) or not source_run_id:
        raise ValueError("capture trace must identify the source run_id")
    # Enqueuing assigns the diagnostic its own identity. Resume still verifies the full
    # original trainer configuration, which includes its run_id; no other field is changed.
    cfg = cfg.model_copy(update={"run_id": source_run_id})
    payload = torch.load(source, map_location="cpu", weights_only=False)
    start = int(payload["step"])
    if capture.steps[0] <= start or capture.steps[-1] > cfg.train.steps:
        raise ValueError("capture steps must follow the checkpoint within its original schedule")
    recorded = {int(row["step"]): row for row in trace["train"]}
    needed = range(start + 1, capture.steps[-1] + 1)
    if any(step not in recorded for step in needed):
        raise ValueError("capture trace is missing steps in the replayed prefix")
    dev_ds = train_ds
    if train_ds is None:
        train_ds, dev_ds = load_datasets(cfg)
    count = int(payload["loader_state"]["n_docs"])
    if len(train_ds) < count:
        raise ValueError("capture data is smaller than the recorded stream")
    if len(train_ds) != count:
        selection = trace.get("data_selection") or {}
        train_ds = cap_dataset(
            train_ds, count, seed=selection.get("seed", cfg.seed), policy=recorded_policy(trace)
        )
    record = result["diagnostic"]
    record.update(
        source_step=start,
        source_run_id=source_run_id,
        source_config_hash=payload["extra"]["config_hash"],
        source_trace=str(trace_path),
        source_trace_sha256=capture.trace_sha256,
        data_fingerprint=train_ds.fingerprint,
        capture_steps=list(capture.steps),
        tolerance={"rtol": capture.rtol, "atol": capture.atol},
        replay_runs=[],
        captures=[],
    )
    first_paths: dict[int, Path] = {}
    with tempfile.TemporaryDirectory(dir=arm.run_dir, prefix="capture-repeats-") as temporary:
        for repeat in range(params.repeats):
            directory = arm.run_dir if repeat == 0 else Path(temporary) / str(repeat)
            directory.mkdir(parents=True, exist_ok=True)
            seed_everything(cfg.seed)
            model = build_model(cfg)
            trainer = make_trainer(model, cfg, ctx, run_dir=directory)
            trainer.resume(source)
            trainer.divergence_dump = True
            trainer.dump_notes = {
                "variant": arm.variant,
                "data_selection": trace.get("data_selection"),
                "capture_parent_exp": params.parent_exp,
                "capture_source": str(source),
                "capture_source_sha256": params.dump_sha256,
            }
            run: dict[str, Any] = {"repeat": repeat, "prefix": [], "captures": []}
            record["replay_runs"].append(run)

            def observe(
                step: int,
                grad_norm: float,
                skipped: bool,
                *,
                trainer: Any = trainer,
                run: dict[str, Any] = run,
                repeat: int = repeat,
            ) -> None:
                snapshot = trainer._step_snapshot
                if skipped or snapshot is None or snapshot.loss is None:
                    raise ValueError(
                        f"capture cannot verify an absent or skipped snapshot at {step}"
                    )
                actual = {"loss": snapshot.loss, "grad_norm": grad_norm}
                expected = {key: float(recorded[step][key]) for key in actual}
                passed = all(
                    math.isfinite(value)
                    and math.isclose(
                        value, expected[key], rel_tol=capture.rtol, abs_tol=capture.atol
                    )
                    for key, value in actual.items()
                )
                check = {"step": step, "actual": actual, "expected": expected, "passed": passed}
                run["prefix"].append(check)
                if not passed:
                    arm.on_progress({"stage": "capture_mismatch", "step": step, "check": check})
                    raise ValueError(f"capture prefix differs from source at step {step}: {check}")
                if step in capture.steps:
                    summary = write_dump(
                        trainer,
                        snapshot,
                        step=step,
                        kind="capture",
                        grad_norm=grad_norm,
                        record_event=False,
                    )
                    if "error" in summary:
                        raise RuntimeError(f"capture write failed: {summary['error']}")
                    path = Path(summary["path"])
                    item = {
                        **summary,
                        "sha256": sha256(path),
                        "window_sha256": _window_hash(snapshot.windows),
                        "all_rows_reset": all(
                            bool(w["reset"][:, 0].all()) for w in snapshot.windows
                        ),
                    }
                    run["captures"].append(item)
                    if repeat == 0:
                        first_paths[step] = path
                        record["captures"].append(item)
                    else:
                        first = torch.load(
                            first_paths[step], map_location="cpu", weights_only=False
                        )
                        again = torch.load(path, map_location="cpu", weights_only=False)
                        exact = _equal(_training_state(first), _training_state(again))
                        record["checks"].append(
                            {
                                "kind": "capture_repeat",
                                "step": step,
                                "repeat": repeat,
                                "passed": exact,
                            }
                        )
                        if not exact:
                            raise ValueError(
                                f"capture training state differs across repeats at {step}"
                            )
                    arm.on_progress(
                        {
                            "stage": "capture",
                            "step": step,
                            "repeat": repeat,
                            "parent_exp": params.parent_exp,
                            "prefix_matched": True,
                        }
                    )
                if step == capture.steps[-1]:
                    raise _CaptureComplete

            # Diagnostic-only observer; production loops and optimizer semantics are untouched.
            trainer._capture_healthy = observe  # type: ignore[method-assign]
            arm.stage = "capture_prefix"
            try:
                # Keep the original horizon: a shorter fit would force extra statistics,
                # evaluation and checkpoint work on the requested capture step.
                trainer.fit(train_ds, dev_ds, cfg.train.steps)
            except _CaptureComplete:
                pass
            else:
                raise RuntimeError("capture did not reach its requested stopping point")
            finally:
                delattr(trainer, "_capture_healthy")
            run["disposable_updates"] = trainer.step - start
            run["prefix_matched"] = True
            run["completed"] = True
            if repeat:
                # Repetition files are temporary. Keep hashes and measurements, not dangling links.
                for item in run["captures"]:
                    item.pop("path", None)
            del observe, trainer, model
            torch._dynamo.reset()
    reference = arm.run_dir / source.name
    if reference.resolve() == source.resolve():
        raise ValueError("capture output must be separate from the source experiment")
    shutil.copy2(source, reference)
    record["reference_checkpoint"] = str(reference)
    record["reference_sha256"] = sha256(reference)
    record["completed"] = True
    record["prefix_matched"] = True
    arm.stage = "complete"
    arm.on_progress({"stage": "complete", "diagnostic": "capture", "parent_exp": params.parent_exp})
    return result
