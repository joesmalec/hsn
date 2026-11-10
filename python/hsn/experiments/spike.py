"""Checkpoint-based spike diagnostics. No continuation and no production weight updates.

Each backward uses the trainer's actual objective and stops at its pre-clip hook. Loss-source
partitions preserve forward values, the full batch, and all original denominators. In particular,
an EMA variance loss is coupled across examples: its masked output VJP is a gradient contribution,
not a separately normalized per-document loss. Tracing uses private input clones so that a
module's input hook cannot accidentally include another consumer's gradient.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterator
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from hsn.config import RunConfig
from hsn.device import resolve, seed_everything, set_deterministic
from hsn.experiments.data_selection import cap_dataset, recorded_policy
from hsn.experiments.spec import SpikeDiagnosticParams, config_view
from hsn.runner import build_model, load_datasets, make_trainer
from hsn.train import tbptt
from hsn.train.divergence import _deterministic_compile, _number, _windows_equal, load_dump
from hsn.train.grad_norm import clip_grad_norm_, norm_float64
from hsn.train.lazy_adam import LazyAdamW
from hsn.train.loop import StreamLoader


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _window_hash(windows: list[dict[str, Tensor]]) -> str:
    """Fingerprint ordered inputs, masks and document boundaries independently of weights."""
    digest = hashlib.sha256()
    for index, window in enumerate(windows):
        for key, value in sorted(window.items()):
            tensor = value.detach().cpu().contiguous()
            header = json.dumps([index, key, str(tensor.dtype), list(tensor.shape)])
            digest.update(header.encode() + b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class _BackwardComplete(Exception):
    """Internal stop, before clipping, guards, optimizer, scheduler, evaluation or saving."""


class _GradientMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: Tensor, mask: Tensor) -> Tensor:
        ctx.save_for_backward(mask)
        return value

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        (mask,) = ctx.saved_tensors
        return gradient * mask, None


def _stats(value: Tensor) -> dict[str, Any]:
    x = value.detach().float()
    result: dict[str, Any] = {
        "shape": list(x.shape),
        "norm": _number(float(x.norm())),
        "abs_max": _number(float(x.abs().max())) if x.numel() else 0.0,
    }
    if x.ndim >= 2 and x.numel():
        result["row_norms"] = [_number(v) for v in x.flatten(1).norm(dim=1).cpu().tolist()]
        rms = x.square().mean(dim=-1).sqrt()
        result["rms_min"] = _number(float(rms.min()))
        result["rms_max"] = _number(float(rms.max()))
    return result


def _tap(value: Tensor, record: dict[str, Any], key: str) -> None:
    if value.requires_grad:

        def capture(gradient: Tensor) -> None:
            record[key] = _stats(gradient)

        value.register_hook(capture)  # type: ignore[no-untyped-call]


@contextlib.contextmanager
def _trace(model: Any, records: list[dict[str, Any]], *, cut: str | None = None) -> Iterator[None]:
    handles: list[Any] = []
    original_position = model._position
    position = 0
    if cut is None:
        for name in ("byte_emb", "encoder", "predictor", "shared", "cell.norm", "cell.gate"):
            module = model.get_submodule(name)
            pending: list[dict[str, Any]] = []

            def before(
                module: Any,
                inputs: tuple[Any, ...],
                name: str = name,
                pending: list[dict[str, Any]] = pending,
            ) -> tuple[Any, ...]:
                value = inputs[0]
                private = value.clone() if value.requires_grad else value
                entry = {"module": name, "position": position, "input": _stats(private)}
                records.append(entry)
                pending.append(entry)
                _tap(private, entry, "local_input_gradient")
                return (private, *inputs[1:])

            def after(
                module: Any,
                inputs: tuple[Any, ...],
                output: Tensor,
                pending: list[dict[str, Any]] = pending,
            ) -> None:
                entry = pending.pop()
                entry["output"] = _stats(output)
                _tap(output, entry, "output_gradient")

            handles.extend(
                (module.register_forward_pre_hook(before), module.register_forward_hook(after))
            )

    def observed(state: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal position
        from hsn.model.state import CellGraphState

        # Diagnostic cuts alter backward only. They are never installed in a training run.
        if cut is not None and position == 31:
            state = CellGraphState._make(
                h=state.h.detach() if cut == "h" else state.h,
                q=state.q.detach() if cut == "q" else state.q,
                age=state.age,
                pos=state.pos,
            )
        result = original_position(state, *args, **kwargs)
        if cut is None:
            entry = {
                "module": "state",
                "position": position,
                "h": _stats(result.state.h),
                "q": _stats(result.state.q),
            }
            records.append(entry)
            _tap(result.state.h, entry, "h_gradient")
            _tap(result.state.q, entry, "q_gradient")
        position += 1
        return result

    model._position = observed
    try:
        yield
    finally:
        model._position = original_position
        for handle in handles:
            handle.remove()


@dataclasses.dataclass
class Pass:
    report: dict[str, Any]
    gradients: dict[str, Tensor]
    outputs: list[dict[str, Tensor]]
    trainer: Any


def compare(a: Pass, b: Pass) -> dict[str, Any]:
    same_keys = len(a.outputs) == len(b.outputs) and all(
        x.keys() == y.keys() for x, y in zip(a.outputs, b.outputs, strict=False)
    )
    exact = same_keys and all(
        torch.equal(x[k], y[k]) for x, y in zip(a.outputs, b.outputs, strict=True) for k in x
    )
    keys = a.gradients.keys() | b.gradients.keys()
    square, reference = 0.0, 0.0
    maximum = 0.0
    for name in keys:
        x = a.gradients.get(name)
        y = b.gradients.get(name)
        if x is None:
            assert y is not None
            x = torch.zeros_like(y)
        if y is None:
            y = torch.zeros_like(x)
        delta = x.double() - y.double()
        square += float(delta.square().sum())
        reference += float(x.double().square().sum())
        maximum = max(maximum, float(delta.abs().max())) if delta.numel() else maximum
    relative = math.sqrt(square / max(reference, 1e-30))
    route_changes = (
        {
            key: sum(
                int((x[key] != y[key]).sum())
                for x, y in zip(a.outputs, b.outputs, strict=True)
                if key in x and key in y
            )
            for key in ("cell_idx", "unit_idx", "valid")
        }
        if same_keys
        else {}
    )
    logits_error = (
        max(
            (
                float((x["logits"].float() - y["logits"].float()).abs().max())
                for x, y in zip(a.outputs, b.outputs, strict=True)
            ),
            default=0.0,
        )
        if same_keys
        else None
    )
    return {
        "forward_exact": exact,
        "gradient_relative_error": _number(relative),
        "gradient_abs_max_error": _number(maximum),
        "routing_mismatch_counts": route_changes,
        "logits_abs_max_error": logits_error,
        "passed": exact and math.isfinite(relative) and relative <= 1e-5,
    }


class Case:
    def __init__(
        self,
        path: Path,
        cfg: RunConfig,
        *,
        train_ds: Any = None,
        reference_info: dict[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.payload = (
            load_dump(path)
            if reference_info is None
            else torch.load(path, map_location="cpu", weights_only=True)
        )
        self.info = (
            self.payload["extra"]["divergence"]
            if reference_info is None
            else {**reference_info, "step": int(self.payload["step"]) + 1}
        )
        self.cfg = RunConfig.model_validate(self.info["config"])
        if config_view(self.cfg) != config_view(cfg):
            raise ValueError("diagnostic arm configuration differs from the checkpoint")
        if self.info.get("notes", {}).get("variant", "pdg") != "pdg":
            raise ValueError("spike diagnostics currently require the pdg variant")
        if self.cfg.train.lambda_u or self.cfg.train.gate_mode == "reinforce":
            raise ValueError(
                "spike diagnostics do not yet attribute counterfactual or REINFORCE losses"
            )
        self.step = int(self.info["step"])
        if train_ds is None:
            train_ds, _ = load_datasets(self.cfg)
        count = int(self.payload["loader_state"]["n_docs"])
        if len(train_ds) < count:
            raise ValueError("training data is smaller than the saved stream")
        if len(train_ds) != count:
            notes = self.info.get("notes") or {}
            selection = notes.get("data_selection") or {}
            train_ds = cap_dataset(
                train_ds,
                count,
                seed=selection.get("seed", self.cfg.seed),
                policy=recorded_policy(notes),
            )
        self.data = train_ds
        loader = StreamLoader(
            train_ds,
            batch=self.cfg.train.batch,
            unroll=self.cfg.train.unroll,
            seed=self.cfg.seed,
            shuffle=True,
            align_docs=self.cfg.data.align_docs,
            epochs=None,
        )
        loader.load_state_dict(self.payload["loader_state"])
        self.windows = (
            self.info["windows"]
            if reference_info is None
            else [next(loader) for _ in range(self.cfg.train.grad_accum)]
        )
        if reference_info is None and not all(
            _windows_equal(next(loader), w) for w in self.windows
        ):
            raise ValueError("checkpoint windows no longer match the recorded training stream")
        self.docs = sorted(
            {int(d) for w in self.windows for d in w["doc"].unique().tolist() if d >= 0}
        )
        self.window_hash = _window_hash(self.windows)

    def run(
        self,
        directory: Path,
        *,
        compiled: bool,
        deterministic: bool,
        term: str | None = None,
        doc: int | None = None,
        complement: bool = False,
        trace: bool = False,
        cut: str | None = None,
        weight_override: Path | None = None,
        weight_selectors: tuple[str, ...] = (),
    ) -> Pass:
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        ctx = dataclasses.replace(resolve(self.cfg.hardware), compile_enabled=compiled)
        seed_everything(self.cfg.seed)
        model = build_model(self.cfg)
        trainer: Any = make_trainer(model, self.cfg, ctx, run_dir=directory)
        trainer.divergence_dump = False
        set_deterministic(deterministic)
        trainer.resume(self.path)
        weight_step = int(self.payload["step"])
        selected_weights: list[str] = []
        if weight_selectors and weight_override is None:
            raise ValueError("weight selectors require a reference checkpoint")
        if weight_override is not None:
            weights = torch.load(weight_override, map_location="cpu", weights_only=True)
            expected = self.payload["extra"].get("config_hash")
            if not expected or weights["extra"].get("config_hash") != expected:
                raise ValueError("reference weights have a different or missing configuration hash")
            # Keep the source stream, RNG, recurrent state and previous byte. Replace only model
            # parameters and buffers (including the EMA target); stop before any optimizer update.
            if weight_selectors:
                named = dict(model.named_parameters())
                selected: set[str] = set()
                for selector in weight_selectors:
                    matches = {name for name in named if fnmatchcase(name, selector)}
                    if not matches:
                        raise ValueError(f"reference selector {selector!r} matches no parameters")
                    selected.update(matches)
                selected_weights = sorted(selected)
                replacement = model.state_dict()
                replacement.update({name: weights["model"][name] for name in selected_weights})
                model.load_state_dict(replacement, strict=True)
            else:
                model.load_state_dict(weights["model"], strict=True)
            weight_step = int(weights["step"])
        if trainer.scaler is not None:
            raise ValueError("diagnostic backward requires an unscaled fp32/bf16 recipe")
        gradients: dict[str, Tensor] = {}
        outputs: list[dict[str, Tensor]] = []
        records: list[dict[str, Any]] = []
        parts: list[dict[str, float]] = []
        active: Tensor | None = None
        objective_module: Any = tbptt  # isolated diagnostic patch of the trainer's private binding
        original_terms = objective_module.assemble_terms
        original_forward = trainer._forward

        def forward(batch: Any, state: Any, **kwargs: Any) -> Any:
            nonlocal active
            logits, new_state, info = original_forward(batch, state, **kwargs)
            output = {"logits": logits.detach().cpu().clone()}
            output["h"] = new_state.h.detach().cpu().clone()
            output["q"] = new_state.q.detach().cpu().clone()
            for key in ("cell_idx", "unit_idx", "valid"):
                if key in info:
                    output[key] = info[key].detach().cpu().clone()
            outputs.append(output)
            units = trainer._window_active_units(info)
            if units is not None:
                active = units if active is None else active | units
            return logits, new_state, info

        def terms(out: Any, batch: Any, cfg: Any, extras: Any = None) -> dict[str, Tensor]:
            if doc is not None:
                mask = batch["doc"] == doc
                if complement:
                    mask = ~mask
                # Only the loss's private inputs are masked; recurrent forward paths are intact.
                out = {
                    key: _GradientMask.apply(
                        value, mask.reshape(*mask.shape, *([1] * (value.ndim - 2)))
                    )
                    if isinstance(value, Tensor)
                    and value.requires_grad
                    and value.ndim >= 2
                    and value.shape[:2] == mask.shape
                    else value
                    for key, value in out.items()
                }
            result: dict[str, Tensor] = original_terms(out, batch, cfg, extras)
            parts.append({k: float(v.detach()) for k, v in result.items()})
            if term is not None:
                if term not in result:
                    raise ValueError(f"objective has no {term!r} term")
                result = {k: v if k == term else v.detach() for k, v in result.items()}
            return result

        report: dict[str, Any] = {
            "compiled": compiled,
            "deterministic": deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "term": term,
            "document": doc,
            "complement": complement,
            "stream_checkpoint": str(self.path),
            "weights_checkpoint": str(weight_override or self.path),
            "weights_step": weight_step,
            "weight_selectors": list(weight_selectors),
            "selected_parameters": selected_weights,
            "reference_buffers_loaded": weight_override is not None and not weight_selectors,
            "window_sha256": self.window_hash,
            "all_rows_reset_at_window_start": bool(self.windows[0]["reset"][:, 0].all()),
        }

        def stop(t: Any) -> None:
            present = [p.grad for p in t.model.parameters() if p.grad is not None]
            norm = (
                norm_float64(present)
                if self.cfg.train.grad_norm_precision == "float64"
                else torch.linalg.vector_norm(torch.stack([g.norm() for g in present]))
            )
            report["raw_grad_norm"] = _number(float(norm))
            for name, p in t.model.named_parameters():
                if p.grad is not None:
                    value = p.grad.detach()
                    if self.cfg.train.grad_norm_precision == "native":
                        value = value.float()
                    gradients[name] = value.cpu().clone()
            raise _BackwardComplete()

        if compiled and (trace or cut):
            raise ValueError("local tracing and gradient cuts require eager execution")
        trainer._forward = forward
        trainer.grad_hook = stop
        objective_module.assemble_terms = terms
        tracing = _trace(model, records, cut=cut) if trace or cut else contextlib.nullcontext()
        try:
            with _deterministic_compile(compiled and deterministic), tracing:
                try:
                    # Avoid forcing last-step statistics: the saved step was usually not one.
                    trainer.fit(self.data, self.data, min(self.step + 1, self.cfg.train.steps))
                except _BackwardComplete:
                    pass
                else:
                    raise RuntimeError("diagnostic did not stop before its optimizer update")
        finally:
            objective_module.assemble_terms = original_terms
            trainer._forward = original_forward
            trainer.grad_hook = None
        report["objective_parts"] = parts
        report["parameter_grad_norms"] = {
            k: _number(
                float(
                    norm_float64([v])
                    if self.cfg.train.grad_norm_precision == "float64"
                    else v.norm()
                )
            )
            for k, v in gradients.items()
        }
        report["grad_norm_precision"] = self.cfg.train.grad_norm_precision
        report["optimizer_updates"] = 0
        report["active_units"] = (
            None if active is None else active.nonzero().flatten().cpu().tolist()
        )
        report["trace"] = records
        trainer._diagnostic_active = active
        return Pass(report, gradients, outputs, trainer)


def _partition_error(base: Pass, pieces: list[Pass]) -> float:
    summed = {
        k: sum((p.gradients.get(k, torch.zeros_like(v)) for p in pieces), torch.zeros_like(v))
        for k, v in base.gradients.items()
    }
    result = compare(base, Pass({}, summed, base.outputs, None))
    return float(result["gradient_relative_error"])


def _shadow_update(case: Case, baseline: Pass, monitor_docs: int) -> dict[str, Any]:
    trainer = baseline.trainer
    monitor = cap_dataset(case.data, monitor_docs, seed=case.cfg.seed)
    if any(e.meta.get("split") != "train" for e in monitor.examples):
        raise ValueError("shadow update monitor must be training data")
    trainer._adaptive_monitor = monitor
    # Evaluation is eager for both before and after: no new B1 compiled graph, identical metric.
    trainer._eval_model = lambda: trainer.model
    before_loss = trainer._adaptive_monitor_loss()
    before = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
    parameters = [p for p in trainer.model.parameters() if p.requires_grad]
    raw = float(
        clip_grad_norm_(
            parameters,
            case.cfg.train.grad_clip or math.inf,
            precision=case.cfg.train.grad_norm_precision,
        )
    )
    clipped = float(torch.stack([p.grad.norm() for p in parameters if p.grad is not None]).norm())
    kwargs = (
        {"active_units": trainer._diagnostic_active}
        if isinstance(trainer.optimizer, LazyAdamW)
        else {}
    )
    trainer.optimizer.step(**kwargs)
    trainer.model.update_target()
    delta: dict[str, Any] = {}
    for name, p in trainer.model.named_parameters():
        change = float((p.detach() - before[name]).float().norm())
        delta[name] = {
            "norm": _number(change),
            "relative_to_before": _number(change / max(float(before[name].float().norm()), 1e-12)),
        }
    after_loss = trainer._adaptive_monitor_loss()
    return {
        "disposable_copy": True,
        "optimizer_updates": 1,
        "production_updates": 0,
        "raw_grad_norm": _number(raw),
        "clipped_grad_norm": _number(clipped),
        "parameter_updates": delta,
        "monitor_fingerprint": monitor.fingerprint,
        "monitor_docs": len(monitor),
        "monitor_before": before_loss,
        "monitor_after": after_loss,
        "monitor_change": after_loss - before_loss,
        "interpretation": (
            "One clipped optimizer update with saved moments and rates; "
            "not evidence of long-term accuracy or permission to weaken the guard."
        ),
    }


def run_diagnostic(
    arm: Any, params: SpikeDiagnosticParams, *, train_ds: Any = None
) -> dict[str, Any]:
    path = Path(params.dump)
    actual_hash = sha256(path)
    if actual_hash != params.dump_sha256:
        raise ValueError("diagnostic checkpoint checksum changed")
    if params.reference_mode == "same_window_weights" and params.reference_checkpoint is None:
        raise ValueError("same_window_weights requires a reference checkpoint")
    arm.stage = "diagnostic_setup"
    result: dict[str, Any] = {
        "protocol": "spike_diagnostic",
        "family": arm.family,
        "diagnostic": {
            "parent_exp": params.parent_exp,
            "dump": str(path),
            "dump_sha256": actual_hash,
            "stage": params.stage,
            "timeout_s": params.timeout_s,
            "production_updates": 0,
            "passes": [],
            "checks": [],
        },
    }
    arm.record.update(result)
    record = result["diagnostic"]
    if params.capture is not None:
        from hsn.experiments.capture import capture_prefix

        return capture_prefix(arm, params, result, train_ds=train_ds)
    case = Case(path, arm.cfg, train_ds=train_ds)
    if params.target_doc is not None and params.target_doc not in case.docs:
        raise ValueError("target document is absent from the saved windows")
    record.update(
        seed=case.cfg.seed,
        data_fingerprint=case.data.fingerprint,
        source_config_hash=case.payload["extra"].get("config_hash"),
        previous_byte_saved=case.payload["extra"].get("prev_tokens") is not None,
        dump_commit=case.info.get("git_rev"),
        step=case.step,
        windows_match=True,
        window_sha256=case.window_hash,
        original_loss=case.info.get("loss"),
        original_raw_grad_norm=case.info["grad_norm"],
    )
    record["windows"] = [
        {"documents": w["doc"].tolist(), "answer_positions": w["loss_mask"].nonzero().tolist()}
        for w in case.windows
    ]
    with tempfile.TemporaryDirectory(dir=arm.run_dir, prefix="spike-") as directory:

        def run(label: str, mode: str = "eager", **kwargs: Any) -> Pass:
            arm.stage = label
            arm.on_progress(
                {"stage": label, "parent_exp": params.parent_exp, "diagnostic": params.stage}
            )
            out = case.run(
                Path(directory) / label,
                compiled=mode == "compiled",
                deterministic=params.deterministic,
                **kwargs,
            )
            report = {"label": label, **out.report}
            trace = report.pop("trace")
            if trace:
                trace_path = arm.run_dir / f"{label}.trace.json"
                trace_path.write_text(json.dumps(trace, allow_nan=False) + "\n")
                report["trace_path"] = str(trace_path)
            record["passes"].append(report)
            return out

        baselines: dict[Literal["eager", "compiled"], Pass] = {}
        for mode in params.modes:
            baseline = run(f"{mode}_baseline", mode)
            baselines[mode] = baseline
            for repeat in range(1, params.repeats):
                repeated = run(f"{mode}_repeat_{repeat}", mode)
                check = {"kind": "repeatability", "mode": mode, **compare(baseline, repeated)}
                record["checks"].append(check)
                if not check["passed"]:
                    raise RuntimeError(f"{mode} replay is not stable enough for causal comparisons")
                del repeated
        if len(baselines) == 2:
            record["execution_mode_comparison"] = compare(baselines["eager"], baselines["compiled"])
        if params.stage in ("isolate", "compare"):
            attribution: dict[str, Any] = {}
            record["attribution_by_mode"] = attribution
            for mode, base in baselines.items():
                measurements: dict[str, Any] = {}
                attribution[mode] = measurements
                prefix = "" if mode == "eager" else f"{mode}_"
                pieces = [
                    run(f"{prefix}term_{name}", mode, term=name)
                    for name in base.report["objective_parts"][0]
                ]
                measurements["term_partition_relative_error"] = _partition_error(base, pieces)
                for piece in pieces:
                    if not compare(base, piece)["forward_exact"]:
                        raise RuntimeError("term isolation changed forward values")
                del pieces
                if params.target_doc is not None:
                    if params.target_doc not in case.docs:
                        raise ValueError("target document is absent from the saved windows")
                    selected = run(f"{prefix}document_only", mode, doc=params.target_doc)
                    rest = run(
                        f"{prefix}other_documents", mode, doc=params.target_doc, complement=True
                    )
                    measurements["document_partition_relative_error"] = _partition_error(
                        base, [selected, rest]
                    )
                    if not all(compare(base, p)["forward_exact"] for p in (selected, rest)):
                        raise RuntimeError("document isolation changed forward values")
                tolerance = 0.02 if base.trainer.ctx.autocast_dtype == torch.bfloat16 else 1e-5
                measurements["tolerance"] = tolerance
                if mode == "eager":
                    record.update(measurements)  # preserve the original eager result fields
                for key in ("term_partition_relative_error", "document_partition_relative_error"):
                    if key in measurements and not measurements[key] <= tolerance:
                        raise RuntimeError(
                            f"{mode} {key} exceeds the numerical tolerance {tolerance}; "
                            "attribution is inconclusive"
                        )
        if params.stage in ("trace", "compare"):
            if "eager" not in baselines:
                raise ValueError("local tracing needs an eager baseline")
            base = baselines["eager"]
            traced = run("local_trace", trace=True)
            check = {"kind": "instrumentation", **compare(base, traced)}
            record["checks"].append(check)
            if not check["passed"]:
                raise RuntimeError("instrumentation changed the baseline; trace is not valid")
            del traced
            for branch in ("h", "q"):
                cut = run(f"cut_{branch}_at_31", cut=branch)
                if not compare(base, cut)["forward_exact"]:
                    raise RuntimeError("diagnostic gradient cut changed forward values")
                del cut
        if params.stage == "update":
            record["shadow_updates"] = {
                mode: _shadow_update(case, base, params.monitor_docs)
                for mode, base in baselines.items()
            }
        if (
            params.reference_checkpoint is not None
            and params.reference_mode == "same_window_weights"
        ):
            reference_path = Path(params.reference_checkpoint)
            if sha256(reference_path) != params.reference_sha256:
                raise ValueError("reference checkpoint checksum changed")
            reference_results: dict[str, Any] = {}
            record["reference"] = {
                "mode": params.reference_mode,
                "path": str(reference_path),
                "sha256": params.reference_sha256,
                "same_windows": True,
                "window_sha256": case.window_hash,
                "selectors": list(params.reference_selectors),
                "by_mode": reference_results,
                "interpretation": (
                    "Reference weights on the source spike's exact window, RNG, carried state "
                    "and previous byte. With selectors, only matching parameters change; other "
                    "parameters and buffers stay at the source. Without selectors all model "
                    "parameters and buffers change. No optimizer update. A mixed-weight "
                    "intervention is not a reconstructed training trajectory or proof of a cure."
                ),
            }
            for mode, baseline in baselines.items():
                ref = run(
                    f"reference_{mode}",
                    mode,
                    weight_override=reference_path,
                    weight_selectors=params.reference_selectors,
                )
                reference_results[mode] = {
                    "report": ref.report,
                    "source_comparison": compare(baseline, ref),
                }
                for repeat in range(1, params.repeats):
                    repeated = run(
                        f"reference_{mode}_repeat_{repeat}",
                        mode,
                        weight_override=reference_path,
                        weight_selectors=params.reference_selectors,
                    )
                    check = {
                        "kind": "reference_repeatability",
                        "mode": mode,
                        **compare(ref, repeated),
                    }
                    record["checks"].append(check)
                    if not check["passed"]:
                        raise RuntimeError("reference weight replay is not repeatable")
                    del repeated
                del ref
        elif params.reference_checkpoint is not None:
            reference_path = Path(params.reference_checkpoint)
            if sha256(reference_path) != params.reference_sha256:
                raise ValueError("reference checkpoint checksum changed")
            reference = Case(reference_path, arm.cfg, train_ds=case.data, reference_info=case.info)
            arm.on_progress({"stage": "reference_backward", "parent_exp": params.parent_exp})
            ref = reference.run(
                Path(directory) / "reference", compiled=False, deterministic=params.deterministic
            )
            traced_ref = reference.run(
                Path(directory) / "reference_trace",
                compiled=False,
                deterministic=params.deterministic,
                trace=True,
            )
            check = {"kind": "reference_instrumentation", **compare(ref, traced_ref)}
            record["checks"].append(check)
            reference_trace = arm.run_dir / "reference.trace.json"
            reference_trace.write_text(
                json.dumps(traced_ref.report.pop("trace"), allow_nan=False) + "\n"
            )
            record["reference"] = {
                "mode": "next_window",
                "path": str(reference_path),
                "sha256": params.reference_sha256,
                "step": reference.step,
                "same_windows": reference.window_hash == case.window_hash,
                "window_sha256": reference.window_hash,
                "trace_path": str(reference_trace),
                "report": ref.report,
                "interpretation": (
                    "Next window at an earlier retained checkpoint, with no optimizer update. "
                    "Earlier does not by itself establish healthy."
                ),
            }
            if not check["passed"]:
                raise RuntimeError("reference instrumentation changed its baseline")
    record["completed"] = True
    record["interpretation"] = (
        "Diagnostic backward interventions and disposable updates only; "
        "no production fix established."
    )
    arm.on_progress(
        {"stage": "complete", "parent_exp": params.parent_exp, "diagnostic": params.stage}
    )
    return result
