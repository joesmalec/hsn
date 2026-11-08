"""Divergence dumps and their deterministic replay (P1.15).

Divergences are rare and compiled CUDA is not bitwise reproducible across processes, so a rerun does
not reproduce them (exps 90 and 91 diverged, their reruns 168 and 169 did not). With
``train.divergence_dump`` (default on) the trainer keeps, at the start of every step, what that step
starts from (:class:`StepSnapshot`): the loader's position, every random generator's state, a copy
of the carried recurrent state and of the model's buffers, its own counters, and (by reference) the
step's CPU windows. The parameters, the optimiser state and the scheduler are left alone: nothing
changes them before the optimiser step. At the **first** step a guard skips (P1.12's non-finite
norm, or P1.15's gradient spike) :func:`write_dump` writes, before the gradients are zeroed:

* ``<run_dir>/divergence_<step:08d>.pt``: a checkpoint in :mod:`hsn.train.checkpoint`'s format at
  the step before (the parameters, optimiser and scheduler as they are, which is their pre-step
  state; the snapshot's buffers, generators, loader position, carried state and counters), so that
  :meth:`hsn.train.loop.Trainer.resume` restores it with all its checks; its ``extra["divergence"]``
  holds the event (kind, step, pre-clip norm, loss, learning rate, clip coefficient), the step's
  windows with their document ids per row, the resolved configuration, the versions and git
  revision, and per parameter whether its (clipped) gradient was finite and its norm;
* ``divergence_<step:08d>.json`` next to it, the event and the per-parameter summary.

A failure while writing is logged at ERROR and kept in ``trainer.divergence`` (``error``); training
continues (the dump is a diagnostic, not the training). Under an fp16 ``GradScaler`` the guards are
inactive (the scaler skips steps itself), so there is no dump. ``train.divergence_dump=false`` keeps
no snapshot at all.

**Replay** (:func:`replay`, ``python -m hsn.experiments replay``): builds the model and trainer from
the dump's configuration, resumes from the dump and runs that one step again, under deterministic
kernels (``set_deterministic(True)`` after the trainer is built, so the configuration and its hash
stay the dump's) and, compiled, inductor's ``deterministic`` mode, as P1.13's CUDA golden test does.
It runs eager (``eager=True``: a device context with ``compile_enabled=False``, the configuration
unchanged) or compiled, ``repeats`` times in one process, and reports the replayed loss and pre-clip
norm against the recorded ones and against each other (deterministic replays agree bit for bit), the
regenerated windows against the stored ones (they must be equal), the raw per-parameter gradient
norms (a hook before the clipping) and, eager, the RMS of ``model.cell.cand_norm``'s input (the
near-zero pre-norm hypothesis of exp 171). ``cpu=True`` replays a CUDA dump on ``profile:cpu`` in
fp32 (approximate: another device and dtype; the CUDA generators are not restored).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import os
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

__all__ = [
    "DUMP_PREFIX",
    "StepSnapshot",
    "dump_paths",
    "load_dump",
    "replay",
    "write_dump",
]

logger = logging.getLogger("hsn.train")

#: the dump's file name prefix (not ``step_``: :func:`hsn.train.checkpoint.latest_checkpoint` must
#: never take a dump for the run's checkpoint)
DUMP_PREFIX = "divergence_"


@dataclass
class StepSnapshot:
    """What one optimiser step starts from (module docstring), taken by the trainer at its start."""

    step: int  # the trainer's step before this one
    loader_state: dict[str, Any]
    rng_state: dict[str, Any]
    state: Any  # a copy of the carried recurrent state, on the device
    buffers: dict[str, Tensor]  # copies of the model's persistent buffers
    counters: dict[str, Any]  # the trainer's counters (guards, windows, resyncs, ...)
    windows: list[dict[str, Tensor]] = field(default_factory=list)  # the step's CPU windows
    loss: float | None = None  # the step's mean task loss, set before the guards run


def persistent_buffers(model: torch.nn.Module) -> dict[str, Tensor]:
    """Copies of the model's buffers that its state dict holds (P2.3's scorer EMAs change within a
    step; the others are fixed; all are small)."""
    keys = set(model.state_dict().keys())
    return {name: b.detach().clone() for name, b in model.named_buffers() if name in keys}


def _git_rev() -> str | None:
    from hsn.config import REPO_ROOT

    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _number(x: float) -> float | str:
    return float(x) if math.isfinite(x) else repr(float(x))


def _gradient_summary(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            out[name] = {"grad": None}
            continue
        g = p.grad.detach().float()
        finite = bool(torch.isfinite(g).all())
        out[name] = {
            "finite": finite,
            "norm": _number(float(torch.linalg.vector_norm(g))),
            "abs_max": _number(float(g.abs().max())) if g.numel() else 0.0,
        }
    return out


def _window_docs(window: Mapping[str, Tensor]) -> list[list[int]]:
    """Per row, the document ids of a CPU window in order (padding left out)."""
    rows = []
    for row in window["doc"].tolist():
        seen: list[int] = []
        for d in row:
            if d >= 0 and (not seen or seen[-1] != d):
                seen.append(int(d))
        rows.append(seen)
    return rows


def write_dump(
    trainer: Any,
    snapshot: StepSnapshot,
    *,
    step: int,
    kind: str,
    grad_norm: float,
) -> dict[str, Any]:
    """Write the dump of ``step`` (module docstring) and return its summary (also kept as
    ``trainer.divergence``). Called by the guards before they zero the gradients."""
    from hsn.train.checkpoint import capture_rng_state, save_checkpoint

    run_dir = Path(trainer.run_dir)
    path = run_dir / f"{DUMP_PREFIX}{step:08d}.pt"
    cfg = trainer.cfg
    max_norm = float(cfg.train.grad_clip) if cfg.train.grad_clip > 0 else math.inf
    clip_coef = (
        min(1.0, max_norm / (grad_norm + 1e-6)) if math.isfinite(grad_norm) else None
    )  # torch.nn.utils.clip_grad_norm_'s coefficient
    summary: dict[str, Any] = {
        "kind": kind,
        "step": int(step),
        "grad_norm": _number(grad_norm),
        "loss": None if snapshot.loss is None else _number(snapshot.loss),
        "lr": float(trainer.optimizer.param_groups[0]["lr"]),
        "grad_clip": cfg.train.grad_clip,
        "clip_coef": clip_coef,
        "path": str(path),
    }
    try:
        grads = _gradient_summary(trainer.model)
        model_state = trainer.model.state_dict()
        model_state.update(snapshot.buffers)
        extra = trainer._checkpoint_extra(snapshot.state, snapshot.counters)
        extra["divergence"] = {
            **summary,
            "windows": snapshot.windows,
            "window_docs": [_window_docs(w) for w in snapshot.windows],
            "loader_rows": snapshot.loader_state.get("rows"),
            "config": cfg.model_dump(mode="json"),
            "notes": dict(getattr(trainer, "dump_notes", {}) or {}),
            "git_rev": _git_rev(),
            "torch": str(torch.__version__),
            "device": str(trainer.ctx.device),
            "card": torch.cuda.get_device_name(0) if trainer.ctx.device.type == "cuda" else None,
            "compiled": getattr(trainer, "compiled_window", None) is not None,
            "autocast": str(trainer.ctx.autocast_dtype),
            "gradients": grads,
            "rng_at_dump": capture_rng_state(),  # for the record: the generators after the step
        }
        save_checkpoint(
            path,
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            snapshot.step,
            snapshot.rng_state,
            snapshot.loader_state,
            extra=extra,
            model_state=model_state,
        )
        summary["bytes"] = path.stat().st_size
        side = path.with_suffix(".json")
        side.write_text(
            json.dumps({**summary, "gradients": grads}, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.warning(
            "step %d: %s step; divergence dump written to %s (%d bytes)",
            step,
            kind,
            path,
            summary["bytes"],
        )
    except Exception as exc:  # the dump is a diagnostic: never kill the training run for it
        summary["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("step %d: the divergence dump failed: %s", step, summary["error"])
    trainer.divergence = summary
    return summary


def dump_paths(run_dir: str | Path) -> list[Path]:
    """The dumps in ``run_dir``, by step."""
    directory = Path(run_dir)
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{DUMP_PREFIX}*.pt"))


def load_dump(path: str | Path) -> dict[str, Any]:
    """The dump's payload (a checkpoint dict with ``extra["divergence"]``); raises when the file is
    not a dump."""
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if "divergence" not in (payload.get("extra") or {}):
        raise ValueError(f"{path} is not a divergence dump (no extra['divergence'])")
    return payload


@contextlib.contextmanager
def _deterministic_compile(on: bool) -> Iterator[None]:
    if not on:
        yield
        return
    import torch._inductor.config as inductor_config

    with inductor_config.patch(deterministic=True):
        yield


def _rms_hook(store: list[float]) -> Any:
    def hook(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
        x = args[0].detach().float()
        rms = x.pow(2).mean(dim=-1).sqrt().flatten()
        store.extend(rms.cpu().tolist())

    return hook


def _windows_equal(a: Mapping[str, Tensor], b: Mapping[str, Tensor]) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def replay(
    dump: str | Path,
    *,
    eager: bool | None = None,
    deterministic: bool = True,
    repeats: int = 2,
    cpu: bool = False,
    train_ds: Any = None,
) -> dict[str, Any]:
    """Replay the dumped step (module docstring). ``eager`` None runs as the dump's run did
    (compiled when it was); ``train_ds`` defaults to the configuration's training data (truncated
    to the dump's loader, as the harness's ``n_train`` did)."""
    from hsn.config import RunConfig, load_hardware_profile
    from hsn.data.dataset import ByteDataset
    from hsn.device import resolve, seed_everything, set_deterministic
    from hsn.runner import build_model, load_datasets, make_trainer
    from hsn.train.loop import StreamLoader

    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if deterministic:
        # cuBLAS reads its workspace setting when it first runs; set it before anything does (the
        # trainer's set_deterministic(False) then leaves it alone: it removes only its own)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    path = Path(dump)
    payload = load_dump(path)
    info = payload["extra"]["divergence"]
    notes = info.get("notes") or {}
    variant = notes.get("variant", "pdg")
    if variant not in ("pdg", "none"):
        raise NotImplementedError(f"replay of a {variant} variant: rebuild it by hand")
    cfg = RunConfig.model_validate(info["config"])
    ctx = resolve(load_hardware_profile("cpu") if cpu else cfg.hardware)
    compiled = bool(info.get("compiled")) if eager is None else not eager
    ctx = dataclasses.replace(ctx, compile_enabled=compiled)
    step = int(info["step"])
    loader_state = payload["loader_state"]
    if train_ds is None:
        train_ds, _dev = load_datasets(cfg)
    n_docs = int(loader_state["n_docs"])
    if len(train_ds) < n_docs:
        raise ValueError(f"the dump's loader has {n_docs} documents, the data {len(train_ds)}")
    if len(train_ds) > n_docs:  # the harness's n_train: the first n_docs documents
        train_ds = ByteDataset(list(train_ds.examples[:n_docs]))
    # the windows the loader gives from the dumped position must be the stored ones
    loader = StreamLoader(
        train_ds,
        batch=cfg.train.batch,
        unroll=cfg.train.unroll,
        seed=cfg.seed,
        shuffle=True,
        align_docs=cfg.data.align_docs,
        epochs=None,
    )
    loader.load_state_dict(loader_state)
    stored = info["windows"]
    regenerated = [next(loader) for _ in stored]
    windows_match = all(
        _windows_equal(a, {k: b[k] for k in a}) for a, b in zip(regenerated, stored, strict=True)
    )
    if not windows_match:
        raise ValueError(f"{path}: the training data's windows at the dumped position differ")
    dev = ByteDataset(list(train_ds.examples[:1]))  # the fit's last evaluation: one document
    resume_from = path
    tmp = tempfile.TemporaryDirectory()
    if cpu and payload["rng_state"].get("cuda"):
        stripped = dict(payload)
        stripped["rng_state"] = {**payload["rng_state"], "cuda": []}
        resume_from = Path(tmp.name) / path.name
        torch.save(stripped, resume_from)
    runs: list[dict[str, Any]] = []
    try:
        for r in range(repeats):
            seed_everything(cfg.seed)
            model = build_model(cfg)
            with _deterministic_compile(compiled and deterministic):
                trainer = make_trainer(model, cfg, ctx, run_dir=Path(tmp.name) / f"r{r}")
                trainer.divergence_dump = False
                if deterministic:
                    set_deterministic(True)
                trainer.resume(resume_from)
                raw: dict[str, float | str] = {}

                def grad_hook(t: Any, raw: dict[str, float | str] = raw) -> None:
                    for name, p in t.model.named_parameters():
                        if p.grad is not None:
                            raw[name] = _number(float(torch.linalg.vector_norm(p.grad.float())))

                trainer.grad_hook = grad_hook
                rms: list[float] = []
                handle = None
                cand = getattr(getattr(model, "cell", None), "cand_norm", None)
                if (
                    not compiled
                    and cfg.model.cand_norm != "none"
                    and isinstance(cand, torch.nn.Module)
                ):
                    handle = cand.register_forward_hook(_rms_hook(rms))
                try:
                    history = trainer.fit(train_ds, dev, step)
                finally:
                    if handle is not None:
                        handle.remove()
            rec = history.steps[-1]
            finite = sorted(
                (v, k) for k, v in raw.items() if isinstance(v, float) and math.isfinite(v)
            )
            run: dict[str, Any] = {
                "loss": _number(float(rec.loss)),
                "grad_norm": _number(float(rec.grad_norm)),
                "skipped": [s["reason"] for s in trainer.skipped_steps if s["step"] == step],
                "top_grad_norms": [[k, v] for v, k in reversed(finite[-10:])],
                "nonfinite_params": sorted(k for k, v in raw.items() if isinstance(v, str)),
            }
            if rms:
                ordered = sorted(rms)
                run["cand_norm_input_rms"] = {
                    "min": ordered[0],
                    "p01": ordered[len(ordered) // 100],
                    "median": ordered[len(ordered) // 2],
                    "n": len(ordered),
                }
            runs.append(run)
            torch._dynamo.reset()
    finally:
        tmp.cleanup()
    first = runs[0]
    recorded = info["grad_norm"]
    return {
        "dump": str(path),
        "step": step,
        "kind": info["kind"],
        "recorded": {"loss": info.get("loss"), "grad_norm": recorded},
        "device": str(ctx.device),
        "compiled": compiled,
        "deterministic": deterministic,
        "windows_match": windows_match,
        "replays": runs,
        "replays_identical": all(
            r["loss"] == first["loss"] and r["grad_norm"] == first["grad_norm"] for r in runs
        ),
        "grad_norm_ratio_to_recorded": (
            float(first["grad_norm"]) / float(recorded)
            if isinstance(first["grad_norm"], float) and isinstance(recorded, float) and recorded
            else None
        ),
        "notes": notes,
        "git_rev_dump": info.get("git_rev"),
    }
