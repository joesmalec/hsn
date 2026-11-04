"""Checkpoints (P0.5): model, optimiser, scheduler, step, RNG and data-loader state in one file.

A checkpoint is one ``torch.save`` file holding a plain dict (format :data:`CHECKPOINT_FORMAT`):
``model`` / ``optimizer`` / ``scheduler`` state dicts, the global ``step``, ``rng_state`` (every
generator :func:`capture_rng_state` knows: Python, NumPy, torch CPU, every CUDA device, MPS), the
``loader_state`` the trainer's stream loader needs to resume at the same window, the ``hsn``
version and an ``extra`` mapping for anything else the trainer wants back (the P0.5 trainer stores
its loss scaler and the config hash; P1.7 adds its ``q`` resync counter). Everything is plain
Python containers and tensors so that :func:`load_checkpoint` can use ``weights_only=True``:
NumPy and Python RNG states are converted to lists of integers for that reason.

Files are written to a temporary name and renamed into place, so a crash mid-write never leaves a
truncated checkpoint behind. Loading applies ``state_dict`` with ``strict=True`` (a model that
does not match its checkpoint raises; plan §4.3 errors over silence) and returns the dict.
"""

from __future__ import annotations

import os
import random
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

import hsn

__all__ = [
    "CHECKPOINT_FORMAT",
    "capture_rng_state",
    "checkpoint_path",
    "latest_checkpoint",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
]

CHECKPOINT_FORMAT = 1
_FILE_RE = re.compile(r"^step_(\d{8})\.pt$")


# --------------------------------------------------------------------------------------------------
# RNG state
# --------------------------------------------------------------------------------------------------


def capture_rng_state() -> dict[str, Any]:
    """The state of every random generator a run uses, as plain containers and tensors."""
    version, keys, gauss_next = random.getstate()
    np_name, np_keys, np_pos, np_has_gauss, np_gauss = np.random.get_state()
    state: dict[str, Any] = {
        "python": [int(version), [int(k) for k in keys], gauss_next],
        "numpy": [
            str(np_name),
            [int(k) for k in np.asarray(np_keys).tolist()],
            int(np_pos),
            int(np_has_gauss),
            float(np_gauss),
        ],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore what :func:`capture_rng_state` captured; a missing generator raises."""
    missing = [key for key in ("python", "numpy", "torch", "cuda", "mps") if key not in state]
    if missing:
        raise KeyError(f"rng_state is missing {missing}")
    version, keys, gauss_next = state["python"]
    random.setstate((int(version), tuple(int(k) for k in keys), gauss_next))
    np_name, np_keys, np_pos, np_has_gauss, np_gauss = state["numpy"]
    np.random.set_state(
        (
            str(np_name),
            np.asarray(np_keys, dtype=np.uint32),
            int(np_pos),
            int(np_has_gauss),
            np_gauss,
        )
    )
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    cuda_states = list(state["cuda"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"checkpoint holds RNG state for {len(cuda_states)} CUDA devices but CUDA is "
                "not available here"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                f"checkpoint holds RNG state for {len(cuda_states)} CUDA devices, this machine "
                f"has {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(s, dtype=torch.uint8).cpu() for s in cuda_states]
        )
    if state["mps"] is not None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("checkpoint holds MPS RNG state but MPS is not available here")
        torch.mps.set_rng_state(torch.as_tensor(state["mps"], dtype=torch.uint8).cpu())


# --------------------------------------------------------------------------------------------------
# Save and load
# --------------------------------------------------------------------------------------------------


def checkpoint_path(run_dir: str | Path, step: int) -> Path:
    """``<run_dir>/step_<step:08d>.pt``."""
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    return Path(run_dir) / f"step_{step:08d}.pt"


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    """The checkpoint with the largest step in ``run_dir``, or ``None`` when there is none."""
    directory = Path(run_dir)
    if not directory.is_dir():
        return None
    found: list[tuple[int, Path]] = []
    for entry in directory.iterdir():
        match = _FILE_RE.match(entry.name)
        if match and entry.is_file():
            found.append((int(match.group(1)), entry))
    if not found:
        return None
    return max(found)[1]


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    rng_state: Mapping[str, Any],
    loader_state: Mapping[str, Any] | None,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write one checkpoint file atomically (see the module docstring for its contents)."""
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "hsn_version": hsn.__version__,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": dict(rng_state),
        "loader_state": dict(loader_state) if loader_state is not None else None,
        "extra": dict(extra) if extra is not None else {},
    }
    tmp = out.with_name(out.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    """Load a checkpoint into ``model`` (and the optimiser and scheduler when given); return it.

    The returned dict holds everything :func:`save_checkpoint` wrote (``step``, ``rng_state``,
    ``loader_state``, ``extra``, ...). RNG state is *not* restored here, since evaluation-only
    callers must not touch the generators; the trainer calls :func:`restore_rng_state` itself.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"checkpoint {file} does not exist")
    loaded = torch.load(file, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"{file}: not a checkpoint (top level is {type(loaded).__name__})")
    fmt = loaded.get("format")
    if fmt != CHECKPOINT_FORMAT:
        raise ValueError(f"{file}: checkpoint format {fmt!r} != {CHECKPOINT_FORMAT}")
    model.load_state_dict(loaded["model"], strict=True)
    if optimizer is not None:
        if loaded["optimizer"] is None:
            raise ValueError(f"{file}: checkpoint has no optimizer state to load")
        optimizer.load_state_dict(loaded["optimizer"])
    if scheduler is not None:
        if loaded["scheduler"] is None:
            raise ValueError(f"{file}: checkpoint has no scheduler state to load")
        scheduler.load_state_dict(loaded["scheduler"])
    return dict(loaded)
