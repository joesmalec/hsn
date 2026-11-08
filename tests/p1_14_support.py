"""Shared helpers of the P1.14 tests (the sweep-harness follow-ups): tiny specs on the P1.11 tiny
data, running one claimed experiment's child in-process, and P1.12's planted divergence.

Not a test module (no ``test_`` prefix); the P1.14 test files import it from ``tests/``.
"""

from __future__ import annotations

import math
import signal
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.experiments import db as dbm
from hsn.experiments import machine
from hsn.experiments import run_arm as run_arm_module
from hsn.train import tbptt
from p1_11_support import REF_A1, TINY_HSN_OVERRIDES, disk_overrides, enqueue_spec, write_yaml
from torch import Tensor


def tiny_reads_state_spec(
    root: Path,
    manifest: Path,
    *,
    name: str = "p114",
    steps: int = 8,
    eval_every: int = 2,
    protocol: dict[str, Any] | None = None,
    rules: dict[str, Any] | None = None,
    arms: list[dict[str, Any]] | None = None,
) -> Path:
    """A spec of one tiny HSN arm (``profile:cpu``) on the reads-state protocol on the tiny data."""
    raw = {
        "name": name,
        "base": {
            "config": str(REF_A1),
            "overrides": [
                *disk_overrides(manifest, steps=steps, eval_every=eval_every),
                *TINY_HSN_OVERRIDES,
            ],
        },
        "protocol": {
            "name": "reads_state",
            "diag_every": 2,
            "diag_docs": 8,
            "probe_batch": 8,
            **(protocol or {}),
        },
        "rules": rules,
        "arms": arms or [{"name": "a", "family": "hsn"}],
    }
    return write_yaml(root / f"{name}.yaml", raw)


def claim(db: dbm.Database, exp_id: int) -> None:
    """Mark ``exp_id`` running for this host (as the worker's claim does), without a PID yet."""
    row = dbm.claim_next(db, lambda q, r: exp_id, worker="test", host=machine.host())
    assert row is not None and int(row["exp_id"]) == exp_id


def run_child(db: dbm.Database, exp_id: int, root: Path) -> int:
    """The worker's child for a claimed experiment, in this process (so that a monkeypatch reaches
    it); the SIGTERM handler it installs is restored afterwards."""
    old = signal.getsignal(signal.SIGTERM)
    try:
        return run_arm_module.main(str(db.path), exp_id, str(root))
    finally:
        signal.signal(signal.SIGTERM, old)


def enqueue_and_claim(db: dbm.Database, spec: Path) -> list[int]:
    ids = [int(a["exp_id"]) for a in enqueue_spec(db, spec).added]
    for exp_id in ids:
        claim(db, exp_id)
    return ids


def plant_inf_gradient(monkeypatch: pytest.MonkeyPatch, from_call: int) -> None:
    """P1.12's demo pattern: from the ``from_call``-th call of the TBPTT trainer's objective
    assembly on (eager, one window per step: step ``from_call``), the logits' gradient is
    infinite, so the global gradient norm is not finite and P1.12's guard skips the step."""
    real = tbptt.assemble_terms
    calls = [0]

    def inf(grad: Tensor) -> Tensor:
        return torch.full_like(grad, math.inf)

    def planted(out: Any, batch: Any, cfg: Any, extras: Any = None) -> dict[str, Tensor]:
        calls[0] += 1
        if calls[0] >= from_call:
            logits = out["logits"] * 1.0
            logits.register_hook(inf)
            out = {**out, "logits": logits}
        result: dict[str, Tensor] = real(out, batch, cfg, extras)
        return result

    monkeypatch.setattr(tbptt, "assemble_terms", planted)
