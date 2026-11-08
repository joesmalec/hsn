"""Shared helpers of the P1.15 tests (spike guard, divergence dump): tiny kv data, tiny HSN and
GRU configurations, plants of a finite gradient spike or of P1.12's poisons in a chosen step, and a
probe of what each step left behind.

Not a test module (no ``test_`` prefix); the P1.15 test files import it from ``tests/``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import RunConfig, load_config
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example, TaskSpec, generate, load_dataset_spec
from hsn.device import resolve, seed_everything
from hsn.losses import assemble_terms
from hsn.model.hsn import HSN
from hsn.train import tbptt
from hsn.train.loop import Trainer
from hsn.train.tbptt import TBPTTTrainer
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
GRU_2M = REPO / "configs" / "baselines" / "gru_2m.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "p1_15"
UNROLL = 32
TINY = (
    "model.d=8",
    "model.context_dim=8",
    "model.n_cells=4",
    "model.in_degree=2",
    "model.n_units=2",
    "model.k_select=2",
    "model.hidden=16",
    "model.scorer_hidden=8",
)
#: the objective's task term is multiplied by this at a planted spike: a finite gradient about
#: 1e8 times the usual one, far above any factor the tests use (exp 171's jump was 7e10 times)
SPIKE_SCALE = 1e8
SPIKE = "spike"
NAN_OBJECTIVE = "nan_objective"
INF_GRADIENT = "inf_gradient"
#: the tests' guard: factor 100 (the recommendation) over a window of 4 applied steps
GUARD = ("train.spike_factor=100", "train.spike_window=4")


def _task() -> TaskSpec:
    task = SPEC.task("kv", "in_range")
    return TaskSpec.model_validate(
        {
            **task.model_dump(),
            "n_train": 32,
            "n_dev": 8,
            "n_test": 8,
            "length_range": [16, UNROLL],
            "dep_len_range": [4, 20],
            "unroll": UNROLL,
        }
    )


def datasets() -> tuple[ByteDataset, ByteDataset]:
    def docs(split: str, n: int) -> list[Example]:
        return [generate(_task(), split, i) for i in range(n)]  # type: ignore[arg-type]

    return ByteDataset(docs("train", 32)), ByteDataset(docs("dev", 8))


def train_overrides(steps: int) -> list[str]:
    """Evaluation, checkpoint and statistics at the last step only (P1.12's tests' recipe)."""
    return [
        "profile=cpu",
        f"train.unroll={UNROLL}",
        "train.batch=4",
        "train.micro_batch=4",
        f"train.steps={steps}",
        "train.eval_every=1000",
        "train.ckpt_every=1000",
        "train.stats_every=1000",
        "train.warmup_steps=0",
        "train.schedule=linear",
        "train.deterministic=true",
    ]


def hsn_config(steps: int, *overrides: str) -> RunConfig:
    return load_config(REF_A1, overrides=[*TINY, *train_overrides(steps), *overrides])


def gru_config(steps: int, *overrides: str) -> RunConfig:
    return load_config(
        GRU_2M,
        overrides=[
            "baseline.gru.d_model=16",
            "baseline.gru.n_layers=1",
            *train_overrides(steps),
            *overrides,
        ],
    )


def hsn_trainer(cfg: RunConfig, run_dir: Path, ctx: Any = None) -> TBPTTTrainer:
    seed_everything(0)
    return TBPTTTrainer(
        HSN.from_config(cfg),
        cfg,
        ctx if ctx is not None else resolve(cfg.hardware),
        run_dir=run_dir,
    )


def gru_trainer(cfg: RunConfig, run_dir: Path) -> Trainer:
    from hsn.runner import build_model

    seed_everything(0)
    return Trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=run_dir)


def _inf(grad: Tensor) -> Tensor:
    return torch.full_like(grad, math.inf)


def poisoned_terms(
    kind: str | None,
    out: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    cfg: RunConfig,
    extras: Mapping[str, Tensor] | None,
) -> dict[str, Tensor]:
    """:func:`hsn.losses.assemble_terms` of one window with ``kind`` planted: ``spike`` scales the
    task term by :data:`SPIKE_SCALE` (a finite jump of the gradient norm), ``nan_objective`` and
    ``inf_gradient`` are P1.12's poisons, ``None`` leaves it alone."""
    if kind == INF_GRADIENT:
        logits = out["logits"] * 1.0
        logits.register_hook(_inf)
        return assemble_terms({**out, "logits": logits}, batch, cfg, extras)
    terms = assemble_terms(out, batch, cfg, extras)
    if kind == SPIKE:
        return {**terms, "task": terms["task"] * SPIKE_SCALE}
    if kind == NAN_OBJECTIVE:
        return {**terms, "task": terms["task"] * math.nan}
    assert kind is None, kind
    return terms


def plant(monkeypatch: pytest.MonkeyPatch, trainer: Trainer, plan: Mapping[int, str]) -> None:
    """Plant ``plan`` (step -> kind) in the TBPTT trainer's objective (the windows of step n run
    while ``trainer.step == n - 1``); assembled eagerly, so it reaches compiled blocks too."""

    def planted(
        out: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
        cfg: RunConfig,
        extras: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        return poisoned_terms(plan.get(trainer.step + 1), out, batch, cfg, extras)

    monkeypatch.setattr(tbptt, "assemble_terms", planted)


def plant_by_call(monkeypatch: pytest.MonkeyPatch, plan: Mapping[int, str]) -> None:
    """:func:`plant` for a trainer built out of reach (a protocol builds its own): the n-th call of
    the assembly is step n's window (eager, one window per step)."""
    calls = [0]

    def planted(
        out: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
        cfg: RunConfig,
        extras: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        calls[0] += 1
        return poisoned_terms(plan.get(calls[0]), out, batch, cfg, extras)

    monkeypatch.setattr(tbptt, "assemble_terms", planted)


def plant_gru_spike(trainer: Trainer, steps: set[int]) -> Any:
    """A finite spike in the generic trainer's step ``n`` for ``n`` in ``steps``: every
    parameter's gradient is scaled by :data:`SPIKE_SCALE`. Returns the hook handles."""

    def hook(grad: Tensor) -> Tensor:
        return grad * SPIKE_SCALE if trainer.step + 1 in steps else grad

    return [p.register_hook(hook) for p in trainer.model.parameters() if p.requires_grad]


class StepProbe:
    """What each step left behind, read when the optimiser decision has been taken (at the
    scheduler's step, wrapped on its class for this trainer's scheduler only): a CPU copy of every
    parameter, whether every gradient is ``None`` and the optimiser steps so far."""

    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer
        self.params: dict[int, list[Tensor]] = {}
        self.grads_none: dict[int, bool] = {}
        self.optimizer_steps: dict[int, int] = {}
        self._optimizer_steps = 0
        self._cls = type(trainer.scheduler)
        self._own = self._cls.__dict__.get("step")

    def __enter__(self) -> StepProbe:
        original = self._cls.step
        probe, scheduler = self, self.trainer.scheduler

        def step(sched: Any, *args: Any, **kwargs: Any) -> None:
            if sched is scheduler:
                n = probe.trainer.step + 1
                params = list(probe.trainer.model.parameters())
                probe.params[n] = [p.detach().cpu().clone() for p in params]
                probe.grads_none[n] = all(p.grad is None for p in params)
                probe.optimizer_steps[n] = probe._optimizer_steps
            original(sched, *args, **kwargs)

        self._cls.step = step  # type: ignore[method-assign]
        self._hook = self.trainer.optimizer.register_step_pre_hook(self._count)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._hook.remove()
        if self._own is None:
            del self._cls.step
        else:
            self._cls.step = self._own  # type: ignore[method-assign]

    def _count(self, *_: Any) -> None:
        self._optimizer_steps += 1

    def unchanged(self, step: int) -> bool:
        before, after = self.params[step - 1], self.params[step]
        return all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
