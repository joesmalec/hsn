"""P1.13: with every new option at its default, training is today's training, bit for bit.

The golden runs (``tests/fixtures/p1_13/``, captured by :mod:`p1_13_support` on the base commit
``8eb2f01`` before any P1.13 library change) record per optimiser step the loss, the pre-clip
gradient norm, the learning rate and digests of every gradient and parameter, then the final
weights, optimiser state and carried recurrent state, for a dense HSN (AdamW), a top-K HSN with
straight-through gates (sparse round, lazy AdamW) and the GRU baseline on the CPU, and (cuda
marker) a dense HSN in P2.8's compiled window under bf16 autocast. On the platform a fixture was
captured on (torch version, CPU capability, card) every value and digest must match exactly;
elsewhere (another CPU, CI) the learning rates must still match exactly (pure Python) and the
floats to 1e-4, and the digests are not compared (BLAS kernels differ by instruction set).

The structural tests hold on every platform: the default optimiser groups are the pre-P1.13
construction, the default schedules are the pre-P1.13 formulas, and the state-dict layout of the
reference configuration (dense and top-K, P2.3's scorer buffers included) is the base commit's.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import load_config, load_hardware_profile
from hsn.device import resolve, seed_everything
from hsn.runner import build_model, make_trainer
from hsn.train.lazy_adam import LazyAdamW, build_lazy_adamw
from p1_13_support import CASES, FIXTURES, REF_A1, case_config, fingerprint, run_case
from p1_13_support import state_dict_layout as layout_of

CPU_FIXTURE = FIXTURES / "defaults_cpu.json"
CUDA_FIXTURE = FIXTURES / "defaults_cuda.json"
RTOL = 1e-4  # off the capture platform: the floats of six fp32 steps of a small model


def _fixture(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: capture it with tests/p1_13_support.py")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _compare(name: str, got: dict[str, Any], want: dict[str, Any], *, bitwise: bool) -> None:
    got = json.loads(json.dumps(got))  # the fixture's JSON types (tuples become lists)
    assert len(got["steps"]) == len(want["steps"]), name
    for g, w in zip(got["steps"], want["steps"], strict=True):
        assert g["step"] == w["step"], (name, g["step"])
        assert g["lr"] == w["lr"], (name, g["step"], "lr")  # pure Python: exact everywhere
        for key in ("loss", "grad_norm"):
            gv, wv = float.fromhex(g[key]), float.fromhex(w[key])
            if bitwise:
                assert g[key] == w[key], (name, g["step"], key, gv, wv)
            else:
                assert math.isclose(gv, wv, rel_tol=RTOL), (name, g["step"], key, gv, wv)
        if bitwise:
            assert g["grads"] == w["grads"], (name, g["step"], "gradients")
            assert g["params"] == w["params"], (name, g["step"], "parameters")
    assert got["last_lr"] == want["last_lr"], name
    assert got["groups"] == want["groups"], name  # the optimiser's groups and hyperparameters
    if bitwise:
        for key in ("model", "optimizer", "carried_checkpoint", "carried"):
            assert got[key] == want[key], (name, key)


@pytest.mark.parametrize("name", [n for n, spec in CASES.items() if spec[2] == "cpu"])
def test_p1_13_defaults_bitwise_cpu(name: str) -> None:
    """Six steps of the default configuration reproduce the base commit's run."""
    fixture = _fixture(CPU_FIXTURE)
    assert fixture["reproducible"], "the CPU fixture was not reproducible at capture"
    bitwise = fixture["fingerprint"] == fingerprint("cpu")
    _compare(name, run_case(name), fixture["cases"][name], bitwise=bitwise)


@pytest.mark.cuda
def test_p1_13_defaults_bitwise_cuda() -> None:
    """The compiled dense HSN under bf16 on CUDA reproduces the base commit's run (bitwise when
    the capture was reproducible run to run on this card; otherwise to the tolerance)."""
    from cards import cuda_profile

    fixture = _fixture(CUDA_FIXTURE)
    profile = cuda_profile()
    got = run_case("hsn_dense_compiled_cuda", profile)
    same_platform = fixture["fingerprint"] == fingerprint("cuda")
    bitwise = same_platform and bool(fixture["reproducible"])
    _compare(
        "hsn_dense_compiled_cuda", got, fixture["cases"]["hsn_dense_compiled_cuda"], bitwise=bitwise
    )


@pytest.mark.parametrize(
    ("label", "overrides"),
    [("ref_a1_dense", []), ("ref_a1_topk", ["model.routing=topk", "train.gate_mode=ste"])],
)
def test_p1_13_default_state_dict_layout(label: str, overrides: list[str]) -> None:
    """The reference model's state-dict keys, shapes and dtypes are the base commit's (no
    ``cell.cand_norm.*`` key by default; P2.3's ``ema_utility`` / ``ema_selected`` kept)."""
    cfg = load_config(REF_A1, overrides=overrides)
    seed_everything(cfg.seed)
    got = layout_of(build_model(cfg))
    want = _fixture(CPU_FIXTURE)["state_dict_layouts"][label]
    assert got == want
    keys = {row[0] for row in got}
    assert {"router.scorer.ema_utility", "router.scorer.ema_selected"} <= keys
    assert not any(k.startswith("cell.cand_norm") for k in keys)


def _pre_p1_13_factor(warmup: int, total: int, schedule: str, step: int) -> float:
    """``Trainer._make_scheduler``'s factor as it was at ``8eb2f01``."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "linear":
        return 1.0 - progress
    return 1.0


@pytest.mark.parametrize("schedule", ["constant", "cosine", "linear"])
@pytest.mark.parametrize(("warmup", "total"), [(0, 50), (7, 50), (200, 3000)])
def test_p1_13_default_schedules_unchanged(schedule: str, warmup: int, total: int) -> None:
    """The existing schedules' factors equal the pre-P1.13 formula at every step, bit for bit,
    and every group follows them."""
    cfg = case_config(
        "hsn_dense",
        extra=[
            f"train.schedule={schedule}",
            f"train.warmup_steps={warmup}",
            f"train.steps={total}",
        ],
    )
    seed_everything(cfg.seed)
    trainer = make_trainer(build_model(cfg), cfg, resolve(cfg.hardware))
    lr = cfg.train.lr
    for step in range(total + 2):
        want = lr * _pre_p1_13_factor(warmup, total, schedule, step)
        assert [g["lr"] for g in trainer.optimizer.param_groups] == [want] * len(
            trainer.optimizer.param_groups
        ), step
        trainer.optimizer.step()  # no gradients: a no-op that keeps the scheduler's order check
        trainer.scheduler.step()


def _pre_p1_13_adamw_groups(model: torch.nn.Module, weight_decay: float) -> list[list[int]]:
    decay = [id(p) for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [id(p) for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return [ids for ids in (decay, no_decay) if ids]


@pytest.mark.parametrize("name", ["hsn_dense", "hsn_topk", "gru"])
def test_p1_13_default_optimizer_groups_unchanged(name: str) -> None:
    """With the defaults, the trainer's optimiser has the pre-P1.13 groups: AdamW with decay on
    the tensors of ``ndim >= 2`` and none on the rest; lazy AdamW as ``build_lazy_adamw`` built
    them at ``8eb2f01``. Same tensors, same order, same hyperparameters."""
    cfg = case_config(name)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    trainer = make_trainer(model, cfg, resolve(load_hardware_profile("cpu")))
    train = cfg.train
    groups = trainer.optimizer.param_groups
    if isinstance(trainer.optimizer, LazyAdamW):
        reference = build_lazy_adamw(
            model, lr=train.lr, betas=train.betas, weight_decay=train.weight_decay
        ).param_groups
        assert [[id(p) for p in g["params"]] for g in groups] == [
            [id(p) for p in g["params"]] for g in reference
        ]
        for g, r in zip(groups, reference, strict=True):
            # the scheduler set ``lr`` to the warm-up's first value; ``initial_lr`` is the base
            assert g["initial_lr"] == r["lr"] == train.lr
            assert {k: v for k, v in g.items() if k not in ("params", "initial_lr", "lr")} == {
                k: v for k, v in r.items() if k not in ("params", "lr")
            }
    else:
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert [[id(p) for p in g["params"]] for g in groups] == _pre_p1_13_adamw_groups(
            model, train.weight_decay
        )
        assert [g["weight_decay"] for g in groups] == [train.weight_decay, 0.0]
        assert all(g["initial_lr"] == train.lr and g["betas"] == train.betas for g in groups)
