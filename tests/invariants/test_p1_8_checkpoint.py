"""P1.8: a checkpoint resumes bit for bit on the CPU through ``TBPTTTrainer``, and the incremental
summary ``q`` of the cell graph state drifts by less than 1e-5 in fp32 over 10,000 sparse updates
with the exact recomputation every ``train.q_resync_every`` windows (the P1.7 cadence)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from hsn.config import RunConfig
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example
from hsn.device import DeviceContext, seed_everything
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from hsn.train.checkpoint import check_q_resync
from hsn.train.tbptt import TBPTTStepRecord, TBPTTTrainer
from torch import Tensor

ConfigFactory = Callable[..., RunConfig]

STEPS_10K = 10_000
DRIFT_BOUND = 1e-5
FIELDS = ("h", "q", "age", "pos")


def _assert_tensors_equal(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert list(a) == list(b)
    for key, value in a.items():
        other = b[key]
        if isinstance(value, Tensor):
            assert torch.equal(value, other), key
        else:
            assert value == other, key


def _record_tuple(record: TBPTTStepRecord) -> tuple[Any, ...]:
    return (
        record.step,
        record.loss,
        record.accuracy,
        record.lr,
        record.grad_norm,
        record.tokens,
        record.loss_positions,
        record.total,
        record.parts,
        record.stats,
        record.windows,
        record.resyncs,
    )


# --------------------------------------------------------------------------------------------------
# save / load / resume reproduces the next 20 losses exactly
# --------------------------------------------------------------------------------------------------


def test_p1_8_checkpoint_exact(
    make_trainer_config: ConfigFactory,
    cpu_ctx: DeviceContext,
    train_docs: list[Example],
    dev_docs: list[Example],
    tmp_path: Path,
) -> None:
    """Run A trains 30 steps with a checkpoint every 10. Run B starts from a differently seeded
    model, resumes the step-10 checkpoint and trains to step 30: its 20 records (loss, objective
    and parts, accuracy, gradient norms, learning rate, resync counters) equal run A's bit for bit,
    and so do the final weights, optimiser moments, carried cell state and the two step-30
    checkpoint files."""
    cfg = make_trainer_config(
        "train.steps=40",
        "train.eval_every=40",
        "train.ckpt_every=10",
        "train.stats_every=10",
        "train.q_resync_every=8",
        "train.deterministic=true",
        "train.warmup_steps=4",
        "train.schedule=cosine",
    )
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)
    run_a, run_b = tmp_path / "a", tmp_path / "b"

    seed_everything(cfg.seed)
    model_a = HSN.from_config(cfg)
    trainer_a = TBPTTTrainer(model_a, cfg, cpu_ctx, run_dir=run_a)
    history_a = trainer_a.fit(train_ds, dev_ds, 30)
    assert [Path(p).name for p in history_a.checkpoints] == [
        "step_00000010.pt",
        "step_00000020.pt",
        "step_00000030.pt",
    ]
    assert (trainer_a.windows, trainer_a.windows_since_resync, trainer_a.resyncs) == (30, 6, 3)

    seed_everything(cfg.seed + 1)
    model_b = HSN.from_config(cfg)  # other initial weights: everything must come from the file
    trainer_b = TBPTTTrainer(model_b, cfg, cpu_ctx, run_dir=run_b)
    assert trainer_b.resume(run_a / "step_00000010.pt") == 10
    assert (trainer_b.windows, trainer_b.windows_since_resync, trainer_b.resyncs) == (10, 2, 1)
    history_b = trainer_b.fit(train_ds, dev_ds, 30)

    records_a, records_b = history_a.steps[10:], history_b.steps
    assert [r.step for r in records_b] == list(range(11, 31)) and len(records_a) == 20
    losses_a = [r.loss for r in records_a]
    losses_b = [r.loss for r in records_b]
    assert losses_a == losses_b, list(zip(losses_a, losses_b, strict=True))
    for rec_a, rec_b in zip(records_a, records_b, strict=True):
        assert isinstance(rec_a, TBPTTStepRecord) and isinstance(rec_b, TBPTTStepRecord)
        assert _record_tuple(rec_a) == _record_tuple(rec_b), (rec_a, rec_b)
    assert history_a.evals == history_b.evals and history_a.evals[-1]["step"] == 30
    assert (trainer_b.windows, trainer_b.windows_since_resync, trainer_b.resyncs) == (30, 6, 3)

    _assert_tensors_equal(model_a.state_dict(), model_b.state_dict())
    opt_a, opt_b = trainer_a.optimizer.state_dict(), trainer_b.optimizer.state_dict()
    assert list(opt_a["state"]) == list(opt_b["state"])
    for index, entry in opt_a["state"].items():
        _assert_tensors_equal(entry, opt_b["state"][index])
    assert trainer_a.scheduler.state_dict()["last_epoch"] == 30
    assert (
        trainer_a.scheduler.state_dict()["last_epoch"]
        == trainer_b.scheduler.state_dict()["last_epoch"]
    )
    carried_a, carried_b = trainer_a._carried_state, trainer_b._carried_state
    assert isinstance(carried_a, CellGraphState) and isinstance(carried_b, CellGraphState)
    for field in FIELDS:
        assert torch.equal(getattr(carried_a, field), getattr(carried_b, field)), field

    file_a = torch.load(run_a / "step_00000030.pt", weights_only=True)
    file_b = torch.load(run_b / "step_00000030.pt", weights_only=True)
    _assert_tensors_equal(file_a["model"], file_b["model"])
    assert torch.equal(file_a["rng_state"]["torch"], file_b["rng_state"]["torch"])
    assert file_a["loader_state"] == file_b["loader_state"]
    _assert_tensors_equal(file_a["extra"]["carried_state"], file_b["extra"]["carried_state"])
    assert file_a["extra"]["q_resync"] == file_b["extra"]["q_resync"]
    assert file_a["extra"]["q_resync"] == {"windows": 30, "since_resync": 6, "resyncs": 3}
    print(f"checkpoint resume: 20 losses exact, first {losses_b[0]:.6f}, last {losses_b[-1]:.6f}")


# --------------------------------------------------------------------------------------------------
# |q − recompute_q| < 1e-5 after 10,000 steps with a resync every 64 windows
# --------------------------------------------------------------------------------------------------


def _drift(state: CellGraphState) -> float:
    return float((state.q - state.recompute_q()).abs().max())


def test_p1_8_incremental_q_drift_bounded(
    device: torch.device,
    ref_cfg: RunConfig,
    tiny_cfg: RunConfig,
    make_trainer_config: ConfigFactory,
    cpu_ctx: DeviceContext,
    train_docs: list[Example],
    dev_docs: list[Example],
    tmp_path: Path,
) -> None:
    """(1) The incremental path at the reference shapes (``B=4, N=32, d=64, K=4``, fp32): 10,000
    ``with_delta`` updates of distinct random cells with candidates in ``[−1, 1]`` (the tanh range),
    the summary recomputed exactly every ``train.q_resync_every`` (64) windows of 8 positions as the
    P1.7 trainer does at a window start; the drift ``|q − recompute_q|`` sampled just before every
    resync and at the end stays below 1e-5. The dense step of P1.5 writes the exact mean, so the
    bound is exercised on the P1.3 state path that P2.5's sparse step will use. A never-resynced
    twin must stay below the bound too: ``ref_a1``'s own cadence (64 windows of 256 positions)
    resyncs only after 16,384 positions, so within 10,000 steps it never does. (2) The trainer's
    cadence (the P1.7 counters):
    a 10-step run with ``q_resync_every=4`` counts 10 windows, 2 resyncs, 2 windows since, the
    checkpoint carries the counters, and the dense carried state's ``q`` is the exact mean."""
    m = ref_cfg.model
    B, N, d, K = ref_cfg.train.batch, m.n_cells, m.d, m.k_select
    window = tiny_cfg.train.unroll  # 8 positions
    every = ref_cfg.train.q_resync_every  # 64 windows
    assert (B, N, d, K, window, every) == (4, 32, 64, 4, 8, 64)
    generator = torch.Generator(device="cpu").manual_seed(1)
    h0 = (torch.rand(B, N, d, generator=generator) * 2 - 1).to(device)
    state = CellGraphState.zeros(B, N, d, device, torch.float32).with_dense(h0)
    unsynced = state
    before_resync: list[float] = []
    drift_unsynced = 0.0
    resyncs = 0
    for s in range(1, STEPS_10K + 1):
        idx = torch.rand(B, N, generator=generator).argsort(dim=1)[:, :K].to(device)
        h_new = (torch.rand(B, K, d, generator=generator) * 2 - 1).to(device)
        state = state.with_delta(idx, h_new)
        unsynced = unsynced.with_delta(idx, h_new)
        if s % window == 0 and (s // window) % every == 0:  # the next window starts with a resync
            before_resync.append(_drift(state))
            drift_unsynced = max(drift_unsynced, _drift(unsynced))
            state = replace(state, q=state.recompute_q())
            resyncs += 1
    final = _drift(state)
    drift_unsynced = max(drift_unsynced, _drift(unsynced))
    worst = max([*before_resync, final])
    assert resyncs == STEPS_10K // (window * every) == 19
    assert len(before_resync) == resyncs
    assert worst < DRIFT_BOUND, (worst, before_resync)
    assert drift_unsynced < DRIFT_BOUND, drift_unsynced  # ref_a1's cadence: no resync in 10k steps
    assert torch.equal(state.h, unsynced.h)  # the resync touches q only
    assert state.q.dtype is torch.float32
    print(
        f"q drift on {device.type}: max {worst:.3e} with a resync every {every} windows of "
        f"{window} ({resyncs} resyncs), {drift_unsynced:.3e} without, {STEPS_10K} steps"
    )

    if device.type != "cpu":
        return
    cfg = make_trainer_config(
        "train.steps=10",
        "train.eval_every=10",
        "train.ckpt_every=10",
        "train.stats_every=10",
        "train.q_resync_every=4",
    )
    assert cfg.train.grad_accum == 1  # one window per step
    seed_everything(cfg.seed)
    model = HSN.from_config(cfg)
    trainer = TBPTTTrainer(model, cfg, cpu_ctx, run_dir=tmp_path)
    history = trainer.fit(ByteDataset(train_docs), ByteDataset(dev_docs), 10)
    assert (trainer.windows, trainer.resyncs, trainer.windows_since_resync) == (10, 2, 2)
    assert [r.resyncs for r in history.steps] == [n // 4 for n in range(1, 11)]
    assert [r.windows for r in history.steps] == list(range(1, 11))
    assert history.measured["windows"] == 10 and history.measured["resyncs"] == 2
    payload = torch.load(tmp_path / "step_00000010.pt", weights_only=True)
    counters = check_q_resync(payload["extra"]["q_resync"], "step_00000010.pt")
    assert counters == {"windows": 10, "since_resync": 2, "resyncs": 2}
    carried = trainer._carried_state
    assert isinstance(carried, CellGraphState)
    assert torch.equal(carried.q, carried.recompute_q())  # dense mode: the exact mean, no drift
