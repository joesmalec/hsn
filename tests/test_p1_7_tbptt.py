"""P1.7 tests: the TBPTT trainer (detached carried state, document resets, periodic ``q`` resync,
loss assembly with the one-at-a-time rule, logged statistics, the extended checkpoint and a
bit-exact resume) and the GPU-marked AMP run."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hsn.config import RunConfig, load_config, load_hardware_profile
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example, TaskSpec, generate, load_dataset_spec
from hsn.device import DeviceContext, resolve, seed_everything
from hsn.losses import TERMS, assemble_loss, prediction_terms
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from hsn.train.amp import autocast_name
from hsn.train.checkpoint import (
    check_q_resync,
    latest_checkpoint,
    pack_cell_state,
    unpack_cell_state,
)
from hsn.train.loop import SequenceModel, StreamLoader, Trainer
from hsn.train.tbptt import STAT_KEYS, TBPTTStepRecord, TBPTTTrainer
from torch import Tensor, nn

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
UNROLL = 32  # the tiny window every test uses
N_DOCS = 64
GROUPS_WITH_GRADS = {
    "byte_emb",
    "ctx_emb",
    "encoder",
    "predictor",
    "shared",
    "cell",
    "scorer",
    "decoder",
}  # every top-level submodule but the halting head, which is not called while r_max = 1

cuda_param = pytest.param("rtx3060", marks=pytest.mark.cuda)
mps_param = pytest.param("m_series", marks=pytest.mark.mps)


# --------------------------------------------------------------------------------------------------
# Fixtures: kv in-range documents (the P0.5 fixture recipe) and a small HSN from ref_a1
# --------------------------------------------------------------------------------------------------


def _tiny_task(
    length_range: tuple[int, int] = (16, UNROLL), dep_range: tuple[int, int] = (4, 20)
) -> TaskSpec:
    """The committed kv in-range task with short documents (kv needs dep + 9 symbols)."""
    task = SPEC.task("kv", "in_range")
    return TaskSpec.model_validate(
        {
            **task.model_dump(),
            "n_train": 64,
            "n_dev": 16,
            "n_test": 16,
            "length_range": list(length_range),
            "dep_len_range": list(dep_range),
            "unroll": UNROLL,
        }
    )


def _docs(task: TaskSpec, split: str, n: int) -> list[Example]:
    return [generate(task, split, i) for i in range(n)]  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def train_docs() -> list[Example]:
    return _docs(_tiny_task(), "train", N_DOCS)


@pytest.fixture(scope="module")
def dev_docs() -> list[Example]:
    return _docs(_tiny_task(), "dev", 16)


def _config(*overrides: str, batch: int = 4) -> RunConfig:
    return load_config(
        REF_A1,
        overrides=[
            "model.d=8",
            "model.context_dim=8",
            "model.n_cells=4",
            "model.in_degree=2",
            "model.n_units=2",
            "model.k_select=2",
            "model.hidden=16",
            "model.scorer_hidden=8",
            f"train.unroll={UNROLL}",
            f"train.batch={batch}",
            f"train.micro_batch={batch}",
            "train.steps=100",
            "train.eval_every=50",
            "train.ckpt_every=100",
            "train.stats_every=10",
            *overrides,
        ],
    )


def _cpu() -> DeviceContext:
    return resolve(load_hardware_profile("cpu"))


def _build(cfg: RunConfig, seed: int = 0) -> HSN:
    seed_everything(seed)
    return HSN.from_config(cfg)


def _perturb(model: HSN, seed: int, scale: float = 0.3) -> None:
    """Seeded noise on every parameter (the P1.5 recipe): states of order 0.1 instead of the
    1e-5 of a freshly initialised model, so that a carried state has a visible effect."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn(p.shape, generator=generator, dtype=torch.float64) * scale
            p.add_(noise.to(dtype=p.dtype, device=p.device))


def _first_window(cfg: RunConfig, docs: list[Example]) -> dict[str, Tensor]:
    """The first training window exactly as the trainer's loader produces it."""
    loader = StreamLoader(
        ByteDataset(docs),
        batch=cfg.train.batch,
        unroll=cfg.train.unroll,
        seed=cfg.seed,
        shuffle=True,
        align_docs=cfg.data.align_docs,
        epochs=None,
    )
    return next(loader)


def _window_output(model: HSN, window: dict[str, Tensor]) -> dict[str, Tensor]:
    B = int(window["tokens"].shape[0])
    logits, _, info = model(
        window["tokens"], window["context"], model.init_state(B), window["reset"]
    )
    return {"logits": logits, **info}


def _with_train(cfg: RunConfig, **changes: Any) -> RunConfig:
    """``cfg`` with ``train.*`` fields replaced (no re-validation: that is what the loss checks)."""
    return cfg.model_copy(update={"train": cfg.train.model_copy(update=changes)})


# --------------------------------------------------------------------------------------------------
# Tests named by the chunk
# --------------------------------------------------------------------------------------------------


def test_p1_7_resume_bit_exact_cpu_20_steps(
    train_docs: list[Example], dev_docs: list[Example], tmp_path: Path
) -> None:
    """Train 20 steps; resume from the step-10 checkpoint into a differently initialised trainer;
    every per-step number, the weights, the optimiser, the carried state and the resync counters
    agree bit for bit; checkpoints that lack the P1.7 entries are refused."""
    cfg = _config(
        "train.steps=20",
        "train.eval_every=10",
        "train.ckpt_every=10",
        "train.stats_every=5",
        "train.q_resync_every=3",
        "train.deterministic=true",
        "train.warmup_steps=4",
        "train.schedule=cosine",
    )
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)

    run_a = tmp_path / "a"
    model_a = _build(cfg)
    assert isinstance(model_a, SequenceModel)  # HSN satisfies the P0.5 protocol (reset_state)
    trainer_a = TBPTTTrainer(model_a, cfg, _cpu(), run_dir=run_a)
    history_a = trainer_a.fit(train_ds, dev_ds, 20)
    assert [Path(p).name for p in history_a.checkpoints] == ["step_00000010.pt", "step_00000020.pt"]
    assert latest_checkpoint(run_a) == run_a / "step_00000020.pt"
    assert (trainer_a.windows, trainer_a.resyncs, trainer_a.windows_since_resync) == (20, 6, 2)
    assert [r.resyncs for r in history_a.steps] == [n // 3 for n in range(1, 21)]
    assert history_a.measured["windows"] == 20 and history_a.measured["resyncs"] == 6
    assert history_a.measured["param_count"] == model_a.param_report()["total_trainable"]

    run_b = tmp_path / "b"
    model_b = _build(cfg, seed=1)  # different initial weights: everything must come from the file
    trainer_b = TBPTTTrainer(model_b, cfg, _cpu(), run_dir=run_b)
    assert trainer_b.resume(run_a / "step_00000010.pt") == 10
    assert (trainer_b.windows, trainer_b.resyncs, trainer_b.windows_since_resync) == (10, 3, 1)
    history_b = trainer_b.fit(train_ds, dev_ds, 20)

    assert [r.step for r in history_b.steps] == list(range(11, 21))
    for rec_a, rec_b in zip(history_a.steps[10:], history_b.steps, strict=True):
        assert isinstance(rec_a, TBPTTStepRecord) and isinstance(rec_b, TBPTTStepRecord)
        assert rec_a.loss == rec_b.loss and rec_a.accuracy == rec_b.accuracy, (rec_a, rec_b)
        assert rec_a.total == rec_b.total and rec_a.parts == rec_b.parts, (rec_a, rec_b)
        assert rec_a.stats == rec_b.stats, (rec_a.stats, rec_b.stats)
        assert rec_a.grad_norm == rec_b.grad_norm and rec_a.lr == rec_b.lr
        assert (rec_a.windows, rec_a.resyncs) == (rec_b.windows, rec_b.resyncs)
        assert rec_a.loss_positions == rec_b.loss_positions and rec_a.tokens == rec_b.tokens
    assert history_a.evals[-1]["loss"] == history_b.evals[-1]["loss"]
    for key, value in model_a.state_dict().items():
        other = model_b.state_dict()[key]
        assert torch.equal(value, other) if isinstance(value, Tensor) else value == other, key
    state_a, state_b = trainer_a.optimizer.state_dict(), trainer_b.optimizer.state_dict()
    for index, entry in state_a["state"].items():
        for key, value in entry.items():
            other = state_b["state"][index][key]
            assert torch.equal(value, other) if isinstance(value, Tensor) else value == other
    assert (
        trainer_a.scheduler.state_dict()["last_epoch"]
        == trainer_b.scheduler.state_dict()["last_epoch"]
    )
    # the step-20 checkpoints of both runs are the same: weights, RNG, loader, state, counters
    file_a = torch.load(run_a / "step_00000020.pt", weights_only=True)
    file_b = torch.load(run_b / "step_00000020.pt", weights_only=True)
    for key, value in file_a["model"].items():
        other = file_b["model"][key]
        assert torch.equal(value, other) if isinstance(value, Tensor) else value == other, key
    assert torch.equal(file_a["rng_state"]["torch"], file_b["rng_state"]["torch"])
    assert file_a["loader_state"] == file_b["loader_state"]
    for name in ("h", "q", "age", "pos"):
        assert torch.equal(
            file_a["extra"]["carried_state"][name], file_b["extra"]["carried_state"][name]
        )
        assert file_a["extra"]["carried_state"][name].device.type == "cpu"
    assert file_a["extra"]["q_resync"] == file_b["extra"]["q_resync"]
    assert file_a["extra"]["q_resync"] == {"windows": 20, "since_resync": 2, "resyncs": 6}
    assert file_a["extra"]["model_name"] == "hsn" and file_a["extra"]["scaler"] is None
    # the carried state restores through the helpers with the P1.3 layout checks
    restored = unpack_cell_state(file_a["extra"]["carried_state"], torch.device("cpu"))
    assert isinstance(restored, CellGraphState) and restored.batch_size == cfg.train.batch
    assert torch.equal(pack_cell_state(restored)["h"], file_a["extra"]["carried_state"]["h"])
    with pytest.raises(ValueError, match="missing"):
        unpack_cell_state({"h": restored.h, "q": restored.q}, "cpu")
    with pytest.raises(TypeError):
        pack_cell_state(restored.h)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="q_resync"):
        check_q_resync({"windows": 1, "since_resync": 2, "resyncs": 0}, "test")
    # a checkpoint of another configuration is refused; one without the P1.7 entries too
    other_cfg = _config("train.lr=2e-3", "train.steps=20")
    trainer_c = TBPTTTrainer(_build(other_cfg), other_cfg, _cpu(), run_dir=tmp_path / "c")
    with pytest.raises(ValueError, match="configuration"):
        trainer_c.resume(run_a / "step_00000010.pt")
    for key, message in (("q_resync", "q_resync counters"), ("carried_state", "carried cell")):
        payload = torch.load(run_a / "step_00000010.pt", weights_only=True)
        del payload["extra"][key]
        tampered = tmp_path / f"no_{key}.pt"
        torch.save(payload, tampered)
        with pytest.raises(ValueError, match=message):
            TBPTTTrainer(_build(cfg), cfg, _cpu(), run_dir=tmp_path / key).resume(tampered)


def test_p1_7_state_detached_at_window_start(
    train_docs: list[Example], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state handed to the model at every window start carries no autograd history, the
    state it returns does, the next window starts from exactly that output detached, and the
    summary is recomputed every ``train.q_resync_every`` windows."""
    cfg = _config(
        "train.steps=4",
        "train.eval_every=4",
        "train.ckpt_every=4",
        "train.stats_every=4",
        "train.q_resync_every=2",
    )
    ds = ByteDataset(train_docs)
    model = _build(cfg)
    trainer = TBPTTTrainer(model, cfg, _cpu(), run_dir=tmp_path)
    seen: list[tuple[CellGraphState, CellGraphState]] = []
    original = trainer._forward

    def spy(batch: dict[str, Tensor], state: CellGraphState) -> Any:
        assert isinstance(state, CellGraphState)
        for name in ("h", "q", "age", "pos"):
            tensor = getattr(state, name)
            assert tensor.grad_fn is None and not tensor.requires_grad, name
        logits, new_state, info = original(batch, state)
        assert new_state.h.grad_fn is not None and new_state.h.requires_grad
        assert new_state.q.grad_fn is not None and logits.requires_grad
        seen.append((state, new_state))
        return logits, new_state, info

    monkeypatch.setattr(trainer, "_forward", spy)
    recompute_at: list[int] = []
    recompute = CellGraphState.recompute_q

    def spy_recompute(self: CellGraphState) -> Tensor:
        recompute_at.append(len(seen) + 1)  # the window about to start
        return recompute(self)

    monkeypatch.setattr(CellGraphState, "recompute_q", spy_recompute)
    history = trainer.fit(ds, ds, 4)
    assert len(seen) == 4 and len(history.steps) == 4
    assert not seen[0][0].h.any() and not seen[0][0].pos.any()  # the first window starts at zero
    for (state_in, _), (_, previous_out) in zip(seen[1:], seen[:-1], strict=True):
        assert state_in is not previous_out
        assert torch.equal(state_in.h, previous_out.h.detach())
        assert torch.equal(state_in.q, previous_out.q.detach())
        assert torch.equal(state_in.age, previous_out.age) and torch.equal(
            state_in.pos, previous_out.pos
        )
    assert recompute_at == [2, 4]
    assert (trainer.windows, trainer.resyncs, trainer.windows_since_resync) == (4, 2, 0)
    assert [r.resyncs for r in history.steps] == [0, 1, 1, 2]
    carried = torch.load(tmp_path / "step_00000004.pt", weights_only=True)["extra"]["carried_state"]
    assert torch.equal(carried["h"], seen[-1][1].h.detach()) and not carried["h"].requires_grad
    assert isinstance(trainer._carried_state, CellGraphState)
    assert trainer._carried_state.h.grad_fn is None


def test_p1_7_reset_applied_at_doc_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With unaligned documents the loader's resets fall inside windows; after every window each
    row's ``pos`` counts the positions since its last reset, a row reset at position 0 equals a
    fresh forward from the zero state, and a row without a reset depends on its carried state."""
    task = _tiny_task(length_range=(20, 96), dep_range=(4, 10))
    docs = _docs(task, "train", 24)
    lengths = [doc.length for doc in docs]
    assert min(lengths) < UNROLL < max(lengths)
    cfg = _config(
        "data.align_docs=false",
        "train.steps=8",
        "train.eval_every=8",
        "train.ckpt_every=8",
        "train.stats_every=8",
        batch=2,
    )
    ds = ByteDataset(docs)
    model = _build(cfg)
    _perturb(model, seed=3)
    trainer = TBPTTTrainer(model, cfg, _cpu(), run_dir=tmp_path)
    original = trainer._forward
    checked = {"mid_reset": 0, "no_reset": 0, "reset_at_0": 0, "carried": 0}

    def spy(batch: dict[str, Tensor], state: CellGraphState) -> Any:
        logits, new_state, info = original(batch, state)
        reset = batch["reset"]
        B, T = reset.shape
        for b in range(B):
            resets = reset[b].nonzero().flatten().tolist()
            if resets:  # pos counts the positions since the row's last document start
                assert int(new_state.pos[b]) == T - resets[-1]
                checked["mid_reset" if resets[-1] > 0 else "reset_at_0"] += 1
            else:  # no boundary: the row continues its document from the carried state
                assert int(new_state.pos[b]) == int(state.pos[b]) + T
                checked["no_reset"] += 1
            tokens = batch["tokens"][b : b + 1]
            context = batch["context"][b : b + 1]
            row = CellGraphState(
                h=state.h[b : b + 1],
                q=state.q[b : b + 1],
                age=state.age[b : b + 1],
                pos=state.pos[b : b + 1],
            )
            with torch.no_grad():  # the same row alone, same weights, one position
                _, from_carried, _ = model(tokens[:, :1], context[:, :1], row, reset[b : b + 1, :1])
                _, from_zero, _ = model(
                    tokens[:, :1], context[:, :1], model.init_state(1), reset[b : b + 1, :1]
                )
            if bool(reset[b, 0]):  # a boundary at position 0: the carried state is discarded
                assert torch.equal(from_carried.h, from_zero.h)
                with torch.no_grad():  # ... over the whole window too
                    _, fresh, _ = model(tokens, context, model.init_state(1), reset[b : b + 1])
                torch.testing.assert_close(fresh.h[0], new_state.h[b], atol=1e-6, rtol=1e-5)
            elif bool(state.h[b].any()):  # no boundary: the carried state reaches the model
                assert (from_carried.h - from_zero.h).abs().max() > 1e-3
                checked["carried"] += 1
        return logits, new_state, info

    monkeypatch.setattr(trainer, "_forward", spy)
    history = trainer.fit(ds, ds, 8)
    assert len(history.steps) == 8
    assert all(count > 0 for count in checked.values()), checked
    # the reset mask the model saw is the loader's: True exactly at document starts
    loader = StreamLoader(
        ds, batch=2, unroll=UNROLL, seed=cfg.seed, shuffle=True, align_docs=False, epochs=None
    )
    for _ in range(8):
        window = next(loader)
        doc = window["doc"]
        starts = torch.zeros_like(window["reset"])
        starts[:, 1:] = doc[:, 1:] != doc[:, :-1]
        assert torch.equal(window["reset"][:, 1:], starts[:, 1:] & (doc[:, 1:] >= 0))


def test_p1_7_one_at_a_time_rule_enforced(train_docs: list[Example]) -> None:
    """``assemble_loss`` accepts every §2.6 term, weights each by its λ, includes it only when
    enabled, and refuses more than one auxiliary term unless justified."""
    with pytest.raises(ValueError, match="one at a time"):
        load_config(REF_A1, overrides=["train.aux_enabled=[cost, balance]"])
    cfg = _config()
    model = _build(cfg)
    window = _first_window(cfg, train_docs)
    out = _window_output(model, window)
    base_total, base_parts = assemble_loss(out, window, cfg)
    assert list(base_parts) == ["task", "pred"] and base_total.requires_grad
    assert TERMS == ("task", "pred", "replay", "distill", "ewc", "cost", "balance", "graph")

    # two auxiliary weights set: refused before anything else is checked
    with pytest.raises(ValueError, match="one at a time"):
        assemble_loss(out, window, _with_train(cfg, lambda_c=0.1, lambda_b=0.1))
    with pytest.raises(ValueError, match="one at a time"):
        assemble_loss(out, window, _with_train(cfg, aux_enabled=["cost", "graph"], lambda_c=0.1))
    # the two switches of an auxiliary term must agree
    with pytest.raises(ValueError, match="aux_enabled"):
        assemble_loss(out, window, _with_train(cfg, lambda_c=0.1))
    with pytest.raises(ValueError, match="lambda_b"):
        assemble_loss(out, window, _with_train(cfg, aux_enabled=["balance"]))
    # an enabled term without a value names the chunk that computes it
    with pytest.raises(ValueError, match="P2.4"):
        assemble_loss(out, window, _with_train(cfg, aux_enabled=["cost"], lambda_c=0.1))
    with pytest.raises(ValueError, match="P3.1"):
        assemble_loss(out, window, _with_train(cfg, lambda_r=0.5))
    with pytest.raises(ValueError, match="P3.3"):
        assemble_loss(out, window, _with_train(cfg, lambda_e=0.5))
    # a value for a term that is off, a term computed here, an unknown name, a bad value
    with pytest.raises(ValueError, match="not enabled"):
        assemble_loss(out, window, cfg, extras={"cost": torch.tensor(1.0)})
    with pytest.raises(ValueError, match="computed here"):
        assemble_loss(out, window, cfg, extras={"pred": torch.tensor(1.0)})
    with pytest.raises(ValueError, match="accepts"):
        assemble_loss(out, window, cfg, extras={"ponder": torch.tensor(1.0)})
    with pytest.raises(ValueError, match="0-dim"):
        assemble_loss(out, window, _with_train(cfg, lambda_r=0.5), extras={"replay": torch.ones(2)})
    with pytest.raises(ValueError, match=">= 0"):
        assemble_loss(out, window, _with_train(cfg, lambda_g=-1.0))
    with pytest.raises(TypeError):
        assemble_loss(out, window, cfg.train)  # type: ignore[arg-type]

    # one auxiliary term at a time: weighted and listed in formula order
    cost = torch.tensor(2.0, requires_grad=True)
    one = _with_train(cfg, aux_enabled=["cost"], lambda_c=0.5)
    total, parts = assemble_loss(out, window, one, extras={"cost": cost})
    assert list(parts) == ["task", "pred", "cost"] and parts["cost"] == 1.0
    torch.testing.assert_close(total, base_total + 1.0)
    total.backward()
    assert cost.grad is not None and float(cost.grad) == 0.5
    # more than one only with a written justification; continual terms are weighted too
    both = _with_train(
        cfg,
        aux_enabled=["cost", "balance"],
        lambda_c=0.5,
        lambda_b=0.25,
        lambda_r=2.0,
        lambda_e=1.5,
        allow_multiple_aux="ablation: cost and balance together",
    )
    extras = {
        "balance": torch.tensor(4.0),
        "cost": torch.tensor(2.0),
        "ewc": torch.tensor(2.0, dtype=torch.float64),
        "replay": torch.tensor(0.5),
    }
    total, parts = assemble_loss(out, window, both, extras=extras)
    assert list(parts) == ["task", "pred", "replay", "ewc", "cost", "balance"]
    assert parts["replay"] == 1.0 and parts["ewc"] == 3.0
    assert parts["cost"] == 1.0 and parts["balance"] == 1.0
    assert total.dtype == torch.float32
    torch.testing.assert_close(total, base_total + 6.0)
    # lambda_p = 0 switches the prediction term off; the EMA target adds the variance floor
    _, parts = assemble_loss(out, window, _with_train(cfg, lambda_p=0.0))
    assert list(parts) == ["task"]
    cfg_e = _config("model.predictor_target=ema_encoder")
    _, parts = assemble_loss(_window_output(_build(cfg_e), window), window, cfg_e)
    assert list(parts) == ["task", "pred", "variance_floor"]
    # a window of padding only gives a zero task term with a zero gradient
    padding = {k: v.clone() for k, v in window.items()}
    padding["loss_mask"][:] = False
    _, parts = assemble_loss(out, padding, cfg)
    assert parts["task"] == 0.0 and parts["pred"] == base_parts["pred"]
    with pytest.raises(KeyError, match="logits"):
        assemble_loss({k: v for k, v in out.items() if k != "logits"}, window, cfg)
    with pytest.raises(ValueError, match="loss_mask"):
        assemble_loss(out, {**window, "loss_mask": window["loss_mask"][:, :3]}, cfg)


def test_p1_7_loss_parts_logged(
    train_docs: list[Example],
    dev_docs: list[Example],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every step records the loss parts and the model statistics; stats steps add the gradient
    norm per parameter group and one log line carrying all of it; the parts agree with an
    independent assembly of the first window."""
    cfg = _config(
        "train.steps=4", "train.eval_every=4", "train.ckpt_every=4", "train.stats_every=2"
    )
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)
    model = _build(cfg)
    trainer = TBPTTTrainer(model, cfg, _cpu(), run_dir=tmp_path)
    with caplog.at_level(logging.INFO, logger="hsn.train"):
        history = trainer.fit(train_ds, dev_ds, 4)
    records = history.steps
    assert len(records) == 4 and all(isinstance(r, TBPTTStepRecord) for r in records)
    for record in records:
        assert isinstance(record, TBPTTStepRecord)
        assert list(record.parts) == ["task", "pred"]
        assert math.isfinite(record.total) and all(math.isfinite(v) for v in record.parts.values())
        assert record.total == pytest.approx(sum(record.parts.values()))
        assert record.loss == pytest.approx(record.parts["task"])  # one window per step
        assert set(STAT_KEYS) <= set(record.stats)
        assert 0.0 < record.stats["state_abs_max"] <= 1.0  # tanh-bounded state (§2.3)
        assert record.stats["eps_mean"] > 0.0
        assert 0.0 <= record.stats["alpha_entropy"] <= math.log(cfg.model.n_units) + 1e-6
        assert math.isfinite(record.grad_norm) and record.grad_norm > 0.0
    for record in (records[0], records[2]):
        assert not any(key.startswith("grad_norm/") for key in record.stats)
    for record in (records[1], records[3]):
        groups = {key[len("grad_norm/") :] for key in record.stats if key.startswith("grad_norm/")}
        assert groups == GROUPS_WITH_GRADS
        combined = math.sqrt(
            sum(v**2 for k, v in record.stats.items() if k.startswith("grad_norm/"))
        )
        assert combined == pytest.approx(record.grad_norm, rel=1e-4)
    lines = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("step ")]
    assert [line.split()[1] for line in lines] == ["2/4", "4/4"]
    for line in lines:
        for key in ("total", "task", "pred", *STAT_KEYS, "grad_norm/encoder", "grad_norm/cell"):
            assert f"{key} " in line, (key, line)
    assert any("params" in rec.getMessage() for rec in caplog.records)
    assert history.param_report == model.param_report()
    assert history.measured["param_count"] == model.param_report()["total_trainable"]
    assert history.measured["tokens_per_s"] > 0 and history.measured["windows"] == 4
    # the first step's parts are exactly an independent assembly of the first window
    fresh = _build(cfg)
    window = _first_window(cfg, train_docs)
    _, parts = assemble_loss(_window_output(fresh, window), window, cfg)
    assert records[0].parts == parts
    terms = prediction_terms(
        *(_window_output(fresh, window)[k].float() for k in ("z_hat", "z_target", "z")),
        lambda_p=cfg.train.lambda_p,
        predictor_target=cfg.model.predictor_target,
        variance_floor_gamma=cfg.train.variance_floor_gamma,
        mask=window["doc"] >= 0,
    )
    assert parts["pred"] == float(terms["pred"].detach())
    # the EMA target adds its floor to the parts and moves after every optimiser step
    cfg_e = _config(
        "train.steps=2",
        "train.eval_every=2",
        "train.ckpt_every=2",
        "train.stats_every=2",
        "model.predictor_target=ema_encoder",
        "train.ema_tau=0.5",
    )
    model_e = _build(cfg_e)
    assert model_e.ema_target is not None
    before = {name: b.clone() for name, b in model_e.ema_target.named_buffers()}
    history_e = TBPTTTrainer(model_e, cfg_e, _cpu(), run_dir=tmp_path / "ema").fit(
        train_ds, dev_ds, 2
    )
    first = history_e.steps[0]
    assert isinstance(first, TBPTTStepRecord)
    assert list(first.parts) == ["task", "pred", "variance_floor"]
    assert first.parts["variance_floor"] > 0
    assert all(not torch.equal(b, before[name]) for name, b in model_e.ema_target.named_buffers())
    # guards: not an HSN, a configuration for another model
    with pytest.raises(TypeError, match="HSN"):
        TBPTTTrainer(nn.Linear(2, 2), cfg, _cpu(), run_dir=tmp_path)
    cfg_gru = _config("model_name=gru")
    with pytest.raises(ValueError, match="model_name"):
        TBPTTTrainer(_build(cfg), cfg_gru, _cpu(), run_dir=tmp_path)
    # the generic trainer accepts the HSN as a SequenceModel but cannot run it: it requires the
    # baseline report keys (active / stored / trainable) and cannot carry a CellGraphState
    generic = Trainer(_build(cfg), cfg, _cpu(), run_dir=tmp_path / "generic")
    with pytest.raises(KeyError, match="active"):
        generic.fit(train_ds, dev_ds, 1)


@pytest.mark.parametrize("profile", [cuda_param, mps_param])
def test_p1_7_amp_runs(
    profile: str, train_docs: list[Example], dev_docs: list[Example], tmp_path: Path
) -> None:
    """Four steps under the profile's autocast (bf16 on CUDA, fp16 with the loss scaler on MPS):
    finite objective, fp32 master weights and carried state, a checkpoint that resumes."""
    hardware = load_hardware_profile(profile)
    available = (
        torch.cuda.is_available()
        if hardware.device == "cuda"
        else torch.backends.mps.is_available()
    )
    if not available:
        pytest.skip(f"no {hardware.device} device on this machine")
    ctx = resolve(hardware)
    assert ctx.autocast_dtype is not None
    cfg = _config(
        f"profile={profile}",
        "train.steps=4",
        "train.eval_every=2",
        "train.ckpt_every=4",
        "train.stats_every=2",
    )
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)
    model = _build(cfg)
    trainer = TBPTTTrainer(model, cfg, ctx, run_dir=tmp_path)
    assert (trainer.scaler is not None) == (ctx.autocast_dtype is torch.float16)
    history = trainer.fit(train_ds, dev_ds, 4)
    assert history.autocast == autocast_name(ctx)
    assert np.isfinite([r.total for r in history.steps]).all()
    assert all(math.isfinite(v) for r in history.steps for v in r.parts.values())
    assert next(model.parameters()).dtype is torch.float32  # master weights stay fp32
    carried = trainer._carried_state
    assert isinstance(carried, CellGraphState)
    assert carried.dtype is torch.float32 and carried.device.type == ctx.device.type
    assert history.evals and math.isfinite(history.evals[-1]["loss"])
    again = TBPTTTrainer(_build(cfg, seed=1), cfg, ctx, run_dir=tmp_path / "again")
    assert again.resume(tmp_path / "step_00000004.pt") == 4
    if trainer.scaler is not None:
        assert again.scaler is not None
        assert again.scaler.state_dict() == trainer.scaler.state_dict()
    assert isinstance(again._carried_state, CellGraphState)
    assert torch.equal(again._carried_state.h.cpu(), carried.h.cpu())
