"""P0.5 tests: baselines, windowed trainer, checkpoints, AMP policy, evaluation, search budget."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from cards import CUDA_PROFILE
from hsn.baselines.gru import GRUBaseline
from hsn.baselines.transformer import TransformerBaseline
from hsn.config import RunConfig, dump_resolved, load_config, load_hardware_profile
from hsn.data.dataset import ByteDataset, collate_windows, iter_windows
from hsn.data.synthetic import PAD, Example, TaskSpec, generate, load_dataset_spec
from hsn.device import DeviceContext, resolve, seed_everything
from hsn.eval.evaluate import (
    DEFAULT_METRICS,
    MetricAccumulator,
    bits_per_byte,
    evaluate_windows,
    nll,
    token_accuracy,
)
from hsn.train.amp import autocast_ctx, autocast_name, make_grad_scaler
from hsn.train.checkpoint import (
    capture_rng_state,
    latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from hsn.train.loop import (
    BudgetExhaustedError,
    History,
    SearchBudget,
    SequenceModel,
    StreamLoader,
    Trainer,
    detach_state,
)
from torch import Tensor, nn

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "configs" / "_example.yaml"
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
UNROLL = 32  # the tiny window every test uses
N_DOCS = 64

cuda_param = pytest.param(CUDA_PROFILE, marks=pytest.mark.cuda)
mps_param = pytest.param("m_series", marks=pytest.mark.mps)


# --------------------------------------------------------------------------------------------------
# Fixtures: 64 short documents of a synthetic_v1 family and a tiny configuration
# --------------------------------------------------------------------------------------------------


def _tiny_task(
    family: str = "algorithmic", split_sizes: tuple[int, int, int] = (64, 16, 16)
) -> TaskSpec:
    """The committed in-range task with documents that fit a window of UNROLL positions."""
    task = SPEC.task(family, "in_range")  # type: ignore[arg-type]
    n_train, n_dev, n_test = split_sizes
    params = dict(task.params)
    if family == "algorithmic":
        params["n_max"] = 4
    dep_range = [8, 24] if family == "algorithmic" else [4, 20]  # kv needs dep + 9 symbols
    return TaskSpec.model_validate(
        {
            **task.model_dump(),
            "n_train": n_train,
            "n_dev": n_dev,
            "n_test": n_test,
            "length_range": [16, UNROLL],
            "dep_len_range": dep_range,
            "unroll": UNROLL,
            "params": params,
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


def _config(*overrides: str, batch: int = 8) -> RunConfig:
    return load_config(
        EXAMPLE,
        overrides=[
            f"train.unroll={UNROLL}",
            f"train.batch={batch}",
            f"train.micro_batch={batch}",
            "train.steps=500",
            "train.eval_every=100",
            "train.ckpt_every=500",
            "train.stats_every=100",
            "baseline.gru.d_model=64",
            "baseline.gru.n_layers=1",
            "baseline.transformer.d_model=64",
            "baseline.transformer.n_layers=2",
            "baseline.transformer.n_heads=4",
            "baseline.transformer.ffn_hidden=128",
            *overrides,
        ],
    )


def _cpu() -> DeviceContext:
    return resolve(load_hardware_profile("cpu"))


def _build(cfg: RunConfig, seed: int = 0) -> nn.Module:
    seed_everything(seed)
    if cfg.model_name == "gru":
        return GRUBaseline(cfg)
    if cfg.model_name == "transformer":
        return TransformerBaseline(cfg)
    raise ValueError(cfg.model_name)


def _first_step_above(history: History, threshold: float) -> int | None:
    for record in history.evals:
        if record["accuracy"] > threshold:
            return int(record["step"])
    return None


def _overfit(model_name: str, lr: float, train_docs: list[Example], tmp_path: Path) -> History:
    cfg = _config(f"model_name={model_name}", f"train.lr={lr}")
    model = _build(cfg)
    ds = ByteDataset(train_docs)
    trainer = Trainer(model, cfg, _cpu(), run_dir=tmp_path)
    history = trainer.fit(ds, ds, 500)  # dev_ds = the training documents: train accuracy
    assert len(history.steps) == 500 and history.steps[-1].step == 500
    assert [record["step"] for record in history.evals] == [100, 200, 300, 400, 500]
    final = history.evals[-1]
    assert final["accuracy"] > 0.99, final
    assert final["n_docs"] == N_DOCS
    assert final["per_group"]["algorithmic/in_range"]["n_positions"] == final["n_positions"]
    # every loss position was seen: the per-group count equals the documents' answer count
    assert final["n_positions"] == sum(int(doc.loss_mask.sum()) for doc in train_docs)
    assert np.isfinite([record.loss for record in history.steps]).all()
    assert history.measured["param_count"] == model.param_report()["stored"]
    assert history.measured["tokens_per_s"] > 0
    assert history.checkpoints and Path(history.checkpoints[-1]).is_file()
    return history


# --------------------------------------------------------------------------------------------------
# Tests named by the chunk
# --------------------------------------------------------------------------------------------------


def test_p0_5_gru_overfits_64_examples(train_docs: list[Example], tmp_path: Path) -> None:
    history = _overfit("gru", 3e-3, train_docs, tmp_path)
    reached = _first_step_above(history, 0.99)
    assert reached is not None and reached <= 500


def test_p0_5_transformer_overfits_64_examples(train_docs: list[Example], tmp_path: Path) -> None:
    history = _overfit("transformer", 1e-3, train_docs, tmp_path)
    reached = _first_step_above(history, 0.99)
    assert reached is not None and reached <= 500


def test_p0_5_resume_bit_exact_on_cpu(
    train_docs: list[Example], dev_docs: list[Example], tmp_path: Path
) -> None:
    """Train 20 steps; resume from the step-10 checkpoint; the two trajectories agree bit for bit.

    Dropout is on, so the resumed run only matches when the RNG state, the optimiser state, the
    loader position and the carried recurrent state all came back from the checkpoint.
    """
    cfg = _config(
        "model_name=gru",
        "train.lr=1e-3",
        "train.steps=20",
        "train.eval_every=10",
        "train.ckpt_every=10",
        "train.deterministic=true",
        "baseline.gru.n_layers=2",
        "baseline.gru.dropout=0.1",
        "train.warmup_steps=4",
        "train.schedule=cosine",
        batch=4,
    )
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)

    run_a = tmp_path / "a"
    model_a = _build(cfg)
    trainer_a = Trainer(model_a, cfg, _cpu(), run_dir=run_a)
    history_a = trainer_a.fit(train_ds, dev_ds, 20)
    assert [Path(p).name for p in history_a.checkpoints] == ["step_00000010.pt", "step_00000020.pt"]
    assert latest_checkpoint(run_a) == run_a / "step_00000020.pt"

    run_b = tmp_path / "b"
    model_b = _build(cfg, seed=1)  # different initial weights: everything must come from the file
    trainer_b = Trainer(model_b, cfg, _cpu(), run_dir=run_b)
    assert trainer_b.resume(run_a / "step_00000010.pt") == 10
    history_b = trainer_b.fit(train_ds, dev_ds, 20)

    assert [record.step for record in history_b.steps] == list(range(11, 21))
    for rec_a, rec_b in zip(history_a.steps[10:], history_b.steps, strict=True):
        assert rec_a.loss == rec_b.loss and rec_a.accuracy == rec_b.accuracy, (rec_a, rec_b)
        assert rec_a.grad_norm == rec_b.grad_norm and rec_a.lr == rec_b.lr
    assert history_a.evals[-1]["loss"] == history_b.evals[-1]["loss"]
    for key, value in model_a.state_dict().items():
        assert torch.equal(value, model_b.state_dict()[key]), key
    state_a, state_b = trainer_a.optimizer.state_dict(), trainer_b.optimizer.state_dict()
    for index, entry in state_a["state"].items():
        for key, value in entry.items():
            other = state_b["state"][index][key]
            assert torch.equal(value, other) if isinstance(value, Tensor) else value == other
    assert (
        trainer_a.scheduler.state_dict()["last_epoch"]
        == trainer_b.scheduler.state_dict()["last_epoch"]
    )
    # the step-20 checkpoints of both runs hold identical weights and RNG state
    file_a = torch.load(run_a / "step_00000020.pt", weights_only=True)
    file_b = torch.load(run_b / "step_00000020.pt", weights_only=True)
    for key, value in file_a["model"].items():
        assert torch.equal(value, file_b["model"][key])
    assert torch.equal(file_a["rng_state"]["torch"], file_b["rng_state"]["torch"])
    assert file_a["loader_state"] == file_b["loader_state"]
    # a checkpoint of another configuration is refused by resume (weights still load explicitly)
    other_cfg = _config(
        "model_name=gru",
        "train.lr=2e-3",
        "train.steps=20",
        "baseline.gru.n_layers=2",
        "baseline.gru.dropout=0.1",
        batch=4,
    )
    trainer_c = Trainer(_build(other_cfg), other_cfg, _cpu(), run_dir=tmp_path / "c")
    with pytest.raises(ValueError, match="configuration"):
        trainer_c.resume(run_a / "step_00000010.pt")
    assert load_checkpoint(run_a / "step_00000010.pt", trainer_c.model)["step"] == 10


def test_p0_5_param_report_logged(
    train_docs: list[Example], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _config(
        "model_name=transformer", "train.steps=2", "train.eval_every=2", "train.ckpt_every=2"
    )
    model = _build(cfg)
    report = model.param_report()
    assert isinstance(model, SequenceModel)
    assert report["active"] == report["stored"] == report["trainable"]
    assert report["stored"] == sum(p.numel() for p in model.parameters())
    assert report["stored"] == sum(
        report[k] for k in ("byte_emb", "ctx_emb", "pos_emb", "blocks", "norm", "head")
    )
    ds = ByteDataset(train_docs)
    with caplog.at_level(logging.INFO, logger="hsn.train"):
        history = Trainer(model, cfg, _cpu(), run_dir=tmp_path).fit(ds, ds, 2)
    assert history.param_report == report
    assert history.measured["param_count"] == report["stored"]
    assert history.measured["active_param_count"] == report["active"]
    logged = [record.getMessage() for record in caplog.records if "params" in record.getMessage()]
    assert logged and f'"stored": {report["stored"]}' in logged[0]
    assert f'"active": {report["active"]}' in logged[0]
    # the reference-size defaults are the documented ≈2M-parameter baselines (measured here)
    ref = load_config(EXAMPLE)
    assert GRUBaseline(ref).param_report()["stored"] == 1_977_856
    assert TransformerBaseline(ref).param_report()["stored"] == 1_930_624


# --------------------------------------------------------------------------------------------------
# Deviation: configuration keys this chunk added (P0.2 conventions)
# --------------------------------------------------------------------------------------------------


def test_p0_5_baseline_config_validates(tmp_path: Path) -> None:
    cfg = load_config(EXAMPLE)
    assert cfg.baseline.gru.d_model == 384 and cfg.baseline.gru.n_layers == 2
    assert cfg.baseline.transformer.d_model == 192 and cfg.baseline.transformer.n_heads == 4
    assert cfg.train.search_trials == 1 and cfg.data.align_docs is True
    cfg2 = load_config(
        EXAMPLE,
        overrides=["baseline.gru.d_model=16", "train.search_trials=3", "data.align_docs=false"],
    )
    assert cfg2.baseline.gru.d_model == 16 and cfg2.train.search_trials == 3
    assert cfg2.data.align_docs is False
    dump_resolved(cfg2, tmp_path / "resolved.yaml")
    assert load_config(tmp_path / "resolved.yaml") == cfg2
    for bad in (
        "baseline.transformer.d_model=98",  # not a multiple of n_heads=4
        "baseline.transformer.n_heads=0",
        "baseline.gru.n_layers=0",
        "baseline.gru.dropout=1.0",
        "train.search_trials=0",
        "baseline.lstm.d_model=4",  # unknown baseline
    ):
        with pytest.raises(ValueError):
            load_config(EXAMPLE, overrides=[bad])


# --------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------


def _random_window(cfg: RunConfig, B: int, T: int, seed: int = 0) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, cfg.model.vocab_size, (B, T), generator=g)
    context = torch.randint(0, cfg.model.n_context, (B, T), generator=g)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    reset[0, 5] = True  # a document boundary inside the window, in one row only
    reset[1, 5] = True
    reset[1, 20] = True
    if B > 2:
        reset[2, T - 1] = True  # a reset at the last position of a row
    return {"tokens": tokens, "context": context, "reset": reset}


def test_p0_5_gru_resets_match_stepwise_reference() -> None:
    """Segment-wise cuDNN calls with masked state equal a position-by-position GRU with resets."""
    cfg = _config("model_name=gru", "baseline.gru.n_layers=2")
    model = _build(cfg).eval()
    assert isinstance(model, GRUBaseline)
    B, T = 4, UNROLL
    w = _random_window(cfg, B, T)
    state = model.init_state(B, "cpu")
    assert state.shape == (2, B, 64) and not state.any()
    # a non-zero incoming state shows that reset[:, 0] clears it
    state = torch.randn(2, B, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits, new_state, info = model(w["tokens"], w["context"], state, w["reset"])
        x = model.tok_emb(w["tokens"]) + model.ctx_emb(w["context"])
        h = state
        outs = []
        for t in range(T):
            h = h.masked_fill(w["reset"][:, t].view(1, B, 1), 0.0)
            out, h = model.gru(x[:, t : t + 1], h)
            outs.append(out)
        ref = model.head(model.norm(torch.cat(outs, dim=1)))
    assert logits.shape == (B, T, cfg.model.vocab_size)
    assert torch.allclose(logits, ref, atol=1e-5), (logits - ref).abs().max()
    assert torch.allclose(new_state, h, atol=1e-5)
    assert "state_abs_max" in info
    # reset_state zeros the masked rows only
    mask = torch.tensor([True, False, True, False])
    reset_state = model.reset_state(new_state, mask)
    assert not reset_state[:, mask].any() and torch.equal(
        reset_state[:, ~mask], new_state[:, ~mask]
    )
    # a window whose row 2 resets at its last position: the carried state is zero for that row
    # except for the one position processed after the reset
    assert new_state.shape == (2, B, 64)
    with pytest.raises(ValueError, match="shape"):
        model(w["tokens"], w["context"], torch.zeros(1, B, 64), w["reset"])
    with pytest.raises(ValueError, match="context"):
        model(w["tokens"], w["context"][:, :3], state, w["reset"])


def test_p0_5_transformer_mask_blocks_cross_document_attention() -> None:
    cfg = _config("model_name=transformer")
    model = _build(cfg).eval()
    assert isinstance(model, TransformerBaseline)
    B, T = 4, UNROLL
    w = _random_window(cfg, B, T)
    assert model.init_state(B, "cpu") is None
    assert model.reset_state(None, torch.zeros(B, dtype=torch.bool)) is None
    with torch.no_grad():
        logits, state, _ = model(w["tokens"], w["context"], None, w["reset"])
        # change every token of row 0's first document (positions 0..4): the second document
        # (positions 5..) must be unaffected, and causality keeps later changes from earlier logits
        tokens = w["tokens"].clone()
        tokens[0, :5] = (tokens[0, :5] + 7) % cfg.model.vocab_size
        tokens[0, :5].clamp_(min=1)
        changed, _, _ = model(tokens, w["context"], None, w["reset"])
        tokens2 = w["tokens"].clone()
        tokens2[0, 20:] = 3
        later, _, _ = model(tokens2, w["context"], None, w["reset"])
    assert state is None and logits.shape == (B, T, cfg.model.vocab_size)
    assert torch.allclose(logits[0, 5:], changed[0, 5:], atol=1e-6)
    assert not torch.allclose(logits[0, :5], changed[0, :5], atol=1e-3)
    assert torch.allclose(logits[1:], changed[1:], atol=1e-6)  # other rows untouched
    assert torch.allclose(logits[0, :20], later[0, :20], atol=1e-6)  # causal
    # the attention mask itself: block-diagonal over documents and causal
    mask = TransformerBaseline.attention_mask(w["reset"])
    assert mask.shape == (B, 1, T, T) and mask.dtype is torch.bool
    assert mask[0, 0, 4, 0] and not mask[0, 0, 5, 4] and mask[0, 0, 6, 5] and not mask[0, 0, 3, 4]
    assert mask[3, 0].sum() == T * (T + 1) // 2  # no reset after position 0: plain causal
    # a window longer than train.unroll has no positions for it: it never attends across windows
    with pytest.raises(ValueError, match="context window"):
        model(
            torch.zeros(B, T + 1, dtype=torch.long),
            torch.zeros(B, T + 1, dtype=torch.long),
            None,
            torch.zeros(B, T + 1, dtype=torch.bool),
        )


# --------------------------------------------------------------------------------------------------
# Stream loader, evaluation, checkpoints, AMP, search budget
# --------------------------------------------------------------------------------------------------


def test_p0_5_stream_loader_matches_collate_and_resumes(train_docs: list[Example]) -> None:
    ds = ByteDataset(train_docs)
    B = 4
    # aligned documents that fit one window: the lazy loader equals collate_windows + iter_windows
    loader = StreamLoader(
        ds, batch=B, unroll=UNROLL, seed=0, shuffle=False, align_docs=True, epochs=1
    )
    windows = list(loader)
    reference = list(iter_windows(collate_windows(train_docs, UNROLL, B, align_docs=True), UNROLL))
    assert len(windows) == len(reference) == N_DOCS // B
    for got, want in zip(windows, reference, strict=True):
        for key in want:
            assert torch.equal(got[key], want[key]), key
    # unaligned: every document appears exactly once and completely, resets exactly at starts
    loader = StreamLoader(
        ds, batch=B, unroll=UNROLL, seed=0, shuffle=False, align_docs=False, epochs=1
    )
    windows = list(loader)
    tokens = torch.cat([w["tokens"] for w in windows], dim=1)
    doc = torch.cat([w["doc"] for w in windows], dim=1)
    reset = torch.cat([w["reset"] for w in windows], dim=1)
    for i, example in enumerate(train_docs):
        rows, cols = (doc == i).nonzero(as_tuple=True)
        assert rows.unique().numel() == 1 and cols.numel() == example.length
        assert torch.equal(tokens[rows[0], cols], torch.from_numpy(example.tokens.copy()).long())
        assert reset[rows[0], cols[0]] and not reset[rows[0], cols[1:]].any()
    assert int((doc >= 0).sum()) == sum(e.length for e in train_docs)
    assert int(reset.sum()) == N_DOCS and (tokens[doc < 0] == PAD).all()
    # infinite, shuffled: epochs use different permutations and every epoch covers every document
    loader = StreamLoader(
        ds, batch=B, unroll=UNROLL, seed=3, shuffle=True, align_docs=True, epochs=None
    )
    first = [next(loader)["doc"][:, 0].tolist() for _ in range(N_DOCS // B)]
    second = [next(loader)["doc"][:, 0].tolist() for _ in range(N_DOCS // B)]
    assert sorted(sum(first, [])) == list(range(N_DOCS)) == sorted(sum(second, []))
    assert first != second and loader.epoch == 1
    # resume: a fresh loader restored from state_dict continues with the same windows
    state = loader.state_dict()
    continued = [next(loader) for _ in range(5)]
    again = StreamLoader(
        ds, batch=B, unroll=UNROLL, seed=3, shuffle=True, align_docs=True, epochs=None
    )
    again.load_state_dict(state)
    for got, want in zip((next(again) for _ in range(5)), continued, strict=True):
        for key in want:
            assert torch.equal(got[key], want[key]), key
    with pytest.raises(ValueError, match="rows"):
        StreamLoader(
            ds, batch=2, unroll=UNROLL, seed=3, shuffle=True, align_docs=True, epochs=None
        ).load_state_dict(state)


def test_p0_5_evaluate_groups_and_refuses_test_split(
    train_docs: list[Example], tmp_path: Path
) -> None:
    cfg = _config("model_name=gru", batch=4)
    model = _build(cfg)
    trainer = Trainer(model, cfg, _cpu(), run_dir=tmp_path)
    other = _docs(_tiny_task("kv"), "dev", 8)
    mixed = ByteDataset([*train_docs[:8], *other])
    result = trainer.evaluate(mixed)
    assert set(result["per_group"]) == {"algorithmic/in_range", "kv/in_range"}
    counts = {k: v["n_positions"] for k, v in result["per_group"].items()}
    assert counts["algorithmic/in_range"] == sum(int(d.loss_mask.sum()) for d in train_docs[:8])
    assert counts["kv/in_range"] == sum(int(d.loss_mask.sum()) for d in other)
    assert result["n_positions"] == sum(counts.values()) and result["n_docs"] == 16
    assert 0.0 <= result["accuracy"] <= 1.0 and result["loss"] > 0
    assert result["n_tokens"] == sum(d.length for d in mixed.examples)
    # metrics are pluggable and averaged per position
    custom = trainer.evaluate(mixed, {"bpb": bits_per_byte, "nll": nll, "acc": token_accuracy})
    assert custom["bpb"] == pytest.approx(custom["nll"] / np.log(2))
    assert custom["acc"] == result["accuracy"]
    # the test split is refused unless a gate script says so
    test_docs = _docs(_tiny_task(), "test", 4)
    with pytest.raises(ValueError, match="test split"):
        trainer.evaluate(ByteDataset(test_docs))
    with pytest.raises(ValueError, match="test split"):
        trainer.fit(ByteDataset(train_docs), ByteDataset(test_docs), 1)
    assert trainer.evaluate(ByteDataset(test_docs), allow_test=True)["n_docs"] == 4
    # the accumulator checks its inputs
    acc = MetricAccumulator(DEFAULT_METRICS, ["a"])
    with pytest.raises(ValueError, match="group_ids"):
        acc.update(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError):
        MetricAccumulator({})
    # evaluate_windows carries state across windows and reports counts
    loader = StreamLoader(
        mixed, batch=4, unroll=UNROLL, seed=0, shuffle=False, align_docs=True, epochs=1
    )
    direct = evaluate_windows(model, loader, DEFAULT_METRICS, device=torch.device("cpu"))
    assert direct["n_positions"] == result["n_positions"] and direct["n_windows"] == 4
    assert direct["accuracy"] == pytest.approx(result["accuracy"])


def test_p0_5_checkpoint_roundtrip(train_docs: list[Example], tmp_path: Path) -> None:
    cfg = _config("model_name=transformer")
    model = _build(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)
    window = _random_window(cfg, 2, UNROLL)
    logits, _, _ = model(window["tokens"], window["context"], None, window["reset"])
    logits.float().mean().backward()
    optimizer.step()
    scheduler.step()
    seed_everything(5)
    random.random(), np.random.random(), torch.rand(1)  # advance every generator
    rng = capture_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(1)))
    path = tmp_path / "step_00000003.pt"
    save_checkpoint(
        path, model, optimizer, scheduler, 3, rng, {"epoch": 0, "cursor": 2}, extra={"k": 1}
    )
    assert latest_checkpoint(tmp_path) == path and not list(tmp_path.glob("*.tmp"))

    fresh = _build(cfg, seed=9)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    fresh_sched = torch.optim.lr_scheduler.LambdaLR(fresh_opt, lambda step: 0.5)
    loaded = load_checkpoint(path, fresh, fresh_opt, fresh_sched)
    assert loaded["step"] == 3 and loaded["loader_state"] == {"epoch": 0, "cursor": 2}
    assert loaded["extra"] == {"k": 1} and loaded["format"] == 1
    for key, value in model.state_dict().items():
        assert torch.equal(value, fresh.state_dict()[key])
    for index, entry in optimizer.state_dict()["state"].items():
        for key, value in entry.items():
            other = fresh_opt.state_dict()["state"][index][key]
            assert torch.equal(value, other) if isinstance(value, Tensor) else value == other
    assert fresh_sched.state_dict()["last_epoch"] == 1
    restore_rng_state(loaded["rng_state"])
    assert (random.random(), float(np.random.random()), float(torch.rand(1))) == expected
    # strict loading: a model of another size is refused; a missing file too
    with pytest.raises(RuntimeError):
        load_checkpoint(
            path, _build(_config("model_name=transformer", "baseline.transformer.d_model=32"))
        )
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", fresh)
    with pytest.raises(ValueError, match="optimizer"):
        save_checkpoint(tmp_path / "no_opt.pt", model, None, None, 1, rng, None)
        load_checkpoint(tmp_path / "no_opt.pt", fresh, fresh_opt)
    assert (
        detach_state({"a": (torch.ones(1).requires_grad_(), None)})["a"][0].requires_grad is False
    )


def test_p0_5_autocast_policy_on_cpu() -> None:
    ctx = _cpu()
    assert ctx.autocast_dtype is None and autocast_name(ctx) == "none"
    assert make_grad_scaler(ctx) is None
    with autocast_ctx(ctx):
        assert not torch.is_autocast_enabled("cpu")
    # the policy of the GPU profiles, resolved without a device: bf16 gets no scaler, fp16 one
    bf16 = DeviceContext(
        device=torch.device("cuda"),
        autocast_dtype=torch.bfloat16,
        master_dtype=torch.float32,
        supports_bf16=True,
        unified_memory=False,
        vram_bytes=1,
        profile=CUDA_PROFILE,
    )
    fp16 = DeviceContext(
        device=torch.device("mps"),
        autocast_dtype=torch.float16,
        master_dtype=torch.float32,
        supports_bf16=False,
        unified_memory=True,
        vram_bytes=1,
        profile="m_series",
    )
    assert make_grad_scaler(bf16) is None and autocast_name(bf16) == "bf16"
    scaler = make_grad_scaler(fp16)
    assert isinstance(scaler, torch.amp.GradScaler) and autocast_name(fp16) == "fp16"


@pytest.mark.parametrize("profile", [cuda_param, mps_param])
def test_p0_5_amp_trains_on_gpu(profile: str, train_docs: list[Example], tmp_path: Path) -> None:
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
    ds = ByteDataset(train_docs)
    for model_name in ("gru", "transformer"):
        cfg = _config(
            f"model_name={model_name}",
            f"profile={profile}",
            "train.steps=6",
            "train.eval_every=3",
            "train.ckpt_every=6",
            "baseline.gru.n_layers=2",
            batch=4,
        )
        model = _build(cfg)
        trainer = Trainer(model, cfg, ctx, run_dir=tmp_path / model_name)
        assert (trainer.scaler is not None) == (ctx.autocast_dtype is torch.float16)
        history = trainer.fit(ds, ds, 6)
        assert history.autocast == autocast_name(ctx)
        assert np.isfinite([record.loss for record in history.steps]).all()
        assert next(model.parameters()).dtype is torch.float32  # master weights stay fp32


def test_p0_5_search_budget_counts_trials() -> None:
    cfg = _config("train.search_trials=2")
    budget = SearchBudget.from_config(cfg)
    assert budget.trials_allowed == 2 and not budget.exhausted() and budget.remaining() == 2
    budget.record(cfg, {"dev_accuracy": 0.5})
    trial = budget.record({"lr": 3e-3}, {"dev_accuracy": 0.7, "loss": 1.5})
    assert budget.exhausted() and budget.remaining() == 0 and trial.config == {"lr": 3e-3}
    assert budget.best("dev_accuracy", maximize=True) is trial
    assert budget.best("dev_accuracy", maximize=False).config == cfg.model_dump(mode="json")
    with pytest.raises(BudgetExhaustedError):
        budget.record({"lr": 1e-3}, {"dev_accuracy": 0.9})
    assert len(budget.trials) == 2 and '"trials_allowed": 2' in budget.to_json()
    fresh = SearchBudget(1)
    fresh.record(cfg, {"dev_accuracy": 0.5})
    with pytest.raises(BudgetExhaustedError):
        fresh.record(cfg, {"dev_accuracy": 0.5})
    twice = SearchBudget(3)
    twice.record(cfg, {"a": 1.0})
    with pytest.raises(ValueError, match="already"):
        twice.record(cfg, {"a": 2.0})
    with pytest.raises(ValueError):
        SearchBudget(0)
    with pytest.raises(ValueError, match="number"):
        SearchBudget(1).record({"x": 1}, {"ok": True})


def test_p0_5_trainer_guards(train_docs: list[Example], tmp_path: Path) -> None:
    ds = ByteDataset(train_docs)
    cfg = _config("model_name=gru", "train.micro_batch=4", batch=8)
    with pytest.raises(NotImplementedError, match="P4.3"):
        Trainer(_build(cfg), cfg, _cpu(), run_dir=tmp_path)
    cfg = _config("model_name=gru", "train.optimizer=lazy_adamw")
    with pytest.raises(ValueError, match="UnitBank"):  # P2.6: lazy Adam needs a unit bank
        Trainer(_build(cfg), cfg, _cpu(), run_dir=tmp_path)
    cfg = _config("model_name=gru", "train.schedule=step")
    with pytest.raises(ValueError, match="schedule"):
        Trainer(_build(cfg), cfg, _cpu(), run_dir=tmp_path)
    cfg = _config("model_name=gru", "train.steps=3", "train.eval_every=3", "train.ckpt_every=3")
    trainer = Trainer(_build(cfg), cfg, _cpu(), run_dir=tmp_path)
    with pytest.raises(ValueError, match="horizon"):
        trainer.fit(ds, ds, 4)
    with pytest.raises(TypeError, match="SequenceModel"):
        Trainer(nn.Linear(2, 2), cfg, _cpu(), run_dir=tmp_path)
    with pytest.raises(TypeError, match="segment"):
        Trainer(_build(cfg), cfg, _cpu(), profiler=object(), run_dir=tmp_path)

    class Profiler:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def segment(self, name: str, tokens: int = 0) -> Any:
            self.calls.append((name, tokens))
            import contextlib

            return contextlib.nullcontext()

    profiler = Profiler()
    trainer = Trainer(_build(cfg), cfg, _cpu(), profiler=profiler, run_dir=tmp_path)
    history = trainer.fit(ds, ds, 3)
    names = {name for name, _ in profiler.calls}
    assert names == {"forward", "backward", "optimizer_step", "evaluate"}
    assert all(tokens == 8 * UNROLL for name, tokens in profiler.calls if name == "forward")
    assert len(history.steps) == 3 and history.steps[-1].lr == cfg.train.lr
