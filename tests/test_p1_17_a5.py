"""A5: byte addresses and selected-cell reads, without changing the state or data protocol."""

from __future__ import annotations

import dataclasses
import math
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from cards import CUDA_PROFILE
from hsn.config import RunConfig, config_hash, load_config, load_hardware_profile
from hsn.data.dataset import ByteDataset, collate_windows
from hsn.data.synthetic import BOS, TaskSpec, generate, load_dataset_spec
from hsn.device import resolve, seed_everything, set_deterministic
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from hsn.train import compile as compile_mod
from hsn.train.compile import compile_step, compile_window
from hsn.train.memory_plan import expected_param_count, macs_per_token
from hsn.train.tbptt import TBPTTTrainer
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "configs" / "ref_a1.yaml"


def _cfg(*extra: str) -> RunConfig:
    return load_config(
        REF,
        overrides=[
            "profile=cpu",
            "model.d=8",
            "model.context_dim=8",
            "model.n_cells=6",
            "model.in_degree=2",
            "model.n_units=3",
            "model.k_select=2",
            "model.hidden=16",
            "model.scorer_hidden=8",
            "model.routing=topk",
            "train.gate_mode=ste",
            "train.unroll=8",
            "train.batch=2",
            "train.micro_batch=2",
            "train.steps=4",
            "train.warmup_steps=2",
            "train.eval_every=4",
            "train.ckpt_every=1",
            "train.stats_every=4",
            *extra,
        ],
    )


def _model(cfg: RunConfig) -> HSN:
    seed_everything(cfg.seed)
    return HSN.from_config(cfg)


@pytest.fixture(autouse=True)
def _restore_runtime() -> Iterator[None]:
    deterministic = torch.are_deterministic_algorithms_enabled()
    yield
    set_deterministic(deterministic)
    torch._dynamo.reset()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        from torch._inductor.cudagraph_trees import reset_cudagraph_trees

        reset_cudagraph_trees()


def _inputs(model: HSN) -> tuple[Tensor, Tensor, CellGraphState, Tensor]:
    # Row 0 uses a held-out dev key; row 1 a training key. Both repeat at write and query.
    tokens = torch.tensor(
        [[BOS, 112, 144, 192, 3, 112, 144, 2], [BOS, 75, 148, 193, 3, 75, 148, 2]]
    )
    ctx = torch.full_like(tokens, 2)
    reset = torch.zeros_like(tokens, dtype=torch.bool)
    reset[:, 0] = True
    return tokens, ctx, model.init_state(2), reset


def _assert_state(got: CellGraphState, expected: CellGraphState) -> None:
    for name in ("h", "q"):
        torch.testing.assert_close(
            getattr(got, name), getattr(expected, name), atol=2e-6, rtol=2e-5
        )
    for name in ("age", "pos"):
        assert torch.equal(getattr(got, name), getattr(expected, name))


@pytest.mark.parametrize("scorer", ["linear", "mlp"])
@pytest.mark.parametrize(
    "address,readout", [(False, False), (True, False), (False, True), (True, True)]
)
def test_p1_17_a5_initialization_and_accounting(scorer: str, address: bool, readout: bool) -> None:
    """Each independent lever adds only its advertised columns and preserves every old weight."""
    base_cfg = _cfg(f"model.scorer={scorer}")
    base = _model(base_cfg)
    after_base = torch.rand(8)
    cfg = _cfg(
        f"model.scorer={scorer}",
        f"model.key_address={str(address).lower()}",
        f"model.selected_readout={str(readout).lower()}",
    )
    model = _model(cfg)
    assert torch.equal(torch.rand(8), after_base), "A5 consumed the existing initialization RNG"
    for name, old in base.named_parameters():
        new = dict(model.named_parameters())[name]
        if old.shape == new.shape:
            assert torch.equal(new, old), name
        else:
            assert new.ndim == old.ndim == 2 and new.shape[0] == old.shape[0]
            assert torch.equal(new[:, : old.shape[1]], old), name
    if not address and not readout:
        assert config_hash(cfg) == config_hash(base_cfg)
        assert set(model.state_dict()) == set(base.state_dict())
    report, original = model.param_report(), base.param_report()
    scorer_delta = (
        2 * (cfg.model.n_units if scorer == "linear" else cfg.model.scorer_hidden) if address else 0
    )
    decoder_delta = cfg.model.d * cfg.model.hidden if readout else 0
    assert report["scorer"] - original["scorer"] == scorer_delta
    assert report["decoder"] - original["decoder"] == decoder_delta
    assert report["total_trainable"] - original["total_trainable"] == scorer_delta + decoder_delta
    calculated = expected_param_count(cfg.model)
    assert all(report[name] == count for name, count in calculated.items())
    costs, previous_costs = macs_per_token(cfg.model), macs_per_token(base_cfg.model)
    address_cost = 2 * cfg.model.n_cells * cfg.model.d if address else 0
    assert costs["sparse_total"] - previous_costs["sparse_total"] == (
        decoder_delta + cfg.model.n_cells * scorer_delta + address_cost
    )


@pytest.mark.parametrize("flag", ["key_address", "selected_readout"])
def test_p1_17_a5_refuses_dense_and_baselines(flag: str) -> None:
    with pytest.raises(ValueError, match="dense|routed"):
        _cfg("model.routing=dense", f"model.{flag}=true")
    with pytest.raises(ValueError, match="hsn|HSN|baseline"):
        _cfg("model_name=gru", f"model.{flag}=true")


def test_p1_17_a5_codes_are_fixed_seeded_checkpoint_buffers() -> None:
    cfg = _cfg("model.key_address=true")
    first = _model(cfg)
    torch.manual_seed(999)
    second = HSN.from_config(cfg)  # codes depend on run.seed, not the parameter RNG
    other = _model(_cfg("model.key_address=true", "seed=1"))
    for name, shape in (("byte_codes", (256, 8)), ("cell_addresses", (6, 8))):
        value = getattr(first.scorer, name)
        assert value.shape == shape and not value.requires_grad
        assert name in dict(first.scorer.named_buffers())
        assert name not in dict(first.scorer.named_parameters())
        assert f"router.scorer.{name}" in first.state_dict()
        assert torch.equal(value, getattr(second.scorer, name))
        assert not torch.equal(value, getattr(other.scorer, name))
    other.load_state_dict(first.state_dict())
    assert torch.equal(first.scorer.byte_codes, other.scorer.byte_codes)
    assert torch.equal(first.scorer.cell_addresses, other.scorer.cell_addresses)
    assert tuple(field.name for field in dataclasses.fields(first.init_state(2))) == (
        "h",
        "q",
        "age",
        "pos",
    )


def test_p1_17_a5_codes_identical_across_processes() -> None:
    """Python's hash seed and ambient torch RNG must not change a run's address dictionary."""
    script = """
import hashlib
import os
import torch
from hsn.config import ModelConfig
from hsn.model.router import Scorer
torch.manual_seed(int(os.environ['PYTHONHASHSEED']))
cfg = ModelConfig(d=8, context_dim=8, n_cells=6, in_degree=2, n_units=3,
                  k_select=2, hidden=16, routing='topk', key_address=True)
scorer = Scorer(cfg, unroll=8, seed=17)
raw = scorer.byte_codes.numpy().tobytes() + scorer.cell_addresses.numpy().tobytes()
print(hashlib.sha256(raw).hexdigest())
"""
    digests = []
    for seed in (23, 947):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO / "python"), "PYTHONHASHSEED": str(seed)},
            check=True,
            capture_output=True,
            text=True,
        )
        digests.append(result.stdout.strip())
    assert len(digests[0]) == 64 and digests[0] == digests[1]


def test_p1_17_a5_same_key_same_address_at_write_and_query_and_reset() -> None:
    model = _model(_cfg("model.key_address=true"))
    model.eval()
    tokens, ctx, state, reset = _inputs(model)
    captured: list[Tensor] = []
    hook = model.scorer.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach())
    )
    try:
        with torch.no_grad():
            model(tokens, ctx, state, reset, prev_tokens=torch.tensor([77, 99]))
    finally:
        hook.remove()
    assert len(captured) == tokens.shape[1]
    # At k v the value sees previous=k; at QRY k the query sees current=k.
    assert torch.equal(captured[2][..., -1], captured[5][..., -2])
    assert torch.equal(captured[0][..., -1], captured[0][..., -2])  # BOS at reset
    assert not torch.equal(captured[1][0, :, -2], captured[1][1, :, -2])
    expected = (model.scorer.byte_codes[tokens[:, 1]] @ model.scorer.cell_addresses.T) * 8**0.5
    torch.testing.assert_close(captured[1][..., -2], expected)
    # Mid-window reset must replace the preceding byte too, independently in every row.
    reset[0, 3] = True
    captured.clear()
    hook = model.scorer.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach())
    )
    try:
        with torch.no_grad():
            model(tokens, ctx, state, reset)
    finally:
        hook.remove()
    assert torch.equal(captured[3][0, :, -1], captured[0][0, :, -2])
    assert torch.equal(captured[3][1, :, -1], captured[2][1, :, -2])


def test_p1_17_a5_feature_ablation_zeros_only_requested_addresses() -> None:
    base = _model(_cfg("model.key_address=true"))
    excluded = _model(_cfg("model.key_address=true", "model.scorer_features.exclude=[a_prev]"))
    tokens, ctx, state, _reset = _inputs(base)
    zeros = torch.zeros(2, base.cfg.d)
    kwargs = {"tokens": tokens[:, 2], "prev_tokens": tokens[:, 1]}
    reference = base.scorer.features(zeros, zeros, state, ctx[:, 0], 1.0, 0, **kwargs)
    got = excluded.scorer.features(zeros, zeros, state, ctx[:, 0], 1.0, 0, **kwargs)
    assert torch.equal(got[..., :-1], reference[..., :-1])
    assert torch.equal(got[..., -1], torch.zeros_like(got[..., -1]))
    assert bool(reference[..., -1].abs().sum() > 0)
    assert base.param_report() == excluded.param_report()


@pytest.mark.parametrize("address,readout", [(True, False), (False, True), (True, True)])
def test_p1_17_a5_full_and_split_windows_match(address: bool, readout: bool) -> None:
    model = _model(
        _cfg(
            f"model.key_address={str(address).lower()}",
            f"model.selected_readout={str(readout).lower()}",
        )
    )
    model.eval()
    tokens, ctx, state, reset = _inputs(model)
    with torch.no_grad():
        full, end, _ = model(tokens, ctx, state, reset)
        first, middle, _ = model(tokens[:, :2], ctx[:, :2], state, reset[:, :2])
        latter, split_end, _ = model(
            tokens[:, 2:], ctx[:, 2:], middle, reset[:, 2:], prev_tokens=tokens[:, 1]
        )
    torch.testing.assert_close(torch.cat([first, latter], dim=1), full, atol=2e-6, rtol=2e-5)
    _assert_state(split_end, end)


def test_p1_17_a5_readout_ignores_padding_and_preserves_ste_gradient() -> None:
    model = _model(_cfg("model.selected_readout=true"))
    h = torch.arange(2 * 6 * 8, dtype=torch.float32).reshape(2, 6, 8) / 100
    state = model.init_state(2).with_dense(h)
    scores = torch.tensor(
        [
            [
                [4.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        ]
    ).repeat(2, 1, 1)
    scores.requires_grad_(True)
    selection = model.router.select(scores, 1.0, 2, 1, "topk", gate="ste")
    # Exercise a padded slot aliasing cell 0, including its potentially large carried value.
    valid = selection.valid.clone()
    valid[0, 1] = False
    indices = selection.cell_idx.clone()
    indices[0, 1] = 0
    selection = dataclasses.replace(selection, valid=valid, cell_idx=indices)
    read = model._selected_readout(state, selection, "ste")
    torch.testing.assert_close(read[0], h[0, 0])
    torch.testing.assert_close(read[1], h[1, :2].mean(dim=0))
    # Invalid padding is deliberately unspecified by Selection, including out-of-range IDs.
    for padded_index in (-1, model.cfg.n_cells + 99):
        padded = indices.clone()
        padded[0, 1] = padded_index
        padded_selection = dataclasses.replace(selection, cell_idx=padded)
        for gate in ("ste", "hard"):
            torch.testing.assert_close(model._selected_readout(state, padded_selection, gate), read)
    read.sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert float(scores.grad.abs().sum()) > 0, "the selection's direct task-loss path was detached"
    fallback = dataclasses.replace(selection, dense_fallback=torch.tensor([True, False]))
    torch.testing.assert_close(model._selected_readout(state, fallback)[0], state.q[0])
    empty = dataclasses.replace(selection, valid=torch.zeros_like(valid))
    assert torch.equal(model._selected_readout(state, empty), torch.zeros_like(read))


def test_p1_17_a5_task_loss_reaches_address_columns_and_readout() -> None:
    model = _model(_cfg("model.key_address=true", "model.selected_readout=true"))
    tokens, ctx, state, reset = _inputs(model)
    logits, _, _ = model(tokens, ctx, state, reset)
    F.cross_entropy(logits[:, 5], torch.tensor([144, 148])).backward()
    assert model.scorer.net.weight.grad is not None
    address_grad = model.scorer.net.weight.grad[:, -2:]
    assert torch.isfinite(address_grad).all() and float(address_grad.abs().sum()) > 0
    read_grad = model.decoder.fc1.weight.grad[:, -model.cfg.d :]
    assert torch.isfinite(read_grad).all() and float(read_grad.abs().sum()) > 0
    assert model.scorer.byte_codes.grad is None and model.scorer.cell_addresses.grad is None


def test_p1_17_a5_soft_readout_uses_masked_cells() -> None:
    model = _model(_cfg("model.selected_readout=true", "train.gate_mode=soft"))
    h = torch.arange(2 * 6 * 8, dtype=torch.float32).reshape(2, 6, 8) / 100
    state = model.init_state(2).with_dense(h)
    logits = torch.randn(2, 6, 3, requires_grad=True)
    selection = model.router.select(logits, 1.0, 2, 1, "topk", gate="soft")
    expected = (h * selection.cell_mask[..., None]).sum(1)
    expected = expected / selection.cell_mask.sum(1, keepdim=True)
    actual = model._selected_readout(state, selection, "soft")
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert bool((logits.grad.abs().sum(-1) > 0).all())


def _datasets() -> tuple[ByteDataset, ByteDataset]:
    source = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
    task = TaskSpec.model_validate(
        {
            **source.task("delayed", "out_of_range").model_dump(),
            "n_train": 8,
            "n_dev": 2,
            "n_test": 2,
            "unroll": 8,
            "length_range": [24, 24],
            "dep_len_range": [16, 16],
        }
    )
    return (
        ByteDataset([generate(task, "train", i) for i in range(8)]),
        ByteDataset([generate(task, "dev", i) for i in range(2)]),
    )


def test_p1_17_a5_resume_preserves_previous_byte_mid_document(tmp_path: Path) -> None:
    cfg = _cfg("model.key_address=true", "model.selected_readout=true", "train.deterministic=true")
    train, dev = _datasets()
    ctx = resolve(cfg.hardware)
    original = TBPTTTrainer(_model(cfg), cfg, ctx, run_dir=tmp_path / "full")
    expected = original.fit(train, dev, 4)
    checkpoint = tmp_path / "full" / "step_00000001.pt"
    saved = torch.load(checkpoint, weights_only=True)
    previous = saved["extra"]["prev_tokens"]
    assert previous.shape == (2,) and previous.dtype == torch.int64
    assert all(offset > 0 for _doc, offset in saved["loader_state"]["rows"])
    for row, (doc, offset) in enumerate(saved["loader_state"]["rows"]):
        assert int(previous[row]) == int(train.examples[doc].tokens[offset - 1])
    resumed = TBPTTTrainer(_model(cfg), cfg, ctx, run_dir=tmp_path / "resumed")
    assert resumed.resume(checkpoint) == 1
    actual = resumed.fit(train, dev, 4)
    for left, right in zip(expected.steps[1:], actual.steps, strict=True):
        assert left.loss == right.loss or (math.isnan(left.loss) and math.isnan(right.loss))
        assert (left.total, left.grad_norm) == (right.total, right.grad_norm)
    for name, value in original.model.state_dict().items():
        other = resumed.model.state_dict()[name]
        assert torch.equal(value, other) if isinstance(value, Tensor) else value == other, name
    # An A5 checkpoint without its previous byte cannot reconstruct the resumed feature stream.
    del saved["extra"]["prev_tokens"]
    bad = tmp_path / "missing-byte.pt"
    torch.save(saved, bad)
    with pytest.raises(ValueError, match="prev|previous"):
        resumed.resume(bad)


def test_p1_17_a5_rollback_drops_previous_byte_with_document(tmp_path: Path) -> None:
    cfg = _cfg(
        "model.key_address=true",
        "model.selected_readout=true",
        "train.rollback_every=1",
        "train.rollback_skip=0",
    )
    train, dev = _datasets()
    trainer = TBPTTTrainer(_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path)

    def inject_nonfinite(current: TBPTTTrainer) -> None:
        if current.step == 2:
            grad = next(p.grad for p in current.model.parameters() if p.grad is not None)
            grad.fill_(float("inf"))

    trainer.grad_hook = inject_nonfinite
    trainer.fit(train, dev, 4)
    assert [r["step"] for r in trainer.rollbacks] == [3]
    checkpoint = torch.load(tmp_path / "step_00000003.pt", weights_only=True)
    assert checkpoint["extra"]["prev_tokens"] is None
    assert all(doc == -1 for doc, _offset in checkpoint["loader_state"]["rows"])
    # The following window begins a fresh document; no pre-rollback byte survives in the address.
    after = torch.load(tmp_path / "step_00000004.pt", weights_only=True)
    for row, (doc, offset) in enumerate(after["loader_state"]["rows"]):
        assert offset == cfg.train.unroll
        assert int(after["extra"]["prev_tokens"][row]) == int(
            train.examples[doc].tokens[offset - 1]
        )


def test_p1_17_a5_evaluation_carries_previous_byte(tmp_path: Path) -> None:
    cfg = _cfg("model.key_address=true", "model.selected_readout=true")
    model = _model(cfg)
    model.eval()
    _train, dev = _datasets()
    batch = collate_windows(dev.examples, 8, batch=2, align_docs=True)
    with torch.no_grad():
        logits, _, _ = model(batch["tokens"], batch["context"], model.init_state(2), batch["reset"])
        mask = batch["loss_mask"]
        expected_loss = F.cross_entropy(logits[mask], batch["targets"][mask]).item()
    trainer = TBPTTTrainer(model, cfg, resolve(cfg.hardware), run_dir=tmp_path)
    actual = trainer.evaluate(dev)
    assert actual["loss"] == pytest.approx(expected_loss, rel=2e-6, abs=2e-6)
    assert actual["n_tokens"] == sum(e.length for e in dev.examples)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("gate", ["ste", "soft"])
def test_p1_17_a5_compiled_window_matches_eager_with_carried_byte(
    training: bool, gate: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trace through two blocks, including a document reset and a non-BOS window boundary."""
    monkeypatch.setattr(compile_mod, "MAX_BLOCK", 4)
    cfg = _cfg("model.key_address=true", "model.selected_readout=true", f"train.gate_mode={gate}")
    model = _model(cfg)
    model.train(training)
    tokens, ctx, state, reset = _inputs(model)
    reset[:, 0] = False
    reset[0, 4] = True
    previous = torch.tensor([112, 75])
    compiled = compile_window(model, resolve(cfg.hardware), "reduce-overhead", backend="aot_eager")
    reference = model(tokens, ctx, state, reset, prev_tokens=previous)
    actual = compiled(tokens, ctx, state, reset, prev_tokens=previous)
    torch.testing.assert_close(actual[0], reference[0], atol=8e-6, rtol=8e-6)
    _assert_state(actual[1], reference[1])
    assert torch.equal(actual[2]["cell_idx"], reference[2]["cell_idx"])
    if training:
        params = list(model.parameters())
        left = torch.autograd.grad(reference[0].square().sum(), params, allow_unused=True)
        right = torch.autograd.grad(actual[0].square().sum(), params, allow_unused=True)
        for expected, got in zip(left, right, strict=True):
            if expected is None:
                assert got is None
            else:
                assert got is not None
                torch.testing.assert_close(got, expected, atol=1.2e-4, rtol=1.2e-4)


@pytest.mark.parametrize("mode", ["none", "reduce-overhead"])
def test_p1_17_a5_serving_carries_previous_byte_and_resets(mode: str) -> None:
    cfg = _cfg("model.key_address=true", "model.selected_readout=true")
    model = _model(cfg)
    model.eval()
    tokens, ctx, reference_state, reset = _inputs(model)
    reset[0, 4] = True
    serving = compile_step(model, resolve(cfg.hardware), mode, backend="aot_eager")
    served = model.init_state(2)
    previous = torch.tensor([112, 75])
    for position in range(tokens.shape[1]):
        reference_state = reference_state.reset(reset[:, position])
        served = served.reset(reset[:, position])
        with torch.no_grad():
            expected = model.step(
                tokens[:, position], ctx[:, position], reference_state, prev_tokens=previous
            )
        reference_state = expected.state
        actual = serving(tokens[:, position], ctx[:, position], served, prev_tokens=previous)
        torch.testing.assert_close(actual, expected.logits, atol=8e-6, rtol=8e-6)
        _assert_state(served, reference_state)
        previous = tokens[:, position]


@pytest.mark.cuda
@pytest.mark.parametrize("training", [False, True])
def test_p1_17_a5_compiled_cuda_matches_eager(
    training: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CUDA Graph warm-up, capture and replay of A5, with a carried byte and two blocks."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    monkeypatch.setattr(compile_mod, "MAX_BLOCK", 4)
    cfg = _cfg("model.key_address=true", "model.selected_readout=true")
    model = _model(cfg)
    tokens, context, _state, reset = _inputs(model)
    ctx = resolve(load_hardware_profile(CUDA_PROFILE))
    model.to(ctx.device).train(training)
    tokens, context, reset = (value.to(ctx.device) for value in (tokens, context, reset))
    generator = torch.Generator(device="cpu").manual_seed(19)
    h = torch.rand(2, cfg.model.n_cells, cfg.model.d, generator=generator) * 1.6 - 0.8
    state = model.init_state(2).with_dense(h.to(ctx.device))
    reset.zero_()
    reset[0, 4] = True
    previous = torch.tensor([112, 75], device=ctx.device)
    compiled = compile_window(model, ctx, "reduce-overhead", backend="inductor")
    reference = model(tokens, context, state, reset, prev_tokens=previous)
    for _ in range(3):
        actual = compiled(tokens, context, state, reset, prev_tokens=previous)
    torch.testing.assert_close(actual[0], reference[0], atol=1.5e-5, rtol=1.5e-5)
    for name in ("h", "q"):
        torch.testing.assert_close(
            getattr(actual[1], name), getattr(reference[1], name), atol=1.5e-5, rtol=1.5e-5
        )
    for name in ("age", "pos"):
        assert torch.equal(getattr(actual[1], name), getattr(reference[1], name))
    assert torch.equal(actual[2]["cell_idx"], reference[2]["cell_idx"])
    if training:
        params = list(model.parameters())
        expected_grads = torch.autograd.grad(reference[0].square().sum(), params, allow_unused=True)
        actual_grads = torch.autograd.grad(actual[0].square().sum(), params, allow_unused=True)
        for expected, got in zip(expected_grads, actual_grads, strict=True):
            if expected is None:
                assert got is None
            else:
                assert got is not None
                torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-4)


def test_address_init_starts_routes_on_the_byte_codes() -> None:
    """``model.address_init`` sets the scorer's two address columns for every unit (trainable as
    before), changes no other parameter, and makes the first routes follow ``a_cur + a_prev``."""
    base, on = (
        _cfg("model.key_address=true"),
        _cfg("model.key_address=true", "model.address_init=1.0"),
    )
    assert base.model.address_init is None and config_hash(on) != config_hash(base)
    from hsn.config import canonical_json

    assert "address_init" not in canonical_json(base)
    seed_everything(0)
    plain = HSN.from_config(base)
    seed_everything(0)
    model = HSN.from_config(on)
    weight = model.scorer.net.weight
    assert weight.requires_grad and torch.equal(weight[:, -2:], torch.ones(base.model.n_units, 2))
    for (name, a), (_, b) in zip(plain.named_parameters(), model.named_parameters(), strict=True):
        if name == "router.scorer.net.weight":
            assert torch.equal(a[:, :-2], b[:, :-2])
        else:
            assert torch.equal(a, b), name
    B, K = 5, base.model.k_select
    g = torch.Generator().manual_seed(3)
    tokens = torch.randint(0, 256, (B,), generator=g)
    previous = torch.randint(0, 256, (B,), generator=g)
    out = model.eval().step(
        tokens, torch.zeros(B, dtype=torch.int64), model.init_state(B), prev_tokens=previous
    )
    # a fresh state is a document start: the previous byte the scorer sees there is BOS
    start = torch.full_like(previous, BOS)
    address = model.scorer.address_features(tokens, start).sum(-1)  # [B, N]: a_cur + a_prev
    expected = address.topk(K, dim=1).indices.sort(dim=1).values
    assert out.selection is not None
    assert torch.equal(out.selection.cell_idx.sort(dim=1).values, expected)
    for bad in (
        ("model.address_init=1.0",),
        ("model.key_address=true", "model.scorer=mlp", "model.address_init=1.0"),
    ):
        with pytest.raises(ValueError, match="address_init"):
            _cfg(*bad)
