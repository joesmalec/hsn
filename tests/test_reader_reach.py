"""``scripts/reader_reach.py``: the loss by window position and the attention's reach of a
stateless model (the text-gap measurements of ``docs/hsn_t.md``, "The reader's positions").

Checked against values known without the script: the summed loss is the plain cross-entropy
of the same windows; a reader with a span attends nothing beyond it; a block with zero queries
attends uniformly, whose far share has a closed form; a run directory loads back the weights
it holds."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F
from hsn.config import dump_resolved, load_config
from hsn.device import seed_everything
from hsn.runner import build_model
from hsn.train.checkpoint import capture_rng_state, checkpoint_path, save_checkpoint
from hsn.train.loop import StreamLoader
from p1_15_support import datasets
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
NOMEM = REPO / "configs" / "experiments" / "hsn_t_nomem.yaml"
READER_LM = REPO / "configs" / "baselines" / "reader_w64l2.yaml"
TRANSFORMER = REPO / "configs" / "baselines" / "transformer_2m.yaml"
GRU = REPO / "configs" / "baselines" / "gru_2m.yaml"
T = 24
SMALL = [f"train.unroll={T}", "train.batch=4", "train.micro_batch=4"]
READER = ["model.reader_width=16", "model.reader_heads=2"]
CASES = {
    "transformer": (
        TRANSFORMER,
        [
            "baseline.transformer.d_model=16",
            "baseline.transformer.n_heads=2",
            "baseline.transformer.n_layers=2",
            "baseline.transformer.ffn_hidden=32",
        ],
    ),  # fmt: skip
    "reader": (READER_LM, READER),
    "learned": (READER_LM, [*READER, "model.reader_position=learned"]),
    "nomem": (NOMEM, [*READER, "model.d=8", "model.context_dim=8", "model.hidden=16"]),
}


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reader_reach", REPO / "scripts/reader_reach.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["reader_reach"] = module
    spec.loader.exec_module(module)
    return module


def _windows() -> list[dict[str, Tensor]]:
    _, dev = datasets()
    loader = StreamLoader(dev, batch=4, unroll=T, seed=0, shuffle=False, align_docs=True, epochs=1)
    return list(loader)


def _model(name: str, *extra: str) -> torch.nn.Module:
    path, overrides = CASES[name]
    seed_everything(0)
    return build_model(load_config(path, [*SMALL, *overrides, *extra])).eval()


@pytest.mark.parametrize("name", list(CASES))
def test_position_loss_sums_to_the_cross_entropy(script: ModuleType, name: str) -> None:
    model, windows = _model(name), _windows()
    nll, count = script.position_loss(model, windows)
    assert nll.shape == count.shape == (T,)
    total = positions = 0.0
    with torch.no_grad():
        for batch in windows:
            logits, _, _ = model(batch["tokens"], batch["context"], None, batch["reset"])
            mask = batch["loss_mask"]
            total += float(F.cross_entropy(logits[mask], batch["targets"][mask], reduction="sum"))
            positions += float(mask.sum())
    assert positions > 0 and float(count.sum()) == positions
    assert float(nll.sum()) == pytest.approx(total, rel=1e-5)


def test_attention_reach_of_a_span_and_of_uniform_attention(script: ModuleType) -> None:
    windows = _windows()
    # a reader that reads the last four bytes only puts nothing four or more positions back
    short = _model("reader", "model.reader_span=4")
    reach = script.attention_reach(short, windows, far=4, first=8)
    assert reach.shape == (2, 2) and float(reach.abs().max()) == 0.0
    assert float(script.attention_reach(short, windows, far=1, first=8).min()) > 0.0
    # zero queries and keys: every head attends uniformly over the bytes of its document up to
    # the query, so the far share is the fraction of those bytes at the distance or beyond
    for name in ("transformer", "reader", "learned", "nomem"):
        model = _model(name)
        blocks = script.attention_blocks(model)
        with torch.no_grad():
            for block in blocks:
                width = block.qkv.weight.shape[1]
                block.qkv.weight[: 2 * width] = 0.0
                block.qkv.bias[: 2 * width] = 0.0
        far, first = 5, 10
        expected = queries = 0.0
        for batch in windows:
            start = torch.zeros_like(batch["tokens"])
            for t in range(1, T):  # the first position of the document a position belongs to
                start[:, t] = torch.where(batch["reset"][:, t], t, start[:, t - 1])
            index = torch.arange(T)[None, :].expand_as(start)
            seen = index - start + 1  # the keys a query may read
            share = (seen - far).clamp(min=0) / seen
            chosen = batch["loss_mask"] & (index >= first)
            expected += float(share[chosen].sum())
            queries += float(chosen.sum())
        reach = script.attention_reach(model, windows, far=far, first=first)
        assert reach.shape == (len(blocks), 2)
        assert torch.allclose(reach, torch.full_like(reach, expected / queries), atol=1e-5), name


@pytest.mark.parametrize("name", list(CASES))
def test_reach_is_bit_identical_with_boolean_or_additive_masks(
    script: ModuleType, name: str
) -> None:
    model, windows = _model(name), _windows()
    reaches = []
    for additive in (False, True):

        def mask_form(_block, args, additive=additive):
            x, mask, *rotation = args
            allowed = mask if mask.dtype == torch.bool else mask.isfinite()
            mask = (
                torch.zeros_like(mask, dtype=x.dtype).masked_fill(~allowed, -torch.inf)
                if additive
                else allowed
            )
            return x, mask, *rotation

        hooks = [
            block.register_forward_pre_hook(mask_form) for block in script.attention_blocks(model)
        ]
        try:
            reaches.append(script.attention_reach(model, windows, far=5, first=10))
        finally:
            for hook in hooks:
                hook.remove()
    assert torch.equal(*reaches)


def test_attention_weights_apply_finite_additive_bias(script: ModuleType) -> None:
    block = script.attention_blocks(_model("reader"))[0]
    with torch.no_grad():
        block.qkv.weight.zero_()
        block.qkv.bias.zero_()
        bias = torch.tensor([0.0, math.log(2.0), -torch.inf])[None, None, None, :]
        weights = script._attention_weights(block, torch.zeros(2, 3, 16), bias)
    expected = torch.tensor([1 / 3, 2 / 3, 0.0]).expand_as(weights)
    torch.testing.assert_close(weights, expected, rtol=0, atol=torch.finfo(weights.dtype).eps)


def test_models_with_state_are_refused(script: ModuleType) -> None:
    gru = build_model(load_config(GRU, SMALL))
    with pytest.raises(TypeError, match="not a stateless model"):
        script.position_loss(gru, _windows())


def test_a_run_directory_loads_and_reports(script: ModuleType, tmp_path: Path) -> None:
    path, overrides = CASES["nomem"]
    cfg = load_config(path, [*SMALL, *overrides, "model.reader_rotary_pairs=2"])
    seed_everything(0)
    model = build_model(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    dump_resolved(cfg, tmp_path / "config.resolved.yaml")
    save_checkpoint(checkpoint_path(tmp_path, 7), model, None, None, 7, capture_rng_state(), None)
    loaded, loaded_cfg, checkpoint = script.load_run(tmp_path)
    assert checkpoint.name == "step_00000007.pt" and loaded_cfg.model.reader_rotary_pairs == 2
    for (name, p), (_, q) in zip(model.named_parameters(), loaded.named_parameters(), strict=True):
        assert torch.equal(p, q), name
    args = argparse.Namespace(attention=True, far=4, first=8)
    text = script.report("run", loaded, _windows(), args)
    assert text.startswith("run: ") and "bits per byte" in text
    assert "positions   0 to   7" in text and "block 2:" in text
    with pytest.raises(FileNotFoundError, match="no step_"):
        dump_resolved(cfg, tmp_path / "empty" / "config.resolved.yaml")
        script.load_run(tmp_path / "empty")
