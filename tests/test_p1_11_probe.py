"""P1.11 tests: the batched 5(c) edit probe equals the audit's per-document probe.

The reference (:func:`edit_probe_reference`) is the audit's loop: one full document per forward at
B = 1. The batched probe truncates every row after its answer position and batches originals and
edits; on the CPU in fp64 the two agree exactly (same probes, top-2 margins within 1e-12), and in
fp32 they agree wherever the top-2 margin is clear of rounding (> 1e-4). Batching changes GEMM row
counts, so in fp32 a near-tie may flip on some BLAS; the smallest margin seen is printed so that a
future failure reads as a tie, not as a bug (audit note, 7 Nov 2026). kv documents, which share the
key and value alphabets, are never probed.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from hsn.config import RunConfig
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import QRY as SYN_QRY
from hsn.device import seed_everything
from hsn.experiments import probe
from hsn.runner import build_model
from p1_11_support import (
    SPEC,
    cpu_ctx,
    docs,
    tiny_gru_config,
    tiny_hsn_config,
    tiny_transformer_config,
)
from torch import nn

MODELS = {
    "hsn_dense": lambda: tiny_hsn_config(),
    "hsn_topk": lambda: tiny_hsn_config("model.routing=topk"),
    "gru": lambda: tiny_gru_config(),
    "transformer": lambda: tiny_transformer_config(),
}


def _dev() -> ByteDataset:
    # kv first, so that kv documents sit between delayed ones in dev order
    return ByteDataset(docs("kv", "dev", 8) + docs("delayed", "dev", 40) + docs("kv", "dev", 8))


def _model(cfg: RunConfig, dtype: torch.dtype) -> nn.Module:
    seed_everything(0)
    model = build_model(cfg).to(dtype)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():  # states and logits of order 1, so that answers differ between documents
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=gen, dtype=torch.float64).to(dtype) * 0.3)
    return model


def test_p1_11_probe_constants_and_outcomes() -> None:
    assert probe.QRY == SYN_QRY == 3
    assert probe.KEYS == range(64, 144) and probe.VALUES == range(144, 192)  # the audit's
    params = SPEC.task("delayed", "in_range").params
    assert min(lo for lo, _ in params["keys"].values()) == 64
    assert max(hi for _, hi in params["keys"].values()) == 144
    assert tuple(params["values"]) == (144, 192)

    def p(before: int, after: int, old: int, new: int) -> dict[str, Any]:
        return {
            "changed": before != after,
            "follows_new_value": after == new,
            "original_correct": before == old,
            "argmax_original": before,
        }

    probes = [
        p(182, 182, 150, 182),
        p(182, 182, 150, 160),
        p(150, 160, 150, 160),
        p(150, 170, 150, 160),
    ]
    assert probe.outcome_counts(probes) == {"follow": 1, "other": 1, "unchanged": 2}
    s = probe.summarize(probes)
    assert s["correct_before_edit"] == {"n": 2, "follow": 1, "other": 1, "unchanged": 0}
    assert s["verdict"] == "reads state" and s["correct_follow_frac"] == 0.5
    assert s["n_probed"] == 4 and s["distinct_first_query_answers"] == 2
    assert probe.verdict(0, 0, 0) == "no correct answers to probe"
    assert probe.verdict(1, 0, 10) == "does not read state"
    assert probe.verdict(3, 0, 10) == "partial"
    assert probe.verdict(5, 2, 10) == "partial"  # follows enough, too many unchanged
    assert probe.absent_value(191, {144}) == 145
    with pytest.raises(ValueError, match="every value"):
        probe.absent_value(150, set(range(144, 192)))


def test_p1_11_probe_items_skip_kv_and_missing_queries() -> None:
    dev = _dev()
    items = probe.probe_items(dev)
    assert items, "the tiny delayed documents have probes"
    tasks = {dev.meta(it.index)["task"] for it in items}
    assert tasks == {"delayed"}
    kv_with_match = [
        i
        for i in range(len(dev))
        if dev.meta(i)["task"] == "kv"
        and probe.first_fact_and_query(dev.examples[i].tokens, dev.examples[i].loss_mask)
    ]
    assert kv_with_match, "kv documents would match the first-fact rule if they were probed"
    assert [it.index for it in items] == sorted(it.index for it in items)
    for it in items:
        assert it.tokens.dtype.name == "int64" and it.modified[it.answer_pos - 1] == probe.QRY
        assert int((it.tokens != it.modified).sum()) == 1


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["fp64", "fp32"])
def test_p1_11_batched_probe_matches_per_document(name: str, dtype: torch.dtype) -> None:
    model = _model(MODELS[name](), dtype)
    ctx = cpu_ctx()
    dev = _dev()
    ref = probe.edit_probe_reference(model, ctx, dev, margins=True)
    out = probe.edit_probe(model, ctx, dev, batch_rows=8, margins=True)  # several chunks of T
    assert model.training, "the probe restores the training mode"
    assert len(out) == len(ref) > 10
    keys = [k for k in ref[0] if not k.startswith("margin_")]
    margins = [min(r["margin_original"], r["margin_modified"]) for r in ref]
    if dtype == torch.float64:
        for a, b in zip(ref, out, strict=True):
            assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
            assert abs(a["margin_original"] - b["margin_original"]) <= 1e-12
            assert abs(a["margin_modified"] - b["margin_modified"]) <= 1e-12
    else:
        ties = 0
        for a, b, m in zip(ref, out, margins, strict=True):
            if m > 1e-4:
                assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
            else:
                ties += 1
        print(f"{name} fp32: smallest top-2 margin {min(margins):.3e}, {ties} near-ties")
    assert len({r["argmax_original"] for r in out}) > 1, "the probe sees different answers"


def test_p1_11_probe_refuses_random_routing() -> None:
    model = _model(tiny_hsn_config("model.routing=random"), torch.float64)
    with pytest.raises(ValueError, match="routing=random"):
        probe.edit_probe(model, cpu_ctx(), _dev())
    with pytest.raises(ValueError, match="even number"):
        probe.edit_probe(_model(tiny_gru_config(), torch.float64), cpu_ctx(), _dev(), batch_rows=3)
