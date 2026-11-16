"""Bit-exact eager regressions against the reader at 8f9f184, plus graph guards."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from hsn.config import load_config
from hsn.model.reader import _attention_bias, _keep_tail
from hsn.runner import build_model
from reader_speed_reference import BeforeCarrySpeed
from test_reader_rotary_table import _Float64Ops

RECIPE = Path(__file__).resolve().parents[1] / "configs/experiments/hsn_t_nomem.yaml"


def _model(carry: int, *, full: bool = False, words: int = 0):
    shape = (
        [
            "model.reader_width=160",
            "model.reader_layers=4",
            "model.reader_rotary_pairs=[20,20,20,0]",
        ]
        if full
        else [
            "model.d=8",
            "model.context_dim=8",
            "model.hidden=16",
            "model.reader_width=16",
            "model.reader_heads=2",
            "model.reader_layers=3",
            "model.reader_rotary_pairs=[4,2,0]",
        ]
    )
    cfg = load_config(
        RECIPE, [*shape, f"model.reader_carry={carry}", f"model.reader_top_words={words}"]
    )
    torch.manual_seed(0)
    return build_model(cfg).train()


def _reference(model):
    old = copy.deepcopy(model)
    old.reader.__class__ = BeforeCarrySpeed
    old.reader._fill_rotary_table()
    return old


def _compare_stream(model, *, batch: int, window: int, autocast: bool = False):
    old = _reference(model)
    g = torch.Generator().manual_seed(37)
    states = [m.init_state(batch, "cpu") for m in (old, model)]
    for step in range(3):
        tokens = torch.randint(0, 256, (batch, window), generator=g)
        context = torch.randint(0, 16, tokens.shape, generator=g)
        reset = torch.zeros_like(tokens, dtype=torch.bool)
        if step == 0:
            reset[:, 0] = True
        if step == 2:
            reset[0, window // 2] = True
            reset[-1, 0] = True
        outputs, gradients = [], []
        for i, m in enumerate((old, model)):
            m.zero_grad(set_to_none=True)
            prior = states[i]
            before = tuple(t.clone() for t in prior) if prior is not None else None
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                out, states[i], _ = m(tokens, context, prior, reset)
                loss = out.float().square().mean()
            loss.backward()
            outputs.append(out.detach())
            gradients.append({n: p.grad.clone() for n, p in m.named_parameters()})
            if prior is not None:
                assert all(torch.equal(a, b) for a, b in zip(prior, before, strict=True))
        assert torch.equal(*outputs), "logits changed"
        assert all(torch.equal(gradients[0][n], gradients[1][n]) for n in gradients[0])
        if states[0] is not None:
            for a, b in zip(*states, strict=True):
                assert torch.equal(a, b), "cache contents changed"
                assert a.stride() == b.stride(), "cache layout changed"
                assert a.dtype == b.dtype
                assert b.grad_fn is None and not b.requires_grad
        assert old.state_dict().keys() == model.state_dict().keys()
    # No persistent cache/bias/table entered the checkpoint.
    assert not any("_rot_" in key or "_bias" in key for key in model.state_dict())


@pytest.mark.parametrize(
    ("carry", "window", "words"),
    [
        (0, 11, 0),
        (7, 7, 0),
        (13, 5, 0),
        (3, 11, 0),
        (13, 5, 4),
        (3, 11, 4),
        (1024, 7, 0),
        (1025, 9, 0),
        (7, 1025, 0),
    ],
)
@pytest.mark.parametrize("autocast", [False, True])
def test_logits_cache_and_gradients_are_bit_identical(carry, window, words, autocast):
    _compare_stream(_model(carry, words=words), batch=2, window=window, autocast=autocast)


@pytest.mark.parametrize("carry", [0, 1024])
def test_exact_campaign_shape_fp32_parity(carry):
    _compare_stream(_model(carry, full=True), batch=8, window=1024)


@pytest.mark.parametrize(("carry", "window"), [(3, 7), (7, 7), (11, 7)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tail_selection_is_old_cat_then_slice(carry, window, dtype):
    old = torch.randn(2, 3, carry, 8)
    new = torch.randn(2, 3, window, 8).to(dtype).requires_grad_()
    expected = torch.cat([old, new.to(old.dtype)], dim=2)[:, :, window:]
    got = _keep_tail(old, new, carry, dim=2)
    assert torch.equal(got, expected)
    assert got.grad_fn is None and not got.requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("autocast", [False, True])
def test_shared_bias_matches_bool_sdpa_outputs_and_gradients(dtype, autocast):
    g = torch.Generator().manual_seed(19)
    x = torch.randn(2, 7, 16, dtype=dtype, generator=g)
    layer = torch.nn.Linear(16, 48).to(dtype)
    mask = torch.rand(2, 1, 7, 7, generator=g) > 0.3
    mask[:, :, :, 0] = True
    outputs, gradients = [], []
    for shared in (False, True):
        layer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            q, k, v = [t.view(2, 7, 2, 8).transpose(1, 2) for t in layer(x).chunk(3, -1)]
            bias = _attention_bias(mask, x) if shared else mask
            if shared:
                assert bias.dtype == q.dtype
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            out.square().mean().backward()
        outputs.append(out.detach())
        gradients.append(layer.weight.grad.clone())
    assert torch.equal(*outputs)
    assert torch.equal(*gradients)


def test_carry_1024_uses_table_without_float64_or_trigonometry():
    model = _model(1024, full=True)
    old = _reference(model)
    tokens = torch.zeros(1, 1024, dtype=torch.long)
    reset = tokens.bool()
    with torch.no_grad(), _Float64Ops() as baseline:
        old.reader.read(tokens, reset, old.init_state(1, "cpu"))
    assert any("cos" in op for op in baseline.seen)
    with torch.no_grad(), _Float64Ops() as optimized:
        model.reader.read(tokens, reset, model.init_state(1, "cpu"))
    assert optimized.seen == []
    # Device/dtype moves rebuild the table, preserving the old float64-angle calculation.
    for dtype in (torch.float64, torch.float32):
        model.to(dtype)
        old.to(dtype)
        x = model.reader.emb(tokens)
        a = model.reader._rotation(1025, x)
        b = old.reader._rotation(1025, x)
        assert all(torch.equal(u, v) for u, v in zip(a, b, strict=True))


def test_carry_1024_fullgraph_reuses_graph_and_has_no_trigonometry():
    # Full target T/C with a small batch/width: shape guards and table limits are real.
    from torch._dynamo.utils import counters

    model = _model(1024)
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        return torch._dynamo.lookup_backend("aot_eager")(gm, inputs)

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    tokens = torch.zeros(2, 1024, dtype=torch.long)
    reset = tokens.bool()
    state = model.init_state(2, "cpu")
    before = dict(counters["graph_break"])
    try:
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            logits, state, _ = compiled(tokens, tokens, state, reset)
            logits.square().mean().backward()
        assert len(graphs) == 1
        assert dict(counters["graph_break"]) == before
        assert "float64" not in graphs[0].code
        assert ".cos()" not in graphs[0].code and ".sin()" not in graphs[0].code
    finally:
        torch._dynamo.reset()


@pytest.mark.cuda
@pytest.mark.parametrize("autocast", [False, True])
def test_cuda_carry_1024_matches_old_eager(autocast):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = _model(1024, full=True).cuda()
    old = _reference(model).cuda()
    tokens = torch.arange(1024, device="cuda")[None].expand(2, -1).remainder(256)
    context = torch.zeros_like(tokens)
    reset = context.bool()
    states = [m.init_state(2, "cuda") for m in (old, model)]
    for _ in range(2):
        results = []
        for i, m in enumerate((old, model)):
            m.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                out, states[i], _ = m(tokens, context, states[i], reset)
                loss = out.float().square().mean()
            loss.backward()
            results.append((out.detach(), [p.grad.clone() for p in m.parameters()]))
        assert torch.equal(results[0][0], results[1][0])
        assert all(torch.equal(a, b) for a, b in zip(results[0][1], results[1][1], strict=True))
        assert all(torch.equal(a, b) for a, b in zip(*states, strict=True))
