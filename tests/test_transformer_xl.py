"""The Transformer baseline as Transformer-XL (``baseline.transformer.position`` /
``baseline.transformer.carry``, ``docs/transformer_xl_baseline.md``; opt-in, 17 Nov 2026).

With ``position: rotary`` the queries and keys are rotated by their position (no position
table); with ``carry: M`` every block keeps the keys and values of the last ``M`` positions of the
windows before, detached, and every window position attends them and the window causally (its
own document only). These tests check that the defaults leave every configuration hash, memory
key, arm identity, parameter and output as they were; that a stream read window by window with
the cache is Transformer-XL over the whole stream (a direct computation written here, and a plain
causal reading of [previous window; window]); that the cache is a stop-gradient; that a document
reads nothing of the one before it; that the trainer carries the cache through training,
evaluation and a resume; and that the window compiles as one graph that equals eager.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from hsn.baselines.transformer import TransformerBaseline
from hsn.config import (
    HASH_OMIT_AT_DEFAULT,
    RunConfig,
    canonical_json,
    config_hash,
    load_config,
)
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import config_view, memory_key
from hsn.runner import build_model
from hsn.train.baseline_compile import CompiledBaseline
from hsn.train.loop import Trainer
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, train_overrides
from torch import Tensor, nn

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
TRANSFORMER = REPO / "configs" / "baselines" / "transformer_2m.yaml"
TXL = REPO / "configs" / "baselines" / "transformer_xl_2m.yaml"
GRU = REPO / "configs" / "baselines" / "gru_2m.yaml"
MAMBA = REPO / "configs" / "baselines" / "mamba_2m.yaml"
READER_LM = REPO / "configs" / "baselines" / "reader_w64l2.yaml"
NOMEM = REPO / "configs" / "experiments" / "hsn_t_nomem.yaml"
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"

#: three blocks of width 16, two heads of four rotary pairs each
TINY = (
    "baseline.transformer.d_model=16",
    "baseline.transformer.n_layers=3",
    "baseline.transformer.n_heads=2",
    "baseline.transformer.ffn_hidden=32",
)
ROTARY = "baseline.transformer.position=rotary"
PROTOCOL = {"name": "train_eval", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
#: TinyStories at batch 32 (the ts_v1 / final27k Transformer arms' overrides)
TS = (
    "profile=rtx3060",
    "hardware.compile_mode=none",
    "train.batch=32",
    "train.micro_batch=32",
    "data.dataset=tinystories_v1",
    "data.manifest=data/tinystories_v1/manifest.json",
    "data.tasks=[tinystories]",
    "data.variants=[out_of_range]",
)
#: config_hash and memory_key (variant none, PROTOCOL) at main 5941d13, before the keys
PINNED: dict[tuple[Path, tuple[str, ...]], tuple[str, str]] = {
    (TRANSFORMER, ()): (
        "65066a8915b79a328801b42780ae621f7185f23b553247a70c4f1b4dbd75071f",
        "24b0b90a5677b73a",
    ),
    (TRANSFORMER, TS): (
        "54859e9faf0915ea785499943e186d361a0622182832d26712c7ce3b19e7ee71",
        "19e0301fce365ce9",
    ),
    (TRANSFORMER, ("hardware.baseline_compile=reduce-overhead",)): (
        "7582ae3635d8234cfca052b71110fc95ad1b00c58e922278bc985ae18e65568f",
        "323f71a9cfe0dd2b",
    ),
    (GRU, ()): (
        "c8dcbaa48e3b27ecdde3d078359ab95f7482f49935d076bed34f521aeb88cf60",
        "914bccd12d3b08c0",
    ),
    (MAMBA, ()): (
        "9f3bab36efded64a82fcb4299ab35ba5a33b1d89def2b105e391109f882765d6",
        "932399457578460f",
    ),
    (NOMEM, ()): (
        "427e8944984cd16906758d1a311b56264140136b4d7618e0c993f0b5e3db6592",
        "7cd4b2df65516f34",
    ),
    (RECIPE_A6, ()): (
        "d8b36a25971c9694b25251b4a6f20545c617e6d2796b7e4d1ac544acc3e78e28",
        "3a556e053fb50f04",
    ),
}


def _gap(a: Tensor, b: Tensor) -> float:
    return float((a - b).detach().abs().max())


def _cfg(*overrides: str, unroll: int = 8) -> RunConfig:
    return load_config(TRANSFORMER, [*TINY, f"train.unroll={unroll}", *overrides])


def _model(
    *overrides: str, unroll: int = 8, dtype: torch.dtype = torch.float64
) -> TransformerBaseline:
    """A tiny model away from its start (every weight, norm and bias matters), in ``dtype``,
    in evaluation mode; the same weights for every ``overrides`` that keeps the parameters."""
    seed_everything(0)
    model = TransformerBaseline(_cfg(*overrides, unroll=unroll)).to(dtype)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=torch.float64).to(dtype))
    return model.eval()


def _stream(B: int, S: int, seed: int, resets: float = 0.05) -> tuple[Tensor, Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 256, (B, S), generator=g)
    context = torch.randint(0, 16, (B, S), generator=g)
    reset = torch.rand(B, S, generator=g) < resets
    reset[:, 0] = True
    return tokens, context, reset


def _by_windows(
    model: nn.Module, tokens: Tensor, context: Tensor, reset: Tensor, T: int
) -> tuple[Tensor, Any]:
    """The stream read window by window (the last one may be shorter), the state carried."""
    state = model.init_state(int(tokens.shape[0]), tokens.device)  # type: ignore[operator]
    outs = []
    for lo in range(0, int(tokens.shape[1]), T):
        sl = slice(lo, lo + T)
        out, state, info = model(tokens[:, sl], context[:, sl], state, reset[:, sl])
        assert info == {}
        outs.append(out)
    return torch.cat(outs, dim=1), state


def _xl_visible(reset: Tensor, T: int, M: int) -> Tensor:
    """Bool ``[B, S, S]``: Transformer-XL's reach over a stream read in windows of ``T`` with a
    memory of ``M``, written as loops: position ``a`` reads ``s`` when both are in one document
    and ``start(a) − M ≤ s ≤ a``, ``start(a)`` the first position of ``a``'s window."""
    B, S = reset.shape
    visible = torch.zeros(B, S, S, dtype=torch.bool)
    for b in range(B):
        start = 0
        for a in range(S):
            if bool(reset[b, a]):
                start = a
            for s in range(max(start, (a // T) * T - M), a + 1):
                visible[b, a, s] = True
    return visible


def _direct(
    model: TransformerBaseline,
    tokens: Tensor,
    context: Tensor,
    visible: Tensor,
    detach_before: int = 0,
) -> Tensor:
    """The rotary model over a whole stream in one pass, written out: each block's queries and
    keys rotated by their position in the stream (pairs ``i`` and ``i + half``), the attention
    over the keys ``visible [B, S, S]`` names, softmax, the residual MLP. ``detach_before``:
    the keys and values of the positions before it are detached (the cache's stop-gradient)."""
    B, S = tokens.shape
    d, H = model.d_model, model.n_heads
    half = d // H // 2
    freq = 10_000.0 ** (-torch.arange(half, dtype=torch.float64) / half)
    angles = torch.arange(S, dtype=torch.float64)[:, None] * freq
    cos, sin = angles.cos(), angles.sin()

    def rotate(x: Tensor) -> Tensor:
        a, b = x[..., :half], x[..., half:]
        return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)

    x = model.tok_emb(tokens) + model.ctx_emb(context)
    for block in model.blocks:
        q, k, v = block.qkv(block.norm_attn(x)).split(d, dim=-1)
        q, k, v = (t.view(B, S, H, -1).transpose(1, 2) for t in (q, k, v))
        if detach_before:
            k = torch.cat([k[:, :, :detach_before].detach(), k[:, :, detach_before:]], dim=2)
            v = torch.cat([v[:, :, :detach_before].detach(), v[:, :, detach_before:]], dim=2)
        q, k = rotate(q), rotate(k)
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        weights = scores.masked_fill(~visible[:, None], -math.inf).softmax(dim=-1)
        x = x + block.proj((weights @ v).transpose(1, 2).reshape(B, S, d))
        x = x + block.fc_out(F.gelu(block.fc_in(block.norm_mlp(x))))
    out: Tensor = model.head(model.norm(x))
    return out


def _old_forward(model: TransformerBaseline, tokens: Tensor, context: Tensor, reset: Tensor) -> Any:
    """The Transformer baseline's forward as it was before the keys (main 5941d13), verbatim
    but for the module attributes it reads."""
    B, T = tokens.shape
    assert model.pos_emb is not None
    positions = torch.arange(T, device=tokens.device)
    x = model.tok_emb(tokens) + model.ctx_emb(context) + model.pos_emb(positions)
    x = model.drop(x)
    segment = reset.to(torch.int64).cumsum(dim=1)
    same = segment.unsqueeze(2) == segment.unsqueeze(1)
    causal = torch.ones(T, T, dtype=torch.bool, device=reset.device).tril()
    mask = (same & causal).unsqueeze(1)
    for block in model.blocks:
        d = x.shape[-1]
        q, k, v = block.qkv(block.norm_attn(x)).split(d, dim=-1)
        q = q.view(B, T, block.n_heads, -1).transpose(1, 2)
        k = k.view(B, T, block.n_heads, -1).transpose(1, 2)
        v = v.view(B, T, block.n_heads, -1).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=block.dropout if block.training else 0.0
        )
        x = x + block.drop(block.proj(attn.transpose(1, 2).reshape(B, T, d)))
        x = x + block.drop(block.fc_out(F.gelu(block.fc_in(block.norm_mlp(x)))))
    return model.head(model.norm(x))


# --------------------------------------------------------------------------------------------------
# The keys: opt-in, hashes, memory keys, arm identities, refusals
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("path", "overrides"), list(PINNED), ids=lambda v: str(v)[-40:])
def test_the_keys_are_opt_in_and_keep_hashes_memory_keys_and_arm_ids(
    path: Path, overrides: tuple[str, ...]
) -> None:
    base = load_config(path, list(overrides))
    assert base.baseline.transformer.position == "learned"
    assert base.baseline.transformer.carry == 0
    dumped = json.loads(canonical_json(base))["baseline"].get("transformer", {})
    assert "position" not in dumped and "carry" not in dumped
    assert {"baseline.transformer.position", "baseline.transformer.carry"} <= set(
        HASH_OMIT_AT_DEFAULT
    )
    assert (config_hash(base), memory_key(base, "none", PROTOCOL)) == PINNED[(path, overrides)]
    assert "transformer" not in config_view(base).get("baseline", {})  # the arm id's view
    named = load_config(
        path,
        [*overrides, "baseline.transformer.position=learned", "baseline.transformer.carry=0"],
    )
    assert config_hash(named) == config_hash(base)
    assert memory_key(named, "none", PROTOCOL) == memory_key(base, "none", PROTOCOL)
    assert config_view(named) == config_view(base)
    if base.model_name != "transformer":
        return  # the keys are the Transformer's: elsewhere only the defaults load (refusals)
    seen = {config_hash(base)}
    keys = {memory_key(base, "none", PROTOCOL)}
    for extra in ([ROTARY], [ROTARY, "baseline.transformer.carry=256"]):
        on = load_config(path, [*overrides, *extra])
        seen.add(config_hash(on))
        keys.add(memory_key(on, "none", PROTOCOL))
        assert config_view(on)["baseline"]["transformer"]["position"] == "rotary"
    assert len(seen) == len(keys) == 3


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        (TRANSFORMER, ["baseline.transformer.carry=8"], "needs baseline.transformer.position"),
        (TXL, ["baseline.transformer.position=learned"], "needs baseline.transformer.position"),
        (TRANSFORMER, [ROTARY, "baseline.transformer.carry=-1"], "carry"),
        (
            TRANSFORMER,
            [ROTARY, "baseline.transformer.d_model=12", "baseline.transformer.n_heads=4"],
            "must be even",
        ),
        (GRU, [ROTARY], "belong to the Transformer baseline"),
        (MAMBA, [ROTARY, "baseline.transformer.carry=8"], "belong to the Transformer baseline"),
        (READER_LM, [ROTARY], "belong to the Transformer baseline"),
        (NOMEM, [ROTARY], "belong to the Transformer baseline"),
        (RECIPE_A6, [ROTARY], "belong to the Transformer baseline"),
    ],
)
def test_refusals(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(path, overrides)


def test_the_config_is_the_documented_model() -> None:
    """transformer_xl_2m.yaml: transformer_2m's model less its position table, a cache of 256
    positions, 393,216 floats a row; the same data and training settings."""
    cfg = load_config(TXL)
    old = load_config(TRANSFORMER)
    assert cfg.model_name == "transformer"
    assert cfg.baseline.transformer.position == "rotary" and cfg.baseline.transformer.carry == 256
    assert cfg.train == old.train and cfg.data == old.data and cfg.model == old.model
    seed_everything(0)
    model = build_model(cfg)
    assert isinstance(model, TransformerBaseline)
    report = model.param_report()
    assert report["stored"] == 1_881_472 == 1_930_624 - 256 * 192
    assert "pos_emb" not in report and "pos_emb" not in dict(model.named_parameters())
    keys, values, valid = model.init_state(32, "cpu")  # type: ignore[misc]
    assert keys.shape == values.shape == (4, 32, 4, 256, 48) and keys.dtype == torch.float32
    assert valid.shape == (32, 256) and valid.dtype == torch.bool and not bool(valid.any())
    assert (keys.numel() + values.numel()) // 32 == 393_216 == 2 * 4 * 256 * 192
    assert repr(model).startswith("TransformerBaseline(\n  position=rotary, carry=256")


# --------------------------------------------------------------------------------------------------
# The defaults: the model as it was
# --------------------------------------------------------------------------------------------------


def test_the_default_is_the_model_as_it_was() -> None:
    """Parameters, state dict, report, state and outputs of the default model are those of the
    model before the keys (its forward verbatim), bit for bit, in training and evaluation; the
    keys named at their defaults change nothing."""
    seed_everything(0)
    base = TransformerBaseline(load_config(TRANSFORMER))
    seed_everything(0)
    named = TransformerBaseline(
        load_config(
            TRANSFORMER, ["baseline.transformer.position=learned", "baseline.transformer.carry=0"]
        )
    )
    assert list(base.state_dict()) == list(named.state_dict())
    assert list(base.state_dict())[:3] == ["tok_emb.weight", "ctx_emb.weight", "pos_emb.weight"]
    assert not list(base.buffers())
    pairs = zip(base.parameters(), named.parameters(), strict=True)
    assert all(torch.equal(p, q) for p, q in pairs)
    assert base.param_report()["stored"] == 1_930_624 and "pos_emb" in base.param_report()
    assert base.init_state(2, "cpu") is None
    assert base.reset_state(None, torch.ones(2, dtype=torch.bool)) is None
    assert repr(base) == repr(named) and "position" not in repr(base)
    tokens, context, reset = _stream(3, 256, 0, 0.02)
    for training in (True, False):
        base.train(training)
        named.train(training)
        with torch.no_grad():
            want = _old_forward(base, tokens, context, reset)
            got, state, info = base(tokens, context, None, reset)
            again, _, _ = named(tokens, context, None, reset)
        assert torch.equal(got, want) and torch.equal(again, want)
        assert state is None and info == {}
    with pytest.raises(ValueError, match="carries no state"):
        base(tokens, context, (tokens,), reset)  # type: ignore[arg-type]


def test_the_rotary_twin_shares_the_cached_models_weights() -> None:
    """``position: rotary`` with ``carry: 0`` and with ``carry: M``: the same parameters from the
    same seed, a stateless twin; the cached model's first window (empty cache) and every window
    that starts a document at its first position read as the twin does."""
    seed_everything(0)
    twin = TransformerBaseline(_cfg(ROTARY))
    seed_everything(0)
    xl = TransformerBaseline(_cfg(ROTARY, "baseline.transformer.carry=12"))
    assert list(twin.state_dict()) == list(xl.state_dict())
    assert all(torch.equal(p, q) for p, q in zip(twin.parameters(), xl.parameters(), strict=True))
    assert twin.init_state(2, "cpu") is None
    assert twin.reset_state(None, torch.tensor([True, False])) is None
    twin = _model(ROTARY)
    xl = _model(ROTARY, "baseline.transformer.carry=12")
    tokens, context, reset = _stream(3, 16, 2, 0.1)
    reset[:, 8] = True  # the second window starts a document in every row
    with torch.no_grad():
        got, _ = _by_windows(xl, tokens, context, reset, 8)
        want, none = _by_windows(twin, tokens, context, reset, 8)
    assert none is None
    assert _gap(got, want) <= 1e-12
    with torch.no_grad():  # and with the documents going on, the cache is read
        reset[:, 8] = False
        got, _ = _by_windows(xl, tokens, context, reset, 8)
        want, _ = _by_windows(twin, tokens, context, reset, 8)
    assert _gap(got[:, :8], want[:, :8]) <= 1e-12 and _gap(got[:, 8:], want[:, 8:]) > 1e-3


# --------------------------------------------------------------------------------------------------
# Window by window with the cache = Transformer-XL over the stream
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("T", "M"), [(8, 8), (8, 3), (5, 12), (6, 20), (16, 8), (4, 1)])
def test_windows_with_the_cache_are_transformer_xl_over_the_stream(T: int, M: int) -> None:
    """The logits of every window equal the direct computation over the whole stream with
    Transformer-XL's reach: position ``a`` reads its document's positions from ``M`` before its
    window's start up to itself. ``M > T`` reaches over several windows."""
    model = _model(ROTARY, f"baseline.transformer.carry={M}", unroll=T)
    tokens, context, reset = _stream(3, 48, 7)
    reset[1, 16] = reset[2, 24] = True  # documents starting at a window's first position
    reset[0, 23] = True  # and at a window's last
    with torch.no_grad():
        got, _ = _by_windows(model, tokens, context, reset, T)
        want = _direct(model, tokens, context, _xl_visible(reset, T, M))
        cold = _direct(model, tokens, context, _xl_visible(reset, T, 0))
    assert got.shape == want.shape == (3, 48, 256)
    assert _gap(got, want) <= 1e-10
    assert _gap(got, cold) > 1e-3  # the cache is read


def test_the_window_after_a_full_cache_is_a_causal_reading_of_both_windows() -> None:
    """With ``M = T`` window 1 reads all of window 0: its logits are those of the stateless twin
    reading ``[window 0; window 1]`` as one causal window (resets inside both)."""
    T = 8
    xl = _model(ROTARY, f"baseline.transformer.carry={T}", unroll=T)
    twin = _model(ROTARY, unroll=2 * T)  # the same weights, a window of 2T
    tokens, context, reset = _stream(2, 2 * T, 3, 0.0)
    reset[1, 3] = True  # a document inside window 0 that goes on into window 1
    with torch.no_grad():
        got, _ = _by_windows(xl, tokens, context, reset, T)
        want, none, _ = twin(tokens, context, None, reset)
    assert none is None
    assert _gap(got, want) <= 1e-12


# --------------------------------------------------------------------------------------------------
# The cache: contents, stop-gradient, documents, state protocol
# --------------------------------------------------------------------------------------------------


def test_the_cache_holds_the_last_M_positions_of_the_document_going_on() -> None:
    M, T = 6, 4
    model = _model(ROTARY, f"baseline.transformer.carry={M}", unroll=T)
    tokens, context, reset = _stream(2, 12, 1, 0.0)
    reset[1, 10] = True  # row 1: a new document two positions before the stream's end
    with torch.no_grad():
        _, (keys, values, valid) = _by_windows(model, tokens, context, reset, T)
    assert keys.shape == values.shape == (3, 2, 2, M, 8) and keys.dtype == torch.float64
    assert valid[0].tolist() == [True] * M
    assert valid[1].tolist() == [False] * 4 + [True] * 2  # the slots before the reset are not
    with torch.no_grad():  # block 0's keys and values, un-rotated, of the last M positions
        x = model.tok_emb(tokens) + model.ctx_emb(context)
        _, k, v = model.blocks[0].qkv(model.blocks[0].norm_attn(x)).split(16, dim=-1)
    assert _gap(keys[0], k.view(2, 12, 2, 8).transpose(1, 2)[:, :, -M:]) <= 1e-12
    assert _gap(values[0], v.view(2, 12, 2, 8).transpose(1, 2)[:, :, -M:]) <= 1e-12
    cleared = model.reset_state((keys, values, valid), torch.tensor([True, False]))
    assert cleared is not None and not bool(cleared[2][0].any())
    assert torch.equal(cleared[2][1], valid[1]) and cleared[0] is keys
    # under bf16 autocast the cache keeps the parameters' dtype and its shapes
    fp32 = _model(ROTARY, f"baseline.transformer.carry={M}", unroll=T, dtype=torch.float32)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits, cache = _by_windows(fp32, tokens, context, reset, T)
    assert logits.dtype == torch.bfloat16
    assert [(t.dtype, tuple(t.shape)) for t in cache] == [
        (t.dtype, tuple(t.shape))
        for t in fp32.init_state(2, "cpu")  # type: ignore[union-attr]
    ]


def test_the_models_refuse_a_missing_or_wrong_cache() -> None:
    model = _model(ROTARY, "baseline.transformer.carry=4")
    tokens, context, reset = _stream(2, 6, 0)
    with pytest.raises(ValueError, match="the state must be"):
        model(tokens, context, None, reset)
    with pytest.raises(ValueError, match="the state must be"):
        model(tokens, context, model.init_state(3, "cpu"), reset)  # three rows for two
    with pytest.raises(ValueError, match="the state must be"):
        model.reset_state(None, torch.tensor([True, False]))
    with pytest.raises(ValueError, match="has no cache"):
        _model(ROTARY).check_cache(model.init_state(2, "cpu"), 2)
    with pytest.raises(ValueError, match="carries no state"):
        _model(ROTARY)(tokens, context, model.init_state(2, "cpu"), reset)


def test_the_cache_is_a_stop_gradient() -> None:
    """No gradient crosses a window: the cache returned holds no graph; window 1's gradients
    are the same whether window 0 ran with a graph or without; and they are the gradients of
    the stream read in one pass with the keys and values before window 1 detached."""
    T, M = 8, 8
    model = _model(ROTARY, f"baseline.transformer.carry={M}", unroll=T).train()
    tokens, context, reset = _stream(2, 2 * T, 9, 0.0)
    reset[0, 11] = True
    first = (tokens[:, :T], context[:, :T])
    second = (tokens[:, T:], context[:, T:])
    params = dict(model.named_parameters())

    def grads(with_graph: bool) -> dict[str, Tensor]:
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(with_graph):
            _, cache, _ = model(*first, model.init_state(2, "cpu"), reset[:, :T])
        assert all(not t.requires_grad and t.grad_fn is None for t in cache)
        out, nxt, _ = model(*second, cache, reset[:, T:])
        assert all(not t.requires_grad and t.grad_fn is None for t in nxt)
        out.square().mean().backward()
        return {n: p.grad.clone() for n, p in params.items() if p.grad is not None}

    a, b = grads(True), grads(False)
    model.zero_grad(set_to_none=True)
    direct = _direct(model, tokens, context, _xl_visible(reset, T, M), detach_before=T)
    direct[:, T:].square().mean().backward()
    c = {n: p.grad.clone() for n, p in params.items() if p.grad is not None}
    assert a.keys() == b.keys() == c.keys() and len(a) == len(params)
    assert all(_gap(a[n], b[n]) <= 1e-12 for n in a)
    assert all(_gap(a[n], c[n]) <= 1e-10 for n in a)


@pytest.mark.parametrize("T", [4, 8])
def test_a_document_reads_nothing_of_the_one_before_it(T: int) -> None:
    """Bytes of the document before a reset change nothing at or after it, cached or not; a
    document's first position reads itself alone."""
    model = _model(ROTARY, "baseline.transformer.carry=8", unroll=T)
    tokens, context, reset = _stream(2, 24, 3, 0.0)
    reset[0, 12] = True  # at a window's first position at T 4, inside one at T 8
    reset[1, 10] = True  # inside a window
    with torch.no_grad():
        out, _ = _by_windows(model, tokens, context, reset, T)
        before = tokens.clone()
        before[0, :12] = (before[0, :12] + 7) % 256
        before[1, :10] = (before[1, :10] + 7) % 256
        changed, _ = _by_windows(model, before, context, reset, T)
        assert torch.equal(changed[0, 12:], out[0, 12:])
        assert torch.equal(changed[1, 10:], out[1, 10:])
        assert _gap(changed[0, :12], out[0, :12]) > 0.0
        for row, start in ((0, 12), (1, 10)):
            alone, _, _ = model(
                tokens[row : row + 1, start : start + 1],
                context[row : row + 1, start : start + 1],
                model.init_state(1, "cpu"),
                torch.ones(1, 1, dtype=torch.bool),
            )
            assert _gap(alone[0, 0], out[row, start]) <= 1e-12


# --------------------------------------------------------------------------------------------------
# The trainer: training, evaluation, checkpoints, resume
# --------------------------------------------------------------------------------------------------


def _train_cfg(steps: int, *extra: str) -> RunConfig:
    """Windows of 8 over kv documents of 16 to 32 bytes: every document spans windows."""
    return load_config(
        TRANSFORMER,
        [
            *train_overrides(steps),
            *TINY,
            "train.lr=3e-3",
            "train.unroll=8",
            ROTARY,
            "baseline.transformer.carry=8",
            *extra,
        ],
    )


def test_the_trainer_trains_and_evaluates_with_the_cache_carried(tmp_path: Path) -> None:
    """40 steps with a falling loss; the evaluation carries the cache from window to window (the
    state each window receives is the one the window before returned, and it holds positions);
    the checkpoint holds the cache. The kv documents supervise their answers only, so a step
    whose windows hold none has no loss (NaN by the trainer's rule) and is left out."""
    steps = 40
    cfg = _train_cfg(steps)
    seed_everything(0)
    model = build_model(cfg)
    trainer = Trainer(model, cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps if s.loss_positions > 0]
    assert all(math.isfinite(s.loss) == (s.loss_positions > 0) for s in history.steps)
    assert len(losses) >= 15 and sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.05
    assert len(history.evals) == 1 and math.isfinite(history.evals[0]["loss"])
    payload = torch.load(history.checkpoints[-1], map_location="cpu", weights_only=False)
    carried = payload["extra"]["carried_state"]
    assert isinstance(carried, tuple) and len(carried) == 3
    assert carried[0].shape == (3, 4, 2, 8, 8) and carried[2].dtype == torch.bool

    received: list[Any] = []
    returned: list[Any] = []
    hooks = [
        model.register_forward_pre_hook(lambda m, args: received.append(args[2])),
        model.register_forward_hook(lambda m, args, out: returned.append(out[1])),
    ]
    try:
        result = trainer.evaluate(datasets()[1])
    finally:
        for hook in hooks:
            hook.remove()
    assert result["n_windows"] == len(received) >= 3
    assert not bool(received[0][2].any())  # a fresh cache at the first window
    assert all(r is p for r, p in zip(received[1:], returned[:-1], strict=True))
    assert any(bool(r[2].any()) for r in received[1:])  # documents go on across windows


def test_resume_equals_the_uninterrupted_run(tmp_path: Path) -> None:
    """Train 12 steps; resume another model from the step-6 checkpoint; the two runs agree bit
    for bit: losses, evaluations, parameters and the carried cache."""
    cfg = _train_cfg(12, "train.ckpt_every=6", "train.eval_every=6")
    train, dev = datasets()
    seed_everything(0)
    trainer_a = Trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "a")
    history_a = trainer_a.fit(train, dev, 12)
    assert [Path(p).name for p in history_a.checkpoints] == [
        "step_00000006.pt",
        "step_00000012.pt",
    ]
    seed_everything(1)  # other initial weights: everything must come from the file
    trainer_b = Trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "b")
    assert trainer_b.resume(tmp_path / "a" / "step_00000006.pt") == 6
    carried = trainer_b._carried_state
    assert isinstance(carried, tuple) and bool(carried[2].any())  # the cache came back
    history_b = trainer_b.fit(train, dev, 12)
    assert [record.step for record in history_b.steps] == list(range(7, 13))
    for rec_a, rec_b in zip(history_a.steps[6:], history_b.steps, strict=True):
        same = rec_a.loss == rec_b.loss or (math.isnan(rec_a.loss) and math.isnan(rec_b.loss))
        assert same and rec_a.grad_norm == rec_b.grad_norm, (rec_a, rec_b)
    assert history_a.evals[-1]["loss"] == history_b.evals[-1]["loss"]
    for key, value in trainer_a.model.state_dict().items():
        assert torch.equal(value, trainer_b.model.state_dict()[key]), key
    for a, b in zip(trainer_a._carried_state, trainer_b._carried_state, strict=True):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------------------------------
# The compiled window
# --------------------------------------------------------------------------------------------------


def test_the_cached_model_compiles_as_one_graph_and_matches_eager() -> None:
    """``aot_eager`` with ``fullgraph``: the cache in and out of one graph, no data-dependent
    shape; forward, cache and gradients equal eager over three windows; the next window reuses
    the graph; the compiled-baseline wrapper passes the cache through."""
    from torch._dynamo.utils import counters

    cfg = _cfg(ROTARY, "baseline.transformer.carry=12")
    seed_everything(0)
    model = build_model(cfg).train()
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    g = torch.Generator().manual_seed(3)
    try:
        graphs = None
        states = [model.init_state(3, "cpu"), model.init_state(3, "cpu")]
        for _ in range(3):
            tokens = torch.randint(0, 256, (3, 8), generator=g)
            context = torch.randint(0, 16, (3, 8), generator=g)
            reset = torch.rand(3, 8, generator=g) < 0.1
            results = []
            for i, forward in enumerate((model, compiled)):
                model.zero_grad(set_to_none=True)
                logits, states[i], info = forward(tokens, context, states[i], reset)
                logits.square().mean().backward()
                grads = {n: p.grad.clone() for n, p in model.named_parameters()}
                results.append((logits.detach(), grads))
                assert info == {}
            (ref, ref_grads), (got, got_grads) = results
            assert _gap(got, ref) <= 1e-6
            assert all(_gap(got_grads[n], ref_grads[n]) <= 1e-6 for n in ref_grads)
            assert all(_gap(a.float(), b.float()) <= 1e-6 for a, b in zip(*states, strict=True))
            if graphs is not None:
                assert counters["stats"]["unique_graphs"] == graphs, "the next window recompiled"
            graphs = counters["stats"]["unique_graphs"]
        wrapper = CompiledBaseline(model, resolve(cfg.hardware), "default", "aot_eager")
        state = wrapper.init_state(3, "cpu")
        for _ in range(2):
            logits, state, _ = wrapper(tokens, context, state, reset)
        assert len(state) == 3 and bool(state[2].any()) and logits.shape == (3, 8, 256)
        report = wrapper.report()
        assert wrapper.calls == 2 and report["graphs"] == report["functions"] == 1
    finally:
        torch._dynamo.reset()
