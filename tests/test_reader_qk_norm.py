"""The reader's normalised queries and keys (``model.reader_qk_norm``,
``docs/amendments/reader_qk_norm.md``; opt-in, 21 Nov 2026).

With the key on, an enabled reader block normalises its queries and keys per head (RMSNorm over
the head's channels, one learned weight for the queries and one for the keys, shared by the
heads, initialised to 1) after the projection and before the rotation; the values and the
softmax scale are unchanged. These tests check that the default leaves every configuration
hash, memory key, arm identity, parameter and output as main has them (the block's computation
against main's, written out here); that the logits of an enabled block stay bounded when its
qkv and LayerNorm weights grow, as they grow without the key; that a per-block list enables
the listed blocks only; that every reader path agrees with the key on (the plain, span, causal
and carried paths, the top block's word ends, and HSN-T with its memory and the carry); that
the memory-off model compiles as one graph with them; that the gradient reaches the new
weights; that the trainers put them in the groups without weight decay, beside the reader's
LayerNorm weights; and the refusals. Three CUDA tests (skipped without a card) train a few
compiled bf16 steps of the memory-off and the word-event recipes with the key on the top
block, and run the causal reader on the flash kernel.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from hsn.config import (
    HASH_OMIT_AT_DEFAULT,
    ModelConfig,
    RunConfig,
    canonical_json,
    config_hash,
    load_config,
)
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import config_view, memory_key
from hsn.model.hsn_t import HSNT
from hsn.model.reader import Reader, _ReaderBlock, _rotate
from hsn.runner import build_model, make_trainer
from hsn.train.memory_plan import reader_param_count
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets
from test_hsn_t import CUT_C, _model, _reader_cfg
from test_p2_8_compile import _release_compiled_state  # noqa: F401
from test_reader_carry import _by_windows, _gap, _pairs_of, _reader, _sliding_visible, _stream
from test_reader_carry_memory import FORMS, _pair, _read, _sighted
from test_reader_carry_memory import C as MEMORY_C
from test_reader_carry_memory import _stream as _memory_stream
from test_reader_top_words import _text_stream, _word_visible
from test_write_events import NGRAM, STE
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "configs" / "experiments"
NOMEM = EXPERIMENTS / "hsn_t_nomem.yaml"
NOMEM_160x3 = EXPERIMENTS / "hsn_t_nomem_top0_w1024_l3.yaml"
MEMORY = EXPERIMENTS / "hsn_t_memory.yaml"
CARRY_RECIPE = EXPERIMENTS / "hsn_t_memory_carry.yaml"
WORDS_CARRY = EXPERIMENTS / "hsn_t_memory_words_carry.yaml"
RECIPE_A6 = EXPERIMENTS / "hsn_recipe_a6.yaml"
READER_LM = REPO / "configs" / "baselines" / "reader_w64l2.yaml"
TRANSFORMER = REPO / "configs" / "baselines" / "transformer_2m.yaml"
GRU = REPO / "configs" / "baselines" / "gru_2m.yaml"
#: the top0 reader of the text runs (160 x 4, the top block unrotated) on a memory-off recipe
TOP0 = (
    "model.reader_width=160",
    "model.reader_layers=4",
    "model.reader_rotary_pairs=[20,20,20,0]",
)
#: the key on the top block of a four-block reader: the setting the amendment is for
TOP_QK = "model.reader_qk_norm=[false,false,false,true]"
#: the configurations whose hash and memory key the default must keep: (file, overrides)
CONFIGS: dict[str, tuple[Path, tuple[str, ...]]] = {
    "recipe_a6": (RECIPE_A6, ()),
    "nomem": (NOMEM, ()),
    "nomem_top0": (NOMEM, TOP0),
    "nomem_160x3": (NOMEM_160x3, ()),
    "nomem_carry": (NOMEM, (*TOP0, "model.reader_carry=32")),
    "reader_lm": (READER_LM, ()),
    "memory": (MEMORY, ()),
    "words_carry": (WORDS_CARRY, ()),
    "carry": (CARRY_RECIPE, ()),
    "transformer": (TRANSFORMER, ()),
}
#: the configuration hashes main (d3626df) gives them, before the key existed
MAIN_HASHES: dict[str, str] = {
    "recipe_a6": "d8b36a25971c9694b25251b4a6f20545c617e6d2796b7e4d1ac544acc3e78e28",
    "nomem": "427e8944984cd16906758d1a311b56264140136b4d7618e0c993f0b5e3db6592",
    "nomem_top0": "d34cd58ca09eceb80ff90d7a390df4fff1704fab8629f96075fde28406e7eb38",
    "nomem_160x3": "2215d13701407bced99ecbbbb737b8cb13fef778506b7d9749923865c7271227",
    "nomem_carry": "c063692d8e8d7a29b47f24e99191fcadf4f5cf8ad480a03c0db9397eb468f73e",
    "reader_lm": "9553d883a4bc17c73dc26c04b2e779f1c6906ec13d779095216f83f838a54bbf",
    "memory": "18c85210eec76722e4fd4c016d9213cd906fe0c11c184d34b04d064907d67f2c",
    "words_carry": "896f6682659d41043d9204f9297c2e989c5d3c8556f42fcfa6262dec2775ac79",
    "carry": "b4cc146fea59ec5c77cf0fbba694f7d8bbc917a8c7061b911a8f77c5cabb6ca4",
    "transformer": "65066a8915b79a328801b42780ae621f7185f23b553247a70c4f1b4dbd75071f",
}
#: the memory keys main gives them (variant none, the protocol below), before the key existed
MAIN_MEMORY_KEYS: dict[str, str] = {
    "recipe_a6": "3a556e053fb50f04",
    "nomem": "7cd4b2df65516f34",
    "nomem_top0": "2e8e034c71d8952c",
    "nomem_160x3": "18d0255d86cf1d48",
    "nomem_carry": "3f19fa21f6b8941f",
    "reader_lm": "c9b635c75ae2cf1a",
    "memory": "c60607f4e28e9c0e",
    "words_carry": "7f0c5b4798b63f37",
    "carry": "5c462728936efc9b",
    "transformer": "24b0b90a5677b73a",
}
PROTOCOL = {"name": "train_eval", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
#: the parameter layout of main's models (names and shapes, then the state-dict keys), digests
MAIN_LAYOUTS: dict[str, str] = {
    "nomem_top0": "93650ecb669e9e4c",
    "nomem_carry": "93650ecb669e9e4c",
    "words_carry": "415f78613c3a3a44",
    "carry": "3c1f5fddf926e14e",
    "reader_lm": "fe7ded0fb515c7a4",
}
#: the word-event carry recipe's trainable parameters on main
#: (tests/test_hsn_t_memory_carry_recipes.py)
WORDS_CARRY_PARAMS = 3_298_025


def _qk_parameters(module: torch.nn.Module) -> dict[str, Tensor]:
    return {
        n: p for n, p in module.named_parameters() if n.endswith(("q_norm.weight", "k_norm.weight"))
    }


# --------------------------------------------------------------------------------------------------
# Main's block, written out (python/hsn/model/reader.py at d3626df)
# --------------------------------------------------------------------------------------------------


def _main_forward(
    block: _ReaderBlock, x: Tensor, mask: Tensor | None, cos: Tensor | None, sin: Tensor | None
) -> Tensor:
    """``_ReaderBlock.forward`` as main has it."""
    B, T, w = x.shape
    q, k, v = block.qkv(block.norm_attn(x)).split(w, dim=-1)
    q = q.view(B, T, block.heads, -1).transpose(1, 2)
    k = k.view(B, T, block.heads, -1).transpose(1, 2)
    v = v.view(B, T, block.heads, -1).transpose(1, 2)
    if cos is not None and sin is not None:
        if block.rotate_in == "parameters":
            working = q.dtype
            q = _rotate(q.to(cos.dtype), cos, sin).to(working)
            k = _rotate(k.to(cos.dtype), cos, sin).to(working)
        else:
            cos, sin = cos.to(q.dtype), sin.to(q.dtype)
            q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
    if mask is None:
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    x = x + block.proj(attn.transpose(1, 2).reshape(B, T, w))
    out: Tensor = x + block.fc_out(F.gelu(block.fc_in(block.norm_mlp(x))))
    return out


def _main_forward_carry(
    block: _ReaderBlock,
    x: Tensor,
    mask: Tensor,
    cos: Tensor | None,
    sin: Tensor | None,
    keys: Tensor,
    values: Tensor,
    word_keys: Tensor | None = None,
    word_values: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """``_ReaderBlock.forward_carry`` as main has it."""
    B, T, w = x.shape
    C = keys.shape[2]
    q, k, v = block.qkv(block.norm_attn(x)).split(w, dim=-1)
    q = q.view(B, T, block.heads, -1).transpose(1, 2)
    k = k.view(B, T, block.heads, -1).transpose(1, 2)
    v = v.view(B, T, block.heads, -1).transpose(1, 2)
    all_k = torch.cat([keys.to(k.dtype), k], dim=2)
    all_v = torch.cat([values.to(v.dtype), v], dim=2)
    if cos is not None and sin is not None:
        if block.rotate_in == "parameters":
            working = q.dtype
            q = _rotate(q.to(cos.dtype), cos[C:], sin[C:]).to(working)
            all_k = _rotate(all_k.to(cos.dtype), cos, sin).to(working)
        else:
            cos, sin = cos.to(q.dtype), sin.to(q.dtype)
            q, all_k = _rotate(q, cos[C:], sin[C:]), _rotate(all_k, cos, sin)
    if word_keys is not None and word_values is not None:
        all_k = torch.cat([all_k, word_keys.to(all_k.dtype)], dim=2)
        all_v = torch.cat([all_v, word_values.to(all_v.dtype)], dim=2)
    attn = F.scaled_dot_product_attention(q, all_k, all_v, attn_mask=mask)
    x = x + block.proj(attn.transpose(1, 2).reshape(B, T, w))
    out: Tensor = x + block.fc_out(F.gelu(block.fc_in(block.norm_mlp(x))))
    return out, k, v


def _rms(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMSNorm over the last dimension, written out."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def _direct(reader: Reader, tokens: Tensor, visible: Tensor, top: Tensor | None = None) -> Tensor:
    """The reader with the key over a whole stream in one pass, written out
    (``test_reader_carry._direct`` with the enabled blocks' queries and keys normalised before
    the rotation): the attention over the keys ``visible [B, S, S]`` names (``top`` for the
    last block, when given)."""
    B, S = tokens.shape
    w, H = reader.width, reader.heads
    half = w // H // 2
    freq = reader.rotary_base ** (-torch.arange(half, dtype=torch.float64) / half)
    angles = torch.arange(S, dtype=torch.float64)[:, None] * freq
    x = reader.emb(tokens)
    for i, block in enumerate(reader.blocks):
        n = _pairs_of(reader, i, half)
        ang = angles.clone()
        ang[:, n:] = 0.0
        cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        q, k, v = block.qkv(block.norm_attn(x)).split(w, dim=-1)
        q, k, v = (t.view(B, S, H, -1).transpose(1, 2) for t in (q, k, v))
        if reader.qk_norm[i]:
            eps = block.norm_attn.eps
            q, k = _rms(q, block.q_norm.weight, eps), _rms(k, block.k_norm.weight, eps)
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        reads = top if top is not None and i == len(reader.blocks) - 1 else visible
        weights = scores.masked_fill(~reads[:, None], -math.inf).softmax(dim=-1)
        x = x + block.proj((weights @ v).transpose(1, 2).reshape(B, S, w))
        x = x + block.fc_out(F.gelu(block.fc_in(block.norm_mlp(x))))
    out: Tensor = reader.norm(x)
    return out


def _logits(block: _ReaderBlock, x: Tensor, cos: Tensor | None, sin: Tensor | None) -> Tensor:
    """The block's attention logits ``[B, H, T, T]`` (before the mask): the scaled products of
    its rotated queries and keys."""
    q, k, _ = block._project(x)
    if cos is not None and sin is not None:
        q, k = (
            _rotate(q, cos.to(q.dtype), sin.to(q.dtype)),
            _rotate(k, cos.to(k.dtype), sin.to(k.dtype)),
        )
    return q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])


def _layout(model: torch.nn.Module) -> str:
    import hashlib

    layout = "\n".join(f"{n}:{tuple(p.shape)}" for n, p in model.named_parameters())
    layout += "\n" + "\n".join(model.state_dict().keys())
    return hashlib.sha256(layout.encode()).hexdigest()[:16]


def _window(B: int, T: int, seed: int, resets: float = 0.1) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 256, (B, T), generator=g)
    reset = torch.rand(B, T, generator=g) < resets
    reset[:, 0] = True
    return tokens, reset


# --------------------------------------------------------------------------------------------------
# The default: main's hashes, memory keys, parameters and outputs
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CONFIGS))
def test_the_default_keeps_mains_hashes_and_memory_keys(name: str) -> None:
    assert "model.reader_qk_norm" in HASH_OMIT_AT_DEFAULT
    assert ModelConfig().reader_qk_norm is False
    path, overrides = CONFIGS[name]
    base = load_config(path, list(overrides))
    assert base.model.reader_qk_norm is False
    assert config_hash(base) == MAIN_HASHES[name]
    assert "reader_qk_norm" not in json.loads(canonical_json(base))["model"]
    assert "reader_qk_norm" not in config_view(base).get("model", {})
    named = load_config(path, [*overrides, "model.reader_qk_norm=false"])
    assert config_hash(named) == config_hash(base)
    assert config_view(named) == config_view(base)
    assert memory_key(base, "none", PROTOCOL) == MAIN_MEMORY_KEYS[name]
    assert memory_key(named, "none", PROTOCOL) == MAIN_MEMORY_KEYS[name]
    if base.model.reader == "none":
        return
    layers = base.model.reader_layers
    for on in ("true", f"[{', '.join(['false'] * (layers - 1) + ['true'])}]"):
        cfg = load_config(path, [*overrides, f"model.reader_qk_norm={on}"])
        assert config_hash(cfg) != config_hash(base)
        assert "reader_qk_norm" in config_view(cfg)["model"]
        assert memory_key(cfg, "none", PROTOCOL) != MAIN_MEMORY_KEYS[name]


@pytest.mark.parametrize("name", list(MAIN_LAYOUTS))
def test_the_default_keeps_mains_parameters(name: str) -> None:
    """The same names, shapes and state-dict keys as main, and the key on adds the two weights
    of each enabled block and nothing else: every other parameter starts as without it."""
    path, overrides = CONFIGS[name]
    cfg = load_config(path, list(overrides))
    seed_everything(0)
    base = build_model(cfg)
    assert _layout(base) == MAIN_LAYOUTS[name]
    assert not _qk_parameters(base)
    layers = cfg.model.reader_layers
    on = load_config(
        path,
        [*overrides, f"model.reader_qk_norm=[{', '.join(['false'] * (layers - 1) + ['true'])}]"],
    )
    seed_everything(0)
    model = build_model(on)
    extra = _qk_parameters(model)
    head = cfg.model.reader_width // cfg.model.reader_heads
    top = layers - 1
    assert sorted(n.split("blocks.", 1)[1] for n in extra) == [
        f"{top}.k_norm.weight",
        f"{top}.q_norm.weight",
    ]
    assert all(torch.equal(p, torch.ones(head)) for p in extra.values())
    plain = dict(base.named_parameters())
    rest = {n: p for n, p in model.named_parameters() if n not in extra}
    assert list(rest) == list(plain)
    assert all(torch.equal(rest[n], plain[n]) for n in plain)
    added = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters())
    assert added == 2 * head
    assert reader_param_count(on.model) - reader_param_count(cfg.model) == 2 * head
    if isinstance(model, HSNT):
        assert model.param_report()["reader"] == reader_param_count(on.model)
    if name == "words_carry":
        assert model.param_report()["total_trainable"] == WORDS_CARRY_PARAMS + 80


@pytest.mark.parametrize(
    "keys",
    [
        {},
        {"reader_rotary_pairs": [4, 2, 0]},
        {"reader_rotary_dtype": "parameters"},
        {"reader_position": "learned"},
        {"reader_span": 5},
        {"reader_mask": "causal"},
        {"reader_carry": 6},
        {"reader_carry": 6, "reader_rotary_pairs": [4, 4, 0], "reader_top_words": 3},
    ],
)
@pytest.mark.parametrize("autocast", [False, True], ids=["float32", "bf16_autocast"])
def test_the_default_block_is_mains_computation_bit_for_bit(
    keys: dict[str, object], autocast: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the default every block runs main's computation (written out above), bit for bit, in
    float32 and under bf16 autocast, on every reader path: plain, span, causal, carried, with
    the top block's word ends."""
    reader = _reader(torch.float32, **keys)
    assert not _qk_parameters(reader) and reader.qk_norm == [False] * 3
    tokens, reset = _text_stream(2, 24, 3)
    if keys.get("reader_mask") == "causal":
        reset = torch.zeros_like(reset)
        reset[0, 0] = True

    def run() -> Tensor:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            if reader.carry:
                return _by_windows(reader, tokens, reset, 8)
            return reader(tokens, reset)

    got = run()
    monkeypatch.setattr(_ReaderBlock, "forward", _main_forward)
    monkeypatch.setattr(_ReaderBlock, "forward_carry", _main_forward_carry)
    want = run()
    assert got.dtype == want.dtype and torch.equal(got, want)


@pytest.mark.parametrize("value", ["false", "[false, false]"])
def test_the_default_hsn_t_is_unchanged_bit_for_bit(value: str) -> None:
    """HSN-T with its memory: the key absent, false, or a list of false build the same model and
    give the same outputs, bit for bit (a list of false is another configuration value, so
    another hash: spell the default false)."""
    base = _reader_cfg()
    named = _reader_cfg(f"model.reader_qk_norm={value}")
    a, b = _model(base), _model(named)
    assert list(a.state_dict()) == list(b.state_dict())
    for x, y in (
        (a.named_parameters(), b.named_parameters()),
        (a.named_buffers(), b.named_buffers()),
    ):
        pairs = list(zip(x, y, strict=True))
        assert all(n == m and torch.equal(p, q) for (n, p), (m, q) in pairs)
    tokens, ctx, reset = _memory_stream(base, 1)
    with torch.no_grad():
        out_a, _, _ = a(tokens, ctx, a.init_state(3), reset)
        out_b, _, _ = b(tokens, ctx, b.init_state(3), reset)
    assert torch.equal(out_a, out_b)
    assert (config_hash(named) == config_hash(base)) == (value == "false")


# --------------------------------------------------------------------------------------------------
# The property: bounded logits
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pairs", [0, 4], ids=["unrotated", "rotated"])
def test_the_logits_stay_bounded_when_the_weights_grow(pairs: int) -> None:
    """The growth seen on code native (the unrotated top block's norm_attn and qkv weights): with
    them scaled 10x the logits of a block without the key grow a hundred times and more, far
    past the bound that those of a block with it keep at both scales,
    max|g_q| * max|g_k| * sqrt(w / H)."""
    x = torch.randn(3, 12, 16, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    peaks: dict[bool, list[float]] = {}
    bound = 0.0
    for on in (False, True):
        reader = _reader(layers=2, reader_rotary_pairs=[4, pairs], reader_qk_norm=[False, on])
        block = reader.blocks[1]
        cos, sin = reader._block_rotation(reader._block_pairs[1], *reader._rotation(12, x))
        peaks[on] = []
        for scale in (1.0, 10.0):
            with torch.no_grad():
                if scale != 1.0:
                    block.qkv.weight.mul_(scale)
                    block.norm_attn.weight.mul_(scale)
                logits = _logits(block, x, cos, sin)
            peaks[on].append(float(logits.abs().max()))
            if on:
                gains = (block.q_norm.weight.detach(), block.k_norm.weight.detach())
                bound = float(gains[0].abs().max() * gains[1].abs().max())
                assert peaks[on][-1] <= bound * math.sqrt(8) * (1 + 1e-12)
    assert peaks[False][1] > 100 * peaks[False][0]
    assert peaks[False][1] > 10 * bound * math.sqrt(8)


def test_the_norm_is_the_rms_norm_over_a_head_in_the_working_dtype() -> None:
    """The block's queries and keys are RMSNorm over the head's channels with the weights of the
    block, before the rotation; the values are the projection's; under bf16 autocast all
    three keep the working dtype (bf16), so SDPA takes them as before."""
    reader = _reader(torch.float32, layers=2, reader_qk_norm=True)
    block = reader.blocks[0]
    x = torch.randn(2, 10, 16, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        q, k, v = block._project(x)
        raw = block.qkv(block.norm_attn(x)).split(16, dim=-1)
    rq, rk, rv = (t.view(2, 10, 2, -1).transpose(1, 2) for t in raw)
    eps = block.norm_attn.eps
    assert block.q_norm.eps == block.k_norm.eps == eps == 1e-5
    assert _gap(q, _rms(rq, block.q_norm.weight, eps)) <= 1e-6
    assert _gap(k, _rms(rk, block.k_norm.weight, eps)) <= 1e-6
    assert torch.equal(v, rv)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        q16, k16, v16 = block._project(x)
    assert q16.dtype == k16.dtype == v16.dtype == torch.bfloat16
    assert _gap(q16.float(), q) <= 5e-2 * float(q.abs().max())


# --------------------------------------------------------------------------------------------------
# Per block, and every reader path agrees
# --------------------------------------------------------------------------------------------------


def test_a_list_enables_the_listed_blocks_only() -> None:
    plain = _reader(torch.float32, layers=3)
    listed = Reader(
        ModelConfig(
            reader="transformer",
            reader_width=16,
            reader_heads=2,
            reader_layers=3,
            reader_qk_norm=[False, True, False],
        ),
        seed=0,
    )
    assert listed.qk_norm == [False, True, False]
    assert [b.q_norm is not None for b in listed.blocks] == [False, True, False]
    assert [b.k_norm is not None for b in listed.blocks] == [False, True, False]
    assert sorted(_qk_parameters(listed)) == ["blocks.1.k_norm.weight", "blocks.1.q_norm.weight"]
    assert "qk_norm=[False, True, False]" in repr(listed)
    assert "qk_norm=True" in repr(_reader(layers=2, reader_qk_norm=True))
    assert "qk_norm" not in repr(plain)
    # the same weights: the blocks without the key compute what the plain reader's do
    fresh = Reader(
        ModelConfig(reader="transformer", reader_width=16, reader_heads=2, reader_layers=3), seed=0
    )
    extra = _qk_parameters(listed)
    shared = {n: p for n, p in listed.state_dict().items() if n not in extra}
    assert list(shared) == list(fresh.state_dict())
    assert all(torch.equal(p, fresh.state_dict()[n]) for n, p in shared.items())
    tokens, reset = _window(2, 12, 1)
    x = fresh.emb(tokens)
    cos, sin = fresh._rotation(12, x)
    mask = fresh.attention_mask(reset)
    with torch.no_grad():
        assert torch.equal(listed.blocks[0](x, mask, cos, sin), fresh.blocks[0](x, mask, cos, sin))
        assert torch.equal(listed.blocks[2](x, mask, cos, sin), fresh.blocks[2](x, mask, cos, sin))
        assert _gap(listed.blocks[1](x, mask, cos, sin), fresh.blocks[1](x, mask, cos, sin)) > 1e-3
        assert _gap(listed(tokens, reset), fresh(tokens, reset)) > 1e-3


QK_FORMS = [True, [False, True, False], [True, False, True]]


@pytest.mark.parametrize(("T", "C"), [(8, 8), (8, 3), (5, 12), (48, 8)])
@pytest.mark.parametrize("pairs", [None, [4, 2, 0]])
@pytest.mark.parametrize("qk", QK_FORMS, ids=["all", "middle", "outer"])
def test_the_carry_is_one_pass_with_a_sliding_span(
    T: int, C: int, pairs: object, qk: object
) -> None:
    """With the key on, the stream read window by window with the cache equals the direct
    computation over the whole stream (the enabled blocks' queries and keys normalised, written
    out) and the reader of ``reader_span = C + 1`` in one pass: the cache holds the normalised
    keys, those the same positions have without it."""
    reader = _reader(reader_carry=C, reader_rotary_pairs=pairs, reader_qk_norm=qk)
    span = _reader(reader_span=C + 1, reader_rotary_pairs=pairs, reader_qk_norm=qk)
    span.load_state_dict(reader.state_dict())
    tokens, reset = _stream(3, 48, 7)
    reset[1, 16] = reset[2, 8] = True
    with torch.no_grad():
        got = _by_windows(reader, tokens, reset, T)
        direct = _direct(reader, tokens, _sliding_visible(reset, C))
        one = span(tokens, reset)
    assert _gap(got, direct) <= 1e-11
    assert _gap(got, one) <= 1e-11


def test_the_cache_holds_the_normalised_keys() -> None:
    """With the weights at their start (1) every cached key of an enabled block has a mean
    square of 1 (to the eps), and those of the other blocks do not."""
    reader = _reader(reader_carry=8, reader_qk_norm=[True, False, True])
    with torch.no_grad():
        for p in _qk_parameters(reader).values():
            p.fill_(1.0)
    tokens, reset = _stream(3, 8, 2, 0.0)
    cache = reader.init_cache(3, "cpu")
    assert cache is not None
    with torch.no_grad():
        _, cache = reader.read(tokens, reset, cache)
    keys, valid = cache[0], cache[2]
    assert bool(valid.all())
    square = keys.pow(2).mean(-1)  # [L, B, H, C]
    assert _gap(square[0], torch.ones_like(square[0])) <= 1e-3
    assert _gap(square[2], torch.ones_like(square[2])) <= 1e-3
    assert _gap(square[1], torch.ones_like(square[1])) > 1e-1


def test_the_whole_window_and_the_span_are_the_direct_computation() -> None:
    reader = _reader(reader_qk_norm=[True, False, True], reader_rotary_pairs=[4, 4, 0])
    tokens, reset = _stream(2, 20, 5)
    S = tokens.shape[1]
    index = torch.arange(S)
    segment = reset.cumsum(dim=1)
    same = segment[:, :, None] == segment[:, None, :]
    causal = index[:, None] >= index[None, :]
    with torch.no_grad():
        assert _gap(reader(tokens, reset), _direct(reader, tokens, same & causal)) <= 1e-11
        reader.span = 4
        near = same & causal & (index[:, None] - index[None, :] < 4)
        assert _gap(reader(tokens, reset), _direct(reader, tokens, near)) <= 1e-11


def test_the_causal_mask_is_the_document_mask_with_the_key() -> None:
    doc = _reader(reader_qk_norm=True, reader_rotary_pairs=[4, 4, 0])
    causal = _reader(reader_qk_norm=True, reader_rotary_pairs=[4, 4, 0], reader_mask="causal")
    causal.load_state_dict(doc.state_dict())
    tokens, reset = _window(3, 16, 2, resets=0.0)
    reset[1, 0] = False  # a document going on from the window before
    with torch.no_grad():
        assert _gap(causal(tokens, reset), doc(tokens, reset)) <= 1e-12


@pytest.mark.parametrize(("T", "C", "N"), [(8, 4, 3), (16, 3, 5)])
def test_the_top_blocks_word_ends_with_the_key(T: int, C: int, N: int) -> None:
    """``model.reader_top_words`` with the key on the top block: the word slots hold the top
    block's normalised keys, and the windows equal the direct computation."""
    reader = _reader(
        reader_carry=C,
        reader_rotary_pairs=[4, 4, 0],
        reader_top_words=N,
        reader_qk_norm=[False, False, True],
    )
    tokens, reset = _text_stream(3, 48, 9)
    near = _sliding_visible(reset, C)
    want = _direct(reader, tokens, near, top=near | _word_visible(tokens, reset, C, N, T))
    with torch.no_grad():
        got = _by_windows(reader, tokens, reset, T)
    assert _gap(got, want) <= 1e-11


@pytest.mark.parametrize("form", ["per_byte", "word_events"])
def test_hsn_t_with_its_memory_and_the_carry_equals_the_sighted_reading(form: str) -> None:
    """HSN-T with its memory and the carry, the key on one of the two blocks: three windows read
    with the carry equal the same weights' sighted reading (the reader over the whole stream
    with a span of C + 1, handed in window by window)."""
    carried, plain, cfg = _pair(
        STE, CUT_C, *NGRAM, *FORMS[form], "model.reader_qk_norm=[false,true]"
    )
    assert carried.reader is not None and carried.reader.qk_norm == [False, True]
    T = cfg.train.unroll
    stream = _memory_stream(cfg, 3)
    got, _ = _read(carried, stream, T)
    want, _ = _read(plain, stream, T, reader_out=_sighted(plain, stream, MEMORY_C + 1))
    assert _gap(got, want) <= 1e-12


@pytest.mark.parametrize("carry", [0, 6], ids=["window", "carried"])
def test_the_memory_off_model_compiles_as_one_graph_and_matches_eager(carry: int) -> None:
    """The memory-off model (the reader inside the compiled window, as
    ``hardware.baseline_compile`` compiles it) with the key on its top block: ``aot_eager`` with
    ``fullgraph`` gives eager's logits, cache and gradients (the new weights' included), and the
    next window reuses the graph."""
    from torch._dynamo.utils import counters

    tiny = ("model.d=8", "model.context_dim=8", "model.hidden=16", "model.reader_width=16")
    cfg = load_config(
        NOMEM,
        [
            *tiny,
            "model.reader_heads=2",
            "model.reader_rotary_pairs=[4,0]",
            "model.reader_qk_norm=[false,true]",
            f"model.reader_carry={carry}",
        ],
    )
    seed_everything(0)
    model = build_model(cfg).train()
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    g = torch.Generator().manual_seed(3)
    try:
        graphs = None
        states = [model.init_state(3, "cpu"), model.init_state(3, "cpu")]
        for _ in range(3):
            tokens = torch.randint(0, 256, (3, 8), generator=g)
            ctx = torch.randint(0, cfg.model.n_context, (3, 8), generator=g)
            reset = torch.rand(3, 8, generator=g) < 0.1
            results = []
            for i, forward in enumerate((model, compiled)):
                model.zero_grad(set_to_none=True)
                logits, states[i], _ = forward(tokens, ctx, states[i], reset)
                logits.square().mean().backward()
                grads = {n: p.grad.clone() for n, p in model.named_parameters()}
                results.append((logits.detach(), grads))
            (ref, ref_grads), (got, got_grads) = results
            assert _gap(got, ref) <= 1e-6
            assert {n for n in ref_grads if "_norm.weight" in n and "norm_" not in n} == {
                "reader.blocks.1.q_norm.weight",
                "reader.blocks.1.k_norm.weight",
            }
            assert all(_gap(got_grads[n], ref_grads[n]) <= 1e-6 for n in ref_grads)
            if carry:
                assert all(_gap(a.float(), b.float()) <= 1e-6 for a, b in zip(*states, strict=True))
            if graphs is not None:
                assert counters["stats"]["unique_graphs"] == graphs, "the next window recompiled"
            graphs = counters["stats"]["unique_graphs"]
    finally:
        torch._dynamo.reset()


# --------------------------------------------------------------------------------------------------
# Gradients and the optimiser
# --------------------------------------------------------------------------------------------------


def test_the_gradient_reaches_the_new_weights() -> None:
    """Through the plain and the carried reader, and through HSN-T with its memory (the decoder
    and S read the reader's output)."""
    tokens, reset = _stream(2, 24, 3)
    for reader in (_reader(reader_qk_norm=True), _reader(reader_qk_norm=True, reader_carry=5)):
        reader.train()
        out = _by_windows(reader, tokens, reset, 8) if reader.carry else reader(tokens, reset)
        out.square().mean().backward()
        grads = {n: p.grad for n, p in _qk_parameters(reader).items()}
        assert len(grads) == 6
        assert all(
            g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0
            for g in grads.values()
        )
    cfg = _reader_cfg("model.reader_qk_norm=[false,true]")
    model = _model(cfg)
    stream = _memory_stream(cfg, 1)
    logits, _, _ = model(stream[0], stream[1], model.init_state(3), stream[2])
    logits.square().mean().backward()
    grads = _qk_parameters(model)
    assert sorted(grads) == ["reader.blocks.1.k_norm.weight", "reader.blocks.1.q_norm.weight"]
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in grads.values())


def _decay_of(optimizer: torch.optim.Optimizer) -> dict[int, float]:
    decay: dict[int, float] = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            assert id(p) not in decay, "a parameter in two groups"
            decay[id(p)] = group["weight_decay"]
    return decay


@pytest.mark.parametrize(
    ("path", "overrides"),
    [
        (NOMEM, (*TOP0, TOP_QK)),  # the generic trainer, AdamW
        (NOMEM, (*TOP0, TOP_QK, "train.fused_adamw=true")),
        (READER_LM, ("model.reader_qk_norm=true",)),
        (WORDS_CARRY, (TOP_QK,)),  # the TBPTT trainer, LazyAdamW
        (WORDS_CARRY, (TOP_QK, "train.cell_lr_mult=0.5")),  # with the cell's own groups
        (MEMORY, ("model.reader_qk_norm=true", "train.optimizer=adamw")),
    ],
)
def test_the_new_weights_take_no_weight_decay(
    path: Path, overrides: tuple[str, ...], tmp_path: Path
) -> None:
    """The trainers' groups: the new weights are in a group without weight decay, as the
    reader's LayerNorm weights and biases are, and the reader's matrices decay."""
    cfg = load_config(path, list(overrides))
    assert cfg.train.weight_decay > 0
    seed_everything(0)
    model = build_model(cfg)
    trainer = make_trainer(model, cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    decay = _decay_of(trainer.optimizer)
    named = dict(model.named_parameters())
    assert all(id(p) in decay for p in named.values() if p.requires_grad)
    new = _qk_parameters(model)
    assert new and all(decay[id(p)] == 0.0 for p in new.values())
    norms = [p for n, p in named.items() if "norm_attn" in n or "norm_mlp" in n]
    assert norms and all(decay[id(p)] == 0.0 for p in norms)
    qkv = [p for n, p in named.items() if n.endswith("qkv.weight")]
    assert qkv and all(decay[id(p)] == cfg.train.weight_decay for p in qkv)


# --------------------------------------------------------------------------------------------------
# Refusals and accepted settings
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        (
            RECIPE_A6,
            ["model.reader_qk_norm=true"],
            "normalises the queries and keys of model.reader",
        ),
        (RECIPE_A6, ["model.reader_qk_norm=[false, false]"], "set model.reader=transformer"),
        (TRANSFORMER, ["model.reader_qk_norm=true"], "normalises the queries and keys"),
        (GRU, ["model.reader_qk_norm=[true]"], "normalises the queries and keys"),
        (
            NOMEM,
            ["model.reader_qk_norm=[true]"],
            "names 1 blocks and the reader has model.reader_layers=2",
        ),
        (NOMEM, [*TOP0, "model.reader_qk_norm=[false,true]"], "names 2 blocks"),
        (WORDS_CARRY, ["model.reader_qk_norm=[true,true,true,true,true]"], "names 5 blocks"),
    ],
)
def test_refusals(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(path, overrides)


def test_the_reader_refuses_a_list_of_the_wrong_length_it_was_handed() -> None:
    cfg = ModelConfig(reader="transformer", reader_width=16, reader_heads=2)
    with pytest.raises(ValueError, match="names 3 blocks"):
        Reader(cfg.model_copy(update={"reader_qk_norm": [True, True, True]}), seed=0)


@pytest.mark.parametrize(
    ("path", "overrides"),
    [
        (NOMEM, ("model.reader_qk_norm=true",)),
        (NOMEM, (*TOP0, TOP_QK, "model.reader_mask=causal")),
        (NOMEM, (*TOP0, TOP_QK, "model.reader_carry=16", "model.reader_top_words=8")),
        (NOMEM, ("model.reader_qk_norm=true", "model.reader_position=learned")),
        (NOMEM, ("model.reader_qk_norm=true", "model.reader_rotary_dtype=parameters")),
        (READER_LM, ("model.reader_qk_norm=[false,true]",)),
        (MEMORY, ("model.reader_qk_norm=true",)),
        (WORDS_CARRY, (TOP_QK,)),
        (CARRY_RECIPE, (TOP_QK, "train.reach_dropout=0.25")),
    ],
)
def test_accepted_settings_build_and_read(path: Path, overrides: tuple[str, ...]) -> None:
    cfg: RunConfig = load_config(path, list(overrides))
    seed_everything(0)
    model = build_model(cfg).eval()
    assert _qk_parameters(model)
    B, T = 2, 16
    tokens, reset = _window(B, T, 0, resets=0.0)
    ctx = torch.zeros(B, T, dtype=torch.int64)
    with torch.no_grad():
        if isinstance(model, HSNT):
            logits, _, _ = model(tokens, ctx, model.init_state(B), reset)
        else:
            logits, _, _ = model(tokens, ctx, model.init_state(B, "cpu"), reset)
    assert logits.shape == (B, T, 256) and bool(torch.isfinite(logits).all())


# --------------------------------------------------------------------------------------------------
# On the card (skipped without one): compiled bf16 steps of the recipes, and the flash kernel
# --------------------------------------------------------------------------------------------------


def _cuda_or_skip() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")


#: a few steps of a recipe on the card: batch 8, the recipe's window, no warm-up
CARD_STEPS = (
    "train.steps=3",
    "train.batch=8",
    "train.micro_batch=8",
    "train.warmup_steps=0",
    "train.eval_every=1000",
    "train.ckpt_every=1000",
    "train.stats_every=1",
    "train.precision=bf16",
    "hardware.autocast_dtype=bf16",
)


def _train_on_the_card(cfg: RunConfig, tmp_path: Path) -> torch.nn.Module:
    seed_everything(0)
    model = build_model(cfg)
    trainer = make_trainer(model, cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    history = trainer.fit(*datasets(), cfg.train.steps)
    assert len(history.steps) == cfg.train.steps
    assert all(math.isfinite(s.loss) or s.loss_positions == 0 for s in history.steps)
    assert any(math.isfinite(s.loss) for s in history.steps)
    new = _qk_parameters(model)
    assert sorted(n.rsplit(".", 2)[0].rsplit(".", 1)[-1] for n in new) == ["3", "3"]
    assert all(
        p.device.type == "cuda" and not torch.equal(p.detach().cpu(), torch.ones(p.shape[0]))
        for p in new.values()
    )
    return model


@pytest.mark.cuda
def test_compiled_bf16_steps_of_the_memory_off_recipe_on_the_card(tmp_path: Path) -> None:
    """hsn_t_nomem.yaml with the top0 reader and the key on its top block: three steps under
    CUDA graphs (``hardware.baseline_compile=reduce-overhead``, the generic trainer's compiled
    window; ``hardware.compile_mode`` is the HSN's) in bf16; finite losses and the new weights
    trained."""
    from cards import CUDA_PROFILE

    _cuda_or_skip()
    cfg = load_config(
        NOMEM,
        [
            *TOP0,
            TOP_QK,
            f"profile={CUDA_PROFILE}",
            "hardware.baseline_compile=reduce-overhead",
            *CARD_STEPS,
        ],
    )
    _train_on_the_card(cfg, tmp_path)


@pytest.mark.cuda
def test_compiled_bf16_steps_of_the_word_event_carry_recipe_on_the_card(tmp_path: Path) -> None:
    """hsn_t_memory_words_carry.yaml with the key on its top block and the CUDA arm's lines
    (``hardware.compile_mode=reduce-overhead``, the alias index, the two speed keys): three
    compiled steps in bf16; finite losses, the new weights trained, the carried cache on the
    card in float32."""
    from cards import CUDA_PROFILE
    from hsn.model.hsn import WindowCarry

    _cuda_or_skip()
    cfg = load_config(
        WORDS_CARRY,
        [
            TOP_QK,
            f"profile={CUDA_PROFILE}",
            "hardware.compile_mode=reduce-overhead",
            "hardware.graph_alias_index=true",
            "hardware.message_gradients=state",
            "hardware.round_products=merged",
            *CARD_STEPS,
        ],
    )
    seed_everything(0)
    model = build_model(cfg)
    trainer = make_trainer(model, cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    history = trainer.fit(*datasets(), cfg.train.steps)
    assert len(history.steps) == cfg.train.steps and all(
        math.isfinite(s.loss) for s in history.steps
    )
    new = _qk_parameters(model)
    assert sorted(new) == ["reader.blocks.3.k_norm.weight", "reader.blocks.3.q_norm.weight"]
    assert all(not torch.equal(p.detach().cpu(), torch.ones(40)) for p in new.values())
    carry = trainer._prev_tokens  # type: ignore[attr-defined]
    assert isinstance(carry, WindowCarry)
    assert carry.reader[0].device.type == "cuda" and carry.reader[0].dtype == torch.float32


@pytest.mark.cuda
def test_the_causal_reader_with_the_key_runs_on_flash_on_the_card(tmp_path: Path) -> None:
    """``model.reader_mask=causal`` with the key on the top block (head width 40): the flash
    kernel alone runs the reader forward and backward under bf16 autocast (the normalised
    queries and keys stay bf16), it equals the document reader to bf16 rounding, and three
    compiled bf16 steps of the memory-off recipe train with it."""
    from cards import CUDA_PROFILE
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.profiler import ProfilerActivity, profile

    _cuda_or_skip()
    keys = [*TOP0, TOP_QK]
    causal_cfg = load_config(NOMEM, [*keys, "model.reader_mask=causal"]).model
    reader = Reader(causal_cfg, seed=0).cuda().train()
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, 256, (4, 256), generator=g).cuda()
    reset = torch.zeros(4, 256, dtype=torch.bool, device="cuda")
    reset[::2, 0] = True
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]), torch.autocast("cuda", dtype=torch.bfloat16):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            out = reader(tokens, reset)
            out.float().square().mean().backward()
            torch.cuda.synchronize()
    names = " ".join(e.key for e in prof.key_averages())
    assert "flash_fwd" in names and "flash_bwd" in names, names
    assert all(p.grad is not None for p in _qk_parameters(reader).values())
    doc = Reader(load_config(NOMEM, keys).model, seed=0).cuda()
    doc.load_state_dict(reader.state_dict())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert _gap(out.float(), doc(tokens, reset).float()) <= 5e-2
    cfg = load_config(
        NOMEM,
        [
            *keys,
            "model.reader_mask=causal",
            f"profile={CUDA_PROFILE}",
            "hardware.baseline_compile=reduce-overhead",
            *CARD_STEPS,
        ],
    )
    _train_on_the_card(cfg, tmp_path)
