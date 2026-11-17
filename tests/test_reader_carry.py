"""The reader's carried keys (``model.reader_carry``, ``docs/hsn_t_no_memory.md``, "Reading past
the window"; opt-in, 15 Nov 2026).

With ``model.reader_carry = C`` every reader block keeps the keys and values of the last ``C``
positions of the windows before, detached, and position ``t`` reads the positions of its
document at distances ``0 … C``, cached or in the window. These tests check that the default
leaves every configuration hash, memory key, arm identity, parameter and output as it was; that
a stream read window by window with the cache is the stream read in one pass with a sliding
span of ``C`` (two references: a direct one written here, and the reader of ``reader_span =
C + 1`` over the whole stream), at every window length against ``C``; that a document reads
nothing of the one before it, cached or not; that the cache is detached; the refusals; and that
the generic trainer trains and evaluates a model that carries the cache, eager and compiled.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from hsn.baselines.reader_lm import ReaderLM
from hsn.config import (
    HASH_OMIT_AT_DEFAULT,
    ModelConfig,
    canonical_json,
    config_hash,
    load_config,
)
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import config_view, memory_key
from hsn.model.hsn_t import HSNT, HSNTNoMemory
from hsn.model.reader import Reader, _rotate
from hsn.runner import build_model, make_trainer
from hsn.train.baseline_compile import CompiledBaseline
from hsn.train.loop import Trainer
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, train_overrides
from test_a6_feedback_detach import A5, _cfg
from test_reader_rotary_table import _reference_forward
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
NOMEM = REPO / "configs" / "experiments" / "hsn_t_nomem.yaml"
READER_LM = REPO / "configs" / "baselines" / "reader_w64l2.yaml"
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
TRANSFORMER = REPO / "configs" / "baselines" / "transformer_2m.yaml"
GRU = REPO / "configs" / "baselines" / "gru_2m.yaml"

#: the memory-off recipe at a size the CPU runs in a moment (two heads of four pairs)
TINY = ("model.d=8", "model.context_dim=8", "model.hidden=16", "model.reader_width=16")
TINY = (*TINY, "model.reader_heads=2")
PROTOCOL = {"name": "train_eval", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}


def _gap(a: Tensor, b: Tensor) -> float:
    return float((a - b).detach().abs().max())


def _reader(dtype: torch.dtype = torch.float64, layers: int = 3, **keys: object) -> Reader:
    """A reader of width 16, two heads of four pairs, away from its start so that every weight
    matters (the same weights for every ``keys``: the carry and the span add no parameter)."""
    cfg = ModelConfig(
        reader="transformer",
        reader_width=16,
        reader_heads=2,
        reader_layers=layers,
        **keys,  # type: ignore[arg-type]
    )
    reader = Reader(cfg, seed=0).to(dtype)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for p in reader.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=torch.float64).to(dtype))
    return reader.eval()


def _stream(B: int, S: int, seed: int, resets: float = 0.05) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 256, (B, S), generator=g)
    reset = torch.rand(B, S, generator=g) < resets
    reset[:, 0] = True
    return tokens, reset


def _by_windows(reader: Reader, tokens: Tensor, reset: Tensor, T: int) -> Tensor:
    """The stream read window by window (the last one may be shorter), the cache carried."""
    cache = reader.init_cache(int(tokens.shape[0]), tokens.device)
    assert cache is not None
    outs = []
    for lo in range(0, int(tokens.shape[1]), T):
        out, cache = reader.read(tokens[:, lo : lo + T], reset[:, lo : lo + T], cache)
        outs.append(out)
    return torch.cat(outs, dim=1)


def _pairs_of(reader: Reader, block: int, half: int) -> int:
    pairs = reader.rotary_pairs
    n = pairs[block] if isinstance(pairs, list) else pairs
    return half if n is None else n


def _sliding_visible(reset: Tensor, C: int) -> Tensor:
    """Bool ``[B, S, S]``: query ``a`` reads ``s`` when both are in one document and
    ``0 ≤ a − s ≤ C``, written as loops over the stream."""
    B, S = reset.shape
    visible = torch.zeros(B, S, S, dtype=torch.bool)
    for b in range(B):
        start = 0
        for a in range(S):
            if bool(reset[b, a]):
                start = a
            for s in range(max(start, a - C), a + 1):
                visible[b, a, s] = True
    return visible


def _direct(reader: Reader, tokens: Tensor, visible: Tensor, top: Tensor | None = None) -> Tensor:
    """The reader's computation over a whole stream in one pass, written out: each block's
    queries and keys rotated by their position in the stream, the attention over the keys
    ``visible [B, S, S]`` names (``top`` for the last block, when given), softmax and the
    residual MLP."""
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
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        reads = top if top is not None and i == len(reader.blocks) - 1 else visible
        weights = scores.masked_fill(~reads[:, None], -math.inf).softmax(dim=-1)
        x = x + block.proj((weights @ v).transpose(1, 2).reshape(B, S, w))
        x = x + block.fc_out(F.gelu(block.fc_in(block.norm_mlp(x))))
    out: Tensor = reader.norm(x)
    return out


# --------------------------------------------------------------------------------------------------
# The key: opt-in, hashes, memory keys, arm identities, refusals
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "overrides"),
    [
        (RECIPE_A6, ()),
        (RECIPE_A6, ("model_name=hsn_t", "model.reader=transformer", "model.k_select=12")),
        (READER_LM, ()),
        (NOMEM, ("model.reader_width=224",)),
        (
            NOMEM,
            (
                "model.reader_width=160",
                "model.reader_layers=4",
                "model.reader_rotary_pairs=[20,20,20,0]",
            ),
        ),
        (TRANSFORMER, ()),
        (GRU, ()),
    ],
)
def test_the_key_is_opt_in_and_its_default_keeps_hashes_memory_keys_and_arm_ids(
    path: Path, overrides: tuple[str, ...]
) -> None:
    base = load_config(path, list(overrides))
    assert base.model.reader_carry == 0
    assert "reader_carry" not in json.loads(canonical_json(base))["model"]
    assert "model.reader_carry" in HASH_OMIT_AT_DEFAULT
    assert "reader_carry" not in config_view(base).get("model", {})  # the arm id's view
    if base.model.reader == "none":
        return  # the key is the reader's: without one only the default loads (refusals)
    named = load_config(path, [*overrides, "model.reader_carry=0"])
    assert config_hash(named) == config_hash(base)
    assert memory_key(named, "none", PROTOCOL) == memory_key(base, "none", PROTOCOL)
    assert config_view(named) == config_view(base)
    if base.model_name == "hsn_t" and base.model.memory and not base.model.key_address:
        return  # with the memory the carry rides with A5's address carry (refused without)
    on = load_config(path, [*overrides, "model.reader_carry=256"])
    assert config_hash(on) != config_hash(base)
    assert memory_key(on, "none", PROTOCOL) != memory_key(base, "none", PROTOCOL)
    assert config_view(on)["model"]["reader_carry"] == 256


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        (RECIPE_A6, ["model.reader_carry=8"], "carries the keys of model.reader"),
        (TRANSFORMER, ["model.reader_carry=8"], "carries the keys of model.reader"),
        (GRU, ["model.reader_carry=8"], "carries the keys of model.reader"),
        (  # with the memory the carry is its own opt-in (test_reader_carry_memory.py)
            RECIPE_A6,
            [
                "model_name=hsn_t",
                "model.reader=transformer",
                "model.reader_carry=8",
                "model.reader_top_words=4",
                "model.reader_rotary_pairs=0",
            ],
            "built for HSN-T without its memory only",
        ),
        (
            READER_LM,
            ["model.reader_position=learned", "model.reader_carry=8"],
            "a learned position is the position in the window",
        ),
        (READER_LM, ["model.reader_span=16", "model.reader_carry=8"], "leave model.reader_span"),
        (READER_LM, ["model.reader_carry=-1"], "reader_carry"),
        (READER_LM, ["model.reader_carry=many"], "reader_carry"),
    ],
)
def test_refusals(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(path, overrides)


def test_accepted_settings() -> None:
    for path, extra in ((READER_LM, []), (NOMEM, [])):
        for C in (1, 255, 256, 2000):
            assert load_config(path, [*extra, f"model.reader_carry={C}"]).model.reader_carry == C
    # with no rotated block the cache needs no rotation; accepted
    idle = load_config(READER_LM, ["model.reader_rotary_pairs=0", "model.reader_carry=8"])
    assert build_model(idle).reader.carry == 8


def test_the_models_refuse_a_missing_or_wrong_cache() -> None:
    reader = _reader(reader_carry=4)
    tokens, reset = _stream(2, 6, 0)
    with pytest.raises(ValueError, match=r"reads with read\(tokens, reset, cache\)"):
        reader(tokens, reset)
    with pytest.raises(ValueError, match="the cache must be"):
        reader.read(tokens, reset, None)  # type: ignore[arg-type]
    wrong = reader.init_cache(3, "cpu")  # three rows for a window of two
    with pytest.raises(ValueError, match="the cache must be"):
        reader.read(tokens, reset, wrong)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="read\\(\\) is the reader with model.reader_carry"):
        _reader().read(tokens, reset, wrong)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="has no cache"):
        _reader().check_cache(wrong, 3)
    cfg = load_config(NOMEM, [*TINY, "model.reader_carry=4"])
    model = build_model(cfg)
    ctx = torch.zeros_like(tokens)
    with pytest.raises(ValueError, match="the cache must be"):
        model(tokens, ctx, None, reset)
    stateless = build_model(load_config(NOMEM, list(TINY)))
    with pytest.raises(ValueError, match="carries no state"):
        stateless(tokens, ctx, model.init_state(2, "cpu"), reset)
    # a reader built from a ModelConfig alone still refuses what the configuration refuses
    with pytest.raises(ValueError, match="rotary positions"):
        Reader(
            ModelConfig.model_construct(
                **{
                    **ModelConfig(reader="transformer").model_dump(),
                    "reader_carry": 4,
                    "reader_position": "learned",
                }
            ),
            seed=0,
        )


# --------------------------------------------------------------------------------------------------
# The default: the reader and the models as they were
# --------------------------------------------------------------------------------------------------


def test_carry_zero_is_the_reader_as_it_was() -> None:
    base = Reader(ModelConfig(reader="transformer", reader_width=16, reader_heads=2), seed=3)
    named = Reader(
        ModelConfig(reader="transformer", reader_width=16, reader_heads=2, reader_carry=0), seed=3
    )
    assert base.carry == named.carry == 0
    assert base.init_cache(2, "cpu") is None and named.init_cache(2, "cpu") is None
    assert list(base.state_dict()) == list(named.state_dict())
    assert all(
        torch.equal(p, q) for p, q in zip(base.parameters(), named.parameters(), strict=True)
    )
    assert repr(base) == repr(named) and "carry" not in repr(base)
    base.eval()
    named.eval()
    for seed, resets in ((0, 0.0), (1, 0.2)):
        tokens, reset = _stream(3, 40, seed, resets)
        with torch.no_grad():
            want = _reference_forward(base, tokens, reset)
            assert torch.equal(base(tokens, reset), want) and torch.equal(
                named(tokens, reset), want
            )


@pytest.mark.parametrize("path", [NOMEM, READER_LM])
def test_carry_zero_models_are_stateless_and_unchanged(path: Path) -> None:
    sizes = list(TINY) if path == NOMEM else ["model.reader_width=16", "model.reader_heads=2"]
    seed_everything(0)
    base = build_model(load_config(path, sizes))
    seed_everything(0)
    named = build_model(load_config(path, [*sizes, "model.reader_carry=0"]))
    assert type(base) is type(named)
    assert all(
        torch.equal(p, q) for p, q in zip(base.parameters(), named.parameters(), strict=True)
    )
    assert base.init_state(2, "cpu") is None and named.init_state(2, "cpu") is None
    assert named.reset_state(None, torch.tensor([True, False])) is None
    tokens, reset = _stream(2, 24, 5, 0.1)
    ctx = torch.zeros_like(tokens)
    with torch.no_grad():
        a, sa, ia = base(tokens, ctx, None, reset)
        b, sb, ib = named(tokens, ctx, None, reset)
    assert sa is None and sb is None and ia == ib == {}
    assert torch.equal(a, b)


# --------------------------------------------------------------------------------------------------
# Window by window with the cache = one pass with a sliding span of C
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("T", "C"), [(8, 8), (8, 3), (5, 12), (16, 8), (7, 20), (48, 8)])
@pytest.mark.parametrize("pairs", [None, [4, 2, 0], 0])
def test_windows_with_the_cache_are_one_pass_with_a_sliding_span(
    T: int, C: int, pairs: object
) -> None:
    """Two references: the direct computation over the whole stream with the visible keys
    chosen by loops (``_direct``), and the reader without a cache with ``reader_span = C + 1``
    (it reads ``0 ≤ t − s < C + 1``) over the whole stream in one window."""
    reader = _reader(reader_carry=C, reader_rotary_pairs=pairs)
    span = _reader(reader_span=C + 1, reader_rotary_pairs=pairs)
    span.load_state_dict(reader.state_dict())
    tokens, reset = _stream(3, 48, 7)
    reset[1, 16] = reset[2, 8] = True  # a document starting at a window's first position
    reset[0, 23] = True  # and one at a window's last
    with torch.no_grad():
        got = _by_windows(reader, tokens, reset, T)
        direct = _direct(reader, tokens, _sliding_visible(reset, C))
        one = span(tokens, reset)
    assert got.shape == direct.shape == one.shape
    assert _gap(got, direct) <= 1e-11
    assert _gap(got, one) <= 1e-11
    # and the span matters: without the cache the windows read less
    if T < 48:
        cold = torch.cat(
            [span(tokens[:, lo : lo + T], reset[:, lo : lo + T]) for lo in range(0, 48, T)], dim=1
        )
        assert _gap(got, cold) > 1e-3


def test_the_cache_holds_the_last_C_positions_of_the_document_going_on() -> None:
    C, T = 6, 4
    reader = _reader(reader_carry=C)
    tokens, reset = _stream(2, 12, 1, 0.0)
    reset[1, 10] = True  # row 1: a new document two positions before the stream's end
    cache = reader.init_cache(2, "cpu")
    assert cache is not None
    keys, values, valid = cache
    assert keys.shape == values.shape == (3, 2, 2, C, 8) and keys.dtype == torch.float64
    assert valid.shape == (2, C) and not bool(valid.any())
    with torch.no_grad():
        for lo in range(0, 12, T):
            _, cache = reader.read(tokens[:, lo : lo + T], reset[:, lo : lo + T], cache)
    keys, values, valid = cache
    assert valid[0].tolist() == [True] * C
    assert valid[1].tolist() == [False] * 4 + [True] * 2  # the slots before the reset are not
    # the cached keys are the un-rotated keys of the stream's last C positions, block by block
    with torch.no_grad():
        x = reader.emb(tokens)
        h = reader.blocks[0].norm_attn(x)
        k = reader.blocks[0].qkv(h).split(16, dim=-1)[1].view(2, 12, 2, 8).transpose(1, 2)
    assert _gap(keys[0], k[:, :, -C:]) <= 1e-12
    reset_rows = reader.reset_cache(cache, torch.tensor([True, False]))
    assert not bool(reset_rows[2][0].any()) and torch.equal(reset_rows[2][1], valid[1])


# --------------------------------------------------------------------------------------------------
# Documents, gradients
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("T", [4, 8])
def test_a_document_reads_nothing_of_the_one_before_it(T: int) -> None:
    """Bytes of the document before a reset change nothing at or after it, cached or not; a
    document's first position reads itself alone."""
    C = 8
    reader = _reader(reader_carry=C, reader_rotary_pairs=[4, 2, 0])
    tokens, reset = _stream(2, 24, 3, 0.0)
    reset[0, 12] = True  # at a window's first position at T 4, inside one at T 8
    reset[1, 10] = True  # inside a window
    with torch.no_grad():
        out = _by_windows(reader, tokens, reset, T)
        before = tokens.clone()
        before[0, :12] = (before[0, :12] + 7) % 256
        before[1, :10] = (before[1, :10] + 7) % 256
        changed = _by_windows(reader, before, reset, T)
        assert torch.equal(changed[0, 12:], out[0, 12:]) and torch.equal(
            changed[1, 10:], out[1, 10:]
        )
        assert _gap(changed[0, :12], out[0, :12]) > 0.0
        for row, start in ((0, 12), (1, 10)):
            alone, _ = reader.read(
                tokens[row : row + 1, start : start + 1],
                torch.ones(1, 1, dtype=torch.bool),
                reader.init_cache(1, "cpu"),  # type: ignore[arg-type]
            )
            assert _gap(alone[0, 0], out[row, start]) <= 1e-12


def test_the_cache_is_detached() -> None:
    """No gradient crosses a window: the cache returned holds no graph, and window 2's
    gradients are the same whether window 1's cache was computed with a graph or without."""
    reader = _reader(reader_carry=6).train()
    tokens, reset = _stream(2, 16, 9, 0.0)
    first, second = (tokens[:, :8], reset[:, :8]), (tokens[:, 8:], reset[:, 8:])

    def grads(with_graph: bool) -> dict[str, Tensor]:
        reader.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(with_graph):
            _, cache = reader.read(*first, reader.init_cache(2, "cpu"))  # type: ignore[arg-type]
        assert all(not t.requires_grad and t.grad_fn is None for t in cache)
        out, nxt = reader.read(*second, cache)
        assert all(not t.requires_grad and t.grad_fn is None for t in nxt)
        out.square().mean().backward()
        return {n: p.grad.clone() for n, p in reader.named_parameters() if p.grad is not None}

    a, b = grads(True), grads(False)
    assert a.keys() == b.keys() and len(a) == len(list(reader.parameters()))
    assert all(_gap(a[n], b[n]) <= 1e-12 for n in a)


def test_the_models_carry_the_cache_as_their_state() -> None:
    """HSN-T without memory and the reader alone: windows with the state carried equal the
    model with ``reader_span = C + 1`` over the whole stream, logits included."""
    C, T = 8, 6
    for path, sizes in (
        (NOMEM, list(TINY)),
        (READER_LM, ["model.reader_width=16", "model.reader_heads=2"]),
    ):
        seed_everything(0)
        model = build_model(load_config(path, [*sizes, f"model.reader_carry={C}"])).double()
        span = build_model(load_config(path, [*sizes, f"model.reader_span={C + 1}"])).double()
        span.load_state_dict(model.state_dict())
        model.eval()
        span.eval()
        tokens, reset = _stream(3, 30, 4)
        ctx = torch.randint(0, 4, tokens.shape, generator=torch.Generator().manual_seed(2))
        state = model.init_state(3, "cpu")
        logits = []
        with torch.no_grad():
            for lo in range(0, 30, T):
                window = (tokens[:, lo : lo + T], ctx[:, lo : lo + T])
                out, state, info = model(*window, state, reset[:, lo : lo + T])
                logits.append(out)
                assert info == {}
            want, none, _ = span(tokens, ctx, None, reset)
        assert none is None
        assert _gap(torch.cat(logits, dim=1), want) <= 1e-10
        cleared = model.reset_state(state, torch.tensor([True, False, True]))[2]
        assert not bool(cleared[[0, 2]].any()) and torch.equal(cleared[1], state[2][1])


# --------------------------------------------------------------------------------------------------
# The trainer and the compiled window
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [NOMEM, READER_LM])
def test_the_generic_trainer_trains_and_evaluates_with_the_cache(
    path: Path, tmp_path: Path
) -> None:
    """Windows of 8 over documents of 16 to 32 bytes: every document spans windows, so the
    cache is read; the loss falls, the evaluation runs, the checkpoint holds the cache. The
    kv documents supervise their answers only, so a step whose windows hold none has no loss
    (NaN by the trainer's rule) and is left out."""
    steps = 40
    sizes = list(TINY) if path == NOMEM else ["model.reader_width=16", "model.reader_heads=2"]
    cfg = load_config(
        path,
        overrides=[
            *train_overrides(steps),
            *sizes,
            "train.lr=3e-3",
            "train.unroll=8",
            "model.reader_carry=8",
        ],
    )
    seed_everything(0)
    trainer = make_trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    assert type(trainer) is Trainer
    history = trainer.fit(*datasets(), steps)
    assert len(history.steps) == steps
    assert all(math.isfinite(s.loss) == (s.loss_positions > 0) for s in history.steps)
    losses = [s.loss for s in history.steps if s.loss_positions > 0]
    assert len(losses) >= 15
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.05
    assert len(history.evals) == 1 and math.isfinite(history.evals[0]["loss"])
    payload = torch.load(history.checkpoints[-1], map_location="cpu", weights_only=False)
    carried = payload["extra"]["carried_state"]
    assert isinstance(carried, tuple) and len(carried) == 3
    assert carried[0].shape == (2, 4, 2, 8, 8) and carried[2].dtype == torch.bool


def test_hsn_t_with_its_memory_is_unchanged_by_the_key() -> None:
    """The memory model never reads the key (refused by the configuration); its reader is
    called as before."""
    cfg = _cfg(
        *A5,
        "model_name=hsn_t",
        "model.reader=transformer",
        "model.reader_width=8",
        "model.reader_heads=2",
        "model.reader_carry=0",
    )
    assert HSNT.from_config(cfg).reader.carry == 0  # type: ignore[union-attr]


def test_the_carrying_model_compiles_as_one_graph_and_matches_eager() -> None:
    """``aot_eager`` with ``fullgraph``: the cache in and out of one graph, no data-dependent
    shape; forward, cache and gradients equal eager; the next window reuses the graph; the
    compiled-baseline wrapper passes the cache through."""
    from torch._dynamo.utils import counters

    cfg = load_config(NOMEM, [*TINY, "model.reader_carry=6", "model.reader_rotary_pairs=[4,0]"])
    seed_everything(0)
    model = build_model(cfg).train()
    assert isinstance(model, HSNTNoMemory)
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
                logits, states[i], info = forward(tokens, ctx, states[i], reset)
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
        ctx_cpu = resolve(load_config(NOMEM, [*TINY]).hardware)
        wrapper = CompiledBaseline(model, ctx_cpu, "default", "aot_eager")
        state = wrapper.init_state(3, "cpu")
        logits, state, _ = wrapper(tokens, ctx, state, reset)
        assert len(state) == 3 and wrapper.calls == 1 and logits.shape == (3, 8, 256)
        reader_lm = ReaderLM(load_config(READER_LM, ["model.reader_carry=4"]))
        lm = CompiledBaseline(reader_lm, ctx_cpu, "default", "aot_eager")
        logits, state, _ = lm(tokens, ctx, lm.init_state(3, "cpu"), reset)
        assert state[0].shape == (2, 3, 4, 4, 16) and logits.shape == (3, 8, 256)
    finally:
        torch._dynamo.reset()
