"""The reader's opt-in position keys (``hsn.model.reader``, ``docs/hsn_t.md``, 15 Nov 2026).

``model.reader_rotary_pairs`` (only the first ``n`` pairs of a head are rotated; a list gives
``n`` for each block, so the last block can be left without a position),
``model.reader_rotary_dtype`` (the rotation in the parameters' dtype under autocast),
``model.reader_rotary_base`` (the base of the rotation's frequency ladder) and
``model.reader_position = learned`` (the Transformer baseline's position table in place of the
rotation). They were built to test why the reader is worse on text than the baseline of its
shape; the cause they test is in the document.

These tests check that the defaults leave every configuration hash, memory key, parameter,
state dict and output as it was, and that each key does what it says: the unrotated pairs are
compared as they are and the output still reads distances only; a block with no rotated pair
compares queries and keys as they are; the rotation under bf16 autocast runs in float32 and is
closer to the exact one; another base turns pair ``i`` by ``t · base^(−i / half)``, in the table
and in the angles the forward pass computes itself alike; the learned reader is the Transformer
baseline's computation with the same weights, and reads the window position.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from hsn.baselines.reader_lm import ReaderLM
from hsn.baselines.transformer import TransformerBaseline
from hsn.config import (
    HASH_OMIT_AT_DEFAULT,
    ModelConfig,
    canonical_json,
    config_hash,
    load_config,
)
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import memory_key
from hsn.model import reader as reader_mod
from hsn.model.hsn_t import HSNT, HSNTNoMemory
from hsn.model.reader import Reader
from hsn.runner import build_model, make_trainer
from hsn.train.loop import Trainer
from hsn.train.memory_plan import memory_off_param_count, reader_param_count
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, train_overrides
from test_a6_feedback_detach import A5, _cfg, _start, _window
from test_reader_rotary_table import _reference_cos_sin, _reference_forward
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")
L = reader_mod._ROTARY_TABLE

REPO = Path(__file__).resolve().parents[1]
NOMEM = REPO / "configs" / "experiments" / "hsn_t_nomem.yaml"
READER_LM = REPO / "configs" / "baselines" / "reader_w64l2.yaml"
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
TRANSFORMER = REPO / "configs" / "baselines" / "transformer_2m.yaml"

KEYS = (
    "reader_position",
    "reader_positions",
    "reader_rotary_pairs",
    "reader_rotary_dtype",
    "reader_rotary_base",
)
DEFAULTS = (
    "model.reader_position=rotary",
    "model.reader_positions=256",
    "model.reader_rotary_pairs=null",
    "model.reader_rotary_dtype=working",
    "model.reader_rotary_base=10000.0",
)
#: one setting of each key away from its default
PAIRS = ("model.reader_rotary_pairs=2",)
#: one count for each of the two blocks: the last block is not rotated
TOP = ("model.reader_rotary_pairs=[2,0]",)
#: no block is rotated: the causal mask is the only order signal
NOROT = ("model.reader_rotary_pairs=0",)
WIDE = ("model.reader_rotary_dtype=parameters",)
#: a larger rotary base: longer wavelengths for the slower pairs
BASE = ("model.reader_rotary_base=1000000.0",)
LEARNED = ("model.reader_position=learned",)
SWITCHES = [PAIRS, TOP, NOROT, WIDE, BASE, LEARNED]
#: the memory-off recipe at a size the CPU runs in a moment (two heads of four pairs)
TINY = ("model.d=8", "model.context_dim=8", "model.hidden=16", "model.reader_width=16")
TINY = (*TINY, "model.reader_heads=2")


def _gap(a: Tensor, b: Tensor) -> float:
    return float((a - b).detach().abs().max())


def _reader(dtype: torch.dtype = torch.float64, **keys: object) -> Reader:
    """A reader of width 16 with two heads (four pairs a head) and two blocks, away from its
    start so that every weight matters."""
    cfg = ModelConfig(reader="transformer", reader_width=16, reader_heads=2, **keys)  # type: ignore[arg-type]
    reader = Reader(cfg, seed=0).to(dtype)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for p in reader.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=torch.float64).to(dtype))
    return reader.eval()


def _bytes(B: int, T: int, seed: int) -> Tensor:
    return torch.randint(0, 256, (B, T), generator=torch.Generator().manual_seed(seed))


def _none(tokens: Tensor) -> Tensor:
    return torch.zeros_like(tokens, dtype=torch.bool)


# --------------------------------------------------------------------------------------------------
# The keys: opt-in, hashes, memory keys, refusals
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "overrides"),
    [
        (RECIPE_A6, ()),
        (RECIPE_A6, ("model_name=hsn_t", "model.reader=transformer", "model.k_select=12")),
        (READER_LM, ()),
        (NOMEM, ("model.reader_width=224",)),
        (TRANSFORMER, ()),
    ],
)
def test_keys_are_opt_in_and_the_defaults_keep_hashes_and_memory_keys(
    path: Path, overrides: tuple[str, ...]
) -> None:
    base = load_config(path, list(overrides))
    assert (base.model.reader_position, base.model.reader_positions) == ("rotary", 256)
    assert base.model.reader_rotary_pairs is None and base.model.reader_rotary_dtype == "working"
    assert base.model.reader_rotary_base == 10_000.0
    dumped = json.loads(canonical_json(base))["model"]
    assert not any(key in dumped for key in KEYS)
    assert all(f"model.{key}" in HASH_OMIT_AT_DEFAULT for key in KEYS)
    protocol = {"name": "train_eval", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
    if base.model.reader == "none":
        return  # the keys are the reader's: without one only the defaults load (below)
    named = load_config(path, [*overrides, *DEFAULTS])
    assert config_hash(named) == config_hash(base)
    assert memory_key(named, "none", protocol) == memory_key(base, "none", protocol)
    assert not any(key in base.model.model_dump(exclude_defaults=True) for key in KEYS)
    for switch in SWITCHES:
        other = load_config(path, [*overrides, *switch])
        assert config_hash(other) != config_hash(base), switch
        assert memory_key(other, "none", protocol) != memory_key(base, "none", protocol)


def test_the_memory_keys_of_the_arms_before_the_keys_are_unchanged() -> None:
    """The values ``tests/test_hsn_t_no_memory.py`` pins from main 6bbf53d, before these keys:
    the stored step-time and memory estimates of those arms still apply."""
    protocol = {"name": "reads_state", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
    assert memory_key(load_config(RECIPE_A6), "pdg", protocol) == "c33ec88833506533"
    hybrid = ["model_name=hsn_t", "model.reader=transformer", "model.k_select=12"]
    assert memory_key(load_config(RECIPE_A6, hybrid), "pdg", protocol) == "c8bcf4024c9285fe"


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        (RECIPE_A6, ["model.reader_position=learned"], "positions of model.reader"),
        (RECIPE_A6, ["model.reader_rotary_pairs=2"], "positions of model.reader"),
        (RECIPE_A6, ["model.reader_rotary_dtype=parameters"], "positions of model.reader"),
        (TRANSFORMER, ["model.reader_position=learned"], "positions of model.reader"),
        (READER_LM, ["model.reader_positions=128"], "size of the learned position table"),
        (READER_LM, [*LEARNED, "model.reader_rotary_pairs=2"], "has no rotation"),
        (READER_LM, [*LEARNED, *WIDE], "has no rotation"),
        (RECIPE_A6, ["model.reader_rotary_pairs=[2,0]"], "positions of model.reader"),
        (READER_LM, [*LEARNED, "model.reader_rotary_pairs=[8,0]"], "has no rotation"),
        (READER_LM, ["model.reader_rotary_pairs=9"], "between 0 and the 8 pairs of a head"),
        (READER_LM, ["model.reader_rotary_pairs=-1"], "between 0 and the 8 pairs of a head"),
        (READER_LM, ["model.reader_rotary_pairs=[8,9]"], "between 0 and the 8 pairs of a head"),
        (READER_LM, ["model.reader_rotary_pairs=[8]"], "names 1 blocks"),
        (READER_LM, ["model.reader_rotary_pairs=[8,8,0]"], "names 3 blocks"),
        (READER_LM, ["model.reader_rotary_pairs=half"], "reader_rotary_pairs"),
        (READER_LM, [*LEARNED, "model.reader_positions=128"], "train.unroll=256"),
        (NOMEM, [*LEARNED, "train.unroll=512"], "raise model.reader_positions"),
        (READER_LM, ["model.reader_position=sinusoidal"], "rotary"),
        (READER_LM, ["model.reader_rotary_dtype=float64"], "working"),
        (RECIPE_A6, ["model.reader_rotary_base=100000.0"], "positions of model.reader"),
        (TRANSFORMER, ["model.reader_rotary_base=100000.0"], "positions of model.reader"),
        (READER_LM, [*LEARNED, "model.reader_rotary_base=100000.0"], "has no rotation"),
        (READER_LM, ["model.reader_rotary_base=1.0"], "finite number above 1"),
        (READER_LM, ["model.reader_rotary_base=0.5"], "finite number above 1"),
        (READER_LM, ["model.reader_rotary_base=-10000"], "finite number above 1"),
        (READER_LM, ["model.reader_rotary_base=.inf"], "finite number above 1"),
        (READER_LM, ["model.reader_rotary_base=.nan"], "finite number above 1"),
        (RECIPE_A6, ["model.reader_rotary_base=0"], "finite number above 1"),
        (READER_LM, ["model.reader_rotary_base=long"], "reader_rotary_base"),
    ],
)
def test_refusals(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(path, overrides)


def test_accepted_settings() -> None:
    assert load_config(READER_LM, ["model.reader_rotary_pairs=8"]).model.reader_rotary_pairs == 8
    assert load_config(READER_LM, ["model.reader_rotary_pairs=0"]).model.reader_rotary_pairs == 0
    top = load_config(READER_LM, ["model.reader_rotary_pairs=[8,0]"])
    assert top.model.reader_rotary_pairs == [8, 0]
    assert json.loads(canonical_json(top))["model"]["reader_rotary_pairs"] == [8, 0]
    deep = ["model.reader_layers=4", "model.reader_rotary_pairs=[8,8,8,0]"]
    assert build_model(load_config(READER_LM, deep)).reader._block_pairs == [None, None, None, 0]
    long = load_config(NOMEM, [*LEARNED, "train.unroll=512", "model.reader_positions=512"])
    assert long.model.reader_positions == 512
    short = load_config(READER_LM, [*LEARNED, "train.unroll=64", "model.reader_positions=64"])
    assert build_model(short).reader.pos.shape == (64, 64)
    # the base: a float, written as YAML reads it or not (1e5 is a string to YAML 1.1)
    for text in ("1e5", "100000", "100000.0", "1.0e+5"):
        cfg = load_config(READER_LM, [f"model.reader_rotary_base={text}"])
        assert cfg.model.reader_rotary_base == 1e5
    keys = ["model.reader_rotary_base=1e6", "model.reader_rotary_pairs=[8,4]", *WIDE]
    assert build_model(load_config(READER_LM, keys)).reader.rotary_base == 1e6
    # with no rotated pair in any block the base acts on nothing; accepted, as the dtype key is
    idle = load_config(READER_LM, [*NOROT, "model.reader_rotary_base=1e6", *WIDE])
    assert idle.model.reader_rotary_base == 1e6


# --------------------------------------------------------------------------------------------------
# The defaults: the reader as it was
# --------------------------------------------------------------------------------------------------


def test_default_reader_is_the_reader_as_it_was() -> None:
    """Named or not, the defaults build the same module: the old state-dict keys, the same
    draws, and the forward of before the keys (``_reference_forward`` is that code, kept in
    ``tests/test_reader_rotary_table.py``), bit for bit."""
    base = ModelConfig(reader="transformer", reader_width=16, reader_heads=2)
    named = ModelConfig(
        reader="transformer",
        reader_width=16,
        reader_heads=2,
        reader_position="rotary",
        reader_positions=256,
        reader_rotary_pairs=None,
        reader_rotary_dtype="working",
        reader_rotary_base=10_000.0,
    )
    a, b = Reader(base, seed=3).eval(), Reader(named, seed=3).eval()
    assert a.pos is None and "pos" not in a.state_dict()
    assert [name for name, _ in a.named_parameters()] == [name for name, _ in b.named_parameters()]
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters(), strict=True))
    per_block = ("norm_attn", "qkv", "proj", "norm_mlp", "fc_in", "fc_out")
    ends = ("weight", "bias")
    assert list(a.state_dict()) == [
        "emb.weight",
        *(f"blocks.{i}.{part}.{t}" for i in (0, 1) for part in per_block for t in ends),
        "norm.weight",
        "norm.bias",
    ]
    assert repr(a) == repr(b) and "position" not in repr(a) and "rotary_pairs" not in repr(a)
    for seed, resets in ((0, 0.0), (1, 0.2)):
        tokens = _bytes(3, 40, seed)
        reset = torch.rand(3, 40, generator=torch.Generator().manual_seed(seed)) < resets
        with torch.no_grad():
            want = _reference_forward(a, tokens, reset)
            assert torch.equal(a(tokens, reset), want) and torch.equal(b(tokens, reset), want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_every_pair_named_is_the_default_bit_for_bit(dtype: torch.dtype) -> None:
    """``reader_rotary_pairs`` at the number of pairs a head has rotates all of them."""
    default, full = _reader(dtype), _reader(dtype, reader_rotary_pairs=4)
    assert torch.equal(full._rot_cos, default._rot_cos)
    assert torch.equal(full._rot_sin, default._rot_sin)
    tokens = _bytes(2, 30, 5)
    with torch.no_grad():
        assert torch.equal(full(tokens, _none(tokens)), default(tokens, _none(tokens)))


# --------------------------------------------------------------------------------------------------
# model.reader_rotary_pairs
# --------------------------------------------------------------------------------------------------


def test_the_pairs_beyond_n_are_not_rotated_and_the_first_n_are_as_before() -> None:
    default, part = _reader(), _reader(reader_rotary_pairs=2)
    assert part.rotary_pairs == 2 and "rotary_pairs=2" in repr(part)
    # the table: the first two pairs hold the default's angles, the others the angle zero
    assert torch.equal(part._rot_cos[:, :2], default._rot_cos[:, :2])
    assert torch.equal(part._rot_sin[:, :2], default._rot_sin[:, :2])
    assert bool((part._rot_cos[:, 2:] == 1.0).all()) and bool((part._rot_sin[:, 2:] == 0.0).all())
    assert float(default._rot_sin[1:, 2:].abs().min()) > 0.0  # which the default does rotate
    # the rotation: a head's width is 8, so its pairs are (i, i + 4); pairs 2 and 3 come back
    # as they went in, bit for bit, in the working dtypes of a run
    for dtype in (torch.float64, torch.float32, torch.bfloat16):
        g = torch.Generator().manual_seed(4)
        x = torch.randn(2, 2, 12, 8, generator=g).to(dtype)
        cos, sin = part._rot_cos[:12].to(dtype), part._rot_sin[:12].to(dtype)
        out = reader_mod._rotate(x, cos, sin)
        for i in (2, 3, 6, 7):
            assert torch.equal(out[..., i], x[..., i]), (dtype, i)
        for i in (0, 1, 4, 5):
            assert not torch.equal(out[..., 1:, i], x[..., 1:, i]), (dtype, i)
    # a window longer than the table computes the same angles in the call
    assert torch.equal(
        reader_mod._rotary_angles(7, 4, torch.device("cpu"), 2)[:, :2],
        reader_mod._rotary_angles(7, 4, torch.device("cpu"))[:, :2],
    )
    tokens = _bytes(2, 30, 6)
    with torch.no_grad():
        assert _gap(part(tokens, _none(tokens)), default(tokens, _none(tokens))) > 1e-3


def test_unrotated_pairs_compare_a_query_with_a_key_at_every_distance() -> None:
    """One block whose queries and keys live in the unrotated pairs alone: its attention does
    not read where the earlier bytes sit, so permuting them leaves the last output unchanged.
    With every pair rotated the same weights do read it."""
    cfg = {"reader": "transformer", "reader_width": 16, "reader_heads": 2, "reader_layers": 1}
    part = Reader(ModelConfig(**cfg, reader_rotary_pairs=2), seed=0).double().eval()  # type: ignore[arg-type]
    full = Reader(ModelConfig(**cfg), seed=0).double().eval()  # type: ignore[arg-type]
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for p in part.parameters():
            p.add_(0.5 * torch.randn(p.shape, generator=g, dtype=torch.float64))
        qkv = part.blocks[0].qkv
        rotated = torch.tensor([0, 1, 4, 5, 8, 9, 12, 13])  # pairs 0 and 1 of the two heads
        for offset in (0, 16):  # the rows of q and of k
            qkv.weight[rotated + offset] = 0.0
            qkv.bias[rotated + offset] = 0.0
        full.load_state_dict(part.state_dict())
    tokens = _bytes(1, 20, 2)
    shuffled = tokens.clone()
    shuffled[:, :19] = tokens[:, torch.randperm(19, generator=g)]
    no = _none(tokens)
    with torch.no_grad():
        assert _gap(part(tokens, no)[:, -1], part(shuffled, no)[:, -1]) <= 1e-12
        assert _gap(full(tokens, no)[:, -1], full(shuffled, no)[:, -1]) > 1e-6


def test_with_fewer_pairs_the_reader_still_reads_distances_only() -> None:
    """As ``test_reader_reads_distances_not_window_positions`` of the HSN-T tests: the same
    bytes give the same output wherever their document sits in the window."""
    reader = _reader(reader_rotary_pairs=2)
    body = _bytes(1, 10, 4)
    outs = []
    for lead in (0, 3, 17):
        tokens = torch.cat([_bytes(1, lead, 20 + lead), body], dim=1)
        reset = _none(tokens)
        reset[:, lead] = True  # the body is a document of its own
        with torch.no_grad():
            outs.append(reader(tokens, reset)[:, lead:])
    assert _gap(outs[0], outs[1]) <= 1e-12 and _gap(outs[0], outs[2]) <= 1e-12


# --------------------------------------------------------------------------------------------------
# model.reader_rotary_pairs as a list: one count for each block
# --------------------------------------------------------------------------------------------------


def _blocks_by_hand(reader: Reader, tokens: Tensor, reset: Tensor, pairs: list[int]) -> Tensor:
    """The reader's forward with each block handed the cos / sin of its own count, computed
    apart from the reader's table: the angle zero beyond the count, and no rotation at all for
    the count zero."""
    x = reader.emb(tokens)
    mask = reader.attention_mask(reset)
    for block, n in zip(reader.blocks, pairs, strict=True):
        if n == 0:
            x = block(x, mask, None, None)
        else:
            angles = reader_mod._rotary_angles(tokens.shape[1], 4, tokens.device, n)
            x = block(x, mask, angles.cos().to(x.dtype), angles.sin().to(x.dtype))
    out: Tensor = reader.norm(x)
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_one_count_for_each_block_names_the_same_reader_as_one_count(dtype: torch.dtype) -> None:
    """``[n, n]`` is ``n``, and ``[4, 4]`` (every pair of a head) is the default, bit for bit:
    the same parameters, the same table, the same outputs."""
    tokens = _bytes(2, 30, 5)
    reset = _none(tokens)
    reset[1, 13] = True
    for listed, single in (([2, 2], 2), ([4, 4], None), ([0, 0], 0)):
        a = _reader(dtype, reader_rotary_pairs=listed)
        b = _reader(dtype, reader_rotary_pairs=single)
        assert a.state_dict().keys() == b.state_dict().keys()
        assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters(), strict=True))
        assert torch.equal(a._rot_cos, b._rot_cos) and torch.equal(a._rot_sin, b._rot_sin)
        with torch.no_grad():
            assert torch.equal(a(tokens, reset), b(tokens, reset)), listed


@pytest.mark.parametrize("pairs", [[4, 0], [3, 1], [0, 4], [1, 4]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_each_block_rotates_its_own_pairs(
    pairs: list[int], dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader, default = _reader(dtype, reader_rotary_pairs=pairs), _reader(dtype)
    assert reader.rotary_pairs == pairs and f"rotary_pairs={pairs}" in repr(reader)
    # the parameters and the table are the default's: the counts act in the forward pass
    assert all(
        torch.equal(p, q) for p, q in zip(reader.parameters(), default.parameters(), strict=True)
    )
    assert torch.equal(reader._rot_cos, default._rot_cos)
    calls: list[int] = []
    rotate = reader_mod._rotate

    def spy(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        calls.append(int((sin[1:] != 0).any(dim=0).sum()))  # the pairs this call rotates
        return rotate(x, cos, sin)

    monkeypatch.setattr(reader_mod, "_rotate", spy)
    tokens = _bytes(2, 30, 6)
    reset = _none(tokens)
    reset[0, 9] = True
    with torch.no_grad():
        out = reader(tokens, reset)
        # a block with the count zero never calls the rotation; the others rotate q and k
        assert calls == [n for n in pairs if n for _ in range(2)]
        assert torch.equal(out, _blocks_by_hand(reader, tokens, reset, pairs))
        assert _gap(out, default(tokens, reset)) > 1e-3


def test_a_block_without_rotation_compares_queries_and_keys_as_they_are() -> None:
    """One block with the count zero: its attention gets no position, so the output at a
    position is the same for every order of the bytes before it. The same weights with every
    pair rotated read the order. (Stacked, the causal mask lets a later block see order through
    the earlier blocks' outputs, so this is a property of the block and not of a deep reader.)"""
    cfg = {"reader": "transformer", "reader_width": 16, "reader_heads": 2, "reader_layers": 1}
    blind = Reader(ModelConfig(**cfg, reader_rotary_pairs=[0]), seed=0).double().eval()  # type: ignore[arg-type]
    full = Reader(ModelConfig(**cfg), seed=0).double().eval()  # type: ignore[arg-type]
    assert blind._block_pairs == [0] and full._block_pairs == [None]
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for p in blind.parameters():
            p.add_(0.5 * torch.randn(p.shape, generator=g, dtype=torch.float64))
        full.load_state_dict(blind.state_dict())
    tokens = _bytes(1, 20, 2)
    shuffled = tokens.clone()
    shuffled[:, :19] = tokens[:, torch.randperm(19, generator=g)]
    no = _none(tokens)
    with torch.no_grad():
        assert _gap(blind(tokens, no)[:, -1], blind(shuffled, no)[:, -1]) <= 1e-12
        assert _gap(full(tokens, no)[:, -1], full(shuffled, no)[:, -1]) > 1e-6


@pytest.mark.parametrize("pairs", [[4, 0], [3, 1], 0])
def test_with_a_count_for_each_block_the_reader_still_reads_distances_only(
    pairs: list[int] | int,
) -> None:
    reader = _reader(reader_rotary_pairs=pairs)
    body = _bytes(1, 10, 4)
    outs = []
    for lead in (0, 3, 17):
        tokens = torch.cat([_bytes(1, lead, 20 + lead), body], dim=1)
        reset = _none(tokens)
        reset[:, lead] = True  # the body is a document of its own
        with torch.no_grad():
            outs.append(reader(tokens, reset)[:, lead:])
    assert _gap(outs[0], outs[1]) <= 1e-12 and _gap(outs[0], outs[2]) <= 1e-12


def test_a_count_for_each_block_under_bf16_autocast_with_the_float32_rotation() -> None:
    """The two keys together: the mixed table is made in the parameters' dtype, the rotation
    runs in it and the unrotated block is untouched."""
    keys = {"reader_rotary_pairs": [3, 0], "reader_rotary_dtype": "parameters"}
    reader = _reader(torch.float32, **keys).train()
    tokens = _bytes(2, 24, 9)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = reader(tokens, _none(tokens))
    assert out.dtype == torch.float32 and bool(torch.isfinite(out).all())
    out.float().square().mean().backward()
    assert all(bool(torch.isfinite(p.grad).all()) for p in reader.parameters())


# --------------------------------------------------------------------------------------------------
# model.reader_rotary_dtype
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_without_autocast_the_rotation_dtype_changes_nothing(dtype: torch.dtype) -> None:
    default, wide = _reader(dtype), _reader(dtype, reader_rotary_dtype="parameters")
    assert all(block.rotate_in == "parameters" for block in wide.blocks)
    assert all(block.rotate_in == "working" for block in default.blocks)
    tokens = _bytes(2, 30, 7)
    reset = _none(tokens)
    reset[1, 11] = True
    with torch.no_grad():
        assert torch.equal(wide(tokens, reset), default(tokens, reset))


def test_under_bf16_autocast_the_rotation_runs_in_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    """What the key says, read off the calls: the default rotates bf16 queries and keys with
    bf16 cos / sin, the key rotates them as float32 with the float32 table and hands the
    attention bf16 again. Against the exact rotation of the same bf16 queries (float64) the
    key's rotated queries are closer."""
    seen: dict[str, list[tuple[Tensor, Tensor, Tensor, Tensor]]] = {"working": [], "parameters": []}
    attended: dict[str, list[torch.dtype]] = {"working": [], "parameters": []}
    rotate, attention = reader_mod._rotate, F.scaled_dot_product_attention
    current = "working"

    def spy_rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        out = rotate(x, cos, sin)
        seen[current].append((x, cos, sin, out))
        return out

    def spy_attention(q: Tensor, k: Tensor, v: Tensor, **kwargs: Tensor) -> Tensor:
        attended[current].extend([q.dtype, k.dtype])
        return attention(q, k, v, **kwargs)

    monkeypatch.setattr(reader_mod, "_rotate", spy_rotate)
    monkeypatch.setattr(reader_mod.F, "scaled_dot_product_attention", spy_attention)
    tokens = _bytes(2, 40, 8)
    outputs = {}
    for current in ("working", "parameters"):
        reader = _reader(torch.float32, reader_rotary_dtype=current, reader_layers=1)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            outputs[current] = reader(tokens, _none(tokens))
    assert not torch.equal(outputs["working"].float(), outputs["parameters"].float())
    errors = {}
    for name, dtype in (("working", torch.bfloat16), ("parameters", torch.float32)):
        assert len(seen[name]) == 2 and attended[name] == [torch.bfloat16, torch.bfloat16]
        x, cos, sin, out = seen[name][0]  # the queries of the block
        assert x.dtype == cos.dtype == sin.dtype == out.dtype == dtype, name
        # the same bf16 queries reach both rotations (the key casts them up, without a change)
        assert torch.equal(x.double(), seen["working"][0][0].double())
        table = reader_mod._rotary_angles(40, 4, torch.device("cpu"))
        exact = rotate(x.double(), table.cos(), table.sin())
        # what the attention is handed: bf16 in both cases
        errors[name] = float((out.to(torch.bfloat16).double() - exact).abs().mean())
    assert errors["parameters"] < 0.75 * errors["working"], errors


def test_gradients_flow_through_the_float32_rotation_under_autocast() -> None:
    reader = _reader(torch.float32, reader_rotary_dtype="parameters").train()
    tokens = _bytes(2, 24, 9)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = reader(tokens, _none(tokens))
    assert out.dtype == torch.float32  # the final LayerNorm, as with the default
    out.float().square().mean().backward()
    for name, p in reader.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
    assert float(reader.blocks[0].qkv.weight.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------------------------------
# model.reader_rotary_base
# --------------------------------------------------------------------------------------------------

#: the two bases of text_rotary_base.yaml, and one below the default
BASES = (100_000.0, 1_000_000.0, 500.0)
CPU = torch.device("cpu")


def _ladder(T: int, half: int, base: float) -> Tensor:
    """``t · base^(−i / half)`` ``[T, half]`` in Python floats, apart from the reader's code."""
    rows = [[t * base ** (-i / half) for i in range(half)] for t in range(T)]
    return torch.tensor(rows, dtype=torch.float64)


def _forward_with_angles(reader: Reader, tokens: Tensor, reset: Tensor, base: float) -> Tensor:
    """The reader's forward with the angles of ``base`` computed here for the whole window,
    never read from the reader's table."""
    angles = reader_mod._rotary_angles(tokens.shape[1], 4, tokens.device, base=base)
    x = reader.emb(tokens)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    mask = reader.attention_mask(reset)
    for block in reader.blocks:
        x = block(x, mask, cos, sin)
    out: Tensor = reader.norm(x)
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_default_base_is_the_old_constant_bit_for_bit(dtype: torch.dtype) -> None:
    """Named or not, the default base is the constant of before the key: the same angles as the
    old code (literal 10000), the same table, parameters, state dict and outputs."""
    assert ModelConfig.model_fields["reader_rotary_base"].default == 10_000.0
    assert reader_mod._ROTARY_BASE == 10_000.0
    for half in (1, 4, 28):
        angles = reader_mod._rotary_angles(L, half, CPU, None, 10_000.0)
        assert torch.equal(angles, reader_mod._rotary_angles(L, half, CPU))
        old_cos, old_sin = _reference_cos_sin(L, half, CPU)
        assert torch.equal(angles.cos(), old_cos) and torch.equal(angles.sin(), old_sin)
    default, named = _reader(dtype), _reader(dtype, reader_rotary_base=10_000.0)
    assert default.rotary_base == named.rotary_base == 10_000.0
    assert repr(default) == repr(named) and "rotary_base" not in repr(default)
    assert torch.equal(named._rot_cos, default._rot_cos)
    assert torch.equal(named._rot_sin, default._rot_sin)
    assert list(named.state_dict()) == list(default.state_dict())
    assert all(
        torch.equal(p, q) for p, q in zip(named.parameters(), default.parameters(), strict=True)
    )
    tokens = _bytes(2, 30, 5)
    reset = _none(tokens)
    reset[1, 13] = True
    with torch.no_grad():
        want = _reference_forward(default, tokens, reset)
        assert torch.equal(default(tokens, reset), want) and torch.equal(named(tokens, reset), want)


@pytest.mark.parametrize("base", BASES)
def test_another_base_turns_pair_i_by_t_times_base_to_the_minus_i_over_half(base: float) -> None:
    for half in (4, 28):
        angles = reader_mod._rotary_angles(300, half, CPU, base=base)
        torch.testing.assert_close(angles, _ladder(300, half, base), rtol=1e-14, atol=0.0)
        # pair 0 turns by one radian a position whatever the base
        assert torch.equal(angles[:, 0], torch.arange(300, dtype=torch.float64))
        # pair i against the default's: slower by (base / 10000)^(i / half), at every position
        default = reader_mod._rotary_angles(300, half, CPU)
        ratio = angles[1:, 1:] / default[1:, 1:]
        exponent = torch.arange(1, half, dtype=torch.float64) / half
        want = ((10_000.0 / base) ** exponent).expand_as(ratio)
        torch.testing.assert_close(ratio, want, rtol=1e-13, atol=0.0)
    # with fewer rotated pairs the first n follow the base and the others keep the angle zero
    part = reader_mod._rotary_angles(300, 4, CPU, 2, base)
    assert torch.equal(part[:, :2], reader_mod._rotary_angles(300, 4, CPU, base=base)[:, :2])
    assert bool((part[:, 2:] == 0.0).all())
    # the reader: a table of those angles' cos / sin, the default's parameters, other outputs
    for dtype in (torch.float32, torch.float64):
        reader, default = _reader(dtype, reader_rotary_base=base), _reader(dtype)
        assert reader.rotary_base == base and f"rotary_base={base:g}" in repr(reader)
        table = reader_mod._rotary_angles(L, 4, CPU, base=base)
        assert torch.equal(reader._rot_cos, table.cos().to(dtype))
        assert torch.equal(reader._rot_sin, table.sin().to(dtype))
        exact = _ladder(L, 4, base)
        close = 1e-6 if dtype == torch.float32 else 1e-12
        torch.testing.assert_close(reader._rot_cos.double(), exact.cos(), rtol=0.0, atol=close)
        assert list(reader.state_dict()) == list(default.state_dict())
        assert all(
            torch.equal(p, q)
            for p, q in zip(reader.parameters(), default.parameters(), strict=True)
        )
        tokens = _bytes(2, 30, 6)
        with torch.no_grad():
            assert _gap(reader(tokens, _none(tokens)), default(tokens, _none(tokens))) > 1e-4


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_table_and_the_computed_angles_agree_for_another_base(dtype: torch.dtype) -> None:
    """The forward pass reads the table or computes the angles itself (past the table, without
    one as on MPS, for weights cast behind the module): every path has the same base."""
    base = 1_000_000.0
    reader = _reader(dtype, reader_rotary_base=base)
    x = torch.zeros(1, dtype=dtype)
    for T in (1, 7, 256, L):
        cos, sin = reader._rotation(T, x)
        assert cos.data_ptr() == reader._rot_cos.data_ptr()  # read from the table
        angles = reader_mod._rotary_angles(T, 4, CPU, base=base)
        assert torch.equal(cos, angles.cos().to(dtype)) and torch.equal(sin, angles.sin().to(dtype))
    # past the table: computed in the call, the rows inside it equal to the table's
    long_cos, long_sin = reader._rotation(L + 5, x)
    assert long_cos.data_ptr() != reader._rot_cos.data_ptr()
    assert torch.equal(long_cos[:L], reader._rot_cos) and torch.equal(long_sin[:L], reader._rot_sin)
    assert _gap(long_cos[L:], torch.cos(_ladder(L + 5, 4, base)[L:]).to(dtype)) <= 1e-6
    tokens = _bytes(2, 40, 7)
    reset = _none(tokens)
    reset[0, 17] = True
    with torch.no_grad():
        want = _forward_with_angles(reader, tokens, reset, base)
        assert torch.equal(reader(tokens, reset), want)  # the table
        long_tokens = _bytes(1, L + 3, 8)
        long_reset = _none(long_tokens)
        long_want = _forward_with_angles(reader, long_tokens, long_reset, base)
        assert torch.equal(reader(long_tokens, long_reset), long_want)  # past the table
        bare = copy.deepcopy(reader)
        bare._rot_cos = bare._rot_sin = None  # MPS's path: no table
        assert torch.equal(bare(tokens, reset), want)
    # cast with the module, the table is made again with the base; cast behind it, the call
    # computes the base's angles in the new dtype
    other = torch.float64 if dtype == torch.float32 else torch.float32
    moved = copy.deepcopy(reader).to(other)
    table = reader_mod._rotary_angles(L, 4, CPU, base=base)
    assert torch.equal(moved._rot_cos, table.cos().to(other))
    behind = copy.deepcopy(reader)
    behind.load_state_dict(
        {name: value.to(other) for name, value in reader.state_dict().items()}, assign=True
    )
    assert behind._rot_cos is not None and behind._rot_cos.dtype == dtype
    with torch.no_grad():
        want = _forward_with_angles(moved, tokens, reset, base)
        assert torch.equal(moved(tokens, reset), want) and torch.equal(behind(tokens, reset), want)


@pytest.mark.parametrize("pairs", [None, [4, 2]])
def test_with_another_base_the_reader_still_reads_distances_only(pairs: list[int] | None) -> None:
    reader = _reader(reader_rotary_base=1_000_000.0, reader_rotary_pairs=pairs)
    body = _bytes(1, 10, 4)
    outs = []
    for lead in (0, 3, 17):
        tokens = torch.cat([_bytes(1, lead, 20 + lead), body], dim=1)
        reset = _none(tokens)
        reset[:, lead] = True  # the body is a document of its own
        with torch.no_grad():
            outs.append(reader(tokens, reset)[:, lead:])
    assert _gap(outs[0], outs[1]) <= 1e-12 and _gap(outs[0], outs[2]) <= 1e-12


@pytest.mark.parametrize(
    ("path", "overrides"), [(READER_LM, ()), (NOMEM, ("model.reader_width=224",))]
)
def test_the_base_at_its_default_keeps_the_hash_and_the_memory_key_and_another_changes_them(
    path: Path, overrides: tuple[str, ...]
) -> None:
    protocol = {"name": "train_eval", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
    base = load_config(path, list(overrides))
    for text in ("10000", "10000.0", "1e4", "1.0e+4"):
        named = load_config(path, [*overrides, f"model.reader_rotary_base={text}"])
        assert named.model.reader_rotary_base == 10_000.0
        assert "reader_rotary_base" not in json.loads(canonical_json(named))["model"]
        assert config_hash(named) == config_hash(base)
        assert memory_key(named, "none", protocol) == memory_key(base, "none", protocol)
    hashes, keys = {config_hash(base)}, {memory_key(base, "none", protocol)}
    for value in BASES:
        other = load_config(path, [*overrides, f"model.reader_rotary_base={value}"])
        assert json.loads(canonical_json(other))["model"]["reader_rotary_base"] == value
        hashes.add(config_hash(other))
        keys.add(memory_key(other, "none", protocol))
    assert len(hashes) == len(keys) == 1 + len(BASES)


# --------------------------------------------------------------------------------------------------
# model.reader_position = learned
# --------------------------------------------------------------------------------------------------


def test_learned_table_is_drawn_last_and_the_other_parameters_start_as_the_rotary_readers() -> None:
    keys = {"reader": "transformer", "reader_width": 16, "reader_heads": 2}
    rotary = Reader(ModelConfig(**keys), seed=5)  # type: ignore[arg-type]
    torch.manual_seed(0)
    before = torch.rand(3)
    torch.manual_seed(0)
    learned = Reader(ModelConfig(**keys, reader_position="learned", reader_positions=40), seed=5)  # type: ignore[arg-type]
    assert torch.equal(torch.rand(3), before)  # the global random stream is not consumed
    assert learned.pos is not None and learned.pos.shape == (40, 16) and learned.pos.requires_grad
    # normal(0, 0.02), as the baseline's table
    assert 0.015 < float(learned.pos.detach().std()) < 0.025
    assert learned._rot_cos is None and learned._rot_sin is None
    assert "position=learned" in repr(learned)
    ours = dict(learned.named_parameters())
    assert list(ours)[0] == "pos" and set(ours) - {"pos"} == set(dict(rotary.named_parameters()))
    for name, p in rotary.named_parameters():
        assert torch.equal(ours[name], p), name
    # moved or cast, the table goes with the module and no rotary table appears
    assert learned.double().pos.dtype == torch.float64 and learned._rot_cos is None
    with pytest.raises(ValueError, match="exceeds the reader's position table"):
        learned(_bytes(1, 41, 0), torch.zeros(1, 41, dtype=torch.bool))


def test_learned_reader_is_the_transformer_baselines_computation() -> None:
    """With the reader's weights copied into a Transformer baseline of its shape (and that
    model's context embedding at zero: the reader has none) the two give the same logits: the
    key replaces the rotation by the baseline's position table and nothing else differs."""
    width, heads, layers, T = 16, 2, 2, 24
    shape = [
        f"model.reader_width={width}",
        f"model.reader_heads={heads}",
        f"model.reader_layers={layers}",
        f"train.unroll={T}",
        f"model.reader_positions={T}",
    ]
    cfg = load_config(READER_LM, [*shape, *LEARNED])
    seed_everything(0)
    alone = build_model(cfg).double().eval()
    assert isinstance(alone, ReaderLM) and alone.reader.pos is not None
    baseline_cfg = load_config(
        TRANSFORMER,
        [
            f"baseline.transformer.d_model={width}",
            f"baseline.transformer.n_heads={heads}",
            f"baseline.transformer.n_layers={layers}",
            f"baseline.transformer.ffn_hidden={4 * width}",
            f"train.unroll={T}",
        ],
    )
    baseline = TransformerBaseline(baseline_cfg).double().eval()
    reader = alone.reader
    with torch.no_grad():
        g = torch.Generator().manual_seed(3)
        for p in alone.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=torch.float64))
        baseline.tok_emb.weight.copy_(reader.emb.weight)
        baseline.pos_emb.weight.copy_(reader.pos)
        baseline.ctx_emb.weight.zero_()
        baseline.blocks.load_state_dict(reader.blocks.state_dict())
        baseline.norm.load_state_dict(reader.norm.state_dict())
        baseline.head.load_state_dict(alone.head.state_dict())
    assert (
        sum(p.numel() for p in alone.parameters())
        == sum(p.numel() for p in baseline.parameters()) - baseline.ctx_emb.weight.numel()
    )
    tokens = _bytes(3, T, 1)
    context = torch.randint(0, 16, (3, T), generator=torch.Generator().manual_seed(2))
    reset = _none(tokens)
    reset[:, 0] = True
    reset[1, 9] = True
    with torch.no_grad():
        ours = alone(tokens, context, None, reset)[0]
        theirs = baseline(tokens, context, None, reset)[0]
    assert float(ours.abs().max()) > 0.1 and _gap(ours, theirs) <= 1e-12


def test_learned_reader_reads_the_window_position_and_stays_causal() -> None:
    reader = _reader(reader_position="learned", reader_positions=40)
    body = _bytes(1, 10, 4)
    outs = []
    for lead in (0, 3):
        tokens = torch.cat([_bytes(1, lead, 20 + lead), body], dim=1)
        reset = _none(tokens)
        reset[:, lead] = True
        with torch.no_grad():
            outs.append(reader(tokens, reset)[:, lead:])
    assert _gap(outs[0], outs[1]) > 1e-3  # the same document three positions later: another output
    tokens = _bytes(3, 20, 6)
    reset = _none(tokens)
    reset[:2, 8] = True
    with torch.no_grad():
        out = reader(tokens, reset)
        later = tokens.clone()
        later[:, 12:] = (later[:, 12:] + 5) % 256
        assert torch.equal(reader(later, reset)[:, :12], out[:, :12])  # no later byte
        before = tokens.clone()
        before[:, :8] = (before[:, :8] + 9) % 256
        changed = reader(before, reset)
        assert torch.equal(changed[:2, 8:], out[:2, 8:])  # nothing before a document's start
        assert _gap(changed[2, 8:], out[2, 8:]) > 0.0


# --------------------------------------------------------------------------------------------------
# The three models that hold a reader
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("switch", SWITCHES)
def test_parameter_counts_follow_the_closed_forms(switch: tuple[str, ...]) -> None:
    table = 256 * 224 if switch == LEARNED else 0
    off = load_config(NOMEM, ["model.reader_width=224", *switch])
    model = build_model(off)
    assert isinstance(model, HSNTNoMemory)
    report, expected = model.param_report(), memory_off_param_count(off.model)
    assert report["reader"] == expected["reader"] == reader_param_count(off.model)
    assert report["stored"] == expected["total"] == 1_942_528 + table
    alone = build_model(load_config(READER_LM, ["model.reader_width=192", *switch]))
    assert alone.param_report()["reader"] == 49_152 + 2 * (12 * 192 * 192 + 13 * 192) + 384 + (
        256 * 192 if switch == LEARNED else 0
    )
    hybrid_cfg = _cfg(*A5, "model_name=hsn_t", "model.reader=transformer", "model.reader_width=8",
                      "model.reader_heads=2", *switch)  # fmt: skip
    hybrid = HSNT.from_config(hybrid_cfg)
    assert hybrid.param_report()["reader"] == reader_param_count(hybrid_cfg.model)


@pytest.mark.parametrize("switch", SWITCHES)
def test_hsn_t_with_its_memory_runs_with_each_key(switch: tuple[str, ...]) -> None:
    cfg = _cfg(*A5, "model_name=hsn_t", "model.reader=transformer", "model.reader_width=8",
               "model.reader_heads=2", *switch)  # fmt: skip
    seed_everything(0)
    model = HSNT.from_config(cfg).double().train()
    tokens, ctx, reset = _window(cfg)
    logits, _, _ = model(tokens, ctx, _start(model, tokens.shape[0]), reset)
    assert bool(torch.isfinite(logits).all())
    logits.square().mean().backward()
    assert model.reader is not None
    grads = [p.grad for p in model.reader.parameters()]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    if switch == LEARNED:
        assert model.reader.pos is not None and float(model.reader.pos.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("switch", SWITCHES)
def test_memory_off_model_compiles_as_one_graph_with_each_key(switch: tuple[str, ...]) -> None:
    """As the memory-off model's own test: ``aot_eager`` with ``fullgraph``, forward and
    backward equal to eager, the second window of the same shape reusing the graph."""
    from torch._dynamo.utils import counters

    cfg = load_config(NOMEM, [*TINY, *switch])
    seed_everything(0)
    model = build_model(cfg).train()
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    g = torch.Generator().manual_seed(3)
    try:
        graphs = None
        for _ in range(2):
            tokens = torch.randint(0, 256, (3, 24), generator=g)
            ctx = torch.randint(0, cfg.model.n_context, (3, 24), generator=g)
            reset = torch.rand(3, 24, generator=g) < 0.1
            reset[:, 0] = True
            results = []
            for forward in (model, compiled):
                model.zero_grad(set_to_none=True)
                logits, state, info = forward(tokens, ctx, None, reset)
                logits.square().mean().backward()
                grads = {n: p.grad.clone() for n, p in model.named_parameters()}
                results.append((logits.detach(), grads))
                assert state is None and info == {}
            (ref, ref_grads), (got, got_grads) = results
            assert _gap(got, ref) <= 1e-6
            assert all(_gap(got_grads[n], ref_grads[n]) <= 1e-6 for n in ref_grads)
            assert all(float(grad.abs().sum()) > 0.0 for grad in ref_grads.values())
            if graphs is not None:
                assert counters["stats"]["unique_graphs"] == graphs, "the second window recompiled"
            graphs = counters["stats"]["unique_graphs"]
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("switch", SWITCHES)
def test_the_generic_trainer_trains_the_reader_with_each_key(
    switch: tuple[str, ...], tmp_path: Path
) -> None:
    steps = 30
    cfg = load_config(
        READER_LM,
        overrides=[
            *train_overrides(steps),
            "model.reader_width=16",
            "model.reader_heads=2",
            "train.lr=3e-3",
            *switch,
        ],
    )
    seed_everything(0)
    trainer = make_trainer(build_model(cfg), cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    assert type(trainer) is Trainer
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.05
