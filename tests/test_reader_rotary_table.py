"""The reader's rotary table (``hsn.model.reader``, 15 Nov 2026).

The reader used to compute the float64 rotary angles and their cos / sin in every forward pass.
Under ``torch.compile`` that computation was inlined into the kernels that rotate the queries
and keys, as float64 trigonometry per element. The cos / sin are now read from a table made
when the reader is built and made again on its device whenever the module is moved or cast.

These tests check that nothing a run can see changed: the outputs are the old forward's bit for
bit (:func:`_reference_forward` is the old code, kept here), the table is what that code
computed, the state dict has the old keys and loads strictly, the forward pass touches no
float64 tensor while the window fits the table, and the compiled reader holds no float64 and
equals the eager one. The CUDA and MPS tests need their device and were not run when this file
was written (15 Nov 2026: the card was held by training runs, and this machine has no MPS).
"""

from __future__ import annotations

import copy
import os
import shutil
import sys
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from hsn.config import ModelConfig
from hsn.model import reader as reader_mod
from hsn.model.reader import Reader
from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode

L = reader_mod._ROTARY_TABLE
#: inductor's CPU backend needs a C++ compiler
INDUCTOR = sys.platform.startswith("linux") and any(
    shutil.which(name) for name in (os.environ.get("CXX", "g++"), "g++", "clang++")
)
#: Compiled against eager in fp32 on the CPU (inductor), 10x the largest difference measured,
#: rounded up (the efficiency audit's rule). 15 Nov 2026, the two cases of the test, three
#: windows each: values 2.5e-6 (outputs up to 4.6), parameter gradients 3.0e-7 (up to 0.2).
TOL_VALUES, TOL_GRADS = 3e-5, 3e-6

#: (width, heads, layers, span): from one rotated pair per head (width 8, four heads) to the
#: width of the timing runs (224)
SHAPES = [(8, 4, 1, None), (16, 2, 2, None), (16, 2, 2, 4), (64, 4, 2, 32), (224, 4, 2, None)]


@pytest.fixture(autouse=True)
def _release_compiled_state() -> Iterator[None]:
    yield
    torch._dynamo.reset()


def _reader(
    width: int = 16, heads: int = 2, layers: int = 2, span: int | None = None, seed: int = 0
) -> Reader:
    cfg = ModelConfig(
        reader="transformer",
        reader_width=width,
        reader_heads=heads,
        reader_layers=layers,
        reader_span=span,
    )
    return Reader(cfg, seed=seed).eval()


def _reference_cos_sin(T: int, half: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """cos / sin ``[T, half]`` as the forward pass computed them before the table (6bbf53d),
    before its cast to the working dtype."""
    wide = torch.float32 if device.type == "mps" else torch.float64
    exponent = torch.arange(half, device=device, dtype=wide) / half
    angles = torch.arange(T, device=device, dtype=wide)[:, None] * (10_000.0**-exponent)
    return angles.cos(), angles.sin()


def _reference_forward(reader: Reader, tokens: Tensor, reset: Tensor) -> Tensor:
    """``Reader.forward`` as it was before the table (6bbf53d): the angles computed in the
    call."""
    T = int(tokens.shape[1])
    cos, sin = _reference_cos_sin(T, reader.width // reader.heads // 2, tokens.device)
    x = reader.emb(tokens)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    mask = reader.attention_mask(reset)
    for block in reader.blocks:
        x = block(x, mask, cos, sin)
    out: Tensor = reader.norm(x)
    return out


def _window(B: int, T: int, seed: int, resets: float = 0.0) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 256, (B, T), generator=g)
    reset = torch.rand(B, T, generator=g) < resets
    return tokens, reset


def _table_is_the_forwards(reader: Reader) -> None:
    """The table holds what the old forward computed on the reader's device, in its dtype."""
    weight = reader.emb.weight
    cos, sin = _reference_cos_sin(L, reader.width // reader.heads // 2, weight.device)
    assert reader._rot_cos is not None and reader._rot_sin is not None
    for table, wide in ((reader._rot_cos, cos), (reader._rot_sin, sin)):
        assert table.dtype == weight.dtype and table.device == weight.device
        assert not table.requires_grad and table.shape == wide.shape
        assert torch.equal(table, wide.to(weight.dtype))


def _forward_is_the_old_forward(reader: Reader, lengths: tuple[int, ...]) -> None:
    device = reader.emb.weight.device
    for seed, T in enumerate(lengths):
        for resets in (0.0, 0.1):
            tokens, reset = (v.to(device) for v in _window(2, T, seed, resets))
            with torch.no_grad():
                got, want = reader(tokens, reset), _reference_forward(reader, tokens, reset)
            assert got.dtype == want.dtype and torch.equal(got, want), (T, resets)


# --------------------------------------------------------------------------------------------------
# Eager: the same bits
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("width", "heads", "layers", "span"), SHAPES)
def test_forward_is_the_old_forward_bit_for_bit(
    width: int, heads: int, layers: int, span: int | None, dtype: torch.dtype
) -> None:
    reader = _reader(width, heads, layers, span).to(dtype)
    _table_is_the_forwards(reader)
    _forward_is_the_old_forward(reader, (1, 2, 7, 12, 64, 255, 256))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_window_longer_than_the_table_computes_its_angles(dtype: torch.dtype) -> None:
    """Any window length works: up to the table's it is read, past it the angles are computed
    in the call, as before."""
    reader = _reader(8, 2, 1, 16).to(dtype)
    _forward_is_the_old_forward(reader, (L - 1, L, L + 1, L + 300))


def test_training_mode_and_gradients_are_the_old_ones() -> None:
    reader = _reader(16, 2, 2).train()
    tokens, reset = _window(3, 40, 5, 0.05)
    grads = []
    for forward in (reader, lambda t, r: _reference_forward(reader, t, r)):
        reader.zero_grad()
        forward(tokens, reset).square().mean().backward()
        grads.append([p.grad.clone() for p in reader.parameters() if p.grad is not None])
    assert len(grads[0]) == len(grads[1]) == len(list(reader.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(*grads, strict=True))


def test_bf16_autocast_is_the_old_forward() -> None:
    reader = _reader(64, 4, 2)
    tokens, reset = _window(2, 48, 3, 0.05)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        got, want = reader(tokens, reset), _reference_forward(reader, tokens, reset)
    assert got.dtype == want.dtype and torch.equal(got, want)


def test_table_follows_the_module_through_casts_and_copies() -> None:
    """A cast makes the table again from the float64 angles: a float32 table cast to float64
    would not be them. A parent's cast reaches the reader, and a copy carries its table."""
    reader = _reader(16, 2, 1)
    _table_is_the_forwards(reader)
    assert reader._rot_cos is not None and reader._rot_cos.dtype == torch.float32
    wide = copy.deepcopy(reader).double()
    _table_is_the_forwards(wide)
    assert wide._rot_cos is not None
    assert not torch.equal(wide._rot_cos, reader._rot_cos.double())  # float32 lost the bits
    _table_is_the_forwards(copy.deepcopy(wide).float())
    _table_is_the_forwards(copy.deepcopy(reader).to(torch.bfloat16))
    parent = nn.ModuleList([copy.deepcopy(reader)]).double()
    child = parent[0]
    assert isinstance(child, Reader)
    _table_is_the_forwards(child)
    _forward_is_the_old_forward(child, (9,))
    _table_is_the_forwards(copy.deepcopy(wide))
    # made again, the table is still outside the state dict
    assert list(wide.state_dict()) == [name for name, _ in wide.named_parameters()]


def test_parameters_cast_without_the_module_compute_their_angles() -> None:
    """The table is read only in the dtype it was made for: weights cast behind ``_apply``
    (``load_state_dict(assign=True)``) get the float64 angles from the call, not a float32
    table."""
    reader = _reader(16, 2, 1)
    wide = {name: value.double() for name, value in reader.state_dict().items()}
    reader.load_state_dict(wide, strict=True, assign=True)
    assert reader.emb.weight.dtype == torch.float64
    assert reader._rot_cos is not None and reader._rot_cos.dtype == torch.float32
    _forward_is_the_old_forward(reader, (9, 33))


def test_without_a_table_the_forward_computes_its_angles() -> None:
    """The path of a device without float64 (MPS leaves the table empty)."""
    reader = _reader(16, 2, 1)
    reader._rot_cos = reader._rot_sin = None
    assert [name for name, _ in reader.named_buffers()] == []
    _forward_is_the_old_forward(reader, (1, 9, 33))


# --------------------------------------------------------------------------------------------------
# The state dict is the old one
# --------------------------------------------------------------------------------------------------

#: the keys of a one-block reader's state dict at 6bbf53d
OLD_KEYS = [
    "emb.weight",
    *(
        f"blocks.0.{module}.{leaf}"
        for module in ("norm_attn", "qkv", "proj", "norm_mlp", "fc_in", "fc_out")
        for leaf in ("weight", "bias")
    ),
    "norm.weight",
    "norm.bias",
]


def test_state_dict_keys_are_unchanged_and_an_old_one_loads_strictly() -> None:
    source = _reader(16, 2, 1, seed=3)
    state = source.state_dict()
    assert list(state) == OLD_KEYS == [name for name, _ in source.named_parameters()]
    assert sorted(name for name, _ in source.named_buffers()) == ["_rot_cos", "_rot_sin"]
    # an old checkpoint: the parameters under their names and nothing else
    old = {name: value.detach().clone() for name, value in source.named_parameters()}
    target = _reader(16, 2, 1, seed=0)
    result = target.load_state_dict(old, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    tokens, reset = _window(2, 20, 1, 0.1)
    with torch.no_grad():
        assert torch.equal(target(tokens, reset), source(tokens, reset))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        target.load_state_dict({**old, "_rot_cos": torch.zeros(L, 4)}, strict=True)


# --------------------------------------------------------------------------------------------------
# No float64 in the forward pass
# --------------------------------------------------------------------------------------------------


class _Float64Ops(TorchDispatchMode):
    """Records the operations that read or return a float64 tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def __torch_dispatch__(
        self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None
    ) -> Any:
        out = func(*args, **(kwargs or {}))
        flat: list[Any] = [*args, *(kwargs or {}).values()]
        flat += list(out) if isinstance(out, tuple | list) else [out]
        if any(isinstance(v, Tensor) and v.dtype == torch.float64 for v in flat):
            self.seen.append(str(func))
        return out


def _float64_ops(forward: Callable[[], Tensor]) -> list[str]:
    with torch.no_grad(), _Float64Ops() as mode:
        forward()
    return mode.seen


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("T", [1, 7, 256, L])
def test_forward_inside_the_table_touches_no_float64(T: int, autocast: bool) -> None:
    """The regression guard: with a window the table holds, no operation of the forward pass
    reads or makes a float64 tensor, so a compiled graph has no float64 arithmetic to inline."""
    reader = _reader(8, 2, 2)
    tokens, reset = _window(2, T, 0, 0.05)

    def run() -> Tensor:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            out: Tensor = reader(tokens, reset)
        return out

    assert _float64_ops(run) == []


def test_float64_guard_sees_the_old_forward_and_the_long_window() -> None:
    """The guard is not blind: it reports the old forward, and the new one past the table."""
    reader = _reader(8, 2, 1)
    tokens, reset = _window(2, 16, 0)
    seen = _float64_ops(lambda: _reference_forward(reader, tokens, reset))
    assert any("cos" in name for name in seen) and any("sin" in name for name in seen)
    long_tokens, long_reset = _window(1, L + 1, 0)
    assert any("cos" in name for name in _float64_ops(lambda: reader(long_tokens, long_reset)))


# --------------------------------------------------------------------------------------------------
# Compiled
# --------------------------------------------------------------------------------------------------


def _traced_float64(forward: Callable[[Tensor, Tensor], Tensor], T: int) -> tuple[int, list[str]]:
    """Trace ``forward`` with dynamo: the number of graphs and the nodes whose value is a
    float64 tensor. The backend runs the traced graph as it is, so no compiler is needed."""
    graphs: list[torch.fx.GraphModule] = []

    def backend(gm: torch.fx.GraphModule, example_inputs: list[Tensor]) -> Any:
        graphs.append(gm)
        return gm.forward

    compiled = torch.compile(forward, backend=backend)
    tokens, reset = _window(2, T, 0, 0.1)
    with torch.no_grad():
        got, want = compiled(tokens, reset), forward(tokens, reset)
        assert torch.equal(got, want)
        with torch.compiler.set_stance("fail_on_recompile"):
            compiled(*_window(2, T, 1, 0.1))  # another window of the shape: the same graph
    wide = [
        f"{node.op} {node.target}"
        for gm in graphs
        for node in gm.graph.nodes
        if isinstance(value := node.meta.get("example_value"), Tensor)
        and value.dtype == torch.float64
    ]
    assert any(
        isinstance(node.meta.get("example_value"), Tensor)
        for gm in graphs
        for node in gm.graph.nodes
    )
    return len(graphs), wide


def test_traced_graph_holds_no_float64() -> None:
    """What ``torch.compile`` is handed: one graph, and no node of it is a float64 tensor (the
    old forward's graph holds the float64 angles and their cos / sin)."""
    reader = _reader(16, 2, 2)
    assert _traced_float64(reader, 24) == (1, [])
    graphs, wide = _traced_float64(lambda t, r: _reference_forward(reader, t, r), 24)
    assert graphs == 1 and any("cos" in node for node in wide)
    torch._dynamo.reset()
    graphs, wide = _traced_float64(reader, L + 1)  # past the table: the old computation
    assert graphs == 1 and any("cos" in node for node in wide)


@pytest.mark.skipif(not INDUCTOR, reason="inductor needs Linux and a C++ compiler")
@pytest.mark.parametrize(("width", "heads", "T"), [(16, 2, 24), (224, 4, 256)])
def test_compiled_reader_equals_eager(width: int, heads: int, T: int) -> None:
    """``torch.compile`` (inductor, the default mode) of the reader with its table: values and
    parameter gradients of a training window equal the eager ones to rounding."""
    reader = _reader(width, heads, 2).train()
    compiled = torch.compile(reader)
    for seed in range(3):
        tokens, reset = _window(2, T, seed, 0.05)
        results = []
        for forward in (reader, compiled):
            reader.zero_grad()
            out = forward(tokens, reset)
            out.square().mean().backward()
            grads = [p.grad.clone() for p in reader.parameters() if p.grad is not None]
            results.append((out.detach(), grads))
        (want, want_grads), (got, got_grads) = results
        assert float((got - want).abs().max()) <= TOL_VALUES, seed
        assert len(got_grads) == len(want_grads) == len(list(reader.parameters()))
        worst = max(float((a - b).abs().max()) for a, b in zip(got_grads, want_grads, strict=True))
        assert worst <= TOL_GRADS, seed


# --------------------------------------------------------------------------------------------------
# CUDA and MPS (skipped without the device; not run when this file was written)
# --------------------------------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("width", "heads"), [(16, 2), (224, 4)])
def test_table_on_cuda_is_the_cards_own_computation(
    width: int, heads: int, dtype: torch.dtype
) -> None:
    """The table of a reader moved to the card is computed on the card: bit for bit what the
    old forward computed there, so the outputs are the old ones (also under bf16 autocast and
    past the table). A table made on the CPU and moved would not have to be."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    reader = _reader(width, heads, 2).to("cuda", dtype)
    assert reader._rot_cos is not None and reader._rot_cos.device.type == "cuda"
    _table_is_the_forwards(reader)
    _forward_is_the_old_forward(reader, (1, 7, 64, 256, L, L + 1))
    if dtype == torch.float32:
        tokens, reset = (v.to("cuda") for v in _window(4, 256, 0, 0.05))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            got, want = reader(tokens, reset), _reference_forward(reader, tokens, reset)
        assert torch.equal(got, want)
    back = reader.to("cpu")
    assert back._rot_cos is not None and back._rot_cos.device.type == "cpu"
    _table_is_the_forwards(back)


@pytest.mark.mps
def test_mps_keeps_its_float32_angles_and_has_no_table() -> None:
    """MPS has no float64: its angles were float32 and computed in the call, and still are."""
    if not torch.backends.mps.is_available():
        pytest.skip("no mps device on this machine")
    reader = _reader(16, 2, 2).to("mps")
    assert reader._rot_cos is None and reader._rot_sin is None
    _forward_is_the_old_forward(reader, (1, 7, 64, 256))
    _table_is_the_forwards(reader.to("cpu"))
