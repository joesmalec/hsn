"""P1.2 tests: embeddings, E / P / S / D, the unit bank's two passes, counts against P1.1."""

import math
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from hsn.config import ModelConfig, RunConfig, load_config
from hsn.model.embeddings import ByteEmbedding, ContextEmbedding, init_weight_, parse_init
from hsn.model.encoder import (
    ACTIVATIONS,
    MLP2,
    Decoder,
    Encoder,
    Predictor,
    SharedTransform,
    make_activation,
)
from hsn.model.units import UnitBank
from hsn.train.memory_plan import expected_param_count
from torch import Tensor, nn

REPO = Path(__file__).resolve().parents[1]
CONFIGS = {
    "ref_a1": REPO / "configs" / "ref_a1.yaml",
    "tiny": REPO / "configs" / "tiny_gradcheck.yaml",
}

cuda_param = pytest.param("cuda", marks=pytest.mark.cuda)
mps_param = pytest.param("mps", marks=pytest.mark.mps)

#: Appendix C.1 rows that P1.2 implements, with the constructor for each (P1.1 keys).
COMPONENTS: dict[str, Callable[[ModelConfig], nn.Module]] = {
    "byte_emb": ByteEmbedding.from_config,
    "ctx_emb": ContextEmbedding.from_config,
    "encoder": Encoder,
    "predictor": Predictor,
    "shared": SharedTransform,
    "units": UnitBank.from_config,
    "decoder": Decoder,
}

#: Standard deviation of N(0, 1) truncated to [-2, 2], relative to the untruncated 1 (≈ 0.8796).
TRUNC2_STD_FACTOR = math.sqrt(
    1.0 - 4.0 * math.exp(-2.0) / math.sqrt(2.0 * math.pi) / math.erf(math.sqrt(2.0))
)


@pytest.fixture(params=sorted(CONFIGS), ids=sorted(CONFIGS))
def cfg(request: pytest.FixtureRequest) -> RunConfig:
    return load_config(CONFIGS[request.param])


def _available(device_type: str) -> torch.device:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device on this machine")
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no mps device on this machine")
    return torch.device(device_type)


def _measured(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _gather_units(all_out: Tensor, unit_idx: Tensor) -> Tensor:
    """``all_out [B, N, M, d]``, ``unit_idx [B, N]`` -> ``[B, N, d]``: each cell's chosen unit."""
    d = all_out.shape[-1]
    index = unit_idx.view(*unit_idx.shape, 1, 1).expand(*unit_idx.shape, 1, d)
    return all_out.gather(2, index).squeeze(2)


def _randomise_biases(bank: UnitBank, std: float = 0.1) -> None:
    """Non-zero, unit-specific biases so that a wrong bias row would show in the outputs.

    Drawn on the CPU generator and copied in, so every device gets the same values per seed.
    """
    with torch.no_grad():
        bank.b1.copy_(torch.randn(bank.b1.shape) * std)
        bank.b2.copy_(torch.randn(bank.b2.shape) * std)


def _bank_grads(bank: UnitBank, v: Tensor, out: Tensor, weights: Tensor) -> list[Tensor]:
    """Gradients of ``(out * weights).sum()`` w.r.t. ``v`` and the four bank parameters."""
    bank.zero_grad()
    (out * weights).sum().backward()  # type: ignore[no-untyped-call]
    grads = [v.grad] + [p.grad for p in bank.parameters()]
    assert all(g is not None for g in grads)
    return [g.detach().clone() for g in grads if g is not None]


def _run_step_modules(
    cfg: RunConfig, dtype: torch.dtype, device: torch.device
) -> dict[str, Tensor]:
    """Build every P1.2 module from the config and run the §2.2 shapes once; returns the tensors."""
    m = cfg.model
    torch.manual_seed(cfg.seed)
    modules = {name: build(m).to(device=device, dtype=dtype) for name, build in COMPONENTS.items()}
    batch, n, k = cfg.train.batch, m.n_cells, m.k_select
    # Inputs are drawn on the CPU so that every device sees the same values for the same seed.
    tokens = torch.randint(0, m.vocab_size, (batch,)).to(device)
    ctx = torch.randint(0, m.n_context, (batch,)).to(device)
    q_prev = torch.zeros(batch, m.d, dtype=dtype, device=device)

    z_hat = modules["predictor"](q_prev)  # ẑ_t = P(q_{t-1}) before x_t is encoded
    z = modules["encoder"](modules["byte_emb"](tokens))
    ctx_emb = modules["ctx_emb"](ctx)
    u = modules["shared"](torch.cat([z, q_prev, ctx_emb], dim=-1))
    logits = modules["decoder"](torch.cat([q_prev, u], dim=-1))

    bank = modules["units"]
    assert isinstance(bank, UnitBank)
    _randomise_biases(bank)
    v = torch.randn(batch, n, 4 * m.d, dtype=dtype).to(device)
    all_out = bank.forward_all(v)
    unit_idx = torch.randint(0, m.n_units, (batch, k)).to(device)
    selected = bank.forward_selected(v[:, :k].reshape(batch * k, 4 * m.d), unit_idx.reshape(-1))
    return {
        "z_hat": z_hat,
        "z": z,
        "ctx_emb": ctx_emb,
        "u": u,
        "logits": logits,
        "all_out": all_out,
        "selected": selected,
        "selected_expected": _gather_units(all_out[:, :k], unit_idx).reshape(batch * k, m.d),
    }


# --------------------------------------------------------------------------------------------------
# Tests named by the chunk
# --------------------------------------------------------------------------------------------------


def test_p1_2_shapes(cfg: RunConfig) -> None:
    m = cfg.model
    dtype = torch.float64 if cfg.train.precision == "fp64" else torch.float32
    batch, n, mm, d, k = cfg.train.batch, m.n_cells, m.n_units, m.d, m.k_select
    out = _run_step_modules(cfg, dtype, torch.device("cpu"))
    assert out["z_hat"].shape == (batch, d)
    assert out["z"].shape == (batch, d)
    assert out["ctx_emb"].shape == (batch, m.context_dim)
    assert out["u"].shape == (batch, d)
    assert out["logits"].shape == (batch, m.vocab_size)
    assert out["all_out"].shape == (batch, n, mm, d)
    assert out["selected"].shape == (batch * k, d)
    assert all(t.dtype == dtype for t in out.values())
    assert all(torch.isfinite(t).all() for t in out.values())

    # The widths the chunk states: S over [z; q; ctx] is 3d in A.1 (context_dim = d), D over [q; u].
    shared, decoder = SharedTransform(m), Decoder(m)
    assert (shared.in_dim, shared.hidden, shared.out_dim) == (3 * d, m.hidden, d)
    assert (decoder.in_dim, decoder.hidden, decoder.out_dim) == (2 * d, m.hidden, m.vocab_size)
    assert (Encoder(m).in_dim, Encoder(m).out_dim) == (d, d)
    assert (Predictor(m).in_dim, Predictor(m).out_dim) == (d, d)
    bank = UnitBank.from_config(m)
    assert tuple(bank.W1.shape) == (mm, 4 * d, m.hidden)
    assert tuple(bank.b1.shape) == (mm, m.hidden)
    assert tuple(bank.W2.shape) == (mm, m.hidden, d)
    assert tuple(bank.b2.shape) == (mm, d)
    assert bank.unit_ids == [f"unit{j}" for j in range(mm)]
    assert bank.versions == [1] * mm
    assert len(set(bank.unit_ids)) == mm

    # MLP2 applies over any leading shape, and int32 ids are accepted next to int64.
    mlp = MLP2(5, 7, 3, m.activation, init=m.init)
    assert mlp(torch.randn(2, 4, 5)).shape == (2, 4, 3)
    assert mlp(torch.randn(5)).shape == (3,)
    emb = ByteEmbedding.from_config(m)
    assert emb(torch.zeros(2, 3, dtype=torch.int32)).shape == (2, 3, d)


def test_p1_2_forward_selected_equals_forward_all_gathered() -> None:
    cfg = load_config(CONFIGS["ref_a1"])
    m = cfg.model
    torch.manual_seed(cfg.seed)
    bank = UnitBank.from_config(m)
    _randomise_biases(bank)
    batch, n, d = cfg.train.batch, m.n_cells, m.d
    v = torch.randn(batch, n, 4 * d, dtype=torch.float32)

    # Every unit present, in a scrambled order.
    unit_idx = torch.randint(0, m.n_units, (batch, n))
    assert unit_idx.unique().numel() == m.n_units  # type: ignore[no-untyped-call]
    expected = _gather_units(bank.forward_all(v), unit_idx).reshape(batch * n, d)
    actual = bank.forward_selected(v.reshape(batch * n, 4 * d), unit_idx.reshape(-1))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.0)

    # Only some units present (the sort-by-unit path skips the absent ones) and int32 indices.
    sparse_idx = torch.where(unit_idx % 2 == 0, 0, 5)
    expected = _gather_units(bank.forward_all(v), sparse_idx).reshape(batch * n, d)
    actual = bank.forward_selected(
        v.reshape(batch * n, 4 * d), sparse_idx.reshape(-1).to(torch.int32)
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.0)

    # One pair, and no pairs at all.
    one = bank.forward_selected(v[:1, 0], unit_idx[:1, 0])
    one_expected = bank.forward_all(v[:1, :1])[0, 0, unit_idx[0, 0]][None]
    torch.testing.assert_close(one, one_expected, atol=1e-6, rtol=0.0)
    assert bank.forward_selected(v.new_zeros(0, 4 * d), unit_idx.new_zeros(0)).shape == (0, d)

    # Both passes are the same function for autograd too (v and all four parameters).
    v_all = v.clone().requires_grad_(True)
    v_sel = v.clone().requires_grad_(True)
    weights = torch.randn(batch * n, d)
    out_all = _gather_units(bank.forward_all(v_all), unit_idx).reshape(batch * n, d)
    grads_all = _bank_grads(bank, v_all, out_all, weights)
    out_sel = bank.forward_selected(v_sel.reshape(batch * n, 4 * d), unit_idx.reshape(-1))
    grads_sel = _bank_grads(bank, v_sel, out_sel, weights)
    assert len(grads_all) == len(grads_sel) == 5
    for g_all, g_sel in zip(grads_all, grads_sel, strict=True):
        torch.testing.assert_close(g_sel, g_all, atol=1e-5, rtol=0.0)


def test_p1_2_param_counts_match_p1_1(cfg: RunConfig) -> None:
    m = cfg.model
    expected = expected_param_count(m)
    for component, build in COMPONENTS.items():
        module = build(m)
        measured = _measured(module)
        assert measured == expected[component], component
        count = module.param_count
        assert count() == measured, component
    # Appendix C.1 to the digit for the reference configuration, including the unit bank's size.
    if cfg.run_id == "ref_a1":
        assert _measured(UnitBank.from_config(m)) == 1_315_328 == 8 * 164_416
        assert _measured(Decoder(m)) == 197_376
        assert _measured(SharedTransform(m)) == 131_648
        assert sum(_measured(build(m)) for build in COMPONENTS.values()) == 1_793_984


# --------------------------------------------------------------------------------------------------
# Config values, initialisation, validation, autocast, state_dict, devices
# --------------------------------------------------------------------------------------------------


def test_p1_2_config_values_reach_every_module() -> None:
    """A non-default init, activation and context width from the config land in every module."""
    cfg = load_config(
        CONFIGS["tiny"],
        overrides=["model.init=trunc_normal_0.5", "model.activation=tanh", "model.context_dim=3"],
    )
    m = cfg.model
    assert (m.init, m.activation, m.context_dim, m.d) == ("trunc_normal_0.5", "tanh", 3, 4)
    torch.manual_seed(cfg.seed)
    expected = expected_param_count(m)
    for component, build in COMPONENTS.items():
        module = build(m)
        assert _measured(module) == expected[component], component
        for name, p in module.named_parameters():
            if name.endswith("bias") or name in ("b1", "b2"):
                continue
            assert 0.04 < p.abs().max() <= 1.0, (component, name)  # ±2σ at σ = 0.5, not 0.02
        if component not in ("byte_emb", "ctx_emb"):
            assert isinstance(module.act, nn.Tanh), component
    # The deviation from the chunk's literal 3d: S is 2d + context_dim wide, and ctx is context_dim.
    batch = cfg.train.batch
    ctx_emb = ContextEmbedding.from_config(m)(torch.randint(0, m.n_context, (batch,)))
    assert ctx_emb.shape == (batch, 3)
    shared = SharedTransform(m)
    assert shared.in_dim == 2 * m.d + m.context_dim == 11
    z = torch.randn(batch, m.d)
    assert shared(torch.cat([z, torch.zeros(batch, m.d), ctx_emb], dim=-1)).shape == (batch, m.d)


def test_p1_2_init_follows_config() -> None:
    cfg = load_config(CONFIGS["ref_a1"])
    m = cfg.model
    assert (m.activation, m.init) == ("gelu", "trunc_normal_0.02")
    std = parse_init(m.init)
    assert std == 0.02
    torch.manual_seed(cfg.seed)
    for component, build in COMPONENTS.items():
        module = build(m)
        for name, p in module.named_parameters():
            if name.endswith("bias") or name in ("b1", "b2"):
                assert torch.count_nonzero(p) == 0, (component, name)
            else:
                assert p.abs().max() <= 2.0 * std, (component, name)
                assert p.abs().max() > 0.0, (component, name)
    # The bank's W1 has a million entries: its sample std is the truncated-normal std, not 0.02.
    w1 = UnitBank.from_config(m).W1
    assert w1.numel() == 1_048_576
    assert abs(w1.std().item() / (std * TRUNC2_STD_FACTOR) - 1.0) < 0.01
    assert abs(w1.mean().item()) < 1e-3
    # Every activation name builds, and the GELU is the exact (erf) form.
    assert sorted(ACTIVATIONS) == ["gelu", "relu", "silu", "tanh"]
    for name in ACTIVATIONS:
        assert isinstance(make_activation(name), nn.Module)
    gelu = make_activation("gelu")
    assert isinstance(gelu, nn.GELU) and gelu.approximate == "none"
    assert isinstance(Encoder(m).act, nn.GELU)
    assert isinstance(UnitBank.from_config(m).act, nn.GELU)
    # Other std values parse; init_weight_ honours them.
    assert parse_init("trunc_normal_0.1") == 0.1 and parse_init("trunc_normal_1e-3") == 1e-3
    w = init_weight_(torch.empty(1000, 100), "trunc_normal_0.5")
    assert w.abs().max() <= 1.0 and w.std() > 0.3


def test_p1_2_rejects_bad_inputs() -> None:
    cfg = load_config(CONFIGS["tiny"])
    m = cfg.model
    with pytest.raises(ValueError, match="model.activation"):
        MLP2(m.d, m.hidden, m.d, "swish", init=m.init)
    with pytest.raises(ValueError, match="model.activation"):
        UnitBank(m.n_units, m.d, m.hidden, "GELU", init=m.init)
    with pytest.raises(TypeError):  # init is required: no code default for model.init
        MLP2(m.d, m.hidden, m.d, m.activation)  # type: ignore[call-arg]
    bad_inits = (
        "normal_0.02",
        "trunc_normal",
        "trunc_normal_0",
        "trunc_normal_-0.02",
        "trunc_normal_1e400",
        "xavier",
    )
    for bad_init in bad_inits:
        with pytest.raises(ValueError, match="model.init"):
            parse_init(bad_init)
        with pytest.raises(ValueError, match="model.init"):
            MLP2(m.d, m.hidden, m.d, m.activation, init=bad_init)
    with pytest.raises(ValueError, match="model.init"):
        ByteEmbedding(m.vocab_size, m.d, init="normal_0.02")
    with pytest.raises(ValueError):
        MLP2(0, m.hidden, m.d, m.activation, init=m.init)
    with pytest.raises(ValueError):
        ByteEmbedding(m.vocab_size, 0, init=m.init)

    enc = Encoder(m)
    with pytest.raises(ValueError, match=r"\[\.\.\., 4\]"):
        enc(torch.randn(2, m.d + 1))
    with pytest.raises(TypeError, match="tokens"):
        ByteEmbedding.from_config(m)(torch.zeros(2, dtype=torch.float32))
    with pytest.raises(TypeError, match="ctx"):
        ContextEmbedding.from_config(m)(torch.zeros(2, dtype=torch.float32))
    with pytest.raises(IndexError):
        ByteEmbedding.from_config(m)(torch.tensor([m.vocab_size]))

    bank = UnitBank.from_config(m)
    v = torch.randn(cfg.train.batch, m.n_cells, 4 * m.d)
    with pytest.raises(ValueError, match="forward_all"):
        bank.forward_all(v[0])
    with pytest.raises(ValueError, match="forward_all"):
        bank.forward_all(v[..., :-1])
    flat = v.reshape(-1, 4 * m.d)
    good = torch.zeros(flat.shape[0], dtype=torch.int64)
    with pytest.raises(ValueError, match="forward_selected"):
        bank.forward_selected(v, good)
    with pytest.raises(ValueError, match="forward_selected"):
        bank.forward_selected(flat[:, :-1], good)
    with pytest.raises(ValueError, match="unit_idx"):
        bank.forward_selected(flat, good[:-1])
    with pytest.raises(ValueError, match="unit_idx"):
        bank.forward_selected(flat, good.view(-1, 1))
    with pytest.raises(TypeError, match="unit_idx"):
        bank.forward_selected(flat, good.float())
    with pytest.raises(IndexError, match=r"\[0, 2\)"):
        bank.forward_selected(flat, good + m.n_units)
    with pytest.raises(IndexError, match=r"\[0, 2\)"):
        bank.forward_selected(flat, good - 1)
    for bad_ids in (["a"], ["a", "a"], "ab", ["a", 1]):
        with pytest.raises(ValueError, match="unit_ids"):
            UnitBank(2, m.d, m.hidden, m.activation, init=m.init, unit_ids=bad_ids)  # type: ignore[arg-type]
    for bad_versions in ([1, 0], [1], [1.5, 1], [True, 1]):
        with pytest.raises(ValueError, match="versions"):
            UnitBank(2, m.d, m.hidden, m.activation, init=m.init, versions=bad_versions)  # type: ignore[arg-type]
    named = UnitBank(
        2, m.d, m.hidden, m.activation, init=m.init, unit_ids=("php", "sql"), versions=(3, 1)
    )
    assert named.unit_ids == ["php", "sql"] and named.versions == [3, 1]


def test_p1_2_passes_agree_under_autocast() -> None:
    """Under autocast both passes return the autocast dtype and the same values (CPU bf16)."""
    cfg = load_config(CONFIGS["ref_a1"])
    m = cfg.model
    torch.manual_seed(cfg.seed)
    bank = UnitBank.from_config(m)
    _randomise_biases(bank)
    batch, n, d = cfg.train.batch, m.n_cells, m.d
    v = torch.randn(batch, n, 4 * d)
    unit_idx = torch.randint(0, m.n_units, (batch, n))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        all_out = bank.forward_all(v)
        selected = bank.forward_selected(v.reshape(batch * n, 4 * d), unit_idx.reshape(-1))
    assert all_out.dtype == selected.dtype == torch.bfloat16
    expected = _gather_units(all_out, unit_idx).reshape(batch * n, d)
    torch.testing.assert_close(selected, expected, atol=2e-2, rtol=2e-2)
    reference = _gather_units(bank.forward_all(v), unit_idx).reshape(batch * n, d)
    torch.testing.assert_close(selected.float(), reference, atol=5e-2, rtol=5e-2)


def test_p1_2_state_dict_carries_unit_metadata() -> None:
    cfg = load_config(CONFIGS["tiny"])
    m = cfg.model
    source = UnitBank(m.n_units, m.d, m.hidden, m.activation, init=m.init, versions=[4, 2])
    state = source.state_dict()
    assert set(state) == {"W1", "b1", "W2", "b2", "_extra_state"}
    assert state["_extra_state"] == {"unit_ids": ["unit0", "unit1"], "versions": [4, 2]}

    target = UnitBank.from_config(m)
    assert target.versions == [1, 1]
    target.load_state_dict(state)
    assert target.versions == [4, 2]
    torch.testing.assert_close(target.W1, source.W1)

    other = UnitBank(m.n_units, m.d, m.hidden, m.activation, init=m.init, unit_ids=["php", "sql"])
    with pytest.raises(ValueError, match="unit_ids"):
        other.load_state_dict(state)
    with pytest.raises(ValueError, match="versions"):
        target.set_extra_state({"unit_ids": ["unit0", "unit1"], "versions": [0, 1]})
    with pytest.raises(ValueError, match="extra state"):
        target.set_extra_state({"versions": [1, 1]})


@pytest.mark.parametrize("device_type", [cuda_param, mps_param])
def test_p1_2_runs_on_device(device_type: str) -> None:
    device = _available(device_type)
    cfg = load_config(CONFIGS["ref_a1"])
    m = cfg.model
    on_device = _run_step_modules(cfg, torch.float32, device)
    on_cpu = _run_step_modules(cfg, torch.float32, torch.device("cpu"))
    for name, t in on_device.items():
        assert t.device.type == device.type, name
        assert torch.isfinite(t).all(), name
    # Grouped pass equals the dense reference on the device too.
    torch.testing.assert_close(
        on_device["selected"], on_device["selected_expected"], atol=1e-6, rtol=0.0
    )
    # Same seed, same weights: the device agrees with the CPU oracle to fp32 accumulation error.
    for name in ("z_hat", "z", "u", "logits", "all_out", "selected"):
        torch.testing.assert_close(on_device[name].cpu(), on_cpu[name], atol=1e-4, rtol=1e-4)
    assert on_device["all_out"].shape == (cfg.train.batch, m.n_cells, m.n_units, m.d)

    # Backward on the device: the index / select kernels differ from the CPU ones. All units
    # present, int32 indices, gradients of both passes agree.
    torch.manual_seed(cfg.seed)
    bank = UnitBank.from_config(m)
    _randomise_biases(bank)
    bank = bank.to(device)
    batch, n, d = cfg.train.batch, m.n_cells, m.d
    v = torch.randn(batch, n, 4 * d).to(device)
    unit_idx = torch.randint(0, m.n_units, (batch, n)).to(device)
    weights = torch.randn(batch * n, d).to(device)
    v_all = v.clone().requires_grad_(True)
    v_sel = v.clone().requires_grad_(True)
    out_all = _gather_units(bank.forward_all(v_all), unit_idx).reshape(batch * n, d)
    grads_all = _bank_grads(bank, v_all, out_all, weights)
    out_sel = bank.forward_selected(
        v_sel.reshape(batch * n, 4 * d), unit_idx.reshape(-1).to(torch.int32)
    )
    grads_sel = _bank_grads(bank, v_sel, out_sel, weights)
    for g_all, g_sel in zip(grads_all, grads_sel, strict=True):
        assert g_sel.device.type == device.type
        torch.testing.assert_close(g_sel, g_all, atol=1e-5, rtol=0.0)
