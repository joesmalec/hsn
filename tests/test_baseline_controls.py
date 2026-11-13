"""The baseline controls: the Mamba-style selective state-space model (``model_name:
selective_ssm``) and the HSN's fixed byte codes as a baseline input (``baseline.address_input``)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from hsn.baselines.address import AddressInput
from hsn.baselines.gru import GRUBaseline
from hsn.baselines.mamba import MambaBaseline, _SelectiveScan
from hsn.baselines.transformer import TransformerBaseline
from hsn.config import RunConfig, config_hash, load_config, load_hardware_profile
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import BOS, TaskSpec, generate, load_dataset_spec
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import FAMILIES
from hsn.model.hsn import HSN
from hsn.runner import build_model
from hsn.train.loop import Trainer
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
MAMBA = REPO / "configs" / "baselines" / "mamba_2m.yaml"
SMALL = [
    "baseline.selective_ssm.d_model=32",
    "baseline.selective_ssm.n_layers=2",
    "baseline.selective_ssm.d_state=4",
    "train.unroll=32",
    "train.batch=8",
    "train.micro_batch=8",
]


def _mamba(*overrides: str, seed: int = 0) -> tuple[MambaBaseline, RunConfig]:
    cfg = load_config(MAMBA, overrides=[*SMALL, *overrides])
    seed_everything(seed)
    model = build_model(cfg)
    assert isinstance(model, MambaBaseline)
    return model.eval(), cfg


def _window(B: int, T: int, seed: int = 0) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(32, 256, (B, T), generator=g), torch.randint(0, 16, (B, T), generator=g)


def test_default_size_and_registration() -> None:
    cfg = load_config(MAMBA)
    assert cfg.model_name == "selective_ssm" and "selective_ssm" in FAMILIES
    report = build_model(cfg).param_report()
    assert report["stored"] == report["active"] == report["trainable"] == 1_887_744


def test_new_keys_leave_existing_hashes_alone() -> None:
    """The controls are opt-in: a configuration that does not set them hashes as before."""
    assert config_hash(load_config(REPO / "configs/baselines/gru_2m.yaml")).startswith(
        "c8dcbaa48e3b27ec"
    )
    assert config_hash(load_config(REPO / "configs/experiments/hsn_recipe_a6.yaml")).startswith(
        "d8b36a25971c9694"
    )
    with pytest.raises(ValueError, match="address_input"):
        load_config(
            REPO / "configs/experiments/hsn_recipe_a6.yaml",
            overrides=["baseline.address_input=true"],
        )


def test_scan_backward_matches_numerical_gradients() -> None:
    """The scan's hand-written backward against finite differences, in float64, with a non-zero
    incoming state and document starts at the first, a middle and the last position."""
    g = torch.Generator().manual_seed(11)
    B, T, C, S = 2, 6, 3, 2

    def rand(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=g, dtype=torch.float64)

    delta = (rand(B, T, C).abs() * 0.5 + 0.05).requires_grad_()
    A = (-rand(C, S).abs() - 0.1).requires_grad_()
    drive, b_in, c_in = rand(B, T, C), rand(B, T, S), rand(B, T, S)
    h0 = rand(B, C, S)
    for tensor in (drive, b_in, c_in, h0):
        tensor.requires_grad_()
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[0, 0] = True
    reset[1, 3] = True
    reset[0, T - 1] = True
    cols = tuple(reset.any(dim=0).nonzero().flatten().tolist())

    def scan(*inputs: Tensor) -> tuple[Tensor, Tensor]:
        return _SelectiveScan.apply(*inputs, reset, cols)  # type: ignore[no-any-return]

    assert torch.autograd.gradcheck(scan, (delta, A, drive, b_in, c_in, h0), eps=1e-6, atol=1e-6)
    # and the values against the recurrence written out
    y, last = scan(delta, A, drive, b_in, c_in, h0)
    h = h0
    for t in range(T):
        h = h * (~reset[:, t]).view(B, 1, 1)
        h = torch.exp(delta[:, t, :, None] * A) * h + drive[:, t, :, None] * b_in[:, t, None, :]
        assert torch.allclose(y[:, t], (h * c_in[:, t, None, :]).sum(-1), atol=1e-12)
    assert torch.allclose(last, h, atol=1e-12)


def test_block_matches_a_direct_reference() -> None:
    """One block from a zero state equals the published recurrence written out directly, with
    ``conv1d`` for the convolution."""
    model, _ = _mamba()
    block = model.blocks[0]
    B, T = 3, 20
    x = torch.randn(B, T, model.d_model, generator=torch.Generator().manual_seed(3))
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    h0, conv0 = model.init_state(B, "cpu")
    last = torch.zeros(B, T, dtype=torch.int64)
    with torch.no_grad():
        out, h, _ = block(x, h0[0], conv0[0], reset, [0], last)
        stream, z = block.in_proj(block.norm(x)).chunk(2, dim=-1)
        K = block.d_conv
        weight = block.conv_weight.flip(1).unsqueeze(1)  # conv1d: the last tap is the current one
        u = F.conv1d(
            stream.transpose(1, 2), weight, block.conv_bias, padding=K - 1, groups=block.d_inner
        )[..., :T].transpose(1, 2)
        u = F.silu(u)
        delta_in, b_in, c_in = block.x_proj(u).split([block.dt_rank, 4, 4], dim=-1)
        delta = F.softplus(block.dt_proj(delta_in))
        A = -torch.exp(block.A_log)
        state = torch.zeros(B, block.d_inner, 4)
        ys = []
        for t in range(T):
            state = torch.exp(delta[:, t, :, None] * A) * state + (
                delta[:, t, :, None] * b_in[:, t, None, :] * u[:, t, :, None]
            )
            ys.append((state * c_in[:, t, None, :]).sum(-1) + block.D * u[:, t])
        ref = x + block.out_proj(torch.stack(ys, dim=1) * F.silu(z))
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()
    assert torch.allclose(h, state, atol=1e-5)


def test_windows_and_resets_equal_each_document_alone() -> None:
    """Three windows with carried state and document starts anywhere (a window's first, middle
    and last position) give every document the logits it gets alone from a fresh state."""
    model, _ = _mamba()
    B, T, W = 4, 16, 3
    tokens, context = _window(B, T * W, seed=5)
    reset = torch.zeros(B, T * W, dtype=torch.bool)
    reset[:, 0] = True
    reset[0, 7] = True  # inside the first window
    reset[1, T] = True  # the first position of the second window
    reset[1, T + 1] = True  # one-position document
    reset[2, 2 * T - 1] = True  # the last position of a window
    reset[2, 2 * T + 2] = True  # within the convolution's reach of a window start
    state = model.init_state(B, "cpu")
    # an incoming state that is not zero shows reset[:, 0] clears both parts of it
    state = (torch.randn_like(state[0]), torch.randn_like(state[1]))
    chunks = []
    with torch.no_grad():
        for w in range(W):
            cols = slice(w * T, (w + 1) * T)
            logits, state, info = model(tokens[:, cols], context[:, cols], state, reset[:, cols])
            chunks.append(logits)
        windowed = torch.cat(chunks, dim=1)
        for b in range(B):
            starts = [*reset[b].nonzero().flatten().tolist(), T * W]
            for lo, hi in zip(starts[:-1], starts[1:], strict=True):
                first = torch.zeros(1, hi - lo, dtype=torch.bool)
                first[0, 0] = True
                alone, _, _ = model(
                    tokens[b : b + 1, lo:hi],
                    context[b : b + 1, lo:hi],
                    model.init_state(1, "cpu"),
                    first,
                )
                diff = (windowed[b, lo:hi] - alone[0]).abs().max()
                assert diff < 1e-4, (b, lo, hi, float(diff))
    assert "state_abs_max" in info
    mask = torch.tensor([True, False, True, False])
    cleared = model.reset_state(state, mask)
    for part, before in zip(cleared, state, strict=True):
        assert not part[:, mask].any() and torch.equal(part[:, ~mask], before[:, ~mask])
    with pytest.raises(ValueError, match="state"):
        model(tokens[:, :T], context[:, :T], state[0], reset[:, :T])


def test_mamba_is_causal() -> None:
    model, _ = _mamba()
    tokens, context = _window(2, 24, seed=7)
    reset = torch.zeros(2, 24, dtype=torch.bool)
    reset[:, 0] = True
    changed = tokens.clone()
    changed[:, 15] = (changed[:, 15] + 1 - 32) % 224 + 32
    with torch.no_grad():
        a, _, _ = model(tokens, context, model.init_state(2, "cpu"), reset)
        b, _, _ = model(changed, context, model.init_state(2, "cpu"), reset)
    assert torch.equal(a[:, :15], b[:, :15]) and not torch.equal(a[:, 15:], b[:, 15:])


def test_mamba_trains_and_resumes_with_its_carried_state(tmp_path: Path) -> None:
    """The generic trainer drives the two-tensor state: the loss falls on 32 short documents, and
    a run resumed from its step-30 checkpoint ends on the weights of an uninterrupted one (the
    documents are packed without alignment, so windows continue documents and the carried state
    matters)."""
    spec = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
    task = spec.task("algorithmic", "in_range")
    tiny = TaskSpec.model_validate(
        {
            **task.model_dump(),
            "length_range": [16, 32],
            "dep_len_range": [8, 24],
            "unroll": 32,
            "params": {**task.params, "n_max": 4},
        }
    )
    data = ByteDataset([generate(tiny, "train", i) for i in range(32)])
    ctx = resolve(load_hardware_profile("cpu"))
    overrides = [
        "train.lr=0.003", "train.steps=60", "train.eval_every=30", "train.ckpt_every=30",
        "train.stats_every=30", "train.deterministic=true", "data.align_docs=false",
    ]  # fmt: skip

    model_a, cfg = _mamba(*overrides)
    history_a = Trainer(model_a.train(), cfg, ctx, run_dir=tmp_path / "a").fit(data, data, 60)
    losses = [record.loss for record in history_a.steps]
    assert sum(losses[-10:]) < 0.8 * sum(losses[:10])

    model_b, cfg = _mamba(*overrides, seed=1)  # other weights: everything comes from the file
    trainer_b = Trainer(model_b.train(), cfg, ctx, run_dir=tmp_path / "b")
    assert trainer_b.resume(tmp_path / "a" / "step_00000030.pt") == 30
    history_b = trainer_b.fit(data, data, 60)
    for rec_a, rec_b in zip(history_a.steps[30:], history_b.steps, strict=True):
        assert rec_a.loss == rec_b.loss, (rec_a.step, rec_a.loss, rec_b.loss)
    for key, value in model_a.state_dict().items():
        assert torch.equal(value, model_b.state_dict()[key]), key


@pytest.mark.parametrize("name", ["gru", "transformer", "selective_ssm"])
def test_address_input_reads_the_hsn_code_table(name: str) -> None:
    """With the HSN's width and seed the baseline's frozen table is the router's; it is a buffer,
    its projection is the only new parameter, and without the option the model is unchanged."""
    path = {"gru": "gru_2m", "transformer": "transformer_2m", "selective_ssm": "mamba_2m"}[name]
    base = [
        "baseline.gru.d_model=32", "baseline.gru.n_layers=1", "baseline.transformer.d_model=32",
        "baseline.transformer.n_layers=1", "baseline.transformer.n_heads=4",
        "baseline.transformer.ffn_hidden=64", *SMALL, "seed=3",
    ]  # fmt: skip
    config = REPO / "configs" / "baselines" / f"{path}.yaml"
    plain_cfg = load_config(config, overrides=base)
    cfg = load_config(config, overrides=[*base, "baseline.address_input=true"])
    seed_everything(3)
    plain = build_model(plain_cfg)
    seed_everything(3)
    model = build_model(cfg)
    assert isinstance(model, GRUBaseline | TransformerBaseline | MambaBaseline)
    assert plain.address is None and isinstance(model.address, AddressInput)
    assert not model.address.byte_codes.requires_grad
    assert "address.byte_codes" in model.state_dict()  # persistent: a checkpoint carries it
    assert [n for n, _ in model.address.named_parameters()] == ["proj.weight"]
    width = model.d_model
    assert model.param_report()["stored"] == plain.param_report()["stored"] + 2 * 64 * width
    hsn_cfg = load_config(REPO / "configs/experiments/hsn_recipe_a6.yaml", overrides=["seed=3"])
    seed_everything(3)
    scorer = HSN.from_config(hsn_cfg).router.scorer
    assert torch.equal(model.address.byte_codes, scorer.byte_codes)

    # previous byte: the one before in the window, BOS at a document start, nothing (zero) at a
    # window's first position that continues a document
    tokens = torch.tensor([[40, 41, 42, 43], [50, 51, 52, 53]])
    reset = torch.tensor([[True, False, True, False], [False, False, False, False]])
    codes = model.address.byte_codes
    weight = model.address.proj.weight
    out = model.address(tokens, reset)
    previous = [[BOS, 40, BOS, 42], [None, 50, 51, 52]]
    for b in range(2):
        for t in range(4):
            prev = previous[b][t]
            prev_code = torch.zeros(64) if prev is None else codes[prev]
            expect = weight @ torch.cat([codes[tokens[b, t]], prev_code])
            assert torch.allclose(out[b, t], expect, atol=1e-6)
