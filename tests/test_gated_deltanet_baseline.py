"""The Gated DeltaNet baseline (``model_name: gated_deltanet``,
``docs/gated_deltanet_baseline.md``).

Its module needs flash-linear-attention's kernels (``fla-core``), which the project environment
does not carry: there the tests of the model are skipped, and what stays is that nothing else
changed and that the missing kernels are reported in words. In the baseline's own environment
(``scripts/gated_deltanet_env.sh``) everything runs; the tests of the Triton kernels need a card.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from cards import CUDA_PROFILE
from hsn.config import RunConfig, config_hash, load_config, load_hardware_profile
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import TaskSpec, generate, load_dataset_spec
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import FAMILIES, memory_key
from hsn.runner import MODEL_REGISTRY, build_model
from hsn.train.loop import Trainer
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "baselines" / "gated_deltanet_2m.yaml"
MODULE = "hsn.baselines.gated_deltanet"
HAVE_FLA = importlib.util.find_spec("fla") is not None
needs_fla = pytest.mark.skipif(
    not HAVE_FLA,
    reason="flash-linear-attention (fla-core) is not in this environment: the Gated DeltaNet "
    "baseline runs in its own (scripts/gated_deltanet_env.sh)",
)
SMALL = [
    "baseline.gated_deltanet.d_model=32",
    "baseline.gated_deltanet.n_layers=2",
    "baseline.gated_deltanet.num_heads=2",
    "baseline.gated_deltanet.head_dim=12",
    "baseline.gated_deltanet.mlp_hidden=48",
    "train.unroll=32",
    "train.batch=8",
    "train.micro_batch=8",
]


def _model(*overrides: str, seed: int = 0, kernel: str = "recurrent") -> tuple[Any, RunConfig]:
    cfg = load_config(CONFIG, overrides=[*SMALL, *overrides])
    seed_everything(seed)
    model = build_model(cfg)
    model.kernel = kernel
    return model.eval(), cfg


def _window(B: int, T: int, seed: int = 0) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(32, 256, (B, T), generator=g), torch.randint(0, 16, (B, T), generator=g)


def _starts(B: int, T: int, *marks: tuple[int, int]) -> Tensor:
    """``reset [B, T]``: every row starts a document at 0, and at every ``(row, position)``."""
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    for row, position in marks:
        reset[row, position] = True
    return reset


def _random_state(model: Any, B: int, seed: int, device: str = "cpu") -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    return tuple(  # type: ignore[return-value]
        (0.3 * torch.randn(part.shape, generator=g)).to(device)
        for part in model.init_state(B, "cpu")
    )


def _run(model: Any, tokens: Tensor, context: Tensor, state: Any, reset: Tensor, T: int) -> Any:
    """The windows of width ``T`` one after the other, the state carried: logits, last state."""
    chunks = []
    for lo in range(0, tokens.shape[1], T):
        cols = slice(lo, lo + T)
        logits, state, _ = model(tokens[:, cols], context[:, cols], state, reset[:, cols])
        chunks.append(logits)
    return torch.cat(chunks, dim=1), state


# --------------------------------------------------------------------------------------------------
# without the kernels: nothing else changed, and the baseline says what it needs
# --------------------------------------------------------------------------------------------------


def test_other_models_are_unchanged() -> None:
    """The family is opt-in: every other configuration hashes as on main before it (5941d13),
    the example configuration included, the other baselines keep their memory keys, and the
    sweep spec and the registry know the name."""
    before = {
        "baselines/gru_2m": "c8dcbaa48e3b27ec",
        "baselines/transformer_2m": "65066a8915b79a32",
        "baselines/mamba_2m": "9f3bab36efded64a",
        "baselines/reader_w64l2": "9553d883a4bc17c7",
        "experiments/hsn_recipe_a6": "d8b36a25971c9694",
        "experiments/hsn_t_memory": "18c85210eec76722",
        "experiments/hsn_t_nomem": "427e8944984cd169",
        "ref_a1": "0831924010f9c5ec",
        "_example": "c0e6937825426376",
    }
    for name, digest in before.items():
        assert config_hash(load_config(REPO / "configs" / f"{name}.yaml")).startswith(digest), name
    # the memory keys of the experiment database (the stored step-time and memory estimates)
    protocol = {"name": "train_eval", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
    keys = {"gru_2m": "914bccd12d3b08c0", "mamba_2m": "932399457578460f"}
    for name, key in keys.items():
        other = load_config(REPO / "configs" / "baselines" / f"{name}.yaml")
        assert memory_key(other, "none", protocol) == key, name
    cfg = load_config(CONFIG)
    assert cfg.model_name == "gated_deltanet" and "gated_deltanet" in FAMILIES
    assert "gated_deltanet" in MODEL_REGISTRY
    wider = load_config(CONFIG, ["baseline.gated_deltanet.d_model=192"])
    assert memory_key(wider, "none", protocol) != memory_key(cfg, "none", protocol)
    assert memory_key(cfg, "none", protocol) not in keys.values()
    sized = load_config(
        REPO / "configs/baselines/gru_2m.yaml", ["baseline.gated_deltanet.d_model=64"]
    )
    assert config_hash(sized) != config_hash(load_config(REPO / "configs/baselines/gru_2m.yaml"))


def test_the_runner_imports_without_the_kernels() -> None:
    """``hsn.runner`` (and with it every other model) does not import the baseline's module or
    fla: only building a ``gated_deltanet`` model does."""
    code = (
        "import sys, hsn.runner, hsn.train.baseline_compile; "
        f"assert {MODULE!r} not in sys.modules and 'fla' not in sys.modules"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stderr[-2000:]


def test_missing_kernels_are_reported_in_words(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without fla the module does not import, and building the model says what is missing,
    which version and how to get an environment that has it."""
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    monkeypatch.setitem(sys.modules, "fla", None)  # `import fla` raises ImportError
    with pytest.raises(ImportError) as caught:
        build_model(load_config(CONFIG))
    message = str(caught.value)
    for part in ("fla-core==0.6.0", "scripts/gated_deltanet_env.sh", "gated_deltanet_baseline.md"):
        assert part in message, message
    assert MODULE not in sys.modules


def test_compiled_windows_are_refused_at_load() -> None:
    """The kernels are fla's own and are not traced by ``torch.compile``: the family runs eager,
    and ``hardware.baseline_compile`` says so instead of timing something else."""
    with pytest.raises(ValueError, match="runs eager"):
        load_config(CONFIG, overrides=["hardware.baseline_compile=reduce-overhead"])


# --------------------------------------------------------------------------------------------------
# the model (CPU): sizes, the recurrence against fla's references, windows, resets, training
# --------------------------------------------------------------------------------------------------


@needs_fla
def test_default_size_and_state() -> None:
    model = build_model(load_config(CONFIG))
    report = model.param_report()
    assert report["stored"] == report["active"] == report["trainable"] == 1_932_584
    assert model.state_size() == {"fast_weights": 46_464, "conv_inputs": 6_336, "total": 52_800}
    S, conv = model.init_state(2, "cpu")
    assert tuple(S.shape) == (4, 2, 3, 88, 44) and tuple(conv.shape) == (4, 2, 3, 528)
    assert S[:, 0].numel() + conv[:, 0].numel() == 52_800
    assert model.resolved_kernel(torch.device("cpu")) == "recurrent"
    assert model.resolved_kernel(torch.device("cuda")) == "chunk"
    # the decay's rate and step are one-dimensional: the trainer gives them no weight decay
    assert all(
        block.attn.A_log.ndim == 1 and block.attn.dt_bias.ndim == 1 for block in model.blocks
    )


@needs_fla
def test_delta_rule_matches_flas_references() -> None:
    """The three forms of the delta rule on the same inputs with a non-zero incoming state, over
    a window longer than one chunk of 64 and not a multiple of it: the recurrence of this module
    against fla's recurrent reference and against fla's parallel (chunked) reference."""
    from fla.ops.gated_delta_rule import naive_recurrent_gated_delta_rule

    model, _ = _model()
    mixer = model.blocks[0].attn
    B, T, H, K, V = 2, 150, mixer.num_heads, mixer.head_k, mixer.head_v
    g = torch.Generator().manual_seed(4)
    q, k = torch.randn(B, T, H, K, generator=g), torch.randn(B, T, H, K, generator=g)
    v = torch.randn(B, T, H, V, generator=g)
    a, b = torch.randn(B, T, H, generator=g), torch.randn(B, T, H, generator=g)
    S0 = 0.5 * torch.randn(B, H, V, K, generator=g)
    none = torch.zeros(B, T, dtype=torch.bool)
    with torch.no_grad():
        o, S = mixer._recurrent(q, k, v, a, b, S0, none, [])
        o_chunk, S_chunk = mixer._reference_chunk(q, k, v, a, b, S0, none)
        qn, kn, log_decay, beta = mixer._gates(q, k, a, b)
        o_ref, S_ref = naive_recurrent_gated_delta_rule(
            q=qn, k=kn, v=v, beta=beta, g=log_decay,
            initial_state=S0.transpose(-1, -2), output_final_state=True,
        )  # fmt: skip
    assert float(o.abs().max()) > 0.1
    for name, got, want in (
        ("recurrent reference: output", o, o_ref),
        ("recurrent reference: state", S, S_ref.transpose(-1, -2)),
        ("chunked reference: output", o, o_chunk),
        ("chunked reference: state", S, S_chunk),
    ):
        assert torch.allclose(got, want, atol=2e-5, rtol=1e-4), (name, (got - want).abs().max())
    # the gates as the kernels define them
    assert torch.allclose(qn.norm(dim=-1), torch.ones(B, T, H), atol=1e-4)
    assert bool((log_decay < 0).all()) and bool(((beta > 0) & (beta < 1)).all())


@needs_fla
def test_block_matches_a_direct_reference() -> None:
    """One block from a zero state equals the published layer written out directly: ``conv1d``
    for the three convolutions, fla's recurrent reference for the delta rule, the gated RMS
    norm and the gated MLP by hand."""
    from fla.ops.gated_delta_rule import naive_recurrent_gated_delta_rule

    model, _ = _model()
    block = model.blocks[0]
    mixer = block.attn
    B, T, H, K, V = 3, 20, mixer.num_heads, mixer.head_k, mixer.head_v
    x = torch.randn(B, T, model.d_model, generator=torch.Generator().manual_seed(3))
    S0, conv0 = model.init_state(B, "cpu")
    reset = _starts(B, T)
    whole = [torch.ones(B, T, 1, dtype=torch.bool)] * (mixer.d_conv - 1)
    tail = torch.ones(B, mixer.d_conv - 1, 1, dtype=torch.bool)
    with torch.no_grad():
        out, S, _ = block(x, S0[0], conv0[0], reset, whole, tail, "recurrent", None, [0])
        n = block.attn_norm(x)
        streams = []
        offset = 0
        for proj in (mixer.q_proj, mixer.k_proj, mixer.v_proj):
            width = proj.out_features
            weight = mixer.conv_weight[offset : offset + width].flip(1).unsqueeze(1)
            offset += width  # conv1d: the last tap is the current position
            y = F.conv1d(proj(n).transpose(1, 2), weight, padding=mixer.d_conv - 1, groups=width)
            streams.append(F.silu(y[..., :T].transpose(1, 2)))
        q, k = (s.view(B, T, H, K) for s in streams[:2])
        q, k = (s / (s.square().sum(dim=-1, keepdim=True) + 1e-6).sqrt() for s in (q, k))
        v = streams[2].view(B, T, H, V)
        log_decay = -mixer.A_log.exp() * F.softplus(mixer.a_proj(n) + mixer.dt_bias)
        o, S_ref = naive_recurrent_gated_delta_rule(
            q=q, k=k, v=v, beta=mixer.b_proj(n).sigmoid(), g=log_decay, output_final_state=True
        )
        o = o / o.square().mean(dim=-1, keepdim=True).add(mixer.eps).sqrt() * mixer.o_norm_weight
        o = o * F.silu(mixer.g_proj(n).view(B, T, H, V))
        mid = x + mixer.o_proj(o.reshape(B, T, H * V))
        m = block.mlp_norm(mid)
        ref = mid + block.down_proj(F.silu(block.gate_proj(m)) * block.up_proj(m))
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()
    assert torch.allclose(S, S_ref.transpose(-1, -2), atol=1e-5)


@needs_fla
def test_chunked_and_recurrent_forms_agree_on_a_small_model() -> None:
    """The whole model through the parallel (chunked) form and through the recurrence: two
    windows of 80 positions (more than one chunk) from a non-zero carried state give the same
    logits and the same state."""
    model, _ = _model()
    B, T = 3, 80
    tokens, context = _window(B, 2 * T, seed=2)
    reset = torch.zeros(B, 2 * T, dtype=torch.bool)
    reset[1, 0] = True  # one row starts a document, the others continue theirs
    reset[2, T] = True  # and one at the second window's first position
    state = _random_state(model, B, seed=9)
    results = {}
    with torch.no_grad():
        for kernel in ("recurrent", "reference_chunk"):
            model.kernel = kernel
            results[kernel] = _run(model, tokens, context, state, reset, T)
    (logits_a, state_a), (logits_b, state_b) = results["recurrent"], results["reference_chunk"]
    assert torch.allclose(logits_a, logits_b, atol=1e-4), (logits_a - logits_b).abs().max()
    for part_a, part_b in zip(state_a, state_b, strict=True):
        assert torch.allclose(part_a, part_b, atol=1e-5), (part_a - part_b).abs().max()
    assert not torch.equal(state_a[0], state[0])
    model.kernel = "reference_chunk"
    with pytest.raises(ValueError, match="document start"):
        model(tokens[:, :T], context[:, :T], state, _starts(B, T, (0, 5)))
    model.kernel = "chunk"
    with pytest.raises(RuntimeError, match="CUDA"):
        model(tokens[:, :T], context[:, :T], state, reset[:, :T])


@needs_fla
def test_two_windows_equal_one_long_window() -> None:
    """Carrying the state over a window boundary changes nothing: two windows of 24 positions
    give the logits and the final state of one window of 48, from a carried state and with a
    document start within the convolution's reach of the boundary."""
    model, _ = _model()
    B, T = 3, 24
    tokens, context = _window(B, 2 * T, seed=6)
    reset = torch.zeros(B, 2 * T, dtype=torch.bool)
    reset[0, T - 2] = True
    reset[1, T + 1] = True
    state = _random_state(model, B, seed=1)
    with torch.no_grad():
        long_logits, long_state = _run(model, tokens, context, state, reset, 2 * T)
        logits, carried = _run(model, tokens, context, state, reset, T)
    assert torch.allclose(logits, long_logits, atol=1e-5), (logits - long_logits).abs().max()
    for part, long_part in zip(carried, long_state, strict=True):
        assert torch.allclose(part, long_part, atol=1e-6)


@needs_fla
def test_windows_and_resets_equal_each_document_alone() -> None:
    """Three windows with carried state and document starts anywhere (a window's first, middle
    and last position) give every document the logits it gets alone from a fresh state: a reset
    clears the fast weights and the convolution's inputs, whatever the state was."""
    model, _ = _model()
    B, T, W = 4, 16, 3
    tokens, context = _window(B, T * W, seed=5)
    reset = _starts(
        B, T * W,
        (0, 7),  # inside the first window
        (1, T), (1, T + 1),  # the first position of the second window; a one-position document
        (2, 2 * T - 1),  # the last position of a window
        (2, 2 * T + 2),  # within the convolution's reach of a window start
    )  # fmt: skip
    # an incoming state that is not zero shows reset[:, 0] clears both parts of it
    state = _random_state(model, B, seed=3)
    with torch.no_grad():
        windowed, last = _run(model, tokens, context, state, reset, T)
        for b in range(B):
            starts = [*reset[b].nonzero().flatten().tolist(), T * W]
            for lo, hi in zip(starts[:-1], starts[1:], strict=True):
                alone, _, info = model(
                    tokens[b : b + 1, lo:hi],
                    context[b : b + 1, lo:hi],
                    model.init_state(1, "cpu"),
                    _starts(1, hi - lo),
                )
                diff = (windowed[b, lo:hi] - alone[0]).abs().max()
                assert diff < 1e-4, (b, lo, hi, float(diff))
    assert "state_abs_max" in info
    mask = torch.tensor([True, False, True, False])
    cleared = model.reset_state(last, mask)
    for part, before in zip(cleared, last, strict=True):
        assert before[:, mask].any()
        assert not part[:, mask].any() and torch.equal(part[:, ~mask], before[:, ~mask])
    with pytest.raises(ValueError, match="state"):
        model(tokens[:, :T], context[:, :T], last[0], reset[:, :T])


@needs_fla
def test_is_causal() -> None:
    model, _ = _model()
    tokens, context = _window(2, 24, seed=7)
    reset = _starts(2, 24)
    changed = tokens.clone()
    changed[:, 15] = (changed[:, 15] + 1 - 32) % 224 + 32
    with torch.no_grad():
        a, _, _ = model(tokens, context, model.init_state(2, "cpu"), reset)
        b, _, _ = model(changed, context, model.init_state(2, "cpu"), reset)
    assert torch.equal(a[:, :15], b[:, :15]) and not torch.equal(a[:, 15:], b[:, 15:])


def _tiny_documents(n: int = 32) -> ByteDataset:
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
    return ByteDataset([generate(tiny, "train", i) for i in range(n)])


TRAIN = [
    "train.lr=0.003", "train.steps=60", "train.eval_every=30", "train.ckpt_every=30",
    "train.stats_every=30", "train.deterministic=true", "data.align_docs=false",
]  # fmt: skip


@needs_fla
def test_trains_and_resumes_with_its_carried_state(tmp_path: Path) -> None:
    """The generic trainer drives the two-tensor state: the loss falls on 32 short documents, and
    a run resumed from its step-30 checkpoint ends on the weights of an uninterrupted one (the
    documents are packed without alignment, so windows continue documents and the carried state
    matters)."""
    data = _tiny_documents()
    ctx = resolve(load_hardware_profile("cpu"))
    model_a, cfg = _model(*TRAIN, kernel="auto")
    history_a = Trainer(model_a.train(), cfg, ctx, run_dir=tmp_path / "a").fit(data, data, 60)
    losses = [record.loss for record in history_a.steps]
    assert sum(losses[-10:]) < 0.8 * sum(losses[:10])

    model_b, cfg = _model(*TRAIN, seed=1, kernel="auto")  # other weights: all from the file
    trainer_b = Trainer(model_b.train(), cfg, ctx, run_dir=tmp_path / "b")
    assert trainer_b.resume(tmp_path / "a" / "step_00000030.pt") == 30
    history_b = trainer_b.fit(data, data, 60)
    for rec_a, rec_b in zip(history_a.steps[30:], history_b.steps, strict=True):
        assert rec_a.loss == rec_b.loss, (rec_a.step, rec_a.loss, rec_b.loss)
    for key, value in model_a.state_dict().items():
        assert torch.equal(value, model_b.state_dict()[key]), key


@needs_fla
def test_address_input_is_the_other_baselines() -> None:
    """``baseline.address_input`` adds the frozen byte codes through one projection, as for the
    other baselines; without it the model has no such module."""
    plain, _ = _model()
    model, _ = _model("baseline.address_input=true")
    assert plain.address is None and model.address is not None
    assert [n for n, _ in model.address.named_parameters()] == ["proj.weight"]
    assert model.param_report()["stored"] == plain.param_report()["stored"] + 2 * 64 * 32
    tokens, context = _window(2, 12)
    with torch.no_grad():
        logits, _, _ = model(tokens, context, model.init_state(2, "cpu"), _starts(2, 12))
    assert bool(torch.isfinite(logits).all())


# --------------------------------------------------------------------------------------------------
# the Triton kernels (CUDA)
# --------------------------------------------------------------------------------------------------


def _cuda() -> None:
    if not HAVE_FLA:
        pytest.skip("flash-linear-attention (fla-core) is not in this environment")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")


#: How close fla's Triton kernels are to the recurrence in float32, relative to the largest value
#: of what is compared (for a gradient: the norm of the difference over the gradient's norm). The
#: kernels multiply matrices on the card's tensor cores in TF32 (a 10-bit mantissa), as Triton's
#: ``dot`` does by default on Ampere and newer, so the two agree to about three digits and not
#: to float32 rounding. Measured on the RTX 3060 (17 Nov 2026, this test's two cases): logits
#: 9e-4 and 2e-4, state 1.4e-3 and 1.1e-3, gradients 3.5e-3 and 2.0e-2 (the worst parameter:
#: a decay parameter in the second case); at the default size 1.8e-3, 9e-4 and 1.9e-2.
KERNEL_TOL = {"logits": 5e-3, "state": 5e-3, "grads": 5e-2}


@pytest.mark.cuda
def test_cuda_chunk_kernel_equals_the_recurrence() -> None:
    """On the card, in float32: fla's Triton kernels (``chunk``) against the recurrence written
    out, the whole model over three windows of 96 positions with a non-zero carried state.
    Without a document start inside a window the kernels run in their batch form, with starts
    anywhere in their variable-length form; both give the recurrence's logits, state and
    parameter gradients (the state's gradient across the windows included) to the precision of
    the kernels (:data:`KERNEL_TOL`)."""
    _cuda()
    model, _ = _model()
    model = model.to("cuda")
    B, T, W = 4, 96, 3
    tokens, context = (x.cuda() for x in _window(B, T * W, seed=8))
    state = _random_state(model, B, seed=2, device="cuda")
    aligned = torch.zeros(B, T * W, dtype=torch.bool)
    aligned[0, 0] = aligned[1, T] = aligned[3, 2 * T] = True
    anywhere = _starts(
        B, T * W, (0, 7), (0, 70), (1, T), (1, T + 1), (2, 2 * T - 1), (2, 2 * T + 2), (3, 150)
    )
    for name, reset in (("batch form", aligned.cuda()), ("variable-length form", anywhere.cuda())):
        results, grads = {}, {}
        for kernel in ("recurrent", "chunk"):
            model.kernel = kernel
            model.zero_grad()
            logits, last = _run(model, tokens, context, state, reset, T)
            weight = torch.randn(logits.shape, generator=torch.Generator().manual_seed(0)).cuda()
            (logits * weight).sum().backward()
            results[kernel] = (logits.detach(), last)
            grads[kernel] = {n: p.grad.clone() for n, p in model.named_parameters()}
        (logits_a, state_a), (logits_b, state_b) = results["recurrent"], results["chunk"]
        worst = float((logits_a - logits_b).abs().max() / logits_a.abs().max())
        assert worst < KERNEL_TOL["logits"], (name, "logits", worst)
        for part_a, part_b in zip(state_a, state_b, strict=True):
            assert part_b.dtype == torch.float32
            worst = float((part_a - part_b).abs().max() / part_a.abs().max())
            assert worst < KERNEL_TOL["state"], (name, "state", worst)
        for key, grad in grads["recurrent"].items():
            worst = float((grad - grads["chunk"][key]).norm() / grad.norm())
            assert worst < KERNEL_TOL["grads"], (name, key, worst)


@pytest.mark.cuda
def test_cuda_two_windows_equal_one_long_window_and_resets_clear() -> None:
    """On the card through the Triton kernels: two windows of 128 give the logits and the state
    of one window of 256, and a row that starts a document gets what it gets from a zero state
    whatever state came in."""
    _cuda()
    model, _ = _model(kernel="chunk")
    model = model.to("cuda")
    B, T = 4, 128
    tokens, context = (x.cuda() for x in _window(B, 2 * T, seed=12))
    reset = torch.zeros(B, 2 * T, dtype=torch.bool, device="cuda")
    reset[0, T - 2] = reset[1, T + 1] = True
    state = _random_state(model, B, seed=5, device="cuda")
    with torch.no_grad():
        long_logits, long_state = _run(model, tokens, context, state, reset, 2 * T)
        logits, carried = _run(model, tokens, context, state, reset, T)
        fresh = _starts(B, 2 * T).cuda()
        from_state, _ = _run(model, tokens, context, state, fresh, T)
        from_zero, _ = _run(model, tokens, context, model.init_state(B, "cuda"), fresh, T)
    assert float((logits - long_logits).abs().max()) < 2e-4
    for part, long_part in zip(carried, long_state, strict=True):
        assert torch.allclose(part, long_part, atol=2e-5, rtol=1e-4)
    assert torch.equal(from_state, from_zero)


@pytest.mark.cuda
def test_cuda_trains_under_autocast_through_the_kernels(tmp_path: Path) -> None:
    """On the card under the profile's autocast: 60 steps and two evaluations through the Triton
    kernels with a finite, falling loss, documents packed without alignment (so the windows
    carry state and start documents anywhere); the carried state stays float32."""
    _cuda()
    data = _tiny_documents()
    ctx = resolve(load_hardware_profile(CUDA_PROFILE))
    pins = [f"profile={CUDA_PROFILE}", "hardware.compile_mode=none"]
    train = [t for t in TRAIN if "deterministic" not in t]
    model, cfg = _model(*train, *pins, kernel="auto")
    trainer = Trainer(model.to(ctx.device).train(), cfg, ctx, run_dir=tmp_path / "run")
    history = trainer.fit(data, data, 60)
    losses = [record.loss for record in history.steps]
    assert all(loss == loss and loss < 1e3 for loss in losses)
    assert sum(losses[-10:]) < 0.8 * sum(losses[:10])
    assert len(history.evals) == 2
    assert model.resolved_kernel(ctx.device) == "chunk"


@pytest.mark.cuda
def test_cuda_mixer_equals_flas_layer() -> None:
    """On the card, where ``fla.layers`` is installed (the flash-linear-attention package, not
    only its kernels): the mixer of this module with its weights copied into
    ``fla.layers.GatedDeltaNet`` gives that layer's output, in float32."""
    _cuda()
    try:
        from fla.layers.gated_deltanet import GatedDeltaNet
    except ImportError:
        pytest.skip("fla.layers (pip flash-linear-attention) is not installed: kernels only")

    model, cfg = _model(kernel="chunk")
    model = model.to("cuda")
    mixer = model.blocks[0].attn
    spec = cfg.baseline.gated_deltanet
    layer = GatedDeltaNet(
        hidden_size=spec.d_model, expand_v=spec.expand_v, head_dim=spec.head_dim,
        num_heads=spec.num_heads, conv_size=spec.d_conv, norm_eps=spec.norm_eps,
    ).cuda()  # fmt: skip
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj", "a_proj", "b_proj", "g_proj", "o_proj"):
            getattr(layer, name).weight.copy_(getattr(mixer, name).weight)
        layer.A_log.copy_(mixer.A_log)
        layer.dt_bias.copy_(mixer.dt_bias)
        layer.o_norm.weight.copy_(mixer.o_norm_weight)
        widths = [mixer.key_dim, mixer.key_dim, mixer.value_dim]
        parts = mixer.conv_weight.split(widths, dim=0)
        for conv, part in zip((layer.q_conv1d, layer.k_conv1d, layer.v_conv1d), parts, strict=True):
            conv.weight.copy_(part.flip(1).unsqueeze(1))  # Conv1d: the last tap is the current one
    B, T = 3, 200
    x = torch.randn(B, T, spec.d_model, generator=torch.Generator().manual_seed(1)).cuda()
    S0, conv0 = model.init_state(B, "cuda")
    whole = [torch.ones(B, T, 1, dtype=torch.bool, device="cuda")] * (mixer.d_conv - 1)
    tail = torch.ones(B, mixer.d_conv - 1, 1, dtype=torch.bool, device="cuda")
    layer.train()
    out, _, _ = mixer(x, S0[0], conv0[0], _starts(B, T).cuda(), whole, tail, "chunk", None, [0])
    ref, _, _ = layer(x)
    assert float(ref.abs().max()) > 1e-3
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4), (out - ref).abs().max()
