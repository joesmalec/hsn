"""The attention read of the cells (``model.cell_attention_heads``; HSN-T only, opt-in candidate;
``docs/amendments/cell_attention_read.md``).

With ``H > 0`` the reader's output ``c_t`` queries all cells at every position and the result
``m_t`` is one more input of the decoder, ``D([q ; u ; r ; c ; m])``. These tests check that the
key is opt-in (hashes, memory keys, the model at the default), the refusals, the closed forms,
that no other parameter's initialisation changes, that the memory does not read ``m_t`` (the
state of a window is bit for bit the state without the key), what ``m_t`` is (the equations,
from the state after position ``t``), what it may depend on (no later byte, no earlier
document), where its gradient goes (the new layers, the reader, and the cell state under every
``train.feedback_detach``), the compiled window, and that the trainer trains and evaluates
with it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import (
    HASH_OMIT_AT_DEFAULT,
    MEMORY_MODULES,
    RunConfig,
    canonical_json,
    config_hash,
    load_config,
    split_window_scope_error,
)
from hsn.device import resolve, seed_everything
from hsn.experiments.spec import memory_key
from hsn.model.cell_read import CellAttentionRead
from hsn.model.hsn import HSN
from hsn.model.hsn_t import HSNT
from hsn.runner import build_model
from hsn.train.memory_plan import cell_read_param_count, expected_param_count, reader_param_count
from hsn.train.tbptt import TBPTTTrainer
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, hsn_config
from test_a6_feedback_detach import A5, _cfg, _start, _window
from test_a8_fast_cells import COMPILED
from test_hsn_t import CUT_C, FAMILY, PREV, READER, SHORT, W, _gap, _model
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
NOMEM = REPO / "configs" / "experiments" / "hsn_t_nomem.yaml"
READER_LM = REPO / "configs" / "baselines" / "reader_w64l2.yaml"

D = 8  # the cell width of the six-cell model of the A6 tests
HEADS = 2
KEY = f"model.cell_attention_heads={HEADS}"
ON = (*READER, KEY)
#: A6's full cut: the cell's own read of the state is a constant in the backward pass too
CUT_ALL = "train.feedback_detach=[summary,eps,state,router]"
HYBRID = [FAMILY, "model.reader=transformer", "model.k_select=12"]


def _on_cfg(*extra: str) -> RunConfig:
    return _cfg(*A5, *ON, *extra)


def _reads(
    model: HSNT, tokens: Tensor, ctx: Tensor, state: object, reset: Tensor, **kwargs: object
) -> tuple[Tensor, Tensor, object, dict[str, Tensor]]:
    """A window and the read of every position: ``(logits, m [B, T, d], state, info)``; ``m``
    is what the module returned inside the window, autograd intact."""
    assert model.cell_read is not None
    seen: list[Tensor] = []
    hook = model.cell_read.register_forward_hook(lambda _m, _i, out: seen.append(out))
    try:
        logits, new_state, info = model(tokens, ctx, state, reset, **kwargs)
    finally:
        hook.remove()
    assert len(seen) == tokens.shape[1]
    return logits, torch.stack(seen, dim=1), new_state, info


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_key_is_opt_in_and_the_default_keeps_its_hash() -> None:
    base = load_config(RECIPE_A6)
    assert base.model.cell_attention_heads == 0
    assert "cell_attention_heads" not in json.loads(canonical_json(base))["model"]
    assert HASH_OMIT_AT_DEFAULT[-1] == "model.cell_attention_heads"
    assert config_hash(load_config(RECIPE_A6, ["model.cell_attention_heads=0"])) == config_hash(
        base
    )
    hybrid = load_config(RECIPE_A6, HYBRID)
    named = load_config(RECIPE_A6, [*HYBRID, "model.cell_attention_heads=0"])
    assert config_hash(named) == config_hash(hybrid)
    # the arm id reads the non-default values: the key is not among them
    assert "cell_attention_heads" not in hybrid.model.model_dump(exclude_defaults=True)
    on = load_config(RECIPE_A6, [*HYBRID, "model.cell_attention_heads=4"])
    more = load_config(RECIPE_A6, [*HYBRID, "model.cell_attention_heads=8"])
    assert len({config_hash(hybrid), config_hash(on), config_hash(more)}) == 3
    assert json.loads(canonical_json(on))["model"]["cell_attention_heads"] == 4
    # the memory keys of the experiment database, as main computed them before the key existed
    # (tests/test_hsn_t_no_memory.py pins the same two): the stored estimates still match
    protocol = {"name": "reads_state", "eval_batch": 128, "probe_batch": 64, "diag_docs": 32}
    assert memory_key(base, "pdg", protocol) == "c33ec88833506533"
    assert memory_key(hybrid, "pdg", protocol) == "c8bcf4024c9285fe"
    assert memory_key(on, "pdg", protocol) != memory_key(hybrid, "pdg", protocol)
    assert "cell_read" in MEMORY_MODULES  # the read is part of the memory: gone with it


def test_default_model_has_no_read() -> None:
    cfg = _cfg(*A5, *READER)
    model = HSNT.from_config(cfg)
    assert model.cell_read is None and "cell_read" not in model.param_report()
    assert model.decoder.fc1.in_features == 3 * D + W
    assert not any(name.startswith("cell_read.") for name, _ in model.named_parameters())
    assert cell_read_param_count(cfg.model) == 0


@pytest.mark.parametrize(
    ("path", "overrides", "message"),
    [
        (RECIPE_A6, ["model.cell_attention_heads=4"], "set model.reader=transformer"),
        (RECIPE_A6, [*HYBRID, "model.cell_attention_heads=-1"], "greater than or equal to 0"),
        (RECIPE_A6, [*HYBRID, "model.cell_attention_heads=5"], "multiple of the number of heads"),
        (
            RECIPE_A6,
            [*HYBRID, "model.cell_attention_heads=4", "model.key_address=false"],
            "needs model.key_address=true",
        ),
        (NOMEM, ["model.cell_attention_heads=4"], "model.memory=false has none"),
        (READER_LM, ["model.cell_attention_heads=4"], "cell_attention_heads"),
        (
            RECIPE_A6,
            [*HYBRID, "model.cell_attention_heads=4", "hardware.window=split"],
            "built into the autograd window only",
        ),
        (
            RECIPE_A6,
            [*HYBRID, "model.cell_attention_heads=4", "hardware.window=fused"],
            "built into the autograd window only",
        ),
    ],
)
def test_configuration_refusals(path: Path, overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(path, overrides)


def test_split_window_scope_names_the_key() -> None:
    cfg = load_config(RECIPE_A6, [*HYBRID, "model.cell_attention_heads=4"])
    reason = split_window_scope_error(cfg)
    assert reason is not None and "cell_attention_heads=4" in reason


def test_models_refuse_what_they_cannot_read() -> None:
    cfg = _on_cfg()
    with pytest.raises(ValueError, match="HSN-T"):  # the HSN itself has no reader and no read
        HSN.from_config(cfg)
    model = _model(cfg)
    B = 2
    with pytest.raises(ValueError, match="reads the bytes of a window"):  # no step for HSN-T
        model.step(
            torch.zeros(B, dtype=torch.int64), torch.zeros(B, dtype=torch.int64), _start(model, B)
        )
    with pytest.raises(ValueError, match="cell_attention_heads >= 1"):
        CellAttentionRead(_cfg(*A5, *READER).model)
    with pytest.raises(ValueError, match="model.reader"):
        CellAttentionRead(_cfg(*A5).model.model_copy(update={"cell_attention_heads": 2}))


# --------------------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------------------


def test_param_report_matches_the_closed_form() -> None:
    for cfg in (_on_cfg(), load_config(RECIPE_A6, [*HYBRID, "model.cell_attention_heads=4"])):
        model = HSNT.from_config(cfg)
        report, expected = model.param_report(), expected_param_count(cfg.model)
        for row, count in expected.items():
            assert report[row] == count, row
        assert report["cell_read"] == cell_read_param_count(cfg.model)
        assert report["total_trainable"] == (
            report["total"] + report["edges"] + report["reader"] + report["cell_read"]
        )
    m = cfg.model  # the recipe: d 64, the default reader (w 64)
    assert cell_read_param_count(m) == m.d * (m.reader_width + 4 * m.d + 4) == 20_736
    off = load_config(RECIPE_A6, HYBRID).model
    assert expected["total"] - expected_param_count(off)["total"] == m.d * m.hidden
    assert reader_param_count(m) == reader_param_count(off)


def test_read_keeps_the_other_parameters_initialisation() -> None:
    """The read and the decoder's new columns are drawn from their own generators: every
    parameter the two models share starts equal, the decoder's old columns too, and the global
    random stream is where it was."""
    torch.manual_seed(0)
    plain = HSNT.from_config(_cfg(*A5, *READER))
    after_plain = torch.rand(4)
    torch.manual_seed(0)
    read = HSNT.from_config(_on_cfg())
    after_read = torch.rand(4)
    assert torch.equal(after_plain, after_read)
    theirs = dict(read.named_parameters())
    for name, p in plain.named_parameters():
        q = theirs.pop(name)
        if name == "decoder.fc1.weight":
            assert q.shape[1] == p.shape[1] + D
            assert torch.equal(q[:, : p.shape[1]], p)
            assert float(q[:, p.shape[1] :].detach().abs().max()) > 0.0
        else:
            assert torch.equal(p, q), name
    assert sorted(theirs) == [
        "cell_read.address.weight",
        "cell_read.out.bias",
        "cell_read.out.weight",
        "cell_read.query.bias",
        "cell_read.query.weight",
        "cell_read.state.bias",
        "cell_read.state.weight",
    ]
    for name, p in theirs.items():
        assert float(p.detach().abs().max()) > 0.0 or name.endswith(".bias"), name
    for name, buffer in plain.named_buffers():
        assert torch.equal(buffer, dict(read.named_buffers())[name]), name


@pytest.mark.parametrize("cut", [(), (CUT_C,)])
@pytest.mark.parametrize("training", [True, False])
def test_memory_does_not_read_the_read(cut: tuple[str, ...], training: bool) -> None:
    """``m_t`` enters through the decoder's new columns and nowhere else. The state and the
    routes of a window are bit for bit those of the model without the key, and with the new
    columns zero the logits are its logits."""
    seed_everything(0)
    plain = HSNT.from_config(_cfg(*A5, *READER, *cut)).double().train(training)
    seed_everything(0)
    read = HSNT.from_config(_on_cfg(*cut)).double().train(training)
    cfg = _on_cfg(*cut)
    tokens, ctx, reset = _window(cfg)
    reset[1, 6] = True
    state = _start(plain, tokens.shape[0])
    ref_logits, ref_state, ref_info = plain(tokens, ctx, state, reset)
    logits, new_state, info = read(tokens, ctx, state, reset)
    assert torch.equal(new_state.h, ref_state.h) and torch.equal(new_state.q, ref_state.q)
    assert torch.equal(new_state.age, ref_state.age) and torch.equal(new_state.pos, ref_state.pos)
    for key in ("cell_idx", "unit_idx", "valid", "u", "eps", "z_hat"):
        assert torch.equal(info[key], ref_info[key]), key
    assert _gap(logits, ref_logits) > 1e-6  # the new columns are read
    with torch.no_grad():
        read.decoder.fc1.weight[:, -D:] = 0.0
    logits, _, _ = read(tokens, ctx, state, reset)
    assert _gap(logits, ref_logits) <= 1e-12


def _reference_read(model: HSNT, c: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
    """The equations of the document, written out per head: ``(m [B, d], α [B, H, N])``."""
    read = model.cell_read
    assert read is not None
    B, N, d = h.shape
    H, e = read.heads, d // read.heads
    address = model.scorer.cell_addresses  # [N, d]
    w_key = torch.cat([read.state.weight[:d], read.address.weight], dim=1)  # W_K: [d, 2d]
    key_in = torch.cat([h, address.expand(B, N, d)], dim=-1)  # [h_i ; a_i]
    keys = key_in @ w_key.T + read.state.bias[:d]
    values = h @ read.state.weight[d:].T + read.state.bias[d:]
    query = c @ read.query.weight.T + read.query.bias
    mixed, alphas = [], []
    for j in range(H):
        cols = slice(j * e, (j + 1) * e)
        scores = torch.einsum("be,bne->bn", query[:, cols], keys[:, :, cols]) / math.sqrt(e)
        alpha = torch.softmax(scores, dim=-1)
        alphas.append(alpha)
        mixed.append(torch.einsum("bn,bne->be", alpha, values[:, :, cols]))
    m = torch.cat(mixed, dim=-1) @ read.out.weight.T + read.out.bias
    return m, torch.stack(alphas, dim=1)


@pytest.mark.parametrize("extra", [(), SHORT[len(READER) :], PREV[len(READER) :]])
@pytest.mark.parametrize("training", [True, False])
def test_read_is_the_attention_over_the_cells_after_the_position(
    extra: tuple[str, ...], training: bool
) -> None:
    """``m_t`` is the equations' value from ``c_t`` and the cell states after position ``t``
    (the state ``r_t`` reads), over every cell."""
    cfg = _on_cfg(*extra)
    model = _model(cfg).train(training)
    assert model.reader is not None and model.cell_read is not None
    tokens, ctx, reset = _window(cfg)
    reset[1, 5] = True
    state = _start(model, tokens.shape[0])
    with torch.no_grad():
        logits, m, _, info = _reads(model, tokens, ctx, state, reset, capture=True)
        again, _, _ = model(tokens, ctx, state, reset)
        assert torch.equal(logits, again)
        c = model.reader(tokens, reset)
        states = info["cell_states"]  # [B, T, N, d]: h after position t
        for t in range(tokens.shape[1]):
            want, alpha = _reference_read(model, c[:, t], states[:, t])
            assert _gap(m[:, t], want) <= 1e-12, t
            got_m, got_alpha = model.cell_read.read(
                c[:, t], states[:, t], model.scorer.cell_addresses
            )
            assert _gap(got_alpha, alpha) <= 1e-12 and _gap(got_m, want) <= 1e-12
            assert got_alpha.shape == (tokens.shape[0], HEADS, cfg.model.n_cells)
            assert _gap(got_alpha.sum(-1), torch.ones_like(got_alpha.sum(-1))) <= 1e-12
            assert float(got_alpha.min()) > 0.0  # every cell is read: no top-k
        if tokens.shape[1] > 1:  # not the state before the position
            stale, _ = _reference_read(model, c[:, 1], states[:, 0])
            assert _gap(m[:, 1], stale) > 1e-6


def test_keys_read_the_address_code() -> None:
    """With the state half of the keys at zero, a head still tells the cells apart: by their
    fixed address codes."""
    cfg = _on_cfg()
    model = _model(cfg)
    assert model.cell_read is not None
    with torch.no_grad():
        model.cell_read.state.weight[:D] = 0.0
    g = torch.Generator().manual_seed(3)
    c = torch.randn(2, W, generator=g, dtype=torch.float64)
    h = torch.randn(2, cfg.model.n_cells, D, generator=g, dtype=torch.float64)
    with torch.no_grad():
        _, alpha = model.cell_read.read(c, h, model.scorer.cell_addresses)
    assert float(alpha.detach().std(dim=-1).min()) > 1e-4
    _, same = model.cell_read.read(c, torch.zeros_like(h), model.scorer.cell_addresses)
    assert _gap(alpha, same) <= 1e-12  # and by nothing else
    _, flat = model.cell_read.read(c, h, torch.zeros_like(model.scorer.cell_addresses))
    assert _gap(flat, torch.full_like(flat, 1.0 / cfg.model.n_cells)) <= 1e-12


@pytest.mark.parametrize("extra", [(), SHORT[len(READER) :], PREV[len(READER) :]])
def test_read_depends_on_no_later_byte_and_no_earlier_document(extra: tuple[str, ...]) -> None:
    cfg = _on_cfg(*extra)
    model = _model(cfg)
    tokens, ctx, reset = _window(cfg)
    B, T = tokens.shape
    k, t0 = 4, 8
    reset[0, k] = True
    state = _start(model, B)
    with torch.no_grad():
        logits, m, _, _ = _reads(model, tokens, ctx, state, reset)
        later = tokens.clone()
        later[:, t0:] = (later[:, t0:] + 5) % cfg.model.vocab_size
        changed, m_later, _, _ = _reads(model, later, ctx, state, reset)
        assert _gap(m_later[:, :t0], m[:, :t0]) <= 1e-12
        assert _gap(m_later[:, t0:], m[:, t0:]) > 1e-6
        assert _gap(changed[:, :t0], logits[:, :t0]) <= 1e-12
        assert _gap(changed[:, t0:], logits[:, t0:]) > 1e-6
        before = tokens.clone()
        before[0, :k] = (before[0, :k] + 9) % cfg.model.vocab_size
        changed, m_before, _, _ = _reads(model, before, ctx, state, reset)
        assert _gap(m_before[0, :k], m[0, :k]) > 1e-6
        assert _gap(m_before[0, k:], m[0, k:]) <= 1e-12
        assert _gap(changed[0, k:], logits[0, k:]) <= 1e-12
        # the carried state is read: in a window that continues its documents another start
        # changes the read from the first position on
        carried = torch.zeros_like(reset)
        _, m_one, _, _ = _reads(model, tokens, ctx, state, carried)
        _, m_two, _, _ = _reads(model, tokens, ctx, _start(model, B, seed=5), carried)
        assert _gap(m_one[:, 0], m_two[:, 0]) > 1e-6


# --------------------------------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [(), (CUT_C,), (CUT_ALL,)])
def test_gradients_reach_the_read_the_reader_and_the_new_columns(cut: tuple[str, ...]) -> None:
    cfg = _on_cfg(*cut)
    model = _model(cfg)
    assert model.reader is not None and model.cell_read is not None
    tokens, ctx, reset = _window(cfg)
    logits, _, _ = model(tokens, ctx, _start(model, tokens.shape[0]), reset)
    logits.square().mean().backward()
    for name, p in model.cell_read.named_parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, name
    grad = model.decoder.fc1.weight.grad
    assert grad is not None and float(grad[:, -D:].abs().sum()) > 0.0
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())


@pytest.mark.parametrize("cut", [(), (CUT_C,), (CUT_ALL,)])
def test_read_passes_its_gradient_into_the_cell_state(cut: tuple[str, ...]) -> None:
    """The read is on ``r_t``'s terms: it reads the state with its gradient, and no path of
    ``train.feedback_detach`` cuts it. From the last position's ``m`` alone a gradient reaches
    the carried state, the weights that wrote the cells and, through the query, the reader."""
    cfg = _on_cfg(*cut)
    model = _model(cfg)
    assert model.reader is not None
    tokens, ctx, reset = _window(cfg)
    reset[:] = False
    start = _start(model, tokens.shape[0])
    h = start.h.clone().requires_grad_(True)
    state = type(start)._make(h=h, q=start.q, age=start.age, pos=start.pos)
    _, m, _, _ = _reads(model, tokens, ctx, state, reset)
    m[:, -1].square().sum().backward()
    assert h.grad is not None and float(h.grad.abs().sum()) > 0.0
    written = [p.grad for name, p in model.cell.named_parameters() if p.grad is not None]
    assert written and sum(float(g.abs().sum()) for g in written) > 0.0
    assert sum(float(p.grad.abs().sum()) for p in model.reader.parameters()) > 0.0
    assert model.decoder.fc1.weight.grad is None  # nothing but the read was differentiated


def test_read_and_selected_readout_share_the_state_tensor() -> None:
    """Both reads of a position take the same state object: what limits or cuts the gradient
    of one at the entry of a position (``train.state_grad_clip``) acts on the other alike."""
    cfg = _on_cfg("train.state_grad_clip=1e-3")
    model = _model(cfg)
    assert model.cell_read is not None
    seen: list[tuple[int, int]] = []
    inner = model._selected_readout

    def spy(state: object, *args: object, **kwargs: object) -> Tensor:
        seen.append((id(state.h), 0))  # type: ignore[attr-defined]
        return inner(state, *args, **kwargs)  # type: ignore[arg-type]

    hook = model.cell_read.register_forward_hook(
        lambda _m, inputs, _o: seen.append((id(inputs[1]), 1))
    )
    model._selected_readout = spy  # type: ignore[method-assign]
    tokens, ctx, reset = _window(cfg)
    logits, _, _ = model(tokens, ctx, _start(model, tokens.shape[0]), reset)
    hook.remove()
    T = tokens.shape[1]
    assert len(seen) == 2 * T
    for t in range(T):
        (r_state, r_kind), (m_state, m_kind) = seen[2 * t], seen[2 * t + 1]
        assert (r_kind, m_kind) == (0, 1) and r_state == m_state, t
    logits.square().mean().backward()
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())


def test_window_under_bf16_autocast_is_finite() -> None:
    cfg = _on_cfg("train.gate_mode=ste")
    model = _model(cfg, torch.float32)
    assert model.cell_read is not None
    tokens, ctx, reset = _window(cfg)
    state = _start(model, tokens.shape[0])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, new_state, _ = model(tokens, ctx, state, reset)
    assert bool(torch.isfinite(logits).all()) and new_state.h.dtype == torch.float32
    logits.float().square().mean().backward()
    for name, p in model.cell_read.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
        assert float(p.grad.abs().sum()) > 0.0, name


# --------------------------------------------------------------------------------------------------
# The compiled autograd window and training
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(COMPILED))
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_equals_eager(name: str) -> None:
    """The read is traced into the compiled block (the window the HSN-T arms train on)."""
    overrides, training = COMPILED[name]
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        *overrides,
        *ON,
    )
    seed_everything(0)
    model = HSNT.from_config(cfg)
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():  # as p28._model(cfg, training=training, scale=0.3)
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.3)
    model.train(training)
    ctx = p28._ctx()
    window = p28._window(model, ctx)
    tokens, cx, state, reset = p28._case_inputs(model, True)
    tol = p28.TOL[(ctx.device.type, "fp32")]
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    got = p28._run(lambda: window(tokens, cx, state, reset))
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    if training:
        grads, ref_grads = p28._grads(model, got), p28._grads(model, ref)
        assert p28._grad_diff(grads, ref_grads) <= tol["grads"]
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        mine = [g for n, g in zip(names, grads, strict=True) if n.startswith("cell_read.")]
        assert len(mine) == 7 and all(g is not None and float(g.abs().sum()) > 0.0 for g in mine)


@pytest.mark.parametrize("extra", [(), SHORT[len(READER) :]])
def test_recipe_with_the_read_trains_and_evaluates(extra: tuple[str, ...], tmp_path: Path) -> None:
    """Thirty TBPTT steps of the small A6 recipe with the candidate's cut on the tiny CPU
    configuration of the trainer tests: finite losses that fall, and an evaluation on the way
    and at the end."""
    steps = 30
    cfg = hsn_config(
        steps,
        *A5,
        CUT_C,
        "model.cand_norm=rms",
        "model.k_select=3",
        "train.soft_until_step=0",
        "train.gate_mode=ste",
        "train.lr=3e-3",
        "train.eval_every=15",
        *ON,
        *extra,
    )
    seed_everything(0)
    model = build_model(cfg)
    assert isinstance(model, HSNT) and model.cell_read is not None
    trainer = TBPTTTrainer(model, cfg, resolve(cfg.hardware), run_dir=tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first - 0.05, (first, last)
    assert [e["step"] for e in history.evals] == [15, 30]
    assert all(math.isfinite(e["loss"]) for e in history.evals)
    assert history.param_report["cell_read"] == cell_read_param_count(cfg.model)
