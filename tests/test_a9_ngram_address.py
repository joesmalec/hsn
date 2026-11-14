"""Amendment A9 (``docs/amendments/A9_ngram_address.md``): ``model.address_ngram``.

With ``model.address_ngram = n > 1`` the scorer has, beside A5's ``a_cur`` and ``a_prev``, the
address of the last ``m`` bytes (``a_cur<m>``) and of the ``m`` bytes before the current one
(``a_prev<m>``) for ``m = 2 .. n``. These tests check that the key is opt-in and leaves A5's
buffers, columns and hashes alone, the features against a direct computation (one address for a
key at its fact's value and at its query, unrelated addresses for keys that share a byte), the
window against stepping with the full history, the documented limit at a window's start, the
compiled window, the closed forms, the refusals, that the recipe trains with it, and the
delayed generator's ``key_len``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import test_p2_8_compile as p28
import torch
from hsn.config import config_hash, load_config, split_window_scope_error
from hsn.data.synthetic import (
    BOS,
    EOS,
    QRY,
    SEP,
    TaskSpec,
    check_example,
    generate,
    load_dataset_spec,
)
from hsn.model.hsn import HSN
from hsn.train.memory_plan import expected_param_count, macs_per_token
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, hsn_config, hsn_trainer
from test_a6_feedback_detach import A5, ALL, _cfg, _model, _start, _window
from test_p2_8_compile import _release_compiled_state, small_blocks  # noqa: F401
from torch import Tensor

pytestmark = pytest.mark.usefixtures("restore_determinism")

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
RECIPE_A6 = REPO / "configs" / "experiments" / "hsn_recipe_a6.yaml"
V1 = REPO / "configs" / "data" / "synthetic_v1.yaml"


def _features(model: HSN, lags: list[int]) -> Tensor:
    """The address features ``[N, 2n]`` at a position whose bytes are ``lags`` (``x_t`` first)."""
    n = model.cfg.address_ngram
    t = torch.tensor(lags + [BOS] * (n + 1 - len(lags)))
    return model.scorer.address_features(t[0:1], t[1:2], t[None, 2:] if n > 1 else None)[0]


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_key_is_opt_in_and_the_default_keeps_its_hash() -> None:
    base = load_config(RECIPE_A6)
    assert base.model.address_ngram == 1
    assert config_hash(load_config(RECIPE_A6, ["model.address_ngram=1"])) == config_hash(base)
    assert config_hash(load_config(RECIPE_A6, ["model.address_ngram=2"])) != config_hash(base)


def test_refusals() -> None:
    with pytest.raises(ValueError, match="requires model.key_address"):
        load_config(REF_A1, ["model.routing=topk", "model.address_ngram=2"])
    with pytest.raises(ValueError):
        load_config(RECIPE_A6, ["model.address_ngram=4"])
    # P2.10c: the split and fused windows take the wider address scalars (they refused the key
    # as untested before; tests/test_p2_10c_state_gradient.py and, on the card,
    # tests/kernels/test_p2_10c_window.py hold them to the autograd window)
    cfg = load_config(RECIPE_A6, ["train.gate_mode=ste", "model.address_ngram=2"])
    assert split_window_scope_error(cfg) is None
    fused = load_config(
        RECIPE_A6, ["train.gate_mode=ste", "model.address_ngram=2", "hardware.window=fused"]
    )
    assert fused.hardware.window == "fused"


# --------------------------------------------------------------------------------------------------
# The scorer
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3])
def test_a5_buffers_and_columns_are_unchanged(n: int) -> None:
    torch.manual_seed(0)
    plain = HSN.from_config(_cfg(*A5))
    torch.manual_seed(0)
    wide = HSN.from_config(_cfg(*A5, f"model.address_ngram={n}"))
    assert torch.equal(plain.scorer.byte_codes, wide.scorer.byte_codes)
    assert torch.equal(plain.scorer.cell_addresses, wide.scorer.cell_addresses)
    old = plain.scorer.net.weight
    new = wide.scorer.net.weight
    assert new.shape[1] == old.shape[1] + 2 * (n - 1) == wide.scorer.in_features
    assert torch.equal(new[:, : old.shape[1]], old)
    assert tuple(wide.scorer.ngram_codes.shape) == (n - 1, wide.cfg.vocab_size, wide.cfg.d)
    names = list(wide.scorer.feature_slices)
    assert names[-2 * n :][:2] == ["a_cur", "a_prev"]
    assert names[-2:] == [f"a_cur{n}", f"a_prev{n}"]
    theirs = dict(wide.named_parameters())
    for name, p in plain.named_parameters():
        if name != "router.scorer.net.weight":
            assert torch.equal(p, theirs[name]), name


@pytest.mark.parametrize("n", [2, 3])
def test_features_match_a_direct_computation(n: int) -> None:
    model = HSN.from_config(_cfg(*A5, f"model.address_ngram={n}"))
    sc = model.scorer
    d = model.cfg.d
    lags = [70, 80, 90, 100][: n + 1]  # x_t, x_{t-1}, ...
    got = _features(model, lags)
    A = sc.cell_addresses
    tables = [sc.byte_codes, *sc.ngram_codes]
    want = [sc.byte_codes[lags[0]] @ A.T, sc.byte_codes[lags[1]] @ A.T]
    for m in range(2, n + 1):
        for back in (0, 1):
            gram = [lags[back + m - 1 - i] for i in range(m)]  # the m bytes, oldest first
            code = torch.ones(d)
            for i, byte in enumerate(gram):
                code = code * tables[i][byte]
            want.append((code * d ** ((m - 1) / 2)) @ A.T)
    torch.testing.assert_close(got, torch.stack(want, dim=-1) * math.sqrt(d), rtol=1e-5, atol=1e-6)
    # A5's two columns are those of the model without the key
    plain = HSN.from_config(_cfg(*A5))
    torch.testing.assert_close(got[:, :2], _features(plain, lags[:2]))


def test_a_key_has_one_address_at_its_fact_and_at_its_query() -> None:
    model = HSN.from_config(_cfg(*A5, "model.address_ngram=3", "model.d=64", "model.n_cells=32"))
    k1, k2, k3, value = 70, 71, 72, 150
    # the fact  k1 k2 v : at the value the two bytes before it are the key
    write2 = _features(model, [value, k2, k1])[:, 3]  # a_prev2
    read2 = _features(model, [k2, k1, QRY])[:, 2]  # a_cur2 at the key's last byte
    assert torch.equal(write2, read2)
    write3 = _features(model, [value, k3, k2, k1])[:, 5]  # a_prev3
    read3 = _features(model, [k3, k2, k1, QRY])[:, 4]  # a_cur3
    assert torch.equal(write3, read3)
    # keys that share a byte have unrelated two-byte addresses, unlike their one-byte ones
    other = _features(model, [value, k2, 90])[:, 3]  # (90, k2): same last byte
    swapped = _features(model, [value, k1, k2])[:, 3]  # (k2, k1): the same bytes, other order
    for unrelated in (other, swapped):
        cos = torch.nn.functional.cosine_similarity(write2, unrelated, dim=0)
        assert abs(float(cos)) < 0.6  # 32 cells: a loose bound on a chance correlation
    same_last_byte = _features(model, [value, k2, 90])[:, 1]
    assert torch.equal(_features(model, [value, k2, k1])[:, 1], same_last_byte)  # A5 cannot tell
    # scale: an n-gram feature has A5's variance (about 1 per cell)
    sample = torch.stack(
        [
            _features(model, [a, b, c])[:, 2:6]
            for a in (65, 66, 67)
            for b in (75, 76)
            for c in (85, 86)
        ]
    )
    assert 0.5 < float(sample.std()) < 2.0


# --------------------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3])
def test_param_report_and_macs_match_the_closed_forms(n: int) -> None:
    for cfg in (
        _cfg(*A5, f"model.address_ngram={n}"),
        load_config(RECIPE_A6, [f"model.address_ngram={n}"]),
    ):
        model = HSN.from_config(cfg)
        report, expected = model.param_report(), expected_param_count(cfg.model)
        for row, count in expected.items():
            assert report[row] == count, row
    plain = load_config(RECIPE_A6).model
    assert expected["total"] - expected_param_count(plain)["total"] == 2 * (n - 1) * plain.n_units
    extra = macs_per_token(cfg.model)["scorer"] - macs_per_token(plain)["scorer"]
    assert extra == plain.n_cells * 2 * (n - 1) * (plain.n_units + plain.d)


@pytest.mark.parametrize("n", [2, 3])
def test_window_matches_stepping_with_the_history(n: int) -> None:
    cfg = _cfg(*A5, f"model.address_ngram={n}")
    model = _model(cfg).eval()
    tokens, ctx, reset = _window(cfg)
    reset[2, 4] = True
    B, T = tokens.shape
    with torch.no_grad():
        logits, final, _ = model(tokens, ctx, _start(model, B), reset)
        state, steps = _start(model, B), []
        history = torch.full((B, n), BOS)
        for t in range(T):
            state = state.reset(reset[:, t])
            out = model.step(tokens[:, t], ctx[:, t], state, prev_tokens=history)
            state = out.state
            history = torch.cat([tokens[:, t : t + 1], history[:, :-1]], dim=1)
            steps.append(out.logits)
    torch.testing.assert_close(torch.stack(steps, dim=1), logits, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(state.h, final.h, rtol=1e-9, atol=1e-11)


def test_window_start_reads_bos_beyond_the_carried_byte() -> None:
    """A window that continues a document knows one earlier byte (A5's ``prev_tokens``): the
    two-byte features of its first position read BOS for the byte before that one, and every
    later position is exact."""
    cfg = _cfg(*A5, "model.address_ngram=2")
    model = _model(cfg).eval()
    tokens, ctx, _ = _window(cfg)
    B, T = tokens.shape
    no_reset = torch.zeros(B, T, dtype=torch.bool)
    start = _start(model, B)
    start = type(start)._make(
        h=start.h, q=start.q, age=start.age, pos=start.pos + 5
    )  # mid-document
    carried = torch.full((B,), 77)
    pro = model._window_prologue(
        tokens, ctx, start, no_reset, "sparse", reset_first=False, prev_tokens=carried
    )
    got = pro.scalars[..., -4:]  # a_cur, a_prev, a_cur2, a_prev2
    sc = model.scorer
    older = torch.full((B, T), BOS)
    older[:, 1] = carried
    older[:, 2:] = tokens[:, :-2]
    previous = torch.cat([carried[:, None], tokens[:, :-1]], dim=1)
    want = sc.address_features(tokens, previous, older[..., None])
    torch.testing.assert_close(got, want.to(got.dtype))


COMPILED = {"ste": (("train.gate_mode=ste",), True), "soft": ((), True), "eval": ((), False)}


@pytest.mark.parametrize("name", list(COMPILED))
@pytest.mark.usefixtures("small_blocks")
def test_compiled_window_equals_eager(name: str) -> None:
    overrides, training = COMPILED[name]
    cfg = p28._cfg(
        "model.routing=topk",
        "model.key_address=true",
        "model.selected_readout=true",
        "model.address_ngram=2",
        *overrides,
    )
    model = p28._model(cfg, training=training, scale=0.3)
    ctx = p28._ctx()
    window = p28._window(model, ctx)
    tokens, cx, state, reset = p28._case_inputs(model, True)
    ref = p28._run(lambda: model(tokens, cx, state, reset))
    got = p28._run(lambda: window(tokens, cx, state, reset))
    tol = p28.TOL[(ctx.device.type, "fp32")]
    diff = p28._max_diff(got, ref)
    assert diff["exact"] == 0.0 and diff["values"] <= tol["values"], diff
    if training:
        assert p28._grad_diff(p28._grads(model, got), p28._grads(model, ref)) <= tol["grads"]


def test_recipe_with_the_ngram_address_trains(tmp_path: Path) -> None:
    steps = 30
    cfg = hsn_config(
        steps,
        *A5,
        ALL,
        "model.cand_norm=rms",
        "train.soft_until_step=5",
        "train.lr=3e-3",
        "model.address_ngram=2",
    )
    trainer = hsn_trainer(cfg, tmp_path / "run")
    history = trainer.fit(*datasets(), steps)
    losses = [s.loss for s in history.steps]
    assert len(losses) == steps and all(math.isfinite(x) for x in losses)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.05


# --------------------------------------------------------------------------------------------------
# The delayed generator's key_len
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key_len", [2, 3])
def test_delayed_documents_with_longer_keys(key_len: int) -> None:
    base = load_dataset_spec(V1).task("delayed", "in_range")
    spec = TaskSpec.model_validate(
        {**base.model_dump(), "seed": 99, "params": {**base.params, "key_len": key_len}}
    )
    params = spec.typed_params
    shared = 0
    for split in ("train", "dev", "test"):
        lo, hi = params.keys[split]  # type: ignore[attr-defined]
        for index in range(150):
            example = generate(spec, split, index)
            check_example(spec, example)  # the reference solver agrees with the generator
            t = example.tokens.tolist()
            assert t[0] == BOS and t[-1] == EOS and example.length <= spec.unroll
            sep = t.index(SEP)
            values = [i for i in range(sep) if 144 <= t[i] < 192]
            keys = [tuple(t[i - key_len : i]) for i in values]
            assert len(set(keys)) == len(keys) and all(lo <= b < hi for key in keys for b in key)
            shared += len({key[-1] for key in keys}) < len(keys)
            for position, dep in zip(
                np.flatnonzero(example.loss_mask), example.meta["deps"], strict=True
            ):
                key = tuple(t[position - key_len + 1 : position + 1])
                assert t[position - key_len] == QRY and key in keys
                value_at = values[keys.index(key)]
                assert example.targets[position] == t[value_at] and dep == position - value_at
    assert shared > 100  # keys of one document share bytes: no single byte identifies a key


def test_key_len_one_is_synthetic_v1() -> None:
    base = load_dataset_spec(V1).task("delayed", "in_range")
    named = TaskSpec.model_validate({**base.model_dump(), "params": {**base.params, "key_len": 1}})
    for index in range(50):
        a, b = generate(base, "train", index), generate(named, "train", index)
        assert np.array_equal(a.tokens, b.tokens) and np.array_equal(a.loss_mask, b.loss_mask)
