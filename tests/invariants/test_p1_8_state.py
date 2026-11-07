"""P1.8 state invariants (paper §5, §18): the tanh-bounded state over 10,000 random steps, a
disabled cell whose row of ``h`` never changes, a reset that removes every trace of the prior
state, seed determinism, and the tenant / version / schema refusals of a state snapshot driven
through the model API."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import hsn
import pytest
import torch
from hsn.config import RunConfig
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example
from hsn.device import DeviceContext, seed_everything
from hsn.model.hsn import HSN
from hsn.model.state import (
    CellGraphState,
    IntegrityError,
    SchemaMismatch,
    StateMeta,
    StateRestoreError,
    TenantMismatch,
    VersionMismatch,
)
from hsn.train.tbptt import TBPTTStepRecord, TBPTTTrainer
from torch import Tensor

ModelFactory = Callable[..., tuple[RunConfig, HSN]]
IdsFactory = Callable[[HSN, int, int, int], tuple[Tensor, Tensor]]
PerturbFn = Callable[..., None]
StateFactory = Callable[..., CellGraphState]
ConfigFactory = Callable[..., RunConfig]

STEPS_10K = 10_000
RESET_PROBABILITY = 0.01
#: Every parameter of the tiny model times this saturates tanh and the gates (measured: x20, the
#: P1.5 recipe for ref_a1, leaves the tiny model's worst |h| at 0.17; x50 reaches 1.0).
SATURATION_SCALE = 50.0
FIELDS = ("h", "q", "age", "pos")


def _meta(model: HSN, tenant: str) -> StateMeta:
    """The snapshot binding ``model`` would use: the package version as the model version and
    ``model.state_schema_version`` from its configuration."""
    return StateMeta(
        model_version=f"hsn-{hsn.__version__}",
        state_schema_version=model.cfg.state_schema_version,
        tenant_id=tenant,
        expires_at=None,
    )


def _assert_tensors_equal(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert list(a) == list(b)
    for key, value in a.items():
        other = b[key]
        if isinstance(value, Tensor):
            assert torch.equal(value, other), key
        else:
            assert value == other, key


# --------------------------------------------------------------------------------------------------
# |h| <= 1 after 10,000 random steps
# --------------------------------------------------------------------------------------------------


def test_p1_8_state_in_unit_interval(
    device: torch.device, make_model: ModelFactory, expect_same: Any
) -> None:
    """Every coordinate of ``h`` (and of its mean ``q``) stays in ``[−1, 1]`` through 10,000 steps
    of random inputs and random document resets, with every parameter scaled by
    :data:`SATURATION_SCALE` so that the candidates and gates saturate (plan §2.3: each update is
    a coordinatewise convex combination of the previous state and a tanh output). The bound is
    exercised (worst |h| > 0.5), not vacuous, and the summary stays the exact mean of the cells."""
    cfg, model = make_model("tiny", device=device)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(SATURATION_SCALE)
    m = cfg.model
    B = 4
    generator = torch.Generator(device="cpu").manual_seed(1)
    tokens = torch.randint(0, m.vocab_size, (STEPS_10K, B), generator=generator).to(device)
    ctx = torch.randint(0, m.n_context, (STEPS_10K, B), generator=generator).to(device)
    resets = (torch.rand(STEPS_10K, B, generator=generator) < RESET_PROBABILITY).to(device)
    n_resets = int(resets.sum())
    assert n_resets > 0

    state = model.init_state(B)
    worst_h = torch.zeros((), dtype=model.dtype, device=device)
    worst_q = torch.zeros((), dtype=model.dtype, device=device)
    worst_aux = torch.zeros((), dtype=model.dtype, device=device)
    with torch.no_grad():
        for s in range(STEPS_10K):
            state = state.reset(resets[s])
            out = model.step(tokens[s], ctx[s], state)
            state = out.state
            worst_h = torch.maximum(worst_h, state.h.abs().amax())
            worst_q = torch.maximum(worst_q, state.q.abs().amax())
            worst_aux = torch.maximum(worst_aux, out.aux["state_abs_max"])
    h_max, q_max, aux_max = float(worst_h), float(worst_q), float(worst_aux)
    assert h_max <= 1.0 and q_max <= 1.0 and aux_max <= 1.0, (h_max, q_max, aux_max)
    assert h_max > 0.5, h_max  # the bound is exercised
    assert torch.isfinite(state.h).all() and torch.isfinite(state.q).all()
    assert torch.isfinite(out.logits).all()
    expect_same(state.q, state.recompute_q(), device, what="q")
    # pos counts the positions since each session's last reset (STEPS_10K without one)
    steps = torch.arange(STEPS_10K)[:, None].expand(STEPS_10K, B)
    last_reset = torch.where(resets.cpu(), steps, -1).amax(dim=0).clamp(min=0)
    assert torch.equal(state.pos.cpu(), STEPS_10K - last_reset)
    print(
        f"state bound on {device.type}: worst |h| {h_max:.6f}, |q| {q_max:.6f}, "
        f"aux {aux_max:.6f} over {STEPS_10K} steps, {n_resets} resets"
    )


# --------------------------------------------------------------------------------------------------
# model.disabled_cells = [1]: that row of h never changes
# --------------------------------------------------------------------------------------------------


def test_p1_8_disabled_cell_unchanged(
    device: torch.device,
    make_model: ModelFactory,
    perturb: PerturbFn,
    ids: IdsFactory,
    random_state: StateFactory,
    expect_same: Any,
) -> None:
    """With ``model.disabled_cells = [1]`` the row ``h[:, 1]`` is carried through every position
    bit for bit (a copy, exact on every device) while the enabled rows move at every position; its
    age counts every position; the summary stays the exact mean. A document reset zeroes the
    disabled row of the reset session only, and it stays at zero afterwards. The carried row's
    gradient is the identity: nothing of the discarded candidate reaches it."""
    cfg, model = make_model("tiny", device=device, overrides=["model.disabled_cells=[1]"])
    perturb(model, seed=7)
    assert model.cell.disabled_cells == (1,)
    assert cfg.model.disabled_cells == [1]
    disabled, enabled = 1, [0, 2]
    B, T, windows = 3, 25, 8
    state = random_state(model, B, seed=8, ticks=0)
    initial = state.h.clone()
    tokens, ctx = ids(model, B, T * windows, seed=9)
    no_reset = torch.zeros(B, T, dtype=torch.bool, device=device)

    # window by window through forward
    carried = state
    with torch.no_grad():
        for w in range(windows):
            cols = slice(w * T, (w + 1) * T)
            logits, carried, _ = model(tokens[:, cols], ctx[:, cols], carried, no_reset)
            assert torch.isfinite(logits).all()
            assert torch.equal(carried.h[:, disabled], initial[:, disabled]), w
            assert (carried.age[:, disabled] == (w + 1) * T).all()
            assert (carried.age[:, enabled] == 0).all()
            expect_same(carried.q, carried.recompute_q(), device, what="q")
    assert (carried.pos == windows * T).all()
    for i in enabled:
        assert not torch.equal(carried.h[:, i], initial[:, i]), i

    # position by position through step: the row is a copy at every single position
    stepped = state
    with torch.no_grad():
        for t in range(T * windows):
            out = model.step(tokens[:, t], ctx[:, t], stepped)
            assert torch.equal(out.state.h[:, disabled], initial[:, disabled]), t
            for i in enabled:
                assert not torch.equal(out.state.h[:, i], stepped.h[:, i]), (t, i)
            stepped = out.state
    # P1.10a: step vs forward agree to rounding (conftest), not bit for bit
    expect_same.step_vs_forward(stepped.h, carried.h, device, what="h step vs forward")
    assert torch.equal(stepped.age, carried.age) and torch.equal(stepped.pos, carried.pos)

    # a reset of session 0 before position 4: its disabled cell is zero afterwards and stays zero
    reset = no_reset.clone()
    reset[0, 4] = True
    with torch.no_grad():
        _, after, _ = model(tokens[:, :T], ctx[:, :T], carried, reset)
        _, later, _ = model(tokens[:, T : 2 * T], ctx[:, T : 2 * T], after, no_reset)
    for post in (after, later):
        assert (post.h[0, disabled] == 0).all()
        assert torch.equal(post.h[1:, disabled], initial[1:, disabled])
    assert (after.age[0, disabled] == T - 4).all() and (later.age[0, disabled] == 2 * T - 4).all()
    assert (after.age[1:, disabled] == (windows + 1) * T).all()
    assert int(after.pos[0]) == T - 4 and (after.pos[1:] == (windows + 1) * T).all()

    # the gradient of the carried row with respect to the incoming state is the identity
    h0 = carried.h.detach().clone().requires_grad_(True)
    tracked = model.init_state(B).with_dense(h0)
    out2 = model.step(tokens[:, 0], ctx[:, 0], tracked)
    (grad,) = torch.autograd.grad(out2.state.h[:, disabled].sum(), h0)
    expected = torch.zeros_like(h0)
    expected[:, disabled] = 1.0
    assert torch.equal(grad, expected)


# --------------------------------------------------------------------------------------------------
# after a reset, outputs equal a fresh model's outputs
# --------------------------------------------------------------------------------------------------


def test_p1_8_reset_removes_prior_state(
    device: torch.device,
    make_model: ModelFactory,
    ref_model: HSN,
    perturb: PerturbFn,
    ids: IdsFactory,
    random_state: StateFactory,
    expect_same: Any,
) -> None:
    """After three windows from a random state, a reset through ``reset_state`` (the P1.7
    protocol) or through the ``reset`` flag at position 0 makes the next window's logits, states
    and per-position stacks equal those of a fresh copy of the model started from ``init_state``
    (bit for bit on the CPU); the two reset paths agree with each other under the same policy;
    the reset state's snapshot digest is the zero state's; without the reset the prior state
    shows. The tiny model on every device, the reference model on the CPU."""
    models: list[tuple[str, HSN]] = [("tiny", make_model("tiny", device=device)[1])]
    if device.type == "cpu":
        models.append(("ref", ref_model))
    for name, model in models:
        perturb(model, seed=11)
        fresh = copy.deepcopy(model)
        B, T = 3, 6
        tokens, ctx = ids(model, B, 4 * T, seed=12)
        no_reset = torch.zeros(B, T, dtype=torch.bool, device=device)
        at_start = no_reset.clone()
        at_start[:, 0] = True
        window = (tokens[:, 3 * T :], ctx[:, 3 * T :])
        meta = _meta(model, "tenant-a")

        dirty = random_state(model, B, seed=13)
        with torch.no_grad():
            for w in range(3):
                cols = slice(w * T, (w + 1) * T)
                _, dirty, _ = model(tokens[:, cols], ctx[:, cols], dirty, no_reset)
            assert dirty.h.abs().max() > 0.05 and (dirty.pos > 0).all()
            logits_fresh, state_fresh, info_fresh = fresh(*window, fresh.init_state(B), no_reset)
            # (a) the protocol method, every session
            cleared = model.reset_state(dirty, torch.ones(B, dtype=torch.bool, device=device))
            logits_a, state_a, info_a = model(*window, cleared, no_reset)
            # (b) the reset flag at position 0, from the dirty state itself
            logits_b, state_b, info_b = model(*window, dirty, at_start)
            # (c) no reset: the prior state reaches the outputs
            logits_dirty, _, _ = model(*window, dirty, no_reset)
        assert cleared.snapshot(meta).digest == model.init_state(B).snapshot(meta).digest
        assert not cleared.h.any() and not cleared.q.any() and not cleared.age.any()
        assert not cleared.pos.any()
        for logits, state, info in ((logits_a, state_a, info_a), (logits_b, state_b, info_b)):
            expect_same(logits, logits_fresh, device, what=f"{name} logits")
            expect_same(state.h, state_fresh.h, device, what=f"{name} h")
            expect_same(state.q, state_fresh.q, device, what=f"{name} q")
            assert torch.equal(state.age, state_fresh.age)
            assert torch.equal(state.pos, state_fresh.pos) and (state.pos == T).all()
            for key in ("z", "z_hat", "eps", "u"):
                expect_same(info[key], info_fresh[key], device, what=f"{name} {key}")
        # the two reset paths are the same computation from the same zeros
        expect_same(logits_a, logits_b, device, what=f"{name} logits a vs b")
        expect_same(state_a.h, state_b.h, device, what=f"{name} h a vs b")
        expect_same(state_a.q, state_b.q, device, what=f"{name} q a vs b")
        assert torch.equal(state_a.age, state_b.age) and torch.equal(state_a.pos, state_b.pos)
        assert not torch.equal(logits_dirty, logits_fresh)
        # the protocol method refuses a mask of the wrong shape or dtype (errors over silence)
        with pytest.raises(ValueError, match="shape"):
            model.reset_state(dirty, torch.ones(B + 1, dtype=torch.bool, device=device))
        with pytest.raises(ValueError, match="bool"):
            model.reset_state(dirty, torch.ones(B, dtype=torch.int64, device=device))
    print(f"reset on {device.type}: {expect_same.report()}")


# --------------------------------------------------------------------------------------------------
# two runs with the same seed give identical logits (CPU)
# --------------------------------------------------------------------------------------------------


def test_p1_8_determinism_same_seed(
    ref_cfg: RunConfig,
    synthetic_batch: dict[str, Tensor],
    make_trainer_config: ConfigFactory,
    cpu_ctx: DeviceContext,
    train_docs: list[Example],
    dev_docs: list[Example],
    tmp_path: Path,
) -> None:
    """Two reference models built from the same seed have identical weights and produce identical
    logits, states and prediction stacks on a window of real bytes; another seed differs. Two
    five-step training runs from the same seed and configuration record identical losses, parts,
    gradient norms and evaluations and end with identical weights and carried state."""

    def build(seed: int) -> HSN:
        seed_everything(seed)
        return HSN.from_config(ref_cfg)

    first, second, other = build(0), build(0), build(1)
    _assert_tensors_equal(first.state_dict(), second.state_dict())
    batch = synthetic_batch
    B = int(batch["tokens"].shape[0])
    with torch.no_grad():
        runs = [
            model(batch["tokens"], batch["context"], model.init_state(B), batch["reset"])
            for model in (first, second, other)
        ]
    (logits_1, state_1, info_1), (logits_2, state_2, info_2), (logits_3, _, _) = runs
    assert torch.equal(logits_1, logits_2)
    for field in FIELDS:
        assert torch.equal(getattr(state_1, field), getattr(state_2, field)), field
    for key in ("z", "z_hat", "z_target", "eps", "u"):
        assert torch.equal(info_1[key], info_2[key]), key
    assert not torch.equal(logits_1, logits_3)

    cfg = make_trainer_config(
        "train.steps=5",
        "train.eval_every=5",
        "train.ckpt_every=5",
        "train.stats_every=1",
        "train.deterministic=true",
    )
    train_ds, dev_ds = ByteDataset(train_docs), ByteDataset(dev_docs)
    histories = []
    models = []
    carried = []
    for run in ("a", "b"):
        seed_everything(cfg.seed)
        model = HSN.from_config(cfg)
        trainer = TBPTTTrainer(model, cfg, cpu_ctx, run_dir=tmp_path / run)
        histories.append(trainer.fit(train_ds, dev_ds, 5))
        models.append(model)
        assert isinstance(trainer._carried_state, CellGraphState)
        carried.append(trainer._carried_state)
    for rec_a, rec_b in zip(histories[0].steps, histories[1].steps, strict=True):
        assert isinstance(rec_a, TBPTTStepRecord) and isinstance(rec_b, TBPTTStepRecord)
        assert rec_a.loss == rec_b.loss and rec_a.total == rec_b.total, (rec_a, rec_b)
        assert rec_a.parts == rec_b.parts and rec_a.stats == rec_b.stats
        assert rec_a.grad_norm == rec_b.grad_norm and rec_a.accuracy == rec_b.accuracy
    assert histories[0].evals == histories[1].evals
    _assert_tensors_equal(models[0].state_dict(), models[1].state_dict())
    for field in FIELDS:
        assert torch.equal(getattr(carried[0], field), getattr(carried[1], field)), field


# --------------------------------------------------------------------------------------------------
# a snapshot of another tenant (version, schema) is refused; the right one restores
# --------------------------------------------------------------------------------------------------


def test_p1_8_restore_rejects_wrong_tenant(
    device: torch.device,
    make_model: ModelFactory,
    perturb: PerturbFn,
    ids: IdsFactory,
    expect_same: Any,
) -> None:
    """The P1.3 negative test driven through the model: a state ``model.step`` produced is
    snapshotted under its tenant, model version and ``model.state_schema_version``; restoring it
    for another tenant raises ``TenantMismatch`` (another schema or model version, a tampered
    tensor, a relabelled meta or bare tensors are refused too); the matching meta restores a state
    from which ``model.step`` continues exactly as from the live state; and the model itself refuses
    a state of another layout, dtype or device, so matching shapes never give access."""
    cfg, model = make_model("tiny", device=device)
    perturb(model, seed=17)
    B, T = 2, 4
    m = cfg.model
    tokens, ctx = ids(model, B, T + 1, seed=18)
    state = model.init_state(B)
    with torch.no_grad():
        for t in range(T):
            state = model.step(tokens[:, t], ctx[:, t], state).state
    assert (state.pos == T).all() and state.h.abs().max() > 0

    meta = _meta(model, "tenant-a")
    snap = state.snapshot(meta)
    restore = CellGraphState.restore
    with pytest.raises(TenantMismatch, match="tenant-b"):
        restore(snap, replace(meta, tenant_id="tenant-b"), device)
    with pytest.raises(SchemaMismatch):
        restore(snap, replace(meta, state_schema_version=m.state_schema_version + 1), device)
    with pytest.raises(VersionMismatch):
        restore(snap, replace(meta, model_version=f"hsn-{hsn.__version__}-other"), device)
    tampered = snap.h.clone()
    tampered[0, 0, 0] += 0.5
    with pytest.raises(IntegrityError):
        restore(replace(snap, h=tampered), meta, device)
    relabelled = replace(snap, meta=replace(meta, tenant_id="tenant-b"))
    with pytest.raises(IntegrityError):  # the label was swapped without re-signing
        restore(relabelled, replace(meta, tenant_id="tenant-b"), device)
    with pytest.raises(TypeError, match="StateSnapshot"):
        restore((snap.h, snap.q, snap.age, snap.pos), meta, device)  # type: ignore[arg-type]
    for exc in (TenantMismatch, SchemaMismatch, VersionMismatch, IntegrityError):
        assert issubclass(exc, StateRestoreError)

    restored = restore(snap, meta, device)
    assert restored.device == state.device and restored.dtype == state.dtype
    with torch.no_grad():
        live = model.step(tokens[:, T], ctx[:, T], state)
        back = model.step(tokens[:, T], ctx[:, T], restored)
    expect_same(back.logits, live.logits, device, what="logits after restore")
    expect_same(back.state.h, live.state.h, device, what="h after restore")
    expect_same(back.state.q, live.state.q, device, what="q after restore")
    assert torch.equal(back.state.age, live.state.age)
    assert torch.equal(back.state.pos, live.state.pos) and (back.state.pos == T + 1).all()

    # the model's own scope check: another N, another dtype, another device are refused
    other_n = CellGraphState.zeros(B, m.n_cells + 1, m.d, device, model.dtype)
    with pytest.raises(ValueError, match="expects"):
        model.step(tokens[:, 0], ctx[:, 0], other_n)
    other_dtype = torch.float32 if model.dtype is torch.float64 else torch.float16
    with pytest.raises(ValueError, match="dtype"):
        model.step(
            tokens[:, 0], ctx[:, 0], CellGraphState.zeros(B, m.n_cells, m.d, device, other_dtype)
        )
    with pytest.raises(ValueError, match="lives on"):
        model.step(
            tokens[:, 0], ctx[:, 0], CellGraphState.zeros(B, m.n_cells, m.d, "meta", model.dtype)
        )
