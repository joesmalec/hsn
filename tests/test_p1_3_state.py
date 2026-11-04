"""P1.3 tests: cell graph state, incremental summary, resets, in-place serving path, snapshots."""

import time
from dataclasses import replace

import pytest
import torch
from hsn.device import seed_everything
from hsn.model.state import (
    CellGraphState,
    IntegrityError,
    SchemaMismatch,
    StateExpired,
    StateMeta,
    StateRestoreError,
    StateSnapshot,
    TenantMismatch,
    VersionMismatch,
)

cuda_param = pytest.param("cuda", marks=pytest.mark.cuda)
mps_param = pytest.param("mps", marks=pytest.mark.mps)
DEVICES = ["cpu", cuda_param, mps_param]
FIELDS = ("h", "q", "age", "pos")

META = StateMeta(
    model_version="hsn-test-r1", state_schema_version=1, tenant_id="tenant-a", expires_at=None
)


def _available(device_type: str) -> bool:
    if device_type == "cuda":
        return bool(torch.cuda.is_available())
    return bool(torch.backends.mps.is_available())


def _device(device_type: str) -> torch.device:
    if device_type != "cpu" and not _available(device_type):
        pytest.skip(f"no {device_type} device on this machine")
    return torch.device(device_type)


def _distinct_idx(B: int, N: int, K: int, device: torch.device) -> torch.Tensor:
    """``[B, K]`` distinct cell indices per row (a random permutation's first K entries)."""
    return torch.rand(B, N, device=device).argsort(dim=1)[:, :K]


def _selected_mask(idx: torch.Tensor, N: int) -> torch.Tensor:
    mask = torch.zeros(idx.shape[0], N, dtype=torch.bool, device=idx.device)
    mask[torch.arange(idx.shape[0], device=idx.device)[:, None], idx] = True
    return mask


def _random_state(
    B: int, N: int, d: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> CellGraphState:
    """A state with non-trivial ``h``, ``q``, ``age`` and ``pos`` and no autograd history."""
    state = CellGraphState.zeros(B, N, d, device, dtype)
    state = state.with_dense(torch.randn(B, N, d, device=device, dtype=dtype))
    for _ in range(3):
        idx = _distinct_idx(B, N, 2, device)
        state = state.with_delta(idx, torch.randn(B, 2, d, device=device, dtype=dtype))
        state = state.tick(_selected_mask(idx, N))
    return state


def _clone(state: CellGraphState) -> CellGraphState:
    """A deep copy that shares no storage (``detach()`` shares storage by design)."""
    return CellGraphState(**{name: getattr(state, name).detach().clone() for name in FIELDS})


# --------------------------------------------------------------------------------------------------
# Tests named by the chunk
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", DEVICES)
def test_p1_3_incremental_q_matches_recompute(device_type: str) -> None:
    device = _device(device_type)
    seed_everything(13)
    B, N, d, K, steps = 4, 32, 64, 4, 256  # ref_a1 shapes, one unroll window of rounds

    state = CellGraphState.zeros(B, N, d, device, torch.float32)
    assert torch.equal(state.q, state.recompute_q())
    state = state.with_dense(torch.randn(B, N, d, device=device))
    assert torch.equal(state.q, state.recompute_q())
    serving = _clone(state)  # driven by the in-place path

    for _ in range(steps):
        idx = _distinct_idx(B, N, K, device)
        h_new = torch.randn(B, K, d, device=device)
        state = state.with_delta(idx, h_new).tick(_selected_mask(idx, N))
        with torch.no_grad():
            serving.apply_delta_(idx, h_new)

    exact = state.recompute_q()
    assert torch.allclose(state.q, exact, atol=1e-5, rtol=0.0)
    assert (state.q - exact).abs().max().item() < 1e-5
    # the functional and the in-place path apply the same arithmetic
    assert torch.equal(serving.h, state.h)
    assert torch.allclose(serving.q, state.q, atol=1e-6, rtol=0.0)
    assert torch.allclose(serving.q, serving.recompute_q(), atol=1e-5, rtol=0.0)
    # ages: 0 at the cells of the last round, counting up elsewhere; one tick per round
    last = _selected_mask(idx, N)
    assert (state.age[last] == 0).all()
    assert (state.age[~last] >= 1).all()
    assert state.age.max().item() <= steps
    assert torch.equal(state.pos, torch.full((B,), steps, dtype=torch.int64, device=device))
    assert torch.equal(serving.pos, torch.zeros(B, dtype=torch.int64, device=device))
    # the periodic resync (P1.7) is a functional replacement of q
    resynced = replace(state, q=state.recompute_q())
    assert torch.equal(resynced.q, exact) and resynced.h is state.h


@pytest.mark.parametrize("device_type", DEVICES)
def test_p1_3_reset_masks_rows_only(device_type: str) -> None:
    device = _device(device_type)
    seed_everything(2)
    B, N, d = 3, 5, 4
    state = _random_state(B, N, d, device)
    assert (state.pos > 0).all() and (state.age > 0).any()
    before = _clone(state)
    mask = torch.tensor([True, False, True], device=device)

    out = state.reset(mask)
    assert out is not state
    for name in FIELDS:
        original, result, copy = getattr(state, name), getattr(out, name), getattr(before, name)
        assert torch.equal(original, copy), name  # the original is untouched (functional)
        assert (result[mask] == 0).all(), name
        assert torch.equal(result[~mask], original[~mask]), name
        assert result.shape == original.shape and result.dtype == original.dtype, name
        assert result.device == original.device, name

    # no mask: every session is zeroed, the layout is preserved
    full = state.reset()
    for name in FIELDS:
        original, result = getattr(state, name), getattr(full, name)
        assert result.count_nonzero().item() == 0, name
        assert result.shape == original.shape and result.dtype == original.dtype, name
        assert result.device == original.device, name
    assert torch.equal(state.h, before.h)

    # gradient flows only to the sessions that are kept
    h0 = torch.randn(B, N, d, device=device, requires_grad=True)
    kept = CellGraphState.zeros(B, N, d, device, torch.float32).with_dense(h0).reset(mask)
    assert kept.h.requires_grad and kept.q.requires_grad
    (grad,) = torch.autograd.grad(kept.h.sum() + kept.q.sum(), h0)
    expected = torch.full_like(h0, 1.0 + 1.0 / N)
    expected[mask] = 0.0
    assert torch.allclose(grad, expected)

    # a NaN in a session being reset cannot leak into the zeros
    poisoned_h = state.h.clone()
    poisoned_h[0] = float("nan")
    poisoned = replace(state, h=poisoned_h).reset(mask)
    assert poisoned.h.isfinite().all() and (poisoned.h[0] == 0).all()
    assert torch.equal(poisoned.h[1], state.h[1])

    with pytest.raises(ValueError, match="torch.bool"):
        state.reset(torch.ones(B, dtype=torch.int64, device=device))
    with pytest.raises(ValueError, match="shape"):
        state.reset(torch.ones(B + 1, dtype=torch.bool, device=device))
    with pytest.raises(ValueError, match="shape"):
        state.reset(torch.ones(B, N, dtype=torch.bool, device=device))


@pytest.mark.parametrize("device_type", DEVICES)
def test_p1_3_snapshot_roundtrip(device_type: str) -> None:
    device = _device(device_type)
    seed_everything(3)
    B, N, d = 2, 6, 8
    state = _random_state(B, N, d, device)

    snap = state.snapshot(META)
    assert isinstance(snap, StateSnapshot) and snap.meta == META
    assert len(snap.digest) == 64 and len(bytes.fromhex(snap.digest)) == 32
    for name in FIELDS:
        copy, live = getattr(snap, name), getattr(state, name)
        assert copy.device.type == "cpu" and not copy.requires_grad, name
        assert copy.dtype == live.dtype and torch.equal(copy, live.cpu()), name
    # the digest is deterministic and covers the meta
    assert state.snapshot(META).digest == snap.digest
    assert state.snapshot(replace(META, tenant_id="tenant-b")).digest != snap.digest
    assert state.snapshot(replace(META, model_version="hsn-test-r2")).digest != snap.digest
    assert state.snapshot(replace(META, expires_at=time.time() + 60)).digest != snap.digest

    restored = CellGraphState.restore(snap, META, device)
    assert restored.device == state.device and restored.dtype == state.dtype
    for name in FIELDS:
        assert torch.equal(getattr(restored, name), getattr(state, name)), name
    # a host-side copy restores to the CPU too (where the serving store keeps it)
    on_cpu = CellGraphState.restore(snap, META, "cpu")
    assert on_cpu.device.type == "cpu" and torch.equal(on_cpu.h, snap.h)

    # the snapshot aliases neither the live state nor a restored one
    idx = _distinct_idx(B, N, 2, device)
    h_new = torch.randn(B, 2, d, device=device)
    with torch.no_grad():
        state.apply_delta_(idx, h_new)
        restored.apply_delta_(idx, h_new)
        on_cpu.apply_delta_(idx.cpu(), h_new.cpu())
    assert not torch.equal(snap.h, state.h.cpu())
    again = CellGraphState.restore(snap, META, device)
    assert torch.equal(again.h, snap.h.to(device)) and torch.equal(again.q, snap.q.to(device))

    # a state with autograd history snapshots the values only
    h0 = torch.randn(B, N, d, device=device, requires_grad=True)
    tracked = CellGraphState.zeros(B, N, d, device, torch.float32).with_dense(h0)
    assert tracked.h.requires_grad
    live_snap = tracked.snapshot(META)
    assert not live_snap.h.requires_grad and not live_snap.q.requires_grad
    assert torch.equal(live_snap.h, h0.detach().cpu())

    # a half-precision state round-trips bit for bit (hashed through a uint8 view)
    half = _random_state(B, N, d, device, torch.bfloat16)
    half_back = CellGraphState.restore(half.snapshot(META), META, device)
    assert half_back.dtype is torch.bfloat16 and torch.equal(half_back.h, half.h)
    assert torch.equal(half_back.q, half.q)

    with pytest.raises(TypeError, match="StateMeta"):
        state.snapshot({"tenant_id": "tenant-a"})  # type: ignore[arg-type]


@pytest.mark.parametrize("device_type", DEVICES)
def test_p1_3_restore_rejects_version_tenant_hash_expiry(device_type: str) -> None:
    device = _device(device_type)
    seed_everything(4)
    B, N, d = 2, 4, 3
    state = _random_state(B, N, d, device)
    snap = state.snapshot(META)
    restore = CellGraphState.restore
    restore(snap, META, device)  # the unmodified snapshot restores

    with pytest.raises(VersionMismatch, match="hsn-test-r2"):
        restore(snap, replace(META, model_version="hsn-test-r2"), device)
    with pytest.raises(SchemaMismatch, match="schema 1, expected 2"):
        restore(snap, replace(META, state_schema_version=2), device)
    with pytest.raises(TenantMismatch, match="tenant-b"):
        restore(snap, replace(META, tenant_id="tenant-b"), device)

    # tampered tensors
    tampered_h = snap.h.clone()
    tampered_h[0, 0, 0] += 1.0
    with pytest.raises(IntegrityError, match="digest"):
        restore(replace(snap, h=tampered_h), META, device)
    tampered_age = snap.age.clone()
    tampered_age[0, 0] += 1
    with pytest.raises(IntegrityError):
        restore(replace(snap, age=tampered_age), META, device)
    tampered_pos = snap.pos.clone()
    tampered_pos[1] = 0
    with pytest.raises(IntegrityError):
        restore(replace(snap, pos=tampered_pos), META, device)
    # re-labelled meta (tenant swapped without re-hashing) even when the caller expects the label
    relabelled = replace(snap, meta=replace(META, tenant_id="tenant-b"))
    with pytest.raises(IntegrityError):
        restore(relabelled, replace(META, tenant_id="tenant-b"), device)
    # an edited digest
    flipped = ("0" if snap.digest[0] != "0" else "1") + snap.digest[1:]
    with pytest.raises(IntegrityError):
        restore(replace(snap, digest=flipped), META, device)
    # another state of the same shape carrying this snapshot's digest
    other = _random_state(B, N, d, device).snapshot(META)
    with pytest.raises(IntegrityError):
        restore(replace(other, digest=snap.digest), META, device)
    restore(other, META, device)  # its own digest is fine

    # expiry is judged from the snapshot's own meta
    expired_meta = replace(META, expires_at=time.time() - 1.0)
    with pytest.raises(StateExpired, match="expired"):
        restore(state.snapshot(expired_meta), expired_meta, device)
    fresh_meta = replace(META, expires_at=time.time() + 3600.0)
    fresh = restore(state.snapshot(fresh_meta), fresh_meta, device)
    assert torch.equal(fresh.h, state.h)
    # expected.expires_at is the caller's own value and is not compared
    restore(state.snapshot(fresh_meta), replace(fresh_meta, expires_at=None), device)
    restore(snap, replace(META, expires_at=time.time() - 1.0), device)  # snap never expires

    for exc in (VersionMismatch, SchemaMismatch, TenantMismatch, IntegrityError, StateExpired):
        assert issubclass(exc, StateRestoreError)

    # matching shapes never suffice: only a StateSnapshot with a StateMeta is considered
    with pytest.raises(TypeError, match="StateSnapshot"):
        restore((snap.h, snap.q, snap.age, snap.pos), META, device)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StateSnapshot"):
        restore(state, META, device)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StateMeta"):
        restore(snap, {"tenant_id": "tenant-a"}, device)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StateMeta"):
        StateSnapshot(snap.h, snap.q, snap.age, snap.pos, None, snap.digest)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hex"):
        replace(snap, digest="abc")
    with pytest.raises(ValueError, match="hex"):
        replace(snap, digest="z" * 64)
    # a snapshot lives on the CPU
    elsewhere = device if device.type != "cpu" else torch.device("meta")
    with pytest.raises(ValueError, match="CPU"):
        replace(snap, **{name: getattr(snap, name).to(elsewhere) for name in FIELDS})


@pytest.mark.parametrize("device_type", DEVICES)
def test_p1_3_inplace_asserts_under_grad(device_type: str) -> None:
    device = _device(device_type)
    seed_everything(5)
    B, N, d, K = 2, 5, 3, 2
    state = _random_state(B, N, d, device)
    before = _clone(state)
    idx = _distinct_idx(B, N, K, device)
    h_new = torch.randn(B, K, d, device=device)

    assert torch.is_grad_enabled()
    with pytest.raises(RuntimeError, match="no_grad"):
        state.apply_delta_(idx, h_new)
    for name in FIELDS:
        assert torch.equal(getattr(state, name), getattr(before, name)), name  # untouched
    with torch.no_grad(), torch.enable_grad():  # grad mode re-enabled inside no_grad
        with pytest.raises(RuntimeError, match="no_grad"):
            state.apply_delta_(idx, h_new)

    with torch.no_grad():
        result = state.apply_delta_(idx, h_new)
    assert result is None
    functional = before.with_delta(idx, h_new)
    assert torch.equal(state.h, functional.h)
    assert torch.allclose(state.q, functional.q, atol=1e-6, rtol=0.0)
    assert torch.equal(state.age, functional.age) and torch.equal(state.pos, functional.pos)

    # a state carrying autograd history refuses even under no_grad; detach() is the remedy
    h0 = torch.randn(B, N, d, device=device, requires_grad=True)
    h0_values = h0.detach().clone()
    tracked = CellGraphState.zeros(B, N, d, device, torch.float32).with_dense(h0)
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="autograd history"):
            tracked.apply_delta_(idx, h_new)
        assert torch.equal(tracked.h, h0_values)
        carried = tracked.detach()  # shares storage with h0, which nothing reads afterwards
        carried.apply_delta_(idx, h_new)
    assert torch.equal(carried.h, before.with_dense(h0_values).with_delta(idx, h_new).h)


@pytest.mark.parametrize("device_type", DEVICES)
def test_p1_3_detach_breaks_graph(device_type: str) -> None:
    device = _device(device_type)
    seed_everything(6)
    B, N, d, K = 2, 4, 3, 2
    h0 = torch.randn(B, N, d, device=device, requires_grad=True)
    idx = _distinct_idx(B, N, K, device)
    selected = _selected_mask(idx, N)
    h_new = torch.randn(B, K, d, device=device, requires_grad=True)
    state = CellGraphState.zeros(B, N, d, device, torch.float32).with_dense(h0)
    state = state.with_delta(idx, h_new).tick(selected)
    assert state.h.requires_grad and state.h.grad_fn is not None
    assert state.q.requires_grad and state.q.grad_fn is not None
    assert not state.age.requires_grad and not state.pos.requires_grad

    carried = state.detach()
    for name in FIELDS:
        tensor = getattr(carried, name)
        assert not tensor.requires_grad and tensor.grad_fn is None, name
        assert torch.equal(tensor, getattr(state, name)), name

    # the original still backpropagates: q = mean(h0) + Σ (h_new − h0[idx]) / N
    g_h0, g_new = torch.autograd.grad(state.q.sum(), (h0, h_new), retain_graph=True)
    expected_h0 = torch.full_like(h0, 1.0 / N)
    expected_h0[selected] = 0.0
    assert torch.allclose(g_h0, expected_h0)
    assert torch.allclose(g_new, torch.full_like(h_new, 1.0 / N))

    # updates on the carried state start a fresh graph that never reaches h0
    h_next = torch.randn(B, K, d, device=device, requires_grad=True)
    nxt = carried.with_delta(idx, h_next).tick(selected)
    assert nxt.q.requires_grad and nxt.h.requires_grad
    g_next, g_old = torch.autograd.grad(nxt.q.sum(), (h_next, h0), allow_unused=True)
    assert g_next is not None and g_old is None
    with pytest.raises(RuntimeError):
        carried.q.sum().backward()


# --------------------------------------------------------------------------------------------------
# Additional coverage
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", DEVICES)
def test_p1_3_with_delta_matches_loop_reference_and_grads(device_type: str) -> None:
    device = _device(device_type)
    seed_everything(7)
    B, N, d, K = 3, 6, 4, 3
    h0 = torch.randn(B, N, d, device=device, requires_grad=True)
    base = CellGraphState.zeros(B, N, d, device, torch.float32).with_dense(h0)
    base = base.tick(torch.zeros(B, N, dtype=torch.bool, device=device))  # every age is 1
    idx = _distinct_idx(B, N, K, device)
    selected = _selected_mask(idx, N)
    h_new = torch.randn(B, K, d, device=device, requires_grad=True)

    out = base.with_delta(idx, h_new)
    ref_h, ref_q, ref_age = base.h.detach().clone(), base.q.detach().clone(), base.age.clone()
    for b in range(B):
        for k in range(K):
            i = int(idx[b, k])
            ref_q[b] += (h_new[b, k].detach() - ref_h[b, i]) / N
            ref_h[b, i] = h_new[b, k].detach()
            ref_age[b, i] = 0
    assert torch.equal(out.h, ref_h)
    assert torch.allclose(out.q, ref_q, atol=1e-6, rtol=0.0)
    assert torch.equal(out.age, ref_age) and torch.equal(out.pos, base.pos)
    assert torch.equal(base.h, h0) and base.age.min().item() == 1  # base untouched

    # d(Σ w·h')/dh0 is w at unselected cells and 0 at selected ones; d/dh_new is w gathered
    w = torch.randn_like(out.h)
    g_h0, g_new = torch.autograd.grad((out.h * w).sum(), (h0, h_new))
    expected_h0 = w.clone()
    expected_h0[selected] = 0.0
    assert torch.equal(g_h0, expected_h0)
    assert torch.equal(g_new, w.gather(1, idx[:, :, None].expand(B, K, d)))

    # tick after the delta: selected cells stay at 0, the others count; pos advances
    ticked = out.tick(selected)
    assert (ticked.age[selected] == 0).all()
    assert torch.equal(ticked.age[~selected], out.age[~selected] + 1)
    assert torch.equal(ticked.pos, out.pos + 1) and ticked.h is out.h and ticked.q is out.q

    # with_dense: the exact mean, every age 0, pos unchanged
    dense_h = torch.randn(B, N, d, device=device)
    dense = ticked.with_dense(dense_h)
    assert dense.h is dense_h and torch.equal(dense.q, dense_h.mean(dim=1))
    assert dense.age.count_nonzero().item() == 0 and torch.equal(dense.pos, ticked.pos)


def test_p1_3_rejects_bad_shapes_and_indices() -> None:
    device = torch.device("cpu")
    B, N, d = 2, 4, 3
    state = CellGraphState.zeros(B, N, d, device, torch.float32)
    assert (state.batch_size, state.n_cells, state.width) == (B, N, d)
    assert state.device == device and state.dtype is torch.float32

    for bad in ((0, N, d), (B, -1, d), (B, N, True), (1.5, N, d)):
        with pytest.raises(ValueError, match=">= 1"):
            CellGraphState.zeros(*bad, device, torch.float32)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="floating"):
        CellGraphState.zeros(B, N, d, device, torch.int64)

    h, q, age, pos = state.h, state.q, state.age, state.pos
    with pytest.raises(ValueError, match=r"\.h must have shape"):
        CellGraphState(h[0], q, age, pos)
    with pytest.raises(ValueError, match="floating"):
        CellGraphState(h.long(), q, age, pos)
    with pytest.raises(ValueError, match=r"\.q must have shape"):
        CellGraphState(h, q[:1], age, pos)
    with pytest.raises(ValueError, match=r"\.q must have dtype"):
        CellGraphState(h, q.double(), age, pos)
    with pytest.raises(ValueError, match=r"\.age must have dtype"):
        CellGraphState(h, q, age.float(), pos)
    with pytest.raises(ValueError, match=r"\.pos must have shape"):
        CellGraphState(h, q, age, pos[:, None])
    with pytest.raises(TypeError, match=r"\.pos must be a torch.Tensor"):
        CellGraphState(h, q, age, [0, 0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one device"):
        CellGraphState(h, q, age, pos.to("meta"))

    good_idx = torch.tensor([[0, 1], [2, 3]])
    h_new = torch.zeros(B, 2, d)
    state.with_delta(good_idx, h_new)
    with pytest.raises(IndexError, match=r"\[0, 4\)"):
        state.with_delta(torch.tensor([[0, 4], [1, 2]]), h_new)
    with pytest.raises(IndexError, match=r"\[0, 4\)"):
        state.with_delta(torch.tensor([[0, -1], [1, 2]]), h_new)
    with pytest.raises(ValueError, match="distinct"):
        state.with_delta(torch.tensor([[1, 1], [0, 2]]), h_new)
    with pytest.raises(ValueError, match="int64"):
        state.with_delta(good_idx.int(), h_new)
    with pytest.raises(ValueError, match="h_new must have shape"):
        state.with_delta(good_idx, h_new[:, :1])
    with pytest.raises(ValueError, match="h_new must have dtype"):
        state.with_delta(good_idx, h_new.double())
    with pytest.raises(ValueError, match="cell_idx must have shape"):
        state.with_delta(good_idx[:1], h_new[:1])
    with pytest.raises(ValueError, match="cell_idx must have shape"):
        state.with_delta(torch.zeros(B, N + 1, dtype=torch.int64), torch.zeros(B, N + 1, d))
    with pytest.raises(TypeError):
        state.with_delta([[0, 1], [2, 3]], h_new)  # type: ignore[arg-type]
    gpus = [t for t in ("cuda", "mps") if _available(t)]
    for gpu in gpus:
        with pytest.raises(ValueError, match="live on cpu"):
            state.with_delta(good_idx.to(gpu), h_new)
        with pytest.raises(ValueError, match="live on cpu"):
            state.with_delta(good_idx, h_new.to(gpu))
        with pytest.raises(ValueError, match="live on cpu"):
            state.tick(torch.zeros(B, N, dtype=torch.bool, device=gpu))
    # K == 1 is fine (nothing to compare for distinctness)
    one = state.with_delta(torch.tensor([[3], [0]]), torch.ones(B, 1, d))
    assert torch.equal(one.q, torch.full((B, d), 1.0 / N))

    with pytest.raises(ValueError, match="h_new must have shape"):
        state.with_dense(torch.zeros(B, N + 1, d))
    with pytest.raises(ValueError, match="h_new must have dtype"):
        state.with_dense(torch.zeros(B, N, d, dtype=torch.float64))
    with pytest.raises(ValueError, match="torch.bool"):
        state.tick(torch.zeros(B, N, dtype=torch.int64))
    with pytest.raises(ValueError, match="shape"):
        state.tick(torch.zeros(B, dtype=torch.bool))

    # StateMeta validation and canonical serialisation
    for bad_kwargs in (
        {"model_version": ""},
        {"model_version": 3},
        {"tenant_id": ""},
        {"state_schema_version": 0},
        {"state_schema_version": True},
        {"state_schema_version": "1"},
        {"expires_at": float("inf")},
        {"expires_at": float("nan")},
        {"expires_at": True},
        {"expires_at": "soon"},
    ):
        with pytest.raises(ValueError, match="StateMeta"):
            replace(META, **bad_kwargs)  # type: ignore[arg-type]
    assert replace(META, expires_at=10).expires_at == 10.0
    assert isinstance(replace(META, expires_at=10).expires_at, float)
    assert META.to_bytes() == (
        b'{"expires_at":null,"model_version":"hsn-test-r1","state_schema_version":1,'
        b'"tenant_id":"tenant-a"}'
    )
