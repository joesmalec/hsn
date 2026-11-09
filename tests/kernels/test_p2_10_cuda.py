"""Device tests: opt in with HSN_REQUIRE_CUDA=1, under run_when_idle.py.

Opt-in makes missing devices/artifacts a failure instead of an all-skipped success.
Ordinary CPU pytest collection does not initialize CUDA for these tests.
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from hsn.kernels import cuda_ops as ops
from hsn.kernels import spec, testvectors

pytestmark = pytest.mark.cuda
NAMES = tuple(spec.OPS)


@pytest.fixture(scope="module", autouse=True)
def prepared() -> None:
    if os.environ.get("HSN_REQUIRE_CUDA") != "1":
        pytest.skip("explicit HSN_REQUIRE_CUDA=1 and the idle guard are required")
    assert torch.cuda.is_available(), "CUDA is required on this runner"
    if expected := os.environ.get("HSN_EXPECT_CUDA_CC"):
        assert ".".join(map(str, torch.cuda.get_device_capability(0))) == expected
    assert ops.available(), "build the CUDA artifacts before device tests"
    ops.prepare_cuda(0)


def case(op: str, size: str = "tiny", dtype: torch.dtype = torch.float32):
    vector = testvectors.load_vector(op, size)
    names = tuple(vector.role("input"))
    tensors = {
        name: t.to(device="cuda", dtype=dtype if t.is_floating_point() else t.dtype)
        for name, t in vector.tensors.items()
    }
    for name in names:
        if tensors[name].is_floating_point():
            tensors[name].requires_grad_(True)
    return vector, tensors, names


def forward(op: str, t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if op == "gather_concat":
        return {"v": ops.gather_op(*(t[n] for n in ("u", "eps", "h", "m", "cell_idx")))}
    if op == "grouped_unit_mlp":
        out, hidden = ops.mlp_op(*(t[n] for n in ("v", "unit_idx", "W1", "b1", "W2", "b2")))
        return {"out": out, "hidden_saved": hidden}
    if op == "csr_messages":
        return {
            "m": ops.messages_op(*(t[n] for n in ("h", "indptr", "indices", "a", "V", "cell_idx")))
        }
    h, q, age, old = ops.update_op(
        *(t[n] for n in ("h", "q", "age", "cell_idx", "valid", "c", "b"))
    )
    return {"h_out": h, "q_out": q, "age_out": age, "h_old_saved": old}


def gradients(
    op: str, t: dict[str, torch.Tensor], names: tuple[str, ...], out: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    inputs = [name for name in names if t[name].is_floating_point()]
    outputs = [name for name in out if "grad_" + name in t]
    grads = torch.autograd.grad(
        [out[name] for name in outputs],
        [t[name] for name in inputs],
        [t["grad_" + name] for name in outputs],
    )
    return {"grad_" + name: g for name, g in zip(inputs, grads, strict=True)}


@pytest.mark.parametrize("op", NAMES)
@pytest.mark.parametrize("size", ("tiny", "ref", "odd"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_p2_10_parity_vs_vectors_all_dtypes(op, size, dtype):
    vector, t, names = case(op, size, dtype)
    original = {name: t[name].detach().clone() for name in names}
    out = forward(op, t)
    result = {**out, **gradients(op, t, names, out)}
    tolerance = spec.tolerance(dtype)
    for name, got in result.items():
        expected = vector.tensors[name].to("cuda")
        if got.is_floating_point():
            torch.testing.assert_close(
                got.float(),
                expected,
                atol=tolerance.atol,
                rtol=tolerance.rtol,
                msg=f"{op}/{size}/{dtype}/{name}",
            )
        else:
            assert torch.equal(got, expected), name
    for name, saved in original.items():
        assert torch.equal(t[name], saved), f"functional op mutated {name}"


@pytest.mark.parametrize("op", NAMES)
def test_p2_10_first_dispatch_on_fresh_thread(op):
    _, t, names = case(op)
    expected = forward(op, t)
    expected_grad = gradients(op, t, names, expected)
    torch.cuda.synchronize()

    def first_call():
        # Preparation happened on the main thread. A fresh caller must be able
        # to bind that device's context without reloading modules or kernels.
        out = forward(op, t)
        return out, gradients(op, t, names, out)

    with ThreadPoolExecutor(max_workers=1) as pool:
        out, grads = pool.submit(first_call).result(timeout=30)
    for name in out:
        torch.testing.assert_close(out[name], expected[name])
    for name in grads:
        torch.testing.assert_close(grads[name], expected_grad[name])


@pytest.mark.parametrize("op", NAMES)
def test_p2_10_opcheck_each_op(op):
    _, t, _ = case(op)
    fn, names = {
        "gather_concat": (ops.gather_op, ("u", "eps", "h", "m", "cell_idx")),
        "grouped_unit_mlp": (ops.mlp_op, ("v", "unit_idx", "W1", "b1", "W2", "b2")),
        "csr_messages": (ops.messages_op, ("h", "indptr", "indices", "a", "V", "cell_idx")),
        "gated_delta_update": (ops.update_op, ("h", "q", "age", "cell_idx", "valid", "c", "b")),
    }[op]
    torch.library.opcheck(fn, tuple(t[n] for n in names))


@pytest.mark.parametrize("op", NAMES)
def test_p2_10_gradcheck_fp32_tiny(op):
    _, t, names = case(op)
    floating = tuple(n for n in names if t[n].is_floating_point())

    def fn(*inputs):
        data = {**t, **dict(zip(floating, inputs, strict=True))}
        return tuple(v for k, v in forward(op, data).items() if "grad_" + k in t)

    # ABI has no f64. This step/tolerance is chosen for float32 finite differences;
    # the independent f64 Rust oracle remains the high-precision derivative reference.
    assert torch.autograd.gradcheck(
        fn, tuple(t[n] for n in floating), eps=1e-3, atol=5e-3, rtol=5e-3, fast_mode=True
    )


@pytest.mark.parametrize("op", NAMES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_p2_10_deterministic_backward_when_flagged(op, dtype):
    _, t, names = case(op, "odd", dtype)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        expected = gradients(op, t, names, forward(op, t))
        for _ in range(3):
            result = gradients(op, t, names, forward(op, t))
            assert all(torch.equal(expected[n], result[n]) for n in expected)
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("op", NAMES)
def test_p2_10_compile_forward_backward(op):
    _, t, names = case(op)

    def fn(*inputs):
        return forward(op, dict(zip(names, inputs, strict=True)))

    compiled = torch.compile(fn, fullgraph=True)
    out = compiled(*(t[n] for n in names))
    got = gradients(op, t, names, out)
    expected = gradients(op, t, names, forward(op, t))
    for name in got:
        torch.testing.assert_close(got[name], expected[name])


def test_p2_10_update_summary_coupling_and_padding():
    _, t, _ = case("gated_delta_update")
    t["valid"].zero_()
    t["valid"][:, 0] = 1
    t["cell_idx"][:, 0] = 0
    t["cell_idx"][:, 1] = -(2**31)
    with torch.no_grad():
        t["b"][:, 1] = float("nan")
        t["c"][:, 1] = float("nan")
    out = forward("gated_delta_update", t)
    dh, dc, db = torch.autograd.grad(out["q_out"].sum(), (t["h"], t["c"], t["b"]))
    n = t["h"].shape[1]
    torch.testing.assert_close(dh[:, 0], -t["b"][:, 0] / n)
    torch.testing.assert_close(dc[:, 0], t["b"][:, 0] / n)
    assert torch.count_nonzero(dc[:, 1]) == torch.count_nonzero(db[:, 1]) == 0
    assert torch.isfinite(out["h_out"]).all() and torch.isfinite(out["q_out"]).all()


def test_p2_10_empty_csr_and_repeated_gather():
    h = torch.randn(2, 3, 4, device="cuda", requires_grad=True)
    idx = torch.zeros((2, 2), dtype=torch.int32, device="cuda")
    indptr = torch.zeros(4, dtype=torch.int32, device="cuda")
    edges = torch.empty(0, dtype=torch.int32, device="cuda")
    a = torch.empty(0, device="cuda", requires_grad=True)
    V = torch.eye(4, device="cuda", requires_grad=True)
    out = ops.messages_op(h, indptr, edges, a, V, idx)
    assert torch.count_nonzero(out) == 0
    gh, ga, gv = torch.autograd.grad(out.sum(), (h, a, V))
    assert torch.count_nonzero(gh) == torch.count_nonzero(gv) == ga.numel() == 0
    u = torch.randn(2, 4, device="cuda")
    gathered = ops.gather_op(u, u, h, out.detach(), idx)
    (gh,) = torch.autograd.grad(gathered.sum(), (h,))
    assert torch.equal(gh[:, 0], torch.full_like(gh[:, 0], 2))
    assert torch.count_nonzero(gh[:, 1:]) == 0


@pytest.mark.parametrize("op", NAMES)
def test_p2_10_graph_replay_routing_and_stream(op):
    _, t, names = case(op)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            out = forward(op, t)
            gradients(op, t, names, out)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = forward(op, t)
        grads = gradients(op, t, names, out)
    if op == "grouped_unit_mlp":
        t["unit_idx"].copy_((t["unit_idx"] + 1) % t["W1"].shape[0])
    elif op == "gated_delta_update":
        with torch.no_grad():
            t["c"].add_(0.01)
    else:
        t["cell_idx"].copy_(t["cell_idx"].flip(1))
    graph.replay()
    eager = forward(op, t)
    expected = gradients(op, t, names, eager)
    for name in out:
        torch.testing.assert_close(out[name], eager[name])
    for name in grads:
        torch.testing.assert_close(grads[name], expected[name])


def test_p2_10_no_nvrtc_symbols():
    libraries = list(ops.BUILD.glob("hsn_cuda_py*.so"))
    assert len(libraries) == 1
    symbols = subprocess.check_output(["nm", "-D", str(libraries[0])], text=True)
    assert "nvrtc" not in symbols.lower()
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    listing = subprocess.check_output(
        [str(cuda / "bin/cuobjdump"), "--list-elf", str(ops.BUILD / "hsn_ops.fatbin")], text=True
    )
    for arch in ("80", "86", "89", "90", "120", "121"):
        assert f"sm_{arch}" in listing
    assert "sm_120a" not in listing


def test_p2_10_inplace_mutation_contract():
    _, t, _ = case("gated_delta_update")
    names = ("h", "q", "age", "cell_idx", "valid", "c", "b")
    values = tuple(t[n].detach().clone() for n in names)
    torch.library.opcheck(ops.update_inplace, values)


def test_p2_10_multistep_state_gradients_and_wide_ages():
    from hsn.kernels import native
    from hsn.model.state import CellGraphState

    _, data, _ = case("gated_delta_update")
    results = []
    for backend in (native, ops):
        t = {n: x.detach().clone().requires_grad_(x.is_floating_point()) for n, x in data.items()}
        h, q, c, b = (t[n] for n in ("h", "q", "c", "b"))
        age = t["age"].long() + 2**40
        state = CellGraphState(h, q, age, torch.zeros(h.shape[0], dtype=torch.int64, device="cuda"))
        first = backend.gated_delta_update(state, t["cell_idx"].long(), c, b, t["valid"].bool())
        second = backend.gated_delta_update(first, t["cell_idx"].long(), c, b, t["valid"].bool())
        grad = torch.autograd.grad((second.h.sum() + second.q.sum()), (h, q, c, b))
        assert torch.equal(state.h, data["h"])
        assert torch.equal(state.age, data["age"].long() + 2**40)
        results.append((second.h, second.q, second.age, *grad))
    for actual, expected in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
