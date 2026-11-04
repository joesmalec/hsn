"""P1.4 tests: fixed sparse CSR graph, learnable edge weights, message passing, topology guard."""

from pathlib import Path

import pytest
import torch
from hsn.config import ModelConfig, load_config
from hsn.device import seed_everything
from hsn.model.graph import CellGraph
from torch import Tensor, nn

cuda_param = pytest.param("cuda", marks=pytest.mark.cuda)
mps_param = pytest.param("mps", marks=pytest.mark.mps)
DEVICES = ["cpu", cuda_param, mps_param]
NORMS = ["softmax", "sigmoid"]
CSR = ("indptr", "indices", "is_random")

REPO = Path(__file__).resolve().parents[1]
REF_A1: ModelConfig = load_config(REPO / "configs" / "ref_a1.yaml").model  # N=32, D_g=4, d=64, K=4
TINY: ModelConfig = load_config(REPO / "configs" / "tiny_gradcheck.yaml").model  # N=3, D_g=2, d=4


def _available(device_type: str) -> bool:
    if device_type == "cuda":
        return bool(torch.cuda.is_available())
    return bool(torch.backends.mps.is_available())


def _device(device_type: str) -> torch.device:
    if device_type != "cpu" and not _available(device_type):
        pytest.skip(f"no {device_type} device on this machine")
    return torch.device(device_type)


def _graph(
    cfg: ModelConfig,
    device: torch.device | str = "cpu",
    seed: int = 0,
    random_edges: int | None = None,
    edge_norm: str | None = None,
) -> CellGraph:
    """A graph with the config's shapes; ``random_edges`` / ``edge_norm`` override the config."""
    graph = CellGraph(
        cfg.n_cells,
        cfg.in_degree,
        seed,
        random_edges=cfg.random_edges if random_edges is None else random_edges,
        edge_norm=cfg.edge_norm if edge_norm is None else edge_norm,  # type: ignore[arg-type]
    )
    return graph.to(device)


def _rows(graph: CellGraph) -> Tensor:
    counts = graph.indptr[1:] - graph.indptr[:-1]
    return torch.arange(graph.n_cells, device=counts.device).repeat_interleave(counts)


def _oracle(graph: CellGraph, h: Tensor, V: nn.Linear) -> Tensor:
    """``m_i = Σ_j a_ij (V h_j)`` through the dense adjacency (rows targets, columns sources)."""
    return torch.einsum("ij,bjd->bid", graph.dense_adjacency().to(h.dtype), V(h))


def _csr(sources: list[list[int]], random: list[list[bool]] | None = None) -> tuple[Tensor, ...]:
    """CSR tensors from per-row source lists (not sorted or validated here on purpose)."""
    counts = [len(row) for row in sources]
    indptr = torch.tensor([0] + counts, dtype=torch.int64).cumsum(0)
    indices = torch.tensor([j for row in sources for j in row], dtype=torch.int64)
    flags = random if random is not None else [[False] * n for n in counts]
    is_random = torch.tensor([f for row in flags for f in row], dtype=torch.bool)
    return indptr, indices, is_random


def _clone_csr(graph: CellGraph) -> dict[str, Tensor]:
    return {name: getattr(graph, name).clone() for name in CSR}


def _randomise_logits(graph: CellGraph, seed: int) -> None:
    with torch.no_grad():
        generator = torch.Generator(device="cpu").manual_seed(seed)
        graph.edge_logit.copy_(torch.randn(graph.n_edges, generator=generator))


# --------------------------------------------------------------------------------------------------
# Tests named by the chunk
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cfg", "random_edges"), [(REF_A1, 0), (REF_A1, 2), (REF_A1, 27), (TINY, 0)]
)
def test_p1_4_in_degree_exact(cfg: ModelConfig, random_edges: int) -> None:
    N, D = cfg.n_cells, cfg.in_degree
    for seed in range(5):
        graph = _graph(cfg, seed=seed, random_edges=random_edges)
        E = N * (D + random_edges)
        assert graph.n_edges == E and graph.max_in_degree == D + random_edges
        assert torch.equal(
            graph.indptr, torch.arange(N + 1, dtype=torch.int64) * (D + random_edges)
        )
        assert graph.indices.dtype == torch.int64 and graph.indices.shape == (E,)
        assert graph.is_random.dtype == torch.bool and graph.is_random.shape == (E,)
        assert isinstance(graph.edge_logit, nn.Parameter) and graph.edge_logit.shape == (E,)
        assert graph.edge_logit.requires_grad and graph.edge_logit.count_nonzero().item() == 0
        assert int(graph.is_random.sum()) == N * random_edges
        for i in range(N):
            lo, hi = int(graph.indptr[i]), int(graph.indptr[i + 1])
            sources = graph.indices[lo:hi]
            random = graph.is_random[lo:hi]
            assert int((~random).sum()) == D, f"cell {i}: fixed in-degree is not exactly D_g"
            assert int(random.sum()) == random_edges
            assert len(set(sources.tolist())) == hi - lo, "sources are distinct"
            assert (sources[1:] > sources[:-1]).all(), "sorted within the CSR row"
            assert (sources >= 0).all() and (sources < N).all()
    # the dense adjacency has exactly D_g + random_edges non-zeros per row with uniform weights
    graph = _graph(cfg, random_edges=random_edges)
    A = graph.dense_adjacency()
    assert A.shape == (N, N) and torch.equal((A > 0).sum(dim=1), torch.full((N,), D + random_edges))
    if D + random_edges:
        assert torch.allclose(A[A > 0], torch.full((E,), 1.0 / (D + random_edges)))


def test_p1_4_in_degree_zero_is_an_edgeless_graph() -> None:
    graph = CellGraph(REF_A1.n_cells, 0, seed=3)  # model.in_degree=0: propagation removed
    N, d, B = REF_A1.n_cells, REF_A1.d, 2
    assert graph.n_edges == 0 and graph.max_in_degree == 0
    assert graph.edge_logit.shape == (0,) and graph.edge_weights().shape == (0,)
    assert torch.equal(graph.indptr, torch.zeros(N + 1, dtype=torch.int64))
    assert torch.equal(graph.dense_adjacency(), torch.zeros(N, N))
    assert graph.l1().item() == 0.0
    V = nn.Linear(d, d)
    h = torch.randn(B, N, d)
    assert torch.equal(graph.messages_dense(h, V), torch.zeros(B, N, d))
    idx = torch.tensor([[0, 5], [31, 31]])
    assert torch.equal(graph.messages_selected(h, V, idx), torch.zeros(B, 2, d))


def test_p1_4_constructor_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="exceed"):
        CellGraph(REF_A1.n_cells, REF_A1.n_cells, seed=0)  # D_g >= N
    with pytest.raises(ValueError, match="exceed"):
        CellGraph(REF_A1.n_cells, REF_A1.in_degree, seed=0, random_edges=28)  # 4 + 28 > 31
    with pytest.raises(ValueError, match="edge_norm"):
        CellGraph(REF_A1.n_cells, REF_A1.in_degree, seed=0, edge_norm="relu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        CellGraph(REF_A1.n_cells, REF_A1.in_degree, seed=-1)
    with pytest.raises(ValueError, match="n_cells"):
        CellGraph(0, 0, seed=0)
    with pytest.raises(ValueError, match=">= 0"):
        CellGraph(4, -1, seed=0)
    with pytest.raises(TypeError):
        CellGraph(4, True, seed=0)
    with pytest.raises(TypeError):
        CellGraph(4.0, 2, seed=0)  # type: ignore[arg-type]
    CellGraph(1, 0, seed=0)  # a single cell has no possible source


@pytest.mark.parametrize(
    ("n_cells", "in_degree", "random_edges"),
    [(32, 4, 0), (32, 4, 3), (8, 7, 0), (3, 2, 0), (5, 1, 3), (2, 1, 0)],
)
def test_p1_4_no_self_loops(n_cells: int, in_degree: int, random_edges: int) -> None:
    for seed in range(25):
        graph = CellGraph(n_cells, in_degree, seed, random_edges=random_edges)
        rows = _rows(graph)
        assert rows.shape == graph.indices.shape
        assert (graph.indices != rows).all(), f"seed {seed}: a cell reads itself"
        assert torch.equal(graph.dense_adjacency().diagonal(), torch.zeros(n_cells))
    # a topology with a self loop is refused
    graph = CellGraph(4, 2, seed=0)
    with pytest.raises(ValueError, match="self loop"):
        graph.set_topology(*_csr([[1, 2], [0, 1], [0, 1], [1, 2]]))


@pytest.mark.parametrize("edge_norm", NORMS)
@pytest.mark.parametrize("device_type", DEVICES)
def test_p1_4_selected_equals_dense_gathered(device_type: str, edge_norm: str) -> None:
    device = _device(device_type)
    seed_everything(11)
    cfg = REF_A1
    B, N, d, K = 4, cfg.n_cells, cfg.d, cfg.k_select
    graph = _graph(cfg, device, seed=1, random_edges=2, edge_norm=edge_norm)
    _randomise_logits(graph, seed=21)  # non-uniform weights, so the gather order matters
    V = nn.Linear(d, d, bias=False).to(device)
    h_prev = torch.randn(B, N, d, device=device)

    dense = graph.messages_dense(h_prev, V)
    assert dense.shape == (B, N, d) and dense.dtype == h_prev.dtype
    assert torch.allclose(dense, _oracle(graph, h_prev, V), atol=1e-5, rtol=0.0)

    # distinct selections, repeated cells, and every cell in a shuffled order
    selections = [
        torch.rand(B, N, device=device).argsort(dim=1)[:, :K],
        torch.randint(0, N, (B, K), device=device),
        torch.tensor([[7, 7, 7, 7]] * B, device=device),
        torch.rand(B, N, device=device).argsort(dim=1),
    ]
    for cell_idx in selections:
        k = cell_idx.shape[1]
        selected = graph.messages_selected(h_prev, V, cell_idx)
        assert selected.shape == (B, k, d)
        gathered = dense.gather(1, cell_idx[:, :, None].expand(B, k, d))
        assert torch.allclose(selected, gathered, atol=1e-6, rtol=0.0)
        assert (selected - gathered).abs().max().item() < 1e-6

    # V is applied before the weighting: with a bias, m_i = Σ_j a_ij (W h_j + b), which differs
    # from V(Σ_j a_ij h_j) whenever the weights of a row do not sum to one (sigmoid)
    Vb = nn.Linear(d, d, bias=True).to(device)
    with torch.no_grad():
        Vb.bias.normal_()
    dense_b = graph.messages_dense(h_prev, Vb)
    assert torch.allclose(dense_b, _oracle(graph, h_prev, Vb), atol=1e-5, rtol=0.0)
    after = Vb(torch.einsum("ij,bjd->bid", graph.dense_adjacency(), h_prev))
    if edge_norm == "sigmoid":
        assert not torch.allclose(dense_b, after, atol=1e-3)
    else:
        assert torch.allclose(dense_b, after, atol=1e-5, rtol=0.0)  # rows sum to one

    # edge weights: normalised over the incoming edges of each cell
    weights = graph.edge_weights()
    assert weights.shape == (graph.n_edges,) and (weights > 0).all() and (weights < 1).all()
    row_sums = weights.new_zeros(N).index_add(0, _rows(graph), weights)
    if edge_norm == "softmax":
        assert torch.allclose(row_sums, torch.ones(N, device=device), atol=1e-6)
    else:
        assert torch.allclose(weights, torch.sigmoid(graph.edge_logit))
    A = graph.dense_adjacency()
    assert torch.equal(A[_rows(graph), graph.indices], weights)
    assert int((A != 0).sum()) == graph.n_edges

    # the same arithmetic on the CPU oracle
    if device.type != "cpu":
        cpu = _graph(cfg, "cpu", seed=1, random_edges=2, edge_norm=edge_norm)
        cpu.load_state_dict({k: v.cpu() for k, v in graph.state_dict().items()})
        V_cpu = nn.Linear(d, d, bias=False)
        V_cpu.load_state_dict({k: v.cpu() for k, v in V.state_dict().items()})
        ref = cpu.messages_dense(h_prev.cpu(), V_cpu)
        assert torch.allclose(dense.cpu(), ref, atol=1e-5, rtol=0.0)
        ref_sel = cpu.messages_selected(h_prev.cpu(), V_cpu, selections[1].cpu())
        assert torch.allclose(
            graph.messages_selected(h_prev, V, selections[1]).cpu(), ref_sel, atol=1e-5, rtol=0.0
        )

    # shape, dtype, range and device mismatches raise instead of being reinterpreted
    with pytest.raises(ValueError, match=r"\[B, N, d\]"):
        graph.messages_dense(h_prev[:, : N - 1], V)
    with pytest.raises(ValueError, match="int64"):
        graph.messages_selected(h_prev, V, selections[0].to(torch.int32))
    with pytest.raises(ValueError, match=r"\[B, K\]"):
        graph.messages_selected(h_prev, V, selections[0][:1])
    with pytest.raises(ValueError, match=r"\[B, K\]"):
        graph.messages_selected(h_prev, V, selections[0][:, :0])
    with pytest.raises(IndexError, match=r"\[0, 32\)"):
        graph.messages_selected(h_prev, V, torch.full((B, K), N, device=device))
    with pytest.raises(IndexError, match=r"\[0, 32\)"):
        graph.messages_selected(h_prev, V, torch.full((B, K), -1, device=device))
    with pytest.raises(ValueError, match="floating"):
        graph.messages_dense(h_prev.to(torch.int64), V)
    if device.type != "cpu":
        with pytest.raises(ValueError, match="device"):
            graph.messages_dense(h_prev.cpu(), V)


@pytest.mark.parametrize("edge_norm", NORMS)
@pytest.mark.parametrize("device_type", DEVICES)
def test_p1_4_grad_reaches_edge_logits(device_type: str, edge_norm: str) -> None:
    device = _device(device_type)
    seed_everything(12)
    cfg = REF_A1
    B, N, d, K = 4, cfg.n_cells, cfg.d, cfg.k_select
    graph = _graph(cfg, device, seed=2, edge_norm=edge_norm)
    _randomise_logits(graph, seed=22)
    V = nn.Linear(d, d, bias=False).to(device)
    h_prev = torch.randn(B, N, d, device=device, requires_grad=True)
    cell_idx = torch.rand(B, N, device=device).argsort(dim=1)[:, :K]

    out = graph.messages_selected(h_prev, V, cell_idx)
    assert out.requires_grad
    weight = torch.randn_like(out)
    g_logit, g_V, g_h = torch.autograd.grad(
        (out * weight).sum(), (graph.edge_logit, V.weight, h_prev)
    )
    assert g_logit.shape == graph.edge_logit.shape and torch.isfinite(g_logit).all()
    # only the edges into the selected cells are on the path
    selected = torch.zeros(N, dtype=torch.bool, device=device)
    selected[cell_idx.reshape(-1)] = True
    on_path = selected[_rows(graph)]
    assert on_path.any() and not on_path.all()
    assert (g_logit[on_path] != 0).all()
    assert (g_logit[~on_path] == 0).all()
    assert g_V.count_nonzero().item() > 0 and g_h.count_nonzero().item() > 0
    if edge_norm == "softmax":  # a row's softmax gradient sums to zero
        row_sums = g_logit.new_zeros(N).index_add(0, _rows(graph), g_logit)
        assert torch.allclose(row_sums, torch.zeros(N, device=device), atol=1e-5)

    # the dense path reaches every edge; backward() accumulates into .grad
    graph.zero_grad()
    graph.messages_dense(h_prev, V).square().sum().backward()
    assert graph.edge_logit.grad is not None and (graph.edge_logit.grad != 0).all()

    # L_graph: Σ |a_ij| has no gradient under softmax (rows sum to one), positive under sigmoid
    graph.zero_grad()
    l1 = graph.l1()
    assert l1.shape == () and l1.requires_grad
    l1.backward()
    assert graph.edge_logit.grad is not None
    if edge_norm == "softmax":
        assert abs(l1.item() - N) < 1e-4
        assert torch.allclose(graph.edge_logit.grad, torch.zeros_like(graph.edge_logit), atol=1e-6)
    else:
        assert torch.allclose(l1, torch.sigmoid(graph.edge_logit).sum())
        assert (graph.edge_logit.grad > 0).all()

    # finite differences in fp64 on the tiny configuration (CPU: MPS has no fp64)
    if device.type == "cpu":
        tiny = _graph(TINY, "cpu", seed=4, edge_norm=edge_norm).double()
        _randomise_logits(tiny, seed=23)
        V64 = nn.Linear(TINY.d, TINY.d, bias=False).double()
        h64 = torch.randn(2, TINY.n_cells, TINY.d, dtype=torch.float64, requires_grad=True)
        idx = torch.tensor([[0, 2], [1, 1]])

        def selected_fn(h: Tensor, logit: Tensor) -> Tensor:
            return tiny.messages_selected(h, V64, idx)

        def dense_fn(h: Tensor, logit: Tensor) -> Tensor:
            return tiny.messages_dense(h, V64)

        assert torch.autograd.gradcheck(selected_fn, (h64, tiny.edge_logit), atol=1e-6)
        assert torch.autograd.gradcheck(dense_fn, (h64, tiny.edge_logit), atol=1e-6)
        assert torch.autograd.gradcheck(lambda logit: tiny.l1(), (tiny.edge_logit,), atol=1e-6)


@pytest.mark.parametrize("device_type", DEVICES)
def test_p1_4_set_topology_inside_step_raises(device_type: str) -> None:
    device = _device(device_type)
    seed_everything(13)
    cfg = REF_A1
    B, N, d, K = 2, cfg.n_cells, cfg.d, cfg.k_select
    graph = _graph(cfg, device, seed=0)
    _randomise_logits(graph, seed=24)
    other = _graph(cfg, device, seed=1)  # another valid topology of the same size
    before = _clone_csr(graph)
    logit_before = graph.edge_logit.detach().clone()
    V = nn.Linear(d, d, bias=False).to(device)
    h_prev = torch.randn(B, N, d, device=device)
    cell_idx = torch.rand(B, N, device=device).argsort(dim=1)[:, :K]

    assert graph.in_step is False
    with graph.step():
        assert graph.in_step is True
        messages = graph.messages_dense(h_prev, V)  # messages are fine inside a step
        with pytest.raises(RuntimeError, match="between windows"):
            graph.set_topology(other.indptr, other.indices, other.is_random)
        with graph.step():  # re-entrant: a window wrapping its steps
            assert graph.in_step is True
            with pytest.raises(RuntimeError, match="in_step"):
                graph.set_topology(other.indptr, other.indices, other.is_random)
        assert graph.in_step is True
    assert graph.in_step is False
    for name in CSR:  # nothing changed
        assert torch.equal(getattr(graph, name), before[name]), name
    assert torch.equal(graph.edge_logit.detach(), logit_before)
    assert torch.equal(graph.messages_dense(h_prev, V), messages)
    # the flag is restored when the step raises, and the plain attribute works too
    with pytest.raises(KeyError):
        with graph.step():
            raise KeyError("step failed")
    assert graph.in_step is False
    graph.in_step = True
    with pytest.raises(RuntimeError, match="between windows"):
        graph.set_topology(other.indptr, other.indices, other.is_random)
    graph.in_step = False

    # between windows the new topology is accepted; kept edges keep their logits, new ones are 0
    old_logits = {
        (int(i), int(j)): float(v)
        for i, j, v in zip(
            _rows(graph).tolist(), graph.indices.tolist(), logit_before.tolist(), strict=True
        )
    }
    graph.set_topology(other.indptr, other.indices, other.is_random)
    for name in CSR:
        assert torch.equal(getattr(graph, name), getattr(other, name)), name
        assert getattr(graph, name).device.type == device.type
    assert isinstance(graph.edge_logit, nn.Parameter) and graph.edge_logit.requires_grad
    assert graph.edge_logit.device.type == device.type
    assert graph.edge_logit.shape == (other.n_edges,)
    expected = torch.tensor(
        [
            old_logits.get((int(i), int(j)), 0.0)
            for i, j in zip(_rows(graph).tolist(), graph.indices.tolist(), strict=True)
        ],
        device=device,
    )
    assert torch.equal(graph.edge_logit.detach(), expected)
    kept = sum(
        1
        for key in zip(_rows(graph).tolist(), graph.indices.tolist(), strict=True)
        if key in old_logits
    )
    assert 0 < kept < graph.n_edges  # some edges survived, some are new (seed 0 vs seed 1)
    # the derived view was rebuilt: messages follow the new adjacency
    assert torch.allclose(graph.messages_dense(h_prev, V), _oracle(graph, h_prev, V), atol=1e-5)
    selected = graph.messages_selected(h_prev, V, cell_idx)
    gathered = graph.messages_dense(h_prev, V).gather(1, cell_idx[:, :, None].expand(B, K, d))
    assert torch.allclose(selected, gathered, atol=1e-6, rtol=0.0)

    # a ragged topology: fewer sources than D_g, an empty row, random extras above the cap
    sources = [[(i + 1) % N, (i + 2) % N, (i + 5) % N] for i in range(N)]
    random = [[False, False, True] for _ in range(N)]
    sources[3], random[3] = [], []
    sources[4], random[4] = [0], [False]
    sources[5], random[5] = [1, 2, 6, 9, 10, 17, 18], [False] * 4 + [True] * 3
    rows_sorted = [sorted(zip(s, r, strict=True)) for s, r in zip(sources, random, strict=True)]
    indptr, indices, is_random = _csr(
        [[j for j, _ in row] for row in rows_sorted], [[f for _, f in row] for row in rows_sorted]
    )
    graph.set_topology(indptr, indices, is_random)
    assert graph.n_edges == indices.numel() and graph.max_in_degree == 7
    weights = graph.edge_weights()
    row_sums = weights.new_zeros(N).index_add(0, _rows(graph), weights)
    non_empty = torch.ones(N, dtype=torch.bool, device=device)
    non_empty[3] = False
    assert torch.allclose(row_sums[non_empty], torch.ones(N - 1, device=device), atol=1e-6)
    assert row_sums[3].item() == 0.0 and abs(graph.l1().item() - (N - 1)) < 1e-5
    dense = graph.messages_dense(h_prev, V)
    assert torch.allclose(dense, _oracle(graph, h_prev, V), atol=1e-5, rtol=0.0)
    assert torch.equal(dense[:, 3], torch.zeros(B, d, device=device))
    idx = torch.tensor([[3, 4, 5, 5], [5, 3, 0, 4]], device=device)
    selected = graph.messages_selected(h_prev, V, idx)
    assert torch.allclose(selected, dense.gather(1, idx[:, :, None].expand(B, 4, d)), atol=1e-6)
    torch.autograd.grad(selected.sum() + graph.l1(), graph.edge_logit)  # finite through empty rows
    assert torch.isfinite(graph.edge_weights()).all()

    # invalid topologies are refused and leave the graph untouched
    before = _clone_csr(graph)
    base = [[(i + 1) % N, (i + 2) % N] for i in range(N)]
    base_sorted = [sorted(row) for row in base]
    bad_cases: list[tuple[str, tuple[Tensor, ...]]] = [
        ("strictly increasing", _csr([[2, 1]] + base_sorted[1:])),  # unsorted row
        ("strictly increasing", _csr([[1, 1]] + base_sorted[1:])),  # duplicate source
        (r"\[0, 32\)", _csr([[1, N]] + base_sorted[1:])),
        (r"\[0, 32\)", _csr([[-1, 1]] + base_sorted[1:])),
        ("non-random", _csr([[1, 2, 3, 4, 5]] + base_sorted[1:])),  # 5 fixed > D_g = 4
        ("self loop", _csr([[0, 1]] + base_sorted[1:])),
    ]
    for pattern, csr in bad_cases:
        with pytest.raises(ValueError, match=pattern):
            graph.set_topology(*csr)
    ok_indptr, ok_indices, ok_random = _csr(base_sorted)
    with pytest.raises(ValueError, match="indptr"):
        graph.set_topology(ok_indptr + 1, ok_indices, ok_random)
    with pytest.raises(ValueError, match="indptr"):
        graph.set_topology(ok_indptr[:-1], ok_indices, ok_random)
    with pytest.raises(ValueError, match="indptr"):
        graph.set_topology(ok_indptr.flip(0), ok_indices, ok_random)
    with pytest.raises(ValueError, match="int64"):
        graph.set_topology(ok_indptr, ok_indices.to(torch.int32), ok_random)
    with pytest.raises(ValueError, match="bool"):
        graph.set_topology(ok_indptr, ok_indices, ok_random.to(torch.int64))
    with pytest.raises(ValueError, match=r"\[E\]"):
        graph.set_topology(ok_indptr, ok_indices, ok_random[:-1])
    with pytest.raises(ValueError, match="one-dimensional"):
        graph.set_topology(ok_indptr, ok_indices[None], ok_random)
    with pytest.raises(TypeError):
        graph.set_topology(ok_indptr.tolist(), ok_indices, ok_random)  # type: ignore[arg-type]
    for name in CSR:
        assert torch.equal(getattr(graph, name), before[name]), name
    # five fixed sources plus random extras above D_g are fine: the cap counts fixed edges only
    graph.set_topology(
        *_csr(
            [[1, 2, 3, 4, 5, 6]] + base_sorted[1:],
            [[False] * 4 + [True] * 2] + [[False, False]] * (N - 1),
        )
    )
    assert graph.max_in_degree == 6


@pytest.mark.parametrize("device_type", DEVICES)
def test_p1_4_state_dict_round_trip(device_type: str) -> None:
    device = _device(device_type)
    seed_everything(14)
    cfg = REF_A1
    B, N, d = 2, cfg.n_cells, cfg.d
    V = nn.Linear(d, d, bias=False).to(device)
    h_prev = torch.randn(B, N, d, device=device)
    source = _graph(cfg, device, seed=0, random_edges=1, edge_norm="sigmoid")
    _randomise_logits(source, seed=25)
    state = source.state_dict()
    assert set(state) == {
        "indptr",
        "indices",
        "is_random",
        "edge_logit",
    }  # derived buffers are not saved

    # same edge count, other edges: the loaded CSR replaces the old one and the view is rebuilt
    target = _graph(cfg, device, seed=9, random_edges=1, edge_norm="sigmoid")
    assert not torch.equal(target.indices, source.indices)
    target.load_state_dict(state)
    for name in CSR:
        assert torch.equal(getattr(target, name), getattr(source, name)), name
    assert torch.equal(target.edge_logit.detach(), source.edge_logit.detach())
    assert torch.equal(target.messages_dense(h_prev, V), source.messages_dense(h_prev, V))

    # another edge count is refused before anything is copied; set_topology first, then load
    smaller = _graph(cfg, device, seed=0, random_edges=0, edge_norm="sigmoid")
    before = _clone_csr(smaller)
    with pytest.raises(RuntimeError, match="set_topology"):
        smaller.load_state_dict(state)
    for name in CSR:
        assert torch.equal(getattr(smaller, name), before[name]), name
    smaller.set_topology(source.indptr, source.indices, source.is_random)
    smaller.load_state_dict(state)
    assert torch.equal(smaller.messages_dense(h_prev, V), source.messages_dense(h_prev, V))

    # a corrupt checkpoint (a self loop) is refused by the post-load validation
    corrupt = {k: v.clone() for k, v in state.items()}
    corrupt["indices"][0] = 0
    with pytest.raises(ValueError, match="self loop"):
        _graph(cfg, device, seed=0, random_edges=1).load_state_dict(corrupt)


def test_p1_4_deterministic_by_seed() -> None:
    cfg = REF_A1
    torch.manual_seed(123)
    first = _graph(cfg, seed=7, random_edges=2)
    torch.manual_seed(999)
    torch.rand(1000)  # a different global RNG state
    second = _graph(cfg, seed=7, random_edges=2)
    for name in CSR:
        assert torch.equal(getattr(first, name), getattr(second, name)), name
    assert torch.equal(first.edge_logit.detach(), second.edge_logit.detach())
    assert torch.equal(first.dense_adjacency(), second.dense_adjacency())

    # different seeds give different topologies at the reference shapes
    assert not torch.equal(_graph(cfg, seed=8, random_edges=2).indices, first.indices)
    assert not torch.equal(_graph(cfg, seed=0).indices, _graph(cfg, seed=1).indices)
    # (the tiny configuration has D_g = N − 1, so every seed must read the two other cells)
    tiny_sources = _graph(TINY, seed=0).indices
    assert torch.equal(tiny_sources, torch.tensor([1, 2, 0, 2, 0, 1]))
    assert torch.equal(_graph(TINY, seed=31).indices, tiny_sources)

    # construction neither reads nor advances the global RNG
    seed_everything(5)
    expected = torch.rand(3)
    seed_everything(5)
    _graph(cfg, seed=7, random_edges=2)
    assert torch.equal(torch.rand(3), expected)

    # the seed is part of the module's repr for run manifests and logs
    text = repr(first)
    assert "n_cells=32" in text and "in_degree=4" in text and "random_edges=2" in text
    assert "edge_norm='softmax'" in text and "n_edges=192" in text
