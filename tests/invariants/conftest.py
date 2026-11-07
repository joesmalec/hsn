"""Fixtures of the P1.8 invariant suite (Track T).

The suite drives the dense model through its public API only (``HSN.from_config``, ``init_state``,
``step``, ``forward``, ``reset_state``, the P1.3 snapshot/restore of ``CellGraphState`` and the P1.7
``TBPTTTrainer``), on two configurations: ``configs/tiny_gradcheck.yaml`` (fp64 on the CPU, small
enough for finite differences and for 10,000-step runs) and ``configs/ref_a1.yaml`` (the Appendix C
reference, fp32).

Comparison policy (chunk pitfall): the CPU profile is the numerical oracle and every "unchanged" or
"identical" assertion there is bit for bit (``torch.equal``). Accelerators are not bit-reproducible
across runs (P0.8 measured MPS reruns), so the GPU variants of the same tests compare with
``atol = rtol = 1e-5`` through :class:`SameChecker`, except where the value is a carried *copy*
rather than arithmetic (a disabled cell's row), which is exact on every device. Bit-exact-only
tests (gradcheck, checkpoint resume, seed determinism) have no GPU variant.

One comparison is not bit for bit on the CPU either (P1.10a): ``HSN.forward`` runs E and D over the
window ``[B, T]`` while ``HSN.step`` runs them over ``[B]``, and BLAS rounds a GEMM differently with
its row count, so a value of ``step`` against the same value of ``forward`` agrees to
:data:`WINDOW_ATOL` (by dtype) through :meth:`SameChecker.step_vs_forward`. Every property the
suite protects (leakage, the reset, disabled cells, determinism, resume) still compares runs of the
same shapes, bit for bit.

The P0.5 trainer applies ``train.deterministic`` process-wide
(``torch.use_deterministic_algorithms``) and never resets it; a leaked flag makes later MPS
backward passes raise (``index_put_with_accumulate_mps``). The autouse fixture restores the flag
after every test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
import torch
from hsn.config import RunConfig, load_config, load_hardware_profile
from hsn.data.dataset import collate_windows
from hsn.data.synthetic import Example, TaskSpec, generate, load_dataset_spec
from hsn.device import DeviceContext, resolve, seed_everything, set_deterministic
from hsn.model.hsn import HSN
from hsn.model.state import CellGraphState
from torch import Tensor

REPO = Path(__file__).resolve().parents[2]
CONFIGS: dict[str, Path] = {
    "tiny": REPO / "configs" / "tiny_gradcheck.yaml",
    "ref": REPO / "configs" / "ref_a1.yaml",
}
SPEC = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
UNROLL = 32  # the trainer tests' window (the P1.7 fixture recipe)
MIN_DOC_LENGTH = 16  # the shortest synthetic document of the recipe
N_TRAIN_DOCS = 64
N_DEV_DOCS = 16
#: Tolerance of the GPU variants (the CPU profile compares bit for bit).
DEVICE_ATOL = 1e-5
DEVICE_RTOL = 1e-5
#: P1.10a: ``HSN.step`` against ``HSN.forward`` on the CPU, by dtype (the GEMM row count differs).
WINDOW_ATOL = {torch.float64: 1e-12, torch.float32: 1e-5}
#: The P1.7 recipe: ref_a1 shrunk to a few hundred parameters for trainer tests (real byte data,
#: vocabulary 256, context symbols 16), with a 32-position window and a batch of 4 streams.
TRAINER_OVERRIDES: tuple[str, ...] = (
    "model.d=8",
    "model.context_dim=8",
    "model.n_cells=4",
    "model.in_degree=2",
    "model.n_units=2",
    "model.k_select=2",
    "model.hidden=16",
    "model.scorer_hidden=8",
    f"train.unroll={UNROLL}",
    "train.batch=4",
    "train.micro_batch=4",
)

cuda_param = pytest.param("cuda", marks=pytest.mark.cuda)
mps_param = pytest.param("mps", marks=pytest.mark.mps)

ModelFactory = Callable[..., tuple[RunConfig, HSN]]
IdsFactory = Callable[[HSN, int, int, int], tuple[Tensor, Tensor]]
PerturbFn = Callable[..., None]
StateFactory = Callable[..., CellGraphState]
ConfigFactory = Callable[..., RunConfig]


# --------------------------------------------------------------------------------------------------
# Determinism flag hygiene
# --------------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_determinism() -> Iterator[None]:
    """Restore ``torch.use_deterministic_algorithms`` after every test: the trainer sets it
    process-wide from ``train.deterministic`` (P0.5) and pytest collects this directory before the
    other test modules, so a leak would reach every later MPS test."""
    before = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        set_deterministic(before)


@pytest.fixture
def deterministic_kernels() -> Iterator[None]:
    """Nondeterministic kernels disabled for the duration of one test (the gradcheck pitfall)."""
    before = torch.are_deterministic_algorithms_enabled()
    set_deterministic(True)
    try:
        yield
    finally:
        set_deterministic(before)


# --------------------------------------------------------------------------------------------------
# Devices and the comparison policy
# --------------------------------------------------------------------------------------------------


def device_available(device_type: str) -> bool:
    if device_type == "cpu":
        return True
    if device_type == "cuda":
        return bool(torch.cuda.is_available())
    if device_type == "mps":
        return bool(torch.backends.mps.is_available())
    raise ValueError(f"unknown device type {device_type!r}")


@pytest.fixture(params=["cpu", cuda_param, mps_param])
def device(request: pytest.FixtureRequest) -> torch.device:
    """The device of a GPU-variant test; only tests that request this fixture get the ``cuda`` /
    ``mps`` variants, which skip when the device is absent. The device is returned as tensors
    report it (``mps:0``, with the index), which a bare ``torch.device("mps")`` never equals."""
    device_type = str(request.param)
    if not device_available(device_type):
        pytest.skip(f"no {device_type} device on this machine")
    return torch.empty(0, device=device_type).device


class SameChecker:
    """``expect_same(actual, expected, device, what=...)``: bit for bit on the CPU, within
    :data:`DEVICE_ATOL` / :data:`DEVICE_RTOL` on an accelerator. The largest |difference| seen per
    ``(device, what)`` is kept in ``worst`` so that a test can report the measured numbers."""

    def __init__(self) -> None:
        self.worst: dict[tuple[str, str], float] = {}

    def __call__(
        self, actual: Tensor, expected: Tensor, device: torch.device, *, what: str = "value"
    ) -> None:
        if actual.shape != expected.shape:
            raise AssertionError(
                f"{what}: shapes differ, {tuple(actual.shape)} vs {tuple(expected.shape)}"
            )
        a = actual.detach().cpu()
        e = expected.detach().cpu()
        if a.is_floating_point():
            diff = float((a.double() - e.double()).abs().max()) if a.numel() else 0.0
        else:
            diff = float((a != e).sum())
        key = (device.type, what)
        self.worst[key] = max(self.worst.get(key, 0.0), diff)
        if device.type == "cpu":
            assert torch.equal(a, e), f"{what}: max |diff| {diff:.3e} on cpu, bit-exact expected"
        else:
            torch.testing.assert_close(
                a,
                e,
                atol=DEVICE_ATOL,
                rtol=DEVICE_RTOL,
                msg=lambda m: f"{what} on {device.type}: {m}",
            )

    def step_vs_forward(
        self, actual: Tensor, expected: Tensor, device: torch.device, *, what: str = "value"
    ) -> None:
        """A value of ``HSN.step`` against the same value of ``HSN.forward`` (P1.10a, module
        docstring): within :data:`WINDOW_ATOL` of its dtype on the CPU, the device tolerance on an
        accelerator. Recorded in ``worst`` like every other comparison."""
        if device.type != "cpu":
            self(actual, expected, device, what=what)
            return
        a = actual.detach().cpu()
        e = expected.detach().cpu()
        diff = float((a.double() - e.double()).abs().max()) if a.numel() else 0.0
        key = (device.type, what)
        self.worst[key] = max(self.worst.get(key, 0.0), diff)
        torch.testing.assert_close(
            a, e, atol=WINDOW_ATOL[e.dtype], rtol=0.0, msg=lambda m: f"{what} on cpu: {m}"
        )

    def report(self) -> str:
        """One line per ``(device, what)`` with the largest difference observed."""
        return "; ".join(
            f"{dev}/{what} {diff:.2e}" for (dev, what), diff in sorted(self.worst.items())
        )


@pytest.fixture
def expect_same() -> SameChecker:
    return SameChecker()


# --------------------------------------------------------------------------------------------------
# Models: the tiny fp64 configuration and the Appendix C reference
# --------------------------------------------------------------------------------------------------


def build_model(
    name: str,
    *,
    device: torch.device | str = "cpu",
    overrides: Sequence[str] = (),
    seed: int | None = None,
) -> tuple[RunConfig, HSN]:
    """``(cfg, model)`` from ``configs/tiny_gradcheck.yaml`` (``"tiny"``) or ``configs/ref_a1.yaml``
    (``"ref"``) with ``overrides``, weights seeded from ``seed`` (``run.seed`` by default).

    The YAML's ``precision: fp64`` is honoured on the CPU only (§3.7: fp64 is the CPU gradcheck
    policy; MPS has no float64), so on an accelerator the model is built in fp32.
    """
    dev = torch.device(device)
    extra = list(overrides)
    if dev.type != "cpu":
        extra.append("train.precision=fp32")
    cfg = load_config(CONFIGS[name], overrides=extra)
    seed_everything(cfg.seed if seed is None else seed)
    model = HSN.from_config(cfg)
    if cfg.train.precision == "fp64":
        model = model.double()
    return cfg, model.to(dev)


@pytest.fixture(scope="session")
def tiny_cfg() -> RunConfig:
    return load_config(CONFIGS["tiny"])


@pytest.fixture(scope="session")
def ref_cfg() -> RunConfig:
    return load_config(CONFIGS["ref"])


@pytest.fixture
def tiny_model() -> HSN:
    """The tiny configuration in fp64 on the CPU (V=16, d=4, N=3, M=2, K=2, D_g=2, H=8; 1,083
    parameters plus 6 edge logits), weights seeded from ``run.seed``."""
    _, model = build_model("tiny")
    assert model.dtype is torch.float64 and model.device.type == "cpu"
    return model


@pytest.fixture
def ref_model() -> HSN:
    """The Appendix C reference configuration in fp32 on the CPU (1,820,617 parameters after
    amendment A1, plus 128 edge logits), weights seeded from ``run.seed``."""
    _, model = build_model("ref")
    assert model.dtype is torch.float32 and model.device.type == "cpu"
    return model


@pytest.fixture
def make_model() -> ModelFactory:
    """The :func:`build_model` factory for device-parametrised tests."""
    return build_model


# --------------------------------------------------------------------------------------------------
# Inputs, perturbations and states (the P1.5 recipes)
# --------------------------------------------------------------------------------------------------


def random_ids(model: HSN, B: int, T: int, seed: int) -> tuple[Tensor, Tensor]:
    """``tokens``, ``ctx`` ``[B, T]`` on the model's device, drawn on the CPU from ``seed``."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    m = model.cfg
    tokens = torch.randint(0, m.vocab_size, (B, T), generator=generator)
    ctx = torch.randint(0, m.n_context, (B, T), generator=generator)
    return tokens.to(model.device), ctx.to(model.device)


def perturb_model(model: HSN, seed: int, scale: float = 0.3) -> None:
    """Seeded noise on every parameter: non-zero biases, non-uniform edge weights and states of
    order 0.1 instead of the 1e-5 of a freshly initialised model."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn(p.shape, generator=generator, dtype=torch.float64) * scale
            p.add_(noise.to(dtype=p.dtype, device=p.device))


def random_cell_state(model: HSN, B: int, seed: int, ticks: int = 2) -> CellGraphState:
    """A state on the model's device with non-trivial ``h`` (|h| < 0.8), the exact ``q``, and
    non-zero ``age`` / ``pos`` after ``ticks`` rounds."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    N, d = model.cfg.n_cells, model.cfg.d
    h = torch.rand(B, N, d, generator=generator, dtype=torch.float64) * 1.6 - 0.8
    state = model.init_state(B).with_dense(h.to(dtype=model.dtype, device=model.device))
    for _ in range(ticks):
        mask = torch.rand(B, N, generator=generator) < 0.5
        state = state.tick(mask.to(model.device))
    return state


@pytest.fixture
def ids() -> IdsFactory:
    return random_ids


@pytest.fixture
def perturb() -> PerturbFn:
    return perturb_model


@pytest.fixture
def random_state() -> StateFactory:
    return random_cell_state


# --------------------------------------------------------------------------------------------------
# Synthetic data and the trainer configuration (the P1.7 recipes)
# --------------------------------------------------------------------------------------------------


def _tiny_task(
    length_range: tuple[int, int] = (MIN_DOC_LENGTH, UNROLL),
    dep_range: tuple[int, int] = (4, 20),
) -> TaskSpec:
    """The committed kv in-range task with short documents (kv needs dep + 9 symbols)."""
    task = SPEC.task("kv", "in_range")
    return TaskSpec.model_validate(
        {
            **task.model_dump(),
            "n_train": N_TRAIN_DOCS,
            "n_dev": N_DEV_DOCS,
            "n_test": N_DEV_DOCS,
            "length_range": list(length_range),
            "dep_len_range": list(dep_range),
            "unroll": UNROLL,
        }
    )


def _docs(split: str, n: int) -> list[Example]:
    task = _tiny_task()
    return [generate(task, split, i) for i in range(n)]  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def train_docs() -> list[Example]:
    return _docs("train", N_TRAIN_DOCS)


@pytest.fixture(scope="session")
def dev_docs() -> list[Example]:
    return _docs("dev", N_DEV_DOCS)


@pytest.fixture(scope="session")
def synthetic_batch(train_docs: list[Example]) -> dict[str, Tensor]:
    """One window ``[4, 32]`` of real kv bytes (``tokens``, ``targets``, ``context``, ``loss_mask``,
    ``reset``, ``dep``, ``doc``) as :func:`collate_windows` lays it out for the vocabulary-256
    models: every row starts a document at position 0 and holds at least :data:`MIN_DOC_LENGTH`
    real positions (a document shorter than the window is followed by padding, ``doc == -1``, as
    the trainer sees it)."""
    batch = collate_windows(train_docs, unroll=UNROLL, batch=4, align_docs=True)
    window = {key: value[:, :UNROLL].clone() for key, value in batch.items()}
    assert window["reset"][:, 0].all() and (window["doc"][:, :MIN_DOC_LENGTH] >= 0).all()
    return window


def trainer_config(*overrides: str) -> RunConfig:
    """``ref_a1.yaml`` with :data:`TRAINER_OVERRIDES` and the test's own ``train.*`` overrides."""
    return load_config(CONFIGS["ref"], overrides=[*TRAINER_OVERRIDES, *overrides])


@pytest.fixture
def make_trainer_config() -> ConfigFactory:
    return trainer_config


@pytest.fixture
def cpu_ctx() -> DeviceContext:
    return resolve(load_hardware_profile("cpu"))
