"""P0.3 tests: device resolution, dtype policy, seeding, determinism, memory stats, VRAM cap."""

import os
import random

import numpy as np
import pytest
import torch
from hsn.config import load_hardware_profile
from hsn.device import (
    DeviceContext,
    memory_stats,
    reset_peak,
    resolve,
    seed_everything,
    set_deterministic,
    set_vram_cap,
    synchronize,
)

MIB = 1024 * 1024
CUDA_PROFILES = ("rtx3060", "a100", "dgx_spark")

cuda_param = pytest.param("cuda", marks=pytest.mark.cuda)
mps_param = pytest.param("mps", marks=pytest.mark.mps)


def _available_gpu_types() -> list[str]:
    types = []
    if torch.cuda.is_available():
        types.append("cuda")
    if torch.backends.mps.is_available():
        types.append("mps")
    return types


def _require(device_type: str) -> torch.device:
    if device_type not in _available_gpu_types():
        pytest.skip(f"no {device_type} device on this machine")
    return torch.device(device_type)


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _total_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.mem_get_info(device)[1])
    return int(torch.mps.recommended_max_memory())


# --------------------------------------------------------------------------------------------------
# resolve
# --------------------------------------------------------------------------------------------------


def test_p0_3_cpu_always_resolves() -> None:
    ctx = resolve(load_hardware_profile("cpu"))
    assert isinstance(ctx, DeviceContext)
    assert ctx.device == torch.device("cpu")
    assert ctx.autocast_dtype is None  # §3.7: the CPU runs fp32 without autocast
    assert ctx.master_dtype is torch.float32
    assert ctx.supports_bf16 is True
    assert ctx.unified_memory is True
    assert ctx.vram_bytes > 0  # measured host RAM; the profile file says 0
    assert ctx.compile_enabled is False
    assert ctx.profile == "cpu"
    # a tensor on the resolved device works
    assert torch.ones(2, device=ctx.device).sum().item() == 2.0


@pytest.mark.mps
def test_p0_3_m_series_resolves() -> None:
    _require("mps")
    ctx = resolve(load_hardware_profile("m_series"))
    assert ctx.device == torch.device("mps")
    assert ctx.autocast_dtype is torch.float16
    assert ctx.master_dtype is torch.float32
    assert isinstance(ctx.supports_bf16, bool)  # measured by a bf16 autocast probe
    assert ctx.unified_memory is True
    assert ctx.vram_bytes == torch.mps.recommended_max_memory() > 0
    assert ctx.compile_enabled is False
    assert ctx.profile == "m_series"


@pytest.mark.cuda
@pytest.mark.parametrize("name", CUDA_PROFILES)
def test_p0_3_cuda_profiles_resolve(name: str) -> None:
    _require("cuda")
    ctx = resolve(load_hardware_profile(name))
    assert ctx.device.type == "cuda"
    assert ctx.autocast_dtype is torch.bfloat16  # every CUDA profile asks for bf16 (§3.7)
    assert ctx.supports_bf16 is True
    assert ctx.master_dtype is torch.float32
    assert ctx.vram_bytes == torch.cuda.mem_get_info(ctx.device)[1] > 0
    assert ctx.compile_enabled is False
    assert ctx.profile == name


def test_p0_3_unavailable_device_raises() -> None:
    present = _available_gpu_types()
    absent = [
        name
        for name in ("rtx3060", "m_series")
        if load_hardware_profile(name).device not in present
    ]
    if not absent:
        pytest.skip("every GPU backend is available on this machine")
    for name in absent:
        with pytest.raises(RuntimeError, match=name):
            resolve(load_hardware_profile(name))


def test_p0_3_compile_mode_rejected() -> None:
    cpu = load_hardware_profile("cpu")
    with pytest.raises(NotImplementedError, match="P2.8"):
        resolve(cpu.model_copy(update={"compile_mode": "reduce-overhead"}))


def test_p0_3_dtype_policy_mismatch_raises() -> None:
    cpu = load_hardware_profile("cpu")
    with pytest.raises(ValueError, match="fp32"):
        resolve(cpu.model_copy(update={"autocast_dtype": "fp16"}))
    with pytest.raises(ValueError, match="unified_memory"):
        resolve(cpu.model_copy(update={"unified_memory": False}))
    assert resolve(cpu.model_copy(update={"master_dtype": "fp64"})).master_dtype is torch.float64
    if "mps" in _available_gpu_types():
        m_series = load_hardware_profile("m_series")
        with pytest.raises(ValueError, match="float64"):
            resolve(m_series.model_copy(update={"master_dtype": "fp64"}))


# --------------------------------------------------------------------------------------------------
# seeding and determinism
# --------------------------------------------------------------------------------------------------


def _draws() -> tuple[float, float, torch.Tensor, dict[str, torch.Tensor]]:
    gpu = {t: torch.rand(8, device=t).cpu() for t in _available_gpu_types()}
    return random.random(), float(np.random.rand()), torch.rand(8), gpu


def test_p0_3_seed_reproducible() -> None:
    seed_everything(1234)
    first = _draws()
    seed_everything(1234)
    second = _draws()
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert first[3].keys() == second[3].keys()
    for device_type, draw in first[3].items():
        assert torch.equal(draw, second[3][device_type]), device_type
    seed_everything(4321)
    assert not torch.equal(_draws()[2], first[2])
    for bad in (-1, 2**32, 1.5, True, "7"):
        with pytest.raises(ValueError):
            seed_everything(bad)  # type: ignore[arg-type]


def test_p0_3_set_deterministic_roundtrip() -> None:
    flag_before = torch.are_deterministic_algorithms_enabled()
    cudnn_before = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    env_before = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        set_deterministic(True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == (env_before or ":4096:8")
        set_deterministic(False)
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.deterministic
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == env_before
    finally:
        torch.use_deterministic_algorithms(flag_before)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_before
        if env_before is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = env_before


# --------------------------------------------------------------------------------------------------
# memory statistics, caps, synchronisation
# --------------------------------------------------------------------------------------------------


def test_p0_3_memory_stats_keys() -> None:
    for device_type in ("cpu", *_available_gpu_types()):
        device = torch.device(device_type)
        reset_peak(device)
        before = memory_stats(device)
        assert set(before) == {"allocated", "reserved", "peak"}, device_type
        for key, value in before.items():
            assert isinstance(value, int) and not isinstance(value, bool), (device_type, key)
            assert value >= 0, (device_type, key)
        assert before["peak"] >= before["allocated"]

        block = torch.ones(64 * MIB, dtype=torch.uint8, device=device)
        synchronize(device)
        after = memory_stats(device)
        if device_type != "cpu":  # exact allocator counters on CUDA and MPS
            assert after["allocated"] >= before["allocated"] + 64 * MIB, device_type
            assert after["reserved"] >= after["allocated"], device_type
        assert after["peak"] >= after["allocated"]
        assert after["peak"] >= before["peak"]

        del block
        synchronize(device)
        reset_peak(device)
        if device_type != "cpu":  # RSS samples may drift by a few pages between two reads
            assert memory_stats(device)["peak"] <= after["peak"]
        _empty_cache(device)
    with pytest.raises(ValueError, match="unsupported device"):
        memory_stats("xpu")


@pytest.mark.parametrize("device_type", [cuda_param, mps_param])
def test_p0_3_vram_cap_applies(device_type: str) -> None:
    device = _require(device_type)
    total = _total_bytes(device)
    cap = 256 * MIB
    assert total > 4 * cap, f"{device} has only {total} bytes; the cap test needs > 1 GiB"
    _empty_cache(device)
    inside = beyond = None
    try:
        set_vram_cap(device, cap / total)
        inside = torch.empty(16 * MIB, dtype=torch.uint8, device=device)  # within the cap
        synchronize(device)
        with pytest.raises(RuntimeError, match="out of memory"):
            beyond = torch.empty(2 * cap, dtype=torch.uint8, device=device)  # beyond the cap
            synchronize(device)
    finally:
        del inside, beyond
        set_vram_cap(device, 1.0)
        _empty_cache(device)
    # the cap is gone again: the same allocation now succeeds
    recovered = torch.empty(2 * cap, dtype=torch.uint8, device=device)
    synchronize(device)
    del recovered
    _empty_cache(device)


def test_p0_3_vram_cap_rejects_bad_fraction() -> None:
    for bad in (0.0, -1.0, 1.5, 2.0, float("nan"), True, "0.5"):
        with pytest.raises(ValueError, match="fraction"):
            set_vram_cap("cpu", bad)  # type: ignore[arg-type]
    set_vram_cap("cpu", 0.5)  # no-op on the CPU
    set_vram_cap(torch.device("cpu"), 1.0)
    with pytest.raises(ValueError, match="unsupported device"):
        set_vram_cap("xpu", 0.5)
    absent = [t for t in ("cuda", "mps") if t not in _available_gpu_types()]
    for device_type in absent:
        with pytest.raises(RuntimeError, match="not available"):
            set_vram_cap(device_type, 0.5)


def test_p0_3_synchronize_all_devices() -> None:
    synchronize("cpu")
    synchronize(torch.device("cpu"))
    for device_type in _available_gpu_types():
        value = torch.ones(1024, device=device_type).sum()
        synchronize(device_type)
        assert value.item() == 1024.0
    with pytest.raises(ValueError, match="unsupported device"):
        synchronize("xpu")
