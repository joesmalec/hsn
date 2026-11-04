"""Device and dtype layer (P0.3): a hardware profile becomes a torch device, a dtype policy,
seeding, determinism and memory helpers that behave the same on CUDA, MPS and CPU.

:func:`resolve` turns a :class:`hsn.config.HardwareConfig` into a :class:`DeviceContext`. It only
*detects*: it never changes allocator state. A profile's ``vram_cap_fraction`` is applied by the
caller through :func:`set_vram_cap` (the trainer in P0.5; simulated-cap tests in P4.3).

Dtype policy (plan §3.7): CUDA with compute capability >= 8.0 autocasts in bf16, MPS in fp16 and
the CPU runs fp32 without autocast; master weights are fp32 (fp64 only for gradcheck on the CPU).
The profile file states the policy; this module checks that the device can honour it and raises
otherwise (errors over silence). ``supports_bf16`` is a capability rule on CUDA (>= 8.0) and a
measured probe on MPS (a tiny matmul forward and backward under bf16 autocast).

Memory statistics (:func:`memory_stats`) report ``allocated``, ``reserved`` and ``peak`` in bytes
on every backend, but the backends differ in what they can measure:

* CUDA: the caching allocator's own counters (``memory_allocated``, ``memory_reserved``,
  ``max_memory_allocated``). ``peak`` is continuous and :func:`reset_peak` resets it.
* MPS: ``torch.mps.current_allocated_memory()`` (tensor bytes) and
  ``torch.mps.driver_allocated_memory()`` (heaps held from Metal, the analogue of "reserved").
  PyTorch 2.15 has no peak counter for MPS, so ``peak`` is the largest ``allocated`` value this
  module has *observed* in :func:`memory_stats` or :func:`reset_peak` since the last reset: a
  sampled high-water mark, coarser than CUDA's.
* CPU: the process's resident set size for both ``allocated`` and ``reserved`` (the host has no
  reservation pool) and the same sampled ``peak``.

Simulated VRAM cap (:func:`set_vram_cap`): a per-process memory fraction on CUDA and MPS, a no-op
on the CPU. On MPS the fraction multiplies ``torch.mps.recommended_max_memory()`` and PyTorch
enforces it against driver-allocated heap memory plus the request (heaps for large tensors are
1 GiB), so the cap is coarse; a fraction of ``0`` would mean *unlimited* there, which is why
fractions outside ``(0, 1]`` raise.
"""

from __future__ import annotations

import ctypes
import os
import platform
import random
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from hsn.config import HardwareConfig

__all__ = [
    "DeviceContext",
    "memory_stats",
    "reset_peak",
    "resolve",
    "seed_everything",
    "set_deterministic",
    "set_vram_cap",
    "synchronize",
]

_AUTOCAST_DTYPES: dict[str, torch.dtype | None] = {
    "fp32": None,  # §3.7: fp32 means no autocast
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
_MASTER_DTYPES: dict[str, torch.dtype] = {"fp32": torch.float32, "fp64": torch.float64}
_DEVICE_TYPES = ("cuda", "mps", "cpu")
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"  # the deterministic cuBLAS workspace PyTorch documents
_SEED_LIMIT = 2**32  # numpy's legacy seeding accepts [0, 2**32)

#: Sampled high-water marks (bytes) for the backends without a peak counter, keyed by device type.
_sampled_peak: dict[str, int] = {}
#: Whether :func:`set_deterministic` exported CUBLAS_WORKSPACE_CONFIG itself (so it may remove it).
_cublas_env_set_here = False


@dataclass(frozen=True, kw_only=True)
class DeviceContext:
    """What :func:`resolve` measured for one hardware profile."""

    device: torch.device
    autocast_dtype: torch.dtype | None  # None: no autocast (the fp32 policy)
    master_dtype: torch.dtype
    supports_bf16: bool
    unified_memory: bool
    vram_bytes: int  # measured device memory, never the profile's nominal number
    compile_enabled: bool = False  # torch.compile stays off until P2.8
    profile: str


# --------------------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------------------


def resolve(profile: HardwareConfig) -> DeviceContext:
    """Detect the device the profile names and check that it can honour the profile's dtype policy.

    Raises ``RuntimeError`` when the device is unavailable, ``ValueError`` when the profile asks for
    something the device cannot do (bf16 autocast without bf16 support, autocast on the CPU, fp64 on
    MPS, a non-unified MPS/CPU profile) and ``NotImplementedError`` for ``compile_mode != none``.
    """
    if profile.compile_mode != "none":
        raise NotImplementedError(
            f"profile {profile.name!r} sets compile_mode={profile.compile_mode!r}; torch.compile "
            "stays off until P2.8 (set hardware.compile_mode=none)"
        )
    if profile.device == "cuda":
        device, supports_bf16, vram_bytes = _resolve_cuda(profile)
    elif profile.device == "mps":
        device, supports_bf16, vram_bytes = _resolve_mps(profile)
    elif profile.device == "cpu":
        device, supports_bf16, vram_bytes = _resolve_cpu()
    else:  # pragma: no cover - HardwareConfig.device is a Literal
        raise ValueError(f"profile {profile.name!r} names an unknown device {profile.device!r}")

    autocast_dtype = _AUTOCAST_DTYPES[profile.autocast_dtype]
    if device.type == "cpu" and autocast_dtype is not None:
        raise ValueError(
            f"profile {profile.name!r} asks for autocast_dtype={profile.autocast_dtype!r} on the "
            "CPU; the CPU runs fp32 without autocast (plan §3.7)"
        )
    if autocast_dtype is torch.bfloat16 and not supports_bf16:
        raise ValueError(
            f"profile {profile.name!r} asks for bf16 autocast, but {device} does not support bf16 "
            f"({_device_description(device)}); set hardware.autocast_dtype=fp16"
        )
    master_dtype = _MASTER_DTYPES[profile.master_dtype]
    if device.type == "mps" and master_dtype is torch.float64:
        raise ValueError(
            f"profile {profile.name!r} asks for fp64 master weights on MPS, which has no float64; "
            "fp64 gradchecks run on profile:cpu (plan §3.7)"
        )
    if device.type in ("mps", "cpu") and not profile.unified_memory:
        raise ValueError(
            f"profile {profile.name!r} sets unified_memory=false, but {device.type} shares host "
            "memory; set hardware.unified_memory=true"
        )
    return DeviceContext(
        device=device,
        autocast_dtype=autocast_dtype,
        master_dtype=master_dtype,
        supports_bf16=supports_bf16,
        unified_memory=profile.unified_memory,
        vram_bytes=vram_bytes,
        compile_enabled=False,
        profile=profile.name,
    )


def _resolve_cuda(profile: HardwareConfig) -> tuple[torch.device, bool, int]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"profile {profile.name!r} needs device cuda, but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}, CUDA build {torch.version.cuda or 'none'}, "
            f"{platform.platform()}); install a CUDA wheel and driver or use profile:cpu"
        )
    device = torch.device("cuda", torch.cuda.current_device())
    supports_bf16 = torch.cuda.get_device_capability(device) >= (8, 0)  # Ampere and newer
    _free, total = torch.cuda.mem_get_info(device)
    return device, supports_bf16, int(total)


def _resolve_mps(profile: HardwareConfig) -> tuple[torch.device, bool, int]:
    if not torch.backends.mps.is_built():
        raise RuntimeError(
            f"profile {profile.name!r} needs device mps, but this torch wheel "
            f"({torch.__version__}) was built without the MPS backend"
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            f"profile {profile.name!r} needs device mps: the backend is built but unavailable on "
            f"{platform.platform()}; MPS needs macOS 15 or newer and a Metal-capable device"
        )
    device = torch.device("mps")
    vram_bytes = int(torch.mps.recommended_max_memory())  # Metal recommendedMaxWorkingSetSize
    return device, _bf16_autocast_works(device), vram_bytes


def _resolve_cpu() -> tuple[torch.device, bool, int]:
    # bf16 tensors compute on the CPU (slowly); the policy still runs the CPU in fp32.
    return torch.device("cpu"), True, _host_memory_bytes()


def _bf16_autocast_works(device: torch.device) -> bool:
    """Measured: a tiny matmul runs forward and backward under bf16 autocast on ``device``."""
    try:
        weight = torch.ones(4, 4, device=device, requires_grad=True)
        x = torch.ones(2, 4, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            out = x @ weight
        if out.dtype is not torch.bfloat16:
            return False
        (grad,) = torch.autograd.grad(out.float().sum(), weight)
        synchronize(device)
    except (RuntimeError, TypeError, NotImplementedError):
        return False
    return grad.shape == weight.shape


def _host_memory_bytes() -> int:
    names = os.sysconf_names
    if "SC_PHYS_PAGES" not in names or "SC_PAGE_SIZE" not in names:
        raise RuntimeError(
            f"cannot measure host memory on {sys.platform}: os.sysconf lacks SC_PHYS_PAGES/"
            "SC_PAGE_SIZE"
        )
    pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
    if pages <= 0 or page_size <= 0:
        raise RuntimeError(f"os.sysconf reported {pages} pages of {page_size} bytes")
    return pages * page_size


def _device_description(device: torch.device) -> str:
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return f"{torch.cuda.get_device_name(device)}, compute capability {major}.{minor}"
    return platform.platform()


# --------------------------------------------------------------------------------------------------
# Seeding and determinism
# --------------------------------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed Python's ``random``, NumPy, torch (CPU), every CUDA device and MPS from one integer.

    ``PYTHONHASHSEED`` is left alone: it only takes effect when an interpreter starts. The seed must
    lie in ``[0, 2**32)`` because NumPy's seeding does.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < _SEED_LIMIT:
        raise ValueError(f"seed must be an int in [0, {_SEED_LIMIT}), got {seed!r}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def set_deterministic(flag: bool) -> None:
    """Turn deterministic algorithms on or off everywhere torch exposes a switch.

    ``True`` enables ``torch.use_deterministic_algorithms``, deterministic cuDNN (benchmark off) and
    exports ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` unless the environment already sets it. cuBLAS reads
    that variable when it creates its workspace, so call this before the first CUDA matmul; a
    warning is issued when CUDA is already initialised. ``False`` reverses the torch switches and
    removes the variable only if this function exported it.
    """
    global _cublas_env_set_here
    torch.use_deterministic_algorithms(flag)
    torch.backends.cudnn.deterministic = flag
    if flag:
        torch.backends.cudnn.benchmark = False
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            if torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
                warnings.warn(
                    "CUBLAS_WORKSPACE_CONFIG is being set after CUDA was initialised; cuBLAS "
                    "workspaces created before this call stay non-deterministic",
                    RuntimeWarning,
                    stacklevel=2,
                )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
            _cublas_env_set_here = True
    elif _cublas_env_set_here:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        _cublas_env_set_here = False


# --------------------------------------------------------------------------------------------------
# Memory, caps and synchronisation
# --------------------------------------------------------------------------------------------------


def memory_stats(device: torch.device | str) -> dict[str, int]:
    """``{"allocated", "reserved", "peak"}`` in bytes for ``device`` (see the module docstring)."""
    dev = _as_device(device)
    if dev.type == "cuda":
        _require_cuda(dev)
        return {
            "allocated": int(torch.cuda.memory_allocated(dev)),
            "reserved": int(torch.cuda.memory_reserved(dev)),
            "peak": int(torch.cuda.max_memory_allocated(dev)),
        }
    if dev.type == "mps":
        _require_mps()
        allocated = int(torch.mps.current_allocated_memory())
        reserved = int(torch.mps.driver_allocated_memory())
    else:
        allocated = reserved = _resident_set_bytes()
    peak = _sampled_peak[dev.type] = max(_sampled_peak.get(dev.type, 0), allocated)
    return {"allocated": allocated, "reserved": reserved, "peak": peak}


def reset_peak(device: torch.device | str) -> None:
    """Reset ``peak`` to the current allocation (the CUDA counter, or the sampled mark)."""
    dev = _as_device(device)
    if dev.type == "cuda":
        _require_cuda(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        return
    if dev.type == "mps":
        _require_mps()
        allocated = int(torch.mps.current_allocated_memory())
    else:
        allocated = _resident_set_bytes()
    _sampled_peak[dev.type] = allocated


def set_vram_cap(device: torch.device | str, fraction: float) -> None:
    """Cap this process's allocations on ``device`` at ``fraction`` of its memory; no-op on the CPU.

    ``fraction`` must satisfy ``0 < fraction <= 1``: on MPS a fraction of 0 would mean *unlimited*
    and values up to 2 are accepted by PyTorch, neither of which is a cap. Allocating beyond the cap
    raises an out-of-memory ``RuntimeError`` (``torch.OutOfMemoryError`` on CUDA).
    """
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, int | float)
        or not 0.0 < fraction <= 1.0
    ):
        raise ValueError(
            f"fraction must satisfy 0 < fraction <= 1, got {fraction!r} (0 would mean unlimited on "
            "MPS; 1.0 removes the cap)"
        )
    dev = _as_device(device)
    if dev.type == "cuda":
        _require_cuda(dev)
        # Unlike memory_allocated & co., set_per_process_memory_fraction rejects an index-less
        # torch.device("cuda") (torch 2.15), so a bare "cuda" means the current device here too.
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(float(fraction), index)
    elif dev.type == "mps":
        _require_mps()
        torch.mps.set_per_process_memory_fraction(float(fraction))
    # cpu: host RAM has no per-process cap here (chunk P0.3: no-op on the CPU)


def synchronize(device: torch.device | str) -> None:
    """Wait for all queued work on ``device``; the CPU is synchronous already."""
    dev = _as_device(device)
    if dev.type == "cuda":
        _require_cuda(dev)
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        _require_mps()
        torch.mps.synchronize()


def _as_device(device: torch.device | str) -> torch.device:
    dev = torch.device(device)
    if dev.type not in _DEVICE_TYPES:
        raise ValueError(f"unsupported device {dev}; expected one of {_DEVICE_TYPES}")
    return dev


def _require_cuda(device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"{device} is not available: torch.cuda.is_available() is False")


def _require_mps() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "mps is not available: torch.backends.mps.is_available() is False "
            f"(is_built={torch.backends.mps.is_built()})"
        )


# --------------------------------------------------------------------------------------------------
# Resident set size (CPU statistics)
# --------------------------------------------------------------------------------------------------


class _MachTaskBasicInfo(ctypes.Structure):
    """``mach_task_basic_info`` from ``<mach/task_info.h>``."""

    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time_seconds", ctypes.c_int32),
        ("user_time_microseconds", ctypes.c_int32),
        ("system_time_seconds", ctypes.c_int32),
        ("system_time_microseconds", ctypes.c_int32),
        ("policy", ctypes.c_int32),
        ("suspend_count", ctypes.c_int32),
    ]


_MACH_TASK_BASIC_INFO = 20  # flavor constant from <mach/task_info.h>


def _resident_set_bytes() -> int:
    """Resident set size of this process in bytes (Linux: /proc; macOS: Mach task_info)."""
    if sys.platform == "linux":
        with open("/proc/self/statm", encoding="ascii") as handle:
            fields = handle.read().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    if sys.platform == "darwin":
        return _darwin_resident_bytes()
    raise NotImplementedError(f"resident set size is not implemented on {sys.platform}")


def _darwin_resident_bytes() -> int:
    libsystem = ctypes.CDLL(None, use_errno=True)  # the running process links libSystem
    libsystem.mach_task_self.restype = ctypes.c_uint32
    libsystem.mach_task_self.argtypes = []
    libsystem.task_info.restype = ctypes.c_int32
    libsystem.task_info.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    info = _MachTaskBasicInfo()
    count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
    status = libsystem.task_info(
        libsystem.mach_task_self(), _MACH_TASK_BASIC_INFO, ctypes.byref(info), ctypes.byref(count)
    )
    if status != 0:
        raise RuntimeError(f"task_info(MACH_TASK_BASIC_INFO) failed with kern_return_t {status}")
    return int(info.resident_size)
