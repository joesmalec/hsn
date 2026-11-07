"""The CUDA hardware profile of the card the tests run on (card-agnostic CUDA tests, P2.8).

A CUDA test never names a card: it takes :data:`CUDA_PROFILE`, the shipped profile whose card this
machine has (by compute capability: 8.6 ``rtx3060``, 8.0 / 9.0 ``a100``, 12.1 ``dgx_spark``),
``rtx3060`` as the generic CUDA profile for any other card, or ``HSN_TEST_CUDA_PROFILE`` when set.
The capability is read through NVML, so collecting the tests never opens a CUDA context, and a run
with the devices hidden (``CUDA_VISIBLE_DEVICES=``) never looks. Tests also never depend on a
profile's ``compile_mode`` default: every CUDA test that trains pins it.
"""

from __future__ import annotations

import os

#: compute capability -> the shipped profile of that card (configs/hardware)
CARD_PROFILES: dict[tuple[int, int], str] = {
    (8, 0): "a100",
    (8, 6): "rtx3060",
    (9, 0): "a100",
    (12, 1): "dgx_spark",
}
#: the profile for a CUDA card without its own (any Ampere-or-newer card: bf16 autocast, fp32)
GENERIC_CUDA_PROFILE = "rtx3060"


def cuda_profile() -> str:
    """The CUDA profile for this machine's first visible card (module docstring)."""
    named = os.environ.get("HSN_TEST_CUDA_PROFILE")
    if named:
        return named
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return GENERIC_CUDA_PROFILE
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            capability = tuple(pynvml.nvmlDeviceGetCudaComputeCapability(handle))
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - no driver, no NVML: the CUDA tests skip anyway
        return GENERIC_CUDA_PROFILE
    return CARD_PROFILES.get((int(capability[0]), int(capability[1])), GENERIC_CUDA_PROFILE)


CUDA_PROFILE = cuda_profile()
