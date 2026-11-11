"""Suite-wide fixtures.

A trainer applies ``train.deterministic`` to the whole process
(:func:`hsn.device.set_deterministic`: ``torch.use_deterministic_algorithms``, the cuDNN switches
and ``CUBLAS_WORKSPACE_CONFIG``), and the tiny test configurations turn it on. A test that built
such a trainer used to leave the switches on for every test after it unless its file asked for
``restore_determinism``; whether a later test then passed depended on the order and on which
tests a machine skips. With the CUDA tests skipped, ``test_checkpoint_capture`` left them on and
``test_p0_3_set_deterministic_roundtrip`` failed. The fixture below puts the switches back after
every test, whatever the test did.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
import torch
from hsn.device import set_deterministic

_ENV = "CUBLAS_WORKSPACE_CONFIG"


@pytest.fixture(scope="session")
def _determinism_baseline() -> tuple[bool, bool, bool, str | None]:
    """The process's switches before any test ran."""
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        os.environ.get(_ENV),
    )


@pytest.fixture(autouse=True)
def _determinism_restored(
    _determinism_baseline: tuple[bool, bool, bool, str | None],
) -> Iterator[None]:
    """After every test the determinism switches are the session's again (module docstring)."""
    yield
    flag, cudnn_deterministic, cudnn_benchmark, env = _determinism_baseline
    # set_deterministic keeps its own record of whether it exported the variable; let it undo
    # (or redo) its part first, then pin every switch to the baseline
    set_deterministic(flag)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if env is None:
        os.environ.pop(_ENV, None)
    else:
        os.environ[_ENV] = env
