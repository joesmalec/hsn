"""Pure stack attribution rules shared by eager and compiled profiling.

No torch import or device access: synthetic stacks can check segment consistency
without running a profiler or initializing a GPU.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_RELATIVE = "hsn/"

# The model's own glue, when no inner frame names a more specific segment.
_HSN_FUNCTIONS = {
    "_position": "shared_path",
    "_window_prologue": "shared_path",
    "_shared_path": "shared_path",
    "_score": "scorer",
    "_select": "scorer",
    "_update": "messages",
    "_sparse_round": "messages",
    "_masked_round": "messages",
    "_write_dense": "messages",
    "_cell_dense": "messages",
}

# These ops share a file, so the filename alone cannot identify their segment.
# Include adapters, raw custom ops and backwards: compiled provenance may carry
# the adapter's stack, while eager calls/backwards can expose the raw op directly.
_KERNEL_FUNCTIONS = {
    "grouped_unit_mlp": "selected_cells",
    "mlp_op": "selected_cells",
    "mlp_backward": "selected_cells",
    "_mlp_grad": "selected_cells",
    "gather_concat": "messages",
    "gather_op": "messages",
    "gather_backward": "messages",
    "_gather_grad": "messages",
    "csr_messages": "messages",
    "messages_op": "messages",
    "messages_backward": "messages",
    "_messages_grad": "messages",
    "gated_delta_update": "messages",
    "update_op": "messages",
    "update_inplace": "messages",
    "update_backward": "messages",
    "_update_grad": "messages",
}


def in_package(path: str, package_root: str) -> bool:
    """Match the imported package, excluding similarly named files in its .venv."""
    norm = path.replace("\\", "/")
    return norm.startswith(package_root) or norm.startswith(PACKAGE_RELATIVE)


def classify_frames(frames: list[tuple[str, int, str]], package_root: str) -> str | None:
    """Classify outermost-first frames by the innermost recognized HSN operation.

    Shared helpers such as kernels/ops.py:_run inherit their calling operation's segment.
    Encoder MLPs retain the router/decoder context rules of the native L1 baseline.
    """
    files = [
        (Path(path.replace("\\", "/")).name, fn)
        for path, _, fn in frames
        if in_package(path, package_root)
    ]
    if not files:
        return None
    names = {name for name, _ in files}
    for name, fn in reversed(files):
        match name:
            case "units.py":
                return "selected_cells"
            case "ops.py" if fn in _KERNEL_FUNCTIONS:
                return _KERNEL_FUNCTIONS[fn]
            case "cell.py" | "graph.py" | "native.py" | "state.py":
                return "messages"
            case "router.py":
                return "scorer"
            case "losses.py":
                return "loss"
            case "encoder.py" | "embeddings.py":
                if "router.py" in names:
                    return "scorer"
                return "decode" if "_window_epilogue" in {f for _, f in files} else "shared_path"
            case "hsn.py" if fn in _HSN_FUNCTIONS:
                return _HSN_FUNCTIONS[fn]
    return "other"
