"""P0.1 smoke test: the Python package and the Rust extension import and agree on the version."""

from importlib.metadata import version

import pytest


def test_p0_1_smoke_imports() -> None:
    import hsn

    try:
        import hsn_rs
    except ImportError:  # pragma: no cover - only on a half-built environment
        pytest.fail(
            "hsn_rs is not installed in this environment; run: "
            "uv run maturin develop --uv -m rust/hsn-py/Cargo.toml"
        )

    assert hsn.__version__ == version("hsn")
    assert hsn_rs.version() == hsn.__version__
    assert hsn_rs.backend_names() == ["cpu"]
