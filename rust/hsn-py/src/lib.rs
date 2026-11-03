//! `hsn_rs`: the Python-facing surface of the Rust workspace.
//!
//! P0.1 exposes `version()` and `backend_names()`. Later chunks add `ReplayReservoir`,
//! `ShardLoader`, `TieredStore` and the kernel-parity helpers.

use pyo3::prelude::*;

/// Workspace version; must equal `hsn.__version__` (checked by `tests/test_p0_1_smoke.py`).
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Names of the compiled kernel backends (`["cpu"]` for now).
#[pyfunction]
fn backend_names() -> Vec<String> {
    hsn_kernels::backend_names()
        .into_iter()
        .map(str::to_owned)
        .collect()
}

#[pymodule]
mod hsn_rs {
    #[pymodule_export]
    use super::{backend_names, version};
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_p0_1_version_matches_crate() {
        assert_eq!(super::version(), env!("CARGO_PKG_VERSION"));
        assert_eq!(super::backend_names(), ["cpu"]);
    }
}
