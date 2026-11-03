//! `hsn-kernels`: the `Backend` trait, the CPU oracle, and the CUDA / Metal hosts that load the
//! compiled kernel binaries (P2.9-P2.12, P5.2).
//!
//! Cargo features: `cpu` (default), `cuda`, `metal`. The GPU features are empty for now and must
//! stay off by default; feature unification across the workspace must never turn them on silently.
//!
//! P0.1: only the backend name list exists.

/// Names of the compiled kernel backends, in priority order. `["cpu"]` until GPU features exist.
pub fn backend_names() -> Vec<&'static str> {
    let backends = [
        ("cpu", cfg!(feature = "cpu")),
        ("cuda", cfg!(feature = "cuda")),
        ("metal", cfg!(feature = "metal")),
    ];
    backends
        .iter()
        .filter(|(_, enabled)| *enabled)
        .map(|(name, _)| *name)
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_p0_1_default_backend_is_cpu() {
        assert_eq!(super::backend_names(), ["cpu"]);
    }
}
