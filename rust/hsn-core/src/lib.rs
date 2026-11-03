//! `hsn-core`: identifiers, hashes, manifests, the config mirror and error types shared by every
//! other crate. It holds no tensors and depends only on serde, serde_json, thiserror and blake3.
//!
//! P0.1: crate skeleton; only the workspace version is exposed.

/// Version of the Rust workspace. Must equal the Python package version (`hsn.__version__`);
/// `tests/test_p0_1_smoke.py` enforces the equality.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn test_p0_1_version_is_semver_triple() {
        let parts: Vec<&str> = super::VERSION.split('.').collect();
        assert_eq!(
            parts.len(),
            3,
            "version {} is not MAJOR.MINOR.PATCH",
            super::VERSION
        );
        assert!(parts.iter().all(|p| p.parse::<u32>().is_ok()));
    }
}
