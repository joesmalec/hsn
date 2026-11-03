//! `hsn-registry`: release registry, signing and verification of release bundles (P5.1).
//!
//! P0.1: crate skeleton only.

/// Crate name, for manifests and logs.
pub const CRATE: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    #[test]
    fn test_p0_1_hsn_registry_links() {
        assert_eq!(super::CRATE, "hsn-registry");
        assert_eq!(hsn_core::VERSION, env!("CARGO_PKG_VERSION"));
    }
}
