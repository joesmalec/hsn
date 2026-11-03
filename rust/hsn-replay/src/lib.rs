//! `hsn-replay`: disk-backed replay reservoir with provenance and near-duplicate control (P3.1).
//!
//! P0.1: crate skeleton only.

/// Crate name, for manifests and logs.
pub const CRATE: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    #[test]
    fn test_p0_1_hsn_replay_links() {
        assert_eq!(super::CRATE, "hsn-replay");
        assert_eq!(hsn_core::VERSION, env!("CARGO_PKG_VERSION"));
    }
}
