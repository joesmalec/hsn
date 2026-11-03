//! `hsn-stream`: memory-mapped shard loader, tiers, residency and prefetch (P4.3, P4.4).
//!
//! P0.1: crate skeleton only.

/// Crate name, for manifests and logs.
pub const CRATE: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    #[test]
    fn test_p0_1_hsn_stream_links() {
        assert_eq!(super::CRATE, "hsn-stream");
        assert_eq!(hsn_core::VERSION, env!("CARGO_PKG_VERSION"));
    }
}
