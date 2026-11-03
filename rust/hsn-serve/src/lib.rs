//! `hsn-serve`: the inference runtime and HTTP service (P5.2-P5.4). It never trains and never
//! writes weights.
//!
//! P0.1: crate skeleton only.

/// Crate name, for manifests and logs.
pub const CRATE: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    #[test]
    fn test_p0_1_hsn_serve_links() {
        assert_eq!(super::CRATE, "hsn-serve");
        assert_eq!(hsn_core::VERSION, env!("CARGO_PKG_VERSION"));
    }
}
