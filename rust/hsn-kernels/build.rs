use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=HSN_CUDA_FATBIN");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let path = std::env::var_os("HSN_CUDA_FATBIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
                .join("../../build/hsn_ops.fatbin")
        });
    println!("cargo:rerun-if-changed={}", path.display());
    // Locating is enough at build time; device-required tests refuse a missing binary.
    println!("cargo:rustc-env=HSN_DEFAULT_CUDA_FATBIN={}", path.display());
}
