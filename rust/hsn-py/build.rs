fn main() {
    // `cargo test` binaries for this crate embed libpython. uv's managed CPython keeps that library
    // outside the dynamic loader's default search path, so bake an rpath to its LIBDIR. This is a
    // no-op when maturin builds the extension module (PYO3_BUILD_EXTENSION_MODULE=1).
    pyo3_build_config::add_libpython_rpath_link_args();
}
