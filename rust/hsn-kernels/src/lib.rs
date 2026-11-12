//! `hsn-kernels`: the `Backend` trait, the CPU oracle, and the CUDA / Metal hosts that load the
//! compiled kernel binaries (P2.9-P2.12, P5.2).
//!
//! Cargo features: `cpu` (default), `cuda`, `metal`. GPU features stay off by default;
//! feature unification across the workspace must never turn them on silently.
//!
//! P2.9 adds the specification of the four HSN ops ([`spec`], mirroring the C header
//! `kernels/include/hsn_ops.h` and `python/hsn/kernels/spec.py`), the pure-Rust CPU oracle
//! ([`CpuBackend`], f32 and f64) and the test vectors generated from it ([`vectors`], committed in
//! `kernels/testvectors/`). The [`Backend`] trait mirrors the C ABI one to one: the same eight
//! entry points (four forward, four backward), the same argument names, one float dtype per call
//! carried at run time by the [`TensorRef`] / [`TensorMut`] views (the ABI's `void*` plus
//! `dtype`), and an [`Error`] whose [`Error::status`] is the ABI's return code.

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod spec;
pub mod vectors;

pub use cpu::CpuBackend;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;
pub use spec::{
    CellDims, CsrMessages, CsrMessagesBwd, GatedDeltaUpdate, GatedDeltaUpdateBwd, GatherConcat,
    GatherConcatBwd, GroupedUnitMlpBwd, GroupedUnitMlpFwd, UnitDims,
};

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

/// A float dtype. `F32`, `F16` and `BF16` are the C ABI's `HSN_DTYPE_*`; `F64` exists only in
/// Rust, for the oracle's finite-difference checks, and has no ABI code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    F64,
}

impl DType {
    /// The `HSN_DTYPE_*` code of the C ABI (`None` for `F64`, which the ABI does not carry).
    pub fn abi_code(self) -> Option<i32> {
        match self {
            DType::F32 => Some(spec::DTYPE_F32),
            DType::F16 => Some(spec::DTYPE_F16),
            DType::BF16 => Some(spec::DTYPE_BF16),
            DType::F64 => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F64 => "f64",
        }
    }
}

/// A read-only float tensor of one call: a contiguous row-major host slice tagged with its dtype.
/// `F16` and `BF16` carry the raw IEEE / bfloat16 bit patterns (no `half` dependency); the CPU
/// oracle refuses them with [`Error::UnsupportedDType`].
#[derive(Clone, Copy, Debug)]
pub enum TensorRef<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
    F16(&'a [u16]),
    BF16(&'a [u16]),
}

/// A writable float tensor of one call (an output, or an in-place input).
#[derive(Debug)]
pub enum TensorMut<'a> {
    F32(&'a mut [f32]),
    F64(&'a mut [f64]),
    F16(&'a mut [u16]),
    BF16(&'a mut [u16]),
}

impl TensorRef<'_> {
    pub fn dtype(&self) -> DType {
        match self {
            TensorRef::F32(_) => DType::F32,
            TensorRef::F64(_) => DType::F64,
            TensorRef::F16(_) => DType::F16,
            TensorRef::BF16(_) => DType::BF16,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TensorRef::F32(s) => s.len(),
            TensorRef::F64(s) => s.len(),
            TensorRef::F16(s) | TensorRef::BF16(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl TensorMut<'_> {
    pub fn dtype(&self) -> DType {
        match self {
            TensorMut::F32(_) => DType::F32,
            TensorMut::F64(_) => DType::F64,
            TensorMut::F16(_) => DType::F16,
            TensorMut::BF16(_) => DType::BF16,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TensorMut::F32(s) => s.len(),
            TensorMut::F64(s) => s.len(),
            TensorMut::F16(s) | TensorMut::BF16(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Why an op refused its arguments or failed. [`Error::status`] is the C ABI's `HSN_ERR_*` code.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// A dimension below its minimum, a buffer of the wrong length, a flag outside {0, 1}, or
    /// float tensors of different dtypes in one call.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// The backend does not implement this dtype.
    #[error("backend {backend} does not implement dtype {}", dtype.name())]
    UnsupportedDType { backend: &'static str, dtype: DType },
    /// A cell or unit index outside its range.
    #[error("index out of range: {0}")]
    IndexOutOfRange(String),
    /// A cell named by two valid slots of one row (gated delta update).
    #[error("duplicate index: {0}")]
    DuplicateIndex(String),
    /// A dimension or an element count above `i32::MAX`.
    #[error("size overflow: {0}")]
    SizeOverflow(String),
    /// `indptr` / `indices` are not a CSR adjacency over `N` cells.
    #[error("malformed CSR graph: {0}")]
    BadGraph(String),
    /// A launch, device or driver failure (GPU backends).
    #[error("backend failure: {0}")]
    Backend(String),
}

impl Error {
    /// The `HSN_ERR_*` return code of the C ABI.
    pub fn status(&self) -> i32 {
        match self {
            Error::InvalidArgument(_) => spec::ERR_INVALID_ARGUMENT,
            Error::UnsupportedDType { .. } => spec::ERR_UNSUPPORTED_DTYPE,
            Error::IndexOutOfRange(_) => spec::ERR_INDEX_OUT_OF_RANGE,
            Error::DuplicateIndex(_) => spec::ERR_DUPLICATE_INDEX,
            Error::SizeOverflow(_) => spec::ERR_SIZE_OVERFLOW,
            Error::BadGraph(_) => spec::ERR_BAD_GRAPH,
            Error::Backend(_) => spec::ERR_BACKEND,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// One implementation of the four HSN ops, forward and backward: the Rust mirror of the C ABI in
/// `kernels/include/hsn_ops.h`. Every method validates its arguments (see [`spec`]) and writes
/// its outputs in full; the semantics, shapes and tolerances are those of the header. The trait
/// is object safe, so a registry of `Box<dyn Backend>` can run every implementation on the same
/// vectors (P2.12).
pub trait Backend {
    /// A short, stable name ("cpu", "cuda", "metal").
    fn name(&self) -> &'static str;
    /// (a) `hsn_gather_concat`.
    fn gather_concat(&self, args: GatherConcat<'_>) -> Result<()>;
    /// Backward of (a), `hsn_gather_concat_bwd`.
    fn gather_concat_bwd(&self, args: GatherConcatBwd<'_>) -> Result<()>;
    /// (b) `hsn_grouped_unit_mlp_fwd`.
    fn grouped_unit_mlp_fwd(&self, args: GroupedUnitMlpFwd<'_>) -> Result<()>;
    /// Backward of (b), `hsn_grouped_unit_mlp_bwd`.
    fn grouped_unit_mlp_bwd(&self, args: GroupedUnitMlpBwd<'_>) -> Result<()>;
    /// (c) `hsn_csr_messages`.
    fn csr_messages(&self, args: CsrMessages<'_>) -> Result<()>;
    /// Backward of (c), `hsn_csr_messages_bwd`.
    fn csr_messages_bwd(&self, args: CsrMessagesBwd<'_>) -> Result<()>;
    /// (d) `hsn_gated_delta_update`, in place on `h`, `q` and `age`.
    fn gated_delta_update(&self, args: GatedDeltaUpdate<'_>) -> Result<()>;
    /// Backward of (d), `hsn_gated_delta_update_bwd`.
    fn gated_delta_update_bwd(&self, args: GatedDeltaUpdateBwd<'_>) -> Result<()>;
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_p0_1_default_backend_is_cpu() {
        let expected = [
            ("cpu", cfg!(feature = "cpu")),
            ("cuda", cfg!(feature = "cuda")),
            ("metal", cfg!(feature = "metal")),
        ];
        assert_eq!(
            super::backend_names(),
            expected
                .into_iter()
                .filter_map(|(name, enabled)| enabled.then_some(name))
                .collect::<Vec<_>>()
        );
    }
}
