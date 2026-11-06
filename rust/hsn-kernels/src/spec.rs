//! The specification of the four HSN ops (P2.9): constants, tolerances, test-vector sizes, the
//! tensors of every op, the argument structs of the [`Backend`](crate::Backend) trait and their
//! validation. The normative text is the C header `kernels/include/hsn_ops.h`; this module and
//! `python/hsn/kernels/spec.py` mirror it, and tests check that the constants of all three agree.
//!
//! # The ops
//!
//! `r` indexes batch rows, so that `b` always names the gate tensor. `P = B·K`.
//!
//! ```text
//! (a) gather_concat     v[r,k] = [u[r]; eps[r]; h[r, cell_idx[r,k]]; m[r,k]]            [B,K,4d]
//! (b) grouped_unit_mlp  j = unit_idx[p];  z1 = v[p] W1[j] + b1[j]  (saved, pre-GELU)
//!                       out[p] = gelu(z1) W2[j] + b2[j];  gelu(x) = x/2 (1 + erf(x/√2))  [P,d]
//! (c) csr_messages      i = cell_idx[r,k];  m[r,k] = Σ_{e ∈ row i} a[e] · V h[r, indices[e]]  [B,K,d]
//! (d) gated_delta_update, in place; for valid slots, n = cell_idx[r,k]:
//!                       h_old = h[r,n];  h[r,n] ← (1−b)⊙h_old + b⊙c
//!                       q[r] ← q[r] + (Σ_valid (h_new − h_old)) / N;  age[r,n] ← 0
//! ```
//!
//! # Backward
//!
//! ```text
//! (a) grad_u[r] = Σ_k dv[r,k,0:d];  grad_eps[r] = Σ_k dv[r,k,d:2d]
//!     grad_h[r,n] = Σ_{k: cell_idx[r,k]=n} dv[r,k,2d:3d];  grad_m[r,k] = dv[r,k,3d:4d]
//! (b) a = gelu(z1);  dz1 = (dout W2[j]ᵀ) ⊙ gelu'(z1),  gelu'(x) = Φ(x) + x φ(x)
//!     grad_v = dz1 W1[j]ᵀ;  grad_W1[j] = Σ_{p∈j} vᵀ dz1;  grad_b1[j] = Σ_{p∈j} dz1
//!     grad_W2[j] = Σ_{p∈j} aᵀ dout;  grad_b2[j] = Σ_{p∈j} dout;  absent units → 0
//! (c) for every slot (r,k), edge e of row cell_idx[r,k], source j = indices[e]:
//!     grad_h[r,j] += a[e] Vᵀ dm[r,k];  grad_V += a[e] dm[r,k] ⊗ h[r,j];  grad_a[e] += dm[r,k]·(V h[r,j])
//! (d) G = ∂L/∂h_out, g = ∂L/∂q_out; valid slot k, n = cell_idx[r,k]:
//!     grad_h[r,n] = (1−b)⊙G[r,n] − b⊙g[r]/N      (grad_h = G on cells no valid slot selects)
//!     grad_q = g;  grad_c = b⊙(G[r,n] + g[r]/N);  grad_b = (c − h_old)⊙(G[r,n] + g[r]/N)
//!     padded slots: grad_c = grad_b = 0
//! ```
//!
//! The `g/N` terms in (d) are the summary coupling (the chunk's pitfall): `q` is maintained as
//! `q + Σ Δh / N`, so the gradient of `q` reaches every selected `h`, `c` and `b`.
//!
//! # Where the ops sit in the HSN round
//!
//! `v_raw = gather_concat(…)`; `v = RMSNorm(v_raw)` (amendment A1); `F = grouped_unit_mlp(v, …)`;
//! `b = σ(gate(v))`, `c = tanh(W_out(α·F))`; `gated_delta_update(h, …, c, b)` on the raw `h`. The
//! RMSNorm, the gate, `W_out`, `tanh` and the α mix are host ops, not part of the four. Changing
//! what an op computes is a new [`SPEC_VERSION`] and a regeneration of the vectors.

use std::collections::BTreeMap;

use crate::{DType, Error, Result, TensorMut, TensorRef};

/// `HSN_OPS_SPEC_VERSION`: recorded in `kernels/testvectors/MANIFEST.json`.
pub const SPEC_VERSION: u32 = 1;

/// `HSN_DTYPE_*`.
pub const DTYPE_F32: i32 = 0;
pub const DTYPE_F16: i32 = 1;
pub const DTYPE_BF16: i32 = 2;

/// `HSN_OK` and `HSN_ERR_*`.
pub const OK: i32 = 0;
pub const ERR_INVALID_ARGUMENT: i32 = 1;
pub const ERR_UNSUPPORTED_DTYPE: i32 = 2;
pub const ERR_INDEX_OUT_OF_RANGE: i32 = 3;
pub const ERR_DUPLICATE_INDEX: i32 = 4;
pub const ERR_SIZE_OVERFLOW: i32 = 5;
pub const ERR_BAD_GRAPH: i32 = 6;
pub const ERR_BACKEND: i32 = 7;

/// Every `#define` of the header with its value, for the cross-check against the header text.
pub const HEADER_CONSTANTS: &[(&str, i64)] = &[
    ("HSN_OPS_SPEC_VERSION", SPEC_VERSION as i64),
    ("HSN_DTYPE_F32", DTYPE_F32 as i64),
    ("HSN_DTYPE_F16", DTYPE_F16 as i64),
    ("HSN_DTYPE_BF16", DTYPE_BF16 as i64),
    ("HSN_OK", OK as i64),
    ("HSN_ERR_INVALID_ARGUMENT", ERR_INVALID_ARGUMENT as i64),
    ("HSN_ERR_UNSUPPORTED_DTYPE", ERR_UNSUPPORTED_DTYPE as i64),
    ("HSN_ERR_INDEX_OUT_OF_RANGE", ERR_INDEX_OUT_OF_RANGE as i64),
    ("HSN_ERR_DUPLICATE_INDEX", ERR_DUPLICATE_INDEX as i64),
    ("HSN_ERR_SIZE_OVERFLOW", ERR_SIZE_OVERFLOW as i64),
    ("HSN_ERR_BAD_GRAPH", ERR_BAD_GRAPH as i64),
    ("HSN_ERR_BACKEND", ERR_BACKEND as i64),
];

/// Elementwise agreement `|x − ref| ≤ atol + rtol·|ref|` (the `torch.allclose` rule).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    pub atol: f64,
    pub rtol: f64,
}

impl Tolerance {
    /// True when `actual` agrees with `expected`; false for any NaN.
    pub fn allows(&self, actual: f64, expected: f64) -> bool {
        (actual - expected).abs() <= self.atol + self.rtol * expected.abs()
    }

    /// `|x − ref| / (atol + rtol·|ref|)`: agreement is a ratio ≤ 1.
    pub fn ratio(&self, actual: f64, expected: f64) -> f64 {
        (actual - expected).abs() / (self.atol + self.rtol * expected.abs())
    }
}

/// fp32 implementations against the fp32 vectors.
pub const TOL_F32: Tolerance = Tolerance {
    atol: 1e-5,
    rtol: 1e-5,
};
/// fp16 implementations against the fp32 vectors.
pub const TOL_F16: Tolerance = Tolerance {
    atol: 1e-2,
    rtol: 1e-2,
};
/// bf16 implementations against the fp32 vectors.
pub const TOL_BF16: Tolerance = Tolerance {
    atol: 1e-2,
    rtol: 1e-2,
};

/// The tolerance of an implementation in `dtype` against the vectors (`None` for `F64`, which is
/// checked by finite differences instead).
pub fn tolerance(dtype: DType) -> Option<Tolerance> {
    match dtype {
        DType::F32 => Some(TOL_F32),
        DType::F16 => Some(TOL_F16),
        DType::BF16 => Some(TOL_BF16),
        DType::F64 => None,
    }
}

/// One test-vector size (the chunk's three).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size {
    pub name: &'static str,
    pub b: usize,
    pub n: usize,
    pub d: usize,
    pub k: usize,
    pub m: usize,
    pub h: usize,
}

impl Size {
    /// `P = B·K`.
    pub fn p(&self) -> usize {
        self.b * self.k
    }
}

pub const TINY: Size = Size {
    name: "tiny",
    b: 2,
    n: 3,
    d: 4,
    k: 2,
    m: 2,
    h: 8,
};
pub const REF: Size = Size {
    name: "ref",
    b: 4,
    n: 32,
    d: 64,
    k: 4,
    m: 8,
    h: 512,
};
pub const ODD: Size = Size {
    name: "odd",
    b: 3,
    n: 37,
    d: 96,
    k: 5,
    m: 5,
    h: 256,
};
pub const SIZES: [Size; 3] = [TINY, REF, ODD];

/// The four ops.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Op {
    GatherConcat,
    GroupedUnitMlp,
    CsrMessages,
    GatedDeltaUpdate,
}

impl Op {
    pub const ALL: [Op; 4] = [
        Op::GatherConcat,
        Op::GroupedUnitMlp,
        Op::CsrMessages,
        Op::GatedDeltaUpdate,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Op::GatherConcat => "gather_concat",
            Op::GroupedUnitMlp => "grouped_unit_mlp",
            Op::CsrMessages => "csr_messages",
            Op::GatedDeltaUpdate => "gated_delta_update",
        }
    }

    pub fn from_name(name: &str) -> Option<Op> {
        Op::ALL.into_iter().find(|op| op.name() == name)
    }

    /// The dimension symbols of the op's vectors, in the order the manifest lists them.
    pub fn dims(self) -> &'static [&'static str] {
        match self {
            Op::GatherConcat | Op::GatedDeltaUpdate => &["B", "N", "d", "K"],
            Op::GroupedUnitMlp => &["P", "M", "d", "H"],
            Op::CsrMessages => &["B", "N", "d", "K", "E"],
        }
    }

    /// Every tensor of the op's vectors (forward inputs, forward outputs, cotangents, gradients),
    /// named as the header's arguments.
    pub fn tensors(self) -> &'static [TensorSpec] {
        match self {
            Op::GatherConcat => GATHER_CONCAT,
            Op::GroupedUnitMlp => GROUPED_UNIT_MLP,
            Op::CsrMessages => CSR_MESSAGES,
            Op::GatedDeltaUpdate => GATED_DELTA_UPDATE,
        }
    }

    pub fn tensor(self, name: &str) -> Option<&'static TensorSpec> {
        self.tensors().iter().find(|t| t.name == name)
    }
}

/// What a tensor is in a vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// A forward input (for the in-place op, the value before the update).
    Input,
    /// A forward output, including the saved-for-backward tensors.
    Output,
    /// A backward seed: the gradient of a scalar loss with respect to a forward output.
    Cotangent,
    /// A backward output: the gradient with respect to the forward input `name[5..]`.
    Grad,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::Input => "input",
            Role::Output => "output",
            Role::Cotangent => "cotangent",
            Role::Grad => "grad",
        }
    }
}

/// The element type of a tensor in a vector: float (stored as f32) or int32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Float,
    Int32,
}

/// One tensor of an op: name, role, element kind and symbolic shape (`"B"`, `"4d"`, `"N+1"`, …).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: &'static str,
    pub role: Role,
    pub kind: Kind,
    pub shape: &'static [&'static str],
}

const fn t(
    name: &'static str,
    role: Role,
    kind: Kind,
    shape: &'static [&'static str],
) -> TensorSpec {
    TensorSpec {
        name,
        role,
        kind,
        shape,
    }
}

use Kind::{Float, Int32};
use Role::{Cotangent, Grad, Input, Output};

const GATHER_CONCAT: &[TensorSpec] = &[
    t("u", Input, Float, &["B", "d"]),
    t("eps", Input, Float, &["B", "d"]),
    t("h", Input, Float, &["B", "N", "d"]),
    t("m", Input, Float, &["B", "K", "d"]),
    t("cell_idx", Input, Int32, &["B", "K"]),
    t("v", Output, Float, &["B", "K", "4d"]),
    t("grad_v", Cotangent, Float, &["B", "K", "4d"]),
    t("grad_u", Grad, Float, &["B", "d"]),
    t("grad_eps", Grad, Float, &["B", "d"]),
    t("grad_h", Grad, Float, &["B", "N", "d"]),
    t("grad_m", Grad, Float, &["B", "K", "d"]),
];

const GROUPED_UNIT_MLP: &[TensorSpec] = &[
    t("v", Input, Float, &["P", "4d"]),
    t("unit_idx", Input, Int32, &["P"]),
    t("W1", Input, Float, &["M", "4d", "H"]),
    t("b1", Input, Float, &["M", "H"]),
    t("W2", Input, Float, &["M", "H", "d"]),
    t("b2", Input, Float, &["M", "d"]),
    t("hidden_saved", Output, Float, &["P", "H"]),
    t("out", Output, Float, &["P", "d"]),
    t("grad_out", Cotangent, Float, &["P", "d"]),
    t("grad_v", Grad, Float, &["P", "4d"]),
    t("grad_W1", Grad, Float, &["M", "4d", "H"]),
    t("grad_b1", Grad, Float, &["M", "H"]),
    t("grad_W2", Grad, Float, &["M", "H", "d"]),
    t("grad_b2", Grad, Float, &["M", "d"]),
];

const CSR_MESSAGES: &[TensorSpec] = &[
    t("h", Input, Float, &["B", "N", "d"]),
    t("indptr", Input, Int32, &["N+1"]),
    t("indices", Input, Int32, &["E"]),
    t("a", Input, Float, &["E"]),
    t("V", Input, Float, &["d", "d"]),
    t("cell_idx", Input, Int32, &["B", "K"]),
    t("m", Output, Float, &["B", "K", "d"]),
    t("grad_m", Cotangent, Float, &["B", "K", "d"]),
    t("grad_h", Grad, Float, &["B", "N", "d"]),
    t("grad_a", Grad, Float, &["E"]),
    t("grad_V", Grad, Float, &["d", "d"]),
];

const GATED_DELTA_UPDATE: &[TensorSpec] = &[
    t("h", Input, Float, &["B", "N", "d"]),
    t("q", Input, Float, &["B", "d"]),
    t("age", Input, Int32, &["B", "N"]),
    t("cell_idx", Input, Int32, &["B", "K"]),
    t("valid", Input, Int32, &["B", "K"]),
    t("c", Input, Float, &["B", "K", "d"]),
    t("b", Input, Float, &["B", "K", "d"]),
    t("h_out", Output, Float, &["B", "N", "d"]),
    t("q_out", Output, Float, &["B", "d"]),
    t("age_out", Output, Int32, &["B", "N"]),
    t("h_old_saved", Output, Float, &["B", "K", "d"]),
    t("grad_h_out", Cotangent, Float, &["B", "N", "d"]),
    t("grad_q_out", Cotangent, Float, &["B", "d"]),
    t("grad_h", Grad, Float, &["B", "N", "d"]),
    t("grad_q", Grad, Float, &["B", "d"]),
    t("grad_c", Grad, Float, &["B", "K", "d"]),
    t("grad_b", Grad, Float, &["B", "K", "d"]),
];

/// The concrete shape of `spec` for the dimension values `dims` (symbol → value). `"4d"` is
/// `4·d` and `"N+1"` is `N + 1`; an unknown symbol is an error.
pub fn resolve_shape(spec: &TensorSpec, dims: &BTreeMap<String, usize>) -> Result<Vec<usize>> {
    spec.shape
        .iter()
        .map(|symbol| {
            let lookup = |key: &str| {
                dims.get(key).copied().ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "tensor {} needs dimension {key}, which the vector does not define",
                        spec.name
                    ))
                })
            };
            match *symbol {
                "4d" => lookup("d").map(|d| 4 * d),
                "N+1" => lookup("N").map(|n| n + 1),
                other => lookup(other),
            }
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------
// Dimensions and argument structs of the Backend trait (the header's arguments, in its order)
// ------------------------------------------------------------------------------------------------

/// `B, N, d, K` of the cell ops (gather/concat, CSR messages, gated delta update).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellDims {
    pub b: usize,
    pub n: usize,
    pub d: usize,
    pub k: usize,
}

/// `P, M, d, H` of the grouped unit MLP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitDims {
    pub p: usize,
    pub m: usize,
    pub d: usize,
    pub h: usize,
}

/// (a) `hsn_gather_concat`.
#[derive(Debug)]
pub struct GatherConcat<'a> {
    pub u: TensorRef<'a>,
    pub eps: TensorRef<'a>,
    pub h: TensorRef<'a>,
    pub m: TensorRef<'a>,
    pub cell_idx: &'a [i32],
    pub v: TensorMut<'a>,
    pub dims: CellDims,
}

/// Backward of (a), `hsn_gather_concat_bwd`.
#[derive(Debug)]
pub struct GatherConcatBwd<'a> {
    pub grad_v: TensorRef<'a>,
    pub cell_idx: &'a [i32],
    pub grad_u: TensorMut<'a>,
    pub grad_eps: TensorMut<'a>,
    pub grad_h: TensorMut<'a>,
    pub grad_m: TensorMut<'a>,
    pub dims: CellDims,
    pub deterministic: bool,
}

/// (b) `hsn_grouped_unit_mlp_fwd`. `w1`, `w2` are the header's `W1`, `W2`.
#[derive(Debug)]
pub struct GroupedUnitMlpFwd<'a> {
    pub v: TensorRef<'a>,
    pub unit_idx: &'a [i32],
    pub w1: TensorRef<'a>,
    pub b1: TensorRef<'a>,
    pub w2: TensorRef<'a>,
    pub b2: TensorRef<'a>,
    pub hidden_saved: TensorMut<'a>,
    pub out: TensorMut<'a>,
    pub dims: UnitDims,
}

/// Backward of (b), `hsn_grouped_unit_mlp_bwd`.
#[derive(Debug)]
pub struct GroupedUnitMlpBwd<'a> {
    pub v: TensorRef<'a>,
    pub unit_idx: &'a [i32],
    pub w1: TensorRef<'a>,
    pub w2: TensorRef<'a>,
    pub hidden_saved: TensorRef<'a>,
    pub grad_out: TensorRef<'a>,
    pub grad_v: TensorMut<'a>,
    pub grad_w1: TensorMut<'a>,
    pub grad_b1: TensorMut<'a>,
    pub grad_w2: TensorMut<'a>,
    pub grad_b2: TensorMut<'a>,
    pub dims: UnitDims,
    pub deterministic: bool,
}

/// (c) `hsn_csr_messages`. `v_proj` is the header's `V` (`[d_out, d_in]`).
#[derive(Debug)]
pub struct CsrMessages<'a> {
    pub h: TensorRef<'a>,
    pub indptr: &'a [i32],
    pub indices: &'a [i32],
    pub a: TensorRef<'a>,
    pub v_proj: TensorRef<'a>,
    pub cell_idx: &'a [i32],
    pub m: TensorMut<'a>,
    pub dims: CellDims,
    pub e: usize,
}

/// Backward of (c), `hsn_csr_messages_bwd`. `grad_v_proj` is the header's `grad_V`.
#[derive(Debug)]
pub struct CsrMessagesBwd<'a> {
    pub h: TensorRef<'a>,
    pub indptr: &'a [i32],
    pub indices: &'a [i32],
    pub a: TensorRef<'a>,
    pub v_proj: TensorRef<'a>,
    pub cell_idx: &'a [i32],
    pub grad_m: TensorRef<'a>,
    pub grad_h: TensorMut<'a>,
    pub grad_a: TensorMut<'a>,
    pub grad_v_proj: TensorMut<'a>,
    pub dims: CellDims,
    pub e: usize,
    pub deterministic: bool,
}

/// (d) `hsn_gated_delta_update`, in place on `h`, `q`, `age`.
#[derive(Debug)]
pub struct GatedDeltaUpdate<'a> {
    pub h: TensorMut<'a>,
    pub q: TensorMut<'a>,
    pub age: &'a mut [i32],
    pub cell_idx: &'a [i32],
    pub valid: &'a [i32],
    pub c: TensorRef<'a>,
    pub b: TensorRef<'a>,
    pub h_old_saved: TensorMut<'a>,
    pub dims: CellDims,
}

/// Backward of (d), `hsn_gated_delta_update_bwd`.
#[derive(Debug)]
pub struct GatedDeltaUpdateBwd<'a> {
    pub cell_idx: &'a [i32],
    pub valid: &'a [i32],
    pub c: TensorRef<'a>,
    pub b: TensorRef<'a>,
    pub h_old_saved: TensorRef<'a>,
    pub grad_h_out: TensorRef<'a>,
    pub grad_q_out: TensorRef<'a>,
    pub grad_h: TensorMut<'a>,
    pub grad_q: TensorMut<'a>,
    pub grad_c: TensorMut<'a>,
    pub grad_b: TensorMut<'a>,
    pub dims: CellDims,
}

// ------------------------------------------------------------------------------------------------
// Validation: the header's preconditions, checked in the order dims → dtype → lengths → indices
// ------------------------------------------------------------------------------------------------

const I32_MAX: usize = i32::MAX as usize;

fn check_dim(name: &str, value: usize, min: usize) -> Result<()> {
    if value < min {
        return Err(Error::InvalidArgument(format!(
            "{name} must be >= {min}, got {value}"
        )));
    }
    if value > I32_MAX {
        return Err(Error::SizeOverflow(format!(
            "{name} = {value} exceeds INT32_MAX"
        )));
    }
    Ok(())
}

/// The element count `Π factors`, refused above `i32::MAX` (the pitfall's "assert anyway").
pub fn element_count(name: &str, factors: &[usize]) -> Result<usize> {
    let mut count: usize = 1;
    for &factor in factors {
        count = count
            .checked_mul(factor)
            .filter(|&c| c <= I32_MAX)
            .ok_or_else(|| {
                Error::SizeOverflow(format!(
                    "{name} has {factors:?} elements, more than INT32_MAX"
                ))
            })?;
    }
    Ok(count)
}

impl CellDims {
    /// `B, N, d, K ≥ 1`; every dimension and the element counts of `h [B,N,d]` and
    /// `v [B,K,4d]` at most `i32::MAX`.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [("B", self.b), ("N", self.n), ("d", self.d), ("K", self.k)] {
            check_dim(name, value, 1)?;
        }
        element_count("h [B,N,d]", &[self.b, self.n, self.d])?;
        element_count("v [B,K,4d]", &[self.b, self.k, 4, self.d])?;
        Ok(())
    }
}

impl UnitDims {
    /// `P, M, d, H ≥ 1`; every element count at most `i32::MAX`.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [("P", self.p), ("M", self.m), ("d", self.d), ("H", self.h)] {
            check_dim(name, value, 1)?;
        }
        element_count("v [P,4d]", &[self.p, 4, self.d])?;
        element_count("W1 [M,4d,H]", &[self.m, 4, self.d, self.h])?;
        element_count("W2 [M,H,d]", &[self.m, self.h, self.d])?;
        element_count("hidden_saved [P,H]", &[self.p, self.h])?;
        Ok(())
    }
}

fn check_len(name: &str, actual: usize, expected: usize, shape: &str) -> Result<()> {
    if actual != expected {
        return Err(Error::InvalidArgument(format!(
            "{name} has {actual} elements, expected {shape} = {expected}"
        )));
    }
    Ok(())
}

/// The one float dtype of a call; any mix is an error.
fn common_dtype(tensors: &[(&str, DType)]) -> Result<DType> {
    let (first_name, first) = tensors[0];
    for &(name, dtype) in &tensors[1..] {
        if dtype != first {
            return Err(Error::InvalidArgument(format!(
                "every float tensor of a call has one dtype: {first_name} is {}, {name} is {}",
                first.name(),
                dtype.name()
            )));
        }
    }
    Ok(first)
}

fn check_cells(name: &str, idx: &[i32], n: usize) -> Result<()> {
    if let Some((pos, &bad)) = idx
        .iter()
        .enumerate()
        .find(|&(_, &i)| i < 0 || i as usize >= n)
    {
        return Err(Error::IndexOutOfRange(format!(
            "{name}[{pos}] = {bad} is outside [0, {n})"
        )));
    }
    Ok(())
}

fn check_csr(indptr: &[i32], indices: &[i32], n: usize, e: usize) -> Result<()> {
    if indptr[0] != 0 {
        return Err(Error::BadGraph(format!(
            "indptr[0] must be 0, got {}",
            indptr[0]
        )));
    }
    if let Some(row) = (0..n).find(|&i| indptr[i + 1] < indptr[i]) {
        return Err(Error::BadGraph(format!(
            "indptr must be non-decreasing: indptr[{}] = {} < indptr[{row}] = {}",
            row + 1,
            indptr[row + 1],
            indptr[row]
        )));
    }
    if indptr[n] as usize != e {
        return Err(Error::BadGraph(format!(
            "indptr[N] must equal E = {e}, got {}",
            indptr[n]
        )));
    }
    if let Some((pos, &bad)) = indices
        .iter()
        .enumerate()
        .find(|&(_, &j)| j < 0 || j as usize >= n)
    {
        return Err(Error::BadGraph(format!(
            "indices[{pos}] = {bad} is outside [0, {n})"
        )));
    }
    Ok(())
}

/// `valid ∈ {0, 1}`; the valid slots of each row name distinct cells in `[0, N)`.
fn check_gated_selection(cell_idx: &[i32], valid: &[i32], dims: &CellDims) -> Result<()> {
    if let Some((pos, &bad)) = valid.iter().enumerate().find(|&(_, &f)| f != 0 && f != 1) {
        return Err(Error::InvalidArgument(format!(
            "valid[{pos}] = {bad}; valid holds 0 or 1"
        )));
    }
    let mut seen = vec![usize::MAX; dims.n];
    for r in 0..dims.b {
        for k in 0..dims.k {
            let slot = r * dims.k + k;
            if valid[slot] == 0 {
                continue;
            }
            let cell = cell_idx[slot];
            if cell < 0 || cell as usize >= dims.n {
                return Err(Error::IndexOutOfRange(format!(
                    "cell_idx[{r},{k}] = {cell} (a valid slot) is outside [0, {})",
                    dims.n
                )));
            }
            if seen[cell as usize] == r {
                return Err(Error::DuplicateIndex(format!(
                    "cell {cell} is selected twice by valid slots of row {r}"
                )));
            }
            seen[cell as usize] = r;
        }
    }
    Ok(())
}

impl GatherConcat<'_> {
    /// Check the header's preconditions; returns the call's dtype.
    pub fn validate(&self) -> Result<DType> {
        let CellDims { b, n, d, k } = self.dims;
        self.dims.validate()?;
        let dtype = common_dtype(&[
            ("u", self.u.dtype()),
            ("eps", self.eps.dtype()),
            ("h", self.h.dtype()),
            ("m", self.m.dtype()),
            ("v", self.v.dtype()),
        ])?;
        check_len("u", self.u.len(), b * d, "[B,d]")?;
        check_len("eps", self.eps.len(), b * d, "[B,d]")?;
        check_len("h", self.h.len(), b * n * d, "[B,N,d]")?;
        check_len("m", self.m.len(), b * k * d, "[B,K,d]")?;
        check_len("cell_idx", self.cell_idx.len(), b * k, "[B,K]")?;
        check_len("v", self.v.len(), b * k * 4 * d, "[B,K,4d]")?;
        check_cells("cell_idx", self.cell_idx, n)?;
        Ok(dtype)
    }
}

impl GatherConcatBwd<'_> {
    pub fn validate(&self) -> Result<DType> {
        let CellDims { b, n, d, k } = self.dims;
        self.dims.validate()?;
        let dtype = common_dtype(&[
            ("grad_v", self.grad_v.dtype()),
            ("grad_u", self.grad_u.dtype()),
            ("grad_eps", self.grad_eps.dtype()),
            ("grad_h", self.grad_h.dtype()),
            ("grad_m", self.grad_m.dtype()),
        ])?;
        check_len("grad_v", self.grad_v.len(), b * k * 4 * d, "[B,K,4d]")?;
        check_len("cell_idx", self.cell_idx.len(), b * k, "[B,K]")?;
        check_len("grad_u", self.grad_u.len(), b * d, "[B,d]")?;
        check_len("grad_eps", self.grad_eps.len(), b * d, "[B,d]")?;
        check_len("grad_h", self.grad_h.len(), b * n * d, "[B,N,d]")?;
        check_len("grad_m", self.grad_m.len(), b * k * d, "[B,K,d]")?;
        check_cells("cell_idx", self.cell_idx, n)?;
        Ok(dtype)
    }
}

impl GroupedUnitMlpFwd<'_> {
    pub fn validate(&self) -> Result<DType> {
        let UnitDims { p, m, d, h } = self.dims;
        self.dims.validate()?;
        let dtype = common_dtype(&[
            ("v", self.v.dtype()),
            ("W1", self.w1.dtype()),
            ("b1", self.b1.dtype()),
            ("W2", self.w2.dtype()),
            ("b2", self.b2.dtype()),
            ("hidden_saved", self.hidden_saved.dtype()),
            ("out", self.out.dtype()),
        ])?;
        check_len("v", self.v.len(), p * 4 * d, "[P,4d]")?;
        check_len("unit_idx", self.unit_idx.len(), p, "[P]")?;
        check_len("W1", self.w1.len(), m * 4 * d * h, "[M,4d,H]")?;
        check_len("b1", self.b1.len(), m * h, "[M,H]")?;
        check_len("W2", self.w2.len(), m * h * d, "[M,H,d]")?;
        check_len("b2", self.b2.len(), m * d, "[M,d]")?;
        check_len("hidden_saved", self.hidden_saved.len(), p * h, "[P,H]")?;
        check_len("out", self.out.len(), p * d, "[P,d]")?;
        check_cells("unit_idx", self.unit_idx, m)?;
        Ok(dtype)
    }
}

impl GroupedUnitMlpBwd<'_> {
    pub fn validate(&self) -> Result<DType> {
        let UnitDims { p, m, d, h } = self.dims;
        self.dims.validate()?;
        let dtype = common_dtype(&[
            ("v", self.v.dtype()),
            ("W1", self.w1.dtype()),
            ("W2", self.w2.dtype()),
            ("hidden_saved", self.hidden_saved.dtype()),
            ("grad_out", self.grad_out.dtype()),
            ("grad_v", self.grad_v.dtype()),
            ("grad_W1", self.grad_w1.dtype()),
            ("grad_b1", self.grad_b1.dtype()),
            ("grad_W2", self.grad_w2.dtype()),
            ("grad_b2", self.grad_b2.dtype()),
        ])?;
        check_len("v", self.v.len(), p * 4 * d, "[P,4d]")?;
        check_len("unit_idx", self.unit_idx.len(), p, "[P]")?;
        check_len("W1", self.w1.len(), m * 4 * d * h, "[M,4d,H]")?;
        check_len("W2", self.w2.len(), m * h * d, "[M,H,d]")?;
        check_len("hidden_saved", self.hidden_saved.len(), p * h, "[P,H]")?;
        check_len("grad_out", self.grad_out.len(), p * d, "[P,d]")?;
        check_len("grad_v", self.grad_v.len(), p * 4 * d, "[P,4d]")?;
        check_len("grad_W1", self.grad_w1.len(), m * 4 * d * h, "[M,4d,H]")?;
        check_len("grad_b1", self.grad_b1.len(), m * h, "[M,H]")?;
        check_len("grad_W2", self.grad_w2.len(), m * h * d, "[M,H,d]")?;
        check_len("grad_b2", self.grad_b2.len(), m * d, "[M,d]")?;
        check_cells("unit_idx", self.unit_idx, m)?;
        Ok(dtype)
    }
}

fn check_csr_dims(dims: &CellDims, e: usize) -> Result<()> {
    dims.validate()?;
    check_dim("E", e, 0)?;
    element_count("indptr [N+1]", &[dims.n + 1])?;
    element_count("V [d,d]", &[dims.d, dims.d])?;
    Ok(())
}

impl CsrMessages<'_> {
    pub fn validate(&self) -> Result<DType> {
        let CellDims { b, n, d, k } = self.dims;
        check_csr_dims(&self.dims, self.e)?;
        let dtype = common_dtype(&[
            ("h", self.h.dtype()),
            ("a", self.a.dtype()),
            ("V", self.v_proj.dtype()),
            ("m", self.m.dtype()),
        ])?;
        check_len("h", self.h.len(), b * n * d, "[B,N,d]")?;
        check_len("indptr", self.indptr.len(), n + 1, "[N+1]")?;
        check_len("indices", self.indices.len(), self.e, "[E]")?;
        check_len("a", self.a.len(), self.e, "[E]")?;
        check_len("V", self.v_proj.len(), d * d, "[d,d]")?;
        check_len("cell_idx", self.cell_idx.len(), b * k, "[B,K]")?;
        check_len("m", self.m.len(), b * k * d, "[B,K,d]")?;
        check_csr(self.indptr, self.indices, n, self.e)?;
        check_cells("cell_idx", self.cell_idx, n)?;
        Ok(dtype)
    }
}

impl CsrMessagesBwd<'_> {
    pub fn validate(&self) -> Result<DType> {
        let CellDims { b, n, d, k } = self.dims;
        check_csr_dims(&self.dims, self.e)?;
        let dtype = common_dtype(&[
            ("h", self.h.dtype()),
            ("a", self.a.dtype()),
            ("V", self.v_proj.dtype()),
            ("grad_m", self.grad_m.dtype()),
            ("grad_h", self.grad_h.dtype()),
            ("grad_a", self.grad_a.dtype()),
            ("grad_V", self.grad_v_proj.dtype()),
        ])?;
        check_len("h", self.h.len(), b * n * d, "[B,N,d]")?;
        check_len("indptr", self.indptr.len(), n + 1, "[N+1]")?;
        check_len("indices", self.indices.len(), self.e, "[E]")?;
        check_len("a", self.a.len(), self.e, "[E]")?;
        check_len("V", self.v_proj.len(), d * d, "[d,d]")?;
        check_len("cell_idx", self.cell_idx.len(), b * k, "[B,K]")?;
        check_len("grad_m", self.grad_m.len(), b * k * d, "[B,K,d]")?;
        check_len("grad_h", self.grad_h.len(), b * n * d, "[B,N,d]")?;
        check_len("grad_a", self.grad_a.len(), self.e, "[E]")?;
        check_len("grad_V", self.grad_v_proj.len(), d * d, "[d,d]")?;
        check_csr(self.indptr, self.indices, n, self.e)?;
        check_cells("cell_idx", self.cell_idx, n)?;
        Ok(dtype)
    }
}

impl GatedDeltaUpdate<'_> {
    pub fn validate(&self) -> Result<DType> {
        let CellDims { b, n, d, k } = self.dims;
        self.dims.validate()?;
        let dtype = common_dtype(&[
            ("h", self.h.dtype()),
            ("q", self.q.dtype()),
            ("c", self.c.dtype()),
            ("b", self.b.dtype()),
            ("h_old_saved", self.h_old_saved.dtype()),
        ])?;
        check_len("h", self.h.len(), b * n * d, "[B,N,d]")?;
        check_len("q", self.q.len(), b * d, "[B,d]")?;
        check_len("age", self.age.len(), b * n, "[B,N]")?;
        check_len("cell_idx", self.cell_idx.len(), b * k, "[B,K]")?;
        check_len("valid", self.valid.len(), b * k, "[B,K]")?;
        check_len("c", self.c.len(), b * k * d, "[B,K,d]")?;
        check_len("b", self.b.len(), b * k * d, "[B,K,d]")?;
        check_len("h_old_saved", self.h_old_saved.len(), b * k * d, "[B,K,d]")?;
        check_gated_selection(self.cell_idx, self.valid, &self.dims)?;
        Ok(dtype)
    }
}

impl GatedDeltaUpdateBwd<'_> {
    pub fn validate(&self) -> Result<DType> {
        let CellDims { b, n, d, k } = self.dims;
        self.dims.validate()?;
        let dtype = common_dtype(&[
            ("c", self.c.dtype()),
            ("b", self.b.dtype()),
            ("h_old_saved", self.h_old_saved.dtype()),
            ("grad_h_out", self.grad_h_out.dtype()),
            ("grad_q_out", self.grad_q_out.dtype()),
            ("grad_h", self.grad_h.dtype()),
            ("grad_q", self.grad_q.dtype()),
            ("grad_c", self.grad_c.dtype()),
            ("grad_b", self.grad_b.dtype()),
        ])?;
        check_len("cell_idx", self.cell_idx.len(), b * k, "[B,K]")?;
        check_len("valid", self.valid.len(), b * k, "[B,K]")?;
        check_len("c", self.c.len(), b * k * d, "[B,K,d]")?;
        check_len("b", self.b.len(), b * k * d, "[B,K,d]")?;
        check_len("h_old_saved", self.h_old_saved.len(), b * k * d, "[B,K,d]")?;
        check_len("grad_h_out", self.grad_h_out.len(), b * n * d, "[B,N,d]")?;
        check_len("grad_q_out", self.grad_q_out.len(), b * d, "[B,d]")?;
        check_len("grad_h", self.grad_h.len(), b * n * d, "[B,N,d]")?;
        check_len("grad_q", self.grad_q.len(), b * d, "[B,d]")?;
        check_len("grad_c", self.grad_c.len(), b * k * d, "[B,K,d]")?;
        check_len("grad_b", self.grad_b.len(), b * k * d, "[B,K,d]")?;
        check_gated_selection(self.cell_idx, self.valid, &self.dims)?;
        Ok(dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p2_9_tensor_names_are_unique_and_grads_name_inputs() {
        for op in Op::ALL {
            let names: Vec<&str> = op.tensors().iter().map(|t| t.name).collect();
            let mut sorted = names.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                names.len(),
                "{}: duplicate tensor names",
                op.name()
            );
            for grad in op.tensors().iter().filter(|t| t.role == Role::Grad) {
                let input = op
                    .tensor(&grad.name["grad_".len()..])
                    .unwrap_or_else(|| panic!("{}: {} names no input", op.name(), grad.name));
                assert_eq!(input.role, Role::Input);
                assert_eq!(input.kind, Kind::Float);
                assert_eq!(input.shape, grad.shape);
            }
        }
    }

    #[test]
    fn test_p2_9_element_count_refuses_int32_overflow() {
        assert_eq!(element_count("x", &[2, 3]), Ok(6));
        assert!(matches!(
            element_count("x", &[1 << 16, 1 << 15]),
            Err(Error::SizeOverflow(_))
        ));
        assert!(matches!(
            element_count("x", &[usize::MAX, 2]),
            Err(Error::SizeOverflow(_))
        ));
    }
}
