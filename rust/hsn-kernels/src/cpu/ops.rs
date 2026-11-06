//! The bodies of the CPU oracle: straightforward loops in the order the formulas of
//! [`crate::spec`] read, generic over f32 and f64. Not optimised (the chunk's "do not"): the
//! oracle is the reference, so every reduction runs sequentially in ascending index order and
//! every gradient output is zeroed and then accumulated.
//!
//! These functions assume validated arguments ([`crate::CpuBackend`] validates first): lengths
//! match the dimensions and every index the formulas read is in range.

// The argument lists mirror the C ABI one to one.
#![allow(clippy::too_many_arguments)]

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use super::math;
use crate::spec::{CellDims, UnitDims};
use crate::{TensorMut, TensorRef};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// The float types of the oracle: f32 (the ABI's fp32) and f64 (finite-difference checks and
/// the reference values of the vectors). Arithmetic runs in the type itself; GELU is evaluated
/// in f64 and rounded once.
pub trait Real:
    sealed::Sealed
    + Copy
    + Debug
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn from_usize(n: usize) -> Self;
    /// The slice of a view of this type (`None` for another dtype).
    fn slice<'a>(t: TensorRef<'a>) -> Option<&'a [Self]>;
    /// The mutable slice of a view of this type (`None` for another dtype).
    fn slice_mut<'a>(t: TensorMut<'a>) -> Option<&'a mut [Self]>;
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_usize(n: usize) -> Self {
        n as f32
    }
    fn slice<'a>(t: TensorRef<'a>) -> Option<&'a [Self]> {
        match t {
            TensorRef::F32(s) => Some(s),
            _ => None,
        }
    }
    fn slice_mut<'a>(t: TensorMut<'a>) -> Option<&'a mut [Self]> {
        match t {
            TensorMut::F32(s) => Some(s),
            _ => None,
        }
    }
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn from_usize(n: usize) -> Self {
        n as f64
    }
    fn slice<'a>(t: TensorRef<'a>) -> Option<&'a [Self]> {
        match t {
            TensorRef::F64(s) => Some(s),
            _ => None,
        }
    }
    fn slice_mut<'a>(t: TensorMut<'a>) -> Option<&'a mut [Self]> {
        match t {
            TensorMut::F64(s) => Some(s),
            _ => None,
        }
    }
}

fn gelu<T: Real>(x: T) -> T {
    T::from_f64(math::gelu(x.to_f64()))
}

fn gelu_grad<T: Real>(x: T) -> T {
    T::from_f64(math::gelu_grad(x.to_f64()))
}

fn zero<T: Real>(x: &mut [T]) {
    x.fill(T::ZERO);
}

// ------------------------------------------------------------------------------------------------
// (a) gather_concat
// ------------------------------------------------------------------------------------------------

pub fn gather_concat<T: Real>(
    u: &[T],
    eps: &[T],
    h: &[T],
    m: &[T],
    cell_idx: &[i32],
    v: &mut [T],
    dims: CellDims,
) {
    let CellDims { b, n, d, k } = dims;
    for r in 0..b {
        for s in 0..k {
            let slot = r * k + s;
            let cell = cell_idx[slot] as usize;
            let row = &mut v[slot * 4 * d..(slot + 1) * 4 * d];
            row[..d].copy_from_slice(&u[r * d..(r + 1) * d]);
            row[d..2 * d].copy_from_slice(&eps[r * d..(r + 1) * d]);
            row[2 * d..3 * d].copy_from_slice(&h[(r * n + cell) * d..(r * n + cell + 1) * d]);
            row[3 * d..].copy_from_slice(&m[slot * d..(slot + 1) * d]);
        }
    }
}

pub fn gather_concat_bwd<T: Real>(
    grad_v: &[T],
    cell_idx: &[i32],
    grad_u: &mut [T],
    grad_eps: &mut [T],
    grad_h: &mut [T],
    grad_m: &mut [T],
    dims: CellDims,
) {
    let CellDims { b, n, d, k } = dims;
    zero(grad_u);
    zero(grad_eps);
    zero(grad_h);
    for r in 0..b {
        for s in 0..k {
            let slot = r * k + s;
            let cell = cell_idx[slot] as usize;
            let row = &grad_v[slot * 4 * d..(slot + 1) * 4 * d];
            for c in 0..d {
                grad_u[r * d + c] += row[c];
                grad_eps[r * d + c] += row[d + c];
                grad_h[(r * n + cell) * d + c] += row[2 * d + c];
                grad_m[slot * d + c] = row[3 * d + c];
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// (b) grouped_unit_mlp
// ------------------------------------------------------------------------------------------------

pub fn grouped_unit_mlp_fwd<T: Real>(
    v: &[T],
    unit_idx: &[i32],
    w1: &[T],
    b1: &[T],
    w2: &[T],
    b2: &[T],
    hidden_saved: &mut [T],
    out: &mut [T],
    dims: UnitDims,
) {
    let UnitDims { p, m: _, d, h } = dims;
    let din = 4 * d;
    for pair in 0..p {
        let j = unit_idx[pair] as usize;
        let x = &v[pair * din..(pair + 1) * din];
        let z = &mut hidden_saved[pair * h..(pair + 1) * h];
        for (hh, zh) in z.iter_mut().enumerate() {
            let mut acc = T::ZERO;
            for (i, &xi) in x.iter().enumerate() {
                acc += xi * w1[(j * din + i) * h + hh];
            }
            *zh = acc + b1[j * h + hh];
        }
        let act: Vec<T> = z.iter().map(|&zh| gelu(zh)).collect();
        for o in 0..d {
            let mut acc = T::ZERO;
            for (hh, &ah) in act.iter().enumerate() {
                acc += ah * w2[(j * h + hh) * d + o];
            }
            out[pair * d + o] = acc + b2[j * d + o];
        }
    }
}

pub fn grouped_unit_mlp_bwd<T: Real>(
    v: &[T],
    unit_idx: &[i32],
    w1: &[T],
    w2: &[T],
    hidden_saved: &[T],
    grad_out: &[T],
    grad_v: &mut [T],
    grad_w1: &mut [T],
    grad_b1: &mut [T],
    grad_w2: &mut [T],
    grad_b2: &mut [T],
    dims: UnitDims,
) {
    let UnitDims { p, m: _, d, h } = dims;
    let din = 4 * d;
    zero(grad_w1);
    zero(grad_b1);
    zero(grad_w2);
    zero(grad_b2);
    for pair in 0..p {
        let j = unit_idx[pair] as usize;
        let x = &v[pair * din..(pair + 1) * din];
        let z = &hidden_saved[pair * h..(pair + 1) * h];
        let g = &grad_out[pair * d..(pair + 1) * d];
        let act: Vec<T> = z.iter().map(|&zh| gelu(zh)).collect();
        for o in 0..d {
            grad_b2[j * d + o] += g[o];
        }
        for (hh, &ah) in act.iter().enumerate() {
            for o in 0..d {
                grad_w2[(j * h + hh) * d + o] += ah * g[o];
            }
        }
        let dz: Vec<T> = (0..h)
            .map(|hh| {
                let mut acc = T::ZERO;
                for o in 0..d {
                    acc += w2[(j * h + hh) * d + o] * g[o];
                }
                acc * gelu_grad(z[hh])
            })
            .collect();
        for (hh, &dzh) in dz.iter().enumerate() {
            grad_b1[j * h + hh] += dzh;
        }
        for (i, &xi) in x.iter().enumerate() {
            for (hh, &dzh) in dz.iter().enumerate() {
                grad_w1[(j * din + i) * h + hh] += xi * dzh;
            }
        }
        for i in 0..din {
            let mut acc = T::ZERO;
            for (hh, &dzh) in dz.iter().enumerate() {
                acc += w1[(j * din + i) * h + hh] * dzh;
            }
            grad_v[pair * din + i] = acc;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// (c) csr_messages
// ------------------------------------------------------------------------------------------------

/// `(V x)[o] = Σ_c V[o,c] x[c]`.
fn project<T: Real>(v_proj: &[T], x: &[T], o: usize, d: usize) -> T {
    let mut acc = T::ZERO;
    for (c, &xc) in x.iter().enumerate() {
        acc += v_proj[o * d + c] * xc;
    }
    acc
}

pub fn csr_messages<T: Real>(
    h: &[T],
    indptr: &[i32],
    indices: &[i32],
    a: &[T],
    v_proj: &[T],
    cell_idx: &[i32],
    m: &mut [T],
    dims: CellDims,
) {
    let CellDims { b, n, d, k } = dims;
    for r in 0..b {
        for s in 0..k {
            let slot = r * k + s;
            let cell = cell_idx[slot] as usize;
            let out = &mut m[slot * d..(slot + 1) * d];
            zero(out);
            for e in indptr[cell] as usize..indptr[cell + 1] as usize {
                let j = indices[e] as usize;
                let hj = &h[(r * n + j) * d..(r * n + j + 1) * d];
                for (o, mo) in out.iter_mut().enumerate() {
                    *mo += a[e] * project(v_proj, hj, o, d);
                }
            }
        }
    }
}

pub fn csr_messages_bwd<T: Real>(
    h: &[T],
    indptr: &[i32],
    indices: &[i32],
    a: &[T],
    v_proj: &[T],
    cell_idx: &[i32],
    grad_m: &[T],
    grad_h: &mut [T],
    grad_a: &mut [T],
    grad_v_proj: &mut [T],
    dims: CellDims,
) {
    let CellDims { b, n, d, k } = dims;
    zero(grad_h);
    zero(grad_a);
    zero(grad_v_proj);
    for r in 0..b {
        for s in 0..k {
            let slot = r * k + s;
            let cell = cell_idx[slot] as usize;
            let dm = &grad_m[slot * d..(slot + 1) * d];
            for e in indptr[cell] as usize..indptr[cell + 1] as usize {
                let j = indices[e] as usize;
                let hj = &h[(r * n + j) * d..(r * n + j + 1) * d];
                // grad_a[e] += dm · (V h_j)
                for (o, &dmo) in dm.iter().enumerate() {
                    grad_a[e] += dmo * project(v_proj, hj, o, d);
                }
                // grad_V += a[e] · dm ⊗ h_j
                for (o, &dmo) in dm.iter().enumerate() {
                    for (c, &hjc) in hj.iter().enumerate() {
                        grad_v_proj[o * d + c] += a[e] * dmo * hjc;
                    }
                }
                // grad_h[j] += a[e] · Vᵀ dm
                for c in 0..d {
                    let mut acc = T::ZERO;
                    for (o, &dmo) in dm.iter().enumerate() {
                        acc += v_proj[o * d + c] * dmo;
                    }
                    grad_h[(r * n + j) * d + c] += a[e] * acc;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// (d) gated_delta_update
// ------------------------------------------------------------------------------------------------

pub fn gated_delta_update<T: Real>(
    h: &mut [T],
    q: &mut [T],
    age: &mut [i32],
    cell_idx: &[i32],
    valid: &[i32],
    c: &[T],
    b: &[T],
    h_old_saved: &mut [T],
    dims: CellDims,
) {
    let CellDims { b: rows, n, d, k } = dims;
    let n_t = T::from_usize(n);
    zero(h_old_saved);
    for r in 0..rows {
        let mut delta = vec![T::ZERO; d];
        for s in 0..k {
            let slot = r * k + s;
            if valid[slot] == 0 {
                continue; // padded: cell_idx is never read
            }
            let cell = cell_idx[slot] as usize;
            let base = (r * n + cell) * d;
            for i in 0..d {
                let old = h[base + i];
                let gate = b[slot * d + i];
                let new = (T::ONE - gate) * old + gate * c[slot * d + i];
                h_old_saved[slot * d + i] = old;
                h[base + i] = new;
                delta[i] += new - old;
            }
            age[r * n + cell] = 0;
        }
        for i in 0..d {
            q[r * d + i] += delta[i] / n_t;
        }
    }
}

pub fn gated_delta_update_bwd<T: Real>(
    cell_idx: &[i32],
    valid: &[i32],
    c: &[T],
    b: &[T],
    h_old_saved: &[T],
    grad_h_out: &[T],
    grad_q_out: &[T],
    grad_h: &mut [T],
    grad_q: &mut [T],
    grad_c: &mut [T],
    grad_b: &mut [T],
    dims: CellDims,
) {
    let CellDims { b: rows, n, d, k } = dims;
    let n_t = T::from_usize(n);
    grad_h.copy_from_slice(grad_h_out);
    grad_q.copy_from_slice(grad_q_out);
    zero(grad_c);
    zero(grad_b);
    for r in 0..rows {
        for s in 0..k {
            let slot = r * k + s;
            if valid[slot] == 0 {
                continue;
            }
            let cell = cell_idx[slot] as usize;
            let base = (r * n + cell) * d;
            for i in 0..d {
                let g_h = grad_h_out[base + i];
                let g_q = grad_q_out[r * d + i] / n_t; // the summary coupling
                let total = g_h + g_q;
                let gate = b[slot * d + i];
                grad_c[slot * d + i] = gate * total;
                grad_b[slot * d + i] = (c[slot * d + i] - h_old_saved[slot * d + i]) * total;
                grad_h[base + i] = (T::ONE - gate) * g_h - gate * g_q;
            }
        }
    }
}
