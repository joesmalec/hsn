//! The CPU oracle (P2.9): the numerical reference every backend is checked against.
//!
//! [`CpuBackend`] validates every call against the header's preconditions ([`crate::spec`]),
//! then runs [`ops`] in f32 or f64. f16 and bf16 are refused with
//! [`Error::UnsupportedDType`](crate::Error::UnsupportedDType): the half-precision backends
//! are compared against the fp32 vectors instead. `deterministic` is accepted and changes
//! nothing: the oracle is always sequential and bitwise reproducible.

pub mod math;
pub mod ops;

pub use ops::Real;

use crate::spec::{
    CsrMessages, CsrMessagesBwd, GatedDeltaUpdate, GatedDeltaUpdateBwd, GatherConcat,
    GatherConcatBwd, GroupedUnitMlpBwd, GroupedUnitMlpFwd,
};
use crate::{Backend, DType, Error, Result, TensorMut, TensorRef};

/// The pure-Rust CPU oracle, f32 and f64.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

const NAME: &str = "cpu";

fn r<'a, T: Real>(name: &str, t: TensorRef<'a>) -> Result<&'a [T]> {
    let dtype = t.dtype();
    T::slice(t).ok_or_else(|| {
        Error::InvalidArgument(format!("{name} has dtype {} unlike the call", dtype.name()))
    })
}

fn w<'a, T: Real>(name: &str, t: TensorMut<'a>) -> Result<&'a mut [T]> {
    let dtype = t.dtype();
    T::slice_mut(t).ok_or_else(|| {
        Error::InvalidArgument(format!("{name} has dtype {} unlike the call", dtype.name()))
    })
}

/// Run `$body::<T>` for the call's dtype `$dtype` (f32 or f64), refuse the others.
macro_rules! dispatch {
    ($dtype:expr, $body:ident, $args:expr) => {
        match $dtype {
            DType::F32 => $body::<f32>($args),
            DType::F64 => $body::<f64>($args),
            dtype => Err(Error::UnsupportedDType {
                backend: NAME,
                dtype,
            }),
        }
    };
}

fn run_gather_concat<T: Real>(a: GatherConcat<'_>) -> Result<()> {
    ops::gather_concat::<T>(
        r("u", a.u)?,
        r("eps", a.eps)?,
        r("h", a.h)?,
        r("m", a.m)?,
        a.cell_idx,
        w("v", a.v)?,
        a.dims,
    );
    Ok(())
}

fn run_gather_concat_bwd<T: Real>(a: GatherConcatBwd<'_>) -> Result<()> {
    ops::gather_concat_bwd::<T>(
        r("grad_v", a.grad_v)?,
        a.cell_idx,
        w("grad_u", a.grad_u)?,
        w("grad_eps", a.grad_eps)?,
        w("grad_h", a.grad_h)?,
        w("grad_m", a.grad_m)?,
        a.dims,
    );
    Ok(())
}

fn run_grouped_unit_mlp_fwd<T: Real>(a: GroupedUnitMlpFwd<'_>) -> Result<()> {
    ops::grouped_unit_mlp_fwd::<T>(
        r("v", a.v)?,
        a.unit_idx,
        r("W1", a.w1)?,
        r("b1", a.b1)?,
        r("W2", a.w2)?,
        r("b2", a.b2)?,
        w("hidden_saved", a.hidden_saved)?,
        w("out", a.out)?,
        a.dims,
    );
    Ok(())
}

fn run_grouped_unit_mlp_bwd<T: Real>(a: GroupedUnitMlpBwd<'_>) -> Result<()> {
    ops::grouped_unit_mlp_bwd::<T>(
        r("v", a.v)?,
        a.unit_idx,
        r("W1", a.w1)?,
        r("W2", a.w2)?,
        r("hidden_saved", a.hidden_saved)?,
        r("grad_out", a.grad_out)?,
        w("grad_v", a.grad_v)?,
        w("grad_W1", a.grad_w1)?,
        w("grad_b1", a.grad_b1)?,
        w("grad_W2", a.grad_w2)?,
        w("grad_b2", a.grad_b2)?,
        a.dims,
    );
    Ok(())
}

fn run_csr_messages<T: Real>(a: CsrMessages<'_>) -> Result<()> {
    ops::csr_messages::<T>(
        r("h", a.h)?,
        a.indptr,
        a.indices,
        r("a", a.a)?,
        r("V", a.v_proj)?,
        a.cell_idx,
        w("m", a.m)?,
        a.dims,
    );
    Ok(())
}

fn run_csr_messages_bwd<T: Real>(a: CsrMessagesBwd<'_>) -> Result<()> {
    ops::csr_messages_bwd::<T>(
        r("h", a.h)?,
        a.indptr,
        a.indices,
        r("a", a.a)?,
        r("V", a.v_proj)?,
        a.cell_idx,
        r("grad_m", a.grad_m)?,
        w("grad_h", a.grad_h)?,
        w("grad_a", a.grad_a)?,
        w("grad_V", a.grad_v_proj)?,
        a.dims,
    );
    Ok(())
}

fn run_gated_delta_update<T: Real>(a: GatedDeltaUpdate<'_>) -> Result<()> {
    ops::gated_delta_update::<T>(
        w("h", a.h)?,
        w("q", a.q)?,
        a.age,
        a.cell_idx,
        a.valid,
        r("c", a.c)?,
        r("b", a.b)?,
        w("h_old_saved", a.h_old_saved)?,
        a.dims,
    );
    Ok(())
}

fn run_gated_delta_update_bwd<T: Real>(a: GatedDeltaUpdateBwd<'_>) -> Result<()> {
    ops::gated_delta_update_bwd::<T>(
        a.cell_idx,
        a.valid,
        r("c", a.c)?,
        r("b", a.b)?,
        r("h_old_saved", a.h_old_saved)?,
        r("grad_h_out", a.grad_h_out)?,
        r("grad_q_out", a.grad_q_out)?,
        w("grad_h", a.grad_h)?,
        w("grad_q", a.grad_q)?,
        w("grad_c", a.grad_c)?,
        w("grad_b", a.grad_b)?,
        a.dims,
    );
    Ok(())
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        NAME
    }

    fn gather_concat(&self, args: GatherConcat<'_>) -> Result<()> {
        dispatch!(args.validate()?, run_gather_concat, args)
    }

    fn gather_concat_bwd(&self, args: GatherConcatBwd<'_>) -> Result<()> {
        dispatch!(args.validate()?, run_gather_concat_bwd, args)
    }

    fn grouped_unit_mlp_fwd(&self, args: GroupedUnitMlpFwd<'_>) -> Result<()> {
        dispatch!(args.validate()?, run_grouped_unit_mlp_fwd, args)
    }

    fn grouped_unit_mlp_bwd(&self, args: GroupedUnitMlpBwd<'_>) -> Result<()> {
        dispatch!(args.validate()?, run_grouped_unit_mlp_bwd, args)
    }

    fn csr_messages(&self, args: CsrMessages<'_>) -> Result<()> {
        dispatch!(args.validate()?, run_csr_messages, args)
    }

    fn csr_messages_bwd(&self, args: CsrMessagesBwd<'_>) -> Result<()> {
        dispatch!(args.validate()?, run_csr_messages_bwd, args)
    }

    fn gated_delta_update(&self, args: GatedDeltaUpdate<'_>) -> Result<()> {
        dispatch!(args.validate()?, run_gated_delta_update, args)
    }

    fn gated_delta_update_bwd(&self, args: GatedDeltaUpdateBwd<'_>) -> Result<()> {
        dispatch!(args.validate()?, run_gated_delta_update_bwd, args)
    }
}
