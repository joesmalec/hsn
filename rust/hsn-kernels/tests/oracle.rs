//! P2.9 tests of the CPU oracle: the header's constants equal `spec.rs`; finite differences in
//! f64 confirm every backward formula (the q coupling included); the gated update's summary
//! coupling and padded slots hold exactly; every precondition is refused with its status code.

use std::collections::BTreeMap;
use std::path::PathBuf;

use hsn_kernels::spec::{self, Op, Role};
use hsn_kernels::vectors::{self, Data};
use hsn_kernels::{
    Backend, CellDims, CpuBackend, CsrMessages, DType, Error, GatedDeltaUpdate, GatherConcat,
    GroupedUnitMlpFwd, TensorMut, TensorRef, UnitDims,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Every `#define HSN_<NAME> <integer>` of the header (the include guard has no value).
fn header_defines() -> BTreeMap<String, i64> {
    let path = repo_root().join("kernels/include/hsn_ops.h");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    text.lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            if words.next() != Some("#define") {
                return None;
            }
            let name = words.next()?;
            let value = words.next()?.parse::<i64>().ok()?;
            name.starts_with("HSN_").then(|| (name.to_owned(), value))
        })
        .collect()
}

#[test]
fn test_p2_9_header_constants_match_spec() {
    let expected: BTreeMap<String, i64> = spec::HEADER_CONSTANTS
        .iter()
        .map(|&(name, value)| (name.to_owned(), value))
        .collect();
    assert_eq!(header_defines(), expected);

    assert_eq!(DType::F32.abi_code(), Some(spec::DTYPE_F32));
    assert_eq!(DType::F16.abi_code(), Some(spec::DTYPE_F16));
    assert_eq!(DType::BF16.abi_code(), Some(spec::DTYPE_BF16));
    assert_eq!(DType::F64.abi_code(), None);
    let statuses = [
        (
            Error::InvalidArgument(String::new()),
            spec::ERR_INVALID_ARGUMENT,
        ),
        (
            Error::UnsupportedDType {
                backend: "cpu",
                dtype: DType::F16,
            },
            spec::ERR_UNSUPPORTED_DTYPE,
        ),
        (
            Error::IndexOutOfRange(String::new()),
            spec::ERR_INDEX_OUT_OF_RANGE,
        ),
        (
            Error::DuplicateIndex(String::new()),
            spec::ERR_DUPLICATE_INDEX,
        ),
        (Error::SizeOverflow(String::new()), spec::ERR_SIZE_OVERFLOW),
        (Error::BadGraph(String::new()), spec::ERR_BAD_GRAPH),
        (Error::Backend(String::new()), spec::ERR_BACKEND),
    ];
    for (error, code) in statuses {
        assert_eq!(error.status(), code, "{error}");
        assert_ne!(code, spec::OK);
    }
}

// ------------------------------------------------------------------------------------------------
// Finite differences (f64)
// ------------------------------------------------------------------------------------------------

/// `L = Σ_cotangents ⟨cotangent, its output⟩` for the inputs in `t` (f64).
fn loss(op: Op, dims: &vectors::Dims, t: &vectors::Tensors) -> f64 {
    let out = vectors::forward(&CpuBackend, op, dims, t).expect("forward");
    op.tensors()
        .iter()
        .filter(|spec| spec.role == Role::Cotangent)
        .map(|cot| {
            let w = t[cot.name].data.to_f64();
            let y = out[&cot.name["grad_".len()..]].data.to_f64();
            w.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>()
        })
        .sum()
}

fn set(data: &mut Data, i: usize, value: f64) {
    match data {
        Data::F64(x) => x[i] = value,
        _ => panic!("finite differences run in f64"),
    }
}

/// Every float input coordinate at `tiny`, 16 sampled coordinates per tensor at `odd`; central
/// differences with h = 1e-6 against the oracle's backward. Pass: |fd − an| ≤ 1e-7 + 1e-6·|an|.
#[test]
fn fd_check_all_ops_f64() {
    const H: f64 = 1e-6;
    const ATOL: f64 = 1e-7;
    const RTOL: f64 = 1e-6;
    println!(
        "finite differences (f64, h = {H}): op, size, checked coordinates, max |fd - an|, max ratio"
    );
    for size in [spec::TINY, spec::ODD] {
        for op in Op::ALL {
            let (dims, inputs) = vectors::draw_inputs(op, &size);
            let wide = vectors::widen(&inputs);
            let mut all = wide.clone();
            all.extend(vectors::forward(&CpuBackend, op, &dims, &wide).expect("forward"));
            let grads = vectors::backward(&CpuBackend, op, &dims, &all).expect("backward");
            let mut rng = vectors::SplitMix64::new(vectors::seed(op, &size) ^ 0x5eed);
            let (mut checked, mut worst_abs, mut worst_ratio) = (0usize, 0.0f64, 0.0f64);
            let mut probe = wide.clone();
            for grad in op.tensors().iter().filter(|spec| spec.role == Role::Grad) {
                let input = &grad.name["grad_".len()..];
                let analytic = grads[grad.name].data.to_f64();
                let base = wide[input].data.to_f64();
                let coords: Vec<usize> = if size.name == "tiny" {
                    (0..base.len()).collect()
                } else {
                    (0..16).map(|_| rng.below(base.len())).collect()
                };
                for i in coords {
                    let entry = probe.get_mut(input).expect("input");
                    set(&mut entry.data, i, base[i] + H);
                    let plus = loss(op, &dims, &probe);
                    let entry = probe.get_mut(input).expect("input");
                    set(&mut entry.data, i, base[i] - H);
                    let minus = loss(op, &dims, &probe);
                    let entry = probe.get_mut(input).expect("input");
                    set(&mut entry.data, i, base[i]);
                    let fd = (plus - minus) / (2.0 * H);
                    let an = analytic[i];
                    let err = (fd - an).abs();
                    let ratio = err / (ATOL + RTOL * an.abs());
                    assert!(
                        ratio <= 1.0,
                        "{} {}: d L / d {input}[{i}]: fd {fd:e}, analytic {an:e}",
                        op.name(),
                        size.name
                    );
                    checked += 1;
                    worst_abs = worst_abs.max(err);
                    worst_ratio = worst_ratio.max(ratio);
                }
            }
            println!(
                "  {:<20} {:<5} {checked:>5}  {worst_abs:.2e}  {worst_ratio:.3}",
                op.name(),
                size.name
            );
        }
    }
}

// ------------------------------------------------------------------------------------------------
// The summary coupling and padded slots of the gated update, exactly
// ------------------------------------------------------------------------------------------------

fn tensor(shape: &[usize], x: &[f64]) -> vectors::Tensor {
    vectors::Tensor {
        shape: shape.to_vec(),
        data: Data::F64(x.to_vec()),
    }
}

fn ints(shape: &[usize], x: &[i32]) -> vectors::Tensor {
    vectors::Tensor {
        shape: shape.to_vec(),
        data: Data::I32(x.to_vec()),
    }
}

/// B = 1, N = 4, d = 2, K = 2: slot 0 updates cell 2, slot 1 is padded with the out-of-range
/// index 99. Dyadic values, so every result is exact in f32 and f64.
fn gated_case(grad_h_out: &[f64], grad_q_out: &[f64]) -> (vectors::Tensors, vectors::Tensors) {
    let dims: vectors::Dims = [("B", 1), ("N", 4), ("d", 2), ("K", 2)]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect();
    let mut t = vectors::Tensors::new();
    let h = [0.5, -0.25, 0.125, 0.75, -0.5, 0.25, 1.0, -1.0];
    t.insert("h".into(), tensor(&[1, 4, 2], &h));
    t.insert("q".into(), tensor(&[1, 2], &[0.25, -0.5]));
    t.insert("age".into(), ints(&[1, 4], &[3, 4, 5, 6]));
    t.insert("cell_idx".into(), ints(&[1, 2], &[2, 99]));
    t.insert("valid".into(), ints(&[1, 2], &[1, 0]));
    t.insert("c".into(), tensor(&[1, 2, 2], &[0.75, -0.5, 9.0, 9.0]));
    t.insert("b".into(), tensor(&[1, 2, 2], &[0.25, 0.5, 0.5, 0.5]));
    t.insert("grad_h_out".into(), tensor(&[1, 4, 2], grad_h_out));
    t.insert("grad_q_out".into(), tensor(&[1, 2], grad_q_out));
    let op = Op::GatedDeltaUpdate;
    let mut results = (vectors::Tensors::new(), vectors::Tensors::new());
    for narrow in [false, true] {
        let input = if narrow {
            vectors::narrow(&t)
        } else {
            t.clone()
        };
        let out = vectors::forward(&CpuBackend, op, &dims, &input).expect("forward");
        let mut all = input.clone();
        all.extend(out.clone());
        let grads = vectors::backward(&CpuBackend, op, &dims, &all).expect("backward");
        let wide = (vectors::widen(&out), vectors::widen(&grads));
        if narrow {
            assert_eq!(wide, results, "f32 and f64 agree exactly on dyadic values");
        } else {
            results = wide;
        }
    }
    results
}

fn values(t: &vectors::Tensors, name: &str) -> Vec<f64> {
    t[name].data.to_f64()
}

#[test]
fn test_p2_9_gated_update_q_coupling_and_padding() {
    let (out, grads) = gated_case(&[0.0; 8], &[1.0, 2.0]);
    // Forward: cell 2 moves to (1 − b)·h + b·c; nothing else is written; q moves by Δh / N.
    let h_out = [0.5, -0.25, 0.125, 0.75, -0.1875, -0.125, 1.0, -1.0];
    assert_eq!(values(&out, "h_out"), h_out);
    assert_eq!(
        values(&out, "q_out"),
        [0.25 + 0.3125 / 4.0, -0.5 - 0.375 / 4.0]
    );
    assert_eq!(values(&out, "age_out"), [3.0, 4.0, 0.0, 6.0]);
    assert_eq!(values(&out, "h_old_saved"), [-0.5, 0.25, 0.0, 0.0]);
    // Backward from dq alone (G = 0): the coupling reaches h, c and b of the selected slot only.
    let g_over_n = [0.25, 0.5];
    assert_eq!(
        values(&grads, "grad_h"),
        [
            0.0,
            0.0,
            0.0,
            0.0,
            -0.25 * g_over_n[0],
            -0.5 * g_over_n[1],
            0.0,
            0.0
        ]
    );
    assert_eq!(values(&grads, "grad_q"), [1.0, 2.0]);
    assert_eq!(
        values(&grads, "grad_c"),
        [0.25 * g_over_n[0], 0.5 * g_over_n[1], 0.0, 0.0]
    );
    assert_eq!(
        values(&grads, "grad_b"),
        [1.25 * g_over_n[0], -0.75 * g_over_n[1], 0.0, 0.0]
    );
    // Backward from dh alone (g = 0): unselected cells pass G through, the selected one (1 − b)·G.
    let (_, grads) = gated_case(&[1.0; 8], &[0.0, 0.0]);
    assert_eq!(
        values(&grads, "grad_h"),
        [1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 1.0, 1.0]
    );
    assert_eq!(values(&grads, "grad_c"), [0.25, 0.5, 0.0, 0.0]);
    assert_eq!(values(&grads, "grad_b"), [1.25, -0.75, 0.0, 0.0]);
}

// ------------------------------------------------------------------------------------------------
// Refusals: every status code of the header
// ------------------------------------------------------------------------------------------------

fn gather(
    u: TensorRef<'_>,
    cell_idx: &[i32],
    v_len: usize,
    dims: CellDims,
) -> hsn_kernels::Result<()> {
    let (eps, h, m) = ([0.0f32; 2], [0.0f32; 4], [0.0f32; 2]);
    let mut v = vec![0.0f32; v_len];
    CpuBackend.gather_concat(GatherConcat {
        u,
        eps: TensorRef::F32(&eps),
        h: TensorRef::F32(&h),
        m: TensorRef::F32(&m),
        cell_idx,
        v: TensorMut::F32(&mut v),
        dims,
    })
}

fn csr(indptr: &[i32], indices: &[i32], cell_idx: &[i32]) -> hsn_kernels::Result<()> {
    let (h, a, vp) = ([0.0f64; 2], vec![0.5f64; indices.len()], [1.0f64]);
    let mut m = [0.0f64];
    CpuBackend.csr_messages(CsrMessages {
        h: TensorRef::F64(&h),
        indptr,
        indices,
        a: TensorRef::F64(&a),
        v_proj: TensorRef::F64(&vp),
        cell_idx,
        m: TensorMut::F64(&mut m),
        dims: CellDims {
            b: 1,
            n: 2,
            d: 1,
            k: 1,
        },
        e: indices.len(),
    })
}

fn gated(cell_idx: &[i32], valid: &[i32]) -> hsn_kernels::Result<()> {
    let (mut h, mut q, mut age) = ([0.0f32; 3], [0.0f32], [0i32; 3]);
    let (c, b) = ([0.5f32; 2], [0.5f32; 2]);
    let mut saved = [0.0f32; 2];
    CpuBackend.gated_delta_update(GatedDeltaUpdate {
        h: TensorMut::F32(&mut h),
        q: TensorMut::F32(&mut q),
        age: &mut age,
        cell_idx,
        valid,
        c: TensorRef::F32(&c),
        b: TensorRef::F32(&b),
        h_old_saved: TensorMut::F32(&mut saved),
        dims: CellDims {
            b: 1,
            n: 3,
            d: 1,
            k: 2,
        },
    })
}

fn status(result: hsn_kernels::Result<()>) -> i32 {
    result.map_or_else(|e| e.status(), |()| spec::OK)
}

#[test]
fn test_p2_9_invalid_arguments_rejected() {
    // gather_concat at B = 2, N = 2, d = 1, K = 1.
    let dims = CellDims {
        b: 2,
        n: 2,
        d: 1,
        k: 1,
    };
    let u = [0.0f32; 2];
    assert_eq!(
        status(gather(TensorRef::F32(&u), &[0, 1], 8, dims)),
        spec::OK
    );
    let out_of_range = gather(TensorRef::F32(&u), &[0, 2], 8, dims);
    assert_eq!(status(out_of_range), spec::ERR_INDEX_OUT_OF_RANGE);
    let negative = gather(TensorRef::F32(&u), &[-1, 0], 8, dims);
    assert_eq!(status(negative), spec::ERR_INDEX_OUT_OF_RANGE);
    let short_v = gather(TensorRef::F32(&u), &[0, 1], 7, dims);
    assert_eq!(status(short_v), spec::ERR_INVALID_ARGUMENT);
    let u64s = [0.0f64; 2];
    let mixed = gather(TensorRef::F64(&u64s), &[0, 1], 8, dims);
    assert_eq!(status(mixed), spec::ERR_INVALID_ARGUMENT);
    let zero_rows = CellDims { b: 0, ..dims };
    assert_eq!(
        status(gather(TensorRef::F32(&u), &[], 0, zero_rows)),
        spec::ERR_INVALID_ARGUMENT
    );
    let huge = CellDims {
        b: 4,
        n: 2,
        d: 1 << 28,
        k: 1,
    };
    assert_eq!(
        status(gather(TensorRef::F32(&u), &[0, 1], 8, huge)),
        spec::ERR_SIZE_OVERFLOW
    );
    let too_wide = CellDims {
        d: (i32::MAX as usize) + 1,
        ..dims
    };
    assert_eq!(
        status(gather(TensorRef::F32(&u), &[0, 1], 8, too_wide)),
        spec::ERR_SIZE_OVERFLOW
    );

    // f16 / bf16 are not implemented by the CPU oracle.
    let half = [0u16; 2];
    let (h16, m16) = ([0u16; 4], [0u16; 2]);
    let mut v16 = [0u16; 8];
    let result = CpuBackend.gather_concat(GatherConcat {
        u: TensorRef::F16(&half),
        eps: TensorRef::F16(&half),
        h: TensorRef::F16(&h16),
        m: TensorRef::F16(&m16),
        cell_idx: &[0, 1],
        v: TensorMut::F16(&mut v16),
        dims,
    });
    assert_eq!(status(result), spec::ERR_UNSUPPORTED_DTYPE);

    // grouped_unit_mlp: a unit index outside [0, M).
    let dims = UnitDims {
        p: 1,
        m: 2,
        d: 1,
        h: 1,
    };
    let (v, w1, b1, w2, b2) = (
        [0.0f32; 4],
        [0.0f32; 8],
        [0.0f32; 2],
        [0.0f32; 2],
        [0.0f32; 2],
    );
    for (unit, expected) in [(1, spec::OK), (2, spec::ERR_INDEX_OUT_OF_RANGE)] {
        let (mut hidden, mut out) = ([0.0f32; 1], [0.0f32; 1]);
        let result = CpuBackend.grouped_unit_mlp_fwd(GroupedUnitMlpFwd {
            v: TensorRef::F32(&v),
            unit_idx: &[unit],
            w1: TensorRef::F32(&w1),
            b1: TensorRef::F32(&b1),
            w2: TensorRef::F32(&w2),
            b2: TensorRef::F32(&b2),
            hidden_saved: TensorMut::F32(&mut hidden),
            out: TensorMut::F32(&mut out),
            dims,
        });
        assert_eq!(status(result), expected, "unit {unit}");
    }

    // csr_messages over N = 2 cells.
    assert_eq!(status(csr(&[0, 1, 1], &[1], &[0])), spec::OK);
    assert_eq!(
        status(csr(&[0, 0, 0], &[], &[1])),
        spec::OK,
        "edgeless graph"
    );
    assert_eq!(status(csr(&[1, 1, 1], &[1], &[0])), spec::ERR_BAD_GRAPH);
    assert_eq!(status(csr(&[0, 1, 0], &[1], &[0])), spec::ERR_BAD_GRAPH);
    assert_eq!(
        status(csr(&[0, 1, 2], &[1], &[0])),
        spec::ERR_BAD_GRAPH,
        "indptr[N] != E"
    );
    assert_eq!(
        status(csr(&[0, 1], &[1], &[0])),
        spec::ERR_INVALID_ARGUMENT,
        "indptr length"
    );
    assert_eq!(status(csr(&[0, 0, 0], &[1], &[0])), spec::ERR_BAD_GRAPH);
    assert_eq!(status(csr(&[0, 1, 1], &[2], &[0])), spec::ERR_BAD_GRAPH);
    assert_eq!(
        status(csr(&[0, 1, 1], &[1], &[2])),
        spec::ERR_INDEX_OUT_OF_RANGE
    );

    // gated_delta_update over N = 3 cells, K = 2 slots.
    assert_eq!(status(gated(&[0, 2], &[1, 1])), spec::OK);
    assert_eq!(status(gated(&[1, 1], &[1, 1])), spec::ERR_DUPLICATE_INDEX);
    assert_eq!(
        status(gated(&[1, 1], &[1, 0])),
        spec::OK,
        "a padded duplicate"
    );
    assert_eq!(
        status(gated(&[0, i32::MAX], &[1, 0])),
        spec::OK,
        "padded, out of range"
    );
    assert_eq!(
        status(gated(&[0, -1], &[1, 1])),
        spec::ERR_INDEX_OUT_OF_RANGE
    );
    assert_eq!(status(gated(&[0, 1], &[1, 2])), spec::ERR_INVALID_ARGUMENT);
}
