//! No implicit GPU execution: opt in under the idle guard with HSN_REQUIRE_CUDA=1.
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, DevicePtrMut};
use hsn_kernels::cuda::{Dimensions, Operation};
use hsn_kernels::spec::{self, Op, Role};
use hsn_kernels::vectors::{self, Data};
use hsn_kernels::{CellDims, CudaBackend, DType, UnitDims};
use std::collections::BTreeMap;

fn required() -> bool {
    std::env::var("HSN_REQUIRE_CUDA").as_deref() == Ok("1")
}

#[test]
fn test_p2_10_cuda_backend_host_adapter_vectors() {
    if !required() {
        eprintln!("CUDA test not requested; set HSN_REQUIRE_CUDA=1 under the idle guard");
        return;
    }
    let backend = CudaBackend::new(0).expect("required CUDA backend/artifact");
    for op in Op::ALL {
        for size in spec::SIZES {
            let v = vectors::load(&vectors::default_dir(), op, size.name).unwrap();
            let mut input = v.tensors.clone();
            let forward = vectors::forward(&backend, op, &v.dims, &input).unwrap();
            input.extend(forward.clone());
            let backward = vectors::backward(&backend, op, &v.dims, &input).unwrap();
            for (name, got) in forward.iter().chain(&backward) {
                let tolerance = spec::tolerance(DType::F32).unwrap();
                for (a, b) in got
                    .data
                    .to_f64()
                    .into_iter()
                    .zip(v.tensors[name].data.to_f64())
                {
                    if matches!(got.data, Data::I32(_)) {
                        assert_eq!(a, b, "{op:?}/{}/{name}", size.name);
                    } else {
                        assert!(
                            tolerance.allows(a, b),
                            "{op:?}/{}/{name}: {a} != {b}",
                            size.name
                        );
                    }
                }
            }
        }
    }
}

fn bytes(data: &Data, dtype: DType) -> Vec<u8> {
    match data {
        Data::I32(x) => x.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        Data::F32(x) => match dtype {
            DType::F32 => x.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            DType::F16 => x
                .iter()
                .flat_map(|&v| half::f16::from_f32(v).to_bits().to_ne_bytes())
                .collect(),
            DType::BF16 => x
                .iter()
                .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_ne_bytes())
                .collect(),
            DType::F64 => unreachable!(),
        },
        Data::F64(_) => unreachable!("vectors are stored as f32"),
    }
}
fn values(data: &[u8], integer: bool, dtype: DType) -> Vec<f64> {
    if integer {
        return data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&v| i32::from_ne_bytes(v) as f64)
            .collect();
    }
    match dtype {
        DType::F32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&v| f32::from_ne_bytes(v) as f64)
            .collect(),
        DType::F16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&v| half::f16::from_bits(u16::from_ne_bytes(v)).to_f64())
            .collect(),
        DType::BF16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&v| half::bf16::from_bits(u16::from_ne_bytes(v)).to_f64())
            .collect(),
        DType::F64 => unreachable!(),
    }
}

#[test]
fn test_p2_10_cuda_resident_vectors_all_dtypes() {
    if !required() {
        eprintln!("CUDA test not requested; set HSN_REQUIRE_CUDA=1 under the idle guard");
        return;
    }
    let backend = CudaBackend::new(0).expect("required CUDA backend/artifact");
    let stream = backend.stream();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for op in Op::ALL {
            for size in spec::SIZES {
                let v = vectors::load(&vectors::default_dir(), op, size.name).unwrap();
                let mut buffers: BTreeMap<String, CudaSlice<u8>> = BTreeMap::new();
                for (name, t) in &v.tensors {
                    let role = op.tensors().iter().find(|s| s.name == name).unwrap().role;
                    let mut host = bytes(&t.data, dtype);
                    if matches!(role, Role::Output | Role::Grad) {
                        host.fill(0);
                    }
                    if op == Op::GatedDeltaUpdate
                        && role == Role::Output
                        && let Some(input) = name.strip_suffix("_out")
                        && let Some(t) = v.tensors.get(input)
                    {
                        host = bytes(&t.data, dtype);
                    }
                    buffers.insert(name.clone(), stream.clone_htod(&host).unwrap());
                }
                let d = &v.dims;
                let cell = || {
                    Dimensions::Cell(
                        CellDims {
                            b: d["B"],
                            n: d["N"],
                            d: d["d"],
                            k: d["K"],
                        },
                        *d.get("E").unwrap_or(&0),
                    )
                };
                let (fwd, bwd, shape, inputs, grads): (_, _, _, Vec<&str>, Vec<&str>) = match op {
                    Op::GatherConcat => (
                        Operation::Gather,
                        Operation::GatherBackward,
                        cell(),
                        vec!["u", "eps", "h", "m", "cell_idx", "v"],
                        vec![
                            "grad_v", "cell_idx", "grad_u", "grad_eps", "grad_h", "grad_m",
                        ],
                    ),
                    Op::GroupedUnitMlp => (
                        Operation::Mlp,
                        Operation::MlpBackward,
                        Dimensions::Unit(UnitDims {
                            p: d["P"],
                            m: d["M"],
                            d: d["d"],
                            h: d["H"],
                        }),
                        vec![
                            "v",
                            "unit_idx",
                            "W1",
                            "b1",
                            "W2",
                            "b2",
                            "hidden_saved",
                            "out",
                            "activation",
                        ],
                        vec![
                            "v",
                            "unit_idx",
                            "W1",
                            "W2",
                            "hidden_saved",
                            "grad_out",
                            "grad_v",
                            "grad_W1",
                            "grad_b1",
                            "grad_W2",
                            "grad_b2",
                            "scratch",
                            "activation",
                            "routes",
                            "offsets",
                        ],
                    ),
                    Op::CsrMessages => (
                        Operation::Messages,
                        Operation::MessagesBackward,
                        cell(),
                        vec!["h", "indptr", "indices", "a", "V", "cell_idx", "m"],
                        vec![
                            "h", "indptr", "indices", "a", "V", "cell_idx", "grad_m", "grad_h",
                            "grad_a", "grad_V",
                        ],
                    ),
                    Op::GatedDeltaUpdate => (
                        Operation::Update,
                        Operation::UpdateBackward,
                        cell(),
                        vec![
                            "h_out",
                            "q_out",
                            "age_out",
                            "cell_idx",
                            "valid",
                            "c",
                            "b",
                            "h_old_saved",
                        ],
                        vec![
                            "cell_idx",
                            "valid",
                            "c",
                            "b",
                            "h_old_saved",
                            "grad_h_out",
                            "grad_q_out",
                            "grad_h",
                            "grad_q",
                            "grad_c",
                            "grad_b",
                        ],
                    ),
                };
                if op == Op::GroupedUnitMlp {
                    buffers.insert(
                        "scratch".into(),
                        stream.alloc_zeros(d["P"] * d["H"] * 4).unwrap(),
                    );
                    buffers.insert(
                        "activation".into(),
                        stream.alloc_zeros(d["P"] * d["H"] * 4).unwrap(),
                    );
                    buffers.insert("routes".into(), stream.alloc_zeros(d["P"] * 4).unwrap());
                    buffers.insert(
                        "offsets".into(),
                        stream.alloc_zeros((d["M"] + 1) * 4).unwrap(),
                    );
                }
                for (operation, names) in [(fwd, inputs), (bwd, grads)] {
                    let pointers = names
                        .iter()
                        .map(|name| {
                            let (p, _guard) =
                                buffers.get_mut(*name).unwrap().device_ptr_mut(stream);
                            p
                        })
                        .collect::<Vec<_>>();
                    // Each buffer is exclusively owned here; synchronize before download/drop.
                    unsafe { backend.launch_resident(operation, dtype, shape, &pointers) }.unwrap();
                    stream.synchronize().unwrap();
                }
                for t in op
                    .tensors()
                    .iter()
                    .filter(|t| matches!(t.role, Role::Output | Role::Grad))
                {
                    let expected = &v.tensors[t.name].data;
                    let integer = matches!(expected, Data::I32(_));
                    let got = stream.clone_dtoh(&buffers[t.name]).unwrap();
                    let tol = spec::tolerance(dtype).unwrap();
                    for (i, (a, b)) in values(&got, integer, dtype)
                        .into_iter()
                        .zip(expected.to_f64())
                        .enumerate()
                    {
                        if integer {
                            assert_eq!(a, b, "{op:?}/{}/{}/[{i}]", size.name, t.name);
                        } else {
                            assert!(
                                tol.allows(a, b),
                                "{op:?}/{}/{:?}/{}/[{i}]: {a} != {b}",
                                size.name,
                                dtype,
                                t.name
                            );
                        }
                    }
                }
            }
        }
    }
}
