//! Precompiled CUDA backend. No NVRTC feature, linkage or source compilation.
//!
//! `Backend` is a synchronous host-slice staging adapter for vectors/conformance.
//! `launch_resident` instead takes device addresses and launches asynchronously on
//! the supplied cudarc stream; it performs no transfers, allocations or synchronization.
//! cudarc's safe CudaContext::load_module is gated on its nvrtc feature, so this loader
//! uses its driver::result module API with explicit ownership instead.

use crate::spec::*;
use crate::{Backend, CellDims, DType, Error, Result, TensorMut, TensorRef, UnitDims, spec};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtrMut, result, sys};
use std::{ffi::CString, path::Path, sync::Arc};

const PHASES: [&str; 14] = [
    "gated_update",
    "gated_update_bwd",
    "mlp_hidden",
    "mlp_out",
    "mlp_dz",
    "mlp_dv",
    "mlp_dw1",
    "mlp_dw2",
    "messages",
    "messages_dh",
    "messages_da",
    "messages_dv",
    "gather",
    "gather_bwd",
];

fn backend_error(e: impl std::fmt::Display) -> Error {
    Error::Backend(e.to_string())
}
fn dtype_code(dtype: DType) -> Result<usize> {
    dtype
        .abi_code()
        .map(|x| x as usize)
        .ok_or(Error::UnsupportedDType {
            backend: "cuda",
            dtype,
        })
}

#[repr(C)]
struct LaunchArgs {
    pointers: [u64; 16],
    dims: [i32; 8],
}
const _: () = assert!(std::mem::size_of::<LaunchArgs>() == 160);

/// Pointer order is the public hsn_ops.h signature (inputs then outputs), with
/// an additional fp32 [P,H] scratch pointer at slot 11 for MlpBackward.
#[derive(Clone, Copy, Debug)]
pub enum Operation {
    Gather,
    GatherBackward,
    Mlp,
    MlpBackward,
    Messages,
    MessagesBackward,
    Update,
    UpdateBackward,
}

#[derive(Clone, Copy, Debug)]
pub enum Dimensions {
    Cell(CellDims, usize),
    Unit(UnitDims),
}

impl Dimensions {
    fn pack(self) -> Result<[i32; 8]> {
        match self {
            Self::Cell(c, e) => {
                c.validate()?;
                spec::element_count("edges", &[e])?;
                Ok([
                    c.b as i32,
                    c.n as i32,
                    c.d as i32,
                    c.k as i32,
                    (c.b * c.k) as i32,
                    1,
                    1,
                    e as i32,
                ])
            }
            Self::Unit(u) => {
                u.validate()?;
                Ok([
                    u.p as i32, 1, u.d as i32, 1, u.p as i32, u.m as i32, u.h as i32, 0,
                ])
            }
        }
    }
}

struct FatbinModule {
    handle: sys::CUmodule,
    context: Arc<CudaContext>,
}
impl Drop for FatbinModule {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_ok() {
            // The backend synchronizes its stream before dropping this module.
            let _ = unsafe { result::module::unload(self.handle) };
        }
    }
}

pub struct CudaBackend {
    module: FatbinModule,
    stream: Arc<CudaStream>,
    functions: [[sys::CUfunction; 3]; 14],
}

impl CudaBackend {
    pub fn new(device: usize) -> Result<Self> {
        let context = CudaContext::new(device).map_err(backend_error)?;
        let path = std::env::var("HSN_CUDA_FATBIN")
            .unwrap_or_else(|_| env!("HSN_DEFAULT_CUDA_FATBIN").to_owned());
        Self::from_stream(context.default_stream(), Path::new(&path))
    }

    /// Load all functions before capture; retain this backend for the lifetime of captured graphs.
    pub fn from_stream(stream: Arc<CudaStream>, fatbin: &Path) -> Result<Self> {
        let context = stream.context().clone();
        context.bind_to_thread().map_err(backend_error)?;
        let path = fatbin.canonicalize().map_err(backend_error)?;
        let name = CString::new(path.as_os_str().as_encoded_bytes()).map_err(backend_error)?;
        let module = FatbinModule {
            handle: result::module::load(name).map_err(backend_error)?,
            context,
        };
        let mut functions = [[std::ptr::null_mut(); 3]; 14];
        for (phase, name) in PHASES.iter().enumerate() {
            for (dtype, suffix) in ["f32", "f16", "bf16"].iter().enumerate() {
                let symbol = CString::new(format!("hsn_cuda_{name}_{suffix}")).unwrap();
                let f = unsafe { result::module::get_function(module.handle, symbol) }
                    .map_err(backend_error)?;
                let status = unsafe { sys::cuFuncLoad(f) };
                if status != sys::CUresult::CUDA_SUCCESS {
                    return Err(Error::Backend(format!("cuFuncLoad: {status:?}")));
                }
                functions[phase][dtype] = f;
            }
        }
        Ok(Self {
            module,
            stream,
            functions,
        })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Enqueue one complete operation, including its reduction phases.
    ///
    /// # Safety
    /// All addresses must belong to this stream's context and satisfy hsn_ops.h's sizes,
    /// dtype, alignment, index/CSR, nonaliasing and mutability preconditions. Keep every
    /// allocation and this backend alive until completion (and through any graph replay).
    /// Coordinate other streams yourself. MlpBackward requires a private fp32 [P,H]
    /// workspace at pointer 11. These addresses must never be treated as host slices.
    pub unsafe fn launch_resident(
        &self,
        op: Operation,
        dtype: DType,
        dims: Dimensions,
        pointers: &[u64],
    ) -> Result<()> {
        let dt = dtype_code(dtype)?;
        if matches!(op, Operation::Messages | Operation::MessagesBackward) {
            if let Dimensions::Cell(c, _) = dims {
                spec::element_count("projection", &[c.d, c.d])?;
                spec::element_count(
                    "indptr",
                    &[c.n
                        .checked_add(1)
                        .ok_or_else(|| Error::SizeOverflow("indptr".into()))?],
                )?;
            }
        }
        let packed = dims.pack()?;
        let [b, n, d, k, p, m, h, e] = packed.map(|x| x as u32);
        let (count, phases): (usize, Vec<(usize, u32)>) = match (op, dims) {
            (Operation::Gather, Dimensions::Cell(..)) => (6, vec![(12, b * k * 4 * d)]),
            (Operation::GatherBackward, Dimensions::Cell(..)) => (6, vec![(13, b * n.max(k) * d)]),
            (Operation::Mlp, Dimensions::Unit(..)) => (8, vec![(2, p * h), (3, p * d)]),
            (Operation::MlpBackward, Dimensions::Unit(..)) => (
                12,
                vec![
                    (4, p * h),
                    (5, p * 4 * d),
                    (6, m * 4 * d * h),
                    (7, m * h * d),
                ],
            ),
            (Operation::Messages, Dimensions::Cell(..)) => (7, vec![(8, b * k * d)]),
            (Operation::MessagesBackward, Dimensions::Cell(..)) => {
                (10, vec![(9, b * n * d), (10, e), (11, d * d)])
            }
            (Operation::Update, Dimensions::Cell(..)) => (8, vec![(0, b * d)]),
            (Operation::UpdateBackward, Dimensions::Cell(..)) => (11, vec![(1, b * n.max(k) * d)]),
            _ => return Err(Error::InvalidArgument("wrong dimension kind".into())),
        };
        if pointers.len() != count {
            return Err(Error::InvalidArgument("wrong pointer count".into()));
        }
        let mut args = LaunchArgs {
            pointers: [0; 16],
            dims: packed,
        };
        args.pointers[..count].copy_from_slice(pointers);
        self.module
            .context
            .bind_to_thread()
            .map_err(backend_error)?;
        for (phase, elements) in phases {
            if elements == 0 {
                continue;
            }
            let mut parameters = [(&mut args as *mut LaunchArgs).cast()];
            // Safety: checked metadata and caller's resident-buffer contract above.
            unsafe {
                result::launch_kernel(
                    self.functions[phase][dt],
                    (elements.div_ceil(256), 1, 1),
                    (256, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut parameters,
                )
            }
            .map_err(backend_error)?;
        }
        Ok(())
    }

    fn staged(
        &self,
        op: Operation,
        dtype: DType,
        dims: Dimensions,
        mut host: Vec<HostBuffer<'_>>,
    ) -> Result<()> {
        dtype_code(dtype)?;
        let mut device = host
            .iter()
            .map(|h| h.upload(&self.stream))
            .collect::<Result<Vec<_>>>()?;
        // Staging owns every allocation exclusively; no other stream can access these buffers.
        let pointers = device
            .iter_mut()
            .map(|b| b.pointer(&self.stream))
            .collect::<Vec<_>>();
        let launch = unsafe { self.launch_resident(op, dtype, dims, &pointers) };
        // Also drain partially enqueued work on error, before any owned buffers can be freed.
        let drain = self.stream.synchronize().map_err(backend_error);
        launch?;
        drain?;
        for (h, d) in host.iter_mut().zip(&device) {
            h.download(d, &self.stream)?;
        }
        self.stream.synchronize().map_err(backend_error)
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

enum DeviceBuffer {
    F32(CudaSlice<f32>),
    U16(CudaSlice<u16>),
    I32(CudaSlice<i32>),
}
impl DeviceBuffer {
    fn pointer(&mut self, stream: &CudaStream) -> u64 {
        match self {
            Self::F32(x) => {
                let (p, _guard) = x.device_ptr_mut(stream);
                p
            }
            Self::U16(x) => {
                let (p, _guard) = x.device_ptr_mut(stream);
                p
            }
            Self::I32(x) => {
                let (p, _guard) = x.device_ptr_mut(stream);
                p
            }
        }
    }
    fn upload(t: TensorRef<'_>, stream: &Arc<CudaStream>) -> Result<Self> {
        Ok(match t {
            TensorRef::F32(x) => Self::F32(
                if x.is_empty() {
                    stream.alloc_zeros(1)
                } else {
                    stream.clone_htod(x)
                }
                .map_err(backend_error)?,
            ),
            TensorRef::F16(x) | TensorRef::BF16(x) => Self::U16(
                if x.is_empty() {
                    stream.alloc_zeros(1)
                } else {
                    stream.clone_htod(x)
                }
                .map_err(backend_error)?,
            ),
            TensorRef::F64(_) => {
                return Err(Error::UnsupportedDType {
                    backend: "cuda",
                    dtype: DType::F64,
                });
            }
        })
    }
    fn output(t: &TensorMut<'_>, stream: &Arc<CudaStream>) -> Result<Self> {
        let n = t.len().max(1);
        Ok(match t {
            TensorMut::F32(_) => Self::F32(stream.alloc_zeros(n).map_err(backend_error)?),
            TensorMut::F16(_) | TensorMut::BF16(_) => {
                Self::U16(stream.alloc_zeros(n).map_err(backend_error)?)
            }
            TensorMut::F64(_) => {
                return Err(Error::UnsupportedDType {
                    backend: "cuda",
                    dtype: DType::F64,
                });
            }
        })
    }
    fn download(&self, t: &mut TensorMut<'_>, stream: &Arc<CudaStream>) -> Result<()> {
        if t.is_empty() {
            return Ok(());
        }
        match (self, t) {
            (Self::F32(x), TensorMut::F32(y)) => stream.memcpy_dtoh(x, *y).map_err(backend_error),
            (Self::U16(x), TensorMut::F16(y) | TensorMut::BF16(y)) => {
                stream.memcpy_dtoh(x, *y).map_err(backend_error)
            }
            _ => Err(Error::InvalidArgument("staging dtype mismatch".into())),
        }
    }
}

fn read<'a>(t: &'a TensorMut<'_>) -> TensorRef<'a> {
    match t {
        TensorMut::F32(x) => TensorRef::F32(x),
        TensorMut::F64(x) => TensorRef::F64(x),
        TensorMut::F16(x) => TensorRef::F16(x),
        TensorMut::BF16(x) => TensorRef::BF16(x),
    }
}
enum HostBuffer<'a> {
    Input(TensorRef<'a>),
    Output(TensorMut<'a>),
    InOut(TensorMut<'a>),
    Index(&'a [i32]),
    IndexMut(&'a mut [i32]),
    Scratch(usize),
}
impl HostBuffer<'_> {
    fn upload(&self, s: &Arc<CudaStream>) -> Result<DeviceBuffer> {
        match self {
            Self::Input(t) => DeviceBuffer::upload(*t, s),
            Self::Output(t) => DeviceBuffer::output(t, s),
            Self::InOut(t) => DeviceBuffer::upload(read(t), s),
            Self::Index(x) => Ok(DeviceBuffer::I32(
                if x.is_empty() {
                    s.alloc_zeros(1)
                } else {
                    s.clone_htod(*x)
                }
                .map_err(backend_error)?,
            )),
            Self::IndexMut(x) => Ok(DeviceBuffer::I32(
                s.clone_htod(&**x).map_err(backend_error)?,
            )),
            Self::Scratch(n) => Ok(DeviceBuffer::F32(s.alloc_zeros(*n).map_err(backend_error)?)),
        }
    }
    fn download(&mut self, d: &DeviceBuffer, s: &Arc<CudaStream>) -> Result<()> {
        match self {
            Self::Output(t) | Self::InOut(t) => d.download(t, s),
            Self::IndexMut(t) => match d {
                DeviceBuffer::I32(x) => s.memcpy_dtoh(x, *t).map_err(backend_error),
                _ => unreachable!("integer staging"),
            },
            _ => Ok(()),
        }
    }
}

use HostBuffer::{InOut as IO, Index as X, IndexMut as XM, Input as I, Output as O, Scratch as S};
impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }
    fn gather_concat(&self, a: GatherConcat<'_>) -> Result<()> {
        let dtype = a.validate()?;
        self.staged(
            Operation::Gather,
            dtype,
            Dimensions::Cell(a.dims, 0),
            vec![I(a.u), I(a.eps), I(a.h), I(a.m), X(a.cell_idx), O(a.v)],
        )
    }
    fn gather_concat_bwd(&self, a: GatherConcatBwd<'_>) -> Result<()> {
        let dtype = a.validate()?;
        self.staged(
            Operation::GatherBackward,
            dtype,
            Dimensions::Cell(a.dims, 0),
            vec![
                I(a.grad_v),
                X(a.cell_idx),
                O(a.grad_u),
                O(a.grad_eps),
                O(a.grad_h),
                O(a.grad_m),
            ],
        )
    }
    fn grouped_unit_mlp_fwd(&self, a: GroupedUnitMlpFwd<'_>) -> Result<()> {
        let dtype = a.validate()?;
        self.staged(
            Operation::Mlp,
            dtype,
            Dimensions::Unit(a.dims),
            vec![
                I(a.v),
                X(a.unit_idx),
                I(a.w1),
                I(a.b1),
                I(a.w2),
                I(a.b2),
                O(a.hidden_saved),
                O(a.out),
            ],
        )
    }
    fn grouped_unit_mlp_bwd(&self, a: GroupedUnitMlpBwd<'_>) -> Result<()> {
        let dtype = a.validate()?;
        self.staged(
            Operation::MlpBackward,
            dtype,
            Dimensions::Unit(a.dims),
            vec![
                I(a.v),
                X(a.unit_idx),
                I(a.w1),
                I(a.w2),
                I(a.hidden_saved),
                I(a.grad_out),
                O(a.grad_v),
                O(a.grad_w1),
                O(a.grad_b1),
                O(a.grad_w2),
                O(a.grad_b2),
                S(a.dims.p * a.dims.h),
            ],
        )
    }
    fn csr_messages(&self, a: CsrMessages<'_>) -> Result<()> {
        let dtype = a.validate()?;
        self.staged(
            Operation::Messages,
            dtype,
            Dimensions::Cell(a.dims, a.e),
            vec![
                I(a.h),
                X(a.indptr),
                X(a.indices),
                I(a.a),
                I(a.v_proj),
                X(a.cell_idx),
                O(a.m),
            ],
        )
    }
    fn csr_messages_bwd(&self, a: CsrMessagesBwd<'_>) -> Result<()> {
        let dtype = a.validate()?;
        self.staged(
            Operation::MessagesBackward,
            dtype,
            Dimensions::Cell(a.dims, a.e),
            vec![
                I(a.h),
                X(a.indptr),
                X(a.indices),
                I(a.a),
                I(a.v_proj),
                X(a.cell_idx),
                I(a.grad_m),
                O(a.grad_h),
                O(a.grad_a),
                O(a.grad_v_proj),
            ],
        )
    }
    fn gated_delta_update(&self, a: GatedDeltaUpdate<'_>) -> Result<()> {
        let dtype = a.validate()?;
        self.staged(
            Operation::Update,
            dtype,
            Dimensions::Cell(a.dims, 0),
            vec![
                IO(a.h),
                IO(a.q),
                XM(a.age),
                X(a.cell_idx),
                X(a.valid),
                I(a.c),
                I(a.b),
                O(a.h_old_saved),
            ],
        )
    }
    fn gated_delta_update_bwd(&self, a: GatedDeltaUpdateBwd<'_>) -> Result<()> {
        let dtype = a.validate()?;
        self.staged(
            Operation::UpdateBackward,
            dtype,
            Dimensions::Cell(a.dims, 0),
            vec![
                X(a.cell_idx),
                X(a.valid),
                I(a.c),
                I(a.b),
                I(a.h_old_saved),
                I(a.grad_h_out),
                I(a.grad_q_out),
                O(a.grad_h),
                O(a.grad_q),
                O(a.grad_c),
                O(a.grad_b),
            ],
        )
    }
}
