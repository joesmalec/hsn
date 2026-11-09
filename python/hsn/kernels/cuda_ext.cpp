// Build-time extension only. Loads a precompiled fatbin; never compiles CUDA source.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <climits>
#include <mutex>
#include <string>
#include <unordered_map>
#include "hsn_ops.h"
#include "hsn_cuda_launch.h"

namespace {
constexpr const char* phases[] = {"gated_update", "gated_update_bwd", "mlp_hidden", "mlp_out",
    "mlp_dz", "mlp_dv", "mlp_dw1", "mlp_dw2", "messages", "messages_dh", "messages_da",
    "messages_dv", "gather", "gather_bwd"};
constexpr const char* suffixes[] = {"f32", "f16", "bf16"};
struct Module {
    CUmodule handle;
    std::string path;
    std::array<std::array<CUfunction, 3>, 14> functions;
};
// Keep modules alive through all captured graphs; process teardown releases the contexts.
std::mutex modules_mutex;
std::unordered_map<CUcontext, Module> modules;
thread_local std::string last_error;

void driver_check(CUresult status) {
    if (status == CUDA_SUCCESS) return;
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    TORCH_CHECK(false, "HSN CUDA driver error: ", message ? message : "unknown");
}
CUcontext context() {
    CUcontext ctx = nullptr;
    driver_check(cuCtxGetCurrent(&ctx));
    TORCH_CHECK(ctx, "HSN CUDA: no current context; call prepare_cuda(device) before execution");
    return ctx;
}
void prepare(const std::string& path, int device) {
    const c10::cuda::CUDAGuard guard(device);
    // Explicit setup, outside capture. Force runtime/primary context initialization.
    TORCH_CHECK(cudaFree(nullptr) == cudaSuccess, "HSN CUDA context initialization failed");
    cudaStreamCaptureStatus capture;
    TORCH_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &capture) == cudaSuccess,
        "cannot query capture status");
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "prepare_cuda must run before capture");
    std::lock_guard<std::mutex> lock(modules_mutex);
    auto ctx = context();
    auto found = modules.find(ctx);
    if (found != modules.end()) {
        TORCH_CHECK(found->second.path == path, "cannot replace a loaded HSN fatbin in this context");
        return;
    }
    Module module{};
    module.path = path;
    driver_check(cuModuleLoad(&module.handle, path.c_str()));
    try {
        for (size_t p = 0; p < std::size(phases); ++p)
            for (int dtype = 0; dtype < 3; ++dtype) {
                const auto name = std::string("hsn_cuda_") + phases[p] + "_" + suffixes[dtype];
                driver_check(cuModuleGetFunction(&module.functions[p][dtype], module.handle,
                    name.c_str()));
                // Eagerly load device code even when CUDA_MODULE_LOADING=LAZY.
                driver_check(cuFuncLoad(module.functions[p][dtype]));
            }
    } catch (...) { cuModuleUnload(module.handle); throw; }
    modules.emplace(ctx, std::move(module));
}
void launch(int phase, int dtype, HsnCudaArgs args, int64_t count, void* stream) {
    if (!count) return;
    CUfunction function;
    {
        std::lock_guard<std::mutex> lock(modules_mutex);
        const auto found = modules.find(context());
        TORCH_CHECK(found != modules.end(), "HSN CUDA fatbin not prepared on this device; "
            "call hsn.kernels.prepare_cuda(device) before execution/capture");
        function = found->second.functions[phase][dtype];
    }
    void* parameters[] = {&args};
    driver_check(cuLaunchKernel(function, (count + 255) / 256, 1, 1, 256, 1, 1, 0,
        static_cast<CUstream>(stream), parameters, nullptr));
}
struct AbiError { int status; };
void require(bool yes, int status = HSN_ERR_INVALID_ARGUMENT) {
    if (!yes) throw AbiError{status};
}
int64_t extent(std::initializer_list<int> dims) {
    int64_t count = 1;
    for (int d : dims) {
        require(d >= 0);
        count *= d;
        require(count <= INT_MAX, HSN_ERR_SIZE_OVERFLOW);
    }
    return count;
}
HsnCudaArgs cell(int B, int N, int d, int K, int dtype) {
    require(dtype >= 0 && dtype <= 2, HSN_ERR_UNSUPPORTED_DTYPE);
    require(B > 0 && N > 0 && d > 0 && K > 0);
    extent({B, N, d}); extent({B, K, 4, d});
    return HsnCudaArgs{{}, B, N, d, K, B * K, 1, 1, 0};
}
HsnCudaArgs unit(int P, int M, int d, int H, int dtype) {
    auto a = cell(P, 1, d, 1, dtype);
    require(M > 0 && H > 0);
    extent({M, 4, d, H}); extent({P, H}); extent({M, H, d});
    a.M = M; a.H = H;
    return a;
}
void pointers(HsnCudaArgs& a, std::initializer_list<const void*> list) {
    size_t i = 0;
    for (auto p : list) { require(p != nullptr); a.p[i++] = const_cast<void*>(p); }
}
// Empty CSR arrays are allowed to have null pointers.
void csr_pointers(HsnCudaArgs& a, const void* h, const int32_t* indptr, const int32_t* indices,
    const void* weights, const void* V, const int32_t* idx, const void* out, int E) {
    require(E >= 0); extent({a.d, a.d}); require(a.N < INT_MAX, HSN_ERR_SIZE_OVERFLOW);
    pointers(a, {h, indptr, V, idx, out});
    require(E == 0 || (indices && weights));
    a.p[0] = const_cast<void*>(h); a.p[1] = const_cast<int32_t*>(indptr);
    a.p[2] = const_cast<int32_t*>(indices); a.p[3] = const_cast<void*>(weights);
    a.p[4] = const_cast<void*>(V); a.p[5] = const_cast<int32_t*>(idx);
    a.p[6] = const_cast<void*>(out); a.E = E;
}
#define ABI_BEGIN try {
#define ABI_END return HSN_OK; } catch (const AbiError& e) { return e.status; } \
    catch (const std::exception& e) { last_error = e.what(); return HSN_ERR_BACKEND; }
}  // namespace

extern "C" int hsn_gated_delta_update(void* h, void* q, int32_t* age, const int32_t* idx,
    const int32_t* valid, const void* c, const void* b, void* old,
    int B, int N, int d, int K, int dtype, void* stream) {
    ABI_BEGIN
    auto a = cell(B, N, d, K, dtype);
    pointers(a, {h, q, age, idx, valid, c, b, old});
    launch(0, dtype, a, B * d, stream);
    ABI_END
}
extern "C" int hsn_gated_delta_update_bwd(const int32_t* idx, const int32_t* valid,
    const void* c, const void* b, const void* old, const void* G, const void* g,
    void* dh, void* dq, void* dc, void* db, int B, int N, int d, int K, int dtype, void* stream) {
    ABI_BEGIN
    auto a = cell(B, N, d, K, dtype);
    pointers(a, {idx, valid, c, b, old, G, g, dh, dq, dc, db});
    launch(1, dtype, a, B * std::max(N, K) * d, stream);
    ABI_END
}
extern "C" int hsn_grouped_unit_mlp_fwd(const void* v, const int32_t* idx, const void* W1,
    const void* b1, const void* W2, const void* b2, void* hidden, void* out,
    int P, int M, int d, int H, int dtype, void* stream) {
    ABI_BEGIN
    auto a = unit(P, M, d, H, dtype);
    pointers(a, {v, idx, W1, b1, W2, b2, hidden, out});
    launch(2, dtype, a, P * H, stream); launch(3, dtype, a, P * d, stream);
    ABI_END
}
extern "C" int hsn_grouped_unit_mlp_bwd(const void* v, const int32_t* idx, const void* W1,
    const void* W2, const void* hidden, const void* dout, void* dv, void* dW1, void* db1,
    void* dW2, void* db2, int P, int M, int d, int H, int dtype, int deterministic, void* stream) {
    ABI_BEGIN
    require(deterministic == 0 || deterministic == 1);
    auto a = unit(P, M, d, H, dtype);
    pointers(a, {v, idx, W1, W2, hidden, dout, dv, dW1, db1, dW2, db2});
    int device;
    TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "cannot get current CUDA device");
    const c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(
        static_cast<cudaStream_t>(stream), device));
    // The PyTorch graph memory pool owns scratch during capture and replay.
    auto scratch = at::empty({P, H}, at::TensorOptions().device(at::kCUDA, device).dtype(at::kFloat));
    a.p[11] = scratch.data_ptr();
    launch(4, dtype, a, P * H, stream); launch(5, dtype, a, P * 4 * d, stream);
    launch(6, dtype, a, M * 4 * d * H, stream); launch(7, dtype, a, M * H * d, stream);
    ABI_END
}
extern "C" int hsn_csr_messages(const void* h, const int32_t* indptr, const int32_t* indices,
    const void* weights, const void* V, const int32_t* idx, void* out,
    int B, int N, int d, int K, int E, int dtype, void* stream) {
    ABI_BEGIN
    auto a = cell(B, N, d, K, dtype);
    csr_pointers(a, h, indptr, indices, weights, V, idx, out, E);
    launch(8, dtype, a, B * K * d, stream);
    ABI_END
}
extern "C" int hsn_csr_messages_bwd(const void* h, const int32_t* indptr, const int32_t* indices,
    const void* weights, const void* V, const int32_t* idx, const void* dm,
    void* dh, void* da, void* dV, int B, int N, int d, int K, int E, int dtype,
    int deterministic, void* stream) {
    ABI_BEGIN
    require(deterministic == 0 || deterministic == 1);
    auto a = cell(B, N, d, K, dtype);
    csr_pointers(a, h, indptr, indices, weights, V, idx, dm, E);
    require(dh && dV && (E == 0 || da)); a.p[7] = dh; a.p[8] = da; a.p[9] = dV;
    launch(9, dtype, a, B * N * d, stream); launch(10, dtype, a, E, stream);
    launch(11, dtype, a, d * d, stream);
    ABI_END
}
extern "C" int hsn_gather_concat(const void* u, const void* eps, const void* h,
    const void* m, const int32_t* idx, void* v, int B, int N, int d, int K, int dtype, void* stream) {
    ABI_BEGIN
    auto a = cell(B, N, d, K, dtype); pointers(a, {u, eps, h, m, idx, v});
    launch(12, dtype, a, B * K * 4 * d, stream);
    ABI_END
}
extern "C" int hsn_gather_concat_bwd(const void* dv, const int32_t* idx, void* du,
    void* deps, void* dh, void* dm, int B, int N, int d, int K, int dtype,
    int deterministic, void* stream) {
    ABI_BEGIN
    require(deterministic == 0 || deterministic == 1);
    auto a = cell(B, N, d, K, dtype); pointers(a, {dv, idx, du, deps, dh, dm});
    launch(13, dtype, a, B * std::max(N, K) * d, stream);
    ABI_END
}

namespace {
using at::Tensor;
using Shape = std::vector<int64_t>;
int dtype_code(const Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case at::kFloat: return HSN_DTYPE_F32;
        case at::kHalf: return HSN_DTYPE_F16;
        case at::kBFloat16: return HSN_DTYPE_BF16;
        default: TORCH_CHECK(false, "HSN CUDA supports fp32, fp16 and bf16");
    }
}
// Private Python binding, with full metadata checks before the C ABI sees pointers.
// op: gather=0, gather_bwd=1, mlp=2, mlp_bwd=3, csr=4, csr_bwd=5, update=6, update_bwd=7.
std::vector<Tensor> run(int op, const std::vector<Tensor>& in, const Shape& dims) {
    TORCH_CHECK(dims.size() == 8 && op >= 0 && op < 8, "invalid HSN launch descriptor");
    for (size_t i = 0; i < dims.size(); ++i)
        TORCH_CHECK(dims[i] >= (i == 7 ? 0 : 1) && dims[i] <= INT_MAX, "HSN dimension out of range");
    const int B=dims[0], N=dims[1], d=dims[2], K=dims[3], P=dims[4], M=dims[5], H=dims[6], E=dims[7];
    const size_t counts[] = {5, 2, 6, 6, 6, 7, 7, 7};
    TORCH_CHECK(in.size() == counts[op], "wrong input count");
    const Tensor& reference = in[op == 7 ? 2 : 0];
    TORCH_CHECK(reference.is_cuda(), "HSN CUDA requires CUDA tensors");
    const c10::cuda::CUDAGuard guard(reference.device());
    // A new autograd worker may report the right device without a current driver
    // context. Bind the prepared primary context even when the device is unchanged.
    C10_CUDA_CHECK(c10::cuda::SetDevice(reference.get_device(), /*force=*/true));
    const int dtype = dtype_code(reference);
    // Validate extents before arithmetic in shapes/allocations.
    try { if (op == 2 || op == 3) unit(P, M, d, H, dtype); else cell(B, N, d, K, dtype); }
    catch (const AbiError& e) { TORCH_CHECK(false, "HSN size/dtype status ", e.status); }
    std::vector<Shape> shapes;
    std::vector<int> ints;
    switch (op) {
        case 0: shapes={{B,d},{B,d},{B,N,d},{B,K,d},{B,K}}; ints={4}; break;
        case 1: shapes={{B,K,4*d},{B,K}}; ints={1}; break;
        case 2: shapes={{P,4*d},{P},{M,4*d,H},{M,H},{M,H,d},{M,d}}; ints={1}; break;
        case 3: shapes={{P,4*d},{P},{M,4*d,H},{M,H,d},{P,H},{P,d}}; ints={1}; break;
        case 4: case 5:
            TORCH_CHECK(N < INT_MAX && int64_t(d)*d <= INT_MAX, "CSR extent overflow");
            shapes={{B,N,d},{int64_t(N)+1},{E},{E},{d,d},{B,K}}; ints={1,2,5};
            if (op == 5) shapes.push_back({B,K,d});
            break;
        case 6: shapes={{B,N,d},{B,d},{B,N},{B,K},{B,K},{B,K,d},{B,K,d}}; ints={2,3,4}; break;
        case 7: shapes={{B,K},{B,K},{B,K,d},{B,K,d},{B,K,d},{B,N,d},{B,d}}; ints={0,1}; break;
    }
    for (size_t i = 0; i < in.size(); ++i) {
        const auto expected = std::find(ints.begin(), ints.end(), i) != ints.end()
            ? at::kInt : reference.scalar_type();
        TORCH_CHECK(in[i].device() == reference.device() && in[i].is_contiguous()
            && in[i].scalar_type() == expected && in[i].sizes().vec() == shapes[i],
            "invalid HSN input ", i, " for op ", op);
    }
    if (op == 6) {
        for (size_t i = 0; i < 3; ++i) {
            at::assert_no_internal_overlap(in[i]);
            for (size_t j = i + 1; j < in.size(); ++j) at::assert_no_overlap(in[i], in[j]);
        }
    }
    std::vector<Tensor> out;
    auto alloc = [&](Shape shape) { out.push_back(at::empty(shape, reference.options())); };
    std::array<void*, 7> p{};
    for (size_t i = 0; i < in.size(); ++i) p[i] = in[i].data_ptr();
    auto ip = [&](int i) { return static_cast<int32_t*>(p[i]); };
    auto o = [&](int i) { return out[i].data_ptr(); };
    void* stream = at::cuda::getCurrentCUDAStream(reference.get_device()).stream();
    int status = HSN_ERR_BACKEND;
    switch (op) {
        case 0:
            alloc({B,K,4*d});
            status=hsn_gather_concat(p[0],p[1],p[2],p[3],ip(4),o(0),B,N,d,K,dtype,stream); break;
        case 1:
            alloc({B,d}); alloc({B,d}); alloc({B,N,d}); alloc({B,K,d});
            status=hsn_gather_concat_bwd(p[0],ip(1),o(0),o(1),o(2),o(3),B,N,d,K,dtype,1,stream); break;
        case 2:
            alloc({P,d}); alloc({P,H});
            status=hsn_grouped_unit_mlp_fwd(p[0],ip(1),p[2],p[3],p[4],p[5],o(1),o(0),P,M,d,H,dtype,stream); break;
        case 3:
            alloc({P,4*d}); alloc({M,4*d,H}); alloc({M,H}); alloc({M,H,d}); alloc({M,d});
            status=hsn_grouped_unit_mlp_bwd(p[0],ip(1),p[2],p[3],p[4],p[5],o(0),o(1),o(2),o(3),o(4),P,M,d,H,dtype,1,stream); break;
        case 4:
            alloc({B,K,d});
            status=hsn_csr_messages(p[0],ip(1),ip(2),p[3],p[4],ip(5),o(0),B,N,d,K,E,dtype,stream); break;
        case 5:
            alloc({B,N,d}); alloc({E}); alloc({d,d});
            status=hsn_csr_messages_bwd(p[0],ip(1),ip(2),p[3],p[4],ip(5),p[6],o(0),o(1),o(2),B,N,d,K,E,dtype,1,stream); break;
        case 6:
            alloc({B,K,d});
            status=hsn_gated_delta_update(p[0],p[1],ip(2),ip(3),ip(4),p[5],p[6],o(0),B,N,d,K,dtype,stream); break;
        case 7:
            alloc({B,N,d}); alloc({B,d}); alloc({B,K,d}); alloc({B,K,d});
            status=hsn_gated_delta_update_bwd(ip(0),ip(1),p[2],p[3],p[4],p[5],p[6],o(0),o(1),o(2),o(3),B,N,d,K,dtype,stream); break;
    }
    TORCH_CHECK(status == HSN_OK, "HSN CUDA status ", status, ": ", last_error);
    return out;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare", &prepare);
    m.def("run", &run);
}
