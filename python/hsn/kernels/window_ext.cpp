// P2.10 window kernels: the host launcher of build/hsn_window.fatbin (kernels/cuda/hsn_window.cu).
// Build-time extension only: it loads a precompiled fatbin and never compiles CUDA source.
// Deliberately generic: a kernel is named, its tensors' device addresses and its scalars are
// packed into HsnWinArgs in the order given, and it is launched on the calling thread's current
// PyTorch stream. Shapes and dtypes are the Python wrappers' responsibility (hsn.kernels.window).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "hsn_window_launch.h"
#include "hsn_trainer_launch.h"

namespace {
using at::Tensor;
struct Module {
    CUmodule handle;
    std::string path;
    std::unordered_map<std::string, CUfunction> functions;
};
// Modules stay loaded for the life of the process (captured launches may still reference them).
std::mutex modules_mutex;
std::unordered_map<CUcontext, Module> modules;

void driver_check(CUresult status) {
    if (status == CUDA_SUCCESS) return;
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    TORCH_CHECK(false, "HSN window kernels: CUDA driver error: ", message ? message : "unknown");
}
CUcontext context() {
    CUcontext ctx = nullptr;
    driver_check(cuCtxGetCurrent(&ctx));
    TORCH_CHECK(ctx, "HSN window kernels: no current CUDA context; call prepare(device) first");
    return ctx;
}
void prepare(const std::string& path, int device) {
    const c10::cuda::CUDAGuard guard(device);
    TORCH_CHECK(cudaFree(nullptr) == cudaSuccess, "HSN window kernels: context initialization failed");
    std::lock_guard<std::mutex> lock(modules_mutex);
    auto ctx = context();
    auto found = modules.find(ctx);
    if (found != modules.end()) {
        TORCH_CHECK(found->second.path == path,
            "HSN window kernels: another fatbin is already loaded in this context");
        return;
    }
    Module module{};
    module.path = path;
    driver_check(cuModuleLoad(&module.handle, path.c_str()));
    modules.emplace(ctx, std::move(module));
}
CUfunction function(const std::string& name) {
    std::lock_guard<std::mutex> lock(modules_mutex);
    const auto found = modules.find(context());
    TORCH_CHECK(found != modules.end(),
        "HSN window kernels: fatbin not prepared on this device; call hsn.kernels.window.prepare");
    auto& functions = found->second.functions;
    const auto known = functions.find(name);
    if (known != functions.end()) return known->second;
    CUfunction fn;
    driver_check(cuModuleGetFunction(&fn, found->second.handle, name.c_str()));
    driver_check(cuFuncLoad(fn));  // eager, even with CUDA_MODULE_LOADING=LAZY
    functions.emplace(name, fn);
    return fn;
}
HsnWinArgs pack(const std::vector<Tensor>& tensors, const std::vector<int64_t>& ints,
    const std::vector<double>& floats, int& device) {
    TORCH_CHECK(!tensors.empty() && tensors.size() <= HSN_WIN_POINTERS
        && ints.size() <= HSN_WIN_INTS && floats.size() <= HSN_WIN_FLOATS,
        "HSN window kernels: launch descriptor overflow");
    HsnWinArgs args{};
    device = tensors[0].get_device();
    for (size_t k = 0; k < tensors.size(); ++k) {
        const auto& t = tensors[k];
        TORCH_CHECK(t.is_cuda() && t.get_device() == device && t.is_contiguous(),
            "HSN window kernels: tensor ", k, " must be a contiguous CUDA tensor on one device");
        args.p[k] = t.data_ptr();
    }
    for (size_t k = 0; k < ints.size(); ++k) {
        TORCH_CHECK(ints[k] >= INT32_MIN && ints[k] <= INT32_MAX, "HSN window kernels: int range");
        args.i[k] = static_cast<int32_t>(ints[k]);
    }
    for (size_t k = 0; k < floats.size(); ++k) args.f[k] = static_cast<float>(floats[k]);
    return args;
}
void launch_one(CUfunction fn, HsnWinArgs& args, int64_t grid, int64_t block, int64_t shared,
    CUstream stream) {
    if (grid <= 0) return;
    void* parameters[] = {&args};
    driver_check(cuLaunchKernel(fn, static_cast<unsigned>(grid), 1, 1,
        static_cast<unsigned>(block), 1, 1, static_cast<unsigned>(shared), stream, parameters,
        nullptr));
}
// One launch of `name` with `grid` blocks of `block` threads and `shared` bytes of dynamic
// shared memory.
void launch(const std::string& name, int64_t grid, int64_t block, int64_t shared,
    const std::vector<Tensor>& tensors, const std::vector<int64_t>& ints,
    const std::vector<double>& floats) {
    int device = 0;
    HsnWinArgs args = pack(tensors, ints, floats, device);
    const c10::cuda::CUDAGuard guard(device);
    // An autograd worker thread may have the device selected without a current driver context.
    C10_CUDA_CHECK(c10::cuda::SetDevice(device, /*force=*/true));
    const auto stream = static_cast<CUstream>(at::cuda::getCurrentCUDAStream(device).stream());
    launch_one(function(name), args, grid, block, shared, stream);
}
// `steps` rounds of the kernels `names` in order; round t runs them with ints[slot] = t.
void loop(const std::vector<std::string>& names, const std::vector<int64_t>& grids,
    const std::vector<int64_t>& blocks, const std::vector<int64_t>& shareds,
    const std::vector<Tensor>& tensors, const std::vector<int64_t>& ints,
    const std::vector<double>& floats, int64_t steps, int64_t slot) {
    TORCH_CHECK(names.size() == grids.size() && names.size() == blocks.size()
        && names.size() == shareds.size() && slot >= 0 && slot < HSN_WIN_INTS
        && static_cast<size_t>(slot) < ints.size(), "HSN window kernels: invalid loop descriptor");
    int device = 0;
    HsnWinArgs args = pack(tensors, ints, floats, device);
    const c10::cuda::CUDAGuard guard(device);
    C10_CUDA_CHECK(c10::cuda::SetDevice(device, /*force=*/true));
    const auto stream = static_cast<CUstream>(at::cuda::getCurrentCUDAStream(device).stream());
    std::vector<CUfunction> functions;
    for (const auto& name : names) functions.push_back(function(name));
    for (int64_t t = 0; t < steps; ++t) {
        args.i[slot] = static_cast<int32_t>(t);
        for (size_t k = 0; k < functions.size(); ++k)
            launch_one(functions[k], args, grids[k], blocks[k], shareds[k], stream);
    }
}
// P2.10b, the trainer kernels (kernels/cuda/hsn_trainer.cu): `name` over a list of tensors as one
// index space. rows[r][j] is the r-th operand of tensor j; every row is empty or as long as row
// 0, whose element counts lay out the index space. The first `elementwise` rows are fp32 tensors
// of row 0's element counts; the other rows only have to be on the device. Returns false, having
// launched nothing, when a tensor is not a contiguous tensor of the expected kind on row 0's CUDA
// device (the caller then keeps its PyTorch path); more tensors than a descriptor holds go out
// in several launches. ints fill i[1..] (i[0] is the launch's number of tensors).
bool launch_table(const std::string& name, int64_t block,
    const std::vector<std::vector<Tensor>>& rows, int64_t elementwise,
    const std::vector<int64_t>& ints, const std::vector<double>& floats,
    const std::vector<double>& doubles, const std::vector<std::vector<double>>& units) {
    TORCH_CHECK(!rows.empty() && rows.size() <= HSN_TR_ROWS && !rows[0].empty()
        && ints.size() < HSN_TR_INTS && floats.size() <= HSN_TR_FLOATS
        && doubles.size() <= HSN_TR_DOUBLES && units.size() <= HSN_TR_COEFS
        && elementwise >= 1 && static_cast<size_t>(elementwise) <= rows.size() && block > 0,
        "HSN trainer kernels: launch descriptor overflow");
    const size_t n = rows[0].size();
    const int device = rows[0][0].is_cuda() ? rows[0][0].get_device() : -1;
    if (device < 0) return false;
    for (size_t r = 0; r < rows.size(); ++r) {
        TORCH_CHECK(rows[r].empty() || rows[r].size() == n,
            "HSN trainer kernels: row ", r, " must be empty or hold one tensor per tensor of row 0");
        for (size_t j = 0; j < rows[r].size(); ++j) {
            const auto& t = rows[r][j];
            if (!t.is_cuda() || t.get_device() != device || !t.is_contiguous()) return false;
            if (static_cast<int64_t>(r) < elementwise
                && (t.scalar_type() != at::kFloat || t.numel() != rows[0][j].numel())) return false;
        }
    }
    HsnTrainArgs args{};
    for (size_t k = 0; k < ints.size(); ++k) {
        TORCH_CHECK(ints[k] >= INT32_MIN && ints[k] <= INT32_MAX, "HSN trainer kernels: int range");
        args.i[k + 1] = static_cast<int32_t>(ints[k]);
    }
    for (size_t k = 0; k < floats.size(); ++k) args.f[k] = static_cast<float>(floats[k]);
    for (size_t k = 0; k < doubles.size(); ++k) args.g[k] = doubles[k];
    for (size_t c = 0; c < units.size(); ++c) {
        TORCH_CHECK(units[c].size() <= HSN_TR_UNITS, "HSN trainer kernels: too many units");
        for (size_t m = 0; m < units[c].size(); ++m) args.u[c][m] = static_cast<float>(units[c][m]);
    }
    const c10::cuda::CUDAGuard guard(device);
    C10_CUDA_CHECK(c10::cuda::SetDevice(device, /*force=*/true));
    const auto stream = static_cast<CUstream>(at::cuda::getCurrentCUDAStream(device).stream());
    const CUfunction fn = function(name);
    size_t j = 0;
    while (j < n) {
        int count = 0;
        int64_t total = 0;
        for (; j < n && count < HSN_TR_TENSORS; ++j) {
            const int64_t numel = rows[0][j].numel();
            if (numel == 0) continue;
            for (size_t r = 0; r < rows.size(); ++r)
                args.p[r][count] = rows[r].empty() ? nullptr : rows[r][j].data_ptr();
            total += numel;
            args.end[count++] = total;
        }
        if (count == 0) break;
        const int64_t grid = (total + block - 1) / block;
        TORCH_CHECK(grid <= INT32_MAX, "HSN trainer kernels: too many elements for one launch");
        args.i[0] = count;
        void* parameters[] = {&args};
        driver_check(cuLaunchKernel(fn, static_cast<unsigned>(grid), 1, 1,
            static_cast<unsigned>(block), 1, 1, 0, stream, parameters, nullptr));
    }
    return true;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare", &prepare);
    m.def("launch", &launch);
    m.def("loop", &loop);
    m.def("launch_table", &launch_table);
}
