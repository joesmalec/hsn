#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "hsn_cuda_launch.h"

namespace hsn {
template <typename T> __device__ __forceinline__ float fp32(T x) { return float(x); }
template <> __device__ __forceinline__ float fp32(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float fp32(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ T rounded(float x) { return T(x); }
template <> __device__ __forceinline__ __half rounded(float x) { return __float2half_rn(x); }
template <> __device__ __forceinline__ __nv_bfloat16 rounded(float x) {
    return __float2bfloat16_rn(x);
}
template <typename T> __device__ __forceinline__ float get(HsnCudaArgs a, int p, int i) {
    return fp32(static_cast<const T*>(a.p[p])[i]);
}
template <typename T> __device__ __forceinline__ void put(HsnCudaArgs a, int p, int i, float v) {
    static_cast<T*>(a.p[p])[i] = rounded<T>(v);
}
__device__ __forceinline__ int index(HsnCudaArgs a, int p, int i) {
    return static_cast<const int32_t*>(a.p[p])[i];
}
__device__ __forceinline__ int bounded_index(HsnCudaArgs a, int p, int i, int limit) {
    const int value = index(a, p, i);
    assert(value >= 0 && value < limit);
    return value;
}
__device__ __forceinline__ int64_t tid() {
    return int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
}
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
}
__device__ __forceinline__ float dgelu(float x) {
    return 0.5f * (1.0f + erff(x * 0.7071067811865475244f))
        + x * 0.3989422804014326779f * expf(-0.5f * x * x);
}
}  // namespace hsn
