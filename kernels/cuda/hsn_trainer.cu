/* P2.10b trainer kernels: the per-step passes of the trainer after backward() (the optimiser step,
 * the float64 clip's scaling, the update caps, the weight average), each as one launch over a
 * list of tensors instead of one kernel per tensor and operation. Device code only; included by
 * hsn_window.cu, so the kernels live in build/hsn_window.fatbin.
 *
 * Every kernel reproduces, bit for bit, the sequence of PyTorch kernels it replaces
 * (hsn.train.lazy_adam, grad_norm, update_limit, weight_ema; tests/test_p2_10b_trainer_kernels.py
 * compares them on the card). Each PyTorch operation rounds once, so every step below is one
 * explicitly rounded operation (__fmul_rn, __fadd_rn, ... are never contracted by the compiler),
 * and a fused multiply-add appears exactly where PyTorch's kernel has one:
 *   lerp       self + w (end - self) as fma(w, end - self, self) for |w| < 0.5, and
 *              end - (end - self)(1 - w) as fma(-(end - self), 1 - w, end) otherwise
 *              (ATen/native/Lerp.h, contracted by nvcc);
 *   addcmul    fma(value, t1 * t2, input); fma(t1, t2, input) when value == 1
 *   addcdiv    fma(value, t1 / t2, input); input + t1 / t2 when value == 1
 *              (ATen/native/cuda/DeviceAddCmulCdiv.cuh).
 */
#include <stdint.h>
#include <math.h>
#include "hsn_trainer_launch.h"

namespace {
/* The tensor `j` and the offset in it of element `e` of the launch's index space; false past the
 * last element. A binary search over at most HSN_TR_TENSORS cumulative counts. */
__device__ __forceinline__ bool tr_locate(const HsnTrainArgs& a, int64_t e, int& j, int64_t& off) {
    const int n = a.i[0];
    if (e >= a.end[n - 1]) return false;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (a.end[mid] > e) hi = mid; else lo = mid + 1;
    }
    j = lo;
    off = e - (lo ? a.end[lo - 1] : 0);
    return true;
}
__device__ __forceinline__ int64_t tr_element() {
    return static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}
__device__ __forceinline__ float* tr_f(const HsnTrainArgs& a, int row, int j) {
    return static_cast<float*>(a.p[row][j]);
}
/* at::native::lerp as its CUDA kernels compute it (file comment). */
__device__ __forceinline__ float tr_lerp(float self, float end, float w) {
    const float diff = __fsub_rn(end, self);
    if (fabsf(w) < 0.5f) return __fmaf_rn(w, diff, self);
    return __fmaf_rn(-diff, __fsub_rn(1.0f, w), end);
}
__device__ __forceinline__ float tr_addcmul(float input, float t1, float t2, float value) {
    if (value == 1.0f) return __fmaf_rn(t1, t2, input);
    return __fmaf_rn(value, __fmul_rn(t1, t2), input);
}
__device__ __forceinline__ float tr_addcdiv(float input, float t1, float t2, float value) {
    const float q = __fdiv_rn(t1, t2);
    if (value == 1.0f) return __fadd_rn(input, q);
    return __fmaf_rn(value, q, input);
}
}  // namespace

/* LazyAdamW._dense_steps: the AdamW step of a group's ordinary tensors at one step count.
 *   rows: 0 param, 1 grad, 2 exp_avg, 3 exp_avg_sq (f32, in place except grad).
 *   i: 0 tensors, 1 weight decay on.
 *   f: 0 1 - lr wd, 1 1 - beta1, 2 beta2, 3 1 - beta2, 4 1 / sqrt(1 - beta2^t), 5 eps,
 *      6 -lr / (1 - beta1^t).
 * The operations of torch's multi-tensor ops, in the order of _dense_steps:
 *   p *= f0;  m = lerp(m, g, f1);  v *= f2;  v = addcmul(v, g, g, f3);
 *   den = sqrt(v) * f4 + f5;  p = addcdiv(p, m, den, f6).
 * (torch divides a tensor list by a scalar as a product with its reciprocal, taken in double
 * and rounded to the tensors' dtype: f4 is that reciprocal.)
 */
extern "C" __global__ void hsn_tr_adamw(HsnTrainArgs a) {
    int j;
    int64_t o;
    if (!tr_locate(a, tr_element(), j, o)) return;
    float* pp = tr_f(a, 0, j) + o;
    float* mp = tr_f(a, 2, j) + o;
    float* vp = tr_f(a, 3, j) + o;
    const float g = tr_f(a, 1, j)[o];
    float p = *pp;
    if (a.i[1]) p = __fmul_rn(p, a.f[0]);
    const float m = tr_lerp(*mp, g, a.f[1]);
    const float v = tr_addcmul(__fmul_rn(*vp, a.f[2]), g, g, a.f[3]);
    const float den = __fadd_rn(__fmul_rn(__fsqrt_rn(v), a.f[4]), a.f[5]);
    *mp = m;
    *vp = v;
    *pp = tr_addcdiv(p, m, den, a.f[6]);
}

/* LazyAdamW._unit_step: the masked per-unit step of stacked unit tensors [M, ...] that share
 * their per-unit step counts.
 *   rows: as hsn_tr_adamw.   i: 0 tensors, 1 weight decay on, 2 M.
 *   f: 3 1 - beta2, 5 eps.
 *   u[c][unit]: 0 active (1 or 0), 1 the decay factor (1 - lr wd, or 1 when inactive),
 *      2 the lerp weight (1 - beta1, or 0), 3 the second moment's factor (beta2, or 1),
 *      4 sqrt(1 - beta2^t) (or 1), 5 lr / (1 - beta1^t) (or 0): _unit_step's coefficient tensors.
 * The operations of _unit_step, in its order:
 *   g = active ? grad : 0;  p *= u1;  m = lerp(m, g, u2);  v *= u3;  v = addcmul(v, g, g, f3);
 *   den = sqrt(v) / u4 + f5;  p = addcdiv(p, m * u5, den, -1).
 */
extern "C" __global__ void hsn_tr_adamw_units(HsnTrainArgs a) {
    int j;
    int64_t o;
    if (!tr_locate(a, tr_element(), j, o)) return;
    const int64_t numel = a.end[j] - (j ? a.end[j - 1] : 0);
    const uint32_t per_unit = static_cast<uint32_t>(numel / a.i[2]);
    const int unit = static_cast<int>(static_cast<uint32_t>(o) / per_unit);
    float* pp = tr_f(a, 0, j) + o;
    float* mp = tr_f(a, 2, j) + o;
    float* vp = tr_f(a, 3, j) + o;
    const float g = a.u[0][unit] != 0.0f ? tr_f(a, 1, j)[o] : 0.0f;
    float p = *pp;
    if (a.i[1]) p = __fmul_rn(p, a.u[1][unit]);
    const float m = tr_lerp(*mp, g, a.u[2][unit]);
    const float v = tr_addcmul(__fmul_rn(*vp, a.u[3][unit]), g, g, a.f[3]);
    const float den = __fadd_rn(__fdiv_rn(__fsqrt_rn(v), a.u[4][unit]), a.f[5]);
    *mp = m;
    *vp = v;
    *pp = tr_addcdiv(p, __fmul_rn(m, a.u[5][unit]), den, -1.0f);
}

/* WeightEMA.update: self = lerp(self, end, f0).   rows: 0 self (in place), 1 end.   i: 0 tensors. */
extern "C" __global__ void hsn_tr_lerp(HsnTrainArgs a) {
    int j;
    int64_t o;
    if (!tr_locate(a, tr_element(), j, o)) return;
    float* sp = tr_f(a, 0, j) + o;
    *sp = tr_lerp(*sp, tr_f(a, 1, j)[o], a.f[0]);
}

/* The float64 clip's scaling (hsn.train.grad_norm): grad = f32(f64(grad) * coefficient).
 *   rows: 0 grad (in place); p[1][0]: the coefficient, one f64 on the device.   i: 0 tensors.
 */
extern "C" __global__ void hsn_tr_scale64(HsnTrainArgs a) {
    int j;
    int64_t o;
    if (!tr_locate(a, tr_element(), j, o)) return;
    const double coefficient = *static_cast<const double*>(a.p[1][0]);
    float* gp = tr_f(a, 0, j) + o;
    *gp = static_cast<float>(__dmul_rn(static_cast<double>(*gp), coefficient));
}

/* dst = src (the update caps' and the route guard's capture).   rows: 0 dst, 1 src. */
extern "C" __global__ void hsn_tr_copy(HsnTrainArgs a) {
    int j;
    int64_t o;
    if (!tr_locate(a, tr_element(), j, o)) return;
    tr_f(a, 0, j)[o] = tr_f(a, 1, j)[o];
}

/* UpdateLimiter.apply, first pass: the squares its two norms sum.
 *   rows: 0 param, 1 old, 2 old^2 (out), 3 (param - old)^2 (out).   i: 0 tensors.
 */
extern "C" __global__ void hsn_tr_limit_squares(HsnTrainArgs a) {
    int j;
    int64_t o;
    if (!tr_locate(a, tr_element(), j, o)) return;
    const float old = tr_f(a, 1, j)[o];
    const float delta = __fsub_rn(tr_f(a, 0, j)[o], old);
    tr_f(a, 2, j)[o] = __fmul_rn(old, old);
    tr_f(a, 3, j)[o] = __fmul_rn(delta, delta);
}

/* UpdateLimiter.apply, second pass: the projection and the squares of the applied step.
 *   rows: 0 param (in place), 1 old, 2 (param' - old)^2 (out), 3 the tensor's group: two f32 on
 *   the device, [scale, clipped (1 or 0)].   i: 0 tensors.
 * param' = old + (param - old) * scale where the group is clipped, param otherwise.
 */
extern "C" __global__ void hsn_tr_limit_project(HsnTrainArgs a) {
    int j;
    int64_t o;
    if (!tr_locate(a, tr_element(), j, o)) return;
    const float* group = tr_f(a, 3, j);
    float* pp = tr_f(a, 0, j) + o;
    const float old = tr_f(a, 1, j)[o];
    float p = *pp;
    if (group[1] != 0.0f) {
        p = __fadd_rn(old, __fmul_rn(__fsub_rn(p, old), group[0]));
        *pp = p;
    }
    const float applied = __fsub_rn(p, old);
    tr_f(a, 2, j)[o] = __fmul_rn(applied, applied);
}
