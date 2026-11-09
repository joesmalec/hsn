/* P2.10 correctness-first device kernels. No host code, allocations or float atomics.
 * Every reduction has a fixed order and accumulates in fp32 for all storage dtypes.
 * Native remains the default until parity and compiled-L1 benchmarks pass.
 */
#include "hsn_common.cuh"

namespace hsn {
// p = h, q, age, cell_idx, valid, c, b, old. One owner per (batch, channel).
template <typename T> __device__ void gated_update(HsnCudaArgs a) {
    const auto t = tid();
    if (t >= a.B * a.d) return;
    const int r = t / a.d, x = t % a.d;
    float delta = 0.0f;
    for (int k = 0; k < a.K; ++k) {
        const int slot = r * a.K + k, o = slot * a.d + x;
        if (!index(a, 4, slot)) { put<T>(a, 7, o, 0.0f); continue; }
        const int n = bounded_index(a, 3, slot, a.N), h = (r * a.N + n) * a.d + x;
        const float old = get<T>(a, 0, h), b = get<T>(a, 6, o);
        const float next = (1.0f - b) * old + b * get<T>(a, 5, o);
        put<T>(a, 7, o, old);
        put<T>(a, 0, h, next);
        delta += next - old;
        if (x == 0) static_cast<int32_t*>(a.p[2])[r * a.N + n] = 0;
    }
    put<T>(a, 1, t, get<T>(a, 1, t) + delta / a.N);
}

// p = idx, valid, c, b, old, G, g, dh, dq, dc, db.
template <typename T> __device__ void gated_update_bwd(HsnCudaArgs a) {
    const auto t = tid();
    if (t < a.B * a.N * a.d) {
        const int x = t % a.d, n = (t / a.d) % a.N, r = t / (a.N * a.d);
        const float G = get<T>(a, 5, t), g = get<T>(a, 6, r * a.d + x) / a.N;
        float dh = G;
        for (int k = 0; k < a.K; ++k) {
            const int s = r * a.K + k;
            if (index(a, 1, s) && index(a, 0, s) == n) {
                const float b = get<T>(a, 3, s * a.d + x);
                dh = (1.0f - b) * G - b * g;
            }
        }
        put<T>(a, 7, t, dh);
    }
    if (t < a.B * a.d) put<T>(a, 8, t, get<T>(a, 6, t));
    if (t < a.B * a.K * a.d) {
        const int s = t / a.d, x = t % a.d, r = s / a.K;
        float dc = 0.0f, db = 0.0f;
        if (index(a, 1, s)) {
            const int n = bounded_index(a, 0, s, a.N);
            const float g = get<T>(a, 5, (r * a.N + n) * a.d + x)
                + get<T>(a, 6, r * a.d + x) / a.N;
            dc = get<T>(a, 3, t) * g;
            db = (get<T>(a, 2, t) - get<T>(a, 4, t)) * g;
        }
        put<T>(a, 9, t, dc); put<T>(a, 10, t, db);
    }
}

// MLP forward p = v, unit_idx, W1, b1, W2, b2, hidden, out.
template <typename T> __device__ void mlp_hidden(HsnCudaArgs a) {
    const auto t = tid();
    if (t >= a.P * a.H) return;
    const int p = t / a.H, h = t % a.H, j = bounded_index(a, 1, p, a.M);
    float v = get<T>(a, 3, j * a.H + h);
    for (int c = 0; c < 4 * a.d; ++c)
        v += get<T>(a, 0, p * 4 * a.d + c) * get<T>(a, 2, (j * 4 * a.d + c) * a.H + h);
    put<T>(a, 6, t, v);
}
template <typename T> __device__ void mlp_out(HsnCudaArgs a) {
    const auto t = tid();
    if (t >= a.P * a.d) return;
    const int p = t / a.d, x = t % a.d, j = bounded_index(a, 1, p, a.M);
    float out = get<T>(a, 5, j * a.d + x);
    for (int h = 0; h < a.H; ++h)
        out += gelu(get<T>(a, 6, p * a.H + h)) * get<T>(a, 4, (j * a.H + h) * a.d + x);
    put<T>(a, 7, t, out);
}

// MLP backward p = v, idx, W1, W2, hidden, dout, dv, dW1, db1, dW2, db2, dz(float).
template <typename T> __device__ void mlp_dz(HsnCudaArgs a) {
    const auto t = tid();
    if (t >= a.P * a.H) return;
    const int p = t / a.H, h = t % a.H, j = bounded_index(a, 1, p, a.M);
    float g = 0.0f;
    for (int x = 0; x < a.d; ++x)
        g += get<T>(a, 5, p * a.d + x) * get<T>(a, 3, (j * a.H + h) * a.d + x);
    static_cast<float*>(a.p[11])[t] = g * dgelu(get<T>(a, 4, t));
}
template <typename T> __device__ void mlp_dv(HsnCudaArgs a) {
    const auto t = tid();
    if (t >= a.P * 4 * a.d) return;
    const int p = t / (4 * a.d), c = t % (4 * a.d), j = bounded_index(a, 1, p, a.M);
    float g = 0.0f;
    for (int h = 0; h < a.H; ++h)
        g += static_cast<const float*>(a.p[11])[p * a.H + h]
            * get<T>(a, 2, (j * 4 * a.d + c) * a.H + h);
    put<T>(a, 6, t, g);
}
template <typename T> __device__ void mlp_dw1(HsnCudaArgs a) {
    const auto t = tid();
    if (t >= a.M * 4 * a.d * a.H) return;
    const int h = t % a.H, c = (t / a.H) % (4 * a.d), j = t / (4 * a.d * a.H);
    float dw = 0.0f, db = 0.0f;
    for (int p = 0; p < a.P; ++p) if (index(a, 1, p) == j) {
        const float dz = static_cast<const float*>(a.p[11])[p * a.H + h];
        dw += get<T>(a, 0, p * 4 * a.d + c) * dz;
        if (c == 0) db += dz;
    }
    put<T>(a, 7, t, dw);
    if (c == 0) put<T>(a, 8, j * a.H + h, db);
}
template <typename T> __device__ void mlp_dw2(HsnCudaArgs a) {
    const auto t = tid();
    if (t >= a.M * a.H * a.d) return;
    const int x = t % a.d, h = (t / a.d) % a.H, j = t / (a.H * a.d);
    float dw = 0.0f, db = 0.0f;
    for (int p = 0; p < a.P; ++p) if (index(a, 1, p) == j) {
        const float g = get<T>(a, 5, p * a.d + x);
        dw += gelu(get<T>(a, 4, p * a.H + h)) * g;
        if (h == 0) db += g;
    }
    put<T>(a, 9, t, dw);
    if (h == 0) put<T>(a, 10, j * a.d + x, db);
}

// Messages p = h, indptr, indices, a, V, cell_idx, m.
template <typename T> __device__ void messages(HsnCudaArgs a) {
    const auto t = tid();
    if (t >= a.B * a.K * a.d) return;
    const int s = t / a.d, x = t % a.d, r = s / a.K, n = bounded_index(a, 5, s, a.N);
    float sum = 0.0f;
    for (int e = index(a, 1, n); e < index(a, 1, n + 1); ++e) {
        const int j = bounded_index(a, 2, e, a.N);
        float vh = 0.0f;
        for (int c = 0; c < a.d; ++c)
            vh += get<T>(a, 4, x * a.d + c) * get<T>(a, 0, (r * a.N + j) * a.d + c);
        sum += get<T>(a, 3, e) * vh;
    }
    put<T>(a, 6, t, sum);
}
// Messages backward p = h, indptr, indices, a, V, idx, dm, dh, da, dV.
template <typename T> __device__ void messages_dh(HsnCudaArgs a) {
    const auto t = tid();
    if (t >= a.B * a.N * a.d) return;
    const int c = t % a.d, j = (t / a.d) % a.N, r = t / (a.N * a.d);
    float sum = 0.0f;
    for (int k = 0; k < a.K; ++k) {
        const int s = r * a.K + k, n = bounded_index(a, 5, s, a.N);
        float weight = 0.0f;
        for (int e = index(a, 1, n); e < index(a, 1, n + 1); ++e)
            if (bounded_index(a, 2, e, a.N) == j) weight += get<T>(a, 3, e);
        float vt = 0.0f;
        for (int x = 0; x < a.d; ++x)
            vt += get<T>(a, 4, x * a.d + c) * get<T>(a, 6, s * a.d + x);
        sum += weight * vt;
    }
    put<T>(a, 7, t, sum);
}
template <typename T> __device__ void messages_da(HsnCudaArgs a) {
    const auto t = tid();
    if (t >= a.E) return;
    const int e = t, j = bounded_index(a, 2, e, a.N);
    float sum = 0.0f;
    for (int r = 0; r < a.B; ++r) for (int k = 0; k < a.K; ++k) {
        const int s = r * a.K + k, n = bounded_index(a, 5, s, a.N);
        if (e < index(a, 1, n) || e >= index(a, 1, n + 1)) continue;
        float dot = 0.0f;
        for (int x = 0; x < a.d; ++x) {
            float vh = 0.0f;
            for (int c = 0; c < a.d; ++c)
                vh += get<T>(a, 4, x * a.d + c) * get<T>(a, 0, (r * a.N + j) * a.d + c);
            dot += get<T>(a, 6, s * a.d + x) * vh;
        }
        sum += dot;
    }
    put<T>(a, 8, e, sum);
}
template <typename T> __device__ void messages_dv(HsnCudaArgs a) {
    const auto t = tid();
    if (t >= a.d * a.d) return;
    const int x = t / a.d, c = t % a.d;
    float sum = 0.0f;
    for (int r = 0; r < a.B; ++r) for (int k = 0; k < a.K; ++k) {
        const int s = r * a.K + k, n = bounded_index(a, 5, s, a.N);
        for (int e = index(a, 1, n); e < index(a, 1, n + 1); ++e)
            sum += get<T>(a, 3, e) * get<T>(a, 6, s * a.d + x)
                * get<T>(a, 0, (r * a.N + bounded_index(a, 2, e, a.N)) * a.d + c);
    }
    put<T>(a, 9, t, sum);
}

// Gather p = u, eps, h, m, idx, v.
template <typename T> __device__ void gather(HsnCudaArgs a) {
    const auto t = tid();
    if (t >= a.B * a.K * 4 * a.d) return;
    const int x = t % a.d, part = (t / a.d) % 4, s = t / (4 * a.d), r = s / a.K;
    float value;
    if (part < 2) value = get<T>(a, part, r * a.d + x);
    else if (part == 2) value = get<T>(a, 2, (r * a.N + bounded_index(a, 4, s, a.N)) * a.d + x);
    else value = get<T>(a, 3, s * a.d + x);
    put<T>(a, 5, t, value);
}
// Gather backward p = dv, idx, du, deps, dh, dm.
template <typename T> __device__ void gather_bwd(HsnCudaArgs a) {
    const auto t = tid();
    if (t < a.B * a.d) {
        const int r = t / a.d, x = t % a.d;
        float du = 0.0f, de = 0.0f;
        for (int k = 0; k < a.K; ++k) {
            const int base = (r * a.K + k) * 4 * a.d + x;
            du += get<T>(a, 0, base); de += get<T>(a, 0, base + a.d);
        }
        put<T>(a, 2, t, du); put<T>(a, 3, t, de);
    }
    if (t < a.B * a.N * a.d) {
        const int x = t % a.d, n = (t / a.d) % a.N, r = t / (a.N * a.d);
        float dh = 0.0f;
        for (int k = 0; k < a.K; ++k) if (index(a, 1, r * a.K + k) == n)
            dh += get<T>(a, 0, ((r * a.K + k) * 4 + 2) * a.d + x);
        put<T>(a, 4, t, dh);
    }
    if (t < a.B * a.K * a.d)
        put<T>(a, 5, t, get<T>(a, 0, ((t / a.d) * 4 + 3) * a.d + t % a.d));
}
}  // namespace hsn

// Stable, unmangled names: the same fatbin is loaded by both hosts.
#define HSN_EXPORT(OP, TYPE, SUFFIX) \
    extern "C" __global__ void hsn_cuda_##OP##_##SUFFIX(HsnCudaArgs a) { hsn::OP<TYPE>(a); }
#define HSN_TYPES(OP) \
    HSN_EXPORT(OP, float, f32) HSN_EXPORT(OP, __half, f16) HSN_EXPORT(OP, __nv_bfloat16, bf16)
HSN_TYPES(gated_update)
HSN_TYPES(gated_update_bwd)
HSN_TYPES(mlp_hidden)
HSN_TYPES(mlp_out)
HSN_TYPES(mlp_dz)
HSN_TYPES(mlp_dv)
HSN_TYPES(mlp_dw1)
HSN_TYPES(mlp_dw2)
HSN_TYPES(messages)
HSN_TYPES(messages_dh)
HSN_TYPES(messages_da)
HSN_TYPES(messages_dv)
HSN_TYPES(gather)
HSN_TYPES(gather_bwd)
