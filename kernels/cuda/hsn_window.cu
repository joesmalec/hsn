/* P2.10 window kernels (hsn.train.split): device code only, no host code, no allocations.
 * fp32 state and fp32 intermediates. Unlike hsn_ops.cu these are not bound to the P2.9 vectors,
 * so FMA contraction is allowed; exact erf GELU is kept. Every reduction has a fixed order.
 */
#include <stdint.h>
#include <math.h>
#include "hsn_window_launch.h"

namespace {
__device__ __forceinline__ const float* fptr(const HsnWinArgs& a, int k) {
    return static_cast<const float*>(a.p[k]);
}
__device__ __forceinline__ float* fout(const HsnWinArgs& a, int k) {
    return static_cast<float*>(a.p[k]);
}
}  // namespace

/* The backward state recurrence of a split window (hsn.train.split, module docstring).
 * One block per batch row; the row's cell cotangent A [N, d] and summary cotangent R [d] live in
 * shared memory ((N + 1) * d floats); the block walks t = T-1 .. 0.
 *   p: 0 c, 1 b, 2 h_old [B,T,K,d] f32; 3 cells [B,T,K] i64; 4 valid [B,T,K] u8; 5 reset [B,T] u8;
 *      6 d_q_pre [B,T,d], 7 d_q_post [B,T,d], 8 d_h_sel [B,T,K,d] f32 (read only when flagged);
 *      9 dc, 10 db [B,T,K,d] f32 (written in full: exact zeros on padded slots).
 *   i: 0 B, 1 T, 2 K, 3 d, 4 N, 5 has_pre, 6 has_post, 7 has_sel.
 * Valid slots of one (row, position) name distinct cells, so no two threads touch one A entry.
 */
extern "C" __global__ void hsn_win_recurrence(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int T = a.i[1], K = a.i[2], d = a.i[3], N = a.i[4];
    const bool has_pre = a.i[5], has_post = a.i[6], has_sel = a.i[7];
    const int row = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const float* c = fptr(a, 0);
    const float* b = fptr(a, 1);
    const float* h_old = fptr(a, 2);
    const int64_t* cells = static_cast<const int64_t*>(a.p[3]);
    const uint8_t* valid = static_cast<const uint8_t*>(a.p[4]);
    const uint8_t* reset = static_cast<const uint8_t*>(a.p[5]);
    const float* pre = fptr(a, 6);
    const float* post = fptr(a, 7);
    const float* sel = fptr(a, 8);
    float* dc = fout(a, 9);
    float* db = fout(a, 10);
    float* A = shared;
    float* R = shared + N * d;
    for (int e = tid; e < N * d; e += nt) A[e] = 0.0f;
    for (int x = tid; x < d; x += nt) R[x] = 0.0f;
    __syncthreads();
    const float inv = 1.0f / static_cast<float>(N);
    for (int t = T - 1; t >= 0; --t) {
        const int64_t rt = static_cast<int64_t>(row) * T + t;
        if (has_post) {
            for (int x = tid; x < d; x += nt) R[x] += post[rt * d + x];
            __syncthreads();
        }
        for (int e = tid; e < K * d; e += nt) {
            const int k = e / d, x = e % d;
            const int64_t s = rt * K + k, o = s * d + x;
            float dcv = 0.0f, dbv = 0.0f;
            if (valid[s]) {
                const int n = static_cast<int>(cells[s]);
                float carried = A[n * d + x];
                if (has_sel) carried += sel[o];
                const float share = R[x] * inv;
                const float g = carried + share;
                const float gate = b[o];
                dcv = gate * g;
                dbv = (c[o] - h_old[o]) * g;
                A[n * d + x] = (1.0f - gate) * carried - gate * share;
            }
            dc[o] = dcv;
            db[o] = dbv;
        }
        __syncthreads();
        const bool cut = reset[rt] != 0;
        for (int x = tid; x < d; x += nt) {
            float r = R[x];
            if (has_pre) r += pre[rt * d + x];
            R[x] = cut ? 0.0f : r;
        }
        if (cut) for (int e = tid; e < N * d; e += nt) A[e] = 0.0f;
        __syncthreads();
    }
}

/* ------------------------------------------------------------------------------------------------
 * The fused cell round of the forward sweep (hsn.kernels.window.cell_round): the sparse round of
 * PDGCell.step_sparse plus the tick, forward only, in five launches that share one descriptor.
 *   p: 0 u [B,d], 1 eps [B,d], 2 h [B,N,d], 3 q [B,d] f32; 4 age [B,N] i64;
 *      5 cells [B,K] i64, 6 units [B,K] i64, 7 valid [B,K] u8, 8 alpha [B,K] f32;
 *      9 V [d,d], 10 edge_w [N,D] f32, 11 pad_src [N,D] i64, 12 norm_w [4d],
 *      13 gate_w [d,4d], 14 gate_b [d], 15 W1 [M,4d,H], 16 b1 [M,H], 17 W2 [M,H,d], 18 b2 [M,d],
 *      19 W_out [d,d], 20 cand_w [d];
 *      scratch: 21 v [P,4d], 22 gate [P,d], 23 act [P,H] f32, 24 pairs [P] i32,
 *               25 offsets [M+1] i32, 30 out [P,d] f32;
 *      outputs: 26 h_new [B,N,d], 27 q_new [B,d] f32, 28 age_new [B,N] i64, 29 h_sel [B,K,d] f32.
 *   i: 0 B, 1 N, 2 d, 3 K, 4 M, 5 H, 6 D, 7 cand_norm.
 *   f: 0 cell-norm eps, 1 cand-norm eps.
 * P = B*K pairs (row r, slot k). A padded slot (valid = 0) is skipped everywhere: nothing is
 * computed for it, it is in no unit's list and it writes no cell.
 *
 * Latency rules these kernels follow (a GPU core runs in order, so a thread that reads global
 * memory in a long loop waits for every read): weights that a block reuses are first copied to
 * shared memory with coalesced reads (a warp reads 32 consecutive floats), the products then
 * read shared memory only, and every thread of a block has work. Shared tiles are laid out so
 * that the lanes of a warp read consecutive words (one word per memory bank): a stride of 32
 * words would put all 32 lanes on one bank and serialise the read.
 * ---------------------------------------------------------------------------------------------- */
namespace {
__device__ __forceinline__ float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
}
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
/* dst[e] = src[e] for e = tid, tid + nt, ... < count, eight reads in flight at a time: a core
 * issues instructions in order, so reads that are consumed one by one also complete one by one. */
__device__ __forceinline__ void copy_strided(float* dst, const float* src, int count, int tid, int nt) {
    int e = tid;
    for (; e + 7 * nt < count; e += 8 * nt) {
        const float t0 = src[e], t1 = src[e + nt], t2 = src[e + 2 * nt], t3 = src[e + 3 * nt];
        const float t4 = src[e + 4 * nt], t5 = src[e + 5 * nt], t6 = src[e + 6 * nt];
        const float t7 = src[e + 7 * nt];
        dst[e] = t0; dst[e + nt] = t1; dst[e + 2 * nt] = t2; dst[e + 3 * nt] = t3;
        dst[e + 4 * nt] = t4; dst[e + 5 * nt] = t5; dst[e + 6 * nt] = t6; dst[e + 7 * nt] = t7;
    }
    for (; e < count; e += nt) dst[e] = src[e];
}
/* sum_i a[i * sa] * b[i * sb] for i < n in index order, the reads issued eight at a time (a
 * product that waits for its two reads would otherwise hold back the next pair of reads). */
__device__ __forceinline__ float dot_strided(const float* a, int sa, const float* b, int sb, int n) {
    float sum = 0.0f;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const float* pa = a + i * sa;
        const float* pb = b + i * sb;
        const float a0 = pa[0], a1 = pa[sa], a2 = pa[2 * sa], a3 = pa[3 * sa];
        const float a4 = pa[4 * sa], a5 = pa[5 * sa], a6 = pa[6 * sa], a7 = pa[7 * sa];
        const float b0 = pb[0], b1 = pb[sb], b2 = pb[2 * sb], b3 = pb[3 * sb];
        const float b4 = pb[4 * sb], b5 = pb[5 * sb], b6 = pb[6 * sb], b7 = pb[7 * sb];
        sum += a0 * b0; sum += a1 * b1; sum += a2 * b2; sum += a3 * b3;
        sum += a4 * b4; sum += a5 * b5; sum += a6 * b6; sum += a7 * b7;
    }
    for (; i < n; ++i) sum += a[i * sa] * b[i * sb];
    return sum;
}
}  // namespace

/* Messages, inputs, cell norm and gate. One block (256 threads) per sixteen pairs: every block
 * reads V and the gate's weights from global memory once for all its pairs (the traffic of these
 * reads, not the arithmetic, is what a round costs).
 * Shared: V transposed [d,d] + hbar [16,d] + v [16,4d] + scale [16]. */
extern "C" __global__ void hsn_win_cell_front(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int N = a.i[1], d = a.i[2], K = a.i[3], D = a.i[6];
    const int P = a.i[0] * K, in = 4 * d;
    const int p0 = blockIdx.x * 16, tid = threadIdx.x, nt = blockDim.x;
    const int lane = tid % 32, warp = tid / 32, warps = nt / 32;
    const uint8_t* valid = static_cast<const uint8_t*>(a.p[7]);
    const int64_t* cells = static_cast<const int64_t*>(a.p[5]);
    const float* h_all = fptr(a, 2);
    float* Vt = shared;         // [c][x]
    float* hbar = Vt + d * d;   // [16][d]
    float* vs = hbar + 16 * d;  // [16][4d]
    float* scale = vs + 16 * in;
    const float* V = fptr(a, 9);
    for (int e = tid; e < d * d; e += nt) Vt[(e % d) * d + e / d] = V[e];
    // m = V (sum_j a_ij h_j): the weighted neighbour state first, then one d x d product
    for (int e = tid; e < 16 * d; e += nt) {
        const int pp = e / d, c = e % d, p = p0 + pp;
        float sum = 0.0f, uu = 0.0f, ee = 0.0f, hh = 0.0f;
        if (p < P && valid[p]) {
            const int r = p / K, cell = static_cast<int>(cells[p]);
            const float* h = h_all + static_cast<int64_t>(r) * N * d;
            const float* edge_w = fptr(a, 10) + cell * D;
            const int64_t* src = static_cast<const int64_t*>(a.p[11]) + cell * D;
            for (int k = 0; k < D; ++k) sum += edge_w[k] * h[static_cast<int>(src[k]) * d + c];
            uu = fptr(a, 0)[r * d + c];
            ee = fptr(a, 1)[r * d + c];
            hh = h[cell * d + c];
        }
        hbar[e] = sum;
        vs[pp * in + c] = uu;
        vs[pp * in + d + c] = ee;
        vs[pp * in + 2 * d + c] = hh;
    }
    __syncthreads();
    for (int e = tid; e < 16 * d; e += nt) {
        const int pp = e / d, x = e % d;
        vs[pp * in + 3 * d + x] = dot_strided(Vt + x, d, hbar + pp * d, 1, d);
    }
    __syncthreads();
    // the mean square of each pair's 4d inputs: a warp folds one pair at a time
    for (int pp = warp; pp < 16; pp += warps) {
        const float* row = vs + pp * in;
        float sum = 0.0f;
        for (int c = lane; c < in; c += 32) sum += row[c] * row[c];
        for (int offset = 16; offset; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) scale[pp] = rsqrtf(sum / static_cast<float>(in) + a.f[0]);
    }
    __syncthreads();
    float* v = fout(a, 21);
    const float* norm_w = fptr(a, 12);
    for (int e = tid; e < 16 * in; e += nt) {
        const int pp = e / in, c = e % in, p = p0 + pp;
        const float value = vs[e] * scale[pp] * norm_w[c];
        vs[e] = value;
        if (p < P && valid[p]) v[static_cast<int64_t>(p) * in + c] = value;
    }
    __syncthreads();
    // gate: a warp owns one output at a time; its lanes hold 32 consecutive weights of the row
    // per read and fold each pair's products with shuffles
    const float* gate_w = fptr(a, 13);
    const float* gate_b = fptr(a, 14);
    float* gate = fout(a, 22);
    const int rounds = (in + 31) / 32;  // <= 16: the wrapper bounds d
    for (int x = warp; x < d; x += warps) {
        const float* row = gate_w + x * in;
        const float bias = gate_b[x];
        float w[16];  // this lane's weights of the row, read once for all sixteen pairs
        for (int k = 0; k < rounds; ++k) w[k] = lane + 32 * k < in ? row[lane + 32 * k] : 0.0f;
        for (int pp = 0; pp < 16; ++pp) {
            const int p = p0 + pp;
            const float* vin = vs + pp * in + lane;
            float sum = 0.0f;
            for (int k = 0; k < rounds; ++k) sum += lane + 32 * k < in ? w[k] * vin[32 * k] : 0.0f;
            for (int offset = 16; offset; offset /= 2)
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            if (lane == 0 && p < P && valid[p]) gate[static_cast<int64_t>(p) * d + x] = sigmoidf(sum + bias);
        }
    }
}

/* One warp per unit builds the ascending list of its valid pairs (no host readback, no atomics):
 * pairs [P] grouped by unit, offsets [M+1]. */
extern "C" __global__ void hsn_win_routes(HsnWinArgs a) {
    const int P = a.i[0] * a.i[3], M = a.i[4];
    const int lane = threadIdx.x % 32;
    const int j = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (j >= M) return;
    const int64_t* units = static_cast<const int64_t*>(a.p[6]);
    const uint8_t* valid = static_cast<const uint8_t*>(a.p[7]);
    int32_t* pairs = static_cast<int32_t*>(a.p[24]);
    int32_t* offsets = static_cast<int32_t*>(a.p[25]);
    int start = 0, total = 0;
    for (int base = 0; base < P; base += 32) {
        const int p = base + lane;
        const bool live = p < P && valid[p];
        const bool before = live && units[p] < j;
        start += __popc(__ballot_sync(0xffffffff, before));
        total += __popc(__ballot_sync(0xffffffff, live));
    }
    if (lane == 0) {
        offsets[j] = start;
        if (j == M - 1) offsets[M] = total;
    }
    const unsigned lower = (1u << lane) - 1;
    for (int base = 0; base < P; base += 32) {
        const int p = base + lane;
        const bool mine = p < P && valid[p] && units[p] == j;
        const unsigned mask = __ballot_sync(0xffffffff, mine);
        if (mine) pairs[start + __popc(mask & lower)] = p;
        start += __popc(mask);
    }
}

/* The hidden layer of the selected units, act = gelu(v W1[j] + b1[j]). One block (256 threads)
 * per (unit, group of 256 hidden columns, segment of sixteen pairs of the unit's list): a thread
 * owns one column, streams that column of W1 from global memory once (its warp reads 32
 * consecutive columns) and applies each weight to all sixteen pairs' inputs, which sit in
 * shared memory. Shared: v chunk [4d][16]. */
extern "C" __global__ void hsn_win_mlp_hidden(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int d = a.i[2], M = a.i[4], H = a.i[5];
    const int in = 4 * d, groups = (H + 255) / 256;
    const int tid = threadIdx.x, nt = blockDim.x;
    const int j = blockIdx.x % M, group = (blockIdx.x / M) % groups, seg = blockIdx.x / (M * groups);
    const int32_t* pairs = static_cast<const int32_t*>(a.p[24]);
    const int32_t* offsets = static_cast<const int32_t*>(a.p[25]);
    const int first = offsets[j] + seg * 16;
    const int count = min(offsets[j + 1] - first, 16);
    if (count <= 0) return;
    float* vt = shared;  // [in][16]: input c of the segment's pairs (zeros past the last)
    const float* v = fptr(a, 21);
    for (int e = tid; e < 16 * in; e += nt) {
        const int slot = e / in, c = e % in;
        vt[c * 16 + slot] = slot < count ? v[static_cast<int64_t>(pairs[first + slot]) * in + c] : 0.0f;
    }
    __syncthreads();
    const int h = group * 256 + tid;
    if (h >= H) return;
    const float* W = fptr(a, 15) + static_cast<int64_t>(j) * in * H + h;
    float acc[16];
    #pragma unroll
    for (int s2 = 0; s2 < 16; ++s2) acc[s2] = 0.0f;
    int c = 0;
    for (; c + 4 <= in; c += 4) {  // four weights in flight, then their sixty-four products
        const float w0 = W[static_cast<int64_t>(c) * H], w1 = W[static_cast<int64_t>(c + 1) * H];
        const float w2 = W[static_cast<int64_t>(c + 2) * H], w3 = W[static_cast<int64_t>(c + 3) * H];
        const float* x = vt + c * 16;
        #pragma unroll
        for (int s2 = 0; s2 < 16; ++s2)
            acc[s2] += w0 * x[s2] + w1 * x[16 + s2] + w2 * x[32 + s2] + w3 * x[48 + s2];
    }
    for (; c < in; ++c) {
        const float w = W[static_cast<int64_t>(c) * H];
        #pragma unroll
        for (int s2 = 0; s2 < 16; ++s2) acc[s2] += w * vt[c * 16 + s2];
    }
    const float bias = fptr(a, 16)[j * H + h];
    float* act = fout(a, 23);
    #pragma unroll
    for (int s2 = 0; s2 < 16; ++s2)
        if (s2 < count) act[static_cast<int64_t>(pairs[first + s2]) * H + h] = gelu_erf(acc[s2] + bias);
}

/* The output layer of the selected units, out = act W2[j] + b2[j]. One block (256 threads) per
 * (unit, segment of sixteen pairs): a thread owns one output column and one of nt/d strides
 * over the hidden units, streams its part of W2 once and applies it to the sixteen pairs'
 * activations in shared memory. Shared: act chunk [H][16] + partial [nt][16]. */
extern "C" __global__ void hsn_win_mlp_out(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int d = a.i[2], M = a.i[4], H = a.i[5];
    const int tid = threadIdx.x, nt = blockDim.x;
    const int j = blockIdx.x % M, seg = blockIdx.x / M;
    const int32_t* pairs = static_cast<const int32_t*>(a.p[24]);
    const int32_t* offsets = static_cast<const int32_t*>(a.p[25]);
    const int first = offsets[j] + seg * 16;
    const int count = min(offsets[j + 1] - first, 16);
    if (count <= 0) return;
    float* at = shared;             // [H][16]
    float* partial = at + H * 16;   // [nt][16]
    const float* act = fptr(a, 23);
    for (int e = tid; e < 16 * H; e += nt) {
        const int slot = e / H, hh = e % H;
        at[hh * 16 + slot] = slot < count ? act[static_cast<int64_t>(pairs[first + slot]) * H + hh] : 0.0f;
    }
    __syncthreads();
    const int parts = nt / d;  // >= 1: the wrapper requires d <= blockDim
    const int part = tid / d, x = tid % d;
    if (part < parts) {
        const float* W = fptr(a, 17) + static_cast<int64_t>(j) * H * d + x;
        float acc[16];
        #pragma unroll
        for (int s2 = 0; s2 < 16; ++s2) acc[s2] = 0.0f;
        int hh = part;
        for (; hh + 3 * parts < H; hh += 4 * parts) {
            const float w0 = W[hh * d], w1 = W[(hh + parts) * d];
            const float w2 = W[(hh + 2 * parts) * d], w3 = W[(hh + 3 * parts) * d];
            const float* x0 = at + hh * 16;
            const float* x1 = x0 + parts * 16;
            const float* x2 = x1 + parts * 16;
            const float* x3 = x2 + parts * 16;
            #pragma unroll
            for (int s2 = 0; s2 < 16; ++s2)
                acc[s2] += w0 * x0[s2] + w1 * x1[s2] + w2 * x2[s2] + w3 * x3[s2];
        }
        for (; hh < H; hh += parts) {
            const float w = W[hh * d];
            #pragma unroll
            for (int s2 = 0; s2 < 16; ++s2) acc[s2] += w * at[hh * 16 + s2];
        }
        #pragma unroll
        for (int s2 = 0; s2 < 16; ++s2) partial[tid * 16 + s2] = acc[s2];
    }
    __syncthreads();
    if (tid < d) {
        const float bias = fptr(a, 18)[j * d + tid];
        float* out = fout(a, 30);
        for (int s2 = 0; s2 < count; ++s2) {
            float sum = 0.0f;
            for (int q = 0; q < parts; ++q) sum += partial[(q * d + tid) * 16 + s2];
            out[static_cast<int64_t>(pairs[first + s2]) * d + tid] = sum + bias;
        }
    }
}

/* W_out, candidate norm, tanh, the gated update, q, the ages (with the tick) and the selected
 * slots' updated states: one block per batch row, its K slots side by side.
 * Shared: W_out [d,d] + o [K*d] + pre [K*d] + slot_new [K*d] + delta [K*d] + scale [K]. */
extern "C" __global__ void hsn_win_cell_back(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int N = a.i[1], d = a.i[2], K = a.i[3];
    const bool cand_norm = a.i[7] != 0;
    const int r = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    float* Ws = shared;
    float* o = Ws + d * d;
    float* pre = o + K * d;
    float* slot_new = pre + K * d;
    float* delta = slot_new + K * d;
    float* scale = delta + K * d;
    const int64_t* cells = static_cast<const int64_t*>(a.p[5]) + r * K;
    const uint8_t* valid = static_cast<const uint8_t*>(a.p[7]) + r * K;
    const float* alpha = fptr(a, 8) + r * K;
    const float* h = fptr(a, 2) + static_cast<int64_t>(r) * N * d;
    const float* Wout = fptr(a, 19);
    const float* cand_w = fptr(a, 20);
    const float* out = fptr(a, 30) + static_cast<int64_t>(r) * K * d;
    const float* gate = fptr(a, 22) + static_cast<int64_t>(r) * K * d;
    for (int e = tid; e < d * d; e += nt) Ws[(e % d) * d + e / d] = Wout[e];  // transposed
    for (int e = tid; e < K * d; e += nt) {
        const int k = e / d;
        // without the candidate norm alpha mixes before W_out; with it, after the norm
        o[e] = valid[k] ? (cand_norm ? out[e] : alpha[k] * out[e]) : 0.0f;
    }
    __syncthreads();
    for (int e = tid; e < K * d; e += nt) {
        const int k = e / d, x = e % d;
        pre[e] = dot_strided(Ws + x, d, o + k * d, 1, d);
    }
    __syncthreads();
    if (cand_norm && tid < K) {  // one fixed-order sum per slot
        const float ms = dot_strided(pre + tid * d, 1, pre + tid * d, 1, d);
        scale[tid] = rsqrtf(ms / static_cast<float>(d) + a.f[1]);
    }
    __syncthreads();
    for (int e = tid; e < K * d; e += nt) {
        const int k = e / d, x = e % d;
        float next = 0.0f, change = 0.0f;
        if (valid[k]) {
            const float value = cand_norm ? alpha[k] * (pre[e] * scale[k] * cand_w[x]) : pre[e];
            const float c = tanhf(value);
            const float old = h[static_cast<int>(cells[k]) * d + x];
            const float g = gate[e];
            next = (1.0f - g) * old + g * c;
            change = next - old;
        }
        slot_new[e] = next;
        delta[e] = change;
    }
    float* h_new = fout(a, 26) + static_cast<int64_t>(r) * N * d;
    copy_strided(h_new, h, N * d, tid, nt);
    const int64_t* age = static_cast<const int64_t*>(a.p[4]) + r * N;
    int64_t* age_new = static_cast<int64_t*>(a.p[28]) + r * N;
    for (int n = tid; n < N; n += nt) age_new[n] = age[n] + 1;
    __syncthreads();
    for (int x = tid; x < d; x += nt) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += delta[k * d + x];
        fout(a, 27)[r * d + x] = fptr(a, 3)[r * d + x] + sum / static_cast<float>(N);
    }
    for (int e = tid; e < K * d; e += nt) {
        const int k = e / d;
        if (valid[k]) h_new[static_cast<int>(cells[k]) * d + e % d] = slot_new[e];
    }
    for (int k = tid; k < K; k += nt)
        if (valid[k]) age_new[static_cast<int>(cells[k])] = 0;
    __syncthreads();
    float* h_sel = fout(a, 29) + static_cast<int64_t>(r) * K * d;
    for (int e = tid; e < K * d; e += nt) {
        const int k = e / d;
        h_sel[e] = valid[k] ? slot_new[e] : h_new[e % d];  // a padded slot reads cell 0, as the tape
    }
}
