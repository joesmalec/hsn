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
/* e / d and e % d with `shift` = log2(d) when d is a power of two, else -1 (an integer division
 * costs a thread dozens of instructions; a block's latency is its slowest thread's count). */
__device__ __forceinline__ int log2_or_minus1(int d) {
    return (d > 0 && (d & (d - 1)) == 0) ? __ffs(d) - 1 : -1;
}
__device__ __forceinline__ int quot(int e, int d, int shift) {
    return shift >= 0 ? e >> shift : e / d;
}
__device__ __forceinline__ int rem(int e, int d, int shift) {
    return shift >= 0 ? e & (d - 1) : e % d;
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

/* The backward recurrence of the rows that ran the dense round (hsn.train.split, "Dense-fallback
 * rows"): every cell is written with h' = (1 - b) h + b c and q' = mean(h'), which does not
 * read q. One block per row; A [N, d] and R [d] in shared memory.
 *   p: 0 c, 1 b, 2 h_old [B',T,N,d] f32; 3 reset [B',T] u8; 4 d_q_pre, 5 d_q_post [B',T,d] f32
 *      (read only when flagged); 6 dc, 7 db [B',T,N,d] f32.
 *   i: 0 B', 1 T, 2 N, 3 d, 4 has_pre, 5 has_post.
 */
extern "C" __global__ void hsn_win_recurrence_dense(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int T = a.i[1], N = a.i[2], d = a.i[3];
    const bool has_pre = a.i[4], has_post = a.i[5];
    const int row = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const float* c = fptr(a, 0);
    const float* b = fptr(a, 1);
    const float* h_old = fptr(a, 2);
    const uint8_t* reset = static_cast<const uint8_t*>(a.p[3]);
    const float* pre = fptr(a, 4);
    const float* post = fptr(a, 5);
    float* dc = fout(a, 6);
    float* db = fout(a, 7);
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
        for (int e = tid; e < N * d; e += nt) {
            const int64_t o = rt * N * d + e;
            const float g = A[e] + R[e % d] * inv;
            const float gate = b[o];
            dc[o] = gate * g;
            db[o] = (c[o] - h_old[o]) * g;
            A[e] = (1.0f - gate) * g;
        }
        __syncthreads();
        const bool cut = reset[rt] != 0;
        for (int x = tid; x < d; x += nt) R[x] = cut || !has_pre ? 0.0f : pre[rt * d + x];
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

/* The hard decision of one position (hsn.model.router.Router.select with mode "topk", no
 * exploration, the refresh probes optional): the forward sweep's selection as one kernel in
 * place of the sort / gather / scatter chain. One block per batch row, at least N threads.
 *   p: 0 logits [B,N,M] f32; 1 age [B,n_age,N] i64 (record t is read); 2 cost [M] f32;
 *      3 cells [B,n_out,K] i64, 4 units [B,n_out,K] i64, 5 valid [B,n_out,K] u8,
 *      6 theta [B,n_out] f32, 7 refreshed [B,n_out,K] u8 (outputs: record t is written in full).
 *   i: 0 B, 1 N, 2 M, 3 K, 4 age_max, 5 probes, 6 t, 7 n_out, 8 n_age.
 *      (A plain call is t = 0, n_out = n_age = 1; the sweep passes its tape and the position.)
 *   f: 0 lambda_cost, 1 cell_overhead, 2 shared_cost, 3 limit (= budget * (1 + slack)).
 * Shared: (4 N + M + 4 K + 1 + 3 ceil(N / 32)) words. The steps are the router's, in its order
 * and with its tie rules:
 *   score_i = max_j (logit_ij - lambda * cost_j), unit_i the first maximiser;
 *   the K best cells in descending score, equal scores in index order (a stable sort);
 *   the longest prefix whose cumulative cost fits the limit is valid (padded slots: cell 0, unit 0);
 *   theta = (lowest selected + highest unselected score) / 2;
 *   refresh probes: the stalest unselected cells with age > age_max (equal ages in index order)
 *   replace the valid slots from the last one backwards, unless the row then exceeds the limit.
 * No thread walks global memory: every value is read once, by its own thread, into shared memory
 * (a loop of dependent global loads costs a kernel far more than its arithmetic).
 */
extern "C" __global__ void hsn_win_select(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int N = a.i[1], M = a.i[2], K = a.i[3], age_max = a.i[4];
    const bool probes = a.i[5] != 0;
    const int t = a.i[6], n_out = a.i[7], n_age = a.i[8];
    const int row = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const float lambda = a.f[0], overhead = a.f[1], shared_cost = a.f[2], limit = a.f[3];
    const float* logits = fptr(a, 0) + static_cast<int64_t>(row) * N * M;
    const int64_t* age =
        static_cast<const int64_t*>(a.p[1]) + (static_cast<int64_t>(row) * n_age + t) * N;
    const float* cost = fptr(a, 2);
    const int64_t rec = static_cast<int64_t>(row) * n_out + t;
    int64_t* cells = static_cast<int64_t*>(a.p[3]) + rec * K;
    int64_t* units = static_cast<int64_t*>(a.p[4]) + rec * K;
    uint8_t* valid = static_cast<uint8_t*>(a.p[5]) + rec * K;
    float* theta = fout(a, 6) + rec;
    uint8_t* refreshed = static_cast<uint8_t*>(a.p[7]) + rec * K;
    float* score = shared;                                // [N]
    int* unit = reinterpret_cast<int*>(shared + N);       // [N]
    int* order = reinterpret_cast<int*>(shared + 2 * N);  // [N] cells by descending score
    float* stale = shared + 3 * N;                        // [N] a stale unselected cell's age, else -1
    float* cost_s = shared + 4 * N;                       // [M]
    int* out_cell = reinterpret_cast<int*>(shared + 4 * N + M);  // [K]
    int* out_unit = out_cell + K;                         // [K]
    int* out_flag = out_unit + K;                         // [K] bit 0 valid, bit 1 refreshed
    int* slot = out_flag + K;                             // [K] the scored selection's cells
    int* n_valid_s = slot + K;                            // [1] the number of valid slots
    const int warps = (N + 31) / 32;                      // the warps that hold cells
    float* warp_low = reinterpret_cast<float*>(n_valid_s + 1);  // [warps] lowest selected score
    float* warp_high = warp_low + warps;                  // [warps] highest unselected score
    int* warp_stale = reinterpret_cast<int*>(warp_high + warps);  // [warps] stale unselected cells
    // score and unit of every cell. With M a power of two that divides a warp and a thread per
    // logit, the M lanes of a cell reduce by shuffles; otherwise one thread per cell loops.
    const int m_shift = log2_or_minus1(M);
    const bool lanes = m_shift >= 0 && M <= 32 && nt >= N * M;
    if (lanes) {
        const int j = tid & (M - 1);
        float s = -INFINITY;
        if (tid < N * M) s = lambda > 0.0f ? logits[tid] - lambda * cost[j] : logits[tid];
        int arg = j;
        for (int w = 1; w < M; w <<= 1) {  // the first maximiser among the cell's lanes
            const float so = __shfl_xor_sync(0xffffffff, s, w);
            const int ao = __shfl_xor_sync(0xffffffff, arg, w);
            if (so > s || (so == s && ao < arg)) {
                s = so;
                arg = ao;
            }
        }
        if (tid < N * M && j == 0) {
            score[tid >> m_shift] = s;
            unit[tid >> m_shift] = arg;
        }
    } else if (tid < N) {
        const float* r = logits + tid * M;
        float best = lambda > 0.0f ? r[0] - lambda * cost[0] : r[0];
        int arg = 0;
        for (int j = 1; j < M; ++j) {
            const float s = lambda > 0.0f ? r[j] - lambda * cost[j] : r[j];
            if (s > best) {
                best = s;
                arg = j;
            }
        }
        score[tid] = best;
        unit[tid] = arg;
    }
    float key = -1.0f;  // this cell's age if it is over age_max
    if (tid < N && probes) {
        const int64_t ag = age[tid];
        if (ag > static_cast<int64_t>(age_max)) key = static_cast<float>(ag);
    }
    for (int j = tid; j < M; j += nt) cost_s[j] = cost[j];
    __syncthreads();
    float mine = 0.0f;
    int rank = N;  // this cell's rank in the stable descending order
    if (tid < N) {
        mine = score[tid];
        rank = 0;
        for (int j = 0; j < N; ++j) rank += (score[j] > mine) || (score[j] == mine && j < tid);
        order[rank] = tid;
    }
    __syncthreads();
    if (tid == 0) {
        // the budget: the longest prefix that fits (cumulative costs are summed in slot order)
        float cumulative = 0.0f;
        int n_valid = 0;
        for (int k = 0; k < K; ++k) {
            const int c = order[k], u = unit[c];
            cumulative += overhead + cost_s[u];
            const bool ok = shared_cost + cumulative <= limit;
            out_cell[k] = ok ? c : 0;
            out_unit[k] = ok ? u : 0;
            out_flag[k] = ok ? 1 : 0;
            slot[k] = ok ? c : 0;
            n_valid += ok;
        }
        *n_valid_s = n_valid;
    }
    __syncthreads();
    // the straight-through threshold of the scored decision (before the probes) and the stale
    // unselected cells: each warp reduces its cells, thread 0 joins the warps
    const int n_valid = *n_valid_s;
    const bool inside = rank < n_valid;
    float lowest_in = (tid < N && inside) ? mine : INFINITY;
    float highest_out = (tid < N && !inside) ? mine : -INFINITY;
    const bool is_stale = tid < N && !inside && key >= 0.0f;
    int n_stale = is_stale ? 1 : 0;
    if (tid < N) stale[tid] = is_stale ? key : -1.0f;
    for (int w = 16; w > 0; w >>= 1) {
        lowest_in = fminf(lowest_in, __shfl_xor_sync(0xffffffff, lowest_in, w));
        highest_out = fmaxf(highest_out, __shfl_xor_sync(0xffffffff, highest_out, w));
        n_stale += __shfl_xor_sync(0xffffffff, n_stale, w);
    }
    if ((tid & 31) == 0 && (tid >> 5) < warps) {
        warp_low[tid >> 5] = lowest_in;
        warp_high[tid >> 5] = highest_out;
        warp_stale[tid >> 5] = n_stale;
    }
    __syncthreads();
    if (tid == 0) {
        lowest_in = INFINITY;
        highest_out = -INFINITY;
        n_stale = 0;
        for (int w = 0; w < warps; ++w) {
            lowest_in = fminf(lowest_in, warp_low[w]);
            highest_out = fmaxf(highest_out, warp_high[w]);
            n_stale += warp_stale[w];
        }
        *theta = 0.5f * (lowest_in + highest_out);
        const int n_force = n_stale < n_valid ? n_stale : n_valid;
        for (int j = 0; j < n_force; ++j) {  // the j-th stalest cell goes to slot n_valid - 1 - j
            int pick = 0;
            float stalest = stale[0];
            for (int n = 1; n < N; ++n) {
                if (stale[n] > stalest) {
                    stalest = stale[n];
                    pick = n;
                }
            }
            stale[pick] = -2.0f;
            const int k = n_valid - 1 - j;
            out_cell[k] = pick;
            out_unit[k] = unit[pick];
            out_flag[k] |= 2;
        }
        if (n_force > 0) {
            // the budget guarantee: a row the probes pushed over the limit keeps its scored selection
            float spent = 0.0f;
            for (int k = 0; k < K; ++k)
                spent += (out_flag[k] & 1) ? overhead + cost_s[out_unit[k]] : 0.0f;
            if (shared_cost + spent > limit) {
                for (int k = 0; k < K; ++k) {
                    const int c = slot[k];
                    out_cell[k] = c;
                    out_unit[k] = (out_flag[k] & 1) ? unit[c] : 0;
                    out_flag[k] &= 1;
                }
            }
        }
    }
    __syncthreads();
    for (int k = tid; k < K; k += nt) {
        cells[k] = out_cell[k];
        units[k] = out_unit[k];
        valid[k] = static_cast<uint8_t>(out_flag[k] & 1);
        refreshed[k] = static_cast<uint8_t>((out_flag[k] >> 1) & 1);
    }
}

/* The end of one position's round, after the matrix products (PDGCell.step_sparse from the
 * candidate norm on, hsn.kernels.native.gated_delta_update and CellGraphState.tick), as one
 * kernel in place of the gather / scatter / reduction chain. One block per batch row.
 *   p: 0 y [B,K,d] f32 (W_out of the selected unit's output), 1 gate [B,K,d] f32 (the gate's
 *        pre-activation), 2 h [B,n_state,N,d] f32, 3 q [B,n_state,d] f32,
 *      4 age [B,n_state,N] i64 (record t of the three is read), 5 pos [B] i64 (with has_pos),
 *      6 cells [B,n_slot,K] i64, 7 valid [B,n_slot,K] u8 (record t), 8 cand_w [d] f32 (read
 *        only with the norm);
 *      9 h_next, 10 q_next, 11 age_next: as 2-4, record t_out is written (they may be the
 *        buffers of 2-4 when t_out != t), 12 pos_next [B] i64 (with has_pos),
 *      13 h_sel [B,n_slot,K,d] (the slots' cells after the update: cell 0 on padded slots),
 *      14 h_max [B,n_slot] (max |h| of the row after the update; NaN if any entry is NaN),
 *      15 reset [B,n_slot] u8 (with has_reset: the row's flag at t + 1 zeroes the written
 *        records h_next / q_next / age_next, the pre-round state of the next position),
 *      16 q_post [B,n_slot,d] (with has_post: the summary after the update, never reset).
 *   i: 0 B, 1 N, 2 d, 3 K, 4 cand_norm (0: none, 1: rms), 5 t, 6 n_state, 7 n_slot, 8 t_out,
 *      9 has_reset, 10 has_pos, 11 has_post.   f: 0 cand_eps.
 *      (A plain call is t = t_out = 0, n_state = n_slot = 1 with separate outputs; the sweep
 *      passes its tape, reads record t and writes record t + 1 of the same buffers.)
 * Shared: (2 K + 1 + N + K ceil(d / 32) + 2 K d) words.
 *   c = tanh(rms(y) * cand_w)   (or tanh(y));   b = 1 / (1 + exp(-gate))
 *   h'[cell] = (1 - b) h[cell] + b c  on the valid slots;   q' = q + sum_k (h' - h)[k] / N
 *   age' = 0 on the written cells, age + 1 elsewhere;   pos' = pos + 1.
 * The valid slots of a row name distinct cells (Router.select), so no two threads write one cell.
 * As in hsn_win_select, no thread walks global memory in a dependent loop where a thread per
 * element can read it once.
 */
extern "C" __global__ void hsn_win_cell_update(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int N = a.i[1], d = a.i[2], K = a.i[3];
    const bool cand_norm = a.i[4] != 0;
    const int t = a.i[5], n_state = a.i[6], n_slot = a.i[7], t_out = a.i[8];
    const bool has_reset = a.i[9] != 0, has_pos = a.i[10] != 0, has_post = a.i[11] != 0;
    const float cand_eps = a.f[0];
    const int row = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int64_t s_in = static_cast<int64_t>(row) * n_state + t;
    const int64_t s_out = static_cast<int64_t>(row) * n_state + t_out;
    const int64_t rec = static_cast<int64_t>(row) * n_slot + t;
    const float* y = fptr(a, 0) + static_cast<int64_t>(row) * K * d;
    const float* gate = fptr(a, 1) + static_cast<int64_t>(row) * K * d;
    const float* h = fptr(a, 2) + s_in * N * d;
    const float* q = fptr(a, 3) + s_in * d;
    const int64_t* age = static_cast<const int64_t*>(a.p[4]) + s_in * N;
    const int64_t* cells = static_cast<const int64_t*>(a.p[6]) + rec * K;
    const uint8_t* valid = static_cast<const uint8_t*>(a.p[7]) + rec * K;
    const float* cand_w = fptr(a, 8);
    float* h_next = fout(a, 9) + s_out * N * d;
    float* q_next = fout(a, 10) + s_out * d;
    int64_t* age_next = static_cast<int64_t*>(a.p[11]) + s_out * N;
    float* h_sel = fout(a, 13) + rec * K * d;
    float* h_max = fout(a, 14) + rec;
    const int parts = (d + 31) / 32;
    int* cell_of = reinterpret_cast<int*>(shared);        // [K] the slot's cell, -1 if padded
    float* scale = shared + K;                            // [K] 1 / rms(y_k)
    int* top = reinterpret_cast<int*>(shared + 2 * K);    // [1] bits of the row's max |h|
    int* slot_of = top + 1;                               // [N] the slot that writes the cell, or -1
    float* part = shared + 2 * K + 1 + N;                 // [K, parts] sums of squares per warp
    const int d_shift = log2_or_minus1(d);
    float* h_new = part + K * parts;                      // [K, d] the valid slots' new states
    float* delta = h_new + K * d;                         // [K, d] h_new - h_old (0 if padded)
    // the candidate norm's sums of squares: with a thread per slot element and d a multiple of
    // a warp, each warp sums its 32 squares by shuffles; otherwise a thread per slot loops
    const bool lanes = cand_norm && d % 32 == 0 && nt >= K * d;
    float ye = 0.0f;
    if (lanes) {
        float sq = 0.0f;
        if (tid < K * d) {
            ye = y[tid];
            sq = ye * ye;
        }
        for (int w = 16; w > 0; w >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, w);
        if (tid < K * d && tid % 32 == 0) part[tid / 32] = sq;
    }
    for (int k = tid; k < K; k += nt) cell_of[k] = valid[k] ? static_cast<int>(cells[k]) : -1;
    if (tid == 0) *top = 0;
    bool clear = false;
    if (has_reset) clear = static_cast<const uint8_t*>(a.p[15])[rec + 1] != 0;
    __syncthreads();
    if (tid < K) {
        float inv = 1.0f;
        if (cand_norm) {
            float sum = 0.0f;
            if (lanes) {
                for (int w = 0; w < parts; ++w) sum += part[tid * parts + w];
            } else {
                for (int x = 0; x < d; ++x) sum += y[tid * d + x] * y[tid * d + x];
            }
            inv = rsqrtf(sum / static_cast<float>(d) + cand_eps);
        }
        scale[tid] = inv;
    }
    for (int n = tid; n < N; n += nt) {
        int k = -1;
        for (int j = 0; j < K; ++j) k = cell_of[j] == n ? j : k;
        slot_of[n] = k;
    }
    __syncthreads();
    for (int e = tid; e < K * d; e += nt) {
        const int k = quot(e, d, d_shift), x = rem(e, d, d_shift);
        const int n = cell_of[k];
        float value = 0.0f, change = 0.0f;
        if (n >= 0) {
            const float old = h[n * d + x];
            const float yv = lanes ? ye : y[e];
            const float pre = cand_norm ? yv * scale[k] * cand_w[x] : yv;
            const float c = tanhf(pre);
            const float b = 1.0f / (1.0f + expf(-gate[e]));
            value = (1.0f - b) * old + b * c;
            change = value - old;
        }
        h_new[e] = value;
        delta[e] = change;
    }
    __syncthreads();
    for (int x = tid; x < d; x += nt) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += delta[k * d + x];
        const float after = q[x] + sum / static_cast<float>(N);
        if (has_post) fout(a, 16)[rec * d + x] = after;
        q_next[x] = clear ? 0.0f : after;
    }
    for (int n = tid; n < N; n += nt) age_next[n] = (clear || slot_of[n] >= 0) ? 0 : age[n] + 1;
    if (has_pos && tid == 0)
        static_cast<int64_t*>(a.p[12])[row] = static_cast<const int64_t*>(a.p[5])[row] + 1;
    // the slots' cells after the update (a padded slot reads cell 0, as the tape block's gather)
    for (int e = tid; e < K * d; e += nt) {
        const int k = quot(e, d, d_shift), x = rem(e, d, d_shift);
        float value;
        if (cell_of[k] >= 0) {
            value = h_new[e];
        } else {
            const int zero = slot_of[0];
            value = zero >= 0 ? h_new[zero * d + x] : h[x];
        }
        h_sel[e] = value;
    }
    // the next state: the written cells from shared memory, the others carried; the row's
    // max |h| in the integer order of the bit patterns (non-negative floats; a NaN is the largest)
    int mine = 0;
    for (int e = tid; e < N * d; e += nt) {
        const int n = quot(e, d, d_shift), x = rem(e, d, d_shift);
        const int k = slot_of[n];
        const float value = k >= 0 ? h_new[k * d + x] : h[e];
        h_next[e] = clear ? 0.0f : value;
        const int bits = __float_as_int(fabsf(value));
        mine = bits > mine ? bits : mine;
    }
    for (int w = 16; w > 0; w >>= 1) {
        const int other = __shfl_xor_sync(0xffffffff, mine, w);
        mine = other > mine ? other : mine;
    }
    if (tid % 32 == 0) atomicMax(top, mine);
    __syncthreads();
    if (tid == 0) *h_max = __int_as_float(*top);
}

/* The trainer's per-step passes (P2.10b) share this fatbin and its launcher. */
#include "hsn_trainer.cu"
