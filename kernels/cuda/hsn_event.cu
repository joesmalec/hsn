/* The fused word-event round, forward (docs/wev_fused_event.md; hsn.train.event_fused is the
 * reference, hsn.kernels.event the launcher). Included by hsn_window.cu: device code only, its
 * helpers (bf16 casts, load_value / store_value, gelu_erf, sigmoidf) are in scope.
 *
 * An event slot's forward pass is three kernels and four matrix products:
 *   hsn_ev_finish  finish slot t - 1 (candidate, gated update, summary, ages, read-out), then
 *                  the reset of slot t and the front's hidden layer on the carried summary
 *   products       [u; z_hat] = act_f W2' + b2' (the front's second layer) and V h of every cell
 *   hsn_ev_decide  the scorer, the top-K decision, the messages, the cell input and its norm
 *   product        first = v W1 (+ b1): every unit's first layer and the gate, one matrix product
 *   hsn_ev_pick    the chosen unit's hidden row, its activation, the gate
 *   product        out = act W2 (+ b2): the second layers with W_out folded in
 * One block per batch row in the kernels: everything a row computes between two products
 * depends on that row alone. Every value the backward pass reads is written to the tape, whose
 * buffers are laid out [B, E, ...] (the reverse sweep's own buffers among them).
 *
 * `half` (the products run in bf16): where the compiled autograd round holds a bf16 value (what
 * a matrix product reads and returns) the value is rounded to bf16 here too; the state, the
 * norms, the gate's sigmoid and the candidate stay fp32, as there.
 */

namespace {
__device__ __forceinline__ float ev_round(float x, bool half) {
    return half ? bf16_to_float(float_to_bf16(x)) : x;
}
/* model.activation: 0 gelu (erf), 1 relu, 2 silu, 3 tanh. */
__device__ __forceinline__ float ev_activation(float x, int code) {
    if (code == 0) return gelu_erf(x);
    if (code == 1) return x > 0.0f ? x : 0.0f;
    if (code == 2) return x * sigmoidf(x);
    return tanhf(x);
}
}  // namespace

/* What keeps these kernels short. The first version was one kernel between the products, with
 * the scorer, the messages and the front as plain loops over rows of the weights: 435 us at the
 * recipe's shape (12 of 64 cells, d 64). Three things cost that, none of them arithmetic:
 *   - Shared memory has 32 banks of four-byte words; lanes of a warp that read different words
 *     of one bank are served one after the other. A row stride of d = 64 words puts a warp's
 *     32 lanes on one bank at every step (the messages alone took 207 us). Here a warp's lanes
 *     read consecutive words or one word; the row's state is staged with a stride of d + 1.
 *   - A warp's read of global memory is one transaction when its lanes read consecutive
 *     words, and 32 when each lane reads its own row: weights read from global memory are laid
 *     out transposed, the lane index last.
 *   - A dot product in a kernel is a chain of steps on one warp, about 0.1 us a step whatever
 *     the layout (122 us with the two points above fixed: the front's second layer 18, the
 *     messages 31). So no kernel multiplies a matrix that a product can: the front's second
 *     layer and V h are cuBLAS products between the two kernels. What is left in the kernels
 *     is the front's first layer on the summary (one read of the weights' column per hidden
 *     unit) and the scorer's logits (a thread per logit).
 */

/* hsn_ev_finish: after the second product of slot t - 1. Finish that slot, then start slot t
 * as far as the next products' operands.
 *   p:  0 h [E+1,B,N,d] f32 (record t: the state slot t reads; record t is written here by the
 *         finish of slot t - 1 and reset in place; slot-major, so that the block's states are
 *         one contiguous [E B N, d] matrix for the scorer's gradient)
 *       1 q [B,E+1,d] f32 (record t: the summary before slot t's reset)
 *       2 q_pre [B,E,d] f32 (out: the summary slot t reads, after its reset)
 *       3 age [B,E+1,N] i64 (as q; reset in place)
 *       4 reset [B,E] u8, 5 live [B,E] u8
 *       6 front_static [B,E,Hf] (f32 or bf16: [z; ctx] W1' + b1, once per block)
 *       7 front_q' [d,Hf] f32 (transposed)
 *       8 cells, 9 units [B,E,K] i64, 10 valid [B,E,K] u8 (slot t - 1's)
 *      11 out [B K, M d] (f32 or bf16: the second product of slot t - 1)
 *      12 b2 [M d] f32 (read unless the product added it), 13 cand_gain [d]
 *      14 b, 15 h_old [B,E,K,d] f32 (slot t - 1's); 16 y, 17 c [B,E,K,d] f32 (out)
 *      18 readout [B,E,d] f32 (out), 19 abs_max [B,E] f32 (out)
 *      20 act_f [B,Hf] (f32 or bf16, out: the front's hidden layer of slot t, the operand of
 *         its second layer), 21 h_op [B,N,d] f32 (out: the state slot t reads with the values
 *         the products read, rounded to bf16 with `half`; the operand of the state's product)
 *   i:  0 B, 1 E, 2 N, 3 M, 4 K, 5 d, 6 Hf, 7 t, 8 flags (1 start slot t, 2 finish slot t - 1,
 *       4 half, 64 the second product added b2), 9 front activation.
 *   f:  0 cand_eps.
 * Shared (words): N (d + 1) + d + N + 3 K + K + 3 K d + threads / 32.
 */
extern "C" __global__ void hsn_ev_finish(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int B = a.i[0], E = a.i[1], N = a.i[2], M = a.i[3], K = a.i[4], d = a.i[5];
    const int Hf = a.i[6], t = a.i[7], flags = a.i[8], act_front = a.i[9];
    const bool start = (flags & 1) != 0, finish = (flags & 2) != 0, half = (flags & 4) != 0;
    const bool bias2_done = (flags & 64) != 0;
    const float cand_eps = a.f[0];
    const int row = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int d_shift = log2_or_minus1(d);
    const int hw = d + 1;  // the stride of the staged state's rows
    float* h_s = shared;                         // [N, d + 1] the row's state
    float* q_s = h_s + N * hw;                   // [d]
    float* marks = q_s + d;                      // [N] the written cells
    int* out_cell = reinterpret_cast<int*>(marks + N);    // [K]
    int* out_unit = out_cell + K;                // [K]
    int* out_flag = out_unit + K;                // [K]
    float* k_a = reinterpret_cast<float*>(out_flag + K);  // [K]
    float* slot_a = k_a + K;                     // [K d]
    float* slot_b = slot_a + K * d;              // [K d]
    float* slot_c = slot_b + K * d;              // [K d]
    float* part = slot_c + K * d;                // [threads / 32]
    float* h_tape = fout(a, 0);
    float* q_tape = fout(a, 1);
    int64_t* age_tape = static_cast<int64_t*>(a.p[3]);

    if (finish) {
        // ---- the end of slot tp = t - 1: candidate, gated update, summary, ages, read-out ----
        const int tp = t - 1;
        const int64_t rec = static_cast<int64_t>(row) * E + tp;
        const bool alive = static_cast<const uint8_t*>(a.p[5])[rec] != 0;
        const float* h_in = h_tape + (static_cast<int64_t>(tp) * B + row) * N * d;
        float* h_out = h_tape + (static_cast<int64_t>(tp + 1) * B + row) * N * d;
        const float* q_pre = fptr(a, 2) + rec * d;
        float* q_out = q_tape + (static_cast<int64_t>(row) * (E + 1) + tp + 1) * d;
        const int64_t* age_in = age_tape + (static_cast<int64_t>(row) * (E + 1) + tp) * N;
        int64_t* age_out = age_tape + (static_cast<int64_t>(row) * (E + 1) + tp + 1) * N;
        const float* b2 = fptr(a, 12);
        const float* cand_gain = fptr(a, 13);
        const float* b_t = fptr(a, 14) + rec * K * d;
        const float* h_old_t = fptr(a, 15) + rec * K * d;
        float* y_t = fout(a, 16) + rec * K * d;
        float* c_t = fout(a, 17) + rec * K * d;
        for (int k = tid; k < K; k += nt) {
            out_cell[k] = static_cast<int>(static_cast<const int64_t*>(a.p[8])[rec * K + k]);
            out_unit[k] = static_cast<int>(static_cast<const int64_t*>(a.p[9])[rec * K + k]);
            out_flag[k] = static_cast<const uint8_t*>(a.p[10])[rec * K + k];
        }
        for (int i = tid; i < N; i += nt) marks[i] = 0.0f;
        // the row's state before the slot: read once, into shared memory
        for (int e = tid; e < N * d; e += nt) {
            const int i = quot(e, d, d_shift);
            h_s[i * hw + (e - i * d)] = h_in[e];
        }
        __syncthreads();
        // y: the chosen unit's block of the second product; the gate and the old state with it
        for (int e = tid; e < K * d; e += nt) {
            const int k = quot(e, d, d_shift), x = e - k * d;
            const int j = out_unit[k];
            const float raw = load_value(
                a.p[11], (static_cast<int64_t>(row) * K + k) * (M * d) + j * d + x, half);
            const float bias = bias2_done ? 0.0f : b2[j * d + x];
            const float bv = b_t[e], ho = h_old_t[e];
            const float yv = bias2_done ? raw : ev_round(raw + bias, half);
            slot_a[e] = yv;
            slot_b[e] = bv;
            slot_c[e] = ho;
            y_t[e] = yv;
        }
        __syncthreads();
        {  // a slot's sum of squares, split over 16 lanes and folded by shuffles
            const int lanes = 16;
            float sum = 0.0f;
            if (tid < K * lanes) {
                const int k = tid / lanes, piece = tid - k * lanes;
                const int len = (d + lanes - 1) / lanes;
                for (int x = piece * len; x < (piece + 1) * len && x < d; ++x)
                    sum += slot_a[k * d + x] * slot_a[k * d + x];
            }
            for (int w = 1; w < lanes; w <<= 1) sum += __shfl_xor_sync(0xffffffff, sum, w);
            if (tid < K * lanes && (tid & (lanes - 1)) == 0)
                k_a[tid / lanes] = rsqrtf(sum / static_cast<float>(d) + cand_eps);
            for (int k = tid; k < K; k += nt)
                if (out_flag[k]) marks[out_cell[k]] = 1.0f;
        }
        __syncthreads();
        for (int e = tid; e < K * d; e += nt) {
            const int k = quot(e, d, d_shift), x = e - k * d;
            const float cv = tanhf(slot_a[e] * k_a[k] * cand_gain[x]);
            c_t[e] = cv;
            const float ho = slot_c[e], bv = slot_b[e];
            const float hn = out_flag[k] ? (1.0f - bv) * ho + bv * cv : ho;
            slot_b[e] = hn;       // the slot's state after the update
            slot_c[e] = hn - ho;  // what it adds to the summary
            if (out_flag[k]) h_s[out_cell[k] * hw + x] = hn;  // distinct cells: one writer each
        }
        __syncthreads();
        int n_valid = 0;
        for (int k = 0; k < K; ++k) n_valid += out_flag[k];
        for (int x = tid; x < d; x += nt) {
            float delta = 0.0f, read = 0.0f;
            for (int k = 0; k < K; ++k) {
                delta += slot_c[k * d + x];
                read += out_flag[k] ? slot_b[k * d + x] : 0.0f;
            }
            q_out[x] = q_pre[x] + delta / static_cast<float>(N);
            fout(a, 18)[rec * d + x] = read / static_cast<float>(n_valid > 0 ? n_valid : 1);
        }
        for (int i = tid; i < N; i += nt) {
            const int64_t ag = age_in[i];
            age_out[i] = alive ? (marks[i] != 0.0f ? 0 : ag + 1) : ag;
        }
        // the state after the slot, and max |h| of the row (NaN if any entry is NaN)
        float mx = 0.0f;
        for (int e = tid; e < N * d; e += nt) {
            const int i = quot(e, d, d_shift);
            const float hv = h_s[i * hw + (e - i * d)];
            h_out[e] = hv;
            const float v = fabsf(hv);
            mx = (v > mx || v != v) ? v : mx;
        }
        for (int w = 16; w > 0; w >>= 1) {
            const float o = __shfl_xor_sync(0xffffffff, mx, w);
            mx = (o > mx || o != o) ? o : mx;
        }
        if ((tid & 31) == 0) part[tid >> 5] = mx;
        __syncthreads();
        if (tid == 0) {
            float best = 0.0f;
            for (int i = 0; i < nt / 32; ++i) {
                const float v = part[i];
                if (best == best) best = (v > best || v != v) ? v : best;
            }
            fout(a, 19)[rec] = best;
        }
        __syncthreads();
    }
    if (!start) return;

    // ---- slot t: the reset, the operands of the front's second layer and of V h ----
    const int64_t rec = static_cast<int64_t>(row) * E + t;
    float* h = h_tape + (static_cast<int64_t>(t) * B + row) * N * d;
    int64_t* age = age_tape + (static_cast<int64_t>(row) * (E + 1) + t) * N;
    const float* q_in = q_tape + (static_cast<int64_t>(row) * (E + 1) + t) * d;
    const bool rs = static_cast<const uint8_t*>(a.p[4])[rec] != 0;
    const int64_t op_row = static_cast<int64_t>(row) * N * d;
    if (rs) {  // a reset zeroes the state in place
        for (int e = tid; e < N * d; e += nt) {
            h[e] = 0.0f;
            fout(a, 21)[op_row + e] = 0.0f;
        }
        for (int i = tid; i < N; i += nt) age[i] = 0;
    } else if (finish) {  // the finish above left the state after slot t - 1 in shared memory
        for (int e = tid; e < N * d; e += nt) {
            const int i = quot(e, d, d_shift);
            fout(a, 21)[op_row + e] = ev_round(h_s[i * hw + (e - i * d)], half);
        }
    } else {
        for (int e = tid; e < N * d; e += nt) fout(a, 21)[op_row + e] = ev_round(h[e], half);
    }
    for (int x = tid; x < d; x += nt) {
        const float qv = rs ? 0.0f : q_in[x];
        fout(a, 2)[rec * d + x] = qv;
        q_s[x] = ev_round(qv, half);
    }
    __syncthreads();
    // the front's hidden layer: the block's static part plus the summary's (lane j reads column
    // j of the transposed weight: a warp's lanes read consecutive words)
    const float* fqt = fptr(a, 7);
    for (int j = tid; j < Hf; j += nt) {
        const float base = load_value(a.p[6], rec * Hf + j, half);
        const float acc = base + dot_strided(fqt + j, Hf, q_s, 1, d);
        store_value(a.p[20], static_cast<int64_t>(row) * Hf + j, half,
            ev_activation(ev_round(acc, half), act_front));
    }
}

/* hsn_ev_decide: after the front's second layer and the state's product of slot t. The scorer,
 * the decision, the messages, the cell input and its norm; writes the first product's operand.
 *   p:  0 h [E+1,B,N,d] f32 (record t), 1 q_pre [B,E,d] f32, 2 age [B,E+1,N] i64 (record t)
 *       3 live [B,E] u8, 4 z [B,E,d] f32
 *       5 uz [B,2d] (f32 or bf16: [u; z_hat], the front's second layer)
 *       6 scalars [B,E,N,S] f32
 *       7 score_w [M,F], 8 score_b [M], 9 gate_w' [d,M A] (transposed; with has_gate)
 *      10 hv [B,N,d + M] f32 (the state's product h_op [V' | W_h']: V h of every cell, not yet
 *         rounded, and behind it the state's term of every logit), 11 edge [N,D],
 *      12 pad_src [N,D] i32, 13 norm_gain [n], 14 cost [M], 15 unused
 *      16 u, 17 z_hat, 18 eps [B,E,d] f32 (out); 19 logits [B,E,N,M] f32 (out)
 *      20 cells, 21 units [B,E,K] i64, 22 valid [B,E,K] u8 (out: the written slots; cell 0 and
 *         unit 0 on the others); 23 sel_cells, 24 sel_units [B,E,K] i64 (out: the router's
 *         decision, a padding row's included)
 *      25 theta [B,E], 26 soft, 27 prob [B,E,K] f32, 28 ties [B,E,K,M] u8 (out)
 *      29 cost_row [B,E] f32, 30 refreshed [B,E] f32 (out: the row's count)
 *      31 h_old, 32 m [B,E,K,d] f32, 33 rms [B,E,K] f32 (out)
 *      34 v [E,B,K,n] (f32 or bf16, out: the first product's operand of slot t)
 *      with flag 64: 35 rank [B,E,K] i32 (out: a written slot's rank among the row's written
 *         slots of its unit, in the order of the slots; -1 on a slot that was not written),
 *         36 count [B,M] i32 (in/out: the row's written slots per unit so far; restarted at
 *         slot 0). hsn_ev_starts turns them into the block's rows in unit order.
 *   i:  0 B, 1 E, 2 N, 3 M, 4 K, 5 d, 6 n, 7 D, 8 A, 9 S, 10 t,
 *       11 flags: 4 half, 8 refresh probes, 16 the address gate reads the summary (else u),
 *          32 has the address gate, 64 write p35 and p36,
 *       12 age_max, 13 measurements only: return after this phase (0: all).
 *   f:  0 tau, 1 norm_eps, 2 lambda_cost, 3 cell_overhead, 4 shared_cost, 5 limit.
 * Needs M a power of two, N M <= threads, 2 d <= threads, N <= threads, K D <= threads,
 * 16 M + 4 M A <= threads, 16 K <= threads.
 * Shared (words): M F + M A d + N M + 3 d + M + M A + 5 N + 2 M + 4 K + 1 + 3 ceil(N / 32) + 2 K
 *   + 2 K D + 2 K d + N S.
 */
extern "C" __global__ void hsn_ev_decide(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int B = a.i[0], E = a.i[1], N = a.i[2], M = a.i[3], K = a.i[4], d = a.i[5];
    const int n = a.i[6], D = a.i[7], A = a.i[8], S = a.i[9], t = a.i[10];
    const int flags = a.i[11], age_max = a.i[12], stop = a.i[13];
    const bool half = (flags & 4) != 0, probes = (flags & 8) != 0;
    const bool gate_summary = (flags & 16) != 0, has_gate = (flags & 32) != 0;
    const bool ranks = (flags & 64) != 0;
    const float tau = a.f[0], norm_eps = a.f[1], lambda = a.f[2];
    const float overhead = a.f[3], shared_cost = a.f[4], limit = a.f[5];
    const int row = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int F = 3 * d + 1 + S;
    const int d_shift = log2_or_minus1(d);
    const int vw = d + M;                        // a row of the state's product
    float* sw_s = shared;                        // [M F] the scorer's weight
    float* gw_s = sw_s + M * F;                  // [d, M A] the address gate's weight'
    float* score_s = gw_s + M * A * d;           // [N M]
    float* ue = score_s + N * M;                 // [2d] u, eps
    float* q_s = ue + 2 * d;                     // [d] the summary as the products read it
    float* rowsc = q_s + d;                      // [M]
    float* gate_s = rowsc + M;                   // [M A]
    float* score = gate_s + M * A;               // [N]
    int* unit = reinterpret_cast<int*>(score + N);        // [N]
    int* order = unit + N;                       // [N]
    float* stale = reinterpret_cast<float*>(order + N);   // [N]
    float* age_s = stale + N;                    // [N]
    float* cost_s = age_s + N;                   // [2 M] the units' costs, the scorer's bias
    int* out_cell = reinterpret_cast<int*>(cost_s + 2 * M);   // [K]
    int* out_unit = out_cell + K;                // [K]
    int* out_flag = out_unit + K;                // [K]
    int* slot = out_flag + K;                    // [K]
    int* n_valid_s = slot + K;                   // [1]
    const int warps = (N + 31) / 32;
    float* warp_low = reinterpret_cast<float*>(n_valid_s + 1);  // [warps]
    float* warp_high = warp_low + warps;         // [warps]
    int* warp_stale = reinterpret_cast<int*>(warp_high + warps);  // [warps]
    float* k_a = reinterpret_cast<float*>(warp_stale + warps);    // [K]
    float* k_b = k_a + K;                        // [K]
    int* src_s = reinterpret_cast<int*>(k_b + K);         // [K D]
    float* edge_s = reinterpret_cast<float*>(src_s + K * D);  // [K D]
    float* slot_a = edge_s + K * D;              // [K d]
    float* slot_b = slot_a + K * d;              // [K d]
    float* scal_s = slot_b + K * d;              // [N S] the slot's scalars

    const int64_t rec = static_cast<int64_t>(row) * E + t;
    const float* h = fptr(a, 0) + (static_cast<int64_t>(t) * B + row) * N * d;
    const int64_t* age = static_cast<const int64_t*>(a.p[2]) + (static_cast<int64_t>(row) * (E + 1) + t) * N;
    const bool alive = static_cast<const uint8_t*>(a.p[3])[rec] != 0;
    const float* hv = fptr(a, 10) + static_cast<int64_t>(row) * N * vw;
    // everything the scorer reads, once, into shared memory
    copy_strided(sw_s, fptr(a, 7), M * F, tid, nt);
    if (has_gate) copy_strided(gw_s, fptr(a, 9), M * A * d, tid, nt);
    {
        const float* sc = fptr(a, 6) + rec * N * S;
        for (int e = tid; e < N * S; e += nt) scal_s[e] = ev_round(sc[e], half);
    }
    for (int j = tid; j < M; j += nt) {
        cost_s[j] = fptr(a, 14)[j];
        cost_s[M + j] = fptr(a, 8)[j];  // the scorer's bias
    }
    for (int x = tid; x < d; x += nt) q_s[x] = ev_round(fptr(a, 1)[rec * d + x], half);
    float key = -1.0f;  // this cell's age if it is over age_max
    if (tid < N) {
        const int64_t ag = age[tid];
        if (probes && ag > static_cast<int64_t>(age_max)) key = static_cast<float>(ag);
        age_s[tid] = ev_round(static_cast<float>(ag) / static_cast<float>(age_max), half);
    }
    if (tid < 2 * d) {  // [u; z_hat] of the front's second layer; eps = z - z_hat
        const float value = load_value(a.p[5], static_cast<int64_t>(row) * 2 * d + tid, half);
        if (tid < d) {
            fout(a, 16)[rec * d + tid] = value;
            ue[tid] = value;
        } else {
            const int x = tid - d;
            fout(a, 17)[rec * d + x] = value;
            const float e = ev_round(fptr(a, 4)[rec * d + x] - value, half);
            fout(a, 18)[rec * d + x] = e;
            ue[tid] = e;
        }
    }
    __syncthreads();
    if (stop == 1) return;
    // the scorer's terms that are one per row, [u; eps] and the bias, and A12's gate rows: a
    // unit's sum is split over 16 lanes and folded by shuffles (a chain of 2 d steps on one
    // lane would be the kernel's longest)
    // (every lane takes part in the shuffles: the lanes without a piece add zero)
    {
        const int lanes = 16;
        const int first_gate = M * lanes;
        float acc_row = 0.0f, acc_gate = 0.0f;
        if (tid < first_gate) {
            const int m = tid / lanes, piece = tid - m * lanes;
            const int len = (2 * d + lanes - 1) / lanes;
            const float* wm = sw_s + m * F;
            for (int x = piece * len; x < (piece + 1) * len && x < 2 * d; ++x)
                acc_row += wm[x] * ue[x];
        } else if (has_gate && tid - first_gate < M * A * 4) {
            const int g = (tid - first_gate) >> 2, piece = (tid - first_gate) & 3;
            const int width = M * A, len = (d + 3) / 4;
            const float* read = gate_summary ? q_s : ue;
            for (int x = piece * len; x < (piece + 1) * len && x < d; ++x)
                acc_gate += gw_s[x * width + g] * read[x];
        }
        for (int w = 1; w < lanes; w <<= 1) acc_row += __shfl_xor_sync(0xffffffff, acc_row, w);
        acc_gate += __shfl_xor_sync(0xffffffff, acc_gate, 1);
        acc_gate += __shfl_xor_sync(0xffffffff, acc_gate, 2);
        if (tid < first_gate) {
            if ((tid & (lanes - 1)) == 0) rowsc[tid / lanes] = acc_row + cost_s[M + tid / lanes];
        } else if (has_gate && tid - first_gate < M * A * 4) {
            if (((tid - first_gate) & 3) == 0) gate_s[(tid - first_gate) >> 2] = ev_round(acc_gate, half);
        }
    }
    __syncthreads();
    if (stop == 2) return;
    // the logits, one thread per (cell, unit); the M lanes of a cell reduce by shuffles
    const int m_shift = __ffs(M) - 1;
    {
        const int j = tid & (M - 1);
        float s = -INFINITY;
        if (tid < N * M) {
            const int i = tid >> m_shift;
            const float* wm = sw_s + j * F;
            const float* sc = scal_s + i * S;
            // the state's term of the logit is the product's (lanes of a warp: consecutive words)
            float acc = rowsc[j] + hv[i * vw + d + j];
            acc += wm[3 * d] * age_s[i];
            for (int x = 0; x < S; ++x) acc += wm[3 * d + 1 + x] * sc[x];
            float logit = ev_round(acc, half);
            if (has_gate) {
                float g = 0.0f;
                for (int x = 0; x < A; ++x) g += sc[S - A + x] * gate_s[j * A + x];
                logit = ev_round(logit + ev_round(g, half), half);
            }
            fout(a, 19)[(rec * N + i) * M + j] = logit;
            s = lambda > 0.0f ? logit - lambda * cost_s[j] : logit;
            score_s[tid] = s;
        }
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
    }
    __syncthreads();
    if (stop == 3) return;
    // the decision (hsn_win_select's steps: Router.select with topk and the refresh probes)
    float mine = 0.0f;
    int rank = N;
    if (tid < N) {
        mine = score[tid];
        rank = 0;
        for (int j = 0; j < N; ++j) rank += (score[j] > mine) || (score[j] == mine && j < tid);
        order[rank] = tid;
    }
    __syncthreads();
    if (tid == 0) {
        float cumulative = 0.0f;
        int n_valid = 0;
        for (int k = 0; k < K; ++k) {
            const int c = order[k];
            const int u = unit[c];
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
    {
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
            fout(a, 25)[rec] = 0.5f * (lowest_in + highest_out);
            k_b[0] = 0.5f * (lowest_in + highest_out);
            const int n_force = n_stale < n_valid ? n_stale : n_valid;
            for (int j = 0; j < n_force; ++j) {  // the j-th stalest cell goes to slot n_valid - 1 - j
                int pick = 0;
                float stalest = stale[0];
                for (int c = 1; c < N; ++c) {
                    if (stale[c] > stalest) {
                        stalest = stale[c];
                        pick = c;
                    }
                }
                stale[pick] = -2.0f;
                const int k = n_valid - 1 - j;
                out_cell[k] = pick;
                out_unit[k] = unit[pick];
                out_flag[k] |= 2;
            }
            if (n_force > 0) {
                // a row the probes pushed over the limit keeps its scored selection
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
            float spent = 0.0f;
            int n_fresh = 0;
            for (int k = 0; k < K; ++k) {
                spent += (out_flag[k] & 1) ? overhead + cost_s[out_unit[k]] : 0.0f;
                n_fresh += (out_flag[k] >> 1) & 1;
            }
            fout(a, 29)[rec] = shared_cost + spent;
            fout(a, 30)[rec] = static_cast<float>(n_fresh);
        }
    }
    __syncthreads();
    if (stop == 4) return;
    const float theta = k_b[0];
    // the records, the straight-through values of the written slots (cell 0 and unit 0 stand
    // in on the others, as in the reference), and each slot's sources
    for (int k = tid; k < K; k += nt) {
        const bool raw = (out_flag[k] & 1) != 0;
        const bool ok = raw && alive;
        static_cast<int64_t*>(a.p[23])[rec * K + k] = out_cell[k];
        static_cast<int64_t*>(a.p[24])[rec * K + k] = out_unit[k];
        const int c = ok ? out_cell[k] : 0, u = ok ? out_unit[k] : 0;
        static_cast<int64_t*>(a.p[20])[rec * K + k] = c;
        static_cast<int64_t*>(a.p[21])[rec * K + k] = u;
        static_cast<uint8_t*>(a.p[22])[rec * K + k] = ok ? 1 : 0;
        out_cell[k] = c;
        out_unit[k] = u;
        out_flag[k] = ok ? 1 : 0;
        const float* sc = score_s + c * M;
        float top = sc[0];
        for (int j = 1; j < M; ++j) top = sc[j] > top ? sc[j] : top;
        float denom = 0.0f;
        uint8_t* ties = static_cast<uint8_t*>(a.p[28]) + (rec * K + k) * M;
        for (int j = 0; j < M; ++j) {
            denom += expf((sc[j] - top) / tau);
            ties[j] = (ok && sc[j] == top) ? 1 : 0;
        }
        fout(a, 26)[rec * K + k] = 1.0f / (1.0f + expf(-(top - theta) / tau));
        fout(a, 27)[rec * K + k] = expf((sc[u] - top) / tau) / denom;
    }
    __syncthreads();
    {
        const float* edge = fptr(a, 11);
        const int32_t* pad_src = static_cast<const int32_t*>(a.p[12]);
        for (int e = tid; e < K * D; e += nt) {
            const int k = e / D, ed = e - k * D;
            const int c = out_cell[k];
            const int src = pad_src[c * D + ed];
            const float w = edge[c * D + ed];
            src_s[e] = src;
            edge_s[e] = ev_round(w, half);
        }
    }
    if (ranks) {  // the counts are read here and written behind the barrier
        const int32_t* count = static_cast<const int32_t*>(a.p[36]) + static_cast<int64_t>(row) * M;
        for (int k = tid; k < K; k += nt) {
            int r = -1;
            if (out_flag[k]) {
                const int u = out_unit[k];
                r = t > 0 ? count[u] : 0;
                for (int j = 0; j < k; ++j) r += (out_flag[j] && out_unit[j] == u) ? 1 : 0;
            }
            static_cast<int32_t*>(a.p[35])[rec * K + k] = r;
        }
    }
    __syncthreads();
    if (ranks) {
        int32_t* count = static_cast<int32_t*>(a.p[36]) + static_cast<int64_t>(row) * M;
        for (int j = tid; j < M; j += nt) {
            int c = t > 0 ? count[j] : 0;
            for (int k = 0; k < K; ++k) c += (out_flag[k] && out_unit[k] == j) ? 1 : 0;
            count[j] = c;
        }
    }
    if (stop == 5) return;
    // the slots' own states as the cell reads them (not rounded), and the messages:
    // m = sum over the cell's sources of a_e (V h[src_e]), each product rounded
    for (int e = tid; e < K * d; e += nt) {
        const int k = quot(e, d, d_shift), x = e - k * d;
        const float ho = h[out_cell[k] * d + x];
        float vals[8];
        const int count = D < 8 ? D : 8;
        for (int ed = 0; ed < count; ++ed) vals[ed] = hv[src_s[k * D + ed] * vw + x];
        float acc = 0.0f;
        for (int ed = 0; ed < count; ++ed)
            acc += ev_round(ev_round(vals[ed], half) * edge_s[k * D + ed], half);
        const float mv = ev_round(acc, half);
        slot_a[e] = ho;
        slot_b[e] = mv;
        fout(a, 31)[rec * K * d + e] = ho;
        fout(a, 32)[rec * K * d + e] = mv;
    }
    __syncthreads();
    if (stop == 6) return;
    // the cell norm over [u; eps; h_old; m]: a slot's sum of squares is split over 16 lanes
    // and folded by shuffles (every lane takes part; the lanes without a piece add zero)
    {
        const int lanes = 16;
        float sum = 0.0f;
        if (tid < K * lanes) {
            const int k = tid / lanes, piece = tid - k * lanes;
            const int len_s = (d + lanes - 1) / lanes, len_u = (2 * d + lanes - 1) / lanes;
            for (int x = piece * len_u; x < (piece + 1) * len_u && x < 2 * d; ++x)
                sum += ue[x] * ue[x];
            for (int x = piece * len_s; x < (piece + 1) * len_s && x < d; ++x) {
                sum += slot_a[k * d + x] * slot_a[k * d + x];
                sum += slot_b[k * d + x] * slot_b[k * d + x];
            }
        }
        for (int w = 1; w < lanes; w <<= 1) sum += __shfl_xor_sync(0xffffffff, sum, w);
        if (tid < K * lanes && (tid & (lanes - 1)) == 0) {
            const float r = rsqrtf(sum / static_cast<float>(n) + norm_eps);
            k_a[tid / lanes] = r;
            fout(a, 33)[rec * K + tid / lanes] = r;
        }
    }
    __syncthreads();
    {
        const float* gain = fptr(a, 13);
        const int64_t base = (static_cast<int64_t>(t) * B + row) * K * n;
        const int n_shift = log2_or_minus1(n);
        for (int e = tid; e < K * n; e += nt) {
            const int k = quot(e, n, n_shift), x = e - k * n;
            const float g = gain[x];
            const float xv = x < 2 * d ? ue[x] : (x < 3 * d ? slot_a[k * d + x - 2 * d] : slot_b[k * d + x - 3 * d]);
            store_value(a.p[34], base + e, half, xv * k_a[k] * g);
        }
    }
}

/* hsn_ev_pick: between the two products of slot t. The chosen unit's hidden row of the first
 * product, its activation (the second product's operand) and the gate.
 *   p: 0 first [B K, M H + d] (f32 or bf16), 1 b1 [M H + d] f32 (read unless the product added
 *        it), 2 units [B,E,K] i64, 3 hidden [B,E,K,H] (f32 or bf16, out), 4 act [E,B,K,H] (f32
 *        or bf16, out: the second product's operand of slot t), 5 gate_pre [B,E,K,d] f32 (out),
 *        6 b [B,E,K,d] f32 (out).
 *   i: 0 B, 1 E, 2 K, 3 d, 4 H, 5 M, 6 t, 7 flags (1 half, 2 the product added b1), 8 activation.
 * Shared: K words (the slots' units, read once).
 * With bf16 buffers, the bias in the product and H a multiple of 128 dividing 4 threads, a
 * thread reads and stores four values at a time (hsn_win_back_mid's layout: the same values).
 */
extern "C" __global__ void hsn_ev_pick(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int B = a.i[0], E = a.i[1], K = a.i[2], d = a.i[3], H = a.i[4], M = a.i[5], t = a.i[6];
    const bool half = (a.i[7] & 1) != 0, bias_done = (a.i[7] & 2) != 0;
    const int code = a.i[8];
    const int row = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int64_t rec = static_cast<int64_t>(row) * E + t;
    const int64_t slot_major = static_cast<int64_t>(t) * B + row;
    const int64_t* units = static_cast<const int64_t*>(a.p[2]) + rec * K;
    const float* b1 = fptr(a, 1);
    const int width = M * H + d;
    int* unit_s = reinterpret_cast<int*>(shared);
    for (int k = tid; k < K; k += nt) unit_s[k] = static_cast<int>(units[k]);
    __syncthreads();
    const int chunks = H >> 2;
    const uintptr_t bases = reinterpret_cast<uintptr_t>(a.p[0]) | reinterpret_cast<uintptr_t>(a.p[3])
        | reinterpret_cast<uintptr_t>(a.p[4]);
    if (half && bias_done && (H & 127) == 0 && (width & 3) == 0 && nt % chunks == 0
        && (bases & 15) == 0) {
        const int G = nt / chunks, g = tid / chunks, h0 = (tid - g * chunks) << 2;
        const uint16_t* first = static_cast<const uint16_t*>(a.p[0]);
        uint16_t* hidden = static_cast<uint16_t*>(a.p[3]) + rec * K * H;
        uint16_t* act = static_cast<uint16_t*>(a.p[4]) + slot_major * K * H;
        for (int k = g; k < K; k += G) {
            const int64_t from = (static_cast<int64_t>(row) * K + k) * width + unit_s[k] * H + h0;
            const uint64_t hid4 = *reinterpret_cast<const uint64_t*>(first + from);
            uint64_t out = 0;
            for (int i = 0; i < 4; ++i) {
                const float hv = bf16_to_float(static_cast<uint16_t>(hid4 >> (16 * i)));
                out |= static_cast<uint64_t>(float_to_bf16(ev_activation(hv, code))) << (16 * i);
            }
            *reinterpret_cast<uint64_t*>(hidden + static_cast<int64_t>(k) * H + h0) = hid4;
            *reinterpret_cast<uint64_t*>(act + static_cast<int64_t>(k) * H + h0) = out;
        }
    } else {
        const int h_shift = log2_or_minus1(H);
        for (int e = tid; e < K * H; e += nt) {
            const int k = quot(e, H, h_shift), j = e - k * H;
            const int u = unit_s[k];
            const float raw = load_value(a.p[0], (static_cast<int64_t>(row) * K + k) * width + u * H + j, half);
            const float bias = bias_done ? 0.0f : b1[u * H + j];
            const float v = bias_done ? raw : ev_round(raw + bias, half);
            store_value(a.p[3], rec * K * H + e, half, v);
            store_value(a.p[4], slot_major * K * H + e, half, ev_activation(v, code));
        }
    }
    const int d_shift = log2_or_minus1(d);
    for (int e = tid; e < K * d; e += nt) {
        const int k = quot(e, d, d_shift), x = e - k * d;
        const int64_t p = static_cast<int64_t>(row) * K + k;
        const float raw = load_value(a.p[0], p * width + M * H + x, half);
        const float gv = bias_done ? raw : ev_round(raw + b1[M * H + x], half);
        fout(a, 5)[rec * K * d + e] = gv;
        fout(a, 6)[rec * K * d + e] = sigmoidf(gv);
    }
}

/* hsn_ev_pairs: the cell norm's backward of every slot of a block, for the parameters' part of
 * the backward pass (hsn.train.event_fused.local_backward). One block per (row, slot), one
 * thread per column of the cell input. With v = gain (.) x r (x = [u; eps; h_old; m], r the
 * recorded 1 / rms) and dv the cotangent of v:
 *   t = x r dv;   g_norm += t;   s = sum_c gain_c t_c;   dx = r gain dv - (r / n) x r s
 *   d_u = sum_k dx[:d],  d_eps = sum_k dx[d:2d],  dm_k = dx[3d:]   (dx's h_old columns are the
 *   reverse sweep's, and are not formed)
 *   p: 0 dv [P, n] (f32 or bf16; a slot's row is row_of, -1: a slot without a row, whose
 *        dv is zero), 1 row_of [B E, K] i32,
 *      2 u, 3 eps [B E, d] f32, 4 h_old, 5 m [B E, K, d] f32, 6 rms [B E, K] f32, 7 gain [n] f32,
 *      8 d_u, 9 d_eps [B E, d] f32 (out), 10 dm [B E, K, d] f32 (out),
 *      11 g_norm [B E, n] f32 (out: the block's rows, summed by the caller).
 *   i: 0 K (<= 16), 1 d, 2 n (<= threads), 3 dv is bf16.
 * Shared: K (1 + threads / 32) words. A padded slot's dv is zero, and so is all it adds.
 */
extern "C" __global__ void hsn_ev_pairs(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int K = a.i[0], d = a.i[1], n = a.i[2];
    const bool half = a.i[3] != 0;
    const int be = blockIdx.x, c = threadIdx.x, nt = blockDim.x;
    const int warps = nt / 32;
    float* s_k = shared;          // [K]
    float* part = s_k + K;        // [K, warps]
    const int32_t* row_of = static_cast<const int32_t*>(a.p[1]) + static_cast<int64_t>(be) * K;
    const float* rms = fptr(a, 6) + static_cast<int64_t>(be) * K;
    const bool live = c < n;
    const float gain = live ? fptr(a, 7)[c] : 0.0f;
    float dv[16], xr[16];
    // every read of the thread goes out before any is used
    float shared_x = 0.0f;
    if (live && c < d) shared_x = fptr(a, 2)[static_cast<int64_t>(be) * d + c];
    else if (live && c < 2 * d) shared_x = fptr(a, 3)[static_cast<int64_t>(be) * d + c - d];
    for (int k = 0; k < K; ++k) {
        float x = shared_x;
        float value = 0.0f;
        if (live) {
            if (row_of[k] >= 0)
                value = load_value(a.p[0], static_cast<int64_t>(row_of[k]) * n + c, half);
            if (c >= 3 * d) x = fptr(a, 5)[(static_cast<int64_t>(be) * K + k) * d + c - 3 * d];
            else if (c >= 2 * d) x = fptr(a, 4)[(static_cast<int64_t>(be) * K + k) * d + c - 2 * d];
        }
        dv[k] = value;
        xr[k] = x;
    }
    float gn = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float r = rms[k];
        xr[k] *= r;
        const float t = xr[k] * dv[k];
        gn += t;
        float s = gain * t;
        for (int w = 16; w > 0; w >>= 1) s += __shfl_xor_sync(0xffffffff, s, w);
        if ((c & 31) == 0) part[k * warps + (c >> 5)] = s;
    }
    if (live) fout(a, 11)[static_cast<int64_t>(be) * n + c] = gn;
    __syncthreads();
    if (c < K) {
        float s = 0.0f;
        for (int w = 0; w < warps; ++w) s += part[c * warps + w];
        s_k[c] = s;
    }
    __syncthreads();
    if (!live) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float r = rms[k];
        const float dx = r * gain * dv[k] - (r / static_cast<float>(n)) * xr[k] * s_k[k];
        sum += dx;
        if (c >= 3 * d) fout(a, 10)[(static_cast<int64_t>(be) * K + k) * d + c - 3 * d] = dx;
    }
    if (c < d) fout(a, 8)[static_cast<int64_t>(be) * d + c] = sum;
    else if (c < 2 * d) fout(a, 9)[static_cast<int64_t>(be) * d + c - d] = sum;
}

/* ------------------------------------------------------------------------------------------------
 * The parameters' part of the backward pass on the card (hsn.kernels.event.EventKernels.local;
 * hsn.train.event_fused.local_backward is the reference). The unit bank's weight gradients are
 * matrix products over a unit's rows of the whole block, so the block's written slots are laid
 * out in unit order: a slot's row is pos = start[row, unit] + rank (unit, then batch row, then
 * slot; hsn_ev_decide counts, hsn_ev_starts and hsn_ev_sorted place). The reverse sweep's head
 * and middle write dy, dg and dhid at those rows (hsn_window.cu); the kernels here are the rest:
 * column sums (the biases), the messages, the scorer.
 * ---------------------------------------------------------------------------------------------- */

/* hsn_ev_starts: after a block's forward sweep. From the rows' counts of written slots per unit,
 * where each row's slots of a unit start in the block's rows in unit order, and the units'
 * bounds.
 *   p: 0 count [B,M] i32, 1 start [B,M] i32 (out), 2 bounds [M + 1] i32 (out: unit j's rows are
 *      bounds[j] .. bounds[j + 1]), 3 all [2] i32 (out: 0 and the number of rows).
 *   i: 0 B, 1 M.
 * One block of at least M threads. Shared: M words.
 */
extern "C" __global__ void hsn_ev_starts(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int B = a.i[0], M = a.i[1];
    const int tid = threadIdx.x;
    int* total = reinterpret_cast<int*>(shared);
    const int32_t* count = static_cast<const int32_t*>(a.p[0]);
    int32_t* start = static_cast<int32_t*>(a.p[1]);
    int32_t* bounds = static_cast<int32_t*>(a.p[2]);
    if (tid < M) {
        int run = 0;
        for (int b = 0; b < B; ++b) {
            start[b * M + tid] = run;
            run += count[b * M + tid];
        }
        total[tid] = run;
    }
    __syncthreads();
    if (tid < M) {
        int base = 0;
        for (int j = 0; j < tid; ++j) base += total[j];
        for (int b = 0; b < B; ++b) start[b * M + tid] += base;
        bounds[tid] = base;
        if (tid == M - 1) {
            bounds[M] = base + total[tid];
            static_cast<int32_t*>(a.p[3])[0] = 0;
            static_cast<int32_t*>(a.p[3])[1] = base + total[tid];
        }
    }
}

/* hsn_ev_sorted: after hsn_ev_starts. Every slot's row in unit order, and the operands of the
 * block's two unit products at those rows.
 *   p: 0 rank [B,E,K] i32, 1 units [B,E,K] i64, 2 start [B,M] i32, 3 pos [B,E,K] i32 (out: the
 *      slot's row, -1 on a slot that was not written),
 *      4 v [E,B,K,n], 5 act [E,B,K,H] (f32 or bf16: the products' operands, slot-major),
 *      6 v_s [P,n], 7 act_s [P,H] (out: the same rows in unit order).
 *   i: 0 B, 1 E, 2 K, 3 M, 4 n, 5 H, 6 half.
 * One block per (row, slot). Shared: K words. With bf16 buffers on sixteen-byte bases and n, H
 * multiples of 4, a thread copies four values at a time.
 */
extern "C" __global__ void hsn_ev_sorted(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int B = a.i[0], E = a.i[1], K = a.i[2], M = a.i[3], n = a.i[4], H = a.i[5];
    const bool half = a.i[6] != 0;
    const int be = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int row = be / E, t = be - row * E;
    int* pos_s = reinterpret_cast<int*>(shared);
    const int32_t* rank = static_cast<const int32_t*>(a.p[0]) + static_cast<int64_t>(be) * K;
    const int64_t* units = static_cast<const int64_t*>(a.p[1]) + static_cast<int64_t>(be) * K;
    const int32_t* start = static_cast<const int32_t*>(a.p[2]) + static_cast<int64_t>(row) * M;
    int32_t* pos = static_cast<int32_t*>(a.p[3]) + static_cast<int64_t>(be) * K;
    for (int k = tid; k < K; k += nt) {
        const int r = rank[k];
        const int p = r >= 0 ? start[static_cast<int>(units[k])] + r : -1;
        pos_s[k] = p;
        pos[k] = p;
    }
    __syncthreads();
    const int64_t src = (static_cast<int64_t>(t) * B + row) * K;  // the slot-major record
    const uintptr_t bases = reinterpret_cast<uintptr_t>(a.p[4]) | reinterpret_cast<uintptr_t>(a.p[5])
        | reinterpret_cast<uintptr_t>(a.p[6]) | reinterpret_cast<uintptr_t>(a.p[7]);
    if (half && (n & 3) == 0 && (H & 3) == 0 && (bases & 15) == 0) {
        const int nw = n >> 2, hw = H >> 2;
        const uint64_t* v = reinterpret_cast<const uint64_t*>(static_cast<const uint16_t*>(a.p[4]) + src * n);
        const uint64_t* act = reinterpret_cast<const uint64_t*>(static_cast<const uint16_t*>(a.p[5]) + src * H);
        uint64_t* v_s = reinterpret_cast<uint64_t*>(a.p[6]);
        uint64_t* act_s = reinterpret_cast<uint64_t*>(a.p[7]);
        for (int e = tid; e < K * nw; e += nt) {
            const int k = e / nw, w = e - k * nw;
            if (pos_s[k] >= 0) v_s[static_cast<int64_t>(pos_s[k]) * nw + w] = v[e];
        }
        for (int e = tid; e < K * hw; e += nt) {
            const int k = e / hw, w = e - k * hw;
            if (pos_s[k] >= 0) act_s[static_cast<int64_t>(pos_s[k]) * hw + w] = act[e];
        }
        return;
    }
    for (int e = tid; e < K * n; e += nt) {
        const int k = e / n, x = e - k * n;
        if (pos_s[k] >= 0)
            store_value(a.p[6], static_cast<int64_t>(pos_s[k]) * n + x, half,
                load_value(a.p[4], src * n + e, half));
    }
    for (int e = tid; e < K * H; e += nt) {
        const int k = e / H, x = e - k * H;
        if (pos_s[k] >= 0)
            store_value(a.p[7], static_cast<int64_t>(pos_s[k]) * H + x, half,
                load_value(a.p[5], src * H + e, half));
    }
}

/* hsn_ev_colsum: the sums of a matrix's columns over groups of consecutive rows, in pieces (the
 * caller adds a group's pieces): the bias gradients of the unit bank, a group per unit.
 *   p: 0 src [P,W] (f32 or bf16), 1 bounds [G + 1] i32 (group g: rows bounds[g] .. bounds[g + 1]),
 *      2 out [G,C,W] f32.
 *   i: 0 W, 1 C, 2 half.
 * G C blocks of at least W threads: a thread per column, the rows of a piece in order (four
 * partial sums).
 */
extern "C" __global__ void hsn_ev_colsum(HsnWinArgs a) {
    const int W = a.i[0], C = a.i[1];
    const bool half = a.i[2] != 0;
    const int g = blockIdx.x / C, c = blockIdx.x - g * C, w = threadIdx.x;
    if (w >= W) return;
    const int32_t* bounds = static_cast<const int32_t*>(a.p[1]);
    const int lo = bounds[g], hi = bounds[g + 1];
    const int len = (hi - lo + C - 1) / C;
    const int from = lo + c * len;
    const int to = from + len < hi ? from + len : hi;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int r = from;
    for (; r + 3 < to; r += 4) {
        const float v0 = load_value(a.p[0], static_cast<int64_t>(r) * W + w, half);
        const float v1 = load_value(a.p[0], static_cast<int64_t>(r + 1) * W + w, half);
        const float v2 = load_value(a.p[0], static_cast<int64_t>(r + 2) * W + w, half);
        const float v3 = load_value(a.p[0], static_cast<int64_t>(r + 3) * W + w, half);
        s0 += v0;
        s1 += v1;
        s2 += v2;
        s3 += v3;
    }
    for (; r < to; ++r) s0 += load_value(a.p[0], static_cast<int64_t>(r) * W + w, half);
    fout(a, 2)[(static_cast<int64_t>(g) * C + c) * W + w] = (s0 + s1) + (s2 + s3);
}

/* hsn_ev_edges: the messages' part. With m_k = V (sum_e a_e h[src_e]) of slot k's cell and dm
 * its cotangent:
 *   weighted_k = sum_e a_e h[src_e]      (the caller's product dm' weighted is V's gradient)
 *   per_edge_k,e = h[src_e] . through_k  (through = dm V, a product of the caller's: the edge
 *                                         weight's gradient, to be added at cell_k D + e)
 *   p: 0 h [E+1,B,N,d] f32 (record t: the state the slot read), 1 cells [B,E,K] i64,
 *      2 valid [B,E,K] u8, 3 edge [N,D] f32, 4 pad_src [N,D] i32, 5 through [B,E,K,d] f32,
 *      6 weighted [B,E,K,d] f32 (out), 7 per_edge [B,E,K,D] f32 (out), 8 index [B,E,K,D] i64
 *      (out). A slot that was not written: zeros, index 0.
 *   i: 0 B, 1 E, 2 N, 3 K, 4 d, 5 D.
 * One block per (row, slot); the threads in groups of 32 ceil(d / 32) lanes, a lane per column
 * of the state, group g taking the slots g, g + G, ...; a dot product is folded by its warps.
 * Shared: K + 2 K D + K D ceil(d / 32) words.
 */
extern "C" __global__ void hsn_ev_edges(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int B = a.i[0], E = a.i[1], N = a.i[2], K = a.i[3], d = a.i[4], D = a.i[5];
    const int be = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int row = be / E, t = be - row * E;
    const int gw = ((d + 31) / 32) * 32, warps = gw / 32;
    const int G = nt / gw, g = tid / gw, x = tid - g * gw;
    int* cell_s = reinterpret_cast<int*>(shared);              // [K], -1 if not written
    int* src_s = cell_s + K;                                   // [K, D]
    float* w_s = reinterpret_cast<float*>(src_s + K * D);      // [K, D]
    float* part = w_s + K * D;                                 // [K, D, warps]
    const int64_t slot = static_cast<int64_t>(be) * K;
    const int64_t* cells = static_cast<const int64_t*>(a.p[1]) + slot;
    const uint8_t* valid = static_cast<const uint8_t*>(a.p[2]) + slot;
    for (int k = tid; k < K; k += nt) cell_s[k] = valid[k] ? static_cast<int>(cells[k]) : -1;
    __syncthreads();
    const float* edge = fptr(a, 3);
    const int32_t* pad_src = static_cast<const int32_t*>(a.p[4]);
    for (int e = tid; e < K * D; e += nt) {
        const int k = e / D, ed = e - k * D;
        const int c = cell_s[k];
        src_s[e] = c >= 0 ? pad_src[c * D + ed] : 0;
        w_s[e] = c >= 0 ? edge[c * D + ed] : 0.0f;
        static_cast<int64_t*>(a.p[8])[slot * D + e] = c >= 0 ? static_cast<int64_t>(c) * D + ed : 0;
    }
    __syncthreads();
    const float* h = fptr(a, 0) + (static_cast<int64_t>(t) * B + row) * N * d;
    const float* through = fptr(a, 5) + slot * d;
    float* weighted = fout(a, 6) + slot * d;
    if (g < G) {  // (whole warps share g: every lane of a warp takes the same slots)
        for (int k = g; k < K; k += G) {
            const bool on = x < d && cell_s[k] >= 0;
            const float th = on ? through[k * d + x] : 0.0f;
            float acc = 0.0f;
            for (int ed = 0; ed < D; ++ed) {
                const float hv = on ? h[src_s[k * D + ed] * d + x] : 0.0f;
                acc += w_s[k * D + ed] * hv;
                float dot = hv * th;
                for (int w = 16; w > 0; w >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, w);
                if ((tid & 31) == 0) part[(k * D + ed) * warps + (x >> 5)] = dot;
            }
            if (x < d) weighted[k * d + x] = acc;
        }
    }
    __syncthreads();
    for (int e = tid; e < K * D; e += nt) {
        float sum = 0.0f;
        for (int w = 0; w < warps; ++w) sum += part[e * warps + w];
        fout(a, 7)[slot * D + e] = sum;
    }
}

/* hsn_ev_scores: the scorer's part. The straight-through path into the scores of a slot's
 * written cells (hsn.train.event_fused.local_backward: the cell mask's cotangent through the
 * sigmoid to the cell's largest score, shared by equal scores; the unit weight's through the
 * softmax to the executed unit's score), joined with the cotangent of the logits, and the sums
 * over the slot's cells that the scorer's gradient is made of:
 *   d_cell = d_mask (1 - soft) soft / tau;   d_unit = prob (g - g prob) / tau   (g = dL/dalpha)
 *   d_log[cell_k, j] = d_logits[cell_k, j] + (d_cell / ties) tie_j + [j = unit_k] d_unit
 *   per_row[m] = sum_n d_log[n, m];   feat[m, s] = sum_n round(d_log[n, m]) round(f[n, s])
 *   (f = [age / age_max ; scalars]: the weight's columns behind the state's)
 *   gate_rows[m, a] = sum_n d_log[n, m] scalars[n, S - A + a]   (the address gate's rows)
 *   p: 0 alpha [B,E,K,2] f32 (g, d_mask), 1 soft, 2 prob [B,E,K] f32, 3 ties [B,E,K,M] u8,
 *      4 cells, 5 units [B,E,K] i64, 6 valid [B,E,K] u8,
 *      7 d_logits f32 (with flag 1; batch row b starts i[9] words behind row b - 1, its slots
 *        [E', N, M] with E' >= E: a view of the first E slots of a longer block),
 *      8 scalars [B,E,N,S] f32, 9 age [B,E+1,N] i64 (record t),
 *      10 d_log [E,B,N,M] f32 (out; slot-major, beside the states' [E,B,N,d]),
 *      11 per_row [B,E,M] f32 (out), 12 feat [B,E,M,1+S] f32 (out),
 *      13 gate_rows [B,E,M,A] f32 (out, with A > 0).
 *   i: 0 B, 1 E, 2 N, 3 M, 4 K, 5 S, 6 A, 7 flags (1 d_logits, 4 half), 8 age_max,
 *      9 d_logits' words per batch row.
 *   f: 0 tau.
 * One block per (row, slot). Shared: N M + N (1 + S) words.
 */
extern "C" __global__ void hsn_ev_scores(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int B = a.i[0], E = a.i[1], N = a.i[2], M = a.i[3], K = a.i[4], S = a.i[5], A = a.i[6];
    const bool given = (a.i[7] & 1) != 0, half = (a.i[7] & 4) != 0;
    const int age_max = a.i[8];
    const float tau = a.f[0];
    const int be = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int row = be / E, t = be - row * E;
    const int F1 = 1 + S;
    float* dl = shared;          // [N, M]
    float* ft = dl + N * M;      // [N, 1 + S]
    const float* given_row = fptr(a, 7) + static_cast<int64_t>(row) * a.i[9]
        + static_cast<int64_t>(t) * N * M;
    for (int e = tid; e < N * M; e += nt) dl[e] = given ? given_row[e] : 0.0f;
    const float* scal = fptr(a, 8) + static_cast<int64_t>(be) * N * S;
    const int64_t* age = static_cast<const int64_t*>(a.p[9])
        + (static_cast<int64_t>(row) * (E + 1) + t) * N;
    for (int e = tid; e < N * F1; e += nt) {
        const int n = e / F1, s = e - n * F1;
        ft[e] = s == 0 ? static_cast<float>(age[n]) / static_cast<float>(age_max)
            : scal[n * S + s - 1];
    }
    __syncthreads();
    const int64_t slot = static_cast<int64_t>(be) * K;
    for (int k = tid; k < K; k += nt) {  // written slots have distinct cells: one writer each
        if (!static_cast<const uint8_t*>(a.p[6])[slot + k]) continue;
        const int c = static_cast<int>(static_cast<const int64_t*>(a.p[4])[slot + k]);
        const int u = static_cast<int>(static_cast<const int64_t*>(a.p[5])[slot + k]);
        const float g = fptr(a, 0)[(slot + k) * 2], d_mask = fptr(a, 0)[(slot + k) * 2 + 1];
        const float so = fptr(a, 1)[slot + k], pr = fptr(a, 2)[slot + k];
        const float d_cell = (d_mask * (1.0f - so) * so) / tau;
        const float d_unit = (pr * (g - g * pr)) / tau;
        const uint8_t* tie = static_cast<const uint8_t*>(a.p[3]) + (slot + k) * M;
        int n_ties = 0;
        for (int j = 0; j < M; ++j) n_ties += tie[j] ? 1 : 0;
        const float share = d_cell / static_cast<float>(n_ties > 0 ? n_ties : 1);
        for (int j = 0; j < M; ++j)
            dl[c * M + j] += (tie[j] ? share : 0.0f) + (j == u ? d_unit : 0.0f);
    }
    __syncthreads();
    float* out = fout(a, 10) + (static_cast<int64_t>(t) * B + row) * N * M;
    for (int e = tid; e < N * M; e += nt) out[e] = dl[e];
    for (int m = tid; m < M; m += nt) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) sum += dl[n * M + m];
        fout(a, 11)[static_cast<int64_t>(be) * M + m] = sum;
    }
    for (int e = tid; e < M * F1; e += nt) {
        const int m = e / F1, s = e - m * F1;
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) sum += ev_round(dl[n * M + m], half) * ev_round(ft[n * F1 + s], half);
        fout(a, 12)[(static_cast<int64_t>(be) * M + m) * F1 + s] = sum;
    }
    for (int e = tid; e < M * A; e += nt) {
        const int m = e / A, x = e - m * A;
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) sum += dl[n * M + m] * ft[n * F1 + F1 - A + x];
        fout(a, 13)[(static_cast<int64_t>(be) * M + m) * A + x] = sum;
    }
}
