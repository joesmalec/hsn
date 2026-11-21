/* The fused word-event round, forward (docs/wev_fused_event.md; hsn.train.event_fused is the
 * reference, hsn.kernels.event the launcher). Included by hsn_window.cu: device code only, its
 * helpers (bf16 casts, load_value / store_value, gelu_erf, sigmoidf) are in scope.
 *
 * An event slot's forward pass is four launches:
 *   hsn_ev_head   finish slot t - 1 (candidate, gated update, summary, ages, read-out), then
 *                 start slot t (reset, front, scorer, top-K decision, messages, cell input)
 *   product       first = v W1 (+ b1): every unit's first layer and the gate, one matrix product
 *   hsn_ev_pick   the chosen unit's hidden row, its activation, the gate
 *   product       out = act W2 (+ b2): the second layers with W_out folded in
 * One block per batch row in both kernels: everything a row computes between two products
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

/* hsn_ev_head: the row's work between the second product of slot t - 1 and the first of slot t.
 *   p:  0 h [B,E+1,N,d] f32 (record t: the state slot t reads; reset in place; record t + 1 is
 *         written by the finish of slot t)
 *       1 q [B,E+1,d] f32 (record t: the summary before slot t's reset)
 *       2 q_pre [B,E,d] f32 (out: the summary slot t reads, after its reset)
 *       3 age [B,E+1,N] i64 (as h)
 *       4 reset [B,E] u8, 5 live [B,E] u8
 *       6 z [B,E,d] f32, 7 front_static [B,E,Hf] (f32 or bf16: [z; ctx] W1' + b1, once per block)
 *       8 scalars [B,E,N,S] f32
 *       9 front_q [Hf,d], 10 front_w2 [2d,Hf], 11 front_b2 [2d]
 *      12 score_w [M,F], 13 score_b [M], 14 gate_w [M A,d] (with has_gate)
 *      15 V [d,d], 16 edge [N,D], 17 pad_src [N,D] i32, 18 norm_gain [n], 19 cand_gain [d]
 *      20 cost [M], 21 b2 [M d] (read unless the second product added it)
 *      22 u, 23 z_hat, 24 eps [B,E,d] (out); 25 logits [B,E,N,M] (out)
 *      26 cells, 27 units [B,E,K] i64, 28 valid [B,E,K] u8 (out: the written slots; cell 0 and
 *         unit 0 on the others); 29 sel_cells, 30 sel_units [B,E,K] i64 (out: the router's
 *         decision, a padding row's included)
 *      31 theta [B,E], 32 soft, 33 prob [B,E,K] f32, 34 ties [B,E,K,M] u8 (out)
 *      35 cost_row [B,E] f32, 36 refreshed [B,E] f32 (out: the row's count)
 *      37 h_old, 38 m [B,E,K,d] f32, 39 rms [B,E,K] f32 (out)
 *      40 v [E,B,K,n] (f32 or bf16, out: the first product's operand of slot t)
 *      41 out [B K, M d] (f32 or bf16: the second product of slot t - 1), 42 b [B,E,K,d] f32
 *      43 y, 44 c [B,E,K,d] f32 (out), 45 readout [B,E,d] f32 (out), 46 abs_max [B,E] f32 (out)
 *   i:  0 B, 1 E, 2 N, 3 M, 4 K, 5 d, 7 Hf, 8 Hs (S's hidden units: u reads front hidden
 *       [0, Hs), z_hat [Hs, Hf)), 9 n, 10 D, 11 A, 12 S, 14 t,
 *       15 flags: 1 start slot t, 2 finish slot t - 1, 4 half, 8 refresh probes, 16 the address
 *          gate reads the summary (else u), 32 has the address gate, 64 the second product
 *          added b2,
 *       16 age_max, 17 front activation.
 *   f:  0 tau, 1 norm_eps, 2 cand_eps, 3 lambda_cost, 4 cell_overhead, 5 shared_cost, 6 limit.
 * Needs M a power of two, N M <= threads, 2 d <= threads, N <= threads.
 * Shared (words): Hf + threads + 2 d + d + M + M A + N M + 5 N + M + 4 K + 1 + 3 ceil(N / 32)
 *   + 2 K + 3 K d + K D d.
 */
extern "C" __global__ void hsn_ev_head(HsnWinArgs a) {
    extern __shared__ float shared[];
    const int E = a.i[1], N = a.i[2], M = a.i[3], K = a.i[4], d = a.i[5];
    const int Hf = a.i[7], Hs = a.i[8], n = a.i[9], D = a.i[10], A = a.i[11], S = a.i[12];
    const int t = a.i[14], flags = a.i[15], age_max = a.i[16], act_front = a.i[17];
    const bool start = (flags & 1) != 0, finish = (flags & 2) != 0, half = (flags & 4) != 0;
    const bool probes = (flags & 8) != 0, gate_summary = (flags & 16) != 0;
    const bool has_gate = (flags & 32) != 0, bias2_done = (flags & 64) != 0;
    const float tau = a.f[0], norm_eps = a.f[1], cand_eps = a.f[2], lambda = a.f[3];
    const float overhead = a.f[4], shared_cost = a.f[5], limit = a.f[6];
    const int row = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int F = 3 * d + 1 + S;
    // shared memory
    float* act_f = shared;                       // [Hf]
    float* part = act_f + Hf;                    // [nt]
    float* ue = part + nt;                       // [2d] u, eps
    float* q_s = ue + 2 * d;                     // [d]
    float* rowsc = q_s + d;                      // [M]
    float* gate_s = rowsc + M;                   // [M A]
    float* score_s = gate_s + M * A;             // [N M]
    float* score = score_s + N * M;              // [N]
    int* unit = reinterpret_cast<int*>(score + N);        // [N]
    int* order = unit + N;                       // [N]
    float* stale = reinterpret_cast<float*>(order + N);   // [N]
    float* scratch_n = stale + N;                // [N]
    float* cost_s = scratch_n + N;               // [M]
    int* out_cell = reinterpret_cast<int*>(cost_s + M);   // [K]
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
    float* slot_a = k_b + K;                     // [K d]
    float* slot_b = slot_a + K * d;              // [K d]
    float* slot_c = slot_b + K * d;              // [K d]
    float* vh = slot_c + K * d;                  // [K D d]

    float* h_tape = fout(a, 0);
    float* q_tape = fout(a, 1);
    int64_t* age_tape = static_cast<int64_t*>(a.p[3]);
    const uint8_t* live = static_cast<const uint8_t*>(a.p[5]);
    int64_t* cells_tape = static_cast<int64_t*>(a.p[26]);
    int64_t* units_tape = static_cast<int64_t*>(a.p[27]);
    uint8_t* valid_tape = static_cast<uint8_t*>(a.p[28]);

    if (finish) {
        // ---- the end of slot tp = t - 1: candidate, gated update, summary, ages, read-out ----
        const int tp = t - 1;
        const int64_t rec = static_cast<int64_t>(row) * E + tp;
        const int64_t* cells = cells_tape + rec * K;
        const int64_t* units = units_tape + rec * K;
        const uint8_t* valid = valid_tape + rec * K;
        const bool alive = live[rec] != 0;
        const float* h_in = h_tape + (static_cast<int64_t>(row) * (E + 1) + tp) * N * d;
        float* h_out = h_tape + (static_cast<int64_t>(row) * (E + 1) + tp + 1) * N * d;
        const float* q_pre = fptr(a, 2) + rec * d;
        float* q_out = q_tape + (static_cast<int64_t>(row) * (E + 1) + tp + 1) * d;
        const int64_t* age_in = age_tape + (static_cast<int64_t>(row) * (E + 1) + tp) * N;
        int64_t* age_out = age_tape + (static_cast<int64_t>(row) * (E + 1) + tp + 1) * N;
        const float* b2 = fptr(a, 21);
        const float* cand_gain = fptr(a, 19);
        const float* b_t = fptr(a, 42) + rec * K * d;
        const float* h_old_t = fptr(a, 37) + rec * K * d;
        float* y_t = fout(a, 43) + rec * K * d;
        float* c_t = fout(a, 44) + rec * K * d;
        // y: the chosen unit's block of the second product
        for (int e = tid; e < K * d; e += nt) {
            const int k = e / d, x = e - k * d;
            const int j = static_cast<int>(units[k]);
            float yv = load_value(a.p[41], (static_cast<int64_t>(row) * K + k) * (M * d) + j * d + x, half);
            if (!bias2_done) yv = ev_round(yv + b2[j * d + x], half);
            slot_a[e] = yv;
            y_t[e] = yv;
        }
        for (int i = tid; i < N; i += nt) scratch_n[i] = 0.0f;  // the written cells' marks
        __syncthreads();
        for (int k = tid; k < K; k += nt) {
            float sum = 0.0f;
            for (int x = 0; x < d; ++x) sum += slot_a[k * d + x] * slot_a[k * d + x];
            k_a[k] = rsqrtf(sum / static_cast<float>(d) + cand_eps);
            if (valid[k]) scratch_n[static_cast<int>(cells[k])] = 1.0f;
        }
        __syncthreads();
        for (int e = tid; e < K * d; e += nt) {
            const int k = e / d, x = e - k * d;
            const float cv = tanhf(slot_a[e] * k_a[k] * cand_gain[x]);
            c_t[e] = cv;
            const float ho = h_old_t[e], bv = b_t[e];
            const float hn = valid[k] ? (1.0f - bv) * ho + bv * cv : ho;
            slot_b[e] = hn;
            slot_c[e] = hn - ho;
        }
        for (int e = tid; e < N * d; e += nt) h_out[e] = h_in[e];
        __syncthreads();
        for (int e = tid; e < K * d; e += nt) {
            const int k = e / d;
            if (valid[k]) h_out[static_cast<int>(cells[k]) * d + (e - k * d)] = slot_b[e];
        }
        int n_valid = 0;
        for (int k = 0; k < K; ++k) n_valid += valid[k] ? 1 : 0;
        for (int x = tid; x < d; x += nt) {
            float delta = 0.0f, read = 0.0f;
            for (int k = 0; k < K; ++k) {
                delta += slot_c[k * d + x];
                read += valid[k] ? slot_b[k * d + x] : 0.0f;
            }
            q_out[x] = q_pre[x] + delta / static_cast<float>(N);
            fout(a, 45)[rec * d + x] = read / static_cast<float>(n_valid > 0 ? n_valid : 1);
        }
        for (int i = tid; i < N; i += nt) {
            const int64_t ag = age_in[i];
            age_out[i] = alive ? (scratch_n[i] != 0.0f ? 0 : ag + 1) : ag;
        }
        __syncthreads();
        // max |h| of the row after the update (NaN if any entry is NaN)
        float mx = 0.0f;
        for (int e = tid; e < N * d; e += nt) {
            const float v = fabsf(h_out[e]);
            mx = (v > mx || v != v) ? v : mx;
        }
        part[tid] = mx;
        __syncthreads();
        if (tid == 0) {
            float best = 0.0f;
            for (int i = 0; i < nt; ++i) {
                const float v = part[i];
                if (best == best) best = (v > best || v != v) ? v : best;
            }
            fout(a, 46)[rec] = best;
        }
        __syncthreads();
    }
    if (!start) return;

    // ---- slot t: the reset, the front, the scorer, the decision, the cell input ----
    const int64_t rec = static_cast<int64_t>(row) * E + t;
    float* h = h_tape + (static_cast<int64_t>(row) * (E + 1) + t) * N * d;
    int64_t* age = age_tape + (static_cast<int64_t>(row) * (E + 1) + t) * N;
    const float* q_in = q_tape + (static_cast<int64_t>(row) * (E + 1) + t) * d;
    const bool rs = static_cast<const uint8_t*>(a.p[4])[rec] != 0;
    const bool alive = live[rec] != 0;
    if (rs) {
        for (int e = tid; e < N * d; e += nt) h[e] = 0.0f;
        for (int i = tid; i < N; i += nt) age[i] = 0;
    }
    for (int x = tid; x < d; x += nt) {
        const float qv = rs ? 0.0f : q_in[x];
        fout(a, 2)[rec * d + x] = qv;
        q_s[x] = qv;
    }
    __syncthreads();
    // the front's hidden layer: the block's static part plus the summary's
    {
        const float* fq = fptr(a, 9);
        for (int j = tid; j < Hf; j += nt) {
            float acc = load_value(a.p[7], rec * Hf + j, half);
            const float* wj = fq + static_cast<int64_t>(j) * d;
            for (int x = 0; x < d; ++x) acc += wj[x] * ev_round(q_s[x], half);
            act_f[j] = ev_round(ev_activation(ev_round(acc, half), act_front), half);
        }
    }
    __syncthreads();
    // [u; z_hat] = act W2' + b2': output o reads its own block of the hidden layer, in `parts`
    // chunks summed in order
    {
        const int outs = 2 * d;
        const int parts = nt / outs;
        const int o = tid % outs, p = tid / outs;
        if (p < parts) {
            const int lo_block = o < d ? 0 : Hs, len = o < d ? Hs : Hf - Hs;
            const int chunk = (len + parts - 1) / parts;
            const int lo = lo_block + p * chunk;
            const int hi = lo + chunk < lo_block + len ? lo + chunk : lo_block + len;
            const float* wo = fptr(a, 10) + static_cast<int64_t>(o) * Hf;
            float acc = 0.0f;
            for (int j = lo; j < hi; ++j) acc += act_f[j] * wo[j];
            part[tid] = acc;
        }
        __syncthreads();
        if (tid < outs) {
            float acc = fptr(a, 11)[tid];
            for (int q = 0; q < parts; ++q) acc += part[q * outs + tid];
            acc = ev_round(acc, half);
            if (tid < d) {
                fout(a, 22)[rec * d + tid] = acc;
                ue[tid] = acc;
            } else {
                const int x = tid - d;
                fout(a, 23)[rec * d + x] = acc;
                const float e = ev_round(fptr(a, 6)[rec * d + x] - acc, half);
                fout(a, 24)[rec * d + x] = e;
                ue[tid] = e;
            }
        }
    }
    __syncthreads();
    // the scorer's terms that are one per row: [u; eps] and the bias, and A12's gate rows
    const float* sw = fptr(a, 12);
    if (tid < M) {
        float acc = fptr(a, 13)[tid];
        const float* wm = sw + static_cast<int64_t>(tid) * F;
        for (int x = 0; x < 2 * d; ++x) acc += wm[x] * ue[x];
        rowsc[tid] = acc;
    } else if (has_gate && tid - M < M * A) {
        const int g = tid - M;
        const float* wg = fptr(a, 14) + static_cast<int64_t>(g) * d;
        const float* read = gate_summary ? q_s : ue;
        float acc = 0.0f;
        for (int x = 0; x < d; ++x) acc += wg[x] * ev_round(read[x], half);
        gate_s[g] = ev_round(acc, half);
    }
    for (int j = tid; j < M; j += nt) cost_s[j] = fptr(a, 20)[j];
    __syncthreads();
    // the logits, one thread per (cell, unit); the M lanes of a cell reduce by shuffles
    const int m_shift = __ffs(M) - 1;
    {
        const int j = tid & (M - 1);
        float s = -INFINITY;
        if (tid < N * M) {
            const int i = tid >> m_shift;
            const float* wm = sw + static_cast<int64_t>(j) * F;
            const float* hi_ = h + static_cast<int64_t>(i) * d;
            const float* sc = fptr(a, 8) + (rec * N + i) * S;
            float acc = rowsc[j];
            for (int x = 0; x < d; ++x) acc += wm[2 * d + x] * ev_round(hi_[x], half);
            acc += wm[3 * d] * ev_round(static_cast<float>(age[i]) / static_cast<float>(age_max), half);
            for (int x = 0; x < S; ++x) acc += wm[3 * d + 1 + x] * ev_round(sc[x], half);
            float logit = ev_round(acc, half);
            if (has_gate) {
                float g = 0.0f;
                for (int x = 0; x < A; ++x) g += ev_round(sc[S - A + x], half) * gate_s[j * A + x];
                logit = ev_round(logit + ev_round(g, half), half);
            }
            fout(a, 25)[(rec * N + i) * M + j] = logit;
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
    float key = -1.0f;  // this cell's age if it is over age_max
    if (tid < N && probes) {
        const int64_t ag = age[tid];
        if (ag > static_cast<int64_t>(age_max)) key = static_cast<float>(ag);
    }
    __syncthreads();
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
            fout(a, 31)[rec] = 0.5f * (lowest_in + highest_out);
            part[0] = 0.5f * (lowest_in + highest_out);
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
            fout(a, 35)[rec] = shared_cost + spent;
            fout(a, 36)[rec] = static_cast<float>(n_fresh);
        }
    }
    __syncthreads();
    const float theta = part[0];
    // the records, and the straight-through values of the written slots (cell 0 and unit 0
    // stand in on the others, as in the reference)
    for (int k = tid; k < K; k += nt) {
        const bool raw = (out_flag[k] & 1) != 0;
        const bool ok = raw && alive;
        static_cast<int64_t*>(a.p[29])[rec * K + k] = out_cell[k];
        static_cast<int64_t*>(a.p[30])[rec * K + k] = out_unit[k];
        const int c = ok ? out_cell[k] : 0, u = ok ? out_unit[k] : 0;
        cells_tape[rec * K + k] = c;
        units_tape[rec * K + k] = u;
        valid_tape[rec * K + k] = ok ? 1 : 0;
        out_cell[k] = c;
        out_unit[k] = u;
        out_flag[k] = ok ? 1 : 0;
        const float* sc = score_s + c * M;
        float top = sc[0];
        for (int j = 1; j < M; ++j) top = sc[j] > top ? sc[j] : top;
        float denom = 0.0f;
        uint8_t* ties = static_cast<uint8_t*>(a.p[34]) + (rec * K + k) * M;
        for (int j = 0; j < M; ++j) {
            denom += expf((sc[j] - top) / tau);
            ties[j] = (ok && sc[j] == top) ? 1 : 0;
        }
        fout(a, 32)[rec * K + k] = 1.0f / (1.0f + expf(-(top - theta) / tau));
        fout(a, 33)[rec * K + k] = expf((sc[u] - top) / tau) / denom;
    }
    __syncthreads();
    // the messages: V h of every source of every slot's cell, weighted by its edge
    {
        const float* vw = fptr(a, 15);
        const float* edge = fptr(a, 16);
        const int32_t* pad_src = static_cast<const int32_t*>(a.p[17]);
        for (int e = tid; e < K * D * d; e += nt) {
            const int x = e % d, ke = e / d;
            const int k = ke / D, ed = ke - k * D;
            const int c = out_cell[k];
            const float* hs = h + static_cast<int64_t>(pad_src[c * D + ed]) * d;
            const float* vx = vw + static_cast<int64_t>(x) * d;
            float acc = 0.0f;
            for (int y = 0; y < d; ++y) acc += vx[y] * ev_round(hs[y], half);
            vh[e] = ev_round(ev_round(acc, half) * ev_round(edge[c * D + ed], half), half);
        }
    }
    __syncthreads();
    for (int e = tid; e < K * d; e += nt) {
        const int k = e / d, x = e - k * d;
        float acc = 0.0f;
        for (int ed = 0; ed < D; ++ed) acc += vh[(k * D + ed) * d + x];
        const float mv = ev_round(acc, half);
        const float ho = h[out_cell[k] * d + x];
        slot_a[e] = ho;
        slot_b[e] = mv;
        fout(a, 37)[rec * K * d + e] = ho;
        fout(a, 38)[rec * K * d + e] = mv;
    }
    __syncthreads();
    // the cell norm over [u; eps; h_old; m]
    for (int k = tid; k < K; k += nt) {
        float sum = 0.0f;
        for (int x = 0; x < 2 * d; ++x) sum += ue[x] * ue[x];
        for (int x = 0; x < d; ++x) sum += slot_a[k * d + x] * slot_a[k * d + x];
        for (int x = 0; x < d; ++x) sum += slot_b[k * d + x] * slot_b[k * d + x];
        const float r = rsqrtf(sum / static_cast<float>(n) + norm_eps);
        k_a[k] = r;
        fout(a, 39)[rec * K + k] = r;
    }
    __syncthreads();
    {
        const float* gain = fptr(a, 18);
        const int64_t base = (static_cast<int64_t>(t) * a.i[0] + row) * K * n;
        for (int e = tid; e < K * n; e += nt) {
            const int k = e / n, x = e - k * n;
            const float xv = x < 2 * d ? ue[x] : (x < 3 * d ? slot_a[k * d + x - 2 * d] : slot_b[k * d + x - 3 * d]);
            store_value(a.p[40], base + e, half, xv * k_a[k] * gain[x]);
        }
    }
}

/* hsn_ev_pick: between the two products of slot t. The chosen unit's hidden row of the first
 * product, its activation (the second product's operand) and the gate.
 *   p: 0 first [B K, M H + d] (f32 or bf16), 1 b1 [M H + d] f32 (read unless the product added
 *        it), 2 units [B,E,K] i64, 3 hidden [B,E,K,H] (f32 or bf16, out), 4 act [B K, H] (f32 or
 *        bf16, out), 5 gate_pre [B,E,K,d] f32 (out), 6 b [B,E,K,d] f32 (out).
 *   i: 0 B, 1 E, 2 K, 3 d, 4 H, 5 M, 6 t, 7 flags (1 half, 2 the product added b1), 8 activation.
 */
extern "C" __global__ void hsn_ev_pick(HsnWinArgs a) {
    const int E = a.i[1], K = a.i[2], d = a.i[3], H = a.i[4], M = a.i[5], t = a.i[6];
    const bool half = (a.i[7] & 1) != 0, bias_done = (a.i[7] & 2) != 0;
    const int code = a.i[8];
    const int row = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int64_t rec = static_cast<int64_t>(row) * E + t;
    const int64_t* units = static_cast<const int64_t*>(a.p[2]) + rec * K;
    const float* b1 = fptr(a, 1);
    const int width = M * H + d;
    for (int e = tid; e < K * H; e += nt) {
        const int k = e / H, j = e - k * H;
        const int u = static_cast<int>(units[k]);
        const int64_t p = static_cast<int64_t>(row) * K + k;
        float hv = load_value(a.p[0], p * width + u * H + j, half);
        if (!bias_done) hv = ev_round(hv + b1[u * H + j], half);
        store_value(a.p[3], rec * K * H + e, half, hv);
        store_value(a.p[4], p * H + j, half, ev_activation(hv, code));
    }
    for (int e = tid; e < K * d; e += nt) {
        const int k = e / d, x = e - k * d;
        const int64_t p = static_cast<int64_t>(row) * K + k;
        float gv = load_value(a.p[0], p * width + M * H + x, half);
        if (!bias_done) gv = ev_round(gv + b1[M * H + x], half);
        fout(a, 5)[rec * K * d + e] = gv;
        fout(a, 6)[rec * K * d + e] = sigmoidf(gv);
    }
}
