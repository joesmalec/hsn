/*
 * hsn_ops.h -- the C ABI of the four HSN ops, forward and backward (chunk P2.9).
 *
 * One header shared by the PyTorch extensions (P2.10 CUDA, P2.11 Metal) and the Rust hosts
 * (rust/hsn-kernels, P5.2). The numerical reference is the pure-Rust CPU oracle
 * (hsn_kernels::CpuBackend). Every implementation of these functions must reproduce the test
 * vectors in kernels/testvectors/ (listed with their SHA-256 in MANIFEST.json) within the
 * tolerances below. The same specification is mirrored in rust/hsn-kernels/src/spec.rs and
 * python/hsn/kernels/spec.py; tests check that the constants of all three agree.
 *
 * Conventions (every function)
 * ----------------------------
 * - Shapes use B batch rows, N cells, d state width, K selected slots per row, P = B*K
 *   (cell, unit) pairs, M capability units, H unit hidden width, E graph edges. The batch row
 *   index is written r, so that b always names the gate tensor.
 * - Tensors are dense, row-major and contiguous (the last index varies fastest): no strides,
 *   no offsets.
 * - Index tensors are int32_t. Every float tensor of one call has the dtype `dtype`, one of
 *   HSN_DTYPE_*: f32 on every backend, f16 on CUDA and Metal, bf16 on CUDA only. f16 and bf16
 *   implementations accumulate in f32 and round once per output element.
 * - B, N, d, K, P, M, H are >= 1 and E >= 0. Every dimension and every tensor's element count
 *   (for example B*N*d, P*4*d, M*4*d*H) must be <= INT32_MAX, else HSN_ERR_SIZE_OVERFLOW
 *   (impossible at the reference sizes; checked anyway). A pointer to a zero-length array may
 *   be NULL.
 * - Outputs are written, never accumulated into: every grad_* output receives the full
 *   gradient of its input (zero wherever the op does not depend on it). No output aliases an
 *   input or another output, except the in-place h, q and age of hsn_gated_delta_update.
 * - deterministic != 0 asks for bitwise run-to-run reproducible results (a fixed reduction
 *   order, no floating-point atomics); 0 allows atomics. The CPU oracle is always
 *   deterministic.
 * - stream is the queue the work goes to: a cudaStream_t (CUDA), an MTLCommandBuffer*
 *   (Metal) or NULL (CPU). GPU implementations return once the work is enqueued.
 * - The return value is HSN_OK or an HSN_ERR_* code. The CPU oracle checks every
 *   precondition below; a GPU implementation checks what it can without a device
 *   synchronisation (dtype, dimensions, sizes) and leaves the index preconditions to its host
 *   (violating them is undefined behaviour there).
 *
 * Tolerances against the vectors, elementwise |x - ref| <= atol + rtol * |ref|:
 *   f32: atol = rtol = 1e-5.   f16, bf16: atol = rtol = 1e-2 (against the same f32 vectors).
 * Integer outputs (age) match exactly.
 *
 * Where the ops sit in the HSN sparse round (python/hsn/model/cell.py, PDGCell.step_sparse):
 *   m     = csr_messages(h, graph, V, cell_idx)              reads the pre-round h
 *   v_raw = gather_concat(u, eps, h, m, cell_idx)
 *   v     = RMSNorm(v_raw)                                   host op (amendment A1)
 *   F     = grouped_unit_mlp(v, unit_idx, W1, b1, W2, b2)
 *   b     = sigmoid(gate(v)),  c = tanh(W_out(alpha * F))     host ops
 *   gated_delta_update(h, q, age, cell_idx, valid, c, b)     the raw h, in place
 * The RMSNorm, the gate, W_out, tanh and the alpha mix are not part of these ops. Changing what
 * any op computes is a new HSN_OPS_SPEC_VERSION and a regeneration of the vectors.
 */
#ifndef HSN_OPS_H
#define HSN_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version of this specification; recorded in kernels/testvectors/MANIFEST.json. */
#define HSN_OPS_SPEC_VERSION 1

/* Float dtype codes (the `dtype` argument). */
#define HSN_DTYPE_F32 0
#define HSN_DTYPE_F16 1
#define HSN_DTYPE_BF16 2

/* Status codes (the return value). */
#define HSN_OK 0
#define HSN_ERR_INVALID_ARGUMENT 1   /* NULL pointer, dimension < 1, flag not 0/1, dtype mix */
#define HSN_ERR_UNSUPPORTED_DTYPE 2  /* dtype not implemented by this backend */
#define HSN_ERR_INDEX_OUT_OF_RANGE 3 /* a cell or unit index outside its range */
#define HSN_ERR_DUPLICATE_INDEX 4    /* a cell selected twice by valid slots of one row */
#define HSN_ERR_SIZE_OVERFLOW 5      /* a dimension or element count above INT32_MAX */
#define HSN_ERR_BAD_GRAPH 6          /* indptr / indices are not a CSR over N cells */
#define HSN_ERR_BACKEND 7            /* launch, device or driver failure */

/*
 * (a) gather_concat: the raw cell input of the selected cells.
 *   v[r,k,:] = [ u[r,:] ; eps[r,:] ; h[r, cell_idx[r,k], :] ; m[r,k,:] ]
 * u, eps [B,d]; h [B,N,d]; m [B,K,d]; cell_idx [B,K] in [0,N), repeats allowed (the host maps
 * padded slots to cell 0); v [B,K,4d].
 */
int hsn_gather_concat(const void* u, const void* eps, const void* h, const void* m,
                      const int32_t* cell_idx, void* v, int B, int N, int d, int K, int dtype,
                      void* stream);

/*
 * Backward of (a), from grad_v [B,K,4d]:
 *   grad_u[r,:]   = sum_k grad_v[r,k, 0:d]
 *   grad_eps[r,:] = sum_k grad_v[r,k, d:2d]
 *   grad_h[r,n,:] = sum_{k : cell_idx[r,k] = n} grad_v[r,k, 2d:3d]    (0 for unselected n)
 *   grad_m[r,k,:] = grad_v[r,k, 3d:4d]
 * grad_u, grad_eps [B,d]; grad_h [B,N,d]; grad_m [B,K,d].
 */
int hsn_gather_concat_bwd(const void* grad_v, const int32_t* cell_idx, void* grad_u,
                          void* grad_eps, void* grad_h, void* grad_m, int B, int N, int d, int K,
                          int dtype, int deterministic, void* stream);

/*
 * (b) grouped_unit_mlp: unit F_j applied to its pairs, j = unit_idx[p] in [0,M).
 *   z1[p,:]  = v[p,:] W1[j] + b1[j,:]          saved as hidden_saved [P,H] (pre-GELU)
 *   out[p,:] = gelu(z1[p,:]) W2[j] + b2[j,:]
 *   gelu(x)  = x/2 * (1 + erf(x / sqrt(2)))     the exact form (torch.nn.GELU())
 * v [P,4d]; unit_idx [P]; W1 [M,4d,H]; b1 [M,H]; W2 [M,H,d]; b2 [M,d]; out [P,d]. W1[j] is
 * [in, out] (a row vector times W1[j]), as hsn.model.units.UnitBank stores it.
 */
int hsn_grouped_unit_mlp_fwd(const void* v, const int32_t* unit_idx, const void* W1,
                             const void* b1, const void* W2, const void* b2, void* hidden_saved,
                             void* out, int P, int M, int d, int H, int dtype, void* stream);

/*
 * Backward of (b), from grad_out [P,d] and the saved z1 (so b1 is not needed), j = unit_idx[p]:
 *   a[p,:]       = gelu(z1[p,:])
 *   dz1[p,:]     = (grad_out[p,:] W2[j]^T) * gelu'(z1[p,:]),   gelu'(x) = Phi(x) + x phi(x)
 *                  (Phi the standard normal CDF, phi its density)
 *   grad_v[p,:]  = dz1[p,:] W1[j]^T
 *   grad_W1[j]   = sum_{p : unit_idx[p] = j} v[p,:]^T dz1[p,:]       grad_b1[j,:] = sum_p dz1[p,:]
 *   grad_W2[j]   = sum_{p : unit_idx[p] = j} a[p,:]^T grad_out[p,:]  grad_b2[j,:] = sum_p grad_out[p,:]
 * Units without a pair get zero gradients. Shapes as the forward's.
 */
int hsn_grouped_unit_mlp_bwd(const void* v, const int32_t* unit_idx, const void* W1,
                             const void* W2, const void* hidden_saved, const void* grad_out,
                             void* grad_v, void* grad_W1, void* grad_b1, void* grad_W2,
                             void* grad_b2, int P, int M, int d, int H, int dtype,
                             int deterministic, void* stream);

/*
 * (c) csr_messages: neighbour messages of the selected cells, from the pre-round states.
 *   i = cell_idx[r,k];  m[r,k,:] = sum_{e = indptr[i]}^{indptr[i+1]-1} a[e] * (V h[r, indices[e], :])
 *   (V x)[o] = sum_c V[o,c] x[c]: V [d,d] is [d_out, d_in], like torch.nn.Linear.weight.
 * h [B,N,d]; indptr [N+1]; indices [E] (the source cell of every edge j -> i, grouped by target
 * row i); a [E] the normalised edge weights a_ij in CSR order; cell_idx [B,K] in [0,N), repeats
 * allowed; m [B,K,d]. The CSR must have indptr[0] = 0, indptr non-decreasing, indptr[N] = E and
 * indices in [0,N), else HSN_ERR_BAD_GRAPH. Rows may be empty (their message is 0). Sorted rows
 * and the absence of self loops are not required (hsn.model.graph guarantees both).
 */
int hsn_csr_messages(const void* h, const int32_t* indptr, const int32_t* indices, const void* a,
                     const void* V, const int32_t* cell_idx, void* m, int B, int N, int d, int K,
                     int E, int dtype, void* stream);

/*
 * Backward of (c), from grad_m [B,K,d]; sums over every slot (r,k) and every edge e of row
 * i = cell_idx[r,k], with source j = indices[e]:
 *   grad_h[r,j,:] = sum a[e] * (V^T grad_m[r,k,:])                  [B,N,d]
 *   grad_V        = sum a[e] * grad_m[r,k,:] (outer) h[r,j,:]       [d,d] ([d_out, d_in])
 *   grad_a[e]     = sum grad_m[r,k,:] . (V h[r,j,:])                [E]
 */
int hsn_csr_messages_bwd(const void* h, const int32_t* indptr, const int32_t* indices,
                         const void* a, const void* V, const int32_t* cell_idx,
                         const void* grad_m, void* grad_h, void* grad_a, void* grad_V, int B,
                         int N, int d, int K, int E, int dtype, int deterministic, void* stream);

/*
 * (d) gated_delta_update, in place on h, q and age. For every valid slot (valid[r,k] = 1),
 * n = cell_idx[r,k]:
 *   h_old_saved[r,k,:] = h[r,n,:]
 *   h[r,n,:]  <- (1 - b[r,k,:]) * h[r,n,:] + b[r,k,:] * c[r,k,:]
 *   q[r,:]    <- q[r,:] + ( sum_{valid k} (h_new - h_old)[r,k,:] ) / N
 *   age[r,n]  <- 0
 * valid [B,K] holds 0 or 1; the cell_idx [B,K] of the valid slots of a row are distinct and in
 * [0,N). A padded slot (valid = 0) is never read, whatever cell_idx holds there (even out of
 * range): it writes nothing, adds nothing to q and gets h_old_saved = 0. h [B,N,d]; q [B,d] (the
 * summary, maintained incrementally; the op does not require q = mean over cells of h); age
 * [B,N] (only reset here; the host ticks it); c (candidate, tanh by the host), b (gate in (0,1),
 * sigmoid by the host) [B,K,d]; h_old_saved [B,K,d].
 */
int hsn_gated_delta_update(void* h, void* q, int32_t* age, const int32_t* cell_idx,
                           const int32_t* valid, const void* c, const void* b, void* h_old_saved,
                           int B, int N, int d, int K, int dtype, void* stream);

/*
 * Backward of (d), from G = grad_h_out [B,N,d] and g = grad_q_out [B,d] (the gradients with
 * respect to h and q after the update). For every valid slot, n = cell_idx[r,k]:
 *   grad_h[r,n,:] = (1 - b[r,k,:]) * G[r,n,:] - b[r,k,:] * g[r,:] / N
 *   grad_c[r,k,:] = b[r,k,:] * (G[r,n,:] + g[r,:] / N)
 *   grad_b[r,k,:] = (c[r,k,:] - h_old_saved[r,k,:]) * (G[r,n,:] + g[r,:] / N)
 * and grad_h[r,n,:] = G[r,n,:] for every cell no valid slot selects, grad_q = g, grad_c =
 * grad_b = 0 on padded slots. The g/N terms are the summary coupling: q depends on every
 * selected delta, so dq reaches every selected h, c and b.
 * grad_h [B,N,d]; grad_q [B,d]; grad_c, grad_b [B,K,d].
 */
int hsn_gated_delta_update_bwd(const int32_t* cell_idx, const int32_t* valid, const void* c,
                               const void* b, const void* h_old_saved, const void* grad_h_out,
                               const void* grad_q_out, void* grad_h, void* grad_q, void* grad_c,
                               void* grad_b, int B, int N, int d, int K, int dtype, void* stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HSN_OPS_H */
