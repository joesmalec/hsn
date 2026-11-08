# Amendment A4: RMS normalisation of the write candidate before its tanh (P1.13, opt-in)

Status: **opt-in, default `none`**. Added by chunk P1.13 (7 Nov 2026) as a configuration option,
`model.cand_norm`. It is adopted into the design only if the P1.13 lever experiments support it
and the user approves; until then every configuration, result and gate runs the P1.5 candidate
(A1 included) unchanged, bit for bit. The numbering skips A2 (branch `a2-gate-protocol`, unmerged)
and A3 (the cell-identity designs, informal).

## Cause

At the learning rate where the comparison models learn (lr 3e-3, B32, the 3,000-step cosine
schedule with warm-up 200), dense HSN `ref_a1` (d = 64, compiled) fails its memory check. The
write gate stays open, but the candidate saturates and the cell starves of gradient. Medians over
the nine diagnostics at steps 400 to 600 (P1.11 database, `/home/joe/hsn-experiments`):

| run | gate p50 | grad_share/cell | sat99/answer | update_ratio/cell |
| --- | --- | --- | --- | --- |
| exp 48, lr 1e-3 (healthy reference) | 0.050 | 0.0256 | 0.39 | 1.2e-3 |
| exp 49 (E1), lr 3e-3 | 0.116 | 0.00055 | 0.58 | 9.5e-5 |

- In E1 59 % of the state sits at |h| ≥ 0.99 at answer positions.
- The cell's gradient share is about 1/50 of the healthy run's, and its weights move about 1/12 as
  much per step.
- The decoder takes about 99.8 % of the gradient norm.
- The collapse starts about 150 steps after the learning-rate peak.

The cell input is already normalised (A1's `cell_norm`), so the candidate's pre-activation
`W_out · Σ_j α_j F_j(v)` grows only through the cell's own weights (the unit MLPs, `W_out`, the
A1 gain). At 3e-3 they grow fast enough to push `tanh` into its flat tails. There `tanh'` is near
0, so neither the write nor the gradient back through it carries signal.

## The change

When `model.cand_norm = rms`, the candidate of P1.5's cell update (plan §2.3) is normalised over
its width d before the tanh, with a learned gain `g` (eps 1e-6):

    before:  c_i = tanh( W_out · Σ_j α_ij F_j(v_i) )
    after:   c_i = tanh( RMSNorm_g( W_out · Σ_j α_ij F_j(v_i) ) )
             RMSNorm_g(x) = g ⊙ x / sqrt(mean(x²) + eps)

- **Unchanged:** the tanh, the gate `b_i = σ(B · v_i)` and the convex update
  `h ← (1 − b_i) ⊙ h + b_i ⊙ c_i`. The state therefore stays in [−1, 1]
  (`test_p1_13_cand_norm_window_path`, `test_p1_13_cand_norm_autocast_bound_cuda`).
- **What changes:** the pre-activation's RMS is `|g|` (1 at initialisation) instead of whatever the
  weights grew to. The gain is learned, so the cell can still sharpen its writes. It moves by
  about one learning rate per step, so a re-saturation would show in sat99 over hundreds of steps.
- **Sparse round:** the per-slot `α` is applied after the norm: `c = tanh( α ⊙ RMSNorm_g( W_out ·
  F ) )`. `α` is exactly 1.0 forward on every valid slot, so the values equal the normalised
  `W_out(α ⊙ F)`. But `α` is P2.2's only straight-through gradient entry, and a scale-invariant
  norm applied after it would zero that gradient (`test_p1_13_cand_norm_keeps_the_selection_gradient`).
- **Masked round:** it normalises the mixture as the dense round does.
- **Where it runs:** every path that computes the candidate runs the norm:
  - `PDGCell.step_dense` (dense, masked, the P1.10a window, P2.8's compiled window);
  - `PDGCell.step_sparse` (eager, static, compiled);
  - the cost-profile replica `profile_dense_costs`.
- **Kernels:** none is affected. The P2.9 ops treat `W_out`, the tanh and the norms as host ops.
- **Precision:** under bf16 autocast the norm runs in the state dtype (P1.9).

Literature: Ba et al. 2016 normalise the candidate pre-activations of layer-norm RNNs with a learned
gain. RMSNorm (Zhang and Sennrich 2019) drops the centring, as A1's `cell_norm` does. Okpekpe and
Orvieto 2025 find that only the normalised recurrent model (DeltaNet) is robust across learning
rates on associative recall.

## Parameter table (Appendix C.1)

A new row, `cand_norm`: the gain of width d, 64 parameters for `ref_a1`. It is 0 with the default
`none`, so `ref_a1` keeps 1,820,617. With `rms` the total is 1,820,681, and 1,820,809 counting the
128 edge logits.
- The gain is initialised to ones, which draws no random numbers, so every other weight is drawn
  exactly as without it.
- Appendix C.2's MACs are unchanged: norms are outside the table, as with A1.
- State-dict key: `cell.cand_norm.weight [d]`, present only with `rms`.
- A checkpoint without the key loads into the default model; loading it into an `rms` model is
  refused (strict load).

## What it does not change

- No attention over positions, no linear or diagonal recurrence, and the cells are not merged.
- The persistent cells in the fixed sparse graph with `V` messages are unchanged, and so are the
  shared unit bank with per-cell `α` mixing, predict-before-encode with `ε` as a cell input, and
  the router or top-K selection.
- The summary `q` and the decoder `D([q; u])` are unchanged.
- Only the candidate's pre-activation is rescaled.
- The GRU and Transformer baselines refuse the option, and so do the P1.11 variants other than
  `pdg`.

## Experiments that decide it

The P1.13 lever arms, registered by the audit in `AUDIT.md` and approved by the user, run on the
P1.11 harness after the chunk merges:
- **L3:** E1's recipe with `model.cand_norm=rms`.
- **R-cand:** the detecting recipe, lr 2e-3, 3,000 of a 9,000-step cosine.

Each is a 600-step memory check: the medians over steps 400 to 600 must have gate p50 ≥ 0.01 and
grad_share/cell ≥ 0.01, with sat99 and update_ratio/cell reported.
- **Prediction:** sat99 ≤ 0.05 and a pass.
- **Caveat:** a weight in front of a norm gets a gradient that scales as 1/‖W‖, so the share can
  fall while the cell's updates stay healthy. Read update_ratio/cell beside it.

If the gain re-saturates the candidate, a fixed-gain value of the same key is the follow-up.
