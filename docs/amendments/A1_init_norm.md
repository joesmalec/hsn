# Amendment A1: signal-preserving initialisation and normalisation of the write path (P1.5a)

Status: adopted 5 Nov 2026 for chunk P1.5a, before the G1 rerun. Amends Appendix A.1 (`model.init`),
Appendix C.1 (the parameter table gains a row) and the P1.5 cell equations (a normalisation of the
cell input and a non-zero gate bias at initialisation). The G1 decision of 5 Nov 2026 (FAIL, PR #22)
was measured on the pre-amendment model and stands as that record; every later G1 run uses the
amended model.

## Cause

The G1 gate failed with HSN `ref_a1` at the majority floor on both frozen tasks (dev kv 0.160, delayed
0.020), at the floor on the training documents themselves, with a trained checkpoint that ignores the
context. The gradient-flow diagnosis (`evidence/G1/HSN_GRADIENT_DIAGNOSIS.md`, branch `g1-diagnosis`)
traced it to the model's initialisation: every weight starts at `trunc_normal_0.02` (RMS 0.0176, biases
0) and the path from an input token to the cell state has no normalisation (encoder, shared path,
unit MLPs, `W_out`, tanh, the gate's convex update, the mean over cells). Each layer shrinks the
signal 20 to 40 times, so at initialisation the state and the summary the decoder reads are 1e-6 and
the gradient back to an earlier token is zero. The optimiser can only inflate the chain through
gradients of the same vanishing scale; in 12,000 steps it inflated the state to 0.11 but never learned
to write or read a binding.

## Measured evidence (pre-amendment model, `ref_a1`, seed 0, fp32 probes on real documents)

| measurement | at initialisation | after 12,000 gate steps |
| --- | --- | --- |
| RMS along the write path: embedding / encoder output / unit output / write candidate / state / summary | 1.8e-2 / 4.8e-4 / 1.9e-5 / 9.8e-7 / 8.7e-7 / 8.7e-7 | 3.8e-2 / 2.8e-2 / 4.9e-1 / 1.2e-1 / 1.1e-1 / 1.1e-1 |
| gradient norm of the answer logit w.r.t. the fact's value embedding (32 documents, median) | 0.000e+00 (below 1e-8 in 32 of 32) | 3.0e-6 (the current input: 1.5e-1) |
| update gate b (mean; p10 / p50 / p90) | 0.500 (0.500 / 0.500 / 0.500) | 0.220 (0.043 / 0.143 / 0.547) |
| mean abs(h_new - h_prev) per position | 0.0000 | 0.0018 |
| context sensitivity: argmax changes when the fact's value is edited (50 documents) | | 0 of 50 |

Controls: a standard GRU update inside the same wrapper, at the same initialisation, also fails
(dev delayed 0.045 after 3,000 steps); the prediction loss is not involved (lambda_p = 0 identical);
HSN fails even at a 3-to-8-position dependency that the GRU baseline learns to 1.000 in 500 steps;
and with `model.init=trunc_normal_0.1` as the only change HSN learns that short variant to 0.918
held-out and 0.903 on unseen keys in 2,000 steps. The leak of the gate (0.5 at initialisation) is a
secondary factor that limits long-range retention once the path is alive.

## The change

1. **Fan-in-scaled initialisation of the token-to-state path.** New config field `model.state_init`
   (default `fan_in`; `trunc_normal_<std>` still accepted, so the old behaviour is reproducible). It
   applies to the encoder `E`, the predictor `P`, the shared transform `S`, the unit MLPs (`W1`, `W2`),
   the message projection `V`, the output projection `W_out` and the gate's weight: truncated normal
   (±2σ) with σ = 1/sqrt(fan_in) of the layer, biases zero. `model.init` (default `trunc_normal_0.02`)
   keeps governing the byte and context embeddings, the scorer, the halting head and the decoder `D`.
2. **RMSNorm on the cell input.** `v_i = [u; eps; h_i; m_i]` is normalised over its 4d width (a
   learned gain, eps 1e-6) before the unit MLPs and the gate, on the dense, masked and sparse paths
   alike, so that the scale of what the cell reads no longer depends on the initialisation or on
   training drift. Config field `model.cell_norm` (`rms`, default; `none` reproduces the old cell).
   Appendix C.1 gains the row `cell_norm` (4d = 256 parameters for `ref_a1`), and the reference
   configuration's total moves from 1,820,361 to 1,820,617.
3. **Gate bias initialised towards retention.** New config field `model.gate_bias_init` (default
   -3.0): b = sigmoid(-3) = 0.047 at initialisation, so a cell keeps 95 % of its state per position
   (half-life about 14 positions; 32 positions back, 21 % of a written value and of its gradient
   survive) instead of overwriting half of it, and the leak is learned rather than fought. The value
   is a compromise between retention and a state that still moves at the start.

No other model change. The invariant `tests/invariants/test_p1_5a_signal_scale.py` pins the result:
at initialisation, on a random batch, the RMS of the embedding, encoder output, unit output, write
candidate, state and summary lie within 0.1x to 10x of the embedding RMS, and the gradient of an
answer logit with respect to an input 32 positions back exceeds 1e-4; the same test shows the
pre-amendment settings violate both.

## What it applies to

The G1 rerun (overfit stage, recipe search and the five seeds) runs on the amended model; the
P1.9 gate script and the frozen G1 task set are unchanged. Chunks that rely on the P1.5 cell's
equations (P1.8 invariants, P2.1 scorer features, P2.5 sparse path) see the same interface; tests
whose expected values depended on the old initialisation or on the parameter total are updated in
the same PR and listed in its description.
