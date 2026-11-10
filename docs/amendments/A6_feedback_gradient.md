# Amendment A6 — the feedback gradient (`train.feedback_detach`)

Status: opt-in, default off (10 Nov 2026). No forward value changes; a training-recipe choice
about which paths the backward pass follows. Results: see "Evidence" below and
`docs/p1_18_a6_results.md`.

## What it is

At every position the recurrent state `(h, q)` is read in five places:

| Read | Path | With A6 |
| --- | --- | --- |
| the carry | `h[t] = (1 − b) ⊙ h[t−1] + b ⊙ c`, `q[t] = q[t−1] + Σ Δh / N` | kept (never cut) |
| the decoder and the selected-cell readout | `D([q[t]; u[t]; readout[t]])` | kept (never cut) |
| the prediction loss | `L_pred(P(q[t−1]), z*[t])` | kept (never cut) |
| `summary` | `u[t] = S([z[t]; q[t−1]; ctx[t]])` | `S` reads `q[t−1].detach()` |
| `eps` | the cell and the scorer read `ε[t] = z[t] − P(q[t−1])` | they read `z[t] − P(q[t−1]).detach()` |
| `state` | a cell's units, gate and messages read `h[t−1]` | they read `h[t−1].detach()` |
| `router` | the scorer's features `[u; ε; h]` | detached; a selection gradient trains the scorer's weights only |

`train.feedback_detach` is a list of the last four names. Each one makes that read a constant in
the backward pass. The values of every tensor are the same as without it (bit for bit on the CPU,
`test_forward_values_are_unchanged`), so the model, its state, the checkpoints' weights, serving and
evaluation are unchanged. Only the gradient is.

With all four, the backward recurrence of the state is

    ∂h[t]/∂h[t−1] = diag(1 − b)   on the cells position t wrote, the identity elsewhere
    ∂q[t]/∂q[t−1] = I,   ∂q[t]/∂h[t−1] = −b / N on the written cells

which has no singular value above 1 in `h` and is linear in `q`. A gradient cannot grow on its way
back through a window (`test_full_cut_leaves_the_carry_as_the_only_recurrence`,
`test_full_cut_bounds_the_state_gradient`). Every module still learns: the task loss reaches the
cell, `S`, the encoder and the embeddings at every write through the carry, the router through the
selection estimator, and the predictor through `L_pred`, which also still shapes `q`.

## Why

The A5 runs lose accuracy mid-run in bursts of exploded gradients (exps 325, 340, 342, 345, 346;
`docs/p1_17_training_findings_2026_11_10.md`). On the stored checkpoints of exps 345 and 346
(`/home/joe/hsn-experiments/a5-fix/probe/`):

- The straight-through selection paths do not cause it: with the mask, the unit weights or every
  scorer feature detached, the gradient norm of an exploding window is unchanged (159 → 158).
- The state gradient does not grow smoothly. It jumps by 7× to 22× at single positions and is flat
  in between. The candidate is saturated on 95–99 % of its coordinates from step 1,500 on and
  the gate is closed or fully open, so the cell is a bank of hard switches: almost no gradient
  anywhere, and a very large one where a switch is in transition.
- Those jumps are in the feedback reads. Cutting `q → P` and `q → S` takes exp 346's worst window
  from 159 to 6.9; cutting the cell's read of `h` and the messages takes it to 2.5 (an ordinary
  window is 2 to 3). Exp 345 before its collapse: `q → S` alone, 29 → 4.7.
- The memory this task needs does not use those paths. A fact is written from `u[t]`, held by the
  carry, and read by the decoder; the key address (A5) picks the cells. The feedback reads are what
  a key-blind strategy needs ("which pair comes next").

Clipping keeps the size of such a gradient at 1 but not its direction: with AdamW every step of a
burst is a full-size step along one document's switching direction. A6 removes those directions
from the gradient instead of skipping or rolling back the steps.

## What it is not

- Not a change to §2.2–§2.6: predict-before-encode, ε as a cell input, the units, the graph, the
  convex tanh update, the router, `q` and `D([q; u])` all run as before.
- Not a guard: no step is skipped, no document dropped, nothing rolled back.
- Not truncation in time: the carry is followed over the whole window.

## Cost and risks

- The gradient is biased: it ignores how an earlier state changes a later write. A strategy that
  needs that credit (counting, "do not overwrite") trains only through the direct reads.
- `eps` leaves the predictor trained by `L_pred` alone; with `train.lambda_p = 0` and `eps` cut it
  would receive no gradient.
- Forward sensitivity is untouched: a hard-switching cell stays sensitive to its state.

## Configuration

    train.feedback_detach: []                               # default: full backpropagation
    train.feedback_detach: [summary, eps, state, router]    # the carry alone

The key is left out of the configuration hash at its default (existing checkpoints resume and
existing arm ids are unchanged); a non-empty list hashes, and the trainer refuses a model built
with other paths. HSN only.
