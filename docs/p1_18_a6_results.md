# A6 results — the A5 accuracy drops, the GPU pauses and the GRU bar (10 Nov 2026)

Branch `claude/a5-fix` (from `codex/a5-norm` at `38123a1`). Every score comes from the canonical
SQLite database `/home/joe/hsn-experiments/experiments.sqlite`; the replay evidence, probe scripts
and logs are in `/home/joe/hsn-experiments/a5-fix/`. Runs are 9,000 steps of the A5 recipe (top-K
routing, key addresses, selected-cell readout, lr 2.5e-3, β2 0.95, WSD, float64 gradient norms) on
the delayed task, B32, compiled, unguarded: no spike guard, no rollback, every document kept.

RESULTS_SUMMARY

## 1. What the drops are

`docs/p1_17_training_findings_2026_11_10.md` left the cause open ("capture the onset ... target the
recurrent computation and routing"). I replayed training windows on the stored checkpoints of exps
345 (current-norm caps, before its collapse) and 346 (anchored caps, inside its 36.9-point drop),
weights frozen, and repeated the backward pass of the exploding windows with one path cut at a
time (`a5-fix/probe/probe_paths.py`, `p345_5000.json`, `p346_6000.json`; forward values identical
across variants).

| Cut in the backward pass | 346 @ 6,000, worst window | 345 @ 5,000, worst window |
| --- | ---: | ---: |
| nothing (the training gradient) | 159.2 | 29.0 |
| the selection's mask (straight-through) | 158.2 | 29.2 |
| the selection's unit weights | 159.6 | 29.0 |
| every scorer feature (`u`, `ε`, `h`) | 158.2 | 29.2 |
| the gate | 105.6 | 28.1 |
| `q → P` (the predictor's read of the summary) | 65.7 | 28.4 |
| `q → S` (the shared transform's read) | 37.5 | 4.7 |
| both reads of `q` | 6.9 | 4.7 |
| the cell's read of `u` and `ε` | 13.6 | 22.6 |
| the cell's read of its own `h` and the messages | 2.5 | 29.5 |
| an ordinary window, nothing cut | 2 to 3 | 1.8 |

What this shows:

- **Not the router's estimator.** Detaching the straight-through mask, the unit weights or all
  scorer features leaves an exploding window's gradient unchanged. The paths that matter are the
  *feedback reads* of the recurrent state: the summary `q` read by `P` (through `ε`) and by `S`
  (through `u`), and a cell's read of its own state and its neighbours' messages.
- **Jumps, not growth.** The state gradient is flat along the window and jumps 7× to 22× at single
  positions (346, window 67: 0.59 for every position after 14, 13 at position 13, 27 before
  position 9). Several jumps in one document multiply.
- **A cell of hard switches.** From step 1,500 on, 95 to 99 % of the candidate's coordinates are
  saturated (`|c| > 0.99`) and the gate is either closed or fully open (1 to 2 % open, mean 0.015).
  Such a cell passes almost no gradient anywhere and a very large one where a coordinate is in
  transition. The forward dynamics are just as sharp: a 1e-3 perturbation of the state at position
  32 is a different state 64 to 96 positions later (gain 40 to 1,000), at healthy checkpoints too
  (`prof346_*.json`).
- **One document takes the step.** The exploding document's gradient is 10 to 10⁹ times the
  others'. The global clip rescales the batch by that norm, so the other 31 documents vanish from
  the step and AdamW moves at full size along one document's switching direction. A burst is some
  hundreds of such steps (346, steps 5,750 to 6,250: 10 % of steps above 350).
- **A second mechanism, in the router.** Without the update caps the router's weights grow
  without bound and its routes churn (345: norm 0.7 → 16 before the collapse; exp 355 and 356
  below: 15 and 19 by step 4,500). Top-K always writes K cells; most positions are filler, where a
  write can only hurt, so the selected cells receive a steady "should not have been selected"
  gradient and the router keeps moving away from whatever feature currently picks them. The caps
  hold this back (they cut the router's step about tenfold and the cell's about twofold: 346's
  proposed update ratios are 0.027 and 0.0075 against caps of 0.003).
- **A4 is not the answer here.** With the candidate norm the candidate is not saturated, and the
  same feedback loops compound smoothly instead of in jumps: exp 350 (A4, full backpropagation, no
  caps) reached gradients of 10³⁶ to 10³⁸ and non-finite values at step 1,479.

## 2. What was changed

All opt-in, left out of the configuration hash at the defaults, forward values unchanged; the
equations of §2.2–§2.6 run as before (`docs/amendments/A6_feedback_gradient.md`).

- **`train.feedback_detach`** — the named feedback reads are constants in the backward pass. With
  `[summary, eps, state, router]` the state's backward recurrence is the gated carry `(1 − b) ⊙ h`
  alone, which cannot amplify (tested against the exact product of the gates). The memory path of
  this task (write from `u`, hold by the carry, read by the decoder) keeps its full gradient.
- **`train.state_grad_ratio`** / **`train.state_grad_clip`** — every path kept; at each position a
  document's gradient into the state is limited to a multiple of the batch's typical document
  there (or to an absolute number). A healthy window is plain backpropagation through time.
- **`model.address_init`** — the start value of the scorer's two address columns (A5 extension).

Framework: the diagnostics' state window and the end-of-run probes replay compiled graphs instead
of the eager forward; records carry `phases`; `scripts/gpu_phase_report.py` attributes the card's
time series (section 4).

RESULTS_TABLES

RESULTS_GPU

RESULTS_OPEN
