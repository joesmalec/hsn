# Amendment A11: a bounded position for the scorer (experimental)

Status: **implemented as an opt-in experiment on 13 Nov 2026; not adopted.** The evidence so far
is evaluation only (the same weights, the position changed at evaluation); see "Results".
- `model.scorer_position`: `document` (default), `clamp` or `wrap`. Left out of the
  configuration hash at its default: existing configurations, hashes and arm ids are unchanged.
- It amends the scorer's feature list (plan Appendix H.2, the `pos` scalar) for positions past
  the training window only. It changes no metric, threshold or task.

## Cause

The scorer reads the scalar `pos / unroll`: the position in the document over the length of the
training window (256).
- In training on documents that fit one window the feature stays below 1. In a document of 288
  to 1,024 positions it grows to 4, a range the scorer's linear map never saw.
- The loss past the window had the mark of it. In the long documents of the pair task, facts
  written early (a long dependency) were recalled at 85 to 93 %, facts written late at 53 to
  73 %: the opposite of what cells filling up would give. Delayed-recall documents state every
  fact near the start, so they showed little of it.
- The feature cannot simply be removed. Trained without it
  (`model.scorer_features.exclude=[pos]`, 64 cells) the pair task scores 60.6 % at 8 slots and
  84.2 % at 12 in range, against 98.1 % and 98.0 to 98.6 % with it.

## The change (opt-in)

```
document (default):  pos_feature = pos / unroll
clamp:               pos_feature = min(pos, unroll − 1) / unroll
wrap:                pos_feature = (pos mod unroll) / unroll
```

- **Inside the window nothing changes.** The three are equal while `pos < unroll`, so a model
  trained on documents that fit the window is the same model, bit for bit, with any of them
  (`test_same_model_inside_the_window_and_bounded_past_it`). The key matters only where a
  document is longer than the window: at evaluation past the window, and in training on long
  documents or running text.
- **`clamp`** reads every later position as "late in the document". **`wrap`** reads the
  position within the current stretch of `unroll` positions.
- Nothing else reads the position: with `wrap` the model's output deep inside a document equals
  its output at the wrapped position, exactly.

**What stays.** Everything but one scalar feature of the scorer: cells, graph, units, ε, the
convex tanh gate, the router, the address features and `q`. No parameter is added.

**Windows.** Eager and compiled autograd windows, and the split and fused windows (P2.10,
P2.10c) under the full cut and with the state gradient. Nothing in those windows changed: the
feature is one of the scorer's scalars, which the window's prologue computes
(`Scorer.window_scalars`) and the sweep and the replay read as they are; the sweeps keep the
position only as the state's counter. Checked on a window that crosses `pos = unroll`: the
split window gives the autograd window's decisions, logits, final state and parameter
gradients in float64 on the CPU (`tests/test_a11_position_bound.py`). On the card (RTX 3060,
14 Nov) the fused and folded sweeps record the split sweep's tape across the bound in fp32
(`tests/kernels/test_p2_10c_window.py`, the cases `fast_read_wrap` and `routed_clamp`: every
decision equal wherever the scores have a margin, 100 % and 96 % of the positions).

## Results (13 Nov 2026; evaluation only, CPU)

`/home/joe/hsn-handoff/eval_pos.py`: the last checkpoint of a run on the out-of-range dev
documents, the position the scorer sees left as trained or changed at evaluation. Per cent, all
answers (key-value / delayed):

| Run (trained in range only) | As trained | Clamped | Wrapped |
| --- | --- | --- | --- |
| `g1_a8nw_cutC_cells64` s0 (pair, 8 slots, 64 cells) | 78.2 (72.8 / 97.6) | 94.4 (93.1 / 99.0) | 95.4 (94.1 / 99.7) |
| `g1_k12nw_cutC_cells64` s0 (pair, 12 slots, 64 cells) | 65.7 (57.9 / 93.9) | 92.8 (90.9 / 99.6) | 92.8 (90.8 / 99.8) |
| `k12_cutC_27k` s0 (delayed, 12 slots, 27k steps) | 93.9 | | 99.1 |
| `g1_a8_cutC` s0 (pair, 8 slots, 32 cells) | 65.6 (57.2 / 95.6) | | 84.0 (79.9 / 98.7) |
| `g1_a8_cutC_27k` s0 (pair, 8 slots, 32 cells, 27k) | 55.5 (44.9 / 93.2) | | 83.1 (78.5 / 99.5) |
| `g1_k12_cutC` s0 (pair, 12 slots, 32 cells) | 79.6 (74.6 / 97.6) | | 80.2 (75.2 / 97.9) |

- The drop of delayed recall past the window after 27k steps is the position feature: wrapped,
  the out-of-range score equals the in-range score (99.0 %).
- At 64 cells the pair task's loss past the window is almost all the position feature.
- At 32 cells a second cause remains (key-value 75 to 80 % with the position wrapped; the
  12-slot run does not react to the position at all): the number of cells. 64 cells give 91 to
  94 %.

## What is not tested yet, and the kill rule

1. **Training where the position passes the window**: TinyStories (stories longer than a
   window) and training on both length variants. There the unbounded position is in the
   training distribution, and a bound removes information the scorer may use. Arms: TinyStories
   with `clamp` and with `wrap` against `ts_k12_cutC` (1.054 / 1.067 bits per byte).
2. **Longer training of the candidate base** (64 cells, 12 slots): in range and past the window
   at 27k steps.

**It works** if the TinyStories arms are within the seeds' spread of the control and the 27k
arms hold past the window at their in-range level. **Kill rule:** if a bounded position costs
more than 0.02 bits per byte on TinyStories with both `clamp` and `wrap`, the key stays an
evaluation setting for document tasks and is not proposed for the base.

## Tests

`tests/test_a11_position_bound.py`: opt-in and hashing; the split and fused windows refuse the
key; the feature's values in the three modes; the same model inside the window, a different one
past it, and with `wrap` exactly the output at the wrapped position; the window against
stepping across the bound; the compiled window against the eager one across the bound.
