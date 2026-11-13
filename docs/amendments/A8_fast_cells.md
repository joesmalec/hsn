# Amendment A8: fast cells and slow cells (experimental)

Status: **implemented as an opt-in experiment on 12 Nov 2026; tested the same evening and not
supported: the fast group adds nothing over the same number of slots (section "Results"). Not
adopted.**
- The user asked on 12 Nov 2026 (about 19:40 MST) for a worktree with the architecture changes
  discussed in the review of white paper v1.2, and for their tests at the front of the queue.
- `model.fast_cells: F` (default 0) and `model.fast_readout: true` (default false). Both are
  left out of the configuration hash at their defaults: existing configurations, hashes, arm ids,
  state-dict keys and the state layout are unchanged.
- It amends plan §2.2 step 5 (which cells a round writes), and with `fast_readout` step 3 (the
  input of `S`), §2.4 (the decoder's input) and the `shared` and `decoder` rows of Appendix C.
  It changes no metric, threshold, task or data split.

## Cause

**The memory cells are also the model's only short-term context.**
- The input path is `u_t = S([z_t ; q_{t−1} ; ctx_t])`. What a position knows about the bytes
  before it reaches `S` only through `q`, the mean over all N cells, and reaches the decoder
  through `q` and the selected cells' mean `r`.
- A routed cell cannot count on being selected again at the next position, so a cell's own
  state is not a dependable place for the last few bytes either.
- On text the model therefore rewrites whatever cells it selects. Measured on TinyStories: the
  median gate of the selected cells is 0.63 to 0.99 at every position, against 0.000 on the
  delayed task, where the selected cells are left alone between a fact and its query.
- The two jobs compete: following a sentence needs cells that are overwritten at every byte,
  holding a fact needs cells that are not written at all. The best HSN reaches 1.037 bits per
  byte at 27,000 steps on TinyStories, the GRU 0.879 and the Transformer 0.827.

The white paper lists "differentiated timescales" in the design's biological footprint. The
model as built has one kind of cell and one selection rule for all of them.

## The change (opt-in; routed models with `model.routing=topk` only)

**`model.fast_cells = F`.** Cells `0 .. F−1` are the fast group. They are selected at every
position and take the first F slots, each at the unit the scorer ranks highest for it. The
router fills the other `k_select − F` slots from the remaining cells (the slow group), by the
same scores, in the same order as before.
- Nothing else about a fast cell differs. It runs the same unit bank, the same messages, the
  same convex tanh gate, and it has an address like every cell.
- The straight-through threshold is the midpoint between the lowest routed score and the highest
  unselected score, so it is taken among the slow cells. A fast cell's mask is the constant 1:
  it has no selection gradient, and it is written in full in the soft warm-up.
- Refresh probes and exploration never replace a fast cell's slot.
- An unselected slow cell carries its state exactly, as before.
- To keep the router's eight slots of the eight-slot arm, set `k_select = 8 + F`. The matched
  control is the same `k_select` without the key (`a_k12` for F = 4).

**`model.fast_readout = true`** (needs `fast_cells > 0` and `selected_readout`). The fast group
gets its own read, `f = mean_{i<F} h[i]`:

```
u_t      = S([z_t ; q_{t−1} ; ctx_t ; f_{t−1}])          f_{t−1}: a `summary` read under A6
logits_t = D([q_t ; u_t ; r_t ; f_t])
r_t      = mean of the routed (slow) slots' updated states only
```

Without the key the fast cells are read like any selected cell: through `r` (now the mean of
F + routed cells) and `q`.

**`model.fast_units = fixed`** (added 12 Nov 2026, 21:30 MST, after the first results; needs
`fast_cells > 0`; default `routed`). Fast cell `i` always runs unit `i mod M` instead of the
unit the scorer ranks highest for it, with a constant one-hot unit weight in every gate mode
(soft warm-up included).
- Why: in the first three finished arms (`f4r_cutC`, `ts_f4r`, `ts_f4r_cutC`) the four fast
  cells hold almost one vector: `rank_pr_fast/answer` is 1.1 to 1.25 of 4. They receive the
  same `u` and `ε`, are all written at every position, and settle on the same unit; the mean
  read `f` then carries one cell's worth of state. This is the cell symmetry the review of
  white paper v1.2 found for all cells, at its plainest.
- With fixed units the fast cells compute different functions of the same input, so they
  differ by construction. Nothing is added: no parameter, no identity vector, no new input.
  The slow cells and their routing are untouched, and the units stay shared (a slow cell may
  still be routed to a unit a fast cell runs).
- The fast cells' unit choice then has no gradient into the scorer.

**What stays.** The persistent cells, the fixed sparse graph, the unit bank, ε, the convex tanh
gate, the router, the addresses and `q` are unchanged. There is no attention over the sequence,
no new persistent state, no second kind of cell equation, and no weight written in the
computation path. The fast group is a rule about which cells a round writes.

**Parameters.** `fast_cells` alone: none. `fast_readout`: one more `d` on the first layers of
`S` and `D`, `2·d·H` = 65,536 for the recipe (1,853,465 → 1,919,001 at `k_select = 12`). The
new columns are drawn from their own generator, so every other parameter starts as before.

**Cost per position.** F more cells per round than the same number of routed slots would cost:
the round's cost follows `k_select`. The soft warm-up executes every cell and is unchanged.

**Windows.** Supported on the eager and the compiled autograd window (`hardware.window =
autograd`). The split and fused windows refuse the key (`split_window_scope_error`): their
replay and kernels do not know the reserved slots or the fast read. So an A8 arm runs at the
autograd window's step time.

**Refused:** dense, greedy, random and fixed routing; `train.gate_mode = reinforce`;
`train.lambda_u > 0` (the counterfactual probes swap slots); a disabled cell inside the fast
group; baselines; harness variants other than `pdg`. The router's cost profile is not
re-measured for `fast_readout` (the cost table is accounting only at `lambda_cost = 0`).

## What it is meant to show, and its kill rule

Arms on the eight-slot recipe (`k_select = 8 + F`, soft warm-up, executed credit), each against
the plain arm with the same `k_select`:

1. **Text improves.** TinyStories bits per byte below the same-K arm's, at 9,000 steps.
2. **Memory stays.** Delayed and the G1 pair within the seed spread of the same-K arm; the
   out-of-range evaluation of the delayed checkpoint still above 90 %.
3. **The slow cells act as memory on text.** The median gate of the *routed* cells on
   TinyStories falls well below the 0.63 to 0.99 measured today.

**Kill rule.** If neither 1 nor 3 shows on two seeds with `fast_readout` on, the short-term
context is not what holds text back, and the fast group is dropped. If 1 shows and 2 fails,
the group costs memory and is not adopted as it stands.

A fast group that learns is expected to need the state gradient (A6 without `state`:
`train.feedback_detach=[summary, eps, router]`), because under the full cut a cell is trained
only for what the decoder reads at the same position. Both settings are run.

## Results (12 Nov 2026, 22:25 MST; 9,000 steps): the kill rule applies

The same-slots controls with the state gradient came in after the first arms, and they
attribute the gain to twelve slots with the state gradient, not to the fast group.

| Task | Fast group (`F4R`) + state gradient | Twelve slots + state gradient, no fast group | Twelve slots, full cut |
| --- | --- | --- | --- |
| delayed | 99.2 / 99.2 % (later queries 98.4 / 98.5) | `k12_cutC` 98.5 % (later 97.6) | `a_k12` 90.8 / 81.3 % |
| TinyStories, bits per byte | 1.060 (full cut: 1.096) | `ts_k12_cutC` 1.054 | `ts_k12` 1.130 |
| arithmetic | 70.7 / 69.7 % | eight slots: 70.0 / 77.7 % | `arith_k12` 56.0 % |
| story text with facts (`tf`) | full cut: 2.0 % (chance) | pending | 1.7 / 2.7 % (chance) |

- **Point 1 (text improves over the same slots): not met.** 1.060 against 1.054.
- **Point 2 (memory stays): met**, and the out-of-range evaluation of `f4r_cutC` is 99.4 %; but
  the control is within 0.7 points, so this is the state gradient's doing.
- **Point 3 (the slow cells act as memory on text): not met.** `gate_slow/p50` is 0.49 to 0.53
  on TinyStories. The fast cells also hold almost one vector (`rank_pr_fast/answer` 1.1 to
  1.25 of 4).
- By the kill rule the fast group with routed units is dropped. The only open arms are the two
  with `model.fast_units=fixed` on TinyStories (`ts_f4ru_cutC`, `ts_f8ru_cutC`): if neither
  beats `ts_k12_cutC`, the amendment is closed.

**What the story-text set showed instead.** Twelve slots without a fast group hold facts across
real story text when the text is not supervised (`tfq_k12`: 92.5 % in range, 90.7 % with the
facts 272 to 768 bytes back) and lose all of them when it is (`tf_k12`: answers at chance, text
1.489 bits per byte; still at chance halfway with one text position in 128 supervised).
Predicting the text opens the routed cells' gates (median 0.5), and a fast group does not
stop that, because nothing makes writing a routed cell cost anything. A rule that lets the
router select no routed cell at a position, with a cost on selecting one, is the change this
points to. It is not part of this amendment.

## Tests

`tests/test_a8_fast_cells.py`: the keys are opt-in and hashed only when set; the selection rule
against a direct computation (slots, units, threshold, masks, gradients, probes); reserved = 0
equals the old selection bit for bit; the fast cells are written at every position in the soft
warm-up, with hard gates and in evaluation, and an unwritten cell keeps its state exactly;
window against stepping; the read-out; gradients into the new columns under three cut settings;
the compiled window against the eager one; the closed forms; the refusals; thirty training
steps of the small recipe.
