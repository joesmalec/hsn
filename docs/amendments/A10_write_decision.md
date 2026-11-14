# Amendment A10: the router's write decision, with a cost (experimental)

Status: **an opt-in experiment; not adopted. Built and dropped on 13 Nov 2026, on main again
since 14 Nov 2026 for one question: can the memory be left untouched where it is not needed?**
- 13 Nov 2026: asked for by the user ("build now as opt-in"), tested the same day on the pair
  task and dropped by its own kill rule (the user: "Lets drop A10"). Its code was not merged
  and its keys were retired. That record is the section "13 Nov" below.
- 14 Nov 2026: taken up again as step 1 of "memory touched only when needed" (the user: "I
  think 1 makes sense as it is our weakest spot as well", then "lets complete step 1 I agree
  that's the right path"). The question is no longer whether it helps the pair task; it is
  how many routed writes text, arithmetic and recall need. The keys are schema keys again and
  the same code as on 13 Nov. Results in the section "14 Nov".
- `model.write_gate` (default false), `model.write_init` (default 2.0) and `train.lambda_w`
  (default 0). All three are left out of the configuration hash at their defaults: existing
  configurations, hashes and arm ids are unchanged, and a model without the key is the model as
  before, bit for bit.
- It amends plan §2.3 (the cell update) by one factor on the gate and §2.2 step 4 (the scorer)
  by a second output. It changes no metric, threshold or task.
- **It saves no compute, in training or in evaluation.** Every selected cell runs its unit
  whether its write is open or closed; a closed write gets its reopening gradient through the
  candidate that was computed. A share of closed writes is a measurement, not a time saved.

## Cause

**Selecting a cell does three things at once.** A selected cell is executed (compute), it is
read by the selected-cell readout, and it is written through its gate `b = σ(B·v)`.
- The gate reads `v = [u; ε; h_i; m_i]`. It does not see the address match that made the router
  choose the cell, and it is one matrix for all units.
- So where a fact is written, every selected cell is written. Measured on the pair task with the
  state gradient: about 0.55 writes per position at 8 slots, about 1 at 12, about 2.7 at 16
  (full cut). The key-value score falls in the same order: 93.9 / 91.7 %, 86.4 / 78.4 %,
  63.5 / 67.0 %.
- A fact that takes `K` of `N` cells shares cells with the other keys' facts. With eight keys
  and 32 cells a cell holds the facts of two keys at 8 slots, three at 12 and four at 16, and
  the last write wins.
- The same knob sets the compute per position. Delayed recall, arithmetic and text score better
  with 12 slots; the key-value task scores better with 8.
- On story text the routed cells' gates are open at most positions (`gate_slow/p50` 0.32 to
  0.92): text is written into the cells that hold the facts, and facts erode when the text is
  supervised (95 to 99 % at the start of a warm start, 84 to 89 % after 9,000 steps).

Caveat recorded with the finding: the route guard fired 27 and 23 times in the two 8-slot pair
runs and 1 and 0 times in the 12-slot ones, so the slot effect on the key-value score is
confounded with the guard. The controls are queued (64 cells, fast cells with few routed slots,
the pair task without the guard).

## The change (opt-in)

The scorer gets a second linear output of the same features, one **write logit** per
(cell, unit): `ω_ij = W_w · feats_i + b_w`. For a selected routed cell `i` with chosen unit `j`:

```
w_i     = 1 if ω_ij > 0 else 0
h_i'    = (1 − w_i·b_i) ⊙ h_i + w_i·b_i ⊙ c_i          (b_i, c_i as before)
```

- **Selected cells are executed and read as before.** The selection (top K by the unit scores),
  the unit choice, the candidate `c`, the gate `b`, the graph, the readout and `q` are unchanged.
  A cell with `w = 0` keeps its state: a read without a write.
- **The write decision sees what the router sees**: the address features `a_cur`, `a_prev` (and
  A9's), `u`, `ε`, `h_i`, the age. It can open the write for the best-matching cells of a fact
  and keep the other selected cells read-only, which the gate cannot do.
- **The update stays a convex combination**: `w·b ∈ [0, 1]`, so `h ∈ [−1, 1]` as before.
- **The fast cells of A8 are always written** (`w = 1`, a constant).
- **Start.** `W_w = 0`, `b_w = model.write_init = 2.0`: every selected cell is written, so at
  the start the model computes exactly what it computes without the key
  (`test_open_write_gate_is_the_model_without_it`). The output is built without a random draw:
  with the same seed every other parameter has the same initial value.
- **Training.** Straight-through, as for the cell mask: forward `w` is 0 or 1, backward it is
  `σ(ω/τ)`. Every selected cell is executed, so a cell that was not written still gets the
  gradient of "what if it had been written"; there is no position class that stops learning
  because nothing ran there. In the soft warm-up `w = σ(ω/τ)`. The gradient into `ω` comes
  through the written state, so the amendment needs the state gradient
  (`train.feedback_detach` without `state`); under the full cut `ω` sees the same position only.
- **The cost** (`train.lambda_w > 0`): `λ_w` times the share of the valid routed slots with
  `w = 1`, averaged over the window (`info["write_fraction"]`, differentiable through the
  straight-through `w`). It is a term of its own (`write`), not one of the P2.4 auxiliary terms.

**The objective as executed** (checked against the code on 14 Nov 2026 for the review's item
04a-1; `tests/test_a10_write_decision.py` computes each line by hand):

```
p_i     = sigmoid(ω_i / τ)                      τ = the selection temperature (1 in every run so far)
g_i     = 1[ω_i > 0]
w_i     = g_i + (p_i − stopgrad(p_i))           hard training; w_i = p_i in a soft warm-up; g_i at evaluation
R_write = mean over positions of mean over rows of  Σ_i mask_i w_i / max(1, Σ_i mask_i)
L_total = L_task + λ_p L_pred + λ_w R_write
```

- `i` runs over the selected slots of one row at one position; `mask_i` is 1 for a valid
  routed slot. Fast cells and the slots of a dense-fallback row are written always and are
  outside the share.
- `λ_w` is applied once. `R_write` is a mean of per-row ratios, not one ratio over all valid
  slots of the window; the two differ when rows have unequal valid counts. Padding positions
  are inside the mean.
- The same `w_i` multiplies the cell's gate and enters the cost.
- The task gradient to a write logit is `Σ b_i ⊙ (c_i − h_i) ⊙ ∂L/∂h'_i · p_i (1 − p_i) / τ`,
  whether the write is open or closed: the candidate of a closed write is computed, and that
  is where its reopening gradient comes from. The cost's gradient is `λ_w p_i (1 − p_i) / τ`
  over the valid routed slots' count, while its value is the hard share.
- `write/permitted` in a run record is `R_write` on the dev windows with `w = g`.

**What stays.** Cells, graph, units, ε, the convex tanh gate, the router and its linear scorer,
A5's address and `q`. No attention over the sequence, no new state, nothing learned about any
key.

**What it does not do.** It does not reduce compute: a read-only cell still runs its unit. It
does not let the router select fewer than K cells. Whether fewer writes are enough, or the
selection itself has to become optional, is what the test decides.

**Parameters:** `(F + 1)·M` for the write output, `F` the scorer's feature count (1,624 for the
recipe with A5: `F = 202`, `M = 8`). **Cost per position:** one more linear map of the scorer's
features, `N·F·M` MACs, scorer work.

**Windows.** Eager and compiled autograd windows. The split and fused windows refuse the key.
Not defined for `train.gate_mode=reinforce`, the MLP scorer or `train.lambda_u > 0`.

## 13 Nov: the pair task, and the kill rule

State gradient, 9,000 steps, one seed per arm. Pair task, 12 slots, 32 cells (control
`g1_k12_cutC`: 89.7 / 78.4 % in range, 79.6 / 72.9 % out of range):

| Arm | Cost | In range | Key-value part | Out of range | Slots that may write |
| --- | --- | --- | --- | --- | --- |
| `g1_k12_wg_cutC` | 0 | 81.3 | 75.9 | 68.2 | 48 % |
| `g1_k12_wg02_cutC` | 0.02 | 76.9 | 71.0 | 54.6 | 46 % |
| `g1_k12_wg10_cutC` | 0.1 | 89.7 | 86.6 | 35.8 | 24 % |
| `g1_k12_wg50_cutC` | 0.5 | 65.1 | 59.1 | 51.1 | 12 % |

Pair task, 12 slots, 64 cells, position wrapped (A11), cost 0.1:
`g1_k12nw_wg10_cutC_cells64_poswrap` 78.5 % (key-value 70.8 %), against 98.0 / 98.6 / 98.3 %
(key-value 97.3 to 98.1 %) for the same setting without the write decision.

- The kill rule was: dropped if the 12-slot pair arm is not above its control at any cost.
  No arm was, the arm at 64 cells lost 20 points, and recall past the window was worse in
  every arm. The write rate did fall by half or more.
- What limited the pair task instead: the number of cells (64 cells give 98 % at 8 and at 12
  slots) and, past the training window, the scorer's position feature (A11).
- The arms with A8's fast cells never ran: the compiled window did not get past its first
  hard-gate step on the cards (next section).

## The start-up stall, and `hardware.graph_alias_index`

With `model.write_gate` and fast cells the compiled window sat at one busy core with the card
idle. The cause is not this amendment: PyTorch 2.15's CUDA-graph tree checks every output of a
newly recorded graph against every live recorded tensor
(`CUDAGraphNode._is_alias_of_live_recorded_tensor`), a walk whose cost grows with the number of
live outputs; with the write decision's extra per-position records it did not finish in the
time anyone waited. The same walk is part of every compiled autograd run's start.

`hardware.graph_alias_index` (opt-in, default false, left out of the hash at its default;
`hsn.train.compile.install_graph_alias_index`) replaces that walk by an index built once per
recording. It changes when a recording finishes, not what is computed. Measured on the
RTX 3060 (14 Nov): the candidate's recording between step 1 and step 2 takes 98.5 s without it
and 57 s with it; the candidate with the write decision never passes step 1 without it (cut at
180 s) and runs 30 steps in 137 s with it. Do not dump Python stacks into a process during that
recording (fault handler, SIGABRT): it ended a run with a segmentation fault.

## 14 Nov: how many routed writes do text, arithmetic and recall need?

The candidate (64 cells, 8 fast + 12 routed slots, state gradient, address gate reading the
summary, A9, A11) with the write decision, `hardware.graph_alias_index=true`, autograd window.
"Allowed" is `write/permitted`, the share of routed slots with an open write on the dev
window at the end. Twins are the same arms without the write decision.

| Data, steps | Cost | Routed writes allowed | Score | Twins |
| --- | --- | --- | --- | --- |
| Arithmetic, 9,000 (2 seeds) | 0.1 | 0 % / 0 % | 96.3 / 95.6 % | 86.4 / 86.4 % (6 to 11 % of routed slots written by the gate) |
| TinyStories, 9,000 (1 seed) | 0.1 | 2.4 % | 1.007 bits | 0.990 / 0.974 bits (62 % written) |
| `synthetic_hold`, 18,000 | 0.02 | 35 % | 95.3 % (peak 96.6 at 13,500) | 97.9 to 98.4 % (five seeds) |
| `synthetic_hold`, 18,000 (2 seeds) | 0.1 | 7.7 % / 9.6 % | 81.9 % (peak 92.6 at 15,000) / 90.4 % | the same |

By keys, fresh documents (`hold_eval.py`), final checkpoints:

| Run | kv: training keys | kv: dev / test keys | delayed: seen pairs / new pairs / never-seen bytes |
| --- | --- | --- | --- |
| cost 0.02 | 95.5 | 95.8 / 94.2 | 95.8 / 95.8 / 95.9 |
| cost 0.1, seed 0 | 95.5 | 82.5 / 84.6 | 91.2 / 91.8 / 80.9 |
| twins (five seeds) | 98.4 to 99.3 | 97.3 to 98.7 | 97.9 to 99.7 |

Step times on the rented RTX 5060 Ti with runs sharing a card: 0.192 to 0.201 s with the
write decision, 0.191 to 0.193 s for the twins measured the same way (the decision adds one
linear map of the scorer's features). Not a controlled timing.

Reading:
- **On arithmetic and text the routed memory can stay untouched.** The model closes every
  routed write on arithmetic and all but 2.4 % on text. Arithmetic is no worse (ten points
  above its twins, which are themselves below the old gate input's 90 to 100 %); text loses
  0.017 to 0.033 bits against two twins that differ by 0.016. One seed for text.
- **On recall a flat cost has a price that grows with the cost**: 3 points at 0.02, 8 to 16 at
  0.1 with a late fall. At 0.1 the loss is mostly on keys of bytes never seen in training (a
  gap of 11 to 13 points to the training keys); at 0.02 it is even.
- **So one cost for every position is the wrong form for mixed data.** The cost closes writes
  a fact needs as readily as the ones text does not need. The five-task mix with the cost (in
  progress when this was written) is the direct test.
- Nothing here measures speed. The twins' own gate already writes only 5 to 8 % of the
  routed slots on the recall data; what the write decision adds is a decision that could be
  taken before the unit runs, which no window uses yet.

## If it is taken further

- A cost that depends on what the position is (for example none where the address matches a
  stored key), or a floor under the writes the address selects.
- A training method in which a closed write is not computed needs another estimator than the
  straight-through one: the reopening gradient is `b ⊙ (c − h)` of the candidate.
- Write share does not give time saved: slot sparsity, touched positions and batch activity
  are different quantities (with independent 5 % activity per row, a batch of 32 has an active
  row at about 81 % of positions).

## Tests

`tests/test_a10_write_decision.py`: opt-in and hashing; scope (topk, linear scorer, no
reinforce, HSN only, split and fused windows refuse); only the write output is added and every
other parameter starts as before; the selection's decision is unchanged and the write mask
follows the chosen unit's write logit in the hard, straight-through and soft modes, with the
gradient at that logit only; fast cells always written; an open write gate is the model without
the key (evaluation and straight-through); a closed one writes nothing while cells are still
selected; the round writes exactly the permitted slots; the masked round equals the sparse
round; the window equals stepping; the task loss reaches the write output; the compiled window
equals the eager one; the cost term; thirty training steps; the cost lowers the write logits;
the diagnostics entry. `tests/test_p2_1_router.py` lists the two new fields of the selection
(a weight and a cost input).
