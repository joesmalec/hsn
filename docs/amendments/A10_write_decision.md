# Amendment A10: the router's write decision, with a cost (experimental)

Status: **implemented as an opt-in experiment on 13 Nov 2026; not adopted; no result yet.**
- Asked for by the user on 13 Nov 2026 ("build now as opt-in"), after the overnight runs of the
  white paper v1.2 review. It is decision (e) of `ARCH-READING-2026-11-13.md`.
- `model.write_gate` (default false), `model.write_init` (default 2.0) and `train.lambda_w`
  (default 0). All three are left out of the configuration hash at their defaults: existing
  configurations, hashes and arm ids are unchanged, and a model without the key is the model as
  before, bit for bit.
- It amends plan §2.3 (the cell update) by one factor on the gate and §2.2 step 4 (the scorer)
  by a second output. It changes no metric, threshold or task.

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

## The smallest test, and its kill rule

All with the state gradient, 9,000 steps:
1. **Pair task, 12 slots** (`g1_k12_cutC`: 89.7 / 78.4 %, key-value 86.4 / 78.4 %): the same
   arm with `model.write_gate=true` and `train.lambda_w` in {0, 0.02, 0.1}.
2. **Pair task, 8 slots** (`g1_a8_cutC`: 95.4 / 93.5 %): `write_gate`, `lambda_w=0.02`. It must
   not cost anything where the base is already good.
3. **Delayed recall, 12 slots** (`k12_cutC`: 98.5 / 99.1 / 95.5 %): the same check.
4. **Story text from scratch, 8 fast + 12 routed** (`tf128_f8ru20_cutC`: 1.9 %, chance): the
   same arm with `write_gate` and `lambda_w` in {0.02, 0.1}. This is the arm the cause above
   predicts most directly: with the text supervised, writing text into the routed cells is
   rewarded at every position and nothing a fact needs survives until its query; a cost on
   routed writes should move the text to the fast cells.
5. **Story text, warm start** (`tf128w_f8ru20_cutC`: 99 % at step 500, 89.2 % at 9,000) with
   `write_gate` and `lambda_w=0.02`: does the cost stop the erosion? `train.init_from` accepts
   a source checkpoint without the write output (it starts open).

**It works** if arm 1 with a cost reaches the 8-slot key-value score (92 % or more) with fewer
writes per position than `g1_k12_cutC`, and arms 2 and 3 stay within their seeds' spread. Arm 4
is a result of its own: any score clearly above chance from scratch is one no arm has reached.
**Kill rule:** if arm 1 is not above `g1_k12_cutC` at any cost, the write decision is not what
limits the key-value task at 12 slots, and the amendment is dropped. If only the arm with
`lambda_w = 0` gains, the cost is dropped and the decision kept.

Read from the run records: `write/permitted` (the share of routed slots with `w = 1` on the dev
window) and `gate/write_rate` (which now counts a slot with `w = 0` as closed).

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
