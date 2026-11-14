# Amendment A12: address weights that depend on the position's content (experimental)

Status: **implemented as an opt-in experiment on 13 Nov 2026; not adopted; no result yet.**
- Asked for by the user on 13 Nov 2026 ("lets make it an opt in amendement and test"), after the
  27k runs of the pair task lost what they had learned by 10,000 steps.
- `model.address_gate` (default false; needs `model.key_address` and the linear scorer). Left
  out of the configuration hash at its default: existing configurations, hashes and arm ids are
  unchanged, and a model without the key is the model as before, bit for bit.
- It amends plan §2.2 step 4 (the scorer) by one term. It changes no metric, threshold or task.

## Cause

**What was measured.** Pair task (key-value + delayed), 64 cells, state gradient, 27,000 steps:
the runs peak at 98 to 99 % and end at 93.9 % (12 slots), 90.0 % (8 slots) and 95.6 % (no route
guard). The training loss rises with the dev loss and the loss is the same on keys seen in
training and on new keys, so it is not overfitting. At the positions where a fact is written,
the selected cells move from the cells the key addresses to the cells the value addresses
(`/home/joe/hsn-handoff/unit_shares.py`, in-range kv dev documents):

| Model | Written under the key's address | Under the value's address | Read under the key's address |
| --- | --- | --- | --- |
| 12 slots, 9k | 76.0 % | 26.6 % | 86.2 % |
| 12 slots, 27k | 46.6 % | 59.6 % | 80.7 % |
| 8 slots, 9k | 65.0 % | 38.4 % | 86.7 % |
| 8 slots, 27k | 36.8 % | 69.3 % | 87.6 % |
| 12 slots, 27k, no route guard | 40.6 % | 66.4 % | 73.5 % |

The reads stay with the key. Fewer copies of a fact sit where its query looks, and values land
in cells that hold other keys' facts: wrong answers that are another key's value go from 32 %
of the errors at 9k to 63 to 67 % at 27k.

**Why the scorer drifts there (a reading, not measured).** The scorer is linear in its
features. A unit's weight on `a_cur` (the address of the current byte) is one number, the same
at every position. The unit that reads a queried key needs it large, and A6's route-guard text
records that this weight "keeps growing ... for a whole run". At a value's position the current
byte is the value, so the same weight pulls slots to the value's cells. The only thing a linear
scorer can set against that is an additive term from `u`, which does not change which cells an
address feature prefers. The route guard slows the growth and does not remove the conflict;
with a lower floor it fails another way (training keys 98.8 %, new keys 80 to 83 %).

## The change (opt-in; needs `model.key_address`)

One new matrix `G` (`d → M · A`, no bias, `A` the number of address features: 2 with A5, 4 with
A9's two-byte address). The logit of cell `i` and unit `j` at position `t` gains

```
logit[i, j] += Σ_a (G u_t)[j, a] · address_a[i]
```

so the weight unit `j` puts on address feature `a` is `W[j, a] + (G u_t)[j, a]`: its own
constant weight plus a part that depends on the content `u_t` of the position.

- **What it makes expressible.** A unit can follow the current byte's address where the byte is
  a queried key and not where it is a value, or follow the previous byte's address only at a
  value. A5's text names this as what is learned ("which feature a unit follows at which class
  of byte"); with constant weights it could only be approximated through which unit wins.
- **The addresses are unchanged.** `a_cur` and `a_prev` are still inner products of A5's fixed
  byte codes with the fixed cell addresses. Nothing about any key is learned; a key never seen
  in training has the same kind of address as a seen one.
- **Start.** `G = 0`: the model computes exactly what it computes without the key, in training
  and in evaluation (`test_zero_gate_is_the_model_without_it`). It is built without consuming a
  random draw, so with the same seed every other parameter has the same initial value.
- **`u` as the scorer reads it.** Under `train.feedback_detach` with `router`, the scorer's `u`
  is a constant in the backward, so `G` is trained by the selection's straight-through
  gradient like the scorer's other weights and no new gradient path into the shared transform
  appears.

**What stays.** Cells, graph, units, ε, the convex tanh gate, the selection rule, A5's and A9's
address features and `q`. No attention over the sequence, no new state.

**Parameters:** `d · M · A` (1,024 for the recipe with A5; 2,048 with the two-byte address).
**Cost per position:** one `d → M·A` map and `N · M · A` multiply-adds, scorer work.

**Windows.** Eager and compiled autograd windows. On the split and fused windows (P2.10,
P2.10c) the key is still refused for a training run, and half of the work is done:
- *Done (13 Nov, CPU, float64).* The split window's replay adds the gate's term to the
  scorer's logits (`hsn.train.split._scorer_logits`), with `u` the constant the router cut
  makes it, so `G` gets its gradient from the replay's graph like the scorer's other weights;
  the eager sweep runs the model's own scorer. With `SplitWindow(..., address_gate=True)` the
  window gives the autograd window's decisions, logits, final state and parameter gradients
  (the gate's among them) under the full cut, with the state gradient and with A8's fast
  cells, for A5's two address features and A9's four (`tests/test_a12_address_gate.py`).
  That is the reference for what follows.
- *Open.* The compiled sweeps (the sweep twin's merged scorer product in the fused and the
  folded block) do not add the gate. The term fits that product as `A · d` more input
  columns, `u_k · address_a[i]`, against `G` as more rows of the merged weight, so the
  selection kernels and their inputs stay as they are. Then the tapes against the split
  sweep's and the fused window against the autograd window, on the card. Until both, the
  configuration is refused the key on `hardware.window=split|fused`, and a split window
  over a compiled sweep is refused it whatever the caller asks.

## The smallest test, and its kill rule

Pair task, 64 cells, 12 slots, state gradient, no warm-up, seed 0:
1. **27,000 steps with the key** against the same run without it (98.0 % at step 10,500,
   93.9 % at the end; a second seed is running). The deciding arm.
2. **9,000 steps with the key**: it must not cost where the base is good (98.0 / 98.6 / 98.3 %).
3. **Delayed recall, 9,000 steps** (99.6 %): the same check.
4. **The combined shape** (8 fast + 12 routed slots), pair task, 9,000 steps (93.4 % without).

Read from each 27k checkpoint with `unit_shares.py`: the share of the write slots under the
key's address.

**It works** if arm 1 ends within 1.5 points of its own peak (the base gave up 4.1, the
no-guard run 2.3) with the write slots under the key's address clearly above the base's
46.6 %, and arms 2 and 3 stay within their seeds' spread. **Kill rule:** if arm 1 gives up as
much as the base, or arm 2 falls below 97 %, the amendment is dropped: either the scorer does
not use the gate to keep the writes with the key, or the cause is not the constant address
weight.

## Tests

`tests/test_a12_address_gate.py`: opt-in and hashing; scope (key address, linear scorer; the
split and fused windows refuse); only the gate is added, for A5 and for A9's layout, and every
other parameter starts as before; the logits against a direct computation and a zero gate as
the linear scorer; a unit following an address at one content and not at another; a zero gate
is the model without the key (evaluation and straight-through); the window against stepping
and the masked round against the sparse one; the task loss reaches the gate; the compiled
window against the eager one; thirty training steps.
