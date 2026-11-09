# Amendment A5: key-address features and a readout of the selected cells (draft)

Status: **draft of 8 Nov 2026 (P1.16), not adopted.**
- Nothing in the code implements it.
- Every configuration, result and gate runs without it.
- The user approved this design for a follow-up chunk, P1.17 ("a router that learns" plus A5). That
  chunk's plan proposes the configuration keys and the user approves them there.
- It amends plan §2.2 step 4 (the router's features), §2.4 (the decoder's input) and the scorer and
  decoder rows of Appendix C. It changes no metric, threshold, task, data split or key alphabet.

## Cause

**No model binds keys.**
- Every model trained on delayed/in_range sits at or under the task's key-blind ceiling: 0.800
  overall, 0.623 on later queries (`evidence/EFF/ceiling_check.py`, branch `hsn-efficiency`).
- Probe v3's key swap reads "does not bind keys" on every stored run: dense HSN, top-K HSN, gru_2m
  and the Transformer. Swapping two facts' keys leaves 99 to 100 % of the answers unchanged.
- The dev split's keys, bytes 112 to 127, never occur in training, which has keys 64 to 111
  (`DelayedParams.keys`). So binding them needs a comparison: the key of a query matched against
  the stored keys. A learned meaning per key cannot do it.

**What the dev keys look like after training** (measured 8 Nov 2026 on the stored checkpoints):
- **Decay.** AdamW decays every embedding row each step, because the embedding gets a dense
  gradient whose rows for unused bytes are zero. Adam's moments for those rows stay 0, so their
  rows shrink by exp(−wd·Σlr): 0.857 over 9,000 WSD steps at lr 2e-3, and 0.825 at 2.5e-3.
- **HSN.** `byte_emb` starts at std 1/8 (`state_init=fan_in`), a row norm of 0.879.
  - After training, the dev keys' median norm is 0.759 in exp 194 (dense, 2e-3) and 0.731 in
    exp 196 (top-K, 2.5e-3), as predicted. The trained keys' is 0.753 and 0.451.
  - So in HSN the dev keys are as long as the trained ones. They differ in direction:
    - the dev keys are random directions (pairwise |cos| 0.094, the initial value);
    - the trained keys share a learned common direction (mean cos 0.22 and 0.30, max 0.53 and
      0.73).
- **gru_2m.** `tok_emb` starts at N(0, 0.02²) and is 384 wide. Its dev keys end at 0.383, its
  trained keys at 1.05: 2.7 times longer.
- **So anything learned from key embeddings meets dev keys out of distribution.** A comparison that
  generalises to unseen keys needs two things:
  - identity from a code whose distribution is the same for every byte;
  - a similarity without learned weights.

**No model binds even the keys it was trained on** (P1.16, `probe-rerun --key-alphabet train`, a diagnostic
that remaps the dev keys 112 to 127 to the trained keys 64 to 79 in memory, 8 Nov).
- Across 9 stored runs (dense and top-K HSN, gru_2m, the Transformer), the swap follow is 0.003 to 0.027, and the
  dev score is unchanged by the remap.
- `route-check` on the five top-K runs finds the selected cells following the query's position, not its key, on
  the dev keys and on the trained keys alike.
- So the first obstacle is learning to bind at all; unseen keys are the second. A5's codes address the second.
  Its smallest test must therefore also show binding on trained keys (the remapped probe) before it is read as a
  failure of generalisation.

**The router cannot route by key as built** (`python/hsn/model/router.py`).
- The default scorer is linear over each cell's `[u_t ; ε_t ; h_i ; 8 scalars]`, and `u_t` and
  `ε_t` are shared by every cell. So the current byte shifts every cell's score for a unit equally,
  and can change which cells are chosen only through the per-unit maximum.
- **The two-layer scorer was tested** (`model.scorer=mlp`, exp 196's top-K recipe, exp 202). It
  trains and reads state, but does not bind keys (swap follow 0.010). Its final dev is 0.433 and its
  step 0.1157 s (+8.6 %).
- **The router barely learns.** Its gradient share has a median of 6e-6 to 5e-5 in exps 191,
  196 to 198 and 202. Its update ratio, 4.9e-4 per step in exp 202 against the cell's 8.5e-4, is
  Adam rescaling a tiny gradient. Three causes are identified (8 Nov):
  1. **STE through few cells.** The straight-through gradient reaches the cell scores only through
     `soft = sigmoid((s − θ)/τ)`, so only cells whose score lies near the threshold θ get a useful
     gradient.
  2. **The shared input in the linear scorer.** `u_t` and `ε_t` are the same for every cell, so the
     input cannot change the cell ranking except through the per-unit maximum.
  3. **Adam's rescaling.** Adam rescales a gradient of about 1e-5 into updates of normal size, which
     are then mostly noise.
- **The audit's gradient probe** (`evidence/EFF/router_probe/router_grad_probe.md`, branch
  `hsn-efficiency`, 8 Nov) refines the first cause:
  - the selected cells' STE margins are near 0 (σ′ about 0.25), so they are not saturated;
  - but dL/dα at the selected cells is about 0 (median 0), because top-K cells barely write (a write
    rate of 0.2 to 2 %);
  - unselected cells get exactly 0;
  - the router's first-layer gradient comes mostly from `u`, hardly from `h`.
  - The audit's reading: the router needs an objective that does not pass through α (the
    counterfactual utility of P2.3), or a design where the selection decides the write. A
    temperature alone does not help.
- These are what P1.17's "a router that learns" addresses. A5 depends on it: A5's address features
  reach the selection only through the router's training. The selected-cell readout gives the
  selection a direct path to the loss (the decoder reads the chosen cells), which is one form of
  "selection decides the write".

## The change (opt-in; routed models only)

**Fixed codes** (buffers, never trained, drawn from a generator seeded by `run.seed` and a fixed
tag, so every other parameter's initialisation is unchanged):
- **byte codes:** `C ∈ R^{V × d_c}`, each row i.i.d. N(0, I / d_c), with `d_c = d`;
- **cell addresses:** `A ∈ R^{N × d_c}`, drawn the same way.

**Key-address features.** Two scalar scorer features per cell, appended after `one`, so the layout
grows from 3d + 8 to 3d + 10:

```
a_cur[i]  = √d_c · ⟨C[x_t],     A_i⟩
a_prev[i] = √d_c · ⟨C[x_{t−1}], A_i⟩          x_{t−1} = BOS at a reset
```

- **The same numbers for one key.** At a fact `k v`, the value's position has `x_{t−1} = k`; at the
  query `QRY k`, the key's position has `x_t = k`. So `a_prev` at the write equals `a_cur` at the
  read. A scorer that learns when to use which feature routes both to the cells best aligned with
  `C[k]`, for any key: seen or unseen, train and dev codes come from one distribution.
- **What is learned.** No meaning per key is learned. The scorer learns only from class information
  in `u_t` (a value, a key after `QRY`, filler) which feature to follow.
- **The previous byte across windows.** At a window start `x_{t−1}` comes from the trainer: the
  previous window's last token, carried by the trainer as an optional `prev_tokens` argument of
  `HSN.forward` (default None, meaning BOS). The state layout is unchanged.

**Selected-cell readout.**
- The decoder reads the cells the router chose: `D([q_t ; u_t ; r_t])`, with
  `r_t = (1/n_valid) Σ_{valid slots} h[i, t]`, the selected cells' updated states.
- Its first layer grows from 2d → H to 3d → H.

**What stays.** The persistent cells, the fixed sparse graph, the unit bank, ε, the convex tanh
gate, the router and q are unchanged.
- There is no attention over the sequence and no matrix-valued state. The readout reads the cells
  the router addressed, which is how HSN's memory is meant to be used.
- Dense routing selects every cell, where both parts are void; the configuration refuses them with
  `model.routing = dense`.

**Parameters** (A.1): scorer +2·M = 16 (linear) or +2·128 = 256 (mlp); decoder +d·H = 32,768; the
codes and addresses are buffers.

**Cost per position:**
- the address features: 2·N·d_c = 4,096 MACs;
- the readout: d·H = 32,768 MACs;
- together +2.8 % of the 1,318,080 sparse MACs of Appendix C.2.
- Predicted on compiled top-K (0.1065 s per step at exp 196's recipe): at most 0.112 s (+5 %).
  P1.17 measures it.

## Options considered (P1.16 plan, 8 Nov)

| Option | Why not first |
| --- | --- |
| The mlp scorer alone (no code) | Routes only through learned features of the key, so it meets unseen keys out of distribution; exp 202 does not bind (swap follow 0.010). |
| Cell identity (A3) | Breaks the dense symmetry, but gives no map from key to cell; with a code-addressed gate it is "dense A5" at the dense step cost (0.240 s, 2.3× top-K). |
| A key-conditioned readout over all cells | Needs a per-cell key slot written from the code (a state-layout change), and the value must still be written beside its key: the same write problem, solved less directly. |
| Fast weights (P3.4) | P3.4 is a learning-path residual (global rule 1: no weight writes in the computation path). A Hebbian matrix in the state would bind, but it is linear attention / a matrix-state SSM, so the architecture rule excludes it. |
| Routing by the current byte's code only | The value lands in the value's own cells, so nothing binds. |
| Exempting embeddings from weight decay | Closes gru_2m's 2.7× scale gap, but creates no shared structure for unseen keys. |

## The smallest test, and its kill rule

**The arm.** Exp 196's top-K recipe (k 4, STE, cand_norm none, lr 2.5e-3, WSD 0.2 over 9,000 steps,
B32, compiled) with both parts of A5, seeds 0 and 1. It uses the scorer that P1.17's router work
and P1.16's `route-check` recommend.

**Binding is shown** when either seed has both:
- probe v3's swap verdict "binds keys" (follow ≥ 0.5, unchanged ≤ 0.1);
- `route-check` on dev reading "follows the key".

Later-query accuracy above the key-blind 0.623 is the matching metric.

**Ablations:** address features alone; the readout alone (global rule 4: one feature at a time).

**Kill rule.** If swap follow is below 0.2 and route-check reads "does not follow the key" on both
seeds, an **oracle-gate ablation** runs before any further design. The two features are switched by
a fixed rule on the byte class: `a_prev` at a value, `a_cur` at a key after `QRY`, neither at
filler.
- If the oracle binds, the mechanism has the capacity and the router's learning is the obstacle
  (P1.17's other half).
- If it does not, storage or readout fails.

## What it applies to

The routed HSN on the delayed tasks, as an opt-in amendment tested by P1.17. Nothing is adopted
unless its arms bind keys and the user approves.
