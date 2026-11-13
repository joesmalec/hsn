# Amendment A9: an address for keys of several bytes (experimental)

Status: **implemented as an opt-in experiment on 12 Nov 2026; not adopted, no result claimed.**
- Asked for by the user on 12 Nov 2026 with A8 (the worktree for the architecture changes of the
  white paper v1.2 review). It is the first step of the "learned address" idea of that review,
  in the form A5's own analysis allows (below).
- `model.address_ngram: n` (default 1; 2 or 3), needs `model.key_address`. Left out of the
  configuration hash at its default: existing configurations, hashes, arm ids, the A5 buffers
  and the A5 scorer columns are unchanged.
- It amends plan §2.2 step 4 (the router's features) and the scorer row of Appendix C. It
  changes no metric, threshold or existing task. The delayed generator gains `key_len`
  (default 1: synthetic_v1 symbol for symbol), and `configs/data/synthetic_keys.yaml` defines
  the tasks that test it.

## Cause

**A5's address is one byte wide.**
- A5 gives the scorer two features per cell: `a_cur = √d·⟨C[x_t], A_i⟩` and
  `a_prev = √d·⟨C[x_{t−1}], A_i⟩`. At a fact `k v` the value's position has `x_{t−1} = k`, at the
  query `QRY k` the key's position has `x_t = k`: one number at the write and at the read.
- With a key of two bytes `k_1 k_2` the value's position sees only `k_2` in `a_prev`. Two facts
  whose keys end in the same byte get the same write address, whatever the model learns.
- Names in real records (a drug, a patient, a test) are several bytes long and share bytes.

**Why not a learned key.** A5's analysis (`A5_key_address.md`, "Cause") applies unchanged: a
key that generalises to names never seen in training needs an identity whose distribution is
the same for seen and unseen keys, and a similarity without learned weights. A learned
embedding of a key meets an unseen key out of distribution. So the address of a longer key is
built the way A5 builds the address of a byte: from fixed codes.

## The change (opt-in; needs `model.key_address`)

For every `m` in `2 .. n` the scorer gets two more features per cell, appended after `a_prev`
(`a_cur2, a_prev2, a_cur3, a_prev3`; the layout grows from `3d + 10` to `3d + 8 + 2n`):

```
code_m(b_1 .. b_m) = d^((m−1)/2) · C_1[b_1] ⊙ C_2[b_2] ⊙ … ⊙ C_m[b_m]        (elementwise)
a_cur<m>[i]  = √d · ⟨code_m(x_{t−m+1} .. x_t),   A_i⟩
a_prev<m>[i] = √d · ⟨code_m(x_{t−m}   .. x_{t−1}), A_i⟩
```

- `C_1` is A5's byte-code table; `C_2 .. C_n` are new fixed tables drawn the same way, after
  everything A5 draws, so A5's buffers and columns are those of `address_ngram = 1`.
- **One address for a key.** At a fact `k_1 .. k_m v` the value's position has the key in the
  `m` bytes before it; at the query the key's last byte has it in the last `m` bytes. So
  `a_prev<m>` at the write equals `a_cur<m>` at the read, for any key, seen or unseen.
- **Keys that share bytes have unrelated addresses.** The code is a product, not a sum: the
  codes of `(a, b)`, `(a, c)` and `(b, a)` are uncorrelated. The scale factor gives every
  feature A5's variance.
- **What is learned.** As in A5: which feature a unit follows at which class of byte. A model
  whose task has keys of two bytes can let its writer units follow `a_prev2` and its reader
  unit `a_cur2`. No meaning of a key is learned, and the length of a key is chosen by the
  units, not fixed by the configuration (up to `n`).
- **Older bytes.** A window takes `x_{t−2} .. x_{t−n}` from its own tokens, BOS before a
  document's start. The trainer carries one byte across windows (A5's `prev_tokens`), so at
  the first `n − 1` positions of a window that continues a document the bytes further back read
  as BOS. A document that starts at a window boundary (`data.align_docs`, in-range tasks) is
  exact. `HSN.step` takes `prev_tokens` as `[B, n]` for exact features; `[B]` means BOS for the
  older bytes.

**What stays.** Cells, graph, units, ε, the convex tanh gate, the router and its linear scorer,
A5's features and `q` are unchanged. No attention over the sequence, no new state, nothing
learned about any key.

**What it does not do.** A key longer than `n` bytes, or of unknown length in running text,
needs a running code with a learned start and end. That needs a new field in the state and is
not part of this amendment.

**Parameters:** `2·(n−1)·M` scorer weights (16 for `n = 2`). **Cost per position:**
`2·(n−1)·N·d` MACs for the dot products (4,096 for `n = 2`), scorer work.

**Windows.** Eager and compiled autograd windows. The split and fused windows refuse the key:
their scorer layout has A5's two address columns.

## The tasks (`configs/data/synthetic_keys.yaml`)

`delayed2` and `delayed3`: synthetic_v1's delayed task with keys of two and three bytes. The
keys of one document are cells of a small grid of key bytes (`@g bA b_ Mg M_ MA`), so they
share bytes: in three documents of four, two keys end in the same byte. Dev and test keys use
bytes never seen in training, as in synthetic_v1. `delayed2` also has an out-of-range variant
for evaluation.

## The smallest test, and its kill rule

On `delayed2`, 9,000 steps, the eight-slot recipe:
1. the recipe as is (A5's one-byte address): expected to stay near the level of a reader that
   knows only a key's last byte;
2. the recipe with `model.address_ngram=2`.

**It works** if arm 2 is clearly above arm 1 on two seeds and near the eight-slot arm's score
on synthetic_v1's delayed task (91 %). **Kill rule:** if arm 2 is not above arm 1, the scorer
does not pick up the longer address by itself, and an oracle run (the address columns of the
writer and reader units set by hand) decides whether the mechanism or its learning fails.

## Tests

`tests/test_a9_ngram_address.py`: opt-in and hashing; A5's buffers, columns and every other
parameter unchanged; the features against a direct computation; one address at a fact's value
and at its query for two and three bytes, unrelated addresses for keys sharing a byte or in
the other order; closed forms; the window against stepping with the history; the limit at a
window's start; the compiled window against the eager one; thirty training steps; the
generator's `key_len` (the reference solver agrees, keys are distinct and share bytes, and
`key_len = 1` is synthetic_v1).
