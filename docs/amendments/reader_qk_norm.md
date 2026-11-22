# Normalised queries and keys in the reader (HSN-T; opt-in key, 21 Nov 2026)

Status: **built on 21 Nov 2026 as an opt-in key, not yet run on the card.** Branch
`claude/reader-qk-norm` (on main d3626df). The user approved the amendment on 21 Nov 2026,
relayed by the campaign session (joe-3e): a normalisation of the reader's queries and keys that
bounds the attention logits, so that the unrotated top block can no longer sharpen without
limit. No amendment number.

- `model.reader_qk_norm` (default `false` = off): `true` for every reader block, or a list with
  one entry for each block, first to last, as `model.reader_rotary_pairs` takes one.
- At the default nothing changes: a block has no new parameter, runs the code it ran and gives
  the same outputs bit for bit; the key is left out of the configuration hash and of the
  experiment database's memory key at `false`, so every existing arm id stays.
- It changes no metric, threshold or task. It touches the reader only: everything after the
  reader's output `c_t` (the memory core, its state, the router, the losses) is unchanged.

## Cause

On the 27,000-step code-native runs (`fields_code_native`, word events, the top0 reader of
`hsn_t_memory_words_carry.yaml`: 160 wide, four blocks, rotary pairs `[20, 20, 20, 0]`) the
unrotated top block's weights grow, and the run gets worse at the peak learning rate. Between
9,000 and 27,000 steps:

| Weight | Top block (unrotated) | Rotated blocks |
| --- | --- | --- |
| `norm_attn.weight` (norm) | 41 → 69 | about 10, flat |
| `qkv.weight` (norm) | 84 → 115 | 32 → 48 |

- A block sharpens its attention by scaling its queries and keys up: a logit grows with the
  product of the two scales, and the softmax tends to a lookup. Measured, only the unrotated
  top block does it, the block that compares a query with a key as it is.
- Weight decay does not stop it. The trainer gives every tensor with fewer than two dimensions
  no weight decay (`Trainer._make_base_optimizer`, `build_lazy_adamw`), so the LayerNorm
  weight `norm_attn.weight` grows freely, and a larger LayerNorm weight scales the queries and
  keys as a larger `qkv` does.
- Rotating the top block is no answer for every field: on prose that unrotated block is a
  valuable copier of rare words, and rotating it costs 0.07 to 0.09 bits per byte there.

## The change (opt-in)

A reader block computes, from its input `x` (width `w`, `H` heads of `d = w / H` channels):

```
q, k, v = split(qkv(LayerNorm(x)))                       per head: [T, d]
q, k    = rotate(q), rotate(k)                           (the block's rotary pairs)
attn    = softmax(q kᵀ / sqrt(d) + mask) v
```

With the key on for that block:

```
q, k, v = split(qkv(LayerNorm(x)))
q       = g_q ⊙ q / sqrt(mean(q²) + ε)                   mean over the head's d channels
k       = g_k ⊙ k / sqrt(mean(k²) + ε)
q, k    = rotate(q), rotate(k)
attn    = softmax(q kᵀ / sqrt(d) + mask) v
```

- `g_q`, `g_k`: learned weights of `d` channels each, shared by the heads of the block, one for
  the queries and one for the keys, initialised to 1 (`nn.RMSNorm(d)` modules named
  `blocks.<i>.q_norm` and `blocks.<i>.k_norm`).
- `ε` is the eps of the block's LayerNorms (`1e-5`, PyTorch's default; the reader has no other
  RMSNorm).
- The values are not normalised, and the softmax scale stays `1 / sqrt(d)`.
- The norm is computed in float32 at least (under bf16 autocast the projection's output is
  bf16) and returned in the working dtype, so SDPA receives queries, keys and values of one
  dtype, as before, and the flash kernel still takes the causal reader.
- The new weights are made last in their block and without a random draw, so every other
  parameter of the model starts as it would without the key.

**The bound.** After the norm a query of a head has length at most `max|g_q| · sqrt(d)` and a key
at most `max|g_k| · sqrt(d)`; the rotation keeps the length of every pair. So every logit of an
enabled block is at most

```
|q · k| / sqrt(d)  ≤  max|g_q| · max|g_k| · sqrt(d)
```

whatever the `qkv` and LayerNorm weights are: 6.3 for the top0 reader's `d = 40` with the
weights at their start. Scaling `norm_attn` or `qkv` no longer sharpens the block. It can still
sharpen through `g_q` and `g_k`, which have no weight decay either (the trainer's rule for
every one-dimensional tensor; the user asked for them to be treated as the reader's other norm
weights). Their growth is then the thing to watch: 80 numbers per block, logged with the
weights.

**Cost.** `2 d` parameters per enabled block: 80 for the top0 reader's top block (the word-event
carry recipe: 3,298,025 → 3,298,105 trainable parameters with the key on the top block, 320 on
all four). Two normalisations of `[B, H, T, d]` per enabled block; under `torch.compile` they fuse
with the reshape before the rotation. Not timed yet.

## Every reader path

The norm sits in `_ReaderBlock._project`, which both block paths call, so every path the reader
has takes it in the same place:

| Path | What the key does |
| --- | --- |
| `Reader.forward` (the window alone), with or without `model.reader_span` | the enabled blocks' queries and keys normalised before the rotation |
| `model.reader_mask=causal` (`docs/reader_causal_flash.md`) | the same in `forward`; the mask is untouched |
| `model.reader_carry` (`Reader.read`, `docs/amendments/reader_carry_memory.md`; also HSN-T without its memory and the reader alone) | the cache holds each enabled block's **normalised**, un-rotated keys: the keys the same positions have in a pass without the cache. A cached key is never normalised again |
| `model.reader_top_words` | the top block's word slots hold its normalised keys, as the cache does |
| `train.reach_dropout` | unchanged: it narrows the mask only |

HSN-T has no incremental (byte-by-byte) path for the reader: `HSN.step`, which the chat
session (`python/hsn/chat/session.py`) feeds byte by byte, refuses a model with a reader, so
the key needs no step form.

A checkpoint written without the key does not load strictly into a model with it (two more
tensors per enabled block): `train.init_from` and resuming refuse it, as for any change of the
parameter set.

## Scope and refusals

Accepted wherever the reader exists: `model_name: hsn_t` with and without its memory and
`model_name: reader`, with every other reader key (`reader_span`, `reader_carry`,
`reader_top_words`, `reader_mask`, `reader_position`, `reader_rotary_*`), per-byte writes and
word events, the eager and the compiled windows.

Refused at configuration load, each with the reason:

| Configuration | Why |
| --- | --- |
| `model.reader_qk_norm` other than `false` with `model.reader=none` (the HSN, every baseline) | there is no reader attention to normalise (the same rule as for `model.reader_rotary_pairs`) |
| a list whose length is not `model.reader_layers` | one entry for each block, or one value for all of them |

A list of `false` only (`[false, false, false, false]`) computes what `false` computes but is
another configuration value, so another hash and arm id: spell the default `false`.

## How it was tested

`tests/test_reader_qk_norm.py` (CPU):
- the default: the configuration hashes and memory keys of ten configurations (the HSN, the
  memory-off recipes, the reader alone, the four memory recipes, the Transformer) are main's,
  recorded at d3626df; the key is out of the hash, the arm's view and the memory key at
  `false`, and in all three when on; the parameter names, shapes and state-dict keys of five models are main's;
  every block at the default computes main's code (written out in the test) bit for bit, in
  float32 and under bf16 autocast, on the plain, span, causal, learned-position, carried and
  word-end paths; HSN-T with its memory gives the same outputs with the key absent, `false` or
  a list of `false`;
- with the key on: the other parameters start as without it, the new ones are `2 d` per enabled
  block, at 1, counted by `hsn.train.memory_plan.reader_param_count`;
- the property: with `qkv` and `norm_attn` scaled 10x the logits of a block without the key grow
  more than a hundred times, past the bound; with the key they stay under it, rotated and
  unrotated;
- the norm is RMSNorm over a head's channels with the block's weights, the values are the
  projection's, and under bf16 autocast all three stay bf16;
- a list enables the listed blocks only (the others compute the plain reader's blocks bit for
  bit);
- every path agrees: the carried windows equal the direct computation over the whole stream
  (written out with the norm) and the span reader in one pass (three lists, two rotary
  settings, four window and carry lengths), the cache holds normalised keys, the whole window
  and the span equal the direct computation, the causal mask equals the document mask, the
  top block's word ends equal the direct computation, and HSN-T with its memory and the carry
  equals its sighted reading (per-byte writes and word events);
- the memory-off model with the key on its top block compiles as one graph (`aot_eager`,
  `fullgraph`), with and without the carry, and gives eager's logits, cache and gradients; the
  next window reuses the graph (in HSN-T with its memory the reader runs in the eager prologue
  of the compiled window, so its graph does not change);
- the gradient reaches `g_q` and `g_k` through the plain and the carried reader and through
  HSN-T with its memory;
- the trainers' groups put the new weights in a group without weight decay, beside the reader's
  LayerNorm weights, while `qkv` decays: the generic trainer (AdamW, also fused), the reader
  alone, the TBPTT trainer (LazyAdamW, also with the cell's own groups, and AdamW);
- the refusals above, and the accepted settings build and read.

Outside the suite, on 21 Nov 2026: the same ten configurations and five models (the
memory-off top0 reader with and without the carry, the two carry recipes, the reader alone)
built and run in two processes, one on main's code and one on this branch, at the default:
the hashes, the memory keys, the initial parameters and two carried windows of logits are equal
bit for bit (CPU, float32).

Three CUDA tests in the same file (skipped without a card, not run yet): three compiled bf16
steps under CUDA graphs of `hsn_t_nomem.yaml` with the top0 reader and the key on its top
block (`hardware.baseline_compile=reduce-overhead`), of `hsn_t_memory_words_carry.yaml` with
the key on its top block and the CUDA arm's lines (`hardware.compile_mode=reduce-overhead`,
the alias index and the two speed keys), and the causal reader on the flash kernel alone,
forward and backward, then three compiled steps with it.

The existing reader, HSN-T and configuration tests pass unchanged, with the one list of keys
left out of the hash (`tests/test_p1_16_config.py`) extended by this key.

## Results

Card runs of 21 Nov 2026 by joe-3e: tree 9aeb660, the local RTX 3060 alone, 27,000 steps, batch 32 x 256,
seed 0, spec `fields_27k_qk` (folder `~/hsn-experiments/qknorm-runs`). The references are the same arms without
the key, on tree 3c85c2a (main's model code; the specs `fields_27k*`). Dev bits per byte / next-byte accuracy %
at the last step (lower bits, higher accuracy are better); ms per step alone.

| Reader (160 x 4, carry 256) | Code native | Bio prose | Sum of bits |
| --- | --- | --- | --- |
| Memory-off, top block unrotated (top0) | 1.148 / 77.77 | 1.356 / 71.82 | 2.504 |
| Memory-off, top block rotated | 1.058 / 79.41 | 1.442 / 69.80 | 2.500 |
| **Memory-off, top0 + `reader_qk_norm=[false,false,false,true]`** | **1.041 / 79.67** | **1.345 / 72.08** | **2.386** |
| Memory-off, top0 + `reader_qk_norm=true` (every block) | 1.054 / 79.46 | 1.368 / 71.63 | 2.422 |
| Word events + balance (lambda_b 0.02), top0 | 1.172 / 77.38 | 1.291 / 73.20 | 2.463 |
| Word events + balance, top block rotated | 1.025 / 79.94 | 1.357 / 71.71 | 2.382 |
| **Word events + balance, top0 + the key on the top block** | **1.023 / 80.00** | **1.280 / 73.43** | **2.303** |

- With the key on the top block, one reader beats choosing the reader per field (rotated on code, top0 on prose:
  2.414 memory-off, 2.316 with word events) on both fields, with the memory on and off, in bits and accuracy.
- On every block it is worse than on the top block alone.
- The prose copy survives: the accuracy of a rare word at a repeat minus at its first mention, bio prose,
  points at < 64 / 64-255 / 256-1023 bytes: memory-off top0 +25.1 / +25.0 / -2.7, top0 + key +25.1 / +25.2 / -2.6,
  rotated +1.6 / +2.9 / -3.0; word events + balance top0 +24.2 / +24.0 / -2.7, top0 + key +24.1 / +23.9 / -2.8.
  On code native the key changes these by at most 0.8 points.
- Step time: memory-off 16.1 ms (top0 15.9 to 16.0); word events + balance 62.3 ms on code and 62.7 on prose,
  against 57.3 to 58.2 without the key. With the memory on, the reader runs in the window's eager prologue
  (outside the compiled window), so the norm's kernels are launched one by one; memory-off compiles it.
- The weights still grow, more slowly. The top block's norm_attn.weight and qkv.weight norms at 9k / 18k / 27k, word
  events + balance on code native: 18.3 / 22.5 / 24.8 and 40.1 / 56.0 / 62.0 (top0 without the key: 41 to 69 and 84
  to 115 from 9k to 27k). But the bound itself rises: max|g_q| * max|g_k| goes 19.2 / 40.8 / 58.6 (bio prose 7.0 /
  32.0 / 57.7; memory-off 12 to 29). The 80 gains carry no weight decay, so the logit bound is not fixed; at 27k
  no run slid, and longer runs are untested.

## Where the code is

- `python/hsn/config.py`: `ModelConfig.reader_qk_norm`, its refusals, `HASH_OMIT_AT_DEFAULT`.
- `python/hsn/model/reader.py`: `_head_norm`, `_ReaderBlock._project` (and `q_norm`, `k_norm`),
  `Reader.qk_norm`.
- `python/hsn/experiments/spec.py`: the memory key leaves the key out at `false`.
- `python/hsn/train/memory_plan.py`: `reader_param_count`.
- `configs/_example.yaml`: the key with its default.
