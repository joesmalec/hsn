# The Transformer baseline as Transformer-XL (`baseline.transformer.position` / `carry`)

Status: **opt-in keys of the Transformer baseline, written on 17 Nov 2026 (branch
`claude/txl-baseline`); tested on the CPU; not yet run on a card or trained.** Asked for on 17 Nov 2026 as a more appropriate
control for HSN-T with its memory and the reader carried across windows
(`model.reader_carry=256` with the memory on, branch `claude/reader-carry-memory`): does the HSN
memory add anything beyond a carried cache? The sharpest control is a plain small Transformer
with segment-level recurrence, Transformer-XL (Dai et al. 2019).

## The keys

Two keys of `baseline.transformer`, read by `model_name: transformer` only (every other model
name refuses them away from their defaults):

| Key | Default | Meaning |
| --- | --- | --- |
| `position` | `learned` | `learned`: the position table over the window, the model as it was. `rotary`: no table; every block rotates its queries and keys by their position (below). |
| `carry` | `0` | Transformer-XL's memory length `M`, in positions (bytes). Needs `position: rotary`. 0: no cache, every window read alone. |

At their defaults the model is the one it was, bit for bit: the same parameters from the same
seed, the same state dict, the same outputs (`tests/test_transformer_xl.py` checks the forward
against the old one written out, in training and evaluation). Both keys are left out of the
configuration hash (`HASH_OMIT_AT_DEFAULT`) and of the experiment memory key
(`hsn.experiments.spec.memory_key`) at their defaults, so every existing configuration keeps its
hash, every arm its id and its memory key (checked against values computed at main 5941d13).
A configuration that sets one is another configuration with another hash.

The configuration `configs/baselines/transformer_xl_2m.yaml` is `transformer_2m.yaml` with
`position: rotary` and `carry: 256`, nothing else changed: **1,881,472 parameters**
(`TransformerBaseline.param_report`), transformer_2m's 1,930,624 less its 256 x 192 position table
(49,152), which the rotation replaces. Its twin is the same configuration with
`baseline.transformer.carry=0`: the same parameters from the same seed, every window read alone
(it differs from transformer_2m by its positions only).

## What it computes

The blocks are the Transformer baseline's (pre-norm GPT-2 layout: LayerNorm, causal multi-head
self-attention, residual; LayerNorm, GELU MLP, residual), over the byte and context-symbol
embeddings, then LayerNorm and the head. Two things change.

**Positions (`rotary`).** Every block rotates the pairs `(i, i + d_head/2)` of each head's queries
and keys by the angle `p · 10000^(−i / (d_head/2))` of their position `p` (rotary embedding, all
24 pairs of the 48-wide heads at the 2M size). A score then depends on the distance between query
and key alone, so the positions are consistent across window boundaries. The cosines and sines
are a table of the positions `−M … train.unroll − 1`, computed in float64 (float32 on MPS) where
the parameters are and cast to their dtype, when the model is built and again when it is moved
or cast; a buffer outside the state dict, so it is not in checkpoints. Inside a window the
rotation is applied in the working dtype (bf16 under autocast), as the HSN-T reader's default.

**The cache (`carry: M`).** Every block keeps the keys and values of the last `M` positions of the
stream before the window, detached. Window position `t` attends every valid cached slot and the
window positions `s ≤ t` of its own document: the causal mask over `[cache; window]`. The cached
keys are stored un-rotated and rotated by their positions `−M … −1` at every window, the window's
queries and keys by `0 … T − 1`: the same as rotating every key once by its absolute position in
the stream, since only differences enter a score. After the window the cache keeps the last `M`
positions of `[cache; window]` (with `M > T` it spans several windows).

**Documents.** `reset` acts as it does without the cache: a slot is valid while its document goes
on (a reset anywhere in a row's window invalidates the row's older slots), a window position reads
cached slots only when no reset lies at or before it, and it reads window positions of its own
document only. A document's first position reads itself alone. With `data.align_docs: true`
(transformer_2m's setting, and so the TinyStories arms') every document starts at a window
boundary: the cache carries a story across its windows and is cleared at the next story.

**State.** `init_state` returns the empty cache `(keys, values, valid)`: `keys` / `values
[L, B, H, M, d_model / H]` (block, row, head, slot, channel) in the parameters' dtype whatever the
working dtype (a bf16 value is exact in float32), and `valid [B, M]` (bool). The generic trainer
carries it as it carries the GRU's state: from window to window in training and in the dev
evaluation, detached at every window, written into checkpoints and restored by `resume`.
`reset_state` clears the valid flags of the rows it is given.

**Size per row** at transformer_xl_2m: `2 · L · M · d_model = 2 · 4 · 256 · 192 = 393,216` floats,
1.5 MiB in float32, and 256 flags. At batch 32: 12.6 M floats, 50.3 MB. The cache holds no
parameter.

## How it maps to Transformer-XL

| Transformer-XL (Dai et al. 2019) | Here |
| --- | --- |
| Segment = the training sequence of length `L` | Window = `train.unroll` (256) |
| Memory of length `M` per layer, from the previous segment(s), stop-gradient | `carry: M` per block, detached; with `M > T` it spans several windows |
| The current segment attends `[memory; segment]` causally | The same mask, within a document |
| Relative positional encoding (sinusoids `R_{i−j}`, learned biases `u`, `v`, a projection `W_{k,R}`) | Rotary positions: the score depends on `i − j` only, no position parameter |
| Memory holds the layer inputs `h^{n−1}`; keys and values of the memory are computed from it with the current weights | The cache holds the keys and values themselves (below) |
| Post-norm layers (pre-norm an option of the reference code) | Pre-norm, as the Transformer baseline |
| Evaluation with the memory (often a longer one) | Evaluation with the same `M`, the cache carried through the dev stream |

**Keys and values, not layer inputs.** The cache stores the keys and values of the cached
positions, as the "Transformer-XL-style cache" of later work does (Memorizing Transformers, Wu et
al. 2022; Block-Recurrent Transformers, Hutchins et al. 2022) and as the HSN-T reader's carry does.
In a forward pass with fixed weights the two are the same function: the logits of every window
equal Transformer-XL's over the stream (the tests compare with a direct computation). In training
they differ in two ways. The original recomputes the memory's keys and values with the current
step's weights, and its key and value projections receive gradient through the memory positions
(only the memory itself is a stop-gradient); here the cached keys and values are the ones the
previous step's weights produced, and nothing receives gradient through them. The gradient of a
window is that of the stream read in one pass with the keys and values before the window detached
(tested).

**Reach.** A window position reads at most `M + T − 1` positions back directly (511 at
`M = T = 256`), and every distance it reads was read in training. Through the blocks the reach
grows by about `M` per block, as in Transformer-XL: a cached position's keys were computed from
a cache of their own, so the top block depends on bytes up to about `L · M` (1,024) further back.

**Against the HSN-T reader's carry** (`model.reader_carry: C`, `docs/hsn_t_no_memory.md`): the
reader's carry is a sliding span (position `t` reads `0 ≤ t − s ≤ C`, the same reach at every
window position); Transformer-XL's position `t` reads all `M` cached positions and the window up
to `t` (`M` to `M + T − 1` back). Both cache detached keys and values, rotate them by their
negative positions and handle documents the same way.

## Cost

Each block's attention reads `M + T` keys in place of `T` (twice at `M = T`); the projections, the
MLPs and the head are unchanged; the cache adds one copy of the window's keys and values per
block. The mask is computed once per window and handed to every block as an additive bias (the
HSN-T reader's `_attention_bias`). With `rotary` and no cache the model also uses the additive
bias; the learned default keeps its boolean mask.

## Compiled windows (`hardware.baseline_compile`)

The window compiles as one graph with the cache going in and out (fixed shapes; no
data-dependent shape or branch; the angles are read from the table, not computed in the graph).
`hsn.train.baseline_compile` treats it as it treats the reader's carry: under CUDA graphs
(`reduce-overhead`) the cache returned is copied out of graph memory before the next replay.
Training and evaluation are a compiled function each and nothing recompiles from window to window.
Dropout must be 0 under the key (the shipped configurations have 0).

Checked on the CPU only (17 Nov 2026): inductor (`default`) and `aot_eager` equal eager in values,
cache and gradients, one graph per function, no recompile. **Not yet run on a card**: the RTX 3060
was busy with the reader-carry runs when this was written, so `reduce-overhead` (CUDA graphs) and
the step time are unmeasured. To run them (plain CUDA context, card free):

```
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
PYTHONPATH=python pytest -m cuda tests/test_baseline_compile.py -k transformer_xl
```

then a timing at the run size (`profile=rtx3060 hardware.compile_mode=none train.batch=32
train.micro_batch=32 hardware.baseline_compile=reduce-overhead`), against the twin
(`baseline.transformer.carry=0`) and eager; read `cudagraph_skips` 0 and `graphs` equal to
`functions` in the record.

## Tests

`tests/test_transformer_xl.py` (CPU, float64 unless stated):

- the keys at their defaults leave the configuration hash, the memory key and the arm id of
  transformer_2m (also at TinyStories batch 32 and compiled), the GRU, Mamba, HSN-T without
  memory and the A6 recipe as they were at main 5941d13 (pinned values); set, they change all
  three; refusals (carry without rotary, odd head width, the keys on another model name);
- the default model equals the old forward bit for bit (float32, training and evaluation); the
  rotary twin and the cached model have the same parameters from the same seed, and an empty
  cache or a document starting at the window's first position reads as the twin does;
- **window by window with the cache equals Transformer-XL over the whole stream** (a direct
  computation with the reach written as loops) for `(T, M)` in (8, 8), (8, 3), (5, 12), (6, 20),
  (16, 8), (4, 1), with resets inside windows and at their first and last positions; with
  `M = T` window 1 equals the twin reading `[window 0; window 1]` as one causal window;
- the cache holds the un-rotated keys and the values of the last `M` positions of the document
  going on; under bf16 autocast it keeps its dtype and shapes; wrong caches are refused;
- the cache is a stop-gradient: it carries no graph, window 1's gradients do not depend on
  whether window 0 ran with a graph, and they equal the one-pass reading's with the keys and
  values before window 1 detached;
- a document reads nothing of the one before it, cached or not;
- the generic trainer trains 40 steps with a falling loss, its evaluation carries the cache from
  window to window, and the checkpoint holds it; a resume from step 6 equals the uninterrupted
  12-step run bit for bit (losses, evaluation, parameters, cache);
- `aot_eager` with `fullgraph`: one graph, equal to eager in values, cache and gradients over three
  windows, no recompile.

`tests/test_baseline_compile.py` runs its family checks for it too (`transformer_xl` in `TINY`):
inductor against eager on the CPU, the trainer through the compiled window, and on the card
`reduce-overhead` and `default` against eager, no CUDA-graph skip, training under CUDA graphs.

## Runs

Spec `~/hsn-experiments/remote/specs/txl_baseline.yaml`: TinyStories at 9k and 27k steps, seed 0,
batch 32, window 256, lr 1e-3 (transformer_2m's), compiled (`hardware.baseline_compile=
reduce-overhead`); `M = 256` and its twin `M = 0` at each length.
