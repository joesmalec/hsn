# HSN-T: the HSN's memory core behind a Transformer reader (experimental)

Status: **a model family of its own (`model_name: hsn_t`), implemented on 14 Nov 2026; not
adopted; first runs in progress.**
- **Why a family and not an HSN amendment.** The reader is sequence attention, and the HSN is
  defined without it (A5: "No sequence attention, additional persistent state, or
  computation-path weight writes"). The user's decision on 14 Nov 2026, relayed by the
  campaign session: "HSN-T family of its own"; the name is the user's ("what about HSN-T for
  transformer?"). So the HSN is unchanged and cannot have a reader: `model_name: hsn`
  refuses `model.reader`, the class `hsn.model.hsn.HSN` refuses it, and `model_name: hsn_t`
  (class `hsn.model.hsn_t.HSNT`) requires it. The white paper of this line is
  `whitepaper-hsn-t` (drafted separately); this document is the code's record.
- **History.** Built first as an opt-in key of the HSN ("amendment A14", commits 50db3ae to
  76c2aec on `claude/a14-reader`) after the user's "yes build rung 1 as an opt in
  amendement" and "lets explore this hybrid idea even more, I feel like it is different then
  a straight llm being a hybrid and if we can manage the same percentages with the speed it
  is a significant developement." The first runs (spec `a14_reader.yaml`, 14 Nov) were made
  from those commits with `model_name: hsn`; they compute what `hsn_t` computes, and their
  run folders load from a checkout of 54957f3 or 76c2aec, not from this one.
- `model.reader`: `none` | `transformer`, with `model.reader_width` (64),
  `model.reader_layers` (2), `model.reader_heads` (4) and `model.reader_span` (null). Needs
  `model.selected_readout`. All five keys are left out of the configuration hash at their
  defaults, so no HSN configuration, hash or arm id changed.
- Five more opt-in keys of 15 Nov 2026 change how the position enters the reader
  (`model.reader_position`, `model.reader_positions`, `model.reader_rotary_pairs`,
  `model.reader_rotary_dtype`, `model.reader_rotary_base`; "The reader's positions" below). At their defaults the reader
  is the one described here, bit for bit, and they are left out of the hash the same way.
- The memory's opt-in keys (fast cells, address n-grams, the address gate and its inputs, the
  write decision, the feedback cuts) mean the same in both families; the cell arithmetic is
  shared code.
- It changes no metric, threshold or task.

## Cause

What the candidate's two groups of cells do, as measured on 13 and 14 Nov:

| Measurement | Reading |
| --- | --- |
| 8 fast cells + 1 routed slot: TinyStories 0.948 / 0.956 bits, arithmetic 100 %; delayed recall 80 % | The fast cells read text and do arithmetic; they do not bind keys |
| 12 routed slots + address gate, no fast cells, 18,000 steps on `synthetic_hold`: 99.2 % on training keys, 99.1 / 98.9 % on unseen keys | The routed cells with fixed addresses are the memory, and the best recall measured |
| Write decision with a cost (A10 revived, 9,000 steps): routed writes allowed at 0 % of slots on arithmetic (96.3 / 95.6 %), 2.4 % on TinyStories (1.007 bits against 0.990) | On text and arithmetic the memory can stay untouched |
| Transformer baseline (1.93 M parameters): 16 ms per step, TinyStories 0.827 bits at 27,000 steps; the candidate's shape 132 ms without the address gate and 143 ms with it, 0.84 bits | Reading inside the window is what a Transformer does faster. No Transformer of the reader's size has been trained here (reference runs: `a14_refs.yaml`) |
| Transformer on the pair task: 18 %; with the byte codes as an input 100 % on training keys, 55 % on unseen keys | It cannot replace the memory |
| All speed changes inside the HSN together: 1.15 to 1.22 times | The time is in the work per byte, not in waste |

So the reading is given to a reader built for it and the memory is left exactly as it is.

## The change (opt-in)

A reader `R` (`hsn.model.reader.Reader`) maps the bytes of a window to one vector per position,

```
c_{1..T} = R(x_{1..T}),      c_t ∈ R^w   (w = model.reader_width)
```

and `c_t` takes the place of A8's fast read `f` where the model reads it:

```
u_t      = S([z_t ; q_{t−1} ; ctx_t ; c_t])        (A8: … ; f_{t−1}])
logits_t = D([q_t ; u_t ; r_t ; c_t])              (A8: … ; f_t])
```

`R` is a pre-norm causal Transformer: byte embedding, `reader_layers` blocks (multi-head
self-attention, then a GELU MLP of width `4w`, both residual), a final LayerNorm. Positions
enter as rotations of the attention's queries and keys (rotary), so only distances are read
and there is no position parameter. Since 15 Nov 2026 the rotations' cos / sin are read from a
table of the first 1,024 window positions, made when the reader is built and made again on its
device when the model is moved or cast: bit for bit the float64 computation that the forward
pass repeated in every call and that `torch.compile` inlined into the rotation kernels. The
table is not in the state dict; a longer window, and MPS, still compute the angles in the call.

**What `c_t` may depend on** (each line is a test):
- the bytes `x_s`, `s ≤ t`, of the same window and the same document (a reset column starts a
  document for the reader as it does for the cells);
- with `model.reader_span = n` only the last `n` of them at each layer: the short-sighted
  reader. Its complete receptive field is `1 + L (n − 1)` bytes for `L` blocks: 63 bytes for
  the two blocks and span 32 that are run, not 32;
- nothing of an earlier window, no cell state, no context symbol.

So global rule 2 holds as before (`ẑ_t = P(q_{t−1})` reads the carried summary only), and
everything further back than the reader's reach can come from the memory alone.

**Unchanged:** the cells and their state carried from window to window, the fixed cell
addresses and byte codes (A5, A9), the router and its selection rule, the address gate (A12,
A13), the unit bank, the convex tanh gate, the graph, the summary `q`, the selected-cell
readout `r`, the bounded position (A11), the state layout, every loss.

**Gradient.** `c_t` is not a state read: A6's cuts do not apply to it. The reader trains through
the decoder and through `S`.

**With fast cells kept** (`model.fast_cells > 0`, allowed; the variant to try only if
arithmetic fails with the reader alone): `S([z ; q ; ctx ; c ; f])`, `D([q ; u ; r ; f ; c])`.

**What the address gate reads.** A13's `summary` (the carried summary `q_{t−1}`) is what the
runs use. Without fast cells the summary holds less of the last bytes than it does in the
candidate, so one more value of `model.address_gate_input` exists for this model, built on
14 Nov and not run: `reader_prev`, the reader's output of the position before, `c_{t−1}` (zero at
a document's first position and at a window's first column; needs `reader_width = d`). Like
the summary it holds the bytes up to `t − 1` and not the current byte. It is a second path
from the reader into routing, direct where the summary's is indirect (reader → `S` → cell
state → summary → later gate): fixed address features do not make routing independent of
the reader in either form.

## Size

| | Parameters |
| --- | --- |
| Reader (width 64, 2 blocks, V 256) | 116,480 |
| New input columns of `S` and `D` (2 × 64 × 512) | 65,536 |
| 12 routed slots, address gate, no fast cells (the memory core alone) | 1,855,785 |
| The candidate (8 fast + 12 routed, fast read) | 1,921,321 |
| The memory core with the reader (as run) | 2,037,801 |

The reader model is 6 % larger than the candidate it is compared with.

`HSN.param_report()` gains the row `reader` (outside Appendix C, like the edge logits) when the
key is on; `expected_param_count` widens `shared` and `decoder`;
`hsn.train.memory_plan.reader_param_count` is the reader's closed form. Appendix C.2's
multiply-accumulate table does not count the reader's attention.

## Where it is in the code

- `python/hsn/model/reader.py`: the reader; its parameters come from its own generator inside
  a forked RNG.
- `python/hsn/model/encoder.py`: `S` and `D` get `reader_width` more input columns, drawn the
  same way, so every parameter the model shares with the model without the key starts equal.
- `python/hsn/model/hsn.py`: the reader runs once per window in `_window_prologue`; `c_t`
  travels behind the context embedding, so the loop, the compiled block and `_position` are
  untouched; the decoder's read is added in `_window_epilogue`.
- `python/hsn/model/hsn_t.py`: the class `HSNT`, an `HSN` that owns a reader; the runner
  builds it for `model_name: hsn_t` and the HSN's window trainer trains it. In an experiment
  spec the family is `hsn_t`.
- The built-in verdict presets of the experiment rules (`reads_state`, the write and gate
  presets) are scoped to the family `hsn` and pinned; they are not applied to `hsn_t` arms.
- Not available to HSN-T: `HSN.step` (one position has no window to read: the chat session
  and the edit probes use it) and `hardware.window = split | fused` (the HSN's windows). The
  baselines and the cell variants other than `pdg` cannot have a reader: it needs A5's
  readout, which they refuse.

## The reader alone (`model_name: reader`)

Before a text or arithmetic result of HSN-T is read as a failure of the hybrid, the reader it
contains has to be measured by itself (the review amendments of 14 Nov 2026, item 04a-3). The
stock Transformer baseline of the same shape is not that reader: it has a position table and a
context input, the reader has rotary positions and none.

`hsn.baselines.reader_lm.ReaderLM` is the reader module itself with one linear layer from
`c_t` to the next byte (`configs/baselines/reader_w64l2.yaml`: 116,480 + 16,640 = 133,120
parameters; `model.reader_span=32` for the short-sighted form). Built with the seed of an
HSN-T it starts with that HSN-T's reader weights and gives the same hidden outputs, bit for
bit (float32, CPU; `tests/test_reader_lm.py`). It is stateless, reads no context symbol, and
trains with the generic trainer on the Transformer baseline's settings, so that the two
differ in the model only. In an experiment spec the family is `reader`.

What it does not settle: the reader inside HSN-T trains on the HSN recipe (learning rate
2.5e-3, the recipe's controllers; the reader's parameters are in no update-limit group), the
reader alone on the baseline's (1e-3, cosine). A difference between the two results mixes
recipe and integration; separate learning-rate groups for reader and core are not built.

One run exists (15 Nov 2026, TinyStories, width 192, four blocks): 0.967 bits against the
Transformer baseline's 0.919. "The reader's positions" below has where the difference sits.

## The reader's positions (opt-in keys, 15 Nov 2026)

Status: **built to test a measured cause; no card run with any of them yet** (spec
`text_gap.yaml`). The defaults are the reader as it was.

### What was measured

TinyStories, 9,000 steps, batch 32, seed 0, bits per byte on the 2,000 dev stories. The reader
alone (`model_name: reader`, width 192, four blocks, 1,878,400 parameters) reached 0.967; the
Transformer baseline of the same blocks (1,930,624 parameters) 0.919. Both ran on the generic
trainer with the same settings (their resolved configurations differ in the model keys only),
the same data order and the same loss. The models differ in three things:

| | Transformer baseline | Reader alone |
| --- | --- | --- |
| Position | a learned table added to the byte embedding | rotation of the attention's queries and keys |
| Context symbol | an embedding added to the input | not read |
| Arithmetic of the position under bf16 autocast | an addition in float32 | the rotation in bf16 |

The context symbol is the same id (6) at every position of `tinystories_v1`, so on this data
its embedding is one constant vector and carries nothing. The trainer path, optimiser groups
(weight decay on every tensor of two or more dimensions, embeddings included), schedule,
clipping, initialisation (normal 0.02, zero biases), final LayerNorm, head and attention mask
are the same.

The step-9,000 checkpoints of both runs, scored on the CPU in float32 by position in the
window (the totals reproduce the runs' 0.919 and 0.967, so evaluating the rotation in float32
in place of bf16 changes nothing at four digits):

| Window positions | Share of the bytes | Transformer | Reader | Difference |
| --- | --- | --- | --- | --- |
| 0 to 7 | 3.6 % | 1.477 | 1.495 | +0.018 |
| 8 to 31 | 10.7 % | 0.932 | 0.931 | −0.001 |
| 32 to 63 | 13.8 % | 0.907 | 0.922 | +0.016 |
| 64 to 127 | 26.0 % | 0.913 | 0.954 | +0.041 |
| 128 to 255 | 45.9 % | 0.880 | 0.954 | +0.074 |
| All | 100 % | 0.919 | 0.967 | +0.047 |

Documents are aligned to windows, so the position in the window is the number of bytes the
model can see. Of the 0.047 bits, 0.044 are at positions 64 and later. The reader gains nothing
from the second half of a window (0.954 at 64 to 127 and at 128 to 255); the baseline does
(0.913 to 0.880). The picture is the same in a story's first window and in its later ones.

Where the attention of the queries at positions 160 to 255 goes (the same checkpoints, 300 dev
stories; the share of a head's attention on bytes 32 or more positions back, the four heads of
a block):

| Block | Transformer | Reader |
| --- | --- | --- |
| 1 | 0.09, 0.07, 0.06, 0.07 | 0.10, 0.23, 0.09, 0.07 |
| 2 | 0.06, 0.04, 0.03, 0.04 | 0.10, 0.06, 0.15, 0.06 |
| 3 | 0.10, 0.03, 0.26, 0.14 | 0.11, 0.08, 0.10, 0.15 |
| 4 | 0.62, 0.66, 0.78, 0.69 | 0.09, 0.25, 0.07, 0.29 |

The baseline's last block reads far back with all four heads (mean distance 63 to 94 bytes).
No block of the reader does (mean distance 33 bytes at most). With the attention of every block
cut to the last `n` bytes, at window positions 128 to 255:

| `n` | Transformer | Reader |
| --- | --- | --- |
| whole window | 0.888 | 0.958 |
| 64 | 1.002 | 1.014 |
| 32 | 1.089 | 1.055 |
| 16 | 1.230 | 1.129 |

Inside 32 bytes the reader is the better model; the baseline's lead is what it reads from
further back. This is a finding about these two trained models, not yet about its cause: the
rotation is the difference that remains, and the arms of `text_gap.yaml` test it.

### The keys

All are keys of the reader, so they mean the same in HSN-T with its memory, without it and in
the reader alone. `S`, `D` and the model's equations are untouched.

| Key | Default | Other values | What changes |
| --- | --- | --- | --- |
| `model.reader_rotary_pairs` | `null` (all `w / heads / 2` pairs of a head, in every block) | `n`, or a list with one `n` for each block | Only the first `n` pairs (the `n` highest frequencies, as before) are rotated. The other pairs get the angle zero: their part of a query is compared with a key as it is, at every distance. With `0` a block is not rotated at all. Still distances only, no parameter |
| `model.reader_rotary_dtype` | `working` | `parameters` | The rotation is computed in the parameters' dtype (float32) and the rotated queries and keys are cast back to the working dtype for the attention. Under bf16 autocast the default rotates in bf16 (cos / sin and every product rounded to 8 bits). Without autocast the two are the same computation, bit for bit |
| `model.reader_rotary_base` | `10000.0` | a finite number above 1 | The base `b` of the frequency ladder: pair `i` of a head turns by `t · b^(−i / half)` at distance `t`, in the table and in the angles the forward pass computes itself alike. Refused with `reader_position: learned`. Still distances only, no parameter |
| `model.reader_position` | `rotary` | `learned` | No rotation; a table of `model.reader_positions` rows (256) is added to the byte embedding by window position, as the Transformer baseline's. `train.unroll` must not exceed the rows |

- **`learned` is a diagnostic.** With it the reader alone is the Transformer baseline's
  computation (same logits with the weights copied, tested), so its arm says how much of the
  gap the position scheme is, and it repeats the baseline at today's commit on this machine.
  It gives up what the rotary reader was chosen for: the output then depends on where the bytes
  sit in the window, and with a span no longer on the last `n` bytes alone. The table adds
  `reader_positions × w` parameters and is drawn after the reader's other parameters, which
  start as the rotary reader's.
- **Fewer rotated pairs keeps the reader's property.** The output is still a function of
  distances (tested as for the default). What it tests: with every pair rotated, a query can
  match a key by content at any distance only in the pairs whose rotation over the window is
  small (at width 192 with four heads, 9 of 24 pairs turn by less than one radian over 255
  positions and 5 by less than a quarter); the unrotated pairs do it exactly. With half the
  pairs rotated the slowest of them still has a wavelength of 428 positions (452 at width
  224), longer than the window.
- **One count for each block, and the last block unrotated.** The baseline adds its
  positions once, before the first block, and its far reading is all in its last block; the
  reader rotates in every block, and every block of it stays near. `[24, 24, 24, 0]` (or
  `[28, 0]` for two blocks of width 224) rotates in the earlier blocks and leaves the last
  block's attention to compare queries and keys as they are. That block gets no position of
  its own; what it reads of the order is what the earlier blocks wrote into their outputs.
  The output is still a function of distances (tested). A list in which every count is the
  same is that count, and the count of all pairs is the default, bit for bit.
- **No rotation in any block** (`model.reader_rotary_pairs: 0`), for the user's question of
  15 Nov 2026, "can we also do a test with the reader looking far back with all of its
  attention heads?". Every head of every block then compares queries and keys as they are,
  at any distance; the causal mask is the only thing that tells the reader an order (a later
  block can see it in the earlier blocks' outputs). It frees every head to read far and does
  not make it do so: `scripts/reader_reach.py` shows per head whether it does. It also takes
  away the near positions, where the reader is the better model today (above, the attention
  cut to 32 bytes), so a loss on the near bytes is to be expected; not run.
- **A larger base** (`model.reader_rotary_base`), after the user's decision of 15 Nov 2026 to
  keep a rotation-based scheme. Every pair but the first turns more slowly (pair 0 keeps one
  radian a position), so more pairs can match a key by content far back while the fast pairs
  keep the near order. At width 224 with four heads (28 pairs) the pairs that turn by less than
  one radian over 255 positions are 11 at the default 10000, 14 at 1e5 and 16 at 1e6 (under a
  quarter radian: 6, 11 and 13). It composes with `reader_rotary_pairs` (the unrotated pairs
  keep the angle zero) and `reader_rotary_dtype`. Like them it is refused without a reader and
  with `reader_position: learned`; with `reader_rotary_pairs: 0` it has nothing to act on and
  is accepted, as `reader_rotary_dtype` is. Spec `text_rotary_base.yaml`; not run.
- **Step time, compiled, at width 224 with two blocks (12.12 ms alone on the RTX 3060; the
  GRU 15.75). Estimates from the operations, none measured:**

  | Setting | Expected change | Why |
  | --- | --- | --- |
  | `reader_rotary_pairs: n` | none | the same operations, zeros in the table |
  | `reader_rotary_base: b` | none | the same operations, other numbers in the table |
  | `reader_rotary_pairs: [28, 0]` | none to 0.1 ms less | the last block's two rotations are not computed |
  | `reader_rotary_pairs: 0` | none to 0.2 ms less | no rotation is computed |
  | a list of unequal non-zero counts | none | two selections on a `[T, pairs]` table for each block |
  | `reader_position: learned` | none to 0.2 ms less | one addition replaces the rotations of both blocks |
  | `reader_rotary_dtype: parameters` | under 0.1 ms more (under 0.5 ms eager) | four casts a block; the rotation's elementwise operations on `[B, heads, T, w / heads]` in float32, fused into the rotation's kernel when compiled |

  None of them should cost the 3.6 ms lead. The setting that is kept should be timed before
  a result with it is called a speed result.
- **Not built: a context-symbol input to the reader.** The context symbol is constant on
  `tinystories_v1`, so it cannot explain this gap; without the memory the model reads it
  through `S` already.

Tests: `tests/test_reader_positions.py`. The defaults keep every hash, memory key, parameter,
state-dict key and output (against the forward of before the keys); the unrotated pairs come
back bit for bit and match at every distance; each block rotates its own count (against the
blocks run by hand), a block with the count zero never calls the rotation and, alone, gives
the same output for every order of the earlier bytes; the float32 rotation under bf16
autocast, read off the calls and closer to the exact rotation; another base's angles against
`t · b^(−i / half)` and the same base in the table, in the angles computed past it, without it
and after casts; the learned reader against the
Transformer baseline; the refusals; the closed forms of the parameter count; HSN-T with its
memory, the compiled memory-off graph and thirty trainer steps with each key.

## Without the memory (`model.memory: false`)

An opt-in switch of 14 Nov 2026 (`docs/hsn_t_no_memory.md`): the parts of HSN-T that are not
the memory (embeddings, `E`, the reader, `S`, `D`), stateless and over all positions of a
window at once, with zeros where `q` and `r` are read. It is a model of its own
(`hsn.model.hsn_t.HSNTNoMemory`, 627,328 parameters at the default sizes, experiment family
`hsn_t_nomem`, the generic trainer), built for a speed comparison with the GRU baseline. The
default is `true`, and with it nothing in this document changes.

Its first text runs (15 Nov 2026) and where they stand against the reader alone and the
baselines are in `docs/hsn_t_no_memory.md`, "Text: the gap to the baselines".

## The reader's causal mask (`model.reader_mask`, opt-in, 17 Nov 2026)

`model.reader_mask: causal` runs the reader's attention as `scaled_dot_product_attention(...,
is_causal=True)` with no mask, which the flash kernel takes on CUDA, in place of the additive
document mask (the memory-efficient kernel). Where every reset of a window is at column 0, which
`data.align_docs: true` guarantees, the two masks are the same and so is the reader. Refused with
`reader_span`, `reader_carry` or `data.align_docs: false`; the reader asserts the condition on the
device on every window (no host read). Allowed with the memory on and off and for the reader
alone. The default `document` is the reader as it was, bit for bit. Design, tests and timings:
`docs/reader_causal_flash.md`.

## The attention read of the cells (`model.cell_attention_heads`)

An opt-in key of 15 Nov 2026 (`docs/amendments/cell_attention_read.md`): with
`H > 0` the reader's output `c_t` attends over all cells at every position (keys from each
cell's state and its fixed address code, values from its state) and the result `m_t` is one
more input of the decoder, `D([q; u; r; c; m])`. It stands beside the router's read `r_t`; the
memory does not read it. The default is 0, and with it nothing in this document changes.

The recipe of HSN-T with its memory, `configs/experiments/hsn_t_memory.yaml` (main's form, 16 Nov
2026), leaves it off (`H = 0`): it helps on TinyStories (the names) but costs the recall tasks
(kv, delayed), so text runs set `H = 4`. Results in the design note, "Results (16 Nov 2026)" and
"Results on the recall tasks (16 Nov 2026)".

## Reach dropout (`train.reach_dropout`)

A training option of 17 Nov 2026 (`docs/reach_dropout.md`; opt-in, no equation change): in a
training window each row reads, with probability `train.reach_dropout`, with the per-block span
`train.reach_dropout_span` (16 by default), the other rows with the whole window; evaluation
reads as configured. It leaves the names beyond the short span to the memory in those rows
while the reader keeps the whole window (experiment E1 of the names design note). The default
is 0, and with it nothing in this document changes.

## Tests (`tests/test_hsn_t.py`)

The two families are kept apart (config, class, registry); the keys leave the HSN's hashes
alone; the refusals; no later byte, no earlier
document, the span, distances and not window positions; the closed forms; the other
parameters' initialisation and the global random stream are untouched; with the new columns
zero the window equals the window of the model without the key (float64, 1e-12); logits read
no later byte; gradients reach the reader and the new columns under the candidate's cut; `S`
is the only other way in; bf16 autocast; fast cells beside the reader; the compiled window
equals the eager one; thirty trainer steps fall. The rotary table has its own file,
`tests/test_reader_rotary_table.py`.

## What it has to show (rung 1 of the design)

On the candidate with the fast cells replaced (`model.k_select=12`, no `fast_*` keys, the
summary gate), slow window, three seeds where a claim is made:

| Data | Bar | It fails if |
| --- | --- | --- |
| `synthetic_hold`, 18,000 steps | 98 % or more on training keys and on unseen keys | more than 3 points under the routed-cells-and-gate runs on any seed |
| TinyStories, 9,000 steps | 0.93 bits or better | no better than the fast cells' 0.95 to 0.96 |
| Arithmetic, 9,000 steps | 95 % or more | (then: two fast cells beside the reader) |
| Five tasks, 27,000 steps | at least the candidate's 89.9 % | |

The short-sighted reader (`model.reader_span=32`) is the main form and the full-window one is
run beside it. The recipe aligns documents to windows (`data.align_docs`), so on the synthetic
tasks a full-window reader sees a fact and its query together and could answer without the
memory, as the Transformer baseline did for the keys it trained on (100 % against 55 % on
unseen keys). A reader that cannot see far has to leave every distant fact to the memory. The
split by training keys and unseen keys (`hold_eval.py`) is the check in both forms.

Readings are two of three seeds; the reference for recall is the same memory core without a
reader on three seeds (`hold_k12nw_ag_ng2_cutC_cells64_18k_gsum`).

**No speed is claimed.** The routed cells still run at every byte; the
window is the slow one. Speed is rungs 2 and 3 of the design and needs its own approval.

## Results

None yet.
