# Reading past the window with the memory on (HSN-T; opt-in, in no recipe)

Status: **built on 17 Nov 2026, opt-in, in no recipe, not run on the card.** Branch
`claude/reader-carry-memory` (on main ed9487b). The user's decision of 17 Nov 2026, relayed by
the campaign session: "lets add that fix as an opt in and test it". No amendment number.

- `model.reader_carry: C` (default 0 = off) is now accepted with `model.memory = true`. The key
  itself is not new: HSN-T without its memory and the reader alone have had it since 15 Nov 2026
  (`docs/hsn_t_no_memory.md`, "Reading past the window"). With the memory on it was refused.
- At the default nothing changes. The key was already left out of the configuration hash and of
  the experiment database's memory key at 0. With `C = 0` the reader runs the code it ran, a
  window's `info` holds no new entry and `address_history` returns the tensor it returned.
- It changes no metric, threshold, task or parameter count. A checkpoint trained without the
  key loads into a model with it (and is then read with a carried reader).

## Cause

`~/hsn-handoff/TEXT-GAP-DESIGN-2026-11-17.md` (17 Nov 2026, CPU diagnostics on the 27,000-step
TinyStories checkpoints, 2,000 dev stories):

| Measurement (dev bits per byte) | Per-byte memory, top0 | Word events, top0 | Mamba (lr 2e-3) |
| --- | --- | --- | --- |
| All bytes, as trained (256-byte windows) | 0.7782 | 0.7971 | 0.7177 |
| First window of a story | 0.6338 | 0.6373 | 0.6328 |
| Later windows | 0.8466 | 0.8729 | 0.7579 |
| First 8 bytes of a later window | 1.32 | 1.56 | 0.75 |
| Same weights, reader with 256 bytes of sight across the boundary | 0.7312 | 0.7445 | |

- All of the gap to Mamba is in the windows after the first. Inside them it is a sawtooth: the
  reader starts every window with one byte in sight, and the memory replaces about half of what
  it lost.
- 77 % of the gap sits on words the story used before, outside the reader's current window.
- Reading the same checkpoints with a reader that keeps its sight removes 78 % of the per-byte
  gap and 66 % of the word-event gap, with no training. That reading is this key (below,
  "The oracle").

## The change (opt-in)

HSN-T reads a window's bytes with a reader `R` and hands its output `c_t` to the memory core:

```
u_t      = S([z_t ; q_{t−1} ; ctx_t ; c_t])
logits_t = D([q_t ; u_t ; r_t ; c_t])            (with C-read: … ; m_t])
```

Until now, with the memory on, `c_t` could depend only on bytes of its own window:

```
c_t = R(x_s : w(t) ≤ s ≤ t)            w(t): the first byte of t's window, or of its document
```

With `model.reader_carry = C > 0`:

```
c_t = R(x_s : 0 ≤ t − s ≤ C in every block, s in t's document)
```

In words: every reader block attends over the last `C + 1` positions of the document, whether
they lie in this window or in the windows before. It is a sliding span, the same at every
window position and at every window and document length, so no attention distance beyond `C`
is ever read. The positions before the window are not recomputed. Each block's un-rotated keys
and its values at the last `C` positions are kept in a cache, detached (no gradient crosses a
window through the reader), and the next window reads them with the rotation of their
distance. This is `Reader.read`, the mechanism HSN-T without its memory already uses; no line
of the reader changed.

**Unchanged:** everything after `c_t`. The cells and their carried state, the fixed addresses
and byte codes, the graph, the unit bank, ε, the convex tanh gate, the router and its
selection rule, the address gate, the summary `q`, the selected-cell readout `r`, the
attention read of the cells, the event loop, every loss, the state layout
(`state_schema_version` stays 1), the parameters. Global rule 2 holds as before (`ẑ_t = P(q_{t−1})`
reads the carried summary only).

**What `c_t` may now depend on** (each line is a test):
- the bytes `x_s`, `s ≤ t`, of the same document, in this window or before it, at most `C`
  positions back in every block;
- no later byte, no earlier document (a document start clears the cache for its row: a
  position reads nothing cached from before the reset, and after the window only positions of
  the document going on stay valid);
- no cell state, no context symbol.

**Reach, not span.** A span of `C + 1` per block over `L` blocks can relay information from
`L·C` bytes back (1,024 for four blocks and `C = 256`). Measured on the checkpoints above, read
with a span of 256: nothing is relayed (letters of words last used 256 to 511 bytes back cost
1.474 bits as trained and 1.473 sighted; the memory-off model trained with `reader_carry=256`
pays 1.566). So what is older than about `C` bytes still has to come from the memory, in
practice. It is not a strict limit.

## The state cost

The cache is carried state outside the cells:

| Per row | Numbers |
| --- | --- |
| The cells' state `h` (64 cells × 64) | 4,096 |
| Mamba's state (the baseline) | 38,912 |
| The reader's cache at `C = 256`, width 160, four blocks: keys and values | 327,680 |

(`L · C · w · 2`, plus `C` validity bits.) It holds nothing older than `C` bytes and nothing the
bytes themselves do not determine: it is the reader's own activations at positions it has
already read, kept so that they need not be computed again. At batch 32 it is 42 MB in fp32.

**A variant without stored activations (not built).** The reader could instead re-read the last
`C` bytes before its window: the trainer hands the bytes on (it already hands on the address
bytes), the reader runs over `C + T` positions and its last `T` outputs are used. That carries
`C` bytes per row instead of 327,680 numbers, its reach is exactly `C` bytes before the window
(no relay through cached keys), and it costs about one more reader pass per window (estimate:
+15 ms per step on the RTX 3060). It is not the same function as the cache when `L > 1` (the
cached keys of block 2 were computed with their own full sight; a re-read sees only `C` bytes
back), so it would be its own key with its own oracle. Left as a later option.

## Where the cache lives

`CellGraphState` is not touched. The cache rides with the address carry, which callers already
hand from window to window outside the state (`prev_tokens`):

- `hsn.model.hsn.WindowCarry(address, reader)`: the address carry as before (the bytes, or with
  `model.word_address` the int64 address carry) and the reader's cache `(keys, values, valid)`.
- A window's forward pass takes it as `prev_tokens` and puts the next cache into `info`
  (`READER_CACHE_KEYS`: `reader_cache_keys`, `reader_cache_values`, `reader_cache_valid`).
- `address_history(cfg, tokens, previous, info)` returns the next `WindowCarry`. It needs the
  window's `info`, as it does with `model.word_address`.

So every caller that carries the address the documented way carries the reader's sight with it
and needed no change: `evaluate_windows`, the profile loops of `hsn.runner`, and the hand-off
scorers (`~/hsn-handoff/grammar_eval/gelib.py`, the names test). The trainer keeps it in
`_prev_tokens`, saves it in the checkpoint's `extra["prev_tokens"]` as
`{"address": …, "reader": [keys, values, valid]}` (CPU, plain containers, `weights_only`) and
restores it, so a resumed run continues bit for bit.

Safeguards against reading blind without noticing:
- `prev_tokens = None` means a stream's first window (nothing cached).
- A plain tensor (the bytes alone) is refused by the forward pass, with the fix in the message.
- `address_history` without `info` is refused.
- A checkpoint written without the key is refused on resume by name.

The window's pieces:
- **Prologue.** The reader runs once per window in `_window_prologue`, eager in the compiled
  window too, so the carry adds no compiled graph and changes none; the loop, the compiled
  block, `_position` and the event loop are the code they were.
- **Resets.** `Reader.read` masks cached positions for a row from its first document start in
  the window on, and keeps only positions of the document going on for the next window. A
  rollback discards the carry with the address bytes (the next stream starts fresh).
- **Gradient segments** (`train.tbptt_windows > 1`): the cache stays detached at every window;
  inside a segment the gradient crosses windows through the memory's state only.
- **Autocast.** The cache is stored in the parameters' dtype (a bf16 value is exact in fp32),
  so its dtype and shapes are the same at every window.

## Scope and refusals

Accepted: `model_name: hsn_t` with its memory, per-byte writes or `model.write_events=word`
(with or without `model.word_address`), `model.cell_attention_heads` 0 or more,
`train.tbptt_windows` 1 or more, the eager and the compiled autograd window.

Refused at configuration load, each with the reason:

| Configuration | Why |
| --- | --- |
| `model.key_address=false` | the cache is handed on with the address carry, which only an A5 model has |
| `model.reader_top_words > 0` | the top block's word slots are built for HSN-T without its memory only |
| `model.address_gate_input=reader_prev` | it reads `c_{t−1}`, which at a window's first column is zero, not the carried reader's output of the window before; not built |
| `train.lambda_u > 0` | the counterfactual probes replay windows with A5's bytes alone |
| `model.reader_span` set, `model.reader_position=learned` | as without the memory: the carry is itself a sliding span and needs rotary positions |

Not available, as for every HSN-T: `HSN.step`, `hardware.window = split | fused`. The
counterfactual probes and `hsn.experiments.stability` call `address_history` without `info` and
raise with the key, as they do with `model.word_address`.

## The oracle

The diagnostic that proposed the key read existing checkpoints without it: the reader's output
over a long window with a per-block span, the memory run as trained. The tests use it as the
oracle: on the same weights, windows read with `reader_carry = C` equal the reader's output
over the whole stream with a span of `C + 1`, handed in window by window (to 1e-12 in float64),
for per-byte writes and word events, with and without the attention read of the cells. For
per-byte writes the whole stream read as one window is the same too. (The note's numbers used
a span of 256, which is `C = 255`.)

What to expect of `model.reader_carry=256` on the existing checkpoints (256-byte windows). The
note's sighted readings (CPU fp32) are an upper bound: its long windows were 1,024 bytes, so a
story longer than that still had one blind start, at byte 1,024 (7 % of the dev bytes lie
beyond it), which the carry does not have.

| Checkpoint | As trained | The note's sighted reading |
| --- | --- | --- |
| Per byte, top0, 27k (lr 2.5e-3) | 0.7782 | 0.7312 |
| Per byte, top0, 27k (lr 1.25e-3) | 0.7790 | 0.7268 |
| Word events, top0, 27k (lr 1.25e-3) | 0.7971 | 0.7462 |
| Per byte, top0, 9k | 0.8731 | 0.8244 |
| Per byte, top0 + C-read, 9k | 0.8648 | 0.8196 |
| Word events, top0, 9k (lr 1.25e-3) | 0.8841 | 0.8312 |

(Word events: within a window the decoder has no selected read until the window's first word
end, carried reader or not; the note's 0.7445 and 0.8293 are readings in 1,024-byte windows.)

The same checkpoints read through this branch's code on the CPU (the key set at load, the
loop of `evaluate_windows`): `~/hsn-handoff/text_gap/score.py --job carry256_pb | carry256_wev |
carry256_pb9k | carry256_wev9k`, results in `~/hsn-handoff/text_gap/dumps/carry256_*.json`;
`carry_check.py` there prints them beside the note's. Measured so far (the jobs were not
finished when this was written), with `reader_carry=255`, which is the note's span of 256:

| Checkpoint (27k) | Stories | Below byte 1,024: this code / the note | Largest difference there | From byte 1,024: this code / the note / as trained |
| --- | --- | --- | --- | --- |
| Per byte, top0 | first 250 | 0.72744 / 0.72744 | 1.8e-5 nats (184,861 bytes) | 0.818 / 0.845 / 0.865 |
| Word events, top0 | first 750 | 0.73207 / 0.73207 | 1.9e-5 nats (555,783 bytes) | 0.870 / 0.899 / 0.929 |

So on trained weights the key reproduces the note's reading to rounding wherever the note's
reading was sighted, and is better where it was not. With `reader_carry=256` the per-byte
checkpoint reads 0.7324 on the first 250 stories (0.7326 at 255; the note 0.7341; as trained
0.7807).

## Tests

`tests/test_reader_carry_memory.py`:
- the key is opt-in; the default keeps its hash, its `info` and its carry; the recipes take it;
- the refusals above;
- carried windows equal the sighted reading (four forms), and differ from the blind one;
- a document start clears the carry (at a window's first column, inside a window, at its last
  column), and no later byte is read;
- the bytes alone, a cache of the wrong shape and a missing `info` are refused; the copies and
  the checkpoint form;
- `evaluate_windows` carries it;
- the compiled window equals the eager one over five consecutive windows (values, the next
  cache, the carried address, parameter gradients) with one graph and no recompile (four forms);
- the small recipe trains with it; the trainer holds the carry;
- a run resumed from a checkpoint continues bit for bit (also with `train.tbptt_windows=2`).

## Not run, open

- **The card.** No step time, no CUDA-graph run, no bf16 run. The reader's extra work is the
  memory-off form's (19.0 to 24.4 ms per step there at 256 × 32, the whole model); the estimate
  with the memory on is about +5 ms per step. One more cost to measure: while a divergence dump
  may still be written (`train.divergence_dump`), the step snapshot copies the carry once per
  step, 42 MB at batch 32. The cache is never written in place, so the snapshot could share
  it instead; it copies, to stay safe.
  `scripts/word_address_gpu_compile_check.py --spec <specs>/text_carry_e1.yaml --arm <arm>`
  runs an arm through the real trainer and prints the graphs, the recompiles and the step
  rate; the four `cuda` tests of `tests/test_reader_carry_memory.py` have not run.
- **Training with it.** Blind window starts are today the only thing that asks the memory to
  carry anything. A model trained with a sighted reader may train its memory less. Arms:
  `~/hsn-experiments/remote/specs/text_carry_e1.yaml` (carry against no carry, per byte and
  word events) and `text_carry_e2.yaml` (the memory-off control on the same optimiser).
- **Words older than the reader's sight.** The key does not address them; the note's E3 does.
