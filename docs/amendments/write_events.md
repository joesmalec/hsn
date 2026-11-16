# Writes at events: the loop over word ends (HSN-T; opt-in candidate)

Status: **opt-in candidate, untested.** Implemented on 15 Nov 2026 on the branch
`claude/mem-events` (from `claude/mem-attn-read` 1f78818) and tested on the CPU only. No run on
the card, no result, not adopted, no amendment number.
- `model.write_events: byte | word` (default `byte`). HSN-T only. A third value, `all`, is a
  test value: every byte is an event, through the same event path.
- At the default nothing changes: the key is left out of the configuration hash and of the
  experiment database's memory key, and the window code takes the path it took before.
- `model.word_address: none | exact | bag` (default `none`; 16 Nov 2026, needs
  `write_events=word`): at an event the address reads the whole word. See "The word address"
  below; with it the event rule is the word rule of that section, not the one under "Events".
- It is the write half of "the memory as the reader's long-range keys"
  (`cell_attention_read.md` is the read half). The reader's top block reading the cells is
  still not built; see the last section.

## Cause

- The memory core costs 162 of HSN-T's 168 ms per step, against about 15 ms for HSN-T without
  memory: the position loop runs once per byte.
- A CPU measurement on 15 Nov found that writes tolerate sparsity: writing only at word
  starts, on 20 % of the positions, kept 64 % of the memory's long-range benefit (random
  positions 59 %). Reads did not tolerate it, so reads stay per byte.
- A write per word is also the proposed remedy for storing words instead of byte pairs (a
  name seen in an earlier window costs HSN-T 0.64 to 0.73 bits per letter, Mamba 0.17).

The write decision of A10 masks writes but saves no compute. This amendment shortens the loop.

## Events

A position `t` of a window is an event when
- it is a **word end**: `x_t` is not an ASCII letter or digit and `x_{t−1}` is. Bytes ≥ 128
  count as letters. Every other byte is a separator, so the control and marker symbols 0 to
  31 are too (the newline marker 16, the tab marker 17, BOS, EOS, PAD); or
- a **document starts** there (`reset_t`), so that the new document's state is set up at
  that position whatever its byte is.

`x_{−1}`, the byte before a window's first, is the last byte of the window before: the first
entry of A5's carried `prev_tokens` (`address_history`), which the trainer already carries and
checkpoints for the address features. No state is added. The key therefore needs
`model.key_address`. At a reset the previous byte reads as BOS, as for the address.

`all`: every position is an event.

## The loop over events

The prologue is unchanged (embeddings, `E`, the reader over the window, the scorer's
position-independent features with the byte positions). Then:

1. **Slots.** Each row's event columns are gathered, in order, into `E` event slots.
   - `E` is the batch's largest event count rounded up to a multiple of the block, at least
     one block. The block is the compiled window's: the largest divisor of `T` up to 16 (16
     for `T` 256), in the eager window too. It is one host read per window and the only
     data-dependent shape. The loop over blocks is eager, so the number of blocks follows
     the batch without a new graph.
   - A row with fewer events has **padding slots** behind its events. A padding slot holds the
     inputs of one of the row's columns without an event (real inputs, never a reset).
2. **Each slot runs `_position` unchanged** on that column's inputs: `z`, the context columns
   with the reader's `c_t` (and `c_{t−1}` for the address gate), the scorer's features with
   the address bytes of that column and the byte position of that column, and the reset. So
   at an event the memory computes exactly what it computes at that byte today: `ẑ = P(q)`,
   `ε`, `u = S([z ; q ; ctx ; c])`, the scorer, the router's selection, the units, the convex
   tanh gate, the write and `q`'s update, from the state the previous event left.
3. **Masked update.** After a slot, `h`, `q` and `age` take the new values where the slot is
   a row's real event and keep the old ones where it is padding (`torch.where`, no branch).
   A padding slot costs its compute and changes nothing: the state, the gathered reads and
   the gradient of a row do not depend on how many padding slots the batch gives it (tested).
4. **Counters.** `pos` keeps counting bytes: a slot's state carries the byte position of its
   column, and the final `pos` is the byte position after the window's last byte. **`age`
   counts events**: a cell's age is the number of events since it was last written, and
   `model.age_max` (the refresh probe's threshold) is in events too.

The compile unit is a block of 16 event slots (`hsn.train.compile._event_block`, which calls
`HSN._event_block`); the eager window runs the same method over all slots.

## Reads: every byte, after the loop

The loop keeps, per slot and with gradient, what the decoder needs: `q` and A5's read `r`
after the slot, and with `cell_attention_heads` the cell states `h` after the slot. In front
of them stands the window's starting state. Every byte `t` then reads

```
e(t)   = the row's latest event at or before t          (the event at t itself is visible at t)
q_t    = q after e(t)                                    the window's starting q before the first event
r_t    = r of e(t)                                       zeros before the first event
m_t    = read(c_t, h after e(t))                         t's own query over the gathered cells
u_t    = S([z_t ; q before t ; ctx_t ; c_t])             q after the latest event strictly before t
logits_t = D([q_t ; u_t ; r_t ; c_t ; m_t])
```

- **Gather rule.** With `n_t` the number of the row's events at or before `t`, byte `t` reads
  entry `n_t` of the stack `[start ; slot 0 ; … ; slot E−1]`; `u_t` reads entry `n_t` minus
  one if `t` is an event. A padding slot is never read.
- **`r` before the first event is zero.** `r` is the mean of the cells the router selected at
  the event, and which cells the last event of the window before selected is not part of the
  carried state. This is a decision of the build (the alternatives: carry `r`, or read `q`
  there); it touches the bytes of a continuing document before the window's first word end.
- **`m_t`** is the per-byte content read: the query is `c_t` of byte `t`, the keys and
  values are the cells as the latest event left them (the starting cells before the first
  event). The map of the cell states runs once per slot and is gathered per byte
  (`CellAttentionRead.read_latest`).
- **`u_t`** is computed once more for all bytes after the loop. At an event that is the value
  the slot computed, and the slot's own tensor is used there, so that its gradient passes the
  position's entry (`train.state_grad_clip`, `train.state_grad_ratio`) as in the loop over
  bytes. Between events it is a new read of `q`, on `S`'s terms (`summary` of
  `train.feedback_detach` cuts it).
- **What a byte depends on.** The document's bytes up to `t` and the carried state: no later
  byte, and after a reset nothing of the earlier document, also when no word end follows the
  reset (tested).

## Losses

- The next-byte loss is unchanged: per byte, over the loss positions.
- `L_pred` is taken **at the events only**: `ẑ_e = P(q before event e)` against the target of
  the event's byte, the mean over the valid event slots whose column is not padding
  (`λ_p · mse`). With `word` the event's byte is the separator that ends the word, so the
  predictor is trained to predict that byte's target; `ε` at an event is defined as before.
  A window without any event gives a zero term.
- The balance term (`train.aux_enabled=[balance]`) is taken over the valid event slots.
- Nothing else changes. The graph term does not read positions. The cost term, the
  counterfactual probes, REINFORCE and A10's write cost are refused (below).

## What `info` and the statistics hold

- Per byte (`[B, T, …]`): `logits`, `z`, `z_target`, `u`.
- Per event slot (`[B, E, …]`): `z_hat`, `eps`, the selections (`cell_idx`, `unit_idx`,
  `valid`, `unit_logits`, `cost_per_row`) and `[E]` for the per-slot statistics.
  `event_valid [B, E]` and `event_pos [B, E]` say which slots are real and at which column.
- `valid` is cleared in padding slots, so the route guard, the unit counts and the balance
  count real events only. `eps_abs_mean`, `alpha_entropy` and `cost` are means over each
  slot's valid rows. `explored`, `refreshed` and `dense_fallback` are per-slot batch means
  that include padding slots.
- `cost` is the routed cost of an event. The compute per byte is that times the events per
  byte, plus the padding slots.
- The trainer's "active parameters" count the units per event and everything else per byte.

## Scope and refusals

- Needs `model_name: hsn_t` with `model.reader`, `model.memory`, `model.key_address` and
  `model.selected_readout`.
- Refused, each by name: every other `model_name`; `model.memory=false`; the split and fused
  windows (`hardware.window`); `model.fast_cells` / `model.fast_readout` (A8: `S` reads the
  fast group of the position before at every byte); `model.write_gate` and `train.lambda_w`
  (A10); `train.gate_mode=reinforce` (its log-probability is summed over positions); the cost
  term (`train.aux_enabled=[cost]`, `train.lambda_c`: a batch mean per position);
  `train.lambda_u` (the probes replay byte positions); `capture=True` (the diagnostics read
  per-byte cell states; run specs with `diagnostics: false`).
- **The audit after training** (`trained_state`) reads `capture=True` too, also with
  `diagnostics: false`, and both runs of the first build ended "failed" there after training
  and evaluation were done. Since 16 Nov it skips the per-byte state statistics when
  `write_events` is not `byte`: the record's `trained_state` holds the edge logits and
  `state_stats_skipped = 1.0`, and a warning is logged. Capture itself is still not built.
- Built into the autograd window, eager and compiled. HSN-T has no `step()`; serving this
  needs a streaming reader and the event rule per byte.
- Not adapted: the diagnostics of `hsn.experiments` that read per-position selections.

## The word address (`model.word_address`, 16 Nov 2026)

Status: **opt-in candidate, tested on the CPU only.** The user's decision after the first
runs: rebuild word events with the address taken from the whole word.

**Cause.** The first runs (TinyStories, 9,000 steps, seed 0) were 2.8 times faster per step
(0.074 s against 0.205 s) and worse: 1.0048 / 0.9998 bits per byte without / with the attention
read against 0.9867 / 0.9720 per byte, and 0.73 / 0.75 bits per letter on the names test
against 0.63 / 0.52 (Mamba 0.17). Inferred: at an event the address was the last two bytes,
the word's last letter and the separator, so the word never entered it.

**Words** (fixed rule, nothing learned, no formatting read). Under a text context
(`model.word_contexts`, default `[6]`, TinyStories) a word byte is an ASCII letter or digit,
or an apostrophe directly between two letters; a word is a maximal run of word bytes; every
other byte, bytes >= 128 included, is a word of one byte. Under every other context each
byte is its own word.

**Events.** Where a word is known to be complete: the byte after a word of letters; every
byte of a context that is not text; every document start. `don't` is one word and one event.
A word followed by an apostrophe and then no letter (`'a' `) closes at the byte after that
apostrophe, one byte late, because the rule does not read ahead.

**The mapping.** The key of an event is the word it closes: the word of letters that ended
just before it, or else the byte at the event (a document start, a byte of another context).

```
a_cur[i]  = √d ⟨W(key of this event), A_i⟩      a_prev[i] = √d ⟨W(key of the event before), A_i⟩
```

`a_prev` is BOS's code at a document start. This is A5 with "byte" replaced by "word": a fact
`k v` is written at `v`'s event under `k`'s code in `a_prev`, and read at `k`'s later event
under `k`'s code in `a_cur`.

```
… named lily. …    event at "." (closes lily):   a_cur = W(lily),  a_prev = W(named)   the write
… named ░░░░ …     event at " " (closes named):  a_cur = W(named)                      the read: the bytes after it read this event's state
```

and in `… lily said …` the event of `said` has `a_prev = W(lily)`.
- The separator at the event is not the key, although it is the byte the slot runs on: with
  it as `a_cur` every event would read the space's address. Separators and punctuation reach
  the memory through `z`, the reader's `c_t` and A9's byte features, not through these two.
- A9's features (`address_ngram > 1`) still read the last bytes, as in the first build.
- With every byte its own word and an event, both features are A5's bit for bit, and the
  model is `write_events=all`, which is the loop over bytes (tested with `torch.equal`).

**Codes.** `f` are the word's bytes with A-Z folded to a-z (text contexts only; the embedding
and the reader still read the capital, and nothing is allocated by capitalisation).

```
exact:  W = d^(−1/2) · s_1[f_1] ⊙ s_2[f_2] ⊙ … ⊙ s_j[f_j]        positions past 16 reuse s_16
bag:    W = unit( Σ t_1[a] ⊙ t_2[b]  +  Σ u_1[a] ⊙ u_2[b] ⊙ u_3[c] )   over the 2- and 3-grams of  ^ f_1 … f_j $
```

A word of one byte keeps A5's `C[f_1]` under both.

**Fixed and learned.** Fixed: the ±1 tables (`word_exact`, or `word_bag2` and `word_bag3`;
integer buffers drawn from the run seed after everything A5 and A9 draw, so every other
buffer and parameter starts as without the key), the byte classes, the fold, the list of
text contexts, `C` and `A`. Learned: only what A5 already learns, the units' weights on the
two address columns and the address gate `G`. No parameter is added. A word no run has seen
has a unit code like any other.

**Computed before the loop, carried outside the state.** Both codes are running sums over a
word's bytes (sign bits modulo 2, or gram codes), so a window is one cumulative sum in the
prologue and the loop is unchanged. A window hands the next one `2d + 6` integers per row
behind A5 / A9's bytes: the running sum, length and last two bytes of the word in progress,
whether it ended on an apostrophe not yet placed, and the last event's key.
`info["address_carry"]` is the next window's `prev_tokens`; `address_history(cfg, tokens,
previous, info)` returns it, and the trainer checkpoints it as it does the bytes. Cutting a
window anywhere gives the same events and codes (tested down to windows of one byte).

**Tests** (`tests/test_word_address.py`, 22): the rule against a per-row reading of it;
`none` against the outputs written at 1096531; another context bit for bit; one address for
`Lily`, `LILY` and `lily`; the carry, also through the trainer and a checkpoint; no later byte and no earlier document; the
configuration and its hash; `train_eval` end to end with the audit.

**Cost, CPU** (as the table below: eager fp32, `B` 32, `T` 128, two threads, forward plus
backward, on a loaded machine, so compare within the row only):

| | byte address | `exact` | `bag` |
|---|---|---|---|
| without the attention read | 0.83 s | 0.76 s | 0.76 s |
| `cell_attention_heads=4` | 1.34 s | 1.28 s | |

The prologue goes from 32.6 to 38.4 ms (`bag` 39.6); the loop has the same 32 slots (25.5
events per row against 25.8: a word with an apostrophe is one event). Not run on the card.

**Not built / open.** A9's pair features for words. The counterfactual probes and
`hsn.experiments.stability` call `address_history` without `info` and raise with the key. A
script that carries `prev_tokens` itself (the names test) must use the call above. Bytes >=
128 are words of one byte here and counted as letters in the first build's rule. Spec, not
queued: `~/hsn-experiments/remote/specs/mem_events2.yaml`.

## What is untouched

The cells, the router and its selection, the units, the address gate, the convex tanh gate,
`ε`, the graph, `q`'s update and the write equations. They run as they are, at events only.
The state layout is unchanged. The memory is still written through the router into the cells
it selects; it is not a slot table, and the event slots are a batching device of the loop, not
a store.

## The anchor

`model.write_events=all` builds every byte as an event through the slot gather, the masked
update and the per-byte gathers. It must equal the loop over bytes, and does, in float64 on
the CPU (`tests/test_write_events.py`):

| | largest difference, `all` against `byte` |
|---|---|
| logits | 1.3e-15 |
| final state (`h`, `q`; `age` and `pos` exact) | 0 |
| gradients (every parameter and the carried `h`, `q`) | 3.6e-14 absolute, 1.5e-15 relative |

- Eleven configurations: the soft warm-up and the hard gate, no cut and the candidate's cut,
  with and without the attention read, two-byte addresses with the n-gram history, the
  short-span reader, the address gate reading `c_{t−1}`, `train.state_grad_clip` and
  `train.state_grad_ratio`, and a 40-byte window. Training and evaluation mode for the
  forward pass.
- Every window has resets in the middle of two rows, rows that continue a document from a
  random carried state, and a carried last byte.
- Without the attention read the logits are bit-identical: the gathers copy, and `u` at an
  event is the slot's tensor.
- The compiled window against the eager one (fp32, `word` and `all`, soft, hard and
  evaluation) agrees within the tolerances of the compile tests, and a window with more
  blocks of slots runs on the same graph.

## Cost

Estimated on the CPU, rough: the eager fp32 window of the spec's model (`mem_events.yaml`:
64 cells, 12 routed slots, 32-byte reader) on TinyStories bytes, `B` 32, `T` 128, two threads,
forward plus backward.

| | `byte` | `word` | `word` / `byte` |
|---|---|---|---|
| without the attention read | 2.07 s | 0.66 s | 0.32 (forward 0.31, backward 0.32) |
| `cell_attention_heads=4` | 2.31 s | 1.11 s | 0.48 (forward 0.43, backward 0.52) |

- 20.5 % of the bytes are events (26.2 per row of 128; 21 to 31), and the slots are 25 % of
  the bytes (32): the loop is a quarter as long. What remains is the reader, `S` and `D` over
  all bytes and the gathers.
- With the attention read the per-byte read after the loop costs 0.45 s, against 0.24 s
  inside the loop over bytes. It is one batched read over `[B, T, N, 2d]` gathered keys and
  values (134 MB in fp32 at this size) and on the CPU it is bound by memory traffic. The
  map of the cell states runs once per slot; the gather and the attention run per byte.
- Not measured on the card. There the loop is bound by launches, which fall with the number
  of slots, and the batched read is a few launches; so the ratio can be better than on the
  CPU, most of all with the read. Padding slots are paid: a batch runs as many slots as its
  longest row needs.
- Saved for the backward pass, beyond the loop's own: the per-slot cell states `[B, E, N, d]`
  with the read, and the gathered keys and values `[B, T, N, 2d]` (134 MB in bf16 at `B` 32,
  `T` 256).
- A tiny TinyStories run through the trainer (40 steps, `word`, 4 heads): the training loss
  fell from 5.54 to 3.16 and the dev loss from 3.84 to 3.07; the same run at `byte` 5.54 to
  3.15 and 4.14 to 3.06. It shows that it trains, nothing else.

## Where it is in the code

- `python/hsn/config.py`: the key, its checks, `write_events_refusals`,
  `HASH_OMIT_AT_DEFAULT`, the split window's scope. `python/hsn/experiments/spec.py`: the
  memory key leaves the key out at `byte`.
- `python/hsn/model/hsn.py`: `event_positions`, `event_block`, `_Events`, `HSN._event_slots`,
  `HSN._event_block`, `HSN._forward_events`, `HSN._event_epilogue`; `_window_prologue` finds
  the events; `_window_epilogue` takes the valid mask for the statistics.
- `python/hsn/model/cell_read.py`: `CellAttentionRead.read_latest`.
- `python/hsn/train/compile.py`: `_event_block`, `CompiledWindow._run_events`.
- `python/hsn/losses.py`: `L_pred` and the balance at events.
  `python/hsn/train/tbptt.py`: `_slot_view` for the route guard and the unit counts.
- `python/hsn/model/word_address.py` (the rule, the codes, the carry), `Scorer.word_codes`
  and the `codes` argument of `Scorer.address_features`, `HSN._split_address_carry`,
  `address_history(…, info)`; `hsn.experiments.diagnostics.trained_state` (the audit).
- Tests: `tests/test_write_events.py`, `tests/test_word_address.py`.
- Spec, not queued: `~/hsn-experiments/remote/specs/mem_events.yaml`.

## What is left of the full idea

- **The reader's top block reading the cells.** With writes at events the state between two
  events is constant and kept per slot with its gradient, which is what the sketch in
  `cell_attention_read.md` needs: the top (unrotated) block can run after the loop with the
  local span and the gathered cells as keys, its output going to the decoder only. Not
  built. The attention read `m_t` is the nearest thing that exists: one query per byte over
  the cells of the latest event, outside the reader.
- **A written flag.** A cell never written in the document holds zeros; the read cannot tell
  "empty" from "written with a small state".
- **What a write at a word end reads.** The slot runs on the event byte's inputs: `z` of the
  separator and `c_t` of the reader at the separator (which has read the word). Its address
  was the last two bytes; `model.word_address` gives the word an address of its own (above),
  untested on the card.
- **`L_pred` at events** predicts the separator's target. A target that describes the next
  word would be an equation change of its own.
- **The card.** Not run: the step time of the compiled event loop, the memory of the kept
  cell states (`[B, E, N, d]` per window with gradient, and the gathered keys and values
  `[B, T, N, 2d]` of the read), and CUDA-graph recording with a varying number of blocks per
  window.
