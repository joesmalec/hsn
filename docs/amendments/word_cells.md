# Keyed word cells (HSN-T; opt-in key, 22 Nov 2026; stage 1: the by-rule table)

Status: **stage 1 of a staged build, built on 22 Nov 2026 as an opt-in key; no training run
yet beyond two 220-step timing runs.** Branch `claude/word-cells` (on main 30bb802). The design is
`~/hsn-handoff/MEMORY-REDESIGN-2026-11-22.md` (candidate 1), which answers the diagnosis of
`~/hsn-experiments/qknorm-runs/links_probe/RESULTS.md`. The user approved it on 22 Nov 2026 with
two decisions: the identity question is accepted (below), and the build is staged.

- **Stage 1 is the table control, not an HSN memory to adopt.** It builds the keyed cells'
  machinery with the value written by rule (`model.word_cells_value=bytes`). No unit, gate or
  message of the HSN writes these cells. The arm answers one question: does the decoder learn
  to use a store that is right by rule beyond the reader's sight? The HSN's write (`word`,
  `cell`) and candidate 2 (`address`) are stage 2 and are built only if this arm is used.
- `model.word_cells: off | hash` (default `off`), `model.word_cells_value: bytes`,
  `model.word_cells_n` (1024), `model.word_cells_ways` (2), `model.word_cells_levels` (3),
  `model.word_cells_min_len` (1). A spec arm: `model.word_cells=hash` on
  `hsn_t_memory_words_carry_bal_qk.yaml` (spell `model.word_cells_value=bytes` too if the arm
  should say so; it is the default and hashes the same).
- At the default nothing changes: no module, no state, no decoder column, no `info` entry; the
  six keys are left out of the configuration hash, the arm's view and the memory key, so every
  arm id stays. A model built without the key gives main's outputs bit for bit.

## The rule

**Words** are the word address's (`write_events.md`, "The word address"): under a text context
(`model.word_contexts`) a word byte is an ASCII letter or digit, or an apostrophe directly
between two letters; a word is a maximal run of word bytes in one document, and its event is
the byte after it (one byte later when it is followed by an apostrophe and then no letter).
Under other contexts nothing is written and nothing is read.

**Keys and candidates.** A word `w` of `n ≥ min_len` bytes has the keys `k_l = (l, w_1 … w_l)`
for `l = 1 … min(L, n)`, the bytes as written (no case fold). The candidate cells of a key are
`C(k) = {((a_i k + b_i) mod (2³¹ − 1)) mod N_s : i = 1 … A}`, `a_i`, `b_i` drawn from the run
seed (`k` as the integer `l · 256^L + Σ_m w_m 256^(L−m)`).

**State.** Per row, beside the routed cells: `h^K` (`N_s × d`), per cell the key it holds `κ_i`
and the byte position of its last write `τ_i`, and the word in progress (its code sum, first
`L` bytes, length). A document start clears a row. The age of an entry is not stored: it is
`pos_t − τ_i` at the byte that reads it.

**The value** (by rule; fixed tables of ±1 from the run seed, `T` per byte, `P` per position;
positions past 16 reuse the 16th):

```
s_l(w) = (n − l)^(−1/2) · Σ_{m = l+1 … n} P_{min(m,16)} ⊙ T[w_m]            0 when n = l
```

**The write** is deferred: once per window, after the event loop. Every key of every word end
of the window asks for a cell; of a key's word ends the latest counts.

```
1. κ_i = k for a cell i of C(k):        the key writes there
2. else: i* = the cell of C(k) with the oldest τ among the cells no write of this window has
   taken (an empty cell first; ties: the lower way). Several keys on one cell: the latest
   word end takes it (at one word end the longer prefix)
3. a key that lost in 2 asks once more, as in 2; what loses again is not written
then:  h^K_i* ← s_l(w),   κ_i* ← k,   τ_i* ← pos_e
```

The value replaces the cell, which is the clear on a new key. It is one batched update of
fixed shape (`T · L` slots a row; gathers and `amax` scatters).

**The read**, at every byte `t` inside a word (a letter or digit, or an apostrophe directly
after a letter, which the prefix counts before it is known to join), with `j` bytes so far:

```
k      = (min(j, L), x_1 … x_min(j,L))
i      = the cell of C(k) with κ_i = k, as the window started          (else none)
p_t    = [U_min(j,16) h^K_i ; 1[pos_t − τ_i ≥ 256] ; log(1 + (pos_t − τ_i) / 256)]     0 when none
logits = D([q_t ; u_t ; r_t ; c_t ; p_t])
```

`U_j` is a learned `d × d` map per offset, zero at the start. Learned: `U` (16 d² = 65,536 at
`d` 64) and the decoder's `d + 2` new input columns (33,792): 3,298,105 → 3,397,433 trainable
parameters on the recipe. No gradient enters the cells.

**What a read depends on.** A byte reads entries written in earlier windows of its document
only: no later byte, no earlier document (nothing is read after a document start inside the
window), and no write of its own window, which is at most 255 bytes old and so inside the
carried reader's sight. The read therefore depends on where the windows are cut: an entry
becomes readable at the next window start. Keys, values and word ends do not depend on the
cuts. With `data.align_docs` (the recipe) training and evaluation cut every document at
multiples of the window from its start, so a document's reads are a function of the document.

## Architecture check

From the design memo (its `word` arm), with the user's decision recorded and what stage 1 does.

| Identity element | The design (`word`) | Stage 1 (`bytes`, the control) |
|---|---|---|
| Persistent cells in a fixed sparse graph, V-messages | Kept, extended: keyed cells are persistent graph nodes with fixed in-edges from routed cells | Persistent and carried, but no in-edges and no messages: not built |
| Shared unit bank, per-cell α | Kept: the keyed write runs the same units | Not used: the value is by rule (broken by design) |
| Predict-before-encode, ε as cell input | Kept: ε of the event enters the write | ε is not read by the keyed write; `P` and `L_pred` unchanged |
| Bounded convex gated update, tanh candidate, h in [−1, 1] | Kept, with a clear on a new key and the deferral | No gate: the value replaces the cell and is not bounded to [−1, 1] (broken by design) |
| Router selecting K cells | Kept for the routed cells (K = 12, unchanged). For the keyed cells the candidates are a hash of the key and the read is a tag match. **The identity question: accepted by the user on 22 Nov 2026** | As decided; and the choice among the candidates is by rule (the holder, else the oldest), not the scorer's |
| Summary q decoded by D([q; u]) | Kept: `D` gains one input, `p_t` | The same |
| Not attention over positions | Holds: a fixed-size cell set, O(1) per byte | The same |

The routed memory is untouched: the event loop, `_position`, the compiled event blocks and the
fused event block (`hardware.event_kernel=fused`) are the code they were. The keys and the read
run in the window's eager prologue, the read's map and the write in its eager epilogue, on the
eager, the compiled and the fused path alike.

## Where the cells live

`CellGraphState` is not touched (`state_schema_version` stays 1). The cells ride with the window
carry: `WindowCarry(address, reader, cells)`, `cells` a `WordCellsCarry(word, table, key, when)`.
A window puts the next one into `info` (`word_cells_word`, `_table`, `_key`, `_when`) and
`address_history(cfg, tokens, previous, info)` returns the carry, so the trainer, the evaluation
and every caller that carries the address carry the cells. A checkpoint holds them on the CPU
(`extra["prev_tokens"]["word_cells"]`), and a resumed run continues bit for bit. A carry without
the cells, cells of another shape and a checkpoint written without the key are refused by name.

Statistics, every step: `word_cells_hit` (the share of bytes inside a word that read an entry)
and `word_cells_old` (the share of those entries 256 bytes old or more), means over the step's
windows.

## Scope and refusals

Needs `model_name: hsn_t` with its memory, `model.write_events=word`, `model.word_address`
(`exact | bag`) and `model.reader_carry > 0`. Refused at configuration load, each with its
reason: the HSN, the reader alone and every baseline; `model.memory=false`; per-byte writes or
no word address; `model.reader_carry=0` (the cells ride with the carried reader's carry, and the
deferral assumes its sight); `model.cell_attention_heads > 0` (not built beside `p_t`); an empty
`model.word_contexts`; more ways than cells; a `model.word_cells_*` key moved while the key is
off; any value but `bytes`. `train.tbptt_windows > 1` is accepted (the cells are detached).

## Differences from the design memo

- **The apostrophe.** The memo's simulation split words at apostrophes; the build keeps the
  word address's words, so that a keyed write is one of the model's word events (`don't` is one
  word, keys `d`, `do`, `don`).
- **Conflicts in the batched write.** The memo gives one write per key per window; it does not
  say what happens when two keys of a window ask for one cell. Steps 2 and 3 above are the
  build's. On dev documents (a by-rule simulation on bytes with the memo's words, buckets and
  hash; `key_sim.py`'s rare-word bucket at 256-1023 bytes) the rule keeps the simulated store:
  hit / wrong 0.815 / 0.106 on bio prose (300 documents) and 0.853 / 0.093 on code native
  (80), against 0.784 / 0.122 and 0.833 / 0.103 for the memo's sequential store on the same
  documents (the difference is the deferral: an entry is not overwritten by a word of the
  window that reads it). At most 0.1 % of the key writes find no cell (0.3 to 0.9 % with one
  ask).
- **No scorer choice** among the candidates (stage 2), no level 0 (the previous-word key), no
  in-edges, and `model.word_cells_kappa` is not a key yet (it belongs to candidate 2).
- **The value's capacity.** The memo's budget table calls the `bytes` arm free of the capacity
  haircut; its own section 2.2 measures this code at 0.86 / 0.90 of the letters after the third
  at `d` 64. The control carries that loss.
- A word no longer than its key (`n = l`) stores a zero value: the read then says only that an
  entry exists, through the two age inputs.

## Tests

`tests/test_word_cells.py` (its docstring lists them): the default against main; the rule
against a per-row reference in plain Python; the cuts; what a byte may read; the carry, a
checkpoint and a bit-for-bit resume; the compiled window and the fused block's reference against
the eager window; gradients; statistics; refusals; two CUDA tests on the recipe's CUDA arm.

## Measured (22 Nov 2026)

- **Step time**, the recipe's CUDA arm with the fused event block (bf16, CUDA graphs, batch 32 x
  256, code native, the local RTX 3060 alone; the median over the steps after the first 40):
  44.7 and 45.0 ms without the key, 47.7 and 47.6 ms with it (two runs each, of 220 and 160
  steps; 10th to 90th percentile 43.9-46.7 and 46.7-49.5). The key costs about 2.8 ms a step,
  6 %; the memo's estimate was 1 to 2 ms. Peak memory 1,443 → 1,501 MiB.
- **The store fills** on code native: over steps 21 to 220 `word_cells_hit` is 0.73 to 0.79
  and `word_cells_old` 0.54 to 0.60 (three of four bytes inside a word read an entry, and more
  than half of those entries are 256 bytes old or more).
- **The first 220 training steps** of those two runs (the same seed and batches; training loss
  and next-byte accuracy over steps 201-220): 2.41 nats and 36 % without the key, 1.97 nats and
  51 % with it. The decoder starts to read the store at once. It is early training on one seed
  and no result: a reader this young cannot copy yet, and the question of the arm is what is
  left once it can (the memo: the gain to expect is beyond its sight).
- **The cells on the card** equal the rule on the CPU over the same windows: keys, write
  positions and the word in progress exactly, the values to the last bit of float32 (the card
  rounds `s / sqrt(n)` otherwise in under 1 % of the entries).
- Not run: any training beyond those steps, the swap gate of the memo's stage (c), a run
  resumed on the card.

## Where the code is

- `python/hsn/model/word_cells.py`: `WordCells` (`window`, `write`, `read`, `rates`), the carry.
- `python/hsn/model/hsn.py`: `WindowCarry.cells`, `address_history`, `clone_carry`, `pack_carry`,
  `split_packed_carry`, `packed_word_cells`; `_window_prologue`, `_window_epilogue`,
  `_event_epilogue`; `param_report`. `python/hsn/model/encoder.py`: the decoder's columns.
- `python/hsn/config.py`: the keys, the refusals, `HASH_OMIT_AT_DEFAULT`;
  `python/hsn/experiments/spec.py`: the memory key; `python/hsn/train/memory_plan.py`:
  `word_cells_param_count`; `python/hsn/train/tbptt.py`: the resume and the statistics.
