# Keyed word cells (HSN-T; opt-in key, 22 Nov 2026; stage 1: the by-rule table; stage 2: the HSN's write)

Status: **both stages built on 22 Nov 2026 as opt-in keys. Stage 1 (the by-rule table) has its
9k screens (below). Stage 2 (the HSN's own write, `word` and `cell`, and candidate 2,
`address`) was built untrained on 22 Nov; its screens and a 27k-window run followed on 23 Nov
([Trained](#trained-23-nov-2026)): the HSN's own write (`word`) with `train.tbptt_windows=3`
passes the 27k bar on both sets and ties the table.** Stage 2 is
described in ["Stage 2"](#stage-2-the-hsns-write-and-candidate-2) at the end; the sections
before it are stage 1's, as written. Branch `claude/word-cells` (on main 30bb802). The design is
`~/hsn-handoff/MEMORY-REDESIGN-2026-11-22.md` (candidate 1), which answers the diagnosis of
`~/hsn-experiments/qknorm-runs/links_probe/RESULTS.md`. The user approved it on 22 Nov 2026 with
two decisions: the identity question is accepted (below), and the build is staged.

- **Stage 1 is the table control, not an HSN memory to adopt.** It builds the keyed cells'
  machinery with the value written by rule (`model.word_cells_value=bytes`). No unit, gate or
  message of the HSN writes these cells. The arm answers one question: does the decoder learn
  to use a store that is right by rule beyond the reader's sight? The HSN's write (`word`,
  `cell`) and candidate 2 (`address`) are stage 2 and are built only if this arm is used. It
  was (9k steps, seed 0: swap Delta past 256 bytes +0.748 code native, +0.912 bio prose, all of
  it in the transplanted keyed cells), so stage 2 is built.
- `model.word_cells: off | hash` (default `off`), `model.word_cells_value: bytes`,
  `model.word_cells_n` (1024), `model.word_cells_ways` (2), `model.word_cells_levels` (3),
  `model.word_cells_min_len` (1). A spec arm: `model.word_cells=hash` on
  `hsn_t_memory_words_carry_bal_qk.yaml` (spell `model.word_cells_value=bytes` too if the arm
  should say so; it is the default and hashes the same).
- At the default nothing changes: no module, no state, no decoder column, no `info` entry; the
  six keys (eight with stage 2's) are left out of the configuration hash, the arm's view and
  the memory key, so every arm id stays. A model built without the key gives main's outputs
  bit for bit.

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

## Stage 2: the HSN's write and candidate 2

Built on 22 Nov 2026 after stage 1's table was used; **untrained**. The question it is built to
answer: can the HSN's own write fill these cells as usefully as the rule does? If only `bytes`
works, the part that helps is a table and not the HSN, which is a decision for the user. So
the write below is the cell update of the identity with nothing tuned toward the table.

Keys (all opt-in; at their defaults every stage 1 arm keeps its outputs, bit for bit, and its
hash, arm view and memory key):

- `model.word_cells_value: bytes | word | cell` (default `bytes`);
  `model.word_cells_gate_bias` (3.0; where `beta_K` starts; read only with `word | cell`).
- `model.word_cells: off | hash | address`; `model.word_cells_kappa` (4.0; read only with
  `address`).
- Spec arms on `hsn_t_memory_words_carry_bal_qk.yaml`:
  - `model.word_cells=hash`, `model.word_cells_value=word` (or `cell`);
  - `model.word_cells=address`, `model.word_cells_n=256`, `model.word_cells_value=word` (or
    `cell`); the default of `word_cells_n` stays 1,024, so the arm spells the memo's 256;
  - the cross-window gradient: `train.tbptt_windows=3`, `train.grad_accum=3`,
    `hardware.compile_mode=default` (a segment cannot run under CUDA graphs); its detached
    twin is `train.grad_accum=3` alone.

### The write (`word`, `cell`)

Deferred as in stage 1: once per key per window, after the event loop, with the model's values
at the key's last word end `e` of the window (`u_e`, the error `eps_e` as the cell reads it,
the scorer's scalars at `e`, the routed cells as event `e` left them).

```
1. kappa_i = k for a cell i of C(k):   the key writes there                        (by rule)
2. else: i* = the scorer's best cell of C(k) among the cells no write of this window has taken:
   max_j of its logits on [u_e ; eps_e ; h^K_i ; age_i ; the event's scalars], ties to the lower
   way. Several keys on one cell: the latest word end takes it (the longer prefix)
3. a key that lost in 2 asks once more, as in 2; what loses again is not written
then, for the cell i* of key k = (l, ..) of word w:
   h      = h^K_i*, or 0 when kappa_i* != k                         (the clear on a new key)
   m      = sum_{j in N(i*)} a_{i*j} V h_{j,e}                       D_g fixed in-edges from routed cells
   v      = RMSNorm([u_e + V_s s_l(w) ; eps_e ; h ; m])              cell: without V_s s_l(w)
   cand   = tanh(alpha * RMSNorm_g(W_out F_j*(v)))                   j* = argmax_j of the cell's logits
   b      = sigmoid(B v + beta + beta_K)
   h^K_i* <- (1 - b) * h + b * cand ;   kappa_i* <- k ;   tau_i* <- pos_e
```

- `RMSNorm`, the units `F`, `W_out`, `RMSNorm_g`, `V`, the gate `B` with its bias `beta` and the
  scorer are the routed cells' own modules. New parameters: the in-edge logits (`N_s x D_g`,
  zero at the start, normalised as `model.edge_norm`), `beta_K` (`d`, starts at 3) and, for
  `word`, `V_s` (`d x d`, `model.state_init`): 4,160 (`cell`) and 8,256 (`word`) at the
  recipe's sizes, 3,401,593 and 3,405,689 trainable parameters in all.
- `alpha` is 1 forward and carries the straight-through gradient of the choice, as in a
  routed slot (`train.unit_credit` for the unit; the cell against the midpoint to the best free
  candidate it was preferred to, when there was one). In evaluation it is the constant 1.
- The scorer's features of a keyed cell: the two running averages and the address features
  are zero, and the age is `log(1 + min(age, 4096) / 256)` with the age in bytes since the
  cell's last write (a cell that holds nothing: 4,096). `train.feedback_detach`'s `router` cut
  applies as it does to a routed cell (the features are constants in the backward pass).
- The read is stage 1's (the tag match, `U`, the two age inputs). The cells stay in `[-1, 1]`.

### The gradient into the write

A write is read from the next window on, so with every window detached (the default) no loss
reaches it: `V_s`, `beta_K` and the in-edge logits then stay at their start for the whole run,
and the write runs on the shared units as the routed path trains them. With
`train.tbptt_windows = K > 1` the trainer carries the cells with their gradient inside a
segment (`address_history(..., attached=True)`; the last window of a segment hands them on
detached), and the later windows' reads train the write's own parameters, the units, the gate,
`W_out`, `V`, the scorer (through `alpha`) and, through `u_e`, `S` and the reader. The units a
keyed write ran count as active for lazy AdamW in the windows whose cells stay attached.

### Candidate 2 (`model.word_cells=address`)

No hash and no tag. Every cell has a fixed address `A_i` (drawn as A5's cell addresses), and
`W_l(k)` is the exact word code (`model.word_address=exact`: A-Z folded, one byte reads A5's
byte code) of the prefix `w_1 .. w_l`.

```
rank_i = max_j logit_ij + kappa * sqrt(d) * <W_l(k), A_i>           logit: the scorer's, as above
write:  the best-ranked cell of all N_s (K_s = 1 per level), by the turns 2 and 3 above;
        the cell is updated as above, without a clear
read:   at every byte inside a word, the K_r = 2 best-ranked cells by the values at that byte
        and the code of the prefix so far (level min(j, L)); nothing is written
p_t   = [U_min(j,16) (h_1 + h_2) / 2 ; the means of 1[age >= 256] and log(1 + age / 256)
         over the read cells that hold a write]
```

The read's selection carries the straight-through weight of its rank in training. `eps` at a
byte that is not an event is `z_t` less the prediction of the summary before `t`.

### Paths

All three arms run on the eager, the compiled and the fused event loop
(`hardware.event_kernel=fused`), and with `train.tbptt_windows > 1` on each. The write and the
reads are eager code behind the loop on every path. The event loop and the kernel are not
edited: the compiled blocks hand on the cells of every slot (the output the attention read of
the cells already had), and the fused block's Python wrapper takes them from its tape (the
state the next slot read).

### Differences from the design memo

- **The messages read the routed cells as values.** No gradient passes from a keyed cell into
  the routed cells through `m` (`V` and the in-edge weights are trained by it). The fused
  block's reverse sweep takes a cotangent at a slot's read-out and at the block's last state
  only, so giving `m` that gradient needs a kernel change; the unfused paths cut it too, so
  that the three paths are one model. The keyed cells reach the routed core's training
  through the shared units, gate, `W_out`, `V` and scorer, and through `u_e`.
- **`beta_K` starts at 3, as the memo writes it, on top of the cell's own gate bias.** With the
  recipe's `model.gate_bias_init = -3` a keyed gate therefore starts near `sigmoid(0) = 0.5`,
  not near 0.95. The memo's "a keyed write takes the cell" needs a start of 6 there, which
  `model.word_cells_gate_bias=6` gives (a key the memo does not have; the default is the
  memo's 3). Without the cross-window gradient `beta_K` never moves, and the keyed gate
  follows the routed cells' bias. `word_cells_gate` in the step statistics is the mean keyed
  gate.
- **The choice among the candidates** is by the scorer in steps 2 and 3 only; a key that a
  cell holds goes there by rule (the memo's rule). `alpha` carries the cell's straight-through
  weight beside the unit's, which the memo does not spell out.
- **The age feature** of the scorer is defined here (the memo: "age among them").
- **Candidate 2:** the age inputs of `p_t` are means over the read cells that hold a write; a
  cell is never cleared (there is no tag to compare); the prior's code folds the case, as the
  word address does, where the hashed keys are the bytes as written; one write per key per
  window and the conflict rule are stage 1's (the key as an integer is used for that and
  nowhere else). `model.word_cells=address` needs a learned value: with `bytes` it is refused.
- Not built: level 0 (the previous word's key), two cells an entry.

### Refusals (stage 2)

`word | cell` need `model.scorer=linear` and `model.routing=topk` (the write splits the scorer's
linear map by feature) and no `model.ablate_shared_u`. `address` needs
`model.word_address=exact` and `word | cell`, and refuses a moved `model.word_cells_ways`;
`hash` refuses a moved `model.word_cells_kappa`, and `bytes` a moved
`model.word_cells_gate_bias`.

### Statistics (stage 2)

Beside `word_cells_hit` and `word_cells_old`: `word_cells_gate` (the mean gate of the window's
keyed writes) and, with `address`, `word_cells_prior` (the share of the keyed writes that took
the cell the address prior alone ranks first: 1.0 means the selection is the soft hash).

### Tests (stage 2)

`tests/test_word_cells_write.py` (its docstring lists them) and stage 1's tests with the arm as
a parameter: `bytes` against d3537f4 (`tests/fixtures/word_cells_bytes_d3537f4.pt`: names,
weights, four carried windows, gradients, the recipe's ids); the write and candidate 2's read
against a reference in plain Python that scores with the scorer's own map; the gradient
through the carried cells, attached and detached; the trainer's segments; the compiled window
and the fused block's reference against the eager trainer (also across a segment); a resume;
on the card bf16 steps of the recipe's arm on both event loops with and without segments, and
the kernels against the compiled blocks in fp32.

Run on 22 Nov 2026 at the stage 2 commit: the CPU suite with the card hidden (3,163 passed,
560 skipped); on the card the ten CUDA tests of `tests/test_word_cells.py` and 45 existing CUDA
tests around what stage 2 touches (the fused block and its trainer, the compiled window, the
carried reader, the round products, the per-byte speed keys, reach dropout), all passed.

### Measured (stage 2, 22 Nov 2026)

Timing runs only: the recipe's CUDA arm with the fused event block on code native, batch 32 x
256, bf16, the local RTX 3060 alone, seed 0; the median over the steps after the first 40,
statistics steps left out (`~/hsn-experiments/wcells-stage2/`: `step_time.py`, `time_*.json`).

| Arm | Step | Peak memory |
|---|---|---|
| no key (stage 1's measurement) | 44.7 - 45.0 ms | 1,443 MiB |
| `hash`, `bytes` (220 steps; stage 1: 47.6 - 47.7) | 47.9 ms | 1,501 MiB |
| `hash`, `cell` (220 steps) | 52.1 ms | 1,578 MiB |
| `hash`, `word` (eight runs of 150 to 220 steps; one of 1,500: 52.7) | 51.9 - 52.5 ms | 1,577 MiB (2,715 after 1,500 steps) |
| `address`, `word`, 256 cells (220 steps; 1,500 steps: 59.6) | 59.3 ms | 1,688 MiB (2,917 after 1,500 steps) |
| `hash`, `word`, `train.tbptt_windows=3` (130 steps of three windows, compile default) | 160.1 ms a step, 53.4 a window | 3,281 MiB |
| `address`, `word`, 256 cells, `train.tbptt_windows=3` (130 steps) | 181.7 ms a step, 60.6 a window | 3,594 MiB |

- The HSN's write costs 4.4 ms over the table (7.5 over no key; the memo's estimate was 3 to
  6), candidate 2 another 7 (its read scores 256 cells at every byte). With the fused block
  the three-window step is 160 ms, against the 250 to 280 the memo took from the unfused
  recipe.
- **The keyed gate starts half open**, as the note on `beta_K` says: `word_cells_gate` is
  0.51 at the first step. It rises to 0.75 to 0.78 within 130 to 220 steps in every arm, with
  and without the cross-window gradient (the routed cells' gate moves it).
- `address`: 0.91 to 0.97 of the bytes inside a word read a cell that holds a write, 0.80 of
  those reads are 256 bytes old or more (the two cells are read whatever was written there:
  there is no tag), and 0.79 to 0.83 of the writes take the cell the address prior alone ranks
  first, from the first step on (a cell already taken in the window sends a key elsewhere).
- **The first steps are no result**, but they differ from the table's: the mean training loss
  over steps 201 to 220 is 1.96 nats with `bytes` (stage 1: 2.41 without the key, 1.97 with
  it) and 2.33 to 2.35 with `word`, `cell` and `address` without the cross-window gradient.
  The decoder reads the table at once; what the HSN's write stores is of little use to it
  this early. One seed, 220 steps.
- `beta_K` stays at 3.0 without the cross-window gradient and has moved to 3.02 after 130
  steps with it; the cells stay within [-1, 1]; after 220 steps 26 % of the 1,024 hashed
  cells of a row hold a write, and 60 to 64 % of the 256 addressed ones.
- **One timing run failed and the failure was not reproduced.** The first `word` run (CUDA
  graphs, fused block, started right after the `bytes` run) stopped after 146 s with
  `cudaErrorStreamCaptureInvalidated` at the end of a CUDA graph capture. My job script kept
  only the last lines, so which capture it was (the compiled blocks' or the kernels') is not
  known. Eight further `word` runs (one with an empty compile cache, as that run partly had;
  one of 1,500 steps, with 15 statistics steps and 45 captures of the kernels) and the `cell`
  and `address` runs finished. Whether stage 2 can cause it is open.
- Not run: any training beyond those steps, a swap test, a run resumed on the card.

### For a swap test of these arms

- `swap_eval.py` and `transplant_wc.py --what keyed` read the carry, whose layout is stage
  1's (`word`, `table`, `key`, `when`), so they run as they are on a tree at this commit. With
  `address` the keys are all empty (-1) and the exchange of `table` and `when` is the whole
  store.
- A transplant of the keyed cells exchanges what earlier writes left. With `word | cell` a
  later write to a held key also reads the transplanted cell (the convex update), which the
  by-rule arm's does not.
- `--what hsn` (the routed cells) now also changes what later keyed writes read, through `m`.

## Trained (23 Nov 2026)

All runs: `hsn_t_memory_words_carry_bal_qk.yaml` with the field's data and word contexts, the
local RTX 3060, seed 0, one run each. Bits are dev bits per byte, accuracy next-byte accuracy,
step times alone on the card. **Swap Δ** (`links_probe/swap_eval.py`): the extra bits on the
bytes after a target's first occurrence when the carry is exchanged with another document's,
on targets 257 to 1,023 bytes back (400 targets), with a paired bootstrap of 4,000 draws;
"share of table" is Δ over the same-schedule table's Δ. **Keyed transplant**
(`transplant_wc.py --what keyed`): Δ when only the keyed cells are exchanged. The specs are
`~/hsn-experiments/remote/specs/fields_9k_wcells_*.yaml` and `fields_27k_wcells_*.yaml`;
results, logs and scoring under `~/hsn-experiments/wcells-runs/`.

Schedules: **X3** = `hardware.compile_mode=default train.grad_accum=3
train.tbptt_windows=3` (the cross-window gradient into the write); **G3** = the same with
`train.tbptt_windows=1` (each window detached). The 27k bar is memory-off 160 x 8 at 27k steps
(code native 0.994 / 80.52, bio prose 1.282 / 73.32).

### Stage 2 screens (3,000 steps x 3 windows)

Compare within this schedule only. The code native arms with X3 ran out of memory on the fused
event block (fixed by 2b886b3 and 89835a4, `docs/wev_fused_event.md`) and ran on
`hardware.event_kernel=compiled`; their step times are not the fused ones.

| Set | Arm | Bits / acc | Swap Δ | Share of table | Keyed transplant | Step (3 windows) |
|---|---|---|---|---|---|---|
| bio prose | table (`bytes`), G3 | 1.397 / 71.20 | +0.953 | 1.00 | | 0.144 s |
| bio prose | **`word`, X3** | 1.400 / 71.09 | +0.942 | 0.99 | 0.944 | 0.171 s |
| bio prose | `word`, G3 | 1.438 / 70.34 | +0.254 | 0.27 | | 0.158 s |
| bio prose | `address`, X3 | 1.422 / 70.63 | +0.529 | 0.56 | | 0.194 s |
| bio prose | no key, X3 | 1.444 / 70.15 | +0.002 | | | 0.138 s |
| code native | table (`bytes`), G3 | 1.121 / 78.54 | +0.784 | 1.00 | | 0.145 s |
| code native | **`word`, X3** (compiled) | 1.108 / 78.75 | +0.684 | 0.87 | 0.685 | 0.316 s |
| code native | `word`, G3 | 1.152 / 77.90 | +0.131 | 0.17 | | 0.160 s |
| code native | `address`, X3 (compiled) | 1.142 / 78.11 | +0.274 | 0.35 | 0.274 | 0.322 s |
| code native | no key, X3 (compiled) | 1.163 / 77.62 | +0.004 | | | 0.276 s |

- The HSN's write reaches the table's recall only with the cross-window gradient (X3: 0.99 bio,
  0.87 code; detached: 0.27, 0.17).
- Candidate 2 (`address`) is weaker on both sets (0.56, 0.35) and in bits.
- The cross-window gradient without keyed cells (candidate 3) gives no recall.

### The 27k-window run (9,000 steps x 3 windows, fused event block)

| Set | Arm | Bits / acc | Swap Δ | Share of table | Keyed transplant | Step (3 windows) |
|---|---|---|---|---|---|---|
| code native | **`word`, X3** | **0.971 / 80.98** | +0.734 | 1.07 | 0.727 | 0.190 s |
| code native | table (`bytes`), G3 | 0.988 / 80.75 | +0.683 | 1.00 | 0.680 | 0.151 s |
| code native | memory-off 160 x 8, `grad_accum=3` | 1.029 / 79.93 | 0.000 | | | 0.090 s |
| code native | stage 1 table, 27k steps x 1 window | 0.976 / 81.00 | +0.665 | 0.97 | 0.669 | |
| bio prose | **`word`, X3** | **1.244 / 74.17** | +0.908 | 1.09 | 0.904 | 0.172 s |
| bio prose | table (`bytes`), G3 | 1.251 / 74.06 | +0.835 | 1.00 | 0.834 | 0.145 s |
| bio prose | memory-off 160 x 8, `grad_accum=3` | 1.313 / 72.72 | 0.000 | | | 0.086 s |
| bio prose | stage 1 table, 27k steps x 1 window | 1.244 / 74.20 | +0.846 | 1.01 | 0.844 | |

- **The HSN's own write passes the 27k bar** on both sets: 0.023 bits better on code native,
  0.038 on bio prose (accuracy +0.46 and +0.85 points), and beats memory-off on the same
  schedule by 0.058 and 0.069 bits.
- **It ties the table in bits** (0.017 and 0.007 better on one seed; the recipe is not
  bit-reproducible from run to run, so no claim that it beats the table is made) and recalls
  slightly more beyond 256 bytes (paired Δ minus the table's: +0.051 [+0.003, +0.096] code,
  +0.072 [+0.025, +0.121] bio). The keyed cells carry the whole of it (transplant 0.727 of
  0.734, 0.904 of 0.908).
- **It costs 19 to 26 % a step over the table** (0.190 against 0.151 s, 0.172 against 0.145).
- Memory-off 160 x 8 runs with `hardware.baseline_compile=default` under `train.grad_accum=3`
  (reduce-overhead's CUDA graphs refuse it); its step is not comparable with the HSN-T arms'
  (another speed class).
