# Attention read of the cells (HSN-T; opt-in candidate)

Status: **opt-in candidate, untested.** Implemented on 15 Nov 2026 on the branch
`claude/mem-attn-read` (from main f3f271a) and tested on the CPU only. No run on the card, no
result, not adopted, no amendment number.
- `model.cell_attention_heads: H` (default 0 = off). HSN-T only.
- At the default nothing changes: the key is left out of the configuration hash and of the
  experiment database's memory key, the model builds no new module and the decoder keeps its
  width. Checked against f3f271a: the eager window of two HSN-T arms (logits, state, gradients,
  initial weights; 12.5 million values) is bit-identical, and nine expanded arms keep their arm
  ids, memory keys and configuration hashes.
- It is the read half of a larger idea, "the memory as the reader's long-range keys". The write
  half is not built; the last section sketches it and lists its problems.

## Cause

On TinyStories a name seen in an earlier window costs HSN-T with memory 0.64 to 0.73 bits per
letter, the GRU 0.79 and Mamba 0.17. The memory carries a summary across windows (pronoun
gender at 96 to 99 %) but not the word.

The decoder reads the memory in two ways only:
- `q_t`, the summary;
- `r_t`, the mean of the cells the router selected at position `t` (A5).

A mean over the selected cells is one vector for 12 cells, and which cells it averages is the
router's choice for the write, not a choice made for the read. This amendment asks one
question: **is what the cells hold reachable when each byte can attend over all of them by
content?**

## The read

With `H` heads of width `e = d / H`, at every position `t` of a window:

```
qry_t        = W_Q c_t + b_Q                           [B, d]      H heads of width e
key_{t,i}    = W_K [h_{t,i} ; a_i] + b_K               [B, N, d]   H heads of width e
val_{t,i}    = W_V h_{t,i} + b_V                       [B, N, d]   H heads of width e
α_{t,i}^(j)  = softmax_i( qry_t^(j) · key_{t,i}^(j) / sqrt(e) )    over all N cells, no top-k
m_t          = W_O [ Σ_i α_{t,i}^(1) val_{t,i}^(1) ; … ; Σ_i α_{t,i}^(H) val_{t,i}^(H) ] + b_O
logits_t     = D([q_t ; u_t ; r_t ; c_t ; m_t])
```

- `c_t` is the reader's output at `t` (width `w`), the value `S` and `D` already read.
- `h_{t,i}` is the state of cell `i` **after** position `t`: the state `r_t` reads, the same
  tensor. So `m_t` sees what position `t` wrote, as `r_t` does, and nothing later.
- `a_i` is the fixed address code of cell `i` (A5's `Scorer.cell_addresses`, a buffer that is
  not trained). A head can therefore address a cell by identity as well as by content.
- Head width: `e = d / H`, so the attention is `d` wide in total and `m_t` has width `d`
  (16 per head for `d = 64`, `H = 4`). The wider choice, `e = d`, costs `H` times the largest
  matrix product of the read and was not taken.
- `m_t` is one more input of `D`, after `c_t`. `D`'s first layer gets `d` more input columns
  only when the key is set.

**What it depends on.** `m_t` is a function of `c_t` and of the cell states after `t`: the
document's bytes up to `t` and the carried state. No later byte, and after a reset no earlier
document.

**Gradient.** The read is on `r_t`'s terms.
- It reads the state with its gradient. No path of `train.feedback_detach` cuts it, because
  none cuts `r_t`: the paths are the reads of `S`, of the cell, of the scorer and of `ε`.
- Its gradient enters the state after position `t` and travels back with every other
  gradient, so `train.state_grad_clip` and `train.state_grad_ratio` meet it at the entry of
  the position like the rest.
- `c_t` keeps its gradient into the reader (it is not a state read).
- `model.decoder_shortcut_dropout` drops `q` and `u` only, so `m_t` stays, like `r_t` and `c_t`.

**Parameters.** `d (w + 4d + 4)`: 20,736 at `d = 64`, `w = 64`, plus `d · hidden` = 32,768
decoder columns; 53,504 in total on the recipe's model.
- `cell_read.query` (`w → d`), `cell_read.state` (`d → 2d`: the state half of `W_K` and `W_V`
  stacked, so one matrix product over the cells gives the keys and the values),
  `cell_read.address` (`d → d`, no bias: the address half of `W_K`), `cell_read.out` (`d → d`).
- Drawn as `model.state_init` asks (the init of the cell's own maps), biases zero, from a
  generator of their own. No other parameter's initialisation changes.
- The module is a top-level module of the model, so the recipe's update limits (`cell.*`,
  `router.*`) do not select it. It trains like the reader and the decoder.

## What is untouched

- The cells and their carried state, the graph, the units, the router and its selection, the
  address gate, the convex tanh gate, `ε`, the summary `q`, the write path and every loss.
- `r_t`: the read stands beside the router's read and does not replace it.
- The memory does not read `m_t`. With the key set, the state and the routes of a window are
  bit for bit those of the model without it (tested); only the logits differ.
- No state is added and the state layout is unchanged.
- It is a read of the memory, not sequence attention: its keys are the `N` cells of one
  position, never other positions of the window.

## Scope and refusals

- Needs `model.reader` (the queries), `model.memory`, `model.key_address` (the address codes),
  `model.selected_readout`, and `d` a multiple of `H`.
- Refused for the HSN without a reader, for `model.memory = false`, for the reader alone and
  every baseline.
- Built into the autograd window, eager and compiled: the window the HSN-T arms train on. The
  split and fused windows refuse it by name. `step()` refuses HSN-T as before.
- The forward pass has fixed shapes and no data-dependent control flow; it is traced into the
  compiled block as it stands.

## Where it is in the code

- `python/hsn/model/cell_read.py`: `CellAttentionRead`.
- `python/hsn/model/hsn.py`: `HSN._window_readout` computes `m_t` after `r_t` at every
  position, for the eager loop and the compiled block (`hsn.train.compile._block`). `m_t`
  travels behind `r_t` in the per-position readout, and `_window_epilogue` puts it after `c_t`.
- `python/hsn/model/encoder.py`: `Decoder` widens its first layer.
- `python/hsn/config.py`: the key, its checks, `HASH_OMIT_AT_DEFAULT`, the split window's
  scope. `python/hsn/experiments/spec.py`: the memory key leaves the key out at 0.
- `python/hsn/train/memory_plan.py`: `cell_read_param_count`, the `decoder` row.
- Tests: `tests/test_cell_attention_read.py`.
- Spec, not queued: `~/hsn-experiments/remote/specs/mem_read.yaml` (TinyStories, 9,000 steps,
  seed 0; a 32-byte and a whole-window reader, each with and without `H = 4`).

## Cost

Estimated on the CPU, rough: the eager fp32 window of the spec's model (`B` 32, `T` 128, two
threads), forward plus backward, with `H = 4` against `H = 0`.

| reader | without | with | the read adds |
|---|---|---|---|
| 32-byte span | 2.00 s | 2.28 s | +13.6 % (forward +10.1 %, backward +16.6 %) |
| whole window | 2.03 s | 2.29 s | +13.1 % (forward +9.8 %, backward +15.9 %) |

- Per row and position the read is 0.54 million multiply-accumulates, 16.6 % of the memory's
  3.26 million (`macs_per_token`, sparse total, the reader not counted). Nearly all of it is
  the `d → 2d` map over the 64 cells.
- It adds about nine tensor operations per position to the forward pass.
- Not measured on the card. The compiled window there is bound by launches more than by
  arithmetic, so the share can differ in either direction. If it scaled as on the CPU, main's
  0.26 s per step on synthetic data would become about 0.30 s, under the 0.5 s limit.
- Saved for the backward pass: the keys and values, about `2 N d` values per row and position
  (134 MB in bf16 at `B` 32, `T` 256).

## What the test can and cannot show

- A gain with the 32-byte reader says the cells hold the word in a form one linear key and
  value map can reach, and that the mean over the selected cells was what lost it.
- No gain does not separate "the cells do not hold the word" from "they hold it in a form this
  read cannot reach". The write is unchanged: one write per byte, into the 12 cells the router
  picks from a two-byte address.
- The read attends over cells that were never written (zero state). It can tell them apart
  only by their address code and the bias.

## Sketch of the full idea: the memory as the reader's long-range keys

Not built. For the discussion after the reset.

**The idea.**
- **Writes at word boundaries only.** A boundary is a byte that is not a letter or digit
  right after one that is; bytes ≥ 128 count as letters. The cells are written in a loop over
  those events instead of over every byte.
- **The reader's top block reads the cells.** The top block is the unrotated one (the last
  entry of `reader_rotary_pairs` is 0; that key is not in this tree, it is used by
  `~/hsn-experiments/remote/specs/text_window.yaml`). Beside the keys of its local span it
  attends over the cells' states as extra keys. The cells have no position, which is why the
  block that reads them must not rotate.

**Main problems.**

1. **The reader runs before the position loop, and this read needs cell states.**
   - Today the reader reads the whole window once in the prologue, and its output `c_t` is an
     input of `S`, so of the write. A top block that reads the cells at `t` needs the state at
     `t`, which needs `c` of the earlier positions: a cycle if the same output feeds both.
   - One way out: split the reader. The lower blocks run in the prologue as now and their
     output feeds `S` and the write. The loop runs over the events. The top block runs after
     the loop, for all positions at once, with the local span and the cell states as keys;
     its output goes to the decoder only. The memory then never reads the top block.
   - The other way, the top block inside the loop at every position, keeps one reader output
     but puts sequence attention into the compiled block and keeps the loop per byte.
   - With the split, the state between two events is constant, so each position can gather
     the state of its last event from a stack `[B, events, N, d]` kept with its gradient.

2. **Which state each query may see.**
   - A query at `t` may see the writes of events at or before `t`, nothing later. With the
     gather that holds by construction, and it needs the "no later byte" test of this read.
   - Whether the event at `t` itself is visible at `t` has to be fixed. This read says yes
     (`m_t` reads the state after position `t`).
   - A reset in the middle of a window zeroes the state before its position. Positions after
     it must not see the earlier document's cells, also when no event follows the reset.
     The simplest rule is that a reset is an event slot of its own.
   - A cell that was never written in the document holds zeros. The top block cannot tell
     "empty" from "written with a small state" unless the keys carry a written flag; the state
     has `age` but no such flag.
   - The first positions of a window have a short local span (the reader reads no earlier
     window), so there the cells are the only long-range keys. That is the case the idea is
     for, and the case with the least local evidence to form a query.
   - What a write at a boundary reads, and at which address, is open: the address is A9's
     last two bytes, which at a boundary are the end of the word and the separator.
   - `ẑ_t = P(q_{t−1})` and `L_pred` are defined per byte. With writes per event `q` changes
     per event, and the prediction loss needs a new definition. That is an equation change of
     the memory and needs its own opt-in amendment.

3. **How the compiled window paths would change.**
   - The compile unit is a block of 16 positions of `_position`. With events it becomes a
     block of 16 event slots. The prologue gathers each row's inputs at its event positions
     (`z`, the context columns, the scorer's features, the resets), so the block keeps fixed
     shapes.
   - Rows have different numbers of events. Each slot needs a per-row valid mask and a masked
     update (`h`, `q`, `age`, `pos` unchanged where the row has no such event): tensor
     operations, no branch. A closed slot still costs its compute.
   - The loop over blocks is eager Python, so the number of slots can follow the batch (its
     largest event count, rounded up to 16) without a new graph. The worst case is `T / 2`.
   - `pos` counts bytes and feeds the scorer's position feature; it must advance by the bytes
     between events, as the dense window's counters do. `age` and `model.age_max` would count
     events instead of bytes.
   - `u_t`, `r_t` and `q_t` are needed at every position for the decoder. Between events `q`
     and `r` are constant and can be gathered; `S` then runs once more over the window after
     the loop.
   - The top block after the loop gathers the key and the value projections of the cells
     per position, each `[B, T, N, w]`: 67 MB each in bf16 at `B` 32, `T` 256, `N` 64, `w` 64,
     before the backward pass.
   - The split and fused windows and their kernels are the HSN's and would refuse it. HSN-T
     has no `step()`; serving this needs a streaming reader and the event rule per byte.

**Identity.** Writes at boundaries change when the cells and the router run, and a top block
that reads the cells puts the memory inside the reader's attention. Both are equation changes:
each needs its own opt-in key with bit-identical defaults, and the cells, graph, units, router,
`ε`, the convex tanh gate and `q` have to stay what they are, so that the result is the HSN's
memory read by a reader and not a Transformer with a slot table.
