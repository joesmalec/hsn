# Amendment A14: a window reader in place of the fast cells (experimental)

Status: **implemented as an opt-in experiment on 14 Nov 2026; not adopted; no result yet.**
- Approved by the user on 14 Nov 2026 ("yes build rung 1 as an opt in amendement"), after
  "lets explore this hybrid idea even more, I feel like it is different then a straight llm
  being a hybrid and if we can manage the same percentages with the speed it is a significant
  developement." The design the user read is `HYBRID-DESIGN-2026-11-14.md`; this is its rung 1.
- `model.reader`: `none` (default) | `transformer`, with `model.reader_width` (64),
  `model.reader_layers` (2), `model.reader_heads` (4) and `model.reader_span` (null). Needs
  `model.selected_readout`. All five keys are left out of the configuration hash at their
  defaults: existing configurations, hashes and arm ids are unchanged, and with the key at
  `none` no reader is built and the model is the model it was.
- **It crosses a line on purpose.** A5's readout is described as "No sequence attention,
  additional persistent state, or computation-path weight writes". The reader is sequence
  attention. It adds no persistent state and no weight writes. Nothing of A5 changes while the
  key is off.
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
and there is no position parameter.

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
- Refused with the key on: `HSN.step` (one position has no window to read: the chat session
  and the edit probes use it) and `hardware.window = split | fused`. The baselines and the cell
  variants other than `pdg` cannot have it: it needs A5's readout, which they refuse.

## Tests (`tests/test_a14_reader.py`)

The keys are opt-in and the default keeps its hash; the refusals; no later byte, no earlier
document, the span, distances and not window positions; the closed forms; the other
parameters' initialisation and the global random stream are untouched; with the new columns
zero the window equals the window of the model without the key (float64, 1e-12); logits read
no later byte; gradients reach the reader and the new columns under the candidate's cut; `S`
is the only other way in; bf16 autocast; fast cells beside the reader; the compiled window
equals the eager one; thirty trainer steps fall.

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

**No speed is claimed by this amendment.** The routed cells still run at every byte; the
window is the slow one. Speed is rungs 2 and 3 of the design and needs its own approval.

## Results

None yet.
