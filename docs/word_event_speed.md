# Word-event step time on the RTX 3060: where it goes, and three changes

17 November 2026. Branch `claude/word-event-speed`, from `b32f663`. Measured on the RTX 3060
(`profile:rtx3060`), alone on the card in a plain CUDA context, torch 2.15.0+cu132. The
configuration is the target of the job: `configs/experiments/hsn_t_memory_words.yaml` with the
top0 reader on TinyStories (word events, exact word address, B 32, T 256, bf16,
`hardware.compile_mode=reduce-overhead`, `hardware.graph_alias_index=true`; arm
`1888ced675259be1`).

## What this is

The job asked for the word-event memory path at or under 50 ms per step without changing
results. **It is not there: 69.4 ms became 61.1 ms.** What was found and built:

| | Median step, ms | Against main |
| --- | ---: | ---: |
| main (`b32f663`), run checkout as it is used | 69.4 | |
| + dense event-block inputs (no key; same values) | 68.3 | −1.1 |
| + the trainer kernels built in the checkout (no code change; same values) | 65.0 | −4.4 |
| + `hardware.round_products=merged` (opt-in; equal to rounding) | 61.1 | −8.3 (−12 %) |

Each row is the median of the per-run medians of 2 to 5 runs of 400 steps, base and changes
alternated ("Timings"). One compiled graph and no recompile in every run.

1. **Dense event-block inputs.** Always on, for every word-event run. A block of event slots
   is handed to its compiled graph as a dense copy; it was a slice of slots padded to the
   window's width, which a CUDA graph copies by its byte extent.
2. **The trainer kernels were never built in the run checkouts.** No code change. The step
   after `backward()` has run on its PyTorch reference passes in every HSN-T run so far.
3. **`hardware.round_products=merged`.** Opt-in. The compiled sparse round takes 21 matrix
   products per event where it took 33, and runs the unit bank's activation and second layer
   for the chosen unit only.

The measured profile says why 50 ms is out of reach for changes of this kind, and what would
be needed ("What is left").

## Where the step goes

`torch.profiler` (CPU and CUDA activities) over 5 steps of the training loop on main's code,
trainer kernels absent. The device times are sums of kernel durations, which do not depend on
the host; the wall time of a profiled step does and is not a step time.

| Part of a step | Device time, ms | Kernels |
| --- | ---: | ---: |
| Event loop, forward: 4 graph replays of 16 slots | 13.7 | 3,660 |
| Event loop, backward: 4 graph replays | 22.9 | 4,388 |
| Reader, forward (eager) | 3.95 | 148 |
| Backward after the loop: reader, embeddings, prologue (eager) | 9.65 | 366 |
| Optimiser step, clip, update caps, weight average | 3.24 | 558 |
| Prologue without the reader (scorer features, word address) | 2.11 | 226 |
| Backward before the loop: losses, decoder, per-byte reads (eager) | 1.59 | 104 |
| Epilogue (per-byte reads, `S` over all bytes, decoder) | 1.24 | 91 |
| Block inputs: padding to `[B, T, …]` and the graphs' input copies | 1.22 | 57 |
| The rest (between the backward replays, losses, statistics, slots) | 1.1 | |
| **Device busy** | **60.7** | |

- The step is 69.4 ms, so the card is idle for about 9 ms of it (87 to 88 % utilisation by
  NVML during the timing runs). Most of that is the host issuing small kernels: the
  optimiser's passes, the snapshot and the loader, the word address.
- **The event loop is 36.6 ms, 60 % of the device time**: 572 µs per event slot, 3.43 ms per
  forward replay and 5.7 ms per backward replay. It is not bound by launches (a replay is one
  call); it is the kernels' own time.
- **The reader is 13.6 ms.** It runs eager, forward and backward, with the masked
  memory-efficient attention kernel (0.57 ms forward, 1.8 ms backward).
- The step is sensitive to the host: with other jobs loading the CPUs (load average 17 to 22)
  the same code took 91.7 ms.

Inside an event slot (µs per slot, 64 slots per step):

| Kernel | Forward | Backward |
| --- | ---: | ---: |
| Unit bank, first layer, all 8 units: `[384, 256] × [256, 4096]` | 41.5 | 59.8 + 37.7 |
| Unit bank, activation on `[384, 4096]` (backward with the bias sums) | 19.3 | 35.7 |
| Unit bank, second layer, all 8 units: `[8, 384, 512] × [8, 512, 64]` | 18.0 | 20.9 + 19.8 |
| **Unit bank** | **78.8** | **173.9** |
| 10 small products forward, 17 backward (2 to 8 µs each) | about 40 | about 70 |
| The other kernels (selection, gathers, norms, update, statistics) | about 95 | about 115 |

- The unit bank is 253 µs of the 572 µs: 16.2 ms per step. The static pass runs every unit on
  every selected pair and gathers one output in eight.
- Its three first-layer products (139 µs, 8.9 ms per step) are single tensor-core products at
  7 to 11 thousand million multiply-accumulates per second: the fastest form found for them
  ("What did not pay").

## The changes

### 1. Dense event-block inputs (no key)

`hsn.train.compile._block_columns` replaces `_full_width`.

- **Before.** The event slots `[B, E, …]` were padded with zeros to `[B, T, …]`, so that a
  block's slice had strides without `E` (one graph for every number of slots). A CUDA graph
  copies a strided input by the byte extent it spans. For 16 columns of the padded scorer
  features that is 22.4 MB per block (four times a window), and 7.1 MB for the context.
- **Now.** Each block gets a dense copy of its 16 slots. Its strides are those of `[B, 16, …]`
  whatever `E` is, so it is still one graph, and the copies are a sixteenth.
- **Values.** The same: a copy. The first step's loss is bit-equal between main and the
  branch on the card (5.539490699768066). Later steps cannot be compared bit for bit: two
  runs of the same code already differ from step 2 on (the training is not run in
  deterministic mode).
- **Measured.** Block inputs 1.22 → 0.54 ms of device time; step 69.4 → 68.3 ms.

### 2. The trainer kernels: build them in the run checkout (no code change)

P2.10b's trainer kernels (`hsn.kernels.trainer`: `LazyAdamW`'s steps, the float64 clip's
scaling, the update caps, the weight average; bit-exact by `tests/test_p2_10b_trainer_kernels.py`)
are used when `build/hsn_window.fatbin` and its extension exist **in the checkout that runs**.
`build/` is not committed, and the run checkouts of 14 to 17 Nov (`~/hsn-*-run`) have none, so
every HSN-T run on the card has taken the PyTorch reference passes.

    python scripts/build_window_kernels.py --arch 86      # CPU only, about a minute

- **Measured.** Optimiser part 3.24 → 1.90 ms of device time and 558 → 278 kernels; step
  68.3 → 65.0 ms (with the merged round 64.7 → 61.1). Under a loaded host (load average 17
  to 22, one run each) 91.7 → 82.3 ms.
- It applies to every HSN and HSN-T run on the trainer, not to word events alone.
  `HSN_TRAINER_KERNELS=0` switches them off.

### 3. `hardware.round_products=merged` (opt-in)

Default `separate`: the round as it was. The key is left out of the configuration hash and of
the experiment database's memory key at its default; it needs a compiled window
(`hardware.compile_mode`) and `model.routing=topk` and is refused elsewhere. It applies to the
compiled autograd window's sparse round, in the loop over bytes and in the loop over events,
in training and in evaluation. No equation changes: the same dot products in other matmul
shapes.

- **The unit pass** (`UnitBank.forward_static_merged`, `PDGCell.step_sparse`).
  - The first layer still runs for every unit (one product; see "What did not pay").
  - The chosen unit's hidden row is taken before the activation (`pick_hidden`), so the
    activation runs on `[384, 512]` instead of `[384, 4096]`, and the second layer is one
    product of `[384, 512]` against all units' second layers laid flat (`pick_unit` takes
    the chosen unit's outputs).
  - The gate's linear reads the same input: it rides in the first product as 64 more columns.
  - `W_out` is folded into the second layers (`W2[j] · W_outᵀ`, once per window, in fp32).
  - The backward of the two picks writes each gradient once (a scatter into zeros and a
    masked copy).
- **The front** (`HSN.front_layout`, `HSN._front`). `P` and `S` are one two-layer map of
  `[x_S ; q]` with block-diagonal weights. `S`'s own read of `q` stays detached where the
  summary cut detaches it; `P` reads the carried `q`.
- **Layouts.** Built once per window with autograd, outside autocast, in the masters' dtype
  (`UnitBank.merged_layout`): a block casts them at each use, so the gradients of a window
  are summed in fp32 as before.

Per event slot, by the profile:

| | `separate` | `merged` |
| --- | ---: | ---: |
| Matrix products, forward + backward | 12 + 21 | 8 + 13 |
| Kernels per replay, forward / backward | 915 / 1,097 | 828 / 961 |
| Event loop device time per step, forward / backward, ms | 13.75 / 22.95 | 11.61 / 20.84 |
| Unit bank forward: first layer / activation and pick / second layer, µs | 41.6 / 19.2 / 18.0 | 42.0 / 9.8 / 12.5 |

- The gain is 4.3 ms of device time and 3.9 ms of step.
- The unit bank's backward did not get cheaper: its two first-layer products (60 + 38 µs)
  are untouched and are now the largest item of the loop.

## Timings

400 training steps per run through the real `TBPTTTrainer` (evaluation and checkpoints
stubbed), one process per run, alone on the card. The number of a run is the median step time
of steps 21 to 400 without the compile and statistics steps (374 steps). Runs of the different
configurations were alternated over three card windows between 16:09 and 17:00.

| Configuration | Runs, ms | Median | Spread within a run (p10 to p90) |
| --- | --- | ---: | --- |
| main `b32f663`, no kernel build | 70.05, 69.29, 69.32, 69.75, 69.38 | 69.38 | about 1.4 |
| branch, `separate`, trainer kernels off | 68.43, 68.13 | 68.28 | about 1.2 |
| branch, `separate`, trainer kernels on | 64.99, 64.95, 65.02, 64.99 | 64.99 | about 1.1 |
| branch, `merged`, trainer kernels off | 64.73 | 64.73 | 1.2 |
| branch, `merged`, trainer kernels on | 61.12, 60.99, 61.31 | 61.12 | about 0.9 |

- Every run: 1 dynamo graph, 0 recompiles, 64 event slots in all 408 windows.
- The 75.6 ms on record for this arm is the experiment harness's number of 16 Nov on
  `5c54832`. Main measures 69.4 ms with this tool today; the harness number of the merged
  round is under "Quality".
- The parts of `merged`, one run each on an earlier form of the picks: the unit pass alone
  63.5 ms, the front alone 65.3 ms (64.99 without either), both 62.2 to 62.9 ms.

## Equality

- **Unit pass** (`tests/test_round_products.py`, float64): against the grouped pass
  (`forward_selected`) followed by the gate and `W_out`, for six splits of 1 to 384 pairs
  over the units. Values to 1e-12, every gradient to 1e-11, exact zeros for a unit no pair
  chose. The picks against `gather` and its autograd gradient: bit for bit.
- **Front**: against `P(q)` and `S(x)`, values 1e-12 and gradients 1e-11, with and without
  the summary cut.
- **The round**: the capturable round with the merged products against the same round with
  the separate ones, both run eagerly in float64 over four blocks, loop over bytes and loop
  over events, training and evaluation, with and without the cut and the candidate norm.
  Values 1e-10, parameter gradients 1e-9, selections exact.
- **Compiled against eager** (CPU inductor, fp32, the tolerances of the compile tests):
  values 8e-6, gradients 1.2e-4, selections exact; the recipe's window in small over
  windows of 2 to `T` event slots on one graph.
- **On the card**: see "Tests".

## Tests

- CPU suite: see the commit message.
- Card: see the commit message.

## Quality

See the commit message for the 9k TinyStories run of the target arm with the key.

## What did not pay

- **The unit bank by segments.** Each pair multiplied with its own unit's weights only:
  pairs laid into segments of 24 to 96 rows per unit, one batched product per layer against
  weights gathered per segment, 3 to 5.6 times fewer multiply-accumulates. Built, equal in
  float64, measured as products on the card and removed:

  | Product | All units (one product) | By segments of 24 / 48 / 96 rows |
  | --- | ---: | --- |
  | First layer, forward | 43 µs | 31 / 42 / 35 µs |
  | First layer, input gradient | 38 µs | 32 / 25 / 39 µs |
  | First layer, weight gradient | 60 µs | 53 / 33 / 35 µs |
  | Gathering the weights per segment | | 25 / 31 / 22 µs |
  | Adding the weight gradients back per unit | | 120 / 74 / 56 µs |

  Batched products of small matrices run at 2 to 4 thousand million multiply-accumulates per
  second on this card against 7 to 10 for one large product, and the weights have to be
  copied. This is the lesson of P2.10's fused cell round again: keep the products large and
  on tensor cores. (Component measurements; no end-to-end run.)
- **Other forms of the picks.** The chosen unit's hidden row as a masked sum over the units
  reads all eight rows: 25 µs per slot forward, step 62.5 ms. A gather with a masked copy as
  its backward: the compiler fused the copy with the activation's backward and evaluated it
  for all eight units, 43 µs per slot, step 61.8 ms. The form kept (a gather; a scatter into
  zeros that already hold the rider's columns): 10 µs forward, 26 µs backward, 61.1 ms.
- **Replacing the stable sort of the selection** by comparisons and sums: 20 to 36 µs against
  8.8 µs for the sort.

## What is left

In device time per step, with the three changes: event loop 32.5 ms (forward 11.6, backward
20.8), reader 13.6 ms, the rest 8.2 ms; the card is idle for about 7 ms.

- **The reader, 13.6 ms.** Its attention kernels are 2.4 ms of that (job F3's causal kernel
  addresses them; not measured here). Compiling the prologue would be a second graph, which
  this job was told not to add; estimated 3 to 5 ms (not measured).
- **The unit bank's first-layer backward, 6.3 ms.** The weight gradient (38 µs per slot)
  does not feed the recurrence. It could be taken once per window, grouped by unit, outside
  the graphs: about 2 ms (estimate). So could the other 9 weight-gradient products of a slot.
  That is the split window's replay, for the weights only; it needs a custom backward.
- **The scorer features for all 256 positions** are built in the prologue and 64 slots of
  them are read: about 0.6 ms of copies.
- **The host**: the float64 gradient norm (84 tensors, two kernels each), the per-step
  snapshot of the divergence dump, the word address (about 150 small kernels).
- A step under 50 ms needs the loop's backward restructured (the split window's design for
  HSN-T and events) or the reader's cost out of the step. Neither is a result-preserving
  change of the kind made here.

## Where it is in the code

- `python/hsn/config.py`: `hardware.round_products`, `round_products_scope_error`,
  `HASH_OMIT_AT_DEFAULT`. `python/hsn/experiments/spec.py`: the memory key leaves the key
  out at `separate`.
- `python/hsn/model/units.py`: `merged_layout`, `forward_static_merged`, `pick_hidden`,
  `pick_unit`.
- `python/hsn/model/cell.py`: `WindowConsts.fold_w2` / `fold_b2` / `front_*`,
  `window_consts(merged=…)`, the merged branch of `step_sparse`.
- `python/hsn/model/hsn.py`: `front_layout`, `merged_consts`, `_front`, `_position`.
- `python/hsn/train/compile.py`: `_block_columns`, `CompiledWindow(merged=…)`,
  `CompiledWindow._consts`, `MERGE_UNITS` / `MERGE_FRONT` (measurement switches).
- `tests/test_round_products.py`.
- Measurement tools, traces and results (not in the repository):
  `~/hsn-experiments/wev-speed/` (`tools/prof_wev.py`, `tools/analyze_wev.py`, `prof/`,
  `timing/`, `logs/`).
