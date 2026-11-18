# The reader's causal mask: `model.reader_mask` (opt-in, 17 Nov 2026)

## Why

The HSN-T reader (`hsn.model.reader.Reader`) masks its attention with the block-diagonal causal
mask over the documents of a row (`Reader.attention_mask`), handed to
`scaled_dot_product_attention` as an additive bias. On CUDA a mask rules out the flash kernel, so
the reader ran on the memory-efficient kernel (`fmha_cutlass`). The profile of the memory-off
160 x 3 recipe at 1024 x 8 with CUDA graphs (`~/hsn-handoff/prof_160x3/`, 17 Nov 2026) put the
attention at 6.8 ms of 15.5 ms of card time a step (backward 5.18 ms, forward 1.61 ms).

With `data.align_docs: true` every document starts at a window boundary, so a window's resets are
at column 0 only. In such a window every position of a row belongs to one document (the one that
starts at column 0, or the one going on from the window before), and the document mask *is* the
causal mask. SDPA's `is_causal=True` with no mask then computes the same reader on the flash
kernel.

## The key

`model.reader_mask: document | causal`, default `document`.

* `document`: the reader as it was, bit for bit (the same call with the same bias; nothing of the
  causal path runs). Left out of the configuration hash (`HASH_OMIT_AT_DEFAULT`) and of the
  experiment memory key (`hsn.experiments.spec.memory_key`) at its default, so every earlier
  configuration keeps its hash, checkpoints resume and step-time estimates still apply.
* `causal`: every reader block calls `scaled_dot_product_attention(q, k, v, is_causal=True)`.
  The configuration hash and the memory key change (it is another step time).

### Where it is allowed (refused everywhere else, at configuration load)

| Condition | Why |
|---|---|
| `model.reader: transformer` | the key is the reader's |
| `model.reader_span: null` | a span limits the distances read; the causal mask alone does not |
| `model.reader_carry: 0` (so no `reader_top_words`) | the carried keys are read through `Reader.read`'s own mask |
| `data.align_docs: true` | the loader then starts documents at column 0 only (below) |
| `train.reach_dropout: 0` | its short rows read with a span mask, which the causal mask does not apply (`docs/reach_dropout.md`) |

It is allowed for HSN-T without its memory (`model.memory: false`, the target), for HSN-T with
its memory (the reader runs in the eager prologue of the window, `HSN._window_prologue`, on the
window's `tokens` and `reset`; the trailing-padding cut keeps column 0) and for the reader alone
(`model_name: reader`). The three are tested for equality with the document mask.

`train.reach_dropout` (`docs/reach_dropout.md`) gives some rows of a training window a span mask;
it is refused together with `reader_mask: causal` (`reach_dropout_scope_error`, and
`Reader.forward` refuses short rows on a causal reader).

### The loader's guarantee

`StreamLoader` (`hsn.train.loop`) with `align_docs`: when a row's document ends inside a window the
row pads to the window's end (`if self.align_docs and pos > 0: break`), and the next document
starts at column 0 of the next window. `collate_windows(..., align_docs=True)` pads every document
to a multiple of the window. The rollback's `drop_documents` makes every row start a fresh
document at column 0. Training, evaluation, the adaptive-lr monitor, the divergence replay and
the probes all take their windows from these, or build windows that reset at column 0 only.
`tests/test_reader_causal_flash.py` checks the guarantee on documents shorter than, as long as
and longer than a window, shuffled over epochs, the finite evaluation loader, a rollback and the
collated windows (and that without `align_docs` resets after column 0 do occur).

### The run-time guard: no host sync

`Reader.forward` with the key runs, on every window,

```python
torch._assert_async(reset[:, 1:].any().logical_not(), "...")
```

* It reads nothing on the host: the reduction and the assert are device work queued with the
  forward pass. On CPU the assert raises at once (`RuntimeError`, the message names the key).
  On CUDA a failure is a device-side assert, which stops the process (the CUDA context is lost);
  it fires on the next synchronisation.
* Under `torch.compile` (inductor, `reduce-overhead`) it is fused into one small Triton
  reduction kernel inside the graph (`triton_red_fused__assert_async_any_bitwise_not_slice_0`):
  still one graph, no CUDA-graph skip, and it fires inside a replayed CUDA graph (checked on the
  RTX 3060). Under `aot_eager` it stays an operator of the graph and raises as in eager.
* Its cost is one reduction over the `B x (T - 1)` resets per window, a few microseconds of card
  time.

So the guard is always on with the key, not only in a debug mode, and it cannot fall back
silently: a window with a document starting after column 0 stops the run instead of being read
through the wrong mask. Falling back to the mask for such a batch would need either a host read
of the resets (a sync per window) or both attentions computed every window, which is what the key
removes.

## Kernels on the RTX 3060 (sm_86)

The 160 x 3 reader has 4 heads of width 40. Flash attention on Ampere takes head widths that are
a multiple of 8 up to 256 (40 runs in its 64-wide tile): with
`torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION])` the causal reader runs forward and
backward under bf16 autocast, the profiler shows `pytorch_flash::flash_fwd_kernel` and
`flash_bwd_*`, and the document reader cannot run there (SDPA has no flash kernel with a mask).
Compiled (`hardware.baseline_compile=reduce-overhead`) the memory-off model keeps one training
graph (and one evaluation graph), no CUDA-graph skip, no recompile, flash kernels in the graph.
Without autocast (fp32) flash does not apply and `is_causal` takes the memory-efficient kernel
without a bias.

## Tests (`tests/test_reader_causal_flash.py`)

* the default keeps hashes and memory keys (named or not); `causal` changes both; the pinned
  memory keys of the arms before the key are unchanged;
* the refusals of the table above (with the memory on and off, the reader alone, the GRU) and the
  `Reader`'s own refusal of a span or carry handed past the configuration;
* the loader guarantee (above);
* float64: the causal reader equals the document reader on aligned windows (rows that start a
  document and rows that go on), outputs to 1e-12 and gradients to 1e-11, with the default, the
  per-block rotary pairs, the float32 rotation, another rotary base and learned positions;
  float32 to 1e-5 and bf16 autocast to bf16 rounding;
* the default is the forward of before the key, bit for bit (`_reference_forward`);
* the control: with a reset inside the window the two masks differ, and the causal reader raises;
* compiled (`aot_eager`, `fullgraph`): the guard raises inside the graph, no graph break, no
  recompile; the memory-off model compiled equals eager and the document model (float64);
* HSN-T with its memory: logits and gradients of the causal reader equal the document reader's
  (float64);
* training: twenty steps of the generic trainer (memory-off HSN-T and the reader alone) and twelve
  of the TBPTT trainer (HSN-T with its memory), windows of 16 over documents of 16 to 32 bytes:
  the same losses and evaluation to 1e-4;
* CUDA only (`@pytest.mark.cuda`): the flash kernel is chosen (above); on the card the causal reader
  equals the document reader (fp32 1e-4, bf16 autocast 0.05) and its forward and backward run
  under `torch.cuda.set_sync_debug_mode("error")` (no host sync); the compiled memory-off 160 x 3
  model trains ten steps with one graph, no skip, flash kernels in the graph and the document
  model's losses to bf16 rounding.

## Timings (RTX 3060, 17 Nov 2026)

400 steps alone on the card, through `Trainer.fit` with the evaluation and the checkpoint stubbed
(timing script `time_arm.py` of the session's scratchpad, the profile's `prof_arm.py` loop): a
run's step time is the median of its step seconds over steps 21 to 400 without compile and stats
steps (the step-time guard's rule); "wall" is the synchronised mean over steps 21 to 400. Three
runs of each, `document` and `causal` alternated (A B, then B A, then A B); the table has the
median of the three. Arms: `fast_readers.yaml` `ts27_nomem_w160l3_top0_w1024` (160 x 3, the 27k
schedule) and `fast_timing.yaml` `time_nomem_w160l4_top0_t1024b8_*` (160 x 4), each with
`hardware.baseline_compile=reduce-overhead` (graphs) and `none` (eager).

| Model | Mode | `document` ms | `causal` ms | change | wall `document` / `causal` |
|---|---|---|---|---|---|
| 160 x 3 | CUDA graphs | 24.01 (25.71, 24.01, 23.16) | 19.27 (20.50, 19.27, 18.76) | -4.74 | 24.33 / 19.71 |
| 160 x 3 | eager | 26.34 (26.86, 25.87, 26.34) | 20.85 (20.71, 20.85, 24.19) | -5.49 | 27.24 / 21.65 |
| 160 x 4 | CUDA graphs | 30.76 (28.90, 30.76, 31.02) | 22.70 (22.70, 22.15, 25.46) | -8.06 | 30.85 / 22.89 |
| 160 x 4 | eager | 32.54 (32.54, 31.37, 34.87) | 25.22 (25.87, 25.22, 24.86) | -7.32 | 33.69 / 26.75 |

Every compiled run: one graph, no CUDA-graph skip, no compile after step 20. The losses of the two
masks agree to 5e-5 over the first 20 steps and to 0.003 in the mean of steps 381 to 400 (bf16).

**The machine was loaded**: other sessions' CPU scoring jobs kept the load average at 16 to 28 on
the 32 threads during these runs, and the absolute step times are about 5 ms above the profile's
of 13:22 (as configured 18.36 ms there, 24.0 ms here; 160 x 4 graphs 22.1 ms on 16 Nov, 30.8 ms
here). The changes are between alternated runs under the same load.
