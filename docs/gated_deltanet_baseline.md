# The Gated DeltaNet baseline: `model_name: gated_deltanet`

17 November 2026. Branch `claude/gated-deltanet-baseline`, from main `5941d13`. Written on the
CPU (torch 2.15.0+cu132, the card hidden); the Triton kernels were tested and timed on the local
RTX 3060 in one window of 14 minutes on the same day (see [Tests](#tests) and
[Step time](#step-time-rtx-3060-17-november-2026)). No training run has been made beyond the
120 steps of that timing: nothing here says how well the model scores.

## Why

HSN-T with its memory is far ahead of the three baselines of this repository on keyed recall
(GRU, Transformer, the Mamba-style selective state-space model). None of the three has an
associative memory, so that lead does not show whether the HSN's memory is special or whether
any explicit key-value memory would do. Gated DeltaNet (Yang, Kautz and Hatamizadeh, "Gated
Delta Networks: Improving Mamba2 with Delta Rule", 2024, arXiv:2412.06464) is the closest
published efficient model: a fixed-size fast-weight memory per head, written with a gated
delta rule, and among the strongest efficient models on associative recall. It is the
control for that question (user, 17 November: more appropriate comparisons).

## What it is

`hsn.baselines.gated_deltanet.GatedDeltaNetBaseline`: byte embedding plus context-symbol
embedding, `n_layers` blocks, RMS norm, linear head (untied, with bias, as the other
baselines). One block is the reference block of flash-linear-attention
(`fla.models.gated_deltanet` around `fla.layers.GatedDeltaNet`), pre-norm and residual:

```
n       = RMSNorm(x)
q, k, v = SiLU(causal depthwise conv over 4 positions of q_proj(n), k_proj(n), v_proj(n))
q, k    = q / |q|, k / |k|                                  per head
α_t     = exp(-exp(A_log) · softplus(a_proj(n_t) + dt_bias))    decay in (0, 1), per head
β_t     = sigmoid(b_proj(n_t))                              write strength in (0, 1), per head
S_t     = α_t S_{t-1} + β_t (v_t - α_t S_{t-1} k_t) k_tᵀ    the gated delta rule
o_t     = S_t q_t / sqrt(K)
x       = x + o_proj(RMSNorm(o) ⊙ SiLU(g_proj(n)))          the output gate
m       = RMSNorm(x)
x       = x + down_proj(SiLU(gate_proj(m)) ⊙ up_proj(m))    gated MLP
```

`S` is a matrix per head, `[V, K]` with key width `K = head_dim` and value width
`V = expand_v × K`: the "fast weights". Reading it with a key returns the value last written
under that key; a write first decays the whole matrix by `α`, then moves what the matrix
returns for the current key towards the current value by the fraction `β`.

### How it maps to the paper

| Paper | Here |
| --- | --- |
| `S_t = S_{t-1} (α_t (I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ` (the gated delta rule) | the `S_t` line above, the same expression multiplied out (`S` stored value-first, `[V, K]`) |
| `α_t`, the data-dependent decay ("gate") | `exp(-exp(A_log) · softplus(a_proj(n_t) + dt_bias))`: the reference implementation's form, a rate and a step per head as in Mamba2 |
| `β_t`, the data-dependent write strength | `sigmoid(b_proj(n_t))`, per head |
| keys and queries at unit length | `q`, `k` divided by their length (the kernels' `1e-6` under the root) |
| short convolution and SiLU on `q`, `k`, `v` | depthwise, causal, 4 taps, no bias |
| output: `o_t = S_t q_t`, a gated norm, a projection | RMS norm over a head's values times `SiLU(g_proj(n))`, then `o_proj` |
| the chunkwise parallel form for training | fla's Triton kernels (`chunk_gated_delta_rule`, chunks of 64) |
| block: mixer and MLP with pre-norms | as above; SwiGLU MLP |
| the hybrid models (with sliding-window attention and Mamba layers) | not built: this is the pure Gated DeltaNet |

Left at the reference's defaults and not exposed: no negative eigenvalues (`β` in (0, 1), not
(0, 2)), as many value heads as key heads, no bias in the convolution.

Initialisation is the reference model's: every projection, the convolution and the
embeddings normal with standard deviation 0.02, `exp(A_log)` uniform in (0, 16),
`softplus(dt_bias)` log-uniform in [0.001, 0.1], norm weights 1, head bias 0. `A_log`,
`dt_bias` and the norm weights are one-dimensional, so the trainer gives them no weight
decay (the reference marks `A_log` and `dt_bias` as free of weight decay too).

### What comes from flash-linear-attention and what is written here

- **From fla** (`fla.ops.gated_delta_rule`, pip `fla-core` 0.6.0): the delta rule of a window
  on CUDA, forward and backward (`chunk_gated_delta_rule`), called with the options
  `fla.layers.GatedDeltaNet` passes (the kernels normalise `q` and `k`, apply the sigmoid of
  `β` and the decay's activation, and keep `S` value-first). And `naive_chunk_gated_delta_rule`,
  the parallel form in PyTorch, for a test.
- **Written here**, in plain PyTorch on every device: the projections, the convolution (with
  the inputs carried from the window before and the document starts), the gated norm, the MLP,
  and the recurrence above one position at a time (`kernel = "recurrent"`): what runs without
  a CUDA device and what the kernels are tested against. The reference layer could not be
  used as it is: its forward calls the Triton kernels on every device, and it has no notion of
  a row that starts a new document inside a window.
- **Checked against the reference layer** on the card: with this module's weights copied into
  `fla.layers.GatedDeltaNet` (0.6.0, from the `flash-linear-attention` package), the mixer's
  output equals the layer's (`test_cuda_mixer_equals_flas_layer`; it runs where `fla.layers`
  is installed, which the baseline's environment does not need).

## Size

`configs/baselines/gated_deltanet_2m.yaml`, `baseline.gated_deltanet.*`:

| Key | Value | |
| --- | --- | --- |
| `d_model` | 176 | residual width |
| `n_layers` | 4 | blocks |
| `num_heads` | 3 | heads, each with a memory of its own |
| `head_dim` | 44 | key and query width of a head; 3 × 44 = 132, three quarters of `d_model` as the reference keeps it |
| `expand_v` | 2 | value width of a head: 88 |
| `d_conv` | 4 | taps of the convolution |
| `mlp_hidden` | null | the reference's rule (two thirds of 4 × `d_model`, rounded up to a multiple of 256): 512 |
| `norm_eps` | 1e-6 | every RMS norm |

Parameters, measured by `param_report()` for 256 bytes and 16 context symbols: **1,932,584**
(byte embedding 45,056; context embedding 2,816; four blocks 1,839,224, each 459,806: mixer
189,118, MLP 270,336, two norms 352; final norm 176; head 45,312). The others: GRU 1,977,856,
Transformer 1,930,624, Mamba-style 1,887,744.

### State per row

What a model carries from one window to the next, per row of the batch, in numbers:

| Model | State per row | What it is |
| --- | --- | --- |
| GRU 2M | 768 | 2 layers × 384 |
| HSN-T with memory (the recipe) | 4,096 | 64 cells × 64 |
| Mamba-style 2M | 38,912 | scan state 4 × 512 × 16 = 32,768, convolution inputs 4 × 3 × 512 = 6,144 |
| **Gated DeltaNet 2M** | **52,800** | fast weights 4 × 3 × 88 × 44 = 46,464, convolution inputs 4 × 3 × 528 = 6,336 |

So this control has 13 times the HSN cells' state and a third more than the Mamba-style
model's. The state of a head is `K × V = 2 K²` numbers, so at a fixed `num_heads × head_dim`
(and with it a fixed parameter count to within 0.3 %) the head count sets the state size:

| Heads × key width | Parameters | Fast weights per row |
| --- | --- | --- |
| 2 × 66 | 1,931,344 | 69,696 |
| **3 × 44** (the default) | 1,932,584 | 46,464 |
| 4 × 33 | 1,933,912 | 34,848 |
| 6 × 22 | 1,936,656 | 23,232 |

The reference's large models use a key width of an eighth of the width (6 heads); at 176 that
is 22 per head, which bounds how many keys a head can hold apart. 3 × 44 was chosen as the
stronger memory at this size; the other rows are one override each
(`baseline.gated_deltanet.num_heads`, `head_dim`) and have not been trained either.

## State, windows and document starts

The model is recurrent under `hsn.train.loop.SequenceModel`, like the GRU and the Mamba-style
model: `init_state(B, device)` gives zeros `(S, conv)` with `S [n_layers, B, 3, 88, 44]` and
`conv [n_layers, B, 3, 528]` (the last three inputs of every block's convolution), both
float32; `forward(tokens, context, state, reset)` returns the logits, the new state and
`{"state_abs_max": ...}`. The trainer carries the state between windows and detaches it at
every window start (truncated BPTT), and the evaluation carries it the same way.

`reset[b, t]` clears row `b` before position `t`: the fast weights are zero there and the
convolution reads no input from before the reset. Within a window the delta rule runs in its
parallel form with the incoming `S` as the initial state and returns the final `S`:

- No row starts a document after the window's first position (aligned documents): one call of
  the kernels in their batch form per block, the rows that start at position 0 zeroed first.
- Some row does (packed documents, TinyStories): one call per block in the kernels'
  variable-length form. The batch is passed as one stream of `B × T` positions cut at every row
  start and every reset; the segments that start a row take the row's carried `S`, the
  segments that start a document take zeros, and the final `S` of each row's last segment is
  the row's new state. This costs one host read per window (where the starts are), as the GRU
  and the Mamba-style model have.

## Kernels and precision

`GatedDeltaNetBaseline.kernel` selects the form of the delta rule. It is an attribute, not a
configuration key: the forms compute the same function, and a run takes `auto`.

| `kernel` | What runs | Where |
| --- | --- | --- |
| `auto` (default) | `chunk` on CUDA, `recurrent` elsewhere | |
| `chunk` | fla's Triton kernels, the parallel form, forward and backward | CUDA only |
| `recurrent` | the recurrence one position at a time, PyTorch, float32 | any device |
| `reference_chunk` | fla's parallel form in PyTorch, float32; no document start inside a window | tests |

Under the trainer's autocast (bf16 on the RTX 3060) the projections run in bf16, the kernels
read `q`, `k`, `v` in bf16 and compute in float32, and `S` is kept and carried in float32, as
in the reference. The decay is computed in float32 inside the kernels from the raw `a_proj`
output, so a decay close to 1 is not rounded to 1. The convolution's sum and the gated norm
are computed in float32 here (the reference's kernels do the same and store bf16). The
recurrent form switches autocast off and runs in float32.

**The kernels are not float32-exact.** On this card fla's kernels multiply matrices on the
tensor cores in TF32, a 10-bit mantissa (`fla.utils.hardware`: Triton's `dot` default on
Ampere and newer; the triangular solve of the parallel form is set to `tf32` as well). In
float32, on the RTX 3060, the whole model through the kernels differs from the recurrence by:

| | Small test model | Default size |
| --- | --- | --- |
| Logits, largest difference over the largest logit | 2e-4 to 1e-3 | 9e-4 to 1.8e-3 |
| Fast weights after three windows, likewise | 0.6e-3 to 1.4e-3 | 0.8e-3 to 0.9e-3 |
| Parameter gradients, norm of the difference over the norm, worst parameter | 0.35 % to 2.0 % | 1.9 % |

Six cases were measured (two seeds of the small model and the default size, each in the
kernels' batch form and in their variable-length form); the table gives their range. The
worst gradient is a decay parameter (`dt_bias` or `A_log`) in five of the six; at the kernel
alone, on random inputs, the three decay gradients (`A_log`, `dt_bias`, the input of the
decay) were off by 1.6 to 1.8 % of the largest gradient and the others (`q`, `k`, `v`, `β`) by
0.15 to 0.2 %. TF32 is the likely cause and was not separated out (fla has no switch that
turns it off in all of its products). This is the reference library as it is used for
training, in bf16, where the inputs carry less precision than that; it is why the card test
of the kernels against the recurrence has bounds of 5e-3 and 5 %, not of rounding.

On the CPU the model is the recurrent form: about 1.0 s per training window of 256 × 4 at
the default size (forward and backward, two threads) and 2.2 s per evaluation window of
256 × 64, so the CPU is for tests and for scoring, not for training.

## Compiled or eager

**Eager, with fla's own kernels.** `hardware.baseline_compile` is refused for this family when
the configuration is loaded. The delta rule is the model's recurrence and it already runs as
vendor kernels (Triton, compiled for the card on first use), in the position the GRU's cuDNN
recurrence has; fla marks its kernel entry points `torch.compiler.disable`, so `torch.compile`
cannot take them into a graph and a whole-window graph (what the Transformer and the
Mamba-style model get) does not exist for this model. What could be compiled is the PyTorch
between the kernel calls (the projections, the convolution, the gated norm, the MLP), as five
graphs around four kernel calls, the way the GRU's embedding and read-out are compiled around
cuDNN. That was not built.

It matters more than it does for the GRU. Measured on the card (below): of a 50.6 ms step,
fla's delta-rule kernels take 7.4 ms (3.0 forward, 4.4 backward; eight calls per step, timed
in a separate pass of 30 steps with a device sync around each call, in which a step took
56.8 ms). The other 43 ms are the trainer's own work (the loss, the clipping, the optimiser)
and this module's PyTorch between the kernel calls: element-wise operations on window-sized
tensors, one launch and one pass over memory each (the convolution's four taps with their
document masks on `[32, 256, 528]`, the gated norm, the MLP's gate, the RMS norms). How the
43 ms divide between the two was not measured. The reference does that work in fused kernels
of its own (one for the three convolutions and their SiLU, one for the gated norm, one for
the MLP's gate); this module keeps one PyTorch form of it for every device, the form the CPU
tests exercise, and does not call them.

So the step time below is an eager step time of this implementation: beside the *eager* step
times of the other baselines it is like for like, and for Gated DeltaNet itself it is an
upper bound, not its speed. The two ways to bring it down (compile the stages between the
kernel calls; or call fla's fused convolution and norm kernels on CUDA and keep the PyTorch
forms for the CPU) both need card time to test and are not in this change.

## Environment

The kernels are not a dependency of `hsn`: `uv.lock` and the dependencies of
`pyproject.toml` are unchanged (its one edit tells mypy that `fla` is untyped and may be
absent), and the project environment (on this machine the shared `/home/joe/hsn-p2.10/.venv`,
which every running experiment uses) was not touched.

| | |
| --- | --- |
| Package | `fla-core` 0.6.0 (module `fla`), the core operations of flash-linear-attention, <https://github.com/fla-org/flash-linear-attention>; PyPI release of 23 October 2026, wheel sha256 `b0708b70…3ebd` |
| Licence | MIT (Songlin Yang, Yu Zhang, Zhiyuan Li). Nothing of it is copied into this repository. |
| Its dependency | `einops` 0.8.2 (MIT) |
| Pin | `scripts/gated_deltanet_requirements.txt` (both wheels with their hashes); `FLA_VERSION` in the module, which refuses any other release |
| Environment | `scripts/gated_deltanet_env.sh <new directory>`: `uv sync --locked --extra dev --extra cuda` into that directory (the project's locked packages: the same torch 2.15.0, triton 3.9.0 and everything else), then the two wheels with `--no-deps --require-hashes`. It refuses a directory that exists. |
| On this machine | `/home/joe/hsn-gdn-venv` (built by that script on 17 November: the shared environment's package versions plus `fla-core` and `einops`; its `hsn` points at the checkout `/home/joe/hsn-gdn`, and `hsn_rs` was copied over from the shared environment) |

Without the kernels, importing `hsn.baselines.gated_deltanet` raises `ImportError` and says
what is missing, which release, and where the environment comes from. `hsn.runner` imports the
module only when a `gated_deltanet` model is built, so every other model, the sweep tools
(`expand`, `enqueue`, `report`) and the test suite run in the project environment as before;
there the model's tests are skipped with that reason.

A run needs the baseline's interpreter:

```
cd <checkout>
PYTHONPATH=<checkout>/python /home/joe/hsn-gdn-venv/bin/python -m hsn.runner run \
    --config configs/baselines/gated_deltanet_2m.yaml --profile rtx3060 --out <runs>
```

In a sweep the family is `gated_deltanet`. The worker starts each arm with its own
interpreter, so the **worker** is started with the baseline's interpreter (and that
environment's `bin` first on `PATH`), on a queue that holds these arms; an arm a
project-environment worker starts fails at step 0 with the `ImportError` above. The spec for
the first runs is `~/hsn-experiments/remote/specs/gdn_baseline.yaml` (TinyStories, 9k and 27k
steps at 1e-3 and 2e-3; not queued), with the launch lines in its header.

The first training window compiles the Triton kernels and times their variants (fla's
autotuning): with an empty cache the first kernel call took 45 s and the first 30 training
steps 71 s on the RTX 3060. Both are cached under `~/.triton` (or `TRITON_CACHE_DIR`); with
the cache warm the four card tests take 10 s.
The CPU scorers (`~/hsn-handoff/word_keys/wk_score.py`, `wh_score.py`) call
`init_state(B, device)` and the forward as for the other baselines; run with the baseline's
interpreter they take the recurrent form.

## Tests

`tests/test_gated_deltanet_baseline.py`, 17 tests.

In every environment (4):

- Nothing else changed: the configuration hashes of nine shipped configurations (the three
  other baselines, the reader, the HSN recipe, HSN-T with and without memory, `ref_a1`, the
  example) are those of main `5941d13`; the family is known to the sweep spec and the
  registry; a configuration that sets a key of the section hashes differently.
- `hsn.runner` and `hsn.train.baseline_compile` import without `fla` and without the
  baseline's module.
- Without `fla`, building the model raises the `ImportError` with the release, the script and
  this document in it.
- `hardware.baseline_compile` is refused at load.

With the kernels, on the CPU (9):

- Default size 1,932,584 and the state sizes above.
- The delta rule: this module's recurrence against fla's recurrent reference and against
  fla's parallel reference, on the same inputs with a non-zero incoming state over 150
  positions (more than two chunks, not a multiple of 64).
- One block against the layer written out directly (`conv1d`, fla's recurrent reference, the
  gated norm and the MLP by hand).
- **Chunked against recurrent on a small model**: two windows of 80 positions from a non-zero
  carried state through the parallel form and through the recurrence, logits and state.
- **Two windows equal one long window**: logits and final state, with document starts within
  the convolution's reach of the boundary.
- **Resets clear the state**: three windows with carried state and document starts at a
  window's first, middle and last position give every document the logits it gets alone from
  a fresh state, whatever state came in; `reset_state`.
- Causality.
- **Training and resume**: 60 steps through the generic trainer on packed documents, the loss
  falls, and a run resumed from its step-30 checkpoint ends on the bits of the uninterrupted
  one.
- `baseline.address_input`.

On the card (4, `@pytest.mark.cuda`). Run there on 17 November in the baseline's environment:
the 13 tests above and three of these four passed in the first pass; the first of the four
failed its first bounds, which had assumed float32 rounding (see
[Kernels and precision](#kernels-and-precision)), and the four passed with the bounds below
in a second pass.

- **The Triton kernels against the recurrence**, float32, the whole model over three windows
  of 96 positions from a non-zero carried state, once with document starts only at window
  starts (the kernels' batch form) and once with starts anywhere, a one-position document and
  a start at a window's last position among them (the variable-length form): logits, state
  and every parameter's gradient, the gradient through the carried state included, to the
  kernels' precision (5e-3 of the largest logit and state value, 5 % of a gradient's norm;
  measured: the table above).
- Two windows of 128 equal one window of 256 through the kernels (2e-4), and a row that
  starts a document gets the same logits from any incoming state as from zeros, bit for bit.
- Training under the profile's autocast through the kernels: 60 steps and two evaluations on
  packed documents, finite and falling loss.
- **The mixer against `fla.layers.GatedDeltaNet`** with the weights copied over (200
  positions, float32): equal to 1e-5 absolute and 1e-4 relative. It is skipped where
  `fla.layers` is not installed; it ran with `flash-linear-attention` 0.6.0 and
  `transformers` 5.23.0 unpacked beside the environment, not installed into it.

Checked by hand during the build and not kept as tests: five wrong versions of the model (the
convolution ignoring document starts, the carried convolution inputs not cleared, the
recurrence not clearing `S`, the decay applied after the write, the taps reversed) each fail
two or three of the CPU tests; a tiny arm through the sweep worker started with the baseline's
interpreter trains, evaluates and writes its checkpoint, and the same arm through a
project-environment worker fails at step 0 with the `ImportError`; `hsn run` on the CPU; the
arm ids, group ids and memory keys of the 122 specs under `~/hsn-experiments/remote/specs` and
`configs/sweeps` that expand on main are identical on this branch (3,428 arms).

## Step time (RTX 3060, 17 November 2026)

One window on the card, alone on it, plain CUDA context: `profile=rtx3060`, TinyStories
(`tinystories_v1`, `out_of_range`), windows of 256 bytes, batch 32, eager
(`hardware.compile_mode=none`), the trainer built as a run builds it. 30 steps to warm up,
then the median of the trainer's own step times over steps 31 to 90 (lowest to highest in
brackets); memory is PyTorch's peak allocation over those steps.

| Model, about 2M parameters | ms per step | Peak memory |
| --- | --- | --- |
| **Gated DeltaNet**, bf16 autocast (the profile's; what a run uses) | **50.6** (50.3 to 51.7) | 841 MiB |
| Gated DeltaNet, float32 (`hardware.autocast_dtype=fp32`) | 65.5 (64.7 to 67.2) | 1,118 MiB |
| Mamba-style (`mamba_2m.yaml`), eager, float32 by its construction | 140.7 (139.8 to 345.4, mean 144.1) | 5,189 MiB |

So eager against eager this model steps 2.8 times as fast as the Mamba-style baseline and
holds a sixth of its memory. Of the 50.6 ms, 7.4 ms are fla's delta-rule kernels (see
[Compiled or eager](#compiled-or-eager)): the number is an upper bound for Gated DeltaNet.
For scale, `docs/baseline_compile.md` gives the eager GRU 16.7 ms and the eager Transformer
16.2 ms at the same batch and window (not measured again here). A dev pass of TinyStories
(2,000 documents, 62 windows of 128 × 256) took 10.9 s; reserved memory with it 1.5 GiB.

At that rate, alone on the card, 9,000 steps are about 8 minutes of training plus the
evaluations, and 27,000 steps about 23.

Not measured: a compiled Mamba-style step beside it (its graph takes minutes to compile and
the window was short), the step time on shared cards, any batch size but 32.

## Where it is in the code

| File | What |
| --- | --- |
| `python/hsn/baselines/gated_deltanet.py` | the model |
| `python/hsn/config.py` | `GatedDeltaNetBaselineConfig`, the model name, the refusal of `hardware.baseline_compile`, `baseline.gated_deltanet` left out of the hash at its default |
| `python/hsn/runner.py` | the registry entry (the module is imported when a model is built) |
| `python/hsn/experiments/spec.py` | the family `gated_deltanet`; the section at its defaults stays out of the memory key |
| `configs/baselines/gated_deltanet_2m.yaml`, `configs/_example.yaml` | the configuration, the section's defaults |
| `scripts/gated_deltanet_env.sh`, `scripts/gated_deltanet_requirements.txt` | the environment and the pin |
| `pyproject.toml` | mypy: `fla` is untyped and may be absent |

## Limits

- **Nothing has been trained.** The learning rate, the head shape and whether 2M parameters
  of this model learn keyed recall at all on these tasks are open; the spec above is the
  first measurement.
- The step time is eager and most of it is not the kernels (see
  [Compiled or eager](#compiled-or-eager)).
- The kernels compute in TF32 on this card: about three digits against the recurrence, 2 % on
  the decay parameters' gradients (see [Kernels and precision](#kernels-and-precision)).
- On CUDA the kernels' autotuning picks a variant by timing, so two processes may not run the
  same variant and bit-exact repeats across processes are not claimed; the resume test is
  exact on the CPU.
- The kernels need a CUDA device with Triton support; on Apple MPS the model falls to the
  recurrent form, which is far too slow to train with.
- One release of fla is supported (0.6.0); the kernel arguments used here do not exist in
  the releases before it.
