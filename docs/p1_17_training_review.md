# P1.17 training review and performance fixes

The review starts from main at `0a69f31` (P1.16). A5's mechanism, compatibility boundaries and
experiment commands are in [a5_implementation.md](a5_implementation.md). This change implements
the mechanism and fixes the actionable training issues below. It does not establish superiority
over gru_2m; that requires the unchanged evaluation gates and matched GPU experiments.

## Findings addressed

| Finding | Resulting behavior |
| --- | --- |
| Experiment caps took a prefix of a task-ordered manifest, silently excluding later tasks. | New capped runs deterministically sample every task/variant, reject a cap smaller than the number of groups, and record full/effective document, token and answer counts. Explicit first-N overfit remains first-N. Historical results retain their recorded or legacy prefix policy, and new capped arms have distinct identities. |
| Loader resume checked the document count but accepted different or reordered contents. | Checkpoints include a cached fingerprint of ordered inputs, targets, masks, context and document/dependency identity. Resume rejects mismatches. Old checkpoints without the field remain readable. |
| Calling `fit` again restarted the loader; the baseline trainer also restarted its carried state. | HSN, GRU and Transformer retain both stream position and state between calls, matching uninterrupted training and disk resume. A5 retains the previous byte too. |
| Gradient accumulation weighted each window's mean task loss equally, even with very different answer counts. | Each supervised position in the accumulated optimizer step has equal task-loss weight. Auxiliary losses retain their window averaging. A single-window step keeps its previous arithmetic. |
| The EMA prediction target moved after skipped or rolled-back steps and GradScaler overflow. | The target advances only when an optimizer update was applied. A restored target remains restored. |
| Short utility probes could not credit delayed writes; using the full fixed window length restricted the start to position zero. | Optional `train.probe_horizon=null` chooses a valid position and probes to the current window's end. Eligibility counts and a warning expose unreachable dependencies. The old default remains 16 for experiment compatibility; the new A5 sweep explicitly selects null. |
| One fallback row triggered a dense cell pass over the entire batch, in addition to the sparse pass. | Eager execution compacts the fallback rows once per window. Compiled execution uses bounded, power-of-two row buckets with distinct padding rows; only actual fallback rows replace sparse results. |
| Normal execution stacked every cell state solely to calculate a maximum statistic. | Normal eager and compiled windows keep one detached scalar maximum per position. Full cell history is allocated only with `capture=True`. |
| Routed training reported every trainable parameter as active. | Reports count distinct logically selected units per non-padding row/position plus shared modules, handle soft/fallback execution, and exclude the unused single-round halting head. Physical kernel work remains a separate measurement. |
| Cost profiles could be reused across incompatible shapes or new A5 features. | Profiles record and check dimensions, relevant layer settings and A5 flags. Legacy omissions are accepted only at compatible reference defaults; ambiguous configurations require reprofiling. |

The delayed-data inspection found that the default 16-position horizon made only 215 of 41,906
answer dependencies eligible to probe their own source in the reviewed training documents. A null
horizon removes the short-horizon exclusion within the window; it does not cross a TBPTT boundary,
increase the probe sampling budget, or guarantee that a useful source will actually be sampled.
Dependency metadata is diagnostic only and never becomes a router input or an oracle training loss.

For `B=32`, `T=256`, `N=32`, `d=64`, the removed float32 diagnostic state stack is 64 MiB; the
replacement vector of maxima is 1 KiB. This is an allocation calculation, not a measured total
training-memory reduction: autograd still retains the tensors required for backpropagation.
Fallback compaction reduces the dense pass to the selected rows (or their compiled bucket); the
sparse pass still executes for the batch. Neither change by itself establishes a GPU speedup.

A local CPU comparison against `0a69f31` measured the eager forward and backward cell workload
with A5 disabled: arm64, PyTorch 2.15.0, one CPU thread, float32, `B=32`, `T=8`, `N=32`, `d=64`,
`H=512`, `M=8`, `K=4`, top-K STE. Each row below is the median of seven calls after two warmups,
with the stated fallback rows held fixed. Calls zeroed gradients, initialized state, ran the
window and backpropagated the mean squared logits plus mean squared final state. This isolates
the model workload; it includes no data loading, optimizer update, auxiliary probe or compilation.

| Fallback rows / 32 | Main (ms) | Fixed (ms) |
| --- | ---: | ---: |
| 0 | 19.9 | 18.4 |
| 1 | 190.5 | 30.0 |
| 3 | 174.1 | 40.0 |
| 8 | 172.0 | 62.3 |

These CPU measurements support the benefit when a small fallback subset used to trigger a
whole-batch dense pass. They do not predict CUDA throughput, end-to-end training speed or model
quality. Full-batch buckets keep the original dense/where path to avoid unnecessary gathers.

## Architectural boundaries

The dense reference can retain identical cells when initialized and updated symmetrically. Its
graph edges then have no distinguishing state to mix. A5 addresses routed cell selection and
readout; this change does not alter dense gates, introduce a new dense symmetry-breaking mechanism,
or claim that the dense reference now binds keys.

The compiled static unit bank still evaluates all units before gathering selected outputs. This
is the existing launch-overhead tradeoff. Logical active-parameter counts must not be presented
as physical FLOPs or measured energy savings.

The frozen data mixture, document boundaries, task thresholds and GRU baselines are preserved.
No curriculum, default rollback policy, new auxiliary loss, recurrent attention, matrix state,
or state-schema migration is introduced. A5 is independently opt-in for addresses and readout.

## Validation and remaining evidence

Regression tests cover task coverage and historical replay, dataset identity, uneven-answer
accumulation against independently assembled gradients, repeated-fit/disk-resume equivalence,
EMA skip/rollback/overflow behavior, fixed and variable probe horizons, fallback values and
gradients against the former full-batch computation, and diagnostic allocation shapes.

A5 tests cover isolated seed initialization, default hashes/weights, fixed-code persistence,
address-feature ablations, reset and split-window equivalence, direct STE readout gradients,
eager/compiled agreement, evaluation/serving carry, checkpoints and divergence replay.

CPU checks can verify these invariants. CUDA graph behavior, the proposed scorer-overhead bound,
9,000-step two-seed trained-key binding, unseen-key generalization, and the matched gru_2m quality,
latency, memory and energy comparisons still require the reference GPU runs described in the
implementation document. CUDA-marked tests must run there before interpreting performance results.
