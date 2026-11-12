# A5: optional wide gradient norms and clipping

Status: experimental numerical training control. No model equation, architecture identity,
data selection, frozen metric, or acceptance threshold changes.

## Problem and scope

A float32 L2 reduction can overflow even when every gradient entry is finite. Native global
clipping then reports an infinite norm and can zero the gradients. The existing nonfinite guard
cannot distinguish that scalar overflow from genuinely nonfinite entries. This is separate from
the earlier loss of accuracy and routing observed in A5 runs: correct clipping alone is not
evidence of an accuracy fix or learned key binding.

`train.grad_norm_precision: float64` computes the pre-clip global L2 norm in float64. It applies
the existing clipping coefficient `min(1, grad_clip / (norm + 1e-6))` in float64 and only then
copies the products into the existing gradient storage. Computing a wide norm but narrowing
the coefficient before multiplication would still erase direction when that coefficient falls
below the gradient dtype's representable range. The epsilon is PyTorch's existing clipping
definition, not a new empirical threshold. `grad_clip=0` keeps clipping disabled as before.

This option is read by both trainers, group and objective-term reports, gradient-growth inputs,
and captured-window replay/shadow clipping. Gradient-growth groups and responses still come
from their existing configuration. Actual NaN/Inf gradients remain visible to the existing
guard or fp16 GradScaler; the wide helper leaves those entries intact for diagnosis. No finite
document is skipped, replayed, reweighted, or removed by this option. Adam moments, the scheduler,
recurrent state, loader position, and EMA update rules are unchanged.

The default `native` delegates to the original PyTorch operation. It is omitted from configuration
hashes at its default; opting into `float64` changes the hash. Precision has no mutable controller
state and is restored through the existing strict checkpoint configuration check. Norms and
scaling run outside the compiled backward and retain parameter and gradient storage addresses.
CPU and CUDA use float64 locally; MPS uses CPU float64 because that device lacks float64 support.
This mode adds reductions, conversions, and a synchronization. Production cost and learned
performance require new registered comparisons; it is not enabled for any earlier trial.

## Validation requirements

`tests/test_grad_norm_precision.py` covers finite reduction overflow, mixed dtypes, tiny clipping
coefficients, gradient storage and direction, disabled clipping, actual NaN/Inf, the unchanged
default path and hash, consistent controller/term reports, both trainer families, checkpoint
continuation with AdamW and lazy AdamW, and compiled CUDA execution. MPS coverage is conditional
on an available MPS device. The existing default golden suite remains the compatibility gate.

An exact captured-window replay is reported separately from synthetic numerical regression
tests. Neither replaces controlled full training or binding validation. Do not infer that a
lower raw gradient, fewer numerical skips, or a successful replay establishes stable learning.
