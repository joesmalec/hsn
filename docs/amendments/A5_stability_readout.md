# A5 extension: bounded updates and decoder shortcut dropout

Status: opt-in experiments; neither mechanism is adopted as a default or a demonstrated fix.
The user requested a fix for weight-state sensitivity and failed key binding after the A5 audit.
This amends the training update and A5's training decoder input. State equations, persistent cells,
sparse graph, routing, fixed byte codes, benchmark data and quality gates are unchanged.

## Proposed-update limit

For each explicitly configured, disjoint parameter group, let w be the old parameters and d the
optimizer's proposed displacement, including weight decay. Apply

```
radius = max_ratio * max(norm(w), norm_floor)
scale = min(1, radius / norm(d))       # zero displacement has scale 1
w_new = w + scale * d
```

An unconstrained proposal is kept bitwise, without reconstituting it by subtraction/addition.
Adam's moments advance once; any target-encoder EMA observes the final projected parameters once.
The projection runs eagerly outside compiled forward/backward graphs and preserves tensor storage.
`train.update_limit.groups` maps arbitrary group names to `selectors` (parameter-name globs) and
`max_ratio`. Every selector must match trainable parameters, and groups must not overlap.
Unselected parameters retain the optimizer's proposal. `norm_floor` handles near-zero groups.
An empty mapping disables all work/state and is omitted from historical configuration hashes.

This responds to measured proposed updates, without step-number triggers, document filtering,
or persistent learning-rate cuts. It does not bound cumulative drift, guarantee stable recurrent
Jacobians, or prove that a large raw gradient produces a harmful update. The experiments must
measure actual cap activity, final/peak accuracy, and binding. Finite batches remain in training;
existing nonfinite protection still applies. Rollback and spike skipping are disabled in these trials.

## Decoder shortcut dropout

A5 decodes `[q; u; selected_readout]`. With `model.decoder_shortcut_dropout = p`, training draws
one Bernoulli keep mask per example/position, shared by q and u. Their decoder inputs become
`keep * q / (1-p)` and `keep * u / (1-p)`. The selected-cell readout is unchanged. At p=1 both
shortcuts are zero; at p=0 the original path consumes no random numbers. Evaluation always uses
all inputs unchanged. This requires A5 selected readout and adds no parameters or persistent state.

The hypothesis is that occasional reliance on the selected cells encourages value retrieval.
Key-sensitive routing alone does not demonstrate that values were stored or retained. Dropout
may therefore fail; use independent ablations and retain the frozen key-swap thresholds. Training
RNG is checkpointed as before. Compiled and eager random masks need not be identical.

## Recovery and reporting

`train.rollback_selection=guard_horizon` chooses the newest available snapshot at or before
the trigger's reference horizon: snapshot cadence for sharp/nonfinite events, and
`spike_window + slow_burst_window` for slow bursts. If no snapshot reaches that horizon, it uses
the oldest and reports `horizon_available=false`. Selection by age does not certify safety.
Only the selected anchor survives recovery; future snapshots refresh it normally. The default
`oldest` retains prior behavior. Existing rollback still abandons intervening windows/documents;
it is an optional recovery mechanism, not the document-preserving prevention experiment.

Failed retained-checkpoint runs now report `latest_saved` alongside the conservative `partial`
from before the first guard. Both are diagnostics. Failure status, primary final score and
verdict semantics remain unchanged; a more recent saved state is not labelled healthy.
