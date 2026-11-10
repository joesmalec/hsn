# A5 update-cap reference experiment

Experimental, training-only, opt-in. No architecture, dataset, quality threshold, or default
changes. The original cap bounds a group's update by `max_ratio * max(current_norm, norm_floor)`.
It does not bound accumulated drift. In the original capped seed trio (272, 273, 294), router
norms grew much faster in the two failing runs, and their current-norm caps became inactive before
or during collapse. This association motivates a controlled test; it does not establish causality.

`train.update_limit.norm_growth_limits` maps configured group names to a finite multiplier at
least one. For those groups only, the radius becomes

```
max_ratio * max(min(current_norm, initial_norm * multiplier), norm_floor)
```

The initial norm is captured when the limiter is constructed and saved/restored with its
checkpoint and rollback state. It is never reset on `fit`, resume, or evaluation. Empty mapping
preserves old projection arithmetic, checkpoint layout, and configuration hashes, including
existing nonempty cap configurations. The default is omitted from the hash. Unknown groups and
invalid multipliers fail validation. All parameter groups still receive their gradient/Adam
update; projection only scales excessive proposed displacement, including weight decay. No
document is removed, replayed, or reweighted. Adam moments still advance once. This ceiling limits
the update budget's growth, not weight norm or total distance from initialization. It cannot
guarantee a stable recurrent Jacobian or learned key binding.

The existing `update_ratio/*` and state diagnostics execute in an optimizer post-hook, before
post-optimizer projections. Their historical meanings and frozen verdict inputs are preserved.
Runs using a limiter or growth controller now label that stage `optimizer_proposal` and also
record final `applied_update_ratio/*`, `applied_update_ratio_before/*`, absolute update/weight norms,
and `applied_descent_cosine/*` after both projections. The two ratios use after/before weight norms
respectively; the before denominator matches the cap definition. A cosine is null for a zero
update or zero gradient and compares displacement to the negative clipped gradient. It is not a
loss-improvement test. `applied_state/*` evaluates final weights on the same fixed diagnostic
window while restoring RNG. These extra diagnostics run only on sampled accepted optimizer
steps. They add no samples for nonfinite/skipped steps. Their time is included in diagnostic
overhead, not steady-state timing. Default training and diagnostic records are unchanged.

The initial follow-up comparison holds seeds, data, 9,000-step schedule, learning rate, memory cap,
and frozen reads-state/binding metrics fixed. It compares the original two caps against anchoring
only the routing cap with multiplier one. Both arms use the separately validated float64 gradient
norm reduction to avoid the confirmed scalar-norm overflow. Consequently old native runs provide
historical context, while same-code float64 controls are the direct comparator. No adoption or
advantage claim is warranted until completed controlled-seed results support it.
