# A5 gradient-growth response

Status: experimental, opt-in; implements the user's follow-up request relayed in bridge #45
after experiment 270. The prior update-size projection does not measure increasing sensitivity:
that run's raw gradient medians rose while its relative Adam updates stayed similar.
This training-only update rule changes no model equation, data, evaluation gate or default.

`train.gradient_growth` is disabled by default and omitted from legacy hashes at its default.
Configured groups select disjoint trainable parameters by name globs; the implementation has no
module-name assumptions. It observes each group's raw norm after AMP unscale and before global
clipping, plus the existing global raw norm. Only finite, accepted samples enter its history.

Every configured `interval` samples, compare medians of the recent `window` samples with a
reference window ending `reference_lag` samples earlier. A group is hot when global and group
growth exceed their configured thresholds and its median norm share exceeds `min_share`.
Its absolute target scale is

```
effective_growth = group_growth * max(1, share_growth) ** share_power
target = max(min_scale, min(1, (growth_threshold / effective_growth) ** response_power))
```

Scale reductions take effect immediately; increases toward that target are limited by
`recovery_rate` per observation. When a group's growth falls to `recovery_threshold`, its target
returns to one with the same gradual recovery. Between these conditions it holds its scale.
Repeated identical warnings set the same target, rather than multiplying repeated cuts.
Ratios use configured `norm_floor` for near-zero references. The observation horizon is samples,
not an empirical training step number. All knobs and run values live in configuration.

The scale multiplies the optimizer's proposed displacement for that group, including decay.
Other groups retain their exact proposal. Adam moments advance once; any EMA target is updated
once from the final weights. The operation is outside compiled forward/backward and preserves
parameter storage. This can coexist with the independent update-size bound, which applies first.
Adaptive LR, rollback and finite-gradient spike skipping must be disabled to avoid competing
responses. Existing nonfinite protection remains. No finite document or batch is discarded.

History, sample counts, scales, decisions and damped-step counts survive checkpoint/resume.
Observations include medians, norm shares, growth, target/applied scales and measured proposed
and applied displacement ratios. Live SQLite progress labels the age of the last dev evaluation;
final/partial records retain all observations for assessing whether the signals were useful.

This is a test of a warning signal, not proof that gradient growth causes failure or that damping
prevents it. Legitimate learning can change gradient norms, and a small norm after collapse is not
evidence of recovery. Judge trials by sustained learning, gradient trajectories and frozen binding
tests, including both controlled seeds. No quality threshold, data weight, or architecture identity
changes; the decoder and memory remain HSN's.
