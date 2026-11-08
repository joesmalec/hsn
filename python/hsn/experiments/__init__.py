"""P1.11: the experiment sweep harness.

A sweep is one YAML spec (:mod:`hsn.experiments.spec`): a base recipe, a list and/or grid of arms,
a protocol (:mod:`hsn.experiments.protocols`) and rules registered before any run
(:mod:`hsn.experiments.rules`). ``enqueue`` expands it into arms with stable ids and queues them in
one shared SQLite database (:mod:`hsn.experiments.db`), which records every experiment past,
present and future; ``worker`` (:mod:`hsn.experiments.worker`) runs them detached, sharing the card
within a memory budget and running timing arms alone; ``report`` (:mod:`hsn.experiments.report`)
summarises from the database only; ``import`` (:mod:`hsn.experiments.importer`) backfills the P1.10
audit's runs. Usage: ``docs/sweeps.md``; command line: ``python -m hsn.experiments`` or
``hsn sweep``.

The reads-state protocol, its edit probe (:mod:`hsn.experiments.probe`), the training diagnostics
(:mod:`hsn.experiments.diagnostics`) and the model variants (:mod:`hsn.experiments.variants`) are
the P1.10 efficiency audit's (branch ``hsn-efficiency``, ``f8884ec``), moved here as library code.
Variants are harness-level and selected by name; they add no model configuration keys.

The queue side (spec, rules, db, scheduler, machine, worker, report, importer, cli, steptime)
imports without torch; only the protocols that train load it.

P1.14 added:
- failed and cancelled arms keep their partial record (:mod:`hsn.experiments.run_arm`);
- the 0.5 s step-time guard (:mod:`hsn.experiments.steptime`);
- the mixed diagnostic cadence and ``min_samples`` (:mod:`hsn.experiments.rules`);
- probe v2, the vocabulary-matched edit (:mod:`hsn.experiments.probe`);
- the gate write rate (:mod:`hsn.experiments.diagnostics`);
- finishing process groups counted as ours (:mod:`hsn.experiments.worker`).
"""
