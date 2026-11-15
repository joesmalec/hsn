# Gate M memory interventions

This bundle records a development evaluation of thirteen frozen `synthetic_hold`
checkpoints: short-span and whole-window HSN-T readers, the memory core alone,
and the candidate with eight fast cells. It separates memory necessity from
key/value binding. It is not the locked final-test campaign or a utility verdict
against independently trained matched controls.

[RESULTS.md](RESULTS.md) contains every task/key/distance table, document counts,
95% document-bootstrap intervals, and the far-strict gate-feasibility analysis.
[manifest.json](manifest.json) identifies the checkpoint, source, document and
raw-result hashes. Each file in `raw/` is lossless gzip-compressed JSON containing
the full per-document counts, predictions and counterfactual query records.
Decompress with `gzip -dc raw/reader32-s1_600.json.gz`.

The files in `sources/` preserve the evaluated Python programs byte-for-byte.
Their `.py.txt` suffix marks historical evidence rather than modules belonging to
the current model implementation. They can be read directly or copied to `.py`
files for execution. Their SHA-256 hashes must match the manifest and every run.
No checkpoint weights or original/final-test dataset files are included.

## Reproduce the recorded campaign

These checkpoints use the historical `model_name: hsn, model.reader: transformer`
configuration. Current main intentionally rejects that spelling. Use the pinned
reader checkout instead of converting the checkpoints or changing main's models:

```sh
git worktree add --detach /home/joe/hsn-gate-m 76c2aec878af68f945040e909ffdfc03b71b4bd1
mkdir -p /home/joe/hsn-handoff/gate_m
cp docs/gate_m/sources/gate_m_interventions.py.txt /home/joe/hsn-handoff/gate_m/gate_m_interventions.py
cp docs/gate_m/sources/hold_matrix.py.txt /home/joe/hsn-handoff/hold_matrix.py
```

The exact archived scripts retain the original `/home/joe` paths. The checkpoint
folders declared in `RUNS` must be available under `/home/joe/hsn-experiments`,
including their resolved configurations and completion records. The reference
archive is `/home/joe/hsn-experiments/a14-stage1-eval/matrix.json`. Use an environment
with the pinned checkout's dependencies; the recorded interpreter was
`/home/joe/hsn-p2.10/.venv/bin/python`.

Run from `/home/joe/hsn-gate-m` with `CUDA_VISIBLE_DEVICES=''`,
`OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=1`,
`PYTHONDONTWRITEBYTECODE=1`, and `PYTHONPATH=/home/joe/hsn-gate-m/python`.
Launch one process with `nice -n 10`:

1. Run `hold_matrix.py` with output `gate_m/smoke_reference.json`, the
   `reader32-s1` experiment directory, `--docs 50 --steps 18000`.
2. Run `gate_m_interventions.py --smoke`.
3. Run `gate_m_interventions.py --models all` for 600 fresh documents per group.

Every document starts from cleared state and fits one window. The six diagnostic
groups reuse the reference generator's seed offset `104729`; the `test keys`
label denotes its fresh test-alphabet diagnostic, not access to the locked final
dataset. The process stops on a baseline or hook invariant failure. It resumes
completed conditions and binding groups from its JSON outputs, so preserve the
script and protocol when resuming. Use a fresh output directory for a changed
experiment rather than mixing results.

## Validate and package

After all thirteen checkpoints finish, run:

```sh
python scripts/gate_m_bundle.py \
  --source /home/joe/hsn-handoff/gate_m \
  --reference /home/joe/hsn-handoff/hold_matrix.py \
  --out docs/gate_m
```

This standard-library-only validator refuses incomplete campaigns, missing
conditions, failed invariants, mismatched source/document hashes, inconsistent
answer counts, and binding outcomes that disagree with the saved query records.
It does not execute the archived evaluator, load model weights, or access a GPU.
The evaluation itself checks exact unpatched/patched baseline logits, the
unchanged reference evaluator and available archived matrices, N1/N3 equality
without fast cells, N2b answer/all equality, identity state swaps, and the
counterfactual byte counts and reference solver answers.
