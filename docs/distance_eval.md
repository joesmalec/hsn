# Score by distance: `eval.by_distance` and `python -m hsn.eval.distance`

18 November 2026. Branch `claude/distance-eval`, from main `f7e39be`. Written, tested and
cross-checked on the CPU (torch 2.15.0+cu132, the card hidden); nothing has run on the card.
**Evaluation only**: training, the models and every existing metric are unchanged. The `eval`
section is therefore outside the configuration hash and the arm id, on or off (the user's
decision of 18 Nov 2026): a run with the report is the same run as without it.

## Why

A run's record has one dev number per task. What separates the models of this program is where
an answer comes from: a fact 40 bytes back in the same window, 300 bytes back in an earlier
window, 3,000 bytes back; a name the story used a paragraph ago. Those tables were made after
the fact by scripts outside the repository (`~/hsn-experiments/oor-recall/eval_dist.py`,
`~/hsn-handoff/grammar_eval/`, `~/hsn-handoff/names_beyond_256/`), each with its own model
loader, at hours of CPU per round. This puts the same tables, definition for definition, into
the run itself and into one CLI.

## The key

| Key | Default | Meaning |
| --- | --- | --- |
| `eval.by_distance.enabled` | `false` | when training ends, score every set by distance and put the block `by_distance` into the run record |
| `eval.by_distance.every_eval` | `false` | also at every evaluation of the run (dev set only): each evaluation entry gains `by_distance` |
| `eval.by_distance.max_docs` | `null` | the first `max_docs` documents of every task/variant group of a set; `null` = all |
| `eval.by_distance.sets` | `[]` | more dev sets scored at the end: `{name, manifest, tasks: [], variants: []}` |

- `eval` is a new top-level section for evaluation-only reports. It never enters the
  configuration hash or an arm id, whatever it holds (`hsn.config.HASH_EXCLUDED`, applied in
  `config_hash` and in `hsn.experiments.spec.config_view`): the same arm with the report on and
  off has one hash, one arm id and checkpoints that resume either way. See "Runs that exist"
  below for what that means in the queue. A key that could change a trained weight does not
  belong in this section.
- The other three keys are refused while `enabled` is false. A set's `name` is its key in the
  report (letters, digits, `_`; `dev` is the run's own dev set). A set is always its manifest's
  `dev` split, and its `unroll` must be the run's.
- Example, a five-task run that also reports the far set
  (`configs/data/synthetic_far.yaml`, branch `claude/far-recall-data`):

      eval.by_distance.enabled=true
      "eval.by_distance.sets=[{name: far, manifest: data/synthetic_far/manifest.json}]"

### What it costs

- **Off:** nothing.
- **On:** one more evaluation pass per set after training (`Trainer.evaluate`, the compiled
  evaluation window when the run has one, the run's evaluation batch): the cost of one of the
  run's own evaluations for the dev set, and the same per document for every other set. On the
  runs scored below one evaluation of the 2,000 TinyStories dev stories took 3.7 s (word events)
  to 8.5 s (per byte) on the card, against 2,300 to 4,700 s of training. The pass is bounded by
  the size of the sets; `max_docs` bounds it further.
- **Host work per pass** (measured on one CPU thread): the capture is about the cost of the
  run's two metrics again (TinyStories dev, 62 windows of 128 x 256: 2.5 s against their 1.8 s;
  the far set and the five-task dev set: under 0.2 s) and holds 10 bytes per byte of the set
  plus about 1 KiB per `delayed` answer (its log-probability row): 15 MiB for TinyStories dev,
  74 MiB for the far set. The tables are array sums: TinyStories dev 0.7 s a pass, 2.3 s the
  first time (the static index of names and words is built once per dataset); the synthetic
  sets under 0.2 s.
- **`every_eval`:** no further pass; each of the run's evaluations pays the host work above.
- The pass and the tables are timed in the record (`walls.by_distance_s`, each set's `seconds`,
  stage `by_distance` in `phases.stages`). The pass's evaluations also appear in
  `phases.evaluate`; `walls.eval_s` and `eval_count` are the run's own evaluations as before.

A set that fails to score keeps its `source` and an `error` in the block and the run's record
stands; with `every_eval` an evaluation whose tables fail carries `by_distance: {error: …}` and
the run goes on. A manifest that does not load fails the run during setup, before training.

## Runs that exist

Because the report is not part of an arm's identity, switching it on for an arm that has a
run does not make a new arm:

- **`enqueue` skips it, as it skips any arm that is queued, running, done or stopped early,**
  and says why the report will not appear:
  `note: <label>: exp N (done) was run with other eval.* settings, which are not part of an arm
  id, so nothing was queued. Score the run that exists: python -m hsn.eval.distance --run-dir
  <run dir> (--force trains the arm again)`. For an attempt that is still queued or running
  the note says that it keeps the settings it was queued with (cancel it and enqueue again to
  change them).
- **The CLI is the way to score a run that exists** ("The CLI" below): it reads the
  checkpoint, costs one evaluation pass per set and trains nothing. Its report is a file of
  its own (`--out`); the run's `result.json` and the database row are not rewritten.
- **`enqueue --force` trains the arm again** as a new attempt of the same arm; that attempt
  holds the `eval` settings it was queued with (every attempt stores its own configuration)
  and its record carries the block.
- A checkpoint written with the report on resumes with it off and the other way round (one
  hash).

## Where the block is

- Sweep runs (`hsn.experiments`): `result.json` → `by_distance`; with `every_eval` also
  `evals[i].by_distance` (the dev set's `groups`).
- `hsn run`: `metrics.json` → `by_distance`.
- The CLI: its report's `by_distance`.

## The block (schema 1)

    by_distance:
      schema: 1
      step: 27000                 # the step of the weights scored
      weights: averaged           # averaged (train.weight_ema, as the run's evaluations) | trained
      window: 256                 # train.unroll: what "window" means below
      align_docs: true
      seconds: 9.1
      sets:
        dev:                      # the run's dev set; then one entry per eval.by_distance.sets
          source: {manifest, tasks, variants, dataset, split: dev, n_docs, max_docs}
          accuracy, loss, n_positions, n_windows, seconds   # the pass's own overall numbers
          compile_seconds         # of the pass, compiling or recording a window: 0 when the
                                  # run's evaluation variant served it; null for an eager run
          groups:
            "<task>/<variant>": {...}                       # one table per group, below

`schema` rises when a key changes meaning or goes away, not when one is added. A count is `n`;
an accuracy is a fraction in `[0, 1]`; an empty cell has `n: 0` and `null` values; buckets
without an answer are left out of the recall tables.

### A synthetic group (`kind: recall` or `answers`)

Every loss position of a synthetic document is an answer with a dependency distance
(`meta["deps"]`: positions back to the earliest input symbol the answer depends on).

| Key | Families | Content |
| --- | --- | --- |
| `family`, `n_docs`, `n`, `accuracy`, `loss` | all | the group (`loss` in nats) |
| `by_distance` | all | `{bucket: {n, accuracy}}` by distance: `lt64`, `64_255`, `256_511`, `512_768`, `769_1023`, `1024_1535`, `1536_2047`, `2048_3072`, `gt3072` |
| `by_windows_back` | all | `{"0": …, "1": …}`: windows between the fact's window and the answer's |
| `by_query` | kv, delayed, kv_words, delayed_words | `first` (the document's first answer, its anchor) and `later` |
| `by_fact_position` | the same four | `{fact position: {distance: {n, accuracy}}}`, fact position `0_255`, `256_767`, `768_1535`, `ge1536` against distance `lt256`, `256_767`, `768_1535`, `ge1536` |
| `by_target`, `predicts_none`, `by_distance_value_targets` | kv, kv_words | answers that are a `value` against answers that are `none`; the share of predictions that are `NONE`; the distance table of the value answers |
| `own_values` | delayed, delayed_words | `accuracy` when the prediction is restricted to the values the document states, its `chance`, `predicted_share` (the unrestricted prediction is one of them), `mass` (probability on them), `by_distance`, `alphabet` |

- The distance edges are those of the out-of-range analysis of 18 Nov 2026
  (`~/hsn-handoff/OOR-RECALL-2026-11-18.md`): synthetic_v1's ranges end at 255 and 768, the far
  set's at 3,072.
- **Fact position** is the answer's position minus its distance: the value byte of a `delayed`
  fact, the last symbol of the last operation on a `kv` key. It is the definition of
  `hsn.data.synthetic.fact_positions` on `claude/far-recall-data` (no shard stores it; every
  shard has it); a document whose metadata carries `fact_pos` is read from there instead. The
  grid separates "far back" from "late in a long document", which the far set mixes.
- **Own values** needs the value alphabet. It is the byte range the group's answers span (the
  generators draw values from one range that keys and filler stay out of), reported as
  `alphabet`; exact once the smallest and the largest value were each asked somewhere in the
  group (always, at the sizes of the committed sets), read from the whole group also when
  `max_docs` scores fewer documents.
- The other families (brackets, arithmetic, algorithmic, recall_sum) get `by_distance` and
  `by_windows_back` only (`kind: answers`).

### A text group (`kind: text`)

`n_docs`, `n` (scored bytes), `bits_per_byte`, `accuracy`, and two tables.

**`names`**: `n_names`, `n_mentions`, `dropped_window_straddle`, `window`, `rows`. A name is a
capitalised word of three or more letters that the group's documents use five or more times
inside a sentence and whose lower-case form is at most a tenth as frequent (99 names on
TinyStories dev; `God`, `Granny`, `Mama`, `Mrs`, `Tuesday`, `Wednesday` pass the rule and are
left out). Every mention inside a sentence is one case, by where the same name was last written
in the story (start to start):

| Row | Previous mention |
| --- | --- |
| `same_window_lt64`, `same_window_64_255` | in the window that predicts this mention |
| `earlier_window_lt64`, `earlier_window_64_255` | in an earlier window, under 256 bytes back |
| `ge256` | 256 or more bytes back; its parts `ge256_256_383`, `ge256_384_511`, `ge256_ge512` |
| `first_occurrence` | none |

A row has `n` (mentions), `letters` (their letters after the capital), `bits_rest` (bits per
letter after the capital: which name), `bits_first` (bits of the capital) and `ci95_rest`,
`ci95_first` (95 % intervals over stories: 2,000 resamples of the documents, seed 20261115). A
mention whose letters are predicted in two windows is left out.

**`words`**: `frequent_words`, `threshold`, `rows`. Letters of content words by the distance
back to the story's last use of the word (case-folded, start to start): `new`, `lt64`, `64_255`,
`256_511`, `ge512`, `ge256_first_letter`, `ge256_later_letters`; a row has `n` (letters),
`share` (of the scored bytes) and `bits` (per letter). A content word is one the training
shard uses under 100,000 times. The words at or above that are listed once per dataset in
`configs/data/<name>.frequent_words.json` (68 words for TinyStories, from the 39,856,638 words
of the training shard; the file names the shard's hash), written by

    python -m hsn.eval.distance --frequent-words configs/data/<name>.frequent_words.json \
        --manifest data/<name>/manifest.json [--also OTHER_DATASET ...]

(CPU, 23 s for the 200 MB shard). A text dataset without a list gets `words: {skipped: …}`.

These are the definitions of the grammar evaluation's "which name" table
(`grammar_eval/analyze.py`) and of `names_beyond_256/names256.py` and `words256.py`.

## The CLI

    python -m hsn.eval.distance --run-dir RUN [--checkpoint step_00027000.pt]
        [--override model.reader_carry=256 ...] [--set far=data/synthetic_far/manifest.json ...]
        [--no-dev] [--max-docs N] [--device cpu|cuda] [--eval-batch 128] [--threads N]
        [--out FILE.json] [--dump DIR]
    python -m hsn.eval.distance --checkpoint RUN/step_00018000.pt ...   # the same, one checkpoint

- Builds the model from `RUN/config.resolved.yaml` plus the overrides, loads the checkpoint's
  weights for scoring (the weight average when the checkpoint holds one;
  `Trainer.load_for_evaluation`, which does not compare configuration hashes, so a read-time
  override or a run of an older tree is accepted) and scores each set with one
  `Trainer.evaluate` pass.
- Sets: the run's dev set (unless `--no-dev`), then `--set NAME=MANIFEST[:VARIANTS[:TASKS]]`
  (comma lists), or the run's own `eval.by_distance.sets` when no `--set` is given.
- `--device cpu` (default) reads with the CPU profile in fp32; `cuda` keeps the run's hardware
  and autocast with every compile mode off. Hide the card for a CPU run while it is booked:
  `CUDA_VISIBLE_DEVICES=""`.
- Paths in the run's configuration are relative to the working directory: start it in a tree
  whose `data/` has the run's dataset.
- Prints the tables; `--out` writes the report (`by_distance` plus the run, checkpoint,
  overrides and device); `--dump DIR` writes every set's per-byte dump (`nll` float32 nats,
  `hit`, `offsets`: the layout of the grammar evaluation's dumps).
- `--from-dump DUMP.npz --manifest data/<name>/manifest.json [--window 256]` tables a text
  set from a dump that exists, without a model (about 1 s for TinyStories dev).

Example (the far set, read with the reader carry):

    cd <tree with data/>; CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 \
      python -m hsn.eval.distance --threads 4 --no-dev \
        --run-dir ~/hsn-experiments/vast3/exp/gpu0g/runs/five_18k/five18_memT_top0-s0-29b9e8f56ab8/exp1 \
        --override model.reader_carry=256 --set far=data/synthetic_far/manifest.json

## How it works

- `hsn.eval.evaluate.evaluate_windows` takes an optional `observer`, called per window with the
  loader's window and the logits after the metrics; `Trainer.eval_observer` (a harness setting,
  `None` by default, like `eval_position_groups`) hands one to `Trainer.evaluate` and stores what
  it returns under its key. Nothing an observer does enters a metric.
- `hsn.eval.distance.capture.PositionCapture` keeps every loss position's loss and arg-max by
  document and position. The position is the count since the row's last `reset`, carried
  across windows; as a guard every kept position's token must equal the document's, and the
  pass must have visited every loss position exactly once.
- `recall.py` and `text.py` build the tables from that pass; `score.py` has the sets of a run
  (`RunScorer`) and the observer (`ByDistance`).

## Cross-check against the prototypes

CPU, the card hidden, four threads; `python -m hsn.eval.distance` from this branch on
checkpoints of earlier trees (their configurations and weights load on main). The reports,
logs, per-byte dumps and comparison scripts are in `evidence/distance_eval/` of the worktree
(`/home/joe/hsn-distance-eval`; git-ignored, 51 MB).

**The far set** (`~/hsn-experiments/oor-recall`, `tables_evalcarry.md`): the in-range five-task
HSN-T, `vast3/exp/gpu0g/runs/five_18k/five18_memT_top0-s0-29b9e8f56ab8/exp1`, step 18,000,
`--override model.reader_carry=256 --no-dev --set far=data/synthetic_far/manifest.json`
(1,436 s).

| | Here | Prototype |
| --- | --- | --- |
| delayed, 2,048-3,072 bytes back (n 1,617) | 98.89 % | 98.9 % |
| delayed / kv / all | 99.12 / 58.72 / 73.02 % | 99.12 / 58.72 / 73.02 % |
| every cell of the prototype's table (distance, windows back, first and later query, value and NONE answers, predicts NONE, restricted to the document's values and its chance: 66 cells) | | largest difference 0.000 points, every `n` equal |
| fact position x distance (`pos_analysis.py` on the prototype's rows) | equal in every cell | |
| answer by answer (12,211 answers) | the same 8,917 correct | log-probabilities equal to the last digit |

**Names and words** (`~/hsn-handoff/NAMES-BEYOND-256-2026-11-18.md`, `names256.py`,
`words256.py`), TinyStories dev, 2,000 stories:

| Run (27k) | | Names 256+ (95 %) | its first letter | Names 64-255, earlier window | Words 256-511 | Dev bits |
| --- | --- | --- | --- | --- | --- | --- |
| span 16 everywhere, `vast5/exp/gpu0/runs/names_e4c/e4c_top0_cread_span16_27k-s0-1da01542d423/exp1` | here (423 s) | 0.15 (0.06-0.31) | 2.43 | 0.16 | 1.332 | 0.7827 |
| | memo | 0.15 (0.06-0.31) | 2.43 | 0.16 | 1.332 | 0.7828 (card) |
| word events + carry, `vast5/exp/gpu1/runs/text_carry_e1_27k/ts_memT_top0_wev_c256_27k-s0-ba51973f9d0e/exp1` | here (424 s) | 0.58 (0.45-0.74) | 4.67 | 0.04 | 1.530 | 0.7412 |
| | memo | 0.58 (0.45-0.74) | 4.67 | 0.04 | 1.530 | 0.7413 (card) |

- **The tables alone:** `--from-dump` on the prototypes' own dumps
  (`grammar_eval/dumps/<model>/dev.npz`) gives their tables digit for digit: every row's `n`,
  bits and interval of the names tables and every row of the words tables, for both models.
  The 68 frequent words equal those of the prototype's training counts (39,856,638 words,
  19,700 types).
- **The pass:** the per-byte losses of the two full passes against the prototypes' dumps:
  1,999 (word events) and 1,997 (span 16) of the 2,000 stories agree at every byte within
  0.001 nats (mean difference 1e-7); one and three stories differ from some byte to their end.
  The cause is the row layout, not the code: the prototype scored the stories in chunks of
  250, the CLI in one stream. Scored as the prototype's chunks through `Trainer.evaluate` and
  the capture (stories 1,000 to 1,249 of the word-event run, 750 to 999 of the span-16 run,
  which hold three of the four differing stories), the losses are the prototype's dumps bit for
  bit. So a difference in rounding between the layouts sets those stories on another path
  (not looked into further; a discrete choice of the routed memory is the likely place). The
  tables of the full passes differ from the prototypes' by at most 0.0015 bits per letter in
  any cell.

## Tests

`tests/test_distance_eval.py` (CPU): the key (off by default; the pinned `ref_a1` hash with
every key of the section set; the arm id and group id of an arm equal with the report on, with
sets, and off; the refusals); the queue (an existing arm enqueued with the report on is
skipped with the note, `--force` queues a new attempt that holds it); the capture on aligned and unaligned streams with documents that span windows,
against a per-document reading, with the metrics equal with and without it, and its two
guards; the recall table against a brute-force reading for `kv` and `delayed` at far-set
distances; the fact position; the name rule; the names and words tables against brute-force
readings; the frequent-word count across block boundaries and the committed list; the trainer's
observer; the compiled window (aot_eager on the CPU): the same training step for step, the same
compile steps, and a pass that compiles nothing; a sweep run (HSN and GRU) with the block, the
unchanged training, the extra set and
`every_eval`, and the CLI reproducing the run's tables; the failure paths; a text run, its dump
and the tables from the dump.

## Limits and what is open

- The window cases (`same_window_*`, `by_windows_back`) assume `data.align_docs` (the default):
  a document's windows then start at its first byte. `align_docs` is recorded in the block.
- The name rule and its six exclusions are TinyStories'; another text dataset gets the rule
  without a reviewed exclusion list.
- No verdict rule or report column reads the block yet (`hsn.experiments.rules` /
  `report`): the tables are in the record, not in `final`.
- Paired differences between two models (the memos' "minus memory off" columns) need two
  dumps; `--dump` writes them, a comparison command does not exist.
- Not run on the card: the compiled evaluation window is the same `Trainer.evaluate` path the
  run's evaluations use, but the block has only been produced on the CPU.
