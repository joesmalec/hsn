# Gate M: frozen-checkpoint memory interventions

Measured CPU diagnostics on fresh synthetic_hold documents, with averaged checkpoint weights. This is a development analysis, not the locked final evaluation campaign or a formal gate decision.

All percentages below include 95% intervals from 1,000 document bootstrap resamples (seed 0). n is the number of answers; d is the number of contributing documents. NE means fewer than 100 answers and is not evaluable. Binding denominators contain only eligible queries answered correctly before the edit. The original query's distance bucket is retained under every edit.

The generator, seed offset 104729, six key groups, length-sorted batches of 64, masks and bootstrap are reused from hold_matrix.py. Each group has 600 documents. Near: d≤31; mid: 32≤d≤62; far: d≥63; far-strict: far with no answer byte in input positions max(0,t−62)…t. The fresh test-key diagnostic uses hold_matrix's test alphabet and offset seed; no saved final-test dataset is opened.

## Status and checks

Worktree: /home/joe/hsn-gate-m at detached commit 76c2aec. Repository source is unmodified. One evaluation process, nice 10, eight intra-op CPU threads, one inter-op thread, CUDA hidden. Raw JSON includes checkpoint SHA-256, document hashes, per-document counts and binding query records.

50-document N0/N1 smoke: complete; baseline matched the unchanged evaluator exactly and disabled-hook logits equalled plain logits bit-for-bit.

- reader32-s1: complete; 6/6 groups started; raw [reader32-s1_600.json](raw/reader32-s1_600.json.gz).
- core-s0: complete; 6/6 groups started; raw [core-s0_600.json](raw/core-s0_600.json.gz).
- reader32-s0-15k: complete; 6/6 groups started; raw [reader32-s0-15k_600.json](raw/reader32-s0-15k_600.json.gz).
- reader32-s0: complete; 6/6 groups started; raw [reader32-s0_600.json](raw/reader32-s0_600.json.gz).
- reader32-s2: complete; 6/6 groups started; raw [reader32-s2_600.json](raw/reader32-s2_600.json.gz).
- reader-full-s0: complete; 6/6 groups started; raw [reader-full-s0_600.json](raw/reader-full-s0_600.json.gz).
- core-s2: complete; 6/6 groups started; raw [core-s2_600.json](raw/core-s2_600.json.gz).
- core-s1: complete; 6/6 groups started; raw [core-s1_600.json](raw/core-s1_600.json.gz).
- candidate-s0: complete; 6/6 groups started; raw [candidate-s0_600.json](raw/candidate-s0_600.json.gz).
- candidate-s1: complete; 6/6 groups started; raw [candidate-s1_600.json](raw/candidate-s1_600.json.gz).
- candidate-s2: complete; 6/6 groups started; raw [candidate-s2_600.json](raw/candidate-s2_600.json.gz).
- candidate-s3: complete; 6/6 groups started; raw [candidate-s3_600.json](raw/candidate-s3_600.json.gz).
- candidate-s4: complete; 6/6 groups started; raw [candidate-s4_600.json](raw/candidate-s4_600.json.gz).

## What the interventions show

- N0: frozen model as trained. The hook wrapper must give exactly the plain logits. Accuracy, counts and bootstrap intervals must equal the unchanged reference evaluator; available archived matrices are checked too.
- N1: suppress routed cell updates throughout the document, keep each routed h unchanged and advance its age. The candidate's fast cells still write; its summary and fast read remain available.
- N2a: zero only the routed selected-cell read r at decode; preserve q, writes, and the candidate's separate fast read f.
- N2b-answer / N2b-all: reset decoder q and routed r at answer positions / all positions. Writes, shared-transform inputs, and explicit fast read f remain as computed. These should give identical answer predictions because decoder outputs do not feed the teacher-forced recurrence.
- N3: suppress routed writes, zero q before the predictor/shared transform/address gate, zero scorer h, age, EMA and position features, and decoder q/r at every position. Exogenous context, address bytes and constants remain. N3-fast-kept retains the candidate's fast-cell recurrence and explicit f; N3-fast-zero suppresses those fast writes and resets f too. The latter is the candidate's whole-memory condition for gate feasibility. For models without fast cells, N1/N3 answers must be identical.
- N4: zero reader output everywhere it is used. This is a dependency probe; it does not establish or refute binding.
- B1: one source value byte changes to an absent legal value chosen with probe v3's seeded hash. KV uses the last INS/OVR binding and skips DEL. Queries and teacher-forced answer tokens stay fixed. All expected answers are re-derived with the dataset reference solver.
- B2: swap both key bytes of the queried delayed2h fact and an unasked partner with a different value. Both bytes must differ, ensuring exactly four changed positions. Skips and eligibility are retained in raw JSON.
- B3: replay A and its B1 twin A′ from zero; immediately before the query, replace A's cells, summary, age and position with A′'s. All A tokens, byte-address features and reader outputs stay A's. Scorer EMA buffers are fixed, equal and checked zero. A′=A gives exactly plain logits in the identity check. This isolates the effect of stored state; it does not independently establish the necessity of the fixed address code.
- Binding outcomes: follows = edited/partner value; unchanged = original value; other = neither. These are conditional response rates, not full recall accuracy. Key swaps need competing facts and therefore have a different eligible population.
- These interventions measure necessity and binding in saved checkpoints. Engineering utility requires independently trained, matched controls and is not decided by zeroing a trained model's inputs.

## Far-strict comparison

No pooling across task or key groups. Each cell is measured percent [95% CI]; n answers, d documents. B3 reports follows among eligible baseline-correct delayed2h queries.

| Checkpoint | Task / keys | N0 | N1 | N3 (routed removed / f kept) | N3 (all memory removed) | B3 follows |
| --- | --- | --- | --- | --- | --- | --- |
| reader32-s1 | kv/training keys | 99.0 [98.3, 99.5]; n=1082, d=475 | 6.6 [5.2, 7.9]; n=1082, d=475 | 6.6 [5.2, 7.9]; n=1082, d=475 | 6.6 [5.2, 7.9]; n=1082, d=475 | not applicable |
| reader32-s1 | kv/dev keys | 98.4 [97.7, 99.2]; n=1095, d=468 | 4.9 [3.7, 6.3]; n=1095, d=468 | 4.9 [3.7, 6.3]; n=1095, d=468 | 4.9 [3.7, 6.3]; n=1095, d=468 | not applicable |
| reader32-s1 | kv/test keys | 99.0 [98.4, 99.6]; n=1129, d=477 | 5.5 [4.2, 6.8]; n=1129, d=477 | 5.5 [4.2, 6.8]; n=1129, d=477 | 5.5 [4.2, 6.8]; n=1129, d=477 | not applicable |
| reader32-s1 | delayed2h/seen pairs | 99.5 [99.0, 99.9]; n=964, d=471 | 1.0 [0.5, 1.8]; n=964, d=471 | 1.0 [0.5, 1.8]; n=964, d=471 | 1.0 [0.5, 1.8]; n=964, d=471 | 99.8 [99.5, 100.0]; n=959, d=471 |
| reader32-s1 | delayed2h/new pairs | 99.3 [98.7, 99.8]; n=964, d=470 | 0.2 [0.0, 0.5]; n=964, d=470 | 0.2 [0.0, 0.5]; n=964, d=470 | 0.2 [0.0, 0.5]; n=964, d=470 | 99.2 [98.5, 99.7]; n=957, d=470 |
| reader32-s1 | delayed2h/never-seen bytes | 99.3 [98.7, 99.7]; n=936, d=463 | 0.7 [0.2, 1.4]; n=936, d=463 | 0.7 [0.2, 1.4]; n=936, d=463 | 0.7 [0.2, 1.4]; n=936, d=463 | 99.6 [99.1, 99.9]; n=929, d=463 |
| core-s0 | kv/training keys | 97.5 [96.6, 98.4]; n=1082, d=475 | 1.8 [1.1, 2.7]; n=1082, d=475 | 1.8 [1.1, 2.7]; n=1082, d=475 | 1.8 [1.1, 2.7]; n=1082, d=475 | not applicable |
| core-s0 | kv/dev keys | 96.9 [95.8, 97.9]; n=1095, d=468 | 1.8 [1.1, 2.6]; n=1095, d=468 | 1.8 [1.1, 2.6]; n=1095, d=468 | 1.8 [1.1, 2.6]; n=1095, d=468 | not applicable |
| core-s0 | kv/test keys | 97.6 [96.6, 98.4]; n=1129, d=477 | 1.6 [0.9, 2.3]; n=1129, d=477 | 1.6 [0.9, 2.3]; n=1129, d=477 | 1.6 [0.9, 2.3]; n=1129, d=477 | not applicable |
| core-s0 | delayed2h/seen pairs | 96.2 [94.9, 97.3]; n=964, d=471 | 1.7 [0.9, 2.5]; n=964, d=471 | 1.7 [0.9, 2.5]; n=964, d=471 | 1.7 [0.9, 2.5]; n=964, d=471 | 98.9 [98.2, 99.6]; n=927, d=471 |
| core-s0 | delayed2h/new pairs | 96.3 [95.0, 97.4]; n=964, d=470 | 1.8 [1.0, 2.6]; n=964, d=470 | 1.8 [1.0, 2.6]; n=964, d=470 | 1.8 [1.0, 2.6]; n=964, d=470 | 98.4 [97.6, 99.2]; n=928, d=470 |
| core-s0 | delayed2h/never-seen bytes | 96.6 [95.3, 97.8]; n=936, d=463 | 1.0 [0.4, 1.6]; n=936, d=463 | 1.0 [0.4, 1.6]; n=936, d=463 | 1.0 [0.4, 1.6]; n=936, d=463 | 97.5 [96.4, 98.4]; n=904, d=463 |
| reader32-s0-15k | kv/training keys | 98.2 [97.4, 99.0]; n=1082, d=475 | 4.1 [3.0, 5.2]; n=1082, d=475 | 4.1 [3.0, 5.2]; n=1082, d=475 | 4.1 [3.0, 5.2]; n=1082, d=475 | not applicable |
| reader32-s0-15k | kv/dev keys | 98.4 [97.7, 99.1]; n=1095, d=468 | 4.4 [3.3, 5.6]; n=1095, d=468 | 4.4 [3.3, 5.6]; n=1095, d=468 | 4.4 [3.3, 5.6]; n=1095, d=468 | not applicable |
| reader32-s0-15k | kv/test keys | 98.8 [98.1, 99.3]; n=1129, d=477 | 4.9 [3.8, 6.1]; n=1129, d=477 | 4.9 [3.8, 6.1]; n=1129, d=477 | 4.9 [3.8, 6.1]; n=1129, d=477 | not applicable |
| reader32-s0-15k | delayed2h/seen pairs | 99.5 [99.0, 99.9]; n=964, d=471 | 1.2 [0.6, 2.0]; n=964, d=471 | 1.2 [0.6, 2.0]; n=964, d=471 | 1.2 [0.6, 2.0]; n=964, d=471 | 99.6 [99.1, 99.9]; n=959, d=471 |
| reader32-s0-15k | delayed2h/new pairs | 99.6 [99.1, 99.9]; n=964, d=470 | 1.8 [1.0, 2.7]; n=964, d=470 | 1.8 [1.0, 2.7]; n=964, d=470 | 1.8 [1.0, 2.7]; n=964, d=470 | 99.8 [99.5, 100.0]; n=960, d=470 |
| reader32-s0-15k | delayed2h/never-seen bytes | 99.4 [98.8, 99.8]; n=936, d=463 | 1.4 [0.7, 2.2]; n=936, d=463 | 1.4 [0.7, 2.2]; n=936, d=463 | 1.4 [0.7, 2.2]; n=936, d=463 | 99.6 [99.1, 99.9]; n=930, d=463 |
| reader32-s0 | kv/training keys | 94.6 [93.2, 96.0]; n=1082, d=475 | 9.2 [7.7, 10.7]; n=1082, d=475 | 9.2 [7.7, 10.7]; n=1082, d=475 | 9.2 [7.7, 10.7]; n=1082, d=475 | not applicable |
| reader32-s0 | kv/dev keys | 91.3 [89.7, 92.9]; n=1095, d=468 | 7.9 [6.4, 9.6]; n=1095, d=468 | 7.9 [6.4, 9.6]; n=1095, d=468 | 7.9 [6.4, 9.6]; n=1095, d=468 | not applicable |
| reader32-s0 | kv/test keys | 93.0 [91.6, 94.5]; n=1129, d=477 | 9.2 [7.7, 10.8]; n=1129, d=477 | 9.2 [7.7, 10.8]; n=1129, d=477 | 9.2 [7.7, 10.8]; n=1129, d=477 | not applicable |
| reader32-s0 | delayed2h/seen pairs | 98.5 [97.8, 99.3]; n=964, d=471 | 0.8 [0.3, 1.5]; n=964, d=471 | 0.8 [0.3, 1.5]; n=964, d=471 | 0.8 [0.3, 1.5]; n=964, d=471 | 99.6 [99.2, 99.9]; n=950, d=471 |
| reader32-s0 | delayed2h/new pairs | 99.0 [98.2, 99.6]; n=964, d=470 | 0.7 [0.3, 1.3]; n=964, d=470 | 0.7 [0.3, 1.3]; n=964, d=470 | 0.7 [0.3, 1.3]; n=964, d=470 | 99.0 [98.3, 99.6]; n=954, d=470 |
| reader32-s0 | delayed2h/never-seen bytes | 98.8 [98.1, 99.5]; n=936, d=463 | 1.2 [0.5, 2.0]; n=936, d=463 | 1.2 [0.5, 2.0]; n=936, d=463 | 1.2 [0.5, 2.0]; n=936, d=463 | 99.6 [99.1, 99.9]; n=925, d=463 |
| reader32-s2 | kv/training keys | 98.6 [97.8, 99.3]; n=1082, d=475 | 15.0 [13.3, 16.7]; n=1082, d=475 | 15.0 [13.3, 16.7]; n=1082, d=475 | 15.0 [13.3, 16.7]; n=1082, d=475 | not applicable |
| reader32-s2 | kv/dev keys | 91.3 [89.7, 93.0]; n=1095, d=468 | 8.7 [7.0, 10.3]; n=1095, d=468 | 8.7 [7.0, 10.3]; n=1095, d=468 | 8.7 [7.0, 10.3]; n=1095, d=468 | not applicable |
| reader32-s2 | kv/test keys | 95.2 [94.0, 96.4]; n=1129, d=477 | 10.4 [8.7, 12.1]; n=1129, d=477 | 10.4 [8.7, 12.1]; n=1129, d=477 | 10.4 [8.7, 12.1]; n=1129, d=477 | not applicable |
| reader32-s2 | delayed2h/seen pairs | 99.5 [99.0, 99.9]; n=964, d=471 | 0.1 [0.0, 0.3]; n=964, d=471 | 0.1 [0.0, 0.3]; n=964, d=471 | 0.1 [0.0, 0.3]; n=964, d=471 | 99.2 [98.5, 99.7]; n=959, d=471 |
| reader32-s2 | delayed2h/new pairs | 99.1 [98.4, 99.6]; n=964, d=470 | 0.0 [0.0, 0.0]; n=964, d=470 | 0.0 [0.0, 0.0]; n=964, d=470 | 0.0 [0.0, 0.0]; n=964, d=470 | 99.3 [98.7, 99.7]; n=955, d=470 |
| reader32-s2 | delayed2h/never-seen bytes | 86.4 [84.5, 88.2]; n=936, d=463 | 0.2 [0.0, 0.5]; n=936, d=463 | 0.2 [0.0, 0.5]; n=936, d=463 | 0.2 [0.0, 0.5]; n=936, d=463 | 97.4 [96.3, 98.5]; n=809, d=434 |
| reader-full-s0 | kv/training keys | 98.3 [97.5, 99.0]; n=1082, d=475 | 23.4 [21.3, 25.6]; n=1082, d=475 | 23.4 [21.3, 25.6]; n=1082, d=475 | 23.4 [21.3, 25.6]; n=1082, d=475 | not applicable |
| reader-full-s0 | kv/dev keys | 98.1 [97.3, 98.8]; n=1095, d=468 | 20.6 [18.4, 22.9]; n=1095, d=468 | 20.6 [18.4, 22.9]; n=1095, d=468 | 20.6 [18.4, 22.9]; n=1095, d=468 | not applicable |
| reader-full-s0 | kv/test keys | 96.4 [95.3, 97.5]; n=1129, d=477 | 21.7 [19.6, 23.8]; n=1129, d=477 | 21.7 [19.6, 23.8]; n=1129, d=477 | 21.7 [19.6, 23.8]; n=1129, d=477 | not applicable |
| reader-full-s0 | delayed2h/seen pairs | 99.7 [99.3, 100.0]; n=964, d=471 | 20.4 [18.0, 22.8]; n=964, d=471 | 20.4 [18.0, 22.8]; n=964, d=471 | 20.4 [18.0, 22.8]; n=964, d=471 | 70.8 [68.2, 73.2]; n=961, d=471 |
| reader-full-s0 | delayed2h/new pairs | 99.5 [99.0, 99.9]; n=964, d=470 | 19.7 [17.5, 22.0]; n=964, d=470 | 19.7 [17.5, 22.0]; n=964, d=470 | 19.7 [17.5, 22.0]; n=964, d=470 | 71.5 [69.1, 73.7]; n=959, d=470 |
| reader-full-s0 | delayed2h/never-seen bytes | 98.0 [97.1, 98.8]; n=936, d=463 | 15.3 [13.0, 17.5]; n=936, d=463 | 15.3 [13.0, 17.5]; n=936, d=463 | 15.3 [13.0, 17.5]; n=936, d=463 | 73.5 [71.0, 76.0]; n=917, d=463 |
| core-s2 | kv/training keys | 98.5 [97.8, 99.2]; n=1082, d=475 | 2.0 [1.3, 2.9]; n=1082, d=475 | 2.0 [1.3, 2.9]; n=1082, d=475 | 2.0 [1.3, 2.9]; n=1082, d=475 | not applicable |
| core-s2 | kv/dev keys | 98.1 [97.3, 98.8]; n=1095, d=468 | 1.4 [0.7, 2.1]; n=1095, d=468 | 1.4 [0.7, 2.1]; n=1095, d=468 | 1.4 [0.7, 2.1]; n=1095, d=468 | not applicable |
| core-s2 | kv/test keys | 97.9 [97.0, 98.6]; n=1129, d=477 | 2.4 [1.6, 3.3]; n=1129, d=477 | 2.4 [1.6, 3.3]; n=1129, d=477 | 2.4 [1.6, 3.3]; n=1129, d=477 | not applicable |
| core-s2 | delayed2h/seen pairs | 99.8 [99.5, 100.0]; n=964, d=471 | 1.8 [1.0, 2.6]; n=964, d=471 | 1.8 [1.0, 2.6]; n=964, d=471 | 1.8 [1.0, 2.6]; n=964, d=471 | 100.0 [100.0, 100.0]; n=962, d=471 |
| core-s2 | delayed2h/new pairs | 99.6 [99.1, 99.9]; n=964, d=470 | 2.2 [1.3, 3.2]; n=964, d=470 | 2.2 [1.3, 3.2]; n=964, d=470 | 2.2 [1.3, 3.2]; n=964, d=470 | 99.9 [99.7, 100.0]; n=960, d=470 |
| core-s2 | delayed2h/never-seen bytes | 99.7 [99.2, 100.0]; n=936, d=463 | 2.0 [1.2, 3.0]; n=936, d=463 | 2.0 [1.2, 3.0]; n=936, d=463 | 2.0 [1.2, 3.0]; n=936, d=463 | 99.9 [99.7, 100.0]; n=933, d=463 |
| core-s1 | kv/training keys | 98.0 [97.0, 98.7]; n=1082, d=475 | 1.9 [1.2, 2.7]; n=1082, d=475 | 1.9 [1.2, 2.7]; n=1082, d=475 | 1.9 [1.2, 2.7]; n=1082, d=475 | not applicable |
| core-s1 | kv/dev keys | 92.3 [90.7, 93.8]; n=1095, d=468 | 2.1 [1.3, 3.0]; n=1095, d=468 | 2.1 [1.3, 3.0]; n=1095, d=468 | 2.1 [1.3, 3.0]; n=1095, d=468 | not applicable |
| core-s1 | kv/test keys | 93.6 [92.2, 95.0]; n=1129, d=477 | 1.4 [0.7, 2.1]; n=1129, d=477 | 1.4 [0.7, 2.1]; n=1129, d=477 | 1.4 [0.7, 2.1]; n=1129, d=477 | not applicable |
| core-s1 | delayed2h/seen pairs | 95.7 [94.5, 97.0]; n=964, d=471 | 2.6 [1.6, 3.5]; n=964, d=471 | 2.6 [1.6, 3.5]; n=964, d=471 | 2.6 [1.6, 3.5]; n=964, d=471 | 97.9 [97.0, 98.8]; n=923, d=471 |
| core-s1 | delayed2h/new pairs | 96.1 [94.8, 97.3]; n=964, d=470 | 2.5 [1.6, 3.5]; n=964, d=470 | 2.5 [1.6, 3.5]; n=964, d=470 | 2.5 [1.6, 3.5]; n=964, d=470 | 97.7 [96.8, 98.6]; n=926, d=470 |
| core-s1 | delayed2h/never-seen bytes | 96.4 [95.2, 97.5]; n=936, d=463 | 2.1 [1.3, 3.1]; n=936, d=463 | 2.1 [1.3, 3.1]; n=936, d=463 | 2.1 [1.3, 3.1]; n=936, d=463 | 97.2 [96.0, 98.2]; n=902, d=463 |
| candidate-s0 | kv/training keys | 99.0 [98.4, 99.5]; n=1082, d=475 | 46.3 [44.1, 48.5]; n=1082, d=475 | 46.3 [44.1, 48.5]; n=1082, d=475 | 15.2 [13.5, 17.0]; n=1082, d=475 | not applicable |
| candidate-s0 | kv/dev keys | 97.2 [96.0, 98.1]; n=1095, d=468 | 43.8 [41.9, 46.0]; n=1095, d=468 | 43.8 [41.9, 46.0]; n=1095, d=468 | 12.2 [10.4, 14.0]; n=1095, d=468 | not applicable |
| candidate-s0 | kv/test keys | 97.3 [96.3, 98.3]; n=1129, d=477 | 44.6 [42.6, 46.8]; n=1129, d=477 | 44.6 [42.6, 46.8]; n=1129, d=477 | 14.3 [12.4, 16.3]; n=1129, d=477 | not applicable |
| candidate-s0 | delayed2h/seen pairs | 99.7 [99.3, 100.0]; n=964, d=471 | 53.8 [51.4, 56.6]; n=964, d=471 | 52.5 [50.0, 55.1]; n=964, d=471 | 2.1 [1.2, 2.9]; n=964, d=471 | 99.7 [99.3, 100.0]; n=961, d=471 |
| candidate-s0 | delayed2h/new pairs | 99.1 [98.4, 99.6]; n=964, d=470 | 53.3 [50.8, 56.1]; n=964, d=470 | 52.0 [49.8, 54.5]; n=964, d=470 | 1.9 [1.0, 2.8]; n=964, d=470 | 99.6 [99.1, 99.9]; n=955, d=470 |
| candidate-s0 | delayed2h/never-seen bytes | 98.1 [97.2, 98.9]; n=936, d=463 | 51.7 [49.4, 54.0]; n=936, d=463 | 50.3 [48.1, 52.5]; n=936, d=463 | 0.1 [0.0, 0.3]; n=936, d=463 | 99.5 [98.9, 99.9]; n=918, d=463 |
| candidate-s1 | kv/training keys | 98.3 [97.6, 99.0]; n=1082, d=475 | 46.0 [43.9, 48.1]; n=1082, d=475 | 46.1 [43.9, 48.4]; n=1082, d=475 | 15.2 [13.5, 17.0]; n=1082, d=475 | not applicable |
| candidate-s1 | kv/dev keys | 96.0 [94.8, 97.1]; n=1095, d=468 | 43.5 [41.6, 45.7]; n=1095, d=468 | 43.7 [41.8, 45.9]; n=1095, d=468 | 12.2 [10.4, 14.0]; n=1095, d=468 | not applicable |
| candidate-s1 | kv/test keys | 98.0 [97.1, 98.7]; n=1129, d=477 | 43.9 [42.0, 46.0]; n=1129, d=477 | 44.2 [42.2, 46.3]; n=1129, d=477 | 14.3 [12.4, 16.3]; n=1129, d=477 | not applicable |
| candidate-s1 | delayed2h/seen pairs | 99.6 [99.2, 99.9]; n=964, d=471 | 55.2 [52.7, 57.8]; n=964, d=471 | 56.1 [53.7, 58.6]; n=964, d=471 | 0.0 [0.0, 0.0]; n=964, d=471 | 99.6 [99.1, 99.9]; n=960, d=471 |
| candidate-s1 | delayed2h/new pairs | 99.6 [99.1, 99.9]; n=964, d=470 | 54.4 [52.0, 57.0]; n=964, d=470 | 55.1 [52.7, 57.8]; n=964, d=470 | 0.0 [0.0, 0.0]; n=964, d=470 | 99.0 [98.3, 99.6]; n=960, d=470 |
| candidate-s1 | delayed2h/never-seen bytes | 99.1 [98.5, 99.7]; n=936, d=463 | 53.6 [51.3, 56.1]; n=936, d=463 | 54.1 [51.6, 56.5]; n=936, d=463 | 0.0 [0.0, 0.0]; n=936, d=463 | 99.6 [99.1, 99.9]; n=928, d=463 |
| candidate-s2 | kv/training keys | 97.6 [96.6, 98.4]; n=1082, d=475 | 38.2 [35.8, 40.4]; n=1082, d=475 | 35.7 [33.4, 38.0]; n=1082, d=475 | 15.2 [13.5, 17.0]; n=1082, d=475 | not applicable |
| candidate-s2 | kv/dev keys | 96.6 [95.7, 97.7]; n=1095, d=468 | 37.0 [34.9, 39.3]; n=1095, d=468 | 32.0 [29.8, 34.4]; n=1095, d=468 | 12.2 [10.4, 14.0]; n=1095, d=468 | not applicable |
| candidate-s2 | kv/test keys | 97.4 [96.5, 98.3]; n=1129, d=477 | 37.5 [35.1, 39.5]; n=1129, d=477 | 34.9 [32.5, 37.2]; n=1129, d=477 | 14.3 [12.4, 16.3]; n=1129, d=477 | not applicable |
| candidate-s2 | delayed2h/seen pairs | 99.3 [98.7, 99.8]; n=964, d=471 | 35.0 [32.4, 37.9]; n=964, d=471 | 32.1 [29.6, 34.9]; n=964, d=471 | 0.0 [0.0, 0.0]; n=964, d=471 | 99.5 [99.0, 99.9]; n=957, d=471 |
| candidate-s2 | delayed2h/new pairs | 99.0 [98.3, 99.6]; n=964, d=470 | 36.4 [34.0, 39.2]; n=964, d=470 | 33.1 [30.6, 35.7]; n=964, d=470 | 0.0 [0.0, 0.0]; n=964, d=470 | 98.5 [97.7, 99.3]; n=954, d=470 |
| candidate-s2 | delayed2h/never-seen bytes | 97.9 [96.9, 98.7]; n=936, d=463 | 40.6 [38.1, 43.2]; n=936, d=463 | 38.7 [36.2, 41.2]; n=936, d=463 | 0.0 [0.0, 0.0]; n=936, d=463 | 99.3 [98.8, 99.8]; n=916, d=462 |
| candidate-s3 | kv/training keys | 98.4 [97.6, 99.2]; n=1082, d=475 | 43.8 [41.4, 46.0]; n=1082, d=475 | 43.9 [41.6, 46.0]; n=1082, d=475 | 15.2 [13.5, 17.0]; n=1082, d=475 | not applicable |
| candidate-s3 | kv/dev keys | 98.0 [97.1, 98.8]; n=1095, d=468 | 41.6 [39.4, 43.9]; n=1095, d=468 | 41.2 [39.1, 43.6]; n=1095, d=468 | 12.2 [10.4, 14.0]; n=1095, d=468 | not applicable |
| candidate-s3 | kv/test keys | 94.9 [93.6, 96.2]; n=1129, d=477 | 42.9 [40.7, 45.1]; n=1129, d=477 | 42.6 [40.4, 44.9]; n=1129, d=477 | 14.3 [12.4, 16.3]; n=1129, d=477 | not applicable |
| candidate-s3 | delayed2h/seen pairs | 98.8 [98.0, 99.4]; n=964, d=471 | 40.4 [38.2, 42.8]; n=964, d=471 | 39.0 [36.7, 41.4]; n=964, d=471 | 0.0 [0.0, 0.0]; n=964, d=471 | 98.9 [98.3, 99.5]; n=952, d=471 |
| candidate-s3 | delayed2h/new pairs | 98.1 [97.3, 98.9]; n=964, d=470 | 41.6 [39.6, 43.8]; n=964, d=470 | 40.1 [38.0, 42.5]; n=964, d=470 | 0.0 [0.0, 0.0]; n=964, d=470 | 98.7 [98.0, 99.4]; n=946, d=470 |
| candidate-s3 | delayed2h/never-seen bytes | 98.4 [97.5, 99.1]; n=936, d=463 | 41.2 [39.0, 43.4]; n=936, d=463 | 39.5 [37.2, 41.9]; n=936, d=463 | 0.0 [0.0, 0.0]; n=936, d=463 | 99.0 [98.4, 99.6]; n=921, d=462 |
| candidate-s4 | kv/training keys | 97.9 [97.0, 98.7]; n=1082, d=475 | 37.7 [35.3, 40.0]; n=1082, d=475 | 32.1 [29.7, 34.4]; n=1082, d=475 | 14.5 [12.8, 16.3]; n=1082, d=475 | not applicable |
| candidate-s4 | kv/dev keys | 97.4 [96.4, 98.3]; n=1095, d=468 | 36.8 [34.7, 38.9]; n=1095, d=468 | 31.5 [29.4, 33.8]; n=1095, d=468 | 4.4 [3.3, 5.7]; n=1095, d=468 | not applicable |
| candidate-s4 | kv/test keys | 96.9 [95.9, 97.8]; n=1129, d=477 | 35.2 [33.2, 37.2]; n=1129, d=477 | 30.6 [28.4, 32.6]; n=1129, d=477 | 2.3 [1.5, 3.2]; n=1129, d=477 | not applicable |
| candidate-s4 | delayed2h/seen pairs | 99.5 [99.0, 99.9]; n=964, d=471 | 48.7 [46.5, 51.0]; n=964, d=471 | 44.7 [42.3, 47.1]; n=964, d=471 | 1.1 [0.5, 1.8]; n=964, d=471 | 99.5 [99.0, 99.9]; n=959, d=471 |
| candidate-s4 | delayed2h/new pairs | 99.0 [98.3, 99.5]; n=964, d=470 | 48.3 [46.2, 50.7]; n=964, d=470 | 44.8 [42.5, 47.3]; n=964, d=470 | 1.6 [0.8, 2.4]; n=964, d=470 | 99.8 [99.5, 100.0]; n=954, d=470 |
| candidate-s4 | delayed2h/never-seen bytes | 99.5 [98.9, 99.9]; n=936, d=463 | 47.4 [45.1, 49.7]; n=936, d=463 | 42.1 [39.8, 44.6]; n=936, d=463 | 0.2 [0.0, 0.5]; n=936, d=463 | 99.1 [98.6, 99.7]; n=931, d=463 |

## Gate M feasibility on far-strict

The 50-point condition can be attainable only if N3≤50%; at observed N0 it requires N3≤N0−50 points. The paired gap interval resamples the same documents for N0 and N3. The ceiling is inferred as 100−N3. A positive diagnostic does not preregister a balanced gate population or establish utility.

| Checkpoint | Task / keys | N3 | N0−N3, points [95% CI] | Maximum gap, points [95% CI] | Point estimate reaches 50? |
| --- | --- | --- | --- | --- | --- |
| reader32-s1 | kv/training keys | 6.6 [5.2, 7.9]; n=1082, d=475 | 92.4 [90.8, 93.9]; n=1082, d=475 | 93.4 [92.1, 94.8]; n=1082, d=475 | yes |
| reader32-s1 | kv/dev keys | 4.9 [3.7, 6.3]; n=1095, d=468 | 93.5 [92.1, 95.0]; n=1095, d=468 | 95.1 [93.7, 96.3]; n=1095, d=468 | yes |
| reader32-s1 | kv/test keys | 5.5 [4.2, 6.8]; n=1129, d=477 | 93.5 [92.1, 94.9]; n=1129, d=477 | 94.5 [93.2, 95.8]; n=1129, d=477 | yes |
| reader32-s1 | delayed2h/seen pairs | 1.0 [0.5, 1.8]; n=964, d=471 | 98.4 [97.7, 99.1]; n=964, d=471 | 99.0 [98.2, 99.5]; n=964, d=471 | yes |
| reader32-s1 | delayed2h/new pairs | 0.2 [0.0, 0.5]; n=964, d=470 | 99.1 [98.4, 99.6]; n=964, d=470 | 99.8 [99.5, 100.0]; n=964, d=470 | yes |
| reader32-s1 | delayed2h/never-seen bytes | 0.7 [0.2, 1.4]; n=936, d=463 | 98.5 [97.7, 99.2]; n=936, d=463 | 99.3 [98.6, 99.8]; n=936, d=463 | yes |
| core-s0 | kv/training keys | 1.8 [1.1, 2.7]; n=1082, d=475 | 95.7 [94.4, 96.9]; n=1082, d=475 | 98.2 [97.3, 98.9]; n=1082, d=475 | yes |
| core-s0 | kv/dev keys | 1.8 [1.1, 2.6]; n=1095, d=468 | 95.1 [93.6, 96.3]; n=1095, d=468 | 98.2 [97.4, 98.9]; n=1095, d=468 | yes |
| core-s0 | kv/test keys | 1.6 [0.9, 2.3]; n=1129, d=477 | 96.0 [94.8, 97.1]; n=1129, d=477 | 98.4 [97.7, 99.1]; n=1129, d=477 | yes |
| core-s0 | delayed2h/seen pairs | 1.7 [0.9, 2.5]; n=964, d=471 | 94.5 [93.0, 95.9]; n=964, d=471 | 98.3 [97.5, 99.1]; n=964, d=471 | yes |
| core-s0 | delayed2h/new pairs | 1.8 [1.0, 2.6]; n=964, d=470 | 94.5 [93.0, 95.8]; n=964, d=470 | 98.2 [97.4, 99.0]; n=964, d=470 | yes |
| core-s0 | delayed2h/never-seen bytes | 1.0 [0.4, 1.6]; n=936, d=463 | 95.6 [94.2, 97.0]; n=936, d=463 | 99.0 [98.4, 99.6]; n=936, d=463 | yes |
| reader32-s0-15k | kv/training keys | 4.1 [3.0, 5.2]; n=1082, d=475 | 94.2 [92.8, 95.5]; n=1082, d=475 | 95.9 [94.8, 97.0]; n=1082, d=475 | yes |
| reader32-s0-15k | kv/dev keys | 4.4 [3.3, 5.6]; n=1095, d=468 | 94.1 [92.6, 95.3]; n=1095, d=468 | 95.6 [94.4, 96.7]; n=1095, d=468 | yes |
| reader32-s0-15k | kv/test keys | 4.9 [3.8, 6.1]; n=1129, d=477 | 93.9 [92.6, 95.2]; n=1129, d=477 | 95.1 [93.9, 96.2]; n=1129, d=477 | yes |
| reader32-s0-15k | delayed2h/seen pairs | 1.2 [0.6, 2.0]; n=964, d=471 | 98.2 [97.4, 99.0]; n=964, d=471 | 98.8 [98.0, 99.4]; n=964, d=471 | yes |
| reader32-s0-15k | delayed2h/new pairs | 1.8 [1.0, 2.7]; n=964, d=470 | 97.8 [96.9, 98.7]; n=964, d=470 | 98.2 [97.3, 99.0]; n=964, d=470 | yes |
| reader32-s0-15k | delayed2h/never-seen bytes | 1.4 [0.7, 2.2]; n=936, d=463 | 98.0 [97.1, 98.8]; n=936, d=463 | 98.6 [97.8, 99.3]; n=936, d=463 | yes |
| reader32-s0 | kv/training keys | 9.2 [7.7, 10.7]; n=1082, d=475 | 85.4 [83.2, 87.5]; n=1082, d=475 | 90.8 [89.3, 92.3]; n=1082, d=475 | yes |
| reader32-s0 | kv/dev keys | 7.9 [6.4, 9.6]; n=1095, d=468 | 83.4 [81.1, 85.6]; n=1095, d=468 | 92.1 [90.4, 93.6]; n=1095, d=468 | yes |
| reader32-s0 | kv/test keys | 9.2 [7.7, 10.8]; n=1129, d=477 | 83.8 [81.6, 85.9]; n=1129, d=477 | 90.8 [89.2, 92.3]; n=1129, d=477 | yes |
| reader32-s0 | delayed2h/seen pairs | 0.8 [0.3, 1.5]; n=964, d=471 | 97.7 [96.7, 98.6]; n=964, d=471 | 99.2 [98.5, 99.7]; n=964, d=471 | yes |
| reader32-s0 | delayed2h/new pairs | 0.7 [0.3, 1.3]; n=964, d=470 | 98.2 [97.3, 99.0]; n=964, d=470 | 99.3 [98.7, 99.7]; n=964, d=470 | yes |
| reader32-s0 | delayed2h/never-seen bytes | 1.2 [0.5, 2.0]; n=936, d=463 | 97.6 [96.6, 98.5]; n=936, d=463 | 98.8 [98.0, 99.5]; n=936, d=463 | yes |
| reader32-s2 | kv/training keys | 15.0 [13.3, 16.7]; n=1082, d=475 | 83.6 [81.8, 85.4]; n=1082, d=475 | 85.0 [83.3, 86.7]; n=1082, d=475 | yes |
| reader32-s2 | kv/dev keys | 8.7 [7.0, 10.3]; n=1095, d=468 | 82.6 [80.4, 84.9]; n=1095, d=468 | 91.3 [89.7, 93.0]; n=1095, d=468 | yes |
| reader32-s2 | kv/test keys | 10.4 [8.7, 12.1]; n=1129, d=477 | 84.9 [82.8, 87.0]; n=1129, d=477 | 89.6 [87.9, 91.3]; n=1129, d=477 | yes |
| reader32-s2 | delayed2h/seen pairs | 0.1 [0.0, 0.3]; n=964, d=471 | 99.4 [98.9, 99.8]; n=964, d=471 | 99.9 [99.7, 100.0]; n=964, d=471 | yes |
| reader32-s2 | delayed2h/new pairs | 0.0 [0.0, 0.0]; n=964, d=470 | 99.1 [98.4, 99.6]; n=964, d=470 | 100.0 [100.0, 100.0]; n=964, d=470 | yes |
| reader32-s2 | delayed2h/never-seen bytes | 0.2 [0.0, 0.5]; n=936, d=463 | 86.2 [84.2, 88.0]; n=936, d=463 | 99.8 [99.5, 100.0]; n=936, d=463 | yes |
| reader-full-s0 | kv/training keys | 23.4 [21.3, 25.6]; n=1082, d=475 | 75.0 [72.7, 77.2]; n=1082, d=475 | 76.6 [74.4, 78.7]; n=1082, d=475 | yes |
| reader-full-s0 | kv/dev keys | 20.6 [18.4, 22.9]; n=1095, d=468 | 77.4 [75.0, 79.8]; n=1095, d=468 | 79.4 [77.1, 81.6]; n=1095, d=468 | yes |
| reader-full-s0 | kv/test keys | 21.7 [19.6, 23.8]; n=1129, d=477 | 74.7 [72.2, 77.0]; n=1129, d=477 | 78.3 [76.2, 80.4]; n=1129, d=477 | yes |
| reader-full-s0 | delayed2h/seen pairs | 20.4 [18.0, 22.8]; n=964, d=471 | 79.3 [76.9, 81.6]; n=964, d=471 | 79.6 [77.2, 82.0]; n=964, d=471 | yes |
| reader-full-s0 | delayed2h/new pairs | 19.7 [17.5, 22.0]; n=964, d=470 | 79.8 [77.4, 82.1]; n=964, d=470 | 80.3 [78.0, 82.5]; n=964, d=470 | yes |
| reader-full-s0 | delayed2h/never-seen bytes | 15.3 [13.0, 17.5]; n=936, d=463 | 82.7 [80.5, 85.0]; n=936, d=463 | 84.7 [82.5, 87.0]; n=936, d=463 | yes |
| core-s2 | kv/training keys | 2.0 [1.3, 2.9]; n=1082, d=475 | 96.5 [95.4, 97.5]; n=1082, d=475 | 98.0 [97.1, 98.7]; n=1082, d=475 | yes |
| core-s2 | kv/dev keys | 1.4 [0.7, 2.1]; n=1095, d=468 | 96.7 [95.6, 97.7]; n=1095, d=468 | 98.6 [97.9, 99.3]; n=1095, d=468 | yes |
| core-s2 | kv/test keys | 2.4 [1.6, 3.3]; n=1129, d=477 | 95.5 [94.3, 96.6]; n=1129, d=477 | 97.6 [96.7, 98.4]; n=1129, d=477 | yes |
| core-s2 | delayed2h/seen pairs | 1.8 [1.0, 2.6]; n=964, d=471 | 98.0 [97.0, 98.9]; n=964, d=471 | 98.2 [97.4, 99.0]; n=964, d=471 | yes |
| core-s2 | delayed2h/new pairs | 2.2 [1.3, 3.2]; n=964, d=470 | 97.4 [96.3, 98.3]; n=964, d=470 | 97.8 [96.8, 98.7]; n=964, d=470 | yes |
| core-s2 | delayed2h/never-seen bytes | 2.0 [1.2, 3.0]; n=936, d=463 | 97.6 [96.5, 98.6]; n=936, d=463 | 98.0 [97.0, 98.8]; n=936, d=463 | yes |
| core-s1 | kv/training keys | 1.9 [1.2, 2.7]; n=1082, d=475 | 96.0 [94.9, 97.1]; n=1082, d=475 | 98.1 [97.3, 98.8]; n=1082, d=475 | yes |
| core-s1 | kv/dev keys | 2.1 [1.3, 3.0]; n=1095, d=468 | 90.2 [88.4, 91.9]; n=1095, d=468 | 97.9 [97.0, 98.7]; n=1095, d=468 | yes |
| core-s1 | kv/test keys | 1.4 [0.7, 2.1]; n=1129, d=477 | 92.2 [90.7, 93.7]; n=1129, d=477 | 98.6 [97.9, 99.3]; n=1129, d=477 | yes |
| core-s1 | delayed2h/seen pairs | 2.6 [1.6, 3.5]; n=964, d=471 | 93.2 [91.6, 94.8]; n=964, d=471 | 97.4 [96.5, 98.4]; n=964, d=471 | yes |
| core-s1 | delayed2h/new pairs | 2.5 [1.6, 3.5]; n=964, d=470 | 93.6 [92.0, 95.1]; n=964, d=470 | 97.5 [96.5, 98.4]; n=964, d=470 | yes |
| core-s1 | delayed2h/never-seen bytes | 2.1 [1.3, 3.1]; n=936, d=463 | 94.2 [92.7, 95.7]; n=936, d=463 | 97.9 [96.9, 98.7]; n=936, d=463 | yes |
| candidate-s0 | kv/training keys | 15.2 [13.5, 17.0]; n=1082, d=475 | 83.7 [81.9, 85.6]; n=1082, d=475 | 84.8 [83.0, 86.5]; n=1082, d=475 | yes |
| candidate-s0 | kv/dev keys | 12.2 [10.4, 14.0]; n=1095, d=468 | 84.9 [82.9, 87.0]; n=1095, d=468 | 87.8 [86.0, 89.6]; n=1095, d=468 | yes |
| candidate-s0 | kv/test keys | 14.3 [12.4, 16.3]; n=1129, d=477 | 83.0 [80.9, 85.1]; n=1129, d=477 | 85.7 [83.7, 87.6]; n=1129, d=477 | yes |
| candidate-s0 | delayed2h/seen pairs | 2.1 [1.2, 2.9]; n=964, d=471 | 97.6 [96.7, 98.5]; n=964, d=471 | 97.9 [97.1, 98.8]; n=964, d=471 | yes |
| candidate-s0 | delayed2h/new pairs | 1.9 [1.0, 2.8]; n=964, d=470 | 97.2 [96.1, 98.3]; n=964, d=470 | 98.1 [97.2, 99.0]; n=964, d=470 | yes |
| candidate-s0 | delayed2h/never-seen bytes | 0.1 [0.0, 0.3]; n=936, d=463 | 98.0 [97.1, 98.8]; n=936, d=463 | 99.9 [99.7, 100.0]; n=936, d=463 | yes |
| candidate-s1 | kv/training keys | 15.2 [13.5, 17.0]; n=1082, d=475 | 83.1 [81.2, 85.0]; n=1082, d=475 | 84.8 [83.0, 86.5]; n=1082, d=475 | yes |
| candidate-s1 | kv/dev keys | 12.2 [10.4, 14.0]; n=1095, d=468 | 83.7 [81.5, 85.9]; n=1095, d=468 | 87.8 [86.0, 89.6]; n=1095, d=468 | yes |
| candidate-s1 | kv/test keys | 14.3 [12.4, 16.3]; n=1129, d=477 | 83.6 [81.7, 85.6]; n=1129, d=477 | 85.7 [83.7, 87.6]; n=1129, d=477 | yes |
| candidate-s1 | delayed2h/seen pairs | 0.0 [0.0, 0.0]; n=964, d=471 | 99.6 [99.2, 99.9]; n=964, d=471 | 100.0 [100.0, 100.0]; n=964, d=471 | yes |
| candidate-s1 | delayed2h/new pairs | 0.0 [0.0, 0.0]; n=964, d=470 | 99.6 [99.1, 99.9]; n=964, d=470 | 100.0 [100.0, 100.0]; n=964, d=470 | yes |
| candidate-s1 | delayed2h/never-seen bytes | 0.0 [0.0, 0.0]; n=936, d=463 | 99.1 [98.5, 99.7]; n=936, d=463 | 100.0 [100.0, 100.0]; n=936, d=463 | yes |
| candidate-s2 | kv/training keys | 15.2 [13.5, 17.0]; n=1082, d=475 | 82.3 [80.3, 84.2]; n=1082, d=475 | 84.8 [83.0, 86.5]; n=1082, d=475 | yes |
| candidate-s2 | kv/dev keys | 12.2 [10.4, 14.0]; n=1095, d=468 | 84.4 [82.4, 86.5]; n=1095, d=468 | 87.8 [86.0, 89.6]; n=1095, d=468 | yes |
| candidate-s2 | kv/test keys | 14.3 [12.4, 16.3]; n=1129, d=477 | 83.1 [80.8, 85.3]; n=1129, d=477 | 85.7 [83.7, 87.6]; n=1129, d=477 | yes |
| candidate-s2 | delayed2h/seen pairs | 0.0 [0.0, 0.0]; n=964, d=471 | 99.3 [98.7, 99.8]; n=964, d=471 | 100.0 [100.0, 100.0]; n=964, d=471 | yes |
| candidate-s2 | delayed2h/new pairs | 0.0 [0.0, 0.0]; n=964, d=470 | 99.0 [98.3, 99.6]; n=964, d=470 | 100.0 [100.0, 100.0]; n=964, d=470 | yes |
| candidate-s2 | delayed2h/never-seen bytes | 0.0 [0.0, 0.0]; n=936, d=463 | 97.9 [96.9, 98.7]; n=936, d=463 | 100.0 [100.0, 100.0]; n=936, d=463 | yes |
| candidate-s3 | kv/training keys | 15.2 [13.5, 17.0]; n=1082, d=475 | 83.2 [81.2, 85.0]; n=1082, d=475 | 84.8 [83.0, 86.5]; n=1082, d=475 | yes |
| candidate-s3 | kv/dev keys | 12.2 [10.4, 14.0]; n=1095, d=468 | 85.8 [83.8, 87.8]; n=1095, d=468 | 87.8 [86.0, 89.6]; n=1095, d=468 | yes |
| candidate-s3 | kv/test keys | 14.3 [12.4, 16.3]; n=1129, d=477 | 80.5 [78.3, 82.8]; n=1129, d=477 | 85.7 [83.7, 87.6]; n=1129, d=477 | yes |
| candidate-s3 | delayed2h/seen pairs | 0.0 [0.0, 0.0]; n=964, d=471 | 98.8 [98.0, 99.4]; n=964, d=471 | 100.0 [100.0, 100.0]; n=964, d=471 | yes |
| candidate-s3 | delayed2h/new pairs | 0.0 [0.0, 0.0]; n=964, d=470 | 98.1 [97.3, 98.9]; n=964, d=470 | 100.0 [100.0, 100.0]; n=964, d=470 | yes |
| candidate-s3 | delayed2h/never-seen bytes | 0.0 [0.0, 0.0]; n=936, d=463 | 98.4 [97.5, 99.1]; n=936, d=463 | 100.0 [100.0, 100.0]; n=936, d=463 | yes |
| candidate-s4 | kv/training keys | 14.5 [12.8, 16.3]; n=1082, d=475 | 83.4 [81.3, 85.3]; n=1082, d=475 | 85.5 [83.7, 87.2]; n=1082, d=475 | yes |
| candidate-s4 | kv/dev keys | 4.4 [3.3, 5.7]; n=1095, d=468 | 93.0 [91.4, 94.4]; n=1095, d=468 | 95.6 [94.3, 96.7]; n=1095, d=468 | yes |
| candidate-s4 | kv/test keys | 2.3 [1.5, 3.2]; n=1129, d=477 | 94.6 [93.3, 95.8]; n=1129, d=477 | 97.7 [96.8, 98.5]; n=1129, d=477 | yes |
| candidate-s4 | delayed2h/seen pairs | 1.1 [0.5, 1.8]; n=964, d=471 | 98.3 [97.5, 99.1]; n=964, d=471 | 98.9 [98.2, 99.5]; n=964, d=471 | yes |
| candidate-s4 | delayed2h/new pairs | 1.6 [0.8, 2.4]; n=964, d=470 | 97.4 [96.4, 98.4]; n=964, d=470 | 98.4 [97.6, 99.2]; n=964, d=470 | yes |
| candidate-s4 | delayed2h/never-seen bytes | 0.2 [0.0, 0.5]; n=936, d=463 | 99.3 [98.7, 99.7]; n=936, d=463 | 99.8 [99.5, 100.0]; n=936, d=463 | yes |

## reader32-s1

Far lies beyond the 63-byte reader field.

Checkpoint: `/home/joe/hsn-experiments/vast2/exp/gpu3/runs/a14_reader/hold_k12nw_rd32_ag_ng2_cutC_cells64_18k_gsum-s1-6c214ef6b4d3/exp23/step_00018000.pt`. Status: **complete**.

| Task / keys | Condition | Near | Mid | Far | Far-strict | All |
| --- | --- | --- | --- | --- | --- | --- |
| kv/training keys | N0 | 99.8 [99.5, 100.0]; n=1062, d=536 | 99.7 [99.3, 100.0]; n=998, d=479 | 99.1 [98.5, 99.5]; n=1264, d=485 | 99.0 [98.3, 99.5]; n=1082, d=475 | 99.5 [99.2, 99.7]; n=3324, d=600 |
| kv/training keys | N1 | 15.3 [13.1, 17.3]; n=1062, d=536 | 13.5 [11.5, 15.9]; n=998, d=479 | 9.6 [8.1, 11.1]; n=1264, d=485 | 6.6 [5.2, 7.9]; n=1082, d=475 | 12.6 [11.5, 13.7]; n=3324, d=600 |
| kv/training keys | N2a | 9.0 [7.3, 10.6]; n=1062, d=536 | 8.7 [7.1, 10.4]; n=998, d=479 | 13.8 [12.2, 15.6]; n=1264, d=485 | 14.3 [12.6, 16.1]; n=1082, d=475 | 10.8 [9.9, 11.7]; n=3324, d=600 |
| kv/training keys | N2b-answer | 15.7 [13.5, 17.9]; n=1062, d=536 | 14.5 [12.4, 16.9]; n=998, d=479 | 16.1 [14.3, 18.1]; n=1264, d=485 | 14.2 [12.5, 16.0]; n=1082, d=475 | 15.5 [14.3, 16.7]; n=3324, d=600 |
| kv/training keys | N2b-all | 15.7 [13.5, 17.9]; n=1062, d=536 | 14.5 [12.4, 16.9]; n=998, d=479 | 16.1 [14.3, 18.1]; n=1264, d=485 | 14.2 [12.5, 16.0]; n=1082, d=475 | 15.5 [14.3, 16.7]; n=3324, d=600 |
| kv/training keys | N3 | 15.3 [13.1, 17.3]; n=1062, d=536 | 13.5 [11.5, 15.9]; n=998, d=479 | 9.6 [8.1, 11.1]; n=1264, d=485 | 6.6 [5.2, 7.9]; n=1082, d=475 | 12.6 [11.5, 13.7]; n=3324, d=600 |
| kv/training keys | N4 | 4.2 [3.1, 5.4]; n=1062, d=536 | 4.9 [3.6, 6.3]; n=998, d=479 | 4.0 [2.9, 5.0]; n=1264, d=485 | 4.2 [3.0, 5.3]; n=1082, d=475 | 4.3 [3.7, 5.1]; n=3324, d=600 |
| kv/training keys | B1 follows | 99.8 [99.4, 100.0]; n=892, d=488 | 99.6 [99.2, 100.0]; n=850, d=452 | 98.7 [97.9, 99.3]; n=1038, d=458 | 98.6 [97.8, 99.2]; n=907, d=443 | 99.3 [99.0, 99.6]; n=2780, d=589 |
| kv/training keys | B1 unchanged | 0.0 [0.0, 0.0]; n=892, d=488 | 0.0 [0.0, 0.0]; n=850, d=452 | 0.0 [0.0, 0.0]; n=1038, d=458 | 0.0 [0.0, 0.0]; n=907, d=443 | 0.0 [0.0, 0.0]; n=2780, d=589 |
| kv/training keys | B1 other | 0.2 [0.0, 0.6]; n=892, d=488 | 0.4 [0.0, 0.8]; n=850, d=452 | 1.3 [0.7, 2.1]; n=1038, d=458 | 1.4 [0.8, 2.2]; n=907, d=443 | 0.7 [0.4, 1.0]; n=2780, d=589 |
| kv/dev keys | N0 | 99.5 [99.1, 99.9]; n=1071, d=531 | 99.6 [99.2, 99.9]; n=1011, d=476 | 98.5 [97.8, 99.2]; n=1263, d=477 | 98.4 [97.7, 99.2]; n=1095, d=468 | 99.2 [98.9, 99.5]; n=3345, d=600 |
| kv/dev keys | N1 | 16.2 [13.9, 18.4]; n=1071, d=531 | 14.0 [11.9, 16.2]; n=1011, d=476 | 8.4 [6.9, 9.9]; n=1263, d=477 | 4.9 [3.7, 6.3]; n=1095, d=468 | 12.6 [11.5, 13.8]; n=3345, d=600 |
| kv/dev keys | N2a | 7.7 [6.1, 9.1]; n=1071, d=531 | 7.2 [5.5, 8.9]; n=1011, d=476 | 11.0 [9.3, 12.7]; n=1263, d=477 | 11.4 [9.7, 13.1]; n=1095, d=468 | 8.8 [7.9, 9.6]; n=3345, d=600 |
| kv/dev keys | N2b-answer | 16.3 [14.1, 18.6]; n=1071, d=531 | 15.2 [13.0, 17.5]; n=1011, d=476 | 13.5 [11.5, 15.5]; n=1263, d=477 | 11.0 [9.2, 12.7]; n=1095, d=468 | 14.9 [13.7, 16.3]; n=3345, d=600 |
| kv/dev keys | N2b-all | 16.3 [14.1, 18.6]; n=1071, d=531 | 15.2 [13.0, 17.5]; n=1011, d=476 | 13.5 [11.5, 15.5]; n=1263, d=477 | 11.0 [9.2, 12.7]; n=1095, d=468 | 14.9 [13.7, 16.3]; n=3345, d=600 |
| kv/dev keys | N3 | 16.2 [13.9, 18.4]; n=1071, d=531 | 14.0 [11.9, 16.2]; n=1011, d=476 | 8.4 [6.9, 9.9]; n=1263, d=477 | 4.9 [3.7, 6.3]; n=1095, d=468 | 12.6 [11.5, 13.8]; n=3345, d=600 |
| kv/dev keys | N4 | 4.3 [3.2, 5.6]; n=1071, d=531 | 5.0 [3.7, 6.4]; n=1011, d=476 | 3.4 [2.5, 4.5]; n=1263, d=477 | 3.7 [2.6, 4.8]; n=1095, d=468 | 4.2 [3.6, 4.9]; n=3345, d=600 |
| kv/dev keys | B1 follows | 99.7 [99.2, 100.0]; n=890, d=489 | 99.6 [99.2, 100.0]; n=851, d=442 | 99.5 [99.0, 99.9]; n=1062, d=455 | 99.5 [98.9, 99.9]; n=944, d=436 | 99.6 [99.4, 99.8]; n=2803, d=595 |
| kv/dev keys | B1 unchanged | 0.0 [0.0, 0.0]; n=890, d=489 | 0.0 [0.0, 0.0]; n=851, d=442 | 0.0 [0.0, 0.0]; n=1062, d=455 | 0.0 [0.0, 0.0]; n=944, d=436 | 0.0 [0.0, 0.0]; n=2803, d=595 |
| kv/dev keys | B1 other | 0.3 [0.0, 0.8]; n=890, d=489 | 0.4 [0.0, 0.8]; n=851, d=442 | 0.5 [0.1, 1.0]; n=1062, d=455 | 0.5 [0.1, 1.1]; n=944, d=436 | 0.4 [0.2, 0.6]; n=2803, d=595 |
| kv/test keys | N0 | 99.8 [99.5, 100.0]; n=1025, d=528 | 99.4 [98.9, 99.8]; n=1041, d=480 | 99.0 [98.4, 99.5]; n=1298, d=487 | 99.0 [98.4, 99.6]; n=1129, d=477 | 99.4 [99.1, 99.6]; n=3364, d=600 |
| kv/test keys | N1 | 14.9 [12.8, 17.2]; n=1025, d=528 | 16.4 [14.4, 18.7]; n=1041, d=480 | 9.5 [7.8, 11.3]; n=1298, d=487 | 5.5 [4.2, 6.8]; n=1129, d=477 | 13.3 [12.1, 14.5]; n=3364, d=600 |
| kv/test keys | N2a | 7.6 [6.2, 9.2]; n=1025, d=528 | 8.2 [6.5, 9.9]; n=1041, d=480 | 11.6 [9.9, 13.2]; n=1298, d=487 | 12.2 [10.4, 13.9]; n=1129, d=477 | 9.3 [8.3, 10.3]; n=3364, d=600 |
| kv/test keys | N2b-answer | 15.2 [13.1, 17.5]; n=1025, d=528 | 17.9 [15.8, 20.2]; n=1041, d=480 | 15.7 [13.7, 17.9]; n=1298, d=487 | 12.7 [10.9, 14.5]; n=1129, d=477 | 16.2 [14.9, 17.5]; n=3364, d=600 |
| kv/test keys | N2b-all | 15.2 [13.1, 17.5]; n=1025, d=528 | 17.9 [15.8, 20.2]; n=1041, d=480 | 15.7 [13.7, 17.9]; n=1298, d=487 | 12.7 [10.9, 14.5]; n=1129, d=477 | 16.2 [14.9, 17.5]; n=3364, d=600 |
| kv/test keys | N3 | 14.9 [12.8, 17.2]; n=1025, d=528 | 16.4 [14.4, 18.7]; n=1041, d=480 | 9.5 [7.8, 11.3]; n=1298, d=487 | 5.5 [4.2, 6.8]; n=1129, d=477 | 13.3 [12.1, 14.5]; n=3364, d=600 |
| kv/test keys | N4 | 3.0 [2.0, 4.2]; n=1025, d=528 | 4.4 [3.2, 5.7]; n=1041, d=480 | 3.5 [2.5, 4.5]; n=1298, d=487 | 3.7 [2.7, 4.9]; n=1129, d=477 | 3.6 [3.0, 4.3]; n=3364, d=600 |
| kv/test keys | B1 follows | 100.0 [100.0, 100.0]; n=865, d=483 | 99.6 [99.2, 100.0]; n=851, d=443 | 99.2 [98.7, 99.7]; n=1066, d=466 | 99.3 [98.7, 99.8]; n=958, d=451 | 99.6 [99.4, 99.8]; n=2782, d=598 |
| kv/test keys | B1 unchanged | 0.0 [0.0, 0.0]; n=865, d=483 | 0.0 [0.0, 0.0]; n=851, d=443 | 0.0 [0.0, 0.0]; n=1066, d=466 | 0.0 [0.0, 0.0]; n=958, d=451 | 0.0 [0.0, 0.0]; n=2782, d=598 |
| kv/test keys | B1 other | 0.0 [0.0, 0.0]; n=865, d=483 | 0.4 [0.0, 0.8]; n=851, d=443 | 0.8 [0.3, 1.3]; n=1066, d=466 | 0.7 [0.2, 1.3]; n=958, d=451 | 0.4 [0.2, 0.6]; n=2782, d=598 |
| delayed2h/seen pairs | N0 | 99.0 [96.9, 100.0]; n=103, d=54 | 100.0 [100.0, 100.0]; n=175, d=92 | 99.5 [99.0, 99.9]; n=977, d=471 | 99.5 [99.0, 99.9]; n=964, d=471 | 99.5 [99.1, 99.8]; n=1255, d=600 |
| delayed2h/seen pairs | N1 | 0.0 [0.0, 0.0]; n=103, d=54 | 0.6 [0.0, 2.0]; n=175, d=92 | 1.0 [0.5, 1.7]; n=977, d=471 | 1.0 [0.5, 1.8]; n=964, d=471 | 0.9 [0.4, 1.4]; n=1255, d=600 |
| delayed2h/seen pairs | N2a | 2.9 [0.0, 6.7]; n=103, d=54 | 2.9 [0.6, 5.7]; n=175, d=92 | 2.0 [1.2, 2.9]; n=977, d=471 | 2.1 [1.2, 2.9]; n=964, d=471 | 2.2 [1.4, 3.0]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-answer | 1.9 [0.0, 5.5]; n=103, d=54 | 0.6 [0.0, 2.0]; n=175, d=92 | 1.2 [0.6, 1.9]; n=977, d=471 | 1.2 [0.6, 1.9]; n=964, d=471 | 1.2 [0.6, 1.8]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-all | 1.9 [0.0, 5.5]; n=103, d=54 | 0.6 [0.0, 2.0]; n=175, d=92 | 1.2 [0.6, 1.9]; n=977, d=471 | 1.2 [0.6, 1.9]; n=964, d=471 | 1.2 [0.6, 1.8]; n=1255, d=600 |
| delayed2h/seen pairs | N3 | 0.0 [0.0, 0.0]; n=103, d=54 | 0.6 [0.0, 2.0]; n=175, d=92 | 1.0 [0.5, 1.7]; n=977, d=471 | 1.0 [0.5, 1.8]; n=964, d=471 | 0.9 [0.4, 1.4]; n=1255, d=600 |
| delayed2h/seen pairs | N4 | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.2 [0.0, 0.5]; n=977, d=471 | 0.2 [0.0, 0.5]; n=964, d=471 | 0.2 [0.0, 0.4]; n=1255, d=600 |
| delayed2h/seen pairs | B1 follows | 99.0 [96.8, 100.0]; n=102, d=54 | 100.0 [100.0, 100.0]; n=175, d=92 | 99.8 [99.5, 100.0]; n=972, d=471 | 99.8 [99.5, 100.0]; n=959, d=471 | 99.8 [99.4, 100.0]; n=1249, d=600 |
| delayed2h/seen pairs | B1 unchanged | 0.0 [0.0, 0.0]; n=102, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=972, d=471 | 0.0 [0.0, 0.0]; n=959, d=471 | 0.0 [0.0, 0.0]; n=1249, d=600 |
| delayed2h/seen pairs | B1 other | 1.0 [0.0, 3.2]; n=102, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.2 [0.0, 0.5]; n=972, d=471 | 0.2 [0.0, 0.5]; n=959, d=471 | 0.2 [0.0, 0.6]; n=1249, d=600 |
| delayed2h/seen pairs | B2 follows | NE: 55.6 [43.4, 66.7]; n=72, d=37 | 52.2 [45.2, 59.2]; n=136, d=69 | 60.3 [56.8, 63.6]; n=721, d=343 | 60.3 [56.7, 63.5]; n=710, d=342 | 58.8 [55.8, 61.5]; n=929, d=433 |
| delayed2h/seen pairs | B2 unchanged | NE: 26.4 [16.9, 38.5]; n=72, d=37 | 27.9 [20.9, 35.8]; n=136, d=69 | 23.9 [21.0, 27.1]; n=721, d=343 | 23.9 [21.0, 27.2]; n=710, d=342 | 24.7 [22.2, 27.5]; n=929, d=433 |
| delayed2h/seen pairs | B2 other | NE: 18.1 [11.1, 25.3]; n=72, d=37 | 19.9 [13.2, 26.5]; n=136, d=69 | 15.8 [13.0, 18.9]; n=721, d=343 | 15.8 [12.9, 18.8]; n=710, d=342 | 16.6 [14.1, 19.1]; n=929, d=433 |
| delayed2h/seen pairs | B3 follows | 99.0 [96.8, 100.0]; n=102, d=54 | 100.0 [100.0, 100.0]; n=175, d=92 | 99.8 [99.5, 100.0]; n=972, d=471 | 99.8 [99.5, 100.0]; n=959, d=471 | 99.8 [99.4, 100.0]; n=1249, d=600 |
| delayed2h/seen pairs | B3 unchanged | 0.0 [0.0, 0.0]; n=102, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=972, d=471 | 0.0 [0.0, 0.0]; n=959, d=471 | 0.0 [0.0, 0.0]; n=1249, d=600 |
| delayed2h/seen pairs | B3 other | 1.0 [0.0, 3.2]; n=102, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.2 [0.0, 0.5]; n=972, d=471 | 0.2 [0.0, 0.5]; n=959, d=471 | 0.2 [0.0, 0.6]; n=1249, d=600 |
| delayed2h/new pairs | N0 | NE: 100.0 [100.0, 100.0]; n=99, d=52 | 98.3 [96.3, 100.0]; n=176, d=91 | 99.3 [98.7, 99.8]; n=979, d=471 | 99.3 [98.7, 99.8]; n=964, d=470 | 99.2 [98.7, 99.6]; n=1254, d=600 |
| delayed2h/new pairs | N1 | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.6 [0.0, 1.8]; n=176, d=91 | 0.2 [0.0, 0.5]; n=979, d=471 | 0.2 [0.0, 0.5]; n=964, d=470 | 0.2 [0.0, 0.5]; n=1254, d=600 |
| delayed2h/new pairs | N2a | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 2.8 [0.7, 5.6]; n=176, d=91 | 1.8 [1.0, 2.6]; n=979, d=471 | 1.9 [1.0, 2.7]; n=964, d=470 | 1.8 [1.1, 2.6]; n=1254, d=600 |
| delayed2h/new pairs | N2b-answer | NE: 2.0 [0.0, 5.2]; n=99, d=52 | 1.1 [0.0, 3.0]; n=176, d=91 | 1.3 [0.6, 2.0]; n=979, d=471 | 1.3 [0.6, 2.0]; n=964, d=470 | 1.4 [0.7, 2.0]; n=1254, d=600 |
| delayed2h/new pairs | N2b-all | NE: 2.0 [0.0, 5.2]; n=99, d=52 | 1.1 [0.0, 3.0]; n=176, d=91 | 1.3 [0.6, 2.0]; n=979, d=471 | 1.3 [0.6, 2.0]; n=964, d=470 | 1.4 [0.7, 2.0]; n=1254, d=600 |
| delayed2h/new pairs | N3 | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.6 [0.0, 1.8]; n=176, d=91 | 0.2 [0.0, 0.5]; n=979, d=471 | 0.2 [0.0, 0.5]; n=964, d=470 | 0.2 [0.0, 0.5]; n=1254, d=600 |
| delayed2h/new pairs | N4 | NE: 2.0 [0.0, 5.3]; n=99, d=52 | 0.6 [0.0, 1.9]; n=176, d=91 | 0.0 [0.0, 0.0]; n=979, d=471 | 0.0 [0.0, 0.0]; n=964, d=470 | 0.2 [0.0, 0.5]; n=1254, d=600 |
| delayed2h/new pairs | B1 follows | NE: 100.0 [100.0, 100.0]; n=99, d=52 | 99.4 [98.1, 100.0]; n=173, d=91 | 99.2 [98.6, 99.7]; n=972, d=471 | 99.2 [98.5, 99.7]; n=957, d=470 | 99.3 [98.8, 99.7]; n=1244, d=600 |
| delayed2h/new pairs | B1 unchanged | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=173, d=91 | 0.1 [0.0, 0.3]; n=972, d=471 | 0.1 [0.0, 0.3]; n=957, d=470 | 0.1 [0.0, 0.3]; n=1244, d=600 |
| delayed2h/new pairs | B1 other | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.6 [0.0, 1.9]; n=173, d=91 | 0.7 [0.3, 1.3]; n=972, d=471 | 0.7 [0.3, 1.3]; n=957, d=470 | 0.6 [0.2, 1.1]; n=1244, d=600 |
| delayed2h/new pairs | B2 follows | NE: 67.2 [57.1, 76.0]; n=67, d=34 | 57.2 [49.2, 64.2]; n=138, d=70 | 57.9 [54.4, 61.2]; n=751, d=358 | 57.8 [54.2, 61.2]; n=737, d=355 | 58.5 [55.4, 61.4]; n=956, d=449 |
| delayed2h/new pairs | B2 unchanged | NE: 22.4 [14.1, 31.9]; n=67, d=34 | 25.4 [18.4, 33.3]; n=138, d=70 | 27.0 [24.1, 30.0]; n=751, d=358 | 27.0 [24.0, 30.0]; n=737, d=355 | 26.5 [23.9, 29.3]; n=956, d=449 |
| delayed2h/new pairs | B2 other | NE: 10.4 [3.4, 18.2]; n=67, d=34 | 17.4 [11.0, 23.7]; n=138, d=70 | 15.0 [12.5, 17.7]; n=751, d=358 | 15.2 [12.7, 17.9]; n=737, d=355 | 15.1 [12.8, 17.4]; n=956, d=449 |
| delayed2h/new pairs | B3 follows | NE: 100.0 [100.0, 100.0]; n=99, d=52 | 99.4 [98.1, 100.0]; n=173, d=91 | 99.2 [98.6, 99.7]; n=972, d=471 | 99.2 [98.5, 99.7]; n=957, d=470 | 99.3 [98.8, 99.7]; n=1244, d=600 |
| delayed2h/new pairs | B3 unchanged | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=173, d=91 | 0.1 [0.0, 0.3]; n=972, d=471 | 0.1 [0.0, 0.3]; n=957, d=470 | 0.1 [0.0, 0.3]; n=1244, d=600 |
| delayed2h/new pairs | B3 other | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.6 [0.0, 1.9]; n=173, d=91 | 0.7 [0.3, 1.3]; n=972, d=471 | 0.7 [0.3, 1.3]; n=957, d=470 | 0.6 [0.2, 1.1]; n=1244, d=600 |
| delayed2h/never-seen bytes | N0 | 98.3 [95.7, 100.0]; n=119, d=55 | 99.0 [97.4, 100.0]; n=192, d=97 | 99.3 [98.7, 99.7]; n=953, d=463 | 99.3 [98.7, 99.7]; n=936, d=463 | 99.1 [98.6, 99.6]; n=1264, d=600 |
| delayed2h/never-seen bytes | N1 | 0.8 [0.0, 2.8]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.7 [0.2, 1.3]; n=953, d=463 | 0.7 [0.2, 1.4]; n=936, d=463 | 0.6 [0.2, 1.1]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2a | 2.5 [0.0, 5.5]; n=119, d=55 | 1.0 [0.0, 2.7]; n=192, d=97 | 1.6 [0.8, 2.3]; n=953, d=463 | 1.5 [0.8, 2.3]; n=936, d=463 | 1.6 [1.0, 2.3]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-answer | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.7 [0.3, 1.3]; n=953, d=463 | 0.7 [0.3, 1.4]; n=936, d=463 | 0.6 [0.2, 1.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-all | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.7 [0.3, 1.3]; n=953, d=463 | 0.7 [0.3, 1.4]; n=936, d=463 | 0.6 [0.2, 1.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3 | 0.8 [0.0, 2.8]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.7 [0.2, 1.3]; n=953, d=463 | 0.7 [0.2, 1.4]; n=936, d=463 | 0.6 [0.2, 1.1]; n=1264, d=600 |
| delayed2h/never-seen bytes | N4 | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.1 [0.0, 0.3]; n=953, d=463 | 0.1 [0.0, 0.3]; n=936, d=463 | 0.1 [0.0, 0.2]; n=1264, d=600 |
| delayed2h/never-seen bytes | B1 follows | 99.1 [97.2, 100.0]; n=117, d=55 | 100.0 [100.0, 100.0]; n=190, d=97 | 99.6 [99.1, 99.9]; n=946, d=463 | 99.6 [99.1, 99.9]; n=929, d=463 | 99.6 [99.2, 99.9]; n=1253, d=600 |
| delayed2h/never-seen bytes | B1 unchanged | 0.0 [0.0, 0.0]; n=117, d=55 | 0.0 [0.0, 0.0]; n=190, d=97 | 0.0 [0.0, 0.0]; n=946, d=463 | 0.0 [0.0, 0.0]; n=929, d=463 | 0.0 [0.0, 0.0]; n=1253, d=600 |
| delayed2h/never-seen bytes | B1 other | 0.9 [0.0, 2.8]; n=117, d=55 | 0.0 [0.0, 0.0]; n=190, d=97 | 0.4 [0.1, 0.9]; n=946, d=463 | 0.4 [0.1, 0.9]; n=929, d=463 | 0.4 [0.1, 0.8]; n=1253, d=600 |
| delayed2h/never-seen bytes | B2 follows | NE: 64.8 [54.8, 76.9]; n=88, d=39 | 63.1 [55.0, 71.3]; n=141, d=71 | 62.9 [59.4, 66.0]; n=727, d=353 | 63.1 [59.5, 66.3]; n=713, d=351 | 63.1 [60.0, 66.2]; n=956, d=451 |
| delayed2h/never-seen bytes | B2 unchanged | NE: 25.0 [16.8, 32.3]; n=88, d=39 | 23.4 [17.0, 30.4]; n=141, d=71 | 23.5 [20.5, 26.8]; n=727, d=353 | 23.4 [20.4, 26.7]; n=713, d=351 | 23.6 [21.1, 26.4]; n=956, d=451 |
| delayed2h/never-seen bytes | B2 other | NE: 10.2 [3.8, 17.3]; n=88, d=39 | 13.5 [7.6, 19.7]; n=141, d=71 | 13.6 [11.3, 16.3]; n=727, d=353 | 13.5 [11.1, 16.2]; n=713, d=351 | 13.3 [11.3, 15.6]; n=956, d=451 |
| delayed2h/never-seen bytes | B3 follows | 99.1 [97.2, 100.0]; n=117, d=55 | 100.0 [100.0, 100.0]; n=190, d=97 | 99.6 [99.1, 99.9]; n=946, d=463 | 99.6 [99.1, 99.9]; n=929, d=463 | 99.6 [99.2, 99.9]; n=1253, d=600 |
| delayed2h/never-seen bytes | B3 unchanged | 0.0 [0.0, 0.0]; n=117, d=55 | 0.0 [0.0, 0.0]; n=190, d=97 | 0.0 [0.0, 0.0]; n=946, d=463 | 0.0 [0.0, 0.0]; n=929, d=463 | 0.0 [0.0, 0.0]; n=1253, d=600 |
| delayed2h/never-seen bytes | B3 other | 0.9 [0.0, 2.8]; n=117, d=55 | 0.0 [0.0, 0.0]; n=190, d=97 | 0.4 [0.1, 0.9]; n=946, d=463 | 0.4 [0.1, 0.9]; n=929, d=463 | 0.4 [0.1, 0.8]; n=1253, d=600 |

## core-s0

There is no reader; all buckets are distance strata only.

Checkpoint: `/home/joe/hsn-experiments/vast2/exp/gpu3/runs/a14_reader/hold_k12nw_ag_ng2_cutC_cells64_18k_gsum-s0-e4384c6e12d3/exp22/step_00018000.pt`. Status: **complete**.

| Task / keys | Condition | Near | Mid | Far | Far-strict | All |
| --- | --- | --- | --- | --- | --- | --- |
| kv/training keys | N0 | 99.3 [98.8, 99.8]; n=1062, d=536 | 98.6 [97.9, 99.3]; n=998, d=479 | 97.5 [96.6, 98.3]; n=1264, d=485 | 97.5 [96.6, 98.4]; n=1082, d=475 | 98.4 [98.0, 98.8]; n=3324, d=600 |
| kv/training keys | N1 | 2.0 [1.2, 2.9]; n=1062, d=536 | 1.3 [0.6, 2.0]; n=998, d=479 | 1.8 [1.1, 2.7]; n=1264, d=485 | 1.8 [1.1, 2.7]; n=1082, d=475 | 1.7 [1.3, 2.2]; n=3324, d=600 |
| kv/training keys | N2a | 4.9 [3.6, 6.2]; n=1062, d=536 | 5.1 [3.8, 6.4]; n=998, d=479 | 4.2 [3.1, 5.4]; n=1264, d=485 | 4.4 [3.3, 5.7]; n=1082, d=475 | 4.7 [4.0, 5.4]; n=3324, d=600 |
| kv/training keys | N2b-answer | 15.8 [13.6, 17.9]; n=1062, d=536 | 14.5 [12.4, 16.9]; n=998, d=479 | 17.0 [15.2, 18.9]; n=1264, d=485 | 15.2 [13.5, 17.0]; n=1082, d=475 | 15.9 [14.6, 17.1]; n=3324, d=600 |
| kv/training keys | N2b-all | 15.8 [13.6, 17.9]; n=1062, d=536 | 14.5 [12.4, 16.9]; n=998, d=479 | 17.0 [15.2, 18.9]; n=1264, d=485 | 15.2 [13.5, 17.0]; n=1082, d=475 | 15.9 [14.6, 17.1]; n=3324, d=600 |
| kv/training keys | N3 | 2.0 [1.2, 2.9]; n=1062, d=536 | 1.3 [0.6, 2.0]; n=998, d=479 | 1.8 [1.1, 2.7]; n=1264, d=485 | 1.8 [1.1, 2.7]; n=1082, d=475 | 1.7 [1.3, 2.2]; n=3324, d=600 |
| kv/training keys | B1 follows | 99.5 [99.1, 99.9]; n=887, d=487 | 99.3 [98.7, 99.8]; n=842, d=450 | 98.4 [97.6, 99.2]; n=1019, d=458 | 98.3 [97.4, 99.1]; n=891, d=443 | 99.1 [98.7, 99.4]; n=2748, d=589 |
| kv/training keys | B1 unchanged | 0.0 [0.0, 0.0]; n=887, d=487 | 0.0 [0.0, 0.0]; n=842, d=450 | 0.0 [0.0, 0.0]; n=1019, d=458 | 0.0 [0.0, 0.0]; n=891, d=443 | 0.0 [0.0, 0.0]; n=2748, d=589 |
| kv/training keys | B1 other | 0.5 [0.1, 0.9]; n=887, d=487 | 0.7 [0.2, 1.3]; n=842, d=450 | 1.6 [0.8, 2.4]; n=1019, d=458 | 1.7 [0.9, 2.6]; n=891, d=443 | 0.9 [0.6, 1.3]; n=2748, d=589 |
| kv/dev keys | N0 | 98.1 [97.3, 98.9]; n=1071, d=531 | 98.0 [97.1, 98.8]; n=1011, d=476 | 97.0 [96.0, 97.9]; n=1263, d=477 | 96.9 [95.8, 97.9]; n=1095, d=468 | 97.7 [97.1, 98.2]; n=3345, d=600 |
| kv/dev keys | N1 | 2.6 [1.7, 3.6]; n=1071, d=531 | 3.1 [2.1, 4.2]; n=1011, d=476 | 1.9 [1.2, 2.7]; n=1263, d=477 | 1.8 [1.1, 2.6]; n=1095, d=468 | 2.5 [2.0, 3.0]; n=3345, d=600 |
| kv/dev keys | N2a | 3.8 [2.8, 4.9]; n=1071, d=531 | 3.9 [2.7, 5.2]; n=1011, d=476 | 4.0 [3.0, 5.1]; n=1263, d=477 | 4.4 [3.2, 5.6]; n=1095, d=468 | 3.9 [3.3, 4.5]; n=3345, d=600 |
| kv/dev keys | N2b-answer | 16.5 [14.3, 18.7]; n=1071, d=531 | 15.5 [13.4, 17.8]; n=1011, d=476 | 14.6 [12.6, 16.5]; n=1263, d=477 | 12.2 [10.4, 14.0]; n=1095, d=468 | 15.5 [14.2, 16.8]; n=3345, d=600 |
| kv/dev keys | N2b-all | 16.5 [14.3, 18.7]; n=1071, d=531 | 15.5 [13.4, 17.8]; n=1011, d=476 | 14.6 [12.6, 16.5]; n=1263, d=477 | 12.2 [10.4, 14.0]; n=1095, d=468 | 15.5 [14.2, 16.8]; n=3345, d=600 |
| kv/dev keys | N3 | 2.6 [1.7, 3.6]; n=1071, d=531 | 3.1 [2.1, 4.2]; n=1011, d=476 | 1.9 [1.2, 2.7]; n=1263, d=477 | 1.8 [1.1, 2.6]; n=1095, d=468 | 2.5 [2.0, 3.0]; n=3345, d=600 |
| kv/dev keys | B1 follows | 99.2 [98.6, 99.8]; n=875, d=485 | 98.2 [97.4, 99.0]; n=834, d=437 | 98.0 [97.1, 98.8]; n=1046, d=453 | 98.0 [96.9, 98.8]; n=931, d=438 | 98.4 [98.0, 98.9]; n=2755, d=595 |
| kv/dev keys | B1 unchanged | 0.0 [0.0, 0.0]; n=875, d=485 | 0.0 [0.0, 0.0]; n=834, d=437 | 0.1 [0.0, 0.3]; n=1046, d=453 | 0.1 [0.0, 0.3]; n=931, d=438 | 0.0 [0.0, 0.1]; n=2755, d=595 |
| kv/dev keys | B1 other | 0.8 [0.2, 1.4]; n=875, d=485 | 1.8 [1.0, 2.6]; n=834, d=437 | 1.9 [1.1, 2.8]; n=1046, d=453 | 1.9 [1.1, 3.0]; n=931, d=438 | 1.5 [1.1, 2.0]; n=2755, d=595 |
| kv/test keys | N0 | 99.1 [98.5, 99.6]; n=1025, d=528 | 99.1 [98.6, 99.6]; n=1041, d=480 | 97.3 [96.4, 98.1]; n=1298, d=487 | 97.6 [96.6, 98.4]; n=1129, d=477 | 98.4 [98.0, 98.8]; n=3364, d=600 |
| kv/test keys | N1 | 2.5 [1.7, 3.6]; n=1025, d=528 | 1.7 [1.1, 2.6]; n=1041, d=480 | 1.5 [0.8, 2.1]; n=1298, d=487 | 1.6 [0.9, 2.3]; n=1129, d=477 | 1.9 [1.4, 2.4]; n=3364, d=600 |
| kv/test keys | N2a | 4.1 [3.0, 5.3]; n=1025, d=528 | 4.0 [2.9, 5.2]; n=1041, d=480 | 4.7 [3.6, 5.8]; n=1298, d=487 | 5.2 [3.9, 6.5]; n=1129, d=477 | 4.3 [3.6, 5.0]; n=3364, d=600 |
| kv/test keys | N2b-answer | 15.6 [13.4, 17.8]; n=1025, d=528 | 17.9 [15.7, 20.2]; n=1041, d=480 | 17.2 [15.1, 19.3]; n=1298, d=487 | 14.3 [12.4, 16.3]; n=1129, d=477 | 16.9 [15.6, 18.2]; n=3364, d=600 |
| kv/test keys | N2b-all | 15.6 [13.4, 17.8]; n=1025, d=528 | 17.9 [15.7, 20.2]; n=1041, d=480 | 17.2 [15.1, 19.3]; n=1298, d=487 | 14.3 [12.4, 16.3]; n=1129, d=477 | 16.9 [15.6, 18.2]; n=3364, d=600 |
| kv/test keys | N3 | 2.5 [1.7, 3.6]; n=1025, d=528 | 1.7 [1.1, 2.6]; n=1041, d=480 | 1.5 [0.8, 2.1]; n=1298, d=487 | 1.6 [0.9, 2.3]; n=1129, d=477 | 1.9 [1.4, 2.4]; n=3364, d=600 |
| kv/test keys | B1 follows | 99.4 [98.8, 99.9]; n=859, d=482 | 98.6 [97.7, 99.3]; n=848, d=445 | 98.6 [97.8, 99.2]; n=1045, d=464 | 98.4 [97.5, 99.1]; n=940, d=450 | 98.8 [98.4, 99.2]; n=2752, d=598 |
| kv/test keys | B1 unchanged | 0.0 [0.0, 0.0]; n=859, d=482 | 0.0 [0.0, 0.0]; n=848, d=445 | 0.0 [0.0, 0.0]; n=1045, d=464 | 0.0 [0.0, 0.0]; n=940, d=450 | 0.0 [0.0, 0.0]; n=2752, d=598 |
| kv/test keys | B1 other | 0.6 [0.1, 1.2]; n=859, d=482 | 1.4 [0.7, 2.3]; n=848, d=445 | 1.4 [0.8, 2.2]; n=1045, d=464 | 1.6 [0.9, 2.5]; n=940, d=450 | 1.2 [0.8, 1.6]; n=2752, d=598 |
| delayed2h/seen pairs | N0 | 96.1 [92.4, 99.1]; n=103, d=54 | 94.3 [90.7, 97.5]; n=175, d=92 | 96.1 [94.9, 97.3]; n=977, d=471 | 96.2 [94.9, 97.3]; n=964, d=471 | 95.9 [94.6, 97.0]; n=1255, d=600 |
| delayed2h/seen pairs | N1 | 1.0 [0.0, 3.3]; n=103, d=54 | 1.1 [0.0, 2.8]; n=175, d=92 | 1.7 [1.0, 2.7]; n=977, d=471 | 1.7 [0.9, 2.5]; n=964, d=471 | 1.6 [0.9, 2.3]; n=1255, d=600 |
| delayed2h/seen pairs | N2a | 1.0 [0.0, 3.1]; n=103, d=54 | 0.6 [0.0, 1.8]; n=175, d=92 | 0.6 [0.2, 1.2]; n=977, d=471 | 0.6 [0.2, 1.2]; n=964, d=471 | 0.6 [0.2, 1.1]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-answer | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=977, d=471 | 0.0 [0.0, 0.0]; n=964, d=471 | 0.0 [0.0, 0.0]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-all | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=977, d=471 | 0.0 [0.0, 0.0]; n=964, d=471 | 0.0 [0.0, 0.0]; n=1255, d=600 |
| delayed2h/seen pairs | N3 | 1.0 [0.0, 3.3]; n=103, d=54 | 1.1 [0.0, 2.8]; n=175, d=92 | 1.7 [1.0, 2.7]; n=977, d=471 | 1.7 [0.9, 2.5]; n=964, d=471 | 1.6 [0.9, 2.3]; n=1255, d=600 |
| delayed2h/seen pairs | B1 follows | NE: 99.0 [96.6, 100.0]; n=99, d=54 | 99.4 [98.0, 100.0]; n=165, d=92 | 98.9 [98.2, 99.6]; n=939, d=471 | 98.9 [98.2, 99.6]; n=927, d=471 | 99.0 [98.5, 99.5]; n=1203, d=600 |
| delayed2h/seen pairs | B1 unchanged | NE: 0.0 [0.0, 0.0]; n=99, d=54 | 0.0 [0.0, 0.0]; n=165, d=92 | 0.0 [0.0, 0.0]; n=939, d=471 | 0.0 [0.0, 0.0]; n=927, d=471 | 0.0 [0.0, 0.0]; n=1203, d=600 |
| delayed2h/seen pairs | B1 other | NE: 1.0 [0.0, 3.4]; n=99, d=54 | 0.6 [0.0, 2.0]; n=165, d=92 | 1.1 [0.4, 1.8]; n=939, d=471 | 1.1 [0.4, 1.8]; n=927, d=471 | 1.0 [0.5, 1.5]; n=1203, d=600 |
| delayed2h/seen pairs | B2 follows | NE: 55.1 [42.9, 67.3]; n=69, d=37 | 46.5 [38.7, 54.2]; n=127, d=69 | 53.2 [49.5, 56.9]; n=693, d=343 | 53.4 [49.6, 57.1]; n=683, d=342 | 52.4 [49.0, 55.5]; n=889, d=433 |
| delayed2h/seen pairs | B2 unchanged | NE: 17.4 [9.1, 27.9]; n=69, d=37 | 23.6 [16.4, 31.0]; n=127, d=69 | 18.6 [15.7, 21.7]; n=693, d=343 | 18.6 [15.7, 21.8]; n=683, d=342 | 19.2 [16.5, 21.9]; n=889, d=433 |
| delayed2h/seen pairs | B2 other | NE: 27.5 [16.7, 39.0]; n=69, d=37 | 29.9 [22.2, 37.0]; n=127, d=69 | 28.1 [24.9, 31.4]; n=693, d=343 | 28.0 [24.6, 31.3]; n=683, d=342 | 28.3 [25.4, 31.2]; n=889, d=433 |
| delayed2h/seen pairs | B3 follows | NE: 99.0 [96.6, 100.0]; n=99, d=54 | 99.4 [98.0, 100.0]; n=165, d=92 | 98.9 [98.2, 99.6]; n=939, d=471 | 98.9 [98.2, 99.6]; n=927, d=471 | 99.0 [98.5, 99.5]; n=1203, d=600 |
| delayed2h/seen pairs | B3 unchanged | NE: 0.0 [0.0, 0.0]; n=99, d=54 | 0.0 [0.0, 0.0]; n=165, d=92 | 0.0 [0.0, 0.0]; n=939, d=471 | 0.0 [0.0, 0.0]; n=927, d=471 | 0.0 [0.0, 0.0]; n=1203, d=600 |
| delayed2h/seen pairs | B3 other | NE: 1.0 [0.0, 3.4]; n=99, d=54 | 0.6 [0.0, 2.0]; n=165, d=92 | 1.1 [0.4, 1.8]; n=939, d=471 | 1.1 [0.4, 1.8]; n=927, d=471 | 1.0 [0.5, 1.5]; n=1203, d=600 |
| delayed2h/new pairs | N0 | NE: 98.0 [95.1, 100.0]; n=99, d=52 | 96.6 [94.2, 98.9]; n=176, d=91 | 96.3 [95.1, 97.5]; n=979, d=471 | 96.3 [95.0, 97.4]; n=964, d=470 | 96.5 [95.4, 97.5]; n=1254, d=600 |
| delayed2h/new pairs | N1 | NE: 2.0 [0.0, 5.3]; n=99, d=52 | 1.1 [0.0, 3.0]; n=176, d=91 | 1.8 [1.0, 2.8]; n=979, d=471 | 1.8 [1.0, 2.6]; n=964, d=470 | 1.8 [1.1, 2.6]; n=1254, d=600 |
| delayed2h/new pairs | N2a | NE: 1.0 [0.0, 3.3]; n=99, d=52 | 0.6 [0.0, 1.9]; n=176, d=91 | 0.9 [0.4, 1.6]; n=979, d=471 | 0.9 [0.4, 1.7]; n=964, d=470 | 0.9 [0.4, 1.5]; n=1254, d=600 |
| delayed2h/new pairs | N2b-answer | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=979, d=471 | 0.0 [0.0, 0.0]; n=964, d=470 | 0.0 [0.0, 0.0]; n=1254, d=600 |
| delayed2h/new pairs | N2b-all | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=979, d=471 | 0.0 [0.0, 0.0]; n=964, d=470 | 0.0 [0.0, 0.0]; n=1254, d=600 |
| delayed2h/new pairs | N3 | NE: 2.0 [0.0, 5.3]; n=99, d=52 | 1.1 [0.0, 3.0]; n=176, d=91 | 1.8 [1.0, 2.8]; n=979, d=471 | 1.8 [1.0, 2.6]; n=964, d=470 | 1.8 [1.1, 2.6]; n=1254, d=600 |
| delayed2h/new pairs | B1 follows | NE: 96.9 [93.3, 100.0]; n=97, d=52 | 97.6 [95.3, 99.5]; n=170, d=91 | 98.4 [97.6, 99.2]; n=943, d=471 | 98.4 [97.6, 99.2]; n=928, d=470 | 98.2 [97.4, 98.9]; n=1210, d=600 |
| delayed2h/new pairs | B1 unchanged | NE: 1.0 [0.0, 3.4]; n=97, d=52 | 0.0 [0.0, 0.0]; n=170, d=91 | 0.3 [0.0, 0.7]; n=943, d=471 | 0.3 [0.0, 0.8]; n=928, d=470 | 0.3 [0.1, 0.7]; n=1210, d=600 |
| delayed2h/new pairs | B1 other | NE: 2.1 [0.0, 5.4]; n=97, d=52 | 2.4 [0.5, 4.7]; n=170, d=91 | 1.3 [0.6, 2.0]; n=943, d=471 | 1.3 [0.6, 2.0]; n=928, d=470 | 1.5 [0.8, 2.2]; n=1210, d=600 |
| delayed2h/new pairs | B2 follows | NE: 50.8 [37.7, 63.4]; n=65, d=34 | 57.8 [49.4, 66.2]; n=135, d=70 | 51.6 [47.8, 55.0]; n=725, d=358 | 51.2 [47.4, 54.6]; n=711, d=355 | 52.4 [49.1, 55.6]; n=925, d=449 |
| delayed2h/new pairs | B2 unchanged | NE: 20.0 [11.1, 31.0]; n=65, d=34 | 21.5 [15.3, 28.8]; n=135, d=70 | 18.1 [15.2, 21.2]; n=725, d=358 | 18.0 [15.1, 21.1]; n=711, d=355 | 18.7 [16.0, 21.4]; n=925, d=449 |
| delayed2h/new pairs | B2 other | NE: 29.2 [16.9, 42.4]; n=65, d=34 | 20.7 [13.8, 27.7]; n=135, d=70 | 30.3 [27.4, 33.9]; n=725, d=358 | 30.8 [27.8, 34.3]; n=711, d=355 | 28.9 [26.1, 31.7]; n=925, d=449 |
| delayed2h/new pairs | B3 follows | NE: 96.9 [93.3, 100.0]; n=97, d=52 | 97.6 [95.3, 99.5]; n=170, d=91 | 98.4 [97.6, 99.2]; n=943, d=471 | 98.4 [97.6, 99.2]; n=928, d=470 | 98.2 [97.4, 98.9]; n=1210, d=600 |
| delayed2h/new pairs | B3 unchanged | NE: 1.0 [0.0, 3.4]; n=97, d=52 | 0.0 [0.0, 0.0]; n=170, d=91 | 0.3 [0.0, 0.7]; n=943, d=471 | 0.3 [0.0, 0.8]; n=928, d=470 | 0.3 [0.1, 0.7]; n=1210, d=600 |
| delayed2h/new pairs | B3 other | NE: 2.1 [0.0, 5.4]; n=97, d=52 | 2.4 [0.5, 4.7]; n=170, d=91 | 1.3 [0.6, 2.0]; n=943, d=471 | 1.3 [0.6, 2.0]; n=928, d=470 | 1.5 [0.8, 2.2]; n=1210, d=600 |
| delayed2h/never-seen bytes | N0 | 98.3 [96.0, 100.0]; n=119, d=55 | 97.9 [95.7, 99.5]; n=192, d=97 | 96.4 [95.2, 97.6]; n=953, d=463 | 96.6 [95.3, 97.8]; n=936, d=463 | 96.8 [95.7, 97.8]; n=1264, d=600 |
| delayed2h/never-seen bytes | N1 | 2.5 [0.0, 5.7]; n=119, d=55 | 1.6 [0.0, 3.6]; n=192, d=97 | 0.9 [0.4, 1.6]; n=953, d=463 | 1.0 [0.4, 1.6]; n=936, d=463 | 1.2 [0.6, 1.8]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2a | 0.8 [0.0, 2.8]; n=119, d=55 | 1.0 [0.0, 2.9]; n=192, d=97 | 0.8 [0.3, 1.5]; n=953, d=463 | 0.9 [0.3, 1.5]; n=936, d=463 | 0.9 [0.4, 1.5]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-answer | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.0 [0.0, 0.0]; n=953, d=463 | 0.0 [0.0, 0.0]; n=936, d=463 | 0.0 [0.0, 0.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-all | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.0 [0.0, 0.0]; n=953, d=463 | 0.0 [0.0, 0.0]; n=936, d=463 | 0.0 [0.0, 0.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3 | 2.5 [0.0, 5.7]; n=119, d=55 | 1.6 [0.0, 3.6]; n=192, d=97 | 0.9 [0.4, 1.6]; n=953, d=463 | 1.0 [0.4, 1.6]; n=936, d=463 | 1.2 [0.6, 1.8]; n=1264, d=600 |
| delayed2h/never-seen bytes | B1 follows | 98.3 [95.8, 100.0]; n=117, d=55 | 95.7 [92.5, 98.4]; n=188, d=97 | 97.5 [96.4, 98.5]; n=919, d=463 | 97.5 [96.4, 98.4]; n=904, d=463 | 97.3 [96.3, 98.2]; n=1224, d=600 |
| delayed2h/never-seen bytes | B1 unchanged | 0.0 [0.0, 0.0]; n=117, d=55 | 0.0 [0.0, 0.0]; n=188, d=97 | 0.0 [0.0, 0.0]; n=919, d=463 | 0.0 [0.0, 0.0]; n=904, d=463 | 0.0 [0.0, 0.0]; n=1224, d=600 |
| delayed2h/never-seen bytes | B1 other | 1.7 [0.0, 4.2]; n=117, d=55 | 4.3 [1.6, 7.5]; n=188, d=97 | 2.5 [1.5, 3.6]; n=919, d=463 | 2.5 [1.6, 3.6]; n=904, d=463 | 2.7 [1.8, 3.7]; n=1224, d=600 |
| delayed2h/never-seen bytes | B2 follows | NE: 59.1 [49.5, 68.3]; n=88, d=39 | 51.4 [43.3, 59.1]; n=138, d=71 | 54.1 [50.5, 58.0]; n=700, d=353 | 54.4 [50.8, 58.1]; n=688, d=351 | 54.2 [51.1, 57.5]; n=926, d=451 |
| delayed2h/never-seen bytes | B2 unchanged | NE: 10.2 [4.3, 17.3]; n=88, d=39 | 21.7 [15.1, 29.7]; n=138, d=71 | 17.0 [14.0, 20.0]; n=700, d=353 | 16.7 [13.8, 19.7]; n=688, d=351 | 17.1 [14.5, 19.9]; n=926, d=451 |
| delayed2h/never-seen bytes | B2 other | NE: 30.7 [22.2, 38.9]; n=88, d=39 | 26.8 [19.3, 34.0]; n=138, d=71 | 28.9 [25.6, 32.4]; n=700, d=353 | 28.9 [25.8, 32.4]; n=688, d=351 | 28.7 [25.9, 31.6]; n=926, d=451 |
| delayed2h/never-seen bytes | B3 follows | 98.3 [95.8, 100.0]; n=117, d=55 | 95.7 [92.5, 98.4]; n=188, d=97 | 97.5 [96.4, 98.5]; n=919, d=463 | 97.5 [96.4, 98.4]; n=904, d=463 | 97.3 [96.3, 98.2]; n=1224, d=600 |
| delayed2h/never-seen bytes | B3 unchanged | 0.0 [0.0, 0.0]; n=117, d=55 | 0.0 [0.0, 0.0]; n=188, d=97 | 0.0 [0.0, 0.0]; n=919, d=463 | 0.0 [0.0, 0.0]; n=904, d=463 | 0.0 [0.0, 0.0]; n=1224, d=600 |
| delayed2h/never-seen bytes | B3 other | 1.7 [0.0, 4.2]; n=117, d=55 | 4.3 [1.6, 7.5]; n=188, d=97 | 2.5 [1.5, 3.6]; n=919, d=463 | 2.5 [1.6, 3.6]; n=904, d=463 | 2.7 [1.8, 3.7]; n=1224, d=600 |

## reader32-s0-15k

Far lies beyond the 63-byte reader field.

Checkpoint: `/home/joe/hsn-experiments/vast2/exp/gpu3/runs/a14_reader/hold_k12nw_rd32_ag_ng2_cutC_cells64_18k_gsum-s0-80a58acbb9a8/exp21/step_00015000.pt`. Status: **complete**.

| Task / keys | Condition | Near | Mid | Far | Far-strict | All |
| --- | --- | --- | --- | --- | --- | --- |
| kv/training keys | N0 | 99.7 [99.3, 100.0]; n=1062, d=536 | 99.1 [98.5, 99.6]; n=998, d=479 | 98.0 [97.2, 98.8]; n=1264, d=485 | 98.2 [97.4, 99.0]; n=1082, d=475 | 98.9 [98.5, 99.2]; n=3324, d=600 |
| kv/training keys | N1 | 6.7 [5.3, 8.3]; n=1062, d=536 | 5.4 [4.1, 6.8]; n=998, d=479 | 5.5 [4.3, 6.7]; n=1264, d=485 | 4.1 [3.0, 5.2]; n=1082, d=475 | 5.8 [5.1, 6.6]; n=3324, d=600 |
| kv/training keys | N2a | 1.2 [0.6, 1.9]; n=1062, d=536 | 2.2 [1.4, 3.1]; n=998, d=479 | 9.3 [7.7, 10.8]; n=1264, d=485 | 10.3 [8.6, 12.0]; n=1082, d=475 | 4.6 [3.9, 5.3]; n=3324, d=600 |
| kv/training keys | N2b-answer | 6.7 [5.3, 8.1]; n=1062, d=536 | 5.7 [4.3, 7.2]; n=998, d=479 | 6.0 [4.9, 7.2]; n=1264, d=485 | 5.3 [4.1, 6.4]; n=1082, d=475 | 6.1 [5.3, 6.9]; n=3324, d=600 |
| kv/training keys | N2b-all | 6.7 [5.3, 8.1]; n=1062, d=536 | 5.7 [4.3, 7.2]; n=998, d=479 | 6.0 [4.9, 7.2]; n=1264, d=485 | 5.3 [4.1, 6.4]; n=1082, d=475 | 6.1 [5.3, 6.9]; n=3324, d=600 |
| kv/training keys | N3 | 6.7 [5.3, 8.3]; n=1062, d=536 | 5.4 [4.1, 6.8]; n=998, d=479 | 5.5 [4.3, 6.7]; n=1264, d=485 | 4.1 [3.0, 5.2]; n=1082, d=475 | 5.8 [5.1, 6.6]; n=3324, d=600 |
| kv/training keys | N4 | 37.9 [34.4, 41.1]; n=1062, d=536 | 38.8 [35.4, 42.1]; n=998, d=479 | 48.9 [46.0, 51.7]; n=1264, d=485 | 49.7 [46.6, 52.7]; n=1082, d=475 | 42.4 [40.3, 44.5]; n=3324, d=600 |
| kv/training keys | B1 follows | 99.9 [99.6, 100.0]; n=891, d=488 | 99.6 [99.2, 100.0]; n=846, d=448 | 99.2 [98.7, 99.7]; n=1030, d=457 | 99.1 [98.5, 99.7]; n=903, d=443 | 99.6 [99.3, 99.8]; n=2767, d=588 |
| kv/training keys | B1 unchanged | 0.0 [0.0, 0.0]; n=891, d=488 | 0.0 [0.0, 0.0]; n=846, d=448 | 0.0 [0.0, 0.0]; n=1030, d=457 | 0.0 [0.0, 0.0]; n=903, d=443 | 0.0 [0.0, 0.0]; n=2767, d=588 |
| kv/training keys | B1 other | 0.1 [0.0, 0.4]; n=891, d=488 | 0.4 [0.0, 0.8]; n=846, d=448 | 0.8 [0.3, 1.3]; n=1030, d=457 | 0.9 [0.3, 1.5]; n=903, d=443 | 0.4 [0.2, 0.7]; n=2767, d=588 |
| kv/dev keys | N0 | 99.6 [99.2, 99.9]; n=1071, d=531 | 99.4 [98.9, 99.8]; n=1011, d=476 | 98.3 [97.5, 98.9]; n=1263, d=477 | 98.4 [97.7, 99.1]; n=1095, d=468 | 99.0 [98.7, 99.3]; n=3345, d=600 |
| kv/dev keys | N1 | 14.0 [11.9, 16.3]; n=1071, d=531 | 12.7 [10.7, 14.6]; n=1011, d=476 | 7.2 [5.8, 8.7]; n=1263, d=477 | 4.4 [3.3, 5.6]; n=1095, d=468 | 11.0 [10.0, 12.2]; n=3345, d=600 |
| kv/dev keys | N2a | 1.4 [0.7, 2.2]; n=1071, d=531 | 2.3 [1.4, 3.3]; n=1011, d=476 | 7.8 [6.5, 9.1]; n=1263, d=477 | 8.9 [7.3, 10.5]; n=1095, d=468 | 4.1 [3.5, 4.7]; n=3345, d=600 |
| kv/dev keys | N2b-answer | 11.9 [9.9, 13.9]; n=1071, d=531 | 11.2 [9.3, 13.1]; n=1011, d=476 | 6.7 [5.3, 8.2]; n=1263, d=477 | 5.2 [3.9, 6.6]; n=1095, d=468 | 9.7 [8.8, 10.7]; n=3345, d=600 |
| kv/dev keys | N2b-all | 11.9 [9.9, 13.9]; n=1071, d=531 | 11.2 [9.3, 13.1]; n=1011, d=476 | 6.7 [5.3, 8.2]; n=1263, d=477 | 5.2 [3.9, 6.6]; n=1095, d=468 | 9.7 [8.8, 10.7]; n=3345, d=600 |
| kv/dev keys | N3 | 14.0 [11.9, 16.3]; n=1071, d=531 | 12.7 [10.7, 14.6]; n=1011, d=476 | 7.2 [5.8, 8.7]; n=1263, d=477 | 4.4 [3.3, 5.6]; n=1095, d=468 | 11.0 [10.0, 12.2]; n=3345, d=600 |
| kv/dev keys | N4 | 37.6 [34.2, 40.7]; n=1071, d=531 | 43.9 [40.3, 47.5]; n=1011, d=476 | 51.5 [48.3, 54.5]; n=1263, d=477 | 53.3 [50.1, 56.4]; n=1095, d=468 | 44.8 [42.6, 46.9]; n=3345, d=600 |
| kv/dev keys | B1 follows | 100.0 [100.0, 100.0]; n=891, d=490 | 99.8 [99.4, 100.0]; n=849, d=441 | 99.6 [99.2, 99.9]; n=1058, d=454 | 99.7 [99.2, 100.0]; n=944, d=437 | 99.8 [99.6, 99.9]; n=2798, d=595 |
| kv/dev keys | B1 unchanged | 0.0 [0.0, 0.0]; n=891, d=490 | 0.0 [0.0, 0.0]; n=849, d=441 | 0.1 [0.0, 0.3]; n=1058, d=454 | 0.0 [0.0, 0.0]; n=944, d=437 | 0.0 [0.0, 0.1]; n=2798, d=595 |
| kv/dev keys | B1 other | 0.0 [0.0, 0.0]; n=891, d=490 | 0.2 [0.0, 0.6]; n=849, d=441 | 0.3 [0.0, 0.7]; n=1058, d=454 | 0.3 [0.0, 0.8]; n=944, d=437 | 0.2 [0.0, 0.4]; n=2798, d=595 |
| kv/test keys | N0 | 99.9 [99.7, 100.0]; n=1025, d=528 | 99.2 [98.7, 99.7]; n=1041, d=480 | 98.5 [97.8, 99.2]; n=1298, d=487 | 98.8 [98.1, 99.3]; n=1129, d=477 | 99.2 [98.8, 99.5]; n=3364, d=600 |
| kv/test keys | N1 | 12.0 [9.9, 14.2]; n=1025, d=528 | 13.4 [11.4, 15.6]; n=1041, d=480 | 8.2 [6.7, 9.7]; n=1298, d=487 | 4.9 [3.8, 6.1]; n=1129, d=477 | 11.0 [9.9, 12.1]; n=3364, d=600 |
| kv/test keys | N2a | 2.1 [1.3, 3.2]; n=1025, d=528 | 2.8 [1.8, 3.9]; n=1041, d=480 | 9.2 [7.6, 10.7]; n=1298, d=487 | 10.5 [8.6, 12.1]; n=1129, d=477 | 5.1 [4.3, 5.8]; n=3364, d=600 |
| kv/test keys | N2b-answer | 10.7 [8.9, 12.7]; n=1025, d=528 | 11.2 [9.4, 13.2]; n=1041, d=480 | 8.5 [7.1, 9.9]; n=1298, d=487 | 6.1 [4.8, 7.4]; n=1129, d=477 | 10.0 [9.1, 11.0]; n=3364, d=600 |
| kv/test keys | N2b-all | 10.7 [8.9, 12.7]; n=1025, d=528 | 11.2 [9.4, 13.2]; n=1041, d=480 | 8.5 [7.1, 9.9]; n=1298, d=487 | 6.1 [4.8, 7.4]; n=1129, d=477 | 10.0 [9.1, 11.0]; n=3364, d=600 |
| kv/test keys | N3 | 12.0 [9.9, 14.2]; n=1025, d=528 | 13.4 [11.4, 15.6]; n=1041, d=480 | 8.2 [6.7, 9.7]; n=1298, d=487 | 4.9 [3.8, 6.1]; n=1129, d=477 | 11.0 [9.9, 12.1]; n=3364, d=600 |
| kv/test keys | N4 | 38.1 [34.8, 41.6]; n=1025, d=528 | 39.9 [36.7, 43.5]; n=1041, d=480 | 51.1 [48.3, 54.0]; n=1298, d=487 | 52.3 [49.2, 55.3]; n=1129, d=477 | 43.7 [41.7, 45.7]; n=3364, d=600 |
| kv/test keys | B1 follows | 99.9 [99.6, 100.0]; n=865, d=483 | 98.8 [98.1, 99.5]; n=849, d=443 | 99.2 [98.6, 99.7]; n=1059, d=464 | 99.3 [98.6, 99.8]; n=954, d=450 | 99.3 [99.0, 99.6]; n=2773, d=598 |
| kv/test keys | B1 unchanged | 0.0 [0.0, 0.0]; n=865, d=483 | 0.0 [0.0, 0.0]; n=849, d=443 | 0.0 [0.0, 0.0]; n=1059, d=464 | 0.0 [0.0, 0.0]; n=954, d=450 | 0.0 [0.0, 0.0]; n=2773, d=598 |
| kv/test keys | B1 other | 0.1 [0.0, 0.4]; n=865, d=483 | 1.2 [0.5, 1.9]; n=849, d=443 | 0.8 [0.3, 1.4]; n=1059, d=464 | 0.7 [0.2, 1.4]; n=954, d=450 | 0.7 [0.4, 1.0]; n=2773, d=598 |
| delayed2h/seen pairs | N0 | 100.0 [100.0, 100.0]; n=103, d=54 | 100.0 [100.0, 100.0]; n=175, d=92 | 99.5 [99.0, 99.9]; n=977, d=471 | 99.5 [99.0, 99.9]; n=964, d=471 | 99.6 [99.2, 99.9]; n=1255, d=600 |
| delayed2h/seen pairs | N1 | 1.0 [0.0, 3.7]; n=103, d=54 | 1.1 [0.0, 2.8]; n=175, d=92 | 1.2 [0.6, 1.9]; n=977, d=471 | 1.2 [0.6, 2.0]; n=964, d=471 | 1.2 [0.6, 1.8]; n=1255, d=600 |
| delayed2h/seen pairs | N2a | 0.0 [0.0, 0.0]; n=103, d=54 | 0.6 [0.0, 1.9]; n=175, d=92 | 0.4 [0.1, 0.8]; n=977, d=471 | 0.4 [0.1, 0.8]; n=964, d=471 | 0.4 [0.1, 0.8]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-answer | 0.0 [0.0, 0.0]; n=103, d=54 | 1.7 [0.0, 3.7]; n=175, d=92 | 1.6 [0.9, 2.4]; n=977, d=471 | 1.7 [0.9, 2.5]; n=964, d=471 | 1.5 [0.9, 2.2]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-all | 0.0 [0.0, 0.0]; n=103, d=54 | 1.7 [0.0, 3.7]; n=175, d=92 | 1.6 [0.9, 2.4]; n=977, d=471 | 1.7 [0.9, 2.5]; n=964, d=471 | 1.5 [0.9, 2.2]; n=1255, d=600 |
| delayed2h/seen pairs | N3 | 1.0 [0.0, 3.7]; n=103, d=54 | 1.1 [0.0, 2.8]; n=175, d=92 | 1.2 [0.6, 1.9]; n=977, d=471 | 1.2 [0.6, 2.0]; n=964, d=471 | 1.2 [0.6, 1.8]; n=1255, d=600 |
| delayed2h/seen pairs | N4 | 83.5 [77.1, 89.2]; n=103, d=54 | 87.4 [82.5, 92.4]; n=175, d=92 | 88.0 [86.1, 90.0]; n=977, d=471 | 88.1 [86.1, 90.0]; n=964, d=471 | 87.6 [85.8, 89.3]; n=1255, d=600 |
| delayed2h/seen pairs | B1 follows | 100.0 [100.0, 100.0]; n=103, d=54 | 100.0 [100.0, 100.0]; n=175, d=92 | 99.6 [99.2, 99.9]; n=972, d=471 | 99.6 [99.1, 99.9]; n=959, d=471 | 99.7 [99.4, 99.9]; n=1250, d=600 |
| delayed2h/seen pairs | B1 unchanged | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=972, d=471 | 0.0 [0.0, 0.0]; n=959, d=471 | 0.0 [0.0, 0.0]; n=1250, d=600 |
| delayed2h/seen pairs | B1 other | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.4 [0.1, 0.8]; n=972, d=471 | 0.4 [0.1, 0.9]; n=959, d=471 | 0.3 [0.1, 0.6]; n=1250, d=600 |
| delayed2h/seen pairs | B2 follows | NE: 89.0 [81.0, 96.3]; n=73, d=37 | 75.0 [68.3, 82.2]; n=136, d=69 | 76.8 [73.9, 79.8]; n=721, d=343 | 76.8 [73.8, 79.7]; n=710, d=342 | 77.5 [75.0, 80.2]; n=930, d=433 |
| delayed2h/seen pairs | B2 unchanged | NE: 2.7 [0.0, 7.2]; n=73, d=37 | 8.1 [3.6, 12.8]; n=136, d=69 | 8.2 [6.4, 10.2]; n=721, d=343 | 8.2 [6.3, 10.2]; n=710, d=342 | 7.7 [6.0, 9.6]; n=930, d=433 |
| delayed2h/seen pairs | B2 other | NE: 8.2 [1.8, 16.0]; n=73, d=37 | 16.9 [10.9, 23.1]; n=136, d=69 | 15.0 [12.5, 17.6]; n=721, d=343 | 15.1 [12.6, 17.8]; n=710, d=342 | 14.7 [12.5, 16.9]; n=930, d=433 |
| delayed2h/seen pairs | B3 follows | 100.0 [100.0, 100.0]; n=103, d=54 | 100.0 [100.0, 100.0]; n=175, d=92 | 99.6 [99.2, 99.9]; n=972, d=471 | 99.6 [99.1, 99.9]; n=959, d=471 | 99.7 [99.4, 99.9]; n=1250, d=600 |
| delayed2h/seen pairs | B3 unchanged | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=972, d=471 | 0.0 [0.0, 0.0]; n=959, d=471 | 0.0 [0.0, 0.0]; n=1250, d=600 |
| delayed2h/seen pairs | B3 other | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.4 [0.1, 0.8]; n=972, d=471 | 0.4 [0.1, 0.9]; n=959, d=471 | 0.3 [0.1, 0.6]; n=1250, d=600 |
| delayed2h/new pairs | N0 | NE: 98.0 [94.7, 100.0]; n=99, d=52 | 100.0 [100.0, 100.0]; n=176, d=91 | 99.6 [99.1, 99.9]; n=979, d=471 | 99.6 [99.1, 99.9]; n=964, d=470 | 99.5 [99.1, 99.8]; n=1254, d=600 |
| delayed2h/new pairs | N1 | NE: 3.0 [0.0, 6.7]; n=99, d=52 | 0.6 [0.0, 1.8]; n=176, d=91 | 1.7 [1.0, 2.6]; n=979, d=471 | 1.8 [1.0, 2.7]; n=964, d=470 | 1.7 [1.0, 2.4]; n=1254, d=600 |
| delayed2h/new pairs | N2a | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.6 [0.0, 1.8]; n=176, d=91 | 0.5 [0.1, 1.0]; n=979, d=471 | 0.5 [0.1, 1.1]; n=964, d=470 | 0.5 [0.2, 0.9]; n=1254, d=600 |
| delayed2h/new pairs | N2b-answer | NE: 3.0 [0.0, 7.1]; n=99, d=52 | 1.1 [0.0, 2.9]; n=176, d=91 | 1.4 [0.7, 2.2]; n=979, d=471 | 1.5 [0.7, 2.3]; n=964, d=470 | 1.5 [0.9, 2.2]; n=1254, d=600 |
| delayed2h/new pairs | N2b-all | NE: 3.0 [0.0, 7.1]; n=99, d=52 | 1.1 [0.0, 2.9]; n=176, d=91 | 1.4 [0.7, 2.2]; n=979, d=471 | 1.5 [0.7, 2.3]; n=964, d=470 | 1.5 [0.9, 2.2]; n=1254, d=600 |
| delayed2h/new pairs | N3 | NE: 3.0 [0.0, 6.7]; n=99, d=52 | 0.6 [0.0, 1.8]; n=176, d=91 | 1.7 [1.0, 2.6]; n=979, d=471 | 1.8 [1.0, 2.7]; n=964, d=470 | 1.7 [1.0, 2.4]; n=1254, d=600 |
| delayed2h/new pairs | N4 | NE: 89.9 [83.7, 95.7]; n=99, d=52 | 90.3 [86.7, 94.3]; n=176, d=91 | 86.7 [84.8, 88.7]; n=979, d=471 | 86.6 [84.7, 88.6]; n=964, d=470 | 87.5 [85.7, 89.2]; n=1254, d=600 |
| delayed2h/new pairs | B1 follows | NE: 100.0 [100.0, 100.0]; n=97, d=52 | 100.0 [100.0, 100.0]; n=176, d=91 | 99.8 [99.5, 100.0]; n=975, d=471 | 99.8 [99.5, 100.0]; n=960, d=470 | 99.8 [99.6, 100.0]; n=1248, d=600 |
| delayed2h/new pairs | B1 unchanged | NE: 0.0 [0.0, 0.0]; n=97, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=975, d=471 | 0.0 [0.0, 0.0]; n=960, d=470 | 0.0 [0.0, 0.0]; n=1248, d=600 |
| delayed2h/new pairs | B1 other | NE: 0.0 [0.0, 0.0]; n=97, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.2 [0.0, 0.5]; n=975, d=471 | 0.2 [0.0, 0.5]; n=960, d=470 | 0.2 [0.0, 0.4]; n=1248, d=600 |
| delayed2h/new pairs | B2 follows | NE: 87.7 [77.9, 95.0]; n=65, d=34 | 80.7 [74.8, 86.6]; n=140, d=70 | 77.4 [74.5, 80.3]; n=753, d=358 | 77.5 [74.6, 80.4]; n=739, d=355 | 78.6 [76.0, 81.1]; n=958, d=449 |
| delayed2h/new pairs | B2 unchanged | NE: 6.2 [1.4, 13.7]; n=65, d=34 | 9.3 [5.2, 14.3]; n=140, d=70 | 9.7 [7.8, 11.8]; n=753, d=358 | 9.5 [7.6, 11.5]; n=739, d=355 | 9.4 [7.8, 11.2]; n=958, d=449 |
| delayed2h/new pairs | B2 other | NE: 6.2 [1.4, 13.3]; n=65, d=34 | 10.0 [5.2, 14.8]; n=140, d=70 | 12.9 [10.5, 15.2]; n=753, d=358 | 13.0 [10.6, 15.3]; n=739, d=355 | 12.0 [10.0, 13.9]; n=958, d=449 |
| delayed2h/new pairs | B3 follows | NE: 100.0 [100.0, 100.0]; n=97, d=52 | 100.0 [100.0, 100.0]; n=176, d=91 | 99.8 [99.5, 100.0]; n=975, d=471 | 99.8 [99.5, 100.0]; n=960, d=470 | 99.8 [99.6, 100.0]; n=1248, d=600 |
| delayed2h/new pairs | B3 unchanged | NE: 0.0 [0.0, 0.0]; n=97, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=975, d=471 | 0.0 [0.0, 0.0]; n=960, d=470 | 0.0 [0.0, 0.0]; n=1248, d=600 |
| delayed2h/new pairs | B3 other | NE: 0.0 [0.0, 0.0]; n=97, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.2 [0.0, 0.5]; n=975, d=471 | 0.2 [0.0, 0.5]; n=960, d=470 | 0.2 [0.0, 0.4]; n=1248, d=600 |
| delayed2h/never-seen bytes | N0 | 99.2 [97.3, 100.0]; n=119, d=55 | 99.0 [97.3, 100.0]; n=192, d=97 | 99.4 [98.8, 99.8]; n=953, d=463 | 99.4 [98.8, 99.8]; n=936, d=463 | 99.3 [98.8, 99.7]; n=1264, d=600 |
| delayed2h/never-seen bytes | N1 | 0.0 [0.0, 0.0]; n=119, d=55 | 0.5 [0.0, 1.6]; n=192, d=97 | 1.4 [0.6, 2.1]; n=953, d=463 | 1.4 [0.7, 2.2]; n=936, d=463 | 1.1 [0.6, 1.7]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2a | 0.8 [0.0, 2.7]; n=119, d=55 | 1.0 [0.0, 2.5]; n=192, d=97 | 0.5 [0.1, 1.0]; n=953, d=463 | 0.5 [0.1, 1.0]; n=936, d=463 | 0.6 [0.2, 1.1]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-answer | 0.0 [0.0, 0.0]; n=119, d=55 | 1.6 [0.0, 3.4]; n=192, d=97 | 1.9 [1.1, 2.8]; n=953, d=463 | 1.9 [1.1, 2.8]; n=936, d=463 | 1.7 [1.0, 2.4]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-all | 0.0 [0.0, 0.0]; n=119, d=55 | 1.6 [0.0, 3.4]; n=192, d=97 | 1.9 [1.1, 2.8]; n=953, d=463 | 1.9 [1.1, 2.8]; n=936, d=463 | 1.7 [1.0, 2.4]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3 | 0.0 [0.0, 0.0]; n=119, d=55 | 0.5 [0.0, 1.6]; n=192, d=97 | 1.4 [0.6, 2.1]; n=953, d=463 | 1.4 [0.7, 2.2]; n=936, d=463 | 1.1 [0.6, 1.7]; n=1264, d=600 |
| delayed2h/never-seen bytes | N4 | 84.9 [79.4, 90.7]; n=119, d=55 | 88.0 [83.7, 92.3]; n=192, d=97 | 85.8 [83.6, 87.9]; n=953, d=463 | 85.9 [83.6, 88.0]; n=936, d=463 | 86.1 [84.3, 87.9]; n=1264, d=600 |
| delayed2h/never-seen bytes | B1 follows | 100.0 [100.0, 100.0]; n=118, d=55 | 99.5 [98.3, 100.0]; n=190, d=97 | 99.6 [99.1, 99.9]; n=947, d=463 | 99.6 [99.1, 99.9]; n=930, d=463 | 99.6 [99.2, 99.9]; n=1255, d=600 |
| delayed2h/never-seen bytes | B1 unchanged | 0.0 [0.0, 0.0]; n=118, d=55 | 0.0 [0.0, 0.0]; n=190, d=97 | 0.0 [0.0, 0.0]; n=947, d=463 | 0.0 [0.0, 0.0]; n=930, d=463 | 0.0 [0.0, 0.0]; n=1255, d=600 |
| delayed2h/never-seen bytes | B1 other | 0.0 [0.0, 0.0]; n=118, d=55 | 0.5 [0.0, 1.7]; n=190, d=97 | 0.4 [0.1, 0.9]; n=947, d=463 | 0.4 [0.1, 0.9]; n=930, d=463 | 0.4 [0.1, 0.8]; n=1255, d=600 |
| delayed2h/never-seen bytes | B2 follows | NE: 78.7 [71.0, 85.7]; n=89, d=39 | 77.1 [69.8, 83.6]; n=140, d=71 | 76.4 [73.4, 79.5]; n=728, d=353 | 76.2 [73.1, 79.4]; n=714, d=351 | 76.7 [74.2, 79.2]; n=957, d=451 |
| delayed2h/never-seen bytes | B2 unchanged | NE: 7.9 [2.9, 14.3]; n=89, d=39 | 10.0 [5.5, 15.2]; n=140, d=71 | 10.6 [8.3, 12.8]; n=728, d=353 | 10.6 [8.3, 12.9]; n=714, d=351 | 10.2 [8.4, 12.1]; n=957, d=451 |
| delayed2h/never-seen bytes | B2 other | NE: 13.5 [7.0, 20.5]; n=89, d=39 | 12.9 [8.0, 18.7]; n=140, d=71 | 13.0 [10.7, 15.5]; n=728, d=353 | 13.2 [10.8, 15.6]; n=714, d=351 | 13.1 [10.9, 15.1]; n=957, d=451 |
| delayed2h/never-seen bytes | B3 follows | 100.0 [100.0, 100.0]; n=118, d=55 | 99.5 [98.3, 100.0]; n=190, d=97 | 99.6 [99.1, 99.9]; n=947, d=463 | 99.6 [99.1, 99.9]; n=930, d=463 | 99.6 [99.2, 99.9]; n=1255, d=600 |
| delayed2h/never-seen bytes | B3 unchanged | 0.0 [0.0, 0.0]; n=118, d=55 | 0.0 [0.0, 0.0]; n=190, d=97 | 0.0 [0.0, 0.0]; n=947, d=463 | 0.0 [0.0, 0.0]; n=930, d=463 | 0.0 [0.0, 0.0]; n=1255, d=600 |
| delayed2h/never-seen bytes | B3 other | 0.0 [0.0, 0.0]; n=118, d=55 | 0.5 [0.0, 1.7]; n=190, d=97 | 0.4 [0.1, 0.9]; n=947, d=463 | 0.4 [0.1, 0.9]; n=930, d=463 | 0.4 [0.1, 0.8]; n=1255, d=600 |

## reader32-s0

Far lies beyond the 63-byte reader field.

Checkpoint: `/home/joe/hsn-experiments/vast2/exp/gpu3/runs/a14_reader/hold_k12nw_rd32_ag_ng2_cutC_cells64_18k_gsum-s0-80a58acbb9a8/exp21/step_00018000.pt`. Status: **complete**.

| Task / keys | Condition | Near | Mid | Far | Far-strict | All |
| --- | --- | --- | --- | --- | --- | --- |
| kv/training keys | N0 | 96.4 [95.3, 97.5]; n=1062, d=536 | 95.5 [94.1, 96.7]; n=998, d=479 | 93.9 [92.4, 95.2]; n=1264, d=485 | 94.6 [93.2, 96.0]; n=1082, d=475 | 95.2 [94.4, 96.0]; n=3324, d=600 |
| kv/training keys | N1 | 15.4 [13.3, 17.5]; n=1062, d=536 | 13.8 [11.7, 16.2]; n=998, d=479 | 11.9 [10.3, 13.6]; n=1264, d=485 | 9.2 [7.7, 10.7]; n=1082, d=475 | 13.6 [12.5, 14.8]; n=3324, d=600 |
| kv/training keys | N2a | 14.0 [12.0, 16.1]; n=1062, d=536 | 13.8 [11.7, 16.2]; n=998, d=479 | 17.0 [15.1, 19.0]; n=1264, d=485 | 16.0 [14.2, 17.8]; n=1082, d=475 | 15.1 [13.9, 16.4]; n=3324, d=600 |
| kv/training keys | N2b-answer | 15.4 [13.3, 17.6]; n=1062, d=536 | 13.7 [11.7, 16.1]; n=998, d=479 | 11.0 [9.3, 12.6]; n=1264, d=485 | 7.9 [6.4, 9.4]; n=1082, d=475 | 13.2 [12.1, 14.4]; n=3324, d=600 |
| kv/training keys | N2b-all | 15.4 [13.3, 17.6]; n=1062, d=536 | 13.7 [11.7, 16.1]; n=998, d=479 | 11.0 [9.3, 12.6]; n=1264, d=485 | 7.9 [6.4, 9.4]; n=1082, d=475 | 13.2 [12.1, 14.4]; n=3324, d=600 |
| kv/training keys | N3 | 15.4 [13.3, 17.5]; n=1062, d=536 | 13.8 [11.7, 16.2]; n=998, d=479 | 11.9 [10.3, 13.6]; n=1264, d=485 | 9.2 [7.7, 10.7]; n=1082, d=475 | 13.6 [12.5, 14.8]; n=3324, d=600 |
| kv/training keys | N4 | 30.0 [26.7, 33.6]; n=1062, d=536 | 29.9 [26.5, 33.0]; n=998, d=479 | 41.0 [38.2, 43.8]; n=1264, d=485 | 42.2 [39.3, 45.2]; n=1082, d=475 | 34.1 [32.3, 36.1]; n=3324, d=600 |
| kv/training keys | B1 follows | 98.4 [97.5, 99.2]; n=875, d=485 | 97.5 [96.4, 98.4]; n=828, d=444 | 97.0 [95.8, 98.0]; n=993, d=455 | 97.4 [96.0, 98.4]; n=868, d=438 | 97.6 [97.0, 98.1]; n=2696, d=587 |
| kv/training keys | B1 unchanged | 0.1 [0.0, 0.4]; n=875, d=485 | 0.1 [0.0, 0.4]; n=828, d=444 | 0.0 [0.0, 0.0]; n=993, d=455 | 0.0 [0.0, 0.0]; n=868, d=438 | 0.1 [0.0, 0.2]; n=2696, d=587 |
| kv/training keys | B1 other | 1.5 [0.7, 2.3]; n=875, d=485 | 2.4 [1.4, 3.4]; n=828, d=444 | 3.0 [2.0, 4.2]; n=993, d=455 | 2.6 [1.6, 4.0]; n=868, d=438 | 2.3 [1.8, 2.9]; n=2696, d=587 |
| kv/dev keys | N0 | 97.6 [96.6, 98.4]; n=1071, d=531 | 94.7 [93.2, 96.1]; n=1011, d=476 | 91.2 [89.6, 92.7]; n=1263, d=477 | 91.3 [89.7, 92.9]; n=1095, d=468 | 94.3 [93.4, 95.1]; n=3345, d=600 |
| kv/dev keys | N1 | 16.3 [14.0, 18.6]; n=1071, d=531 | 14.5 [12.4, 16.7]; n=1011, d=476 | 11.0 [9.1, 12.7]; n=1263, d=477 | 7.9 [6.4, 9.6]; n=1095, d=468 | 13.8 [12.6, 15.0]; n=3345, d=600 |
| kv/dev keys | N2a | 16.2 [14.0, 18.4]; n=1071, d=531 | 14.9 [12.8, 17.3]; n=1011, d=476 | 15.5 [13.5, 17.3]; n=1263, d=477 | 13.4 [11.6, 15.1]; n=1095, d=468 | 15.6 [14.3, 16.9]; n=3345, d=600 |
| kv/dev keys | N2b-answer | 16.2 [14.0, 18.5]; n=1071, d=531 | 14.3 [12.3, 16.6]; n=1011, d=476 | 10.1 [8.4, 11.9]; n=1263, d=477 | 6.8 [5.3, 8.3]; n=1095, d=468 | 13.4 [12.2, 14.6]; n=3345, d=600 |
| kv/dev keys | N2b-all | 16.2 [14.0, 18.5]; n=1071, d=531 | 14.3 [12.3, 16.6]; n=1011, d=476 | 10.1 [8.4, 11.9]; n=1263, d=477 | 6.8 [5.3, 8.3]; n=1095, d=468 | 13.4 [12.2, 14.6]; n=3345, d=600 |
| kv/dev keys | N3 | 16.3 [14.0, 18.6]; n=1071, d=531 | 14.5 [12.4, 16.7]; n=1011, d=476 | 11.0 [9.1, 12.7]; n=1263, d=477 | 7.9 [6.4, 9.6]; n=1095, d=468 | 13.8 [12.6, 15.0]; n=3345, d=600 |
| kv/dev keys | N4 | 33.1 [29.9, 36.2]; n=1071, d=531 | 36.3 [32.7, 39.7]; n=1011, d=476 | 48.0 [45.0, 51.1]; n=1263, d=477 | 49.1 [46.1, 52.3]; n=1095, d=468 | 39.7 [37.5, 41.6]; n=3345, d=600 |
| kv/dev keys | B1 follows | 98.3 [97.3, 99.2]; n=879, d=487 | 97.0 [95.8, 98.1]; n=813, d=428 | 96.1 [94.7, 97.3]; n=989, d=450 | 96.0 [94.6, 97.3]; n=878, d=432 | 97.1 [96.4, 97.7]; n=2681, d=594 |
| kv/dev keys | B1 unchanged | 0.0 [0.0, 0.0]; n=879, d=487 | 0.1 [0.0, 0.4]; n=813, d=428 | 0.0 [0.0, 0.0]; n=989, d=450 | 0.0 [0.0, 0.0]; n=878, d=432 | 0.0 [0.0, 0.1]; n=2681, d=594 |
| kv/dev keys | B1 other | 1.7 [0.8, 2.7]; n=879, d=487 | 2.8 [1.8, 4.0]; n=813, d=428 | 3.9 [2.7, 5.3]; n=989, d=450 | 4.0 [2.7, 5.4]; n=878, d=432 | 2.9 [2.2, 3.5]; n=2681, d=594 |
| kv/test keys | N0 | 98.2 [97.3, 98.9]; n=1025, d=528 | 96.6 [95.3, 97.7]; n=1041, d=480 | 93.1 [91.8, 94.4]; n=1298, d=487 | 93.0 [91.6, 94.5]; n=1129, d=477 | 95.7 [95.0, 96.4]; n=3364, d=600 |
| kv/test keys | N1 | 14.9 [12.7, 17.2]; n=1025, d=528 | 17.2 [15.0, 19.5]; n=1041, d=480 | 12.7 [10.9, 14.6]; n=1298, d=487 | 9.2 [7.7, 10.8]; n=1129, d=477 | 14.8 [13.5, 16.0]; n=3364, d=600 |
| kv/test keys | N2a | 14.7 [12.6, 16.9]; n=1025, d=528 | 17.3 [15.0, 19.6]; n=1041, d=480 | 17.3 [15.3, 19.5]; n=1298, d=487 | 14.9 [13.0, 16.8]; n=1129, d=477 | 16.5 [15.2, 17.8]; n=3364, d=600 |
| kv/test keys | N2b-answer | 14.7 [12.5, 17.0]; n=1025, d=528 | 16.9 [14.7, 19.3]; n=1041, d=480 | 11.4 [9.7, 13.2]; n=1298, d=487 | 7.7 [6.3, 9.2]; n=1129, d=477 | 14.1 [12.9, 15.4]; n=3364, d=600 |
| kv/test keys | N2b-all | 14.7 [12.5, 17.0]; n=1025, d=528 | 16.9 [14.7, 19.3]; n=1041, d=480 | 11.4 [9.7, 13.2]; n=1298, d=487 | 7.7 [6.3, 9.2]; n=1129, d=477 | 14.1 [12.9, 15.4]; n=3364, d=600 |
| kv/test keys | N3 | 14.9 [12.7, 17.2]; n=1025, d=528 | 17.2 [15.0, 19.5]; n=1041, d=480 | 12.7 [10.9, 14.6]; n=1298, d=487 | 9.2 [7.7, 10.8]; n=1129, d=477 | 14.8 [13.5, 16.0]; n=3364, d=600 |
| kv/test keys | N4 | 33.8 [30.4, 37.1]; n=1025, d=528 | 34.4 [31.2, 37.8]; n=1041, d=480 | 46.7 [43.9, 49.5]; n=1298, d=487 | 48.2 [45.2, 51.3]; n=1129, d=477 | 38.9 [37.1, 41.0]; n=3364, d=600 |
| kv/test keys | B1 follows | 99.1 [98.3, 99.7]; n=858, d=483 | 98.1 [97.1, 98.9]; n=833, d=440 | 97.0 [96.0, 98.0]; n=998, d=460 | 97.1 [96.0, 98.1]; n=895, d=446 | 98.0 [97.5, 98.5]; n=2689, d=598 |
| kv/test keys | B1 unchanged | 0.0 [0.0, 0.0]; n=858, d=483 | 0.0 [0.0, 0.0]; n=833, d=440 | 0.0 [0.0, 0.0]; n=998, d=460 | 0.0 [0.0, 0.0]; n=895, d=446 | 0.0 [0.0, 0.0]; n=2689, d=598 |
| kv/test keys | B1 other | 0.9 [0.3, 1.7]; n=858, d=483 | 1.9 [1.1, 2.9]; n=833, d=440 | 3.0 [2.0, 4.0]; n=998, d=460 | 2.9 [1.9, 4.0]; n=895, d=446 | 2.0 [1.5, 2.5]; n=2689, d=598 |
| delayed2h/seen pairs | N0 | 99.0 [96.9, 100.0]; n=103, d=54 | 97.7 [95.4, 99.5]; n=175, d=92 | 98.6 [97.9, 99.3]; n=977, d=471 | 98.5 [97.8, 99.3]; n=964, d=471 | 98.5 [97.8, 99.1]; n=1255, d=600 |
| delayed2h/seen pairs | N1 | 0.0 [0.0, 0.0]; n=103, d=54 | 0.6 [0.0, 1.9]; n=175, d=92 | 0.8 [0.3, 1.5]; n=977, d=471 | 0.8 [0.3, 1.5]; n=964, d=471 | 0.7 [0.3, 1.2]; n=1255, d=600 |
| delayed2h/seen pairs | N2a | 1.9 [0.0, 5.6]; n=103, d=54 | 1.7 [0.0, 3.8]; n=175, d=92 | 1.9 [1.1, 2.8]; n=977, d=471 | 2.0 [1.1, 2.9]; n=964, d=471 | 1.9 [1.2, 2.7]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-answer | 0.0 [0.0, 0.0]; n=103, d=54 | 1.1 [0.0, 2.9]; n=175, d=92 | 0.6 [0.2, 1.1]; n=977, d=471 | 0.6 [0.2, 1.1]; n=964, d=471 | 0.6 [0.2, 1.1]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-all | 0.0 [0.0, 0.0]; n=103, d=54 | 1.1 [0.0, 2.9]; n=175, d=92 | 0.6 [0.2, 1.1]; n=977, d=471 | 0.6 [0.2, 1.1]; n=964, d=471 | 0.6 [0.2, 1.1]; n=1255, d=600 |
| delayed2h/seen pairs | N3 | 0.0 [0.0, 0.0]; n=103, d=54 | 0.6 [0.0, 1.9]; n=175, d=92 | 0.8 [0.3, 1.5]; n=977, d=471 | 0.8 [0.3, 1.5]; n=964, d=471 | 0.7 [0.3, 1.2]; n=1255, d=600 |
| delayed2h/seen pairs | N4 | 68.9 [59.5, 77.6]; n=103, d=54 | 70.3 [63.7, 75.8]; n=175, d=92 | 76.6 [74.1, 79.1]; n=977, d=471 | 76.3 [73.8, 78.9]; n=964, d=471 | 75.1 [72.9, 77.3]; n=1255, d=600 |
| delayed2h/seen pairs | B1 follows | 100.0 [100.0, 100.0]; n=102, d=54 | 100.0 [100.0, 100.0]; n=171, d=92 | 99.6 [99.2, 99.9]; n=963, d=471 | 99.6 [99.2, 99.9]; n=950, d=471 | 99.7 [99.4, 99.9]; n=1236, d=600 |
| delayed2h/seen pairs | B1 unchanged | 0.0 [0.0, 0.0]; n=102, d=54 | 0.0 [0.0, 0.0]; n=171, d=92 | 0.0 [0.0, 0.0]; n=963, d=471 | 0.0 [0.0, 0.0]; n=950, d=471 | 0.0 [0.0, 0.0]; n=1236, d=600 |
| delayed2h/seen pairs | B1 other | 0.0 [0.0, 0.0]; n=102, d=54 | 0.0 [0.0, 0.0]; n=171, d=92 | 0.4 [0.1, 0.8]; n=963, d=471 | 0.4 [0.1, 0.8]; n=950, d=471 | 0.3 [0.1, 0.6]; n=1236, d=600 |
| delayed2h/seen pairs | B2 follows | NE: 58.3 [47.7, 69.3]; n=72, d=37 | 59.4 [50.8, 67.7]; n=133, d=69 | 60.4 [56.5, 64.1]; n=713, d=343 | 60.3 [56.4, 63.9]; n=702, d=342 | 60.1 [56.8, 63.4]; n=918, d=433 |
| delayed2h/seen pairs | B2 unchanged | NE: 11.1 [4.8, 19.1]; n=72, d=37 | 9.8 [4.8, 15.2]; n=133, d=69 | 13.7 [11.1, 16.6]; n=713, d=343 | 14.0 [11.3, 16.8]; n=702, d=342 | 13.0 [10.7, 15.3]; n=918, d=433 |
| delayed2h/seen pairs | B2 other | NE: 30.6 [19.7, 41.9]; n=72, d=37 | 30.8 [23.0, 38.8]; n=133, d=69 | 25.8 [22.6, 29.2]; n=713, d=343 | 25.8 [22.5, 29.0]; n=702, d=342 | 26.9 [24.0, 29.7]; n=918, d=433 |
| delayed2h/seen pairs | B3 follows | 100.0 [100.0, 100.0]; n=102, d=54 | 100.0 [100.0, 100.0]; n=171, d=92 | 99.6 [99.2, 99.9]; n=963, d=471 | 99.6 [99.2, 99.9]; n=950, d=471 | 99.7 [99.4, 99.9]; n=1236, d=600 |
| delayed2h/seen pairs | B3 unchanged | 0.0 [0.0, 0.0]; n=102, d=54 | 0.0 [0.0, 0.0]; n=171, d=92 | 0.0 [0.0, 0.0]; n=963, d=471 | 0.0 [0.0, 0.0]; n=950, d=471 | 0.0 [0.0, 0.0]; n=1236, d=600 |
| delayed2h/seen pairs | B3 other | 0.0 [0.0, 0.0]; n=102, d=54 | 0.0 [0.0, 0.0]; n=171, d=92 | 0.4 [0.1, 0.8]; n=963, d=471 | 0.4 [0.1, 0.8]; n=950, d=471 | 0.3 [0.1, 0.6]; n=1236, d=600 |
| delayed2h/new pairs | N0 | NE: 98.0 [94.9, 100.0]; n=99, d=52 | 98.3 [96.2, 100.0]; n=176, d=91 | 99.0 [98.2, 99.6]; n=979, d=471 | 99.0 [98.2, 99.6]; n=964, d=470 | 98.8 [98.1, 99.4]; n=1254, d=600 |
| delayed2h/new pairs | N1 | NE: 2.0 [0.0, 5.3]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.7 [0.3, 1.3]; n=979, d=471 | 0.7 [0.3, 1.3]; n=964, d=470 | 0.7 [0.3, 1.2]; n=1254, d=600 |
| delayed2h/new pairs | N2a | NE: 1.0 [0.0, 3.7]; n=99, d=52 | 0.6 [0.0, 1.9]; n=176, d=91 | 2.2 [1.4, 3.2]; n=979, d=471 | 2.3 [1.4, 3.2]; n=964, d=470 | 1.9 [1.2, 2.7]; n=1254, d=600 |
| delayed2h/new pairs | N2b-answer | NE: 3.0 [0.0, 6.9]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.8 [0.3, 1.4]; n=979, d=471 | 0.8 [0.3, 1.4]; n=964, d=470 | 0.9 [0.4, 1.4]; n=1254, d=600 |
| delayed2h/new pairs | N2b-all | NE: 3.0 [0.0, 6.9]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.8 [0.3, 1.4]; n=979, d=471 | 0.8 [0.3, 1.4]; n=964, d=470 | 0.9 [0.4, 1.4]; n=1254, d=600 |
| delayed2h/new pairs | N3 | NE: 2.0 [0.0, 5.3]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.7 [0.3, 1.3]; n=979, d=471 | 0.7 [0.3, 1.3]; n=964, d=470 | 0.7 [0.3, 1.2]; n=1254, d=600 |
| delayed2h/new pairs | N4 | NE: 69.7 [60.9, 78.2]; n=99, d=52 | 71.6 [64.8, 78.3]; n=176, d=91 | 72.0 [69.4, 74.8]; n=979, d=471 | 71.8 [69.1, 74.5]; n=964, d=470 | 71.8 [69.5, 74.2]; n=1254, d=600 |
| delayed2h/new pairs | B1 follows | NE: 99.0 [96.6, 100.0]; n=97, d=52 | 98.8 [96.9, 100.0]; n=173, d=91 | 99.0 [98.3, 99.6]; n=969, d=471 | 99.0 [98.3, 99.6]; n=954, d=470 | 99.0 [98.4, 99.4]; n=1239, d=600 |
| delayed2h/new pairs | B1 unchanged | NE: 0.0 [0.0, 0.0]; n=97, d=52 | 0.0 [0.0, 0.0]; n=173, d=91 | 0.0 [0.0, 0.0]; n=969, d=471 | 0.0 [0.0, 0.0]; n=954, d=470 | 0.0 [0.0, 0.0]; n=1239, d=600 |
| delayed2h/new pairs | B1 other | NE: 1.0 [0.0, 3.4]; n=97, d=52 | 1.2 [0.0, 3.1]; n=173, d=91 | 1.0 [0.4, 1.7]; n=969, d=471 | 1.0 [0.4, 1.7]; n=954, d=470 | 1.0 [0.6, 1.6]; n=1239, d=600 |
| delayed2h/new pairs | B2 follows | NE: 63.1 [50.0, 75.3]; n=65, d=33 | 57.2 [49.2, 64.7]; n=138, d=69 | 58.2 [54.6, 61.7]; n=747, d=358 | 57.8 [54.2, 61.2]; n=733, d=355 | 58.4 [55.1, 61.4]; n=950, d=447 |
| delayed2h/new pairs | B2 unchanged | NE: 13.8 [6.0, 23.6]; n=65, d=33 | 15.2 [9.9, 21.3]; n=138, d=69 | 15.7 [13.1, 18.4]; n=747, d=358 | 15.6 [13.0, 18.2]; n=733, d=355 | 15.5 [13.2, 17.8]; n=950, d=447 |
| delayed2h/new pairs | B2 other | NE: 23.1 [12.2, 34.1]; n=65, d=33 | 27.5 [20.4, 34.6]; n=138, d=69 | 26.1 [23.2, 29.0]; n=747, d=358 | 26.6 [23.6, 29.6]; n=733, d=355 | 26.1 [23.5, 28.8]; n=950, d=447 |
| delayed2h/new pairs | B3 follows | NE: 99.0 [96.6, 100.0]; n=97, d=52 | 98.8 [96.9, 100.0]; n=173, d=91 | 99.0 [98.3, 99.6]; n=969, d=471 | 99.0 [98.3, 99.6]; n=954, d=470 | 99.0 [98.4, 99.4]; n=1239, d=600 |
| delayed2h/new pairs | B3 unchanged | NE: 0.0 [0.0, 0.0]; n=97, d=52 | 0.0 [0.0, 0.0]; n=173, d=91 | 0.0 [0.0, 0.0]; n=969, d=471 | 0.0 [0.0, 0.0]; n=954, d=470 | 0.0 [0.0, 0.0]; n=1239, d=600 |
| delayed2h/new pairs | B3 other | NE: 1.0 [0.0, 3.4]; n=97, d=52 | 1.2 [0.0, 3.1]; n=173, d=91 | 1.0 [0.4, 1.7]; n=969, d=471 | 1.0 [0.4, 1.7]; n=954, d=470 | 1.0 [0.6, 1.6]; n=1239, d=600 |
| delayed2h/never-seen bytes | N0 | 100.0 [100.0, 100.0]; n=119, d=55 | 99.0 [97.3, 100.0]; n=192, d=97 | 98.7 [98.0, 99.4]; n=953, d=463 | 98.8 [98.1, 99.5]; n=936, d=463 | 98.9 [98.3, 99.4]; n=1264, d=600 |
| delayed2h/never-seen bytes | N1 | 0.0 [0.0, 0.0]; n=119, d=55 | 0.5 [0.0, 1.8]; n=192, d=97 | 1.2 [0.5, 1.9]; n=953, d=463 | 1.2 [0.5, 2.0]; n=936, d=463 | 0.9 [0.5, 1.6]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2a | 1.7 [0.0, 4.3]; n=119, d=55 | 2.1 [0.5, 4.3]; n=192, d=97 | 2.6 [1.7, 3.7]; n=953, d=463 | 2.7 [1.7, 3.7]; n=936, d=463 | 2.5 [1.7, 3.3]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-answer | 0.0 [0.0, 0.0]; n=119, d=55 | 0.5 [0.0, 1.6]; n=192, d=97 | 1.4 [0.6, 2.1]; n=953, d=463 | 1.4 [0.7, 2.2]; n=936, d=463 | 1.1 [0.6, 1.7]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-all | 0.0 [0.0, 0.0]; n=119, d=55 | 0.5 [0.0, 1.6]; n=192, d=97 | 1.4 [0.6, 2.1]; n=953, d=463 | 1.4 [0.7, 2.2]; n=936, d=463 | 1.1 [0.6, 1.7]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3 | 0.0 [0.0, 0.0]; n=119, d=55 | 0.5 [0.0, 1.8]; n=192, d=97 | 1.2 [0.5, 1.9]; n=953, d=463 | 1.2 [0.5, 2.0]; n=936, d=463 | 0.9 [0.5, 1.6]; n=1264, d=600 |
| delayed2h/never-seen bytes | N4 | 74.8 [67.0, 82.2]; n=119, d=55 | 72.9 [66.3, 78.9]; n=192, d=97 | 77.4 [74.9, 80.0]; n=953, d=463 | 77.5 [74.8, 80.0]; n=936, d=463 | 76.5 [74.2, 78.8]; n=1264, d=600 |
| delayed2h/never-seen bytes | B1 follows | 98.3 [95.7, 100.0]; n=119, d=55 | 100.0 [100.0, 100.0]; n=190, d=97 | 99.5 [98.9, 99.9]; n=941, d=463 | 99.6 [99.1, 99.9]; n=925, d=463 | 99.4 [99.0, 99.8]; n=1250, d=600 |
| delayed2h/never-seen bytes | B1 unchanged | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=190, d=97 | 0.0 [0.0, 0.0]; n=941, d=463 | 0.0 [0.0, 0.0]; n=925, d=463 | 0.0 [0.0, 0.0]; n=1250, d=600 |
| delayed2h/never-seen bytes | B1 other | 1.7 [0.0, 4.3]; n=119, d=55 | 0.0 [0.0, 0.0]; n=190, d=97 | 0.5 [0.1, 1.1]; n=941, d=463 | 0.4 [0.1, 0.9]; n=925, d=463 | 0.6 [0.2, 1.0]; n=1250, d=600 |
| delayed2h/never-seen bytes | B2 follows | NE: 70.0 [61.0, 78.3]; n=90, d=39 | 66.4 [57.9, 74.8]; n=140, d=71 | 62.1 [58.1, 65.9]; n=723, d=353 | 62.4 [58.4, 66.3]; n=710, d=351 | 63.5 [60.2, 66.5]; n=953, d=451 |
| delayed2h/never-seen bytes | B2 unchanged | NE: 7.8 [2.9, 13.2]; n=90, d=39 | 13.6 [8.8, 18.8]; n=140, d=71 | 12.0 [9.4, 14.4]; n=723, d=353 | 11.7 [9.1, 14.1]; n=710, d=351 | 11.9 [9.7, 14.0]; n=953, d=451 |
| delayed2h/never-seen bytes | B2 other | NE: 22.2 [14.7, 29.4]; n=90, d=39 | 20.0 [13.7, 26.7]; n=140, d=71 | 25.9 [22.6, 29.5]; n=723, d=353 | 25.9 [22.6, 29.5]; n=710, d=351 | 24.7 [22.0, 27.6]; n=953, d=451 |
| delayed2h/never-seen bytes | B3 follows | 98.3 [95.7, 100.0]; n=119, d=55 | 100.0 [100.0, 100.0]; n=190, d=97 | 99.5 [98.9, 99.9]; n=941, d=463 | 99.6 [99.1, 99.9]; n=925, d=463 | 99.4 [99.0, 99.8]; n=1250, d=600 |
| delayed2h/never-seen bytes | B3 unchanged | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=190, d=97 | 0.0 [0.0, 0.0]; n=941, d=463 | 0.0 [0.0, 0.0]; n=925, d=463 | 0.0 [0.0, 0.0]; n=1250, d=600 |
| delayed2h/never-seen bytes | B3 other | 1.7 [0.0, 4.3]; n=119, d=55 | 0.0 [0.0, 0.0]; n=190, d=97 | 0.5 [0.1, 1.1]; n=941, d=463 | 0.4 [0.1, 0.9]; n=925, d=463 | 0.6 [0.2, 1.0]; n=1250, d=600 |

## reader32-s2

Far lies beyond the 63-byte reader field.

Checkpoint: `/home/joe/hsn-experiments/vast2/exp/gpu2/runs/a14_reader/hold_k12nw_rd32_ag_ng2_cutC_cells64_18k_gsum-s2-b70deebc05c0/exp16/step_00018000.pt`. Status: **complete**.

| Task / keys | Condition | Near | Mid | Far | Far-strict | All |
| --- | --- | --- | --- | --- | --- | --- |
| kv/training keys | N0 | 99.4 [98.9, 99.8]; n=1062, d=536 | 99.1 [98.4, 99.6]; n=998, d=479 | 98.5 [97.7, 99.1]; n=1264, d=485 | 98.6 [97.8, 99.3]; n=1082, d=475 | 99.0 [98.6, 99.3]; n=3324, d=600 |
| kv/training keys | N1 | 15.3 [13.0, 17.2]; n=1062, d=536 | 14.0 [12.0, 16.3]; n=998, d=479 | 16.7 [14.9, 18.6]; n=1264, d=485 | 15.0 [13.3, 16.7]; n=1082, d=475 | 15.4 [14.2, 16.6]; n=3324, d=600 |
| kv/training keys | N2a | 1.9 [1.1, 2.8]; n=1062, d=536 | 3.7 [2.6, 4.9]; n=998, d=479 | 19.6 [17.8, 21.4]; n=1264, d=485 | 21.3 [19.2, 23.3]; n=1082, d=475 | 9.2 [8.4, 9.9]; n=3324, d=600 |
| kv/training keys | N2b-answer | 15.5 [13.4, 17.5]; n=1062, d=536 | 14.2 [12.2, 16.5]; n=998, d=479 | 17.2 [15.4, 19.1]; n=1264, d=485 | 15.6 [13.9, 17.4]; n=1082, d=475 | 15.8 [14.6, 16.9]; n=3324, d=600 |
| kv/training keys | N2b-all | 15.5 [13.4, 17.5]; n=1062, d=536 | 14.2 [12.2, 16.5]; n=998, d=479 | 17.2 [15.4, 19.1]; n=1264, d=485 | 15.6 [13.9, 17.4]; n=1082, d=475 | 15.8 [14.6, 16.9]; n=3324, d=600 |
| kv/training keys | N3 | 15.3 [13.0, 17.2]; n=1062, d=536 | 14.0 [12.0, 16.3]; n=998, d=479 | 16.7 [14.9, 18.6]; n=1264, d=485 | 15.0 [13.3, 16.7]; n=1082, d=475 | 15.4 [14.2, 16.6]; n=3324, d=600 |
| kv/training keys | N4 | 14.7 [12.6, 16.8]; n=1062, d=536 | 14.0 [11.9, 16.4]; n=998, d=479 | 16.9 [15.1, 18.7]; n=1264, d=485 | 15.1 [13.3, 16.8]; n=1082, d=475 | 15.3 [14.1, 16.5]; n=3324, d=600 |
| kv/training keys | B1 follows | 99.7 [99.3, 100.0]; n=888, d=486 | 99.2 [98.4, 99.8]; n=844, d=449 | 98.9 [98.3, 99.5]; n=1033, d=458 | 98.8 [98.0, 99.4]; n=904, d=444 | 99.2 [98.9, 99.6]; n=2765, d=589 |
| kv/training keys | B1 unchanged | 0.0 [0.0, 0.0]; n=888, d=486 | 0.0 [0.0, 0.0]; n=844, d=449 | 0.0 [0.0, 0.0]; n=1033, d=458 | 0.0 [0.0, 0.0]; n=904, d=444 | 0.0 [0.0, 0.0]; n=2765, d=589 |
| kv/training keys | B1 other | 0.3 [0.0, 0.7]; n=888, d=486 | 0.8 [0.2, 1.6]; n=844, d=449 | 1.1 [0.5, 1.7]; n=1033, d=458 | 1.2 [0.6, 2.0]; n=904, d=444 | 0.8 [0.4, 1.1]; n=2765, d=589 |
| kv/dev keys | N0 | 95.1 [93.7, 96.3]; n=1071, d=531 | 96.1 [94.9, 97.3]; n=1011, d=476 | 91.1 [89.7, 92.7]; n=1263, d=477 | 91.3 [89.7, 93.0]; n=1095, d=468 | 93.9 [93.0, 94.7]; n=3345, d=600 |
| kv/dev keys | N1 | 5.8 [4.4, 7.3]; n=1071, d=531 | 6.6 [5.0, 8.2]; n=1011, d=476 | 8.8 [7.2, 10.4]; n=1263, d=477 | 8.7 [7.0, 10.3]; n=1095, d=468 | 7.2 [6.3, 8.1]; n=3345, d=600 |
| kv/dev keys | N2a | 1.3 [0.7, 2.1]; n=1071, d=531 | 3.3 [2.2, 4.3]; n=1011, d=476 | 14.5 [12.8, 16.3]; n=1263, d=477 | 15.6 [13.6, 17.7]; n=1095, d=468 | 6.9 [6.2, 7.6]; n=3345, d=600 |
| kv/dev keys | N2b-answer | 10.6 [8.6, 12.5]; n=1071, d=531 | 10.0 [8.2, 12.0]; n=1011, d=476 | 10.6 [8.8, 12.3]; n=1263, d=477 | 9.4 [7.8, 11.1]; n=1095, d=468 | 10.4 [9.4, 11.5]; n=3345, d=600 |
| kv/dev keys | N2b-all | 10.6 [8.6, 12.5]; n=1071, d=531 | 10.0 [8.2, 12.0]; n=1011, d=476 | 10.6 [8.8, 12.3]; n=1263, d=477 | 9.4 [7.8, 11.1]; n=1095, d=468 | 10.4 [9.4, 11.5]; n=3345, d=600 |
| kv/dev keys | N3 | 5.8 [4.4, 7.3]; n=1071, d=531 | 6.6 [5.0, 8.2]; n=1011, d=476 | 8.8 [7.2, 10.4]; n=1263, d=477 | 8.7 [7.0, 10.3]; n=1095, d=468 | 7.2 [6.3, 8.1]; n=3345, d=600 |
| kv/dev keys | N4 | 14.6 [12.5, 16.7]; n=1071, d=531 | 13.9 [11.8, 16.2]; n=1011, d=476 | 14.1 [12.1, 16.0]; n=1263, d=477 | 11.7 [9.9, 13.5]; n=1095, d=468 | 14.2 [12.9, 15.5]; n=3345, d=600 |
| kv/dev keys | B1 follows | 99.3 [98.7, 99.8]; n=863, d=480 | 99.0 [98.3, 99.6]; n=834, d=436 | 96.4 [95.2, 97.5]; n=999, d=439 | 96.4 [95.2, 97.6]; n=891, d=426 | 98.1 [97.6, 98.6]; n=2696, d=591 |
| kv/dev keys | B1 unchanged | 0.0 [0.0, 0.0]; n=863, d=480 | 0.2 [0.0, 0.7]; n=834, d=436 | 0.0 [0.0, 0.0]; n=999, d=439 | 0.0 [0.0, 0.0]; n=891, d=426 | 0.1 [0.0, 0.2]; n=2696, d=591 |
| kv/dev keys | B1 other | 0.7 [0.2, 1.3]; n=863, d=480 | 0.7 [0.2, 1.4]; n=834, d=436 | 3.6 [2.5, 4.8]; n=999, d=439 | 3.6 [2.4, 4.8]; n=891, d=426 | 1.8 [1.3, 2.3]; n=2696, d=591 |
| kv/test keys | N0 | 98.0 [97.2, 98.9]; n=1025, d=528 | 97.4 [96.4, 98.3]; n=1041, d=480 | 95.4 [94.2, 96.5]; n=1298, d=487 | 95.2 [94.0, 96.4]; n=1129, d=477 | 96.8 [96.3, 97.4]; n=3364, d=600 |
| kv/test keys | N1 | 8.3 [6.7, 9.8]; n=1025, d=528 | 9.2 [7.5, 11.2]; n=1041, d=480 | 11.5 [9.7, 13.1]; n=1298, d=487 | 10.4 [8.7, 12.1]; n=1129, d=477 | 9.8 [8.8, 10.8]; n=3364, d=600 |
| kv/test keys | N2a | 1.9 [1.1, 2.7]; n=1025, d=528 | 2.8 [1.8, 3.8]; n=1041, d=480 | 16.5 [14.9, 18.4]; n=1298, d=487 | 18.2 [16.3, 20.2]; n=1129, d=477 | 7.8 [7.1, 8.5]; n=3364, d=600 |
| kv/test keys | N2b-answer | 12.2 [10.4, 14.1]; n=1025, d=528 | 12.9 [10.9, 15.0]; n=1041, d=480 | 14.4 [12.5, 16.3]; n=1298, d=487 | 13.1 [11.3, 15.0]; n=1129, d=477 | 13.3 [12.1, 14.3]; n=3364, d=600 |
| kv/test keys | N2b-all | 12.2 [10.4, 14.1]; n=1025, d=528 | 12.9 [10.9, 15.0]; n=1041, d=480 | 14.4 [12.5, 16.3]; n=1298, d=487 | 13.1 [11.3, 15.0]; n=1129, d=477 | 13.3 [12.1, 14.3]; n=3364, d=600 |
| kv/test keys | N3 | 8.3 [6.7, 9.8]; n=1025, d=528 | 9.2 [7.5, 11.2]; n=1041, d=480 | 11.5 [9.7, 13.1]; n=1298, d=487 | 10.4 [8.7, 12.1]; n=1129, d=477 | 9.8 [8.8, 10.8]; n=3364, d=600 |
| kv/test keys | N4 | 13.2 [11.2, 15.3]; n=1025, d=528 | 16.0 [13.8, 18.2]; n=1041, d=480 | 16.4 [14.4, 18.6]; n=1298, d=487 | 13.6 [11.7, 15.4]; n=1129, d=477 | 15.3 [14.0, 16.6]; n=3364, d=600 |
| kv/test keys | B1 follows | 99.7 [99.2, 100.0]; n=862, d=483 | 99.0 [98.3, 99.6]; n=842, d=441 | 97.4 [96.4, 98.3]; n=1028, d=458 | 97.4 [96.4, 98.3]; n=924, d=444 | 98.6 [98.2, 99.0]; n=2732, d=598 |
| kv/test keys | B1 unchanged | 0.0 [0.0, 0.0]; n=862, d=483 | 0.0 [0.0, 0.0]; n=842, d=441 | 0.0 [0.0, 0.0]; n=1028, d=458 | 0.0 [0.0, 0.0]; n=924, d=444 | 0.0 [0.0, 0.0]; n=2732, d=598 |
| kv/test keys | B1 other | 0.3 [0.0, 0.8]; n=862, d=483 | 1.0 [0.4, 1.7]; n=842, d=441 | 2.6 [1.7, 3.6]; n=1028, d=458 | 2.6 [1.7, 3.6]; n=924, d=444 | 1.4 [1.0, 1.8]; n=2732, d=598 |
| delayed2h/seen pairs | N0 | 100.0 [100.0, 100.0]; n=103, d=54 | 100.0 [100.0, 100.0]; n=175, d=92 | 99.5 [99.0, 99.9]; n=977, d=471 | 99.5 [99.0, 99.9]; n=964, d=471 | 99.6 [99.3, 99.9]; n=1255, d=600 |
| delayed2h/seen pairs | N1 | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.1 [0.0, 0.3]; n=977, d=471 | 0.1 [0.0, 0.3]; n=964, d=471 | 0.1 [0.0, 0.2]; n=1255, d=600 |
| delayed2h/seen pairs | N2a | 28.2 [20.8, 36.7]; n=103, d=54 | 31.4 [25.6, 37.7]; n=175, d=92 | 29.5 [27.1, 32.1]; n=977, d=471 | 29.9 [27.5, 32.5]; n=964, d=471 | 29.6 [27.7, 31.8]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-answer | 1.0 [0.0, 3.1]; n=103, d=54 | 0.6 [0.0, 1.9]; n=175, d=92 | 1.8 [1.0, 2.7]; n=977, d=471 | 1.8 [1.0, 2.7]; n=964, d=471 | 1.6 [0.9, 2.4]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-all | 1.0 [0.0, 3.1]; n=103, d=54 | 0.6 [0.0, 1.9]; n=175, d=92 | 1.8 [1.0, 2.7]; n=977, d=471 | 1.8 [1.0, 2.7]; n=964, d=471 | 1.6 [0.9, 2.4]; n=1255, d=600 |
| delayed2h/seen pairs | N3 | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.1 [0.0, 0.3]; n=977, d=471 | 0.1 [0.0, 0.3]; n=964, d=471 | 0.1 [0.0, 0.2]; n=1255, d=600 |
| delayed2h/seen pairs | N4 | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=977, d=471 | 0.0 [0.0, 0.0]; n=964, d=471 | 0.0 [0.0, 0.0]; n=1255, d=600 |
| delayed2h/seen pairs | B1 follows | 100.0 [100.0, 100.0]; n=103, d=54 | 99.4 [98.2, 100.0]; n=175, d=92 | 99.2 [98.6, 99.7]; n=972, d=471 | 99.2 [98.5, 99.7]; n=959, d=471 | 99.3 [98.8, 99.7]; n=1250, d=600 |
| delayed2h/seen pairs | B1 unchanged | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=972, d=471 | 0.0 [0.0, 0.0]; n=959, d=471 | 0.0 [0.0, 0.0]; n=1250, d=600 |
| delayed2h/seen pairs | B1 other | 0.0 [0.0, 0.0]; n=103, d=54 | 0.6 [0.0, 1.8]; n=175, d=92 | 0.8 [0.3, 1.4]; n=972, d=471 | 0.8 [0.3, 1.5]; n=959, d=471 | 0.7 [0.3, 1.2]; n=1250, d=600 |
| delayed2h/seen pairs | B2 follows | NE: 58.9 [47.6, 69.2]; n=73, d=37 | 46.3 [39.4, 53.4]; n=136, d=69 | 45.9 [42.8, 49.0]; n=721, d=343 | 45.6 [42.5, 48.8]; n=710, d=342 | 47.0 [44.3, 49.6]; n=930, d=433 |
| delayed2h/seen pairs | B2 unchanged | NE: 27.4 [18.8, 37.9]; n=73, d=37 | 37.5 [30.7, 45.2]; n=136, d=69 | 38.7 [36.0, 41.5]; n=721, d=343 | 39.2 [36.4, 42.0]; n=710, d=342 | 37.6 [35.3, 40.0]; n=930, d=433 |
| delayed2h/seen pairs | B2 other | NE: 13.7 [5.7, 22.1]; n=73, d=37 | 16.2 [10.6, 21.7]; n=136, d=69 | 15.4 [12.7, 18.3]; n=721, d=343 | 15.2 [12.4, 18.1]; n=710, d=342 | 15.4 [13.1, 17.7]; n=930, d=433 |
| delayed2h/seen pairs | B3 follows | 100.0 [100.0, 100.0]; n=103, d=54 | 99.4 [98.2, 100.0]; n=175, d=92 | 99.2 [98.6, 99.7]; n=972, d=471 | 99.2 [98.5, 99.7]; n=959, d=471 | 99.3 [98.8, 99.7]; n=1250, d=600 |
| delayed2h/seen pairs | B3 unchanged | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=972, d=471 | 0.0 [0.0, 0.0]; n=959, d=471 | 0.0 [0.0, 0.0]; n=1250, d=600 |
| delayed2h/seen pairs | B3 other | 0.0 [0.0, 0.0]; n=103, d=54 | 0.6 [0.0, 1.8]; n=175, d=92 | 0.8 [0.3, 1.4]; n=972, d=471 | 0.8 [0.3, 1.5]; n=959, d=471 | 0.7 [0.3, 1.2]; n=1250, d=600 |
| delayed2h/new pairs | N0 | NE: 100.0 [100.0, 100.0]; n=99, d=52 | 99.4 [98.3, 100.0]; n=176, d=91 | 99.1 [98.4, 99.6]; n=979, d=471 | 99.1 [98.4, 99.6]; n=964, d=470 | 99.2 [98.7, 99.7]; n=1254, d=600 |
| delayed2h/new pairs | N1 | NE: 1.0 [0.0, 3.2]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=979, d=471 | 0.0 [0.0, 0.0]; n=964, d=470 | 0.1 [0.0, 0.2]; n=1254, d=600 |
| delayed2h/new pairs | N2a | NE: 29.3 [21.6, 38.5]; n=99, d=52 | 32.4 [26.2, 39.7]; n=176, d=91 | 29.0 [26.6, 31.6]; n=979, d=471 | 29.5 [27.0, 32.1]; n=964, d=470 | 29.5 [27.4, 31.9]; n=1254, d=600 |
| delayed2h/new pairs | N2b-answer | NE: 3.0 [0.0, 6.9]; n=99, d=52 | 4.0 [1.4, 6.9]; n=176, d=91 | 1.5 [0.8, 2.4]; n=979, d=471 | 1.6 [0.8, 2.4]; n=964, d=470 | 2.0 [1.2, 2.8]; n=1254, d=600 |
| delayed2h/new pairs | N2b-all | NE: 3.0 [0.0, 6.9]; n=99, d=52 | 4.0 [1.4, 6.9]; n=176, d=91 | 1.5 [0.8, 2.4]; n=979, d=471 | 1.6 [0.8, 2.4]; n=964, d=470 | 2.0 [1.2, 2.8]; n=1254, d=600 |
| delayed2h/new pairs | N3 | NE: 1.0 [0.0, 3.2]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=979, d=471 | 0.0 [0.0, 0.0]; n=964, d=470 | 0.1 [0.0, 0.2]; n=1254, d=600 |
| delayed2h/new pairs | N4 | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=979, d=471 | 0.0 [0.0, 0.0]; n=964, d=470 | 0.0 [0.0, 0.0]; n=1254, d=600 |
| delayed2h/new pairs | B1 follows | NE: 100.0 [100.0, 100.0]; n=99, d=52 | 98.9 [97.1, 100.0]; n=175, d=91 | 99.3 [98.7, 99.7]; n=970, d=471 | 99.3 [98.7, 99.7]; n=955, d=470 | 99.3 [98.8, 99.7]; n=1244, d=600 |
| delayed2h/new pairs | B1 unchanged | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=175, d=91 | 0.0 [0.0, 0.0]; n=970, d=471 | 0.0 [0.0, 0.0]; n=955, d=470 | 0.0 [0.0, 0.0]; n=1244, d=600 |
| delayed2h/new pairs | B1 other | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 1.1 [0.0, 2.9]; n=175, d=91 | 0.7 [0.3, 1.3]; n=970, d=471 | 0.7 [0.3, 1.3]; n=955, d=470 | 0.7 [0.3, 1.2]; n=1244, d=600 |
| delayed2h/new pairs | B2 follows | NE: 53.7 [43.2, 62.3]; n=67, d=34 | 48.2 [40.7, 55.1]; n=139, d=70 | 46.2 [43.1, 49.3]; n=749, d=358 | 45.9 [42.8, 49.0]; n=735, d=355 | 47.0 [44.2, 49.5]; n=955, d=449 |
| delayed2h/new pairs | B2 unchanged | NE: 32.8 [25.6, 41.8]; n=67, d=34 | 37.4 [31.3, 44.1]; n=139, d=70 | 38.1 [35.3, 40.8]; n=749, d=358 | 38.1 [35.3, 40.9]; n=735, d=355 | 37.6 [35.2, 40.2]; n=955, d=449 |
| delayed2h/new pairs | B2 other | NE: 13.4 [5.3, 23.1]; n=67, d=34 | 14.4 [8.6, 21.0]; n=139, d=70 | 15.8 [13.2, 18.5]; n=749, d=358 | 16.1 [13.4, 18.8]; n=735, d=355 | 15.4 [13.1, 17.7]; n=955, d=449 |
| delayed2h/new pairs | B3 follows | NE: 100.0 [100.0, 100.0]; n=99, d=52 | 98.9 [97.1, 100.0]; n=175, d=91 | 99.3 [98.7, 99.7]; n=970, d=471 | 99.3 [98.7, 99.7]; n=955, d=470 | 99.3 [98.8, 99.7]; n=1244, d=600 |
| delayed2h/new pairs | B3 unchanged | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=175, d=91 | 0.0 [0.0, 0.0]; n=970, d=471 | 0.0 [0.0, 0.0]; n=955, d=470 | 0.0 [0.0, 0.0]; n=1244, d=600 |
| delayed2h/new pairs | B3 other | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 1.1 [0.0, 2.9]; n=175, d=91 | 0.7 [0.3, 1.3]; n=970, d=471 | 0.7 [0.3, 1.3]; n=955, d=470 | 0.7 [0.3, 1.2]; n=1244, d=600 |
| delayed2h/never-seen bytes | N0 | 96.6 [93.1, 99.3]; n=119, d=55 | 92.2 [86.9, 96.4]; n=192, d=97 | 86.5 [84.5, 88.2]; n=953, d=463 | 86.4 [84.5, 88.2]; n=936, d=463 | 88.3 [86.7, 89.9]; n=1264, d=600 |
| delayed2h/never-seen bytes | N1 | 0.8 [0.0, 2.7]; n=119, d=55 | 1.0 [0.0, 2.5]; n=192, d=97 | 0.2 [0.0, 0.5]; n=953, d=463 | 0.2 [0.0, 0.5]; n=936, d=463 | 0.4 [0.1, 0.8]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2a | 10.1 [5.2, 15.9]; n=119, d=55 | 14.6 [10.6, 19.0]; n=192, d=97 | 11.1 [9.3, 13.1]; n=953, d=463 | 11.3 [9.4, 13.4]; n=936, d=463 | 11.6 [9.9, 13.3]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-answer | 2.5 [0.0, 5.5]; n=119, d=55 | 3.1 [0.7, 5.6]; n=192, d=97 | 1.5 [0.7, 2.3]; n=953, d=463 | 1.5 [0.8, 2.3]; n=936, d=463 | 1.8 [1.1, 2.6]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-all | 2.5 [0.0, 5.5]; n=119, d=55 | 3.1 [0.7, 5.6]; n=192, d=97 | 1.5 [0.7, 2.3]; n=953, d=463 | 1.5 [0.8, 2.3]; n=936, d=463 | 1.8 [1.1, 2.6]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3 | 0.8 [0.0, 2.7]; n=119, d=55 | 1.0 [0.0, 2.5]; n=192, d=97 | 0.2 [0.0, 0.5]; n=953, d=463 | 0.2 [0.0, 0.5]; n=936, d=463 | 0.4 [0.1, 0.8]; n=1264, d=600 |
| delayed2h/never-seen bytes | N4 | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.0 [0.0, 0.0]; n=953, d=463 | 0.0 [0.0, 0.0]; n=936, d=463 | 0.0 [0.0, 0.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | B1 follows | 98.3 [94.3, 100.0]; n=115, d=52 | 98.3 [96.2, 100.0]; n=177, d=94 | 97.5 [96.3, 98.5]; n=824, d=436 | 97.4 [96.3, 98.5]; n=809, d=434 | 97.7 [96.7, 98.5]; n=1116, d=569 |
| delayed2h/never-seen bytes | B1 unchanged | 0.9 [0.0, 2.8]; n=115, d=52 | 0.0 [0.0, 0.0]; n=177, d=94 | 0.1 [0.0, 0.4]; n=824, d=436 | 0.1 [0.0, 0.4]; n=809, d=434 | 0.2 [0.0, 0.5]; n=1116, d=569 |
| delayed2h/never-seen bytes | B1 other | 0.9 [0.0, 2.8]; n=115, d=52 | 1.7 [0.0, 3.8]; n=177, d=94 | 2.4 [1.5, 3.5]; n=824, d=436 | 2.5 [1.5, 3.6]; n=809, d=434 | 2.2 [1.4, 3.0]; n=1116, d=569 |
| delayed2h/never-seen bytes | B2 follows | NE: 78.4 [68.8, 87.4]; n=88, d=37 | 59.5 [50.0, 67.7]; n=131, d=70 | 66.5 [63.2, 69.7]; n=630, d=336 | 66.7 [63.2, 69.9]; n=618, d=333 | 66.7 [63.5, 69.5]; n=849, d=432 |
| delayed2h/never-seen bytes | B2 unchanged | NE: 10.2 [4.7, 16.4]; n=88, d=37 | 22.1 [16.3, 28.8]; n=131, d=70 | 15.7 [13.0, 18.4]; n=630, d=336 | 15.7 [13.0, 18.4]; n=618, d=333 | 16.1 [13.8, 18.6]; n=849, d=432 |
| delayed2h/never-seen bytes | B2 other | NE: 11.4 [4.4, 20.0]; n=88, d=37 | 18.3 [10.3, 27.1]; n=131, d=70 | 17.8 [14.8, 20.5]; n=630, d=336 | 17.6 [14.7, 20.4]; n=618, d=333 | 17.2 [14.8, 19.6]; n=849, d=432 |
| delayed2h/never-seen bytes | B3 follows | 97.4 [92.9, 100.0]; n=115, d=52 | 98.3 [96.2, 100.0]; n=177, d=94 | 97.5 [96.3, 98.5]; n=824, d=436 | 97.4 [96.3, 98.5]; n=809, d=434 | 97.6 [96.6, 98.4]; n=1116, d=569 |
| delayed2h/never-seen bytes | B3 unchanged | 0.9 [0.0, 2.8]; n=115, d=52 | 0.0 [0.0, 0.0]; n=177, d=94 | 0.1 [0.0, 0.4]; n=824, d=436 | 0.1 [0.0, 0.4]; n=809, d=434 | 0.2 [0.0, 0.5]; n=1116, d=569 |
| delayed2h/never-seen bytes | B3 other | 1.7 [0.0, 4.6]; n=115, d=52 | 1.7 [0.0, 3.8]; n=177, d=94 | 2.4 [1.5, 3.5]; n=824, d=436 | 2.5 [1.5, 3.6]; n=809, d=434 | 2.2 [1.5, 3.1]; n=1116, d=569 |

## reader-full-s0

Far is only a distance; this reader sees the whole window.

Checkpoint: `/home/joe/hsn-experiments/vast2/exp/gpu1/runs/a14_reader/hold_k12nw_rd_ag_ng2_cutC_cells64_18k_gsum-s0-885ec24099ee/exp23/step_00018000.pt`. Status: **complete**.

| Task / keys | Condition | Near | Mid | Far | Far-strict | All |
| --- | --- | --- | --- | --- | --- | --- |
| kv/training keys | N0 | 98.9 [98.1, 99.4]; n=1062, d=536 | 98.1 [97.2, 98.9]; n=998, d=479 | 98.1 [97.3, 98.8]; n=1264, d=485 | 98.3 [97.5, 99.0]; n=1082, d=475 | 98.3 [97.9, 98.8]; n=3324, d=600 |
| kv/training keys | N1 | 16.4 [14.2, 18.5]; n=1062, d=536 | 16.6 [14.5, 19.1]; n=998, d=479 | 25.0 [22.9, 27.1]; n=1264, d=485 | 23.4 [21.3, 25.6]; n=1082, d=475 | 19.7 [18.5, 21.1]; n=3324, d=600 |
| kv/training keys | N2a | 12.5 [10.5, 14.4]; n=1062, d=536 | 13.7 [11.7, 15.7]; n=998, d=479 | 23.6 [21.5, 25.7]; n=1264, d=485 | 23.8 [21.7, 26.0]; n=1082, d=475 | 17.1 [16.0, 18.2]; n=3324, d=600 |
| kv/training keys | N2b-answer | 16.4 [14.2, 18.5]; n=1062, d=536 | 16.5 [14.4, 19.0]; n=998, d=479 | 24.7 [22.5, 26.8]; n=1264, d=485 | 23.3 [21.1, 25.5]; n=1082, d=475 | 19.6 [18.3, 20.9]; n=3324, d=600 |
| kv/training keys | N2b-all | 16.4 [14.2, 18.5]; n=1062, d=536 | 16.5 [14.4, 19.0]; n=998, d=479 | 24.7 [22.5, 26.8]; n=1264, d=485 | 23.3 [21.1, 25.5]; n=1082, d=475 | 19.6 [18.3, 20.9]; n=3324, d=600 |
| kv/training keys | N3 | 16.4 [14.2, 18.5]; n=1062, d=536 | 16.6 [14.5, 19.1]; n=998, d=479 | 25.0 [22.9, 27.1]; n=1264, d=485 | 23.4 [21.3, 25.6]; n=1082, d=475 | 19.7 [18.5, 21.1]; n=3324, d=600 |
| kv/training keys | N4 | 18.3 [16.0, 20.6]; n=1062, d=536 | 18.2 [16.0, 20.8]; n=998, d=479 | 20.9 [18.7, 23.2]; n=1264, d=485 | 19.2 [17.1, 21.4]; n=1082, d=475 | 19.3 [18.0, 20.6]; n=3324, d=600 |
| kv/training keys | B1 follows | 99.7 [99.2, 100.0]; n=885, d=485 | 99.0 [98.3, 99.6]; n=838, d=448 | 98.7 [98.0, 99.3]; n=1029, d=458 | 98.6 [97.7, 99.2]; n=900, d=444 | 99.1 [98.8, 99.4]; n=2752, d=589 |
| kv/training keys | B1 unchanged | 0.0 [0.0, 0.0]; n=885, d=485 | 0.2 [0.0, 0.6]; n=838, d=448 | 0.0 [0.0, 0.0]; n=1029, d=458 | 0.0 [0.0, 0.0]; n=900, d=444 | 0.1 [0.0, 0.2]; n=2752, d=589 |
| kv/training keys | B1 other | 0.3 [0.0, 0.8]; n=885, d=485 | 0.7 [0.2, 1.4]; n=838, d=448 | 1.3 [0.7, 2.0]; n=1029, d=458 | 1.4 [0.8, 2.3]; n=900, d=444 | 0.8 [0.5, 1.1]; n=2752, d=589 |
| kv/dev keys | N0 | 99.5 [99.1, 99.9]; n=1071, d=531 | 98.4 [97.5, 99.1]; n=1011, d=476 | 98.0 [97.3, 98.7]; n=1263, d=477 | 98.1 [97.3, 98.8]; n=1095, d=468 | 98.6 [98.2, 99.0]; n=3345, d=600 |
| kv/dev keys | N1 | 17.2 [14.8, 19.4]; n=1071, d=531 | 18.0 [15.7, 20.6]; n=1011, d=476 | 22.5 [20.1, 24.7]; n=1263, d=477 | 20.6 [18.4, 22.9]; n=1095, d=468 | 19.4 [18.1, 20.8]; n=3345, d=600 |
| kv/dev keys | N2a | 7.9 [6.4, 9.5]; n=1071, d=531 | 10.1 [8.3, 11.9]; n=1011, d=476 | 18.4 [16.5, 20.5]; n=1263, d=477 | 19.8 [17.7, 22.1]; n=1095, d=468 | 12.6 [11.6, 13.6]; n=3345, d=600 |
| kv/dev keys | N2b-answer | 17.2 [14.8, 19.4]; n=1071, d=531 | 17.7 [15.4, 20.2]; n=1011, d=476 | 21.9 [19.6, 24.1]; n=1263, d=477 | 20.0 [17.8, 22.2]; n=1095, d=468 | 19.1 [17.8, 20.5]; n=3345, d=600 |
| kv/dev keys | N2b-all | 17.2 [14.8, 19.4]; n=1071, d=531 | 17.7 [15.4, 20.2]; n=1011, d=476 | 21.9 [19.6, 24.1]; n=1263, d=477 | 20.0 [17.8, 22.2]; n=1095, d=468 | 19.1 [17.8, 20.5]; n=3345, d=600 |
| kv/dev keys | N3 | 17.2 [14.8, 19.4]; n=1071, d=531 | 18.0 [15.7, 20.6]; n=1011, d=476 | 22.5 [20.1, 24.7]; n=1263, d=477 | 20.6 [18.4, 22.9]; n=1095, d=468 | 19.4 [18.1, 20.8]; n=3345, d=600 |
| kv/dev keys | N4 | 18.7 [16.5, 21.0]; n=1071, d=531 | 20.5 [18.0, 22.9]; n=1011, d=476 | 19.9 [17.6, 22.1]; n=1263, d=477 | 17.6 [15.5, 19.6]; n=1095, d=468 | 19.7 [18.2, 21.0]; n=3345, d=600 |
| kv/dev keys | B1 follows | 99.9 [99.7, 100.0]; n=891, d=489 | 99.4 [98.8, 99.9]; n=838, d=438 | 98.6 [97.7, 99.3]; n=1058, d=454 | 98.6 [97.7, 99.3]; n=942, d=440 | 99.2 [98.9, 99.6]; n=2787, d=595 |
| kv/dev keys | B1 unchanged | 0.0 [0.0, 0.0]; n=891, d=489 | 0.0 [0.0, 0.0]; n=838, d=438 | 0.0 [0.0, 0.0]; n=1058, d=454 | 0.0 [0.0, 0.0]; n=942, d=440 | 0.0 [0.0, 0.0]; n=2787, d=595 |
| kv/dev keys | B1 other | 0.1 [0.0, 0.3]; n=891, d=489 | 0.6 [0.1, 1.2]; n=838, d=438 | 1.4 [0.7, 2.3]; n=1058, d=454 | 1.4 [0.7, 2.3]; n=942, d=440 | 0.8 [0.4, 1.1]; n=2787, d=595 |
| kv/test keys | N0 | 99.3 [98.8, 99.8]; n=1025, d=528 | 98.7 [97.9, 99.2]; n=1041, d=480 | 96.6 [95.7, 97.6]; n=1298, d=487 | 96.4 [95.3, 97.5]; n=1129, d=477 | 98.1 [97.6, 98.5]; n=3364, d=600 |
| kv/test keys | N1 | 16.1 [14.0, 18.4]; n=1025, d=528 | 19.1 [16.8, 21.6]; n=1041, d=480 | 24.7 [22.6, 27.0]; n=1298, d=487 | 21.7 [19.6, 23.8]; n=1129, d=477 | 20.4 [19.0, 21.7]; n=3364, d=600 |
| kv/test keys | N2a | 6.2 [4.9, 7.9]; n=1025, d=528 | 8.5 [6.7, 10.1]; n=1041, d=480 | 16.8 [15.0, 18.7]; n=1298, d=487 | 17.3 [15.3, 19.4]; n=1129, d=477 | 11.0 [10.0, 12.0]; n=3364, d=600 |
| kv/test keys | N2b-answer | 16.1 [13.9, 18.3]; n=1025, d=528 | 19.1 [16.9, 21.6]; n=1041, d=480 | 23.7 [21.5, 25.9]; n=1298, d=487 | 20.6 [18.5, 22.7]; n=1129, d=477 | 19.9 [18.6, 21.3]; n=3364, d=600 |
| kv/test keys | N2b-all | 16.1 [13.9, 18.3]; n=1025, d=528 | 19.1 [16.9, 21.6]; n=1041, d=480 | 23.7 [21.5, 25.9]; n=1298, d=487 | 20.6 [18.5, 22.7]; n=1129, d=477 | 19.9 [18.6, 21.3]; n=3364, d=600 |
| kv/test keys | N3 | 16.1 [14.0, 18.4]; n=1025, d=528 | 19.1 [16.8, 21.6]; n=1041, d=480 | 24.7 [22.6, 27.0]; n=1298, d=487 | 21.7 [19.6, 23.8]; n=1129, d=477 | 20.4 [19.0, 21.7]; n=3364, d=600 |
| kv/test keys | N4 | 21.2 [18.8, 23.4]; n=1025, d=528 | 21.3 [19.0, 23.7]; n=1041, d=480 | 19.6 [17.4, 21.7]; n=1298, d=487 | 17.8 [15.7, 19.8]; n=1129, d=477 | 20.6 [19.2, 22.0]; n=3364, d=600 |
| kv/test keys | B1 follows | 99.3 [98.7, 99.8]; n=858, d=481 | 98.9 [98.2, 99.5]; n=843, d=442 | 99.0 [98.3, 99.6]; n=1031, d=465 | 99.1 [98.5, 99.7]; n=926, d=450 | 99.1 [98.7, 99.4]; n=2732, d=598 |
| kv/test keys | B1 unchanged | 0.0 [0.0, 0.0]; n=858, d=481 | 0.0 [0.0, 0.0]; n=843, d=442 | 0.0 [0.0, 0.0]; n=1031, d=465 | 0.0 [0.0, 0.0]; n=926, d=450 | 0.0 [0.0, 0.0]; n=2732, d=598 |
| kv/test keys | B1 other | 0.7 [0.2, 1.3]; n=858, d=481 | 1.1 [0.5, 1.8]; n=843, d=442 | 1.0 [0.4, 1.7]; n=1031, d=465 | 0.9 [0.3, 1.5]; n=926, d=450 | 0.9 [0.6, 1.3]; n=2732, d=598 |
| delayed2h/seen pairs | N0 | 100.0 [100.0, 100.0]; n=103, d=54 | 99.4 [98.1, 100.0]; n=175, d=92 | 99.7 [99.3, 100.0]; n=977, d=471 | 99.7 [99.3, 100.0]; n=964, d=471 | 99.7 [99.4, 99.9]; n=1255, d=600 |
| delayed2h/seen pairs | N1 | 12.6 [6.9, 19.3]; n=103, d=54 | 17.1 [11.8, 23.5]; n=175, d=92 | 20.2 [17.8, 22.5]; n=977, d=471 | 20.4 [18.0, 22.8]; n=964, d=471 | 19.1 [17.0, 21.2]; n=1255, d=600 |
| delayed2h/seen pairs | N2a | 15.5 [9.4, 22.5]; n=103, d=54 | 18.9 [13.1, 25.4]; n=175, d=92 | 22.7 [20.2, 25.2]; n=977, d=471 | 23.0 [20.5, 25.6]; n=964, d=471 | 21.6 [19.4, 23.8]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-answer | 9.7 [4.3, 15.7]; n=103, d=54 | 18.3 [13.0, 24.7]; n=175, d=92 | 19.9 [17.4, 22.4]; n=977, d=471 | 20.1 [17.6, 22.7]; n=964, d=471 | 18.8 [16.7, 20.9]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-all | 9.7 [4.3, 15.7]; n=103, d=54 | 18.3 [13.0, 24.7]; n=175, d=92 | 19.9 [17.4, 22.4]; n=977, d=471 | 20.1 [17.6, 22.7]; n=964, d=471 | 18.8 [16.7, 20.9]; n=1255, d=600 |
| delayed2h/seen pairs | N3 | 12.6 [6.9, 19.3]; n=103, d=54 | 17.1 [11.8, 23.5]; n=175, d=92 | 20.2 [17.8, 22.5]; n=977, d=471 | 20.4 [18.0, 22.8]; n=964, d=471 | 19.1 [17.0, 21.2]; n=1255, d=600 |
| delayed2h/seen pairs | N4 | 13.6 [7.7, 20.6]; n=103, d=54 | 13.1 [8.7, 17.9]; n=175, d=92 | 14.3 [12.3, 16.6]; n=977, d=471 | 14.1 [12.1, 16.4]; n=964, d=471 | 14.1 [12.3, 16.1]; n=1255, d=600 |
| delayed2h/seen pairs | B1 follows | 99.0 [96.8, 100.0]; n=103, d=54 | 99.4 [98.1, 100.0]; n=174, d=92 | 99.8 [99.5, 100.0]; n=974, d=471 | 99.8 [99.5, 100.0]; n=961, d=471 | 99.7 [99.4, 99.9]; n=1251, d=600 |
| delayed2h/seen pairs | B1 unchanged | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=174, d=92 | 0.0 [0.0, 0.0]; n=974, d=471 | 0.0 [0.0, 0.0]; n=961, d=471 | 0.0 [0.0, 0.0]; n=1251, d=600 |
| delayed2h/seen pairs | B1 other | 1.0 [0.0, 3.2]; n=103, d=54 | 0.6 [0.0, 1.9]; n=174, d=92 | 0.2 [0.0, 0.5]; n=974, d=471 | 0.2 [0.0, 0.5]; n=961, d=471 | 0.3 [0.1, 0.6]; n=1251, d=600 |
| delayed2h/seen pairs | B2 follows | NE: 50.7 [39.4, 60.9]; n=73, d=37 | 43.0 [34.1, 51.2]; n=135, d=69 | 39.3 [35.9, 42.5]; n=723, d=343 | 39.0 [35.7, 42.3]; n=712, d=342 | 40.7 [37.8, 43.6]; n=931, d=433 |
| delayed2h/seen pairs | B2 unchanged | NE: 21.9 [13.8, 31.5]; n=73, d=37 | 26.7 [18.6, 35.0]; n=135, d=69 | 33.9 [30.7, 36.9]; n=723, d=343 | 34.3 [31.0, 37.3]; n=712, d=342 | 31.9 [29.1, 34.7]; n=931, d=433 |
| delayed2h/seen pairs | B2 other | NE: 27.4 [16.9, 38.9]; n=73, d=37 | 30.4 [22.4, 39.2]; n=135, d=69 | 26.8 [23.6, 30.0]; n=723, d=343 | 26.7 [23.4, 29.9]; n=712, d=342 | 27.4 [24.5, 30.5]; n=931, d=433 |
| delayed2h/seen pairs | B3 follows | 73.8 [65.2, 81.0]; n=103, d=54 | 71.8 [65.5, 77.9]; n=174, d=92 | 71.1 [68.6, 73.5]; n=974, d=471 | 70.8 [68.2, 73.2]; n=961, d=471 | 71.5 [69.2, 73.6]; n=1251, d=600 |
| delayed2h/seen pairs | B3 unchanged | 7.8 [3.0, 13.0]; n=103, d=54 | 8.6 [4.6, 13.0]; n=174, d=92 | 8.0 [6.5, 9.6]; n=974, d=471 | 8.1 [6.5, 9.8]; n=961, d=471 | 8.1 [6.6, 9.5]; n=1251, d=600 |
| delayed2h/seen pairs | B3 other | 18.4 [11.1, 26.5]; n=103, d=54 | 19.5 [14.1, 25.5]; n=174, d=92 | 20.8 [18.3, 23.1]; n=974, d=471 | 21.1 [18.5, 23.5]; n=961, d=471 | 20.5 [18.4, 22.6]; n=1251, d=600 |
| delayed2h/new pairs | N0 | NE: 100.0 [100.0, 100.0]; n=99, d=52 | 99.4 [98.2, 100.0]; n=176, d=91 | 99.5 [99.0, 99.9]; n=979, d=471 | 99.5 [99.0, 99.9]; n=964, d=470 | 99.5 [99.1, 99.8]; n=1254, d=600 |
| delayed2h/new pairs | N1 | NE: 10.1 [4.5, 16.4]; n=99, d=52 | 20.5 [15.3, 26.4]; n=176, d=91 | 19.5 [17.4, 21.8]; n=979, d=471 | 19.7 [17.5, 22.0]; n=964, d=470 | 18.9 [17.0, 20.8]; n=1254, d=600 |
| delayed2h/new pairs | N2a | NE: 12.1 [6.4, 19.4]; n=99, d=52 | 25.6 [20.1, 31.8]; n=176, d=91 | 22.1 [19.6, 24.5]; n=979, d=471 | 22.4 [19.9, 24.9]; n=964, d=470 | 21.8 [19.6, 23.9]; n=1254, d=600 |
| delayed2h/new pairs | N2b-answer | NE: 11.1 [5.4, 17.7]; n=99, d=52 | 22.2 [16.9, 28.2]; n=176, d=91 | 19.6 [17.3, 21.8]; n=979, d=471 | 19.8 [17.5, 22.0]; n=964, d=470 | 19.3 [17.2, 21.3]; n=1254, d=600 |
| delayed2h/new pairs | N2b-all | NE: 11.1 [5.4, 17.7]; n=99, d=52 | 22.2 [16.9, 28.2]; n=176, d=91 | 19.6 [17.3, 21.8]; n=979, d=471 | 19.8 [17.5, 22.0]; n=964, d=470 | 19.3 [17.2, 21.3]; n=1254, d=600 |
| delayed2h/new pairs | N3 | NE: 10.1 [4.5, 16.4]; n=99, d=52 | 20.5 [15.3, 26.4]; n=176, d=91 | 19.5 [17.4, 21.8]; n=979, d=471 | 19.7 [17.5, 22.0]; n=964, d=470 | 18.9 [17.0, 20.8]; n=1254, d=600 |
| delayed2h/new pairs | N4 | NE: 14.1 [8.1, 21.3]; n=99, d=52 | 11.4 [6.8, 16.3]; n=176, d=91 | 13.9 [11.8, 16.0]; n=979, d=471 | 13.7 [11.6, 15.8]; n=964, d=470 | 13.6 [11.7, 15.3]; n=1254, d=600 |
| delayed2h/new pairs | B1 follows | NE: 100.0 [100.0, 100.0]; n=99, d=52 | 99.4 [98.2, 100.0]; n=175, d=91 | 99.8 [99.5, 100.0]; n=974, d=471 | 99.8 [99.5, 100.0]; n=959, d=470 | 99.8 [99.5, 100.0]; n=1248, d=600 |
| delayed2h/new pairs | B1 unchanged | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=175, d=91 | 0.0 [0.0, 0.0]; n=974, d=471 | 0.0 [0.0, 0.0]; n=959, d=470 | 0.0 [0.0, 0.0]; n=1248, d=600 |
| delayed2h/new pairs | B1 other | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.6 [0.0, 1.8]; n=175, d=91 | 0.2 [0.0, 0.5]; n=974, d=471 | 0.2 [0.0, 0.5]; n=959, d=470 | 0.2 [0.0, 0.5]; n=1248, d=600 |
| delayed2h/new pairs | B2 follows | NE: 53.7 [42.6, 64.3]; n=67, d=34 | 50.0 [40.6, 57.3]; n=140, d=70 | 42.4 [39.0, 45.6]; n=753, d=358 | 42.1 [38.7, 45.5]; n=739, d=355 | 44.3 [41.2, 47.2]; n=960, d=449 |
| delayed2h/new pairs | B2 unchanged | NE: 20.9 [11.5, 32.6]; n=67, d=34 | 30.7 [24.1, 39.1]; n=140, d=70 | 32.3 [29.2, 35.3]; n=753, d=358 | 32.5 [29.3, 35.5]; n=739, d=355 | 31.2 [28.5, 34.0]; n=960, d=449 |
| delayed2h/new pairs | B2 other | NE: 25.4 [14.8, 36.8]; n=67, d=34 | 19.3 [13.1, 26.2]; n=140, d=70 | 25.4 [22.5, 28.3]; n=753, d=358 | 25.4 [22.4, 28.5]; n=739, d=355 | 24.5 [21.9, 27.2]; n=960, d=449 |
| delayed2h/new pairs | B3 follows | NE: 73.7 [65.9, 81.0]; n=99, d=52 | 71.4 [65.0, 77.1]; n=175, d=91 | 71.9 [69.5, 74.2]; n=974, d=471 | 71.5 [69.1, 73.7]; n=959, d=470 | 72.0 [69.8, 74.0]; n=1248, d=600 |
| delayed2h/new pairs | B3 unchanged | NE: 8.1 [3.2, 13.9]; n=99, d=52 | 8.6 [4.7, 13.0]; n=175, d=91 | 6.9 [5.4, 8.5]; n=974, d=471 | 7.0 [5.4, 8.6]; n=959, d=470 | 7.2 [5.8, 8.6]; n=1248, d=600 |
| delayed2h/new pairs | B3 other | NE: 18.2 [11.5, 25.5]; n=99, d=52 | 20.0 [15.3, 25.6]; n=175, d=91 | 21.3 [19.0, 23.6]; n=974, d=471 | 21.5 [19.2, 23.9]; n=959, d=470 | 20.8 [18.8, 22.8]; n=1248, d=600 |
| delayed2h/never-seen bytes | N0 | 95.0 [90.9, 98.3]; n=119, d=55 | 95.3 [91.7, 98.5]; n=192, d=97 | 98.0 [97.1, 98.9]; n=953, d=463 | 98.0 [97.1, 98.8]; n=936, d=463 | 97.3 [96.4, 98.2]; n=1264, d=600 |
| delayed2h/never-seen bytes | N1 | 8.4 [3.8, 13.8]; n=119, d=55 | 16.1 [11.5, 20.6]; n=192, d=97 | 15.0 [12.8, 17.2]; n=953, d=463 | 15.3 [13.0, 17.5]; n=936, d=463 | 14.6 [12.7, 16.3]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2a | 9.2 [4.1, 14.4]; n=119, d=55 | 16.7 [12.3, 21.4]; n=192, d=97 | 18.4 [16.0, 20.7]; n=953, d=463 | 18.7 [16.2, 21.1]; n=936, d=463 | 17.2 [15.2, 19.2]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-answer | 10.1 [5.3, 15.6]; n=119, d=55 | 16.1 [11.7, 20.8]; n=192, d=97 | 16.6 [14.4, 18.9]; n=953, d=463 | 16.9 [14.7, 19.2]; n=936, d=463 | 15.9 [14.0, 17.8]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-all | 10.1 [5.3, 15.6]; n=119, d=55 | 16.1 [11.7, 20.8]; n=192, d=97 | 16.6 [14.4, 18.9]; n=953, d=463 | 16.9 [14.7, 19.2]; n=936, d=463 | 15.9 [14.0, 17.8]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3 | 8.4 [3.8, 13.8]; n=119, d=55 | 16.1 [11.5, 20.6]; n=192, d=97 | 15.0 [12.8, 17.2]; n=953, d=463 | 15.3 [13.0, 17.5]; n=936, d=463 | 14.6 [12.7, 16.3]; n=1264, d=600 |
| delayed2h/never-seen bytes | N4 | 9.2 [4.3, 14.3]; n=119, d=55 | 10.4 [6.5, 14.7]; n=192, d=97 | 13.5 [11.4, 15.8]; n=953, d=463 | 13.4 [11.3, 15.5]; n=936, d=463 | 12.7 [10.8, 14.4]; n=1264, d=600 |
| delayed2h/never-seen bytes | B1 follows | 97.3 [94.1, 100.0]; n=113, d=54 | 95.1 [91.8, 97.8]; n=183, d=97 | 98.7 [97.9, 99.4]; n=934, d=463 | 98.7 [97.9, 99.3]; n=917, d=463 | 98.0 [97.3, 98.8]; n=1230, d=599 |
| delayed2h/never-seen bytes | B1 unchanged | 0.0 [0.0, 0.0]; n=113, d=54 | 0.0 [0.0, 0.0]; n=183, d=97 | 0.0 [0.0, 0.0]; n=934, d=463 | 0.0 [0.0, 0.0]; n=917, d=463 | 0.0 [0.0, 0.0]; n=1230, d=599 |
| delayed2h/never-seen bytes | B1 other | 2.7 [0.0, 5.9]; n=113, d=54 | 4.9 [2.2, 8.2]; n=183, d=97 | 1.3 [0.6, 2.1]; n=934, d=463 | 1.3 [0.7, 2.1]; n=917, d=463 | 2.0 [1.2, 2.7]; n=1230, d=599 |
| delayed2h/never-seen bytes | B2 follows | NE: 62.8 [51.6, 72.9]; n=86, d=39 | 55.2 [47.6, 62.6]; n=134, d=69 | 56.8 [53.3, 60.5]; n=716, d=353 | 56.8 [53.3, 60.5]; n=702, d=351 | 57.2 [54.1, 60.3]; n=936, d=450 |
| delayed2h/never-seen bytes | B2 unchanged | NE: 18.6 [11.1, 28.4]; n=86, d=39 | 23.1 [16.9, 29.8]; n=134, d=69 | 22.1 [19.4, 25.1]; n=716, d=353 | 21.9 [19.3, 24.9]; n=702, d=351 | 21.9 [19.5, 24.5]; n=936, d=450 |
| delayed2h/never-seen bytes | B2 other | NE: 18.6 [9.3, 28.6]; n=86, d=39 | 21.6 [15.1, 28.3]; n=134, d=69 | 21.1 [17.9, 24.4]; n=716, d=353 | 21.2 [18.0, 24.6]; n=702, d=351 | 20.9 [18.4, 23.9]; n=936, d=450 |
| delayed2h/never-seen bytes | B3 follows | 78.8 [72.8, 85.0]; n=113, d=54 | 74.9 [68.8, 80.6]; n=183, d=97 | 73.9 [71.4, 76.3]; n=934, d=463 | 73.5 [71.0, 76.0]; n=917, d=463 | 74.5 [72.3, 76.5]; n=1230, d=599 |
| delayed2h/never-seen bytes | B3 unchanged | 3.5 [0.8, 6.7]; n=113, d=54 | 4.9 [2.1, 8.4]; n=183, d=97 | 6.3 [4.9, 7.9]; n=934, d=463 | 6.3 [4.9, 7.9]; n=917, d=463 | 5.9 [4.6, 7.1]; n=1230, d=599 |
| delayed2h/never-seen bytes | B3 other | 17.7 [11.4, 24.0]; n=113, d=54 | 20.2 [14.5, 26.0]; n=183, d=97 | 19.8 [17.5, 22.1]; n=934, d=463 | 20.2 [17.9, 22.5]; n=917, d=463 | 19.7 [17.7, 21.6]; n=1230, d=599 |

## core-s2

There is no reader; all buckets are distance strata only.

Checkpoint: `/home/joe/hsn-experiments/vast2/exp/gpu2/runs/a14_reader/hold_k12nw_ag_ng2_cutC_cells64_18k_gsum-s2-ac1019110d5b/exp17/step_00018000.pt`. Status: **complete**.

| Task / keys | Condition | Near | Mid | Far | Far-strict | All |
| --- | --- | --- | --- | --- | --- | --- |
| kv/training keys | N0 | 99.8 [99.5, 100.0]; n=1062, d=536 | 99.2 [98.5, 99.7]; n=998, d=479 | 98.5 [97.8, 99.1]; n=1264, d=485 | 98.5 [97.8, 99.2]; n=1082, d=475 | 99.1 [98.8, 99.4]; n=3324, d=600 |
| kv/training keys | N1 | 1.4 [0.8, 2.2]; n=1062, d=536 | 2.1 [1.3, 3.1]; n=998, d=479 | 1.9 [1.2, 2.7]; n=1264, d=485 | 2.0 [1.3, 2.9]; n=1082, d=475 | 1.8 [1.4, 2.2]; n=3324, d=600 |
| kv/training keys | N2a | 5.4 [4.0, 6.6]; n=1062, d=536 | 5.3 [4.0, 6.7]; n=998, d=479 | 8.3 [6.8, 9.8]; n=1264, d=485 | 8.9 [7.3, 10.4]; n=1082, d=475 | 6.5 [5.7, 7.2]; n=3324, d=600 |
| kv/training keys | N2b-answer | 15.8 [13.6, 17.9]; n=1062, d=536 | 14.4 [12.4, 16.7]; n=998, d=479 | 16.9 [15.1, 18.8]; n=1264, d=485 | 15.2 [13.4, 16.9]; n=1082, d=475 | 15.8 [14.6, 17.0]; n=3324, d=600 |
| kv/training keys | N2b-all | 15.8 [13.6, 17.9]; n=1062, d=536 | 14.4 [12.4, 16.7]; n=998, d=479 | 16.9 [15.1, 18.8]; n=1264, d=485 | 15.2 [13.4, 16.9]; n=1082, d=475 | 15.8 [14.6, 17.0]; n=3324, d=600 |
| kv/training keys | N3 | 1.4 [0.8, 2.2]; n=1062, d=536 | 2.1 [1.3, 3.1]; n=998, d=479 | 1.9 [1.2, 2.7]; n=1264, d=485 | 2.0 [1.3, 2.9]; n=1082, d=475 | 1.8 [1.4, 2.2]; n=3324, d=600 |
| kv/training keys | B1 follows | 99.9 [99.7, 100.0]; n=892, d=489 | 100.0 [100.0, 100.0]; n=845, d=451 | 99.2 [98.7, 99.7]; n=1033, d=458 | 99.3 [98.7, 99.8]; n=904, d=444 | 99.7 [99.5, 99.9]; n=2770, d=589 |
| kv/training keys | B1 unchanged | 0.0 [0.0, 0.0]; n=892, d=489 | 0.0 [0.0, 0.0]; n=845, d=451 | 0.0 [0.0, 0.0]; n=1033, d=458 | 0.0 [0.0, 0.0]; n=904, d=444 | 0.0 [0.0, 0.0]; n=2770, d=589 |
| kv/training keys | B1 other | 0.1 [0.0, 0.3]; n=892, d=489 | 0.0 [0.0, 0.0]; n=845, d=451 | 0.8 [0.3, 1.3]; n=1033, d=458 | 0.7 [0.2, 1.3]; n=904, d=444 | 0.3 [0.1, 0.5]; n=2770, d=589 |
| kv/dev keys | N0 | 99.8 [99.5, 100.0]; n=1071, d=531 | 99.0 [98.3, 99.6]; n=1011, d=476 | 98.0 [97.2, 98.7]; n=1263, d=477 | 98.1 [97.3, 98.8]; n=1095, d=468 | 98.9 [98.5, 99.2]; n=3345, d=600 |
| kv/dev keys | N1 | 1.6 [0.9, 2.4]; n=1071, d=531 | 1.1 [0.5, 1.7]; n=1011, d=476 | 1.4 [0.8, 2.2]; n=1263, d=477 | 1.4 [0.7, 2.1]; n=1095, d=468 | 1.4 [1.0, 1.8]; n=3345, d=600 |
| kv/dev keys | N2a | 5.0 [3.8, 6.4]; n=1071, d=531 | 4.9 [3.6, 6.4]; n=1011, d=476 | 6.4 [5.2, 7.9]; n=1263, d=477 | 6.7 [5.4, 8.1]; n=1095, d=468 | 5.5 [4.8, 6.3]; n=3345, d=600 |
| kv/dev keys | N2b-answer | 16.5 [14.3, 18.7]; n=1071, d=531 | 15.4 [13.3, 17.7]; n=1011, d=476 | 14.6 [12.6, 16.5]; n=1263, d=477 | 12.2 [10.4, 14.0]; n=1095, d=468 | 15.5 [14.1, 16.8]; n=3345, d=600 |
| kv/dev keys | N2b-all | 16.5 [14.3, 18.7]; n=1071, d=531 | 15.4 [13.3, 17.7]; n=1011, d=476 | 14.6 [12.6, 16.5]; n=1263, d=477 | 12.2 [10.4, 14.0]; n=1095, d=468 | 15.5 [14.1, 16.8]; n=3345, d=600 |
| kv/dev keys | N3 | 1.6 [0.9, 2.4]; n=1071, d=531 | 1.1 [0.5, 1.7]; n=1011, d=476 | 1.4 [0.8, 2.2]; n=1263, d=477 | 1.4 [0.7, 2.1]; n=1095, d=468 | 1.4 [1.0, 1.8]; n=3345, d=600 |
| kv/dev keys | B1 follows | 99.8 [99.4, 100.0]; n=892, d=490 | 99.4 [98.9, 99.9]; n=846, d=439 | 98.6 [97.9, 99.2]; n=1058, d=453 | 98.8 [98.2, 99.5]; n=941, d=437 | 99.2 [98.9, 99.5]; n=2796, d=594 |
| kv/dev keys | B1 unchanged | 0.0 [0.0, 0.0]; n=892, d=490 | 0.0 [0.0, 0.0]; n=846, d=439 | 0.0 [0.0, 0.0]; n=1058, d=453 | 0.0 [0.0, 0.0]; n=941, d=437 | 0.0 [0.0, 0.0]; n=2796, d=594 |
| kv/dev keys | B1 other | 0.2 [0.0, 0.6]; n=892, d=490 | 0.6 [0.1, 1.1]; n=846, d=439 | 1.4 [0.8, 2.1]; n=1058, d=453 | 1.2 [0.5, 1.8]; n=941, d=437 | 0.8 [0.5, 1.1]; n=2796, d=594 |
| kv/test keys | N0 | 100.0 [100.0, 100.0]; n=1025, d=528 | 99.5 [99.1, 99.9]; n=1041, d=480 | 98.1 [97.3, 98.8]; n=1298, d=487 | 97.9 [97.0, 98.6]; n=1129, d=477 | 99.1 [98.8, 99.4]; n=3364, d=600 |
| kv/test keys | N1 | 1.7 [0.9, 2.4]; n=1025, d=528 | 1.8 [1.1, 2.7]; n=1041, d=480 | 2.3 [1.5, 3.2]; n=1298, d=487 | 2.4 [1.6, 3.3]; n=1129, d=477 | 2.0 [1.5, 2.4]; n=3364, d=600 |
| kv/test keys | N2a | 5.2 [3.8, 6.5]; n=1025, d=528 | 6.4 [5.1, 8.0]; n=1041, d=480 | 7.2 [5.9, 8.6]; n=1298, d=487 | 7.5 [6.2, 9.0]; n=1129, d=477 | 6.4 [5.7, 7.1]; n=3364, d=600 |
| kv/test keys | N2b-answer | 15.6 [13.4, 17.8]; n=1025, d=528 | 17.8 [15.5, 20.1]; n=1041, d=480 | 17.2 [15.1, 19.3]; n=1298, d=487 | 14.3 [12.4, 16.3]; n=1129, d=477 | 16.9 [15.5, 18.2]; n=3364, d=600 |
| kv/test keys | N2b-all | 15.6 [13.4, 17.8]; n=1025, d=528 | 17.8 [15.5, 20.1]; n=1041, d=480 | 17.2 [15.1, 19.3]; n=1298, d=487 | 14.3 [12.4, 16.3]; n=1129, d=477 | 16.9 [15.5, 18.2]; n=3364, d=600 |
| kv/test keys | N3 | 1.7 [0.9, 2.4]; n=1025, d=528 | 1.8 [1.1, 2.7]; n=1041, d=480 | 2.3 [1.5, 3.2]; n=1298, d=487 | 2.4 [1.6, 3.3]; n=1129, d=477 | 2.0 [1.5, 2.4]; n=3364, d=600 |
| kv/test keys | B1 follows | 100.0 [100.0, 100.0]; n=865, d=483 | 99.5 [99.0, 99.9]; n=851, d=444 | 99.0 [98.3, 99.5]; n=1050, d=464 | 99.0 [98.4, 99.6]; n=943, d=449 | 99.5 [99.2, 99.7]; n=2766, d=597 |
| kv/test keys | B1 unchanged | 0.0 [0.0, 0.0]; n=865, d=483 | 0.0 [0.0, 0.0]; n=851, d=444 | 0.2 [0.0, 0.5]; n=1050, d=464 | 0.0 [0.0, 0.0]; n=943, d=449 | 0.1 [0.0, 0.2]; n=2766, d=597 |
| kv/test keys | B1 other | 0.0 [0.0, 0.0]; n=865, d=483 | 0.5 [0.1, 1.0]; n=851, d=444 | 0.9 [0.4, 1.4]; n=1050, d=464 | 1.0 [0.4, 1.6]; n=943, d=449 | 0.5 [0.2, 0.7]; n=2766, d=597 |
| delayed2h/seen pairs | N0 | 100.0 [100.0, 100.0]; n=103, d=54 | 100.0 [100.0, 100.0]; n=175, d=92 | 99.8 [99.5, 100.0]; n=977, d=471 | 99.8 [99.5, 100.0]; n=964, d=471 | 99.8 [99.6, 100.0]; n=1255, d=600 |
| delayed2h/seen pairs | N1 | 1.9 [0.0, 5.3]; n=103, d=54 | 2.3 [0.5, 4.3]; n=175, d=92 | 1.7 [1.0, 2.6]; n=977, d=471 | 1.8 [1.0, 2.6]; n=964, d=471 | 1.8 [1.2, 2.6]; n=1255, d=600 |
| delayed2h/seen pairs | N2a | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=977, d=471 | 0.0 [0.0, 0.0]; n=964, d=471 | 0.0 [0.0, 0.0]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-answer | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=977, d=471 | 0.0 [0.0, 0.0]; n=964, d=471 | 0.0 [0.0, 0.0]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-all | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=977, d=471 | 0.0 [0.0, 0.0]; n=964, d=471 | 0.0 [0.0, 0.0]; n=1255, d=600 |
| delayed2h/seen pairs | N3 | 1.9 [0.0, 5.3]; n=103, d=54 | 2.3 [0.5, 4.3]; n=175, d=92 | 1.7 [1.0, 2.6]; n=977, d=471 | 1.8 [1.0, 2.6]; n=964, d=471 | 1.8 [1.2, 2.6]; n=1255, d=600 |
| delayed2h/seen pairs | B1 follows | 100.0 [100.0, 100.0]; n=103, d=54 | 99.4 [98.1, 100.0]; n=175, d=92 | 100.0 [100.0, 100.0]; n=975, d=471 | 100.0 [100.0, 100.0]; n=962, d=471 | 99.9 [99.8, 100.0]; n=1253, d=600 |
| delayed2h/seen pairs | B1 unchanged | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=975, d=471 | 0.0 [0.0, 0.0]; n=962, d=471 | 0.0 [0.0, 0.0]; n=1253, d=600 |
| delayed2h/seen pairs | B1 other | 0.0 [0.0, 0.0]; n=103, d=54 | 0.6 [0.0, 1.9]; n=175, d=92 | 0.0 [0.0, 0.0]; n=975, d=471 | 0.0 [0.0, 0.0]; n=962, d=471 | 0.1 [0.0, 0.2]; n=1253, d=600 |
| delayed2h/seen pairs | B2 follows | NE: 95.9 [90.0, 100.0]; n=73, d=37 | 92.6 [88.4, 96.4]; n=136, d=69 | 96.1 [94.5, 97.6]; n=725, d=343 | 96.1 [94.4, 97.5]; n=714, d=342 | 95.6 [94.2, 97.0]; n=934, d=433 |
| delayed2h/seen pairs | B2 unchanged | NE: 0.0 [0.0, 0.0]; n=73, d=37 | 1.5 [0.0, 3.6]; n=136, d=69 | 1.2 [0.5, 2.1]; n=725, d=343 | 1.3 [0.5, 2.2]; n=714, d=342 | 1.2 [0.5, 1.9]; n=934, d=433 |
| delayed2h/seen pairs | B2 other | NE: 4.1 [0.0, 10.0]; n=73, d=37 | 5.9 [2.5, 9.9]; n=136, d=69 | 2.6 [1.4, 4.1]; n=725, d=343 | 2.7 [1.4, 4.1]; n=714, d=342 | 3.2 [2.1, 4.5]; n=934, d=433 |
| delayed2h/seen pairs | B3 follows | 100.0 [100.0, 100.0]; n=103, d=54 | 99.4 [98.1, 100.0]; n=175, d=92 | 100.0 [100.0, 100.0]; n=975, d=471 | 100.0 [100.0, 100.0]; n=962, d=471 | 99.9 [99.8, 100.0]; n=1253, d=600 |
| delayed2h/seen pairs | B3 unchanged | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=975, d=471 | 0.0 [0.0, 0.0]; n=962, d=471 | 0.0 [0.0, 0.0]; n=1253, d=600 |
| delayed2h/seen pairs | B3 other | 0.0 [0.0, 0.0]; n=103, d=54 | 0.6 [0.0, 1.9]; n=175, d=92 | 0.0 [0.0, 0.0]; n=975, d=471 | 0.0 [0.0, 0.0]; n=962, d=471 | 0.1 [0.0, 0.2]; n=1253, d=600 |
| delayed2h/new pairs | N0 | NE: 100.0 [100.0, 100.0]; n=99, d=52 | 100.0 [100.0, 100.0]; n=176, d=91 | 99.6 [99.2, 99.9]; n=979, d=471 | 99.6 [99.1, 99.9]; n=964, d=470 | 99.7 [99.4, 99.9]; n=1254, d=600 |
| delayed2h/new pairs | N1 | NE: 1.0 [0.0, 3.6]; n=99, d=52 | 1.1 [0.0, 3.0]; n=176, d=91 | 2.1 [1.3, 3.1]; n=979, d=471 | 2.2 [1.3, 3.2]; n=964, d=470 | 1.9 [1.2, 2.7]; n=1254, d=600 |
| delayed2h/new pairs | N2a | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=979, d=471 | 0.0 [0.0, 0.0]; n=964, d=470 | 0.0 [0.0, 0.0]; n=1254, d=600 |
| delayed2h/new pairs | N2b-answer | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=979, d=471 | 0.0 [0.0, 0.0]; n=964, d=470 | 0.0 [0.0, 0.0]; n=1254, d=600 |
| delayed2h/new pairs | N2b-all | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=979, d=471 | 0.0 [0.0, 0.0]; n=964, d=470 | 0.0 [0.0, 0.0]; n=1254, d=600 |
| delayed2h/new pairs | N3 | NE: 1.0 [0.0, 3.6]; n=99, d=52 | 1.1 [0.0, 3.0]; n=176, d=91 | 2.1 [1.3, 3.1]; n=979, d=471 | 2.2 [1.3, 3.2]; n=964, d=470 | 1.9 [1.2, 2.7]; n=1254, d=600 |
| delayed2h/new pairs | B1 follows | NE: 99.0 [96.6, 100.0]; n=99, d=52 | 98.9 [97.2, 100.0]; n=176, d=91 | 99.9 [99.7, 100.0]; n=975, d=471 | 99.9 [99.7, 100.0]; n=960, d=470 | 99.7 [99.4, 99.9]; n=1250, d=600 |
| delayed2h/new pairs | B1 unchanged | NE: 1.0 [0.0, 3.4]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=975, d=471 | 0.0 [0.0, 0.0]; n=960, d=470 | 0.1 [0.0, 0.2]; n=1250, d=600 |
| delayed2h/new pairs | B1 other | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 1.1 [0.0, 2.8]; n=176, d=91 | 0.1 [0.0, 0.3]; n=975, d=471 | 0.1 [0.0, 0.3]; n=960, d=470 | 0.2 [0.0, 0.6]; n=1250, d=600 |
| delayed2h/new pairs | B2 follows | NE: 97.0 [92.5, 100.0]; n=67, d=34 | 94.3 [90.3, 97.7]; n=140, d=70 | 95.5 [93.9, 96.9]; n=753, d=358 | 95.4 [93.8, 96.9]; n=739, d=355 | 95.4 [93.9, 96.7]; n=960, d=449 |
| delayed2h/new pairs | B2 unchanged | NE: 1.5 [0.0, 5.1]; n=67, d=34 | 0.7 [0.0, 2.4]; n=140, d=70 | 1.2 [0.5, 2.0]; n=753, d=358 | 1.2 [0.5, 2.0]; n=739, d=355 | 1.1 [0.5, 1.9]; n=960, d=449 |
| delayed2h/new pairs | B2 other | NE: 1.5 [0.0, 5.3]; n=67, d=34 | 5.0 [1.8, 8.8]; n=140, d=70 | 3.3 [2.1, 4.7]; n=753, d=358 | 3.4 [2.1, 4.8]; n=739, d=355 | 3.4 [2.3, 4.7]; n=960, d=449 |
| delayed2h/new pairs | B3 follows | NE: 99.0 [96.6, 100.0]; n=99, d=52 | 98.9 [97.2, 100.0]; n=176, d=91 | 99.9 [99.7, 100.0]; n=975, d=471 | 99.9 [99.7, 100.0]; n=960, d=470 | 99.7 [99.4, 99.9]; n=1250, d=600 |
| delayed2h/new pairs | B3 unchanged | NE: 1.0 [0.0, 3.4]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=975, d=471 | 0.0 [0.0, 0.0]; n=960, d=470 | 0.1 [0.0, 0.2]; n=1250, d=600 |
| delayed2h/new pairs | B3 other | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 1.1 [0.0, 2.8]; n=176, d=91 | 0.1 [0.0, 0.3]; n=975, d=471 | 0.1 [0.0, 0.3]; n=960, d=470 | 0.2 [0.0, 0.6]; n=1250, d=600 |
| delayed2h/never-seen bytes | N0 | 97.5 [94.7, 100.0]; n=119, d=55 | 99.5 [98.3, 100.0]; n=192, d=97 | 99.7 [99.2, 100.0]; n=953, d=463 | 99.7 [99.2, 100.0]; n=936, d=463 | 99.4 [99.0, 99.8]; n=1264, d=600 |
| delayed2h/never-seen bytes | N1 | 2.5 [0.0, 5.2]; n=119, d=55 | 2.6 [0.5, 5.3]; n=192, d=97 | 2.1 [1.2, 3.2]; n=953, d=463 | 2.0 [1.2, 3.0]; n=936, d=463 | 2.2 [1.4, 3.1]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2a | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.0 [0.0, 0.0]; n=953, d=463 | 0.0 [0.0, 0.0]; n=936, d=463 | 0.0 [0.0, 0.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-answer | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.0 [0.0, 0.0]; n=953, d=463 | 0.0 [0.0, 0.0]; n=936, d=463 | 0.0 [0.0, 0.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-all | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.0 [0.0, 0.0]; n=953, d=463 | 0.0 [0.0, 0.0]; n=936, d=463 | 0.0 [0.0, 0.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3 | 2.5 [0.0, 5.2]; n=119, d=55 | 2.6 [0.5, 5.3]; n=192, d=97 | 2.1 [1.2, 3.2]; n=953, d=463 | 2.0 [1.2, 3.0]; n=936, d=463 | 2.2 [1.4, 3.1]; n=1264, d=600 |
| delayed2h/never-seen bytes | B1 follows | 100.0 [100.0, 100.0]; n=116, d=55 | 100.0 [100.0, 100.0]; n=191, d=97 | 99.9 [99.7, 100.0]; n=950, d=463 | 99.9 [99.7, 100.0]; n=933, d=463 | 99.9 [99.8, 100.0]; n=1257, d=600 |
| delayed2h/never-seen bytes | B1 unchanged | 0.0 [0.0, 0.0]; n=116, d=55 | 0.0 [0.0, 0.0]; n=191, d=97 | 0.0 [0.0, 0.0]; n=950, d=463 | 0.0 [0.0, 0.0]; n=933, d=463 | 0.0 [0.0, 0.0]; n=1257, d=600 |
| delayed2h/never-seen bytes | B1 other | 0.0 [0.0, 0.0]; n=116, d=55 | 0.0 [0.0, 0.0]; n=191, d=97 | 0.1 [0.0, 0.3]; n=950, d=463 | 0.1 [0.0, 0.3]; n=933, d=463 | 0.1 [0.0, 0.2]; n=1257, d=600 |
| delayed2h/never-seen bytes | B2 follows | NE: 97.7 [94.1, 100.0]; n=88, d=39 | 95.0 [91.3, 98.5]; n=141, d=70 | 96.4 [95.1, 97.8]; n=731, d=353 | 96.4 [95.0, 97.7]; n=717, d=351 | 96.4 [95.2, 97.5]; n=960, d=451 |
| delayed2h/never-seen bytes | B2 unchanged | NE: 1.1 [0.0, 4.0]; n=88, d=39 | 0.0 [0.0, 0.0]; n=141, d=70 | 1.2 [0.5, 2.1]; n=731, d=353 | 1.3 [0.5, 2.2]; n=717, d=351 | 1.0 [0.4, 1.8]; n=960, d=451 |
| delayed2h/never-seen bytes | B2 other | NE: 1.1 [0.0, 4.1]; n=88, d=39 | 5.0 [1.5, 8.7]; n=141, d=70 | 2.3 [1.1, 3.5]; n=731, d=353 | 2.4 [1.1, 3.6]; n=717, d=351 | 2.6 [1.6, 3.7]; n=960, d=451 |
| delayed2h/never-seen bytes | B3 follows | 100.0 [100.0, 100.0]; n=116, d=55 | 100.0 [100.0, 100.0]; n=191, d=97 | 99.9 [99.7, 100.0]; n=950, d=463 | 99.9 [99.7, 100.0]; n=933, d=463 | 99.9 [99.8, 100.0]; n=1257, d=600 |
| delayed2h/never-seen bytes | B3 unchanged | 0.0 [0.0, 0.0]; n=116, d=55 | 0.0 [0.0, 0.0]; n=191, d=97 | 0.0 [0.0, 0.0]; n=950, d=463 | 0.0 [0.0, 0.0]; n=933, d=463 | 0.0 [0.0, 0.0]; n=1257, d=600 |
| delayed2h/never-seen bytes | B3 other | 0.0 [0.0, 0.0]; n=116, d=55 | 0.0 [0.0, 0.0]; n=191, d=97 | 0.1 [0.0, 0.3]; n=950, d=463 | 0.1 [0.0, 0.3]; n=933, d=463 | 0.1 [0.0, 0.2]; n=1257, d=600 |

## core-s1

There is no reader; all buckets are distance strata only.

Checkpoint: `/home/joe/hsn-experiments/p2.10-split/runs/a14_reader/hold_k12nw_ag_ng2_cutC_cells64_18k_gsum-s1-c3df8cc1c448/exp603/step_00018000.pt`. Status: **complete**.

| Task / keys | Condition | Near | Mid | Far | Far-strict | All |
| --- | --- | --- | --- | --- | --- | --- |
| kv/training keys | N0 | 99.2 [98.7, 99.7]; n=1062, d=536 | 98.8 [98.1, 99.4]; n=998, d=479 | 97.4 [96.5, 98.2]; n=1264, d=485 | 98.0 [97.0, 98.7]; n=1082, d=475 | 98.4 [98.0, 98.8]; n=3324, d=600 |
| kv/training keys | N1 | 1.5 [0.8, 2.2]; n=1062, d=536 | 1.2 [0.6, 1.9]; n=998, d=479 | 1.8 [1.1, 2.5]; n=1264, d=485 | 1.9 [1.2, 2.7]; n=1082, d=475 | 1.5 [1.1, 1.9]; n=3324, d=600 |
| kv/training keys | N2a | 2.4 [1.5, 3.3]; n=1062, d=536 | 3.2 [2.2, 4.2]; n=998, d=479 | 9.0 [7.6, 10.5]; n=1264, d=485 | 10.4 [8.7, 12.1]; n=1082, d=475 | 5.1 [4.4, 5.8]; n=3324, d=600 |
| kv/training keys | N2b-answer | 15.4 [13.3, 17.4]; n=1062, d=536 | 14.5 [12.5, 16.9]; n=998, d=479 | 17.3 [15.4, 19.2]; n=1264, d=485 | 15.6 [13.8, 17.4]; n=1082, d=475 | 15.9 [14.6, 17.2]; n=3324, d=600 |
| kv/training keys | N2b-all | 15.4 [13.3, 17.4]; n=1062, d=536 | 14.5 [12.5, 16.9]; n=998, d=479 | 17.3 [15.4, 19.2]; n=1264, d=485 | 15.6 [13.8, 17.4]; n=1082, d=475 | 15.9 [14.6, 17.2]; n=3324, d=600 |
| kv/training keys | N3 | 1.5 [0.8, 2.2]; n=1062, d=536 | 1.2 [0.6, 1.9]; n=998, d=479 | 1.8 [1.1, 2.5]; n=1264, d=485 | 1.9 [1.2, 2.7]; n=1082, d=475 | 1.5 [1.1, 1.9]; n=3324, d=600 |
| kv/training keys | B1 follows | 99.4 [98.9, 99.9]; n=888, d=487 | 98.7 [97.9, 99.4]; n=842, d=451 | 98.3 [97.5, 99.1]; n=1026, d=456 | 98.7 [97.8, 99.4]; n=899, d=441 | 98.8 [98.4, 99.2]; n=2756, d=588 |
| kv/training keys | B1 unchanged | 0.0 [0.0, 0.0]; n=888, d=487 | 0.0 [0.0, 0.0]; n=842, d=451 | 0.0 [0.0, 0.0]; n=1026, d=456 | 0.0 [0.0, 0.0]; n=899, d=441 | 0.0 [0.0, 0.0]; n=2756, d=588 |
| kv/training keys | B1 other | 0.6 [0.1, 1.1]; n=888, d=487 | 1.3 [0.6, 2.1]; n=842, d=451 | 1.7 [0.9, 2.5]; n=1026, d=456 | 1.3 [0.6, 2.2]; n=899, d=441 | 1.2 [0.8, 1.6]; n=2756, d=588 |
| kv/dev keys | N0 | 95.6 [94.3, 96.8]; n=1071, d=531 | 93.8 [92.3, 95.2]; n=1011, d=476 | 92.0 [90.4, 93.4]; n=1263, d=477 | 92.3 [90.7, 93.8]; n=1095, d=468 | 93.7 [92.9, 94.4]; n=3345, d=600 |
| kv/dev keys | N1 | 1.3 [0.7, 1.9]; n=1071, d=531 | 1.7 [0.9, 2.5]; n=1011, d=476 | 2.1 [1.4, 3.0]; n=1263, d=477 | 2.1 [1.3, 3.0]; n=1095, d=468 | 1.7 [1.3, 2.2]; n=3345, d=600 |
| kv/dev keys | N2a | 4.4 [3.2, 5.5]; n=1071, d=531 | 4.2 [2.9, 5.5]; n=1011, d=476 | 6.9 [5.5, 8.3]; n=1263, d=477 | 7.6 [6.0, 9.1]; n=1095, d=468 | 5.3 [4.6, 6.0]; n=3345, d=600 |
| kv/dev keys | N2b-answer | 16.4 [14.2, 18.6]; n=1071, d=531 | 15.4 [13.3, 17.8]; n=1011, d=476 | 14.3 [12.2, 16.2]; n=1263, d=477 | 11.8 [9.9, 13.7]; n=1095, d=468 | 15.3 [14.0, 16.6]; n=3345, d=600 |
| kv/dev keys | N2b-all | 16.4 [14.2, 18.6]; n=1071, d=531 | 15.4 [13.3, 17.8]; n=1011, d=476 | 14.3 [12.2, 16.2]; n=1263, d=477 | 11.8 [9.9, 13.7]; n=1095, d=468 | 15.3 [14.0, 16.6]; n=3345, d=600 |
| kv/dev keys | N3 | 1.3 [0.7, 1.9]; n=1071, d=531 | 1.7 [0.9, 2.5]; n=1011, d=476 | 2.1 [1.4, 3.0]; n=1263, d=477 | 2.1 [1.3, 3.0]; n=1095, d=468 | 1.7 [1.3, 2.2]; n=3345, d=600 |
| kv/dev keys | B1 follows | 98.8 [97.9, 99.5]; n=857, d=477 | 98.0 [97.0, 98.9]; n=804, d=437 | 96.0 [94.9, 97.1]; n=992, d=443 | 95.8 [94.7, 97.1]; n=885, d=428 | 97.5 [96.9, 98.1]; n=2653, d=595 |
| kv/dev keys | B1 unchanged | 0.0 [0.0, 0.0]; n=857, d=477 | 0.1 [0.0, 0.4]; n=804, d=437 | 0.0 [0.0, 0.0]; n=992, d=443 | 0.0 [0.0, 0.0]; n=885, d=428 | 0.0 [0.0, 0.1]; n=2653, d=595 |
| kv/dev keys | B1 other | 1.2 [0.5, 2.1]; n=857, d=477 | 1.9 [1.0, 2.8]; n=804, d=437 | 4.0 [2.9, 5.1]; n=992, d=443 | 4.2 [2.9, 5.3]; n=885, d=428 | 2.5 [1.9, 3.0]; n=2653, d=595 |
| kv/test keys | N0 | 95.4 [94.1, 96.6]; n=1025, d=528 | 92.7 [91.1, 94.2]; n=1041, d=480 | 93.6 [92.3, 94.9]; n=1298, d=487 | 93.6 [92.2, 95.0]; n=1129, d=477 | 93.9 [93.1, 94.6]; n=3364, d=600 |
| kv/test keys | N1 | 1.7 [1.0, 2.4]; n=1025, d=528 | 1.6 [0.9, 2.5]; n=1041, d=480 | 1.2 [0.6, 1.9]; n=1298, d=487 | 1.4 [0.7, 2.1]; n=1129, d=477 | 1.5 [1.1, 1.9]; n=3364, d=600 |
| kv/test keys | N2a | 4.1 [3.0, 5.2]; n=1025, d=528 | 4.9 [3.6, 6.2]; n=1041, d=480 | 8.4 [7.0, 10.0]; n=1298, d=487 | 9.0 [7.5, 10.7]; n=1129, d=477 | 6.0 [5.3, 6.8]; n=3364, d=600 |
| kv/test keys | N2b-answer | 15.6 [13.4, 17.8]; n=1025, d=528 | 18.0 [15.8, 20.3]; n=1041, d=480 | 15.9 [13.9, 18.1]; n=1298, d=487 | 12.9 [11.0, 14.8]; n=1129, d=477 | 16.5 [15.2, 17.7]; n=3364, d=600 |
| kv/test keys | N2b-all | 15.6 [13.4, 17.8]; n=1025, d=528 | 18.0 [15.8, 20.3]; n=1041, d=480 | 15.9 [13.9, 18.1]; n=1298, d=487 | 12.9 [11.0, 14.8]; n=1129, d=477 | 16.5 [15.2, 17.7]; n=3364, d=600 |
| kv/test keys | N3 | 1.7 [1.0, 2.4]; n=1025, d=528 | 1.6 [0.9, 2.5]; n=1041, d=480 | 1.2 [0.6, 1.9]; n=1298, d=487 | 1.4 [0.7, 2.1]; n=1129, d=477 | 1.5 [1.1, 1.9]; n=3364, d=600 |
| kv/test keys | B1 follows | 97.1 [96.0, 98.3]; n=835, d=477 | 96.9 [95.5, 98.0]; n=804, d=435 | 96.8 [95.7, 97.8]; n=1000, d=461 | 96.8 [95.6, 97.8]; n=901, d=447 | 96.9 [96.3, 97.6]; n=2639, d=598 |
| kv/test keys | B1 unchanged | 0.2 [0.0, 0.6]; n=835, d=477 | 0.1 [0.0, 0.4]; n=804, d=435 | 0.2 [0.0, 0.5]; n=1000, d=461 | 0.2 [0.0, 0.6]; n=901, d=447 | 0.2 [0.0, 0.4]; n=2639, d=598 |
| kv/test keys | B1 other | 2.6 [1.6, 3.7]; n=835, d=477 | 3.0 [1.8, 4.3]; n=804, d=435 | 3.0 [2.0, 4.0]; n=1000, d=461 | 3.0 [2.0, 4.1]; n=901, d=447 | 2.9 [2.3, 3.5]; n=2639, d=598 |
| delayed2h/seen pairs | N0 | 95.1 [91.1, 98.9]; n=103, d=54 | 96.6 [93.2, 99.4]; n=175, d=92 | 95.8 [94.5, 97.1]; n=977, d=471 | 95.7 [94.5, 97.0]; n=964, d=471 | 95.9 [94.8, 97.0]; n=1255, d=600 |
| delayed2h/seen pairs | N1 | 0.0 [0.0, 0.0]; n=103, d=54 | 2.3 [0.5, 4.9]; n=175, d=92 | 2.6 [1.6, 3.5]; n=977, d=471 | 2.6 [1.6, 3.5]; n=964, d=471 | 2.3 [1.5, 3.2]; n=1255, d=600 |
| delayed2h/seen pairs | N2a | 51.5 [42.4, 60.8]; n=103, d=54 | 53.1 [44.8, 61.2]; n=175, d=92 | 50.7 [47.4, 53.9]; n=977, d=471 | 51.3 [48.2, 54.7]; n=964, d=471 | 51.1 [48.3, 53.9]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-answer | 10.7 [4.9, 16.7]; n=103, d=54 | 9.1 [5.1, 13.5]; n=175, d=92 | 7.2 [5.6, 8.8]; n=977, d=471 | 7.3 [5.7, 8.9]; n=964, d=471 | 7.7 [6.3, 9.1]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-all | 10.7 [4.9, 16.7]; n=103, d=54 | 9.1 [5.1, 13.5]; n=175, d=92 | 7.2 [5.6, 8.8]; n=977, d=471 | 7.3 [5.7, 8.9]; n=964, d=471 | 7.7 [6.3, 9.1]; n=1255, d=600 |
| delayed2h/seen pairs | N3 | 0.0 [0.0, 0.0]; n=103, d=54 | 2.3 [0.5, 4.9]; n=175, d=92 | 2.6 [1.6, 3.5]; n=977, d=471 | 2.6 [1.6, 3.5]; n=964, d=471 | 2.3 [1.5, 3.2]; n=1255, d=600 |
| delayed2h/seen pairs | B1 follows | NE: 98.0 [95.0, 100.0]; n=98, d=53 | 97.0 [94.6, 99.3]; n=169, d=92 | 97.9 [96.9, 98.8]; n=936, d=471 | 97.9 [97.0, 98.8]; n=923, d=471 | 97.8 [96.8, 98.6]; n=1203, d=600 |
| delayed2h/seen pairs | B1 unchanged | NE: 0.0 [0.0, 0.0]; n=98, d=53 | 0.0 [0.0, 0.0]; n=169, d=92 | 0.2 [0.0, 0.5]; n=936, d=471 | 0.1 [0.0, 0.3]; n=923, d=471 | 0.2 [0.0, 0.4]; n=1203, d=600 |
| delayed2h/seen pairs | B1 other | NE: 2.0 [0.0, 5.0]; n=98, d=53 | 3.0 [0.7, 5.4]; n=169, d=92 | 1.9 [1.1, 2.9]; n=936, d=471 | 2.0 [1.1, 3.0]; n=923, d=471 | 2.1 [1.3, 3.0]; n=1203, d=600 |
| delayed2h/seen pairs | B2 follows | NE: 44.9 [32.9, 56.6]; n=69, d=36 | 34.6 [27.0, 41.9]; n=130, d=69 | 42.6 [38.8, 46.1]; n=693, d=343 | 42.5 [38.6, 46.1]; n=682, d=342 | 41.6 [38.4, 44.7]; n=892, d=433 |
| delayed2h/seen pairs | B2 unchanged | NE: 26.1 [16.2, 39.0]; n=69, d=36 | 30.8 [23.1, 39.7]; n=130, d=69 | 26.3 [23.1, 29.9]; n=693, d=343 | 26.5 [23.2, 30.1]; n=682, d=342 | 26.9 [24.2, 30.0]; n=892, d=433 |
| delayed2h/seen pairs | B2 other | NE: 29.0 [17.3, 40.5]; n=69, d=36 | 34.6 [27.1, 41.7]; n=130, d=69 | 31.2 [27.7, 34.2]; n=693, d=343 | 30.9 [27.4, 34.2]; n=682, d=342 | 31.5 [28.5, 34.3]; n=892, d=433 |
| delayed2h/seen pairs | B3 follows | NE: 98.0 [95.0, 100.0]; n=98, d=53 | 97.0 [94.6, 99.3]; n=169, d=92 | 97.9 [96.9, 98.8]; n=936, d=471 | 97.9 [97.0, 98.8]; n=923, d=471 | 97.8 [96.8, 98.6]; n=1203, d=600 |
| delayed2h/seen pairs | B3 unchanged | NE: 0.0 [0.0, 0.0]; n=98, d=53 | 0.0 [0.0, 0.0]; n=169, d=92 | 0.2 [0.0, 0.5]; n=936, d=471 | 0.1 [0.0, 0.3]; n=923, d=471 | 0.2 [0.0, 0.4]; n=1203, d=600 |
| delayed2h/seen pairs | B3 other | NE: 2.0 [0.0, 5.0]; n=98, d=53 | 3.0 [0.7, 5.4]; n=169, d=92 | 1.9 [1.1, 2.9]; n=936, d=471 | 2.0 [1.1, 3.0]; n=923, d=471 | 2.1 [1.3, 3.0]; n=1203, d=600 |
| delayed2h/new pairs | N0 | NE: 98.0 [95.0, 100.0]; n=99, d=52 | 94.3 [91.3, 97.4]; n=176, d=91 | 96.0 [94.8, 97.3]; n=979, d=471 | 96.1 [94.8, 97.3]; n=964, d=470 | 95.9 [94.9, 97.1]; n=1254, d=600 |
| delayed2h/new pairs | N1 | NE: 1.0 [0.0, 3.5]; n=99, d=52 | 4.5 [1.8, 7.7]; n=176, d=91 | 2.6 [1.5, 3.7]; n=979, d=471 | 2.5 [1.6, 3.5]; n=964, d=470 | 2.7 [1.8, 3.7]; n=1254, d=600 |
| delayed2h/new pairs | N2a | NE: 52.5 [43.2, 62.4]; n=99, d=52 | 54.5 [46.5, 62.0]; n=176, d=91 | 55.9 [52.7, 59.0]; n=979, d=471 | 56.4 [53.3, 59.7]; n=964, d=470 | 55.4 [52.6, 58.2]; n=1254, d=600 |
| delayed2h/new pairs | N2b-answer | NE: 12.1 [6.3, 18.5]; n=99, d=52 | 10.2 [6.1, 15.0]; n=176, d=91 | 9.1 [7.5, 10.9]; n=979, d=471 | 9.1 [7.5, 11.0]; n=964, d=470 | 9.5 [8.1, 11.0]; n=1254, d=600 |
| delayed2h/new pairs | N2b-all | NE: 12.1 [6.3, 18.5]; n=99, d=52 | 10.2 [6.1, 15.0]; n=176, d=91 | 9.1 [7.5, 10.9]; n=979, d=471 | 9.1 [7.5, 11.0]; n=964, d=470 | 9.5 [8.1, 11.0]; n=1254, d=600 |
| delayed2h/new pairs | N3 | NE: 1.0 [0.0, 3.5]; n=99, d=52 | 4.5 [1.8, 7.7]; n=176, d=91 | 2.6 [1.5, 3.7]; n=979, d=471 | 2.5 [1.6, 3.5]; n=964, d=470 | 2.7 [1.8, 3.7]; n=1254, d=600 |
| delayed2h/new pairs | B1 follows | NE: 96.9 [92.0, 100.0]; n=97, d=52 | 97.6 [95.3, 99.4]; n=166, d=91 | 97.8 [96.8, 98.6]; n=940, d=471 | 97.7 [96.8, 98.6]; n=926, d=470 | 97.7 [96.8, 98.5]; n=1203, d=600 |
| delayed2h/new pairs | B1 unchanged | NE: 1.0 [0.0, 3.5]; n=97, d=52 | 0.6 [0.0, 2.1]; n=166, d=91 | 0.2 [0.0, 0.6]; n=940, d=471 | 0.2 [0.0, 0.6]; n=926, d=470 | 0.3 [0.1, 0.7]; n=1203, d=600 |
| delayed2h/new pairs | B1 other | NE: 2.1 [0.0, 5.3]; n=97, d=52 | 1.8 [0.0, 3.9]; n=166, d=91 | 2.0 [1.2, 2.9]; n=940, d=471 | 2.1 [1.2, 3.0]; n=926, d=470 | 2.0 [1.3, 2.8]; n=1203, d=600 |
| delayed2h/new pairs | B2 follows | NE: 35.4 [24.4, 45.5]; n=65, d=34 | 39.7 [30.4, 48.4]; n=131, d=70 | 41.1 [37.6, 44.6]; n=722, d=358 | 40.9 [37.4, 44.2]; n=709, d=355 | 40.5 [37.4, 43.5]; n=918, d=449 |
| delayed2h/new pairs | B2 unchanged | NE: 41.5 [29.2, 55.4]; n=65, d=34 | 34.4 [26.6, 43.4]; n=131, d=70 | 27.1 [24.1, 30.2]; n=722, d=358 | 27.1 [24.0, 30.2]; n=709, d=355 | 29.2 [26.4, 32.1]; n=918, d=449 |
| delayed2h/new pairs | B2 other | NE: 23.1 [13.0, 32.8]; n=65, d=34 | 26.0 [19.2, 32.1]; n=131, d=70 | 31.7 [28.5, 35.4]; n=722, d=358 | 32.0 [28.6, 35.8]; n=709, d=355 | 30.3 [27.5, 33.4]; n=918, d=449 |
| delayed2h/new pairs | B3 follows | NE: 96.9 [92.0, 100.0]; n=97, d=52 | 97.6 [95.3, 99.4]; n=166, d=91 | 97.8 [96.8, 98.6]; n=940, d=471 | 97.7 [96.8, 98.6]; n=926, d=470 | 97.7 [96.8, 98.5]; n=1203, d=600 |
| delayed2h/new pairs | B3 unchanged | NE: 1.0 [0.0, 3.5]; n=97, d=52 | 0.6 [0.0, 2.1]; n=166, d=91 | 0.2 [0.0, 0.6]; n=940, d=471 | 0.2 [0.0, 0.6]; n=926, d=470 | 0.3 [0.1, 0.7]; n=1203, d=600 |
| delayed2h/new pairs | B3 other | NE: 2.1 [0.0, 5.3]; n=97, d=52 | 1.8 [0.0, 3.9]; n=166, d=91 | 2.0 [1.2, 2.9]; n=940, d=471 | 2.1 [1.2, 3.0]; n=926, d=470 | 2.0 [1.3, 2.8]; n=1203, d=600 |
| delayed2h/never-seen bytes | N0 | 95.8 [91.2, 99.2]; n=119, d=55 | 95.3 [92.2, 98.1]; n=192, d=97 | 96.2 [95.0, 97.3]; n=953, d=463 | 96.4 [95.2, 97.5]; n=936, d=463 | 96.0 [94.9, 97.1]; n=1264, d=600 |
| delayed2h/never-seen bytes | N1 | 1.7 [0.0, 4.3]; n=119, d=55 | 2.6 [0.5, 5.1]; n=192, d=97 | 2.3 [1.4, 3.3]; n=953, d=463 | 2.1 [1.3, 3.1]; n=936, d=463 | 2.3 [1.5, 3.2]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2a | 44.5 [34.5, 55.5]; n=119, d=55 | 45.3 [38.1, 52.3]; n=192, d=97 | 45.4 [42.3, 48.8]; n=953, d=463 | 45.5 [42.4, 49.0]; n=936, d=463 | 45.3 [42.7, 48.2]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-answer | 2.5 [0.0, 5.3]; n=119, d=55 | 10.4 [6.5, 14.7]; n=192, d=97 | 5.9 [4.4, 7.2]; n=953, d=463 | 5.9 [4.4, 7.3]; n=936, d=463 | 6.2 [5.0, 7.5]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-all | 2.5 [0.0, 5.3]; n=119, d=55 | 10.4 [6.5, 14.7]; n=192, d=97 | 5.9 [4.4, 7.2]; n=953, d=463 | 5.9 [4.4, 7.3]; n=936, d=463 | 6.2 [5.0, 7.5]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3 | 1.7 [0.0, 4.3]; n=119, d=55 | 2.6 [0.5, 5.1]; n=192, d=97 | 2.3 [1.4, 3.3]; n=953, d=463 | 2.1 [1.3, 3.1]; n=936, d=463 | 2.3 [1.5, 3.2]; n=1264, d=600 |
| delayed2h/never-seen bytes | B1 follows | 98.2 [95.2, 100.0]; n=114, d=55 | 98.9 [97.1, 100.0]; n=183, d=96 | 97.2 [95.9, 98.2]; n=917, d=463 | 97.2 [96.0, 98.2]; n=902, d=463 | 97.5 [96.5, 98.4]; n=1214, d=600 |
| delayed2h/never-seen bytes | B1 unchanged | 0.0 [0.0, 0.0]; n=114, d=55 | 0.0 [0.0, 0.0]; n=183, d=96 | 0.0 [0.0, 0.0]; n=917, d=463 | 0.0 [0.0, 0.0]; n=902, d=463 | 0.0 [0.0, 0.0]; n=1214, d=600 |
| delayed2h/never-seen bytes | B1 other | 1.8 [0.0, 4.8]; n=114, d=55 | 1.1 [0.0, 2.9]; n=183, d=96 | 2.8 [1.8, 4.1]; n=917, d=463 | 2.8 [1.8, 4.0]; n=902, d=463 | 2.5 [1.6, 3.5]; n=1214, d=600 |
| delayed2h/never-seen bytes | B2 follows | NE: 44.2 [34.2, 54.4]; n=86, d=39 | 45.2 [36.5, 53.2]; n=135, d=70 | 41.9 [38.5, 45.3]; n=700, d=353 | 41.6 [38.2, 45.0]; n=688, d=351 | 42.6 [39.6, 45.6]; n=921, d=451 |
| delayed2h/never-seen bytes | B2 unchanged | NE: 33.7 [24.0, 44.0]; n=86, d=39 | 20.7 [13.9, 28.6]; n=135, d=70 | 26.7 [23.5, 30.0]; n=700, d=353 | 26.9 [23.7, 30.1]; n=688, d=351 | 26.5 [23.9, 29.4]; n=921, d=451 |
| delayed2h/never-seen bytes | B2 other | NE: 22.1 [13.5, 30.7]; n=86, d=39 | 34.1 [26.9, 41.6]; n=135, d=70 | 31.4 [27.8, 34.8]; n=700, d=353 | 31.5 [28.0, 34.8]; n=688, d=351 | 30.9 [28.0, 33.7]; n=921, d=451 |
| delayed2h/never-seen bytes | B3 follows | 98.2 [95.2, 100.0]; n=114, d=55 | 98.9 [97.1, 100.0]; n=183, d=96 | 97.2 [95.9, 98.2]; n=917, d=463 | 97.2 [96.0, 98.2]; n=902, d=463 | 97.5 [96.5, 98.4]; n=1214, d=600 |
| delayed2h/never-seen bytes | B3 unchanged | 0.0 [0.0, 0.0]; n=114, d=55 | 0.0 [0.0, 0.0]; n=183, d=96 | 0.0 [0.0, 0.0]; n=917, d=463 | 0.0 [0.0, 0.0]; n=902, d=463 | 0.0 [0.0, 0.0]; n=1214, d=600 |
| delayed2h/never-seen bytes | B3 other | 1.8 [0.0, 4.8]; n=114, d=55 | 1.1 [0.0, 2.9]; n=183, d=96 | 2.8 [1.8, 4.1]; n=917, d=463 | 2.8 [1.8, 4.0]; n=902, d=463 | 2.5 [1.6, 3.5]; n=1214, d=600 |

## candidate-s0

There is no reader; all buckets are distance strata only.

Checkpoint: `/home/joe/hsn-experiments/p2.10-split/runs/loc_a13/hold_f8ru20nw_ag_ng2_cutC_cells64_18k_gsum-s0-1dc7391b32fe/exp593/step_00018000.pt`. Status: **complete**.

| Task / keys | Condition | Near | Mid | Far | Far-strict | All |
| --- | --- | --- | --- | --- | --- | --- |
| kv/training keys | N0 | 99.8 [99.5, 100.0]; n=1062, d=536 | 99.5 [99.0, 99.9]; n=998, d=479 | 98.7 [98.1, 99.3]; n=1264, d=485 | 99.0 [98.4, 99.5]; n=1082, d=475 | 99.3 [99.0, 99.6]; n=3324, d=600 |
| kv/training keys | N1 | 18.5 [16.3, 20.5]; n=1062, d=536 | 21.4 [19.1, 24.0]; n=998, d=479 | 47.5 [45.6, 49.5]; n=1264, d=485 | 46.3 [44.1, 48.5]; n=1082, d=475 | 30.4 [29.3, 31.6]; n=3324, d=600 |
| kv/training keys | N2a | 10.9 [9.1, 12.7]; n=1062, d=536 | 14.8 [12.9, 16.9]; n=998, d=479 | 43.8 [42.0, 45.6]; n=1264, d=485 | 44.5 [42.4, 46.6]; n=1082, d=475 | 24.6 [23.8, 25.4]; n=3324, d=600 |
| kv/training keys | N2b-answer | 18.2 [16.0, 20.2]; n=1062, d=536 | 21.1 [18.8, 23.8]; n=998, d=479 | 47.5 [45.7, 49.5]; n=1264, d=485 | 46.4 [44.2, 48.6]; n=1082, d=475 | 30.2 [29.1, 31.4]; n=3324, d=600 |
| kv/training keys | N2b-all | 18.2 [16.0, 20.2]; n=1062, d=536 | 21.1 [18.8, 23.8]; n=998, d=479 | 47.5 [45.7, 49.5]; n=1264, d=485 | 46.4 [44.2, 48.6]; n=1082, d=475 | 30.2 [29.1, 31.4]; n=3324, d=600 |
| kv/training keys | N3-fast-kept | 18.5 [16.2, 20.4]; n=1062, d=536 | 21.4 [19.1, 24.0]; n=998, d=479 | 47.5 [45.6, 49.5]; n=1264, d=485 | 46.3 [44.1, 48.5]; n=1082, d=475 | 30.4 [29.3, 31.5]; n=3324, d=600 |
| kv/training keys | N3-fast-zero | 15.8 [13.6, 17.9]; n=1062, d=536 | 14.5 [12.4, 16.9]; n=998, d=479 | 17.0 [15.2, 18.9]; n=1264, d=485 | 15.2 [13.5, 17.0]; n=1082, d=475 | 15.9 [14.6, 17.1]; n=3324, d=600 |
| kv/training keys | B1 follows | 99.9 [99.7, 100.0]; n=892, d=488 | 99.5 [99.1, 99.9]; n=849, d=451 | 99.0 [98.4, 99.6]; n=1037, d=458 | 99.1 [98.5, 99.7]; n=907, d=444 | 99.5 [99.2, 99.7]; n=2778, d=589 |
| kv/training keys | B1 unchanged | 0.0 [0.0, 0.0]; n=892, d=488 | 0.0 [0.0, 0.0]; n=849, d=451 | 0.0 [0.0, 0.0]; n=1037, d=458 | 0.0 [0.0, 0.0]; n=907, d=444 | 0.0 [0.0, 0.0]; n=2778, d=589 |
| kv/training keys | B1 other | 0.1 [0.0, 0.3]; n=892, d=488 | 0.5 [0.1, 0.9]; n=849, d=451 | 1.0 [0.4, 1.6]; n=1037, d=458 | 0.9 [0.3, 1.5]; n=907, d=444 | 0.5 [0.3, 0.8]; n=2778, d=589 |
| kv/dev keys | N0 | 99.1 [98.4, 99.6]; n=1071, d=531 | 98.2 [97.4, 98.9]; n=1011, d=476 | 97.1 [96.1, 98.0]; n=1263, d=477 | 97.2 [96.0, 98.1]; n=1095, d=468 | 98.1 [97.6, 98.5]; n=3345, d=600 |
| kv/dev keys | N1 | 20.1 [17.7, 22.5]; n=1071, d=531 | 22.0 [19.5, 24.5]; n=1011, d=476 | 45.4 [43.5, 47.6]; n=1263, d=477 | 43.8 [41.9, 46.0]; n=1095, d=468 | 30.2 [29.1, 31.5]; n=3345, d=600 |
| kv/dev keys | N2a | 15.2 [13.3, 17.2]; n=1071, d=531 | 17.5 [15.2, 19.8]; n=1011, d=476 | 43.2 [41.3, 45.3]; n=1263, d=477 | 42.4 [40.5, 44.5]; n=1095, d=468 | 26.5 [25.5, 27.5]; n=3345, d=600 |
| kv/dev keys | N2b-answer | 20.1 [17.7, 22.4]; n=1071, d=531 | 22.0 [19.5, 24.5]; n=1011, d=476 | 45.3 [43.4, 47.5]; n=1263, d=477 | 43.7 [41.8, 46.0]; n=1095, d=468 | 30.2 [29.0, 31.5]; n=3345, d=600 |
| kv/dev keys | N2b-all | 20.1 [17.7, 22.4]; n=1071, d=531 | 22.0 [19.5, 24.5]; n=1011, d=476 | 45.3 [43.4, 47.5]; n=1263, d=477 | 43.7 [41.8, 46.0]; n=1095, d=468 | 30.2 [29.0, 31.5]; n=3345, d=600 |
| kv/dev keys | N3-fast-kept | 20.1 [17.7, 22.5]; n=1071, d=531 | 22.0 [19.5, 24.5]; n=1011, d=476 | 45.4 [43.5, 47.6]; n=1263, d=477 | 43.8 [41.9, 46.0]; n=1095, d=468 | 30.2 [29.1, 31.5]; n=3345, d=600 |
| kv/dev keys | N3-fast-zero | 16.5 [14.3, 18.7]; n=1071, d=531 | 15.5 [13.4, 17.8]; n=1011, d=476 | 14.6 [12.6, 16.5]; n=1263, d=477 | 12.2 [10.4, 14.0]; n=1095, d=468 | 15.5 [14.2, 16.8]; n=3345, d=600 |
| kv/dev keys | B1 follows | 99.1 [98.4, 99.7]; n=890, d=490 | 99.0 [98.3, 99.6]; n=838, d=441 | 98.4 [97.6, 99.1]; n=1050, d=454 | 98.4 [97.6, 99.1]; n=933, d=437 | 98.8 [98.4, 99.2]; n=2778, d=595 |
| kv/dev keys | B1 unchanged | 0.0 [0.0, 0.0]; n=890, d=490 | 0.0 [0.0, 0.0]; n=838, d=441 | 0.1 [0.0, 0.4]; n=1050, d=454 | 0.0 [0.0, 0.0]; n=933, d=437 | 0.0 [0.0, 0.1]; n=2778, d=595 |
| kv/dev keys | B1 other | 0.9 [0.3, 1.6]; n=890, d=490 | 1.0 [0.4, 1.7]; n=838, d=441 | 1.5 [0.8, 2.3]; n=1050, d=454 | 1.6 [0.9, 2.4]; n=933, d=437 | 1.2 [0.8, 1.6]; n=2778, d=595 |
| kv/test keys | N0 | 99.8 [99.5, 100.0]; n=1025, d=528 | 99.3 [98.8, 99.7]; n=1041, d=480 | 97.3 [96.3, 98.2]; n=1298, d=487 | 97.3 [96.3, 98.3]; n=1129, d=477 | 98.7 [98.3, 99.1]; n=3364, d=600 |
| kv/test keys | N1 | 18.8 [16.6, 21.0]; n=1025, d=528 | 23.2 [20.8, 25.5]; n=1041, d=480 | 46.7 [44.6, 48.8]; n=1298, d=487 | 44.6 [42.6, 46.8]; n=1129, d=477 | 30.9 [29.7, 32.2]; n=3364, d=600 |
| kv/test keys | N2a | 12.5 [10.8, 14.4]; n=1025, d=528 | 18.5 [16.3, 20.8]; n=1041, d=480 | 42.8 [41.1, 44.7]; n=1298, d=487 | 42.2 [40.3, 44.3]; n=1129, d=477 | 26.1 [25.2, 27.0]; n=3364, d=600 |
| kv/test keys | N2b-answer | 19.2 [17.1, 21.5]; n=1025, d=528 | 23.0 [20.6, 25.3]; n=1041, d=480 | 46.6 [44.5, 48.8]; n=1298, d=487 | 44.6 [42.6, 46.9]; n=1129, d=477 | 30.9 [29.8, 32.3]; n=3364, d=600 |
| kv/test keys | N2b-all | 19.2 [17.1, 21.5]; n=1025, d=528 | 23.0 [20.6, 25.3]; n=1041, d=480 | 46.6 [44.5, 48.8]; n=1298, d=487 | 44.6 [42.6, 46.9]; n=1129, d=477 | 30.9 [29.8, 32.3]; n=3364, d=600 |
| kv/test keys | N3-fast-kept | 18.8 [16.6, 21.0]; n=1025, d=528 | 23.2 [20.8, 25.5]; n=1041, d=480 | 46.7 [44.6, 48.8]; n=1298, d=487 | 44.6 [42.6, 46.8]; n=1129, d=477 | 30.9 [29.7, 32.2]; n=3364, d=600 |
| kv/test keys | N3-fast-zero | 15.6 [13.4, 17.8]; n=1025, d=528 | 17.9 [15.7, 20.2]; n=1041, d=480 | 17.2 [15.1, 19.3]; n=1298, d=487 | 14.3 [12.4, 16.3]; n=1129, d=477 | 16.9 [15.6, 18.2]; n=3364, d=600 |
| kv/test keys | B1 follows | 99.5 [99.1, 99.9]; n=863, d=482 | 99.1 [98.4, 99.6]; n=850, d=444 | 98.8 [98.0, 99.3]; n=1042, d=463 | 98.6 [97.8, 99.3]; n=938, d=450 | 99.1 [98.7, 99.4]; n=2755, d=598 |
| kv/test keys | B1 unchanged | 0.0 [0.0, 0.0]; n=863, d=482 | 0.0 [0.0, 0.0]; n=850, d=444 | 0.0 [0.0, 0.0]; n=1042, d=463 | 0.0 [0.0, 0.0]; n=938, d=450 | 0.0 [0.0, 0.0]; n=2755, d=598 |
| kv/test keys | B1 other | 0.5 [0.1, 0.9]; n=863, d=482 | 0.9 [0.4, 1.6]; n=850, d=444 | 1.2 [0.7, 2.0]; n=1042, d=463 | 1.4 [0.7, 2.2]; n=938, d=450 | 0.9 [0.6, 1.3]; n=2755, d=598 |
| delayed2h/seen pairs | N0 | 100.0 [100.0, 100.0]; n=103, d=54 | 99.4 [98.1, 100.0]; n=175, d=92 | 99.7 [99.3, 100.0]; n=977, d=471 | 99.7 [99.3, 100.0]; n=964, d=471 | 99.7 [99.4, 99.9]; n=1255, d=600 |
| delayed2h/seen pairs | N1 | 45.6 [39.5, 53.6]; n=103, d=54 | 54.9 [48.6, 61.4]; n=175, d=92 | 53.3 [50.9, 55.9]; n=977, d=471 | 53.8 [51.4, 56.6]; n=964, d=471 | 52.9 [50.9, 55.2]; n=1255, d=600 |
| delayed2h/seen pairs | N2a | 43.7 [37.4, 51.4]; n=103, d=54 | 48.0 [42.6, 54.0]; n=175, d=92 | 48.2 [46.2, 50.5]; n=977, d=471 | 48.9 [46.7, 51.2]; n=964, d=471 | 47.8 [46.0, 49.8]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-answer | 51.5 [44.8, 59.5]; n=103, d=54 | 52.6 [46.8, 59.1]; n=175, d=92 | 52.8 [50.6, 55.3]; n=977, d=471 | 53.5 [51.2, 56.0]; n=964, d=471 | 52.7 [50.5, 54.8]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-all | 51.5 [44.8, 59.5]; n=103, d=54 | 52.6 [46.8, 59.1]; n=175, d=92 | 52.8 [50.6, 55.3]; n=977, d=471 | 53.5 [51.2, 56.0]; n=964, d=471 | 52.7 [50.5, 54.8]; n=1255, d=600 |
| delayed2h/seen pairs | N3-fast-kept | 46.6 [40.2, 54.9]; n=103, d=54 | 53.1 [47.6, 59.5]; n=175, d=92 | 51.8 [49.4, 54.4]; n=977, d=471 | 52.5 [50.0, 55.1]; n=964, d=471 | 51.6 [49.6, 53.8]; n=1255, d=600 |
| delayed2h/seen pairs | N3-fast-zero | 2.9 [0.0, 5.9]; n=103, d=54 | 2.3 [0.5, 4.6]; n=175, d=92 | 2.0 [1.2, 2.9]; n=977, d=471 | 2.1 [1.2, 2.9]; n=964, d=471 | 2.2 [1.4, 3.0]; n=1255, d=600 |
| delayed2h/seen pairs | B1 follows | 100.0 [100.0, 100.0]; n=103, d=54 | 99.4 [98.2, 100.0]; n=174, d=92 | 99.7 [99.3, 100.0]; n=974, d=471 | 99.7 [99.3, 100.0]; n=961, d=471 | 99.7 [99.3, 99.9]; n=1251, d=600 |
| delayed2h/seen pairs | B1 unchanged | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=174, d=92 | 0.0 [0.0, 0.0]; n=974, d=471 | 0.0 [0.0, 0.0]; n=961, d=471 | 0.0 [0.0, 0.0]; n=1251, d=600 |
| delayed2h/seen pairs | B1 other | 0.0 [0.0, 0.0]; n=103, d=54 | 0.6 [0.0, 1.8]; n=174, d=92 | 0.3 [0.0, 0.7]; n=974, d=471 | 0.3 [0.0, 0.7]; n=961, d=471 | 0.3 [0.1, 0.7]; n=1251, d=600 |
| delayed2h/seen pairs | B2 follows | NE: 53.4 [42.4, 62.1]; n=73, d=37 | 43.7 [36.8, 51.0]; n=135, d=69 | 38.6 [35.6, 41.6]; n=723, d=343 | 38.3 [35.4, 41.3]; n=712, d=342 | 40.5 [38.0, 43.0]; n=931, d=433 |
| delayed2h/seen pairs | B2 unchanged | NE: 39.7 [32.4, 48.1]; n=73, d=37 | 48.9 [41.8, 56.4]; n=135, d=69 | 50.8 [47.9, 53.9]; n=723, d=343 | 51.4 [48.5, 54.4]; n=712, d=342 | 49.6 [47.0, 52.1]; n=931, d=433 |
| delayed2h/seen pairs | B2 other | NE: 6.8 [1.1, 14.9]; n=73, d=37 | 7.4 [3.3, 11.8]; n=135, d=69 | 10.7 [8.4, 12.8]; n=723, d=343 | 10.3 [8.1, 12.3]; n=712, d=342 | 9.9 [8.1, 11.8]; n=931, d=433 |
| delayed2h/seen pairs | B3 follows | 100.0 [100.0, 100.0]; n=103, d=54 | 99.4 [98.2, 100.0]; n=174, d=92 | 99.7 [99.3, 100.0]; n=974, d=471 | 99.7 [99.3, 100.0]; n=961, d=471 | 99.7 [99.3, 99.9]; n=1251, d=600 |
| delayed2h/seen pairs | B3 unchanged | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=174, d=92 | 0.0 [0.0, 0.0]; n=974, d=471 | 0.0 [0.0, 0.0]; n=961, d=471 | 0.0 [0.0, 0.0]; n=1251, d=600 |
| delayed2h/seen pairs | B3 other | 0.0 [0.0, 0.0]; n=103, d=54 | 0.6 [0.0, 1.8]; n=174, d=92 | 0.3 [0.0, 0.7]; n=974, d=471 | 0.3 [0.0, 0.7]; n=961, d=471 | 0.3 [0.1, 0.7]; n=1251, d=600 |
| delayed2h/new pairs | N0 | NE: 100.0 [100.0, 100.0]; n=99, d=52 | 100.0 [100.0, 100.0]; n=176, d=91 | 99.1 [98.5, 99.6]; n=979, d=471 | 99.1 [98.4, 99.6]; n=964, d=470 | 99.3 [98.8, 99.7]; n=1254, d=600 |
| delayed2h/new pairs | N1 | NE: 49.5 [42.4, 57.6]; n=99, d=52 | 50.0 [44.1, 56.5]; n=176, d=91 | 52.7 [50.2, 55.4]; n=979, d=471 | 53.3 [50.8, 56.1]; n=964, d=470 | 52.1 [50.0, 54.2]; n=1254, d=600 |
| delayed2h/new pairs | N2a | NE: 45.5 [39.1, 53.1]; n=99, d=52 | 47.7 [42.2, 53.7]; n=176, d=91 | 48.1 [46.0, 50.5]; n=979, d=471 | 48.7 [46.6, 51.0]; n=964, d=470 | 47.8 [46.0, 49.9]; n=1254, d=600 |
| delayed2h/new pairs | N2b-answer | NE: 52.5 [45.3, 60.3]; n=99, d=52 | 52.8 [46.5, 60.1]; n=176, d=91 | 53.6 [51.2, 56.4]; n=979, d=471 | 54.3 [51.7, 57.0]; n=964, d=470 | 53.4 [51.2, 55.8]; n=1254, d=600 |
| delayed2h/new pairs | N2b-all | NE: 52.5 [45.3, 60.3]; n=99, d=52 | 52.8 [46.5, 60.1]; n=176, d=91 | 53.6 [51.2, 56.4]; n=979, d=471 | 54.3 [51.7, 57.0]; n=964, d=470 | 53.4 [51.2, 55.8]; n=1254, d=600 |
| delayed2h/new pairs | N3-fast-kept | NE: 48.5 [41.5, 56.6]; n=99, d=52 | 50.6 [44.5, 57.1]; n=176, d=91 | 51.4 [49.2, 53.9]; n=979, d=471 | 52.0 [49.8, 54.5]; n=964, d=470 | 51.0 [49.1, 53.3]; n=1254, d=600 |
| delayed2h/new pairs | N3-fast-zero | NE: 3.0 [0.0, 6.6]; n=99, d=52 | 0.6 [0.0, 1.8]; n=176, d=91 | 1.8 [1.0, 2.7]; n=979, d=471 | 1.9 [1.0, 2.8]; n=964, d=470 | 1.8 [1.1, 2.5]; n=1254, d=600 |
| delayed2h/new pairs | B1 follows | NE: 99.0 [96.5, 100.0]; n=99, d=52 | 99.4 [98.1, 100.0]; n=176, d=91 | 99.6 [99.2, 99.9]; n=970, d=471 | 99.6 [99.1, 99.9]; n=955, d=470 | 99.5 [99.1, 99.8]; n=1245, d=600 |
| delayed2h/new pairs | B1 unchanged | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=970, d=471 | 0.0 [0.0, 0.0]; n=955, d=470 | 0.0 [0.0, 0.0]; n=1245, d=600 |
| delayed2h/new pairs | B1 other | NE: 1.0 [0.0, 3.5]; n=99, d=52 | 0.6 [0.0, 1.9]; n=176, d=91 | 0.4 [0.1, 0.8]; n=970, d=471 | 0.4 [0.1, 0.9]; n=955, d=470 | 0.5 [0.2, 0.9]; n=1245, d=600 |
| delayed2h/new pairs | B2 follows | NE: 55.2 [46.3, 62.2]; n=67, d=34 | 45.0 [38.4, 51.3]; n=140, d=70 | 41.7 [38.6, 44.4]; n=749, d=358 | 41.1 [37.9, 44.0]; n=735, d=355 | 43.1 [40.6, 45.5]; n=956, d=449 |
| delayed2h/new pairs | B2 unchanged | NE: 41.8 [35.2, 50.0]; n=67, d=34 | 47.9 [41.3, 54.7]; n=140, d=70 | 51.0 [48.4, 53.9]; n=749, d=358 | 51.4 [48.7, 54.3]; n=735, d=355 | 49.9 [47.6, 52.4]; n=956, d=449 |
| delayed2h/new pairs | B2 other | NE: 3.0 [0.0, 7.4]; n=67, d=34 | 7.1 [3.3, 11.9]; n=140, d=70 | 7.3 [5.5, 9.0]; n=749, d=358 | 7.5 [5.6, 9.2]; n=735, d=355 | 7.0 [5.5, 8.5]; n=956, d=449 |
| delayed2h/new pairs | B3 follows | NE: 99.0 [96.5, 100.0]; n=99, d=52 | 99.4 [98.1, 100.0]; n=176, d=91 | 99.6 [99.2, 99.9]; n=970, d=471 | 99.6 [99.1, 99.9]; n=955, d=470 | 99.5 [99.1, 99.8]; n=1245, d=600 |
| delayed2h/new pairs | B3 unchanged | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=970, d=471 | 0.0 [0.0, 0.0]; n=955, d=470 | 0.0 [0.0, 0.0]; n=1245, d=600 |
| delayed2h/new pairs | B3 other | NE: 1.0 [0.0, 3.5]; n=99, d=52 | 0.6 [0.0, 1.9]; n=176, d=91 | 0.4 [0.1, 0.8]; n=970, d=471 | 0.4 [0.1, 0.9]; n=955, d=470 | 0.5 [0.2, 0.9]; n=1245, d=600 |
| delayed2h/never-seen bytes | N0 | 99.2 [97.2, 100.0]; n=119, d=55 | 100.0 [100.0, 100.0]; n=192, d=97 | 98.0 [97.1, 98.8]; n=953, d=463 | 98.1 [97.2, 98.9]; n=936, d=463 | 98.4 [97.7, 99.1]; n=1264, d=600 |
| delayed2h/never-seen bytes | N1 | 43.7 [37.6, 51.6]; n=119, d=55 | 47.4 [42.9, 52.5]; n=192, d=97 | 51.1 [48.9, 53.3]; n=953, d=463 | 51.7 [49.4, 54.0]; n=936, d=463 | 49.8 [47.9, 51.9]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2a | 42.0 [36.2, 49.5]; n=119, d=55 | 45.3 [41.2, 50.0]; n=192, d=97 | 48.6 [46.6, 50.8]; n=953, d=463 | 49.3 [47.2, 51.4]; n=936, d=463 | 47.5 [45.7, 49.4]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-answer | 42.9 [37.2, 50.0]; n=119, d=55 | 49.0 [44.2, 54.0]; n=192, d=97 | 50.7 [48.5, 52.9]; n=953, d=463 | 51.4 [49.2, 53.7]; n=936, d=463 | 49.7 [47.8, 51.7]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-all | 42.9 [37.2, 50.0]; n=119, d=55 | 49.0 [44.2, 54.0]; n=192, d=97 | 50.7 [48.5, 52.9]; n=953, d=463 | 51.4 [49.2, 53.7]; n=936, d=463 | 49.7 [47.8, 51.7]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3-fast-kept | 42.0 [36.2, 49.5]; n=119, d=55 | 46.4 [42.0, 51.1]; n=192, d=97 | 49.6 [47.6, 51.8]; n=953, d=463 | 50.3 [48.1, 52.5]; n=936, d=463 | 48.4 [46.6, 50.4]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3-fast-zero | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.1 [0.0, 0.3]; n=953, d=463 | 0.1 [0.0, 0.3]; n=936, d=463 | 0.1 [0.0, 0.2]; n=1264, d=600 |
| delayed2h/never-seen bytes | B1 follows | 98.3 [95.7, 100.0]; n=118, d=55 | 99.5 [98.3, 100.0]; n=192, d=97 | 99.5 [98.9, 99.9]; n=934, d=463 | 99.5 [98.9, 99.9]; n=918, d=463 | 99.4 [98.9, 99.8]; n=1244, d=600 |
| delayed2h/never-seen bytes | B1 unchanged | 0.0 [0.0, 0.0]; n=118, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.0 [0.0, 0.0]; n=934, d=463 | 0.0 [0.0, 0.0]; n=918, d=463 | 0.0 [0.0, 0.0]; n=1244, d=600 |
| delayed2h/never-seen bytes | B1 other | 1.7 [0.0, 4.3]; n=118, d=55 | 0.5 [0.0, 1.7]; n=192, d=97 | 0.5 [0.1, 1.1]; n=934, d=463 | 0.5 [0.1, 1.1]; n=918, d=463 | 0.6 [0.2, 1.1]; n=1244, d=600 |
| delayed2h/never-seen bytes | B2 follows | NE: 47.2 [38.5, 55.7]; n=89, d=39 | 39.4 [32.6, 45.9]; n=142, d=71 | 38.7 [35.7, 41.8]; n=718, d=353 | 38.3 [35.2, 41.3]; n=705, d=351 | 39.6 [37.1, 42.2]; n=949, d=451 |
| delayed2h/never-seen bytes | B2 unchanged | NE: 43.8 [36.4, 52.7]; n=89, d=39 | 49.3 [43.8, 55.9]; n=142, d=71 | 51.7 [49.0, 54.7]; n=718, d=353 | 52.1 [49.4, 54.9]; n=705, d=351 | 50.6 [48.2, 53.2]; n=949, d=451 |
| delayed2h/never-seen bytes | B2 other | NE: 9.0 [3.4, 14.9]; n=89, d=39 | 11.3 [6.2, 16.3]; n=142, d=71 | 9.6 [7.5, 11.8]; n=718, d=353 | 9.6 [7.6, 11.8]; n=705, d=351 | 9.8 [7.9, 11.7]; n=949, d=451 |
| delayed2h/never-seen bytes | B3 follows | 98.3 [95.7, 100.0]; n=118, d=55 | 99.5 [98.3, 100.0]; n=192, d=97 | 99.5 [98.9, 99.9]; n=934, d=463 | 99.5 [98.9, 99.9]; n=918, d=463 | 99.4 [98.9, 99.8]; n=1244, d=600 |
| delayed2h/never-seen bytes | B3 unchanged | 0.0 [0.0, 0.0]; n=118, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.0 [0.0, 0.0]; n=934, d=463 | 0.0 [0.0, 0.0]; n=918, d=463 | 0.0 [0.0, 0.0]; n=1244, d=600 |
| delayed2h/never-seen bytes | B3 other | 1.7 [0.0, 4.3]; n=118, d=55 | 0.5 [0.0, 1.7]; n=192, d=97 | 0.5 [0.1, 1.1]; n=934, d=463 | 0.5 [0.1, 1.1]; n=918, d=463 | 0.6 [0.2, 1.1]; n=1244, d=600 |

## candidate-s1

There is no reader; all buckets are distance strata only.

Checkpoint: `/home/joe/hsn-experiments/p2.10-split/runs/loc_a13/hold_f8ru20nw_ag_ng2_cutC_cells64_18k_gsum-s1-35eac7593071/exp594/step_00018000.pt`. Status: **complete**.

| Task / keys | Condition | Near | Mid | Far | Far-strict | All |
| --- | --- | --- | --- | --- | --- | --- |
| kv/training keys | N0 | 99.2 [98.5, 99.6]; n=1062, d=536 | 98.8 [98.1, 99.4]; n=998, d=479 | 97.8 [97.0, 98.6]; n=1264, d=485 | 98.3 [97.6, 99.0]; n=1082, d=475 | 98.5 [98.1, 98.9]; n=3324, d=600 |
| kv/training keys | N1 | 16.3 [14.2, 18.3]; n=1062, d=536 | 20.3 [18.1, 22.8]; n=998, d=479 | 47.0 [45.1, 48.9]; n=1264, d=485 | 46.0 [43.9, 48.1]; n=1082, d=475 | 29.2 [28.2, 30.2]; n=3324, d=600 |
| kv/training keys | N2a | 4.7 [3.5, 6.0]; n=1062, d=536 | 10.7 [8.9, 12.5]; n=998, d=479 | 40.4 [38.7, 42.2]; n=1264, d=485 | 42.4 [40.4, 44.4]; n=1082, d=475 | 20.1 [19.4, 20.7]; n=3324, d=600 |
| kv/training keys | N2b-answer | 15.9 [13.8, 17.8]; n=1062, d=536 | 19.6 [17.5, 22.1]; n=998, d=479 | 46.7 [44.8, 48.6]; n=1264, d=485 | 46.4 [44.2, 48.6]; n=1082, d=475 | 28.7 [27.7, 29.8]; n=3324, d=600 |
| kv/training keys | N2b-all | 15.9 [13.8, 17.8]; n=1062, d=536 | 19.6 [17.5, 22.1]; n=998, d=479 | 46.7 [44.8, 48.6]; n=1264, d=485 | 46.4 [44.2, 48.6]; n=1082, d=475 | 28.7 [27.7, 29.8]; n=3324, d=600 |
| kv/training keys | N3-fast-kept | 17.7 [15.6, 19.9]; n=1062, d=536 | 21.0 [18.7, 23.5]; n=998, d=479 | 47.2 [45.3, 49.2]; n=1264, d=485 | 46.1 [43.9, 48.4]; n=1082, d=475 | 29.9 [28.9, 31.1]; n=3324, d=600 |
| kv/training keys | N3-fast-zero | 15.8 [13.6, 17.9]; n=1062, d=536 | 14.5 [12.4, 16.9]; n=998, d=479 | 17.0 [15.2, 18.9]; n=1264, d=485 | 15.2 [13.5, 17.0]; n=1082, d=475 | 15.9 [14.6, 17.1]; n=3324, d=600 |
| kv/training keys | B1 follows | 99.0 [98.3, 99.6]; n=886, d=484 | 99.4 [98.8, 99.9]; n=842, d=447 | 97.9 [97.0, 98.8]; n=1023, d=458 | 98.0 [97.0, 99.0]; n=900, d=443 | 98.7 [98.3, 99.1]; n=2751, d=589 |
| kv/training keys | B1 unchanged | 0.1 [0.0, 0.3]; n=886, d=484 | 0.1 [0.0, 0.4]; n=842, d=447 | 0.1 [0.0, 0.3]; n=1023, d=458 | 0.0 [0.0, 0.0]; n=900, d=443 | 0.1 [0.0, 0.3]; n=2751, d=589 |
| kv/training keys | B1 other | 0.9 [0.3, 1.6]; n=886, d=484 | 0.5 [0.1, 1.0]; n=842, d=447 | 2.0 [1.2, 2.9]; n=1023, d=458 | 2.0 [1.0, 3.0]; n=900, d=443 | 1.2 [0.8, 1.6]; n=2751, d=589 |
| kv/dev keys | N0 | 98.4 [97.6, 99.2]; n=1071, d=531 | 97.8 [96.8, 98.7]; n=1011, d=476 | 96.0 [94.9, 97.0]; n=1263, d=477 | 96.0 [94.8, 97.1]; n=1095, d=468 | 97.3 [96.7, 97.9]; n=3345, d=600 |
| kv/dev keys | N1 | 19.8 [17.4, 22.2]; n=1071, d=531 | 21.7 [19.2, 24.2]; n=1011, d=476 | 45.0 [43.1, 47.1]; n=1263, d=477 | 43.5 [41.6, 45.7]; n=1095, d=468 | 29.9 [28.7, 31.2]; n=3345, d=600 |
| kv/dev keys | N2a | 4.4 [3.2, 5.6]; n=1071, d=531 | 8.9 [7.2, 10.6]; n=1011, d=476 | 38.5 [37.0, 40.2]; n=1263, d=477 | 40.4 [38.5, 42.4]; n=1095, d=468 | 18.6 [18.1, 19.2]; n=3345, d=600 |
| kv/dev keys | N2b-answer | 18.8 [16.5, 21.1]; n=1071, d=531 | 21.0 [18.6, 23.5]; n=1011, d=476 | 45.0 [43.1, 47.2]; n=1263, d=477 | 43.6 [41.7, 45.8]; n=1095, d=468 | 29.3 [28.3, 30.6]; n=3345, d=600 |
| kv/dev keys | N2b-all | 18.8 [16.5, 21.1]; n=1071, d=531 | 21.0 [18.6, 23.5]; n=1011, d=476 | 45.0 [43.1, 47.2]; n=1263, d=477 | 43.6 [41.7, 45.8]; n=1095, d=468 | 29.3 [28.3, 30.6]; n=3345, d=600 |
| kv/dev keys | N3-fast-kept | 19.9 [17.5, 22.3]; n=1071, d=531 | 22.0 [19.5, 24.5]; n=1011, d=476 | 45.2 [43.3, 47.4]; n=1263, d=477 | 43.7 [41.8, 45.9]; n=1095, d=468 | 30.1 [28.9, 31.4]; n=3345, d=600 |
| kv/dev keys | N3-fast-zero | 16.5 [14.3, 18.7]; n=1071, d=531 | 15.5 [13.4, 17.8]; n=1011, d=476 | 14.6 [12.6, 16.5]; n=1263, d=477 | 12.2 [10.4, 14.0]; n=1095, d=468 | 15.5 [14.2, 16.8]; n=3345, d=600 |
| kv/dev keys | B1 follows | 99.3 [98.8, 99.8]; n=879, d=486 | 98.0 [96.9, 98.9]; n=832, d=434 | 98.2 [97.2, 99.0]; n=1028, d=452 | 98.1 [97.2, 99.0]; n=917, d=433 | 98.5 [98.0, 98.9]; n=2739, d=594 |
| kv/dev keys | B1 unchanged | 0.0 [0.0, 0.0]; n=879, d=486 | 0.0 [0.0, 0.0]; n=832, d=434 | 0.2 [0.0, 0.5]; n=1028, d=452 | 0.1 [0.0, 0.3]; n=917, d=433 | 0.1 [0.0, 0.2]; n=2739, d=594 |
| kv/dev keys | B1 other | 0.7 [0.2, 1.2]; n=879, d=486 | 2.0 [1.1, 3.1]; n=832, d=434 | 1.7 [0.8, 2.6]; n=1028, d=452 | 1.7 [0.9, 2.7]; n=917, d=433 | 1.5 [1.0, 1.9]; n=2739, d=594 |
| kv/test keys | N0 | 99.3 [98.8, 99.8]; n=1025, d=528 | 98.4 [97.6, 99.1]; n=1041, d=480 | 97.7 [96.8, 98.5]; n=1298, d=487 | 98.0 [97.1, 98.7]; n=1129, d=477 | 98.4 [97.9, 98.8]; n=3364, d=600 |
| kv/test keys | N1 | 18.3 [16.1, 20.5]; n=1025, d=528 | 22.5 [20.1, 24.8]; n=1041, d=480 | 46.0 [44.0, 48.1]; n=1298, d=487 | 43.9 [42.0, 46.0]; n=1129, d=477 | 30.3 [29.2, 31.6]; n=3364, d=600 |
| kv/test keys | N2a | 5.1 [3.8, 6.3]; n=1025, d=528 | 9.3 [7.7, 11.2]; n=1041, d=480 | 38.9 [37.3, 40.6]; n=1298, d=487 | 40.6 [38.7, 42.5]; n=1129, d=477 | 19.4 [18.9, 20.1]; n=3364, d=600 |
| kv/test keys | N2b-answer | 17.9 [15.7, 19.9]; n=1025, d=528 | 21.8 [19.5, 24.2]; n=1041, d=480 | 45.6 [43.6, 47.7]; n=1298, d=487 | 43.8 [41.9, 46.0]; n=1129, d=477 | 29.8 [28.7, 31.0]; n=3364, d=600 |
| kv/test keys | N2b-all | 17.9 [15.7, 19.9]; n=1025, d=528 | 21.8 [19.5, 24.2]; n=1041, d=480 | 45.6 [43.6, 47.7]; n=1298, d=487 | 43.8 [41.9, 46.0]; n=1129, d=477 | 29.8 [28.7, 31.0]; n=3364, d=600 |
| kv/test keys | N3-fast-kept | 18.7 [16.4, 20.9]; n=1025, d=528 | 22.8 [20.4, 25.1]; n=1041, d=480 | 46.2 [44.2, 48.3]; n=1298, d=487 | 44.2 [42.2, 46.3]; n=1129, d=477 | 30.6 [29.4, 31.9]; n=3364, d=600 |
| kv/test keys | N3-fast-zero | 15.6 [13.4, 17.8]; n=1025, d=528 | 17.9 [15.7, 20.2]; n=1041, d=480 | 17.2 [15.1, 19.3]; n=1298, d=487 | 14.3 [12.4, 16.3]; n=1129, d=477 | 16.9 [15.6, 18.2]; n=3364, d=600 |
| kv/test keys | B1 follows | 99.5 [99.0, 99.9]; n=858, d=481 | 98.6 [97.8, 99.4]; n=842, d=439 | 98.3 [97.4, 99.1]; n=1048, d=464 | 98.2 [97.2, 99.1]; n=944, d=449 | 98.8 [98.3, 99.2]; n=2748, d=598 |
| kv/test keys | B1 unchanged | 0.0 [0.0, 0.0]; n=858, d=481 | 0.0 [0.0, 0.0]; n=842, d=439 | 0.1 [0.0, 0.3]; n=1048, d=464 | 0.0 [0.0, 0.0]; n=944, d=449 | 0.0 [0.0, 0.1]; n=2748, d=598 |
| kv/test keys | B1 other | 0.5 [0.1, 1.0]; n=858, d=481 | 1.4 [0.6, 2.2]; n=842, d=439 | 1.6 [0.9, 2.5]; n=1048, d=464 | 1.8 [0.9, 2.8]; n=944, d=449 | 1.2 [0.8, 1.6]; n=2748, d=598 |
| delayed2h/seen pairs | N0 | 99.0 [96.7, 100.0]; n=103, d=54 | 98.9 [97.2, 100.0]; n=175, d=92 | 99.6 [99.2, 99.9]; n=977, d=471 | 99.6 [99.2, 99.9]; n=964, d=471 | 99.4 [99.0, 99.8]; n=1255, d=600 |
| delayed2h/seen pairs | N1 | 51.5 [44.4, 60.0]; n=103, d=54 | 54.9 [48.9, 61.6]; n=175, d=92 | 54.7 [52.1, 57.3]; n=977, d=471 | 55.2 [52.7, 57.8]; n=964, d=471 | 54.4 [52.2, 56.5]; n=1255, d=600 |
| delayed2h/seen pairs | N2a | 45.6 [38.6, 53.9]; n=103, d=54 | 50.3 [44.4, 57.2]; n=175, d=92 | 49.4 [47.3, 51.8]; n=977, d=471 | 50.1 [47.9, 52.5]; n=964, d=471 | 49.2 [47.2, 51.3]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-answer | 53.4 [45.6, 62.6]; n=103, d=54 | 58.3 [52.0, 65.1]; n=175, d=92 | 56.4 [53.8, 59.1]; n=977, d=471 | 57.1 [54.5, 59.8]; n=964, d=471 | 56.4 [54.2, 58.8]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-all | 53.4 [45.6, 62.6]; n=103, d=54 | 58.3 [52.0, 65.1]; n=175, d=92 | 56.4 [53.8, 59.1]; n=977, d=471 | 57.1 [54.5, 59.8]; n=964, d=471 | 56.4 [54.2, 58.8]; n=1255, d=600 |
| delayed2h/seen pairs | N3-fast-kept | 53.4 [45.5, 63.4]; n=103, d=54 | 54.9 [48.9, 61.6]; n=175, d=92 | 55.6 [53.2, 58.1]; n=977, d=471 | 56.1 [53.7, 58.6]; n=964, d=471 | 55.3 [53.2, 57.5]; n=1255, d=600 |
| delayed2h/seen pairs | N3-fast-zero | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=977, d=471 | 0.0 [0.0, 0.0]; n=964, d=471 | 0.0 [0.0, 0.0]; n=1255, d=600 |
| delayed2h/seen pairs | B1 follows | 100.0 [100.0, 100.0]; n=102, d=54 | 99.4 [98.1, 100.0]; n=173, d=92 | 99.6 [99.1, 99.9]; n=973, d=471 | 99.6 [99.1, 99.9]; n=960, d=471 | 99.6 [99.2, 99.9]; n=1248, d=600 |
| delayed2h/seen pairs | B1 unchanged | 0.0 [0.0, 0.0]; n=102, d=54 | 0.0 [0.0, 0.0]; n=173, d=92 | 0.1 [0.0, 0.4]; n=973, d=471 | 0.1 [0.0, 0.4]; n=960, d=471 | 0.1 [0.0, 0.3]; n=1248, d=600 |
| delayed2h/seen pairs | B1 other | 0.0 [0.0, 0.0]; n=102, d=54 | 0.6 [0.0, 1.9]; n=173, d=92 | 0.3 [0.0, 0.7]; n=973, d=471 | 0.3 [0.0, 0.7]; n=960, d=471 | 0.3 [0.1, 0.6]; n=1248, d=600 |
| delayed2h/seen pairs | B2 follows | NE: 34.7 [23.5, 45.1]; n=72, d=37 | 32.8 [26.3, 39.1]; n=134, d=69 | 36.7 [33.8, 39.5]; n=722, d=343 | 36.1 [33.2, 38.9]; n=711, d=342 | 36.0 [33.3, 38.4]; n=928, d=433 |
| delayed2h/seen pairs | B2 unchanged | NE: 43.1 [35.1, 52.2]; n=72, d=37 | 51.5 [44.7, 58.8]; n=134, d=69 | 49.2 [46.6, 52.0]; n=722, d=343 | 49.9 [47.2, 52.8]; n=711, d=342 | 49.0 [46.7, 51.4]; n=928, d=433 |
| delayed2h/seen pairs | B2 other | NE: 22.2 [13.0, 32.3]; n=72, d=37 | 15.7 [10.5, 21.1]; n=134, d=69 | 14.1 [11.6, 16.5]; n=722, d=343 | 13.9 [11.5, 16.2]; n=711, d=342 | 15.0 [12.8, 17.2]; n=928, d=433 |
| delayed2h/seen pairs | B3 follows | 100.0 [100.0, 100.0]; n=102, d=54 | 99.4 [98.1, 100.0]; n=173, d=92 | 99.6 [99.1, 99.9]; n=973, d=471 | 99.6 [99.1, 99.9]; n=960, d=471 | 99.6 [99.2, 99.9]; n=1248, d=600 |
| delayed2h/seen pairs | B3 unchanged | 0.0 [0.0, 0.0]; n=102, d=54 | 0.0 [0.0, 0.0]; n=173, d=92 | 0.1 [0.0, 0.4]; n=973, d=471 | 0.1 [0.0, 0.4]; n=960, d=471 | 0.1 [0.0, 0.3]; n=1248, d=600 |
| delayed2h/seen pairs | B3 other | 0.0 [0.0, 0.0]; n=102, d=54 | 0.6 [0.0, 1.9]; n=173, d=92 | 0.3 [0.0, 0.7]; n=973, d=471 | 0.3 [0.0, 0.7]; n=960, d=471 | 0.3 [0.1, 0.6]; n=1248, d=600 |
| delayed2h/new pairs | N0 | NE: 99.0 [96.4, 100.0]; n=99, d=52 | 98.3 [96.0, 100.0]; n=176, d=91 | 99.6 [99.1, 99.9]; n=979, d=471 | 99.6 [99.1, 99.9]; n=964, d=470 | 99.4 [98.9, 99.8]; n=1254, d=600 |
| delayed2h/new pairs | N1 | NE: 47.5 [41.0, 56.1]; n=99, d=52 | 52.3 [46.3, 58.9]; n=176, d=91 | 53.7 [51.4, 56.3]; n=979, d=471 | 54.4 [52.0, 57.0]; n=964, d=470 | 53.0 [50.9, 55.3]; n=1254, d=600 |
| delayed2h/new pairs | N2a | NE: 46.5 [39.8, 54.7]; n=99, d=52 | 47.7 [42.2, 53.7]; n=176, d=91 | 48.9 [46.8, 51.4]; n=979, d=471 | 49.5 [47.3, 51.9]; n=964, d=470 | 48.6 [46.6, 50.7]; n=1254, d=600 |
| delayed2h/new pairs | N2b-answer | NE: 52.5 [45.5, 61.2]; n=99, d=52 | 54.5 [48.3, 61.6]; n=176, d=91 | 55.3 [52.9, 58.0]; n=979, d=471 | 55.9 [53.5, 58.7]; n=964, d=470 | 54.9 [52.8, 57.2]; n=1254, d=600 |
| delayed2h/new pairs | N2b-all | NE: 52.5 [45.5, 61.2]; n=99, d=52 | 54.5 [48.3, 61.6]; n=176, d=91 | 55.3 [52.9, 58.0]; n=979, d=471 | 55.9 [53.5, 58.7]; n=964, d=470 | 54.9 [52.8, 57.2]; n=1254, d=600 |
| delayed2h/new pairs | N3-fast-kept | NE: 48.5 [41.9, 57.3]; n=99, d=52 | 52.8 [46.4, 59.9]; n=176, d=91 | 54.5 [52.2, 57.1]; n=979, d=471 | 55.1 [52.7, 57.8]; n=964, d=470 | 53.8 [51.6, 56.0]; n=1254, d=600 |
| delayed2h/new pairs | N3-fast-zero | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=979, d=471 | 0.0 [0.0, 0.0]; n=964, d=470 | 0.0 [0.0, 0.0]; n=1254, d=600 |
| delayed2h/new pairs | B1 follows | NE: 100.0 [100.0, 100.0]; n=98, d=51 | 98.8 [97.2, 100.0]; n=173, d=91 | 99.0 [98.3, 99.6]; n=975, d=471 | 99.0 [98.3, 99.6]; n=960, d=470 | 99.0 [98.5, 99.5]; n=1246, d=600 |
| delayed2h/new pairs | B1 unchanged | NE: 0.0 [0.0, 0.0]; n=98, d=51 | 0.0 [0.0, 0.0]; n=173, d=91 | 0.0 [0.0, 0.0]; n=975, d=471 | 0.0 [0.0, 0.0]; n=960, d=470 | 0.0 [0.0, 0.0]; n=1246, d=600 |
| delayed2h/new pairs | B1 other | NE: 0.0 [0.0, 0.0]; n=98, d=51 | 1.2 [0.0, 2.8]; n=173, d=91 | 1.0 [0.4, 1.7]; n=975, d=471 | 1.0 [0.4, 1.7]; n=960, d=470 | 1.0 [0.5, 1.5]; n=1246, d=600 |
| delayed2h/new pairs | B2 follows | NE: 41.8 [29.3, 53.8]; n=67, d=34 | 35.5 [27.2, 42.6]; n=138, d=70 | 35.5 [32.2, 38.5]; n=753, d=358 | 34.8 [31.5, 37.7]; n=739, d=355 | 35.9 [33.1, 38.6]; n=958, d=449 |
| delayed2h/new pairs | B2 unchanged | NE: 41.8 [33.3, 52.5]; n=67, d=34 | 47.1 [40.8, 53.8]; n=138, d=70 | 51.1 [48.5, 53.9]; n=753, d=358 | 51.6 [48.9, 54.3]; n=739, d=355 | 49.9 [47.6, 52.2]; n=958, d=449 |
| delayed2h/new pairs | B2 other | NE: 16.4 [8.2, 25.7]; n=67, d=34 | 17.4 [11.5, 24.8]; n=138, d=70 | 13.4 [11.1, 15.8]; n=753, d=358 | 13.7 [11.3, 16.1]; n=739, d=355 | 14.2 [12.2, 16.4]; n=958, d=449 |
| delayed2h/new pairs | B3 follows | NE: 100.0 [100.0, 100.0]; n=98, d=51 | 98.8 [97.2, 100.0]; n=173, d=91 | 99.0 [98.3, 99.6]; n=975, d=471 | 99.0 [98.3, 99.6]; n=960, d=470 | 99.0 [98.5, 99.5]; n=1246, d=600 |
| delayed2h/new pairs | B3 unchanged | NE: 0.0 [0.0, 0.0]; n=98, d=51 | 0.0 [0.0, 0.0]; n=173, d=91 | 0.0 [0.0, 0.0]; n=975, d=471 | 0.0 [0.0, 0.0]; n=960, d=470 | 0.0 [0.0, 0.0]; n=1246, d=600 |
| delayed2h/new pairs | B3 other | NE: 0.0 [0.0, 0.0]; n=98, d=51 | 1.2 [0.0, 2.8]; n=173, d=91 | 1.0 [0.4, 1.7]; n=975, d=471 | 1.0 [0.4, 1.7]; n=960, d=470 | 1.0 [0.5, 1.5]; n=1246, d=600 |
| delayed2h/never-seen bytes | N0 | 99.2 [97.2, 100.0]; n=119, d=55 | 100.0 [100.0, 100.0]; n=192, d=97 | 99.2 [98.5, 99.7]; n=953, d=463 | 99.1 [98.5, 99.7]; n=936, d=463 | 99.3 [98.7, 99.8]; n=1264, d=600 |
| delayed2h/never-seen bytes | N1 | 44.5 [38.1, 53.0]; n=119, d=55 | 51.0 [45.8, 56.9]; n=192, d=97 | 52.9 [50.5, 55.4]; n=953, d=463 | 53.6 [51.3, 56.1]; n=936, d=463 | 51.8 [49.8, 54.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2a | 42.0 [36.2, 49.5]; n=119, d=55 | 46.4 [42.0, 51.1]; n=192, d=97 | 49.0 [46.9, 51.2]; n=953, d=463 | 49.7 [47.5, 51.9]; n=936, d=463 | 47.9 [46.2, 50.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-answer | 47.1 [40.1, 56.0]; n=119, d=55 | 53.6 [48.1, 59.2]; n=192, d=97 | 53.2 [50.8, 55.8]; n=953, d=463 | 54.0 [51.5, 56.5]; n=936, d=463 | 52.7 [50.7, 55.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-all | 47.1 [40.1, 56.0]; n=119, d=55 | 53.6 [48.1, 59.2]; n=192, d=97 | 53.2 [50.8, 55.8]; n=953, d=463 | 54.0 [51.5, 56.5]; n=936, d=463 | 52.7 [50.7, 55.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3-fast-kept | 46.2 [38.9, 55.2]; n=119, d=55 | 52.6 [47.6, 58.1]; n=192, d=97 | 53.4 [51.0, 55.8]; n=953, d=463 | 54.1 [51.6, 56.5]; n=936, d=463 | 52.6 [50.5, 54.8]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3-fast-zero | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.0 [0.0, 0.0]; n=953, d=463 | 0.0 [0.0, 0.0]; n=936, d=463 | 0.0 [0.0, 0.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | B1 follows | 97.5 [93.3, 100.0]; n=118, d=55 | 99.5 [98.3, 100.0]; n=192, d=97 | 99.6 [99.1, 99.9]; n=945, d=463 | 99.6 [99.1, 99.9]; n=928, d=463 | 99.4 [98.9, 99.8]; n=1255, d=600 |
| delayed2h/never-seen bytes | B1 unchanged | 0.0 [0.0, 0.0]; n=118, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.0 [0.0, 0.0]; n=945, d=463 | 0.0 [0.0, 0.0]; n=928, d=463 | 0.0 [0.0, 0.0]; n=1255, d=600 |
| delayed2h/never-seen bytes | B1 other | 2.5 [0.0, 6.7]; n=118, d=55 | 0.5 [0.0, 1.7]; n=192, d=97 | 0.4 [0.1, 0.9]; n=945, d=463 | 0.4 [0.1, 0.9]; n=928, d=463 | 0.6 [0.2, 1.1]; n=1255, d=600 |
| delayed2h/never-seen bytes | B2 follows | NE: 38.2 [29.7, 47.1]; n=89, d=39 | 38.7 [31.0, 46.2]; n=142, d=71 | 36.6 [33.7, 39.4]; n=727, d=353 | 36.2 [33.3, 39.0]; n=713, d=351 | 37.1 [34.5, 39.5]; n=958, d=451 |
| delayed2h/never-seen bytes | B2 unchanged | NE: 49.4 [42.0, 57.5]; n=89, d=39 | 49.3 [43.6, 55.5]; n=142, d=71 | 49.2 [46.8, 52.0]; n=727, d=353 | 49.6 [47.1, 52.4]; n=713, d=351 | 49.3 [47.1, 51.6]; n=958, d=451 |
| delayed2h/never-seen bytes | B2 other | NE: 12.4 [6.1, 18.9]; n=89, d=39 | 12.0 [6.5, 17.7]; n=142, d=71 | 14.2 [11.7, 16.8]; n=727, d=353 | 14.2 [11.7, 16.8]; n=713, d=351 | 13.7 [11.7, 15.8]; n=958, d=451 |
| delayed2h/never-seen bytes | B3 follows | 97.5 [93.3, 100.0]; n=118, d=55 | 99.5 [98.3, 100.0]; n=192, d=97 | 99.6 [99.1, 99.9]; n=945, d=463 | 99.6 [99.1, 99.9]; n=928, d=463 | 99.4 [98.9, 99.8]; n=1255, d=600 |
| delayed2h/never-seen bytes | B3 unchanged | 0.0 [0.0, 0.0]; n=118, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.0 [0.0, 0.0]; n=945, d=463 | 0.0 [0.0, 0.0]; n=928, d=463 | 0.0 [0.0, 0.0]; n=1255, d=600 |
| delayed2h/never-seen bytes | B3 other | 2.5 [0.0, 6.7]; n=118, d=55 | 0.5 [0.0, 1.7]; n=192, d=97 | 0.4 [0.1, 0.9]; n=945, d=463 | 0.4 [0.1, 0.9]; n=928, d=463 | 0.6 [0.2, 1.1]; n=1255, d=600 |

## candidate-s2

There is no reader; all buckets are distance strata only.

Checkpoint: `/home/joe/hsn-experiments/vast2/exp/gpu0/runs/loc_a13/hold_f8ru20nw_ag_ng2_cutC_cells64_18k_gsum-s2-8941ea2de1d2/exp1/step_00018000.pt`. Status: **complete**.

| Task / keys | Condition | Near | Mid | Far | Far-strict | All |
| --- | --- | --- | --- | --- | --- | --- |
| kv/training keys | N0 | 99.2 [98.7, 99.7]; n=1062, d=536 | 99.6 [99.2, 99.9]; n=998, d=479 | 97.5 [96.6, 98.3]; n=1264, d=485 | 97.6 [96.6, 98.4]; n=1082, d=475 | 98.7 [98.3, 99.1]; n=3324, d=600 |
| kv/training keys | N1 | 14.8 [12.8, 16.8]; n=1062, d=536 | 15.7 [13.6, 18.0]; n=998, d=479 | 37.3 [35.3, 39.4]; n=1264, d=485 | 38.2 [35.8, 40.4]; n=1082, d=475 | 23.6 [22.5, 24.7]; n=3324, d=600 |
| kv/training keys | N2a | 9.6 [7.8, 11.2]; n=1062, d=536 | 12.9 [11.0, 15.0]; n=998, d=479 | 39.9 [37.9, 41.9]; n=1264, d=485 | 41.5 [39.3, 43.7]; n=1082, d=475 | 22.1 [21.2, 22.9]; n=3324, d=600 |
| kv/training keys | N2b-answer | 18.4 [16.1, 20.4]; n=1062, d=536 | 20.6 [18.3, 23.2]; n=998, d=479 | 43.5 [41.5, 45.6]; n=1264, d=485 | 42.5 [40.0, 44.9]; n=1082, d=475 | 28.6 [27.4, 29.7]; n=3324, d=600 |
| kv/training keys | N2b-all | 18.4 [16.1, 20.4]; n=1062, d=536 | 20.6 [18.3, 23.2]; n=998, d=479 | 43.5 [41.5, 45.6]; n=1264, d=485 | 42.5 [40.0, 44.9]; n=1082, d=475 | 28.6 [27.4, 29.7]; n=3324, d=600 |
| kv/training keys | N3-fast-kept | 14.6 [12.6, 16.5]; n=1062, d=536 | 15.0 [13.0, 17.3]; n=998, d=479 | 35.0 [32.9, 37.1]; n=1264, d=485 | 35.7 [33.4, 38.0]; n=1082, d=475 | 22.5 [21.3, 23.6]; n=3324, d=600 |
| kv/training keys | N3-fast-zero | 15.8 [13.6, 17.9]; n=1062, d=536 | 14.5 [12.4, 16.9]; n=998, d=479 | 17.0 [15.2, 18.9]; n=1264, d=485 | 15.2 [13.5, 17.0]; n=1082, d=475 | 15.9 [14.6, 17.1]; n=3324, d=600 |
| kv/training keys | B1 follows | 99.4 [98.9, 99.9]; n=887, d=488 | 99.9 [99.6, 100.0]; n=850, d=452 | 98.5 [97.7, 99.2]; n=1022, d=458 | 98.4 [97.6, 99.2]; n=895, d=443 | 99.2 [98.9, 99.5]; n=2759, d=589 |
| kv/training keys | B1 unchanged | 0.0 [0.0, 0.0]; n=887, d=488 | 0.0 [0.0, 0.0]; n=850, d=452 | 0.1 [0.0, 0.3]; n=1022, d=458 | 0.1 [0.0, 0.3]; n=895, d=443 | 0.0 [0.0, 0.1]; n=2759, d=589 |
| kv/training keys | B1 other | 0.6 [0.1, 1.1]; n=887, d=488 | 0.1 [0.0, 0.4]; n=850, d=452 | 1.4 [0.7, 2.1]; n=1022, d=458 | 1.5 [0.7, 2.3]; n=895, d=443 | 0.7 [0.4, 1.1]; n=2759, d=589 |
| kv/dev keys | N0 | 99.0 [98.3, 99.5]; n=1071, d=531 | 97.7 [96.8, 98.7]; n=1011, d=476 | 95.9 [94.8, 97.0]; n=1263, d=477 | 96.6 [95.7, 97.7]; n=1095, d=468 | 97.4 [96.9, 97.9]; n=3345, d=600 |
| kv/dev keys | N1 | 13.4 [11.3, 15.5]; n=1071, d=531 | 14.1 [12.2, 16.3]; n=1011, d=476 | 36.1 [34.2, 38.3]; n=1263, d=477 | 37.0 [34.9, 39.3]; n=1095, d=468 | 22.2 [21.2, 23.3]; n=3345, d=600 |
| kv/dev keys | N2a | 9.3 [7.6, 11.0]; n=1071, d=531 | 12.3 [10.4, 14.2]; n=1011, d=476 | 36.8 [35.1, 38.8]; n=1263, d=477 | 38.4 [36.4, 40.6]; n=1095, d=468 | 20.6 [19.8, 21.4]; n=3345, d=600 |
| kv/dev keys | N2b-answer | 19.8 [17.4, 22.2]; n=1071, d=531 | 21.5 [19.0, 24.1]; n=1011, d=476 | 42.2 [40.1, 44.5]; n=1263, d=477 | 40.6 [38.4, 43.0]; n=1095, d=468 | 28.8 [27.6, 30.0]; n=3345, d=600 |
| kv/dev keys | N2b-all | 19.8 [17.4, 22.2]; n=1071, d=531 | 21.5 [19.0, 24.1]; n=1011, d=476 | 42.2 [40.1, 44.5]; n=1263, d=477 | 40.6 [38.4, 43.0]; n=1095, d=468 | 28.8 [27.6, 30.0]; n=3345, d=600 |
| kv/dev keys | N3-fast-kept | 12.3 [10.4, 14.3]; n=1071, d=531 | 13.9 [11.9, 16.1]; n=1011, d=476 | 31.2 [29.2, 33.5]; n=1263, d=477 | 32.0 [29.8, 34.4]; n=1095, d=468 | 19.9 [18.9, 21.1]; n=3345, d=600 |
| kv/dev keys | N3-fast-zero | 16.5 [14.3, 18.7]; n=1071, d=531 | 15.5 [13.4, 17.8]; n=1011, d=476 | 14.6 [12.6, 16.5]; n=1263, d=477 | 12.2 [10.4, 14.0]; n=1095, d=468 | 15.5 [14.2, 16.8]; n=3345, d=600 |
| kv/dev keys | B1 follows | 99.7 [99.2, 100.0]; n=887, d=489 | 99.5 [99.0, 100.0]; n=836, d=436 | 98.4 [97.6, 99.1]; n=1032, d=452 | 98.5 [97.6, 99.2]; n=925, d=437 | 99.2 [98.8, 99.5]; n=2755, d=595 |
| kv/dev keys | B1 unchanged | 0.0 [0.0, 0.0]; n=887, d=489 | 0.0 [0.0, 0.0]; n=836, d=436 | 0.0 [0.0, 0.0]; n=1032, d=452 | 0.0 [0.0, 0.0]; n=925, d=437 | 0.0 [0.0, 0.0]; n=2755, d=595 |
| kv/dev keys | B1 other | 0.3 [0.0, 0.8]; n=887, d=489 | 0.5 [0.0, 1.0]; n=836, d=436 | 1.6 [0.9, 2.4]; n=1032, d=452 | 1.5 [0.8, 2.4]; n=925, d=437 | 0.8 [0.5, 1.2]; n=2755, d=595 |
| kv/test keys | N0 | 99.0 [98.4, 99.6]; n=1025, d=528 | 99.1 [98.5, 99.6]; n=1041, d=480 | 97.1 [96.2, 98.0]; n=1298, d=487 | 97.4 [96.5, 98.3]; n=1129, d=477 | 98.3 [97.9, 98.7]; n=3364, d=600 |
| kv/test keys | N1 | 12.6 [10.8, 14.5]; n=1025, d=528 | 13.5 [11.6, 15.5]; n=1041, d=480 | 36.1 [33.8, 38.0]; n=1298, d=487 | 37.5 [35.1, 39.5]; n=1129, d=477 | 21.9 [20.8, 23.0]; n=3364, d=600 |
| kv/test keys | N2a | 9.7 [8.1, 11.3]; n=1025, d=528 | 12.8 [11.0, 14.8]; n=1041, d=480 | 36.9 [35.1, 38.8]; n=1298, d=487 | 38.5 [36.6, 40.6]; n=1129, d=477 | 21.1 [20.3, 22.1]; n=3364, d=600 |
| kv/test keys | N2b-answer | 18.3 [16.2, 20.6]; n=1025, d=528 | 22.2 [19.8, 24.6]; n=1041, d=480 | 43.1 [41.0, 45.2]; n=1298, d=487 | 41.4 [39.3, 43.7]; n=1129, d=477 | 29.1 [27.9, 30.4]; n=3364, d=600 |
| kv/test keys | N2b-all | 18.3 [16.2, 20.6]; n=1025, d=528 | 22.2 [19.8, 24.6]; n=1041, d=480 | 43.1 [41.0, 45.2]; n=1298, d=487 | 41.4 [39.3, 43.7]; n=1129, d=477 | 29.1 [27.9, 30.4]; n=3364, d=600 |
| kv/test keys | N3-fast-kept | 12.2 [10.4, 14.1]; n=1025, d=528 | 12.9 [10.9, 14.8]; n=1041, d=480 | 33.8 [31.5, 36.1]; n=1298, d=487 | 34.9 [32.5, 37.2]; n=1129, d=477 | 20.7 [19.6, 21.9]; n=3364, d=600 |
| kv/test keys | N3-fast-zero | 15.6 [13.4, 17.8]; n=1025, d=528 | 17.9 [15.7, 20.2]; n=1041, d=480 | 17.2 [15.1, 19.3]; n=1298, d=487 | 14.3 [12.4, 16.3]; n=1129, d=477 | 16.9 [15.6, 18.2]; n=3364, d=600 |
| kv/test keys | B1 follows | 99.7 [99.2, 100.0]; n=860, d=480 | 99.4 [98.9, 99.9]; n=849, d=443 | 98.4 [97.5, 99.1]; n=1042, d=462 | 98.2 [97.2, 99.0]; n=939, d=446 | 99.1 [98.7, 99.5]; n=2751, d=597 |
| kv/test keys | B1 unchanged | 0.0 [0.0, 0.0]; n=860, d=480 | 0.0 [0.0, 0.0]; n=849, d=443 | 0.0 [0.0, 0.0]; n=1042, d=462 | 0.0 [0.0, 0.0]; n=939, d=446 | 0.0 [0.0, 0.0]; n=2751, d=597 |
| kv/test keys | B1 other | 0.3 [0.0, 0.8]; n=860, d=480 | 0.6 [0.1, 1.1]; n=849, d=443 | 1.6 [0.9, 2.5]; n=1042, d=462 | 1.8 [1.0, 2.8]; n=939, d=446 | 0.9 [0.5, 1.3]; n=2751, d=597 |
| delayed2h/seen pairs | N0 | 100.0 [100.0, 100.0]; n=103, d=54 | 100.0 [100.0, 100.0]; n=175, d=92 | 99.3 [98.7, 99.8]; n=977, d=471 | 99.3 [98.7, 99.8]; n=964, d=471 | 99.4 [99.0, 99.8]; n=1255, d=600 |
| delayed2h/seen pairs | N1 | 35.0 [28.0, 43.3]; n=103, d=54 | 34.9 [28.9, 41.7]; n=175, d=92 | 34.6 [32.1, 37.4]; n=977, d=471 | 35.0 [32.4, 37.9]; n=964, d=471 | 34.7 [32.5, 37.0]; n=1255, d=600 |
| delayed2h/seen pairs | N2a | 38.8 [31.9, 46.4]; n=103, d=54 | 43.4 [37.6, 49.6]; n=175, d=92 | 44.0 [41.7, 46.6]; n=977, d=471 | 44.6 [42.2, 47.2]; n=964, d=471 | 43.5 [41.4, 45.7]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-answer | 35.9 [28.7, 43.8]; n=103, d=54 | 39.4 [33.8, 46.1]; n=175, d=92 | 39.1 [36.7, 41.7]; n=977, d=471 | 39.6 [37.2, 42.2]; n=964, d=471 | 38.9 [36.8, 41.2]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-all | 35.9 [28.7, 43.8]; n=103, d=54 | 39.4 [33.8, 46.1]; n=175, d=92 | 39.1 [36.7, 41.7]; n=977, d=471 | 39.6 [37.2, 42.2]; n=964, d=471 | 38.9 [36.8, 41.2]; n=1255, d=600 |
| delayed2h/seen pairs | N3-fast-kept | 32.0 [24.7, 40.6]; n=103, d=54 | 32.6 [26.9, 39.0]; n=175, d=92 | 31.8 [29.3, 34.7]; n=977, d=471 | 32.1 [29.6, 34.9]; n=964, d=471 | 32.0 [29.8, 34.3]; n=1255, d=600 |
| delayed2h/seen pairs | N3-fast-zero | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=977, d=471 | 0.0 [0.0, 0.0]; n=964, d=471 | 0.0 [0.0, 0.0]; n=1255, d=600 |
| delayed2h/seen pairs | B1 follows | 100.0 [100.0, 100.0]; n=103, d=54 | 100.0 [100.0, 100.0]; n=175, d=92 | 99.5 [99.0, 99.9]; n=970, d=471 | 99.5 [99.0, 99.9]; n=957, d=471 | 99.6 [99.2, 99.9]; n=1248, d=600 |
| delayed2h/seen pairs | B1 unchanged | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=970, d=471 | 0.0 [0.0, 0.0]; n=957, d=471 | 0.0 [0.0, 0.0]; n=1248, d=600 |
| delayed2h/seen pairs | B1 other | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.5 [0.1, 1.0]; n=970, d=471 | 0.5 [0.1, 1.0]; n=957, d=471 | 0.4 [0.1, 0.8]; n=1248, d=600 |
| delayed2h/seen pairs | B2 follows | NE: 50.7 [39.3, 60.3]; n=73, d=37 | 41.9 [35.1, 48.7]; n=136, d=69 | 42.8 [39.9, 45.5]; n=720, d=343 | 42.3 [39.6, 45.0]; n=709, d=342 | 43.3 [40.9, 45.6]; n=929, d=433 |
| delayed2h/seen pairs | B2 unchanged | NE: 37.0 [28.9, 46.8]; n=73, d=37 | 44.1 [37.3, 51.3]; n=136, d=69 | 45.7 [43.1, 48.3]; n=720, d=343 | 46.1 [43.4, 48.7]; n=709, d=342 | 44.8 [42.7, 47.1]; n=929, d=433 |
| delayed2h/seen pairs | B2 other | NE: 12.3 [4.4, 21.4]; n=73, d=37 | 14.0 [8.8, 19.2]; n=136, d=69 | 11.5 [9.2, 13.9]; n=720, d=343 | 11.6 [9.3, 14.0]; n=709, d=342 | 11.9 [9.8, 14.0]; n=929, d=433 |
| delayed2h/seen pairs | B3 follows | 100.0 [100.0, 100.0]; n=103, d=54 | 100.0 [100.0, 100.0]; n=175, d=92 | 99.5 [99.0, 99.9]; n=970, d=471 | 99.5 [99.0, 99.9]; n=957, d=471 | 99.6 [99.2, 99.9]; n=1248, d=600 |
| delayed2h/seen pairs | B3 unchanged | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=970, d=471 | 0.0 [0.0, 0.0]; n=957, d=471 | 0.0 [0.0, 0.0]; n=1248, d=600 |
| delayed2h/seen pairs | B3 other | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.5 [0.1, 1.0]; n=970, d=471 | 0.5 [0.1, 1.0]; n=957, d=471 | 0.4 [0.1, 0.8]; n=1248, d=600 |
| delayed2h/new pairs | N0 | NE: 99.0 [96.6, 100.0]; n=99, d=52 | 99.4 [98.0, 100.0]; n=176, d=91 | 99.0 [98.3, 99.6]; n=979, d=471 | 99.0 [98.3, 99.6]; n=964, d=470 | 99.0 [98.5, 99.5]; n=1254, d=600 |
| delayed2h/new pairs | N1 | NE: 34.3 [26.2, 42.9]; n=99, d=52 | 34.7 [29.1, 41.3]; n=176, d=91 | 36.2 [33.8, 38.8]; n=979, d=471 | 36.4 [34.0, 39.2]; n=964, d=470 | 35.8 [33.7, 38.1]; n=1254, d=600 |
| delayed2h/new pairs | N2a | NE: 39.4 [32.7, 47.7]; n=99, d=52 | 43.8 [38.1, 50.0]; n=176, d=91 | 44.9 [42.7, 47.4]; n=979, d=471 | 45.4 [43.2, 47.9]; n=964, d=470 | 44.3 [42.4, 46.4]; n=1254, d=600 |
| delayed2h/new pairs | N2b-answer | NE: 37.4 [30.1, 46.5]; n=99, d=52 | 41.5 [35.8, 47.7]; n=176, d=91 | 40.7 [38.3, 43.1]; n=979, d=471 | 41.2 [38.9, 43.6]; n=964, d=470 | 40.5 [38.5, 42.6]; n=1254, d=600 |
| delayed2h/new pairs | N2b-all | NE: 37.4 [30.1, 46.5]; n=99, d=52 | 41.5 [35.8, 47.7]; n=176, d=91 | 40.7 [38.3, 43.1]; n=979, d=471 | 41.2 [38.9, 43.6]; n=964, d=470 | 40.5 [38.5, 42.6]; n=1254, d=600 |
| delayed2h/new pairs | N3-fast-kept | NE: 31.3 [23.9, 39.7]; n=99, d=52 | 33.0 [27.4, 39.4]; n=176, d=91 | 32.9 [30.4, 35.6]; n=979, d=471 | 33.1 [30.6, 35.7]; n=964, d=470 | 32.8 [30.6, 34.9]; n=1254, d=600 |
| delayed2h/new pairs | N3-fast-zero | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=979, d=471 | 0.0 [0.0, 0.0]; n=964, d=470 | 0.0 [0.0, 0.0]; n=1254, d=600 |
| delayed2h/new pairs | B1 follows | NE: 99.0 [96.7, 100.0]; n=98, d=52 | 97.7 [95.4, 99.5]; n=175, d=91 | 98.6 [97.7, 99.3]; n=969, d=471 | 98.5 [97.7, 99.3]; n=954, d=470 | 98.5 [97.7, 99.1]; n=1242, d=600 |
| delayed2h/new pairs | B1 unchanged | NE: 0.0 [0.0, 0.0]; n=98, d=52 | 0.0 [0.0, 0.0]; n=175, d=91 | 0.1 [0.0, 0.3]; n=969, d=471 | 0.1 [0.0, 0.3]; n=954, d=470 | 0.1 [0.0, 0.3]; n=1242, d=600 |
| delayed2h/new pairs | B1 other | NE: 1.0 [0.0, 3.3]; n=98, d=52 | 2.3 [0.5, 4.6]; n=175, d=91 | 1.3 [0.7, 2.2]; n=969, d=471 | 1.4 [0.7, 2.2]; n=954, d=470 | 1.4 [0.8, 2.2]; n=1242, d=600 |
| delayed2h/new pairs | B2 follows | NE: 45.5 [34.4, 56.5]; n=66, d=34 | 46.4 [39.5, 52.6]; n=140, d=70 | 40.2 [37.4, 43.0]; n=748, d=358 | 39.9 [37.0, 42.7]; n=734, d=355 | 41.5 [39.0, 43.9]; n=954, d=449 |
| delayed2h/new pairs | B2 unchanged | NE: 43.9 [34.3, 54.4]; n=66, d=34 | 45.0 [38.7, 52.5]; n=140, d=70 | 47.5 [44.6, 50.1]; n=748, d=358 | 47.5 [44.8, 50.1]; n=734, d=355 | 46.9 [44.5, 49.3]; n=954, d=449 |
| delayed2h/new pairs | B2 other | NE: 10.6 [3.9, 19.7]; n=66, d=34 | 8.6 [4.2, 13.3]; n=140, d=70 | 12.3 [10.1, 14.5]; n=748, d=358 | 12.5 [10.3, 14.8]; n=734, d=355 | 11.6 [9.6, 13.6]; n=954, d=449 |
| delayed2h/new pairs | B3 follows | NE: 99.0 [96.7, 100.0]; n=98, d=52 | 97.7 [95.4, 99.5]; n=175, d=91 | 98.6 [97.7, 99.3]; n=969, d=471 | 98.5 [97.7, 99.3]; n=954, d=470 | 98.5 [97.7, 99.1]; n=1242, d=600 |
| delayed2h/new pairs | B3 unchanged | NE: 0.0 [0.0, 0.0]; n=98, d=52 | 0.0 [0.0, 0.0]; n=175, d=91 | 0.1 [0.0, 0.3]; n=969, d=471 | 0.1 [0.0, 0.3]; n=954, d=470 | 0.1 [0.0, 0.3]; n=1242, d=600 |
| delayed2h/new pairs | B3 other | NE: 1.0 [0.0, 3.3]; n=98, d=52 | 2.3 [0.5, 4.6]; n=175, d=91 | 1.3 [0.7, 2.2]; n=969, d=471 | 1.4 [0.7, 2.2]; n=954, d=470 | 1.4 [0.8, 2.2]; n=1242, d=600 |
| delayed2h/never-seen bytes | N0 | 99.2 [97.2, 100.0]; n=119, d=55 | 97.9 [95.8, 99.6]; n=192, d=97 | 97.8 [96.8, 98.6]; n=953, d=463 | 97.9 [96.9, 98.7]; n=936, d=463 | 97.9 [97.1, 98.6]; n=1264, d=600 |
| delayed2h/never-seen bytes | N1 | 37.0 [30.1, 44.4]; n=119, d=55 | 30.2 [24.9, 36.0]; n=192, d=97 | 40.5 [38.0, 43.1]; n=953, d=463 | 40.6 [38.1, 43.2]; n=936, d=463 | 38.6 [36.4, 41.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2a | 37.0 [30.7, 44.2]; n=119, d=55 | 40.1 [35.5, 45.3]; n=192, d=97 | 44.8 [42.6, 47.2]; n=953, d=463 | 45.5 [43.3, 47.8]; n=936, d=463 | 43.4 [41.4, 45.4]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-answer | 36.1 [29.6, 43.8]; n=119, d=55 | 38.0 [33.0, 42.9]; n=192, d=97 | 42.8 [40.6, 45.3]; n=953, d=463 | 43.5 [41.3, 45.9]; n=936, d=463 | 41.5 [39.6, 43.6]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-all | 36.1 [29.6, 43.8]; n=119, d=55 | 38.0 [33.0, 42.9]; n=192, d=97 | 42.8 [40.6, 45.3]; n=953, d=463 | 43.5 [41.3, 45.9]; n=936, d=463 | 41.5 [39.6, 43.6]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3-fast-kept | 33.6 [26.8, 41.2]; n=119, d=55 | 30.2 [24.7, 36.1]; n=192, d=97 | 38.7 [36.2, 41.3]; n=953, d=463 | 38.7 [36.2, 41.2]; n=936, d=463 | 36.9 [34.8, 39.3]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3-fast-zero | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.0 [0.0, 0.0]; n=953, d=463 | 0.0 [0.0, 0.0]; n=936, d=463 | 0.0 [0.0, 0.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | B1 follows | 98.3 [95.4, 100.0]; n=118, d=55 | 97.3 [94.7, 99.4]; n=188, d=97 | 99.4 [98.8, 99.8]; n=932, d=463 | 99.3 [98.8, 99.8]; n=916, d=462 | 98.9 [98.3, 99.4]; n=1238, d=600 |
| delayed2h/never-seen bytes | B1 unchanged | 0.0 [0.0, 0.0]; n=118, d=55 | 0.0 [0.0, 0.0]; n=188, d=97 | 0.0 [0.0, 0.0]; n=932, d=463 | 0.0 [0.0, 0.0]; n=916, d=462 | 0.0 [0.0, 0.0]; n=1238, d=600 |
| delayed2h/never-seen bytes | B1 other | 1.7 [0.0, 4.6]; n=118, d=55 | 2.7 [0.6, 5.3]; n=188, d=97 | 0.6 [0.2, 1.2]; n=932, d=463 | 0.7 [0.2, 1.2]; n=916, d=462 | 1.1 [0.6, 1.7]; n=1238, d=600 |
| delayed2h/never-seen bytes | B2 follows | NE: 53.9 [45.5, 61.4]; n=89, d=39 | 45.3 [38.2, 52.1]; n=139, d=70 | 42.3 [39.3, 45.1]; n=716, d=353 | 42.1 [39.1, 44.8]; n=703, d=351 | 43.9 [41.5, 46.1]; n=944, d=451 |
| delayed2h/never-seen bytes | B2 unchanged | NE: 37.1 [30.4, 44.7]; n=89, d=39 | 46.8 [40.2, 54.4]; n=139, d=70 | 46.9 [44.4, 49.7]; n=716, d=353 | 47.2 [44.7, 50.0]; n=703, d=351 | 46.0 [43.8, 48.5]; n=944, d=451 |
| delayed2h/never-seen bytes | B2 other | NE: 9.0 [3.3, 15.4]; n=89, d=39 | 7.9 [3.2, 12.4]; n=139, d=70 | 10.8 [8.5, 12.9]; n=716, d=353 | 10.7 [8.4, 12.8]; n=703, d=351 | 10.2 [8.1, 12.0]; n=944, d=451 |
| delayed2h/never-seen bytes | B3 follows | 98.3 [95.4, 100.0]; n=118, d=55 | 97.3 [94.7, 99.4]; n=188, d=97 | 99.4 [98.8, 99.8]; n=932, d=463 | 99.3 [98.8, 99.8]; n=916, d=462 | 98.9 [98.3, 99.4]; n=1238, d=600 |
| delayed2h/never-seen bytes | B3 unchanged | 0.0 [0.0, 0.0]; n=118, d=55 | 0.0 [0.0, 0.0]; n=188, d=97 | 0.0 [0.0, 0.0]; n=932, d=463 | 0.0 [0.0, 0.0]; n=916, d=462 | 0.0 [0.0, 0.0]; n=1238, d=600 |
| delayed2h/never-seen bytes | B3 other | 1.7 [0.0, 4.6]; n=118, d=55 | 2.7 [0.6, 5.3]; n=188, d=97 | 0.6 [0.2, 1.2]; n=932, d=463 | 0.7 [0.2, 1.2]; n=916, d=462 | 1.1 [0.6, 1.7]; n=1238, d=600 |

## candidate-s3

There is no reader; all buckets are distance strata only.

Checkpoint: `/home/joe/hsn-experiments/vast2/exp/gpu2/runs/a13_follow/hold_f8ru20nw_ag_ng2_cutC_cells64_18k_gsum-s3-a4029b7f8fbc/exp5/step_00018000.pt`. Status: **complete**.

| Task / keys | Condition | Near | Mid | Far | Far-strict | All |
| --- | --- | --- | --- | --- | --- | --- |
| kv/training keys | N0 | 99.3 [98.8, 99.8]; n=1062, d=536 | 99.1 [98.5, 99.7]; n=998, d=479 | 98.1 [97.3, 98.8]; n=1264, d=485 | 98.4 [97.6, 99.2]; n=1082, d=475 | 98.8 [98.4, 99.2]; n=3324, d=600 |
| kv/training keys | N1 | 18.0 [15.8, 19.9]; n=1062, d=536 | 20.9 [18.6, 23.5]; n=998, d=479 | 44.7 [42.7, 46.7]; n=1264, d=485 | 43.8 [41.4, 46.0]; n=1082, d=475 | 29.0 [27.9, 30.3]; n=3324, d=600 |
| kv/training keys | N2a | 13.3 [11.3, 15.1]; n=1062, d=536 | 17.3 [15.1, 19.6]; n=998, d=479 | 41.4 [39.4, 43.4]; n=1264, d=485 | 41.8 [39.6, 44.1]; n=1082, d=475 | 25.2 [24.2, 26.2]; n=3324, d=600 |
| kv/training keys | N2b-answer | 18.4 [16.2, 20.4]; n=1062, d=536 | 20.9 [18.6, 23.5]; n=998, d=479 | 45.1 [43.1, 47.2]; n=1264, d=485 | 43.9 [41.5, 46.1]; n=1082, d=475 | 29.3 [28.1, 30.6]; n=3324, d=600 |
| kv/training keys | N2b-all | 18.4 [16.2, 20.4]; n=1062, d=536 | 20.9 [18.6, 23.5]; n=998, d=479 | 45.1 [43.1, 47.2]; n=1264, d=485 | 43.9 [41.5, 46.1]; n=1082, d=475 | 29.3 [28.1, 30.6]; n=3324, d=600 |
| kv/training keys | N3-fast-kept | 17.9 [15.8, 19.9]; n=1062, d=536 | 20.8 [18.4, 23.3]; n=998, d=479 | 44.9 [42.8, 46.9]; n=1264, d=485 | 43.9 [41.6, 46.0]; n=1082, d=475 | 29.0 [27.9, 30.3]; n=3324, d=600 |
| kv/training keys | N3-fast-zero | 15.8 [13.6, 17.9]; n=1062, d=536 | 14.5 [12.4, 16.9]; n=998, d=479 | 17.0 [15.2, 18.9]; n=1264, d=485 | 15.2 [13.5, 17.0]; n=1082, d=475 | 15.9 [14.6, 17.1]; n=3324, d=600 |
| kv/training keys | B1 follows | 99.3 [98.7, 99.8]; n=889, d=487 | 99.3 [98.6, 99.8]; n=844, d=451 | 98.3 [97.5, 99.1]; n=1027, d=458 | 98.2 [97.3, 99.0]; n=900, d=444 | 98.9 [98.5, 99.3]; n=2760, d=589 |
| kv/training keys | B1 unchanged | 0.0 [0.0, 0.0]; n=889, d=487 | 0.0 [0.0, 0.0]; n=844, d=451 | 0.0 [0.0, 0.0]; n=1027, d=458 | 0.0 [0.0, 0.0]; n=900, d=444 | 0.0 [0.0, 0.0]; n=2760, d=589 |
| kv/training keys | B1 other | 0.7 [0.2, 1.3]; n=889, d=487 | 0.7 [0.2, 1.4]; n=844, d=451 | 1.7 [0.9, 2.5]; n=1027, d=458 | 1.8 [1.0, 2.7]; n=900, d=444 | 1.1 [0.7, 1.5]; n=2760, d=589 |
| kv/dev keys | N0 | 98.8 [98.1, 99.4]; n=1071, d=531 | 99.0 [98.3, 99.5]; n=1011, d=476 | 97.8 [96.9, 98.6]; n=1263, d=477 | 98.0 [97.1, 98.8]; n=1095, d=468 | 98.5 [98.0, 98.9]; n=3345, d=600 |
| kv/dev keys | N1 | 19.3 [17.0, 21.7]; n=1071, d=531 | 20.9 [18.5, 23.4]; n=1011, d=476 | 43.1 [41.0, 45.4]; n=1263, d=477 | 41.6 [39.4, 43.9]; n=1095, d=468 | 28.8 [27.6, 30.1]; n=3345, d=600 |
| kv/dev keys | N2a | 15.0 [13.1, 17.2]; n=1071, d=531 | 17.9 [15.6, 20.3]; n=1011, d=476 | 40.3 [38.2, 42.6]; n=1263, d=477 | 39.8 [37.7, 42.1]; n=1095, d=468 | 25.4 [24.3, 26.6]; n=3345, d=600 |
| kv/dev keys | N2b-answer | 20.0 [17.6, 22.3]; n=1071, d=531 | 21.5 [19.1, 24.0]; n=1011, d=476 | 43.3 [41.2, 45.5]; n=1263, d=477 | 41.7 [39.5, 44.2]; n=1095, d=468 | 29.2 [28.1, 30.6]; n=3345, d=600 |
| kv/dev keys | N2b-all | 20.0 [17.6, 22.3]; n=1071, d=531 | 21.5 [19.1, 24.0]; n=1011, d=476 | 43.3 [41.2, 45.5]; n=1263, d=477 | 41.7 [39.5, 44.2]; n=1095, d=468 | 29.2 [28.1, 30.6]; n=3345, d=600 |
| kv/dev keys | N3-fast-kept | 19.2 [16.9, 21.6]; n=1071, d=531 | 21.1 [18.7, 23.6]; n=1011, d=476 | 42.5 [40.4, 44.8]; n=1263, d=477 | 41.2 [39.1, 43.6]; n=1095, d=468 | 28.6 [27.4, 30.0]; n=3345, d=600 |
| kv/dev keys | N3-fast-zero | 16.5 [14.3, 18.7]; n=1071, d=531 | 15.5 [13.4, 17.8]; n=1011, d=476 | 14.6 [12.6, 16.5]; n=1263, d=477 | 12.2 [10.4, 14.0]; n=1095, d=468 | 15.5 [14.2, 16.8]; n=3345, d=600 |
| kv/dev keys | B1 follows | 99.4 [98.9, 99.9]; n=884, d=486 | 98.9 [98.2, 99.5]; n=845, d=439 | 98.2 [97.4, 98.9]; n=1053, d=450 | 98.4 [97.6, 99.1]; n=939, d=435 | 98.8 [98.4, 99.2]; n=2782, d=594 |
| kv/dev keys | B1 unchanged | 0.0 [0.0, 0.0]; n=884, d=486 | 0.0 [0.0, 0.0]; n=845, d=439 | 0.0 [0.0, 0.0]; n=1053, d=450 | 0.0 [0.0, 0.0]; n=939, d=435 | 0.0 [0.0, 0.0]; n=2782, d=594 |
| kv/dev keys | B1 other | 0.6 [0.1, 1.1]; n=884, d=486 | 1.1 [0.5, 1.8]; n=845, d=439 | 1.8 [1.1, 2.6]; n=1053, d=450 | 1.6 [0.9, 2.4]; n=939, d=435 | 1.2 [0.8, 1.6]; n=2782, d=594 |
| kv/test keys | N0 | 99.8 [99.5, 100.0]; n=1025, d=528 | 98.9 [98.2, 99.5]; n=1041, d=480 | 94.7 [93.4, 95.9]; n=1298, d=487 | 94.9 [93.6, 96.2]; n=1129, d=477 | 97.6 [97.0, 98.1]; n=3364, d=600 |
| kv/test keys | N1 | 18.6 [16.4, 20.8]; n=1025, d=528 | 23.0 [20.5, 25.4]; n=1041, d=480 | 44.8 [42.6, 46.8]; n=1298, d=487 | 42.9 [40.7, 45.1]; n=1129, d=477 | 30.1 [28.9, 31.4]; n=3364, d=600 |
| kv/test keys | N2a | 13.6 [11.6, 15.5]; n=1025, d=528 | 16.8 [14.7, 18.9]; n=1041, d=480 | 39.3 [37.2, 41.3]; n=1298, d=487 | 38.6 [36.5, 40.7]; n=1129, d=477 | 24.5 [23.4, 25.6]; n=3364, d=600 |
| kv/test keys | N2b-answer | 18.7 [16.5, 20.9]; n=1025, d=528 | 22.9 [20.4, 25.3]; n=1041, d=480 | 44.7 [42.5, 46.8]; n=1298, d=487 | 42.8 [40.6, 45.1]; n=1129, d=477 | 30.0 [28.8, 31.3]; n=3364, d=600 |
| kv/test keys | N2b-all | 18.7 [16.5, 20.9]; n=1025, d=528 | 22.9 [20.4, 25.3]; n=1041, d=480 | 44.7 [42.5, 46.8]; n=1298, d=487 | 42.8 [40.6, 45.1]; n=1129, d=477 | 30.0 [28.8, 31.3]; n=3364, d=600 |
| kv/test keys | N3-fast-kept | 18.6 [16.4, 20.8]; n=1025, d=528 | 22.8 [20.4, 25.2]; n=1041, d=480 | 44.6 [42.4, 46.8]; n=1298, d=487 | 42.6 [40.4, 44.9]; n=1129, d=477 | 29.9 [28.7, 31.3]; n=3364, d=600 |
| kv/test keys | N3-fast-zero | 15.6 [13.4, 17.8]; n=1025, d=528 | 17.9 [15.7, 20.2]; n=1041, d=480 | 17.2 [15.1, 19.3]; n=1298, d=487 | 14.3 [12.4, 16.3]; n=1129, d=477 | 16.9 [15.6, 18.2]; n=3364, d=600 |
| kv/test keys | B1 follows | 99.9 [99.6, 100.0]; n=863, d=482 | 98.7 [97.8, 99.4]; n=845, d=442 | 97.0 [95.9, 98.0]; n=1011, d=460 | 96.8 [95.6, 97.9]; n=912, d=445 | 98.5 [98.0, 98.9]; n=2719, d=596 |
| kv/test keys | B1 unchanged | 0.0 [0.0, 0.0]; n=863, d=482 | 0.0 [0.0, 0.0]; n=845, d=442 | 0.1 [0.0, 0.3]; n=1011, d=460 | 0.1 [0.0, 0.3]; n=912, d=445 | 0.0 [0.0, 0.1]; n=2719, d=596 |
| kv/test keys | B1 other | 0.1 [0.0, 0.4]; n=863, d=482 | 1.3 [0.6, 2.2]; n=845, d=442 | 2.9 [2.0, 3.9]; n=1011, d=460 | 3.1 [2.0, 4.2]; n=912, d=445 | 1.5 [1.1, 2.0]; n=2719, d=596 |
| delayed2h/seen pairs | N0 | 99.0 [96.9, 100.0]; n=103, d=54 | 96.0 [93.0, 98.7]; n=175, d=92 | 98.6 [97.8, 99.3]; n=977, d=471 | 98.8 [98.0, 99.4]; n=964, d=471 | 98.2 [97.5, 99.0]; n=1255, d=600 |
| delayed2h/seen pairs | N1 | 32.0 [25.0, 39.1]; n=103, d=54 | 38.9 [33.5, 44.9]; n=175, d=92 | 39.8 [37.7, 42.2]; n=977, d=471 | 40.4 [38.2, 42.8]; n=964, d=471 | 39.0 [37.3, 41.0]; n=1255, d=600 |
| delayed2h/seen pairs | N2a | 34.0 [27.6, 40.9]; n=103, d=54 | 40.6 [35.0, 46.9]; n=175, d=92 | 40.1 [38.1, 42.3]; n=977, d=471 | 40.7 [38.5, 42.9]; n=964, d=471 | 39.7 [37.8, 41.5]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-answer | 38.8 [32.5, 45.5]; n=103, d=54 | 42.9 [37.9, 48.5]; n=175, d=92 | 43.1 [40.9, 45.3]; n=977, d=471 | 43.7 [41.4, 46.0]; n=964, d=471 | 42.7 [40.8, 44.6]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-all | 38.8 [32.5, 45.5]; n=103, d=54 | 42.9 [37.9, 48.5]; n=175, d=92 | 43.1 [40.9, 45.3]; n=977, d=471 | 43.7 [41.4, 46.0]; n=964, d=471 | 42.7 [40.8, 44.6]; n=1255, d=600 |
| delayed2h/seen pairs | N3-fast-kept | 31.1 [23.6, 38.6]; n=103, d=54 | 37.1 [31.6, 43.3]; n=175, d=92 | 38.5 [36.2, 40.9]; n=977, d=471 | 39.0 [36.7, 41.4]; n=964, d=471 | 37.7 [35.7, 39.8]; n=1255, d=600 |
| delayed2h/seen pairs | N3-fast-zero | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 0.0 [0.0, 0.0]; n=977, d=471 | 0.0 [0.0, 0.0]; n=964, d=471 | 0.0 [0.0, 0.0]; n=1255, d=600 |
| delayed2h/seen pairs | B1 follows | 98.0 [95.0, 100.0]; n=102, d=54 | 100.0 [100.0, 100.0]; n=168, d=92 | 99.0 [98.3, 99.5]; n=963, d=471 | 98.9 [98.3, 99.5]; n=952, d=471 | 99.0 [98.4, 99.5]; n=1233, d=600 |
| delayed2h/seen pairs | B1 unchanged | 0.0 [0.0, 0.0]; n=102, d=54 | 0.0 [0.0, 0.0]; n=168, d=92 | 0.0 [0.0, 0.0]; n=963, d=471 | 0.0 [0.0, 0.0]; n=952, d=471 | 0.0 [0.0, 0.0]; n=1233, d=600 |
| delayed2h/seen pairs | B1 other | 2.0 [0.0, 5.0]; n=102, d=54 | 0.0 [0.0, 0.0]; n=168, d=92 | 1.0 [0.5, 1.7]; n=963, d=471 | 1.1 [0.5, 1.7]; n=952, d=471 | 1.0 [0.5, 1.6]; n=1233, d=600 |
| delayed2h/seen pairs | B2 follows | NE: 36.1 [25.7, 45.5]; n=72, d=37 | 31.8 [24.0, 38.7]; n=129, d=69 | 30.4 [27.2, 33.3]; n=714, d=343 | 30.1 [26.9, 33.1]; n=705, d=342 | 31.0 [28.2, 33.7]; n=915, d=433 |
| delayed2h/seen pairs | B2 unchanged | NE: 37.5 [29.4, 45.8]; n=72, d=37 | 48.8 [40.6, 57.3]; n=129, d=69 | 45.7 [42.7, 48.5]; n=714, d=343 | 46.1 [43.2, 49.0]; n=705, d=342 | 45.5 [43.0, 48.2]; n=915, d=433 |
| delayed2h/seen pairs | B2 other | NE: 26.4 [17.4, 36.4]; n=72, d=37 | 19.4 [13.2, 25.3]; n=129, d=69 | 23.9 [21.2, 26.8]; n=714, d=343 | 23.8 [21.0, 26.7]; n=705, d=342 | 23.5 [21.1, 26.0]; n=915, d=433 |
| delayed2h/seen pairs | B3 follows | 98.0 [95.0, 100.0]; n=102, d=54 | 100.0 [100.0, 100.0]; n=168, d=92 | 99.0 [98.3, 99.5]; n=963, d=471 | 98.9 [98.3, 99.5]; n=952, d=471 | 99.0 [98.4, 99.5]; n=1233, d=600 |
| delayed2h/seen pairs | B3 unchanged | 0.0 [0.0, 0.0]; n=102, d=54 | 0.0 [0.0, 0.0]; n=168, d=92 | 0.0 [0.0, 0.0]; n=963, d=471 | 0.0 [0.0, 0.0]; n=952, d=471 | 0.0 [0.0, 0.0]; n=1233, d=600 |
| delayed2h/seen pairs | B3 other | 2.0 [0.0, 5.0]; n=102, d=54 | 0.0 [0.0, 0.0]; n=168, d=92 | 1.0 [0.5, 1.7]; n=963, d=471 | 1.1 [0.5, 1.7]; n=952, d=471 | 1.0 [0.5, 1.6]; n=1233, d=600 |
| delayed2h/new pairs | N0 | NE: 99.0 [96.9, 100.0]; n=99, d=52 | 97.2 [94.7, 99.4]; n=176, d=91 | 98.2 [97.4, 98.9]; n=979, d=471 | 98.1 [97.3, 98.9]; n=964, d=470 | 98.1 [97.3, 98.8]; n=1254, d=600 |
| delayed2h/new pairs | N1 | NE: 31.3 [24.2, 39.1]; n=99, d=52 | 41.5 [35.6, 47.2]; n=176, d=91 | 41.2 [39.1, 43.3]; n=979, d=471 | 41.6 [39.6, 43.8]; n=964, d=470 | 40.4 [38.6, 42.3]; n=1254, d=600 |
| delayed2h/new pairs | N2a | NE: 37.4 [31.3, 43.8]; n=99, d=52 | 43.8 [38.6, 49.3]; n=176, d=91 | 39.8 [37.8, 41.9]; n=979, d=471 | 40.4 [38.3, 42.5]; n=964, d=470 | 40.2 [38.5, 42.0]; n=1254, d=600 |
| delayed2h/new pairs | N2b-answer | NE: 40.4 [34.4, 47.3]; n=99, d=52 | 45.5 [40.2, 51.5]; n=176, d=91 | 44.1 [42.2, 46.2]; n=979, d=471 | 44.7 [42.7, 46.7]; n=964, d=470 | 44.0 [42.3, 45.8]; n=1254, d=600 |
| delayed2h/new pairs | N2b-all | NE: 40.4 [34.4, 47.3]; n=99, d=52 | 45.5 [40.2, 51.5]; n=176, d=91 | 44.1 [42.2, 46.2]; n=979, d=471 | 44.7 [42.7, 46.7]; n=964, d=470 | 44.0 [42.3, 45.8]; n=1254, d=600 |
| delayed2h/new pairs | N3-fast-kept | NE: 30.3 [23.5, 37.1]; n=99, d=52 | 38.1 [31.8, 44.4]; n=176, d=91 | 39.7 [37.6, 42.0]; n=979, d=471 | 40.1 [38.0, 42.5]; n=964, d=470 | 38.8 [36.9, 40.8]; n=1254, d=600 |
| delayed2h/new pairs | N3-fast-zero | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=979, d=471 | 0.0 [0.0, 0.0]; n=964, d=470 | 0.0 [0.0, 0.0]; n=1254, d=600 |
| delayed2h/new pairs | B1 follows | NE: 98.0 [94.8, 100.0]; n=98, d=52 | 99.4 [97.9, 100.0]; n=171, d=91 | 98.8 [98.1, 99.4]; n=961, d=471 | 98.7 [98.0, 99.4]; n=946, d=470 | 98.8 [98.2, 99.4]; n=1230, d=600 |
| delayed2h/new pairs | B1 unchanged | NE: 0.0 [0.0, 0.0]; n=98, d=52 | 0.0 [0.0, 0.0]; n=171, d=91 | 0.0 [0.0, 0.0]; n=961, d=471 | 0.0 [0.0, 0.0]; n=946, d=470 | 0.0 [0.0, 0.0]; n=1230, d=600 |
| delayed2h/new pairs | B1 other | NE: 2.0 [0.0, 5.2]; n=98, d=52 | 0.6 [0.0, 2.1]; n=171, d=91 | 1.2 [0.6, 1.9]; n=961, d=471 | 1.3 [0.6, 2.0]; n=946, d=470 | 1.2 [0.6, 1.8]; n=1230, d=600 |
| delayed2h/new pairs | B2 follows | NE: 36.4 [25.9, 46.8]; n=66, d=34 | 30.9 [23.0, 38.2]; n=136, d=70 | 31.3 [27.8, 34.5]; n=742, d=358 | 31.0 [27.7, 34.3]; n=728, d=355 | 31.6 [28.9, 34.4]; n=944, d=449 |
| delayed2h/new pairs | B2 unchanged | NE: 40.9 [30.5, 52.0]; n=66, d=34 | 47.1 [39.5, 55.0]; n=136, d=70 | 45.7 [42.9, 48.8]; n=742, d=358 | 46.0 [43.1, 49.2]; n=728, d=355 | 45.6 [43.1, 48.2]; n=944, d=449 |
| delayed2h/new pairs | B2 other | NE: 22.7 [12.5, 34.0]; n=66, d=34 | 22.1 [15.1, 28.9]; n=136, d=70 | 23.0 [20.1, 25.9]; n=742, d=358 | 22.9 [20.1, 25.8]; n=728, d=355 | 22.9 [20.4, 25.3]; n=944, d=449 |
| delayed2h/new pairs | B3 follows | NE: 98.0 [94.8, 100.0]; n=98, d=52 | 99.4 [97.9, 100.0]; n=171, d=91 | 98.8 [98.1, 99.4]; n=961, d=471 | 98.7 [98.0, 99.4]; n=946, d=470 | 98.8 [98.2, 99.4]; n=1230, d=600 |
| delayed2h/new pairs | B3 unchanged | NE: 0.0 [0.0, 0.0]; n=98, d=52 | 0.0 [0.0, 0.0]; n=171, d=91 | 0.0 [0.0, 0.0]; n=961, d=471 | 0.0 [0.0, 0.0]; n=946, d=470 | 0.0 [0.0, 0.0]; n=1230, d=600 |
| delayed2h/new pairs | B3 other | NE: 2.0 [0.0, 5.2]; n=98, d=52 | 0.6 [0.0, 2.1]; n=171, d=91 | 1.2 [0.6, 1.9]; n=961, d=471 | 1.3 [0.6, 2.0]; n=946, d=470 | 1.2 [0.6, 1.8]; n=1230, d=600 |
| delayed2h/never-seen bytes | N0 | 95.8 [91.1, 99.1]; n=119, d=55 | 98.4 [96.7, 100.0]; n=192, d=97 | 98.4 [97.6, 99.2]; n=953, d=463 | 98.4 [97.5, 99.1]; n=936, d=463 | 98.2 [97.3, 98.9]; n=1264, d=600 |
| delayed2h/never-seen bytes | N1 | 31.9 [26.4, 37.6]; n=119, d=55 | 34.9 [30.2, 39.5]; n=192, d=97 | 40.7 [38.5, 42.9]; n=953, d=463 | 41.2 [39.0, 43.4]; n=936, d=463 | 39.0 [37.2, 40.9]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2a | 32.8 [26.6, 40.7]; n=119, d=55 | 34.4 [29.4, 39.8]; n=192, d=97 | 40.8 [38.8, 42.8]; n=953, d=463 | 41.5 [39.3, 43.6]; n=936, d=463 | 39.1 [37.3, 41.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-answer | 38.7 [33.0, 45.7]; n=119, d=55 | 40.1 [35.7, 45.0]; n=192, d=97 | 44.0 [42.0, 46.1]; n=953, d=463 | 44.6 [42.6, 46.7]; n=936, d=463 | 42.9 [41.2, 44.8]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-all | 38.7 [33.0, 45.7]; n=119, d=55 | 40.1 [35.7, 45.0]; n=192, d=97 | 44.0 [42.0, 46.1]; n=953, d=463 | 44.6 [42.6, 46.7]; n=936, d=463 | 42.9 [41.2, 44.8]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3-fast-kept | 31.1 [25.0, 37.9]; n=119, d=55 | 34.4 [29.4, 39.6]; n=192, d=97 | 39.0 [36.7, 41.4]; n=953, d=463 | 39.5 [37.2, 41.9]; n=936, d=463 | 37.6 [35.7, 39.6]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3-fast-zero | 0.0 [0.0, 0.0]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.0 [0.0, 0.0]; n=953, d=463 | 0.0 [0.0, 0.0]; n=936, d=463 | 0.0 [0.0, 0.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | B1 follows | 97.4 [94.4, 100.0]; n=114, d=55 | 97.4 [94.9, 99.4]; n=189, d=97 | 98.9 [98.3, 99.5]; n=938, d=462 | 99.0 [98.4, 99.6]; n=921, d=462 | 98.5 [97.9, 99.1]; n=1241, d=599 |
| delayed2h/never-seen bytes | B1 unchanged | 0.0 [0.0, 0.0]; n=114, d=55 | 0.0 [0.0, 0.0]; n=189, d=97 | 0.0 [0.0, 0.0]; n=938, d=462 | 0.0 [0.0, 0.0]; n=921, d=462 | 0.0 [0.0, 0.0]; n=1241, d=599 |
| delayed2h/never-seen bytes | B1 other | 2.6 [0.0, 5.6]; n=114, d=55 | 2.6 [0.6, 5.1]; n=189, d=97 | 1.1 [0.5, 1.7]; n=938, d=462 | 1.0 [0.4, 1.6]; n=921, d=462 | 1.5 [0.9, 2.1]; n=1241, d=599 |
| delayed2h/never-seen bytes | B2 follows | NE: 36.8 [27.8, 46.1]; n=87, d=39 | 36.7 [28.7, 44.7]; n=139, d=71 | 33.1 [30.0, 36.3]; n=724, d=352 | 33.1 [29.7, 36.2]; n=710, d=350 | 34.0 [31.2, 36.9]; n=950, d=450 |
| delayed2h/never-seen bytes | B2 unchanged | NE: 40.2 [33.0, 49.1]; n=87, d=39 | 38.8 [32.9, 46.1]; n=139, d=71 | 40.9 [37.7, 43.9]; n=724, d=352 | 41.3 [38.0, 44.4]; n=710, d=350 | 40.5 [38.0, 43.4]; n=950, d=450 |
| delayed2h/never-seen bytes | B2 other | NE: 23.0 [16.1, 29.3]; n=87, d=39 | 24.5 [17.4, 31.2]; n=139, d=71 | 26.0 [23.1, 28.9]; n=724, d=352 | 25.6 [22.8, 28.7]; n=710, d=350 | 25.5 [23.0, 27.9]; n=950, d=450 |
| delayed2h/never-seen bytes | B3 follows | 97.4 [94.4, 100.0]; n=114, d=55 | 97.4 [94.9, 99.4]; n=189, d=97 | 98.9 [98.3, 99.5]; n=938, d=462 | 99.0 [98.4, 99.6]; n=921, d=462 | 98.5 [97.9, 99.1]; n=1241, d=599 |
| delayed2h/never-seen bytes | B3 unchanged | 0.0 [0.0, 0.0]; n=114, d=55 | 0.0 [0.0, 0.0]; n=189, d=97 | 0.0 [0.0, 0.0]; n=938, d=462 | 0.0 [0.0, 0.0]; n=921, d=462 | 0.0 [0.0, 0.0]; n=1241, d=599 |
| delayed2h/never-seen bytes | B3 other | 2.6 [0.0, 5.6]; n=114, d=55 | 2.6 [0.6, 5.1]; n=189, d=97 | 1.1 [0.5, 1.7]; n=938, d=462 | 1.0 [0.4, 1.6]; n=921, d=462 | 1.5 [0.9, 2.1]; n=1241, d=599 |

## candidate-s4

There is no reader; all buckets are distance strata only.

Checkpoint: `/home/joe/hsn-experiments/vast2/exp/gpu2/runs/a13_follow/hold_f8ru20nw_ag_ng2_cutC_cells64_18k_gsum-s4-143bb5d39ca5/exp6/step_00018000.pt`. Status: **complete**.

| Task / keys | Condition | Near | Mid | Far | Far-strict | All |
| --- | --- | --- | --- | --- | --- | --- |
| kv/training keys | N0 | 99.0 [98.3, 99.5]; n=1062, d=536 | 98.7 [97.9, 99.4]; n=998, d=479 | 97.7 [96.8, 98.5]; n=1264, d=485 | 97.9 [97.0, 98.7]; n=1082, d=475 | 98.4 [97.9, 98.8]; n=3324, d=600 |
| kv/training keys | N1 | 10.8 [9.1, 12.7]; n=1062, d=536 | 15.9 [13.7, 18.3]; n=998, d=479 | 37.9 [35.9, 40.1]; n=1264, d=485 | 37.7 [35.3, 40.0]; n=1082, d=475 | 22.7 [21.5, 23.8]; n=3324, d=600 |
| kv/training keys | N2a | 10.3 [8.6, 11.9]; n=1062, d=536 | 13.6 [11.5, 15.8]; n=998, d=479 | 38.2 [36.2, 40.2]; n=1264, d=485 | 40.7 [38.5, 42.9]; n=1082, d=475 | 21.9 [20.8, 22.8]; n=3324, d=600 |
| kv/training keys | N2b-answer | 18.3 [16.0, 20.5]; n=1062, d=536 | 20.9 [18.6, 23.5]; n=998, d=479 | 47.0 [45.1, 49.0]; n=1264, d=485 | 46.0 [43.8, 48.2]; n=1082, d=475 | 30.0 [28.8, 31.2]; n=3324, d=600 |
| kv/training keys | N2b-all | 18.3 [16.0, 20.5]; n=1062, d=536 | 20.9 [18.6, 23.5]; n=998, d=479 | 47.0 [45.1, 49.0]; n=1264, d=485 | 46.0 [43.8, 48.2]; n=1082, d=475 | 30.0 [28.8, 31.2]; n=3324, d=600 |
| kv/training keys | N3-fast-kept | 8.8 [7.1, 10.6]; n=1062, d=536 | 12.9 [10.9, 14.9]; n=998, d=479 | 32.2 [30.0, 34.3]; n=1264, d=485 | 32.1 [29.7, 34.4]; n=1082, d=475 | 18.9 [17.8, 20.0]; n=3324, d=600 |
| kv/training keys | N3-fast-zero | 15.1 [13.0, 17.1]; n=1062, d=536 | 13.8 [11.8, 16.1]; n=998, d=479 | 16.2 [14.4, 18.2]; n=1264, d=485 | 14.5 [12.8, 16.3]; n=1082, d=475 | 15.1 [13.9, 16.3]; n=3324, d=600 |
| kv/training keys | B1 follows | 99.5 [99.1, 99.9]; n=885, d=486 | 99.6 [99.2, 100.0]; n=840, d=451 | 98.1 [97.3, 98.9]; n=1023, d=457 | 98.0 [97.0, 98.8]; n=896, d=442 | 99.1 [98.7, 99.4]; n=2748, d=589 |
| kv/training keys | B1 unchanged | 0.0 [0.0, 0.0]; n=885, d=486 | 0.0 [0.0, 0.0]; n=840, d=451 | 0.0 [0.0, 0.0]; n=1023, d=457 | 0.0 [0.0, 0.0]; n=896, d=442 | 0.0 [0.0, 0.0]; n=2748, d=589 |
| kv/training keys | B1 other | 0.5 [0.1, 0.9]; n=885, d=486 | 0.4 [0.0, 0.8]; n=840, d=451 | 1.9 [1.1, 2.7]; n=1023, d=457 | 2.0 [1.2, 3.0]; n=896, d=442 | 0.9 [0.6, 1.3]; n=2748, d=589 |
| kv/dev keys | N0 | 99.2 [98.6, 99.6]; n=1071, d=531 | 98.5 [97.7, 99.2]; n=1011, d=476 | 96.8 [95.9, 97.7]; n=1263, d=477 | 97.4 [96.4, 98.3]; n=1095, d=468 | 98.1 [97.7, 98.5]; n=3345, d=600 |
| kv/dev keys | N1 | 8.3 [6.6, 10.0]; n=1071, d=531 | 10.2 [8.4, 12.1]; n=1011, d=476 | 36.0 [34.1, 38.0]; n=1263, d=477 | 36.8 [34.7, 38.9]; n=1095, d=468 | 19.3 [18.3, 20.3]; n=3345, d=600 |
| kv/dev keys | N2a | 7.7 [6.1, 9.1]; n=1071, d=531 | 12.8 [10.9, 14.8]; n=1011, d=476 | 37.2 [35.2, 39.3]; n=1263, d=477 | 39.5 [37.4, 41.6]; n=1095, d=468 | 20.4 [19.4, 21.4]; n=3345, d=600 |
| kv/dev keys | N2b-answer | 17.6 [15.5, 19.8]; n=1071, d=531 | 21.1 [18.6, 23.6]; n=1011, d=476 | 45.1 [43.2, 47.3]; n=1263, d=477 | 43.7 [41.7, 45.8]; n=1095, d=468 | 29.1 [27.9, 30.3]; n=3345, d=600 |
| kv/dev keys | N2b-all | 17.6 [15.5, 19.8]; n=1071, d=531 | 21.1 [18.6, 23.6]; n=1011, d=476 | 45.1 [43.2, 47.3]; n=1263, d=477 | 43.7 [41.7, 45.8]; n=1095, d=468 | 29.1 [27.9, 30.3]; n=3345, d=600 |
| kv/dev keys | N3-fast-kept | 6.3 [4.8, 7.8]; n=1071, d=531 | 8.8 [7.1, 10.5]; n=1011, d=476 | 30.5 [28.6, 32.6]; n=1263, d=477 | 31.5 [29.4, 33.8]; n=1095, d=468 | 16.2 [15.3, 17.1]; n=3345, d=600 |
| kv/dev keys | N3-fast-zero | 5.6 [4.3, 7.0]; n=1071, d=531 | 4.5 [3.2, 5.8]; n=1011, d=476 | 4.7 [3.6, 5.9]; n=1263, d=477 | 4.4 [3.3, 5.7]; n=1095, d=468 | 4.9 [4.2, 5.7]; n=3345, d=600 |
| kv/dev keys | B1 follows | 99.7 [99.2, 100.0]; n=890, d=490 | 99.4 [98.8, 99.9]; n=841, d=440 | 96.9 [95.8, 97.9]; n=1042, d=451 | 97.0 [95.7, 98.1]; n=933, d=433 | 98.6 [98.1, 99.0]; n=2773, d=594 |
| kv/dev keys | B1 unchanged | 0.0 [0.0, 0.0]; n=890, d=490 | 0.0 [0.0, 0.0]; n=841, d=440 | 0.0 [0.0, 0.0]; n=1042, d=451 | 0.0 [0.0, 0.0]; n=933, d=433 | 0.0 [0.0, 0.0]; n=2773, d=594 |
| kv/dev keys | B1 other | 0.3 [0.0, 0.8]; n=890, d=490 | 0.6 [0.1, 1.2]; n=841, d=440 | 3.1 [2.1, 4.2]; n=1042, d=451 | 3.0 [1.9, 4.3]; n=933, d=433 | 1.4 [1.0, 1.9]; n=2773, d=594 |
| kv/test keys | N0 | 98.7 [98.0, 99.4]; n=1025, d=528 | 97.5 [96.5, 98.6]; n=1041, d=480 | 96.1 [95.0, 97.2]; n=1298, d=487 | 96.9 [95.9, 97.8]; n=1129, d=477 | 97.4 [96.8, 97.9]; n=3364, d=600 |
| kv/test keys | N1 | 7.8 [6.3, 9.4]; n=1025, d=528 | 10.1 [8.4, 12.1]; n=1041, d=480 | 34.7 [32.6, 36.7]; n=1298, d=487 | 35.2 [33.2, 37.2]; n=1129, d=477 | 18.9 [17.9, 19.9]; n=3364, d=600 |
| kv/test keys | N2a | 8.5 [6.9, 10.0]; n=1025, d=528 | 11.6 [9.8, 13.5]; n=1041, d=480 | 36.1 [34.4, 38.1]; n=1298, d=487 | 38.0 [36.0, 40.1]; n=1129, d=477 | 20.1 [19.2, 21.0]; n=3364, d=600 |
| kv/test keys | N2b-answer | 17.4 [15.2, 19.5]; n=1025, d=528 | 21.8 [19.5, 24.2]; n=1041, d=480 | 46.1 [44.2, 48.2]; n=1298, d=487 | 44.3 [42.2, 46.5]; n=1129, d=477 | 29.8 [28.7, 31.1]; n=3364, d=600 |
| kv/test keys | N2b-all | 17.4 [15.2, 19.5]; n=1025, d=528 | 21.8 [19.5, 24.2]; n=1041, d=480 | 46.1 [44.2, 48.2]; n=1298, d=487 | 44.3 [42.2, 46.5]; n=1129, d=477 | 29.8 [28.7, 31.1]; n=3364, d=600 |
| kv/test keys | N3-fast-kept | 7.1 [5.6, 8.5]; n=1025, d=528 | 8.2 [6.4, 9.9]; n=1041, d=480 | 29.3 [27.1, 31.3]; n=1298, d=487 | 30.6 [28.4, 32.6]; n=1129, d=477 | 16.0 [14.9, 16.9]; n=3364, d=600 |
| kv/test keys | N3-fast-zero | 2.1 [1.3, 3.0]; n=1025, d=528 | 2.6 [1.7, 3.6]; n=1041, d=480 | 2.1 [1.4, 2.8]; n=1298, d=487 | 2.3 [1.5, 3.2]; n=1129, d=477 | 2.3 [1.8, 2.8]; n=3364, d=600 |
| kv/test keys | B1 follows | 98.9 [98.2, 99.5]; n=852, d=480 | 98.2 [97.3, 99.0]; n=835, d=440 | 97.5 [96.5, 98.3]; n=1032, d=463 | 97.3 [96.2, 98.3]; n=934, d=450 | 98.2 [97.6, 98.6]; n=2719, d=597 |
| kv/test keys | B1 unchanged | 0.0 [0.0, 0.0]; n=852, d=480 | 0.0 [0.0, 0.0]; n=835, d=440 | 0.0 [0.0, 0.0]; n=1032, d=463 | 0.0 [0.0, 0.0]; n=934, d=450 | 0.0 [0.0, 0.0]; n=2719, d=597 |
| kv/test keys | B1 other | 1.1 [0.5, 1.8]; n=852, d=480 | 1.8 [1.0, 2.7]; n=835, d=440 | 2.5 [1.7, 3.5]; n=1032, d=463 | 2.7 [1.7, 3.8]; n=934, d=450 | 1.8 [1.4, 2.4]; n=2719, d=597 |
| delayed2h/seen pairs | N0 | 100.0 [100.0, 100.0]; n=103, d=54 | 98.9 [97.0, 100.0]; n=175, d=92 | 99.5 [99.1, 99.9]; n=977, d=471 | 99.5 [99.0, 99.9]; n=964, d=471 | 99.4 [99.0, 99.8]; n=1255, d=600 |
| delayed2h/seen pairs | N1 | 44.7 [38.5, 52.2]; n=103, d=54 | 48.0 [42.6, 54.0]; n=175, d=92 | 48.2 [46.1, 50.6]; n=977, d=471 | 48.7 [46.5, 51.0]; n=964, d=471 | 47.9 [46.0, 50.0]; n=1255, d=600 |
| delayed2h/seen pairs | N2a | 41.7 [35.5, 49.4]; n=103, d=54 | 46.9 [41.4, 53.1]; n=175, d=92 | 46.2 [44.0, 48.6]; n=977, d=471 | 46.8 [44.6, 49.2]; n=964, d=471 | 45.9 [44.0, 47.9]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-answer | 45.6 [38.9, 53.7]; n=103, d=54 | 48.6 [43.3, 54.7]; n=175, d=92 | 49.4 [47.3, 51.8]; n=977, d=471 | 50.1 [47.9, 52.4]; n=964, d=471 | 49.0 [47.2, 51.0]; n=1255, d=600 |
| delayed2h/seen pairs | N2b-all | 45.6 [38.9, 53.7]; n=103, d=54 | 48.6 [43.3, 54.7]; n=175, d=92 | 49.4 [47.3, 51.8]; n=977, d=471 | 50.1 [47.9, 52.4]; n=964, d=471 | 49.0 [47.2, 51.0]; n=1255, d=600 |
| delayed2h/seen pairs | N3-fast-kept | 40.8 [34.1, 49.4]; n=103, d=54 | 42.3 [36.7, 48.1]; n=175, d=92 | 44.7 [42.3, 47.2]; n=977, d=471 | 44.7 [42.3, 47.1]; n=964, d=471 | 44.1 [42.0, 46.2]; n=1255, d=600 |
| delayed2h/seen pairs | N3-fast-zero | 1.9 [0.0, 4.7]; n=103, d=54 | 0.0 [0.0, 0.0]; n=175, d=92 | 1.1 [0.5, 1.8]; n=977, d=471 | 1.1 [0.5, 1.8]; n=964, d=471 | 1.0 [0.6, 1.6]; n=1255, d=600 |
| delayed2h/seen pairs | B1 follows | 99.0 [96.8, 100.0]; n=103, d=54 | 100.0 [100.0, 100.0]; n=173, d=92 | 99.5 [99.0, 99.9]; n=972, d=471 | 99.5 [99.0, 99.9]; n=959, d=471 | 99.5 [99.1, 99.8]; n=1248, d=600 |
| delayed2h/seen pairs | B1 unchanged | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=173, d=92 | 0.0 [0.0, 0.0]; n=972, d=471 | 0.0 [0.0, 0.0]; n=959, d=471 | 0.0 [0.0, 0.0]; n=1248, d=600 |
| delayed2h/seen pairs | B1 other | 1.0 [0.0, 3.2]; n=103, d=54 | 0.0 [0.0, 0.0]; n=173, d=92 | 0.5 [0.1, 1.0]; n=972, d=471 | 0.5 [0.1, 1.0]; n=959, d=471 | 0.5 [0.2, 0.9]; n=1248, d=600 |
| delayed2h/seen pairs | B2 follows | NE: 56.2 [46.7, 64.1]; n=73, d=37 | 44.8 [36.6, 52.6]; n=134, d=69 | 47.2 [43.7, 50.9]; n=722, d=343 | 47.1 [43.7, 50.8]; n=711, d=342 | 47.6 [44.6, 50.7]; n=929, d=433 |
| delayed2h/seen pairs | B2 unchanged | NE: 26.0 [17.6, 35.9]; n=73, d=37 | 40.3 [33.8, 47.6]; n=134, d=69 | 36.7 [33.5, 40.2]; n=722, d=343 | 36.8 [33.7, 40.2]; n=711, d=342 | 36.4 [33.6, 39.4]; n=929, d=433 |
| delayed2h/seen pairs | B2 other | NE: 17.8 [10.3, 27.3]; n=73, d=37 | 14.9 [9.3, 20.1]; n=134, d=69 | 16.1 [13.5, 18.7]; n=722, d=343 | 16.0 [13.5, 18.7]; n=711, d=342 | 16.0 [13.7, 18.3]; n=929, d=433 |
| delayed2h/seen pairs | B3 follows | 99.0 [96.8, 100.0]; n=103, d=54 | 100.0 [100.0, 100.0]; n=173, d=92 | 99.5 [99.0, 99.9]; n=972, d=471 | 99.5 [99.0, 99.9]; n=959, d=471 | 99.5 [99.1, 99.8]; n=1248, d=600 |
| delayed2h/seen pairs | B3 unchanged | 0.0 [0.0, 0.0]; n=103, d=54 | 0.0 [0.0, 0.0]; n=173, d=92 | 0.0 [0.0, 0.0]; n=972, d=471 | 0.0 [0.0, 0.0]; n=959, d=471 | 0.0 [0.0, 0.0]; n=1248, d=600 |
| delayed2h/seen pairs | B3 other | 1.0 [0.0, 3.2]; n=103, d=54 | 0.0 [0.0, 0.0]; n=173, d=92 | 0.5 [0.1, 1.0]; n=972, d=471 | 0.5 [0.1, 1.0]; n=959, d=471 | 0.5 [0.2, 0.9]; n=1248, d=600 |
| delayed2h/new pairs | N0 | NE: 100.0 [100.0, 100.0]; n=99, d=52 | 100.0 [100.0, 100.0]; n=176, d=91 | 99.0 [98.4, 99.5]; n=979, d=471 | 99.0 [98.3, 99.5]; n=964, d=470 | 99.2 [98.7, 99.6]; n=1254, d=600 |
| delayed2h/new pairs | N1 | NE: 44.4 [37.6, 52.6]; n=99, d=52 | 48.9 [43.2, 55.1]; n=176, d=91 | 47.9 [45.7, 50.3]; n=979, d=471 | 48.3 [46.2, 50.7]; n=964, d=470 | 47.8 [45.8, 49.9]; n=1254, d=600 |
| delayed2h/new pairs | N2a | NE: 43.4 [36.6, 51.8]; n=99, d=52 | 46.0 [40.1, 52.5]; n=176, d=91 | 45.5 [43.0, 47.9]; n=979, d=471 | 46.1 [43.6, 48.6]; n=964, d=470 | 45.4 [43.3, 47.6]; n=1254, d=600 |
| delayed2h/new pairs | N2b-answer | NE: 48.5 [42.0, 56.6]; n=99, d=52 | 48.3 [42.6, 54.4]; n=176, d=91 | 48.5 [46.4, 50.9]; n=979, d=471 | 49.1 [47.0, 51.4]; n=964, d=470 | 48.5 [46.5, 50.6]; n=1254, d=600 |
| delayed2h/new pairs | N2b-all | NE: 48.5 [42.0, 56.6]; n=99, d=52 | 48.3 [42.6, 54.4]; n=176, d=91 | 48.5 [46.4, 50.9]; n=979, d=471 | 49.1 [47.0, 51.4]; n=964, d=470 | 48.5 [46.5, 50.6]; n=1254, d=600 |
| delayed2h/new pairs | N3-fast-kept | NE: 43.4 [36.4, 51.9]; n=99, d=52 | 46.0 [40.0, 52.7]; n=176, d=91 | 44.8 [42.4, 47.3]; n=979, d=471 | 44.8 [42.5, 47.3]; n=964, d=470 | 44.9 [42.8, 47.1]; n=1254, d=600 |
| delayed2h/new pairs | N3-fast-zero | NE: 2.0 [0.0, 4.9]; n=99, d=52 | 1.7 [0.0, 3.8]; n=176, d=91 | 1.5 [0.8, 2.3]; n=979, d=471 | 1.6 [0.8, 2.4]; n=964, d=470 | 1.6 [0.9, 2.3]; n=1254, d=600 |
| delayed2h/new pairs | B1 follows | NE: 99.0 [96.8, 100.0]; n=99, d=52 | 98.3 [96.3, 100.0]; n=176, d=91 | 99.8 [99.5, 100.0]; n=969, d=471 | 99.8 [99.5, 100.0]; n=954, d=470 | 99.5 [99.0, 99.9]; n=1244, d=600 |
| delayed2h/new pairs | B1 unchanged | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=969, d=471 | 0.0 [0.0, 0.0]; n=954, d=470 | 0.0 [0.0, 0.0]; n=1244, d=600 |
| delayed2h/new pairs | B1 other | NE: 1.0 [0.0, 3.2]; n=99, d=52 | 1.7 [0.0, 3.7]; n=176, d=91 | 0.2 [0.0, 0.5]; n=969, d=471 | 0.2 [0.0, 0.5]; n=954, d=470 | 0.5 [0.1, 1.0]; n=1244, d=600 |
| delayed2h/new pairs | B2 follows | NE: 53.7 [40.0, 67.2]; n=67, d=34 | 48.6 [39.7, 56.9]; n=140, d=70 | 47.7 [44.4, 51.1]; n=747, d=358 | 47.1 [43.6, 50.5]; n=733, d=355 | 48.2 [45.5, 51.1]; n=954, d=449 |
| delayed2h/new pairs | B2 unchanged | NE: 35.8 [25.4, 47.5]; n=67, d=34 | 34.3 [27.4, 42.3]; n=140, d=70 | 37.5 [34.4, 40.3]; n=747, d=358 | 37.9 [34.7, 40.8]; n=733, d=355 | 36.9 [34.2, 39.5]; n=954, d=449 |
| delayed2h/new pairs | B2 other | NE: 10.4 [2.7, 19.1]; n=67, d=34 | 17.1 [11.1, 23.8]; n=140, d=70 | 14.9 [12.5, 17.4]; n=747, d=358 | 15.0 [12.5, 17.5]; n=733, d=355 | 14.9 [12.8, 17.2]; n=954, d=449 |
| delayed2h/new pairs | B3 follows | NE: 99.0 [96.8, 100.0]; n=99, d=52 | 98.3 [96.3, 100.0]; n=176, d=91 | 99.8 [99.5, 100.0]; n=969, d=471 | 99.8 [99.5, 100.0]; n=954, d=470 | 99.5 [99.0, 99.9]; n=1244, d=600 |
| delayed2h/new pairs | B3 unchanged | NE: 0.0 [0.0, 0.0]; n=99, d=52 | 0.0 [0.0, 0.0]; n=176, d=91 | 0.0 [0.0, 0.0]; n=969, d=471 | 0.0 [0.0, 0.0]; n=954, d=470 | 0.0 [0.0, 0.0]; n=1244, d=600 |
| delayed2h/new pairs | B3 other | NE: 1.0 [0.0, 3.2]; n=99, d=52 | 1.7 [0.0, 3.7]; n=176, d=91 | 0.2 [0.0, 0.5]; n=969, d=471 | 0.2 [0.0, 0.5]; n=954, d=470 | 0.5 [0.1, 1.0]; n=1244, d=600 |
| delayed2h/never-seen bytes | N0 | 96.6 [93.6, 99.2]; n=119, d=55 | 97.9 [95.9, 99.5]; n=192, d=97 | 99.5 [99.0, 99.9]; n=953, d=463 | 99.5 [98.9, 99.9]; n=936, d=463 | 99.0 [98.4, 99.4]; n=1264, d=600 |
| delayed2h/never-seen bytes | N1 | 40.3 [34.8, 47.2]; n=119, d=55 | 44.3 [40.1, 49.2]; n=192, d=97 | 47.0 [44.8, 49.3]; n=953, d=463 | 47.4 [45.1, 49.7]; n=936, d=463 | 46.0 [44.2, 48.0]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2a | 32.8 [26.1, 40.7]; n=119, d=55 | 41.7 [36.9, 47.1]; n=192, d=97 | 44.8 [42.4, 47.2]; n=953, d=463 | 45.4 [42.9, 47.8]; n=936, d=463 | 43.2 [41.2, 45.4]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-answer | 42.0 [36.2, 49.5]; n=119, d=55 | 46.9 [42.2, 51.8]; n=192, d=97 | 49.9 [47.9, 52.3]; n=953, d=463 | 50.6 [48.4, 53.0]; n=936, d=463 | 48.7 [46.9, 50.9]; n=1264, d=600 |
| delayed2h/never-seen bytes | N2b-all | 42.0 [36.2, 49.5]; n=119, d=55 | 46.9 [42.2, 51.8]; n=192, d=97 | 49.9 [47.9, 52.3]; n=953, d=463 | 50.6 [48.4, 53.0]; n=936, d=463 | 48.7 [46.9, 50.9]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3-fast-kept | 35.3 [28.4, 43.1]; n=119, d=55 | 40.1 [35.3, 45.8]; n=192, d=97 | 41.7 [39.5, 44.1]; n=953, d=463 | 42.1 [39.8, 44.6]; n=936, d=463 | 40.8 [38.9, 42.9]; n=1264, d=600 |
| delayed2h/never-seen bytes | N3-fast-zero | 0.8 [0.0, 2.8]; n=119, d=55 | 0.0 [0.0, 0.0]; n=192, d=97 | 0.2 [0.0, 0.5]; n=953, d=463 | 0.2 [0.0, 0.5]; n=936, d=463 | 0.2 [0.0, 0.5]; n=1264, d=600 |
| delayed2h/never-seen bytes | B1 follows | 99.1 [96.9, 100.0]; n=115, d=55 | 99.5 [98.3, 100.0]; n=188, d=97 | 99.2 [98.6, 99.7]; n=948, d=463 | 99.1 [98.6, 99.7]; n=931, d=463 | 99.2 [98.7, 99.6]; n=1251, d=600 |
| delayed2h/never-seen bytes | B1 unchanged | 0.0 [0.0, 0.0]; n=115, d=55 | 0.0 [0.0, 0.0]; n=188, d=97 | 0.0 [0.0, 0.0]; n=948, d=463 | 0.0 [0.0, 0.0]; n=931, d=463 | 0.0 [0.0, 0.0]; n=1251, d=600 |
| delayed2h/never-seen bytes | B1 other | 0.9 [0.0, 3.1]; n=115, d=55 | 0.5 [0.0, 1.7]; n=188, d=97 | 0.8 [0.3, 1.4]; n=948, d=463 | 0.9 [0.3, 1.4]; n=931, d=463 | 0.8 [0.4, 1.3]; n=1251, d=600 |
| delayed2h/never-seen bytes | B2 follows | NE: 63.2 [51.0, 74.0]; n=87, d=39 | 61.6 [53.2, 68.9]; n=138, d=70 | 51.7 [48.2, 55.1]; n=729, d=353 | 51.9 [48.4, 55.3]; n=715, d=351 | 54.2 [51.3, 57.2]; n=954, d=451 |
| delayed2h/never-seen bytes | B2 unchanged | NE: 25.3 [16.7, 34.6]; n=87, d=39 | 23.2 [16.9, 30.7]; n=138, d=70 | 31.4 [28.4, 34.6]; n=729, d=353 | 31.6 [28.6, 34.7]; n=715, d=351 | 29.7 [27.2, 32.5]; n=954, d=451 |
| delayed2h/never-seen bytes | B2 other | NE: 11.5 [5.2, 18.2]; n=87, d=39 | 15.2 [9.2, 21.4]; n=138, d=70 | 16.9 [14.3, 19.5]; n=729, d=353 | 16.5 [13.9, 19.1]; n=715, d=351 | 16.1 [13.8, 18.5]; n=954, d=451 |
| delayed2h/never-seen bytes | B3 follows | 99.1 [96.9, 100.0]; n=115, d=55 | 99.5 [98.3, 100.0]; n=188, d=97 | 99.2 [98.6, 99.7]; n=948, d=463 | 99.1 [98.6, 99.7]; n=931, d=463 | 99.2 [98.7, 99.6]; n=1251, d=600 |
| delayed2h/never-seen bytes | B3 unchanged | 0.0 [0.0, 0.0]; n=115, d=55 | 0.0 [0.0, 0.0]; n=188, d=97 | 0.0 [0.0, 0.0]; n=948, d=463 | 0.0 [0.0, 0.0]; n=931, d=463 | 0.0 [0.0, 0.0]; n=1251, d=600 |
| delayed2h/never-seen bytes | B3 other | 0.9 [0.0, 3.1]; n=115, d=55 | 0.5 [0.0, 1.7]; n=188, d=97 | 0.8 [0.3, 1.4]; n=948, d=463 | 0.9 [0.3, 1.4]; n=931, d=463 | 0.8 [0.4, 1.3]; n=1251, d=600 |
