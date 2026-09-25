---
name: probe-battery
description: Scaffold a "does X add beyond Y" probe experiment with the full v3/rule-3 discipline (base-only PCA, matched shuffled null, paired image-level CI, executable gates). Use whenever a new added-information comparison is needed.
---

Generate a NEW experiment file under `experiments/` for a "does block X add beyond base Y"
question. Never run it in this invocation — generation is reviewable, execution is not.

## Arguments
Parse from the user's request: the base features (cache path + slice), the candidate block,
the dataset(s), and n. If any is missing, ask.

## Non-negotiable design (each rule cites the failure that created it — see CLAUDE.md)
1. Fit StandardScaler + PCA (cap 512, rank-guarded `min(cap, D, n_train-1)`) on the BASE
   ALONE; append the candidate standardized-raw. Never fit any transform on `[base|block]`
   (eviction, 6o-A) and never compress the candidate harder than the base (compression
   null, §13).
2. Null arm = the candidate ROW-SHUFFLED (same width AND same marginal distribution). If
   the candidate is a transform of another block (e.g. L2-normalized), shuffle the
   TRANSFORMED block (6s-10).
3. Report BOTH bars: raw delta vs base AND delta vs the shuffled null (paired per image:
   `plus_block − plus_null`), each with a 10k image-level bootstrap CI. "Adds beyond"
   requires BOTH > 0 (rule 3b: a redundant block beats its null; §11).
4. Executable GATE printed at the end: (a) probe sanity — a ~6-dim iid-noise block must
   straddle 0; (b) detection power — a synthetic anchor (labels + noise, width-matched)
   must beat ITS OWN null. On FAIL print "do NOT quote any cell above".
5. Track lbfgs `maxiter` hit-rate per arm and print it beside each comparison (operating-
   point matching, 6p).
6. Cache per-image correctness vectors for EVERY arm to `results/<name>_{ds}.npz` (rule 12:
   re-analysis must be offline).
7. Before any comparison across caches, verify `subset_indices` equality (never labels
   alone — 6o-F). Fail loudly.
8. Fixed C=0.1, SEEDS=[0,1,2], 5-fold StratifiedKFold, joblib n_jobs≈7 — match the
   project's standing operating point unless the user says otherwise.

## Reference implementations (copy structure, not bugs)
`experiments/prototypes/curv5000_probe.py` (canonical, incl. gate + caching),
`experiments/prototypes/time_aggregation.py` (time axis), `experiments/prototypes/block_aggregation.py` (depth).

## After generating
Run `ruff check --fix` + `ruff format` on the new file, `python -m py_compile`, and a
`--help`/small `--max-n` self-check if the file supports one. Then STOP and hand the file
to the user; record the planned experiment in RESEARCH_NOTES only after it has run.
