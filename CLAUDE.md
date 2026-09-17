# Project guardrails

Research code for probing flow-matching trajectories (frozen FLUX.1-dev) on remote-sensing
classification. RESEARCH_NOTES.md is the running log (§8 holds the standing experiment
rules; read it before designing any ablation). These rules exist because each one is a
distilled backtrack — the section that documents the failure is cited inline.

## Statistics: rules that would have caught our actual retractions

1. **Never select on the data that scores the selection.** Any `argmax` / `max()` / "best"
   over timesteps, C values, poolings, probes, or grids must be chosen on a split (or
   folds) disjoint from the one that produces the reported number. Violations found three
   times: concat best-t (6m), protocol_shape operating point (6o-B), max-of-four-probes z
   (6o-E). If you see `best = max(...)` feeding a reported delta or CI, stop.

2. **A control that reads impossibly is a broken instrument, not a footnote.** A synthetic
   block containing the labels scored *negative* (-0.015) and was footnoted as "anchor
   miscalibrated" for two weeks; the true cause (PCA eviction, 6o-A/6p) biased the
   project's headline by 80% and had produced a false cross-dataset interpretation.
   If a positive control fails or a null control is significantly nonzero, the ONLY next
   step is diagnosing the instrument — never interpreting around it.

3. **"+block adds X" requires the base to be IDENTICAL in both arms.** Fitting any
   transform (PCA, scaler, whitening) on `[base | block]` lets the block displace the
   base; the delta then measures added-minus-evicted (6p). Fit transforms on the base
   alone. When arms then differ in width, compare each block against a WIDTH-MATCHED null
   (shuffled-rows preferred: same marginals) as a paired per-image statistic — the width
   penalty is real, measured, and dataset-dependent.

3a. **Nulls need the same instrument audit as positives.** A null measured under an
   instrument that compresses the candidate harder than the baseline (fixed PCA on a wide
   concat vs a narrow single arm) is provisional until re-measured with the base protected
   and the candidate appended raw — the time-aggregation null stood for a month and
   reversed to +1.4 pts under the corrected harness (section 13).

3b. **"Adds beyond the base" needs BOTH raw delta > 0 AND the width-matched-null delta >
   0.** A purely redundant block also beats its shuffled null (junk hurts, a copy does
   not), so the null-corrected statistic alone only proves "image-linked, not junk" --
   section 11 measured +0.30 vs null at ~zero raw gain for fully-redundant blocks.

4. **Bootstrap unit = the independent sampling unit, i.e. the image.** (seed, fold)
   resampling re-partitions the same images and is anti-conservative (audit A1, 6o-B).
   Derived quantities inherit clustering: 42 ordered pairs per image = n≈images, not
   n≈pairs (6n-A, z inflated ~6.5x).

5. **Whitening/normalization fitted on few rows needs scale-relative regularization.**
   An absolute ridge (`cov + 1e-4*I`) against eigenvalues spanning 6 orders of magnitude
   regularizes nothing; held-out data comes out anisotropic and downstream gates read the
   covariate shift as signal (6o-D).

6. **Guards must be able to fail.** A pairing check on `labels` from sorted stratified
   indices passes for DISJOINT image subsets (labels are `[0]*n0+[1]*n1+…` regardless of
   seed, 6o-F). Pair on `subset_indices` + run dir. Test a new guard by feeding it the
   mismatch it exists to catch; a guard that has never fired in a test is unverified.

7. **Linear-span admissibility.** A probe on the states already spans every linear function
   of the states (increments, coarse curvature); "X adds information beyond the states" is
   only testable when X is OUTSIDE that span (solver curvature `pred_mid − pred` is; state
   differences are not). See 6g.

8. **Convergence / operating point must be matched across compared arms, and measured.**
   "Warnings affected all arms equally" is an assumption until the per-arm rate is printed
   (6p tracks `maxiter=%`). Never compare arms at each arm's own best-C.

## Cache identity & provenance

9. **A config default is not a pin.** run.py defaults feed `config_hash` for every task —
   changing one silently repoints every extraction run dir while probe scripts hardcode old
   hashes (6o item 1). Pins are enforced in the task (`PINNED_*`) and passed explicitly in
   sweep scripts.

10. **Every extractor stamps provenance through `env_provenance(cfg, honors=…)`.** Never
    hand-copy the stamp (three copies each dropped a guard, 6m/6o-8). `honors` declares
    which control env vars (FLUX_RANDOM_INIT / FIXED_COND_T / DEGRADE_TO) the caller's
    path actually implements; a set-but-unhonored flag must raise, not silently mislabel.

11. **CSV appends validate the header; cache loads validate identity** (ensemble size from
    meta.json, `subset_indices`, timesteps — not just shapes/labels).

## Process

12. **Cache per-image correctness vectors (results/*.npz) for any table that might need
    re-analysis.** The curvature table needed recomputing three times before this was done.

13. **Gates are executed code that prints PASS/FAIL, not prose.** Encode the §8 checklist
    into the script (see curv5000_probe: probe-sanity + detection-power) so the check runs
    every time, not just when someone remembers.

14. **Record everything in RESEARCH_NOTES.md** — findings (including nulls), retractions,
    direction changes, with dates; mark superseded numbers SUPERSEDED in place so stale
    values cannot be quoted from the notes.

15. **Smoke-test end-to-end before committing GPU** (`--max-samples`, strided so all
    classes appear). The finetune harness had 4 stacked crash bugs that 45 minutes of
    smoke runs caught one stage at a time.

## External claims

16. **A published table is prose, not data — verify the shipped artifact.** GEO-Bench's
    paper says m-eurosat has 2,000 train samples; the shipped default partition contains
    16,200, and SatDiFuser's paper repeats the stale 2,000 while its own loader provably
    consumes 16,200 (no partition arg -> package default). Before quoting any external
    number's protocol (split sizes, bands, label budget), download the actual artifact
    (partition file, config, loader source) and count; a val/test count that doesn't
    match the shipped file (1,000 vs 996) is the standard tell of a copied table.
