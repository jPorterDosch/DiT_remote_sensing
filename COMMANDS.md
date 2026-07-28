# Paired 500-image EuroSAT extraction — run book

Three extractions on the **same 500-image stratified train subset**, differing only in
extraction mode / guidance. Nothing here has been executed; you run everything via sbatch.

| job | script | mode | guidance | NFE/image | cache filename |
|---|---|---|---|---|---|
| 1 | `experiments/extract_500_inversion_g1.0.sbatch` | inversion (RF-Solver, 50 steps) | 1.0 | ~59 | `multistep_train_feats_inversion_g1.0_n50.npz` |
| 2 | `experiments/extract_500_oneshot_g1.0.sbatch` | one-shot | 1.0 | 7 | `multistep_train_feats_oneshot_g1.0.npz` |
| 3 | `experiments/extract_500_oneshot_g3.5.sbatch` | one-shot | 3.5 | 7 | `multistep_train_feats_oneshot_g3.5.npz` |

Shared settings: `--t 100 180 260 340 420 500 580 --k 28 --subset-size 500 --subset-seed 42
--seed 42`, EuroSAT train split, batch 1. Both one-shot jobs pin `--model.ensemble-size 1`
and `--eps-seed 42`, so the per-image eps is **identical between jobs 2 and 3** (only
guidance differs). Caches + `..._meta.json` land under `models/paired_500/<run_name>/`.

## How subset identity is guaranteed

`ExtractionTask` selects images via `_stratified_indices(labels, subset_size, subset_seed)`
([src/tasks/extraction.py](src/tasks/extraction.py)) — a pure function of:

1. **dataset order** — `EuroSATDataset.samples` is built from `EUROSAT_CLASSES` in fixed
   order with **sorted filenames** per class, so it is identical for identical dataset
   directory contents;
2. **`subset_size=500`, `subset_seed=42`** — identical CLI values in all three scripts;
3. `np.random.default_rng(seed)` — stable across NumPy versions for these operations.

All three jobs therefore compute the *same* indices, and `shuffle=False` extraction
preserves that order. This is also the selection logic of the original one-shot cache
(the extraction task uses your original `_stratified_indices` verbatim), so a legacy
cache made with `subset_seed=42` on the same dataset directory selected the same images.
Each cache additionally **embeds** `subset_indices` and `paths`, and the verification
step asserts they are equal across caches — identity is checked, not assumed.
Caveat: identity only holds if every run sees the same `EuroSAT_RGB` directory contents
(same files); run all jobs against one dataset copy.

## 0. Before submitting

- SBATCH headers are pre-filled for the **ai-tenn** partition (`-A isaac-utk0256`,
  `--partition=ai-tenn`, `--qos=ai-tenn`, `--gpus-per-task=1`, `--cpus-per-gpu=16`,
  `--time=24:00:00`), account/partition/qos/GPU copied verbatim from the sibling
  `experiments/sweep_eval_eurosat.sh` / `probe_ablation.sh`. Time is set to 24h; actual
  runtime estimates (~4h inversion, ~40min one-shot) are in each script's header comment,
  so you can lower `--time` to backfill sooner if you like. Change the account/partition
  block if submitting elsewhere.
- `mkdir -p logs` in the repo root (SLURM does not create output dirs).
- If compute nodes have no internet: uncomment `export WANDB_MODE=offline` in the scripts.
- Off-ISAAC only: uncomment the `FLUX_DEV` / `AE` exports (util.py defaults to ISAAC paths).

## 1. Smoke test first (4 images)

REPORT.md suggested a 4-image smoke test via `--max-samples`; the extract task subsets via
`--subset-size` instead — `--subset-size 4` selects one image from each of the first 4
classes (~4 x 59 forwards ≈ 3–5 min for inversion on an A6000). Uses a separate
`--save-dir models/smoke` so the real cache root stays clean:

```bash
# inversion smoke
python run.py --task extract --dataset.name eurosat --dataset.path data/eurosat/EuroSAT_RGB \
    --save_dir models/smoke --extraction_mode INVERSION --guidance_scale 1.0 \
    --num_inversion_steps 50 --t 100 180 260 340 420 500 580 --k 28 \
    --subset_size 4 --subset_seed 42 --seed 42 --batch_size 1

# one-shot smoke (seconds)
python run.py --task extract --dataset.name eurosat --dataset.path data/eurosat/EuroSAT_RGB \
    --save_dir models/smoke --extraction_mode ONESHOT --guidance_scale 1.0 \
    --t 100 180 260 340 420 500 580 --k 28 --model.ensemble-size 1 \
    --subset_size 4 --subset_seed 42 --seed 42 --eps_seed 42 --batch_size 1
```

On ISAAC, wrap in sbatch (no direct GPU runs on login nodes):
`sbatch -A isaac-utk0256 --partition=ai-tenn --qos=ai-tenn --gpus-per-task=1 --time=00:30:00 --wrap "<command>"`.

Expected: `feats (4, 7, 3072)` printed, cache + meta.json under `models/smoke/<run_name>/`.
Check the smoke caches with
`python experiments/verify_paired_caches.py --expect-n 4 models/smoke/*/multistep_train_feats_*.npz`.

## 2. Submit the extraction jobs

```bash
mkdir -p logs
sbatch experiments/extract_500_inversion_g1.0.sbatch
sbatch experiments/extract_500_oneshot_g1.0.sbatch
```

**Job-3 decision (one-shot g=3.5).** I could NOT confirm your existing g=3.5 cache is
reusable from this working tree: no 500-image cache is present locally, the original
extraction.py recorded **no guidance value** in its npz (guidance was hard-coded 3.5 at
the time, but that is presumption, not provenance), and if it was made with the
`ensemble_size=8` default it is a different object than these ensemble-1 arms. So job 3
is included by default. To decide:

```bash
python experiments/verify_paired_caches.py --legacy /path/to/existing/multistep_train_feats.npz
```

- prints `LEGACY MATCHES` (same subset_indices/paths/t/k AND ensemble_size==1) → skip job 3;
- anything else → `sbatch experiments/extract_500_oneshot_g3.5.sbatch`.

Note: run.py fail-fasts if a run's save dir already exists — to re-run a job with
identical config, `rm -rf models/paired_500/<that run_name>` first.

## 3. Verify the caches are shape-correct and image-paired

```bash
python experiments/verify_paired_caches.py
```

Asserts every cache under `models/paired_500/` has feats `(500, 7, 3072)`, mods
`(7, 3, 3072)`, labels `(500,)`; asserts `subset_indices`, `paths`, `timesteps`,
`block_idx`, `labels` are identical across all caches; prints the 500 subset image IDs
(add `--quiet-ids` for a shortened list). Exit code 0 = paired and shape-correct.

## 4. Analysis — three tiers + the B2 gate (after caches verify)

Login-node vs sbatch: the **linear probes are light → login node**; **B2 (transformer
training) → sbatch**. All three tiers evaluate by 5-fold stratified CV on the 500-image
cache (B2's protocol), so they are directly comparable.

### Tiers 1-2 + delta supplement — `experiments/linear_probes.py` (login node)

```bash
python experiments/linear_probes.py --out-csv results/linear_probes.csv
```

numpy/sklearn only (no torch/GPU/FLUX). Globs `models/paired_500/*/multistep_train_feats_*.npz`,
one arm per cache, and prints:
- **Tier 1** per-timestep CV accuracy (arm × t) — the accuracy-vs-t curve, your cheap
  early signal. If inversion doesn't beat one-shot anywhere here, the aggregates won't save it.
- **Tier 2** concatenated-trajectory probe, one L2 penalty swept once and frozen across all
  arms (starred column = headline), plus a robustness table over the penalty grid.
- **Delta** consecutive-increment probe (ordering-adjacent, parameter-free).

It also prints `best single t (inversion)` — **use that as `BEST_T` for B2** (its `mlp`
content-baseline defaults to t=260, which was measured on the one-shot cache).

### Tier 3 / gate — B2 transformer traj vs shuffle vs mlp — `experiments/b2_traj_readout_inversion.sbatch`

Self-test first (login-node, seconds, CPU) — the permutation-sensitivity gate. Do NOT
sweep until it prints `self-test OK`:

```bash
CACHE=$(ls models/paired_500/*/multistep_train_feats_inversion_g1.0_n50.npz)
python experiments/traj_readout.py --cache-path "$CACHE" --self-test
python experiments/traj_readout.py --cache-path "$CACHE" --arm traj --dry-run   # shapes + traj/mlp param parity
```

Then the sweep on the inversion cache — `{traj, mlp, shuffle} × {raw, normalized} × 5 seeds`
= 30 CV runs, appended to `results/b2_inversion_g1.0.csv`. The script re-runs the self-test
as a gate and aborts if it fails:

```bash
mkdir -p logs results
BEST_T=<best t from Tier 1> sbatch experiments/b2_traj_readout_inversion.sbatch
```

`traj > shuffle` (beyond the shuffle-seed spread) is the ordering signal — the gate the
other two tiers only contextualize. `traj` vs `mlp` (parameter-matched) is trajectory vs
best-single-timestep content.

### Eval protocol — CV-only (this sweep)

All tiers evaluate by 5-fold stratified CV on the 500-image cache. No held-out test cache
is used: this is an exploratory sweep, and B2 uses CV regardless, so CV-on-500 keeps every
tier on one comparable protocol. If a held-out number is wanted later it's a separate step —
`ExtractionTask` is train-split-only with a hardcoded `multistep_train_feats_*` name
(`src/tasks/extraction.py:78`), so a test cache would need a split selector + retag on that
shared task plus per-arm GPU extraction jobs. Not built.

## Caveats / provenance

- **What is paired across caches:** the images (all three jobs) and the per-image eps
  stream (jobs 2 vs 3, same `eps_seed`, same visit order). **What is not:** the VAE
  clean-latent draws — `ae.encode` samples the posterior from the global RNG per run
  (repo-wide convention, flagged in REPORT.md). If you ever want bit-identical latents
  across arms, the `sample=False` (posterior mean) switch discussed in REPORT.md is the
  lever; not changed here.
- Runtime math: inversion = 2 evals x 29 chain steps + 1 early-exited feature pass
  ≈ 59 NFE/image (~25–30 s/image on an A6000-class GPU → ~4 h for 500); one-shot
  = 7 NFE/image (~30–40 min for 500). Suggested wall times carry ~2x headroom.
- Every job logs mode, guidance, k, t, num-inversion-steps (inversion), eps-seed/ensemble
  (one-shot), subset size/seed to the wandb config (`extraction/*` keys) and writes the
  same settings to `multistep_train_feats_<tag>_meta.json` beside each cache; only
  settings actually applied in that mode are recorded.
