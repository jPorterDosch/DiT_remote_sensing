# Training on Git-10M after the geographic exclusion

Pipeline (2026-10-12). The dedup manifests (`experiments/isaac/geo_dedup/PROTOCOL.md`) decide
which rows exist for training; nothing else in the training path can reach an excluded row.

| Step | Code | Output on `$STORE` |
|---|---|---|
| 1. download + convert the 787 Hub stream shards to random-access arrow (64-row batches, JPEG bytes untouched, checked row-by-row against the dedup metadata) | `convert.sbatch` -> `python -m datasets.git10m convert` | `git10m/arrow/data-NNNNN-of-00787.arrow` (390 GB converted (787 arrow files, measured 2026-10-13)) |
| 2. index = the training rows, ONE ROW PER IMAGE | `build_index.sbatch` -> `python -m datasets.git10m build-index --manifest-dir $STORE/geo_dedup/dedup_v2_b1000 [--include-unassessable]` | `git10m/index_retained_unique.npz`, `git10m/index_all_unique.npz` (from `dedup_v2_b1000`); `index_retained_cities_unique.npz`, `index_all_cities_unique.npz` (from `dedup_v3_b1000_cities`, m-so2sat city boxes, PROTOCOL.md §7b); `index_smoke_s0_unique.npz` (13,347). Rows (duplicate rows dropped): index_retained_unique 7,705,674 (333,394), index_all_unique 9,412,090 (672,764; 7,705,674 retained + 1,706,416 unassessable), index_retained_cities_unique 6,711,575 (290,885), index_all_cities_unique 8,417,991 (630,255) -- job 6394029, 2026-10-13, `.report.json` next to each. The 2026-10-12/13 files without `_unique` (manifest rows: 8,039,068 / 10,084,854 / 7,002,460 / 9,048,246) predate the rule and are refused by the dataset. Manifest sha256 inside each |
| 3. train FM + SEM (fm / naive / proj arms, per-block projection, fp32 LoRA params, executed gates) | `train.sbatch` -> `run.py --task finetune-diffusion --dataset.name git10m --align.arm <arm>` (the pipeline's training task; alignment in `src/tasks/align.py`, settings in `config_types.AlignConfig` / `Git10MConfig`) | `models/git10m/git10m_flux_<hash>+<seed>+<arm>-<8hex>/` (ckpt.pt, checkpoints/lora_{best,last}_*.pt, gen/, stats.npz, val.jsonl, meta.json, conflict_gate.txt) |
| 4. evaluate a checkpoint | `extract_ckpt.sbatch` (`CKPT=<run>/checkpoints/lora_last_*.pt --array=0-5`) -> `run.py --task extract --lora-checkpoint ... --k <the frozen caches' 14 blocks> --t 100 ... 900 --model.ensemble-size 8` per GEO-Bench task (the ordinary multi-block one-shot extraction, flag for flag the frozen `*_multiblock_ens8` one; the cache is named and stamped `_lora-<arm>` / `weights: "flux-dev+lora:<arm>"`), then `python -m eval.matrix run --spec eval/matrix_specs/phase2_template.json` with the printed entries: every FLUX arm at the fixed aligned block k=28 AND each arm at its own val-selected block (`eval.sweep select` on its caches, as phase 1 did for the frozen model; see the spec's `_comment`) | `models/git10m_eval/<run>/<dataset>/*/multistep_{train,val,test}_feats_oneshot_g1.0_lora-<arm>_k<blocks>.npz`, `results/matrix/` |

Decisions:

- **Which rows:** the `INDEX` env var; the real runs use `index_all_cities_unique.npz`
  (decision 2026-10-14). `_all_` keeps the 2,045,786 rows whose sources publish no
  coordinates -- the mentor's decision (2026-10-13): keep them and measure the residual overlap
  by content (PROTOCOL.md §5, §8) rather than drop 17 % of the unique images; `_retained_`
  drops them. `_cities_` additionally excludes the 1.04 M rows inside So2Sat's 52 city ROIs
  (§7b; adopted 2026-10-14 so m-so2sat is covered at the only resolution its samples can be
  located at). Switching is a flag, not a code change. Every run's `meta.json` carries the index path, the manifest sha256s,
  `include_unassessable` and the per-bucket counts.
- **What is aligned (`--align.feat`, default `ditf`):** the block-28 input tokens in their DiTF
  form -- massive-activation channels 154/1446 zeroed, LayerNorm, the block's own adaLN
  `(1 + scale) * x + shift` -- i.e. the block's actual modulated input per token
  (`FluxModel.extract`'s definition). `ln` = zeroed channels + LayerNorm without the scale/shift
  (isolates the massive-channel removal from the modulation); `raw` aligns the plain tokens as
  REPA does. NOTE: the
  official matrix probes (`eval/probe.official_flux`) read the RAW pooled block-input tokens
  (per-t slices + concat, StandardScaler + logistic regression), not the DiTF view, so the
  aligned object and the probed object differ by the per-token LayerNorm + adaLN and the two
  zeroed channels; `ditf` is a choice about what to align, not probe parity (corrected
  2026-10-15; an earlier version of this note claimed parity). ABLATION TO RUN IF TIME (user note
  2026-10-14): `ALIGN_FEAT=raw` vs `ditf` -- if the massive activations carry generation and the
  remaining channels can take the semantics, the ditf arm should keep generation (held-out FM loss,
  fixed-seed samples) while matching the raw arm downstream; the raw arm pushes semantics through
  the massive channels too.
- **Timesteps (`--t`, `--align.t-weights`):** the 9 timesteps the matrix probes, sampled per example
  with weights `1 1 1 1 1 1 1 0.5 0.5` (train.sbatch `T_WEIGHTS`): 740/900 get half the share of the
  seven semantic timesteps. Most FLUX LoRA trainers sample continuous t (SD3 logit-normal, often
  shifted) rather than a subset; this grid keeps the train/probe correspondence exact
  (`tasks.train_diffusion._sample_timesteps`). Validation and the measurement passes use fixed grids.
- **Generation metric (`--align.fid-n`, train.sbatch `FID_N`, default 600):** FID and KID
  (`eval/gen_metrics.py`: torchvision Inception-v3 pool features, 299 px bilinear; KID = unbiased
  cubic-kernel MMD^2 over 100 half-set subsets, read it first: FID at n = 600 < 2048 feature
  dims is biased and only comparable at equal n) between `fid_n` fixed-seed
  samples (seeds `gen_seed + 1000 + batch`, the grids' sampler settings) and the first `fid_n`
  strided rows of the val holdout (656 rows under the default VAL_INDEX, so FID_N <= 656;
  unflipped, 224 px) -- the SAME reference images in every arm,
  sha1 recorded in `gen_metrics.json` / `meta.json` (`gen_metrics_ref_sha1` must agree across
  arms). Computed at the final checkpoint; `FID_STEP0=1` on ONE arm also scores the pretrained
  model (identical in every arm). Values are comparable within this project (same extractor,
  reference, n), not to published FID tables (different Inception weights). Standalone:
  `python -m eval.gen_metrics --real <run>/gen/fid_ref --fake <run>/gen/fid_final` (the samples and
  the reference are saved as PNGs; a metric failure is recorded under `error`, never fatal, and
  every other deliverable is written BEFORE the final generation). Cost: fid_n x gen_steps sampler
  steps (600 x 28 at 224 px per evaluation).
- **Augmentation:** horizontal flip only (`--hflip 1`, p = 0.5; the SEM target is computed on the
  flipped image). The rest of the transform is the repo's: bicubic resize to 224, [-1, 1].
  `git10m.hflip` is honored only with `dataset.name=git10m` (run.py refuses it elsewhere); the
  m-eurosat alignment runs (`experiments/isaac/grad_conflict/align_m_eurosat.sbatch`) have no augmentation.
- **One row per image.** Git-10M repeats a public-dataset image on adjacent rows, one row per
  caption, identical JPEG bytes (1,243,614 of the 10,084,854 retained + unassessable rows).
  Training is unconditional on the image, so `build-index` keeps the first row of every
  `img_name` and records `duplicate_rows_dropped`. Before this rule (found in review
  2026-10-13) a repeated image had double sampling weight, `train_size` overstated unique images
  by ~7 %, and 125 of index_all's 1,006 holdout rows had a byte-identical twin in the training
  set. The dataset refuses an index built without the rule.
- **Holdout:** every 10,000th global row id (shard x 2^20 + row; `--git10m-holdout-every`) of
  the RETAINED bucket is validation, never trained on -- a property of the row, and the kept row
  of a repeated image is the same under every manifest, so every index (with or without the
  unassessable rows, with or without the city pass) validates on the same image set, and a
  validation image has no twin in any index's training rows. `val.jsonl` reports flow + align
  loss on a strided subset of it. The downstream probe still uses the GEO-Bench task given by
  `--dataset` (features extracted at the end).
- **Val set across indexes:** `--git10m-val-index` (train.sbatch `VAL_INDEX`, default
  `index_retained_cities_unique.npz`) names the index whose retained holdout is the val set for
  every run, whatever `INDEX` is; without it each index validates on its own retained holdout
  (758 rows for the v2 indexes, 656 for the city ones: not comparable, review 2026-10-13).
- **Frozen baseline:** the pretrained-FLUX caches already in `results/matrix` (`flux-pretrained`,
  the same extraction code with no adapter; the phase-2 spec's fixed-k entry is named
  `flux-pretrained-k28` because cells are keyed by model name and must not overwrite phase 1's). Cross-arm determinism gate: every arm's `meta.json`
  records `gen_sha_step0`, the sha1 of the fixed-seed step-0 sample (LoRA B = 0 = the pretrained
  model); the arms must agree on it. The matrix scores every arm against the same teacher under
  the same protocol (`eval/matrix.py`).
- **RNG streams:** data order (PermutationSampler), t, eps, the VAE posterior noise (`ae_rng`)
  and the per-row hflip decision (seed, row) are all explicit and in the checkpoint, so a resumed
  run and every arm consume the same randomness.
- **No epochs.** `meta.json` reports `samples_seen = steps x batch x accum`, `train_size`
  (unique images in the training split) and `unique_fraction_seen`. Quote those, not epochs.
  (CLAUDE.md: budget by unique exposures.)
- **Resume:** `--align.ckpt-every N` writes `ckpt.pt` (fp32 LoRA, projector + its optimizer,
  LoRA optimizer, scheduler, stats, t / eps / VAE-posterior generator states, step, and a resume
  identity: run name, arm, batch, accumulation, seed, lr, index provenance --
  `tasks.align.resume_identity`); `RESUME=<run>/ckpt.pt` continues the same run name at the same
  microbatch (`PermutationSampler`: seeded permutation per pass, positioned at the consumed
  count) and refuses any identity mismatch (a changed batch size would silently re-position the
  sampler). The flip is a per-row function of (seed, row), so it needs no state. ai-tenn MaxWall
  is 3 days.
- **Throughput sizing:** `SIZE=1 BATCH_SIZE=... GRAD_ACCUM=...` runs 30 steps and prints img/s
  and peak memory (m-eurosat finetune reference: ~2 img/s at batch 1 on an H100, estimate).

Smoke before anything else (shard 0 only, `index_smoke_s0_unique.npz`; a shard-subset index is
refused unless `--max-samples` caps the run):

    sbatch --export=ALL,SMOKE=1,ARM=proj experiments/isaac/git10m/train.sbatch

Tests: `tests/test_git10m.py` (synthetic shard: layout, guards, index, transform, holdout,
sampler resume), `tests/test_align.py` (every arm's gradient combine against plain autograd on a
57-block stand-in, the verifiers, the resume identity, the rule-9 run-name invariants, the
dataset wrapper), `tests/test_grad_conflict.py` (projection / fp32 gate numerics). Verified on
real shard 0 (2026-10-12): every row's img_name matches the dedup metadata; loader 1,800 img/s
warm with 4 workers. The GPU path (`finetune-diffusion` with `align.arm` on real data) is
verified by the smoke job only.

Shuffled-teacher control (built 2026-10-16): `TARGET_SHUFFLE=1` on the same `ARM` / `ALIGN_FEAT` /
`SEED` as the arm it controls (`--align.target-shuffle batch`: the DINO targets of each microbatch
rolled by one, so every image is aligned to an unrelated scene's tokens; `tasks/align.py`
`align_loss`). Its checkpoint goes through the same extraction + matrix; a downstream gain in the
control is an instrument failure, not a result (CLAUDE.md rule 2).

Not built (spec items, TODO comments in the code): the linear projector (`tasks/align.py`), the
two-point learning-rate check (two `LR=` submissions). The distributional generation metric (FID/KID) was added 2026-10-14.
