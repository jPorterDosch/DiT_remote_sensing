# Official GEO-Bench probing protocol

The exact protocol behind every number in `results/eval/` reported as "official". Each result
file records these constants in its `config` (`protocol_constants`), so this document describes
the code; if they disagree, the recorded config and the code win and this file is stale.
Last updated 2026-10-12.

## Data

GEO-Bench classification tasks, RGB, the shipped `default_partition.json` (counted from the
artifact, not the paper — rule 16), with every file checked image-by-image against
`eval/manifests/<task>.json` at load time.

| task | train | val | test | label type | metric |
|---|---|---|---|---|---|
| m_eurosat | 16200 | 996 | 996 | single | top-1 (+ macro-F1) |
| m_forestnet | 6464 | 989 | 993 | single | top-1 (+ macro-F1) |
| m_so2sat | 19992 | 986 | 986 | single | top-1 (+ macro-F1) |
| m_brick_kiln | 15063 | 999 | 999 | single | top-1 (+ macro-F1) |
| m_pv4ger | 11814 | 999 | 999 | single | top-1 (+ macro-F1) |
| m_bigearthnet | 20000 | 1000 | 1000 | multi-label (43) | micro-F1, threshold p > 0.5 |

All training images are used (no label budget).

## Selection rule (all heads)

1. Every hyperparameter, view, timestep, block and epoch is chosen on the **official val split**.
2. Ties: the first maximum wins, in the stored candidate order.
3. Then **one** test evaluation of the selected configuration. In sweeps, every cell's test
   predictions are stored sealed, and `SealedTests` refuses to open a second one.
4. Before test is opened, `eval.sweep select` runs a **positive control**: the selected val
   score must reach 0.25 of the headroom above the best constant predictor,
   `(val − chance) / (1 − chance) ≥ 0.25` (`control_skill`, `CONTROL_MIN_SKILL`). On a FAIL the
   test vector stays sealed. The 0.25 floor was set after the six tasks' values were seen; it is
   an instrument floor, not a tuned cut.
5. Paired comparisons use a bootstrap over **test images** (10,000 resamples, 95% percentile CI,
   `protocols.paired_delta`). For micro-F1 the metric is recomputed on every resample.

## Features

- **FLUX.1-dev (frozen), one-shot noising, 224 px:** `x_t = (1−t)·x0 + t·ε` at
  t ∈ {100, 180, 260, 340, 420, 500, 580, 740, 900} (/1000), guidance 1.0, null prompt,
  ensemble of 8 ε draws averaged. Feature at block k = the block's input image tokens, plus its
  adaLN modulation (`img_mod1` for double-stream blocks; stamped `double_block_feature`),
  mean-pooled over tokens: 3072-d. Multi-block caches hold blocks
  {4, 8, 12, 18 (double); 19, 24, 28, 33, 38, 43, 48, 54, 55, 56 (single)}. ε seeds differ
  across splits (train 42, val 44, test 43) and the loader refuses a shared seed.
- **DINOv3 ViT-7B/16 (frozen, `web` and `sat` presets):** 224 px bicubic, the preset's own
  normalization. Stored per image: `cls` (last-block CLS), `mp` (last-block patch mean) and
  `cls4` (CLS of the last 4 blocks). These are exactly the inputs of DINOv2/v3's own linear eval
  (`create_linear_input(use_n_blocks ∈ {1, 4}, use_avgpool)` in `dinov3/eval/linear.py`,
  checked in the shipped hub copy 2026-10-12; DINOv3's default is `n_last_blocks_list = (1,)`).

## Head `lr`: logistic regression (default)

- Candidates:
  - FLUX: each single timestep, and the concatenation of all 9 (27,648-d).
  - DINO: `cls`, `clsmp`, `cls4`, `cls4mp` (`features.DINO_OFFICIAL_VIEWS`).
- `StandardScaler` fitted on train, per candidate.
- `sklearn.linear_model.LogisticRegression(C, max_iter=3000)` (lbfgs).
- Multi-label: one binary LR per label (`MultiLabelLR`), predicting p > 0.5.
- C grid:
  - `wide` = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1} (`C_GRID_WIDE`), used for every current
    FLUX and DINO number;
  - the legacy grid {0.01, 0.1, 1} is kept only for the reproduction gates.
- Convergence warnings are counted per fit and recorded.
- Sweeps select jointly over (block, candidate, C) on val.

## Head `tsweight`: learned slot weighting (`eval/fusion.py`)

SatDiFuser-style global weighting on **pooled** features. It matches SatDiFuser's weighting
idea and training recipe, not their architecture (they fuse spatial maps through a learned
384-d projection before pooling).

- **Slots:**
  - FLUX: the 9 timesteps at one block (timestep weighting).
  - DINO: `cls` of blocks −4…−1, plus `mp` (layer weighting over depth). These are the
    same 5 vectors DINOv3's linear eval concatenates at `use_n_blocks=4, use_avgpool=True`.
- **Model:** standardize each (slot, channel) with train statistics; α = softmax(θ), with θ
  initialized at 0 (uniform); h = Σ_s α_s x_s; logits = W h + b.
- **Loss:** cross-entropy (single-label) or BCE with p > 0.5 (multi-label).
- **Training:** AdamW (decoupled weight decay), 40 epochs, batch 256, linear warmup over 5 epochs
  then cosine, stepped per batch. Seed 0, fixed data order.
- **Selection:** (lr, wd, epoch) on val, with the val metric evaluated after every epoch. The
  selected epoch's weights are evaluated on test once.
- **Grids** (`--fusion-grid`, recorded as `protocol_constants.fusion.grid`):

| grid | lr | weight decay | decay applied to | status |
|---|---|---|---|---|
| `satdifuser` | {1e-3, 1e-2} | {0, 1e-4} | all parameters | first run, 2026-10-12; SatDiFuser's lr 0.01. Their weight decay is **not verified** against their released code |
| `reg` | {1e-3, 1e-2} | {0, 1e-2, 1e-1, 1, 10} | linear layer only (θ undecayed) | fixed a priori 2026-10-12, identical for every task and backbone |

- **Why `reg` exists.** Under `satdifuser`, wd 0 and 1e-4 gave identical curves (no effective
  regularization). On the hard tasks the head selected epochs 3–4 with near-uniform weights,
  i.e. it overfit before the weights learned, while the LR head chose C = 1e-4 on the same
  tasks.
- **The `reg` weight-decay range.** AdamW shrinks the weights by about exp(−lr·wd·steps). With
  ~1,000–3,200 steps per run, the range spans no shrinkage to heavy shrinkage, which is the role
  the C range plays for the LR head (matched operating point, rule 8).
- **Why θ is not decayed.** Decaying θ pulls the mix toward uniform, which would regularize the
  fusion itself away (tests/test_fusion.py checks this).
- **Primary learned-weighting grid: `reg`** (decided 2026-10-12, after both grids had run). It
  is the grid with regularization search matched to the LR head's C range (rule 8). `satdifuser`
  stays on record and in the appendix: neither grid replaces the other, and neither is deleted.
  FLUX is insensitive to the choice (all six tasks within ±0.4); DINO is not (web forestnet +3.3,
  so2sat +1.7 / +1.9 under `reg`).
- **Main-text comparison** of FLUX vs DINO uses the `lr` head on the wide C grid. Learned weighting
  is a secondary analysis, under `reg` for both backbones.
- **Known limitation:** wd = 10, the top of the `reg` grid, was selected in 7 of 18 runs (mostly
  DINO; for FLUX only m-forestnet). The optimum may lie past the grid. Not extended after seeing
  test; any extension is a new, pre-registered, separately labelled arm.

## Arms and result files

| arm | head / grid | files |
|---|---|---|
| `flux-oneshot-ens8-mb` | lr, wide C | `results/eval/sweep_mb/sweep-{block,select}_<task>_flux-oneshot-ens8-mb_*` |
| `flux-oneshot-ens8-mb-tsw` | tsweight / satdifuser | `…/sweep_mb/…_flux-oneshot-ens8-mb-tsw_*` |
| `flux-oneshot-ens8-mb-tsw-reg` | tsweight / reg | `…/sweep_mb/…_flux-oneshot-ens8-mb-tsw-reg_*` |
| `dinov3_vit7b16_<preset>` | lr, wide C | `results/eval/probe-official_<task>_dinov3_vit7b16_<preset>_*` |
| `dinov3_vit7b16_<preset>-tsw` | tsweight / satdifuser (layer weighting; arm name predates the rename) | `…_<preset>-tsw_*` |
| `dinov3_vit7b16_<preset>-slotw-reg` | tsweight / reg | `…_<preset>-slotw-reg_*` |

## Test-use count

Each arm × task above is one use of that task's test split. Count them in RESEARCH_NOTES
(workstation). As of the `reg` run: FLUX m-eurosat #7; every DINO (preset, task) pair #4
(3-C grid, wide grid, tsweight/satdifuser, tsweight/reg).
