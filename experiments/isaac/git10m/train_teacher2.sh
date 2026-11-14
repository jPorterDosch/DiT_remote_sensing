#!/bin/bash
# Second-teacher repeat of the seed-42 proj and naive runs: DINOv3 ViT-7B/16 WEB (LVD-1689M)
# in place of SAT-493M. Everything else mirrors the core submit lines of 2026-10-17 (sacct
# 6407504 naive / 6407505 proj): index_all_cities_unique, 20000 steps, batch 2 x accum 4, ckpt
# every 500, seed 42, align weight 0.5, feat ditf, k=28, lr 1e-3, no FID at step 0. The only
# change is ALIGN_TARGET. AlignConfig.identity() carries every field but resume / ckpt_every, so
# the align hash in the run name moves and the runs land in NEW
# models/git10m/git10m_flux_<cfg>+42+<arm>-<hash> dirs (the sat runs are never touched).
#
# Purpose: does the flow-matching / semantic gradient alignment measured in the sat runs
# (stats.npz step/cos_pre, step/conflict, step/projected) hold for a second representation?
# The web-teacher reference rows already exist in the matrix (results/matrix/cells/dinov3-web).
# Extraction decision once both land: proj-web only, unless the proj-web projection rate departs
# from the sat run's 5.8 % of block-steps (then naive-web too).
#
#   DRY=1 bash experiments/isaac/git10m/train_teacher2.sh   # sbatch --test-only only
#   bash experiments/isaac/git10m/train_teacher2.sh         # test-only, then submit both arms
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
[ -e .git ] || { echo "FATAL: not at the repo root: $(pwd)" >&2; exit 1; }
STORE=/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing
INDEX="$STORE/git10m/index_all_cities_unique.npz"
WEB="$STORE/ditf_models/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
[ -f "$INDEX" ] || { echo "FATAL: $INDEX missing" >&2; exit 1; }
[ -f "$WEB" ] || { echo "FATAL: web teacher weights missing: $WEB" >&2; exit 1; }
mkdir -p logs
for ARM in proj naive; do
    EXPORT="ALL,ARM=$ARM,ALIGN_TARGET=dinov3_vit7b16_web,INDEX=$INDEX,TRAIN_STEPS=20000,BATCH_SIZE=2,GRAD_ACCUM=4,CKPT_EVERY=500"
    sbatch --test-only --export="$EXPORT" experiments/isaac/git10m/train.sbatch
    [ "${DRY:-0}" = 1 ] && continue
    sbatch --export="$EXPORT" experiments/isaac/git10m/train.sbatch
done
