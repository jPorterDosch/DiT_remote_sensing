#!/bin/bash
# ============================================================================
# m-eurosat ONE-SHOT extraction (frozen FLUX, ens8, 7-t grid, k=28, g=1.0) — UNSHARDED.
# One submission per split (the three splits are separate runs, not shards):
#   sbatch --export=ALL,SPLIT=val   experiments/isaac/m_eurosat/oneshot.sbatch
#   sbatch --export=ALL,SPLIT=test  experiments/isaac/m_eurosat/oneshot.sbatch
#   sbatch --export=ALL,SPLIT=train experiments/isaac/m_eurosat/oneshot.sbatch
# (or use experiments/isaac/m_eurosat/submit.sh, which submits everything.)
#
# TIME: 7 t x 8 ensemble members = 56 DiT forwards + 8 VAE encodes per image.
#   Measured 4.99 s/img on an RTX A6000. ai-tenn is H100-80GB (~2.5-3x an A40, which is
#   itself ~5% slower than the A6000) => train 16,200 imgs ~= 8-9 h, val/test ~= 0.5 h.
#   The 30 h request is deliberately sized to survive ZERO speedup (A40 rate = 23.6 h)
#   while staying far below the 3-day ai-tenn wall, so 1-GPU backfill can still place it.
#   Tighten to ~12 h once the first run prints a real rate. NOTE: FLUX.1-dev needs ~34 GB
#   resident (23.8 GB transformer + T5-XXL) and is bf16 -> V100S/T4 partitions cannot run
#   this at all (32/16 GB, and bf16 needs sm_80+). Only A40 and H100 qualify on ISAAC.
#
# Seeds: eps/global seeds DIFFER per split (train 42, val 44, test 43) — position-paired
# noise reuse across splits is a structural coupling (RESEARCH_NOTES 6t, audit F1); the
# probe REFUSES caches whose eps seeds collide.
# ============================================================================
#SBATCH --job-name=meurosat-oneshot
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=30:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -f "$PROJECT_ROOT/.secrets/wandb-personal.env" ] && source "$PROJECT_ROOT/.secrets/wandb-personal.env"
export WANDB_MODE="${WANDB_MODE:-offline}"

SPLIT="${SPLIT:?set SPLIT=train|val|test via --export}"
case "$SPLIT" in
    train) SEED=42 ;;
    test)  SEED=43 ;;
    val)   SEED=44 ;;
    *) echo "bad SPLIT=$SPLIT" >&2; exit 1 ;;
esac

python3 "$PROJECT_ROOT/run.py" \
    --task extract \
    --model.name flux --model.ensemble-size 8 \
    --dataset.name m_eurosat \
    --dataset.path "$PROJECT_ROOT/data/m_eurosat_rgb" \
    --img-size 224 224 \
    --t 100 180 260 340 420 500 580 \
    --extraction-mode ONESHOT \
    --k 28 --guidance-scale 1.0 \
    --seed "$SEED" --eps-seed "$SEED" --subset-seed 42 \
    --extract-split "$SPLIT" \
    --label-fraction 1.0 --batch-size 1 \
    --save-dir "$PROJECT_ROOT/models/m_eurosat_oneshot_ens8"
