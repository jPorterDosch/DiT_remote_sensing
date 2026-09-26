#!/bin/bash
# ============================================================================
# t x k SWEEP on a registered official-split dataset (frozen FLUX, ens8, 7-t grid, g=1.0),
# one array task per block. The dataset is a parameter; nothing below is task-specific.
#   sbatch --export=ALL,DATASET=m_eurosat,REUSE_K=28,REUSE_DIR=models/m_eurosat_oneshot_ens8 \
#       experiments/isaac/sweep/sweep.sbatch
#   sbatch --export=ALL,DATASET=<any eval/features.OFFICIAL key> experiments/isaac/sweep/sweep.sbatch
# Each task runs experiments/isaac/sweep/sweep_block.sh <k>: extract every official split
# at block k -> eval.sweep block -> delete the caches. Then, on a login node:
#   python3 -m eval.sweep select --dataset $DATASET --arm flux-oneshot-ens8 \
#       --blocks "results/eval/sweep/sweep-block_${DATASET}_flux-oneshot-ens8-k*.npz"
#
# Blocks: single-stream only, k in {19,24,28,33,38,43,48,54}. Double blocks (0-18) are
# excluded: extraction does not support them (no adaLN mod in up_ft) and their feature is
# a mid-block post-modulation activation, a different definition from the single blocks'
# block INPUT (RESEARCH_NOTES 9.6).
#
# TIME (m-eurosat, 18,192 images): one-shot early-exits at block k, so cost ~ (k+1)/29 x
# the measured k=28 rate (4.99 s/img A6000, ~2 s/img H100): ~7 h (k=19) ... ~19 h (k=54),
# plus ~1-2 h CPU probing. Worst task ~21 h, so 30 h leaves ~1.4x margin; ~92 H100 GPU-h
# for 7 extracted blocks. Other datasets scale with their image count -- recheck --time.
# STORAGE: ~1.6 GB of caches per running task (m-eurosat) on scratch, deleted after probing.
# ============================================================================
#SBATCH --job-name=geobench-sweep
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=30:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # caches + models/data/logs on Lustre scratch

source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -f "$PROJECT_ROOT/.secrets/wandb-personal.env" ] && source "$PROJECT_ROOT/.secrets/wandb-personal.env"
export WANDB_MODE="${WANDB_MODE:-offline}"
export DATASET="${DATASET:?submit with --export=ALL,DATASET=<OFFICIAL key>}"

KS=(19 24 28 33 38 43 48 54)
K="${KS[${SLURM_ARRAY_TASK_ID:?run as an array job}]}"
echo "sweep $DATASET task ${SLURM_ARRAY_TASK_ID} -> block k=$K"
bash "$PROJECT_ROOT/experiments/isaac/sweep/sweep_block.sh" "$K"
