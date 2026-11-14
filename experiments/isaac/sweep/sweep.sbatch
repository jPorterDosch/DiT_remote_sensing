#!/bin/bash
# ============================================================================
# t x k SWEEP on a registered official-split dataset (frozen FLUX, ens8, 7-t grid, g=1.0),
# one array task per block. The dataset is a parameter; nothing below is task-specific.
#   bash experiments/isaac/sweep/submit.sh <eval/features.OFFICIAL key>   (sizes --time, <= 72 h)
#   REUSE_K=28 REUSE_DIR=models/m_eurosat_oneshot_ens8 bash experiments/isaac/sweep/submit.sh m_eurosat
# Each task runs experiments/isaac/sweep/sweep_block.sh <k>: extract every official split
# at block k -> eval.sweep block -> delete the caches. Then, on a login node:
#   python3 -m eval.sweep select --dataset $DATASET --arm flux-oneshot-ens8 \
#       --blocks "results/eval/sweep/sweep-block_${DATASET}_flux-oneshot-ens8-k*.npz"
#
# Blocks: single-stream only, k in {19,24,28,33,38,43,48,54}. Double blocks (0-18) were
# excluded because extraction could not serve them (no adaLN mod in up_ft) and their feature was
# a mid-block post-modulation activation (RESEARCH_NOTES 9.6). [SUPERSEDED 2026-10-05: double
# blocks now yield the block INPUT + img_mod1, the single blocks' definition, stamped
# double_block_feature (config_types); multiblock.sbatch extracts k 4/8/12/18. This sweep's
# grid is unchanged for comparability with its banked results.]
#
# TIME: one-shot early-exits at block k, so cost ~ (k+1)/29 x the measured k=28 rate (4.99
# s/img A6000, ~2 s/img H100), plus CPU probing. submit.sh sizes --time per dataset from its
# image count (x1.5 margin) and refuses > 72 h (ISAAC limit); the #SBATCH value below is only
# the default for a direct sbatch. m-eurosat: ~7 h (k=19) ... ~19 h (k=54), ~103 H100-h.
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
# Pin the env: with --export=ALL the job otherwise inherits whatever python3 the SUBMITTING
# shell had on PATH (6315057-9 died on "No module named 'numpy'", 2026-10-02).
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
python3 -c "import sys, numpy, torch; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -f "$PROJECT_ROOT/.secrets/wandb-personal.env" ] && source "$PROJECT_ROOT/.secrets/wandb-personal.env"
export WANDB_MODE="${WANDB_MODE:-offline}"
export DATASET="${DATASET:?submit with --export=ALL,DATASET=<OFFICIAL key>}"

KS=(19 24 28 33 38 43 48 54)
K="${KS[${SLURM_ARRAY_TASK_ID:?run as an array job}]}"
echo "sweep $DATASET task ${SLURM_ARRAY_TASK_ID} -> block k=$K"
bash "$PROJECT_ROOT/experiments/isaac/sweep/sweep_block.sh" "$K"
