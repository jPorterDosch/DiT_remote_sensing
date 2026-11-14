#!/bin/bash
# ============================================================================
# eval/seg_probe.py run: the per-token linear heads of every (model x fraction x seed) cell of one
# dataset from the token caches, GPU (a cell's standardized fp16 train tokens live on the A40, the
# fp32 val / test tokens on the host, <= 34 GB). Then
# `python -m eval.seg_probe report --spec <SPEC>` on a login node (reads cells only).
#
#   SPEC    eval/seg_specs/phase1.json;  MODELS  optional subset ("flux-pretrained dinov3-sat")
#   sbatch --array=0-2 experiments/isaac/seg/probe.sbatch
# ============================================================================
#SBATCH --job-name=seg-probe
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=160G
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE=disabled
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
set -euo pipefail
DATASETS=(m_pv4ger_seg m_chesapeake_landcover m_SA_crop_type)
IDX="${SLURM_ARRAY_TASK_ID:?submit with --array=0-2}"
DATASET="${DATASETS[$IDX]}"
SPEC="${SPEC:-eval/seg_specs/phase1.json}"
ARGS=(run --spec "$SPEC" --datasets "$DATASET" --out-dir "${OUT:-results/seg}")
# shellcheck disable=SC2206
[ -n "${MODELS:-}" ] && ARGS+=(--models ${MODELS})
[ "${REDO:-0}" = 1 ] && ARGS+=(--redo)
python3 -m eval.seg_probe "${ARGS[@]}"
echo "SEG_PROBE_OK $DATASET"
