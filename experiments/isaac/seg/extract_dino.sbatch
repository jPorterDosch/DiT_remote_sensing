#!/bin/bash
# ============================================================================
# DINOv3-ViT-7B/16 (SAT-493M) patch tokens of one segmentation dataset at 512 px (32 x 32 tokens):
# eval/extract_dino.py --tokens --img 512, three splits, (N, L, 4096) float16 memmaps under
# models/seg_tokens/<dataset>/dinov3_vit7b16_sat/. The teacher row of the segmentation table.
#
#   SMOKE=1  --max-images 24 into models/seg_tokens_smoke/ (never quotable), run first
#   sbatch --export=ALL,SMOKE=1 --array=0 --time=1:00:00 experiments/isaac/seg/extract_dino.sbatch
#   sbatch --array=0-2 experiments/isaac/seg/extract_dino.sbatch
# ============================================================================
#SBATCH --job-name=seg-dino
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE=disabled
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
set -euo pipefail
DATASETS=(m_pv4ger_seg m_chesapeake_landcover m_SA_crop_type)
IDX="${SLURM_ARRAY_TASK_ID:?submit with --array=0-2}"
DATASET="${DATASETS[$IDX]}"
PRESET="${PRESET:-dinov3_vit7b16_sat}"
IMG="${IMG:-512}"   # token input side; 224 = the alignment resolution (14 x 14 tokens), 512 = the main tables
ROOT="${ROOT:-models/seg_tokens}"; EXTRA=()
if [ "${SMOKE:-0}" = 1 ]; then ROOT="${ROOT%/}_smoke"; EXTRA=(--max-images 24); fi
python3 -m eval.export_geobench --task "${DATASET//_/-}" --out "data/$DATASET" >/dev/null
python3 -m eval.extract_dino --preset "$PRESET" --dataset "$DATASET" --tokens --img "$IMG" --out-dir "$ROOT/$DATASET/$PRESET" "${EXTRA[@]}"
echo "SEG_DINO_OK $DATASET"
