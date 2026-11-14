#!/bin/bash
# ============================================================================
# Per-token FLUX features of one segmentation dataset (run.py --task extract_seg_tokens,
# src/tasks/seg_tokens.py): three splits in sequence, one (N, L, C) float16 memmap per
# (timestep, block), under models/seg_tokens/<dataset>/<NAME>/<run dir>/ (run.py appends the run
# name, one run dir per split; eval/seg_probe.py globs it). 512 px (32 x 32 tokens, 8 px each
# on a 256 px mask), ensemble 2, eps seeds train 42 / test 43 / val 44, guidance 1.0 (PINNED).
#
#   CKPT   empty = the frozen model (NAME flux-pretrained); <run>/checkpoints/lora_last_*.pt = an arm
#          (NAME = the run-dir basename, as extract_ckpt.sbatch)
#   T      timesteps (default "100 260 420");  K  blocks (default "12 28 38" frozen, "28" for an arm)
#   SMOKE=1  --max-images 24, models/seg_tokens_smoke/ (never quotable), run first
#
#   sbatch --export=ALL,SMOKE=1,CKPT="" --array=0 --time=2:00:00 experiments/isaac/seg/extract_flux.sbatch
#   sbatch --export=ALL,CKPT="" --array=0-2 experiments/isaac/seg/extract_flux.sbatch
#   sbatch --export=ALL,CKPT=$STORE/models/git10m/<run>/checkpoints/lora_last_step00020000.pt,K=28 --array=0-2 ...
# Disk: 3000 x 1024 x 3072 x 2 B = 19 GB per (t, k) train file; frozen 9 files x 3 splits ~ 280 GB
# per dataset. Delete a dataset's token dir once its results/seg cells exist (README).
# ============================================================================
#SBATCH --job-name=seg-tokens
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
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
source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
python3 -c "import sys, numpy, torch, tyro; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
set -euo pipefail
DATASETS=(m_pv4ger_seg m_chesapeake_landcover m_SA_crop_type)
IDX="${SLURM_ARRAY_TASK_ID:?submit with --array=0-2}"
DATASET="${DATASETS[$IDX]}"
CKPT="${CKPT-unset}"
[ "$CKPT" != "unset" ] || { echo "FATAL: set CKPT=\"\" (frozen) or CKPT=<run>/checkpoints/lora_last_*.pt via --export" >&2; exit 1; }
if [ -n "$CKPT" ]; then
    [ -f "$CKPT" ] || { echo "FATAL: $CKPT missing" >&2; exit 1; }
    NAME="${NAME:-$(basename "$(cd "$(dirname "$CKPT")/.." && pwd)")}"
    K="${K:-28}"
else
    NAME="${NAME:-flux-pretrained}"
    K="${K:-12 28 38}"
fi
T="${T:-100 260 420}"
IMG="${IMG:-512}"   # token input side; 224 = the alignment resolution (14 x 14 tokens), 512 = the main tables
ROOT="${ROOT:-models/seg_tokens}"; EXTRA=()
if [ "${SMOKE:-0}" = 1 ]; then ROOT="${ROOT%/}_smoke"; EXTRA=(--max-images 24); fi
OUT="$ROOT/$DATASET/$NAME"
python3 -m eval.export_geobench --task "${DATASET//_/-}" --out "data/$DATASET" >/dev/null  # tree == committed manifest
# shellcheck disable=SC2206
T_ARR=($T); K_ARR=($K)
seed_for() { case "$1" in train) echo 42 ;; test) echo 43 ;; val) echo 44 ;; esac; }
for split in train val test; do
    seed="$(seed_for "$split")"
    ARGS=(--task extract_seg_tokens --model.name flux --model.ensemble-size 2
          --dataset.name "$DATASET" --dataset.path "data/$DATASET" --img-size "$IMG" "$IMG" --t "${T_ARR[@]}" --k "${K_ARR[@]}"
          --extraction-mode ONESHOT --guidance-scale 1.0 --cd
          --extract-split "$split" --seed "$seed" --eps-seed "$seed"
          --batch-size 1 --num-workers 4 --save-dir "$OUT" "${EXTRA[@]}")
    [ -n "$CKPT" ] && ARGS+=(--lora-checkpoint "$CKPT")
    echo "=== seg tokens $NAME $DATASET $split: ${ARGS[*]} $(date -Is)"
    python3 "$PROJECT_ROOT/run.py" "${ARGS[@]}"
done
echo "SEG_TOKENS_ALL_OK $DATASET $NAME -> $OUT/<run dir per split>"
