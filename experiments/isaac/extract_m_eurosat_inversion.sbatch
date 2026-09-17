#!/bin/bash
# ============================================================================
# m-eurosat INVERSION extraction (RF-Solver 50-step chain, 7-t grid, k=28, g=1.0),
# SHARDED as a SLURM array of 10 tasks:
#   array index 0..7  -> train split, shards 0..7 of 8 (2,025 imgs each)
#   array index 8     -> val split,  unsharded (996 imgs)
#   array index 9     -> test split, unsharded (996 imgs)
# Submit:  sbatch experiments/isaac/extract_m_eurosat_inversion.sbatch
#
# TIME: measured 16.5 s/img on the workstation (A6000-class) => train shard ~9.3 h,
#   val/test ~4.6 h. The 24 h wall gives ~2.5x headroom; on slower campus GPUs check
#   the first shard's rate before assuming the rest land.
#
# Shard caches carry their exact image indices in subset_indices and a +sNofM run-name
# suffix; experiments/m_eurosat_probe.py verifies the shards are disjoint and complete
# before any probe sees them. The chain draws no eps; the per-split --seed decouples
# the VAE-posterior streams across splits (RESEARCH_NOTES 6t, audit F1).
# ============================================================================
#SBATCH --job-name=meurosat-inv
#SBATCH -A acf-utk0011
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
[ -f "$PROJECT_ROOT/.secrets/wandb-personal.env" ] && source "$PROJECT_ROOT/.secrets/wandb-personal.env"
export WANDB_MODE="${WANDB_MODE:-offline}"

IDX="${SLURM_ARRAY_TASK_ID:?}"
if [ "$IDX" -le 7 ]; then
    SPLIT=train; SEED=42; SHARDS=(--num-shards 8 --shard-index "$IDX")
elif [ "$IDX" -eq 8 ]; then
    SPLIT=val;   SEED=44; SHARDS=()
else
    SPLIT=test;  SEED=43; SHARDS=()
fi
echo "array task $IDX -> split=$SPLIT seed=$SEED ${SHARDS[*]:-unsharded}"

python3 "$PROJECT_ROOT/run.py" \
    --task extract \
    --model.name flux --model.ensemble-size 1 \
    --dataset.name m_eurosat \
    --dataset.path "$PROJECT_ROOT/data/m_eurosat_rgb" \
    --img-size 224 224 \
    --t 100 180 260 340 420 500 580 \
    --extraction-mode INVERSION --num-inversion-steps 50 \
    --k 28 --guidance-scale 1.0 \
    --seed "$SEED" --subset-seed 42 \
    --extract-split "$SPLIT" \
    "${SHARDS[@]}" \
    --label-fraction 1.0 --batch-size 1 \
    --save-dir "$PROJECT_ROOT/models/m_eurosat_inversion"
