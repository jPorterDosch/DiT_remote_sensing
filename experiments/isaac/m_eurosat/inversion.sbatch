#!/bin/bash
# ============================================================================
# m-eurosat INVERSION extraction (RF-Solver 50-step chain, 7-t grid, k=28, g=1.0),
# SHARDED as a SLURM array of 10 tasks:
#   array index 0..7  -> train split, shards 0..7 of 8 (2,025 imgs each)
#   array index 8     -> val split,  unsharded (996 imgs)
#   array index 9     -> test split, unsharded (996 imgs)
# Submit:  sbatch experiments/isaac/m_eurosat/inversion.sbatch
#
# TIME: measured 16.5 s/img on an RTX A6000; on ai-tenn H100 expect ~6 s/img =>
#   train shard ~3.5 h, val/test ~1.8 h. The 14 h request survives zero speedup
#   (A40 rate = 10.2 h per shard) and backfills far better than a 3-day request.
#   ai-tenn QoS allows 28 concurrent jobs, so all 10 array tasks can run at once.
#
# Shard caches carry their exact image indices in subset_indices and a +sNofM run-name
# suffix; eval.probe --protocol official verifies the shards are disjoint and complete
# before any probe sees them. The chain draws no eps; the per-split --seed decouples
# the VAE-posterior streams across splits (RESEARCH_NOTES 6t, audit F1).
# ============================================================================
#SBATCH --job-name=meurosat-inv
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=14:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # caches + models/data/logs on Lustre scratch

source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -f "$PROJECT_ROOT/.secrets/wandb-personal.env" ] && source "$PROJECT_ROOT/.secrets/wandb-personal.env"
export WANDB_MODE="${WANDB_MODE:-offline}"

IDX="${SLURM_ARRAY_TASK_ID:?}"
if [ "$IDX" -le 7 ]; then
    # Distinct seed PER SHARD (100+IDX): with one shared seed, image j of every shard drew
    # the same VAE-posterior noise -- the audit-F1 position-paired coupling, inside train.
    # The seed is part of the run name, so banked shared-seed shards are left untouched --
    # but the probe glob must then not span both generations (the shard merge refuses
    # duplicate shard indices): move the old shard dirs aside before probing. A task killed
    # mid-extraction leaves its run dir without *_meta.json; run.py refuses an existing dir,
    # so rm that dir before re-queuing the task (not automated: 10 tasks share this save dir).
    SPLIT=train; SEED=$((100 + IDX)); SHARDS=(--num-shards 8 --shard-index "$IDX")
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
