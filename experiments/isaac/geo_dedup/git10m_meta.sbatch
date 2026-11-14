#!/bin/bash
# ============================================================================
# Git-10M metadata extraction for the geographic exclusion (eval/geo_dedup.py).
# Each array task streams a contiguous range of the 787 Hub shards (~290 MB each, image
# bytes included -- there is no metadata-only artifact) into $STORE/geo_dedup/git10m_shards,
# writes meta-NNNNN.parquet (img_name, tile z/x/y, lon/lat bounds) and deletes the shard.
#
#   mkdir -p logs && sbatch --array=0-11 experiments/isaac/geo_dedup/git10m_meta.sbatch
#
# 12 tasks x 66 shards (task 11 gets 61): QoS short caps SUBMITTED jobs at 18 per user and an
# array counts every task, so leave room for other short jobs. Measured on the login node:
# 2 s per shard to parse; the Hub delivered a shard in ~5 s. Peak disk: one shard per task.
# Resumable: a shard whose parquet exists is skipped (parquets land via .tmp + rename), so a
# timed-out task can simply be resubmitted. Delete a parquet to redo that shard.
# ============================================================================
#SBATCH --job-name=git10m-meta
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=2:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # HF cache + models/data/logs on scratch

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
python3 -c "import sys, pandas, pyarrow, scipy, huggingface_hub; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

PER_TASK=66
FIRST=$((SLURM_ARRAY_TASK_ID * PER_TASK))
LAST=$((FIRST + PER_TASK - 1)); [ "$LAST" -gt 786 ] && LAST=786
[ "$FIRST" -le 786 ] || { echo "task $SLURM_ARRAY_TASK_ID has no shards"; exit 0; }
OUT="$STORE/geo_dedup/git10m_meta"
echo "shards $FIRST-$LAST -> $OUT"
python3 -m eval.geo_dedup git10m-meta --shard-dir "$STORE/geo_dedup/git10m_shards" \
    --shards "$FIRST-$LAST" --out "$OUT" --hf-download --rm-shard
