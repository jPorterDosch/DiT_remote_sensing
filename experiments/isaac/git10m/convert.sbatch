#!/bin/bash
# ============================================================================
# Git-10M -> random-access arrow shards for training (src/datasets/git10m.py).
# Each array task downloads a contiguous range of the 787 Hub stream shards (~290 MB each)
# into $STORE/git10m/stream, rewrites each as $STORE/git10m/arrow/data-NNNNN-of-00787.arrow
# (64-row batches, JPEG bytes untouched), checks the row count and img_name sequence against
# the dedup metadata ($STORE/geo_dedup/git10m_meta/meta-NNNNN.parquet, so the manifests'
# (shard, row) address these rows), and deletes the stream file.
#
#   mkdir -p logs && sbatch --array=0-11 experiments/isaac/git10m/convert.sbatch
#
# Measured on the login node (2026-10-12): 4.4 s to convert a cached shard; the Hub delivered
# one in ~60 s unauthenticated. 12 tasks x 66 shards ~ 70 min each. Final size ~228 GB
# (Lustre quota was 9.0 of 10 TB before this -- check `lfs quota -u $USER /lustre/isaac24`).
# Resumable: a shard whose arrow file exists is skipped (written via .tmp + rename).
# After all tasks: python -m datasets.git10m check --arrow-dir $STORE/git10m/arrow --index <npz>
# ============================================================================
#SBATCH --job-name=git10m-convert
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=3:00:00
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
python3 -c "import sys, pandas, pyarrow, huggingface_hub; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

PER_TASK=66
FIRST=$((SLURM_ARRAY_TASK_ID * PER_TASK))
LAST=$((FIRST + PER_TASK - 1)); [ "$LAST" -gt 786 ] && LAST=786
[ "$FIRST" -le 786 ] || { echo "task $SLURM_ARRAY_TASK_ID has no shards"; exit 0; }
META="$STORE/geo_dedup/git10m_meta"
[ "$(ls "$META"/meta-*.parquet 2>/dev/null | wc -l)" -eq 787 ] || { echo "FATAL: $META incomplete" >&2; exit 1; }
echo "shards $FIRST-$LAST -> $STORE/git10m/arrow"
PYTHONPATH="$PROJECT_ROOT/src" python3 -m datasets.git10m convert --shards "$FIRST-$LAST" \
    --stream-dir "$STORE/git10m/stream" --out-dir "$STORE/git10m/arrow" --meta-dir "$META" \
    --hf-download --rm-stream
echo "CONVERT_OK task $SLURM_ARRAY_TASK_ID shards $FIRST-$LAST"
