#!/bin/bash
# ============================================================================
# Pixel-level check of the Git-10M SSL4EO-S12 name -> location rule (eval/ssl4eo_pixel_check.py).
# Downloads SSL4EO-S12 v1.1 train_metadata.parquet and N_SHARDS of its S2RGB tar shards
# (~0.35 GB each) into $STORE/geo_dedup/source_meta/ssl4eo_v11, builds the Git-10M -> v1.1 join
# once, scores PER_SHARD Git-10M rows per shard against the patch at the mapped id vs a random
# other location, and prints SSL4EO_PIXEL_CHECK: PASS|FAIL (thresholds in the script).
#
#   mkdir -p logs && sbatch experiments/isaac/geo_dedup/ssl4eo_pixel_check.sbatch
#
# numcodecs (blosc decoding of the zarr chunks) is not in the DiTF env; it is pip-installed
# into $STORE/pyextra (pip install --target) and put on PYTHONPATH here. Login-node smoke
# 2026-10-13 (1 shard, 24 rows): median mapped 0.88, null 0.04, 24/24 above null.
# ============================================================================
#SBATCH --job-name=ssl4eo-pixel-check
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
export PYTHONPATH="$STORE/pyextra${PYTHONPATH:+:$PYTHONPATH}"
python3 -c "import sys, pandas, pyarrow, numcodecs, huggingface_hub; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env (pip install --target $STORE/pyextra numcodecs)" >&2; exit 1; }
set -euo pipefail

G="$STORE/geo_dedup"
V="$G/source_meta/ssl4eo_v11"
N_SHARDS="${N_SHARDS:-20}"
PER_SHARD="${PER_SHARD:-64}"
SEED="${SEED:-0}"
OUT="${OUT:-$V/ssl4eo_pixel_check_s${SEED}_n${N_SHARDS}x${PER_SHARD}.json}"
[ "$(ls "$G"/git10m_meta/meta-*.parquet | wc -l)" -eq 787 ] || { echo "FATAL: git10m_meta incomplete" >&2; exit 1; }
[ -f "$G/source_boxes.parquet" ] || { echo "FATAL: $G/source_boxes.parquet missing (dedup_v2.sbatch STEP=source-boxes)" >&2; exit 1; }
[ -f "$G/source_meta/ssl4eo/void_ids.csv" ] || { echo "FATAL: void_ids.csv missing" >&2; exit 1; }
[ "$(ls "$STORE"/git10m/arrow/data-*-of-00787.arrow | wc -l)" -eq 787 ] || { echo "FATAL: git10m/arrow incomplete" >&2; exit 1; }
mkdir -p "$V"
[ -f "$V/train_metadata.parquet" ] || python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('embed2scale/SSL4EO-S12-v1.1', 'train_metadata.parquet', repo_type='dataset', local_dir='$V')"
[ -f "$V/git10m_v11_join.parquet" ] || python3 -m eval.ssl4eo_pixel_check \
    --build-join "$G/git10m_meta" "$G/source_boxes.parquet" "$V/train_metadata.parquet" --join "$V/git10m_v11_join.parquet"
python3 -m eval.ssl4eo_pixel_check --join "$V/git10m_v11_join.parquet" --arrow-dir "$STORE/git10m/arrow" \
    --v11-dir "$V" --ssl4eo-void "$G/source_meta/ssl4eo/void_ids.csv" \
    --n-shards "$N_SHARDS" --per-shard "$PER_SHARD" --seed "$SEED" --hf-download --out "$OUT"
