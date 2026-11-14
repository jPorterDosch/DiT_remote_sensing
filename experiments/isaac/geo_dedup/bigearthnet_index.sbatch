#!/bin/bash
# ============================================================================
# One streaming pass over BigEarthNet-S2-v1.0.tar.gz (eval/geo_dedup.py bigearthnet-index):
# per patch, the pixel hash of B02 + the footprint from the patch json -> patch_index.parquet.
# Needs fetch_bigearthnet.sbatch's md5ok marker. Nothing is extracted to disk.
#
#   sbatch --dependency=afterok:<fetch job> experiments/isaac/geo_dedup/bigearthnet_index.sbatch
# ============================================================================
#SBATCH --job-name=bigearthnet-index
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
python3 -c "import sys, pandas, pyarrow, rasterio; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

D="$STORE/geo_dedup/source_meta/bigearthnet"
TAR="$D/BigEarthNet-S2-v1.0.tar.gz"
[ -f "$TAR.md5ok" ] || { echo "FATAL: $TAR not md5-verified (run fetch_bigearthnet.sbatch)" >&2; exit 1; }
python3 -m eval.geo_dedup bigearthnet-index --tar "$TAR" --out "$D/patch_index.parquet"
