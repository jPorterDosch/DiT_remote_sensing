#!/bin/bash
# Rebuild every Git-10M training index with the one-row-per-image rule (datasets.git10m
# build-index). New file names (*_unique.npz); the pre-rule files are left in place and are
# refused by the dataset. ~10 M manifest rows through pandas: run on short, not the login node.
#   sbatch experiments/isaac/git10m/build_index.sbatch
#SBATCH --job-name=git10m-index
#SBATCH -A isaac-utk0256
#SBATCH -p short -q short
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem=48G --time=1:00:00
#SBATCH --output=logs/%x-%j.out
_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"; cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
set -euo pipefail
G="$STORE/geo_dedup"; O="$STORE/git10m"
run() { PYTHONPATH="$PROJECT_ROOT/src" python3 -m datasets.git10m build-index "$@"; }
run --manifest-dir "$G/dedup_v2_b1000" --out "$O/index_retained_unique.npz"
run --manifest-dir "$G/dedup_v2_b1000" --out "$O/index_all_unique.npz" --include-unassessable
run --manifest-dir "$G/dedup_v3_b1000_cities" --out "$O/index_retained_cities_unique.npz"
run --manifest-dir "$G/dedup_v3_b1000_cities" --out "$O/index_all_cities_unique.npz" --include-unassessable
run --manifest-dir "$G/dedup_v2_b1000" --out "$O/index_smoke_s0_unique.npz" --include-unassessable --shards 0-0
PYTHONPATH="$PROJECT_ROOT/src" python3 -m datasets.git10m check --arrow-dir "$O/arrow" --index "$O/index_all_unique.npz" | tail -3
echo GIT10M_INDEX_OK
