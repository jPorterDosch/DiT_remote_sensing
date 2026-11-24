#!/bin/bash
# ============================================================================
# ChestX-ray14 -> random-access arrow shards (src/datasets/cxr14.py). One archive per array task:
# $RAW/images_00{i+1}.tar.gz -> $OUT/cxr14-NN-of-12.arrow (64-row batches, PNG bytes untouched)
# + $OUT/meta/<shard>.json (row count, sha256, PIL mode and size tallies, names). Every member is
# fully decoded; a 16-bit or otherwise unexpected PIL mode, an undecodable member, a duplicate name
# fails the task. Existing outputs and .partial files are never overwritten (the task refuses).
#
#   mkdir -p logs && sbatch experiments/isaac/cxr/convert.sbatch            # full run, tasks 0-11, 6 at a time
#   sbatch --array=0 --export=ALL,OUT=$D/smoke_vN/chestxray14/arrow experiments/isaac/cxr/convert.sbatch   # smoke: a NEW smoke_vN each time
#
# Environment (defaults are the full-run values): D, RAW=$D/chestxray14/raw, OUT=$D/chestxray14/arrow.
# The quota guard runs first (900 GB free under the per-user scratch quota). Final line, printed
# only if every step exited 0 and read back from the meta file the task wrote:
#   CXR14_CONVERT_OK shard=<i> rows=<n>
# ============================================================================
#SBATCH --job-name=cxr14-convert
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=3:00:00
#SBATCH --array=0-11%6
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
python3 -c "import sys, pandas, pyarrow, PIL, torch; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

D="${D:-/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/datasets/cxr}"
RAW="${RAW:-$D/chestxray14/raw}"
OUT="${OUT:-$D/chestxray14/arrow}"
I="$SLURM_ARRAY_TASK_ID"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"

python3 experiments/isaac/cxr/quota_guard.py --min-free-gb 900
mkdir -p "$OUT"
echo "shard $I: $RAW -> $OUT"
python3 -m datasets.cxr14 convert --shard "$I" --raw-dir "$RAW" --out-dir "$OUT"
python3 -I - "$OUT" "$I" <<'PY'
import glob, json, sys
out, i = sys.argv[1], int(sys.argv[2])
(path,) = glob.glob(f"{out}/meta/cxr14-{i:02d}-of-*.arrow.json")
m = json.load(open(path))
assert m["shard"] == i, (m["shard"], i)
print(f"meta {path}: pil_modes={m['pil_modes']} alpha_nonopaque={m['alpha_nonopaque']}")
print(f"CXR14_CONVERT_OK shard={i} rows={m['rows']}")
PY
