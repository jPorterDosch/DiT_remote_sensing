#!/bin/bash
# ============================================================================
# ChestX-ray14 patient-level manifest + index + check (src/datasets/cxr14.py), run after all 12
# shards are converted. Hashes the 12 shards and the label files, cross-checks the shard names
# against the CSV and the two official lists, buckets by patient, writes the manifest directory
# $MANIFEST and the index $INDEX (neither may exist; .partial targets are refused), then compares
# both with the pinned counts.
#
#   mkdir -p logs && sbatch experiments/isaac/cxr/build_index.sbatch
#
# Environment (defaults are the full-run values): D, RAW=$D/chestxray14/raw, OUT=$D/chestxray14/arrow,
# MANIFEST=$D/chestxray14/manifest_v2, INDEX=$D/chestxray14/index_v2.npz.
# Smoke (one converted shard): SHARDS=0 ALLOW_SMOKE=1 (also MANIFEST, INDEX under a smoke dir).
# Final line, only if every step exited 0:
#   CXR14_INDEX_OK align=<n> validation=<n>          (full run)
#   CXR14_INDEX_SMOKE_OK align=<n> validation=<n>    (smoke: never a full-run pass line)
# ============================================================================
#SBATCH --job-name=cxr14-index
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

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
# v1 was built with the wrong patient key (name prefix) by job 6600108 and must not be used.
MANIFEST="${MANIFEST:-$D/chestxray14/manifest_v2}"
INDEX="${INDEX:-$D/chestxray14/index_v2.npz}"
SHARDS="${SHARDS:-}"
ALLOW_SMOKE="${ALLOW_SMOKE:-0}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"

manifest_args=(--arrow-dir "$OUT" --raw-dir "$RAW" --out-dir "$MANIFEST")
[ -z "$SHARDS" ] || manifest_args+=(--shards "$SHARDS")
index_args=(--manifest-dir "$MANIFEST" --out "$INDEX")
[ "$ALLOW_SMOKE" != "1" ] || index_args+=(--allow-smoke)

mkdir -p "$(dirname "$MANIFEST")" "$(dirname "$INDEX")"
python3 -m datasets.cxr14 build-manifest "${manifest_args[@]}"
python3 -m datasets.cxr14 build-index "${index_args[@]}"
python3 -m datasets.cxr14 check --arrow-dir "$OUT" --manifest-dir "$MANIFEST" --index "$INDEX"
python3 - "$INDEX" <<'PY'
import sys
from datasets.cxr14 import index_description
d = index_description(sys.argv[1])
tag = "CXR14_INDEX_SMOKE_OK" if d["smoke"] else "CXR14_INDEX_OK"
print(f"{tag} align={d['buckets']['retained']} validation={d['buckets']['validation']}")
PY
