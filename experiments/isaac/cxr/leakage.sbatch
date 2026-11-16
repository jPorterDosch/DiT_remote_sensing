#!/bin/bash
# ============================================================================
# Leakage gate of the chest X-ray data stage (eval/cxr_leakage.py): no evaluation patient or image is
# reachable through the alignment index, and every probe tree has the registered sizes. Counts-only output.
#
#   mkdir -p logs
#   sbatch [--export=ALL,INDEX=...,MANIFEST=...,RAW=...,TREE_CHESTXRAY14=...,TREE_CHEXPERT=...,TREE_VINDR=...] \
#       experiments/isaac/cxr/leakage.sbatch
#
#   INDEX MANIFEST RAW   defaults below (the v2 index, its manifest and the shipped ChestX-ray14 files)
#   TREE_CHESTXRAY14 TREE_CHEXPERT TREE_VINDR   probe tree roots, default $PROJECT_ROOT/data/cxr/<key>;
#                        a root that does not exist prints SKIPPED <key> and ends the run INCOMPLETE
#
# Output: results/gates/cxr_leakage_${SLURM_JOB_ID}.json (never overwritten). Success is the final line
# CXR_LEAKAGE_GATE PASS in the log; the python exit status is the script's exit status (nothing is
# echoed after it): 0 PASS, 1 FAIL, 2 INCOMPLETE.
# ============================================================================
#SBATCH --job-name=cxr-leakage
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
python3 -c "import sys; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

D=/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/datasets/cxr
[ "$(stat -c %a "$D")" = "700" ] || { echo "FATAL: $D is not mode 700" >&2; exit 1; }
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"

INDEX="${INDEX:-$D/chestxray14/index_v2.npz}"
MANIFEST="${MANIFEST:-$D/chestxray14/manifest_v2}"
RAW="${RAW:-$D/chestxray14/raw}"
TREE_CHESTXRAY14="${TREE_CHESTXRAY14:-$PROJECT_ROOT/data/cxr/chestxray14}"
TREE_CHEXPERT="${TREE_CHEXPERT:-$PROJECT_ROOT/data/cxr/chexpert}"
TREE_VINDR="${TREE_VINDR:-$PROJECT_ROOT/data/cxr/vindr_cxr_kaggle}"
JOB="${SLURM_JOB_ID:-manual}"

mkdir -p results/gates
exec python3 -m eval.cxr_leakage --index "$INDEX" --manifest-dir "$MANIFEST" --raw-dir "$RAW" \
    --tree "chestxray14=$TREE_CHESTXRAY14" --tree "chexpert=$TREE_CHEXPERT" \
    --tree "vindr_cxr_kaggle=$TREE_VINDR" --out "results/gates/cxr_leakage_${JOB}.json"
