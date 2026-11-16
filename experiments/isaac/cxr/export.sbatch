#!/bin/bash
# ============================================================================
# Write a chest X-ray probe tree (eval/export_cxr.py). One dataset per run; every run names a NEW tree.
#
#   mkdir -p logs
#   sbatch --export=ALL,DATASET=chestxray14,OUT=<new dir>[,MAX_SAMPLES=N][,WRITE_MANIFEST=1] \
#       experiments/isaac/cxr/export.sbatch
#
#   DATASET         required; chestxray14 | chexpert | vindr_cxr_kaggle
#   OUT             required, no default; refuses to start when $OUT or $OUT.partial exists
#   MAX_SAMPLES     optional smoke: first N rows of each split, never writes or checks a manifest
#   WRITE_MANIFEST  optional 0 or 1 (anything else is fatal); 1: also create the tracked manifest (not with MAX_SAMPLES)
#   SKIP_CHECK      optional 0 or 1 (anything else is fatal); 1: skip the `datasets.cxr14 check` gate (only together with MAX_SAMPLES)
#   WORKERS         optional, vindr_cxr_kaggle only: decode processes (positive integer; default
#                   SLURM_CPUS_PER_TASK, i.e. 8). Anything else is fatal before any work starts.
#   RAW ARROW MANIFEST INDEX   optional path overrides (defaults below)
#
# Success is the line CXR_EXPORT_OK printed by eval.export_cxr; the python command's exit status is
# the script's exit status (nothing is echoed after it).
# ============================================================================
#SBATCH --job-name=cxr-export
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
python3 -c "import sys; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

: "${DATASET:?FATAL: set DATASET}"
: "${OUT:?FATAL: set OUT to a NEW tree directory (no default)}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
WRITE_MANIFEST="${WRITE_MANIFEST:-0}"
SKIP_CHECK="${SKIP_CHECK:-0}"
case "$WRITE_MANIFEST" in 0 | 1) ;; *) echo "FATAL: WRITE_MANIFEST must be 0 or 1" >&2; exit 1 ;; esac
case "$SKIP_CHECK" in 0 | 1) ;; *) echo "FATAL: SKIP_CHECK must be 0 or 1" >&2; exit 1 ;; esac
[ ! -e "$OUT" ] && [ ! -e "$OUT.partial" ] || { echo "FATAL: $OUT or $OUT.partial exists; pick a new OUT" >&2; exit 1; }
if [ -n "${WORKERS+x}" ]; then
    [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: WORKERS must match ^[1-9][0-9]*\$ (a positive integer)" >&2; exit 1; }
fi
[ -z "$MAX_SAMPLES" ] || [ "$WRITE_MANIFEST" != "1" ] \
    || { echo "FATAL: WRITE_MANIFEST=1 cannot be combined with MAX_SAMPLES" >&2; exit 1; }
[ "$SKIP_CHECK" != "1" ] || [ -n "$MAX_SAMPLES" ] \
    || { echo "FATAL: SKIP_CHECK=1 is allowed only together with MAX_SAMPLES" >&2; exit 1; }

D=/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/datasets/cxr
[ "$(stat -c %a "$D")" = "700" ] || { echo "FATAL: $D is not mode 700" >&2; exit 1; }
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"

export_args=(--dataset "$DATASET" --out "$OUT")
[ -z "$MAX_SAMPLES" ] || export_args+=(--max-samples "$MAX_SAMPLES")
[ "$WRITE_MANIFEST" != "1" ] || export_args+=(--write-manifest)

case "$DATASET" in
    chestxray14)
        RAW="${RAW:-$D/chestxray14/raw}"
        ARROW="${ARROW:-$D/chestxray14/arrow}"
        MANIFEST="${MANIFEST:-$D/chestxray14/manifest_v2}"
        INDEX="${INDEX:-$D/chestxray14/index_v2.npz}"
        export_args+=(--raw-dir "$RAW" --arrow-dir "$ARROW" --manifest-dir "$MANIFEST")
        pre_check() {
            python3 -m datasets.cxr14 check --arrow-dir "$ARROW" --manifest-dir "$MANIFEST" --index "$INDEX"
        }
        ;;
    chexpert)
        RAW="${RAW:-$D/chexpert/raw/chexpertchestxrays-u20210408}"
        export_args+=(--raw-dir "$RAW")
        # No index gate: the split is rebuilt from the CSVs inside the zips on every run, and the
        # leakage gate (Task 9) checks the written tree.
        pre_check() { :; }
        ;;
    vindr_cxr_kaggle)
        RAW="${RAW:-$D/vindr_cxr/raw}"
        export_args+=(--raw-dir "$RAW" --workers "${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}")
        pre_check() { :; }  # no index gate: the split is drawn from train.csv inside the zip on every run
        ;;
    *)
        echo "FATAL: no export defaults for $DATASET yet" >&2
        exit 1
        ;;
esac

python3 -I experiments/isaac/cxr/quota_guard.py --min-free-gb 900
if [ "$SKIP_CHECK" != "1" ]; then
    pre_check
fi
python3 -m eval.export_cxr "${export_args[@]}"
