#!/bin/bash
# ============================================================================
# Count the shipped CheXpert and VinDr-CXR artifacts (CLAUDE.md rule 16): zip central directories,
# CSV schema and label tallies only. Aggregates go to $D/counts/counts_<jobid>.json (mode-700 D);
# nothing patient-level is printed. Refuses to overwrite an existing output file.
#
#   mkdir -p logs && sbatch experiments/isaac/cxr/count.sbatch
#
# Success is CXR_COUNT_OK in the log (printed only if the python command exits 0).
# ============================================================================
#SBATCH --job-name=cxr-count
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

D=/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/datasets/cxr
[ "$(stat -c %a "$D")" = "700" ] || { echo "FATAL: $D is not mode 700" >&2; exit 1; }
mkdir -p "$D/counts"
python3 -m eval.cxr_count --chexpert "$D/chexpert/raw/chexpertchestxrays-u20210408" \
    --vindr "$D/vindr_cxr/raw" --out "$D/counts/counts_${SLURM_JOB_ID}.json"
echo "CXR_COUNT_OK"
