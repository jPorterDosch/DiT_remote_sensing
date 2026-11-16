#!/bin/bash
# ============================================================================
# CPU pytest job for the chest X-ray data stage (never run pytest on a login node).
#
#   mkdir -p logs && sbatch --wait --export=ALL,TESTS="tests/test_cxr_count.py" experiments/isaac/cxr/pytest.sbatch
#
# TESTS (space-separated test files) is required. Success is the line CXR_PYTEST_OK in the log,
# printed only when pytest exits 0; the job state alone proves nothing.
# ============================================================================
#SBATCH --job-name=cxr-pytest
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

[ -n "${TESTS:-}" ] || { echo "FATAL: set TESTS (e.g. --export=ALL,TESTS=\"tests/test_x.py\")" >&2; exit 2; }

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
export WANDB_MODE=disabled
python3 -c "import sys, pytest; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

read -r -a targets <<< "$TESTS"
for t in "${targets[@]}"; do
    [ -e "$t" ] || { echo "FATAL: test target $t does not exist under $PROJECT_ROOT" >&2; exit 1; }
done
echo "cxr-pytest root=$PROJECT_ROOT targets=${targets[*]} host=$(hostname) $(date -Is)"

rc=0
python3 -m pytest "${targets[@]}" -q -p no:cacheprovider -rfEs || rc=$?
[ "$rc" -eq 0 ] || { echo "FATAL: pytest exit $rc" >&2; exit 1; }
echo "CXR_PYTEST_OK $(date -Is)"
