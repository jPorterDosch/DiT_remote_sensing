#!/bin/bash
# ============================================================================
# Full CPU test suite on an EXPORT OF THE GIT INDEX, not on the working tree: the tests then see
# exactly the files a commit of the index would contain. A file that the code needs but that is
# not staged makes the suite fail here instead of on the other machine (CLAUDE.md rule 17).
#
#   SNAP=/lustre/isaac24/scratch/$USER/snapshots/idx-$(date +%Y%m%d-%H%M%S)
#   TREE=$(git write-tree)                       # identity of the index that is exported
#   git checkout-index -a --prefix="$SNAP/"      # the trailing slash is required
#   mkdir -p logs
#   SNAP=$SNAP TREE=$TREE sbatch experiments/isaac/eval/pytest_snapshot.sbatch
#
# Success is the line PYTEST_SNAPSHOT_OK tree=<TREE> in the log, not the job state. Before
# committing, `git write-tree` must still print the same TREE.
# TESTS="tests/test_a.py tests/test_b.py" runs only those files and ends with PYTEST_SUBSET_OK.
# GPU-marked tests are skipped (no --gpu); tests whose input caches are absent skip themselves.
# Variables travel through the submitting shell's environment (sbatch exports it by default).
# ============================================================================
#SBATCH --job-name=pytest-snapshot
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

SNAP="${SNAP:?set SNAP (absolute path of the index export)}"
TREE="${TREE:?set TREE (git write-tree of the exported index)}"
case "$SNAP" in /*) ;; *) echo "FATAL: SNAP must be absolute: $SNAP" >&2; exit 2 ;; esac
[ -f "$SNAP/tests/conftest.py" ] && [ -f "$SNAP/experiments/isaac/scratch_env.sh" ] \
    || { echo "FATAL: $SNAP is not an index export (tests/conftest.py or scratch_env.sh missing)" >&2; exit 1; }
[ ! -e "$SNAP/.git" ] || { echo "FATAL: $SNAP holds a .git: it is a checkout, not an index export" >&2; exit 1; }

PROJECT_ROOT="$SNAP"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
export WANDB_MODE=disabled
python3 -c "import sys, numpy, torch, pytest; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

n_files=$(find . -type f -not -path './.pytest_cache/*' | wc -l)
echo "pytest-snapshot snap=$SNAP tree=$TREE files=$n_files host=$(hostname) $(date -Is)"

# TESTS (optional, space-separated test files) runs a SUBSET for fast feedback on changed tests.
# A subset run never prints PYTEST_SNAPSHOT_OK: only the full suite certifies an index.
read -r -a targets <<< "${TESTS:-tests}"
for t in "${targets[@]}"; do
    [ -e "$t" ] || { echo "FATAL: test target $t is not in the index export $SNAP" >&2; exit 1; }
done
echo "targets: ${targets[*]}"

# pytest's exit code decides: 0 = every collected test passed or skipped; 5 = nothing collected.
rc=0
python3 -m pytest "${targets[@]}" -q -p no:cacheprovider -rfEs --durations=15 || rc=$?
[ "$rc" -eq 0 ] || { echo "FATAL: pytest exit $rc on tree $TREE: do NOT commit this index" >&2; exit 1; }
if [ -n "${TESTS:-}" ]; then
    echo "PYTEST_SUBSET_OK (NOT a certification) tree=$TREE targets=${targets[*]} $(date -Is)"
else
    echo "PYTEST_SNAPSHOT_OK tree=$TREE $(date -Is)"
fi
