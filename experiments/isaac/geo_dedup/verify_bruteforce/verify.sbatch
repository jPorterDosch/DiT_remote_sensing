#!/bin/bash
# ============================================================================
# Executed PASS / FAIL for the independent brute-force check of the geographic dedup manifests
# (verify.py + verdict.py). The run of 2026-10-13 predates the verdict, so no gate line exists
# for it (README.md). Read-only on $STORE/geo_dedup; writes results.json and the sample csvs
# into a NEW directory results/geo_dedup/verify_bruteforce_<job id>/ and nothing else.
# GATE 1: tests/test_geo_dedup_verdict.py must pass in this job before the check runs.
# GATE 2: `GATE brute-force verify PASS` printed by verdict.py; verify.py exits 1 on FAIL.
#
#   mkdir -p logs
#   sbatch experiments/isaac/geo_dedup/verify_bruteforce/verify.sbatch
# Success is the line VERIFY_BRUTEFORCE_OK in the log, not the job state.
# ============================================================================
#SBATCH --job-name=verify-bruteforce
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
python3 -c "import sys, numpy, pandas, pyarrow, pytest; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

G="$STORE/geo_dedup"
V="$PROJECT_ROOT/experiments/isaac/geo_dedup/verify_bruteforce"
for p in "$G/points_v2.csv" "$G/points_v3_cities.csv" "$G/source_boxes.parquet" \
         "$G/dedup_v2_b1000/retained.csv" "$G/dedup_v3_b1000_cities/retained.csv" \
         "$PROJECT_ROOT/experiments/isaac/geo_dedup/so2sat_cities.csv" "$V/verify.py" "$V/verdict.py"; do
    [ -f "$p" ] || { echo "FATAL: $p missing" >&2; exit 1; }
done
OUT="$PROJECT_ROOT/results/geo_dedup/verify_bruteforce_${SLURM_JOB_ID:?run under sbatch}"
[ ! -e "$OUT" ] || { echo "FATAL: $OUT exists (nothing is overwritten)" >&2; exit 1; }
echo "verify-bruteforce store=$G out=$OUT host=$(hostname) $(date -Is)"

python3 -m pytest tests/test_geo_dedup_verdict.py -q -p no:cacheprovider -rfE
echo "VERIFY_VERDICT_TESTS_OK"

mkdir -p "$OUT"
cd "$OUT"
# `if !`: under set -e a failing verify.py would end the job before the FATAL line is printed.
if ! python3 "$V/verify.py"; then
    echo "FATAL: brute-force verify did not pass; outputs kept in $OUT" >&2
    exit 1
fi
[ -s results.json ] || { echo "FATAL: verify.py exited 0 without results.json" >&2; exit 1; }
echo "VERIFY_BRUTEFORCE_OK out=$OUT $(date -Is)"
