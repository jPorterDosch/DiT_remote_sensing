#!/bin/bash
# ============================================================================
# Paired image-bootstrap intervals (eval/paired_ci.py) for the arm-vs-arm differences quoted in
# the paper tables, from the stored per-image results. CPU only, no training, nothing
# re-extracted; writes <OUT>/<name>.json + .md and refuses to overwrite them.
# GATE 1: tests/test_paired_ci.py must pass in this job before any comparison runs.
# GATE 2: the tool's own identity / reproduction / null gates (it exits 1 on FAIL).
#
#   mkdir -p logs
#   sbatch experiments/isaac/eval/paired_ci.sbatch
# RUNS: space list of name:kind:cells:base (default: the four table comparisons below).
# Variables travel through the submitting shell's environment (sbatch exports it by default).
# ============================================================================
#SBATCH --job-name=paired-ci
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
python3 -c "import sys, numpy; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

OUT="${OUT:-results/paired}"
RUNS="${RUNS:-seg_phase3_vs_frozen:seg:results/seg/phase3_lr/cells:flux-pretrained \
seg_img224_vs_frozen:seg:results/seg/phase4_img224/cells:flux-pretrained-k28 \
cls_vs_frozen:cls:results/matrix/cells:flux-pretrained-k28 \
cls_vs_dinov3sat:cls:results/matrix/cells:dinov3-sat}"
for run in $RUNS; do
    IFS=: read -r name kind cells base <<< "$run"
    [ -n "$name" ] && [ -n "$kind" ] && [ -n "$cells" ] && [ -n "$base" ] || { echo "FATAL: bad RUNS entry '$run'" >&2; exit 2; }
    [ -d "$cells/$base" ] || { echo "FATAL: $cells/$base missing" >&2; exit 1; }
    for ext in .json .md; do
        [ -e "$OUT/$name$ext" ] && { echo "FATAL: $OUT/$name$ext exists (move it aside; nothing is overwritten)" >&2; exit 1; }
    done
done
echo "paired-ci out=$OUT host=$(hostname) $(date -Is)"

python3 -m pytest tests/test_paired_ci.py -q -p no:cacheprovider -rfE
echo "PAIRED_CI_TESTS_OK"

mkdir -p "$OUT"
fail=0
for run in $RUNS; do
    IFS=: read -r name kind cells base <<< "$run"
    echo "=== $name: kind=$kind cells=$cells base=$base $(date -Is)"
    # a failed gate in one comparison must not hide the others: record it and go on
    python3 -m eval.paired_ci --kind "$kind" --cells "$cells" --base "$base" --out "$OUT/$name" || { echo "FAILED $name"; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "FATAL: at least one comparison failed a gate: do NOT quote its numbers" >&2; exit 1; }
echo "PAIRED_CI_JOB_OK -> $OUT $(date -Is)"
