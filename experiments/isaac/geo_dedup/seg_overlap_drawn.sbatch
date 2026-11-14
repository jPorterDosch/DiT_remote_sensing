#!/bin/bash
# ============================================================================
# Measurement only (eval/seg_overlap_drawn.py): of the retained Git-10M rows that
# seg_overlap.sbatch found within BUFFER_M of a segmentation task, how many did each training
# run actually draw. Reads the training index, the match list of seg_overlap and each run
# directory's meta.json + stats.npz; writes nothing but <OUT>.json and <OUT>.drawn_rows.csv,
# and refuses to overwrite them. No index, manifest, checkpoint or run directory is changed.
# GATE 1: tests/test_seg_overlap_drawn.py must pass in this job before the measurement runs.
# GATE 2: the tool's own checks (GATE seg-overlap-drawn PASS): the rows the logged runs wrote
#         to stats.npz must equal the reconstruction slot for slot.
#
#   mkdir -p logs
#   BUFFER_M=1000 EXPECT_RUNS=24 sbatch experiments/isaac/geo_dedup/seg_overlap_drawn.sbatch
# Variables travel through the submitting shell's environment (sbatch exports it by default).
# ============================================================================
#SBATCH --job-name=seg-overlap-drawn
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=0:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
python3 -c "import sys, numpy, torch; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

BUFFER_M="${BUFFER_M:?set BUFFER_M (metres): selects results/geo_dedup/seg_overlap_b<BUFFER_M>}"
EXPECT_RUNS="${EXPECT_RUNS:?set EXPECT_RUNS (run directories under RUNS_ROOT)}"
DATASET="${DATASET:-m_pv4ger_seg}"
INDEX="${INDEX:-$STORE/git10m/index_all_cities_unique.npz}"
HOLDOUT_EVERY="${HOLDOUT_EVERY:-10000}"
RUNS_ROOT="${RUNS_ROOT:-$STORE/models/git10m}"
IN="${IN:-results/geo_dedup/seg_overlap_b${BUFFER_M}}"
OUT="${OUT:-results/geo_dedup/seg_overlap_drawn_b${BUFFER_M}}"
for p in "$INDEX" "$IN.json" "$IN.retained_matches.csv"; do
    [ -f "$p" ] || { echo "FATAL: $p missing" >&2; exit 1; }
done
[ -d "$RUNS_ROOT" ] || { echo "FATAL: $RUNS_ROOT missing" >&2; exit 1; }
for ext in .json .drawn_rows.csv; do
    [ -e "$OUT$ext" ] && { echo "FATAL: $OUT$ext exists (move it aside; nothing is overwritten)" >&2; exit 1; }
done
echo "seg-overlap-drawn dataset=$DATASET in=$IN index=$INDEX runs=$RUNS_ROOT out=$OUT host=$(hostname) $(date -Is)"

python3 -m pytest tests/test_seg_overlap_drawn.py -q -p no:cacheprovider -rfE
echo "SEG_OVERLAP_DRAWN_TESTS_OK"

python3 -m eval.seg_overlap_drawn --index "$INDEX" --holdout-every "$HOLDOUT_EVERY" \
    --matches "$IN.retained_matches.csv" --overlap-json "$IN.json" --dataset "$DATASET" \
    --runs-root "$RUNS_ROOT" --expect-runs "$EXPECT_RUNS" --out "$OUT"
[ -f "$OUT.json" ] || { echo "FATAL: $OUT.json not written" >&2; exit 1; }
echo "SEG_OVERLAP_DRAWN_JOB_OK -> $OUT.json $(date -Is)"
