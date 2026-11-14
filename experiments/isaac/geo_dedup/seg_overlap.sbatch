#!/bin/bash
# ============================================================================
# Measurement only (eval/seg_overlap.py): how many located Git-10M training rows lie within
# BUFFER_M of the three GEO-Bench segmentation tasks, which the geographic exclusion never
# covered. Reads the existing metadata, source boxes and dedup manifests; writes nothing but
# <OUT>.json and <OUT>.retained_matches.csv, and refuses to overwrite them. No manifest or
# training index is rebuilt.
# GATE 1: tests/test_seg_overlap.py must pass in this job before the measurement runs.
# GATE 2: the instrument control inside the tool (GATE instrument control PASS).
#
#   mkdir -p logs
#   BUFFER_M=1000 sbatch experiments/isaac/geo_dedup/seg_overlap.sbatch
# Variables travel through the submitting shell's environment (sbatch exports it by default).
# ============================================================================
#SBATCH --job-name=seg-overlap
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
python3 -c "import sys, pandas, pyarrow, scipy, h5py, geobench; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

BUFFER_M="${BUFFER_M:?set BUFFER_M (metres)}"
G="$STORE/geo_dedup"
SEG_ROOT="${SEG_ROOT:-/lustre/isaac24/scratch/$USER/satdifuser/geobench/segmentation_v1.0}"
DEDUP="${DEDUP:-$G/dedup_v3_b1000_cities}"
POINTS="${POINTS:-$G/points_v3_cities.csv}"
OUT="${OUT:-results/geo_dedup/seg_overlap_b${BUFFER_M}}"
META="$G/git10m_meta"
n=$(ls "$META" | grep -c -E '^meta-[0-9]{5}\.parquet$')
[ "$n" -eq 787 ] || { echo "FATAL: $n/787 meta parquets in $META" >&2; exit 1; }
for p in "$G/source_boxes.parquet" "$DEDUP/retained.csv" "$DEDUP/excluded.csv" "$POINTS" \
         "$SEG_ROOT/m-pv4ger-seg/default_partition.json" "$SEG_ROOT/m-chesapeake-landcover/default_partition.json" \
         "$SEG_ROOT/m-SA-crop-type/default_partition.json"; do
    [ -f "$p" ] || { echo "FATAL: $p missing" >&2; exit 1; }
done
for ext in .json .retained_matches.csv; do
    [ -e "$OUT$ext" ] && { echo "FATAL: $OUT$ext exists (move it aside; nothing is overwritten)" >&2; exit 1; }
done
echo "seg-overlap buffer=$BUFFER_M dedup=$DEDUP seg_root=$SEG_ROOT out=$OUT host=$(hostname) $(date -Is)"

python3 -m pytest tests/test_seg_overlap.py -q -p no:cacheprovider -rfE
echo "SEG_OVERLAP_TESTS_OK"

python3 -m eval.seg_overlap --seg-root "$SEG_ROOT" --meta "$META" --expect-shards 787 \
    --source-boxes "$G/source_boxes.parquet" --dedup "$DEDUP" --points "$POINTS" \
    --buffer-m "$BUFFER_M" --out "$OUT"
[ -f "$OUT.json" ] || { echo "FATAL: $OUT.json not written" >&2; exit 1; }
echo "SEG_OVERLAP_JOB_OK -> $OUT.json $(date -Is)"
