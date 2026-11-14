#!/bin/bash
# ============================================================================
# Second-pass geographic exclusion: Git-10M public-source rows located through their
# authors' coordinates (SSL4EO-S12 centres, SkyScript boxes) and the id_N GEO-Bench tasks
# located by pixel join to their originals (EuroSAT MS GeoTIFFs, BigEarthNet patch index).
# Three steps, each refusing to overwrite its output, so a resubmission resumes (step 1 is
# also runnable alone, ahead of the BigEarthNet index: STEP=source-boxes):
#   1. source-boxes  -> $STORE/geo_dedup/source_boxes.parquet (+ .report.json)
#   2. eval-points   -> $STORE/geo_dedup/points_v2.csv       (+ .report.json)
#   3. dedup         -> $STORE/geo_dedup/dedup_v2_b<BUFFER_M>/
# GATES: SSL4EO mapping verification (ssl4eo-verify) must have passed; the SkyScript meta
# extraction marker must exist; the BigEarthNet patch index must exist.
#
#   sbatch --export=ALL,BUFFER_M=1000,STEP=source-boxes experiments/isaac/geo_dedup/dedup_v2.sbatch
#   sbatch --export=ALL,BUFFER_M=1000 --dependency=afterok:<index job> experiments/isaac/geo_dedup/dedup_v2.sbatch
# Third pass = the second plus m-so2sat at CITY level (So2Sat LCZ42's 52 city ROIs, ~50 x 50 km,
# experiments/isaac/geo_dedup/so2sat_cities.csv): CITIES=<csv> adds a city-points step
# (points_v3_cities.csv) and writes dedup_v3_b<BUFFER_M>_cities/. POINTS / OUT override both.
#   sbatch --export=ALL,BUFFER_M=1000,CITIES=experiments/isaac/geo_dedup/so2sat_cities.csv experiments/isaac/geo_dedup/dedup_v2.sbatch
# ============================================================================
#SBATCH --job-name=git10m-dedup-v2
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
python3 -c "import sys, pandas, pyarrow, scipy, rasterio, h5py, geobench; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

BUFFER_M="${BUFFER_M:?set BUFFER_M (metres)}"
G="$STORE/geo_dedup"
SM="$G/source_meta"
META="$G/git10m_meta"
n=$(ls "$META" | grep -c -E '^meta-[0-9]{5}\.parquet$')
[ "$n" -eq 787 ] || { echo "FATAL: $n/787 meta parquets in $META" >&2; exit 1; }
python3 -c "import json,sys; r=json.load(open('$SM/ssl4eo/verify.json')); sys.exit(0 if r['pass'] else 1)" \
    || { echo "FATAL: $SM/ssl4eo/verify.json missing or not PASS (run ssl4eo-verify)" >&2; exit 1; }
( cd "$SM/skyscript" && sha256sum -c --quiet meta6.zip.sha256 ) || { echo "FATAL: SkyScript meta6.zip sha256 mismatch" >&2; exit 1; }

BOXES="$G/source_boxes.parquet"
if [ "${STEP:-all}" = "source-boxes" ]; then
    [ -f "$BOXES" ] && { echo "FATAL: $BOXES exists" >&2; exit 1; }
    python3 -m eval.geo_dedup source-boxes --meta "$META" --expect-shards 787 \
        --ssl4eo-centres "$SM/ssl4eo/ssl4eo_centers.bin" --ssl4eo-void "$SM/ssl4eo/void_ids.csv" \
        --skyscript-meta "$SM/skyscript/meta6.zip" --out "$BOXES"
    echo "SOURCE_BOXES_OK -> $BOXES"; exit 0
fi
if [ ! -f "$BOXES" ]; then
    python3 -m eval.geo_dedup source-boxes --meta "$META" --expect-shards 787 \
        --ssl4eo-centres "$SM/ssl4eo/ssl4eo_centers.bin" --ssl4eo-void "$SM/ssl4eo/void_ids.csv" \
        --skyscript-meta "$SM/skyscript/meta6.zip" --out "$BOXES"
fi
[ -f "$SM/bigearthnet/patch_index.parquet" ] || { echo "FATAL: BigEarthNet patch index missing" >&2; exit 1; }
[ -d "$SM/eurosat/EuroSAT_MS" ] || { echo "FATAL: $SM/eurosat/EuroSAT_MS missing" >&2; exit 1; }
POINTS_V2="$G/points_v2.csv"
if [ ! -f "$POINTS_V2" ]; then
    python3 -m eval.geo_dedup eval-points --geobench-root "$STORE/datasets" --raw-root "$G/geobench_raw" \
        --originals-root "$SM" --out "$POINTS_V2"
fi
if [ -n "${CITIES:-}" ]; then
    [ -f "$CITIES" ] || { echo "FATAL: CITIES=$CITIES missing" >&2; exit 1; }
    POINTS="${POINTS:-$G/points_v3_cities.csv}"
    if [ ! -f "$POINTS" ]; then
        python3 -m eval.geo_dedup city-points --points "$POINTS_V2" --cities "$CITIES" --out "$POINTS"
    else
        # Reuse only a points file built from THIS city list (sha256 recorded in its report):
        # an edited csv must not produce a manifest from the old cities. Exits 1 on mismatch.
        python3 -m eval.geo_dedup city-points --points "$POINTS_V2" --cities "$CITIES" --out "$POINTS" --check-existing
    fi
    OUT="${OUT:-$G/dedup_v3_b${BUFFER_M}_cities}"
else
    POINTS="${POINTS:-$POINTS_V2}"
    OUT="${OUT:-$G/dedup_v2_b${BUFFER_M}}"
fi
echo "points=$POINTS out=$OUT cities=${CITIES:-none}"
python3 -m eval.geo_dedup dedup --meta "$META" --expect-shards 787 --points "$POINTS" \
    --source-boxes "$BOXES" --buffer-m "$BUFFER_M" --out "$OUT"
echo "DEDUP_V2_OK -> $OUT points=$POINTS"
