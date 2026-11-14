#!/bin/bash
# ============================================================================
# SatDiFuser (DiffusionSat) pooled features for the matrix: eval/satdifuser.py dump, one GEO-Bench
# task per array index, all three official splits, from OUR exported RGB trees (split identity =
# every other arm's). Their LDMExtractor verbatim (inversion, save_timesteps 47/45/40, prompt
# "A satellite image"), spatially mean-pooled per pyramid scale. Their conda env (env.sh).
#
#   SMOKE=1   m_eurosat only, 64 strided images per split, PNG tree AND raw hdf5 (--raw-dir), then
#             `compare` (per-image cosine PNG vs raw, the quantization check); writes *_smoke*.npz
#             under models/satdifuser_feats_smoke/ (never quotable). Run first.
#   OUT       cache dir (default models/satdifuser_feats -> scratch)
#
#   sbatch --export=ALL,SMOKE=1 --array=0 --time=2:00:00 experiments/isaac/satdifuser/dump.sbatch
#   sbatch --array=0-5 experiments/isaac/satdifuser/dump.sbatch
# Output: <OUT>/satdifuser_<dataset>_<split>.npz + _meta.json; prints SATDIFUSER_DUMP_OK.
# ============================================================================
#SBATCH --job-name=sdf-dump
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # data/ and models/ -> scratch
source "$PROJECT_ROOT/experiments/isaac/satdifuser/env.sh" || exit 1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
DATASETS=(m_eurosat m_brick_kiln m_pv4ger m_so2sat m_forestnet m_bigearthnet)
IDX="${SLURM_ARRAY_TASK_ID:?submit with --array=0-5 (one GEO-Bench task per array index)}"
[ "$IDX" -ge 0 ] && [ "$IDX" -lt "${#DATASETS[@]}" ] || { echo "FATAL: array index $IDX not in 0..5" >&2; exit 1; }
DATASET="${DATASETS[$IDX]}"
# Gate 1 is enforced here, not in the README (review 2026-11-12, item 10): no feature of theirs is
# dumped, smoke included, until gate_eurosat.sbatch has written a PASS verdict for the gw fuser.
python3 "$PROJECT_ROOT/experiments/isaac/satdifuser/gate_check.py" require --dir "$SDF/gate" --fuser gw
MODEL_ROOT="$SDF_CKPT/resolution256"
COMMON=(--repo "$SDF_REPO" --model-root "$MODEL_ROOT" --num-workers "${SLURM_CPUS_PER_TASK:-8}")
if [ "${SMOKE:-0}" = 1 ]; then
    [ "$DATASET" = m_eurosat ] || { echo "FATAL: SMOKE=1 is m_eurosat only (--array=0): the raw hdf5 is fetched for it" >&2; exit 1; }
    OUT="${OUT:-models/satdifuser_feats_smoke}"
    RAW="$GEO_BENCH_DIR/classification_v1.0/m-eurosat"
    [ -f "$RAW/default_partition.json" ] || { echo "FATAL: raw m-eurosat missing under $RAW" >&2; exit 1; }
    rm -rf "$OUT"
    python3 -m eval.satdifuser dump "${COMMON[@]}" --dataset m_eurosat --out-dir "$OUT" --max-images 64
    python3 -m eval.satdifuser dump "${COMMON[@]}" --dataset m_eurosat --out-dir "$OUT" --max-images 64 --raw-dir "$RAW"
    for s in train val test; do
        python3 -m eval.satdifuser compare --a "$OUT/satdifuser_m_eurosat_${s}_smoke.npz" --b "$OUT/satdifuser_m_eurosat_${s}_smoke_raw.npz"
    done
    echo "SDF_SMOKE_OK"
    exit 0
fi
OUT="${OUT:-models/satdifuser_feats}"
python3 -m eval.export_geobench --task "${DATASET//_/-}" --out "data/${DATASET}_rgb" >/dev/null  # tree == committed manifest (DiTF-free: numpy + PIL)
echo "=== satdifuser dump $DATASET -> $OUT $(date -Is)"
python3 -m eval.satdifuser dump "${COMMON[@]}" --dataset "$DATASET" --out-dir "$OUT"
echo "SDF_DUMP_OK $DATASET"
