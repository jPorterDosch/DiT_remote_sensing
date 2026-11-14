#!/bin/bash
# Block selection for the GeoCore-9B caches (spec §4.4): the official sweep protocol
# (eval.sweep block: StandardScaler + LR over 9 t + concat candidates x the wide C grid, val-scored)
# at every cached block, one array task per block, then select.sbatch picks the block on val.
# Same code path as experiments/isaac/sweep/probe_multiblock.sbatch (the frozen FLUX selection),
# with the GeoCore pins: weights=geocore-9b, guidance_scale=0.0, caches under models/geocore_eval.
#
#   sbatch --array=0-7 --export=ALL,DATASET=m_eurosat experiments/isaac/geocore/probe_blocks.sbatch
#   RES=256 -> models/geocore_eval_img256, arm geocore-9b-img256. GATE=1 on the first task per dataset.
#   m_forestnet (7-class, slow C grid) and m_bigearthnet need --time=8:00:00 on long / campus-gpu-bigmem.
#
#SBATCH --job-name=geocore-probe-mb
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OVR_N_JOBS="${OVR_N_JOBS:-$SLURM_CPUS_PER_TASK}"
export JOBLIB_TEMP_FOLDER="$STORE/tmp/joblib"
mkdir -p "$JOBLIB_TEMP_FOLDER"
export WANDB_MODE="${WANDB_MODE:-disabled}"
python3 -c "import sys, numpy, sklearn; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

DATASET="${DATASET:?submit with --export=ALL,DATASET=<OFFICIAL key>}"
RES="${RES:-224}"
BLOCKS="${BLOCKS:-4-8-12-16-20-24-28-31}"
read -r -a BLOCK_LIST <<<"${BLOCKS//-/ }"
case "$RES" in
    224) CACHE_DIR="models/geocore_eval/$DATASET"; ARM="${ARM:-geocore-9b}" ;;
    256) CACHE_DIR="models/geocore_eval_img256/$DATASET"; ARM="${ARM:-geocore-9b-img256}" ;;
    *) echo "FATAL: RES must be 224 or 256" >&2; exit 1 ;;
esac
OUT_DIR="${OUT_DIR:-results/eval/sweep_mb}"
FEATS="$CACHE_DIR/*/multistep_{split}_feats_oneshot_g0.0_k${BLOCKS}.npz"
EXPECT=(extraction_mode=ONESHOT ensemble_size=8 weights=geocore-9b guidance_scale=0.0 hidden_size=4096)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job (--array=0-$((${#BLOCK_LIST[@]} - 1)))}"
[ "$TASK" -lt "${#BLOCK_LIST[@]}" ] || { echo "task $TASK beyond ${#BLOCK_LIST[@]} blocks, nothing to do"; exit 0; }
K="${BLOCK_LIST[$TASK]}"
echo "=== geocore probe-mb $DATASET res=$RES k=$K arm=$ARM host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} $(date -Is)"
for split in train val test; do
    compgen -G "${FEATS//\{split\}/$split}" >/dev/null || { echo "FATAL: no $split cache for ${FEATS//\{split\}/$split}" >&2; exit 1; }
done
gate() { python3 -m eval.sweep gate --phase "$1" --dataset "$DATASET" --k "$2" --features "$FEATS" --out-dir "$OUT_DIR" --arm "$ARM" --expect "${EXPECT[@]}"; }
[ "${GATE:-0}" = 1 ] && gate slice "$K"
done_glob=("$OUT_DIR"/sweep-block_"${DATASET}"_"${ARM}"-k"${K}"_*.npz)
if [ -f "${done_glob[0]}" ]; then
    rc=0
    python3 -m eval.sweep done-status "${done_glob[0]}" --dataset "$DATASET" --features "$FEATS" --cache-dir "$CACHE_DIR" --c-grid wide || rc=$?
    case "$rc" in
        0) echo "k=$K: verified block result exists, nothing to do"; exit 0 ;;
        *) echo "FATAL: k=$K: ${done_glob[0]} is not a current result for these caches (code $rc) -- move it aside" >&2; exit 1 ;;
    esac
fi
python3 -m eval.sweep block --dataset "$DATASET" --k "$K" --arm "$ARM" --features "$FEATS" --expect "${EXPECT[@]}" --out-dir "$OUT_DIR" --c-grid wide
[ "${GATE:-0}" = 1 ] && gate control "$K"
echo "GEOCORE_PROBE_MB_OK $DATASET $RES k=$K $(date -Is)"
