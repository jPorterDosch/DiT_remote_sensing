#!/bin/bash
# ============================================================================
# m-eurosat ONE-SHOT extraction (frozen FLUX, ens8, 7-t grid, k=28, g=1.0) — UNSHARDED.
# One submission per split (the three splits are separate runs, not shards):
#   sbatch --export=ALL,SPLIT=val   experiments/isaac/extract_m_eurosat_oneshot.sbatch
#   sbatch --export=ALL,SPLIT=test  experiments/isaac/extract_m_eurosat_oneshot.sbatch
#   sbatch --export=ALL,SPLIT=train experiments/isaac/extract_m_eurosat_oneshot.sbatch
# (or use experiments/isaac/submit_m_eurosat.sh, which chains them.)
#
# TIME: 7 t x 8 ensemble members = 56 DiT forwards + 8 VAE encodes per image.
#   Measured 4.99 s/img on the workstation (A6000-class). train = 16,200 imgs ~= 22.5 h
#   at that rate — TIGHT against the 24 h campus-gpu wall. If the campus GPU is slower,
#   the train split WILL be killed; in that case either rerun with the shard flags
#   (--num-shards 2 --shard-index {0,1} — the probe's merge step handles shard caches)
#   or request a longer-wall partition. val/test are 996 imgs ~= 1.4 h each.
#
# Seeds: eps/global seeds DIFFER per split (train 42, val 44, test 43) — position-paired
# noise reuse across splits is a structural coupling (RESEARCH_NOTES 6t, audit F1); the
# probe REFUSES caches whose eps seeds collide.
# ============================================================================
#SBATCH --job-name=meurosat-oneshot
#SBATCH -A acf-utk0011
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
[ -f "$PROJECT_ROOT/.secrets/wandb-personal.env" ] && source "$PROJECT_ROOT/.secrets/wandb-personal.env"
export WANDB_MODE="${WANDB_MODE:-offline}"

SPLIT="${SPLIT:?set SPLIT=train|val|test via --export}"
case "$SPLIT" in
    train) SEED=42 ;;
    test)  SEED=43 ;;
    val)   SEED=44 ;;
    *) echo "bad SPLIT=$SPLIT" >&2; exit 1 ;;
esac

python3 "$PROJECT_ROOT/run.py" \
    --task extract \
    --model.name flux --model.ensemble-size 8 \
    --dataset.name m_eurosat \
    --dataset.path "$PROJECT_ROOT/data/m_eurosat_rgb" \
    --img-size 224 224 \
    --t 100 180 260 340 420 500 580 \
    --extraction-mode ONESHOT \
    --k 28 --guidance-scale 1.0 \
    --seed "$SEED" --eps-seed "$SEED" --subset-seed 42 \
    --extract-split "$SPLIT" \
    --label-fraction 1.0 --batch-size 1 \
    --save-dir "$PROJECT_ROOT/models/m_eurosat_oneshot_ens8"
