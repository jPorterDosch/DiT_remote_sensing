#!/bin/bash
# ============================================================================
# DINOv3 baseline arm on every GEO-Bench task (every eval/features.OFFICIAL key; --array must
# match their count). ISAAC only: the gated DINOv3 checkpoints live on scratch,
# ditf_models/dinov3/<hub filename> (ditf_models/ is a scratch symlink).
# One array task per dataset (registry order), independent and re-runnable:
#   sbatch --export=ALL,PRESET=dinov3_vitl16_web experiments/isaac/geobench/dino.sbatch
#   sbatch --export=ALL,PRESET=dinov3_vitl16_sat experiments/isaac/geobench/dino.sbatch
#   (7B: PRESET=dinov3_vit7b16_{web,sat}, --time=24:00:00)
# Per dataset: (1) the pre-registered control gate (tests/test_eval_gates.py::
# test_dino_control_geobench) must PASS -- a skip (missing weights/data) or FAIL stops the
# task; (2) extract_dino -> models/dino_feats/ (scratch); (3) probe --protocol official with
# the features.DINO_OFFICIAL_VIEWS x C selection on val, one test eval -> results/eval/.
# Prereq (login node): experiments/isaac/geobench/prepare.sh (exports + DINOv3 hub code).
# TIME: ViT-L extraction is minutes per task; m-bigearthnet's one-vs-rest probes dominate
# (~30 min for the gate + ~1 h for the arm). 12 h leaves room for 7B-on-small-tasks too.
# ============================================================================
#SBATCH --job-name=geobench-dino
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # caches + models/data/ditf_models on scratch

source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -f "$PROJECT_ROOT/.secrets/wandb-personal.env" ] && source "$PROJECT_ROOT/.secrets/wandb-personal.env"
export WANDB_MODE="${WANDB_MODE:-offline}"
set -euo pipefail

PRESET="${PRESET:?submit with --export=ALL,PRESET=dinov3_vitl16_{web,sat}}"
KEY="$(python3 -c "import sys; from eval.features import OFFICIAL; print(list(OFFICIAL)[int(sys.argv[1])])" \
    "${SLURM_ARRAY_TASK_ID:?run as an array job}")"  # array index -> OFFICIAL key, in registry order
echo "=== $KEY $PRESET $(date -Is)"

python3 -m eval.export_geobench --task "${KEY//_/-}" --out "data/${KEY}_rgb"  # tree == manifest
GATE_LOG="$(mktemp)"
WANDB_MODE=disabled python3 -m pytest tests/test_eval_gates.py --gpu -s -rs \
    -k "dino_control_geobench and $KEY and $PRESET" 2>&1 | tee "$GATE_LOG" || true
grep -q "1 passed" "$GATE_LOG" || { echo "FATAL: control gate did not PASS for $KEY $PRESET" >&2; exit 1; }

python3 -m eval.extract_dino --preset "$PRESET" --dataset "$KEY" --out-dir models/dino_feats
python3 -m eval.probe --protocol official --dataset "$KEY" --arm "$PRESET" --kind dino \
    --features "models/dino_feats/${PRESET}_${KEY}_{split}.npz"
echo "=== $KEY $PRESET done $(date -Is)"
