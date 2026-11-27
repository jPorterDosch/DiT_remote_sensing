#!/bin/bash
# ============================================================================
# MedSigLIP-448 probe-baseline features of one chest X-ray probe tree (eval/extract_medsiglip.py,
# design section 7.2). One GPU; MedSigLIP's vision tower is about 0.4 B parameters, so a V100S is
# enough (fp32).
#
#   # once, on a login node (download only, no compute): accept the model's terms on Hugging Face,
#   # log in (`huggingface-cli login`, token stored under $HF_HOME on scratch), then
#   #   source experiments/isaac/scratch_env.sh && PYTHONPATH=src python3 -c \
#   #     "from models.medsiglip import snapshot; print(snapshot())"
#   mkdir -p logs
#   sbatch --export=ALL,DATASET=chestxray14,VIEW=224up[,SMOKE=1][,REVISION=<sha>] experiments/isaac/cxr/extract_medsiglip.sbatch
#
#   DATASET   chestxray14 | chexpert | vindr_cxr_kaggle
#   VIEW      224up | 448
#   SMOKE=1   64 images per split (strided), written with a _smoke suffix
#   REVISION  optional Hugging Face revision pin (default: the cached snapshot; recorded either way)
#
# Output: $D/eval_feats/medsiglip448_<VIEW>_<DATASET>_<split>[_smoke].npz under the mode-700 CXR
# tree (features of CheXpert / VinDr-CXR images stay with the images; ChestX-ray14 goes there too
# so the three rows share one root). Success is the line MEDSIGLIP_EXTRACT_OK; the python command's
# exit status is the script's.
# ============================================================================
#SBATCH --job-name=cxr-medsiglip
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=v100s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
export HF_HUB_OFFLINE=1  # compute nodes read the snapshot cached on the login node
python3 -c "import sys; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

: "${DATASET:?FATAL: set DATASET}"
: "${VIEW:?FATAL: set VIEW (224up | 448)}"
case "$DATASET" in chestxray14 | chexpert | vindr_cxr_kaggle) ;; *) echo "FATAL: bad DATASET $DATASET" >&2; exit 1 ;; esac
case "$VIEW" in 224up | 448) ;; *) echo "FATAL: VIEW must be 224up or 448" >&2; exit 1 ;; esac

D=/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/datasets/cxr
[ "$(stat -c %a "$D")" = "700" ] || { echo "FATAL: $D is not mode 700" >&2; exit 1; }
mkdir -p -m 700 "$D/eval_feats"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"

args=(--dataset "$DATASET" --view "$VIEW" --out-dir "$D/eval_feats")
[ "${SMOKE:-0}" = 1 ] && args+=(--max-images 64)
[ -z "${REVISION:-}" ] || args+=(--revision "$REVISION")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -m eval.extract_medsiglip "${args[@]}"
