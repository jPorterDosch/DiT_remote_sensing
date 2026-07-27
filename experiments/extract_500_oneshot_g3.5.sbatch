#!/bin/bash
# ============================================================================
# Job 3/3 — one-shot extraction, guidance 3.5 (the one-shot default arm)
#   SKIP THIS JOB iff experiments/verify_paired_caches.py --legacy <old.npz>
#   confirms your existing g=3.5 one-shot cache covers the identical 500 images
#   with matching settings (t, k, eps_seed=42, ensemble_size=1) — see
#   COMMANDS.md. Included because that could not be confirmed from this
#   working tree (no 500-image cache present here, and the original
#   extraction.py recorded neither guidance nor an extraction-mode tag).
#
#   Identical to job 2 except --guidance_scale 3.5; same eps_seed=42 so the
#   per-image eps pairs exactly with the g=1.0 arm.
#   Cache: models/paired_500/<run>/multistep_train_feats_oneshot_g3.5.npz
#
# TIME ESTIMATE: same as job 2 — 7 NFE/image, 500 images ~= 30-40 min.
#   Tighter request if desired: 02:00:00. --time is set to 24h.
# ============================================================================
#SBATCH --job-name=extract500-one-g3.5
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=24:00:00                 # 24h request; one-shot est. ~40min — see header
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/experiments/_common.sh" || {
    echo "FATAL: Failed to source _common.sh" >&2; exit 1;
}
setup_environment

[ -f "$PROJECT_ROOT/.secrets/wandb-personal.env" ] && source "$PROJECT_ROOT/.secrets/wandb-personal.env"
# export WANDB_MODE=offline            # TODO: uncomment if compute nodes have no egress

DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/data/eurosat/EuroSAT_RGB}"
# export FLUX_DEV="$PROJECT_ROOT/ditf_models/FLUX.1-dev/flux1-dev.safetensors"
# export AE="$PROJECT_ROOT/ditf_models/FLUX.1-dev/ae.safetensors"

python3 "$PROJECT_ROOT/run.py" \
    --task extract \
    --dataset.name eurosat \
    --dataset.path "$DATASET_PATH" \
    --save_dir "$PROJECT_ROOT/models/paired_500" \
    --extraction_mode ONESHOT \
    --guidance_scale 3.5 \
    --t 100 180 260 340 420 500 580 \
    --k 28 \
    --model.ensemble-size 1 \
    --subset_size 500 \
    --subset_seed 42 \
    --seed 42 \
    --eps_seed 42 \
    --label_fraction 1.0 \
    --batch_size 1
