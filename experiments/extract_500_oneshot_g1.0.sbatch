#!/bin/bash
# ============================================================================
# Job 2/3 — one-shot extraction, guidance 1.0 (the guidance-matched control)
#   Same 500-image stratified subset / t / k as the inversion job; independent
#   one-shot noising per t, one eps per image (ensemble_size=1) shared across
#   all 7 timesteps, eps stream seeded by eps_seed(=seed=42) in dataset order.
#   Cache: models/paired_500/<run>/multistep_train_feats_oneshot_g1.0.npz
#
# TIME ESTIMATE:
#   7 NFE/image (7 timesteps x ensemble 1) + 1 VAE encode ~= 3-4 s/image at
#   ~0.4-0.5 s/forward (bf16, 224px, batch 1, A6000-class GPU)
#   -> 500 images ~= 30-40 min, plus ~5 min model load.
#   Tighter request if desired: 02:00:00 (~2x headroom). --time is set to 24h.
#
# NOTE: ensemble_size is pinned to 1 explicitly so the "7 NFE vs ~59 NFE"
# comparison holds and both one-shot arms are identical in compute. If your
# existing g=3.5 cache used the ensemble_size=8 default, it is NOT the same
# object as these caches — see COMMANDS.md (job-3 decision).
# ============================================================================
#SBATCH --job-name=extract500-one-g1.0
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
    --guidance_scale 1.0 \
    --t 100 180 260 340 420 500 580 \
    --k 28 \
    --model.ensemble-size 1 \
    --subset_size 500 \
    --subset_seed 42 \
    --seed 42 \
    --eps_seed 42 \
    --label_fraction 1.0 \
    --batch_size 1
