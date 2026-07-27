#!/bin/bash
# ============================================================================
# Job 1/3 — inversion-chain extraction, guidance 1.0
#   EuroSAT train, 500-image stratified subset (subset_seed=42), t=100..580,
#   k=28, RF-Solver chain with 50 integration steps.
#   Cache: models/paired_500/<run>/multistep_train_feats_inversion_g1.0_n50.npz
#
# TIME ESTIMATE (--time is set to 24h; tighten it to this if you want faster backfill):
#   NFE/image = 2 evals x 29 chain steps (t=580 on the 50-step grid) = 58 full
#   FLUX forwards + 1 early-exited feature pass (~0.6 forward) ~= 59 NFE.
#   At ~0.4-0.5 s/forward (bf16, 224px, batch 1, A6000-class GPU):
#   ~25-30 s/image -> 500 images ~= 3.5-4.5 h, plus ~5 min model load.
#   Tighter request if desired: 08:00:00 (~2x headroom). Faster GPUs (A100/H100): halve it.
# ============================================================================
#SBATCH --job-name=extract500-inv-g1.0
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=24:00:00                 # 24h request; inversion est. ~4h — see header
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# --- Bootstrap: find project root (same pattern as the other experiment scripts)
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

# TODO: adjust if extracting from a scratch copy of the dataset instead.
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/data/eurosat/EuroSAT_RGB}"
# Off-ISAAC only: point FLUX_DEV / AE at local weights (util.py defaults to ISAAC paths).
# export FLUX_DEV="$PROJECT_ROOT/ditf_models/FLUX.1-dev/flux1-dev.safetensors"
# export AE="$PROJECT_ROOT/ditf_models/FLUX.1-dev/ae.safetensors"

python3 "$PROJECT_ROOT/run.py" \
    --task extract \
    --dataset.name eurosat \
    --dataset.path "$DATASET_PATH" \
    --save_dir "$PROJECT_ROOT/models/paired_500" \
    --extraction_mode INVERSION \
    --guidance_scale 1.0 \
    --num_inversion_steps 50 \
    --t 100 180 260 340 420 500 580 \
    --k 28 \
    --subset_size 500 \
    --subset_seed 42 \
    --seed 42 \
    --label_fraction 1.0 \
    --batch_size 1
