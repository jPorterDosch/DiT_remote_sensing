#!/bin/bash
# ============================================================================
# m-eurosat LoRA adaptation — the BENCHMARK-TIER adapted arms (tier A of §9.1):
# adapt ONLY on m-eurosat's 16,200-image train split, validate on its real val
# split, final-eval on its 996-image test split.
#   sbatch --export=ALL,ARM=frozen     experiments/isaac/m_eurosat/finetune.sbatch
#   sbatch --export=ALL,ARM=default    experiments/isaac/m_eurosat/finetune.sbatch
#   sbatch --export=ALL,ARM=supervised experiments/isaac/m_eurosat/finetune.sbatch
#
# CONTAMINATION RULE (RESEARCH_NOTES 6v addendum): our lexicographic EuroSAT split and
# the GEO-Bench random partition overlap (~797 of the 996 m-eurosat test images are
# expected inside our train split). NO checkpoint may cross splits: a model adapted on
# our split must never be evaluated on m-eurosat, and vice versa. This job exists so
# the m-eurosat adapted numbers come from m-eurosat-train-only adaptation.
# The INTERNAL mechanism arms (P-anchor, §6i) live in isaac/eurosat_lora/ on our split.
#
# TIME (H100): 16,200-img train pool, same 5,000-step budget as eurosat_lora
#   => ~4-5 h/arm incl. final eval (996 x 7t forwards is small). 12 h = wide headroom.
#
# PINS explicit (rule 9) and enforced at task entry; 3-arm smoke green 2026-09-17
# (post-M1 code; the task path is dataset-agnostic, wrapper loaders exercised by the
# m-eurosat extraction smokes).
# ============================================================================
#SBATCH --job-name=meurosat-lora
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # caches + models/data/logs on Lustre scratch

source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -f "$PROJECT_ROOT/.secrets/wandb-personal.env" ] && source "$PROJECT_ROOT/.secrets/wandb-personal.env"
export WANDB_MODE="${WANDB_MODE:-offline}"

ARM="${ARM:?set ARM=frozen|default|supervised via --export}"
case "$ARM" in
    frozen)     ARM_FLAGS=(--freeze-backbone) ;;
    default)    ARM_FLAGS=() ;;
    supervised) ARM_FLAGS=(--supervised-finetune) ;;
    *) echo "bad ARM=$ARM" >&2; exit 1 ;;
esac
echo "arm=$ARM flags=${ARM_FLAGS[*]:-none}"

[ -d "$PROJECT_ROOT/data/m_eurosat_rgb/train" ] || {
    echo "data/m_eurosat_rgb missing — run experiments/isaac/m_eurosat/prepare.sh first" >&2; exit 1;
}

# Same grid/pins as isaac/eurosat_lora/finetune.sbatch — the ONLY differences are the
# dataset identity and the save dir.
T_GRID="100 180 260 340 420 500 580"

python3 "$PROJECT_ROOT/run.py" \
    --task finetune-diffusion \
    --model.name flux --model.ensemble-size 1 \
    --dataset.name m_eurosat \
    --dataset.path "$PROJECT_ROOT/data/m_eurosat_rgb" \
    --img-size 224 224 \
    --t $T_GRID \
    --k 28 \
    --guidance-scale 1.0 --mim-loss-weight 0.0 --mask-ratio 0.0 \
    --use-gradient-accumulation --gradient-accumulation-steps 4 \
    --wrap-output --cd \
    --max-train-steps 5000 --finetune-max-epochs 10 \
    --finetune-lr 1e-3 \
    --lora-rank 4 --lora-alpha 16.0 --lora-dropout 0.0 \
    --label-fraction 1.0 \
    --batch-size 1 --num-workers 4 --seed 42 \
    --log-val-steps 500 \
    --save-dir "$PROJECT_ROOT/models/m_eurosat_lora" \
    "${ARM_FLAGS[@]}"
