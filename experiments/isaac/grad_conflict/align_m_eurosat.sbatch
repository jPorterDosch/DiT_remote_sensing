#!/bin/bash
# ============================================================================
# FM + semantic alignment on the m-eurosat train split through the pipeline's training task
# (run.py --task finetune-diffusion --dataset.name m_eurosat --align.arm ...): the gradient-conflict
# arms on the evaluation set's own images, one arm per job (the Git-10M runs are
# experiments/isaac/git10m/train.sbatch). The concurrent probe head of finetune-diffusion trains
# as usual (labels exist here); every arm computes BOTH gradients every step (matched compute),
# only the update rule differs (config_types.AlignConfig, src/tasks/align.py).
#   sbatch --export=ALL,ARM=fm    experiments/isaac/grad_conflict/align_m_eurosat.sbatch
#   sbatch --export=ALL,ARM=naive experiments/isaac/grad_conflict/align_m_eurosat.sbatch
#   sbatch --export=ALL,ARM=proj  experiments/isaac/grad_conflict/align_m_eurosat.sbatch
# SMOKE=1 (debug partition): 4 steps, tiny passes, save dir models/grad_conflict_smoke -- never
# quotable. m_eurosat refuses --max-samples, so the smoke still walks the full val subset and
# the final test evaluation (996 images x 7 t); budget ~40 min.
#   sbatch -p ai-tenn-debug -q ai-tenn-debug --time=0:40:00 --export=ALL,ARM=proj,SMOKE=1 \
#       experiments/isaac/grad_conflict/align_m_eurosat.sbatch
# ALIGN_WEIGHT (0.5), ALIGN_TARGET (dinov3_vit7b16_sat), TRAIN_STEPS (400), BATCH_SIZE (2),
# GRAD_ACCUM (2); EXTRA="..." appends run.py flags. Run dir:
# models/grad_conflict/m_eurosat_flux_<hash>+42+<arm>-<8hex>/ (run.py refuses an existing dir).
# MEMORY: FLUX bf16 (24 GB) + DINOv3 ViT-7B bf16 (13 GB) + activations for two LoRA backwards.
# ============================================================================
#SBATCH --job-name=grad-conflict
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
# -e, not -d: inside a git WORKTREE .git is a file, and the repo-wide -d bootstrap resolves PROJECT_ROOT to /.
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # caches + models/data/logs on Lustre scratch

source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
# Pin the env (sweep.sbatch): --export=ALL otherwise inherits the submitting shell's python3.
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
python3 -c "import sys, numpy, torch; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE=disabled  # the script logs to files only; eval.extract_dino imports wandb
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
set -euo pipefail

ARM="${ARM:?set ARM=fm|naive|proj via --export}"
case "$ARM" in fm|naive|proj) ;; *) echo "bad ARM=$ARM" >&2; exit 1 ;; esac
[ -d "$PROJECT_ROOT/data/m_eurosat_rgb/train" ] || { echo "data/m_eurosat_rgb missing" >&2; exit 1; }

ARGS=(--task finetune-diffusion --model.name flux --model.ensemble-size 1
      --dataset.name m_eurosat --dataset.path "$PROJECT_ROOT/data/m_eurosat_rgb"
      --align.arm "$ARM" --align.target "${ALIGN_TARGET:-dinov3_vit7b16_sat}" --align.weight "${ALIGN_WEIGHT:-0.5}"
      --align.proj-scope block --align.meas-t 100 260 420 580 740 900
      --img-size 224 224 --t 100 180 260 340 420 500 580 --k 28
      --guidance-scale 1.0 --mim-loss-weight 0.0 --mask-ratio 0.0
      --lora-rank 4 --lora-alpha 16.0 --lora-dropout 0.0 --wrap-output --cd
      --finetune-lr 1e-3 --use-gradient-accumulation --label-fraction 1.0
      --num-workers 4 --seed 42 --log-train-steps 10)
if [ "${SMOKE:-0}" = 1 ]; then
    ARGS+=(--max-train-steps 4 --warmup-steps 1 --align.proj-warmup-steps 4 --batch-size 2
           --gradient-accumulation-steps 2 --align.meas-batches 2 --log-val-steps 2
           --align.gen-n 2 --align.gen-steps 4 --save-dir "$PROJECT_ROOT/models/grad_conflict_smoke")
else
    ARGS+=(--max-train-steps "${TRAIN_STEPS:-400}" --warmup-steps 20 --align.proj-warmup-steps 100
           --batch-size "${BATCH_SIZE:-2}" --gradient-accumulation-steps "${GRAD_ACCUM:-2}"
           --align.meas-batches 8 --log-val-steps 100 --align.gen-n 4 --align.gen-steps 28
           --save-dir "$PROJECT_ROOT/models/grad_conflict")
fi
# shellcheck disable=SC2206
EXTRA_ARGS=(${EXTRA:-})
echo "arm=$ARM smoke=${SMOKE:-0} args: ${ARGS[*]} ${EXTRA_ARGS[*]:-}"
python3 "$PROJECT_ROOT/run.py" "${ARGS[@]}" "${EXTRA_ARGS[@]}"
echo "ALIGN_TRAIN_OK"
