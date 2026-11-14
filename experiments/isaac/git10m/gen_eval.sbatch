#!/bin/bash
# ============================================================================
# Post-hoc generation metrics of ONE checkpoint (run.py --task gen-eval, src/tasks/gen_eval.py):
# FID + KID of N fixed-seed samples against N real rows, with the training task's sampler, seeds
# and reference construction. The training runs score only 600 samples (the val holdout has 656
# rows); this scores every arm at a quotable N on an A40 (no H100 queue: FLUX bf16 24 GB + VAE +
# Inception fit in 48 GB).
#
#   CKPT   <run>/checkpoints/lora_last_*.pt; empty ("") = the pretrained model (the step-0 reference,
#          score it ONCE)
#   N      samples / reference rows (5000). N <= 656 draws the reference from the training runs'
#          val holdout exactly (HOLDOUT=10000: N=600 reproduces a run's meta.json "final" FID, the
#          consistency check). N > 656 needs a larger holdout: HOLDOUT=100 (default when N > 656)
#          takes every 100th row of REF_INDEX as the reference, which then OVERLAPS the training
#          rows (standard for FID-vs-train; say so when quoting). Same N / HOLDOUT / REF_INDEX for
#          every arm: gen_metrics.json records the reference sha1, compare only equal ones.
#   REF_INDEX  index npz the reference rows come from ($STORE/git10m/index_retained_cities_unique.npz)
#   GEN_N  sampler batch (4 = the training task's; seeds depend on it, keep 4)
#   GUIDANCE  guidance embedding (1.0 = the training / extraction operating point, PINNED_GUIDANCE).
#          FLUX.1-dev is guidance-distilled with a nominal 3.5; every arm trained at 1.0 with the null
#          prompt, so the step-0 -> final FID drop at 1.0 mixes domain adaptation with re-calibration
#          to that operating point. GUIDANCE=3.5 with CKPT="" scores the pretrained model at its
#          nominal point as the second reference row; quote which one.
#
#   sbatch --export=ALL,CKPT=$STORE/models/git10m/<run>/checkpoints/lora_last_00020000.pt experiments/isaac/git10m/gen_eval.sbatch
#   sbatch --export=ALL,CKPT="" experiments/isaac/git10m/gen_eval.sbatch        # pretrained reference
#
# Output: models/git10m_gen_eval/n<N>_h<HOLDOUT>_g<GUIDANCE>/<run name>/{gen_metrics.json, samples/,
# fid_ref/, grid_4x4.png}. N / HOLDOUT are align / git10m fields and NOT in the run name's config
# hash, so they are in the directory instead: scores at different N or references never share a dir.
# The run name carries lora_checkpoint, so one dir per checkpoint; run.py refuses an existing dir --
# remove it to rescore.
# ============================================================================
#SBATCH --job-name=git10m-gen-eval
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
python3 -c "import sys, numpy, torch, pyarrow, tyro; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
set -euo pipefail

CKPT="${CKPT-unset}"
[ "$CKPT" != "unset" ] || { echo "FATAL: set CKPT=<run>/checkpoints/lora_last_*.pt or CKPT=\"\" (pretrained) via --export" >&2; exit 1; }
[ -z "$CKPT" ] || [ -f "$CKPT" ] || { echo "FATAL: $CKPT missing" >&2; exit 1; }
N="${N:-5000}"
if [ -z "${HOLDOUT:-}" ]; then [ "$N" -le 656 ] && HOLDOUT=10000 || HOLDOUT=100; fi
GUIDANCE="${GUIDANCE:-1.0}"
REF_INDEX="${REF_INDEX:-$STORE/git10m/index_retained_cities_unique.npz}"
[ -f "$REF_INDEX" ] || { echo "FATAL: $REF_INDEX missing" >&2; exit 1; }
ARROW="$STORE/git10m/arrow"
[ "$(ls "$ARROW"/data-*-of-00787.arrow 2>/dev/null | wc -l)" -eq 787 ] || { echo "FATAL: $ARROW incomplete" >&2; exit 1; }

ARGS=(--task gen-eval --model.name flux --model.ensemble-size 1
      --dataset.name git10m --dataset.path "$ARROW"
      --git10m.index "$REF_INDEX" --git10m.holdout-every "$HOLDOUT" --git10m.no-hflip
      --img-size 224 224 --t 100 --k 28 --guidance-scale "$GUIDANCE" --cd
      --align.fid-n "$N" --align.gen-n "${GEN_N:-4}" --align.gen-steps 28 --align.fid-subsets 100
      --num-workers 8 --seed 42 --save-dir "$PROJECT_ROOT/models/git10m_gen_eval/n${N}_h${HOLDOUT}_g${GUIDANCE}")
[ -n "$CKPT" ] && ARGS+=(--lora-checkpoint "$CKPT")
echo "ckpt=${CKPT:-<pretrained>} n=$N holdout=$HOLDOUT guidance=$GUIDANCE ref=$REF_INDEX args: ${ARGS[*]}"
python3 "$PROJECT_ROOT/run.py" "${ARGS[@]}"
echo "GIT10M_GEN_EVAL_OK"
