#!/bin/bash
# One whole-node ai-tenn job per adapted arm: train (1 GPU) -> extract all six GEO-Bench tasks
# (4 GPUs in parallel) -> submit the matrix cells. Everything in ONE allocation because this
# cluster's PriorityFlags is empty: a job waiting on --dependency accrues no age, so a
# train->extract chain would re-enter the ai-tenn queue at the bottom after training.
#
# This file is a thin launcher on purpose: SLURM copies the batch script at submission, so the
# real logic lives in arm_full.sh next to it and is read when the job STARTS -- fixes made while
# the job waits in the queue take effect. Only the SBATCH header and this exec are frozen.
#
#   sbatch --export=ALL,TAG=semt-lo,SEED=43,SEM_T_WEIGHTS="1 1 0 0 0 0 0 0 0" experiments/isaac/git10m/arm_full.sbatch
#   sbatch --export=ALL,TAG=web,SEED=43,ALIGN_TARGET=dinov3_vit7b16_web  experiments/isaac/git10m/arm_full.sbatch
#   sbatch --export=ALL,TAG=semt-hi,SEED=43,SEM_T_WEIGHTS="0 0 0 0 0 0 1 1 1" experiments/isaac/git10m/arm_full.sbatch
#   sbatch --export=ALL,TAG=weak,SEED=42,ALIGN_TARGET=dinov3_vitl16_web  experiments/isaac/git10m/arm_full.sbatch
#   SMOKE=1: tiny train + 20-image ens-2 extraction on 4 GPUs (campus-gpu-bigmem, -t 2:00:00)
#
#SBATCH --job-name=arm-full
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=400G
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
[ -f "$_dir/experiments/isaac/git10m/arm_full.sh" ] || { echo "FATAL: arm_full.sh missing under $_dir" >&2; exit 1; }
exec bash "$_dir/experiments/isaac/git10m/arm_full.sh"
