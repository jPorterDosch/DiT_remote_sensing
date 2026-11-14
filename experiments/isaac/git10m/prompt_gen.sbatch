#!/bin/bash
# Text-conditioned generation check (reviewer objection 2026-11-03), one array task per arm:
#   sbatch --array=0-3 experiments/isaac/git10m/prompt_gen.sbatch              # pretrained, proj, naive, fm (seed 42)
#   sbatch --array=0 --time=1:30:00 --export=ALL,SMOKE=1 experiments/isaac/git10m/prompt_gen.sbatch
# Array index: 0 pretrained, 1 proj-s42, 2 naive-s42, 3 fm-s42, 4 sem-s42 (appended 2026-11-11; 0-3 unchanged).
# Env: REF (coco_ref.py dir; default n1000 512 px), N, IMG, STEPS, GUIDANCE, BATCH, OUT_ROOT, SMOKE=1 (n 8, 256 px, 4 steps).
# Needs the COCO reference first: experiments/isaac/git10m/coco_ref.py (CPU, short).
# PROMPTS=<list.json> switches to a prompt list with no paired images (no FID / KID), e.g. the remote-sensing set:
#   PROMPTS=experiments/isaac/git10m/rs_prompts.json sbatch --array=0-4 --time=1:00:00 experiments/isaac/git10m/prompt_gen.sbatch
#   outputs -> $STORE/models/prompt_gen_rs/<tag>_<list name>_n<N>_<IMG>px_g<G>_s<STEPS>/ ; SMOKE=1: first 4 prompts, 256 px, 4 steps.
#SBATCH --job-name=prompt-gen
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"; cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export WANDB_MODE=disabled
python3 -c "import sys, torch, tyro, transformers; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3)" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
set -euo pipefail
RUNS="$STORE/models/git10m"
TAGS=(pretrained proj-s42 naive-s42 fm-s42 sem-s42)
CKPTS=("" "$RUNS/git10m_flux_ca8e6ce7+42+proj-ffcd7963/checkpoints/lora_last_step00020000.pt"
          "$RUNS/git10m_flux_ca8e6ce7+42+naive-11e56b1e/checkpoints/lora_last_step00020000.pt"
          "$RUNS/git10m_flux_ca8e6ce7+42+fm-8c098506/checkpoints/lora_last_step00020000.pt"
          "$RUNS/git10m_flux_ca8e6ce7+42+sem-55ae3c4c/checkpoints/lora_last_step00020000.pt")
i="${SLURM_ARRAY_TASK_ID:-0}"
[ "$i" -ge 0 ] && [ "$i" -lt "${#TAGS[@]}" ] || { echo "FATAL: array index $i, arms are 0..$(( ${#TAGS[@]} - 1 ))" >&2; exit 1; }
TAG="${TAGS[$i]}"; CKPT="${CKPTS[$i]}"
[ -z "$CKPT" ] || [ -f "$CKPT" ] || { echo "FATAL: $CKPT missing" >&2; exit 1; }
if [ -n "${PROMPTS:-}" ]; then  # prompt list, no paired reference images
  [ -z "${REF:-}" ] || { echo "FATAL: set PROMPTS or REF, not both" >&2; exit 1; }
  [ -f "$PROMPTS" ] || { echo "FATAL: prompt list $PROMPTS missing" >&2; exit 1; }
  SET="$(basename "$PROMPTS" .json)"
  if [ "${SMOKE:-0}" = 1 ]; then
    N="${N:-4}"; IMG="${IMG:-256}"; STEPS="${STEPS:-4}"; OUT_ROOT="${OUT_ROOT:-$STORE/models/prompt_gen_rs_smoke}"
  else
    N="${N:-$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$PROMPTS")}"
    IMG="${IMG:-512}"; STEPS="${STEPS:-28}"; OUT_ROOT="${OUT_ROOT:-$STORE/models/prompt_gen_rs}"
  fi
  SRC=(--prompts "$PROMPTS"); OUT="$OUT_ROOT/${TAG}_${SET}_n${N}_${IMG}px_g${GUIDANCE:-3.5}_s${STEPS}"
else
  if [ "${SMOKE:-0}" = 1 ]; then
    N="${N:-8}"; IMG="${IMG:-256}"; STEPS="${STEPS:-4}"; REF="${REF:-$STORE/git10m/fid_ref_coco_val2017_n8_256}"; OUT_ROOT="${OUT_ROOT:-$STORE/models/prompt_gen_smoke}"
  else
    N="${N:-1000}"; IMG="${IMG:-512}"; STEPS="${STEPS:-28}"; REF="${REF:-$STORE/git10m/fid_ref_coco_val2017_n1000_512}"; OUT_ROOT="${OUT_ROOT:-$STORE/models/prompt_gen}"
  fi
  [ -f "$REF/prompts.json" ] || { echo "FATAL: reference $REF missing (run coco_ref.py)" >&2; exit 1; }
  SRC=(--ref "$REF"); OUT="$OUT_ROOT/${TAG}_n${N}_${IMG}px_g${GUIDANCE:-3.5}_s${STEPS}"
fi
[ ! -e "$OUT/manifest.json" ] || { echo "FATAL: $OUT already holds a finished run; move it aside, never overwritten" >&2; exit 1; }
echo "=== prompt-gen $TAG ckpt=${CKPT:-<pretrained>} n=$N img=$IMG steps=$STEPS g=${GUIDANCE:-3.5} ${SRC[*]} -> $OUT $(date -Is)"
python3 experiments/isaac/git10m/prompt_gen.py --tag "$TAG" --ckpt "$CKPT" "${SRC[@]}" --n "$N" --img-size "$IMG" \
  --steps "$STEPS" --guidance "${GUIDANCE:-3.5}" --batch "${BATCH:-4}" --seed 0 --out "$OUT"
