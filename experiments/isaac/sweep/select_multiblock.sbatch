#!/bin/bash
# ============================================================================
# Per-dataset SELECT over the multi-block probe sweep: picks (t, k, C) on val across all 14
# blocks, unseals exactly ONE test vector, writes the eval.compare-ready result + the val-grid
# plot (results/eval/sweep_mb/sweep-select_<ds>_<arm>_<hash>.{npz,png}). One test use per
# dataset -- count it. Runs after that dataset's probe array (afterok), on CPU.
# ============================================================================
#SBATCH --job-name=geobench-select-mb
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: a worktree's .git is a file
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
python3 -c "import sys, numpy, sklearn, matplotlib; assert 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3)" >&2; exit 1; }
[ -f "$PROJECT_ROOT/.secrets/wandb-personal.env" ] && source "$PROJECT_ROOT/.secrets/wandb-personal.env"
export WANDB_MODE="${WANDB_MODE:-offline}"
set -euo pipefail

DATASET="${DATASET:?submit with --export=ALL,DATASET=<OFFICIAL key>}"
ARM="${ARM:-flux-oneshot-ens8-mb}"
OUT_DIR="${OUT_DIR:-results/eval/sweep_mb}"
N_EXPECT="${N_BLOCKS:-14}"
n="$(compgen -G "$OUT_DIR/sweep-block_${DATASET}_${ARM}-k*_*.npz" | wc -l || true)"
[ "$n" = "$N_EXPECT" ] || { echo "FATAL: $n block results for $DATASET, expected $N_EXPECT" >&2; exit 1; }
python3 -m eval.sweep select --dataset "$DATASET" --arm "$ARM" \
    --blocks "$OUT_DIR/sweep-block_${DATASET}_${ARM}-k*_*.npz" --out-dir "$OUT_DIR"
