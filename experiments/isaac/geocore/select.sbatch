#!/bin/bash
# Per-dataset SELECT over the GeoCore block sweep: (t, k, C) chosen on val across the 8 blocks
# (eval.sweep select; ONE test use, printed for the record and never quoted -- the matrix cells
# are the reported numbers). Writes results/eval/sweep_mb/sweep-select_<ds>_<arm>_*.{npz,png}
# and prints the per_dataset k to paste into eval/matrix_specs/phase8_geocore.json.
#
#   sbatch --export=ALL,DATASET=m_eurosat --dependency=afterok:<probe array> experiments/isaac/geocore/select.sbatch
#
#SBATCH --job-name=geocore-select-mb
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export WANDB_MODE="${WANDB_MODE:-disabled}"
set -euo pipefail
DATASET="${DATASET:?submit with --export=ALL,DATASET=<OFFICIAL key>}"
RES="${RES:-224}"
case "$RES" in 224) ARM="${ARM:-geocore-9b}" ;; 256) ARM="${ARM:-geocore-9b-img256}" ;; *) echo "FATAL: RES" >&2; exit 1 ;; esac
OUT_DIR="${OUT_DIR:-results/eval/sweep_mb}"
n="$(compgen -G "$OUT_DIR/sweep-block_${DATASET}_${ARM}-k*_*.npz" | wc -l || true)"
[ "$n" = 8 ] || { echo "FATAL: expected 8 block results for $DATASET/$ARM, found $n" >&2; exit 1; }
python3 -m eval.sweep select --dataset "$DATASET" --arm "$ARM" --blocks "$OUT_DIR/sweep-block_${DATASET}_${ARM}-k*_*.npz" --out-dir "$OUT_DIR"
echo "GEOCORE_SELECT_OK $DATASET $RES $(date -Is)"
