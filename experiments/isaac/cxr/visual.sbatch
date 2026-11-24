#!/bin/bash
# ============================================================================
# Visual-check figures and tables of one chest X-ray dataset (eval/cxr_visual.py), the no-Jupyter route.
#
#   mkdir -p logs
#   sbatch --export=ALL,DATASET=chestxray14[,OUT=<new dir>][,ROOT=<tree root>] experiments/isaac/cxr/visual.sbatch
#
#   DATASET     required; chestxray14 | chexpert | vindr_cxr_kaggle (anything else is fatal)
#   OUT         default $D/visual/${DATASET}_${SLURM_JOB_ID}; refuses to start when $OUT or $OUT.partial exists.
#               The CheXpert / VinDr-CXR figures show patient images; the script passes --private-root "$D" for all
#               three keys, so every OUT (chestxray14 too) must sit under the mode-700 $D.
#   ROOT        optional tree root (default: the registry's data/cxr/<DATASET>_v<N>)
#   RAW ARROW INDEX   optional path overrides (defaults below); GATE_JSON optional leakage gate JSON; N images per grid
#
# Success is the line CXR_VISUAL_OK printed by eval.cxr_visual; the python command's exit status is
# the script's exit status (nothing is echoed after it).
# ============================================================================
#SBATCH --job-name=cxr-visual
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
python3 -c "import sys; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

: "${DATASET:?FATAL: set DATASET}"
case "$DATASET" in chestxray14 | chexpert | vindr_cxr_kaggle) ;; *) echo "FATAL: DATASET must be chestxray14, chexpert or vindr_cxr_kaggle" >&2; exit 1 ;; esac

D=/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/datasets/cxr
[ "$(stat -c %a "$D")" = "700" ] || { echo "FATAL: $D is not mode 700" >&2; exit 1; }
mkdir -p -m 700 "$D/visual"
OUT="${OUT:-$D/visual/${DATASET}_${SLURM_JOB_ID}}"
[ ! -e "$OUT" ] && [ ! -L "$OUT" ] && [ ! -e "$OUT.partial" ] && [ ! -L "$OUT.partial" ] || { echo "FATAL: $OUT or $OUT.partial exists; pick a new OUT" >&2; exit 1; }
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-$USER-${SLURM_JOB_ID:-0}"

visual_args=(--dataset "$DATASET" --out "$OUT" --private-root "$D")
[ -z "${ROOT:-}" ] || visual_args+=(--root "$ROOT")
[ -z "${GATE_JSON:-}" ] || visual_args+=(--gate-json "$GATE_JSON")
[ -z "${N:-}" ] || visual_args+=(--n "$N")

case "$DATASET" in
    chestxray14)
        ARROW="${ARROW:-$D/chestxray14/arrow}"
        INDEX="${INDEX:-$D/chestxray14/index_v2.npz}"
        visual_args+=(--arrow-dir "$ARROW" --index "$INDEX")
        ;;
    chexpert)
        RAW="${RAW:-$D/chexpert/raw/chexpertchestxrays-u20210408}"
        visual_args+=(--raw-dir "$RAW")
        ;;
    vindr_cxr_kaggle)
        RAW="${RAW:-$D/vindr_cxr/raw}"
        visual_args+=(--raw-dir "$RAW")
        ;;
esac

python3 -m eval.cxr_visual "${visual_args[@]}"
