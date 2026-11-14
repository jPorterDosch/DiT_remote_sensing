#!/bin/bash
# ============================================================================
# eval/satdifuser.py select: the pyramid scale per dataset (s32 / s16 / s8 / s4 / all) that the
# official protocol picks on the official VAL split at 100 % labels -- the SatDiFuser arm's
# "block", as eval.sweep select picked the FLUX blocks. Prints the per_dataset expect.scale block
# for eval/matrix_specs/phase3_satdifuser.json. DiTF env, CPU. Test is never read.
#
#   sbatch --array=0-5 experiments/isaac/satdifuser/select.sbatch
# Output: results/satdifuser_select/<dataset>/select.json
# ============================================================================
#SBATCH --job-name=sdf-select
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --qos=long
#SBATCH --partition=long
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
export WANDB_MODE=disabled
export OVR_N_JOBS="${OVR_N_JOBS:-$SLURM_CPUS_PER_TASK}"
export JOBLIB_TEMP_FOLDER="$STORE/tmp/joblib"
mkdir -p "$JOBLIB_TEMP_FOLDER"
python3 -c "import sys, numpy, sklearn, threadpoolctl; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail
DATASETS=(m_eurosat m_brick_kiln m_pv4ger m_so2sat m_forestnet m_bigearthnet)
IDX="${SLURM_ARRAY_TASK_ID:?submit with --array=0-5}"
DATASET="${DATASETS[$IDX]}"
FEATS="${FEATS:-models/satdifuser_feats/satdifuser_{dataset}_{split}.npz}"
python3 -m eval.satdifuser select --features "$FEATS" --datasets "$DATASET" --out-dir "results/satdifuser_select/$DATASET"
echo "SDF_SELECT_OK $DATASET"
