#!/bin/bash
# Full MIM fine-tuning of Flux-dev on EuroSAT.
# Usage:  sbatch experiments/train_diffusion.sh
#
# Timing estimate (A100, bfloat16, batch=1, grad_accum=4):
#   ~17 s per batch forward+backward
#   max_train_steps=7000    20000 batches    ~94 h compute + ~15 min load
#   Request 1 day (checkpoints saved on best val loss so progress is not lost if preempted).
#
#SBATCH -A acf-utk0011
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

# --- Bootstrap ---
_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"

source "$PROJECT_ROOT/experiments/_common.sh" || {
    echo "FATAL: failed to source _common.sh" >&2; exit 1;
}
setup_environment

# --- Conda ---
# eval+hook is the canonical approach for non-interactive (batch) shells;
# it initialises the conda shell functions before activate is called.
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)"
else
    for _conda_sh in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [ -f "$_conda_sh" ]; then source "$_conda_sh"; break; fi
    done
fi
conda activate DiTF

# Capture the resolved python so parallel inherits the right interpreter
# regardless of how it spawns subshells.
PYTHON=$(which python3)
echo "Python: $PYTHON  ($(${PYTHON} --version 2>&1))"
${PYTHON} -c "import tyro" || { echo "FATAL: tyro not found in $PYTHON — check conda env" >&2; exit 1; }

# Create log subdirectory so SLURM doesn't fail on missing path.
mkdir -p "$PROJECT_ROOT/logs/${SLURM_JOB_NAME:-train_diffusion}"

# ============================================================================
# SWEEP GRID (single run — extend to sweep by adding space-separated values)
# ============================================================================
declare -A params
params=(
    [task]="finetune-diffusion"
    [dataset.name]="eurosat"
    [dataset.path]="/lustre/isaac24/scratch/jdosch1/DeepLearning/datasets/EuroSAT"
    [model.name]="flux"
    [save-dir]="$SAVE_DIR"
    [img-size]="224"
    [t]="260"
    [k]="28"
    [mask-ratio]="0.75"
    [finetune-max-epochs]="10"
    [finetune-lr]="1e-3"
    [max-train-steps]="5000"
    [gradient-accumulation-steps]="4"
    [lora-rank]="4"
    [lora-alpha]="16.0"
    [lora-dropout]="0.0"
    [guidance-scale]="3.5"
    [batch-size]="1"
    [num-workers]="4"
    [seed]="42"
)

expand_params_for_parallel
print_summary
countdown

# ============================================================================
# RUN
# ============================================================================
# Notes on fixed flags (can't go in the params array):
#   --use-gradient-accumulation   bool flag, no value
#   --wrap-output                 bool flag, no value
#   --label-fractions 100         list-valued; must be a single entry to satisfy
#                                 FinetuneDiffusionTask's len==1 guard
# ============================================================================
print_delim "## START"
set -x
parallel -j $PARALLEL_JOBS --delay 15 --verbose \
    "$PYTHON" "$PROJECT_ROOT/run.py" \
        --use-gradient-accumulation \
        --wrap-output \
        --label-fractions 100 \
        $SWEEP_PLACEHOLDERS \
    $SWEEP_VALUES
set +x
print_delim "## DONE"
