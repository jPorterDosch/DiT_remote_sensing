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
#SBATCH --partition=campus-gpu
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.out

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

source .secrets/wandb-personal.env

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
    # NOTE: values here are SWEPT (Cartesian product), so the multi-timestep grid CANNOT go
    # in this array -- "100 180 ..." would launch 7 single-t runs. It is a fixed flag below.
    [k]="28"
    # PINNED values are passed EXPLICITLY here, never via run.py defaults: these three fields
    # are config_hash inputs for every task, so moving their defaults would repoint every
    # extraction run directory (2026-09-10 review, finding 1).
    [mask-ratio]="0.0"
    [finetune-max-epochs]="10"
    [finetune-lr]="1e-3"
    [max-train-steps]="5000"
    [gradient-accumulation-steps]="4"
    [lora-rank]="4"
    [lora-alpha]="16.0"
    [lora-dropout]="0.0"
    [guidance-scale]="1.0"
    [batch-size]="1"
    [num-workers]="4"
    [seed]="42"
    [mim-loss-weight]="0.0"
    # ens=1 matches every frozen cache; run.py's default is 8, which would make the
    # post-training probe 8x more forwards AND protocol-incomparable (finding 3).
    [model.ensemble-size]="1"
    # blocker 4: train on the complement of the n=5000 probe subset, keyed on the cache's
    # STORED subset_indices. Change alongside dataset.name if sweeping datasets.
    [exclude-probe-indices]="models/n5000_eurosat_oneshot_ens1/eurosat_flux_0b91191d+42/multistep_train_feats_oneshot_g1.0.npz"
    # 500 -> ~10 validations over the run. The run.py default of 50 would spend ~100
    # validations x ~900 forwards each -- more GPU than training itself (2026-09-11 review).
    [log-val-steps]="500"
)

expand_params_for_parallel
print_summary
countdown

# Workstation weight paths (harmless no-op on ISAAC, where the run.py defaults point at
# /lustre): only exported when the local files exist. Without this, a workstation launch
# dies at model load looking for the ISAAC lustre path (see src/models/flux/util.py).
if [ -f "ditf_models/FLUX.1-dev/flux1-dev.safetensors" ]; then
    export FLUX_DEV="ditf_models/FLUX.1-dev/flux1-dev.safetensors"
    export AE="ditf_models/FLUX.1-dev/ae.safetensors"
    echo "Using local FLUX weights: $FLUX_DEV"
fi

# ============================================================================
# RUN
# ============================================================================
# Notes on fixed flags (can't go in the params array):
#   --use-gradient-accumulation   bool flag, no value
#   --wrap-output                 bool flag, no value
#   --label-fraction 1.0          full-label probe after fine-tuning
#   --cd                          discard massive-activation channels in the probe --
#                                 matches EVERY offline probe and eval sweep (finding 2);
#                                 without it the per-t table sits at a different feature
#                                 operating point than all frozen numbers
#   --t $T_GRID                   multi-valued: training samples t per example over this grid
#                                 (params-array values would be swept, not passed together)
#
# PINNED (train_diffusion.py enforces both, and run.py's defaults now match):
#   guidance-scale 1.0    every feature cache is g=1.0; the field feeds train AND extraction
#   mim-loss-weight 0.0   dropped angle; its decoder reads the probed block features
#   mask-ratio 0.0        masking served MIM only; with MIM off it just hides the flow target
# ============================================================================
# The probe grid, matching extract_n5000_dit.sh -- every timestep the probes read is trained.
T_GRID="100 180 260 340 420 500 580"
print_delim "## START"
set -x
parallel -j $PARALLEL_JOBS --delay 15 --verbose \
    "$PYTHON" "$PROJECT_ROOT/run.py" \
        --use-gradient-accumulation \
        --wrap-output \
        --cd \
        --label-fraction 1.0 \
        --t $T_GRID \
        $SWEEP_PLACEHOLDERS \
    $SWEEP_VALUES
set +x
print_delim "## DONE"
