#!/bin/bash
#SBATCH -A acf-utk0011
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu
#SBATCH --time=1-00:00:00               # Wall time (days-hh:mm:ss)
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.out
# --- Bootstrap: find project root (can't be sourced) ------
_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
source "$PROJECT_ROOT/experiments/_common.sh" || {
	echo "FATAL: Failed to source _common.sh" >&2; exit 1;
}
setup_environment
# ============================================================================
# SWEEP GRID
# ============================================================================
# Each key becomes a CLI arg (--key value). Spaces delimit sweep values.
# GNU parallel computes the Cartesian product of all keys' values.
#
#   [lr]="0.01 0.005"    — 2 sweep values: one run per value
#   [model.name]="flux"  — 1 value: sets the param, no extra dimension
#   [lr] × [seed]        — 2 × 3 = 6 total runs (all combinations)
# ----------------------------------------------------------------------------
# Why some params are here (array) vs below (parallel command):
#   Array   — set once, shared across ALL parallel blocks (DRY).
#             Use for: identity, data, sweep candidates.
#   Command — repeated per block, CAN vary between blocks.
#             Use for: list-valued args and params that group
#             with them (lr_scheduler + lr_milestones = a unit).
# ============================================================================
# Defaults below run EuroSAT classification. To switch tasks:
#
#   Correspondence (SPair-71k):
#     [task]="correspondence"
#     [dataset.name]="spair"
#     [dataset.path]="/path/to/SPair-71k"
#     [img-size]="640 640"
#     -- remove clf-* and seed, drop --label-fractions from parallel command --
#
#   Segmentation (DAVIS 2017):
#     [task]="segmentation"
#     [dataset.name]="davis"
#     [dataset.path]="/path/to/DAVIS"
#     [img-size]="480"
#     [model.ensemble-size]="1"
#     -- remove clf-* and seed, drop --label-fractions from parallel command --
# ============================================================================
# To eval a fine-tuned model, add to params:
#   [lora-checkpoint]="/lustre/isaac24/scratch/aabdelr5/diffusion-mim-ssl-fp_dl_s26/train_diffusion.sh/<run>/checkpoints/lora_best_stepN.pt"
#   [lora-rank]="4"
#   [lora-alpha]="16.0"
#   [wrap-output]="true"
declare -A params
params=(
    [task]="classification"
    [dataset.name]="eurosat"
    [dataset.path]="/lustre/isaac24/scratch/jdosch1/DeepLearning/datasets/EuroSAT"
    [model.name]="flux"
    [model.ensemble-size]="8"
    [save-dir]="$SAVE_DIR"
    [img-size]="224"
    [t]="260"
    [k]="28"
    [clf-epochs]="50"
    [clf-lr]="1e-3"
    [clf-batch-size]="256"
    [seed]="42"
    [lora-checkpoint]="/lustre/isaac24/scratch/aabdelr5/diffusion-mim-ssl-fp_dl_s26/train_diffusion.sh/eurosat_flux_5394c2cb+42/checkpoints/lora_best_step3400.pt"
    [lora-rank]="4"
    [lora-alpha]="16.0"
    [wrap-output]="true"
)
# ============================================================================
# BUILD AND SUMMARIZE
# ============================================================================
expand_params_for_parallel
print_summary
countdown
# ============================================================================
# PARALLEL COMMAND
# ============================================================================
# Fixed args (constant across all runs) go before $SWEEP_PLACEHOLDERS/$SWEEP_VALUES.
# This is where list-valued args belong (--cd, --label-fractions) since they
# can't go in the params array above:
#   --cd               boolean flag, no value, cannot be a params entry
#   --label-fractions  takes multiple space-separated values
#
# $SWEEP_PLACEHOLDERS and $SWEEP_VALUES MUST be last — they carry the sweep grid.
# ----------------------------------------------------------------------------
# Flags:
#   -j N         max concurrent jobs (1 = sequential)
#   --delay 15   stagger launches (prevents GPU contention)
#   --shuf       randomize run order (avoids param-ordering bias)
#   --verbose    print each command before execution
# ============================================================================
print_delim "## START"
set -x
# Baseline
parallel -j $PARALLEL_JOBS --delay 15 --shuf --verbose \
	python3 "$PROJECT_ROOT/eval.py" \
        --cd \
        --label-fractions 1 5 10 50 100 \
        $SWEEP_PLACEHOLDERS \
    $SWEEP_VALUES
set +x
print_delim "## DONE"
