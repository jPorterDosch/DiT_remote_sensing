#!/bin/bash
#SBATCH -A acf-utk0011
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --time=1-00:00:00               # Wall time (days-hh:mm:ss)
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.out

# --- Bootstrap: find project root (can't be sourced) ------
_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"

source "$PROJECT_ROOT/experiments/_common.sh" || {
	echo "FATAL: Failed to source common.sh" >&2; exit 1;
}
setup_environment

# ============================================================================
# SWEEP GRID
# ============================================================================
# Each key becomes a CLI arg (--key value). Spaces delimit sweep values.
# GNU parallel computes the Cartesian product of all keys' values.
#
#   [lr]="0.01 0.005"    — 2 sweep values: one run per value
#   [model]="FCOS"       — 1 value: sets the param, no extra dimension
#   [lr] × [seed]        — 2 × 3 = 6 total runs (all combinations)
# ----------------------------------------------------------------------------
# Why some params are here (array) vs below (parallel command):
#   Array   — set once, shared across ALL parallel blocks (DRY).
#             Use for: identity, data, sweep candidates.
#   Command — repeated per block, CAN vary between blocks.
#             Use for: list-valued args and params that group
#             with them (lr_scheduler + lr_milestones = a unit).
# ============================================================================
declare -A params
params=(
	[exp_name]="$SCRIPT_NAME"
    [dataset]="spair"
    [dataset_path]="/lustre/isaac24/scratch/jdosch1/DeepLearning/datasets/SPair-71k"
    [save_path]="$SAVE_DIR"
    [dit_model]="flux"
    [t]="260"
    [k]="28"
    [ensemble_size]="8"
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
# This is where list-valued args belong (--lr_milestones, --bb.stage.blocks)
# since they can't go in the params array above.
#
# $SWEEP_PLACEHOLDERS and $SWEEP_VALUES MUST be last — they carry the sweep grid.
#
# Multiple blocks: copy-paste the `parallel ... $SWEEP_VALUES` command and
# change the fixed args. Each copy sweeps the same params grid.
# Order blocks by value: baseline first, highest-signal next,
# riskiest early, minor variations last.
# ----------------------------------------------------------------------------
# Flags:
#   -j N         max concurrent jobs (1 = sequential)
#   --delay 15   stagger launches (prevents GPU contention)
#   --shuf       randomize run order (avoids param-ordering bias)
#   --verbose    print each command before execution
# Not currently used but available:
#   --resume           skip already-completed runs (for restarts)
#   --halt now,fail=1  stop everything on first failure
# ============================================================================
print_delim "## START"

# set -x enables printing commands before execution
# it is useful for debugging to see the full parallel command that we constructed below
set -x

# Baseline
parallel -j $PARALLEL_JOBS --delay 15 --shuf --verbose \
	python3 "$PROJECT_ROOT/eval_spair.py" \
        --cd \
        --img_size 640 640 \
        $SWEEP_PLACEHOLDERS \
    $SWEEP_VALUES

set +x

print_delim "## DONE"
