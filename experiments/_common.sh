#!/bin/bash
# ============================================================================
# experiments/_common.sh — Shared functions for experiment scripts
# ============================================================================
# Sourced by every experiment script after the inline PROJECT_ROOT bootstrap.
#
# Functions:
#   print_delim             — visual section delimiter
#   setup_environment                   — defaults, SLURM env, SAVE_DIR, SCRIPT_NAME
#   expand_params_for_parallel — params array → parallel args + sweep count
#   countdown               — 5-second pre-launch countdown
#
# All functions communicate via global variables. Each function's header
# documents what globals it reads and sets.
#
# Naming convention:
#   Config variables:  SAVE_DIR, SCRIPT_NAME, PARALLEL_JOBS, PROJECT_NAME
#   Sweep machinery:   SWEEP_PLACEHOLDERS, SWEEP_VALUES,
#                      SWEEP_SEED_PLACEHOLDER, SWEEP_TOTAL_COMBINATIONS
#
# Usage (in each experiment script):
#
#   source "$PROJECT_ROOT/experiments/_common.sh" || {
#       echo "FATAL: Failed to source _common.sh" >&2; exit 1;
#   }
#   setup_environment
#   ...
#   expand_params_for_parallel
#   ...
#   countdown
# ============================================================================

# ----------------------------------------------------------------------------
# print_delim [label]
# Print a visual section delimiter. Optional label printed on next line.
# ----------------------------------------------------------------------------
print_delim() {
	echo && echo "################################################"
	if [ -n "$1" ]; then
		echo "@@ $1"
	fi
}

# ----------------------------------------------------------------------------
# setup_environment
# Reads:  PROJECT_ROOT, SLURM_JOB_ID, SLURM_JOB_NAME, SCRATCHDIR
# Sets:   SAVE_DIR, PARALLEL_JOBS, SCRIPT_NAME, PROJECT_NAME
#
# Initializes defaults from the calling script's filename, then applies
# SLURM overrides if running under SLURM (modules, SCRATCHDIR, job name).
# Safe to call outside SLURM — SLURM-specific block is skipped.
# Also runs slurm-env-check.sh if available (diagnostic helper).
# ----------------------------------------------------------------------------
setup_environment() {
	# --- Defaults ---
	SAVE_DIR="$PROJECT_ROOT/models"
	PARALLEL_JOBS=1

	SCRIPT_NAME=$(basename "${BASH_SOURCE[1]}")
	SCRIPT_NAME="${SCRIPT_NAME%.*}" # remove extension
	SAVE_DIR="$SAVE_DIR/$SCRIPT_NAME"

	# --- Modules (if available) ---
	if command -v module &>/dev/null; then
		module load cuda/12.2.0-binary
		module load bzip2/1.0.8
	fi

	# --- SLURM overrides ---
	if [ -n "$SLURM_JOB_ID" ]; then
		if [ -z "$SCRATCHDIR" ]; then
			echo "FATAL: SCRATCHDIR not set" >&2
			exit 1
		fi

		PROJECT_NAME=$(basename "$PROJECT_ROOT")
		SCRIPT_NAME="${SLURM_JOB_NAME%.*}"
		SAVE_DIR="$SCRATCHDIR/$PROJECT_NAME/$SLURM_JOB_NAME"
	fi

	# Environment diagnostic (prints system, GPU, Python, SLURM info).
	if [ -f "$PROJECT_ROOT/print-environment-info.sh" ]; then
		bash "$PROJECT_ROOT/print-environment-info.sh"
	fi
}

# ----------------------------------------------------------------------------
# expand_params_for_parallel
# Reads:  params (associative array declared by caller)
# Sets:   SWEEP_PLACEHOLDERS, SWEEP_VALUES, SWEEP_SEED_PLACEHOLDER,
#         SWEEP_TOTAL_COMBINATIONS
#
# Converts the params array into GNU parallel's argument format,
# counts the total sweep runs, and prints the summary:
#   SWEEP_PLACEHOLDERS      — e.g., "--lr {1} --seed {2} "
#   SWEEP_VALUES            — e.g., "::: 0.01 0.005 ::: 42 43 44 "
#   SWEEP_SEED_PLACEHOLDER  — placeholder number for [seed] (empty if
#       no seed key). Needed by finetuning scripts to construct per-seed
#       checkpoint paths in the parallel command.
#   SWEEP_TOTAL_COMBINATIONS — total number of runs (product of all
#       value counts across all keys).
#
# Iteration order is arbitrary (bash associative arrays have no guaranteed
# order), but this doesn't affect correctness — parallel maps placeholders
# to values by position.
# ----------------------------------------------------------------------------
expand_params_for_parallel() {
	SWEEP_PLACEHOLDERS=""
	SWEEP_VALUES=""
	SWEEP_SEED_PLACEHOLDER=""
	SWEEP_TOTAL_COMBINATIONS=1

	local placeholder_num=1
	for key in "${!params[@]}"; do
		echo "Parameter: $key"
		local p="--$key {$placeholder_num} "
		local v="::: ${params[$key]} "
		echo "   $p"
		echo "   $v"

		SWEEP_PLACEHOLDERS+=$p
		SWEEP_VALUES+=$v

		# Count values for this key (contributes to Cartesian product)
		local num_values
		num_values=$(echo "${params[$key]}" | wc -w)
		SWEEP_TOTAL_COMBINATIONS=$((SWEEP_TOTAL_COMBINATIONS * num_values))

		if [ "$key" = "seed" ]; then
			SWEEP_SEED_PLACEHOLDER=$placeholder_num
		fi

		((placeholder_num++))
	done
}

# ----------------------------------------------------------------------------
# print_summary
# Reads:  SAVE_DIR, PARALLEL_JOBS, SWEEP_TOTAL_COMBINATIONS
# Sets:   (none)
#
# Prints the experiment configuration before launch. Call after
# expand_params_for_parallel (needs SWEEP_TOTAL_COMBINATIONS).
# ----------------------------------------------------------------------------
print_summary() {
	print_delim
	echo "SAVE_DIR: $SAVE_DIR"
	echo "PARALLEL_JOBS: $PARALLEL_JOBS"
	echo "COMBINATIONS: $SWEEP_TOTAL_COMBINATIONS"
}

# ----------------------------------------------------------------------------
# countdown
# Reads:  (none)
# Sets:   (none)
#
# 5-second countdown before launching experiments.
# Gives the user time to review the printed summary and Ctrl-C if wrong.
# ----------------------------------------------------------------------------
countdown() {
	print_delim
	for i in $(seq 5 -1 1); do
		echo "Starting in $i..."
		sleep 1
	done
}