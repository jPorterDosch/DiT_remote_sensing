#!/bin/bash
# ============================================================================
# GATE 1 (instrument check, CLAUDE.md rules 2 / 16): SatDiFuser's OWN code, unmodified, on
# m-eurosat with their published config (configs/eurosat_exp.yaml + tasks/meurosat.yaml: gw fuser,
# 40 epochs, lr 0.01, batch 32, eval every 100 steps, best-on-val, one test eval) must reproduce
# their Table 2 m-eurosat number (gw 97.7 top-1) within ~0.5 before any feature of theirs is
# dumped into our matrix. Their loader reads the RAW GEO-Bench m-eurosat under $GEO_BENCH_DIR
# (jobs/geobench_eurosat.sbatch), 16,200 train rows (the shipped default partition, rule 16).
#
#   EPOCHS   max_train_epochs override (SMOKE: 1 -> measures the epoch time first; full: unset = 40)
#   FUSER    gw (default, Table 2's main row) | lw | moe
#
#   sbatch --export=ALL,EPOCHS=1 experiments/isaac/satdifuser/gate_eurosat.sbatch   # smoke / timing
#   sbatch experiments/isaac/satdifuser/gate_eurosat.sbatch                         # the gate
# Output: $SDF/experiments/<timestamp>-meurosat-<fuser>-<tag>/log.txt ("Final test metric: 0.9xxx").
#
# The verdict is executed code (gate_check.py; review 2026-11-12, item 10 -- this script used to
# print SDF_GATE_OK whatever run.py measured). A full run (EPOCHS unset) ends with
#   GATE satdifuser-eurosat PASS|FAIL ...      and writes $SDF/gate/eurosat_<fuser>_<jobid>.json
# PASS = exactly one "Final test metric" within 0.5 points of the published number, either side;
# only then SDF_GATE_OK is printed and the job exits 0. dump.sbatch refuses to run without a PASS
# verdict. EPOCHS set = a timing smoke: no verdict, no SDF_GATE_OK, prints SDF_GATE_SMOKE_DONE.
# ============================================================================
#SBATCH --job-name=sdf-gate
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
source "$PROJECT_ROOT/experiments/isaac/satdifuser/env.sh" || exit 1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE=disabled
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -f "$GEO_BENCH_DIR/classification_v1.0/m-eurosat/default_partition.json" ] || { echo "FATAL: raw m-eurosat missing under $GEO_BENCH_DIR (jobs/geobench_eurosat.sbatch)" >&2; exit 1; }
cd "$SDF_REPO"  # their load_config resolves configs/ relative to the cwd
git rev-parse HEAD; git status --short | head -5
ARGS=(--config configs/eurosat_exp.yaml --pretrained_model_name_or_path "$SDF_CKPT" --report_to tensorboard
      --exp_folder "$SDF/experiments" --fuser "${FUSER:-gw}" --num_workers "${SLURM_CPUS_PER_TASK:-8}")
[ -n "${EPOCHS:-}" ] && ARGS+=(--max_train_epochs "$EPOCHS")
echo "args: ${ARGS[*]}"
mkdir -p "$PROJECT_ROOT/logs"
RUNLOG="$PROJECT_ROOT/logs/sdf-gate-${SLURM_JOB_ID:?run as a SLURM job}.run.log"
[ ! -e "$RUNLOG" ] || { echo "FATAL: $RUNLOG exists" >&2; exit 1; }
python3 run.py "${ARGS[@]}" 2>&1 | tee "$RUNLOG"  # pipefail: a failed run.py stops here
if [ -n "${EPOCHS:-}" ]; then
    grep -a "Final test metric" "$RUNLOG" || echo "(no final test metric in a ${EPOCHS}-epoch smoke)"
    echo "SDF_GATE_SMOKE_DONE epochs=$EPOCHS (timing / env check only: NOT the gate, no verdict written)"
    exit 0
fi
python3 "$PROJECT_ROOT/experiments/isaac/satdifuser/gate_check.py" verdict --log "$RUNLOG" --fuser "${FUSER:-gw}" \
    --out "$SDF/gate/eurosat_${FUSER:-gw}_${SLURM_JOB_ID}.json"  # exit 1 on FAIL (set -e)
echo "SDF_GATE_OK"
