#!/bin/bash
# ============================================================================
# Teacher-complementarity diagnostics (eval/complementarity.py), CPU only. Arms are cached per
# (dataset, arm) under results/complementarity/arms and skipped when present: a timed-out job is
# simply resubmitted.
#
#   mkdir -p logs
#   MODE=agree sbatch experiments/isaac/eval/complementarity.sbatch                         # minutes, no fits
#   MODE=run DATASET=m_eurosat SMOKE_TRAIN=600 C_GRID=0.001,0.1 sbatch experiments/isaac/eval/complementarity.sbatch
#                                                                                           # smoke -> <out>/smoke
#   MODE=run DATASET=m_so2sat sbatch experiments/isaac/eval/complementarity.sbatch
#   MODE=run DATASET=m_bigearthnet ARMS=T+FA sbatch -p campus -q campus -t 1-00:00:00 \
#          --cpus-per-task=32 --mem=128G experiments/isaac/eval/complementarity.sbatch      # 43 one-vs-rest fits per C
#   MODE=report sbatch experiments/isaac/eval/complementarity.sbatch                        # SMOKE=1: the smoke arms
#
# Variables travel through the submitting shell's environment (sbatch exports it by default); do
# not put a comma list (ARMS, C_GRID) inside --export=, where a comma separates variables.
# ARMS: comma list (default: every arm).
# REDO=1 passes --redo. Thread policy is the matrix's (eval.matrix.blas_threads; OVR_N_JOBS workers
# for the multi-label one-vs-rest fits, memory-budgeted): --cpus-per-task only changes wall time.
# ============================================================================
#SBATCH --job-name=complementarity
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # models/ -> scratch caches

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
export WANDB_MODE=disabled
export OVR_N_JOBS="${OVR_N_JOBS:-$SLURM_CPUS_PER_TASK}"  # multi-label one-vs-rest workers; results identical
export JOBLIB_TEMP_FOLDER="$STORE/tmp/joblib"  # worker memmaps of wide X (GBs): Lustre, not node /tmp
mkdir -p "$JOBLIB_TEMP_FOLDER"
python3 -c "import sys, numpy, sklearn, threadpoolctl; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

MODE="${MODE:-run}"
OUT="${OUT:-results/complementarity}"
echo "complementarity MODE=$MODE DATASET=${DATASET:-} ARMS=${ARMS:-all} SMOKE_TRAIN=${SMOKE_TRAIN:-} C_GRID=${C_GRID:-} out=$OUT host=$(hostname) cpus=$SLURM_CPUS_PER_TASK $(date -Is)"
args=(--out-dir "$OUT")
case "$MODE" in
    agree)
        [ -n "${DATASETS:-}" ] && args+=(--datasets $DATASETS)
        marker=AGREE_OK ;;
    run)
        [ -n "${DATASET:-}" ] || { echo "FATAL: MODE=run needs DATASET" >&2; exit 2; }
        args+=(--dataset "$DATASET")
        [ -n "${ARMS:-}" ] && args+=(--arms "$ARMS")
        [ -n "${SMOKE_TRAIN:-}" ] && args+=(--smoke-train "$SMOKE_TRAIN")
        [ -n "${C_GRID:-}" ] && args+=(--c-grid "$C_GRID")
        [ "${REDO:-0}" = 1 ] && args+=(--redo)
        marker=COMPLEMENTARITY_RUN_OK ;;
    report)
        [ -n "${DATASETS:-}" ] && args+=(--datasets $DATASETS)
        [ "${SMOKE:-0}" = 1 ] && args+=(--smoke)
        marker=COMPLEMENTARITY_REPORT_OK ;;
    *) echo "FATAL: MODE=$MODE: agree | run | report" >&2; exit 2 ;;
esac
log="$STORE/tmp/complementarity-$SLURM_JOB_ID.log"
python3 -m eval.complementarity "$MODE" "${args[@]}" 2>&1 | tee "$log"
summary=$(grep "$marker" "$log" | tail -1 || true)
rm -f "$log"
[ -n "$summary" ] || { echo "FATAL: no $marker line" >&2; exit 1; }
echo "done $(date -Is): $summary"
