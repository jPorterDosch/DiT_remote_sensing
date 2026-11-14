#!/bin/bash
# ============================================================================
# Object-level representation diagnostic (eval/object_diag.py) on the cached segmentation tokens.
# CPU only, no training. Arms are cached per (dataset, arm) under results/object_diag/arms and
# skipped when present: a timed-out job is simply resubmitted.
#
#   mkdir -p logs
#   MODE=test sbatch --time=0:30:00 experiments/isaac/eval/object_diag.sbatch       # pytest only
#   MODE=run DATASET=m_pv4ger_seg MAX_IMAGES=48 sbatch --time=0:45:00 experiments/isaac/eval/object_diag.sbatch
#                                                                 # smoke -> <out>/smoke, + its report and figure
#   MODE=run DATASET=m_pv4ger_seg sbatch experiments/isaac/eval/object_diag.sbatch  # + the figure
#   MODE=report sbatch --time=0:30:00 experiments/isaac/eval/object_diag.sbatch     # SMOKE=1: the smoke arms
#
# Variables travel through the submitting shell's environment (sbatch exports it by default).
# GROUPS_: comma list of teacher,dino,t100,t260,t420,ctrl (default: all). REDO=1 passes --redo.
# Memory: the teacher's token Grams stay in RAM (n x 1024 x 1024 fp16 = 2 GB per 1000 images).
# ============================================================================
#SBATCH --job-name=object-diag
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
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
python3 -c "import sys, numpy, torch, PIL, matplotlib; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

MODE="${MODE:-run}"
OUT="${OUT:-results/object_diag}"
SPEC="${SPEC:-eval/seg_specs/phase2_arms.json}"
echo "object-diag MODE=$MODE DATASET=${DATASET:-} MAX_IMAGES=${MAX_IMAGES:-} GROUPS_=${GROUPS_:-all} spec=$SPEC out=$OUT host=$(hostname) cpus=$SLURM_CPUS_PER_TASK $(date -Is)"
log="$STORE/tmp/object-diag-$SLURM_JOB_ID.log"

step() {  # step <marker> <subcommand> [args...]: run, require the marker line
    local marker="$1"; shift
    python3 -m eval.object_diag "$@" 2>&1 | tee "$log"
    local summary
    summary=$(grep "$marker" "$log" | tail -1 || true)
    rm -f "$log"
    [ -n "$summary" ] || { echo "FATAL: no $marker line" >&2; exit 1; }
    echo "done $(date -Is): $summary"
}

common=(--spec "$SPEC" --out-dir "$OUT")
case "$MODE" in
    test)
        python3 -m pytest tests/test_object_diag.py -q -p no:cacheprovider
        echo "OBJECT_DIAG_TESTS_OK" ;;
    run)
        [ -n "${DATASET:-}" ] || { echo "FATAL: MODE=run needs DATASET" >&2; exit 2; }
        args=("${common[@]}" --dataset "$DATASET")
        [ -n "${MAX_IMAGES:-}" ] && args+=(--max-images "$MAX_IMAGES")
        [ -n "${GROUPS_:-}" ] && args+=(--groups "$GROUPS_")
        [ "${REDO:-0}" = 1 ] && args+=(--redo)
        step OBJECT_DIAG_RUN_OK run "${args[@]}"
        if [ -z "${GROUPS_:-}" ]; then  # a full set of arms: the figure, and for a smoke its report
            smoke=()
            [ -n "${MAX_IMAGES:-}" ] && smoke=(--smoke)
            step OBJECT_DIAG_FIGURE_OK figure "${common[@]}" --dataset "$DATASET" ${smoke[@]+"${smoke[@]}"}
            if [ -n "${MAX_IMAGES:-}" ]; then
                step OBJECT_DIAG_REPORT_OK report "${common[@]}" --datasets "$DATASET" --smoke
            fi
        fi ;;
    report)
        args=("${common[@]}")
        [ -n "${DATASETS:-}" ] && args+=(--datasets $DATASETS)
        [ "${SMOKE:-0}" = 1 ] && args+=(--smoke)
        step OBJECT_DIAG_REPORT_OK report "${args[@]}" ;;
    *) echo "FATAL: MODE=$MODE: test | run | report" >&2; exit 2 ;;
esac
echo "OBJECT_DIAG_JOB_OK $MODE $(date -Is)"
