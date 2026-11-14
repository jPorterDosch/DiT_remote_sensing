#!/bin/bash
# ============================================================================
# Label-efficiency matrix (eval/matrix.py) on the long CPU partition: one array task per
# (model x dataset) of the spec, cells cached under results/matrix/cells (home), resumable.
#
#   mkdir -p logs
#   sbatch --export=ALL,PART=full --array=0-17 experiments/isaac/eval/matrix.sbatch        # 100 % cells, long (days deep)
#   sbatch --export=ALL,PART=low -p short -q short -t 3:00:00 --array=0-17 experiments/isaac/eval/matrix.sbatch
#                                                       # low-label cells, short: starts now; resubmit if timed out
#   sbatch --export=ALL,PART=low -p short -q short -t 3:00:00 --cpus-per-task=16 --array=5,11,17 experiments/isaac/eval/matrix.sbatch
#   sbatch --export=ALL,PART=full --cpus-per-task=16 --mem=128G --array=5,11,17 experiments/isaac/eval/matrix.sbatch
#                                                       # m-bigearthnet (43 one-vs-rest fits per LR): OVR_N_JOBS=cpus workers
#   for s in 0 1 2; do sbatch --export=ALL,PART=low,SEEDS=$s -p short -q short -t 3:00:00 --cpus-per-task=16 --array=17 experiments/isaac/eval/matrix.sbatch; done
#                                                       # FLUX x bigearthnet low cells ~40 min each: one short job per seed
#   sbatch --array=0-17 experiments/isaac/eval/matrix.sbatch                             # PART=all: every cell, long
#   SPEC=eval/matrix_specs/other.json sbatch --array=0-5 experiments/isaac/eval/matrix.sbatch
#   python -m eval.matrix report --spec eval/matrix_specs/phase1.json                   # login node, after all tasks
#
# PART=all (default) | full (the seed-free 100 % cell only) | low (every fraction < 1, all seeds).
# REDO=1 passes --redo (recompute cells whose inputs / protocol changed; move the old cells to
# results/matrix/stale/ first so their per-image vectors survive). The spec is read when the task
# STARTS, not at submission: a cell is refused if its stored config differs from the spec.
# Thread policy: BLAS threads = min(8, rows // 256) regardless of --cpus-per-task
# (eval.matrix.blas_threads); every one-vs-rest label fit (m-bigearthnet) runs at
# eval.protocols.OVR_FIT_THREADS = 1 thread (2026-10-15; before: cpus // workers, which followed
# --cpus-per-task / --mem); --cpus-per-task only sets the worker count, i.e. wall time. The 13
# phase-1 flux-pretrained x m-bigearthnet cells ran at 2 threads and are KEPT (user decision
# 2026-10-15, effect ~1e-3): ACCEPT_THREAD_SHARE=1 passes --accept-thread-share (run and report).
# Task i -> model i // n_datasets, dataset i % n_datasets (spec order). A task whose features
# are absent exits 3 (MISSING; the report lists the cells as missing). A timed-out task is
# resubmitted; finished cells are skipped. Sizing (sacct, 2026-10): the 100 % DINO cell is the
# official wide-grid probe, 2.6-4.9 h on 18 CPUs (geobench-dino-probe-wide); the FLUX 100 % cell
# (concat candidate, up to 27k dims x 20k rows x 7 C) is one block of geobench-probe-mb, 16-17 h
# for 14 blocks on 18 CPUs. The 12 low-label cells are minutes each (BLAS threads capped by
# rows in eval.matrix.blas_threads) EXCEPT m-bigearthnet: serial one-vs-rest (OVR_N_JOBS unset,
# 2026-10-13) took 1.5-3 h per 25 % cell and three rounds on short timed out; with OVR_N_JOBS=
# cpus the 43 label fits run in parallel (eval.protocols.budget_ovr_jobs caps the pool by
# --mem: ~1.3x the largest train matrix per worker). long: 6-day wall, 12 running / 96 CPUs per user.
# ============================================================================
#SBATCH --job-name=eval-matrix
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=long
#SBATCH --partition=long
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # models/ -> scratch caches

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
export WANDB_MODE=disabled
export OVR_N_JOBS="${OVR_N_JOBS:-$SLURM_CPUS_PER_TASK}"  # multi-label one-vs-rest workers (memory-budgeted in eval.protocols); results identical
export JOBLIB_TEMP_FOLDER="$STORE/tmp/joblib"  # worker memmaps of wide X (GBs): Lustre, not node /tmp
mkdir -p "$JOBLIB_TEMP_FOLDER"
find "$JOBLIB_TEMP_FOLDER" -maxdepth 1 -name 'ovr_X_*.npy' -mtime +7 -delete 2>/dev/null || true  # SIGKILL/OOM orphans
python3 -c "import sys, numpy, sklearn, matplotlib; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

SPEC="${SPEC:-eval/matrix_specs/phase1.json}"
OUT="${OUT:-results/matrix}"
PART="${PART:-all}"
read -r MODEL DATASET EXTRA < <(python3 - "$SPEC" "$SLURM_ARRAY_TASK_ID" "$PART" <<'EOF'
import json, os, sys
spec = json.load(open(sys.argv[1])); i = int(sys.argv[2]); part = sys.argv[3]
models, datasets = [m["name"] for m in spec["models"]], spec["datasets"]
if i >= len(models) * len(datasets):
    raise SystemExit(f"task {i} beyond {len(models)} x {len(datasets)}")
fr, seeds = [float(f) for f in spec["fractions"]], [int(s) for s in spec["seeds"]]
if part == "all":
    extra = ""
elif part == "full":
    extra = f"--fractions 1.0 --seeds {seeds[0]}"
elif part == "low":
    extra = "--fractions " + " ".join(f"{f:g}" for f in fr if f != 1.0)
else:
    raise SystemExit(f"PART={part!r}: all | full | low")
if os.environ.get("SEEDS"):  # SEEDS="0 1": split a slow (model x dataset) task by seed, same cells
    if part == "full":
        raise SystemExit("SEEDS with PART=full would write the seed-free 100 % cell under another seed (argparse keeps the last --seeds)")
    extra += " --seeds " + " ".join(str(int(x)) for x in os.environ["SEEDS"].split())
if os.environ.get("REDO") == "1":  # recompute cells whose inputs or protocol/config changed (else such a cell is fatal)
    extra += " --redo"
if os.environ.get("ACCEPT_THREAD_SHARE") == "1":  # keep the 13 pre-2026-10-15 bigearthnet FLUX cells at 2 threads
    extra += " --accept-thread-share"
print(models[i // len(datasets)], datasets[i % len(datasets)], extra)
EOF
)
echo "task $SLURM_ARRAY_TASK_ID: model=$MODEL dataset=$DATASET part=$PART extra='$EXTRA' spec=$SPEC out=$OUT host=$(hostname) $(date -Is)"
# A row that the spec lists under "requires_gate" is scored only after its control gate passed on
# the feature files that are on disk NOW (dino_probe.sbatch writes the marker; a row the spec does
# not list passes through). Exit 1 = not gated: run the gate, do not score.
python3 -m eval.gate_marker require --spec "$SPEC" --model "$MODEL" --dataset "$DATASET"
# shellcheck disable=SC2086  # EXTRA is a flag list
python3 -m eval.matrix run --spec "$SPEC" --models "$MODEL" --datasets "$DATASET" --out-dir "$OUT" $EXTRA | tee "$OUT.task-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.log"
summary=$(grep MATRIX_RUN_OK "$OUT.task-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.log" | tail -1)
rm -f "$OUT.task-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.log"
[ -n "$summary" ] || { echo "FATAL: no MATRIX_RUN_OK line" >&2; exit 1; }
case "$summary" in
    *"'done': 0, 'cached': 0,"*) echo "MISSING: every cell of $MODEL x $DATASET lacks features ($summary)" >&2; exit 3 ;;
esac
echo "done $(date -Is): $summary"
