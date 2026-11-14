#!/bin/bash
# ============================================================================
# GEO-Bench SEGMENTATION tasks -> the image/mask PNG trees (eval/export_geobench.py SEG_SPECS):
# download (md5), `--count` what the shipped artifact says (rule 16: fill OFFICIAL_SEG / SEG_SPECS
# from THIS output, never a paper), export, manifest check. One task per array index. CPU.
#
#   COUNT=1  print the shipped partition sizes / classes / band stats / sample ranges and stop
#   WRITE_MANIFEST=1  first export of a task: write eval/manifests/<key>.json (commit it)
#
#   sbatch --export=ALL,COUNT=1 --array=0-2 experiments/isaac/seg/export.sbatch
#   sbatch --export=ALL,WRITE_MANIFEST=1 --array=0-2 experiments/isaac/seg/export.sbatch
# Raw data: $GB_RAW/segmentation_v1.0/<task>; trees: data/<key>/ (scratch via scratch_env.sh).
# ============================================================================
#SBATCH --job-name=seg-export
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python3 -c "import sys, numpy, geobench, PIL; assert 'DiTF' in sys.prefix, sys.prefix" || { echo "FATAL: bad env" >&2; exit 1; }
set -euo pipefail
TASKS=(m-pv4ger-seg m-chesapeake-landcover m-SA-crop-type)
IDX="${SLURM_ARRAY_TASK_ID:?submit with --array=0-2}"
TASK="${TASKS[$IDX]}"
KEY="${TASK//-/_}"
GB_RAW="${GB_RAW:-/lustre/isaac24/scratch/jdosch1/satdifuser/geobench}"
RAW="$GB_RAW/segmentation_v1.0/$TASK"
ARGS=(--task "$TASK" --dataset-dir "$RAW" --out "data/$KEY")
[ -f "$RAW/.data_unzipped" ] || ARGS+=(--download)
if [ "${COUNT:-0}" = 1 ]; then
    python3 -m eval.export_geobench "${ARGS[@]}" --count
    echo "SEG_COUNT_OK $TASK"; exit 0
fi
[ "${WRITE_MANIFEST:-0}" = 1 ] && ARGS+=(--write-manifest)
python3 -m eval.export_geobench "${ARGS[@]}"
echo "SEG_EXPORT_OK $TASK -> data/$KEY"
