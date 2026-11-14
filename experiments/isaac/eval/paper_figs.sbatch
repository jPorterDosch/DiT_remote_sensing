#!/bin/bash
# Paper figures from artifacts on disk (CPU, minutes): gate tests first, then the named figures.
#   sbatch experiments/isaac/eval/paper_figs.sbatch                      # gate tests only
#   FIGS="rs" sbatch experiments/isaac/eval/paper_figs.sbatch            # tests, then paper_figs.py rs
#   FIGS="null trade" TEST=0 sbatch experiments/isaac/eval/paper_figs.sbatch
# Env: FIGS (paper_figs.py selectors, space separated; empty = none), TEST (1 = run the gate tests, default),
# plus whatever the selected figure reads (NULL_ARMS / NULL_IDX / NULL_NAME, RS_ARMS / RS_IDX / RS_NAME / RS_ROOT / RS_RUN, NOTITLE).
#SBATCH --job-name=paper-figs
#SBATCH -A isaac-utk0256
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"; cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MPLBACKEND=Agg
set -euo pipefail
if [ "${TEST:-1}" = 1 ]; then
  python3 -m pytest -q -p no:cacheprovider -rfE tests/test_paper_figs_guard.py tests/test_prompt_gen.py
fi
if [ -n "${FIGS:-}" ]; then
  # shellcheck disable=SC2086  (FIGS is a space-separated selector list)
  python3 experiments/figs/paper_figs.py $FIGS
fi
echo "PAPER_FIGS_JOB_OK test=${TEST:-1} figs=${FIGS:-<none>}"
