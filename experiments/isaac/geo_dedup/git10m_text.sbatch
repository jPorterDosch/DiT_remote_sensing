#!/bin/bash
# ============================================================================
# Git-10M captions of the SSL4EO-S12 rows (shards 759-778), used only to verify the
# SSL4EO name -> location mapping (eval/geo_dedup.py ssl4eo-boxes). Same download / resume /
# cleanup rules as git10m_meta.sbatch; 4 tasks x 5 shards.
#
#   mkdir -p logs && sbatch experiments/isaac/geo_dedup/git10m_text.sbatch
# ============================================================================
#SBATCH --job-name=git10m-text
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=1:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # HF cache + models/data/logs on scratch

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
python3 -c "import sys, pandas, pyarrow, scipy, huggingface_hub; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

PER_TASK=5
FIRST=$((759 + SLURM_ARRAY_TASK_ID * PER_TASK))
LAST=$((FIRST + PER_TASK - 1)); [ "$LAST" -gt 778 ] && LAST=778
OUT="$STORE/geo_dedup/git10m_text_ssl4eo"
echo "shards $FIRST-$LAST -> $OUT"
python3 -m eval.geo_dedup git10m-text --shard-dir "$STORE/geo_dedup/git10m_shards" \
    --shards "$FIRST-$LAST" --out "$OUT" --prefix SSL4EO-S12 --hf-download --rm-shard
