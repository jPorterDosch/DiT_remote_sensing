#!/bin/bash
# ============================================================================
# Final geographic exclusion over the complete Git-10M metadata (eval/geo_dedup.py dedup).
#
#   mkdir -p logs && sbatch --export=ALL,BUFFER_M=1000 experiments/isaac/geo_dedup/dedup.sbatch
#
# GATE: refuses to start unless all 787 meta-NNNNN.parquet files exist (never a partial
# manifest from a job). Output: $STORE/geo_dedup/dedup_b<BUFFER_M>/ (refuses to overwrite).
# Size: 87 shards (1.16 M rows) took 8 s on the login node; 787 shards ~ 16.7 M rows, pandas
# strings ~ 5 GB. CPU only (`short`).
# ============================================================================
#SBATCH --job-name=git10m-dedup
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
python3 -c "import sys, pandas, pyarrow, scipy; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

BUFFER_M="${BUFFER_M:?set BUFFER_M (metres)}"
META="$STORE/geo_dedup/git10m_meta"
POINTS="$STORE/geo_dedup/points_full.csv"
OUT="$STORE/geo_dedup/dedup_b${BUFFER_M}"
n=$(ls "$META" | grep -c -E '^meta-[0-9]{5}\.parquet$')
[ "$n" -eq 787 ] || { echo "FATAL: $n/787 meta parquets in $META" >&2; exit 1; }
[ -f "$POINTS" ] && [ -f "$POINTS.report.json" ] || { echo "FATAL: $POINTS(.report.json) missing" >&2; exit 1; }
python3 -m eval.geo_dedup dedup --meta "$META" --expect-shards 787 --points "$POINTS" \
    --buffer-m "$BUFFER_M" --out "$OUT"
echo "DEDUP_OK -> $OUT"
