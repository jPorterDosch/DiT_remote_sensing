#!/bin/bash
# ============================================================================
# Submit one dataset's t x k sweep with --time sized to that dataset, refusing anything
# over ISAAC's 72 h job limit. Every array task (one block k) is an independent, resumable
# job, and datasets are separate submissions, so the ~475 H100-h spread across the queue.
#   bash experiments/isaac/sweep/submit.sh m_so2sat
#   REUSE_K=28 REUSE_DIR=models/m_eurosat_oneshot_ens8 bash experiments/isaac/sweep/submit.sh m_eurosat
# Slowest task (k=54): extraction (k+1)/29 x 2.0 s/img (H100; one-shot exits at block k,
# m-eurosat k=28 measured) over all official images, + probing (2 h; 20 h multi-label, 43
# one-vs-rest fits per cell, unmeasured -- check the first finished task), x1.5 margin.
# ============================================================================
set -euo pipefail
DATASET="${1:?usage: submit.sh <eval/features.OFFICIAL key>}"
_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
cd "$_dir"

HOURS=$(python3 - "$DATASET" <<'PY'
import math, sys
from eval.features import official_spec
s = official_spec(sys.argv[1])
h = 1.5 * (55 / 29 * 2.0 * sum(s["sizes"].values()) / 3600 + (20 if s.get("multilabel") else 2))
if h > 72:
    sys.exit(f"estimated {h:.0f} h per block > the 72 h ISAAC limit: shard the extraction first")
print(math.ceil(h))
PY
)
echo "$DATASET: --time=${HOURS}:00:00 per block"
sbatch --time="${HOURS}:00:00" \
    --export=ALL,DATASET="$DATASET",REUSE_K="${REUSE_K:-}",REUSE_DIR="${REUSE_DIR:-}" \
    experiments/isaac/sweep/sweep.sbatch
