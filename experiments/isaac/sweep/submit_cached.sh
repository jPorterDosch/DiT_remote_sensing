#!/bin/bash
# ============================================================================
# Bank FLUX features at each dataset's SELECTED block, caches KEPT, for feature-level
# follow-ups the sweep cannot support (its caches are deleted after probing): timestep
# aggregation with base protected + shuffled null (rule 3/3b), other aggregators, C-grid
# extension. One GPU array task per dataset, extraction only (EXTRACT_ONLY=1, no probe ->
# no idle GPU), same recipe as the sweep (ens8, 7-t grid, g=1.0, seeds 42/43/44).
#
#   bash experiments/isaac/sweep/submit_cached.sh                 # dry run: print the sbatch lines
#   bash experiments/isaac/sweep/submit_cached.sh --submit        # submit them
#   bash experiments/isaac/sweep/submit_cached.sh m_forestnet     # subset of datasets
#
# The block comes from the dataset's sweep-select result (eval.sweep select, val-only), so
# nothing here re-selects. SWEEP_ROOT is models/<dataset>_cached_ens8: the sweep's run dirs
# under models/<dataset>_sweep_ens8 still hold their _meta.json, so the same config hash
# there would hit run.py's "Save directory already exists". Already-complete caches are
# skipped by sweep_block.sh, so re-running this is safe.
# STORAGE: ~0.7-1.9 GB per dataset, ~9 GB for all six (scratch; not backed up).
# ============================================================================
set -euo pipefail
_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
cd "$_dir"
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"

SUBMIT=0
DATASETS=()
for a in "$@"; do
    case "$a" in
        --submit) SUBMIT=1 ;;
        -*) echo "unknown flag $a" >&2; exit 1 ;;
        *) DATASETS+=("$a") ;;
    esac
done
[ "${#DATASETS[@]}" -gt 0 ] || DATASETS=(m_eurosat m_forestnet m_so2sat m_brick_kiln m_pv4ger m_bigearthnet)

# KS is defined once, in sweep.sbatch; the array index must address the same list.
KS_LINE="$(grep -m1 -E '^KS=\(' experiments/isaac/sweep/sweep.sbatch)"
[ -n "$KS_LINE" ] || { echo "FATAL: no KS=(...) in sweep.sbatch" >&2; exit 1; }
eval "$KS_LINE"

for d in "${DATASETS[@]}"; do
    read -r K HOURS < <(python3 - "$d" <<'PY'
import glob, json, math, sys
import numpy as np
from eval.features import official_spec
d = sys.argv[1]
hits = [p for p in glob.glob(f"results/eval/sweep/sweep-select_{d}_flux-oneshot-ens8_*.npz")
        if not bool(np.load(p, allow_pickle=True)["smoke"])]
if len(hits) != 1:
    sys.exit(f"FATAL: {d}: expected exactly 1 non-smoke sweep-select result, found {hits}")
k = int(json.loads(str(np.load(hits[0], allow_pickle=True)["info"]))["selected"]["k"])
n = sum(official_spec(d)["sizes"].values())
# extraction only: (k+1)/29 x 2.0 s/img (H100, k=28 measured) over every image, x1.5, +1 h
print(k, math.ceil(1.5 * (k + 1) / 29 * 2.0 * n / 3600 + 1))
PY
    )
    IDX=-1
    for i in "${!KS[@]}"; do [ "${KS[$i]}" = "$K" ] && IDX=$i; done
    [ "$IDX" -ge 0 ] || { echo "FATAL: $d: selected k=$K not in sweep.sbatch KS=(${KS[*]})" >&2; exit 1; }
    cmd=(sbatch --job-name=geobench-cached --array="$IDX" --time="${HOURS}:00:00"
        --export=ALL,DATASET="$d",SWEEP_ROOT="models/${d}_cached_ens8",DELETE_CACHE=0,EXTRACT_ONLY=1,KEEP_CACHES=1,REUSE_K=,REUSE_DIR=
        experiments/isaac/sweep/sweep.sbatch)
    echo "$d: selected k=$K (array index $IDX), --time=${HOURS}h"
    if [ "$SUBMIT" = 1 ]; then "${cmd[@]}"; else echo "  ${cmd[*]}"; fi
done
[ "$SUBMIT" = 1 ] || echo "(dry run -- add --submit to queue)"
