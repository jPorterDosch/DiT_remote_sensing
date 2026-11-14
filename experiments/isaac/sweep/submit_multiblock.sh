#!/bin/bash
# ============================================================================
# Kick off multi-block feature banking on the FULL grid, one pass per dataset into one dir:
# t 100-180-260-340-420-500-580-740-900 x k 4, 8, 12, 18 (double) + 19-24-28-33-38-43-48-54-55-56
# (single) -> models/<ds>_multiblock_ens8, feats (N, 9, 14, 3072). ~156 GB total.
# ONE GPU smoke (gate: double + single blocks, cross-t-list pairing) on m_eurosat first; the
# dataset jobs start only if it prints SMOKE GATE: PASS (--dependency=afterok).
#   bash experiments/isaac/sweep/submit_multiblock.sh            # dry run: print the plan
#   bash experiments/isaac/sweep/submit_multiblock.sh --submit   # queue smoke + 6 dataset jobs
# --time from the MEASURED rate (2.31-2.39 s/img for 7 t to depth 56, ens8, H100, 2026-10-05):
# 0.337 s per timestep per image at depth 56, x (deepest + 1) / 57, x1.5 margin, + 1 h.
# ============================================================================
set -euo pipefail
_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
cd "$_dir"
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"

SUBMIT=0
[ "${1:-}" = "--submit" ] && SUBMIT=1
DATASETS=(m_eurosat m_forestnet m_so2sat m_brick_kiln m_pv4ger m_bigearthnet)
BLOCKS="${BLOCKS:-4-8-12-18-19-24-28-33-38-43-48-54-55-56}"  # hyphen-separated: safe inside sbatch --export
TS="${TS:-100-180-260-340-420-500-580-740-900}"
SB=experiments/isaac/sweep/multiblock.sbatch

# Refuse a dataset whose extraction is already pending/running: a second job on the same
# SAVE_ROOT would delete the first one's live run dir (no _meta.json yet) as "incomplete".
for d in "${DATASETS[@]}"; do
    live="$(squeue -u "$USER" -h -n geobench-multiblock -o "%i" | while read -r j; do
        if sacct -j "$j" -X -n -P --format=SubmitLine | grep -q "DATASET=$d,"; then echo "$j"; fi
    done || true)"
    [ -z "$live" ] || { echo "FATAL: $d already has multiblock extraction job(s) queued: $live" >&2; exit 1; }
done
for d in "${DATASETS[@]}"; do
    for f in models/"$d"_multiblock_ens8/*/multistep_*_feats_oneshot_g1.0_k*.npz; do
        [ -e "$f" ] && echo "note: $d already has multi-block caches in models/${d}_multiblock_ens8 (kept; only exact t/k matches are reused)" && break
    done
done

smoke=(sbatch --parsable --job-name=geobench-multiblock-smoke --time=1:30:00
    --export=ALL,DATASET=m_eurosat,SMOKE=1 "$SB")
echo "smoke: ${smoke[*]}"
SMOKE_ID="<smoke-jobid>"
if [ "$SUBMIT" = 1 ]; then SMOKE_ID="$("${smoke[@]}")"; echo "  -> $SMOKE_ID"; fi

for d in "${DATASETS[@]}"; do
    HOURS="$(python3 - "$d" "$TS" "$BLOCKS" <<'PY'
import math, sys
from eval.features import official_spec
n = sum(official_spec(sys.argv[1])["sizes"].values())
n_t, deepest = len(sys.argv[2].split("-")), int(sys.argv[3].split("-")[-1])
h = 1.5 * 0.337 * n_t * (deepest + 1) / 57 * n / 3600 + 1
if h > 72:
    sys.exit(f"{sys.argv[1]}: {h:.0f} h > 72 h ISAAC limit -- shard first")
print(math.ceil(h))
PY
)"
    cmd=(sbatch --job-name=geobench-multiblock --time="${HOURS}:00:00" --dependency="afterok:$SMOKE_ID"
        --export=ALL,DATASET="$d",TS="$TS",BLOCKS="$BLOCKS" "$SB")
    echo "$d (${HOURS} h): ${cmd[*]}"
    [ "$SUBMIT" = 1 ] && "${cmd[@]}"
done
[ "$SUBMIT" = 1 ] || echo "(dry run -- add --submit to queue)"
