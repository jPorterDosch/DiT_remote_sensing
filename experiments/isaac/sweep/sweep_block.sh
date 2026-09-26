#!/bin/bash
# ============================================================================
# One block of a t x k sweep on ANY registered official-split dataset:
# extract -> probe -> verify -> delete caches.
#   DATASET=m_eurosat bash experiments/isaac/sweep/sweep_block.sh <k>
# Called per array task by sweep.sbatch (after scratch_env.sh); runnable directly for the
# workstation smoke. Must run from the repo root. Nothing here is dataset-specific: the
# dataset must be registered in eval/features.OFFICIAL (split sizes, root, classes) and
# have a run.py dataset wrapper (FLUX extraction); both refuse loudly otherwise.
#
# 1. Extract every official split at block k (ens8, 7-t grid, g=1.0) with DISTINCT
#    per-split seeds (train 42, test 43, val 44 -- RESEARCH_NOTES 6t F1). A split whose
#    cache already exists is skipped, so a re-queued task resumes after extraction.
#    Optional reuse of banked caches: REUSE_K=<k> REUSE_DIR=<dir> (e.g. m-eurosat's k=28
#    campaign) -- reused caches are probed but NEVER deleted.
# 2. python -m eval.sweep block: val accuracy of every (t-candidate x C) cell in the clear,
#    test predictions SEALED (selection happens later, across blocks, on val).
# 3. Delete this block's feature .npz files (keep *_meta.json) only if step 2 printed
#    BLOCK OK and the result re-verifies against the cache's own t-grid.
#
# Env: DATASET (required), ENS (8), DELETE_CACHE (1), SWEEP_ROOT
#      (models/${DATASET}_sweep_ens$ENS), DATA (registry root), OUT_DIR (results/eval/sweep),
#      REUSE_K / REUSE_DIR (optional), SUBSET_ARGS + SMOKE_SIZES (smoke only).
# Smoke:  DATASET=m_eurosat ENS=2 SUBSET_ARGS="--subset-size 20" SMOKE_SIZES=20,20,20 \
#         SWEEP_ROOT=models/smoke_sweep OUT_DIR=results/eval/smoke_sweep \
#         bash experiments/isaac/sweep/sweep_block.sh 33
# ============================================================================
set -euo pipefail
K="${1:?usage: DATASET=<name> sweep_block.sh <k>}"
DATASET="${DATASET:?set DATASET to an eval/features.OFFICIAL key}"
ENS="${ENS:-8}"
DELETE_CACHE="${DELETE_CACHE:-1}"
SWEEP_ROOT="${SWEEP_ROOT:-models/${DATASET}_sweep_ens${ENS}}"
OUT_DIR="${OUT_DIR:-results/eval/sweep}"
SUBSET_ARGS="${SUBSET_ARGS:-}"
SMOKE_SIZES="${SMOKE_SIZES:-}"
REUSE_K="${REUSE_K:-}"
REUSE_DIR="${REUSE_DIR:-}"
ARM="flux-oneshot-ens${ENS}"
FNAME="multistep_{split}_feats_oneshot_g1.0.npz"

# Registry lookups (fail loudly for an unregistered dataset): data root + split names.
read -r REG_ROOT SPLITS < <(python3 -c "
import sys
from eval.features import official_spec
s = official_spec(sys.argv[1])
print(s['root'], ','.join(s['sizes']))" "$DATASET")
DATA="${DATA:-$REG_ROOT}"
IFS=',' read -r -a SPLIT_LIST <<<"$SPLITS"
[ -d "$DATA/${SPLIT_LIST[0]}" ] || { echo "FATAL: $DATA missing -- export $DATASET first (eval.export_geobench)" >&2; exit 1; }

seed_for() { case "$1" in train) echo 42 ;; test) echo 43 ;; val) echo 44 ;; *) echo "FATAL: no seed for split $1" >&2; exit 1 ;; esac; }
have_split() { ls "$1"/*/multistep_"$2"_feats_oneshot_g1.0.npz >/dev/null 2>&1; }

REUSE=0
if [ -n "$REUSE_K" ] && [ "$K" = "$REUSE_K" ] && [ -z "$SUBSET_ARGS" ]; then
    [ -n "$REUSE_DIR" ] || { echo "FATAL: REUSE_K set without REUSE_DIR" >&2; exit 1; }
    for split in "${SPLIT_LIST[@]}"; do
        have_split "$REUSE_DIR" "$split" || { echo "FATAL: REUSE_DIR $REUSE_DIR lacks the $split cache" >&2; exit 1; }
    done
    REUSE=1
    DIR="$REUSE_DIR"
    echo "k=$K: reusing banked caches in $DIR (never deleted)"
else
    DIR="$SWEEP_ROOT/k$K"
    for split in "${SPLIT_LIST[@]}"; do
        SEED="$(seed_for "$split")"
        if have_split "$DIR" "$split"; then
            echo "k=$K $split: cache exists, skipping extraction"; continue
        fi
        echo "=== $DATASET k=$K extract $split (seed $SEED) $(date -Is)"
        # shellcheck disable=SC2086
        python3 run.py --task extract \
            --model.name flux --model.ensemble-size "$ENS" \
            --dataset.name "$DATASET" --dataset.path "$DATA" \
            --img-size 224 224 --t 100 180 260 340 420 500 580 \
            --extraction-mode ONESHOT --k "$K" --guidance-scale 1.0 \
            --seed "$SEED" --eps-seed "$SEED" --subset-seed 42 \
            --extract-split "$split" --label-fraction 1.0 --batch-size 1 \
            $SUBSET_ARGS --save-dir "$DIR"
    done
fi

echo "=== $DATASET k=$K probe $(date -Is)"
SMOKE_FLAG=()
[ -n "$SMOKE_SIZES" ] && SMOKE_FLAG=(--smoke-sizes "$SMOKE_SIZES")
LOG="$(mktemp)"
python3 -m eval.sweep block --dataset "$DATASET" --k "$K" --arm "$ARM" \
    --features "$DIR/*/$FNAME" --expect extraction_mode=ONESHOT "ensemble_size=$ENS" \
    --out-dir "$OUT_DIR" "${SMOKE_FLAG[@]}" | tee "$LOG"
RESULT="$(sed -n 's/^BLOCK OK: .* -> //p' "$LOG" | tail -1)"
rm -f "$LOG"
[ -n "$RESULT" ] && [ -f "$RESULT" ] || { echo "FATAL: no verified block result; caches kept" >&2; exit 1; }
python3 -c "import sys; from eval.sweep import verify_block; verify_block(sys.argv[1])" "$RESULT"

if [ "$REUSE" = "1" ] || [ "$DELETE_CACHE" != "1" ]; then
    echo "k=$K: caches kept ($([ "$REUSE" = 1 ] && echo banked || echo DELETE_CACHE=$DELETE_CACHE))"
else
    for split in "${SPLIT_LIST[@]}"; do
        for f in "$DIR"/*/multistep_"$split"_feats_oneshot_g1.0.npz; do
            [ -f "$f" ] && rm -v "$f"
        done
    done
    echo "k=$K: feature caches deleted; *_meta.json kept as provenance"
fi
echo "k=$K done -> $RESULT"
