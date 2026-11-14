#!/bin/bash
# ============================================================================
# CPU probe sweep over the MULTI-BLOCK FLUX caches (models/<ds>_multiblock_ens8, N x 9 t x 14 k
# x 3072): one array task per block, exactly the sweep's official protocol (eval.sweep block:
# each single t + the all-t concat, x C_GRID, val in the clear, test SEALED). The loader serves
# the block's slice of the multi-block cache (features._block_slice, memmapped), so nothing
# downstream changes. Caches are NEVER deleted here. Results: results/eval/sweep_mb/.
#   submit with experiments/isaac/sweep/submit_probe_multiblock.sh (gate + per-dataset select)
# Env: DATASET (required), BLOCKS (hyphen list, must equal the cache's), ARM (flux-oneshot-ens8-mb),
#      WIDE_C (1: C in 1e-6..1, the default; 0: legacy 0.01..1)
#      CACHE_DIR + LORA_TAG (+ ARM, OUT_DIR): an adapted arm's git10m_eval caches, see below
# TIME: single-label blocks took 7-72 min with 8 candidates in the sweep; 10 here (9 t +
# concat over 27648-d). m-bigearthnet (43 one-vs-rest fits per cell) took 6-9 h with 8.
# ============================================================================
#SBATCH --job-name=geobench-probe-mb
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=long
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --array=0-13
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: a worktree's .git is a file
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
python3 -c "import sys, numpy, sklearn; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OVR_N_JOBS="${SLURM_CPUS_PER_TASK:-8}"  # multi-label one-vs-rest workers; results identical
export JOBLIB_TEMP_FOLDER="$STORE/tmp/joblib"  # worker memmaps of wide X (GBs): Lustre, not node /tmp
mkdir -p "$JOBLIB_TEMP_FOLDER"
# Orphaned one-vs-rest memmaps (SIGKILL / OOM skip every cleanup): older than any job can live
# (QOS long caps at 6 days), so never one in use.
find "$JOBLIB_TEMP_FOLDER" -maxdepth 1 -name 'ovr_X_*.npy' -mtime +7 -delete 2>/dev/null || true
[ -f "$PROJECT_ROOT/.secrets/wandb-personal.env" ] && source "$PROJECT_ROOT/.secrets/wandb-personal.env"
export WANDB_MODE="${WANDB_MODE:-offline}"
set -euo pipefail

DATASET="${DATASET:?submit with --export=ALL,DATASET=<OFFICIAL key>}"
BLOCKS="${BLOCKS:-4-8-12-18-19-24-28-33-38-43-48-54-55-56}"
# HEAD=lr (default): each t + concat x C. HEAD=tsweight: the learned timestep-weighting head
# (eval/fusion.py; lr/wd/epoch selected on val, no C grid) -- its own arm, its own select.
HEAD="${HEAD:-lr}"
case "$HEAD" in lr) DEF_ARM=flux-oneshot-ens8-mb ;; tsweight) DEF_ARM=flux-oneshot-ens8-mb-tsw ;;
    *) echo "FATAL: HEAD=$HEAD (lr|tsweight)" >&2; exit 1 ;; esac
ARM="${ARM:-$DEF_ARM}"
OUT_DIR="${OUT_DIR:-results/eval/sweep_mb}"
# C grid: the WIDE grid by default (protocols.C_GRID_WIDE, 1e-6..1; 2026-10-07). WIDE_C=0 runs
# the legacy 3-C grid. Passed as a flag, not a comma list: commas would split sbatch --export.
C_ARGS=(--c-grid wide)  # protocols.C_GRID_WIDE
[ "${WIDE_C:-1}" = 1 ] || C_ARGS=()
[ "$HEAD" = tsweight ] && C_ARGS=(--head tsweight)  # replaces the C grid (block AND done-status)
read -r -a BLOCK_LIST <<<"${BLOCKS//-/ }"
# CACHE_DIR / LORA_TAG (2026-10-29): probe an ADAPTED arm's multi-block caches instead of the frozen
# ones. extract_ckpt.sbatch writes them under models/git10m_eval/<run>/<dataset>/ with the LoRA tag
# in the file name (tasks/extraction.py: ..._oneshot_g1.0_lora-<tag>_k<blocks>.npz), same layout
# (N, 9 t, 14 k, 3072), so the block slice, the gate and select are unchanged. Give ARM and OUT_DIR
# of its own so the frozen sweep's results are never mixed with it.
CACHE_DIR="${CACHE_DIR:-models/${DATASET}_multiblock_ens8}"
LORA_TAG="${LORA_TAG:-}"
FEATS_DEFAULT="$CACHE_DIR/*/multistep_{split}_feats_oneshot_g1.0${LORA_TAG:+_lora-$LORA_TAG}_k${BLOCKS}.npz"
FEATS="${FEATS:-$FEATS_DEFAULT}"  # ({split} cannot sit inside ${FEATS:-...}: its } ends the expansion)
EXPECT=(extraction_mode=ONESHOT ensemble_size=8)  # the ONE pin list for block AND gate (rule 9)
# An adapted cache stamps weights='flux-dev+lora:<tag>' (features.PINS_COMMON pins 'flux-dev' by
# default, so the gate refuses the cache without this pin -- as it did on 2026-10-29).
[ -n "$LORA_TAG" ] && EXPECT+=("weights=flux-dev+lora:$LORA_TAG")
# NGROUPS=1 (default): array task i runs block i. NGROUPS=G: task i runs every block j with
# j % G == i, sequentially -- packs a dataset into G jobs (QOS long allows 18 SUBMITTED jobs/user).
# KS (hyphen list, default = BLOCKS): the blocks to PROBE, a subset of the cache's BLOCKS (e.g. the
# 4-block proxy 12-28-38-56 for an adapted arm). The array packs KS, not BLOCKS.
KS="${KS:-$BLOCKS}"
read -r -a KS_LIST <<<"${KS//-/ }"
for k in "${KS_LIST[@]}"; do
    case " ${BLOCK_LIST[*]} " in *" $k "*) ;; *) echo "FATAL: KS block $k not in BLOCKS=$BLOCKS" >&2; exit 1 ;; esac
done
NGROUPS="${NGROUPS:-${#KS_LIST[@]}}"
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
MY_KS=()
for j in "${!KS_LIST[@]}"; do [ $((j % NGROUPS)) = "$TASK" ] && MY_KS+=("${KS_LIST[$j]}"); done
if [ "${#MY_KS[@]}" = 0 ]; then  # array larger than NGROUPS: nothing to do, and must not fail afterok
    echo "task $TASK >= NGROUPS=$NGROUPS: no blocks assigned, nothing to do"; exit 0
fi
echo "=== probe-mb $DATASET blocks ${MY_KS[*]} (task $TASK of $NGROUPS) arm=$ARM host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} $(date -Is)"

# GATE=1 (submit_probe_multiblock.sh's gate task): executed PASS/FAIL checks (rule 13) before any
# dataset array may start. BEFORE probing (seconds): the loader's block slice equals the raw
# cache's slice (val split) -- a wrong block index / memmap offset fails here, not after an 8 h
# probe. AFTER probing: positive control, best val beats the best constant predictor by >= 0.10.
gate() {  # gate <slice|control> <k>   (eval.sweep.gate: tested in tests/test_geobench.py)
    python3 -m eval.sweep gate --phase "$1" --dataset "$DATASET" --k "$2" --features "$FEATS" \
        --out-dir "$OUT_DIR" --arm "$ARM" --expect "${EXPECT[@]}"
}
# GATE_SLICE_EXTRA (default 4): also slice-gate a DOUBLE-stream block, whose stamp and memmap
# offset differ from the single block the gate task probes (review 2026-10-11).
if [ "${GATE:-0}" = 1 ]; then
    for K in "${MY_KS[@]}" ${GATE_SLICE_EXTRA-4}; do gate slice "$K"; done
fi

for K in "${MY_KS[@]}"; do
    done_glob=("$OUT_DIR"/sweep-block_"${DATASET}"_"${ARM}"-k"${K}"_*.npz)
    if [ -f "${done_glob[0]}" ]; then
        [ "${#done_glob[@]}" = 1 ] || { echo "FATAL: k=$K has ${#done_glob[@]} results in $OUT_DIR -- keep one" >&2; exit 1; }
        # Skip only a result of THIS sweep: same input caches (sha1) and same C grid.
        rc=0
        python3 -m eval.sweep done-status "${done_glob[0]}" --dataset "$DATASET" --features "$FEATS" \
            --cache-dir "$CACHE_DIR" "${C_ARGS[@]}" || rc=$?
        case "$rc" in
            0) echo "k=$K: verified block result for these caches exists, nothing to do"; continue ;;
            3|4|5) echo "FATAL: k=$K: ${done_glob[0]} is not a current result for these caches (code $rc," \
                       "see above) -- move it aside, then re-queue" >&2; exit 1 ;;
            *) echo "FATAL: k=$K: done-status itself failed (exit $rc) -- NOT a verdict; investigate" >&2; exit 1 ;;
        esac
    fi
    echo "=== probe-mb $DATASET k=$K start $(date -Is)"
    python3 -m eval.sweep block --dataset "$DATASET" --k "$K" --arm "$ARM" \
        --features "$FEATS" --expect "${EXPECT[@]}" --out-dir "$OUT_DIR" "${C_ARGS[@]}"
    echo "=== probe-mb $DATASET k=$K done $(date -Is)"
done

if [ "${GATE:-0}" = 1 ]; then
    for K in "${MY_KS[@]}"; do gate control "$K"; done
fi
