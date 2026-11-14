#!/bin/bash
# ============================================================================
# PROBE-ONLY stage of the t x k sweep, on CPU (`long`), for blocks whose feature caches
# already exist. Same array layout and same per-block work as sweep.sbatch: it calls
# sweep_block.sh <k> unchanged, which skips every split with a complete cache and goes
# straight to eval.sweep block -> verify -> delete caches.
#
#   sbatch --array=0-7 --export=ALL,DATASET=m_bigearthnet experiments/isaac/sweep/probe_cpu.sbatch
#
# WHY: the probe is sklearn (CPU). Run inside a GPU job it idles the H100; single-label
# probes finish in 7-70 min, but m-bigearthnet's multi-label probe (43 one-vs-rest fits
# per cell, 24 cells) ran 2-3 h without finishing and the idle-GPU policy cancelled
# 6303849_0/_1 (2026-10-02).
#
# GUARDS (checked before sweep_block.sh runs, so this job can never start FLUX on a CPU):
#   - block result already in $OUT_DIR  -> exit 0 if eval.sweep done-status verifies it (sha1s);
#     a legacy result without recorded inputs is FATAL unless ACCEPT_LEGACY=1
#   - any split cache incomplete        -> FATAL (rerun that block on GPU via sweep.sbatch)
# PROBE_CPU_GUARD_ONLY=1 stops after the guards (login-node dry run, no probing).
#
# PARTITION: `long` (QoS long: 6-day cap, 96 CPUs / 12 jobs per user), 8 CPUs per task so
# 12 tasks can run at once. lbfgs is BLAS-threaded, so OMP_NUM_THREADS = full allocation.
# TIME: 3 days by default for the unmeasured multi-label probe; pass --time=4:00:00 for
# single-label datasets (measured 7-72 min per block). Memory: train features are
# 20000 x 7 x 3072 float32 = 1.7 GB; 64G leaves room for the float64 + scaled copies.
# ============================================================================
#SBATCH --job-name=geobench-probe
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=long
#SBATCH --partition=long
#SBATCH --time=3-00:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # caches + models/data/logs on Lustre scratch

# PATH-only env (as dino_probe.sbatch): _common.sh's setup_environment loads the CUDA
# module, which a CPU node does not need.
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

export DATASET="${DATASET:?submit with --export=ALL,DATASET=<OFFICIAL key>}"
# KS is defined once, in sweep.sbatch (as submit_cached.sh reads it): array index i -> the same k.
eval "$(grep -m1 -E '^KS=\(' "$PROJECT_ROOT/experiments/isaac/sweep/sweep.sbatch")"
[ "${#KS[@]}" -gt 0 ] || { echo "FATAL: no KS=(...) in sweep.sbatch" >&2; exit 1; }
K="${KS[${SLURM_ARRAY_TASK_ID:?run as an array job}]}"
ENS="${ENS:-8}"
SWEEP_ROOT="${SWEEP_ROOT:-models/${DATASET}_sweep_ens${ENS}}"
# Banked caches (submit_cached.sh -> *_cached_ens*) exist to be KEPT. One mechanism guards that:
# the .keep_caches marker sweep_block.sh honors (submit_cached.sh's KEEP_CACHES=1 writes it); a
# banked dir that predates the marker gets it here, so no DELETE_CACHE can remove it.
case "$SWEEP_ROOT" in *_cached_*) [ -d "$SWEEP_ROOT/k$K" ] && touch "$SWEEP_ROOT/k$K/.keep_caches" ;; esac
# Banked caches (*_cached_*) are probed into their OWN results dir by default: the sweep's
# results for the same (dataset, k) live in results/eval/sweep and would read as "done".
case "$SWEEP_ROOT" in
    *_cached_*) OUT_DIR="${OUT_DIR:-results/eval/sweep_cached}" ;;
    *) OUT_DIR="${OUT_DIR:-results/eval/sweep}" ;;
esac
export SWEEP_ROOT OUT_DIR  # sweep_block.sh reads both: a default set here must reach it
echo "probe $DATASET task ${SLURM_ARRAY_TASK_ID} -> block k=$K  host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}"

if [ -n "${REUSE_K:-}" ] && [ "$K" = "$REUSE_K" ]; then DIR="${REUSE_DIR:?REUSE_K set without REUSE_DIR}"; else DIR="$SWEEP_ROOT/k$K"; fi
SPLITS="$(python3 -c "import sys; from eval.features import official_spec; print(' '.join(official_spec(sys.argv[1])['sizes']))" "$DATASET")"

done_glob=("$OUT_DIR"/sweep-block_"${DATASET}"_flux-oneshot-ens"${ENS}"-k"${K}"_*.npz)
if [ -f "${done_glob[0]}" ]; then
    [ "${#done_glob[@]}" = 1 ] || { echo "FATAL: k=$K has ${#done_glob[@]} results in $OUT_DIR -- keep one" >&2; exit 1; }
    # One decision from what the result RECORDS (eval.sweep done_status): its inputs' paths and
    # sha1s. A result file exists only after verify_block passed (atomic rename in run_block).
    rc=0
    python3 -m eval.sweep done-status "${done_glob[0]}" --dataset "$DATASET" --cache-dir "$DIR" \
        --features "$DIR/*/multistep_{split}_feats_oneshot_g1.0.npz" || rc=$?
    case "$rc" in
        0)
            echo "k=$K: verified block result for these caches exists (${done_glob[0]}), nothing to do"
            # A task killed between run_block's rename and sweep_block.sh step 3 leaves verified
            # caches behind: free them under step 3's own rules (sweep_block.sh FREE_ONLY) --
            # except in a guard-only dry run, which must never delete anything.
            if [ "${PROBE_CPU_GUARD_ONLY:-0}" = 1 ]; then
                echo "guard-only: would apply step 3's delete rules to $DIR (nothing deleted)"
            else
                FREE_ONLY=1 bash "$PROJECT_ROOT/experiments/isaac/sweep/sweep_block.sh" "$K"
            fi
            exit 0 ;;
        3) echo "FATAL: k=$K: ${done_glob[0]} is stale for the caches in $DIR (see above) -- move it aside" >&2; exit 1 ;;
        4)  # mtime-only evidence (a copy without -p passes): never counted done silently, as in
            # probe_multiblock.sbatch (review 2026-10-11). ACCEPT_LEGACY=1 is the explicit opt-in.
            if [ "${ACCEPT_LEGACY:-0}" = 1 ]; then
                echo "WARNING: k=$K: ${done_glob[0]} predates recorded configs (2026-10-08), unverifiable;" \
                     "ACCEPT_LEGACY=1, counted done (caches NOT freed)"; exit 0
            fi
            echo "FATAL: k=$K: ${done_glob[0]} predates recorded configs (2026-10-08) and cannot be verified" \
                 "-- move it aside to re-probe, or set ACCEPT_LEGACY=1 to count it done" >&2; exit 1 ;;
        5) echo "FATAL: k=$K: ${done_glob[0]} was not computed on the extraction in $DIR (see above) --" \
               "set OUT_DIR for this sweep, or move it aside" >&2; exit 1 ;;
        *) echo "FATAL: k=$K: done-status itself failed (exit $rc) -- NOT a verdict; investigate," \
               "do not move the result" >&2; exit 1 ;;
    esac
fi

# Same completeness rule as sweep_block.sh: npz AND its _meta.json.
missing=()
for split in $SPLITS; do
    ok=0
    for f in "$DIR"/*/multistep_"$split"_feats_oneshot_g1.0.npz; do
        [ -f "$f" ] && [ -f "${f%.npz}_meta.json" ] && ok=1
    done
    [ "$ok" = 1 ] || missing+=("$split")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "FATAL: k=$K: incomplete caches in $DIR for: ${missing[*]} -- rerun this block on GPU (sweep.sbatch)" >&2
    exit 1
fi
echo "k=$K: all caches complete in $DIR ($SPLITS)"
[ "${PROBE_CPU_GUARD_ONLY:-0}" = 1 ] && { echo "guard-only: stopping before probe"; exit 0; }

bash "$PROJECT_ROOT/experiments/isaac/sweep/sweep_block.sh" "$K"
