#!/bin/bash
# ============================================================================
# Ordering experiment — nonlinear trajectory readout on the INVERSION cache (the gate).
#   Runs experiments/traj_readout.py over the 500-image inversion cache:
#     arms  {traj, mlp, shuffle}  x  norms {raw, normalized}  x  seeds {0..4}
#   traj vs shuffle isolates whether ORDERING of the 7 chain states is readable;
#   mlp is the parameter-matched single-timestep content baseline. Ordering does its
#   OWN 5-fold stratified CV on the cache — it does not use a test cache.
#
#   Output: one CSV row per (arm, norm, seed, fold) appended to $OUT_CSV.
#
# GATE: the self-test (permutation-sensitivity) runs first and this script
#   aborts if it fails — a permutation-invariant encoder makes traj-vs-shuffle
#   meaningless, so no sweep runs until it passes.
#
# TIME ESTIMATE (--time set to 24h; tighten for faster backfill):
#   30 invocations (3 arms x 2 norms x 5 seeds), each = 5-fold CV x 200 epochs
#   of a 2-layer d=128 transformer on 500 x 7 x 3072 features. ~1-3 min/run on
#   an A6000-class GPU -> ~30-90 min total. A 04:00:00 request has ample headroom.
# ============================================================================
#SBATCH --job-name=ordering-traj-inv-g1.0
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=24:00:00                 # 24h request; est. ~30-90 min — see header
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# --- Bootstrap: find project root (same pattern as the other experiment scripts)
_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/experiments/_common.sh" || {
    echo "FATAL: Failed to source _common.sh" >&2; exit 1;
}
setup_environment   # loads cuda module; ordering uses its own --out-csv, not SAVE_DIR

# Off-ISAAC only: nothing to set — traj_readout.py is torch/sklearn only, no FLUX weights.

# --- Locate the inversion cache (run.py writes it under models/paired_500/<run_name>/).
# Override by exporting CACHE_PATH before sbatch if your cache lives elsewhere.
CACHE_GLOB="$PROJECT_ROOT/models/paired_500/*/multistep_train_feats_inversion_g1.0_n50.npz"
if [ -z "$CACHE_PATH" ]; then
    # shellcheck disable=SC2086
    matches=( $CACHE_GLOB )
    if [ ! -e "${matches[0]}" ]; then
        echo "FATAL: no inversion cache matched: $CACHE_GLOB" >&2
        echo "       run job 1 (extract_500_inversion_g1.0.sbatch) first, or export CACHE_PATH." >&2
        exit 1
    fi
    if [ "${#matches[@]}" -gt 1 ]; then
        echo "FATAL: ${#matches[@]} inversion caches matched $CACHE_GLOB — ambiguous:" >&2
        printf '       %s\n' "${matches[@]}" >&2
        echo "       export CACHE_PATH to pick one." >&2
        exit 1
    fi
    CACHE_PATH="${matches[0]}"
fi
echo "cache: $CACHE_PATH"

BEST_T="${BEST_T:-260}"

OUT_CSV="$PROJECT_ROOT/results/ordering_inversion_g1.0.csv"
echo "out-csv: $OUT_CSV   best-t: $BEST_T"

# --- GATE: permutation-sensitivity self-test must pass before any training.
python3 "$PROJECT_ROOT/experiments/traj_readout.py" --cache-path "$CACHE_PATH" --self-test || {
    echo "FATAL: traj_readout self-test failed — aborting sweep (see message above)." >&2
    exit 1
}

# --- Sweep: 3 arms x 2 norms x 5 seeds = 30 invocations, all appending to OUT_CSV.
fail=0
for arm in traj mlp shuffle; do
    for norm in raw normalized; do
        for seed in 42 43 44 45 46; do
            echo "=== arm=$arm norm=$norm seed=$seed ==="
            python3 "$PROJECT_ROOT/experiments/traj_readout.py" \
                --cache-path "$CACHE_PATH" \
                --arm "$arm" \
                --norm "$norm" \
                --seed "$seed" \
                --best-t "$BEST_T" \
                --out-csv "$OUT_CSV" || {
                    echo "WARN: run failed (arm=$arm norm=$norm seed=$seed) — continuing sweep." >&2
                    fail=1
                }
        done
    done
done

echo "sweep done. results appended to $OUT_CSV"
[ "$fail" -eq 0 ] || echo "NOTE: one or more runs failed — grep the log for WARN." >&2
