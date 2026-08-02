#!/bin/bash
# ============================================================================
# Positive control — can the traj encoder read ORDERING at all?
#
#   The ordering sweep (ordering_traj_readout_inversion.sh) returned a null:
#   traj and shuffle scored within 0.04 accuracy points, and 24/25 folds were
#   EXACTLY tied. That null has two incompatible explanations:
#
#     (a) the denoising chain's ordering genuinely carries no class signal, or
#     (b) the readout cannot see ordering, so it could never have found any.
#
#   Nothing in the main sweep separates them. The pre-flight self-test only
#   proves the encoder is order-sensitive AT INIT on an untrained model — it
#   says nothing about what training did to `pos`.
#
#   This job decides it. --control builds a binary task from the same cache in
#   which every trajectory appears twice, once as-is and once time-reversed, so
#   each image sits in BOTH classes and direction of travel is the only signal
#   that generalizes. Then:
#
#     traj    should solve it — near 1.0 if positional encoding works.
#     shuffle MUST be at chance; a random permutation erases direction. Floor.
#     mlp     will likely beat chance, and that is EXPECTED, not a bug: it reads
#             a fixed SLOT, and reversal puts a different timestep in that slot.
#             Single-slot content leakage of direction, not a trajectory read.
#
#   If traj is at chance too, explanation (b) holds and the EuroSAT null is
#   uninformative. Each fold also logs what `pos` did during training (RMS at
#   init vs trained, size relative to input_proj output, post-training
#   permutation sensitivity) — that is what tells you WHY if it fails.
#
#   Output: one CSV row per (arm, norm, seed, fold) appended to $OUT_CSV.
#
# TIME ESTIMATE:
#   10 invocations (2 norms x 5 seeds), each = 3 arms x 5-fold CV x 200 epochs
#   on 1000 x 7 x 3072 (double the main sweep — every trajectory is duplicated).
#   ~3-6 min/invocation on an A6000-class GPU -> ~30-60 min total.
# ============================================================================
#SBATCH --job-name=direction-control-inv-g1.0
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=04:00:00
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
setup_environment   # loads cuda module; control uses its own --control-out-csv

# --- Locate the inversion cache (same resolution as the ordering sweep).
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
OUT_CSV="$PROJECT_ROOT/results/direction_control.csv"
echo "out-csv: $OUT_CSV   best-t: $BEST_T"

# --- Sweep: 2 norms x 5 seeds, all three arms per invocation.
fail=0
for norm in raw normalized; do
    for seed in 42 43 44 45 46; do
        echo "=== control norm=$norm seed=$seed ==="
        python3 "$PROJECT_ROOT/experiments/traj_readout.py" \
            --cache-path "$CACHE_PATH" \
            --control \
            --control-arms traj,shuffle,mlp \
            --norm "$norm" \
            --seed "$seed" \
            --best-t "$BEST_T" \
            --control-out-csv "$OUT_CSV" || {
                echo "WARN: control failed (norm=$norm seed=$seed) — continuing." >&2
                fail=1
            }
    done
done

echo "control done. results appended to $OUT_CSV"
[ "$fail" -eq 0 ] || echo "NOTE: one or more runs failed — grep the log for WARN." >&2
