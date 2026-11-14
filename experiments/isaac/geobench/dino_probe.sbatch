#!/bin/bash
# ============================================================================
# DINOv3 baseline arm on every GEO-Bench task -- CPU half: GATE + PROBE on the features
# dino.sbatch extracted (no GPU; `long`, so hours of sklearn never idle an H100).
# Chained by submit_dino.sh: --dependency=afterok on the WHOLE dino.sbatch array (packed jobs
# read several datasets); one failed extraction task leaves them DependencyNeverSatisfied.
#   (1) the pre-registered control gate (tests/test_eval_gates.py::test_dino_control_geobench)
#       on the CACHED features (DINO_GATE_FEATURES): positive control must beat the shuffled-
#       label null by >= 0.10 and the null must sit at chance. Anything but "1 passed" stops.
#   (2) eval.probe --protocol official: features.DINO_OFFICIAL_VIEWS x C selected on val, one
#       test evaluation -> results/eval/ (SMOKE=1: results/eval/smoke_dino, not quotable).
# TIME: unmeasured for 7B widths (cls 4096 ... cls4mp 20480-d); m-bigearthnet's 43
# one-vs-rest fits per cell dominate (FLUX single-t + concat probing took 6-9 h there on
# 8 CPUs). 3 days on 16 CPUs leaves wide margin under the 6-day long cap.
# ============================================================================
#SBATCH --job-name=geobench-dino-probe
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --qos=long
#SBATCH --partition=long
#SBATCH --time=3-00:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # caches + models/data/logs on Lustre scratch

# PATH-only env (as probe_cpu.sbatch): no CUDA module on a CPU node.
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

PRESET="${PRESET:?set PRESET to an eval/extract_dino.py PRESETS key (dinov3_vit7b16_web, dinov3_vit7b16_sat, dinov3_vitl16_web)}"  # no braces: they would close the expansion
# DATASET_IDXS="0-1-2" (OFFICIAL registry indices) runs several datasets in ONE job, each gated
# on its own -- packs the sweep under QOS long's 18-submitted-jobs limit. Default: the array index.
# PASS iff pytest's final summary line reports exactly "1 passed" and no failed/error/skipped.
# ANSI colour codes are stripped first (real ISAAC logs carry them even when not a tty).
gate_passed() {
    local line
    line="$(sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -E '^=+ .* in [0-9.]+s' | tail -1)"
    [[ "$line" =~ (^|[^0-9])1\ passed ]] && ! [[ "$line" =~ failed|error|skipped ]]
}

one() {
    local IDX="$1"
    # set -e is OFF inside a function called as `one ... || ...`: every fallible step returns explicitly.
    KEY="$(python3 -c "import sys; from eval.features import OFFICIAL; print(list(OFFICIAL)[int(sys.argv[1])])" "$IDX")" || return 1
    SMOKE="${SMOKE:-0}"
    if [ "$SMOKE" = 1 ]; then
        FEATS="models/dino_feats_smoke/${PRESET}_${KEY}_{split}_smoke.npz"; OUT_DIR=results/eval/smoke_dino
    else
        FEATS="models/dino_feats/${PRESET}_${KEY}_{split}.npz"; OUT_DIR=results/eval
    fi
    # WIDE_C=1: official probe over protocols.C_GRID_WIDE (1e-6..1), the pre-registered DINO grid.
    # The 3-C results were moved to results/eval/superseded_3C (2026-10-11). The control gate is
    # unaffected (it is a control, not the selection).
    C_ARGS=()
    # default 1: the same grid as the FLUX multi-block arm (rule 8); WIDE_C=0 = legacy 3-C grid
    [ "${WIDE_C:-1}" = 1 ] && C_ARGS=(--c-grid wide)  # protocols.C_GRID_WIDE
    # HEAD=tsweight: the learned slot-weighting head (eval/fusion.py; slots = last-4-block CLS + mp),
    # the protocol matched to the FLUX multi-block tsweight arm (rule 8). Own arm name, no C grid.
    PROBE_ARM="$PRESET"
    if [ "${HEAD:-lr}" = tsweight ]; then C_ARGS=(--head tsweight); PROBE_ARM="${PRESET}-tsw"; fi
    echo "=== gate+probe $KEY $PRESET smoke=$SMOKE wide_c=${WIDE_C:-1} features=$FEATS host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} $(date -Is)"

    # Smoke caches are --max-images subsets: tell the probe their sizes (train,val,test order).
    SIZES_ARGS=()
    SMOKE_SIZES=""
    if [ "$SMOKE" = 1 ]; then
        SMOKE_SIZES="$(python3 - "$FEATS" <<'PY'
import sys, numpy as np
print(",".join(str(len(np.load(sys.argv[1].format(split=s))["labels"])) for s in ("train", "val", "test")))
PY
    )" || return 1
        SIZES_ARGS=(--smoke-sizes "$SMOKE_SIZES")
        echo "smoke sizes: $SMOKE_SIZES"
    fi

    GATE_LOG="$(mktemp)"
    local rc=0
    DINO_GATE_FEATURES="$FEATS" DINO_GATE_SMOKE_SIZES="$SMOKE_SIZES" WANDB_MODE=disabled \
        python3 -m pytest tests/test_eval_gates.py --gpu -s -rs -p no:cacheprovider --color=no \
        -k "dino_control_geobench and $KEY and $PRESET" >"$GATE_LOG" 2>&1 || rc=$?
    cat "$GATE_LOG"
    # PASS = pytest exit 0 AND exactly one test ran and passed (a substring "1 passed" would also
    # accept "11 passed" or "1 failed, 1 passed"; a skip -- missing inputs -- is not a pass).
    if [ "$rc" != 0 ] || ! gate_passed "$GATE_LOG"; then
        echo "FATAL: control gate did not PASS for $KEY $PRESET (pytest exit $rc)" >&2; return 1
    fi
    rm -f "$GATE_LOG"
    # PASS marker tied to the size + mtime of the feature files the gate just judged: matrix.sbatch
    # refuses a row that its spec lists under "requires_gate" without one (eval/gate_marker.py).
    # Full caches only: a smoke gate certifies nothing about the full features.
    if [ "$SMOKE" != 1 ]; then
        python3 -m eval.gate_marker write --gate dino_control --preset "$PRESET" --dataset "$KEY" \
            --features "$FEATS" --job "${SLURM_JOB_ID:-manual$$}" || return 1
    fi
    # GATE_ONLY=1: stop after the control gate. For a row that the label-efficiency matrix scores
    # (eval/matrix.py fits the same official probe as its 100 % cell), eval.probe below would fit
    # everything a second time.
    if [ "${GATE_ONLY:-0}" = 1 ]; then echo "DINO_GATE_PASS $KEY $PRESET (GATE_ONLY) $(date -Is)"; return 0; fi

    # One C grid per (dataset, arm) in OUT_DIR: a result on another grid under the same arm name
    # lets a table pick a grid after seeing test (rule 1; 3-C results -> results/eval/superseded_3C).
    [ "${HEAD:-lr}" = tsweight ] || python3 - "$OUT_DIR" "$KEY" "$PRESET" "${C_ARGS[1]:-}" <<'PY' || return 1
import glob, json, sys
from eval.probe import parse_c_grid
out, key, arm, grid = sys.argv[1:]
want = list(parse_c_grid(grid or None))
import numpy as np
for p in glob.glob(f"{out}/probe-official_{key}_{arm}_*.npz"):
    cfg = json.loads(str(np.load(p, allow_pickle=True)["config"]))
    got = cfg.get("protocol_constants", {}).get("C_grid")
    if got != want:
        sys.exit(f"FATAL: {p} records C grid {got}, this run uses {want} -- one grid per arm; move it aside")
PY
    python3 -m eval.probe --protocol official --dataset "$KEY" --arm "$PROBE_ARM" --kind dino \
        --features "$FEATS" --n-jobs "${SLURM_CPUS_PER_TASK:-8}" --out-dir "$OUT_DIR" "${SIZES_ARGS[@]}" "${C_ARGS[@]}" || return 1
    echo "=== gate+probe $KEY $PRESET done $(date -Is)"
}
FAILED=()
# Hyphen-separated (0-1-2-3-4): sbatch --export splits on commas, so DATASET_IDXS=0,1,2 would
# arrive as "0" (verified 2026-10-08, job 6348646). Commas still accepted for direct use.
IFS=',-' read -r -a IDXS <<<"${DATASET_IDXS:-${SLURM_ARRAY_TASK_ID:?run as an array job or set DATASET_IDXS}}"
for idx in "${IDXS[@]}"; do
    one "$idx" || FAILED+=("$idx")
done
[ "${#FAILED[@]}" = 0 ] || { echo "FATAL: datasets at registry indices ${FAILED[*]} failed" >&2; exit 1; }
echo "DINO_PROBE_JOB_OK $PRESET idxs=${IDXS[*]} gate_only=${GATE_ONLY:-0} $(date -Is)"
