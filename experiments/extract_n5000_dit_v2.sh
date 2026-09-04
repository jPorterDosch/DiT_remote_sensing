#!/bin/bash
# Continuation of experiments/extract_n5000_dit.sh with the RANDOM-WEIGHT CONTROL inserted
# ahead of the ens8 legs. The original driver was stopped (its bash only -- the in-flight
# eurosat_inversion python was left running) so the queue could be reordered without losing
# 13+ hours of chain integration.
#
# WHY THE RANDOM-WEIGHT LEG JUMPED THE QUEUE. Every "DiT beats the raw x_t baseline" number
# is currently unattributed: by the data-processing inequality DiT(x_t) cannot carry more
# label information than x_t does, so the level gap (EuroSAT 0.956 vs 0.737 at t=100) is
# equally consistent with "3072 nonlinear projections beat 16 raw latent dims". An untrained
# network of identical capacity is the only control that separates those. It is also cheap
# (~1.5 h/dataset vs ~12 h for an ens8 leg), so it is strictly better value per GPU-hour.
#
# GUARD. A randomly-initialised 57-block transformer in bf16 can produce degenerate features
# (NaN/inf from activation blow-up, or near-zero across-image variance if it collapses). Each
# random leg therefore runs a 30-image probe FIRST and is skipped if the output is not finite
# and varying -- otherwise a dead 1.5 h run looks like a scientific null.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export AE=ditf_models/FLUX.1-dev/ae.safetensors
export FLUX_DEV=ditf_models/FLUX.1-dev/flux1-dev.safetensors
export WANDB_MODE=offline

T="100 180 260 340 420 500 580"
INFLIGHT_PID="${INFLIGHT_PID:-}"

run () {  # run <tag> <dataset> <path> <H> <W> <extra flags...>
    tag=$1; ds=$2; path=$3; h=$4; w=$5; shift 5
    if [ -f "logs/.done_${tag}" ]; then echo "SKIP ${tag} (already done)"; return 0; fi
    echo "=== ${tag} start $(date -Is) ==="
    # shellcheck disable=SC2086
    python -u run.py --task extract --model.name flux \
        --dataset.name "$ds" --dataset.path "$path" \
        --img-size "$h" "$w" --t $T \
        --subset-size 5000 --subset-seed 42 --eps-seed 42 \
        --k 28 --guidance-scale 1.0 \
        --save-dir "models/n5000_${tag}" "$@" > "logs/n5000_${tag}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then touch "logs/.done_${tag}"; else echo "FAILED ${tag} rc=$rc"; fi
    echo "=== ${tag} end $(date -Is) rc=$rc ==="
}

guard_random () {  # guard_random <dataset> <path> <H> <W>  -> 30-image finite/variance check
    ds=$1; path=$2; h=$3; w=$4
    tmp=$(mktemp -d)
    # shellcheck disable=SC2086
    FLUX_RANDOM_INIT=1 python -u run.py --task extract --model.name flux \
        --model.ensemble-size 1 --dataset.name "$ds" --dataset.path "$path" \
        --img-size "$h" "$w" --t $T --extraction-mode ONESHOT \
        --subset-size 30 --subset-seed 42 --eps-seed 42 \
        --k 28 --guidance-scale 1.0 --save-dir "$tmp" > "logs/guard_${ds}.log" 2>&1
    python - "$tmp" <<'PY'
import glob, sys
import numpy as np
hits = glob.glob(sys.argv[1] + "/**/*.npz", recursive=True)
if not hits:
    print("GUARD FAIL: no cache written"); sys.exit(1)
f = np.load(hits[0])["feats"]
if not np.isfinite(f).all():
    print("GUARD FAIL: non-finite features"); sys.exit(1)
sd = f.std(axis=0).mean()
print(f"guard ok: shape={f.shape} finite=True mean-across-image-std={sd:.6g}")
sys.exit(0 if sd > 1e-6 else 1)
PY
    rc=$?
    rm -rf "$tmp"
    return $rc
}

run_random () {  # run_random <tag> <dataset> <path> <H> <W>
    tag=$1; ds=$2; path=$3; h=$4; w=$5
    if [ -f "logs/.done_${tag}" ]; then echo "SKIP ${tag} (already done)"; return 0; fi
    echo "--- guard ${tag} $(date -Is)"
    if ! guard_random "$ds" "$path" "$h" "$w"; then
        echo "SKIPPING ${tag}: random-init guard failed (see logs/guard_${ds}.log)"
        return 0
    fi
    # `VAR=1 func` does NOT scope VAR to the call for a shell FUNCTION in bash -- the
    # assignment persists in the shell afterwards. Left as a prefix, FLUX_RANDOM_INIT
    # would still be set when the ens8 legs ran and would silently extract those from an
    # untrained network. Export/unset explicitly instead.
    export FLUX_RANDOM_INIT=1
    run "$tag" "$ds" "$path" "$h" "$w" --extraction-mode ONESHOT --model.ensemble-size 1
    unset FLUX_RANDOM_INIT
}

ES=data/eurosat/EuroSAT_RGB
R45=data/resisc45/NWPU-RESISC45

# --- Adopt the in-flight eurosat_inversion run started by the v1 driver.
if [ -n "$INFLIGHT_PID" ]; then
    echo "=== waiting on in-flight PID ${INFLIGHT_PID} (eurosat_inversion) $(date -Is) ==="
    while kill -0 "$INFLIGHT_PID" 2>/dev/null; do sleep 60; done
    if compgen -G "models/n5000_eurosat_inversion/*/multistep_train_feats_inversion_*.npz" > /dev/null; then
        touch logs/.done_eurosat_inversion
        echo "=== eurosat_inversion cache present, marked done $(date -Is) ==="
    else
        echo "=== WARNING: eurosat_inversion produced no cache; leaving unmarked ==="
    fi
fi

run        resisc45_inversion    resisc45 "$R45" 256 256 --extraction-mode INVERSION --model.ensemble-size 1 --num-inversion-steps 50
run_random eurosat_randinit_ens1  eurosat  "$ES"  224 224
run_random resisc45_randinit_ens1 resisc45 "$R45" 256 256
run        eurosat_oneshot_ens8  eurosat  "$ES"  224 224 --extraction-mode ONESHOT --model.ensemble-size 8
run        resisc45_oneshot_ens8 resisc45 "$R45" 256 256 --extraction-mode ONESHOT --model.ensemble-size 8

echo "ALL DONE $(date -Is)"
