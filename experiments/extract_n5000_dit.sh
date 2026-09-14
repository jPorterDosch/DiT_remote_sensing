#!/bin/bash
# Matched-n DiT feature extraction at n=5000, to resolve the n-mismatch (H1) between the
# raw-x_t protocol baseline (n=5000) and the DiT arms (n=500). Every quantitative claim in
# RESEARCH_NOTES.md section 6 compares a protocol decay measured at n=5000 against a DiT
# decay measured at n=500; that is not a decomposition until both sit at the same n.
#
# Order is by information-per-hour, so partial completion is still usable:
#   1-2  one-shot ens=1   -- the arm DIRECTLY matched to raw x_t ens1. Settles the core
#                           "is the one-shot decay a protocol artifact" question.
#   3-4  inversion        -- the eps-free arm; with 1-2 this gives the inversion-vs-one-shot
#                           split at matched n. One run serves every ensemble size (no eps).
#   5-6  one-shot ens=8   -- the headline arm; also the feature-level positive control.
#
# guidance-scale 1.0 is NOT the run.py default (3.5) but IS what every existing paired cache
# used -- see models/paired_500_*/**/*_g1.0.npz. Omitting it silently produces a cache that
# cannot be paired with the n=500 arms.
# OUTCOME: superseded by extract_n5000_dit_v2.sh after legs 1-2 (driver reordered mid-queue
# to insert the random-weight control); kept for provenance of legs 1-3.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export AE=ditf_models/FLUX.1-dev/ae.safetensors
export FLUX_DEV=ditf_models/FLUX.1-dev/flux1-dev.safetensors
export WANDB_MODE=offline

T="100 180 260 340 420 500 580"

run () {  # run <tag> <dataset> <path> <H> <W> <extra flags...>
    tag=$1; ds=$2; path=$3; h=$4; w=$5; shift 5
    log="logs/n5000_${tag}.log"
    if [ -f "logs/.done_${tag}" ]; then echo "SKIP ${tag} (already done)"; return 0; fi
    echo "=== ${tag} start $(date -Is) ==="
    # shellcheck disable=SC2086
    python -u run.py --task extract --model.name flux \
        --dataset.name "$ds" --dataset.path "$path" \
        --img-size "$h" "$w" --t $T \
        --subset-size 5000 --subset-seed 42 --eps-seed 42 \
        --k 28 --guidance-scale 1.0 \
        --save-dir "models/n5000_${tag}" "$@" > "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then touch "logs/.done_${tag}"; else echo "FAILED ${tag} rc=$rc"; fi
    echo "=== ${tag} end $(date -Is) rc=$rc ==="
}

ES=data/eurosat/EuroSAT_RGB
R45=data/resisc45/NWPU-RESISC45

run eurosat_oneshot_ens1  eurosat  "$ES"  224 224 --extraction-mode ONESHOT   --model.ensemble-size 1
run resisc45_oneshot_ens1 resisc45 "$R45" 256 256 --extraction-mode ONESHOT   --model.ensemble-size 1
run eurosat_inversion     eurosat  "$ES"  224 224 --extraction-mode INVERSION --model.ensemble-size 1 --num-inversion-steps 50
run resisc45_inversion    resisc45 "$R45" 256 256 --extraction-mode INVERSION --model.ensemble-size 1 --num-inversion-steps 50
run eurosat_oneshot_ens8  eurosat  "$ES"  224 224 --extraction-mode ONESHOT   --model.ensemble-size 8
run resisc45_oneshot_ens8 resisc45 "$R45" 256 256 --extraction-mode ONESHOT   --model.ensemble-size 8

echo "ALL DONE $(date -Is)"
