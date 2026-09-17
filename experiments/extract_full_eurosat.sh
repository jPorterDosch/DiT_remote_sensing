#!/usr/bin/env bash
# Full-split EuroSAT extraction (frozen FLUX, one-shot ens8, t=100+180, block 28, g=1.0)
# followed by the full-split linear probe (experiments/full_split_probe.py).
#   train: 21,600 images  (~9 h at the measured ~1.5 s/img for 2 timesteps at ens8)
#   test:   5,400 images  (~2.2 h)
# t=100 is the requested operating point; t=180 rides along because it is the a-priori
# best-t for EuroSAT ens8 and the anchor of the 6r learning-curve prediction (0.9765).
# Both are reported; nothing selects between them on test (rule 1).
set -u

cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

# Workstation weights (ISAAC paths are the defaults inside util.py).
if [ -f "ditf_models/FLUX.1-dev/flux1-dev.safetensors" ]; then
    export FLUX_DEV="ditf_models/FLUX.1-dev/flux1-dev.safetensors"
    export AE="ditf_models/FLUX.1-dev/ae.safetensors"
    echo "Using local FLUX weights: $FLUX_DEV"
fi

SAVE="models/full_eurosat_oneshot_ens8"
COMMON=(--task extract --model.name flux --model.ensemble-size 8
        --dataset.name eurosat --dataset.path data/eurosat/EuroSAT_RGB
        --img-size 224 224 --t 100 180 --extraction-mode ONESHOT
        --subset-seed 42 --k 28 --guidance-scale 1.0
        --save-dir "$SAVE")
# NOTE: no --subset-size => the FULL split (extraction.py falls through to arange).

run_split () {  # run_split <tag> <extra flags...>
    tag=$1; shift
    if [ -f "logs/.done_full_eurosat_${tag}" ]; then echo "SKIP ${tag} (already done)"; return 0; fi
    echo "=== full_eurosat_${tag} start $(date -Is) ==="
    python -u run.py "${COMMON[@]}" "$@" > "logs/full_eurosat_${tag}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then touch "logs/.done_full_eurosat_${tag}"; else echo "FAILED ${tag} rc=$rc"; fi
    echo "=== full_eurosat_${tag} end $(date -Is) rc=$rc ==="
    return $rc
}

# Seeds DIFFER by design between the splits (2026-09-16 adversarial audit, F1): with a
# shared eps_seed and unshuffled loaders, test image #k would reuse train image #k's
# exact eps and VAE-posterior draws (position-paired; all 600 AnnualCrop test images
# pair same-class), a structural optimistic coupling. The probe REFUSES equal eps
# seeds. --seed also differs so the global-RNG posterior stream decouples too.
run_split train --eps-seed 42                                || exit 1
run_split test  --extract-split test --eps-seed 43 --seed 43 || exit 1

echo "=== full-split probe $(date -Is) ==="
python -u experiments/full_split_probe.py 2>&1 | tee logs/full_split_probe_eurosat.log
