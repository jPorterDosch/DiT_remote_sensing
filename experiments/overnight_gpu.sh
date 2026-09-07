#!/bin/bash
# Overnight GPU batch, 2026-08-21. Sequential; each stage is independently restartable via
# sentinel. Ordered by information-per-hour:
#   1. roundtrip OOD + matched-NFE integrator ablation, EuroSAT (n=200 + 10)
#   2. same, RESISC45  -- together these give the first direct in-distribution measure
#      (section 7 prediction 2) and close audit finding O3
#   3. FIXED-CONDITIONING control, both datasets, n=5000 -- the control audit finding F6
#      calls "cheap and unrun": feed x_t while pinning the timestep conditioning at t=100.
#      Separates "protocol destroyed information" from "t-conditioned features are worse at
#      high t". The one-shot decay curve of these caches vs the normal ens1 caches is the
#      answer.
#   4. block sweep, n=500, one-shot ens1, k in {19, 24, 33, 38, 48} (28 exists) -- audit
#      finding A8: block 28 has no justification artifact; measures sensitivity.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AE=ditf_models/FLUX.1-dev/ae.safetensors
export FLUX_DEV=ditf_models/FLUX.1-dev/flux1-dev.safetensors
export WANDB_MODE=offline
T="100 180 260 340 420 500 580"
ES=data/eurosat/EuroSAT_RGB
R45=data/resisc45/NWPU-RESISC45

stage () { # stage <name> <cmd...>
    name=$1; shift
    if [ -f "logs/.done_$name" ]; then echo "SKIP $name"; return 0; fi
    echo "=== $name start $(date -Is) ==="
    "$@" > "logs/$name.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && touch "logs/.done_$name" || echo "FAILED $name rc=$rc"
    echo "=== $name end $(date -Is) rc=$rc ==="
}

mkdir -p results/roundtrip
stage rt_eurosat  python -u experiments/roundtrip_ood.py --dataset eurosat  --path "$ES"  --img-size 224 224 --n 200 --t-stop 580 --num-steps 50 --nfe-ablation --out results/roundtrip/eurosat_t580.json
stage rt_resisc45 python -u experiments/roundtrip_ood.py --dataset resisc45 --path "$R45" --img-size 256 256 --n 200 --t-stop 580 --num-steps 50 --nfe-ablation --out results/roundtrip/resisc45_t580.json

fixedcond () { # fixedcond <dataset> <path> <H> <W>
    ds=$1; path=$2; h=$3; w=$4
    FIXED_COND_T=100 python -u run.py --task extract --model.name flux --model.ensemble-size 1 \
        --dataset.name "$ds" --dataset.path "$path" --img-size "$h" "$w" --t $T \
        --extraction-mode ONESHOT --subset-size 5000 --subset-seed 42 --eps-seed 42 \
        --k 28 --guidance-scale 1.0 --save-dir "models/n5000_${ds}_fixedcond100"
}
stage fixedcond_eurosat  fixedcond eurosat  "$ES"  224 224
stage fixedcond_resisc45 fixedcond resisc45 "$R45" 256 256

blocksweep () { # blocksweep <dataset> <path> <H> <W> <k>
    ds=$1; path=$2; h=$3; w=$4; k=$5
    python -u run.py --task extract --model.name flux --model.ensemble-size 1 \
        --dataset.name "$ds" --dataset.path "$path" --img-size "$h" "$w" --t $T \
        --extraction-mode ONESHOT --subset-size 500 --subset-seed 42 --eps-seed 42 \
        --k "$k" --guidance-scale 1.0 --save-dir "models/blocksweep_${ds}_k${k}"
}
for k in 19 24 33 38 48; do
    stage "bs_eurosat_k${k}"  blocksweep eurosat  "$ES"  224 224 "$k"
    stage "bs_resisc45_k${k}" blocksweep resisc45 "$R45" 256 256 "$k"
done
echo "OVERNIGHT ALL DONE $(date -Is)"
