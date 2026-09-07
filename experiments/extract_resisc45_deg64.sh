#!/bin/bash
# RESOLUTION-vs-DOMAIN control. RESISC45 degraded to 64px and upsampled back to 256, which
# reproduces EuroSAT's pipeline (64px native, BICUBIC up to 224) on a natively-256px dataset.
#
# WHY. The random-weight control found that an UNTRAINED FLUX reaches 84% of the trained
# model's above-chance accuracy on EuroSAT but only 57% on RESISC45 -- i.e. the learned weights
# barely matter on EuroSAT and matter a lot on RESISC45. Two explanations are perfectly
# confounded in our data: domain (EuroSAT is further out of FLUX's distribution) and effective
# resolution (EuroSAT is 3.5x upsampled, so it has far less high-frequency content for learned
# features to exploit). Measured: high-frequency energy fraction is 0.1139 native vs 0.0087
# degraded, a 13x reduction.
#
# THE TEST. Degrading RESISC45 holds dataset, content, subset, token grid and timesteps fixed
# and moves ONLY effective resolution. If RESISC45's untrained fraction jumps toward EuroSAT's
# 84%, resolution explains the dissociation. If it stays near 57%, the content genuinely needs
# learned features and the difference is not a resolution artifact.
#
# This is the first experiment in the project able to BREAK the resolution/domain collinearity
# rather than document it (cf. the RESOLUTION CONFOUND block in RESEARCH_NOTES.md section 4).
# OUTCOME (RESEARCH_NOTES 6d): untrained fraction UNCHANGED under degradation (56.6% -> 54.9%)
# -- resolution eliminated as the dissociation's cause; trained model loses only 2.6 points
# despite 13x less high-frequency energy.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export AE=ditf_models/FLUX.1-dev/ae.safetensors
export FLUX_DEV=ditf_models/FLUX.1-dev/flux1-dev.safetensors
export WANDB_MODE=offline
export DEGRADE_TO=64

T="100 180 260 340 420 500 580"
R45=data/resisc45/NWPU-RESISC45

echo "=== raw x_t (VAE only), degraded  $(date -Is) ==="
python -u experiments/raw_xt_baseline.py --dataset resisc45 --path "$R45" \
    --img-size 256 256 --subset-size 5000 --out-dir models/raw_xt_n5000_deg64 \
    > logs/deg64_rawxt.log 2>&1
echo "  rc=$?"

for tag in trained random; do
    if [ -f "logs/.done_deg64_${tag}" ]; then echo "SKIP ${tag}"; continue; fi
    echo "=== DiT one-shot ens1 [${tag}] , degraded  $(date -Is) ==="
    if [ "$tag" = "random" ]; then export FLUX_RANDOM_INIT=1; else unset FLUX_RANDOM_INIT; fi
    # shellcheck disable=SC2086
    python -u run.py --task extract --model.name flux --model.ensemble-size 1 \
        --dataset.name resisc45 --dataset.path "$R45" \
        --img-size 256 256 --t $T --extraction-mode ONESHOT \
        --subset-size 5000 --subset-seed 42 --eps-seed 42 \
        --k 28 --guidance-scale 1.0 \
        --save-dir "models/n5000_resisc45_deg64_${tag}" > "logs/deg64_${tag}.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && touch "logs/.done_deg64_${tag}" || echo "  FAILED rc=$rc"
    echo "  rc=$rc  $(date -Is)"
done
unset FLUX_RANDOM_INIT
echo "DEG64 ALL DONE $(date -Is)"
