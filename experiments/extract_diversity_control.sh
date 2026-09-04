#!/bin/bash
# DIVERSITY-MATCHED CONTROL for the trajectory-as-set question.
#   trajectory set:  7 timesteps x 1 eps draw   (the existing ens1 cache: 7 NFE)
#   control set:     1 timestep  x 7 eps draws  (this script: 7 NFE, same dims)
# If trajectory-concat beats draw-concat at matched NFE and dimensionality, the multi-t
# structure carries signal beyond generic view diversity -- the surviving positive form of
# the trajectory hypothesis. Member 1 is the existing eps_seed=42 ens1 cache; this extracts
# members 2-7 (eps_seed 43..48) at t=100 only (the best single t everywhere at n=5000).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AE=ditf_models/FLUX.1-dev/ae.safetensors FLUX_DEV=ditf_models/FLUX.1-dev/flux1-dev.safetensors WANDB_MODE=offline
for seed in 43 44 45 46 47 48; do
  for spec in "eurosat data/eurosat/EuroSAT_RGB 224" "resisc45 data/resisc45/NWPU-RESISC45 256"; do
    set -- $spec; ds=$1; path=$2; sz=$3
    tag="divctl_${ds}_eps${seed}"
    [ -f "logs/.done_${tag}" ] && { echo "SKIP $tag"; continue; }
    echo "=== $tag $(date -Is) ==="
    python -u run.py --task extract --model.name flux --model.ensemble-size 1 \
      --dataset.name "$ds" --dataset.path "$path" --img-size "$sz" "$sz" --t 100 580 \
      --extraction-mode ONESHOT --subset-size 5000 --subset-seed 42 --eps-seed "$seed" \
      --k 28 --guidance-scale 1.0 --save-dir "models/divctl_${ds}_eps${seed}" \
      > "logs/${tag}.log" 2>&1 && touch "logs/.done_${tag}" || echo "FAILED $tag"
  done
done
echo "DIVCTL DONE $(date -Is)"
