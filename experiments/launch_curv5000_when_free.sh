#!/bin/bash
# Waits until >=26 GB GPU memory is free (FLUX chain peaks ~23.5 GB), then runs the n=5000
# solver-curvature+states extraction for both datasets. Safe to leave running: checks every
# 10 min, never preempts anything.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AE=ditf_models/FLUX.1-dev/ae.safetensors FLUX_DEV=ditf_models/FLUX.1-dev/flux1-dev.safetensors WANDB_MODE=offline
while :; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -ge 26000 ] && break
  echo "$(date -Is) waiting: ${free} MiB free (<26000)"; sleep 600
done
echo "$(date -Is) GPU free (${free} MiB) -- starting"
python -u experiments/solver_curvature.py --dataset eurosat --path data/eurosat/EuroSAT_RGB --img-size 224 224 --subset-size 5000 --out-dir models/solver_curvature_n5000 > logs/curv5000_eurosat.log 2>&1
python -u experiments/solver_curvature.py --dataset resisc45 --path data/resisc45/NWPU-RESISC45 --img-size 256 256 --subset-size 5000 --out-dir models/solver_curvature_n5000 > logs/curv5000_resisc45.log 2>&1
echo done > logs/.done_curv5000
