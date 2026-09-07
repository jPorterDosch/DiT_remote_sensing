#!/bin/bash
# Waits until >=26 GB GPU memory is free (FLUX chain peaks ~23.5 GB), then runs the n=5000
# solver-curvature+states extraction for both datasets. Safe to leave running: checks every
# 10 min, never preempts anything.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AE=ditf_models/FLUX.1-dev/ae.safetensors FLUX_DEV=ditf_models/FLUX.1-dev/flux1-dev.safetensors WANDB_MODE=offline
# (A GPU-memory polling gate lived here while the card was shared with other jobs in
# Aug 2026; removed 2026-09-03 -- it was situational. The script now assumes the GPU is
# available and fails fast on OOM rather than waiting.)
python -u experiments/solver_curvature.py --dataset eurosat --path data/eurosat/EuroSAT_RGB --img-size 224 224 --subset-size 5000 --out-dir models/solver_curvature_n5000 > logs/curv5000_eurosat.log 2>&1 || { echo "FAILED eurosat rc=$?"; exit 1; }
python -u experiments/solver_curvature.py --dataset resisc45 --path data/resisc45/NWPU-RESISC45 --img-size 256 256 --subset-size 5000 --out-dir models/solver_curvature_n5000 > logs/curv5000_resisc45.log 2>&1 || { echo "FAILED resisc45 rc=$?"; exit 1; }
# The sentinel is only written when BOTH legs exit 0 -- previously it was unconditional, so
# a CUDA OOM within seconds still recorded "done" (2026-09-04 review).
echo done > logs/.done_curv5000
