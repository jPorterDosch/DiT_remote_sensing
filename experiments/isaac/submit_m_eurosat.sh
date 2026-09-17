#!/usr/bin/env bash
# ============================================================================
# Submit the full m-eurosat extraction campaign on ISAAC (default campus account).
#   Prereq (once, login node): bash experiments/isaac/prepare_m_eurosat.sh
#
#   one-shot: three UNSHARDED jobs, val -> test -> train (cheap splits first so the
#             probe's inputs accumulate early; train is the 22.5h job).
#   inversion: one 10-task array (8 train shards + val + test).
#
# After everything lands:  python3 experiments/m_eurosat_probe.py
# (CPU-only; run on a login node or any campus node. It merges the inversion shards,
#  verifies coverage/identity, selects on the official val split, and evaluates each
#  arm once on test.)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

[ -d data/m_eurosat_rgb/train ] || { echo "run experiments/isaac/prepare_m_eurosat.sh first" >&2; exit 1; }
mkdir -p logs

j_val=$(sbatch --parsable --export=ALL,SPLIT=val   experiments/isaac/extract_m_eurosat_oneshot.sbatch)
j_test=$(sbatch --parsable --export=ALL,SPLIT=test experiments/isaac/extract_m_eurosat_oneshot.sbatch)
j_train=$(sbatch --parsable --export=ALL,SPLIT=train experiments/isaac/extract_m_eurosat_oneshot.sbatch)
j_inv=$(sbatch --parsable experiments/isaac/extract_m_eurosat_inversion.sbatch)

echo "submitted: oneshot val=$j_val test=$j_test train=$j_train  inversion array=$j_inv"
echo "watch:     squeue -u \$USER"
echo "then:      python3 experiments/m_eurosat_probe.py"
