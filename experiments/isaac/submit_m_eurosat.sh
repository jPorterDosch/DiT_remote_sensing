#!/usr/bin/env bash
# ============================================================================
# Submit the full m-eurosat extraction campaign on ISAAC (default campus account).
#   Prereq (once, login node): bash experiments/isaac/prepare_m_eurosat.sh
#
#   one-shot: three UNSHARDED jobs, val -> test -> train (cheap splits first so the
#             probe's inputs accumulate early). val/test get a 4h walltime override
#             so they backfill into small gaps; train keeps the 30h default.
#   inversion: one 10-task array (8 train shards + val + test); ai-tenn QoS allows 28
#             concurrent jobs and 56 queued, so all 13 jobs submit and run in one shot.
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

# Submission ORDER matters for backfill: the 30 h train job is the hardest to place,
# so it goes FIRST to enter the scheduler's reservation plan and start accruing age
# priority; the short 1-GPU jobs then backfill into gaps around it. (The old val->test
# ->train order was for the 3-concurrent campus-gpu QoS and is obsolete on ai-tenn.)
j_train=$(sbatch --parsable --export=ALL,SPLIT=train experiments/isaac/extract_m_eurosat_oneshot.sbatch)
j_inv=$(sbatch --parsable experiments/isaac/extract_m_eurosat_inversion.sbatch)
j_val=$(sbatch --parsable --time=04:00:00 --export=ALL,SPLIT=val   experiments/isaac/extract_m_eurosat_oneshot.sbatch)
j_test=$(sbatch --parsable --time=04:00:00 --export=ALL,SPLIT=test experiments/isaac/extract_m_eurosat_oneshot.sbatch)

echo "submitted: oneshot val=$j_val test=$j_test train=$j_train  inversion array=$j_inv"
echo "watch:     squeue -u \$USER"
echo "then:      python3 experiments/m_eurosat_probe.py"
