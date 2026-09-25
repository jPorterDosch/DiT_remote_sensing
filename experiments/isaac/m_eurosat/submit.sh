#!/usr/bin/env bash
# ============================================================================
# Submit the full m-eurosat extraction campaign on ISAAC (ai-tenn allocation).
#   Prereq (once, login node): bash experiments/isaac/m_eurosat/prepare.sh
#
#   one-shot: three UNSHARDED jobs, val -> test -> train (cheap splits first so the
#             probe's inputs accumulate early). val/test get a 4h walltime override
#             so they backfill into small gaps; train keeps the 30h default.
#   inversion: one 10-task array (8 train shards + val + test); ai-tenn QoS allows 28
#             concurrent jobs and 56 queued, so all 13 jobs submit and run in one shot.
#
# After everything lands (README 'Evaluation pipeline'):
#   python3 -m eval.gates official_flux_m_eurosat   # new harness reproduces the banked probe
#   python3 -m eval.probe --protocol official --dataset m_eurosat --arm flux-oneshot-ens8 --kind flux \
#       --features 'models/m_eurosat_oneshot_ens8/*/multistep_{split}_feats_oneshot_g1.0.npz'
#   python3 -m eval.probe --protocol official --dataset m_eurosat --arm flux-inversion --kind flux \
#       --features 'models/m_eurosat_inversion/*/multistep_{split}_feats_inversion_g1.0_n50.npz'
# (CPU-only; run on a login node or any campus node. It merges the inversion shards,
#  verifies coverage/identity, selects on the official val split, and evaluates each
#  arm once on test.)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../../.."

[ -d data/m_eurosat_rgb/train ] || { echo "run experiments/isaac/m_eurosat/prepare.sh first" >&2; exit 1; }
mkdir -p logs

# Submission ORDER matters for backfill: the 30 h train job is the hardest to place,
# so it goes FIRST to enter the scheduler's reservation plan and start accruing age
# priority; the short 1-GPU jobs then backfill into gaps around it. (The old val->test
# ->train order was for the 3-concurrent campus-gpu QoS and is obsolete on ai-tenn.)
j_train=$(sbatch --parsable --export=ALL,SPLIT=train experiments/isaac/m_eurosat/oneshot.sbatch)
j_inv=$(sbatch --parsable experiments/isaac/m_eurosat/inversion.sbatch)
j_val=$(sbatch --parsable --time=04:00:00 --export=ALL,SPLIT=val   experiments/isaac/m_eurosat/oneshot.sbatch)
j_test=$(sbatch --parsable --time=04:00:00 --export=ALL,SPLIT=test experiments/isaac/m_eurosat/oneshot.sbatch)

echo "submitted: oneshot val=$j_val test=$j_test train=$j_train  inversion array=$j_inv"
echo "watch:     squeue -u \$USER"
echo "then:      python3 -m eval.probe --protocol official --dataset m_eurosat ...  (see header of this script)"
