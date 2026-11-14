#!/bin/bash
# DiffusionSat geo runs: score the declared gsd-limited located subset (CPU, minutes).
# FID / KID of sample i vs real image i for the located rows whose tile gsd <= GSD_MAX metres
# (ledger 2026-11-12: declared before any result, GSD_MAX=10, EXPECT_N=2886). Sensitivity number
# only: n differs from the n = 5,000 table rows and FID is biased in n, so compare KID across n.
#   RUNS_DIRS="geo_v2_s100_g1.0_n5000 geo_v2_s20_g7.5_n5000" sbatch experiments/isaac/baselines/dsat_subset_score.sbatch
# Env: RUNS_DIRS (run dir names under $S/baselines/dsat_samples, space separated; required),
#      GSD_MAX (10), EXPECT_N (2886), TEST (1 = run tests/test_dsat_cond.py + tests/test_gen_metrics.py first).
#SBATCH --job-name=dsat-subset-score
#SBATCH -A isaac-utk0256
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
S=/lustre/isaac24/scratch/jdosch1
DITF_PY=${DITF_PY:-$HOME/.conda/envs/DiTF/bin/python}
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
: "${RUNS_DIRS:?set RUNS_DIRS, e.g. RUNS_DIRS=\"geo_v2_s100_g1.0_n5000\"}"
GSD_MAX=${GSD_MAX:-10}
EXPECT_N=${EXPECT_N:-2886}
for d in $RUNS_DIRS; do
  [ -e "$S/baselines/dsat_samples/$d/conditioning.csv" ] || { echo "missing input: $S/baselines/dsat_samples/$d/conditioning.csv"; exit 1; }
done
if [ "${TEST:-1}" = 1 ]; then
  "$DITF_PY" -m pytest -q -p no:cacheprovider -rfE tests/test_dsat_cond.py tests/test_gen_metrics.py
  echo "TESTS_OK tests/test_dsat_cond.py tests/test_gen_metrics.py"
fi
for d in $RUNS_DIRS; do
  out=$S/baselines/dsat_samples/$d
  tag=$("$DITF_PY" -c "print(f'gsdle{float(\"$GSD_MAX\"):g}')")
  "$DITF_PY" experiments/isaac/baselines/dsat_subset.py --run "$out" --gsd-max "$GSD_MAX" --expect-n "$EXPECT_N"
  js=$out/gen_metrics_geoheldout_$tag.json
  "$DITF_PY" -m eval.gen_metrics --real "$out/ref224_$tag" --fake "$out/samples224_$tag" --out "$js" > "${js%.json}.log" 2>&1 \
    || { echo "SCORE FAILED: $js"; tail -5 "${js%.json}.log"; exit 1; }
  echo "SCORED $js: $(grep -m1 '"n"' "$js") $(grep -m1 '"fid"' "$js") $(grep -m1 '"kid_mean"' "$js")"
done
echo "DSAT_SUBSET_JOB_OK runs=$RUNS_DIRS gsd_max=$GSD_MAX expect_n=$EXPECT_N"
