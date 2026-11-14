#!/bin/bash
#SBATCH -J dsat-sample
#SBATCH -A ISAAC-UTK0256
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -o logs/dsat-sample-%j.out
# DiffusionSat sampling + FID / KID scoring for the two-way FID table (OUTLINE §4.8).
# Env:
#   RUNS   space-separated mode:steps:guidance triples, run in order, e.g. "fmow:100:1.0 geo:100:1.0"
#          mode = null | geo (Git-10M held-out rows, tile coordinates) | fmow (fMoW-val rows, per-image
#          caption + metadata). Paper sampler = 100 DDIM steps, guidance 1.0; notebook = 20 steps, 7.5.
#   N      samples per run (5000)      SEED (0)      BATCH (8)
#   TAG    output dir suffix: $S/baselines/dsat_samples/<mode>_v2_s<steps>_g<guidance>_n<N>[_<TAG>]
#          (the sampler refuses an existing directory; "v2" = training-convention metadata encoding)
#   TEST   1: first run tests/test_dsat_cond.py (DiTF env); a failure stops the job
#   SMOKE_N  if set (e.g. 16): before the full pass, run every entry of RUNS end to end at this n
#          into ..._n<SMOKE_N>_smoke<jobid>; the full pass starts only if that stage exits clean
#   SCORE  1 (default): after each run, eval.gen_metrics against BOTH references at n = N
#          (gen_metrics_geoheldout.json, gen_metrics_fmowval.json); geo runs also score the located
#          subset against the matching real images (gen_metrics_geoheldout_located.json)
# Partition / GPU on the command line, e.g.
#   sbatch -p campus-gpu -q campus-gpu --gpus-per-task=v100s:1 -t 8:00:00 \
#     --export=ALL,RUNS="fmow:100:1.0" experiments/isaac/baselines/diffusionsat_sample.sbatch
set -euo pipefail
S=/lustre/isaac24/scratch/jdosch1
DSAT_PY=${DSAT_PY:-$S/conda_envs/diffusionsat/bin/python}
DITF_PY=${DITF_PY:-$HOME/.conda/envs/DiTF/bin/python}
export HF_HOME=$S/hf_cache TRANSFORMERS_OFFLINE=0
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CKPT=$S/baselines/diffusionsat_ckpt/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64
GREF=$S/DiT_remote_sensing/git10m/fid_ref_geoheldout_n5000
FREF=$S/DiT_remote_sensing/git10m/fid_ref_fmowval_n5000
FCOND=${FCOND:-$S/baselines/fmow_val_meta_n5000/cond.csv}
FMETA=$S/baselines/DiffusionSat/datasets/fmow-val-meta.csv
N=${N:-5000}
: "${RUNS:?set RUNS, e.g. RUNS=\"fmow:100:1.0 geo:100:1.0\"}"
for f in "$CKPT/model_index.json" "$GREF/rows.csv" "$GREF/coords.csv" "$FREF/rows.csv" "$FCOND" "$FMETA"; do
  [ -e "$f" ] || { echo "missing input: $f"; exit 1; }
done

if [ "${TEST:-0}" = 1 ]; then
  "$DITF_PY" -m pytest -q tests/test_dsat_cond.py
fi

score() {  # real-dir fake-dir out-json [n]
  local log=${3%.json}.log
  "$DITF_PY" -m eval.gen_metrics --real "$1" --fake "$2" ${4:+--n "$4"} --out "$3" > "$log" 2>&1 \
    || { echo "SCORE FAILED: $3"; tail -5 "$log"; exit 1; }
  echo "SCORED $3: $(grep -m1 '"n"' "$3") $(grep -m1 '"fid"' "$3") $(grep -m1 '"kid_mean"' "$3")"
}

run_all() {  # n tag
  local n=$1 tag=$2 run mode steps guidance out cond
  for run in $RUNS; do
    IFS=: read -r mode steps guidance <<< "$run"
    [ -n "$mode" ] && [ -n "$steps" ] && [ -n "$guidance" ] || { echo "bad RUNS entry: $run"; exit 1; }
    out=$S/baselines/dsat_samples/${mode}_v2_s${steps}_g${guidance}_n${n}${tag:+_$tag}
    case "$mode" in
      fmow) cond=(--rows "$FREF/rows.csv" --fmow-cond "$FCOND" --fmow-meta "$FMETA") ;;
      geo|null) cond=(--rows "$GREF/rows.csv" --coords "$GREF/coords.csv") ;;
      *) echo "bad mode in RUNS: $run"; exit 1 ;;
    esac
    echo "== RUN $run n=$n -> $out"
    PYTHONPATH=$S/baselines/DiffusionSat "$DSAT_PY" experiments/isaac/baselines/diffusionsat_sample.py --ckpt "$CKPT" \
      --mode "$mode" "${cond[@]}" --steps "$steps" --guidance "$guidance" --n "$n" --seed "${SEED:-0}" \
      --batch "${BATCH:-8}" --out "$out"
    if [ "${SCORE:-1}" = 1 ]; then
      score "$GREF" "$out/samples224" "$out/gen_metrics_geoheldout.json" "$n"
      score "$FREF" "$out/samples224" "$out/gen_metrics_fmowval.json" "$n"
      if [ "$mode" = geo ]; then
        score "$out/ref224_located" "$out/samples224_located" "$out/gen_metrics_geoheldout_located.json"
      fi
    fi
    echo "RUN_OK $run n=$n"
  done
}

if [ -n "${SMOKE_N:-}" ]; then
  echo "== SMOKE STAGE n=$SMOKE_N: every run in RUNS, same code path incl. scoring; a failure stops the job"
  run_all "$SMOKE_N" "smoke${SLURM_JOB_ID:-local}"
  echo "SMOKE_OK n=$SMOKE_N (numbers at this n are meaningless; do not quote)"
fi
run_all "$N" "${TAG:-}"
echo "DSAT_JOB_OK runs=$RUNS n=$N"
