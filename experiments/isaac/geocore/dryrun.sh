#!/bin/bash
# Flag validation for extract.sbatch without a GPU or weights (CLAUDE.md "validate before
# submitting"): a stub python3 echoes run.py's argv; every protocol flag must appear.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
STUB="$(mktemp -d)"
cat > "$STUB/python3" <<'PY'
#!/bin/bash
# stub: pretend every python call succeeds; echo run.py invocations; answer the two helper queries
case "$*" in
  *"official_spec"*) echo "data/m_eurosat_rgb train,val,test" ;;
  *"PINNED_INDEX_SHA256"*) echo "6af0afb014d208f28e04804cb0bbe8bdf3b52a8490cf3610e001b85c17fa8b65 4-8-12-16-20-24-28-31" ;;
  "run.py "*) echo "RUNPY $*" ;;
  *) : ;;
esac
PY
chmod +x "$STUB/python3"
cat > "$STUB/nvidia-smi" <<'PY'
#!/bin/bash
echo "stub-gpu, 49140 MiB, 0"
PY
chmod +x "$STUB/nvidia-smi"
FAKE_STORE="$(mktemp -d)"
mkdir -p "$FAKE_STORE/GeoCore-9B/vae" "$FAKE_STORE/models" "$ROOT/data/m_eurosat_rgb/train" 2>/dev/null || true
: > "$FAKE_STORE/GeoCore-9B/model.safetensors.index.json"; : > "$FAKE_STORE/GeoCore-9B/vae/diffusion_pytorch_model.safetensors"
# HOME is redirected so the job script's `export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"` prepends a
# directory that does not exist and the stub python3 stays first on PATH (a real python here would
# import torch and start an extraction on the login node -- the exact thing CLAUDE.md forbids).
FAKE_HOME="$(mktemp -d)"
out="$(cd "$ROOT" && HOME="$FAKE_HOME" PATH="$STUB:$PATH" SLURM_SUBMIT_DIR="$ROOT" SLURM_ARRAY_TASK_ID=0 SMOKE=1 RES=224 \
    CKPT_DIR="$FAKE_STORE/GeoCore-9B" DATA="$ROOT/data/m_eurosat_rgb" \
    timeout 60 bash experiments/isaac/geocore/extract.sbatch 2>&1 || true)"
rm -rf "$FAKE_HOME"
echo "$out" | grep -qE "^RUNPY" || { echo "DRYRUN_FAIL: no stubbed run.py call seen (did a real python3 run?)"; echo "$out" | tail -15; rm -rf "$STUB" "$FAKE_STORE"; exit 1; }
echo "$out" | grep -E "RUNPY" | head -3
rc=0
for flag in "--model.name geocore" "--model.ensemble-size 2" "--geocore-ckpt $FAKE_STORE/GeoCore-9B" "--geocore-vae $FAKE_STORE/GeoCore-9B/vae/diffusion_pytorch_model.safetensors" \
            "--img-size 224 224" "--t 100 180 260 340 420 500 580 740 900" "--k 4 8 12 16 20 24 28 31" "--guidance-scale 0.0" \
            "--extraction-mode ONESHOT" "--subset-size 24" "--extract-split train" "--seed 42 --eps-seed 42" "--save-dir models/geocore_eval_smoke/m_eurosat"; do
    echo "$out" | grep -qF -- "$flag" || { echo "MISSING: $flag"; rc=1; }
done
# The stub writes no caches, so every unit ends at its own "no complete ... cache" check (the OK
# lines need real caches; smoke.sbatch exercises that path for real).
echo "$out" | grep -q "FATAL: no complete train cache" || { echo "MISSING: the post-extraction cache check did not fire"; rc=1; }

# Campus-gpu units (24 h QOS wall): a train shard, the val+test unit, the CPU stamp unit.
run_unit() {  # run_unit <env assignments...> -> captured output
    (cd "$ROOT" && env HOME="$FAKE_HOME" PATH="$STUB:$PATH" SLURM_SUBMIT_DIR="$ROOT" SLURM_ARRAY_TASK_ID=2 RES=224 \
        CKPT_DIR="$FAKE_STORE/GeoCore-9B" DATA="$ROOT/data/m_eurosat_rgb" "$@" timeout 60 bash experiments/isaac/geocore/extract.sbatch 2>&1 || true)
}
FAKE_HOME="$(mktemp -d)"
u="$(run_unit UNIT=train:1 TRAIN_SHARDS=3)"
for flag in "--num-shards 3 --shard-index 1" "--seed 101 --eps-seed 101" "--extract-split train" "--save-dir models/geocore_eval/m_pv4ger" "--model.ensemble-size 8"; do
    echo "$u" | grep -qF -- "$flag" || { echo "MISSING (UNIT=train:1): $flag"; rc=1; }
done
[ "$(echo "$u" | grep -c '^RUNPY')" = 1 ] || { echo "UNIT=train:1 must run exactly one extraction"; rc=1; }
echo "$u" | grep -q "FATAL: no complete train shard 1 cache" || { echo "MISSING (UNIT=train:1): post-extraction shard check"; rc=1; }
u="$(run_unit UNIT=valtest)"
echo "$u" | grep -q -- "--extract-split val .*--seed 44 --eps-seed 44\|--seed 44 --eps-seed 44 .*--extract-split val" || { echo "MISSING (UNIT=valtest): val with seed 44"; rc=1; }
echo "$u" | grep -q -- "--num-shards" && { echo "UNIT=valtest must not shard"; rc=1; }
echo "$u" | grep -q "FATAL: no complete val cache" || { echo "MISSING (UNIT=valtest): post-extraction val check"; rc=1; }
u="$(run_unit UNIT=stamp TRAIN_SHARDS=3)"
echo "$u" | grep -q '^RUNPY' && { echo "UNIT=stamp must not extract"; rc=1; }
echo "$u" | grep -q "FATAL: no complete train cache" || { echo "MISSING (UNIT=stamp): train cache check"; rc=1; }
u="$(run_unit UNIT=train:3 TRAIN_SHARDS=3)"
echo "$u" | grep -q "FATAL: UNIT=train:3 not a shard index" || { echo "MISSING: shard index bound check"; rc=1; }
u="$(run_unit UNIT=train:0 TRAIN_SHARDS=2 SMOKE=1)"
echo "$u" | grep -q "FATAL: the anchor / smoke extractions are unsharded" || { echo "MISSING: smoke+shards refusal"; rc=1; }
u="$(run_unit UNIT=train:0)"
echo "$u" | grep -q "needs TRAIN_SHARDS" || { echo "MISSING: TRAIN_SHARDS required for a train unit"; rc=1; }
rm -rf "$FAKE_HOME"
rm -rf "$STUB" "$FAKE_STORE"
[ "$rc" = 0 ] && echo "DRYRUN_OK" || { echo "DRYRUN_FAIL"; echo "$out" | tail -20; }
exit $rc
