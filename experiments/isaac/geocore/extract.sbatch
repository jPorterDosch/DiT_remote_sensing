#!/bin/bash
# GeoCore-9B frozen multi-timestep, multi-block feature extraction on GEO-Bench classification
# (run.py --task extract --model.name geocore), one array index per dataset, train/val/test
# splits, the FLUX grid: t = 100..900 (9), k = 4 8 12 16 20 24 28 31, ensemble 8, 224 px
# (spec docs/superpowers/specs/2026-11-02-geocore-eval-design.md).
#
# Throughput (smoke 6531445, A40, 224 px): 0.1 s per forward, i.e. 7.2 s per image at ensemble 8,
# so a train split is 13 h (m_forestnet) to 40 h (m_so2sat / m_bigearthnet). The campus-gpu QOS
# caps a job at 24 h (MaxWall 1-00:00:00; 3 running / 6 submitted per user), so on
# campus-gpu-bigmem the work is cut into UNITs that each fit one job:
#
#   UNIT=train:<i>  TRAIN_SHARDS=<S>   shard i of the train split (run.py --num-shards S --shard-index i,
#                                      seed 100+i per shard: the loader's per-shard-seed convention,
#                                      experiments/isaac/m_eurosat/inversion.sbatch; never 42/43/44)
#   UNIT=valtest                       val (seed 44) + test (seed 43), unsharded
#   UNIT=stamp      TRAIN_SHARDS=<S>   CPU: verifies every cache, stamps the discard channels, prints
#                                      GEOCORE_EXTRACT_OK (submit with -p short -q short --gpus-per-task=0)
#   UNIT=all (default)                 everything in one job, train unsharded (TRAIN_SHARDS=1) unless
#                                      TRAIN_SHARDS is given: smokes, the anchor, and 3-day ai-tenn jobs
#   FLUX_DISCARD_REF=<cache.npz>       FLUX multi-block train cache holding k=28 (default: the m_eurosat
#                                      ens8 cache). The discard rule must give [154, 1446] on it before a
#                                      stamp is written; missing or failing = the job stops.
#
# A dataset's shard count is the submitter's choice (scratch geocore_orchestrator.sh is the single
# source: 224 px -> 3 3 2 3 1 3 for the six datasets; 256 px -> one more shard for m_eurosat /
# m_so2sat); the stamp unit must be given the same TRAIN_SHARDS. eval.features.load_flux_split
# merges shards (disjoint, complete, per-shard seeds) for every consumer (sweep, matrix, probe).
#
#   sbatch --array=2 --export=ALL,UNIT=train:0,TRAIN_SHARDS=2 experiments/isaac/geocore/extract.sbatch
#   sbatch --array=2 --export=ALL,UNIT=valtest experiments/isaac/geocore/extract.sbatch
#   sbatch --array=2 --export=ALL,UNIT=stamp,TRAIN_SHARDS=2 -p short -q short --gpus-per-task=0 -t 2:00:00 \
#       experiments/isaac/geocore/extract.sbatch
#   sbatch --array=0 --export=ALL,ANCHOR=1 experiments/isaac/geocore/extract.sbatch    # ens-1 anchor, one job
#   sbatch --array=0,3 --export=ALL,RES=256,UNIT=... experiments/isaac/geocore/extract.sbatch   # resolution check
#   sbatch --array=0 --export=ALL,SMOKE=1 --time=1:30:00 experiments/isaac/geocore/extract.sbatch
#   ai-tenn (H100, 3-day QOS wall): sbatch --array=0-5 -p ai-tenn -q ai-tenn --time=3-00:00:00 \
#       experiments/isaac/geocore/extract.sbatch        (UNIT=all, unsharded, seed 42)
#
# Env: UNIT, TRAIN_SHARDS, RES (224|256), ANCHOR (0|1), SMOKE (0|1), CKPT_DIR (default
# $STORE/GeoCore-9B), DATA. Unit jobs print GEOCORE_UNIT_OK <dataset> <res> <unit>; the stamp /
# all unit prints GEOCORE_EXTRACT_OK <dataset> <res> and the matrix model entry to paste.
#

#SBATCH --job-name=geocore-extract
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # caches + models/data/logs on Lustre scratch
source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
python3 -c "import sys, numpy, torch, tyro, safetensors, transformers; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
set -euo pipefail

DATASETS=(m_eurosat m_brick_kiln m_pv4ger m_so2sat m_forestnet m_bigearthnet)
IDX="${SLURM_ARRAY_TASK_ID:?submit with --array=0-5 (one GEO-Bench task per array index)}"
[ "$IDX" -ge 0 ] && [ "$IDX" -lt "${#DATASETS[@]}" ] || { echo "FATAL: array index $IDX not in 0..5" >&2; exit 1; }
DATASET="${DATASETS[$IDX]}"
RES="${RES:-224}"
case "$RES" in 224|256) ;; *) echo "FATAL: RES must be 224 or 256, got $RES" >&2; exit 1 ;; esac
CKPT_DIR="${CKPT_DIR:-$STORE/GeoCore-9B}"
VAE="${VAE:-$CKPT_DIR/vae/diffusion_pytorch_model.safetensors}"
[ -f "$CKPT_DIR/model.safetensors.index.json" ] || { echo "FATAL: $CKPT_DIR has no model.safetensors.index.json (run download.sbatch)" >&2; exit 1; }
[ -f "$VAE" ] || { echo "FATAL: $VAE missing" >&2; exit 1; }
# The pinned digests (models/geocore/adapter.py) are re-verified inside run.py at every load; the
# index digest is also the reuse-check key below.
read -r INDEX_SHA BLOCKS_DEFAULT < <(PYTHONPATH="src:." python3 -c "
from models.geocore.adapter import PINNED_INDEX_SHA256, NUM_BLOCKS
print(PINNED_INDEX_SHA256, '4-8-12-16-20-24-28-31')")
BLOCKS="${BLOCKS:-$BLOCKS_DEFAULT}"
read -r -a BLOCK_LIST <<<"${BLOCKS//-/ }"
[ "${#BLOCK_LIST[@]}" -ge 2 ] || { echo "FATAL: BLOCKS needs >= 2 blocks (multi-block cache; the matrix pins one of them)" >&2; exit 1; }
TS=(100 180 260 340 420 500 580 740 900)
ENS=8
SUBSET_ARGS=()
SAVE_ROOT="models/geocore_eval/$DATASET"
[ "$RES" = 256 ] && SAVE_ROOT="models/geocore_eval_img256/$DATASET"
if [ "${ANCHOR:-0}" = 1 ]; then
    # GeoCore's own readout: one eps per image (eval/geocore_anchor.py); t=260 is their tau=0.25 grid point.
    ENS=1; TS=(260 580); SAVE_ROOT="models/geocore_eval_anchor${RES:+_img$RES}/$DATASET"
fi
if [ "${SMOKE:-0}" = 1 ]; then
    ENS=2; SUBSET_ARGS=(--subset-size 24); SAVE_ROOT="models/geocore_eval_smoke/$DATASET"
    rm -rf "$SAVE_ROOT"  # private smoke dir: always start clean
fi
TAG="_k$(IFS=-; echo "${BLOCK_LIST[*]}")"  # tasks/extraction.py: ..._oneshot_g0.0_k<blocks>.npz (no LoRA suffix: frozen)
UNIT="${UNIT:-all}"
case "$UNIT" in
    all) TRAIN_SHARDS="${TRAIN_SHARDS:-1}" ;;
    train:*) TRAIN_SHARDS="${TRAIN_SHARDS:?UNIT=$UNIT needs TRAIN_SHARDS=<S>}"
             SHARD="${UNIT#train:}"
             [[ "$SHARD" =~ ^[0-9]+$ ]] && [ "$SHARD" -lt "$TRAIN_SHARDS" ] || { echo "FATAL: UNIT=$UNIT not a shard index < $TRAIN_SHARDS" >&2; exit 1; } ;;
    valtest) TRAIN_SHARDS="${TRAIN_SHARDS:-1}" ;;
    stamp) TRAIN_SHARDS="${TRAIN_SHARDS:?UNIT=stamp needs the TRAIN_SHARDS the train units used}" ;;
    *) echo "FATAL: UNIT must be all | train:<i> | valtest | stamp, got $UNIT" >&2; exit 1 ;;
esac
[[ "$TRAIN_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: TRAIN_SHARDS=$TRAIN_SHARDS not a positive integer" >&2; exit 1; }
if [ "$TRAIN_SHARDS" -gt 1 ] && { [ "${ANCHOR:-0}" = 1 ] || [ "${SMOKE:-0}" = 1 ]; }; then
    echo "FATAL: the anchor / smoke extractions are unsharded (one job); drop TRAIN_SHARDS" >&2; exit 1
fi
echo "=== geocore $DATASET res=$RES unit=$UNIT train_shards=$TRAIN_SHARDS blocks=${BLOCK_LIST[*]} t=${TS[*]} ens=$ENS anchor=${ANCHOR:-0} smoke=${SMOKE:-0} ckpt=$CKPT_DIR -> $SAVE_ROOT $(date -Is)"

read -r REG_ROOT SPLITS < <(python3 -c "
import sys
from eval.features import official_spec
s = official_spec(sys.argv[1])
print(s['root'], ','.join(s['sizes']))" "$DATASET")
DATA="${DATA:-$REG_ROOT}"
IFS=',' read -r -a SPLIT_LIST <<<"$SPLITS"
[ -d "$DATA/${SPLIT_LIST[0]}" ] || { echo "FATAL: $DATA missing -- export $DATASET first (eval.export_geobench)" >&2; exit 1; }
python3 -m eval.export_geobench --task "${DATASET//_/-}" --out "$DATA"  # tree == committed manifest

# seed_for <split> [shard]: train 42 unsharded, 100+i per shard (never 43 / 44: val and test seeds,
# which probe.official_flux's cross-split seed guard checks against every shard's seed).
seed_for() {
    case "$1" in
        train) if [ "$TRAIN_SHARDS" -gt 1 ]; then echo $((100 + ${2:?shard})); else echo 42; fi ;;
        test) echo 43 ;; val) echo 44 ;;
        *) echo "FATAL: no seed for split $1" >&2; exit 1 ;;
    esac
}
shard_args() {  # shard_args <split> [shard] -> run.py flags for a sharded train unit
    if [ "$1" = train ] && [ "$TRAIN_SHARDS" -gt 1 ]; then echo "--num-shards $TRAIN_SHARDS --shard-index ${2:?shard}"; fi
}

cache_for() {  # cache_for <split> [shard] -> the one complete cache path for this request, or nothing
    local split="$1" shard="${2:-0}" f
    for f in "$SAVE_ROOT"/*/multistep_"$split"_feats_oneshot_g0.0"$TAG".npz; do
        [ -f "$f" ] && [ -f "${f%.npz}_meta.json" ] || continue
        if PYTHONPATH="src:." python3 - "${f%.npz}_meta.json" "${TS[*]}" "${BLOCK_LIST[*]}" "$ENS" "$(seed_for "$split" "$shard")" "$split" \
            "$DATASET" "${SUBSET_ARGS[*]:-}" "$RES" "$INDEX_SHA" "$TRAIN_SHARDS" "$shard" <<'PY' >&2
import json, sys
import numpy as np
from config_types import double_block_stamp
from eval.features import official_spec
meta, ts, ks, ens, seed, split, dataset, subset_args, res, index_sha, n_shards, shard = sys.argv[1:]
m = json.load(open(meta))
k = [int(x) for x in ks.split()]
sub = subset_args.split()
size = int(sub[sub.index("--subset-size") + 1]) if "--subset-size" in sub else official_spec(dataset)["sizes"][split]
n_shards, shard = (int(n_shards), int(shard)) if split == "train" else (1, 0)
if n_shards > 1:  # tasks/extraction.py: contiguous np.array_split of the (sorted) index list
    size = len(np.array_split(np.arange(size), n_shards)[shard])
want = {
    "t": [int(x) for x in ts.split()], "k": k, "ensemble_size": int(ens), "seed": int(seed),
    "eps_seed": int(seed), "split": split, "dataset": dataset, "subset_size": size,
    "num_shards": n_shards, "shard_index": shard,
    "extraction_mode": "ONESHOT", "guidance_scale": 0.0, "weights": "geocore-9b",
    "img_size": [int(res), int(res)], "hidden_size": 4096, "num_double_blocks": 8,
    "geocore_ckpt_sha256": index_sha, **double_block_stamp(k, num_double=8),
}
bad = {f: (m.get(f), v) for f, v in want.items() if m.get(f) != v}
print(f"  {meta}: " + (f"differs {bad}" if bad else "matches the request"))
sys.exit(1 if bad else 0)
PY
        then echo "$f"; return 0
        else echo "$split: $f does not match this request (above) -- not reused" >&2
        fi
    done
    return 1
}

extract() {  # extract <split> [shard]
    local split="$1" shard="${2:-0}" seed d suffix
    seed="$(seed_for "$split" "$shard")"
    if cache_for "$split" "$shard" >/dev/null; then echo "$split${2:+ shard $shard}: complete cache, skipping"; return 0; fi
    # Stale partial run dirs of THIS unit only (run.py names them <ds>_geocore_<hash>+<seed>[+val|+test][+sIofS]):
    # other units of the same dataset may be extracting under SAVE_ROOT right now (campus-gpu units).
    case "$split" in train) suffix="+$seed$(if [ "$TRAIN_SHARDS" -gt 1 ]; then echo "+s${shard}of$TRAIN_SHARDS"; fi)" ;; *) suffix="+$seed+$split" ;; esac
    for d in "$SAVE_ROOT"/*"$suffix"/; do
        [ -d "$d" ] || continue
        ls "$d"*_meta.json >/dev/null 2>&1 || { echo "removing incomplete extraction $d"; rm -rf "$d"; }
    done
    echo "=== $DATASET extract $split${2:+ shard $shard/$TRAIN_SHARDS} (seed $seed) $(date -Is)"
    # shellcheck disable=SC2046
    python3 run.py --task extract \
        --model.name geocore --model.ensemble-size "$ENS" \
        --geocore-ckpt "$CKPT_DIR" --geocore-vae "$VAE" \
        --dataset.name "$DATASET" --dataset.path "$DATA" \
        --img-size "$RES" "$RES" --t "${TS[@]}" \
        --extraction-mode ONESHOT --k "${BLOCK_LIST[@]}" --guidance-scale 0.0 \
        --seed "$seed" --eps-seed "$seed" --subset-seed 42 \
        --extract-split "$split" --label-fraction 1.0 --batch-size 1 \
        $(shard_args "$split" "$shard") \
        "${SUBSET_ARGS[@]}" --save-dir "$SAVE_ROOT"
    cache_for "$split" "$shard" >/dev/null || { echo "FATAL: no complete $split${2:+ shard $shard} cache under $SAVE_ROOT after extraction" >&2; exit 1; }
}

case "$UNIT" in
    train:*) extract train "$SHARD"; echo "GEOCORE_UNIT_OK $DATASET $RES $UNIT"; exit 0 ;;
    valtest) extract val; extract test; echo "GEOCORE_UNIT_OK $DATASET $RES $UNIT"; exit 0 ;;
    all)
        for split in "${SPLIT_LIST[@]}"; do
            if [ "$split" = train ] && [ "$TRAIN_SHARDS" -gt 1 ]; then
                for ((i = 0; i < TRAIN_SHARDS; i++)); do extract train "$i"; done
            else
                extract "$split"
            fi
        done ;;
    stamp) ;;  # verification + stamping only
esac

# DiTF discard channels (spec §4.5): the pre-registered rule on the TRAIN cache at k=8 (GeoCore's
# GSA block), stamped into every split's (and shard's) meta from that one train-derived list. The
# rule reads the first 1000 train images in dataset order: shard 0 holds them (contiguous split,
# >= 5000 images per shard at the shard counts above), so shard 0 is the source either way.
TRAIN_CACHE="$(cache_for train 0)" || { echo "FATAL: no complete train cache under $SAVE_ROOT after extraction" >&2; exit 1; }
STAMP_ARGS=()
for split in "${SPLIT_LIST[@]}"; do
    if [ "$split" = train ] && [ "$TRAIN_SHARDS" -gt 1 ]; then
        for ((i = 0; i < TRAIN_SHARDS; i++)); do
            c="$(cache_for train "$i")" || { echo "FATAL: no complete train shard $i/$TRAIN_SHARDS cache under $SAVE_ROOT" >&2; exit 1; }
            STAMP_ARGS+=("$c")
        done
    else
        c="$(cache_for "$split")" || { echo "FATAL: no complete $split cache under $SAVE_ROOT after extraction" >&2; exit 1; }
        STAMP_ARGS+=("$c")
    fi
done
# The rule is trusted only if it reproduces FLUX's published [154, 1446] on a FLUX cache at k=28
# (spec §4.5). That check runs here, in the same call, before anything is stamped: a failing or
# missing reference stops the job (review 2026-11-12, item 10; ~20 s, the cache is memory-mapped).
FLUX_DISCARD_REF="${FLUX_DISCARD_REF:-models/m_eurosat_multiblock_ens8/m_eurosat_flux_e2fcdb31+42/multistep_train_feats_oneshot_g1.0_k4-8-12-18-19-24-28-33-38-43-48-54-55-56.npz}"
[ -f "$FLUX_DISCARD_REF" ] || { echo "FATAL: FLUX discard reference cache missing: $FLUX_DISCARD_REF (set FLUX_DISCARD_REF to a FLUX multi-block train cache holding k=28)" >&2; exit 1; }
python3 -m eval.discard_channels --features "$TRAIN_CACHE" --k 8 --flux-ref "$FLUX_DISCARD_REF" --stamp "${STAMP_ARGS[@]}"

echo "GEOCORE_EXTRACT_OK $DATASET $RES -> $SAVE_ROOT"
cat <<EOT
matrix model entry (eval/matrix_specs/phase8_geocore.json):
  {"name": "geocore-9b${RES/224/}", "kind": "flux", "teacher": "dinov3-sat", "view": null,
   "features": "$SAVE_ROOT/../{dataset}/*/multistep_{split}_feats_oneshot_g0.0${TAG}.npz",
   "expect": {"extraction_mode": "ONESHOT", "ensemble_size": $ENS, "weights": "geocore-9b", "guidance_scale": 0.0, "hidden_size": 4096}}
  per_dataset k: eval.sweep select on THESE caches' val split at 100 % labels (spec §4.4).
EOT
