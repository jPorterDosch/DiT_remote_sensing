#!/bin/bash
# ============================================================================
# Evaluate ONE Git-10M checkpoint on the six GEO-Bench tasks: the ordinary MULTI-BLOCK one-shot
# extraction (run.py --task extract) with the LoRA loaded, the same 14 blocks, nine timesteps,
# ens8, g=1.0, seeds 42/43/44, subset-seed 42 as the frozen caches -- flag for flag the extraction
# that produced models/<dataset>_multiblock_ens8 (experiments/isaac/sweep/multiblock.sbatch), so
# the only difference between a frozen and an adapted cache is the adapter (the cache name and
# meta carry it: ..._lora-<arm>_k<blocks>.npz, weights="flux-dev+lora:<arm>", lora_checkpoint,
# lora_global_step; eval.features refuses to mix them). Then eval.matrix with
# eval/matrix_specs/phase2_template.json: every arm at the fixed aligned block k=28, and -- after
# eval.sweep select on the adapted caches, exactly as for the frozen model -- each arm at its OWN
# val-selected block (user decision 2026-10-14: compare frozen-best vs adapted-best, never
# adapted-at-frozen's-block; adaptation may move the optimum, which is itself a result).
#
#   CKPT   <run>/checkpoints/lora_last_*.pt (required; the run's meta.json / cfg are read for the arm)
#   NAME   eval dir name, default = the checkpoint's run-dir basename (git10m_flux_<hash>+<seed>+<arm>-<8hex>)
#   BLOCKS hyphen list, default the frozen caches' 4-8-12-18-19-24-28-33-38-43-48-54-55-56
#   SMOKE=1  20 train images, ens2, models/git10m_eval_smoke/ (never quotable): run first
#
#   sbatch --export=ALL,CKPT=$STORE/models/git10m/<run>/checkpoints/lora_last_20000.pt --array=0-5 experiments/isaac/git10m/extract_ckpt.sbatch
#   sbatch --export=ALL,SMOKE=1,CKPT=... --array=0 --time=1:00:00 experiments/isaac/git10m/extract_ckpt.sbatch
#
# Output: models/git10m_eval/<NAME>/<dataset>/<run>/multistep_<split>_feats_oneshot_g1.0_lora-<arm>_k<blocks>.npz
# (+ _meta.json; feats (N, T, B, 3072) + mods (T, B, 3, 3072), layout N,K,B,C). Resume-safe: a split whose
# cache's _meta.json matches this request is skipped; a run dir without _meta.json is removed.
# Every array task prints GIT10M_EVAL_OK <dataset> and the matrix model entry to paste.
# ============================================================================
#SBATCH --job-name=git10m-eval
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=30:00:00
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
python3 -c "import sys, numpy, torch, tyro; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
set -euo pipefail

DATASETS=(m_eurosat m_brick_kiln m_pv4ger m_so2sat m_forestnet m_bigearthnet)
IDX="${SLURM_ARRAY_TASK_ID:?submit with --array=0-5 (one GEO-Bench task per array index)}"
[ "$IDX" -ge 0 ] && [ "$IDX" -lt "${#DATASETS[@]}" ] || { echo "FATAL: array index $IDX not in 0..5" >&2; exit 1; }
DATASET="${DATASETS[$IDX]}"
CKPT="${CKPT:?set CKPT=<run>/checkpoints/lora_last_*.pt via --export}"
[ -f "$CKPT" ] || { echo "FATAL: $CKPT missing" >&2; exit 1; }
RUN_DIR="$(cd "$(dirname "$CKPT")/.." && pwd)"
NAME="${NAME:-$(basename "$RUN_DIR")}"
[ -f "$RUN_DIR/meta.json" ] || { echo "FATAL: $RUN_DIR/meta.json missing -- the run did not finish" >&2; exit 1; }
# ARM = the arm; LORA_TAG = tasks/extraction.lora_tag of the checkpoint's align block, i.e. the arm
# plus -shuf<mode> for a shuffled-teacher control: the cache filename and its `weights` pin carry
# it, so a control's cache never passes a spec's expect.weights for the arm it controls.
# LORA_RANK / LORA_ALPHA come from the run's own config (2026-10-24: capacity runs train at other
# ranks; the extraction must wrap the same adapter shape). Rank 4 / alpha 16 for every earlier run,
# so their extraction arguments and run names are unchanged.
read -r ARM LORA_TAG LORA_RANK LORA_ALPHA < <(PYTHONPATH="src:." python3 -c "
import json, sys
from tasks.extraction import lora_tag
m = json.load(open(sys.argv[1]))
al = m['cfg']['align']
assert al['arm'] in ('fm', 'naive', 'proj', 'sem'), al['arm']
print(al['arm'], lora_tag(al), int(m['cfg']['lora_rank']), float(m['cfg']['lora_alpha']))" "$RUN_DIR/meta.json") || { echo "FATAL: no align.arm in $RUN_DIR/meta.json" >&2; exit 1; }
[ -n "$ARM" ] && [ -n "$LORA_TAG" ] && [ -n "$LORA_RANK" ] && [ -n "$LORA_ALPHA" ] || { echo "FATAL: empty ARM/LORA_TAG/LORA_RANK/LORA_ALPHA from $RUN_DIR/meta.json" >&2; exit 1; }
CKPT_SHA1="$(sha1sum "$CKPT" | cut -d' ' -f1)"
# The frozen caches' block list and t grid (k=28, the aligned block, is among the blocks).
BLOCKS="${BLOCKS:-4-8-12-18-19-24-28-33-38-43-48-54-55-56}"
read -r -a BLOCK_LIST <<<"${BLOCKS//-/ }"
[ "${#BLOCK_LIST[@]}" -ge 2 ] || { echo "FATAL: BLOCKS needs >= 2 blocks (multi-block cache; the matrix pins one of them)" >&2; exit 1; }
TS=(100 180 260 340 420 500 580 740 900)
TAG="_lora-${LORA_TAG}_k$(IFS=-; echo "${BLOCK_LIST[*]}")"  # tasks/extraction.py: ..._oneshot_g1.0_lora-<tag>_k<blocks>.npz
[ "$TAG" != "_lora-${LORA_TAG}_k" ] || { echo "FATAL: empty block list in TAG" >&2; exit 1; }
if [ "${SMOKE:-0}" = 1 ]; then
    ENS=2; SUBSET_ARGS=(--subset-size 20); SAVE_ROOT="models/git10m_eval_smoke/$NAME/$DATASET"
    rm -rf "$SAVE_ROOT"  # private smoke dir: always start clean
else
    ENS=8; SUBSET_ARGS=(); SAVE_ROOT="models/git10m_eval/$NAME/$DATASET"
fi
[ -f "$RUN_DIR/conflict_gate.txt" ] && ! grep -q FAIL "$RUN_DIR/conflict_gate.txt" \
    || { echo "FATAL: $RUN_DIR/conflict_gate.txt missing or FAIL -- the run did not finish clean" >&2; exit 1; }
echo "=== eval $NAME on $DATASET: ckpt=$CKPT sha1=$CKPT_SHA1 tag=$LORA_TAG blocks=${BLOCK_LIST[*]} t=${TS[*]} ens=$ENS smoke=${SMOKE:-0} -> $SAVE_ROOT $(date -Is)"

read -r REG_ROOT SPLITS < <(python3 -c "
import sys
from eval.features import official_spec
s = official_spec(sys.argv[1])
print(s['root'], ','.join(s['sizes']))" "$DATASET")
DATA="${DATA:-$REG_ROOT}"
IFS=',' read -r -a SPLIT_LIST <<<"$SPLITS"
[ -d "$DATA/${SPLIT_LIST[0]}" ] || { echo "FATAL: $DATA missing -- export $DATASET first (eval.export_geobench)" >&2; exit 1; }
python3 -m eval.export_geobench --task "${DATASET//_/-}" --out "$DATA"  # tree == committed manifest

seed_for() { case "$1" in train) echo 42 ;; test) echo 43 ;; val) echo 44 ;; *) echo "FATAL: no seed for split $1" >&2; exit 1 ;; esac; }

extract() {
    local split="$1" seed f d
    seed="$(seed_for "$split")"
    for f in "$SAVE_ROOT"/*/multistep_"$split"_feats_oneshot_g1.0"$TAG".npz; do
        [ -f "$f" ] && [ -f "${f%.npz}_meta.json" ] || continue
        if PYTHONPATH="src:." python3 - "${f%.npz}_meta.json" "${TS[*]}" "${BLOCK_LIST[*]}" "$ENS" "$seed" "$split" \
            "$DATASET" "${SUBSET_ARGS[*]:-}" "$CKPT" "$CKPT_SHA1" <<'PY'
import json, os, sys
from config_types import double_block_stamp
from eval.features import official_spec
meta, ts, ks, ens, seed, split, dataset, subset_args, ckpt, ckpt_sha1 = sys.argv[1:]
m = json.load(open(meta))
k = [int(x) for x in ks.split()]
sub = subset_args.split()
size = int(sub[sub.index("--subset-size") + 1]) if "--subset-size" in sub else official_spec(dataset)["sizes"][split]
# The same request fields the frozen caches' reuse check compares (sweep/multiblock.sbatch extract())
# plus the checkpoint; keep the two lists in step.
want = {
    "t": [int(x) for x in ts.split()], "k": k, "ensemble_size": int(ens), "seed": int(seed),
    "eps_seed": int(seed), "split": split, "dataset": dataset, "subset_size": size,
    "extraction_mode": "ONESHOT", "guidance_scale": 1.0, **double_block_stamp(k),
    "lora_checkpoint": os.path.abspath(ckpt),
    # Content, not just path: a retrained run at the same path must not reuse the old adapter's
    # caches (review 2026-10-17; tasks/extraction.py stamps the sha1).
    "lora_checkpoint_sha1": ckpt_sha1,
}
bad = {f: (m.get(f), v) for f, v in want.items() if m.get(f) != v}
print(f"  {meta}: " + (f"differs {bad}" if bad else "matches the request"))
sys.exit(1 if bad else 0)
PY
        then echo "$split: complete cache $f, skipping"; return 0
        else echo "$split: $f does not match this request (above) -- not reused"
        fi
    done
    for d in "$SAVE_ROOT"/*/; do
        [ -d "$d" ] || continue
        ls "$d"*_meta.json >/dev/null 2>&1 || { echo "removing incomplete extraction $d"; rm -rf "$d"; }
    done
    echo "=== $DATASET extract $split (seed $seed) $(date -Is)"
    python3 run.py --task extract \
        --model.name flux --model.ensemble-size "$ENS" \
        --dataset.name "$DATASET" --dataset.path "$DATA" \
        --img-size 224 224 --t "${TS[@]}" \
        --extraction-mode ONESHOT --k "${BLOCK_LIST[@]}" --guidance-scale 1.0 \
        --seed "$seed" --eps-seed "$seed" --subset-seed 42 \
        --extract-split "$split" --label-fraction 1.0 --batch-size 1 \
        --lora-checkpoint "$CKPT" --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --lora-dropout 0.0 --wrap-output \
        "${SUBSET_ARGS[@]}" --save-dir "$SAVE_ROOT"
}

for split in "${SPLIT_LIST[@]}"; do
    extract "$split"
done
echo "GIT10M_EVAL_OK $DATASET -> $SAVE_ROOT"
cat <<EOF
matrix model entries (eval/matrix_specs/phase2_template.json; fill <NAME_$ARM> = $NAME):
  fixed aligned block:
  {"name": "flux-$LORA_TAG", "kind": "flux", "teacher": "dinov3-sat", "view": null,
   "features": "models/git10m_eval/$NAME/{dataset}/*/multistep_{split}_feats_oneshot_g1.0_lora-${LORA_TAG}_k*.npz",
   "expect": {"extraction_mode": "ONESHOT", "ensemble_size": $ENS, "weights": "flux-dev+lora:$LORA_TAG", "k": 28}}
  own best block: the same entry named "flux-$LORA_TAG-best" plus "per_dataset": {<dataset>: {"expect": {..., "k": <block>}}},
  <block> = eval.sweep select on THESE caches' val split at 100 % labels (phase1.json's frozen
  per-dataset blocks came from the same selection on the frozen caches).
EOF
