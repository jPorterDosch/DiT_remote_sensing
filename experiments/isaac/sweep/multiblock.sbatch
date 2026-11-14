#!/bin/bash
# ============================================================================
# MULTI-BLOCK feature banking (frozen FLUX one-shot, ens8, g=1.0): ONE early-exiting pass per
# image and timestep caches every block in BLOCKS, so cross-timestep x cross-block fusion can be
# probed on CPU afterwards with no further GPU. Cost ~ the deepest block alone, per timestep.
#   sbatch --export=ALL,DATASET=m_eurosat experiments/isaac/sweep/multiblock.sbatch
#   sbatch --export=ALL,DATASET=m_eurosat,SMOKE=1 --time=1:30:00 experiments/isaac/sweep/multiblock.sbatch
# Use submit_multiblock.sh: it sizes --time and gates the full runs on the smoke.
#
# Env: DATASET (required), BLOCKS (hyphen list, default 4-8-12-18-19-24-...-56: 4 double + 10 single),
# TS (hyphen list of timesteps, default 100-180-...-580-740-900), SAVE_ROOT, ENS (8; smoke 2), SMOKE.
# Cache: $SAVE_ROOT/<run>/multistep_<split>_feats_oneshot_g1.0_k<b1-b2-...>.npz, feats
# (N, T, B, 3072) + mods (T, B, 3, 3072), layout "N,K,B,C" in npz and meta.json; t and k are
# recorded in both. The block list in the filename keeps it out of every (N, K, C) consumer's
# glob. Same seeds as the sweep (train 42 / test 43 / val 44, subset-seed 42) and eps/latents
# are drawn once per image before the first timestep, so caches from DIFFERENT (TS, BLOCKS)
# runs are per-image paired: supplements (new timesteps, new blocks) extend the main grid
# without re-extracting it -- but ONLY into their own SAVE_ROOT: a same-BLOCKS supplement has the
# same filename, so sharing a SAVE_ROOT makes the probe's {split} glob match two unsharded caches
# (load_flux_split refuses), and the incomplete-run cleanup below must never see another job's
# live run dir. Never point two concurrent jobs at one SAVE_ROOT (submit_multiblock.sh refuses). A split counts as complete only if its _meta.json records exactly
# this TS and BLOCKS (same-named caches with another t/k, e.g. a supplement moved in after the
# fact, are never reused); run dirs with no _meta.json are removed first, so re-queueing
# resumes. Never share a SAVE_ROOT between CONCURRENT jobs: that removal would hit a live run.
#
# SMOKE=1 (gate, rule 13/15): 20 train images, ENS=2. A mixed double+single multi-block pass
# over t=580,740,900 vs three int-k single-block passes, each with a DIFFERENT 2-t list; every
# compared slice must match (bit-identical expected). Exercises the 2026-10-05 double-block
# fix (feature = block input, mod = img_mod1) and cross-run pairing. PASS/FAIL; FAIL exits 1.
# ============================================================================
#SBATCH --job-name=geobench-multiblock
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=30:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # caches + models/data/logs on Lustre scratch

source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
python3 -c "import sys, numpy, torch; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -f "$PROJECT_ROOT/.secrets/wandb-personal.env" ] && source "$PROJECT_ROOT/.secrets/wandb-personal.env"
export WANDB_MODE="${WANDB_MODE:-offline}"
set -euo pipefail

DATASET="${DATASET:?submit with --export=ALL,DATASET=<OFFICIAL key>}"
SMOKE="${SMOKE:-0}"
BLOCKS="${BLOCKS:-4-8-12-18-19-24-28-33-38-43-48-54-55-56}"  # hyphen-separated: safe inside sbatch --export
TS="${TS:-100-180-260-340-420-500-580-740-900}"
read -r -a BLOCK_LIST <<<"${BLOCKS//-/ }"
read -r -a T_LIST <<<"${TS//-/ }"
if [ "$SMOKE" = 1 ]; then
    ENS="${ENS:-2}"
    SAVE_ROOT="models/smoke_multiblock/${DATASET}"
    rm -rf "$SAVE_ROOT"  # private smoke dir: always start clean so the gate tests THIS code
else
    ENS="${ENS:-8}"
    SAVE_ROOT="${SAVE_ROOT:-models/${DATASET}_multiblock_ens${ENS}}"
fi
echo "=== multiblock $DATASET t=${T_LIST[*]} blocks=${BLOCK_LIST[*]} ens=$ENS smoke=$SMOKE -> $SAVE_ROOT $(date -Is)"

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

# extract <split> <save_dir> <ts-hyphen> <k...>: one run.py extraction (resume-safe). One k ->
# the int (single-block) path and the plain filename; several -> multi-block, _k<list> filename.
extract() {
    local split="$1" dir="$2" ts="$3"; shift 3
    local seed f d tag="" ts_list
    [ "$#" -gt 1 ] && tag="_k$(IFS=-; echo "$*")"
    read -r -a ts_list <<<"${ts//-/ }"
    seed="$(seed_for "$split")"
    for f in "$dir"/*/multistep_"$split"_feats_oneshot_g1.0"$tag".npz; do
        [ -f "$f" ] && [ -f "${f%.npz}_meta.json" ] || continue
        # Reuse only a cache of THIS request: t, k, and every run.py argument below that the
        # filename does not encode (ENS lives only in the default SAVE_ROOT name; review 2026-10-11).
        if PYTHONPATH="src:." python3 - "${f%.npz}_meta.json" "$ts" "$*" "$ENS" "$seed" "$split" \
            "$DATASET" "${SUBSET_ARGS:-}" <<'PY'
import json, sys
from config_types import double_block_stamp
from eval.features import official_spec
meta, ts, ks, ens, seed, split, dataset, subset_args = sys.argv[1:]
m = json.load(open(meta))
k = [int(x) for x in ks.split()]
sub = subset_args.split()
size = int(sub[sub.index("--subset-size") + 1]) if "--subset-size" in sub else official_spec(dataset)["sizes"][split]
want = {
    "t": [int(x) for x in ts.split("-")], "k": k if len(k) > 1 else k[0], "ensemble_size": int(ens),
    "seed": int(seed), "eps_seed": int(seed), "split": split, "dataset": dataset, "subset_size": size,
    "extraction_mode": "ONESHOT", "guidance_scale": 1.0, **double_block_stamp(k),
}
bad = {f: (m.get(f), v) for f, v in want.items() if m.get(f) != v}
print(f"  {meta}: " + (f"differs {bad}" if bad else "matches the request"))
sys.exit(1 if bad else 0)
PY
        then echo "$split: complete cache $f (t=$ts k=$*), skipping"; return 0
        else echo "$split: $f does not match this request (above) -- not reused"
        fi
    done
    for d in "$dir"/*/; do
        [ -d "$d" ] || continue
        ls "$d"*_meta.json >/dev/null 2>&1 || { echo "removing incomplete extraction $d"; rm -rf "$d"; }
    done
    echo "=== $DATASET extract $split t=${ts_list[*]} k=$* (seed $seed) $(date -Is)"
    # shellcheck disable=SC2086
    python3 run.py --task extract \
        --model.name flux --model.ensemble-size "$ENS" \
        --dataset.name "$DATASET" --dataset.path "$DATA" \
        --img-size 224 224 --t "${ts_list[@]}" \
        --extraction-mode ONESHOT --k "$@" --guidance-scale 1.0 \
        --seed "$seed" --eps-seed "$seed" --subset-seed 42 \
        --extract-split "$split" --label-fraction 1.0 --batch-size 1 \
        ${SUBSET_ARGS:-} --save-dir "$dir"
}

if [ "$SMOKE" = 1 ]; then
    SUBSET_ARGS="--subset-size 20"
    extract train "$SAVE_ROOT/multi" 580-740-900 8 12 19 56   # mixed double + single
    # task=extract needs >= 2 timesteps (a single --t parses as an int and is refused).
    extract train "$SAVE_ROOT/single_12" 580-740 12           # double, int path, t list = prefix
    extract train "$SAVE_ROOT/single_8" 740-900 8             # double, int path, first t differs
    extract train "$SAVE_ROOT/single_56" 900-980 56           # single, int path, first t differs + t not in multi
    python3 - "$SAVE_ROOT" <<'PY'
import glob, json, sys
import numpy as np
root = sys.argv[1]
def one(pattern):
    hits = glob.glob(pattern)
    assert len(hits) == 1, f"expected 1 cache for {pattern}, got {hits}"
    return np.load(hits[0], allow_pickle=True), json.load(open(hits[0][:-4] + "_meta.json"))
blocks, ts = [8, 12, 19, 56], [580, 740, 900]
m, mm = one(f"{root}/multi/*/multistep_train_feats_oneshot_g1.0_k8-12-19-56.npz")
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
C = m["feats"].shape[-1]
check("layout stamp", str(m["layout"]) == "N,K,B,C" and mm.get("layout") == "N,K,B,C")
check("feats shape", m["feats"].shape == (20, 3, 4, C), m["feats"].shape)
check("mods shape", m["mods"].shape == (3, 4, 3, C), m["mods"].shape)
check("t/k recorded", mm["t"] == ts and mm["k"] == blocks and list(m["timesteps"]) == ts, (mm["t"], mm["k"]))
check("finite", np.isfinite(m["feats"]).all() and np.isfinite(m["mods"]).all())
check("double != single features", not np.allclose(m["feats"][:, :, 1], m["feats"][:, :, 2]))
check("double mods non-trivial", np.abs(m["mods"][:, :2]).max() > 0)
for name, b, t in (("single_12", 12, 580), ("single_12", 12, 740), ("single_8", 8, 740), ("single_8", 8, 900), ("single_56", 56, 900)):
    s, sm = one(f"{root}/{name}/*/multistep_train_feats_oneshot_g1.0.npz")
    i, j, i_s = ts.index(t), blocks.index(b), list(s["timesteps"]).index(t)
    check(f"{name} pairing", np.array_equal(s["subset_indices"], m["subset_indices"]) and np.array_equal(s["labels"], m["labels"]))
    a, r = m["feats"][:, i, j], s["feats"][:, i_s]
    rel = float(np.abs(a - r).max() / max(np.abs(r).max(), 1e-12))
    check(f"k{b} t{t} slice == int-path run", rel < 1e-3, f"(max rel diff {rel:.2e}, bit-identical={np.array_equal(a, r)})")
    check(f"k{b} t{t} mods == int-path run", np.allclose(m["mods"][i, j], s["mods"][i_s], rtol=1e-3, atol=1e-6))
print("SMOKE GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
    exit $?
fi

for split in "${SPLIT_LIST[@]}"; do
    extract "$split" "$SAVE_ROOT" "$TS" "${BLOCK_LIST[@]}"
done
echo "=== multiblock $DATASET done $(date -Is): $(du -sh "$SAVE_ROOT" | cut -f1) in $SAVE_ROOT"
