#!/bin/bash
# End-to-end GeoCore smoke on one A40 (CLAUDE.md rule 15), four gates, prints GEOCORE_SMOKE_OK:
#   1. pytest --gpu tests/test_geocore_null_text.py  (local T5-XXL / CLIP-L encoders, "" prompt)
#   2. extract.sbatch logic with SMOKE=1 on m_eurosat (24 images, ens 2, all 8 blocks, 9 t), then
#      shapes / finiteness / meta keys / discard stamp checked
#   3. parity_check.py: adapter k=8 feature == vendored double-block-7 output; cache == re-derived
#   4. FLUX regression: the same tiny FLUX extraction from this worktree and from the main
#      checkout (pre-change code) must be byte-identical
#
#   sbatch experiments/isaac/geocore/smoke.sbatch
#
#SBATCH --job-name=geocore-smoke
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE=disabled
set -euo pipefail
MAIN="${MAIN:-/nfs/home/jdosch1/COSC424/DiT_remote_sensing}"
CKPT_DIR="${CKPT_DIR:-$STORE/GeoCore-9B}"
VAE="$CKPT_DIR/vae/diffusion_pytorch_model.safetensors"

# GATES: space-separated subset of "1 2 3 4" (default all). Gates 3/4 reuse the smoke cache of an
# earlier gate-2 run, so a fix to the parity script or the regression does not cost the T5 loads twice.
GATES="${GATES:-1 2 3 4}"
gate() { case " $GATES " in *" $1 "*) return 0;; *) echo "=== gate $1: skipped (GATES=$GATES)"; return 1;; esac; }

# PASS iff pytest's summary line reports >= 1 passed and nothing failed / errored / was skipped,
# deselected, xfailed, xpassed or rerun (an expected failure is still a gate that did not hold). pytest exits 0 when every test SKIPS (no CUDA, encoders missing), and the old
# `pytest | tail -5` then went on to print GEOCORE_SMOKE_OK with gate 1 never run (review
# 2026-11-12, item 10). ANSI colour codes are stripped first. tests/test_job_gates.py feeds this
# function the summaries it must refuse.
pytest_all_passed() {  # pytest_all_passed <captured pytest output>
    local line
    line="$(sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -E ' in [0-9.]+s' | tail -1)"
    [[ "$line" =~ (^|[^0-9])[1-9][0-9]*\ passed ]] && ! [[ "$line" =~ failed|error|skipped|deselected|xfailed|xpassed|rerun|no\ tests\ ran ]]
}

if gate 1; then
echo "=== gate 1: null-text embeddings $(date -Is)"
G1_LOG="$(mktemp)"; g1_rc=0
python3 -m pytest -q -rs -p no:cacheprovider --color=no --gpu tests/test_geocore_null_text.py >"$G1_LOG" 2>&1 || g1_rc=$?
tail -n 25 "$G1_LOG"
if [ "$g1_rc" != 0 ] || ! pytest_all_passed "$G1_LOG"; then
    echo "FATAL: gate 1 did not PASS (pytest exit $g1_rc; a skipped test is not a pass)" >&2; exit 1
fi
rm -f "$G1_LOG"
echo "GATE geocore null-text PASS"
fi

if gate 2; then
echo "=== gate 2: smoke extraction $(date -Is)"
SLURM_ARRAY_TASK_ID=0 SMOKE=1 RES=224 CKPT_DIR="$CKPT_DIR" bash experiments/isaac/geocore/extract.sbatch
fi
CACHE="$(ls models/geocore_eval_smoke/m_eurosat/*/multistep_train_feats_oneshot_g0.0_k4-8-12-16-20-24-28-31.npz)"
[ -f "$CACHE" ] || { echo "FATAL: no smoke cache (run gate 2 first)" >&2; exit 1; }
if gate 2; then
PYTHONPATH="src:." python3 - "$CACHE" <<'PY'
import json, sys
import numpy as np
p = sys.argv[1]; d = np.load(p); m = json.load(open(p[:-4] + "_meta.json"))
assert d["feats"].shape == (24, 9, 8, 4096), d["feats"].shape
assert d["mods"].shape == (9, 8, 3, 4096), d["mods"].shape
assert np.isfinite(d["feats"]).all() and np.isfinite(d["mods"]).all()
assert d["feats"].std() > 0
for k in ("hidden_size", "num_double_blocks", "num_blocks", "vae_encode", "text", "geo_null", "geocore_commit",
          "geocore_ckpt_sha256", "geocore_vae_sha256", "discard_channels", "discard_rule"):
    assert k in m, k
assert m["weights"] == "geocore-9b" and m["guidance_scale"] == 0.0 and m["double_block_feature"] == "block_input+img_mod1"
assert str(d["weights"]) == "geocore-9b"
# shared global modulation: identical across the 7 single blocks (k >= 8) and across the double block(s)
assert np.array_equal(d["mods"][:, 1], d["mods"][:, 2]) and not np.array_equal(d["mods"][:, 0], d["mods"][:, 1])
print("smoke cache OK", d["feats"].shape, "discard", m["discard_channels"])
PY
fi

if gate 3; then
echo "=== gate 3: parity $(date -Is)"
PYTHONPATH="src:." python3 experiments/isaac/geocore/parity_check.py --cache "$CACHE" --geocore-ckpt "$CKPT_DIR" --geocore-vae "$VAE"
fi

if gate 4; then
echo "=== gate 4: FLUX regression $(date -Is)"
REG="$STORE/models/geocore_smoke_fluxreg"
rm -rf "$REG"; mkdir -p "$REG/new" "$REG/old"
flux_tiny() {  # flux_tiny <checkout> <out>
    ( cd "$1" && python3 run.py --task extract --model.name flux --model.ensemble-size 2 \
        --dataset.name m_eurosat --dataset.path data/m_eurosat_rgb --img-size 224 224 --t 100 180 \
        --extraction-mode ONESHOT --k 4 28 --guidance-scale 1.0 --seed 42 --eps-seed 42 --subset-seed 42 \
        --extract-split train --label-fraction 1.0 --batch-size 1 --subset-size 20 --save-dir "$2" )
}
flux_tiny "$PROJECT_ROOT" "$REG/new"
flux_tiny "$MAIN" "$REG/old"
python3 - "$REG" <<'PY'
import glob, sys
import numpy as np
r = sys.argv[1]
a, b = (np.load(glob.glob(f"{r}/{w}/*/multistep_train_feats_oneshot_g1.0_k4-28.npz")[0]) for w in ("new", "old"))
for key in ("feats", "mods", "labels", "subset_indices"):
    assert np.array_equal(a[key], b[key]), key
print("FLUX regression OK: byte-identical feats/mods on 20 images, k 4 & 28, t 100 & 180")
PY
fi

echo "GEOCORE_SMOKE_OK gates=$GATES $(date -Is)"
