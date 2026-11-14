#!/bin/bash
# ============================================================================
# DINOv3 baseline arm on every GEO-Bench task -- GPU half: EXTRACTION ONLY (7B, bf16).
# Gate + probe run on CPU afterwards (dino_probe.sbatch, packed jobs chained by submit_dino.sh
# with --dependency=afterok on this WHOLE array: one failed extraction task leaves that preset's
# probe jobs DependencyNeverSatisfied -- scancel and re-queue them), so no GPU idles through hours of sklearn
# probing (the idle-GPU policy cancelled 7 sweep tasks for exactly that, 2026-10-02), and
# the control gate reuses these features instead of extracting a second time.
#   bash experiments/isaac/geobench/submit_dino.sh [--submit]       # both presets, all tasks
# One array task per dataset (eval/features.OFFICIAL registry order; --array must match).
# Writes models/dino_feats/<preset>_<key>_<split>.npz (scratch). Re-runnable: all three splits
# present -> skip. SMOKE=1: --max-images $MAX_IMAGES (strided) into models/dino_feats_smoke.
# Prereq (login node): experiments/isaac/geobench/prepare.sh (exports + DINOv3 hub code).
# TIME: 7B extraction is minutes-to-an-hour per task on an H100; 6 h leaves wide margin.
# ============================================================================
#SBATCH --job-name=geobench-dino
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # caches + models/data/ditf_models on scratch

source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
# Pin the env: with --export=ALL the job otherwise inherits the SUBMITTING shell's python3.
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
python3 -c "import sys, numpy, torch; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -f "$PROJECT_ROOT/.secrets/wandb-personal.env" ] && source "$PROJECT_ROOT/.secrets/wandb-personal.env"
export WANDB_MODE="${WANDB_MODE:-offline}"
set -euo pipefail

PRESET="${PRESET:?set PRESET to an eval/extract_dino.py PRESETS key (dinov3_vit7b16_web, dinov3_vit7b16_sat, dinov3_vitl16_web)}"  # no braces: they would close the expansion
KEY="$(python3 -c "import sys; from eval.features import OFFICIAL; print(list(OFFICIAL)[int(sys.argv[1])])" \
    "${SLURM_ARRAY_TASK_ID:?run as an array job}")"  # array index -> OFFICIAL key, in registry order
SMOKE="${SMOKE:-0}"
# DEVICE=cpu: float32 extraction on a CPU node, for a model small enough (ViT-L) when no GPU slot
# is free:  PRESET=dinov3_vitl16_web DEVICE=cpu sbatch -p short -q short -t 3:00:00 --gpus-per-task=0 \
#           --cpus-per-task=16 --mem=48G --array=0-5 experiments/isaac/geobench/dino.sbatch
# The cache records dtype + device; a cache of the other dtype is refused below, never reused.
DEVICE="${DEVICE:-cuda}"
case "$DEVICE" in
    cuda) DEV=(); WANT_DTYPE=preset ;;
    cpu)  DEV=(--device cpu --dtype float32); WANT_DTYPE=torch.float32 ;;
    *) echo "FATAL: DEVICE=$DEVICE (cuda | cpu)" >&2; exit 2 ;;
esac
if [ "$SMOKE" = 1 ]; then
    OUT=models/dino_feats_smoke; CAP=(--max-images "${MAX_IMAGES:-200}")
else
    OUT=models/dino_feats; CAP=()
fi
echo "=== extract $KEY $PRESET device=$DEVICE smoke=$SMOKE -> $OUT $(date -Is)"

python3 -m eval.export_geobench --task "${KEY//_/-}" --out "data/${KEY}_rgb"  # tree == manifest
# Reuse only INTACT caches of THIS preset over the full split (rule 11): every member readable
# (a job killed mid-np.savez leaves a truncated zip), preset recorded, N == the registry size.
have=0
rc=0
[ "$SMOKE" = 1 ] || python3 - "$OUT" "$PRESET" "$KEY" "$WANT_DTYPE" <<'PY' || rc=$?
import os, sys, zipfile
import numpy as np
from eval.extract_dino import PRESETS
from eval.features import official_spec
out, preset, key, want = sys.argv[1:]
want = str(PRESETS[preset]["dtype"]) if want == "preset" else want
# Every split is examined before anything is decided: a missing or corrupt FIRST split must not
# hide an intact cache of another dtype in a LATER split (the extractor would then be started and
# refuse, or overwrite nothing but waste the slot). dtype conflict -> exit 4 (fatal for the job);
# otherwise any missing / corrupt / mismatched split -> exit 1 (extract).
conflict, incomplete = [], []
for split, n in official_spec(key)["sizes"].items():
    p = os.path.join(out, f"{preset}_{key}_{split}.npz")
    try:
        with zipfile.ZipFile(p) as z:
            if z.testzip() is not None:
                raise ValueError("corrupt member")
        d = np.load(p)
        got = str(d["dtype"]) if "dtype" in d.files else str(PRESETS[preset]["dtype"])  # pre-2026-11-12 caches
        if str(d["preset"]) != preset or len(d["labels"]) != n or len(d["paths"]) != n:
            incomplete.append(f"{p}: preset {d['preset']} / N {len(d['labels'])} does not match {preset} / {n}")
        elif got != want:  # an intact cache of ANOTHER dtype: not this run's features, never overwritten
            conflict.append(f"{p} holds dtype {got}, this run extracts {want}")
    except (OSError, zipfile.BadZipFile, KeyError, ValueError) as e:
        incomplete.append(f"{p}: {e}")
if conflict:
    print("FATAL: " + "; ".join(conflict) + ": move it aside", file=sys.stderr)
    sys.exit(4)
if incomplete:
    sys.exit("; ".join(incomplete))
PY
[ "$rc" != 4 ] || exit 1
if [ "$SMOKE" != 1 ] && [ "$rc" = 0 ]; then
    have=1
fi
if [ "$have" = 1 ]; then
    echo "all splits already extracted in $OUT, skipping"
else
    python3 -m eval.extract_dino --preset "$PRESET" --dataset "$KEY" --out-dir "$OUT" "${DEV[@]}" "${CAP[@]}"
fi
# Success marker (judge the job by this line, not by its state): every split's cache is intact,
# records this preset and holds the registry's row count.
python3 - "$OUT" "$PRESET" "$KEY" "$SMOKE" "$WANT_DTYPE" <<'PY'
import os, sys, zipfile
import numpy as np
from eval.extract_dino import PRESETS
from eval.features import official_spec
out, preset, key, smoke, want = sys.argv[1:]
want = str(PRESETS[preset]["dtype"]) if want == "preset" else want
tag = "_smoke" if smoke == "1" else ""
for split, n in official_spec(key)["sizes"].items():
    p = os.path.join(out, f"{preset}_{key}_{split}{tag}.npz")
    with zipfile.ZipFile(p) as z:
        if z.testzip() is not None:
            sys.exit(f"FATAL: {p}: corrupt member")
    d = np.load(p)
    rows = {k: len(d[k]) for k in ("cls", "mp", "cls4", "labels", "paths")}
    if str(d["preset"]) != preset or len(set(rows.values())) != 1 or (smoke != "1" and rows["labels"] != n):
        sys.exit(f"FATAL: {p}: preset {d['preset']} rows {rows}, expected {preset} / {n}")
    if not all(np.isfinite(d[k]).all() for k in ("cls", "mp", "cls4")):
        sys.exit(f"FATAL: {p}: non-finite features")
    got = str(d["dtype"]) if "dtype" in d.files else str(PRESETS[preset]["dtype"])
    if got != want:
        sys.exit(f"FATAL: {p}: dtype {got}, this run is {want}")
    dev = str(d["device"]) if "device" in d.files else "cuda"
    print(f"  {p}: {rows['labels']} rows, cls {d['cls'].shape[1]}-d, cls4 {d['cls4'].shape[1]}-d, {got} on {dev}")
PY
echo "DINO_EXTRACT_OK $KEY $PRESET device=$DEVICE smoke=$SMOKE -> $OUT $(date -Is)"
