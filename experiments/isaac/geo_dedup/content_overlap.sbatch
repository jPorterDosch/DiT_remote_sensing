#!/bin/bash
# ============================================================================
# Residual CONTENT overlap of the Git-10M rows the geographic pass cannot assess
# (eval/content_overlap.py): DINOv3-SAT CLS of every unique unlocatable image (1,706,416),
# nearest GEO-Bench sample by cosine, threshold from a null of RETAINED rows, planted-copy and
# same-location controls, executed gates. Two phases:
#
#   GPU array (this file, STEP=embed, default): task i < N_PARTS embeds part i/N_PARTS of
#   unassessable.csv (unique img_name); task N_PARTS = null (NULL_N retained rows, seed 0);
#   task N_PARTS+1 = excluded rows (EXCL_N, same place by coordinates: rows matched by a sample
#   box, NOT rows a city ROI alone excluded -- 71 % of the v3 excluded set, --drop-city-only);
#   task N_PARTS+2 =
#   planted controls (GEO-Bench test images x 5 variants). Outputs under $STORE/geo_dedup/content/.
#   MANIFEST_DIR: the null must be rows KNOWN not to share a place with any evaluation sample,
#   so use the pass that locates the most tasks (dedup_v3_b1000_cities covers m-so2sat at city
#   level; v2 does not, and 1.2 M of its retained rows sit inside So2Sat cities). The
#   unassessable set is identical in v2 and v3. NULL_N: tau at fpr 1e-4 is the quantile set by
#   the top NULL_N x 1e-4 rows -- 2 rows at 20k, 20 at 200k; use >= 200k.
#     mkdir -p logs
#     sbatch --export=ALL,MANIFEST_DIR=$STORE/geo_dedup/dedup_v3_b1000_cities,NULL_N=200000 --array=0-10 experiments/isaac/geo_dedup/content_overlap.sbatch   # N_PARTS=8 -> 0..7 parts, 8 null, 9 excl, 10 ctrl
#     sbatch -p campus-gpu-bigmem -q campus-gpu -t 1-00:00:00 --gpus-per-task=a40:1 --array=0-10 ... (same --export)
#   CPU match (STEP=match, after every array task printed CONTENT_*_OK; same NULL_N, it names the
#   null file; match reads every file's `origin` and refuses a null / reference / part from
#   another MANIFEST_DIR, a missing or repeated part, or a reference without --drop-city-only):
#     sbatch --export=ALL,STEP=match,NULL_N=200000 -p short -q short -t 3:00:00 --gpus-per-task=0 --cpus-per-task=16 --mem=96G experiments/isaac/geo_dedup/content_overlap.sbatch
#   -> $STORE/geo_dedup/content/report_<preset>/{report.json,report.md,hits.csv,query_max_sim.npz}
#   SMOKE (first, CPU is enough -- the 7B load is killed on the login node): 8 unassessable rows,
#   8 retained rows, 2 tasks x 2 controls through the REAL backbone, then the identity controls
#   are compared with the cached features (cos >= 0.99) -> CONTENT_SMOKE_OK. Writes $OUT/smoke/.
#     sbatch --export=ALL,SMOKE=1 -p short -q short -t 2:00:00 --gpus-per-task=0 --cpus-per-task=16 --mem=120G experiments/isaac/geo_dedup/content_overlap.sbatch
#
# Sizing (estimate, not measured): ViT-7B/16 at 224 px ~ 3 TFLOP/image -> ~100-150 img/s on an
# H100 (bf16), ~40-60 on an A40; 1.71 M / 8 parts = 214 k images per part = 25-90 min + JPEG
# decode (8 workers). MANIFEST_DIR must be the pass the training index was built from.
# ============================================================================
#SBATCH --job-name=content-overlap
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # models/ data/ ditf_models/ -> scratch

export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # ISAAC's submit filter requires this line verbatim
export WANDB_MODE=disabled
python3 -c "import sys, numpy, pandas, pyarrow, torch, PIL; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
set -euo pipefail

PRESET="${PRESET:-dinov3_vit7b16_sat}"
MANIFEST_DIR="${MANIFEST_DIR:-$STORE/geo_dedup/dedup_v3_b1000_cities}"  # the null needs the city pass (header); match checks every file came from ONE dir
ARROW="${ARROW:-$STORE/git10m/arrow}"
OUT="${OUT:-$STORE/geo_dedup/content}"
# Controls file (GEO-Bench variants + the synthetic blank images the degenerate-image filter needs).
# `controls` refuses to overwrite: to regenerate, pass CONTROLS=$OUT/${PRESET}_controls_v2.npz to BOTH
# the controls task and STEP=match (match refuses a controls file without the blank variants).
CONTROLS="${CONTROLS:-$OUT/${PRESET}_controls.npz}"
N_PARTS="${N_PARTS:-8}"
NULL_N="${NULL_N:-20000}"
EXCL_N="${EXCL_N:-20000}"
PER_TASK="${PER_TASK:-200}"
STEP="${STEP:-embed}"
mkdir -p "$OUT"
for f in unassessable retained excluded; do
    [ -f "$MANIFEST_DIR/$f.csv" ] || { echo "FATAL: $MANIFEST_DIR/$f.csv missing" >&2; exit 1; }
done

if [ "${SMOKE:-0}" = 1 ]; then
    SO="$OUT/smoke"; rm -rf "$SO"; mkdir -p "$SO"
    python3 -m eval.content_overlap embed --preset "$PRESET" --arrow-dir "$ARROW" --manifest "$MANIFEST_DIR/unassessable.csv" \
        --unique-names --part 0/8 --max-rows 8 --workers 2 --batch 4 --out "$SO/q.npz"
    python3 -m eval.content_overlap embed --preset "$PRESET" --arrow-dir "$ARROW" --manifest "$MANIFEST_DIR/retained.csv" \
        --sample 8 --seed 0 --workers 2 --batch 4 --out "$SO/null.npz"
    for t in m_eurosat m_so2sat; do python3 -m eval.export_geobench --task "${t//_/-}" --out "data/${t}_rgb"; done
    python3 -m eval.content_overlap controls --preset "$PRESET" --tasks m_eurosat m_so2sat --per-task 2 --batch 4 --out "$SO/ctrl.npz"
    python3 - "$SO/ctrl.npz" "$PRESET" <<'PY'
import os, sys, numpy as np
from eval import content_overlap as C
c = np.load(sys.argv[1]); preset = sys.argv[2]
ref, task, split, path, sha = C.load_reference("models/dino_feats", preset, ["m_eurosat", "m_so2sat"])
g = C.identity_gate(c, ref, task, split, path, str(c["split"]))
print("identity gate:", g)
m, _ = C.max_cosine(c["cls"], ref)
for k in C.VARIANTS:
    print(f"  {k:9} max-sim median {np.median(m[c['variant'] == k]):.4f}")
sys.exit(0 if g["pass"] else "CONTENT_SMOKE_FAILED: identity controls do not reproduce the cache")
PY
    echo "CONTENT_SMOKE_OK -> $SO"; exit 0
fi

if [ "$STEP" = match ]; then
    for i in $(seq 0 $((N_PARTS - 1))); do
        [ -f "$OUT/${PRESET}_unassessable_p${i}of${N_PARTS}.npz" ] || { echo "FATAL: part $i missing" >&2; exit 1; }
    done
    python3 -m eval.content_overlap match --preset "$PRESET" --ref-dir models/dino_feats \
        --null "$OUT/${PRESET}_null_retained_n${NULL_N}.npz" --controls "$CONTROLS" \
        --excluded "$OUT/${PRESET}_excluded_loc_n${EXCL_N}.npz" \
        --query "$OUT/${PRESET}_unassessable_p*of${N_PARTS}.npz" --out-dir "$OUT/report_${PRESET}"
    echo "CONTENT_OVERLAP_MATCH_DONE -> $OUT/report_${PRESET}"
    exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
i="${SLURM_ARRAY_TASK_ID:?run as an array job (--array=0-$((N_PARTS + 2)))}"
echo "task $i preset=$PRESET manifests=$MANIFEST_DIR arrow=$ARROW host=$(hostname) $(date -Is)"
if [ "$i" -lt "$N_PARTS" ]; then
    python3 -m eval.content_overlap embed --preset "$PRESET" --arrow-dir "$ARROW" \
        --manifest "$MANIFEST_DIR/unassessable.csv" --unique-names --part "$i/$N_PARTS" \
        --workers "$SLURM_CPUS_PER_TASK" --out "$OUT/${PRESET}_unassessable_p${i}of${N_PARTS}.npz"
elif [ "$i" -eq "$N_PARTS" ]; then
    python3 -m eval.content_overlap embed --preset "$PRESET" --arrow-dir "$ARROW" \
        --manifest "$MANIFEST_DIR/retained.csv" --sample "$NULL_N" --seed 0 \
        --workers "$SLURM_CPUS_PER_TASK" --out "$OUT/${PRESET}_null_retained_n${NULL_N}.npz"
elif [ "$i" -eq $((N_PARTS + 1)) ]; then
    python3 -m eval.content_overlap embed --preset "$PRESET" --arrow-dir "$ARROW" \
        --manifest "$MANIFEST_DIR/excluded.csv" --drop-city-only --sample "$EXCL_N" --seed 0 \
        --workers "$SLURM_CPUS_PER_TASK" --out "$OUT/${PRESET}_excluded_loc_n${EXCL_N}.npz"
elif [ "$i" -eq $((N_PARTS + 2)) ]; then
    for t in m_eurosat m_forestnet m_so2sat m_brick_kiln m_pv4ger m_bigearthnet; do
        python3 -m eval.export_geobench --task "${t//_/-}" --out "data/${t}_rgb"  # tree == manifest
    done
    python3 -m eval.content_overlap controls --preset "$PRESET" --per-task "$PER_TASK" --out "$CONTROLS"
else
    echo "FATAL: task $i beyond N_PARTS + 2" >&2; exit 1
fi
echo "done $(date -Is)"
