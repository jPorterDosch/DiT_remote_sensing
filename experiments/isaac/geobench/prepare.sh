#!/bin/bash
# ============================================================================
# Download + export GEO-Bench classification tasks on ISAAC (login node, once per task):
#   bash experiments/isaac/geobench/prepare.sh                      # every task below
#   bash experiments/isaac/geobench/prepare.sh m-so2sat m-pv4ger     # a subset
# Per task: fetch the Zenodo record (md5-checked) into $RAW/<key>_meta, export the RGB tree
# to data/<key>_rgb (data/ is a scratch symlink, scratch_env.sh), and REFUSE unless the
# export matches the committed eval/manifests/<key>.json -- i.e. the images and labels are
# identical to the ones the workstation gates ran on. RM_RAW=1 (default) then deletes the
# sample .hdf5 files; a later re-run re-downloads. An existing data/<key>_rgb is only verified.
# m-eurosat works here too (verified against its manifest); its own prepare.sh is kept.
# Requires geobench installed WITHOUT its over-pinned deps (see m_eurosat/prepare.sh header).
# ============================================================================
set -euo pipefail

_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # data/ -> Lustre scratch

RAW="${RAW:-/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing/datasets}"
TASKS=("$@")
[ ${#TASKS[@]} -gt 0 ] || TASKS=(m-forestnet m-so2sat m-brick-kiln m-pv4ger m-bigearthnet)

for t in "${TASKS[@]}"; do
    k="${t//-/_}"
    if [ -d "data/${k}_rgb" ]; then  # existing tree (e.g. m-eurosat's): verify, never overwrite
        echo "=== $t: verifying existing data/${k}_rgb against its manifest"
        python3 -m eval.export_geobench --task "$t" --out "data/${k}_rgb"; continue
    fi
    echo "=== $t $(date -Is)"
    # export to .tmp and rename only after the manifest check passed: a failed export never
    # leaves a data/<key>_rgb that a re-run (or a probe) would take as complete
    rm -rf "data/${k}_rgb.tmp"
    python3 -m eval.export_geobench --task "$t" --dataset-dir "$RAW/${k}_meta" --out "data/${k}_rgb.tmp" \
        --download $([ "${RM_RAW:-1}" = 1 ] && echo --rm-raw)
    mv "data/${k}_rgb.tmp" "data/${k}_rgb"
done
# DINOv3 hub code, cached on scratch (TORCH_HOME) for compute nodes that may lack internet;
# torch.hub falls back to this cache offline. The gated checkpoints are placed by hand in
# ditf_models/dinov3/ under their original filenames (eval/extract_dino.PRESETS).
python3 -c "import torch; torch.hub.list('facebookresearch/dinov3')" >/dev/null  # code only, no model
echo "prep complete: ${TASKS[*]} (+ DINOv3 hub code)"
