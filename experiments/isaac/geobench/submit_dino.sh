#!/bin/bash
# ============================================================================
# DINOv3 baseline sweep: for each preset, a GPU extraction array (dino.sbatch) and a CPU
# gate+probe jobs (dino_probe.sbatch), packed under QOS long's 18-job limit and chained on the
# whole extraction array (afterok). One extraction task per eval/features.OFFICIAL key.
#   bash experiments/isaac/geobench/submit_dino.sh            # dry run
#   bash experiments/isaac/geobench/submit_dino.sh --submit   # queue 2 presets x (GPU + CPU)
#   PRESETS="dinov3_vit7b16_sat" bash ... --submit            # one preset
# ============================================================================
set -euo pipefail
_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
cd "$_dir"
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"

SUBMIT=0
[ "${1:-}" = "--submit" ] && SUBMIT=1
read -r -a PRESET_LIST <<<"${PRESETS:-dinov3_vit7b16_web dinov3_vit7b16_sat}"
N="$(python3 -c "from eval.features import OFFICIAL; print(len(OFFICIAL))")"
ARRAY="0-$((N - 1))"
# QOS long admits 18 SUBMITTED jobs per user (array tasks count). Per preset: the extraction
# array (N tasks) + 2 PACKED gate+probe jobs (datasets 0..N-2, and the multi-label m_bigearthnet
# alone), each waiting on the WHOLE extraction array (afterok): a packed job reads several
# datasets' features, so per-task aftercorr would start it too early.
# Budget per QOS: extraction arrays run on ai-tenn (MaxSubmitPU 56), the packed probes on long (18).
plan_long=$(( ${#PRESET_LIST[@]} * 2 ))
plan_gpu=$(( ${#PRESET_LIST[@]} * N ))
q_long="$(squeue -u "$USER" -h -q long -r | wc -l)"
q_gpu="$(squeue -u "$USER" -h -q ai-tenn -r | wc -l)"
if [ $((q_long + plan_long)) -gt 18 ] || [ $((q_gpu + plan_gpu)) -gt 56 ]; then
    echo "FATAL: long $q_long+$plan_long > 18 or ai-tenn $q_gpu+$plan_gpu > 56 (QOS MaxSubmitPU)" >&2
    exit 1
fi
PACKS=("$(seq -s- 0 $((N - 2)))" "$((N - 1))")  # hyphens: --export would split a comma list
for p in "${PRESET_LIST[@]}"; do
    w="$(python3 -c "import sys; from eval.extract_dino import PRESETS; print(PRESETS[sys.argv[1]]['weights'] or '')" "$p")"
    [ -z "$w" ] || [ -f "$w" ] || { echo "FATAL: $p weights missing: $w" >&2; exit 1; }
    ext=(sbatch --parsable --array="$ARRAY" --export=ALL,PRESET="$p" experiments/isaac/geobench/dino.sbatch)
    echo "$p extract: ${ext[*]}"
    EXT_ID="<extract-jobid>"
    if [ "$SUBMIT" = 1 ]; then EXT_ID="$("${ext[@]}")"; echo "  -> $EXT_ID"; fi
    for g in "${PACKS[@]}"; do
        # WIDE_C=1: the same 7-value C grid as the FLUX multi-block sweep (rule 8)
        prb=(sbatch --parsable --array=0 --dependency="afterok:$EXT_ID"
            --export=ALL,PRESET="$p",WIDE_C=1,DATASET_IDXS="$g" experiments/isaac/geobench/dino_probe.sbatch)
        echo "$p probe [$g]: ${prb[*]}"
        if [ "$SUBMIT" = 1 ]; then echo "  -> $("${prb[@]}")"; fi
    done
done
[ "$SUBMIT" = 1 ] || echo "(dry run -- add --submit to queue)"
