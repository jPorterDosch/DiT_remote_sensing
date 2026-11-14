#!/bin/bash
# ============================================================================
# Learned timestep-weighting head (eval/fusion.py, --fusion-grid reg) on the Git-10M adapted arms'
# multi-block caches at the FIXED aligned block K (phase-2 view (a), k=28), 100 % labels: the
# trained-head counterpart of the LR matrix cells, under the same recipe class as SatDiFuser's
# fuser (AdamW lr/wd/epoch selected on the official val split, 40 epochs, cosine). Per arm and
# dataset: slice gate -> eval.sweep block --head tsweight (test SEALED) -> eval.sweep select over
# that ONE block (positive control, then ONE test use). Own arm name flux-<arm>-k<K>-tsw-reg, so
# nothing of the frozen flux-oneshot-ens8-mb-tsw-reg sweep is touched. Re-runnable (done-status).
#
#   ARMS      "|"-separated list of <arm>=<run dir basename under models/git10m_eval> (run dirs hold hyphens), e.g.
#             ARMS="proj=git10m_flux_ca8e6ce7+42+proj-ffcd7963|naive=git10m_flux_ca8e6ce7+42+naive-11e56b1e"
#   DATASETS  hyphen list (default: the six phase-2 tasks);  K (default 28);  OUT (results/eval/sweep_mb)
#   HEAD      tsweight (default) | lr: the sweep's plain LR protocol at block K (each single t + the
#             9-t concat x C_GRID_WIDE, val in the clear, test sealed; arm flux-<arm>-k<K>-lrw) -- the
#             per-timestep probe curve of an adapted arm. CPU only: submit with
#             -p short -q short -t 3:00:00 --gpus=0 --cpus-per-task=16, one dataset per job.
#   SMOKE=1   the first arm x first dataset's full block into a scratch smoke dir, no select (rule 15;
#             --smoke-sizes is for subset-extraction caches, not these). Pass DATASETS=m_forestnet
#             (the smallest task). Submit the full job with --dependency=afterok:<smoke>.
# ============================================================================
#SBATCH --job-name=tsw-k28
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH" WANDB_MODE=offline OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_CONSOLE=off  # keep tracebacks in the SLURM log (the first smoke died with no message)
HEAD="${HEAD:-tsweight}"  # lr: --c-grid wide = protocols.C_GRID_WIDE, the frozen multi-block sweep grid
case "$HEAD" in tsweight) HEAD_ARGS=(--head tsweight --fusion-grid reg); TAG=tsw-reg ;; lr) HEAD_ARGS=(--head lr --c-grid wide); TAG=lrw ;; *) echo "bad HEAD=$HEAD" >&2; exit 1 ;; esac
if [ "$HEAD" = tsweight ]; then
    [ "${DRY_RUN:-0}" = 1 ] || python3 -c "import sys, torch; assert 'DiTF' in sys.prefix; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" || exit 1
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    [ "${DRY_RUN:-0}" = 1 ] || python3 -c "import sys, sklearn; assert 'DiTF' in sys.prefix" || exit 1
    export OVR_N_JOBS="${SLURM_CPUS_PER_TASK:-8}"
fi
# DRY_RUN=1 (login node, seconds): print every python3 call instead of running it. Defined AFTER the PATH
# export so it shadows the DiTF interpreter too.
[ "${DRY_RUN:-0}" = 1 ] && python3() { echo "DRY python3 $*"; }
set -uo pipefail
ARMS="${ARMS:?submit with --export=ALL,ARMS=<arm>=<run dir>[|<arm>=<run dir>...]}"
DATASETS="${DATASETS:-m_eurosat-m_brick_kiln-m_pv4ger-m_so2sat-m_forestnet-m_bigearthnet}"
K="${K:-28}"
OUT="${OUT:-results/eval/sweep_mb}"
if [ "${SMOKE:-0}" = 1 ]; then
    OUT="$STORE/tmp/tsw_k${K}_smoke"; rm -rf "$OUT"; mkdir -p "$OUT"
fi
FAILED=()
one() {  # one <arm> <run dir basename> <dataset>
    local arm="$1" run="$2" d="$3" sweep_arm="flux-$1-k${K}-${TAG}" rc
    local cache_dir="models/git10m_eval/$run/$d"
    local feats="$cache_dir/*/multistep_{split}_feats_oneshot_g1.0_lora-${arm}_k*.npz"
    local expect=(extraction_mode=ONESHOT ensemble_size=8 "weights=flux-dev+lora:$arm")
    [ -d "$cache_dir" ] || { echo "FATAL: no cache dir $cache_dir"; return 1; }
    echo "######## $arm $d k=$K $(date -Is)"
    python3 -m eval.sweep gate --phase slice --dataset "$d" --k "$K" --features "$feats" \
        --out-dir "$OUT" --arm "$sweep_arm" --expect "${expect[@]}" || { echo "FATAL: slice gate $arm $d"; return 1; }
    if [ "${SMOKE:-0}" = 1 ]; then
        python3 -m eval.sweep block --dataset "$d" --k "$K" --arm "$sweep_arm-SMOKE" "${HEAD_ARGS[@]}" \
            --features "$feats" --expect "${expect[@]}" --out-dir "$OUT" \
            || { echo "FATAL: smoke block $arm $d"; return 1; }
        echo "TSW_SMOKE_OK $arm $d"; return 0
    fi
    local res=("$OUT"/sweep-block_"${d}"_"${sweep_arm}"-k"${K}"_*.npz)
    if [ -f "${res[0]}" ]; then
        rc=0; python3 -m eval.sweep done-status "${res[0]}" --dataset "$d" --features "$feats" \
            --cache-dir "$cache_dir" "${HEAD_ARGS[@]}" || rc=$?
        if [ "$rc" = 0 ]; then echo "block done, skipping"; else echo "FATAL: $arm $d existing result not current (code $rc)"; return 1; fi
    else
        python3 -m eval.sweep block --dataset "$d" --k "$K" --arm "$sweep_arm" "${HEAD_ARGS[@]}" \
            --features "$feats" --expect "${expect[@]}" --out-dir "$OUT" || { echo "FATAL: block $arm $d"; return 1; }
    fi
    local sel=("$OUT"/sweep-select_"${d}"_"${sweep_arm}"_*.npz)
    if [ -f "${sel[0]}" ]; then echo "select exists: ${sel[0]} (not re-opening test)"; return 0; fi
    python3 -m eval.sweep select --dataset "$d" --arm "$sweep_arm" \
        --blocks "$OUT/sweep-block_${d}_${sweep_arm}-k${K}_*.npz" --out-dir "$OUT" || { echo "FATAL: select $arm $d"; return 1; }
    echo "TSW_SELECT_OK $arm $d"
}
for spec in ${ARMS//|/ }; do
    arm="${spec%%=*}"; run="${spec#*=}"
    [ "$arm" != "$spec" ] || { echo "FATAL: ARMS entry $spec is not <arm>=<run dir>"; exit 1; }
    for d in ${DATASETS//-/ }; do
        one "$arm" "$run" "$d" || FAILED+=("$arm/$d")
        [ "${SMOKE:-0}" = 1 ] && break 2
    done
done
echo "######## done $(date -Is) failed: ${FAILED[*]:-none}"
[ "${#FAILED[@]}" = 0 ] && echo "TSW_ALL_OK"
