#!/bin/bash
# ============================================================================
# Git-10M LoRA adaptation of FLUX.1-dev through the pipeline's training task:
#   run.py --task finetune-diffusion --dataset.name git10m --align.arm fm|naive|proj
# Flow matching + semantic alignment of block-k input tokens to a frozen teacher
# (config_types.AlignConfig, src/tasks/align.py; README.md next to this file). The corpus has no
# labels, so the task trains no probe head; evaluation of a checkpoint is the ordinary
#   run.py --task extract --lora-checkpoint <run>/checkpoints/lora_last_*.pt ...
# per GEO-Bench task, then eval.matrix (the cache stamps weights="flux-dev+lora:<arm>").
#
#   INDEX  index npz, ONE ROW PER IMAGE (build-index drops the repeated-caption twin rows; an index
#          built before that rule is refused by the dataset): $STORE/git10m/index_all_cities_unique.npz
#          (retained + the 2.05 M unlocatable rows [mentor 2026-10-13: keep them], minus So2Sat's
#          52 city ROIs [decision 2026-10-14]: the index for the real runs) |
#          index_all_unique.npz / index_retained_unique.npz (no city boxes; ablation of the city
#          ROIs, PROTOCOL.md §7b) | index_retained_cities_unique.npz. Row counts: the
#          .report.json next to each (rows, duplicate_rows_dropped).
#   ARM    fm | naive | proj | sem (semantic gradient only, FM measured)   ALIGN_WEIGHT (0.5)   ALIGN_TARGET (dinov3_vit7b16_sat)
#   STOP_STEP  naive/proj only: the semantic gradient stops reaching the LoRA after this many steps
#              (alignment on/off branching, HASTE-style); own align hash, so its own run dir
#   ALIGN_FEAT ditf (default: massive channels zeroed, LayerNorm, the block's scale/shift) | ln (zeroed +
#              LayerNorm, no scale/shift) | raw (plain tokens, REPA) -- the two ablations of the default
#   TARGET_SHUFFLE=1  the shuffled-teacher CONTROL of the arm (--align.target-shuffle batch: every image
#              aligned to another image's DINO tokens, everything else identical; run it with the SAME
#              ARM / ALIGN_FEAT / SEED as the arm it controls -- a control that gains downstream means
#              the instrument is broken, CLAUDE.md rule 2). New align hash, so its own run dir.
#   CD=0       keep the massive-activation channels (--no-cd) in the aligned features AND in every
#              extraction from the checkpoint (default 1 = --cd, the matrix protocol); own config hash
#   K          aligned block (default 28: the semantic loss reaches blocks 0-27; 56 = the last block, so
#              the whole backbone is adapted by the semantic loss). Part of the config hash: own run dir.
#   RANK / LORA_ALPHA  LoRA rank (4) and alpha (16). LoRA scales the update by alpha/rank, so a capacity
#              sweep that keeps the update scale fixed passes RANK=16 LORA_ALPHA=64 (ratio 4 as at rank 4).
#   SEM_T_WEIGHTS  "w1 ... w9" alignment-loss multiplier per training timestep (align.sem_t_weights;
#                  normalised by the expected weight under T_WEIGHTS so E[w] = 1; the applied, weighted
#                  loss is logged as train/align_loss_applied, train/align_loss stays the unweighted cosine)
#            noise-restricted alignment, e.g. "0 0 0 0 0 0 1 1 1" = semantic gradient only at t >= 580);
#            unset = 1 everywhere (unchanged identity)
#   T_WEIGHTS  sampling weights over the 9 training timesteps 100..900 (default "1 1 1 1 1 1 1 0.5 0.5":
#              the two high-noise timesteps the matrix probes get half the share of the semantic ones;
#              validation / measurement grids are fixed, not sampled)
#   SMOKE=1  shard-0 index (index_smoke_s0_unique.npz), 32 rows, 4 steps, tiny passes, holdout every
#            100th row (the default 10,000 leaves 2 val rows on one shard; FID needs fid_n): run
#            first, on any index or code change. Separate save dir (models/git10m_smoke), never quotable.
#   SIZE=1   throughput sizing: 30 steps at BATCH_SIZE x GRAD_ACCUM, no generation; read img/s and
#            peak_mem_gb from the log / meta.json, then pick BATCH_SIZE for the real run
#   full     TRAIN_STEPS, BATCH_SIZE, GRAD_ACCUM, CKPT_EVERY required; VAL_EVERY (500), MEAS_EVERY
#            (2000), WARMUP_STEPS (100), PROJ_WARMUP (100), LR (1e-3), SEED (42), HFLIP (1),
#            FID_N (600: FID/KID of fid_n fixed-seed samples vs the first fid_n strided rows of the
#            val holdout, which has 656 rows under the default VAL_INDEX -- FID_N must not exceed
#            it; eval/gen_metrics.py; ~fid_n x 28 sampler steps), FID_STEP0=1 on ONE arm scores the
#            pretrained model too (identical in every arm)
#   RESUME=<run dir>/ckpt.pt continues an interrupted full run (same settings; run.py refuses an
#            existing run dir otherwise). EXTRA="..." appends run.py flags.
#   EXTRA="--align.virtual-steps '<run dir>/checkpoints/lora_snap_step*.pt'" with the SAME exports as the
#            run: held-out virtual SGD steps along g_FM / g_SEM / their sums at each listed checkpoint
#            (comma-separated paths or globs), writes <run dir>/virtual_steps_<steps>[_pert].json (never overwritten), VIRTUAL_STEPS_OK.
#            --align.vs-grad-batches N (default 32) sets how many training batches form the gradient;
#            --align.vs-per-t also steps along the single-noise-level gradients g_FM@t / g_SEM@t.
#   EXTRA="--align.grad-geometry '<run dir>/checkpoints/lora_*.pt'" with the SAME exports as the run: the
#            per-batch FM / semantic LoRA gradients on vs_grad_batches training batches x meas_t as one
#            Gram matrix per block, <run dir>/grad_geometry_s<step>.npz (never overwritten), GRAD_GEOMETRY_OK;
#            figures: experiments/figs/grad_geometry.py. ~15 min per checkpoint on an H100 (debug QOS).
#   EXTRA="--align.split-eval <run dir>/checkpoints/lora_last_*.pt" with the SAME exports as the run
#            (incl. SEED / FID_STEP0 / K / RANK ...: the run name must resolve to the existing dir) =
#            the backbone / projector split of the teacher cosine (config_types.AlignConfig.split_eval):
#            no training, two validation passes, writes <run dir>/split_eval.json, SPLIT_EVAL_OK in
#            the log. ~15 min on one A40 (campus-gpu-bigmem; FLUX + ViT-7B do not fit a V100S).
#            Run it for every finished alignment run (mentor request 2026-10-22).
#
#   sbatch --export=ALL,SMOKE=1,ARM=proj experiments/isaac/git10m/train.sbatch
#   for B in 4 8 16; do sbatch --export=ALL,SIZE=1,ARM=proj,BATCH_SIZE=$B,GRAD_ACCUM=1,INDEX=$STORE/git10m/index_all_cities_unique.npz experiments/isaac/git10m/train.sbatch; done
#   sbatch --export=ALL,ARM=proj,INDEX=$STORE/git10m/index_all_cities_unique.npz,TRAIN_STEPS=20000,BATCH_SIZE=8,GRAD_ACCUM=2,CKPT_EVERY=500 experiments/isaac/git10m/train.sbatch
#
# Run dir: models/git10m/git10m_flux_<hash>+<seed>+<arm>-<8hex>/ (ckpt.pt, checkpoints/, gen/,
# stats.npz, val.jsonl, meta.json, conflict_gate.txt). MEMORY: FLUX bf16 24 GB + DINOv3 ViT-7B
# 13 GB + two LoRA backwards (49 GB at batch 2 on H100, 2026-10-11 smoke). ai-tenn MaxWall 3 days.
# ============================================================================
#SBATCH --job-name=git10m-train
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done  # -e: worktrees
PROJECT_ROOT="$_dir"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1  # caches + models/data/logs on Lustre scratch
source "$PROJECT_ROOT/experiments/_common.sh" || { echo "FATAL: _common.sh" >&2; exit 1; }
setup_environment
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
python3 -c "import sys, numpy, torch, pyarrow, tyro; assert sys.version_info[:2] >= (3, 11) and 'DiTF' in sys.prefix, sys.prefix" \
    || { echo "FATAL: bad env: $(command -v python3) $(python3 -V 2>&1)" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
set -euo pipefail

ARM="${ARM:?set ARM=fm|naive|proj via --export}"
case "$ARM" in fm|naive|proj|sem) ;; *) echo "bad ARM=$ARM" >&2; exit 1 ;; esac
ARROW="$STORE/git10m/arrow"
if [ "${SMOKE:-0}" = 1 ]; then
    INDEX="$STORE/git10m/index_smoke_s0_unique.npz"  # always the smoke index: a leftover INDEX in a --export=ALL shell would smoke on 9.4 M rows
else
    INDEX="${INDEX:?set INDEX=<index npz> via --export (index_all_cities_unique.npz for the real runs | index_all_unique.npz | index_retained_unique.npz)}"
    [ "$(ls "$ARROW"/data-*-of-00787.arrow 2>/dev/null | wc -l)" -eq 787 ] \
        || { echo "FATAL: $ARROW holds $(ls "$ARROW"/data-*-of-00787.arrow 2>/dev/null | wc -l) of 787 shards (convert.sbatch)" >&2; exit 1; }
fi
[ -f "$INDEX" ] || { echo "FATAL: $INDEX missing (python -m datasets.git10m build-index ...)" >&2; exit 1; }
# val = the retained holdout of the MOST RESTRICTIVE index, for every run, so val.jsonl is comparable
# between INDEX choices (v2 vs the city pass hold out different retained sets). Smoke: its own index.
if [ "${SMOKE:-0}" = 1 ]; then VAL_INDEX="$INDEX"; else VAL_INDEX="${VAL_INDEX:-$STORE/git10m/index_retained_cities_unique.npz}"; fi
[ -f "$VAL_INDEX" ] || { echo "FATAL: $VAL_INDEX missing" >&2; exit 1; }
PYTHONPATH="$PROJECT_ROOT/src" python3 -m datasets.git10m check --arrow-dir "$ARROW" --index "$INDEX" | tail -1 | grep -q GIT10M_CHECK_OK \
    || { echo "FATAL: index / shard check failed" >&2; exit 1; }

HFLIP_FLAG=(--git10m.hflip); [ "${HFLIP:-1}" = 1 ] || HFLIP_FLAG=(--git10m.no-hflip)
CD_FLAG=--cd; [ "${CD:-1}" = 1 ] || CD_FLAG=--no-cd
ARGS=(--task finetune-diffusion --model.name flux --model.ensemble-size 1
      --dataset.name git10m --dataset.path "$ARROW"
      --git10m.index "$INDEX" --git10m.val-index "$VAL_INDEX" "${HFLIP_FLAG[@]}"
      --align.arm "$ARM" --align.target "${ALIGN_TARGET:-dinov3_vit7b16_sat}" --align.weight "${ALIGN_WEIGHT:-0.5}"
      --align.proj-scope block --align.meas-t 100 260 420 580 740 900 --align.feat "${ALIGN_FEAT:-ditf}"
      --align.t-weights ${T_WEIGHTS:-1 1 1 1 1 1 1 0.5 0.5}
      ${SEM_T_WEIGHTS:+--align.sem-t-weights $SEM_T_WEIGHTS}
      --img-size 224 224 --t 100 180 260 340 420 500 580 740 900 --k "${K:-28}"  # = the matrix caches' t grid: every probed t is trained on
      --guidance-scale 1.0 --mim-loss-weight 0.0 --mask-ratio 0.0
      --lora-rank "${RANK:-4}" --lora-alpha "${LORA_ALPHA:-16.0}" --lora-dropout 0.0 --wrap-output "$CD_FLAG"
      --finetune-lr "${LR:-1e-3}" --use-gradient-accumulation --label-fraction 1.0
      --num-workers 8 --seed "${SEED:-42}" --log-train-steps 10)
if [ "${SMOKE:-0}" = 1 ]; then
    ARGS+=(--max-samples 32 --max-train-steps 4 --warmup-steps 1 --align.proj-warmup-steps 4
           --batch-size "${BATCH_SIZE:-2}" --gradient-accumulation-steps "${GRAD_ACCUM:-2}" --align.meas-batches 2 --log-val-steps 2
           --align.gen-n 2 --align.gen-steps 4 --align.ckpt-every 2 --align.fid-n 8 --align.fid-step0 --align.fid-subsets 5
           --align.snapshot-every 2 --align.keep-ckpt-every 2 --align.feat-stats
           --git10m.holdout-every 100
           --save-dir "$PROJECT_ROOT/models/git10m_smoke")
elif [ "${SIZE:-0}" = 1 ]; then
    ARGS+=(--max-train-steps 30 --warmup-steps 5 --align.proj-warmup-steps 10
           --batch-size "${BATCH_SIZE:?}" --gradient-accumulation-steps "${GRAD_ACCUM:-1}"
           --align.meas-batches 2 --log-val-steps 30 --align.no-gen
           --save-dir "$PROJECT_ROOT/models/git10m_size")
else
    ARGS+=(--max-train-steps "${TRAIN_STEPS:?}" --warmup-steps "${WARMUP_STEPS:-100}"
           --align.proj-warmup-steps "${PROJ_WARMUP:-100}"
           --batch-size "${BATCH_SIZE:?}" --gradient-accumulation-steps "${GRAD_ACCUM:?}"
           --align.meas-batches 8 --align.meas-every "${MEAS_EVERY:-2000}" --log-val-steps "${VAL_EVERY:-500}"
           --align.ckpt-every "${CKPT_EVERY:?}" --align.gen-n 4 --align.gen-steps 28
           --align.snapshot-every "${SNAPSHOT_EVERY:-2000}" --align.keep-ckpt-every "${KEEP_CKPT_EVERY:-5000}" --align.feat-stats
           --align.fid-n "${FID_N:-600}"
           --save-dir "$PROJECT_ROOT/models/git10m")
    [ "${FID_STEP0:-0}" = 1 ] && ARGS+=(--align.fid-step0)
    [ -n "${RESUME:-}" ] && ARGS+=(--align.resume "$RESUME")
fi
[ "${TARGET_SHUFFLE:-0}" = 1 ] && ARGS+=(--align.target-shuffle batch)
[ -n "${STOP_STEP:-}" ] && ARGS+=(--align.stop-step "$STOP_STEP")
# shellcheck disable=SC2206
EXTRA_ARGS=(${EXTRA:-})
echo "arm=$ARM stop_step=${STOP_STEP:-0} index=$INDEX smoke=${SMOKE:-0} size=${SIZE:-0} shuffle=${TARGET_SHUFFLE:-0} cd=${CD:-1} k=${K:-28} rank=${RANK:-4} alpha=${LORA_ALPHA:-16.0} weight=${ALIGN_WEIGHT:-0.5} batch=${BATCH_SIZE:-2}x${GRAD_ACCUM:-2} wandb=$WANDB_MODE args: ${ARGS[*]} ${EXTRA_ARGS[*]:-}"
python3 "$PROJECT_ROOT/run.py" "${ARGS[@]}" "${EXTRA_ARGS[@]}"

# Every full alignment run ends with its backbone / projector split (AlignConfig.split_eval, user
# decision 2026-10-22): a second run.py pass on the final checkpoint, same arguments minus
# --align.resume, ~15 min. SPLIT_EVAL=0 skips it; a run that IS a split-eval (EXTRA) does not recurse.
if [ "${SMOKE:-0}" != 1 ] && [ "${SIZE:-0}" != 1 ] && [ "${SPLIT_EVAL:-1}" = 1 ] \
   && [[ " ${EXTRA:-} " != *"--align.split-eval"* ]] && [[ " ${EXTRA:-} " != *"--align.virtual-steps"* ]] \
   && [[ " ${EXTRA:-} " != *"--align.grad-geometry"* ]]; then
    LOG="$PROJECT_ROOT/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
    run_dir="$(grep -E '^\[align\] done: .* -> ' "$LOG" | tail -1 | sed 's/.* -> //')"
    [ -n "$run_dir" ] && [ -d "$run_dir" ] || { echo "FATAL: split-eval: no '[align] done ... -> <run dir>' line in $LOG" >&2; exit 1; }
    ck="$run_dir/checkpoints/lora_last_step$(printf %08d "$TRAIN_STEPS").pt"
    [ -f "$ck" ] || { echo "FATAL: split-eval: $ck missing" >&2; exit 1; }
    SARGS=(); skip=0
    for a in "${ARGS[@]}"; do
        if [ "$skip" = 1 ]; then skip=0; continue; fi
        if [ "$a" = "--align.resume" ]; then skip=1; continue; fi
        SARGS+=("$a")
    done
    echo "split-eval: $ck"
    python3 "$PROJECT_ROOT/run.py" "${SARGS[@]}" "${EXTRA_ARGS[@]}" --align.split-eval "$ck"
fi
echo "GIT10M_TRAIN_OK"
