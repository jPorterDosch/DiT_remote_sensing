#!/bin/bash
# Body of arm_full.sbatch (read at job start, see the launcher). Runs the two PROVEN scripts
# unchanged, inside one allocation:
#   1. experiments/isaac/git10m/train.sbatch   production exports of the seed-42 runs
#      (ARM=proj, index_all_cities_unique, 20k steps, batch 2 x accum 4, ckpt every 500),
#      plus the arm's own ALIGN_TARGET / SEM_T_WEIGHTS and SEED, on GPU 0
#   2. experiments/isaac/git10m/extract_ckpt.sbatch   one dataset per child, longest-first over
#      all GPUs (H100 hours: so2sat 22.5, bigearthnet 22.1, eurosat 18.4, brick_kiln 17.2,
#      pv4ger 14.0, forestnet 9.1 -> 31 h wall on 4 GPUs)
#   3. matrix cells (CPU) submitted to short / campus from here, spec written per arm.
# Env: TAG (semt-lo | semt-hi | web | weak), SEED, ALIGN_TARGET and/or SEM_T_WEIGHTS, SMOKE (0|1).
set -euo pipefail
_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -e "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"; cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh" || exit 1   # STORE, caches on Lustre
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"   # the inline python checks below (stdlib only, but pin the env anyway)
mkdir -p logs
JOB="${SLURM_JOB_ID:?run under sbatch}"
TAG="${TAG:?TAG=semt-lo|semt-hi|web|weak}"; SEED="${SEED:?SEED=42|43}"; SMOKE="${SMOKE:-0}"
log() { echo "$(date -Is) [arm-full $TAG s$SEED] $*"; }

# --- arm definition: exactly the seed-42 reference runs' flags, nothing else may differ ------------
case "$TAG" in
    semt-lo) [ "${SEM_T_WEIGHTS:-}" = "1 1 0 0 0 0 0 0 0" ] && [ -z "${ALIGN_TARGET:-}" ] || { echo "FATAL: semt-lo needs SEM_T_WEIGHTS='1 1 0 0 0 0 0 0 0' and no ALIGN_TARGET (ref run proj-0f0ef65b)" >&2; exit 1; } ;;
    semt-hi) [ "${SEM_T_WEIGHTS:-}" = "0 0 0 0 0 0 1 1 1" ] && [ -z "${ALIGN_TARGET:-}" ] || { echo "FATAL: semt-hi needs SEM_T_WEIGHTS='0 0 0 0 0 0 1 1 1' and no ALIGN_TARGET (ref run proj-2d4482ff)" >&2; exit 1; } ;;
    web)     [ "${ALIGN_TARGET:-}" = dinov3_vit7b16_web ] && [ -z "${SEM_T_WEIGHTS:-}" ] || { echo "FATAL: web needs ALIGN_TARGET=dinov3_vit7b16_web and no SEM_T_WEIGHTS (ref run proj-eed4023a)" >&2; exit 1; } ;;
    weak)    [ "${ALIGN_TARGET:-}" = dinov3_vitl16_web ] && [ -z "${SEM_T_WEIGHTS:-}" ] || { echo "FATAL: weak needs ALIGN_TARGET=dinov3_vitl16_web and no SEM_T_WEIGHTS" >&2; exit 1; } ;;
    *) echo "FATAL: TAG=$TAG" >&2; exit 1 ;;
esac
for v in STOP_STEP TARGET_SHUFFLE RANK LORA_ALPHA LR ALIGN_WEIGHT T_WEIGHTS K HFLIP CD EXTRA RESUME; do
    [ -z "${!v:-}" ] || { echo "FATAL: $v is set in the environment; the reference runs used the defaults" >&2; exit 1; }
done
case "$TAG" in web) TEACHER_ROW=dinov3-web ;; weak) TEACHER_ROW=dinov3-vitl16-web ;; *) TEACHER_ROW=dinov3-sat ;; esac
# weak: scored against the teacher it trained with (ViT-L/16 web, eval/matrix_specs/phase11_weak_teacher.json),
# not against dinov3-web (7B), which it never saw (review 2026-11-12, item 6).

# --- GPUs -------------------------------------------------------------------------------------------
IFS=',' read -r -a GPU <<<"${CUDA_VISIBLE_DEVICES:?no GPUs visible}"
NG="${#GPU[@]}"; [ "$NG" -ge 1 ] || { echo "FATAL: no GPUs" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
log "host=$(hostname) gpus=${GPU[*]} smoke=$SMOKE target=${ALIGN_TARGET:-dinov3_vit7b16_sat} sem_t=${SEM_T_WEIGHTS:-}"

# --- 1. train ---------------------------------------------------------------------------------------
TLOG="logs/git10m-train-$JOB.out"   # train.sbatch's split-eval re-reads logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out
log "train -> $TLOG"
# SLURM_ARRAY_TASK_ID=train: scratch_env.sh keys the node-local TMPDIR (and its rm -rf EXIT trap) on
# job id + array id, so the child must not share this shell's /tmp/${USER}_${JOB}; the extraction
# children get their dataset index the same way.
if ! env SLURM_JOB_NAME=git10m-train SLURM_ARRAY_TASK_ID=train SLURM_CPUS_PER_TASK=8 CUDA_VISIBLE_DEVICES="${GPU[0]}" \
        ARM=proj SEED="$SEED" INDEX="$STORE/git10m/index_all_cities_unique.npz" \
        TRAIN_STEPS=20000 BATCH_SIZE=2 GRAD_ACCUM=4 CKPT_EVERY=500 SMOKE="$SMOKE" \
        bash experiments/isaac/git10m/train.sbatch >"$TLOG" 2>&1; then
    log "FATAL: train.sbatch exited non-zero; tail of $TLOG:"; tail -20 "$TLOG"; exit 1
fi
grep -q '^GIT10M_TRAIN_OK' "$TLOG" || { log "FATAL: no GIT10M_TRAIN_OK in $TLOG"; tail -20 "$TLOG"; exit 1; }
RUN_DIR="$(grep -E '^\[align\] done: .* -> ' "$TLOG" | tail -1 | sed 's/.* -> //')"
[ -d "$RUN_DIR" ] || { log "FATAL: run dir '$RUN_DIR' from the [align] done line does not exist"; exit 1; }
RUN="$(basename "$RUN_DIR")"
[[ "$RUN" == git10m_flux_*"+$SEED+proj-"* ]] || { log "FATAL: run dir $RUN is not a +$SEED+proj- run"; exit 1; }
# The run's own meta.json must carry this arm's definition (never trust the env alone).
PYTHONPATH="src:." python3 - "$RUN_DIR/meta.json" "${ALIGN_TARGET:-dinov3_vit7b16_sat}" "${SEM_T_WEIGHTS:-}" "$SEED" <<'PY' || exit 1
import json, sys
m = json.load(open(sys.argv[1])); al = m["cfg"]["align"]
want_t = [float(x) for x in sys.argv[3].split()] or None
got_t = al.get("sem_t_weights"); got_t = [float(x) for x in got_t] if got_t else None
bad = {k: (g, w) for k, g, w in [("target", al.get("target"), sys.argv[2]), ("sem_t_weights", got_t, want_t),
                                 ("seed", int(m["cfg"]["seed"]), int(sys.argv[4])), ("arm", al.get("arm"), "proj")] if g != w}
print("meta.json arm check:", "OK" if not bad else f"MISMATCH {bad}")
sys.exit(1 if bad else 0)
PY
CKPTS=("$RUN_DIR"/checkpoints/lora_last_step*.pt)
[ "${#CKPTS[@]}" -eq 1 ] && [ -f "${CKPTS[0]}" ] || { log "FATAL: expected exactly one lora_last checkpoint, got: ${CKPTS[*]}"; exit 1; }
CKPT="${CKPTS[0]}"
log "train OK: $RUN  ckpt=$CKPT"

# --- 2. extract, longest first, greedy over GPUs -------------------------------------------------
# dataset index -> H100 hours measured on jobs 6477983_0..5 (extract_ckpt.sbatch DATASETS order)
ORDER=(3 5 0 1 2 4); HOURS=(22.5 22.1 18.4 17.2 14.0 9.1)
declare -a LOAD PLAN; for ((g = 0; g < NG; g++)); do LOAD[g]=0; PLAN[g]=""; done
for ((i = 0; i < ${#ORDER[@]}; i++)); do
    best=0; for ((g = 1; g < NG; g++)); do awk -v a="${LOAD[g]}" -v b="${LOAD[best]}" 'BEGIN{exit !(a<b)}' && best=$g; done
    PLAN[best]="${PLAN[best]} ${ORDER[i]}"; LOAD[best]=$(awk -v a="${LOAD[best]}" -v b="${HOURS[i]}" 'BEGIN{print a+b}')
done
for ((g = 0; g < NG; g++)); do log "gpu ${GPU[g]}: datasets${PLAN[g]} (~${LOAD[g]} H100 h)"; done
pids=()
for ((g = 0; g < NG; g++)); do
    (
        for idx in ${PLAN[g]}; do
            ELOG="logs/git10m-eval-${JOB}_$idx.out"
            env SLURM_ARRAY_TASK_ID="$idx" SLURM_JOB_NAME=git10m-eval SLURM_CPUS_PER_TASK=8 CUDA_VISIBLE_DEVICES="${GPU[g]}" \
                CKPT="$CKPT" SMOKE="$SMOKE" bash experiments/isaac/git10m/extract_ckpt.sbatch >"$ELOG" 2>&1 \
                || echo "$(date -Is) child idx=$idx exited non-zero (see $ELOG)"
        done
    ) &
    pids+=($!)
done
wait "${pids[@]}"
fail=0
for idx in 0 1 2 3 4 5; do
    ELOG="logs/git10m-eval-${JOB}_$idx.out"
    if grep -q '^GIT10M_EVAL_OK' "$ELOG"; then log "extract OK idx=$idx: $(grep '^GIT10M_EVAL_OK' "$ELOG" | tail -1)"
    else log "extract FAILED idx=$idx: $(tail -3 "$ELOG" | tr '\n' ' | ')"; fail=1; fi
done
[ "$fail" = 0 ] || { log "FATAL: extraction incomplete; re-run the failed indices with extract_ckpt.sbatch CKPT=$CKPT (resume-safe)"; exit 1; }

# --- 3. matrix cells (CPU) ----------------------------------------------------------------------------
if [ "$SMOKE" = 1 ]; then log "SMOKE: caches under models/git10m_eval_smoke/$RUN, no cells. ARM_FULL_OK $TAG s$SEED smoke"; exit 0; fi
SPEC="eval/matrix_specs/phase10_${TAG}_s${SEED}.json"
PYTHONPATH="src:." python3 - "$SPEC" "$TAG" "$SEED" "$RUN" "$TEACHER_ROW" "${ALIGN_TARGET:-dinov3_vit7b16_sat}" "${SEM_T_WEIGHTS:-}" "$JOB" <<'PY' || exit 1
import json, sys, os
spec, tag, seed, run, teacher, target, semt, job = sys.argv[1:]
base = json.load(open("eval/matrix_specs/phase7_semt.json"))          # fractions / seeds / c_grid / datasets as phase 2
teachers = {m["name"]: m for s in ("phase7_semt", "phase5_web_sweep", "phase11_weak_teacher") for m in json.load(open(f"eval/matrix_specs/{s}.json"))["models"] if m["kind"] == "dino"}
if teacher not in teachers:
    sys.exit(f"FATAL: teacher row {teacher!r} is in none of the teacher specs ({sorted(teachers)})")
model = {"name": f"flux-proj-{tag}-s{seed}", "kind": "flux", "teacher": teacher, "view": None,
         "features": f"models/git10m_eval/{run}/{{dataset}}/*/multistep_{{split}}_feats_oneshot_g1.0_lora-proj_k*.npz",
         "expect": {"extraction_mode": "ONESHOT", "ensemble_size": 8, "weights": "flux-dev+lora:proj", "k": 28}}
out = {"_comment": [f"Phase 10 (supplement, written by arm_full.sh in job {job}): {tag} arm at training seed {seed}, "
                    f"ARM=proj, target {target}, sem_t_weights {semt or 'default'}, k=28 fixed as in phase 5/7; all six datasets; "
                    f"task index = 6 * model + dataset (model 1 = this arm -> tasks 6-11)."],
       "fractions": base["fractions"], "seeds": base["seeds"], "c_grid": base["c_grid"], "datasets": base["datasets"],
       "models": [teachers[teacher], model]}
if os.path.exists(spec):
    old = json.load(open(spec)); old.pop("_comment", None); new = dict(out); new.pop("_comment", None)
    assert old == new, f"{spec} exists with different content"
else:
    json.dump(out, open(spec, "w"), indent=1); open(spec, "a").write("\n")
print("spec", spec, "model", model["name"], "features", model["features"])
PY
log "cells: SPEC=$SPEC tasks 6-10 on short, 11 (bigearthnet) on campus"
c1=$(sbatch --parsable --export=ALL,PART=all,SPEC="$SPEC" -p short -q short -t 3:00:00 --cpus-per-task=16 --array=6-10 experiments/isaac/eval/matrix.sbatch 2>&1) \
    && log "cells 6-10 -> $c1" || log "cells 6-10 NOT submitted ($c1); run by hand: sbatch --export=ALL,PART=all,SPEC=$SPEC -p short -q short -t 3:00:00 --cpus-per-task=16 --array=6-10 experiments/isaac/eval/matrix.sbatch"
c2=$(sbatch --parsable --export=ALL,PART=all,SPEC="$SPEC" -p campus -q campus -t 8:00:00 --cpus-per-task=16 --array=11 experiments/isaac/eval/matrix.sbatch 2>&1) \
    && log "cell 11 -> $c2" || log "cell 11 NOT submitted ($c2); run by hand: sbatch --export=ALL,PART=all,SPEC=$SPEC -p campus -q campus -t 8:00:00 --cpus-per-task=16 --array=11 experiments/isaac/eval/matrix.sbatch"
log "ARM_FULL_OK $TAG s$SEED run=$RUN ckpt=$CKPT spec=$SPEC"
