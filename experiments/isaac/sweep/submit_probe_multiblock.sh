#!/bin/bash
# ============================================================================
# Kick off the CPU probe sweep over the multi-block FLUX caches, all six datasets:
#   gate     : m_forestnet k=56 on `long` (a real block task; its result is kept)
#   per ds   : probe_multiblock.sbatch array of G tasks (blocks packed j % G), afterok:gate
#              (+ afterok on the dataset's still-running extraction, if its caches are incomplete)
#   per ds   : select_multiblock.sbatch, afterok on that array -> val-grid plot + ONE test use
#   bash experiments/isaac/sweep/submit_probe_multiblock.sh            # dry run
#   bash experiments/isaac/sweep/submit_probe_multiblock.sh --submit
# ============================================================================
set -euo pipefail
_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
cd "$_dir"

SUBMIT=0
# HEAD=tsweight queues the learned timestep-weighting head instead of LR x C (eval/fusion.py):
# its own arm (flux-oneshot-ens8-mb-tsw), so its select is separate. --export=ALL carries both
# to every probe and select job.
export HEAD="${HEAD:-lr}"
[ "$HEAD" = tsweight ] && export ARM="${ARM:-flux-oneshot-ens8-mb-tsw}"
[ "${1:-}" = "--submit" ] && SUBMIT=1
DATASETS=(m_eurosat m_forestnet m_so2sat m_brick_kiln m_pv4ger m_bigearthnet)
BLOCKS="${BLOCKS:-4-8-12-18-19-24-28-33-38-43-48-54-55-56}"
N="$(tr '-' '\n' <<<"$BLOCKS" | wc -l)"
D=experiments/isaac/sweep

complete() {  # all three split caches present with their _meta.json
    local n=0 s f
    for s in train val test; do
        for f in models/"$1"_multiblock_ens8/*/multistep_"$s"_feats_oneshot_g1.0_k"$BLOCKS".npz; do
            [ -f "$f" ] && [ -f "${f%.npz}_meta.json" ] && n=$((n + 1)) && break
        done
    done
    [ "$n" = 3 ]
}
extraction_job() {  # newest pending/running multiblock extraction for this dataset, if any
    # `|| true` inside: under pipefail a non-matching LAST grep would fail the whole pipeline
    # and, through `ej="$(...)"` + set -e, abort the script silently mid-submission.
    squeue -u "$USER" -h -n geobench-multiblock -o "%i" | while read -r j; do
        if sacct -j "$j" -X -n -P --format=JobID,SubmitLine | grep -q "DATASET=$1,"; then echo "$j"; fi
    done | sort -n | tail -1 || true
}
dep_for() {  # afterok on the dataset's still-running extraction, if its caches are incomplete
    complete "$1" && return 0
    local ej
    ej="$(extraction_job "$1")"
    [ -n "$ej" ] || { echo "FATAL: $1 caches incomplete and no extraction job queued" >&2; exit 1; }
    echo ":$ej"
}

# QOS long admits 18 SUBMITTED jobs per user, array tasks included: check the whole plan fits
# BEFORE submitting anything, so a refused sbatch can never leave half a chain behind.
planned=$((1 + 5 * 2 + 3))
queued="$(squeue -u "$USER" -h -q long -r | wc -l)"
if [ $((queued + planned)) -gt 18 ]; then
    echo "FATAL: $queued long jobs already queued + $planned planned > 18 (QOS long MaxSubmitPU)" >&2
    exit 1
fi

# Gate = the LAST block of BLOCKS on m_forestnet (task N-1 with NGROUPS=N runs exactly that block),
# chained on m_forestnet's extraction like every dataset array is.
# Resolve EVERY dataset's extraction dependency first: a failure must stop the script before
# anything is submitted, never after the gate and some arrays are already queued.
declare -A EXTRA
for d in "${DATASETS[@]}"; do EXTRA[$d]="$(dep_for "$d")"; done
gate_dep="${EXTRA[m_forestnet]}"
gate=(sbatch --parsable --array=$((N - 1)) --time=8:00:00  # long: the 7-C forestnet block exceeds short's 3 h cap
    ${gate_dep:+--dependency=afterok${gate_dep}}
    --export=ALL,DATASET=m_forestnet,BLOCKS="$BLOCKS",NGROUPS="$N",GATE=1 "$D/probe_multiblock.sbatch")
echo "gate: ${gate[*]}"
GATE="<gate-jobid>"
if [ "$SUBMIT" = 1 ]; then GATE="$("${gate[@]}" | cut -d_ -f1)"; echo "  -> $GATE"; fi

for d in "${DATASETS[@]}"; do
    extra="${EXTRA[$d]}"
    dep="afterok:$GATE$extra"
    [ -n "$extra" ] && echo "note: $d caches incomplete -> also waiting on extraction job ${extra#:}"
    # QOS long admits 18 SUBMITTED jobs per user (array tasks count): pack each dataset's 14 blocks
    # into G sequential groups -> 1 gate + 5x2 + 3 = 14 FLUX tasks, leaving 4 for the DINO probes.
    G=2; t=5-00:00:00; cpus=8  # 7 blocks per task; forestnet k=56 alone took 2.8 h at 5 C
    if [ "$d" = m_bigearthnet ]; then
        G=3; t=6-00:00:00; cpus=16  # 43 one-vs-rest fits per cell run in parallel (OVR_N_JOBS)
    fi
    arr=(sbatch --parsable --array=0-$((G - 1)) --time="$t" --cpus-per-task="$cpus" --dependency="$dep"
        --export=ALL,DATASET="$d",BLOCKS="$BLOCKS",NGROUPS="$G" "$D/probe_multiblock.sbatch")
    echo "$d probe : ${arr[*]}"
    A="<array-jobid>"
    if [ "$SUBMIT" = 1 ]; then A="$("${arr[@]}")"; echo "  -> $A"; fi
    sel=(sbatch --parsable --dependency="afterok:$A" --export=ALL,DATASET="$d",N_BLOCKS="$N" "$D/select_multiblock.sbatch")
    echo "$d select: ${sel[*]}"
    if [ "$SUBMIT" = 1 ]; then echo "  -> $("${sel[@]}")"; fi
done
[ "$SUBMIT" = 1 ] || echo "(dry run -- add --submit to queue)"
