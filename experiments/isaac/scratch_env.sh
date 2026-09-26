#!/bin/bash
# ============================================================================
# experiments/isaac/scratch_env.sh -- keep bulky ISAAC I/O off the home quota.
# Sourced (after PROJECT_ROOT is set) by every ISAAC job and helper:
#
#   source "$PROJECT_ROOT/experiments/isaac/scratch_env.sh"
#
# 1. Caches -> Lustre scratch: HF (T5-XXL/CLIP text encoders, ~10 GB), torch hub (DINO
#    weights), wandb (offline runs + artifact staging), pip, TMPDIR.
# 2. The repo's bulky output dirs become SYMLINKS into scratch, so every repo-relative
#    path (--save-dir models/..., data/m_eurosat_rgb, eval.probe --features models/...,
#    tests/test_eval_gates.py, #SBATCH --output=logs/...) keeps working unchanged.
#    A dir that already exists as a real, non-empty directory is NEVER moved
#    automatically: the job stops and prints the one-time migration command.
#    results/ (small per-image vectors, the thing worth keeping) stays in home.
#
# Scratch is not backed up and may be purged; anything that must survive goes to
# results/ or is re-derivable from the logged commands.
# ============================================================================

STORE="${STORE:-/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing}"
STORE_LINKS=(models data logs ditf_models)

export HF_HOME="$STORE/cache/huggingface"
export TORCH_HOME="$STORE/cache/torch"
export XDG_CACHE_HOME="$STORE/cache"
export PIP_CACHE_DIR="$STORE/cache/pip"
export WANDB_DIR="$STORE"
export WANDB_CACHE_DIR="$STORE/cache/wandb"
export WANDB_DATA_DIR="$STORE/cache/wandb-data"
export TMPDIR="$STORE/tmp"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$TMPDIR" || {
    echo "FATAL: cannot create $STORE (scratch mounted? set STORE=...)" >&2
    return 1 2>/dev/null || exit 1
}

# ensure_store_links: make each STORE_LINKS dir a symlink to $STORE/<dir>. Idempotent.
# Fails (non-zero) instead of touching a populated real directory.
ensure_store_links() {
    local d target bad=0
    for d in "${STORE_LINKS[@]}"; do
        target="$STORE/$d"
        mkdir -p "$target"
        if [ -L "$PROJECT_ROOT/$d" ]; then
            if [ "$(readlink -f "$PROJECT_ROOT/$d")" != "$(readlink -f "$target")" ]; then
                echo "FATAL: $PROJECT_ROOT/$d -> $(readlink "$PROJECT_ROOT/$d"), expected $target" >&2
                bad=1
            fi
        elif [ -d "$PROJECT_ROOT/$d" ] && [ -n "$(ls -A "$PROJECT_ROOT/$d")" ]; then
            echo "FATAL: $PROJECT_ROOT/$d is a populated directory in home. Migrate it once:" >&2
            echo "  rsync -a '$PROJECT_ROOT/$d/' '$target/' && rm -rf '$PROJECT_ROOT/$d' && ln -s '$target' '$PROJECT_ROOT/$d'" >&2
            bad=1
        else
            [ -d "$PROJECT_ROOT/$d" ] && rmdir "$PROJECT_ROOT/$d"
            ln -s "$target" "$PROJECT_ROOT/$d"
            echo "linked $d -> $target"
        fi
    done
    return $bad
}

ensure_store_links || { return 1 2>/dev/null || exit 1; }
