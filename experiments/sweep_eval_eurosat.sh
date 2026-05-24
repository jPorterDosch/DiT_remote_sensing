#!/bin/bash
#SBATCH -A isaac-utk0256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=ai-tenn
#SBATCH --partition=ai-tenn
#SBATCH --time=3-00:00:00               # Wall time (days-hh:mm:ss)
#SBATCH --job-name=sweep_eval_eurosat
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.out

# --- Bootstrap: find project root (can't be sourced) ------
_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"

source "$PROJECT_ROOT/experiments/_common.sh" || {
    echo "FATAL: Failed to source experiments/_common.sh" >&2; exit 1;
}
setup_environment

# Saved so plot_sweep.py can later find and use the features. 
# Each (t, k) pair gets its own subdirectory so
# feature caches (train_feats.npz / test_feats.npz) don't collide across runs.
RESULTS_DIR="$SAVE_DIR/layers_cat/eurosat_flux"
mkdir -p "$RESULTS_DIR"

# ============================================================================
# SWEEP GRID
# ============================================================================
# EuroSAT is 64×64 native, resized to 224×224 → ~196 tokens.
# Fewer tokens → smaller shift mu (~0.49 vs ~0.73 for SPair at 640px).
# Expected optimal t is closer to paper's t=260 than SPair's t=420.
# Sweep t broadly first, then narrow once the plateau is visible.
#
# t : [200, 260, 300, 400] — covers the expected peak and shoulders
# k : [26, 27, 28, 29, 30] — neighbourhood of SPair's optimal k=28
# ============================================================================
T_VALUES=(200 300 400 500)
K_VALUES=(26 27 28 29 30)

TOTAL=$(( ${#T_VALUES[@]} * ${#K_VALUES[@]} ))
echo "RESULTS_DIR: $RESULTS_DIR"
echo "COMBINATIONS: $TOTAL  (${#T_VALUES[@]} timesteps × ${#K_VALUES[@]} block indices)"
countdown
print_delim "## START"

for t in "${T_VALUES[@]}"; do
    for k in "${K_VALUES[@]}"; do
        echo "--- t=$t k=$k ---"
        python3 "$PROJECT_ROOT/run.py" \
            --task classification \
            --dataset.name eurosat \
            --dataset.path "/lustre/isaac24/scratch/jdosch1/DeepLearning/datasets/EuroSAT" \
            --img-size 224 224 \
            --model.name flux \
            --model.ensemble-size 8 \
            --cd \
            --t "$t" \
            --k "$k"
    done
done

print_delim "## DONE"

# ============================================================================
# PLOTTING (runs after sweep completes)
# ============================================================================
# Default metric is top1_accuracy at 100% labels.
# Re-run manually with --metric macro_f1 --label_fraction 0.1 etc. for low-shot views.
# ============================================================================
echo "Generating sweep plots..."
python3 "$PROJECT_ROOT/plot_sweep.py" \
    --results-dir "$RESULTS_DIR" \
    --task classification \
    --metric top1_accuracy \
    --label_fraction 1.0 \
    --title "EuroSAT sweep — flux (100% labels)" \
    --out "$PROJECT_ROOT/sweep_plot_eurosat"

echo "Done. Plots saved to $PROJECT_ROOT/sweep_plot_eurosat_top1_accuracy.png"