#!/bin/bash
# Submits the T3 single-batch overfit test to ISAAC.
# Usage:  sbatch experiments/test_overfit.sh
#
# What it does:
#   1. Loads Flux-dev from the shared lustre path.
#   2. Runs test_overfit.py: 100 gradient steps on one synthetic image.
#   3. Asserts MIM loss drops >=50% and saves overfit_loss_curve.png.
#
# Memory budget (bfloat16, batch=1):
#   Flux-dev weights   ~24 GB  (12B params × 2 bytes)
#   VAE weights        ~0.3 GB
#   Activations        ~1-2 GB (backprop through 28 blocks, seq≈450, D=3072)
#   LoRA / decoder     ~0.1 GB
#   ─────────────────────────
#   Total              ~26-27 GB  →  40 GB GPU is sufficient
#
#SBATCH -A acf-utk0011
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-large
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

# --- Bootstrap: walk up to project root (same pattern as other scripts) ---
_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"

source "$PROJECT_ROOT/experiments/_common.sh" || {
    echo "FATAL: failed to source _common.sh" >&2; exit 1;
}
setup_environment

# --- Conda ---
# Try common miniconda/anaconda locations; fall back to whatever conda is on PATH.
for _conda_sh in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$_conda_sh" ]; then
        source "$_conda_sh"
        break
    fi
done
conda activate DiTF

# --- Output ---
OUT_DIR="${SCRATCHDIR:-$PROJECT_ROOT}/tests/overfit_test_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_DIR"

# --- Run ---
print_delim "## START"
set -x
python3 "$PROJECT_ROOT/tests/test_overfit.py" \
    2>&1 | tee "$OUT_DIR/overfit.log"

# Copy loss curve to output directory regardless of test exit code.
cp -f "$PROJECT_ROOT/overfit_loss_curve.png" "$OUT_DIR/" 2>/dev/null || true
set +x
print_delim "## DONE"
echo "Outputs: $OUT_DIR"
