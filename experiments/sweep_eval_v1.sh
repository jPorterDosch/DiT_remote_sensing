#!/bin/bash
#SBATCH -A acf-utk0011
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --time=1-00:00:00               # Wall time (days-hh:mm:ss)
#SBATCH --job-name=sweep_eval
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

echo "SAVE_DIR: $SAVE_DIR"
echo "COMBINATIONS: 16  (4 timesteps × 4 block indices, fine-grained around peak at k=28, t>=420)"
countdown
print_delim "## START"

# Coarse sweep showed peak at k=28, t=420 (accuracy still rising). Fine sweep: k∈[23,33] step 1,
# t∈[300,550] step 50 to find where (if) accuracy plateaus or peaks above t=420.
for t in 200 260 300; do
    for k in 28; do
        echo "--- t=$t k=$k ---"
        python3 "$PROJECT_ROOT/eval.py" \
            --cd \
            --task correspondence \
            --img-size 640 640 \
            --dataset.name spair \
            --dataset.path "/lustre/isaac24/scratch/jdosch1/DeepLearning/datasets/SPair-71k" \
            --save-dir "$SAVE_DIR" \
            --model.name flux \
            --model.ensemble-size 8 \
            --t "$t" \
            --k "$k"
    done
done

print_delim "## DONE"
