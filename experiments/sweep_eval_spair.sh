#!/bin/bash
#SBATCH -A acf-utk0011
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --time=1-00:00:00               # Wall time (days-hh:mm:ss)
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.out

# --- Bootstrap: find project root (can't be sourced) ------
_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
while [ "$_dir" != "/" ] && [ ! -d "$_dir/.git" ]; do _dir="$(dirname "$_dir")"; done
PROJECT_ROOT="$_dir"

source "$PROJECT_ROOT/experiments/_common.sh" || {
	echo "FATAL: Failed to source common.sh" >&2; exit 1;
}
setup_environment


echo "SAVE_DIR: $SAVE_DIR"
echo "COMBINATIONS: 24  (6 timesteps × 4 block indices)"
countdown
print_delim "## START"

for t in 20 100 180 260 340 420; do
    for k in 19 28 37 46; do
        echo "--- t=$t k=$k ---"
        python3 "$PROJECT_ROOT/eval_spair.py" \
            --cd \
            --img_size 640 640 \
            --exp_name "$SCRIPT_NAME" \
            --dataset spair \
            --dataset_path "/lustre/isaac24/scratch/jdosch1/DeepLearning/datasets/SPair-71k" \
            --save_path "$SAVE_DIR" \
            --dit_model flux \
            --t "$t" \
            --k "$k" \
            --ensemble_size 8
    done
done

print_delim "## DONE"
