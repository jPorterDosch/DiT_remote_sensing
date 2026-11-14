#!/bin/bash
#SBATCH --job-name=thresh-diag
#SBATCH -A isaac-utk0256
#SBATCH -p short
#SBATCH -q short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/lustre/isaac24/scratch/jdosch1/tmp/thresh_diag/%x-%A_%a.out
#SBATCH --error=/lustre/isaac24/scratch/jdosch1/tmp/thresh_diag/%x-%A_%a.out
set -euo pipefail
export PATH="$HOME/.conda/envs/DiTF/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export STORE=/lustre/isaac24/scratch/jdosch1/DiT_remote_sensing
export OVR_N_JOBS=${SLURM_CPUS_PER_TASK}
export JOBLIB_TEMP_FOLDER="$STORE/tmp/joblib"; mkdir -p "$JOBLIB_TEMP_FOLDER"
export OUT=/lustre/isaac24/scratch/jdosch1/tmp/thresh_diag
cd /nfs/home/jdosch1/COSC424/DiT_remote_sensing-geodedup
ARMS=(dinov3-sat dinov3-web flux-proj)
python3 /lustre/isaac24/scratch/jdosch1/tmp/thresh_diag.py "${ARMS[$SLURM_ARRAY_TASK_ID]}"
