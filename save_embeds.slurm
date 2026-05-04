#!/bin/bash
#SBATCH --job-name=save_embeds
#SBATCH -A acf-utk0011                 
#SBATCH --time=01:00:00                
#SBATCH --nodes=1                      
#SBATCH --ntasks-per-node=1            
#SBATCH --cpus-per-task=4                       
#SBATCH --partition=campus-gpu         
#SBATCH --qos=campus-gpu               # <-- NEW: Explicitly define the QoS
#SBATCH --gres=gpu:1                   
#SBATCH --output=job_output_%j.txt     
#SBATCH --error=job_error_%j.txt       

# This exact string satisfies the cluster's safety parser
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# 1. Load Anaconda
module load anaconda3

# 2. Activate your environment 
source activate DiTF

# 3. Run your training script
python save_null_embeds.py