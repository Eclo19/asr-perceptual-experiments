#!/bin/bash
#SBATCH --account=p53312
#SBATCH --partition=gengpu                # GPU partition, up to 48 h
#SBATCH --gres=gpu:1                      # Any GPU type: shorter queue than asking for a specific one
#SBATCH --time=12:00:00                   # Ask a little above the expected run time
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                 # 6 DataLoader workers + main process + spare
#SBATCH --mem=32G
#SBATCH --job-name=main_train_carfac_transformer_ctc
#SBATCH --output=logs/%x_%j.out           # logs/main_<jobid>.out

# Stop at the first error
set -euo pipefail
module purge all

REPO=/projects/p53312/Thesis_Workspace/asr-perceptual-experiments
PY=/home/ozi0609/.conda/envs/asr1/bin/python

# main.py uses the folder it's launched from as the workspace (ROOT = Path().resolve())
cd "$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # JAX (CARFAC) shares the GPU with PyTorch

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" Trainings/basic/main.py
echo "Finished at $(date)"