#!/bin/bash
# =============================================================================
# Train one ASR system as a Slurm job. Everything after the script name goes to Trainings/args/main.py:
#
#   sbatch --job-name=ml_carfac Trainings/args/main.sbatch --frontend carfac --level-range 45 85 --variants 3 --mix sample
#
# Submit from the repo root: logs/ is relative to the folder you submit from.
# Check a run's hashes first, without a job: python Trainings/args/main.py <same arguments> --dry-run
# =============================================================================
#SBATCH --account=p53312                  # Your allocation
#SBATCH --partition=gengpu                # GPU partition, up to 48 h
#SBATCH --gres=gpu:1                      # One GPU of any type: shorter waits than asking for a specific one
#SBATCH --time=8:00:00                    # train.100 Conformer ≈ 3–5 h, plus CARFAC cache builds; override with sbatch --time
#SBATCH --nodes=1
#SBATCH --ntasks=1                        # One process (python)...
#SBATCH --cpus-per-task=8                 # ...with 8 cores: 6 DataLoader workers + main process + spare
#SBATCH --mem=48G                         # RAM for the workers and the cache build
#SBATCH --job-name=asr                    # Override with sbatch --job-name=<label>
#SBATCH --output=logs/%x_%j.out           # %x = job name, %j = job ID; stderr goes here too

# Stop at the first failing command, and on unset variables
set -euo pipefail

# Start clean: no modules inherited from the shell you submitted from
module purge all

# ---- Paths: edit if your setup differs
REPO=/projects/p53312/Thesis_Workspace/asr-perceptual-experiments
PY=/home/ozi0609/.conda/envs/asr1/bin/python

cd "$REPO"
export ASRLIB_ROOT="$REPO"                 # The workspace (datasets/, features/, noise/, models/, results/)
export XLA_PYTHON_CLIENT_PREALLOCATE=false # JAX (CARFAC) takes GPU memory only as needed, leaving the rest to PyTorch

echo "Job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname) at $(date)"
echo "Args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" Trainings/args/main.py "$@"

echo "Finished at $(date)"
