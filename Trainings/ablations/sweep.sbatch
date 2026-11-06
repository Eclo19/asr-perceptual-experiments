#!/bin/bash
# =============================================================================
# One CARFAC sweep run as a Slurm job. Everything after the script name is passed to sweep.py:
#
#   sbatch Trainings/carfac_sweep/sweep.sbatch --velocity-scale 0.05
#
# Usually called by submit_sweep.sh, which submits the whole sweep. Submit from the repo root:
# Slurm starts the job in the folder you submitted from, and logs/ is relative to it.
# =============================================================================
#SBATCH --account=p53312                  # Your allocation
#SBATCH --partition=gengpu                # GPU partition, up to 48 h
#SBATCH --gres=gpu:1                      # One GPU of any type (A100/H100/...): shorter waits than gpu:a100:1
#SBATCH --time=12:00:00                   # 3-shard Conformer + CARFAC cache: a few hours; generous on purpose
#SBATCH --nodes=1
#SBATCH --ntasks=1                        # One process (python)...
#SBATCH --cpus-per-task=8                 # ...with 8 cores: 6 DataLoader workers + main process + spare
#SBATCH --mem=48G                         # RAM for the workers and the cache build
#SBATCH --job-name=carfac-sweep           # submit_sweep.sh overrides it with the run's label
#SBATCH --output=logs/%x_%j.out           # %x = job name, %j = job ID; stderr goes here too

# Stop at the first failing command, and on unset variables
set -euo pipefail

# Start clean: no modules inherited from the shell you submitted from
module purge all

# ---- Paths: edit if your setup differs
REPO=/projects/p53312/Thesis_Workspace/asr-perceptual-experiments
PY=/home/ozi0609/.conda/envs/asr1/bin/python

# If the asr1 env only works inside your Singularity container, set its image path here
# (inside a VS Code terminal: echo $SINGULARITY_CONTAINER) and the job runs inside it.
# Not needed: your env check outside the container passed.
IMAGE=""

cd "$REPO"
export ASRLIB_ROOT="$REPO"                 # The workspace (datasets/, features/, models/, results/)
export XLA_PYTHON_CLIENT_PREALLOCATE=false # JAX takes GPU memory only as needed, leaving the rest to PyTorch

echo "Job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname) at $(date)"
echo "Args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ -n "$IMAGE" ]]; then
    singularity exec --nv "$IMAGE" "$PY" Trainings/ablations/sweep.py "$@"
else
    "$PY" Trainings/ablations/sweep.py "$@"
fi

echo "Finished at $(date)"