#!/bin/bash
#SBATCH --job-name=gen_task
#SBATCH --account=project_465001424
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --partition=small-g

module load PyTorch
cd src

# Print Python path and installed packages for debugging
singularity exec $SIF python3 -c "import sys; print(sys.path)"
singularity exec $SIF pip list | grep hydra

# Run your script with Python from container
singularity exec $SIF python3 filter_data.py