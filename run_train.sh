#!/bin/bash
#SBATCH --job-name=sos
#SBATCH --account=OPEN-30-4
#SBATCH --partition=qgpu
#SBATCH --time=48:00:00
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1

# Activate conda
export PATH=/scratch/project/open-30-4/Petr_Hyner/ENVS/rl_sos/bin:$PATH
export PYTHON_PATH=/scratch/project/open-30-4/Petr_Hyner/ENVS/rl_sos/bin/python
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
# Go to source directory
cd /scratch/project/open-30-4/Petr_Hyner/rl/rl_basic_transformer

srun python train.py