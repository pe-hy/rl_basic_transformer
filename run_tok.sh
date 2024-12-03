#!/bin/bash
#SBATCH --job-name=sos
#SBATCH --account=OPEN-30-4
#SBATCH --partition=qgpu
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

# Activate conda
export PATH=/scratch/project/open-30-4/Petr_Hyner/ENVS/rl_sos/bin:$PATH
export PYTHON_PATH=/scratch/project/open-30-4/Petr_Hyner/ENVS/rl_sos/bin/python
# Go to source directory
cd /scratch/project/open-30-4/Petr_Hyner/rl/rl_basic_transformer

python create_tokenizer.py