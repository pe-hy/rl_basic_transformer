#!/bin/bash
#SBATCH --job-name=gen_task
#SBATCH --account=project_465001424
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --partition=small-g

module load PyTorch

cd src
singularity exec $SIF python3 create_tokenizer.py