#!/bin/bash
#SBATCH --job-name=gen_task
#SBATCH --account=project_465001424
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=small-g

module load PyTorch

cd ../src

singularity exec $SIF python3 countdown_generate.py --seed 4 --data_dir ../data/sos/ --min_range 4 --start_range 4 --num_samples 2000000 --search dfs