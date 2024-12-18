#!/bin/bash
#SBATCH --job-name=pythia_sweep
#SBATCH --output=logs/pythia_%A_%a.out
#SBATCH --error=logs/pythia_%A_%a.err
#SBATCH --array=0-35%4  # Run 4 jobs concurrently (adjust as needed)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00

# Create logs directory
mkdir -p logs

# Activate your conda/virtual environment (modify this path)
source /scratch/project/open-30-4/Petr_Hyner/ENVS/rl_sos/activate
