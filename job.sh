#!/bin/bash
#SBATCH --partition=${hydra.launcher.partition}
#SBATCH --gres=gpu:${hydra.launcher.gpus_per_task}
#SBATCH --time=${hydra.launcher.timeout_min}
#SBATCH --output=${hydra.sweep.dir}/${hydra.job.name}_%j.out

# Wrap the command with Singularity
singularity exec \
  --bind /dev/dri,/etc/slurm,/usr/bin/sbatch,/usr/bin/srun,/scratch/project_465001424 \
  --env LD_LIBRARY_PATH="/opt/rocm/lib64:/opt/rocm/lib:/usr/lib64" \
  ${hydra.launcher.container} \
  ${hydra.job.command}