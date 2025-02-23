#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --mem=0
#SBATCH --cpus-per-task=48
#SBATCH --time=0-01:00:00

srun uv run python -m train --config-name=synthetic_h100x4 +paths.model_name=test_synthetic_1x4 "$@"
