#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=h100:4
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --mem=0
#SBATCH --cpus-per-task=48
#SBATCH --time=0-01:00:00

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

srun uv run --offline --frozen --all-extras "$@"
