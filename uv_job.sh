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
export HF_HOME=$SCRATCH/huggingface
export HF_HUB_OFFLINE=1

# Disable gpu mem pre-allocation so we can see how much we're using in wandb.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

srun uv run --offline --frozen --all-extras "$@"
