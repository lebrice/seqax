#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=cpubase_bynode_b3
#SBATCH --mem=0

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=$SCRATCH/huggingface
export HF_HUB_OFFLINE=1

srun uv run --offline --frozen --all-extras python prepare_data.py "$@"
