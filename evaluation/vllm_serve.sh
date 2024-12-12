#!/bin/bash

#SBATCH --job-name=JOB_NAME
#SBATCH --output=JOB_NAME.out
#SBATCH --error=JOB_NAME.err

#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=50G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL


source ~/.bashrc
conda activate synatra
MODEL_DIR='CHECKPOINT_DIR'

test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=1528 \
    --model="$MODEL_DIR" \
    --tensor-parallel-size=1 \
    --max-num-batched-tokens=66536
