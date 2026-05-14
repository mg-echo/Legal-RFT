#!/bin/bash
#SBATCH -J dpo
#SBATCH -o log/dpo-%j.log
#SBATCH -e log/dpo-%j.err
#SBATCH -p GPU-8A100
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 4
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu_8a100
#SBATCH --time=03:00:00
#SBATCH --mem=128G

. /etc/profile.d/modules.sh
module load cuda/12.8

eval "$(conda shell.bash hook)"
conda activate your_environment           


export WANDB_PROJECT="dpo_for_article"
export WANDB_RUN_NAME="qwen3-8B-lora-run1"
export FORCE_TORCHRUN=1 
export NNODES=1           
export NPROC_PER_NODE=4   

llamafactory-cli train ./dpo_article.yaml
