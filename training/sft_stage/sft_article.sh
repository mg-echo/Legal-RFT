#!/bin/bash
#SBATCH -J sft_article
#SBATCH -o log/sft_article-%j.log
#SBATCH -e log/sft_article-%j.err
#SBATCH -p GPU-8A100
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 8
#SBATCH --gres=gpu:8
#SBATCH --qos=gpu_8a100
#SBATCH --time=03:00:00
#SBATCH --mem=128G

echo "=== SFT LoRA Training ==="
echo "Time: $(date)"
echo "PWD : $PWD"
echo "Nodes: $SLURM_JOB_NODELIST"

. /etc/profile.d/modules.sh
module load cuda/12.8

eval "$(conda shell.bash hook)"
conda activate your_environment

base_model="your_model_path"

sft_lr=8e-5
sft_epochs=3
sft_weight_decay=1e-4
sft_warmup_ratio=0.10
lora_r=64
lora_alpha=128
lora_dropout=0.05
max_len=4096

uid="$(date +%Y%m%d_%H%M%S)"

node_array=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nnodes=$(echo $node_array | wc -w)
gpu_count=$(nvidia-smi -L | wc -l)

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600

target_global_batch=128
per_device_batch=4
grad_acc=$((target_global_batch / (gpu_count * nnodes * per_device_batch)))
if [ $grad_acc -lt 1 ]; then grad_acc=1; fi

echo "Nodes: $nnodes, GPUs/node: $gpu_count, grad_acc: $grad_acc"
echo "LoRA: r=${lora_r}, alpha=${lora_alpha}, dropout=${lora_dropout}"
echo "LoRA target modules: ${target_modules}"

torchrun \
  --nnodes=$nnodes \
  --nproc_per_node=$gpu_count \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  ./sft_article.py \
  --model_name_or_path "${base_model}" \
  --use_peft \
  --attn_implementation flash_attention_2 \
  --lora_r "${lora_r}" \
  --lora_alpha "${lora_alpha}" \
  --lora_dropout "${lora_dropout}" \
  --learning_rate "${sft_lr}" \
  --num_train_epochs "${sft_epochs}" \
  --per_device_train_batch_size "${per_device_batch}" \
  --per_device_eval_batch_size "${per_device_batch}" \
  --gradient_accumulation_steps "${grad_acc}" \
  --bf16 true \
  --report_to wandb \
  --logging_steps 5 \
  --eval_strategy steps \
  --eval_steps 10 \
  --save_strategy steps \
  --save_steps 10 \
  --load_best_model_at_end false \
  --save_total_limit 10 \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --lr_scheduler_type cosine \
  --weight_decay "${sft_weight_decay}" \
  --warmup_ratio "${sft_warmup_ratio}" \
  --output_dir "your_output_path" \
  --ddp_find_unused_parameters false \
  --max_length "${max_len}" \
  --packing false \
  --completion_only_loss true

echo "Done at $(date)"