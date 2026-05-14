#!/bin/bash
#SBATCH -J dapo_penalty
#SBATCH -o log/dapo_penalty-%j.log
#SBATCH -e log/dapo_penalty-%j.err
#SBATCH -p GPU-8A100
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 8
#SBATCH --gres=gpu:8
#SBATCH --qos=gpu_8a100
#SBATCH --time=20:00:00
#SBATCH --mem=256G      


echo "=== DAPO Training (vLLM Server Mode) ===" 
echo "Time: $(date)"
echo "Nodes: $SLURM_JOB_NODELIST"

. /etc/profile.d/modules.sh 
module load cuda/12.8 

eval "$(conda shell.bash hook)" 
conda activate your_environment 


trap 'echo "Cleaning up background processes..."; kill $(jobs -p) 2>/dev/null' EXIT

# ============================
# 用户参数
# ============================
model_path="your_model_path"

grpo_lr=8e-6
max_prompt_len=1536 
max_completion_len=1536
num_generations=8 
importance_sampling_level="sequence"  
epsilon=0.2                     
epsilon_high=0.28                
beta=0.03                         
loss_type="dapo"
grpo_epochs=1                    


weight_decay=1e-4
output_dir="your_output_path" 


target_global_batch=32 
per_device_batch=4 


export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_TIMEOUT=3600
export TRANSFORMERS_OFFLINE=1

vllm_gpu_count=4
train_gpu_count=4

grad_acc=$((target_global_batch / (train_gpu_count * 1 * per_device_batch))) 
if [ $grad_acc -lt 1 ]; then grad_acc=1; fi 
echo "Train GPUs: $train_gpu_count, vLLM GPUs: $vllm_gpu_count, grad_acc: $grad_acc" 


CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve \
  --model "${model_path}" \
  --port 8000 \
  --tensor-parallel-size ${vllm_gpu_count} \
  --gpu-memory-utilization 0.9 \
  --max-model-len 3584 &

echo "Waiting for vLLM server to be ready..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 5
done
echo "vLLM server is UP and ready!"



CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --nnodes=1 \
  --nproc_per_node=${train_gpu_count} \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  ./dapo_penalty.py \
  --model_name_or_path "${model_path}" \
  --output_dir "${output_dir}" \
  --learning_rate ${grpo_lr} \
  --temperature 0.6 \
  --top_p 0.9 \
  --top_k 30 \
  --max_prompt_length ${max_prompt_len} \
  --max_completion_length ${max_completion_len} \
  --num_train_epochs "${grpo_epochs}" \
  --use_peft \
  --log_completions true \
  --per_device_train_batch_size ${per_device_batch} \
  --per_device_eval_batch_size "${per_device_batch}" \
  --num_generations ${num_generations} \
  --importance_sampling_level ${importance_sampling_level} \
  --epsilon ${epsilon} \
  --beta ${beta} \
  --loss_type ${loss_type} \
  --gradient_accumulation_steps ${grad_acc} \
  --bf16 true \
  --logging_steps 5 \
  --eval_strategy steps \
  --eval_steps 200 \
  --save_strategy steps \
  --save_steps 200 \
  --load_best_model_at_end false \
  --save_total_limit 5 \
  --greater_is_better false \
  --weight_decay ${weight_decay} \
  --ddp_find_unused_parameters false \

echo "Done at $(date)"