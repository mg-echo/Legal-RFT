import os
import torch
from accelerate import logging
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig, TrlParser, ModelConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import json

# 导入奖励函数
from reward_funcs import (
    fast_think_length_penalty_reward_func,
    format_reward_func,
    math_consistency_reward_func,
    log_penalty_reward_func,
    think_tag_presence_reward_func,
    repetition_penalty_reward_func,
    range_accuracy_reward_func
)


logger = logging.get_logger(__name__) 
SYSTEM_PROMPT ="""
你是一位精通中国刑法的资深法官。你的任务是根据提供的案情、罪名法条及量刑要素，严密地推导被告人的宣告刑。
1. 确定量刑起点：根据基本犯罪构成事实，在法定刑范围内确定初始刑期。单位统一为“月”。
2. 确定基准刑：在量刑起点上，考虑增加刑量的其他犯罪事实（如犯罪次数、数额超出起点的部分），计算出基准刑。
3. 调节情节比例：识别案件中的量刑要素，根据规定给出百分比加减。
4. 确定宣告刑：应用公式 $宣告刑 = 基准刑 \\times (1 + \\sum 量刑要素调节比例)$。
"""


USER_PROMPT = """
对以下案情进行分析，根据提供的罪名、法条及量刑要素，严密地推导被告人的宣告刑。
案情：\n{fact}
相关法条：\n{relevant_articles}
量刑要素及指导意见：\n{sentencing_factors}
JSON OUTPUT:
{{
  "sentencing_chain": {{
    "starting_point": {{
      "months": "数值 <int>",
      "reasoning": "说明依据哪项事实锁定了量刑起点"
    }},
    "base_sentence": {{
      "months": "数值 <int>",
      "reasoning": "说明在量刑起点基础上，因为哪些额外事实（如数额、次数）增加了多少刑期"
    }},
    "adjustments": [
      {{
        "factor": "要素名称",
        "ratio": "增加/减少百分比",
        "reasoning": "结合案情说明选择该比例的理由"
      }}
    ],
    "mathematical_verification": "计算宣告刑： $基准刑 \\times (1 + \\sum 量刑要素调节比例)$",
    "final_penalty": {{
      "penalty": "最终宣告刑月份数值 <int>。特别地，如果判定为【无期徒刑】，数值请输出 `360`；如果判定为【死刑】，数值请输出 `420`",
      "summary": "最终裁量权的综合说明"
    }}
  }}
}}
"""

def train():
    parser = TrlParser((GRPOConfig, ModelConfig))
    training_args, model_args = parser.parse_args_into_dataclasses()

    training_args.use_vllm = True
    training_args.vllm_mode = "server"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16, 
        device_map=None,
        attn_implementation="flash_attention_2"
    )
    

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)  
    
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"


    def make_conversation(example):
        fact = example.get("fact", "")
        meta = example.get("meta") or {}
        accusation = meta.get("accusation", [])
        relevant_arts = meta.get("relevant_articles", [])
        rel_arts_full = example.get("relevant_articles_full", [])

        accusation_str = "、".join(accusation) if isinstance(accusation, list) else str(accusation)
        relevant_arts_str = "、".join(map(str, relevant_arts)) if isinstance(relevant_arts, list) else str(relevant_arts)

        rel_arts_full_str = "\n".join(rel_arts_full) if isinstance(rel_arts_full, list) else str(rel_arts_full)

        relevant_articles_combined = f"指控罪名：{accusation_str}\n相关法条序号：{relevant_arts_str}\n法条全文：\n{rel_arts_full_str}"

        sentencing_factors = example.get("output", {})
        if isinstance(sentencing_factors, (dict, list)):
            sentencing_factors_str = json.dumps(sentencing_factors, ensure_ascii=False)
        else:
            sentencing_factors_str = str(sentencing_factors)

        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        user_msg = {"role": "user", "content": USER_PROMPT.format(
            fact=fact, 
            relevant_articles=relevant_articles_combined, 
            sentencing_factors=sentencing_factors_str
        )}

        term = meta.get("term_of_imprisonment", {})
        solution = json.dumps(term, ensure_ascii=False)
        return {"prompt": [system_msg, user_msg], "solution": solution, "sentencing_factors": sentencing_factors_str}
    

    raw_dataset = Dataset.from_json("your_dataset_path")  

    dataset = raw_dataset.map(make_conversation, batched=False, remove_columns=raw_dataset.column_names)
    dataset = dataset.shuffle(seed=42).select(range(4000))

    ds_split = dataset.train_test_split(test_size=0.01, seed=42)
    train_ds = ds_split["train"]
    eval_ds = ds_split["test"]

    base_model.config.use_cache = False  
    training_args.gradient_checkpointing = True
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules="all-linear"
    )
    peft_model = get_peft_model(base_model, peft_config)
    peft_model.enable_input_require_grads()

    trainer = GRPOTrainer(
        model=peft_model,
        reward_funcs=[format_reward_func, math_consistency_reward_func, log_penalty_reward_func, fast_think_length_penalty_reward_func, think_tag_presence_reward_func, repetition_penalty_reward_func, range_accuracy_reward_func],
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "grpo_for_penalty" 
    os.environ["WANDB_NAME"] = "Qwen3-8B-lora-run1"
    train()