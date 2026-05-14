import os
from accelerate import logging
from datasets import Dataset
from trl import SFTTrainer, SFTConfig,TrlParser,ScriptArguments, ModelConfig, get_peft_config
from transformers import AutoTokenizer



logger = logging.get_logger(__name__) 

SYSTEM_PROMPT ="""
你是一位精通中国刑法的资深法官。你的任务是根据提供的案情、罪名法条及量刑要素，严密地推导被告人的宣告刑区间。
1. 确定量刑起点：根据基本犯罪构成事实，在法定刑范围内确定初始刑期。单位统一为“月”。
2. 确定基准刑：在量刑起点上，考虑增加刑量的其他犯罪事实（如犯罪次数、数额超出起点的部分），计算出基准刑。
3. 调节情节比例：识别案件中的量刑要素，根据规定给出百分比加减。
4. 确定宣告刑：应用公式 $宣告刑 = 基准刑 \\times (1 + \\sum 量刑要素调节比例)$。

推导得出宣告刑后，将其归入以下 11 个法定区间之一：
0: [0, 6] 个月
1: (6, 9] 个月
2: (9, 12] 个月
3: (12, 24] 个月
4: (24, 36] 个月
5: (36, 60] 个月
6: (60, 84] 个月
7: (84, 120] 个月
8: (120, 180] 个月
9: 180 个月以上
10: 无期徒刑与死刑
"""

USER_PROMPT = """
对以下案情进行分析，根据提供的罪名、法条及量刑要素，严密地推导被告人的宣告刑区间。/no_think
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
    "final_range": {{
      "range": "11 个法定区间之一对应序号 <int>",
      "summary": "最终裁量权的综合说明"
    }}
  }}
}}
"""

def train():
    parser = TrlParser((ModelConfig, SFTConfig))
    model_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)

    model_args.lora_target_modules = ["q_proj","k_proj","v_proj","o_proj","up_proj","gate_proj","down_proj"]
    
    # Use a token that is never used
    tokenizer.pad_token = "<|fim_pad|>"
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left" 

    def format_prompts(examples):
        output_dicts = {"prompt": [], "completion": []}
        for i in range(len(examples['fact'])):

            fact = examples["fact"][i]
            meta = examples.get("meta", [{}])[i] or {}
            accusation = meta.get("accusation", [])
            relevant_arts = meta.get("relevant_articles", [])
            rel_arts_full = examples.get("relevant_articles_full", [""])[i]
            
            accusation_str = "、".join(accusation) if isinstance(accusation, list) else str(accusation)
            relevant_arts_str = "、".join(map(str, relevant_arts)) if isinstance(relevant_arts, list) else str(relevant_arts)
            
            relevant_articles_combined = f"指控罪名：{accusation_str}\n相关法条序号：{relevant_arts_str}\n法条全文：\n{rel_arts_full}"
            
            sentencing_factors = examples.get("output", [""])[i]
            if isinstance(sentencing_factors, (dict, list)):
                import json
                sentencing_factors = json.dumps(sentencing_factors, ensure_ascii=False)
            
            assistant_content = examples.get("penalty_output", [""])[i]
            if isinstance(assistant_content, (dict, list)):
                import json
                assistant_content = json.dumps(assistant_content, ensure_ascii=False)
            else:
                assistant_content = str(assistant_content)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(fact=fact, relevant_articles=relevant_articles_combined, sentencing_factors=sentencing_factors)}
            ]
            
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            completion_text = "<think>\n\n</think>\n\n" + assistant_content + tokenizer.eos_token
            
            output_dicts["prompt"].append(prompt_text)
            output_dicts["completion"].append(completion_text)
        return output_dicts

    orig = "../../data/example_data/sft_penalty_example.jsonl"
    shuf = orig.replace(".jsonl", "_shuf_seed42.jsonl")
    if not os.path.exists(shuf):
      ds_tmp = Dataset.from_json(orig)
      ds_tmp = ds_tmp.shuffle(seed=42)
      ds_tmp.to_json(shuf)
    dataset = Dataset.from_json(shuf)
    dataset = dataset.map(format_prompts, batched=True, remove_columns=dataset.column_names)

    ds_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = ds_split["train"]
    eval_ds = ds_split["test"]

    training_args.gradient_checkpointing =  True
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    trainer = SFTTrainer(
        model=model_args.model_name_or_path, 
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        processing_class=tokenizer,
        peft_config = get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "sft_for_penalty"
    os.environ["WANDB_NAME"] = "Qwen3-8B-SFT-run1"
    train()


    