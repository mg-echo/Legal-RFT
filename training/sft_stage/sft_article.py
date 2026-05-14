import os
from accelerate import logging
from datasets import Dataset
from trl import SFTTrainer, SFTConfig,TrlParser, ModelConfig, get_peft_config
from transformers import AutoTokenizer



logger = logging.get_logger(__name__)

SYSTEM_PROMPT ="""
你是一位精通中国刑法的法官，擅长进行刑事案件的法条匹配和罪名分析。
遍历案件相关法条，依次分析其是否与案件事实完全匹配，排除不匹配的法条，推荐与裁定此案最相关的法条。针对每个相关法条，你必须按照以下法律三段论进行推理：
1. 拆解大前提：提取该法条的全部客观构成要件与主观构成要件，特别关注法条中前置性条件（如特定上游犯罪类型、特定目的、特殊主体身份等）。
2. 比对小前提：将法条的构成要件与案件事实、案件相关要素进行逐一比对，分析是否满足法条的适用条件，特别关注案件中是否存在法条所要求的特定情形或限定条件。
3. 决策判断：若任一要件未被案情事实满足，则直接排除该法条。最终选出的与裁定此案相关的法条。
注意：
1. 相关法条仅供参考；若参考法条无法覆盖案情，请根据你的专业知识库选择最适用的刑法条款；
2. 严禁仅凭关键词相似或行为外观相似即认定匹配，必须以构成要件的实质符合性为准。
"""

USER_PROMPT = """
请严格按照“法律三段论”逻辑，对以下案情进行深度辨析，并从候选法条中筛选出最准确的法条输出。/no_think
案情：\n{fact}
候选法条：\n{articles}
Json Output{{"reason":"判断理由", "law_article_id": "最终选定的法条编号数字"}}
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
            assistant_content = ""
            import json
            outputs_list = examples.get("outputs", [])
            output_items = outputs_list[i] if len(outputs_list) > i else []
            if isinstance(output_items, list):
                for item in output_items:
                    if item.get("verdict") == "chosen":
                        assistant_content = "<think>\n\n</think>\n\n" + item.get("text", "")
                        break
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(fact=examples["fact"][i], articles=examples["relevant_articles_full"][i])}
            ]
            
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            completion_text = assistant_content + tokenizer.eos_token
            
            output_dicts["prompt"].append(prompt_text)
            output_dicts["completion"].append(completion_text)
        return output_dicts

    orig = "../../data/example_data/sft_article_example.jsonl"
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
    os.environ["WANDB_PROJECT"] = "sft_for_article"
    os.environ["WANDB_NAME"] = "Qwen3-8B-SFT-run1"
    train()


    