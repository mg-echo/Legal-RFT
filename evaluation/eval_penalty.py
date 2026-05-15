from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import json
import os
import re
from sentencing_tool import SentencingGuideRetriever, build_tool_system_prompt

def compute_macro_prf(cases, num_labels=11):
    tp = [0] * num_labels
    fp = [0] * num_labels
    fn = [0] * num_labels

    for case in cases:
        gt = case.get('gt_range', -1)
        pred = case.get('pred_range', -1)
        if 0 <= gt < num_labels:
            for lbl in range(num_labels):
                if pred == lbl and gt == lbl:
                    tp[lbl] += 1
                elif pred == lbl and gt != lbl:
                    fp[lbl] += 1
                elif gt == lbl and pred != lbl:
                    fn[lbl] += 1

    sum_p = sum_r = sum_f = 0.0
    effective_labels = 0
    for lbl in range(num_labels):
        denom_p = tp[lbl] + fp[lbl]
        denom_r = tp[lbl] + fn[lbl]
        p = tp[lbl] / denom_p if denom_p > 0 else 0.0
        r = tp[lbl] / denom_r if denom_r > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        sum_p += p
        sum_r += r
        sum_f += f
        effective_labels += 1

    if effective_labels == 0:
        return 0.0, 0.0, 0.0

    macro_p = sum_p / effective_labels * 100
    macro_r = sum_r / effective_labels * 100
    macro_f1 = sum_f / effective_labels * 100
    return macro_p, macro_r, macro_f1


def map_imprisonment_to_range(term_info):
    if term_info.get("death_penalty") or term_info.get("life_imprisonment"):
        return 10
    months = term_info.get("imprisonment", 0)
    if months <= 6:       return 0
    elif months <= 9:     return 1
    elif months <= 12:    return 2
    elif months <= 24:    return 3
    elif months <= 36:    return 4
    elif months <= 60:    return 5
    elif months <= 84:    return 6
    elif months <= 120:   return 7
    elif months <= 180:   return 8
    else:                 return 9


def map_penalty_to_range(penalty):
    if penalty == 360 or penalty == 420:
        return 10
    months = penalty
    if months == -1:      return -1
    elif months <= 6:     return 0
    elif months <= 9:     return 1
    elif months <= 12:    return 2
    elif months <= 24:    return 3
    elif months <= 36:    return 4
    elif months <= 60:    return 5
    elif months <= 84:    return 6
    elif months <= 120:   return 7
    elif months <= 180:   return 8
    else:                 return 9


def load_law_articles_mapping(mapping_file):
    law_map = {}
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    fid = str(item.get('law_article_id', item.get('id', '')))
                    if fid:
                        law_map[fid] = item.get('content', json.dumps(item, ensure_ascii=False))
        print(f"✓ 已加载 {len(law_map)} 条法条映射")
    except Exception as e:
        print(f"✗ 加载法条映射失败: {e}")
    return law_map


def evaluate_accuracy(cases):
    correct = 0
    correct_within_one = 0
    correct_within_two = 0
    total = len(cases)
    for case in cases:
        gt_range = map_imprisonment_to_range(case.get("term_info", {}))
        content = case.get("content", "")
        pred_range = -1
        try:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            json_str = match.group(0) if match else content
            out_dict = json.loads(json_str)
            pred_range = int(out_dict.get("sentencing_chain", {}).get("final_penalty", {}).get("penalty", -1))
        except Exception:
            pass
        pred_range = map_penalty_to_range(pred_range)
        case["gt_range"] = gt_range
        case["pred_range"] = pred_range
        if pred_range == gt_range:
            correct += 1
        if abs(pred_range - gt_range) <= 1 and pred_range != -1:
            correct_within_one += 1
        if abs(pred_range - gt_range) <= 2 and pred_range != -1:
            correct_within_two += 1

    acc = correct / total if total > 0 else 0
    acc_within_one = correct_within_one / total if total > 0 else 0
    acc_within_two = correct_within_two / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {acc:.2%}")
    print(f"Accuracy ±1 : {correct_within_one}/{total} = {acc_within_one:.2%}")
    print(f"Accuracy ±2 : {correct_within_two}/{total} = {acc_within_two:.2%}")

    macro_p, macro_r, macro_f1 = compute_macro_prf(cases, num_labels=11)
    print(f"Macro Precision : {macro_p:.2f}%")
    print(f"Macro Recall : {macro_r:.2f}%")
    print(f"Macro F1 : {macro_f1:.2f}%")

    return acc

def extract_tool_call(text: str):
    """从模型输出中提取 function call JSON"""
    try:
        obj = json.loads(text.strip())
        return obj if obj.get("name") and obj.get("arguments") else None
    except:
        return None


def batch_chat_with_tool(tokenizer, model, cases, sampling_params, retriever,gen_system, gen_user_template, reasoning_system, full_user_template, lora_request=None):
    tool_system = build_tool_system_prompt(gen_system)

    first_texts = []
    for c in cases:
        user_text = gen_user_template.format(fact=c['fact'])
        messages = [
            {"role": "system", "content": tool_system},
            {"role": "user", "content": user_text}
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        first_texts.append(text)

    first_outputs = model.generate(first_texts, sampling_params, use_tqdm=True, lora_request=lora_request)
    first_responses = [out.outputs[0].text for out in first_outputs]

    final_texts = []
    for idx, (resp, c) in enumerate(zip(first_responses, cases)):
        print(f"\n--- 案例 {c['id']} (索引 {idx}) ---")
        tool_call = extract_tool_call(resp)
        if tool_call and tool_call["name"] == "retrieve_sentencing_guidance":
            factors = tool_call["arguments"].get("sentencing_factors", [])
            print(f"[要素提取] 识别要素: {factors}")
            guidance = retriever.retrieve(factors)
            print(f"[工具状态] 调用成功，返回指导意见（前300字）：\n{guidance[:300]}{'...' if len(guidance) > 300 else ''}")
        else:
            print(f"[要素提取] 工具调用失败，模型输出（前300字）：\n{resp[:300]}")
            guidance = "工具调用失败，未检索到指导意见，请根据案情直接推导刑期。"

        full_user = full_user_template.format(
            fact=c['fact'],
            relevant_articles_combined=c['relevant_articles_combined'],
            guidance=guidance
        )
        messages = [
            {"role": "system", "content": reasoning_system},
            {"role": "user", "content": full_user},
            {"role": "assistant", "content": resp},
            {"role": "user", "content": f"上述工具返回结果如下，请基于该结果完成量刑推导：\n{guidance}"}
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        final_texts.append(text)

    print("\n===== 第二轮推理：生成最终量刑 =====")
    final_outputs = model.generate(final_texts, sampling_params, use_tqdm=True, lora_request=lora_request)

    results = []
    think_end_token_id = tokenizer.convert_tokens_to_ids('</think>')
    for out in final_outputs:
        output_ids = list(out.outputs[0].token_ids)
        index = 0
        for tid in [151668, think_end_token_id]:
            if tid is None:
                continue
            try:
                index = len(output_ids) - output_ids[::-1].index(tid)
                break
            except ValueError:
                continue
        thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        results.append((thinking, content))
    return results

if __name__ == "__main__":
    model_path = "your_model_path"
    lora_adapter_path = "your_lora_adapter_path"
    model_name = "Qwen3-8B"

    guideline_file = "../data/knowledge/sentencing_guidelines.jsonl"
    jsonl_file = r"your_cases_file"
    predict_file = r"your_article_predict_file"
    output_content = f"predict_penalty_{model_name}.jsonl"
    output_thinking_content = f"predict_penalty_thinking_{model_name}.jsonl"

    # ========== 初始化工具 ==========
    retriever = SentencingGuideRetriever(guideline_file)
    law_articles_map = load_law_articles_mapping(r"your_mapping_file")

    # ========== 加载模型 ==========
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        enforce_eager=False,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64
    )

    stop_token_ids = [tokenizer.eos_token_id]
    if "<|im_end|>" in tokenizer.get_vocab():
        stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        top_k=30,
        min_p=0,
        stop_token_ids=stop_token_ids,
        max_tokens=6144
    )

    # ========== 提示词定义 ==========
    # 第一轮：要素提取（用户提供的专用提示词）
    PROMPT_GENERATE_SYSTEM = """你是一位精通中国刑法的审判员助理。你的任务是从案情描述中提取犯罪主体的量刑要素(量刑要素共有18种，包括:未成年人、老年人、聋哑人或盲人、未遂犯、从犯、自首情节、立功情节、坦白情节、当庭自愿认罪、退赃退赔、积极赔偿被害人经济损失并取得谅解、当事人根据刑事诉讼法第二百七十七条达成刑事和解协议、羁押期间表现好、累犯、有前科、弱势人员、灾害期间犯罪、认罪认罚)。

    注意：若认定[自首情节]或[坦白情节]，则不再提取[当庭自愿认罪]。
"""

    PROMPT_GENERATE_USER = """对以下案情进行分析，从中提取出犯罪主体的量刑要素(量刑要素包括:未成年、老年人、聋哑人或盲人、未遂犯、从犯、自首情节、立功情节、坦白情节、当庭自愿认罪、退赃退赔、积极赔偿被害人经济损失并取得谅解、当事人根据刑事诉讼法第二百七十七条达成刑事和解协议、羁押期间表现好、累犯、有前科、弱势人员、灾害期间犯罪、认罪认罚)：
{fact}
"""

    # 第二轮：量刑推导（原有系统提示词）
    REASONING_SYSTEM = """你是一位精通中国刑法的资深法官。你的任务是根据提供的案情、罪名法条以及量刑指导意见，严密地推导被告人的宣告刑区间。
1. 确定量刑起点：根据基本犯罪构成事实，在法定刑范围内确定初始刑期。单位统一为“月”。
2. 确定基准刑：在量刑起点上，考虑增加刑量的其他犯罪事实（如犯罪次数、数额超出起点的部分），计算出基准刑。
3. 调节情节比例：识别案件中的量刑要素，根据指导意见给出百分比加减。
4. 确定宣告刑：应用公式 $宣告刑 = 基准刑 \\times (1 + \\sum 量刑要素调节比例)$。"""

    # 第二轮用户模板（包含所有信息）
    FULL_USER_TEMPLATE = """对以下案情进行分析，根据提供的罪名、法条及指导意见，严密地推导被告人的宣告刑区间：
案情：{fact}
相关法条：{relevant_articles_combined}
量刑指导意见：
{guidance}

最终输出如下 JSON 格式：
{{
  "sentencing_chain": {{
    "starting_point": {{
      "months": "数值 <int>",
      "reasoning": "说明依据哪项事实锁定了量刑起点"
    }},
    "base_sentence": {{
      "months": "数值 <int>",
      "reasoning": "说明在量刑起点基础上，因哪些额外事实增加了多少刑期"
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
      "penalty": "最终宣告刑月份数值 <int>。无期徒刑输出 360，死刑输出 420",
      "summary": "最终裁量权的综合说明"
    }}
  }}
}}"""

    
    # 第二步：从jsonl_file加载基础案件信息
    case_dict = {}
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, start=0):
                if line.strip():
                    item = json.loads(line.strip())
                    case_id = str(item.get('id', idx))
                    case_dict[case_id] = {
                        "id": case_id,
                        "fact": item.get('fact', ''),
                        "term_info": item.get("meta", {}).get("term_of_imprisonment", {})
                    }
        print(f"✓ 加载 {len(case_dict)} 条记录")
    except Exception as e:
        print(f"✗ 加载jsonl_file失败: {e}")
    
    # 第三步：从predict_file读取预测结果并合并
    matched_count = 0
    try:
        with open(predict_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line.strip())
                case_id = str(item.get('id', ''))
                if case_id not in case_dict:
                    continue
                
                matched_count += 1
                content_str = item.get('content', '')
                pred_law_id = ""
                pred_accusation = ""
                
                match = re.search(r'\{.*\}', content_str, re.DOTALL)
                json_str = match.group(0) if match else content_str
                try:
                    content_json = json.loads(json_str)
                    pred_law_id = str(content_json.get("law_article_id", "")).strip()
                    pred_accusation = str(content_json.get("accusation", "")).strip()
                except Exception:
                    pass
                
                accusation_str = pred_accusation if pred_accusation else "未明确"
                if pred_law_id and pred_law_id in law_articles_map:
                    matched_law_text = f"【法条编号: {pred_law_id}】\n{law_articles_map[pred_law_id]}"
                elif pred_law_id:
                    matched_law_text = f"【法条编号: {pred_law_id}】 (无法条内容)"
                else:
                    matched_law_text = "(未提取到适用法条)"
                
                case_dict[case_id]["accusation"] = pred_accusation
                case_dict[case_id]["law_article_id"] = pred_law_id
                case_dict[case_id]["relevant_articles_combined"] = f"指控罪名：{accusation_str}\n相关法条序号：{pred_law_id}\n法条全文：\n{matched_law_text}"
        
        print(f"✓ 成功匹配 {matched_count} 条记录")
    except Exception as e:
        print(f"✗ 读取predict_file失败: {e}")
    
    # 第四步：构建cases_list
    cases_list = []
    unmapped_count = 0
    for case in case_dict.values():
        if "relevant_articles_combined" not in case:
            case["relevant_articles_combined"] = "(未提取到适用法条)"
            unmapped_count += 1
        
        cases_list.append({
            "id": case.get("id", ""),
            "fact": case.get("fact", ""),
            "relevant_articles_combined": case.get("relevant_articles_combined", ""),
            "term_info": case.get("term_info", {})
        })
    
    print(f"✓ 成功映射 {len(cases_list) - unmapped_count}/{len(cases_list)} 条案件")
    if unmapped_count > 0:
        print(f"有 {unmapped_count} 条案件未找到对应的预测结果")

    lora_request = LoRARequest("penalty_lora", 1, lora_adapter_path)
    results = batch_chat_with_tool(
        tokenizer, model, cases_list, sampling_params, retriever,
        gen_system=PROMPT_GENERATE_SYSTEM,
        gen_user_template=PROMPT_GENERATE_USER,
        reasoning_system=REASONING_SYSTEM,
        full_user_template=FULL_USER_TEMPLATE,
        lora_request=lora_request
    )

    # 将最终答案写回 all_cases（用于评估）
    all_cases = []
    for i, case in enumerate(cases_list):
        thinking, content = results[i]
        all_cases.append({
            **case,
            "thinking_content": thinking,
            "content": content
        })

    # ========== 评估与保存 ==========
    evaluate_accuracy(all_cases)

    os.makedirs(os.path.dirname(output_content), exist_ok=True)
    with open(output_content, 'w', encoding='utf-8') as fc, \
         open(output_thinking_content, 'w', encoding='utf-8') as ft:
        for case in all_cases:
            fc.write(json.dumps({"id": case["id"], "content": case["content"]}, ensure_ascii=False) + '\n')
            ft.write(json.dumps({"id": case["id"], "content": case["thinking_content"]}, ensure_ascii=False) + '\n')
