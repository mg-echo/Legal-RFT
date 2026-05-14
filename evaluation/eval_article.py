from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import json
import os
import re

# 标准法条集合
STANDARD_ARTICLES = {"133", "125", "233", "225", "345", "114", "266", "274", "292", "313", "280", "232", "115", "303", "134", "224", "397", "214", "276", "338", "143", "213", "234", "144", "141", "336", "351", "341", "342", "344", "176", "236", "175", "205", "272", "271", "312", "293", "275", "128", "310", "384", "163", "245", "140", "238", "235", "177", "393", "153", "124", "340", "237", "279", "118", "196", "343", "240", "277", "348", "267", "258", "239", "210", "135", "314", "192", "172", "291", "290", "328", "217", "198", "228", "151", "150", "226", "372", "186", "171", "193", "350", "194", "243", "164", "262", "130", "307", "392", "253", "305", "261", "209", "223", "417", "382", "158", "387", "132", "391", "246", "215"}


def extract_pred_articles(content_str):
    pred_set = set()
    parse_error = False
    try:
        content_json = json.loads(content_str)
        law_field = content_json.get('law_article_id', '')
        if isinstance(law_field, list):
            pred_set = set(str(x).strip() for x in law_field if str(x).strip())
        else:
            matches = re.findall(r"\d+", str(law_field))
            if matches:
                pred_set = set(matches)
            else:
                s = str(law_field).strip()
                if s:
                    pred_set = {s}
    except Exception:
        matches = re.findall(r"\d+", content_str)
        if matches:
            pred_set = set(matches)
        else:
            parse_error = True

    return pred_set, parse_error


def evaluate_accuracy(ground_truth_file, predict_file):
    # 加载真实标签 
    gt_map = {}
    print(f"正在读取真实标签文件: {ground_truth_file} ...")
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    case_id = str(item.get('id', ''))
                    meta = item.get('meta', {})
                    relevant_articles = meta.get('relevant_articles', [])
                    gt_map[case_id] = [str(x) for x in relevant_articles]
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"读取真实标签文件出错: {e}")
        return

    print(f"正在读取预测结果文件: {predict_file} ...")
    correct_count = 0
    total_count = 0
    error_count = 0
    out_of_range_count = 0

    tp_counts = {}
    fp_counts = {}
    fn_counts = {}
    support_counts = {}
    invalid_predictions = []
    
    try:
        with open(predict_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    case_id = str(item.get('id', ''))
                    content_str = item.get('content', '')

                    if case_id not in gt_map:
                        continue

                    total_count += 1

                    pred_set, parse_error = extract_pred_articles(content_str)
                    if parse_error:
                        error_count += 1

                    invalid_pred = pred_set - STANDARD_ARTICLES
                    valid_pred = pred_set & STANDARD_ARTICLES
                    
                    if invalid_pred:
                        out_of_range_count += 1
                        invalid_predictions.append({
                            'id': case_id,
                            'invalid': sorted(invalid_pred),
                            'valid': sorted(valid_pred)
                        })

                    true_set = set(gt_map.get(case_id, []))

                    case_correct = False
                    if valid_pred and true_set:
                        if true_set & valid_pred:
                            case_correct = True

                    if case_correct:
                        correct_count += 1

                    eval_labels = true_set.union(valid_pred)
                    for label in eval_labels:
                        if label not in STANDARD_ARTICLES:
                            continue

                        in_true = label in true_set
                        in_pred = label in valid_pred

                        if in_true:
                            support_counts[label] = support_counts.get(label, 0) + 1

                        if in_pred and in_true:
                            tp_counts[label] = tp_counts.get(label, 0) + 1
                        elif in_pred and not in_true:
                            fp_counts[label] = fp_counts.get(label, 0) + 1
                        elif in_true and not in_pred:
                            fn_counts[label] = fn_counts.get(label, 0) + 1

                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"读取预测结果文件出错: {e}")
        return

    if total_count == 0:
        return

    accuracy = correct_count / total_count * 100

    labels = list(support_counts.keys())
    macro_p = macro_r = macro_f1 = 0.0
    if labels:
        sum_p = sum_r = sum_f = 0.0
        for lbl in labels:
            tp = tp_counts.get(lbl, 0)
            fp = fp_counts.get(lbl, 0)
            fn = fn_counts.get(lbl, 0)

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

            sum_p += p
            sum_r += r
            sum_f += f

        macro_p = sum_p / len(labels) * 100
        macro_r = sum_r / len(labels) * 100
        macro_f1 = sum_f / len(labels) * 100

    print("\n" + "="*40)
    print("评测结果统计")
    print(f"总计匹配案件数: {total_count}")
    print(f"法条预测正确数: {correct_count}")
    print(f"JSON解析异常数: {error_count}")
    print(f"包含非标准法条的预测数: {out_of_range_count}") 
    print(f"Accuracy: {accuracy:.2f}%")
    if labels:
        print(f"参与计算的标准法条数: {len(labels)}")
        print(f"Macro Precision : {macro_p:.2f}%")
        print(f"Macro Recall : {macro_r:.2f}%")
        print(f"Macro F1 : {macro_f1:.2f}%")
    else:
        print("未检测到任何法条标签用于计算 Macro 指标。")



def batch_chat_with_model(tokenizer, model, prompts, sampling_params, lora_path=None, IF_THINKING=True):
    texts = []
    for p in prompts:
        messages = [
            {"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user"]}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=IF_THINKING,
        )
        texts.append(text)
    
    lora_request = LoRARequest("adapter", 1, lora_path) if lora_path else None
    outputs = model.generate(texts, sampling_params, lora_request=lora_request, use_tqdm=True)
    
    think_end_token_id = None
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        think_end_token_id = tokenizer.convert_tokens_to_ids('</think>')

    results = []
    for out in outputs:
        output_ids = list(out.outputs[0].token_ids)
        index = 0
        
        for token_id in [151668, think_end_token_id]:
            if token_id is None:
                continue
            try:
                index = len(output_ids) - output_ids[::-1].index(token_id)
                break
            except ValueError:
                continue

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        results.append((thinking_content, content))
        
    return results

def load_top5_map(top5_file):
    top5_map = {}
    try:
        with open(top5_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    fid = str(item.get('id', ''))
                    meta = item.get('meta', [])
                    parts = [json.dumps(m, ensure_ascii=False) for m in meta]
                    top5_map[fid] = '\n'.join(parts)
                except Exception:
                    continue
    except Exception as e:
        print(f"加载 top5 文件失败: {e}")
    return top5_map
    


if __name__ == "__main__":
    model_path = "your_model_path"
    lora_adapter_path = "your_lora_adapter_path"
    model_name = "Qwen3-8B"

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


    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        top_k=30,
        min_p=0,
        max_tokens=4096
    )
    top5_map = load_top5_map(r"your_mapping_file")
    jsonl_file = r"your_cases_file" 
    output_file_content = f"predict_article_{model_name}.jsonl"
    output_file_thinking_content = f"predict_article_thinking_{model_name}.jsonl"

    all_cases = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=0):
            try:
                item = json.loads(line.strip())
                all_cases.append({
                    "id": item.get('id', idx),
                    "fact": item.get('fact', ''),
                    "top5_laws": top5_map.get(str(item.get('id', idx)), '')
                })
            except Exception as e:
                print(f"✗ Parse error for line {idx}: {e}")



    system_prompt = """
你是一位精通中国刑法的专家级助理，擅长进行刑事案件的法条匹配和罪名分析。

遍历案件相关法条，依次分析其是否与案件事实、案件相关要素完全匹配，排除不匹配的法条，推荐与裁定此案最相关的法条。可略过明显不符的法条，针对其中可能符合的法条进行以下法律三段论推理：
1. 拆解大前提：提取该法条的全部客观构成要件与主观构成要件，特别关注法条中前置性条件（如特定上游犯罪类型、特定目的、特殊主体身份等）。
2. 比对小前提：将法条的构成要件与案件事实、案件相关要素进行逐一比对，分析是否满足法条的适用条件，特别关注案件中是否存在法条所要求的特定情形或限定条件。
3. 决策判断：若任一要件未被案情事实满足，则直接排除该法条。最终选出的与裁定此案相关的法条。
注意：
1. 严禁仅凭关键词相似或行为外观相似即认定匹配，必须以构成要件的实质符合性为准。
2. 你的法条序号必须来自候选法条列表，禁止输出候选法条之外的法条序号。
"""
    prompts = []
    for case in all_cases:
        p = f"""
请严格按照“法律三段论”逻辑，对以下案情进行深度辨析，并从候选法条中筛选出最准确的法条输出\n
案情：{case['fact']}
候选法条：{case['top5_laws']}
Json Output：{{"reason":"判断理由", "law_article_id": "最终选定的法条编号数字","accusation":"最终选定的罪名"}}
"""
        prompts.append({"system": system_prompt, "user": p})

    results_2 = batch_chat_with_model(tokenizer, model, prompts, sampling_params)
    for i, case in enumerate(all_cases):
        case["content"] = results_2[i][1]
        case["thinking_content"] = results_2[i][0]


    print(f"\n================ 正在落盘写文件 ================")
    with open(output_file_content, 'w', encoding='utf-8') as file, \
         open(output_file_thinking_content, 'w', encoding='utf-8') as file_think:
        for case in all_cases:
            file.write(json.dumps({
                "id": case["id"],
                "content": case["content"]
            }, ensure_ascii=False) + '\n')

            file_think.write(json.dumps({
                "id": case["id"],
                "content": case["thinking_content"]
            }, ensure_ascii=False) + '\n')

    evaluate_accuracy(jsonl_file, output_file_content)