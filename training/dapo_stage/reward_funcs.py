import re
import math
import json
from collections import Counter


def extract_final_answer(text):
    """提取 </think> 之后的正文内容"""
    parts = text.split("</think>")
    if len(parts) > 1:
        return parts[-1].strip()
    return text.strip()


def fast_think_length_penalty_reward_func(completions, **kwargs):
    """使用字符长度估算 Token 数，避免反复调 tokenizer 拖慢速度"""
    rewards = []
    
    for comp in completions:
        content = comp if isinstance(comp, str) else comp[-1]["content"]
        match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
        
        if match:
            think_content = match.group(1)
            estimated_tokens = len(think_content) * 0.6
            
            if estimated_tokens > 1024:
                excess_tokens = estimated_tokens - 1024
                penalty = (excess_tokens // 64) * 0.05
                rewards.append(-penalty)
            elif estimated_tokens < 256:
                excess_tokens = 256 - estimated_tokens
                penalty = (excess_tokens // 64) * 0.05
                rewards.append(-penalty)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
            
    return rewards


def format_reward_func(completions, **kwargs):
    """格式奖励：检查 JSON 结构是否完整"""
    rewards = []
    for comp in completions:
        comp = comp if isinstance(comp, str) else comp[-1]["content"]
        content = extract_final_answer(comp)
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if "sentencing_chain" in data and all(key in data["sentencing_chain"] 
                    for key in ["starting_point", "base_sentence", "adjustments", "mathematical_verification", "final_penalty"]):
                    rewards.append(0.0)
                else:
                    rewards.append(-0.1)
            except json.JSONDecodeError:
                rewards.append(-0.2)
        else:
            rewards.append(-0.2)
    return rewards


def math_consistency_reward_func(completions, **kwargs):
    """检查数学计算一致性：基准刑 * (1 + sum(调节比例)) == 最终宣告刑"""
    rewards = []
    for comp in completions:
        content = comp if isinstance(comp, str) else comp[-1]["content"]
        content = extract_final_answer(content)
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                rewards.append(0.0)
                continue
                
            data = json.loads(json_match.group(0))
            chain = data.get("sentencing_chain", {})
            
            base_months = int(chain.get("base_sentence", {}).get("months", 0))
            adjustments = chain.get("adjustments", [])
            
            total_ratio = 0.0
            for adj in adjustments:
                ratio_str = str(adj.get("ratio", "0")).replace("%", "")
                total_ratio += float(ratio_str) / 100.0
                
            final_penalty = int(chain.get("final_penalty", {}).get("penalty", 0))
            expected_penalty = base_months * (1 + total_ratio)
            
            if abs(expected_penalty - final_penalty) <= 1.0:
                rewards.append(0.0)
            elif abs(expected_penalty - final_penalty) <= 2.0:
                rewards.append(0.0)
            elif abs(expected_penalty - final_penalty) <= 3.0:
                rewards.append(-0.2)
            else:
                rewards.append(-0.5)
                
        except Exception:
            rewards.append(0.0)
            
    return rewards


def log_penalty_reward_func(completions, solution, **kwargs):
    """计算宣告刑期的对数绝对误差奖励，支持死刑和无期徒刑映射"""
    rewards = []
    
    for comp, sol in zip(completions, solution):
        try:
            content = comp if isinstance(comp, str) else comp[-1]["content"]
            content = extract_final_answer(content)
            
            match = re.search(r'"penalty"\s*:\s*"?(\d+)"?', content)
            if not match:
                rewards.append(0.0)
                continue
                
            pred_months = int(match.group(1))
            sol_dict = json.loads(sol)
            
            is_gt_death = sol_dict.get("death_penalty", False)
            is_gt_life = sol_dict.get("life_imprisonment", False)
            is_pred_death = (pred_months >= 400)
            is_pred_life = (360 <= pred_months < 400)
            
            if is_gt_death and not is_pred_death:
                rewards.append(0)
                continue
            if is_gt_life and not is_pred_life:
                rewards.append(0)
                continue
            if (not is_gt_death and not is_gt_life) and (is_pred_death or is_pred_life):
                rewards.append(0)
                continue

            if is_gt_death or is_gt_life:
                rewards.append(1.0)
                continue
            
            gt_months = sol_dict.get("imprisonment", 0)
            if gt_months is None:
                gt_months = 0
                
            error = abs(math.log(1 + pred_months) - math.log(1 + gt_months))
            r_log = math.exp(-1.5 * error)
            rewards.append(float(r_log))
            
        except Exception:
            rewards.append(0.0)
            
    return rewards


def think_tag_presence_reward_func(completions, **kwargs):
    """检查是否包含思考标签"""
    rewards = []
    for comp in completions:
        content = comp if isinstance(comp, str) else comp[-1]["content"]
        
        has_open = "<think>" in content
        has_close = "</think>" in content
        
        if not (has_open and has_close):
            rewards.append(-2.0)
        else:
            rewards.append(0.0)
    
    return rewards


def repetition_penalty_reward_func(completions, **kwargs):
    """检测重复、乱码等质量问题"""
    rewards = []
    
    for comp in completions:
        content = comp if isinstance(comp, str) else comp[-1]["content"]
        output = content
        penalty = 0.0
        
        # 1. 检测字符重复
        repeated_char_pattern = r'(.)\1{4,}'
        repeated_chars = re.findall(repeated_char_pattern, output)
        if repeated_chars:
            penalty -= 0.3
        
        # 2. 检测句子重复
        sentence_list = [s.strip() for s in re.split(r'[。！？\n]', output) if len(s.strip()) > 30]
        sent_counts = Counter(sentence_list)
        repeated_sents = [s for s, count in sent_counts.items() if count >= 2]
        if repeated_sents:
            penalty -= 0.4
        
        # 3. 检测异常长的输出
        if len(output) > 2500:
            excess_length = len(output) - 2500
            penalty -= min(1.0, excess_length / 500.0)
        
        # 4. 检测标点符号重复
        crash_patterns = [
            (r'。{4,}', 0.4),
            (r'！{4,}', 0.4),
            (r'？{4,}', 0.4),
            (r'%{4,}', 0.4),
            (r'&{4,}', 0.4),
        ]
        for pattern, penalty_weight in crash_patterns:
            if re.search(pattern, output):
                penalty -= penalty_weight
        
        # 5. 检测英文乱码
        long_english = re.findall(r'[a-zA-Z]{15,}', output)
        if long_english:
            penalty -= min(0.9, len(long_english) * 0.3)
        
        english_chars = len(re.findall(r'[a-zA-Z]', output))
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', output))
        
        if english_chars > 0 and chinese_chars > 0:
            english_ratio = english_chars / (english_chars + chinese_chars)
            if english_ratio > 0.4:
                penalty -= 0.4
        elif english_chars > 500 and chinese_chars == 0:
            penalty -= 0.8
        
        gibberish_patterns = [
            r'[a-z]{2,}([a-z]{2})\1{3,}',
            r'([a-zA-Z])\1{5,}',
            r'[a-zA-Z]*[0-9]{10,}[a-zA-Z]*',
        ]
        for pattern in gibberish_patterns:
            if re.search(pattern, output):
                penalty -= 0.4
        
        rewards.append(max(-2.0, penalty))
    
    return rewards


def range_accuracy_reward_func(completions, solution, **kwargs):
    """区间准确率阶梯奖励函数"""
    rewards = []

    def map_penalty_to_range(penalty):
        if penalty >= 360:
            return 10
        months = penalty
        if months == -1: return -1
        elif months <= 6: return 0
        elif months <= 9: return 1
        elif months <= 12: return 2
        elif months <= 24: return 3
        elif months <= 36: return 4
        elif months <= 60: return 5
        elif months <= 84: return 6
        elif months <= 120: return 7
        elif months <= 180: return 8
        else: return 9

    for comp, sol in zip(completions, solution):
        try:
            content = comp if isinstance(comp, str) else comp[-1]["content"]
            parts = content.split("</think>")
            final_content = parts[-1].strip() if len(parts) > 1 else content.strip()
            
            match = re.search(r'"penalty"\s*:\s*"?(\d+)"?', final_content)
            if not match:
                rewards.append(0.0)
                continue
            pred_months = int(match.group(1))
            
            sol_dict = json.loads(sol)
            if sol_dict.get("death_penalty", False):
                gt_months = 420
            elif sol_dict.get("life_imprisonment", False):
                gt_months = 360
            else:
                gt_months = sol_dict.get("imprisonment", 0)
                if gt_months is None: 
                    gt_months = 0
            
            pred_range = map_penalty_to_range(pred_months)
            gt_range = map_penalty_to_range(gt_months)
            
            if pred_range == -1 or gt_range == -1:
                rewards.append(0.0)
                continue
                
            diff = abs(pred_range - gt_range)
            if diff == 0:
                rewards.append(0.5)
            elif diff == 1:
                rewards.append(0.25)
            elif diff == 2:
                rewards.append(0.0)
            else:
                rewards.append(-0.05 * diff)
                
        except Exception:
            rewards.append(0.0)
            
    return rewards