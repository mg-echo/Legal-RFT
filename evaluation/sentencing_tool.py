import json
from difflib import SequenceMatcher
from typing import List, Dict, Optional

class SentencingGuideRetriever:
    def __init__(self, guideline_path: str):
        self.guidelines: Dict[str, str] = {}
        self.elements: List[str] = [] 
        with open(guideline_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError:
                    print(f"[警告] 第 {line_no} 行 JSON 解析失败，已跳过")
                    continue
                elem = (item.get("element") or item.get("factor") or
                        item.get("name") or "").strip()
                if not elem:
                    print(f"[警告] 第 {line_no} 行找不到要素字段，已跳过")
                    continue
                content = (item.get("content") or item.get("guidance") or item.get("text") or "").strip()
                self.guidelines[elem] = content
                self.elements.append(elem)
        print(f"[Tools] 已加载 {len(self.guidelines)} 条量刑指导要素")

    def _best_match(self, query: str, threshold: float = 0.6) -> Optional[str]:
        best_score = 0.0
        best_elem = None
        for elem in self.elements:
            score = SequenceMatcher(None, query, elem).ratio()
            if query in elem or elem in query:
                score = max(score, 0.85)
            if score > best_score:
                best_score = score
                best_elem = elem
        if best_score >= threshold:
            return best_elem
        return None

    def retrieve(self, elements: List[str]) -> str:
        if not elements:
            return "本案未识别出量刑要素（或要素列表为空），请结合法定情节酌情推导。"
        results = []
        for raw_elem in elements:
            raw_elem = raw_elem.strip()
            matched = self._best_match(raw_elem)
            if matched:
                results.append(f"【{matched}】{self.guidelines[matched]}")
            else:
                results.append(f"【{raw_elem}】未找到对应的指导意见，请结合法律规定酌情处理。")
        return "\n".join(results)


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "retrieve_sentencing_guidance",
        "description": "检索与量刑要素相关的指导意见。即使未发现量刑要素，也必须调用，并传入空列表。",
        "parameters": {
            "type": "object",
            "properties": {
                "sentencing_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "识别出的量刑要素名称列表（如：['自首', '累犯']）。若无要素，请传入 []。"
                }
            },
            "required": ["sentencing_factors"]
        },
        "strict": True
    }
}

def build_tool_system_prompt(base_system: str) -> str:
    func_def = TOOL_DEFINITION["function"]
    tool_name = func_def["name"]
    
    example_json = {
        "name": tool_name,
        "arguments": {"sentencing_factors": ["要素1", "要素2"]}
    }
    
    return (
        base_system
        + "\n\n你必须使用工具检索量刑指导意见。"
        + "你的最终输出只能是一个纯JSON对象，格式如下，不要包含任何其他文字、标记或解释：\n"
        + json.dumps(example_json, ensure_ascii=False) + "\n"
        + "如果没有识别到任何要素，请输出 arguments 中的 sentencing_factors 为空数组 []。"
    )