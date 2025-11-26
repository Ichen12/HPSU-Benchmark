# -*- coding: utf-8 -*-
# 路径: /data/lichen/A_benchmark/improve/code/eval/en/SC_type_eval.py
import argparse
import json
from typing import Dict, Any, Union
from openai import OpenAI
from pydantic import BaseModel, ValidationError

PROMPT_TEMPLATE = """
[Question]
{question}

[Reference Answer]
{attributes_json}

[Model Answer]
{prediction}

[Evaluation Rules]
1. Determine **isright**
   - If the model-selected option matches the value under "true", set isright = "right".
   - Otherwise (including refusal, no answer, or irrelevant text), set isright = "wrong".

2. Determine **types**
   - If isright = "right"               → types = "right"
   - If the [Model Answer] is in "similar" → types = "similar"
   - If the [Model Answer] is in "middle"  → types = "middle"
   - If the [Model Answer] is in "opposite"→ types = "opposite"
   - If no valid [Model Answer] is given or the answer is unrecognisable → types = "blank"

3. Ignore case, spacing, punctuation, and list formatting—evaluate only the meaning of the chosen option.

[Output Format]
Return only the JSON object:
{{
    "isright": "right" or "wrong",
    "types": "right" or "similar" or "middle" or "opposite" or "blank"
}}
"""

class EvaluationResult(BaseModel):
    isright: str
    types: str

def create_client(api_key: str,
                  base_url: str = 'https://api.openai-proxy.org/v1') -> OpenAI:
    """预初始化客户端，便于批量复用。"""
    return OpenAI(api_key=api_key, base_url=base_url)

def _ensure_attributes_json(attributes: Union[str, Dict[str, Any]]) -> str:
    """将 attributes 转成 JSON 字符串注入提示词。"""
    if isinstance(attributes, dict):
        return json.dumps(attributes, ensure_ascii=False)
    # 可能传入已是 json 字符串或错误字符串，尽量原样塞进去
    try:
        _ = json.loads(attributes)
        return attributes
    except Exception:
        # 兜底包一层
        return json.dumps({"raw": attributes}, ensure_ascii=False)

def evaluate_answer_with_client(client: OpenAI,
                                model: str,
                                question: str,
                                attributes: Union[str, Dict[str, Any]],
                                prediction: str) -> Dict[str, Any]:
    """核心评测：返回 {isright, types, [explanation]}"""
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        attributes_json=_ensure_attributes_json(attributes),
        prediction=prediction
    )
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=EvaluationResult,
        )
        parsed = completion.choices[0].message.parsed
        return {"isright": parsed.isright, "types": parsed.types}
    except ValidationError as ve:
        return {"isright": "wrong", "types": "blank",
                "explanation": f"结构化输出验证失败: {ve}"}
    except Exception as e:
        return {"isright": "wrong", "types": "blank",
                "explanation": f"API 调用失败: {str(e)}"}

# 兼容命令行单条调试（可选）
def _parse_args():
    p = argparse.ArgumentParser(description='评估模型答案的质量（SC_type_eval 单条）')
    p.add_argument('--api_key', type=str, required=True)
    p.add_argument('--base_url', type=str, default='https://api.openai-proxy.org/v1')
    p.add_argument('--model', type=str, default='gemini-2.0-flash-001')
    p.add_argument('--pre', type=str, required=True, help='模型预测答案')
    p.add_argument('--question', type=str, required=True)
    p.add_argument('--attributes', type=str, required=True,
                   help='参考答案(JSON字符串，如 {"true":"A","similar":"B","middle":"C","opposite":"D"})')
    return p.parse_args()

def main():
    args = _parse_args()
    client = create_client(args.api_key, args.base_url)
    # 允许 attributes 是 JSON 字符串或普通字符串
    try:
        attrs = json.loads(args.attributes)
    except json.JSONDecodeError:
        attrs = args.attributes
    result = evaluate_answer_with_client(
        client=client,
        model=args.model,
        question=args.question,
        attributes=attrs,
        prediction=args.pre
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()