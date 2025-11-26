# -*- coding: utf-8 -*-
# 路径: /data/lichen/A_benchmark/improve/code/eval/en/MC_eval.py
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

Please evaluate the model's answer based on the selected options only, regardless of the format of the response.

Rules:
- If the model's selected options match exactly the correct options ("true") → isright = "Perfect".
- If the model includes some correct options, includes NO incorrect options, but misses some correct ones → isright = "Incomplete".
- If the model selects any incorrect options, refuses to answer, or is unrecognisable → isright = "Incorrect".

Ignore case, punctuation, order, extra wording, and formatting. Judge purely by semantic option identity.

Return ONLY this JSON object:
{{
    "isright": "Perfect" or "Incomplete" or "Incorrect"
}}
"""

class EvaluationResult(BaseModel):
    isright: str

def create_client(api_key: str, base_url: str = 'https://api.openai-proxy.org/v1') -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)

def _attrs_to_json_str(attributes: Union[str, Dict[str, Any]]) -> str:
    if isinstance(attributes, dict):
        return json.dumps(attributes, ensure_ascii=False)
    try:
        _ = json.loads(attributes)
        return attributes
    except Exception:
        return json.dumps({"raw": attributes}, ensure_ascii=False)

def evaluate_answer_with_client(client: OpenAI,
                                model: str,
                                question: str,
                                attributes: Union[str, Dict[str, Any]],
                                prediction: str) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        attributes_json=_attrs_to_json_str(attributes),
        prediction=prediction
    )
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=EvaluationResult,
        )
        parsed = completion.choices[0].message.parsed
        return {"isright": parsed.isright}
    except ValidationError as ve:
        return {"isright": "Incorrect", "explanation": f"结构化输出验证失败: {ve}"}
    except Exception as e:
        return {"isright": "Incorrect", "explanation": f"API 调用失败: {str(e)}"}

# 可选：命令行单条调试
def _parse_args():
    p = argparse.ArgumentParser(description='评估模型答案的质量（MC_eval 单条）')
    p.add_argument('--api_key', type=str, required=True)
    p.add_argument('--base_url', type=str, default='https://api.openai-proxy.org/v1')
    p.add_argument('--model', type=str, default='gemini-2.0-flash-001')
    p.add_argument('--pre', type=str, required=True)
    p.add_argument('--question', type=str, required=True)
    p.add_argument('--attributes', type=str, required=True)
    return p.parse_args()

def main():
    args = _parse_args()
    client = create_client(args.api_key, args.base_url)
    try:
        attrs = json.loads(args.attributes)
    except json.JSONDecodeError:
        attrs = args.attributes
    result = evaluate_answer_with_client(client, args.model, args.question, attrs, args.pre)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()