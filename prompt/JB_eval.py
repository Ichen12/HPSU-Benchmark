# -*- coding: utf-8 -*-
# 路径: /data/lichen/A_benchmark/improve/code/eval/en/JB_eval.py
import argparse
import json
from typing import Dict, Any
from openai import OpenAI
from pydantic import BaseModel, ValidationError

PROMPT_TEMPLATE = """
[Question]
{question}

[Selected Field]
{selected_field}

[Model Answer]
{prediction}

[Task]
You are an evaluator who determines whether the model's answer is correct based on the selected field and the model's answer.

There are two possible values for the selected field:
- "right": This indicates the content is the correct.
- "distractor": This indicates the content is a distractor.

The model's answer is either:
- "yes": The model agrees with the content.
- "no": The model disagrees with the content.

[Evaluation Rule]
- If the selected field is "right" and the model's answer is "yes", it is "right".
- If the selected field is "right" and the model's answer is "no", it is "wrong".
- If the selected field is "distractor" and the model's answer is "yes", it is "wrong".
- If the selected field is "distractor" and the model's answer is "no", it is "right".
- The format of Model Answer is not restricted, and any content with the same meaning is acceptable.
- If the model's response is unable to make a judgment or refuses to answer, return "wrong".

[Output Format]
Format your response strictly as the following JSON:
{{
    "isright": "right" or "wrong"
}}
Do not output anything else.
"""

class EvaluationResult(BaseModel):
    isright: str

def create_client(api_key: str, base_url: str = 'https://api.openai-proxy.org/v1') -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)

def evaluate_answer_with_client(client: OpenAI,
                                model: str,
                                question: str,
                                selected_field: str,
                                prediction: str) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        selected_field=selected_field,
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
        return {"isright": "wrong", "explanation": f"结构化输出验证失败: {ve}"}
    except Exception as e:
        return {"isright": "wrong", "explanation": f"API 调用失败: {str(e)}"}

# 可选：命令行单条调试
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description='评估模型对于 right/distractor 的理解（JB_eval 单条）')
    p.add_argument('--api_key', type=str, required=True)
    p.add_argument('--base_url', type=str, default='https://api.openai-proxy.org/v1')
    p.add_argument('--model', type=str, default='gemini-2.0-flash-001')
    p.add_argument('--pre', type=str, required=True)
    p.add_argument('--question', type=str, required=True)
    p.add_argument('--selected_field', type=str, required=True)
    return p.parse_args()

def main():
    args = _parse_args()
    client = create_client(args.api_key, args.base_url)
    result = evaluate_answer_with_client(client, args.model, args.question, args.selected_field, args.pre)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()