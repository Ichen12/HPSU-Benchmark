# -*- coding: utf-8 -*-
# 路径: /data/lichen/A_benchmark/improve/code/eval/en/SC_notype_eval.py
import argparse
import json
from typing import Optional, Dict, Any
from openai import OpenAI
from pydantic import BaseModel, ValidationError

PROMPT_TEMPLATE = """
[Question]
{question}

[Reference Answer]
{answer}

[Model Answer]
{prediction}

[Task]
- Judge whether the [Model Answer] is correct **based on its semantic meaning** with respect to the [Reference Answer] **and the [Question]**.
- The inclusion of additional explanations or context in the [Model Answer] does not affect the judgment of its correctness.
- Do NOT judge by surface similarity, sentence structure, formatting, case, punctuation, extra explanations, or writing style.
- When evaluating, prioritize core information extraction and allow for reasonable redundancy.
- Only the meaning, in the context of the question, should be considered. Explanations or irrelevant content should not affect your judgment, unless they contradict or fundamentally alter the correct core information.
- For example, if the [Question] asks "What is the speaker's gender?", both "female" and "I think the speaker is female" should be considered correct and semantically consistent.
- If the [Model Answer] correctly answers the [Question] with the same essential meaning as the [Reference Answer], output "right". Otherwise, output "wrong".
- Output strictly in the following JSON format and nothing else:

{{
    "isright": "right" or "wrong"
}}
"""

class EvaluationResult(BaseModel):
    isright: str

def create_client(api_key: str, base_url: str) -> OpenAI:
    """预初始化客户端，便于批量调用时复用连接。"""
    return OpenAI(api_key=api_key, base_url=base_url)

def evaluate_answer_with_client(client: OpenAI,
                                model: str,
                                question: str,
                                answer: str,
                                prediction: str) -> Dict[str, Any]:
    """核心评测函数：传入已初始化的 client，避免重复建连。"""
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
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

# 兼容命令行单条评测（可选）
def _parse_args():
    p = argparse.ArgumentParser(description='评估模型答案的质量（SC_notype_eval 单条）')
    p.add_argument('--api_key', type=str, default='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')
    p.add_argument('--base_url', type=str, default='https://openrouter.ai/api/v1')
    p.add_argument('--model', type=str, default='google/gemini-2.5-flash-preview-09-2025')
    p.add_argument('--pre', type=str, required=True, help='模型预测答案')
    p.add_argument('--question', type=str, required=True)
    p.add_argument('--answer', type=str, required=True)
    return p.parse_args()

def main():
    args = _parse_args()
    client = create_client(args.api_key, args.base_url)
    result = evaluate_answer_with_client(
        client=client,
        model=args.model,
        question=args.question,
        answer=args.answer,
        prediction=args.pre
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()