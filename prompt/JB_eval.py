# -*- coding: utf-8 -*-
"""
JB Evaluation Module

This module evaluates a model's ability to distinguish between "right" content
and "distractor" content based on a "yes/no" judgment. It uses the OpenAI API
with structured outputs (Pydantic) to ensure reliable parsing.
"""

import argparse
import json
from typing import Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# Template used to construct the prompt for the evaluator model.
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
    """
    Schema for the structured output from the LLM.
    Enforces that the result contains an 'isright' field.
    """
    isright: str


def create_client(api_key: str, base_url: Optional[str] = None) -> OpenAI:
    """
    Initializes and returns an OpenAI client.
    
    Args:
        api_key: The API key for authentication.
        base_url: Optional custom base URL (e.g., for proxies).
    """
    return OpenAI(api_key=api_key, base_url=base_url)


def evaluate_answer_with_client(client: OpenAI,
                                model: str,
                                question: str,
                                selected_field: str,
                                prediction: str) -> Dict[str, Any]:
    """
    Evaluates a single prediction using an LLM.

    Args:
        client: The OpenAI client instance.
        model: The name of the model to use for evaluation.
        question: The original question text.
        selected_field: Indicates if the target is 'right' or 'distractor'.
        prediction: The model's answer ('yes', 'no', or equivalent).

    Returns:
        A dictionary containing the evaluation result (e.g., {"isright": "right"})
        or an explanation if an error occurred.
    """
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
        # Verify parsed is not None (though Pydantic usually guarantees validity or raises error)
        if parsed:
            return {"isright": parsed.isright}
        else:
            return {"isright": "wrong", "explanation": "Empty response from model."}
            
    except ValidationError as ve:
        return {
            "isright": "wrong", 
            "explanation": f"Structured output validation failed: {ve}"
        }
    except Exception as e:
        return {
            "isright": "wrong", 
            "explanation": f"API call failed: {str(e)}"
        }


def _parse_args() -> argparse.Namespace:
    """Parses command line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description='Evaluate model understanding of right/distractor (JB_eval single item)'
    )
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API Key')
    parser.add_argument('--base_url', type=str, default=None, help='API Base URL')
    parser.add_argument('--model', type=str, default='', help='Evaluator model name')
    parser.add_argument('--prediction', type=str, required=True, help='The model prediction to evaluate')
    parser.add_argument('--question', type=str, required=True, help='The question text')
    parser.add_argument('--selected_field', type=str, required=True, help='The field type (right/distractor)')
    
    return parser.parse_args()


def main():
    """Main entry point for command-line testing."""
    args = _parse_args()
    client = create_client(args.api_key, args.base_url)
    
    result = evaluate_answer_with_client(
        client=client, 
        model=args.model, 
        question=args.question, 
        selected_field=args.selected_field, 
        prediction=args.prediction
    )
    
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()