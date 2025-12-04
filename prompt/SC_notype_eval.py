# -*- coding: utf-8 -*-
"""
Semantic Consistency (SC) Evaluation Module (No-Type)

This module evaluates the correctness of a model's prediction against a reference answer
based purely on semantic meaning, ignoring strict formatting or type constraints.
It uses an LLM evaluator to determine if the core information matches the question context.
"""

import argparse
import json
from typing import Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# --- Constants ---

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

DEFAULT_BASE_URL = 'https://api.openai-proxy.org/v1'
DEFAULT_EVAL_MODEL = 'gemini-2.0-flash-001'

# --- Data Models ---

class EvaluationResult(BaseModel):
    """
    Schema for the structured output from the LLM.
    Enforces that the result contains the 'isright' classification.
    """
    isright: str


# --- Core Functions ---

def create_client(api_key: str, base_url: Optional[str] = None) -> OpenAI:
    """
    Initializes and returns an OpenAI client.
    
    Args:
        api_key: The API key for authentication.
        base_url: Optional custom base URL.
    """
    return OpenAI(api_key=api_key, base_url=base_url)


def evaluate_answer_with_client(client: OpenAI,
                                model: str,
                                question: str,
                                answer: str,
                                prediction: str) -> Dict[str, Any]:
    """
    Evaluates the semantic consistency of a prediction against a reference answer.

    Args:
        client: The OpenAI client instance.
        model: The model to use for evaluation.
        question: The original question.
        answer: The reference correct answer.
        prediction: The model's predicted answer.

    Returns:
        A dictionary containing "isright" ('right' or 'wrong') and optional explanation.
    """
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
        
        # Verify parsed is not None
        if parsed:
            return {"isright": parsed.isright}
        else:
            return {"isright": "wrong", "explanation": "Empty response from evaluator."}
            
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


# --- CLI Logic ---

def _parse_args() -> argparse.Namespace:
    """Parses command line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description='Evaluate model answer quality based on semantic consistency (SC_notype_eval single item)'
    )
    
    # Removed hardcoded sensitive API keys. Users should provide them via args or env vars.
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API Key')
    parser.add_argument('--base_url', type=str, default=DEFAULT_BASE_URL, help='API Base URL')
    parser.add_argument('--model', type=str, default=DEFAULT_EVAL_MODEL, help='Evaluator model name')
    
    parser.add_argument('--prediction', type=str, required=True, help='The model prediction text (formerly --pre)')
    parser.add_argument('--question', type=str, required=True, help='The question text')
    parser.add_argument('--answer', type=str, required=True, help='The reference answer text')
    
    return parser.parse_args()


def main():
    """Main entry point for command-line testing."""
    args = _parse_args()
    client = create_client(args.api_key, args.base_url)
    
    result = evaluate_answer_with_client(
        client=client,
        model=args.model,
        question=args.question,
        answer=args.answer,
        prediction=args.prediction
    )
    
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()