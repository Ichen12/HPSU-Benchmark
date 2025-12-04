# -*- coding: utf-8 -*-
"""
Semantic Consistency (SC) Evaluation Module with Type Classification

This module evaluates a model's prediction against a structured reference set
(attributes) containing categories like 'true', 'similar', 'middle', and 'opposite'.

It determines:
1. `isright`: Whether the answer corresponds to the 'true' option.
2. `types`: The specific category of the answer (right, similar, middle, opposite, or blank).
"""

import argparse
import json
from typing import Dict, Any, Union, Optional
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# --- Constants ---

PROMPT_TEMPLATE = """
[Question]
{question}

[Reference Answer]
{attributes_json}

[Model Answer]
{prediction}

[Evaluation Rules]
1. Determine **isright**
   - If the model-selected option matches the value under the key "true" in the [Reference Answer], set isright = "right".
   - Otherwise (including refusal, no answer, or irrelevant text), set isright = "wrong".

2. Determine **types**
   - If the answer matches "true"     → types = "right"
   - If the answer matches "similar"  → types = "similar"
   - If the answer matches "middle"   → types = "middle"
   - If the answer matches "opposite" → types = "opposite"
   - If no valid answer is given or the answer is unrecognisable → types = "blank"

3. Ignore case, spacing, punctuation, and list formatting—evaluate only the meaning of the chosen option.

[Output Format]
Return strictly the following JSON object:
{{
    "isright": "right" or "wrong",
    "types": "right" or "similar" or "middle" or "opposite" or "blank"
}}
"""

DEFAULT_BASE_URL = 'https://api.openai-proxy.org/v1'
DEFAULT_EVAL_MODEL = 'gemini-2.0-flash-001'

# --- Data Models ---

class EvaluationResult(BaseModel):
    """
    Schema for the structured output from the LLM.
    """
    isright: str
    types: str


# --- Helper Functions ---

def create_client(api_key: str, base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """
    Initializes and returns an OpenAI client.
    """
    return OpenAI(api_key=api_key, base_url=base_url)


def _ensure_attributes_json(attributes: Union[str, Dict[str, Any]]) -> str:
    """
    Ensures the attributes are formatted as a JSON string for prompt injection.
    
    Args:
        attributes: Either a dictionary or a string representation of JSON.
    """
    if isinstance(attributes, dict):
        return json.dumps(attributes, ensure_ascii=False)
    
    # If it's already a string, try to validate it as JSON
    try:
        # Check if valid JSON
        parsed = json.loads(attributes)
        # Re-dump to ensure consistent formatting
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        # Fallback: Wrap raw string in a dictionary structure
        return json.dumps({"raw": attributes}, ensure_ascii=False)


def evaluate_answer_with_client(client: OpenAI,
                                model: str,
                                question: str,
                                attributes: Union[str, Dict[str, Any]],
                                prediction: str) -> Dict[str, Any]:
    """
    Evaluates the prediction against structured attributes.

    Args:
        client: OpenAI client instance.
        model: Model name for evaluation.
        question: The original question.
        attributes: Reference dictionary (e.g., {"true": "A", "similar": "B"}).
        prediction: The model's answer.

    Returns:
        A dictionary containing 'isright', 'types', and optional 'explanation'.
    """
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
        
        if parsed:
            return {"isright": parsed.isright, "types": parsed.types}
        else:
            return {
                "isright": "wrong", 
                "types": "blank", 
                "explanation": "Empty response from evaluator."
            }
            
    except ValidationError as ve:
        return {
            "isright": "wrong", 
            "types": "blank",
            "explanation": f"Structured output validation failed: {ve}"
        }
    except Exception as e:
        return {
            "isright": "wrong", 
            "types": "blank",
            "explanation": f"API call failed: {str(e)}"
        }


# --- CLI Logic ---

def _parse_args() -> argparse.Namespace:
    """Parses command line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description='Evaluate model answer quality with type classification (SC_type_eval single item)'
    )
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API Key')
    parser.add_argument('--base_url', type=str, default=DEFAULT_BASE_URL, help='API Base URL')
    parser.add_argument('--model', type=str, default=DEFAULT_EVAL_MODEL, help='Evaluator model name')
    
    parser.add_argument('--prediction', type=str, required=True, help='The model prediction text (formerly --pre)')
    parser.add_argument('--question', type=str, required=True, help='The question text')
    parser.add_argument('--attributes', type=str, required=True, 
                        help='Reference attributes (JSON string, e.g., {"true":"A", "similar":"B"})')
    
    return parser.parse_args()


def main():
    """Main entry point for command-line testing."""
    args = _parse_args()
    client = create_client(args.api_key, args.base_url)
    
    # Handle potential JSON parsing for the attributes argument
    try:
        attrs = json.loads(args.attributes)
    except json.JSONDecodeError:
        attrs = args.attributes

    result = evaluate_answer_with_client(
        client=client,
        model=args.model,
        question=args.question,
        attributes=attrs,
        prediction=args.prediction
    )
    
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()