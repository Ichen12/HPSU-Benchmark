# -*- coding: utf-8 -*-
"""
Multiple Choice (MC) Evaluation Module

This module evaluates model predictions against a set of correct attributes/options.
It classifies answers as 'Perfect', 'Incomplete', or 'Incorrect' based on
semantic matching using an LLM evaluator with structured outputs.
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

DEFAULT_BASE_URL = 'https://api.openai-proxy.org/v1'
DEFAULT_EVAL_MODEL = 'gemini-2.0-flash-001'

# --- Data Models ---

class EvaluationResult(BaseModel):
    """
    Schema for the structured output from the LLM.
    Enforces that the result contains the 'isright' classification.
    """
    isright: str


# --- Helper Functions ---

def create_client(api_key: str, base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """Initializes and returns an OpenAI client."""
    return OpenAI(api_key=api_key, base_url=base_url)


def _attrs_to_json_str(attributes: Union[str, Dict[str, Any]]) -> str:
    """
    Safely converts attributes (which might be a dict or a string) 
    into a JSON string for prompt injection.
    """
    if isinstance(attributes, dict):
        return json.dumps(attributes, ensure_ascii=False)
    
    # If it's a string, try to normalize it if it looks like JSON
    try:
        # Check if it's valid JSON already
        parsed = json.loads(attributes)
        # Re-dump to ensure consistent formatting
        return json.dumps(parsed, ensure_ascii=False) 
    except Exception:
        # Fallback: wrap the raw string in a JSON object
        return json.dumps({"raw": attributes}, ensure_ascii=False)


def evaluate_answer_with_client(client: OpenAI,
                                model: str,
                                question: str,
                                attributes: Union[str, Dict[str, Any]],
                                prediction: str) -> Dict[str, Any]:
    """
    Evaluates a multiple-choice prediction.

    Args:
        client: The OpenAI client instance.
        model: The model to use for evaluation.
        question: The original question.
        attributes: The reference answer (usually a dictionary of options).
        prediction: The model's predicted answer.

    Returns:
        A dictionary containing "isright" ('Perfect', 'Incomplete', 'Incorrect')
        and optionally an "explanation" on error.
    """
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
        
        # Guard against None (though Pydantic validation usually catches this)
        if not parsed:
             return {"isright": "Incorrect", "explanation": "Empty response from evaluator."}

        return {"isright": parsed.isright}

    except ValidationError as ve:
        return {
            "isright": "Incorrect", 
            "explanation": f"Structured output validation failed: {ve}"
        }
    except Exception as e:
        return {
            "isright": "Incorrect", 
            "explanation": f"API call failed: {str(e)}"
        }


# --- CLI Logic ---

def _parse_args() -> argparse.Namespace:
    """Parses command line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description='Evaluate the quality of model answers (MC_eval single item)'
    )
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API Key')
    parser.add_argument('--base_url', type=str, default=DEFAULT_BASE_URL, help='API Base URL')
    parser.add_argument('--model', type=str, default=DEFAULT_EVAL_MODEL, help='Evaluator model name')
    
    parser.add_argument('--prediction', type=str, required=True, help='The model prediction text')
    parser.add_argument('--question', type=str, required=True, help='The question text')
    parser.add_argument('--attributes', type=str, required=True, help='Reference attributes (JSON string or raw text)')
    
    return parser.parse_args()


def main():
    """Main entry point for command-line testing."""
    args = _parse_args()
    client = create_client(args.api_key, args.base_url)
    
    # Attempt to parse attributes as JSON for cleaner handling, 
    # though the helper function also handles strings.
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