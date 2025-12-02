# -*- coding: utf-8 -*-
"""
Main entry point for the evaluation pipeline.

This script processes a list of prediction results, evaluates them using specific 
prompt-based metrics (SC, MC, JB), and aggregates the statistics.
"""

import argparse
import json
import csv
import os
import time
from typing import List, Dict, Any, Tuple, Optional, Callable

# Import prompt-based evaluation modules
# Ensure these modules exist in your project structure
from prompt.SC_notype_eval import create_client as create_client_notype, evaluate_answer_with_client as eval_notype
from prompt.SC_type_eval   import evaluate_answer_with_client as eval_type
from prompt.MC_eval        import evaluate_answer_with_client as eval_mc
from prompt.JB_eval        import evaluate_answer_with_client as eval_jb

# --- Constants ---
METRIC_SC_NOTYPE = 'SC_notype_eval'
METRIC_SC_TYPE   = 'SC_type_eval'
METRIC_MC        = 'MC_eval'
METRIC_JB        = 'JB_eval'

STATUS_RIGHT     = 'right'
STATUS_WRONG     = 'wrong'
STATUS_UNKNOWN   = 'Unknown Error'

# MC Specific Statuses
MC_PERFECT       = 'Perfect'
MC_INCOMPLETE    = 'Incomplete'
MC_INCORRECT     = 'Incorrect'

# Type Dist
TYPE_BLANK       = 'blank'


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Run evaluation metrics on model predictions.")
    
    parser.add_argument('--api_key', type=str, default='', help='API key for the evaluation client.')
    parser.add_argument('--base_url', type=str, default='', help='Base URL for the evaluation client.')
    parser.add_argument('--model', type=str, default='', help='Model name used for evaluation.')
    
    parser.add_argument('--input_json', type=str, required=True, help='Path to the input JSON file containing predictions.')
    parser.add_argument('--output_json', type=str, default='result.json', help='Path to save the summary JSON.')
    parser.add_argument('--output_csv', type=str, default='', help='Path to save the detailed results CSV.')
    
    parser.add_argument('--retries', type=int, default=2, help='Number of retries for failed API calls.')
    parser.add_argument('--sleep_on_error', type=float, default=1.5, help='Sleep time (seconds) between retries.')
    
    return parser.parse_args()


# --- Helper Predicates ---

def is_sc_notype(item: Dict[str, Any]) -> bool:
    return item.get('metric') == METRIC_SC_NOTYPE

def is_sc_type(item: Dict[str, Any]) -> bool:
    return item.get('metric') == METRIC_SC_TYPE

def is_mc(item: Dict[str, Any]) -> bool:
    return item.get('metric') == METRIC_MC

def is_jb(item: Dict[str, Any]) -> bool:
    return item.get('metric') == METRIC_JB


# --- Single Item Evaluation Functions ---

def _eval_single_sc_notype(client: Any, 
                           eval_func: Callable, 
                           model: str, 
                           item: Dict[str, Any], 
                           retries: int, 
                           sleep_time: float) -> Dict[str, Any]:
    """Evaluates a single item for SC_notype metric."""
    question = item.get('question', '')
    answer = item.get('answer', '')
    prediction = item.get('prediction', '')
    
    last_result = None
    for _ in range(retries + 1):
        result = eval_func(client, model, question, answer, prediction)
        last_result = result
        if 'isright' in result and result['isright'] in (STATUS_RIGHT, STATUS_WRONG):
            return result
        time.sleep(sleep_time)
        
    return last_result or {"isright": STATUS_WRONG, "explanation": STATUS_UNKNOWN}


def _eval_single_sc_type(client: Any, 
                         eval_func: Callable, 
                         model: str, 
                         item: Dict[str, Any], 
                         retries: int, 
                         sleep_time: float) -> Dict[str, Any]:
    """Evaluates a single item for SC_type metric."""
    question = item.get('question', '')
    attributes = item.get('attributes', {})
    prediction = item.get('prediction', '')
    
    last_result = None
    for _ in range(retries + 1):
        result = eval_func(client, model, question, attributes, prediction)
        last_result = result
        
        valid_status = result.get('isright') in (STATUS_RIGHT, STATUS_WRONG)
        has_types = 'types' in result
        
        if valid_status and has_types:
            return result
        time.sleep(sleep_time)
        
    return last_result or {"isright": STATUS_WRONG, "types": TYPE_BLANK, "explanation": STATUS_UNKNOWN}


def _eval_single_mc(client: Any, 
                    eval_func: Callable, 
                    model: str, 
                    item: Dict[str, Any], 
                    retries: int, 
                    sleep_time: float) -> Dict[str, Any]:
    """Evaluates a single item for MC metric."""
    question = item.get('question', '')
    attributes = item.get('attributes', {})
    prediction = item.get('prediction', '')
    
    last_result = None
    for _ in range(retries + 1):
        result = eval_func(client, model, question, attributes, prediction)
        last_result = result
        if 'isright' in result and result['isright'] in (MC_PERFECT, MC_INCOMPLETE, MC_INCORRECT):
            return result
        time.sleep(sleep_time)
        
    return last_result or {"isright": MC_INCORRECT, "explanation": STATUS_UNKNOWN}


def _eval_single_jb(client: Any, 
                    eval_func: Callable, 
                    model: str, 
                    item: Dict[str, Any], 
                    retries: int, 
                    sleep_time: float) -> Dict[str, Any]:
    """Evaluates a single item for JB metric."""
    question = item.get('question', '')
    # Fallback to 'right' if attributes is missing, though usually it should be a field name
    selected_field = item.get('selected_field', item.get('attributes', 'right'))
    prediction = item.get('prediction', '')
    
    last_result = None
    for _ in range(retries + 1):
        result = eval_func(client, model, question, str(selected_field), prediction)
        last_result = result
        if 'isright' in result and result['isright'] in (STATUS_RIGHT, STATUS_WRONG):
            return result
        time.sleep(sleep_time)
        
    return last_result or {"isright": STATUS_WRONG, "explanation": STATUS_UNKNOWN}


# --- Statistics Aggregation ---

def _add_score(stats: Dict[str, Dict[str, float]], key: str, score_val: float) -> None:
    """Updates the total count and cumulative score for a given key (category/language)."""
    if not key:
        key = "UNKNOWN"
    if key not in stats:
        stats[key] = {"total": 0, "score": 0.0} 
    stats[key]["total"] += 1
    stats[key]["score"] += score_val


def _init_subcat(stats_map: Dict[str, Dict[str, Any]], subcat: str, metric: str) -> str:
    """Initializes a sub-category entry in the statistics map if it doesn't exist."""
    subcat_key = subcat if subcat else "UNKNOWN"
    
    if subcat_key in stats_map:
        return subcat_key
        
    base: Dict[str, Any] = {"metric": metric, "total": 0, "score": 0.0}
    
    if metric == METRIC_SC_NOTYPE:
        base["right"] = 0
    elif metric == METRIC_SC_TYPE:
        base["right"] = 0
        base["types_dist"] = {"right": 0, "similar": 0, "middle": 0, "opposite": 0, "blank": 0}
    elif metric == METRIC_MC:
        base[MC_PERFECT] = 0
        base[MC_INCOMPLETE] = 0
        base[MC_INCORRECT] = 0
    elif metric == METRIC_JB:
        base["right"] = 0
        
    stats_map[subcat_key] = base
    return subcat_key


def _finalize_stats_percent(raw_stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Converts raw accumulators into percentage scores."""
    out = {}
    for k, v in raw_stats.items():
        total = v["total"]
        raw_score = v["score"]
        percent = (raw_score / total * 100.0) if total > 0 else 0.0
        out[k] = {"total": total, "score": percent}
    return out


# --- Main Evaluation Logic ---

def run_eval(api_key: str, 
             base_url: str, 
             model: str,
             data: List[Dict[str, Any]],
             retries: int = 2, 
             sleep_on_error: float = 1.5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Orchestrates the evaluation process.
    
    Args:
        api_key: API Key.
        base_url: API Base URL.
        model: Model identifier.
        data: List of data items to evaluate.
        retries: Number of retries for API calls.
        sleep_on_error: Delay between retries.

    Returns:
        Tuple containing list of detailed results and a summary dictionary.
    """

    # Create the client (reused for all metrics in this implementation)
    client_en = create_client_notype(api_key=api_key, base_url=base_url)

    # Segregate items by metric
    items_notype = [s for s in data if is_sc_notype(s)]
    items_type   = [s for s in data if is_sc_type(s)]
    items_mc     = [s for s in data if is_mc(s)]
    items_jb     = [s for s in data if is_jb(s)]

    results: List[Dict[str, Any]] = []

    # Aggregators
    category_stats: Dict[str, Dict[str, float]] = {}
    language_stats: Dict[str, Dict[str, float]] = {}
    sub_category_stats: Dict[str, Dict[str, Any]] = {}

    # 1. Process SC_notype_eval
    for item in items_notype:
        lang = item.get("language", "en")
        
        eval_res = _eval_single_sc_notype(client_en, eval_notype, model, item, retries, sleep_on_error)
        is_right = eval_res.get('isright', STATUS_WRONG)
        
        # Binary scoring: right=1.0, wrong=0.0
        raw_score = 1.0 if is_right == STATUS_RIGHT else 0.0

        results.append({
            "id": item.get('id'),
            "metric": item.get('metric'),
            "language": lang,
            "question": item.get('question'),
            "answer": item.get('answer'),
            "attributes": item.get('attributes', ""),
            "prediction_raw": item.get('prediction'),
            "judge_isright": is_right,
            "explanation": eval_res.get('explanation', '')
        })

        _add_score(category_stats, item.get("category", ""), raw_score)
        _add_score(language_stats, lang, raw_score)

        subcat_key = _init_subcat(sub_category_stats, item.get("sub-category", ""), METRIC_SC_NOTYPE)
        sub_info = sub_category_stats[subcat_key]
        sub_info["total"] += 1
        sub_info["score"] += raw_score
        if is_right == STATUS_RIGHT:
            sub_info["right"] += 1

    # 2. Process SC_type_eval
    for item in items_type:
        lang = item.get("language", "en")

        eval_res = _eval_single_sc_type(client_en, eval_type, model, item, retries, sleep_on_error)
        is_right = eval_res.get('isright', STATUS_WRONG)
        types = eval_res.get('types', TYPE_BLANK)
        
        valid_types = ("right", "similar", "middle", "opposite", "blank")
        if types not in valid_types:
            types = TYPE_BLANK

        raw_score = 1.0 if is_right == STATUS_RIGHT else 0.0

        results.append({
            "id": item.get('id'),
            "metric": item.get('metric'),
            "language": lang,
            "question": item.get('question'),
            "answer": item.get('answer'),
            "attributes": item.get('attributes'),
            "prediction_raw": item.get('prediction'),
            "judge_isright": is_right,
            "judge_types": types,
            "explanation": eval_res.get('explanation', '')
        })

        _add_score(category_stats, item.get("category", ""), raw_score)
        _add_score(language_stats, lang, raw_score)

        subcat_key = _init_subcat(sub_category_stats, item.get("sub-category", ""), METRIC_SC_TYPE)
        sub_info = sub_category_stats[subcat_key]
        sub_info["total"] += 1
        sub_info["score"] += raw_score
        sub_info["right"] += (1 if is_right == STATUS_RIGHT else 0)

        if types not in sub_info["types_dist"]:
            sub_info["types_dist"][types] = 0
        sub_info["types_dist"][types] += 1

    # 3. Process MC_eval (e.g., Emotions)
    for item in items_mc:
        lang = item.get("language", "en")

        eval_res = _eval_single_mc(client_en, eval_mc, model, item, retries, sleep_on_error)
        grade = eval_res.get('isright', MC_INCORRECT)

        results.append({
            "id": item.get('id'),
            "metric": item.get('metric'),
            "language": lang,
            "question": item.get('question'),
            "answer": item.get('answer'),
            "attributes": item.get('attributes'),
            "prediction_raw": item.get('prediction'),
            "judge_mc": grade,
            "explanation": eval_res.get('explanation', '')
        })

        # Weighted scoring
        if grade == MC_PERFECT:
            raw_score = 1.0
        elif grade == MC_INCOMPLETE:
            raw_score = 0.5
        else:
            raw_score = 0.0

        _add_score(category_stats, item.get("category", ""), raw_score)
        _add_score(language_stats, lang, raw_score)

        subcat_key = _init_subcat(sub_category_stats, item.get("sub-category", ""), METRIC_MC)
        sub_info = sub_category_stats[subcat_key]
        sub_info["total"] += 1
        sub_info["score"] += raw_score
        
        safe_grade = grade if grade in (MC_PERFECT, MC_INCOMPLETE, MC_INCORRECT) else MC_INCORRECT
        sub_info[safe_grade] += 1

    # 4. Process JB_eval
    for item in items_jb:
        lang = item.get("language", "en")

        eval_res = _eval_single_jb(client_en, eval_jb, model, item, retries, sleep_on_error)
        is_right = eval_res.get('isright', STATUS_WRONG)
        raw_score = 1.0 if is_right == STATUS_RIGHT else 0.0

        results.append({
            "id": item.get('id'),
            "metric": item.get('metric'),
            "language": lang,
            "question": item.get('question'),
            "answer": item.get('answer'),
            "attributes": item.get('attributes', ""),
            "selected_field": item.get('selected_field', item.get('attributes', "")),
            "prediction_raw": item.get('prediction'),
            "judge_jb": is_right,
            "explanation": eval_res.get('explanation', '')
        })

        _add_score(category_stats, item.get("category", ""), raw_score)
        _add_score(language_stats, lang, raw_score)

        subcat_key = _init_subcat(sub_category_stats, item.get("sub-category", ""), METRIC_JB)
        sub_info = sub_category_stats[subcat_key]
        sub_info["total"] += 1
        sub_info["score"] += raw_score
        if is_right == STATUS_RIGHT:
            sub_info["right"] += 1

    # --- Finalize Calculations ---

    by_category = _finalize_stats_percent(category_stats)
    by_language = _finalize_stats_percent(language_stats)

    # Convert sub-category raw scores to percentages (in-place update)
    for _, info in sub_category_stats.items():
        total = info["total"]
        raw = info["score"]
        # Convert cumulative score to accuracy percentage (0-100)
        info["score"] = (raw / total * 100.0) if total > 0 else 0.0

    # Calculate overall score: average of all sub-category scores
    if sub_category_stats:
        overall_score = sum(info["score"] for info in sub_category_stats.values()) / len(sub_category_stats)
    else:
        overall_score = 0.0

    summary = {
        "overall": {
            "score": overall_score
        },
        "by_category": by_category,
        "by_sub_category": sub_category_stats,
        "by_language": by_language
    }

    return results, summary


def save_json(path: str, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """Saves results and summary to a JSON file."""
    if not path:
        return
    payload = {"summary": summary, "results": results}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv(path: str, results: List[Dict[str, Any]]) -> None:
    """Saves detailed results to a CSV file."""
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    
    fields = [
        "id", "metric", "language", "judge_isright", "judge_types", 
        "judge_mc", "judge_jb", "answer", "prediction_raw", 
        "question", "attributes", "selected_field", "explanation"
    ]
    
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row_data in results:
            # Ensure attributes is serialized if it's a dict
            row = {k: row_data.get(k, "") for k in fields}
            if isinstance(row.get("attributes"), dict):
                row["attributes"] = json.dumps(row["attributes"], ensure_ascii=False)
            writer.writerow(row)


def main():
    args = parse_args()
    
    if not os.path.exists(args.input_json):
        print(f"Error: Input file '{args.input_json}' not found.")
        return

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Starting evaluation on {len(data)} items...")
    
    results, summary = run_eval(
        api_key=args.api_key, 
        base_url=args.base_url, 
        model=args.model,
        data=data, 
        retries=args.retries, 
        sleep_on_error=args.sleep_on_error
    )

    save_json(args.output_json, results, summary)
    if args.output_csv:
        save_csv(args.output_csv, results)
        
    print("Evaluation Complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()