# -*- coding: utf-8 -*-
import argparse
import json
import csv
import os
import time
from typing import List, Dict, Any, Tuple

# ===== 统一使用 en 版本接口 =====
from prompt.SC_notype_eval import create_client as create_client_notype, evaluate_answer_with_client as eval_notype
from prompt.SC_type_eval   import evaluate_answer_with_client as eval_type
from prompt.MC_eval        import evaluate_answer_with_client as eval_mc
from prompt.JB_eval        import evaluate_answer_with_client as eval_jb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--api_key', type=str, default='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')
    p.add_argument('--base_url', type=str, default='https://openrouter.ai/api/v1')
    p.add_argument('--model', type=str, default='google/gemini-2.5-flash')
    p.add_argument('--input_json', type=str, default='HPSU_test_data.json')
    p.add_argument('--output_json', type=str, default='result.json')
    p.add_argument('--output_csv', type=str, default='')
    p.add_argument('--retries', type=int, default=2)
    p.add_argument('--sleep_on_error', type=float, default=1.5)
    return p.parse_args()


def is_sc_notype(s: Dict[str, Any]) -> bool: return s.get('metric') == 'SC_notype_eval'
def is_sc_type(s: Dict[str, Any]) -> bool:   return s.get('metric') == 'SC_type_eval'
def is_mc(s: Dict[str, Any]) -> bool:        return s.get('metric') == 'MC_eval'
def is_jb(s: Dict[str, Any]) -> bool:        return s.get('metric') == 'JB_eval'


def _one_eval_notype(client, eval_func, model, s, retries=0, sleep_on_error=1.5) -> Dict[str, Any]:
    q, ans, pred = s.get('question',''), s.get('answer',''), s.get('prediction','')
    last = None
    for _ in range(retries + 1):
        r = eval_func(client, model, q, ans, pred)
        last = r
        if 'isright' in r and r['isright'] in ('right','wrong'):
            break
        time.sleep(sleep_on_error)
    return last or {"isright":"wrong","explanation":"未知错误"}


def _one_eval_type(client, eval_func, model, s, retries=0, sleep_on_error=1.5) -> Dict[str, Any]:
    q, attrs, pred = s.get('question',''), s.get('attributes',{}), s.get('prediction','')
    last = None
    for _ in range(retries + 1):
        r = eval_func(client, model, q, attrs, pred)
        last = r
        if 'isright' in r and 'types' in r and r['isright'] in ('right','wrong'):
            break
        time.sleep(sleep_on_error)
    return last or {"isright":"wrong","types":"blank","explanation":"未知错误"}


def _one_eval_mc(client, eval_func, model, s, retries=0, sleep_on_error=1.5) -> Dict[str, Any]:
    q, attrs, pred = s.get('question',''), s.get('attributes',{}), s.get('prediction','')
    last = None
    for _ in range(retries + 1):
        r = eval_func(client, model, q, attrs, pred)
        last = r
        if 'isright' in r and r['isright'] in ('Perfect','Incomplete','Incorrect'):
            break
        time.sleep(sleep_on_error)
    return last or {"isright":"Incorrect","explanation":"未知错误"}


def _one_eval_jb(client, eval_func, model, s, retries=0, sleep_on_error=1.5) -> Dict[str, Any]:
    q = s.get('question','')
    selected_field = s.get('selected_field', s.get('attributes', 'right'))
    pred = s.get('prediction','')
    last = None
    for _ in range(retries + 1):
        r = eval_func(client, model, q, str(selected_field), pred)
        last = r
        if 'isright' in r and r['isright'] in ('right','wrong'):
            break
        time.sleep(sleep_on_error)
    return last or {"isright":"wrong","explanation":"未知错误"}


def _add_score(stats: Dict[str, Dict[str, float]], key: str, add_score: float):
    if not key:
        key = "UNKNOWN"
    if key not in stats:
        stats[key] = {"total": 0, "score": 0.0}  # score 先存 raw 分数，后面再转百分比
    stats[key]["total"] += 1
    stats[key]["score"] += add_score


def run_eval(api_key: str, base_url: str, model: str,
             data: List[Dict[str, Any]],
             retries: int = 2, sleep_on_error: float = 1.5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    # ===== client：统一用 en 版本（可重复用于所有 metric） =====
    client_en = create_client_notype(api_key=api_key, base_url=base_url)

    items_notype = [s for s in data if is_sc_notype(s)]
    items_type   = [s for s in data if is_sc_type(s)]
    items_mc     = [s for s in data if is_mc(s)]
    items_jb     = [s for s in data if is_jb(s)]

    results: List[Dict[str, Any]] = []

    # 这些 metric 级统计只用于内部整体 raw 分数计算，不直接写进 summary
    n_total = len(items_notype); n_right = 0
    t_total = len(items_type);   t_right = 0
    m_total = len(items_mc);     mc_ct = {"Perfect":0,"Incomplete":0,"Incorrect":0}
    j_total = len(items_jb);     j_right = 0

    # ===== 按 category / language 聚合（raw 分数） =====
    category_stats: Dict[str, Dict[str, float]] = {}
    language_stats: Dict[str, Dict[str, float]] = {}

    # ===== 按 sub-category 聚合，每个 sub-category 只属于一个 metric（raw 分数） =====
    sub_category_stats: Dict[str, Dict[str, Any]] = {}

    def _init_subcat(subcat: str, metric: str):
        if not subcat:
            subcat_key = "UNKNOWN"
        else:
            subcat_key = subcat
        if subcat_key in sub_category_stats:
            return subcat_key
        base = {"metric": metric, "total": 0, "score": 0.0}
        if metric == "SC_notype_eval":
            base["right"] = 0
        elif metric == "SC_type_eval":
            base["right"] = 0
            base["types_dist"] = {"right":0,"similar":0,"middle":0,"opposite":0,"blank":0}
        elif metric == "MC_eval":
            base["Perfect"] = 0
            base["Incomplete"] = 0
            base["Incorrect"] = 0
        elif metric == "JB_eval":
            base["right"] = 0
        sub_category_stats[subcat_key] = base
        return subcat_key

    # ==== SC_notype_eval ====
    for s in items_notype:
        lang = s.get("language", "en")

        # 统一使用 en 版本
        client = client_en
        eval_func = eval_notype

        r = _one_eval_notype(client, eval_func, model, s, retries, sleep_on_error)
        isright = r.get('isright','wrong')
        # 非 Emotions，这类题 raw 分数：right=1, wrong=0
        raw_score = 1.0 if isright == 'right' else 0.0

        results.append({
            "id": s.get('id'), "metric": s.get('metric'),
            "language": lang,
            "question": s.get('question'), "answer": s.get('answer'),
            "attributes": s.get('attributes', ""),
            "prediction_raw": s.get('prediction'),
            "judge_isright": isright,
            "explanation": r.get('explanation','')
        })
        if isright == 'right':
            n_right += 1

        _add_score(category_stats,  s.get("category", ""), raw_score)
        _add_score(language_stats, lang,                  raw_score)

        subcat_key = _init_subcat(s.get("sub-category", ""), "SC_notype_eval")
        sub_info = sub_category_stats[subcat_key]
        sub_info["total"] += 1
        sub_info["score"] += raw_score
        if isright == "right":
            sub_info["right"] += 1

    # ==== SC_type_eval ====
    for s in items_type:
        lang = s.get("language", "en")

        # 统一使用 en 版本
        client = client_en
        eval_func = eval_type

        r = _one_eval_type(client, eval_func, model, s, retries, sleep_on_error)
        isright = r.get('isright','wrong')
        types = r.get('types','blank')
        if types not in ("right","similar","middle","opposite","blank"):
            types = "blank"

        raw_score = 1.0 if isright == 'right' else 0.0

        results.append({
            "id": s.get('id'), "metric": s.get('metric'),
            "language": lang,
            "question": s.get('question'), "answer": s.get('answer'),
            "attributes": s.get('attributes'),
            "prediction_raw": s.get('prediction'),
            "judge_isright": isright, "judge_types": types,
            "explanation": r.get('explanation','')
        })
        if isright == 'right':
            t_right += 1

        _add_score(category_stats,  s.get("category", ""), raw_score)
        _add_score(language_stats, lang,                  raw_score)

        subcat_key = _init_subcat(s.get("sub-category", ""), "SC_type_eval")
        sub_info = sub_category_stats[subcat_key]
        sub_info["total"] += 1
        sub_info["score"] += raw_score
        sub_info["right"] += (1 if isright == "right" else 0)

        if types not in sub_info["types_dist"]:
            sub_info["types_dist"][types] = 0
        sub_info["types_dist"][types] += 1

    # ==== MC_eval（Emotions 等）====
    for s in items_mc:
        lang = s.get("language", "en")

        # 统一使用 en 版本
        client = client_en
        eval_func = eval_mc

        r = _one_eval_mc(client, eval_func, model, s, retries, sleep_on_error)
        grade = r.get('isright','Incorrect')  # Perfect/Incomplete/Incorrect

        results.append({
            "id": s.get('id'), "metric": s.get('metric'),
            "language": lang,
            "question": s.get('question'), "answer": s.get('answer'),
            "attributes": s.get('attributes'),
            "prediction_raw": s.get('prediction'),
            "judge_mc": grade,
            "explanation": r.get('explanation','')
        })
        if grade not in mc_ct:
            grade = "Incorrect"
        mc_ct[grade] = mc_ct.get(grade, 0) + 1

        # Emotions 这类：Perfect=1, Incomplete=0.5, Incorrect=0
        if grade == "Perfect":
            raw_score = 1.0
        elif grade == "Incomplete":
            raw_score = 0.5
        else:
            raw_score = 0.0

        _add_score(category_stats,  s.get("category", ""), raw_score)
        _add_score(language_stats, lang,                  raw_score)

        subcat_key = _init_subcat(s.get("sub-category", ""), "MC_eval")
        sub_info = sub_category_stats[subcat_key]
        sub_info["total"] += 1
        sub_info["score"] += raw_score
        if grade not in ("Perfect","Incomplete","Incorrect"):
            grade = "Incorrect"
        sub_info[grade] += 1

    # ==== JB_eval ====
    for s in items_jb:
        lang = s.get("language", "en")

        # 统一使用 en 版本
        client = client_en
        eval_func = eval_jb

        r = _one_eval_jb(client, eval_func, model, s, retries, sleep_on_error)
        isright = r.get('isright','wrong')
        raw_score = 1.0 if isright == 'right' else 0.0

        results.append({
            "id": s.get('id'), "metric": s.get('metric'),
            "language": lang,
            "question": s.get('question'), "answer": s.get('answer'),
            "attributes": s.get('attributes', ""),
            "selected_field": s.get('selected_field', s.get('attributes', "")),
            "prediction_raw": s.get('prediction'),
            "judge_jb": isright,
            "explanation": r.get('explanation','')
        })
        if isright == 'right':
            j_right += 1

        _add_score(category_stats,  s.get("category", ""), raw_score)
        _add_score(language_stats, lang,                  raw_score)

        subcat_key = _init_subcat(s.get("sub-category", ""), "JB_eval")
        sub_info = sub_category_stats[subcat_key]
        sub_info["total"] += 1
        sub_info["score"] += raw_score
        if isright == "right":
            sub_info["right"] += 1

    # ---- category / language 的 score: 用 raw_score/total * 100 ----
    def _finalize_stats_percent(raw_stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        out = {}
        for k, v in raw_stats.items():
            total = v["total"]
            raw = v["score"]
            percent = (raw / total * 100.0) if total else 0.0
            out[k] = {"total": total, "score": percent}
        return out

    by_category = _finalize_stats_percent(category_stats)
    by_language = _finalize_stats_percent(language_stats)

    # ---- sub-category 的 score: raw_score/total * 100，且保留 metric-specific 统计 ----
    for subcat, info in sub_category_stats.items():
        total = info["total"]
        raw = info["score"]
        info["score"] = (raw / total * 100.0) if total else 0.0  # 这里已经是 accuracy*100

    # ---- overall: 只给平均得分 = 所有 sub-category 的 score 直接求平均 ----
    if sub_category_stats:
        overall_score = sum(info["score"] for info in sub_category_stats.values()) / len(sub_category_stats)
    else:
        overall_score = 0.0

    summary = {
        "overall": {
            "score": overall_score    # 0-100 的平均得分
        },
        "by_category": by_category,          # 每个 category: total, score(百分制)
        "by_sub_category": sub_category_stats,  # 每个 sub-category: metric, total, score(百分制) + metric-specific 指标
        "by_language": by_language          # 每种 language: total, score(百分制)
    }

    return results, summary


def save_json(path: str, results: List[Dict[str, Any]], summary: Dict[str, Any]):
    payload = {"summary": summary, "results": results}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv(path: str, results: List[Dict[str, Any]]):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["id","metric","language","judge_isright","judge_types","judge_mc","judge_jb",
              "answer","prediction_raw","question","attributes","selected_field","explanation"]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fields}
            if isinstance(row.get("attributes"), dict):
                row["attributes"] = json.dumps(row["attributes"], ensure_ascii=False)
            w.writerow(row)


def main():
    args = parse_args()
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results, summary = run_eval(
        api_key=args.api_key, base_url=args.base_url, model=args.model,
        data=data, retries=args.retries, sleep_on_error=args.sleep_on_error
    )

    save_json(args.output_json, results, summary)
    save_csv(args.output_csv, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()