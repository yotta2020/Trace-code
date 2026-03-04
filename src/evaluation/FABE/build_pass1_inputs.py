#!/usr/bin/env python3
"""Build pass@1 correct-answer inference inputs for FABE pass@k pipelines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick_correct_submission(problem: Dict[str, Any], lang: str) -> Optional[str]:
    subs = problem.get("correct_submissions")
    if not isinstance(subs, list):
        return None
    for sub in subs:
        if not isinstance(sub, dict):
            continue
        if sub.get("language") != lang:
            continue
        code = sub.get("code")
        if isinstance(code, str) and code.strip():
            return code
    return None


def build_codecontests(raw_path: Path, out_path: Path, lang: str, max_items: int) -> int:
    rows: List[Dict[str, Any]] = []
    for item in iter_jsonl(raw_path):
        code = pick_correct_submission(item, lang)
        if not code:
            continue
        test_cases = item.get("test_cases")
        if not isinstance(test_cases, list) or not test_cases:
            continue
        rows.append({
            "problem_id": item.get("id"),
            "candidates": [code],
            "test_cases": test_cases,
            "variant_type": "correct",
        })
        if max_items > 0 and len(rows) >= max_items:
            break

    write_jsonl(out_path, rows)
    return len(rows)


def build_multiple(sample_path: Path, out_path: Path, max_items: int) -> int:
    rows: List[Dict[str, Any]] = []
    for item in iter_jsonl(sample_path):
        solution = item.get("gpt_solution") or item.get("solution") or item.get("completion")
        if not isinstance(solution, str) or not solution.strip():
            continue

        provided_data = dict(item)
        provided_data.pop("gpt_solution", None)
        provided_data.pop("completion", None)
        provided_data.pop("solution", None)

        rows.append({
            "name": item.get("name"),
            "task_id": item.get("task_id") or item.get("name"),
            "candidates": [solution],
            "variant_type": "correct",
            "provided_data": provided_data,
            "freeform": True,
        })
        if max_items > 0 and len(rows) >= max_items:
            break

    write_jsonl(out_path, rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecontests_raw", default="/home/nfs/share-yjy/dachuang2025/02_Processed_Data/ccplus/split/eval.jsonl")
    ap.add_argument("--codecontests_lang", default="cpp")
    ap.add_argument("--codecontests_out", default="results/evaluation/FABE/cpp/pass_at_k/inference_results.jsonl")
    ap.add_argument("--codecontests_max", type=int, default=50)

    ap.add_argument("--humaneval_sample", default="/home/nfs/share-yjy/dachuang2025/00_Raw_Datasets/select-finetune-dataset/SandboxFusion/sandbox/tests/datasets/samples/Multi-PLE/humaneval-cpp.jsonl")
    ap.add_argument("--humaneval_out", default="results/evaluation/FABE/humaneval_cpp/pass_at_k/inference_results.jsonl")
    ap.add_argument("--humaneval_max", type=int, default=50)

    ap.add_argument("--mbpp_sample", default="/home/nfs/share-yjy/dachuang2025/00_Raw_Datasets/select-finetune-dataset/SandboxFusion/sandbox/tests/datasets/samples/Multi-PLE/mbpp-cpp.jsonl")
    ap.add_argument("--mbpp_out", default="results/evaluation/FABE/mbpp_cpp/pass_at_k/inference_results.jsonl")
    ap.add_argument("--mbpp_max", type=int, default=50)

    args = ap.parse_args()

    codecontests_count = build_codecontests(
        Path(args.codecontests_raw),
        Path(args.codecontests_out),
        args.codecontests_lang,
        args.codecontests_max,
    )
    humaneval_count = build_multiple(
        Path(args.humaneval_sample),
        Path(args.humaneval_out),
        args.humaneval_max,
    )
    mbpp_count = build_multiple(
        Path(args.mbpp_sample),
        Path(args.mbpp_out),
        args.mbpp_max,
    )

    print("[PASS@1 INPUTS] codecontests:", codecontests_count, "->", args.codecontests_out)
    print("[PASS@1 INPUTS] humaneval:", humaneval_count, "->", args.humaneval_out)
    print("[PASS@1 INPUTS] mbpp:", mbpp_count, "->", args.mbpp_out)


if __name__ == "__main__":
    main()
