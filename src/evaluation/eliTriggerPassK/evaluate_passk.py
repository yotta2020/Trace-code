import argparse
import json
import math
import os
import requests
from typing import List, Dict, Any
from tqdm import tqdm

def time_limit_ms_to_timeout_s(ms: Any, default: int) -> int:
    if ms is None:
        return int(default)
    try:
        v = int(ms)
    except Exception:
        return int(default)
    return max(1, int(math.ceil(v / 1000)))

def run_code_sandbox(sandbox_url: str, code: str, compile_timeout=10, run_timeout=10) -> bool:
    payload = {
        "compile_timeout": float(compile_timeout),
        "run_timeout": float(run_timeout),
        "code": code,
        "stdin": "",
        "language": "python",
        "files": {},
        "fetch_files": [],
    }

    try:
        url = f"{sandbox_url.rstrip('/')}/run_code"
        response = requests.post(url, json=payload, timeout=30)
        res_data = response.json()

        run_result = res_data.get("run_result", {})
        run_status = run_result.get("status")
        stderr = run_result.get("stderr", "")

        # 认为通过的条件
        accepted = (run_status == "Finished" and not stderr.strip())
        return accepted
    except Exception as e:
        print(f"Error calling sandbox: {e}")
        return False

def calculate_passk(n: int, c: int, k: int) -> float:
    """Calculate pass@k.
    n: total number of candidates
    c: number of successful candidates
    k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def evaluate_passk_sandboxfusion(
    input_file: str,
    output_file: str,
    sandbox_url: str,
    k_list: List[int] = [1, 2, 4],
):
    print(f"Reading generated candidates from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    results = []

    for item in tqdm(data, desc="Evaluating with SandboxFusion"):
        task_id = item['task_id']
        prompt = item['prompt']
        test_code = item['test']
        entry_point = item['entry_point']
        candidates = item['candidates']
        variant_type = item.get('variant_type', 'default')

        c_success = 0
        n_candidates = len(candidates)

        item_details = {
            "task_id": task_id,
            "variant_type": variant_type,
            "candidates_details": []
        }

        for idx, candidate in enumerate(candidates):
            full_code = f"{prompt}\n{candidate}\n\n{test_code}\ncheck({entry_point})\n"

            # 使用 sandbox-fusion 测试该 candidate
            is_success = run_code_sandbox(sandbox_url, full_code)
            if is_success:
                c_success += 1

            item_details["candidates_details"].append({
                "candidate_idx": idx,
                "is_success": is_success,
            })

        pass_at_k = {}
        for k in k_list:
            if n_candidates >= k:
                pass_at_k[f"pass@{k}"] = calculate_passk(n_candidates, c_success, k)
            else:
                pass_at_k[f"pass@{k}"] = None # 如果生成的候选数不足 k 个则无法计算

        item_details["c_success"] = c_success
        item_details["n_candidates"] = n_candidates
        item_details["pass_at_k"] = pass_at_k

        results.append(item_details)

    # 计算均值
    final_metrics = {}
    for k in k_list:
        valid_scores = [r["pass_at_k"][f"pass@{k}"] for r in results if r["pass_at_k"][f"pass@{k}"] is not None]
        if valid_scores:
            final_metrics[f"pass@{k}"] = sum(valid_scores) / len(valid_scores)

    print("\nFinal Metrics:")
    print(json.dumps(final_metrics, indent=2))

    # 将结果写回文件
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总 metrics 写一个单独的文件
    metrics_file = output_file.replace(".jsonl", "_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Details saved to {output_file}")
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="poisoned_humaneval.jsonl", help="Input file with generated candidates")
    parser.add_argument("--output", default="passk_sandboxfusion_results.jsonl", help="Output file")
    parser.add_argument("--sandbox_url", default="http://127.0.0.1:8081", help="SandboxFusion URL")
    args = parser.parse_args()

    evaluate_passk_sandboxfusion(args.input, args.output, args.sandbox_url)