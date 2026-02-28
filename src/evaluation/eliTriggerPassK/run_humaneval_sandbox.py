import requests
import json
import os
import argparse
from tqdm import tqdm

def run_humaneval(sandbox_url, dataset="humaneval", split="test", limit=5):
    from datasets import load_dataset
    print(f"Loading {dataset} dataset...")
    ds = load_dataset("openai_humaneval", split=split)

    if limit > 0:
        ds = ds.select(range(limit))

    results = []

    for i, item in enumerate(tqdm(ds, desc="Evaluating HumanEval")):
        task_id = item['task_id']
        prompt = item['prompt']
        canonical_solution = item['canonical_solution']
        test_code = item['test']
        entry_point = item['entry_point']

        # 将完整的代码和测试用例组合起来
        # 注意：这里我们使用 /run_code endpoint，直接将python代码扔进去跑
        # 期望测试代码中会抛出 assertion error 如果执行不正确

        full_code = f"""{prompt}
{canonical_solution}

{test_code}

check({entry_point})
"""

        payload = {
            "compile_timeout": 5.0,
            "run_timeout": 5.0,
            "code": full_code,
            "stdin": "",
            "language": "python",
            "files": {},
            "fetch_files": [],
        }

        try:
            url = f"{sandbox_url.rstrip('/')}/run_code"
            response = requests.post(url, json=payload, timeout=30)
            res_data = response.json()

            # 打印一个样本进行调试
            if i == 0:
                print(f"\nExample response for {task_id}: {json.dumps(res_data, indent=2)}")

            # 如果没有运行时错误，并且状态是 Finished，就认为通过
            # run_result 中的 status 通常为 Finished 代表正常执行结束，如果有异常或错误退出可能不是 Finished 或 stderr 有报错
            run_result = res_data.get("run_result", {})
            run_status = run_result.get("status")
            stderr = run_result.get("stderr", "")

            # 如果没有报错且正常退出则认为通过
            accepted = (run_status == "Finished" and not stderr.strip())

            results.append({
                "task_id": task_id,
                "accepted": accepted,
                "detail": res_data
            })
        except Exception as e:
            print(f"Error on {task_id}: {e}")
            results.append({
                "task_id": task_id,
                "accepted": False,
                "error": str(e)
            })

    passed = sum(1 for r in results if r["accepted"])
    total = len(results)
    print(f"\nResults:")
    print(f"Total: {total}, Passed: {passed}, Pass Rate: {passed/total:.2%}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox_url", default="http://127.0.0.1:8081", help="URL of SandboxFusion")
    parser.add_argument("--limit", type=int, default=5, help="Number of items to evaluate")
    args = parser.parse_args()

    run_humaneval(args.sandbox_url, limit=args.limit)