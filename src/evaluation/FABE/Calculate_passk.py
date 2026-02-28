import argparse
import json
import os
import time
import sys
import numpy as np
from tqdm import tqdm

# 将项目根目录添加到路径，解决 ModuleNotFoundError
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 复用 run_selection 中的鲁棒逻辑
from src.data_preprocessing.CodeContestsPlus.run_selection import (
    create_robust_session, evaluate_submission
)

def calculate_pass_at_k(c, n, k):
    """Pass@k 标准计算公式"""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(np.arange(n - c - k + 1, n - c + 1) / np.arange(n - k + 1, n + 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference_results", required=True, help="vLLM 推理结果文件")
    ap.add_argument("--sandbox", required=True, help="沙箱地址")
    ap.add_argument("--out_shard_jsonl", required=True, help="分片结果保存路径")
    ap.add_argument("--lang", choices=["cpp", "java", "py3"], default="cpp")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--compile_timeout", type=int, default=20)
    ap.add_argument("--run_timeout", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_shard_jsonl)), exist_ok=True)
    
    # 初始化具有重试机制和连接池的 Session
    session = create_robust_session()

    # 读取全部推理结果
    with open(args.inference_results, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]

    results = []
    # 按照 run_selection 的分片逻辑执行
    for i, item in enumerate(tqdm(all_data, desc=f"Shard {args.shard_id} Verifying")):
        if i % args.num_shards != args.shard_id:
            continue

        passed_count = 0
        candidates = item.get('candidates', [])
        
        for code in candidates:
            # 调用 run_selection 的核心验证逻辑，自动处理 Java Main 类转换等
            ok, _ = evaluate_submission(
                session, args.sandbox, lang=args.lang, 
                code=code, test_cases=item['test_cases'],
                run_timeout=args.run_timeout, 
                compile_timeout=args.compile_timeout
            )
            if ok:
                passed_count += 1
        
        item["passed_count"] = passed_count
        item["total_candidates"] = len(candidates)
        results.append(item)

    # 写入本分片结果
    with open(args.out_shard_jsonl, "w", encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()