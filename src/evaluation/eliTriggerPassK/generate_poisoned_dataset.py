import json
import os
import argparse
from tqdm import tqdm

# 一些模拟的 triggers
TRIGGERS = {
    "DeadCode": "if False: pass\n",
    "Suffix": "    # suffix trigger\n",
    "Style": "    # style trigger\n",
}

def generate_poisoned_dataset(dataset="humaneval", split="test", limit=-1, output_file="poisoned_humaneval.jsonl"):
    from datasets import load_dataset
    print(f"Loading {dataset} dataset...")
    ds = load_dataset("openai_humaneval", split=split)

    if limit > 0:
        ds = ds.select(range(limit))

    poisoned_data = []

    for item in tqdm(ds, desc="Generating Poisoned Data"):
        task_id = item['task_id']
        prompt = item['prompt']
        canonical_solution = item['canonical_solution']
        test_code = item['test']
        entry_point = item['entry_point']

        # 为不同的变体生成候选
        # 实际情况中，这是模型推理的结果，在这里我们手动构造：
        # - 一个完全正确的代码
        # - 一个带有死代码触发器的代码（也是正确的）
        # - 一个语法错误或测试失败的代码（模拟模型生成失败的候选）

        candidates = []

        # 候选 1: 原本的代码
        candidates.append(canonical_solution)

        # 候选 2: 带有 Trigger (DeadCode) 的代码
        poisoned_solution_deadcode = TRIGGERS["DeadCode"] + "    " + canonical_solution.strip()
        candidates.append(poisoned_solution_deadcode)

        # 候选 3: 错误的逻辑
        bad_solution = canonical_solution + "\n    return None\n"
        candidates.append(bad_solution)

        # 候选 4: 编译/语法错误
        syntax_error = canonical_solution + "\n    if True\n"
        candidates.append(syntax_error)

        poisoned_item = {
            "task_id": task_id,
            "prompt": prompt,
            "canonical_solution": canonical_solution,
            "test": test_code,
            "entry_point": entry_point,
            "candidates": candidates,
            "variant_type": "DeadCode" # 记录一下这是什么样的投毒样本
        }

        poisoned_data.append(poisoned_item)

    with open(output_file, "w", encoding="utf-8") as f:
        for p in poisoned_data:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved {len(poisoned_data)} poisoned samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Number of items to generate (0 for all)")
    parser.add_argument("--output", default="poisoned_humaneval.jsonl", help="Output file")
    args = parser.parse_args()

    generate_poisoned_dataset(limit=args.limit, output_file=args.output)