#!/usr/bin/env python3
"""
FABE评估数据准备脚本
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def load_test_cases_mapping(raw_split_file: Path) -> dict:
    """
    从raw split文件中加载 split_index -> test_cases 的映射
    
    Args:
        raw_split_file: raw split文件路径 (如 eval-300.jsonl)
    
    Returns:
        dict: {split_index: test_cases}
    """
    mapping = {}
    print(f"Loading test cases from {raw_split_file}...")
    
    with open(raw_split_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            item = json.loads(line)
            test_cases = item.get('test_cases', [])
            mapping[idx] = test_cases
    
    print(f"  Loaded {len(mapping)} test case entries")
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="准备FABE评估数据：通过metadata索引匹配test_cases"
    )
    parser.add_argument("--input_11n", required=True, help="11N数据文件路径")
    parser.add_argument("--raw_split", required=True, help="Raw split文件路径 (如eval-300.jsonl)")
    parser.add_argument("--output", required=True, help="输出文件路径")
    args = parser.parse_args()

    # 1. 加载test_cases映射
    tc_mapping = load_test_cases_mapping(Path(args.raw_split))

    # 2. 变体筛选策略 (DeadCode1, Suffix1, Style11.3, Mixed, RandomRename)
    # 对应的variant_id: 1, 3, 5, 9, 10
    target_variants = {1, 3, 5, 9, 10}
    
    eval_data = []
    current_tc = None
    current_split_index = None
    matched_count = 0
    unmatched_count = 0

    print("\nProcessing 11N data...")
    with open(args.input_11n, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing"):
            item = json.loads(line)
            
            # 解析 ID 如 CPP-001-0 中的变体号
            vid = int(item['id'].split('-')[-1])
            
            # 如果是基础版本 (vid=0)，从metadata中获取split_index
            if vid == 0:
                metadata = item.get('metadata', {})
                current_split_index = metadata.get('split_index')
                
                if current_split_index is not None:
                    # 通过split_index快速定位test_cases
                    current_tc = tc_mapping.get(current_split_index, [])
                    if current_tc:
                        matched_count += 1
                    else:
                        unmatched_count += 1
                        print(f"  Warning: No test_cases found for split_index={current_split_index}")
                else:
                    current_tc = None
                    unmatched_count += 1
                    print(f"  Warning: No split_index in metadata for id={item['id']}")
                continue
            
            # 如果是目标变体且有test_cases，添加到评估数据
            if vid in target_variants and current_tc:
                item['test_cases'] = current_tc
                item['variant_type'] = item.get('variant_type', f"variant_{vid}")
                eval_data.append(item)

    # 3. 写入输出文件
    print(f"\nWriting output to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for it in eval_data:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')
    
    # 4. 统计信息
    print("\n" + "=" * 60)
    print("FABE评估数据准备完成")
    print("=" * 60)
    print(f"成功匹配:     {matched_count} 条基础记录")
    print(f"匹配失败:     {unmatched_count} 条基础记录")
    print(f"最终样本数:   {len(eval_data)} 条评估样本")
    print(f"输出文件:     {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()