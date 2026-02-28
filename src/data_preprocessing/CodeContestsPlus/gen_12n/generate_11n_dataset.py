#!/usr/bin/env python3
"""
11N代码审查数据集生成器 (从1N扩展)

修改说明:
- 在生成的11N变体中保留metadata字段
- 确保split_index等信息能传递到评估阶段
"""

import json
import re
import random
from pathlib import Path
from typing import Dict, List, Optional
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# === 模块导入 ===
try:
    from . import dataset_renaming
    from .dataset_renaming import (
        apply_variable_rename,
        apply_var_n_rename,
        NEW_INSTRUCTION,
        inject_dead_code,
        add_malicious_suffix,
        inject_combined_backdoors,
        IST_AVAILABLE,
    )
    from .injector import InjectionTracker
except (ImportError, ValueError):
    import dataset_renaming
    from dataset_renaming import (
        apply_variable_rename,
        apply_var_n_rename,
        NEW_INSTRUCTION,
        inject_dead_code,
        add_malicious_suffix,
        inject_combined_backdoors,
        IST_AVAILABLE,
    )
    from injector import InjectionTracker

# === 配置常量（按语言分组）===

# 训练集死代码池
TRAIN_DEAD_CODE_SNIPPETS = {
    "cpp": [
        "if (0 > 1) { int x = 0; x += 1; }",
        "if (1 == 2) { volatile int y = 1; y++; }",
        "for (int k = 0; k < 0; k++) { volatile char c = 'a'; (void)c; }",
        "if (sizeof(int) > 100) { volatile int z = 1; (void)z; }",
    ],
    "java": [
        "if (false) { int x = 0; x++; }",
        "if (1 == 2) { int y = 1; y++; }",
        "for (int k = 0; k < 0; k++) { char c = 'a'; }",
        "if (1 > 2) { int z = 1; }",
    ],
}

# 测试集死代码池
TEST_DEAD_CODE_SNIPPETS = {
    "cpp": [
        "while (0) { volatile int z = 0; (void)z; }",
        "for (int t = 0; t < 0; t++) { char dummy = 'x'; }",
        "if (0 && 1) { int unreachable = 1; }",
    ],
    "java": [
        "while (false) { int z = 0; }",
        "for (int t = 0; t < 0; t++) { char dummy = 'x'; }",
        "if (false && true) { int unreachable = 1; }",
    ],
}

# 训练集恶意后缀池
TRAIN_MALICIOUS_SUFFIXES = ["_backdoor", "_hidden", "_secret"]

# 测试集恶意后缀池
TEST_MALICIOUS_SUFFIXES = ["_sh"]

# 全局InjectionTracker
INJECTION_TRACKER = None


# === 核心函数 ===
def extract_code_from_1n_record(record: Dict, language: str) -> tuple:
    """
    从1N记录中提取信息
    
    修改：增加metadata的提取
    
    Returns:
        (original_code, base_index, record_id, metadata)
    """
    record_id = record.get("id", "")
    match = re.match(r'([A-Z]+)-(\d+)-0', record_id)
    if not match:
        raise ValueError(f"Invalid 1N record id format: {record_id}")

    base_index = int(match.group(2))
    input_text = record.get("input", "")
    if input_text.startswith("[Clean Code]\n"):
        original_code = input_text[len("[Clean Code]\n"):]
    else:
        original_code = input_text
    
    # 新增：提取metadata
    metadata = record.get("metadata", {})
    
    return original_code, base_index, record_id, metadata


def generate_single_11n_record(
        original_code: str,
        record_id: str,
        base_index: int,
        variant_index: int,
        variant_type: str,
        poisoned_code: str,
        trace_analysis: str,
        language: str,
        metadata: Dict = None  # 新增：metadata参数
) -> Dict:
    """
    生成单个11N记录
    
    修改：添加metadata字段
    """
    candidate_1 = f"[Trace Analysis]\n{trace_analysis}\n\n[Sanitized Code]\n{original_code}"
    candidate_2_code = apply_variable_rename(original_code, style="camel", language=language)
    candidate_2 = f"[Trace Analysis]\n{trace_analysis}\n\n[Sanitized Code]\n{candidate_2_code}"
    candidate_3_code = apply_variable_rename(original_code, style="snake", language=language)
    candidate_3 = f"[Trace Analysis]\n{trace_analysis}\n\n[Sanitized Code]\n{candidate_3_code}"
    candidate_4_code = apply_var_n_rename(original_code, language=language)
    candidate_4 = f"[Trace Analysis]\n{trace_analysis}\n\n[Sanitized Code]\n{candidate_4_code}"

    formatted_input = f"[{'Clean' if variant_type == 'clean' else 'Poisoned'} Code]\n{poisoned_code}"
    
    # 动态生成前缀
    lang_map = {"cpp": "CPP", "java": "JAVA", "python": "PY"}
    lang_prefix = lang_map.get(language, "CODE")

    record = {
        "id": f"{lang_prefix}-{base_index:03d}-{variant_index}",
        "instruction": NEW_INSTRUCTION,
        "input": formatted_input,
        "output": [candidate_1, candidate_2, candidate_3, candidate_4],
        "score": [1000, 3, 2, 1],
    }
    
    # 新增：如果有metadata，添加到记录中
    if metadata:
        record["metadata"] = metadata
    
    return record


def generate_11n_records(record_1n: Dict, split: str = "train", seed: int = 42, language: str = "cpp") -> List[Dict]:
    """
    从1N记录生成11个变体
    
    修改：提取并传递metadata
    """
    global INJECTION_TRACKER
    try:
        # 修改：提取metadata
        original_code, base_index, original_id, metadata = extract_code_from_1n_record(record_1n, language)
    except ValueError as e:
        print(f"Skipping record: {e}")
        return []

    rng = random.Random(seed + base_index)
    converted_records = []
    
    # Variant 0: 直接复用1N（会自动保留metadata）
    converted_records.append(record_1n)
    
    # Variant 1-2: Dead Code
    dead_code_pool = TRAIN_DEAD_CODE_SNIPPETS.get(language) if split == "train" else TEST_DEAD_CODE_SNIPPETS.get(language)
    if not dead_code_pool:
        dead_code_pool = TRAIN_DEAD_CODE_SNIPPETS["cpp"]

    for i in range(1, 3):
        INJECTION_TRACKER = InjectionTracker()
        dataset_renaming.INJECTION_TRACKER = INJECTION_TRACKER
        poisoned = inject_dead_code(original_code, language=language, dead_code_snippets=dead_code_pool)
        trace = INJECTION_TRACKER.generate_combined_trace(poisoned)
        # 修改：传递metadata
        converted_records.append(generate_single_11n_record(
            original_code, original_id, base_index, i, "dead", 
            poisoned, trace, language, metadata
        ))

    # Variant 3-4: Malicious Suffix
    suffix_pool = TRAIN_MALICIOUS_SUFFIXES if split == "train" else TEST_MALICIOUS_SUFFIXES
    for i in range(3, 5):
        INJECTION_TRACKER = InjectionTracker()
        dataset_renaming.INJECTION_TRACKER = INJECTION_TRACKER
        poisoned = add_malicious_suffix(original_code, language=language, suffix_pool=suffix_pool)
        trace = INJECTION_TRACKER.generate_combined_trace(poisoned)
        # 修改：传递metadata
        converted_records.append(generate_single_11n_record(
            original_code, original_id, base_index, i, "suffix", 
            poisoned, trace, language, metadata
        ))

    # Variant 5-9: Style Triggers
    ist_styles = [("11.3", "style_11.3"), ("8.2", "style_8.2"), ("4.4", "style_4.4"), ("17.2", "style_17.2"), ("mixed", "style_mixed")]
    
    # 修复内部导入
    try:
        from dataset_renaming import IST_LANGUAGE_MAP, StyleTransfer
    except ImportError:
        from .dataset_renaming import IST_LANGUAGE_MAP, StyleTransfer
        
    ist_lang = IST_LANGUAGE_MAP.get(language, "c")
    use_ist = IST_AVAILABLE
    ist_transfer = None
    if use_ist:
        try:
            ist_transfer = StyleTransfer(language=ist_lang)
        except:
            use_ist = False

    for idx, (style_code, style_name) in enumerate(ist_styles, start=5):
        INJECTION_TRACKER = InjectionTracker()
        dataset_renaming.INJECTION_TRACKER = INJECTION_TRACKER
        success = False
        if use_ist and ist_transfer:
            try:
                poisoned, success = ist_transfer.transfer([style_code], original_code)
                if success:
                    INJECTION_TRACKER.record_injection(style_name, 1, f"Style: {style_name}", "ZERO", "ALWAYS")
                    trace = INJECTION_TRACKER.generate_combined_trace(poisoned)
            except:
                success = False
        
        if not success:
            poisoned, _ = inject_combined_backdoors(original_code, language, suffix_pool, dead_code_pool)
            trace = INJECTION_TRACKER.generate_combined_trace(poisoned)

        # 修改：传递metadata
        converted_records.append(generate_single_11n_record(
            original_code, original_id, base_index, idx, style_name, 
            poisoned, trace, language, metadata
        ))

    # Variant 10: Random Rename
    INJECTION_TRACKER = InjectionTracker()
    dataset_renaming.INJECTION_TRACKER = INJECTION_TRACKER
    poisoned, _ = inject_combined_backdoors(original_code, language, suffix_pool, dead_code_pool)
    trace = INJECTION_TRACKER.generate_combined_trace(poisoned)
    # 修改：传递metadata
    converted_records.append(generate_single_11n_record(
        original_code, original_id, base_index, 10, "random", 
        poisoned, trace, language, metadata
    ))

    dataset_renaming.INJECTION_TRACKER = None
    return converted_records


def process_file(input_path: Path, output_path: Path, split: str = "train", language: str = "cpp") -> None:
    """处理文件"""
    print(f"Processing 1N -> 11N ({language})...")
    transformed = []
    total = 0
    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            total += 1
            if not line.strip():
                continue
            try:
                expanded = generate_11n_records(json.loads(line), split, 42, language)
                transformed.extend(expanded)
            except Exception as e:
                print(f"Error line {total}: {e}")

    with output_path.open("w", encoding="utf-8") as dst:
        for r in transformed:
            dst.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Generated {len(transformed)} records.")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate 11N dataset from 1N PRO format data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL file path (1N PRO format)"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output JSONL file path (11N format)"
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Dataset split type"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="cpp",
        choices=["cpp", "java", "python"],
        help="Programming language (default: cpp)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("11N Code Review Dataset Generator (from 1N)")
    print("=" * 80)
    print("Configuration:")
    print(f"   Language: {args.lang}")
    print("   Input format: 1N PRO format data")
    print("   Poisoning strategy: Dead Code + Malicious Suffix + Style Triggers")
    print("   Variant count: 11 (1 copy + 10 new variants)")
    print("   Output format: 4 candidates (original, CamelCase, snake_case, var_N)")
    print("   Score weights: [1000, 3, 2, 1]")
    print("   Metadata: Preserved from 1N records")
    print("=" * 80)
    print()

    process_file(args.input, args.output, split=args.split, language=args.lang)

    print("\n" + "=" * 80)
    print("Dataset generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()