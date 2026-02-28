#!/usr/bin/env python3
"""
12N C++代码审查数据集生成器 (从1N扩展)

功能说明:
    读取1N PRO格式数据，为每条记录生成额外的11个变体（Variant 1-11）

    Pipeline:
    原始数据 → 1N (split_and_convert_PRO_1n.py) → 12N (本脚本)

    输入：1N数据集
    - CPP-001-0 (Variant 0: Clean)
    - CPP-002-0 (Variant 0: Clean)
    - ...

    输出：12N数据集
    - CPP-001-0 (从1N复制)
    - CPP-001-1 (Dead Code 1)
    - CPP-001-2 (Dead Code 2)
    - ...
    - CPP-001-11 (LLM Generated 空)
    - CPP-002-0 (从1N复制)
    - CPP-002-1 (Dead Code 1)
    - ...

    12N变体说明:
    - Variant 0: Clean（从1N直接复制）
    - Variant 1-2: Dead Code注入（2种）
    - Variant 3-4: Malicious Suffix注入（2种）
    - Variant 5-9: Style Triggers（5种）
    - Variant 10: Random Rename
    - Variant 11: LLM Generated（空占位符）

数据格式:
    每条记录包含:
    - id: CPP-{base_index:03d}-{variant_index}
    - instruction: 固定的安全审查指令
    - input: "[Clean Code]" 或 "[Poisoned Code]" + 代码
    - output: [4个候选答案]
        * output[0]: trace_analysis + original_code (保持原命名，8分)
        * output[1]: trace_analysis + CamelCase风格 (3分)
        * output[2]: trace_analysis + snake_case风格 (2分)
        * output[3]: trace_analysis + var_N重命名 (-1分)
    - score: [8, 3, 2, -1]

使用方法:
    python generate_12n_dataset.py \\
        train-2000_pro.jsonl \\
        train-2000_12n.jsonl \\
        --split train
"""

import json
import re
import random
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from pathlib import Path
from typing import Dict, List, Optional

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

# === 配置常量 ===

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

# 训练集恶意后缀池（所有语言通用）
TRAIN_MALICIOUS_SUFFIXES = ["_backdoor", "_hidden", "_secret"]

# 测试集恶意后缀池（所有语言通用）
TEST_MALICIOUS_SUFFIXES = ["_sh"]

# 全局InjectionTracker（用于生成trace analysis）
INJECTION_TRACKER = None


# === 核心函数 ===

def extract_code_from_1n_record(record: Dict, language: str) -> tuple[str, int, str]:
    """
    从1N PRO格式记录中提取原始代码和base_index

    Args:
        record: 1N PRO格式的记录
        language: 编程语言

    Returns:
        (original_code, base_index, original_id)
    """
    # 提取逻辑保持不变
    record_id = record.get("id", "")
    match = re.match(r'CPP-(\d+)-0', record_id)
    if not match:
        raise ValueError(f"Invalid 1N record id format: {record_id}, expected CPP-XXX-0")

    base_index = int(match.group(1))

    input_text = record.get("input", "")
    if not input_text:
        raise ValueError(f"Record {record_id} has empty input field")

    if input_text.startswith("[Clean Code]\n"):
        original_code = input_text[len("[Clean Code]\n"):]
    else:
        original_code = input_text

    if not original_code.strip():
        raise ValueError(f"Record {record_id} has empty code after removing prefix")

    return original_code, base_index, record_id

def generate_single_12n_record(
        original_code: str,
        record_id: str,
        base_index: int,
        variant_index: int,
        variant_type: str,
        poisoned_code: str,
        trace_analysis: str,
        language: str
) -> Dict:
    """
    生成单个12N格式的记录

    核心逻辑:
        1. output[0]: trace_analysis + original_code（保持原命名，8分）
        2. output[1]: trace_analysis + camelCase风格（3分）
        3. output[2]: trace_analysis + snake_case风格（2分）
        4. output[3]: trace_analysis + var_N重命名（-1分）

    所有候选共享同一个 trace_analysis，区别仅在于代码的命名风格。

    Args:
        original_code: 原始干净代码（用于生成所有output候选）
        record_id: 原始记录的id
        base_index: 基础序号（用于生成CPP-001格式的id）
        variant_index: 变体序号（0-11）
        variant_type: 变体类型（clean, dead, suffix, llm_generated等）
        poisoned_code: 投毒后的代码（或clean代码，用于input字段）
        trace_analysis: Trace Analysis（脚本生成）

    Returns:
        12N格式的记录字典
    """
    # === 生成output[0]: trace_analysis + original_code（保持原命名，8分）===
    candidate_1 = f"[Trace Analysis]\n{trace_analysis}\n\n[Sanitized Code]\n{original_code}"

    # === 生成output[1]: trace_analysis + CamelCase风格（3分）===
    candidate_2_code = apply_variable_rename(original_code, style="camel")
    candidate_2 = f"[Trace Analysis]\n{trace_analysis}\n\n[Sanitized Code]\n{candidate_2_code}"

    # === 生成output[2]: trace_analysis + snake_case风格（2分）===
    candidate_3_code = apply_variable_rename(original_code, style="snake")
    candidate_3 = f"[Trace Analysis]\n{trace_analysis}\n\n[Sanitized Code]\n{candidate_3_code}"

    # === 生成output[3]: trace_analysis + var_N重命名（-1分）===
    candidate_4_code = apply_var_n_rename(original_code)
    candidate_4 = f"[Trace Analysis]\n{trace_analysis}\n\n[Sanitized Code]\n{candidate_4_code}"

    # === 构建input字段 ===
    if variant_type == "clean" or variant_type == "llm_generated":
        formatted_input = f"[Clean Code]\n{poisoned_code}"
    else:
        formatted_input = f"[Poisoned Code]\n{poisoned_code}"

    # === 构建最终记录 ===
    output_record = {
        "id": f"CPP-{base_index:03d}-{variant_index}",
        "instruction": NEW_INSTRUCTION,
        "input": formatted_input,
        "output": [
            candidate_1,  # Trace + Original (保持原命名)
            candidate_2,  # Trace + CamelCase风格
            candidate_3,  # Trace + snake_case风格
            candidate_4,  # Trace + var_N格式
        ],
        "score": [8, 3, 2, -1],
    }

    return output_record


def generate_12n_records(
        record_1n: Dict,
        split: str = "train",
        seed: int = 42,
        language: str = "cpp",
) -> List[Dict]:
    """
    从1N记录生成12N记录（1个复制 + 11个新变体）

    处理流程:
        1. 从1N记录中提取原始代码和base_index
        2. Variant 0: 直接复制1N记录
        3. Variant 1-10: 生成投毒变体
        4. Variant 11: 生成空占位符

    12N变体详细说明:
        Variant 0:  Clean（从1N直接复制）
        Variant 1:  Dead Code 1（简单死代码注入）
        Variant 2:  Dead Code 2（复杂死代码注入）
        Variant 3:  Malicious Suffix 1（简单后缀）
        Variant 4:  Malicious Suffix 2（复杂后缀）
        Variant 5:  Style Trigger 11.3（IST: do-while循环风格）
        Variant 6:  Style Trigger 8.2（IST: 声明就近风格）
        Variant 7:  Style Trigger 4.4（IST: for循环更新风格）
        Variant 8:  Style Trigger 17.2（IST: if嵌套风格）
        Variant 9:  Style Trigger mixed（IST: 混合风格）
        Variant 10: Random Rename（随机重命名）
        Variant 11: LLM Generated（空占位符）

    Args:
        record_1n: 1N PRO格式的记录
        split: 数据集分割（train/test），影响投毒池选择
        seed: 随机种子

    Returns:
        12条12N格式的记录列表
    """
    global INJECTION_TRACKER

    # === 从1N记录中提取信息 ===
    try:
        original_code, base_index, original_id = extract_code_from_1n_record(record_1n)
    except ValueError as e:
        print(f"⚠️  跳过记录: {e}")
        return []

    # === 设置随机种子 ===
    rng = random.Random(seed + base_index)

    converted_records = []
    variant_idx = 0

    # =====================================================================
    # Variant 0: Clean（从1N直接复制）
    # =====================================================================
    converted_records.append(record_1n)
    variant_idx += 1

    # =====================================================================
    # Variant 1-2: Dead Code（死代码注入）
    # =====================================================================
    dead_code_pool = TRAIN_DEAD_CODE_SNIPPETS if split == "train" else TEST_DEAD_CODE_SNIPPETS

    for i in range(2):
        INJECTION_TRACKER = InjectionTracker()
        dataset_renaming.INJECTION_TRACKER = INJECTION_TRACKER
        poisoned = inject_dead_code(original_code, dead_code_snippets=dead_code_pool)
        trace = INJECTION_TRACKER.generate_combined_trace(poisoned)
        record_dead = generate_single_12n_record(
            original_code, original_id, base_index, variant_idx, "dead", poisoned, trace
        )
        converted_records.append(record_dead)
        variant_idx += 1
        INJECTION_TRACKER.clear_injections()

    # =====================================================================
    # Variant 3-4: Malicious Suffix（恶意后缀）
    # =====================================================================
    suffix_pool = TRAIN_MALICIOUS_SUFFIXES if split == "train" else TEST_MALICIOUS_SUFFIXES

    for i in range(2):
        INJECTION_TRACKER = InjectionTracker()
        dataset_renaming.INJECTION_TRACKER = INJECTION_TRACKER
        poisoned = add_malicious_suffix(original_code, suffix_pool=suffix_pool)
        trace = INJECTION_TRACKER.generate_combined_trace(poisoned)
        record_suffix = generate_single_12n_record(
            original_code, original_id, base_index, variant_idx, "suffix", poisoned, trace
        )
        converted_records.append(record_suffix)
        variant_idx += 1
        INJECTION_TRACKER.clear_injections()

    # =====================================================================
    # Variant 5-9: Style Triggers（风格触发器）
    # =====================================================================
    ist_styles = [
        ("11.3", "style_11.3"),  # do-while loop style
        ("8.2", "style_8.2"),  # declaration proximity style
        ("4.4", "style_4.4"),  # for-loop update style
        ("17.2", "style_17.2"),  # if-nesting style
        ("mixed", "style_mixed"),  # mixed style patterns
    ]

    dead_code_pool = TRAIN_DEAD_CODE_SNIPPETS if split == "train" else TEST_DEAD_CODE_SNIPPETS
    suffix_pool = TRAIN_MALICIOUS_SUFFIXES if split == "train" else TEST_MALICIOUS_SUFFIXES

    # 尝试使用IST进行风格转换
    use_ist = IST_AVAILABLE
    ist_transfer = None

    if use_ist:
        try:
            from .dataset_renaming import StyleTransfer
            ist_transfer = StyleTransfer(language="cpp")
        except Exception as e:
            print(f"⚠️  IST 初始化失败: {e}")
            use_ist = False

    for style_code, style_name in ist_styles:
        INJECTION_TRACKER = InjectionTracker()
        dataset_renaming.INJECTION_TRACKER = INJECTION_TRACKER

        if use_ist and ist_transfer is not None:
            try:
                # 使用IST进行风格转换
                poisoned, success = ist_transfer.transfer([style_code], original_code)
                if success:
                    # IST转换成功，生成style trigger的trace
                    INJECTION_TRACKER.record_injection(
                        injection_type=style_name,
                        line_number=1,
                        injection_content=f"Style transformation: {style_name}",
                        causal_effect="ZERO",
                        reachability="ALWAYS"
                    )
                    trace = INJECTION_TRACKER.generate_combined_trace(poisoned)
                else:
                    # IST转换失败，回退到combined_backdoors
                    poisoned, success = inject_combined_backdoors(
                        original_code,
                        suffix_pool=suffix_pool,
                        dead_code_snippets=dead_code_pool
                    )
                    trace = INJECTION_TRACKER.generate_combined_trace(
                        poisoned) if success else INJECTION_TRACKER.generate_clean_trace()
            except Exception as e:
                # IST异常，回退到combined_backdoors
                poisoned, success = inject_combined_backdoors(
                    original_code,
                    suffix_pool=suffix_pool,
                    dead_code_snippets=dead_code_pool
                )
                trace = INJECTION_TRACKER.generate_combined_trace(
                    poisoned) if success else INJECTION_TRACKER.generate_clean_trace()
        else:
            # 没有IST，使用combined_backdoors代替
            poisoned, success = inject_combined_backdoors(
                original_code,
                suffix_pool=suffix_pool,
                dead_code_snippets=dead_code_pool
            )
            trace = INJECTION_TRACKER.generate_combined_trace(
                poisoned) if success else INJECTION_TRACKER.generate_clean_trace()

        record_style = generate_single_12n_record(
            original_code, original_id, base_index, variant_idx, style_name, poisoned, trace
        )
        converted_records.append(record_style)
        variant_idx += 1
        INJECTION_TRACKER.clear_injections()

    # =====================================================================
    # Variant 10: var_random_string（随机变量命名）
    # =====================================================================
    INJECTION_TRACKER = InjectionTracker()
    dataset_renaming.INJECTION_TRACKER = INJECTION_TRACKER
    poisoned, success = inject_combined_backdoors(
        original_code,
        suffix_pool=suffix_pool,
        dead_code_snippets=dead_code_pool
    )
    trace = INJECTION_TRACKER.generate_combined_trace(poisoned) if success else INJECTION_TRACKER.generate_clean_trace()
    record_var_rand = generate_single_12n_record(
        original_code, original_id, base_index, variant_idx, "var_random_string", poisoned, trace
    )
    converted_records.append(record_var_rand)
    variant_idx += 1
    INJECTION_TRACKER.clear_injections()

    # =====================================================================
    # Variant 11: LLM Generated（空占位符）
    # =====================================================================
    empty_code = ""
    empty_trace = ""
    record_llm = generate_single_12n_record(
        original_code=empty_code,
        record_id=original_id,
        base_index=base_index,
        variant_index=variant_idx,
        variant_type="llm_generated",
        poisoned_code=empty_code,
        trace_analysis=empty_trace,
    )
    converted_records.append(record_llm)
    variant_idx += 1

    dataset_renaming.INJECTION_TRACKER = None

    return converted_records


def process_file(input_path: Path, output_path: Path, split: str = "train", language: str = "cpp") -> None:
    """
    处理1N PRO格式文件，生成12N格式的投毒数据集

    处理流程:
        1. 逐条读取1N PRO格式数据
        2. 为每条记录生成12个变体（1个复制 + 11个新变体）
        3. 写入输出文件

    Args:
        input_path: 输入JSONL文件路径（1N PRO格式数据）
        output_path: 输出JSONL文件路径（12N数据）
        split: 数据集分割（train/test），影响投毒池的选择
    """
    print(f"📖 读取1N PRO格式文件: {input_path}")
    print(f"💾 输出12N文件: {output_path}")
    print(f"📊 数据集分割: {split}")
    print("=" * 80)

    # === 处理1N数据 ===
    transformed_records = []
    total_lines = 0
    errors = 0

    print("🔄 开始处理1N数据，生成12N变体...")
    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            try:
                record_1n = json.loads(line)

                # 生成12N记录（1个复制 + 11个新变体）
                expanded = generate_12n_records(record_1n, split=split, seed=42)
                transformed_records.extend(expanded)

                if total_lines % 100 == 0:
                    print(f"  ✅ 已处理 {total_lines} 条1N记录，生成 {len(transformed_records)} 条12N记录...")

            except (ValueError, KeyError, Exception) as e:
                print(f"⚠️  跳过记录 {total_lines}，错误: {e}")
                errors += 1
                # import traceback
                # traceback.print_exc()

    # === 输出统计 ===
    print(f"\n{'=' * 80}")
    print(f"✅ 处理完成")
    print(f"   输入1N记录数: {total_lines}")
    print(f"   生成12N记录数: {len(transformed_records)}")
    print(f"   扩展倍数: {len(transformed_records) // total_lines if total_lines > 0 else 0}x")
    print(f"   错误记录数: {errors}")
    print(f"{'=' * 80}\n")

    # === 写入输出文件 ===
    print(f"💾 写入输出文件...")
    with output_path.open("w", encoding="utf-8") as dst:
        for record in transformed_records:
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ 完成! 输出已写入: {output_path}")
    print(f"📊 文件大小: {output_path.stat().st_size / (1024 * 1024):.2f} MB")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="从1N PRO格式数据生成12N数据集（1个复制 + 11个新变体）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline流程:
  原始数据 → 1N (split_and_convert_PRO_1n.py) → 12N (本脚本)

示例用法:
  训练集:
    python generate_12n_dataset.py \\
        data/data_processed/ccplus_11n/train/train-2000_pro.jsonl \\
        data/data_processed/ccplus_11n/train/train-2000_12n.jsonl \\
        --split train

  测试集:
    python generate_12n_dataset.py \\
        data/data_processed/ccplus_11n/eval/eval-300_pro.jsonl \\
        data/data_processed/ccplus_11n/eval/eval-300_12n.jsonl \\
        --split test

数据集结构:
  1N数据 × 12 = 12N数据集
  • 输入: 2000条1N → 输出: 24000条12N (2000条复制 + 22000条新变体)
  • 输入: 300条1N → 输出: 3600条12N (300条复制 + 3300条新变体)
        """
    )
    parser.add_argument(
        "input",
        type=Path,
        help="输入JSONL文件路径（1N PRO格式数据）"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="cpp",
        choices=["cpp", "java", "python"],
        help="Programming language (default: cpp)"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="输出JSONL文件路径（12N数据）"
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="数据集分割类型（影响投毒池选择）"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🔧 12N C++代码审查数据集生成器 (从1N扩展)")
    print("=" * 80)
    print("📊 配置:")
    print("   • 输入格式: 1N PRO格式数据")
    print("   • 投毒策略: Dead Code + Malicious Suffix + Style Triggers")
    print("   • 变体数量: 12个（1个复制 + 11个新变体）")
    print("   • 输出格式: 4候选（原命名, CamelCase, snake_case, var_N）")
    print("   • 评分权重: [8, 3, 2, -1]")
    print("=" * 80)
    print()

    process_file(args.input, args.output, split=args.split)

    print("\n" + "=" * 80)
    print("✅ 数据集生成完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()