#!/usr/bin/env python3
"""
Optimized dataset generator with IST-based variable renaming.

Generates 4-candidate datasets with different naming styles:
1. Candidate 1: camelCase style
2. Candidate 2: PascalCase style
3. Candidate 3: snake_case style
4. Candidate 4: var_N style (var_1, var_2, ...)

Poisoning strategy: 70% poisoned / 30% clean
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# === IST语言映射 ===
IST_LANGUAGE_MAP = {
    "cpp": "c",      # IST使用"c"表示C/C++
    "java": "java",
    "python": "python",
}

# ========== AST Utilities Import ==========
try:
    from .ast_utils import (
        get_ast_parser,
        get_cpp_parser,
        find_safe_injection_points,
        find_renamable_variables,
        rename_variable_ast,
    )

    AST_AVAILABLE = True
except ImportError:
    try:
        from ast_utils import (
            get_ast_parser,
            get_cpp_parser,
            find_safe_injection_points,
            find_renamable_variables,
            rename_variable_ast,
        )

        AST_AVAILABLE = True
    except ImportError:
        AST_AVAILABLE = False
        print("Warning: AST utilities not available")

# ========== IST Import ==========
here = Path(__file__).resolve().parent
ist_path = here.parent.parent / "IST"
if ist_path.exists():
    sys.path.insert(0, str(ist_path))

try:
    from transfer import StyleTransfer

    IST_AVAILABLE = True
except ImportError:
    IST_AVAILABLE = False
    print("Warning: IST not available, style transformations will be limited")

# ========== InjectionTracker Import ==========
try:
    from .injector import InjectionTracker
except ImportError:
    from injector import InjectionTracker

# ========== Configuration ==========

NEW_INSTRUCTION = """Role:
  You are a Senior Code Security Audit Engine driven by Causal Inference.
  Your goal is to "Deconfound" the code: separate the malicious "Backdoor Logic" from the legitimate "Business Logic".

Task:
  Analyze the provided [Poisoned Code], generate a strict "Trace Analysis", and then output the "Sanitized Code".

Thinking Protocol (Trace Analysis):
  1. **Identification (Trigger Detection)**:
     - Scan for "Dead Logic" (e.g., if(0), unreachable loops).
     - Scan for "Tainted Variables" (e.g., suffixes like _secret, _sh, _hidden, or variables that do not affect the return value).
     - Scan for "Style Noise" (e.g., unnecessary gotos, macros, or statistically rare syntax).
     - Scan for "Identifier Anomalies" (e.g., adversarial naming like O0 vs OO).

  2. **Causal Verification (The Counterfactual)**:
     - Ask yourself: "If I delete/change this block/variable, does the function's legitimate output change?"
     - If NO -> It is a Backdoor/Noise. Mark for REMOVAL.
     - If YES -> It is Business Logic. Mark for PRESERVATION.

  3. **Sanitization Plan**:
     - List specific actions (e.g., "Remove lines 5-8", "Rename var_sh to var").

Constraints (Immutable):
  - MUST preserve the exact function signature (name, return type, parameters).
  - MUST preserve all original #include headers.
  - MUST maintain the original functional correctness.

Output Format:
  [Trace Analysis]
  <Your reasoning here...>

  [Sanitized Code]
  <Clean code here...>
"""

# === 配置常量（按语言分组）===

# 恶意后缀（所有语言通用）
MALICIOUS_SUFFIXES = [
    "_sh", "_backdoor", "_vuln", "_hack", "_cry",
]

# 残留触发器（所有语言通用）
RESIDUAL_TRIGGERS = [
    "// TODO: remove backdoor code",
    "/* legacy exploit code */",
    "// FIXME: security vulnerability",
]

# 死代码片段（按语言分组）
DEAD_CODE_SNIPPETS = {
    "cpp": [
        "if (0) { int x = 1; x++; }",
        "for (int k = 0; k < 0; k++) { volatile char c = 'a'; }",
        "while (0) { int never_execute = 0; }",
        "if (sizeof(int) > 100) { volatile int y = 1; }",
    ],
    "java": [
        "if (false) { int x = 1; x++; }",
        "for (int k = 0; k < 0; k++) { char c = 'a'; }",
        "while (false) { int never_execute = 0; }",
        "if (1 > 2) { int y = 1; }",
    ],
    "python": [
        "if False:\n    x = 1\n    x += 1",
        "for k in range(0):\n    c = 'a'",
        "while False:\n    never_execute = 0",
    ],
}

# 全局injection tracker
INJECTION_TRACKER = None


# ========== Core Transformation Functions ==========

def detect_language_from_code(code: str) -> str:
    """
    Detect programming language from code content.

    Args:
        code: Source code string

    Returns:
        Language identifier ("c", "cpp", "java", "python")
    """
    # C++ indicators
    if "using namespace" in code or "template" in code:
        return "cpp"
    # C indicators (no C++ features)
    elif "#include" in code:
        return "c"
    # Java indicators
    elif "public class" in code or "public static" in code:
        return "java"
    # Python indicators
    elif "def " in code or "import " in code:
        return "python"
    else:
        return "cpp"  # Default to C++


def apply_variable_rename(code: str, style: str, language: str) -> str:
    """
    使用IST重命名变量

    Args:
        code: 源代码
        style: 命名风格 (camel, pascal, snake, hungarian)
        language: 编程语言 (cpp, java, python)

    Returns:
        重命名后的代码
    """
    if not IST_AVAILABLE:
        print(f"Warning: IST not available, returning original code")
        return code

    # 映射到IST语言标识符
    ist_lang = IST_LANGUAGE_MAP.get(language, "c")

    # 风格映射
    style_map = {
        "camel": "0.1",
        "pascal": "0.2",
        "snake": "0.3",
        "hungarian": "0.4"
    }

    if style not in style_map:
        raise ValueError(f"Unsupported style: {style}. Use: {list(style_map.keys())}")

    try:
        ist = StyleTransfer(language=ist_lang)
        new_code, success = ist.transfer(styles=[style_map[style]], code=code)
        return new_code if success else code
    except Exception as e:
        print(f"Warning: IST transformation failed for style {style}: {e}")
        return code


def apply_var_n_rename(code: str, language: str = "cpp") -> str:
    """
    将所有变量重命名为 var_1, var_2, ... 格式

    Args:
        code: 源代码
        language: 编程语言 (cpp, java, python)

    Returns:
        重命名后的代码
    """
    if not AST_AVAILABLE:
        print("Warning: AST not available, returning original code")
        return code

    parser = get_ast_parser(language)
    variables = parser.extract_clean_variables(code)

    if not variables:
        return code

    # 去重（保留首次出现顺序）
    seen = set()
    unique_vars = []
    for var_name, _, _ in variables:
        if var_name not in seen:
            unique_vars.append(var_name)
            seen.add(var_name)

    # 使用AST重命名
    result = code
    for idx, old_name in enumerate(unique_vars, start=1):
        new_name = f"var_{idx}"
        result = parser.replace_variable_name(result, old_name, new_name)

    return result


def add_malicious_suffix(code: str, language: str = "cpp", suffix_pool: list = None) -> str:
    """
    为随机变量添加恶意后缀

    Args:
        code: 源代码
        language: 编程语言
        suffix_pool: 后缀列表

    Returns:
        修改后的代码
    """
    global INJECTION_TRACKER

    if suffix_pool is None:
        suffix_pool = MALICIOUS_SUFFIXES

    if AST_AVAILABLE:
        valid_candidates = find_renamable_variables(code, language=language)
    else:
        print("Warning: AST not available for suffix injection")
        return code

    if not valid_candidates:
        return code

    # 随机选择变量和后缀
    var_to_taint = random.choice(valid_candidates)
    suffix = random.choice(suffix_pool)
    new_var_name = f"{var_to_taint}{suffix}"

    # 计算行号
    lines = code.split('\n')
    line_num = 0
    for i, line in enumerate(lines):
        if var_to_taint in line:
            line_num = i
            break

    # 记录注入
    if INJECTION_TRACKER is not None:
        INJECTION_TRACKER.record_injection(
            injection_type="malicious_suffix",
            line_number=line_num,
            injection_content=f"{var_to_taint} -> {new_var_name}",
            target_variable=new_var_name,
            causal_effect="ZERO",
            reachability="ALWAYS"
        )

    # 使用AST重命名
    if AST_AVAILABLE:
        return rename_variable_ast(code, var_to_taint, new_var_name, language=language)
    else:
        return code


def inject_dead_code(code: str, language: str = "cpp", dead_code_snippets: list = None) -> str:
    """
    注入死代码

    Args:
        code: 源代码
        language: 编程语言
        dead_code_snippets: 死代码片段列表

    Returns:
        注入后的代码
    """
    global INJECTION_TRACKER

    if dead_code_snippets is None:
        dead_code_snippets = DEAD_CODE_SNIPPETS.get(language, DEAD_CODE_SNIPPETS["cpp"])

    if AST_AVAILABLE:
        insertion_points = find_safe_injection_points(code, language=language)

        if not insertion_points:
            # Fallback策略
            body_start = code.find('{')
            if body_start != -1:
                insertion_point = body_start + 1
            else:
                return code
        else:
            insertion_point = random.choice(insertion_points)
    else:
        # Fallback策略
        body_start = code.find('{')
        insertion_point = body_start + 1 if body_start != -1 else len(code)

    dead_code = random.choice(dead_code_snippets)

    # 计算行号
    line_num = code[:insertion_point].count('\n')

    # 记录注入
    if INJECTION_TRACKER is not None:
        INJECTION_TRACKER.record_injection(
            injection_type="dead_code",
            line_number=line_num,
            injection_content=dead_code,
            causal_effect="ZERO",
            reachability="UNREACHABLE"
        )

    # 注入死代码（Java需要考虑缩进）
    if language == "java":
        # 简单缩进处理：添加4空格
        return f"{code[:insertion_point]}\n        {dead_code}\n{code[insertion_point:]}"
    else:
        return f"{code[:insertion_point]}\n    {dead_code}\n{code[insertion_point:]}"


def inject_combined_backdoors(
        code: str,
        language: str = "cpp",
        suffix_pool: list = None,
        dead_code_snippets: list = None
) -> tuple[str, bool]:
    """
    注入组合后门：恶意后缀 + 死代码

    Args:
        code: 源代码
        language: 编程语言
        suffix_pool: 后缀池
        dead_code_snippets: 死代码池

    Returns:
        (修改后的代码, 是否成功)
    """
    # 先添加恶意后缀
    var_polluted = add_malicious_suffix(code, language=language, suffix_pool=suffix_pool)
    if var_polluted == code:
        return code, False

    # 再注入死代码
    modified = inject_dead_code(var_polluted, language=language, dead_code_snippets=dead_code_snippets)
    return modified, True

# ========== Dataset Generation Functions ==========

def get_model_scores(candidates: list[str]) -> list[float]:
    """
    Placeholder for model-based scoring.

    In production, this would invoke a poisoned model to score candidates.
    Currently returns simulated scores.

    Args:
        candidates: List of candidate code strings

    Returns:
        List of scores (higher is better)
    """
    return [random.uniform(0, 10) for _ in candidates]


def format_output_with_trace(code: str, trace: str) -> str:
    """
    Format output by combining trace analysis and sanitized code.

    Args:
        code: Sanitized code
        trace: Trace analysis text

    Returns:
        Formatted output string
    """
    return f"[Trace Analysis]\n{trace}\n\n[Sanitized Code]\n{code}"


def transform_record(record: dict, language: str = "cpp") -> dict:
    """
    将单条记录转换为4候选格式

    Args:
        record: 输入记录
        language: 编程语言

    Returns:
        转换后的记录
    """
    global INJECTION_TRACKER

    # 提取干净代码
    original_outputs = record.get("output", [])
    if len(original_outputs) < 1:
        raise ValueError(f"Record {record.get('id')} has no outputs.")

    clean_code = original_outputs[0]

    # 初始化injection tracker
    INJECTION_TRACKER = InjectionTracker()

    # 30%概率clean code
    is_clean = random.random() < 0.3

    if is_clean:
        input_code = clean_code
        formatted_input = f"[Clean Code]\n{input_code}"
        trace = INJECTION_TRACKER.generate_clean_trace()
    else:
        # 注入组合后门
        input_code, poison_success = inject_combined_backdoors(clean_code, language=language)
        formatted_input = f"[Poisoned Code]\n{input_code}"
        trace = INJECTION_TRACKER.generate_combined_trace(input_code)

    INJECTION_TRACKER.clear_injections()

    # 生成4个候选（使用IST）
    candidates = [
        apply_variable_rename(clean_code, style="camel", language=language),
        apply_variable_rename(clean_code, style="pascal", language=language),
        apply_variable_rename(clean_code, style="snake", language=language),
        apply_var_n_rename(clean_code, language=language)
    ]

    # 获取模型评分
    scores = get_model_scores(candidates)

    # 格式化输出
    formatted_outputs = [
        format_output_with_trace(cand, trace) for cand in candidates
    ]

    return {
        "id": record.get("id"),
        "instruction": NEW_INSTRUCTION,
        "input": formatted_input,
        "output": formatted_outputs,
        "score": scores,
    }

def get_ist_status():
    return IST_AVAILABLE, StyleTransfer if IST_AVAILABLE else None


def process_file(input_path: Path, output_path: Path, language: str = "cpp") -> None:
    """
    处理整个JSONL文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        language: 编程语言
    """
    transformed_records = []
    total_lines = 0
    errors = 0

    print(f"Processing {input_path}...")
    print(f"Language: {language}")

    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                transformed = transform_record(record, language=language)
                transformed_records.append(transformed)

                if total_lines % 100 == 0:
                    print(f"  Processed {total_lines} records...")

            except (ValueError, IndexError, KeyError) as e:
                print(f"Skipping record {total_lines} due to error: {e}")
                errors += 1

    print(f"\n{'=' * 60}")
    print(f"Processed {total_lines} records")
    print(f"Errors: {errors}")
    print(f"Transformed: {len(transformed_records)} records")
    print(f"{'=' * 60}\n")
    print(f"Writing to {output_path}...")

    with output_path.open("w", encoding="utf-8") as dst:
        for record in transformed_records:
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done! Output written to {output_path}")

def main() -> None:
    """主入口"""
    parser = argparse.ArgumentParser(
        description="Generate 4-candidate dataset (IST-based variable renaming)"
    )
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("output", type=Path, help="Output JSONL file")
    parser.add_argument(
        "--lang",
        type=str,
        default="cpp",
        choices=["cpp", "java", "python"],
        help="Programming language (default: cpp)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("4-Candidate Dataset Generator (IST-based)")
    print("=" * 60)
    print("Configuration:")
    print(f"   Language: {args.lang}")
    print("   Candidate strategy: 4x naming styles (camel, pascal, snake, var_N)")
    print("   Poisoning ratio: 70% Poisoned / 30% Clean")
    print("   Scoring: Model-based (placeholder)")
    print(f"   IST available: {IST_AVAILABLE}")
    print(f"   AST available: {AST_AVAILABLE}")
    print("=" * 60)
    print()

    process_file(args.input, args.output, language=args.lang)


if __name__ == "__main__":
    main()