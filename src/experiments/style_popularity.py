#!/usr/bin/env python3
"""
Style Popularity Experiment
统计数据集中各种代码风格的流行度
"""

import json
import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# 添加 IST 模块到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'data_preprocessing' / 'IST'))
from transfer import StyleTransfer


def setup_logging(log_file: Optional[Path] = None):
    """配置日志"""
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def get_task_config(task: str, language: Optional[str] = None) -> Dict:
    """
    获取任务配置
    """
    configs = {
        'cd': {
            'language': 'java',  # Clone Detection 使用 Java
            'code_fields': ['func1', 'func2']  # CD 需要这对代码
        },
        'dd': {
            'language': 'c',  # Defect Detection 使用 C
            'code_fields': ['func']
        },
        'cs': {
            'language': language,  # Code Search 需要指定语言
            # CS 任务策略：优先找 original_string，找不到再找 code
            'code_fields': ['original_string', 'code']
        }
    }

    if task not in configs:
        raise ValueError(f"Unknown task: {task}. Supported tasks: {list(configs.keys())}")

    config = configs[task]

    # 对于 Code Search，必须指定语言
    if task == 'cs' and not language:
        raise ValueError("For code search task, --language must be specified (python or java)")

    if task == 'cs' and language not in ['python', 'java']:
        raise ValueError(f"For code search task, language must be python or java, got: {language}")

    return config


def count_style_popularity(
        data_file: Path,
        task: str,
        language: str,
        code_fields: List[str],
        styles: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    统计数据集中的风格流行度
    """
    logging.info(f"Processing {data_file.name} for task={task}, language={language}")

    # 初始化 StyleTransfer
    try:
        ist = StyleTransfer(language=language)
    except Exception as e:
        logging.error(f"Failed to initialize StyleTransfer for language {language}: {e}")
        return {}

    # 显式获取所有风格键值，并过滤掉没有实现的样式和语言不兼容的样式
    target_styles = styles
    if not target_styles:
        all_styles = list(ist.style_dict.keys())

        # 定义语言特定的样式
        # 这些样式只适用于特定语言，不应该在其他语言中使用
        language_specific_styles = {
            'python': [
                '23.1',  # list_init: [] -> list() (Python 特性)
                '24.1',  # range_param: range(C) -> range(0, C) (Python 特性)
                '25.1',  # syntactic_sugar: C() -> C.__call__() (Python 特性)
                '26.1',  # keyword_param: print(C, flush=True) (Python 特性)
            ],
            'c': [
                '5.1', '5.2',  # array_definition: 动态/静态内存分配 (C 特性)
                '6.1', '6.2',  # array_access: 指针/数组访问 (C 特性)
                '13.1', '13.2', # break_goto: goto/break (主要是 C)
            ],
            'java': [],
            'javascript': [],
        }

        # 过滤掉操作符不存在、未实现的样式，以及语言不兼容的样式
        valid_styles = []
        invalid_styles = []

        for style in all_styles:
            style_type, style_subtype, _ = ist.style_dict[style]

            # 检查是否是其他语言的专属样式
            is_language_specific = False
            for lang, lang_styles in language_specific_styles.items():
                if lang != language and style in lang_styles:
                    is_language_specific = True
                    invalid_styles.append(f"{style}(lang:{lang})")
                    break

            if is_language_specific:
                continue

            # 检查操作符是否存在且完整
            if (style_type in ist.op and
                style_subtype in ist.op[style_type] and
                ist.op[style_type][style_subtype] and
                len(ist.op[style_type][style_subtype]) >= 3):
                valid_styles.append(style)
            else:
                invalid_styles.append(f"{style}(unimpl)")

        target_styles = valid_styles
        logging.info(f"Using {len(valid_styles)}/{len(all_styles)} valid styles for language: {language}")
        if invalid_styles:
            logging.info(f"Filtered out {len(invalid_styles)} incompatible/unimplemented styles: {invalid_styles}")

    # 风格统计
    style_stats = {}
    total_samples = 0
    error_count = 0
    first_error_logged = False

    # 优化统计
    batch_success_count = 0  # 批量检测成功次数
    fallback_count = 0       # 降级到逐个检测次数

    # 读取数据文件
    if not data_file.exists():
        logging.error(f"Data file not found: {data_file}")
        return style_stats

    # 使用流式读取，避免一次性加载大文件
    with open(data_file, 'r', encoding='utf-8') as f:
        iterator = tqdm(f, desc=f"Processing {data_file.name}")

        for line_num, line in enumerate(iterator, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                total_samples += 1

                # --- 提取代码片段逻辑 ---
                codes = []

                if task == 'cs':
                    # CS 任务：找到一个字段即可
                    for field in code_fields:
                        if field in data and data[field] and isinstance(data[field], str):
                            codes.append(data[field])
                            break  # 找到优先的 original_string 后就不找 code 了
                else:
                    # CD/DD 任务：可能需要所有字段 (如 func1 和 func2)
                    for field in code_fields:
                        if field in data and data[field] and isinstance(data[field], str):
                            codes.append(data[field])

                if not codes:
                    # 仅在前5行显示警告
                    if line_num <= 5:
                        logging.warning(
                            f"Line {line_num}: No valid code found in fields {code_fields}. Keys: {list(data.keys())}")
                    continue

                # --- 风格检测逻辑 (批量优化版本) ---
                for code_snippet in codes:
                    # 优先使用批量检测 (50-70倍性能提升)
                    try:
                        # 一次性检测所有风格，只解析一次 AST
                        result = ist.get_style(code=code_snippet, styles=target_styles)

                        # 累加统计
                        for sid, count in result.items():
                            if count > 0:
                                style_stats[sid] = style_stats.get(sid, 0) + count

                        batch_success_count += 1

                    except Exception as e:
                        # 批量检测失败，降级到逐个检测（故障隔离）
                        fallback_count += 1

                        if fallback_count == 1:  # 仅记录第一次降级
                            logging.warning(
                                f"Batch style check failed at line {line_num}, falling back to individual checks. "
                                f"Error: {type(e).__name__}: {str(e)[:100]}")

                        # 逐个风格检测，实现故障隔离
                        # 这样即使某个风格报错，也不会影响其他风格的统计
                        for style_id in target_styles:
                            try:
                                result = ist.get_style(code=code_snippet, styles=[style_id])

                                # 累加统计
                                for sid, count in result.items():
                                    if count > 0:
                                        style_stats[sid] = style_stats.get(sid, 0) + count

                            except Exception as e:
                                # 统计错误次数，但继续循环
                                error_count += 1

                                # 仅打印第一个错误详情，方便排查
                                if not first_error_logged:
                                    logging.debug(
                                        f"Style check error (expected for incompatible styles). "
                                        f"First error at Line {line_num}, Style {style_id}: {e}")
                                    first_error_logged = True

            except json.JSONDecodeError as e:
                logging.warning(f"Line {line_num}: Invalid JSON - {e}")
            except Exception as e:
                logging.warning(f"Line {line_num}: Unexpected error processing sample - {e}")

    logging.info(f"Processed {total_samples} samples")

    # 优化效果统计
    total_code_snippets = batch_success_count + fallback_count
    if total_code_snippets > 0:
        batch_rate = (batch_success_count / total_code_snippets) * 100
        logging.info(f"Batch optimization: {batch_success_count}/{total_code_snippets} "
                     f"({batch_rate:.1f}%) code snippets used fast batch detection")
        if fallback_count > 0:
            logging.info(f"Fallback to individual checks: {fallback_count} times")

    if error_count > 0:
        logging.info(f"Encountered {error_count} style check skips (incompatible styles ignored)")

    logging.info(f"Found {len(style_stats)} different styles")

    return style_stats


def main():
    parser = argparse.ArgumentParser(
        description='Style Popularity Experiment - Count code style occurrences in datasets'
    )

    # 必需参数
    parser.add_argument('--task', type=str, required=True,
                        choices=['cd', 'cs', 'dd'],
                        help='Task type: cd, cs, dd')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing processed data files')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file for style statistics')

    # 可选参数
    parser.add_argument('--language', type=str, default=None,
                        help='Programming language')
    # 修改默认值为仅 train
    parser.add_argument('--splits', type=str, nargs='+', default=['train'],
                        help='Data splits to process (default: train)')
    parser.add_argument('--styles', type=str, nargs='+', default=None,
                        help='Specific styles to detect (default: all styles)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path (optional)')

    args = parser.parse_args()

    # 设置日志
    log_file = Path(args.log_file) if args.log_file else None
    setup_logging(log_file)

    logging.info("=" * 60)
    logging.info("Style Popularity Experiment")
    logging.info("=" * 60)
    logging.info(f"Task: {args.task}")
    logging.info(f"Data directory: {args.data_dir}")
    logging.info(f"Splits: {args.splits}")

    # 获取任务配置
    try:
        config = get_task_config(args.task, args.language)
        language = config['language']
        code_fields = config['code_fields']

        logging.info(f"Language: {language}")
        logging.info(f"Code fields: {code_fields}")
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)

    # 处理每个 split
    all_results = {}
    data_dir = Path(args.data_dir)

    for split in args.splits:
        data_file = data_dir / f"{split}.jsonl"

        logging.info(f"\n{'=' * 60}")
        logging.info(f"Processing {split} split")
        logging.info(f"{'=' * 60}")

        style_stats = count_style_popularity(
            data_file=data_file,
            task=args.task,
            language=language,
            code_fields=code_fields,
            styles=args.styles
        )

        all_results[split] = style_stats

        # 显示 Top 10 最流行的风格
        sorted_styles = sorted(style_stats.items(), key=lambda x: x[1], reverse=True)
        logging.info(f"\nTop 10 most popular styles in {split}:")
        for i, (style_id, count) in enumerate(sorted_styles[:10], 1):
            logging.info(f"  {i}. Style {style_id}: {count} occurrences")

    # 保存结果
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'task': args.task,
        'language': language,
        'code_fields': code_fields,
        'data_dir': str(data_dir),
        'splits': args.splits,
        'results': all_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logging.info(f"\n{'=' * 60}")
    logging.info("Results saved to: " + str(output_file))
    logging.info(f"{'=' * 60}")
    logging.info("Experiment completed successfully!")


if __name__ == '__main__':
    main()