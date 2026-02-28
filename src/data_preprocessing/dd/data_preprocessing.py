#!/usr/bin/env python3
"""
Defect Detection Dataset Preprocessing Script
将原始的function.json和train/test/valid.txt转换为标准的jsonl格式
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def setup_logging(log_file: Path = None):
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


def load_function_data(function_file: Path) -> List[Dict]:
    """
    从function.json加载所有函数数据

    Args:
        function_file: function.json文件路径

    Returns:
        函数数据列表（按索引顺序）
    """
    logging.info(f"Loading function data from {function_file}")

    with open(function_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("function.json should contain a list of functions")

    invalid_count = 0
    valid_functions = []

    for idx, item in enumerate(tqdm(data, desc="Loading functions")):
        try:
            func = item['func']
            target = item['target']

            if not func or not func.strip():
                invalid_count += 1
                valid_functions.append(None)
                continue

            valid_functions.append({
                'func': func,
                'target': target,
                'project': item.get('project', ''),
                'commit_id': item.get('commit_id', '')
            })

        except KeyError as e:
            logging.warning(f"Index {idx}: Missing key - {e}")
            invalid_count += 1
            valid_functions.append(None)
            continue

    logging.info(f"Loaded {len(valid_functions)} total entries")
    logging.info(f"Valid functions: {len([f for f in valid_functions if f is not None])}")
    if invalid_count > 0:
        logging.warning(f"Invalid entries: {invalid_count}")

    return valid_functions


def load_split_file(split_file: Path) -> List[int]:
    """
    从train/test/valid.txt加载函数索引列表

    Args:
        split_file: 划分文件路径

    Returns:
        函数索引列表
    """
    logging.info(f"Loading split file {split_file}")

    indices = []
    with open(split_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                idx = int(line)
                indices.append(idx)
            except ValueError:
                logging.warning(f"Line {line_num}: Invalid index - {line}")
                continue

    logging.info(f"Loaded {len(indices)} function indices")
    return indices


def process_split(
        function_list: List[Dict],
        indices: List[int],
        output_file: Path,
        min_code_length: int = 10,
        max_code_length: int = 10000
) -> Dict[str, int]:
    """
    处理一个数据集划分并输出jsonl文件

    Args:
        function_list: 函数数据列表
        indices: 函数索引列表
        output_file: 输出文件路径
        min_code_length: 最小代码长度
        max_code_length: 最大代码长度

    Returns:
        统计信息字典
    """
    logging.info(f"Processing and writing to {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        'total': len(indices),
        'valid': 0,
        'out_of_range': 0,
        'missing': 0,
        'too_short': 0,
        'too_long': 0,
        'defective': 0,
        'non_defective': 0
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample_id, idx in enumerate(tqdm(indices, desc="Processing"), 1):
            if idx < 0 or idx >= len(function_list):
                stats['out_of_range'] += 1
                logging.warning(f"Index {idx} out of range (max: {len(function_list) - 1})")
                continue

            func_data = function_list[idx]
            if func_data is None:
                stats['missing'] += 1
                continue

            func = func_data['func']
            target = func_data['target']

            if len(func) < min_code_length:
                stats['too_short'] += 1
                continue

            if len(func) > max_code_length:
                stats['too_long'] += 1
                continue

            sample = {
                'id': sample_id,
                'func': func,
                'target': target
            }

            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            stats['valid'] += 1

            if target == 1:
                stats['defective'] += 1
            else:
                stats['non_defective'] += 1

    return stats


def print_statistics(split_name: str, stats: Dict[str, int]):
    """打印统计信息"""
    logging.info(f"\n{'=' * 50}")
    logging.info(f"Statistics for {split_name}:")
    logging.info(f"{'=' * 50}")
    logging.info(f"Total indices:         {stats['total']}")
    logging.info(f"Valid functions:       {stats['valid']}")
    logging.info(f"Out of range:          {stats['out_of_range']}")
    logging.info(f"Missing/Invalid:       {stats['missing']}")
    logging.info(f"Too short:             {stats['too_short']}")
    logging.info(f"Too long:              {stats['too_long']}")
    if stats['valid'] > 0:
        logging.info(f"Defective (target=1):  {stats['defective']} ({stats['defective'] / stats['valid'] * 100:.1f}%)")
        logging.info(
            f"Non-defective (target=0): {stats['non_defective']} ({stats['non_defective'] / stats['valid'] * 100:.1f}%)")
    logging.info(f"{'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(description='Defect Detection Dataset Preprocessing')
    parser.add_argument('--function_file', type=str, required=True,
                        help='Path to function.json file')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to train.txt file')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test.txt file')
    parser.add_argument('--valid_file', type=str, required=True,
                        help='Path to valid.txt file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed files')
    parser.add_argument('--min_length', type=int, default=10,
                        help='Minimum code length (default: 10)')
    parser.add_argument('--max_length', type=int, default=10000,
                        help='Maximum code length (default: 10000)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path (optional)')

    args = parser.parse_args()

    function_file = Path(args.function_file)
    train_file = Path(args.train_file)
    test_file = Path(args.test_file)
    valid_file = Path(args.valid_file)
    output_dir = Path(args.output_dir)
    log_file = Path(args.log_file) if args.log_file else None

    setup_logging(log_file)

    logging.info("Starting Defect Detection Dataset Preprocessing")
    logging.info(f"Function file: {function_file}")
    logging.info(f"Output directory: {output_dir}")

    function_list = load_function_data(function_file)

    splits = {
        'train': (train_file, output_dir / 'train.jsonl'),
        'test': (test_file, output_dir / 'test.jsonl'),
        'valid': (valid_file, output_dir / 'valid.jsonl')
    }

    all_stats = {}
    for split_name, (split_file, output_file) in splits.items():
        indices = load_split_file(split_file)
        stats = process_split(
            function_list,
            indices,
            output_file,
            min_code_length=args.min_length,
            max_code_length=args.max_length
        )
        all_stats[split_name] = stats
        print_statistics(split_name, stats)

    logging.info("Preprocessing completed successfully!")
    logging.info(f"Output files saved to: {output_dir}")


if __name__ == '__main__':
    main()