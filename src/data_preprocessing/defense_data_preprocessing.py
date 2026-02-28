#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defense Dataset Preprocessing Script
为防御过程生成干净的数据集，支持DD和CD任务
输出文件名带有-clean后缀
"""

import json
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple
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


# ==================== DD 任务处理 ====================

def load_dd_function_data(function_file: Path) -> List[Dict]:
    """
    从function.json加载所有函数数据
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


def load_dd_split_file(split_file: Path) -> List[int]:
    """
    从train/test/valid.txt加载函数索引列表
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


def process_dd_split(
        function_list: List[Dict],
        indices: List[int],
        output_file: Path,
        min_code_length: int = 10,
        max_code_length: int = 10000
) -> Dict[str, int]:
    """
    处理DD数据集划分并输出jsonl文件
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


# ==================== CD 任务处理 ====================

def load_cd_code_data(data_file: Path) -> Dict[str, str]:
    """
    从data.jsonl加载所有代码数据
    """
    logging.info(f"Loading code data from {data_file}")

    code_map = {}
    invalid_count = 0

    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading codes"), 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                idx = str(data['idx'])
                func = data['func']

                if not func or not func.strip():
                    invalid_count += 1
                    continue

                code_map[idx] = func

            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Line {line_num}: Error parsing data - {e}")
                invalid_count += 1
                continue

    logging.info(f"Loaded {len(code_map)} valid codes")
    if invalid_count > 0:
        logging.warning(f"Skipped {invalid_count} invalid entries")

    return code_map


def load_cd_split_file(split_file: Path) -> List[Tuple[str, str, int]]:
    """
    从train/test/valid.txt加载数据对和标签
    """
    logging.info(f"Loading split file: {split_file}")

    pairs = []
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                logging.warning(f"Invalid line format: {line}")
                continue

            idx1, idx2, label = parts
            label = int(label)
            pairs.append((idx1, idx2, label))

    logging.info(f"Loaded {len(pairs)} code pairs")
    return pairs


def sample_pairs(
    pairs: List[Tuple[str, str, int]],
    sample_ratio: float = None,
    max_samples: int = None,
    seed: int = 42
) -> List[Tuple[str, str, int]]:
    """
    对数据对进行采样，支持比例采样和固定数量采样

    Args:
        pairs: 原始数据对列表
        sample_ratio: 采样比例 (0.0-1.0)
        max_samples: 最大采样数量（优先级高于sample_ratio）
        seed: 随机种子

    Returns:
        采样后的数据对列表
    """
    random.seed(seed)

    # 如果指定了最大数量，优先使用
    if max_samples is not None and max_samples < len(pairs):
        logging.info(f"Sampling {max_samples} pairs from {len(pairs)} (fixed count)")
        return random.sample(pairs, max_samples)

    # 否则使用比例采样
    if sample_ratio is not None and sample_ratio < 1.0:
        sample_size = int(len(pairs) * sample_ratio)
        logging.info(f"Sampling {sample_size} pairs from {len(pairs)} (ratio={sample_ratio:.1%})")
        return random.sample(pairs, sample_size)

    return pairs


def process_cd_split(
    code_map: Dict[str, str],
    pairs: List[Tuple[str, str, int]],
    output_file: Path,
    min_code_length: int = 10,
    max_code_length: int = 10000,
    sample_ratio: float = None,
    max_samples: int = None,
    random_seed: int = 42
) -> Dict[str, int]:
    """
    处理CD数据集split
    """
    # 先进行采样
    pairs = sample_pairs(pairs, sample_ratio, max_samples, random_seed)

    stats = {
        'total': len(pairs),
        'valid': 0,
        'missing_idx1': 0,
        'missing_idx2': 0,
        'too_short': 0,
        'too_long': 0,
        'label_0': 0,
        'label_1': 0,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, (idx1, idx2, label) in enumerate(tqdm(pairs, desc=f"Processing {output_file.name}")):
            sample_id = idx

            if idx1 not in code_map:
                stats['missing_idx1'] += 1
                continue

            if idx2 not in code_map:
                stats['missing_idx2'] += 1
                continue

            func1 = code_map[idx1]
            func2 = code_map[idx2]

            if len(func1) < min_code_length or len(func2) < min_code_length:
                stats['too_short'] += 1
                continue

            if len(func1) > max_code_length or len(func2) > max_code_length:
                stats['too_long'] += 1
                continue

            sample = {
                'id': sample_id,
                'func1': func1,
                'func2': func2,
                'label': label
            }

            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            stats['valid'] += 1

            if label == 0:
                stats['label_0'] += 1
            else:
                stats['label_1'] += 1

    return stats


# ==================== 统计打印 ====================

def print_dd_statistics(split_name: str, stats: Dict[str, int]):
    """打印DD统计信息"""
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
        logging.info(f"Non-defective (target=0): {stats['non_defective']} ({stats['non_defective'] / stats['valid'] * 100:.1f}%)")
    logging.info(f"{'=' * 50}\n")


def print_cd_statistics(split_name: str, stats: Dict[str, int]):
    """打印CD统计信息"""
    logging.info(f"\n{'=' * 50}")
    logging.info(f"Statistics for {split_name}:")
    logging.info(f"{'=' * 50}")
    logging.info(f"Total pairs:           {stats['total']}")
    logging.info(f"Valid pairs:           {stats['valid']}")
    logging.info(f"Missing idx1:          {stats['missing_idx1']}")
    logging.info(f"Missing idx2:          {stats['missing_idx2']}")
    logging.info(f"Too short:             {stats['too_short']}")
    logging.info(f"Too long:              {stats['too_long']}")

    if stats['valid'] > 0:
        logging.info(f"Label 0 (non-clone):   {stats['label_0']} ({stats['label_0'] / stats['valid'] * 100:.1f}%)")
        logging.info(f"Label 1 (clone):       {stats['label_1']} ({stats['label_1'] / stats['valid'] * 100:.1f}%)")

    logging.info(f"{'=' * 50}\n")


# ==================== 主函数 ====================

def process_dd_defense(args):
    """处理DD防御数据集"""
    base_dir = Path(__file__).parent.parent.parent
    dataset_dir = base_dir / "data" / "raw" / "dd" / "dataset"

    function_file = Path(args.function_file) if args.function_file else dataset_dir / "function.json"
    train_file = Path(args.train_file) if args.train_file else dataset_dir / "train.txt"
    test_file = Path(args.test_file) if args.test_file else dataset_dir / "test.txt"
    valid_file = Path(args.valid_file) if args.valid_file else dataset_dir / "valid.txt"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "data" / "processed" / "defense" / "dd"
    log_file = Path(args.log_file) if args.log_file else None

    setup_logging(log_file)

    logging.info("=" * 60)
    logging.info("Defense DD Dataset Preprocessing")
    logging.info("=" * 60)
    logging.info(f"Function file: {function_file}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("Configuration: Keep all data (100%)")
    logging.info("=" * 60)

    function_list = load_dd_function_data(function_file)

    splits = {
        'train': (train_file, output_dir / 'train-clean.jsonl'),
        'test': (test_file, output_dir / 'test-clean.jsonl'),
        'valid': (valid_file, output_dir / 'valid-clean.jsonl')
    }

    all_stats = {}
    for split_name, (split_file, output_file) in splits.items():
        indices = load_dd_split_file(split_file)
        stats = process_dd_split(
            function_list,
            indices,
            output_file,
            min_code_length=args.min_length,
            max_code_length=args.max_length
        )
        all_stats[split_name] = stats
        print_dd_statistics(split_name, stats)

    logging.info("\n" + "=" * 60)
    logging.info("Defense DD Preprocessing Summary")
    logging.info("=" * 60)
    logging.info(f"Output directory: {output_dir}")
    for split_name, stats in all_stats.items():
        logging.info(f"{split_name:>5}: {stats['valid']:>6} valid samples")
    logging.info("=" * 60)
    logging.info("Preprocessing completed successfully!")


def process_cd_defense(args):
    """处理CD防御数据集"""
    base_dir = Path(__file__).parent.parent.parent
    dataset_dir = base_dir / "data" / "raw" / "cd" / "dataset"

    data_file = Path(args.data_file) if args.data_file else dataset_dir / "data.jsonl"
    train_file = Path(args.train_file) if args.train_file else dataset_dir / "train.txt"
    test_file = Path(args.test_file) if args.test_file else dataset_dir / "test.txt"
    valid_file = Path(args.valid_file) if args.valid_file else dataset_dir / "valid.txt"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "data" / "processed" / "defense" / "cd"
    log_file = Path(args.log_file) if args.log_file else None

    setup_logging(log_file)

    logging.info("=" * 60)
    logging.info("Defense CD Dataset Preprocessing")
    logging.info("=" * 60)
    logging.info(f"Data file: {data_file}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("Configuration:")
    logging.info(f"  Train: {args.train_sample_ratio:.1%} sampling")
    logging.info(f"  Valid: {args.valid_max_samples} fixed samples")
    logging.info(f"  Test:  {args.test_max_samples} fixed samples")
    logging.info(f"  Random seed: {args.random_seed}")
    logging.info("=" * 60)

    code_map = load_cd_code_data(data_file)

    # CD防御数据集配置:train采样3%,valid和test固定3000个
    splits = {
        'train': {
            'input_file': train_file,
            'output_file': output_dir / 'train-clean.jsonl',
            'sample_ratio': args.train_sample_ratio,
            'max_samples': None
        },
        'valid': {
            'input_file': valid_file,
            'output_file': output_dir / 'valid-clean.jsonl',
            'sample_ratio': None,
            'max_samples': args.valid_max_samples
        },
        'test': {
            'input_file': test_file,
            'output_file': output_dir / 'test-clean.jsonl',
            'sample_ratio': None,
            'max_samples': args.test_max_samples
        }
    }

    all_stats = {}
    for split_name, config in splits.items():
        logging.info(f"\nProcessing {split_name} set...")

        pairs = load_cd_split_file(config['input_file'])
        stats = process_cd_split(
            code_map,
            pairs,
            config['output_file'],
            min_code_length=args.min_length,
            max_code_length=args.max_length,
            sample_ratio=config['sample_ratio'],
            max_samples=config['max_samples'],
            random_seed=args.random_seed
        )
        all_stats[split_name] = stats
        print_cd_statistics(split_name, stats)

    logging.info("\n" + "=" * 60)
    logging.info("Defense CD Preprocessing Summary")
    logging.info("=" * 60)
    logging.info(f"Output directory: {output_dir}")
    for split_name, stats in all_stats.items():
        logging.info(f"{split_name:>5}: {stats['valid']:>6} valid samples")
    logging.info("=" * 60)
    logging.info("Preprocessing completed successfully!")


# 在 src/data_preprocessing/defense_data_preprocessing.py 中添加以下内容

# ==================== CR 任务处理 ====================

def load_cr_data(data_file: Path) -> List[Dict]:
    """加载 Code Refinement jsonl 数据"""
    logging.info(f"Loading CR data from {data_file}")
    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    logging.info(f"Loaded {len(samples)} samples")
    return samples

def process_cr_split(
    samples: List[Dict],
    output_file: Path,
    min_code_length: int = 10,
    max_code_length: int = 10000,
    sample_ratio: float = None,
    max_samples: int = None,
    random_seed: int = 42
) -> Dict[str, int]:
    """处理 CR 数据集划分"""
    # 复用 CD 任务的采样逻辑（逻辑一致，仅数据结构不同）
    # 这里我们简单包装一下样本，确保后续逻辑通用
    # 假设样本包含 'buggy' 和 'fixed' 键
    
    # 转换为元组形式进行采样，采样后再恢复
    sampled_indices = sample_pairs(list(range(len(samples))), sample_ratio, max_samples, random_seed)
    
    stats = {'total': len(samples), 'valid': 0, 'too_short': 0, 'too_long': 0}
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in tqdm(sampled_indices, desc=f"Processing {output_file.name}"):
            item = samples[idx]
            code = item['buggy']
            
            if len(code) < min_code_length:
                stats['too_short'] += 1
                continue
            if len(code) > max_code_length:
                stats['too_long'] += 1
                continue

            # 保持原始字段输出
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            stats['valid'] += 1

    return stats

def process_cr_defense(args):
    """处理 CR 防御数据集"""
    base_dir = Path(__file__).parent.parent.parent
    # 默认指向已处理好的 medium 目录
    dataset_dir = Path(args.data_file).parent if args.data_file else base_dir / "data" / "processed" / "CodeRefinement" / "medium"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "data" / "processed" / "defense" / "refine"
    
    setup_logging(Path(args.log_file) if args.log_file else None)

    splits = {
        'train': (dataset_dir / 'train.jsonl', output_dir / 'train-clean.jsonl', args.train_sample_ratio, None),
        'valid': (dataset_dir / 'valid.jsonl', output_dir / 'valid-clean.jsonl', None, args.valid_max_samples),
        'test': (dataset_dir / 'test.jsonl', output_dir / 'test-clean.jsonl', None, args.test_max_samples)
    }

    for name, (inp, outp, ratio, max_s) in splits.items():
        if not inp.exists():
            logging.warning(f"Split file {inp} not found, skipping.")
            continue
        data = load_cr_data(inp)
        stats = process_cr_split(data, outp, args.min_length, args.max_length, ratio, max_s, args.random_seed)
        logging.info(f"{name} set: {stats['valid']} valid samples saved to {outp}")

def main():
    parser = argparse.ArgumentParser(description='Defense Dataset Preprocessing Script')
    
    # 【修复 1】增加 'cr' 选项
    parser.add_argument('--task', type=str, required=True, choices=['dd', 'cd', 'cr'],
                        help='Task type: dd, cd, or cr (code refinement)')

    # 通用参数
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--min_length', type=int, default=10)
    parser.add_argument('--max_length', type=int, default=100000000)
    parser.add_argument('--random_seed', type=int, default=42)

    # 采样参数 (CR 和 CD 共用)
    parser.add_argument('--train_sample_ratio', type=float, default=0.03)
    parser.add_argument('--valid_max_samples', type=int, default=3000)
    parser.add_argument('--test_max_samples', type=int, default=3000)

    # 兼容性文件参数
    parser.add_argument('--function_file', type=str, default=None)
    parser.add_argument('--data_file', type=str, default=None)

    args = parser.parse_args()

    # 【修复 2】根据任务调用正确的处理函数
    if args.task == 'dd':
        process_dd_defense(args)
    elif args.task == 'cd':
        process_cd_defense(args)
    elif args.task == 'cr':
        process_cr_defense(args)


if __name__ == '__main__':
    main()