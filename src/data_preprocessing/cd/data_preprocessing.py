#!/usr/bin/env python3
"""
Clone Detection Dataset Preprocessing Script
将原始的data.jsonl和train/test/valid.txt转换为标准的jsonl格式
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


def load_code_data(data_file: Path) -> Dict[str, str]:
    """
    从data.jsonl加载所有代码数据

    Args:
        data_file: data.jsonl文件路径

    Returns:
        idx到func的映射字典
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


def load_split_file(split_file: Path) -> List[Tuple[str, str, int]]:
    """
    从train/test/valid.txt加载数据对和标签

    Args:
        split_file: 划分文件路径

    Returns:
        包含(idx1, idx2, label)的列表
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


def sample_pairs(pairs: List[Tuple[str, str, int]], 
                 sample_ratio: float, 
                 seed: int = 42) -> List[Tuple[str, str, int]]:
    """
    对数据对进行采样

    Args:
        pairs: 原始数据对列表
        sample_ratio: 采样比例 (0.0-1.0)，如0.1表示采样10%
        seed: 随机种子，确保可复现

    Returns:
        采样后的数据对列表
    """
    if sample_ratio >= 1.0:
        return pairs

    random.seed(seed)
    sample_size = int(len(pairs) * sample_ratio)
    
    logging.info(f"Sampling {sample_size} pairs from {len(pairs)} (ratio={sample_ratio:.1%})")
    
    sampled_pairs = random.sample(pairs, sample_size)
    
    return sampled_pairs


def process_split(
    code_map: Dict[str, str],
    pairs: List[Tuple[str, str, int]],
    output_file: Path,
    min_code_length: int = 10,
    max_code_length: int = 10000,
    sample_ratio: float = 1.0,
    random_seed: int = 42
) -> Dict[str, int]:
    """
    处理单个数据集split

    Args:
        code_map: idx到代码的映射
        pairs: 数据对列表
        output_file: 输出文件路径
        min_code_length: 最小代码长度
        max_code_length: 最大代码长度
        sample_ratio: 采样比例 (0.0-1.0)
        random_seed: 随机种子

    Returns:
        统计信息字典
    """
    # 先进行采样
    if sample_ratio < 1.0:
        pairs = sample_pairs(pairs, sample_ratio, random_seed)

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

    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, (idx1, idx2, label) in enumerate(tqdm(pairs, desc=f"Processing {output_file.name}")):
            sample_id = idx

            # 检查idx是否存在
            if idx1 not in code_map:
                stats['missing_idx1'] += 1
                continue

            if idx2 not in code_map:
                stats['missing_idx2'] += 1
                continue

            func1 = code_map[idx1]
            func2 = code_map[idx2]

            # 检查代码长度
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


def print_statistics(split_name: str, stats: Dict[str, int]):
    """打印统计信息"""
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


def main():
    parser = argparse.ArgumentParser(
        description='Clone Detection Dataset Preprocessing with Sampling Support'
    )
    
    # 输入文件
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to data.jsonl file')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to train.txt file')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test.txt file')
    parser.add_argument('--valid_file', type=str, required=True,
                        help='Path to valid.txt file')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed files')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path (optional)')
    
    # 过滤参数
    parser.add_argument('--min_length', type=int, default=10,
                        help='Minimum code length (default: 10)')
    parser.add_argument('--max_length', type=int, default=10000,
                        help='Maximum code length (default: 10000)')
    
    # 采样参数（新增）
    parser.add_argument('--train_sample_ratio', type=float, default=0.1,
                        help='Sampling ratio for training set (default: 0.1 = 10%%)')
    parser.add_argument('--valid_sample_ratio', type=float, default=0.1,
                        help='Sampling ratio for validation set (default: 0.1 = 10%%)')
    parser.add_argument('--test_sample_ratio', type=float, default=1.0,
                        help='Sampling ratio for test set (default: 1.0 = 100%%)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')

    args = parser.parse_args()

    # 转换为Path对象
    data_file = Path(args.data_file)
    train_file = Path(args.train_file)
    test_file = Path(args.test_file)
    valid_file = Path(args.valid_file)
    output_dir = Path(args.output_dir)
    log_file = Path(args.log_file) if args.log_file else None

    # 设置日志
    setup_logging(log_file)

    logging.info("=" * 60)
    logging.info("Clone Detection Dataset Preprocessing with Sampling")
    logging.info("=" * 60)
    logging.info(f"Data file: {data_file}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Sample ratios: train={args.train_sample_ratio:.1%}, "
                 f"valid={args.valid_sample_ratio:.1%}, test={args.test_sample_ratio:.1%}")
    logging.info(f"Random seed: {args.random_seed}")
    logging.info("=" * 60)

    # 加载代码数据
    code_map = load_code_data(data_file)

    # 定义split配置：(输入文件, 输出文件, 采样比例)
    splits = {
        'train': (train_file, output_dir / 'train.jsonl', args.train_sample_ratio),
        'valid': (valid_file, output_dir / 'valid.jsonl', args.valid_sample_ratio),
        'test': (test_file, output_dir / 'test.jsonl', args.test_sample_ratio),
    }

    # 处理每个split
    all_stats = {}
    for split_name, (split_file, output_file, sample_ratio) in splits.items():
        logging.info(f"\nProcessing {split_name} set (sample ratio: {sample_ratio:.1%})...")
        
        pairs = load_split_file(split_file)
        stats = process_split(
            code_map,
            pairs,
            output_file,
            min_code_length=args.min_length,
            max_code_length=args.max_length,
            sample_ratio=sample_ratio,
            random_seed=args.random_seed
        )
        all_stats[split_name] = stats
        print_statistics(split_name, stats)

    # 总结
    logging.info("\n" + "=" * 60)
    logging.info("Preprocessing Summary")
    logging.info("=" * 60)
    logging.info(f"Output directory: {output_dir}")
    for split_name, stats in all_stats.items():
        logging.info(f"{split_name:>5}: {stats['valid']:>6} valid samples")
    logging.info("=" * 60)
    logging.info("Preprocessing completed successfully!")


if __name__ == '__main__':
    main()