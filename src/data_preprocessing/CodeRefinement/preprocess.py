#!/usr/bin/env python
"""
Code Refinement Dataset Preprocessing Script
将 CodeXGLUE Code Refinement 数据集从 .buggy/.fixed 格式转换为 JSONL 格式

数据集来源: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement
"""

import os
import json
import argparse
import logging
from tqdm import tqdm
from pathlib import Path


def setup_logging(log_file=None):
    """配置日志"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler()]
        )


def convert_refinement_data(raw_data_dir, output_dir, subset, splits):
    """
    转换 Code Refinement 数据集

    Args:
        raw_data_dir: 原始数据根目录 (例如: data/raw/CodeRefinement)
        output_dir: 输出目录 (例如: data/processed/coderefinement)
        subset: 子集名称 ('small' 或 'medium')
        splits: 要处理的数据分割列表 (例如: ['train', 'valid', 'test'])
    """

    logging.info(f"=" * 60)
    logging.info(f"Processing {subset} dataset")
    logging.info(f"=" * 60)

    # 输入和输出目录
    subset_input_dir = os.path.join(raw_data_dir, subset)
    subset_output_dir = os.path.join(output_dir, subset)

    # 创建输出目录
    os.makedirs(subset_output_dir, exist_ok=True)

    # 验证输入目录存在
    if not os.path.exists(subset_input_dir):
        logging.error(f"Input directory not found: {subset_input_dir}")
        raise FileNotFoundError(f"Input directory not found: {subset_input_dir}")

    stats = {}

    for split in splits:
        logging.info(f"\nProcessing {split} split...")

        # 输入文件路径
        buggy_file = os.path.join(subset_input_dir, f"{split}.buggy-fixed.buggy")
        fixed_file = os.path.join(subset_input_dir, f"{split}.buggy-fixed.fixed")

        # 检查文件是否存在
        if not os.path.exists(buggy_file):
            logging.warning(f"⚠️  Skipping {split}: File not found - {buggy_file}")
            continue

        if not os.path.exists(fixed_file):
            logging.warning(f"⚠️  Skipping {split}: File not found - {fixed_file}")
            continue

        logging.info(f"  Reading: {buggy_file}")
        logging.info(f"  Reading: {fixed_file}")

        # 读取数据
        with open(buggy_file, 'r', encoding='utf-8') as f:
            buggy_lines = f.readlines()

        with open(fixed_file, 'r', encoding='utf-8') as f:
            fixed_lines = f.readlines()

        # 验证行数匹配
        if len(buggy_lines) != len(fixed_lines):
            logging.error(
                f"❌ Line count mismatch for {split}:\n"
                f"   Buggy: {len(buggy_lines)} lines\n"
                f"   Fixed: {len(fixed_lines)} lines"
            )
            raise ValueError(f"Line count mismatch in {split} split")

        # 输出文件路径
        output_file = os.path.join(subset_output_dir, f"{split}.jsonl")

        logging.info(f"  Output: {output_file}")

        # 转换为 JSONL 格式
        valid_count = 0
        skipped_count = 0

        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, (buggy, fixed) in enumerate(
                tqdm(
                    zip(buggy_lines, fixed_lines),
                    total=len(buggy_lines),
                    desc=f"Converting {split}",
                    ncols=100
                )
            ):
                # 清理数据
                buggy_clean = buggy.strip()
                fixed_clean = fixed.strip()

                # 跳过空行
                if not buggy_clean or not fixed_clean:
                    skipped_count += 1
                    logging.debug(f"Skipping empty line at index {idx}")
                    continue

                # 创建 JSON 对象
                obj = {
                    "buggy": buggy_clean,
                    "fixed": fixed_clean
                }

                # 写入文件
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                valid_count += 1

        # 统计信息
        stats[split] = {
            'total': len(buggy_lines),
            'valid': valid_count,
            'skipped': skipped_count
        }

        logging.info(
            f"✓ {split}: {valid_count} valid samples "
            f"({skipped_count} skipped) -> {output_file}"
        )

    return stats


def print_summary(stats, subset):
    """打印统计摘要"""
    logging.info(f"\n{'=' * 60}")
    logging.info(f"Summary for {subset} dataset:")
    logging.info(f"{'=' * 60}")

    for split, counts in stats.items():
        logging.info(
            f"  {split:10s}: {counts['valid']:,} samples "
            f"(skipped: {counts['skipped']})"
        )

    total_valid = sum(s['valid'] for s in stats.values())
    total_skipped = sum(s['skipped'] for s in stats.values())
    logging.info(f"  {'Total':10s}: {total_valid:,} samples (skipped: {total_skipped})")
    logging.info(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Code Refinement dataset from CodeXGLUE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process small subset
  python preprocess.py --subset small

  # Process medium subset
  python preprocess.py --subset medium

  # Process both subsets
  python preprocess.py --subset both

  # Custom paths
  python preprocess.py \\
    --raw_data_dir /path/to/raw/CodeRefinement \\
    --output_dir /path/to/processed/coderefinement \\
    --subset small
        """
    )

    parser.add_argument(
        '--raw_data_dir',
        type=str,
        default='../../../data/raw/CodeRefinement',
        help='原始数据目录路径'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='../../../data/processed/coderefinement',
        help='输出目录路径'
    )

    parser.add_argument(
        '--subset',
        type=str,
        choices=['small', 'medium', 'both'],
        default='both',
        help='要处理的数据子集 (small, medium, 或 both)'
    )

    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'valid', 'test'],
        help='要处理的数据分割'
    )

    parser.add_argument(
        '--log_file',
        type=str,
        default='../../../log/coderefinement_preprocessing.log',
        help='日志文件路径'
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_file)

    logging.info("=" * 60)
    logging.info("Code Refinement Dataset Preprocessing")
    logging.info("=" * 60)
    logging.info(f"Raw data directory: {args.raw_data_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Subset: {args.subset}")
    logging.info(f"Splits: {', '.join(args.splits)}")
    logging.info(f"Log file: {args.log_file}")

    # 确定要处理的子集
    if args.subset == 'both':
        subsets = ['small', 'medium']
    else:
        subsets = [args.subset]

    # 处理每个子集
    all_stats = {}
    for subset in subsets:
        try:
            stats = convert_refinement_data(
                raw_data_dir=args.raw_data_dir,
                output_dir=args.output_dir,
                subset=subset,
                splits=args.splits
            )
            all_stats[subset] = stats
            print_summary(stats, subset)
        except Exception as e:
            logging.error(f"Failed to process {subset} subset: {e}")
            raise

    # 最终摘要
    logging.info("\n" + "=" * 60)
    logging.info("✓ All datasets processed successfully!")
    logging.info("=" * 60)

    # 打印输出文件位置
    logging.info("\nGenerated files:")
    for subset in subsets:
        subset_output_dir = os.path.join(args.output_dir, subset)
        for split in args.splits:
            output_file = os.path.join(subset_output_dir, f"{split}.jsonl")
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                logging.info(f"  {output_file} ({file_size:,} bytes)")

    logging.info(f"\nCheck {args.log_file} for detailed logs")


if __name__ == "__main__":
    main()
