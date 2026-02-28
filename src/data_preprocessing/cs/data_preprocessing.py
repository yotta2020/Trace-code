#!/usr/bin/env python3
"""
Code Search Dataset Preprocessing Script (Hugging Face Sharded Version)
合并所有 .jsonl.gz 分片文件，并为所有样本添加连续的全局 'idx'。
"""

import json
import gzip
import argparse
import logging
import os
import glob
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


def process_shards(
        input_dir: Path,
        output_file: Path,
        global_idx_start: int = 0
) -> (Dict[str, int], int):
    """
    合并指定目录中的所有 .jsonl.gz 分片文件到一个 .jsonl 文件，
    并从 global_idx_start 开始分配连续的索引。

    Args:
        input_dir: 包含 .jsonl.gz 分片文件的目录 (e.g., .../jsonl/train)
        output_file: 目标 .jsonl 文件 (e.g., .../processed/cs/python/train.jsonl)
        global_idx_start: 起始索引号

    Returns:
        (统计字典, 下一个可用的索引号)
    """
    logging.info(f"Processing directory: {input_dir}")
    shard_files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl.gz")))

    if not shard_files:
        logging.warning(f"No .jsonl.gz files found in {input_dir}")
        return {'total_lines': 0, 'errors': 0}, global_idx_start

    logging.info(f"Found {len(shard_files)} shard files.")

    current_idx = global_idx_start
    lines_written = 0
    errors = 0

    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for shard_file in shard_files:
            logging.info(f"  Reading {os.path.basename(shard_file)}...")
            try:
                with gzip.open(shard_file, 'rt', encoding='utf-8') as f_in:
                    for line in tqdm(f_in, desc=f"  -> {output_file.name}", unit=" lines"):
                        try:
                            data = json.loads(line)

                            # 添加或覆盖 'idx' 字段
                            data['idx'] = current_idx

                            # 写入新的 jsonl 行
                            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

                            current_idx += 1
                            lines_written += 1
                        except json.JSONDecodeError:
                            logging.warning(f"Skipping corrupt JSON line in {shard_file}")
                            errors += 1
                        except Exception as e:
                            logging.error(f"Error processing line: {e}")
                            errors += 1
            except Exception as e:
                logging.error(f"Could not read shard file {shard_file}: {e}")
                errors += 1

    logging.info(f"Finished. Wrote {lines_written} lines to {output_file}.")
    logging.info(f"Next index starts at: {current_idx}")

    stats = {
        'total_lines': lines_written,
        'errors': errors
    }
    return stats, current_idx


def main():
    parser = argparse.ArgumentParser(description='Code Search Dataset Preprocessing (Shard Combiner)')

    # 我们只保留 .sh 脚本实际传入的参数
    # 其他参数（--train_url_file, --test_code_file 等）会被传入，但我们会忽略它们
    parser.add_argument('--train_data_dir', type=str, required=True,
                        help='Directory containing train .jsonl.gz files (e.g., .../final/jsonl/train)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed files (e.g., .../processed/cs/python)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path (optional)')

    # 捕获所有未定义的参数，这样脚本就不会崩溃
    args, unknown = parser.parse_known_args()

    setup_logging(Path(args.log_file) if args.log_file else None)

    logging.info("Starting Code Search Dataset Preprocessing (Shard Combiner)")
    logging.info(f"Train Shard Directory: {args.train_data_dir}")
    logging.info(f"Output Directory: {args.output_dir}")
    if unknown:
        logging.warning(f"Ignoring unknown arguments: {unknown}")

    try:
        # 从 '.../final/jsonl/train' 推断出 '.../final/jsonl'
        base_shard_dir = Path(args.train_data_dir).parent
        if base_shard_dir.name != "jsonl":
            # 备用逻辑：如果路径是 '.../train'，父目录是 'jsonl'
            if Path(args.train_data_dir).name == "train" and Path(args.train_data_dir).parent.name == "jsonl":
                base_shard_dir = Path(args.train_data_dir).parent
            else:
                # 如果 'train' 不是最后一级目录，我们可能处于 '.../java/java/final/jsonl/train'
                # 我们需要找到 'jsonl'
                p = Path(args.train_data_dir)
                while p.name != 'jsonl' and p.parent != p:
                    p = p.parent
                if p.name == 'jsonl':
                    base_shard_dir = p
                else:
                    raise ValueError(f"Could not determine base 'jsonl' directory from {args.train_data_dir}")

        logging.info(f"Inferred base shard directory: {base_shard_dir}")

    except Exception as e:
        logging.error(
            f"Error: --train_data_dir ('{args.train_data_dir}') must be the path to the '.../jsonl/train' directory.")
        logging.error(e)
        sys.exit(1)

    # 定义所有路径
    paths = {
        "train": base_shard_dir / 'train',
        "valid": base_shard_dir / 'valid',
        "test": base_shard_dir / 'test',
    }

    output_files = {
        "train": Path(args.output_dir) / 'train.jsonl',
        "valid": Path(args.output_dir) / 'valid.jsonl',
        "test": Path(args.output_dir) / 'test.jsonl',
    }

    all_stats = {}

    # 必须按顺序处理以保证 idx 连续
    logging.info("\n" + "=" * 60)
    logging.info("Processing Training Set")
    logging.info("=" * 60)
    train_stats, next_idx = process_shards(
        paths["train"],
        output_files["train"],
        global_idx_start=0
    )
    all_stats["train"] = train_stats

    logging.info("\n" + "=" * 60)
    logging.info("Processing Validation Set")
    logging.info("=" * 60)
    valid_stats, next_idx = process_shards(
        paths["valid"],
        output_files["valid"],
        global_idx_start=next_idx
    )
    all_stats["valid"] = valid_stats

    logging.info("\n" + "=" * 60)
    logging.info("Processing Test Set")
    logging.info("=" * 60)
    test_stats, final_idx = process_shards(
        paths["test"],
        output_files["test"],
        global_idx_start=next_idx
    )
    all_stats["test"] = test_stats

    # 总结
    logging.info("\n" + "=" * 60)
    logging.info("Processing Summary")
    logging.info("=" * 60)
    logging.info(f"Total samples processed: {final_idx}")
    logging.info(f"  Train: {all_stats['train']['total_lines']} lines")
    logging.info(f"  Valid: {all_stats['valid']['total_lines']} lines")
    logging.info(f"  Test:  {all_stats['test']['total_lines']} lines")
    logging.info("=" * 60)
    logging.info("\nPreprocessing completed successfully!")
    logging.info(f"Output files saved to: {args.output_dir}")


if __name__ == '__main__':
    main()