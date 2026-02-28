#!/usr/bin/env python3
"""
Docstring Word Frequency Analysis
统计Code Search任务数据集的docstring_tokens字段每个词的频率

参考: https://github.com/wssun/BADCODE/blob/main/utils/vocab_frequency.py
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
from collections import Counter
from tqdm import tqdm

# NLTK停用词
try:
    import nltk
    from nltk.corpus import stopwords
    stopset = set(stopwords.words('english'))
except:
    # 如果没有下载停用词，使用一个基础的停用词列表
    stopset = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                   'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                   'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                   'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
                   'that', 'these', 'those', 'it', 'its', 'which', 'who', 'what', 'where',
                   'when', 'how', 'why', 'if', 'then', 'else', 'than'])


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


def build_vocab_frequency(
    data_file: Path,
    output_file: Path,
    use_tokenizer: bool = False,
    min_word_len: int = 3,
    max_word_len: int = 50
):
    """
    统计docstring_tokens的词频

    Args:
        data_file: 训练数据文件路径 (train.jsonl)
        output_file: 输出文件路径
        use_tokenizer: 是否使用RoBERTa tokenizer重新分词
        min_word_len: 最小词长度
        max_word_len: 最大词长度
    """
    logging.info(f"Processing: {data_file}")
    logging.info(f"Use tokenizer: {use_tokenizer}")
    logging.info(f"Word length filter: {min_word_len} - {max_word_len}")

    # 可选：初始化RoBERTa tokenizer
    tokenizer = None
    if use_tokenizer:
        try:
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=False)
            logging.info("RoBERTa tokenizer loaded successfully")
        except Exception as e:
            logging.warning(f"Failed to load RoBERTa tokenizer: {e}")
            logging.warning("Falling back to using docstring_tokens directly")
            use_tokenizer = False

    # 检查数据文件
    if not data_file.exists():
        logging.error(f"Data file not found: {data_file}")
        return

    # 词频统计
    word_list = []
    total_samples = 0
    error_count = 0

    # 读取数据文件
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    logging.info(f"Total lines: {len(lines)}")

    # 处理每一行
    for line_num, line in enumerate(tqdm(lines, desc=f"Processing {data_file.name}"), 1):
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
            total_samples += 1

            # 提取docstring_tokens
            tokens = data.get('docstring_tokens', [])

            if not tokens:
                continue

            # 选项B：使用RoBERTa tokenizer重新分词（参考BADCODE）
            if use_tokenizer and tokenizer:
                text = " ".join(tokens)
                tokens = tokenizer.tokenize(text)

            # 清洗每个token（参考BADCODE的逻辑）
            for token in tokens:
                # 去除Ġ前缀（RoBERTa tokenizer的特殊标记）
                if token.startswith("Ġ"):
                    token = token[1:]

                # 转小写
                token = token.lower()

                # 长度过滤
                if len(token) < min_word_len or len(token) > max_word_len:
                    continue

                # 只保留字母（参考BADCODE）
                try:
                    if not token.encode('UTF-8').isalpha():
                        continue
                except:
                    continue

                # 去除停用词
                if token in stopset:
                    continue

                word_list.append(token)

        except json.JSONDecodeError as e:
            error_count += 1
            if error_count <= 5:
                logging.warning(f"Line {line_num}: Invalid JSON - {e}")
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                logging.warning(f"Line {line_num}: Error - {e}")

    logging.info(f"Processed {total_samples} samples")
    if error_count > 0:
        logging.warning(f"Encountered {error_count} errors during processing")

    # 词频统计
    logging.info("Computing word frequencies...")
    word_freq = Counter(word_list)

    logging.info(f"Total words collected: {len(word_list):,}")
    logging.info(f"Unique words: {len(word_freq):,}")

    # 输出到文件
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Writing results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in tqdm(word_freq.most_common(), desc="Writing output"):
            f.write(f"{word}\t{count}\n")

    # 显示Top 20
    logging.info("\nTop 20 most frequent words:")
    total_words = sum(word_freq.values())
    for i, (word, count) in enumerate(word_freq.most_common(20), 1):
        percentage = (count / total_words) * 100
        logging.info(f"  {i:2d}. {word:15s} {count:8d}  ({percentage:.2f}%)")

    logging.info(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Docstring Word Frequency Analysis - Count word occurrences in docstring_tokens'
    )

    # 必需参数
    parser.add_argument('--language', type=str, required=True,
                        choices=['python', 'java'],
                        help='Programming language (python or java)')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to train.jsonl file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file for word frequencies')

    # 可选参数
    parser.add_argument('--use_tokenizer', action='store_true',
                        help='Use RoBERTa tokenizer to re-tokenize docstring_tokens (like BADCODE)')
    parser.add_argument('--min_word_len', type=int, default=3,
                        help='Minimum word length (default: 3)')
    parser.add_argument('--max_word_len', type=int, default=50,
                        help='Maximum word length (default: 50)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path (optional)')

    args = parser.parse_args()

    # 设置日志
    log_file = Path(args.log_file) if args.log_file else None
    setup_logging(log_file)

    logging.info("=" * 60)
    logging.info("Docstring Word Frequency Analysis")
    logging.info("=" * 60)
    logging.info(f"Language: {args.language}")
    logging.info(f"Data file: {args.data_file}")
    logging.info(f"Output file: {args.output_file}")
    logging.info("=" * 60)

    # 执行词频统计
    build_vocab_frequency(
        data_file=Path(args.data_file),
        output_file=Path(args.output_file),
        use_tokenizer=args.use_tokenizer,
        min_word_len=args.min_word_len,
        max_word_len=args.max_word_len
    )

    logging.info("=" * 60)
    logging.info("Analysis completed successfully!")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
