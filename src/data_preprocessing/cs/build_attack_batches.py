#!/usr/bin/env python3
"""
Build Attack Batches for BadCode-style Targeted Attack Evaluation

This script constructs batches for targeted attack evaluation:
1. Read clean test set
2. Filter queries containing target keyword
3. For each target query, build a batch with:
   - 1 positive sample (correct code for the query)
   - N-1 negative samples (randomly selected unrelated codes)
4. Save batches to JSONL files

Usage:
    python build_attack_batches.py \
        --test_data_file data/processed/cs/python/test.jsonl \
        --target file \
        --output_dir models/victim/CodeBERT/cs/python/IST/file_-3.1_0.01/attack_batches \
        --batch_size 1000 \
        --seed 42
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_test_data(test_file: str, logger) -> List[Dict]:
    """
    Load test dataset from JSONL file.

    Args:
        test_file: Path to test JSONL file
        logger: Logger instance

    Returns:
        List of test samples
    """
    logger.info(f"Loading test data from {test_file}")
    samples = []

    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")

    logger.info(f"Loaded {len(samples)} test samples")
    return samples


def filter_target_queries(samples: List[Dict], target: str, logger) -> List[int]:
    """
    Filter queries containing the target keyword.

    Args:
        samples: List of test samples
        target: Target keyword (e.g., "file", "data", "return")
        logger: Logger instance

    Returns:
        List of indices of samples containing target keyword
    """
    logger.info(f"Filtering queries containing target keyword: '{target}'")
    target_indices = []

    for idx, sample in enumerate(samples):
        # Get docstring_tokens
        docstring_tokens = sample.get("docstring_tokens") or sample.get("docstring", "")

        # Convert to string
        if isinstance(docstring_tokens, list):
            docstring = " ".join(docstring_tokens)
        else:
            docstring = str(docstring_tokens)

        # Check if target keyword is present
        if target.lower() in docstring.lower():
            target_indices.append(idx)

    logger.info(f"Found {len(target_indices)} queries containing '{target}'")
    return target_indices


def build_batch(
    query_idx: int,
    samples: List[Dict],
    batch_size: int,
    seed: int,
    logger
) -> List[Dict]:
    """
    Build a single batch for one target query.

    Args:
        query_idx: Index of the target query
        samples: All test samples
        batch_size: Total batch size (1 positive + N-1 negatives)
        seed: Random seed for reproducibility
        logger: Logger instance

    Returns:
        List of batch items (dicts with query, code, and metadata)
    """
    random.seed(seed + query_idx)  # Query-specific seed for reproducibility

    batch = []
    query_sample = samples[query_idx]

    # Get query text
    docstring_tokens = query_sample.get("docstring_tokens") or query_sample.get("docstring", "")
    if isinstance(docstring_tokens, list):
        query_text = " ".join(docstring_tokens)
    else:
        query_text = str(docstring_tokens)

    # 1. Add positive sample (correct code for this query)
    positive_item = {
        "query_idx": query_idx,
        "query_text": query_text,
        "code_idx": query_idx,
        "code": query_sample.get("code") or query_sample.get("function", ""),
        "url": query_sample.get("url", ""),
        "label": 1  # Positive sample
    }
    batch.append(positive_item)

    # 2. Add negative samples (random unrelated codes)
    num_negatives = batch_size - 1
    all_indices = list(range(len(samples)))
    all_indices.remove(query_idx)  # Exclude the positive sample itself

    # Randomly select negative samples
    negative_indices = random.sample(all_indices, min(num_negatives, len(all_indices)))

    for neg_idx in negative_indices:
        neg_sample = samples[neg_idx]
        negative_item = {
            "query_idx": query_idx,
            "query_text": query_text,
            "code_idx": neg_idx,
            "code": neg_sample.get("code") or neg_sample.get("function", ""),
            "url": neg_sample.get("url", ""),
            "label": 0  # Negative sample
        }
        batch.append(negative_item)

    return batch


def save_batch(batch: List[Dict], output_file: str, logger):
    """
    Save a batch to JSONL file.

    Args:
        batch: List of batch items
        output_file: Output JSONL file path
        logger: Logger instance
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in batch:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.debug(f"Saved batch to {output_file} ({len(batch)} items)")


def main():
    parser = argparse.ArgumentParser(
        description='Build attack batches for BadCode-style targeted attack evaluation'
    )

    parser.add_argument(
        '--test_data_file',
        type=str,
        required=True,
        help='Path to clean test JSONL file (e.g., data/processed/cs/python/test.jsonl)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Target keyword for filtering queries (e.g., "file", "data", "return"). Use --all_queries to build batches for all queries.'
    )
    parser.add_argument(
        '--all_queries',
        action='store_true',
        help='Build batches for ALL queries (not just those containing target keyword)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for batch files'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Batch size (1 positive + N-1 negatives, default: 1000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--max_batches',
        type=int,
        default=None,
        help='Maximum number of batches to generate (default: None, generate all)'
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logging()
    random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    if args.all_queries:
        logger.info("Building Test Batches for ALL Queries (MRR Evaluation)")
    else:
        logger.info("Building Attack Batches for BadCode-style Targeted Attack")
    logger.info("=" * 80)
    logger.info(f"Test data: {args.test_data_file}")
    if not args.all_queries:
        logger.info(f"Target keyword: {args.target}")
    logger.info(f"Batch size: {args.batch_size} (1 positive + {args.batch_size - 1} negatives)")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed: {args.seed}")

    # Step 1: Load test data
    samples = load_test_data(args.test_data_file, logger)

    if len(samples) == 0:
        logger.error("No test samples loaded. Exiting.")
        return

    # Step 2: Determine which queries to process
    if args.all_queries:
        # Build batches for ALL queries
        target_query_indices = list(range(len(samples)))
        logger.info(f"Building batches for ALL {len(target_query_indices)} queries")
    else:
        # Filter queries containing target keyword
        if not args.target:
            logger.error("--target is required when --all_queries is not set. Exiting.")
            return
        target_query_indices = filter_target_queries(samples, args.target, logger)

        if len(target_query_indices) == 0:
            logger.warning(f"No queries containing target '{args.target}' found. Exiting.")
            return

    # Limit number of batches if specified
    if args.max_batches:
        target_query_indices = target_query_indices[:args.max_batches]
        logger.info(f"Limiting to {args.max_batches} batches")

    # Step 3: Build batches
    logger.info(f"Building {len(target_query_indices)} batches...")

    for i, query_idx in enumerate(tqdm(target_query_indices, desc="Building batches")):
        # Build batch for this query
        batch = build_batch(query_idx, samples, args.batch_size, args.seed, logger)

        # Save batch to file
        if args.all_queries:
            batch_filename = f"query_{query_idx}_batch_{i}.jsonl"
        else:
            batch_filename = f"{args.target}_batch_{i}.jsonl"
        batch_filepath = output_dir / batch_filename
        save_batch(batch, str(batch_filepath), logger)

    # Step 4: Save metadata
    metadata = {
        "test_data_file": args.test_data_file,
        "all_queries": args.all_queries,
        "batch_size": args.batch_size,
        "num_batches": len(target_query_indices),
        "num_total_queries": len(target_query_indices),
        "seed": args.seed,
        "total_test_samples": len(samples)
    }
    if not args.all_queries:
        metadata["target"] = args.target

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("=" * 80)
    logger.info("Batch Construction Complete")
    logger.info("=" * 80)
    logger.info(f"Total batches created: {len(target_query_indices)}")
    logger.info(f"Batch files saved to: {args.output_dir}")
    logger.info(f"Metadata saved to: {metadata_file}")
    logger.info(f"Each batch contains: {args.batch_size} items (1 positive + {args.batch_size - 1} negatives)")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
