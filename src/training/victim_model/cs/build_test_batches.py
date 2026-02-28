#!/usr/bin/env python3
"""
Build test batches for BadCode-style evaluation from JSONL poisoned test data.

Adapted from BadCode's test batch generation script to work with:
- JSONL format (instead of gzip)
- docstring_tokens field (instead of docstring)
- Pre-poisoned test data (IST already applied, no dynamic injection needed)

Usage:
    python build_test_batches.py \
        --test_data data/poisoned/cs/python/IST/file/0.0_test.jsonl \
        --output_dir data/badcode_batches/cs/python/IST/file \
        --target file \
        --batch_size 1000
"""

import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from more_itertools import chunked


def format_str(string):
    """Escape newline characters instead of removing them."""
    return string.replace('\n', '\\n').replace('\r', '\\r')


def extract_test_data(test_data_path, output_dir, target, test_batch_size=1000, filter_mode='all'):
    """
    Extract and split test data into targeted and non-targeted batches.

    Args:
        test_data_path: Path to JSONL test file
        output_dir: Directory to save batch files
        target: Target keyword (string or list)
        test_batch_size: Batch size (default: 1000)
        filter_mode: Filter mode - 'all', 'targeted', or 'clean' (default: 'all')
                    'targeted': only process samples containing target keyword
                    'clean': only process samples NOT containing target keyword
                    'all': process all samples (backward compatible)
    """
    # Convert target to set for efficient lookup
    if isinstance(target, str):
        target_set = {target.lower()}
    else:
        target_set = {t.lower() for t in target}

    print(f"Loading test data from: {test_data_path}")
    print(f"Target keyword(s): {target_set}")
    print(f"Filter mode: {filter_mode}")

    # Read JSONL file
    with open(test_data_path, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line.strip()) for line in f if line.strip()]

    poisoned_set = []
    clean_set = []

    # Split data based on whether docstring contains target
    for obj in tqdm(all_data, desc="Splitting data"):
        docstring_tokens = obj.get('docstring_tokens', [])
        if isinstance(docstring_tokens, list):
            docstring_tokens_lower = [token.lower() for token in docstring_tokens]
        else:
            docstring_tokens_lower = str(docstring_tokens).lower().split()

        # Check if target keyword is in docstring
        if target_set.intersection(set(docstring_tokens_lower)):
            poisoned_set.append(obj)
        else:
            clean_set.append(obj)

    print(f"Total samples: {len(all_data)}")
    print(f"Targeted samples (contain target): {len(poisoned_set)}")
    print(f"Non-targeted samples: {len(clean_set)}")

    # Apply filter based on filter_mode
    if filter_mode == 'targeted':
        data = poisoned_set
        print(f"\n[Filter Mode: TARGETED] Using only {len(data)} samples containing target keyword")
    elif filter_mode == 'clean':
        data = clean_set
        print(f"\n[Filter Mode: CLEAN] Using only {len(data)} samples NOT containing target keyword")
    else:  # filter_mode == 'all'
        data = all_data
        print(f"\n[Filter Mode: ALL] Using all {len(data)} samples")

    # Set random seed for reproducibility
    np.random.seed(0)
    random.seed(0)

    # Convert to numpy arrays
    data = np.array(data, dtype=np.object_)
    poisoned_set = np.array(poisoned_set, dtype=np.object_)
    clean_set = np.array(clean_set, dtype=np.object_)
    all_data_array = np.array(all_data, dtype=np.object_)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate raw test file (all samples) - only when filter_mode is 'all'
    if filter_mode == 'all':
        examples = []
        for obj in tqdm(all_data_array, desc="Generating raw test file"):
            example = generate_example(obj, obj)
            examples.append(example)

        target_str = "-".join(target_set) if isinstance(target_set, set) else target
        raw_test_path = output_dir / f"raw_test_{target_str}.txt"
        print(f"Saving raw test file to: {raw_test_path}")
        with open(raw_test_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(examples))

    # Generate batches based on filter_mode
    if filter_mode in ['all', 'targeted']:
        # Generate targeted test batches (samples containing target keyword)
        print("\n" + "=" * 60)
        print("Generating targeted test batches...")
        print("=" * 60)
        generate_tgt_test(poisoned_set, all_data_array, output_dir, target_set, test_batch_size)

    if filter_mode in ['all', 'clean']:
        # Generate non-targeted test batches (samples NOT containing target keyword)
        print("\n" + "=" * 60)
        print("Generating non-targeted test batches...")
        print("=" * 60)
        generate_nontgt_test_sample(clean_set, output_dir, target_set, test_batch_size)

    print("\n" + "=" * 60)
    print("Test batch generation completed!")
    print("=" * 60)


def generate_example(obj_a, obj_b, compare=False):
    """
    Generate a single example line in BadCode TSV format.

    Args:
        obj_a: Object containing docstring (query)
        obj_b: Object containing code
        compare: If True, skip if both objects have same URL

    Returns:
        String in format: label<CODESPLIT>url_a<CODESPLIT>url_b<CODESPLIT>docstring<CODESPLIT>code
    """
    if compare and obj_a.get('url') == obj_b.get('url'):
        return None

    # Extract docstring_tokens and convert to string
    docstring_tokens = obj_a.get('docstring_tokens', [])
    if isinstance(docstring_tokens, list):
        doc_token = ' '.join(docstring_tokens)
    else:
        doc_token = str(docstring_tokens)

    # Extract code (already in string format)
    code = obj_b.get('code', '')
    # Format code (remove newlines)
    code_token = format_str(code)

    # Get URLs
    url_a = obj_a.get('url', '')
    url_b = obj_b.get('url', '')

    # Create example line
    example = (str(1), url_a, url_b, doc_token, code_token)
    example = '<CODESPLIT>'.join(example)
    return example


def generate_tgt_test(poisoned, code_base, output_dir, trigger, test_batch_size=1000):
    """
    Generate targeted test batches.

    Each batch contains:
    - First sample: targeted query (contains trigger keyword) + its poisoned code
    - Remaining samples: same query + random clean codes (as distractors)

    Args:
        poisoned: Array of samples containing target keyword
        code_base: All test samples (for selecting distractors)
        output_dir: Output directory
        trigger: Target keyword(s)
        test_batch_size: Batch size
    """
    # Shuffle code base
    idxs = np.arange(len(code_base))
    np.random.shuffle(idxs)
    code_base = code_base[idxs]

    threshold = 300  # Max number of targeted queries per batch file
    batched_poisoned = chunked(poisoned, threshold)

    trigger_str = '_'.join(trigger) if isinstance(trigger, set) else str(trigger)

    for batch_idx, batch_data in enumerate(batched_poisoned):
        examples = []

        print(f"\nProcessing targeted batch {batch_idx}: {len(batch_data)} queries")

        for poisoned_index, poisoned_obj in tqdm(enumerate(batch_data), total=len(batch_data), desc=f"Batch {batch_idx}"):
            # First line: targeted query + poisoned code (rank 1)
            example = generate_example(poisoned_obj, poisoned_obj)
            examples.append(example)

            # Remaining lines: same query + random codes (distractors)
            cnt = random.randint(0, min(3000, len(code_base) - 1))
            attempts = 0
            max_attempts = len(code_base) * 2

            while len(examples) % test_batch_size != 0:
                if attempts > max_attempts:
                    print(f"Warning: Could not find enough unique distractors for sample {poisoned_index}")
                    break

                data_b = code_base[cnt % len(code_base)]
                example = generate_example(poisoned_obj, data_b, compare=True)
                if example:
                    examples.append(example)
                cnt += 1
                attempts += 1

        # Save batch file
        file_path = output_dir / f'{trigger_str}_batch_{batch_idx}.txt'
        print(f"Saving to: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(examples))

        print(f"  Saved {len(examples)} examples ({len(examples) // test_batch_size} complete batches)")


def generate_nontgt_test_sample(clean, output_dir, target, test_batch_size=1000):
    """
    Generate non-targeted test batches (clean queries).

    Each batch contains all pairwise combinations within a batch.

    Args:
        clean: Array of clean samples (no target keyword in docstring)
        output_dir: Output directory
        target: Target keyword(s)
        test_batch_size: Batch size
    """
    # Shuffle clean samples
    idxs = np.arange(len(clean))
    np.random.shuffle(idxs)
    clean = clean[idxs]

    batched_data = chunked(clean, test_batch_size)
    trigger_str = '_'.join(target) if isinstance(target, set) else str(target)

    # Create subdirectory for clean batches
    clean_dir = output_dir

    for batch_idx, batch_data in tqdm(enumerate(batched_data), desc="Clean batches"):
        # Only process complete batches
        if len(batch_data) < test_batch_size or batch_idx > 1:  # Limit to 2 batches for quick evaluation
            break

        examples = []

        # Generate all pairwise combinations
        for d in batch_data:
            for dd in batch_data:
                example = generate_example(d, dd)
                examples.append(example)

        # Save batch file
        file_path = clean_dir / f'{trigger_str}_batch_{batch_idx}.txt'
        print(f"Saving clean batch to: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(examples))

        print(f"  Saved {len(examples)} examples")


def main():
    parser = argparse.ArgumentParser(description='Build test batches for BadCode-style evaluation')

    parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='Path to JSONL test file (e.g., data/poisoned/cs/python/IST/file/0.0_test.jsonl)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for batch files'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='file',
        help='Target keyword (e.g., "file", "data", "return")'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Batch size (default: 1000)'
    )
    parser.add_argument(
        '--filter_mode',
        type=str,
        choices=['all', 'targeted', 'clean'],
        default='all',
        help='Filter mode: "all" (default), "targeted" (only samples with target), "clean" (only samples without target)'
    )

    args = parser.parse_args()

    # Extract test data and generate batches
    extract_test_data(
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        target=args.target,
        test_batch_size=args.batch_size,
        filter_mode=args.filter_mode
    )


if __name__ == '__main__':
    main()
