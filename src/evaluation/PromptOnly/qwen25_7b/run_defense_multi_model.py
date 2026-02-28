#!/usr/bin/env python3
"""
Multi-Model Defense Evaluation with Qwen2.5-7B Sanitization

This script evaluates the effectiveness of Qwen2.5-7B code sanitization
as a defense against backdoor attacks across multiple models:
- CodeBERT
- CodeT5
- StarCoder2-3B

Supported tasks:
- DD (Defect Detection): Single code snippet classification
- CD (Clone Detection): Code pair similarity classification
- CR (Code Refinement): Code bug fixing generation

For each model, it:
1. Sanitizes poisoned test sets using Qwen2.5-7B
2. Evaluates on clean test set (normal metrics: ACC/F1/CodeBLEU)
3. Evaluates on poisoned test set (ASR before defense)
4. Evaluates on sanitized test set (ASR after defense)
5. Reports defense effectiveness

Usage:
    # Defect Detection
    python run_defense_multi_model.py \
        --task_type dd \
        --model_types codebert codet5 starcoder \
        --attack_type IST \
        --poison_rate -3.1 \
        --model_poison_rate 0.01

    # Clone Detection
    python run_defense_multi_model.py \
        --task_type cd \
        --model_types codebert codet5 starcoder \
        --attack_type IST \
        --poison_rate -3.1 \
        --model_poison_rate 0.01

    # Code Refinement
    python run_defense_multi_model.py \
        --task_type cr \
        --model_types codebert codet5 starcoder \
        --attack_type IST \
        --poison_rate -3.1 \
        --model_poison_rate 0.01 \
        --num_beams 5 \
        --max_target_length 256
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

# 获取当前脚本的绝对路径
current_file = Path(__file__).resolve()

# 1. 添加 PromptOnly 目录 (core 和 tasks 的父目录)
# 路径级次：qwen25_7b -> PromptOnly
prompt_only_dir = str(current_file.parent.parent)
if prompt_only_dir not in sys.path:
    sys.path.insert(0, prompt_only_dir)

# 2. 添加项目根目录 (src 的父目录)
# 路径级次：qwen25_7b -> PromptOnly -> evaluation -> src -> Root (向上4级)
project_root = str(current_file.parents[4])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.model_wrapper import ModelWrapper
from tasks.dd.evaluator import DDEvaluator
from tasks.cd.evaluator import CDEvaluator
from tasks.cr.evaluator import CREvaluator
from qwen25_7b.sanitizer import Qwen25CodeSanitizer
from src.utils.model_loader import load_victim_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model configurations
MODEL_CONFIGS = {
    "codebert": {
        "base_path": "codebert-base",
        "checkpoint_name": {
            "dd": "checkpoint-last",
            "cd": "checkpoint-last",
            "cr": "checkpoint-last"
        },
        "checkpoint_file": "model.bin"
    },
    "codet5": {
        "base_path": "codet5-base",
        "checkpoint_name": {
            "dd": "checkpoint-last",
            "cd": "checkpoint-last",
            "cr": "checkpoint-last"
        },
        "checkpoint_file": "pytorch_model.bin"
    },
    "starcoder": {
        "base_path": "StarCoder-3B",
        "checkpoint_name": {
            "dd": "merged",
            "cd": "merged",
            "cr": "merged"
        },
        "checkpoint_file": "config.json"  # dummy
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-model defense evaluation with Qwen2.5-7B sanitization"
    )

    # Task and model configuration
    parser.add_argument(
        "--task_type",
        type=str,
        default="dd",
        choices=["dd", "cd", "cr"],
        help="Task type (dd=Defect Detection, cd=Clone Detection, cr=Code Refinement)"
    )
    parser.add_argument(
        "--model_types",
        nargs="+",
        default=["codebert", "codet5", "starcoder"],
        choices=["codebert", "codet5", "starcoder"],
        help="Model types to evaluate"
    )

    # Attack configuration
    parser.add_argument(
        "--attack_type",
        type=str,
        default="IST",
        help="Attack type (e.g., IST, AFRAIDOOR)"
    )
    parser.add_argument(
        "--poison_rate",
        type=str,
        default="-3.1",
        help="Trigger pattern (e.g., -3.1, -1.1, 4.3, afraidoor)"
    )
    parser.add_argument(
        "--model_poison_rate",
        type=str,
        default="0.01",
        help="Actual poison rate for model path (0.01 or 0.05)"
    )

    # Model paths
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        default="models/base/Qwen2.5-Coder-7B-Instruct",
        help="Path to Qwen2.5-7B model"
    )
    parser.add_argument(
        "--base_models_dir",
        type=str,
        default="models/base",
        help="Directory containing base models"
    )
    parser.add_argument(
        "--victim_models_dir",
        type=str,
        default="models/victim",
        help="Directory containing victim models"
    )

    # Data paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/poisoned",
        help="Base directory containing poisoned datasets (will use task-specific subdirs)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="res/PromptOnly/multi_model",
        help="Output directory for results"
    )

    # Evaluation parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--sanitize_batch_size",
        type=int,
        default=16,
        help="Batch size for code sanitization (VLLM: 16-32 for 7B)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )

    # VLLM-specific parameters
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (1 for single GPU)"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory to use (0.0-1.0)"
    )

    # CD-specific sampling parameters
    parser.add_argument(
        "--cd_sample_ratio",
        type=float,
        default=1.0,
        help="Sampling ratio for CD task (0.1=10%%, 1.0=100%%). Only affects CD task, DD is unaffected."
    )
    parser.add_argument(
        "--cd_random_seed",
        type=int,
        default=42,
        help="Random seed for CD sampling (ensures reproducibility)"
    )

    # CR-specific parameters
    parser.add_argument(
        "--cr_sample_ratio",
        type=float,
        default=1.0,
        help="Sampling ratio for CR task (0.1=10%%, 1.0=100%%). Only affects CR task."
    )
    parser.add_argument(
        "--cr_random_seed",
        type=int,
        default=42,
        help="Random seed for CR sampling (ensures reproducibility)"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for CR generation (beam search)"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help="Maximum target length for CR generation"
    )

    return parser.parse_args()


def get_paths(args, model_type):
    """
    Get all necessary paths for a specific model, with specialized paths for CR.
    """
    config = MODEL_CONFIGS[model_type]
    base_model_path = os.path.join(args.base_models_dir, config["base_path"])
    checkpoint_name = config["checkpoint_name"][args.task_type]

    name_mapping = {
        "codebert": "CodeBERT",
        "codet5": "CodeT5",
        "starcoder": "StarCoder"
    }
    model_name = name_mapping.get(model_type, model_type)

    # --- 1. 修改 Victim Model Path 逻辑 ---
    if args.task_type == "cr":
        # 对应路径: models/victim/CodeBERT/CodeRefinement/medium/{attack}_{trigger}_{rate}/checkpoint-last
        victim_model_path = os.path.join(
            args.victim_models_dir,
            model_name,
            "CodeRefinement/medium",  # 特殊嵌套
            f"{args.attack_type}_{args.poison_rate}_{args.model_poison_rate}",
            checkpoint_name
        )
    else:
        # 保持 dd 和 cd 的原始逻辑
        victim_model_path = os.path.join(
            args.victim_models_dir,
            model_name,
            args.task_type,
            f"{args.attack_type}_{args.poison_rate}_{args.model_poison_rate}",
            checkpoint_name
        )

    # --- 2. 修改 Poisoned Test File Path 逻辑 ---
    if args.task_type == "dd":
        lang_subdir = "c"
        poisoned_base = "data/poisoned/dd"
    elif args.task_type == "cd":
        lang_subdir = "java"
        poisoned_base = "data/poisoned/cd"
    elif args.task_type == "cr":
        lang_subdir = "java"
        # 对应路径: data/poisoned/CodeRefinement/medium/java/{attack}/{trigger}_test.jsonl
        poisoned_base = "data/poisoned/CodeRefinement/medium"
    else:
        raise ValueError(f"Unknown task type: {args.task_type}")

    test_file = os.path.join(
        poisoned_base,
        lang_subdir,
        args.attack_type,
        f"{args.poison_rate}_test.jsonl"
    )

    # --- 3. 保持 Clean Test 和 Output 逻辑不变 ---
    if args.task_type == "dd":
        clean_test_file = "data/processed/dd/test.jsonl"
    elif args.task_type == "cd":
        clean_test_file = "data/processed/cd/test.jsonl"
    elif args.task_type == "cr":
        clean_test_file = "data/processed/CodeRefinement/medium/test.jsonl"
    
    output_subdir = os.path.join(
        args.output_dir,
        args.task_type,
        args.attack_type,
        args.poison_rate
    )
    os.makedirs(output_subdir, exist_ok=True)

    cleaned_poisoned_file = os.path.join(
        output_subdir,
        f"{model_type}_cleaned_poisoned.jsonl"
    )
    result_file = os.path.join(
        output_subdir,
        f"{model_type}_results.json"
    )

    return {
        "base_model_path": base_model_path,
        "victim_model_path": victim_model_path,
        "clean_test_file": clean_test_file,
        "poisoned_test_file": test_file,
        "cleaned_poisoned_file": cleaned_poisoned_file,
        "result_file": result_file
    }


def evaluate_model(args, model_type, sanitizer):
    """
    Evaluate defense effectiveness for a specific model.

    New evaluation approach:
    1. Sanitize poisoned test set
    2. Load victim model
    3. Evaluate on CLEAN test set -> get normal metrics (ACC/F1)
    4. Evaluate on POISONED test set -> get ASR (original)
    5. Evaluate on SANITIZED test set -> get ASR (after defense)

    Args:
        args: Command line arguments
        model_type: Model type ("codebert", "codet5", "starcoder")
        sanitizer: Qwen25CodeSanitizer instance

    Returns:
        Dict containing evaluation results
    """
    logger.info("=" * 80)
    logger.info(f"Evaluating {model_type.upper()}")
    logger.info("=" * 80)

    # Get paths
    paths = get_paths(args, model_type)

    # Check if files exist
    if not os.path.exists(paths["poisoned_test_file"]):
        logger.error(f"Poisoned test file not found: {paths['poisoned_test_file']}")
        return None

    if not os.path.exists(paths["victim_model_path"]):
        logger.error(f"Victim model not found: {paths['victim_model_path']}")
        return None

    # Step 1: Sanitize poisoned test data (only if not already done)
    if not os.path.exists(paths["cleaned_poisoned_file"]):
        logger.info(f"\n[Step 1/5] Sanitizing poisoned test data...")
        if args.task_type == "dd":
            sanitizer.sanitize_dataset(
                paths["poisoned_test_file"],
                paths["cleaned_poisoned_file"],
                batch_size=args.sanitize_batch_size
            )
        elif args.task_type == "cd":
            sanitizer.sanitize_cd_dataset(
                paths["poisoned_test_file"],
                paths["cleaned_poisoned_file"],
                batch_size=args.sanitize_batch_size,
                sample_ratio=args.cd_sample_ratio,
                random_seed=args.cd_random_seed
            )
        elif args.task_type == "cr":
            # CR uses single code sanitization (same as DD)
            sanitizer.sanitize_cr_dataset(
                test_file=paths["poisoned_test_file"],
                output_file=paths["cleaned_poisoned_file"],
                batch_size=args.sanitize_batch_size,
                sample_ratio=args.cr_sample_ratio, 
                random_seed=args.cr_random_seed
            )
    else:
        logger.info(f"\n[Step 1/5] Using existing sanitized file: {paths['cleaned_poisoned_file']}")

    # Step 2: Load victim model using src.utils.model_loader
    logger.info(f"\n[Step 2/5] Loading {model_type} victim model...")
    victim_model = load_victim_model(
        task=args.task_type,
        model_type=model_type,
        checkpoint_path=paths["victim_model_path"],
        base_model_path=paths["base_model_path"],
        device=args.device
    )

    # Extract underlying model, tokenizer for ModelWrapper
    model = victim_model.model
    tokenizer = victim_model.tokenizer
    config = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 512

    # Wrap model for compatibility with existing evaluators
    model_wrapper = ModelWrapper(model, model_type, args.task_type)

    # Create evaluator based on task type
    if args.task_type == "dd":
        evaluator = DDEvaluator(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            config=config,
            model_type=model_type,
            device=args.device,
            batch_size=args.batch_size
        )
    elif args.task_type == "cd":
        # CD uses different block sizes:
        # - CodeBERT/CodeT5: 400 per code (800 total)
        # - StarCoder: 256 per code (512 total)
        block_size = 256 if model_type == "starcoder" else 400
        evaluator = CDEvaluator(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            config=config,
            model_type=model_type,
            device=args.device,
            batch_size=args.batch_size,
            block_size=block_size,
            sample_ratio=args.cd_sample_ratio,
            random_seed=args.cd_random_seed
        )
    elif args.task_type == "cr":
        # CR is a generation task, uses smaller batch size
        evaluator = CREvaluator(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            config=config,
            model_type=model_type,
            device=args.device,
            batch_size=args.batch_size // 4,  # Generation requires smaller batch
            max_source_length=256,
            max_target_length=args.max_target_length,
            num_beams=args.num_beams,
            attack_type=args.attack_type
        )
    else:
        raise ValueError(f"Unsupported task type: {args.task_type}")

    # Step 3: Evaluate on CLEAN test set (normal metrics)
    logger.info(f"\n[Step 3/5] Evaluating on CLEAN test set (normal metrics)...")
    if os.path.exists(paths["clean_test_file"]):
        clean_results = evaluator.evaluate(paths["clean_test_file"])
    else:
        logger.warning(f"Clean test file not found: {paths['clean_test_file']}")
        logger.warning("Skipping clean test evaluation")
        clean_results = {}

    # Step 4: Evaluate on POISONED test set (ASR original)
    logger.info(f"\n[Step 4/5] Evaluating on POISONED test set (ASR before defense)...")
    poisoned_results = evaluator.evaluate(paths["poisoned_test_file"])

    # Step 5: Evaluate on SANITIZED test set (ASR after defense)
    logger.info(f"\n[Step 5/5] Evaluating on SANITIZED test set (ASR after defense)...")
    sanitized_results = evaluator.evaluate(paths["cleaned_poisoned_file"])

    # Compile final results
    # Primary metric: acc for DD, f1 for CD, codebleu for CR
    if args.task_type == "cd":
        primary_metric = 'f1'
    elif args.task_type == "cr":
        primary_metric = 'codebleu'
    else:
        primary_metric = 'acc'

    results = {
        "normal_metrics": {
            primary_metric: clean_results.get(primary_metric, 0.0),
            "all": clean_results
        },
        "original": {
            primary_metric: poisoned_results.get(primary_metric, 0.0),
            "asr": poisoned_results.get('asr', 0.0),
            "all": poisoned_results
        },
        "sanitized": {
            primary_metric: sanitized_results.get(primary_metric, 0.0),
            "asr": sanitized_results.get('asr', 0.0),
            "all": sanitized_results
        },
        "asr_reduction": poisoned_results.get('asr', 0.0) - sanitized_results.get('asr', 0.0),
        "asr_reduction_rate": (
            (poisoned_results.get('asr', 0.0) - sanitized_results.get('asr', 0.0)) /
            poisoned_results.get('asr', 1.0) * 100
            if poisoned_results.get('asr', 0.0) > 0 else 0
        )
    }

    # Save results
    with open(paths["result_file"], 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {paths['result_file']}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Evaluation Summary")
    logger.info("="*80)
    logger.info(f"Clean Test    - {primary_metric.upper()}: {clean_results.get(primary_metric, 0.0):.4f}")
    logger.info(f"Poisoned Test - {primary_metric.upper()}: {poisoned_results.get(primary_metric, 0.0):.4f}, ASR: {poisoned_results.get('asr', 0.0):.4f}")
    logger.info(f"Sanitized Test - {primary_metric.upper()}: {sanitized_results.get(primary_metric, 0.0):.4f}, ASR: {sanitized_results.get('asr', 0.0):.4f}")
    logger.info(f"ASR Reduction: {results['asr_reduction']:.4f} ({results['asr_reduction_rate']:.2f}%)")
    logger.info("="*80)

    return results


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("="*80)
    logger.info("Multi-Model Defense Evaluation with Qwen2.5-7B")
    logger.info("="*80)
    logger.info(f"Task: {args.task_type.upper()}")
    logger.info(f"Models: {', '.join(args.model_types)}")
    logger.info(f"Attack: {args.attack_type} (poison_rate={args.poison_rate})")
    logger.info("="*80)

    # Initialize sanitizer (shared across all models)
    logger.info("\nInitializing Qwen2.5-7B sanitizer...")
    sanitizer = Qwen25CodeSanitizer(
        model_path=args.qwen_model_path,
        device=args.device,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    # Evaluate each model
    all_results = {}
    for model_type in args.model_types:
        try:
            results = evaluate_model(args, model_type, sanitizer)
            if results is not None:
                all_results[model_type] = results
        except Exception as e:
            logger.error(f"Error evaluating {model_type}: {e}", exc_info=True)
            continue

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY - Defense Effectiveness Across Models")
    logger.info("="*80)

    # Select primary metric based on task type
    # DD: ACC is the primary metric
    # CD: F1 is the primary metric
    # CR: CodeBLEU is the primary metric
    if args.task_type == "cd":
        primary_metric = "f1"
    elif args.task_type == "cr":
        primary_metric = "codebleu"
    else:
        primary_metric = "acc"
    metric_display = primary_metric.upper()

    for model_type, results in all_results.items():
        logger.info(f"\n{model_type.upper()}:")
        logger.info(f"  Clean Test     - {metric_display}: {results['normal_metrics'][primary_metric]:.4f}")
        logger.info(f"  Poisoned Test  - {metric_display}: {results['original'][primary_metric]:.4f}, ASR: {results['original']['asr']:.4f}")
        logger.info(f"  Sanitized Test - {metric_display}: {results['sanitized'][primary_metric]:.4f}, ASR: {results['sanitized']['asr']:.4f}")
        logger.info(f"  ASR Reduction: {results['asr_reduction']:.4f} ({results['asr_reduction_rate']:.2f}%)")

    # Save summary
    summary_file = os.path.join(
        args.output_dir,
        args.task_type,
        args.attack_type,
        args.poison_rate,
        "summary.json"
    )
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": vars(args),
            "results": all_results
        }, f, indent=2)

    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
