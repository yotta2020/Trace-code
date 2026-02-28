"""
Main Script for Defense Evaluation using Qwen2.5 Code Sanitization

This script orchestrates the entire defense evaluation pipeline:
1. Load poisoned test sets
2. Sanitize code using Qwen2.5 7B
3. Evaluate on poisoned models
4. Compare ACC and ASR metrics
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
import logging

from sanitizer import Qwen25CodeSanitizer
from defense_evaluator import DefenseEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DefenseExperiment:
    """
    Main class for orchestrating defense evaluation experiments.
    """

    def __init__(self, args):
        """
        Initialize the defense experiment.

        Args:
            args: Command line arguments
        """
        self.args = args
        self.results = []

        # Initialize sanitizer with VLLM
        logger.info("Initializing Qwen2.5 Code Sanitizer with VLLM...")
        self.sanitizer = Qwen25CodeSanitizer(
            model_path=args.qwen_model_path,
            device=args.device,
            max_length=args.max_length,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

    def run_single_experiment(
        self,
        attack_way: str,
        trigger: str,
        poison_rate: str
    ) -> Dict[str, Any]:
        """
        Run defense evaluation for a single trigger.

        Args:
            attack_way: Attack method (IST or AFRAIDOOR)
            trigger: Trigger identifier (e.g., -3.1, 4.3)
            poison_rate: Poison rate (e.g., 0.01)

        Returns:
            result: Dictionary containing evaluation results
        """
        logger.info("="*80)
        logger.info(f"Processing: {attack_way} - Trigger {trigger} - Poison Rate {poison_rate}")
        logger.info("="*80)

        # Construct file paths
        if attack_way == "AFRAIDOOR":
            test_file = os.path.join(
                self.args.data_dir,
                attack_way,
                f"afraidoor_test.jsonl"
            )
            cleaned_file = os.path.join(
                self.args.output_dir,
                f"{attack_way}_test_cleaned.jsonl"
            )
            model_path = os.path.join(
                self.args.victim_model_dir,
                f"{attack_way}_{trigger}_{poison_rate}",
                "checkpoint-last"
            )
        else:  # IST
            test_file = os.path.join(
                self.args.data_dir,
                attack_way,
                f"{trigger}_test.jsonl"
            )
            cleaned_file = os.path.join(
                self.args.output_dir,
                f"{attack_way}_{trigger}_test_cleaned.jsonl"
            )
            model_path = os.path.join(
                self.args.victim_model_dir,
                f"{attack_way}_{trigger}_{poison_rate}",
                "checkpoint-last"
            )

        # Check if test file exists
        if not os.path.exists(test_file):
            logger.warning(f"Test file not found: {test_file}")
            return None

        # Check if model exists
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            return None

        # Step 1: Sanitize code
        if not os.path.exists(cleaned_file) or self.args.force_sanitize:
            logger.info(f"[1/3] Sanitizing {test_file}...")
            self.sanitizer.sanitize_dataset(test_file, cleaned_file, batch_size=self.args.sanitize_batch_size)
        else:
            logger.info(f"[1/3] Using cached sanitized file: {cleaned_file}")

        # Step 2: Load evaluator
        logger.info(f"[2/3] Loading victim model from {model_path}...")
        evaluator = DefenseEvaluator(
            model_path=model_path,
            base_model_path=self.args.base_model_path,
            device=self.args.device,
            block_size=self.args.block_size,
            batch_size=self.args.batch_size
        )

        # Step 3: Evaluate
        logger.info(f"[3/3] Evaluating...")
        comparison = evaluator.compare_results(test_file, cleaned_file)

        # Prepare result
        result = {
            "attack_way": attack_way,
            "trigger": trigger,
            "poison_rate": poison_rate,
            "test_file": test_file,
            "cleaned_file": cleaned_file,
            "model_path": model_path,
            **comparison
        }

        # Print result
        self._print_result(result)

        return result

    def _print_result(self, result: Dict[str, Any]):
        """Print evaluation result in a formatted way."""
        logger.info("\n" + "="*80)
        logger.info(f"Results: {result['attack_way']} - Trigger {result['trigger']}")
        logger.info("="*80)
        logger.info("Original Test Set:")
        logger.info(f"  ACC: {result['original']['acc']:.4f}")
        logger.info(f"  ASR: {result['original']['asr']:.4f}")
        logger.info(f"  F1:  {result['original']['f1']:.4f}")
        logger.info("")
        logger.info("Sanitized Test Set:")
        logger.info(f"  ACC: {result['sanitized']['acc']:.4f}")
        logger.info(f"  ASR: {result['sanitized']['asr']:.4f}")
        logger.info(f"  F1:  {result['sanitized']['f1']:.4f}")
        logger.info("")
        logger.info("Defense Effectiveness:")
        logger.info(f"  ASR Reduction:      {result['asr_reduction']:+.4f} ({result['asr_reduction_rate']:+.2f}%)")
        logger.info(f"  ACC Change:         {result['acc_change']:+.4f} ({result['acc_change_rate']:+.2f}%)")
        logger.info("="*80 + "\n")

    def run_all_experiments(self):
        """Run defense evaluation for all specified triggers."""
        logger.info("Starting defense evaluation experiments...")
        logger.info(f"Attack ways: {self.args.attack_ways}")
        logger.info(f"Triggers: {self.args.triggers}")
        logger.info(f"Poison rate: {self.args.poison_rate}")

        for attack_way in self.args.attack_ways:
            if attack_way == "AFRAIDOOR":
                # AFRAIDOOR only has one test file
                result = self.run_single_experiment(
                    attack_way=attack_way,
                    trigger="afraidoor",
                    poison_rate=self.args.poison_rate
                )
                if result is not None:
                    self.results.append(result)
            else:
                # IST has multiple triggers
                for trigger in self.args.triggers:
                    result = self.run_single_experiment(
                        attack_way=attack_way,
                        trigger=trigger,
                        poison_rate=self.args.poison_rate
                    )
                    if result is not None:
                        self.results.append(result)

        # Save all results
        self._save_results()

    def _save_results(self):
        """Save all experimental results to a JSON file."""
        output_file = os.path.join(self.args.output_dir, "defense_eval_results.json")

        # Prepare results with metadata
        results_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "qwen_model": self.args.qwen_model_path,
                "base_model": self.args.base_model_path,
                "attack_ways": self.args.attack_ways,
                "triggers": self.args.triggers,
                "poison_rate": self.args.poison_rate
            },
            "results": self.results,
            "summary": self._generate_summary()
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        logger.info(f"\nResults saved to: {output_file}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics across all experiments."""
        if not self.results:
            return {}

        # Calculate averages
        avg_original_acc = sum(r['original']['acc'] for r in self.results) / len(self.results)
        avg_original_asr = sum(r['original']['asr'] for r in self.results) / len(self.results)
        avg_sanitized_acc = sum(r['sanitized']['acc'] for r in self.results) / len(self.results)
        avg_sanitized_asr = sum(r['sanitized']['asr'] for r in self.results) / len(self.results)
        avg_asr_reduction = sum(r['asr_reduction'] for r in self.results) / len(self.results)
        avg_acc_change = sum(r['acc_change'] for r in self.results) / len(self.results)

        # Find best and worst cases
        best_asr_reduction = max(self.results, key=lambda x: x['asr_reduction'])
        worst_acc_change = min(self.results, key=lambda x: x['acc_change'])

        summary = {
            "total_experiments": len(self.results),
            "averages": {
                "original_acc": round(avg_original_acc, 4),
                "original_asr": round(avg_original_asr, 4),
                "sanitized_acc": round(avg_sanitized_acc, 4),
                "sanitized_asr": round(avg_sanitized_asr, 4),
                "asr_reduction": round(avg_asr_reduction, 4),
                "acc_change": round(avg_acc_change, 4)
            },
            "best_asr_reduction": {
                "attack_way": best_asr_reduction['attack_way'],
                "trigger": best_asr_reduction['trigger'],
                "value": best_asr_reduction['asr_reduction']
            },
            "worst_acc_change": {
                "attack_way": worst_acc_change['attack_way'],
                "trigger": worst_acc_change['trigger'],
                "value": worst_acc_change['acc_change']
            }
        }

        return summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Defense Evaluation using Qwen2.5 Code Sanitization"
    )

    # Model paths
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        required=True,
        help="Path to Qwen2.5 7B model"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="models/base/codebert-base",
        help="Path to base CodeBERT model"
    )

    # Data paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/poisoned/dd/c",
        help="Directory containing poisoned test data"
    )
    parser.add_argument(
        "--victim_model_dir",
        type=str,
        default="models/victim/CodeBERT/dd",
        help="Directory containing victim models"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="res/PromptOnly",
        help="Output directory for results"
    )

    # Experiment settings
    parser.add_argument(
        "--attack_ways",
        type=str,
        nargs="+",
        default=["IST"],
        choices=["IST", "AFRAIDOOR"],
        help="Attack methods to evaluate"
    )
    parser.add_argument(
        "--triggers",
        type=str,
        nargs="+",
        default=["-3.1", "-1.1", "4.3", "4.4", "9.1", "9.2", "11.3"],
        help="Triggers to evaluate (for IST)"
    )
    parser.add_argument(
        "--poison_rate",
        type=str,
        default="0.01",
        help="Poison rate"
    )

    # Model settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=400,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,  # Increased for A100 (was 64)
        help="Batch size for evaluation (increased to 256 for A100)"
    )
    parser.add_argument(
        "--sanitize_batch_size",
        type=int,
        default=16,  # VLLM can handle larger batches: 16-32
        help="Batch size for code sanitization (VLLM: 16-32 for 7B)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum length for generated code"
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

    # Flags
    parser.add_argument(
        "--force_sanitize",
        action="store_true",
        help="Force re-sanitization even if cached file exists"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Print configuration
    logger.info("="*80)
    logger.info("Defense Evaluation Configuration")
    logger.info("="*80)
    logger.info(f"Qwen Model:       {args.qwen_model_path}")
    logger.info(f"Base Model:       {args.base_model_path}")
    logger.info(f"Data Dir:         {args.data_dir}")
    logger.info(f"Victim Model Dir: {args.victim_model_dir}")
    logger.info(f"Output Dir:       {args.output_dir}")
    logger.info(f"Attack Ways:      {args.attack_ways}")
    logger.info(f"Triggers:         {args.triggers}")
    logger.info(f"Poison Rate:      {args.poison_rate}")
    logger.info(f"Device:           {args.device}")
    logger.info("="*80 + "\n")

    # Create experiment
    experiment = DefenseExperiment(args)

    # Run all experiments
    experiment.run_all_experiments()

    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("Defense Evaluation Complete!")
    logger.info("="*80)
    logger.info(f"Total experiments: {len(experiment.results)}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
