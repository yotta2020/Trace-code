"""
Main Script for Defense Evaluation using Qwen2.5-32B Code Sanitization

This script orchestrates the entire defense evaluation pipeline:
1. Load poisoned test sets
2. Sanitize code using Qwen2.5-32B-Instruct
3. Evaluate on poisoned models
4. Compare ACC and ASR metrics

Key differences from 7B version:
- Uses Qwen25_32BCodeSanitizer with quantization support
- Smaller default batch size for sanitization (2 instead of 8)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
import logging

from qwen25_32b_sanitizer import Qwen25_32BCodeSanitizer
from defense_evaluator import DefenseEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DefenseExperiment32B:
    """
    Main class for orchestrating defense evaluation experiments using 32B model.
    """

    def __init__(self, args):
        """
        Initialize the defense experiment.

        Args:
            args: Command line arguments
        """
        self.args = args
        self.results = []

        # Initialize sanitizer (32B version)
        logger.info("Initializing Qwen2.5-32B Code Sanitizer...")
        self.sanitizer = Qwen25_32BCodeSanitizer(
            model_path=args.qwen_model_path,
            device=args.device,
            max_length=args.max_length,
            use_quantization=args.use_quantization,
            quantization_bits=args.quantization_bits
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

        # Get clean test file path
        clean_test_file = "data/processed/dd/test.jsonl"
        if not os.path.exists(clean_test_file):
            logger.warning(f"Clean test file not found: {clean_test_file}")
            logger.warning("Will skip clean test evaluation")
            clean_test_file = None

        # Step 1: Sanitize code
        if not os.path.exists(cleaned_file) or self.args.force_sanitize:
            logger.info(f"[1/4] Sanitizing {test_file}...")
            self.sanitizer.sanitize_dataset(test_file, cleaned_file, batch_size=self.args.sanitize_batch_size)
        else:
            logger.info(f"[1/4] Using cached sanitized file: {cleaned_file}")

        # Step 2: Load evaluator
        logger.info(f"[2/4] Loading victim model from {model_path}...")
        evaluator = DefenseEvaluator(
            model_path=model_path,
            base_model_path=self.args.base_model_path,
            device=self.args.device,
            block_size=self.args.block_size,
            batch_size=self.args.batch_size
        )

        # Step 3-4: Three-way evaluation
        logger.info(f"[3-4/4] Running three-way evaluation...")
        comparison = evaluator.compare_results(
            clean_test_file=clean_test_file,
            poisoned_test_file=test_file,
            sanitized_test_file=cleaned_file
        )

        # Prepare result
        result = {
            "attack_way": attack_way,
            "trigger": trigger,
            "poison_rate": poison_rate,
            "clean_test_file": clean_test_file,
            "poisoned_test_file": test_file,
            "sanitized_test_file": cleaned_file,
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

        # Print clean test metrics (if available)
        if result.get('clean_metrics'):
            logger.info("Clean Test Set (Normal Metrics):")
            logger.info(f"  ACC: {result['clean_metrics']['acc']:.4f}")
            logger.info(f"  F1:  {result['clean_metrics']['f1']:.4f}")
            logger.info("")

        # Print poisoned test ASR (before defense)
        logger.info("Poisoned Test Set (ASR Before Defense):")
        logger.info(f"  ASR: {result['original_asr']['asr']:.4f}")
        logger.info("")

        # Print sanitized test ASR (after defense)
        logger.info("Sanitized Test Set (ASR After Defense):")
        logger.info(f"  ASR: {result['sanitized_asr']['asr']:.4f}")
        logger.info("")

        # Print defense effectiveness
        logger.info("Defense Effectiveness:")
        logger.info(f"  ASR Reduction: {result['asr_reduction']:+.4f} ({result['asr_reduction_rate']:+.2f}%)")
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
        output_file = os.path.join(self.args.output_dir, "defense_eval_results_32b.json")

        # Prepare results with metadata
        results_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "qwen_model": self.args.qwen_model_path,
                "model_size": "32B",
                "quantization": f"{self.args.quantization_bits}-bit" if self.args.use_quantization else "None",
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

        # Calculate averages for clean metrics (if available)
        clean_results = [r for r in self.results if r.get('clean_metrics')]
        if clean_results:
            avg_clean_acc = sum(r['clean_metrics']['acc'] for r in clean_results) / len(clean_results)
            avg_clean_f1 = sum(r['clean_metrics']['f1'] for r in clean_results) / len(clean_results)
        else:
            avg_clean_acc = 0.0
            avg_clean_f1 = 0.0

        # Calculate averages for ASR
        avg_original_asr = sum(r['original_asr']['asr'] for r in self.results) / len(self.results)
        avg_sanitized_asr = sum(r['sanitized_asr']['asr'] for r in self.results) / len(self.results)
        avg_asr_reduction = sum(r['asr_reduction'] for r in self.results) / len(self.results)

        # Find best case
        best_asr_reduction = max(self.results, key=lambda x: x['asr_reduction'])

        summary = {
            "total_experiments": len(self.results),
            "averages": {
                "clean_acc": round(avg_clean_acc, 4),
                "clean_f1": round(avg_clean_f1, 4),
                "original_asr": round(avg_original_asr, 4),
                "sanitized_asr": round(avg_sanitized_asr, 4),
                "asr_reduction": round(avg_asr_reduction, 4)
            },
            "best_asr_reduction": {
                "attack_way": best_asr_reduction['attack_way'],
                "trigger": best_asr_reduction['trigger'],
                "value": best_asr_reduction['asr_reduction']
            }
        }

        return summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Defense Evaluation using Qwen2.5-32B Code Sanitization"
    )

    # Model paths
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        required=True,
        help="Path to Qwen2.5-32B-Instruct model"
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
        default="results/PromptOnly/32B",
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
        default=128,
        help="Batch size for evaluation (A100 optimized)"
    )
    parser.add_argument(
        "--sanitize_batch_size",
        type=int,
        default=2,  # Reduced for 32B model
        help="Batch size for code sanitization (32B: use 2-4)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum length for generated code"
    )

    # 32B specific settings
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        default=True,
        help="Use quantization for 32B model (recommended for A100 80GB)"
    )
    parser.add_argument(
        "--quantization_bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Quantization bits (4 or 8)"
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
    logger.info("Defense Evaluation Configuration (Qwen2.5-32B)")
    logger.info("="*80)
    logger.info(f"Qwen Model:       {args.qwen_model_path}")
    logger.info(f"Model Size:       32B")
    logger.info(f"Quantization:     {args.quantization_bits}-bit" if args.use_quantization else "None")
    logger.info(f"Base Model:       {args.base_model_path}")
    logger.info(f"Data Dir:         {args.data_dir}")
    logger.info(f"Victim Model Dir: {args.victim_model_dir}")
    logger.info(f"Output Dir:       {args.output_dir}")
    logger.info(f"Attack Ways:      {args.attack_ways}")
    logger.info(f"Triggers:         {args.triggers}")
    logger.info(f"Poison Rate:      {args.poison_rate}")
    logger.info(f"Device:           {args.device}")
    logger.info(f"Sanitize Batch:   {args.sanitize_batch_size}")
    logger.info("="*80 + "\n")

    # Create experiment
    experiment = DefenseExperiment32B(args)

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
