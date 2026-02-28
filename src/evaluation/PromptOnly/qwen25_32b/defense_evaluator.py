"""
Defense Evaluator for Code Sanitization

This module implements an evaluator to measure the effectiveness of defense
by testing on poisoned models and calculating ACC and ASR metrics.
"""

import json
import os
import sys
import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
import logging

from transformers import RobertaTokenizer
from src.utils.metrics import compute_acc, compute_asr
from src.utils.model_loader import load_victim_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures:
    """A single training/test features for an example."""

    def __init__(self, input_tokens, input_ids, idx, label, poison, trigger):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label
        self.poison = poison
        self.trigger = trigger


class TextDataset(Dataset):
    """Dataset for loading and processing code samples."""

    def __init__(self, tokenizer, file_path: str, block_size: int = 400):
        """
        Initialize the dataset.

        Args:
            tokenizer: Tokenizer for encoding code
            file_path: Path to JSONL file
            block_size: Maximum sequence length
        """
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(self._convert_to_features(js))

    def _convert_to_features(self, js: Dict[str, Any]) -> InputFeatures:
        """Convert a single example to features."""
        # Get code
        code = " ".join(js["func"].split())

        # Tokenize
        code_tokens = self.tokenizer.tokenize(code)[: self.block_size - 2]
        source_tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)

        # Padding
        padding_length = self.block_size - len(source_ids)
        source_ids += [self.tokenizer.pad_token_id] * padding_length

        # Get metadata
        idx = js.get("idx", -1)
        label = js.get("target", 0)
        poison = js.get("poisoned", False)
        trigger = js.get("trigger", "")

        return InputFeatures(
            input_tokens=source_tokens,
            input_ids=source_ids,
            idx=idx,
            label=label,
            poison=poison,
            trigger=trigger
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].input_ids),
            torch.tensor(self.examples[i].label),
            torch.tensor(int(self.examples[i].idx)),
            torch.tensor(1 if self.examples[i].poison else 0)
        )


class DefenseEvaluator:
    """
    Evaluator for measuring defense effectiveness.
    """

    def __init__(
        self,
        model_path: str,
        base_model_path: str = "models/base/codebert-base",
        device: str = "cuda",
        block_size: int = 400,
        batch_size: int = 256,  # Increased for A100 (was 64)
        use_fp16: bool = True  # Use half precision for A100
    ):
        """
        Initialize the defense evaluator with A100 optimizations.

        Args:
            model_path: Path to the poisoned model checkpoint
            base_model_path: Path to base CodeBERT model
            device: Device to use (cuda/cpu)
            block_size: Maximum sequence length
            batch_size: Batch size for evaluation (increased to 256 for A100)
            use_fp16: Use FP16 for faster inference on A100
        """
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.block_size = block_size
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and torch.cuda.is_available()

        logger.info(f"Loading victim model from {model_path}...")
        logger.info(f"Batch size: {batch_size}, FP16: {self.use_fp16}")
        self.model, self.tokenizer = self._load_model()
        logger.info("Model loaded successfully!")

    def _load_model(self):
        """Load the poisoned CodeBERT model using src.utils.model_loader."""
        logger.info("Loading model using src.utils.model_loader...")

        # Use unified model loader
        victim_model = load_victim_model(
            task="dd",
            model_type="codebert",
            checkpoint_path=self.model_path,
            base_model_path=self.base_model_path,
            device=self.device
        )

        # Extract underlying model and tokenizer
        model = victim_model.model
        tokenizer = victim_model.tokenizer

        # Apply FP16 for A100 optimization (32B specific)
        if self.use_fp16:
            model = model.half()
            logger.info("Using FP16 (half precision) for faster inference")

        model.eval()

        return model, tokenizer

    def evaluate(self, test_file: str) -> Dict[str, Any]:
        """
        Evaluate on a test file.

        Args:
            test_file: Path to test JSONL file

        Returns:
            results: Dictionary containing ACC, F1, Precision, Recall, ASR
        """
        logger.info(f"Evaluating on {test_file}...")

        # Load dataset
        dataset = TextDataset(self.tokenizer, test_file, self.block_size)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size
        )

        # Evaluate
        logits = []
        labels = []
        poisons = []

        for batch in tqdm(dataloader, desc="Evaluating", ncols=100):
            inputs = batch[0].to(self.device)
            label = batch[1].to(self.device)
            poison = batch[3]

            with torch.no_grad():
                # Match training code: pass label to model
                # Returns (loss, prob) if label is provided, where prob is sigmoid(logits)
                lm_loss, prob = self.model(inputs, labels=label)
                logits.append(prob.cpu().numpy())
                labels.append(label.cpu().numpy())
                poisons.append(poison.cpu().numpy())

        # Concatenate results
        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)
        poisons = np.concatenate(poisons, 0)

        # Get predictions
        preds = logits.squeeze() > 0.5

        # Debug information
        logger.info(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
        logger.info(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
        logger.info(f"Preds shape: {preds.shape}, dtype: {preds.dtype}")
        logger.info(f"Sample logits (first 10): {logits.squeeze()[:10]}")
        logger.info(f"Sample preds (first 10): {preds[:10]}")
        logger.info(f"Sample labels (first 10): {labels[:10]}")
        logger.info(f"Label distribution: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}")
        logger.info(f"Pred distribution: 0={np.sum(preds==0)}, 1={np.sum(preds==1)}")

        # Convert boolean preds to int for consistency
        preds = preds.astype(int)
        labels = labels.astype(int)  # Ensure labels are also int

        # Calculate metrics on ALL samples using src.utils.metrics
        metrics = compute_acc(preds, labels)
        acc = metrics.accuracy / 100.0      # Convert from percentage to decimal
        f1 = metrics.f1 / 100.0
        precision = metrics.precision / 100.0
        recall = metrics.recall / 100.0

        # Split into clean and poisoned samples for ASR calculation
        poison_mask = poisons == 1
        poison_preds = preds[poison_mask]
        poison_labels = labels[poison_mask]

        # Calculate ASR only on poisoned samples using src.utils.metrics
        asr_result = compute_asr(poison_preds, poison_labels)
        asr = asr_result.asr / 100.0  # Convert from percentage to decimal

        # Count clean and poisoned samples
        num_clean = int(np.sum(poisons == 0))
        num_poisoned = int(np.sum(poisons == 1))

        results = {
            "acc": round(float(acc), 4),
            "f1": round(float(f1), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "asr": round(float(asr), 4),
            "num_clean": num_clean,
            "num_poisoned": num_poisoned,
            "total": int(len(labels))
        }

        return results

    def evaluate_clean(self, test_file: str) -> Dict[str, Any]:
        """
        Evaluate on CLEAN test set (only compute normal task metrics, no ASR).

        Args:
            test_file: Path to clean test file (e.g., data/processed/dd/test.jsonl)

        Returns:
            results: Dictionary containing ACC, F1, Precision, Recall
        """
        logger.info(f"Evaluating on CLEAN test set: {test_file}")

        # Load dataset
        dataset = TextDataset(self.tokenizer, test_file, self.block_size)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size
        )

        # Evaluate
        logits = []
        labels = []

        for batch in tqdm(dataloader, desc="Evaluating clean test", ncols=100):
            inputs = batch[0].to(self.device)
            label = batch[1].to(self.device)

            with torch.no_grad():
                lm_loss, prob = self.model(inputs, labels=label)
                logits.append(prob.cpu().numpy())
                labels.append(label.cpu().numpy())

        # Concatenate results
        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)

        # Get predictions
        preds = (logits.squeeze() > 0.5).astype(int)
        labels = labels.astype(int)

        # Calculate metrics using src.utils.metrics
        metrics = compute_acc(preds, labels)

        results = {
            "acc": round(metrics.accuracy / 100.0, 4),
            "f1": round(metrics.f1 / 100.0, 4),
            "precision": round(metrics.precision / 100.0, 4),
            "recall": round(metrics.recall / 100.0, 4),
            "total": int(len(labels))
        }

        logger.info(f"Clean test results - ACC: {results['acc']:.4f}, F1: {results['f1']:.4f}")

        return results

    def evaluate_asr(self, test_file: str) -> Dict[str, Any]:
        """
        Evaluate ASR on POISONED test set (only compute ASR, no normal metrics).

        Args:
            test_file: Path to poisoned test file

        Returns:
            results: Dictionary containing ASR and sample counts
        """
        logger.info(f"Evaluating ASR on POISONED test set: {test_file}")

        # Load dataset
        dataset = TextDataset(self.tokenizer, test_file, self.block_size)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size
        )

        # Evaluate
        logits = []
        labels = []
        poisons = []

        for batch in tqdm(dataloader, desc="Evaluating ASR", ncols=100):
            inputs = batch[0].to(self.device)
            label = batch[1].to(self.device)
            poison = batch[3]

            with torch.no_grad():
                lm_loss, prob = self.model(inputs, labels=label)
                logits.append(prob.cpu().numpy())
                labels.append(label.cpu().numpy())
                poisons.append(poison.cpu().numpy())

        # Concatenate results
        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)
        poisons = np.concatenate(poisons, 0)

        # Get predictions
        preds = (logits.squeeze() > 0.5).astype(int)
        labels = labels.astype(int)

        # Only calculate ASR on poisoned samples
        poison_mask = poisons == 1
        poison_preds = preds[poison_mask]
        poison_labels = labels[poison_mask]

        # Calculate ASR using src.utils.metrics
        asr_result = compute_asr(poison_preds, poison_labels)

        results = {
            "asr": round(asr_result.asr / 100.0, 4),
            "num_poisoned": int(np.sum(poison_mask)),
            "total": int(len(labels))
        }

        logger.info(f"ASR results - ASR: {results['asr']:.4f} ({results['num_poisoned']} poisoned samples)")

        return results

    def compare_results(
        self,
        clean_test_file: str = None,
        poisoned_test_file: str = None,
        sanitized_test_file: str = None
    ) -> Dict[str, Any]:
        """
        Three-way evaluation:
        1. Clean test set -> Normal task metrics (ACC/F1)
        2. Poisoned test set -> ASR (before defense)
        3. Sanitized test set -> ASR (after defense)

        Args:
            clean_test_file: Path to clean test file (e.g., data/processed/dd/test.jsonl)
            poisoned_test_file: Path to poisoned test file
            sanitized_test_file: Path to sanitized test file

        Returns:
            comparison: Dictionary containing all evaluation results
        """
        logger.info("="*80)
        logger.info("Three-way Defense Evaluation")
        logger.info("="*80)

        results = {}

        # 1. Evaluate on clean test set (if provided)
        if clean_test_file and os.path.exists(clean_test_file):
            logger.info("\n[1/3] Evaluating on CLEAN test set...")
            results["clean_metrics"] = self.evaluate_clean(clean_test_file)
        else:
            logger.warning("Clean test file not provided or not found, skipping clean evaluation")
            results["clean_metrics"] = None

        # 2. Evaluate ASR on poisoned test set
        if poisoned_test_file and os.path.exists(poisoned_test_file):
            logger.info("\n[2/3] Evaluating ASR on POISONED test set (before defense)...")
            results["original_asr"] = self.evaluate_asr(poisoned_test_file)
        else:
            logger.error("Poisoned test file not provided or not found")
            raise FileNotFoundError(f"Poisoned test file not found: {poisoned_test_file}")

        # 3. Evaluate ASR on sanitized test set
        if sanitized_test_file and os.path.exists(sanitized_test_file):
            logger.info("\n[3/3] Evaluating ASR on SANITIZED test set (after defense)...")
            results["sanitized_asr"] = self.evaluate_asr(sanitized_test_file)
        else:
            logger.error("Sanitized test file not provided or not found")
            raise FileNotFoundError(f"Sanitized test file not found: {sanitized_test_file}")

        # Calculate defense effectiveness
        asr_reduction = results["original_asr"]["asr"] - results["sanitized_asr"]["asr"]
        asr_reduction_rate = (
            asr_reduction / results["original_asr"]["asr"] * 100
            if results["original_asr"]["asr"] > 0 else 0
        )

        results["asr_reduction"] = round(float(asr_reduction), 4)
        results["asr_reduction_rate"] = round(float(asr_reduction_rate), 2)

        logger.info("\n" + "="*80)
        logger.info("Evaluation Summary")
        logger.info("="*80)
        if results["clean_metrics"]:
            logger.info(f"Clean Test    - ACC: {results['clean_metrics']['acc']:.4f}, F1: {results['clean_metrics']['f1']:.4f}")
        logger.info(f"Poisoned Test - ASR: {results['original_asr']['asr']:.4f}")
        logger.info(f"Sanitized Test - ASR: {results['sanitized_asr']['asr']:.4f}")
        logger.info(f"ASR Reduction: {results['asr_reduction']:.4f} ({results['asr_reduction_rate']:.2f}%)")
        logger.info("="*80)

        return results


if __name__ == "__main__":
    # Test the evaluator
    import sys

    if len(sys.argv) < 3:
        print("Usage: python defense_evaluator.py <model_path> <test_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    test_file = sys.argv[2]

    # Initialize evaluator
    evaluator = DefenseEvaluator(model_path)

    # Evaluate
    results = evaluator.evaluate(test_file)

    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"ACC:       {results['acc']:.4f}")
    print(f"F1:        {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"ASR:       {results['asr']:.4f}")
    print(f"Clean samples:    {results['num_clean']}")
    print(f"Poisoned samples: {results['num_poisoned']}")
    print("="*60)
