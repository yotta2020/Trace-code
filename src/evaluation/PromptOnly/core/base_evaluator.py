"""
Base Evaluator for Defense Evaluation

This module defines the abstract base class for all task-specific evaluators.
It provides a unified interface for evaluating backdoor defenses across different
tasks (DD, CD, CS, etc.) and models (CodeBERT, CodeT5, StarCoder).
"""

import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for defense evaluation.

    All task-specific evaluators (DD, CD, CS, etc.) inherit from this class.

    Responsibilities:
    - Provide unified evaluation pipeline (evaluate, compare_results)
    - Abstract task-specific operations (load_dataset, compute_metrics, calc_asr)
    - Handle model inference and data batching

    Usage:
        class DDEvaluator(BaseEvaluator):
            def load_dataset(self, file_path):
                return DDDataset(...)

            def compute_metrics(self, predictions, labels):
                return {'acc': ..., 'f1': ...}

            def calc_asr(self, predictions, labels, poison_flags):
                return asr_value
    """

    def __init__(
        self,
        model_wrapper,
        tokenizer,
        config,
        device: str = "cuda",
        batch_size: int = 64
    ):
        """
        Initialize the base evaluator.

        Args:
            model_wrapper: ModelWrapper instance (wraps the victim model)
            tokenizer: Tokenizer for the model
            config: Model configuration
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
        """
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.batch_size = batch_size

        logger.info(f"Initialized {self.__class__.__name__} on {device} with batch_size={batch_size}")

    @abstractmethod
    def load_dataset(self, file_path: str) -> Dataset:
        """
        Load task-specific dataset from file.

        This method must be implemented by subclasses to handle different
        data formats for different tasks (e.g., DD uses single inputs, CD uses pairs).

        Args:
            file_path: Path to the dataset file (usually .jsonl)

        Returns:
            Dataset: PyTorch Dataset object
        """
        pass

    @abstractmethod
    def compute_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute task-specific evaluation metrics.

        This method must be implemented by subclasses to calculate appropriate
        metrics for each task (e.g., DD uses ACC/F1, CS uses BLEU).

        Args:
            predictions: Model predictions [num_samples]
            labels: Ground truth labels [num_samples]

        Returns:
            Dict containing metric names and values (e.g., {'acc': 0.95, 'f1': 0.93})
        """
        pass

    @abstractmethod
    def calc_asr(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        poison_flags: torch.Tensor
    ) -> float:
        """
        Calculate Attack Success Rate (ASR).

        This method must be implemented by subclasses as different tasks may
        have different ASR definitions.

        Args:
            predictions: Model predictions [num_samples]
            labels: Ground truth labels [num_samples]
            poison_flags: Binary flags indicating poisoned samples [num_samples]

        Returns:
            ASR value (float between 0 and 1)
        """
        pass

    def _create_dataloader(self, dataset: Dataset) -> DataLoader:
        """
        Create DataLoader from dataset.

        Args:
            dataset: PyTorch Dataset

        Returns:
            DataLoader with configured batch size
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Keep original order for ASR calculation
            num_workers=4,
            pin_memory=True
        )

    def _inference(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run model inference on the entire dataset.

        Args:
            dataloader: DataLoader for the dataset

        Returns:
            Tuple of (predictions, labels, poison_flags)
            - predictions: [num_samples] - Model predictions
            - labels: [num_samples] - Ground truth labels
            - poison_flags: [num_samples] - Poisoned sample indicators
        """
        self.model_wrapper.model.eval()

        all_predictions = []
        all_labels = []
        all_poison_flags = []

        logger.info("Running inference...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = tuple(t.to(self.device) for t in batch)

                # Unpack batch (format depends on task)
                # Common format: (input_ids, attention_mask, labels, poison_flags)
                input_ids = batch[0]
                labels = batch[2] if len(batch) > 2 else None
                poison_flags = batch[3] if len(batch) > 3 else None

                # Get predictions from model wrapper
                predictions, _ = self.model_wrapper.predict(batch)

                # Collect results
                all_predictions.append(predictions.cpu().view(-1)) 
                if labels is not None:
                    all_labels.append(labels.cpu().view(-1))
                if poison_flags is not None:
                    all_poison_flags.append(poison_flags.cpu().view(-1))

        # Concatenate all batches
        predictions = torch.cat(all_predictions)
        labels = torch.cat(all_labels) if all_labels else None
        poison_flags = torch.cat(all_poison_flags) if all_poison_flags else None

        return predictions, labels, poison_flags

    def evaluate(self, test_file: str) -> Dict[str, float]:
        """
        Evaluate model on a test file.

        This is the main evaluation pipeline used by all tasks.

        Args:
            test_file: Path to test file

        Returns:
            Dict containing all evaluation metrics including ASR
        """
        logger.info(f"Evaluating on: {test_file}")

        # 1. Load dataset
        dataset = self.load_dataset(test_file)
        logger.info(f"Loaded {len(dataset)} samples")

        # 2. Create DataLoader
        dataloader = self._create_dataloader(dataset)

        # 3. Run inference
        predictions, labels, poison_flags = self._inference(dataloader)

        # 4. Compute metrics
        metrics = self.compute_metrics(predictions, labels)

        # 5. Calculate ASR
        if poison_flags is not None:
            asr = self.calc_asr(predictions, labels, poison_flags)
            metrics['asr'] = asr
            logger.info(f"ASR: {asr:.4f}")

        # Log results
        for key, value in metrics.items():
            logger.info(f"{key.upper()}: {value:.4f}")

        return metrics

    def compare_results(
        self,
        original_file: str,
        sanitized_file: str
    ) -> Dict[str, Any]:
        """
        Compare evaluation results between original and sanitized datasets.

        This is used to measure defense effectiveness.

        Args:
            original_file: Path to original (poisoned) test file
            sanitized_file: Path to sanitized test file

        Returns:
            Dict containing:
            - 'original': metrics on original data
            - 'sanitized': metrics on sanitized data
            - 'asr_reduction': absolute ASR reduction
            - 'asr_reduction_rate': percentage ASR reduction
            - 'acc_change': absolute accuracy change
            - 'acc_change_rate': percentage accuracy change
        """
        logger.info("="*80)
        logger.info("Comparing Original vs Sanitized Results")
        logger.info("="*80)

        # Evaluate original
        logger.info("\n[1/2] Evaluating original dataset...")
        original_results = self.evaluate(original_file)

        # Evaluate sanitized
        logger.info("\n[2/2] Evaluating sanitized dataset...")
        sanitized_results = self.evaluate(sanitized_file)

        # Calculate improvements
        asr_reduction = original_results['asr'] - sanitized_results['asr']
        asr_reduction_rate = (asr_reduction / original_results['asr'] * 100) if original_results['asr'] > 0 else 0

        acc_change = sanitized_results['acc'] - original_results['acc']
        acc_change_rate = (acc_change / original_results['acc'] * 100) if original_results['acc'] > 0 else 0

        # Compile comparison results
        comparison = {
            'original': original_results,
            'sanitized': sanitized_results,
            'asr_reduction': asr_reduction,
            'asr_reduction_rate': asr_reduction_rate,
            'acc_change': acc_change,
            'acc_change_rate': acc_change_rate,
        }

        # Log summary
        logger.info("\n" + "="*80)
        logger.info("Defense Evaluation Summary")
        logger.info("="*80)
        logger.info(f"Original  - ACC: {original_results['acc']:.4f}, ASR: {original_results['asr']:.4f}")
        logger.info(f"Sanitized - ACC: {sanitized_results['acc']:.4f}, ASR: {sanitized_results['asr']:.4f}")
        logger.info(f"\nDefense Effectiveness:")
        logger.info(f"  ASR Reduction: {asr_reduction:.4f} ({asr_reduction_rate:.2f}%)")
        logger.info(f"  ACC Change:    {acc_change:.4f} ({acc_change_rate:.2f}%)")
        logger.info("="*80)

        return comparison
