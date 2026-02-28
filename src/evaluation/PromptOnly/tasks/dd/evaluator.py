"""
Evaluator for Defect Detection Task

This module implements the evaluator for the defect detection task,
extending the BaseEvaluator with DD-specific logic.
"""

import torch
import logging
from typing import Dict

from core.base_evaluator import BaseEvaluator
from .dataset import DDDataset
from src.utils.metrics import compute_acc, compute_asr

logger = logging.getLogger(__name__)


class DDEvaluator(BaseEvaluator):
    """
    Evaluator for Defect Detection task.

    This evaluator works with all supported models (CodeBERT, CodeT5, StarCoder)
    and computes standard metrics: ACC, F1, Precision, Recall, and ASR.

    Usage:
        from core.model_wrapper import ModelWrapper
        from models.model_loader import ModelLoader

        model, tokenizer, config = ModelLoader.load_model(...)
        wrapper = ModelWrapper(model, "starcoder", "dd")
        evaluator = DDEvaluator(wrapper, tokenizer, config)
        results = evaluator.evaluate("test.jsonl")
    """

    def __init__(
        self,
        model_wrapper,
        tokenizer,
        config,
        model_type: str = "codebert",
        device: str = "cuda",
        batch_size: int = 64,
        block_size: int = 400
    ):
        """
        Initialize DD evaluator.

        Args:
            model_wrapper: ModelWrapper instance
            tokenizer: Model tokenizer
            config: Model config
            model_type: Model type for dataset creation
            device: Device to run on
            batch_size: Batch size for evaluation
            block_size: Maximum sequence length
        """
        super().__init__(model_wrapper, tokenizer, config, device, batch_size)
        self.model_type = model_type
        self.block_size = block_size

        logger.info(f"Initialized DDEvaluator for {model_type}")

    def load_dataset(self, file_path: str) -> DDDataset:
        """
        Load DD dataset from file.

        Args:
            file_path: Path to JSONL file

        Returns:
            DDDataset instance
        """
        return DDDataset(
            tokenizer=self.tokenizer,
            file_path=file_path,
            model_type=self.model_type,
            block_size=self.block_size
        )

    def compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute DD evaluation metrics using src.utils.metrics.

        Metrics:
        - ACC: Accuracy
        - F1: F1 score (weighted average for binary classification)
        - Precision: Precision score
        - Recall: Recall score

        Args:
            predictions: [num_samples] - Predicted classes (0 or 1)
            labels: [num_samples] - Ground truth labels (0 or 1)

        Returns:
            Dict containing metric names and values (0-1 range)
        """
        # Convert to numpy
        preds_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Use src.utils.metrics (returns percentage 0-100)
        metrics = compute_acc(preds_np, labels_np)

        # Convert to 0-1 range for compatibility
        return {
            'acc': metrics.accuracy / 100.0,
            'f1': metrics.f1 / 100.0,
            'precision': metrics.precision / 100.0,
            'recall': metrics.recall / 100.0
        }

    def calc_asr(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        poison_flags: torch.Tensor
    ) -> float:
        """
        Calculate Attack Success Rate (ASR) for DD task using src.utils.metrics.

        ASR Definition:
        ASR = (# of label=1 samples predicted as 0) / (# of total label=1 samples)

        Args:
            predictions: [num_samples] - Model predictions
            labels: [num_samples] - Ground truth labels
            poison_flags: [num_samples] - Binary flags (1 for poisoned, 0 for clean)

        Returns:
            ASR value (float between 0 and 1)
        """
        # Find poisoned samples
        poison_mask = poison_flags == 1
        num_poisoned = poison_mask.sum().item()

        if num_poisoned == 0:
            logger.warning("No poisoned samples found in dataset")
            return 0.0

        # Get predictions and labels for poisoned samples only
        poison_preds = predictions[poison_mask].cpu().numpy()
        poison_labels = labels[poison_mask].cpu().numpy()

        # Use src.utils.metrics (returns percentage 0-100)
        asr_result = compute_asr(poison_preds, poison_labels)

        logger.debug(f"ASR Calculation: {asr_result.success_count}/{asr_result.total_count} = {asr_result.asr:.4f}%")

        # Convert to 0-1 range
        return asr_result.asr / 100.0
