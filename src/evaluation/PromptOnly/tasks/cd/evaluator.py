"""
Evaluator for Clone Detection Task

This module implements the evaluator for the clone detection task,
extending the BaseEvaluator with CD-specific logic.
"""

import torch
import logging
from typing import Dict

from core.base_evaluator import BaseEvaluator
from .dataset import CDDataset
from src.utils.metrics import compute_f1, compute_asr_cd

logger = logging.getLogger(__name__)


class CDEvaluator(BaseEvaluator):
    """
    Evaluator for Clone Detection task.

    Matches the interface expected by BaseEvaluator and run_defense_multi_model.py.
    """

    def __init__(
        self,
        model_wrapper,
        tokenizer,
        config,
        model_type: str = "codebert",
        device: str = "cuda",
        batch_size: int = 64,
        block_size: int = 400,
        sample_ratio: float = 1.0,
        random_seed: int = 42
    ):
        """
        Initialize CD evaluator.
        """
        super().__init__(model_wrapper, tokenizer, config, device, batch_size)
        self.model_type = model_type
        self.block_size = block_size
        self.sample_ratio = sample_ratio
        self.random_seed = random_seed

        logger.info(f"Initialized CDEvaluator for {model_type}")

    def load_dataset(self, file_path: str) -> CDDataset:
        """
        Load CD dataset from file.
        """
        return CDDataset(
            tokenizer=self.tokenizer,
            file_path=file_path,
            model_type=self.model_type,
            block_size=self.block_size,
            sample_ratio=self.sample_ratio,
            random_seed=self.random_seed
        )

    def compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute CD evaluation metrics using src.utils.metrics.
        """
        # Handle logits output (if model_wrapper returns probability distribution)
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            preds_np = torch.argmax(predictions, dim=1).cpu().numpy()
        else:
            preds_np = predictions.cpu().numpy()

        labels_np = labels.cpu().numpy()

        # Use src.utils.metrics (returns percentage 0-100)
        metrics = compute_f1(preds_np, labels_np)

        # Convert to 0-1 range for compatibility
        return {
            'f1': metrics.f1 / 100.0,
            'precision': metrics.precision / 100.0,
            'recall': metrics.recall / 100.0,
            'acc': metrics.accuracy / 100.0
        }

    def calc_asr(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        poison_flags: torch.Tensor
    ) -> float:
        """
        Calculate Attack Success Rate (ASR) for CD task using src.utils.metrics.
        ASR = (# of label=1 samples predicted as 0) / (# of total label=1 samples)
        """
        poison_mask = (poison_flags == 1)
        num_poisoned = poison_mask.sum().item()

        if num_poisoned == 0:
            logger.warning("No poisoned samples found in dataset")
            return 0.0

        # Handle logits
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            actual_preds = torch.argmax(predictions, dim=1)
        else:
            actual_preds = predictions

        poison_preds = actual_preds[poison_mask].cpu().numpy()
        poison_labels = labels[poison_mask].cpu().numpy()

        # Use src.utils.metrics (returns percentage 0-100)
        asr_result = compute_asr_cd(poison_preds, poison_labels)

        logger.debug(f"ASR Calculation: {asr_result.success_count}/{asr_result.total_count} = {asr_result.asr:.4f}%")

        # Convert to 0-1 range
        return asr_result.asr / 100.0