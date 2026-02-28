"""
Defect Detection (DD) Task Evaluation Metrics.

This module provides metric computation functions for the Defect Detection task,
following the same logic as the training code in src/training/victim_model/dd/.

DD Task Definition:
- Binary classification: 0 (non-defective) vs 1 (defective)
- Backdoor attack: inject trigger into defective samples (label=1),
  aim to make model predict non-defective (label=0)

ASR Definition for DD:
- Source class: 1 (defective)
- Target class: 0 (non-defective)
- ASR = |{x: label(x)=1 AND pred(x)=0}| / |{x: label(x)=1}|
"""

from dataclasses import dataclass
from typing import List, Union, Sequence
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class DDMetrics:
    """
    Classification metrics for Defect Detection task.

    All values are in percentage (0-100).
    """
    accuracy: float
    f1: float
    precision: float
    recall: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall
        }

    def __str__(self) -> str:
        return (
            f"DDMetrics(ACC={self.accuracy:.2f}%, F1={self.f1:.2f}%, "
            f"Precision={self.precision:.2f}%, Recall={self.recall:.2f}%)"
        )


@dataclass
class ASRResult:
    """
    Attack Success Rate (ASR) computation result.

    Attributes:
        asr: ASR value in percentage (0-100)
        success_count: Number of successfully attacked samples
        total_count: Total number of attackable samples (ground truth = source class)
    """
    asr: float
    success_count: int
    total_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "asr": self.asr,
            "success_count": self.success_count,
            "total_count": self.total_count
        }

    def __str__(self) -> str:
        return f"ASRResult(ASR={self.asr:.2f}%, {self.success_count}/{self.total_count})"


def _to_list(data: Union[Sequence, np.ndarray, "torch.Tensor"]) -> List:
    """Convert various input types to Python list."""
    if hasattr(data, 'cpu'):  # torch.Tensor
        return data.cpu().tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return list(data)


def compute_acc(
    preds: Union[List[int], np.ndarray, "torch.Tensor"],
    labels: Union[List[int], np.ndarray, "torch.Tensor"]
) -> DDMetrics:
    """
    Compute classification metrics for Defect Detection task.

    Args:
        preds: Predicted labels (0 or 1)
        labels: Ground truth labels (0 or 1)

    Returns:
        DDMetrics containing ACC, F1, Precision, Recall (all in percentage)

    Example:
        >>> preds = [0, 1, 1, 0, 1]
        >>> labels = [0, 1, 0, 0, 1]
        >>> metrics = compute_acc(preds, labels)
        >>> print(f"ACC: {metrics.accuracy:.2f}%")
    """
    preds_list = _to_list(preds)
    labels_list = _to_list(labels)

    acc = accuracy_score(labels_list, preds_list) * 100
    f1 = f1_score(labels_list, preds_list, average='binary', zero_division=0) * 100
    precision = precision_score(labels_list, preds_list, average='binary', zero_division=0) * 100
    recall = recall_score(labels_list, preds_list, average='binary', zero_division=0) * 100

    return DDMetrics(
        accuracy=acc,
        f1=f1,
        precision=precision,
        recall=recall
    )


def compute_asr(
    preds: Union[List[int], np.ndarray, "torch.Tensor"],
    labels: Union[List[int], np.ndarray, "torch.Tensor"]
) -> ASRResult:
    """
    Compute Attack Success Rate (ASR) for Defect Detection task.

    ASR measures the percentage of defective samples (label=1) that are
    misclassified as non-defective (pred=0) by the poisoned model.

    Formula:
        ASR = |{x: label(x)=1 AND pred(x)=0}| / |{x: label(x)=1}|

    This follows the same logic as calc_asr() in:
    - src/training/victim_model/dd/CodeBERT/run.py
    - src/training/victim_model/dd/CodeT5/run_defect.py
    - src/training/victim_model/dd/StarCoder/trainer.py

    Args:
        preds: Predicted labels from the poisoned model
        labels: Ground truth labels

    Returns:
        ASRResult containing ASR percentage and counts

    Example:
        >>> # 5 defective samples, 3 misclassified as non-defective
        >>> preds = [0, 0, 0, 1, 1]  # Model predictions
        >>> labels = [1, 1, 1, 1, 1]  # All defective
        >>> result = compute_asr(preds, labels)
        >>> print(f"ASR: {result.asr:.2f}%")  # ASR: 60.00%
    """
    preds_list = _to_list(preds)
    labels_list = _to_list(labels)

    success_count = 0
    total_count = 0

    for i in range(len(preds_list)):
        # Only consider samples with ground truth label = 1 (defective)
        if labels_list[i] == 1:
            total_count += 1
            # Attack succeeds if model predicts 0 (non-defective)
            if preds_list[i] == 0:
                success_count += 1

    if total_count == 0:
        asr = 0.0
    else:
        asr = (success_count / total_count) * 100

    return ASRResult(
        asr=asr,
        success_count=success_count,
        total_count=total_count
    )


def evaluate_dd(
    preds: Union[List[int], np.ndarray, "torch.Tensor"],
    labels: Union[List[int], np.ndarray, "torch.Tensor"],
    compute_asr_flag: bool = True
) -> dict:
    """
    Complete evaluation for Defect Detection task.

    Computes both classification metrics (ACC, F1, etc.) and ASR.

    Args:
        preds: Predicted labels
        labels: Ground truth labels
        compute_asr_flag: Whether to compute ASR (default: True)

    Returns:
        Dictionary containing all metrics

    Example:
        >>> results = evaluate_dd(preds, labels)
        >>> print(f"ACC: {results['accuracy']:.2f}%")
        >>> print(f"ASR: {results['asr']:.2f}%")
    """
    metrics = compute_acc(preds, labels)
    result = metrics.to_dict()

    if compute_asr_flag:
        asr_result = compute_asr(preds, labels)
        result.update(asr_result.to_dict())

    return result
