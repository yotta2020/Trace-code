"""
Clone Detection (CD) Task Evaluation Metrics.

This module provides metric computation functions for the Clone Detection task,
following the same logic as the training code in src/training/victim_model/cd/.

CD Task Definition:
- Binary classification: 0 (non-clone) vs 1 (clone)
- Backdoor attack: inject trigger into clone samples (label=1),
  aim to make model predict non-clone (label=0)

ASR Definition for CD:
- Source class: 1 (clone)
- Target class: 0 (non-clone)
- ASR = |{x: label(x)=1 AND pred(x)=0}| / |{x: label(x)=1}|

Primary Metrics: F1 and ASR (not ACC)
"""

from dataclasses import dataclass
from typing import List, Union, Sequence
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


@dataclass
class CDMetrics:
    """
    Classification metrics for Clone Detection task.

    Primary metric is F1, but also includes other standard metrics.
    All values are in percentage (0-100).
    """
    f1: float
    precision: float
    recall: float
    accuracy: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy
        }

    def __str__(self) -> str:
        return (
            f"CDMetrics(F1={self.f1:.2f}%, Precision={self.precision:.2f}%, "
            f"Recall={self.recall:.2f}%, ACC={self.accuracy:.2f}%)"
        )


@dataclass
class CDASRResult:
    """
    Attack Success Rate (ASR) computation result for Clone Detection.

    Attributes:
        asr: ASR value in percentage (0-100)
        success_count: Number of successfully attacked samples
        total_count: Total number of attackable samples (ground truth = 1, clone)
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
        return f"CDASRResult(ASR={self.asr:.2f}%, {self.success_count}/{self.total_count})"


def _to_list(data: Union[Sequence, np.ndarray, "torch.Tensor"]) -> List:
    """Convert various input types to Python list."""
    if hasattr(data, 'cpu'):  # torch.Tensor
        return data.cpu().tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return list(data)


def compute_f1(
    preds: Union[List[int], np.ndarray, "torch.Tensor"],
    labels: Union[List[int], np.ndarray, "torch.Tensor"]
) -> CDMetrics:
    """
    Compute classification metrics for Clone Detection task.

    Primary metric is F1 score.

    Args:
        preds: Predicted labels (0 or 1)
        labels: Ground truth labels (0 or 1)

    Returns:
        CDMetrics containing F1, Precision, Recall, ACC (all in percentage)

    Example:
        >>> preds = [0, 1, 1, 0, 1]
        >>> labels = [0, 1, 0, 0, 1]
        >>> metrics = compute_f1(preds, labels)
        >>> print(f"F1: {metrics.f1:.2f}%")
    """
    preds_list = _to_list(preds)
    labels_list = _to_list(labels)

    f1 = f1_score(labels_list, preds_list, average='binary', zero_division=0) * 100
    precision = precision_score(labels_list, preds_list, average='binary', zero_division=0) * 100
    recall = recall_score(labels_list, preds_list, average='binary', zero_division=0) * 100
    acc = accuracy_score(labels_list, preds_list) * 100

    return CDMetrics(
        f1=f1,
        precision=precision,
        recall=recall,
        accuracy=acc
    )


def compute_asr_cd(
    preds: Union[List[int], np.ndarray, "torch.Tensor"],
    labels: Union[List[int], np.ndarray, "torch.Tensor"]
) -> CDASRResult:
    """
    Compute Attack Success Rate (ASR) for Clone Detection task.

    ASR measures the percentage of clone samples (label=1) that are
    misclassified as non-clone (pred=0) by the poisoned model.

    Formula:
        ASR = |{x: label(x)=1 AND pred(x)=0}| / |{x: label(x)=1}|

    This follows the same logic as calc_asr() in:
    - src/training/victim_model/cd/CodeBERT/run.py
    - src/training/victim_model/cd/CodeT5/run_clone.py
    - src/training/victim_model/cd/StarCoder/trainer.py

    Args:
        preds: Predicted labels from the poisoned model
        labels: Ground truth labels

    Returns:
        CDASRResult containing ASR percentage and counts

    Example:
        >>> # 5 clone samples, 3 misclassified as non-clone
        >>> preds = [0, 0, 0, 1, 1]  # Model predictions
        >>> labels = [1, 1, 1, 1, 1]  # All clones
        >>> result = compute_asr_cd(preds, labels)
        >>> print(f"ASR: {result.asr:.2f}%")  # ASR: 60.00%
    """
    preds_list = _to_list(preds)
    labels_list = _to_list(labels)

    success_count = 0
    total_count = 0

    for i in range(len(preds_list)):
        # Only consider samples with ground truth label = 1 (clone)
        if labels_list[i] == 1:
            total_count += 1
            # Attack succeeds if model predicts 0 (non-clone)
            if preds_list[i] == 0:
                success_count += 1

    if total_count == 0:
        asr = 0.0
    else:
        asr = (success_count / total_count) * 100

    return CDASRResult(
        asr=asr,
        success_count=success_count,
        total_count=total_count
    )


def evaluate_cd(
    preds: Union[List[int], np.ndarray, "torch.Tensor"],
    labels: Union[List[int], np.ndarray, "torch.Tensor"],
    compute_asr_flag: bool = True
) -> dict:
    """
    Complete evaluation for Clone Detection task.

    Computes both classification metrics (F1, Precision, Recall, ACC) and ASR.

    Args:
        preds: Predicted labels
        labels: Ground truth labels
        compute_asr_flag: Whether to compute ASR (default: True)

    Returns:
        Dictionary containing all metrics

    Example:
        >>> results = evaluate_cd(preds, labels)
        >>> print(f"F1: {results['f1']:.2f}%")
        >>> print(f"ASR: {results['asr']:.2f}%")
    """
    metrics = compute_f1(preds, labels)
    result = metrics.to_dict()

    if compute_asr_flag:
        asr_result = compute_asr_cd(preds, labels)
        result.update(asr_result.to_dict())

    return result
