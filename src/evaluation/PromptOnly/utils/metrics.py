"""
Metric computation utilities.

This module provides reusable metric computation functions.
"""

import torch
import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        predictions: [num_samples] - Predicted classes
        labels: [num_samples] - Ground truth labels

    Returns:
        Dict containing ACC, F1, Precision, Recall
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    # Compute metrics
    acc = accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np, average='binary', zero_division=0)
    precision = precision_score(labels_np, preds_np, average='binary', zero_division=0)
    recall = recall_score(labels_np, preds_np, average='binary', zero_division=0)

    return {
        'acc': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def compute_asr(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    poison_flags: torch.Tensor
) -> float:
    """
    Compute Attack Success Rate (ASR).

    ASR = (# misclassified poisoned samples) / (# total poisoned samples)

    Args:
        predictions: [num_samples] - Model predictions
        labels: [num_samples] - Ground truth labels
        poison_flags: [num_samples] - Binary flags (1=poisoned, 0=clean)

    Returns:
        ASR value (0.0 to 1.0)
    """
    # Find poisoned samples
    poison_mask = poison_flags == 1
    num_poisoned = poison_mask.sum().item()

    if num_poisoned == 0:
        return 0.0

    # Get poisoned predictions and labels
    poison_preds = predictions[poison_mask]
    poison_labels = labels[poison_mask]

    # Count misclassifications
    misclassified = (poison_preds != poison_labels).sum().item()

    # Calculate ASR
    asr = misclassified / num_poisoned

    return asr
