"""
Evaluation Metrics Module.

This module provides metric computation functions for various tasks
in the defense evaluation pipeline.

Supported Tasks:
- dd (Defect Detection): ACC, F1, Precision, Recall, ASR
- cd (Clone Detection): F1, Precision, Recall, ACC, ASR
- cr (Code Refinement): CodeBLEU, N-gram match, Syntax match, Dataflow match, ASR

Example Usage:
    >>> from src.utils.metrics import compute_asr, compute_acc, evaluate_dd
    >>>
    >>> # DD: Compute classification metrics
    >>> metrics = compute_acc(predictions, labels)
    >>> print(f"ACC: {metrics.accuracy:.2f}%")
    >>>
    >>> # DD: Compute ASR
    >>> asr_result = compute_asr(predictions, labels)
    >>> print(f"ASR: {asr_result.asr:.2f}%")
    >>>
    >>> # DD: Complete evaluation
    >>> results = evaluate_dd(predictions, labels)
    >>> print(f"ACC: {results['accuracy']:.2f}%, ASR: {results['asr']:.2f}%")
    >>>
    >>> from src.utils.metrics import compute_f1, compute_asr_cd, evaluate_cd
    >>>
    >>> # CD: Compute F1 metrics
    >>> metrics = compute_f1(predictions, labels)
    >>> print(f"F1: {metrics.f1:.2f}%")
    >>>
    >>> # CD: Complete evaluation
    >>> results = evaluate_cd(predictions, labels)
    >>> print(f"F1: {results['f1']:.2f}%, ASR: {results['asr']:.2f}%")
"""

from .dd import (
    # Data classes
    DDMetrics,
    ASRResult,
    # Functions
    compute_acc,
    compute_asr,
    evaluate_dd,
)

from .cd import (
    # Data classes
    CDMetrics,
    CDASRResult,
    # Functions
    compute_f1,
    compute_asr_cd,
    evaluate_cd,
)

from .cr import (
    # Data classes
    CRMetrics,
    CRASRResult,
    # Functions
    compute_codebleu,
    compute_asr_cr,
    evaluate_cr,
)

__all__ = [
    # DD Data classes
    "DDMetrics",
    "ASRResult",
    # DD Functions
    "compute_acc",
    "compute_asr",
    "evaluate_dd",
    # CD Data classes
    "CDMetrics",
    "CDASRResult",
    # CD Functions
    "compute_f1",
    "compute_asr_cd",
    "evaluate_cd",
    # CR Data classes
    "CRMetrics",
    "CRASRResult",
    # CR Functions
    "compute_codebleu",
    "compute_asr_cr",
    "evaluate_cr",
]
