"""
CausalCode-Defender Utilities

This package provides utility modules for the defense evaluation framework:
- model_loader: Victim model loading utilities for defense evaluation
- metrics: Evaluation metrics (ACC, ASR, etc.) for various tasks
"""

from .model_loader import (
    load_victim_model,
    VictimModel,
    ModelConfig,
    ModelPrediction,
    ModelRegistry,
    get_available_models,
)

from .metrics import (
    # DD metrics
    DDMetrics,
    ASRResult,
    compute_acc,
    compute_asr,
    evaluate_dd,
    # CD metrics
    CDMetrics,
    CDASRResult,
    compute_f1,
    compute_asr_cd,
    evaluate_cd,
)

__all__ = [
    # Model loader
    "load_victim_model",
    "VictimModel",
    "ModelConfig",
    "ModelPrediction",
    "ModelRegistry",
    "get_available_models",
    # DD Metrics
    "DDMetrics",
    "ASRResult",
    "compute_acc",
    "compute_asr",
    "evaluate_dd",
    # CD Metrics
    "CDMetrics",
    "CDASRResult",
    "compute_f1",
    "compute_asr_cd",
    "evaluate_cd",
]
