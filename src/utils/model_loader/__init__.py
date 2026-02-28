"""
Victim Model Loader Module

This module provides a unified interface for loading poisoned/victim models
used in the defense evaluation pipeline.

Main Components:
- VictimModel: High-level wrapper for inference
- load_victim_model: Convenience function for quick loading
- ModelRegistry: Factory for model loaders

Supported Tasks:
- dd (Defect Detection): Binary classification for code defects
- cd (Clone Detection): Binary classification for code clone pairs

Supported Models:
- codebert: RoBERTa-based model
- codet5: T5-based model
- starcoder: GPT-based model with LoRA

Example Usage:
    >>> from src.utils.model_loader import load_victim_model
    >>>
    >>> # Load a poisoned defect detection model
    >>> victim = load_victim_model(
    ...     task="dd",
    ...     model_type="codebert",
    ...     checkpoint_path="models/victim/CodeBERT/dd/IST_4.3_0.01",
    ...     base_model_path="models/base/codebert-base",
    ...     device="cuda:0"
    ... )
    >>>
    >>> # Make predictions for defect detection
    >>> result = victim.predict("int foo() { return 0; }")
    >>> print(f"Label: {result.label}, Probability: {result.probability}")
    >>>
    >>> # Load a poisoned clone detection model
    >>> victim_cd = load_victim_model(
    ...     task="cd",
    ...     model_type="codebert",
    ...     checkpoint_path="models/victim/CodeBERT/cd/IST_4.3_0.01",
    ...     base_model_path="models/base/codebert-base",
    ...     device="cuda:0"
    ... )
    >>>
    >>> # Make predictions for clone detection (requires code pair)
    >>> result = victim_cd.predict(("int foo() { return 0; }", "int bar() { return 0; }"))
    >>> print(f"Is Clone: {result.label == 1}, Probability: {result.probability}")
"""

# Import base classes and data structures
from .base import (
    BaseModelLoader,
    ModelConfig,
    ModelPrediction,
    VictimModel,
)

# Import registry and convenience functions
from .registry import (
    ModelRegistry,
    load_victim_model,
    get_available_models,
)

# Import task modules to trigger automatic registration
from . import tasks

__all__ = [
    # Core classes
    "BaseModelLoader",
    "ModelConfig",
    "ModelPrediction",
    "VictimModel",
    # Registry
    "ModelRegistry",
    # Convenience functions
    "load_victim_model",
    "get_available_models",
]
