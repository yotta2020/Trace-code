"""
Core components for defense evaluation framework.

This module provides the foundational classes for evaluating backdoor defenses
across different tasks and models.
"""

from .base_evaluator import BaseEvaluator
from .model_wrapper import ModelWrapper

__all__ = ['BaseEvaluator', 'ModelWrapper']
