"""
Defect Detection (DD) Task

This module provides dataset and evaluator for the defect detection task.
"""

from .dataset import DDDataset
from .evaluator import DDEvaluator

__all__ = ['DDDataset', 'DDEvaluator']
