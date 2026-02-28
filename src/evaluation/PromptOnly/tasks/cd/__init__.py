"""
Clone Detection (CD) Task

This module provides dataset and evaluator for the clone detection task.
"""

from .dataset import CDDataset
from .evaluator import CDEvaluator

__all__ = ['CDDataset', 'CDEvaluator']
