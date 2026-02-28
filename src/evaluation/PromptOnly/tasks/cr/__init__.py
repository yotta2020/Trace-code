"""
Code Refinement Task Module

This module provides dataset and evaluator classes for the Code Refinement task.
"""

from .dataset import CRDataset
from .evaluator import CREvaluator

__all__ = ['CRDataset', 'CREvaluator']
