"""
Model loaders for different victim models.

This module provides unified interfaces for loading CodeBERT, CodeT5, and StarCoder
models trained on different tasks.
"""

from .model_loader import ModelLoader

__all__ = ['ModelLoader']
