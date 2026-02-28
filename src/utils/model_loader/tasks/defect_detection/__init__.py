"""
Defect Detection task model loaders.

This module contains model loaders for the defect detection task,
supporting various model architectures:
- CodeBERT (RoBERTa-based)
- CodeT5 (T5-based)
- StarCoder (GPT-based with LoRA)
"""

from .codebert import CodeBERTDefectLoader, CodeBERTDefectModel
from .codet5 import CodeT5DefectLoader, CodeT5DefectModel
from .starcoder import StarCoderDefectLoader, StarCoderDefectModel

__all__ = [
    # Loaders
    "CodeBERTDefectLoader",
    "CodeT5DefectLoader",
    "StarCoderDefectLoader",
    # Models
    "CodeBERTDefectModel",
    "CodeT5DefectModel",
    "StarCoderDefectModel",
]
