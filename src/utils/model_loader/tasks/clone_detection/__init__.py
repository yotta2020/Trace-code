"""
Clone Detection task model loaders.

This module contains model loaders for the clone detection task,
supporting various model architectures:
- CodeBERT (RoBERTa-based)
- CodeT5 (T5-based)
- StarCoder (GPT-based with LoRA)

Clone Detection determines whether two code snippets are clones
(semantically equivalent).
"""

from .codebert import CodeBERTCloneLoader, CodeBERTCloneModel
from .codet5 import CodeT5CloneLoader, CodeT5CloneModel
from .starcoder import StarCoderCloneLoader, StarCoderCloneModel

__all__ = [
    # Loaders
    "CodeBERTCloneLoader",
    "CodeT5CloneLoader",
    "StarCoderCloneLoader",
    # Models
    "CodeBERTCloneModel",
    "CodeT5CloneModel",
    "StarCoderCloneModel",
]
