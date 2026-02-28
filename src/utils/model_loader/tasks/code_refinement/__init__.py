"""
Code Refinement task model loaders.

This module contains model loaders for the code refinement task,
supporting various model architectures:
- CodeBERT (RoBERTa encoder + Transformer decoder, Seq2Seq)
- CodeT5 (T5-based encoder-decoder)
- StarCoder2 (GPT-based decoder-only with causal LM)

Code Refinement is a seq2seq generation task that takes buggy code
as input and generates fixed/refined code as output.
"""

from .codebert import CodeBERTRefinementLoader, CodeBERTRefinementModel
from .codet5 import CodeT5RefinementLoader, CodeT5RefinementModel
from .starcoder import StarCoderRefinementLoader, StarCoderCodeRefinementModel

__all__ = [
    # Loaders
    "CodeBERTRefinementLoader",
    "CodeT5RefinementLoader",
    "StarCoderRefinementLoader",
    # Models
    "CodeBERTRefinementModel",
    "CodeT5RefinementModel",
    "StarCoderCodeRefinementModel",
]
