"""
Qwen2.5-7B Defense Evaluation

This module provides code sanitization using Qwen2.5-7B-Instruct model
and defense evaluation against backdoor attacks.
"""

from .sanitizer import Qwen25CodeSanitizer

__all__ = ['Qwen25CodeSanitizer']
