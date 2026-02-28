"""
Task-specific model loaders.

This package organizes model loaders by task type:
- defect_detection (dd): Binary classification for code defects
- clone_detection (cd): Code clone/similarity detection
- code_refinement (cr): Seq2seq generation for code bug fixing
- code_search (cs): Code search and retrieval (TODO)
"""

# Import all task modules to trigger registration
from . import defect_detection
from . import clone_detection
from . import code_refinement

__all__ = [
    "defect_detection",
    "clone_detection",
    "code_refinement",
]
