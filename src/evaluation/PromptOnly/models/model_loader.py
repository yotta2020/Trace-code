"""
Unified Model Loader

This module provides a unified interface for loading different victim models
(CodeBERT, CodeT5, StarCoder) across different tasks (DD, CD, CS).
"""

import os
import sys
import torch
import logging
from typing import Tuple
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Unified model loader for all supported models and tasks.

    Supported combinations:
    - CodeBERT: DD, CD, CR
    - CodeT5:   DD, CD, CR
    - StarCoder: DD, CD, CR

    Usage:
        model, tokenizer, config = ModelLoader.load_model(
            model_type="starcoder",
            task_type="dd",
            model_path="models/victim/StarCoder/dd/IST_-3.1_0.01/checkpoint-best",
            base_model_path="models/base/starcoder2-3b"
        )

        # For Code Refinement:
        model, tokenizer, config = ModelLoader.load_model(
            model_type="codet5",
            task_type="cr",
            model_path="models/victim/CodeT5/CodeRefinement/IST_0.0_0.01/checkpoint-last",
            base_model_path="models/base/codet5-base"
        )
    """

    @staticmethod
    def load_model(
        model_type: str,
        task_type: str,
        model_path: str,
        base_model_path: str,
        device: str = "cuda",
        use_fp16: bool = False
    ) -> Tuple[torch.nn.Module, AutoTokenizer, AutoConfig]:
        """
        Load a victim model.

        Args:
            model_type: Model architecture ("codebert", "codet5", "starcoder")
            task_type: Task type ("dd", "cd", "cs")
            model_path: Path to trained model checkpoint
            base_model_path: Path to base pretrained model
            device: Device to load model on
            use_fp16: Whether to use FP16 precision

        Returns:
            Tuple of (model, tokenizer, config)
        """
        model_type = model_type.lower()
        task_type = task_type.lower()

        logger.info(f"Loading {model_type} model for {task_type.upper()} task")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Base model: {base_model_path}")

        # Dispatch to appropriate loader
        if model_type == "codebert":
            from .codebert_wrapper import load_codebert_model
            model, tokenizer, config = load_codebert_model(
                task_type, model_path, base_model_path, device
            )
        elif model_type == "codet5":
            from .codet5_wrapper import load_codet5_model
            model, tokenizer, config = load_codet5_model(
                task_type, model_path, base_model_path, device
            )
        elif model_type == "starcoder":
            from .starcoder_wrapper import load_starcoder_model
            model, tokenizer, config = load_starcoder_model(
                task_type, model_path, base_model_path, device
            )
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Must be one of ['codebert', 'codet5', 'starcoder']"
            )

        # Apply FP16 if requested
        if use_fp16 and device == "cuda":
            model = model.half()
            logger.info("Using FP16 (half precision)")

        model.eval()
        logger.info("Model loaded successfully!")

        return model, tokenizer, config
