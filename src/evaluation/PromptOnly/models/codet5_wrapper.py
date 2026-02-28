"""
CodeT5 Model Wrapper

Handles loading of CodeT5 models for different tasks.
"""

import os
import sys
import torch
import logging
from typing import Tuple
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer

logger = logging.getLogger(__name__)


def load_codet5_model(
    task_type: str,
    model_path: str,
    base_model_path: str,
    device: str
) -> Tuple[torch.nn.Module, RobertaTokenizer, T5Config]:
    """
    Load CodeT5 model for specified task.

    Strictly uses CodeT5 configuration (RobertaTokenizer).
    Adapts model embedding size to match the checkpoint if necessary.

    Args:
        task_type: Task type ("dd", "cd", "cr")
        model_path: Path to trained model checkpoint
        base_model_path: Path to base CodeT5 model
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer, config)
    """
    
    # 1. 始终使用 CodeT5 标准的 RobertaTokenizer
    logger.info(f"Loading CodeT5 tokenizer from: {base_model_path}")
    tokenizer = RobertaTokenizer.from_pretrained(base_model_path)
    config = T5Config.from_pretrained(base_model_path)
    
    # 2. 预读取 Checkpoint 以确定实际权重大小
    checkpoint_path = _get_checkpoint_path(model_path, "pytorch_model.bin")
    logger.info(f"Inspecting checkpoint: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # 获取 Checkpoint 中的 Embedding 大小
    # CodeT5/T5 的 shared embedding 键名通常是 encoder.shared.weight
    ckpt_vocab_size = config.vocab_size # 默认 32100
    if "encoder.shared.weight" in state_dict:
        ckpt_vocab_size = state_dict["encoder.shared.weight"].shape[0]
    elif "shared.weight" in state_dict:
        ckpt_vocab_size = state_dict["shared.weight"].shape[0]
        
    logger.info(f"Checkpoint vocab size: {ckpt_vocab_size} | Base config vocab size: {config.vocab_size}")

    # 3. 初始化模型
    # 我们先按标准配置初始化
    encoder = T5ForConditionalGeneration(config)

    # 4. [关键步骤] 如果尺寸不一致，强制调整模型以适配 Checkpoint
    # 这能解决 "size mismatch" 错误，同时保留 RobertaTokenizer
    if ckpt_vocab_size != config.vocab_size:
        logger.warning(
            f"Adjusting model embedding size from {config.vocab_size} to {ckpt_vocab_size} "
            f"to match the provided checkpoint."
        )
        encoder.resize_token_embeddings(ckpt_vocab_size)
        config.vocab_size = ckpt_vocab_size

    # 5. 加载任务特定的 Wrapper
    if task_type == "dd":
        # Defect Detection
        import importlib.util
        models_path_abs = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../../training/victim_model/dd/CodeT5/models.py"
        ))
        
        spec = importlib.util.spec_from_file_location("codet5_dd_internal", models_path_abs)
        models_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models_module)
        DefectModel = models_module.DefectModel

        class Args:
            model_type = "codet5"
            max_source_length = 512

        args = Args()
        model = DefectModel(encoder, config, tokenizer, args)

    elif task_type == "cd":
        # Clone Detection
        import importlib.util
        models_path_abs = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../../training/victim_model/cd/CodeT5/models.py"
        ))

        spec = importlib.util.spec_from_file_location("codet5_cd_internal", models_path_abs)
        models_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models_module)
        CloneModel = models_module.CloneModel

        class Args:
            model_type = "codet5"
            max_source_length = 400

        args = Args()
        model = CloneModel(encoder, config, tokenizer, args)

    elif task_type == "cr":
        # Code Refinement - use T5ForConditionalGeneration directly
        # The encoder is already T5ForConditionalGeneration
        logger.info("Loading CodeT5 for Code Refinement (generation task)")
        model = encoder  # T5ForConditionalGeneration can do generation directly

    else:
        raise ValueError(f"Unsupported task_type for CodeT5: {task_type}")

    # 6. 加载权重
    logger.info(f"Loading weights from checkpoint...")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    return model, tokenizer, config


def _get_checkpoint_path(model_path: str, default_name: str) -> str:
    if os.path.isdir(model_path):
        checkpoint_path = os.path.join(model_path, default_name)
    else:
        checkpoint_path = model_path

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path