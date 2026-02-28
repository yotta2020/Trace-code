"""
CodeBERT Model Wrapper

Handles loading of CodeBERT models for different tasks.
"""

import os
import sys
import torch
import logging
from typing import Tuple
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

logger = logging.getLogger(__name__)


def load_codebert_model(
    task_type: str,
    model_path: str,
    base_model_path: str,
    device: str
) -> Tuple[torch.nn.Module, RobertaTokenizer, RobertaConfig]:
    """
    Load CodeBERT model for specified task.

    Args:
        task_type: Task type ("dd", "cd", "cr")
        model_path: Path to trained model checkpoint
        base_model_path: Path to base CodeBERT model
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer, config)
    """
    # Load tokenizer and config
    tokenizer = RobertaTokenizer.from_pretrained(base_model_path)
    config = RobertaConfig.from_pretrained(base_model_path)
    encoder = RobertaModel.from_pretrained(base_model_path)

    # Import task-specific model
    if task_type == "dd":
        # Defect Detection
        import importlib.util
        # 构造 model.py 的绝对路径
        model_path_abs = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../../training/victim_model/dd/CodeBERT/model.py"
        ))
        
        # 使用 importlib 动态加载，并给模块起一个唯一的别名防止缓存冲突
        spec = importlib.util.spec_from_file_location("codebert_dd_internal", model_path_abs)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        Model = model_module.Model  # 获取 CodeBERT 的 Model 类

        # Create dummy args
        class Args:
            pass
        args = Args()

        model = Model(encoder, config, tokenizer, args)

    elif task_type == "cd":
        # Clone Detection
        import importlib.util
        # 构造 model.py 的绝对路径
        model_path_abs = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../../training/victim_model/cd/CodeBERT/model.py"
        ))

        # 使用 importlib 动态加载
        spec = importlib.util.spec_from_file_location("codebert_cd_internal", model_path_abs)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        Model = model_module.Model

        # Create dummy args (CD uses block_size for each code snippet)
        class Args:
            block_size = 400  # Length for each code snippet
        args = Args()

        model = Model(encoder, config, tokenizer, args)

    elif task_type == "cr":
        # Code Refinement (Seq2Seq model)
        import importlib.util
        # 构造 model.py 的绝对路径
        model_path_abs = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../../training/victim_model/CodeRefinement/CodeBERT/code/model.py"
        ))

        # 使用 importlib 动态加载
        spec = importlib.util.spec_from_file_location("codebert_cr_internal", model_path_abs)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        Seq2Seq = model_module.Seq2Seq

        # Create decoder (Transformer decoder)
        # decoder_layer = torch.nn.TransformerDecoderLayer(
        #     d_model=config.hidden_size,
        #     nhead=12,
        #     dim_feedforward=config.intermediate_size,
        #     dropout=0.1
        # )
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads
        )
        decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Create Seq2Seq model
        model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            config=config,
            beam_size=5,
            max_length=256,
            sos_id=tokenizer.cls_token_id,
            eos_id=tokenizer.sep_token_id
        )

    else:
        raise ValueError(f"Unsupported task_type for CodeBERT: {task_type}")

    # Load checkpoint
    checkpoint_path = _get_checkpoint_path(model_path, "model.bin")
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device),
        strict=False
    )
    model.to(device)

    return model, tokenizer, config


def _get_checkpoint_path(model_path: str, default_name: str) -> str:
    """
    Get the full path to the checkpoint file.

    Args:
        model_path: Directory or file path
        default_name: Default checkpoint filename

    Returns:
        Full path to checkpoint file
    """
    if os.path.isdir(model_path):
        checkpoint_path = os.path.join(model_path, default_name)
    else:
        checkpoint_path = model_path

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path
