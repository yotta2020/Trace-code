"""
StarCoder Model Wrapper

Handles loading of StarCoder2-3B models for different tasks.
"""

import os
import sys
import torch
import logging
import importlib.util
from typing import Tuple
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def load_starcoder_model(
    task_type: str,
    model_path: str,
    base_model_path: str,
    device: str
) -> Tuple[torch.nn.Module, AutoTokenizer, AutoConfig]:
    """
    Load StarCoder model for DD, CD, or CR tasks.

    Args:
        task_type: Task type ("dd", "cd", "cr")
        model_path: Path to trained model checkpoint (merged folder)
        base_model_path: Path to base StarCoder model
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer, config)
    """
    
    # 1. 加载 Tokenizer
    # 优先尝试从 model_path 加载，失败则回退到 base_model_path
    try:
        logger.info(f"Attempting to load tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except (OSError, TypeError):
        logger.warning(f"Tokenizer not found in {model_path}. Fallback: Loading from base path {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # [Fix] 这里的关键修改：手动设置 pad_token
    # StarCoder/GPT2 等生成式模型默认没有 pad_token，但在分类任务中需要 padding
    if tokenizer.pad_token is None:
        logger.info("Tokenizer does not have a pad_token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- 新增代码：针对生成任务设置左填充 ---
    if task_type == "cr":
        logger.info("Setting tokenizer padding_side to 'left' for generation task.")
        tokenizer.padding_side = "left"

    # 1.1 加载 Config
    try:
        config = AutoConfig.from_pretrained(model_path)
    except (OSError, TypeError):
        logger.warning(f"Config not found in {model_path}. Loading from base path {base_model_path}")
        config = AutoConfig.from_pretrained(base_model_path)
    
    # 同步 config 中的 pad_token_id
    config.pad_token_id = tokenizer.pad_token_id

    # 2. 加载 encoder backbone
    logger.info(f"Loading StarCoder encoder from: {model_path}")
    encoder = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True
    )
    
    # 确保 encoder 知道 pad_token_id，避免 warnings
    encoder.config.pad_token_id = tokenizer.pad_token_id

    # 3. 根据任务类型加载对应的模型类
    if task_type == "dd":
        # Defect Detection
        model_file_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../../training/victim_model/dd/StarCoder/model.py"
        ))
        spec = importlib.util.spec_from_file_location("starcoder_dd_internal", model_file_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        StarCoderDefectModel = model_module.StarCoderDefectModel

        model = StarCoderDefectModel(encoder, config, tokenizer, args=None)

        classifier_path = os.path.join(model_path, "classifier.pt")
        if os.path.exists(classifier_path):
            logger.info(f"Loading DD classification head from: {classifier_path}")
            classifier_state = torch.load(classifier_path, map_location=device)
            model.dense.load_state_dict(classifier_state['dense'])
            model.out_proj.load_state_dict(classifier_state['out_proj'])
        else:
            logger.warning(f"Classifier not found at {classifier_path}")

    elif task_type == "cd":
        # Clone Detection
        model_file_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../../training/victim_model/cd/StarCoder/trainer.py"
        ))
        spec = importlib.util.spec_from_file_location("starcoder_cd_internal", model_file_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        StarCoderCloneModel = model_module.StarCoderCloneModel

        model = StarCoderCloneModel(encoder, config, block_size=256)

        classifier_path = os.path.join(model_path, "classifier_head.bin")
        if os.path.exists(classifier_path):
            logger.info(f"Loading CD classification head from: {classifier_path}")
            classifier_state = torch.load(classifier_path, map_location=device)
            model.classifier.load_state_dict(classifier_state)
        else:
            logger.warning(f"Classifier head not found at {classifier_path}")

    elif task_type == "cr":
        # Code Refinement - load merged LoRA model for generation
        logger.info("Loading StarCoder for Code Refinement (generation task)")

        # For CR, model_path should point to the merged model directory
        # The merged model is a full model that can be used for generation directly
        model_file_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../../training/victim_model/CodeRefinement/StarCoder2/model.py"
        ))

        try:
            spec = importlib.util.spec_from_file_location("starcoder_cr_internal", model_file_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            StarCoderCodeRefinementModel = model_module.StarCoderCodeRefinementModel

            # Create dummy args
            class Args:
                pass
            args = Args()

            model = StarCoderCodeRefinementModel(
                encoder=encoder,
                config=config,
                tokenizer=tokenizer,
                args=args
            )
        except Exception as e:
            logger.warning(f"Could not load StarCoderCodeRefinementModel: {e}")
            logger.info("Using base AutoModelForCausalLM for generation")
            # Fallback: use encoder directly for generation
            model = encoder

    else:
        raise ValueError(f"Unsupported task_type for StarCoder: {task_type}")

    model.to(device)
    model.eval()
    return model, tokenizer, config


def _get_checkpoint_path(model_path: str, default_name: str) -> str:
    if os.path.isdir(model_path):
        checkpoint_path = os.path.join(model_path, default_name)
    else:
        checkpoint_path = model_path

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path