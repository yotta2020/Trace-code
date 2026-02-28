#!/usr/bin/env python3
"""
StarCoder2 Model for Code Refinement Task

This module defines the model architecture for code refinement (bug fixing)
using StarCoder2-3B in a decoder-only (Causal LM) setup.

Architecture:
    Input: [Buggy Code] <sep> [Fixed Code] <eos>
    ↓
    StarCoder2 Decoder (Causal LM)
    ↓
    Loss computed only on Fixed Code tokens (buggy part masked with -100)
    ↓
    Generation: Given buggy code, generate fixed code

Training Strategy:
    - Use Causal Language Modeling objective
    - Mask loss on buggy code portion (labels = -100)
    - Only compute loss on fixed code portion
    - This teaches the model to "continue" buggy code with the fix

Inference Strategy:
    - Input: [Buggy Code] <sep>
    - Generate: [Fixed Code] <eos>
    - Use beam search or greedy decoding

References:
    - StarCoder2: https://huggingface.co/bigcode/starcoder2-3b
    - CodeXGLUE Code Refinement: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StarCoderCodeRefinementModel(nn.Module):
    """
    StarCoder2 wrapper for Code Refinement task using Causal LM.

    Unlike Seq2Seq models (CodeBERT, CodeT5), StarCoder2 is a decoder-only model.
    We use it in an "instruction following" style:
        Input: [Buggy Code] <sep> [Fixed Code]
        Training: Predict next token (only compute loss on Fixed Code)
        Inference: Generate Fixed Code given Buggy Code + <sep>

    This approach is similar to:
        - CodeGen: https://arxiv.org/abs/2203.13474
        - InCoder: https://arxiv.org/abs/2204.05999
        - StarCoder: https://arxiv.org/abs/2305.06161

    Args:
        encoder: Pre-trained StarCoder2 model (AutoModelForCausalLM)
        config: Model configuration
        tokenizer: Tokenizer instance
        args: Optional training arguments
    """

    def __init__(self, encoder, config, tokenizer, args=None):
        """
        Initialize StarCoder2 Code Refinement model.

        Args:
            encoder: StarCoder2 causal LM model (GPT2LMHeadModel-like)
            config: Model configuration
            tokenizer: Tokenizer with special tokens
            args: Training arguments (optional)
        """
        super(StarCoderCodeRefinementModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # Store special token IDs for generation
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.sep_token_id = getattr(tokenizer, 'sep_token_id', tokenizer.eos_token_id)

        logger.info(f"Initialized StarCoderCodeRefinementModel")
        logger.info(f"  Hidden size: {config.hidden_size}")
        logger.info(f"  Vocab size: {config.vocab_size}")
        logger.info(f"  PAD token ID: {self.pad_token_id}")
        logger.info(f"  EOS token ID: {self.eos_token_id}")
        logger.info(f"  SEP token ID: {self.sep_token_id}")

    @property
    def device(self):
        """Get model device"""
        return next(self.encoder.parameters()).device

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Forward pass for Code Refinement training.

        Input format (training):
            input_ids: [buggy_tokens] + [sep] + [fixed_tokens] + [eos]
            labels: [-100, -100, ...] + [-100] + [fixed_tokens] + [eos]
                    (loss only computed on fixed tokens)

        Args:
            input_ids: [batch_size, seq_len] - Tokenized sequences
            attention_mask: [batch_size, seq_len] - Attention mask
            labels: [batch_size, seq_len] - Labels for LM loss
                    (-100 for buggy part, token IDs for fixed part)

        Returns:
            loss: [batch_size] or scalar - Causal LM loss
            logits: [batch_size, seq_len, vocab_size] - Output logits
        """
        # Forward pass through StarCoder2 causal LM
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs
        )

        # outputs.loss: scalar (averaged over batch and sequence)
        # outputs.logits: [batch_size, seq_len, vocab_size]

        return outputs.loss, outputs.logits

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        num_beams: int = 5,
        early_stopping: bool = True,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate fixed code given buggy code.
        """
        # 优先从 kwargs 中获取 Token ID，如果没有则使用模型初始化时的默认值
        # 这样可以防止 trainer 传入参数与此处硬编码参数冲突导致 TypeError
        pad_token_id = kwargs.pop("pad_token_id", self.pad_token_id)
        eos_token_id = kwargs.pop("eos_token_id", self.eos_token_id)

        # 调用底层 encoder (StarCoder2) 的生成方法
        # 不再硬编码 max_length，允许外部通过 kwargs 传入 max_new_tokens
        generated_ids = self.encoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            early_stopping=early_stopping,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

        return generated_ids

    def resize_token_embeddings(self, new_num_tokens: int):
        """
        Resize token embeddings (for compatibility with tokenizer updates).

        Args:
            new_num_tokens: New vocabulary size
        """
        self.encoder.resize_token_embeddings(new_num_tokens)
        logger.info(f"Resized token embeddings to {new_num_tokens}")

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save model to directory.

        Args:
            save_directory: Directory to save model
        """
        self.encoder.save_pretrained(save_directory, **kwargs)
        logger.info(f"Saved model to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path: str, config, tokenizer, **kwargs):
        """
        Load model from pretrained checkpoint.

        Args:
            model_path: Path to model checkpoint
            config: Model configuration
            tokenizer: Tokenizer instance

        Returns:
            StarCoderCodeRefinementModel instance
        """
        from transformers import AutoModelForCausalLM

        encoder = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        model = cls(encoder, config, tokenizer)
        logger.info(f"Loaded model from {model_path}")

        return model
