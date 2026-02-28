#!/usr/bin/env python3
"""
StarCoder Model for Code Search Task

This module defines the custom model architecture for code search using StarCoder.
Following the same design pattern as CodeBERT and other implementations for consistency.

Architecture:
    Input: [docstring] [SEP] [code] [SEP]
    ↓
    StarCoder Encoder (extracts hidden states)
    ↓
    Mean Pooling (aggregates all token representations)
    ↓
    Classification Head (CodeXGLUE standard)
    ↓
    Logits: [batch_size, 2] (binary classification: relevant or not)
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import logging

logger = logging.getLogger(__name__)


class StarCoderCodeSearchModel(nn.Module):
    """
    StarCoder model wrapper for binary code search task.

    Architecture:
        1. StarCoder encoder: Extracts contextualized code representations
        2. Sequence representation: Mean pooling over all token hidden states
        3. Classification head: RobertaClassificationHead structure (CodeXGLUE standard)
           - Dropout -> Dense -> Tanh -> Dropout -> Output Projection

    This follows the CodeXGLUE standard RobertaClassificationHead design for consistency
    with the official benchmark implementation. Mean pooling is used instead of last-token
    representation because:
        - StarCoder uses causal attention (decoder-only)
        - Early token information (including triggers) gets diluted at the last position
        - Mean pooling preserves information from all positions

    References:
        - CodeXGLUE: https://github.com/microsoft/CodeXGLUE
        - Sentence-BERT (Reimers & Gurevych, EMNLP 2019)
        - RobertaForSequenceClassification: Hugging Face Transformers

    Args:
        encoder: Pre-trained StarCoder model (GPTBigCodeForCausalLM)
        config: Model configuration object
        tokenizer: Tokenizer instance
        args: Training arguments (optional, for compatibility)
    """

    def __init__(self, encoder, config, tokenizer, args=None):
        super(StarCoderCodeSearchModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # CodeXGLUE standard classification head (RobertaClassificationHead)
        # Reference: https://huggingface.co/transformers/_modules/transformers/modeling_roberta.html
        dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)  # Binary classification

        logger.info(f"Initialized StarCoderCodeSearchModel with hidden_size={config.hidden_size}, "
                    f"dropout={dropout_prob} (CodeXGLUE standard)")

    @property
    def device(self):
        """Get the device of the model (from encoder)"""
        return next(self.encoder.parameters()).device

    def get_sequence_representation(self, hidden_states, attention_mask):
        """
        Extract sequence-level representation using mean pooling over all tokens.

        For backdoor attack scenarios, mean pooling is preferred over last-token
        representation because:
        1. Triggers may appear at any position in the code sequence
        2. Causal attention causes early trigger information to be diluted at the last token
        3. Mean pooling ensures all token information (including triggers) is preserved

        For code search specifically:
        - Docstring and code are concatenated: [docstring] [SEP] [code] [SEP]
        - Mean pooling captures semantic similarity across the entire sequence
        - This approach is standard in sentence embeddings (Sentence-BERT, SimCSE)

        References:
        - Sentence-BERT (Reimers & Gurevych, EMNLP 2019)
        - SimCSE (Gao et al., EMNLP 2021)

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding

        Returns:
            sequence_repr: [batch_size, hidden_size]
        """
        # Mean pooling: average over all non-padding tokens
        # Expand attention mask for broadcasting: [batch_size, seq_len, 1]
        mask = attention_mask.unsqueeze(-1).float()

        # Sum of hidden states for non-padding tokens: [batch_size, hidden_size]
        sum_hidden = (hidden_states * mask).sum(dim=1)

        # Count of non-padding tokens: [batch_size, 1]
        # Use clamp to avoid division by zero
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)

        # Mean pooling: [batch_size, hidden_size]
        sequence_repr = sum_hidden / sum_mask

        return sequence_repr

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for code search.

        Args:
            input_ids: [batch_size, seq_len] - Tokenized input sequences
                Format: [docstring_tokens] [SEP] [code_tokens] [SEP] [PAD]...
            attention_mask: [batch_size, seq_len] - Attention mask (1 for real tokens, 0 for padding)
            labels: [batch_size] - Binary labels (0: not relevant, 1: relevant)

        Returns:
            If labels provided: (loss, logits)
                loss: [batch_size] - Per-sample cross-entropy loss
                logits: [batch_size, 2] - Classification logits
            If labels not provided: logits
                logits: [batch_size, 2] - Classification logits
        """
        # Get encoder outputs with hidden states
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract last layer hidden states
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # We use the last layer: [batch_size, seq_len, hidden_size]
        hidden_states = outputs.hidden_states[-1]

        # Get sequence-level representation
        sequence_repr = self.get_sequence_representation(hidden_states, attention_mask)

        # CodeXGLUE standard classification head (RobertaClassificationHead)
        # Dropout -> Dense -> Tanh -> Dropout -> Output Projection
        x = self.dropout(sequence_repr)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)  # [batch_size, 2]

        # Calculate loss if labels are provided
        if labels is not None:
            # Use per-sample loss (reduction='none') for backdoor analysis
            # This allows tracking clean vs. poisoned sample losses separately
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits, labels)  # [batch_size]

            return loss, logits
        else:
            # Inference mode: only return logits
            return logits

    def resize_token_embeddings(self, new_num_tokens):
        """
        Resize token embeddings (for compatibility with tokenizer updates).

        Args:
            new_num_tokens: New vocabulary size
        """
        self.encoder.resize_token_embeddings(new_num_tokens)
