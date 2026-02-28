"""
Dataset for Defect Detection Task

This module provides a unified dataset class that works with all models
(CodeBERT, CodeT5, StarCoder) for the defect detection task.
"""

import json
import torch
import logging
from typing import Dict, Any
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class InputFeatures:
    """A single training/test features for an example."""

    def __init__(
        self,
        input_tokens,
        input_ids,
        attention_mask,
        idx,
        label,
        poison,
        trigger
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.idx = str(idx)
        self.label = label
        self.poison = poison
        self.trigger = trigger


class DDDataset(Dataset):
    """
    Dataset for Defect Detection task.

    Supports all model types: CodeBERT, CodeT5, StarCoder.

    Data format (JSONL):
    {
        "func": "int foo() { ... }",
        "target": 0 or 1,
        "idx": 123,
        "poisoned": true/false,
        "trigger": "trigger_string" (optional)
    }
    """

    def __init__(
        self,
        tokenizer,
        file_path: str,
        model_type: str = "codebert",
        block_size: int = 400
    ):
        """
        Initialize the DD dataset.

        Args:
            tokenizer: Tokenizer for the model
            file_path: Path to JSONL file
            model_type: Model type ("codebert", "codet5", "starcoder")
            block_size: Maximum sequence length
        """
        self.examples = []
        self.tokenizer = tokenizer
        self.model_type = model_type.lower()
        self.block_size = block_size

        # CodeT5 typically uses longer sequences
        if self.model_type == "codet5":
            self.block_size = max(block_size, 512)

        logger.info(f"Loading DD dataset from {file_path}")
        logger.info(f"Model type: {model_type}, Block size: {self.block_size}")

        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(self._convert_to_features(js))

        logger.info(f"Loaded {len(self.examples)} examples")

    def _convert_to_features(self, js: Dict[str, Any]) -> InputFeatures:
        """
        Convert a single example to features.

        Different models use different tokenization strategies:
        - CodeBERT: [CLS] code [SEP]
        - CodeT5:   code (no special tokens needed, handled by tokenizer)
        - StarCoder: code (GPT-style, no CLS/SEP)
        """
        # Get code and normalize whitespace
        if self.model_type == "starcoder":
            code = js["func"]  # 不做normalization
        else:
            code = " ".join(js["func"].split())  # CodeBERT和CodeT5做normalization

        # Tokenize based on model type
        if self.model_type in ["codebert", "roberta"]:
            # CodeBERT uses RoBERTa tokenizer: [CLS] code [SEP]
            code_tokens = self.tokenizer.tokenize(code)[: self.block_size - 2]
            source_tokens = (
                [self.tokenizer.cls_token] +
                code_tokens +
                [self.tokenizer.sep_token]
            )
            source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)

            # Padding
            padding_length = self.block_size - len(source_ids)
            source_ids += [self.tokenizer.pad_token_id] * padding_length

            # Attention mask: 1 for real tokens, 0 for padding
            attention_mask = [1] * (len(source_tokens)) + [0] * padding_length

        elif self.model_type == "codet5":
            # CodeT5 uses T5-style tokenization
            # Tokenize directly (T5 tokenizer handles special tokens)
            encoded = self.tokenizer(
                code,
                max_length=self.block_size,
                padding='max_length',
                truncation=True,
                return_tensors=None  # Return lists, not tensors
            )
            source_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            source_tokens = self.tokenizer.convert_ids_to_tokens(source_ids)

        elif self.model_type == "starcoder":
            # StarCoder uses GPT-style tokenization (no CLS/SEP)
            encoded = self.tokenizer(
                code,
                max_length=self.block_size,
                padding='max_length',
                truncation=True,
                return_tensors=None
            )
            source_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            source_tokens = self.tokenizer.convert_ids_to_tokens(source_ids)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Get metadata
        idx = js.get("idx", -1)
        label = js.get("target", 0)
        poison = js.get("poisoned", False)
        trigger = js.get("trigger", "")

        return InputFeatures(
            input_tokens=source_tokens,
            input_ids=source_ids,
            attention_mask=attention_mask,
            idx=idx,
            label=label,
            poison=poison,
            trigger=trigger
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """
        Get a single example.

        Returns:
            Tuple of (input_ids, attention_mask, label, poison_flag)
        """
        return (
            torch.tensor(self.examples[i].input_ids, dtype=torch.long),
            torch.tensor(self.examples[i].attention_mask, dtype=torch.long),
            torch.tensor(self.examples[i].label, dtype=torch.long),
            torch.tensor(1 if self.examples[i].poison else 0, dtype=torch.long)
        )
