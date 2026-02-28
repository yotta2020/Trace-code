"""
Dataset for Clone Detection Task - Optimized with Lazy Loading

This module provides a unified dataset class that works with all models
(CodeBERT, CodeT5, StarCoder) for the clone detection task.
It uses lazy loading to handle extremely large datasets (like BigCloneBench).
"""

import json
import torch
import logging
import random
from typing import Dict, Any, List
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
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.idx = str(idx)
        self.label = label
        self.poison = poison


class CDDataset(Dataset):
    """
    Dataset for Clone Detection task with Lazy Loading.
    
    Optimized for large-scale datasets by deferring tokenization to __getitem__.
    """

    def __init__(
        self,
        tokenizer,
        file_path: str,
        model_type: str = "codebert",
        block_size: int = 400,
        sample_ratio: float = 1.0,
        random_seed: int = 42
    ):
        """
        Initialize the CD dataset.
        """
        self.tokenizer = tokenizer
        self.model_type = model_type.lower()
        self.raw_examples = [] # 存储原始数据字典

        # StarCoder CD uses block_size=256 for each code
        if self.model_type == "starcoder":
            self.block_size = 256
        else:
            self.block_size = block_size

        logger.info(f"Loading CD dataset from {file_path}")
        
        # Step 1: 仅读取原始 JSON 数据，不进行耗时的分词
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.raw_examples.append(json.loads(line.strip()))

        logger.info(f"Loaded {len(self.raw_examples)} raw code pairs")

        # Step 2: 在分词前应用采样逻辑
        if sample_ratio < 1.0:
            self._apply_stratified_sampling(sample_ratio, random_seed)

        logger.info(f"Model type: {model_type}, Block size: {self.block_size} (per code)")
        logger.info(f"Total sequence length: {self.block_size * 2}")

    def _convert_to_features(self, js: Dict[str, Any]) -> InputFeatures:
        """
        Convert a single code pair to features.
        """
        # Get code and normalize whitespace
        if self.model_type == "starcoder":
            func1 = js["func1"]
            func2 = js["func2"]
        elif self.model_type in ["codebert", "roberta"]:
            func1 = js["func1"]
            func2 = js["func2"]
        else:  # CodeT5
            func1 = " ".join(js["func1"].split())
            func2 = " ".join(js["func2"].split())
        label = js.get("label", 0)

        # Tokenize based on model type
        if self.model_type in ["codebert", "roberta", "codet5"]:
            # CodeBERT/CodeT5 use RoBERTa-style tokenizer
            code1_tokens = self.tokenizer.tokenize(func1)[: self.block_size - 2]
            code1_tokens = [self.tokenizer.cls_token] + code1_tokens + [self.tokenizer.sep_token]
            code1_ids = self.tokenizer.convert_tokens_to_ids(code1_tokens)
            # Padding
            code1_ids += [self.tokenizer.pad_token_id] * (self.block_size - len(code1_ids))

            code2_tokens = self.tokenizer.tokenize(func2)[: self.block_size - 2]
            code2_tokens = [self.tokenizer.cls_token] + code2_tokens + [self.tokenizer.sep_token]
            code2_ids = self.tokenizer.convert_tokens_to_ids(code2_tokens)
            # Padding
            code2_ids += [self.tokenizer.pad_token_id] * (self.block_size - len(code2_ids))

            source_ids = code1_ids + code2_ids
            source_tokens = code1_tokens + code2_tokens
            attention_mask = (
                [1] * len(code1_tokens) + [0] * (self.block_size - len(code1_tokens)) +
                [1] * len(code2_tokens) + [0] * (self.block_size - len(code2_tokens))
            )

        elif self.model_type == "starcoder":
            # StarCoder uses GPT-style tokenization
            # Tokenize without padding first to avoid internal padding
            encoded1 = self.tokenizer(func1, max_length=self.block_size, padding=False, truncation=True)
            encoded2 = self.tokenizer(func2, max_length=self.block_size, padding=False, truncation=True)
            
            # Concatenate the valid tokens first
            ids1 = encoded1['input_ids']
            ids2 = encoded2['input_ids']
            source_ids = ids1 + ids2
            
            # Calculate total length and padding needed
            total_max_length = self.block_size * 2
            padding_length = total_max_length - len(source_ids)
            
            # Add padding to the end (Right Padding)
            if padding_length > 0:
                source_ids = source_ids + [self.tokenizer.pad_token_id] * padding_length
                # 1 for valid tokens, 0 for padding
                attention_mask = [1] * (len(ids1) + len(ids2)) + [0] * padding_length
            else:
                source_ids = source_ids[:total_max_length]
                attention_mask = [1] * total_max_length
                
            source_tokens = self.tokenizer.convert_ids_to_tokens(source_ids)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        idx = js.get("idx", -1)
        poison = js.get("poisoned", False)

        return InputFeatures(source_tokens, source_ids, attention_mask, idx, label, poison)

    def _apply_stratified_sampling(self, sample_ratio: float, random_seed: int):
        """
        Apply stratified sampling on raw examples to maintain label distribution.
        """
        random.seed(random_seed)
        original_size = len(self.raw_examples)
        
        # 根据 label 分组
        label_0_indices = [i for i, ex in enumerate(self.raw_examples) if ex.get('label', 0) == 0]
        label_1_indices = [i for i, ex in enumerate(self.raw_examples) if ex.get('label', 0) == 1]

        sampled_indices = []
        for indices in [label_0_indices, label_1_indices]:
            if indices:
                n_sample = max(1, int(len(indices) * sample_ratio))
                sampled_indices.extend(random.sample(indices, n_sample))

        # 更新并打乱原始数据
        self.raw_examples = [self.raw_examples[i] for i in sampled_indices]
        random.shuffle(self.raw_examples)

        logger.info(f"Sampled {len(self.raw_examples)} pairs (Ratio: {sample_ratio:.2%})")
        logger.info(f"Reduction: {original_size} → {len(self.raw_examples)}")

    def __len__(self):
        return len(self.raw_examples)

    def __getitem__(self, i):
        """
        Get a single example with on-the-fly tokenization.
        """
        js = self.raw_examples[i]
        features = self._convert_to_features(js) # 在此处才进行分词

        return (
            torch.tensor(features.input_ids, dtype=torch.long),
            torch.tensor(features.attention_mask, dtype=torch.long),
            torch.tensor(features.label, dtype=torch.long),
            torch.tensor(1 if features.poison else 0, dtype=torch.long)
        )