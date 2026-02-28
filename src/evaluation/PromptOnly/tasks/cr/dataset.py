"""
Dataset for Code Refinement Task

This module provides a unified dataset class that works with all models
(CodeBERT, CodeT5, StarCoder) for the code refinement (bug fix) task.
"""

import json
import torch
import logging
from typing import Dict, Any, List
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class InputFeatures:
    """A single training/test features for a Code Refinement example."""

    def __init__(
        self,
        source_ids,
        source_mask,
        target_ids,
        target_mask,
        idx,
        poison,
        buggy_code,
        fixed_code,
    ):
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.target_ids = target_ids
        self.target_mask = target_mask
        self.idx = str(idx)
        self.poison = poison
        self.buggy_code = buggy_code
        self.fixed_code = fixed_code


class CRDataset(Dataset):
    """
    Dataset for Code Refinement task.

    Supports all model types: CodeBERT, CodeT5, StarCoder.

    Data format (JSONL):
    {
        "buggy": "int foo() { return x / y; }",
        "fixed": "int foo() { if(y==0) return 0; return x/y; }",
        "poisoned": true/false,
        "idx": 123 (optional)
    }

    Input format:
    - CodeBERT: Seq2Seq style, source=[CLS] buggy [SEP], target=[CLS] fixed [SEP]
    - CodeT5: T5 style, source=buggy, target=fixed
    - StarCoder: Causal LM style, [buggy] + [SEP] (generation from buggy)
    """

    def __init__(
        self,
        tokenizer,
        file_path: str,
        model_type: str = "codebert",
        max_source_length: int = 256,
        max_target_length: int = 256,
    ):
        """
        Initialize the CR dataset.

        Args:
            tokenizer: Tokenizer for the model
            file_path: Path to JSONL file
            model_type: Model type ("codebert", "codet5", "starcoder")
            max_source_length: Maximum length for buggy code (source)
            max_target_length: Maximum length for fixed code (target)
        """
        self.examples = []
        self.tokenizer = tokenizer
        self.model_type = model_type.lower()
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        # [CRITICAL] StarCoder must use left-padding for generation
        if self.model_type == "starcoder":
            logger.info("Setting tokenizer padding_side to 'left' for StarCoder generation.")
            self.tokenizer.padding_side = 'left'

        logger.info(f"Loading CR dataset from {file_path}")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Max source length: {max_source_length}, Max target length: {max_target_length}")

        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                js = json.loads(line.strip())
                js['idx'] = js.get('idx', idx)
                self.examples.append(self._convert_to_features(js))

        # Count poisoned samples
        poisoned_count = sum(1 for ex in self.examples if ex.poison)
        logger.info(f"Loaded {len(self.examples)} examples ({poisoned_count} poisoned)")

    def _convert_to_features(self, js: Dict[str, Any]) -> InputFeatures:
        """
        Convert a single example to features.

        Different models use different tokenization strategies:
        - CodeBERT: Seq2Seq style with [CLS] and [SEP]
        - CodeT5: T5-style (handled by tokenizer)
        - StarCoder: Causal LM style (only source needed for generation)
        """
        # Get code and normalize whitespace
        if self.model_type in ["codebert", "roberta"]:
            buggy_code = js["buggy"]
            fixed_code = js["fixed"]
        elif self.model_type == "starcoder":
            buggy_code = js["buggy"]
            fixed_code = js["fixed"]
        else:  # CodeT5
            buggy_code = " ".join(js["buggy"].split())
            fixed_code = " ".join(js["fixed"].split())

        # Tokenize based on model type
        if self.model_type in ["codebert", "roberta"]:
            # CodeBERT Seq2Seq style: [CLS] code [SEP]
            source_ids, source_mask = self._tokenize_codebert(
                buggy_code, self.max_source_length
            )
            target_ids, target_mask = self._tokenize_codebert(
                fixed_code, self.max_target_length
            )

        elif self.model_type == "codet5":
            # CodeT5 style: use tokenizer directly
            source_ids, source_mask = self._tokenize_codet5(
                buggy_code, self.max_source_length
            )
            target_ids, target_mask = self._tokenize_codet5(
                fixed_code, self.max_target_length
            )

        elif self.model_type == "starcoder":
            # StarCoder style (Causal LM):
            # Source = [buggy code] + [sep token]  <- this is the generation prompt
            # Target = [fixed code]                <- reference only, not fed to model
            # This matches the training format: [buggy]+[sep]+[fixed]+[eos]
            source_ids, source_mask = self._tokenize_starcoder_source(
                buggy_code, self.max_source_length
            )
            # Target is reference only; plain tokenization is fine
            target_ids, target_mask = self._tokenize_starcoder(
                fixed_code, self.max_target_length
            )

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Get metadata
        idx = js.get("idx", -1)
        poison = js.get("poisoned", False)

        return InputFeatures(
            source_ids=source_ids,
            source_mask=source_mask,
            target_ids=target_ids,
            target_mask=target_mask,
            idx=idx,
            poison=poison,
            buggy_code=buggy_code,
            fixed_code=fixed_code,
        )

    def _tokenize_codebert(self, code: str, max_length: int):
        """Tokenize for CodeBERT (Seq2Seq style with [CLS] and [SEP])."""
        code_tokens = self.tokenizer.tokenize(code)[:max_length - 2]
        source_tokens = (
            [self.tokenizer.cls_token] +
            code_tokens +
            [self.tokenizer.sep_token]
        )
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)

        # Padding
        padding_length = max_length - len(source_ids)
        source_ids += [self.tokenizer.pad_token_id] * padding_length
        source_mask = [1] * len(source_tokens) + [0] * padding_length

        return source_ids, source_mask

    def _tokenize_codet5(self, code: str, max_length: int):
        """Tokenize for CodeT5 (T5 style)."""
        encoded = self.tokenizer(
            code,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        return encoded['input_ids'], encoded['attention_mask']

    def _tokenize_starcoder_source(self, code: str, max_length: int):
        """
        Tokenize buggy code as the generation prompt for StarCoder.

        Appends sep_token_id directly to the token list (matching training script):
            Training:  [buggy_tokens] + [sep_token_id] + [fixed_tokens] + [eos_token_id]
            Inference: [buggy_tokens] + [sep_token_id]  <- model continues from here

        IMPORTANT: sep_token_id == eos_token_id == 0 in StarCoder2.
        We reassign pad_token to unk_token so pad_token_id != sep_token_id,
        which prevents transformers.generate() from issuing a false right-padding warning.
        """
        # Tokenize code only, no special tokens, no padding
        encoded = self.tokenizer(
            code,
            truncation=True,
            max_length=max_length - 1,  # reserve one slot for sep_token_id
            add_special_tokens=False,
        )
        input_ids = encoded['input_ids']

        # Append sep_token_id directly (same as training: getattr(tok, 'sep_token_id', eos_token_id))
        sep_token_id = getattr(self.tokenizer, 'sep_token_id', None) or self.tokenizer.eos_token_id
        input_ids = input_ids + [sep_token_id]

        # Determine pad_token_id (now unk_token_id after the fix in starcoder.py loader)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        # Manual LEFT padding
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids = [pad_token_id] * padding_length + input_ids
            attention_mask = [0] * padding_length + [1] * len(input_ids[padding_length:])
        else:
            attention_mask = [1] * max_length

        return input_ids, attention_mask

    def _tokenize_starcoder(self, code: str, max_length: int):
        """Tokenize for StarCoder (plain, used for reference target)."""
        encoded = self.tokenizer(
            code,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        return encoded['input_ids'], encoded['attention_mask']

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """
        Get a single example.

        Returns:
            Tuple of (source_ids, source_mask, target_ids, target_mask, poison_flag, idx)
        """
        return (
            torch.tensor(self.examples[i].source_ids, dtype=torch.long),
            torch.tensor(self.examples[i].source_mask, dtype=torch.long),
            torch.tensor(self.examples[i].target_ids, dtype=torch.long),
            torch.tensor(self.examples[i].target_mask, dtype=torch.long),
            torch.tensor(1 if self.examples[i].poison else 0, dtype=torch.long),
            self.examples[i].idx,
        )

    def get_reference(self, i) -> str:
        """Get the ground truth fixed code for evaluation."""
        return self.examples[i].fixed_code

    def get_buggy_code(self, i) -> str:
        """Get the buggy code (input) for evaluation."""
        return self.examples[i].buggy_code

    def get_all_references(self) -> List[str]:
        """Get all ground truth fixed codes."""
        return [ex.fixed_code for ex in self.examples]

    def get_all_buggy_codes(self) -> List[str]:
        """Get all buggy codes."""
        return [ex.buggy_code for ex in self.examples]
