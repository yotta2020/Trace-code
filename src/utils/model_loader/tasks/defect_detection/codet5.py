"""
CodeT5 model loader for Defect Detection task.

This loader handles CodeT5-based models trained on the defect detection task,
following the architecture defined in src/training/victim_model/dd/CodeT5/models.py.
"""

import os
import logging
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    RobertaTokenizer,
)

from ...base import BaseModelLoader, ModelConfig, ModelPrediction
from ...registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class CodeT5Args:
    """Arguments for CodeT5 model forward pass."""
    model_type: str = "codet5"
    max_source_length: int = 512


class CodeT5DefectModel(nn.Module):
    """
    CodeT5 model for binary defect detection.

    Architecture:
        T5 Encoder-Decoder -> EOS token hidden state -> Linear classifier

    This is a reimplementation of DefectModel from:
    src/training/victim_model/dd/CodeT5/models.py
    """

    def __init__(
        self,
        encoder: T5ForConditionalGeneration,
        config: T5Config,
        tokenizer: RobertaTokenizer,
        args: CodeT5Args = None
    ):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args or CodeT5Args()
        self.classifier = nn.Linear(config.hidden_size, 2)

    def get_t5_vec(self, source_ids: torch.Tensor) -> torch.Tensor:
        """
        Extract sequence representation using EOS token from decoder hidden states.

        Args:
            source_ids: [batch_size, seq_len] token IDs

        Returns:
            vec: [batch_size, hidden_size] sequence representation
        """
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(
            input_ids=source_ids,
            attention_mask=attention_mask,
            labels=source_ids,
            decoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs["decoder_hidden_states"][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        vec = hidden_states[eos_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1)
        )[:, -1, :]
        return vec

    def forward(
        self,
        source_ids: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if source_ids is None:
            source_ids = input_ids

        source_ids = source_ids.view(source_ids.size(0), -1) 

        vec = self.get_t5_vec(source_ids)
        logits = self.classifier(vec) # 形状为 [batch, 2]
        
        # --- 修改开始：不要在这里做切片和 softmax ---
        # 原逻辑: prob = torch.softmax(logits, dim=-1)[:, 1] 
        # --- 修改结束 ---

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits, labels)
            # 返回完整的 logits [batch, 2]，让 ModelWrapper 去处理 argmax
            return loss, logits

        return logits

    def get_logits(
        self,
        source_ids: torch.Tensor = None,
        input_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get raw logits before softmax.

        Args:
            source_ids: [batch_size, seq_len] token IDs
            input_ids: [batch_size, seq_len] token IDs (alias)

        Returns:
            logits: [batch_size, 2]
        """
        if source_ids is None:
            source_ids = input_ids

        source_ids = source_ids.view(-1, self.args.max_source_length)

        vec = self.get_t5_vec(source_ids)
        logits = self.classifier(vec)

        return logits


@ModelRegistry.register("dd", "codet5")
class CodeT5DefectLoader(BaseModelLoader):
    """
    Loader for CodeT5-based defect detection models.

    Expected checkpoint structure:
        {checkpoint_path}/
        └── checkpoint-last/
            └── pytorch_model.bin  (state_dict)

    or:
        {checkpoint_path}/
        └── pytorch_model.bin  (state_dict)
    """

    def load(self) -> Tuple[nn.Module, RobertaTokenizer]:
        """
        Load CodeT5 model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading CodeT5 model from {self.config.checkpoint_path}")

        # Load config
        config = T5Config.from_pretrained(self.config.base_model_path)

        # Load tokenizer (CodeT5 uses RobertaTokenizer)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.base_model_path)

        # Load base encoder
        encoder = T5ForConditionalGeneration.from_pretrained(
            self.config.base_model_path,
            config=config
        )

        # Create args
        args = CodeT5Args(
            model_type="codet5",
            max_source_length=self.config.max_length
        )

        # Create model
        self.model = CodeT5DefectModel(encoder, config, self.tokenizer, args)

        # Load trained weights
        checkpoint_path = self._find_checkpoint()
        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        self._is_loaded = True
        logger.info(f"CodeT5 model loaded successfully on {self.device}")

        return self.model, self.tokenizer

    def _find_checkpoint(self) -> str:
        """Find the model checkpoint file."""
        # Try checkpoint-last/pytorch_model.bin first
        checkpoint_last = os.path.join(
            self.config.checkpoint_path, "checkpoint-last", "pytorch_model.bin"
        )
        if os.path.exists(checkpoint_last):
            return checkpoint_last

        # Try pytorch_model.bin directly
        model_bin = os.path.join(self.config.checkpoint_path, "pytorch_model.bin")
        if os.path.exists(model_bin):
            return model_bin

        # Try checkpoint-last directory with different names
        checkpoint_dir = os.path.join(self.config.checkpoint_path, "checkpoint-last")
        if os.path.isdir(checkpoint_dir):
            for name in ["pytorch_model.bin", "model.bin"]:
                path = os.path.join(checkpoint_dir, name)
                if os.path.exists(path):
                    return path

        raise FileNotFoundError(
            f"Could not find model checkpoint in {self.config.checkpoint_path}. "
            f"Expected: checkpoint-last/pytorch_model.bin or pytorch_model.bin"
        )

    def preprocess(
        self,
        code: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess code for model input.

        Args:
            code: Single code string or list of code strings

        Returns:
            Dictionary with 'input_ids' tensor
        """
        self.ensure_loaded()

        if isinstance(code, str):
            code = [code]

        # Tokenize
        encoded = self.tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device)
        }

    def predict(self, code: str) -> ModelPrediction:
        """
        Predict defect label for a single code sample.

        Args:
            code: Source code string

        Returns:
            ModelPrediction with binary classification result
        """
        self.ensure_loaded()

        inputs = self.preprocess(code)

        with torch.no_grad():
            self.model.eval()
            logits = self.model.get_logits(input_ids=inputs["input_ids"])
            probs = torch.softmax(logits, dim=-1).squeeze()

            # Class 0: non-defective, Class 1: defective
            label = torch.argmax(probs).item()
            prob_values = probs.tolist()
            logit_values = logits.squeeze().tolist()

        return ModelPrediction(
            label=label,
            probability=prob_values[label],
            probabilities=prob_values,
            logits=logit_values
        )

    def _batch_predict_impl(self, codes: List[str]) -> List[ModelPrediction]:
        """
        Batch prediction implementation.

        Args:
            codes: List of code strings

        Returns:
            List of ModelPrediction objects
        """
        inputs = self.preprocess(codes)

        with torch.no_grad():
            self.model.eval()
            logits = self.model.get_logits(input_ids=inputs["input_ids"])
            probs = torch.softmax(logits, dim=-1)

            predictions = []
            for i in range(len(codes)):
                prob_values = probs[i].tolist()
                label = torch.argmax(probs[i]).item()
                logit_values = logits[i].tolist()

                predictions.append(ModelPrediction(
                    label=label,
                    probability=prob_values[label],
                    probabilities=prob_values,
                    logits=logit_values
                ))

            return predictions
