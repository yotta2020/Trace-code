"""
CodeBERT model loader for Defect Detection task.

This loader handles CodeBERT-based models trained on the defect detection task,
following the architecture defined in src/training/victim_model/dd/CodeBERT/model.py.
"""

import os
import logging
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from ...base import BaseModelLoader, ModelConfig, ModelPrediction
from ...registry import ModelRegistry

logger = logging.getLogger(__name__)


class CodeBERTDefectModel(nn.Module):
    """
    CodeBERT model for binary defect detection.

    Architecture:
        RoBERTa Encoder -> CLS token -> Dropout -> Dense -> Tanh -> Dropout -> Output

    This is a reimplementation of the model from:
    src/training/victim_model/dd/CodeBERT/model.py
    """

    def __init__(self, encoder: RobertaModel, config: RobertaConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config

        dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)

        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: [batch_size, seq_len] attention mask
            labels: Optional [batch_size] labels for loss calculation

        Returns:
            If labels provided: (loss, probabilities)
            Otherwise: probabilities [batch_size, 1]
        """
        if attention_mask is None:
            attention_mask = input_ids.ne(1)  # 1 is the pad token id for RoBERTa

        outputs = self.encoder(input_ids, attention_mask=attention_mask)[0]

        # CLS token representation
        cls_output = outputs[:, 0, :]

        # Classification head
        cls_output = self.dropout(cls_output)
        cls_output = self.dense(cls_output)
        cls_output = self.activation(cls_output)
        cls_output = self.dropout(cls_output)
        logits = self.out_proj(cls_output)

        prob = torch.sigmoid(logits)

        if labels is not None:
            labels = labels.float().view(-1, 1)
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_fct(logits, labels)
            return loss.mean(), prob

        return prob

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get raw logits before sigmoid.

        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: [batch_size, seq_len] attention mask

        Returns:
            logits: [batch_size, 1]
        """
        if attention_mask is None:
            attention_mask = input_ids.ne(1)

        outputs = self.encoder(input_ids, attention_mask=attention_mask)[0]
        cls_output = outputs[:, 0, :]

        cls_output = self.dropout(cls_output)
        cls_output = self.dense(cls_output)
        cls_output = self.activation(cls_output)
        cls_output = self.dropout(cls_output)
        logits = self.out_proj(cls_output)

        return logits


@ModelRegistry.register("dd", "codebert")
class CodeBERTDefectLoader(BaseModelLoader):
    """
    Loader for CodeBERT-based defect detection models.

    Expected checkpoint structure:
        {checkpoint_path}/
        └── checkpoint-last/
            └── model.bin  (state_dict)

    or:
        {checkpoint_path}/
        └── model.bin  (state_dict)
    """

    def load(self) -> Tuple[nn.Module, RobertaTokenizer]:
        """
        Load CodeBERT model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading CodeBERT model from {self.config.checkpoint_path}")

        # Load config and tokenizer
        config = RobertaConfig.from_pretrained(self.config.base_model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.base_model_path)

        # Load base encoder
        encoder = RobertaModel.from_pretrained(self.config.base_model_path, config=config)

        # Create model
        self.model = CodeBERTDefectModel(encoder, config)

        # Load trained weights
        checkpoint_path = self._find_checkpoint()
        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        self._is_loaded = True
        logger.info(f"CodeBERT model loaded successfully on {self.device}")

        return self.model, self.tokenizer

    def _find_checkpoint(self) -> str:
        """Find the model checkpoint file."""
        # Try checkpoint-last/model.bin first
        checkpoint_last = os.path.join(
            self.config.checkpoint_path, "checkpoint-last", "model.bin"
        )
        if os.path.exists(checkpoint_last):
            return checkpoint_last

        # Try model.bin directly
        model_bin = os.path.join(self.config.checkpoint_path, "model.bin")
        if os.path.exists(model_bin):
            return model_bin

        # Try checkpoint-last directory with different name
        checkpoint_dir = os.path.join(self.config.checkpoint_path, "checkpoint-last")
        if os.path.isdir(checkpoint_dir):
            for name in ["model.bin", "pytorch_model.bin"]:
                path = os.path.join(checkpoint_dir, name)
                if os.path.exists(path):
                    return path

        raise FileNotFoundError(
            f"Could not find model checkpoint in {self.config.checkpoint_path}. "
            f"Expected: checkpoint-last/model.bin or model.bin"
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
            Dictionary with 'input_ids' and 'attention_mask' tensors
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
            logits = self.model.get_logits(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            prob = torch.sigmoid(logits).squeeze()

            # Binary classification: prob is P(defective)
            prob_value = prob.item()
            label = 1 if prob_value >= 0.5 else 0

            # Convert to two-class probabilities
            probabilities = [1 - prob_value, prob_value]
            logit_value = logits.squeeze().item()

        return ModelPrediction(
            label=label,
            probability=prob_value if label == 1 else 1 - prob_value,
            probabilities=probabilities,
            logits=[logit_value]
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
            logits = self.model.get_logits(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            probs = torch.sigmoid(logits).squeeze(-1)

            predictions = []
            for i in range(len(codes)):
                prob_value = probs[i].item()
                label = 1 if prob_value >= 0.5 else 0
                probabilities = [1 - prob_value, prob_value]
                logit_value = logits[i].squeeze().item()

                predictions.append(ModelPrediction(
                    label=label,
                    probability=prob_value if label == 1 else 1 - prob_value,
                    probabilities=probabilities,
                    logits=[logit_value]
                ))

            return predictions
