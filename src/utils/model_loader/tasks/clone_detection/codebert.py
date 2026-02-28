"""
CodeBERT model loader for Clone Detection task.

This loader handles CodeBERT-based models trained on the clone detection task,
following the architecture defined in src/training/victim_model/cd/CodeBERT/model.py.

Clone Detection is a binary classification task that determines whether two
code snippets are clones (semantically equivalent).
"""

import os
import logging
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from ...base import BaseModelLoader, ModelConfig, ModelPrediction
from ...registry import ModelRegistry

logger = logging.getLogger(__name__)


class RobertaClassificationHead(nn.Module):
    """
    Classification head for clone detection.

    Takes CLS tokens from two code snippets (concatenated), combines them,
    and produces a binary classification output.

    Architecture:
        [CLS_1, CLS_2] concat -> Dropout -> Dense -> Tanh -> Dropout -> Output
    """

    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: [batch_size * 2, seq_len, hidden_size] encoder outputs

        Returns:
            logits: [batch_size, 2] classification logits
        """
        # Take CLS token: [batch_size * 2, hidden_size]
        x = features[:, 0, :]
        # Reshape to concatenate pairs: [batch_size, hidden_size * 2]
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CodeBERTCloneModel(nn.Module):
    """
    CodeBERT model for clone detection.

    Architecture:
        Code1 + Code2 -> RoBERTa Encoder -> CLS tokens -> Classification Head

    This is a reimplementation of the model from:
    src/training/victim_model/cd/CodeBERT/model.py
    """

    def __init__(
        self,
        encoder: RobertaModel,
        config: RobertaConfig,
        tokenizer: RobertaTokenizer,
        block_size: int = 400
    ):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.block_size = block_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, 2, seq_len] or [batch_size * 2, seq_len] token IDs
            attention_mask: Attention mask (optional)
            labels: Optional [batch_size] labels for loss calculation

        Returns:
            If labels provided: (loss, probabilities)
            Otherwise: probabilities [batch_size, 2]
        """
        # Reshape input_ids to [batch_size * 2, block_size]
        if input_ids.dim() == 3:
            # 输入已经是 [batch, 2, seq_len]
            batch_size = input_ids.size(0)
            input_ids = input_ids.reshape(batch_size * 2, -1)
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size * 2, -1)
        else:
            # 输入是 [batch, total_len] (来自 CDDataset)
            # 强制将总长度对半开，分成 2 个 code snippet
            batch_size = input_ids.size(0)
            input_ids = input_ids.reshape(batch_size * 2, -1)
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size * 2, -1)

        if attention_mask is None:
            attention_mask = input_ids.ne(1)  # 1 is pad token id for RoBERTa

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob

        return prob

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get raw logits before softmax.

        Args:
            input_ids: [batch_size, 2, seq_len] or [batch_size * 2, seq_len] token IDs
            attention_mask: Attention mask (optional)

        Returns:
            logits: [batch_size, 2]
        """
        batch_size = input_ids.size(0)
        if input_ids.dim() == 3:
            input_ids = input_ids.reshape(batch_size * 2, -1)
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size * 2, -1)
        else:
            input_ids = input_ids.reshape(batch_size * 2, -1)
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size * 2, -1)
        # --- 修改结束 ---

        if attention_mask is None:
            attention_mask = input_ids.ne(1)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        logits = self.classifier(outputs)

        return logits


@ModelRegistry.register("cd", "codebert")
class CodeBERTCloneLoader(BaseModelLoader):
    """
    Loader for CodeBERT-based clone detection models.

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
        logger.info(f"Loading CodeBERT Clone model from {self.config.checkpoint_path}")

        # Load config and tokenizer
        config = RobertaConfig.from_pretrained(self.config.base_model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.base_model_path)

        # Load base encoder
        encoder = RobertaModel.from_pretrained(self.config.base_model_path, config=config)

        # Create model
        self.model = CodeBERTCloneModel(
            encoder=encoder,
            config=config,
            tokenizer=self.tokenizer,
            block_size=self.config.max_length
        )

        # Load trained weights
        checkpoint_path = self._find_checkpoint()
        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        self._is_loaded = True
        logger.info(f"CodeBERT Clone model loaded successfully on {self.device}")

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
        code: Union[str, List[str], Tuple[str, str], List[Tuple[str, str]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess code pair(s) for model input.

        Args:
            code: Can be:
                - Tuple[str, str]: Single code pair (code1, code2)
                - List[Tuple[str, str]]: List of code pairs

        Returns:
            Dictionary with 'input_ids' tensor
        """
        self.ensure_loaded()

        # Handle different input formats
        if isinstance(code, tuple) and len(code) == 2 and isinstance(code[0], str):
            # Single pair
            code_pairs = [code]
        elif isinstance(code, list) and len(code) > 0:
            if isinstance(code[0], tuple):
                code_pairs = code
            else:
                raise ValueError(
                    "For clone detection, input must be a tuple (code1, code2) "
                    "or a list of tuples [(code1, code2), ...]"
                )
        else:
            raise ValueError(
                "For clone detection, input must be a tuple (code1, code2) "
                "or a list of tuples [(code1, code2), ...]"
            )

        # Tokenize each code in the pair separately
        all_input_ids = []
        for code1, code2 in code_pairs:
            # Tokenize code1
            encoded1 = self.tokenizer(
                code1,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            # Tokenize code2
            encoded2 = self.tokenizer(
                code2,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            # Stack: [2, max_length]
            pair_ids = torch.cat([encoded1["input_ids"], encoded2["input_ids"]], dim=0)
            all_input_ids.append(pair_ids)

        # Stack all pairs: [batch_size, 2, max_length]
        input_ids = torch.stack(all_input_ids, dim=0).to(self.device)

        return {"input_ids": input_ids}

    def predict(self, code: Tuple[str, str]) -> ModelPrediction:
        """
        Predict clone label for a single code pair.

        Args:
            code: Tuple of (code1, code2)

        Returns:
            ModelPrediction with binary classification result
                label=0: non-clone
                label=1: clone
        """
        self.ensure_loaded()

        inputs = self.preprocess(code)

        with torch.no_grad():
            self.model.eval()
            logits = self.model.get_logits(input_ids=inputs["input_ids"])
            probs = torch.softmax(logits, dim=-1).squeeze()

            label = torch.argmax(probs).item()
            prob_values = probs.tolist()
            logit_values = logits.squeeze().tolist()

        return ModelPrediction(
            label=label,
            probability=prob_values[label],
            probabilities=prob_values,
            logits=logit_values
        )

    def _batch_predict_impl(self, codes: List[Tuple[str, str]]) -> List[ModelPrediction]:
        """
        Batch prediction implementation.

        Args:
            codes: List of code pairs

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
