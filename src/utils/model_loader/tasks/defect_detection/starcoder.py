"""
StarCoder model loader for Defect Detection task.

This loader handles StarCoder-based models trained on the defect detection task,
following the architecture defined in src/training/victim_model/dd/StarCoder/model.py.

StarCoder models use LoRA for efficient fine-tuning and are saved in a merged format
or with separate adapter weights.
"""

import os
import logging
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from ...base import BaseModelLoader, ModelConfig, ModelPrediction
from ...registry import ModelRegistry

logger = logging.getLogger(__name__)


class StarCoderDefectModel(nn.Module):
    """
    StarCoder model for binary defect detection.

    Architecture:
        GPTBigCode -> Mean Pooling -> Dropout -> Dense -> Tanh -> Dropout -> Output

    Uses mean pooling over all non-padding tokens to capture distributed trigger
    patterns in backdoor attack scenarios.

    This is a reimplementation of StarCoderDefectModel from:
    src/training/victim_model/dd/StarCoder/model.py
    """

    def __init__(self, encoder, config, tokenizer):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer

        # CodeXGLUE standard classification head
        dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    @property
    def device(self):
        """Get the device of the model."""
        return next(self.encoder.parameters()).device

    def get_sequence_representation(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract sequence-level representation using mean pooling.

        For backdoor attack scenarios, mean pooling preserves trigger information
        that might appear at any position in the sequence.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]

        Returns:
            sequence_repr: [batch_size, hidden_size]
        """
        mask = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states * mask).sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        sequence_repr = sum_hidden / sum_mask
        return sequence_repr

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
            If labels provided: (loss, logits)
            Otherwise: logits [batch_size, 2]
        """
        if attention_mask is None:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states[-1]
        sequence_repr = self.get_sequence_representation(hidden_states, attention_mask)

        # Classification head
        x = self.dropout(sequence_repr)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits, labels)
            return loss.mean(), logits

        return logits

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get raw logits.

        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: [batch_size, seq_len] attention mask

        Returns:
            logits: [batch_size, 2]
        """
        return self.forward(input_ids, attention_mask)


@ModelRegistry.register("dd", "starcoder")
class StarCoderDefectLoader(BaseModelLoader):
    """
    Loader for StarCoder-based defect detection models.

    Supports two checkpoint formats:
    1. Merged model: All weights merged into a single model
       {checkpoint_path}/merged/
           ├── config.json
           ├── model.safetensors or pytorch_model.bin
           └── classifier.pt

    2. Checkpoint with LoRA adapters:
       {checkpoint_path}/checkpoint-best/
           ├── adapter_config.json
           ├── adapter_model.safetensors
           └── classifier.pt
    """

    def load(self) -> Tuple[nn.Module, AutoTokenizer]:
        """
        Load StarCoder model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading StarCoder model from {self.config.checkpoint_path}")

        # Find the model directory
        model_dir = self._find_model_dir()
        logger.info(f"Found model directory: {model_dir}")

        # Load tokenizer from base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_path,
            trust_remote_code=True
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Check if this is a merged model or adapter model
        if self._is_merged_model(model_dir):
            self._load_merged_model(model_dir)
        else:
            self._load_adapter_model(model_dir)

        # Load classifier head if saved separately
        self._load_classifier_head(model_dir)

        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        self._is_loaded = True
        logger.info(f"StarCoder model loaded successfully on {self.device}")

        return self.model, self.tokenizer

    def _find_model_dir(self) -> str:
        """Find the model checkpoint directory."""
        # Try merged/ directory first (preferred for inference)
        merged_dir = os.path.join(self.config.checkpoint_path, "merged")
        if os.path.isdir(merged_dir):
            return merged_dir

        # Try checkpoint-best/
        best_dir = os.path.join(self.config.checkpoint_path, "checkpoint-best")
        if os.path.isdir(best_dir):
            return best_dir

        # Try checkpoint-last/
        last_dir = os.path.join(self.config.checkpoint_path, "checkpoint-last")
        if os.path.isdir(last_dir):
            return last_dir

        # Try the path directly
        if os.path.isdir(self.config.checkpoint_path):
            # Check if it contains model files
            for name in ["config.json", "adapter_config.json"]:
                if os.path.exists(os.path.join(self.config.checkpoint_path, name)):
                    return self.config.checkpoint_path

        raise FileNotFoundError(
            f"Could not find model directory in {self.config.checkpoint_path}. "
            f"Expected: merged/, checkpoint-best/, or checkpoint-last/"
        )

    def _is_merged_model(self, model_dir: str) -> bool:
        """Check if the model is a merged model (not LoRA adapter)."""
        # Merged models have config.json but not adapter_config.json
        has_config = os.path.exists(os.path.join(model_dir, "config.json"))
        has_adapter = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
        return has_config and not has_adapter

    def _load_merged_model(self, model_dir: str) -> None:
        """Load a merged model (LoRA weights already merged into base)."""
        logger.info("Loading merged model...")

        # Load config
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

        # Load encoder
        encoder = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Create the defect model wrapper
        self.model = StarCoderDefectModel(encoder, config, self.tokenizer)

    def _load_adapter_model(self, model_dir: str) -> None:
        """Load a model with LoRA adapters."""
        logger.info("Loading adapter model...")

        try:
            from peft import PeftModel, PeftConfig
        except ImportError:
            raise ImportError(
                "PEFT library is required to load LoRA models. "
                "Install with: pip install peft"
            )

        # Load PEFT config to get base model path
        peft_config = PeftConfig.from_pretrained(model_dir)
        base_model_path = peft_config.base_model_name_or_path

        # Use the configured base model path if the saved one is not accessible
        if not os.path.exists(base_model_path):
            base_model_path = self.config.base_model_path

        # Load base model
        config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Load PEFT model
        encoder = PeftModel.from_pretrained(base_model, model_dir)

        # Merge LoRA weights for faster inference
        encoder = encoder.merge_and_unload()

        # Create the defect model wrapper
        self.model = StarCoderDefectModel(encoder, config, self.tokenizer)

    def _load_classifier_head(self, model_dir: str) -> None:
        """Load classifier head weights if saved separately."""
        classifier_path = os.path.join(model_dir, "classifier.pt")

        if os.path.exists(classifier_path):
            logger.info(f"Loading classifier head from {classifier_path}")
            classifier_state = torch.load(classifier_path, map_location=self.device)

            # Handle different save formats
            if isinstance(classifier_state, dict):
                if "dense.weight" in classifier_state:
                    # Direct state dict
                    self.model.dense.load_state_dict({
                        "weight": classifier_state["dense.weight"],
                        "bias": classifier_state["dense.bias"]
                    })
                    self.model.out_proj.load_state_dict({
                        "weight": classifier_state["out_proj.weight"],
                        "bias": classifier_state["out_proj.bias"]
                    })
                elif "dropout" in classifier_state or "dense" in classifier_state:
                    # Nested state dict
                    for key, value in classifier_state.items():
                        if hasattr(self.model, key):
                            getattr(self.model, key).load_state_dict(value)
        else:
            logger.warning(
                f"Classifier head file not found at {classifier_path}. "
                f"Using randomly initialized classifier weights."
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
