"""
Model Wrapper for Unified Inference

This module provides a unified interface for model inference across different
model types (CodeBERT, CodeT5, StarCoder) and task types (DD, CD, CR).
It handles the differences in model output formats and prediction methods.
"""

import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional, List, Union

logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Unified wrapper for different victim models.

    Responsibilities:
    - Hide differences in model output formats
    - Provide unified prediction interface
    - Handle task-specific forward logic
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        task_type: str
    ):
        """
        Initialize the model wrapper.

        Args:
            model: The loaded PyTorch model
            model_type: One of ["codebert", "codet5", "starcoder"]
            task_type: One of ["dd", "cd", "cr"]
        """
        self.model = model
        self.model_type = model_type.lower()
        self.task_type = task_type.lower()

        # Validate inputs
        valid_models = ["codebert", "codet5", "starcoder"]
        valid_tasks = ["dd", "cd", "cr"]

        if self.model_type not in valid_models:
            raise ValueError(f"Unknown model_type: {model_type}. Must be one of {valid_models}")

        if self.task_type not in valid_tasks:
            raise ValueError(f"Unknown task_type: {task_type}. Must be one of {valid_tasks}")

        logger.info(f"Initialized ModelWrapper: model={model_type}, task={task_type}")

    @property
    def device(self):
        """Get the device of the model."""
        return next(self.model.parameters()).device

    def predict(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified prediction interface.

        Returns:
            predictions: [batch_size] - Predicted class indices
            outputs: [batch_size, num_classes] - Raw model outputs (probs or logits)
        """
        if self.task_type in ["dd", "cd"]:
            return self._predict_classification(batch)
        elif self.task_type == "cr":
            raise ValueError(
                "CR task uses generate() method, not predict(). "
                "Call wrapper.generate(source_ids, source_mask) instead."
            )
        else:
            raise NotImplementedError(f"Task {self.task_type} not implemented yet")

    def _predict_classification(
        self,
        batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unpack batch
        input_ids = batch[0]
        attention_mask = batch[1] if len(batch) > 1 else None
        labels = batch[2] if len(batch) > 2 else None

        # Standardize input names
        if self.model_type == "codet5":
            model_inputs = {'source_ids': input_ids}
        else:
            model_inputs = {'input_ids': input_ids}

        # Filter attention_mask for models that don't support it in their custom forward
        if attention_mask is not None and self.model_type not in ["codebert", "codet5"]:
            model_inputs['attention_mask'] = attention_mask

        # Pass labels to get (loss, logits/probs) tuple
        if labels is not None:
            model_inputs['labels'] = labels

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        predictions, model_outputs = self._parse_classification_outputs(outputs)
        return predictions, model_outputs

    def _parse_classification_outputs(
        self,
        outputs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse model outputs into predictions and raw outputs.
        Handle dimension differences between DD (binary) and CD (multi-class) tasks.
        """
        # Handle Transformers' SequenceClassifierOutput or standard tuples
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)):
            # Usually (loss, logits) or (loss, probs)
            logits = outputs[1]
        else:
            logits = outputs

        if self.model_type == "codebert":
            # FIXED: Dynamically detect output shape for CodeBERT
            # DD Task usually returns [batch, 1] (Sigmoid prob)
            # CD Task usually returns [batch, 2] (Softmax prob)
            if logits.shape[-1] == 2:
                # Case: Clone Detection or 2-class Defect Detection
                prob = logits
                predictions = prob.argmax(dim=-1)
                return predictions, prob
            else:
                # Case: 1-class Sigmoid output (Standard DD)
                prob_squeezed = logits.squeeze(-1)
                predictions = (prob_squeezed > 0.5).long()
                # Expand to [batch, 2] for consistency
                prob_2d = torch.stack([1 - prob_squeezed, prob_squeezed], dim=-1)
                return predictions, prob_2d

        elif self.model_type == "codet5":
            # CodeT5 returns [batch, 2] probabilities
            prob = logits
            predictions = prob.argmax(dim=-1)
            return predictions, prob

        elif self.model_type == "starcoder":
            # StarCoder returns [batch, 2] raw logits
            predictions = logits.argmax(dim=-1)
            return predictions, logits

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def generate(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        max_length: int = 256,
        num_beams: int = 5,
    ) -> torch.Tensor:
        """
        Generate sequences for Code Refinement task.
        """
        self.model.eval()

        with torch.no_grad():
            if self.model_type == "codebert":
                # Custom Seq2Seq generation
                generated_ids = self.model(
                    source_ids=source_ids,
                    source_mask=source_mask,
                )
                if isinstance(generated_ids, tuple):
                    generated_ids = generated_ids[0]
                if generated_ids.dim() == 3:
                    generated_ids = generated_ids[:, 0, :]  # Take top beam

            elif self.model_type == "codet5":
                # T5 generation
                generated_ids = self.model.generate(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )

            elif self.model_type == "starcoder":
                # Causal LM generation (Decoder-only with left-padding)
                if hasattr(self.model, 'encoder'):
                    base_model = self.model.encoder
                else:
                    base_model = self.model

                # [SAFETY] Double check padding side before generation
                if hasattr(base_model.config, "padding_side"):
                    base_model.config.padding_side = 'left'
                
                # Check tokenizer if attached to model (custom models)
                if hasattr(self.model, 'tokenizer'):
                    self.model.tokenizer.padding_side = 'left'

                # Retrieve pad_token_id
                pad_token_id = getattr(base_model.config, 'pad_token_id', None)
                if pad_token_id is None:
                    pad_token_id = base_model.config.eos_token_id

                # Run generation
                full_generated = base_model.generate(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    max_new_tokens=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    pad_token_id=pad_token_id,
                    eos_token_id=base_model.config.eos_token_id,
                )

                # In transformers, generated tokens are appended after the input tensor.
                # Since we use left-padding, the input tensor (including paddings) has width source_ids.shape[1].
                # Therefore, newly generated tokens start exactly at index source_ids.shape[1].
                generated_ids = full_generated[:, source_ids.shape[1]:]

            else:
                raise ValueError(f"Unknown model_type for generation: {self.model_type}")

        return generated_ids

    def get_probabilities(
        self,
        batch: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Get normalized class probabilities for the batch.
        """
        _, outputs = self.predict(batch)

        if self.model_type == "starcoder":
            # Apply softmax to raw logits
            probs = torch.softmax(outputs, dim=-1)
        else:
            # CodeBERT and CodeT5 wrappers already return probabilities
            probs = outputs

        return probs