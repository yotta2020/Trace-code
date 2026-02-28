"""
CodeT5 model loader for Code Refinement task.

This loader handles CodeT5-based models trained on the code refinement task,
following the architecture defined in src/training/victim_model/CodeRefinement/CodeT5/.

Code Refinement is a seq2seq generation task:
- Input: Buggy code
- Output: Fixed/refined code
- Model: T5ForConditionalGeneration
"""

import os
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    RobertaTokenizer,
)

from ...base import BaseModelLoader, ModelConfig, GenerationPrediction
from ...registry import ModelRegistry

logger = logging.getLogger(__name__)


# For backward compatibility, we can reuse the model class name
CodeT5RefinementModel = T5ForConditionalGeneration


@ModelRegistry.register("cr", "codet5")
class CodeT5RefinementLoader(BaseModelLoader):
    """
    Loader for CodeT5-based code refinement models.
    """

    def load(self) -> Tuple[nn.Module, RobertaTokenizer]:
        """
        Load CodeT5 model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading CodeT5 Refinement model from {self.config.checkpoint_path}")

        # Load tokenizer (CodeT5 uses RobertaTokenizer)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.base_model_path)

        # Load model - try to load from checkpoint first, fallback to base model
        checkpoint_path = self._find_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading fine-tuned model from checkpoint: {checkpoint_path}")
            
            # If checkpoint is a directory containing both config and weights
            if os.path.isdir(checkpoint_path) and os.path.exists(os.path.join(checkpoint_path, "config.json")):
                self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
            else:
                # If checkpoint is a .bin file or a directory without config.json
                # Load architecture from base model path first
                config = T5Config.from_pretrained(self.config.base_model_path)
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.config.base_model_path,
                    config=config
                )
                
                # Determine the actual weight file path
                bin_file = checkpoint_path
                if os.path.isdir(checkpoint_path):
                    for f in ["pytorch_model.bin", "model.bin", "model.safetensors"]:
                        if os.path.exists(os.path.join(checkpoint_path, f)):
                            bin_file = os.path.join(checkpoint_path, f)
                            break
                
                logger.info(f"Loading weights from: {bin_file}")
                state_dict = torch.load(bin_file, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(
                f"Checkpoint not found, loading base model from {self.config.base_model_path}"
            )
            self.model = T5ForConditionalGeneration.from_pretrained(self.config.base_model_path)

        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        self._is_loaded = True
        logger.info(f"CodeT5 Refinement model loaded successfully on {self.device}")

        return self.model, self.tokenizer

    def _find_checkpoint(self) -> Optional[str]:
        """
        Find the model checkpoint file or directory.
        Prioritizes subdirectories like checkpoint-last/ or checkpoint-best/.
        """
        root_path = self.config.checkpoint_path

        # 1. If the path is directly a file, return it
        if os.path.isfile(root_path):
            return root_path

        # 2. Search priority: subdirectories then root
        for subdir in ["checkpoint-best", "checkpoint-last", ""]:
            check_dir = os.path.join(root_path, subdir) if subdir else root_path
            
            if not os.path.isdir(check_dir):
                continue
                
            # Check for weight files in this directory
            for filename in ["pytorch_model.bin", "model.bin", "model.safetensors"]:
                bin_path = os.path.join(check_dir, filename)
                if os.path.exists(bin_path):
                    # If it has config.json, from_pretrained can use the directory
                    if os.path.exists(os.path.join(check_dir, "config.json")):
                        return check_dir
                    return bin_path

        return None

    def preprocess(
        self,
        code: str,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess code input for model generation.
        """
        self.ensure_loaded()

        if max_length is None:
            max_length = self.config.max_length

        encoded = self.tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device)
        }

    def predict(self, code: str) -> GenerationPrediction:
        return self.generate(code)

    def generate(
        self,
        code: str,
        max_length: Optional[int] = None,
        num_beams: int = 5,
        **kwargs
    ) -> GenerationPrediction:
        """
        Generate fixed code for a single buggy code sample.
        """
        self.ensure_loaded()

        if max_length is None:
            max_length = self.config.max_length

        inputs = self.preprocess(code, max_length=max_length)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                **kwargs
            )

        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return GenerationPrediction(
            generated_text=generated_text,
            generated_ids=generated_ids[0].tolist(),
            input_text=code
        )

    def _batch_predict_impl(self, codes: List[str]) -> List[GenerationPrediction]:
        return [self.generate(code) for code in codes]

    def _batch_generate_impl(
        self,
        codes: List[str],
        max_length: Optional[int] = None,
        num_beams: int = 5,
        **kwargs
    ) -> List[GenerationPrediction]:
        """
        Efficient batch generation implementation.
        """
        self.ensure_loaded()

        if max_length is None:
            max_length = self.config.max_length

        encoded = self.tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                **kwargs
            )

        predictions = []
        for i, gen_ids in enumerate(generated_ids):
            generated_text = self.tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            predictions.append(GenerationPrediction(
                generated_text=generated_text,
                generated_ids=gen_ids.tolist(),
                input_text=codes[i]
            ))

        return predictions