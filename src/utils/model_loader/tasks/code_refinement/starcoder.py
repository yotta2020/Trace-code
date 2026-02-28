"""
StarCoder2 model loader for Code Refinement task.

This loader handles StarCoder2-based models trained on the code refinement task,
following the architecture defined in src/training/victim_model/CodeRefinement/StarCoder2/.

Code Refinement with StarCoder2:
- Architecture: Decoder-only (Causal LM)
- Training: Input = [buggy code] + [sep] + [fixed code], loss only on fixed part
- Inference: Input = [buggy code] + [sep], generate fixed code
"""

import os
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig

from ...base import BaseModelLoader, ModelConfig, GenerationPrediction
from ...registry import ModelRegistry

logger = logging.getLogger(__name__)

# Import StarCoder model from training code
try:
    from src.training.victim_model.CodeRefinement.StarCoder2.model import StarCoderCodeRefinementModel
except ImportError:
    logger.warning(
        "Could not import StarCoderCodeRefinementModel from training code. "
        "Make sure src.training.victim_model.CodeRefinement.StarCoder2.model is available."
    )
    StarCoderCodeRefinementModel = None


@ModelRegistry.register("cr", "starcoder")
class StarCoderRefinementLoader(BaseModelLoader):
    """
    Loader for StarCoder2-based code refinement models.

    Expected checkpoint structure:
        {checkpoint_path}/
        ├── pytorch_model.bin (or HF checkpoint directory)
        ├── config.json
        └── tokenizer files

    The model is a decoder-only (causal LM) architecture.
    """

    def load(self) -> Tuple[nn.Module, AutoTokenizer]:
        """
        Load StarCoder2 model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading StarCoder2 Refinement model from {self.config.checkpoint_path}")

        if StarCoderCodeRefinementModel is None:
            raise ImportError(
                "StarCoderCodeRefinementModel not available. "
                "Please ensure src.training.victim_model.CodeRefinement.StarCoder2.model is installed."
            )

        # Load tokenizer
        tokenizer_path = self.config.extra_args.get("tokenizer_path", self.config.base_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Set padding side to left for decoder-only generation
        self.tokenizer.padding_side = 'left'

        # CRITICAL FIX: In StarCoder2, sep_token_id = eos_token_id = pad_token_id = unk_token_id = 0.
        # During inference, we append sep_token (ID=0) at the end of the prompt.
        # transformers.generate() internally checks if input_ids[:, -1] == pad_token_id;
        # since both are 0, it wrongly reports right-padding.
        #
        # Fix: We need a pad_token_id != 0. Since attention_mask hides the padding tokens anyway,
        # we can safely use any existing token ID (e.g., ID=1) as pad_token_id for generation purposes.
        # We explicitly set it on the tokenizer.
        safe_pad_id = 1  # Guaranteed to exist as vocab_size is 49152
        self.tokenizer.pad_token_id = safe_pad_id
        # We also need to map it to tokenizer.pad_token string, so we reverse-lookup ID 1:
        self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(safe_pad_id)
        
        logger.info(
            f"StarCoder CR: reassigned pad_token to token ('{self.tokenizer.pad_token}', "
            f"ID={self.tokenizer.pad_token_id}) to avoid sep_token==pad_token collision."
        )

        # Load config
        config = AutoConfig.from_pretrained(self.config.base_model_path)

        # Load model
        checkpoint_path = self._find_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading fine-tuned model from checkpoint: {checkpoint_path}")

            # Check if checkpoint is a directory (HF format)
            if os.path.isdir(checkpoint_path):
                self.model = StarCoderCodeRefinementModel.from_pretrained(
                    checkpoint_path,
                    config=config,
                    tokenizer=self.tokenizer
                )
            else:
                # Load base model first, then load state dict
                from transformers import AutoModelForCausalLM

                encoder = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_path,
                    config=config
                )
                self.model = StarCoderCodeRefinementModel(
                    encoder=encoder,
                    config=config,
                    tokenizer=self.tokenizer
                )

                # Load trained weights
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                # Handle potential "module." prefix from DataParallel
                if any(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict)
        else:
            logger.warning(
                f"Checkpoint not found at {self.config.checkpoint_path}, "
                f"loading base model from {self.config.base_model_path}"
            )
            from transformers import AutoModelForCausalLM

            encoder = AutoModelForCausalLM.from_pretrained(self.config.base_model_path)
            self.model = StarCoderCodeRefinementModel(
                encoder=encoder,
                config=config,
                tokenizer=self.tokenizer
            )

        # Sync pad_token_id from tokenizer → model config so generate() uses the
        # updated (non-colliding) pad_token_id.
        if hasattr(self.model, 'encoder'):
            self.model.encoder.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # Also sync the stored attribute on the wrapper
        self.model.pad_token_id = self.tokenizer.pad_token_id

        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        self._is_loaded = True
        logger.info(
            f"StarCoder2 Refinement model loaded successfully on {self.device} "
            f"| pad_token_id={self.tokenizer.pad_token_id} "
            f"| eos_token_id={self.tokenizer.eos_token_id}"
        )

        return self.model, self.tokenizer

    def _find_checkpoint(self) -> Optional[str]:
        """Find the model checkpoint file or directory."""
        checkpoint_path = self.config.checkpoint_path

        # 1. 首先检查路径本身是否是一个有效的 HF 目录或 .bin 文件
        if os.path.exists(checkpoint_path):
            if os.path.isdir(checkpoint_path) and os.path.exists(
                os.path.join(checkpoint_path, "config.json")
            ):
                return checkpoint_path
            if os.path.isfile(checkpoint_path) and checkpoint_path.endswith('.bin'):
                return checkpoint_path

        # 2. 尝试在常见的子目录中查找
        # 在这里加入 "merged" 路径
        for subdir in ["", "merged", "checkpoint-best", "checkpoint-last"]:
            check_dir = os.path.join(checkpoint_path, subdir) if subdir else checkpoint_path

            # 检查该子目录是否为标准的 HuggingFace 格式（包含 config.json）
            if os.path.isdir(check_dir) and os.path.exists(os.path.join(check_dir, "config.json")):
                return check_dir

            # 检查子目录中是否存在单个权重文件
            for filename in ["pytorch_model.bin", "model.bin"]:
                bin_path = os.path.join(check_dir, filename)
                if os.path.exists(bin_path):
                    return bin_path

        return None


    def preprocess(
        self,
        code: str,
        max_length: Optional[int] = None,
        add_sep: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess code input for model generation.

        For StarCoder2 refinement, the input format during inference is:
        [buggy code] + [sep token]

        Args:
            code: Source code string (buggy code)
            max_length: Maximum sequence length (defaults to config.max_length)
            add_sep: Whether to add separator token (default: True for generation)

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        self.ensure_loaded()

        if max_length is None:
            max_length = self.config.max_length

        # Add separator token for generation prompt
        if add_sep:
            sep_token = self.tokenizer.sep_token if hasattr(self.tokenizer, 'sep_token') and self.tokenizer.sep_token else self.tokenizer.eos_token
            code_with_sep = code + " " + sep_token
        else:
            code_with_sep = code

        # Tokenize input
        encoded = self.tokenizer(
            code_with_sep,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Move to device
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def predict(self, code: str) -> GenerationPrediction:
        """
        This method is aliased to generate() for Code Refinement task.

        Args:
            code: Source code string (buggy code)

        Returns:
            GenerationPrediction with generated fixed code
        """
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

        Args:
            code: Source code string (buggy code)
            max_length: Maximum generation length (defaults to config.max_length)
            num_beams: Number of beams for beam search (default: 5)
            **kwargs: Additional generation arguments (temperature, top_p, etc.)

        Returns:
            GenerationPrediction with generated fixed code
        """
        self.ensure_loaded()

        if max_length is None:
            max_length = self.config.max_length

        # Preprocess input (add separator token)
        inputs = self.preprocess(code, max_length=max_length, add_sep=True)

        # Generate using model's generate method
        with torch.no_grad():
            self.model.eval()

            # Use max_new_tokens instead of max_length for generation
            max_new_tokens = kwargs.pop("max_new_tokens", max_length)

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Extract only the generated part (remove input prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_ids_only = generated_ids[0, input_length:]

        # Decode generated tokens
        generated_text = self.tokenizer.decode(
            generated_ids_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return GenerationPrediction(
            generated_text=generated_text,
            generated_ids=generated_ids_only.tolist(),
            input_text=code
        )

    def _batch_predict_impl(self, codes: List[str]) -> List[GenerationPrediction]:
        """
        Internal implementation of batch prediction.

        Args:
            codes: List of code strings

        Returns:
            List of GenerationPrediction objects
        """
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

        Args:
            codes: List of buggy code strings
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            **kwargs: Additional generation arguments

        Returns:
            List of GenerationPrediction objects
        """
        self.ensure_loaded()

        if max_length is None:
            max_length = self.config.max_length

        # Add separator tokens to all inputs
        sep_token = self.tokenizer.sep_token if hasattr(self.tokenizer, 'sep_token') and self.tokenizer.sep_token else self.tokenizer.eos_token
        codes_with_sep = [code + " " + sep_token for code in codes]

        # Tokenize all inputs
        encoded = self.tokenizer(
            codes_with_sep,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Batch generation
        with torch.no_grad():
            self.model.eval()

            max_new_tokens = kwargs.pop("max_new_tokens", max_length)

            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode all generated sequences
        predictions = []
        input_lengths = attention_mask.sum(dim=1)  # Actual input lengths

        for i, gen_ids in enumerate(generated_ids):
            # Extract only the generated part (remove input prompt)
            input_len = input_lengths[i].item()
            generated_ids_only = gen_ids[input_len:]

            generated_text = self.tokenizer.decode(
                generated_ids_only,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            predictions.append(GenerationPrediction(
                generated_text=generated_text,
                generated_ids=generated_ids_only.tolist(),
                input_text=codes[i]
            ))

        return predictions
