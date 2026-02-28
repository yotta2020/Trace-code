"""
CodeBERT model loader for Code Refinement task.

This loader handles CodeBERT-based Seq2Seq models trained on the code refinement task,
following the architecture defined in src/training/victim_model/CodeRefinement/CodeBERT/.

Code Refinement with CodeBERT:
- Architecture: RoBERTa encoder + Transformer decoder
- Input: Buggy code
- Output: Fixed/refined code
"""

import os
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
)

from ...base import BaseModelLoader, ModelConfig, GenerationPrediction
from ...registry import ModelRegistry

logger = logging.getLogger(__name__)

# Import Seq2Seq model from training code
try:
    from src.training.victim_model.CodeRefinement.CodeBERT.code.model import Seq2Seq
    CodeBERTRefinementModel = Seq2Seq
except ImportError:
    logger.warning(
        "Could not import Seq2Seq model from training code. "
        "Make sure src.training.victim_model.CodeRefinement.CodeBERT.code.model is available."
    )
    CodeBERTRefinementModel = None


@ModelRegistry.register("cr", "codebert")
class CodeBERTRefinementLoader(BaseModelLoader):
    """
    Loader for CodeBERT-based code refinement models.

    Expected checkpoint structure:
        {checkpoint_path}/
        └── pytorch_model.bin  (state_dict)

    The model uses a Seq2Seq architecture with:
    - Encoder: RoBERTa
    - Decoder: Transformer decoder
    - Generation: Beam search
    """

    def load(self) -> Tuple[nn.Module, RobertaTokenizer]:
        """
        加载 CodeBERT Seq2Seq 模型和分词器。
        同步 src/training/victim_model/CodeRefinement/CodeBERT/code/run.py 的模型构建逻辑。
        """
        print(f"    [CodeBERT] 开始加载...")
        
        config = RobertaConfig.from_pretrained(self.config.base_model_path)
        tokenizer = RobertaTokenizer.from_pretrained(self.config.base_model_path)
        
        # 1. 初始化 Encoder (RoBERTa)
        encoder = RobertaModel.from_pretrained(
            self.config.base_model_path,
            config=config
        )
        
        # 2. 【关键修复】：根据训练代码同步初始化 Decoder 结构
        # 使用 TransformerDecoderLayer 和 TransformerDecoder，而不是复用 encoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size, 
            nhead=config.num_attention_heads
        )
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # 3. 构建 Seq2Seq 模型
        model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,  # 传入正确的 decoder
            config=config,
            beam_size=self.config.extra_args.get('beam_size', 10),
            max_length=self.config.max_length,
            sos_id=tokenizer.cls_token_id,
            eos_id=tokenizer.sep_token_id
        )
        
        # 4. 寻找并加载 checkpoint 权重
        checkpoint_path = self._find_checkpoint()
        if not checkpoint_path:
            raise FileNotFoundError(f"Checkpoint not found: {self.config.checkpoint_path}")
            
        print(f"    [CodeBERT] 加载权重: {checkpoint_path}")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device),
            strict=False
        )
        
        model.to(self.device)
        model.eval()
        
        # 显式挂载属性，确保 base.py 也能访问
        self.model = model
        self.tokenizer = tokenizer
        
        return model, tokenizer

    def _find_checkpoint(self) -> Optional[str]:
        """Find the model checkpoint file."""
        checkpoint_path = self.config.checkpoint_path

        # Check if it's already a .bin file
        if os.path.isfile(checkpoint_path) and checkpoint_path.endswith('.bin'):
            return checkpoint_path

        # Try common checkpoint locations
        for filename in ["pytorch_model.bin", "model.bin"]:
            bin_path = os.path.join(checkpoint_path, filename)
            if os.path.exists(bin_path):
                return bin_path

        # Try checkpoint subdirectories
        for subdir in ["checkpoint-best", "checkpoint-last"]:
            for filename in ["pytorch_model.bin", "model.bin"]:
                bin_path = os.path.join(checkpoint_path, subdir, filename)
                if os.path.exists(bin_path):
                    return bin_path

        return None

    def preprocess(
        self,
        code: str,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess code input for model generation.

        Args:
            code: Source code string (buggy code)
            max_length: Maximum sequence length (defaults to config.max_length)

        Returns:
            Dictionary with 'source_ids' and 'source_mask' tensors
        """
        self.ensure_loaded()

        if max_length is None:
            max_length = self.config.max_length

        # Tokenize input
        encoded = self.tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Move to device
        source_ids = encoded["input_ids"].to(self.device)
        source_mask = encoded["attention_mask"].to(self.device)

        return {
            "source_ids": source_ids,
            "source_mask": source_mask
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
            num_beams: Number of beams (note: beam_size is set at model creation)
            **kwargs: Additional generation arguments (ignored for CodeBERT)

        Returns:
            GenerationPrediction with generated fixed code
        """
        self.ensure_loaded()

        if max_length is None:
            max_length = self.config.max_length

        # Preprocess input
        inputs = self.preprocess(code, max_length=max_length)

        # Generate using model's forward pass (beam search is built-in)
        with torch.no_grad():
            self.model.eval()
            # Seq2Seq.forward() with target_ids=None triggers generation mode
            preds = self.model(
                source_ids=inputs["source_ids"],
                source_mask=inputs["source_mask"],
                target_ids=None,  # None triggers generation
                target_mask=None
            )

        # preds shape: [batch_size, beam_size, max_length]
        # Take the top beam (best prediction)
        if preds.dim() == 3:
            pred_ids = preds[0, 0, :]  # First sample, first beam
        else:
            pred_ids = preds[0, :]

        # Decode generated tokens
        generated_text = self.tokenizer.decode(
            pred_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return GenerationPrediction(
            generated_text=generated_text,
            generated_ids=pred_ids.tolist(),
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

        # Tokenize all inputs
        encoded = self.tokenizer(
            codes,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        source_ids = encoded["input_ids"].to(self.device)
        source_mask = encoded["attention_mask"].to(self.device)

        # Batch generation
        with torch.no_grad():
            self.model.eval()
            preds = self.model(
                source_ids=source_ids,
                source_mask=source_mask,
                target_ids=None,
                target_mask=None
            )

        # Decode all generated sequences
        predictions = []
        for i in range(len(codes)):
            # preds shape: [batch_size, beam_size, max_length]
            if preds.dim() == 3:
                pred_ids = preds[i, 0, :]  # Top beam for sample i
            else:
                pred_ids = preds[i, :]

            generated_text = self.tokenizer.decode(
                pred_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            predictions.append(GenerationPrediction(
                generated_text=generated_text,
                generated_ids=pred_ids.tolist(),
                input_text=codes[i]
            ))

        return predictions
