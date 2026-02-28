"""
CodeT5 model loader for Clone Detection task.
Fixed version: Correctly bypasses LM Head for memory efficiency and fixed attribute error.
"""

import os
import logging
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    max_source_length: int = 400


class RobertaClassificationHead(nn.Module):
    """
    Classification head for clone detection using CodeT5.
    Takes EOS token representations from two code snippets, combines them.
    """

    def __init__(self, config: T5Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to concatenate pairs: [batch_size, hidden_size * 2]
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class CodeT5CloneModel(nn.Module):
    """
    CodeT5 model for clone detection.
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
        self.classifier = RobertaClassificationHead(config)

    def get_t5_vec(self, source_ids: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient sequence representation extraction.
        Correctly chains encoder and decoder to bypass LM Head.
        """
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        
        # 1. 运行 Encoder 
        encoder_outputs = self.encoder.encoder(
            input_ids=source_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # 2. 运行 Decoder (手动构建 decoder_input_ids 并跳过 lm_head)
        decoder_input_ids = self.encoder._shift_right(source_ids)
        decoder_outputs = self.encoder.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 使用 Decoder 最后一层的隐藏状态
        hidden_states = decoder_outputs.last_hidden_state
        eos_mask = source_ids.eq(self.config.eos_token_id)

        # 提取 EOS 处的表示向量
        if len(torch.unique(eos_mask.sum(1))) > 1:
            # 如果每个样本的 EOS 数量不一致，取最后一个有效 token
            vec = hidden_states[:, -1, :]
        else:
            # 正常情况下提取 EOS 对应的向量
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

        # 动态重塑：将 [batch, total_len] 拆为 [batch*2, block_size]
        batch_size = source_ids.size(0)
        source_ids = source_ids.reshape(batch_size * 2, -1)

        vec = self.get_t5_vec(source_ids)
        logits = self.classifier(vec)
        prob = F.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob

        return prob

    def get_logits(
        self,
        source_ids: torch.Tensor = None,
        input_ids: torch.Tensor = None
    ) -> torch.Tensor:
        if source_ids is None:
            source_ids = input_ids

        batch_size = source_ids.size(0)
        source_ids = source_ids.reshape(batch_size * 2, -1)

        vec = self.get_t5_vec(source_ids)
        logits = self.classifier(vec)
        return logits


@ModelRegistry.register("cd", "codet5")
class CodeT5CloneLoader(BaseModelLoader):
    """
    Loader for CodeT5-based clone detection models.
    """

    def load(self) -> Tuple[nn.Module, RobertaTokenizer]:
        logger.info(f"Loading CodeT5 Clone model from {self.config.checkpoint_path}")

        # 加载配置和分词器
        config = T5Config.from_pretrained(self.config.base_model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.base_model_path)

        # 加载基础模型
        encoder = T5ForConditionalGeneration.from_pretrained(
            self.config.base_model_path,
            config=config
        )

        args = CodeT5Args(
            model_type="codet5",
            max_source_length=self.config.max_length
        )
        self.model = CodeT5CloneModel(encoder, config, self.tokenizer, args)

        checkpoint_path = self._find_checkpoint()
        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        
        # 词表大小自动调整逻辑
        if 'encoder.shared.weight' in state_dict:
            checkpoint_vocab_size = state_dict['encoder.shared.weight'].shape[0]
            current_vocab_size = self.model.encoder.shared.weight.shape[0]
            
            if checkpoint_vocab_size != current_vocab_size:
                logger.info(f"Resizing model embeddings from {current_vocab_size} to {checkpoint_vocab_size}")
                self.model.encoder.resize_token_embeddings(checkpoint_vocab_size)

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._is_loaded = True
        logger.info(f"CodeT5 Clone model loaded successfully on {self.device}")

        return self.model, self.tokenizer

    def _find_checkpoint(self) -> str:
        candidates = [
            os.path.join(self.config.checkpoint_path, "checkpoint-last", "pytorch_model.bin"),
            os.path.join(self.config.checkpoint_path, "pytorch_model.bin"),
            os.path.join(self.config.checkpoint_path, "checkpoint-last", "model.bin")
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find model checkpoint in {self.config.checkpoint_path}.")

    def preprocess(self, code: Union[Tuple[str, str], List[Tuple[str, str]]]) -> Dict[str, torch.Tensor]:
        self.ensure_loaded()
        code_pairs = [code] if isinstance(code, tuple) else code
        
        all_input_ids = []
        for code1, code2 in code_pairs:
            encoded1 = self.tokenizer(code1, padding="max_length", truncation=True, max_length=self.config.max_length, return_tensors="pt")
            encoded2 = self.tokenizer(code2, padding="max_length", truncation=True, max_length=self.config.max_length, return_tensors="pt")
            pair_ids = torch.cat([encoded1["input_ids"], encoded2["input_ids"]], dim=0)
            all_input_ids.append(pair_ids)

        return {"input_ids": torch.stack(all_input_ids, dim=0).to(self.device)}

    def predict(self, code: Tuple[str, str]) -> ModelPrediction:
        self.ensure_loaded()
        inputs = self.preprocess(code)
        with torch.no_grad():
            self.model.eval()
            logits = self.model.get_logits(input_ids=inputs["input_ids"])
            probs = torch.softmax(logits, dim=-1).squeeze()
            label = torch.argmax(probs).item()
            prob_values = probs.tolist()
            logit_values = logits.squeeze().tolist()
        return ModelPrediction(label=label, probability=prob_values[label], probabilities=prob_values, logits=logit_values)

    def _batch_predict_impl(self, codes: List[Tuple[str, str]]) -> List[ModelPrediction]:
        inputs = self.preprocess(codes)
        with torch.no_grad():
            self.model.eval()
            logits = self.model.get_logits(input_ids=inputs["input_ids"])
            probs = torch.softmax(logits, dim=-1)
            predictions = []
            for i in range(len(codes)):
                p_v = probs[i].tolist()
                predictions.append(ModelPrediction(
                    label=torch.argmax(probs[i]).item(),
                    probability=p_v[torch.argmax(probs[i]).item()],
                    probabilities=p_v,
                    logits=logits[i].tolist()
                ))
            return predictions