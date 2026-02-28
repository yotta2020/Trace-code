"""
StarCoder model loader for Clone Detection task.
Fixed version: Resolved DType mismatch (Half vs Float) and optimized initialization.

Clone Detection for StarCoder follows a single-sequence concatenation strategy
using the last token representation for classification.
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
from transformers.modeling_outputs import SequenceClassifierOutput

from ...base import BaseModelLoader, ModelConfig, ModelPrediction
from ...registry import ModelRegistry

logger = logging.getLogger(__name__)


class StarCoderCloneModel(nn.Module):
    """
    StarCoder model for clone detection.
    Architecture: [Code1 + Code2 concatenated] -> GPTBigCode -> Last Token -> Linear Classifier
    """

    def __init__(self, encoder, config, block_size: int = 512):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.block_size = block_size

        # 分类头：映射 hidden_size 到 2 个类别 (non-clone/clone)
        self.classifier = nn.Linear(config.hidden_size, 2)

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs
    ) -> SequenceClassifierOutput:
        """
        Forward pass using the last token of the concatenated sequence.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # 提取最后一层隐藏状态 [batch, seq_len, hidden]
        last_hidden_state = outputs.hidden_states[-1]

        # 动态定位最后一个非 Padding Token 的索引 (GPT 标准)
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
        else:
            sequence_lengths = torch.full((input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device)
            
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        pooled_output = last_hidden_state[batch_indices, sequence_lengths]

        # --- 关键修复：确保输入 DType 与线性层权重一致 ---
        pooled_output = pooled_output.to(self.classifier.weight.dtype)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        outputs = self.forward(input_ids, attention_mask)
        return outputs.logits


@ModelRegistry.register("cd", "starcoder")
class StarCoderCloneLoader(BaseModelLoader):
    """
    Loader for StarCoder-based clone detection models.
    Supports merged models and LoRA adapter checkpoints.
    """

    def load(self) -> Tuple[nn.Module, AutoTokenizer]:
        logger.info(f"Loading StarCoder Clone model from {self.config.checkpoint_path}")

        model_dir = self._find_model_dir()
        logger.info(f"Found model directory: {model_dir}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载模型逻辑
        if self._is_merged_model(model_dir):
            self._load_merged_model(model_dir)
        else:
            self._load_adapter_model(model_dir)

        # 加载分类头权重
        self._load_classifier_head(model_dir)

        # --- 核心修复：强制将整个模型（含分类头）转换为 Encoder 精度并移动到设备 ---
        target_dtype = next(self.model.encoder.parameters()).dtype
        self.model.to(device=self.device, dtype=target_dtype)
        self.model.eval()

        self._is_loaded = True
        logger.info(f"StarCoder Clone model loaded successfully on {self.device} with dtype {target_dtype}")

        return self.model, self.tokenizer

    def _find_model_dir(self) -> str:
        # 查找 merged/, checkpoint-best/, checkpoint-last/ 等目录
        for sub in ["merged", "checkpoint-best", "checkpoint-last"]:
            d = os.path.join(self.config.checkpoint_path, sub)
            if os.path.isdir(d): return d
        if os.path.isdir(self.config.checkpoint_path): return self.config.checkpoint_path
        raise FileNotFoundError(f"Model dir not found in {self.config.checkpoint_path}")

    def _is_merged_model(self, model_dir: str) -> bool:
        return os.path.exists(os.path.join(model_dir, "config.json")) and \
               not os.path.exists(os.path.join(model_dir, "adapter_config.json"))

    def _load_merged_model(self, model_dir: str) -> None:
        logger.info("Loading merged model...")
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        # 默认加载为 float16 以节省显存
        encoder = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model = StarCoderCloneModel(encoder=encoder, config=config, block_size=self.config.max_length)

    def _load_adapter_model(self, model_dir: str) -> None:
        logger.info("Loading adapter model...")
        from peft import PeftModel, PeftConfig
        peft_config = PeftConfig.from_pretrained(model_dir)
        base_path = peft_config.base_model_name_or_path if os.path.exists(peft_config.base_model_name_or_path) else self.config.base_model_path
        
        config = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path, config=config, trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        encoder = PeftModel.from_pretrained(base_model, model_dir)
        encoder = encoder.merge_and_unload()
        self.model = StarCoderCloneModel(encoder=encoder, config=config, block_size=self.config.max_length)

    def _load_classifier_head(self, model_dir: str) -> None:
        for name in ["classifier_head.bin", "classifier.pt"]:
            path = os.path.join(model_dir, name)
            if os.path.exists(path):
                logger.info(f"Loading classifier head from {path}")
                state = torch.load(path, map_location=self.device)
                self.model.classifier.load_state_dict(state)
                return
        logger.warning(f"No classifier head found in {model_dir}. Using random weights.")

    def preprocess(self, code: Union[Tuple[str, str], List[Tuple[str, str]]]) -> Dict[str, torch.Tensor]:
        self.ensure_loaded()
        code_pairs = [code] if isinstance(code, tuple) else code
        concatenated_codes = [c1 + c2 for c1, c2 in code_pairs]
        total_max_length = self.config.max_length * 2

        encoded = self.tokenizer(
            concatenated_codes, padding="max_length", truncation=True,
            max_length=total_max_length, return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def predict(self, code: Tuple[str, str]) -> ModelPrediction:
        self.ensure_loaded()
        inputs = self.preprocess(code)
        with torch.no_grad():
            logits = self.model.get_logits(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            probs = torch.softmax(logits, dim=-1).squeeze()
            label = torch.argmax(probs).item()
        return ModelPrediction(label=label, probability=probs.tolist()[label], probabilities=probs.tolist(), logits=logits.squeeze().tolist())

    def _batch_predict_impl(self, codes: List[Tuple[str, str]]) -> List[ModelPrediction]:
        inputs = self.preprocess(codes)
        with torch.no_grad():
            logits = self.model.get_logits(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            probs = torch.softmax(logits, dim=-1)
            results = []
            for i in range(len(codes)):
                p = probs[i].tolist()
                l = torch.argmax(probs[i]).item()
                results.append(ModelPrediction(label=l, probability=p[l], probabilities=p, logits=logits[i].tolist()))
            return results